// M9 model surface: verify the transformer building blocks (RoPE op +
// RMSNorm/SiLU/SwiGLU compositions) and a full llama-style decoder block
// assembled from them + dot + fused attn_decode, against a from-scratch CPU
// reference. This is the "wiring" proof for turning the M6-M9 kernels into a
// runnable LLM — it confirms the pure array ops compose into a correct layer.
//
// Backend-agnostic (array API): on CUDA the GPU kernels run and are checked vs
// the independent CPU reference here; on a CPU build the array ops fall to their
// own ref path, still checked vs this independent reference. Returns nonzero on
// any mismatch so it can gate CI.

#include <tensorlib.h>

#include <cmath>
#include <cstdio>
#include <vector>

using tl::array;
using vec = std::vector<float>;

namespace {

// Fixed xorshift fill in ~[-1,1); deterministic across runs.
vec rnd(size_t n, uint32_t seed) {
  vec v(n);
  uint32_t s = seed * 2654435761u + 1;
  for (auto& x : v) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    x = static_cast<int32_t>(s) * (1.0f / 2147483648.0f);
  }
  return v;
}

double maxrel(const vec& got, const vec& ref) {
  double m = 0;
  for (size_t i = 0; i < got.size(); i++)
    m = std::max(m, std::fabs((double)got[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
  return m;
}

vec to_host(const array& a) {
  a.eval();
  const float* p = a.raw();
  return vec(p, p + a.size());
}

// ---- CPU references (independent of the library) ----

// C[M,N] = A[M,K] · B[K,N]
vec matmul(const vec& A, const vec& B, int64_t M, int64_t K, int64_t N) {
  vec C(M * N, 0.0f);
  for (int64_t i = 0; i < M; i++)
    for (int64_t k = 0; k < K; k++) {
      float a = A[i * K + k];
      for (int64_t j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
    }
  return C;
}

vec rmsnorm_ref(const vec& x, const vec& w, int64_t rows, int64_t D, float eps) {
  vec out(x.size());
  for (int64_t r = 0; r < rows; r++) {
    double ms = 0;
    for (int64_t d = 0; d < D; d++) ms += (double)x[r * D + d] * x[r * D + d];
    ms /= D;
    float inv = (float)(1.0 / std::sqrt(ms + eps));
    for (int64_t d = 0; d < D; d++) out[r * D + d] = x[r * D + d] * inv * w[d];
  }
  return out;
}

// half-split RoPE (matches tl_rope): rows = H*T, position = pos + (r % T).
vec rope_ref(const vec& x, int64_t rows, int64_t T, int64_t D, int64_t pos,
             float base) {
  vec out(x.size());
  int64_t half = D / 2;
  for (int64_t r = 0; r < rows; r++) {
    int64_t t = T ? r % T : 0;
    double p = (double)(pos + t);
    for (int64_t j = 0; j < half; j++) {
      double theta = std::pow((double)base, -2.0 * (double)j / D);
      double ang = p * theta;
      float c = (float)std::cos(ang), s = (float)std::sin(ang);
      float x0 = x[r * D + j], x1 = x[r * D + j + half];
      out[r * D + j] = x0 * c - x1 * s;
      out[r * D + j + half] = x0 * s + x1 * c;
    }
  }
  return out;
}

bool report(const char* name, double mr) {
  bool ok = mr < 2e-4;
  std::printf("  %-22s maxrel %.2e  %s\n", name, mr, ok ? "OK" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  const bool gpu = tl::gpu_available();
  if (gpu) tl::use_gpu();
  std::printf("LLM block check — %s backend\n", gpu ? "GPU" : "CPU");
  bool ok = true;

  // ---- unit: RoPE (decode [H,D] and prefill [H,T,D]) ----
  {
    int64_t H = 4, D = 128, pos = 7;
    vec xh = rnd(H * D, 1);
    array x = array::from(xh, {H, D});
    ok &= report("rope [H,D]",
                 maxrel(to_host(array::rope(x, pos)),
                        rope_ref(xh, H, 1, D, pos, 10000.0f)));

    int64_t T = 5;
    vec x3 = rnd(H * T * D, 2);
    array X = array::from(x3, {H, T, D});
    ok &= report("rope [H,T,D]",
                 maxrel(to_host(array::rope(X, pos)),
                        rope_ref(x3, H * T, T, D, pos, 10000.0f)));
  }

  // ---- unit: RMSNorm / SiLU / SwiGLU ----
  {
    int64_t R = 6, D = 512;
    vec xh = rnd(R * D, 3), wh = rnd(D, 4);
    array x = array::from(xh, {R, D}), w = array::from(wh, {D});
    ok &= report("rmsnorm",
                 maxrel(to_host(array::rmsnorm(x, w, 1e-5f)),
                        rmsnorm_ref(xh, wh, R, D, 1e-5f)));

    vec silu_ref(xh.size());
    for (size_t i = 0; i < xh.size(); i++)
      silu_ref[i] = xh[i] / (1.0f + std::exp(-xh[i]));
    ok &= report("silu", maxrel(to_host(array::silu(x)), silu_ref));

    vec gh = rnd(R * D, 5), uh = rnd(R * D, 6);
    array g = array::from(gh, {R, D}), u = array::from(uh, {R, D});
    vec sg_ref(gh.size());
    for (size_t i = 0; i < gh.size(); i++)
      sg_ref[i] = (gh[i] / (1.0f + std::exp(-gh[i]))) * uh[i];
    ok &= report("swiglu", maxrel(to_host(array::swiglu(g, u)), sg_ref));
  }

  // ---- a full llama decoder block for one decode token ----
  // RMSNorm -> qkv proj -> RoPE(q,k) -> append to ctx -> attn -> out proj ->
  // residual -> RMSNorm -> SwiGLU MLP -> residual. K/V context given as arrays.
  {
    const int64_t H = 4, hd = 128, Dm = H * hd, ffn = 1024, ctx = 8, pos = ctx;
    const float eps = 1e-5f, scale = 1.0f / std::sqrt((float)hd);

    vec xh = rnd(Dm, 10);
    vec Wq = rnd(Dm * Dm, 11), Wk = rnd(Dm * Dm, 12), Wv = rnd(Dm * Dm, 13);
    vec Wo = rnd(Dm * Dm, 14), Wg = rnd(Dm * ffn, 15), Wu = rnd(Dm * ffn, 16);
    vec Wd = rnd(ffn * Dm, 17), n1 = rnd(Dm, 18), n2 = rnd(Dm, 19);
    vec Kc = rnd(H * ctx * hd, 20), Vc = rnd(H * ctx * hd, 21);
    // scale weights down so the deep matmul chain stays in a sane range
    auto sc = [](vec v, float f) { for (auto& x : v) x *= f; return v; };
    Wq = sc(Wq, 0.04f); Wk = sc(Wk, 0.04f); Wv = sc(Wv, 0.04f);
    Wo = sc(Wo, 0.04f); Wg = sc(Wg, 0.04f); Wu = sc(Wu, 0.04f);
    Wd = sc(Wd, 0.03f);

    // ---- library block (array ops; GPU when available) ----
    array x = array::from(xh, {1, Dm});
    array wq = array::from(Wq, {Dm, Dm}), wk = array::from(Wk, {Dm, Dm});
    array wv = array::from(Wv, {Dm, Dm}), wo = array::from(Wo, {Dm, Dm});
    array wg = array::from(Wg, {Dm, ffn}), wu = array::from(Wu, {Dm, ffn});
    array wd = array::from(Wd, {ffn, Dm});
    array nw1 = array::from(n1, {Dm}), nw2 = array::from(n2, {Dm});

    array h = array::rmsnorm(x, nw1, eps);
    array q = array::rope(h.dot(wq).reshape({H, hd}), pos);
    array k = array::rope(h.dot(wk).reshape({H, hd}), pos);
    array v = h.dot(wv).reshape({H, hd});
    vec kh = to_host(k), vh = to_host(v);  // for host-side context assembly

    // append the new token's k,v to the given context: [H, ctx+1, hd]
    vec Kf(H * (ctx + 1) * hd), Vf(H * (ctx + 1) * hd);
    for (int64_t hh = 0; hh < H; hh++) {
      for (int64_t j = 0; j < ctx; j++)
        for (int64_t d = 0; d < hd; d++) {
          Kf[(hh * (ctx + 1) + j) * hd + d] = Kc[(hh * ctx + j) * hd + d];
          Vf[(hh * (ctx + 1) + j) * hd + d] = Vc[(hh * ctx + j) * hd + d];
        }
      for (int64_t d = 0; d < hd; d++) {
        Kf[(hh * (ctx + 1) + ctx) * hd + d] = kh[hh * hd + d];
        Vf[(hh * (ctx + 1) + ctx) * hd + d] = vh[hh * hd + d];
      }
    }
    array Ka = array::from(Kf, {H, ctx + 1, hd});
    array Va = array::from(Vf, {H, ctx + 1, hd});
    array attn = array::attn_decode(q, Ka, Va, scale).reshape({1, Dm});
    array x2 = x + attn.dot(wo);
    array h2 = array::rmsnorm(x2, nw2, eps);
    array mlp = array::swiglu(h2.dot(wg), h2.dot(wu)).dot(wd);
    array out = x2 + mlp;
    vec got = to_host(out);

    // ---- from-scratch CPU reference of the identical block ----
    vec hn = rmsnorm_ref(xh, n1, 1, Dm, eps);
    vec qh = rope_ref(matmul(hn, Wq, 1, Dm, Dm), H, 1, hd, pos, 10000.0f);
    vec kr = rope_ref(matmul(hn, Wk, 1, Dm, Dm), H, 1, hd, pos, 10000.0f);
    vec vr = matmul(hn, Wv, 1, Dm, Dm);
    vec attn_ref(H * hd);
    std::vector<double> s(ctx + 1);
    for (int64_t hh = 0; hh < H; hh++) {
      double mx = -1e30;
      for (int64_t j = 0; j <= ctx; j++) {
        double dot = 0;
        for (int64_t d = 0; d < hd; d++) {
          float kv = (j < ctx) ? Kc[(hh * ctx + j) * hd + d] : kr[hh * hd + d];
          dot += (double)qh[hh * hd + d] * kv;
        }
        s[j] = dot * scale;
        mx = std::max(mx, s[j]);
      }
      double sum = 0;
      for (int64_t j = 0; j <= ctx; j++) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
      for (int64_t d = 0; d < hd; d++) {
        double acc = 0;
        for (int64_t j = 0; j <= ctx; j++) {
          float vv = (j < ctx) ? Vc[(hh * ctx + j) * hd + d] : vr[hh * hd + d];
          acc += s[j] * vv;
        }
        attn_ref[hh * hd + d] = (float)(acc / sum);
      }
    }
    vec proj = matmul(attn_ref, Wo, 1, Dm, Dm);
    vec x2r(Dm);
    for (int64_t i = 0; i < Dm; i++) x2r[i] = xh[i] + proj[i];
    vec h2r = rmsnorm_ref(x2r, n2, 1, Dm, eps);
    vec gate = matmul(h2r, Wg, 1, Dm, ffn), up = matmul(h2r, Wu, 1, Dm, ffn);
    vec sw(ffn);
    for (int64_t i = 0; i < ffn; i++)
      sw[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
    vec down = matmul(sw, Wd, 1, ffn, Dm);
    vec ref(Dm);
    for (int64_t i = 0; i < Dm; i++) ref[i] = x2r[i] + down[i];

    ok &= report("decoder block", maxrel(got, ref));
  }

  std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
  return ok ? 0 : 1;
}
