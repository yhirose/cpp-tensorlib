// M9 end-to-end: a multi-layer llama-style decode loop wired to the persistent
// cuda::kv_cache (GQA), driven greedily, verified vs a from-scratch CPU
// reference at every step. This is the "runnable LLM" proof — it composes the
// pure array ops (rmsnorm/rope/swiglu + dot) for the per-token math and bridges
// each layer's attention to the stateful cache via array::native(). Random-but-
// fixed weights + integer token ids indexing a random embedding table (no GGUF /
// tokenizer yet — that is the follow-on to actually chat).
//
// CUDA-only (uses tl::cuda::kv_cache). Returns nonzero on any logits mismatch or
// divergent greedy token, so it gates the wiring.

#include <tensorlib.h>

#include <cmath>
#include <cstdio>
#include <vector>

using tl::array;
using vec = std::vector<float>;

namespace {

// --- config: small but real (GQA, multi-layer); head_dim=128 (kernel-fixed) ---
constexpr int64_t hd = 128, HQ = 4, HKV = 2, Dm = HQ * hd;  // 512
constexpr int64_t ffn = 1024, NL = 3, VOCAB = 48, MAXC = 64;
constexpr int64_t GROUP = HQ / HKV;
const float EPS = 1e-5f, SCALE = 1.0f / 11.3137085f /* 1/sqrt(128) */;

vec rnd(size_t n, uint32_t seed, float mul = 1.0f) {
  vec v(n);
  uint32_t s = seed * 2654435761u + 1;
  for (auto& x : v) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    x = static_cast<int32_t>(s) * (1.0f / 2147483648.0f) * mul;
  }
  return v;
}

struct LayerW {
  vec Wq, Wk, Wv, Wo, Wg, Wu, Wd, n1, n2;
};
struct Weights {
  vec embed;       // [VOCAB, Dm]
  std::vector<LayerW> layers;
  vec fnorm;       // [Dm]
  vec lm_head;     // [Dm, VOCAB]
};

Weights make_weights() {
  Weights w;
  w.embed = rnd(VOCAB * Dm, 1, 0.1f);
  w.fnorm = rnd(Dm, 2, 0.1f);
  for (auto& x : w.fnorm) x += 1.0f;  // norm weights ~1
  w.lm_head = rnd(Dm * VOCAB, 3, 0.04f);
  for (int64_t l = 0; l < NL; l++) {
    LayerW lw;
    uint32_t b = 100u + l * 20u;
    lw.Wq = rnd(Dm * HQ * hd, b + 1, 0.04f);
    lw.Wk = rnd(Dm * HKV * hd, b + 2, 0.04f);
    lw.Wv = rnd(Dm * HKV * hd, b + 3, 0.04f);
    lw.Wo = rnd(HQ * hd * Dm, b + 4, 0.04f);
    lw.Wg = rnd(Dm * ffn, b + 5, 0.04f);
    lw.Wu = rnd(Dm * ffn, b + 6, 0.04f);
    lw.Wd = rnd(ffn * Dm, b + 7, 0.03f);
    lw.n1 = rnd(Dm, b + 8, 0.1f);
    lw.n2 = rnd(Dm, b + 9, 0.1f);
    for (auto& x : lw.n1) x += 1.0f;
    for (auto& x : lw.n2) x += 1.0f;
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// ---- CPU reference ----
vec matmul(const vec& A, const vec& B, int64_t M, int64_t K, int64_t N) {
  vec C(M * N, 0.0f);
  for (int64_t i = 0; i < M; i++)
    for (int64_t k = 0; k < K; k++) {
      float a = A[i * K + k];
      for (int64_t j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
    }
  return C;
}
vec rmsnorm_ref(const vec& x, const vec& w, int64_t D) {
  vec o(x.size());
  double ms = 0;
  for (int64_t d = 0; d < D; d++) ms += (double)x[d] * x[d];
  float inv = (float)(1.0 / std::sqrt(ms / D + EPS));
  for (int64_t d = 0; d < D; d++) o[d] = x[d] * inv * w[d];
  return o;
}
void rope_ref(vec& x, int64_t H, int64_t pos) {
  int64_t half = hd / 2;
  for (int64_t h = 0; h < H; h++)
    for (int64_t j = 0; j < half; j++) {
      double theta = std::pow(10000.0, -2.0 * (double)j / hd);
      double ang = (double)pos * theta;
      float c = (float)std::cos(ang), s = (float)std::sin(ang);
      float x0 = x[h * hd + j], x1 = x[h * hd + j + half];
      x[h * hd + j] = x0 * c - x1 * s;
      x[h * hd + j + half] = x0 * s + x1 * c;
    }
}

// Host KV cache: Kc[l], Vc[l] laid out [HKV, MAXC, hd]; pos shared.
struct RefState {
  std::vector<vec> Kc, Vc;
  int64_t pos = 0;
  RefState() : Kc(NL, vec(HKV * MAXC * hd)), Vc(NL, vec(HKV * MAXC * hd)) {}
};

vec cpu_step(const Weights& w, RefState& st, int64_t id, int64_t pos) {
  vec x(w.embed.begin() + id * Dm, w.embed.begin() + (id + 1) * Dm);
  for (int64_t l = 0; l < NL; l++) {
    const LayerW& L = w.layers[l];
    vec h = rmsnorm_ref(x, L.n1, Dm);
    vec q = matmul(h, L.Wq, 1, Dm, HQ * hd);
    vec k = matmul(h, L.Wk, 1, Dm, HKV * hd);
    vec v = matmul(h, L.Wv, 1, Dm, HKV * hd);
    rope_ref(q, HQ, pos);
    rope_ref(k, HKV, pos);
    // append k,v at row `pos`
    for (int64_t hh = 0; hh < HKV; hh++)
      for (int64_t d = 0; d < hd; d++) {
        st.Kc[l][(hh * MAXC + pos) * hd + d] = k[hh * hd + d];
        st.Vc[l][(hh * MAXC + pos) * hd + d] = v[hh * hd + d];
      }
    int64_t vlen = pos + 1;
    vec attn(HQ * hd);
    std::vector<double> sc(vlen);
    for (int64_t hh = 0; hh < HQ; hh++) {
      int64_t kvh = hh / GROUP;
      const float* Kh = &st.Kc[l][kvh * MAXC * hd];
      const float* Vh = &st.Vc[l][kvh * MAXC * hd];
      double mx = -1e30;
      for (int64_t j = 0; j < vlen; j++) {
        double dot = 0;
        for (int64_t d = 0; d < hd; d++) dot += (double)q[hh * hd + d] * Kh[j * hd + d];
        sc[j] = dot * SCALE;
        mx = std::max(mx, sc[j]);
      }
      double sum = 0;
      for (int64_t j = 0; j < vlen; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
      for (int64_t d = 0; d < hd; d++) {
        double acc = 0;
        for (int64_t j = 0; j < vlen; j++) acc += sc[j] * Vh[j * hd + d];
        attn[hh * hd + d] = (float)(acc / sum);
      }
    }
    vec proj = matmul(attn, L.Wo, 1, HQ * hd, Dm);
    for (int64_t i = 0; i < Dm; i++) x[i] += proj[i];
    vec h2 = rmsnorm_ref(x, L.n2, Dm);
    vec g = matmul(h2, L.Wg, 1, Dm, ffn), u = matmul(h2, L.Wu, 1, Dm, ffn);
    vec swg(ffn);
    for (int64_t i = 0; i < ffn; i++)
      swg[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
    vec down = matmul(swg, L.Wd, 1, ffn, Dm);
    for (int64_t i = 0; i < Dm; i++) x[i] += down[i];
  }
  vec xf = rmsnorm_ref(x, w.fnorm, Dm);
  return matmul(xf, w.lm_head, 1, Dm, VOCAB);
}

int64_t argmax(const vec& v) {
  int64_t bi = 0;
  for (int64_t i = 1; i < (int64_t)v.size(); i++)
    if (v[i] > v[bi]) bi = i;
  return bi;
}

// ---- GPU model (array ops + kv_cache bridge) ----
struct GpuLayer {
  array Wq, Wk, Wv, Wo, Wg, Wu, Wd, n1, n2;
  tl::cuda::kv_cache cache;
};
struct GpuModel {
  array embed, fnorm, lm_head;
  std::vector<GpuLayer> layers;
};

array mat(const vec& v, tl::shape_t s) { return array::from(v, std::move(s)); }

GpuModel build_gpu(const Weights& w) {
  GpuModel m{mat(w.embed, {VOCAB, Dm}), mat(w.fnorm, {Dm}),
             mat(w.lm_head, {Dm, VOCAB}), {}};
  for (const auto& L : w.layers) {
    GpuLayer g{mat(L.Wq, {Dm, HQ * hd}), mat(L.Wk, {Dm, HKV * hd}),
               mat(L.Wv, {Dm, HKV * hd}), mat(L.Wo, {HQ * hd, Dm}),
               mat(L.Wg, {Dm, ffn}),      mat(L.Wu, {Dm, ffn}),
               mat(L.Wd, {ffn, Dm}),      mat(L.n1, {Dm}),
               mat(L.n2, {Dm}),           {}};
    g.cache.init(HKV, MAXC, hd);
    m.layers.push_back(std::move(g));
  }
  return m;
}

vec gpu_step(GpuModel& m, int64_t id, int64_t pos) {
  array x = m.embed.slice(id, 1).clone();  // [1, Dm] embedding row (contiguous)
  for (auto& L : m.layers) {
    array h = array::rmsnorm(x, L.n1, EPS);
    array q = array::rope(h.dot(L.Wq).reshape({HQ, hd}), pos);
    array k = array::rope(h.dot(L.Wk).reshape({HKV, hd}), pos);
    array v = h.dot(L.Wv);  // [1, HKV*hd] == [HKV, hd] bytes
    q.eval(); k.eval(); v.eval();
    L.cache.append(k.native(), v.native());
    array a_out = array::empty({HQ, hd});
    L.cache.attn(q.native(), a_out.native(), HQ, SCALE);
    array a = a_out.reshape({1, Dm});
    array x1 = x + a.dot(L.Wo);
    array h2 = array::rmsnorm(x1, L.n2, EPS);
    array mlp = array::swiglu(h2.dot(L.Wg), h2.dot(L.Wu)).dot(L.Wd);
    array x2 = x1 + mlp;
    x2.eval();  // materialize the residual stream (bounds the graph + lifetimes)
    x = x2;
    (void)q; (void)k; (void)v; (void)a_out;  // kept alive to here (kernel order)
  }
  array logits = array::rmsnorm(x, m.fnorm, EPS).dot(m.lm_head);
  logits.eval();
  const float* p = logits.raw();
  return vec(p, p + VOCAB);
}

}  // namespace

int main() {
  if (!tl::gpu_available()) {
    std::printf("no CUDA device — skipping decode-loop check\n");
    return 0;
  }
  tl::use_gpu();
  std::printf("LLM decode-loop check — %lld layers, %lld q / %lld kv heads "
              "(GQA %lld), Dm=%lld, vocab=%lld\n",
              (long long)NL, (long long)HQ, (long long)HKV, (long long)GROUP,
              (long long)Dm, (long long)VOCAB);

  Weights w = make_weights();
  GpuModel gm = build_gpu(w);
  RefState rs;

  std::vector<int64_t> prompt = {1, 7, 3, 9, 2, 5};
  const int64_t NGEN = 6;
  double worst = 0;
  bool ids_ok = true;

  int64_t next = 0;
  int64_t pos = 0;
  std::vector<int64_t> gpu_gen, cpu_gen;
  for (size_t i = 0; i < prompt.size() + NGEN; i++) {
    int64_t id = (i < prompt.size()) ? prompt[i]
                                     : next;  // feed prompt, then own output
    vec lg = gpu_step(gm, id, pos);
    vec lr = cpu_step(w, rs, id, pos);
    double mr = 0;
    for (int64_t j = 0; j < VOCAB; j++)
      mr = std::max(mr, std::fabs((double)lg[j] - lr[j]) / (1.0 + std::fabs(lr[j])));
    worst = std::max(worst, mr);
    int64_t gnext = argmax(lg), cnext = argmax(lr);
    if (i + 1 >= prompt.size()) {  // this logits row predicts a generated token
      gpu_gen.push_back(gnext);
      cpu_gen.push_back(cnext);
      if (gnext != cnext) ids_ok = false;
    }
    next = gnext;  // greedy feedback (GPU drives; CPU tracked in parallel)
    pos++;
  }

  std::printf("  logits maxrel over %zu steps: %.2e %s\n",
              prompt.size() + NGEN, worst, worst < 2e-3 ? "OK" : "FAIL");
  std::printf("  greedy tokens  gpu:");
  for (auto t : gpu_gen) std::printf(" %lld", (long long)t);
  std::printf("\n                 cpu:");
  for (auto t : cpu_gen) std::printf(" %lld", (long long)t);
  std::printf("\n  token sequences %s\n", ids_ok ? "MATCH" : "DIVERGE");

  bool ok = worst < 2e-3 && ids_ok;
  std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
  return ok ? 0 : 1;
}
