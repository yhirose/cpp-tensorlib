// The nn layer's tests (layers: include/gpu_abi.h): the fused ops over plain
// tensors -- softmax and logsumexp, the norms, RoPE, attention and its pullback,
// cross-entropy, Adam, the bf16 and int4 GEMVs -- and the generic compositions
// that stand in for their kernels.

#include "test_util.h"

#include <ggml_quant.h>

TEST_CASE("softmax: own CPU rows across the pool match the ref oracle") {
  // rank 1, 2 and 3, a transposed (strided) view, and a shape big enough to
  // split across threads
  CHECK(cpu_matches_ref([&] { return random_array({5}, 901).softmax(); }));
  CHECK(cpu_matches_ref([&] { return random_array({7, 33}, 902).softmax(); }));
  CHECK(cpu_matches_ref([&] { return random_array({4, 6, 40}, 903).softmax(); }));
  CHECK(cpu_matches_ref([&] { return random_array({40, 6}, 904).transpose().softmax(); }));
  CHECK(cpu_matches_ref([&] { return random_array({8, 256, 256}, 905).softmax(); }));
}

TEST_CASE("cpu::exp_shifted: within a few ulp of libm over softmax's domain") {
  // Every 997th float bit pattern in [-88, 0] (~1.1M values), the exponents a
  // max-shifted softmax produces.
  const float lo = -88.0f;
  uint32_t b_lo;
  std::memcpy(&b_lo, &lo, 4);
  std::vector<float> xs;
  xs.reserve((b_lo - 0x80000000u) / 997 + 2);
  for (uint32_t b = b_lo; b > 0x80000000u; b -= 997) {
    float x;
    std::memcpy(&x, &b, 4);
    xs.push_back(x);
  }
  xs.push_back(0.0f);
  std::vector<float> ys(xs.size());
  tl::cpu::exp_shifted(ys.data(), xs.data(), 1, static_cast<int64_t>(xs.size()), 0.0f);
  // Relative error against double-precision exp, over normal results.
  double worst = 0;
  for (size_t i = 0; i < xs.size(); i++) {
    double want = std::exp(double(xs[i]));
    if (want < 1.1754944e-38) continue;  // denormal results: flushed near there
    worst = std::max(worst, std::abs(double(ys[i]) - want) / want);
  }
  MESSAGE("exp_shifted worst relative error: " << worst);
  CHECK(worst < 1.1920929e-7);  // measured 7.7e-8 (~1.3 ulp) on AVX2
  CHECK(ys.back() == 1.0f);

  float special[] = {-INFINITY, -1000.0f, NAN, 0.0f};
  float out[4];
  tl::cpu::exp_shifted(out, special, 1, 4, 0.0f);
  CHECK(out[0] == 0.0f);
  CHECK(out[1] == 0.0f);
  CHECK(std::isnan(out[2]));
  CHECK(out[3] == 1.0f);
}

TEST_CASE("cpu::exp_shifted: place in the run, and the returned sum") {
  // 37 = four full lanes and a tail of 5. Contiguous body, tail, and a strided
  // run must round the same element the same way, or a row's softmax would
  // depend on its length and layout; the tail's padding lanes add nothing.
  std::vector<float> src(37);
  for (size_t i = 0; i < src.size(); i++) src[i] = -0.37f * float(i) + 0.11f;
  std::vector<float> whole(37), strided(18);
  float sum = tl::cpu::exp_shifted(whole.data(), src.data(), 1, 37, 0.25f);
  for (int64_t i = 0; i < 37; i++) {
    float one;
    tl::cpu::exp_shifted(&one, src.data() + i, 1, 1, 0.25f);
    CHECK(one == whole[i]);
  }
  tl::cpu::exp_shifted(strided.data(), src.data() + 1, 2, 18, 0.25f);
  for (int64_t i = 0; i < 18; i++) CHECK(strided[i] == whole[1 + 2 * i]);

  double want = 0;
  for (float e : whole) want += e;
  CHECK(sum == doctest::Approx(want).epsilon(1e-6));
  CHECK(tl::cpu::exp_shifted(nullptr, src.data(), 1, 37, 0.25f) == sum);
  CHECK(tl::cpu::exp_shifted(nullptr, src.data(), 1, 0, 0.25f) == 0.0f);
}

namespace {
// The update every optimizer loop writes as ops today; adam_step has to land on
// these same numbers, so the test states the composition once and compares.
struct adam_ref {
  array m, v, p;
};
adam_ref compose_adam(const array& p0, const array& m0, const array& v0,
                      const array& g, float lr, float b1, float b2, float eps,
                      float bc1, float bc2) {
  auto m = (m0 * b1 + g * (1.0f - b1)).eval();
  auto v = (v0 * b2 + g * g * (1.0f - b2)).eval();
  auto p = (p0 - (m * (lr / bc1)) /
                     (tl::pow(v * (1.0f / bc2), 0.5f) + eps)).eval();
  return {m, v, p};
}
}  // namespace

TEST_CASE("adam_step lands on the numbers the composition does") {
  const float lr = 3e-4f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
  const float bc1 = 1.0f - b1 * b1, bc2 = 1.0f - b2 * b2;  // the second step
  auto p0 = random_array({4, 6}, 1400);
  auto m0 = random_array({4, 6}, 1401);
  auto v0 = tl::pow(random_array({4, 6}, 1402), 2.0f).eval();  // v is a square
  auto g = random_array({4, 6}, 1403);
  auto want = compose_adam(p0, m0, v0, g, lr, b1, b2, eps, bc1, bc2);

  auto p = p0.clone(), m = m0.clone(), v = v0.clone();
  CHECK(tl::array::adam_step(p, m, v, g, lr, b1, b2, eps, bc1, bc2));
  CHECK(allclose(m, want.m));
  CHECK(allclose(v, want.v));
  CHECK(allclose(p, want.p));
  CHECK(!allclose(p0, p));  // in place on the clones, not on the originals

  // shape and bias-correction misuse are the caller's bug, not a decline
  auto wrong = random_array({4, 5}, 1404);
  CHECK_THROWS_AS(tl::array::adam_step(p, m, v, wrong, lr, b1, b2, eps, bc1, bc2),
                  std::invalid_argument);
  CHECK_THROWS_AS(tl::array::adam_step(p, m, v, g, lr, b1, b2, eps, 0.0f, bc2),
                  std::invalid_argument);
}

TEST_CASE("adam_step on a device buffer matches the same composition") {
  if (!tl::gpu_available()) return;
  const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  const float bc1 = 1.0f - b1 * b1 * b1, bc2 = 1.0f - b2 * b2 * b2;
  auto prev = tl::device_;
  tl::use_gpu();
  auto p0 = random_array({129, 40}, 1410);  // a tail past the block size
  auto m0 = random_array({129, 40}, 1411);
  auto v0 = tl::pow(random_array({129, 40}, 1412), 2.0f).eval();
  auto g = random_array({129, 40}, 1413);
  auto want = compose_adam(p0, m0, v0, g, lr, b1, b2, eps, bc1, bc2);

  auto p = p0.clone(), m = m0.clone(), v = v0.clone();
  bool ran = tl::array::adam_step(p, m, v, g, lr, b1, b2, eps, bc1, bc2);
  tl::device_ = prev;
  CHECK(ran);  // total: the kernel here, the host loop on a backend without one
  CHECK(allclose(m, want.m, 1e-4f, 1e-6f));
  CHECK(allclose(v, want.v, 1e-4f, 1e-6f));
  CHECK(allclose(p, want.p, 1e-4f, 1e-6f));
}

TEST_CASE("adam_step's first device step reads host-born state rather than the mirror") {
  // The optimizer's pattern: p, m and v are made on the host and first touched
  // by the device inside the step itself. A kernel that marks them written
  // without reading them up first works on a device-born clone and computes
  // on the mirror's stale contents here. Non-zero m0/v0 on purpose: a fresh
  // device allocation reads as zeros, which would let host zeros pass by luck.
  if (!tl::gpu_available()) return;
  const float lr = 3e-4f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
  const float bc1 = 1.0f - b1, bc2 = 1.0f - b2;  // the first step
  auto prev = tl::device_;
  tl::use_cpu();
  auto p0 = random_array({64, 32}, 1420);
  auto m0 = random_array({64, 32}, 1422);
  auto v0 = tl::pow(random_array({64, 32}, 1423), 2.0f).eval();
  auto g = random_array({64, 32}, 1421);
  auto want = compose_adam(p0, m0, v0, g, lr, b1, b2, eps, bc1, bc2);
  auto p = p0.clone(), m = m0.clone(), v = v0.clone();  // host copies still
  tl::use_gpu();
  bool ran = tl::array::adam_step(p, m, v, g, lr, b1, b2, eps, bc1, bc2);
  tl::device_ = prev;
  CHECK(ran);
  CHECK(allclose(m, want.m, 1e-4f, 1e-6f));
  CHECK(allclose(v, want.v, 1e-4f, 1e-6f));
  CHECK(allclose(p, want.p, 1e-4f, 1e-6f));
}

TEST_CASE("rope GPU dispatch matches the CPU oracle") {
  // [H,D] (decode shape, T=1).
  auto x = array::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 4});
  CHECK(matches_gpu_oracle([&] { return array::rope(x, 0, 10000.0f); }));
  CHECK(matches_gpu_oracle([&] { return array::rope(x, 5, 10000.0f); }));

  // [H,T,D] (prefill shape): row r's position is pos + (r % T).
  auto y = array::full({2, 3, 8}, 0.5f);
  CHECK(matches_gpu_oracle([&] { return array::rope(y, 0, 10000.0f); }));
  CHECK(matches_gpu_oracle([&] { return array::rope(y, 4, 10000.0f); }));

  // fused epilogue composes with rope (same as tanh/sin/cos do).
  CHECK(matches_gpu_oracle(
      [&] { return array::rope(x, 2, 10000.0f) * 2.0f - 1.0f; }));

  // correctness spot-check: pos=0 rotates every row by angle 0 (position=0
  // regardless of theta), so rope is the identity there.
  auto r0 = array::rope(x, 0, 10000.0f).eval();
  const float* px = x.raw();
  const float* pr = r0.raw();
  for (int64_t i = 0; i < x.size(); i++) CHECK(pr[i] == px[i]);
}

TEST_CASE("layer_norm matches its composition, across the pool and on the GPU") {
  // The nine-op composition the fused node replaces.
  auto composed = [](const array& x, const array& g, const array& b) {
    int last = static_cast<int>(x.rank()) - 1;
    auto mu = x.mean(last, true);
    auto diff = x - mu;
    auto var = (diff * diff).mean(last, true);
    return diff * (1.0f / (var + 1e-5f).sqrt()) * g + b;
  };
  auto x1 = random_array({5}, 1101);
  auto g5 = random_array({5}, 1102), b5 = random_array({5}, 1103);
  auto x2 = random_array({7, 33}, 1104);
  auto g33 = random_array({33}, 1105), b33 = random_array({1, 33}, 1106);
  auto x3 = random_array({4, 6, 40}, 1107);
  auto g40 = random_array({1, 40}, 1108), b40 = random_array({40}, 1109);
  auto xt = random_array({40, 6}, 1110).transpose();  // [6, 40], strided rows
  auto xw = random_array({3, 700}, 1111);             // a row wider than 256
  auto g700 = random_array({700}, 1112), b700 = random_array({700}, 1113);

  auto matches_composed = [&](const array& x, const array& g, const array& b) {
    return tl::allclose(array::layer_norm(x, g, b).eval(),
                        composed(x, g, b).eval(), 1e-4f, 1e-5f);
  };
  CHECK(matches_composed(x1, g5, b5));
  CHECK(matches_composed(x2, g33, b33));
  CHECK(matches_composed(x3, g40, b40));
  CHECK(matches_composed(xt, g40, b40));
  CHECK(matches_composed(xw, g700, b700));

  auto x4 = random_array({64, 512}, 1114);
  auto g512 = random_array({512}, 1115), b512 = random_array({512}, 1116);
  CHECK(cpu_matches_ref([&] { return array::layer_norm(x4, g512, b512); }));

  CHECK(matches_gpu_oracle([&] { return array::layer_norm(x2, g33, b33); }));
  CHECK(matches_gpu_oracle([&] { return array::layer_norm(x3, g40, b40); }));
  CHECK(matches_gpu_oracle([&] { return array::layer_norm(xw, g700, b700); }));
  CHECK(matches_gpu_oracle([&] { return array::layer_norm(x4, g512, b512); }));
  // fused epilogue composes with the kernel's store.
  CHECK(matches_gpu_oracle(
      [&] { return array::layer_norm(x2, g33, b33) * 2.0f - 1.0f; }));

  CHECK_THROWS(array::layer_norm(x2, random_array({32}, 1117), b33));
  CHECK_THROWS(array::layer_norm(x2, g33, random_array({2, 33}, 1118)));
}

TEST_CASE("layer_norm_bwd matches the composed pullback on the GPU and the own CPU") {
  // The closed form the fused kernels replace, spelled with the unfused ops
  // on the CPU: dx = s·(ĝ − mean ĝ − x̂·mean(ĝ⊙x̂)), dγ = Σ dy⊙x̂, dβ = Σ dy.
  auto composed = [](const array& x, const array& g, const array& dy) {
    int last = static_cast<int>(x.rank()) - 1;
    int64_t d = x.shape().back(), rows = x.size() / d;
    auto mu = x.mean(last, true);
    auto diff = x - mu;
    auto var = (diff * diff).mean(last, true);
    auto s = 1.0f / (var + 1e-5f).sqrt();
    auto xhat = diff * s;
    auto gh = dy * g;
    auto dx = s * (gh - gh.mean(last, true) - xhat * (gh * xhat).mean(last, true));
    auto dg = (dy * xhat).reshape({rows, d}).sum(0);
    auto db = dy.reshape({rows, d}).sum(0);
    return std::array<array, 3>{dx.eval(), dg.eval(), db.eval()};
  };
  auto prev = tl::device_;
  bool on_gpu = false;
  auto check = [&](const array& x, const array& g, const array& dy) {
    if (on_gpu) tl::use_gpu();
    auto got = tl::array::layer_norm_bwd(x, g, dy);
    tl::use_cpu();
    // The own CPU takes every contiguous input, and so does a backend that
    // has the kernel; one without it declines and its caller composes the
    // form above.
    if (!on_gpu) REQUIRE(got.has_value());
    if (tl::gpu::has_layer_norm_bwd) REQUIRE(got.has_value());
    if (!got) return;
    auto want = composed(x, g, dy);
    for (int i = 0; i < 3; i++) {
      CHECK(got->at(i).shape() == want[i].shape());
      CHECK(tl::allclose(got->at(i).eval(), want[i], 1e-4f, 1e-5f));
    }
  };
  // A short input (one row chunk, a partial column strip), a rank-3 one with
  // gamma as [1, d], a row wider than the 256-thread block, 200 rows in 64-row
  // chunks with a partial last (64·3 + 8), and 4200 rows where the chunks
  // grow past 64 rows to stay at 64 of them (66 each, the last 42) — the own
  // CPU chunks its column sums the same way.
  for (bool gpu : {false, true}) {
    if (gpu && !tl::gpu_available()) continue;
    on_gpu = gpu;
    check(random_array({7, 33}, 1201), random_array({33}, 1202),
          random_array({7, 33}, 1203));
    check(random_array({4, 6, 40}, 1204), random_array({1, 40}, 1205),
          random_array({4, 6, 40}, 1206));
    check(random_array({3, 700}, 1207), random_array({700}, 1208),
          random_array({3, 700}, 1209));
    check(random_array({200, 300}, 1210), random_array({300}, 1211),
          random_array({200, 300}, 1212));
    check(random_array({4200, 40}, 1219), random_array({40}, 1220),
          random_array({4200, 40}, 1221));
  }
  auto x = random_array({7, 33}, 1213), dy = random_array({7, 33}, 1214);
  CHECK_THROWS(tl::array::layer_norm_bwd(x, random_array({32}, 1215), dy));
  CHECK_THROWS(tl::array::layer_norm_bwd(x, random_array({2, 33}, 1216), dy));
  CHECK_THROWS(tl::array::layer_norm_bwd(x, random_array({33}, 1217),
                                         random_array({33, 7}, 1218)));
  CHECK_THROWS(tl::array::layer_norm_bwd(random_array({1}, 1222),
                                         array::full({}, 1.0f),
                                         random_array({1}, 1223)));
  tl::device_ = prev;
}

TEST_CASE("bf16 storage: round-trip, widen fallback, dot fast path") {
  // round-trip: bf16 keeps the top 8 mantissa bits (values below are exact)
  auto w = array::from({1.0f, -2.5f, 0.15625f, 3.0f, -0.75f, 4.0f}, {3, 2});
  auto wb = w.to_bf16();
  CHECK(wb.dt() == tl::dtype::bf16);
  CHECK(wb.shape() == tl::shape_t{3, 2});
  auto back = wb.to_f32();
  CHECK(back.dt() == tl::dtype::f32);
  for (int64_t i = 0; i < 3; i++)
    for (int64_t j = 0; j < 2; j++) CHECK(back.at({i, j}) == w.at({i, j}));

  // views stay legal (unlike q4): to_f32() walks their offset and strides
  CHECK(tl::array_equal(wb.slice(1, 2).to_f32(), back.slice(1, 2)));
  CHECK(tl::array_equal(wb.slice(1, 1, 1).to_f32(), back.slice(1, 1, 1)));
  CHECK(tl::array_equal(wb.transpose().to_f32(), back.transpose()));

  // rounding: values needing more than 8 mantissa bits round to nearest-even
  auto x = array::from({1.00390625f});  // 1 + 2^-8: exactly between bf16 steps
  float rt = x.to_bf16().to_f32().at({0});
  CHECK(rt == doctest::Approx(1.0f).epsilon(0.01));

  // direct element access on bf16 must throw (weight-container semantics)
  CHECK_THROWS(wb.at({0, 0}));

  // decode-shaped dot: a(1,K) @ Wbf16(K,N) — fast path on CUDA, widen
  // fallback everywhere else; both must match the f32 result within bf16
  // weight precision (~2^-8 relative).
  const int64_t K = 64, N = 48;
  std::vector<float> av(K), wv(K * N);
  unsigned s = 42;
  auto rnd = [&] {
    s = s * 1664525u + 1013904223u;
    return (float)(int)(s >> 9) / (1 << 22) - 1.0f;
  };
  for (auto& v : av) v = rnd();
  for (auto& v : wv) v = rnd();
  auto a = array::from(av, {1, K});
  auto wf = array::from(wv, {K, N});
  auto ref = a.dot(wf);
  auto got = a.dot(wf.to_bf16());
  CHECK(got.shape() == ref.shape());
  for (int64_t j = 0; j < N; j += 7) {
    float r = ref.at({0, j}), g = got.at({0, j});
    CHECK(g == doctest::Approx(r).epsilon(0.02));
  }

  // epilogue still applies through the bf16 path (dot result * 2 + 1)
  auto fused = a.dot(wf.to_bf16()) * 2.0f + 1.0f;
  float r0 = ref.at({0, 0}) * 2.0f + 1.0f;
  CHECK(fused.at({0, 0}) == doctest::Approx(r0).epsilon(0.02));

  // non-decode shape (M>1) with a bf16 operand: widen fallback, still correct
  auto a2 = array::from({1.0f, 0.0f, 0.0f, 1.0f}, {2, 2});
  auto w2 = array::from({1.5f, 2.5f, -3.0f, 0.5f}, {2, 2});
  auto g2 = a2.dot(w2.to_bf16());
  CHECK(g2.at({0, 0}) == doctest::Approx(1.5f));
  CHECK(g2.at({1, 1}) == doctest::Approx(0.5f));

  // bf16 feeding a non-dot op widens too (add)
  auto sum = w2.to_bf16().to_f32() + w2;
  CHECK(sum.at({0, 1}) == doctest::Approx(5.0f));
}

TEST_CASE("fused decode attention matches an explicit softmax(qKt)V") {
  // Small, D=128 (the kernel's supported head dim); 2 heads, ctx=5.
  const int64_t H = 2, D = 128, C = 5;
  auto mk = [](tl::shape_t s, unsigned seed) {
    size_t n = 1;
    for (auto d : s) n *= (size_t)d;
    std::vector<float> v(n);
    unsigned st = seed;
    for (auto& x : v) {
      st = st * 1664525u + 1013904223u;
      x = (float)(int)(st >> 9) / (1 << 22) - 1.0f;
    }
    return tl::array::from(std::move(v), std::move(s));
  };
  auto q = mk({H, D}, 1), K = mk({H, C, D}, 2), V = mk({H, C, D}, 3);
  float scale = 1.0f / std::sqrt((float)D);

  auto got = tl::array::attn_decode(q, K, V, scale);
  CHECK(got.shape() == tl::shape_t{H, D});

  // Independent expected value: softmax over ctx of scale*q·K[j], then ·V.
  for (int64_t h = 0; h < H; h++) {
    std::vector<float> s(C);
    float mx = -1e30f;
    for (int64_t j = 0; j < C; j++) {
      float acc = 0;
      for (int64_t d = 0; d < D; d++)
        acc += q.at({h, d}) * K.at({h, j, d});
      s[j] = acc * scale;
      mx = std::max(mx, s[j]);
    }
    float sum = 0;
    for (int64_t j = 0; j < C; j++) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
    for (int64_t d : {0, 37, 127}) {
      float e = 0;
      for (int64_t j = 0; j < C; j++) e += s[j] * V.at({h, j, d});
      e /= sum;
      CHECK(got.at({h, d}) == doctest::Approx(e).epsilon(1e-4));
    }
  }

  // the device kernel against the ref oracle at both head widths, past the
  // tiny-tensor cutoffs, with a context off the kernel's key stride and one
  // of a single key
  for (int64_t d : {64, 128}) {
    for (int64_t c : {2048, 2045, 1}) {
      const float sc = 1.0f / std::sqrt((float)d);
      CHECK(matches_gpu_oracle([&] {
        return tl::array::attn_decode(random_array({8, d}, 850),
                                      random_array({8, c, d}, 851),
                                      random_array({8, c, d}, 852), sc);
      }));
    }
  }

  // shape validation: q must be [H,D] rank-2, K/V rank-3 and equal
  CHECK_THROWS(tl::array::attn_decode(K, K, V, scale));       // q rank 3
  CHECK_THROWS(tl::array::attn_decode(q, K, q.reshape({H, D}), scale));  // V rank 2
}

TEST_CASE("fused prefill attention matches an explicit causal softmax(qKt)V") {
  // D=64 (a kernel head dim), 2 heads, T=7: row t attends the keys 0..t and
  // no further. Same explicit expectation as the decode test, once per row.
  const int64_t H = 2, T = 7, D = 64;
  auto q = random_array({H, T, D}, 810), K = random_array({H, T, D}, 811),
       V = random_array({H, T, D}, 812);
  float scale = 1.0f / std::sqrt((float)D);

  auto got = tl::array::attn_prefill(q, K, V, scale);
  CHECK(got.shape() == tl::shape_t{H, T, D});

  for (int64_t h = 0; h < H; h++) {
    for (int64_t t = 0; t < T; t++) {
      std::vector<float> s(t + 1);
      float mx = -1e30f;
      for (int64_t j = 0; j <= t; j++) {
        float acc = 0;
        for (int64_t d = 0; d < D; d++) acc += q.at({h, t, d}) * K.at({h, j, d});
        s[j] = acc * scale;
        mx = std::max(mx, s[j]);
      }
      float sum = 0;
      for (int64_t j = 0; j <= t; j++) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
      for (int64_t d : {0, 31, 63}) {
        float e = 0;
        for (int64_t j = 0; j <= t; j++) e += s[j] * V.at({h, j, d});
        e /= sum;
        CHECK(got.at({h, t, d}) == doctest::Approx(e).epsilon(1e-4));
      }
    }
  }

  // the own-CPU path (query tiles of 64 rows across the pool, each a sgemm
  // for its scores, a causal softmax, a sgemm for its context) against the
  // scalar reference, at a size past the tiny-tensor cutoffs and at the tile
  // edges: one head, fewer rows than a tile, rows off the tile, one row, and
  // enough rows that the work splits across threads.
  struct { int64_t h, t; } shapes[] = {{3, 40}, {1, 64}, {2, 130}, {1, 1}, {4, 200}};
  int seed = 813;
  for (auto s : shapes) {
    const int sq = seed++, sk = seed++, sv = seed++;  // fixed: build() runs twice
    CHECK(cpu_matches_ref([&] {
      return tl::array::attn_prefill(random_array({s.h, s.t, 64}, sq),
                                     random_array({s.h, s.t, 64}, sk),
                                     random_array({s.h, s.t, 64}, sv), scale);
    }));
  }

  // the device kernel against the ref oracle at both head widths, with rows
  // off every query tile (70) and a single row
  for (int64_t d : {64, 128}) {
    for (int64_t t : {70, 1}) {
      const float sc = 1.0f / std::sqrt((float)d);
      CHECK(matches_gpu_oracle([&] {
        return tl::array::attn_prefill(random_array({3, t, d}, 820),
                                       random_array({3, t, d}, 821),
                                       random_array({3, t, d}, 822), sc);
      }));
    }
  }

  // shape validation: q, K, V must all be [H,T,D] and agree
  CHECK_THROWS(tl::array::attn_prefill(q.reshape({H * T, D}), K, V, scale));
  CHECK_THROWS(tl::array::attn_prefill(q, K, V.reshape({H, D, T}), scale));
}

TEST_CASE("attention and rope read a strided view as its packed copy") {
  // A window along T (or D) is neither contiguous nor at the buffer's head;
  // the row-indexed kernels must see the same values the packed copy holds.
  const int64_t H = 2, T = 5;
  for (int64_t D : {8, 64}) {
    float scale = 1.0f / std::sqrt((float)D);
    auto win = [&](int seed) {
      return random_array({H, T + 3, D}, seed).slice(1, 2, T);
    };
    auto q = win(830), K = win(831), V = win(832);
    CHECK(allclose(tl::array::attn_prefill(q, K, V, scale).eval(),
                   tl::array::attn_prefill(q.clone(), K.clone(), V.clone(),
                                           scale).eval()));
    CHECK(allclose(array::rope(q, 3, 10000.0f).eval(),
                   array::rope(q.clone(), 3, 10000.0f).eval()));
    auto q1 = random_array({H, D + 2}, 833).slice(1, 1, D);
    CHECK(allclose(tl::array::attn_decode(q1, K, V, scale).eval(),
                   tl::array::attn_decode(q1.clone(), K.clone(), V.clone(),
                                          scale).eval()));
    CHECK(allclose(array::rope(q1, 3, 10000.0f).eval(),
                   array::rope(q1.clone(), 3, 10000.0f).eval()));
  }
}

// The attention pullback's query half checked on one device: the own CPU
// (on_gpu false) or the GPU.
static void check_attn_bwd_dq(bool on_gpu, int64_t D = 64) {
  if (on_gpu) tl::use_gpu(); else tl::use_cpu();

  // T sits off the 32-query tile (and the own CPU's 64-row one), so the last
  // block carries rows past the end and the first tile it walks is a partial
  // one.
  const int64_t H = 3, T = 70;
  auto q = random_array({H, T, D}, 900), K = random_array({H, T, D}, 901),
       V = random_array({H, T, D}, 902), dO = random_array({H, T, D}, 903);
  const float scale = 1.0f / std::sqrt((float)D);

  auto out = tl::array::attn_prefill(q, K, V, scale);
  auto got = tl::array::attn_prefill_bwd_dq(q, K, V, dO, out, scale);
  // The own CPU always takes it, and so does a backend that has the kernel —
  // while one without it declines and its caller composes the unfused form,
  // which the gradient tests above cover.
  if (!on_gpu || tl::gpu::has_attn_prefill_dq) REQUIRE(got.has_value());
  if (!got) {
    MESSAGE("no fused attention pullback on this backend — skipping");
    return;
  }
  const array& dq = got->first;
  const array& stats = got->second;
  CHECK(dq.shape() == tl::shape_t{H, T, D});
  CHECK(stats.shape() == tl::shape_t{2, H, T});

  // Row t of head h, spelled out: the causal softmax of its scores, then
  // dq_t = scale · Σ_j P_tj (dP_tj − Δ_t) k_j with dP_tj = dO_t·v_j and
  // Δ_t = Σ_j P_tj dP_tj. L_t is that softmax's m + log l.
  std::vector<float> p(T), dp(T);
  for (int64_t h = 0; h < H; h++) {
    for (int64_t t = 0; t < T; t++) {
      float mx = -1e30f;
      for (int64_t j = 0; j <= t; j++) {
        float s = 0;
        for (int64_t d = 0; d < D; d++) s += q.at({h, t, d}) * K.at({h, j, d});
        p[j] = s * scale;
        mx = std::max(mx, p[j]);
      }
      float sum = 0;
      for (int64_t j = 0; j <= t; j++) {
        p[j] = std::exp(p[j] - mx);
        sum += p[j];
      }
      float delta = 0;
      for (int64_t j = 0; j <= t; j++) {
        p[j] /= sum;
        float g = 0;
        for (int64_t d = 0; d < D; d++) g += dO.at({h, t, d}) * V.at({h, j, d});
        dp[j] = g;
        delta += p[j] * g;
      }
      CHECK(stats.at({0, h, t}) ==
            doctest::Approx(mx + std::log(sum)).epsilon(1e-4));
      CHECK(stats.at({1, h, t}) == doctest::Approx(delta).epsilon(1e-3));
      for (int64_t d : {int64_t{0}, D / 2, D - 1}) {
        float e = 0;
        for (int64_t j = 0; j <= t; j++)
          e += p[j] * (dp[j] - delta) * K.at({h, j, d});
        CHECK(dq.at({h, t, d}) == doctest::Approx(e * scale).epsilon(1e-3));
      }
    }
  }

  // Same shape rule as the forward, extended to the two gradients.
  CHECK_THROWS(tl::array::attn_prefill_bwd_dq(q, K, V, dO.reshape({H, D, T}),
                                              out, scale));
}

TEST_CASE("the fused pullback's dq and logsumexp match explicit softmax math") {
  auto prev = tl::device_;
  check_attn_bwd_dq(false);
  check_attn_bwd_dq(false, 48);  // the own CPU takes any head width
  if (tl::gpu_available()) {
    check_attn_bwd_dq(true);
    check_attn_bwd_dq(true, 128);
  }
  tl::device_ = prev;
}

// The key/value half, on one device as check_attn_bwd_dq.
static void check_attn_bwd_dkv(bool on_gpu, int64_t D = 64) {
  if (on_gpu) tl::use_gpu(); else tl::use_cpu();

  const int64_t H = 3, T = 70;
  auto q = random_array({H, T, D}, 900), K = random_array({H, T, D}, 901),
       V = random_array({H, T, D}, 902), dO = random_array({H, T, D}, 903);
  const float scale = 1.0f / std::sqrt((float)D);

  auto out = tl::array::attn_prefill(q, K, V, scale);
  auto dqs = tl::array::attn_prefill_bwd_dq(q, K, V, dO, out, scale);
  auto got = dqs ? tl::array::attn_prefill_bwd_dkv(q, K, V, dO, dqs->second,
                                                   scale)
                 : std::nullopt;
  if (!on_gpu || tl::gpu::has_attn_prefill_dkv) REQUIRE(got.has_value());  // as above
  if (!got) {
    MESSAGE("no fused attention pullback on this backend — skipping");
    return;
  }
  const array& dK = got->first;
  const array& dV = got->second;
  CHECK(dK.shape() == tl::shape_t{H, T, D});
  CHECK(dV.shape() == tl::shape_t{H, T, D});

  // Key j collects from every query that can see it:
  //   dv_j = Σ_{i≥j} P_ij dO_i,  dk_j = scale · Σ_{i≥j} P_ij (dP_ij − Δ_i) q_i.
  // Building the head's whole causal P keeps the expectation readable.
  std::vector<float> P((size_t)T * T, 0.f), dP((size_t)T * T, 0.f), delta(T);
  for (int64_t h = 0; h < H; h++) {
    for (int64_t i = 0; i < T; i++) {
      float* pr = P.data() + i * T;
      float* dr = dP.data() + i * T;
      float mx = -1e30f;
      for (int64_t j = 0; j <= i; j++) {
        float s = 0;
        for (int64_t d = 0; d < D; d++) s += q.at({h, i, d}) * K.at({h, j, d});
        pr[j] = s * scale;
        mx = std::max(mx, pr[j]);
      }
      float sum = 0;
      for (int64_t j = 0; j <= i; j++) {
        pr[j] = std::exp(pr[j] - mx);
        sum += pr[j];
      }
      delta[i] = 0;
      for (int64_t j = 0; j <= i; j++) {
        pr[j] /= sum;
        float gd = 0;
        for (int64_t d = 0; d < D; d++) gd += dO.at({h, i, d}) * V.at({h, j, d});
        dr[j] = gd;
        delta[i] += pr[j] * gd;
      }
    }
    // The first key (every query sees it), one off the 32-key tile, one past
    // the own CPU's 64-key tile, and the last (one query sees it).
    for (int64_t j : {0, 37, 64, 69}) {
      for (int64_t d : {int64_t{0}, D / 2, D - 1}) {
        float ev = 0, ek = 0;
        for (int64_t i = j; i < T; i++) {
          const float p = P[(size_t)i * T + j];
          ev += p * dO.at({h, i, d});
          ek += p * (dP[(size_t)i * T + j] - delta[i]) * q.at({h, i, d});
        }
        CHECK(dV.at({h, j, d}) == doctest::Approx(ev).epsilon(1e-3));
        CHECK(dK.at({h, j, d}) == doctest::Approx(ek * scale).epsilon(1e-3));
      }
    }
  }

  // stats is [2,H,T], not the forward's output.
  CHECK_THROWS(tl::array::attn_prefill_bwd_dkv(q, K, V, dO, out, scale));
}

TEST_CASE("the fused pullback's dK and dV match explicit softmax math") {
  auto prev = tl::device_;
  check_attn_bwd_dkv(false);
  check_attn_bwd_dkv(false, 48);
  if (tl::gpu_available()) {
    check_attn_bwd_dkv(true);
    check_attn_bwd_dkv(true, 128);
  }
  tl::device_ = prev;
}

TEST_CASE("decode GEMV matches the ref oracle") {
  // a [1,K] · W [K,N] is the decode projection, which the GPU takes through
  // its own GEMV rather than the M=1 GEMM. Qwen-0.5B's two projection shapes,
  // then an N off the kernel's column tile and a single-column weight.
  struct { int64_t k, n; } shapes[] = {
      {896, 4864}, {4864, 896}, {896, 4865}, {896, 1}};
  // Tolerance: a K-long dot summed in a different order (the kernel splits K
  // across threadgroups) differs in the last bits, and an output near zero is
  // cancellation, so the relative error there is meaningless. The absolute
  // agreement below is ~1e-4 on outputs of order 10-100; a kernel that lost or
  // misplaced a term is off by far more than atol.
  const float rtol = 1e-3f, atol = 1e-3f;
  int seed = 870;
  for (auto s : shapes) {
    const int ka = seed++, kw = seed++;  // fixed: build() runs twice
    CHECK(matches_gpu_oracle(
        [&] {
          return random_array({1, s.k}, ka).dot(random_array({s.k, s.n}, kw));
        },
        rtol, atol));
    // bf16 weights take their own kernel. The oracle widens the same bf16
    // values, so this compares the GEMV, not the cast.
    CHECK(matches_gpu_oracle(
        [&] {
          return random_array({1, s.k}, ka)
              .dot(random_array({s.k, s.n}, kw).to_bf16());
        },
        rtol, atol));
  }
  // int4 weights, one threadgroup per output row: K a multiple of the group.
  for (auto s : {std::pair<int64_t, int64_t>{896, 4864},
                 std::pair<int64_t, int64_t>{4864, 1}}) {
    const int ka = seed++, kw = seed++;
    CHECK(matches_gpu_oracle(
        [&] {
          return random_array({1, s.first}, ka)
              .dot(random_array({s.first, s.second}, kw).to_q4());
        },
        rtol, atol));
  }
}

TEST_CASE("q4 weight storage: decode dot + widen fallback vs dequant oracle") {
  // W [K,N], K a multiple of 256; a [1,K]. int4 quant error on random data is
  // large (a small weight in a big-maxabs group rounds coarsely), so the right
  // oracle is a.dot(dequant(Wq)) — this isolates the GEMV/packing from quant
  // error. The GPU kernel and the CPU widen-fallback must both match it.
  const int64_t K = 256, N = 48;
  std::vector<float> av(K), wv(K * N);
  unsigned st = 5;
  auto rnd = [&] {
    st = st * 1664525u + 1013904223u;
    return (float)(int)(st >> 9) / (1 << 22) - 1.0f;
  };
  for (auto& v : av) v = rnd();
  for (auto& v : wv) v = rnd();
  auto a = tl::array::from(av, {1, K});
  auto w = tl::array::from(wv, {K, N});
  auto wq = w.to_q4();
  CHECK(wq.dt() == tl::dtype::q4);
  CHECK(wq.shape() == tl::shape_t{K, N});  // logical shape preserved

  auto deq = wq.to_f32();  // exactly what every path multiplies against
  CHECK(deq.dt() == tl::dtype::f32);
  CHECK(deq.shape() == tl::shape_t{K, N});
  auto ref = a.dot(deq);   // oracle: dot against the dequantized weights
  auto got = a.dot(wq);    // GPU decode GEMV, or widen-fallback on CPU
  CHECK(got.shape() == ref.shape());
  for (int64_t j = 0; j < N; j += 5) {
    CHECK(got.at({0, j}) == doctest::Approx(ref.at({0, j})).epsilon(1e-3));
  }

  // dequant stays within int4's step of the original (per element)
  double maxerr = 0;
  for (int64_t k = 0; k < K; k++)
    for (int64_t j = 0; j < N; j++)
      maxerr = std::max(maxerr, (double)std::fabs(deq.at({k, j}) - w.at({k, j})));
  CHECK(maxerr < 0.2);  // symmetric int4, scale = maxabs/7

  // direct element access on q4 throws (weight-container semantics) — and
  // names the actual storage kind (q4), not a hardcoded "bf16" (raw()/
  // data() used to assume any non-f32 storage was bf16).
  try {
    wq.at({0, 0});
    FAIL("expected a throw");
  } catch (const std::exception& e) {
    CHECK(std::string(e.what()).find("q4") != std::string::npos);
    CHECK(std::string(e.what()).find("bf16") == std::string::npos);
  }

  // views of q4 throw instead of silently reading the packed buffer from its
  // start with the view's shape (the layout has no per-element strides)
  CHECK_THROWS_AS(wq.slice(0, 32, 32), std::logic_error);
  CHECK_THROWS_AS(wq.transpose(), std::logic_error);
  CHECK_THROWS_AS(wq.reshape({N, K}), std::logic_error);

  // clone byte-copies the packed buffer (q4_bytes, not size()*4)
  auto wqc = wq.clone();
  CHECK(wqc.dt() == tl::dtype::q4);
  CHECK(tl::array_equal(wqc.to_f32(), deq));

  // non-decode shape (M=2) widens q4 -> f32; matches the same dequant oracle
  auto a2v = std::vector<float>(2 * K);
  for (auto& v : a2v) v = rnd();
  auto a2 = tl::array::from(a2v, {2, K});
  auto ref2 = a2.dot(deq);
  auto got2 = a2.dot(wq);
  CHECK(got2.at({1, 3}) == doctest::Approx(ref2.at({1, 3})).epsilon(1e-3));
}

TEST_CASE("logsumexp: softmax's denominator, one value per row") {
  auto a = array::from({1, 2, 3, -1, 0, 1}, {2, 3});
  auto out = a.logsumexp(1);
  CHECK(out.shape() == tl::shape_t{2});
  CHECK(out.at({0}) == doctest::Approx(std::log(std::exp(1.0f) +
                                                std::exp(2.0f) +
                                                std::exp(3.0f))));
  CHECK(out.at({1}) == doctest::Approx(std::log(std::exp(-1.0f) +
                                                std::exp(0.0f) +
                                                std::exp(1.0f))));
}

TEST_CASE("logsumexp: stable where a naive sum of exps overflows") {
  auto a = array::from({100, 101, 102}, {1, 3});
  auto out = a.logsumexp(1);
  CHECK(out.at({0}) == doctest::Approx(102.0f + std::log(std::exp(-2.0f) +
                                                         std::exp(-1.0f) +
                                                         1.0f)));
}

TEST_CASE("logsumexp: keepdims, and an axis with no fused kernel of its own") {
  auto a = random_array({4, 5}, 44);
  CHECK(a.logsumexp(1, true).shape() == tl::shape_t{4, 1});
  // Off the last axis array::logsumexp composes; transposing puts the same
  // numbers on the fused path, so the two must agree.
  auto ax0 = a.logsumexp(0);
  auto by_hand = a.transpose().logsumexp(1);
  CHECK(ax0.shape() == tl::shape_t{5});
  for (int64_t j = 0; j < 5; j++) {
    CHECK(ax0.at({j}) == doctest::Approx(by_hand.at({j})));
  }
}

TEST_CASE("logsumexp: own CPU rows match the ref oracle") {
  CHECK(cpu_matches_ref([&] { return random_array({4, 5}, 46).logsumexp(1); }));
  CHECK(cpu_matches_ref([&] { return random_array({7, 33}, 47).logsumexp(1); }));
  CHECK(cpu_matches_ref([&] { return random_array({40, 6}, 48).transpose().logsumexp(1); }));
  CHECK(cpu_matches_ref([&] { return random_array({64, 4099}, 49).logsumexp(1); }));
  // A masked row, and a row with nothing unmasked: the second has no finite
  // max, so the own-CPU path hands it to the fold, which answers -inf.
  const float inf = std::numeric_limits<float>::infinity();
  auto out = array::from({0, -inf, 1, -inf, -inf, -inf}, {2, 3}).logsumexp(1).eval();
  CHECK(out.at({0}) == doctest::Approx(std::log(1.0f + std::exp(1.0f))));
  CHECK(out.at({1}) == -inf);
}

TEST_CASE("logsumexp: GPU dispatch matches the ref oracle") {
  CHECK(matches_gpu_oracle([&] {
    auto a = random_array({8, 512}, 45);
    return a.logsumexp(1);
  }));
}

// xent_bwd on one device: the own CPU (on_gpu false) or the GPU.
static void check_xent_bwd(bool on_gpu) {
  if (on_gpu) tl::use_gpu(); else tl::use_cpu();
  const int64_t N = 8, C = 32;
  auto logits = random_array({N, C}, 46);
  auto targets = array::from({0, 5, 31, 12, 7, 7, 1, 30}, {N});
  auto g = random_array({N}, 47);
  auto got = tl::array::xent_bwd(logits, logits.logsumexp(1), targets, g);
  // The own CPU always takes it, and so do the CUDA and Metal kernels; WebGPU
  // declines and its caller composes the form checked against right here.
  if (!on_gpu) REQUIRE(got.has_value());
#if !defined(TENSORLIB_WEBGPU)
  REQUIRE(got.has_value());
#endif
  if (!got) {
    MESSAGE("no fused cross-entropy pullback on this backend — skipping");
    return;
  }
  CHECK(got->shape() == tl::shape_t{N, C});
  auto onehot = tl::scatter_to_axis(targets, tl::array::ones({N}), C);
  auto want = (logits.softmax() - onehot) * g.reshape({N, 1});
  tl::eval(*got, want);
  for (int64_t i = 0; i < N; i++) {
    for (int64_t j = 0; j < C; j++) {
      CHECK(got->at({i, j}) ==
            doctest::Approx(want.at({i, j})).epsilon(1e-4));
    }
  }
}

TEST_CASE("xent_bwd: the fused pullback matches (softmax - onehot) * g") {
  // A CUDA build with no driver (the CI's fallback job) has the kernel
  // compiled in but no device to run it on, so the device pass sits behind
  // this, not just behind the build's own #if.
  auto prev = tl::device_;
  check_xent_bwd(false);
  if (tl::gpu_available()) check_xent_bwd(true);
  tl::device_ = prev;
}

// Partial RoPE (Qwen3.5: 64 of a 256-wide head) against a host reference: the
// first `rot` dims of each row rotate as a half-split RoPE of width rot, the
// rest pass through. A backend runs its own kernel or the composition (the
// "generic compositions" case holds the two together); this holds either to
// an independent program.
TEST_CASE("partial rope rotates the first rot dims and keeps the rest") {
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;
  tl::use_gpu();
  const int64_t H = 3, T = 4, D = 256, rot = 64, pos = 11;
  const float base = 1e7f;
  array xh = random_array({H * T, D}, 990);
  xh.eval();
  std::vector<float> want((size_t)H * T * D);
  for (int64_t r = 0; r < H * T; r++) {
    const float* x = xh.raw() + r * D;
    float* w = want.data() + r * D;
    for (int64_t j = 0; j < D; j++) w[j] = x[j];
    const double p = (double)(pos + r % T);
    for (int64_t j = 0; j < rot / 2; j++) {
      const double ang = p * std::pow((double)base, -2.0 * j / rot);
      const float c = (float)std::cos(ang), s = (float)std::sin(ang);
      w[j] = x[j] * c - x[j + rot / 2] * s;
      w[j + rot / 2] = x[j] * s + x[j + rot / 2] * c;
    }
  }
  array x = xh.clone();
  x.eval();
  array o = array::empty({H * T, D});
  REQUIRE(tl::gpu::rope(x.device_span(), o.device_span(), H * T, T, D, pos, base, {}, rot));
  tl::gpu::flush();
  CHECK(tl::allclose(o, array::from(std::move(want), {H * T, D}), 1e-5f, 1e-5f));
  tl::device_ = prev;
}

// N rows of K weights in a GGUF block format (ggml type 12 Q4_K, 14 Q6_K, 8
// Q8_0) as one buffer (for the caller to move to the device), and the same rows
// dequantized by the CPU reference. Random bytes are valid blocks; only the half scales are set
// (a random one could be inf or NaN): Q4_K d, dmin at the start, Q6_K d at
// byte 208, Q8_0 d at the start, each a small positive half ~ 2^-5 .. 2^-4.
struct quant_rows {
  array W;               // the blocks, packed into floats
  std::vector<float> w;  // [N,K], dequantized
};
static quant_rows random_quant_rows(int type, int64_t N, int64_t K, unsigned seed) {
  const auto t = static_cast<tl::gguf::ggml_type>(type);
  const bool q4 = t == tl::gguf::ggml_type::q4_k, q8 = t == tl::gguf::ggml_type::q8_0;
  uint64_t elems, block_bytes;
  tl::gguf::ggml_block(t, elems, block_bytes);
  const int64_t nb = K / (int64_t)elems, bb = (int64_t)block_bytes;
  std::mt19937 rng(seed);
  std::vector<unsigned char> bytes((size_t)N * nb * bb);
  for (auto& b : bytes) b = (unsigned char)rng();
  auto set_half = [&](size_t at) {
    const uint16_t h = (uint16_t)(0x2800 + (rng() & 0x3ff));
    std::memcpy(&bytes[at], &h, 2);
  };
  for (int64_t i = 0; i < N * nb; i++) {
    if (q4) {
      set_half((size_t)i * bb);
      set_half((size_t)i * bb + 2);
    } else {
      set_half((size_t)i * bb + (q8 ? 0 : 208));
    }
  }
  quant_rows r;
  r.w.resize((size_t)N * K);
  for (int64_t n = 0; n < N; n++)
    tl::gguf::dequant_range(t, bytes.data() + (size_t)n * nb * bb, 0, (size_t)K,
                            &r.w[(size_t)n * K]);
  std::vector<float> packed((bytes.size() + 3) / 4);
  std::memcpy(packed.data(), bytes.data(), bytes.size());
  r.W = array::from(std::move(packed), {(int64_t)((bytes.size() + 3) / 4)});
  return r;
}

// The generic compositions (gpu_ops.h, tier 1 out of tier 0) against the op
// as the backend runs it. Where the backend has the fused kernel the two are
// different programs that must agree; where it does not, the op is the
// composition and this is the composition against itself, and the model-path
// test above holds it to the array oracle.
TEST_CASE("generic compositions agree with the fused kernels") {
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;
  tl::use_gpu();
  namespace gpu = tl::gpu;
  namespace gen = tl::gpu::generic;
  auto dev = [](const array& a) {
    array c = a.clone();
    c.eval();
    return c;
  };
  auto same = [](const array& got, const array& want, float tol) {
    return tl::allclose(got, want, tol, tol);
  };

  SUBCASE("rmsnorm, rmsnorm_res and swiglu") {
    const int64_t rows = 3, n = 896, ff = 512;
    array x = dev(random_array({rows, n}, 950)), d = dev(random_array({rows, n}, 951));
    array w = dev(random_array({n}, 952));
    array a = array::empty({rows, n}), b = array::empty({rows, n});
    REQUIRE(gpu::rmsnorm(x.device_span(), w.device_span(), a.device_span(), n, 1e-6f, rows));
    REQUIRE(gen::rmsnorm(x.device_span(), w.device_span(), b.device_span(), n, 1e-6f, rows));
    tl::gpu::flush();
    CHECK(same(a, b, 1e-5f));
    // Rows small enough that eps carries the reciprocal (mean(x^2) ~ 1e-8
    // against eps 1e-6): a composition that dropped eps would be off tenfold.
    array tiny = dev(x * 1e-4f);
    REQUIRE(gpu::rmsnorm(tiny.device_span(), w.device_span(), a.device_span(), n, 1e-6f, rows));
    REQUIRE(gen::rmsnorm(tiny.device_span(), w.device_span(), b.device_span(), n, 1e-6f, rows));
    tl::gpu::flush();
    CHECK(same(a, b, 1e-5f));

    array xa = array::empty({rows, n}), ha = array::empty({rows, n});
    array xb = array::empty({rows, n}), hb = array::empty({rows, n});
    REQUIRE(gpu::rmsnorm_res(x.device_span(), d.device_span(), w.device_span(), xa.device_span(),
                             ha.device_span(), n, 1e-6f, rows));
    REQUIRE(gen::rmsnorm_res(x.device_span(), d.device_span(), w.device_span(), xb.device_span(),
                             hb.device_span(), n, 1e-6f, rows));
    tl::gpu::flush();
    CHECK(same(xa, xb, 0.0f));
    CHECK(same(ha, hb, 1e-5f));

    array gu = dev(random_array({rows, 2 * ff}, 953));
    array oa = array::empty({rows, ff}), ob = array::empty({rows, ff});
    REQUIRE(gpu::swiglu(gu.device_span(), oa.device_span(), ff, rows));
    REQUIRE(gen::swiglu(gu.device_span(), ob.device_span(), ff, rows));
    tl::gpu::flush();
    CHECK(same(oa, ob, 1e-5f));
  }

  SUBCASE("the f32 decode GEMV") {
    const int64_t K = 896, N = 1152;
    array a = dev(random_array({1, K}, 954)), B = dev(random_array({K, N}, 955));
    array ya = array::empty({1, N}), yb = array::empty({1, N});
    REQUIRE(gpu::gemv_f32(a.device_span(), B.device_span(), ya.device_span(), N, K));
    REQUIRE(gen::gemv_f32(a.device_span(), B.device_span(), yb.device_span(), N, K));
    tl::gpu::flush();
    CHECK(same(ya, yb, 1e-3f));
  }

  SUBCASE("rmsnorm_rope against rmsnorm then rope, and the gated norm") {
    // Qwen3's full turn (D 128) and Qwen3.5's partial one (D 256, 64 dims), a
    // chunk of T tokens per head. Where the backend fuses them the two routes
    // are different programs that agree to an ulp (Metal's fast math rounds
    // each kernel's products its own way); where it does not,
    // the fused call is the two ops and this is them against themselves.
    struct shape { int64_t D, rot; };
    for (const shape s : {shape{128, 128}, shape{256, 64}}) {
      const int64_t H = 6, T = 5, rows = H * T, pos = 17;
      const float base = 1e6f, eps = 1e-6f;
      array x = random_array({rows, s.D}, 1090), w = dev(random_array({s.D}, 1091));
      // Each side flushed on its own, the reference's norm into a buffer of
      // its own: both turn in place, which a backend whose launches cannot
      // write what they read (WebGPU) must route around, not drop with its
      // batch and leave a side as it was.
      array xa = dev(x), xd = dev(x), xn = array::empty({rows, s.D});
      REQUIRE(gpu::rmsnorm(xd.device_span(), w.device_span(), xn.device_span(), s.D, eps, rows));
      REQUIRE(gpu::rope(xn.device_span(), xn.device_span(), rows, T, s.D, pos, base, {}, s.rot));
      tl::gpu::flush();
      REQUIRE(gpu::rmsnorm_rope(xa.device_span(), w.device_span(), rows, T, s.D, pos, base,
                                s.rot, eps));
      tl::gpu::flush();
      CHECK(same(xa, xn, 1e-7f));
    }
    // silu(z) * rmsnorm(o) * w over o [T*H, D], the gate a row's head of z.
    // The fused form against the composition a decoder runs without it
    // (qwen.h): the gate rows gathered ahead of o into one [z | o] buffer,
    // rmsnorm on o, then swiglu over the pair, to an ulp as rmsnorm_rope;
    // D 300 takes a row past one 256-wide pass, unevenly over the lanes.
    if (gpu::has_gated_rmsnorm) {
      for (const int64_t D : {128, 300}) {
        const int64_t T = 3, H = 4, n = T * H * D, ldz = H * D + 9;
        const float eps = 1e-6f;
        array o = random_array({T * H, D}, 1092), z = random_array({T, ldz}, 1093);
        o.eval();
        z.eval();
        std::vector<float> zo((size_t)(2 * n));
        for (int64_t r = 0; r < T * H; r++)
          for (int64_t d = 0; d < D; d++)
            zo[r * D + d] = z.raw()[(r / H) * ldz + (r % H) * D + d];
        std::copy(o.raw(), o.raw() + n, zo.begin() + n);
        array od = dev(o), zd = dev(z), w = dev(random_array({D}, 1094));
        array buf = dev(array::from(std::move(zo), {2 * n}));
        array oa = array::empty({T * H, D}), ob = array::empty({T * H, D});
        REQUIRE(gpu::gated_rmsnorm(od.device_span(), zd.device_span(), ldz, w.device_span(),
                                   oa.device_span(), false, T, H, D, eps));
        const gpu::span op = buf.device_span().at(n * 4);
        REQUIRE(gpu::rmsnorm(op, w.device_span(), op, D, eps, T * H));
        REQUIRE(gpu::swiglu(buf.device_span(), ob.device_span(), n));
        tl::gpu::flush();
        CHECK(same(oa, ob, 1e-7f));
      }
    }
  }

  SUBCASE("the cache: fill, append, decode and prefill attention") {
    const int64_t D = 64, HKV = 2, HQ = 14, MAXC = 64, T = 20, T2 = 7;
    const float scale = 1.0f / std::sqrt((float)D);
    array K = dev(random_array({HKV, T, D}, 960)), V = dev(random_array({HKV, T, D}, 961));
    array Ka = dev(array::zeros({HKV, MAXC, D})), Va = dev(array::zeros({HKV, MAXC, D}));
    array Kb = dev(array::zeros({HKV, MAXC, D})), Vb = dev(array::zeros({HKV, MAXC, D}));
    REQUIRE(gpu::kv_fill(Ka.device_span(), Va.device_span(), K.device_span(), V.device_span(), T,
                         MAXC, HKV, D));
    REQUIRE(gen::kv_fill(Kb.device_span(), Vb.device_span(), K.device_span(), V.device_span(), T,
                         MAXC, HKV, D, 0));
    array k1 = dev(random_array({HKV, D}, 962)), v1 = dev(random_array({HKV, D}, 963));
    REQUIRE(gpu::kv_append(Ka.device_span(), Va.device_span(), k1.device_span(), v1.device_span(),
                           T, MAXC, HKV, D));
    REQUIRE(gen::kv_append(Kb.device_span(), Vb.device_span(), k1.device_span(), v1.device_span(),
                           T, MAXC, HKV, D));
    tl::gpu::flush();
    CHECK(same(Ka, Kb, 0.0f));
    CHECK(same(Va, Vb, 0.0f));

    // A decode step over the T + 1 cached rows, by both routes.
    array q = dev(random_array({HQ, D}, 964));
    array oa = array::empty({HQ, D}), ob = array::empty({HQ, D});
    REQUIRE(gpu::attn_decode(q.device_span(), Ka.device_span(), Va.device_span(), oa.device_span(),
                             HQ, HKV, T + 1, MAXC, D, scale));
    REQUIRE(gen::attn_decode(q.device_span(), Ka.device_span(), Va.device_span(), ob.device_span(),
                             HQ, HKV, T + 1, MAXC, D, scale, false));
    tl::gpu::flush();
    CHECK(same(oa, ob, 1e-4f));

    // A prefill of T2 more rows after them: the mask has to open the T + 1
    // rows already cached and close the keys past each query.
    array K2 = dev(random_array({HKV, T2, D}, 965)), V2 = dev(random_array({HKV, T2, D}, 966));
    REQUIRE(gpu::kv_fill(Ka.device_span(), Va.device_span(), K2.device_span(), V2.device_span(),
                         T2, MAXC, HKV, D, false, T + 1));
    array qp = dev(random_array({HQ, T2, D}, 967));
    array pa = array::empty({HQ, T2, D}), pb = array::empty({HQ, T2, D});
    REQUIRE(gpu::attn_prefill(qp.device_span(), Ka.device_span(), Va.device_span(),
                              pa.device_span(), HQ, HKV, T2, MAXC, D, scale, false, T + 1));
    REQUIRE(gen::attn_prefill(qp.device_span(), Ka.device_span(), Va.device_span(),
                              pb.device_span(), HQ, HKV, T2, MAXC, D, scale, false, T + 1));
    tl::gpu::flush();
    CHECK(same(pa, pb, 1e-4f));
  }

  SUBCASE("the inference prefill (tensor cores, f16 operands) at each head width") {
    // A chunk after cached rows, its length off every tile: the causal bound
    // across the tile seams and the zero-filled tail. f16 operands hold it to
    // ~1e-3 of the exact composition; a backend without the form runs the
    // exact op itself.
    for (int64_t D : {64, 128, 256}) {
      const int64_t HKV = 2, HQ = 8, MAXC = 160, P0 = 37, T = 77;
      const float scale = 1.0f / std::sqrt((float)D);
      array Kc = dev(random_array({HKV, MAXC, D}, 1010)), Vc = dev(random_array({HKV, MAXC, D}, 1011));
      array qp = dev(random_array({HQ, T, D}, 1012));
      array pa = array::empty({HQ, T, D}), pb = array::empty({HQ, T, D});
      REQUIRE(gpu::attn_prefill_f16(qp.device_span(), Kc.device_span(), Vc.device_span(),
                                    pa.device_span(), HQ, HKV, T, MAXC, D, scale, false, P0));
      REQUIRE(gen::attn_prefill(qp.device_span(), Kc.device_span(), Vc.device_span(),
                                pb.device_span(), HQ, HKV, T, MAXC, D, scale, false, P0));
      tl::gpu::flush();
      CHECK(same(pa, pb, 2e-3f));
    }
  }

  SUBCASE("the cache at a head width no kernel has (Qwen3.5's 256)") {
    // The ops take the composition rather than declining: a decoder ignores
    // their bool, so a decline would leave the cache silently unwritten.
    const int64_t D = 256, HKV = 2, HQ = 8, MAXC = 16, T = 5;
    const float scale = 1.0f / std::sqrt((float)D);
    array K = dev(random_array({HKV, T, D}, 1000)), V = dev(random_array({HKV, T, D}, 1001));
    array Ka = dev(array::zeros({HKV, MAXC, D})), Va = dev(array::zeros({HKV, MAXC, D}));
    array Kb = dev(array::zeros({HKV, MAXC, D})), Vb = dev(array::zeros({HKV, MAXC, D}));
    REQUIRE(gpu::kv_fill(Ka.device_span(), Va.device_span(), K.device_span(), V.device_span(), T,
                         MAXC, HKV, D));
    REQUIRE(gen::kv_fill(Kb.device_span(), Vb.device_span(), K.device_span(), V.device_span(), T,
                         MAXC, HKV, D, 0));
    array k1 = dev(random_array({HKV, D}, 1002)), v1 = dev(random_array({HKV, D}, 1003));
    REQUIRE(gpu::kv_append(Ka.device_span(), Va.device_span(), k1.device_span(), v1.device_span(),
                           T, MAXC, HKV, D));
    REQUIRE(gen::kv_append(Kb.device_span(), Vb.device_span(), k1.device_span(), v1.device_span(),
                           T, MAXC, HKV, D));
    tl::gpu::flush();
    CHECK(same(Ka, Kb, 0.0f));
    CHECK(same(Va, Vb, 0.0f));
    array q = dev(random_array({HQ, D}, 1004));
    array oa = array::empty({HQ, D}), ob = array::empty({HQ, D});
    REQUIRE(gpu::attn_decode(q.device_span(), Ka.device_span(), Va.device_span(), oa.device_span(),
                             HQ, HKV, T + 1, MAXC, D, scale));
    REQUIRE(gen::attn_decode(q.device_span(), Kb.device_span(), Vb.device_span(), ob.device_span(),
                             HQ, HKV, T + 1, MAXC, D, scale, false));
    tl::gpu::flush();
    CHECK(same(oa, ob, 1e-4f));
  }

  SUBCASE("Gated DeltaNet: ssm_conv and gated_delta") {
    // Qwen3.5-0.8B's shapes: 16 heads of 128, a 4-tap conv over q|k|v. The
    // conv's input rows are a fused projection's, wider than C.
    const int64_t H = 16, dk = 128, dv = 128, K = 4, C = 2 * H * dk + H * dv;
    const int64_t T = 5, ld = C + 3 * H;
    array w = dev(random_array({K, C}, 980));
    array r = random_array({2 * H}, 984);
    r.eval();
    std::vector<float> adt_h(r.raw(), r.raw() + 2 * H);
    for (int64_t h = 0; h < H; h++) adt_h[h] = -std::fabs(adt_h[h]) - 0.1f;  // A < 0
    array adt = dev(array::from(std::move(adt_h), {2 * H}));
    array x = dev(random_array({T, ld}, 990));
    array ab = dev(random_array({T, ld}, 983));  // alpha|beta at each row's start
    array cs0 = dev(random_array({K - 1, C}, 981));
    array S0 = dev(random_array({H, dk, dv}, 986) * 0.1f);
    // Three routes from the same state: all T tokens in one call (the
    // prefill's), one call a token (the decode's), and the composition.
    array csa = dev(cs0), csb = dev(cs0), csc = dev(cs0);
    array Sa = dev(S0), Sb = dev(S0), Sc = dev(S0);
    array ca = array::empty({T, C}), cb = array::empty({T, C}), cc = array::empty({T, C});
    array oa = array::empty({T, H * dv}), ob = array::empty({T, H * dv}),
          oc = array::empty({T, H * dv});
    REQUIRE(gpu::ssm_conv(x.device_span(), ld, csa.device_span(), w.device_span(),
                          ca.device_span(), C, K, T));
    REQUIRE(gen::ssm_conv(x.device_span(), ld, csc.device_span(), w.device_span(),
                          cc.device_span(), C, K, T));
    for (int64_t t = 0; t < T; t++) {
      REQUIRE(gpu::ssm_conv(x.device_span().at(t * ld * 4), ld, csb.device_span(),
                            w.device_span(), cb.device_span().at(t * C * 4), C, K));
    }
    REQUIRE(gpu::gated_delta(ca.device_span(), ab.device_span(), ld, adt.device_span(),
                             Sa.device_span(), oa.device_span(), H, dk, dv, T, 1e-6f));
    REQUIRE(gen::gated_delta(cc.device_span(), ab.device_span(), ld, adt.device_span(),
                             Sc.device_span(), oc.device_span(), H, dk, dv, T, 1e-6f));
    for (int64_t t = 0; t < T; t++) {
      REQUIRE(gpu::gated_delta(cb.device_span().at(t * C * 4), ab.device_span().at(t * ld * 4),
                               0, adt.device_span(), Sb.device_span(),
                               ob.device_span().at(t * H * dv * 4), H, dk, dv, 1, 1e-6f));
    }
    tl::gpu::flush();
    CHECK(same(ca, cb, 0.0f));  // the same kernel, a token at a time
    CHECK(same(csa, csb, 0.0f));
    CHECK(same(oa, ob, 0.0f));
    CHECK(same(Sa, Sb, 0.0f));
    CHECK(same(ca, cc, 1e-5f));  // the kernel against the composition
    CHECK(same(csa, csc, 0.0f));
    // Two tokens, fewer than the K-1 the window holds: the state left behind
    // keeps one row of the old one.
    {
      array s2a = dev(cs0), s2b = dev(cs0);
      array c2a = array::empty({2, C}), c2b = array::empty({2, C});
      REQUIRE(gpu::ssm_conv(x.device_span(), ld, s2a.device_span(), w.device_span(),
                            c2a.device_span(), C, K, 2));
      for (int64_t t = 0; t < 2; t++) {
        REQUIRE(gpu::ssm_conv(x.device_span().at(t * ld * 4), ld, s2b.device_span(),
                              w.device_span(), c2b.device_span().at(t * C * 4), C, K));
      }
      tl::gpu::flush();
      CHECK(same(c2a, c2b, 0.0f));
      CHECK(same(s2a, s2b, 0.0f));
    }
    CHECK(same(oa, oc, 1e-4f));
    CHECK(same(Sa, Sc, 1e-4f));
  }

  SUBCASE("the gated delta rule over a prefill: chunked against token by token") {
    // 77 tokens: past one chunk, with a partial last one. The chunked form
    // regroups the same f32 sums, so the two agree to rounding, outputs and
    // the state left behind.
    const int64_t H = 16, dk = 128, dv = 128, C = 2 * H * dk + H * dv, T = 77;
    const int64_t ld = 2 * H + 5;
    array r = random_array({2 * H}, 1020);
    r.eval();
    std::vector<float> adt_h(r.raw(), r.raw() + 2 * H);
    for (int64_t h = 0; h < H; h++) adt_h[h] = -std::fabs(adt_h[h]) - 0.1f;  // A < 0
    array adt = dev(array::from(std::move(adt_h), {2 * H}));
    array qkv = dev(random_array({T, C}, 1021)), ab = dev(random_array({T, ld}, 1022));
    array S0 = dev(random_array({H, dk, dv}, 1023) * 0.1f), Sa = dev(S0), Sb = dev(S0);
    array oa = array::empty({T, H * dv}), ob = array::empty({T, H * dv});
    REQUIRE(gpu::gated_delta(qkv.device_span(), ab.device_span(), ld, adt.device_span(),
                             Sa.device_span(), oa.device_span(), H, dk, dv, T, 1e-6f));
    for (int64_t t = 0; t < T; t++) {
      REQUIRE(gpu::gated_delta(qkv.device_span().at(t * C * 4), ab.device_span().at(t * ld * 4),
                               0, adt.device_span(), Sb.device_span(),
                               ob.device_span().at(t * H * dv * 4), H, dk, dv, 1, 1e-6f));
    }
    tl::gpu::flush();
    CHECK(same(oa, ob, 1e-4f));
    CHECK(same(Sa, Sb, 1e-4f));
  }

  SUBCASE("the gated delta rule with fewer key heads than value heads") {
    // Qwen3.8-27B: 16 key heads for 48 value heads, value head h reading key
    // head h % Hk. The reference expands q and k to a head each (the layout
    // the rule was built for) and runs the equal-heads path, per token and
    // chunked; the key-head form must agree, on the backend and in the composition.
    const int64_t H = 6, Hk = 2, dk = 128, dv = 128, T = 77;
    const int64_t C = 2 * Hk * dk + H * dv, Cf = 2 * H * dk + H * dv, ld = 2 * H + 5;
    array r = random_array({2 * H}, 1030);
    r.eval();
    std::vector<float> adt_h(r.raw(), r.raw() + 2 * H);
    for (int64_t h = 0; h < H; h++) adt_h[h] = -std::fabs(adt_h[h]) - 0.1f;  // A < 0
    array adt = dev(array::from(std::move(adt_h), {2 * H}));
    array qkv = random_array({T, C}, 1031), ab = dev(random_array({T, ld}, 1032));
    qkv.eval();
    std::vector<float> full((size_t)T * Cf);
    for (int64_t t = 0; t < T; t++) {
      const float* src = qkv.raw() + t * C;
      float* dst = full.data() + t * Cf;
      for (int64_t h = 0; h < H; h++) {  // q, then k, a copy of key head h % Hk
        std::copy(src + (h % Hk) * dk, src + (h % Hk + 1) * dk, dst + h * dk);
        std::copy(src + Hk * dk + (h % Hk) * dk, src + Hk * dk + (h % Hk + 1) * dk,
                  dst + H * dk + h * dk);
      }
      std::copy(src + 2 * Hk * dk, src + C, dst + 2 * H * dk);
    }
    array qkv_d = dev(qkv), full_d = dev(array::from(std::move(full), {T, Cf}));
    array S0 = dev(random_array({H, dk, dv}, 1033) * 0.1f);
    array Sref = dev(S0), Sa = dev(S0), Sb = dev(S0), Sc = dev(S0);
    array oref = array::empty({T, H * dv}), oa = array::empty({T, H * dv}),
          ob = array::empty({T, H * dv}), oc = array::empty({T, H * dv});
    REQUIRE(gpu::gated_delta(full_d.device_span(), ab.device_span(), ld, adt.device_span(),
                             Sref.device_span(), oref.device_span(), H, dk, dv, T, 1e-6f));
    REQUIRE(gpu::gated_delta(qkv_d.device_span(), ab.device_span(), ld, adt.device_span(),
                             Sa.device_span(), oa.device_span(), H, dk, dv, T, 1e-6f, Hk));
    for (int64_t t = 0; t < T; t++) {  // one token at a time: the token kernel
      REQUIRE(gpu::gated_delta(qkv_d.device_span().at(t * C * 4), ab.device_span().at(t * ld * 4),
                               0, adt.device_span(), Sb.device_span(),
                               ob.device_span().at(t * H * dv * 4), H, dk, dv, 1, 1e-6f, Hk));
    }
    REQUIRE(gen::gated_delta(qkv_d.device_span(), ab.device_span(), ld, adt.device_span(),
                             Sc.device_span(), oc.device_span(), H, dk, dv, T, 1e-6f, Hk));
    tl::gpu::flush();
    CHECK(same(oa, oref, 1e-4f));
    CHECK(same(Sa, Sref, 1e-4f));
    CHECK(same(ob, oref, 1e-4f));
    CHECK(same(Sb, Sref, 1e-4f));
    CHECK(same(oc, oref, 1e-4f));
    CHECK(same(Sc, Sref, 1e-4f));
  }

  SUBCASE("GGML block-quantized dequantization is bit for bit ggml's") {
    // One block of each format, the 8th of a Qwen3-0.6B Q4_K_M tensor; the
    // hash is over the 256 float bit patterns ggml's dequantize_row_q4_K /
    // _q6_K give (checked against libggml over every block-quantized value of two
    // models, 7.5e8 of them, none differing).
    static const unsigned char q4k_block[144] = {
      198, 8, 211, 20, 238, 244, 255, 240, 171, 236, 255, 233, 134, 147, 7, 221,
      35, 180, 91, 134, 100, 200, 9, 51, 195, 5, 202, 146, 98, 27, 179, 39,
      163, 244, 175, 153, 158, 164, 117, 47, 72, 124, 160, 18, 13, 64, 119, 103,
      135, 106, 23, 137, 44, 240, 165, 134, 85, 10, 7, 108, 131, 56, 167, 92,
      129, 163, 86, 28, 59, 117, 157, 134, 117, 101, 55, 164, 201, 75, 126, 69,
      58, 169, 130, 104, 248, 200, 79, 128, 167, 181, 12, 72, 106, 89, 196, 67,
      205, 170, 226, 101, 48, 162, 151, 185, 137, 187, 149, 104, 130, 105, 102, 240,
      210, 162, 84, 73, 95, 128, 59, 214, 92, 100, 10, 152, 124, 246, 217, 123,
      138, 106, 74, 124, 81, 118, 166, 41, 185, 150, 165, 180, 169, 133, 67, 152};
    static const unsigned char q80_block[34] = {
      85, 14, 199, 194, 180, 240, 5, 250, 201, 23, 216, 22, 90, 250, 199, 57,
      95, 56, 63, 127, 199, 9, 216, 221, 178, 57, 247, 252, 169, 216, 51, 70,
      51, 39};
    static const unsigned char q6k_block[210] = {
      201, 96, 163, 73, 13, 3, 70, 136, 56, 147, 172, 242, 209, 18, 168, 255,
      19, 208, 98, 9, 212, 63, 89, 128, 63, 77, 222, 22, 73, 207, 131, 49,
      201, 67, 166, 180, 61, 8, 32, 188, 137, 177, 198, 57, 205, 38, 50, 169,
      116, 17, 117, 24, 232, 21, 128, 99, 57, 174, 214, 244, 185, 11, 15, 51,
      99, 85, 236, 120, 25, 202, 17, 52, 205, 228, 59, 201, 196, 69, 150, 43,
      1, 30, 209, 149, 27, 211, 203, 163, 195, 204, 100, 246, 147, 67, 145, 106,
      60, 117, 92, 245, 84, 26, 206, 77, 52, 134, 1, 62, 180, 178, 74, 192,
      33, 124, 183, 190, 14, 155, 191, 5, 246, 103, 113, 237, 164, 254, 205, 80,
      80, 173, 215, 105, 244, 14, 82, 80, 156, 225, 222, 168, 68, 227, 92, 161,
      135, 195, 84, 53, 36, 150, 186, 156, 153, 84, 23, 159, 236, 107, 182, 90,
      241, 185, 102, 21, 77, 185, 16, 231, 73, 106, 16, 145, 69, 229, 218, 86,
      180, 116, 35, 122, 79, 87, 149, 37, 169, 148, 154, 137, 171, 89, 10, 163,
      185, 90, 166, 114, 128, 133, 95, 178, 125, 100, 134, 81, 109, 171, 132, 160,
      47, 129};
    auto fnv = [](const float* y) {
      uint64_t h = 1469598103934665603ull;
      for (int i = 0; i < 256; i++) {
        uint32_t b;
        std::memcpy(&b, &y[i], 4);
        for (int k = 0; k < 4; k++) h = (h ^ ((b >> (8 * k)) & 0xff)) * 1099511628211ull;
      }
      return h;
    };
    float y[256];
    tl::gguf::dequant_range(tl::gguf::ggml_type::q4_k, q4k_block, 0, 256, y);
    CHECK(fnv(y) == 0x84924d0639eee2c3ull);
    {  // Q8_0: one 32-weight block, hashed over its 32 values
      float y8[256] = {};
      tl::gguf::dequant_range(tl::gguf::ggml_type::q8_0, q80_block, 0, 32, y8);
      uint64_t h = 1469598103934665603ull;
      for (int i = 0; i < 32; i++) {
        uint32_t b;
        std::memcpy(&b, &y8[i], 4);
        for (int k = 0; k < 4; k++) h = (h ^ ((b >> (8 * k)) & 0xff)) * 1099511628211ull;
      }
      CHECK(h == 0x393d7402241899c6ull);
    }
    tl::gguf::dequant_range(tl::gguf::ggml_type::q6_k, q6k_block, 0, 256, y);
    CHECK(fnv(y) == 0x1da2050114e9c32bull);
  }

  SUBCASE("GEMV over block-quantized rows against the dequantized weights") {
    // Only a backend that reads block-quantized rows natively runs this; no
    // composition stands in for it, and a model refuses such weights without it.
    if (gpu::has_gemv_quant) {
      for (int type : {12, 14, 8}) {
        const int64_t N = 37, K = 1024;
        const quant_rows q = random_quant_rows(type, N, K, 7 + type);
        array a = random_array({K}, 1040 + type);
        a.eval();
        std::vector<double> ref(N, 0.0);
        for (int64_t n = 0; n < N; n++)
          for (int64_t k = 0; k < K; k++) ref[n] += (double)q.w[(size_t)n * K + k] * a.raw()[k];
        array W = dev(q.W), ad = dev(a), y = array::empty({N});
        REQUIRE(gpu::gemv_quant(type, ad.device_span(), W.device_span(), y.device_span(), N, K));
        // A Q4_K weight off its 4-byte alignment is refused, not misread.
        if (type == 12)
          CHECK_FALSE(gpu::gemv_quant(type, ad.device_span(), W.device_span().at(2),
                                      y.device_span(), N, K));
        // So is an activation off 16 bytes (the kernels' vector loads).
        CHECK_FALSE(gpu::gemv_quant(type, ad.device_span().at(4), W.device_span(),
                                    y.device_span(), N, K));
        tl::gpu::flush();
        y.eval();
        double worst = 0;
        for (int64_t n = 0; n < N; n++)
          worst = std::max(worst, std::fabs(y.raw()[n] - ref[n]) / (1.0 + std::fabs(ref[n])));
        CHECK(worst < 1e-5);
      }
    }
  }

  SUBCASE("GEMM over block-quantized rows against the dequantized weights") {
    // The prefill's form: C's columns [off, off + n) of rows ldc apart, the
    // columns around them left as they were. M on both sides of the
    // backend's tile choice.
    if (gpu::has_gemm_quant) {
      for (int type : {12, 14, 8}) {
        for (int64_t M : {37, 100}) {
          const int64_t N = 70, K = 512, ldc = N + 6, off = 2;
          const quant_rows q = random_quant_rows(type, N, K, 70 + type);
          array a = random_array({M, K}, 1060 + type);
          a.eval();
          array W = dev(q.W), ad = dev(a);
          array C = dev(array::from(std::vector<float>((size_t)(M * ldc), 7.0f), {M, ldc}));
          REQUIRE(gpu::gemm_quant(type, false, ad.device_span(), W.device_span(),
                                  C.device_span().at(off * 4), M, N, K, ldc));
          tl::gpu::flush();
          C.eval();
          double worst = 0;
          bool kept = true;
          for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < ldc; j++) {
              const float got = C.raw()[i * ldc + j];
              if (j < off || j >= off + N) {
                kept = kept && got == 7.0f;
                continue;
              }
              double ref = 0, mag = 0;
              for (int64_t k = 0; k < K; k++) {
                const double t = (double)q.w[(size_t)(j - off) * K + k] * a.raw()[i * K + k];
                ref += t;
                mag += std::fabs(t);
              }
              worst = std::max(worst, std::fabs(got - ref) / mag);
            }
          }
          // f32 sums, ~2e-7 of the terms' magnitude here; a misread weight
          // is off by whole terms.
          CHECK(worst < 2e-6);
          CHECK(kept);
        }
      }
    }
  }

  SUBCASE("Q4_K GEMV on int8 activations against the float one") {
    // Only a backend with the int8-activation GEMV has a second result to compare.
    if (gpu::has_gemv_quant && gpu::int8_activations) {
      // The activation rounded to 8 bits per 32 values (llama.cpp's q8_1): the
      // result differs from the float kernel's by that rounding, ~0.5% of the
      // result's rms on random data, and not by more.
      const int64_t N = 41, K = 2048, nb = K / 256, bb = 144;
      std::mt19937 rng(77);
      std::vector<unsigned char> bytes((size_t)N * nb * bb);
      for (auto& b : bytes) b = (unsigned char)rng();
      for (int64_t i = 0; i < N * nb; i++)
        for (size_t at : {(size_t)i * bb, (size_t)i * bb + 2}) {
          const uint16_t h = (uint16_t)(0x2800 + (rng() & 0x3ff));
          std::memcpy(&bytes[at], &h, 2);
        }
      std::vector<float> packed((bytes.size() + 3) / 4);
      std::memcpy(packed.data(), bytes.data(), bytes.size());
      array W = dev(array::from(std::move(packed), {(int64_t)((bytes.size() + 3) / 4)}));
      array a = dev(random_array({K}, 1077));
      array qa = array::empty({K / 4 + K / 32 * 2});  // K int8, then K/32 (scale, sum)
      array yf = array::empty({N}), yq = array::empty({N});
      REQUIRE(gpu::gemv_quant(12, a.device_span(), W.device_span(), yf.device_span(), N, K));
      REQUIRE(gpu::quantize_act_q8(a.device_span(), qa.device_span(), K));
      REQUIRE(gpu::gemv_q4k_q8(qa.device_span(), W.device_span(), yq.device_span(), N, K));
      tl::gpu::flush();
      yf.eval();
      yq.eval();
      double sq = 0, ref = 0;
      for (int64_t n = 0; n < N; n++) {
        const double d = yq.raw()[n] - yf.raw()[n];
        sq += d * d;
        ref += (double)yf.raw()[n] * yf.raw()[n];
      }
      CHECK(std::sqrt(sq / ref) < 2e-2);
      CHECK(std::sqrt(sq / ref) > 0);  // it is a different computation, not a copy
    }
  }

  SUBCASE("rope, split_heads, merge_heads and argmax") {
    const int64_t H = 14, T = 5, D = 64, ld = H * D;
    array x = dev(random_array({H * T, D}, 970)), b = dev(random_array({H * T, D}, 971));
    array ra = array::empty({H * T, D}), rb = array::empty({H * T, D});
    REQUIRE(gpu::rope(x.device_span(), ra.device_span(), H * T, T, D, 37, 1e6f, b.device_span()));
    REQUIRE(gen::rope(x.device_span(), rb.device_span(), H * T, T, D, 37, 1e6f, b.device_span(), D));
    tl::gpu::flush();
    CHECK(same(ra, rb, 1e-4f));
    REQUIRE(gpu::rope(x.device_span(), ra.device_span(), H * T, T, D, 37, 1e6f));
    REQUIRE(gen::rope(x.device_span(), rb.device_span(), H * T, T, D, 37, 1e6f, {}, D));
    tl::gpu::flush();
    CHECK(same(ra, rb, 1e-4f));
    // Partial (a quarter of the row, as Qwen3.5), with the bias riding along
    // the pass-through dims too.
    REQUIRE(gpu::rope(x.device_span(), ra.device_span(), H * T, T, D, 37, 1e6f,
                      b.device_span(), D / 4));
    REQUIRE(gen::rope(x.device_span(), rb.device_span(), H * T, T, D, 37, 1e6f,
                      b.device_span(), D / 4));
    tl::gpu::flush();
    CHECK(same(ra, rb, 1e-4f));

    array src = dev(random_array({T, ld}, 972)), bias = dev(random_array({H, D}, 973));
    array sa = array::empty({H, T, D}), sb = array::empty({H, T, D});
    REQUIRE(gpu::split_heads(src.device_span(), bias.device_span(), sa.device_span(), T, ld, 0, H, D));
    REQUIRE(gen::split_heads(src.device_span(), bias.device_span(), sb.device_span(), T, ld, 0, H, D));
    tl::gpu::flush();
    CHECK(same(sa, sb, 0.0f));
    REQUIRE(gpu::split_heads(src.device_span(), {}, sa.device_span(), T, ld, 0, H, D));
    REQUIRE(gen::split_heads(src.device_span(), {}, sb.device_span(), T, ld, 0, H, D));
    array ma = array::empty({T, ld}), mb = array::empty({T, ld});
    REQUIRE(gpu::merge_heads(sa.device_span(), ma.device_span(), T, H, D));
    REQUIRE(gen::merge_heads(sa.device_span(), mb.device_span(), T, H, D));
    tl::gpu::flush();
    CHECK(same(sa, sb, 0.0f));
    CHECK(same(ma, mb, 0.0f));
    CHECK(same(ma, src, 0.0f));

    std::vector<float> v(10000, -1.0f);
    v[4321] = v[4322] = 5.0f;
    array a = dev(array::from(v, {(int64_t)v.size()}));
    int64_t ia = -1, ib = -1;
    REQUIRE(gpu::argmax(a.device_span(), (int64_t)v.size(), &ia));
    REQUIRE(gen::argmax(a.device_span(), (int64_t)v.size(), &ib));
    CHECK(ia == 4321);
    CHECK(ib == 4321);
  }

  SUBCASE("xent_bwd and adam_step") {
    const int64_t rows = 5, cols = 37;
    array x = dev(random_array({rows, cols}, 980));
    array lse = dev(random_array({rows}, 981));
    array tgt = dev(array::from({3, 0, 36, 17, 9}, {rows}));
    array g = dev(random_array({rows}, 982));
    array oa = array::empty({rows, cols}), ob = array::empty({rows, cols});
    REQUIRE(gpu::xent_bwd(x.device_span(), lse.device_span(), tgt.device_span(),
                          g.device_span(), oa.device_span(), rows, cols));
    REQUIRE(gen::xent_bwd(x.device_span(), lse.device_span(), tgt.device_span(),
                          g.device_span(), ob.device_span(), rows, cols));
    tl::gpu::flush();
    CHECK(same(oa, ob, 1e-5f));

    const int64_t n = 200;
    auto p0 = random_array({n}, 983), m0 = random_array({n}, 984);
    auto v0 = tl::pow(random_array({n}, 985), 2.0f).eval();  // v is a square
    array grad = dev(random_array({n}, 986));
    array pa = dev(p0), ma = dev(m0), va = dev(v0);
    array pb = dev(p0), mb = dev(m0), vb = dev(v0);
    const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, lr_over_bc1 = 0.01f,
                inv_bc2 = 1.05f;
    REQUIRE(gpu::adam_step(pa.device_span(), ma.device_span(), va.device_span(),
                           grad.device_span(), n, beta1, beta2, eps,
                           lr_over_bc1, inv_bc2));
    REQUIRE(gen::adam_step(pb.device_span(), mb.device_span(), vb.device_span(),
                           grad.device_span(), n, beta1, beta2, eps,
                           lr_over_bc1, inv_bc2));
    tl::gpu::flush();
    CHECK(same(pa, pb, 1e-4f));
    CHECK(same(ma, mb, 1e-5f));
    CHECK(same(va, vb, 1e-4f));
  }

  tl::device_ = prev;
}
