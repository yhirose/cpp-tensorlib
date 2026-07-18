#include "doctest.h"

#include <tensorlib.h>

#include <random>

using tl::array;

namespace {

array random_array(tl::shape_t shape, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(d);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return array::from(std::move(v), std::move(shape));
}

// Evaluate `build()` twice — accelerated and forced through the ref oracle —
// and require the results to agree.
template <typename F>
bool matches_oracle(F build, float rtol = 1e-4f, float atol = 1e-5f) {
  tl::use_accelerate_ = true;
  auto fast = build().eval();
  // Oracle = ref::: disable both accelerated CPU backends (accel and the
  // own-CPU GEMM), else dot would be validated against cpu:: not ref::.
  tl::use_accelerate_ = false;
  tl::cpu::enabled_ = false;
  auto oracle = build().eval();
  tl::use_accelerate_ = true;
  tl::cpu::enabled_ = true;
  return tl::allclose(fast, oracle, rtol, atol);
}

// Evaluate `build()` through the own-CPU backend (accel off, cpu on) and
// through ref (both off); require agreement. Validates cpu::sgemm — on any
// platform, since it never touches Metal/Accelerate.
template <typename F>
bool cpu_matches_ref(F build, float rtol = 2e-3f, float atol = 2e-4f) {
  tl::use_accelerate_ = false;
  tl::cpu::enabled_ = true;
  auto fast = build().eval();
  tl::cpu::enabled_ = false;
  auto oracle = build().eval();
  tl::use_accelerate_ = true;
  tl::cpu::enabled_ = true;
  return tl::allclose(fast, oracle, rtol, atol);
}

// Evaluate `build()` in gpu mode and forced through the ref oracle; require
// agreement. No-op (trivially true) where no Metal device exists.
template <typename F>
bool matches_gpu_oracle(F build, float rtol = 1e-4f, float atol = 1e-5f) {
  if (!tl::gpu_available()) return true;
  auto prev = tl::device_;
  tl::use_gpu();
  auto fast = build().eval();
  tl::use_cpu();
  tl::use_accelerate_ = false;
  tl::cpu::enabled_ = false;
  auto oracle = build().eval();
  tl::use_accelerate_ = true;
  tl::cpu::enabled_ = true;
  tl::device_ = prev;
  return tl::allclose(fast, oracle, rtol, atol);
}

}  // namespace

TEST_CASE("creation and introspection") {
  auto a = array::zeros({2, 3});
  CHECK(a.rank() == 2);
  CHECK(a.size() == 6);
  CHECK(a.contiguous());
  CHECK(a.at({1, 2}) == 0.0f);

  auto b = array::full({2, 2}, 7.0f);
  CHECK(b.at({1, 1}) == 7.0f);

  auto c = array::from({1, 2, 3, 4}, {2, 2});
  CHECK(c.at({0, 1}) == 2.0f);
  CHECK(c.at({1, 0}) == 3.0f);

  // rank-0 scalar
  auto s = array::full({}, 5.0f);
  CHECK(s.size() == 1);
  CHECK(s.item() == 5.0f);

  CHECK_THROWS(array::from({1, 2, 3}, {2, 2}));
}

TEST_CASE("elementwise with broadcasting") {
  auto a = array::from({1, 2, 3, 4, 5, 6}, {2, 3});
  auto row = array::from({10, 20, 30});

  auto c = a + row;  // (2,3) + (3)
  CHECK(c.at({0, 0}) == 11.0f);
  CHECK(c.at({1, 2}) == 36.0f);

  auto col = array::from({100, 200}, {2, 1});
  auto d = a * col;  // (2,3) * (2,1)
  CHECK(d.at({0, 2}) == 300.0f);
  CHECK(d.at({1, 0}) == 800.0f);

  auto e = 2.0f * a - 1.0f;
  CHECK(e.at({0, 0}) == 1.0f);
  CHECK(e.at({1, 2}) == 11.0f);

  auto p = tl::pow(a, 2.0f);
  CHECK(p.at({1, 0}) == 16.0f);

  CHECK_THROWS(a + array::zeros({2, 4}));
}

TEST_CASE("views: transpose / reshape / slice are zero-copy") {
  auto a = array::from({1, 2, 3, 4, 5, 6}, {2, 3});

  auto t = a.transpose();
  CHECK(t.shape() == tl::shape_t{3, 2});
  CHECK(!t.contiguous());
  CHECK(t.at({2, 0}) == 3.0f);
  CHECK(t.at({0, 1}) == 4.0f);

  // ops on a transposed view read through strides
  auto tt = t + 0.0f;
  CHECK(tt.contiguous());
  CHECK(tt.at({2, 1}) == 6.0f);

  auto r = a.reshape({3, 2});
  CHECK(r.at({2, 1}) == 6.0f);

  auto s = a.slice(1, 1);
  CHECK(s.shape() == tl::shape_t{1, 3});
  CHECK(s.at({0, 0}) == 4.0f);

  // views share storage with the base
  a.data()[0] = 100.0f;
  CHECK(t.at({0, 0}) == 100.0f);
  CHECK(s.at({0, 0}) == 4.0f);  // slice starts at row 1, unaffected
}

TEST_CASE("views over a lazy source defer without a batch boundary") {
  // A view of a still-lazy computation composes strides at eval instead of
  // forcing the source to materialize — keeping it in the same eval batch.
  // Correctness: deferring must match materializing the source first.
  auto lazy_prod = [] {
    auto A = array::from({1, 2, 3, 4, 5, 6}, {2, 3});
    auto B = array::from({1, 0, 2, 0, 1, 0, 3, 1, 0, 2, 1, 1}, {3, 4});
    return A.dot(B);  // [2,4], lazy
  };
  auto eager_prod = [&] {
    auto p = lazy_prod();
    p.eval();
    return p;
  };

  CHECK(allclose(lazy_prod().transpose(), eager_prod().transpose()));
  CHECK(allclose(lazy_prod().reshape({4, 2}), eager_prod().reshape({4, 2})));
  CHECK(allclose(lazy_prod().slice(1, 1), eager_prod().slice(1, 1)));
  // view of view: transpose (non-contiguous) then reshape must clone
  CHECK(allclose(lazy_prod().transpose().reshape({8}),
                 eager_prod().transpose().reshape({8})));
  CHECK(allclose(lazy_prod().transpose().transpose(),
                 eager_prod().transpose().transpose()));
  // a lazy transposed view feeding a matmul (gemm reads composed strides)
  auto L = array::full({5, 4}, 0.5f);
  CHECK(allclose(L.dot(lazy_prod().transpose()),
                 L.dot(eager_prod().transpose())));

  // Batch-collapse: transposing a lazy product used to force its own eval
  // (a boundary); now build triggers no evaluation and the whole chain
  // evaluates as one batch (each detail::run_ bumps visit_counter).
  uint64_t before = tl::detail::visit_counter;
  auto K = array::full({8, 6}, 0.3f).dot(array::full({6, 4}, 0.1f));  // lazy
  auto scores = array::full({5, 4}, 0.2f).dot(K.transpose());
  CHECK(tl::detail::visit_counter == before);  // build did not evaluate
  scores.eval();
  CHECK(tl::detail::visit_counter == before + 1);  // exactly one batch
}

TEST_CASE("dot") {
  auto a = array::from({1, 2, 3, 4, 5, 6}, {2, 3});
  auto b = array::from({7, 8, 9, 10, 11, 12}, {3, 2});
  auto c = a.dot(b);
  CHECK(c.shape() == tl::shape_t{2, 2});
  CHECK(c.at({0, 0}) == 58.0f);
  CHECK(c.at({0, 1}) == 64.0f);
  CHECK(c.at({1, 0}) == 139.0f);
  CHECK(c.at({1, 1}) == 154.0f);

  // transposed operand (strided view, no materialization)
  auto d = a.dot(a.transpose());
  CHECK(d.at({0, 0}) == 14.0f);
  CHECK(d.at({1, 1}) == 77.0f);

  // vector promotions
  auto x = array::from({1, 1, 1});
  CHECK(a.dot(x).shape() == tl::shape_t{2});
  CHECK(a.dot(x).at({1}) == 15.0f);
  CHECK(x.dot(x).item() == 3.0f);

  CHECK_THROWS(a.dot(a));
}

TEST_CASE("reductions") {
  auto a = array::from({1, 2, 3, 4, 5, 6}, {2, 3});
  CHECK(a.sum() == 21.0f);
  CHECK(a.mean() == doctest::Approx(3.5f));
  CHECK(a.max() == 6.0f);
  CHECK(a.argmax() == 5);

  auto s0 = a.sum(0);
  CHECK(s0.shape() == tl::shape_t{3});
  CHECK(s0.at({0}) == 5.0f);
  CHECK(s0.at({2}) == 9.0f);

  auto s1 = a.sum(1, true);
  CHECK(s1.shape() == tl::shape_t{2, 1});
  CHECK(s1.at({0, 0}) == 6.0f);
  CHECK(s1.at({1, 0}) == 15.0f);

  auto m1 = a.mean(1);
  CHECK(m1.at({0}) == doctest::Approx(2.0f));

  auto mx = a.max(0);
  CHECK(mx.at({1}) == 5.0f);

  auto am = array::from({3, 9, 1, 8, 2, 7}, {2, 3}).argmax(1);
  CHECK(am.at({0}) == 1.0f);
  CHECK(am.at({1}) == 0.0f);

  // negative axis
  CHECK(a.sum(-1).at({0}) == 6.0f);

  // last-axis max/mean go through the local-accumulator path (inner == 1)
  CHECK(a.max(1).at({0}) == 3.0f);   // max(1,2,3)
  CHECK(a.max(1).at({1}) == 6.0f);   // max(4,5,6)
  CHECK(a.mean(1).at({1}) == doctest::Approx(5.0f));  // mean(4,5,6)

  // reduction over a strided view
  CHECK(a.transpose().sum(0).at({1}) == 15.0f);

  // rank-3 contiguous reduce over each axis: exercises the outer x axis x
  // inner split of the fast path (inner = 4, 2, 1 respectively).
  auto b = array::from({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
  auto b0 = b.sum(0);  // inner=4
  CHECK(b0.shape() == tl::shape_t{2, 2});
  CHECK(b0.at({0, 0}) == 6.0f);   // 1+5
  CHECK(b0.at({1, 1}) == 12.0f);  // 4+8
  auto b1 = b.sum(1);  // inner=2
  CHECK(b1.at({0, 0}) == 4.0f);   // 1+3
  CHECK(b1.at({1, 1}) == 14.0f);  // 6+8
  auto b2 = b.sum(2);  // inner=1 (contiguous running sum)
  CHECK(b2.at({0, 0}) == 3.0f);   // 1+2
  CHECK(b2.at({1, 1}) == 15.0f);  // 7+8
}

TEST_CASE("activations") {
  auto a = array::from({-1, 0, 1});
  auto r = a.relu();
  CHECK(r.at({0}) == 0.0f);
  CHECK(r.at({2}) == 1.0f);

  auto s = array::zeros({3}).sigmoid();
  CHECK(s.at({1}) == doctest::Approx(0.5f));

  auto sm = array::from({1, 2, 3, 1, 2, 3}, {2, 3}).softmax();
  CHECK(sm.sum(1).at({0}) == doctest::Approx(1.0f));
  CHECK(sm.at({0, 2}) == doctest::Approx(0.66524096f));
  // numerical stability: huge logits must not overflow
  auto big = array::from({1000, 1001}).softmax();
  CHECK(big.at({1}) == doctest::Approx(0.7310586f));

  CHECK(tl::allclose(a.exp().log(), a));
}

TEST_CASE("concat") {
  auto a = array::from({1, 2, 3, 4}, {2, 2});
  auto b = array::from({5, 6}, {1, 2});
  auto c = tl::concat({a, b});
  CHECK(c.shape() == tl::shape_t{3, 2});
  CHECK(c.at({2, 1}) == 6.0f);

  // concat of a strided view
  auto d = tl::concat({a.transpose(), b});
  CHECK(d.at({0, 1}) == 3.0f);

  CHECK_THROWS(tl::concat({a, array::zeros({1, 3})}));
}

TEST_CASE("edge shapes") {
  // Lesson from silarray: aligned-only tests pass while edges rot.
  auto one = array::ones({1, 1});
  CHECK(one.dot(one).item() == 1.0f);

  auto row = array::ones({1, 5});
  auto col = array::ones({5, 1});
  CHECK(row.dot(col).item() == 5.0f);
  CHECK(col.dot(row).shape() == tl::shape_t{5, 5});

  auto empty = array::zeros({0, 3});
  CHECK(empty.size() == 0);
  CHECK(empty.sum() == 0.0f);
  CHECK((empty + empty).size() == 0);
}

TEST_CASE("accelerated backend matches the ref oracle") {
  // silarray lesson: edge shapes (1×1, single row/col, odd sizes) are where
  // backend paths rot. Every dispatchable op class, aligned and edge.
  auto a = random_array({33, 17}, 1);
  auto b = random_array({17, 9}, 2);
  auto c = random_array({33, 17}, 3);

  CHECK(matches_oracle([&] { return a.dot(b); }));
  CHECK(matches_oracle([&] { return a.dot(b) * 0.5f + 1.0f; }));  // alpha path
  CHECK(matches_oracle([&] { return a.transpose().dot(c); }));    // CblasTrans
  CHECK(matches_oracle([&] { return a.dot(c.transpose()); }));
  CHECK(matches_oracle([&] { return a.transpose().dot(c).transpose().dot(b); }));
  CHECK(matches_oracle([&] { return random_array({1, 1}, 4).dot(random_array({1, 1}, 5)); }));
  CHECK(matches_oracle([&] { return random_array({1, 7}, 6).dot(random_array({7, 1}, 7)); }));
  CHECK(matches_oracle([&] { return random_array({64}, 8).dot(random_array({64, 3}, 9)); }));

  CHECK(matches_oracle([&] { return a + c; }));
  CHECK(matches_oracle([&] { return a - c; }));
  CHECK(matches_oracle([&] { return a * c; }));
  CHECK(matches_oracle([&] { return a / (c + 2.0f); }));
  CHECK(matches_oracle([&] { return a * 3.0f - 2.0f; }));  // fused affine
  CHECK(matches_oracle([&] { return a.exp(); }));
  CHECK(matches_oracle([&] { return (a + 2.0f).log(); }));
  CHECK(matches_oracle([&] { return (a + 2.0f).sqrt(); }));
  CHECK(matches_oracle([&] { return a.relu(); }));
  CHECK(matches_oracle([&] { return (a + c) * 2.0f; }));  // binary + epilogue

  // training-step shape: W - x^T@g*lr, the whole chain
  auto x = random_array({8, 33}, 10);
  auto g = random_array({8, 9}, 11);
  auto w = random_array({33, 9}, 12);
  CHECK(matches_oracle([&] { return w - x.transpose().dot(g) * 0.01f; }));
}

TEST_CASE("own CPU GEMM matches the ref oracle") {
  // cpu::sgemm — BLIS blocking + packing + microkernel. Edge tiles (MR=8,
  // NR=8 boundaries), single row/col, odd sizes, transposed operands
  // (stride-aware packing, no materialization), and the fused epilogue.
  struct { int64_t m, k, n; } shapes[] = {
      {8, 8, 8}, {1, 1, 1}, {7, 3, 5}, {16, 16, 16}, {33, 17, 31},
      {100, 784, 50}, {1, 128, 64}, {64, 128, 1}, {129, 7, 130},
      {256, 256, 256}, {9, 9, 9},
  };
  int seed = 40;
  for (auto s : shapes) {
    auto a = random_array({s.m, s.k}, seed++);
    auto b = random_array({s.k, s.n}, seed++);
    CHECK(cpu_matches_ref([&] { return a.dot(b); }));
    CHECK(cpu_matches_ref([&] { return a.dot(b) * 0.5f + 1.0f; }));
  }
  auto p = random_array({40, 24}, 70);
  auto q = random_array({40, 12}, 71);
  auto r = random_array({24, 12}, 72);
  CHECK(cpu_matches_ref([&] { return p.transpose().dot(q); }));
  CHECK(cpu_matches_ref([&] { return p.dot(r.transpose().transpose()); }));
  CHECK(cpu_matches_ref([&] { return q.dot(r.transpose()); }));
  // vector promotions and the training-step chain
  CHECK(cpu_matches_ref([&] { return random_array({64}, 73).dot(random_array({64, 5}, 74)); }));
  auto x = random_array({8, 33}, 75);
  auto g = random_array({8, 9}, 76);
  auto w = random_array({33, 9}, 77);
  CHECK(cpu_matches_ref([&] { return w - x.transpose().dot(g) * 0.01f; }));
}

TEST_CASE("metal backend matches the ref oracle") {
  auto a = random_array({33, 17}, 21);
  auto b = random_array({33, 17}, 22);

  CHECK(matches_gpu_oracle([&] { return a + b; }));
  CHECK(matches_gpu_oracle([&] { return a - b; }));
  CHECK(matches_gpu_oracle([&] { return a * b; }));
  CHECK(matches_gpu_oracle([&] { return a / (b + 2.0f); }));
  CHECK(matches_gpu_oracle([&] { return a.exp(); }));
  CHECK(matches_gpu_oracle([&] { return (a + 2.0f).log(); }));
  CHECK(matches_gpu_oracle([&] { return (a + 2.0f).sqrt(); }));
  CHECK(matches_gpu_oracle([&] { return a.sigmoid(); }));
  CHECK(matches_gpu_oracle([&] { return a.relu(); }));
  CHECK(matches_gpu_oracle([&] { return (a * 3.0f) - 1.0f; }));  // fused affine
  CHECK(matches_gpu_oracle([&] { return (a + b) * 2.0f - 1.0f; }));  // epilogue in kernel

  // deep elementwise chain: one command buffer, many dispatches, one flush
  CHECK(matches_gpu_oracle([&] {
    auto x = a;
    for (int i = 0; i < 8; i++) x = (x * 1.01f + b * 0.1f).relu();
    return x;
  }));

  // mixed CPU/GPU graphs: dot runs on CPU (no Metal gemm yet) — exercises
  // the cpu_barrier handoff in both directions
  auto w = random_array({17, 5}, 23);
  CHECK(matches_gpu_oracle([&] { return (a + b).dot(w); }));          // gpu → cpu
  CHECK(matches_gpu_oracle([&] { return (a.dot(w) + 1.0f).relu(); }));  // cpu → gpu
  CHECK(matches_gpu_oracle([&] { return ((a + b).dot(w) * 0.5f).sigmoid().sum(1); }));

  // batch eval of multiple gpu roots
  if (tl::gpu_available()) {
    tl::use_gpu();
    auto c = a + b;
    auto d = a * b;
    tl::eval(c, d);
    tl::use_cpu();
    CHECK(tl::allclose(c, a.clone() + b.clone(), 1e-5f, 1e-6f));
    CHECK(tl::allclose(d, a.clone() * b.clone(), 1e-5f, 1e-6f));
  }
}

TEST_CASE("metal broadcast / pow / mean kernels match oracle") {
  // The rank-2 broadcast kernel (badd_..bpow_), the flat pow_ kernel, and the
  // mean->row_sum lowering keep bias/gamma/layer-norm chains on the GPU. Odd
  // sizes exercise the 32x8 threadgroup edge tiles.
  auto m = random_array({33, 17}, 31);
  auto row = random_array({1, 17}, 32);   // gamma/beta-style row vector
  auto col = random_array({33, 1}, 33);   // bias/rowmax-style column vector
  auto one = random_array({1, 1}, 34);    // scalar

  CHECK(matches_gpu_oracle([&] { return m + row; }));
  CHECK(matches_gpu_oracle([&] { return m - col; }));
  CHECK(matches_gpu_oracle([&] { return m * row; }));
  CHECK(matches_gpu_oracle([&] { return m / (col * col + 1.0f); }));
  CHECK(matches_gpu_oracle([&] { return col + m; }));   // broadcast on the left
  CHECK(matches_gpu_oracle([&] { return m + one; }));
  CHECK(matches_gpu_oracle([&] { return (m + row) * 2.0f - 1.0f; }));  // epilogue

  // pow: same-shape (flat kernel) and broadcast exponent
  auto e = random_array({33, 17}, 35);
  CHECK(matches_gpu_oracle([&] { return tl::pow(m * m + 1.0f, e); }));
  CHECK(matches_gpu_oracle([&] { return tl::pow(m * m + 1.0f, -0.5f); }));

  // mean over the last axis lowers to row_sum * (1/cols) in the kernel
  CHECK(matches_gpu_oracle([&] { return m.mean(1, true); }));
  CHECK(matches_gpu_oracle([&] { return m.mean(1); }));
  CHECK(matches_gpu_oracle([&] { return m.mean(1, true) * 3.0f + 1.0f; }));

  // layer-norm-shaped chain: the op mix that used to ping-pong CPU<->GPU
  auto gamma = random_array({1, 17}, 36);
  auto beta = random_array({1, 17}, 37);
  CHECK(matches_gpu_oracle([&] {
    auto mu = m.mean(1, true);
    auto d = m - mu;
    auto var = (d * d).mean(1, true);
    return d * tl::pow(var + 1e-5f, -0.5f) * gamma + beta;
  }));
}

TEST_CASE("metal SGEMM / softmax / reductions match oracle") {
  if (!tl::gpu_available()) return;

  // Tile-boundary coverage for the 32x32x16 kernel: aligned and every edge
  // (M/N/K not multiples of 32/16), transposed operands, fused epilogue.
  struct { int64_t m, k, n; } shapes[] = {
      {32, 16, 32}, {64, 64, 64}, {33, 17, 31}, {1, 1, 1},
      {1, 128, 1}, {100, 784, 50}, {50, 50, 10}, {17, 3, 129},
      // STEEL band (NN, n >= 48): aligned, M-edge, N-edge, corner, K-rem
      {128, 64, 128}, {70, 32, 128}, {128, 32, 100}, {70, 30, 100},
      {97, 33, 65}, {16, 16, 48}, {200, 129, 64},
  };
  int seed = 30;
  for (auto s : shapes) {
    auto a = random_array({s.m, s.k}, seed++);
    auto b = random_array({s.k, s.n}, seed++);
    CHECK(matches_gpu_oracle([&] { return a.dot(b); }, 2e-3f, 2e-4f));
    CHECK(matches_gpu_oracle([&] { return a.dot(b) * 0.5f + 1.0f; }, 2e-3f, 2e-4f));
  }

  // transposed operands read in place (trans_a / trans_b loaders)
  auto p = random_array({40, 24}, 60);
  auto q = random_array({40, 12}, 61);
  auto rr = random_array({24, 12}, 62);
  CHECK(matches_gpu_oracle([&] { return p.transpose().dot(q); }, 2e-3f, 2e-4f));
  CHECK(matches_gpu_oracle([&] { return p.dot(rr.transpose().transpose()); }, 2e-3f, 2e-4f));
  CHECK(matches_gpu_oracle([&] { return q.dot(rr.transpose()); }, 2e-3f, 2e-4f));

  // STEEL transposed-operand band (_ta_/_tb_ transposing loaders): aligned,
  // M/N edges, K remainder, both BM bands (m < 97 → 32×64). TT falls back
  // to the simple-tile family — covered as the third form.
  struct { int64_t m, k, n; } tshapes[] = {
      {128, 64, 128}, {70, 30, 100}, {97, 33, 65}, {16, 16, 48}, {33, 128, 64},
  };
  for (auto s : tshapes) {
    auto at = random_array({s.k, s.m}, seed++);  // A stored transposed
    auto b2 = random_array({s.k, s.n}, seed++);
    auto a2 = random_array({s.m, s.k}, seed++);
    auto bt = random_array({s.n, s.k}, seed++);  // B stored transposed
    CHECK(matches_gpu_oracle([&] { return at.transpose().dot(b2); }, 2e-3f, 2e-4f));
    CHECK(matches_gpu_oracle([&] { return a2.dot(bt.transpose()); }, 2e-3f, 2e-4f));
    CHECK(matches_gpu_oracle([&] { return at.transpose().dot(bt.transpose()); },
                             2e-3f, 2e-4f));
  }

  // softmax over the last axis, incl. wide rows (cols > threadgroup width)
  for (int64_t cols : {1, 10, 63, 256, 1000}) {
    auto x = random_array({7, cols}, 70 + static_cast<int>(cols));
    CHECK(matches_gpu_oracle([&] { return x.softmax(); }));
  }

  // last-axis row reductions
  auto m = random_array({13, 47}, 80);
  CHECK(matches_gpu_oracle([&] { return m.sum(1); }));
  CHECK(matches_gpu_oracle([&] { return m.sum(1, true); }));
  CHECK(matches_gpu_oracle([&] { return m.max(1); }));
  CHECK(matches_gpu_oracle([&] { return m.sum(1) * 2.0f; }));  // fused epilogue

  // full MLP forward on GPU end to end (gemm → sigmoid → gemm → softmax)
  auto x = random_array({100, 784}, 90);
  auto w1 = random_array({784, 50}, 91);
  auto w2 = random_array({50, 10}, 92);
  CHECK(matches_gpu_oracle(
      [&] { return x.dot(w1).sigmoid().dot(w2).softmax(); }, 2e-3f, 2e-4f));
}

TEST_CASE("lazy graph: build then eval") {
  auto a = array::from({1, 2, 3, 4}, {2, 2});
  auto b = array::ones({2, 2});

  auto c = a + b;            // lazy node
  CHECK(c.shape() == tl::shape_t{2, 2});  // shape known before eval
  CHECK(c.at({1, 1}) == 5.0f);            // access forces eval

  // batch eval: one topological pass over multiple roots
  auto d = a * b;
  auto e = a - b;
  tl::eval(d, e);
  CHECK(d.at({0, 1}) == 2.0f);
  CHECK(e.at({0, 0}) == 0.0f);

  // shape errors surface at build time, not at eval
  CHECK_THROWS(a + array::zeros({3, 3}));
  CHECK_THROWS(a.dot(array::zeros({3, 3})));
}

TEST_CASE("affine fusion composes scalar chains") {
  auto a = array::from({1, 2, 3, 4}, {2, 2});

  // chain of scalar ops folds into one node: ((a*2)+1)*3 = a*6+3
  auto c = ((a * 2.0f) + 1.0f) * 3.0f;
  CHECK(c.at({0, 0}) == 9.0f);
  CHECK(c.at({1, 1}) == 27.0f);

  // s-a and 1/s forms
  auto d = (10.0f - a) / 2.0f;
  CHECK(d.at({0, 0}) == 4.5f);
  auto r = 12.0f / a;
  CHECK(r.at({0, 1}) == 6.0f);

  // epilogue folds into a binary producer: (a+a)*2-1
  auto e = (a + a) * 2.0f - 1.0f;
  CHECK(e.at({1, 0}) == 11.0f);

  // ... and into dot (the gemm-epilogue shape, fused at graph level)
  auto w = array::ones({2, 2});
  auto f = a.dot(w) * 0.5f;
  CHECK(f.at({0, 0}) == 1.5f);
  CHECK(f.at({1, 1}) == 3.5f);

  // ... and into reductions
  auto g = a.sum(1) * 10.0f;
  CHECK(g.at({0}) == 30.0f);
}

TEST_CASE("fusion must not corrupt shared intermediates") {
  auto a = array::from({1, 2, 3, 4}, {2, 2});

  // t is consumed by two fused epilogues AND read directly: the composed
  // copies must leave t's own node untouched.
  auto t = a + a;
  auto u = t * 2.0f;
  auto v = t * 3.0f;
  CHECK(u.at({0, 0}) == 4.0f);
  CHECK(v.at({0, 0}) == 6.0f);
  CHECK(t.at({0, 0}) == 2.0f);

  // diamond: the shared node evaluates once, both consumers see it
  auto c = a + a;
  auto d = c * c;
  CHECK(d.at({1, 1}) == 64.0f);

  // graph rebuild on materialized results
  auto e = (a + a).eval();
  auto f = e + 1.0f;
  CHECK(f.at({0, 0}) == 3.0f);
}

TEST_CASE("sum_to reduces broadcast dims (VJP of broadcasting)") {
  auto a = array::from({1, 2, 3, 4, 5, 6}, {2, 3});

  auto s = a.sum_to({3});  // leading dim sums away
  CHECK(s.shape() == tl::shape_t{3});
  CHECK(s.at({0}) == 5.0f);
  CHECK(s.at({2}) == 9.0f);

  auto k = a.sum_to({1, 3});  // size-1 dim kept
  CHECK(k.shape() == tl::shape_t{1, 3});
  CHECK(k.at({0, 1}) == 7.0f);

  auto r = a.sum_to({2, 1});  // row sums
  CHECK(r.at({0, 0}) == 6.0f);
  CHECK(r.at({1, 0}) == 15.0f);

  CHECK(tl::allclose(a.sum_to({2, 3}), a));  // identity
  CHECK(a.sum_to({}).item() == 21.0f);       // total

  // lazy chain and epilogue fusion compose with it
  auto t = (a * 2.0f).sum_to({3}) * 0.5f;
  CHECK(t.at({0}) == 5.0f);

  // the grad-accumulation pattern: g broadcast up, then reduced back
  auto row = array::from({10, 20, 30});
  auto g = (a + row).sum_to({3});
  CHECK(g.shape() == tl::shape_t{3});

  CHECK_THROWS(a.sum_to({4}));
  CHECK_THROWS(a.sum_to({3, 2}));
}

TEST_CASE("comparisons and where") {
  auto x = array::from({-2, -1, 0, 1, 2});

  auto m = x > 0.0f;
  CHECK(m.at({0}) == 0.0f);
  CHECK(m.at({3}) == 1.0f);
  CHECK((x <= 0.0f).at({2}) == 1.0f);
  CHECK((x == -1.0f).at({1}) == 1.0f);
  CHECK((x != 0.0f).at({2}) == 0.0f);

  // relu backward: g * (x > 0)
  auto g = array::ones({5});
  auto gx = g * (x > 0.0f);
  CHECK(gx.at({1}) == 0.0f);
  CHECK(gx.at({4}) == 1.0f);
  CHECK(tl::allclose(gx, x.relu() > 0.0f));

  // array vs array with broadcasting
  auto a = array::from({1, 2, 3, 4}, {2, 2});
  auto row = array::from({2, 3});
  auto ge = a >= row;
  CHECK(ge.at({0, 0}) == 0.0f);
  CHECK(ge.at({1, 0}) == 1.0f);

  // where with broadcast condition
  auto w = tl::where(a > 2.0f, a, array::zeros({2, 2}));
  CHECK(w.at({0, 0}) == 0.0f);
  CHECK(w.at({1, 1}) == 4.0f);

  // clamp pattern: where(x > hi, hi, x)
  auto c = tl::where(x > 1.0f, array::full({}, 1.0f), x);
  CHECK(c.at({4}) == 1.0f);
  CHECK(c.at({0}) == -2.0f);
}

TEST_CASE("in-place add_ accumulates gradients") {
  auto a = array::from({1, 2, 3, 4}, {2, 2});

  auto g = array::zeros({2, 2});
  g.add_(a);
  g.add_(a);
  CHECK(g.at({0, 0}) == 2.0f);
  CHECK(g.at({1, 1}) == 8.0f);

  // broadcasting contribution
  g.add_(array::from({10, 20}));
  CHECK(g.at({0, 0}) == 12.0f);
  CHECK(g.at({1, 1}) == 28.0f);

  // lazy rhs is evaluated; lazy lhs materializes first
  auto h = (a * 0.0f);
  h.add_(a + a);
  CHECK(h.at({0, 1}) == 4.0f);

  // accumulating through a view mutates the base storage
  auto base = array::zeros({3, 2});
  base.slice(1, 1).add_(array::ones({1, 2}));
  CHECK(base.at({0, 0}) == 0.0f);
  CHECK(base.at({1, 0}) == 1.0f);
  CHECK(base.at({2, 1}) == 0.0f);

  CHECK_THROWS(g.add_(array::zeros({3, 3})));
}

TEST_CASE("lazy chains through views and activations") {
  auto a = array::from({1, 2, 3, 4, 5, 6}, {2, 3});

  // view of a lazy result forces eval of the base, then stays zero-copy
  auto t = (a * 2.0f).transpose();
  CHECK(t.shape() == tl::shape_t{3, 2});
  CHECK(t.at({2, 0}) == 6.0f);

  // activation chain stays lazy end to end
  auto s = (a - 3.0f).relu().sum(1);
  CHECK(s.at({0}) == 0.0f);
  CHECK(s.at({1}) == 6.0f);  // (1+2+3)

  // softmax of a fused affine
  auto sm = (a * 0.0f).softmax();
  CHECK(sm.at({0, 0}) == doctest::Approx(1.0f / 3.0f));
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

  // shape validation: q must be [H,D] rank-2, K/V rank-3 and equal
  CHECK_THROWS(tl::array::attn_decode(K, K, V, scale));       // q rank 3
  CHECK_THROWS(tl::array::attn_decode(q, K, q.reshape({H, D}), scale));  // V rank 2
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

  // direct element access on q4 throws (weight-container semantics)
  CHECK_THROWS(wq.at({0, 0}));

  // non-decode shape (M=2) widens q4 -> f32; matches the same dequant oracle
  auto a2v = std::vector<float>(2 * K);
  for (auto& v : a2v) v = rnd();
  auto a2 = tl::array::from(a2v, {2, K});
  auto ref2 = a2.dot(deq);
  auto got2 = a2.dot(wq);
  CHECK(got2.at({1, 3}) == doctest::Approx(ref2.at({1, 3})).epsilon(1e-3));
}
