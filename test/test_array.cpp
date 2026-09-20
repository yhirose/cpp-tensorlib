#include "doctest.h"

#include <kv_cache.h>
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

TEST_CASE("broadcast_to widens with stride 0 and stays a view") {
  auto a = array::from({1, 2, 3}, {1, 3});

  auto b = a.broadcast_to({4, 3});
  CHECK(b.shape() == tl::shape_t{4, 3});
  CHECK(!b.contiguous());
  CHECK(b.at({0, 0}) == 1.0f);
  CHECK(b.at({3, 2}) == 3.0f);
  // a view: the widened axis reads the source's single row
  a.data()[1] = 20.0f;
  CHECK(b.at({2, 1}) == 20.0f);

  // rank expansion: missing leading axes widen too
  auto row = array::from({5, 6}, {2});
  auto r3 = row.broadcast_to({2, 2, 2});
  CHECK(r3.shape() == tl::shape_t{2, 2, 2});
  CHECK(r3.at({1, 1, 0}) == 5.0f);

  // same shape is a no-op, and an incompatible one throws
  CHECK(a.broadcast_to({1, 3}).shape() == tl::shape_t{1, 3});
  CHECK_THROWS_AS(a.broadcast_to({4, 4}), std::invalid_argument);

  // sum_to is its dual: widen then reduce returns the scaled source
  auto back = a.broadcast_to({4, 3}).sum_to({1, 3});
  CHECK(back.at({0, 0}) == 4.0f);
  CHECK(back.at({0, 1}) == 80.0f);
}

TEST_CASE("broadcast_to feeds ops and clones like a materialized copy") {
  auto src = random_array({1, 64}, 7);
  auto other = random_array({32, 64}, 8);

  // an op against the view matches the same op against a real copy
  auto widened = src.broadcast_to({32, 64});
  auto materialized = widened.clone();
  CHECK(materialized.contiguous());
  CHECK(allclose(widened, materialized));
  CHECK(allclose((widened + other).eval(), (materialized + other).eval()));
  CHECK(allclose((widened * other).eval(), (materialized * other).eval()));

  // and through the ref oracle, so the accelerated paths are checked too
  CHECK(matches_oracle([&] { return src.broadcast_to({32, 64}) + other; }));
  CHECK(matches_oracle([&] { return src.broadcast_to({32, 64}) * other; }));
}

TEST_CASE("clone of a strided view gathers the same elements on the device") {
  // clone()'s device arm (gpu::copy_nd) is a gather over the view's strides,
  // the host arm a strided walk. A transpose, a rank-3 permutation and a
  // stride-0 widening cover the layouts a training step clones.
  auto base = random_array({3, 5, 7}, 1500);
  auto row = random_array({1, 6}, 1501);
  CHECK(matches_gpu_oracle([&] { return base.transpose().clone(); }));
  CHECK(matches_gpu_oracle([&] { return base.transpose({1, 2, 0}).clone(); }));
  CHECK(matches_gpu_oracle([&] { return row.broadcast_to({4, 6}).clone(); }));
}

TEST_CASE("an elementwise op on a widened view matches the oracle") {
  auto src = random_array({1, 64}, 21);
  // 2.0f / x builds recip's node, so it needs a source away from zero
  auto pos_src = random_array({1, 64}, 22).exp().eval();

  // softmax's normalizer sums along the axis, so the repeats belong in the sum
  CHECK(matches_oracle([&] { return src.broadcast_to({32, 64}).softmax(); }));
  CHECK(matches_oracle([&] { return src.broadcast_to({32, 64}) * 2.0f + 1.0f; }));
  CHECK(matches_oracle([&] { return tl::pow(src.broadcast_to({32, 64}), 2.0f); }));
  CHECK(matches_oracle([&] { return src.broadcast_to({32, 64}) > 0.0f; }));
  CHECK(matches_oracle([&] { return src.broadcast_to({32, 64}).exp(); }));
  CHECK(matches_oracle([&] { return src.broadcast_to({32, 64}).clamp(-0.5f, 0.5f); }));
  CHECK(matches_oracle([&] { return 2.0f / pos_src.broadcast_to({32, 64}); }));
}

TEST_CASE("axis reductions off the last axis match the oracle") {
  auto deep = random_array({96, 40}, 11);   // 96 rows: the blocked kernel
  auto shallow = random_array({7, 40}, 12);  // 7 rows: the flat one
  auto r3 = random_array({12, 5, 9}, 13);

  CHECK(matches_oracle([&] { return deep.sum(0); }));
  CHECK(matches_oracle([&] { return deep.sum(0, true); }));
  CHECK(matches_oracle([&] { return deep.mean(0); }));
  CHECK(matches_oracle([&] { return deep.mean(0, true); }));
  CHECK(matches_oracle([&] { return shallow.sum(0); }));
  CHECK(matches_oracle([&] { return shallow.mean(0); }));
  CHECK(matches_oracle([&] { return r3.sum(0); }));
  CHECK(matches_oracle([&] { return r3.sum(1); }));
  CHECK(matches_oracle([&] { return r3.mean(1, true); }));

  // A fused affine on the reduction: sum_ax leaves the epilogue to the shared
  // pass while mean_ax folds 1/dim into one of its own, so both spellings have
  // to come out the same as the oracle's.
  CHECK(matches_oracle([&] { return deep.sum(0) * 2.0f + 1.0f; }));
  CHECK(matches_oracle([&] { return deep.mean(0) * 3.0f - 0.5f; }));

  // The last axis keeps its row kernel, and max still has no dual to take.
  CHECK(matches_oracle([&] { return deep.sum(1); }));
  CHECK(matches_oracle([&] { return deep.max(0); }));
  CHECK(matches_oracle([&] { return deep.mean(1); }));
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

TEST_CASE("batched dot: rank-3 matches per-slice 2-D dot") {
  auto a = random_array({2, 4, 3}, 1);
  auto b = random_array({2, 3, 5}, 2);
  auto c = a.dot(b);
  CHECK(c.shape() == tl::shape_t{2, 4, 5});
  for (int64_t bi = 0; bi < 2; bi++) {
    auto a2 = a.slice(0, bi, 1).reshape({4, 3});
    auto b2 = b.slice(0, bi, 1).reshape({3, 5});
    auto expected = a2.dot(b2);
    for (int64_t i = 0; i < 4; i++) {
      for (int64_t j = 0; j < 5; j++) {
        CHECK(c.at({bi, i, j}) == doctest::Approx(expected.at({i, j})));
      }
    }
  }
}

TEST_CASE("batched dot: rank-4 (two batch axes) matches per-slice 2-D dot") {
  auto a = random_array({2, 3, 4, 5}, 3);
  auto b = random_array({2, 3, 5, 6}, 4);
  auto c = a.dot(b);
  CHECK(c.shape() == tl::shape_t{2, 3, 4, 6});
  for (int64_t p = 0; p < 2; p++) {
    for (int64_t q = 0; q < 3; q++) {
      auto a2 = a.slice(0, p, 1).slice(1, q, 1).reshape({4, 5});
      auto b2 = b.slice(0, p, 1).slice(1, q, 1).reshape({5, 6});
      auto expected = a2.dot(b2);
      for (int64_t i = 0; i < 4; i++) {
        for (int64_t j = 0; j < 6; j++) {
          CHECK(c.at({p, q, i, j}) == doctest::Approx(expected.at({i, j})));
        }
      }
    }
  }
}

TEST_CASE("batched dot: mismatched batch dims or rank throws") {
  auto a = random_array({2, 4, 3}, 5);
  auto b = random_array({3, 3, 5}, 6);   // batch dim 2 vs 3
  CHECK_THROWS(a.dot(b));
  auto c = random_array({2, 3, 4, 3}, 7);  // rank 4 vs rank 3
  CHECK_THROWS(a.dot(c));
}

TEST_CASE("batched dot: GPU dispatch matches the ref oracle") {
  CHECK(matches_gpu_oracle([&] {
    auto a = random_array({4, 6, 8}, 11);
    auto b = random_array({4, 8, 5}, 12);
    return a.dot(b);
  }));
  CHECK(matches_gpu_oracle([&] {
    auto a = random_array({2, 3, 6, 8}, 13);
    auto b = random_array({2, 3, 8, 5}, 14);
    return a.dot(b);
  }));
  // Permuted (transposed-slice) operands — attention's q·kᵀ — take the
  // kernel's transposed operand instead of declining to the scalar oracle.
  CHECK(matches_gpu_oracle([&] {
    auto q = random_array({4, 6, 8}, 15);
    auto k = random_array({4, 6, 8}, 16);
    return q.dot(k.transpose({0, 2, 1}));
  }));
  CHECK(matches_gpu_oracle([&] {
    auto a = random_array({2, 3, 8, 6}, 17);
    auto b = random_array({2, 3, 8, 5}, 18);
    return a.transpose({0, 1, 3, 2}).dot(b);
  }));
  // A batch slice that starts past the buffer's head: one head of attention.
  CHECK(matches_gpu_oracle([&] {
    auto q = random_array({4, 6, 8}, 19);
    auto k = random_array({4, 6, 8}, 20);
    return q.slice(0, 1, 2).dot(k.slice(0, 1, 2).transpose({0, 2, 1}));
  }));
  // The same past a permuted batch, which the one-launch gemm hands back to
  // the per-slice loop.
  CHECK(matches_gpu_oracle([&] {
    auto a = random_array({2, 3, 6, 8}, 26);
    auto b = random_array({2, 3, 8, 5}, 27);
    return a.transpose({1, 0, 2, 3}).slice(0, 1, 2).dot(
        b.transpose({1, 0, 2, 3}).slice(0, 1, 2));
  }));
}

TEST_CASE("batched dot: one-launch GPU gemm matches the ref oracle") {
  // Shapes the register-blocked kernel takes whole, with the batch folded into
  // its grid: a whole tile per slice; m and n off the tile; an underfilled
  // grid with a long K (split-K, so z = batch × S); the attention probs·v
  // shape (NN) and scores shape (NT); and rank 4 whose two batch axes
  // collapse to one stride. Then the layouts the one-launch path must hand
  // back to the per-slice loop: n not a multiple of 4 on NN, K odd, a
  // permuted batch axis, a size-1 batch axis in the middle (which still
  // collapses), and the fused epilogue (which split-K folds into its partials).
  struct { int64_t batch, m, k, n; } shapes[] = {
      {4, 128, 64, 128}, {3, 129, 64, 130}, {4, 200, 1024, 136},
      {8, 256, 256, 64}, {2, 96, 8, 32},
  };
  int seed = 700;
  for (auto s : shapes) {
    auto A = random_array({s.batch, s.m, s.k}, seed++);
    auto B = random_array({s.batch, s.k, s.n}, seed++);
    auto Bt = random_array({s.batch, s.n, s.k}, seed++);
    auto At = random_array({s.batch, s.k, s.m}, seed++);
    const float rtol = 1e-4f, atol = std::max(1e-5f, 2e-7f * (float)s.k);
    CHECK(matches_gpu_oracle([&] { return A.dot(B); }, rtol, atol));
    CHECK(matches_gpu_oracle([&] { return A.dot(Bt.transpose({0, 2, 1})); }, rtol, atol));
    CHECK(matches_gpu_oracle([&] { return At.transpose({0, 2, 1}).dot(B); }, rtol, atol));
    CHECK(matches_gpu_oracle([&] { return A.dot(B) * 0.5f + 1.0f; }, rtol, atol));
  }
  auto A4 = random_array({2, 3, 64, 32}, seed++);
  auto B4 = random_array({2, 3, 32, 64}, seed++);
  CHECK(matches_gpu_oracle([&] { return A4.dot(B4); }));
  CHECK(matches_gpu_oracle([&] { return A4.transpose({1, 0, 2, 3}).dot(B4.transpose({1, 0, 2, 3})); }));
  CHECK(matches_gpu_oracle([&] {
    return A4.reshape({2, 1, 3, 64, 32}).dot(B4.reshape({2, 1, 3, 32, 64}));
  }));
  CHECK(matches_gpu_oracle([&] {
    return random_array({4, 64, 8}, 800).dot(random_array({4, 8, 33}, 801));
  }));
  CHECK(matches_gpu_oracle([&] {
    return random_array({4, 33, 17}, 802).dot(random_array({4, 17, 31}, 803));
  }));
}

TEST_CASE("batched dot: own CPU gemm per slice matches the ref oracle") {
  // The same shapes through cpu_bdot_ (cpu::sgemm per slice, strides passed
  // through) against ref::bdot, plain and with permuted operands.
  auto a = random_array({4, 6, 8}, 21);
  auto b = random_array({4, 8, 5}, 22);
  auto k = random_array({4, 6, 8}, 23);
  auto a4 = random_array({2, 3, 8, 6}, 24);
  auto b4 = random_array({2, 3, 8, 5}, 25);
  CHECK(cpu_matches_ref([&] { return a.dot(b); }));
  CHECK(cpu_matches_ref([&] { return a.dot(k.transpose({0, 2, 1})); }));
  CHECK(cpu_matches_ref([&] { return a4.transpose({0, 1, 3, 2}).dot(b4); }));
  CHECK(cpu_matches_ref([&] { return a.dot(b) * 0.5f + 1.0f; }));  // epilogue
  // A batch slice that starts past the buffer's head: one head of attention.
  CHECK(cpu_matches_ref([&] { return a.slice(0, 1, 2).dot(b.slice(0, 1, 2)); }));
  CHECK(cpu_matches_ref([&] {
    return a.slice(0, 3, 1).dot(k.slice(0, 3, 1).transpose({0, 2, 1}));
  }));
}

TEST_CASE("elementwise: own CPU runs across the pool match the walker oracle") {
  // Every layout ew_plan_for collapses to runs — same shape, a scalar, a row
  // and a column vector, a leading broadcast axis (the attention mask), a
  // size-1 axis in the middle, a slice with a gap between rows, rank 0, and a
  // transposed view (a gather along the run) on either side, both, or next to
  // a broadcast row; a shape big enough to split across threads; unary through
  // the same driver.
  auto a = random_array({6, 7, 40}, 921), b = random_array({6, 7, 40}, 922);
  auto lead = random_array({1, 7, 40}, 923), mid = random_array({6, 1, 40}, 924);
  auto m = random_array({30, 33}, 925), rowv = random_array({1, 33}, 926);
  auto colv = random_array({30, 1}, 927), t = random_array({33, 30}, 928);
  CHECK(cpu_matches_ref([&] { return a + b; }));
  CHECK(cpu_matches_ref([&] { return a * 0.5f; }));
  CHECK(cpu_matches_ref([&] { return a + lead; }));
  CHECK(cpu_matches_ref([&] { return mid * a; }));
  CHECK(cpu_matches_ref([&] { return m + rowv; }));
  CHECK(cpu_matches_ref([&] { return colv - m; }));
  CHECK(cpu_matches_ref([&] { return m.slice(1, 3, 20) + rowv.slice(1, 3, 20); }));
  CHECK(cpu_matches_ref([&] { return m + t.transpose(); }));
  CHECK(cpu_matches_ref([&] { return t.transpose() - m; }));
  CHECK(cpu_matches_ref([&] { return t.transpose() * t.transpose(); }));
  CHECK(cpu_matches_ref([&] { return t.transpose() + rowv; }));
  CHECK(cpu_matches_ref([&] {
    return random_array({64, 512}, 933).transpose() + random_array({512, 64}, 934);
  }));
  CHECK(cpu_matches_ref([&] { return random_array({}, 929) + random_array({}, 930); }));
  CHECK(cpu_matches_ref([&] { return random_array({8, 256, 256}, 931) + random_array({1, 256, 256}, 932); }));
  CHECK(cpu_matches_ref([&] { return a.exp(); }));
  CHECK(cpu_matches_ref([&] { return t.transpose().exp(); }));
  CHECK(cpu_matches_ref([&] { return m.slice(1, 3, 20).exp(); }));
  // where through the same driver: a broadcast mask over both branches, a
  // scalar branch, and a transposed branch
  CHECK(cpu_matches_ref([&] { return tl::where(lead > 0.0f, a, b); }));
  CHECK(cpu_matches_ref([&] { return tl::where(a > 0.0f, a, array::full({}, -1.0f)); }));
  CHECK(cpu_matches_ref([&] { return tl::where(m > 0.0f, t.transpose(), m); }));
}

TEST_CASE("axis reductions: own CPU slabs across the pool match the oracle") {
  // The contiguous reduce driver split over the pool: the last axis (rows),
  // a middle axis (outer slabs), axis 0 of a matrix (inner columns, one
  // slab), sum / mean / max, at sizes past the thread cap; a strided view
  // stays on the walker.
  auto a = random_array({8, 256, 256}, 941), m = random_array({512, 512}, 942);
  CHECK(cpu_matches_ref([&] { return a.sum(2); }));
  CHECK(cpu_matches_ref([&] { return a.mean(2, true); }));
  CHECK(cpu_matches_ref([&] { return a.max(2); }));
  CHECK(cpu_matches_ref([&] { return a.sum(1); }));
  CHECK(cpu_matches_ref([&] { return a.max(0); }));
  CHECK(cpu_matches_ref([&] { return m.sum(0); }));
  CHECK(cpu_matches_ref([&] { return m.mean(0, true); }));
  CHECK(cpu_matches_ref([&] { return m.transpose().sum(1); }));
}

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

TEST_CASE("GPU GEMM: transposed operands take the fast path and match the ref oracle") {
  // The CUDA register-blocked kernel takes each operand plain or as the
  // transposed view of a contiguous array (NN/NT/TN/TT), predicated at the M
  // and N block edges and, on an underfilled grid, split over K. One shape per
  // class: a whole tile; m and n off the tile; an underfilled grid with a long
  // K (4 blocks, K=1024: split 8 ways); n not a multiple of 4 (NT still
  // qualifies — B's float4 runs along K); the attention-scores shape; and K
  // odd, which every layout must hand to the plain kernel.
  struct { int64_t m, k, n; } shapes[] = {
      {128, 64, 128}, {129, 64, 130}, {200, 1024, 136},
      {64, 8, 33},    {512, 1024, 512}, {33, 17, 31},
  };
  int seed = 300;
  for (auto s : shapes) {
    auto A = random_array({s.m, s.k}, seed++), At = random_array({s.k, s.m}, seed++);
    auto B = random_array({s.k, s.n}, seed++), Bt = random_array({s.n, s.k}, seed++);
    // float32 accumulation over K terms of magnitude <= 1 drifts by ~K·ε from
    // the oracle's own order (measured 8e-5 at K=1024, the same on NN), so the
    // absolute tolerance scales with K; the relative one stays.
    const float rtol = 1e-4f, atol = std::max(1e-5f, 2e-7f * (float)s.k);
    CHECK(matches_gpu_oracle([&] { return A.dot(B); }, rtol, atol));
    CHECK(matches_gpu_oracle([&] { return A.dot(Bt.transpose()); }, rtol, atol));
    CHECK(matches_gpu_oracle([&] { return At.transpose().dot(B); }, rtol, atol));
    CHECK(matches_gpu_oracle([&] { return At.transpose().dot(Bt.transpose()); }, rtol, atol));
    // the fused affine epilogue: split-K scales every partial and adds the
    // offset from split 0 only, so the split shapes check it too
    CHECK(matches_gpu_oracle([&] { return A.dot(Bt.transpose()) * 0.5f + 1.0f; }, rtol, atol));
  }
}

TEST_CASE("auto mode derives its matmul threshold on the first eval") {
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;
  tl::use_auto();
  // a graph eval (a constant plus a scalar folds eagerly and never reaches run_)
  random_array({8, 8}, 500).dot(random_array({8, 8}, 501)).eval();
  int64_t t = tl::auto_matmul_threshold();
  tl::device_ = prev;
  // one of the census sizes, or the baked value when the GPU won none of them
  CHECK((t == 96 * 96 * 96 || t == 128 * 128 * 128 || t == 192 * 192 * 192 ||
         t == 256 * 256 * 256 ||
         t == tl::auto_threshold_(tl::kernel_class::matmul)));
  CHECK(tl::cpu::min_work_per_thread_() > 0);
}

TEST_CASE("the auto census keeps out of the caller's defer_flush scope") {
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;
  const int64_t saved = tl::detail::graph::auto_matmul_;
  tl::detail::graph::auto_matmul_ = -1;  // census again, in a scope this time
  tl::use_auto();
  {
    // What an autograd walk opens; the census must not borrow it (the
    // CensusScope in calibrate_auto_ says what goes wrong when it does).
    tl::defer_flush defer;
    random_array({8, 8}, 700).dot(random_array({8, 8}, 701)).eval();
    CHECK(!tl::gpu::pending());
  }
  tl::detail::graph::auto_matmul_ = saved;
  tl::device_ = prev;
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

TEST_CASE("dot + row bias fuses into the gemm and matches the unfused sum") {
  // graph::fuse_dot_bias_ makes the bias a third input of the dot; the CUDA
  // gemm adds it in its store (tail and split-K shapes below, both tiles), and
  // every other path after the epilogue. Checked against the GPU/ref oracles
  // and against the unfused sum (an evaluated dot plus the bias).
  struct { int64_t m, k, n; } shapes[] = {
      {3, 4, 5}, {129, 64, 130}, {256, 512, 512}, {512, 1024, 512}, {200, 1024, 136},
  };
  int seed = 1300;
  for (auto s : shapes) {
    auto A = random_array({s.m, s.k}, seed++);
    auto B = random_array({s.k, s.n}, seed++), Bt = random_array({s.n, s.k}, seed++);
    auto bias = random_array({s.n}, seed++), bias_row = random_array({1, s.n}, seed++);
    const float rtol = 1e-4f, atol = std::max(1e-5f, 2e-7f * (float)s.k);
    CHECK(matches_gpu_oracle([&] { return A.dot(B) + bias; }, rtol, atol));
    CHECK(matches_gpu_oracle([&] { return bias_row + A.dot(Bt.transpose()); }, rtol, atol));
    CHECK(matches_gpu_oracle([&] { return A.dot(B) * 0.5f + 1.0f + bias; }, rtol, atol));
    CHECK(matches_gpu_oracle([&] { return (A.dot(B) + bias) * 2.0f - 1.0f; }, rtol, atol));
    CHECK(tl::allclose((A.dot(B) + bias).eval(), A.dot(B).eval() + bias, rtol, atol));
    // a shared dot: the fused copy leaves the plain product to its other reader
    auto y = A.dot(B);
    auto z = y + bias_row;
    CHECK(tl::allclose(z.eval() - bias_row, y.eval(), rtol, atol));
  }
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

TEST_CASE("sum_to: own CPU passes match the walker oracle") {
  // Summed axes leading, trailing, in the middle and on both sides; size-1
  // axes on the input; a full reduce; a transposed and a broadcast view (both
  // copied out first); sizes past the thread cap.
  auto a = random_array({6, 7, 40}, 951), big = random_array({2048, 256}, 952);
  auto one = random_array({6, 1, 40}, 953);
  CHECK(cpu_matches_ref([&] { return a.sum_to({40}); }));
  CHECK(cpu_matches_ref([&] { return a.sum_to({1, 1, 40}); }));
  CHECK(cpu_matches_ref([&] { return a.sum_to({6, 1, 40}); }));
  CHECK(cpu_matches_ref([&] { return a.sum_to({6, 7, 1}); }));
  CHECK(cpu_matches_ref([&] { return a.sum_to({7, 1}); }));
  CHECK(cpu_matches_ref([&] { return a.sum_to({}); }));
  CHECK(cpu_matches_ref([&] { return a.sum_to({6, 7, 40}); }));
  CHECK(cpu_matches_ref([&] { return one.sum_to({40}); }));
  CHECK(cpu_matches_ref([&] { return big.sum_to({256}); }));
  CHECK(cpu_matches_ref([&] { return big.sum_to({2048, 1}); }));
  CHECK(cpu_matches_ref([&] { return big.transpose().sum_to({256, 1}); }));
  CHECK(cpu_matches_ref([&] { return random_array({40}, 954).broadcast_to({6, 7, 40}).sum_to({7, 1}); }));
  // The bias gradient adds each column in row order, as the walker does.
  auto fast = big.sum_to({256}).eval();
  tl::cpu::enabled_ = false;
  auto oracle = big.sum_to({256}).eval();
  tl::cpu::enabled_ = true;
  CHECK(std::memcmp(fast.raw(), oracle.raw(), 256 * sizeof(float)) == 0);
}

TEST_CASE("clone: own CPU copies any layout exactly") {
  // Contiguous, a permuted view (innermost contiguous), a transposed view
  // (innermost strided), a broadcast view (stride 0), and a slice with a gap
  // between rows — each bit for bit what the walker copies, -0.0 included.
  auto x = random_array({8, 16, 4, 32}, 961);
  auto m = random_array({300, 200}, 962);
  std::vector<float> signs = {-0.0f, 1.0f, -2.0f, 0.0f, -0.0f, 3.0f};
  auto z = array::from(signs, {2, 3});
  auto exact = [](auto build) {
    auto fast = build().clone();
    tl::cpu::enabled_ = false;
    auto oracle = build().clone();
    tl::cpu::enabled_ = true;
    return fast.shape() == oracle.shape() &&
           std::memcmp(fast.raw(), oracle.raw(), fast.size() * sizeof(float)) == 0;
  };
  CHECK(exact([&] { return m; }));
  CHECK(exact([&] { return x.transpose({0, 2, 1, 3}); }));
  CHECK(exact([&] { return m.transpose(); }));
  CHECK(exact([&] { return z.transpose(); }));
  CHECK(exact([&] { return random_array({1, 200}, 963).broadcast_to({300, 200}); }));
  CHECK(exact([&] { return m.slice(1, 10, 150); }));
}

TEST_CASE("sum_to GPU dispatch matches the CPU oracle") {
  auto a = array::full({8, 16}, 2.0f);
  auto row = array::from(std::vector<float>(16, 1.0f), {16});

  // The concrete caller: a broadcast-add's backward reducing a [N,D]
  // gradient down to a [D] bias gradient (Linear layer bias, LayerNorm
  // gamma/beta).
  CHECK(matches_gpu_oracle([&] { return a.sum_to({16}); }));
  CHECK(matches_gpu_oracle([&] { return a.sum_to({1, 16}); }));  // size-1 kept
  CHECK(matches_gpu_oracle([&] { return a.sum_to({8, 1}); }));   // row sums
  CHECK(matches_gpu_oracle([&] { return a.sum_to({}); }));       // full reduce
  CHECK(matches_gpu_oracle([&] { return (a + row).sum_to({16}); }));

  // Rank 3 (Transformer-shaped): [N,S,D] gradient summed to a [D] bias grad.
  auto x = array::full({4, 8, 32}, 1.5f);
  CHECK(matches_gpu_oracle([&] { return x.sum_to({32}); }));
  CHECK(matches_gpu_oracle([&] { return x.sum_to({1, 1, 32}); }));
  CHECK(matches_gpu_oracle([&] { return x.sum_to({4, 1, 1}); }));

  // Deep reductions (64 rows or more take a threadgroup per output) on random
  // data: a bias gradient over more rows than the 256 threads, one short of a
  // full stride, and a rank-3 one keeping its middle axis.
  auto deep = random_array({300, 40}, 1240);
  CHECK(matches_gpu_oracle([&] { return deep.sum_to({40}); }));
  CHECK(matches_gpu_oracle([&] { return random_array({64, 9}, 1241).sum_to({9}); }));
  auto deep3 = random_array({70, 6, 24}, 1242);
  CHECK(matches_gpu_oracle([&] { return deep3.sum_to({6, 1}); }));
  CHECK(matches_gpu_oracle([&] { return deep3.sum_to({1, 1, 24}); }));

  // Non-contiguous input (a view) falls back to the CPU oracle honestly
  // (gpu_sum_to_ gates on a.contiguous()) rather than mis-dispatching.
  auto t = array::from({1, 2, 3, 4, 5, 6}, {2, 3}).transpose();  // shape (3,2)
  CHECK(matches_gpu_oracle([&] { return t.sum_to({2}); }));
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

TEST_CASE("comparisons GPU dispatch matches the CPU oracle") {
  auto p = array::from({-3, -1, 0, 2, 4, -5, 6, 1}, {2, 4});
  auto q = array::from({1, -1, 0, 1, 3, -5, 7, 0}, {2, 4});

  // scalar s: the tensor-scalar kernels (gt_s ...), the ReLU/LeakyReLU/Clip
  // backward-gate shape.
  CHECK(matches_gpu_oracle([&] { return p > 0.0f; }));
  // an explicit size-1 b (bstride=0), which stays a binary compare.
  CHECK(matches_gpu_oracle([&] { return p > array::full({}, 0.0f); }));
  CHECK(matches_gpu_oracle([&] { return p <= array::full({1}, 1.0f); }));
  CHECK(matches_gpu_oracle([&] { return p < 0.0f; }));
  CHECK(matches_gpu_oracle([&] { return p >= 1.0f; }));
  CHECK(matches_gpu_oracle([&] { return p <= -1.0f; }));
  CHECK(matches_gpu_oracle([&] { return p == 0.0f; }));
  CHECK(matches_gpu_oracle([&] { return p != 0.0f; }));

  // same-shape b (bstride=1).
  CHECK(matches_gpu_oracle([&] { return p > q; }));
  CHECK(matches_gpu_oracle([&] { return p == q; }));
  CHECK(matches_gpu_oracle([&] { return p != q; }));

  // the actual masked_gate composition: gy * (x > 0).
  auto gy = array::full({2, 4}, 1.5f);
  CHECK(matches_gpu_oracle([&] { return gy * (p > 0.0f); }));
  CHECK(matches_gpu_oracle([&] { return gy * (p.relu() > 0.0f); }));

  // a genuine N-D broadcast (not scalar, not same-shape): declines to the
  // CPU oracle honestly rather than mis-dispatching.
  auto row = array::from({1, -1, 0, 1});
  CHECK(matches_gpu_oracle([&] { return p >= row; }));
}

TEST_CASE("tanh/sin/cos/clamp GPU dispatch matches the CPU oracle") {
  auto x = array::from({-2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 5.0f}, {2, 3});

  CHECK(matches_gpu_oracle([&] { return x.tanh(); }));
  CHECK(matches_gpu_oracle([&] { return x.sin(); }));
  CHECK(matches_gpu_oracle([&] { return x.cos(); }));
  CHECK(matches_gpu_oracle([&] { return x.clamp(-1.0f, 1.0f); }));

  // fused epilogue composes with tanh/sin/cos (same as exp_/sqrt_ do).
  CHECK(matches_gpu_oracle([&] { return x.tanh() * 2.0f - 1.0f; }));

  // clamp composed with a comparison: the actual Clip backward gate
  // (x_data.ge(lo) * x_data.le(hi)).
  CHECK(matches_gpu_oracle([&] {
    return (x >= -1.0f) * (x <= 1.0f);
  }));

  // correctness spot-checks, not just self-consistency with the oracle.
  auto c = x.clamp(-1.0f, 1.0f).eval();
  CHECK(c.at({0, 0}) == -1.0f);  // -2.0 clamped to lo
  CHECK(c.at({0, 1}) == -0.5f);  // within range, unchanged
  CHECK(c.at({1, 2}) == 1.0f);   // 5.0 clamped to hi
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
    // The own CPU takes every contiguous input, and so do the CUDA and Metal
    // kernels; WebGPU declines and its caller composes the form above.
    if (!on_gpu) REQUIRE(got.has_value());
#if !defined(TENSORLIB_WEBGPU)
    REQUIRE(got.has_value());
#endif
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

TEST_CASE("concat GPU dispatch matches the CPU oracle") {
  auto a = array::from({1, 2, 3, 4}, {2, 2});
  auto b = array::from({5, 6, 7, 8}, {2, 2});

  // axis 0 (the old default) still works and dispatches to GPU.
  CHECK(matches_gpu_oracle([&] { return tl::concat({a, b}); }));
  CHECK(matches_gpu_oracle([&] { return tl::concat({a, b}, 0); }));

  // axis 1 -- the new case (KV-cache append is along a non-leading axis).
  CHECK(matches_gpu_oracle([&] { return tl::concat({a, b}, 1); }));

  // three parts, rank 3, a middle axis -- (2,1,2)+(2,3,2)+(2,2,2) on axis 1.
  auto p = array::from({1, 2, 3, 4}, {2, 1, 2});
  auto q = array::ones({2, 3, 2});
  auto s = array::full({2, 2, 2}, 9.0f);
  CHECK(matches_gpu_oracle([&] { return tl::concat({p, q, s}, 1); }));

  // fused epilogue composes with a concat operand (same as pad/fold do).
  CHECK(matches_gpu_oracle([&] { return tl::concat({a, b}, 1) * 2.0f - 1.0f; }));

  // correctness spot-checks, not just self-consistency with the oracle.
  auto c = tl::concat({a, b}, 1).eval();
  CHECK(c.shape() == tl::shape_t{2, 4});
  CHECK(c.at({0, 0}) == 1.0f);
  CHECK(c.at({0, 2}) == 5.0f);
  CHECK(c.at({1, 3}) == 8.0f);

  CHECK_THROWS(tl::concat({a, array::zeros({3, 2})}, 1));  // axis-1 rows must match
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
  // The own CPU always takes it, and so do the CUDA and Metal kernels — while
  // WebGPU declines and its caller composes the unfused form, which the
  // gradient tests above cover.
  if (!on_gpu) REQUIRE(got.has_value());
#if !defined(TENSORLIB_WEBGPU)
  REQUIRE(got.has_value());
#endif
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
  if (!on_gpu) REQUIRE(got.has_value());
#if !defined(TENSORLIB_WEBGPU)
  REQUIRE(got.has_value());  // as above
#endif
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

// `unfold`: the sliding-window view a differentiable im2col (or any other
// windowed op) is built out of. See array.h's doc comment for what's still
// missing (lazy-graph integration, a `fold`/scatter-add backward, GPU).
TEST_CASE("unfold: 1-D sliding window matches a hand-written stride walk") {
  auto a = array::from({0, 1, 2, 3, 4, 5});
  auto w = a.unfold(0, 3, 1);
  CHECK(w.shape() == tl::shape_t{4, 3});
  float expected[4][3] = {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
  for (int64_t i = 0; i < 4; i++) {
    for (int64_t j = 0; j < 3; j++) {
      CHECK(w.at({i, j}) == expected[i][j]);
    }
  }
}

TEST_CASE("unfold: two axes composed gives im2col-style 2x2 non-overlapping blocks") {
  // A 4x4 "image" (values 0..15), unfolded on both spatial axes with a 2x2,
  // stride-2 window — the exact config test_conv.cul hand-verified for the
  // culebra port's nested-loop im2col (block sums [[10,18],[42,50]]).
  std::vector<float> img(16);
  for (int i = 0; i < 16; i++) img[i] = static_cast<float>(i);
  auto x = array::from(img, {4, 4});

  auto windows = x.unfold(0, 2, 2).unfold(1, 2, 2);  // [2,2,2,2]: (oh,ow,kh,kw)
  CHECK(windows.shape() == tl::shape_t{2, 2, 2, 2});

  // Block (0,0) should be the top-left 2x2 patch {0,1,4,5}.
  CHECK(windows.at({0, 0, 0, 0}) == 0);
  CHECK(windows.at({0, 0, 0, 1}) == 1);
  CHECK(windows.at({0, 0, 1, 0}) == 4);
  CHECK(windows.at({0, 0, 1, 1}) == 5);
  // Block (1,1) (bottom-right) should be {10,11,14,15}.
  CHECK(windows.at({1, 1, 0, 0}) == 10);
  CHECK(windows.at({1, 1, 0, 1}) == 11);
  CHECK(windows.at({1, 1, 1, 0}) == 14);
  CHECK(windows.at({1, 1, 1, 1}) == 15);

  // Every element of a window sums to the same per-block totals im2col's
  // matmul-with-a-ones-kernel would produce (10, 18, 42, 50).
  auto block_sum = [&](int64_t bi, int64_t bj) {
    float s = 0;
    for (int64_t ki = 0; ki < 2; ki++)
      for (int64_t kj = 0; kj < 2; kj++) s += windows.at({bi, bj, ki, kj});
    return s;
  };
  CHECK(block_sum(0, 0) == 10);
  CHECK(block_sum(0, 1) == 18);
  CHECK(block_sum(1, 0) == 42);
  CHECK(block_sum(1, 1) == 50);
}

TEST_CASE("unfold: overlapping windows share storage with the source (true view)") {
  auto base = array::zeros({6});
  auto w = base.unfold(0, 3, 1);  // overlapping: step < size
  CHECK(w.shape() == tl::shape_t{4, 3});

  // Mutate the base in place; the view must see it, proving no copy was
  // made at unfold() time (this is the whole performance argument).
  base.add_(array::from({1, 2, 3, 4, 5, 6}));
  CHECK(w.at({0, 0}) == 1);
  CHECK(w.at({0, 2}) == 3);
  CHECK(w.at({1, 0}) == 2);   // overlaps window 0's last element's neighbor
  CHECK(w.at({3, 2}) == 6);
}

TEST_CASE("slice: any axis, both the eager and the lazy-graph path") {
  // 3x4, values row-major 0..11. Column-slice (axis 1) is the case the old
  // axis-0-only `slice` couldn't express at all.
  std::vector<float> v(12);
  for (int i = 0; i < 12; i++) v[i] = static_cast<float>(i);
  auto x = array::from(v, {3, 4});

  auto eager = x.slice(1, 1, 2);  // columns 1..2 of every row
  CHECK(eager.shape() == tl::shape_t{3, 2});
  CHECK(eager.at({0, 0}) == 1);
  CHECK(eager.at({1, 1}) == 6);
  CHECK(eager.at({2, 0}) == 9);

  // Through a still-lazy source, this exercises the view_ node's eval path
  // (view_axes now carries the sliced axis instead of always meaning 0).
  auto lazy_src = x + 0.0f;  // forces a lazy `affine` node, not yet evaluated
  auto lazy = lazy_src.slice(1, 1, 2);
  CHECK(lazy.shape() == tl::shape_t{3, 2});
  CHECK(lazy.at({0, 0}) == 1);
  CHECK(lazy.at({1, 1}) == 6);
  CHECK(lazy.at({2, 0}) == 9);

  CHECK_THROWS(x.slice(1, 3, 2));  // out of range
  CHECK_THROWS(x.slice(5, 0, 1));  // bad axis
}

TEST_CASE("pad: single axis places the source at the right offset in a zero buffer") {
  auto x = array::from({1, 2, 3}, {1, 3});
  auto padded = x.pad(1, 2, 1);  // 2 zeros before, 1 after, on axis 1
  CHECK(padded.shape() == tl::shape_t{1, 6});
  float expected[6] = {0, 0, 1, 2, 3, 0};
  for (int j = 0; j < 6; j++) CHECK(padded.at({0, j}) == expected[j]);

  // asymmetric before/after and a lazy source both work.
  auto lazy_padded = (x + 0.0f).pad(1, 0, 2);
  CHECK(lazy_padded.shape() == tl::shape_t{1, 5});
  float expected2[5] = {1, 2, 3, 0, 0};
  for (int j = 0; j < 5; j++) CHECK(lazy_padded.at({0, j}) == expected2[j]);
}

TEST_CASE("pad + unfold composed on both spatial axes reproduces a padded im2col") {
  // 4x4 image 0..15, pad=1 on H and W, then a 2x2/stride-2 window on each —
  // the full pad -> unfold -> unfold pipeline a differentiable conv2d needs,
  // with no host-side loop anywhere in this test.
  std::vector<float> img(16);
  for (int i = 0; i < 16; i++) img[i] = static_cast<float>(i);
  auto x = array::from(img, {4, 4});

  auto padded = x.pad(0, 1, 1).pad(1, 1, 1);
  CHECK(padded.shape() == tl::shape_t{6, 6});
  CHECK(padded.at({0, 0}) == 0);   // padding
  CHECK(padded.at({1, 1}) == 0);   // original (0,0)
  CHECK(padded.at({2, 2}) == 5);   // original (1,1)

  auto windows = padded.unfold(0, 2, 2).unfold(1, 2, 2);  // [3,3,2,2]
  CHECK(windows.shape() == tl::shape_t{3, 3, 2, 2});
  // Top-left window is entirely padding-and-one-real-pixel: {0,0,0,0}.
  CHECK(windows.at({0, 0, 0, 0}) == 0);
  CHECK(windows.at({0, 0, 1, 1}) == 0);  // padded(1,1) == original(0,0) == 0
  // Center window (1,1) covers original rows/cols 1..2 exactly.
  CHECK(windows.at({1, 1, 0, 0}) == 5);   // original (1,1)
  CHECK(windows.at({1, 1, 1, 1}) == 10);  // original (2,2)
}

TEST_CASE("a view over a lazy source still gets its fused affine epilogue") {
  // Regression test: `graph::affine` fuses `* s + o` onto ANY unevaluated,
  // non-constant node — view_ (transpose/reshape/slice) included — but
  // `eval_one`'s view_ case used to `store()` and `return` immediately,
  // bypassing the epilogue application entirely and silently dropping the
  // scale/offset. Needs a source large enough to skip the eager-tiny fast
  // path (kEagerTiny) so the arithmetic actually goes through the lazy
  // graph instead of computing eagerly.
  const int64_t n = 5000;
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; i++) v[static_cast<size_t>(i)] = static_cast<float>(i);
  auto base = array::from(v);

  auto lazy = base + 0.0f;  // an unevaluated `affine` node, not yet stored
  auto sliced = lazy.slice(10, 3);
  auto scaled = sliced * 10.0f + 1.0f;
  CHECK(scaled.at({0}) == doctest::Approx(101.0f));  // 10*1
  CHECK(scaled.at({1}) == doctest::Approx(111.0f));  // 10*11 + 1
  CHECK(scaled.at({2}) == doctest::Approx(121.0f));  // 10*12 + 1

  // Same shape of bug through unfold's window view.
  auto windowed = lazy.unfold(0, 3, 1) * 2.0f + 5.0f;
  CHECK(windowed.at({7, 0}) == doctest::Approx(2.0f * 7 + 5));
}

TEST_CASE("slice/pad/unfold accept a negative axis, matching sum/mean/max's convention") {
  std::vector<float> v(12);
  for (int i = 0; i < 12; i++) v[static_cast<size_t>(i)] = static_cast<float>(i);
  auto x = array::from(v, {3, 4});  // rank 2: axis -1 == axis 1

  auto s_pos = x.slice(1, 1, 2);
  auto s_neg = x.slice(-1, 1, 2);
  CHECK(array_equal(s_pos, s_neg));

  auto p_pos = x.pad(1, 1, 0);
  auto p_neg = x.pad(-1, 1, 0);
  CHECK(array_equal(p_pos, p_neg));

  auto u_pos = x.unfold(1, 2, 2);
  auto u_neg = x.unfold(-1, 2, 2);
  CHECK(array_equal(u_pos, u_neg));

  CHECK_THROWS(x.slice(-3, 0, 1));  // out of range even after normalizing
}

TEST_CASE("fold: non-overlapping windows reconstruct the source exactly") {
  auto x = array::from({1, 2, 3, 4, 5, 6});
  auto w = x.unfold(0, 2, 2);  // windows [1,2] [3,4] [5,6], no overlap
  auto rec = w.fold(0, 6, 2);
  CHECK(rec.shape() == tl::shape_t{6});
  CHECK(array_equal(rec, x));
}

TEST_CASE("fold: overlapping windows accumulate, matching a hand-computed sum") {
  auto x = array::from({1, 2, 3, 4, 5, 6});
  auto w = x.unfold(0, 3, 1);  // 4 windows, step 1 < size 3: heavy overlap
  auto rec = w.fold(0, 6, 1);
  // position i's value = i+1, contributed once per window covering it.
  // pos0: {w0}, pos1: {w0,w1}, pos2: {w0,w1,w2}, pos3: {w1,w2,w3},
  // pos4: {w2,w3}, pos5: {w3} — each contributing (i+1) per window.
  float expected[6] = {1, 4, 9, 12, 10, 6};
  for (int i = 0; i < 6; i++) CHECK(rec.at({i}) == expected[i]);

  // Sum conservation: fold never drops or duplicates a source element, it
  // only ever redirects each one to exactly one destination slot.
  CHECK(rec.sum() == doctest::Approx(w.sum()));
}

TEST_CASE("fold: 2-D pad -> unfold -> unfold -> fold -> fold round-trips a non-overlapping tiling") {
  std::vector<float> img(16);
  for (int i = 0; i < 16; i++) img[i] = static_cast<float>(i);
  auto x = array::from(img, {4, 4});
  auto windows = x.unfold(0, 2, 2).unfold(1, 2, 2);  // [2,2,2,2], no overlap

  // fold undoes the two unfolds in reverse order. Undoing the W-unfold
  // (axis 1) drops the trailing kw axis and widens axis 1 back to 4,
  // leaving [oh=2, W=4, kh=2] — the H-unfold's own trailing axis (kh) is
  // still there, one `fold` away from being undone too.
  auto after_w = windows.fold(1, 4, 2);
  CHECK(after_w.shape() == tl::shape_t{2, 4, 2});
  auto rec = after_w.fold(0, 4, 2);
  CHECK(rec.shape() == tl::shape_t{4, 4});
  CHECK(array_equal(rec, x));
}

TEST_CASE("fold accepts a negative axis, matching slice/pad/unfold") {
  auto x = array::from({1, 2, 3, 4, 5, 6});
  auto w = x.unfold(0, 3, 1);
  // The axis fold's bounds-check against is the *reconstructed* (pre-unfold)
  // rank, which is 1 here (x is rank 1) — so -1, not w's own rank, is what
  // maps back to 0.
  CHECK(array_equal(w.fold(0, 6, 1), w.fold(-1, 6, 1)));
}

TEST_CASE("pad and fold are real lazy-graph nodes, not an eager batch boundary") {
  // Large enough to skip the eager-tiny fast path, so `+ 0.0f` genuinely
  // builds an unevaluated `affine` node rather than computing immediately.
  const int64_t n = 5000;
  std::vector<float> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; i++) v[static_cast<size_t>(i)] = static_cast<float>(i);
  auto base = array::from(v);
  auto lazy = base + 0.0f;
  CHECK_FALSE(lazy.materialized());

  // pad() must not force `lazy` to materialize just to build the node.
  auto padded = lazy.pad(0, 2, 0);
  CHECK_FALSE(padded.materialized());
  CHECK_FALSE(lazy.materialized());  // still not forced by pad() itself

  // ...and the fused affine epilogue (the earlier view_ bug's exact shape)
  // must survive through a pad_ node too, since `graph::affine` fuses onto
  // any unevaluated non-constant node, pad_ included.
  auto scaled = padded * 10.0f + 1.0f;
  CHECK(scaled.at({0}) == doctest::Approx(1.0f));    // pad: 0*10+1
  CHECK(scaled.at({2}) == doctest::Approx(1.0f));    // original[0]=0: 0*10+1
  CHECK(scaled.at({3}) == doctest::Approx(11.0f));   // original[1]=1: 1*10+1

  // Same for fold: build it from a still-lazy unfold view over a lazy
  // source, and confirm the epilogue still applies.
  auto lazy2 = base + 0.0f;
  auto windowed = lazy2.unfold(0, 3, 1);
  CHECK_FALSE(windowed.materialized());
  auto folded = windowed.fold(0, n, 1);
  CHECK_FALSE(folded.materialized());
  auto folded_scaled = folded * 2.0f + 3.0f;
  // position 0 is covered by exactly one window -> raw fold value == 0
  CHECK(folded_scaled.at({0}) == doctest::Approx(3.0f));  // 0*2+3
}

TEST_CASE("pad: GPU dispatch matches the ref oracle") {
  // matches_gpu_oracle is a no-op (trivially true) where no GPU device
  // exists — real coverage needs a CUDA (or, once written, Metal) backend.
  std::vector<float> v(37);
  for (int i = 0; i < 37; i++) v[static_cast<size_t>(i)] = static_cast<float>(i) - 10.0f;
  auto x = array::from(v, {1, 37});
  CHECK(matches_gpu_oracle([&] { return x.pad(1, 5, 8); }));
  CHECK(matches_gpu_oracle([&] { return x.pad(0, 2, 3); }));
}

TEST_CASE("pad + unfold composed on the GPU matches the CPU im2col path") {
  std::vector<float> img(64);
  for (int i = 0; i < 64; i++) img[static_cast<size_t>(i)] = static_cast<float>(i);
  auto x = array::from(img, {8, 8});
  CHECK(matches_gpu_oracle([&] {
    return x.pad(0, 1, 1).pad(1, 1, 1).unfold(0, 2, 2).unfold(1, 2, 2);
  }));
}

TEST_CASE("fold: GPU dispatch matches the ref oracle on a contiguous window tensor") {
  // Built via array::from directly (not through unfold's view) so
  // gpu_fold_'s contiguous-input requirement is actually exercised here,
  // rather than silently falling back to the CPU because the view chain
  // that normally feeds fold() is itself non-contiguous.
  std::vector<float> overlapping = {1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6};
  auto w = array::from(overlapping, {4, 3});  // 4 windows of 3, step 1: overlap
  CHECK(matches_gpu_oracle([&] { return w.fold(0, 6, 1); }));  // exercises atomicAdd

  std::vector<float> tiled = {1, 2, 3, 4, 5, 6};
  auto w2 = array::from(tiled, {3, 2});  // 3 windows of 2, step 2: no overlap
  CHECK(matches_gpu_oracle([&] { return w2.fold(0, 6, 2); }));
}

TEST_CASE("fold: 2-D round trip matches the CPU oracle regardless of which "
         "backend actually ran it") {
  // The unfold-view chain feeding fold here is generally non-contiguous, so
  // this mainly re-confirms overall correctness (CPU fallback included) —
  // the contiguous-input case above is what actually pins down the GPU
  // kernel's own correctness.
  std::vector<float> img(64);
  for (int i = 0; i < 64; i++) img[static_cast<size_t>(i)] = static_cast<float>(i);
  auto x = array::from(img, {8, 8});
  CHECK(matches_gpu_oracle([&] {
    auto windows = x.unfold(0, 2, 2).unfold(1, 2, 2);
    return windows.fold(1, 8, 2).fold(0, 8, 2);
  }));
}

TEST_CASE("index_select: gathers rows by a 1-D index array") {
  auto table = array::from({1, 2, 3, 4, 5, 6, 7, 8}, {4, 2});  // 4 rows of 2
  auto idx = array::from({2, 0, 2}, {3});
  auto out = table.index_select(idx);
  CHECK(out.shape() == tl::shape_t{3, 2});
  CHECK(out.at({0, 0}) == doctest::Approx(5));
  CHECK(out.at({0, 1}) == doctest::Approx(6));
  CHECK(out.at({1, 0}) == doctest::Approx(1));
  CHECK(out.at({1, 1}) == doctest::Approx(2));
  CHECK(out.at({2, 0}) == doctest::Approx(5));
  CHECK(out.at({2, 1}) == doctest::Approx(6));

  // Rows that are not dense: a transposed table, rows [1 3 5] and [2 4 6].
  auto t = array::from({1, 2, 3, 4, 5, 6}, {3, 2}).transpose();
  auto got = t.index_select(array::from({1, 0, 1}, {3})).eval();
  CHECK(tl::allclose(got, array::from({2, 4, 6, 1, 3, 5, 2, 4, 6}, {3, 3})));
}

TEST_CASE("index_select: own CPU rows across the pool match the ref oracle") {
  auto idx = [](int64_t n, int64_t limit) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; i++) v[i] = float((i * 7919) % limit);
    return array::from(std::move(v), {n});
  };
  CHECK(cpu_matches_ref([&] {
    return random_array({512, 256}, 911).index_select(idx(1024, 512));
  }));
  CHECK(cpu_matches_ref([&] {
    return random_array({64, 6, 40}, 912).index_select(idx(300, 64));
  }));
  // Rows that are not contiguous: a transposed table.
  CHECK(cpu_matches_ref([&] {
    return random_array({96, 128}, 913).transpose().index_select(idx(400, 128));
  }));
  CHECK(cpu_matches_ref([&] {
    return random_array({50}, 914).index_select(idx(70, 50));
  }));
}

TEST_CASE("index_select: GPU dispatch matches the ref oracle") {
  CHECK(matches_gpu_oracle([&] {
    auto table = random_array({64, 16}, 21);
    auto idx = array::from({5, 5, 0, 63, 30, 5}, {6});
    return table.index_select(idx);
  }));
}

TEST_CASE("index_add: index_select's dual, scatter-adds rows by index") {
  auto idx = array::from({1, 1, 0}, {3});   // row 1 hit twice: must accumulate
  auto values = array::from({10, 20, 30, 40, 1, 2}, {3, 2});
  auto out = tl::index_add(idx, values, {4, 2});
  CHECK(out.shape() == tl::shape_t{4, 2});
  CHECK(out.at({0, 0}) == doctest::Approx(1));   // row 0 <- values[2]
  CHECK(out.at({0, 1}) == doctest::Approx(2));
  CHECK(out.at({1, 0}) == doctest::Approx(40));  // row 1 <- values[0]+values[1]
  CHECK(out.at({1, 1}) == doctest::Approx(60));
  CHECK(out.at({2, 0}) == doctest::Approx(0));   // untouched row stays zero
  CHECK(out.at({3, 0}) == doctest::Approx(0));

  // Indices as a strided view (stride 0): every row lands on row 2.
  auto same = tl::index_add(array::from({2}, {1}).broadcast_to({3}), values,
                            {4, 2});
  CHECK(same.at({2, 0}) == doctest::Approx(41));
  CHECK(same.at({2, 1}) == doctest::Approx(62));
  CHECK(same.at({0, 0}) == doctest::Approx(0));
}

TEST_CASE("index_add: is index_select's exact transpose (dot-product check)") {
  // <index_select(a, idx), g> == <a, index_add(idx, g, a.shape())> for any
  // a/g -- the standard way to confirm a scatter and a gather are duals.
  auto a = random_array({20, 5}, 31);
  auto idx = array::from({3, 3, 7, 0, 19, 3}, {6});
  auto gathered = a.index_select(idx);
  auto g = random_array({6, 5}, 32);
  float lhs = (gathered * g).eval().sum();
  auto scattered = tl::index_add(idx, g, {20, 5});
  float rhs = (a * scattered).eval().sum();
  CHECK(lhs == doctest::Approx(rhs).epsilon(1e-4));
}

TEST_CASE("index_add: GPU dispatch matches the ref oracle (repeated indices)") {
  CHECK(matches_gpu_oracle([&] {
    auto idx = array::from({5, 5, 5, 0, 63, 30, 5, 12}, {8});
    auto values = random_array({8, 16}, 22);
    return tl::index_add(idx, values, {64, 16});
  }));
  // More source rows than one 2048-row scan block, a target hit in both
  // blocks and one hit by none, and rows wider than the 256 threads.
  std::vector<float> ix(4500);
  for (size_t i = 0; i < ix.size(); i++) ix[i] = float((i * 7) % 37);
  ix[100] = ix[3000] = 36.0f;
  auto idx = array::from(std::move(ix), {4500});
  auto values = random_array({4500, 300}, 23);
  CHECK(matches_gpu_oracle([&] { return tl::index_add(idx, values, {40, 300}); },
                           1e-4f, 1e-4f));
}

TEST_CASE("index_select/index_add: bad shapes throw") {
  auto table = random_array({4, 2}, 33);
  auto idx2d = random_array({3, 1}, 34);
  CHECK_THROWS(table.index_select(idx2d));  // indices must be rank-1
  auto idx = array::from({0, 1}, {2});
  auto bad_values = random_array({3, 2}, 35);  // 3 rows for 2 indices
  CHECK_THROWS(tl::index_add(idx, bad_values, {4, 2}));
}

TEST_CASE("scatter_to_axis: one-hot scatter into a new trailing axis") {
  auto idx = array::from({2, 0, 1}, {3});
  auto values = array::from({5, 7, 9}, {3});
  auto out = tl::scatter_to_axis(idx, values, 4);
  CHECK(out.shape() == tl::shape_t{3, 4});
  std::vector<float> expected = {0, 0, 5, 0,   // row 0: value at slot 2
                                 7, 0, 0, 0,   // row 1: value at slot 0
                                 0, 9, 0, 0};  // row 2: value at slot 1
  for (int64_t i = 0; i < 3; i++) {
    for (int64_t k = 0; k < 4; k++) {
      CHECK(out.at({i, k}) ==
           doctest::Approx(expected[static_cast<size_t>(i * 4 + k)]));
    }
  }
}

TEST_CASE("scatter_to_axis: is the exact shape Pooling's own backward needs "
         "(2-D window axis)") {
  // argmax/max over a [n, oh, ow, kh*kw] window axis, scattered back --
  // the same shape a pooling layer's hand-derived backward composes with
  // fold() to reconstruct the padded input gradient.
  auto idx = array::from({0, 3, 2, 1}, {2, 2});  // [n=2, ow=2], indices < 4
  auto values = array::from({1, 2, 3, 4}, {2, 2});
  auto out = tl::scatter_to_axis(idx, values, 4);
  CHECK(out.shape() == tl::shape_t{2, 2, 4});
  CHECK(out.at({0, 0, 0}) == doctest::Approx(1));
  CHECK(out.at({0, 1, 3}) == doctest::Approx(2));
  CHECK(out.at({1, 0, 2}) == doctest::Approx(3));
  CHECK(out.at({1, 1, 1}) == doctest::Approx(4));
  CHECK(out.sum() == doctest::Approx(1 + 2 + 3 + 4));  // every other slot is 0
}

TEST_CASE("index ops: a label outside the target throws at eval") {
  // Unchecked, these were an out-of-bounds read (index_select) and two
  // out-of-bounds WRITES (index_add, scatter_to_axis). The check is at eval
  // rather than at construction, because that is when the labels exist: they
  // are a tensor, and this one may itself be the result of a graph.
  auto table = random_array({4, 2}, 37);
  CHECK_THROWS(table.index_select(array::from({0, 4}, {2})).eval());
  CHECK_THROWS(table.index_select(array::from({-1}, {1})).eval());

  auto row = random_array({1, 2}, 38);
  CHECK_THROWS(tl::index_add(array::from({4}, {1}), row, {4, 2}).eval());

  auto vals = random_array({3}, 39);
  CHECK_THROWS(tl::scatter_to_axis(array::from({0, 1, 4}, {3}), vals, 4).eval());

  // The last in-range label is still in range.
  CHECK_NOTHROW(table.index_select(array::from({3}, {1})).eval());
  CHECK_NOTHROW(
      tl::scatter_to_axis(array::from({0, 1, 3}, {3}), vals, 4).eval());
}

TEST_CASE("gather_from_axis: scatter_to_axis's dual, one element per position") {
  auto src = array::from({0, 0, 5, 0,   // row 0: the labelled slot holds 5
                          7, 0, 0, 0,   // row 1: 7
                          0, 9, 0, 0},  // row 2: 9
                         {3, 4});
  auto idx = array::from({2, 0, 1}, {3});
  auto out = tl::gather_from_axis(src, idx);
  CHECK(out.shape() == tl::shape_t{3});
  CHECK(out.at({0}) == doctest::Approx(5));
  CHECK(out.at({1}) == doctest::Approx(7));
  CHECK(out.at({2}) == doctest::Approx(9));
}

TEST_CASE("gather_from_axis: undoes scatter_to_axis (2-D window axis)") {
  auto idx = array::from({0, 3, 2, 1}, {2, 2});
  auto values = array::from({1, 2, 3, 4}, {2, 2});
  auto out = tl::gather_from_axis(tl::scatter_to_axis(idx, values, 4), idx);
  CHECK(out.shape() == tl::shape_t{2, 2});
  for (int64_t i = 0; i < 2; i++) {
    for (int64_t j = 0; j < 2; j++) {
      CHECK(out.at({i, j}) == doctest::Approx(values.at({i, j})));
    }
  }
}

TEST_CASE("gather_from_axis: GPU dispatch matches the ref oracle") {
  CHECK(matches_gpu_oracle([&] {
    auto src = random_array({4, 16}, 41);
    return tl::gather_from_axis(src, array::from({15, 0, 7, 3}, {4}));
  }));
  CHECK(matches_gpu_oracle([&] {
    auto src = random_array({2, 3, 8}, 42);
    return tl::gather_from_axis(src, array::from({0, 7, 3, 3, 1, 6}, {2, 3}));
  }));
}

TEST_CASE("gather_from_axis: bad shapes and labels throw") {
  auto src = random_array({3, 4}, 43);
  // indices must be src's shape minus the trailing axis
  CHECK_THROWS(tl::gather_from_axis(src, array::from({0, 1}, {2})));
  CHECK_THROWS(tl::gather_from_axis(src, array::from({0, 1, 2}, {3}).reshape(
                                             {3, 1})));
  // and label the trailing axis, checked at eval like the scatter's own
  CHECK_THROWS(tl::gather_from_axis(src, array::from({0, 4, 1}, {3})).eval());
  CHECK_NOTHROW(tl::gather_from_axis(src, array::from({0, 3, 1}, {3})).eval());
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

TEST_CASE("scatter_to_axis: mismatched indices/values shape throws") {
  auto idx = array::from({0, 1}, {2});
  auto values = random_array({3}, 36);
  CHECK_THROWS(tl::scatter_to_axis(idx, values, 4));
}

TEST_CASE("scatter_to_axis: GPU dispatch matches the ref oracle") {
  CHECK(matches_gpu_oracle([&] {
    std::mt19937 rng(41);
    std::uniform_int_distribution<int> dist(0, 7);
    std::vector<float> idx_v(64 * 32);
    for (auto& x : idx_v) x = static_cast<float>(dist(rng));
    auto idx = array::from(idx_v, {64, 32});
    auto values = random_array({64, 32}, 42);
    return tl::scatter_to_axis(idx, values, 8);
  }));
}

// --- broadcast binop GPU dispatch: was a hard stub on CUDA (binary_bcast
// unconditionally returned false) regardless of rank; now real at rank 2
// and, via a new N-D kernel, at any other rank too (a Transformer's
// [N,S,D] - [N,S,1] LayerNorm mean-subtract is rank 3). ---

TEST_CASE("broadcast binop: GPU dispatch matches the ref oracle (rank 2)") {
  CHECK(matches_gpu_oracle([&] {
    auto x = random_array({64, 32}, 51);
    auto bias = random_array({1, 32}, 52);
    return x + bias;
  }));
  CHECK(matches_gpu_oracle([&] {
    auto x = random_array({64, 32}, 53);
    auto col = random_array({64, 1}, 54);
    return x - col;
  }));
  CHECK(matches_gpu_oracle([&] {
    auto x = random_array({64, 32}, 55);
    return x * array::full({}, 2.5f);
  }));
}

TEST_CASE("broadcast binop: GPU dispatch matches the ref oracle (rank 3, "
         "LayerNorm-shaped)") {
  CHECK(matches_gpu_oracle([&] {
    auto x = random_array({4, 16, 32}, 56);         // [N, S, D]
    auto mean = random_array({4, 16, 1}, 57);        // [N, S, 1]
    return x - mean;
  }));
  CHECK(matches_gpu_oracle([&] {
    auto x = random_array({4, 16, 32}, 58);
    auto gamma = random_array({1, 32}, 59);          // [1, D], rank promoted
    return x * gamma;
  }));
}

TEST_CASE("broadcast binop: GPU dispatch matches the ref oracle (rank 4, "
         "attention-mask-shaped)") {
  CHECK(matches_gpu_oracle([&] {
    auto scores = random_array({2, 4, 8, 8}, 60);    // [batch, heads, S, S]
    auto bias = random_array({1, 1, 8, 8}, 61);
    return scores + bias;
  }));
}

// --- where() GPU dispatch: had none on any backend before this. ---

TEST_CASE("where: GPU dispatch matches the ref oracle (rank 2)") {
  CHECK(matches_gpu_oracle([&] {
    auto cond = random_array({8, 8}, 71) > 0.0f;
    auto a = random_array({8, 8}, 72);
    auto b = random_array({8, 8}, 73);
    return tl::where(cond, a, b);
  }));
}

TEST_CASE("where: GPU dispatch matches the ref oracle (rank 4, attention-"
         "mask-shaped, broadcasting the mask over batch and heads)") {
  CHECK(matches_gpu_oracle([&] {
    auto mask = random_array({1, 1, 8, 8}, 74) > 0.0f;
    auto scores = random_array({2, 4, 8, 8}, 75);
    return tl::where(mask, scores, array::full({}, -1.0e9f));
  }));
}

// --- tensor-scalar ops: pow(x, s) and x OP s carry s in the node and as a
// kernel argument, not as a rank-0 operand.

TEST_CASE("tensor-scalar ops match the ref oracle on every path") {
  auto x = random_array({64, 32}, 83).relu() + 0.1f;
  auto p = array::from({-3, -1, 0, 2, 4, -5, 6, 1}, {2, 4});
  for (float s : {2.0f, 0.5f, -0.5f, -1.0f, 3.0f}) {
    CHECK(matches_gpu_oracle([&] { return tl::pow(x, s); }));
    CHECK(cpu_matches_ref([&] { return tl::pow(x, s); }));
  }
  // The fused epilogue: LayerNorm's `(var + eps).pow(-0.5)` then an affine.
  CHECK(matches_gpu_oracle([&] { return tl::pow(x + 1e-5f, -0.5f) * 2.0f + 1.0f; }));
  CHECK(matches_gpu_oracle([&] { return (p >= 1.0f) * 3.0f - 1.0f; }));
  CHECK(matches_gpu_oracle([&] { return (x < 0.5f) * x; }));

  // Against std::pow directly, past the tiny-tensor cutoff (eval_one's path).
  auto big = random_array({128, 64}, 84).relu() + 0.1f;
  auto src = big.eval();
  std::vector<float> want;
  for (int64_t i = 0; i < 128; i++) {
    for (int64_t j = 0; j < 64; j++) want.push_back(std::pow(src.at({i, j}), -0.5f));
  }
  CHECK(tl::allclose(tl::pow(big, -0.5f), array::from(want, {128, 64}), 1e-5f, 1e-6f));

  // The eager tiny path.
  auto t = tl::pow(array::from({4.0f, 9.0f}, {2}), -0.5f).eval();
  CHECK(t.at({0}) == doctest::Approx(0.5f));
  CHECK(t.at({1}) == doctest::Approx(1.0f / 3.0f));
  auto m = (p >= 0.0f).eval();
  CHECK(m.at({0, 0}) == 0.0f);
  CHECK(m.at({0, 2}) == 1.0f);
}

// --- host/device coherence on the CPU fast paths (CUDA's mirror storage).
// A buffer the GPU last wrote lives on the device until a CPU access pulls
// it back; the tiny-tensor fast paths and add_ used to read/write the host
// mirror directly, so in auto mode an MNIST step's small gradient gemms
// consumed stale activations. Producing on the GPU and consuming in cpu
// mode is the same hand-off without depending on the auto thresholds.

TEST_CASE("CPU fast paths see what the GPU last wrote") {
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;

  tl::use_gpu();
  auto x = array::ones({784, 10});
  auto q = array::ones({30, 784}).dot(x).eval();  // [30, 10] of 784

  tl::use_cpu();
  auto s = array::ones({10, 30}).dot(q).eval();  // tiny gemm: fast-dot path
  CHECK(s.at({0, 0}) == 30.0f * 784.0f);
  auto e = (q + array::ones({30, 10})).eval();   // tiny same-shape: fast-ew
  CHECK(e.at({0, 0}) == 785.0f);

  tl::device_ = prev;
}

TEST_CASE("add_ invalidates the device copy the GPU will read next") {
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;

  tl::use_gpu();
  auto x = array::ones({784, 10});
  auto w = array::ones({30, 784});
  w.dot(x).eval();                   // the GPU reads w: mirror now BOTH
  w.add_(array::ones({30, 784}));    // CPU write: host 2, device must not stay 1
  auto c = w.dot(x).eval();
  CHECK(c.at({0, 0}) == 2.0f * 784.0f);

  tl::device_ = prev;
}

// tl::profile: the evaluator opens a scope per op under the caller's own, and
// what a backend launches lands under the innermost one. CUDA and Metal stamp
// a per-launch time; every backend counts.
TEST_CASE("profile: scopes nest into paths and launches land under them") {
  using tl::profile::row;
  auto a = random_array({96, 96}, 1500), b = random_array({96, 96}, 1501);
  a.eval();
  b.eval();

  tl::profile::start();
  {
    tl::profile::scope phase("phase");
    tl::profile::scope mm("mm");
    a.dot(b).eval();
  }
  tl::profile::stop();
  auto rows = tl::profile::rows();

  auto find = [&](std::string_view path, row::kind_t kind) -> const row* {
    auto it = std::find_if(rows.begin(), rows.end(), [&](const row& r) {
      return r.path == path && r.kind == kind;
    });
    return it == rows.end() ? nullptr : &*it;
  };
  const row* phase = find("phase", row::kind_t::scope);
  REQUIRE(phase);
  CHECK(phase->count == 1);
  CHECK(phase->host_us > 0);
  const row* dot = find("phase/mm/dot", row::kind_t::scope);
  REQUIRE(dot);
  CHECK(dot->count == 1);
  CHECK(phase->host_us >= dot->host_us);

  const bool on_device =
      tl::gpu_available() && tl::device_ == tl::device_type::gpu;
  std::vector<const row*> launches;
  for (const row& r : rows) {
    if (r.path == "phase/mm/dot" && r.kind == row::kind_t::launch) {
      launches.push_back(&r);
    }
  }
  if (on_device) {
    // the matmul dispatched at least one kernel, and the batch's flush was
    // waited for under the scope that evaluated it
    CHECK(!launches.empty());
    const row* wait = find("phase/mm", row::kind_t::wait);
    REQUIRE(wait);
    CHECK(wait->count >= 1);
#if !defined(TENSORLIB_WEBGPU)
    for (const row* r : launches) {
      CHECK(r->device_timed == r->count);
      CHECK(r->device_us > 0);
    }
#endif
#ifdef __APPLE__
    CHECK(tl::profile::summarize().batches >= 1);
#endif
  } else {
    CHECK(launches.empty());  // the CPU path launches nothing
  }
  CHECK(tl::profile::summarize().scopes >= 3);

  // Stopped: the next evaluation leaves no trace.
  {
    tl::profile::scope after("after");
    a.dot(b).eval();
  }
  rows = tl::profile::rows();
  CHECK(!find("after", row::kind_t::scope));

  // A new session starts from nothing.
  tl::profile::start();
  tl::profile::stop();
  CHECK(tl::profile::rows().empty());
}

// An eager op names itself only past its operand checks: one that declines
// there (a strided operand, or the oracle with the own CPU off) leaves no row.
TEST_CASE("profile: an eager op that declines on its operands leaves no row") {
  auto x = random_array({8, 32}, 1510), g = random_array({32}, 1511),
       dy = random_array({8, 32}, 1512);
  auto strided = random_array({32, 8}, 1513).transpose();
  auto prev = tl::device_;
  tl::use_cpu();
  tl::profile::start();
  CHECK(!tl::array::layer_norm_bwd(strided, g, dy));
  tl::cpu::enabled_ = false;
  CHECK(!tl::array::layer_norm_bwd(x, g, dy));
  tl::cpu::enabled_ = true;
  if (tl::gpu_available()) {
    tl::use_gpu();
    CHECK(!tl::array::layer_norm_bwd(strided, g, dy));
  }
  tl::profile::stop();
  tl::device_ = prev;
  for (const auto& r : tl::profile::rows()) {
    CHECK(r.path.find("layer_norm_bwd") == std::string::npos);
  }
}

// A buffer released while the device still has work queued against it goes
// straight back to the pool for device work, but not for the host to fill.
// Here the logsumexp is encoded but not run when its temporary dies (xent_bwd
// is eager and keeps only the kernel), and ones() asks for the same 32 bytes:
// handed that buffer and filled at once, its 1s would be overwritten by the
// queued logsumexp.
TEST_CASE("host-filled storage skips buffers the pending batch may write") {
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;
  tl::use_gpu();
  auto logits = random_array({8, 32}, 48);
  auto targets = array::from({0, 5, 31, 12, 7, 7, 1, 30}, {8});
  auto g = random_array({8}, 49);
  auto got = tl::array::xent_bwd(logits, logits.logsumexp(1), targets, g);
  auto ones = tl::array::ones({8});
  tl::eval(ones);
  for (int64_t i = 0; i < 8; i++) CHECK(ones.at({i}) == 1.0f);
  tl::device_ = prev;
}

TEST_CASE("the KV cache and the decode step's kernels match their array forms") {
  // The model path runs on raw device buffers through gpu::, outside the lazy
  // graph; each kernel here is checked against the array composition that
  // defines it, on the GPU (no-op where there is none). Shapes are Qwen2's
  // head geometry (D=64, 14 q heads over 2 kv heads) at a context past the
  // split-KV cutoff, plus the D=128 instantiation.
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;
  tl::use_gpu();
  namespace gpu = tl::gpu;

  // A device buffer from host values, evaluated so native() is valid.
  auto dev = [](const array& a) {
    array c = a.clone();
    c.eval();
    return c;
  };
  auto same = [](const array& got, const array& want, float tol) {
    return tl::allclose(got, want, tol, tol);
  };

  SUBCASE("kv_cache: append + attn, prefill, in f32 and bf16") {
    for (int64_t D : {64, 128}) {
      for (tl::dtype kv : {tl::dtype::f32, tl::dtype::bf16}) {
        const int64_t HQ = 14, HKV = 2, MAXC = 300, T = 260;
        const float scale = 1.0f / std::sqrt((float)D);
        tl::kv_cache cache;
        REQUIRE(cache.init(HKV, MAXC, D, kv));
        // T steps appended one at a time, then a query over all of them.
        auto K = random_array({HKV, T, D}, 900), V = random_array({HKV, T, D}, 901);
        for (int64_t t = 0; t < T; t++) {
          array k = dev(K.slice(1, t, 1).clone());  // [HKV,1,D]
          array v = dev(V.slice(1, t, 1).clone());
          REQUIRE(cache.append(k.native(), v.native()));
        }
        CHECK(cache.pos == T);
        array q = dev(random_array({HQ, D}, 902));
        array out = array::empty({HQ, D});
        REQUIRE(cache.attn(q.native(), out.native(), HQ, scale));
        // Explicit: each q head attends its kv head's T rows.
        auto Kr = kv == tl::dtype::bf16 ? K.to_bf16().to_f32() : K;
        auto Vr = kv == tl::dtype::bf16 ? V.to_bf16().to_f32() : V;
        std::vector<float> want((size_t)HQ * D);
        for (int64_t h = 0; h < HQ; h++) {
          const int64_t kh = h / (HQ / HKV);
          std::vector<double> s(T);
          double m = -1e300, sum = 0;
          for (int64_t j = 0; j < T; j++) {
            double a = 0;
            for (int64_t d = 0; d < D; d++) a += (double)q.at({h, d}) * Kr.at({kh, j, d});
            s[j] = a * scale;
            m = std::max(m, s[j]);
          }
          for (auto& x : s) sum += (x = std::exp(x - m));
          for (int64_t d = 0; d < D; d++) {
            double e = 0;
            for (int64_t j = 0; j < T; j++) e += s[j] * Vr.at({kh, j, d});
            want[h * D + d] = (float)(e / sum);
          }
        }
        tl::gpu::flush();
        CHECK(same(out, array::from(want, {HQ, D}), 1e-4f));

        // The same T rows as one prefill into a fresh cache: the block's own
        // causal attention, then a decode step after it agrees with above.
        tl::kv_cache c2;
        REQUIRE(c2.init(HKV, MAXC, D, kv));
        array qp = dev(random_array({HQ, T, D}, 903));
        array Kd = dev(K), Vd = dev(V);
        array op = array::empty({HQ, T, D});
        REQUIRE(c2.prefill(qp.native(), Kd.native(), Vd.native(), op.native(), T,
                           HQ, scale));
        CHECK(c2.pos == T);
        array out2 = array::empty({HQ, D});
        REQUIRE(c2.attn(q.native(), out2.native(), HQ, scale));
        tl::gpu::flush();
        CHECK(same(out2, out, 1e-5f));
        // Row t of the prefill output is the decode of q row t over keys 0..t.
        const int64_t t = T - 1;
        array qt = dev(qp.slice(1, t, 1).reshape({HQ, D}).clone());
        array ot = array::empty({HQ, D});
        REQUIRE(c2.attn(qt.native(), ot.native(), HQ, scale));
        tl::gpu::flush();
        CHECK(same(ot, op.slice(1, t, 1).reshape({HQ, D}), 1e-4f));
      }
    }
  }

  SUBCASE("rmsnorm, rmsnorm_res and swiglu per row") {
    const int64_t rows = 3, n = 896, ff = 4864;
    array x = dev(random_array({rows, n}, 910)), d = dev(random_array({rows, n}, 911));
    array w = dev(random_array({n}, 912));
    array h = array::empty({rows, n}), xo = array::empty({rows, n}),
          h2 = array::empty({rows, n});
    REQUIRE(gpu::rmsnorm(x.native(), w.native(), h.native(), n, 1e-6f, rows));
    REQUIRE(gpu::rmsnorm_res(x.native(), d.native(), w.native(), xo.native(),
                             h2.native(), n, 1e-6f, rows));
    tl::gpu::flush();
    CHECK(same(h, array::rmsnorm(x, w, 1e-6f), 1e-5f));
    CHECK(same(xo, x + d, 1e-6f));
    CHECK(same(h2, array::rmsnorm(x + d, w, 1e-6f), 1e-5f));

    // A row small enough that eps carries the reciprocal: at this scale
    // mean(x²) is ~1e-8 against eps 1e-6, so dropping eps changes the result
    // by a factor of ten rather than the last bits.
    array tiny = dev(x * 1e-4f);
    array ht = array::empty({rows, n});
    REQUIRE(gpu::rmsnorm(tiny.native(), w.native(), ht.native(), n, 1e-6f, rows));
    tl::gpu::flush();
    CHECK(same(ht, array::rmsnorm(tiny, w, 1e-6f), 1e-5f));

    array gu = dev(random_array({rows, 2 * ff}, 913));
    array o = array::empty({rows, ff});
    REQUIRE(gpu::swiglu(gu.native(), o.native(), ff, rows));
    tl::gpu::flush();
    array gate = gu.slice(1, 0, ff), up = gu.slice(1, ff, ff);
    CHECK(same(o, array::swiglu(gate, up), 1e-5f));
  }

  SUBCASE("rope with a fused bias") {
    const int64_t H = 14, D = 64;
    array x = dev(random_array({H, D}, 920)), b = dev(random_array({H, D}, 921));
    array o = array::empty({H, D});
    REQUIRE(gpu::rope(x.native(), o.native(), H, 1, D, 37, 1e6f, b.native()));
    tl::gpu::flush();
    CHECK(same(o, array::rope(x + b, 37, 1e6f), 1e-5f));
  }

  SUBCASE("argmax reads back the smallest index among ties") {
    std::vector<float> v(151936, -1.0f);
    v[77777] = 5.0f;
    v[77778] = 5.0f;  // a tie: the smaller index wins, as the host scan does
    array a = dev(array::from(v, {(int64_t)v.size()}));
    int64_t idx = -1;
    REQUIRE(gpu::argmax(a.native(), (int64_t)v.size(), &idx));
    CHECK(idx == 77777);
  }

  SUBCASE("split_heads and merge_heads are inverses through the fused layout") {
    const int64_t T = 5, H = 18, D = 64, ld = H * D;
    array src = dev(random_array({T, ld}, 930)), bias = dev(random_array({H, D}, 931));
    array heads = array::empty({H, T, D}), back = array::empty({T, ld});
    REQUIRE(gpu::split_heads(src.native(), bias.native(), heads.native(), T, ld, 0,
                             H, D));
    REQUIRE(gpu::merge_heads(heads.native(), back.native(), T, H, D));
    tl::gpu::flush();
    // heads[h, t, :] = src[t, h*D:(h+1)*D] + bias[h]
    CHECK(same(heads, src.reshape({T, H, D}).transpose({1, 0, 2}) +
                          bias.reshape({H, 1, D}), 1e-6f));
    CHECK(same(back, src + bias.reshape({1, ld}), 1e-6f));
  }

  tl::device_ = prev;
}
