#include "doctest.h"

#include <tensorlib.h>

using tl::array;

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

  // reduction over a strided view
  CHECK(a.transpose().sum(0).at({1}) == 15.0f);
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
