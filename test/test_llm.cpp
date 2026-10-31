// The llm layer's tests (layers: include/gpu_abi.h): the KV cache and the decode
// step's kernels, the graph replay, the GGUF reader.

#include "test_util.h"

#include <gguf.h>
#include <kv_cache.h>

// A BF16 tensor (ggml type 30, as BF16 GGUFs ship their weights) is read with
// its size and bytes as the file has them. The smallest GGUF v3 that holds
// one: no metadata, one 1-D tensor of four values, the data section aligned
// to the default 32.
TEST_CASE("gguf reads a BF16 tensor") {
  std::string f;
  auto put = [&](const void* p, size_t n) { f.append(static_cast<const char*>(p), n); };
  auto u32 = [&](uint32_t v) { put(&v, 4); };
  auto u64 = [&](uint64_t v) { put(&v, 8); };
  put("GGUF", 4);
  u32(3);  // version
  u64(1);  // tensors
  u64(0);  // metadata pairs
  const std::string name = "w";
  u64(name.size());
  put(name.data(), name.size());
  u32(1);   // dims
  u64(4);   // ne[0]
  u32(30);  // GGML_TYPE_BF16
  u64(0);   // offset in the data section
  f.resize((f.size() + 31) / 32 * 32, '\0');
  // Exact in bf16, the last near the top of its range.
  const float want[4] = {1.0f, -2.5f, 0.15625f, std::ldexp(1.0f, 127)};
  uint16_t raw[4];
  for (int i = 0; i < 4; i++) {
    uint32_t bits;
    std::memcpy(&bits, &want[i], 4);
    raw[i] = static_cast<uint16_t>(bits >> 16);
  }
  put(raw, sizeof(raw));
  const std::string path = "gguf_bf16_test.gguf";  // the test's working directory
  { std::ofstream(path, std::ios::binary).write(f.data(), (std::streamsize)f.size()); }
  {
    tl::gguf::model m(path);
    const auto* t = m.tensor("w");
    REQUIRE(t != nullptr);
    CHECK(t->type == tl::gguf::ggml_type::bf16);
    CHECK(t->nbytes == 8);
    const auto* got = static_cast<const uint16_t*>(t->data);
    for (int i = 0; i < 4; i++) CHECK(tl::bf16_to_f32(got[i]) == want[i]);
  }
  std::remove(path.c_str());
}

// A replayed graph keeps the host mirrors in step as its launches would: what
// it writes is device-current afterwards, so a host read after each replay
// sees that replay's result rather than the copy the previous read brought up.
TEST_CASE("a graph replay leaves what it wrote device-current") {
  namespace gpu = tl::gpu;
  if (!tl::gpu_available() || !gpu::graph_available()) return;
  auto prev = tl::device_;
  tl::use_gpu();
  tl::storage a = tl::storage::make(4), one = tl::storage::make(4), b = tl::storage::make(4);
  const float zeros[4] = {0, 0, 0, 0}, ones[4] = {1, 1, 1, 1};
  gpu::upload(a.native, zeros, 4);
  gpu::upload(one.native, ones, 4);
  // a += 1, through b (no launch writes what it reads).
  REQUIRE(gpu::capture_begin());
  gpu::binary(gpu::kop::add, a.device_span(), one.device_span(), b.device_span(), 4, 1, 0);
  gpu::unary(gpu::kop::affine, b.device_span(), a.device_span(), 4, 1, 0);
  gpu::graph_exec g = gpu::capture_end();
  REQUIRE(g != nullptr);
  for (int replay = 1; replay <= 2; replay++) {
    REQUIRE(gpu::graph_launch(g));
    gpu::cpu_barrier();
    gpu::sync_to_host(a.native, false);
    CHECK(a.ptr[0] == doctest::Approx((float)replay));
  }
  gpu::graph_destroy(g);
  tl::device_ = prev;
}

TEST_CASE("the KV cache and the decode step's kernels match their array forms") {
  // The model path runs on raw device buffers through gpu::, outside the lazy
  // graph; each kernel here is checked against the array composition that
  // defines it. Shapes are Qwen2's head geometry (D=64, 14 q heads over 2 kv
  // heads) at a context past the split-KV cutoff, plus the D=128
  // instantiation. A backend without a kernel takes the generic composition,
  // which has to agree too; a bf16 cache needs the backend's own attention.
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;
  tl::use_gpu();
  namespace gpu = tl::gpu;

  // A device buffer from host values, evaluated so native() is valid.

  SUBCASE("kv_cache: append + attn, prefill, in f32 and f16") {
    // K and V as an f16 cache holds them (the reference reads those values).
    auto held16 = [](const array& a) {
      std::vector<float> v(a.raw(), a.raw() + a.size());
      for (float& x : v) x = tl::f16_to_f32(tl::f32_to_f16(x));
      return array::from(std::move(v), a.shape());
    };
    for (int64_t D : {64, 128, 256}) {
      for (const bool f16 : {false, true}) {
        if (f16 && !gpu::caps::kv_f16) continue;
        CAPTURE(D);
        CAPTURE(f16);
        const int64_t HQ = 14, HKV = 2, MAXC = 300, T = 260;
        const float scale = 1.0f / std::sqrt((float)D);
        tl::kv_cache cache;
        REQUIRE(cache.init(HKV, MAXC, D, f16));
        // T steps appended one at a time, then a query over all of them.
        auto K = random_array({HKV, T, D}, 900), V = random_array({HKV, T, D}, 901);
        for (int64_t t = 0; t < T; t++) {
          array k = dev(K.slice(1, t, 1).clone());  // [HKV,1,D]
          array v = dev(V.slice(1, t, 1).clone());
          REQUIRE(cache.append(k.device_span(), v.device_span()));
        }
        CHECK(cache.pos == T);
        array q = dev(random_array({HQ, D}, 902));
        array out = array::empty({HQ, D});
        REQUIRE(cache.attn(q.device_span(), out.device_span(), HQ, scale));
        // Explicit: each q head attends its kv head's T rows.
        auto Kr = f16 ? held16(K) : K;
        auto Vr = f16 ? held16(V) : V;
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
        REQUIRE(c2.init(HKV, MAXC, D, f16));
        array qp = dev(random_array({HQ, T, D}, 903));
        array Kd = dev(K), Vd = dev(V);
        array op = array::empty({HQ, T, D});
        REQUIRE(c2.prefill(qp.device_span(), Kd.device_span(), Vd.device_span(), op.device_span(), T,
                           HQ, scale));
        CHECK(c2.pos == T);
        array out2 = array::empty({HQ, D});
        REQUIRE(c2.attn(q.device_span(), out2.device_span(), HQ, scale));
        tl::gpu::flush();
        CHECK(same(out2, out, 1e-5f));
        // Row t of the prefill output is the decode of q row t over keys 0..t.
        const int64_t t = T - 1;
        array qt = dev(qp.slice(1, t, 1).reshape({HQ, D}).clone());
        array ot = array::empty({HQ, D});
        REQUIRE(c2.attn(qt.device_span(), ot.device_span(), HQ, scale));
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
    REQUIRE(gpu::rmsnorm(x.device_span(), w.device_span(), h.device_span(), n, 1e-6f, rows));
    REQUIRE(gpu::rmsnorm_res(x.device_span(), d.device_span(), w.device_span(), xo.device_span(),
                             h2.device_span(), n, 1e-6f, rows));
    tl::gpu::flush();
    CHECK(same(h, array::rmsnorm(x, w, 1e-6f), 1e-5f));
    CHECK(same(xo, x + d, 1e-6f));
    CHECK(same(h2, array::rmsnorm(x + d, w, 1e-6f), 1e-5f));

    // A row small enough that eps carries the reciprocal: at this scale
    // mean(x²) is ~1e-8 against eps 1e-6, so dropping eps changes the result
    // by a factor of ten rather than the last bits.
    array tiny = dev(x * 1e-4f);
    array ht = array::empty({rows, n});
    REQUIRE(gpu::rmsnorm(tiny.device_span(), w.device_span(), ht.device_span(), n, 1e-6f, rows));
    tl::gpu::flush();
    CHECK(same(ht, array::rmsnorm(tiny, w, 1e-6f), 1e-5f));

    array gu = dev(random_array({rows, 2 * ff}, 913));
    array o = array::empty({rows, ff});
    REQUIRE(gpu::swiglu(gu.device_span(), o.device_span(), ff, rows));
    tl::gpu::flush();
    array gate = gu.slice(1, 0, ff), up = gu.slice(1, ff, ff);
    CHECK(same(o, array::swiglu(gate, up), 1e-5f));
  }

  SUBCASE("rope with a fused bias") {
    const int64_t H = 14, D = 64;
    array x = dev(random_array({H, D}, 920)), b = dev(random_array({H, D}, 921));
    array o = array::empty({H, D});
    REQUIRE(gpu::rope(x.device_span(), o.device_span(), H, 1, D, 37, 1e6f, b.device_span()));
    tl::gpu::flush();
    CHECK(same(o, array::rope(x + b, 37, 1e6f), 1e-5f));
  }

  SUBCASE("argmax reads back the smallest index among ties") {
    std::vector<float> v(151936, -1.0f);
    v[77777] = 5.0f;
    v[77778] = 5.0f;  // a tie: the smaller index wins, as the host scan does
    array a = dev(array::from(v, {(int64_t)v.size()}));
    int64_t idx = -1;
    REQUIRE(gpu::argmax(a.device_span(), (int64_t)v.size(), &idx));
    CHECK(idx == 77777);
  }

  SUBCASE("the [N,K] bf16 weight layout: row GEMV and the prefill GEMM") {
    // GGML stores a linear weight as [out, in], and both of these read it in
    // place. The oracle is the same product through the array API, against
    // the weights ROUNDED to bf16 — the kernels see those bits, so comparing
    // with the f32 originals would measure the rounding, not the kernel.
    //
    // Shapes are deliberately ragged: N = 96 leaves a half-width column
    // block, M = 37 a partial row block (so both edge stores run), and
    // K = 72 a partial K tile on top of the GEMV's k % 8 == 0 requirement.
    const int64_t N = 96, K = 72, M = 37;
    array W = random_array({N, K}, 940);
    array Wb = dev(W.to_bf16()), Wt = dev(W.to_bf16().to_f32().transpose());

    if (gpu::caps::row_gemv) {
      array a = dev(random_array({1, K}, 941));
      array y = array::empty({1, N});
      REQUIRE(gpu::gemv_bf16_row(a.device_span(), Wb.device_span(), y.device_span(), N, K));
      tl::gpu::flush();
      CHECK(same(y, a.dot(Wt), 1e-4f));
      // W from a base off the kernel's 16-byte loads, and the column-major
      // form's ([K,N], N % 8 == 0: the 8-wide path) likewise.
      const std::vector<float> want = host_values(y);
      array Ws = dev(with_lead(W, 2).to_bf16()), ys = dev(array::zeros({1, N}));
      check_misaligned(
          "row GEMV, W off 16 bytes",
          [&] {
            return gpu::gemv_bf16_row(a.device_span(), Ws.device_span().at(4), ys.device_span(), N,
                                      K);
          },
          [&] { return host_close(host_values(ys), want, 1e-4f); });
      array Bs = dev(with_lead(Wt, 2).to_bf16()), yc = dev(array::zeros({1, N}));
      check_misaligned(
          "column GEMV, B off 16 bytes",
          [&] {
            return gpu::gemv_bf16(a.device_span(), Bs.device_span().at(4), yc.device_span(), N, K);
          },
          [&] { return host_close(host_values(yc), want, 1e-4f); });
    }
    if (gpu::caps::bf16_gemm) {
      // The GEMM may round its activations to bf16 too (CUDA's runs on the
      // tensor cores), so the oracle reads the rounded ones as well: exact
      // inputs either way, and any gap left is the kernel's.
      array A = dev(random_array({M, K}, 942).to_bf16().to_f32());
      array C = array::empty({M, N});
      REQUIRE(gpu::gemm_bf16_nt(A.device_span(), Wb.device_span(), C.device_span(), M, N, K));
      tl::gpu::flush();
      CHECK(same(C, A.dot(Wt), 1e-4f));
      // A, then C, from a base off the tensor cores' vectors (and A off the
      // SIMT tiles' float4s), then W one bf16 off (off the 4 bytes Metal's
      // wider loads need) and two (on them, off 16): computed as above, never
      // refused, since the prefill ignores the result and a refusal would
      // leave its chunk unwritten.
      const std::vector<float> want = host_values(C);
      array As = dev(with_lead(A, 1)), W2s = dev(with_lead(W, 1).to_bf16()),
            W4s = dev(with_lead(W, 2).to_bf16());
      const struct {
        const char* what;
        tl::gpu::span a, w;
        int64_t c_lead;
      } cases[] = {
          {"prefill GEMM, A off 16 bytes", As.device_span().at(4), Wb.device_span(), 0},
          {"prefill GEMM, C off 8 bytes", A.device_span(), Wb.device_span(), 1},
          {"prefill GEMM, W off 4 bytes", A.device_span(), W2s.device_span().at(2), 0},
          {"prefill GEMM, W off 16 bytes", A.device_span(), W4s.device_span().at(4), 0},
      };
      for (const auto& c : cases) {
        array Cs = dev(array::zeros({M * N + 1}));
        check_misaligned(
            c.what,
            [&] { return gpu::gemm_bf16_nt(c.a, c.w, Cs.device_span().at(4 * c.c_lead), M, N, K); },
            [&] { return host_close(after_lead(Cs, c.c_lead, M * N), want, 1e-4f); }, true);
      }
      // A product past one 64-row band, and K not a multiple of 8 (Metal's
      // vectorized STEEL loader takes whole 8-element vectors of W's rows,
      // its reader the rest): both raggedly against the tiles as above.
      for (const int64_t k2 : {136, 76}) {
        const int64_t M2 = 130, N2 = 200;
        array W2 = random_array({N2, k2}, 943);
        array W2b = dev(W2.to_bf16()), W2t = dev(W2.to_bf16().to_f32().transpose());
        array A2 = dev(random_array({M2, k2}, 944).to_bf16().to_f32());
        array C2 = array::empty({M2, N2});
        REQUIRE(gpu::gemm_bf16_nt(A2.device_span(), W2b.device_span(), C2.device_span(), M2, N2,
                                  k2));
        tl::gpu::flush();
        CHECK(same(C2, A2.dot(W2t), 1e-4f));
        // The f16-sum form (below) past one band too (K = 76 isn't whole
        // vectors: Metal declines it, and this is the f32 fallback). Its
        // 136-term sums, up to 16 in size, round by up to about 0.06 in f16;
        // a misplaced tile would be off by the products themselves.
        array C2h = array::empty({M2, N2});
        REQUIRE(gpu::gemm_bf16_nt_f16acc(A2.device_span(), W2b.device_span(), C2h.device_span(),
                                         M2, N2, k2));
        tl::gpu::flush();
        CHECK(same(C2h, A2.dot(W2t), 0.25f));
      }
      // The f16-sum form: the same exact inputs (bf16 values sit inside f16),
      // so only the f16 sums' rounding is left. A backend without the form
      // sums in f32 and passes trivially.
      array C16 = array::empty({M, N});
      REQUIRE(gpu::gemm_bf16_nt_f16acc(A.device_span(), Wb.device_span(), C16.device_span(), M,
                                       N, K));
      tl::gpu::flush();
      CHECK(same(C16, A.dot(Wt), 1e-2f));
      // W off as above: the backend's f16 sums, or (where it declines) the
      // f32 GEMM the op falls back to, so never refused; within the f16
      // rounding of the f32 product either way.
      for (const auto& c : {cases[2], cases[3]}) {
        array C16s = array::empty({M, N});
        const std::string what = std::string("f16-sum ") + c.what;
        check_misaligned(
            what.c_str(),
            [&] {
              return gpu::gemm_bf16_nt_f16acc(A.device_span(), c.w, C16s.device_span(), M, N, K);
            },
            [&] { return host_close(host_values(C16s), want, 1e-2f); }, true);
      }
      // Row m of the GEMM is the GEMV of row m — the prefill and the decode
      // paths have to agree on the same weight, which is what makes a
      // prompt's last token continue into generation.
      if (gpu::caps::row_gemv) {
        array row = dev(A.slice(0, 7, 1).clone());
        array y1 = array::empty({1, N});
        REQUIRE(gpu::gemv_bf16_row(row.device_span(), Wb.device_span(), y1.device_span(), N, K));
        tl::gpu::flush();
        CHECK(same(y1, C.slice(0, 7, 1), 1e-4f));
      }
    }
  }

  SUBCASE("split_heads and merge_heads are inverses through the fused layout") {
    const int64_t T = 5, H = 18, D = 64, ld = H * D;
    array src = dev(random_array({T, ld}, 930)), bias = dev(random_array({H, D}, 931));
    array heads = array::empty({H, T, D}), back = array::empty({T, ld});
    REQUIRE(gpu::split_heads(src.device_span(), bias.device_span(), heads.device_span(), T, ld, 0,
                             H, D));
    REQUIRE(gpu::merge_heads(heads.device_span(), back.device_span(), T, H, D));
    tl::gpu::flush();
    // heads[h, t, :] = src[t, h*D:(h+1)*D] + bias[h]
    CHECK(same(heads, src.reshape({T, H, D}).transpose({1, 0, 2}) +
                          bias.reshape({H, 1, D}), 1e-6f));
    CHECK(same(back, src + bias.reshape({1, ld}), 1e-6f));
  }

  tl::device_ = prev;
}
