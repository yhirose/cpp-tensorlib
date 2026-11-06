// The llm layer's tests (layers: include/gpu_abi.h): the KV cache and the decode
// step's kernels, the graph replay, the GGUF reader.

#include "test_quant.h"
#include "test_util.h"

#include <gguf.h>
#include <kv_cache.h>
#include <tokenizer.h>

// The sampler's definition (gpu_abi_llm.h), which the device samplers are held
// to: its random bits are Philox4x32-10's (Random123's known answers), and its
// draws follow the probabilities temperature, top_k and top_p leave.
TEST_CASE("sample_ref draws as temperature, top_k and top_p say") {
  namespace gpu = tl::gpu;
  const struct {
    uint32_t c[4], k[2], want[4];
  } kat[] = {
      {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}},
      {{0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
       {0xffffffff, 0xffffffff},
       {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd}},
      {{0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
       {0xa4093822, 0x299f31d0},
       {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1}},
  };
  for (const auto& t : kat) {
    uint32_t out[4];
    gpu::philox(t.c[0], t.c[1], t.c[2], t.c[3], t.k[0], t.k[1], out);
    CHECK(std::equal(out, out + 4, t.want));
  }

  // The open uniform the Gumbel noise takes stays inside (0, 1) at both ends.
  CHECK(gpu::sample_uniform(0xFFFFFFFFu, true) < 1.0f);
  CHECK(gpu::sample_uniform(0u, true) > 0.0f);

  // Each setting's share of 4000 draws over three tokens of probability
  // 0.5, 0.3, 0.2, against what it leaves them (within 0.03, about 4 sigma).
  const float l[] = {std::log(0.2f), std::log(0.5f), std::log(0.3f)};
  const struct {
    const char* what;
    gpu::sampling s;
    double want[3];
  } cases[] = {
      {"top_k 3", {1.0f, 3, 1.0f, 1}, {0.2, 0.5, 0.3}},
      {"top_p 0.75 keeps the two most likely", {1.0f, 3, 0.75f, 2}, {0.0, 0.625, 0.375}},
      {"top_k 2", {1.0f, 2, 1.0f, 3}, {0.0, 0.625, 0.375}},
      {"temperature 2 over the whole vocabulary", {2.0f, 0, 1.0f, 4},
       {0.4472 / 1.7020, 0.7071 / 1.7020, 0.5477 / 1.7020}},
  };
  for (const auto& c : cases) {
    INFO(c.what);
    int hits[3] = {0, 0, 0};
    for (uint32_t d = 0; d < 4000; d++) hits[gpu::sample_ref(l, 3, c.s, d)]++;
    for (int i = 0; i < 3; i++) CHECK(std::abs(hits[i] / 4000.0 - c.want[i]) < 0.03);
  }
  // Greedy, top_k 1 and a top_p below the largest probability are argmax;
  // NaN and -inf are never drawn.
  CHECK(gpu::sample_ref(l, 3, {0.0f, 0, 1.0f, 5}, 0) == 1);
  CHECK(gpu::sample_ref(l, 3, {1.0f, 1, 1.0f, 5}, 0) == 1);
  CHECK(gpu::sample_ref(l, 3, {1.0f, 3, 0.1f, 5}, 0) == 1);
  // Equal logits are ranked by index: the smaller holds the first half of
  // the draw's uniform.
  const float tie[] = {1.0f, 1.0f};
  for (uint32_t d = 0; d < 20; d++) {
    const float u = gpu::sample_uniform(gpu::philox(d, gpu::sample_pick_element, 0, 0, 5, 0));
    CHECK(gpu::sample_ref(tie, 2, {1.0f, 2, 1.0f, 5}, d) == (u < 0.5f ? 0 : 1));
  }
  // A NaN or a -inf logit is never drawn, and leaves the others' shares as
  // they are: two equal logits beside them are both drawn. Either end of the
  // vector, as a scan meets them first or last.
  const float odd[2][4] = {{NAN, 0.0f, -INFINITY, 0.0f}, {0.0f, -INFINITY, 0.0f, NAN}};
  for (const float* v : odd) {
    for (int top_k : {3, 0}) {
      CAPTURE(v[0]);
      CAPTURE(top_k);
      int hits[4] = {0, 0, 0, 0};
      for (uint32_t d = 0; d < 50; d++) hits[gpu::sample_ref(v, 4, {1.0f, top_k, 1.0f, 6}, d)]++;
      for (int i = 0; i < 4; i++) CHECK((hits[i] > 0) == (v[i] == 0.0f));
    }
  }
  // What a sampler refuses: top_k past the limit, top_p without top_k.
  CHECK(gpu::sampling{0.7f, 20, 0.8f, 0}.valid());
  CHECK_FALSE(gpu::sampling{0.7f, gpu::policy::max_top_k + 1, 1.0f, 0}.valid());
  CHECK_FALSE(gpu::sampling{0.7f, 0, 0.9f, 0}.valid());
  CHECK_FALSE(gpu::sampling{-1.0f, 20, 1.0f, 0}.valid());
  CHECK_FALSE(gpu::sampling{0.7f, 20, 0.0f, 0}.valid());
  CHECK_FALSE(gpu::sampling{1e-39f, 20, 1.0f, 0}.valid());  // 1 / t is inf
}

// A GGUF file's bytes as the tests below write them, little-endian as the format is.
struct gguf_bytes {
  std::string f;
  void put(const void* p, size_t n) { f.append(static_cast<const char*>(p), n); }
  void u32(uint32_t v) { put(&v, 4); }
  void u64(uint64_t v) { put(&v, 8); }
  void str(const std::string& s) { u64(s.size()), put(s.data(), s.size()); }
  void write(const std::string& path) const {
    std::ofstream(path, std::ios::binary).write(f.data(), (std::streamsize)f.size());
  }
};

// A BF16 tensor (ggml type 30, as BF16 GGUFs ship their weights) is read with
// its size and bytes as the file has them. The smallest GGUF v3 that holds
// one: no metadata, one 1-D tensor of four values, the data section aligned
// to the default 32.
TEST_CASE("gguf reads a BF16 tensor") {
  gguf_bytes g;
  g.put("GGUF", 4);
  g.u32(3);  // version
  g.u64(1);  // tensors
  g.u64(0);  // metadata pairs
  g.str("w");
  g.u32(1);   // dims
  g.u64(4);   // ne[0]
  g.u32(30);  // GGML_TYPE_BF16
  g.u64(0);   // offset in the data section
  g.f.resize((g.f.size() + 31) / 32 * 32, '\0');
  // Exact in bf16, the last near the top of its range.
  const float want[4] = {1.0f, -2.5f, 0.15625f, std::ldexp(1.0f, 127)};
  uint16_t raw[4];
  for (int i = 0; i < 4; i++) {
    uint32_t bits;
    std::memcpy(&bits, &want[i], 4);
    raw[i] = static_cast<uint16_t>(bits >> 16);
  }
  g.put(raw, sizeof(raw));
  const std::string path = "gguf_bf16_test.gguf";  // the test's working directory
  g.write(path);
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

// A byte-level BPE splits the text by the pre-tokenizer its file names
// (tokenizer.ggml.pre), and refuses one it does not have. A vocabulary of a few
// tokens, each rule's split telling in the ids: Llama 3 takes digits three at a
// time and a whole word the vocabulary has whatever the merges; Qwen3.5 keeps a
// combining mark (U+0301) in its word, where Qwen2 starts the next word at it,
// and makes a word of a mark no letter follows, where Qwen2's punctuation takes it.
TEST_CASE("the BPE pre-tokenizer is the one the file names") {
  // Bytes 0xCC 0x81 (U+0301) as their GPT-2 symbols, U+00CC and U+0123.
  const std::string cc = "\xC3\x8C", x81 = "\xC4\xA3";
  const std::vector<std::string> tokens = {"1",   "2", "3",  "4",     "12", "123",    "a",
                                           "b",   "ab", "<eos>", cc,  x81,  "a" + cc, ".",
                                           x81 + "."},
                                 merges = {"1 2", "12 3", "a " + cc, x81 + " ."};
  // GGUF v3 with no tensors: the tokenizer's metadata only.
  auto file = [&](const std::string& pre) {
    gguf_bytes g;
    auto strs = [&](const char* key, const std::vector<std::string>& v) {
      g.str(key), g.u32(9), g.u32(8), g.u64(v.size());  // array of strings
      for (const auto& s : v) g.str(s);
    };
    g.put("GGUF", 4), g.u32(3), g.u64(0), g.u64(6);  // version, tensors, metadata pairs
    g.str("tokenizer.ggml.model"), g.u32(8), g.str("gpt2");
    g.str("tokenizer.ggml.pre"), g.u32(8), g.str(pre);
    strs("tokenizer.ggml.tokens", tokens);
    strs("tokenizer.ggml.merges", merges);
    g.str("tokenizer.ggml.token_type"), g.u32(9), g.u32(5), g.u64(tokens.size());  // i32s
    for (size_t i = 0; i < tokens.size(); i++) g.u32(tokens[i] == "<eos>" ? 3 : 1);
    g.str("tokenizer.ggml.eos_token_id"), g.u32(4), g.u32(9);
    const std::string path = "tokenizer_pre_test.gguf";  // the test's working directory
    g.write(path);
    return path;
  };
  auto ids = [&](const std::string& pre, const std::string& text) {
    const std::string path = file(pre);
    const std::vector<int> v = tl::tokenizer(path).encode(text);
    std::remove(path.c_str());
    return v;
  };
  using v = std::vector<int>;
  CHECK(ids("qwen2", "1234") == v{0, 1, 2, 3});
  CHECK(ids("llama-bpe", "1234") == v{5, 3});
  CHECK(ids("qwen2", "ab") == v{6, 7});
  CHECK(ids("llama-bpe", "ab") == v{8});
  CHECK(ids("qwen2", "a\xCC\x81" "b") == v{6, 10, 11, 7});
  CHECK(ids("qwen35", "a\xCC\x81" "b") == v{12, 11, 7});
  CHECK(ids("qwen2", "1\xCC\x81.") == v{0, 10, 14});
  CHECK(ids("qwen35", "1\xCC\x81.") == v{0, 10, 11, 13});
  const std::string path = file("not-a-pre-tokenizer");
  CHECK_THROWS_AS(tl::tokenizer{path}, std::runtime_error);
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

// A model's weights sit in a device-only buffer, so they are in memory once:
// kernels read what upload_bytes wrote, and the host holds no pointer to it.
TEST_CASE("a device-only buffer: kernels read what upload_bytes wrote, the host cannot") {
  namespace gpu = tl::gpu;
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;
  tl::use_gpu();
  tl::storage w;
  try {
    w = tl::storage::make_device(4, 16, tl::dtype::f32);
  } catch (const std::runtime_error&) {
    MESSAGE("this backend keeps no device-only buffers");
    tl::device_ = prev;
    return;
  }
  CHECK(w.ptr == nullptr);
  const float vals[4] = {1, 2, 3, 4}, ones[4] = {1, 1, 1, 1};
  gpu::upload_bytes(w.native, 0, vals, sizeof vals);
  tl::storage one = tl::storage::make(4), out = tl::storage::make(4);
  gpu::upload(one.native, ones, 4);
  REQUIRE(gpu::binary(gpu::kop::add, w.device_span(), one.device_span(), out.device_span(), 4, 1, 0));
  gpu::cpu_barrier();
  gpu::sync_to_host(out.native, false);
  for (int i = 0; i < 4; i++) CHECK(out.ptr[i] == vals[i] + 1);
#if defined(TENSORLIB_CUDA) && !defined(__APPLE__) && !defined(TENSORLIB_HOST_GPU)
  // A mirrored backend has no host copy to bring the bytes into.
  CHECK_THROWS_AS(gpu::sync_to_host(w.native, false), std::logic_error);
#endif
  // Released, its buffer must not come back to an alloc that promises a host
  // pointer (the pools key by size alone).
  const int64_t bytes = 256;
  w = tl::storage::make_device(bytes / 4, bytes, tl::dtype::f32);
  w = tl::storage{};
  tl::storage s = tl::storage::make(bytes / 4);
  REQUIRE(s.ptr != nullptr);
  s.ptr[0] = 5.0f;
  CHECK(s.ptr[0] == 5.0f);
  tl::device_ = prev;
}

TEST_CASE("the KV cache and the decode step's kernels match their array forms") {
  // The model path runs on raw device buffers through gpu::, outside the lazy
  // graph; each kernel here is checked against the array composition that
  // defines it. Shapes are Qwen2's head geometry (D=64, 14 q heads over 2 kv
  // heads) at a context past the split-KV cutoff, and the other head widths
  // (128, 256, 512). A backend without a kernel takes the generic composition,
  // which has to agree too; a bf16 cache needs the backend's own attention.
  if (!tl::gpu_available()) return;
  auto prev = tl::device_;
  tl::use_gpu();
  namespace gpu = tl::gpu;

  // A device buffer from host values, evaluated so native() is valid.

  SUBCASE("kv_cache: append + attn, prefill, in f32 and f16") {
    // K and V as an f16 cache holds them (the reference reads those values).
    auto held16 = [](std::vector<float> v) {
      for (float& x : v) x = tl::f16_to_f32(tl::f32_to_f16(x));
      return v;
    };
    for (int64_t D : {64, 128, 256, 512}) {
      for (const bool f16 : {false, true}) {
        if (f16 && !gpu::caps::kv_f16) continue;
        CAPTURE(D);
        CAPTURE(f16);
        const int64_t HQ = 14, HKV = 2, MAXC = 300, T = 260;
        const float scale = 1.0f / std::sqrt((float)D);
        tl::kv_cache cache;
        REQUIRE(cache.init(HKV, MAXC, D, f16));
        // T steps appended one at a time, then a query over all of them. K and V
        // go up once in step order ([T,HKV,D]), so step t's rows are one span.
        auto K = random_array({HKV, T, D}, 900), V = random_array({HKV, T, D}, 901);
        array Kt = dev(K.transpose({1, 0, 2}).clone()), Vt = dev(V.transpose({1, 0, 2}).clone());
        for (int64_t t = 0; t < T; t++) {
          const int64_t at = t * HKV * D * 4;
          REQUIRE(cache.append(Kt.device_span().at(at), Vt.device_span().at(at)));
        }
        CHECK(cache.pos == T);
        array q = dev(random_array({HQ, D}, 902));
        array out = array::empty({HQ, D});
        REQUIRE(cache.attn(q.device_span(), out.device_span(), HQ, scale));
        // Explicit: each q head attends its kv head's T rows.
        const std::vector<float> qh = host_values(q),
                                 Kr = f16 ? held16(host_values(K)) : host_values(K),
                                 Vr = f16 ? held16(host_values(V)) : host_values(V);
        std::vector<float> want((size_t)HQ * D);
        for (int64_t h = 0; h < HQ; h++) {
          const int64_t kh = h / (HQ / HKV);
          std::vector<double> s(T);
          double m = -1e300, sum = 0;
          for (int64_t j = 0; j < T; j++) {
            double a = 0;
            for (int64_t d = 0; d < D; d++)
              a += (double)qh[(size_t)(h * D + d)] * Kr[(size_t)((kh * T + j) * D + d)];
            s[j] = a * scale;
            m = std::max(m, s[j]);
          }
          for (auto& x : s) sum += (x = std::exp(x - m));
          for (int64_t d = 0; d < D; d++) {
            double e = 0;
            for (int64_t j = 0; j < T; j++) e += s[j] * Vr[(size_t)((kh * T + j) * D + d)];
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

  SUBCASE("kv_cache: a sliding window attends only the last rows") {
    // The decode step at position T - 1 of a cache with a window of W attends the keys
    // [T - W, T) (gpu::attn_window_lo). Two keys of q head 0's kv head are planted to
    // dominate its scores: row lo - 1, which the window leaves out, and row lo, which
    // it keeps, so either off by one moves the output by O(1), not by noise. A window
    // of 100 runs in one block, one of 300 split.
    auto held16 = [](std::vector<float> v) {
      for (float& x : v) x = tl::f16_to_f32(tl::f32_to_f16(x));
      return v;
    };
    for (int64_t D : {64, 256}) {
      for (const bool f16 : {false, true}) {
        if (f16 && !gpu::caps::kv_f16) continue;
        for (const int64_t W : {100, 300}) {
          CAPTURE(D);
          CAPTURE(f16);
          CAPTURE(W);
          const int64_t HQ = 4, HKV = 2, MAXC = 600, T = 520, lo = T - W;
          const float scale = 1.0f / std::sqrt((float)D);
          array q = dev(random_array({HQ, D}, 920));
          std::vector<float> Kh = host_values(random_array({HKV, T, D}, 921)),
                             Vh = host_values(random_array({HKV, T, D}, 922));
          const std::vector<float> qh = host_values(q);
          double qq = 0;
          for (int64_t d = 0; d < D; d++) qq += (double)qh[d] * qh[d];
          for (const auto [row, s, v] : {std::tuple<int64_t, double, float>{lo - 1, 20.0, 5.0f},
                                         {lo, 8.0, -5.0f}}) {
            for (int64_t d = 0; d < D; d++) {  // kv head 0: score s against q head 0
              Kh[(size_t)(row * D + d)] = (float)(s / (scale * qq)) * qh[d];
              Vh[(size_t)(row * D + d)] = v;
            }
          }
          if (f16) {
            Kh = held16(Kh);
            Vh = held16(Vh);
          }
          tl::kv_cache cache;
          REQUIRE(cache.init(HKV, MAXC, D, f16, W));
          array K = array::from(std::vector<float>(Kh), {HKV, T, D}),
                V = array::from(std::vector<float>(Vh), {HKV, T, D});
          for (int64_t t = 0; t < T; t++) {
            array k = dev(K.slice(1, t, 1).clone()), v = dev(V.slice(1, t, 1).clone());
            REQUIRE(cache.append(k.device_span(), v.device_span()));
          }
          array out = array::empty({HQ, D});
          REQUIRE(cache.attn(q.device_span(), out.device_span(), HQ, scale));
          std::vector<float> want((size_t)HQ * D);
          for (int64_t h = 0; h < HQ; h++) {
            const int64_t kh = h / (HQ / HKV);
            std::vector<double> sc(T);
            double m = -1e300, sum = 0;
            for (int64_t j = lo; j < T; j++) {
              double a = 0;
              for (int64_t d = 0; d < D; d++)
                a += (double)qh[(size_t)(h * D + d)] * Kh[(size_t)((kh * T + j) * D + d)];
              sc[j] = a * scale;
              m = std::max(m, sc[j]);
            }
            for (int64_t j = lo; j < T; j++) sum += (sc[j] = std::exp(sc[j] - m));
            for (int64_t d = 0; d < D; d++) {
              double e = 0;
              for (int64_t j = lo; j < T; j++) e += sc[j] * Vh[(size_t)((kh * T + j) * D + d)];
              want[(size_t)(h * D + d)] = (float)(e / sum);
            }
          }
          tl::gpu::flush();
          CHECK(same(out, array::from(want, {HQ, D}), 1e-4f));

          // The same rows as one prefill into a fresh windowed cache (the exact form):
          // its last query, at position T - 1, attends the window the decode above did.
          tl::kv_cache c2;
          REQUIRE(c2.init(HKV, MAXC, D, f16, W));
          array qp = dev(random_array({HQ, T, D}, 923));
          array Kd = dev(K), Vd = dev(V), op = array::empty({HQ, T, D});
          REQUIRE(c2.prefill(qp.device_span(), Kd.device_span(), Vd.device_span(), op.device_span(),
                             T, HQ, scale, /*exact=*/true));
          array qt = dev(qp.slice(1, T - 1, 1).reshape({HQ, D}).clone());
          array ot = array::empty({HQ, D});
          REQUIRE(cache.attn(qt.device_span(), ot.device_span(), HQ, scale));
          tl::gpu::flush();
          CHECK(same(op.slice(1, T - 1, 1).reshape({HQ, D}), ot, 1e-4f));
        }
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
    REQUIRE(gpu::rope(x.device_span(), o.device_span(), H, 1, D, 37, {1e6f}, b.device_span()));
    tl::gpu::flush();
    CHECK(same(o, array::rope(x + b, 37, 1e6f), 1e-5f));
  }

  SUBCASE("rope reading its position from the device matches rope, scaled or not") {
    // A captured decode step reads pos from a device scalar; Gemma 3's global
    // layers scale it, Llama 3's factors divide its pairs' angles. Where the
    // backend has the form, the same rotation bit for bit.
    const int64_t H = 8, D = 256;
    array x = dev(random_array({H, D}, 922));
    const array fac = dev(array::from(rope_factors(D / 2), {D / 2}));
    array pos = dev(array::zeros({1}));
    const uint32_t p = 37;
    float bits;
    std::memcpy(&bits, &p, 4);
    tl::gpu::upload(pos.device_span().buf, &bits, 1);
    for (const auto [scale, factors] : {std::pair{1.0f, false}, {0.125f, false}, {0.125f, true}}) {
      CAPTURE(scale);
      CAPTURE(factors);
      if (factors && !gpu::caps::rope_factors) continue;
      const gpu::rope_freq rf{1e6f, 0, scale, factors ? fac.device_span() : gpu::span{}};
      array a = array::empty({H, D}), b = array::empty({H, D});
      REQUIRE(gpu::rope(x.device_span(), a.device_span(), H, 1, D, p, rf));
      if (gpu::rope_dpos(x.device_span(), b.device_span(), H, 1, D, pos.device_span(), rf)) {
        tl::gpu::flush();
        CHECK(same(a, b, 0.0f));
      }
    }
  }

  SUBCASE("argmax reads back the smallest index among ties, or leaves it on the device") {
    // A backend may split the vector across groups and reduce their winners:
    // ties within one stretch and across two, a maximum at the ragged end, and
    // a vector shorter than any stretch (all negative, so no pair a reduction
    // starts from may win).
    const struct {
      const char* what;
      int64_t n;
      std::vector<int64_t> at;
      float top;
      int64_t want;
    } cases[] = {
        {"tie side by side", 151936, {77777, 77778}, 5.0f, 77777},
        {"tie far apart", 151936, {150000, 100}, 5.0f, 100},
        {"maximum at the ragged end", 100003, {100002}, 5.0f, 100002},
        {"a short all-negative vector", 5, {3}, -0.5f, 3},
    };
    for (const auto& c : cases) {
      INFO(c.what);
      std::vector<float> v((size_t)c.n);
      for (int64_t i = 0; i < c.n; i++) v[(size_t)i] = -1.0f - (float)(i % 7);
      for (int64_t i : c.at) v[(size_t)i] = c.top;
      array a = dev(array::from(v, {c.n}));
      int64_t idx = -1;
      REQUIRE(gpu::argmax(a.device_span(), c.n, &idx));
      CHECK(idx == c.want);
      if (gpu::has_argmax_to) {  // the same index, left on the device as an f32
        array d = dev(array::zeros({1}));
        REQUIRE(gpu::argmax_to(a.device_span(), c.n, d.device_span()));
        CHECK(host_values(d)[0] == (float)c.want);
      }
    }
  }

  SUBCASE("sample_to draws sample_ref's token on the device, and counts its draws") {
    // A vocabulary-sized vector that groups split, its top two tied far apart
    // and the next two tied a group's width apart (one thread's stretch); a
    // vector long enough that the groups' keys take a second launch before
    // the one that draws; a short one drawn in one group, its top three tied
    // that far apart; one shorter than k that holds what is never drawn, and
    // one with nothing to draw. Each setting draws six times on one counter;
    // the token of each draw is the host definition's for that draw number.
    // A backend without the form draws on the host (the decoder's fallback).
    if (gpu::has_sample_to) {
      std::vector<float> big = host_values(random_array({151936}, 990));
      for (float& x : big) x *= 6.0f;
      big[1234] = big[150000] = 9.0f;
      big[777] = big[777 + 256] = 8.5f;
      std::vector<float> longer = host_values(random_array({300000}, 992));
      for (float& x : longer) x *= 4.0f;
      std::vector<float> one_group = host_values(random_array({600}, 991));
      one_group[5] = one_group[5 + 256] = one_group[5 + 512] = 3.0f;
      const std::vector<float> small = {-INFINITY, 2.0f, NAN, 1.0f, 1.5f},
                               none = {-INFINITY, NAN, -INFINITY};
      const struct {
        const char* what;
        const std::vector<float>* v;
        gpu::sampling s;
      } cases[] = {
          {"Qwen's setting", &big, {0.7f, 20, 0.8f, 7}},
          {"the widest k, no top_p", &big, {1.5f, 64, 1.0f, 8}},
          {"a narrow top_p", &big, {1.0f, 5, 0.5f, 9}},
          {"the whole vocabulary (Gumbel-max)", &big, {3.0f, 0, 1.0f, 10}},
          {"three launches", &longer, {1.0f, 64, 0.9f, 14}},
          {"ties in one group", &one_group, {1.0f, 3, 1.0f, 13}},
          {"fewer tokens than k", &small, {2.0f, 20, 1.0f, 11}},
          {"nothing to draw", &none, {1.0f, 20, 1.0f, 12}},
      };
      for (const auto& c : cases) {
        INFO(c.what);
        const auto n = static_cast<int64_t>(c.v->size());
        array a = dev(array::from(*c.v, {n})), ctr = dev(array::zeros({1})),
              d = dev(array::zeros({1}));
        for (uint32_t draw = 0; draw < 6; draw++) {
          CAPTURE(draw);
          REQUIRE(gpu::sample_to(a.device_span(), n, c.s, ctr.device_span(), d.device_span()));
          CHECK(host_values(d)[0] == (float)gpu::sample_ref(c.v->data(), n, c.s, draw));
        }
        uint32_t draws = 0;
        const float bits = host_values(ctr)[0];
        std::memcpy(&draws, &bits, 4);
        CHECK(draws == 6);
      }
    }
  }

  SUBCASE("read_later sees the work queued before it, not after") {
    // A decode step run ahead: the host queues the next step between
    // read_later and finish_read, and the value read is what the work before
    // read_later left, once that work has run (a long GEMM ahead of it, so a
    // read that does not wait sees the old value). Plainly and under a
    // profile, where a backend may commit each launch on its own.
    for (bool profiled : {false, true}) {
      CAPTURE(profiled);
      tl::gpu::flush();
      if (profiled) tl::profile::start();
      using tl::gpu::kop;
      const int64_t n = 384;
      array A = dev(random_array({n, n}, 995)), C = dev(array::zeros({n, n}));
      array zero = dev(array::zeros({1})), x = dev(array::from(std::vector<float>{3.0f}, {1})),
            y = dev(array::zeros({1}));
      auto gemm = [&] {
        return gpu::gemm(A.device_span(), n, false, A.device_span(), n, false, C.device_span(), n,
                         n, n, 1.0f, 0.0f);
      };
      REQUIRE(gemm());
      REQUIRE(gpu::unary(kop::affine, zero.device_span(), x.device_span(), 1, 1.0f, 7.0f));
      gpu::pending_read rx, ry;
      REQUIRE(gpu::read_later(x.device_span(), rx));
      REQUIRE(gemm());  // the next step, queued before x is read; it leaves x alone
      REQUIRE(gpu::unary(kop::affine, zero.device_span(), y.device_span(), 1, 1.0f, 9.0f));
      REQUIRE(gpu::read_later(y.device_span(), ry));
      float vx = 0.0f, vy = 0.0f;
      REQUIRE(gpu::finish_read(rx, &vx));
      REQUIRE(gpu::finish_read(ry, &vy));
      CHECK(vx == 7.0f);
      CHECK(vy == 9.0f);

      // src let go after read_later, its storage handed to the next step's
      // write, and the queue drained before finish_read: still the value read.
      gpu::pending_read rz;
      {
        array z = array::empty({1});
        REQUIRE(gpu::unary(kop::affine, zero.device_span(), z.device_span(), 1, 1.0f, 5.0f));
        REQUIRE(gpu::read_later(z.device_span(), rz));
      }
      array w = array::empty({1});
      REQUIRE(gpu::unary(kop::affine, zero.device_span(), w.device_span(), 1, 1.0f, 11.0f));
      gpu::pending_read dropped;  // a read never finished costs nothing
      REQUIRE(gpu::read_later(w.device_span(), dropped));
      tl::gpu::flush();
      float vz = 0.0f;
      REQUIRE(gpu::finish_read(rz, &vz));
      CHECK(vz == 5.0f);
      if (profiled) tl::profile::stop();
    }
  }

  SUBCASE("embed_row gathers the row an index on the device names") {
    // A decode step run ahead takes its input from the token argmax_to left: a
    // row of the embedding table in whichever form the model keeps it, [V, K]
    // or its transpose, f32, bf16 or GGUF blocks. Past the table, zeros.
    if (gpu::has_embed_row) {
      const int64_t V = 300, K = 256;
      const array T = random_array({V, K}, 990), Tb = T.to_bf16().to_f32();
      const std::vector<float> f = host_values(T), fb = host_values(Tb);
      const array Wf = dev(T), Wft = dev(T.transpose()), Wb = dev(T.to_bf16()),
                  Wbt = dev(T.transpose().to_bf16());
      // The row gathered.
      auto gather = [&](int type, const array& W, int64_t rs, int64_t cs, int64_t n, int64_t rows,
                        float scale = 1.0f) {
        array idx = dev(array::from(std::vector<float>{(float)n}, {1})), out = dev(array::zeros({K}));
        REQUIRE(gpu::embed_row(type, W.device_span(), idx.device_span(), out.device_span(), K,
                               rows, rs, cs, scale));
        return host_values(out);
      };
      auto row = [&](int type, const array& W, int64_t rs, int64_t cs, int64_t n) {
        return gather(type, W, rs, cs, n, V);
      };
      // A table of V - 1 rows: its index V - 1 is past it, and the row there
      // holds data (past the buffer itself a backend may read zeros anyway).
      auto past = [&](int type, const array& W, int64_t rs, int64_t cs) {
        return gather(type, W, rs, cs, V - 1, V - 1) == std::vector<float>((size_t)K, 0.0f);
      };
      auto want = [&](const std::vector<float>& w, int64_t n) {
        std::vector<float> r((size_t)K, 0.0f);
        if (n < V) std::copy(w.begin() + n * K, w.begin() + (n + 1) * K, r.begin());
        return r;
      };
      auto close = [](const std::vector<float>& got, const std::vector<float>& w) {
        for (size_t i = 0; i < w.size(); i++)
          if (!(std::fabs(got[i] - w[i]) <= 1e-6f * (1.0f + std::fabs(w[i])))) return false;
        return true;
      };
      std::vector<std::pair<int, quant_rows>> qs;
      if (gpu::has_gemv_quant)
        for (const block_format& f : block_formats)
          if (reads_quant(f.type, K))
            qs.push_back({f.type, random_quant_rows(f.type, V, K, 990 + f.type)});
      // Scaled (Gemma's embedding): each value times the scale, an f32 product,
      // required of every backend.
      auto scaled = [&](int type, const array& W, int64_t rs, int64_t cs, int64_t n,
                        const std::vector<float>& w, bool exact) {
        const float s = 50.596443f;
        const std::vector<float> got = gather(type, W, rs, cs, n, V, s);
        std::vector<float> r = want(w, n);
        for (float& v : r) v *= s;
        if (exact) CHECK(got == r);
        else CHECK(close(got, r));
      };
      for (int64_t n : {int64_t(0), int64_t(137), V - 1, V}) {
        INFO("row " << n);
        CHECK(row(0, Wf, K, 1, n) == want(f, n));
        scaled(0, Wf, K, 1, n, f, true);
        CHECK(row(0, Wft, 1, V, n) == want(f, n));
        if (gpu::caps::row_gemv) {  // a backend that decodes from bf16 weights
          CHECK(row(30, Wb, K, 1, n) == want(fb, n));
          CHECK(row(30, Wbt, 1, V, n) == want(fb, n));
        }
        for (const auto& [type, q] : qs) {
          INFO("type " << type);
          CHECK(close(row(type, dev(q.W), 0, 0, n), want(q.w, n)));
          scaled(type, dev(q.W), 0, 0, n, q.w, false);
        }
      }
      CHECK(past(0, Wf, K, 1));
      if (gpu::caps::row_gemv) CHECK(past(30, Wb, K, 1));
      for (const auto& [type, q] : qs) {
        INFO("type " << type);
        CHECK(past(type, dev(q.W), 0, 0));
      }
    }
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
      // The activations as they are, not rounded to bf16: the GEMM keeps them
      // f32 (CUDA's tensor cores as bf16 pairs, to ~2^-16), as the decode's
      // GEMVs do, so the batched prefill is the token-by-token one. Rounded,
      // they would be off by ~5e-3 here.
      array A = dev(random_array({M, K}, 942));
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
        array A2 = dev(random_array({M2, k2}, 944));
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
      // The f16-sum form: the activations rounded to f16 (2^-11) and the sums
      // f16, the second by far the larger. A backend without the form sums in
      // f32 and passes trivially.
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
