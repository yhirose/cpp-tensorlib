// The logits of every forward path on fixed prompts, written to one file: run it
// before and after a change and compare the two files byte for byte (cmp), so a
// change meant to leave the numbers alone is checked on all of them at once, and
// one meant to move them shows where.
//
//   tensorlib_dump_logits model.gguf out.bin [fixed]
//
// Some paths are not bit-reproducible: the GEMVs that split K add their partial
// sums with atomics, in no fixed order, so two runs of the same code differ in
// the last bits (tools/compare_logits.py holds them to that spread). `fixed`
// turns the split off (CUDA's set_no_splitk) so that every section is the same
// bytes run after run and a change that should leave the numbers alone can be
// held to identity; the split-K paths themselves are then not the ones run.
//
// Each build (the chat's weights: kept as blocks for a quantized file, else bf16;
// and the f32 one) runs the imperative step token by token, the captured decoder
// (its begin(), batched where the build allows, then step()), and the batched
// prefill in its fast and its exact form. A section is its name, its float count
// and the floats; the names and a checksum are also printed, so a difference can
// be found without a second tool.

#include "qwen.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace qm = qwenmodel;

namespace {

std::FILE* out = nullptr;

void section(const std::string& name, const float* v, int64_t n) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a over the bytes
  const auto* b = reinterpret_cast<const unsigned char*>(v);
  for (int64_t i = 0; i < n * 4; i++) h = (h ^ b[i]) * 1099511628211ull;
  std::printf("  %-36s %016llx\n", name.c_str(), (unsigned long long)h);
  const uint32_t len = (uint32_t)name.size();
  const int64_t count = n;
  std::fwrite(&len, 4, 1, out);
  std::fwrite(name.data(), 1, len, out);
  std::fwrite(&count, 8, 1, out);
  std::fwrite(v, 4, (size_t)n, out);
}

// Spread-out token ids, the same on every run.
std::vector<int> prompt(int64_t n, int64_t vocab) {
  std::vector<int> ids((size_t)n);
  for (int64_t i = 0; i < n; i++) ids[(size_t)i] = (int)(100 + (i * 7919 + 13) % (std::min<int64_t>(vocab, 30000) - 100));
  return ids;
}

void run(qm::Model& M, const std::string& tag) {
  const int64_t V = M.c.VOCAB, steps = 6;
  auto logits = [&] { return qm::host_read(M.scratch.logits); };
  // Token by token on the imperative step, then greedy steps.
  {
    qm::reset_cache(M);
    const std::vector<int> ids = prompt(8, V);
    int64_t pos = 0, tok = 0;
    for (int id : ids) tok = qm::step_imperative(M, id, pos++);
    section(tag + "/imperative/prompt", logits(), V);
    for (int64_t s = 0; s < steps; s++) {
      tok = qm::step_imperative(M, tok, pos++);
      section(tag + "/imperative/step" + std::to_string(s), logits(), V);
    }
  }
  // The chat's decoder over a prompt long enough to batch.
  {
    qm::reset_cache(M);
    qm::captured_decoder dec;
    int64_t tok = dec.begin(M, prompt(200, V));
    section(tag + (dec.batched ? "/decoder/batched" : "/decoder/tokens"), logits(), V);
    for (int64_t s = 0; s < steps; s++) {
      tok = dec.step(M, tok);
      section(tag + "/decoder/step" + std::to_string(s), logits(), V);
    }
    dec.destroy();
  }
  // The batched prefill alone, fast and exact.
  const std::vector<int> ids = prompt(300, V);
  if (qm::can_prefill_batched(M, (int64_t)ids.size())) {
    for (const bool exact : {false, true}) {
      qm::reset_cache(M);
      qm::prefill_batched(M, ids, qm::PREFILL_CHUNK, exact);
      section(tag + (exact ? "/prefill/exact" : "/prefill/fast"), logits(), V);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("usage: %s model.gguf out.bin\n", argv[0]);
    return 2;
  }
  if (!tl::gpu_available()) {
    std::printf("no GPU device\n");
    return 2;
  }
  qm::gg::model m(argv[1]);
  out = std::fopen(argv[2], "wb");
  if (!out) return 2;
  tl::use_gpu();
  if (argc > 3 && std::string(argv[3]) == "fixed") {
#ifdef TENSORLIB_CUDA
    tl::cuda::set_no_splitk(true);
#else
    std::printf("`fixed` is CUDA's (its split-K GEMVs); nothing to fix here\n");
#endif
  }
  {
    qm::Model M = qm::build(m, tl::dtype::bf16, false, false, 4096, /*oracle=*/false,
                             /*f16_acc=*/true, /*quant=*/qm::quant_file(m));
    run(M, M.quant ? "blocks" : "bf16");
  }
  {
    qm::Model M = qm::build(m, tl::dtype::f32, false, false, 4096, /*oracle=*/false);
    run(M, "f32");
  }
  std::fclose(out);
  return 0;
}
