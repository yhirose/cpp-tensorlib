// The logits of every forward path on fixed tokens, written to one file: run it
// before and after a change and compare the two files byte for byte (cmp), so a
// change meant to leave the numbers alone is checked on all of them at once, and
// one meant to move them shows where.
//
//   tensorlib_dump_logits model.gguf out.bin [fixed]
//
// Some paths are not bit-reproducible: the GEMVs and GEMMs that split K add their
// partial sums with atomics, in no fixed order, so two runs of the same code
// differ in the last bits (tools/compare_logits.py holds them to that spread).
// `fixed` turns the split off (CUDA's set_no_splitk) so that every section is the
// same bytes run after run and a change that should leave the numbers alone can
// be held to identity; the split-K paths themselves are then not the ones run.
//
// The tokens come first (tokens/prompt, tokens/steps, as floats; tools/
// llama_logits.cpp reads them). Each build (the chat's weights: kept as blocks for
// a quantized file, else bf16; and the f32 one) runs the prompt, then the steps'
// tokens one at a time, through each path: the imperative step token by token,
// the captured decoder (its begin(), batched where the build allows, then step()),
// and the batched prefill in its exact form (begin()'s is the fast one). A section
// is named build/path/what, what being prompt (the logits after the prompt) or
// stepN (after the N+1-th step's token), so the same path/what in two builds is
// the same computation: the f32 build's imperative sections, f32 throughout, are
// the reference tools/precision_logits.py holds the others to. A section is its
// name, its float count and the floats; the names and a checksum are also
// printed, so a difference can be found without a second tool.

#include "qwen.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace qm = qwenmodel;

namespace {

std::FILE* out = nullptr;

void dump_section(const std::string& name, const float* v, int64_t n) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a over the bytes
  const auto* b = reinterpret_cast<const unsigned char*>(v);
  for (int64_t i = 0; i < n * 4; i++) h = (h ^ b[i]) * 1099511628211ull;
  std::printf("  %-36s %016llx\n", name.c_str(), (unsigned long long)h);
  const uint32_t len = (uint32_t)name.size();
  std::fwrite(&len, 4, 1, out);
  std::fwrite(name.data(), 1, len, out);
  std::fwrite(&n, 8, 1, out);
  std::fwrite(v, 4, (size_t)n, out);
}

// The prompt, then the tokens the steps feed.
constexpr int64_t kPrompt = 200, kSteps = 6;
std::vector<int> tokens(int64_t vocab) { return qm::spread_ids(kPrompt + kSteps, vocab); }

void run(qm::Model& M, const std::string& tag) {
  const int64_t V = M.c.VOCAB;
  auto logits = [&] { return qm::host_read(M.scratch.logits); };
  const std::vector<int> all = tokens(V), ids(all.begin(), all.begin() + kPrompt);
  // Token by token on the imperative step.
  {
    qm::reset_cache(M);
    int64_t pos = 0;
    const float* lg = nullptr;
    for (int id : ids) lg = qm::imperative_logits(M, id, pos++);
    dump_section(tag + "/imperative/prompt", lg, V);
    for (int64_t s = 0; s < kSteps; s++) {
      lg = qm::imperative_logits(M, all[(size_t)(kPrompt + s)], pos++);
      dump_section(tag + "/imperative/step" + std::to_string(s), lg, V);
    }
  }
  // The chat's decoder: the prompt through begin(), batched where it can.
  {
    qm::reset_cache(M);
    qm::captured_decoder dec;
    dec.begin(M, ids);
    std::printf("  (decoder prompt %s)\n", dec.batched ? "batched" : "token by token");
    dump_section(tag + "/decoder/prompt", logits(), V);
    for (int64_t s = 0; s < kSteps; s++) {
      dec.step(M, all[(size_t)(kPrompt + s)]);
      dump_section(tag + "/decoder/step" + std::to_string(s), logits(), V);
    }
    dec.destroy();
  }
  // The batched prefill in its exact form (f32 sums; begin() ran the fast one).
  if (qm::can_prefill_batched(M, (int64_t)ids.size())) {
    qm::reset_cache(M);
    qm::prefill_batched(M, ids, qm::PREFILL_CHUNK, /*exact=*/true);
    dump_section(tag + "/prefill_exact/prompt", logits(), V);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("usage: %s model.gguf out.bin [fixed]\n", argv[0]);
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
    std::printf("`fixed` is CUDA's (its split-K GEMVs and GEMMs); nothing to fix here\n");
#endif
  }
  {
    const std::vector<int> all = tokens(qm::config_from_gguf(m).VOCAB);
    const std::vector<float> f(all.begin(), all.end());
    dump_section("tokens/prompt", f.data(), kPrompt);
    dump_section("tokens/steps", f.data() + kPrompt, kSteps);
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
