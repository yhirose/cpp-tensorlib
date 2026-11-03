// llama.cpp's logits for the tokens a tensorlib_dump_logits file holds (its
// tokens/prompt and tokens/steps sections, bench/models/dump_logits.cpp), in that
// file's format, so tools/precision_logits.py holds them to the same f32
// reference: the prompt in llama_decode calls of 512 tokens, then the steps'
// tokens one at a time, every layer on the GPU and llama.cpp's other defaults (a
// context as long as the tokens). Sections llama/decoder/prompt and
// llama/decoder/stepN.
//
// Not part of the build (it links llama.cpp's):
//   L=path/to/llama.cpp
//   g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include tools/llama_logits.cpp \
//       -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o llama_logits
//   ./llama_logits model.gguf dump.bin llama.bin
#include <llama.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

// The sections of a dump: name -> floats.
static std::map<std::string, std::vector<float>> read_dump(const char* path) {
  std::map<std::string, std::vector<float>> out;
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return out;
  uint32_t len = 0;
  while (std::fread(&len, 4, 1, f) == 1) {
    std::string name(len, '\0');
    int64_t n = 0;
    if (std::fread(name.data(), 1, len, f) != len || std::fread(&n, 8, 1, f) != 1) break;
    std::vector<float> v((size_t)n);
    if (std::fread(v.data(), 4, (size_t)n, f) != (size_t)n) break;
    out[name] = std::move(v);
  }
  std::fclose(f);
  return out;
}

static std::FILE* out;
static void section(const std::string& name, const float* v, int64_t n) {
  const uint32_t len = (uint32_t)name.size();
  std::fwrite(&len, 4, 1, out);
  std::fwrite(name.data(), 1, len, out);
  std::fwrite(&n, 8, 1, out);
  std::fwrite(v, 4, (size_t)n, out);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s model.gguf dump.bin out.bin\n", argv[0]);
    return 2;
  }
  auto dump = read_dump(argv[2]);
  if (!dump.count("tokens/prompt") || !dump.count("tokens/steps")) {
    std::printf("%s has no tokens/prompt and tokens/steps\n", argv[2]);
    return 2;
  }
  std::vector<llama_token> prompt, steps;
  for (float t : dump["tokens/prompt"]) prompt.push_back((llama_token)t);
  for (float t : dump["tokens/steps"]) steps.push_back((llama_token)t);

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  llama_model* model = llama_model_load_from_file(argv[1], mp);
  if (!model) return 1;
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = (uint32_t)std::max<size_t>(1024, (prompt.size() + steps.size() + 255) / 256 * 256);
  cp.n_batch = 512;
  cp.n_ubatch = 512;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) return 1;
  const int64_t V = llama_vocab_n_tokens(llama_model_get_vocab(model));
  out = std::fopen(argv[3], "wb");
  if (!out) return 2;
  for (size_t at = 0; at < prompt.size(); at += cp.n_batch) {
    const int32_t n = (int32_t)std::min<size_t>(cp.n_batch, prompt.size() - at);
    if (llama_decode(ctx, llama_batch_get_one(prompt.data() + at, n))) return 1;
  }
  section("llama/decoder/prompt", llama_get_logits_ith(ctx, -1), V);
  for (size_t s = 0; s < steps.size(); s++) {
    if (llama_decode(ctx, llama_batch_get_one(&steps[s], 1))) return 1;
    section("llama/decoder/step" + std::to_string(s), llama_get_logits_ith(ctx, -1), V);
  }
  std::fclose(out);
  return 0;
}
