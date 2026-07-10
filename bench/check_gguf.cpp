// GGUF v3 reader check: parse a real model file (Qwen2.5-0.5B-Instruct fp16 by
// default, argv[1] overrides), print the header + key hyperparameters + the
// first slice of the tensor manifest, and hard-assert the known ground truth
// (as produced by the reference gguf-py). Pure host code — no GPU involved.
// Returns nonzero on any mismatch so it can gate CI (where the model exists).

#include <gguf.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

template <typename A, typename B>
void check(const char* what, const A& got, const B& want) {
  if (!(got == static_cast<A>(want))) {
    std::printf("FAIL %s\n", what);
    failures++;
  }
}

const char* type_name(tl::gguf::ggml_type t) {
  using gt = tl::gguf::ggml_type;
  switch (t) {
    case gt::f32: return "F32";
    case gt::f16: return "F16";
    case gt::q4_0: return "Q4_0";
    case gt::q4_1: return "Q4_1";
    case gt::q5_0: return "Q5_0";
    case gt::q5_1: return "Q5_1";
    case gt::q8_0: return "Q8_0";
    case gt::q8_1: return "Q8_1";
    case gt::q2_k: return "Q2_K";
    case gt::q3_k: return "Q3_K";
    case gt::q4_k: return "Q4_K";
    case gt::q5_k: return "Q5_K";
    case gt::q6_k: return "Q6_K";
    case gt::q8_k: return "Q8_K";
  }
  return "?";
}

std::string dims_str(const std::vector<uint64_t>& d) {
  std::string s = "[";
  for (size_t i = 0; i < d.size(); i++)
    s += (i ? "," : "") + std::to_string(d[i]);
  return s + "]";
}

}  // namespace

int main(int argc, char** argv) {
  const char* path =
      argc > 1 ? argv[1] : "/home/yuji/models/qwen2.5-0.5b-instruct-fp16.gguf";

  tl::gguf::model m(path);

  std::printf("file        : %s\n", path);
  std::printf("version     : %u\n", m.version());
  std::printf("tensor_count: %" PRIu64 "\n", m.tensor_count());
  std::printf("alignment   : %" PRIu64 "\n", m.alignment());
  std::printf("kv_count    : %zu\n", m.metadata().size());

  const auto& arch = m.kv("general.architecture").as_str();
  uint32_t blocks = m.kv("qwen2.block_count").as_u32();
  uint32_t embd = m.kv("qwen2.embedding_length").as_u32();
  uint32_t heads = m.kv("qwen2.attention.head_count").as_u32();
  uint32_t heads_kv = m.kv("qwen2.attention.head_count_kv").as_u32();
  float freq_base = m.kv("qwen2.rope.freq_base").as_f32();
  const auto& tok_model = m.kv("tokenizer.ggml.model").as_str();
  const auto& tokens = m.kv("tokenizer.ggml.tokens");
  size_t n_tokens = tokens.arr.size();

  std::printf("architecture     : %s\n", arch.c_str());
  std::printf("block_count      : %u\n", blocks);
  std::printf("embedding_length : %u\n", embd);
  std::printf("head_count       : %u\n", heads);
  std::printf("head_count_kv    : %u\n", heads_kv);
  std::printf("rope.freq_base   : %.1f\n", static_cast<double>(freq_base));
  std::printf("tokenizer.model  : %s\n", tok_model.c_str());
  std::printf("tokens           : %zu strings\n", n_tokens);

  std::printf("\nfirst tensors:\n");
  const auto& ts = m.tensors();
  for (size_t i = 0; i < ts.size() && i < 14; i++) {
    const auto& t = ts[i];
    std::printf("  %-28s %-4s %-16s off=%-10" PRIu64 " nbytes=%" PRIu64 "\n",
                t.name.c_str(), type_name(t.type), dims_str(t.dims).c_str(),
                t.rel_offset, t.nbytes);
  }

  // ---- ground truth (reference gguf-py manifest of this exact file) --------
  check("version==3", m.version(), 3u);
  check("tensor_count==291", m.tensor_count(), 291u);
  check("alignment==32", m.alignment(), 32u);
  check("architecture==qwen2", arch, std::string("qwen2"));
  check("block_count==24", blocks, 24u);
  check("embedding_length==896", embd, 896u);
  check("head_count==14", heads, 14u);
  check("head_count_kv==2", heads_kv, 2u);
  check("rope.freq_base==1e6", freq_base, 1000000.0f);
  check("tokenizer.model==gpt2", tok_model, std::string("gpt2"));
  check("tokens len==151936", n_tokens, size_t{151936});
  // as_str_array must round-trip the same count of decoded strings.
  check("as_str_array len", tokens.as_str_array().size(), size_t{151936});

  const auto* emb = m.tensor("token_embd.weight");
  if (!emb) {
    std::printf("FAIL token_embd.weight missing\n");
    failures++;
  } else {
    check("token_embd dims", emb->dims,
          std::vector<uint64_t>{896, 151936});
    check("token_embd type==f16", emb->type, tl::gguf::ggml_type::f16);
    check("token_embd nbytes", emb->nbytes, uint64_t{896} * 151936 * 2);
    check("token_embd data set", emb->data != nullptr, true);
  }
  const auto* qb = m.tensor("blk.0.attn_q.bias");
  if (!qb) {
    std::printf("FAIL blk.0.attn_q.bias missing\n");
    failures++;
  } else {
    check("attn_q.bias dims", qb->dims, std::vector<uint64_t>{896});
    check("attn_q.bias type==f32", qb->type, tl::gguf::ggml_type::f32);
  }
  check("tensor(bogus)==nullptr", m.tensor("no.such.tensor") == nullptr, true);

  if (failures) {
    std::printf("\n%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("\nALL OK\n");
  return 0;
}
