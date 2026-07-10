#pragma once
// Auto-generated from HF Qwen2.5-0.5B-Instruct tokenizer. {text, expected ids}.
#include <cstdint>
#include <string>
#include <vector>
namespace tokoracle {
struct Case { std::string text; std::vector<int> ids; };
inline std::vector<Case> cases() { return {
  { std::string(R"TOK(Hello, world!)TOK"), {9707,11,1879,0} },
  { std::string(R"TOK(The quick brown fox jumps over the lazy dog.)TOK"), {785,3974,13876,38835,34208,916,279,15678,5562,13} },
  { std::string(R"TOK(日本語のテスト)TOK"), {101059,102819,15767,140592} },
  { std::string(R"TOK(123  456
789)TOK"), {16,17,18,220,220,19,20,21,198,22,23,24} },
  { std::string(R"TOK(  leading spaces and	tab)TOK"), {220,6388,12621,323,58149} },
  { std::string(R"TOK(Give me a short introduction to large language models.)TOK"), {35127,752,264,2805,16800,311,3460,4128,4119,13} },
  { std::string(R"TOK(<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Give me a short introduction to large language models.<|im_end|>
<|im_start|>assistant
)TOK"), {151644,8948,198,2610,525,1207,16948,11,3465,553,54364,14817,13,1446,525,264,10950,17847,13,151645,198,151644,872,198,35127,752,264,2805,16800,311,3460,4128,4119,13,151645,198,151644,77091,198} },
};}
}
