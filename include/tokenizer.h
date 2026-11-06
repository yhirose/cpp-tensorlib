#pragma once
// The tokenizers GGUF files name in tokenizer.ggml.model, loaded from the
// file's metadata (tokenizer.ggml.*):
//
//  - "gpt2": byte-level BPE, the family Qwen2, GPT-2 and their relatives share.
//    The pre-tokenizer regex the file names (tokenizer.ggml.pre: Qwen2's,
//    Qwen3.5's, Llama 3's; hand-rolled, no std::regex), the byte->unicode
//    symbol mapping and rank-ordered merges; token for token HF's fast
//    tokenizer (checked against Qwen2TokenizerFast) and llama.cpp's
//    llama-tokenize (Qwen3.5, Llama 3). Text that is not UTF-8 keeps its bytes,
//    where llama.cpp reads each bad byte as U+FFFD.
//  - "llama": SentencePiece BPE (Gemma, Llama 2). Spaces become U+2581, then
//    adjacent pieces merge highest score first where the pair is a token, as
//    llama.cpp's SPM tokenizer does (checked against its llama-tokenize); a
//    piece the vocabulary lacks falls back to its bytes' <0xXX> tokens.
//  - "plamo3": PLaMo 3's unigram (encode_plamo_): the cut of the text into
//    pieces whose scores sum highest, after hard cuts around its markers and
//    runs of one character, as PFN's tokenization_plamo.py and llama.cpp's
//    plamo2 tokenizer (checked against its llama-tokenize); a character no
//    piece covers is written as its bytes' <0xXX> tokens.
//
// Either way special tokens in the text (CONTROL and USER_DEFINED) are matched
// whole, and decode() gives back the raw UTF-8 (special tokens as their text).
//
// Header-only, C++17, stdlib only (gguf.h adds POSIX mmap).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gguf.h"
#include "tokenizer_data.h"

namespace tl {

class tokenizer {
 public:
  explicit tokenizer(const std::string& gguf_path) {
    gguf::model m(gguf_path);

    tokens_ = m.kv("tokenizer.ggml.tokens").as_str_array();
    token2id_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); i++)
      token2id_.emplace(tokens_[i], static_cast<int>(i));

    const auto& tt = m.kv("tokenizer.ggml.token_type");
    const std::string model = m.kv("tokenizer.ggml.model").as_str();
    if (model == "llama") kind_ = kind::spm;
    else if (model == "plamo3") kind_ = kind::plamo;
    else if (model != "gpt2") throw std::runtime_error("tokenizer: no tokenizer \"" + model + "\"");
    if (kind_ != kind::bpe) {  // both score pieces and fall back to <0xXX> tokens
      const auto& sc = m.kv("tokenizer.ggml.scores");
      scores_.reserve(sc.arr.size());
      for (const auto& v : sc.arr) scores_.push_back(static_cast<float>(v.f));
      if (scores_.size() != tokens_.size())
        throw std::runtime_error("tokenizer: one score a token");
      for (int b = 0; b < 256; b++) {
        char name[7];
        std::snprintf(name, sizeof name, "<0x%02X>", b);
        auto it = token2id_.find(name);
        byte_id_[b] = it == token2id_.end() ? -1 : it->second;
        if (byte_id_[b] >= 0) byte_of_[byte_id_[b]] = static_cast<uint8_t>(b);
      }
    }
    if (kind_ == kind::spm) {
      // SentencePiece's default is to prefix a space; Gemma says not to.
      add_space_prefix_ = !m.has("tokenizer.ggml.add_space_prefix") ||
                          m.kv("tokenizer.ggml.add_space_prefix").u != 0;
    } else if (kind_ == kind::plamo) {
      // The runs cut around: the file's (PFN's tokenizer_config), else 4 and 2 as
      // llama.cpp has them (its converter does not write the keys).
      for (auto [key, n] : {std::pair{"tokenizer.ggml.break_around_repeated_chars_threshold", &plamo_repeat_},
                            std::pair{"tokenizer.ggml.break_around_consecutive_spaces_threshold", &plamo_spaces_}})
        if (m.has(key)) *n = m.kv(key).as_u32();
      build_plamo_(tt);
    } else {
      pre_ = pre_rule_(m.has("tokenizer.ggml.pre") ? m.kv("tokenizer.ggml.pre").as_str() : "");
      auto merges = m.kv("tokenizer.ggml.merges").as_str_array();
      merge_rank_.reserve(merges.size() * 2);
      for (size_t i = 0; i < merges.size(); i++)
        merge_rank_.emplace(merges[i], static_cast<int>(i));
    }

    // Special tokens are matched atomically in encode: CONTROL (type 3, the chat
    // markers) and USER_DEFINED (type 4: HF "added tokens" that are not marked
    // special, e.g. Qwen3's <think> / </think>). Both are literal text the model
    // was trained to see as one id, so neither may fall through to BPE.
    // Indexed by first byte, longest first (Gemma has 6,415 of them).
    for (size_t i = 0; i < tt.arr.size() && i < tokens_.size(); i++)
      if ((tt.arr[i].i == 3 || tt.arr[i].i == 4) && !tokens_[i].empty())
        specials_[static_cast<uint8_t>(tokens_[i][0])].emplace_back(tokens_[i],
                                                                    static_cast<int>(i));
    for (auto& v : specials_)
      std::stable_sort(v.begin(), v.end(),
                       [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    eos_id_ = static_cast<int>(m.kv("tokenizer.ggml.eos_token_id").as_u32());
    // The tokens that end a generation, as llama.cpp's vocab gathers them: EOS,
    // the file's EOT and EOM if it names them, and the tokens a chat template
    // closes a turn with, found by their text whatever their type (Gemma's EOS
    // is <eos>, but its turns end in <end_of_turn>; Qwen's in <|im_end|>).
    eog_ids_.push_back(eos_id_);
    for (const char* key : {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"})
      if (m.has(key)) eog_ids_.push_back(static_cast<int>(m.kv(key).as_u32()));
    for (size_t i = 0; i < tokens_.size(); i++)
      for (const char* s : {"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>",
                            "<|endoftext|>", "<|eom_id|>", "<|end_of_text|>", "<EOT>",
                            "_<EOT>", "<end_of_utterance>"})
        if (tokens_[i] == s) eog_ids_.push_back(static_cast<int>(i));
    // Qwen3.5 has no BOS (the template never adds one); -1 marks "none".
    bos_id_ = m.has("tokenizer.ggml.bos_token_id")
                  ? static_cast<int>(m.kv("tokenizer.ggml.bos_token_id").as_u32())
                  : -1;

    for (uint32_t b = 0; b < 256; b++)
      cp2byte_.emplace(tokdata::byte2cp[b], static_cast<uint8_t>(b));
  }

  int eos_id() const { return eos_id_; }
  // Whether `id` ends a generation (EOS, EOT, or a turn's closing token).
  bool is_eog(int id) const {
    return std::find(eog_ids_.begin(), eog_ids_.end(), id) != eog_ids_.end();
  }
  int bos_id() const { return bos_id_; }

  // Encode raw UTF-8 text to token ids. Special-token strings embedded in the
  // text (e.g. "<|im_start|>") are matched atomically; no BOS/EOS are added.
  std::vector<int> encode(const std::string& text) const {
    std::vector<int> out;
    size_t seg_start = 0;  // start of the pending normal-text segment
    size_t i = 0;
    bool after_special = true;  // SentencePiece prefixes a space here (add_space_prefix)
    while (i < text.size()) {
      // Longest special-token match at position i, if any.
      int best_id = -1;
      size_t best_len = 0;
      for (const auto& sp : specials_[static_cast<uint8_t>(text[i])]) {
        if (text.compare(i, sp.first.size(), sp.first) == 0) {
          best_len = sp.first.size();
          best_id = sp.second;
          break;
        }
      }
      if (best_id >= 0) {
        encode_segment_(text.data() + seg_start, i - seg_start, after_special, out);
        out.push_back(best_id);
        after_special = true;
        i += best_len;
        seg_start = i;
      } else {
        i++;
      }
    }
    encode_segment_(text.data() + seg_start, text.size() - seg_start, after_special, out);
    return out;
  }

  // Decode ids back to raw UTF-8 bytes (inverse of the byte2cp mapping).
  std::string decode(const std::vector<int>& ids) const {
    if (kind_ != kind::bpe) return decode_pieces_(ids);
    std::string joined;
    for (int id : ids) {
      if (id < 0 || static_cast<size_t>(id) >= tokens_.size())
        throw std::runtime_error("tokenizer: id out of range");
      joined += tokens_[static_cast<size_t>(id)];
    }
    std::string out;
    out.reserve(joined.size());
    size_t p = 0;
    while (p < joined.size()) {
      uint32_t cp = next_cp_(joined, p);
      auto it = cp2byte_.find(cp);
      if (it == cp2byte_.end())
        throw std::runtime_error("tokenizer: codepoint not in byte map");
      out.push_back(static_cast<char>(it->second));
    }
    return out;
  }

 private:
  // ---- UTF-8 helpers --------------------------------------------------------
  // Decode the codepoint starting at s[p], advancing p. Assumes valid UTF-8
  // (vocab strings and user text; malformed trailing bytes just pass through
  // as their own "codepoint" which the byte-level path never produces).
  static uint32_t next_cp_(const std::string& s, size_t& p) {
    uint8_t c0 = static_cast<uint8_t>(s[p]);
    if (c0 < 0x80) { p += 1; return c0; }
    if ((c0 & 0xE0) == 0xC0 && p + 1 < s.size()) {
      uint32_t cp = (c0 & 0x1Fu) << 6 | (static_cast<uint8_t>(s[p + 1]) & 0x3Fu);
      p += 2; return cp;
    }
    if ((c0 & 0xF0) == 0xE0 && p + 2 < s.size()) {
      uint32_t cp = (c0 & 0x0Fu) << 12 |
                    (static_cast<uint8_t>(s[p + 1]) & 0x3Fu) << 6 |
                    (static_cast<uint8_t>(s[p + 2]) & 0x3Fu);
      p += 3; return cp;
    }
    if ((c0 & 0xF8) == 0xF0 && p + 3 < s.size()) {
      uint32_t cp = (c0 & 0x07u) << 18 |
                    (static_cast<uint8_t>(s[p + 1]) & 0x3Fu) << 12 |
                    (static_cast<uint8_t>(s[p + 2]) & 0x3Fu) << 6 |
                    (static_cast<uint8_t>(s[p + 3]) & 0x3Fu);
      p += 4; return cp;
    }
    p += 1;  // malformed byte: treat as one unit
    return c0;
  }

  static void append_cp_(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
      s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  // ---- encode pipeline for one normal-text segment --------------------------
  // One run of text between special tokens; after_special: it follows one
  // (or opens the text), where SentencePiece may prefix a space.
  void encode_segment_(const char* data, size_t len, bool after_special,
                       std::vector<int>& out) const {
    if (len == 0) return;
    if (kind_ == kind::spm) return encode_spm_(data, len, after_special, out);
    if (kind_ == kind::plamo) return encode_plamo_(data, len, out);
    std::string seg(data, len);

    // Decode to codepoints, keeping each codepoint's byte offset (off[k] is the
    // byte start of cp[k]; off[n] == len so a pretoken [i,e) spans bytes
    // [off[i], off[e])).
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    size_t p = 0;
    while (p < seg.size()) {
      off.push_back(p);
      cp.push_back(next_cp_(seg, p));
    }
    off.push_back(seg.size());

    for (auto [i, e] : pretokenize_(cp, pre_))
      bpe_(data + off[i], off[e] - off[i], out);
  }

  // The byte-level BPE's pre-tokenizers, as llama.cpp's llama_vocab::load maps
  // the names (tokenizer.ggml.pre) and llm_tokenizer_bpe spells the regexes. The
  // ones here share Qwen2's shape,
  //   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+
  //   |\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  // and differ in the digit run, the classes and one rule of the merges:
  struct pretok_rule {
    size_t digits = 1;           // \p{N}{1,digits}: Llama 3 takes three at a time
    bool marks = false;          // [\p{L}\p{M}]+ words, \p{M} not punctuation (Qwen3.5)
    bool ignore_merges = false;  // a pretoken the vocabulary has is that token (Llama 3)
  };
  static pretok_rule pre_rule_(const std::string& pre) {
    for (const char* n : {"qwen2", "deepseek-r1-qwen", "kormo", "megrez"})
      if (pre == n) return {};
    if (pre == "qwen35") return {1, true, false};
    for (const char* n : {"llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral",
                          "midm-2.0", "lfm2", "jina-v5-nano"})
      if (pre == n) return {3, false, true};
    throw std::runtime_error("tokenizer: no BPE pre-tokenizer \"" + pre + "\"");
  }

  // The pre-tokenizer regex (pretok_rule), hand-rolled. Returns [start,end)
  // codepoint spans. At each position the FIRST matching alternative wins.
  static std::vector<std::pair<size_t, size_t>> pretokenize_(
      const std::vector<uint32_t>& cp, const pretok_rule& rule) {
    using tokdata::is_L;
    using tokdata::is_N;
    using tokdata::is_WS;
    auto word = [&](uint32_t c) { return is_L(c) || (rule.marks && tokdata::is_M(c)); };
    auto punct = [&](uint32_t c) { return !is_WS(c) && !word(c) && !is_N(c); };
    auto lower = [](uint32_t c) -> uint32_t {
      return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    };

    std::vector<std::pair<size_t, size_t>> toks;
    const size_t n = cp.size();
    size_t i = 0;
    while (i < n) {
      size_t e = 0;

      // A1 — contraction: '(?:[sdmt]|ll|ve|re) (ASCII-case-insensitive).
      if (cp[i] == 0x27) {
        if (i + 1 < n) {
          uint32_t c1 = lower(cp[i + 1]);
          if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
            e = i + 2;
          } else if (i + 2 < n) {
            uint32_t c2 = lower(cp[i + 2]);
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                (c1 == 'l' && c2 == 'l'))
              e = i + 3;
          }
        }
      }

      // A2 — word: [^\r\n\p{L}\p{N}]?\p{L}+ (Qwen3.5: [\p{L}\p{M}]+). The prefix
      // counts only when a word follows; else the regex backtracks to none, so a
      // mark (in both classes) that no word follows is a word by itself.
      if (!e) {
        size_t j = static_cast<size_t>(-1);
        if (cp[i] != 13 && cp[i] != 10 && !is_L(cp[i]) && !is_N(cp[i]) && i + 1 < n &&
            word(cp[i + 1])) {
          j = i + 1;  // optional prefix used
        } else if (word(cp[i])) {
          j = i;
        }
        if (j != static_cast<size_t>(-1)) {
          while (j < n && word(cp[j])) j++;
          e = j;
        }
      }

      // A3 — digits: \p{N}{1,rule.digits}
      if (!e && is_N(cp[i])) {
        e = i + 1;
        while (e < n && e - i < rule.digits && is_N(cp[e])) e++;
      }

      // A4 — punct run: ?[^\s\p{L}\p{N}]+[\r\n]* (Qwen3.5: and not \p{M})
      if (!e) {
        size_t j = i;
        bool ok = true;
        if (cp[i] == 32) {
          if (i + 1 < n && punct(cp[i + 1])) j = i + 1; else ok = false;
        }
        if (ok && punct(cp[j])) {
          while (j < n && punct(cp[j])) j++;
          while (j < n && (cp[j] == 13 || cp[j] == 10)) j++;
          e = j;
        }
      }

      // A5/A6/A7 — whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
      if (!e) {
        size_t w = i;
        while (w < n && is_WS(cp[w])) w++;
        size_t last_nl = static_cast<size_t>(-1);
        for (size_t k = i; k < w; k++)
          if (cp[k] == 13 || cp[k] == 10) last_nl = k;
        if (last_nl != static_cast<size_t>(-1)) e = last_nl + 1;  // A5
        else if (w == n) e = w;                                   // A6 (to end)
        else if (w - 1 > i) e = w - 1;  // A6, leave last ws char for A7
        else e = w;                     // A7 single ws char
      }

      // Every codepoint starts some alternative's match (a word's A2, a digit's
      // A3, whitespace's A5-A7, the rest A4), so each step moves on.
      if (e <= i || e > n)
        throw std::logic_error("tokenizer: a pre-tokenizer step matched nothing");
      toks.emplace_back(i, e);
      i = e;
    }
    return toks;
  }

  // Byte->symbol mapping + BPE merge for one pretoken's raw bytes.
  void bpe_(const char* bytes, size_t nbytes, std::vector<int>& out) const {
    std::vector<std::string> sym;
    sym.reserve(nbytes);
    for (size_t k = 0; k < nbytes; k++) {
      std::string s;
      append_cp_(s, tokdata::byte2cp[static_cast<uint8_t>(bytes[k])]);
      sym.push_back(std::move(s));
    }
    if (pre_.ignore_merges && sym.size() > 1) {
      std::string whole;
      for (const auto& s : sym) whole += s;
      if (auto it = token2id_.find(whole); it != token2id_.end()) {
        out.push_back(it->second);
        return;
      }
    }

    while (sym.size() > 1) {
      int best_rank = -1;
      size_t best_k = 0;
      std::string key;
      for (size_t k = 0; k + 1 < sym.size(); k++) {
        key.assign(sym[k]);
        key += ' ';
        key += sym[k + 1];
        auto it = merge_rank_.find(key);
        if (it != merge_rank_.end() &&
            (best_rank < 0 || it->second < best_rank)) {
          best_rank = it->second;
          best_k = k;
        }
      }
      if (best_rank < 0) break;
      sym[best_k] += sym[best_k + 1];
      sym.erase(sym.begin() + static_cast<ptrdiff_t>(best_k) + 1);
    }

    for (const auto& s : sym) {
      auto it = token2id_.find(s);
      if (it == token2id_.end())
        throw std::runtime_error("tokenizer: symbol not in vocab: " + s);
      out.push_back(it->second);
    }
  }

  // ---- SentencePiece BPE (tokenizer.ggml.model "llama") ----------------------
  // Spaces as U+2581; one piece a UTF-8 character; then, until none is left,
  // the adjacent pair whose concatenation is the highest-scoring token (the
  // leftmost among equal scores) becomes one piece. A piece that is no token
  // is emitted as its bytes' <0xXX> tokens. llama.cpp's
  // llm_tokenizer_spm_session, which keeps a priority queue of candidate pairs.
  void encode_spm_(const char* data, size_t len, bool after_special,
                   std::vector<int>& out) const {
    std::string text;
    if (add_space_prefix_ && after_special) text = "\xe2\x96\x81";
    for (size_t k = 0; k < len; k++) {
      if (data[k] == ' ') text += "\xe2\x96\x81";
      else text += data[k];
    }
    struct piece {
      size_t at, n;
      int prev, next;
    };
    std::vector<piece> ps;
    for (size_t at = 0; at < text.size();) {
      const auto c = static_cast<uint8_t>(text[at]);
      // llama.cpp's unicode_len_utf8: by the lead byte, a stray continuation one byte.
      const size_t n = std::min<size_t>(text.size() - at, c < 0xC0 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4);
      ps.push_back({at, n, static_cast<int>(ps.size()) - 1, static_cast<int>(ps.size()) + 1});
      at += n;
    }
    if (ps.empty()) return;
    ps.back().next = -1;
    struct pair_ {
      int left, right;
      float score;
      size_t n;
    };
    // The top is the highest score, the leftmost among equals.
    auto lower = [](const pair_& a, const pair_& b) {
      return a.score < b.score || (a.score == b.score && a.left > b.left);
    };
    std::priority_queue<pair_, std::vector<pair_>, decltype(lower)> queue(lower);
    auto offer = [&](int l, int r) {
      if (l < 0 || r < 0) return;
      auto it = token2id_.find(text.substr(ps[l].at, ps[l].n + ps[r].n));
      if (it != token2id_.end()) queue.push({l, r, scores_[it->second], ps[l].n + ps[r].n});
    };
    for (size_t k = 1; k < ps.size(); k++) offer(static_cast<int>(k) - 1, static_cast<int>(k));
    while (!queue.empty()) {
      const pair_ b = queue.top();
      queue.pop();
      piece &l = ps[b.left], &r = ps[b.right];
      if (l.n == 0 || r.n == 0 || l.n + r.n != b.n) continue;  // a side merged since
      l.n += r.n;
      r.n = 0;
      l.next = r.next;
      if (r.next >= 0) ps[r.next].prev = b.left;
      offer(l.prev, b.left);
      offer(b.left, l.next);
    }
    for (int k = 0; k != -1; k = ps[k].next) {
      auto it = token2id_.find(text.substr(ps[k].at, ps[k].n));
      if (it != token2id_.end()) {
        out.push_back(it->second);
        continue;
      }
      for (size_t j = 0; j < ps[k].n; j++) {
        const int id = byte_id_[static_cast<uint8_t>(text[ps[k].at + j])];
        if (id < 0) throw std::runtime_error("tokenizer: no byte token for a piece");
        out.push_back(id);
      }
    }
  }
  // A <0xXX> token to its byte, any other to its text: SentencePiece's with U+2581
  // back to a space and a leading space the encoder prefixed dropped, PLaMo 3's as is.
  std::string decode_pieces_(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
      if (id < 0 || static_cast<size_t>(id) >= tokens_.size())
        throw std::runtime_error("tokenizer: id out of range");
      if (const auto b = byte_of_.find(id); b != byte_of_.end()) {
        out += static_cast<char>(b->second);
        continue;
      }
      const std::string& t = tokens_[id];
      if (kind_ != kind::spm) {
        out += t;
        continue;
      }
      for (size_t k = 0; k < t.size(); k++) {
        if (t.compare(k, 3, "\xe2\x96\x81") == 0) {
          out += ' ';
          k += 2;
        } else {
          out += t[k];
        }
      }
    }
    if (kind_ == kind::spm && add_space_prefix_ && !out.empty() && out[0] == ' ') out.erase(0, 1);
    return out;
  }

  // ---- PLaMo 3 (tokenizer.ggml.model "plamo3") ---------------------------------
  // A unigram over code points: of every way to cut a run of text into pieces
  // (tokens other than the <0xXX> bytes), the one whose scores sum highest, as
  // PFN's tokenization_plamo.py and llama.cpp's llm_tokenizer_plamo2 have it.
  // Scores are rounded to integers (score * 1e4); among equal sums the longer
  // piece wins at each position; a code point no piece covers costs 1e7 and is
  // written as its UTF-8 bytes' tokens (a candidate at every position, which
  // loses ties: it is taken even where a longer piece starts, when that piece
  // leads nowhere better). Before that, hard cuts: around each <|plamo:NAME|>
  // (NAME up to 64 code points, neither '|' nor white space, Python's \s),
  // around each run of plamo_repeat_ or more of one code point other than '\n'
  // and of plamo_spaces_ or more spaces, and at U+EE00 (the reference's
  // boundary piece, which no token holds, so a cut there is the same; dropped).
  static constexpr int32_t plamo_unknown_ = -10000000;  // the score of a byte fallback
  void build_plamo_(const gguf::metadata_value& tt) {
    for (int b = 0; b < 256; b++)
      if (byte_id_[b] < 0) throw std::runtime_error("tokenizer: a plamo3 vocabulary without <0xXX>");
    plamo_nodes_.push_back({-1, 0});
    for (size_t id = 0; id < tokens_.size(); id++) {
      if ((id < tt.arr.size() && tt.arr[id].i == 6) || tokens_[id].empty()) continue;  // BYTE
      const std::vector<uint32_t> cps = plamo_cps_(tokens_[id].data(), tokens_[id].size());
      if (std::find(cps.begin(), cps.end(), 0xEE00u) != cps.end())
        throw std::runtime_error("tokenizer: a plamo3 piece holds U+EE00, the cut mark");
      int32_t node = 0;
      for (uint32_t c : cps) {
        auto [it, added] = plamo_edges_.try_emplace(plamo_edge_(node, c),
                                                    static_cast<int32_t>(plamo_nodes_.size()));
        if (added) plamo_nodes_.push_back({-1, 0});
        node = it->second;
      }
      // A later token of the same text replaces an earlier one, as in both references.
      const float s = scores_[id];
      plamo_nodes_[node] = std::isfinite(s) ? plamo_node{static_cast<int32_t>(id),
                                                         static_cast<int32_t>(std::round(s * 1e4))}
                                            : plamo_node{-1, 0};
    }
  }
  static uint64_t plamo_edge_(int32_t node, uint32_t c) {
    return static_cast<uint64_t>(node) << 21 | c;
  }
  // UTF-8 to code points as llama.cpp's unicode_cpts_from_utf8 reads it: a byte
  // that does not start a well-formed, shortest sequence is one U+FFFD.
  static std::vector<uint32_t> plamo_cps_(const char* s, size_t n) {
    std::vector<uint32_t> out;
    out.reserve(n);
    for (size_t p = 0; p < n;) {
      const auto b = [&](size_t k) { return static_cast<uint8_t>(s[p + k]); };
      const size_t len = b(0) < 0x80 ? 1 : b(0) < 0xC0 ? 0 : b(0) < 0xE0 ? 2 : b(0) < 0xF0 ? 3 : b(0) < 0xF8 ? 4 : 0;
      uint32_t c = len == 1 ? b(0) : len ? b(0) & (0x7Fu >> len) : 0;
      bool ok = len && p + len <= n;
      for (size_t k = 1; ok && k < len; k++) {
        ok = (b(k) & 0xC0) == 0x80;
        c = c << 6 | (b(k) & 0x3Fu);
      }
      static constexpr uint32_t least[] = {0, 0, 0x80, 0x800, 0x10000};
      if (ok && (c < least[len] || c > 0x10FFFF)) ok = false;
      out.push_back(ok ? c : 0xFFFD);
      p += ok ? len : 1;
    }
    return out;
  }
  void encode_plamo_(const char* data, size_t len, std::vector<int>& out) const {
    const std::vector<uint32_t> cp = plamo_cps_(data, len);
    const size_t n = cp.size();
    std::vector<bool> cut(n + 1, false);
    // <|plamo:NAME|>
    static constexpr char prefix[] = "<|plamo:";
    const size_t pn = sizeof prefix - 1;
    for (size_t i = 0; i + pn <= n;) {
      if (!std::equal(prefix, prefix + pn, cp.begin() + static_cast<ptrdiff_t>(i))) {
        i++;
        continue;
      }
      size_t j = i + pn;
      while (j < n && j - (i + pn) < 64 && cp[j] != '|' && !tokdata::is_WS(cp[j])) j++;
      if (j + 1 < n && cp[j] == '|' && cp[j + 1] == '>') {
        cut[i] = cut[j + 2] = true;
        i = j + 2;
      } else {
        i++;
      }
    }
    // Runs of one code point, within the cuts above.
    for (size_t i = 0; i < n;) {
      size_t e = i + 1;
      while (e < n && cp[e] == cp[i] && !cut[e]) e++;
      if ((plamo_repeat_ && cp[i] != '\n' && e - i >= plamo_repeat_) ||
          (plamo_spaces_ && cp[i] == ' ' && e - i >= plamo_spaces_))
        cut[i] = cut[e] = true;
      i = e;
    }
    size_t start = 0;
    for (size_t e = 0; e <= n; e++) {
      const bool mark = e < n && cp[e] == 0xEE00;
      if (e == n || cut[e] || mark) {
        plamo_dp_(cp.data() + start, e - start, out);
        start = e + mark;
      }
    }
  }
  // The best cut of one run, from its end: cost[i] the least cost of cp[i, n),
  // the pieces at i tried longest first so a tie keeps the longer.
  void plamo_dp_(const uint32_t* cp, size_t n, std::vector<int>& out) const {
    if (n == 0) return;
    std::vector<int64_t> cost(n + 1, int64_t{1} << 60);
    std::vector<int32_t> len(n + 1, 0), id(n + 1, -1);
    cost[n] = 0;
    std::vector<std::pair<int32_t, int32_t>> at;  // (length, node) of the pieces at i
    for (size_t i = n; i-- > 0;) {
      at.clear();
      int32_t node = 0;
      for (size_t k = i; k < n; k++) {
        auto it = plamo_edges_.find(plamo_edge_(node, cp[k]));
        if (it == plamo_edges_.end()) break;
        node = it->second;
        if (plamo_nodes_[node].id >= 0) at.emplace_back(static_cast<int32_t>(k + 1 - i), node);
      }
      for (auto p = at.rbegin(); p != at.rend(); ++p) {
        const int64_t s = cost[i + p->first] - plamo_nodes_[p->second].score;
        if (s < cost[i]) cost[i] = s, len[i] = p->first, id[i] = plamo_nodes_[p->second].id;
      }
      if (cost[i + 1] - plamo_unknown_ < cost[i]) cost[i] = cost[i + 1] - plamo_unknown_, len[i] = 1, id[i] = -1;
    }
    for (size_t i = 0; i < n; i += static_cast<size_t>(len[i])) {
      if (id[i] >= 0) {
        out.push_back(id[i]);
        continue;
      }
      std::string bytes;
      append_cp_(bytes, cp[i]);
      for (char b : bytes) out.push_back(byte_id_[static_cast<uint8_t>(b)]);
    }
  }
  std::vector<std::string> tokens_;                    // id -> vocab string
  std::unordered_map<std::string, int> token2id_;      // vocab string -> id
  std::unordered_map<std::string, int> merge_rank_;    // "A B" -> rank (gpt2)
  // CONTROL + USER_DEFINED tokens by first byte, longest first.
  std::vector<std::pair<std::string, int>> specials_[256];
  std::unordered_map<uint32_t, uint8_t> cp2byte_;      // inverse byte2cp (gpt2)
  pretok_rule pre_;                                    // (gpt2)
  enum class kind { bpe, spm, plamo };                 // tokenizer.ggml.model gpt2, llama, plamo3
  kind kind_ = kind::bpe;
  std::vector<float> scores_;                          // a token's score (spm, plamo)
  bool add_space_prefix_ = true;                       // (spm)
  int byte_id_[256] = {};                              // <0xXX>'s id (spm, plamo)
  std::unordered_map<int, uint8_t> byte_of_;           // and back (spm, plamo)
  // (plamo) The pieces' trie: a node's edge by code point, and the token that
  // ends at the node (-1: none) with its score * 1e4.
  struct plamo_node {
    int32_t id, score;
  };
  std::vector<plamo_node> plamo_nodes_;
  std::unordered_map<uint64_t, int32_t> plamo_edges_;
  size_t plamo_repeat_ = 4, plamo_spaces_ = 2;  // (plamo) the runs cut around; 0: none
  int eos_id_ = -1;
  std::vector<int> eog_ids_;  // is_eog's ids
  int bos_id_ = -1;
};

}  // namespace tl
