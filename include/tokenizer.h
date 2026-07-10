#pragma once
// Qwen2 (GPT-2 byte-level BPE) tokenizer. Loads the vocab/merges/special
// tokens from a GGUF model's metadata (tokenizer.ggml.*) and implements the
// GPT-2 pretokenizer regex (hand-rolled — no std::regex), the byte->unicode
// symbol mapping, and greedy rank-ordered BPE merging. encode() reproduces
// HF Qwen2TokenizerFast token-for-token; decode() inverts it back to raw
// UTF-8 bytes (special tokens decode to their literal <|...|> text).
//
// Header-only, C++17, stdlib only (gguf.h adds POSIX mmap).

#include <cstdint>
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

    auto merges = m.kv("tokenizer.ggml.merges").as_str_array();
    merge_rank_.reserve(merges.size() * 2);
    for (size_t i = 0; i < merges.size(); i++)
      merge_rank_.emplace(merges[i], static_cast<int>(i));

    // Special (CONTROL, type==3) tokens are matched atomically in encode.
    const auto& tt = m.kv("tokenizer.ggml.token_type");
    for (size_t i = 0; i < tt.arr.size() && i < tokens_.size(); i++)
      if (tt.arr[i].i == 3)
        specials_.emplace_back(tokens_[i], static_cast<int>(i));

    eos_id_ = static_cast<int>(m.kv("tokenizer.ggml.eos_token_id").as_u32());
    bos_id_ = static_cast<int>(m.kv("tokenizer.ggml.bos_token_id").as_u32());

    for (uint32_t b = 0; b < 256; b++)
      cp2byte_.emplace(tokdata::byte2cp[b], static_cast<uint8_t>(b));
  }

  int eos_id() const { return eos_id_; }
  int bos_id() const { return bos_id_; }

  // Encode raw UTF-8 text to token ids. Special-token strings embedded in the
  // text (e.g. "<|im_start|>") are matched atomically; no BOS/EOS are added.
  std::vector<int> encode(const std::string& text) const {
    std::vector<int> out;
    size_t seg_start = 0;  // start of the pending normal-text segment
    size_t i = 0;
    while (i < text.size()) {
      // Longest special-token match at position i, if any.
      int best_id = -1;
      size_t best_len = 0;
      for (const auto& sp : specials_) {
        const std::string& s = sp.first;
        if (s.size() > best_len && text.compare(i, s.size(), s) == 0) {
          best_len = s.size();
          best_id = sp.second;
        }
      }
      if (best_id >= 0) {
        encode_segment_(text.data() + seg_start, i - seg_start, out);
        out.push_back(best_id);
        i += best_len;
        seg_start = i;
      } else {
        i++;
      }
    }
    encode_segment_(text.data() + seg_start, text.size() - seg_start, out);
    return out;
  }

  // Decode ids back to raw UTF-8 bytes (inverse of the byte2cp mapping).
  std::string decode(const std::vector<int>& ids) const {
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
  void encode_segment_(const char* data, size_t len,
                       std::vector<int>& out) const {
    if (len == 0) return;
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

    for (auto [i, e] : pretokenize_(cp))
      bpe_(data + off[i], off[e] - off[i], out);
  }

  // Hand-rolled Qwen2/GPT-2 pretokenizer regex. Returns [start,end) codepoint
  // spans. At each position the FIRST matching alternative wins.
  static std::vector<std::pair<size_t, size_t>> pretokenize_(
      const std::vector<uint32_t>& cp) {
    using tokdata::is_L;
    using tokdata::is_N;
    using tokdata::is_WS;
    auto punct = [](uint32_t c) { return !is_WS(c) && !is_L(c) && !is_N(c); };
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

      // A2 — word: [^\r\n\p{L}\p{N}]?\p{L}+
      if (!e) {
        size_t j = static_cast<size_t>(-1);
        if (cp[i] != 13 && cp[i] != 10 && !is_L(cp[i]) && !is_N(cp[i])) {
          if (i + 1 < n && is_L(cp[i + 1])) j = i + 1;  // optional prefix used
        } else if (is_L(cp[i])) {
          j = i;
        }
        if (j != static_cast<size_t>(-1)) {
          while (j < n && is_L(cp[j])) j++;
          e = j;
        }
      }

      // A3 — single digit: \p{N}
      if (!e && is_N(cp[i])) e = i + 1;

      // A4 — punct run: ?[^\s\p{L}\p{N}]+[\r\n]*
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

  std::vector<std::string> tokens_;                    // id -> vocab string
  std::unordered_map<std::string, int> token2id_;      // vocab string -> id
  std::unordered_map<std::string, int> merge_rank_;    // "A B" -> rank
  std::vector<std::pair<std::string, int>> specials_;  // CONTROL tokens
  std::unordered_map<uint32_t, uint8_t> cp2byte_;      // inverse byte2cp
  int eos_id_ = -1;
  int bos_id_ = -1;
};

}  // namespace tl
