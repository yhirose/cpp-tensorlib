#!/usr/bin/env python3
"""Ground truth for check_tokenizer from llama.cpp's own tokenizer.

    tools/tokenizer_oracle.py LLAMA_TOKENIZE MODEL.gguf NAME > bench/host/tokenizer_oracle_NAME.h

Runs llama-tokenize (llama.cpp's tools/tokenize, special tokens parsed, no BOS
added: what tl::tokenizer::encode does) over the cases below and writes them as
{text, ids} for bench/host/check_tokenizer.cpp. The cases are the ones a
tokenizer gets wrong first: runs of spaces, newlines and tabs, characters the
vocabulary lacks (byte fallback), scripts other than Latin, code, and a chat
template's special tokens.
"""

import ast
import os
import subprocess
import sys
import tempfile

CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "日本語のテスト。東京は晴れです。",
    "123  456\n789",
    "  leading spaces and\ttab",
    "trailing spaces   ",
    "line one\n\nline three\n\n\n\nline seven",
    "emoji: 🙂🚀 and symbols ∑ ≈ ∞",
    "rare: 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 \U0001fae0 ༺༻",
    "def f(x):\n    return x * 2  # double\n",
    "Ünïcödé façade naïve café — “quotes” ‘and’ …",
    "<start_of_turn>user\nGive me a short introduction to large language models.<end_of_turn>\n"
    "<start_of_turn>model\n",
    "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n4<end_of_turn>\n",
    # Runs and repeats, where adjacent candidate pairs tie on score and merges
    # leave stale ones behind.
    "zzzzzzz aaaaaaaaaaa ........ ~~~~~ hhhhhhhhhhh!!!!",
    "ababababab xyzxyzxyzxyz qqqqqqqqqqqqqqqqqqqqqqqqqqq",
    "Donaudampfschifffahrtsgesellschaftskapitän",
    # Where the BPE pre-tokenizers differ: combining marks, which Qwen3.5's counts
    # with the letters (Thai and Devanagari vowel signs, decomposed accents, emoji
    # variation selectors), and digit runs, which Llama 3's takes three at a time.
    "Thai: สวัสดีครับ ภาษาไทย; Hindi: नमस्ते दुनिया",
    "decomposed: cafe\u0301 nai\u0308ve, a lone mark \u0301abc, emoji: ❤️ ✌️ 1️⃣",
    "digits: 1234567890, 3.14159, 1,000,000 and ٣٤٥٦٧٨",
    # Words Llama 3's vocabulary has whole but its merges never reach, which its
    # pre-tokenizer takes as they are (ignore_merges).
    "Tiếng Việt: nhiều việc ngoài; Türkçe: değiş, oluştur",
    # Not UTF-8: a stray continuation byte (one byte, as llama.cpp's
    # unicode_len_utf8 has it, so the letters after it still merge) and a lead
    # byte the text cuts short.
    b"stray \x80bytes then a cut \xe3\x81",
]


# A family's own cases, after CASES: what its tokenizer does that no other's does.
EXTRA = {
    # PLaMo 3: hard cuts around <|plamo:NAME|> (64 code points at most, no '|' or
    # white space in NAME), runs of 4+ of one character (not newlines) and 2+
    # spaces, and U+EE00, a cut that is dropped; its translation prompt.
    "plamo3": [
        "helloooo and hellooo, aaaa bbbbbbbb, ----====, 😀😀😀😀😀",
        "two  spaces, three   spaces, tabs\t\t\t\t and newlines\n\n\n\n\nend",
        "<|plamo:msg|>known<|plamo:foo|>unknown<|plamo:a b|>space<|plamo:|>empty<|plamo:x|y|>bar",
        "<|plamo:" + "n" * 64 + "|>sixty-four <|plamo:" + "n" * 65 + "|>sixty-five",
        "mark\uee00in\uee00\uee00the middle\uee00",
        "\ufeffbyte order mark first",
        # White space ends a marker's NAME as Python's \\s has it (U+001C-001F too).
        "<|plamo:a\x1cb|>, <|plamo:a\u3000b|>, <|plamo:a\u00a0b|>",
        # A character that is only ever inside longer pieces: its bytes win at a
        # position where a piece starts (PFN's fallback is a candidate everywhere).
        "\u05d9\u05e8\u05d9 \u05d9\u05d3\u05d9",
        "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input"
        "<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>The weather in Tokyo is sunny today."
        "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>",
        "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input"
        "<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>今日の東京は晴れです。"
        "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>",
    ],
    # Gemma 4 (BPE with SentencePiece's spaces): every space is U+2581, the text is
    # cut into runs of non-newlines and of newlines, and each run merges over its
    # characters (not bytes). Its turns and channels are special tokens.
    "gemma4": [
        "  two spaces, then\n\n  an indent and a trailing tab\t",
        "\n\n\n",
        # The newline runs the vocabulary has whole (up to 31) are one token each.
        "".join(f"x{'\n' * n}" for n in range(1, 41)),
        "\n",
        " ",
        "one\n\n\n\n\n\n\n\ntwo\n",
        "non-breaking space and ideographic　space",
        "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n",
        "<|turn>model\n<|channel>thought\n<channel|>Four.<turn|>",
        "<|tool_response>{\"ok\": true}<eos>",
        "<|image>x<|audio>y<|\"|>z<|\"|>",
        "ＡＢＣ全角 and 한국어 and हिन्दी and 🇯🇵👨‍👩‍👧",
    ],
}


def raw(text):
    return text if isinstance(text, bytes) else text.encode("utf-8")


def cpp_string(text):
    """A C++ string literal of the case's bytes: raw where it is UTF-8, escaped where not."""
    if isinstance(text, str):
        return f'std::string(R"TOK({text})TOK")'
    body = "".join(chr(b) if 32 <= b < 127 and chr(b) not in '\\"' else f"\\x{b:02x}\"\"" for b in text)
    return f'std::string("{body}")'


def tokenize(exe, model, text):
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(raw(text))
        path = f.name
    try:
        out = subprocess.run(
            [exe, "-m", model, "-f", path, "--ids", "--no-bos", "--no-escape", "--log-disable"],
            check=True, capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    return ast.literal_eval(out.strip().splitlines()[-1])


def main():
    exe, model, name = sys.argv[1:4]
    print("#pragma once")
    print(f"// Generated by tools/tokenizer_oracle.py from llama.cpp's llama-tokenize on")
    print(f"// {os.path.basename(model)}. {{text, expected ids}}.")
    print("#include <string>\n#include <vector>")
    print(f"namespace tokoracle_{name} {{")
    print("struct Case { std::string text; std::vector<int> ids; };")
    print("inline std::vector<Case> cases() { return {")
    for text in CASES + EXTRA.get(name, []):
        ids = tokenize(exe, model, text)
        print(f'  {{ {cpp_string(text)}, {{{",".join(map(str, ids))}}} }},')
    print("};}\n}")


if __name__ == "__main__":
    main()
