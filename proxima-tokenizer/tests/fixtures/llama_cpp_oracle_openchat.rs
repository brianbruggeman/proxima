// Oracle capture: llama.cpp's own tokenizer + greedy decoder, run against
// the real openchat-3.5-1210 GGUF this crate's other real-vocab tests
// already depend on (`OPENCHAT_GGUF_PATH` in `proxima-tokenizer/src/gguf.rs`).
//
// Why this file exists: three rounds of forward-pass parity debugging ran
// against a target token asserted from intuition (`"The capital of France
// is"` -> `" Paris"`). It is wrong -- llama.cpp's own greedy decode gives
// `" known"` (`"The capital of France is known for its rich"`). Belief is
// not evidence; this file is the evidence, captured directly from the
// incumbent (guiding-principles §14, §9).
//
// Provenance
// ----------
// llama.cpp commit:  b25346221dadb9101aa9dda55431dde4d3596943
//                     (`git -C /Users/brianbruggeman/repos/others/llama.cpp log -1 --format=%H`)
// model file:         openchat-3.5-1210.Q4_K_S.gguf
//                     (TheBloke/openchat-3.5-1210-GGUF), 4140385376 bytes
//                     sha1 not pinned -- byte size + host path is the
//                     reproducibility anchor, matching this crate's existing
//                     `OPENCHAT_GGUF_PATH` convention.
// vocab shape:        `tokenizer.ggml.model = "llama"` (SentencePiece/unigram,
//                     scores present, no merges); `add_bos_token = true`,
//                     `add_eos_token = false`, bos=1, eos=32000.
//
// Commands
// --------
// Prompt token ids (`prompt_ids` below), one call per prompt:
//   llama-tokenize -m <model> -p '<prompt>' --ids
// (default: adds BOS, matching `tokenizer.ggml.add_bos_token = true` above --
// no `--no-bos` passed.)
//
// Generation (`generated_ids`/`generated_pieces` below), one call per prompt:
//   llama-cli -m <model> -p '<prompt>' -n 4 --temp 0 --top-k 1 -s 42 \
//     -no-cnv --no-warmup --simple-io --verbose-prompt
// Flags, and why: `--temp 0` + `--top-k 1` both force pure greedy argmax
// (belt and suspenders -- either alone is sufficient, together they leave no
// ambiguity about which sampler stage is doing the picking); `-s 42` pins
// the RNG seed so the run is reproducible even though greedy sampling at
// temp=0 does not consult it; `-no-cnv` disables conversation/chat-template
// wrapping so the prompt is tokenized exactly as typed, not wrapped in a
// chat template that would change the ids; `--no-warmup` skips the empty
// warmup decode so the perf/log lines are easier to read (does not affect
// the sampled ids); `--simple-io` avoids readline/terminal escape codes
// contaminating captured stdout; `--verbose-prompt` prints the exact
// `id -> piece` table this file's ids/pieces were read from.
//
// The ids llama-cli's sampler actually chose during generation are not
// printed directly (no `--print-token-ids`-style flag exists on this
// build). They were recovered by re-tokenizing the exact printed
// continuation text (`--verbose-prompt` on `<prompt><generated text>`) and
// taking the ids past the already-verified `prompt_ids` prefix -- confirmed
// exact-prefix-match against the independently captured `prompt_ids` for
// all 10 cases below, so this is not an approximation for a
// SentencePiece/unigram vocab with no segmentation ambiguity at these
// boundaries.
//
// Determinism check (required, and it passed): case `factual_completion`
// was captured twice, independently, back to back. Both runs produced
// byte-identical generated text (`" known for its rich"`) and therefore
// identical retokenized ids -- the oracle is reproducible under these
// flags. The second run is not duplicated as a fixture case (an identical
// prompt is not a second distinct oracle fact); this comment is the record
// of that check.
//
// What is deliberately NOT covered here: `generated_ids`/`generated_pieces`
// are the target for a later forward-pass parity test (comparing this
// crate's own logits-driven greedy decode against llama.cpp's), not
// exercised by any test in this commit. Only `prompt_ids` is asserted today
// (`encode_with_bos_eos_matches_llama_cpp_oracle_prompt_ids` in
// `proxima-tokenizer/src/gguf.rs`).
pub(crate) struct OracleCase {
    pub(crate) name: &'static str,
    pub(crate) prompt: &'static str,
    pub(crate) prompt_ids: &'static [u32],
    pub(crate) generated_ids: &'static [u32],
    pub(crate) generated_pieces: &'static [&'static str],
}

pub(crate) const ORACLE_CASES: &[OracleCase] = &[
    // factual completion -- the exact prompt this crate's parity debugging
    // was asserting a wrong target for. Greedy continues "is" with "known",
    // not "Paris".
    OracleCase {
        name: "factual_completion",
        prompt: "The capital of France is",
        prompt_ids: &[1, 415, 5565, 302, 4843, 349],
        generated_ids: &[2651, 354, 871, 6708],
        generated_pieces: &[" known", " for", " its", " rich"],
    },
    // list continuation -- structured/numeric pattern, not prose; exercises
    // digit-by-digit segmentation (each digit is its own vocab piece here).
    OracleCase {
        name: "list_continuation",
        prompt: "1, 2, 3, 4,",
        prompt_ids: &[1, 28705, 28740, 28725, 28705, 28750, 28725, 28705, 28770, 28725, 28705, 28781, 28725],
        generated_ids: &[28705, 28782, 28725, 28705],
        generated_pieces: &[" ", "5", ",", " "],
    },
    // code fragment -- indentation (leading-space run tokens) and an
    // embedded newline in both the prompt and the generated continuation.
    OracleCase {
        name: "code_fragment",
        prompt: "def fibonacci(n):\n    if n <= 1:",
        prompt_ids: &[
            1, 801, 16182, 266, 28127, 28732, 28711, 1329, 13, 2287, 513, 307, 5042, 28705, 28740, 28747,
        ],
        generated_ids: &[13, 5390, 604, 307],
        generated_pieces: &["\n", "       ", " return", " n"],
    },
    // multibyte UTF-8 (Japanese) -- prompt has no vocab merges for these
    // characters (each kana/kanji is its own single-codepoint piece), and
    // generation continues with further multibyte pieces.
    OracleCase {
        name: "multibyte_utf8_japanese",
        prompt: "こんにちは、世界",
        prompt_ids: &[1, 28705, 29543, 29585, 29174, 30173, 29277, 29041, 30050, 29822],
        generated_ids: &[28991, 28993, 29522, 29501],
        generated_pieces: &["中", "の", "プ", "ロ"],
    },
    // single-word prompt -- minimal pretokenize span, one content token
    // plus BOS.
    OracleCase {
        name: "single_word",
        prompt: "Hello",
        prompt_ids: &[1, 22557],
        generated_ids: &[28725, 13, 13, 28737],
        generated_pieces: &[",", "\n", "\n", "I"],
    },
    // empty prompt -- BOS-only encode path; exercises `encode` on `""`
    // directly rather than skipping it.
    OracleCase {
        name: "empty_prompt",
        prompt: "",
        prompt_ids: &[1],
        generated_ids: &[422, 28705, 28750, 28734],
        generated_pieces: &[" #", " ", "2", "0"],
    },
    // short-answer format -- candidate for early termination: greedy only
    // produced 2 visible generated tokens here (not 4) before the printed
    // continuation stopped growing, even though `-n 4` was requested. This
    // is exactly the kind of oracle behaviour a hand-picked target token
    // would never have surfaced.
    OracleCase {
        name: "short_answer_early_stop",
        prompt: "Q: What is 2+2?\nA:",
        prompt_ids: &[1, 1186, 28747, 1824, 349, 28705, 28750, 28806, 28750, 28804, 13, 28741, 28747],
        generated_ids: &[28705, 28781],
        generated_pieces: &[" ", "4"],
    },
    // multibyte UTF-8 (Vietnamese diacritics) -- combining/precomposed
    // characters split across multiple vocab pieces (e.g. "Việt" ->
    // " Vi" + "ệ" + "t"), a real-world byte-fallback/multibyte case distinct
    // from the Japanese one above.
    OracleCase {
        name: "multibyte_utf8_vietnamese",
        prompt: "Cửa Việt is a town in",
        prompt_ids: &[1, 334, 30389, 28708, 11004, 29539, 28707, 349, 264, 3736, 297],
        generated_ids: &[2332, 29508, 817, 365],
        generated_pieces: &[" Qu", "ả", "ng", " B"],
    },
    // long/repetitive prompt -- a full sentence followed by the start of an
    // exact repeat, checking segmentation stays stable over a longer,
    // multi-word-token context (e.g. "fox" splits as " f" + "ox").
    OracleCase {
        name: "long_repeated_context",
        prompt: "The quick brown fox jumps over the lazy dog. The",
        prompt_ids: &[1, 415, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 3914, 28723, 415],
        generated_ids: &[2936, 9060, 285, 1142],
        generated_pieces: &[" quick", " brown", " f", "ox"],
    },
    // code fragment with a quote character -- exercises `("` as its own
    // vocab piece and a generated continuation with no leading space
    // (`"Welcome` picks up mid-word after the quote).
    OracleCase {
        name: "code_fragment_quote",
        prompt: "print(\"",
        prompt_ids: &[1, 2682, 618],
        generated_ids: &[28780, 23079, 298, 272],
        generated_pieces: &["W", "elcome", " to", " the"],
    },
];
