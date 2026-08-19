//! Builds a [`Vocab`] from a [`proxima_gguf::ParsedGguf`]'s metadata.
//! Feature-gated (`gguf`) so the tokenizer core has no hard dependency on
//! the GGUF reader -- a caller who already has tokens/merges from
//! somewhere else (a plain JSON vocab, a hand-built test fixture) never
//! pulls this module in.
//!
//! # Real metadata keys
//!
//! Dumped directly from
//! `~/repos/others/llama.cpp/models/ggml-vocab-llama-bpe.gguf` (a
//! vocab-only fixture, `tensor_count: 0`) rather than assumed from memory
//! -- every key below is one this crate has actually seen on the wire:
//!
//! | key | type | on the real fixture |
//! |---|---|---|
//! | `tokenizer.ggml.model` | string | `"gpt2"` (byte-level BPE) on the llama-bpe fixture; `"llama"` (SentencePiece/SPM) on the openchat-3.5-1210 fixture below -- this is the key [`vocab_from_metadata`] dispatches the encoder on |
//! | `tokenizer.ggml.pre` | string | `"llama-bpe"` (selects `LLAMA_VOCAB_PRE_TYPE_LLAMA3`'s pretokenizer regex in llama.cpp) |
//! | `tokenizer.ggml.tokens` | array\<string\> | 128256 entries, index == token id |
//! | `tokenizer.ggml.token_type` | array\<i32\> | 128256 entries, parallel to `tokens` (see [`crate::vocab::TokenType::from_raw`]) |
//! | `tokenizer.ggml.merges` | array\<string\> | 280147 entries, each `"left right"` space-separated, priority order -- `"gpt2"` vocabs only |
//! | `tokenizer.ggml.scores` | array\<f32\> | one per token, `"llama"` vocabs only -- dumped from the real 32002-token openchat-3.5-1210 fixture (`tokenizer.ggml.model = "llama"`, no `merges` key at all) |
//! | `tokenizer.ggml.bos_token_id` | u32 | `128000` |
//! | `tokenizer.ggml.eos_token_id` | u32 | `128001` |
//!
//! `tokenizer.ggml.unknown_token_id` and `tokenizer.ggml.padding_token_id`
//! are read too, but absent on the llama-bpe fixture -- byte-level BPE has
//! no OOV case (every byte has a base token), so llama.cpp's own gguf
//! writer omits `unknown_token_id` for this vocab family.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::{MetadataArray, MetadataValue, ParsedGguf};

use crate::error::TokenizerError;
use crate::vocab::Vocab;

const MODEL_KEY: &str = "tokenizer.ggml.model";
const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const MERGES_KEY: &str = "tokenizer.ggml.merges";
const SCORES_KEY: &str = "tokenizer.ggml.scores";
const BOS_KEY: &str = "tokenizer.ggml.bos_token_id";
const EOS_KEY: &str = "tokenizer.ggml.eos_token_id";
const UNKNOWN_KEY: &str = "tokenizer.ggml.unknown_token_id";

/// Builds a [`Vocab`] from `metadata`'s tokenizer keys, selecting the
/// merges-driven ([`Vocab::new`]) or scores-driven ([`Vocab::new_unigram`])
/// constructor from `tokenizer.ggml.model` -- never a caller flag, matching
/// llama.cpp's own dispatch (`tokenizer_model == "gpt2"` /
/// `tokenizer_model == "llama"`, `llama-vocab.cpp:1405-1428`).
///
/// # Errors
///
/// [`TokenizerError::MissingMetadataKey`] if `tokenizer.ggml.tokens` or
/// `tokenizer.ggml.model` is absent, or if the declared model's required
/// companion array (`merges` for `"gpt2"`, `scores` for `"llama"`) is
/// missing -- a vocab that declares one family but carries neither or both
/// arrays is exactly this case, named by which key came up empty.
/// [`TokenizerError::UnsupportedTokenizerModel`] for any other
/// `tokenizer.ggml.model` value. [`TokenizerError::WrongMetadataType`] if a
/// present key has the wrong GGUF value type. Anything [`Vocab::new`]/
/// [`Vocab::new_unigram`] can fail with otherwise (a malformed merge rule,
/// a missing base byte token, a scores/tokens length mismatch).
pub fn vocab_from_metadata(metadata: &ParsedGguf) -> Result<Vocab, TokenizerError> {
    let tokens = string_array(metadata, TOKENS_KEY)?.ok_or(TokenizerError::MissingMetadataKey { key: TOKENS_KEY })?;
    let bos_token_id = u32_scalar(metadata, BOS_KEY)?;
    let eos_token_id = u32_scalar(metadata, EOS_KEY)?;
    let unknown_token_id = u32_scalar(metadata, UNKNOWN_KEY)?;
    let model = string_scalar(metadata, MODEL_KEY)?.ok_or(TokenizerError::MissingMetadataKey { key: MODEL_KEY })?;

    match model.as_str() {
        "gpt2" => {
            let merges =
                string_array(metadata, MERGES_KEY)?.ok_or(TokenizerError::MissingMetadataKey { key: MERGES_KEY })?;
            Vocab::new(tokens, &merges, bos_token_id, eos_token_id, unknown_token_id)
        }
        "llama" => {
            let scores =
                f32_array(metadata, SCORES_KEY)?.ok_or(TokenizerError::MissingMetadataKey { key: SCORES_KEY })?;
            Vocab::new_unigram(tokens, scores, bos_token_id, eos_token_id, unknown_token_id)
        }
        other => Err(TokenizerError::UnsupportedTokenizerModel { model: String::from(other) }),
    }
}

fn string_array(metadata: &ParsedGguf, key: &'static str) -> Result<Option<Vec<String>>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::Array(MetadataArray::String(values))) => Ok(Some(values.clone())),
        Some(_) => Err(TokenizerError::WrongMetadataType { key }),
    }
}

fn f32_array(metadata: &ParsedGguf, key: &'static str) -> Result<Option<Vec<f32>>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::Array(MetadataArray::F32(values))) => Ok(Some(values.clone())),
        Some(_) => Err(TokenizerError::WrongMetadataType { key }),
    }
}

fn string_scalar(metadata: &ParsedGguf, key: &'static str) -> Result<Option<String>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(TokenizerError::WrongMetadataType { key }),
    }
}

fn u32_scalar(metadata: &ParsedGguf, key: &'static str) -> Result<Option<u32>, TokenizerError> {
    match metadata.metadata_value(key) {
        None => Ok(None),
        Some(MetadataValue::U32(value)) => Ok(Some(*value)),
        Some(_) => Err(TokenizerError::WrongMetadataType { key }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn missing_tokens_key_is_an_error() {
        let metadata = ParsedGguf {
            version: 3,
            tensor_count: 0,
            kv_count: 0,
            metadata: Vec::new(),
            tensors: Vec::new(),
            data_offset: 0,
            alignment: 32,
        };
        let error = vocab_from_metadata(&metadata).expect_err("no tokens key");
        assert!(matches!(
            error,
            TokenizerError::MissingMetadataKey { key: TOKENS_KEY }
        ));
    }

    /// Loads the real llama-bpe vocab fixture and confirms the exact
    /// metadata keys/types documented in this module's doc comment are
    /// what actually round-trips through `proxima-gguf`.
    /// `#[ignore]`d: depends on a sibling checkout that may not exist on
    /// every host.
    #[test]
    #[ignore = "depends on a real .gguf checkout outside this repo"]
    fn builds_a_vocab_from_the_real_llama_bpe_fixture() {
        let candidate = Path::new(
            "/Users/brianbruggeman/repos/others/llama.cpp/models/ggml-vocab-llama-bpe.gguf",
        );
        if !candidate.exists() {
            eprintln!("no real .gguf found at {candidate:?}, skipping");
            return;
        }
        let (parsed, _bytes) = proxima_gguf::edge::read_file(candidate).expect("parse real gguf file");
        let vocab = vocab_from_metadata(&parsed).expect("builds vocab from real metadata");
        assert_eq!(vocab.len(), 128_256);
        assert_eq!(vocab.bos_token_id(), Some(128_000));
        assert_eq!(vocab.eos_token_id(), Some(128_001));
        println!("real fixture vocab: {} tokens", vocab.len());
    }

    /// The decode path end to end at the real openchat-3.5-1210 vocab
    /// scale (32002 tokens): a synthetic logits vector with a known peak
    /// runs through [`crate::sample::greedy_pick`], and the resulting id
    /// decodes ([`crate::decode`]) back to the exact expected string --
    /// including a multi-byte UTF-8 token, so the byte-level decode path
    /// is exercised, not just the id lookup. Only the metadata region is
    /// read (growing-buffer `parse_complete` loop, matching
    /// `proxima-gguf/src/restack.rs`'s `real_mixtral_file` module) -- the
    /// 3.9 GB tensor payload is never touched. `#[ignore]`d: depends on a
    /// host-local model cache outside this repo.
    const OPENCHAT_GGUF_PATH: &str =
        "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

    /// Reads only the metadata region of the real openchat-3.5-1210 GGUF
    /// (growing-buffer `parse_complete` loop, matching
    /// `proxima-gguf/src/restack.rs`'s `real_mixtral_file` module) and
    /// builds a [`Vocab`] from it -- the 3.9 GB tensor payload is never
    /// touched. `None` if the host-local model cache this crate's real-vocab
    /// tests depend on is absent.
    fn load_real_openchat_vocab() -> Option<Vocab> {
        use std::io::{Read, Seek, SeekFrom};

        let candidate = Path::new(OPENCHAT_GGUF_PATH);
        if !candidate.exists() {
            eprintln!("no real openchat .gguf found at {candidate:?}, skipping");
            return None;
        }

        let mut file = std::fs::File::open(candidate).expect("open host-local openchat gguf fixture");
        let mut header_buf = Vec::new();
        let parsed = 'grow: {
            for cap in [4usize << 20, 16 << 20, 64 << 20] {
                header_buf.resize(cap, 0);
                file.seek(SeekFrom::Start(0)).expect("seek to file start");
                let read = file.read(&mut header_buf).expect("read gguf header region");
                header_buf.truncate(read);
                if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header_buf) {
                    break 'grow parsed;
                }
            }
            panic!("gguf metadata region did not fit in 64 MiB");
        };
        Some(vocab_from_metadata(&parsed).expect("builds vocab from real openchat metadata"))
    }

    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn greedy_decode_at_real_openchat_vocab_scale() {
        use crate::sample::greedy_pick;

        let Some(vocab) = load_real_openchat_vocab() else { return };
        assert_eq!(vocab.len(), 32_002, "real openchat-3.5-1210 vocab must have exactly 32002 tokens");

        // three distinct tokens, picked by inspecting the real vocab
        // (`tokenizer.ggml.model = "llama"`, a SentencePiece/unigram
        // vocab, confirmed via `tokenizer.ggml.scores` present and
        // `tokenizer.ggml.merges` absent): the BOS control token, a
        // plain ASCII subword, and a multi-byte UTF-8 katakana token.
        let cases: [(u32, &str); 3] = [(1, "<s>"), (450, "de"), (30_000, "ァ")];

        for (token_id, expected_text) in cases {
            let mut logits = alloc::vec![0.0f32; vocab.len()];
            logits[token_id as usize] = 100.0;

            let picked = greedy_pick(&logits).expect("logits are non-empty");
            assert_eq!(picked, token_id, "greedy pick must recover the peaked token id");

            let decoded = decode_ids_for_test(&vocab, &[picked]);
            assert_eq!(decoded, expected_text, "decode must recover the exact expected text");
        }

        // degenerate control: a flat, constant logits vector carries no
        // signal. it must NOT resolve to any of the peaked ids above --
        // otherwise a broken argmax (e.g. one that always returns a fixed
        // index) could pass the assertions above by coincidence.
        let flat_logits = alloc::vec![1.0f32; vocab.len()];
        let flat_pick = greedy_pick(&flat_logits).expect("flat logits are non-empty");
        assert_eq!(flat_pick, 0, "ties resolve deterministically to the lowest id");
        for (token_id, _) in cases {
            assert_ne!(flat_pick, token_id, "a flat vector must not coincidentally hit a real peak's id");
        }
    }

    #[cfg(test)]
    fn decode_ids_for_test(vocab: &Vocab, ids: &[u32]) -> String {
        crate::decode(ids, vocab).expect("decodes real vocab ids")
    }

    /// The measured gap this module exists to close: the openchat-3.5-1210
    /// checkpoint (`tokenizer.ggml.model = "llama"`) declares no
    /// `tokenizer.ggml.merges`, so `crate::bpe::encode_pretoken`'s
    /// merges-driven path degenerated to one token per byte (25 tokens for
    /// a 24-char prompt). Asserts the exact id sequence -- ground truth
    /// obtained by looking up the real vocab's own scores/token-id table
    /// for each expected piece (`"\u{2581}The"`, `"\u{2581}capital"`, ...),
    /// dumped directly from this fixture, not guessed or taken from a
    /// second tokenizer implementation.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn unigram_encode_the_capital_of_france_is_not_one_token_per_byte() {
        let Some(vocab) = load_real_openchat_vocab() else { return };
        assert!(vocab.is_unigram(), "openchat-3.5-1210 declares tokenizer.ggml.model = \"llama\"");

        let prompt = "The capital of France is";
        let ids = crate::encode(prompt, &vocab).expect("encodes against real vocab");
        assert!(
            ids.len() < 10,
            "expected a handful of subword pieces, got {} ids (one-token-per-byte regression): {ids:?}",
            ids.len()
        );

        let expected_piece_ids = ["\u{2581}The", "\u{2581}capital", "\u{2581}of", "\u{2581}France", "\u{2581}is"]
            .map(|piece| vocab.token_id(piece).unwrap_or_else(|| panic!("{piece:?} must be in the real vocab")));
        assert_eq!(
            ids, expected_piece_ids,
            "must segment into exactly the real vocab's subword pieces for this prompt"
        );
        assert_eq!(ids.len(), 5, "the sequence=25 bug produced one id per byte; this must be ~5, not 25");

        let with_bos = crate::encode_with_bos_eos(prompt, &vocab, true, false).expect("encodes with bos");
        assert_eq!(with_bos.len(), 6, "add_bos_token = true on this checkpoint, so 5 pieces + 1 bos");
        assert_eq!(with_bos.first().copied(), vocab.bos_token_id());

        let decoded = crate::decode(&ids, &vocab).expect("decodes against real vocab");
        assert_eq!(decoded, prompt, "round trip must recover the exact prompt");
    }

    /// Round-trip is the hard correctness gate (matching this crate's
    /// existing philosophy for the llama-bpe fixture,
    /// `round_trips_against_the_real_vocab` above): multi-byte UTF-8, and a
    /// raw byte with no multi-char vocab piece (only its `<0xXX>`
    /// byte-fallback token), both must survive encode-then-decode exactly.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn unigram_round_trips_multibyte_utf8_and_byte_fallback() {
        let Some(vocab) = load_real_openchat_vocab() else { return };
        for text in [
            "The capital of France is Paris",
            "нещо на Български",
            "🚀 (normal) ✅",
            " this is 🦙.cpp",
            "Cửa Việt",
            "\u{0007}", // BEL control char: no multi-char piece, must fall back to <0x07>
            "",
        ] {
            let ids = crate::encode(text, &vocab).expect("encodes against real vocab");
            let decoded = crate::decode(&ids, &vocab).expect("decodes against real vocab");
            assert_eq!(decoded, text, "round trip failed for {text:?} (ids: {ids:?})");
        }
    }

    /// Degenerate control for the exact regression this module fixes: a
    /// merges-driven vocab's byte-level BPE encoder, run over a scores-only
    /// vocab's tokens by mistake, would produce one id per byte. Confirms
    /// the real fix path never does that for an ordinary ASCII sentence --
    /// this must be impossible to regress silently back to `sequence=25`.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn unigram_encode_never_degenerates_to_one_token_per_byte() {
        let Some(vocab) = load_real_openchat_vocab() else { return };
        let sentence = "The quick brown fox jumps over the lazy dog";
        let ids = crate::encode(sentence, &vocab).expect("encodes against real vocab");
        assert!(
            ids.len() < sentence.len(),
            "one-token-per-byte regression: {} ids for a {}-char sentence",
            ids.len(),
            sentence.len()
        );
        assert!(ids.len() < 15, "expected roughly word-scale segmentation, got {} ids: {ids:?}", ids.len());
    }

    include!("../tests/fixtures/llama_cpp_oracle_openchat.rs");

    /// The measurement this fixture exists to make possible: for every one
    /// of the 10 real-world prompts llama.cpp's own tokenizer + greedy
    /// decoder were run against (see `llama_cpp_oracle_openchat.rs` for the
    /// exact commands and provenance), `crate::encode_with_bos_eos` must
    /// reproduce llama.cpp's own prompt token ids exactly. This is a
    /// measurement, not a belief -- the parity debugging this fixture
    /// replaces was asserting a target token from intuition, and it was
    /// wrong.
    ///
    /// Only `prompt_ids` is checked here. `generated_ids`/`generated_pieces`
    /// on each case are the target for a later forward-pass parity test
    /// (this crate's own logits-driven decode vs. llama.cpp's), not
    /// exercised by this test.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn encode_with_bos_eos_matches_llama_cpp_oracle_prompt_ids() {
        let Some(vocab) = load_real_openchat_vocab() else { return };

        for case in ORACLE_CASES {
            let ids = crate::encode_with_bos_eos(case.prompt, &vocab, true, false)
                .unwrap_or_else(|error| panic!("{}: encode_with_bos_eos failed: {error}", case.name));
            assert_eq!(
                ids.as_slice(),
                case.prompt_ids,
                "{}: our encoder's ids for {:?} must match llama.cpp's own tokenization (got {ids:?})",
                case.name,
                case.prompt
            );
            // not asserted against our own decode yet (that is the
            // forward-pass parity test this fixture exists to enable) --
            // only shape-checked here, so the capture itself is internally
            // consistent.
            assert_eq!(
                case.generated_ids.len(),
                case.generated_pieces.len(),
                "{}: captured generated_ids/generated_pieces must be parallel arrays",
                case.name
            );
        }
    }
}
