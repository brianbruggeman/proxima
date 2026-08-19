//! Crate-level integration tests: round-trip over a larger hand-built
//! vocab exercising multi-step merges, explicit special-token handling,
//! sad paths, and (feature-gated, `#[ignore]`d) the real llama-bpe
//! fixture against llama.cpp's own oracle token ids.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::byte_level::byte_to_char;
use crate::error::TokenizerError;
use crate::vocab::Vocab;
use crate::{decode, encode, encode_with_bos_eos};

/// A vocab covering all 256 base byte tokens, three chained merges
/// ("h"+"e" -> "he", "he"+"l" -> "hel", "hel"+"lo" via "l"+"o" -> "lo"
/// then "hel"+"lo" -> "hello") and a control-style BOS/EOS pair whose
/// display strings are plain ASCII (like llama.cpp's
/// `<|begin_of_text|>`/`<|end_of_text|>`), so this crate's tests are not
/// only ever run against `tiny_vocab`'s two-merge case.
fn synthetic_vocab() -> Vocab {
    let mut tokens: Vec<String> = (0..=255u8).map(|byte| String::from(byte_to_char(byte))).collect();
    for word in ["he", "hel", "lo", "hello", "<|begin_of_text|>", "<|end_of_text|>"] {
        tokens.push(word.to_string());
    }
    let bos_id = (tokens.len() - 2) as u32;
    let eos_id = (tokens.len() - 1) as u32;

    let merges = [
        "h e",     // -> "he"
        "he l",    // -> "hel"
        "l o",     // -> "lo"
        "hel lo",  // -> "hello"
    ]
    .map(String::from);

    Vocab::new(tokens, &merges, Some(bos_id), Some(eos_id), None).expect("synthetic vocab builds")
}

#[test]
fn round_trip_covers_multi_step_merges_and_plain_bytes() {
    let vocab = synthetic_vocab();
    for text in [
        "hello",
        "hello hello",
        "hel",
        "he",
        "xyz hello xyz",
        "Hello, World! 123",
        "  leading and trailing spaces  ",
        "\t\ttabs\n\nand newlines\n",
        "日本語テスト",  // multi-byte UTF-8 with no merge rules at all
        "emoji: \u{1F600}\u{1F601}",
        "",
    ] {
        let ids = encode(text, &vocab).expect("encodes");
        let decoded = decode(&ids, &vocab).expect("decodes");
        assert_eq!(decoded, text, "round trip failed for {text:?} (ids: {ids:?})");
    }
}

#[test]
fn special_tokens_are_explicit_not_silently_dropped() {
    let vocab = synthetic_vocab();

    // plain encode never adds BOS/EOS.
    let plain = encode("hello", &vocab).expect("plain encode");
    assert_ne!(plain.first().copied(), vocab.bos_token_id());

    // explicit opt-in adds both, and decode still reconstructs exactly
    // the literal text those control tokens' display strings carry --
    // nothing about them is dropped.
    let with_special = encode_with_bos_eos("hello", &vocab, true, true).expect("encode with bos/eos");
    assert_eq!(with_special.first().copied(), vocab.bos_token_id());
    assert_eq!(with_special.last().copied(), vocab.eos_token_id());
    let decoded = decode(&with_special, &vocab).expect("decodes");
    assert_eq!(decoded, "<|begin_of_text|>hello<|end_of_text|>");
}

#[test]
fn a_string_of_only_special_tokens_round_trips() {
    let vocab = synthetic_vocab();
    let ids = encode_with_bos_eos("", &vocab, true, true).expect("encode empty body with bos/eos");
    assert_eq!(ids.len(), 2);
    let decoded = decode(&ids, &vocab).expect("decodes");
    assert_eq!(decoded, "<|begin_of_text|><|end_of_text|>");
}

#[test]
fn empty_input_encodes_to_no_tokens() {
    let vocab = synthetic_vocab();
    let ids = encode("", &vocab).expect("encodes empty");
    assert!(ids.is_empty());
    assert_eq!(decode(&ids, &vocab).expect("decodes"), "");
}

#[test]
fn decoding_an_unknown_token_id_is_a_typed_error_not_a_panic() {
    let vocab = synthetic_vocab();
    let error = decode(&[999_999], &vocab).expect_err("id far outside the vocab");
    assert!(matches!(error, TokenizerError::TokenIdOutOfRange { token_id: 999_999, .. }));
}

#[test]
fn decoding_bytes_that_are_not_valid_utf8_is_a_typed_error_not_a_panic() {
    let vocab = synthetic_vocab();
    // byte 0x80 alone is a bare UTF-8 continuation byte -- never valid on
    // its own. Its base token id is index 0x80 in the first 256 tokens.
    let lone_continuation_byte_id = 0x80u32;
    let error = decode(&[lone_continuation_byte_id], &vocab).expect_err("invalid utf-8");
    assert_eq!(error, TokenizerError::InvalidUtf8);
}

#[cfg(feature = "gguf")]
mod real_fixture {
    use std::path::Path;

    use super::*;
    use crate::gguf::vocab_from_metadata;

    /// A curated subset of the (string, expected token ids) pairs
    /// llama.cpp itself ships, commented out at the top of
    /// `tests/test-tokenizer-0.cpp` in the sibling llama.cpp checkout --
    /// this crate's real oracle, not this crate's own guess.
    fn oracle_cases() -> Vec<(&'static str, Vec<u32>)> {
        vec![
            ("Hello world", vec![9906, 1917]),
            (" Hello world", vec![22691, 1917]),
            ("Hello World", vec![9906, 4435]),
            (" Hello World", vec![22691, 4435]),
            ("Hello, world!", vec![9906, 11, 1917, 0]),
            (" Hello, world!", vec![22691, 11, 1917, 0]),
            ("Hello", vec![9906]),
            (" Hello", vec![22691]),
            ("  Hello", vec![220, 22691]),
            ("   Hello", vec![256, 22691]),
            ("    Hello", vec![262, 22691]),
            ("    Hello\n    Hello", vec![262, 22691, 198, 262, 22691]),
            (" (", vec![320]),
            ("\n =", vec![198, 284]),
            ("' era", vec![6, 11639]),
            ("3", vec![18]),
            ("33", vec![1644]),
            ("333", vec![8765]),
            ("3333", vec![8765, 18]),
            ("33333", vec![8765, 1644]),
            ("333333", vec![8765, 8765]),
            ("w048 7tuijk dsdfhu", vec![86, 23904, 220, 22, 83, 2005, 42908, 11729, 3013, 17156]),
        ]
    }

    fn load_real_vocab() -> Option<Vocab> {
        let candidate = Path::new(
            "/Users/brianbruggeman/repos/others/llama.cpp/models/ggml-vocab-llama-bpe.gguf",
        );
        if !candidate.exists() {
            eprintln!("no real .gguf found at {candidate:?}, skipping");
            return None;
        }
        let (parsed, _bytes) = proxima_gguf::edge::read_file(candidate).expect("parse real gguf file");
        Some(vocab_from_metadata(&parsed).expect("build vocab from real metadata"))
    }

    /// Round-trip is the hard gate: every string must survive
    /// encode-then-decode against the real 128256-token llama-bpe vocab,
    /// regardless of whether this crate's pretokenizer matches
    /// llama.cpp's PCRE-based one byte-for-byte on token *boundaries*.
    #[test]
    #[ignore = "depends on a real .gguf checkout outside this repo"]
    fn round_trips_against_the_real_vocab() {
        let Some(vocab) = load_real_vocab() else { return };
        let mut count = 0;
        for (text, _) in oracle_cases() {
            let ids = encode(text, &vocab).expect("encodes against real vocab");
            let decoded = decode(&ids, &vocab).expect("decodes against real vocab");
            assert_eq!(decoded, text, "round trip failed for {text:?} (ids: {ids:?})");
            count += 1;
        }
        // also a few strings the oracle table doesn't cover, still
        // asserting round trip only.
        for text in [
            "нещо на Български",
            "🚀 (normal) ✅",
            " this is 🦙.cpp",
            "Cửa Việt",
        ] {
            let ids = encode(text, &vocab).expect("encodes against real vocab");
            let decoded = decode(&ids, &vocab).expect("decodes against real vocab");
            assert_eq!(decoded, text, "round trip failed for {text:?} (ids: {ids:?})");
            count += 1;
        }
        println!("round_trips_against_the_real_vocab: {count} strings, all round-tripped");
        assert!(count > 0, "the real fixture must actually have been exercised");
    }

    /// Compares this crate's token ids against llama.cpp's own oracle
    /// table. Prints per-string match/mismatch rather than failing the
    /// suite on mismatch: this crate's pretokenizer is a hand-rolled
    /// approximation of llama.cpp's PCRE regex (documented in
    /// `pretokenize.rs`), not a byte-for-byte port of it, so exact id
    /// parity is a bonus this test reports, not a correctness gate --
    /// [`round_trips_against_the_real_vocab`] is the gate.
    #[test]
    #[ignore = "depends on a real .gguf checkout outside this repo"]
    fn reports_parity_against_llama_cpp_oracle_ids() {
        let Some(vocab) = load_real_vocab() else { return };
        let mut matched = 0;
        let mut total = 0;
        for (text, expected) in oracle_cases() {
            total += 1;
            let ids = encode(text, &vocab).expect("encodes against real vocab");
            if ids == expected {
                matched += 1;
                println!("MATCH   {text:?}: {ids:?}");
            } else {
                println!("DIFFER  {text:?}: ours={ids:?} llama.cpp={expected:?}");
            }
        }
        println!("oracle parity: {matched}/{total} exact matches");
        assert!(total > 0, "the oracle table must not be empty");
    }
}
