//! gemma4 tokenizer parity against llama.cpp (commit f1ea20621) token ids,
//! vendored under `tests/fixtures/llama-gemma4-tokenize/` (see its README for
//! the exact `llama-tokenize` command). llama.cpp is the oracle for the
//! fixtures only; it is never a runtime dependency. Each test loads the real
//! gemma4 GGUF blob from the Ollama store and prints `SKIP` loudly if the
//! blob is absent on the host.

#![cfg(feature = "gguf")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::OnceLock;

use proxima_tokenizer::{Vocab, decode, encode_with_bos_eos, gguf::vocab_from_metadata};

const BLOB_SHA256: &str = "3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
const BOS_ID: u32 = 2;

fn blob_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME is set");
    PathBuf::from(home).join(format!(".ollama/models/blobs/sha256-{BLOB_SHA256}"))
}

fn load_vocab() -> Option<Vocab> {
    let path = blob_path();
    let Ok(mut file) = File::open(&path) else {
        eprintln!("SKIP gemma4 tokenizer oracle tests: blob absent at {path:?}");
        return None;
    };
    let mut header = Vec::new();
    for capacity in [4usize << 20, 16 << 20, 64 << 20] {
        header.resize(capacity, 0);
        file.seek(SeekFrom::Start(0)).expect("seek to blob start");
        let read = file.read(&mut header).expect("read gguf header region");
        header.truncate(read);
        if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header) {
            return Some(vocab_from_metadata(&parsed).expect("vocab from real gemma4 metadata"));
        }
    }
    panic!("gguf metadata region did not fit in 64 MiB of {path:?}");
}

fn vocab() -> Option<&'static Vocab> {
    static VOCAB: OnceLock<Option<Vocab>> = OnceLock::new();
    VOCAB.get_or_init(load_vocab).as_ref()
}

fn parse_ids(text: &str) -> Vec<u32> {
    text.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|id| id.trim().parse().expect("fixture id parses"))
        .collect()
}

fn assert_matches_llama(vocab: &Vocab, input: &str, expected_ids: &str) {
    let expected = parse_ids(expected_ids);
    let actual = encode_with_bos_eos(input, vocab, true, false).expect("encodes");
    let first_diff = actual.iter().zip(&expected).position(|(left, right)| left != right);
    assert_eq!(
        actual, expected,
        "llama.cpp mismatch; lengths {} vs {}, first differing index {first_diff:?}",
        actual.len(),
        expected.len()
    );
}

macro_rules! oracle_case {
    ($test:ident, $name:literal) => {
        #[test]
        fn $test() {
            let Some(vocab) = vocab() else { return };
            assert_matches_llama(
                vocab,
                include_str!(concat!("fixtures/llama-gemma4-tokenize/", $name, ".txt")),
                include_str!(concat!("fixtures/llama-gemma4-tokenize/", $name, ".ids")),
            );
        }
    };
}

oracle_case!(gemma4_tokenizer_matches_llama_ascii_single_spaces, "ascii_single_spaces");
oracle_case!(gemma4_tokenizer_matches_llama_chat_template_user_turn, "chat_template_user_turn");
oracle_case!(gemma4_tokenizer_matches_llama_cjk, "cjk");
oracle_case!(gemma4_tokenizer_matches_llama_curly_quote_leading_space, "curly_quote_leading_space");
oracle_case!(gemma4_tokenizer_matches_llama_curly_quotes, "curly_quotes");
oracle_case!(gemma4_tokenizer_matches_llama_emoji, "emoji");
oracle_case!(gemma4_tokenizer_matches_llama_latin1_tab_crlf, "latin1_tab_crlf");
oracle_case!(gemma4_tokenizer_matches_llama_newline_double, "newline_double");
oracle_case!(gemma4_tokenizer_matches_llama_newline_mixed_paragraphs, "newline_mixed_paragraphs");
oracle_case!(gemma4_tokenizer_matches_llama_newline_single, "newline_single");
oracle_case!(gemma4_tokenizer_matches_llama_newline_triple_only, "newline_triple_only");
oracle_case!(gemma4_tokenizer_matches_llama_space_runs_edges, "space_runs_edges");
oracle_case!(gemma4_tokenizer_matches_llama_space_runs_inner, "space_runs_inner");
oracle_case!(gemma4_tokenizer_matches_llama_war_and_peace_2000, "war_and_peace_2000");
oracle_case!(gemma4_tokenizer_matches_llama_war_and_peace_30000, "war_and_peace_30000");

#[test]
fn gemma4_tokenizer_vocab_lookups_by_string() {
    let Some(vocab) = vocab() else { return };
    assert_eq!(vocab.token_id("\u{2581}\u{201c}"), Some(999));
    assert_eq!(vocab.token_id("\n"), Some(107));
    assert_eq!(vocab.token_id("\n\n"), Some(108));
    assert_eq!(vocab.bos_token_id(), Some(BOS_ID));
    assert!(vocab.is_char_level_bpe());
}

#[test]
fn gemma4_tokenizer_round_trips_real_text_shapes() {
    let Some(vocab) = vocab() else { return };
    let texts = [
        "a\nb\n\nc\n\n\nd",
        "She said \u{201c}hello\u{201d} and left.",
        "\u{4eca}\u{65e5}\u{306f}\u{5929}\u{6c17}\u{304c}\u{3044}\u{3044}",
        "launch \u{1F680} today \u{1F44D}\u{1F3FD}",
        "caf\u{e9} na\u{ef}ve \u{a9} 2024",
        "col1\tcol2\r\nline two\r\n",
    ];
    for text in texts {
        let ids = encode_with_bos_eos(text, vocab, false, false).expect("encodes");
        assert_eq!(decode(&ids, vocab).expect("decodes"), text, "round trip of {text:?}");
    }
}

#[test]
fn gemma4_tokenizer_decodes_byte_fallback_tokens_to_utf8() {
    let Some(vocab) = vocab() else { return };
    let ids: Vec<u32> = ["<0xC3>", "<0xA9>"]
        .iter()
        .map(|token| vocab.token_id(token).expect("byte fallback token exists"))
        .collect();
    assert_eq!(decode(&ids, vocab).expect("decodes"), "\u{e9}");
}

#[test]
fn gemma4_tokenizer_ascii_single_space_ids_equal_the_pre_fix_gpt2_path() {
    let Some(vocab) = vocab() else { return };
    let pre_fix_gpt2_ids = [2, 818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799, 236761];
    let actual = encode_with_bos_eos("The quick brown fox jumps over the lazy dog.", vocab, true, false)
        .expect("encodes");
    assert_eq!(actual, pre_fix_gpt2_ids);
}

#[test]
fn gemma4_tokenizer_legacy_byte_level_path_still_splits_newlines_into_gpt2_remap_tokens() {
    let Some(vocab) = vocab() else { return };
    let tokens: Vec<String> = (0..vocab.len() as u32)
        .map(|id| vocab.token_str(id).expect("in range").to_string())
        .collect();
    let legacy = Vocab::new(tokens, &[], Some(BOS_ID), None, None).expect("legacy-shaped vocab");
    assert!(!legacy.is_char_level_bpe());
    let ids = encode_with_bos_eos("\n\n", &legacy, true, false).expect("encodes");
    let remapped_newline = legacy.token_id("\u{10a}").expect("gpt2 remap of 0x0A is present");
    assert_eq!(ids, [BOS_ID, remapped_newline, remapped_newline]);
    assert_ne!(ids, parse_ids(include_str!("fixtures/llama-gemma4-tokenize/newline_double.ids")));
}
