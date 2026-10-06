//! Pre-tokenizer parity against llama.cpp (commit f1ea20621) token ids for
//! digit-heavy real text, vendored under `tests/fixtures/llama-pre-tokenize/`
//! (see its README for the exact `llama-tokenize` command). llama.cpp is the
//! oracle for the fixtures only; it is never a runtime dependency. Each test
//! loads a real GGUF vocab from the Ollama store (or llama.cpp's vocab-only
//! fixture) and panics when it is absent: a test that cannot load its vocab
//! has asserted nothing.

#![cfg(feature = "gguf")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use proxima_tokenizer::pretokenize::PreType;
use proxima_tokenizer::{Vocab, encode, gguf::vocab_from_metadata};

const LLAMA_BPE_VOCAB: &str = "/Users/brianbruggeman/repos/others/llama.cpp/models/ggml-vocab-llama-bpe.gguf";
const DEEPSEEK_CODER_33B_NO_PRE: &str = "/Users/brianbruggeman/.lmstudio/models/TheBloke/deepseek-coder-33B-instruct-GGUF/deepseek-coder-33b-instruct.Q4_K_S.gguf";
const QWEN2_BLOB_SHA256: &str = "c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515";
const QWEN3_BLOB_SHA256: &str = "a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f";
const QWEN35_BLOB_SHA256: &str = "afb707b6b8fac6e475acc42bc8380fc0b8d2e0e4190be5a969fbf62fcc897db5";
const QWEN35MOE_BLOB_SHA256: &str = "f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d";
const GRANITE_MOE_BLOB_SHA256: &str = "cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";

fn ollama_blob(sha256: &str) -> PathBuf {
    let home = std::env::var("HOME").expect("HOME is set");
    PathBuf::from(home).join(format!(".ollama/models/blobs/sha256-{sha256}"))
}

fn load_vocab(path: &PathBuf) -> Vocab {
    let mut file = File::open(path).unwrap_or_else(|error| panic!("open {path:?}: {error}"));
    let mut header = Vec::new();
    for capacity in [4usize << 20, 16 << 20, 64 << 20] {
        header.resize(capacity, 0);
        file.seek(SeekFrom::Start(0)).expect("seek to gguf start");
        let read = file.read(&mut header).expect("read gguf header region");
        header.truncate(read);
        if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header) {
            return vocab_from_metadata(&parsed).expect("vocab from real gguf metadata");
        }
    }
    panic!("gguf metadata region did not fit in 64 MiB of {path:?}");
}

fn parse_ids(text: &str) -> Vec<u32> {
    text.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|id| id.trim().parse().expect("fixture id parses"))
        .collect()
}

macro_rules! family_cases {
    ($family:literal) => {
        [
            case!($family, "arithmetic_with_contractions"),
            case!($family, "code_with_numbers"),
            case!($family, "combining_marks_with_digits"),
            case!($family, "csv_rows"),
            case!($family, "dates_times"),
            case!($family, "digit_run_lengths"),
            case!($family, "indic_thai_arabic_with_marks"),
            case!($family, "large_and_decimal_numbers"),
            case!($family, "non_ascii_digits"),
            case!($family, "phone_numbers"),
            case!($family, "prices_invoice"),
            case!($family, "roman_numerals_and_circled_symbols"),
            case!($family, "version_strings"),
        ]
    };
}

macro_rules! case {
    ($family:literal, $name:literal) => {
        (
            $name,
            include_str!(concat!("fixtures/llama-pre-tokenize/texts/", $name, ".txt")),
            include_str!(concat!("fixtures/llama-pre-tokenize/", $family, "/", $name, ".ids")),
        )
    };
}

fn assert_family_matches_llama(vocab: &Vocab, expected_pre: PreType, cases: &[(&str, &str, &str)]) {
    assert_eq!(cases.len(), 13, "every vendored text is exercised");
    let mismatched: Vec<&str> = cases
        .iter()
        .filter(|(_, text, expected_ids)| {
            encode(text, vocab).expect("encodes") != parse_ids(expected_ids)
        })
        .map(|(name, _, _)| *name)
        .collect();
    assert!(
        mismatched.is_empty(),
        "ids differ from llama.cpp on {mismatched:?}"
    );
    assert_eq!(vocab.pre_type(), expected_pre, "pre type read from the gguf");
}

#[test]
fn llama3_digits_group_in_threes_like_llama() {
    let vocab = load_vocab(&PathBuf::from(LLAMA_BPE_VOCAB));
    assert_family_matches_llama(&vocab, PreType::GroupedDigits, &family_cases!("llama3"));
}

#[test]
fn qwen2_digits_split_one_per_token_like_llama() {
    let vocab = load_vocab(&ollama_blob(QWEN2_BLOB_SHA256));
    assert_family_matches_llama(&vocab, PreType::SingleDigit, &family_cases!("qwen2"));
}

#[test]
fn qwen3_digits_split_one_per_token_like_llama() {
    let vocab = load_vocab(&ollama_blob(QWEN3_BLOB_SHA256));
    assert_family_matches_llama(&vocab, PreType::SingleDigit, &family_cases!("qwen3"));
}

#[test]
fn qwen35_digits_split_one_per_token_and_marks_join_words_like_llama() {
    let vocab = load_vocab(&ollama_blob(QWEN35_BLOB_SHA256));
    assert_family_matches_llama(&vocab, PreType::SingleDigitMarks, &family_cases!("qwen35"));
}

#[test]
fn qwen35moe_digits_split_one_per_token_and_marks_join_words_like_llama() {
    let vocab = load_vocab(&ollama_blob(QWEN35MOE_BLOB_SHA256));
    assert_family_matches_llama(&vocab, PreType::SingleDigitMarks, &family_cases!("qwen35moe"));
}

#[test]
fn granite_refact_vocabulary_isolates_digits_like_llama() {
    let vocab = load_vocab(&ollama_blob(GRANITE_MOE_BLOB_SHA256));
    assert_family_matches_llama(&vocab, PreType::DigitIsolatedGpt2, &family_cases!("granite_moe"));
}

#[test]
fn deepseek_coder_without_a_pre_key_uses_the_default_split_like_llama() {
    let path = PathBuf::from(DEEPSEEK_CODER_33B_NO_PRE);
    let vocab = load_vocab(&path);
    assert_family_matches_llama(&vocab, PreType::Default, &family_cases!("deepseek_coder_33b_no_pre"));
}

fn host_local_vector_vocab(name: &str) -> PathBuf {
    let dir = std::env::var("LLAMA_CPP_MODELS_DIR")
        .unwrap_or_else(|_| panic!("set LLAMA_CPP_MODELS_DIR to llama.cpp's models/ directory holding ggml-vocab-{name}.gguf"));
    let path = PathBuf::from(dir).join(format!("ggml-vocab-{name}.gguf"));
    assert!(path.exists(), "{path:?} is absent; LLAMA_CPP_MODELS_DIR must name llama.cpp's models/ directory");
    path
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-starcoder.gguf is not vendored"]
fn digit_isolated_gpt2_splits_every_digit_like_llama_starcoder() {
    let vocab = load_vocab(&host_local_vector_vocab("starcoder"));
    assert_family_matches_llama(&vocab, PreType::DigitIsolatedGpt2, &family_cases!("starcoder"));
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-refact.gguf is not vendored"]
fn digit_isolated_gpt2_splits_every_digit_like_llama_refact() {
    let vocab = load_vocab(&host_local_vector_vocab("refact"));
    assert_family_matches_llama(&vocab, PreType::DigitIsolatedGpt2, &family_cases!("refact"));
}

macro_rules! non_ascii_digit_cases {
    ($family:literal) => {
        [
            non_ascii_digit_case!($family, "arabic_indic_digit_runs"),
            non_ascii_digit_case!($family, "ascii_beside_non_ascii_digits"),
            non_ascii_digit_case!($family, "devanagari_digit_runs"),
            non_ascii_digit_case!($family, "fullwidth_digit_runs"),
            non_ascii_digit_case!($family, "superscript_fraction_runs"),
        ]
    };
}

macro_rules! non_ascii_digit_case {
    ($family:literal, $name:literal) => {
        (
            $name,
            include_str!(concat!("fixtures/llama-pre-tokenize/texts_non_ascii_digits/", $name, ".txt")),
            include_str!(concat!("fixtures/llama-pre-tokenize/", $family, "/", $name, ".ids")),
        )
    };
}

fn assert_non_ascii_digit_runs_match_llama(vocab: &Vocab, cases: &[(&str, &str, &str)]) {
    assert_eq!(cases.len(), 5, "every non-ASCII digit text is exercised");
    let mismatched: Vec<&str> = cases
        .iter()
        .filter(|(_, text, expected_ids)| encode(text, vocab).expect("encodes") != parse_ids(expected_ids))
        .map(|(name, _, _)| *name)
        .collect();
    assert!(mismatched.is_empty(), "ids differ from llama.cpp on {mismatched:?}");
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-starcoder.gguf is not vendored"]
fn digit_isolated_gpt2_splits_non_ascii_digit_runs_like_llama_starcoder() {
    let vocab = load_vocab(&host_local_vector_vocab("starcoder"));
    assert_non_ascii_digit_runs_match_llama(&vocab, &non_ascii_digit_cases!("starcoder"));
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-refact.gguf is not vendored"]
fn digit_isolated_gpt2_splits_non_ascii_digit_runs_like_llama_refact() {
    let vocab = load_vocab(&host_local_vector_vocab("refact"));
    assert_non_ascii_digit_runs_match_llama(&vocab, &non_ascii_digit_cases!("refact"));
}
