//! Pre-tokenizer parity against the test vectors llama.cpp (commit f1ea20621)
//! ships next to its vocab-only fixtures: `models/ggml-vocab-<name>.gguf` with
//! `<name>.gguf.inp` (cases joined by `\n__ggml_vocab_test__\n`) and
//! `<name>.gguf.out` (one line of token ids per case), parsed the way
//! `tests/test-tokenizer-0.cpp:70-125` parses them and tokenized with no
//! special tokens added.
//!
//! The vectors and the GGUFs are read from the directory named by
//! `LLAMA_CPP_MODELS_DIR` (llama.cpp's `models/`) and the tests that need them
//! stay ignored; they panic with the missing path when run without it. The
//! vectors are not vendored: they hold emoji the repo's commit guard rejects.

#![cfg(feature = "gguf")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use proxima_tokenizer::pretokenize::PreType;
use proxima_tokenizer::{Vocab, encode, gguf::vocab_from_metadata};

const CASE_SEPARATOR: &str = "\n__ggml_vocab_test__\n";
const MODELS_DIR_ENV: &str = "LLAMA_CPP_MODELS_DIR";
const MODELS_DIR_ABSENT: &str = "the llama.cpp models/ directory holding ggml-vocab-*.gguf";

fn host_local_gguf(name: &str) -> PathBuf {
    let dir = std::env::var(MODELS_DIR_ENV)
        .unwrap_or_else(|_| panic!("set {MODELS_DIR_ENV} to {MODELS_DIR_ABSENT}"));
    let path = PathBuf::from(dir).join(format!("ggml-vocab-{name}.gguf"));
    assert!(path.exists(), "{path:?} is absent; {MODELS_DIR_ENV} must name {MODELS_DIR_ABSENT}");
    path
}

fn load_vocab(path: &Path) -> Vocab {
    let mut file = File::open(path).unwrap_or_else(|error| panic!("open {path:?}: {error}"));
    let mut header = Vec::new();
    for capacity in [4usize << 20, 16 << 20, 64 << 20] {
        header.resize(capacity, 0);
        file.seek(SeekFrom::Start(0)).expect("seek to gguf start");
        let read = file.read(&mut header).expect("read gguf header region");
        header.truncate(read);
        if let Ok(parsed) = proxima_gguf::pipe::parse_complete(&header) {
            return vocab_from_metadata(&parsed).expect("vocab from vocab-only gguf metadata");
        }
    }
    panic!("gguf metadata region did not fit in 64 MiB of {path:?}");
}

fn split_cases(raw: &str) -> Vec<&str> {
    let mut cases = Vec::new();
    let mut position = 0;
    while position < raw.len() {
        match raw[position..].find(CASE_SEPARATOR) {
            Some(offset) => {
                cases.push(&raw[position..position + offset]);
                position += offset + CASE_SEPARATOR.len();
            }
            None => {
                cases.push(&raw[position..]);
                break;
            }
        }
    }
    cases
}

fn parse_cases(name: &str) -> Vec<(String, Vec<u32>)> {
    let read = |suffix: &str| {
        let path = host_local_gguf(name).with_extension(format!("gguf.{suffix}"));
        fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path:?}: {error}"))
    };
    let inputs = split_cases(read("inp").leak());
    let outputs: Vec<&str> = read("out").leak().lines().collect();
    assert_eq!(inputs.len(), outputs.len(), "{name}: .inp and .out case counts differ");
    inputs
        .into_iter()
        .zip(outputs)
        .map(|(input, output)| {
            let ids = output
                .split_whitespace()
                .map(|id| id.parse().expect("vector id parses"))
                .collect();
            (input.to_owned(), ids)
        })
        .collect()
}

fn assert_vectors_match_llama(vocab: &Vocab, name: &str, expected_pre: PreType) -> usize {
    assert_eq!(vocab.pre_type(), expected_pre, "{name}: pre type read from the gguf");
    let cases = parse_cases(name);
    assert!(!cases.is_empty(), "{name}: no vector cases were parsed");
    let mismatched: Vec<&str> = cases
        .iter()
        .filter(|(input, expected)| &encode(input, vocab).expect("encodes") != expected)
        .map(|(input, _)| input.as_str())
        .collect();
    assert!(mismatched.is_empty(), "{name}: ids differ from llama.cpp on {mismatched:?}");
    eprintln!("VECTORS {name}: {} cases asserted", cases.len());
    cases.len()
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-gpt-2.gguf is not vendored"]
fn gpt2_word_regex_tokenizes_llama_gpt2_vectors_like_llama() {
    assert_vectors_match_llama(&load_vocab(&host_local_gguf("gpt-2")), "gpt-2", PreType::Gpt2);
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-mpt.gguf is not vendored"]
fn gpt2_word_regex_tokenizes_llama_mpt_vectors_like_llama() {
    assert_vectors_match_llama(&load_vocab(&host_local_gguf("mpt")), "mpt", PreType::Gpt2);
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-starcoder.gguf is not vendored"]
fn digit_isolated_gpt2_tokenizes_llama_starcoder_vectors_like_llama() {
    assert_vectors_match_llama(&load_vocab(&host_local_gguf("starcoder")), "starcoder", PreType::DigitIsolatedGpt2);
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-refact.gguf is not vendored"]
fn digit_isolated_gpt2_tokenizes_llama_refact_vectors_like_llama() {
    assert_vectors_match_llama(&load_vocab(&host_local_gguf("refact")), "refact", PreType::DigitIsolatedGpt2);
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-command-r.gguf is 10 MB, not vendored"]
fn digit_isolated_gpt2_tokenizes_llama_command_r_vectors_like_llama() {
    let vocab = load_vocab(&host_local_gguf("command-r"));
    assert_vectors_match_llama(&vocab, "command-r", PreType::DigitIsolatedGpt2);
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-falcon.gguf is 2.3 MB, not vendored"]
fn falcon_pre_split_tokenizes_llama_falcon_vectors_like_llama() {
    let vocab = load_vocab(&host_local_gguf("falcon"));
    assert_vectors_match_llama(&vocab, "falcon", PreType::Falcon);
}

#[test]
#[ignore = "needs LLAMA_CPP_MODELS_DIR: ggml-vocab-qwen2.gguf is 5.9 MB, not vendored"]
fn qwen2_regex_tokenizes_llama_qwen2_vectors_like_llama() {
    let vocab = load_vocab(&host_local_gguf("qwen2"));
    assert_vectors_match_llama(&vocab, "qwen2", PreType::SingleDigit);
}
