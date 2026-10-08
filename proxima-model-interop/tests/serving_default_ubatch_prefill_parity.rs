//! The serving default's prefill chunk against the single pass.
//!
//! `ServingConfig::default().ubatch_size` is [`DEFAULT_UBATCH_SIZE`] (512), at
//! or above `omega`'s `TILED_GEMM_MIN_TOKENS`, so a real prompt reaches the
//! tiled GEMM and row-tiled attention kernels in every chunk. The benches run
//! `ubatch_size: 0`, one evaluation of every row. This file drives a
//! 971-token prompt through `LoadedModel::generate_with_serving_config` (the
//! serving path) both ways and asserts the greedy token ids are equal, on
//! gemma4-E2B (sliding-ring attention, the row-tiled kernel) and granite-moe
//! (expert-grouped tiled GEMM). 971 tokens is two chunks at 512 (512 + 459),
//! both above the 160-row threshold.
//!
//! A missing checkpoint fails with its path and env override; it never skips,
//! so zero models checked cannot read as a pass. Model-loading tests must run
//! with `--test-threads 1` (nextest `-j 1`).

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{
    DEFAULT_UBATCH_SIZE, GPU_LAYERS_ALL, LoadedModel, PromptCacheConfig, ServingConfig,
};

const GEMMA4_E2B_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
const GEMMA4_E2B_ENV: &str = "PROXIMA_ARCH_GEMMA4_E2B_GGUF";
const GRANITE_MOE_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
const GRANITE_MOE_ENV: &str = "PROXIMA_ARCH_GRANITE_MOE_GGUF";

const PROMPT_TOKENS: usize = 971;
const GENERATED_TOKENS: usize = 16;

/// Real Sherlock Holmes prose, the same passage `gemma4_tiled_gemm_defaults_full_logit_vector_diff.rs`
/// and `src/generate/tests_all.rs` use for long prefills.
const PASSAGE: &str = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer\u{2014}excellent for drawing the veil from men's motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.\n";

fn token_count(text: &str, vocab: &proxima_tokenizer::Vocab) -> usize {
    proxima_tokenizer::encode(text, vocab)
        .expect("tokenize the passage with the checkpoint's own vocab")
        .len()
}

/// The shortest word-boundary prefix of the repeated passage that reaches
/// [`PROMPT_TOKENS`] tokens under this checkpoint's own tokenizer.
fn prompt_of_971_tokens(vocab: &proxima_tokenizer::Vocab) -> (String, usize) {
    let corpus = PASSAGE.repeat(PROMPT_TOKENS.div_ceil(40));
    let mut prompt = String::new();
    for word in corpus.split_inclusive(char::is_whitespace) {
        prompt.push_str(word);
        let count = token_count(&prompt, vocab);
        if count >= PROMPT_TOKENS {
            return (prompt, count);
        }
    }
    panic!("the repeated passage never reached {PROMPT_TOKENS} tokens");
}

fn serving_config(ubatch_size: u32) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        reasoning_budget: 0,
        prompt_cache: PromptCacheConfig::off(),
        ubatch_size,
        ..ServingConfig::default()
    }
}

fn assert_default_chunk_matches_single_pass(name: &str, env: &str, default_path: &str) {
    let path = std::env::var(env).unwrap_or_else(|_| default_path.to_string());
    assert!(
        Path::new(&path).exists(),
        "checkpoint {name} is missing at {path}: stage it there or set {env}"
    );
    let file = File::open(&path).expect("open the real checkpoint");
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real checkpoint header");
    let model = LoadedModel::load(&parsed, bytes).expect("bind the real checkpoint");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("build the vocab from the checkpoint metadata");
    let (prompt, prompt_tokens) = prompt_of_971_tokens(&vocab);
    assert!(
        prompt_tokens >= PROMPT_TOKENS,
        "{name}: prompt is {prompt_tokens} tokens, wanted at least {PROMPT_TOKENS}"
    );

    let default_config = ServingConfig::default();
    assert_eq!(default_config.ubatch_size, DEFAULT_UBATCH_SIZE);
    let (default_ids, _text, _stopped) = model
        .generate_with_serving_config(&prompt, GENERATED_TOKENS, serving_config(default_config.ubatch_size))
        .unwrap_or_else(|error| panic!("{name}: default-ubatch generate failed: {error}"));
    let (single_pass_ids, _text, _stopped) = model
        .generate_with_serving_config(&prompt, GENERATED_TOKENS, serving_config(0))
        .unwrap_or_else(|error| panic!("{name}: ubatch-0 generate failed: {error}"));

    assert!(!default_ids.is_empty(), "{name}: generated no ids");
    assert_eq!(
        default_ids, single_pass_ids,
        "{name}: ubatch {} ids differ from the ubatch-0 single pass on a {prompt_tokens}-token prompt",
        default_config.ubatch_size
    );
}

#[proxima::test]
async fn gemma4_e2b_default_ubatch_prefill_ids_equal_the_single_pass_on_971_tokens() {
    assert_default_chunk_matches_single_pass("gemma4_e2b", GEMMA4_E2B_ENV, GEMMA4_E2B_PATH);
}

#[proxima::test]
async fn granite_moe_default_ubatch_prefill_ids_equal_the_single_pass_on_971_tokens() {
    assert_default_chunk_matches_single_pass("granite_moe", GRANITE_MOE_ENV, GRANITE_MOE_PATH);
}
