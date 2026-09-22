//! Byte-equivalence harness for the gemma4 cached-attention recognizer
//! extension (`metal-fuse-attn-decode`): runs real greedy decode on the
//! real gemma4-E2B checkpoint and, on the decode steps named by
//! `PROXIMA_METAL_FUSE_ATTN_PARITY_STEPS`, `decode.rs::run_attn_fuse_parity_probe`
//! builds a fused and an unfused plan for the SAME step's program/outputs
//! and diffs every `BoundOpKind::CachedAttention` node plus the logits root
//! as raw bit patterns before the real (fused) evaluation runs.
//!
//! Copy of `decode_gbps_baseline.rs`'s load/config shape -- this file adds
//! zero new instrumentation, only a runnable entry point that takes its
//! prompt from `PROXIMA_PROMPT` instead of a fixed string, so the same
//! binary drives the short "capital of France" parity check and a longer
//! padded-KV prompt without a recompile.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs::File;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};

const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const DEFAULT_PROMPT: &str = "The capital of France is";
const MAX_TOKENS: usize = 8;

fn main() {
    let prompt = std::env::var("PROXIMA_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string());

    let file = File::open(MODEL_PATH).expect("open gemma4-E2B blob");
    // SAFETY: `file` is dropped at the end of this scope, but the mapping
    // stays valid past that -- POSIX `mmap`/`munmap` semantics, same
    // pattern every real-checkpoint example in this crate already uses.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map gemma4-E2B blob");
    let parsed = parse_complete(&bytes).expect("parse gemma4-E2B header");
    let model = LoadedModel::load(&parsed, &bytes).expect("bind gemma4-E2B");

    let serving_config = ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    };

    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("builds vocab via the current gemma4 dispatch");
    let prompt_token_count = proxima_tokenizer::encode(&prompt, &vocab)
        .expect("tokenize prompt for step accounting")
        .len();
    eprintln!(
        "attn_parity_probe run=start prompt={prompt:?} prompt_token_count={prompt_token_count} max_tokens={MAX_TOKENS}"
    );
    let (token_ids, text, stopped_by_eos) = model
        .generate_with_serving_config(&prompt, MAX_TOKENS, serving_config)
        .expect("greedy decode on the real gemma4-E2B checkpoint");
    eprintln!(
        "attn_parity_probe run=done tokens_generated={} stopped_by_eos={stopped_by_eos} text={text:?}",
        token_ids.len()
    );
}
