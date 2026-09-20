//! MEASUREMENT 0a/0b baseline: real gemma4-E2B decode, steady-state
//! wall-clock vs the CLEAN aggregate `metal_stage_totals().gpu_exec_ticks`
//! counter (never the per-dispatch stage-boundary path, which is known to
//! inflate 50-100x on this box -- `gemma4_dispatch_profile_widths.rs`'s own
//! doc names that floor-doubling defect).
//!
//! Reuses the ALREADY-WIRED `PROXIMA_DEBUG_METAL_STAGES` env-var convention
//! (`generate/load_model.rs::emit_token_breakdown`/`emit_token_breakdown_metal`)
//! -- this file adds zero new instrumentation, only a runnable entry point
//! over the same `generate_with_serving_config` call
//! `gemma4_correctness_gate.rs` already exercises on the real checkpoint.
//! Run with `PROXIMA_DEBUG_METAL_STAGES=1` set in the environment; every
//! decode step then emits one `token_breakdown_wall` and one
//! `token_breakdown_metal` line to stderr, parsed by the caller.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs::File;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};

const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const MAX_TOKENS: usize = 48;

fn main() {
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_none() {
        eprintln!(
            "decode_gbps_baseline: PROXIMA_DEBUG_METAL_STAGES not set -- \
             per-step token_breakdown_wall/token_breakdown_metal lines will \
             not be emitted; set it in the environment before running."
        );
    }

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

    // Chat-template prompt (same convention as `gemma4_correctness_gate.rs`)
    // chosen because its locked greedy completion runs the full 46-token
    // budget without an early EOS -- a short factual completion like
    // "The capital of France is" hits EOS after ~5 tokens, too few steps
    // for a steady-state decode average.
    let prompt = "<|turn>user\nWhich of these is smaller in size: a hippopotamus or a large office building?<turn|>\n<|turn>model\n";
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("builds vocab via the current gemma4 dispatch");
    let prompt_token_count = proxima_tokenizer::encode(prompt, &vocab)
        .expect("tokenize prompt for cached_len accounting")
        .len();
    eprintln!(
        "decode_gbps_baseline run=start prompt={prompt:?} prompt_token_count={prompt_token_count} max_tokens={MAX_TOKENS}"
    );
    let (token_ids, text, stopped_by_eos) = model
        .generate_with_serving_config(prompt, MAX_TOKENS, serving_config)
        .expect("greedy decode on the real gemma4-E2B checkpoint");
    eprintln!(
        "decode_gbps_baseline run=done tokens_generated={} stopped_by_eos={stopped_by_eos} text={text:?}",
        token_ids.len()
    );
}
