//! P14 hard parity probe for gemma4-E2B's default-off speculative decode
//! loop: runs the SAME prompt, greedy, 64 tokens, with
//! `PROXIMA_SPECULATIVE_DECODE` off then on, and reports whether the two
//! token id streams are byte-identical -- not library surface, a one-shot
//! diagnostic (same convention as `gemma4_real_weight_parity.rs`).
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::env;
use std::fs::File;
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{LoadedModel, ServingConfig};

fn main() {
    let mut args = env::args().skip(1);
    let model_path = args
        .next()
        .unwrap_or_else(|| "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd".to_string());
    let prompt = args
        .next()
        .unwrap_or_else(|| "Write a detailed history of the Roman Empire:".to_string());
    let max_tokens: usize = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);

    let file = File::open(&model_path).expect("open model");
    // SAFETY: `file` remains alive while the read-only mapping is borrowed.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map model");
    let parsed = parse_complete(&bytes).expect("parse model");
    let model = LoadedModel::load(&parsed, &bytes).expect("bind model");

    let serving_config = ServingConfig {
        gpu_layers: 0,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    };

    unsafe {
        env::remove_var("PROXIMA_SPECULATIVE_DECODE");
    }
    let off_started = Instant::now();
    let (off_ids, _off_text, _off_eos) = model
        .generate_with_serving_config(&prompt, max_tokens, serving_config)
        .expect("OFF decode");
    let off_elapsed = off_started.elapsed();

    unsafe {
        env::set_var("PROXIMA_SPECULATIVE_DECODE", "1");
    }
    let on_started = Instant::now();
    let (on_ids, _on_text, _on_eos) = model
        .generate_with_serving_config(&prompt, max_tokens, serving_config)
        .expect("ON decode");
    let on_elapsed = on_started.elapsed();
    unsafe {
        env::remove_var("PROXIMA_SPECULATIVE_DECODE");
    }

    let first_divergence = off_ids
        .iter()
        .zip(on_ids.iter())
        .position(|(left, right)| left != right);
    let identical = off_ids == on_ids;

    println!("off_ids = {off_ids:?}");
    println!("on_ids  = {on_ids:?}");
    println!("identical = {identical}");
    println!("first_divergence = {first_divergence:?}");
    println!(
        "off_ms_total = {:.3} on_ms_total = {:.3}",
        off_elapsed.as_secs_f64() * 1000.0,
        on_elapsed.as_secs_f64() * 1000.0
    );
    println!(
        "off_ms_per_tok = {:.3} on_ms_per_tok = {:.3}",
        off_elapsed.as_secs_f64() * 1000.0 / off_ids.len().max(1) as f64,
        on_elapsed.as_secs_f64() * 1000.0 / on_ids.len().max(1) as f64
    );
}
