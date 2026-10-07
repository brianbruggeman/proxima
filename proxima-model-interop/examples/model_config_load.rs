//! A model is a configuration, end to end: load the hand-written granite moe config over the
//! real checkpoint with `LoadedModel::load_with_descriptor`, decode with the serving loop, and
//! compare the ids with the llama.cpp ids vendored under `tests/fixtures/llama-parity`.
//! The `on_token` callback and `SpeculativeDecodeStats` are the loop's observation hooks.
//!
//! Companion to `proxima-tensor/docs/a-model-is-a-configuration.md` and
//! `proxima-tensor/docs/the-serving-loop.md`.
//!
//! Usage (the checkpoint is Ollama's granite3.1-moe:1b blob; override with
//! `PROXIMA_ARCH_GRANITE_MOE_GGUF`):
//! `cargo run -p proxima-model-interop --example model_config_load --features std,metal`

#![allow(clippy::expect_used)]

use core::ops::ControlFlow;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{LoadedModel, Phase, PromptCacheConfig, ServingConfig, SpeculativeDecodeStats};
use proxima_tensor::spec::{ForwardProgram, ModelDescriptor, build_forward};

const DEFAULT_CHECKPOINT: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";

fn fixture(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(relative)
}

fn llama_case() -> (Vec<u32>, Vec<u32>) {
    let text = std::fs::read_to_string(fixture("llama-parity/granite_moe/llama_ids.json")).expect("reads the vendored llama.cpp ids");
    let records: Vec<serde_json::Value> = serde_json::from_str(&text).expect("a json array of records");
    let ids_of = |value: &serde_json::Value| -> Vec<u32> {
        value.as_array().expect("ids are an array").iter().map(|id| u32::try_from(id.as_u64().expect("an id is unsigned")).expect("an id fits u32")).collect()
    };
    (ids_of(&records[0]["prompt_ids"]), ids_of(&records[0]["generated_ids"]))
}

fn main() {
    let checkpoint = std::env::var("PROXIMA_ARCH_GRANITE_MOE_GGUF").unwrap_or_else(|_| DEFAULT_CHECKPOINT.to_owned());
    let file = File::open(&checkpoint).expect("the granite checkpoint is staged");
    let mapping = unsafe { Mmap::map(&file) }.expect("maps the checkpoint read-only");
    let parsed = parse_complete(&mapping).expect("parses the GGUF header");

    let descriptor: ModelDescriptor = conflaguration::from_file(&fixture("model-configs/granite_moe.toml")).expect("the config loads");
    conflaguration::Validate::validate(&descriptor).expect("the config is internally consistent");
    let model = LoadedModel::load_with_descriptor(&parsed, &mapping, &descriptor).expect("the config binds over the checkpoint");

    let ForwardProgram { program, .. } = build_forward(&descriptor).expect("the config lowers");
    assert_eq!(model.op_count(), program.len(), "the loaded model runs the config's lowering");
    println!("loaded: {} ops, the same count build_forward gives the config", model.op_count());

    let (prompt_ids, llama_ids) = llama_case();
    let serving_config = ServingConfig { prompt_cache: PromptCacheConfig::off(), ..ServingConfig::default() };
    let mut phases = Vec::new();
    let mut stats = SpeculativeDecodeStats::default();
    let (generated, _text, _stopped) = model
        .generate_from_ids_with_speculative_stats(
            &prompt_ids,
            llama_ids.len(),
            &serving_config,
            &mut |event| {
                phases.push(event.phase);
                ControlFlow::Continue(())
            },
            &mut stats,
            None,
        )
        .expect("decodes");

    assert!(!generated.is_empty(), "decoded zero tokens");
    assert_eq!(generated, llama_ids, "ids differ from llama.cpp f1ea20621");
    println!("semantic: {} generated ids equal llama.cpp's", generated.len());

    let prefill_events = phases.iter().filter(|phase| matches!(phase, Phase::Prefill { .. })).count();
    let token_events = phases.iter().filter(|phase| matches!(phase, Phase::Token)).count();
    println!("on_token: {prefill_events} prefill event, {token_events} token events");
    assert_eq!(stats.prefill_steps, 1, "one prefill evaluation");
    assert_eq!(stats.decode_steps, 31, "every later token is one single-row decode step");
    println!(
        "serving machine: prefill {} decode {} accept {} rollback {}",
        stats.prefill_steps, stats.decode_steps, stats.accept_steps, stats.rollback_steps
    );

    let mut seen = 0usize;
    let (stopped, _text, _eos) = model
        .generate_from_ids(&prompt_ids, llama_ids.len(), &serving_config, &mut |_event| {
            seen += 1;
            if seen == 3 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
        })
        .expect("decodes");
    println!("on_token Break at the third event: {} ids kept after {seen} events", stopped.len());
}
