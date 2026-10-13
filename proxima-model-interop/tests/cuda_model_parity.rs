//! Model-level CUDA parity: greedy decode on the CUDA backend must produce the same token ids
//! as the CPU path on a real checkpoint. The CPU path is the oracle here, so this is a
//! determinism check between backends, not an external-oracle check (that is
//! `gemma4_correctness_gate.rs`). Fixtures are env-selected like `arch_data_baseline.rs` and
//! skipped when absent like `gemma4_correctness_gate.rs`.

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};

const GEMMA4_E2B_ENV: &str = "PROXIMA_ARCH_GEMMA4_E2B_GGUF";
const GRANITE_MOE_ENV: &str = "PROXIMA_ARCH_GRANITE_MOE_GGUF";

const PROMPT: &str = "The capital of France is";
const TOKENS: usize = 16;

fn config(gpu_layers: i32) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers,
        ..ServingConfig::default()
    }
}

fn assert_cuda_matches_cpu(name: &str, env: &str) {
    let path = std::env::var(env).unwrap_or_else(|_| {
        panic!("{name} checkpoint env var {env} is unset: set it to the path of a {name} gguf")
    });
    assert!(
        Path::new(&path).exists(),
        "{name} checkpoint is missing at {path} (from {env}): stage the gguf there"
    );
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    // SAFETY: the checkpoint file is not written or truncated while this read-only mapping lives.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real checkpoint header");
    let model = LoadedModel::load(&parsed, bytes).expect("bind the real checkpoint");

    let (cpu_ids, cpu_text, _) = model
        .generate_with_serving_config(PROMPT, TOKENS, config(0))
        .unwrap_or_else(|error| panic!("{name} cpu greedy decode failed: {error}"));
    let (cuda_ids, cuda_text, _) = model
        .generate_with_serving_config(PROMPT, TOKENS, config(GPU_LAYERS_ALL))
        .unwrap_or_else(|error| panic!("{name} cuda greedy decode failed: {error}"));

    println!(
        "{name} tokens={} cpu={cpu_ids:?} cuda={cuda_ids:?} cpu_text={cpu_text:?} cuda_text={cuda_text:?}",
        cpu_ids.len()
    );
    assert!(!cpu_ids.is_empty(), "{name}: cpu oracle produced no tokens");
    assert_eq!(
        cuda_ids, cpu_ids,
        "{name}: cuda greedy token ids diverge from the cpu path"
    );
}

#[proxima::test]
async fn cuda_greedy_decode_matches_cpu_on_gemma4_e2b() {
    assert_cuda_matches_cpu("gemma4_e2b", GEMMA4_E2B_ENV);
}

#[proxima::test]
async fn cuda_greedy_decode_matches_cpu_on_granite_moe() {
    assert_cuda_matches_cpu("granite_moe", GRANITE_MOE_ENV);
}
