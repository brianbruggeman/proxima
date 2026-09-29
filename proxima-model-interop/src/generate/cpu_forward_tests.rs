#![allow(clippy::expect_used)]

use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;

use crate::LoadedModel;
use crate::serving::ServingConfig;

const PROMPT: &str = "The capital of France is";

const TOKENS: usize = 4;

fn cpu_config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: 0,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

/// The CPU interpreter lowers the same program the GPU planner fuses, so a
/// feature that rewrites the program for a GPU-only op must not reach a
/// `gpu_layers = 0` forward. Built under the production feature set
/// (`std,metal,metal-fuse-attn-decode,identity-copy-alias,metal-tiled-gemm`)
/// this generated `NotLowerable` at the first fused node.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo"]
fn gemma4_cpu_forward_lowers_under_the_production_feature_set() {
    let model_path = crate::test_support::gemma4_e2b_gguf_path();
    crate::test_support::require_fixture(&model_path, Some("PROXIMA_GEMMA4_E2B_GGUF"));
    let file = File::open(&model_path).expect("open the real gemma4-E2B checkpoint");
    // SAFETY: read-only mapping of a file nothing else writes during the test.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B header");
    let model = LoadedModel::load(&parsed, bytes).expect("bind the real gemma4-E2B checkpoint");

    let (token_ids, _text, _stopped_by_eos) = model
        .generate_with_serving_config(PROMPT, TOKENS, cpu_config())
        .expect("a gpu_layers = 0 forward must lower on the CPU interpreter");

    assert!(
        !token_ids.is_empty(),
        "the CPU forward must sample at least one token for {PROMPT:?}"
    );
}
