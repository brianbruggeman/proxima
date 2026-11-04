#![allow(clippy::expect_used)]

use core::ops::ControlFlow;

use proxima_gguf::parse_complete;

use super::chunked_prefill_tests::dense_checkpoint;
use super::{LoadedModel, LogitsSink, NodeValuesSink, ServingConfig};
use crate::serving::{GPU_LAYERS_ALL, PromptCacheConfig};

const NEW_TOKENS: usize = 3;

fn gpu_config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    }
}

fn cpu_config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: 0,
        prompt_cache: PromptCacheConfig::off(),
        numeric_policy: ServingConfig::default().numeric_policy.with_flash_narrowing(true),
        ..ServingConfig::default()
    }
}

fn prompt_of(width: usize, salt: usize) -> Vec<u32> {
    (0..width).map(|index| ((index * 7 + salt * 13 + 3) % 250 + 1) as u32).collect()
}

struct Decoded {
    logits: Vec<Vec<f32>>,
    repacks: u64,
    kernels: Vec<String>,
}

fn decode(model: &LoadedModel<'_>, prompt: &[u32]) -> Decoded {
    decode_on(model, prompt, gpu_config())
}

fn decode_on(model: &LoadedModel<'_>, prompt: &[u32], config: ServingConfig<'static>) -> Decoded {
    let mut runtime = model.backend_runtime(&config);
    let mut logits = Vec::new();
    let (_, _, _, _) = model
        .run_decode_loop_through_cache(
            prompt.to_vec(),
            &[],
            NEW_TOKENS,
            &config,
            &mut runtime,
            None,
            &mut LogitsSink::Collect(&mut logits),
            &mut NodeValuesSink::Discard,
            &mut |_| ControlFlow::Continue(()),
            None,
            true,
            None,
            None,
        )
        .expect("decode on the tiny dense checkpoint through cuda");
    let repacks = runtime
        .plans
        .values()
        .filter_map(omega::backend::cuda_refit_counters)
        .map(|counters| counters[4])
        .sum();
    let mut kernels: Vec<String> = runtime
        .plans
        .iter()
        .filter(|(key, _)| key.0 == prompt.len())
        .filter_map(|(_, plan)| omega::backend::cuda_kernel_identities(plan))
        .flatten()
        .collect();
    kernels.sort();
    Decoded { logits, repacks, kernels }
}

fn assert_logits_equal(reused: &[Vec<f32>], fresh: &[Vec<f32>], width: usize) {
    assert_eq!(reused.len(), fresh.len(), "width {width}: logit row count");
    for (row, (got, want)) in reused.iter().zip(fresh).enumerate() {
        for (column, (left, right)) in got.iter().zip(want).enumerate() {
            assert!(
                (left - right).abs() <= 1e-4,
                "width {width} logit row {row} column {column}: repacked {left} vs cpu oracle {right}"
            );
        }
    }
}

#[test]
fn a_prefill_width_reached_by_row_repack_matches_an_exact_bind() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the tiny dense GGUF");
    let reused = LoadedModel::load(&parsed, &bytes).expect("load the reused model");
    let widths = [20, 37, 61, 44, 277, 511, 17, 2, 5, 130];
    let mut repacks = 0;

    for (salt, width) in widths.into_iter().enumerate() {
        let prompt = prompt_of(width, salt);
        let fresh_model = LoadedModel::load(&parsed, &bytes).expect("load a fresh model");
        let exact = decode(&fresh_model, &prompt);
        let oracle = decode_on(&fresh_model, &prompt, cpu_config());
        let moved = decode(&reused, &prompt);
        assert_eq!(moved.kernels, exact.kernels, "width {width}: the repacked plan runs the kernels an exact bind emits");
        assert_logits_equal(&moved.logits, &oracle.logits, width);
        repacks += moved.repacks;
    }

    assert!(repacks > 0, "no width was reached by a row repack, so nothing was proven");
}
