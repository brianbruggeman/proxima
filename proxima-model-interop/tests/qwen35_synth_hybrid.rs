//! Landable, file-driven regression for the `qwen35` hybrid (SSM +
//! attention) load/decode path, over a SYNTHETIC fixture --
//! [`synth_qwen35_gguf`]'s own generator (reused here via `#[path]` rather
//! than duplicated, `examples/synth_qwen35_gguf.rs`), since no real
//! Qwen3.8-27B checkpoint exists on this box
//! (`/private/tmp/.../scratchpad/qwen38-path.md`'s leading finding). Random
//! weights of the right shape and quant mix exercise the same graph
//! (`proxima_tensor::spec::qwen35_forward_program`) and the same Metal/CPU
//! dispatch classification a real checkpoint would, since neither depends
//! on weight VALUES.
//!
//! `#[ignore]`d: a debug build of this forward over `embedding=5120,
//! feed_forward=20480` takes minutes, not seconds (measured: a debug CPU
//! decode of 8 tokens did not finish in 6 minutes; `cargo test --release`
//! is required to run this in a reasonable time), the same convention
//! `real_lfm2_checkpoint.rs`/`real_smollm2_checkpoint.rs` use for their own
//! multi-GB fixtures.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

#[path = "../examples/synth_qwen35_gguf.rs"]
mod fixture;

use proxima_gguf::types::GgmlType;
use proxima_gguf::writer::{GgufModel, write_complete};
#[cfg(all(feature = "metal", target_os = "macos"))]
use proxima_model_interop::GPU_LAYERS_ALL;
use proxima_model_interop::{LoadedModel, ServingConfig};

fn build_fixture_bytes() -> Vec<u8> {
    let mut metadata = fixture::architecture_metadata();
    metadata.extend(fixture::tokenizer_metadata());

    let mut tensors = vec![fixture::matmul_tensor(
        "token_embd.weight",
        fixture::EMBEDDING,
        fixture::VOCAB,
        GgmlType::Q6_K,
        1,
    )];
    for layer in 0..fixture::BLOCK_COUNT {
        let is_attention = (layer + 1).is_multiple_of(fixture::FULL_ATTENTION_INTERVAL);
        tensors.extend(fixture::layer_tensors(
            layer,
            is_attention,
            u64::from(layer) * 1000 + 100,
        ));
    }
    tensors.push(fixture::vector_tensor(
        "output_norm.weight",
        fixture::EMBEDDING,
        999_999,
    ));

    let model = GgufModel {
        version: 3,
        metadata,
        tensors,
    };
    write_complete(&model).expect("write synthetic qwen35 gguf")
}

/// `crate::generate::supported_serving_config`'s field set, mirrored by
/// hand (`pub(crate)`, unreachable from an integration test): the only
/// `ServingConfig` this forward implements end to end today
/// (`generate.rs:1497-1513`; `F32` KV storage, no flash-attention kernel,
/// no batching loop, no reasoning-budget split -- every rejection is a
/// real, typed `UnsupportedServingConfig`, none of it qwen35-specific).
fn supported_config(gpu_layers: i32) -> ServingConfig<'static> {
    ServingConfig {
        model_path: "synthetic-qwen35-fixture",
        context_length: 256,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

/// Loads the synthetic fixture and greedy-decodes 2 tokens on CPU --
/// garbage text is expected (LCG-filled weights), this only proves the
/// hybrid attention+state-space graph runs end to end through the real
/// `qwen35` bind + forward-program path
/// (`proxima_model_interop::qwen35::bind_qwen35_weights` /
/// `qwen35_forward_program`), not that its output means anything.
#[test]
#[ignore = "debug build: minutes, not seconds -- run with --release (see module doc)"]
fn qwen35_hybrid_synthetic_fixture_decodes_on_cpu() {
    let file_bytes = build_fixture_bytes();
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parse synthetic qwen35 gguf");
    let loaded = LoadedModel::load(&parsed, &file_bytes).expect("load synthetic qwen35 checkpoint");

    let outcome = loaded.generate_with_serving_config("hi", 2, supported_config(0));

    match outcome {
        Ok((ids, text, _stopped_by_eos)) => {
            println!("cpu: generated_ids={ids:?} generated_text={text:?}");
            assert!(!ids.is_empty(), "at least one token must decode");
        }
        Err(error) => panic!("synthetic qwen35 cpu forward failed: {error}"),
    }
}

/// [`qwen35_hybrid_synthetic_fixture_decodes_on_cpu`]'s Metal counterpart --
/// only compiled when this crate's `metal` feature is on, same gate every
/// other Metal-path test in this crate uses.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "debug build: minutes, not seconds -- run with --release (see module doc)"]
fn qwen35_hybrid_synthetic_fixture_decodes_on_metal() {
    let file_bytes = build_fixture_bytes();
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parse synthetic qwen35 gguf");
    let loaded = LoadedModel::load(&parsed, &file_bytes).expect("load synthetic qwen35 checkpoint");

    let outcome = loaded.generate_with_serving_config("hi", 2, supported_config(GPU_LAYERS_ALL));

    match outcome {
        Ok((ids, text, _stopped_by_eos)) => {
            println!("metal: generated_ids={ids:?} generated_text={text:?}");
            assert!(!ids.is_empty(), "at least one token must decode");
        }
        Err(error) => panic!("synthetic qwen35 metal forward failed: {error}"),
    }
}

/// ROW 531 fix: the full-graph decode arm (`qwen35moe_pre_gather: false`,
/// `PROXIMA_QWEN35MOE_PRE_GATHER=false`) now places recurrent state, conv
/// history and dense-attention KV roots the same way the pre-gather arm
/// already did (`ssm_placement_enabled`/`dense_attention_placement_enabled`,
/// `generate.rs`), so device residency is a Metal-backend property, not a
/// pre-gather-mode property. Runs the full-graph arm twice over the same
/// fixture and asserts bit-identical ids -- placement changes where a root
/// lives, never what the forward computes -- then asserts the readback
/// counter (`omega::metal::READBACK_CALLS`) stayed bounded by the token
/// count once state/KV stop round-tripping through the host every step.
#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#[test]
#[ignore = "debug build: minutes, not seconds -- run with --release (see module doc)"]
fn qwen35_full_graph_metal_places_state_and_bounds_readback() {
    let file_bytes = build_fixture_bytes();
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parse synthetic qwen35 gguf");

    let mut config = supported_config(GPU_LAYERS_ALL);
    config.qwen35moe_pre_gather = false;
    let token_count = 8;

    let loaded_first =
        LoadedModel::load(&parsed, &file_bytes).expect("load synthetic qwen35 checkpoint");
    let _ = omega::metal::metal_stage_totals();
    let (first_ids, _, _) = loaded_first
        .generate_with_serving_config("hi", token_count, config)
        .expect("synthetic qwen35 full-graph metal forward failed (first run)");
    let totals = omega::metal::metal_stage_totals();

    let loaded_second =
        LoadedModel::load(&parsed, &file_bytes).expect("load synthetic qwen35 checkpoint");
    let (second_ids, _, _) = loaded_second
        .generate_with_serving_config("hi", token_count, config)
        .expect("synthetic qwen35 full-graph metal forward failed (second run)");

    assert!(!first_ids.is_empty(), "at least one token must decode");
    assert_eq!(
        first_ids, second_ids,
        "placed full-graph decode must be bit-identical across repeated runs of the same fixture"
    );
    assert!(
        totals.readback_calls <= first_ids.len() as u64,
        "readback_calls={} exceeds the token count ({}); state/KV roots are \
         reading back to host instead of staying device-resident (ROW 531 invariant 2)",
        totals.readback_calls,
        first_ids.len(),
    );
}
