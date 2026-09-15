//! The one-evaluation qwen35moe prefill (M=13, `crate::qwen35moe::
//! qwen35moe_forward_program_at_width`) is wrong on the real checkpoint on
//! Metal from layer 0, yet the isolated M-position mixer at real dims
//! (`omega/tests/qwen35_mixer_multi_position_metal_parity.rs`) and the
//! position-slice op alone are both clean -- so the defect needs the
//! multi-layer graph (buffer-arena slot reuse across layers, or the MoE/
//! attention surround), not one mixer in isolation. This builds the REAL
//! forward-program builder (`qwen35moe_forward_program_at_width`, never a
//! second hand-rolled copy of its graph) over a small SYNTHETIC checkpoint --
//! `hybrid_moe_program_builds_one_gdn_and_one_attention_layer`'s own
//! `Architecture` literal, widened to 4 layers/8 experts so the graph still
//! contains at least one GDN layer, one full-attention layer, and one MoE
//! block -- and compares every layer's `block_output`
//! (`Qwen35MoeLayerDiagnostics::block_output`) between the CPU reference and
//! Metal at M=13 positions.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_model_interop::qwen35moe::hparams::{Architecture, LayerKind};
use proxima_model_interop::qwen35moe::qwen35moe_forward_program_at_width;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{NumericPolicy, Op, QuantizedBlock, block_node_ids, infer};

const WIDTH: u32 = 13;

/// `hybrid_moe_program_builds_one_gdn_and_one_attention_layer`'s own
/// dimensions, widened to `layer_count` blocks (`full_attention_interval=4`
/// so exactly one of every four layers is a real-attention layer -- with
/// `layer_count<4` every layer stays GDN, which is how the layer-count sweep
/// below isolates whether the defect needs a full-attention layer present at
/// all) and 8 experts top-2 so the MoE block's `top_k>1` branch is exercised.
fn synthetic_architecture(layer_count: u32) -> Architecture {
    let layer_kinds = (0..layer_count)
        .map(|layer| LayerKind::from_interval(layer, 4))
        .collect::<Vec<_>>();
    let kv_heads_by_layer = layer_kinds
        .iter()
        .map(|kind| match kind {
            LayerKind::Gdn => 0,
            LayerKind::Attention => 1,
        })
        .collect();
    Architecture {
        vocab: 16,
        embedding: 8,
        query_heads: 2,
        kv_heads_by_layer,
        attn_head_dim: 4,
        rope_dims: 2,
        rope_dimension_sections: vec![1],
        rope_mrope_interleaved: false,
        block_count: layer_count,
        full_attention_interval: 4,
        rope_freq_base: 10_000.0,
        rms_epsilon: 1e-6,
        ssm_conv_kernel: 2,
        ssm_state_size: 2,
        ssm_group_count: 1,
        ssm_time_step_rank: 2,
        ssm_inner_size: 4,
        v_head_reordered: false,
        expert_count: 8,
        expert_used_count: 2,
        expert_feed_forward: 4,
        expert_shared_feed_forward: 4,
        layer_kinds,
    }
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// Seeds every `Op::Input` leaf in `program` deterministically by node id --
/// `ids` gets token ids `0..width` (fed as `Float32`, `embedding_lookup`'s
/// own established convention, see `row_376_cached_attention_batched.rs`),
/// `eps`/`cached_len` get the values a real serving loop would pass, and
/// every weight/rope/state leaf gets an `Lcg`-seeded fill of the right size
/// so this test never has to enumerate qwen35moe's ~20 per-layer weight
/// names by hand.
fn seed_named_inputs(program: &[Op], symbols: &[u64]) -> Vec<(String, Vec<f32>)> {
    let shapes = infer(program, symbols).expect("qwen35moe forward program infers its own shapes");
    block_node_ids(program)
        .into_iter()
        .map(|node| {
            let Op::Input { name, .. } = &program[node.0 as usize] else {
                unreachable!("block_node_ids only ever returns Op::Input nodes")
            };
            let name = name
                .clone()
                .expect("every qwen35moe forward-program input is named");
            let count: usize = shapes
                .of(node)
                .iter()
                .map(|extent| *extent as usize)
                .product();
            let data = if name == "ids" {
                (0..count as u32).map(|value| value as f32).collect()
            } else if name == "eps" {
                vec![1e-6_f32; count]
            } else if name == "cached_len" {
                vec![0.0_f32]
            } else {
                random_vec(node.0 as u64 + 1, count)
            };
            (name, data)
        })
        .collect()
}

/// Max-abs diff at the last position, relative to that row's own norm --
/// zero when both sides agree, and immune to the row's absolute scale
/// varying layer to layer.
fn relative_error_at_last_position(found: &[f32], wanted: &[f32], row_length: usize) -> f32 {
    let start = wanted.len() - row_length;
    let found_row = &found[start..];
    let wanted_row = &wanted[start..];
    let row_norm: f32 = wanted_row
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    found_row
        .iter()
        .zip(wanted_row.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max)
        / row_norm
}

/// Evaluates every layer's `block_output` on the CPU reference and on Metal
/// for a `layer_count`-layer synthetic architecture at `WIDTH` positions,
/// returning `(layer_index, relative_error_at_last_position)` per layer.
fn layer_parity(layer_count: u32) -> Vec<(usize, f32)> {
    let architecture = synthetic_architecture(layer_count);
    let (program, _roots, _layer_roots, _moe_sites, diagnostics) =
        qwen35moe_forward_program_at_width(&architecture, Some(WIDTH))
            .expect("the qwen35moe forward program lowers at the synthetic architecture's width");

    // symbol 0 is the prompt-width axis, pinned static by `Some(WIDTH)` and
    // unread here; symbol 1 is the attention layer's cached-kv length
    // (`kv_cache.{layer}.k_first` et al, `program.rs`'s own `Extent::
    // Symbolic(1)`) -- zero because this is a one-shot prefill with no prior
    // cache, matching the real production one-evaluation prefill call.
    let symbols = vec![u64::from(WIDTH), 0];
    let named = seed_named_inputs(&program, &symbols);
    let named_f32: Vec<(&str, &[f32])> = named
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    let named_quantized: Vec<(&str, QuantizedBlock<'_>)> = named
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data.as_slice())))
        .collect();

    let outputs: Vec<_> = diagnostics.iter().map(|layer| layer.block_output).collect();

    let cpu = proxima_tensor::cpu::evaluate_named(&program, &symbols, &named_f32, &outputs)
        .expect("cpu reference evaluates every layer's block_output");
    let plan = omega::plan_named(
        &program,
        &symbols,
        &named_quantized,
        &outputs,
        NumericPolicy::default(),
    )
    .expect("metal plan builds for every layer's block_output");
    let metal = omega::execute_plan_named(&plan, &named_quantized)
        .expect("metal evaluates every layer's block_output");

    outputs
        .iter()
        .enumerate()
        .map(|(layer_index, node)| {
            let cpu_values = cpu
                .get(*node)
                .expect("cpu produced this layer's block_output")
                .0;
            let metal_values = metal
                .get(*node)
                .expect("metal produced this layer's block_output")
                .0;
            let embedding = architecture.embedding as usize;
            (
                layer_index,
                relative_error_at_last_position(metal_values, cpu_values, embedding),
            )
        })
        .collect()
}

/// The M=13 program at 4 synthetic layers (>=1 GDN, >=1 full-attention,
/// MoE top-2 on every layer) -- Metal must match the CPU reference's
/// `block_output` at every layer, matching what the checkpoint-level bug
/// report says is violated from layer 0 on the real model.
#[test]
fn metal_matches_cpu_per_layer_block_output_on_four_layer_synthetic_program() {
    let results = layer_parity(4);
    for (layer_index, relative_error) in &results {
        std::println!(
            "qwen35moe_program_parity layers=4 layer={layer_index} relative_error={relative_error:e}"
        );
    }
    let failing: Vec<_> = results.iter().filter(|(_, error)| *error > 1e-3).collect();
    assert!(
        failing.is_empty(),
        "metal disagrees with cpu on block_output for layers {failing:?} of the 4-layer synthetic \
         qwen35moe program at M={WIDTH} -- full per-layer table printed above"
    );
}

/// Bisects the smallest layer count (1..=4) at which Metal first disagrees
/// with the CPU reference, to localize the M-position-program-only defect
/// the isolated single-mixer/position-slice fixtures do not reproduce.
#[test]
fn metal_cpu_divergence_layer_count_bisection() {
    for layer_count in 1..=4u32 {
        let results = layer_parity(layer_count);
        for (layer_index, relative_error) in &results {
            std::println!(
                "qwen35moe_program_parity layers={layer_count} layer={layer_index} relative_error={relative_error:e}"
            );
        }
        let first_failure = results.iter().find(|(_, error)| *error > 1e-3);
        if let Some((layer_index, relative_error)) = first_failure {
            std::println!(
                "qwen35moe_program_parity smallest_diverging_layer_count={layer_count} \
                 first_diverging_layer={layer_index} relative_error={relative_error:e}"
            );
            return;
        }
    }
    std::println!(
        "qwen35moe_program_parity no divergence found for layer counts 1..=4 at M={WIDTH}"
    );
}
