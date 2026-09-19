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

use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};
use proxima_model_interop::qwen35moe::hparams::{Architecture, LayerKind};
use proxima_model_interop::qwen35moe::qwen35moe_forward_program_at_width;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{NumericPolicy, Op, QuantizedBlock, block_node_ids, infer};

const WIDTH: u32 = 13;

/// The checkpoint's own dims (`omega/tests/
/// qwen35_mixer_multi_position_metal_parity.rs`'s `KEY_DIM=2048,
/// VALUE_DIM=4096, KV_HEADS=16, GROUP=2, L_CACHE=4`, plus
/// `attn_head_dim=256, query_heads=16, kv_heads(attention)=2` -- chosen so
/// `query_heads*attn_head_dim*2 == 8192`, the coordinator's "qkv width
/// 8192" for the attention layer's fused Q-gate projection, matching the
/// GDN in-proj's own `qkv_dim = 2*2048+4096 == 8192`). `expert_feed_forward`
/// is pinned to `QK_K` (256) -- the smallest width a `ffn_gate_exps`/
/// `ffn_up_exps` row can be and still be one whole Q4_K block, keeping the
/// MoE small per the brief while every quantized leaf's trailing axis stays
/// `QK_K`-aligned (2048, 4096, 8192, 512, 256 are all multiples of 256).
fn real_dims_architecture(layer_count: u32) -> Architecture {
    let layer_kinds = (0..layer_count)
        .map(|layer| LayerKind::from_interval(layer, 4))
        .collect::<Vec<_>>();
    let kv_heads_by_layer = layer_kinds
        .iter()
        .map(|kind| match kind {
            LayerKind::Gdn => 0,
            LayerKind::Attention => 2,
        })
        .collect();
    Architecture {
        vocab: 32,
        embedding: 2048,
        query_heads: 16,
        kv_heads_by_layer,
        attn_head_dim: 256,
        rope_dims: 64,
        rope_dimension_sections: vec![1],
        rope_mrope_interleaved: false,
        block_count: layer_count,
        full_attention_interval: 4,
        rope_freq_base: 10_000.0,
        rms_epsilon: 1e-6,
        ssm_conv_kernel: 4,
        ssm_state_size: 128,
        ssm_group_count: 16,
        ssm_time_step_rank: 32,
        ssm_inner_size: 4096,
        v_head_reordered: false,
        expert_count: 8,
        expert_used_count: 2,
        expert_feed_forward: 256,
        expert_shared_feed_forward: 256,
        layer_kinds,
    }
}

/// The named leaves the coordinator's brief calls out for Q4_K seeding --
/// GDN in-proj (`attn_qkv`) and its gate (`attn_gate`), the SSM out-proj
/// (`ssm_out`), attention q/k/v/o (`attn_q`/`attn_k`/`attn_v`/
/// `attn_output`), and the routed expert gate/up/down
/// (`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`) -- every other leaf
/// (norms, `ssm_alpha`/`ssm_beta`/`ssm_a`/`ssm_dt`, rope, ids, the small
/// vocab embedding table) stays `Float32`.
fn is_large_projection_leaf(name: &str) -> bool {
    [
        "attn_qkv",
        "attn_gate",
        "ssm_out",
        "attn_q.",
        "attn_k.",
        "attn_v.",
        "attn_output",
        "ffn_gate_exps",
        "ffn_up_exps",
        "ffn_down_exps",
    ]
    .iter()
    .any(|needle| name.contains(needle))
}

/// Packs `values` (row-major, `row_length` elements per row) into Q4_K
/// blocks via `proxima_gguf`'s own real codec (`omega/tests/
/// gdn_sequence_projection_parity.rs`'s established pattern) -- the codec's
/// block layout defines validity, so this never hand-rolls the packed byte
/// format itself.
fn quantize_rows(values: &[f32], row_length: usize) -> Vec<u8> {
    assert_eq!(
        row_length % QK_K,
        0,
        "a quantized leaf's trailing axis must be QK_K-aligned"
    );
    let packed_row_bytes = row_length / QK_K * BLOCK_BYTES;
    let mut packed = vec![0_u8; values.len() / row_length * packed_row_bytes];
    for (row, packed_row) in values
        .chunks_exact(row_length)
        .zip(packed.chunks_exact_mut(packed_row_bytes))
    {
        quantize(row, packed_row).expect("each output row is one valid q4_k block");
    }
    packed
}

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

/// Evaluates every layer's `block_output` on the CPU reference and on Metal
/// for a `layer_count`-layer REAL-dims architecture at `width` positions
/// (`ids = 0..width`), with every [`is_large_projection_leaf`] weight
/// packed as `QuantizedBlock::Q4K` and every other input `Float32` -- both
/// backends bind [`Op::Input`] leaves positionally
/// (`omega::execute`/`proxima_tensor::cpu::evaluate_quantized_exact`'s own
/// doc), so `blocks` is built in [`block_node_ids`]'s own order rather than
/// by name.
fn quantized_layer_parity(layer_count: u32, width: u32) -> Vec<(usize, f32)> {
    let architecture = real_dims_architecture(layer_count);
    let (program, _roots, _layer_roots, _moe_sites, diagnostics) =
        qwen35moe_forward_program_at_width(&architecture, Some(width))
            .expect("the qwen35moe forward program lowers at the real-dims architecture's width");

    let symbols = vec![u64::from(width), 0];
    let shapes = infer(&program, &symbols)
        .expect("real-dims qwen35moe forward program infers its own shapes");

    let node_ids = block_node_ids(&program);
    let leaves: Vec<(String, Vec<f32>, bool)> = node_ids
        .iter()
        .map(|node| {
            let Op::Input { name, .. } = &program[node.0 as usize] else {
                unreachable!("block_node_ids only ever returns Op::Input nodes")
            };
            let name = name
                .clone()
                .expect("every qwen35moe forward-program input is named");
            let count: usize = shapes
                .of(*node)
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
            let quantized = is_large_projection_leaf(&name);
            (name, data, quantized)
        })
        .collect();

    // Owned per-leaf storage (`quantized_storage` for the packed Q4_K
    // leaves) has to outlive `blocks_owned`'s borrows, so it is built first
    // and indexed by the same `leaves` order `blocks_owned` walks below.
    let quantized_storage: Vec<Vec<u8>> = node_ids
        .iter()
        .zip(leaves.iter())
        .filter(|(_, (_, _, quantized))| *quantized)
        .map(|(node, (_, data, _))| {
            let row_length = *shapes
                .of(*node)
                .last()
                .expect("every input has at least one axis") as usize;
            quantize_rows(data, row_length)
        })
        .collect();

    let mut quantized_storage_iter = quantized_storage.iter();
    let blocks_owned: Vec<QuantizedBlock<'_>> = leaves
        .iter()
        .map(|(_, data, quantized)| {
            if *quantized {
                let bytes = quantized_storage_iter
                    .next()
                    .expect("one packed buffer per quantized leaf");
                QuantizedBlock::Packed { codec: Codec::Q4K, bytes: bytes.as_slice() }
            } else {
                QuantizedBlock::Float32(data.as_slice())
            }
        })
        .collect();

    let outputs: Vec<_> = diagnostics.iter().map(|layer| layer.block_output).collect();

    let cpu =
        proxima_tensor::cpu::evaluate_quantized_exact(&program, &symbols, &blocks_owned, &outputs)
            .expect("cpu reference evaluates every layer's block_output on quantized weights");
    let metal = omega::execute(
        &program,
        &symbols,
        &blocks_owned,
        &outputs,
        NumericPolicy::default(),
    )
    .expect("metal evaluates every layer's block_output on quantized weights");

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

/// The one cell the coordinator asked for: the real production program
/// builder at the checkpoint's own dims, 4 layers (>=1 GDN, >=1
/// full-attention, MoE top-2), M=13, with every large projection leaf
/// (SSM/attention in-proj, out-proj, expert gate/up/down) as `Q4_K` rather
/// than `Float32` -- Metal vs the CPU reference, per layer, at the last
/// position.
#[test]
#[ignore = "RED: layer 0 diverges 9.7e-2 at M=13 (layers 1-3 clean, 0e0), and every layer diverges \
            1.0e-1..1.3e-1 at M=1 with the identical Q4_K weights -- the defect reproduces already at \
            a single position, so it is a Q4_K-weight-path Metal/CPU disagreement, not specifically a \
            multi-row one; unresolved, tracked for the qwen35moe checkpoint-prefill bug"]
fn metal_matches_cpu_per_layer_block_output_at_real_dims_with_q4k_weights() {
    let results = quantized_layer_parity(4, WIDTH);
    for (layer_index, relative_error) in &results {
        std::println!(
            "qwen35moe_program_parity_q4k dims=real layers=4 width={WIDTH} layer={layer_index} relative_error={relative_error:e}"
        );
    }
    let failing: Vec<_> = results.iter().filter(|(_, error)| *error > 1e-2).collect();
    if !failing.is_empty() {
        let single_position = quantized_layer_parity(4, 1);
        for (layer_index, relative_error) in &single_position {
            std::println!(
                "qwen35moe_program_parity_q4k dims=real layers=4 width=1 layer={layer_index} relative_error={relative_error:e}"
            );
        }
    }
    assert!(
        failing.is_empty(),
        "metal disagrees with cpu on block_output for layers {failing:?} of the 4-layer real-dims \
         q4k qwen35moe program at M={WIDTH} -- full per-layer table (and the M=1 cross-check, if run) \
         printed above"
    );
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
