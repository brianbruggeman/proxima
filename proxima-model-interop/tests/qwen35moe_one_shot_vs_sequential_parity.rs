//! The Q4_K cell (`qwen35moe_program_metal_cpu_layer_parity.rs`) is moot:
//! the real-checkpoint layer-0 tap sweep gives byte-identical numbers on
//! the CPU engine and on Metal, which clears the executor -- the width-13
//! ONE-SHOT program (`qwen35moe_forward_program_at_width(arch, Some(13))`)
//! must itself compute something different from the SEQUENTIAL 13-step
//! program (`Some(1)`, called once per position with cache threaded between
//! calls) even on a single CPU engine. This is that comparison: checkpoint-
//! free, `f32`, small synthetic layers, CPU only. Every cache leaf threaded
//! between sequential calls mirrors `proxima_model_interop::generate`'s own
//! `SsmLayerCache::advance`/`Qwen35DenseAttentionCache::append` (read there
//! for the exact rows/order this file reproduces without depending on
//! anything private in that module).

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_model_interop::qwen35moe::hparams::{Architecture, LayerKind};
use proxima_model_interop::qwen35moe::{Qwen35MoeLayerDiagnostics, qwen35moe_forward_program_at_width};
use proxima_tensor::spec::Qwen35LayerRoots;
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Op, block_node_ids, infer};

const LAYERS: u32 = 4;
const WIDTH: u32 = 13;

/// `hybrid_moe_program_builds_one_gdn_and_one_attention_layer`'s own small
/// dims (`proxima-model-interop/src/qwen35moe/program.rs`), widened to 4
/// layers/8 experts -- byte-identical to `qwen35moe_program_metal_cpu_layer_
/// parity.rs`'s own `synthetic_architecture`, duplicated here rather than
/// shared across files since this file carries no `metal`/`macos` gate.
fn synthetic_architecture(layer_count: u32) -> Architecture {
    let layer_kinds = (0..layer_count).map(|layer| LayerKind::from_interval(layer, 4)).collect::<Vec<_>>();
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

/// FNV-1a over the leaf's own name -- every weight/norm leaf gets the SAME
/// seed (and so the SAME values) regardless of which of the two programs
/// (one-shot width-13, or one of the 13 sequential width-1 calls) declares
/// the `Op::Input`, since the two programs assign different `NodeId`s to
/// the same-named leaf. This is the one property `layer_parity`'s
/// node-id-seeded fixture (`qwen35moe_program_metal_cpu_layer_parity.rs`)
/// does not need, because that file only ever builds ONE program.
fn seed_for_name(name: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A synthetic, position-dependent row for `rope_cos`/`rope_sin` --
/// deterministic in `(absolute_position, pair_index)` alone (never in which
/// program built the leaf), so row `t` of the one-shot program's `[WIDTH,
/// pairs]` array is bit-identical to the sequential run's own row at
/// absolute position `t`. Values need not be real RoPE angles (this
/// fixture never checks rotary correctness), only that both programs agree
/// on which row means which position.
fn rope_row(is_cos: bool, absolute_position: u32, pairs: usize) -> Vec<f32> {
    (0..pairs)
        .map(|pair| {
            let angle = (absolute_position as f32 + 1.0) * (pair as f32 + 1.0) * 0.037;
            if is_cos { angle.cos() } else { angle.sin() }
        })
        .collect()
}

/// Seeds every `Op::Input` leaf in `program`. `absolute_position` is the
/// index of this call's FIRST new row (`0` for the one-shot width-13
/// program, `t` for the sequential program's `t`-th single-step call) --
/// `ids`/`rope_cos`/`rope_sin` are position-dependent and keyed off it,
/// `cached_len` is a bare scalar equal to it (the prior kv/conv/state
/// history length, `0` on the one-shot program and every sequential call's
/// own `t`), and `kv_cache.*`/`ssm_cache.*` leaves are overwritten by the
/// caller afterward with the actual threaded cache (`cache_overrides`).
fn seed_named_inputs(
    program: &[Op],
    symbols: &[u64],
    absolute_position: u32,
    cache_overrides: &std::collections::HashMap<String, Vec<f32>>,
) -> Vec<(String, Vec<f32>)> {
    let shapes = infer(program, symbols).expect("qwen35moe forward program infers its own shapes");
    block_node_ids(program)
        .into_iter()
        .map(|node| {
            let Op::Input { name, .. } = &program[node.0 as usize] else {
                unreachable!("block_node_ids only ever returns Op::Input nodes")
            };
            let name = name.clone().expect("every qwen35moe forward-program input is named");
            let extents: Vec<usize> = shapes.of(node).iter().map(|extent| *extent as usize).collect();
            let count: usize = extents.iter().product();
            let data = if let Some(overridden) = cache_overrides.get(&name) {
                overridden.clone()
            } else if name == "ids" {
                (0..count as u32).map(|row| (absolute_position + row) as f32).collect()
            } else if name == "eps" {
                vec![1e-6_f32; count]
            } else if name == "cached_len" {
                vec![absolute_position as f32]
            } else if name == "rope_cos" || name == "rope_sin" {
                let pairs = *extents.last().expect("rope tables have a trailing pair axis");
                let width = count / pairs.max(1);
                (0..width)
                    .flat_map(|row| rope_row(name == "rope_cos", absolute_position + row as u32, pairs))
                    .collect()
            } else {
                random_vec(seed_for_name(&name), count)
            };
            (name, data)
        })
        .collect()
}

fn named_f32(named: &[(String, Vec<f32>)]) -> Vec<(&str, &[f32])> {
    named.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect()
}

/// Per-layer cache threaded between sequential single-step calls, one entry
/// per layer matching `Qwen35LayerRoots`'s own per-layer discriminant.
enum LayerCache {
    Ssm { conv_history: Vec<f32>, state: Vec<f32> },
    DenseAttention { k_first: Vec<f32>, k_second: Vec<f32>, k_pass: Vec<f32>, v: Vec<f32> },
}

/// The last `row_length` elements of `data` -- the one-shot program's
/// multi-row output narrowed to the single row a decode-shaped (`M=1`)
/// result can be compared against directly.
fn last_row(data: &[f32], row_length: usize) -> &[f32] {
    &data[data.len() - row_length..]
}

/// Relative-to-row-norm max-abs diff between two EQUAL-length rows.
fn relative_error(found: &[f32], wanted: &[f32]) -> f32 {
    assert_eq!(found.len(), wanted.len(), "relative_error compares two rows of the same width");
    let row_norm: f32 = wanted.iter().map(|value| value * value).sum::<f32>().sqrt().max(1e-6);
    found.iter().zip(wanted.iter()).map(|(actual, expected)| (actual - expected).abs()).fold(0.0, f32::max) / row_norm
}

/// The one-evaluation width-13 program: seeds every leaf at
/// `absolute_position=0` with empty/zero caches (a fresh prefill, matching
/// production's own one-shot call), evaluates every layer's `block_output`
/// plus `logits`, and returns `(per_layer_block_output, logits,
/// layer0_state_out, layer0_qkv_mixed_row12)` for the two extra root
/// comparisons the coordinator asked for.
fn run_one_shot() -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let architecture = synthetic_architecture(LAYERS);
    let (program, roots, layer_roots, _moe_sites, diagnostics) =
        qwen35moe_forward_program_at_width(&architecture, Some(WIDTH))
            .expect("the one-shot qwen35moe forward program lowers");
    let symbols = vec![u64::from(WIDTH), 0];
    let empty_caches = std::collections::HashMap::new();
    let named = seed_named_inputs(&program, &symbols, 0, &empty_caches);
    let blocks = named_f32(&named);

    let mut outputs: Vec<_> = diagnostics.iter().map(|layer: &Qwen35MoeLayerDiagnostics| layer.block_output).collect();
    outputs.push(roots.logits);
    let layer0_ssm_taps = diagnostics[0].ssm_taps.clone().expect("layer 0 is a synthetic GDN layer");
    outputs.push(layer0_ssm_taps.state_out);
    outputs.push(layer0_ssm_taps.qkv_mixed);

    let evaluated = proxima_tensor::cpu::evaluate_named(&program, &symbols, &blocks, &outputs)
        .expect("cpu evaluates the one-shot program's diagnostics and logits");

    let per_layer_block_output: Vec<Vec<f32>> =
        diagnostics.iter().map(|layer| evaluated.get(layer.block_output).expect("block_output produced").0.to_vec()).collect();
    let logits = evaluated.get(roots.logits).expect("logits produced").0.to_vec();
    let state_out = evaluated.get(layer0_ssm_taps.state_out).expect("layer 0 state_out produced").0.to_vec();
    let qkv_mixed = evaluated.get(layer0_ssm_taps.qkv_mixed).expect("layer 0 qkv_mixed produced").0.to_vec();

    assert_eq!(layer_roots.len(), diagnostics.len(), "one layer_roots entry per diagnostics entry");
    (per_layer_block_output, logits, state_out, qkv_mixed)
}

/// The 13-step sequential program: builds `qwen35moe_forward_program_at_
/// width(arch, Some(1))` fresh per step (width never changes across steps,
/// only `symbols`/cache leaves do -- a real decode loop would reuse one
/// resolved plan, but this file is about the PROGRAM's own numbers, not
/// plan-reuse), threads `LayerCache` between steps exactly as
/// `SsmLayerCache::advance`/`Qwen35DenseAttentionCache::append` do, and
/// returns the same four-tuple `run_one_shot` does, PLUS layer 0's final
/// `state_out` and its 13th call's own `qkv_mixed` row for the coordinator's
/// two extra root comparisons.
fn run_sequential() -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let architecture = synthetic_architecture(LAYERS);
    // `ssm_cache.{layer}.conv_history`/`.state` are declared over the FULL
    // per-row width (`qkv_dim`/the recurrent state's own 4 axes), never a
    // bare row count -- `conv_history_len`/`state_len` are element counts,
    // matching `SsmLayerCache`'s own doc ("both lengths come from the
    // program's own declared Op::Input shapes").
    let ssm_key_dim = architecture.ssm_state_size * architecture.ssm_group_count;
    let qkv_dim = 2 * ssm_key_dim + architecture.ssm_inner_size;
    let conv_history_len = ((architecture.ssm_conv_kernel - 1) * qkv_dim) as usize;
    let ssm_group = architecture.ssm_time_step_rank / architecture.ssm_group_count.max(1);
    let head_v_dim = architecture.ssm_inner_size / architecture.ssm_time_step_rank.max(1);
    let state_len = (architecture.ssm_state_size * head_v_dim * architecture.ssm_group_count * ssm_group) as usize;
    let mut caches: Vec<LayerCache> = architecture
        .layer_kinds
        .iter()
        .map(|kind| match kind {
            LayerKind::Gdn => LayerCache::Ssm { conv_history: vec![0.0; conv_history_len], state: vec![0.0; state_len] },
            LayerKind::Attention => {
                LayerCache::DenseAttention { k_first: Vec::new(), k_second: Vec::new(), k_pass: Vec::new(), v: Vec::new() }
            }
        })
        .collect();

    let mut per_layer_block_output: Vec<Vec<f32>> = vec![Vec::new(); LAYERS as usize];
    let mut logits_last = Vec::new();
    let mut layer0_final_state_out = Vec::new();
    let mut layer0_last_qkv_mixed = Vec::new();

    for absolute_position in 0..WIDTH {
        let (program, roots, layer_roots, _moe_sites, diagnostics) =
            qwen35moe_forward_program_at_width(&architecture, Some(1))
                .expect("the sequential step-width-1 qwen35moe forward program lowers");
        let symbols = vec![1u64, u64::from(absolute_position)];

        let mut overrides = std::collections::HashMap::new();
        for (layer, cache) in caches.iter().enumerate() {
            match cache {
                LayerCache::Ssm { conv_history, state } => {
                    overrides.insert(format!("ssm_cache.{layer}.conv_history"), conv_history.clone());
                    overrides.insert(format!("ssm_cache.{layer}.state"), state.clone());
                }
                LayerCache::DenseAttention { k_first, k_second, k_pass, v } => {
                    overrides.insert(format!("kv_cache.{layer}.k_first"), k_first.clone());
                    overrides.insert(format!("kv_cache.{layer}.k_second"), k_second.clone());
                    overrides.insert(format!("kv_cache.{layer}.k_pass"), k_pass.clone());
                    overrides.insert(format!("kv_cache.{layer}.v"), v.clone());
                }
            }
        }
        let named = seed_named_inputs(&program, &symbols, absolute_position, &overrides);
        let blocks = named_f32(&named);

        let mut outputs: Vec<_> = diagnostics.iter().map(|layer| layer.block_output).collect();
        outputs.push(roots.logits);
        for (layer, layer_root) in layer_roots.iter().enumerate() {
            match layer_root {
                Qwen35LayerRoots::Ssm { qkv_mixed, state_out } => {
                    outputs.push(*qkv_mixed);
                    outputs.push(*state_out);
                }
                Qwen35LayerRoots::DenseAttention((first, second, pass, value)) => {
                    outputs.push(*first);
                    outputs.push(*second);
                    outputs.push(*pass);
                    outputs.push(*value);
                }
                Qwen35LayerRoots::Attention(_) => unreachable!("synthetic architecture never uses the even/odd shape"),
            }
            let _ = layer;
        }

        let evaluated = proxima_tensor::cpu::evaluate_named(&program, &symbols, &blocks, &outputs)
            .expect("cpu evaluates this sequential step's diagnostics, logits and cache roots");

        for (layer, layer_diagnostics) in diagnostics.iter().enumerate() {
            per_layer_block_output[layer] = evaluated.get(layer_diagnostics.block_output).expect("block_output produced").0.to_vec();
        }
        logits_last = evaluated.get(roots.logits).expect("logits produced").0.to_vec();

        for (layer, layer_root) in layer_roots.iter().enumerate() {
            match (layer_root, &mut caches[layer]) {
                (Qwen35LayerRoots::Ssm { qkv_mixed, state_out }, LayerCache::Ssm { conv_history, state }) => {
                    let qkv_mixed_new = evaluated.get(*qkv_mixed).expect("qkv_mixed produced").0;
                    let state_new = evaluated.get(*state_out).expect("state_out produced").0;
                    conv_history.extend_from_slice(qkv_mixed_new);
                    let drop = conv_history.len().saturating_sub(conv_history_len);
                    conv_history.copy_within(drop.., 0);
                    conv_history.truncate(conv_history.len() - drop);
                    state.clear();
                    state.extend_from_slice(state_new);
                    if layer == 0 {
                        layer0_final_state_out = state_new.to_vec();
                        layer0_last_qkv_mixed = qkv_mixed_new.to_vec();
                    }
                }
                (
                    Qwen35LayerRoots::DenseAttention((first, second, pass, value)),
                    LayerCache::DenseAttention { k_first, k_second, k_pass, v },
                ) => {
                    k_first.extend_from_slice(evaluated.get(*first).expect("rotated_k_new_first produced").0);
                    k_second.extend_from_slice(evaluated.get(*second).expect("rotated_k_new_second produced").0);
                    k_pass.extend_from_slice(evaluated.get(*pass).expect("k_pass produced").0);
                    v.extend_from_slice(evaluated.get(*value).expect("v_new produced").0);
                }
                _ => unreachable!("layer_roots and caches share the same per-layer discriminant"),
            }
        }
    }

    (per_layer_block_output, logits_last, layer0_final_state_out, layer0_last_qkv_mixed)
}

/// The decisive, checkpoint-free comparison: does the width-13 ONE-SHOT
/// program compute the same thing as 13 sequential width-1 calls with the
/// cache threaded exactly as the production decode loop threads it -- on a
/// single CPU engine, with no Metal/executor involved at all.
#[test]
#[ignore = "RED, checkpoint-free, CPU-only: layer0_qkv_mixed_row12 matches the sequential decode \
            EXACTLY (0e0) but layer 0 block_output already diverges 54% and layer0_state_out 60% -- \
            the M-position mixer's causal-conv/projection stage (qkv_mixed) is correct, but its \
            internal recurrent state scan across the M rows produces a different final state/output \
            than 13 true sequential folds, and that divergence compounds through layers 1-3 (63-82%) \
            and into logits (69%); the width-13 PROGRAM's own SSM recurrence is the first diverging \
            quantity, not the executor. Unresolved, tracked for the qwen35moe checkpoint-prefill bug"]
fn one_shot_program_matches_sequential_decode_on_synthetic_layers() {
    let (one_shot_layers, one_shot_logits, one_shot_state_out, one_shot_qkv_mixed) = run_one_shot();
    let (sequential_layers, sequential_logits, sequential_state_out, sequential_qkv_mixed) = run_sequential();

    let embedding = 8usize;
    let vocab = 16usize;
    let mut first_divergence: Option<String> = None;
    for (layer_index, (one_shot, sequential)) in one_shot_layers.iter().zip(sequential_layers.iter()).enumerate() {
        let error = relative_error(sequential, last_row(one_shot, embedding));
        std::println!("one_shot_vs_sequential layer={layer_index} block_output_relative_error={error:e}");
        if error > 1e-3 && first_divergence.is_none() {
            first_divergence = Some(format!("layer {layer_index} block_output"));
        }
    }
    let logits_error = relative_error(&sequential_logits, last_row(&one_shot_logits, vocab));
    std::println!("one_shot_vs_sequential logits_relative_error={logits_error:e}");
    if logits_error > 1e-3 && first_divergence.is_none() {
        first_divergence = Some("logits".to_owned());
    }

    let state_out_error = relative_error(&sequential_state_out, &one_shot_state_out);
    std::println!("one_shot_vs_sequential layer0_state_out_relative_error={state_out_error:e}");
    let qkv_mixed_error =
        relative_error(&sequential_qkv_mixed, last_row(&one_shot_qkv_mixed, sequential_qkv_mixed.len()));
    std::println!("one_shot_vs_sequential layer0_qkv_mixed_row12_relative_error={qkv_mixed_error:e}");

    assert!(
        first_divergence.is_none(),
        "one-shot width-13 program disagrees with the 13-step sequential decode at {first_divergence:?} \
         (full per-layer/logits/root table printed above) -- CPU-only, no Metal involved, so this is the \
         width-13 PROGRAM computing something different from the sequential program, not an executor bug"
    );
}

