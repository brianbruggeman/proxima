//! Coverage that was previously ZERO: every existing metal-parity test
//! (`qwen35moe_program_metal_cpu_layer_parity.rs`, `omega/tests/
//! cached_attention_coop_load_parity.rs`, `omega/tests/wgpu_parity.rs`, ...)
//! exercises qwen35moe's GDN/attention schedule or the generic single-RoPE
//! `mistral_cached` engine -- none reference gemma4's real per-layer
//! schedule at all. This builds the REAL forward-program builder
//! (`proxima_tensor::spec::lfm2_forward_program_with_experts`, the exact
//! function [`crate::gemma4::bind::Gemma4Arch::bind`] hands its own
//! `gemma4_layer_schedule` to -- never a second hand-rolled copy of its
//! graph) over a small SYNTHETIC checkpoint that reproduces gemma4's own
//! dual-RoPE schedule: `sliding_window_pattern` alternates sliding/global
//! layers, sliding layers read `rope_cos_swa`/`rope_sin_swa` (freq_base=1e4,
//! dimension_count=256 -- the real checkpoint's `rope.freq_base_swa`/
//! `rope.dimension_count_swa`) and global layers read `rope_cos`/`rope_sin`
//! (freq_base=1e6, dimension_count=512 -- the real checkpoint's
//! `rope.freq_base`/`rope.dimension_count`), matching
//! `gemma4::bind::gemma4_layer_schedule`'s own per-layer `RopeTableSel`/
//! `RopePairing::SplitHalf` wiring (that function and its caller,
//! `Gemma4Arch::bind`, are both crate-private/`std`-gated, so this test
//! replicates the SCHEDULE VALUES inline rather than calling them --
//! `proxima_tensor::spec::tests::gemma4_synthetic_parity_localizes_first_divergence`
//! establishes this is the correct, CPU-proven shape for that schedule; this
//! file adds the Metal side that CPU-only test never had). Both RoPE tables
//! are built by gemma4's own real
//! `proxima_model_interop::gemma4::program::gemma4_sliding_rope_table`
//! function (public, reused verbatim), not a hand-rolled angle formula.
//!
//! `lfm2_forward_program_with_experts` has no per-layer-taps counterpart
//! (unlike qwen35moe's `_at_width`, which returns
//! `Qwen35MoeLayerDiagnostics::block_output` per layer, or `mistral_cached`'s
//! own `_and_layer_taps` twin) -- there is no way to request an
//! intermediate layer's residual without hand-rolling a second copy of the
//! graph up to that point, which this file deliberately does not do. Per-
//! layer localization is instead done the same way
//! `qwen35moe_program_metal_cpu_layer_parity.rs`'s own
//! `metal_cpu_divergence_layer_count_bisection` does it: one whole program
//! build per prefix length (`sliding_pattern[..layer_count]`,
//! `block_count=layer_count`), each build's own final `logits` compared
//! Metal vs CPU, bisecting the smallest layer count at which they diverge.
//!
//! GREEN, not RED: `crate::generate::decode`'s step loop runs `self.program`
//! (`BoundProgram::program`, i.e. exactly `lfm2_forward_program_with_experts`'s
//! own output) directly, with no `mistral_cached`-family substitution for
//! gemma4 anywhere in that file (grep confirmed) -- so the working
//! hypothesis this file set out to check ("gemma4 routes through
//! single-RoPE `mistral_cached`") does not hold at THIS layer: both tests
//! below pass at `relative_error` ~1e-7 (float32 noise floor), every layer
//! count 1..=4, both sliding and global RoPE tables exercised. This is real,
//! previously-absent coverage that the dual-rope schedule composition is
//! Metal-clean at small float32 dims -- it does not rule out a divergence
//! that only appears with real quantized weights (Q3_K/Q5_1 routed
//! experts, Q4_K attention projections -- exactly where
//! `qwen35moe_program_metal_cpu_layer_parity.rs`'s own real-dims-with-Q4K
//! test is the one that stays red while its float32 synthetic sibling
//! passes), which a follow-up item would need to add the same way that file
//! did (`quantize_rows`/`is_large_projection_leaf`) rather than this one
//! guessing at a failure it did not observe.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_model_interop::gemma4::program::gemma4_sliding_rope_table;
use proxima_tensor::spec::{
    Activation, AttentionScoreScale, EmbeddingScale, ExpertGatingFunc, FfnCombination,
    LayerAttentionConfig, LayerFfnConfig, LayerKind, LayerSchedule, ParallelDenseMoeConfig,
    RopePairing, RopeTableSel, ValueSourceKind, lfm2_forward_program_with_experts,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{NumericPolicy, Op, QuantizedBlock, block_node_ids, infer};

const WIDTH: u32 = 5;

const VOCAB: u32 = 6;
const EMBEDDING: u32 = 16;
const FEED_FORWARD: u32 = 8;
const EXPERT_FEED_FORWARD: u32 = 8;
const QUERY_HEADS: u32 = 2;
const KV_HEADS: u32 = 1;
const EXPERT_COUNT: u32 = 4;
const EXPERT_USED_COUNT: u32 = 2;

/// The real checkpoint's own asymmetric head width (`attention_forward.rs`'s
/// own doc on [`LayerAttentionConfig`]: "Gemma 4's sliding (head_dim=256,
/// kv_heads=8) vs full (head_dim=512, kv_heads=2)") -- kept at the real
/// magnitude because it is also the RoPE `dimension_count`/`dimension_count_swa`
/// this checkpoint's metadata carries (`bind.rs`'s own doc: "rope.
/// dimension_count=512 is head_dim, i.e. n_rot"), so shrinking it would stop
/// this test from reproducing the real angle schedule at all. `query_heads`/
/// `kv_heads` are shrunk (every other synthetic gemma4 fixture in this
/// workspace does the same) since GQA group width does not interact with
/// the RoPE dimension.
const HEAD_DIM_FULL: u32 = 512;
const HEAD_DIM_SWA: u32 = 256;
const SLIDING_WINDOW: u32 = 2;

/// `rope.freq_base` / `rope.dimension_count` on the real checkpoint.
const ROPE_FREQ_BASE: f32 = 1.0e6;
/// `rope.freq_base_swa` / `rope.dimension_count_swa` on the real checkpoint.
const ROPE_FREQ_BASE_SWA: f32 = 1.0e4;

/// Alternating sliding/global -- `sliding_window_pattern`'s own real shape
/// (a `bool` per layer, true = sliding), truncated to a prefix by
/// [`layer_parity`] for the bisection sweep.
const SLIDING_PATTERN: [bool; 4] = [true, false, true, false];

/// Replicates `gemma4::bind::gemma4_layer_schedule`'s own per-layer
/// `LayerAttentionConfig`/`LayerFfnConfig` values (that function is
/// crate-private, reachable only from inside `proxima-model-interop`'s own
/// `src/`, never from this external `tests/` binary) at the synthetic dims
/// above, for `sliding_pattern`'s own layers in order -- every field here is
/// copied verbatim from `gemma4::bind::gemma4_layer_schedule`, not
/// reinvented.
fn gemma4_synthetic_schedule(sliding_pattern: &[bool]) -> Vec<LayerSchedule> {
    let ffn = LayerFfnConfig {
        post_attention_norm: true,
        combination: FfnCombination::ParallelDenseMoe(ParallelDenseMoeConfig {
            dense_post_norm: true,
            routed_post_norm: true,
            combined_post_norm: true,
            routed_pre_norm: true,
            router_scale: true,
            expert_output_scale: true,
        }),
        output_scale: true,
        routed_gating: ExpertGatingFunc::Softmax,
        routed_expert_bias: false,
        activation: Activation::GeluTanh,
    };
    sliding_pattern
        .iter()
        .map(|&is_sliding| {
            let attention = if is_sliding {
                LayerAttentionConfig {
                    head_dim: HEAD_DIM_SWA,
                    kv_heads: KV_HEADS,
                    mask_window: Some(SLIDING_WINDOW),
                    value_source_kind: ValueSourceKind::ProjectedV,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos_swa",
                        sin_name: "rope_sin_swa",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: HEAD_DIM_SWA / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                }
            } else {
                LayerAttentionConfig {
                    head_dim: HEAD_DIM_FULL,
                    kv_heads: KV_HEADS,
                    mask_window: None,
                    value_source_kind: ValueSourceKind::SharedWithKey,
                    rope_table: RopeTableSel {
                        cos_name: "rope_cos",
                        sin_name: "rope_sin",
                    },
                    rope_pairing: RopePairing::SplitHalf {
                        pairs: HEAD_DIM_FULL / 2,
                    },
                    score_scale: AttentionScoreScale::Unscaled,
                    value_norm: true,
                }
            };
            LayerSchedule {
                kind: LayerKind::Attention,
                attention,
                ffn,
            }
        })
        .collect()
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// Seeds every [`Op::Input`] leaf in `program` deterministically by node id
/// -- `ids` gets token ids `0..width` (`embedding_lookup`'s own established
/// Float32-fed convention, `qwen35moe_program_metal_cpu_layer_parity.rs`'s
/// own `seed_named_inputs`), `eps` the real serving value, the four RoPE
/// leaves their real angle tables (via gemma4's own
/// [`gemma4_sliding_rope_table`], never a hand-rolled formula here), and
/// every weight/norm leaf an `Lcg`-seeded fill of the right size so this
/// test never has to enumerate gemma4's ~20 per-layer weight names by hand.
fn seed_named_inputs(program: &[Op], symbols: &[u64], positions: &[usize]) -> Vec<(String, Vec<f32>)> {
    let shapes = infer(program, symbols).expect("gemma4 synthetic forward program infers its own shapes");
    let (cos_full, sin_full) = gemma4_sliding_rope_table(positions, ROPE_FREQ_BASE, HEAD_DIM_FULL);
    let (cos_swa, sin_swa) = gemma4_sliding_rope_table(positions, ROPE_FREQ_BASE_SWA, HEAD_DIM_SWA);
    block_node_ids(program)
        .into_iter()
        .map(|node| {
            let Op::Input { name, .. } = &program[node.0 as usize] else {
                unreachable!("block_node_ids only ever returns Op::Input nodes")
            };
            let name = name
                .clone()
                .expect("every gemma4 forward-program input is named");
            let count: usize = shapes
                .of(node)
                .iter()
                .map(|extent| *extent as usize)
                .product();
            let data = match name.as_str() {
                "ids" => (0..count as u32).map(|value| value as f32).collect(),
                "eps" => vec![1e-6_f32; count],
                "rope_cos" => cos_full.clone(),
                "rope_sin" => sin_full.clone(),
                "rope_cos_swa" => cos_swa.clone(),
                "rope_sin_swa" => sin_swa.clone(),
                _ => random_vec(node.0 as u64 + 1, count),
            };
            (name, data)
        })
        .collect()
}

/// Max-abs diff at the last position, relative to that row's own norm --
/// zero when both sides agree, immune to the row's absolute scale varying
/// with `row_length` (`VOCAB` for the logits row this file compares).
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

/// Builds the `layer_count`-layer prefix of [`SLIDING_PATTERN`] through the
/// REAL [`lfm2_forward_program_with_experts`] engine at [`WIDTH`] positions,
/// evaluates its final `logits` on the CPU reference and on Metal, and
/// returns the relative error between them at the last position.
fn logits_relative_error(layer_count: u32) -> f32 {
    let sliding_pattern = &SLIDING_PATTERN[..layer_count as usize];
    let schedule = gemma4_synthetic_schedule(sliding_pattern);
    let (program, logits, _moe_sites) = lfm2_forward_program_with_experts(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        EXPERT_FEED_FORWARD,
        QUERY_HEADS,
        layer_count,
        EXPERT_COUNT,
        EXPERT_USED_COUNT,
        0,
        0,
        &schedule,
        Some(EmbeddingScale::Sqrt),
        None,
        false,
    )
    .expect("the gemma4-shaped forward program lowers at the synthetic architecture's width");

    let symbols = vec![u64::from(WIDTH)];
    let positions: Vec<usize> = (0..WIDTH as usize).collect();
    let named = seed_named_inputs(&program, &symbols, &positions);
    let named_f32: Vec<(&str, &[f32])> = named
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    let named_quantized: Vec<(&str, QuantizedBlock<'_>)> = named
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data.as_slice())))
        .collect();

    let cpu = proxima_tensor::cpu::evaluate_named(&program, &symbols, &named_f32, &[logits])
        .expect("cpu reference evaluates the gemma4 synthetic program's logits");
    let plan = omega::plan_named(
        &program,
        &symbols,
        &named_quantized,
        &[logits],
        NumericPolicy::default(),
    )
    .expect("metal plan builds for the gemma4 synthetic program's logits");
    let metal = omega::execute_plan_named(&plan, &named_quantized)
        .expect("metal evaluates the gemma4 synthetic program's logits");

    let cpu_values = cpu.get(logits).expect("cpu produced logits").0;
    let metal_values = metal.get(logits).expect("metal produced logits").0;
    relative_error_at_last_position(metal_values, cpu_values, VOCAB as usize)
}

/// The one cell this file exists to cover: a small SYNTHETIC gemma4 program
/// with real dual-RoPE dims (sliding `freq_base_swa=1e4, dimension_count_swa=256`,
/// global `freq_base=1e6, dimension_count=512`), 4 layers alternating
/// sliding/global, M=5 -- Metal vs the CPU reference on the final logits.
/// GREEN at ~1e-7 relative error (see this file's own top-level doc for why
/// that is a real, checked result rather than the RED this file set out to
/// confirm).
#[test]
fn metal_matches_cpu_logits_on_four_layer_synthetic_gemma4_program() {
    let relative_error = logits_relative_error(4);
    std::println!("gemma4_program_parity layers=4 width={WIDTH} relative_error={relative_error:e}");
    assert!(
        relative_error <= 1e-3,
        "metal disagrees with cpu on the 4-layer synthetic gemma4 program's logits at M={WIDTH}: \
         relative_error={relative_error:e}"
    );
}

/// Bisects the smallest layer count (1..=4) at which Metal first disagrees
/// with the CPU reference on the gemma4 synthetic program's logits --
/// `qwen35moe_program_metal_cpu_layer_parity.rs`'s own
/// `metal_cpu_divergence_layer_count_bisection`, adapted to gemma4's lack of
/// a per-layer-taps builder (one whole-program build per prefix length
/// instead of one build's per-layer diagnostics).
#[test]
fn metal_cpu_divergence_layer_count_bisection() {
    for layer_count in 1..=4u32 {
        let relative_error = logits_relative_error(layer_count);
        std::println!(
            "gemma4_program_parity layers={layer_count} relative_error={relative_error:e}"
        );
        if relative_error > 1e-3 {
            std::println!(
                "gemma4_program_parity smallest_diverging_layer_count={layer_count} \
                 relative_error={relative_error:e}"
            );
            return;
        }
    }
    std::println!("gemma4_program_parity no divergence found for layer counts 1..=4 at M={WIDTH}");
}
