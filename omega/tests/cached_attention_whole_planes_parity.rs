//! Metal-vs-CPU parity for the whole-plane cached attention (`ScorePlanes::Whole`): a
//! non-causal self-attention (`proxima_tensor::spec::non_causal_attention_heads`, the shape of
//! whisper's encoder) binds as one fused `CachedAttention` where the backend's `Executes` grants
//! `Form::WholeAttention`. The reference is the UNFUSED chain on the CPU (strict `bit_exact`, no
//! flash narrowing), so the gate measures the per-row Metal kernel against the plain
//! score/softmax/value reduces, not against another fused kernel.
//!
//! Two cells: a small shape that a failure can be read by eye, and whisper's encoder shape
//! (1500 query rows x 1500 keys x 6 heads x head_dim 64). No scalar type is spelled: every
//! element is whatever `Lcg::next_unit` draws and the evaluators accept.

#![cfg(all(
    feature = "metal",
    feature = "cached-attention-streaming",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_numeric::{NumericFormat, NumericPolicy};
use proxima_tensor::bind::ScorePlanes;
use proxima_tensor::config::TensorExecutionConfig;
use proxima_tensor::cpu::{BufferPools, evaluate_quantized_named_with_config};
use proxima_tensor::spec::{input_leaf, non_causal_attention_heads, scalar_constant_like};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{BoundOpKind, Extent, NodeId, Op, bind_with_fusion_for, infer};

mod support;
use support::{as_named_blocks, gpu_oracle_numeric_policy};

fn attention_program(rows: u32, keys: u32, heads: u32, head_dim: u32) -> (Vec<Op>, NodeId) {
    let scale = match head_dim {
        4 => 0.5,
        _ => 0.125,
    };
    let dtype = NumericFormat::host_element()
        .to_dtype()
        .expect("the host element is a scalar dtype");
    let leaf = |program: &mut Vec<Op>, name: &str, extent: [u32; 3]| {
        let extents = extent.iter().map(|&size| Extent::Static(size)).collect();
        input_leaf(program, dtype, extents, name)
    };
    let mut program = Vec::new();
    let query = leaf(&mut program, "q", [rows, heads, head_dim]);
    let key = leaf(&mut program, "k", [keys, heads, head_dim]);
    let value = leaf(&mut program, "v", [keys, heads, head_dim]);
    let inverse_root = scalar_constant_like(&mut program, scale, query).expect("the scale builds");
    let root = non_causal_attention_heads(
        &mut program,
        &[(query, key)],
        value,
        inverse_root,
        "u",
        "i",
        "d",
    )
    .expect("the attention builds");
    (program, root)
}

/// The two policies the metal side runs under: the oracle's bit-exact one keeps the per-row
/// kernel (no reassociation granted), and the serving one grants the split-merge and tree reduce
/// the row-tiled form needs, so the same op runs the simdgroup-matrix kernel.
fn metal_policies() -> [(&'static str, NumericPolicy); 2] {
    [
        ("per-row", gpu_oracle_numeric_policy()),
        (
            "row-tiled",
            NumericPolicy::llama_relaxed().with_flash_narrowing(true),
        ),
    ]
}

fn assert_metal_binds_one_whole_attention(
    program: &[Op],
    root: NodeId,
    policy: NumericPolicy,
) {
    let shapes = infer(program, &[]).expect("the attention infers");
    let resolved = bind_with_fusion_for(
        program,
        &shapes,
        &[root],
        true,
        policy,
        &omega::metal::executes(),
    )
    .expect("the attention binds under metal's grants");
    let whole = resolved
        .iter()
        .filter(|bound| {
            matches!(
                bound.kind,
                BoundOpKind::CachedAttention {
                    planes: ScorePlanes::Whole,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        whole, 1,
        "metal grants Form::WholeAttention, so the chain binds as one whole cached attention: {resolved:?}"
    );
}

fn assert_whole_attention_parity(rows: u32, keys: u32, heads: u32, head_dim: u32) {
    for (kernel, policy) in metal_policies() {
        assert_whole_attention_parity_under(kernel, policy, (rows, keys, heads, head_dim), false);
    }
}

/// Few query rows over many keys: the serving policy picks the split-KV decode form over one
/// head_dim plane, and the oracle policy keeps the per-row kernel.
fn assert_whole_attention_split_parity(rows: u32, keys: u32, heads: u32, head_dim: u32) {
    for (kernel, policy) in metal_policies() {
        assert_whole_attention_parity_under(kernel, policy, (rows, keys, heads, head_dim), true);
    }
}

fn assert_whole_attention_parity_under(
    kernel: &str,
    policy: NumericPolicy,
    (rows, keys, heads, head_dim): (u32, u32, u32, u32),
    decode_split_expected: bool,
) {
    let (program, root) = attention_program(rows, keys, heads, head_dim);
    assert_metal_binds_one_whole_attention(&program, root, policy);

    let mut lcg = Lcg(0x5eed_0683);
    let mut draw = |count: u32| (0..count).map(|_| lcg.next_unit()).collect::<Vec<_>>();
    let owned = vec![
        ("q".to_string(), draw(rows * heads * head_dim)),
        ("k".to_string(), draw(keys * heads * head_dim)),
        ("v".to_string(), draw(keys * heads * head_dim)),
    ];
    let named = as_named_blocks(&owned);

    let config = TensorExecutionConfig::builder()
        .numeric(NumericPolicy::bit_exact())
        .build();
    let mut free_buffers = BufferPools::default();
    let mut validated = None;
    let reference = evaluate_quantized_named_with_config(
        &program,
        &[],
        &named,
        &[root],
        (&mut free_buffers, &mut validated),
        None,
        &config,
    )
    .expect("the cpu evaluates the unfused chain");
    let expected = reference.root();

    let plan = omega::plan_named(&program, &[], &named, &[root], policy)
        .expect("metal plans the whole-plane attention");
    let kernel_names = plan.kernel_keys().expect("the pipeline identities resolve");
    let row_tiled = kernel_names.iter().any(|name| name.contains("_cbrt"));
    // the row-tiled form needs whole 16-wide head-dim fragments, so the 4-wide cell stays per-row
    let admits_row_tiled =
        kernel == "row-tiled" && head_dim.is_multiple_of(16) && !decode_split_expected;
    assert_eq!(
        row_tiled, admits_row_tiled,
        "{kernel}: the whole attention picked the wrong kernel form: {kernel_names:?}"
    );
    let decode_split = kernel_names.iter().any(|name| name.contains("_cbds"));
    assert_eq!(
        decode_split,
        kernel == "row-tiled" && decode_split_expected,
        "{kernel}: the whole attention picked the wrong decode form: {kernel_names:?}"
    );
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the whole-plane attention on a real device");
    let actual = metal.root();

    assert_eq!(
        actual.len(),
        expected.len(),
        "{rows}x{keys}x{heads}x{head_dim}"
    );
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(
            0.0,
            |worst, value| if value > worst { value } else { worst },
        );
    let max_diff = expected
        .iter()
        .zip(actual.iter())
        .map(|(want, got)| (want - got).abs())
        .fold(0.0, |worst, gap| if gap > worst { gap } else { worst });
    assert!(max_magnitude > 0.0, "the reference output is not all zero");
    let relative = max_diff / max_magnitude;
    eprintln!(
        "whole-plane parity ({kernel}): {rows}x{keys}x{heads}x{head_dim} max_diff={max_diff} \
         max_magnitude={max_magnitude} relative={relative}"
    );
    // measured 3.1e-6 at the encoder shape and 1.5e-7 at the small one; 1e-5 is the partial-rotary
    // gate's own bound and leaves a 3x margin over the worst cell
    let tolerance = 1e-5;
    assert!(
        relative < tolerance,
        "{kernel} {rows}x{keys}x{heads}x{head_dim}: metal disagrees with the unfused cpu chain: \
         relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn whole_attention_three_rows_five_keys_three_heads_matches_the_unfused_cpu_chain() {
    assert_whole_attention_parity(3, 5, 3, 4);
}

#[test]
fn whole_attention_at_whisper_encoder_shape_matches_the_unfused_cpu_chain() {
    assert_whole_attention_parity(1500, 1500, 6, 64);
}

#[test]
fn whole_attention_with_rows_that_are_not_a_tile_multiple_matches_the_unfused_cpu_chain() {
    assert_whole_attention_parity(1003, 1003, 6, 64);
}

#[test]
fn whole_attention_with_a_short_ragged_row_tile_matches_the_unfused_cpu_chain() {
    assert_whole_attention_parity(77, 77, 2, 64);
}

#[test]
fn whole_attention_one_row_over_the_encoder_keys_takes_the_decode_split_and_matches_the_unfused_cpu_chain() {
    assert_whole_attention_split_parity(1, 1500, 6, 64);
}

#[test]
fn whole_attention_three_ragged_rows_over_a_short_key_run_takes_the_decode_split_and_matches_the_unfused_cpu_chain() {
    assert_whole_attention_split_parity(3, 211, 3, 40);
}
