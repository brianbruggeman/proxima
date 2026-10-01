//! Byte-identity gate for the cached-attention kernels: every kernel text
//! below must equal the text `main` emitted before `metal-attn-split-rows`
//! existed. The goldens under `omega/tests/fixtures/attn_split_decode/` are
//! recorded by running [`record_main_goldens`] on an unpatched export of
//! `main` (the recorder uses only `emit` and `emit_cached_attention_merge`,
//! whose signatures did not change), once with no attention feature and once
//! with `metal-attn-split-decode`; `attn_split_tests` compares the patched
//! tree against them, so the same gate covers the feature-off build and the
//! K=1 forms under the row-tiled feature.
//!
//! Self-contained on purpose: the file compiles unchanged against both trees.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use std::path::PathBuf;

use super::*;

pub(super) fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/attn_split_decode")
}

pub(super) fn attention_op(
    operand_count: u32,
    groups: u64,
    head_dim: u64,
    cached_key_rows: u64,
    new_key_rows: u64,
    cached_lower_inclusive: i64,
) -> BoundOp {
    BoundOp {
        node: NodeId(operand_count),
        dtype: DType::Float32,
        extents: vec![1, 1, groups, head_dim],
        kind: BoundOpKind::CachedAttention {
            operands: (0..operand_count)
                .map(|index| {
                    (
                        NodeId(index),
                        Layout {
                            base: 0,
                            strides: vec![1_i64].into(),
                        },
                        None,
                    )
                })
                .collect(),
            query_rows: 1,
            cached_key_rows,
            new_key_rows,
            kv_heads: 1,
            query_groups: groups,
            head_dim,
            rotary_dim: head_dim,
            scale: 1.0,
            cached_lower_inclusive,
            new_upper_inclusive: 0,
        },
    }
}

/// [`attention_op`] at `rows` new query rows with one new key per row, the
/// shape speculative verify binds: eight query heads over one kv head.
#[cfg(feature = "metal-attn-split-rows")]
pub(super) fn attention_rows_op(
    operand_count: u32,
    groups: u64,
    head_dim: u64,
    cached_key_rows: u64,
    rows: u64,
    cached_lower_inclusive: i64,
) -> BoundOp {
    let mut op = attention_op(
        operand_count,
        groups,
        head_dim,
        cached_key_rows,
        rows,
        cached_lower_inclusive,
    );
    op.extents = vec![rows, 1, groups, head_dim];
    let BoundOpKind::CachedAttention { query_rows, .. } = &mut op.kind else {
        unreachable!("attention_op always builds a CachedAttention kind");
    };
    *query_rows = rows;
    op
}

/// File-name prefix of this build's goldens: the decode split changes the
/// text of the cases it serves and the interleaved merge layout, so a build
/// with it compares against goldens recorded with it.
pub(super) const GOLDEN_PREFIX: &str = if cfg!(feature = "metal-attn-split-decode") {
    "main_ds_"
} else {
    "main_"
};

/// Every golden case: its name, the op, and the policy it renders under.
pub(super) fn golden_cases() -> Vec<(&'static str, BoundOp, NumericPolicy)> {
    #[cfg(feature = "metal-attn-split-decode")]
    {
        let mut cases = base_golden_cases();
        cases.extend(decode_split_golden_cases());
        cases
    }
    #[cfg(not(feature = "metal-attn-split-decode"))]
    base_golden_cases()
}

/// The two-range decode op at the three bucket capacities 33/513/2049, global
/// (head_dim 512, unwindowed) and sliding (head_dim 256, window 512).
#[cfg(feature = "metal-attn-split-decode")]
fn decode_split_golden_cases() -> Vec<(&'static str, BoundOp, NumericPolicy)> {
    let relaxed = NumericPolicy::llama_relaxed();
    vec![
        (
            "decode_global_hd512_cap33",
            attention_op(9, 8, 512, 32, 1, i64::MIN),
            relaxed,
        ),
        (
            "decode_global_hd512_cap513",
            attention_op(9, 8, 512, 512, 1, i64::MIN),
            relaxed,
        ),
        (
            "decode_global_hd512_cap2049",
            attention_op(9, 8, 512, 2048, 1, i64::MIN),
            relaxed,
        ),
        (
            "decode_sliding_hd256_cap33",
            attention_op(9, 8, 256, 32, 1, -511),
            relaxed,
        ),
        (
            "decode_sliding_hd256_cap513",
            attention_op(9, 8, 256, 512, 1, -511),
            relaxed,
        ),
        (
            "decode_sliding_hd256_cap2049",
            attention_op(9, 8, 256, 2048, 1, -511),
            relaxed,
        ),
    ]
}

fn base_golden_cases() -> Vec<(&'static str, BoundOp, NumericPolicy)> {
    vec![
        (
            "two_range_sliding_hd256_c512_relaxed",
            attention_op(9, 8, 256, 512, 1, -511),
            NumericPolicy::llama_relaxed(),
        ),
        (
            "two_range_global_hd512_c2048_relaxed",
            attention_op(9, 8, 512, 2048, 1, i64::MIN),
            NumericPolicy::llama_relaxed(),
        ),
        (
            "two_range_sliding_hd256_c512_bit_exact",
            attention_op(9, 8, 256, 512, 1, -511),
            NumericPolicy::bit_exact(),
        ),
        (
            "single_range_dynamic_hd8_relaxed",
            attention_op(9, 1, 8, 0, 256, i64::MIN),
            NumericPolicy::llama_relaxed(),
        ),
        (
            "static_eight_operand_hd64_relaxed",
            attention_op(8, 4, 64, 32, 1, i64::MIN),
            NumericPolicy::llama_relaxed(),
        ),
    ]
}

/// Where a rendered kernel's shared codec preamble ends and the kernel itself
/// begins: every cached-attention kernel declares its `Uniforms` first.
const KERNEL_MARKER: &str = "struct Uniforms";

/// The preamble and the kernel body of one rendered source.
fn split_preamble(source: &str) -> (&str, &str) {
    let marker = source
        .find(KERNEL_MARKER)
        .expect("every cached-attention kernel declares its Uniforms");
    source.split_at(marker)
}

/// The shared preamble every golden kernel renders under, stored once so the
/// per-case goldens stay a few kilobytes of kernel text each.
pub(super) fn preamble_of(op: &BoundOp, policy: NumericPolicy) -> String {
    let kernel = emit(op, &PackedOperands::new(), policy).expect("the golden fixture emits");
    split_preamble(&kernel.source).0.to_string()
}

/// The kernel body, then the merge kernel body when one follows, joined by a
/// marker line so one file pins both. Each is rendered without its preamble;
/// [`preamble_of`] pins that separately and the merge's is asserted equal.
pub(super) fn rendered(op: &BoundOp, policy: NumericPolicy) -> String {
    let kernel = emit(op, &PackedOperands::new(), policy).expect("the golden fixture emits");
    let preamble = split_preamble(&kernel.source).0.to_string();
    let mut text = split_preamble(&kernel.source).1.to_string();
    if let Some(merge) =
        emit_cached_attention_merge(op, policy).expect("the golden fixture's merge emits")
    {
        let (merge_preamble, merge_body) = split_preamble(&merge.source);
        assert_eq!(
            merge_preamble, preamble,
            "the merge renders under the same preamble as its partial"
        );
        text.push_str("\n//// merge\n");
        text.push_str(merge_body);
    }
    text
}

#[test]
#[ignore = "recorder: run on an unpatched export of main to write the goldens the gate compares"]
fn record_main_goldens() {
    let dir = golden_dir();
    std::fs::create_dir_all(&dir).expect("the golden directory is creatable");
    let cases = golden_cases();
    assert!(!cases.is_empty(), "zero golden cases would record nothing");
    let preamble = preamble_of(&cases[0].1, cases[0].2);
    std::fs::write(dir.join(format!("{GOLDEN_PREFIX}preamble.msl")), &preamble)
        .expect("the preamble golden is writable");
    for (name, op, policy) in &cases {
        assert_eq!(
            preamble_of(op, *policy),
            preamble,
            "{name}: one preamble must serve every golden case"
        );
        let path = dir.join(format!("{GOLDEN_PREFIX}{name}.msl"));
        std::fs::write(&path, rendered(op, *policy)).expect("the golden file is writable");
    }
}

/// gemma4-E2B-shaped `CachedSoftmaxWeights` at one query row: eight attention
/// rows, head_dim 256, the cached range at `cached_key_rows`. Layouts are the
/// recognizer's for that op: cached scores keyed `[key, row]`, new scores
/// `[row]`, new value `[row, dim]`.
#[cfg(feature = "metal-wide-cooperative-reduce")]
pub(super) fn softmax_weights_op(cached_key_rows: u64) -> BoundOp {
    let layout = |strides: &[i64]| Layout {
        base: 0,
        strides: strides.into(),
    };
    BoundOp {
        node: NodeId(10),
        dtype: DType::Float32,
        extents: vec![1, cached_key_rows, 1, 8],
        kind: BoundOpKind::CachedSoftmaxWeights {
            operands: vec![
                (NodeId(0), layout(&[8, 1]), None),
                (NodeId(1), layout(&[1]), None),
                (NodeId(2), layout(&[0, 1]), None),
            ],
            cached_weight_sum: NodeId(11),
            new_weight_sum: NodeId(12),
            new_attended: NodeId(13),
            cached_key_rows,
            new_key_rows: 1,
            query_rows: 1,
            attention_rows: 8,
            head_dim: 256,
        },
    }
}

#[cfg(feature = "metal-wide-cooperative-reduce")]
pub(super) fn softmax_weights_golden_cases() -> Vec<(&'static str, BoundOp)> {
    vec![
        ("csw_k1_cached32", softmax_weights_op(32)),
        ("csw_k1_cached512", softmax_weights_op(512)),
        ("csw_k1_cached2048", softmax_weights_op(2048)),
    ]
}

#[cfg(feature = "metal-wide-cooperative-reduce")]
pub(super) fn softmax_weights_rendered(op: &BoundOp) -> String {
    let kernel = emit(op, &PackedOperands::new(), NumericPolicy::bit_exact())
        .expect("the softmax weights golden fixture emits");
    split_preamble(&kernel.source).1.to_string()
}

#[cfg(feature = "metal-wide-cooperative-reduce")]
#[test]
#[ignore = "recorder: run on an unpatched export of main to write the goldens the gate compares"]
fn record_main_softmax_weights_goldens() {
    let dir = golden_dir();
    std::fs::create_dir_all(&dir).expect("the golden directory is creatable");
    let cases = softmax_weights_golden_cases();
    assert!(!cases.is_empty(), "zero golden cases would record nothing");
    for (name, op) in &cases {
        let path = dir.join(format!("main_{name}.msl"));
        std::fs::write(&path, softmax_weights_rendered(op)).expect("the golden file is writable");
    }
}
