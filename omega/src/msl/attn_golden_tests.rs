//! Feature-off byte-identity gate for the cached-attention kernels: every
//! kernel text below must equal the text `main` emitted before
//! `metal-attn-split-decode` existed. The goldens under
//! `omega/tests/fixtures/attn_split_decode/` are recorded by running
//! [`record_main_goldens`] on an unpatched export of `main` (the recorder
//! uses only `emit` and `emit_cached_attention_merge`, whose signatures did
//! not change); `attn_split_tests` compares the patched tree against them.
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

/// Every golden case: its name, the op, and the policy it renders under.
pub(super) fn golden_cases() -> Vec<(&'static str, BoundOp, NumericPolicy)> {
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

/// The kernel source, then the merge kernel source when one follows, joined
/// by a marker line so one file pins both.
pub(super) fn rendered(op: &BoundOp, policy: NumericPolicy) -> String {
    let kernel = emit(op, &PackedOperands::new(), policy).expect("the golden fixture emits");
    let mut text = kernel.source;
    if let Some(merge) =
        emit_cached_attention_merge(op, policy).expect("the golden fixture's merge emits")
    {
        text.push_str("\n//// merge\n");
        text.push_str(&merge.source);
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
    for (name, op, policy) in &cases {
        let path = dir.join(format!("main_{name}.msl"));
        std::fs::write(&path, rendered(op, *policy)).expect("the golden file is writable");
    }
}
