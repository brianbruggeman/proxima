//! Compile gate for the K-row attention kernels: the row-tiled partial at the
//! two sliding and global layer shapes, and the K-row `CachedSoftmaxWeights`
//! at the widths its cached range selects, each handed to the real Metal
//! compiler. The emitter's own tests check the text; only the toolchain checks
//! that the text is Metal (`simdgroup_matrix` loads and transposes, threadgroup
//! array sizes, the grid attributes).
//!
//! No device is used. A missing `xcrun`/`metal` fails the test, never skips it.

#![cfg(feature = "metal-attn-split-rows")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use omega::{Kernel, PackedOperands};
use proxima_tensor::{BoundOp, BoundOpKind, DType, Layout, NodeId, NumericPolicy};

fn layout(strides: &[i64]) -> Layout {
    Layout {
        base: 0,
        strides: strides.into(),
    }
}

fn row_tiled_attention(
    kv_heads: u64,
    query_groups: u64,
    head_dim: u64,
    cached_key_rows: u64,
    rows: u64,
    lower: i64,
) -> BoundOp {
    BoundOp {
        node: NodeId(9),
        dtype: DType::Float32,
        extents: vec![rows, kv_heads, query_groups, head_dim],
        kind: BoundOpKind::CachedAttention {
            operands: (0..9)
                .map(|index| (NodeId(index), layout(&[1]), None))
                .collect(),
            query_rows: rows,
            cached_key_rows,
            new_key_rows: rows,
            kv_heads,
            query_groups,
            head_dim,
            rotary_dim: head_dim,
            scale: 1.0,
            cached_lower_inclusive: lower,
            new_upper_inclusive: 0,
        },
    }
}

fn k_row_softmax_weights(rows: u64, cached_key_rows: u64) -> BoundOp {
    BoundOp {
        node: NodeId(10),
        dtype: DType::Float32,
        extents: vec![rows, cached_key_rows, 1, 8],
        kind: BoundOpKind::CachedSoftmaxWeights {
            operands: vec![
                (NodeId(0), layout(&[cached_key_rows as i64 * 8, 8, 1]), None),
                (NodeId(1), layout(&[rows as i64 * 8, 8, 1]), None),
                (NodeId(2), layout(&[256, 0, 1]), None),
            ],
            cached_weight_sum: NodeId(11),
            new_weight_sum: NodeId(12),
            new_attended: NodeId(13),
            cached_key_rows,
            new_key_rows: rows,
            query_rows: rows,
            attention_rows: rows * 8,
            head_dim: 256,
        },
    }
}

fn compile(label: &str, kernel: &Kernel) {
    let directory = tempfile::tempdir().expect("tempdir creation must not fail in ci");
    let metal_path = directory.path().join("kernel.metal");
    let air_path = directory.path().join("kernel.air");
    std::fs::write(&metal_path, &kernel.source).expect("write metal source to a temp file");

    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-c"])
        .arg(&metal_path)
        .arg("-o")
        .arg(&air_path)
        .output()
        .unwrap_or_else(|error| {
            panic!("metal toolchain unavailable ({error}) -- this is a red gate, not a skip")
        });
    assert!(
        output.status.success(),
        "{label}: metal compile failed for entry `{}`:\n--- stderr ---\n{}",
        kernel.entry,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn the_row_tiled_partial_compiles_with_the_metal_toolchain() {
    let policy = NumericPolicy::llama_relaxed();
    let mut compiled = 0_usize;
    for (label, kv_heads, query_groups, head_dim, cached_key_rows, rows, lower) in [
        ("sliding", 1_u64, 8_u64, 256_u64, 512_u64, 17_u64, -511_i64),
        ("global", 1, 8, 512, 2048, 17, i64::MIN),
        ("sliding_one_split", 1, 8, 256, 32, 17, -511),
        ("sliding_prefill", 1, 8, 256, 512, 971, -511),
        ("global_prefill", 1, 8, 512, 992, 971, i64::MIN),
        ("two_group_prefill", 8, 2, 64, 1024, 1000, i64::MIN),
        ("two_group_one_fragment", 8, 2, 64, 32, 8, i64::MIN),
    ] {
        let op = row_tiled_attention(
            kv_heads,
            query_groups,
            head_dim,
            cached_key_rows,
            rows,
            lower,
        );
        let partial = omega::emit(&op, &PackedOperands::new(), policy).expect("the partial emits");
        assert!(partial.entry.ends_with("_rt"), "{label}: {}", partial.entry);
        compile(label, &partial);
        compiled += 1;
    }
    assert_eq!(
        compiled, 7,
        "the two layer shapes, the one-split direct-output store, prefill widths and a query-group size of two"
    );
}

#[test]
fn the_k_row_softmax_weights_kernels_compile_with_the_metal_toolchain() {
    let mut compiled = 0_usize;
    for rows in [2_u64, 49] {
        for cached_key_rows in [32_u64, 512, 2048] {
            let op = k_row_softmax_weights(rows, cached_key_rows);
            let kernel = omega::emit(&op, &PackedOperands::new(), NumericPolicy::bit_exact())
                .expect("the K-row softmax weights kernel emits");
            compile(&format!("rows {rows} cached {cached_key_rows}"), &kernel);
            compiled += 1;
        }
    }
    assert_eq!(compiled, 6, "2 row counts x 3 widths");
}
