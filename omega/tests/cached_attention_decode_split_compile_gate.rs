//! Compile gate for the decode split kernels: the partial at the
//! two gemma4-E2B layer shapes (global head_dim 512, sliding head_dim 256) and
//! the narrow head the parity fixture runs (head_dim 64, where a lane owns
//! fewer float4 than the layout has slots), each handed to the real Metal
//! compiler at every bucket capacity that selects a different split and
//! simdgroup count. The emitter's own tests check the text; only the toolchain
//! checks that the text is Metal (the nested register arrays, the shuffle
//! ladder, the float4 threadgroup array).
//!
//! No device is used. A missing `xcrun`/`metal` fails the test, never skips it.

#![cfg(feature = "metal-attn-split-decode")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::process::Command;

use omega::{Kernel, PackedOperands};
use proxima_tensor::{BoundOp, BoundOpKind, DType, Layout, NodeId, NumericPolicy};

fn decode_attention(head_dim: u64, cached_key_rows: u64, lower: i64) -> BoundOp {
    let layout = Layout {
        base: 0,
        strides: vec![1_i64].into(),
    };
    BoundOp {
        node: NodeId(9),
        dtype: DType::Float32,
        extents: vec![1, 1, 8, head_dim],
        kind: BoundOpKind::CachedAttention {
            operands: (0..9)
                .map(|index| (NodeId(index), layout.clone(), None))
                .collect(),
            query_rows: 1,
            cached_key_rows,
            new_key_rows: 1,
            kv_heads: 1,
            query_groups: 8,
            head_dim,
            rotary_dim: head_dim,
            scale: 1.0,
            cached_lower_inclusive: lower,
            new_upper_inclusive: 0,
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
fn the_decode_split_partial_compiles_with_the_metal_toolchain() {
    let policy = NumericPolicy::llama_relaxed();
    let mut partials = 0_usize;
    for (shape, head_dim, lower) in [
        ("sliding", 256_u64, -511_i64),
        ("global", 512, i64::MIN),
        ("narrow", 64, i64::MIN),
    ] {
        for cached_key_rows in [31_u64, 32, 512, 2048, 4096] {
            let op = decode_attention(head_dim, cached_key_rows, lower);
            let label = format!("{shape} head_dim {head_dim} cached {cached_key_rows}");
            let partial =
                omega::emit(&op, &PackedOperands::new(), policy).expect("the partial emits");
            assert!(partial.entry.ends_with("_ds"), "{label}: {}", partial.entry);
            compile(&label, &partial);
            partials += 1;
        }
    }
    assert_eq!(partials, 15, "3 head dims x 5 bucket capacities");
}

fn with_constant(kernel: &Kernel, name: &str, value: u64) -> Kernel {
    let marker = format!("constexpr short {name} = ");
    let start = kernel
        .source
        .find(&marker)
        .unwrap_or_else(|| panic!("the emitted kernel declares `{marker}`"))
        + marker.len();
    let end = start
        + kernel.source[start..]
            .find(';')
            .expect("a constexpr declaration ends in a semicolon");
    let mut rewritten = kernel.clone();
    rewritten.source = format!("{}{value}{}", &kernel.source[..start], &kernel.source[end..]);
    rewritten
}

/// `[attention_decode]` is a build-time key, so one build compiles one lane
/// layout; the sweep harness (`examples/attn_decode_split_probe.rs`) rewrites
/// the two constants in the emitted text instead. This is the same rewrite
/// over every legal layout, so a layout the defaults do not select is still
/// known to be Metal before a sweep reaches it.
#[test]
fn every_lane_layout_and_load_batch_the_sweep_reaches_compiles() {
    let policy = NumericPolicy::llama_relaxed();
    let mut compiled = 0_usize;
    for (head_dim, cached_key_rows, lower) in [
        (256_u64, 512_u64, -511_i64),
        (512, 2048, i64::MIN),
        (64, 512, i64::MIN),
    ] {
        let op = decode_attention(head_dim, cached_key_rows, lower);
        let base = omega::emit(&op, &PackedOperands::new(), policy).expect("the partial emits");
        for lanes in [8_u64, 16, 32] {
            for batch in [1_u64, 2, 4] {
                let kernel = with_constant(&with_constant(&base, "lanes_per_key", lanes), "batch", batch);
                compile(
                    &format!("head_dim {head_dim} lanes {lanes} batch {batch}"),
                    &kernel,
                );
                compiled += 1;
            }
        }
    }
    assert_eq!(compiled, 27, "3 head dims x 3 lane layouts x 3 load batches");
}
