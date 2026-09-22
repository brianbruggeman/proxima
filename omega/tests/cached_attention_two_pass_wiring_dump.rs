//! Rootcause round 9, wiring gate 2(b): dumps the qwen35 partial-rotary
//! `BoundOpKind::CachedAttention` op's emitted MSL text (the SAME fixture
//! `cached_attention_partial_rotary_parity.rs` proves CPU/Metal parity
//! against) to a fixed path, so a caller can build this binary once with
//! `metal-fuse-attn-decode` OFF and once ON and diff the two dumps --
//! `two_pass` is never set on a qwen op (Part A's `via_gemma_template`
//! scoping), so the two dumps must be byte-identical.

#![cfg(all(feature = "metal", feature = "cached-attention-streaming", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::PackedOperands;
use proxima_tensor::{BoundOpKind, bind};

mod support;
use support::{production_numeric_policy, qwen35_partial_rotary_forward_fixture};

#[test]
fn dump_qwen35_partial_rotary_op_text() {
    let (program, symbols, roots, _owned) = qwen35_partial_rotary_forward_fixture(1, 40, 37);
    let policy = production_numeric_policy();
    let shapes = proxima_tensor::infer(&program, &symbols).expect("qwen35 partial-rotary fixture infers");
    let resolved = bind(&program, &shapes, &roots, policy).expect("qwen35 partial-rotary fixture binds");
    let fused = resolved
        .iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .expect("the qwen35 chain fuses into a CachedAttention op");

    let kernel = omega::emit(fused, &PackedOperands::new(), policy)
        .unwrap_or_else(|error| panic!("emit failed: {error}"));

    let path = std::env::var("CACHED_ATTENTION_TWO_PASS_WIRING_DUMP_PATH")
        .expect("CACHED_ATTENTION_TWO_PASS_WIRING_DUMP_PATH must be set to the output file path");
    std::fs::write(&path, format!("entry={}\n{}", kernel.entry, kernel.source))
        .unwrap_or_else(|error| panic!("write {path}: {error}"));
    println!("dump_qwen35_partial_rotary_op_text entry={} path={path}", kernel.entry);
}
