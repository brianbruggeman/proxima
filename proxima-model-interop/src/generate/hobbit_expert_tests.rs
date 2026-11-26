//! Real routed-checkpoint gate for the HOBBIT expert policy.
//!
//! The full token-level decode is measured by
//! `examples/hobbit_experts.rs`; this gate proves the target model exposes
//! the per-layer routed boundaries the pre-gather path requires. It reads only
//! the directory and binds zero-copy stacks, so it is cheap, and it fails
//! loudly when the blob is absent rather than skipping.

#![cfg(all(test, feature = "std"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::{bind_checkpoint, metadata_str};

#[test]
fn hobbit_expert_real_qwen35moe_exposes_routed_boundaries() {
    let path = std::env::var_os("PROXIMA_QWEN35MOE_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(
                "/Users/brianbruggeman/.ollama/models/blobs/sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d",
            )
        });
    assert!(
        path.is_file(),
        "the HOBBIT expert gate needs the real qwen35moe GGUF at PROXIMA_QWEN35MOE_GGUF or {}; \
         this gate fails loudly rather than skipping",
        path.display()
    );
    let file = std::fs::File::open(&path).expect("open qwen35moe");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("map qwen35moe");
    let parsed = proxima_gguf::pipe::parse_complete(&mapping).expect("parse qwen35moe");
    assert_eq!(
        metadata_str(&parsed, "general.architecture").expect("architecture"),
        "qwen35moe"
    );
    let bound = bind_checkpoint(&parsed, &mapping).expect("bind qwen35moe");
    assert_eq!(
        bound.router_roots.len(),
        40,
        "each qwen35moe layer exposes the router node"
    );
    assert_eq!(
        bound.moe_layer_diagnostics.len(),
        40,
        "each qwen35moe layer exposes the pre-gather boundary"
    );
}

/// A per-layer contiguous low stack is already the ordinary block under the
/// checkpoint name, so the GPU must not build a second high table beside it.
/// A per-expert plan still needs the table because its substitution lives in
/// the slab, not in the named blocks.
#[test]
fn hobbit_universal_gpu_builds_the_table_only_for_a_per_expert_plan() {
    use super::decode::needs_expert_source_table;

    assert!(!needs_expert_source_table(true, false));
    assert!(needs_expert_source_table(true, true));
    assert!(needs_expert_source_table(false, false));
}
