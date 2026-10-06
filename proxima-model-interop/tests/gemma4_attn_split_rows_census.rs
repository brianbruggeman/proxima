//! Bind census for the row-tiled attention form on the real gemma4-E2B
//! checkpoint: with `metal-attn-split-rows` on and the production numeric
//! policy (`NumericPolicy::llama_relaxed()`), the speculative-verify program
//! at K = 2 and K = 8 new rows over a 1632-key global bucket (KV about 1615)
//! must bind every attention layer as `BoundOpKind::CachedAttention` (35
//! layers), none as `CachedSoftmaxWeights`, each emitting the row-tiled
//! kernel, and the attention chain must be exactly 70 dispatches (a partial
//! and a merge per layer) where the unfused chain was 13 ops per layer.
//!
//! Header plus shape inference plus bind only: nothing here runs a device, and
//! the multi-GB payload stays demand-paged. `#[ignore]`d because it needs the
//! real blob; [`require_fixture`] fails loudly (naming the env var and the
//! path) instead of returning success having counted nothing.

#![cfg(feature = "metal-attn-split-rows")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;

use memmap2::Mmap;
use omega::{Binding, PackedOperands};
use proxima_gguf::parse_complete;
use proxima_model_interop::{KvLayout, bind_speculative_verify, bind_symbols, symbols};
use proxima_tensor::bind::{BoundOpKind, prune_dead};
use proxima_tensor::spec::Qwen35LayerRoots;
use proxima_tensor::{NodeId, NumericPolicy, bind_with_fusion, infer};

const GEMMA4_E2B_DEFAULT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const ATTENTION_LAYERS: usize = 35;
const GLOBAL_HEAD_DIM: u64 = 512;
const SLIDING_WINDOW: usize = 512;
const GLOBAL_BUCKET: usize = 1632;
const DISPATCHES_PER_LAYER: usize = 2;

/// `(new rows, threadgroups of a global layer, threadgroups of a sliding
/// layer)` of the partial: one per `(row tile, split)` at one kv head. Global:
/// one row per tile, 26 splits of 1634 / 1640 keys against the 256-threadgroup
/// target. Sliding (the 512-key ring, three rows per tile): 9 splits.
const EXPECTED_THREADGROUPS: [(usize, u64, u64); 2] = [(2, 52, 9), (8, 208, 27)];

fn gemma4_e2b_gguf_path() -> String {
    std::env::var("PROXIMA_GEMMA4_E2B_GGUF").unwrap_or_else(|_| GEMMA4_E2B_DEFAULT_PATH.to_string())
}

fn require_fixture(path: &str, env_var: &str) {
    assert!(
        std::path::Path::new(path).exists(),
        "no host-local gguf fixture at {path}: set {env_var} to a valid checkpoint path, or stage one at this default path"
    );
}

fn verify_step_outputs(logits_root: NodeId, layer_roots: &[Qwen35LayerRoots]) -> Vec<NodeId> {
    let mut outputs = vec![logits_root];
    for roots_for_layer in layer_roots {
        if let Qwen35LayerRoots::Attention((even, odd, value)) = roots_for_layer {
            outputs.extend([*even, *odd, *value]);
        }
    }
    outputs
}

#[proxima::test]
#[ignore = "requires the real, local gemma4 E2B GGUF blob (library/gemma4:e2b-it-qat); set PROXIMA_GEMMA4_E2B_GGUF"]
async fn gemma4_e2b_verify_binds_35_row_tiled_attention_ops_in_70_dispatches() {
    let path = gemma4_e2b_gguf_path();
    require_fixture(&path, "PROXIMA_GEMMA4_E2B_GGUF");
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");
    let bound_program = bind_speculative_verify(&parsed, bytes, KvLayout::SlidingRing, None)
        .expect("bind the real gemma4-E2B checkpoint's speculative verify program")
        .expect("gemma4 binds a speculative verify program");
    let outputs = verify_step_outputs(bound_program.logits_root, &bound_program.layer_roots);
    let policy = NumericPolicy::llama_relaxed();
    let mut censused = 0_usize;

    for (rows, global_threadgroups, sliding_threadgroups) in EXPECTED_THREADGROUPS {
        let mut step_symbols = bind_symbols(rows, GLOBAL_BUCKET, &[], false)
            .expect("bind_symbols: a multi-row verify step is legal when the program is not single-position");
        step_symbols[usize::from(symbols::SLIDING_KV_BOUND)] =
            GLOBAL_BUCKET.min(SLIDING_WINDOW) as u64;
        let shapes = infer(&bound_program.program, &step_symbols)
            .expect("shape inference over the real verify program");
        let fused = bind_with_fusion(&bound_program.program, &shapes, &outputs, true, policy)
            .expect("bind_with_fusion (fused) over the real verify program");
        let pruned = prune_dead(fused, &outputs);

        let attention: Vec<_> = pruned
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
            .collect();
        let softmax_weights = pruned
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedSoftmaxWeights { .. }))
            .count();
        println!(
            "gemma4_attn_split_rows_census rows={rows}: cached_attention={} cached_softmax_weights={softmax_weights} bound_ops={}",
            attention.len(),
            pruned.len()
        );
        assert_eq!(
            attention.len(),
            ATTENTION_LAYERS,
            "rows {rows}: CachedAttention count"
        );
        assert_eq!(
            softmax_weights, 0,
            "rows {rows}: CachedSoftmaxWeights count"
        );

        let mut dispatches = 0_usize;
        for bound in attention {
            let BoundOpKind::CachedAttention { head_dim, .. } = &bound.kind else {
                unreachable!("filtered to CachedAttention above");
            };
            let kernel = omega::emit(bound, &PackedOperands::new(), policy)
                .expect("the bound verify attention op emits");
            assert!(
                kernel.entry.ends_with("_rt"),
                "rows {rows} node {}: the op must take the row-tiled form, got {}",
                bound.node.0,
                kernel.entry
            );
            let width = kernel
                .grid
                .threadgroup_width
                .expect("the row-tiled form fixes its threadgroup width");
            let expected = if *head_dim == GLOBAL_HEAD_DIM {
                global_threadgroups
            } else {
                sliding_threadgroups
            };
            assert_eq!(
                kernel.grid.threads / width,
                expected,
                "rows {rows} node {} head_dim {head_dim}: threadgroups",
                bound.node.0
            );
            dispatches += 1 + usize::from(kernel.bindings.contains(&Binding::Scratch));
        }
        assert_eq!(
            dispatches,
            ATTENTION_LAYERS * DISPATCHES_PER_LAYER,
            "rows {rows}: attention dispatches (partial + merge per layer)"
        );
        censused += 1;
    }
    assert_eq!(
        censused,
        EXPECTED_THREADGROUPS.len(),
        "every row count must have been censused"
    );
}
