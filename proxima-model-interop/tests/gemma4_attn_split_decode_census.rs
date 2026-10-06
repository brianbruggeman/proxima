//! Bind census for the split-KV decode attention form on the real gemma4-E2B
//! checkpoint: with `metal-attn-split-decode` on and the production numeric
//! policy (`NumericPolicy::llama_relaxed()`, `ServingConfig::default()`), the
//! decode program at KV buckets 32, 512 and 2048 must bind every attention
//! layer as `BoundOpKind::CachedAttention` (35 layers) and none as
//! `CachedSoftmaxWeights`, and each op's emitted grid must carry the
//! threadgroup count the partial kernel's `(head, split)` layout implies.
//!
//! Header plus shape inference plus bind only: nothing here runs a device, and
//! the multi-GB payload stays demand-paged. `#[ignore]`d because it needs the
//! real blob; [`require_fixture`] fails loudly (naming the env var and the
//! path) instead of returning success having counted nothing.

#![cfg(feature = "metal-attn-split-decode")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;

use memmap2::Mmap;
use omega::PackedOperands;
use proxima_gguf::parse_complete;
use proxima_model_interop::{KvLayout, bind_checkpoint_with_kv_layout, bind_symbols, symbols};
use proxima_tensor::bind::{BoundOpKind, prune_dead};
use proxima_tensor::spec::LayerCacheRoots;
use proxima_tensor::{DType, NodeId, NumericPolicy, bind_with_fusion, infer};

const GEMMA4_E2B_DEFAULT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const ATTENTION_LAYERS: usize = 35;
const GLOBAL_HEAD_DIM: u64 = 512;
const SLIDING_WINDOW: usize = 512;

/// `(kv bucket, threadgroups of a global layer, threadgroups of a sliding
/// layer, threads per global threadgroup, threads per sliding threadgroup)`:
/// eight query heads times the split count, and 32 lanes times the simdgroups
/// `[attention_decode].keys_per_simdgroup` spreads a split over. The sliding
/// ring caps its extent at the 512-key window, so its split count stops at 17
/// and its simdgroup count at the four an 8-keys-per-simdgroup target gives a 31-key split.
const EXPECTED_LAUNCH: [(usize, u64, u64, u64, u64); 3] = [
    (32, 16, 16, 96, 96),
    (512, 136, 136, 128, 128),
    (2048, 256, 136, 128, 128),
];

fn gemma4_e2b_gguf_path() -> String {
    std::env::var("PROXIMA_GEMMA4_E2B_GGUF").unwrap_or_else(|_| GEMMA4_E2B_DEFAULT_PATH.to_string())
}

fn require_fixture(path: &str, env_var: &str) {
    assert!(
        std::path::Path::new(path).exists(),
        "no host-local gguf fixture at {path}: set {env_var} to a valid checkpoint path, or stage one at this default path"
    );
}

fn production_step_outputs(logits_root: NodeId, layer_roots: &[LayerCacheRoots]) -> Vec<NodeId> {
    let mut outputs = vec![logits_root];
    for roots_for_layer in layer_roots {
        if let LayerCacheRoots::Attention((even, odd, value)) = roots_for_layer {
            outputs.extend([*even, *odd, *value]);
        }
    }
    outputs
}

#[proxima::test]
#[ignore = "requires the real, local gemma4 E2B GGUF blob (library/gemma4:e2b-it-qat); set PROXIMA_GEMMA4_E2B_GGUF"]
async fn gemma4_e2b_decode_binds_35_cached_attention_ops_and_no_softmax_weights_at_every_bucket() {
    let path = gemma4_e2b_gguf_path();
    require_fixture(&path, "PROXIMA_GEMMA4_E2B_GGUF");
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");
    let bound_program = bind_checkpoint_with_kv_layout(&parsed, bytes, KvLayout::SlidingRing)
        .expect("bind the real gemma4-E2B checkpoint's production decode program");
    let outputs = production_step_outputs(bound_program.logits_root, &bound_program.layer_roots);
    let policy = NumericPolicy::llama_relaxed();

    for (bucket, global_threadgroups, sliding_threadgroups, global_width, sliding_width) in
        EXPECTED_LAUNCH
    {
        let mut step_symbols = bind_symbols(1, bucket, &[], true).expect(
            "bind_symbols: single_position_step=true with one new token is a legal decode step",
        );
        step_symbols[usize::from(symbols::SLIDING_KV_BOUND)] = bucket.min(SLIDING_WINDOW) as u64;
        let shapes = infer(&bound_program.program, &step_symbols)
            .expect("shape inference over the real program");
        let fused = bind_with_fusion(&bound_program.program, &shapes, &outputs, true, policy)
            .expect("bind_with_fusion (fused) over the real decode program");
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
            "gemma4_attn_split_decode_census bucket={bucket}: cached_attention={} cached_softmax_weights={softmax_weights} bound_ops={}",
            attention.len(),
            pruned.len()
        );
        assert_eq!(
            attention.len(),
            ATTENTION_LAYERS,
            "bucket {bucket}: CachedAttention count"
        );
        assert_eq!(
            softmax_weights, 0,
            "bucket {bucket}: CachedSoftmaxWeights count"
        );

        for bound in attention {
            let BoundOpKind::CachedAttention { head_dim, .. } = &bound.kind else {
                unreachable!("filtered to CachedAttention above");
            };
            assert_eq!(
                bound.dtype,
                DType::Float32,
                "bucket {bucket} node {}: the K and V planes the kernel streams are f32, which is the byte count the split kernel moves",
                bound.node.0
            );
            let kernel = omega::emit(bound, &PackedOperands::new(), policy)
                .expect("the bound decode attention op emits");
            assert!(
                kernel.entry.ends_with("_ds"),
                "bucket {bucket} node {}: the op must take the split form, got {}",
                bound.node.0,
                kernel.entry
            );
            let width = kernel
                .grid
                .threadgroup_width
                .expect("the split form fixes its threadgroup width");
            let (expected, expected_width) = if *head_dim == GLOBAL_HEAD_DIM {
                (global_threadgroups, global_width)
            } else {
                (sliding_threadgroups, sliding_width)
            };
            assert_eq!(
                kernel.grid.threads / width,
                expected,
                "bucket {bucket} node {} head_dim {head_dim}: threadgroups",
                bound.node.0
            );
            assert_eq!(
                width, expected_width,
                "bucket {bucket} node {} head_dim {head_dim}: threads per threadgroup",
                bound.node.0
            );
        }
    }
}
