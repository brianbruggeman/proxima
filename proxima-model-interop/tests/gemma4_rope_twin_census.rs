//! Census of what `fuse_twin_elementwise` changes in gemma4-E2B's single-token decode plan: binds
//! the production decode program (`bind_checkpoint` -> `bind_with_fusion` with `fuse_cached_attention:
//! false`, the shape `gemma4_epilogue_sources_census.rs` binds), prunes dead nodes the way `omega`'s
//! plan preparation does, then runs the twin pass and compares the resolved op counts.
//!
//! The expected count is derived from the checkpoint's own layer schedule, not fitted: every one of
//! the 35 layers rotates Q (one `fused_rope_pair`), and the 15 layers that own their KV rotate K
//! too, so 35 + 15 = 50 rotated tensors, each two plain elementwise ops before the pass (the 100
//! `RoPE` rows of the r8 census) and one `ElementwiseTwin` after it.
//!
//! Needs the host-local gemma4-E2B checkpoint and fails loudly, naming the env var and
//! path when it is absent ([`require_fixture`]) so an explicit run never passes having executed
//! nothing.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{bind_checkpoint, bind_symbols};
use proxima_tensor::bind::BoundOp;
use proxima_tensor::spec::LayerCacheRoots;
use proxima_tensor::{
    BoundOpKind, NodeId, NumericPolicy, bind_with_fusion, fuse_twin_elementwise, infer, prune_dead,
};

const GEMMA4_E2B_DEFAULT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const NEW_COUNT: usize = 1;
const KV_BUCKET_EXTENT: usize = 32;

const LAYERS: usize = 35;
const OWN_KV_LAYERS: usize = 15;
const ROTATED_TENSORS: usize = LAYERS + OWN_KV_LAYERS;
const DISPATCHES_PER_ROTATED_TENSOR_BEFORE: usize = 2;

fn gemma4_e2b_gguf_path() -> String {
    std::env::var("PROXIMA_GEMMA4_E2B_GGUF").unwrap_or_else(|_| GEMMA4_E2B_DEFAULT_PATH.to_string())
}

fn require_fixture(path: &str, env_var: &str) {
    assert!(
        Path::new(path).exists(),
        "no host-local gguf fixture at {path}: set {env_var} to a valid checkpoint path, or stage one at this default path"
    );
}

fn production_step_outputs(logits_root: NodeId, layer_roots: &[LayerCacheRoots]) -> Vec<NodeId> {
    let mut outputs = vec![logits_root];
    for roots_for_layer in layer_roots {
        match roots_for_layer {
            LayerCacheRoots::Attention((even, odd, value)) => {
                outputs.extend([*even, *odd, *value]);
            }
            LayerCacheRoots::SharedFromLayer(_) => {}
            LayerCacheRoots::DenseAttention(_) | LayerCacheRoots::Ssm { .. } => {
                panic!("gemma4 E2B's own layer schedule is Attention/SharedFromLayer only")
            }
        }
    }
    outputs
}

fn kind_histogram(bound_ops: &[BoundOp]) -> BTreeMap<&'static str, usize> {
    let mut histogram = BTreeMap::new();
    for bound in bound_ops {
        *histogram.entry(bound.kind.name()).or_insert(0) += 1;
    }
    histogram
}

#[test]
fn gemma4_rope_twin_census() {
    let path = gemma4_e2b_gguf_path();
    require_fixture(&path, "PROXIMA_GEMMA4_E2B_GGUF");
    let file = File::open(&path).expect("open the gemma4-E2B checkpoint");
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the gemma4-E2B checkpoint header");
    let bound_program = bind_checkpoint(&parsed, bytes)
        .expect("bind the gemma4-E2B production decode program");
    let outputs = production_step_outputs(bound_program.logits_root, &bound_program.layer_roots);
    let symbols = bind_symbols(
        NEW_COUNT,
        KV_BUCKET_EXTENT,
        &[],
        bound_program.single_position_step,
    )
    .expect("bind_symbols: gemma4 declares no extra symbolic step_inputs");
    let shapes =
        infer(&bound_program.program, &symbols).expect("shape inference over the decode program");
    let policy = NumericPolicy::llama_relaxed().with_epilogue_sources(true);
    let bound_ops = bind_with_fusion(&bound_program.program, &shapes, &outputs, false, policy)
        .expect("bind_with_fusion over the gemma4-E2B decode program");
    let before = prune_dead(bound_ops, &outputs);

    let after = fuse_twin_elementwise(before.clone(), &bound_program.program);

    let twins = after
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::ElementwiseTwin { .. }))
        .count();
    println!(
        "rope twin census: before={} after={} twins={twins}",
        before.len(),
        after.len()
    );
    let before_kinds = kind_histogram(&before);
    let after_kinds = kind_histogram(&after);
    let names: std::collections::BTreeSet<&str> =
        before_kinds.keys().chain(after_kinds.keys()).copied().collect();
    for name in names {
        println!(
            "rope twin census kind={name} before={} after={}",
            before_kinds.get(name).copied().unwrap_or(0),
            after_kinds.get(name).copied().unwrap_or(0)
        );
    }

    assert_eq!(
        twins, ROTATED_TENSORS,
        "35 Q rotations + 15 own-KV K rotations, one twin each"
    );
    assert_eq!(
        before.len() - after.len(),
        ROTATED_TENSORS * (DISPATCHES_PER_ROTATED_TENSOR_BEFORE - 1),
        "every twin removes exactly one dispatch: 100 rope dispatches become 50"
    );
}
