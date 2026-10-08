//! Census of the norm-family reduces in gemma4-E2B's single-token decode
//! program, bound the way `gemma4_epilogue_sources_census.rs` binds it
//! (serving numeric policy, `fuse_cached_attention: false`, dead nodes
//! pruned). It classifies every `Keep::Reduce` by the extent of the axis it
//! folds and by whether it carries a broadcast epilogue, and pins the one
//! class llama does not have: the per-layer-input RMSNorm over `[1, 256]`.
//!
//! llama.cpp computes that norm once over `[256, 35]` (`norm`, `fuse=3`,
//! `evidence/slice0/llama_ops/e2b_ops.tsv`); `ple_layer_input`
//! (`proxima-tensor/src/spec/attention_forward.rs`) builds it once per layer
//! over a `d + layer * 256` window of a flat `[s, 35 * 256]` tensor, so the
//! program holds `LAYERS` reduces where llama has 1. That difference is the
//! `LAYERS - 1 = 34` norm-family dispatches the r8 census left untraced
//! (276 against 242). Every other norm class matches llama one for one:
//! hidden-width 176, Q 35, K 15, V 15 (`census_groups.csv`).
//!
//! `#[ignore]`d: needs the host-local gemma4-E2B checkpoint, and fails loudly
//! naming the env var and path when it is absent so an explicit run never
//! passes having executed nothing.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{bind_checkpoint, bind_symbols};
use proxima_tensor::spec::LayerCacheRoots;
use proxima_tensor::{
    BoundOp, BoundOpKind, Keep, NodeId, NumericPolicy, bind_with_fusion, infer, prune_dead,
};

const GEMMA4_E2B_DEFAULT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const NEW_COUNT: usize = 1;
const KV_BUCKET_EXTENT: usize = 32;

const LAYERS: usize = 35;
const PLE_DIM: u64 = 256;
const LLAMA_PLE_NORMS: usize = 1;

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

fn bind_decode(policy: NumericPolicy) -> Vec<BoundOp> {
    let path = gemma4_e2b_gguf_path();
    require_fixture(&path, "PROXIMA_GEMMA4_E2B_GGUF");
    let file = File::open(&path).expect("open the gemma4-E2B checkpoint");
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the gemma4-E2B checkpoint header");
    let bound_program =
        bind_checkpoint(&parsed, bytes).expect("bind the gemma4-E2B production decode program");
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
    let bound_ops = bind_with_fusion(&bound_program.program, &shapes, &outputs, false, policy)
        .expect("bind_with_fusion over the gemma4-E2B decode program");
    prune_dead(bound_ops, &outputs)
}

fn folded_reduces_by_extents(bound_ops: &[BoundOp]) -> BTreeMap<Vec<u64>, usize> {
    let mut histogram = BTreeMap::new();
    for bound in bound_ops {
        if matches!(&bound.kind, BoundOpKind::Reduce { keep: Keep::Reduce, .. }) {
            *histogram.entry(bound.extents.clone()).or_insert(0) += 1;
        }
    }
    histogram
}

#[test]
#[ignore = "needs host-local gemma4-E2B gguf"]
fn gemma4_decode_holds_one_per_layer_input_norm_per_layer_where_llama_holds_one() {
    let bound_ops = bind_decode(NumericPolicy::llama_relaxed());
    let histogram = folded_reduces_by_extents(&bound_ops);
    for (extents, count) in &histogram {
        println!("norm family census reduce extents={extents:?} count={count}");
    }

    let per_layer_input_norms = histogram.get(&vec![1, PLE_DIM]).copied().unwrap_or(0);
    assert_eq!(
        per_layer_input_norms, LAYERS,
        "reduces over a [1, {PLE_DIM}] row: one per layer-input window, llama has {LLAMA_PLE_NORMS}"
    );
    assert_eq!(
        per_layer_input_norms - LLAMA_PLE_NORMS,
        34,
        "the norm-family dispatches with no llama counterpart"
    );
}
