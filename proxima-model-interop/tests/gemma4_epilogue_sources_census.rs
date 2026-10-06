//! Census of what `NumericPolicy::epilogue_sources` changes in gemma4-E2B's
//! single-token decode plan: binds the production decode program
//! (`bind_checkpoint` -> `bind_with_fusion` with `fuse_cached_attention:
//! false`, the same shape `gemma4_attention_chain_census.rs` binds) once with
//! the switch off and once on, prunes dead nodes the way `omega`'s plan
//! preparation does, and compares the resolved op counts and per-kind
//! histograms.
//!
//! The absorbed count is derived from the checkpoint's own layer schedule,
//! not fitted: 35 layers x (3 hidden RMSNorm tails + Q-norm tail + attention
//! combine) = 35 x 5 = 175, plus 15 own-KV layers x 2 (K-norm, V-norm) = 30,
//! 175 + 30 = 205. The widened rule reaches each of those tails because the
//! consumer's first reduce operand is a multi-reader projection output that
//! the first-operand rule stops at.
//!
//! `#[ignore]`d: needs the host-local gemma4-E2B checkpoint, and fails loudly
//! naming the env var and path when it is absent ([`require_fixture`]) so an
//! explicit run never passes having executed nothing.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{bind_checkpoint, bind_symbols};
use proxima_tensor::bind::BoundOp;
use proxima_tensor::spec::Qwen35LayerRoots;
use proxima_tensor::{NodeId, NumericPolicy, Op, bind_with_fusion, infer, prune_dead};

const GEMMA4_E2B_DEFAULT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const NEW_COUNT: usize = 1;
const KV_BUCKET_EXTENT: usize = 32;

const EXPECTED_OFF: usize = 1591;
const EXPECTED_ON: usize = 1386;
const EXPECTED_ABSORBED: usize = 205;

const LAYERS: usize = 35;
const OWN_KV_LAYERS: usize = 15;
const PER_LAYER_TAILS: usize = 3 + 1 + 1;
const PER_OWN_KV_LAYER_TAILS: usize = 2;

fn gemma4_e2b_gguf_path() -> String {
    std::env::var("PROXIMA_GEMMA4_E2B_GGUF").unwrap_or_else(|_| GEMMA4_E2B_DEFAULT_PATH.to_string())
}

fn require_fixture(path: &str, env_var: &str) {
    assert!(
        Path::new(path).exists(),
        "no host-local gguf fixture at {path}: set {env_var} to a valid checkpoint path, or stage one at this default path"
    );
}

fn production_step_outputs(logits_root: NodeId, layer_roots: &[Qwen35LayerRoots]) -> Vec<NodeId> {
    let mut outputs = vec![logits_root];
    for roots_for_layer in layer_roots {
        match roots_for_layer {
            Qwen35LayerRoots::Attention((even, odd, value)) => {
                outputs.extend([*even, *odd, *value]);
            }
            Qwen35LayerRoots::SharedFromLayer(_) => {}
            Qwen35LayerRoots::DenseAttention(_) | Qwen35LayerRoots::Ssm { .. } => {
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

fn bind_decode(
    program: &[Op],
    shapes: &proxima_tensor::Shapes,
    outputs: &[NodeId],
    policy: NumericPolicy,
) -> Vec<BoundOp> {
    let bound_ops = bind_with_fusion(program, shapes, outputs, false, policy)
        .expect("bind_with_fusion over the gemma4-E2B decode program");
    prune_dead(bound_ops, outputs)
}

#[test]
#[ignore = "needs host-local gemma4-E2B gguf"]
fn gemma4_epilogue_sources_census() {
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

    let switch_off = NumericPolicy::llama_relaxed();
    let switch_on = switch_off.with_epilogue_sources(true);
    let off = bind_decode(&bound_program.program, &shapes, &outputs, switch_off);
    let on = bind_decode(&bound_program.program, &shapes, &outputs, switch_on);

    let absorbed = off.len() - on.len();
    println!(
        "epilogue_sources census: off={} on={} absorbed={absorbed}",
        off.len(),
        on.len()
    );
    let off_kinds = kind_histogram(&off);
    let on_kinds = kind_histogram(&on);
    let names: std::collections::BTreeSet<&str> =
        off_kinds.keys().chain(on_kinds.keys()).copied().collect();
    for name in names {
        let before = off_kinds.get(name).copied().unwrap_or(0);
        let after = on_kinds.get(name).copied().unwrap_or(0);
        println!("epilogue_sources census kind={name} off={before} on={after}");
    }

    assert_eq!(
        LAYERS * PER_LAYER_TAILS + OWN_KV_LAYERS * PER_OWN_KV_LAYER_TAILS,
        EXPECTED_ABSORBED,
        "35 x (3 hidden + Q-norm + attention combine) + 15 x (K-norm, V-norm) = 175 + 30"
    );
    assert_eq!(off.len(), EXPECTED_OFF, "switch-off op count");
    assert_eq!(on.len(), EXPECTED_ON, "switch-on op count");
    assert_eq!(
        absorbed, EXPECTED_ABSORBED,
        "ops absorbed by the widened rule"
    );
}
