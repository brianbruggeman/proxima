#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

//! What a `kv_bound_extent` bucket crossing changes in the bound decode plan
//! of the real gemma4-E2B checkpoint, and that
//! [`proxima_tensor::refit_cached_attention_rows`] reproduces a fresh bind of
//! it. Binds the production decode program (`GEMMA4.bind_with_kv_layout` with
//! the default sliding-ring layout, the `LoadedModel::load` registry path) at
//! two adjacent bucket extents with production fusion and the production
//! symbol layout (the sliding-window symbol bound to `min(extent, window)`),
//! prunes, and checks:
//!
//! - nothing unread survives the prune (the `causal_mask_merged` iotas the
//!   fused attention op absorbed are gone);
//! - the only ops that differ across the crossing are fused cached-attention
//!   ops (35 below the sliding window, 7 global-layer ops above it);
//! - applying the refit patches to the lower plan yields exactly the upper
//!   plan, op for op.

use std::collections::BTreeSet;
use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{Architecture, GEMMA4, KvLayout, bind_symbols, symbols};
use proxima_tensor::bind::{BoundOp, BoundOpKind};
use proxima_tensor::spec::Qwen35LayerRoots;
use proxima_tensor::{
    NodeId, NumericPolicy, Op, Shapes, bind_with_fusion, dead_resolved_nodes, infer, prune_dead,
    refit_cached_attention_rows,
};

const SLIDING_WINDOW: usize = 512;

const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

fn production_step_outputs(logits_root: NodeId, layer_roots: &[Qwen35LayerRoots]) -> Vec<NodeId> {
    let mut outputs = vec![logits_root];
    for roots_for_layer in layer_roots {
        if let Qwen35LayerRoots::Attention((even, odd, value)) = roots_for_layer {
            outputs.extend([*even, *odd, *value]);
        }
    }
    outputs
}

fn resolve(
    program: &[Op],
    outputs: &[NodeId],
    kv_bound_extent: usize,
    single_position: bool,
) -> (Shapes, Vec<BoundOp>) {
    let mut bound = bind_symbols(1, kv_bound_extent, &[], single_position)
        .expect("gemma4 declares no extra step inputs");
    bound[usize::from(symbols::SLIDING_KV_BOUND)] = kv_bound_extent.min(SLIDING_WINDOW) as u64;
    let shapes = infer(program, &bound).expect("shape inference over the real program");
    let resolved = bind_with_fusion(
        program,
        &shapes,
        outputs,
        true,
        NumericPolicy::llama_relaxed(),
    )
    .expect("bind_with_fusion over the real program");
    (shapes, prune_dead(resolved, outputs))
}

#[proxima::test]
async fn gemma4_bucket_crossing_is_local_to_cached_attention_and_refits_exactly() {
    let Ok(file) = File::open(REAL_GEMMA4_E2B_GGUF_PATH) else {
        eprintln!("skipping: real gemma4-E2B blob not found at {REAL_GEMMA4_E2B_GGUF_PATH}");
        return;
    };
    // SAFETY: the checkpoint is a read-only mapping no other process writes.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the real gemma4-E2B checkpoint header");
    let bound_program = GEMMA4
        .bind_with_kv_layout(&parsed, &mapping, KvLayout::SlidingRing)
        .expect("bind the real gemma4-E2B decode program");
    let outputs = production_step_outputs(bound_program.logits_root, &bound_program.layer_roots);
    let program = &bound_program.program;
    let single = bound_program.single_position_step;

    let crossings = [
        (32usize, 64usize, 35usize),
        (224, 256, 35),
        (480, 512, 35),
        (512, 544, 7),
        (1152, 1184, 7),
        (2016, 2048, 7),
    ];
    for (low, high, expected_patches) in crossings {
        let (low_shapes, low_plan) = resolve(program, &outputs, low, single);
        let (high_shapes, high_plan) = resolve(program, &outputs, high, single);
        let label = format!("crossing {low}->{high}");

        assert!(
            dead_resolved_nodes(&low_plan, &outputs).is_empty(),
            "{label}: an unread node survived the prune"
        );
        assert_eq!(low_plan.len(), high_plan.len(), "{label}: op count moved");
        let differing: BTreeSet<usize> = (0..low_plan.len())
            .filter(|position| low_plan[*position] != high_plan[*position])
            .collect();
        assert_eq!(
            differing.len(),
            expected_patches,
            "{label}: ops that differ"
        );
        assert!(
            differing.iter().all(|position| matches!(
                low_plan[*position].kind,
                BoundOpKind::CachedAttention { .. }
            )),
            "{label}: a non-attention op differs across the crossing"
        );

        let patches = refit_cached_attention_rows(&low_plan, program, &low_shapes, &high_shapes)
            .unwrap_or_else(|| {
                panic!("{label}: refit declined a crossing that is local to attention")
            });
        assert_eq!(
            patches
                .iter()
                .map(|(position, _)| *position)
                .collect::<BTreeSet<_>>(),
            differing,
            "{label}: refit patches a different set of positions than a fresh bind changes"
        );
        let mut refit = low_plan.clone();
        for (position, patched) in patches {
            refit[position] = patched;
        }
        assert_eq!(
            refit, high_plan,
            "{label}: refit plan differs from a fresh bind"
        );
        println!(
            "gemma4_refit {label} ops={} patched={}",
            refit.len(),
            differing.len()
        );
    }
}
