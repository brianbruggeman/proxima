//! OBSERVED census of gemma4-E2B's attention chain, per layer, for a
//! single-token decode step (`s=1`) at a bucketed KV extent of 32 -- the
//! same `(new_count, kv_bucket_extent, outputs)` plan key
//! `residency_caches.rs:1878` (`Self::evaluate`'s own `shape` tuple)
//! resolves against. Binds the exact program
//! [`proxima_model_interop::Architecture::bind`] builds for gemma4
//! (`Gemma4Arch::bind` -> `bind_gemma4_with_last_row_only(.., true)`,
//! private to `proxima-model-interop`, so this test calls the trait method
//! via the public [`GEMMA4`] static instead of reproducing its body), then
//! runs [`proxima_tensor::bind_with_fusion`] TWICE over that SAME program:
//! once with `fuse_cached_attention: false` (the PRODUCTION shape --
//! `bind_cached_attention_fusion`'s own early return,
//! `cached_attention_epilogue_liveness.rs:19-20`, means `false` skips
//! cached-attention recognition entirely and returns `bind_plain`'s
//! unfused chain verbatim, independent of whether the caller's crate was
//! even compiled with `cached-attention-streaming`), once with `true`
//! (kept as a second column so the fused/unfused delta per layer is an
//! OBSERVED number, not an inferred one).
//!
//! `outputs` (the requested output `NodeId`s) is reproduced by hand from
//! `generate/decode.rs`'s own `roots` assembly (around line 2595-2679) for
//! the plain, non-debug, non-monolithic-prefill step: `[logits_root]` plus,
//! per layer, `(even, odd, value)` for [`Qwen35LayerRoots::Attention`] and
//! nothing for [`Qwen35LayerRoots::SharedFromLayer`] (gemma4 E2B's shared-KV
//! trailing layers read their donor layer's own already-requested nodes
//! in-graph, per that variant's own doc).
//!
//! Per-layer attention-chain classification (unfused case): a `BoundOp`
//! strictly between a layer's q-projection reduce and its wo-projection
//! reduce is "other" (norm/rope/K,V-projection) when any of its DIRECT
//! `operands()` is a named `Op::Input` leaf whose name ends in `.weight`
//! (every projection weight and every RMSNorm scale) or starts with
//! `rope_cos`/`rope_sin` (`attention_forward.rs:2385-2984`,
//! `gemma4/bind.rs:774-818`'s own `rope_cos_swa`/`rope_sin_swa` naming) --
//! everything else in the span (score product+reduce, softmax max/exp/sum/
//! divide, AV weighted-sum reduce, the cached/new merge) is "chain". In the
//! fused case a chain BoundOp is simply `BoundOpKind::CachedAttention`.
//!
//! Skips (does not fail) when the real blob is absent -- same posture as
//! `gemma4_correctness_gate.rs:112-117`.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write;

use memmap2::Mmap;
use omega::PackedOperands;
use proxima_gguf::parse_complete;
use proxima_model_interop::{Architecture, GEMMA4, bind_symbols};
use proxima_tensor::bind::{BoundOp, BoundOpKind};
use proxima_tensor::spec::Qwen35LayerRoots;
use proxima_tensor::{NodeId, NumericPolicy, Op, bind_with_fusion, infer, prune_dead};

const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

const ABSORBED_NODES_LOG_PATH: &str = "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/f00a0e26-f6a4-4429-b155-6f5915575ad2/scratchpad/attn_parity/census/absorbed_nodes.txt";

const CENSUS_LAYERS: [u32; 5] = [0, 4, 14, 15, 34];
const GRID_CENSUS_LAYERS: [u32; 3] = [0, 4, 15];

const NEW_COUNT: usize = 1;
const KV_BUCKET_EXTENT: usize = 32;

/// Every real `Op::Input` leaf's own name, keyed by its `NodeId` -- a
/// program's `NodeId` is exactly its position in `program` (`ShapeTable::push`,
/// `proxima-tensor/src/shape.rs:86`), so this is a plain positional scan, not
/// a second numbering scheme.
fn named_input_nodes(program: &[Op]) -> BTreeMap<NodeId, &str> {
    program
        .iter()
        .enumerate()
        .filter_map(|(index, op)| op.name().map(|name| (NodeId(index as u32), name)))
        .collect()
}

fn find_named_node(named: &BTreeMap<NodeId, &str>, name: &str) -> Option<NodeId> {
    named
        .iter()
        .find(|(_, candidate)| **candidate == name)
        .map(|(node, _)| *node)
}

/// [`decode.rs`]'s own `roots` assembly for a plain, non-speculative,
/// non-monolithic-prefill decode step (`generate/decode.rs:2595-2679`):
/// `[logits_root]` then, per layer in order, the real cache-owning layer's
/// `(even, odd, value)` triple or nothing for a shared-KV layer.
fn production_step_outputs(logits_root: NodeId, layer_roots: &[Qwen35LayerRoots]) -> Vec<NodeId> {
    let mut outputs = Vec::with_capacity(1 + layer_roots.len() * 3);
    outputs.push(logits_root);
    for roots_for_layer in layer_roots {
        match roots_for_layer {
            Qwen35LayerRoots::Attention((even, odd, value)) => {
                outputs.push(*even);
                outputs.push(*odd);
                outputs.push(*value);
            }
            Qwen35LayerRoots::SharedFromLayer(_) => {}
            Qwen35LayerRoots::DenseAttention(_) | Qwen35LayerRoots::Ssm { .. } => {
                panic!("gemma4 E2B's own layer schedule is Attention/SharedFromLayer only")
            }
        }
    }
    outputs
}

fn truncated_debug(kind: &BoundOpKind) -> String {
    let full = format!("{kind:?}");
    if full.len() > 200 {
        format!("{}...<{} bytes total>", &full[..200], full.len())
    } else {
        full
    }
}

fn read_sources(bound: &BoundOp) -> Vec<u32> {
    bound.operands().iter().map(|(node, _, _)| node.0).collect()
}

/// Locates one layer's `(q_reduce_index, wo_reduce_index)` inside a bound
/// list, independently of any other bind's own indices -- `BoundOp::node`
/// carries the ORIGINAL program `NodeId` regardless of fusion, so the same
/// weight-node lookup applied to two different `bind_with_fusion` outputs
/// finds each bind's own (possibly different) position for the same
/// logical projection.
fn find_layer_span_bounds(
    bound_ops: &[BoundOp],
    named: &BTreeMap<NodeId, &str>,
    layer: u32,
) -> Option<(usize, usize)> {
    let q_weight_node = find_named_node(named, &format!("blk.{layer}.attn_q.weight"))?;
    let wo_weight_node = find_named_node(named, &format!("blk.{layer}.attn_output.weight"))?;
    let q_reduce_index = bound_ops.iter().position(|bound| {
        matches!(bound.kind, BoundOpKind::Reduce { .. })
            && bound.operands().iter().any(|(node, _, _)| *node == q_weight_node)
    })?;
    let wo_reduce_index = bound_ops.iter().position(|bound| {
        matches!(bound.kind, BoundOpKind::Reduce { .. })
            && bound.operands().iter().any(|(node, _, _)| *node == wo_weight_node)
    })?;
    (wo_reduce_index > q_reduce_index).then_some((q_reduce_index, wo_reduce_index))
}

/// `true` when `node` is a named leaf this census counts as a
/// projection/norm/rope anchor -- see this module's own doc for the exact
/// rule and its citations.
fn is_projection_norm_or_rope_leaf(name: &str) -> bool {
    name.ends_with(".weight") || name.starts_with("rope_cos") || name.starts_with("rope_sin")
}

/// Classifies one `BoundOp` strictly between a layer's q-projection reduce
/// and its wo-projection reduce. Fused case: the single
/// [`BoundOpKind::CachedAttention`] step. Unfused case: everything whose
/// direct operands name no projection/norm/rope leaf -- see module doc.
fn is_chain_op(bound: &BoundOp, named: &BTreeMap<NodeId, &str>, fused: bool) -> bool {
    if fused {
        return matches!(bound.kind, BoundOpKind::CachedAttention { .. });
    }
    !bound.operands().iter().any(|(node, _, _)| {
        named
            .get(node)
            .is_some_and(|name| is_projection_norm_or_rope_leaf(name))
    })
}

struct LayerChainCensus {
    layer: u32,
    chain: usize,
    other: usize,
}

/// One full [`bind_with_fusion`] pass plus its own per-layer chain census --
/// run twice by [`gemma4_attention_chain_census`], once per `fuse_cached_attention`
/// setting, so the fused/unfused delta is two runs of the SAME code, not two
/// hand-written classifiers.
#[allow(clippy::too_many_arguments)]
fn run_one_bind(
    label: &str,
    program: &[Op],
    shapes: &proxima_tensor::Shapes,
    outputs: &[NodeId],
    named: &BTreeMap<NodeId, &str>,
    block_count: u32,
    fuse_cached_attention: bool,
    numeric_policy: NumericPolicy,
) -> (Vec<BoundOp>, Vec<LayerChainCensus>) {
    let bound_ops = bind_with_fusion(program, shapes, outputs, fuse_cached_attention, numeric_policy)
        .unwrap_or_else(|error| panic!("bind_with_fusion[{label}] failed: {error}"));

    println!(
        "gemma4_attention_chain_census[{label}]: total_bound_ops={} new_count={NEW_COUNT} \
         kv_bucket_extent={KV_BUCKET_EXTENT} requested_outputs={}",
        bound_ops.len(),
        outputs.len()
    );
    if !fuse_cached_attention {
        const OBSERVED_PRODUCTION_TOTAL: usize = 1661;
        println!(
            "gemma4_attention_chain_census[{label}]: total_bound_ops={} \
             equals_stored_production_log_1661={} -- requested_outputs={} new_count={NEW_COUNT} \
             kv_bucket_extent={KV_BUCKET_EXTENT} (this test's own s/bucket/outputs, spelled out so \
             a mismatch can be attributed rather than guessed)",
            bound_ops.len(),
            bound_ops.len() == OBSERVED_PRODUCTION_TOTAL,
            outputs.len()
        );
    }

    for (index, bound) in bound_ops.iter().enumerate() {
        println!(
            "boundop[{label}] index={index} node={} kind={} extents={:?} reads={:?} debug={}",
            bound.node.0,
            bound.kind.name(),
            bound.extents,
            read_sources(bound),
            truncated_debug(&bound.kind)
        );
    }

    let mut per_layer = Vec::new();
    for layer in 0..block_count {
        let Some((q_reduce_index, wo_reduce_index)) =
            find_layer_span_bounds(&bound_ops, named, layer)
        else {
            println!("layer={layer}[{label}] census: span bounds not found -- skipped");
            continue;
        };
        let span = &bound_ops[q_reduce_index + 1..wo_reduce_index];
        let chain_count = span
            .iter()
            .filter(|bound| is_chain_op(bound, named, fuse_cached_attention))
            .count();
        let other_count = span.len() - chain_count;
        per_layer.push(LayerChainCensus {
            layer,
            chain: chain_count,
            other: other_count,
        });
        println!(
            "layer={layer}[{label}] q_reduce_index={q_reduce_index} wo_reduce_index={wo_reduce_index} \
             span_len={} chain_ops={chain_count} other_ops={other_count}",
            span.len()
        );
        if CENSUS_LAYERS.contains(&layer) {
            println!("layer={layer}[{label}] verbatim span:");
            for (offset, bound) in span.iter().enumerate() {
                println!(
                    "  layer={layer}[{label}] span_offset={offset} boundop_index={} node={} \
                     kind={} extents={:?} reads={:?} chain={} debug={}",
                    q_reduce_index + 1 + offset,
                    bound.node.0,
                    bound.kind.name(),
                    bound.extents,
                    read_sources(bound),
                    is_chain_op(bound, named, fuse_cached_attention),
                    truncated_debug(&bound.kind)
                );
            }
        }
        if !fuse_cached_attention && GRID_CENSUS_LAYERS.contains(&layer) {
            let packed_operands = PackedOperands::new();
            for (offset, bound) in span.iter().enumerate() {
                if !is_chain_op(bound, named, false) {
                    continue;
                }
                match omega::emit(bound, &packed_operands, numeric_policy) {
                    Ok(kernel) => println!(
                        "grid[{label}] layer={layer} span_offset={offset} \
                         boundop_index={} kind={} threads={} threadgroup_width={:?} depth={}",
                        q_reduce_index + 1 + offset,
                        bound.kind.name(),
                        kernel.grid.threads,
                        kernel.grid.threadgroup_width,
                        kernel.grid.depth
                    ),
                    Err(error) => println!(
                        "grid[{label}] layer={layer} span_offset={offset} boundop_index={} \
                         kind={} emit_error={error} (empty PackedOperands -- unquantized-read \
                         assumption; a quantized operand here would legitimately fail emit)",
                        q_reduce_index + 1 + offset,
                        bound.kind.name()
                    ),
                }
            }
        }
    }
    (bound_ops, per_layer)
}

#[proxima::test]
async fn gemma4_attention_chain_census() {
    let Ok(file) = File::open(REAL_GEMMA4_E2B_GGUF_PATH) else {
        eprintln!("skipping: real gemma4-E2B blob not found at {REAL_GEMMA4_E2B_GGUF_PATH}");
        return;
    };
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");

    // The production decode entry point: `ArchitectureTrait::bind`'s
    // `Gemma4Arch` impl, `last_row_only: true`
    // (`gemma4/bind.rs:1071-1078`) -- byte-for-byte the program
    // `LoadedModel::load`'s registry path binds for real decode.
    let bound_program = GEMMA4
        .bind(&parsed, bytes)
        .expect("bind the real gemma4-E2B checkpoint's production decode program");

    let named = named_input_nodes(&bound_program.program);
    let outputs = production_step_outputs(bound_program.logits_root, &bound_program.layer_roots);

    let symbols = bind_symbols(NEW_COUNT, KV_BUCKET_EXTENT, &[], bound_program.single_position_step)
        .expect("bind_symbols: gemma4 declares no extra symbolic step_inputs");
    let shapes =
        infer(&bound_program.program, &symbols).expect("shape inference over the real program");

    let numeric_policy = NumericPolicy::llama_relaxed();
    let block_count = bound_program.architecture.block_count;

    // PRODUCTION shape first: `fuse_cached_attention: false`.
    // `bind_cached_attention_fusion`'s own early return
    // (`cached_attention_epilogue_liveness.rs:19-20`) means this is the
    // unfused `bind_plain` chain verbatim, matching every gemma4 layer's
    // real production plan (feature `metal-fuse-attn-decode` off).
    let (unfused_bound_ops, unfused_per_layer) = run_one_bind(
        "unfused_production",
        &bound_program.program,
        &shapes,
        &outputs,
        &named,
        block_count,
        false,
        numeric_policy,
    );

    // Second column: the SAME program/outputs, `fuse_cached_attention: true`
    // -- kept so the fused/unfused delta per layer is observed, not derived.
    let (fused_bound_ops, fused_per_layer) = run_one_bind(
        "fused_feature_on",
        &bound_program.program,
        &shapes,
        &outputs,
        &named,
        block_count,
        true,
        numeric_policy,
    );

    println!(
        "gemma4_attention_chain_census: totals unfused_production={} fused_feature_on={}",
        unfused_bound_ops.len(),
        fused_bound_ops.len()
    );
    println!("layer,chain_unfused_production,other_unfused_production,chain_fused_feature_on,other_fused_feature_on");
    let fused_by_layer: BTreeMap<u32, &LayerChainCensus> =
        fused_per_layer.iter().map(|entry| (entry.layer, entry)).collect();
    for unfused_entry in &unfused_per_layer {
        let fused_entry = fused_by_layer.get(&unfused_entry.layer);
        println!(
            "{},{},{},{},{}",
            unfused_entry.layer,
            unfused_entry.chain,
            unfused_entry.other,
            fused_entry.map_or(-1, |entry| entry.chain as i64),
            fused_entry.map_or(-1, |entry| entry.other as i64),
        );
    }

    let sum_chain_unfused: usize = unfused_per_layer.iter().map(|entry| entry.chain).sum();
    let sum_chain_fused: usize = fused_per_layer.iter().map(|entry| entry.chain).sum();
    println!(
        "gemma4_attention_chain_census: observed_layers={} \
         HEURISTIC_COUNT_sum_chain_ops_unfused_production={sum_chain_unfused} \
         (direct-operand rule, own doc above; not an upper bound -- known over-inclusion, e.g. \
         RMS sum-of-squares reduces whose only direct operand is an intermediate node; omissions \
         not excluded either) sum_chain_ops_fused_feature_on={sum_chain_fused}",
        unfused_per_layer.len()
    );

    // The OBSERVED attention-chain ownership under this exact binding
    // configuration: set difference between the unfused bind's node ids and
    // the fused bind's node ids, per layer, computed off `BoundOp::node`
    // (the ORIGINAL program `NodeId`, stable across both binds -- fusion
    // changes which BoundOps survive into the returned `Vec`, never a
    // node's own numbering). `absorbed` = present unfused, absent fused
    // (folded into the fused `CachedAttention` step); `only_fused` = present
    // fused, absent unfused (expected empty or the anchor node whose KIND
    // changed from `Reduce`/`Elementwise` to `CachedAttention` without a new
    // id -- `bind_cached_attention_fusion`'s own `planning_outputs.push(fused.node)`,
    // `cached_attention_epilogue_liveness.rs:58-60`).
    let mut absorbed_log = File::create(ABSORBED_NODES_LOG_PATH)
        .unwrap_or_else(|error| panic!("create {ABSORBED_NODES_LOG_PATH}: {error}"));
    let mut total_absorbed = 0usize;
    for layer in 0..block_count {
        let Some((unfused_q, unfused_wo)) =
            find_layer_span_bounds(&unfused_bound_ops, &named, layer)
        else {
            continue;
        };
        let Some((fused_q, fused_wo)) = find_layer_span_bounds(&fused_bound_ops, &named, layer)
        else {
            continue;
        };
        let unfused_span = &unfused_bound_ops[unfused_q + 1..unfused_wo];
        let fused_span = &fused_bound_ops[fused_q + 1..fused_wo];
        let fused_ids: BTreeSet<u32> = fused_span.iter().map(|bound| bound.node.0).collect();
        let unfused_ids: BTreeSet<u32> = unfused_span.iter().map(|bound| bound.node.0).collect();
        let absorbed: Vec<&BoundOp> = unfused_span
            .iter()
            .filter(|bound| !fused_ids.contains(&bound.node.0))
            .collect();
        let only_fused: Vec<&BoundOp> = fused_span
            .iter()
            .filter(|bound| !unfused_ids.contains(&bound.node.0))
            .collect();
        total_absorbed += absorbed.len();
        println!(
            "gemma4_attention_chain_census: layer={layer} absorbed_count={} only_fused_count={}",
            absorbed.len(),
            only_fused.len()
        );
        writeln!(
            absorbed_log,
            "layer={layer} absorbed_count={} only_fused_count={}",
            absorbed.len(),
            only_fused.len()
        )
        .expect("write absorbed_nodes.txt header line");
        for bound in &absorbed {
            writeln!(
                absorbed_log,
                "  layer={layer} ABSORBED node={} kind={} extents={:?} reads={:?}",
                bound.node.0,
                bound.kind.name(),
                bound.extents,
                read_sources(bound)
            )
            .expect("write absorbed_nodes.txt absorbed line");
        }
        for bound in &only_fused {
            writeln!(
                absorbed_log,
                "  layer={layer} ONLY_FUSED node={} kind={} extents={:?} reads={:?}",
                bound.node.0,
                bound.kind.name(),
                bound.extents,
                read_sources(bound)
            )
            .expect("write absorbed_nodes.txt only_fused line");
        }
    }
    println!(
        "gemma4_attention_chain_census: total_absorbed_nodes={total_absorbed} \
         (expected to equal unfused_total - fused_total = {} - {} = {}) \
         absorbed_nodes_log={ABSORBED_NODES_LOG_PATH}",
        unfused_bound_ops.len(),
        fused_bound_ops.len(),
        unfused_bound_ops.len() as i64 - fused_bound_ops.len() as i64
    );

    // Attribution of the 2-op gap (1663 here vs. the stored 1661
    // `ENCODE_DISPATCH_CALLS` log): `bind_with_fusion` alone never prunes a
    // node unreachable from `outputs` -- that pass is `prune_dead`, called
    // by `omega`'s OWN plan-preparation step, `prepare_uniforms_pack.rs:132-167`
    // (`prepare`, the function `plan`/`plan_named` calls before a `Plan`
    // ever reaches `execute`), never by `bind_with_fusion` itself. This
    // census called `bind_with_fusion` directly and stopped there, so it
    // never ran the SAME dead-node prune `prepare` runs immediately
    // afterward with `effective_outputs == outputs` unchanged (`outputs` is
    // non-empty here, so `prepare_uniforms_pack.rs:126-129`'s
    // `effective_outputs = outputs.to_vec()` arm applies verbatim). Running
    // the identical `prune_dead(unfused_bound_ops, &outputs)` call here is
    // pure graph pruning -- no device, no dispatch -- so it stays inside
    // this census's own "no GPU execution" bound.
    let pruned = prune_dead(unfused_bound_ops.clone(), &outputs);
    let pruned_ids: BTreeSet<u32> = pruned.iter().map(|bound| bound.node.0).collect();
    let dropped_by_prune: Vec<&BoundOp> = unfused_bound_ops
        .iter()
        .filter(|bound| !pruned_ids.contains(&bound.node.0))
        .collect();
    println!(
        "gemma4_attention_chain_census: prune_dead(unfused_production, outputs).len()={} \
         (unfused_production before pruning: {}; stored production log: 1661) \
         dropped_by_prune_dead={}",
        pruned.len(),
        unfused_bound_ops.len(),
        dropped_by_prune.len()
    );
    for bound in &dropped_by_prune {
        println!(
            "gemma4_attention_chain_census: prune_dead dropped node={} kind={} extents={:?} \
             reads={:?} debug={}",
            bound.node.0,
            bound.kind.name(),
            bound.extents,
            read_sources(bound),
            truncated_debug(&bound.kind)
        );
    }
    if pruned.len() != 1661 {
        println!(
            "gemma4_attention_chain_census: prune_dead(unfused_production).len()={} still != \
             1661 -- the {}-op residual is NOT attributed by this census (candidates not ruled \
             out: `resident_skip` for a plan-time-constant `Iota`/`Constant` leaf on a WARM call \
             (`placements_execute_named.rs:333-336`), only fires when `ServingConfig::plan_time_constants` \
             is `true` -- `false` by default, `proxima-model-interop/src/serving.rs:605` -- and \
             this census does not know the stored log's own `ServingConfig`; a `metal-horizontal-merge` \
             multi-position merge folding two BoundOps into one `ENCODE_DISPATCH_CALLS` increment \
             (`device_buffers_arena_plan.rs:1387`), off by default per this crate's own `metal` \
             feature list (`Cargo.toml:117-131`, `metal-horizontal-merge` absent) -- neither is \
             confirmed against the artifact that produced 1661, which this census did not open)",
            pruned.len(),
            pruned.len() as i64 - 1661
        );
    } else {
        println!(
            "gemma4_attention_chain_census: prune_dead(unfused_production, outputs).len() == 1661 \
             -- the 2-op gap between the raw bind_with_fusion output (1663) and the stored \
             ENCODE_DISPATCH_CALLS log (1661) attributes to `prune_dead` dropping the \
             {}-node dead set printed above, a pass this census's earlier bind_with_fusion-only \
             report never ran.",
            dropped_by_prune.len()
        );
    }
    println!("gemma4_attention_chain_census: physical launches remain unmeasured");

    // (c) physical Metal dispatches: NOT observed in this census -- no plan
    // was executed, and even an executed plan would not give a 1:1 BoundOp
    // count. `encode_op`'s cached-attention split+merge path
    // (`omega/src/metal/arena_encode_dispatch_finish.rs:1043` the split
    // dispatch, `:1099` the merge dispatch) issues TWO `dispatch` calls for
    // ONE `BoundOpKind::CachedAttention` bound op whenever bucketed KV
    // widens the key extent past the merged length. The sole physical
    // dispatch site in this codebase is `resident_nocopy_cache.rs:1333`
    // (`encoder.dispatchThreads_threadsPerThreadgroup`), called from both
    // the split and the merge sites above -- so no BoundOp count from either
    // bind above is (c)'s dispatch count, and this census does not report
    // (c) as a number.
    println!(
        "gemma4_attention_chain_census: physical_dispatch_count=not_observed_in_this_census \
         (no execution; BoundOpKind::CachedAttention's split+merge path can issue 2 dispatches \
         for 1 bound op -- arena_encode_dispatch_finish.rs:1043,1099 -- and the only physical \
         dispatch call is resident_nocopy_cache.rs:1333)"
    );
}
