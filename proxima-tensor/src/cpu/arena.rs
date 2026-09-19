use super::*;

pub(super) type MatmulCohort = ThreadCohort<TensorError>;
pub(super) type MatmulSession<'a> = CohortSession<'a, TensorError>;

/// The result of running a tensor program: every requested output's data
/// and shape, plus the peak number of live intermediate buffers the run
/// held at once, where that backend tracks it.
///
/// One type shared by every backend this crate and `omega` ship — a CPU run
/// and a Metal run report the identical shape, so a parity test compares
/// them directly with no adapter on either side. `peak_live_buffers` is
/// `Some` only where a backend actually counts it: [`evaluate`] and
/// [`evaluate_parallel`] track it against their own `Vec<Option<Vec<f32>>>`
/// buffer table (see [`Evaluated::peak_live_buffers`]'s own doc); a device
/// backend whose buffer lifetime is managed by its own allocator (Metal's
/// retain/release) reports `None` rather than a number that would not mean
/// the same thing.
#[derive(Debug)]
pub struct Evaluated {
    pub(super) root: NodeId,
    pub(super) results: Vec<(NodeId, Vec<u64>, Vec<f32>)>,
    pub(super) peak_live_buffers: Option<usize>,
    pub(super) placed: BTreeSet<NodeId>,
}

impl Evaluated {
    /// Builds a result directly from a backend's own bookkeeping. Public so
    /// a sibling backend (`omega`'s Metal driver) can report through this
    /// same type instead of minting its own — see the type's own doc for
    /// why one shared shape matters here.
    #[must_use]
    pub fn from_parts(
        root: NodeId,
        results: Vec<(NodeId, Vec<u64>, Vec<f32>)>,
        peak_live_buffers: Option<usize>,
    ) -> Self {
        Self::from_parts_with_placed(root, results, peak_live_buffers, BTreeSet::new())
    }

    /// Same contract as [`Evaluated::from_parts`], plus `placed`: the set of
    /// requested output nodes a backend deliberately skipped copying into
    /// `results` because their bytes already live in the caller's own
    /// buffer (`omega`'s `execute_plan_with_placements`) — [`Evaluated::get`]
    /// still reports `None` for one of these (there is no owned data here to
    /// hand back), but [`Evaluated::is_placed`] tells a caller that `None`
    /// meant "ask the placement, not the evaluator" rather than "this node
    /// was never computed at all".
    #[must_use]
    pub fn from_parts_with_placed(
        root: NodeId,
        results: Vec<(NodeId, Vec<u64>, Vec<f32>)>,
        peak_live_buffers: Option<usize>,
        placed: BTreeSet<NodeId>,
    ) -> Self {
        Self {
            root,
            results,
            peak_live_buffers,
            placed,
        }
    }

    #[must_use]
    pub fn root(&self) -> &[f32] {
        self.get(self.root).map_or(&[], |(data, _)| data)
    }

    #[must_use]
    pub fn shape(&self) -> &[u64] {
        self.get(self.root).map_or(&[], |(_, shape)| shape)
    }

    /// The data and shape of a specific requested output, or `None` if
    /// `node` was not in the `outputs` passed to [`evaluate`] — or, on a
    /// backend that reports placements (see [`Evaluated::is_placed`]), if
    /// `node`'s bytes were placed straight into a caller-owned buffer
    /// instead of being copied here. Call [`Evaluated::is_placed`] first to
    /// tell those two `None` cases apart.
    #[must_use]
    pub fn get(&self, node: NodeId) -> Option<(&[f32], &[u64])> {
        self.results
            .iter()
            .find(|(candidate, _, _)| *candidate == node)
            .map(|(_, shape, data)| (data.as_slice(), shape.as_slice()))
    }

    /// `true` if `node` was a requested output whose bytes a backend placed
    /// directly into the caller's own buffer rather than copying into this
    /// `Evaluated` — see [`Evaluated::from_parts_with_placed`]. A `false`
    /// return means either `node` was copied normally (check [`Evaluated::get`])
    /// or it was never a requested output at all.
    #[must_use]
    pub fn is_placed(&self, node: NodeId) -> bool {
        self.placed.contains(&node)
    }

    /// The most buffers ([`Op::Input`] inputs and computed intermediates)
    /// held live at any one point during the run, on a backend that counts
    /// it — `None` otherwise (see this type's own doc). The one number that
    /// proves streaming buffer lifetime is doing something: a long unary
    /// chain should hold a small constant, not one buffer per op.
    #[must_use]
    pub const fn peak_live_buffers(&self) -> Option<usize> {
        self.peak_live_buffers
    }

    /// Surrenders every result's storage into `scratch` instead of letting
    /// it drop, so a caller done reading this result can hand the same
    /// memory to [`evaluate_with_scratch`]'s next call — the counterpart to
    /// that function's pool, letting a caller that already called it once
    /// avoid that function's per-call output allocation on every call after
    /// the first. A caller that never calls this just lets `Evaluated` drop
    /// normally, exactly as today.
    pub fn into_scratch(self, scratch: &mut Vec<Vec<f32>>) {
        scratch.extend(self.results.into_iter().map(|(_, _, data)| data));
    }
}

/// Everything [`evaluate`] and [`evaluate_parallel`] must agree on before
/// either one is free to choose how a single nest actually runs.
///
/// Each buffer-table slot is a [`Cow`]: `Borrowed` for an [`Op::Input`]
/// slice straight out of the caller's `blocks` (never written, never
/// retired — see [`prepare`]), `Owned` for a computed intermediate this
/// evaluator holds. `Cow<[f32]>`'s `Owned` associate is `Vec<f32>` (`[T]:
/// ToOwned<Owned = Vec<T>>`), so this is exactly the borrowed-or-owned shape
/// this table needs, with `Clone`/`Deref`/`into_owned` already provided —
/// no hand-rolled type earns a place next to it.
pub(super) struct Prepared<'block> {
    pub(super) root: NodeId,
    pub(super) shapes: shape::Shapes,
    pub(super) effective_outputs: Vec<NodeId>,
    pub(super) buffers: Vec<Option<Cow<'block, [f32]>>>,
    pub(super) resolved: Vec<BoundOp>,
    pub(super) retires: Vec<Vec<NodeId>>,
}

/// The preamble [`evaluate`] and [`evaluate_parallel`] share: shape
/// inference, output resolution, block binding, stride resolution, and
/// per-node buffer retirement. Neither evaluator's own body decides any of
/// this — the two diverge only in how one already-resolved [`BoundOp`]
/// node gets executed.
///
/// Stride resolution (and the fusion it decides) runs over the whole
/// program at once — see `bind::bind`'s docs. Buffer retirement below
/// is a separate, finer-grained liveness question over the *emitted* node
/// sequence: a held zip's own consumption is deferred to whenever its
/// consumer materializes it, which can be well after the expression
/// position `live::annotate` reasons about, so retirement here is computed
/// fresh over `resolved`, not reused from the fusion pass's liveness.
pub(super) fn prepare<'block>(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&'block [f32]],
    outputs: &[NodeId],
) -> Result<Prepared<'block>, TensorError> {
    let shapes = shape::infer(program, symbols)?;
    // `evaluate`/`evaluate_parallel`'s own `blocks: &[&[f32]]` is f32-only,
    // so neither has a quantized weight to offer this gate yet — see
    // `reject_non_float32`'s doc for what the empty set here is standing in
    // for and what `matmul_q4k_f32` covers instead.
    reject_non_float32(program, &BTreeSet::new())?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };
    // a node the structural pass above exempted as an unreferenced dead leaf
    // can still be exactly what THIS call asked to get back — see
    // `reject_non_float32_outputs`'s own doc for why that stays a separate,
    // always-run, per-call check rather than folding `outputs` into the
    // cached pass above.
    reject_non_float32_outputs(program, &BTreeSet::new(), &effective_outputs)?;

    let block_nodes = block_node_ids(program);
    if blocks.len() != block_nodes.len() {
        return Err(TensorError::InputCountMismatch {
            expected: block_nodes.len(),
            found: blocks.len(),
        });
    }

    let mut buffers: Vec<Option<Cow<'block, [f32]>>> = vec![None; program.len()];
    for (node, data) in block_nodes.iter().zip(blocks.iter()) {
        let expected = element_count(shapes.of(*node));
        if data.len() != expected {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected,
                found: data.len(),
            });
        }
        buffers[node.0 as usize] = Some(Cow::Borrowed(data));
    }

    let resolved = bind::bind(
        program,
        &shapes,
        &effective_outputs,
        NumericPolicy::bit_exact(),
    )?;
    let retires = node_retirement(&resolved, &effective_outputs);
    Ok(Prepared {
        root,
        shapes,
        effective_outputs,
        buffers,
        resolved,
        retires,
    })
}

/// Both evaluators reach the same [`Evaluated`] the same way once their
/// execution loop is done: read each requested output's shape and data
/// back out of the (by-then-retired-down) buffer table.
///
/// Takes `buffers` by value — both callers own the table and drop it right
/// after this returns — so the common case (each output node named once)
/// moves its data out instead of cloning it. A node named twice in
/// `effective_outputs` cannot be moved out twice: `repeats_later` detects
/// that and puts a clone back for the later occurrence to take instead, so
/// duplicate outputs still resolve correctly, just without the free move.
pub(super) fn finish(
    shapes: &shape::Shapes,
    effective_outputs: &[NodeId],
    mut buffers: Vec<Option<Cow<'_, [f32]>>>,
    root: NodeId,
    peak_live_buffers: usize,
) -> Evaluated {
    let results = effective_outputs
        .iter()
        .enumerate()
        .map(|(position, node)| {
            let shape = shapes.of(*node).to_vec();
            let repeats_later = effective_outputs[position + 1..].contains(node);
            let data = match buffers[node.0 as usize].take() {
                Some(buffer) => {
                    if repeats_later {
                        buffers[node.0 as usize] = Some(buffer.clone());
                    }
                    buffer.into_owned()
                }
                None => Vec::new(),
            };
            (*node, shape, data)
        })
        .collect();

    Evaluated::from_parts(root, results, Some(peak_live_buffers))
}

pub use crate::gdn::{GdnPrefillScan, GdnPrefillShape, run_gdn_prefill_scan};

/// Run a tensor program to f32 data.
///
/// `blocks` binds [`Op::Input`] inputs positionally, in the order they
/// appear in `program` — the local, single-partition case; a distributed
/// evaluator would instead resolve blocks by
/// [`name`](Op::name). `outputs` selects which nodes to return data for;
/// an empty slice means the root (the program's last expression) only.
///
/// Every call starts and ends with an empty reuse pool — see
/// [`evaluate_with_scratch`] for the same contract with a caller-carried
/// pool that survives across calls.
pub fn evaluate(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    evaluate_pooled(program, symbols, blocks, outputs, &mut free_buffers)
}

/// Same contract as [`evaluate`], but binds [`Op::Input`] inputs by
/// [`Op::name`] instead of position — the counterpart [`evaluate`]'s own doc
/// promises for the distributed case: a partition crossing a wire
/// (`partition::partition_at`) renumbers a program, so positional order is
/// gone by the time a consumer half receives its inputs, and only `name`
/// survives the cut.
///
/// Every `named` entry here is float32 by construction (this function's own
/// signature), so this is `evaluate_named_via_arena` — [`StaticArena`]'s
/// buffer reuse, law 6∘5 weight packing, and dead/static-node skip, all
/// reached through a bounded process-wide cache (`ARENA_CACHE`/
/// `checkout_arena`, both private) rather than a caller-owned handle.
/// `evaluate_quantized_named`/[`evaluate_quantized_with_scratch`] remain the
/// entry point for a caller that mixes in real quantized weight blocks — a
/// capability [`StaticArena`] does not carry — but this function never
/// reaches that loop, since it never has anything but `Float32` blocks to
/// hand it.
///
/// `peak_live_buffers()` on the result is always `None` here (see
/// [`Evaluated`]'s own doc): [`StaticArena`]'s buffers persist across calls
/// rather than being freed and recounted per call, so there is no
/// per-call high-water mark to report, the same reason
/// [`evaluate_named_with_arena`] itself reports `None`.
pub fn evaluate_named(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, &[f32])],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    evaluate_named_via_arena(program, symbols, named, outputs)
}

/// One output-set-dependent decision the planner made while building
/// `resolved` from `program` -- never evaluation state, only what
/// [`bind::bind`], [`dead_resolved_nodes`], [`node_retirement`],
/// `epilogue_fuse_plan` and `layer_norm_cluster_plan` decided given the
/// requested output set. `into` names the [`BoundOp`] the decision folded
/// `node` into, or the resolved position it retired against; `consumers`
/// is the fused-form consumer count at the moment of the decision, where
/// known (`0` where the underlying pass does not track it, e.g. `dead`/`retired`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanDecision {
    pub node: NodeId,
    pub kind: &'static str,
    pub decision: &'static str,
    pub into: Option<NodeId>,
    pub consumers: u32,
}

/// Same admission pipeline [`evaluate_named`] runs -- [`shape::infer`],
/// [`bind::bind`], [`dead_resolved_nodes`], [`node_retirement`], and the
/// epilogue/layer-norm rewrite worklist -- but returns every decision those
/// passes made instead of allocating buffers and interpreting `resolved`.
/// `named` is accepted only for signature parity with [`evaluate_named`]:
/// every decision here depends on `program`/`symbols`/`outputs` alone, never
/// on tensor bytes, so a caller comparing two output sets on the identical
/// program can call this twice and diff the two `Vec<PlanDecision>` directly
/// -- built for a downstream consumer whose own test harness cannot easily
/// enable this crate's `instrument` feature to read the equivalent `debug!`
/// events `epilogue_fuse_plan`/`layer_norm_cluster_plan`/
/// [`dead_resolved_nodes`]/[`node_retirement`] already emit at the identical
/// decision points.
///
/// # Errors
/// The same errors [`evaluate_named`] raises during planning: shape
/// inference failure, a non-`Float32` node, or an out-of-range output.
pub fn plan_trace_named(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, &[f32])],
    outputs: &[NodeId],
) -> Result<Vec<PlanDecision>, TensorError> {
    let _ = named;
    let shapes = shape::infer(program, symbols)?;
    reject_non_float32(program, &BTreeSet::new())?;
    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };
    reject_non_float32_outputs(program, &BTreeSet::new(), &effective_outputs)?;

    let resolved = bind::bind(
        program,
        &shapes,
        &effective_outputs,
        NumericPolicy::bit_exact(),
    )?;
    let mut decisions = Vec::new();

    let bound_nodes: BTreeSet<NodeId> = resolved.iter().map(|computed| computed.node).collect();
    for index in 0..program.len() {
        let node = NodeId(index as u32);
        if bound_nodes.contains(&node) {
            continue;
        }
        let into = resolved
            .iter()
            .find(|computed| {
                computed
                    .operands()
                    .iter()
                    .any(|(operand, ..)| *operand == node)
            })
            .map(|computed| computed.node);
        decisions.push(PlanDecision {
            node,
            kind: "raw_op",
            decision: "absorbed",
            into,
            consumers: 0,
        });
    }

    for node in dead_resolved_nodes(&resolved, &effective_outputs) {
        decisions.push(PlanDecision {
            node,
            kind: "resolved_node",
            decision: "dead",
            into: None,
            consumers: 0,
        });
    }

    for (position, retired_here) in node_retirement(&resolved, &effective_outputs)
        .iter()
        .enumerate()
    {
        for &node in retired_here {
            decisions.push(PlanDecision {
                node,
                kind: "resolved_node",
                decision: "retired",
                into: resolved.get(position).map(|computed| computed.node),
                consumers: 0,
            });
        }
    }

    let (epilogue_fuse, layer_norm_cluster, _fires) = run_rewrite_worklist(
        &resolved,
        program.len(),
        &effective_outputs,
        &BTreeMap::new(),
    );
    for (&reduce_node, &(index, ..)) in &epilogue_fuse {
        decisions.push(PlanDecision {
            node: reduce_node,
            kind: "epilogue_fuse",
            decision: "fused",
            into: resolved.get(index).map(|computed| computed.node),
            consumers: 1,
        });
    }
    for (&r2_node, cluster) in &layer_norm_cluster {
        decisions.push(PlanDecision {
            node: r2_node,
            kind: "layer_norm_cluster",
            decision: "fused",
            into: resolved
                .get(cluster.tail_index)
                .map(|computed| computed.node),
            consumers: 1,
        });
    }

    Ok(decisions)
}

/// Same contract as [`evaluate`], plus one capability a caller cannot get
/// from that function: `scratch` seeds this run's buffer-reuse pool instead
/// of starting it empty, and receives back whatever the pool held once the
/// run finished — most usefully, whatever a prior call's
/// [`Evaluated::into_scratch`] deposited into it. A caller that runs the
/// same program (or same-shaped programs) repeatedly and feeds each
/// result's storage back through `into_scratch` skips `evaluate`'s per-call
/// output allocation on every call after the first, without this crate ever
/// exposing an allocator or a persistent handle — the caller still decides
/// `scratch`'s lifetime, this function only ever borrows it. `evaluate`
/// itself is exactly this function with `scratch` starting, and ending,
/// empty.
pub fn evaluate_with_scratch(
    program: &[Op],
    symbols: &[u64],
    blocks: &[&[f32]],
    outputs: &[NodeId],
    scratch: &mut Vec<Vec<f32>>,
) -> Result<Evaluated, TensorError> {
    evaluate_pooled(program, symbols, blocks, outputs, scratch)
}

/// A [`bind::bind`]-shaped execution plan whose per-node output storage is
/// allocated exactly once and reused, unchanged in size, across every call
/// to [`evaluate_named_with_arena`] against the SAME `program` — the
/// static-arena counterpart to [`evaluate_named`]. [`evaluate_named`] itself
/// now reaches this SAME machinery by default, through a bounded process
/// cache (`ARENA_CACHE`) rather than a caller-held handle — see that
/// function's own doc. What a caller still gets ONLY by holding a
/// `StaticArena` directly: a GUARANTEED warm arena regardless of how many
/// OTHER distinct `(program, symbols, outputs)` shapes the process is
/// currently evaluating (the cache's own bound, `ARENA_CACHE_CAPACITY`,
/// can evict a shape `evaluate_named` alone would otherwise have to rebuild
/// on the next call), no mutex checkout/checkin per call, and the
/// in-place rebind lever ([`evaluate_named_with_arena_in_place`]) a shared
/// cache cannot offer since a rebind aliases buffers the caller alone
/// still owns identity of.
///
/// A caller running an identically-shaped program hundreds of times in a
/// row (a training loop, per `docs/discipline.md` ROW 164) that wants that
/// guarantee, rather than betting on the shared cache staying warm, still
/// builds and holds its own arena — the ONLY thing THIS type exists to buy
/// beyond what `evaluate_named` already does for free, per this crate's own
/// binary-question gate (`AGENTS.md`,
/// guiding-principles §1: "what can a caller do that they could not
/// before").
pub struct StaticArena {
    pub(super) resolved: Vec<BoundOp>,
    pub(super) shapes: shape::Shapes,
    pub(super) effective_outputs: Vec<NodeId>,
    pub(super) root: NodeId,
    pub(super) input_names: Vec<(NodeId, String)>,
    pub(super) buffers: Vec<Option<Vec<f32>>>,
    /// Every [`input_names`] node [`build_static_arena_with_constants`]'s own
    /// `constant_inputs` loop already bound with real data before this arena
    /// was handed back — `bind_named_inputs_into_arena`'s own `require_all`
    /// check reads this so a caller who stops re-passing an already-bound
    /// weight every call is not an unbound-input error: the byte compare
    /// this field lets a caller skip is the same 132 MB memcmp
    /// `docs/discipline.md`'s rebind-identity task measured at ~4.46 ms/call
    /// of BGE's sealed evaluate. A name absent here still goes through the
    /// full bind-and-compare path the instant a caller DOES re-pass it
    /// (`bind_named_inputs_into_arena`'s `Some(data)` arm never consults
    /// this set), so a re-sent constant is exactly as safe as before.
    pub(super) constant_bound: BTreeSet<NodeId>,
    /// Nodes in `resolved` with zero consumers among every other resolved
    /// node's own operands and no membership in `effective_outputs` —
    /// `docs/discipline.md` ROW 166's own dead node (`bind`'s
    /// `differentiate_elementwise` builds both a `Multiply`'s operand
    /// contributions before `route_contribution`'s `is_unwanted_input` gate
    /// ever runs, so the unwanted one is bound but never read). Computed
    /// once here, against `BoundOp::operands()` — which already carries
    /// every source a fused/composed body absorbed, since fusion moves a
    /// source node into the fusing op's own physical operand list rather
    /// than leaving a separate reference behind — so a node consumed only
    /// inside a composed body is correctly counted live. `run_resolved_nodes_in_arena`
    /// skips these; every other field above (`resolved`, `shapes`,
    /// `effective_outputs`, `bind::bind`'s own fusion decisions) is
    /// untouched, which is what keeps every fusion/eligibility decision
    /// exactly as ROW 166's graph-level attempt found it, before either the
    /// node it deleted or the sibling that regressed once it was gone
    /// existed as candidates to remove.
    pub(super) dead: BTreeSet<NodeId>,
    /// `resolved` nodes whose [`BoundOpKind`] is `Constant` or `Iota` —
    /// `docs/discipline.md` ROW 174's own found lever: a `Constant`'s value
    /// is a literal baked into the `BoundOp` at `bind::bind` time
    /// (`run_constant`'s whole body is `output.fill(value)`, no operand
    /// read) and an `Iota`'s output is derived purely from its own position
    /// in `BoundOp::extents` (`run_iota`, also no operand read) — both are
    /// call-invariant by construction, since neither `BoundOpKind::operands()`
    /// entry exists for either variant (`bind.rs`'s own `operands()` match
    /// returns `&[]` for `Iota | Constant`). Run once inside
    /// `build_static_arena`, then skipped forever by
    /// `run_resolved_nodes_in_arena`, the same "computed once, cheap to
    /// consult" shape `dead` above already uses. Never overlaps `dead`: a
    /// node here is either a requested output (kept, still run once) or
    /// consumed by a live sibling (kept, still run once) — `dead_resolved_nodes`
    /// only drops nodes with zero consumers and no output membership, which
    /// this field does not gate on.
    pub(super) static_nodes: BTreeSet<NodeId>,
    /// Law 6∘5 weight packing (`docs/rewrite-algebra.md` section 6): every
    /// width-tile [`Keep::Reduce`] node whose `b` operand was named in
    /// [`build_static_arena_with_constants`]'s own `constant_inputs`, keyed
    /// by the REDUCE node's [`NodeId`] (not the weight's), since
    /// [`run_resolved_nodes_in_arena`] looks this map up per resolved node
    /// on its way to deciding whether to route through
    /// [`run_reduce`]'s packed arm. Always empty off `aarch64` or with the
    /// bench/test escape valve `set_pack_at_plan_time_enabled` (aarch64-only)
    /// flipped off -- [`build_packed_width_panels`] is the only populator and it checks
    /// the valve before scanning.
    pub(super) packed_width_panels: BTreeMap<NodeId, PackedWidthPanels>,
    /// `run_rewrite_worklist`'s law 1/2 admission (`docs/rewrite-algebra.md`
    /// §8), computed exactly once here rather than per
    /// [`evaluate_named_with_arena`] call -- the entire point of an arena is
    /// amortizing work that would otherwise repeat every step, and a
    /// fusion plan is call-invariant for exactly the same reason
    /// `resolved`/`shapes` are: it is derived purely from `resolved`'s own
    /// structure, which never changes across calls against this arena.
    /// Reduce `NodeId` -> (consumer's `resolved` index, fire position,
    /// kind, hoist axis), identical shape to
    /// [`evaluate_quantized_with_scratch`]'s own local of the same name.
    pub(super) epilogue_fuse_plan: SingleHopEpiloguePlan,
    /// `run_rewrite_worklist`'s law 2 cluster upgrade, same call-invariance
    /// argument as `epilogue_fuse_plan` above.
    pub(super) layer_norm_cluster_plan: BTreeMap<NodeId, LayerNormClusterPlan>,
    /// Union of every node `epilogue_fuse_plan`/`layer_norm_cluster_plan`
    /// makes dead weight -- `run_resolved_nodes_in_arena` skips these
    /// exactly like `dead`/`static_nodes`, except the skip is a fusion
    /// decision rather than a liveness one: the node IS a real,
    /// non-dead operand elsewhere in `resolved` (so it stays out of
    /// `dead`), its buffer is simply filled by a fused kernel at
    /// `epilogue_fuse_fire_at`/`layer_norm_cluster_fire_at`'s own
    /// position instead of by `run_node_into` at its own natural position.
    pub(super) epilogue_fuse_skip: BTreeSet<NodeId>,
    /// `epilogue_fuse_plan` grouped by `fire_position`, same shape and
    /// same reasoning as `evaluate_quantized_with_scratch`'s own local:
    /// lets the arena's per-position loop ask "does anything fire here"
    /// in O(1) instead of scanning the whole plan every position.
    pub(super) epilogue_fuse_fire_at: BTreeMap<usize, Vec<NodeId>>,
    /// `layer_norm_cluster_plan` grouped by `fire_position`, sibling to
    /// `epilogue_fuse_fire_at` above.
    pub(super) layer_norm_cluster_fire_at: BTreeMap<usize, Vec<NodeId>>,
}

// `StaticArena::dead`'s own computation -- every `resolved` node neither
// consumed by another resolved node nor named in `effective_outputs`
// (`docs/discipline.md` ROW 166/167) -- now lives at
// [`bind::dead_resolved_nodes`] so a stateless GPU backend can reuse the
// identical scan to drop a dead node from its own dispatch list outright,
// instead of this arena's own execution-time skip path (which a driver with
// no persistent arena has no analogue for).

/// `StaticArena::static_nodes` documents: every LIVE `resolved` node whose
/// [`BoundOpKind`] is `Constant` or `Iota` — excludes anything already in
/// `dead` (no reason to run a dead constant even once) since callers pass
/// `dead` alongside this set to build the union `run_resolved_nodes_in_arena`
/// skips. See `docs/discipline.md` ROW 174.
pub(super) fn static_resolved_nodes(
    resolved: &[BoundOp],
    dead: &BTreeSet<NodeId>,
) -> BTreeSet<NodeId> {
    resolved
        .iter()
        .filter(|computed| {
            matches!(
                computed.kind,
                BoundOpKind::Constant { .. } | BoundOpKind::Iota
            )
        })
        .map(|computed| computed.node)
        .filter(|node| !dead.contains(node))
        .collect()
}

/// Builds a [`StaticArena`] for `program`: runs shape inference and
/// [`bind::bind`] once, then pre-sizes every node's output buffer (both
/// [`Op::Input`] slots and every computed [`BoundOp`]) at its final,
/// call-invariant length. Every subsequent [`evaluate_named_with_arena`]
/// call against this arena reuses these SAME allocations — no per-step
/// `shape::infer`, no per-step `bind::bind`, no per-step `Vec` allocation,
/// and no [`evaluate_with_scratch`]-style runtime best-fit search over a
/// shared pool (`docs/discipline.md` ROW 159 measured that search losing)
/// — each node's buffer lives at a fixed index for the arena's whole
/// lifetime.
///
/// # Errors
/// The same shape/dtype/output errors [`evaluate_named`] itself raises,
/// since this runs the identical `prepare`-shaped validation once up front
/// instead of on every call.
pub fn build_static_arena(
    program: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
) -> Result<StaticArena, TensorError> {
    build_static_arena_with_constants(program, symbols, outputs, &[])
}

/// [`build_static_arena`], plus one capability it does not have: `constant_inputs`
/// binds a set of [`Op::Input`] names with their real, call-invariant data
/// (a weight tensor loaded once from a checkpoint, never re-derived per
/// step) BEFORE the static pass runs, and every width-tile [`Keep::Reduce`]
/// node reading one of them as its 2-D `b` operand gets that operand
/// packed once, right here, into a panel layout the width-tile kernel reads
/// sequentially (law 6∘5, `docs/rewrite-algebra.md` section 6, private
/// `PackedWidthPanels`/`build_packed_width_panels` within this module).
/// `constant_inputs` entries are bound exactly like an initial
/// [`evaluate_named_with_arena`] call's `named` would bind them; a caller
/// that later re-sends the SAME name via `evaluate_named_with_arena`
/// overwrites the identical bytes (weights do not change across BGE
/// sentences), so the packed panel built here stays valid for the arena's
/// whole lifetime without needing to be rebuilt.
///
/// Default-on since `docs/discipline.md` ROW 207's promotion, `aarch64`
/// only — off aarch64, or with the bench/test escape valve
/// `set_pack_at_plan_time_enabled` (aarch64-only) flipped off, this is
/// byte-for-byte [`build_static_arena`] plus binding `constant_inputs` into
/// the arena's input buffers, no packing performed.
///
/// # Errors
/// The same errors [`build_static_arena`] raises, plus
/// [`TensorError::UnboundInputName`] if a `constant_inputs` name is not one
/// of `program`'s [`Op::Input`] names, and [`TensorError::InputSizeMismatch`]
/// if its data does not match that input's inferred shape.
pub fn build_static_arena_with_constants(
    program: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
    constant_inputs: &[(&str, &[f32])],
) -> Result<StaticArena, TensorError> {
    let shapes = shape::infer(program, symbols)?;
    reject_non_float32(program, &BTreeSet::new())?;

    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };
    reject_non_float32_outputs(program, &BTreeSet::new(), &effective_outputs)?;

    let block_nodes = block_node_ids(program);
    let mut input_names = Vec::with_capacity(block_nodes.len());
    let mut buffers: Vec<Option<Vec<f32>>> = vec![None; program.len()];
    for node in &block_nodes {
        let name = program[node.0 as usize]
            .name()
            .ok_or(TensorError::UnnamedInput(*node))?;
        input_names.push((*node, String::from(name)));
        buffers[node.0 as usize] = Some(vec![0.0f32; element_count(shapes.of(*node))]);
    }

    let resolved = bind::bind(
        program,
        &shapes,
        &effective_outputs,
        NumericPolicy::bit_exact(),
    )?;
    for computed in &resolved {
        buffers[computed.node.0 as usize] = Some(vec![0.0f32; node_output_len(computed)]);
        // `state_out` (ROW 547, `docs/discipline.md`): `GatedDeltaNet`'s own
        // second output never appears as any resolved node's `.node`, so the
        // loop above never sizes its slot -- size it here, once, the same
        // way every other resolved node's own buffer is pre-sized.
        if let BoundOpKind::GatedDeltaNet {
            state_out,
            head_k_dim,
            head_v_dim,
            num_v_heads,
            ..
        } = &computed.kind
        {
            let state_len = (*head_k_dim * *head_v_dim * *num_v_heads) as usize;
            buffers[state_out.0 as usize] = Some(vec![0.0f32; state_len]);
        }
        // ROW 569: `MoeTopK`'s own `2 * top_k` extra outputs never appear as
        // any resolved node's `.node` either, for the same reason `state_out`
        // does not -- size every one of them here, one scalar slot apiece
        // (this slice's `n_tokens == 1` decode-only shape).
        if let BoundOpKind::MoeTopK {
            routes,
            weights,
            weight_total,
            ..
        } = &computed.kind
        {
            for extra_node in moe_topk_extra_node_order(routes, weights, *weight_total) {
                buffers[extra_node.0 as usize] = Some(vec![0.0f32; 1]);
            }
        }
        // `round_outputs[1..]` (ROW 569's own shape, generalized): the k-1
        // round-sibling nodes this fusion dropped from `resolved` never
        // appear as any resolved node's own `.node` either -- size each
        // one's slot here, same length as `round_outputs[0]`'s own
        // `node_output_len` (every round shares one output shape).
        if let BoundOpKind::RoundBatchedReduce { round_outputs, .. } = &computed.kind {
            let round_len = node_output_len(computed);
            for extra_node in round_outputs.iter().skip(1) {
                buffers[extra_node.0 as usize] = Some(vec![0.0f32; round_len]);
            }
        }
    }
    let dead = dead_resolved_nodes(&resolved, &effective_outputs);
    let static_nodes = static_resolved_nodes(&resolved, &dead);

    // Computed exactly once, here, never inside `run_resolved_nodes_in_arena`
    // -- fusion is call-invariant over `resolved`'s own structure, the same
    // amortization argument this arena already applies to `shapes`/`resolved`
    // themselves. `quantized_weights` is always empty: a `StaticArena` is
    // f32-only (`reject_non_float32` above), so `epilogue_fuse_plan`'s own
    // quantized-weight exclusion guard is a no-op here, matching
    // `evaluate_named`'s own all-`Float32` call into this same machinery.
    let (epilogue_fuse_plan, layer_norm_cluster_plan, rewrite_fires) = run_rewrite_worklist(
        &resolved,
        program.len(),
        &effective_outputs,
        &BTreeMap::new(),
    );
    record_rewrite_engine_fires(&rewrite_fires);
    let epilogue_fuse_skip: BTreeSet<NodeId> = epilogue_fuse_plan
        .values()
        .map(|(index, ..)| resolved[*index].node)
        .chain(
            layer_norm_cluster_plan
                .values()
                .flat_map(|cluster| cluster.skip),
        )
        .collect();
    let epilogue_fuse_fire_at: BTreeMap<usize, Vec<NodeId>> = {
        let mut grouped: BTreeMap<usize, Vec<NodeId>> = BTreeMap::new();
        for (reduce_node, (_, fire_position, ..)) in &epilogue_fuse_plan {
            if layer_norm_cluster_plan.contains_key(reduce_node) {
                continue;
            }
            grouped
                .entry(*fire_position)
                .or_default()
                .push(*reduce_node);
        }
        grouped
    };
    let layer_norm_cluster_fire_at: BTreeMap<usize, Vec<NodeId>> = {
        let mut grouped: BTreeMap<usize, Vec<NodeId>> = BTreeMap::new();
        for (reduce_node, cluster) in &layer_norm_cluster_plan {
            grouped
                .entry(cluster.fire_position)
                .or_default()
                .push(*reduce_node);
        }
        grouped
    };

    // bound before the static pass so a `Constant`/`Iota` node consuming one
    // of these (unlikely for a weight but not disallowed) sees the real
    // value too — matches `bind_named_inputs_into_arena`'s own per-slot
    // copy, just against the zero-initialized buffer this function itself
    // just built above.
    let mut constant_bound = BTreeSet::new();
    for (name, data) in constant_inputs {
        let (node, _) = input_names
            .iter()
            .find(|(_, candidate)| candidate == name)
            .ok_or_else(|| TensorError::UnboundInputName(String::from(*name)))?;
        let slot = buffers[node.0 as usize].as_mut().ok_or(TensorError::NotLowerable {
            node: *node,
            reason: "static arena has no pre-sized slot for this constant input -- build_static_arena did not size it",
        })?;
        if slot.len() != data.len() {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected: slot.len(),
                found: data.len(),
            });
        }
        slot.copy_from_slice(data);
        constant_bound.insert(*node);
    }

    for computed in &resolved {
        if !static_nodes.contains(&computed.node) {
            continue;
        }
        let node_index = computed.node.0 as usize;
        let mut output = buffers[node_index].take().ok_or(TensorError::NotLowerable {
            node: computed.node,
            reason: "static arena has no pre-sized slot for this resolved node -- build_static_arena did not size it",
        })?;
        run_node_into(computed, &buffers, None, None, None, false, &mut output)?;
        buffers[node_index] = Some(output);
    }

    #[cfg(target_arch = "aarch64")]
    let packed_width_panels =
        build_packed_width_panels(&resolved, &shapes, &buffers, &input_names, constant_inputs);
    #[cfg(not(target_arch = "aarch64"))]
    let packed_width_panels = BTreeMap::new();

    Ok(StaticArena {
        resolved,
        shapes,
        effective_outputs,
        root,
        input_names,
        buffers,
        constant_bound,
        dead,
        static_nodes,
        packed_width_panels,
        epilogue_fuse_plan,
        layer_norm_cluster_plan,
        epilogue_fuse_skip,
        epilogue_fuse_fire_at,
        layer_norm_cluster_fire_at,
    })
}

/// Runs `arena`'s already-[`bind::bind`]-resolved program once against this
/// step's `named` host buffers, writing every input and every computed
/// node's output into the SAME per-node storage [`build_static_arena`]
/// sized once. The only allocation on this call's own path is the small,
/// output-count-sized clone [`Evaluated`] needs to hand results back to a
/// caller that runs another step against the same arena immediately after
/// — the arena's own buffers must survive this call, so results are copied
/// out, never moved out (unlike `evaluate_pooled`'s one-shot table).
///
/// # Errors
/// [`TensorError::UnboundInputName`] if `named` has no entry for one of
/// the program's [`Op::Input`] names; [`TensorError::InputSizeMismatch`] if
/// a bound buffer's length no longer matches the size [`build_static_arena`]
/// fixed for it (a genuinely different-shaped call, not a training step).
pub fn evaluate_named_with_arena(
    arena: &mut StaticArena,
    named: &[(&str, &[f32])],
) -> Result<Evaluated, TensorError> {
    bind_named_inputs_into_arena(arena, named, true)?;
    run_resolved_nodes_in_arena(arena)?;

    let results = arena
        .effective_outputs
        .iter()
        .map(|node| {
            let shape = arena.shapes.of(*node).to_vec();
            let data = arena.buffers[node.0 as usize]
                .as_deref()
                .unwrap_or(&[])
                .to_vec();
            (*node, shape, data)
        })
        .collect();

    Ok(Evaluated::from_parts(arena.root, results, None))
}

/// Reinterprets `slice` as raw bytes, no copy -- `f32` has no padding and
/// `u8` has alignment 1, so any valid `&[f32]` pointer/length is a valid
/// `&[u8]` view of the identical `size_of_val(slice)` bytes.
///
/// # Safety
/// None required beyond `slice` itself being a valid `&[f32]`, which the
/// caller already has by construction -- this is a pointer-cast view, never
/// a mutation, so no aliasing or lifetime hazard beyond what `slice` already
/// carries.
pub(super) fn f32_slice_as_bytes(slice: &[f32]) -> &[u8] {
    // SAFETY: `f32` is `Copy`, has no padding bytes, and `u8`'s alignment
    // (1) never exceeds `f32`'s -- reinterpreting the same backing memory as
    // `size_of_val(slice)` bytes is always valid.
    unsafe {
        core::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), core::mem::size_of_val(slice))
    }
}

/// [`evaluate_named_with_arena`]'s own input-binding loop, factored out so
/// [`evaluate_named_with_arena_in_place`] can reuse it with `require_all =
/// false`: a name absent from `named` is treated as "already correct in
/// the arena" (the in-place rebind lever put it there) rather than an
/// error, instead of duplicating the loop body.
pub(super) fn bind_named_inputs_into_arena(
    arena: &mut StaticArena,
    named: &[(&str, &[f32])],
    require_all: bool,
) -> Result<(), TensorError> {
    #[cfg(feature = "instrument")]
    let compare_started = instrument::read_ticks();
    for (node, name) in &arena.input_names {
        let found = named
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, data)| *data);
        let data = match found {
            Some(data) => data,
            // A name `build_static_arena_with_constants`'s own `constant_inputs`
            // loop already bound is correct in the arena right now without
            // being re-passed -- `require_all` asks "is every genuine input
            // satisfied", not "is every name present in `named`", and a
            // constant-bound weight satisfies the former by construction.
            None if require_all && arena.constant_bound.contains(node) => continue,
            None if require_all => return Err(TensorError::UnboundInputName(name.clone())),
            None => continue,
        };
        let slot = arena.buffers[node.0 as usize].as_mut().ok_or(TensorError::NotLowerable {
            node: *node,
            reason: "static arena has no pre-sized slot for this input node -- build_static_arena did not size it",
        })?;
        if slot.len() != data.len() {
            return Err(TensorError::InputSizeMismatch {
                node: *node,
                expected: slot.len(),
                found: data.len(),
            });
        }
        // `checkout_arena` derives `constant_inputs` from `program` structure
        // alone, so a name it treats as call-invariant is a STRUCTURAL guess
        // (an `Op::Input` leaf, no more), never a caller promise -- this is
        // the guess's own soundness check. A rebind that changes the bytes
        // proves the guess wrong for this node, so every panel packed from it
        // is dropped here, before `run_resolved_nodes_in_arena` ever reads
        // one -- the arena falls back to `run_node_into`'s unpacked path for
        // that node for the rest of its life. A rebind that repeats the same
        // bytes (every real BGE weight, every call) leaves `packed_width_panels`
        // untouched, so the amortized packing gain survives.
        //
        // Compared as raw BYTES (`f32_slice_as_bytes`), not `f32::eq` --
        // this loop's own contract is "did the bytes change", the exact
        // question a byte compare answers and `[f32]::eq` does not: IEEE
        // `NaN != NaN` would report "changed" for a bit-identical resend,
        // and `-0.0 == 0.0` would report "unchanged" for a genuine sign-bit
        // flip, missing a real change the buffer needed. Byte comparison
        // also lets `[u8]::eq` take the stdlib's memcmp fast path instead of
        // a scalar per-element float compare -- measured 11.4ms/call of a
        // ~19ms step for BGE's 132MB of initializers before this change
        // (`docs/discipline.md`, hit-cost regression task, 2026-09-01).
        if f32_slice_as_bytes(slot.as_slice()) != f32_slice_as_bytes(data) {
            slot.copy_from_slice(data);
            if !arena.packed_width_panels.is_empty() {
                arena
                    .packed_width_panels
                    .retain(|_, panel| panel.source != *node);
            }
        }
    }
    #[cfg(feature = "instrument")]
    instrument::record_bind_rebind_compare_ticks(instrument::elapsed_ticks(compare_started));
    Ok(())
}

/// `docs/discipline.md` ROW 174/234's own per-node profile: a coarse,
/// four-way label distinguishing [`BoundOpKind`] variants, not the
/// finer `Path::{DotFast,WidthFast,ConvTile,Generic}` route census
/// [`instrument::reduce_path_totals`]/[`instrument::reduce_gemm_path_totals`]
/// already answer for reduce-shaped nodes specifically -- this label is
/// keyed by [`NodeId`] instead, the axis nothing else in this module covers.
#[cfg(feature = "instrument")]
pub(super) fn arena_node_kind_label(kind: &BoundOpKind) -> &'static str {
    match kind {
        BoundOpKind::Elementwise { .. } => "elementwise",
        BoundOpKind::Reduce { .. } => "reduce",
        BoundOpKind::RoundBatchedReduce { .. } => "round_batched_reduce",
        BoundOpKind::Iota => "iota",
        BoundOpKind::Constant { .. } => "constant",
        BoundOpKind::CachedAttention { .. } => "cached_attention",
        BoundOpKind::GatedDeltaNet { .. } => "gated_delta_net",
        BoundOpKind::MoeTopK { .. } => "moe_topk",
    }
}

/// [`evaluate_named_with_arena`]'s own resolved-node execution loop,
/// factored out so [`evaluate_named_with_arena_in_place`] shares the
/// identical execution path rather than a second copy of it.
pub(super) fn run_resolved_nodes_in_arena(arena: &mut StaticArena) -> Result<(), TensorError> {
    #[cfg(feature = "epilogue-profile-probe")]
    let reduce_nodes = epilogue_profile_reduce_flags(&arena.resolved, arena.buffers.len());
    for position in 0..arena.resolved.len() {
        let computed = &arena.resolved[position];
        let node = computed.node;
        // `arena.epilogue_fuse_skip` sits alongside `dead`/`static_nodes` as
        // a third reason this position's own `run_node_into` never runs --
        // unlike the other two, the skipped node is neither dead nor
        // call-invariant: its buffer is filled below, by a fused kernel, at
        // this SAME plan's own `fire_position` (which may be this position
        // or an earlier one, never later -- `epilogue_fuse_plan`/
        // `layer_norm_cluster_plan`'s own `fire_position >= index` guard).
        let skip_compute = arena.dead.contains(&node)
            || arena.static_nodes.contains(&node)
            || arena.epilogue_fuse_skip.contains(&node);
        if !skip_compute {
            let node_index = node.0 as usize;
            let mut output = arena.buffers[node_index].take().ok_or(TensorError::NotLowerable {
                node,
                reason: "static arena has no pre-sized slot for this resolved node -- build_static_arena did not size it",
            })?;
            #[cfg(feature = "epilogue-profile-probe")]
            let profile_start = std::time::Instant::now();
            #[cfg(feature = "instrument")]
            let node_profile_started = instrument::read_ticks();
            // law 6∘5: a node with a plan-time-packed `b` operand skips
            // `run_node_into`'s dispatch entirely and calls `run_reduce`
            // directly with the packed panel -- `run_node_into`'s own
            // signature stays untouched (13 other call sites, `docs/
            // rewrite-algebra.md` admission rule scopes this lever to the
            // arena's own execution loop, not a crate-wide dispatch change).
            // Always `None`/never-taken off `aarch64` or with the bench/test
            // escape valve `set_pack_at_plan_time_enabled(false)`, since
            // `packed_width_panels` is always empty there.
            // `state_out` (ROW 547, `docs/discipline.md`): kept empty and
            // unread for every kind but `GatedDeltaNet`. `moe_topk_extra`
            // (ROW 569) is the same shape, generalized: kept empty and
            // unread for every kind but `MoeTopK`.
            let mut gdn_state = Vec::new();
            let mut moe_topk_extra = Vec::new();
            let mut round_extra: Vec<Vec<f32>> = Vec::new();
            match arena.packed_width_panels.get(&node) {
                Some(packed) => run_reduce(computed, &arena.buffers, &mut output, Some(packed))?,
                None => run_node_into_with_round_sink(
                    computed,
                    &arena.buffers,
                    None,
                    None,
                    None,
                    false,
                    &mut output,
                    Some(&mut gdn_state),
                    Some(&mut moe_topk_extra),
                    Some(&mut round_extra),
                )?,
            }
            #[cfg(feature = "epilogue-profile-probe")]
            epilogue_profile_record(
                computed,
                &reduce_nodes,
                profile_start.elapsed().as_nanos() as u64,
            );
            // `docs/discipline.md` ROW 174/234: the only counter in this
            // loop keyed by NodeId rather than kind/phase -- see
            // `instrument::record_arena_node_ticks`'s own doc for why this
            // is a cached-bool no-op unless a caller has explicitly opted
            // into `PROXIMA_ARENA_PER_NODE=1`.
            #[cfg(feature = "instrument")]
            instrument::record_arena_node_ticks(
                node,
                arena_node_kind_label(&computed.kind),
                &computed.extents,
                instrument::elapsed_ticks(node_profile_started),
            );
            arena.buffers[node_index] = Some(output);
            if let BoundOpKind::GatedDeltaNet { state_out, .. } = &computed.kind {
                arena.buffers[state_out.0 as usize] = Some(gdn_state);
            }
            if let BoundOpKind::MoeTopK {
                routes,
                weights,
                weight_total,
                ..
            } = &computed.kind
            {
                for (extra_node, value) in moe_topk_extra_node_order(routes, weights, *weight_total)
                    .zip(moe_topk_extra.iter().copied())
                {
                    arena.buffers[extra_node.0 as usize] = Some(vec![value]);
                }
            }
            if let BoundOpKind::RoundBatchedReduce { round_outputs, .. } = &computed.kind {
                for (extra_node, value) in round_outputs.iter().skip(1).zip(round_extra) {
                    arena.buffers[extra_node.0 as usize] = Some(value);
                }
            }
        }
        // `evaluate_quantized_with_scratch`'s own identical fire blocks,
        // ported verbatim onto arena buffers: same maps, same admission,
        // same monomorphic kernels, only the buffer type (`Vec<f32>`
        // in-place instead of `take_or_allocate`'s pooled `Cow::Owned`)
        // differs, since an arena's own consumer slot is already pre-sized
        // and never freed -- `take()` it, write into it, put it back, no
        // allocation on this call's own path.
        if let Some(reduce_nodes) = arena.epilogue_fuse_fire_at.get(&position) {
            for reduce_node in reduce_nodes.clone() {
                let Some(&(consumer_index, _, kind, hoist_axis, epilogue_slots)) =
                    arena.epilogue_fuse_plan.get(&reduce_node)
                else {
                    continue;
                };
                let consumer_node = arena.resolved[consumer_index].node;
                let mut output = arena.buffers[consumer_node.0 as usize].take().ok_or(TensorError::NotLowerable {
                    node: consumer_node,
                    reason: "static arena has no pre-sized slot for this epilogue-fused node -- build_static_arena did not size it",
                })?;
                let reduce_values = arena.buffers[reduce_node.0 as usize]
                    .as_deref()
                    .unwrap_or(&[]);
                let fuse_started = std::time::Instant::now();
                apply_epilogue_fused_monomorphic(
                    EpilogueFuseKernel {
                        kind,
                        hoist_axis,
                        slots: epilogue_slots,
                    },
                    &arena.resolved[consumer_index],
                    reduce_node,
                    reduce_values,
                    &arena.buffers,
                    &mut output,
                );
                EPILOGUE_FUSE_NANOS.fetch_add(
                    fuse_started.elapsed().as_nanos() as u64,
                    EpilogueFuseOrdering::Relaxed,
                );
                EPILOGUE_FUSE_HITS.fetch_add(1, EpilogueFuseOrdering::Relaxed);
                EPILOGUE_FUSE_ELEMENTS
                    .fetch_add(output.len() as u64, EpilogueFuseOrdering::Relaxed);
                arena.buffers[consumer_node.0 as usize] = Some(output);
            }
        }
        if let Some(reduce_nodes) = arena.layer_norm_cluster_fire_at.get(&position) {
            for reduce_node in reduce_nodes.clone() {
                let Some(cluster) = arena.layer_norm_cluster_plan.get(&reduce_node) else {
                    continue;
                };
                let tail_node = arena.resolved[cluster.tail_index].node;
                let mut output = arena.buffers[tail_node.0 as usize].take().ok_or(TensorError::NotLowerable {
                    node: tail_node,
                    reason: "static arena has no pre-sized slot for this layer-norm-cluster node -- build_static_arena did not size it",
                })?;
                let fuse_started = std::time::Instant::now();
                apply_layer_norm_cluster_fused(
                    &arena.resolved[cluster.tail_index],
                    cluster.x_node,
                    cluster.row_axis,
                    LayerNormTailSlots {
                        reciprocal_n: cluster.tail_reciprocal_n_slot,
                        epsilon: cluster.tail_epsilon_slot,
                        gamma: cluster.tail_gamma_slot,
                        beta: cluster.tail_beta_slot,
                    },
                    &arena.buffers,
                    &mut output,
                );
                LAYER_NORM_CLUSTER_NANOS.fetch_add(
                    fuse_started.elapsed().as_nanos() as u64,
                    EpilogueFuseOrdering::Relaxed,
                );
                LAYER_NORM_CLUSTER_HITS.fetch_add(1, EpilogueFuseOrdering::Relaxed);
                LAYER_NORM_CLUSTER_ELEMENTS
                    .fetch_add(output.len() as u64, EpilogueFuseOrdering::Relaxed);
                arena.buffers[tail_node.0 as usize] = Some(output);
            }
        }
    }
    Ok(())
}

/// One [`evaluate_named`] call's worth of graph identity: the SAME
/// `(program, symbols, outputs)` triple [`build_static_arena`] itself
/// binds. `program`/`symbols`/`outputs` are cloned once, at
/// [`checkin_arena`] time, so a later [`checkout_arena`] call can prove a
/// hit by FULL VALUE equality (`Op` and `NodeId` both derive `PartialEq`)
/// rather than by pointer identity alone — a freed-and-reallocated
/// `Vec<Op>` could otherwise alias a stale entry's address and hand back
/// the wrong arena for a same-address, different-graph call.
pub(super) struct CachedArena {
    pub(super) program: Vec<Op>,
    pub(super) symbols: Vec<u64>,
    pub(super) outputs: Vec<NodeId>,
    pub(super) arena: StaticArena,
}

/// [`ARENA_CACHE`]'s own bound: the most distinct `(program, symbols,
/// outputs)` shapes this process keeps a warm [`StaticArena`] for at once.
/// A caller cycling through more distinct shapes than this evicts the
/// least-recently-checked-in entry (index 0, a FIFO queue) rather than
/// growing without limit — see [`checkin_arena`].
pub(super) const ARENA_CACHE_CAPACITY: usize = 8;

/// Bounded, process-wide memoization of a [`StaticArena`] by graph
/// identity, so [`evaluate_named`]'s own default path reaches
/// [`run_resolved_nodes_in_arena`] — this module's ONE loop over a resolved
/// graph — without a caller ever building or owning an arena itself. Owned
/// by this ONE static, bounded by [`ARENA_CACHE_CAPACITY`] entries (never
/// an unbounded map keyed by pointer identity).
///
/// Guarded by a plain `std::sync::Mutex`, matching this workspace's own
/// established pattern for exactly this shape of state (`proxima-http`,
/// `proxima-intercept`, `proxima-net` all guard synchronous shared state
/// the same way; there is no separate lock-tier crate in this workspace to
/// route through instead). The lock is held only for a checkout/checkin —
/// a linear scan plus a `Vec::remove`/`push` over at most
/// [`ARENA_CACHE_CAPACITY`] entries — NEVER across the run itself: an
/// arena's buffers are mutated in place across a call, so sharing one
/// arena INSTANCE between two overlapping callers would be unsound
/// regardless of locking, and [`checkout_arena`] instead hands out
/// exclusive ownership for the call's duration. Two callers racing the
/// SAME key each get their own private arena and duplicate one build; both
/// check back in afterward, so the cache is warm again for the next call —
/// wasted work under first-use concurrency, never a data race or a wrong
/// answer.
pub(super) static ARENA_CACHE: Mutex<Vec<CachedArena>> = Mutex::new(Vec::new());

/// [`ARENA_CACHE`]'s own lock, recovered rather than propagated on
/// poisoning (`instrument::worker_busy_snapshot`'s own established
/// pattern in this crate): a panic while another caller held the lock
/// leaves the `Vec<CachedArena>` in whatever state that caller's own
/// operation reached, never a torn/half-written entry (every mutation
/// under this lock is a whole-`Vec` `remove`/`push`), so recovering and
/// continuing is safe and matches this crate's own "no unwrap/expect
/// outside tests" discipline better than propagating a poison panic would.
pub(super) fn lock_arena_cache() -> MutexGuard<'static, Vec<CachedArena>> {
    ARENA_CACHE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Engagement evidence for the cache above: how many [`checkout_arena`]
/// calls built a fresh [`StaticArena`] (a miss) versus reused a checked-in
/// one (a hit) — same "a gate that cannot report its N is not a gate" shape
/// [`epilogue_fuse_totals`] already serves for the fusion laws. Read via
/// [`arena_cache_totals`].
pub(super) static ARENA_CACHE_BUILDS: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);
pub(super) static ARENA_CACHE_HITS: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);

/// `(builds, hits)` since process start or the last [`arena_cache_reset`].
#[must_use]
pub fn arena_cache_totals() -> (u64, u64) {
    (
        ARENA_CACHE_BUILDS.load(EpilogueFuseOrdering::Relaxed),
        ARENA_CACHE_HITS.load(EpilogueFuseOrdering::Relaxed),
    )
}

/// Test/bench-only reset: empties [`ARENA_CACHE`] and zeroes
/// [`arena_cache_totals`], so a re-prove command measuring "N arenas built
/// over M calls" gets a clean process-relative count instead of whatever
/// an earlier pass in the same process already warmed.
#[doc(hidden)]
pub fn arena_cache_reset() {
    lock_arena_cache().clear();
    ARENA_CACHE_BUILDS.store(0, EpilogueFuseOrdering::Relaxed);
    ARENA_CACHE_HITS.store(0, EpilogueFuseOrdering::Relaxed);
}

/// [`arena_packed_node_count`], summed over every [`StaticArena`] currently
/// checked into [`ARENA_CACHE`] — the engagement number law 6∘5 packing
/// promises on [`evaluate_named`]'s own default path, readable without a
/// caller holding any arena itself.
#[doc(hidden)]
#[must_use]
pub fn arena_cache_packed_node_count() -> usize {
    lock_arena_cache()
        .iter()
        .map(|entry| arena_packed_node_count(&entry.arena))
        .sum()
}

/// Checks a [`StaticArena`] matching `(program, symbols, outputs)` out of
/// [`ARENA_CACHE`], building one via [`build_static_arena_with_constants`] on
/// a miss. See [`ARENA_CACHE`]'s own doc for the lock discipline and the
/// concurrent-miss case.
///
/// `named` is [`evaluate_named`]'s own binding data, threaded in here so a
/// miss can derive `constant_inputs` from it directly -- `program` alone
/// never carries which of its [`Op::Input`] leaves are call-invariant
/// weights (an ONNX initializer and a graph runtime input both lower to the
/// same `Op::Input` shape, `proxima-onnx/src/lower.rs`'s own module doc:
/// "Initializers become named `Op::Input` leaves"), so `named` -- the one
/// place actual data is present -- is the only thing this function CAN
/// derive from without widening [`evaluate_named`]'s own public signature or
/// asking a caller to say which name is which.
///
/// Every genuine `Op::Input` name in `named` (never a stray extra --
/// filtered against [`block_node_ids`] so an unrelated `named` entry can
/// never trip [`build_static_arena_with_constants`]'s own `UnboundInputName`
/// check) is offered as a `constant_inputs` candidate. This is a STRUCTURAL
/// guess, not a promise: [`build_packed_width_panels`] only actually packs
/// the subset that is ALSO the 2-D `b` operand of a width-tile-eligible
/// `Reduce`, and [`bind_named_inputs_into_arena`]'s own rebind check drops
/// any packed panel the instant a later call proves the guess wrong for that
/// node (see that function's own doc) -- so a name that turns out to vary
/// call-to-call never serves stale data, it just stops being packed.
///
/// `proxima_autograd::train::fit` never reaches this function at all: it
/// holds its own arena via a direct [`build_static_arena`] call (empty
/// `constant_inputs`, `docs/discipline.md` ROW 207's own training-safety
/// invariant), so a trainable parameter is never even offered as a
/// candidate here, structurally, regardless of anything below.
pub(super) fn checkout_arena(
    program: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
    named: &[(&str, &[f32])],
) -> Result<StaticArena, TensorError> {
    {
        let mut cache = lock_arena_cache();
        #[cfg(feature = "instrument")]
        let compare_started = instrument::read_ticks();
        let found = cache.iter().position(|entry| {
            entry.program == program && entry.symbols == symbols && entry.outputs == outputs
        });
        #[cfg(feature = "instrument")]
        instrument::record_checkout_arena_key_compare_ticks(instrument::elapsed_ticks(
            compare_started,
        ));
        if let Some(index) = found {
            ARENA_CACHE_HITS.fetch_add(1, EpilogueFuseOrdering::Relaxed);
            return Ok(cache.remove(index).arena);
        }
    }
    ARENA_CACHE_BUILDS.fetch_add(1, EpilogueFuseOrdering::Relaxed);
    let input_names = block_node_ids(program);
    let constant_inputs: Vec<(&str, &[f32])> = named
        .iter()
        .filter(|(name, _)| {
            input_names
                .iter()
                .any(|node| program[node.0 as usize].name() == Some(*name))
        })
        .copied()
        .collect();
    build_static_arena_with_constants(program, symbols, outputs, &constant_inputs)
}

/// Returns `arena` to [`ARENA_CACHE`] for the next [`checkout_arena`] call
/// against the same `(program, symbols, outputs)` key. Never called after a
/// failed bind/run: an arena `run_resolved_nodes_in_arena` errored out of
/// partway through may hold a `take()`n slot that was never restored, and
/// re-offering that arena for reuse would turn one call's failure into every
/// later call's — dropping it and rebuilding fresh on the next miss is the
/// safe, self-healing choice (see [`checkout_arena`]'s own callers).
pub(super) fn checkin_arena(
    program: &[Op],
    symbols: &[u64],
    outputs: &[NodeId],
    arena: StaticArena,
) {
    let mut cache = lock_arena_cache();
    if cache.len() >= ARENA_CACHE_CAPACITY {
        cache.remove(0);
    }
    cache.push(CachedArena {
        program: program.to_vec(),
        symbols: symbols.to_vec(),
        outputs: outputs.to_vec(),
        arena,
    });
}

/// [`evaluate_named`]'s own body: binds `named` into a cached
/// [`StaticArena`] (see [`checkout_arena`]/[`checkin_arena`]) and runs it
/// through [`run_resolved_nodes_in_arena`] — this module's one loop over a
/// resolved graph. A caller gets arena buffer reuse, law 6∘5 weight packing
/// ([`StaticArena::packed_width_panels`]), and the dead/static-node skip
/// that loop already carries, without ever building or owning an arena
/// itself, and without [`evaluate_named`]'s own signature changing to carry
/// one.
pub(super) fn evaluate_named_via_arena(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, &[f32])],
    outputs: &[NodeId],
) -> Result<Evaluated, TensorError> {
    let root = program
        .len()
        .checked_sub(1)
        .map(|last| NodeId(last as u32))
        .ok_or(TensorError::Empty)?;
    for output in outputs {
        if output.0 as usize >= program.len() {
            return Err(TensorError::UnknownOutput(*output));
        }
    }
    let effective_outputs: Vec<NodeId> = if outputs.is_empty() {
        vec![root]
    } else {
        outputs.to_vec()
    };

    let mut arena = checkout_arena(program, symbols, &effective_outputs, named)?;
    let run = bind_named_inputs_into_arena(&mut arena, named, true)
        .and_then(|()| run_resolved_nodes_in_arena(&mut arena));
    match run {
        Ok(()) => {
            let results = arena
                .effective_outputs
                .iter()
                .map(|node| {
                    let shape = arena.shapes.of(*node).to_vec();
                    let data = arena.buffers[node.0 as usize]
                        .as_deref()
                        .unwrap_or(&[])
                        .to_vec();
                    (*node, shape, data)
                })
                .collect();
            let evaluated = Evaluated::from_parts(arena.root, results, None);
            checkin_arena(program, symbols, &effective_outputs, arena);
            Ok(evaluated)
        }
        Err(error) => Err(error),
    }
}

/// `docs/discipline.md` ROW 181's profile-gate probe: a flat bitmap over
/// `arena.buffers`' own node-index space marking every `Keep::Reduce` fold
/// -- built once per [`run_resolved_nodes_in_arena`] call from
/// `arena.resolved`, execution-level only (reads what `bind::bind` already
/// produced, never calls it). Feeds [`is_post_reduce_epilogue`]'s lookup.
pub(super) fn epilogue_profile_reduce_flags(resolved: &[BoundOp], node_count: usize) -> Vec<bool> {
    let mut flags = vec![false; node_count];
    for computed in resolved {
        if matches!(
            computed.kind,
            BoundOpKind::Reduce {
                keep: Keep::Reduce,
                ..
            }
        ) {
            flags[computed.node.0 as usize] = true;
        }
    }
    flags
}

/// ROW 181's own structural detector: an `Elementwise` node is a
/// post-reduce epilogue when exactly one operand is a non-broadcast,
/// non-gathered reference to a `Keep::Reduce` fold's output and every
/// other operand is rank-0/broadcast (ANY stride zero, per real mnist
/// evidence below -- not ALL strides zero) -- the shape a bias-add,
/// batchnorm-scale-shift, or clip-after-matmul epilogue always takes. A
/// non-broadcast operand that is NOT a reduce output (a residual add
/// between two full tensors, say) rules the node out immediately.
///
/// The broadcast test was first written as "every stride is zero" (true
/// rank-0 only) and measured ZERO epilogue hits on the real mnist graph --
/// `examples/epilogue_dump_shapes.rs`'s dump showed the real per-channel
/// bias operand feeding node 25 carries `strides=[0, 1, 0, 0]` (zero on
/// batch/H/W, nonzero on the channel axis it varies over), which the
/// all-zero test rejected as "some other elementwise shape" -- a detection
/// bug, not an absence of epilogues. "ANY stride zero" is the correct
/// broadcast test: it accepts both a true rank-0 scalar (every stride
/// zero) and a per-axis broadcast operand (bias, batchnorm params) that
/// varies over fewer axes than the reduce operand it accompanies, while
/// still rejecting a second full-shape tensor (a residual add) whose
/// strides are nonzero on every axis the reduce operand's are.
pub(super) fn is_post_reduce_epilogue(computed: &BoundOp, reduce_nodes: &[bool]) -> bool {
    let BoundOpKind::Elementwise { operands, .. } = &computed.kind else {
        return false;
    };
    let mut reduce_operand_count = 0usize;
    for (node, layout, gather) in operands {
        let is_broadcast = layout.strides.contains(&0);
        if is_broadcast {
            continue;
        }
        let is_reduce_output = reduce_nodes.get(node.0 as usize).copied().unwrap_or(false);
        if is_reduce_output && gather.is_none() {
            reduce_operand_count += 1;
        } else {
            return false;
        }
    }
    reduce_operand_count == 1
}

/// `docs/discipline.md` ROW 190's own widened admission: [`is_post_reduce_epilogue`]
/// requires the reduce operand to be the element-for-element walked one (a
/// non-broadcast layout) — structurally true for the mnist conv/matmul-bias
/// shape, structurally FALSE for a `LayerNormalization` tail, whose
/// mean/variance reduce is read through a keepdims BROADCAST instead (see
/// [`EpilogueKind::LayerNorm`]'s own doc). This is the consumer-side-only
/// widening the map called for: same admission QUESTION ("is this node a
/// sole consumer of exactly one reduce, plus broadcasts") answered by a
/// second, disjoint shape test rather than by loosening the first — a
/// candidate this function accepts can never ALSO satisfy
/// [`is_post_reduce_epilogue`] (that function demands the reduce be
/// non-broadcast; this one demands it be broadcast), so trying both in
/// sequence never double-admits one candidate under two kinds.
///
/// Returns `(primary_slot, reduce_slot)`: `primary_slot` is the sole
/// non-broadcast operand (the element-for-element walked one — `centered`,
/// never itself a reduce); `reduce_slot` is the sole BROADCAST operand whose
/// node is a `Keep::Reduce` output. Every other operand is either a
/// non-reduce broadcast (`gamma`/`beta`/scalars) or absent.
pub(super) fn is_post_reduce_epilogue_broadcast_reduce(
    computed: &BoundOp,
    reduce_nodes: &[bool],
) -> Option<(usize, usize)> {
    let BoundOpKind::Elementwise { operands, .. } = &computed.kind else {
        return None;
    };
    let mut primary_slot: Option<usize> = None;
    let mut reduce_slot: Option<usize> = None;
    for (slot, (node, layout, gather)) in operands.iter().enumerate() {
        let is_broadcast = layout.strides.contains(&0);
        let is_reduce_output =
            gather.is_none() && reduce_nodes.get(node.0 as usize).copied().unwrap_or(false);
        if is_broadcast {
            if is_reduce_output {
                if reduce_slot.is_some() {
                    return None;
                }
                reduce_slot = Some(slot);
            }
        } else {
            if is_reduce_output || primary_slot.is_some() {
                return None;
            }
            primary_slot = Some(slot);
        }
    }
    primary_slot.zip(reduce_slot)
}

/// Is `layout` a `LayerNorm` reduce operand's own keepdims-broadcast
/// addressing of `extents` — zero stride on the LAST axis (the reduced,
/// hidden/feature axis, broadcast back over it) and standard row-major
/// addressing over every OTHER axis, matching the reduce's own smaller
/// physical buffer (kept axes only, contiguous). Sibling to
/// [`epilogue_is_contiguous_row_major`], which validates the DIFFERENT
/// shape the other three [`EpilogueKind`]s need (full-rank contiguous, no
/// dropped axis).
pub(super) fn epilogue_reduce_operand_matches_leading_axes(
    layout: &bind::Layout,
    extents: &[u64],
) -> bool {
    if layout.base != 0 || layout.strides.len() != extents.len() {
        return false;
    }
    let Some(leading_extents) = extents.len().checked_sub(1).map(|last| &extents[..last]) else {
        return false;
    };
    if layout.strides.last().copied().unwrap_or(0) != 0 {
        return false;
    }
    let mut expected = 1i64;
    for (axis, &extent) in leading_extents.iter().enumerate().rev() {
        if layout.strides.get(axis).copied().unwrap_or(0) != expected {
            return false;
        }
        expected *= extent as i64;
    }
    true
}

/// Is `layout` a `LayerNorm` `gamma`/`beta` operand's own broadcast: zero
/// stride on every axis except the LAST (contiguous, stride 1 — a plain
/// `[hidden]`-shaped tensor broadcast over every leading axis).
pub(super) fn epilogue_broadcast_operand_matches_last_axis(layout: &bind::Layout) -> bool {
    let Some((&last_stride, leading_strides)) = layout.strides.split_last() else {
        return false;
    };
    layout.base == 0 && last_stride == 1 && leading_strides.iter().all(|&stride| stride == 0)
}

/// Is `layout` a pure rank-0 scalar broadcast — every stride zero. Used for
/// `LayerNorm`'s `1/N` and `eps` operands, both lower-time constants.
pub(super) fn epilogue_is_scalar_broadcast(layout: &bind::Layout) -> bool {
    layout.strides.iter().all(|&stride| stride == 0)
}

/// `docs/discipline.md` ROW 184 Phase 3: the three exact op-sequence +
/// operand-wiring shapes a real diagnostic dump (`epilogue_fuse_plan`'s own
/// insertion site, temporarily instrumented, reverted before commit) found
/// among the real mnist model's 5 [`is_post_reduce_epilogue`]-matched
/// consumers -- `NodeId(25)`/`(32)`/`(55)` (bias-add then relu-clip,
/// 2-step), `NodeId(46)` (bias-add + relu-clip + batchnorm, 8-step),
/// `NodeId(69)` (bias-add + batchnorm, no relu, 7-step). A structural
/// match, never a `NodeId`/name lookup -- the same discipline
/// `detect_adam_update_roles` (this module) already established for a
/// differently-shaped fused chain. Anything NOT matching one of these three
/// exact shapes returns `None`, which [`epilogue_fuse_plan`] treats as "do
/// not fuse this node at all" -- the UNFUSED two-pass path runs instead,
/// never ROW 183's own interpreted `apply_body`-per-element evaluator
/// (measured +17.4% e2e slower).
/// `docs/discipline.md` ROW 190: `LayerNorm` is a fourth, structurally
/// distinct shape from `Clip`/`ClipNorm`/`Norm` — a real BERT-style ONNX
/// export's `LayerNormalization` unrolling, confirmed against BGE-small's
/// own 25 LayerNorm sites via `epilogue_dump_shapes`-style instrumentation
/// (run and reverted, never landed):
/// `((centered) / sqrt(reduce * (1/N) + eps)) * gamma + beta`. The other
/// three kinds all walk the REDUCE operand element-for-element (it IS the
/// per-position accumulator, bias-added at every output position); here the
/// reduce (a `ReduceMean`-derived sum-of-squared-centered-values) is instead
/// consumed through a KEEPDIMS broadcast — one value per row, re-read for
/// every column in that row — while the operand walked element-for-element
/// is `centered` (`x - mean`), itself an ordinary prior elementwise node,
/// never a reduce. `gamma`/`beta` compound this: they vary over the LAST
/// (hidden/feature) axis, the opposite axis from the one the reduce
/// broadcasts over — the one shape this kind's own kernel arm hard-codes,
/// verified structurally at admission (`epilogue_reduce_operand_matches_leading_axes`
/// / `epilogue_broadcast_operand_matches_last_axis` below), never assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EpilogueKind {
    /// `max(reduce + bias, zero)`.
    Clip,
    /// `((max(reduce + bias, zero) - mean) / sqrt(var + eps)) * gamma + beta`.
    ClipNorm,
    /// `((reduce + bias - mean) / sqrt(var + eps)) * gamma + beta`, no relu.
    Norm,
    /// `(centered / sqrt(reduce * (1/N) + eps)) * gamma + beta` — BERT-style
    /// `LayerNormalization`'s own unrolled tail; see this variant's own doc.
    LayerNorm,
}

/// `arg`, resolved one edge: `Some((op, args))` when `arg` is a
/// [`StepArg::Step`] pointing at a real entry in `body.steps`, `None` for a
/// [`StepArg::Operand`] (a leaf — nothing upstream to walk) or a dangling
/// index. The one primitive [`match_epilogue`]'s per-kind matchers below
/// build on: walking `StepArg::Step` edges rather than reading `steps[i]` by
/// position is what makes recognition invariant to how many steps upstream
/// canonicalization folded away (`push_canonical_step`, `bind.rs`) — the
/// defect `cpu.rs:2570-2626`'s "hidden=1 confluence gap" doc names.
pub(super) fn edge(body: &ComposedBody, arg: StepArg) -> Option<(ScalarOp, &[StepArg])> {
    match arg {
        StepArg::Step(index) => {
            let step = body.steps.get(index as usize)?;
            Some((step.op, step.args.as_slice()))
        }
        StepArg::Operand(_) => None,
    }
}

/// `edge`, specialized to a binary op — `Clip`/`ClipNorm`/`Norm`/`LayerNorm`
/// are all built entirely from unary (`SquareRoot`) and binary
/// (`Add`/`Subtract`/`Multiply`/`Divide`/`Maximum`) scalar steps, never a
/// three-arg `Select`.
pub(super) fn binary_edge(
    body: &ComposedBody,
    arg: StepArg,
) -> Option<(ScalarOp, StepArg, StepArg)> {
    let (op, args) = edge(body, arg)?;
    let [left, right] = args else { return None };
    Some((op, *left, *right))
}

pub(super) fn unary_edge(body: &ComposedBody, arg: StepArg) -> Option<(ScalarOp, StepArg)> {
    let (op, args) = edge(body, arg)?;
    let [only] = args else { return None };
    Some((op, *only))
}

/// `arg` is `expected_op(Operand(left), Operand(right))` — OR, when
/// `push_canonical_step` eliminated that step because one side was
/// `expected_op`'s own identity element, `arg` is directly whichever operand
/// survived the fold. Both are the same algebraic value; a matcher that only
/// accepted the first shape is exactly the positional brittleness this
/// module replaces. Used for the `reduce + bias` step every kind but
/// `LayerNorm` opens with, and reused by `LayerNorm`'s `reduce * (1/N)` step
/// — the concrete case `docs/discipline.md`'s hidden=1 confluence note
/// names, generalized here to any identity-eliminable binary step.
pub(super) fn matches_binary_or_eliminated(
    body: &ComposedBody,
    arg: StepArg,
    expected_op: ScalarOp,
    left_operand: u16,
    right_operand: u16,
) -> bool {
    match binary_edge(body, arg) {
        Some((op, StepArg::Operand(left), StepArg::Operand(right))) => {
            op == expected_op && left == left_operand && right == right_operand
        }
        Some(_) => false,
        None => arg == StepArg::Operand(left_operand) || arg == StepArg::Operand(right_operand),
    }
}

/// `Maximum(reduce + bias, zero)` — the `max(reduce + bias, 0)` shape every
/// `Clip`-carrying kind opens with, structural over which arg feeds
/// `Maximum` AND over which operand slot each of `bias`/`zero` landed at.
///
/// `push_canonical_step`'s own commutative canonicalization
/// (`bind.rs:1882-1932`) sorts a 2-operand `Add`/`Maximum` pair by
/// `step_arg_sort_key`, and `compose_body`'s leaf presort
/// (`bind.rs:1959-1961`) sorts the SOURCE `(NodeId, IndexMap)` pairs by
/// `NodeId` before that -- together these guarantee canonical STEP shape
/// but never promise `bias` lands at operand slot 1 or `zero` at slot 2 (the
/// literal indices this function required before ROW NNN). The only anchor
/// this function can trust is `reduce_slot`, independently established by
/// [`is_post_reduce_epilogue`] before `match_epilogue` is ever called --
/// every other operand's SLOT is discovered here, never assumed. Returns
/// `(bias_slot, zero_slot)`.
pub(super) fn matches_clip_head(
    body: &ComposedBody,
    arg: StepArg,
    reduce_slot: usize,
) -> Option<(usize, usize)> {
    let (op, left, right) = binary_edge(body, arg)?;
    if op != ScalarOp::Maximum {
        return None;
    }
    let (head, zero_slot) = split_step_operand(left, right)?;
    let bias_slot = resolve_other_operand(body, head, ScalarOp::Add, reduce_slot)?;
    Some((bias_slot, zero_slot))
}

/// Splits a canonical binary edge's two args into "the recursed side" (a
/// [`StepArg::Step`]) and "the leaf operand slot" (a [`StepArg::Operand`]),
/// regardless of which side `push_canonical_step`'s sort put first --
/// tries both orders rather than assuming Step-before-Operand, so this
/// stays correct even if the canonicalization rule ever changes.
pub(super) fn split_step_operand(left: StepArg, right: StepArg) -> Option<(StepArg, usize)> {
    match (left, right) {
        (step @ StepArg::Step(_), StepArg::Operand(operand)) => Some((step, operand as usize)),
        (StepArg::Operand(operand), step @ StepArg::Step(_)) => Some((step, operand as usize)),
        _ => None,
    }
}

/// `arg` is `expected_op(known, other)` in either operand order, OR (when
/// `push_canonical_step` eliminated the whole step because `known`'s
/// sibling was `expected_op`'s own identity element) `arg` IS `known`
/// directly -- the same "authored or eliminated" duality
/// [`matches_binary_or_eliminated`] documents, generalized to discover
/// `other`'s slot instead of checking it against a literal. Returns `None`
/// (never a guess) when the eliminated survivor is `known` itself: the
/// role's own operand no longer exists in the composed body, so a caller
/// depending on reading it (every current kernel arm does) cannot safely
/// fuse this shape -- structurally unreachable under
/// `NumericPolicy::bit_exact` today, since `identity_element_signed_zero_nan`
/// never fires there, but handled as a rejection rather than an assumption.
pub(super) fn resolve_other_operand(
    body: &ComposedBody,
    arg: StepArg,
    expected_op: ScalarOp,
    known_slot: usize,
) -> Option<usize> {
    let known = StepArg::Operand(u16::try_from(known_slot).ok()?);
    if arg == known {
        return None;
    }
    let (op, left, right) = binary_edge(body, arg)?;
    if op != expected_op {
        return None;
    }
    if left == known {
        operand_index(right)
    } else if right == known {
        operand_index(left)
    } else {
        None
    }
}

/// `arg` as a plain leaf operand slot, or `None` when it is a [`StepArg::Step`].
pub(super) fn operand_index(arg: StepArg) -> Option<usize> {
    match arg {
        StepArg::Operand(index) => Some(index as usize),
        StepArg::Step(_) => None,
    }
}

/// `((relu_result - mean) / sqrt(var + eps)) * gamma + beta` — the norm tail
/// shared by `ClipNorm`/`Norm`, parameterized over the operand slots each
/// carries the clipped/unclipped reduce at, since that is the only place the
/// two kinds' operand numbering diverges.
pub(super) fn matches_norm_tail(
    body: &ComposedBody,
    result: StepArg,
    mean_operand: u16,
    var_operand: u16,
    eps_operand: u16,
    gamma_operand: u16,
    beta_operand: u16,
) -> Option<StepArg> {
    let (ScalarOp::Add, scaled, StepArg::Operand(beta)) = binary_edge(body, result)? else {
        return None;
    };
    if beta != beta_operand {
        return None;
    }
    let (ScalarOp::Multiply, normalized, StepArg::Operand(gamma)) = binary_edge(body, scaled)?
    else {
        return None;
    };
    if gamma != gamma_operand {
        return None;
    }
    let (ScalarOp::Divide, centered, std) = binary_edge(body, normalized)? else {
        return None;
    };
    let (ScalarOp::SquareRoot, var_eps) = unary_edge(body, std)? else {
        return None;
    };
    if !matches_binary_or_eliminated(body, var_eps, ScalarOp::Add, var_operand, eps_operand) {
        return None;
    }
    let (ScalarOp::Subtract, reduce_head, StepArg::Operand(mean)) = binary_edge(body, centered)?
    else {
        return None;
    };
    if mean != mean_operand {
        return None;
    }
    Some(reduce_head)
}

/// Structural epilogue recognizer: walks backward from `body.steps.last()`
/// through [`StepArg::Step`] edges, matching each of the four kinds by which
/// step feeds which and what op it applies — never `steps.len() == N` or
/// `steps[i]`. Invariant to how many steps upstream canonicalization folded
/// (`push_canonical_step`, `bind.rs`) in the same bind call; same signature
/// as the `detect_epilogue_kind` this replaces, so `epilogue_fuse_plan`'s
/// call sites are unchanged.
/// Recognizes one of the four [`EpilogueKind`]s and, for `Clip`/`Norm` (the
/// two kinds this crate currently fuses through a landed law test),
/// discovers the `(bias_slot, zero_slot)` `Clip`'s own kernel arm needs
/// rather than assuming a literal position -- `zero_slot` is `None` for
/// `Norm` (no relu clamp). `ClipNorm`/`LayerNorm`'s tail operands
/// (`mean`/`variance`/`epsilon`/`gamma`/`beta`) still assume the
/// pre-canonicalization literal slots this row did not touch -- see ROW
/// NNN's residual note; neither kind has a landed rewrite-law test today.
/// Discovered non-reduce operand slots a kernel arm needs to read, keyed by
/// which [`EpilogueKind`] found them -- `ClipNorm` carries none today (ROW
/// NNN's residual: its kernel arm still reads the pre-canonicalization
/// literal slots `apply_epilogue_fused_monomorphic` always used).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EpilogueSlots {
    Clip {
        bias: usize,
        zero: usize,
    },
    LayerNorm {
        primary: usize,
        /// `None` when `reduce * (1/N)` folded away because `1/N == 1.0`
        /// exactly (`hidden == 1`) -- `eliminate_identity_multiply` always
        /// fires for this (bit-exact under every `NumericPolicy`, unlike
        /// the `Add`/`Maximum` identity folds), so the kernel substitutes
        /// the literal `1.0` rather than reading a slot that does not exist.
        reciprocal_n: Option<usize>,
        epsilon: usize,
        gamma: usize,
        beta: usize,
    },
    Other,
}

pub(super) fn match_epilogue(
    body: &ComposedBody,
    reduce_slot: usize,
    primary_slot: Option<usize>,
) -> Option<(EpilogueKind, EpilogueSlots)> {
    let last_index = u16::try_from(body.steps.len().checked_sub(1)?).ok()?;
    let result = StepArg::Step(last_index);

    if let Some((bias, zero)) = matches_clip_head(body, result, reduce_slot) {
        return Some((EpilogueKind::Clip, EpilogueSlots::Clip { bias, zero }));
    }

    if let Some(reduce_head) = matches_norm_tail(body, result, 3, 4, 5, 6, 7)
        && matches_clip_head(body, reduce_head, reduce_slot).is_some()
    {
        return Some((EpilogueKind::ClipNorm, EpilogueSlots::Other));
    }

    if let Some(reduce_head) = matches_norm_tail(body, result, 2, 3, 4, 5, 6)
        && resolve_other_operand(body, reduce_head, ScalarOp::Add, reduce_slot).is_some()
    {
        return Some((EpilogueKind::Norm, EpilogueSlots::Other));
    }

    // `LayerNorm`: `((primary) / sqrt(reduce * reciprocal_n + epsilon)) *
    // gamma + beta`. `primary_slot` is already discovered by
    // `is_post_reduce_epilogue_broadcast_reduce` before `match_epilogue` is
    // ever called (the raw `BoundOp` operand array, unaffected by
    // `compose_body`'s commutative reordering); every other slot here is
    // discovered against that anchor and against `reduce_slot`, never
    // assumed literal -- `compose_body`'s leaf presort places `gamma`,
    // `epsilon`, `reduce`/`reciprocal_n` at whatever slot their relative
    // `NodeId` earns them, whichever kind's chain they belong to.
    let primary_slot = primary_slot?;
    let (op, left, right) = binary_edge(body, result)?;
    if op != ScalarOp::Add {
        return None;
    }
    let (scaled, beta) = split_step_operand(left, right)?;

    let (op, left, right) = binary_edge(body, scaled)?;
    if op != ScalarOp::Multiply {
        return None;
    }
    let (normalized, gamma) = split_step_operand(left, right)?;

    let (op, left, right) = binary_edge(body, normalized)?;
    if op != ScalarOp::Divide {
        return None;
    }
    // `Divide` is never canonical-reordered (not in `ScalarOp::is_associative`),
    // so its authored (numerator, denominator) order survives intact.
    if left != StepArg::Operand(u16::try_from(primary_slot).ok()?) {
        return None;
    }
    let std = right;

    let (op, var_eps) = unary_edge(body, std)?;
    if op != ScalarOp::SquareRoot {
        return None;
    }

    let (op, left, right) = binary_edge(body, var_eps)?;
    if op != ScalarOp::Add {
        return None;
    }
    // `var_arg` (the `reduce * reciprocal_n` sub-expression) is a `Step` in
    // the ordinary case, but a bare `Operand(reduce_slot)` when `hidden==1`
    // makes `reciprocal_n == 1.0` exactly and `push_canonical_step`'s
    // ALWAYS-ON `Multiply`-by-one fold (`identity_element_bitexact`, unlike
    // the policy-gated `Add`/`Maximum` folds) eliminates the multiply --
    // `split_step_operand` cannot disambiguate that case (both sides are
    // then plain `Operand`s), so the reduce anchor is checked FIRST.
    let reduce_operand = StepArg::Operand(u16::try_from(reduce_slot).ok()?);
    let (var_arg, epsilon) = if left == reduce_operand || matches!(left, StepArg::Step(_)) {
        (left, operand_index(right)?)
    } else if right == reduce_operand || matches!(right, StepArg::Step(_)) {
        (right, operand_index(left)?)
    } else {
        return None;
    };

    let reciprocal_n = if var_arg == reduce_operand {
        None
    } else {
        Some(resolve_other_operand(
            body,
            var_arg,
            ScalarOp::Multiply,
            reduce_slot,
        )?)
    };
    Some((
        EpilogueKind::LayerNorm,
        EpilogueSlots::LayerNorm {
            primary: primary_slot,
            reciprocal_n,
            epsilon,
            gamma,
            beta,
        },
    ))
}

/// Is `layout` the reduce operand's own contiguous row-major addressing of
/// `extents` -- `strides[axis] == extents[axis+1..].product()`, the standard
/// row-major stride formula -- confirmed structurally before the monomorphized
/// kernel is allowed to walk the reduce's own buffer as a flat,
/// monotonically-incrementing slice instead of re-deriving
/// [`bind::Layout::offset_of`] every element. `false` sends the candidate
/// back to the unfused path (never the interpreted one).
pub(super) fn epilogue_is_contiguous_row_major(layout: &bind::Layout, extents: &[u64]) -> bool {
    if layout.base != 0 || layout.strides.len() != extents.len() {
        return false;
    }
    let mut expected = 1i64;
    for (axis, &extent) in extents.iter().enumerate().rev() {
        if layout.strides.get(axis).copied().unwrap_or(0) != expected {
            return false;
        }
        expected *= extent as i64;
    }
    true
}

/// The single axis every non-reduce operand's own broadcast varies over, or
/// `None` when every one is a true rank-0 scalar -- [`is_post_reduce_epilogue`]
/// already established every non-reduce operand carries at least one zero
/// stride; this asks the stronger, kernel-specific question the monomorphized
/// loop below needs: is there EXACTLY one such varying axis, shared by every
/// operand that varies at all, so hoisting can read each broadcast operand
/// once per outer-loop column instead of once per element. `Err(())` means
/// two operands disagree on which axis they vary over -- an unsupported shape,
/// sent back to the unfused path.
pub(super) fn epilogue_hoist_axis(
    operands: &[(NodeId, bind::Layout, Option<bind::Lookup>)],
    reduce_slot: usize,
) -> Result<Option<usize>, ()> {
    let mut axis: Option<usize> = None;
    for (slot, (_, layout, _)) in operands.iter().enumerate() {
        if slot == reduce_slot {
            continue;
        }
        for (candidate, stride) in layout.strides.iter().enumerate() {
            if *stride != 0 {
                match axis {
                    None => axis = Some(candidate),
                    Some(existing) if existing != candidate => return Err(()),
                    Some(_) => {}
                }
            }
        }
    }
    Ok(axis)
}

/// `docs/discipline.md` ROW 183 Phase 2: which `Keep::Reduce` producers can
/// have their sole [`is_post_reduce_epilogue`]-matched consumer's own body
/// evaluated early, keyed by the reduce's own [`NodeId`] to `(consumer's
/// `resolved` index, the position at which every input the fused write
/// needs is finally available)`. Eligibility, each a real correctness
/// requirement, not a style choice:
/// - the reduce has EXACTLY one real (non-gather) consumer among every
///   `resolved` node's own operands — otherwise fusing would still leave a
///   second reader needing the reduce's own unfused value, so both the raw
///   store and the fused write are required regardless, buying nothing.
/// - the reduce is not itself a requested output — an un-fused caller still
///   needs to read its own buffer.
/// - the producer is a plain (`out_scatter: None`) `Keep::Reduce` fold, and
///   no operand the fused write reads is a `quantized_weights` entry — a
///   quantized weight node never occupies a `buffers` slot at all (it lives
///   in the caller's own `quantized_weights` map instead, per
///   `evaluate_quantized_with_scratch`'s own `QuantizedBlock` match), and a
///   scatter reduce's output is not a plain row-major walk of `output_axes`,
///   so [`apply_epilogue_fused_monomorphic`]'s hoisted-column addressing does
///   not apply to either shape.
/// - `fire_position`, the max over the producer's own `resolved` position
///   and every OTHER operand's own `resolved` position (a `NodeId` absent
///   from `resolved` — a `block_node`/`Op::Input` — is always ready, so it
///   contributes nothing to the max). **This is NOT simply the producer's
///   own position.** A first build of this row fired eagerly at the
///   producer's own position and required every broadcast operand's
///   `resolved` position to be strictly earlier — real mnist measured
///   ZERO fusion hits under that rule (`epilogue_fuse_totals() ==
///   (0, 0)`, an N==0 tripwire per the "evidence" section: a probe that
///   never fires is not evidence of anything), because a bias/scale
///   `Constant` is commonly lowered by ONNX one program position AFTER the
///   reduce it feeds, not before — the SAME reordering-vs-`NodeId` gap the
///   MoE FFN fixture's own NaN already caught for `NodeId` MAGNITUDE
///   (`docs/discipline.md`'s note above this one), reproduced here as a
///   second, distinct instance for `resolved` POSITION ORDER itself: `NodeId`
///   backward-reference order guarantees an operand is ready by its
///   CONSUMER's own original position, never by an arbitrary EARLIER
///   position this row chooses to fire at. Firing at `fire_position`
///   instead of the producer's own position is what makes both facts true
///   at once.
pub(super) fn epilogue_fuse_plan(
    resolved: &[BoundOp],
    node_count: usize,
    effective_outputs: &[NodeId],
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
) -> SingleHopEpiloguePlan {
    if !EPILOGUE_FUSE_ENABLED.load(EpilogueFuseOrdering::Relaxed) {
        return BTreeMap::new();
    }
    let reduce_nodes = epilogue_profile_reduce_flags(resolved, node_count);
    let mut consumer_counts: BTreeMap<NodeId, u32> = BTreeMap::new();
    let mut node_position: BTreeMap<NodeId, usize> = BTreeMap::new();
    for (index, computed) in resolved.iter().enumerate() {
        node_position.insert(computed.node, index);
        for (operand, _layout, gather) in computed.operands() {
            if gather.is_none() {
                *consumer_counts.entry(*operand).or_insert(0) += 1;
            }
        }
    }
    let mut plan = BTreeMap::new();
    for (index, computed) in resolved.iter().enumerate() {
        let BoundOpKind::Elementwise { operands, body } = &computed.kind else {
            continue;
        };
        // Two disjoint admission shapes (`docs/discipline.md` ROW 190's own
        // widening): [`is_post_reduce_epilogue`] admits Clip/ClipNorm/Norm,
        // where the reduce is the element-for-element walked operand;
        // [`is_post_reduce_epilogue_broadcast_reduce`] admits `LayerNorm`,
        // where the reduce is read through a keepdims broadcast instead and
        // a DIFFERENT, non-reduce operand (`centered`) is walked
        // element-for-element. A candidate can satisfy at most one — see
        // that function's own doc for why.
        let (reduce_slot, primary_slot) = if is_post_reduce_epilogue(computed, &reduce_nodes) {
            let Some(reduce_slot) = operands.iter().position(|(node, layout, gather)| {
                gather.is_none()
                    && !layout.strides.contains(&0)
                    && reduce_nodes.get(node.0 as usize).copied().unwrap_or(false)
            }) else {
                continue;
            };
            (reduce_slot, None)
        } else if let Some((primary_slot, reduce_slot)) =
            is_post_reduce_epilogue_broadcast_reduce(computed, &reduce_nodes)
        {
            (reduce_slot, Some(primary_slot))
        } else {
            continue;
        };
        let Some((kind, epilogue_slots)) = match_epilogue(body, reduce_slot, primary_slot) else {
            continue;
        };
        // the monomorphized kernels below encode a FIXED operand wiring per
        // `kind` (verified against the real diagnostic dump) — a candidate
        // that structurally matches admission but wires operands into
        // different slots is real, just not a shape this row builds a
        // kernel for, so it falls back to unfused rather than risk
        // misreading the wrong slot. `Clip`/`Norm`/`LayerNorm` verify this
        // dynamically inside `match_epilogue` itself (`reduce_slot`/
        // `primary_slot` are the anchors `matches_clip_head`/
        // `resolve_other_operand`/the `Divide` check matched against), so
        // this check is a tautology for them; `ClipNorm` still assumes the
        // pre-canonicalization literal slot (ROW NNN's residual).
        let expected_reduce_slot = match kind {
            EpilogueKind::Clip | EpilogueKind::Norm | EpilogueKind::LayerNorm => reduce_slot,
            EpilogueKind::ClipNorm => 0,
        };
        if reduce_slot != expected_reduce_slot {
            continue;
        }
        let reduce_node = operands[reduce_slot].0;
        if consumer_counts.get(&reduce_node).copied().unwrap_or(0) != 1 {
            continue;
        }
        if effective_outputs.contains(&reduce_node) {
            #[cfg(feature = "instrument")]
            debug!(
                node = reduce_node.0,
                kind = "epilogue_fuse",
                decision = "rejected_requested_output",
                consumers = consumer_counts.get(&reduce_node).copied().unwrap_or(0),
                "epilogue fuse admission rejected -- reduce node is a requested output"
            );
            continue;
        }
        let Some(&producer_position) = node_position.get(&reduce_node) else {
            continue;
        };
        let producer = &resolved[producer_position];
        // A producer `bind::bind`'s own `reduce-epilogue-fusion` already gave
        // a non-identity `epilogue_body` (e.g. a sigmoid/silu gate folded
        // onto the fold) must be declined here, not just admitted as "plain":
        // `run_resolved_nodes_in_arena` skips `run_node_into` entirely for
        // whatever this function marks as a fusion SOURCE and instead reads
        // its RAW pre-epilogue value inside `apply_epilogue_fused_monomorphic`
        // -- silently dropping `bind::bind`'s own epilogue with no error, the
        // exact double-fusion collision this crate's docs/discipline.md
        // reduce-epilogue-fusion ROW never anticipated a SECOND, independent
        // epilogue-fusion pass existing at all.
        let is_plain_reduce = matches!(
            &producer.kind,
            BoundOpKind::Reduce {
                keep: Keep::Reduce,
                out_scatter: None,
                epilogue_body,
                epilogue_operands,
                ..
            } if reduce_epilogue_is_identity(epilogue_body, epilogue_operands)
        );
        if !is_plain_reduce {
            continue;
        }
        if operands
            .iter()
            .any(|(node, ..)| *node != reduce_node && quantized_weights.contains_key(node))
        {
            continue;
        }
        let hoist_axis = if kind == EpilogueKind::LayerNorm {
            // `LayerNorm`'s own geometry (see [`EpilogueKind::LayerNorm`]'s
            // doc): the reduce broadcasts over the LAST axis (hidden), so the
            // (before, at, after) split hoists at the SECOND-TO-LAST axis —
            // `at` walks rows, `after` walks the hidden axis `gamma`/`beta`
            // vary over (handled by a dedicated inner-axis read in
            // [`apply_epilogue_fused_monomorphic`], never this function's
            // single-axis `epilogue_hoist_axis`, which cannot express two
            // disjoint broadcast axes at once).
            let Some(row_axis) = computed.extents.len().checked_sub(2) else {
                continue;
            };
            let EpilogueSlots::LayerNorm {
                primary: primary_slot_index,
                reciprocal_n: reciprocal_n_slot,
                epsilon: epsilon_slot,
                gamma: gamma_slot,
                beta: beta_slot,
            } = epilogue_slots
            else {
                continue;
            };
            let (Some(primary), Some(epsilon), Some(gamma), Some(beta)) = (
                operands.get(primary_slot_index),
                operands.get(epsilon_slot),
                operands.get(gamma_slot),
                operands.get(beta_slot),
            ) else {
                continue;
            };
            if !epilogue_reduce_operand_matches_leading_axes(
                &operands[reduce_slot].1,
                &computed.extents,
            ) {
                continue;
            }
            if !epilogue_is_contiguous_row_major(&primary.1, &computed.extents) {
                continue;
            }
            // `reciprocal_n_slot` is `None` at `hidden == 1` (the multiply
            // folded away, `EpilogueSlots::LayerNorm`'s own doc) -- nothing
            // to validate structurally in that case since the kernel
            // substitutes the literal `1.0` rather than reading a slot.
            let reciprocal_n_ok = reciprocal_n_slot.is_none_or(|slot| {
                operands
                    .get(slot)
                    .is_some_and(|(_, layout, _)| epilogue_is_scalar_broadcast(layout))
            });
            let scalar_slots_ok = reciprocal_n_ok && epilogue_is_scalar_broadcast(&epsilon.1);
            let affine_slots_ok = epilogue_broadcast_operand_matches_last_axis(&gamma.1)
                && epilogue_broadcast_operand_matches_last_axis(&beta.1);
            if !scalar_slots_ok || !affine_slots_ok {
                continue;
            }
            Some(row_axis)
        } else {
            if !epilogue_is_contiguous_row_major(&operands[reduce_slot].1, &computed.extents) {
                continue;
            }
            let Ok(hoist_axis) = epilogue_hoist_axis(operands, reduce_slot) else {
                continue;
            };
            hoist_axis
        };
        let fire_position = operands
            .iter()
            .filter(|(node, ..)| *node != reduce_node)
            .map(|(node, ..)| {
                node_position
                    .get(node)
                    .copied()
                    .unwrap_or(producer_position)
            })
            .chain(core::iter::once(producer_position))
            .max()
            .unwrap_or(producer_position);
        // defensive: a valid topological program always has every operand
        // ready strictly before its consumer's own original position, so
        // this should never actually trip -- kept as a real guard rather
        // than an assumption, per this row's own two already-caught
        // ordering surprises above.
        if fire_position >= index {
            continue;
        }
        #[cfg(feature = "instrument")]
        debug!(
            node = reduce_node.0,
            kind = "epilogue_fuse",
            decision = "fused",
            into = computed.node.0,
            consumers = consumer_counts.get(&reduce_node).copied().unwrap_or(0),
            "epilogue fuse admitted -- reduce node folded into consumer's epilogue"
        );
        plan.insert(
            reduce_node,
            (index, fire_position, kind, hoist_axis, epilogue_slots),
        );
    }
    plan
}

/// `docs/discipline.md` ROW 183's own re-provable hit counter — how many
/// times [`apply_epilogue_fused_monomorphic`] actually ran, over how many elements,
/// snapshot-and-reset per call so a caller running several forward passes
/// back to back gets one pass's own count, not a running total. Exists so a
/// re-prove command can assert N > 0 rather than trust that the plan built
/// above actually fired (principle "evidence": a gate that cannot report
/// its N is not a gate).
pub(super) static EPILOGUE_FUSE_HITS: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);
pub(super) static EPILOGUE_FUSE_ELEMENTS: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);
pub(super) static EPILOGUE_FUSE_NANOS: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);

/// `docs/discipline.md` ROW 186's own paired-bench escape: a process-wide
/// switch [`epilogue_fuse_plan`] consults once per `evaluate_quantized_with_scratch`
/// call (never per element -- this is a plan-build-time gate, not a hot-path
/// branch), defaulting to `true` now that the fusion this crate's default
/// build runs is `EpilogueKind`'s monomorphized kernel. A future paired
/// bench/test needing the pre-ROW-184 unfused two-pass arm for comparison
/// calls [`set_epilogue_fuse_enabled`] rather than rebuilding the crate under
/// a separate cargo feature -- the same in-process toggle
/// `real_mnist_accuracy.rs`/`epilogue_fuse_alloc.rs` already reset counters
/// through ([`epilogue_fuse_reset`]), extended to also gate the plan itself.
pub(super) static EPILOGUE_FUSE_ENABLED: EpilogueFuseAtomicBool = EpilogueFuseAtomicBool::new(true);

/// Bench/test-only escape valve (see [`EPILOGUE_FUSE_ENABLED`]'s own doc):
/// flips whether [`epilogue_fuse_plan`] fires at all, process-wide, for
/// every `evaluate_named`/`evaluate_quantized_with_scratch` call from this
/// point forward. Not part of this crate's taught public surface -- a
/// caller composing tensor programs never needs this; a paired bench
/// comparing the fused default against the unfused arm does.
#[doc(hidden)]
pub fn set_epilogue_fuse_enabled(enabled: bool) {
    EPILOGUE_FUSE_ENABLED.store(enabled, EpilogueFuseOrdering::Relaxed);
}

/// `docs/discipline.md` ROW 188's own paired-bench escape, same shape as
/// [`EPILOGUE_FUSE_ENABLED`]: a process-wide switch [`run_reduce`] consults
/// once per bound op (never per element) before routing a
/// [`neon_tile_plan`]-shaped GEMM to Accelerate's `cblas_sgemm` instead of
/// the explicit NEON 6x4 microkernel. Defaults to DISABLED, unlike
/// `EPILOGUE_FUSE_ENABLED` -- ROW 188's own gate run found this route
/// (default-on) intercepting `neon_tile_full_output`'s own NEON-targeted
/// tolerance tests (sizes 257/260: sgemm's different summation order pushed
/// RMS error past the naive-f32 comparison those tests assert), so the
/// route stays opt-in via this toggle (or the paired bench's own process)
/// until either those tests are re-scoped to the active route or the
/// e2e/accuracy gate earns the production default -- gate 1's own "default
/// off until the full stack wins" discipline, applied to a platform-cfg
/// route rather than a Cargo feature.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) static ACCELERATE_GEMM_ENABLED: EpilogueFuseAtomicBool =
    EpilogueFuseAtomicBool::new(false);
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) static ACCELERATE_GEMM_HITS: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) static ACCELERATE_GEMM_DECLINED: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);

/// Bench/test-only escape valve (see `ACCELERATE_GEMM_ENABLED`'s own doc).
#[doc(hidden)]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn set_accelerate_gemm_enabled(enabled: bool) {
    ACCELERATE_GEMM_ENABLED.store(enabled, EpilogueFuseOrdering::Relaxed);
}

/// Runtime evidence the Accelerate route actually fired, not just compiled:
/// `(hits, declined)` where `declined` counts a `neon_tile_plan` gate pass
/// that fell through to NEON anyway (non-contiguous output row or a
/// non-zero reduce seed). Snapshot-only; a re-prove command resets via
/// process restart, same as `neon_tile_counters` (behind `instrument`).
#[must_use]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn accelerate_gemm_totals() -> (u64, u64) {
    (
        ACCELERATE_GEMM_HITS.load(EpilogueFuseOrdering::Relaxed),
        ACCELERATE_GEMM_DECLINED.load(EpilogueFuseOrdering::Relaxed),
    )
}

#[must_use]
pub fn epilogue_fuse_totals() -> (u64, u64, u64) {
    (
        EPILOGUE_FUSE_HITS.load(EpilogueFuseOrdering::Relaxed),
        EPILOGUE_FUSE_ELEMENTS.load(EpilogueFuseOrdering::Relaxed),
        EPILOGUE_FUSE_NANOS.load(EpilogueFuseOrdering::Relaxed),
    )
}

pub fn epilogue_fuse_reset() {
    EPILOGUE_FUSE_HITS.store(0, EpilogueFuseOrdering::Relaxed);
    EPILOGUE_FUSE_ELEMENTS.store(0, EpilogueFuseOrdering::Relaxed);
    EPILOGUE_FUSE_NANOS.store(0, EpilogueFuseOrdering::Relaxed);
}
