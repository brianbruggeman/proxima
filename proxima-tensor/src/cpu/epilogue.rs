use super::*;

/// `docs/discipline.md` ROW 184 Phase 3: `epilogue_fuse_plan`'s own
/// [`EpilogueKind`] classification, walked by a monomorphized loop instead of
/// ROW 183's per-element [`apply_body`] interpreter (measured 32.90 ns/element,
/// +17.4% e2e SLOWER than the unfused two-pass path it was meant to replace).
/// Every broadcast/scalar operand this body reads is hoisted to ONCE PER
/// OUTER-LOOP COLUMN (`hoist_axis`) rather than re-derived from
/// [`bind::Layout::offset_of`] on every element — including the invariant
/// `sqrt(var + eps)` sub-expression inside the two batchnorm shapes, which
/// depends on none of the per-element reduce value, so hoisting it out of the
/// inner loop is bit-identical to recomputing it at every position (the SAME
/// argument ROW 179's own rank-0 hoist already established, generalized here
/// from "invariant across the whole call" to "invariant across the inner
/// loop"). The per-element body itself is fixed Rust source per `kind` — a
/// `match` on a 3-variant enum, never a runtime step loop — so each op
/// executes in the SAME left-to-right, non-fused-multiply-add order
/// `apply_body` would have produced (`a * b + c` as two separate f32 rounding
/// steps, never `a.mul_add(b, c)`), which is what keeps this bit-identical to
/// the two-pass unfused path rather than merely close to it.
///
/// Zero heap allocation: `read` below closes over stack-resident state only,
/// `coordinate` is a fixed `[u64; MAX_INLINE_RANK]` array, and every hoisted
/// scalar is a plain `f32` local — closing ROW 183's own named residual (that
/// row's interpreter allocated 3 small `Vec`s per fusion hit).
/// The three fields [`epilogue_fuse_plan`] discovers together for one
/// admitted node -- bundled so [`apply_epilogue_fused_monomorphic`] stays
/// under clippy's argument-count gate rather than taking each separately.
#[derive(Debug, Clone, Copy)]
pub(super) struct EpilogueFuseKernel {
    pub(super) kind: EpilogueKind,
    pub(super) hoist_axis: Option<usize>,
    pub(super) slots: EpilogueSlots,
}

pub(super) fn apply_epilogue_fused_monomorphic<B: Deref<Target = [f32]>>(
    kernel: EpilogueFuseKernel,
    consumer: &BoundOp,
    reduce_node: NodeId,
    reduce_values: &[f32],
    buffers: &[Option<B>],
    output: &mut [f32],
) {
    let EpilogueFuseKernel {
        kind,
        hoist_axis,
        slots: epilogue_slots,
    } = kernel;
    let BoundOpKind::Elementwise { operands, .. } = &consumer.kind else {
        return;
    };
    let extents = &consumer.extents;
    let rank = extents.len();
    let axis = hoist_axis.unwrap_or(rank);
    let before: u64 = extents
        .get(..axis.min(rank))
        .map_or(1, |prefix| prefix.iter().product::<u64>().max(1));
    let at: u64 = if axis < rank { extents[axis] } else { 1 };
    let after: u64 = extents
        .get(axis.saturating_add(1)..)
        .map_or(1, |suffix| suffix.iter().product::<u64>().max(1));

    let read = |slot: usize, column: u64| -> f32 {
        let (node, layout, _gather) = &operands[slot];
        let mut coordinate = [0u64; bind::MAX_INLINE_RANK];
        if axis < rank {
            coordinate[axis] = column;
        }
        let offset = layout.offset_of(&coordinate[..rank.min(bind::MAX_INLINE_RANK)]);
        let source: &[f32] = if *node == reduce_node {
            reduce_values
        } else {
            buffers[node.0 as usize].as_deref().unwrap_or(&[])
        };
        usize::try_from(offset)
            .ok()
            .and_then(|index| source.get(index))
            .copied()
            .unwrap_or(0.0)
    };
    // `LayerNorm`'s own second broadcast axis (the LAST/hidden axis
    // `gamma`/`beta` vary over, disjoint from `axis` above): a dedicated
    // read keyed on the innermost loop position instead of `column`, since
    // `read`'s single `coordinate[axis]` cannot express two independently
    // varying axes at once (see [`EpilogueKind::LayerNorm`]'s own doc).
    let inner_axis = rank.saturating_sub(1);
    let read_inner = |slot: usize, inner: u64| -> f32 {
        let (node, layout, _gather) = &operands[slot];
        let mut coordinate = [0u64; bind::MAX_INLINE_RANK];
        if inner_axis < rank && inner_axis < bind::MAX_INLINE_RANK {
            coordinate[inner_axis] = inner;
        }
        let offset = layout.offset_of(&coordinate[..rank.min(bind::MAX_INLINE_RANK)]);
        let source: &[f32] = buffers[node.0 as usize].as_deref().unwrap_or(&[]);
        usize::try_from(offset)
            .ok()
            .and_then(|index| source.get(index))
            .copied()
            .unwrap_or(0.0)
    };

    let mut reduce_index = 0usize;
    let mut out_index = 0usize;
    for outer in 0..before {
        for column in 0..at {
            match kind {
                EpilogueKind::Clip => {
                    // `epilogue_fuse_plan` only ever inserts a `Clip` plan
                    // entry alongside the `EpilogueSlots::Clip` variant
                    // `match_epilogue` discovered for it (`cpu.rs`'s
                    // `plan.insert` call), so `(1, 2)` is a never-taken
                    // defensive fallback, not a real default.
                    let (bias_slot, zero_slot) = match epilogue_slots {
                        EpilogueSlots::Clip { bias, zero } => (bias, zero),
                        EpilogueSlots::LayerNorm { .. } | EpilogueSlots::Other => (1, 2),
                    };
                    let bias = read(bias_slot, column);
                    let zero = read(zero_slot, 0);
                    for _ in 0..after {
                        let value = reduce_values[reduce_index];
                        reduce_index += 1;
                        output[out_index] = (value + bias).max(zero);
                        out_index += 1;
                    }
                }
                EpilogueKind::ClipNorm => {
                    let bias = read(1, column);
                    let zero = read(2, 0);
                    let mean = read(3, column);
                    let variance = read(4, column);
                    let epsilon = read(5, 0);
                    let gamma = read(6, column);
                    let beta = read(7, column);
                    let denominator = (variance + epsilon).sqrt();
                    for _ in 0..after {
                        let value = reduce_values[reduce_index];
                        reduce_index += 1;
                        let biased = (value + bias).max(zero);
                        let centered = biased - mean;
                        let normalized = centered / denominator;
                        output[out_index] = normalized * gamma + beta;
                        out_index += 1;
                    }
                }
                EpilogueKind::Norm => {
                    let bias = read(1, column);
                    let mean = read(2, column);
                    let variance = read(3, column);
                    let epsilon = read(4, 0);
                    let gamma = read(5, column);
                    let beta = read(6, column);
                    let denominator = (variance + epsilon).sqrt();
                    for _ in 0..after {
                        let value = reduce_values[reduce_index];
                        reduce_index += 1;
                        let biased = value + bias;
                        let centered = biased - mean;
                        let normalized = centered / denominator;
                        output[out_index] = normalized * gamma + beta;
                        out_index += 1;
                    }
                }
                EpilogueKind::LayerNorm => {
                    // `epilogue_fuse_plan` only ever inserts a `LayerNorm`
                    // plan entry alongside the `EpilogueSlots::LayerNorm`
                    // `match_epilogue` discovered for it; the fallback here
                    // is a never-taken defensive default, matching `Clip`'s
                    // own `unwrap_or` shape above.
                    let (primary_slot, reciprocal_n_slot, epsilon_slot, gamma_slot, beta_slot) =
                        match epilogue_slots {
                            EpilogueSlots::LayerNorm {
                                primary,
                                reciprocal_n,
                                epsilon,
                                gamma,
                                beta,
                            } => (primary, reciprocal_n, epsilon, gamma, beta),
                            EpilogueSlots::Clip { .. } | EpilogueSlots::Other => {
                                (0, Some(2), 3, 4, 5)
                            }
                        };
                    // `reduce_values` here holds the variance-sum reduce's
                    // OWN materialized buffer -- shape `(before, at)`
                    // (hidden already dropped by the reduce), addressed
                    // directly rather than via `reduce_index`'s per-output-
                    // element counter (that counter walks `before*at*after`
                    // elements; this reduce needs exactly one read per
                    // `(outer, column)` row, reused across every `after`
                    // position in that row).
                    let primary_node = operands[primary_slot].0;
                    let primary = buffers[primary_node.0 as usize].as_deref().unwrap_or(&[]);
                    // `reciprocal_n_slot` is `None` only when `hidden == 1`
                    // folded `reduce * (1/N)` to the bare reduce operand
                    // (`1/N == 1.0` exactly) -- `EpilogueSlots::LayerNorm`'s
                    // own doc. The literal `1.0` is the correct substitute,
                    // not a guess: it is the exact value that eliminated.
                    let reciprocal_n = reciprocal_n_slot.map_or(1.0, |slot| read(slot, 0));
                    let epsilon = read(epsilon_slot, 0);
                    let row_index = (outer as usize)
                        .saturating_mul(at as usize)
                        .saturating_add(column as usize);
                    let sum_squared = reduce_values.get(row_index).copied().unwrap_or(0.0);
                    let variance = sum_squared * reciprocal_n;
                    let denominator = (variance + epsilon).sqrt();
                    for inner in 0..after {
                        let centered = primary.get(out_index).copied().unwrap_or(0.0);
                        let gamma = read_inner(gamma_slot, inner);
                        let beta = read_inner(beta_slot, inner);
                        let normalized = centered / denominator;
                        output[out_index] = normalized * gamma + beta;
                        out_index += 1;
                    }
                }
            }
        }
    }
}

/// `docs/discipline.md` ROW 204: one row of a `LayerNorm` cluster's own
/// mean/variance reduction, as a sans-IO FSM (phase + running accumulators)
/// rather than a flat function — mirrors `ConvReluStage`'s own ring+step
/// idiom (`proxima-onnx/benches/support/tile_pipeline.rs`), not the other
/// three [`EpilogueKind`] arms' per-element loop shape. [`advance`] is
/// driven to completion in one tight loop by
/// [`apply_layer_norm_cluster_fused`] below (no yielding/resumability wired
/// this row); the phase+accumulator shape is what a future band-streaming
/// driver would advance incrementally without a rewrite of this type.
/// Four independent accumulators per lane (`docs/discipline.md` ROW 200's
/// own lesson: a single-accumulator serial FMA chain is the landmine),
/// summed once per phase rather than once per element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum LayerNormRowPhase {
    Sum,
    Variance,
    Affine,
    Done,
}

pub(super) const LAYER_NORM_ROW_LANES: usize = 4;

#[derive(Debug, Clone, Copy)]
pub(super) struct LayerNormRowFsm {
    pub(super) phase: LayerNormRowPhase,
    pub(super) accumulators: [f32; LAYER_NORM_ROW_LANES],
    pub(super) mean: f32,
    pub(super) denominator: f32,
}

impl LayerNormRowFsm {
    fn new() -> Self {
        Self {
            phase: LayerNormRowPhase::Sum,
            accumulators: [0.0; LAYER_NORM_ROW_LANES],
            mean: 0.0,
            denominator: 0.0,
        }
    }

    /// One state transition. `row` is this row's own `hidden`-wide slice --
    /// `Sum`/`Variance` each walk it once; `Affine` does not re-walk it at
    /// all (the per-column affine write happens in the caller, after
    /// `finish`, since it also needs `gamma`/`beta`/`output` this FSM does
    /// not own). Returns `true` once the FSM reaches `Done`.
    fn advance(&mut self, row: &[f32], reciprocal_n: f32, epsilon: f32) -> bool {
        match self.phase {
            LayerNormRowPhase::Sum => {
                self.accumulators = [0.0; LAYER_NORM_ROW_LANES];
                for chunk in row.chunks(LAYER_NORM_ROW_LANES) {
                    for (lane, &value) in chunk.iter().enumerate() {
                        self.accumulators[lane] += value;
                    }
                }
                let sum: f32 = self.accumulators.iter().sum();
                self.mean = sum * reciprocal_n;
                self.phase = LayerNormRowPhase::Variance;
                false
            }
            LayerNormRowPhase::Variance => {
                self.accumulators = [0.0; LAYER_NORM_ROW_LANES];
                for chunk in row.chunks(LAYER_NORM_ROW_LANES) {
                    for (lane, &value) in chunk.iter().enumerate() {
                        let deviation = value - self.mean;
                        self.accumulators[lane] += deviation * deviation;
                    }
                }
                let sum_squared_deviation: f32 = self.accumulators.iter().sum();
                let variance = sum_squared_deviation * reciprocal_n;
                self.denominator = (variance + epsilon).sqrt();
                self.phase = LayerNormRowPhase::Affine;
                false
            }
            LayerNormRowPhase::Affine => {
                self.phase = LayerNormRowPhase::Done;
                true
            }
            LayerNormRowPhase::Done => true,
        }
    }

    /// `(mean, denominator)` — valid once `advance` has reached at least
    /// `Affine` (two real transitions run).
    fn finish(&self) -> (f32, f32) {
        (self.mean, self.denominator)
    }
}

/// `docs/discipline.md` ROW 204's own plan-build-time record: everything
/// [`apply_layer_norm_cluster_fused`] needs to compute a `LayerNorm` site's
/// mean/variance directly from `x`, bypassing the four upstream dispatches
/// (`R1` sum, `E1` mean, `E2` centered, `R2` sum-of-squares) [`epilogue_fuse_plan`]'s
/// single-hop `LayerNorm` admission (ROW 190/191) already fuses only the
/// LAST of. Keyed by the same `R2` `NodeId` the single-hop plan uses, so a
/// caller upgrading one to the other never double-keys.
pub(super) struct LayerNormClusterPlan {
    pub(super) tail_index: usize,
    /// `[R1, E2, R2]` — every node this cluster's own fusion makes dead
    /// weight, added to the executor's skip set alongside the tail itself
    /// (which the single-hop plan already skips). No standalone `E1` node
    /// exists in `resolved` at all -- `bind`'s own elementwise fusion
    /// already absorbed `mean = R1 * (1/N)` into `E2`'s own composed body
    /// (see `layer_norm_cluster_plan`'s own doc on `E2`'s admission).
    pub(super) skip: [NodeId; 3],
    pub(super) x_node: NodeId,
    pub(super) row_axis: usize,
    pub(super) fire_position: usize,
    /// The tail's own `(reciprocal_n, epsilon, gamma, beta)` operand slots,
    /// discovered by `match_epilogue`'s `EpilogueSlots::LayerNorm` rather
    /// than assumed literal -- `compose_body`'s commutative canonicalization
    /// does not guarantee these land at any fixed position (ROW NNN).
    pub(super) tail_reciprocal_n_slot: usize,
    pub(super) tail_epsilon_slot: usize,
    pub(super) tail_gamma_slot: usize,
    pub(super) tail_beta_slot: usize,
}

/// `docs/discipline.md` ROW 204: widens ROW 190/191's single-hop `LayerNorm`
/// epilogue fusion (reduce-of-squares -> affine tail only) to the FULL
/// five-dispatch cluster the map found `bind` collapses a BERT-style
/// `LayerNormalization` into. Walks EVERY `EpilogueKind::LayerNorm`
/// admission `single_hop` already found back UP `resolved`'s own `NodeId`
/// back-references (execution-level only, exactly ROW 190/191's own
/// constraint — never touches `bind::bind`) to confirm the exact
/// `R1(sum x) -> E1(mean=R1/N) -> E2(centered=x-mean) -> R2(sum
/// centered^2) -> tail` wiring a real BERT export produces. Anything not
/// matching this EXACT shape (wrong op, wrong operand count, a shared
/// operand read by anything other than the next node in the chain) is left
/// untouched — `single_hop`'s own tail-only fusion still applies to it.
pub(super) fn layer_norm_cluster_plan(
    resolved: &[BoundOp],
    effective_outputs: &[NodeId],
    single_hop: &SingleHopEpiloguePlan,
) -> BTreeMap<NodeId, LayerNormClusterPlan> {
    if single_hop.is_empty() {
        return BTreeMap::new();
    }
    let mut node_position: BTreeMap<NodeId, usize> = BTreeMap::new();
    let mut consumer_counts: BTreeMap<NodeId, u32> = BTreeMap::new();
    for (index, computed) in resolved.iter().enumerate() {
        node_position.insert(computed.node, index);
        for (operand, _layout, gather) in computed.operands() {
            if gather.is_none() {
                *consumer_counts.entry(*operand).or_insert(0) += 1;
            }
        }
    }
    let ready_position = |node: NodeId| node_position.get(&node).copied().unwrap_or(0);
    let sole_consumer = |node: NodeId| consumer_counts.get(&node).copied().unwrap_or(0) == 1;

    let mut clusters = BTreeMap::new();
    for (&r2_node, &(tail_index, _fire_position, kind, hoist_axis, epilogue_slots)) in single_hop {
        if kind != EpilogueKind::LayerNorm {
            continue;
        }
        let Some(row_axis) = hoist_axis else { continue };
        if effective_outputs.contains(&r2_node) {
            #[cfg(feature = "instrument")]
            debug!(
                node = r2_node.0,
                kind = "layer_norm_cluster",
                decision = "rejected_requested_output",
                consumers = consumer_counts.get(&r2_node).copied().unwrap_or(0),
                "layer norm cluster admission rejected -- r2 node is a requested output"
            );
            continue;
        }
        let tail = &resolved[tail_index];
        let BoundOpKind::Elementwise {
            operands: tail_operands,
            ..
        } = &tail.kind
        else {
            continue;
        };
        // `e2` (`centered = x - mean`) is the tail's own PRIMARY operand --
        // `epilogue_fuse_plan`'s `match_epilogue` already discovered its
        // slot dynamically (`EpilogueSlots::LayerNorm::primary`), since
        // `compose_body`'s commutative canonicalization does not guarantee
        // it lands at slot 0 (`tail_operands.first()`'s own prior
        // assumption -- ROW NNN's fix).
        let EpilogueSlots::LayerNorm {
            primary,
            reciprocal_n: tail_reciprocal_n_slot,
            epsilon: tail_epsilon_slot,
            gamma: tail_gamma_slot,
            beta: tail_beta_slot,
        } = epilogue_slots
        else {
            continue;
        };
        let Some((e2_node, ..)) = tail_operands.get(primary) else {
            continue;
        };
        let e2_node = *e2_node;

        // R2: sum of squared E2 -- one operand (E2), element_body squares it.
        let Some(&r2_index) = node_position.get(&r2_node) else {
            continue;
        };
        let r2 = &resolved[r2_index];
        let BoundOpKind::Reduce {
            element_body: r2_body,
            reduce_op: r2_op,
            init: r2_init,
            operands: r2_operands,
            ..
        } = &r2.kind
        else {
            continue;
        };
        if *r2_op != ScalarOp::Add || *r2_init != ReduceInit::Zero {
            continue;
        }
        // Bind does not deduplicate a repeated operand: squaring `E2`
        // lowers to TWO operand slots, both physically `e2_node`, combined
        // by `Multiply(Operand(0), Operand(1))` -- confirmed against the
        // real BGE graph via a temporary `LN_CLUSTER_DIAG`-gated eprintln
        // (added and removed this session, never landed): every one of the
        // 25 sites showed `operands=[e2_node, e2_node]`,
        // `args=[Operand(0), Operand(1)]`, never the naive
        // single-slot-read-twice shape this check first assumed.
        let is_square_body = r2_body.steps.len() == 1
            && r2_body.steps[0].op == ScalarOp::Multiply
            && r2_body.steps[0].args == [StepArg::Operand(0), StepArg::Operand(1)];
        let square_operands_ok =
            r2_operands.len() == 2 && r2_operands[0].0 == e2_node && r2_operands[1].0 == e2_node;
        if !is_square_body || !square_operands_ok {
            continue;
        }
        // E2 must feed exactly R2 (which reads it TWICE, per operand slot
        // -- see the squaring note above) and the tail (once, its primary
        // slot, matched below) -- `consumer_counts` counts each OPERAND
        // SLOT reference, not each distinct consumer node, so the real
        // gate is a total of 3 (2 from R2's own doubled read + 1 from the
        // tail), not 2.
        if consumer_counts.get(&e2_node).copied().unwrap_or(0) != 3 {
            continue;
        }

        // E2: centered = x - mean, where `bind`'s own elementwise fusion
        // has ALREADY absorbed `E1` (`mean = R1 * (1/N)`) into E2's own
        // composed body -- there is no standalone `E1` node in `resolved`
        // at all (confirmed against the real BGE graph via a temporary
        // `LN_CLUSTER_DIAG`-gated eprintln, added and removed this
        // session, never landed: every site showed a 2-step body,
        // `Multiply(Operand(1), Operand(2)) -> Subtract(Operand(0),
        // Step(0))`, operands `[x, R1 (broadcast), 1/N (scalar)]`, not the
        // 1-step `Subtract(x, e1_node)` shape this check first assumed).
        let Some(&e2_index) = node_position.get(&e2_node) else {
            continue;
        };
        let e2 = &resolved[e2_index];
        let BoundOpKind::Elementwise {
            body: e2_body,
            operands: e2_operands,
        } = &e2.kind
        else {
            continue;
        };
        let is_centered_body = e2_body.steps.len() == 2
            && e2_body.steps[0].op == ScalarOp::Multiply
            && e2_body.steps[0].args == [StepArg::Operand(1), StepArg::Operand(2)]
            && e2_body.steps[1].op == ScalarOp::Subtract
            && e2_body.steps[1].args == [StepArg::Operand(0), StepArg::Step(0)];
        if !is_centered_body {
            continue;
        }
        let [(x_node, x_layout, x_gather), first, second] = e2_operands.as_slice() else {
            continue;
        };
        // `mean = R1 * reciprocal_n` is a `Multiply` -- `compose_body`'s own
        // commutative canonicalization (`bind.rs`'s leaf presort) sorts
        // this pair's SOURCE `(NodeId, IndexMap)` by `NodeId` before slot
        // assignment, and `reciprocal_n` (a `Constant` authored early in
        // program order) routinely carries a SMALLER `NodeId` than `R1` (a
        // `Reduce` whose own operand chain is authored after it) -- e.g.
        // this row's own `layer_norm_cluster_program` fixture assigns
        // `reciprocal_n=NodeId(3)`, `r1=NodeId(5)`, landing `reciprocal_n`
        // at operand slot 1 and `r1` at slot 2, the REVERSE of what a fixed
        // `[r1, reciprocal_n]` positional destructure assumed. Discovered
        // structurally here instead (`r1` is the leading-axes-broadcast
        // reduce operand, `reciprocal_n` is the true scalar), never
        // assumed by slot -- `continue` when neither or both operands are
        // scalar rather than guessing which is which.
        let (
            (r1_node, r1_layout_in_e2, r1_gather),
            (reciprocal_n_node, reciprocal_n_layout, reciprocal_n_gather),
        ) = match (
            epilogue_is_scalar_broadcast(&first.1),
            epilogue_is_scalar_broadcast(&second.1),
        ) {
            (false, true) => (first, second),
            (true, false) => (second, first),
            _ => continue,
        };
        if x_gather.is_some() || r1_gather.is_some() || reciprocal_n_gather.is_some() {
            continue;
        }
        if !epilogue_is_contiguous_row_major(x_layout, &e2.extents) {
            continue;
        }
        if !epilogue_reduce_operand_matches_leading_axes(r1_layout_in_e2, &e2.extents) {
            continue;
        }
        if !epilogue_is_scalar_broadcast(reciprocal_n_layout) {
            continue;
        }
        // A real BGE export folds `1/N` into TWO independent `Constant`
        // nodes -- one feeding `E2`'s own mean sub-expression, one feeding
        // the tail's own variance-scale step -- never the same `NodeId`
        // (confirmed against the real BGE graph via a temporary
        // `LN_CLUSTER_DIAG`-gated eprintln, added and removed this
        // session, never landed: every site showed two DISTINCT `NodeId`s,
        // e.g. `217` vs `223`). The kernel below reads `1/N` once, from
        // the tail's own operand slot, for both the mean and variance
        // passes, so this admission requires the two constants carry the
        // SAME VALUE (read directly from `BoundOpKind::Constant`, not
        // inferred from `NodeId` identity) rather than being the same node.
        let Some(&reciprocal_n_index) = node_position.get(reciprocal_n_node) else {
            continue;
        };
        // `None` means `hidden == 1` folded the tail's own `reduce *
        // reciprocal_n` to the bare reduce operand (`EpilogueSlots::LayerNorm`'s
        // own doc) -- the documented confluence gap between this law's
        // structural admission and law 3's identity elimination
        // (`rewrite_law_equivalence.rs`'s own `law2` doc), not a shape this
        // cluster fusion fires for.
        let Some(tail_reciprocal_n_slot) = tail_reciprocal_n_slot else {
            continue;
        };
        let Some(&tail_reciprocal_n_index) =
            node_position.get(&tail_operands[tail_reciprocal_n_slot].0)
        else {
            continue;
        };
        let (
            BoundOpKind::Constant {
                value: reciprocal_n_e2_value,
            },
            BoundOpKind::Constant {
                value: reciprocal_n_tail_value,
            },
        ) = (
            &resolved[reciprocal_n_index].kind,
            &resolved[tail_reciprocal_n_index].kind,
        )
        else {
            continue;
        };
        if (reciprocal_n_e2_value - reciprocal_n_tail_value).abs() > 1e-9 {
            continue;
        }
        if !sole_consumer(*r1_node) {
            continue;
        }

        // R1: sum of x, plain (Identity element body).
        let Some(&r1_index) = node_position.get(r1_node) else {
            continue;
        };
        let r1 = &resolved[r1_index];
        let BoundOpKind::Reduce {
            element_body: r1_body,
            reduce_op: r1_op,
            init: r1_init,
            operands: r1_operands,
            ..
        } = &r1.kind
        else {
            continue;
        };
        if *r1_op != ScalarOp::Add || *r1_init != ReduceInit::Zero {
            continue;
        }
        let is_identity_body = r1_body.steps.len() == 1
            && r1_body.steps[0].op == ScalarOp::Identity
            && r1_body.steps[0].args == [StepArg::Operand(0)];
        if !is_identity_body || r1_operands.len() != 1 {
            continue;
        }
        let (r1_x_node, r1_x_layout, r1_x_gather) = &r1_operands[0];
        if r1_x_gather.is_some() || *r1_x_node != *x_node {
            continue;
        }
        if !epilogue_is_contiguous_row_major(r1_x_layout, &r1.extents) {
            continue;
        }
        if r1.extents != r2.extents {
            continue;
        }

        // No quantized weight ever occupies a `buffers` slot -- see
        // `epilogue_fuse_plan`'s own identical guard.
        if r1.extents.is_empty() {
            continue;
        }

        let fire_position = [
            ready_position(*x_node),
            ready_position(*reciprocal_n_node),
            ready_position(tail_operands[tail_reciprocal_n_slot].0),
            ready_position(tail_operands[tail_epsilon_slot].0),
            ready_position(tail_operands[tail_gamma_slot].0),
            ready_position(tail_operands[tail_beta_slot].0),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        // A real BGE export lowers `gamma`/`beta` right next to the
        // `LayerNorm` site itself -- measured AFTER `E2` in program order
        // at every one of the 25 sites, so `fire_position` (the max above)
        // regularly lands past `E2`'s own original position. `x` is
        // otherwise scheduled to be freed there by `node_retirement`'s own
        // schedule (built BEFORE this plan, from the unmodified graph,
        // where `E2` is `x`'s real last consumer) -- the caller's own
        // `layer_norm_cluster_keepalive` set (built from this plan, see
        // its own doc at the call site) defers that ONE retirement event
        // to `fire_position` instead of dropping it, so `x` survives
        // regardless of which side of `E2` this fusion ends up firing on.
        if fire_position >= tail_index {
            continue;
        }

        #[cfg(feature = "instrument")]
        debug!(
            node = r2_node.0,
            kind = "layer_norm_cluster",
            decision = "fused",
            into = tail.node.0,
            consumers = consumer_counts.get(&r2_node).copied().unwrap_or(0),
            x_node = x_node.0,
            "layer norm cluster admitted -- r2/e2/r1 folded into the layer norm tail"
        );
        clusters.insert(
            r2_node,
            LayerNormClusterPlan {
                tail_index,
                skip: [*r1_node, e2_node, r2_node],
                x_node: *x_node,
                row_axis,
                fire_position,
                tail_reciprocal_n_slot,
                tail_epsilon_slot,
                tail_gamma_slot,
                tail_beta_slot,
            },
        );
    }
    clusters
}

pub(super) static LAYER_NORM_CLUSTER_HITS: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);
pub(super) static LAYER_NORM_CLUSTER_ELEMENTS: EpilogueFuseAtomicU64 =
    EpilogueFuseAtomicU64::new(0);
pub(super) static LAYER_NORM_CLUSTER_NANOS: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);

/// `docs/discipline.md` ROW 204's own re-provable hit counter, same shape as
/// [`epilogue_fuse_totals`] -- `(hits, elements, nanos)`, snapshot-and-reset
/// per call.
#[must_use]
pub fn layer_norm_cluster_totals() -> (u64, u64, u64) {
    (
        LAYER_NORM_CLUSTER_HITS.load(EpilogueFuseOrdering::Relaxed),
        LAYER_NORM_CLUSTER_ELEMENTS.load(EpilogueFuseOrdering::Relaxed),
        LAYER_NORM_CLUSTER_NANOS.load(EpilogueFuseOrdering::Relaxed),
    )
}

pub fn layer_norm_cluster_reset() {
    LAYER_NORM_CLUSTER_HITS.store(0, EpilogueFuseOrdering::Relaxed);
    LAYER_NORM_CLUSTER_ELEMENTS.store(0, EpilogueFuseOrdering::Relaxed);
    LAYER_NORM_CLUSTER_NANOS.store(0, EpilogueFuseOrdering::Relaxed);
}

/// One entry in [`run_rewrite_worklist`]'s applied-substitution log --
/// `docs/rewrite-algebra.md` §8's own "depth as an observable, not a hidden
/// implementation detail". Recorded once per admitted node, never per
/// element, so this is plan-build-time bookkeeping, not a hot-path cost.
pub(super) struct RewriteFire {
    pub(super) depth: usize,
    pub(super) law: &'static str,
    pub(super) node: NodeId,
}

/// `docs/rewrite-algebra.md` §8's worklist engine: explicit state (a
/// candidate set per depth, an applied-substitution log, a depth counter),
/// **no call recursion** -- two loop-free passes, each popping its own
/// worklist once. It orchestrates the two LANDED laws
/// ([`epilogue_fuse_plan`] = law 1/2 epilogue + row-statistic absorption,
/// [`layer_norm_cluster_plan`] = law 2's full five-dispatch cluster upgrade)
/// as law instances, never reimplementing either detector's own admission
/// logic.
///
/// **Depth 1** fires law 1/2 directly against the raw, bound graph -- the
/// shapes `epilogue_fuse_plan` already detects in one pass over all of
/// `resolved`, unaware of any prior substitution.
///
/// **Depth 2 re-enqueues ONLY the neighborhood depth 1 changed.**
/// [`layer_norm_cluster_plan`] already takes `single_hop` (depth 1's own
/// output map) as its sole per-node candidate set -- it never rescans
/// `resolved` for new candidates outside that set. That parameter IS the
/// worklist re-enqueue this engine performs; the debug assertion below
/// proves it holds rather than assuming it.
///
/// **Depth bound.** Per §7's own termination argument, node count is a
/// non-negative integer every law strictly decreases, so `resolved.len()`
/// is a hard ceiling on total applications, not a magic constant -- data
/// derived from the call's own input, asserted per depth, never consulted
/// to loop further (a correctly implemented law cannot exceed it; hitting
/// it is a bug in a law's own progress reporting, not a legitimately deep
/// fixpoint).
///
/// **Residual, named honestly**: laws 3/5/6 are not engine laws here. Law 3
/// (prologue absorption) runs inside `bind::bind` before this function ever
/// sees `resolved` -- §8's own "`bind.rs` is untouched" boundary, restated:
/// widening this engine to re-open what `bind.rs` already fused is exactly
/// the landmine ROW 166 (`docs/discipline.md:14974`) describes. Law 6∘5
/// (weight packing, [`build_packed_width_panels`]) runs even earlier, over
/// the pre-bind `Op` program at `evaluate_quantized_with_scratch`'s own
/// call site, gated behind [`PACK_AT_PLAN_TIME_ENABLED`] -- folding it into
/// this worklist is future work, not attempted in this landing. Law 4
/// (same-input widening) and law 2's softmax instantiation are PROPOSED,
/// not landed anywhere in this tree, so they have no detector to
/// orchestrate. **The hidden=1 confluence gap stands as documented**: law 3
/// runs at bind time, before this engine's own depth 1 candidate set is
/// even formed, so a `mean` node law 3 already folded via
/// `eliminate_identity_multiply` at `hidden=1` is invisible to depth 1's
/// admission test -- this engine does not close that gap and does not make
/// it worse, since it fires the exact same two detectors, in the exact same
/// order, over the exact same `resolved` list `epilogue_fuse_plan` always
/// received.
/// Law 1/2's own single-hop admission map: reduce node -> (consumer's
/// `resolved` index, fire position, kind, hoist axis). Named here only to
/// keep [`run_rewrite_worklist`]'s signature legible -- the shape itself is
/// [`epilogue_fuse_plan`]'s, unchanged.
/// `(consumer_index, fire_position, kind, hoist_axis, epilogue_slots)` --
/// `epilogue_slots` is [`match_epilogue`]'s own discovered operand-role
/// mapping, against the actual composed-body operand numbering rather than
/// a pre-canonicalization literal.
pub(super) type SingleHopEpiloguePlan =
    BTreeMap<NodeId, (usize, usize, EpilogueKind, Option<usize>, EpilogueSlots)>;

pub(super) fn run_rewrite_worklist(
    resolved: &[BoundOp],
    node_count: usize,
    effective_outputs: &[NodeId],
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
) -> (
    SingleHopEpiloguePlan,
    BTreeMap<NodeId, LayerNormClusterPlan>,
    Vec<RewriteFire>,
) {
    let depth_bound = resolved.len();
    let mut fires = Vec::new();

    let single_hop = epilogue_fuse_plan(resolved, node_count, effective_outputs, quantized_weights);
    debug_assert!(
        single_hop.len() <= depth_bound,
        "law 1/2 admitted more nodes than exist in the resolved graph -- termination argument violated"
    );
    fires.extend(single_hop.keys().map(|node| RewriteFire {
        depth: 1,
        law: "law1_2_epilogue_absorption",
        node: *node,
    }));

    let depth1_candidates: BTreeSet<NodeId> = single_hop.keys().copied().collect();
    let cluster = layer_norm_cluster_plan(resolved, effective_outputs, &single_hop);
    debug_assert!(
        cluster.keys().all(|node| depth1_candidates.contains(node)),
        "law 2's cluster upgrade fired on a node outside depth 1's own worklist -- re-enqueue scoping violated"
    );
    debug_assert!(
        cluster.len() <= depth_bound,
        "law 2 admitted more nodes than exist in the resolved graph -- termination argument violated"
    );
    fires.extend(cluster.keys().map(|node| RewriteFire {
        depth: 2,
        law: "law2_layer_norm_cluster_upgrade",
        node: *node,
    }));

    (single_hop, cluster, fires)
}

pub(super) static REWRITE_DEPTH1_FIRES: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);
pub(super) static REWRITE_DEPTH2_FIRES: EpilogueFuseAtomicU64 = EpilogueFuseAtomicU64::new(0);

pub(super) fn record_rewrite_engine_fires(fires: &[RewriteFire]) {
    let mut depth1 = 0u64;
    let mut depth2 = 0u64;
    let mut seen: BTreeSet<(usize, NodeId)> = BTreeSet::new();
    for fire in fires {
        debug_assert!(
            seen.insert((fire.depth, fire.node)),
            "a node fired twice at the same depth -- worklist re-enqueue bug"
        );
        match fire.depth {
            1 => {
                debug_assert_eq!(fire.law, "law1_2_epilogue_absorption");
                depth1 += 1;
            }
            2 => {
                debug_assert_eq!(fire.law, "law2_layer_norm_cluster_upgrade");
                depth2 += 1;
            }
            other => debug_assert!(false, "rewrite engine fired at unexpected depth {other}"),
        }
    }
    if depth1 > 0 {
        REWRITE_DEPTH1_FIRES.fetch_add(depth1, EpilogueFuseOrdering::Relaxed);
    }
    if depth2 > 0 {
        REWRITE_DEPTH2_FIRES.fetch_add(depth2, EpilogueFuseOrdering::Relaxed);
    }
}

/// `docs/rewrite-algebra.md` §8's own re-provable per-depth trace -- how
/// many nodes fired at depth 1 (raw-graph law 1/2 admission) vs depth 2
/// (law 2's re-enqueued cluster upgrade), snapshot-and-reset per call, same
/// shape as [`epilogue_fuse_totals`]/[`layer_norm_cluster_totals`]. Exists
/// so a re-prove command can assert per-depth `N > 0` on a real graph
/// rather than trust the worklist fired at the depths its own doc claims.
#[must_use]
pub fn rewrite_engine_depth_fires() -> (u64, u64) {
    (
        REWRITE_DEPTH1_FIRES.load(EpilogueFuseOrdering::Relaxed),
        REWRITE_DEPTH2_FIRES.load(EpilogueFuseOrdering::Relaxed),
    )
}

pub fn rewrite_engine_reset() {
    REWRITE_DEPTH1_FIRES.store(0, EpilogueFuseOrdering::Relaxed);
    REWRITE_DEPTH2_FIRES.store(0, EpilogueFuseOrdering::Relaxed);
}

/// One kernel call per `LayerNorm` site instead of `R1`/`E1`/`E2`/`R2`'s own
/// four dispatches plus the tail -- reads `x` directly, drives
/// [`LayerNormRowFsm`] to completion per row (mean, then the STABLE
/// two-pass variance -- sum of squared DEVIATIONS from the mean this row's
/// own FSM just computed, never the mean-of-squares identity `norm.rs`'s
/// own doc bans), then writes the normalize+affine step reusing
/// [`EpilogueKind::LayerNorm`]'s own arithmetic. Writes ONLY `tail`'s own
/// buffer -- `x` is read, never mutated.
/// The tail's own `(reciprocal_n, epsilon, gamma, beta)` operand slots
/// [`layer_norm_cluster_plan`] discovered -- bundled so
/// [`apply_layer_norm_cluster_fused`] stays under clippy's argument-count
/// gate rather than taking each separately.
#[derive(Debug, Clone, Copy)]
pub(super) struct LayerNormTailSlots {
    pub(super) reciprocal_n: usize,
    pub(super) epsilon: usize,
    pub(super) gamma: usize,
    pub(super) beta: usize,
}

pub(super) fn apply_layer_norm_cluster_fused<B: Deref<Target = [f32]>>(
    tail: &BoundOp,
    x_node: NodeId,
    row_axis: usize,
    tail_slots: LayerNormTailSlots,
    buffers: &[Option<B>],
    output: &mut [f32],
) {
    let BoundOpKind::Elementwise { operands, .. } = &tail.kind else {
        return;
    };
    let extents = &tail.extents;
    let rank = extents.len();
    let axis = row_axis.min(rank);
    let before: u64 = extents
        .get(..axis)
        .map_or(1, |prefix| prefix.iter().product::<u64>().max(1));
    let at: u64 = extents.get(axis).copied().unwrap_or(1);
    let hidden: u64 = extents
        .get(axis.saturating_add(1)..)
        .map_or(1, |suffix| suffix.iter().product::<u64>().max(1));

    let x = buffers[x_node.0 as usize].as_deref().unwrap_or(&[]);
    let inner_axis = rank.saturating_sub(1);
    let read_inner = |slot: usize, inner: u64| -> f32 {
        let (node, layout, _gather) = &operands[slot];
        let mut coordinate = [0u64; bind::MAX_INLINE_RANK];
        if inner_axis < rank && inner_axis < bind::MAX_INLINE_RANK {
            coordinate[inner_axis] = inner;
        }
        let offset = layout.offset_of(&coordinate[..rank.min(bind::MAX_INLINE_RANK)]);
        let source: &[f32] = buffers[node.0 as usize].as_deref().unwrap_or(&[]);
        usize::try_from(offset)
            .ok()
            .and_then(|index| source.get(index))
            .copied()
            .unwrap_or(0.0)
    };
    let read_scalar = |slot: usize| -> f32 { read_inner(slot, 0) };

    let reciprocal_n = read_scalar(tail_slots.reciprocal_n);
    let epsilon = read_scalar(tail_slots.epsilon);
    let hidden_usize = hidden as usize;
    let mut row_index = 0usize;
    let mut out_index = 0usize;
    for _ in 0..before {
        for _ in 0..at {
            let row_start = row_index * hidden_usize;
            let row = x.get(row_start..row_start + hidden_usize).unwrap_or(&[]);
            let mut fsm = LayerNormRowFsm::new();
            while !fsm.advance(row, reciprocal_n, epsilon) {}
            let (mean, denominator) = fsm.finish();
            for inner in 0..hidden {
                let value = row.get(inner as usize).copied().unwrap_or(0.0);
                let gamma = read_inner(tail_slots.gamma, inner);
                let beta = read_inner(tail_slots.beta, inner);
                let normalized = (value - mean) / denominator;
                output[out_index] = normalized * gamma + beta;
                out_index += 1;
            }
            row_index += 1;
        }
    }
}

/// ROW 181's three profile buckets: (a) every `Keep::Reduce` fold --
/// tile-routed and generic combined, since splitting those per-node would
/// duplicate `neon_tile_plan`'s own six-condition gate rather than reuse
/// it (the tile/generic split is corroborated separately via
/// `instrument::totals()`'s existing `path_*` counters in the same
/// profiling run); (b) [`is_post_reduce_epilogue`] matches; (c) everything
/// else (non-epilogue elementwise, iota, constant, scan).
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_REDUCE_NANOS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_REDUCE_CALLS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_EPILOGUE_NANOS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_EPILOGUE_CALLS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_OTHER_NANOS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_OTHER_CALLS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);
/// `docs/discipline.md` ROW 201/202: bucket (a) above merges the 96
/// GEMM-shaped `MatMul` folds with 74 small non-GEMM reduces (LayerNorm
/// mean/variance, final mean-pooling) that share `Keep::Reduce` but pay a
/// very different per-node cost. These two counters are ADDITIVE detail
/// recorded alongside (never instead of) `EPILOGUE_PROFILE_REDUCE_*` above
/// -- every reduce call increments both its bucket-(a) aggregate and
/// exactly one of these two, via the same [`reduce_is_gemm_shaped`]
/// classifier, so `EPILOGUE_PROFILE_REDUCE_*` stays byte-identical to what
/// it measured before this split existed.
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_REDUCE_GEMM_NANOS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_REDUCE_GEMM_CALLS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_REDUCE_SMALL_NANOS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);
#[cfg(feature = "epilogue-profile-probe")]
pub(super) static EPILOGUE_PROFILE_REDUCE_SMALL_CALLS: EpilogueProfileAtomicU64 =
    EpilogueProfileAtomicU64::new(0);

#[cfg(feature = "epilogue-profile-probe")]
pub(super) fn epilogue_profile_record(
    computed: &BoundOp,
    reduce_nodes: &[bool],
    elapsed_nanos: u64,
) {
    if matches!(
        computed.kind,
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            ..
        }
    ) {
        EPILOGUE_PROFILE_REDUCE_NANOS.fetch_add(elapsed_nanos, EpilogueProfileOrdering::Relaxed);
        EPILOGUE_PROFILE_REDUCE_CALLS.fetch_add(1, EpilogueProfileOrdering::Relaxed);
        if reduce_is_gemm_shaped(computed) {
            EPILOGUE_PROFILE_REDUCE_GEMM_NANOS
                .fetch_add(elapsed_nanos, EpilogueProfileOrdering::Relaxed);
            EPILOGUE_PROFILE_REDUCE_GEMM_CALLS.fetch_add(1, EpilogueProfileOrdering::Relaxed);
        } else {
            EPILOGUE_PROFILE_REDUCE_SMALL_NANOS
                .fetch_add(elapsed_nanos, EpilogueProfileOrdering::Relaxed);
            EPILOGUE_PROFILE_REDUCE_SMALL_CALLS.fetch_add(1, EpilogueProfileOrdering::Relaxed);
        }
    } else if is_post_reduce_epilogue(computed, reduce_nodes) {
        EPILOGUE_PROFILE_EPILOGUE_NANOS.fetch_add(elapsed_nanos, EpilogueProfileOrdering::Relaxed);
        EPILOGUE_PROFILE_EPILOGUE_CALLS.fetch_add(1, EpilogueProfileOrdering::Relaxed);
    } else {
        EPILOGUE_PROFILE_OTHER_NANOS.fetch_add(elapsed_nanos, EpilogueProfileOrdering::Relaxed);
        EPILOGUE_PROFILE_OTHER_CALLS.fetch_add(1, EpilogueProfileOrdering::Relaxed);
    }
}

/// Snapshot of ROW 181's three profile buckets:
/// `(reduce_nanos, reduce_calls, epilogue_nanos, epilogue_calls, other_nanos, other_calls)`.
#[cfg(feature = "epilogue-profile-probe")]
#[must_use]
pub fn epilogue_profile_totals() -> (u64, u64, u64, u64, u64, u64) {
    (
        EPILOGUE_PROFILE_REDUCE_NANOS.load(EpilogueProfileOrdering::Relaxed),
        EPILOGUE_PROFILE_REDUCE_CALLS.load(EpilogueProfileOrdering::Relaxed),
        EPILOGUE_PROFILE_EPILOGUE_NANOS.load(EpilogueProfileOrdering::Relaxed),
        EPILOGUE_PROFILE_EPILOGUE_CALLS.load(EpilogueProfileOrdering::Relaxed),
        EPILOGUE_PROFILE_OTHER_NANOS.load(EpilogueProfileOrdering::Relaxed),
        EPILOGUE_PROFILE_OTHER_CALLS.load(EpilogueProfileOrdering::Relaxed),
    )
}

/// ROW 202's own split of bucket (a): `(gemm_nanos, gemm_calls,
/// small_nanos, small_calls)`. `gemm_nanos + small_nanos ==
/// EPILOGUE_PROFILE_REDUCE_NANOS` and likewise for calls, by construction
/// in `epilogue_profile_record` above (both counters increment on the
/// same call, from the same classifier).
#[cfg(feature = "epilogue-profile-probe")]
#[must_use]
pub fn epilogue_profile_reduce_split_totals() -> (u64, u64, u64, u64) {
    (
        EPILOGUE_PROFILE_REDUCE_GEMM_NANOS.load(EpilogueProfileOrdering::Relaxed),
        EPILOGUE_PROFILE_REDUCE_GEMM_CALLS.load(EpilogueProfileOrdering::Relaxed),
        EPILOGUE_PROFILE_REDUCE_SMALL_NANOS.load(EpilogueProfileOrdering::Relaxed),
        EPILOGUE_PROFILE_REDUCE_SMALL_CALLS.load(EpilogueProfileOrdering::Relaxed),
    )
}

/// Resets ROW 181's three profile buckets to zero -- called between the
/// warm-up pass and the timed sweep so warm-up compilation/caching effects
/// never pollute the attributed breakdown. Also resets ROW 202's gemm/small
/// split counters, in lockstep -- both sides of the split must reset on the
/// same call or a later `epilogue_profile_reduce_split_totals()` read could
/// straddle two different measurement windows.
#[cfg(feature = "epilogue-profile-probe")]
pub fn epilogue_profile_reset() {
    EPILOGUE_PROFILE_REDUCE_NANOS.store(0, EpilogueProfileOrdering::Relaxed);
    EPILOGUE_PROFILE_REDUCE_CALLS.store(0, EpilogueProfileOrdering::Relaxed);
    EPILOGUE_PROFILE_EPILOGUE_NANOS.store(0, EpilogueProfileOrdering::Relaxed);
    EPILOGUE_PROFILE_EPILOGUE_CALLS.store(0, EpilogueProfileOrdering::Relaxed);
    EPILOGUE_PROFILE_OTHER_NANOS.store(0, EpilogueProfileOrdering::Relaxed);
    EPILOGUE_PROFILE_OTHER_CALLS.store(0, EpilogueProfileOrdering::Relaxed);
    EPILOGUE_PROFILE_REDUCE_GEMM_NANOS.store(0, EpilogueProfileOrdering::Relaxed);
    EPILOGUE_PROFILE_REDUCE_GEMM_CALLS.store(0, EpilogueProfileOrdering::Relaxed);
    EPILOGUE_PROFILE_REDUCE_SMALL_NANOS.store(0, EpilogueProfileOrdering::Relaxed);
    EPILOGUE_PROFILE_REDUCE_SMALL_CALLS.store(0, EpilogueProfileOrdering::Relaxed);
}

/// `docs/discipline.md` ROW 180's dynamic-elision probe: the SAME skip
/// check `run_resolved_nodes_in_arena` already runs (`dead`/`static_nodes`,
/// both fixed forever at [`build_static_arena`] time), unioned with a THIRD
/// set the caller derives fresh every call from a per-step mask -- a block
/// live one step and skipped the next, which `dead`/`static_nodes` cannot
/// express since both are computed once and never revisited. `named` is
/// expected to carry ONLY this step's live blocks (`bind_named_inputs_into_arena`'s
/// `require_all = false`, the same relaxation
/// [`evaluate_named_with_arena_in_place`] already uses) -- a masked-off
/// block's caller-side buffer is never even copied into the arena, not just
/// never computed, so the traffic this probe measures is the SAME traffic a
/// caller genuinely elides (no read of the skipped block's data at all).
/// Execution-level only: `resolved`/`shapes`/`effective_outputs` (every
/// `bind::bind` fusion decision) are untouched, exactly as ROW 167 found for
/// the fixed dead-set -- graph-level removal (ROW 166) is not this
/// mechanism and is not attempted here. A node this step's `skip` names that
/// is NOT also live in `arena.buffers` keeps its stale prior-step value,
/// same as `evaluate_named_with_arena_in_place`'s own rebind aliasing
/// already relies on for untouched nodes.
#[cfg(feature = "dynamic-elision-probe")]
pub fn evaluate_named_with_arena_masked(
    arena: &mut StaticArena,
    named: &[(&str, &[f32])],
    skip: &BTreeSet<NodeId>,
) -> Result<Evaluated, TensorError> {
    bind_named_inputs_into_arena(arena, named, false)?;
    for computed in &arena.resolved {
        if arena.dead.contains(&computed.node)
            || arena.static_nodes.contains(&computed.node)
            || skip.contains(&computed.node)
        {
            continue;
        }
        let node_index = computed.node.0 as usize;
        let mut output = arena.buffers[node_index].take().ok_or(TensorError::NotLowerable {
            node: computed.node,
            reason: "static arena has no pre-sized slot for this resolved node -- build_static_arena did not size it",
        })?;
        run_node_into(
            computed,
            &arena.buffers,
            None,
            None,
            None,
            false,
            &mut output,
        )?;
        arena.buffers[node_index] = Some(output);
    }

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

/// Reads `node`'s current buffer straight out of `arena` -- a borrow, not a
/// clone. Valid for any node [`build_static_arena`] pre-sized: an
/// [`Op::Input`] slot or a resolved node's output, in whichever state the
/// arena is in right now (freshly computed, or holding a value
/// [`evaluate_named_with_arena_in_place`]'s rebind aliasing swapped in).
#[must_use]
pub fn arena_output(arena: &StaticArena, node: NodeId) -> Option<&[f32]> {
    arena
        .buffers
        .get(node.0 as usize)
        .and_then(Option::as_deref)
}

/// Engagement evidence for law 6∘5 (weight packing): how many `resolved`
/// nodes `build_static_arena_with_constants` packed a `b` operand for, so a
/// re-prove command can assert `N > 0` on a real graph instead of trusting
/// packing ran — same "a gate that cannot report its N is not a gate" shape
/// [`epilogue_fuse_totals`]/[`rewrite_engine_depth_fires`] already serve for
/// the two fusion laws.
#[must_use]
pub fn arena_packed_node_count(arena: &StaticArena) -> usize {
    arena.packed_width_panels.len()
}

/// Diagnostic-only measurement for the rebind-identity task (`docs/
/// discipline.md`, 2026-09-01): `(packed_bytes, unpacked_bytes)` across
/// every [`StaticArena::input_names`] entry, split by whether that node
/// backs a live [`PackedWidthPanels::source`] (any panel, since a weight
/// can in principle back more than one). Answers "how much of the rebind
/// compare's own byte volume would a packed-only compare actually cover"
/// BEFORE assuming that direction helps -- never called from
/// [`bind_named_inputs_into_arena`]'s own hot path.
#[doc(hidden)]
#[must_use]
pub fn arena_named_byte_split(arena: &StaticArena) -> (usize, usize) {
    let mut packed_bytes = 0usize;
    let mut unpacked_bytes = 0usize;
    for (node, _) in &arena.input_names {
        let bytes = arena.buffers[node.0 as usize]
            .as_deref()
            .map_or(0, core::mem::size_of_val);
        let backs_panel = arena
            .packed_width_panels
            .values()
            .any(|panel| panel.source == *node);
        if backs_panel {
            packed_bytes += bytes;
        } else {
            unpacked_bytes += bytes;
        }
    }
    (packed_bytes, unpacked_bytes)
}

/// The in-place counterpart to [`evaluate_named_with_arena`]: a caller
/// whose `rebind` targets stay resident IN the arena across steps (a
/// training loop's own parameters and optimizer state, per
/// `docs/discipline.md` ROW 164's own named residual -- "the rebind
/// targets... get cloned out because arena buffers must survive the next
/// call") never pays that clone at all. `named` here carries ONLY this
/// step's genuinely-new bindings (a batch, a step counter) -- every
/// `rebind` name is expected to already be resident from the PRIOR call's
/// own aliasing swap (or, on the very first call, from a `named` entry the
/// caller supplied once up front).
///
/// After running `program`, every `(computed, input_name)` pair in
/// `rebind` is spliced directly into place with `Vec::swap`: the
/// computed node's freshly written buffer BECOMES the `input_name` node's
/// buffer, and the stale old input buffer moves to the computed node's own
/// slot, where the next call's resolved-node pass fully overwrites it
/// again (matching `evaluate_pooled`'s own "every write position gets
/// overwritten before any read" contract) -- zero allocation, zero `f32`
/// copied, only two `Vec<f32>` headers exchanged.
///
/// What a caller can do with this that [`evaluate_named_with_arena`] alone
/// cannot: run N steps of a `rebind`-shaped loop with the state-carrying
/// buffers touched exactly zero times between [`build_static_arena`] and
/// the caller's own final read-out, instead of a clone-out-then-copy-in
/// pair on every single step.
///
/// # Errors
/// The same errors [`evaluate_named_with_arena`] raises for the `named`
/// bindings it IS given, plus [`TensorError::UnboundInputName`] if a
/// `rebind` pair names an [`Op::Input`] `build_static_arena` never bound,
/// and [`TensorError::InputSizeMismatch`] if a `rebind` pair's computed
/// and input buffers are not the same length (a genuinely different-shaped
/// rebind, not the same-program repeated-step case this exists for).
pub fn evaluate_named_with_arena_in_place(
    arena: &mut StaticArena,
    named: &[(&str, &[f32])],
    loss: NodeId,
    rebind: &[(NodeId, &str)],
) -> Result<f32, TensorError> {
    bind_named_inputs_into_arena(arena, named, false)?;
    run_resolved_nodes_in_arena(arena)?;

    let loss_value = arena_output(arena, loss)
        .and_then(|data| data.first().copied())
        .unwrap_or(0.0);

    for (computed, name) in rebind {
        let input_node = arena
            .input_names
            .iter()
            .find(|(_, candidate)| candidate == name)
            .map(|(node, _)| *node)
            .ok_or_else(|| TensorError::UnboundInputName(String::from(*name)))?;
        let computed_index = computed.0 as usize;
        let input_index = input_node.0 as usize;
        let computed_len = arena.buffers[computed_index].as_ref().map_or(0, Vec::len);
        let input_len = arena.buffers[input_index].as_ref().map_or(0, Vec::len);
        if computed_len != input_len {
            return Err(TensorError::InputSizeMismatch {
                node: input_node,
                expected: input_len,
                found: computed_len,
            });
        }
        arena.buffers.swap(computed_index, input_index);
    }

    Ok(loss_value)
}

/// Reads `name`'s current resident buffer straight out of `arena` -- a
/// borrow, not a clone. `name` is any [`Op::Input`] [`build_static_arena`]
/// bound; the value returned reflects whatever the arena currently holds
/// for it, whether bound by the last [`evaluate_named_with_arena`]/
/// [`evaluate_named_with_arena_in_place`] call's own `named` or spliced in
/// by [`evaluate_named_with_arena_in_place`]'s rebind aliasing -- the
/// read-out a caller uses once, at the end of a run, to pull a `rebind`
/// loop's final state back into its own owned buffers.
#[must_use]
pub fn arena_named_input<'arena>(arena: &'arena StaticArena, name: &str) -> Option<&'arena [f32]> {
    let node = arena
        .input_names
        .iter()
        .find(|(_, candidate)| candidate == name)
        .map(|(node, _)| *node)?;
    arena_output(arena, node)
}

/// One [`evaluate_quantized`]-bound block: either a plain `f32`
/// [`Op::Input`] buffer, exactly what [`evaluate`]'s own `blocks: &[&[f32]]`
/// carries, or the raw packed bytes of a `Q4_K`-quantized weight matrix.
/// [`evaluate`]'s `blocks` parameter has no way to carry the second case — a
/// quantized weight has no legitimate `&[f32]` view to hand through it
/// without dequantizing first, which would defeat the entire point (see
/// [`matmul_q4k_f32`]'s doc on what dequantizing first costs). Both variants
/// bind positionally, in the same [`Op::Input`] program order [`evaluate`]'s
/// `blocks` already uses — one binding convention, not two.
#[derive(Debug, Clone, Copy)]
pub enum QuantizedBlock<'a> {
    Float32(&'a [f32]),
    Int32(&'a [i32]),
    Q4K(&'a [u8]),
    /// Raw packed `Q5_K` bytes -- same super-block shape as [`Self::Q4K`]
    /// (256 elements, 8 sub-blocks of 32) plus a `qh` high-bit plane; see
    /// [`proxima_gguf::quant::q5_k`] for the on-disk layout this borrows
    /// unchanged.
    Q5K(&'a [u8]),
    /// Raw packed `Q3_K` bytes -- 256 elements, 16 sub-blocks of 16, one
    /// signed 6-bit scale per sub-block and no per-sub-block min (`x =
    /// d*sc*q`); see [`proxima_gguf::quant::q3_k`] for the on-disk layout
    /// this borrows unchanged.
    Q3K(&'a [u8]),
    /// Raw packed `Q2_K` bytes -- 256 elements, 16 sub-blocks of 16, one
    /// 4-bit scale AND 4-bit min per sub-block, packed one byte per
    /// sub-block (`x = d*sc*q - dmin*m`); see
    /// [`proxima_gguf::quant::q2_k`] for the on-disk layout this borrows
    /// unchanged. No `dot_fn_for` entry (no shared int8-wide-fold path,
    /// same reasoning as [`Self::Q3K`]) -- this codec's only CPU path is
    /// the scalar dequantize-then-fold `dot_q2k_f32`.
    Q2K(&'a [u8]),
    /// Raw packed `Q6_K` bytes -- 256 elements, 16 sub-blocks of 16, one
    /// signed 8-bit scale per sub-block and no `dmin` term; see
    /// [`proxima_gguf::quant::q6_k`] for the on-disk layout this borrows
    /// unchanged.
    Q6K(&'a [u8]),
    /// Raw packed `Q8_0` bytes -- 32-element blocks, one `f16` scale per
    /// block, no sub-block structure at all; see
    /// [`proxima_gguf::quant::q8_0`] for the on-disk layout this borrows
    /// unchanged. The one variant this enum carries that the growable
    /// per-layer key/value context cache (`proxima-model-interop`'s
    /// `LayerCache`) actually binds -- its rows are `HEAD_DIM / 2`
    /// elements wide, small enough that `Q4_K`/`Q5_K`/`Q6_K`'s 256-element
    /// super-blocks would straddle more than one cached position, while a
    /// 32-element `Q8_0` block divides a typical head dimension evenly.
    Q8_0(&'a [u8]),
    /// Raw packed `Q4_0` bytes -- 32-element blocks, one `f16` scale per
    /// block, no sub-block structure and no shared super-block with the
    /// K-quant family; see [`proxima_gguf::quant::q4_0`] for the on-disk
    /// layout this borrows unchanged. Legacy llama.cpp's simplest and most
    /// widely distributed 4-bit format -- unlike [`Self::Q4K`], no
    /// sub-block scale/min hierarchy, just `value = scale * (nibble - 8)`.
    Q4_0(&'a [u8]),
    /// Raw packed `Q5_1` bytes -- 32-element blocks, one `f16` scale and
    /// one `f16` min per block plus a 4-byte 5th-bit plane, no shared
    /// super-block with the K-quant family; see
    /// [`proxima_gguf::quant::q5_1`] for the on-disk layout this borrows
    /// unchanged. The target checkpoint's SSM tensor codec (252 tensors,
    /// 0.64 GB) -- decode-only, same reasoning as [`Self::Q4_0`] for why no
    /// `dot_fn_for` entry exists (no shared int8-wide-fold path).
    Q5_1(&'a [u8]),
    /// Raw packed `Q5_0` bytes -- 32-element blocks, one `f16` scale per
    /// block plus a 4-byte 5th-bit plane, no per-block min term (unlike
    /// [`Self::Q5_1`]) and no shared super-block with the K-quant family;
    /// see [`proxima_gguf::quant::q5_0`] for the on-disk layout this
    /// borrows unchanged. `Q5_1` with the min dropped -- `value = d *
    /// (level - 16)`, `level` the 5-bit nibble+`qh` union -- equivalently
    /// `Q4_0` with a 5th bit. gemma4's `blk.{1..29}.ffn_down_exps.weight`
    /// codec -- decode-only, same reasoning as [`Self::Q5_1`] for why no
    /// `dot_fn_for` entry exists (no shared int8-wide-fold path).
    Q5_0(&'a [u8]),
    /// Raw packed `IQ4_NL` bytes -- 32-element blocks, one `f16` scale per
    /// block, byte-identical shape to [`Self::Q4_0`] but with a non-linear
    /// codebook (`kvalues_iq4nl`) instead of a fixed `nibble - 8` recenter;
    /// see [`proxima_gguf::quant::iq4_nl`] for the on-disk layout this
    /// borrows unchanged. The target checkpoint's per-layer-token-embedding
    /// ngram table codec (dims `[160, 320001536]`, 28.8 GB) -- decode-only.
    Iq4Nl(&'a [u8]),
    /// Raw packed `IQ2_XS` bytes -- 256-element super-blocks, one `f16`
    /// scale, 8 sub-block scale nibbles, and a 512-entry grid + 128-entry
    /// sign codebook per element group; see
    /// [`proxima_gguf::quant::iq2_xs`] for the on-disk layout this borrows
    /// unchanged. Part of the `UD-Q2_K_XL` codec set -- decode-only.
    Iq2Xs(&'a [u8]),
    /// Raw packed `IQ3_XXS` bytes -- 256-element super-blocks, one `f16`
    /// scale, 8 packed 32-bit aux words (4-bit sub-block scale + four 7-bit
    /// sign fields), and a 256-entry grid codebook shared with no other
    /// format; see [`proxima_gguf::quant::iq3_xxs`] for the on-disk layout
    /// this borrows unchanged. Part of the `UD-Q2_K_XL` codec set --
    /// decode-only.
    Iq3Xxs(&'a [u8]),
    /// Raw packed IEEE-754 binary16 bytes, little-endian, two per element,
    /// no block or scale structure at all -- unlike every other
    /// non-`Float32` variant above, a half-precision weight is not
    /// quantized, only narrower: each element converts to `f32` entirely on
    /// its own, with no neighbours' scale to consult. See [`matmul_f16_f32`]
    /// for the composed convert-then-fold kernel this variant reaches, and
    /// [`proxima_gguf::quant::f16`] for the on-disk layout this borrows
    /// unchanged.
    Float16(&'a [u8]),
    /// Raw packed `bfloat16` bytes, little-endian, two per element -- same
    /// per-element (non-block) shape as [`Self::Float16`], but a different
    /// bit layout (8-bit exponent, 7-bit mantissa) needing its own
    /// conversion. See [`matmul_bf16_f32`] and [`proxima_gguf::quant::bf16`].
    BFloat16(&'a [u8]),
}

/// One expert's own gathered-reduce weight, resolved by
/// [`ExpertSource::entry`] at read time -- a [`QuantizedBlock`] naming its
/// own codec (Q4_K, Q2_K, Q8_0, ... any of them, independently per entry,
/// so a hi-precision copy and a lo-precision copy of the same layer's
/// expert can coexist in one [`ExpertSource`]) plus the declared
/// `[out_dim, in_dim]` shape `run_reduce_quantized`'s own program-derived
/// `rows`/`k` must agree with before this entry's bytes are ever dotted
/// against an activation row.
#[derive(Debug, Clone, Copy)]
pub struct ExpertEntry<'a> {
    pub block: QuantizedBlock<'a>,
    pub out_dim: u32,
    pub in_dim: u32,
    /// Captured once, at the start of the evaluation step that borrows this
    /// entry's [`ExpertSource`] -- not consulted by `run_reduce_quantized`
    /// itself (there is nothing yet to compare it against mid-step), and
    /// carried here so a caller snapshotting a table for one step can tell
    /// two entries for the same expert slot apart across steps. The borrow
    /// itself is what actually prevents promotion/eviction from changing
    /// bytes under a running step (see [`ExpertSource`]'s own doc); `epoch`
    /// is the caller-visible label for which snapshot a step ran against.
    pub epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertPayloadSpan {
    pub offset: u32,
    pub length: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ExpertPayloadArena<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) spans: &'a [Option<ExpertPayloadSpan>],
}

impl<'a> ExpertPayloadArena<'a> {
    #[must_use]
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    #[must_use]
    pub const fn spans(self) -> &'a [Option<ExpertPayloadSpan>] {
        self.spans
    }
}

/// A per-expert weight table for `run_reduce_quantized`'s gathered-reduce
/// read path, standing in for one contiguous [`QuantizedBlock`] stack
/// (`proxima-gguf::restack`'s own byte-concatenation contract) when a
/// caller needs experts whose codec, or whose underlying allocation, differ
/// from each other -- e.g. one expert promoted to a higher-precision copy
/// while its siblings stay at the checkpoint's native codec.
///
/// Borrowed for the duration of exactly one evaluation step: the executor
/// snapshots (builds) this table when the step begins, and every gathered
/// read inside that step resolves through the SAME borrowed `entries`
/// slice. A residency policy that promotes or evicts an expert between
/// steps does so by handing the NEXT step's evaluation a new `ExpertSource`
/// borrow over the new bytes -- it can never mutate bytes a running step
/// already borrowed, because the borrow checker, not a runtime lock, is
/// what forbids it. This is the whole of the lifetime contract: codec,
/// address, and layout agree for the entry's whole `'a` lifetime, and that
/// lifetime never outlives one step.
///
/// [`bind_moe_stacked_experts`](../../proxima-model-interop/src/bind.rs)'s
/// own default construction slices one contiguous stack into `entries` that
/// alias the stack's own bytes -- zero copy, and bit-identical to reading
/// the stack directly (see `cpu.rs`'s own `expert_source_matches_stack_gather`
/// test).
#[derive(Debug, Clone, Copy)]
pub struct ExpertSource<'a> {
    pub(super) entries: &'a [ExpertEntry<'a>],
    pub(super) selected_expert_ids: Option<&'a [u32]>,
    pub(super) packed_arena: Option<ExpertPayloadArena<'a>>,
}

impl<'a> ExpertSource<'a> {
    /// Borrows `entries` as-is, one per expert index -- `entries[e]` is
    /// expert `e`'s own weight. No validation here; [`Self::entry`] is
    /// where an out-of-range or wrong-shape expert becomes a typed error,
    /// at the point `run_reduce_quantized` actually needs it.
    #[must_use]
    pub const fn new(entries: &'a [ExpertEntry<'a>]) -> Self {
        Self {
            entries,
            selected_expert_ids: None,
            packed_arena: None,
        }
    }

    /// Borrows a caller-owned fixed-capacity route list for backend staging.
    /// The entries remain indexed by their original expert ID; the list only
    /// controls which payloads a backend uploads for this snapshot.
    #[must_use]
    pub const fn with_selected_expert_ids(
        entries: &'a [ExpertEntry<'a>],
        selected_expert_ids: &'a [u32],
    ) -> Self {
        Self {
            entries,
            selected_expert_ids: Some(selected_expert_ids),
            packed_arena: None,
        }
    }

    pub fn with_selected_expert_arena(
        entries: &'a [ExpertEntry<'a>],
        selected_expert_ids: &'a [u32],
        bytes: &'a [u8],
        spans: &'a [Option<ExpertPayloadSpan>],
    ) -> Result<Self, TensorError> {
        Self::with_expert_arena(entries, Some(selected_expert_ids), bytes, spans)
    }

    /// Borrows one mmap-backed arena containing every expert payload. The
    /// descriptor table remains dense in route space, so the Metal backend can
    /// bind the arena before routing without staging selected bytes first.
    pub fn with_all_expert_arena(
        entries: &'a [ExpertEntry<'a>],
        bytes: &'a [u8],
        spans: &'a [Option<ExpertPayloadSpan>],
    ) -> Result<Self, TensorError> {
        if spans.len() != entries.len() {
            return Err(TensorError::InvalidExpertPayloadArena {
                reason: "all-expert arena span count does not match the expert table",
            });
        }
        for (entry, span) in entries.iter().zip(spans) {
            let span = span.ok_or(TensorError::InvalidExpertPayloadArena {
                reason: "all-expert arena is missing an expert payload span",
            })?;
            let packed_bytes =
                entry
                    .block
                    .packed_bytes()
                    .ok_or(TensorError::InvalidExpertPayloadArena {
                        reason: "all-expert arena entries must use packed bytes",
                    })?;
            if usize::try_from(span.length).ok() != Some(packed_bytes.len()) {
                return Err(TensorError::InvalidExpertPayloadArena {
                    reason: "all-expert arena span length does not match its expert payload",
                });
            }
        }
        Self::with_expert_arena(entries, None, bytes, spans)
    }

    fn with_expert_arena(
        entries: &'a [ExpertEntry<'a>],
        selected_expert_ids: Option<&'a [u32]>,
        bytes: &'a [u8],
        spans: &'a [Option<ExpertPayloadSpan>],
    ) -> Result<Self, TensorError> {
        if spans.len() < entries.len() {
            return Err(TensorError::InvalidExpertPayloadArena {
                reason: "arena span table is shorter than the expert table",
            });
        }
        let mut present = Vec::new();
        for span in spans.iter().flatten() {
            let end = usize::try_from(span.offset)
                .ok()
                .and_then(|offset| {
                    usize::try_from(span.length)
                        .ok()
                        .and_then(|length| offset.checked_add(length))
                })
                .ok_or(TensorError::InvalidExpertPayloadArena {
                    reason: "arena span endpoint overflowed",
                })?;
            if end > bytes.len() {
                return Err(TensorError::InvalidExpertPayloadArena {
                    reason: "arena span exceeds arena bytes",
                });
            }
            present.push((usize::try_from(span.offset).unwrap_or(usize::MAX), end));
        }
        present.sort_unstable();
        if present.windows(2).any(|window| window[1].0 < window[0].1) {
            return Err(TensorError::InvalidExpertPayloadArena {
                reason: "arena spans overlap",
            });
        }
        Ok(Self {
            entries,
            selected_expert_ids,
            packed_arena: Some(ExpertPayloadArena { bytes, spans }),
        })
    }

    /// Returns the immutable per-expert snapshot so a device backend can
    /// materialize the same table without changing the step's borrowed
    /// lifetime contract.
    #[must_use]
    pub const fn entries(&self) -> &'a [ExpertEntry<'a>] {
        self.entries
    }

    /// Returns the optional caller-owned route list used by a backend to
    /// compact staged payload bytes. `None` means every entry is selected.
    #[must_use]
    pub const fn selected_expert_ids(&self) -> Option<&'a [u32]> {
        self.selected_expert_ids
    }

    #[must_use]
    pub const fn packed_arena(&self) -> Option<ExpertPayloadArena<'a>> {
        self.packed_arena
    }

    /// Resolves expert `index`'s own entry, rejecting a shape that
    /// disagrees with the program's own resolved `[expected_out,
    /// expected_in]` for this gathered reduce -- [`TensorError::ExpertSourceShapeMismatch`],
    /// naming `node` and `expert` for the caller, never a silent
    /// wrong-shape dot product.
    ///
    /// Public (not just `run_reduce_quantized`'s own internal use) so a
    /// residency policy building an [`ExpertSource`] -- `proxima-model-interop`'s
    /// `ExpertSlab` among them -- can assert what a snapshot resolves
    /// without a full evaluation.
    #[must_use = "an unused resolved expert entry is a no-op"]
    pub fn entry(
        &self,
        node: NodeId,
        index: usize,
        expected_out: u32,
        expected_in: u32,
    ) -> Result<ExpertEntry<'a>, TensorError> {
        let entry = *self
            .entries
            .get(index)
            .ok_or(TensorError::GatherIndexOutOfRange {
                node,
                index: index as i64,
                extent: self.entries.len() as u64,
            })?;
        if entry.out_dim != expected_out || entry.in_dim != expected_in {
            return Err(TensorError::ExpertSourceShapeMismatch {
                node,
                expert: index as u32,
                entry_out: entry.out_dim,
                entry_in: entry.in_dim,
                expected_out,
                expected_in,
            });
        }
        Ok(entry)
    }
}

/// [`ExpertSource`]'s own default construction: `stack` is one contiguous
/// [`QuantizedBlock`] carrying `expert_count` back-to-back `[out_dim,
/// in_dim]` slabs (`proxima-gguf::restack`'s own byte-concatenation
/// contract) -- the entries this builds ALIAS `stack`'s own bytes, one
/// `stack.packed_bytes().len() / expert_count`-wide slice per expert, zero
/// copy, and bit-identical to reading the stack directly at
/// `expert_index * per_expert_bytes` the way `run_reduce_quantized`'s own
/// non-`ExpertSource` gather branch does today (see this crate's own
/// `expert_source_matches_stack_gather` test).
///
/// # Errors
/// [`TensorError::ExpertStackNotAligned`] if `stack`'s packed byte length is
/// not a whole multiple of `expert_count`; [`TensorError::EmptyExpertPayload`]
/// if `stack` carries zero packed bytes for a nonzero `expert_count` (`0 % n
/// == 0` passes the alignment check above, but a `0`-wide chunk has no bytes
/// to alias, and `chunks_exact` panics if asked for a zero-width chunk).
#[must_use = "an unused expert table alias is a no-op"]
pub fn expert_entries_from_stack(
    stack: QuantizedBlock<'_>,
    expert_count: usize,
    out_dim: u32,
    in_dim: u32,
    epoch: u64,
) -> Result<Vec<ExpertEntry<'_>>, TensorError> {
    let bytes = stack
        .packed_bytes()
        .ok_or(TensorError::ExpertStackNotAligned {
            expert_count,
            bytes: 0,
        })?;
    if expert_count == 0 || !bytes.len().is_multiple_of(expert_count) {
        return Err(TensorError::ExpertStackNotAligned {
            expert_count,
            bytes: bytes.len(),
        });
    }
    if bytes.is_empty() {
        return Err(TensorError::EmptyExpertPayload { expert_count });
    }
    let per_expert_bytes = bytes.len() / expert_count;
    // Owned `Vec` output, not a lazy borrowed view: `ExpertSource` (this
    // function's only real consumer) holds `&'a [ExpertEntry<'a>]` -- a
    // materialized slice, not an iterator or index-computed accessor -- so
    // the table has to exist somewhere the caller can take `&entries` of.
    // Called once per evaluation step (see `ExpertSource`'s own doc: "the
    // executor snapshots this table when the step begins"), not once per
    // position inside a step, so this is bounded setup-path allocation
    // (`expert_count` entries), never a hot-path allocation.
    Ok(bytes
        .chunks_exact(per_expert_bytes)
        .map(|chunk| ExpertEntry {
            block: stack.with_bytes(chunk),
            out_dim,
            in_dim,
            epoch,
        })
        .collect())
}

impl<'a> QuantizedBlock<'a> {
    /// The packed byte slice underneath any codec, or `None` for
    /// [`Self::Float32`] (which carries `&[f32]`, not packed bytes) --
    /// [`ExpertSource`]'s own gather read uses this to swap an
    /// [`ExpertEntry`]'s bytes into the per-codec matmul dispatch below
    /// without re-deriving which variant carries a `&[u8]` payload a second
    /// time. Bound to `'a`, not `&self`'s own borrow, so a caller reading
    /// this out of a short-lived local (e.g. one loop iteration's own
    /// resolved [`ExpertEntry`]) still gets bytes that outlive that local.
    #[must_use]
    pub const fn packed_bytes(&self) -> Option<&'a [u8]> {
        match self {
            QuantizedBlock::Float32(_) | QuantizedBlock::Int32(_) => None,
            QuantizedBlock::Q4K(bytes)
            | QuantizedBlock::Q5K(bytes)
            | QuantizedBlock::Q3K(bytes)
            | QuantizedBlock::Q2K(bytes)
            | QuantizedBlock::Q6K(bytes)
            | QuantizedBlock::Q8_0(bytes)
            | QuantizedBlock::Q4_0(bytes)
            | QuantizedBlock::Q5_1(bytes)
            | QuantizedBlock::Q5_0(bytes)
            | QuantizedBlock::Iq4Nl(bytes)
            | QuantizedBlock::Iq2Xs(bytes)
            | QuantizedBlock::Iq3Xxs(bytes)
            | QuantizedBlock::Float16(bytes)
            | QuantizedBlock::BFloat16(bytes) => Some(bytes),
        }
    }

    /// Packed-block `(block_bytes, block_elements)` footprint for this
    /// codec, or `None` for [`Self::Float32`]/[`Self::Int32`] (decoded
    /// values, not packed blocks) -- the one table
    /// `run_reduce_quantized`, `build_matmul_stage_plan`, and
    /// `dequantize_row` all compose down to instead of each hand-matching
    /// the same per-codec constants a second and third time. Sourced from
    /// the same `proxima_gguf::quant::*` block-size constants
    /// [`proxima_gguf::GgmlType::block_layout`] wraps for its own codec
    /// table.
    #[must_use]
    pub const fn block_layout(&self) -> Option<(usize, usize)> {
        match self {
            QuantizedBlock::Float32(_) | QuantizedBlock::Int32(_) => None,
            QuantizedBlock::Q4K(_) => Some((Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS)),
            QuantizedBlock::Q5K(_) => Some((Q5K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS)),
            QuantizedBlock::Q3K(_) => Some((Q3K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS)),
            QuantizedBlock::Q2K(_) => Some((Q2K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS)),
            QuantizedBlock::Q6K(_) => Some((Q6K_BLOCK_BYTES, Q4K_BLOCK_ELEMENTS)),
            QuantizedBlock::Q8_0(_) => Some((Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS)),
            QuantizedBlock::Q4_0(_) => Some((Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS)),
            QuantizedBlock::Q5_1(_) => Some((Q5_1_BLOCK_BYTES, Q5_1_BLOCK_ELEMENTS)),
            QuantizedBlock::Q5_0(_) => Some((Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMENTS)),
            QuantizedBlock::Iq4Nl(_) => Some((IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMENTS)),
            QuantizedBlock::Iq2Xs(_) => Some((IQ2_XS_BLOCK_BYTES, IQ2_XS_BLOCK_ELEMENTS)),
            QuantizedBlock::Iq3Xxs(_) => Some((IQ3_XXS_BLOCK_BYTES, IQ3_XXS_BLOCK_ELEMENTS)),
            QuantizedBlock::Float16(_) | QuantizedBlock::BFloat16(_) => {
                Some((HALF_PRECISION_ELEMENT_BYTES, 1))
            }
        }
    }

    /// Rewraps `self`'s own codec discriminant around a different `'a`
    /// byte slice -- [`expert_entries_from_stack`]'s own per-expert slicing,
    /// and `run_reduce_quantized`'s per-position gather read, both need "the
    /// same codec, different bytes" without restating this crate's own
    /// codec list a second time. [`Self::Float32`] cannot itself hold
    /// `&'a [u8]` (it wraps `&'a [f32]`), so that arm rewraps to an
    /// arbitrary packed variant (`Q4K`) instead -- never reached in
    /// practice, since [`Self::packed_bytes`] already returns `None` for
    /// `Float32` and every caller of this method checks that first before
    /// ever reaching here. Kept total rather than partial so this stays a
    /// plain function, not a fallible one, at every other call site.
    #[must_use]
    pub const fn with_bytes(&self, bytes: &'a [u8]) -> Self {
        match self {
            QuantizedBlock::Float32(_) | QuantizedBlock::Int32(_) => QuantizedBlock::Q4K(bytes),
            QuantizedBlock::Q4K(_) => QuantizedBlock::Q4K(bytes),
            QuantizedBlock::Q5K(_) => QuantizedBlock::Q5K(bytes),
            QuantizedBlock::Q3K(_) => QuantizedBlock::Q3K(bytes),
            QuantizedBlock::Q2K(_) => QuantizedBlock::Q2K(bytes),
            QuantizedBlock::Q6K(_) => QuantizedBlock::Q6K(bytes),
            QuantizedBlock::Q8_0(_) => QuantizedBlock::Q8_0(bytes),
            QuantizedBlock::Q4_0(_) => QuantizedBlock::Q4_0(bytes),
            QuantizedBlock::Q5_1(_) => QuantizedBlock::Q5_1(bytes),
            QuantizedBlock::Q5_0(_) => QuantizedBlock::Q5_0(bytes),
            QuantizedBlock::Iq4Nl(_) => QuantizedBlock::Iq4Nl(bytes),
            QuantizedBlock::Iq2Xs(_) => QuantizedBlock::Iq2Xs(bytes),
            QuantizedBlock::Iq3Xxs(_) => QuantizedBlock::Iq3Xxs(bytes),
            QuantizedBlock::Float16(_) => QuantizedBlock::Float16(bytes),
            QuantizedBlock::BFloat16(_) => QuantizedBlock::BFloat16(bytes),
        }
    }

    /// Element count this block decodes to, derived from its own codec's
    /// block geometry -- [`proxima_gguf::quant`]'s per-format
    /// `blocks_for_bytes`/`elements_for_blocks` pair, never `bytes.len()`
    /// directly (a `Q4_K` super-block is 144 bytes carrying 256 elements;
    /// bytes and elements are not the same unit). The one definition every
    /// caller that needs "how many f32 elements does this packed buffer
    /// decode to" composes down to, instead of restating this per-codec
    /// table itself.
    ///
    /// # Errors
    /// [`TensorError::PackedBlockBytesNotAMultiple`] if a non-[`Self::Float32`]
    /// variant's byte length is not a whole multiple of its codec's block
    /// size -- never legitimate GGUF, only ever corrupt, truncated, or
    /// misattributed bytes.
    pub fn element_count(&self) -> Result<usize, TensorError> {
        let (codec, bytes, block_bytes, blocks) = match self {
            QuantizedBlock::Float32(data) => return Ok(data.len()),
            QuantizedBlock::Int32(data) => return Ok(data.len()),
            QuantizedBlock::Q4K(bytes) => (
                "q4_k",
                bytes.len(),
                q4_k::BLOCK_BYTES,
                q4_k::blocks_for_bytes(bytes.len()).map(q4_k::elements_for_blocks),
            ),
            QuantizedBlock::Q5K(bytes) => (
                "q5_k",
                bytes.len(),
                q5_k::BLOCK_BYTES,
                q5_k::blocks_for_bytes(bytes.len()).map(q5_k::elements_for_blocks),
            ),
            QuantizedBlock::Q3K(bytes) => (
                "q3_k",
                bytes.len(),
                q3_k::BLOCK_BYTES,
                q3_k::blocks_for_bytes(bytes.len()).map(q3_k::elements_for_blocks),
            ),
            QuantizedBlock::Q2K(bytes) => (
                "q2_k",
                bytes.len(),
                q2_k::BLOCK_BYTES,
                q2_k::blocks_for_bytes(bytes.len()).map(q2_k::elements_for_blocks),
            ),
            QuantizedBlock::Q6K(bytes) => (
                "q6_k",
                bytes.len(),
                q6_k::BLOCK_BYTES,
                q6_k::blocks_for_bytes(bytes.len()).map(q6_k::elements_for_blocks),
            ),
            QuantizedBlock::Q8_0(bytes) => (
                "q8_0",
                bytes.len(),
                q8_0::BLOCK_BYTES,
                q8_0::blocks_for_bytes(bytes.len()).map(q8_0::elements_for_blocks),
            ),
            QuantizedBlock::Q4_0(bytes) => (
                "q4_0",
                bytes.len(),
                q4_0::BLOCK_BYTES,
                q4_0::blocks_for_bytes(bytes.len()).map(q4_0::elements_for_blocks),
            ),
            QuantizedBlock::Q5_1(bytes) => (
                "q5_1",
                bytes.len(),
                q5_1::BLOCK_BYTES,
                q5_1::blocks_for_bytes(bytes.len()).map(q5_1::elements_for_blocks),
            ),
            QuantizedBlock::Q5_0(bytes) => (
                "q5_0",
                bytes.len(),
                q5_0::BLOCK_BYTES,
                q5_0::blocks_for_bytes(bytes.len()).map(q5_0::elements_for_blocks),
            ),
            QuantizedBlock::Iq4Nl(bytes) => (
                "iq4_nl",
                bytes.len(),
                iq4_nl::BLOCK_BYTES,
                iq4_nl::blocks_for_bytes(bytes.len()).map(iq4_nl::elements_for_blocks),
            ),
            QuantizedBlock::Iq2Xs(bytes) => (
                "iq2_xs",
                bytes.len(),
                iq2_xs::BLOCK_BYTES,
                iq2_xs::blocks_for_bytes(bytes.len()).map(iq2_xs::elements_for_blocks),
            ),
            QuantizedBlock::Iq3Xxs(bytes) => (
                "iq3_xxs",
                bytes.len(),
                iq3_xxs::BLOCK_BYTES,
                iq3_xxs::blocks_for_bytes(bytes.len()).map(iq3_xxs::elements_for_blocks),
            ),
            // f16/bf16 blocks are one element wide (`QK_F16`/`QK_BF16` == 1),
            // so a block count already IS the element count -- neither module
            // exposes its own `elements_for_blocks`.
            QuantizedBlock::Float16(bytes) => (
                "float16",
                bytes.len(),
                gguf_f16::BLOCK_BYTES,
                gguf_f16::blocks_for_bytes(bytes.len()),
            ),
            QuantizedBlock::BFloat16(bytes) => (
                "bfloat16",
                bytes.len(),
                gguf_bf16::BLOCK_BYTES,
                gguf_bf16::blocks_for_bytes(bytes.len()),
            ),
        };
        blocks.ok_or(TensorError::PackedBlockBytesNotAMultiple {
            codec,
            bytes,
            block_bytes,
        })
    }
}
