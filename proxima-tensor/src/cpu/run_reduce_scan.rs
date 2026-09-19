use super::*;

/// One `(node, expert)` key's selection history: `count` since the last
/// [`snapshot_expert_selection_top_n`] drain, and `ema` -- updated toward
/// `1.0` by [`EXPERT_SELECTION_EMA_ALPHA`] on every selection
/// ([`ExpertSelectionEntry::record`]'s own doc). Both fields are the raw
/// inputs a per-node precision-allocation rule needs; this module emits
/// them and implements neither the rule nor its threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ExpertSelectionEntry {
    pub(super) count: u64,
    pub(super) ema: f64,
}

impl ExpertSelectionEntry {
    const ZERO: Self = Self { count: 0, ema: 0.0 };

    /// One selection event: increments `count`, and moves `ema` toward
    /// `1.0` by the standard exponential-moving-average recurrence `ema +=
    /// alpha * (target - ema)` with `target = 1.0` (a selection just
    /// happened) -- a rarely-selected expert's `ema` stays near `0.0`
    /// (mostly non-events between rare `record` calls would pull it back
    /// down were this method also called for a "not selected" observation,
    /// which it is not: see this struct's own doc for what a fuller
    /// per-step hotness signal would still need).
    fn record(&mut self) {
        self.count += 1;
        self.ema += EXPERT_SELECTION_EMA_ALPHA * (1.0 - self.ema);
    }
}

/// Per-(gathered-reduce-node, expert) selection history for
/// [`run_reduce_quantized`]'s own `expert_index` resolution below -- the
/// popularity input a per-node precision policy (choosing which experts to
/// bind at a lower precision, e.g. `proxima-model-interop`'s own
/// `ServingConfig::weight_precision`) would consume.
///
/// Keyed by the gathered reduce op's own [`NodeId`] rather than a decoded
/// "layer number": [`run_reduce_quantized`] is architecture-agnostic and
/// never learns which layer it evaluates, but every mixture-of-experts
/// forward program this crate builds emits exactly one gathered reduce node
/// per layer (`spec.rs`'s own `gathered_expert_product`), so a compiled
/// program's own `NodeId` is already a stable per-layer identity a caller
/// holding that program can map back to a layer number -- no second
/// index-to-layer table needed here.
///
/// `Mutex`-guarded `BTreeMap`, the same recovered-on-poison shape
/// [`ARENA_CACHE`]/[`lock_arena_cache`] already establish in this file, not
/// a lock-free flat array: `expert_count` is a per-checkpoint runtime value
/// with no `sized.rs` build-time constant to size an array from, so an
/// unbounded sparse `(node, expert)` key is the honest shape -- and
/// recording happens once per gathered position per round
/// (`expert_used_count` rounds x sequence length), an order of magnitude
/// below the per-mac hot loop the `MATMUL_Q4K_MACS` counters instrument, so
/// the lock is held for a `BTreeMap` insert, not inside the matmul kernel
/// itself.
pub(super) static EXPERT_SELECTION_COUNTS: Mutex<BTreeMap<(u32, u32), ExpertSelectionEntry>> =
    Mutex::new(BTreeMap::new());

/// [`EXPERT_SELECTION_COUNTS`]'s own lock, recovered rather than propagated
/// on poisoning -- [`lock_arena_cache`]'s own established pattern in this
/// file: a panic while another caller held the lock leaves counts in
/// whatever state that caller's own increment reached, never a torn entry
/// (every mutation under this lock is one `BTreeMap` insert-or-increment),
/// so recovering and continuing is safe.
pub(super) fn lock_expert_selection_counts()
-> MutexGuard<'static, BTreeMap<(u32, u32), ExpertSelectionEntry>> {
    EXPERT_SELECTION_COUNTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Records one expert selection: `node` is the gathered reduce's own
/// [`NodeId`] ([`run_reduce_quantized`]'s `weight_node`/`resolved.node`),
/// `expert` the resolved `expert_index` a real routed token selected.
pub(super) fn record_expert_selection(node: NodeId, expert: u32) {
    let mut counts = lock_expert_selection_counts();
    counts
        .entry((node.0, expert))
        .or_insert(ExpertSelectionEntry::ZERO)
        .record();
}

/// Drains `EXPERT_SELECTION_COUNTS` and returns its `top_n` entries by
/// raw `count`, highest first, each as `(node, expert, count, ema)` -- the
/// same drain-and-reset contract
/// `proxima_telemetry::metric::Counter::snapshot_and_reset` gives a
/// single scalar counter, generalized to this table's `(node, expert)` key:
/// a later decode step's counts start from zero, never accumulated across
/// the whole run, so a caller reads one step's own routing popularity, not
/// a running total that never resets. `ema` rides alongside `count` on every
/// entry -- both are DynaExq-style allocation-rule inputs, not two separate
/// queries.
#[must_use]
pub fn snapshot_expert_selection_top_n(top_n: usize) -> Vec<(NodeId, u32, u64, f64)> {
    let mut counts = lock_expert_selection_counts();
    let mut entries: Vec<(NodeId, u32, u64, f64)> = core::mem::take(&mut *counts)
        .into_iter()
        .map(|((node, expert), entry)| (NodeId(node), expert, entry.count, entry.ema))
        .collect();
    entries.sort_unstable_by_key(|entry| core::cmp::Reverse(entry.2));
    entries.truncate(top_n);
    entries
}

/// `crate::cpu`'s own `expert_selection` structured event: `node`/`expert`/
/// `count`/`ema` for each of [`snapshot_expert_selection_top_n`]'s top-8
/// entries, one event per call -- a caller's decode loop calls this once
/// per step, the same "once per step, not once per token" granularity
/// `proxima-model-interop`'s own `report_op_timings`-style diagnostics
/// already use for other per-step summaries. Compiled out entirely when the
/// `instrument` feature is off, matching every other diagnostic event in
/// this crate.
///
/// Not yet wired into `evaluate_quantized_with_scratch`'s own per-decode-step
/// call boundary -- that function's `Ok(...)` return sits deep inside a
/// multi-thousand-line body this slice did not attempt to blind-edit
/// without a compiler to check the result against; a caller (or a future
/// slice) calls this explicitly once per step in the meantime.
#[cfg(feature = "instrument")]
pub fn emit_expert_selection_event() {
    for (node, expert, count, ema) in snapshot_expert_selection_top_n(8) {
        proxima_telemetry::info!(
            node = node.0,
            expert,
            count,
            ema,
            "expert_selection: top-8 (node, expert) selection counts and ema-hotness this step"
        );
    }
}

/// [`run_reduce_quantized`]'s admission contract, checked BEFORE it ever
/// touches a buffer: the fused reduce's `element_body` must be exactly one
/// step, `Multiply`, over exactly two physical operands (the packed weight
/// and one already-materialized activation leaf) -- the ONE shape
/// [`matmul_q4k_f32`] (and its sibling K-quant/legacy kernels) actually
/// implements, `activation_row = &activation[...]`, a raw slice read with no
/// re-application of any composed step. `resolved.operands()` alone cannot
/// tell a bare `W * a` apart from a fused `W * (x * sigmoid(g))` -- both
/// still carry exactly the physical buffers their leaves resolved to, and a
/// naive "first operand that isn't the weight" pick silently returns `x`
/// instead of `x * sigmoid(g)` for the latter, computing `matmul(W, x)`
/// where the graph asked for `sum(W * (x * sigmoid(g)))`. Checking
/// `element_body`'s own step count is what tells the two apart: a bare
/// product is [`ComposedBody::leaf`]`(Multiply)`, one step, args
/// `[Operand(_), Operand(_)]`; anything the planner fused beyond that shows
/// up here as more steps, or a `Step` arg referencing an earlier one, either
/// of which must reject rather than silently pick a leaf. See
/// `docs/discipline.md` ROW 431.
#[allow(dead_code)]
pub(super) fn packed_reduce_activation_operand(
    resolved: &BoundOp,
    weight_node: NodeId,
) -> Result<NodeId, TensorError> {
    let BoundOpKind::Reduce {
        element_body,
        operands,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_reduce_quantized is only called for a Keep::Reduce fold")
    };
    let is_bare_weight_activation_product = element_body.steps.len() == 1
        && element_body.steps[0].op == ScalarOp::Multiply
        && matches!(
            element_body.steps[0].args.as_slice(),
            [StepArg::Operand(_), StepArg::Operand(_)]
        )
        && operands.len() == 2;
    if !is_bare_weight_activation_product {
        return Err(TensorError::NotLowerable {
            node: resolved.node,
            reason: "packed reduce admits only W\u{b7}a with a materialized a",
        });
    }
    operands
        .iter()
        .map(|(node, _, _)| *node)
        .find(|node| *node != weight_node)
        .ok_or(TensorError::NotLowerable {
            node: resolved.node,
            reason: "quantized matmul reduce has no activation operand",
        })
}

/// [`run_reduce`]'s quantized-weight branch: `resolved` is the fused
/// `Reduce(Elementwise(Multiply))` matmul shape, `weight_node` one of its two
/// operands, packed `Q4_K` bytes rather than a bound `f32` buffer. The other
/// operand is the plain `f32` activation, already sitting in `buffers` like
/// any other node — read straight out of the same table [`run_reduce`]'s f32
/// path uses, no second buffer convention for it.
///
/// [`matmul_q4k_f32`] itself only knows one activation vector times one
/// weight matrix — batch-1. A real forward pass batches every sequence
/// position through the same weight in one call (`mistral_forward_program`'s
/// `wq` node alone folds `s`, `h`, and `d` together into one packed-row
/// dimension: `"ihd->shdi"` broadcasts the same `[s, i]` activation across
/// every head, so the physical weight row a given `(h, d)` pair needs is
/// `h * head_dim + d`, exactly GGUF's own on-disk row order for a
/// `[embedding_in, embedding_out]` projection reinterpreted as heads x
/// head_dim). Rather than re-deriving that per-op axis grouping here, `k`
/// (the contraction width) and `rows` (the packed weight's own row count)
/// are both derived from data already at hand — `k` from `resolved`'s own
/// reduced dims exactly as [`run_reduce`] computes them, `rows` from
/// `weights.len()` divided by `k`'s worth of packed bytes — so `rows` comes
/// out correct regardless of how many *program* output axes the weight's
/// flat row dimension was split into. `leading_total = output.len() / rows`
/// then folds every one of those non-reduced output axes (`s`, `h`, `d`,
/// ...) into one batch loop, one [`matmul_q4k_f32`] call per position.
// `expert_source` is this slice's own addition, pushing this already-large
// interpreter arm one argument past clippy's default threshold; splitting
// it into a params struct here would cost every existing positional call
// site (there is exactly one) more churn than the eight arguments cost a
// reader.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_reduce_quantized<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    weight_block: QuantizedBlock,
    weight_node: NodeId,
    expert_source: Option<ExpertSource<'_>>,
    session: Option<&MatmulSession<'_>>,
    exact_activations: bool,
    output: &mut [f32],
) -> Result<(), TensorError> {
    // proxima-debugger diagnostic: whole-function timer, once per matmul
    // node (the same granularity `evaluate_quantized`'s per-node-kind
    // table uses) -- localizes whether a gap between a node's total wall
    // time and the sum of matmul_rows_threaded's own timers sits inside
    // this function's position loop or outside it entirely.
    #[cfg(feature = "instrument")]
    let diag_reduce_quantized_started = instrument::read_ticks();
    // `session` only reaches a call site when its codec's own `q{4,5,6}k-int8-dot`
    // feature is on (the arms below); a build with every one of those off
    // never reads it, so bind it unconditionally here rather than let a
    // rare feature combination trip an unused-parameter warning.
    let _ = session;
    // A growable cache (`Q8_0`, see `QuantizedBlock::Q8_0`'s own doc) binds
    // a zero-length weight buffer on its very first call (`cached_len ==
    // 0`), which makes this reduce's own output axes multiply out to zero
    // elements too -- nothing to write, and no legal `rows` (weight rows /
    // contraction width) to derive from an empty buffer. A static weight
    // matmul (`Q4_K`/`Q5_K`/`Q6_K`) never binds an empty operand, so this
    // is additive for that path, not a behavior change.
    if output.is_empty() {
        return Ok(());
    }
    let activation_node = packed_reduce_activation_operand(resolved, weight_node)?;
    let activation =
        buffers[activation_node.0 as usize]
            .as_deref()
            .ok_or(TensorError::NotLowerable {
                node: activation_node,
                reason: "quantized matmul activation operand has no bound buffer",
            })?;
    // ROW 140's own redundant-quantize hypothesis check, unbatched-path
    // twin of `build_matmul_stage_plan`'s call: this function's own
    // wide-fold arms below (`matmul_q4k_q8k_f32_impl` et al.) each quantize
    // `activation` fresh, so if two sibling matmul nodes (`ffn_gate`,
    // `ffn_up`) both reach `run_reduce_quantized` with the SAME
    // `activation_node`, this key sees two calls for one distinct node.
    #[cfg(feature = "instrument")]
    instrument::record_quantize_activation_call(activation_node);

    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        unreachable!("run_reduce_quantized is only called for a Keep::Reduce fold")
    };
    // Single-sourced from the same resolved axis structure `run_reduce`
    // reads (`resolve_reduce_axis_shape`), not a second, independent
    // derivation from raw packed-weight byte lengths -- that second
    // derivation is what let this shape drift out of step with the whole
    // reduce's own `output_axes`/`extents` on a cached-attention fold (see
    // `ReduceAxisShape`'s own doc).
    let axis_shape = resolve_reduce_axis_shape(resolved, output_axes.as_slice());
    let contraction_width: u64 = axis_shape.reduction_extents.iter().product();
    let shape_error = || TensorError::NotLowerable {
        node: resolved.node,
        reason: "quantized matmul batch shape does not evenly divide by its packed weight rows",
    };
    let shared_axis_error = || TensorError::NotLowerable {
        node: resolved.node,
        reason: "quantized matmul activation varies along an output axis its packed weight also \
                 varies along -- not a flat weight matmul this interpreter can express",
    };
    let k = usize::try_from(contraction_width).map_err(|_| shape_error())?;

    let weight_layout = resolved
        .operands()
        .iter()
        .find(|(node, _, _)| *node == weight_node)
        .map(|(_, layout, _)| layout)
        .ok_or_else(shape_error)?;
    let activation_layout = resolved
        .operands()
        .iter()
        .find(|(node, _, _)| *node == activation_node)
        .map(|(_, layout, _)| layout)
        .ok_or_else(shape_error)?;

    // Every output axis the packed weight varies over (nonzero stride) is
    // one of its own physical rows; every output axis it broadcasts over
    // (stride 0) is a batch position the same rows are reused for across —
    // `matmul_q4k_f32`'s target shape, `[rows, k] x [k] -> [rows]` called
    // once per batch position, never a byte-length division. An axis the
    // activation ALSO varies over while the weight does too cannot be
    // folded into either bucket: the loop below dots ONE activation vector
    // against every packed row, which only holds when the activation is
    // constant across whichever axes the weight's own rows enumerate.
    // A packed weight carrying `IndexMap::Computed` over its own leading
    // (batch) axis -- `moe_block.toml`'s `expert_w` gather -- selects one
    // whole `[rows, k]` expert slab per batch position out of an
    // `[n_experts, rows, k]` stack (`proxima-gguf/src/restack.rs`'s own
    // module doc: byte concatenation, block-aligned by construction). The
    // gathered axis itself never appears in `output_axes`/`resolved.extents`
    // at all -- only the token axis that *drives* the gather does, and that
    // axis already lands in the broadcast (`stride == 0`) bucket below, same
    // as any other batch position. `leading_axes`/`leading_extents` are only
    // populated when a gather is present, so the non-gathered path (every
    // codec this crate ran before Mixtral) allocates nothing extra here.
    let weight_gather = resolved
        .operands()
        .iter()
        .find(|(node, _, _)| *node == weight_node)
        .and_then(|(_, _, gather)| gather.clone());
    let mut rows_total: u64 = 1;
    let mut leading_total_u64: u64 = 1;
    let mut leading_axes: Vec<u16> = Vec::new();
    let mut leading_extents: Vec<u64> = Vec::new();
    for axis in output_axes.as_slice() {
        let extent = resolved.extents[*axis as usize];
        if weight_layout.stride(*axis) != 0 {
            if activation_layout.stride(*axis) != 0 {
                // proxima-debugger: node35 shared-axis diagnostic -- proves
                // which axis, extent, and operand pair triggered the
                // conflict before removal per the debugging skill.
                #[cfg(feature = "instrument")]
                debug!(
                    reduce_node = resolved.node.0,
                    weight_node = weight_node.0,
                    activation_node = activation_node.0,
                    axis = *axis,
                    extent,
                    weight_gather_present = weight_gather.is_some(),
                    output_axes_count = output_axes.as_slice().len() as u32,
                    "run_reduce_quantized shared_axis_error: activation and weight both vary along the same output axis"
                );
                return Err(shared_axis_error());
            }
            rows_total *= extent;
        } else {
            leading_total_u64 *= extent;
            if weight_gather.is_some() {
                leading_axes.push(*axis);
                leading_extents.push(extent);
            }
        }
    }
    let rows = usize::try_from(rows_total).map_err(|_| shape_error())?;
    let leading_total = usize::try_from(leading_total_u64).map_err(|_| shape_error())?;

    // decision point: rows/k/leading_total right where the wide-fold-vs-
    // per-position branch below reads them -- the shape this reduce
    // actually resolved to, whichever plan runs it.
    #[cfg(feature = "instrument")]
    debug!(
        reduce_node = resolved.node.0,
        rows = rows as u64,
        k = k as u64,
        leading_total = leading_total as u64,
        weight_gather_present = weight_gather.is_some(),
        "run_reduce_quantized shape resolved"
    );

    if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some()
        && (resolved.extents.len() == 5 || activation.len() == 7 * 4096)
    {
        eprintln!(
            "gdn_projection_run reduce={} activation={} k={} rows={} leading={} extents={:?} output_axes={:?} weight_layout={weight_layout:?} activation_layout={activation_layout:?} activation_first={:?}",
            resolved.node.0,
            activation_node.0,
            k,
            rows,
            leading_total,
            resolved.extents,
            output_axes,
            &activation[..activation.len().min(8)],
        );
    }

    // `QuantizedBlock::block_layout` is the one per-codec
    // `(block_bytes, block_elements)` table -- `Q8_0`'s 32-element block is
    // the odd one out (the growable key/value context cache's rows are
    // `HEAD_DIM / 2` wide, too narrow for a 256-element K-quant super-block
    // to divide evenly without straddling more than one cached position),
    // which is exactly why that table is keyed per codec rather than one
    // shared constant.
    if let QuantizedBlock::Int32(_) = weight_block {
        unreachable!("integer index blocks never enter quantized matmul")
    }
    let weights = weight_block.packed_bytes().ok_or_else(shape_error)?;
    let (block_bytes, block_elements) = weight_block.block_layout().ok_or_else(shape_error)?;
    if k == 0 || rows == 0 || !weights.len().is_multiple_of(block_bytes) {
        return Err(shape_error());
    }
    // A gathered weight packs `n_experts` (`weight_gather.extent`) whole
    // `[rows, k]` slabs back to back -- `per_expert_bytes` below is the same
    // byte-concatenation arithmetic `proxima-gguf::restack::plan_stack`
    // already validated at load time (block-aligned by construction, per
    // that module's own doc), re-derived here rather than trusted blind so
    // a stacked buffer that does NOT evenly divide by `rows * k` blocks is a
    // typed `NotLowerable`, never a silent wrong-expert read. The
    // non-gathered check below (`total_weight_elements != rows * k`) stays
    // byte-for-byte what it always was.
    let per_expert_bytes = if weight_gather.is_some() {
        let per_expert_elements = rows.checked_mul(k).ok_or_else(shape_error)?;
        if per_expert_elements == 0 || !per_expert_elements.is_multiple_of(block_elements) {
            return Err(shape_error());
        }
        (per_expert_elements / block_elements) * block_bytes
    } else {
        0
    };
    if let Some(gather) = weight_gather.as_ref() {
        let expert_count = usize::try_from(gather.extent).map_err(|_| shape_error())?;
        let expected_total_bytes = per_expert_bytes
            .checked_mul(expert_count)
            .ok_or_else(shape_error)?;
        // A source-backed gather deliberately has no monolithic stack in the
        // segment's bindings. Each position resolves its own entry below, so
        // validating the absent stack length here would reject the exact
        // residency path this reducer is meant to execute.
        if expert_source.is_none() && weights.len() != expected_total_bytes {
            return Err(shape_error());
        }
    } else {
        // The packed buffer's own byte length must hold exactly the
        // structurally-derived `rows * k` elements -- still validated, just no
        // longer the SOURCE `rows` is derived from.
        let total_weight_elements = (weights.len() / block_bytes) * block_elements;
        if total_weight_elements != rows * k {
            return Err(shape_error());
        }
    }
    if output.len() != leading_total * rows || activation.len() != leading_total * k {
        return Err(shape_error());
    }

    #[cfg(feature = "instrument")]
    counter!(instrument::MATMUL_REDUCE_QUANTIZED_CALLS, 1);

    // `Q4_K` (216 of 225 matmul weights in a real forward, 40.37e9 of
    // 42.66e9 macs) folds every position into one `matmul_q4k_q8k_f32_impl`
    // call so each weight row's bytes are streamed once and its dot reused
    // across `leading_total` positions, instead of the per-position loop
    // below re-streaming the whole weight matrix once per position.
    // `Q5_K`/`Q6_K` (9 of 225 weights) use the identically-shaped
    // `matmul_q5k_q8k_f32_impl`/`matmul_q6k_q8k_f32_impl` wide calls below.
    // A gathered weight varies its whole slab per position -- the wide fold
    // below streams `weights` once and reuses it across every position in
    // `leading_total`, which only holds when every position dots against the
    // SAME weight. Skipping it here (rather than teaching it a per-position
    // slab swap) routes a gathered node into the per-position loop below,
    // which resolves the gather itself.
    #[cfg(feature = "q4k-int8-dot")]
    if leading_total == 1
        && !exact_activations
        && weight_gather.is_none()
        && let QuantizedBlock::Packed { codec: Codec::Q4K, bytes: _ } = weight_block
    {
        #[cfg(feature = "instrument")]
        let diag_call_started = instrument::read_ticks();
        let wide = matmul_q4k_q8k_f32_impl(weights, rows, activation, leading_total, session)?;
        #[cfg(feature = "instrument")]
        {
            let diag_call_ticks = instrument::elapsed_ticks(diag_call_started);
            let diag_call_macs = (rows as u64) * (k as u64) * (leading_total as u64);
            counter!(instrument::MATMUL_Q4K_MACS, diag_call_macs);
            counter!(instrument::MATMUL_Q4K_CALL_TICKS, diag_call_ticks);
            instrument::record_q4k_shape_call(
                rows as u64,
                (k as u64) * leading_total as u64,
                diag_call_macs,
                diag_call_ticks,
            );
        }
        // `wide` is row-major `[row][position]` — `matmul_rows_threaded`'s
        // natural shape, weight row as the parallel axis. `output` here is
        // position-major `[position][row]` (the layout the per-position
        // loop below writes, and what downstream consumes unchanged) — this
        // copy is the transpose back, `O(rows * leading_total)`, far
        // cheaper than the weight stream it replaces.
        #[cfg(feature = "instrument")]
        let diag_transpose_started = instrument::read_ticks();
        transpose_wide_to_output(&wide, rows, leading_total, session, output)?;
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_Q4K_TRANSPOSE_TICKS,
            instrument::elapsed_ticks(diag_transpose_started)
        );
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_REDUCE_QUANTIZED_TICKS,
            instrument::elapsed_ticks(diag_reduce_quantized_started)
        );
        return Ok(());
    }

    // `Q5_K`'s wide-fold arm -- same mechanism as the `Q4_K` arm above,
    // `matmul_q5k_q8k_f32_impl` in place of `matmul_q4k_q8k_f32_impl`. No
    // per-codec transpose-tick counter exists for `Q5_K` (only
    // `MATMUL_Q4K_TRANSPOSE_TICKS` does); the transpose's cost is still
    // captured inside `MATMUL_REDUCE_QUANTIZED_TICKS` below, which is not
    // codec-specific. Gated the same way the `Q4_K` arm above is -- a
    // gathered weight cannot use this single-flat-matrix wide fold.
    #[cfg(feature = "q5k-int8-dot")]
    if !exact_activations
        && weight_gather.is_none()
        && let QuantizedBlock::Packed { codec: Codec::Q5K, bytes: _ } = weight_block
    {
        #[cfg(feature = "instrument")]
        let diag_call_started = instrument::read_ticks();
        let wide = matmul_q5k_q8k_f32_impl(weights, rows, activation, leading_total, session)?;
        #[cfg(feature = "instrument")]
        {
            let diag_call_ticks = instrument::elapsed_ticks(diag_call_started);
            let diag_call_macs = (rows as u64) * (k as u64) * (leading_total as u64);
            counter!(instrument::MATMUL_Q5K_MACS, diag_call_macs);
            counter!(instrument::MATMUL_Q5K_CALL_TICKS, diag_call_ticks);
        }
        transpose_wide_to_output(&wide, rows, leading_total, session, output)?;
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_REDUCE_QUANTIZED_TICKS,
            instrument::elapsed_ticks(diag_reduce_quantized_started)
        );
        return Ok(());
    }

    // `Q6_K`'s wide-fold arm -- same mechanism as the `Q4_K` arm above,
    // `matmul_q6k_q8k_f32_impl` in place of `matmul_q4k_q8k_f32_impl`. Gated
    // the same way the `Q4_K` arm above is.
    #[cfg(feature = "q6k-int8-dot")]
    if !exact_activations
        && weight_gather.is_none()
        && let QuantizedBlock::Packed { codec: Codec::Q6K, bytes: _ } = weight_block
    {
        #[cfg(feature = "instrument")]
        let diag_call_started = instrument::read_ticks();
        let wide = matmul_q6k_q8k_f32_impl(weights, rows, activation, leading_total, session)?;
        #[cfg(feature = "instrument")]
        {
            let diag_call_ticks = instrument::elapsed_ticks(diag_call_started);
            let diag_call_macs = (rows as u64) * (k as u64) * (leading_total as u64);
            counter!(instrument::MATMUL_Q6K_MACS, diag_call_macs);
            counter!(instrument::MATMUL_Q6K_CALL_TICKS, diag_call_ticks);
        }
        transpose_wide_to_output(&wide, rows, leading_total, session, output)?;
        #[cfg(feature = "instrument")]
        counter!(
            instrument::MATMUL_REDUCE_QUANTIZED_TICKS,
            instrument::elapsed_ticks(diag_reduce_quantized_started)
        );
        return Ok(());
    }

    // A gathered weight resolves one expert slab out of the stacked buffer
    // per position, using the same coordinate -> `Layout::offset_of`
    // machinery `fill_gather_cursors` uses for the dense f32 gather path
    // (`fill_gather_cursors`/`GatherCursor`, this module) -- reused here as
    // a byte-offset computation layered on the packed weight's byte buffer,
    // not a second index-resolution mechanism. `leading_coordinate`/
    // `full_coordinate` stay empty `Vec`s (no allocation) on the
    // non-gathered path.
    let mut leading_coordinate = if weight_gather.is_some() {
        vec![0u64; leading_axes.len()]
    } else {
        Vec::new()
    };
    let mut full_coordinate = if weight_gather.is_some() {
        vec![0u64; resolved.extents.len()]
    } else {
        Vec::new()
    };

    #[cfg(feature = "instrument")]
    counter!(instrument::MATMUL_POSITION_LOOP_ITERS, leading_total as u64);
    for position in 0..leading_total {
        let activation_row = &activation[position * k..(position + 1) * k];
        let mut expert_entry: Option<ExpertEntry<'_>> = None;
        let weights: &[u8] = if let Some(gather) = weight_gather.as_ref() {
            unflatten_into(position as u64, &leading_extents, &mut leading_coordinate);
            merge_coordinates_into(
                &leading_axes,
                &leading_coordinate,
                &[],
                &[],
                &mut full_coordinate,
            );
            let index_buffer = buffer_of(buffers, gather.indices)?;
            let index_offset = usize::try_from(gather.index_layout.offset_of(&full_coordinate))
                .map_err(|_| shape_error())?;
            let raw_index = *index_buffer.get(index_offset).ok_or_else(shape_error)?;
            let expert_index = raw_index as i64;
            if expert_index < 0 || expert_index as u64 >= gather.extent {
                return Err(TensorError::GatherIndexOutOfRange {
                    node: resolved.node,
                    index: expert_index,
                    extent: gather.extent,
                });
            }
            record_expert_selection(resolved.node, expert_index as u32);
            #[cfg(feature = "instrument")]
            debug!(
                weight_node = weight_node.0,
                reduce_node = resolved.node.0,
                position = position as u64,
                expert_index = expert_index as u64,
                "proxima-debugger route: gathered weight selects expert"
            );
            // An `ExpertSource` resolves expert `expert_index`'s own entry
            // instead of slicing a fixed offset out of one contiguous
            // stack -- each entry names its own codec and bytes
            // independently (see `ExpertSource`'s own doc), so a mixed
            // table (e.g. one promoted expert at a higher precision) reads
            // correctly here without this loop knowing anything changed.
            if let Some(source) = expert_source.as_ref() {
                let rows_u32 = u32::try_from(rows).map_err(|_| shape_error())?;
                let k_u32 = u32::try_from(k).map_err(|_| shape_error())?;
                let entry = source.entry(resolved.node, expert_index as usize, rows_u32, k_u32)?;
                #[cfg(feature = "instrument")]
                debug!(
                    weight_node = weight_node.0,
                    reduce_node = resolved.node.0,
                    expert_index = expert_index as u64,
                    epoch = entry.epoch,
                    codec = ?entry.block,
                    "proxima-debugger route: resolved expert_source entry"
                );
                let bytes = entry.block.packed_bytes().ok_or_else(shape_error)?;
                expert_entry = Some(entry);
                bytes
            } else {
                let start = expert_index as usize * per_expert_bytes;
                weights
                    .get(start..start + per_expert_bytes)
                    .ok_or_else(shape_error)?
            }
        } else {
            weights
        };
        // `expert_entry` names its own codec (possibly different from
        // `weight_block`'s -- a mixed-codec `ExpertSource`), so the
        // per-position dispatch below matches on ITS discriminant when one
        // was resolved, falling back to `weight_block`'s own discriminant
        // (paired with `weights` above, already re-sliced for this
        // position) on every path that predates `ExpertSource`.
        let dispatch_block = expert_entry.map_or(weight_block, |entry| entry.block);
        // proxima-debugger diagnostic: per-position, per-codec call timer
        // plus `rows * k` mac count -- localizes whether the missing 2x is
        // inside one codec's kernel (ns/mac far above the isolated
        // single-threaded bench) or purely dispatch overhead multiplied by
        // `leading_total` separate `matmul_rows_threaded` rounds (one per
        // position, never folded into a single wider row-batch).
        #[cfg(feature = "instrument")]
        let diag_call_started = instrument::read_ticks();
        let result = match dispatch_block {
            QuantizedBlock::Int32(_) => {
                unreachable!("integer index blocks never enter quantized dispatch")
            }
            QuantizedBlock::Float32(_) => return Err(shape_error()),
            QuantizedBlock::Packed { codec: Codec::Q4K, bytes: _ } => {
                // reachable when `exact_activations` is set (the wide fold
                // above declines whenever it is) or a gathered/MoE weight
                // routes here directly; otherwise unreachable when
                // `q4k-int8-dot` is on, since the wide fold above already
                // handled and returned for every non-gathered `Q4K` weight.
                #[cfg(feature = "q4k-int8-dot")]
                {
                    if exact_activations {
                        matmul_q4k_f32(weights, rows, activation_row)?
                    } else {
                        matmul_q4k_q8k_f32_impl(weights, rows, activation_row, 1, session)?
                    }
                }
                #[cfg(not(feature = "q4k-int8-dot"))]
                {
                    matmul_q4k_f32(weights, rows, activation_row)?
                }
            }
            QuantizedBlock::Packed { codec: Codec::Q5K, bytes: _ } => {
                // same shape as the `Q4K` arm above.
                #[cfg(feature = "q5k-int8-dot")]
                {
                    if exact_activations {
                        matmul_q5k_f32(weights, rows, activation_row)?
                    } else {
                        matmul_q5k_q8k_f32_impl(weights, rows, activation_row, 1, session)?
                    }
                }
                #[cfg(not(feature = "q5k-int8-dot"))]
                {
                    matmul_q5k_f32(weights, rows, activation_row)?
                }
            }
            QuantizedBlock::Packed { codec: Codec::Q6K, bytes: _ } => {
                // same shape as the `Q4K` arm above.
                #[cfg(feature = "q6k-int8-dot")]
                {
                    if exact_activations {
                        matmul_q6k_f32(weights, rows, activation_row)?
                    } else {
                        matmul_q6k_q8k_f32_impl(weights, rows, activation_row, 1, session)?
                    }
                }
                #[cfg(not(feature = "q6k-int8-dot"))]
                {
                    matmul_q6k_f32(weights, rows, activation_row)?
                }
            }
            // No `q3k-int8-dot` kernel exists yet -- `dot_q3k_f32`'s
            // dequantize-then-fold path is `Q3_K`'s only codec path,
            // unconditionally, unlike `Q4K`/`Q5K`/`Q6K` above which fall
            // back to this same shape only when their own int8-dot feature
            // is off.
            QuantizedBlock::Packed { .. } => {
                let kernel = dispatch_block.matmul_f32_kernel().ok_or_else(shape_error)?;
                kernel(weights, rows, activation_row)?
            }
        };
        #[cfg(feature = "instrument")]
        {
            let diag_call_ticks = instrument::elapsed_ticks(diag_call_started);
            let diag_call_macs = (rows as u64) * (k as u64);
            // Matches `dispatch_block` (the codec that actually executed
            // just above), never `weight_block` (the reduce's declared
            // codec) -- a mixed-codec `ExpertSource` can dispatch a
            // different codec per position (see the comment on
            // `dispatch_block`'s own definition above), and attributing to
            // `weight_block` would count e.g. a `Q2_K` expert's macs/ticks
            // as `Q4_K` whenever the stack's OWN declared codec was `Q4_K`.
            match dispatch_block {
                QuantizedBlock::Float32(_) | QuantizedBlock::Int32(_) => {}
                QuantizedBlock::Packed { codec: Codec::Q4K, bytes: _ } => {
                    counter!(instrument::MATMUL_Q4K_MACS, diag_call_macs);
                    counter!(instrument::MATMUL_Q4K_CALL_TICKS, diag_call_ticks);
                    instrument::record_q4k_shape_call(
                        rows as u64,
                        k as u64,
                        diag_call_macs,
                        diag_call_ticks,
                    );
                }
                QuantizedBlock::Packed { codec: Codec::Q5K, bytes: _ } => {
                    counter!(instrument::MATMUL_Q5K_MACS, diag_call_macs);
                    counter!(instrument::MATMUL_Q5K_CALL_TICKS, diag_call_ticks);
                }
                QuantizedBlock::Packed { codec: Codec::Q6K, bytes: _ } => {
                    counter!(instrument::MATMUL_Q6K_MACS, diag_call_macs);
                    counter!(instrument::MATMUL_Q6K_CALL_TICKS, diag_call_ticks);
                }
                QuantizedBlock::Packed { codec: Codec::Q3K, bytes: _ } => {}
                QuantizedBlock::Packed { codec: Codec::Q2K, bytes: _ } => {}
                QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: _ } => {}
                QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: _ } => {}
                QuantizedBlock::Packed { codec: Codec::Q5_1, bytes: _ } => {}
                QuantizedBlock::Packed { codec: Codec::Q5_0, bytes: _ } => {}
                QuantizedBlock::Packed { codec: Codec::Iq4Nl, bytes: _ } => {}
                QuantizedBlock::Packed { codec: Codec::Iq2Xs, bytes: _ } => {}
                QuantizedBlock::Packed { codec: Codec::Iq3Xxs, bytes: _ } => {}
                QuantizedBlock::Packed { codec: Codec::Float16, bytes: _ } | QuantizedBlock::Packed { codec: Codec::BFloat16, bytes: _ } => {}
            }
        }
        output[position * rows..(position + 1) * rows].copy_from_slice(&result);
    }
    #[cfg(feature = "instrument")]
    counter!(
        instrument::MATMUL_REDUCE_QUANTIZED_TICKS,
        instrument::elapsed_ticks(diag_reduce_quantized_started)
    );
    Ok(())
}

/// [`run_node_into`]'s entry point whenever a caller has any quantized
/// weight bound at all ([`evaluate_quantized`], the only one) — checks
/// [`quantized_operand`] and routes to [`run_reduce_quantized`] or falls
/// through to the plain [`run_reduce`] unchanged. Kept as its own function,
/// not folded into `run_reduce` itself, so `run_reduce`'s own compiled body
/// — what every other caller ([`evaluate`], [`evaluate_parallel`],
/// [`run_reduce_typed`]'s `f32` specialization) reaches through
/// `run_node_into`'s `quantized_weights: None` arm — carries none of this
/// check's machine code: measured to hold `run_reduce`'s own instruction
/// count exactly at its pre-quantization baseline (8629 lines, 40 `fmla`,
/// `sweep_gemm --release`) precisely because a caller that never binds a
/// quantized weight never reaches this function at all, not even through a
/// branch it doesn't take.
pub(super) fn run_reduce_with_quantized_weights<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
    session: Option<&MatmulSession<'_>>,
    exact_activations: bool,
    output: &mut [f32],
) -> Result<(), TensorError> {
    if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some()
        && output.len() >= 4_096
        && resolved.extents.len() == 4
    {
        eprintln!(
            "quantized_reduce_candidate node={} output_len={} extents={:?} quantized_operand={:?} exact_activations={}",
            resolved.node.0,
            output.len(),
            resolved.extents,
            quantized_operand(resolved, quantized_weights).map(|node| node.0),
            exact_activations,
        );
    }
    if let Some(weight_node) = quantized_operand(resolved, quantized_weights) {
        let weight_block =
            quantized_weights
                .get(&weight_node)
                .copied()
                .ok_or(TensorError::NotLowerable {
                    node: weight_node,
                    reason: "quantized weight node has no bound byte buffer",
                })?;
        // `expert_sources` is keyed by the SAME weight node `quantized_weights`
        // uses -- a caller wiring a per-step `ExpertSource` snapshot (see
        // `ExpertSlab`/`page_expert` in `proxima-model-interop`) hands it here
        // under the gathered weight's own `NodeId`, and every other node in
        // this evaluation (nothing to gather, or no snapshot at all) reads
        // `None` exactly as it did before this slice existed.
        let expert_source = expert_sources.and_then(|sources| sources.get(&weight_node).copied());
        return run_reduce_quantized(
            resolved,
            buffers,
            weight_block,
            weight_node,
            expert_source,
            session,
            exact_activations,
            output,
        );
    }
    run_reduce(resolved, buffers, output, None)
}

/// The axis structure one `Keep::Reduce` fold's own binding already carries:
/// which of `resolved.extents`'s axes are reduced away versus kept in the
/// output, and the extents on each side of that split. Both the dense f32
/// path ([`run_reduce`]) and the quantized-weight path
/// ([`run_reduce_quantized`]) need exactly this — the only shape a
/// `Keep::Reduce` fold has — so it is derived once, here, rather than twice:
/// [`run_reduce_quantized`] used to re-derive its own `k` by dividing raw
/// packed-weight byte lengths instead of reading `output_axes` the way this
/// does, which is what let its shape drift out of step with this one
/// (`proxima-tensor` cached-attention quantized-seam fix).
pub(super) struct ReduceAxisShape {
    pub(super) reduction_dims: Vec<u16>,
    pub(super) leading_output_axes: Vec<u16>,
    pub(super) last_output_dim: Option<u16>,
    pub(super) leading_extents: Vec<u64>,
    pub(super) reduction_extents: Vec<u64>,
    pub(super) width: usize,
}

/// `leading_output_axes` elides any axis of extent 1: a size-1 axis has
/// exactly one legal coordinate, so it contributes nothing to addressing
/// (`full_coordinate`'s zero-init already leaves that slot at `0` forever —
/// every caller below builds `full_coordinate` as `vec![0u64; extents.len()]`
/// once and never writes an omitted axis's slot). Squeezing it out here,
/// once per bound op, is what lets `neon_tile_plan`/`width_tile_plan`'s
/// shared `leading_output_axes.len() == 1` gate see THROUGH a batch axis a
/// lowering pass never flattened into the token axis instead of counting it
/// (`docs/discipline.md` ROW 196: BGE's `MatMul([1,seq,384], [384,N])` keeps
/// `batch=1` as its own leading axis, so both AArch64 tile kernels decline
/// for 72 of 96 matmuls). Sound at ANY position among the leading axes, not
/// only the first: each survives independently in `leading_extents`
/// (`leading_product` unaffected -- multiplying by 1 is a no-op) and each
/// axis's `full_coordinate` slot is touched only through this same
/// leading-axis list, so dropping one changes no other axis's addressing.
///
/// If EVERY raw leading axis has extent 1 (BGE's own `M=1` sentence — a
/// single-token batch: `MatMul([1,1,384],[384,N])`), the filter above would
/// empty the list entirely, and both tile gates' `leading_output_axes.len()
/// == 1` check rejects `len() == 0` exactly as it rejects `len() == 2` —
/// `docs/discipline.md` ROW 200: M=1 shapes reached no tile at all, worse
/// than the M=7 row-remainder regression (ROW 199) since they fell all the
/// way to the fully-generic scalar loop. `leading_output_axes_raw[0]` is put
/// back in that one case: both tile-plan builders (`neon_tile_plan`,
/// `width_tile_plan`) and `run_reduce`'s own row-remainder macro all index
/// `leading_output_axes[0]` downstream to read that axis's real stride, so
/// an axis index must exist even though its extent is 1 and every legal
/// coordinate on it is `0`. Which raw axis is restored is immaterial when
/// more than one had extent 1 — each of those axes' `full_coordinate` slot
/// stays at its zero-init value regardless of which one is kept in the
/// list, per this function's own "sound at any position" note above — so
/// `leading_extents` for the kept axis is `[1]`, `leading_total` (product)
/// is `1`, and the tile plans address exactly the single row this shape has.
pub(super) fn resolve_reduce_axis_shape(
    resolved: &BoundOp,
    output_axes: &[u16],
) -> ReduceAxisShape {
    let reduction_dims: Vec<u16> = (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .collect();
    let (leading_output_axes_raw, last_output_dim) = output_axes_split(output_axes);
    let mut leading_output_axes: Vec<u16> = leading_output_axes_raw
        .iter()
        .copied()
        .filter(|&dim| resolved.extents[dim as usize] != 1)
        .collect();
    if leading_output_axes.is_empty()
        && let Some(&first_axis) = leading_output_axes_raw.first()
    {
        leading_output_axes.push(first_axis);
    }

    let leading_extents: Vec<u64> = leading_output_axes
        .iter()
        .map(|dim| resolved.extents[*dim as usize])
        .collect();
    let reduction_extents: Vec<u64> = reduction_dims
        .iter()
        .map(|dim| resolved.extents[*dim as usize])
        .collect();
    let width = last_output_dim.map_or(1, |dim| resolved.extents[dim as usize] as usize);

    ReduceAxisShape {
        reduction_dims,
        leading_output_axes,
        last_output_dim,
        leading_extents,
        reduction_extents,
        width,
    }
}

/// True when `operand_index`'s physical layout walks the WHOLE
/// `reduction_dims` range as one contiguous, stride-1 span once every
/// element of it is visited in `reduction_dims`'s own (outer-to-inner)
/// order — the row-major contiguity chain: the innermost dim's stride is
/// exactly 1, and each dim outward is exactly the product of every
/// extent nested inside it. `reduction_dims.len() == 1` degenerates to
/// today's original single-dim check (`stride(dims[0]) == 1`) unchanged.
/// `docs/discipline.md` ROW 148 measured this true for mnist's own first
/// FC layer (both operands: a rank-3 `[c,h,w]` activation and a
/// matching-shaped weight, neither ever reshaped through an explicit
/// flatten) and false for `Conv`'s materialized `windowed` operand, whose
/// `ci` axis sits outside the window's `oh`/`ow` axes in memory — the
/// exact mechanism `run_reduce`'s own `reduction_strides` doc cites.
pub(super) fn reduction_is_fully_flat(
    resolved: &BoundOp,
    reduction_dims: &[u16],
    operand_index: usize,
) -> bool {
    max_flat_reduction_suffix_len(resolved, reduction_dims, operand_index) == reduction_dims.len()
}

/// [`reduction_is_fully_flat`]'s own row-major contiguity chain, generalized
/// from a bool to a count: how many of `reduction_dims`'s TRAILING entries
/// (innermost-first, matching that function's own `.iter().rev()` walk) this
/// operand reads as one contiguous stride-1 span before the chain first
/// breaks. `reduction_dims.len()` degenerates to today's original
/// whole-range check unchanged. `docs/discipline.md` ROW 149 uses this to
/// find `Conv`'s own inner-contiguous-block boundary (`ky,kx`, length 2)
/// once the outer `ci` axis breaks the chain `Conv`'s materialized
/// `windowed` operand never satisfies as a whole.
pub(super) fn max_flat_reduction_suffix_len(
    resolved: &BoundOp,
    reduction_dims: &[u16],
    operand_index: usize,
) -> usize {
    let view = &resolved.operands()[operand_index].1;
    let mut expected: i64 = 1;
    let mut len = 0usize;
    for &dim in reduction_dims.iter().rev() {
        if view.stride(dim) != expected {
            break;
        }
        expected = expected.saturating_mul(resolved.extents[dim as usize] as i64);
        len += 1;
    }
    len
}

/// The dense f32 GEMM interpreter: NEON dot/width tiles then a generic
/// fallback. Never sees a quantized weight — [`run_node_into`] routes any
/// call with `quantized_weights: Some(_)` through
/// [`run_reduce_with_quantized_weights`] instead, so this function's own
/// signature and body stay exactly what they were before quantized weights
/// existed anywhere in this module.
pub(super) fn run_reduce<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
    packed_width: Option<&PackedWidthPanels>,
) -> Result<(), TensorError> {
    if std::env::var_os("PROXIMA_DEBUG_DENSE_DIGEST").is_some() && resolved.node.0 == 1265 {
        let output_axes = match &resolved.kind {
            BoundOpKind::Reduce { output_axes, .. } => output_axes.as_slice(),
            _ => &[],
        };
        eprintln!(
            "dense_reduce node={} output_len={} extents={:?} output_axes={:?} operands={:?}",
            resolved.node.0,
            output.len(),
            resolved.extents,
            output_axes,
            resolved.operands(),
        );
    }
    // only the `aarch64` width-tile block below reads this; every other
    // target's caller always passes `None`, so name it used here rather
    // than at every non-aarch64 call site.
    #[cfg(not(target_arch = "aarch64"))]
    let _ = packed_width;
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        out_layout,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_reduce is only called for a Keep::Reduce fold")
    };
    let raw = operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let shape = body_shape(body);
    let mut operand_values = vec![0.0f32; raw.len()];
    let mut step_values = vec![0.0f32; body.steps.len()];

    let ReduceAxisShape {
        reduction_dims,
        leading_output_axes,
        last_output_dim,
        leading_extents,
        reduction_extents,
        width,
    } = resolve_reduce_axis_shape(resolved, output_axes.as_slice());
    // every downstream site in this function takes `&[u16]` (`neon_tile_plan`,
    // `width_tile_plan`, `conv_gemm_tile_plan`, `merge_coordinates_into`) --
    // shadow to a borrow once here rather than touch each call site.
    let leading_output_axes: &[u16] = &leading_output_axes;

    // loop-invariant: neither `last_output_dim` nor the operand views change
    // across the whole node, so this stride table is built once instead of
    // once per (leading, reduction) coordinate pair — up to ~1e6 times for a
    // 1024^3 GEMM (`proxima-tensor/docs/discipline.md` ROW 2).
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| last_output_dim.map_or(0, |dim| view.stride(dim)))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor>> = (0..raw.len()).map(|_| None).collect();
    let mut leading_coordinate = vec![0u64; leading_extents.len()];
    let mut reduction_coordinate = vec![0u64; reduction_extents.len()];
    let mut full_coordinate = vec![0u64; resolved.extents.len()];
    let reduction_total = odometer_len(&reduction_extents);

    // A matmul with a transposed right-hand operand (ggml's own `mul_mat`
    // layout) has a bad width-dim stride on one operand but a GOOD stride on
    // the contraction dim `k` — both operands read `k` contiguously.
    // `reduction_strides` is `strides`'s sibling table for the whole
    // contraction range, computed once per bound op the same way;
    // `body_shape_is_affine_fast_path` is reused verbatim, just handed a
    // different dim's stride table (`proxima-tensor/docs/discipline.md`
    // ROW 10). A multi-dim contraction (`reduction_dims.len() > 1`, e.g. a
    // matmul-shaped fold whose weight operand was never reshaped through an
    // explicit flatten — mnist's own first FC layer reduces directly over
    // its rank-3 `[c,h,w]` activation) qualifies too, via
    // [`reduction_is_fully_flat`], PROVIDED every dim composes as one
    // contiguous row-major span for that operand: `docs/discipline.md` ROW
    // 148 measured this true for such an FC layer but false for `Conv`'s
    // own materialized `windowed` operand (its `ci` axis sits outside the
    // window's `oh`/`ow` axes in memory — see ROW 148's own row-major
    // layout trace), so `Conv` correctly stays ineligible here, unchanged.
    let reduction_strides: Vec<i64> = (0..resolved.operands().len())
        .map(|index| {
            if reduction_is_fully_flat(resolved, &reduction_dims, index) {
                1
            } else {
                i64::MAX
            }
        })
        .collect();

    // Resolved ONCE per bound op, never per element: whether every physical
    // operand the body shape actually reads is gather-free with a width-dim
    // stride of 0 or 1. When it holds, the width loop below skips
    // `gather_cursors`'s per-element `Option` check and `operand_values`'s
    // per-element copy entirely, reading straight-line out of `raw`'s own
    // subslices instead (`proxima-tensor/docs/discipline.md` ROW 3).
    // `Generic` bodies and any gathered operand fall back to the loop
    // unchanged. The width path wins the tie against the dot path below,
    // the ordering every ROW 3/10 measurement was taken under.
    let fast_path = body_shape_is_affine_fast_path(resolved, &shape, &strides);
    let reduction_fast_path = !fast_path
        && !reduction_dims.is_empty()
        && body_shape_is_affine_fast_path(resolved, &shape, &reduction_strides);

    #[cfg(feature = "instrument")]
    let mut counters = KernelCounters::default();
    #[cfg(feature = "instrument")]
    let path = if reduction_fast_path {
        Path::DotFast
    } else if fast_path {
        Path::WidthFast
    } else {
        Path::Generic
    };
    // route-census task (2026-09-01): computed once here, alongside `path`,
    // so every one of this function's four `record_reduce_path_ticks` call
    // sites can also feed `instrument::record_reduce_gemm_path_ticks` --
    // see that function's own doc for why the all-reduce `path` counters
    // alone cannot isolate the 96 `MatMul` folds from BGE's 74 small
    // single-operand reduces.
    #[cfg(feature = "instrument")]
    let is_gemm = reduce_is_gemm_shaped(resolved);
    // per-path-kind wall time (residual-profile task, 2026-08-30): started
    // once `path` is known, committed at whichever of this function's three
    // early returns (or its own tail) actually fires — see
    // `instrument::record_reduce_path_ticks`'s own doc for why this is a
    // NEW, `run_reduce`-only timer rather than a reuse of `PATH_WIDTH_FAST`/
    // `PATH_GENERIC` (those are invocation counts shared with
    // `run_elementwise_range`'s own, unrelated `Path` usage).
    #[cfg(feature = "instrument")]
    let commit_started = instrument::read_ticks();

    let seed = initial_value(*init).unwrap_or(0.0);
    let leading_total = odometer_len(&leading_extents);

    // Ported from `width-wt`: the `[k,n]`-layout twin of the dot-path tile
    // below. `reduction_fast_path` (dot tile) and `fast_path` (this tile) are
    // mutually exclusive by construction (`reduction_fast_path = !fast_path
    // && ..`), so `width_tile_plan`'s own stride gate never fires on a node
    // the dot tile already claimed — no ordering dependency between the two
    // blocks, only one of them is ever `Some`.
    //
    // whole block gated to aarch64: `try_run_width_tile` is a constant-`false`
    // stub everywhere else, so building `WidthPathContext` to hand it is dead
    // work, not just a dead type.
    #[cfg(target_arch = "aarch64")]
    {
        let width_path_context = WidthPathContext {
            resolved,
            shape: &shape,
            strides: &strides,
            reduce_op: *reduce_op,
            init: *init,
            leading_output_axes,
            reduction_dims: &reduction_dims,
            last_output_dim,
            width,
            out_layout,
        };
        // ROW 209: Accelerate/AMX route for the width tile, behind the SAME
        // `ACCELERATE_GEMM_ENABLED` toggle ROW 188/189 already gate -- see
        // `try_run_accelerate_width_gemm`'s own doc for why this is the
        // route that actually intercepts BGE's GEMMs (`try_run_width_tile`
        // below never runs when this arm already returned). Declines fall
        // straight into the existing `try_run_width_tile` call unchanged,
        // same "try, then fall back" shape as the dot/conv routes.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if ACCELERATE_GEMM_ENABLED.load(EpilogueFuseOrdering::Relaxed)
            && let Some(plan) = width_tile_plan(&width_path_context)
        {
            if try_run_accelerate_width_gemm(&plan, &raw, output) {
                ACCELERATE_GEMM_HITS.fetch_add(1, EpilogueFuseOrdering::Relaxed);
                #[cfg(feature = "instrument")]
                {
                    let distinct_operand_elements: u64 =
                        raw.iter().map(|buffer| buffer.len() as u64).sum();
                    counters.kernel_calls += 1;
                    counters.mac_ops += leading_total * width as u64 * reduction_total;
                    counters.operand_loads += (leading_total + width as u64) * reduction_total;
                    counters.leading_iters += leading_total;
                    counters.output_writes += leading_total * width as u64;
                    counters.commit(path, distinct_operand_elements);
                    instrument::record_reduce_path_ticks(
                        path,
                        instrument::elapsed_ticks(commit_started),
                    );
                }
                return Ok(());
            }
            ACCELERATE_GEMM_DECLINED.fetch_add(1, EpilogueFuseOrdering::Relaxed);
        }
        #[cfg(feature = "instrument")]
        let width_tile_counters_before = width_tile_counters();
        #[cfg(feature = "instrument")]
        let width_tile_row_remainder_before = (
            width_tile_row_remainder_invocations(),
            width_tile_row_remainder_elements(),
        );
        let disable_width_tile_for_debug = std::env::var_os("PROXIMA_DISABLE_WIDTH_TILE").is_some()
            && resolved.extents.len() == 4
            && resolved.extents.first().copied().unwrap_or(0) > 1;
        if !disable_width_tile_for_debug
            && try_run_width_tile(&width_path_context, &raw, packed_width, output)
        {
            // the tile's own early return skips the rest of this function
            // (including the `counters.commit` call every other path reaches),
            // so this is instrument's only chance to record the node — read
            // back the invocation/fallback deltas the tile itself already
            // counted instead of re-deriving them from `leading_total`/`width`.
            #[cfg(feature = "instrument")]
            {
                let (_, invocations_after, fallback_after) = width_tile_counters();
                let (_, invocations_before, fallback_before) = width_tile_counters_before;
                let invocations_delta = invocations_after - invocations_before;
                let fallback_delta = fallback_after - fallback_before;
                let (row_remainder_invocations_after, row_remainder_elements_after) = (
                    width_tile_row_remainder_invocations(),
                    width_tile_row_remainder_elements(),
                );
                let (row_remainder_invocations_before, row_remainder_elements_before) =
                    width_tile_row_remainder_before;
                let row_remainder_invocations_delta =
                    row_remainder_invocations_after - row_remainder_invocations_before;
                let row_remainder_elements_delta =
                    row_remainder_elements_after - row_remainder_elements_before;
                let tile_cols = (WIDTH_TILE_VECS * 4) as u64;
                let tile_elements = WIDTH_TILE_ROWS as u64 * tile_cols;
                counters.kernel_calls +=
                    invocations_delta + row_remainder_invocations_delta + fallback_delta;
                counters.mac_ops += invocations_delta * tile_elements * reduction_total;
                counters.operand_loads += invocations_delta
                    * (WIDTH_TILE_ROWS + WIDTH_TILE_VECS) as u64
                    * reduction_total;
                // row-remainder tiles (`ROWS` = 2 or 1, ROW 200) contribute the
                // SAME per-call shape as the main tile, `(ROWS + WIDTH_TILE_VECS)
                // * reduction_total` operand loads, just at a narrower `ROWS` —
                // `row_remainder_elements_delta / tile_cols` recovers the EXACT
                // sum of `ROWS` across every remainder call (not an average: each
                // call's own `elements = ROWS * tile_cols`, so the division is
                // exact before the sum), letting this stay a precise identity
                // rather than an approximation from the aggregate counters alone.
                let row_remainder_rows_sum = row_remainder_elements_delta / tile_cols;
                counters.mac_ops += row_remainder_elements_delta * reduction_total;
                counters.operand_loads += (row_remainder_rows_sum
                    + row_remainder_invocations_delta * WIDTH_TILE_VECS as u64)
                    * reduction_total;
                counters.mac_ops += fallback_delta * reduction_total;
                counters.operand_loads += fallback_delta * 2 * reduction_total;
                counters.leading_iters += leading_total;
                counters.output_writes += leading_total * width as u64;
                let distinct_operand_elements: u64 =
                    raw.iter().map(|buffer| buffer.len() as u64).sum();
                counters.commit(path, distinct_operand_elements);
                let elapsed = instrument::elapsed_ticks(commit_started);
                instrument::record_reduce_path_ticks(path, elapsed);
                if is_gemm {
                    instrument::record_reduce_gemm_path_ticks(path, elapsed);
                }
            }
            return Ok(());
        }
    }

    // `Conv`'s own reduce shape (`docs/discipline.md` ROW 148/149): the
    // reduction body's two operands own DISJOINT subsets of
    // `leading_output_axes` (weight varies only with `co`, the materialized
    // `windowed` operand only with `n,oy` plus the width dim `ox`), so
    // neither `width_tile_plan` nor `neon_tile_plan` below can ever engage —
    // both require `leading_output_axes.len() == 1`, a single axis BOTH
    // operands share. Tried only when both of those already declined
    // (`fast_path`/`reduction_fast_path` both false), so this can never
    // steal a node either existing tile already claims.
    #[cfg(target_arch = "aarch64")]
    if !fast_path && !reduction_fast_path {
        let conv_context = ConvGemmContext {
            resolved,
            shape: &shape,
            reduce_op: *reduce_op,
            init: *init,
            leading_output_axes,
            reduction_dims: &reduction_dims,
            last_output_dim,
            out_layout,
        };
        if let Some(plan) = conv_gemm_tile_plan(&conv_context) {
            // ROW 189: Accelerate/AMX route for `Conv`'s own two-level-blocked
            // GEMM, behind the SAME `ACCELERATE_GEMM_ENABLED` toggle ROW 188
            // gated the flat route with -- see `try_run_accelerate_conv_gemm`'s
            // own doc for why no packing is needed. Falls through to the NEON
            // tile below on any decline (overflow, negative stride, non-zero
            // seed), same "try, then fall back" shape as the flat route above.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let conv_accelerated = ACCELERATE_GEMM_ENABLED.load(EpilogueFuseOrdering::Relaxed)
                && try_run_accelerate_conv_gemm(&plan, &raw, output);
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let conv_accelerated = false;
            if conv_accelerated {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                ACCELERATE_GEMM_HITS.fetch_add(1, EpilogueFuseOrdering::Relaxed);
            } else {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                if ACCELERATE_GEMM_ENABLED.load(EpilogueFuseOrdering::Relaxed) {
                    ACCELERATE_GEMM_DECLINED.fetch_add(1, EpilogueFuseOrdering::Relaxed);
                }
                run_conv_gemm_tile(&plan, &raw, output);
            }
            #[cfg(feature = "instrument")]
            {
                // `PATH_CONV_TILE` (via `counters.commit` below) is this
                // path's own unambiguous gate-pass signal — deliberately NOT
                // `NEON_TILE_GATE_PASSES`, which `neon_tile_plan`'s own dot
                // tile already owns and a shared assertion in this file's
                // tests reads as "the dot tile fired".
                counters.kernel_calls += (plan.m_total as u64).div_ceil(TILE_ROWS as u64)
                    * (plan.n_total as u64).div_ceil(TILE_COLS as u64)
                    * plan.outer_extent;
                counters.mac_ops += plan.m_total as u64 * plan.n_total as u64 * reduction_total;
                counters.operand_loads +=
                    (plan.m_total as u64 + plan.n_total as u64) * reduction_total;
                counters.leading_iters += plan.m_total as u64;
                counters.output_writes += plan.m_total as u64 * plan.n_total as u64;
                let distinct_operand_elements: u64 =
                    raw.iter().map(|buffer| buffer.len() as u64).sum();
                counters.commit(Path::ConvTile, distinct_operand_elements);
                let elapsed = instrument::elapsed_ticks(commit_started);
                instrument::record_reduce_path_ticks(Path::ConvTile, elapsed);
                if is_gemm {
                    instrument::record_reduce_gemm_path_ticks(Path::ConvTile, elapsed);
                }
            }
            return Ok(());
        }
    }

    // ROW 188: Accelerate/AMX route, tried before the NEON tile below on
    // the SAME `reduction_fast_path` gate `neon_tile_plan` already proves
    // gather-free and contraction-contiguous. `full_coordinate`/`running`
    // are still all-zero here -- neither `try_run_width_tile` nor
    // `conv_gemm_tile_plan` above ever receive them (both take only their
    // own context struct plus `raw`/`output`) -- so this is the same
    // "leading=0, reduction=0" base offset the NEON row-strip loop below
    // recomputes per `TILE_ROWS`-row strip, taken ONCE for the whole `m x n`
    // block instead: the entire point of routing to a BLAS call is that no
    // caller-side tiling is needed. Only `seed == 0.0` (the reduce's
    // additive identity) routes here -- a non-zero seed would need a
    // `beta=1.0` pre-fill this route does not yet pay for, so it falls
    // through to the NEON tile unchanged (documented residual, ROW 188).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if reduction_fast_path
        && seed == 0.0
        && ACCELERATE_GEMM_ENABLED.load(EpilogueFuseOrdering::Relaxed)
        && let Some(plan) = neon_tile_plan(
            resolved,
            &shape,
            *reduce_op,
            !matches!(init, ReduceInit::FirstElement),
            &reduction_strides,
            &strides,
            leading_output_axes,
        )
    {
        let out_row_stride = out_layout.stride(leading_output_axes[0]);
        let out_col_stride = last_output_dim.map_or(0, |dim| out_layout.stride(dim));
        fill_running_offsets(resolved, &full_coordinate, &mut running);
        let base_a = running[plan.index_a] as usize;
        let base_b = running[plan.index_b] as usize;
        let base_c = out_layout.offset_of(&full_coordinate) as usize;
        // SAFETY: `neon_tile_plan`'s gate proves both operands gather-free
        // with contraction stride 1 across `reduction_total` elements from
        // `base_a`/`base_b`, the same bound the NEON tile below relies on
        // for its own `raw[plan.index_a]`/`raw[plan.index_b]` reads;
        // `output` is this function's own `&mut [f32]` parameter, sized by
        // the caller to `out_layout`'s extents, so `base_c + (m-1)*ldc +
        // (n-1)` stays in-bounds whenever `out_row_stride >= 0`.
        let accelerated = out_row_stride >= 0
            && unsafe {
                try_run_accelerate_sgemm(
                    raw[plan.index_a],
                    base_a,
                    plan.row_stride_a,
                    raw[plan.index_b],
                    base_b,
                    plan.col_stride_b,
                    output,
                    base_c,
                    out_row_stride as usize,
                    leading_total as usize,
                    width,
                    reduction_total as usize,
                    out_col_stride,
                    0.0,
                    true,
                )
            };
        if accelerated {
            ACCELERATE_GEMM_HITS.fetch_add(1, EpilogueFuseOrdering::Relaxed);
            #[cfg(feature = "instrument")]
            {
                let distinct_operand_elements: u64 =
                    raw.iter().map(|buffer| buffer.len() as u64).sum();
                counters.kernel_calls += 1;
                counters.mac_ops += leading_total * width as u64 * reduction_total;
                counters.operand_loads += (leading_total + width as u64) * reduction_total;
                counters.leading_iters += leading_total;
                counters.output_writes += leading_total * width as u64;
                counters.commit(path, distinct_operand_elements);
                let elapsed = instrument::elapsed_ticks(commit_started);
                instrument::record_reduce_path_ticks(path, elapsed);
                if is_gemm {
                    instrument::record_reduce_gemm_path_ticks(path, elapsed);
                }
            }
            return Ok(());
        }
        ACCELERATE_GEMM_DECLINED.fetch_add(1, EpilogueFuseOrdering::Relaxed);
    }

    // Resolved ONCE per bound op: an explicit-NEON 6x4 microkernel for the
    // exact GEMM shape `reduction_fast_path` already isolates. Ported from
    // ggml tinyBLAS's `gemm_bloc` — see `neon_tile_plan` and
    // `gemm_tile_neon` docs for the six-condition gate and why the
    // accumulator type (not the loop shape) is what makes it fit in
    // registers (`proxima-tensor/docs/discipline.md`, attempts 1 and 2).
    #[cfg(target_arch = "aarch64")]
    let tile_plan = if reduction_fast_path {
        neon_tile_plan(
            resolved,
            &shape,
            *reduce_op,
            !matches!(init, ReduceInit::FirstElement),
            &reduction_strides,
            &strides,
            leading_output_axes,
        )
    } else {
        None
    };
    #[cfg(target_arch = "aarch64")]
    let tiled_leading_rows = tile_plan.as_ref().map_or(0, |_| {
        if leading_total >= TILE_ROWS as u64 {
            leading_total - leading_total % TILE_ROWS as u64
        } else {
            0
        }
    });
    #[cfg(target_arch = "aarch64")]
    let tiled_width_cols = tile_plan.as_ref().map_or(0, |_| width - width % TILE_COLS);

    // Set to `tiled_leading_rows + rows_remaining` below when the
    // row-remainder tile pass runs, so the untiled loop after this block
    // starts past whatever rows the remainder pass already computed.
    #[cfg(target_arch = "aarch64")]
    let mut main_loop_start = tiled_leading_rows;

    // accumulated locally across the whole bound op and committed once
    // below, never as a per-element atomic inside the tiled loops — a
    // per-element atomic would perturb the throughput it exists to measure.
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let mut neon_tile_fallback_elements = 0u64;
    // `NEON_TILE_INVOCATIONS`/`NEON_TILE_ROW_REMAINDER_*` used to be a
    // `fetch_add(1)` per tile call — ~43,520 atomics per 1024^3 GEMM (one
    // per 6x4 output tile), the same per-call-in-a-hot-loop shape the
    // per-element counters this module's docs warn about. Tallied locally
    // and committed once per bound op instead, same as the fallback count
    // right above.
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let mut neon_tile_invocations = 0u64;
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let mut neon_tile_row_remainder_invocations = 0u64;
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    let mut neon_tile_row_remainder_elements = 0u64;

    #[cfg(target_arch = "aarch64")]
    if let Some(plan) = &tile_plan {
        #[cfg(feature = "instrument")]
        NEON_TILE_GATE_PASSES.fetch_add(1, Ordering::Relaxed);
        let leading_axis = leading_output_axes[0] as usize;
        let out_stride = last_output_dim.map_or(0, |dim| out_layout.stride(dim));

        // Column-panel width: bound the inner sweep to a slice of `b` that
        // stays resident in L2 across the whole row-strip pass below,
        // instead of re-streaming all of `b` past L2 once per 6-row strip
        // (`neon_column_panel_cols`'s doc has the budget arithmetic).
        let panel_cols = neon_column_panel_cols(reduction_total, tiled_width_cols);

        let mut panel_start = 0usize;
        loop {
            let panel_end = (panel_start + panel_cols).min(tiled_width_cols);
            let mut leading_flat = 0u64;
            while leading_flat < tiled_leading_rows {
                unflatten_into(leading_flat, &leading_extents, &mut leading_coordinate);
                merge_coordinates_into(
                    leading_output_axes,
                    &leading_coordinate,
                    &[],
                    &[],
                    &mut full_coordinate,
                );
                full_coordinate[reduction_dims[0] as usize] = 0;
                if let Some(dim) = last_output_dim {
                    full_coordinate[dim as usize] = 0;
                }
                fill_running_offsets(resolved, &full_coordinate, &mut running);
                let base_a = running[plan.index_a] as usize;
                let base_b0 = running[plan.index_b] as usize;

                let mut out_prefixes = [0i64; TILE_ROWS];
                for (row, prefix) in out_prefixes.iter_mut().enumerate() {
                    full_coordinate[leading_axis] = leading_flat + row as u64;
                    *prefix = out_layout.offset_of(&full_coordinate);
                }

                let mut col = panel_start;
                while col < panel_end {
                    let base_b = base_b0 + col * plan.col_stride_b;
                    let mut tile_out = [[seed; TILE_COLS]; TILE_ROWS];
                    // `neon_tile_plan`'s gate already proved: no gathers, both
                    // contraction strides == 1, and `reduction_total` elements
                    // read contiguously from `base_a`/`base_b` on every row and
                    // column this tile visits, so every offset the kernel forms
                    // stays within the source slices.
                    unsafe {
                        gemm_tile_neon::<TILE_ROWS>(
                            KStridedTile {
                                data: raw[plan.index_a],
                                base: base_a as i64,
                                k_stride: plan.row_stride_a as i64,
                            },
                            KStridedTile {
                                data: raw[plan.index_b],
                                base: base_b as i64,
                                k_stride: plan.col_stride_b as i64,
                            },
                            reduction_total as usize,
                            &mut tile_out,
                        );
                    }
                    #[cfg(feature = "instrument")]
                    {
                        neon_tile_invocations += 1;
                        counters.kernel_calls += 1;
                        counters.mac_ops += (TILE_ROWS * TILE_COLS) as u64 * reduction_total;
                        counters.operand_loads += (TILE_ROWS + TILE_COLS) as u64 * reduction_total;
                    }
                    for (tile_row, &out_prefix) in tile_out.iter().zip(out_prefixes.iter()) {
                        for (column, &value) in tile_row.iter().enumerate() {
                            let position = out_prefix + out_stride * (col + column) as i64;
                            output[position as usize] = value;
                        }
                    }
                    col += TILE_COLS;
                }

                // the column tail (`width % TILE_COLS` leftover columns) only
                // needs computing once per row, not once per panel — run it
                // on whichever panel reaches the tiled boundary (exactly one
                // does, including the degenerate single-panel case).
                if panel_end == tiled_width_cols && tiled_width_cols < width {
                    let fold = DotFold {
                        len: reduction_total as usize,
                        init: seed,
                        seeded: true,
                    };
                    for (row, &out_prefix) in out_prefixes.iter().enumerate() {
                        full_coordinate[leading_axis] = leading_flat + row as u64;
                        if let Some(dim) = last_output_dim {
                            full_coordinate[dim as usize] = tiled_width_cols as u64;
                        }
                        fill_running_offsets(resolved, &full_coordinate, &mut running);
                        for n in tiled_width_cols..width {
                            let value = reduce_dot_fast(
                                &shape,
                                *reduce_op,
                                &raw,
                                &running,
                                &reduction_strides,
                                fold,
                            );
                            output[(out_prefix + out_stride * n as i64) as usize] = value;
                            #[cfg(feature = "instrument")]
                            {
                                neon_tile_fallback_elements += 1;
                                counters.kernel_calls += 1;
                                counters.mac_ops += reduction_total;
                                for &operand_stride in &reduction_strides {
                                    counters.operand_loads += if operand_stride == 1 {
                                        reduction_total
                                    } else {
                                        1
                                    };
                                }
                            }
                            for (offset, stride) in running.iter_mut().zip(&strides) {
                                *offset += stride;
                            }
                        }
                    }
                }

                #[cfg(feature = "instrument")]
                {
                    let mut writes = (TILE_ROWS * (panel_end - panel_start)) as u64;
                    if panel_end == tiled_width_cols && tiled_width_cols < width {
                        writes += (TILE_ROWS * (width - tiled_width_cols)) as u64;
                    }
                    counters.output_writes += writes;
                    if panel_start == 0 {
                        counters.leading_iters += TILE_ROWS as u64;
                    }
                }

                leading_flat += TILE_ROWS as u64;
            }
            if panel_end >= tiled_width_cols {
                break;
            }
            panel_start = panel_end;
        }
        // every panel-loop exit leaves the row-strip sweep at exactly
        // `tiled_leading_rows` (each panel processes the same full row
        // range); the remainder pass below picks up from there.
        let mut leading_flat = tiled_leading_rows;

        // Leftover rows after the 6-row main pass are always in `1..=5`
        // (`leading_total mod TILE_ROWS`, `TILE_ROWS == 6`). Every one of
        // those widths gets its own `gemm_tile_neon` instantiation — same
        // kernel body as the main loop above, monomorphised at the exact
        // leftover width instead of testing a single fixed threshold. Zero
        // leftover rows means the remainder pass is skipped entirely; there
        // is no scalar fallback left for any `M`.
        let rows_remaining = leading_total - tiled_leading_rows;

        // one instantiation per possible leftover width; body is identical
        // across widths so a macro avoids five hand-duplicated copies.
        macro_rules! row_remainder_tile {
            ($rows:literal) => {{
                let leading_axis = leading_output_axes[0] as usize;
                let out_stride = last_output_dim.map_or(0, |dim| out_layout.stride(dim));
                unflatten_into(leading_flat, &leading_extents, &mut leading_coordinate);
                merge_coordinates_into(
                    leading_output_axes,
                    &leading_coordinate,
                    &[],
                    &[],
                    &mut full_coordinate,
                );
                full_coordinate[reduction_dims[0] as usize] = 0;
                if let Some(dim) = last_output_dim {
                    full_coordinate[dim as usize] = 0;
                }
                fill_running_offsets(resolved, &full_coordinate, &mut running);
                let base_a = running[plan.index_a] as usize;
                let base_b0 = running[plan.index_b] as usize;

                let mut out_prefixes = [0i64; $rows];
                for (row, prefix) in out_prefixes.iter_mut().enumerate() {
                    full_coordinate[leading_axis] = leading_flat + row as u64;
                    *prefix = out_layout.offset_of(&full_coordinate);
                }

                let mut col = 0usize;
                while col < tiled_width_cols {
                    let base_b = base_b0 + col * plan.col_stride_b;
                    let mut tile_out = [[seed; TILE_COLS]; $rows];
                    // same preconditions `neon_tile_plan`'s gate already
                    // proved for the main 6-row pass; only the row count
                    // differs.
                    unsafe {
                        gemm_tile_neon::<$rows>(
                            KStridedTile {
                                data: raw[plan.index_a],
                                base: base_a as i64,
                                k_stride: plan.row_stride_a as i64,
                            },
                            KStridedTile {
                                data: raw[plan.index_b],
                                base: base_b as i64,
                                k_stride: plan.col_stride_b as i64,
                            },
                            reduction_total as usize,
                            &mut tile_out,
                        );
                    }
                    #[cfg(feature = "instrument")]
                    {
                        neon_tile_row_remainder_invocations += 1;
                        neon_tile_row_remainder_elements += ($rows * TILE_COLS) as u64;
                        counters.kernel_calls += 1;
                        counters.mac_ops += ($rows * TILE_COLS) as u64 * reduction_total;
                        counters.operand_loads += ($rows + TILE_COLS) as u64 * reduction_total;
                    }
                    for (tile_row, &out_prefix) in tile_out.iter().zip(out_prefixes.iter()) {
                        for (column, &value) in tile_row.iter().enumerate() {
                            let position = out_prefix + out_stride * (col + column) as i64;
                            output[position as usize] = value;
                        }
                    }
                    col += TILE_COLS;
                }

                if tiled_width_cols < width {
                    let fold = DotFold {
                        len: reduction_total as usize,
                        init: seed,
                        seeded: true,
                    };
                    for (row, &out_prefix) in out_prefixes.iter().enumerate() {
                        full_coordinate[leading_axis] = leading_flat + row as u64;
                        if let Some(dim) = last_output_dim {
                            full_coordinate[dim as usize] = tiled_width_cols as u64;
                        }
                        fill_running_offsets(resolved, &full_coordinate, &mut running);
                        for n in tiled_width_cols..width {
                            let value = reduce_dot_fast(
                                &shape,
                                *reduce_op,
                                &raw,
                                &running,
                                &reduction_strides,
                                fold,
                            );
                            output[(out_prefix + out_stride * n as i64) as usize] = value;
                            #[cfg(feature = "instrument")]
                            {
                                neon_tile_fallback_elements += 1;
                                counters.kernel_calls += 1;
                                counters.mac_ops += reduction_total;
                                for &operand_stride in &reduction_strides {
                                    counters.operand_loads += if operand_stride == 1 {
                                        reduction_total
                                    } else {
                                        1
                                    };
                                }
                            }
                            for (offset, stride) in running.iter_mut().zip(&strides) {
                                *offset += stride;
                            }
                        }
                    }
                }

                #[cfg(feature = "instrument")]
                {
                    counters.leading_iters += $rows as u64;
                    counters.output_writes += ($rows * width) as u64;
                }

                leading_flat += $rows as u64;
                main_loop_start = leading_flat;
            }};
        }

        match rows_remaining {
            0 => {}
            5 => row_remainder_tile!(5),
            4 => row_remainder_tile!(4),
            3 => row_remainder_tile!(3),
            2 => row_remainder_tile!(2),
            1 => row_remainder_tile!(1),
            _ => unreachable!("rows_remaining must be < TILE_ROWS (6) after the main tiled pass"),
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    let main_loop_start = 0u64;

    // guards both the loop AND the allocation below it: a bound op the
    // width-tile or NEON-tile path fully covers leaves `main_loop_start ==
    // leading_total` (measured: the 1024^3 contiguous GEMM never reaches
    // this branch at all), and this fallback loop's own accumulator —
    // hoisted once per bound op rather than once per output row, same
    // reasoning as `output`'s own hoist above — has nothing to do in that
    // case; allocating it anyway would pay for storage this loop then never
    // touches.
    if main_loop_start < leading_total {
        let mut accumulator = vec![seed; width];
        for leading_flat in main_loop_start..leading_total {
            unflatten_into(leading_flat, &leading_extents, &mut leading_coordinate);
            accumulator.fill(seed);
            let mut seeded = !matches!(init, ReduceInit::FirstElement);

            if reduction_fast_path {
                // Fold along `k` (contiguous on every operand read here) instead
                // of accumulating across the width dim `n` — one full contraction
                // per output position, in the same k=0..K sequential order the
                // generic loop below would visit, so results stay bit-identical.
                merge_coordinates_into(
                    leading_output_axes,
                    &leading_coordinate,
                    &[],
                    &[],
                    &mut full_coordinate,
                );
                full_coordinate[reduction_dims[0] as usize] = 0;
                if let Some(dim) = last_output_dim {
                    full_coordinate[dim as usize] = 0;
                }
                fill_running_offsets(resolved, &full_coordinate, &mut running);
                let fold = DotFold {
                    len: reduction_total as usize,
                    init: initial_value(*init).unwrap_or(0.0),
                    seeded,
                };
                for slot in &mut accumulator {
                    *slot = reduce_dot_fast(
                        &shape,
                        *reduce_op,
                        &raw,
                        &running,
                        &reduction_strides,
                        fold,
                    );
                    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
                    if tile_plan.is_some() {
                        neon_tile_fallback_elements += 1;
                    }
                    #[cfg(feature = "instrument")]
                    {
                        counters.kernel_calls += 1;
                        counters.mac_ops += reduction_total;
                        for &operand_stride in &reduction_strides {
                            counters.operand_loads += if operand_stride == 0 {
                                1
                            } else {
                                reduction_total
                            };
                        }
                    }
                    for (offset, stride) in running.iter_mut().zip(&strides) {
                        *offset += stride;
                    }
                }
            } else {
                for reduction_flat in 0..reduction_total {
                    unflatten_into(
                        reduction_flat,
                        &reduction_extents,
                        &mut reduction_coordinate,
                    );
                    merge_coordinates_into(
                        leading_output_axes,
                        &leading_coordinate,
                        &reduction_dims,
                        &reduction_coordinate,
                        &mut full_coordinate,
                    );
                    fill_running_offsets(resolved, &full_coordinate, &mut running);

                    if fast_path {
                        reduce_width_fast(
                            &shape,
                            *reduce_op,
                            &raw,
                            &running,
                            &strides,
                            &mut accumulator,
                            seeded,
                        );
                        #[cfg(feature = "instrument")]
                        {
                            counters.kernel_calls += 1;
                            counters.mac_ops += width as u64;
                            for &stride in &strides {
                                counters.operand_loads +=
                                    if stride == 0 { 1 } else { width as u64 };
                            }
                        }
                        seeded = true;
                        continue;
                    }

                    fill_gather_cursors(
                        resolved,
                        buffers,
                        &full_coordinate,
                        last_output_dim,
                        &mut gather_cursors,
                    )?;

                    for slot in &mut accumulator {
                        for (index, data) in raw.iter().enumerate() {
                            let mut offset = running[index];
                            if let Some(cursor) = gather_cursors[index].as_mut() {
                                offset += cursor.fetch_and_advance(resolved.node)?;
                            }
                            operand_values[index] = data[offset as usize];
                            running[index] += strides[index];
                        }
                        let value = eval_body_shape(&shape, &operand_values, &mut step_values);
                        *slot = if seeded {
                            apply_scalar_op(*reduce_op, &[*slot, value])
                        } else {
                            value
                        };
                        #[cfg(feature = "instrument")]
                        {
                            counters.kernel_calls += 1;
                            counters.mac_ops += 1;
                            counters.operand_loads += raw.len() as u64;
                        }
                    }
                    seeded = true;
                }
            }

            merge_coordinates_into(
                leading_output_axes,
                &leading_coordinate,
                &[],
                &[],
                &mut full_coordinate,
            );
            let out_prefix = out_layout.offset_of(&full_coordinate);
            let out_stride = last_output_dim.map_or(0, |dim| out_layout.stride(dim));
            for (slot, value) in accumulator.iter().enumerate() {
                output[(out_prefix + out_stride * slot as i64) as usize] = *value;
            }
            #[cfg(feature = "instrument")]
            {
                counters.leading_iters += 1;
                counters.output_writes += accumulator.len() as u64;
            }
        }
    }
    #[cfg(feature = "instrument")]
    {
        let distinct_operand_elements: u64 = raw.iter().map(|buffer| buffer.len() as u64).sum();
        counters.commit(path, distinct_operand_elements);
        let elapsed = instrument::elapsed_ticks(commit_started);
        instrument::record_reduce_path_ticks(path, elapsed);
        if is_gemm {
            instrument::record_reduce_gemm_path_ticks(path, elapsed);
        }
    }
    #[cfg(all(target_arch = "aarch64", feature = "instrument"))]
    {
        NEON_TILE_FALLBACK_ELEMENTS.fetch_add(neon_tile_fallback_elements, Ordering::Relaxed);
        NEON_TILE_INVOCATIONS.fetch_add(neon_tile_invocations, Ordering::Relaxed);
        NEON_TILE_ROW_REMAINDER_INVOCATIONS
            .fetch_add(neon_tile_row_remainder_invocations, Ordering::Relaxed);
        NEON_TILE_ROW_REMAINDER_ELEMENTS
            .fetch_add(neon_tile_row_remainder_elements, Ordering::Relaxed);
        // computed once from `width`/`tiled_width_cols`, both already in
        // scope from earlier in this call — never re-checked per iteration.
        if tile_plan.is_some() && tiled_width_cols < width {
            counter!(instrument::NEON_TILE_COLUMN_TAIL_PRESENT, 1);
        }
    }
    Ok(())
}

pub(super) fn run_scan<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        out_layout,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_scan is only called for a Keep::Scan fold")
    };
    let raw = operand_buffers(resolved, buffers)?;
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let innermost_dim = outer_extents.len() as u16;
    let body = resolved.element_body();
    let shape = body_shape(body);
    let mut operand_values = vec![0.0f32; raw.len()];
    let mut step_values = vec![0.0f32; body.steps.len()];
    // loop-invariant: see the identical hoist in `run_elementwise`
    // (`proxima-tensor/docs/discipline.md` ROW 2).
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| view.stride(innermost_dim))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor>> = (0..raw.len()).map(|_| None).collect();
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    let mut accumulator = initial_value(*init).unwrap_or(0.0);
    let mut seeded = !matches!(init, ReduceInit::FirstElement);

    // Same operand-side gate as `run_elementwise`/`run_reduce`, plus one
    // scan-specific condition: the fast path writes into a contiguous
    // `&mut [f32]` output slice, so it additionally requires the output's
    // own width-dim stride to be 1 (`proxima-tensor/docs/discipline.md` ROW 5).
    // A strided output (real but rarer) falls back to the per-element loop
    // unchanged, named rather than silently narrowed.
    let operand_fast_path = body_shape_is_affine_fast_path(resolved, &shape, &strides);

    for outer_flat in 0..odometer_len(outer_extents) {
        unflatten_into(outer_flat, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);
        let out_running = out_layout.offset_of(&outer_coordinate);
        let out_stride = out_layout.stride(innermost_dim);

        if operand_fast_path && out_stride == 1 {
            let out_base = out_running as usize;
            let out_slice = &mut output[out_base..out_base + inner_len];
            accumulator = scan_width_fast(
                &shape,
                *reduce_op,
                &raw,
                &running,
                &strides,
                out_slice,
                ScanState {
                    seeded,
                    accumulator,
                },
            );
            seeded = true;
            continue;
        }

        fill_gather_cursors(
            resolved,
            buffers,
            &outer_coordinate,
            Some(innermost_dim),
            &mut gather_cursors,
        )?;
        let mut out_running = out_running;

        for _ in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(resolved.node)?;
                }
                operand_values[index] = data[offset as usize];
                running[index] += strides[index];
            }
            let value = eval_body_shape(&shape, &operand_values, &mut step_values);
            accumulator = if seeded {
                apply_scalar_op(*reduce_op, &[accumulator, value])
            } else {
                value
            };
            seeded = true;
            output[out_running as usize] = accumulator;
            out_running += out_stride;
        }
    }
    Ok(())
}

/// A [`ComposedBody`] classified once per node, outside the per-element
/// loop, into the shape its evaluation actually needs. `Unary`/`Binary` are
/// the overwhelmingly common post-fusion case (a single [`ScalarOp`] over
/// one or two freshly-read operands — a bare elementwise op, or the product
/// step a `Reduce(Elementwise(Multiply))` fusion folds straight into the
/// accumulator) and skip `apply_body`'s per-element step loop and its
/// dynamic `StepArg` dispatch entirely. `Generic` is the fallback for a
/// deeper fused chain (multiple `BodyStep`s referencing earlier steps).
///
/// Classifying here — once, before any element is visited — is what lets
/// [`eval_body_shape`] avoid re-deciding "is this one step or several" on
/// every one of a node's iteration-space elements; profiling
/// (`proxima-tensor/docs/discipline.md` ROW 0) found that per-element redecision,
/// via an out-of-line `apply_body` call and its computed jump table, was
/// 51.9% of self-time on a 1024^3 GEMM.
pub(super) enum BodyShape<'a> {
    Unary(ScalarOp, u16),
    Binary(ScalarOp, u16, u16),
    /// The bias-corrected Adam update chain (`docs/discipline.md` ROW 179),
    /// bias correction absorbed in-line for BOTH `m` and `v` (the actual
    /// fused shape `optimizer::adam_step` builds — `recip_bias1`/
    /// `recip_bias2` are each a live, single-consumer 4-step sub-chain
    /// (`step*ln(beta) -> exp -> 1-that -> reciprocal`), not pre-materialized
    /// scalar inputs the way an EARLIER version of this detector, and
    /// ROW 176's own simplified microbench, both assumed): 16 `BodyStep`s
    /// total, detected structurally by [`detect_adam_update_roles`] on op
    /// sequence + `StepArg` wiring — never on a node's own identity or name.
    /// Carries the source [`ComposedBody`] too, purely so
    /// [`eval_body_shape`]'s slow gather fallback can still walk it through
    /// [`apply_body`] exactly like [`Generic`](Self::Generic) does; the fast
    /// dedicated kernel ([`elementwise_width_fused_adam_update`]) never
    /// touches that field.
    FusedAdamUpdate(AdamUpdateRoles, &'a ComposedBody),
    Generic(&'a ComposedBody),
}

/// The eleven physical operand slots [`BodyShape::FusedAdamUpdate`] reads,
/// named by the role each plays in the Adam update math — `m`/`v`/`param`
/// are the three full-shape, unit-stride tensors; every other field is a
/// rank-0 broadcast scalar (`step_for_bias1`/`step_for_bias2` are the SAME
/// logical training-step value, read at two separate operand slots because
/// `bind::compose_operand` freshly resolves each occurrence rather than
/// deduplicating by `NodeId` — same for `one_for_bias1`/`one_for_bias2`,
/// both the literal `1.0`). Every field is a `StepArg::Operand` index into
/// the SAME `BoundOp::operands()` slice every other `BodyShape` variant
/// already indexes into (`Unary`/`Binary`'s own `u16` fields), not a new
/// addressing scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AdamUpdateRoles {
    pub(super) param: u16,
    pub(super) learning_rate: u16,
    pub(super) m: u16,
    pub(super) one_for_bias1: u16,
    pub(super) step_for_bias1: u16,
    pub(super) ln_beta1: u16,
    pub(super) v: u16,
    pub(super) one_for_bias2: u16,
    pub(super) step_for_bias2: u16,
    pub(super) ln_beta2: u16,
    pub(super) epsilon: u16,
}

/// Structural detector for [`BodyShape::FusedAdamUpdate`] (`docs/discipline.md`
/// ROW 179): matches `body`'s own 16 [`BodyStep`]s against the exact op
/// sequence and `StepArg` wiring [`optimizer::adam_step`]'s fusion produces
/// — every check is on `step.op`/`step.args` shape alone, never on a
/// `NodeId`, a name, or which physical buffer a caller happens to bind to
/// an operand slot. `None` on any mismatch (wrong step count, wrong op at
/// any position, or a `StepArg` referencing the wrong earlier step/operand
/// role) falls through to [`BodyShape::Generic`] untouched — the same
/// conservative-precondition shape `window_copy_operand` (ROW 153/154) and
/// `StaticArena::static_nodes` (ROW 174/175) already use for a dedicated
/// fast path beside the general one.
pub(super) fn detect_adam_update_roles(body: &ComposedBody) -> Option<AdamUpdateRoles> {
    let [
        step0,
        step1,
        step2,
        step3,
        step4,
        step5,
        step6,
        step7,
        step8,
        step9,
        step10,
        step11,
        step12,
        step13,
        step14,
        step15,
    ] = body.steps.as_slice()
    else {
        return None;
    };
    let (step_for_bias1, ln_beta1) = match (step0.op, step0.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(step_for_bias1), StepArg::Operand(ln_beta1)]) => {
            (*step_for_bias1, *ln_beta1)
        }
        _ => return None,
    };
    if !matches!(
        (step1.op, step1.args.as_slice()),
        (ScalarOp::Exponential, [StepArg::Step(0)])
    ) {
        return None;
    }
    let one_for_bias1 = match (step2.op, step2.args.as_slice()) {
        (ScalarOp::Subtract, [StepArg::Operand(one_for_bias1), StepArg::Step(1)]) => *one_for_bias1,
        _ => return None,
    };
    if !matches!(
        (step3.op, step3.args.as_slice()),
        (ScalarOp::Reciprocal, [StepArg::Step(2)])
    ) {
        return None;
    }
    let m = match (step4.op, step4.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(m), StepArg::Step(3)]) => *m,
        _ => return None,
    };
    let (step_for_bias2, ln_beta2) = match (step5.op, step5.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(step_for_bias2), StepArg::Operand(ln_beta2)]) => {
            (*step_for_bias2, *ln_beta2)
        }
        _ => return None,
    };
    if !matches!(
        (step6.op, step6.args.as_slice()),
        (ScalarOp::Exponential, [StepArg::Step(5)])
    ) {
        return None;
    }
    let one_for_bias2 = match (step7.op, step7.args.as_slice()) {
        (ScalarOp::Subtract, [StepArg::Operand(one_for_bias2), StepArg::Step(6)]) => *one_for_bias2,
        _ => return None,
    };
    if !matches!(
        (step8.op, step8.args.as_slice()),
        (ScalarOp::Reciprocal, [StepArg::Step(7)])
    ) {
        return None;
    }
    let v = match (step9.op, step9.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(v), StepArg::Step(8)]) => *v,
        _ => return None,
    };
    if !matches!(
        (step10.op, step10.args.as_slice()),
        (ScalarOp::SquareRoot, [StepArg::Step(9)])
    ) {
        return None;
    }
    let epsilon = match (step11.op, step11.args.as_slice()) {
        (ScalarOp::Add, [StepArg::Step(10), StepArg::Operand(epsilon)]) => *epsilon,
        _ => return None,
    };
    if !matches!(
        (step12.op, step12.args.as_slice()),
        (ScalarOp::Reciprocal, [StepArg::Step(11)])
    ) {
        return None;
    }
    if !matches!(
        (step13.op, step13.args.as_slice()),
        (ScalarOp::Multiply, [StepArg::Step(4), StepArg::Step(12)])
    ) {
        return None;
    }
    let learning_rate = match (step14.op, step14.args.as_slice()) {
        (ScalarOp::Multiply, [StepArg::Operand(learning_rate), StepArg::Step(13)]) => {
            *learning_rate
        }
        _ => return None,
    };
    let param = match (step15.op, step15.args.as_slice()) {
        (ScalarOp::Subtract, [StepArg::Operand(param), StepArg::Step(14)]) => *param,
        _ => return None,
    };
    Some(AdamUpdateRoles {
        param,
        learning_rate,
        m,
        one_for_bias1,
        step_for_bias1,
        ln_beta1,
        v,
        one_for_bias2,
        step_for_bias2,
        ln_beta2,
        epsilon,
    })
}

pub(super) fn body_shape(body: &ComposedBody) -> BodyShape<'_> {
    if let [step] = body.steps.as_slice() {
        match step.args.as_slice() {
            [StepArg::Operand(a)] => return BodyShape::Unary(step.op, *a),
            [StepArg::Operand(a), StepArg::Operand(b)] => {
                return BodyShape::Binary(step.op, *a, *b);
            }
            _ => {}
        }
    }
    if let Some(roles) = detect_adam_update_roles(body) {
        return BodyShape::FusedAdamUpdate(roles, body);
    }
    BodyShape::Generic(body)
}

/// Evaluates one iteration step against a pre-classified [`BodyShape`].
/// `#[inline(always)]` plus a `shape` that never changes across a node's own
/// loop nest is what lets LLVM hoist the shape match itself out of the hot
/// loop (loop-invariant code motion over a value that provably doesn't
/// change), rather than re-running it every element the way a direct
/// `apply_body` call forced.
#[inline(always)]
pub(super) fn eval_body_shape(
    shape: &BodyShape,
    operand_values: &[f32],
    step_values: &mut [f32],
) -> f32 {
    match *shape {
        BodyShape::Unary(op, a) => apply_scalar_op(op, &[operand_values[a as usize]]),
        BodyShape::Binary(op, a, b) => apply_scalar_op(
            op,
            &[operand_values[a as usize], operand_values[b as usize]],
        ),
        // The gather-fallback loop never reaches the dedicated kernel (that
        // requires the affine fast path -- `fused_adam_update_is_affine_fast_path`)
        // so a `FusedAdamUpdate` here just walks its own carried `ComposedBody`
        // exactly like `Generic`, bit-identical either way.
        BodyShape::FusedAdamUpdate(_, body) | BodyShape::Generic(body) => {
            apply_body(body, operand_values, step_values)
        }
    }
}

/// True when a physical operand at `index` is gather-free and has a
/// non-negative constant width-dim stride. Negative strides are rejected
/// because every [`OperandSpan`] is built with `stride as usize`, and a
/// negative value would wrap.
pub(super) fn operand_is_affine(resolved: &BoundOp, strides: &[i64], index: u16) -> bool {
    let (_, _, gather) = &resolved.operands()[index as usize];
    gather.is_none() && strides[index as usize] >= 0
}

/// [`operand_is_affine`] narrowed to the strides [`reduce_width_fast`] and
/// `scan_width_fast` should actually take. Their strided arms are correct
/// for any stride, but correctness is not the question this gate answers.
/// Unlike [`run_elementwise`], whose alternative is the per-element
/// interpreter at 16.2 ns/element, a reduce that fails this gate falls
/// through to the contraction-dim dot path and its NEON tile — a faster
/// kernel, not a slower one. Admitting stride > 1 here stole those nodes
/// into a scalar width walk and measured `reduce_f32_dense` at 180.1 ms of
/// prefill against 81.0 ms for the same nodes on the dot path
/// (`proxima-tensor/docs/discipline.md` ROW 66). The two gates genuinely
/// disagree on which strides they accept, and the disagreement is a
/// measurement.
pub(super) fn operand_is_unit_or_broadcast(
    resolved: &BoundOp,
    strides: &[i64],
    index: u16,
) -> bool {
    operand_is_affine(resolved, strides, index) && strides[index as usize] <= 1
}

/// True when every physical operand [`BodyShape`] actually reads (one for
/// `Unary`, up to two for `Binary` — `Generic` never qualifies here) has a
/// width-dim stride of 0 or 1 ([`operand_is_unit_or_broadcast`]). Checked
/// once per bound op, never per element — the same discipline [`body_shape`]
/// already applies to the op/arity decision. Shared by [`run_reduce`] and
/// [`run_scan`], whose own straight-line arms have no `Generic` case, so
/// `Generic` staying `false` here is load-bearing, not merely conservative.
/// [`run_elementwise`]'s own `Generic` fast path is a separate, WIDER gate:
/// [`generic_body_is_affine_fast_path`].
pub(super) fn body_shape_is_affine_fast_path(
    resolved: &BoundOp,
    shape: &BodyShape,
    strides: &[i64],
) -> bool {
    match *shape {
        BodyShape::Unary(_, a) => operand_is_unit_or_broadcast(resolved, strides, a),
        BodyShape::Binary(_, a, b) => {
            operand_is_unit_or_broadcast(resolved, strides, a)
                && operand_is_unit_or_broadcast(resolved, strides, b)
        }
        // A reduce/scan body is never fused with the Adam-chain shape in
        // this crate (it is a straight-line elementwise chain, not a
        // reduce's own per-step combine) -- treated exactly like `Generic`,
        // conservatively false, so `reduce_dot_fast`/`scan_width_fast` never
        // see this variant either.
        BodyShape::FusedAdamUpdate(..) | BodyShape::Generic(_) => false,
    }
}

/// The physical operand index of a window-materialize-shaped identity copy,
/// when `run_elementwise_range`'s own block sweep (`block_dim`/`block_extent`,
/// ROW 150) is engaged — `None` otherwise. `window_materialize`
/// (`proxima-onnx/src/lower.rs`) shapes its output `[n,c,oh,ow,kh,kw]`; once
/// ROW 147's identity-multiply elimination collapses the all-ones stamp
/// away, this op's body is exactly `BodyShape::Unary(ScalarOp::Identity, _)`
/// — a bare copy from a source read whose `kw` axis is already guaranteed
/// contiguous — checked explicitly here via `strides[operand] == 1`, NOT
/// inferred from `fast_path` alone: `fast_path`'s own gate
/// (`operand_is_unit_or_broadcast`) admits stride 0 (a genuine broadcast)
/// as well as stride 1, and `MaxPool`'s `Indices` machinery
/// (`proxima-onnx/src/lower.rs`'s `coordinate_image`) hits exactly that —
/// a `window_materialize` over a value that varies only along `kh`, not
/// `kw`, composing to `Unary(Identity, _)` with the operand's OWN `kw`
/// stride at 0, not 1 (found live by `maxpool_indices_row_major_...`
/// panicking `out of range for slice of length 4` before this check was
/// added, `docs/discipline.md` ROW 154). The `kh` axis (`block_dim`)
/// sits at a regular, arbitrary-sign stride, unconstrained here. The gate
/// stays deliberately narrow on SHAPE (`Unary(Identity, _)` plus a live
/// block plus a genuinely contiguous inner read), not on axis names or a
/// `window_materialize`-specific tag: `Layout::offset_of`'s exact linearity
/// (ROW 150's own proof) makes [`window_copy_block`] correct for ANY
/// operand whose body happens to match this shape, window-materialize or
/// not (`docs/discipline.md` ROW 153's own rung-2 charter).
pub(super) fn window_copy_operand(
    shape: &BodyShape,
    fast_path: bool,
    block_extent: u64,
    strides: &[i64],
) -> Option<u16> {
    match *shape {
        BodyShape::Unary(ScalarOp::Identity, operand)
            if fast_path && block_extent > 1 && strides[operand as usize] == 1 =>
        {
            Some(operand)
        }
        _ => None,
    }
}

/// [`window_copy_operand`]'s block: `block_extent` `inner_len`-wide rows,
/// contiguous within each row, each row offset from the previous by
/// `row_stride` (any sign/magnitude — matches the per-step block loop this
/// replaces, which places no non-negativity requirement on `block_strides`
/// unlike the inner-width `strides` array). Bypasses
/// [`elementwise_width_fast`]'s per-row `BodyShape`/`ScalarOp` dispatch and
/// [`OperandSpan`] construction entirely: the shape is already known
/// constant for the whole block, so nothing is left to branch on per row —
/// each row is a plain slice-to-slice copy. An `inner_len == 3` (mnist's
/// own `kw`) hand-unrolled scalar variant was tried and measured
/// indistinguishable-to-worse than this `copy_from_slice` loop on 3 of 4
/// benched shapes (one shape's apparent win did not survive a second
/// sample — outlier noise, not signal); kept this simpler single form
/// rather than carry a second, unproven-faster code path
/// (`docs/discipline.md` ROW 154).
#[inline(always)]
pub(super) fn window_copy_block(
    source: &[f32],
    src_base: i64,
    row_stride: i64,
    block_extent: u64,
    inner_len: usize,
    out: &mut [f32],
) {
    let mut base = src_base;
    let mut out_offset = 0usize;
    for _ in 0..block_extent {
        let start = base as usize;
        out[out_offset..out_offset + inner_len].copy_from_slice(&source[start..start + inner_len]);
        out_offset += inner_len;
        base += row_stride;
    }
}

/// [`run_elementwise`]'s own eligibility gate for its `Generic` fast path
/// ([`elementwise_width_generic`]): every `StepArg::Operand` any step in
/// `body` references must be gather-free with a non-negative constant
/// stride ([`operand_is_affine`]) — ANY constant stride, not only 0 or 1.
/// A stride-2 RoPE body (`specs/rope.toml`'s `s,2*i->si`) is what this
/// width exists for: it used to fail here and fall to the per-element
/// interpreter at 16.2 ns/element, against 2.2 ns/element on this path.
pub(super) fn generic_body_is_affine_fast_path(
    resolved: &BoundOp,
    body: &ComposedBody,
    strides: &[i64],
) -> bool {
    body.steps
        .iter()
        .flat_map(|step| step.args.iter())
        .all(|arg| match arg {
            StepArg::Operand(index) => operand_is_affine(resolved, strides, *index),
            StepArg::Step(_) => true,
        })
}

/// [`run_elementwise`]'s eligibility gate for [`BodyShape::FusedAdamUpdate`]'s
/// dedicated kernel (`docs/discipline.md` ROW 179) — strictly NARROWER than
/// [`generic_body_is_affine_fast_path`] above (which admits any non-negative
/// constant stride): [`elementwise_width_fused_adam_update`] slices `m`/`v`/
/// `param` directly (`&raw[idx][base..base+width]`), so those three roles
/// must be exactly unit-stride (`strides[idx] == 1`), and reads every other
/// role as one hoisted scalar each, so those eight roles must be exactly
/// stride-0 (a genuine call-invariant broadcast, never a per-row-only
/// broadcast — the same distinction `axes_flat_chain`'s own doc, ROW 178,
/// already draws). Any role failing its own required stride (a caller
/// somehow binding a strided/gathered buffer to one of these eleven slots)
/// falls through to `BodyShape::Generic`'s existing tiled path untouched —
/// this gate, not [`detect_adam_update_roles`]'s structural match, is what
/// makes that fall-through safe.
pub(super) fn fused_adam_update_is_affine_fast_path(
    resolved: &BoundOp,
    roles: AdamUpdateRoles,
    strides: &[i64],
) -> bool {
    let is_unit_stride =
        |index: u16| operand_is_affine(resolved, strides, index) && strides[index as usize] == 1;
    let is_broadcast_scalar =
        |index: u16| operand_is_affine(resolved, strides, index) && strides[index as usize] == 0;
    is_unit_stride(roles.m)
        && is_unit_stride(roles.v)
        && is_unit_stride(roles.param)
        && is_broadcast_scalar(roles.learning_rate)
        && is_broadcast_scalar(roles.one_for_bias1)
        && is_broadcast_scalar(roles.step_for_bias1)
        && is_broadcast_scalar(roles.ln_beta1)
        && is_broadcast_scalar(roles.one_for_bias2)
        && is_broadcast_scalar(roles.step_for_bias2)
        && is_broadcast_scalar(roles.ln_beta2)
        && is_broadcast_scalar(roles.epsilon)
}

/// The width loop's straight-line fast path: reads each physical operand's
/// value for the whole width span at once (a contiguous `&[f32]` subslice
/// when its stride is 1, a single hoisted scalar read when its stride is 0),
/// with no `operand_values` scratch copy, no `gather_cursors` `Option`
/// check, and no per-element `running`/`strides` bookkeeping — `running`
/// gives each operand's width span its starting offset, and
/// [`body_shape_is_affine_fast_path`]'s precondition guarantees every
/// operand here has stride 0 or 1, so `stride == 1` is the only branch left
/// to make (once, not per element) between a slice read and a scalar
/// broadcast. Iterates `accumulator` in the same slot order the generic path
/// does, combining via the same `apply_scalar_op` calls in the same order,
/// so output is bit-identical (`proxima-tensor/docs/discipline.md` ROW 3).
/// One operand's width-span read shape for [`reduce_width_fast`]'s
/// straight-line arms: `stride == 1` reads `data[base..base+width]` as a real
/// subslice, `stride == 0` reads `data[base]` once and broadcasts it across
/// every position, and any other value walks `base, base + stride,
/// base + 2 * stride, ...` — [`operand_is_affine`] admits any non-negative
/// stride, so all three shapes reach here. A bare `contiguous: bool` used to
/// stand in for this field: it could only ever distinguish "stride 1 or not",
/// which made a stride-2 body (RoPE's `x[2*i]`/`x[2*i+1]` reads) unrepresentable
/// in every accelerated kernel and forced it onto the per-element interpreter
/// for good. Bundling the three fields keeps `reduce_width_binary` under
/// clippy's argument-count lint without reaching for `#[allow]`.
#[derive(Clone, Copy)]
pub(super) struct OperandSpan<'a> {
    pub(super) data: &'a [f32],
    pub(super) base: usize,
    pub(super) stride: usize,
}

impl OperandSpan<'_> {
    /// distinguishes a real walk from the stride-0/1 shapes the existing
    /// monomorphic arms already handle, so those arms stay untouched.
    #[inline(always)]
    pub(super) fn is_strided(self) -> bool {
        self.stride > 1
    }

    /// collapses broadcast (`position * 0 == 0`) and contiguous
    /// (`position * 1 == position`) into the same expression as any other
    /// constant stride, so the strided fallback needs no separate broadcast arm.
    #[inline(always)]
    pub(super) fn at(self, position: usize) -> f32 {
        self.data[self.base + position * self.stride]
    }
}

#[inline(always)]
pub(super) fn reduce_width_fast(
    shape: &BodyShape,
    reduce_op: ScalarOp,
    raw: &[&[f32]],
    running: &[i64],
    strides: &[i64],
    accumulator: &mut [f32],
    seeded: bool,
) {
    let span_of = |index: u16| {
        let index = index as usize;
        OperandSpan {
            data: raw[index],
            base: running[index] as usize,
            stride: strides[index] as usize,
        }
    };
    match *shape {
        BodyShape::Unary(op, a) => {
            reduce_width_unary(op, reduce_op, span_of(a), accumulator, seeded);
        }
        BodyShape::Binary(op, a, b) => {
            reduce_width_binary(op, reduce_op, span_of(a), span_of(b), accumulator, seeded);
        }
        BodyShape::FusedAdamUpdate(..) | BodyShape::Generic(_) => {
            unreachable!("fast path is never entered for a Generic or FusedAdamUpdate body shape")
        }
    }
}

#[inline(always)]
pub(super) fn combine_reduction(
    reduce_op: ScalarOp,
    previous: f32,
    value: f32,
    seeded: bool,
) -> f32 {
    if seeded {
        apply_scalar_op(reduce_op, &[previous, value])
    } else {
        value
    }
}

/// Dispatches once per call (never per element) on `op`, then on
/// `reduce_op` — but only when `reduce_op` is one of the four ops a fold
/// realistically combines with (`Add`/`Multiply`/`Maximum`/`Minimum`: sum,
/// product, max-pool, min-pool). Each of the 28 (7 unary op x 4 reduce op)
/// arms hands two concrete, non-capturing closures to
/// [`reduce_width_unary_monomorphic`] — a distinct generic instantiation
/// per pair, so the width loop inside contains the literal arithmetic
/// (`-a`, `a.sqrt()`, `acc.max(v)`, ...) inlined straight into the loop
/// body, with no runtime branch and no indirect call
/// (`proxima-tensor/docs/discipline.md` ROW 4). `seeded` is also resolved here,
/// not per element — [`reduce_width_unary_monomorphic`] branches on it
/// once, outside its loops, rather than once per element the way
/// [`combine_reduction`] used to. A `reduce_op` outside that set of four
/// (`Subtract`/`Divide`/`Greater`/`Equal` as a fold combiner — legal by
/// the type system since both have arity 2, not a real reduction any
/// current caller constructs) falls back to
/// [`reduce_width_unary_scalar_dispatch`], the ROW 3 implementation:
/// correct, not accelerated, named rather than silently narrowed away.
pub(super) fn reduce_width_unary(
    op: ScalarOp,
    reduce_op: ScalarOp,
    span: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) {
    macro_rules! unary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => reduce_width_unary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc + v,
                    span,
                    accumulator,
                    seeded,
                ),
                ScalarOp::Multiply => reduce_width_unary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc * v,
                    span,
                    accumulator,
                    seeded,
                ),
                ScalarOp::Maximum => reduce_width_unary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc.max(v),
                    span,
                    accumulator,
                    seeded,
                ),
                ScalarOp::Minimum => reduce_width_unary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc.min(v),
                    span,
                    accumulator,
                    seeded,
                ),
                _ => reduce_width_unary_scalar_dispatch(op, reduce_op, span, accumulator, seeded),
            }
        };
    }
    match op {
        ScalarOp::Identity => unary_op_arm!(|a: f32| a),
        ScalarOp::Negate => unary_op_arm!(|a: f32| -a),
        ScalarOp::Reciprocal => unary_op_arm!(|a: f32| 1.0 / a),
        ScalarOp::Exponential => unary_op_arm!(|a: f32| a.exp()),
        ScalarOp::Logarithm => unary_op_arm!(|a: f32| a.ln()),
        ScalarOp::SquareRoot => unary_op_arm!(|a: f32| a.sqrt()),
        ScalarOp::Tanh => unary_op_arm!(|a: f32| a.tanh()),
        _ => reduce_width_unary_scalar_dispatch(op, reduce_op, span, accumulator, seeded),
    }
}

/// One monomorphized instantiation per (op, reduce_op) pair `reduce_width_unary`
/// selects. `seeded` is branched on ONCE, outside both loops (not per
/// element) — the loop bodies below each contain exactly one call to `op`
/// and, in the seeded case, one call to `reduce`, both of which are
/// non-capturing closures the compiler inlines directly into the loop,
/// leaving a single concrete arithmetic operation per element. A strided
/// span (stride > 1) delegates to [`reduce_width_unary_monomorphic_strided`]
/// before either arm below runs, so the stride-0/stride-1 arms here never
/// see anything but the two shapes they were always tuned for.
#[inline(always)]
pub(super) fn reduce_width_unary_monomorphic<F, R>(
    op: F,
    reduce: R,
    span: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if span.is_strided() {
        return reduce_width_unary_monomorphic_strided(op, reduce, span, accumulator, seeded);
    }
    if span.stride == 1 {
        let slice = &span.data[span.base..span.base + accumulator.len()];
        if seeded {
            for (slot, &raw_value) in accumulator.iter_mut().zip(slice) {
                *slot = reduce(*slot, op(raw_value));
            }
        } else {
            for (slot, &raw_value) in accumulator.iter_mut().zip(slice) {
                *slot = op(raw_value);
            }
        }
    } else {
        let value = op(span.data[span.base]);
        if seeded {
            for slot in accumulator.iter_mut() {
                *slot = reduce(*slot, value);
            }
        } else {
            for slot in accumulator.iter_mut() {
                *slot = value;
            }
        }
    }
}

/// Mirrors the stride-1 arm of [`reduce_width_unary_monomorphic`] one
/// position at a time via [`OperandSpan::at`] instead of a contiguous slice
/// read, so a stride > 1 body folds in the exact same left-to-right order as
/// the stride-1 case — never routed through a reassociating multi-accumulator
/// fold, which would silently change output for this newly-widened case.
#[inline(always)]
pub(super) fn reduce_width_unary_monomorphic_strided<F, R>(
    op: F,
    reduce: R,
    span: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) where
    F: Fn(f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if seeded {
        for (position, slot) in accumulator.iter_mut().enumerate() {
            *slot = reduce(*slot, op(span.at(position)));
        }
    } else {
        for (position, slot) in accumulator.iter_mut().enumerate() {
            *slot = op(span.at(position));
        }
    }
}

/// The pre-ROW-4 (ROW 3) implementation, kept as the fallback for a
/// `reduce_op` outside {Add, Multiply, Maximum, Minimum} — same numerical
/// result as [`reduce_width_unary_monomorphic`], dispatched per element via
/// [`apply_scalar_op`]/[`combine_reduction`] instead of an inlined closure.
/// [`OperandSpan::at`] already generalizes over every stride, so this needs
/// no separate strided sibling — one loop over positions covers stride 0, 1,
/// and any wider constant stride alike.
pub(super) fn reduce_width_unary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    span: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) {
    for (position, slot) in accumulator.iter_mut().enumerate() {
        let value = apply_scalar_op(op, &[span.at(position)]);
        *slot = combine_reduction(reduce_op, *slot, value, seeded);
    }
}

/// Same discipline as [`reduce_width_unary`], for the two-operand case: 8
/// binary-arity body ops x 4 accelerated reduce ops = 32 monomorphized
/// instantiations of [`reduce_width_binary_monomorphic`], selected by one
/// nested match evaluated once per call. A `reduce_op` outside the
/// accelerated four falls back to [`reduce_width_binary_scalar_dispatch`].
pub(super) fn reduce_width_binary(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) {
    // the width-accumulating twin of `reduce_dot_binary`'s multiply-add arm.
    // For a `[k,n]`-laid-out matmul this is the inner loop: `a` is one scalar
    // at the current `(m, k)`, `b` is a contiguous row of `n` — an axpy, and
    // the single densest multiply-accumulate in the crate. `!a.is_strided()
    // && !b.is_strided()` keeps a real stride (e.g. 2) out of this block
    // explicitly — its own `(false, false)` arm below would otherwise read a
    // strided operand once and silently treat it as a broadcast.
    if FUSED_MULTIPLY_ADD
        && seeded
        && matches!((op, reduce_op), (ScalarOp::Multiply, ScalarOp::Add))
        && !a.is_strided()
        && !b.is_strided()
    {
        let width = accumulator.len();
        match (a.stride == 1, b.stride == 1) {
            (true, true) => {
                let slice_a = &a.data[a.base..a.base + width];
                let slice_b = &b.data[b.base..b.base + width];
                for ((slot, &value_a), &value_b) in accumulator.iter_mut().zip(slice_a).zip(slice_b)
                {
                    *slot = value_a.mul_add(value_b, *slot);
                }
                return;
            }
            (true, false) => {
                let slice_a = &a.data[a.base..a.base + width];
                let value_b = b.data[b.base];
                for (slot, &value_a) in accumulator.iter_mut().zip(slice_a) {
                    *slot = value_a.mul_add(value_b, *slot);
                }
                return;
            }
            (false, true) => {
                let value_a = a.data[a.base];
                let slice_b = &b.data[b.base..b.base + width];
                for (slot, &value_b) in accumulator.iter_mut().zip(slice_b) {
                    *slot = value_a.mul_add(value_b, *slot);
                }
                return;
            }
            (false, false) => {}
        }
    }
    macro_rules! binary_op_arm {
        ($f:expr) => {
            match reduce_op {
                ScalarOp::Add => reduce_width_binary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc + v,
                    a,
                    b,
                    accumulator,
                    seeded,
                ),
                ScalarOp::Multiply => reduce_width_binary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc * v,
                    a,
                    b,
                    accumulator,
                    seeded,
                ),
                ScalarOp::Maximum => reduce_width_binary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc.max(v),
                    a,
                    b,
                    accumulator,
                    seeded,
                ),
                ScalarOp::Minimum => reduce_width_binary_monomorphic(
                    $f,
                    |acc: f32, v: f32| acc.min(v),
                    a,
                    b,
                    accumulator,
                    seeded,
                ),
                _ => reduce_width_binary_scalar_dispatch(op, reduce_op, a, b, accumulator, seeded),
            }
        };
    }
    match op {
        ScalarOp::Add => binary_op_arm!(|x: f32, y: f32| x + y),
        ScalarOp::Subtract => binary_op_arm!(|x: f32, y: f32| x - y),
        ScalarOp::Multiply => binary_op_arm!(|x: f32, y: f32| x * y),
        ScalarOp::Divide => binary_op_arm!(|x: f32, y: f32| x / y),
        ScalarOp::Maximum => binary_op_arm!(|x: f32, y: f32| x.max(y)),
        ScalarOp::Minimum => binary_op_arm!(|x: f32, y: f32| x.min(y)),
        ScalarOp::Greater => binary_op_arm!(|x: f32, y: f32| f32::from(u8::from(x > y))),
        ScalarOp::Equal => {
            binary_op_arm!(|x: f32, y: f32| f32::from(u8::from((x - y).abs() == 0.0)))
        }
        _ => reduce_width_binary_scalar_dispatch(op, reduce_op, a, b, accumulator, seeded),
    }
}

#[inline(always)]
pub(super) fn reduce_width_binary_monomorphic<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if a.is_strided() || b.is_strided() {
        return reduce_width_binary_monomorphic_strided(op, reduce, a, b, accumulator, seeded);
    }
    let width = accumulator.len();
    match (a.stride == 1, b.stride == 1) {
        (true, true) => {
            let slice_a = &a.data[a.base..a.base + width];
            let slice_b = &b.data[b.base..b.base + width];
            if seeded {
                for ((slot, &value_a), &value_b) in accumulator.iter_mut().zip(slice_a).zip(slice_b)
                {
                    *slot = reduce(*slot, op(value_a, value_b));
                }
            } else {
                for ((slot, &value_a), &value_b) in accumulator.iter_mut().zip(slice_a).zip(slice_b)
                {
                    *slot = op(value_a, value_b);
                }
            }
        }
        (true, false) => {
            let slice_a = &a.data[a.base..a.base + width];
            let value_b = b.data[b.base];
            if seeded {
                for (slot, &value_a) in accumulator.iter_mut().zip(slice_a) {
                    *slot = reduce(*slot, op(value_a, value_b));
                }
            } else {
                for (slot, &value_a) in accumulator.iter_mut().zip(slice_a) {
                    *slot = op(value_a, value_b);
                }
            }
        }
        (false, true) => {
            let value_a = a.data[a.base];
            let slice_b = &b.data[b.base..b.base + width];
            if seeded {
                for (slot, &value_b) in accumulator.iter_mut().zip(slice_b) {
                    *slot = reduce(*slot, op(value_a, value_b));
                }
            } else {
                for (slot, &value_b) in accumulator.iter_mut().zip(slice_b) {
                    *slot = op(value_a, value_b);
                }
            }
        }
        (false, false) => {
            let value_a = a.data[a.base];
            let value_b = b.data[b.base];
            let value = op(value_a, value_b);
            if seeded {
                for slot in accumulator.iter_mut() {
                    *slot = reduce(*slot, value);
                }
            } else {
                for slot in accumulator.iter_mut() {
                    *slot = value;
                }
            }
        }
    }
}

/// Mirrors [`reduce_width_binary_monomorphic`]'s fold order one position at a
/// time via [`OperandSpan::at`], for the case at least one of `a`/`b` has a
/// stride > 1 — never reassociated, so output stays bit-identical to what the
/// scalar interpreter would produce for the same body.
#[inline(always)]
pub(super) fn reduce_width_binary_monomorphic_strided<F, R>(
    op: F,
    reduce: R,
    a: OperandSpan,
    b: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) where
    F: Fn(f32, f32) -> f32,
    R: Fn(f32, f32) -> f32,
{
    if seeded {
        for (position, slot) in accumulator.iter_mut().enumerate() {
            *slot = reduce(*slot, op(a.at(position), b.at(position)));
        }
    } else {
        for (position, slot) in accumulator.iter_mut().enumerate() {
            *slot = op(a.at(position), b.at(position));
        }
    }
}

/// The pre-ROW-4 (ROW 3) implementation, kept as the fallback for a
/// `reduce_op` outside {Add, Multiply, Maximum, Minimum}. [`OperandSpan::at`]
/// already generalizes over every stride, so one loop over positions covers
/// stride 0, 1, and any wider constant stride alike.
pub(super) fn reduce_width_binary_scalar_dispatch(
    op: ScalarOp,
    reduce_op: ScalarOp,
    a: OperandSpan,
    b: OperandSpan,
    accumulator: &mut [f32],
    seeded: bool,
) {
    for (position, slot) in accumulator.iter_mut().enumerate() {
        let value = apply_scalar_op(op, &[a.at(position), b.at(position)]);
        *slot = combine_reduction(reduce_op, *slot, value, seeded);
    }
}
