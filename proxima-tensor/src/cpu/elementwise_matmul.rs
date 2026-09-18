use super::*;

pub(super) fn run_elementwise_dispatch<B: Deref<Target = [f32]> + Sync>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    session: Option<&MatmulSession<'_>>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    let Some(session) = session else {
        return run_elementwise(resolved, buffers, output);
    };
    if element_count(&resolved.extents) < PARALLEL_THRESHOLD {
        return run_elementwise(resolved, buffers, output);
    }
    let workers = matmul_worker_count();
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let outer_len = odometer_len(outer_extents) as usize;
    if workers <= 1 || outer_len < 2 || inner_len == 0 {
        return run_elementwise(resolved, buffers, output);
    }

    // `row_chunk_count`'s own `MIN_MACS_PER_CHUNK` floor is tuned to a
    // matmul dot-product's per-mac cost (`matmul_rows_threaded`'s doc); an
    // elementwise op's per-element cost is a small constant number of
    // scalar ops, not a mac, so reusing that floor left almost every node
    // (median 12288 elements, `MIN_MACS_PER_CHUNK` 500,000) computing
    // exactly one chunk — no different from the sequential fallback
    // (measured: `elementwise` term unchanged at ~25.6 ms with that floor
    // applied here). `PARALLEL_THRESHOLD` above already gates whether
    // splitting is worth a round-open at all, so once past it this reuses
    // `evaluate_node_parallel`'s own chunk-count policy instead
    // (`workers * OVERSUBSCRIBE`, `evaluate_node_parallel`'s own doc),
    // capped at one row per chunk so `chunk_len` never rounds to zero.
    //
    // A per-node OR per-chunk element-count floor was tried here (three
    // variants: a flat total-element cutoff, the same cutoff scoped to
    // `Unary`/`Binary` bodies only, and a `MIN_MACS_PER_CHUNK`-shaped
    // per-chunk floor) to stop the real openchat-3.5 decode loop's small
    // elementwise nodes (4096/14336 total elements) from opening a
    // `CohortSession::run` round for work this crate's own measured
    // 0.38ns/element rate finishes before the round would even open. Every
    // variant either left decode's round count unchanged (its actual
    // splitting nodes turned out to be `Generic`-shaped, not `Unary`/
    // `Binary`, so a shape-scoped floor missed them) or measurably regressed
    // the real forward pass's (`runs_one_real_forward_pass_and_greedy_picks_
    // a_real_token`) own comparably-sized `Generic` nodes, which DO benefit
    // from splitting (`DIAG evaluate_quantized node_kind=elementwise
    // total_ms`: 20.327ms baseline vs 25.570ms flat-floor vs 23.851ms
    // per-chunk-floor, all worse). A per-chunk-sized `Generic` node in the
    // forward pass and a whole small `Generic` node in decode land on
    // IDENTICAL element counts (14336 either way), so no floor keyed on
    // element count alone can tell them apart — the real discriminator
    // (round-trip cost vs achievable parallelism for that specific node's
    // shape) was not isolated within this investigation's budget. Left
    // unchanged rather than shipped with a measured prefill regression;
    // see this landing's own log for the full measurement trail.
    let chunk_count = (workers * OVERSUBSCRIBE).min(outer_len);
    let chunk_len = outer_len.div_ceil(chunk_count);
    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = &mut *output;
    let mut outer_start = 0usize;
    while !remaining.is_empty() {
        let take = chunk_len.min(remaining.len() / inner_len);
        let (slice, rest) = remaining.split_at_mut(take * inner_len);
        remaining = rest;
        chunk_ranges.push((outer_start, slice.as_mut_ptr() as usize, slice.len()));
        outer_start += take;
    }
    if chunk_ranges.len() < 2 {
        return run_elementwise(resolved, buffers, output);
    }

    let round = ElementwiseRowRound {
        resolved,
        buffers,
        inner_len,
        chunk_ranges: &chunk_ranges,
    };
    #[cfg(feature = "instrument")]
    {
        counter!(instrument::ELEMENTWISE_COHORT_ROUNDS, 1);
    }
    let report = session.run(&round);
    if let Some(error) = report.first_error {
        return Err(error);
    }
    if report.abandoned > 0 {
        return Err(TensorError::ThreadedChunkFailed {
            chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
            reason: alloc::string::String::from(
                "cohort member panicked while running this elementwise chunk",
            ),
        });
    }
    Ok(())
}

/// [`run_elementwise_dispatch`]'s cohort dispatch shape — the same
/// relationship [`RowRound`] has to [`matmul_rows_threaded`], one round
/// over `(outer_start, chunk_address, len)` ranges of the flattened outer
/// position space, run through [`CohortSession::run`]. `resolved`/`buffers`
/// stay ordinary borrows for the round's whole lifetime, the same argument
/// [`RowRound`]'s own doc makes.
pub(super) struct ElementwiseRowRound<'round, B> {
    pub(super) resolved: &'round BoundOp,
    pub(super) buffers: &'round [Option<B>],
    pub(super) inner_len: usize,
    pub(super) chunk_ranges: &'round [(usize, usize, usize)],
}

impl<B> CohortRound<TensorError> for ElementwiseRowRound<'_, B>
where
    B: Deref<Target = [f32]> + Sync,
{
    fn chunks(&self) -> usize {
        self.chunk_ranges.len()
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        let (outer_start, slice_address, slice_len) = self.chunk_ranges[chunk.0];
        // SAFETY: unique to this chunk by construction (`split_at_mut` in
        // `run_elementwise_dispatch` before the round starts); the parent
        // `output` outlives every reconstructed slice because
        // `CohortSession::run` does not return until every member has
        // reported done.
        let chunk_output =
            unsafe { core::slice::from_raw_parts_mut(slice_address as *mut f32, slice_len) };
        let outer_end = outer_start + slice_len / self.inner_len;
        run_elementwise_range(
            self.resolved,
            self.buffers,
            outer_start,
            outer_end,
            chunk_output,
        )
    }
}

/// One cohort round holding an ORDERED sequence of parallel stages, so a
/// whole run of graph nodes pays ONE round open/close instead of one each.
///
/// This exists because the per-node round is the measured binding constraint
/// on decode (`proxima-tensor/docs/discipline.md` ROW 68): a decode
/// elementwise node holds ~33 us of work and opening a round costs ~25 us, so
/// splitting one node across the cohort measured 2x WORSE, while leaving it
/// serial measured no thread scaling at all (`reduce_f32_dense` 3.859 ms at 1
/// worker, 3.952 at 8). Neither end of that trade is acceptable; the way out
/// is to stop paying per node. ggml's CPU backend runs its whole graph on a
/// persistent team with a cheap barrier between nodes, which is the shape
/// this reproduces.
///
/// No new `prime` primitive was needed. [`CohortRound`] hands out a flat
/// chunk space off a monotonic `fetch_add` cursor (`prime/src/os/cohort.rs`
/// `cursor`), so chunk `i` is always CLAIMED before chunk `i + 1`. Laying
/// stages out consecutively in that space therefore means every chunk of
/// stage `s - 1` has an owner before any chunk of stage `s` is claimed, and a
/// member that reaches stage `s` can simply wait on a per-stage counter.
///
/// Deadlock-free by induction on stage index: a member waiting at stage `s`
/// waits only on stage `s - 1` chunks, each of which has an owner that is
/// either running it or waiting on a strictly earlier stage; the
/// lowest-stage active member is therefore always running, never waiting.
/// The counter is bumped even when a chunk FAILS, so one stage's error can
/// never hang the members behind it — the error still propagates through
/// `CohortSession`'s own report.
///
/// Requires the default all-chunks completion policy: a `FanInCompletion`
/// that stops dispatch early would strand a stage's chunks unclaimed and
/// hang the stage behind it.
///
/// Gated to `cfg(test)` until its consumer lands: the semantics below are
/// the load-bearing, easy-to-get-wrong half (claim order, barrier,
/// deadlock-freedom, error publication), so they are proven FIRST and
/// separately from the graph-walking change that will use them. Wiring the
/// executor onto this is what removes the gate.
///
/// `stage_offsets` (length `stage_count + 1`, strictly increasing,
/// `stage_offsets[0] == 0`) replaces a single uniform `chunks_per_stage`:
/// stage `s` owns chunks `stage_offsets[s]..stage_offsets[s + 1]`, so a
/// matmul-reduce stage (many row-chunks, real cross-worker parallelism) and
/// an elementwise/reduce stage (one chunk, `run_node_into`'s own serial
/// body) can share the SAME round without the narrower stage paying for
/// chunks it never needed — a uniform `chunks_per_stage` would have forced
/// every stage to either match the matmul stage's width (every one-node
/// stage now split into phantom sub-chunks with nothing to parallelize) or
/// the elementwise stage's width (every matmul stage capped at one chunk,
/// serializing the dominant-cost computation onto a single worker). Both
/// are exactly the failure `docs/discipline.md` ROW 96 measured when it
/// tried the uniform-width version of this idea against non-matmul nodes
/// only.
#[cfg(any(test, feature = "cohort-staged-graph"))]
pub(super) struct StagedRound<'round, Run> {
    pub(super) stage_offsets: &'round [usize],
    /// completed-chunk count for each stage, indexed by stage.
    pub(super) completed: &'round [AtomicUsize],
    pub(super) run_stage_chunk: Run,
}

#[cfg(any(test, feature = "cohort-staged-graph"))]
impl<Run> CohortRound<TensorError> for StagedRound<'_, Run>
where
    Run: Fn(usize, usize) -> Result<(), TensorError> + Sync,
{
    fn chunks(&self) -> usize {
        self.stage_offsets.last().copied().unwrap_or(0)
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        // `stage_offsets` is strictly increasing starting at 0, so the
        // number of offsets `<= chunk.0` is always `stage + 1` for the
        // owning stage `s` -- `partition_point`'s own contract (first index
        // whose predicate is false) hands that back directly.
        let stage = self
            .stage_offsets
            .partition_point(|&offset| offset <= chunk.0)
            - 1;
        let within_stage = chunk.0 - self.stage_offsets[stage];
        if let Some(previous) = stage.checked_sub(1) {
            let previous_len = self.stage_offsets[previous + 1] - self.stage_offsets[previous];
            while self.completed[previous].load(Ordering::Acquire) < previous_len {
                core::hint::spin_loop();
            }
        }
        let outcome = (self.run_stage_chunk)(stage, within_stage);
        self.completed[stage].fetch_add(1, Ordering::Release);
        outcome
    }
}

pub(super) fn run_elementwise<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let (outer_extents, _) = split_innermost(&resolved.extents);
    let outer_len = odometer_len(outer_extents) as usize;
    run_elementwise_range(resolved, buffers, 0, outer_len, output)
}

/// [`run_elementwise`]'s whole computation, restricted to
/// `[outer_start, outer_end)` of the flattened outer-position space —
/// `run_elementwise` itself is the `0..outer_len` case. Every outer
/// position is independent (see [`run_elementwise_dispatch`]'s doc), so
/// narrowing the range changes nothing about what any position computes,
/// only how many of them this call covers; `output` is indexed relative to
/// `outer_start` (`out_base` below), matching the disjoint sub-slice a
/// caller like [`ElementwiseRowRound`] hands in.
pub(super) fn run_elementwise_range<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    outer_start: usize,
    outer_end: usize,
    output: &mut [f32],
) -> Result<(), TensorError> {
    #[cfg(feature = "instrument")]
    let diag_setup_started = instrument::read_ticks();
    let (outer_extents, inner_len) = split_innermost(&resolved.extents);
    let innermost_dim = outer_extents.len() as u16;
    let raw = operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let shape = body_shape(body);
    let mut operand_values = vec![0.0f32; raw.len()];
    // loop-invariant: the innermost dim's stride never depends on the outer
    // coordinate, so it is computed once for the whole node, not once per
    // outer position (`proxima-tensor/docs/discipline.md` ROW 2).
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| view.stride(innermost_dim))
        .collect();
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor>> = (0..raw.len()).map(|_| None).collect();
    let mut outer_coordinate = vec![0u64; outer_extents.len()];

    // The dim immediately outside the vectorized `inner_len` width — Conv's
    // own `window_materialize` multiply (`proxima-onnx/src/lower.rs`) shapes
    // its output `[n,c,oh,ow,kh,kw]`, so `kw` alone lands in `inner_len`
    // (3 elements) and `kh` (also 3) is this dim, otherwise walked one
    // `unflatten_into`+`fill_running_offsets` call at a time same as every
    // other outer dim (`docs/discipline.md` residual-profile session,
    // 2026-08-30: measured 12.9 ns/element on this op, ~34x this crate's own
    // 0.38 ns/element monomorphic figure, entirely fixed per-call overhead
    // amortized over only 3 elements — MAC_OPS/OUTPUT_WRITES showed no slow
    // gather path engaged at all). `block_strides` stays empty (never
    // indexed) whenever `block_extent <= 1`, the common case for every
    // rank-1-outer-extents or `kh == 1` shape.
    let block_dim = if outer_extents.is_empty() {
        None
    } else {
        Some((outer_extents.len() - 1) as u16)
    };
    let block_extent = outer_extents.last().copied().unwrap_or(1);
    let block_strides: Vec<i64> = if block_extent > 1 {
        resolved
            .operands()
            .iter()
            .map(|(_, view, _)| block_dim.map_or(0, |dim| view.stride(dim)))
            .collect()
    } else {
        Vec::new()
    };

    // `Unary`/`Binary` share `run_reduce`'s own gate (ROW 3); `Generic`
    // (a fused multi-step chain) gets its own, narrower gate that only
    // `run_elementwise` acts on — every operand the body shape reads is
    // gather-free and affine with a width-dim stride of 0 or 1
    // (`proxima-tensor/docs/discipline.md` ROW 5).
    // `FusedAdamUpdate` (`docs/discipline.md` ROW 179) gets its own gate,
    // narrower than `Generic`'s: `fused_adam_update_is_affine_fast_path`
    // requires exact unit/zero strides, not `Generic`'s wider "any
    // non-negative constant stride" admission, because the dedicated kernel
    // slices `m`/`v`/`param` directly rather than walking `OperandSpan`.
    let fast_path = match shape {
        BodyShape::Generic(generic_body) => {
            generic_body_is_affine_fast_path(resolved, generic_body, &strides)
        }
        BodyShape::FusedAdamUpdate(roles, _) => {
            fused_adam_update_is_affine_fast_path(resolved, roles, &strides)
        }
        _ => body_shape_is_affine_fast_path(resolved, &shape, &strides),
    };
    // rung 2 (`docs/discipline.md` ROW 153's own charter): when the block
    // above is engaged AND the body is a bare identity copy (`window_materialize`'s
    // post-ROW-147 collapsed form), every row the block loop would otherwise
    // walk through `elementwise_width_fast`'s per-row shape/op dispatch is a
    // plain contiguous `inner_len`-wide read at a fixed row stride — computed
    // once per call, same discipline as `fast_path`/`block_strides` above.
    let window_copy_operand = window_copy_operand(&shape, fast_path, block_extent, &strides);
    // Row-flattening (`docs/discipline.md` ROW 178): `window_copy_operand`
    // already collapses ITS narrower shape (a bare identity copy, one block
    // dim) to one memcpy-shaped call per block; this is the same collapse
    // for the GENERAL case — every operand's address across the WHOLE
    // `outer_extents` odometer (every outer dim, not just the last one)
    // composing as a single contiguous stride-`strides[operand]` span,
    // reusing [`axes_flat_chain`] (ROW 148's own reduce-side helper,
    // de-gated from `aarch64`-only below since this call site is
    // architecture-generic) with `unit = strides[operand] * inner_len`: the
    // address one outer step away must land exactly one row past where the
    // current row's own width span ends, at the SAME per-element stride. A
    // stride-0 (broadcast) operand collapses `unit` to 0, which
    // `axes_flat_chain` already treats as "every nonzero-extent axis in the
    // chain must ALSO be stride 0" — a genuinely global scalar (Adam's
    // `beta1`/`beta2`/`eps`/`lr` constants, stride 0 in every dim) passes
    // this for free; a per-row-only broadcast (stride 0 in the width dim,
    // nonzero across an outer dim) correctly FAILS it, since that address
    // is a step function of the flattened index, not affine in it, and
    // cannot be expressed as one [`elementwise_width_fast`] call. Deferred
    // to `window_copy_operand`'s own narrower, already-proven-optimal path
    // (ROW 153/154) when both apply.
    let full_range_flat = fast_path && window_copy_operand.is_none() && {
        let outer_axes: Vec<u16> = (0..innermost_dim).collect();
        elementwise_rows_are_flat(resolved, &outer_axes, &strides, inner_len)
    };
    #[cfg(feature = "instrument")]
    {
        counter!(
            instrument::ELEMENTWISE_SETUP_TICKS,
            instrument::elapsed_ticks(diag_setup_started)
        );
    }
    #[cfg(feature = "instrument")]
    let diag_step_values_started = instrument::read_ticks();
    // `elementwise_width_generic` is the only reader of `step_values`
    // (`elementwise_width_fast`'s own doc: "`Unary`/`Binary` ignore it"), and
    // the slow scalar path's `eval_body_shape` matches the same way — a
    // `Unary`/`Binary` shape never touches it either. Sizing this for every
    // shape at `body.steps.len() * inner_len` paid a real
    // `inner_len`-element (4096/14336 `f32`) heap allocation per node even
    // when nothing ever read it back; only `Generic` needs the fused
    // per-step row table at all. `full_range_flat` widens the row this call
    // covers to the WHOLE `[outer_start, outer_end)` span, so the table must
    // be sized against that wider width, not `inner_len` alone — otherwise
    // `elementwise_width_generic`'s own internal `GENERIC_WIDTH_TILE`
    // chunking (ROW 175) would index past a table sized for one narrow row.
    let flat_width = (outer_end - outer_start) * inner_len;
    let mut step_values = match shape {
        BodyShape::Generic(_) => {
            let effective_width = if full_range_flat {
                flat_width
            } else {
                inner_len
            };
            vec![
                0.0f32;
                body.steps.len()
                    * if fast_path {
                        effective_width.min(GENERIC_WIDTH_TILE)
                    } else {
                        1
                    }
            ]
        }
        // The dedicated kernel (`elementwise_width_fused_adam_update`) never
        // reads `step_values` -- only the slow per-element gather fallback
        // (`eval_body_shape` -> `apply_body`, reached when `fast_path` is
        // false) needs one scalar row per step, the same shape `Generic`'s
        // own `else { 1 }` branch already sizes for.
        BodyShape::FusedAdamUpdate(..) => {
            vec![0.0f32; if fast_path { 0 } else { body.steps.len() }]
        }
        BodyShape::Unary(..) | BodyShape::Binary(..) => Vec::new(),
    };
    #[cfg(feature = "instrument")]
    {
        counter!(
            instrument::ELEMENTWISE_STEP_VALUES_TICKS,
            instrument::elapsed_ticks(diag_step_values_started)
        );
        counter!(instrument::ELEMENTWISE_RANGE_CALLS, 1);
    }
    #[cfg(feature = "instrument")]
    let diag_loop_started = instrument::read_ticks();
    #[cfg(feature = "instrument")]
    let mut counters = KernelCounters::default();
    #[cfg(feature = "instrument")]
    let path = if fast_path {
        Path::WidthFast
    } else {
        Path::Generic
    };

    let mut outer_position = outer_start;
    while outer_position < outer_end {
        unflatten_into(outer_position as u64, outer_extents, &mut outer_coordinate);
        fill_running_offsets(resolved, &outer_coordinate, &mut running);

        // The whole `[outer_start, outer_end)` range collapsed to one flat
        // span (`full_range_flat`, computed once above): a SINGLE
        // `elementwise_width_fast` call over `flat_width` elements replaces
        // what would otherwise be `outer_end - outer_start` separate
        // per-row calls — node 132's own `[784,128]` shape turns 784 calls
        // into 1 (`docs/discipline.md` ROW 178). `running` is already
        // correct for `outer_position == outer_start` (just computed
        // above); every subsequent row's own address is exactly
        // `strides[operand]` past the previous element by construction of
        // the flatten precondition, so `elementwise_width_fast` walking the
        // combined width at that SAME stride reads every row without a
        // second odometer step.
        if full_range_flat {
            elementwise_width_fast(&shape, &raw, &running, &strides, output, &mut step_values);
            #[cfg(feature = "instrument")]
            {
                let elements = output.len() as u64;
                counter!(instrument::ELEMENTWISE_FLAT_RANGE_HITS, 1);
                counter!(
                    instrument::ELEMENTWISE_FLAT_RANGE_ROWS,
                    (outer_end - outer_start) as u64
                );
                counters.leading_iters += (outer_end - outer_start) as u64;
                counters.kernel_calls += 1;
                counters.output_writes += elements;
                for &stride in &strides {
                    counters.operand_loads += if stride == 0 { 1 } else { elements };
                }
            }
            outer_position = outer_end;
            continue;
        }

        // Blocked sweep of `block_dim` (see its own doc above): only when the
        // fast width path is already engaged (so every operand here is
        // gather-free — `Layout::offset_of` is exactly linear in the
        // coordinate, `bind.rs`, so `offset_of(coord + h*e_dim) ==
        // offset_of(coord) + h*stride(dim)` is exact, not approximate), this
        // position starts a fresh sweep (`block_dim`'s own coordinate is 0),
        // and a full `block_extent`-long run still fits before `outer_end`
        // (a parallel chunk boundary mid-sweep falls through to the
        // per-position path below, same as an unaligned `outer_start`).
        if fast_path
            && block_extent > 1
            && outer_coordinate.last() == Some(&0)
            && outer_position + block_extent as usize <= outer_end
        {
            let out_base = (outer_position - outer_start) * inner_len;
            let out_slice = &mut output[out_base..out_base + block_extent as usize * inner_len];
            if let Some(operand) = window_copy_operand {
                let operand = operand as usize;
                window_copy_block(
                    raw[operand],
                    running[operand],
                    block_strides[operand],
                    block_extent,
                    inner_len,
                    out_slice,
                );
                #[cfg(feature = "instrument")]
                {
                    let elements = block_extent * inner_len as u64;
                    counters.leading_iters += block_extent;
                    counters.kernel_calls += block_extent;
                    counters.output_writes += elements;
                    counters.operand_loads += elements;
                }
            } else {
                for step in 0..block_extent {
                    let step_base = step as usize * inner_len;
                    elementwise_width_fast(
                        &shape,
                        &raw,
                        &running,
                        &strides,
                        &mut out_slice[step_base..step_base + inner_len],
                        &mut step_values,
                    );
                    #[cfg(feature = "instrument")]
                    {
                        counters.leading_iters += 1;
                        counters.kernel_calls += 1;
                        counters.output_writes += inner_len as u64;
                        for &stride in &strides {
                            counters.operand_loads +=
                                if stride == 0 { 1 } else { inner_len as u64 };
                        }
                    }
                    if step + 1 < block_extent {
                        for (slot, block_stride) in running.iter_mut().zip(&block_strides) {
                            *slot += block_stride;
                        }
                    }
                }
            }
            outer_position += block_extent as usize;
            continue;
        }

        let out_base = (outer_position - outer_start) * inner_len;
        #[cfg(feature = "instrument")]
        {
            counters.leading_iters += 1;
        }

        if fast_path {
            let out_slice = &mut output[out_base..out_base + inner_len];
            elementwise_width_fast(
                &shape,
                &raw,
                &running,
                &strides,
                out_slice,
                &mut step_values,
            );
            #[cfg(feature = "instrument")]
            {
                counters.kernel_calls += 1;
                counters.output_writes += inner_len as u64;
                for &stride in &strides {
                    counters.operand_loads += if stride == 0 { 1 } else { inner_len as u64 };
                }
            }
            outer_position += 1;
            continue;
        }

        fill_gather_cursors(
            resolved,
            buffers,
            &outer_coordinate,
            Some(innermost_dim),
            &mut gather_cursors,
        )?;

        for step in 0..inner_len {
            for (index, data) in raw.iter().enumerate() {
                let mut offset = running[index];
                if let Some(cursor) = gather_cursors[index].as_mut() {
                    offset += cursor.fetch_and_advance(resolved.node)?;
                }
                operand_values[index] = data[offset as usize];
                running[index] += strides[index];
            }
            output[out_base + step] = eval_body_shape(&shape, &operand_values, &mut step_values);
            #[cfg(feature = "instrument")]
            {
                counters.kernel_calls += 1;
                counters.output_writes += 1;
                counters.operand_loads += raw.len() as u64;
            }
        }
        outer_position += 1;
    }
    #[cfg(feature = "instrument")]
    {
        let diag_loop_ticks = instrument::elapsed_ticks(diag_loop_started);
        counter!(instrument::ELEMENTWISE_LOOP_TICKS, diag_loop_ticks);
        let distinct_operand_elements: u64 = raw.iter().map(|buffer| buffer.len() as u64).sum();
        // achieved-ns/element split by `BodyShape` (nsper task, 2026-08-21):
        // `Unary`/`Binary` is the monomorphic kernel this crate's own
        // 0.38ns/element figure (`cpu.rs:2159`) was measured against;
        // `Generic` is the fused multi-step body. Both axes read
        // `counters.output_writes`, the exact element count this call wrote
        // (identical whether `fast_path` did or didn't fire -- see the loop
        // above), never re-derived from extents.
        match shape {
            BodyShape::Generic(_) => {
                counter!(instrument::ELEMENTWISE_LOOP_TICKS_GENERIC, diag_loop_ticks);
                counter!(
                    instrument::ELEMENTWISE_ELEMENTS_GENERIC,
                    counters.output_writes
                );
                if fast_path {
                    counter!(
                        instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_FAST,
                        diag_loop_ticks
                    );
                    counter!(
                        instrument::ELEMENTWISE_ELEMENTS_GENERIC_FAST,
                        counters.output_writes
                    );
                } else {
                    counter!(
                        instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_SLOW,
                        diag_loop_ticks
                    );
                    counter!(
                        instrument::ELEMENTWISE_ELEMENTS_GENERIC_SLOW,
                        counters.output_writes
                    );
                }
            }
            BodyShape::FusedAdamUpdate(..) => {
                counter!(instrument::ELEMENTWISE_LOOP_TICKS_GENERIC, diag_loop_ticks);
                counter!(
                    instrument::ELEMENTWISE_ELEMENTS_GENERIC,
                    counters.output_writes
                );
                if fast_path {
                    counter!(
                        instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_FAST,
                        diag_loop_ticks
                    );
                    counter!(
                        instrument::ELEMENTWISE_ELEMENTS_GENERIC_FAST,
                        counters.output_writes
                    );
                    counter!(
                        instrument::ELEMENTWISE_LOOP_TICKS_FUSED_ADAM,
                        diag_loop_ticks
                    );
                    counter!(
                        instrument::ELEMENTWISE_ELEMENTS_FUSED_ADAM,
                        counters.output_writes
                    );
                    counter!(instrument::ELEMENTWISE_FUSED_ADAM_HITS, 1);
                } else {
                    counter!(
                        instrument::ELEMENTWISE_LOOP_TICKS_GENERIC_SLOW,
                        diag_loop_ticks
                    );
                    counter!(
                        instrument::ELEMENTWISE_ELEMENTS_GENERIC_SLOW,
                        counters.output_writes
                    );
                }
            }
            BodyShape::Unary(..) | BodyShape::Binary(..) => {
                counter!(
                    instrument::ELEMENTWISE_LOOP_TICKS_MONOMORPHIC,
                    diag_loop_ticks
                );
                counter!(
                    instrument::ELEMENTWISE_ELEMENTS_MONOMORPHIC,
                    counters.output_writes
                );
                if window_copy_operand.is_some() {
                    // rung 2 (ROW 153/154): same per-call constant `fast_path`
                    // already splits on, one level narrower — this call's
                    // block-aligned rows took the specialized row-segment copy,
                    // not `elementwise_width_fast`'s per-row dispatch.
                    counter!(
                        instrument::ELEMENTWISE_LOOP_TICKS_WINDOW_COPY,
                        diag_loop_ticks
                    );
                    counter!(
                        instrument::ELEMENTWISE_ELEMENTS_WINDOW_COPY,
                        counters.output_writes
                    );
                }
                if fast_path {
                    counter!(
                        instrument::ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_FAST,
                        diag_loop_ticks
                    );
                    counter!(
                        instrument::ELEMENTWISE_ELEMENTS_MONOMORPHIC_FAST,
                        counters.output_writes
                    );
                } else {
                    counter!(
                        instrument::ELEMENTWISE_LOOP_TICKS_MONOMORPHIC_SLOW,
                        diag_loop_ticks
                    );
                    counter!(
                        instrument::ELEMENTWISE_ELEMENTS_MONOMORPHIC_SLOW,
                        counters.output_writes
                    );
                }
            }
        }
        instrument::record_elementwise_call_size(counters.output_writes);
        counters.commit(path, distinct_operand_elements);
    }
    if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some()
        && outer_start == 0
        && resolved.extents.as_slice() == [7, 16, 512, 2048]
        && output.len() >= 4
    {
        eprintln!(
            "elementwise_candidate node={} extents={:?} first={:?}",
            resolved.node.0,
            resolved.extents,
            &output[..4]
        );
    }
    Ok(())
}

/// Which of `resolved`'s operands, if any, is a `Q4_K`-packed weight named
/// in `quantized_weights` — [`run_reduce`]'s own gate for routing to
/// [`matmul_q4k_f32`] instead of the f32 tile/generic paths below. Only a
/// `Keep::Reduce` fold can match: `quantized_weights` only ever names a node
/// [`reject_non_float32`] already proved (via [`is_quantized_matmul_operand`])
/// feeds exactly one such fold, so this need not re-check the shape, only
/// find which physical operand it is.
pub(super) fn quantized_operand(
    resolved: &BoundOp,
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
) -> Option<NodeId> {
    if !matches!(
        resolved.kind,
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            ..
        }
    ) {
        return None;
    }
    resolved
        .operands()
        .iter()
        .map(|(node, _, _)| *node)
        .find(|node| quantized_weights.contains_key(node))
}

/// `docs/discipline.md` ROW 202's own named residual, split by node
/// identity rather than output shape -- an earlier attempt at this
/// classifier used `output_axes`'s own trailing-axis extent (`width`,
/// [`resolve_reduce_axis_shape`]'s own field) and MEASURED wrong: BGE's
/// LayerNorm mean/variance reduce keeps `[batch, seq]` as its own
/// `output_axes` (the hidden axis is what gets reduced away), so its
/// trailing axis is `seq` -- extent 7/8/9, NOT 1 -- and the width-based
/// check misclassified every one of the 74 small reduces as GEMM-shaped
/// whenever `M > 1` (a `PROXIMA_DIAG_REDUCE_SHAPE_DUMP=1` dump against the
/// real BGE graph, `bge_epilogue_profile`, caught this before it landed).
/// The mechanism that actually separates the two populations is operand
/// count: a real matmul reads TWO distinct tensors (activation and
/// weight, or query and key, or attention-weights and value -- every one
/// of the 96 GEMM-shaped reduces in the dump carried two distinct
/// `NodeId`s), while LayerNorm's mean (`sum(X)`) reads ONE, and its
/// variance (`sum(X*X)`) reads the SAME node twice (`operand_count=2`,
/// `distinct_operands=1`) -- confirmed on all 74 small reduces in the same
/// dump (50 LayerNorm mean/variance pairs across BGE's 25 LayerNorms, plus
/// 24 softmax max/sum-over-key-dim reduces, all single-operand). Zero
/// allocation: walks `operands()`'s own slice (already resolved by
/// `bind::bind`) rather than paying [`resolve_reduce_axis_shape`]'s `Vec`s
/// -- diagnostic-only, called once per node from a `#[cfg]`-gated counter
/// site, never from the hot compute path `run_reduce` itself takes.
#[cfg(any(feature = "instrument", feature = "epilogue-profile-probe"))]
pub(super) fn reduce_is_gemm_shaped(resolved: &BoundOp) -> bool {
    let operands = resolved.operands();
    let Some((first_node, _, _)) = operands.first() else {
        return false;
    };
    operands.iter().any(|(node, _, _)| node != first_node)
}

#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) use crate::sized::MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH;

/// The `wide` (`[row][position]`) -> `output` (`[position][row]`) transpose
/// copy-back the `Q4_K`/`Q5_K`/`Q6_K` wide-fold arms of
/// [`run_reduce_quantized`] pay -- dispatched across the cohort when a
/// `session` is open and `rows * leading_total` clears
/// [`MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH`]. Splits on `position`, the same
/// outer axis [`run_elementwise_dispatch`] splits on: each position range
/// writes a contiguous, disjoint `rows`-wide slice of `output` (safe
/// [`slice::split_at_mut`], no raw pointer needed for the write side),
/// reading a strided range of `wide` (a shared `&[f32]`, never mutated).
/// Falls straight through to the plain serial loop whenever any gate fails:
/// no session, too few elements, or fewer than two position chunks to split
/// into.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) fn transpose_wide_to_output(
    wide: &[f32],
    rows: usize,
    leading_total: usize,
    session: Option<&MatmulSession<'_>>,
    output: &mut [f32],
) -> Result<(), TensorError> {
    let serial = |output: &mut [f32]| {
        for row in 0..rows {
            for position in 0..leading_total {
                output[position * rows + row] = wide[row * leading_total + position];
            }
        }
    };
    let Some(session) = session else {
        serial(output);
        return Ok(());
    };
    if rows.saturating_mul(leading_total) < MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH {
        serial(output);
        return Ok(());
    }
    let workers = matmul_worker_count();
    if workers <= 1 || leading_total < 2 {
        serial(output);
        return Ok(());
    }
    let chunk_count = (workers * OVERSUBSCRIBE).min(leading_total);
    let chunk_len = leading_total.div_ceil(chunk_count);
    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = &mut *output;
    let mut position_start = 0usize;
    while !remaining.is_empty() {
        let take = chunk_len.min(remaining.len() / rows);
        let (slice, rest) = remaining.split_at_mut(take * rows);
        remaining = rest;
        chunk_ranges.push((position_start, slice.as_mut_ptr() as usize, slice.len()));
        position_start += take;
    }
    if chunk_ranges.len() < 2 {
        serial(output);
        return Ok(());
    }
    let round = TransposeRound {
        wide,
        rows,
        leading_total,
        chunk_ranges: &chunk_ranges,
    };
    let report = session.run(&round);
    if report.abandoned > 0 {
        return Err(TensorError::ThreadedChunkFailed {
            chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
            reason: alloc::string::String::from(
                "cohort member panicked while running this transpose chunk",
            ),
        });
    }
    Ok(())
}

/// [`transpose_wide_to_output`]'s cohort dispatch shape: one round over
/// `(position_start, out_ptr, out_len)` ranges of `output`'s position axis,
/// run through [`CohortSession::run`]. No error path -- pure data movement,
/// nothing here can fail the way a matmul row's dot product can.
#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
pub(super) struct TransposeRound<'round> {
    pub(super) wide: &'round [f32],
    pub(super) rows: usize,
    pub(super) leading_total: usize,
    pub(super) chunk_ranges: &'round [(usize, usize, usize)],
}

#[cfg(any(
    feature = "q4k-int8-dot",
    feature = "q5k-int8-dot",
    feature = "q6k-int8-dot"
))]
impl CohortRound<TensorError> for TransposeRound<'_> {
    fn chunks(&self) -> usize {
        self.chunk_ranges.len()
    }

    fn run_chunk(&self, chunk: ChunkIndex) -> Result<(), TensorError> {
        let (position_start, out_ptr, out_len) = self.chunk_ranges[chunk.0];
        // SAFETY: unique to this chunk by construction (`split_at_mut` in
        // `transpose_wide_to_output` before the round starts); the parent
        // `output` outlives every reconstructed slice because
        // `CohortSession::run` does not return until every member has
        // reported done.
        let chunk_output = unsafe { core::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        let position_count = out_len / self.rows;
        for local_position in 0..position_count {
            let position = position_start + local_position;
            for row in 0..self.rows {
                chunk_output[local_position * self.rows + row] =
                    self.wide[row * self.leading_total + position];
            }
        }
        Ok(())
    }
}

/// Below this many consecutive [`is_staged_batch_eligible`] nodes,
/// [`run_staged_batch`] is not worth calling: a run of one node has nothing
/// to amortize a round-open against, so [`evaluate_quantized_with_scratch`]
/// falls through to the plain per-node call for it, exactly as
/// `cohort-staged-graph` off always does. Threaded through the build-time
/// sizing config (principle 12) as of ROW 98 -- see
/// `crate::sized::STAGED_BATCH_MIN_LEN`'s own doc for the measurement
/// record.
#[cfg(feature = "cohort-staged-graph")]
pub(super) use crate::sized::STAGED_BATCH_MIN_LEN;

/// One codec's row-dot kernel: `(weight_row, activation_q8k) -> dot`. Every
/// `Q4_K`/`Q5_K`/`Q6_K` kernel (`dot_q4k_q8k`/`dot_q5k_q8k`/`dot_q6k_q8k`)
/// shares this exact signature, which is what makes [`dot_fn_for`]'s return
/// type — and therefore [`MatmulStagePlan::dot_fn`] — the same concrete type
/// regardless of which codec a given matmul node uses.
#[cfg(feature = "cohort-staged-graph")]
pub(super) type MatmulRowDotFn = fn(&[u8], &[u8]) -> Result<f32, TensorError>;

/// Selects the row-dot kernel for one quantized-matmul-reduce node's own
/// codec as a plain `fn` pointer rather than naming a distinct codec
/// function at a distinct closure-literal source location -- what makes a
/// `Q4_K` node's own row-chunk work and a `Q5_K` or `Q6_K` node's the exact
/// same concrete Rust type: the ONE thing `docs/discipline.md` ROW 96 named
/// as the actual blocker to folding matmul stages into a shared round
/// ("each codec's closure is a distinct type") does not apply once the
/// codec choice is a captured VALUE ([`MatmulStagePlan::dot_fn`]) instead of
/// a name baked into the closure body. `None` for `Q8_0` (no shared
/// `Q8_K`-activation wide-fold path exists for it — see
/// [`run_reduce_quantized`]'s own per-position loop, which dequantizes
/// `Q8_0` row-by-row instead) and for `Float32` (not a quantized weight at
/// all); either leaves that node on the existing unbatched path via
/// [`is_staged_batch_eligible`].
#[cfg(feature = "cohort-staged-graph")]
pub(super) fn dot_fn_for(weight_block: QuantizedBlock<'_>) -> Option<MatmulRowDotFn> {
    match weight_block {
        #[cfg(feature = "q4k-int8-dot")]
        QuantizedBlock::Q4K(_) => Some(dot_q4k_q8k),
        #[cfg(feature = "q5k-int8-dot")]
        QuantizedBlock::Q5K(_) => Some(dot_q5k_q8k),
        #[cfg(feature = "q6k-int8-dot")]
        QuantizedBlock::Q6K(_) => Some(dot_q6k_q8k),
        _ => None,
    }
}

/// Whether `resolved` belongs in a [`run_staged_batch`] run: ONLY a
/// quantized-weight matmul fold whose own codec has a [`dot_fn_for`] entry
/// (`Q4_K`/`Q5_K`/`Q6_K` built with that codec's own `-int8-dot` feature).
/// Every other kind — elementwise, dense f32 reduce, scan, iota, constant —
/// is deliberately NOT eligible here, unlike `docs/discipline.md` ROW 96's
/// own version of this function, which admitted them. Measured, not
/// assumed (ROW 97): a mixed run (ROW 96's non-matmul kinds ALSO admitted,
/// alongside this session's matmul fold) measured `rounds` RISE 6972 ->
/// 10355 (+48.5%) — the non-matmul kinds still open a round wherever grouped
/// that they opened zero of before, exactly ROW 96's own finding, and it
/// swamps the matmul-fold's own savings. Restricting eligibility to matmul
/// alone measured `rounds` FALL 6972 -> 4412 (-36.7%), because a run this
/// narrow only ever replaces rounds that already existed (one per matmul
/// node) with fewer, larger ones — it can never ADD a round where none
/// existed, which is the one property ROW 96's broader version lacked.
#[cfg(feature = "cohort-staged-graph")]
pub(super) fn is_staged_batch_eligible(
    resolved: &BoundOp,
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
) -> bool {
    // A reduce-epilogue-fused node reads its epilogue operand's buffer
    // (`apply_reduce_epilogue`) at the SAME position it computes its own
    // fold — but `run_staged_batch` commits every stage's output into
    // `buffers` only after the whole round returns (this function's own
    // doc), and `staged_batch_run_end`'s independence check walks
    // `operands()` alone, never `epilogue_operands`. Grouping such a node
    // into a round alongside the reduce its epilogue reads would either read
    // an uncommitted buffer (`operand buffer missing at evaluation time`) or
    // silently skip the epilogue if it lands in a round of its own — no
    // staged renderer exists for `epilogue_body` at all, so this stays a
    // capability rejection (same shape `bind::bind_with_fusion`'s own doc
    // names for a backend with no epilogue renderer) rather than a silent
    // wrong answer. Excluded here, it falls through to the always-correct
    // `run_node_into` path below instead.
    if let BoundOpKind::Reduce {
        epilogue_body,
        epilogue_operands,
        ..
    } = &resolved.kind
        && !reduce_epilogue_is_identity(epilogue_body, epilogue_operands)
    {
        return false;
    }
    match quantized_operand(resolved, quantized_weights) {
        None => false,
        Some(weight_node) => quantized_weights
            .get(&weight_node)
            .is_some_and(|block| dot_fn_for(*block).is_some()),
    }
}

/// The end (exclusive) of the maximal run of [`is_staged_batch_eligible`]
/// nodes starting at `start` — [`evaluate_quantized_with_scratch`]'s own
/// walk over `resolved` reuses this rather than recomputing a dependency
/// DAG: `resolved` is already topologically ordered (`bind::bind`'s own
/// doc), so a contiguous run of eligible positions is, by construction,
/// exactly as independent-of-everything-outside-the-run as any single node
/// already is of everything before it. [`run_staged_batch`] commits every
/// stage's output into the real `buffers` table only after the whole round
/// returns, so a node added to the run must read every operand from OUTSIDE
/// the run — a node at an earlier position that is ALSO in this run has not
/// been written into `buffers` yet when an in-run consumer's stage would
/// need to read it (`docs/discipline.md` ROW 96 caught exactly this via
/// `spec::tests::a_cached_decode_step_matches_the_uncached_forward_pass_exactly`).
/// `resolved`'s topological order means any such dependency is necessarily
/// on an earlier position, so checking positions `start..end` — never
/// anything after `end` — is exhaustive.
#[cfg(feature = "cohort-staged-graph")]
pub(super) fn staged_batch_run_end(
    resolved: &[BoundOp],
    start: usize,
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
) -> usize {
    let mut end = start;
    while end < resolved.len() && is_staged_batch_eligible(&resolved[end], quantized_weights) {
        let reads_from_this_run = resolved[end].all_read_sources().any(|(operand, _, _)| {
            resolved[start..end]
                .iter()
                .any(|produced| produced.node == *operand)
        });
        if reads_from_this_run {
            break;
        }
        end += 1;
    }
    end
}

/// One quantized-matmul-reduce node's own row-parallel work, prepared
/// BEFORE [`run_staged_batch`] opens its round: activation quantization
/// (`Some(session)` — safe here because this runs strictly before
/// `session.run(&round)` is ever called, so there is no in-flight round for
/// a second `session.run` to collide with; see [`build_matmul_stage_plan`]'s
/// own call site) and the [`row_chunk_count`]-many `(row_start, ptr, len)`
/// ranges into `wide`, split via `split_at_mut` exactly the way
/// [`matmul_rows_threaded`] itself splits its own `output` — same
/// single-writer argument, one stage's own chunks instead of one call's.
#[cfg(feature = "cohort-staged-graph")]
pub(super) struct MatmulStagePlan<'plan> {
    pub(super) weights: &'plan [u8],
    // `Arc<[u8]>`, not `Vec<u8>`: `docs/discipline.md` ROW 140's own
    // measured hypothesis check (`instrument::quantize_activation_call_stats`,
    // 225 calls / 129 distinct activation nodes on the real checkpoint) --
    // `attn_q`/`attn_k`/`attn_v` (and `ffn_gate`/`ffn_up`) are consecutive
    // [`is_staged_batch_eligible`] nodes reading the SAME `activation_node`,
    // so [`build_matmul_stage_plan`]'s own `staged_quantize_cache` quantizes
    // it once and every later sibling in the same run clones the `Arc`
    // (one atomic refcount bump) instead of re-running
    // [`quantize_row_q8k_dispatch`]'s per-superblock scale search. A `Vec<u8>`
    // field would still force a byte-for-byte memcpy per cache hit; `Arc<[u8]>`
    // makes the reuse itself allocation-free.
    pub(super) activation_q8k: Arc<[u8]>,
    pub(super) row_bytes: usize,
    pub(super) q8k_row_bytes: usize,
    /// `leading_total` — [`matmul_rows_threaded`]'s own `width` parameter:
    /// how many positions are folded into one row's own dot.
    pub(super) width: usize,
    pub(super) rows: usize,
    pub(super) dot_fn: MatmulRowDotFn,
    /// row-major (`[row][position]`) scratch this stage's chunks write
    /// into; transposed to the node's real position-major output by
    /// [`run_staged_batch`] after the round returns, the same transpose
    /// [`run_reduce_quantized`]'s own wide-fold arms pay unbatched.
    pub(super) wide: Vec<f32>,
    pub(super) chunk_ranges: Vec<(usize, usize, usize)>,
}

#[cfg(feature = "cohort-staged-graph")]
impl MatmulStagePlan<'_> {
    fn run_chunk(&self, within_stage: usize) -> Result<(), TensorError> {
        let (row_start, address, length) = self.chunk_ranges[within_stage];
        // SAFETY: identical single-writer argument to `RowRound::run_chunk`
        // — carved via `split_at_mut` before the round opens (see this
        // plan's own construction in `build_matmul_stage_plan`), one chunk
        // per pointer, `self.wide` never pushed/resized again until the
        // round returns and `run_staged_batch` reads it back through `&self`.
        let chunk_output = unsafe { core::slice::from_raw_parts_mut(address as *mut f32, length) };
        for (offset, slot) in chunk_output.chunks_exact_mut(self.width).enumerate() {
            let row = row_start + offset;
            let start = row * self.row_bytes;
            let weight_row = &self.weights[start..start + self.row_bytes];
            for (position, output_slot) in slot.iter_mut().enumerate() {
                let q8k_start = position * self.q8k_row_bytes;
                *output_slot = (self.dot_fn)(
                    weight_row,
                    &self.activation_q8k[q8k_start..q8k_start + self.q8k_row_bytes],
                )?;
            }
        }
        Ok(())
    }
}

/// Builds `resolved`'s own [`MatmulStagePlan`], or `None` when its shape
/// does not clear [`quantized_matmul_workers`]'s threshold — too little
/// work to parallelize, the same threshold [`run_reduce_quantized`] itself
/// checks before calling [`matmul_rows_threaded`]. `None` here means
/// [`run_staged_batch`] falls back to running this ONE node through the
/// plain [`run_node_into`] path (a single one-chunk stage, `session: None`),
/// identical to what every other batch-eligible node kind already does.
///
/// Shape derivation duplicates (rather than extracts from)
/// [`run_reduce_quantized`]'s own `rows`/`k`/`leading_total` derivation —
/// deliberately, so this feature's own additions never touch that already
/// bit-exact-verified function's body; the copy stays narrow and close to
/// it so a future drift shows up in a diff, not behind an extra call this
/// session did not have budget to verify against every one of
/// `run_reduce_quantized`'s own edge cases (the `Q8_0` growable-cache
/// `output.is_empty()` early return among them — moot here since `Q8_0` is
/// never [`dot_fn_for`]-eligible, but a reason to keep the two copies
/// separate rather than partially shared).
#[cfg(feature = "cohort-staged-graph")]
pub(super) fn build_matmul_stage_plan<'weights>(
    resolved: &BoundOp,
    buffers: &[Option<Cow<'_, [f32]>>],
    weight_block: QuantizedBlock<'weights>,
    weight_node: NodeId,
    session: &MatmulSession<'_>,
    quantize_cache: &mut [Option<Arc<[u8]>>],
) -> Result<Option<MatmulStagePlan<'weights>>, TensorError> {
    let activation_node = resolved
        .operands()
        .iter()
        .map(|(node, _, _)| *node)
        .find(|node| *node != weight_node)
        .ok_or(TensorError::NotLowerable {
            node: resolved.node,
            reason: "quantized matmul reduce has no activation operand",
        })?;
    let activation =
        buffers[activation_node.0 as usize]
            .as_deref()
            .ok_or(TensorError::NotLowerable {
                node: activation_node,
                reason: "quantized matmul activation operand has no bound buffer",
            })?;
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        unreachable!("build_matmul_stage_plan is only called for a Keep::Reduce fold")
    };
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

    // A gathered weight operand (`moe_block.toml`'s `expert_w`) picks a
    // different expert slab per position -- this staged precompute assumes
    // one flat weight matrix shared across the whole round
    // (`run_stage_chunk`'s own dispatch). Rather than duplicate
    // `run_reduce_quantized`'s gather resolution a second time in this
    // already-deliberately-duplicated shape derivation, bail to `None`: the
    // same fallback this function already takes for `dot_fn_for`/
    // `quantized_matmul_workers` ineligibility, which routes back through
    // `run_node_into`'s plain (gather-aware) path.
    if resolved
        .operands()
        .iter()
        .any(|(node, _, gather)| *node == weight_node && gather.is_some())
    {
        return Ok(None);
    }

    let mut rows_total: u64 = 1;
    let mut leading_total_u64: u64 = 1;
    for axis in output_axes.as_slice() {
        let extent = resolved.extents[*axis as usize];
        if weight_layout.stride(*axis) != 0 {
            if activation_layout.stride(*axis) != 0 {
                // proxima-debugger: node35 shared-axis diagnostic, staged
                // (`build_matmul_stage_plan`) call site.
                #[cfg(feature = "instrument")]
                debug!(
                    reduce_node = resolved.node.0,
                    weight_node = weight_node.0,
                    activation_node = activation_node.0,
                    axis = *axis,
                    extent,
                    "build_matmul_stage_plan shared_axis_error: activation and weight both vary along the same output axis"
                );
                return Err(shared_axis_error());
            }
            rows_total *= extent;
        } else {
            leading_total_u64 *= extent;
        }
    }
    let rows = usize::try_from(rows_total).map_err(|_| shape_error())?;
    let leading_total = usize::try_from(leading_total_u64).map_err(|_| shape_error())?;

    // decision point: whether the staged-batch plan (vs the plain
    // run_reduce_quantized path) gets built for this reduce, and the
    // rows/k/leading it derives -- the shape a multi-row divergence between
    // this path and run_reduce_quantized's own is diagnosed from.
    #[cfg(feature = "instrument")]
    debug!(
        reduce_node = resolved.node.0,
        rows = rows as u64,
        k = k as u64,
        leading_total = leading_total as u64,
        "build_matmul_stage_plan shape resolved"
    );

    if std::env::var_os("PROXIMA_DEBUG_GDN_COMPARE").is_some()
        && (activation.len() == 7 * 2048 || activation.len() == 2048)
    {
        eprintln!(
            "gdn_projection reduce={} activation={} k={} rows={} leading={} extents={:?} output_axes={:?} weight_layout={weight_layout:?} activation_layout={activation_layout:?} activation_first={:?}",
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
    // `(block_bytes, block_elements)` table; see `run_reduce_quantized`'s
    // own call for why `block_elements` is keyed per codec rather than one
    // shared constant.
    if let QuantizedBlock::Int32(_) = weight_block {
        unreachable!("integer index blocks never enter quantized matmul")
    }
    let weights = weight_block.packed_bytes().ok_or_else(shape_error)?;
    let (block_bytes, block_elements) = weight_block.block_layout().ok_or_else(shape_error)?;
    if k == 0 || rows == 0 || !weights.len().is_multiple_of(block_bytes) {
        return Err(shape_error());
    }
    let total_weight_elements = (weights.len() / block_bytes) * block_elements;
    if total_weight_elements != rows * k {
        return Err(shape_error());
    }
    if activation.len() != leading_total * k {
        return Err(shape_error());
    }
    if !k.is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(shape_error());
    }

    let Some(dot_fn) = dot_fn_for(weight_block) else {
        return Ok(None);
    };
    let Some(workers) = quantized_matmul_workers(rows, activation.len()) else {
        return Ok(None);
    };

    let block_count = activation.len() / Q4K_BLOCK_ELEMENTS;
    let q8k_row_bytes = (k / Q4K_BLOCK_ELEMENTS) * Q8K_BLOCK_BYTES;
    // ROW 140's own fix: the SAME `activation_node` feeds every one of
    // `attn_q`/`attn_k`/`attn_v` (and `ffn_gate`/`ffn_up`) -- 129 distinct
    // activation nodes measured against 225 quantize calls on the real
    // checkpoint before this cache existed
    // (`instrument::quantize_activation_call_stats`, ROW 140's own doc).
    // `quantize_cache` is scoped to ONE `evaluate_quantized_with_scratch`
    // call (one decode/prefill step) -- see that function's own
    // `staged_quantize_cache` local. A hit clones the `Arc` (one atomic
    // refcount bump, no bytes touched); a miss pays the real quantize once
    // and seeds the cache for whichever sibling node reads this same
    // activation next.
    let activation_q8k: Arc<[u8]> =
        if let Some(cached) = quantize_cache[activation_node.0 as usize].as_ref() {
            #[cfg(feature = "instrument")]
            instrument::record_quantize_activation_cache_hit();
            Arc::clone(cached)
        } else {
            let mut buffer = vec![0u8; block_count * Q8K_BLOCK_BYTES];
            // `Some(session)` here is safe, unlike inside a stage's own
            // `run_stage_chunk` closure: this call runs during the precompute
            // pass, strictly BEFORE `run_staged_batch` opens its round
            // (`session.run(&round)` has not been called yet), so there is no
            // in-flight round for a second `session.run` to collide with.
            // Matches `run_reduce_quantized`'s own unbatched call exactly (same
            // function, same session), so a wide (prefill-shaped) activation
            // keeps its existing parallel quantize instead of losing it just
            // because this node got folded.
            //
            // instrumentation-only: a DEDICATED counter (`STAGED_MATMUL_QUANTIZE_TICKS`),
            // not a second call site into `MATMUL_QUANTIZE_ACTIVATION_TICKS` -- see
            // that counter's own doc for why sharing it across both call sites broke
            // `matmul_split`'s own nested-subset arithmetic. Before this counter
            // existed, this call site had no attribution at all: the staged path's
            // own quantize cost (160/225 matmul nodes per step, ROW97/98's dominant
            // bucket) was invisible.
            #[cfg(feature = "instrument")]
            let diag_staged_quantize_started = instrument::read_ticks();
            // ROW 140's own redundant-quantize hypothesis check: recorded on a
            // CACHE MISS only, i.e. once per distinct activation node this step
            // actually pays a real quantize for -- see
            // `instrument::quantize_activation_call_stats`'s own doc.
            #[cfg(feature = "instrument")]
            instrument::record_quantize_activation_call(activation_node);
            quantize_row_q8k_dispatch(activation, &mut buffer, Some(session))?;
            #[cfg(feature = "instrument")]
            counter!(
                instrument::STAGED_MATMUL_QUANTIZE_TICKS,
                instrument::elapsed_ticks(diag_staged_quantize_started)
            );
            let shared: Arc<[u8]> = Arc::from(buffer);
            quantize_cache[activation_node.0 as usize] = Some(Arc::clone(&shared));
            shared
        };
    #[cfg(feature = "instrument")]
    {
        let macs = (rows as u64)
            .saturating_mul(k as u64)
            .saturating_mul(leading_total as u64);
        counter!(instrument::STAGED_MATMUL_MACS, macs);
        counter!(instrument::STAGED_MATMUL_NODES, 1);
    }

    let row_bytes = weights.len() / rows;
    let chunk_count = row_chunk_count(rows, workers, k.saturating_mul(leading_total));
    let chunk_len = rows.div_ceil(chunk_count);
    let mut wide = vec![0.0f32; rows * leading_total];
    let mut chunk_ranges = Vec::with_capacity(chunk_count);
    let mut remaining = wide.as_mut_slice();
    let mut row_start = 0usize;
    while !remaining.is_empty() {
        let take_rows = chunk_len.min(remaining.len() / leading_total);
        let (slice, rest) = remaining.split_at_mut(take_rows * leading_total);
        remaining = rest;
        chunk_ranges.push((row_start, slice.as_mut_ptr() as usize, slice.len()));
        row_start += take_rows;
    }

    Ok(Some(MatmulStagePlan {
        weights,
        activation_q8k,
        row_bytes,
        q8k_row_bytes,
        width: leading_total,
        rows,
        dot_fn,
        wide,
        chunk_ranges,
    }))
}

/// Runs `run` (a maximal [`is_staged_batch_eligible`] slice of `resolved`,
/// starting at `resolved[run_start]`) as ONE [`StagedRound`] instead of one
/// `CohortSession::run` per node — the fix `docs/discipline.md` ROW 68/90/96
/// point at: threads stay resident and busy-spin through every stage of the
/// whole run behind a single round-open/wake, INCLUDING the quantized-matmul
/// stages that are ~87% of a forward's own wall time (ROW 96 folded every
/// OTHER kind and measured `rounds` rise, not fall, because those kinds
/// already opened zero rounds on their own — matmul is where the existing
/// per-node rounds actually live).
///
/// A node with a [`MatmulStagePlan`] becomes a many-chunk stage (real
/// cross-worker row parallelism, [`MatmulStagePlan::run_chunk`]); a matmul
/// node too small to parallelize (see [`build_matmul_stage_plan`]'s own
/// doc — the only way a node in `run` lacks a plan, since
/// [`is_staged_batch_eligible`] admits nothing else) is a single one-chunk
/// stage running the exact [`run_node_into`] call the unbatched path would
/// have made, `session: None` because that specific node would ALSO
/// serial-fallback with a real session (the same `quantized_matmul_workers`
/// threshold gates both), never because of round reentrancy.
///
/// Every output buffer for the whole run is allocated up front (reusing
/// [`take_or_allocate`], the same pool [`evaluate_quantized_with_scratch`]'s
/// per-node path already draws from) so the round's own closure can hold a
/// raw pointer to each stage's own disjoint slot before the round opens —
/// retirement (`retires`) is applied after the round returns, in run order,
/// so the final `buffers`/`free_buffers` state this leaves is identical to
/// running every node in `run` one at a time; the only difference is that a
/// buffer whose last use falls inside the run is held slightly longer
/// (until the run's own round returns) instead of being freed the instant
/// its consumer finishes — bounded by one run's own total output size, not
/// the whole step's.
#[cfg(feature = "cohort-staged-graph")]
#[allow(clippy::too_many_arguments)]
pub(super) fn run_staged_batch(
    run: &[BoundOp],
    run_start: usize,
    buffers: &mut [Option<Cow<'_, [f32]>>],
    quantized_weights: &BTreeMap<NodeId, QuantizedBlock>,
    expert_sources: Option<&BTreeMap<NodeId, ExpertSource<'_>>>,
    session: &MatmulSession<'_>,
    free_buffers: &mut Vec<Vec<f32>>,
    retires: &[Vec<NodeId>],
    live_now: &mut usize,
    quantize_cache: &mut [Option<Arc<[u8]>>],
) -> Result<(), TensorError> {
    let mut run_outputs: Vec<Vec<f32>> = run
        .iter()
        .map(|node| take_or_allocate(free_buffers, node_output_len(node)))
        .collect();
    let buffers_ref: &[Option<Cow<'_, [f32]>>] = buffers;

    let mut plans: Vec<Option<MatmulStagePlan<'_>>> = Vec::with_capacity(run.len());
    for node in run {
        let plan = match quantized_operand(node, quantized_weights) {
            Some(weight_node) => {
                let weight_block = quantized_weights.get(&weight_node).copied().ok_or(
                    TensorError::NotLowerable {
                        node: weight_node,
                        reason: "quantized weight node has no bound byte buffer",
                    },
                )?;
                build_matmul_stage_plan(
                    node,
                    buffers_ref,
                    weight_block,
                    weight_node,
                    session,
                    quantize_cache,
                )?
            }
            None => None,
        };
        plans.push(plan);
    }

    let stage_offsets: Vec<usize> = core::iter::once(0)
        .chain(plans.iter().scan(0usize, |total, plan| {
            *total += plan.as_ref().map_or(1, |plan| plan.chunk_ranges.len());
            Some(*total)
        }))
        .collect();
    let output_slots: Vec<(usize, usize)> = run_outputs
        .iter_mut()
        .map(|buffer| (buffer.as_mut_ptr() as usize, buffer.len()))
        .collect();
    let completed: Vec<AtomicUsize> = (0..run.len()).map(|_| AtomicUsize::new(0)).collect();

    let round = StagedRound {
        stage_offsets: &stage_offsets,
        completed: &completed,
        run_stage_chunk: |stage: usize, within: usize| -> Result<(), TensorError> {
            match &plans[stage] {
                Some(plan) => plan.run_chunk(within),
                None => {
                    let computed = &run[stage];
                    let (address, length) = output_slots[stage];
                    // SAFETY: single-writer argument identical to
                    // `ElementwiseRowRound`/`RowRound`'s own `split_at_mut`-carved
                    // ranges — one stage owns this whole slot (`plans[stage]`
                    // is `None`, so this stage is exactly one chunk).
                    let output =
                        unsafe { core::slice::from_raw_parts_mut(address as *mut f32, length) };
                    // `run_staged_batch` is only ever entered from the
                    // `!exact_activations` staged-batch arm above, so this
                    // sub-call always wants the int8 path.
                    run_node_into(
                        computed,
                        buffers_ref,
                        Some(quantized_weights),
                        expert_sources,
                        None,
                        false,
                        output,
                    )
                }
            }
        },
    };

    // instrumentation-only: times `session.run(&round)` as a whole, the
    // same granularity `matmul_rows_threaded`'s own `MATMUL_OWN_CHUNK_TICKS`
    // uses for the unbatched leader-claim-and-wait call -- not per-chunk
    // (that would perturb the very thing being measured, see this module's
    // own doc) and not folded into `MATMUL_OWN_CHUNK_TICKS` itself, since
    // that counter's own denominator (`MATMUL_DISPATCH_CALLS`/per-node call
    // count) means something different for a run that folds several matmul
    // nodes into one round.
    #[cfg(feature = "instrument")]
    let diag_staged_round_started = instrument::read_ticks();
    let report = session.run(&round);
    #[cfg(feature = "instrument")]
    counter!(
        instrument::STAGED_MATMUL_ROUND_TICKS,
        instrument::elapsed_ticks(diag_staged_round_started)
    );
    if let Some(error) = report.first_error {
        return Err(error);
    }
    if report.abandoned > 0 {
        return Err(TensorError::ThreadedChunkFailed {
            chunk: report.first_abandoned.map_or(0, |chunk| chunk.0 + 1),
            reason: alloc::string::String::from(
                "cohort member panicked while running a staged graph batch",
            ),
        });
    }

    // `Some(session)` here is safe for the identical reason
    // `build_matmul_stage_plan`'s own quantize call is: this loop runs
    // strictly AFTER `session.run(&round)` above has already returned, so
    // the round this session was driving is closed before any of these
    // transpose calls open a new one.
    #[cfg(feature = "instrument")]
    let diag_staged_transpose_started = instrument::read_ticks();
    for (offset, node_output) in run_outputs.iter_mut().enumerate() {
        if let Some(plan) = &plans[offset] {
            transpose_wide_to_output(
                &plan.wide,
                plan.rows,
                plan.width,
                Some(session),
                node_output,
            )?;
        }
    }
    #[cfg(feature = "instrument")]
    counter!(
        instrument::STAGED_MATMUL_TRANSPOSE_TICKS,
        instrument::elapsed_ticks(diag_staged_transpose_started)
    );

    for (offset, node_output) in run_outputs.into_iter().enumerate() {
        let node = run[offset].node;
        buffers[node.0 as usize] = Some(Cow::Owned(node_output));
        *live_now += 1;
        for retired in &retires[run_start + offset] {
            if retire_into(buffers, *retired, free_buffers) {
                *live_now -= 1;
            }
        }
    }
    Ok(())
}

/// [`Op::Reduce`] with a data-dependent (scatter) `out_map`, `f32` only —
/// the dedicated sequential path every fast path in [`run_reduce`] stays
/// ineligible for (`bind::BoundOp::split`'s own doc has the reason: no chunk
/// rebase story for `out_scatter`, so a scatter never reaches the NEON/dot/
/// width tiles above, which all assume a `Keep::Reduce` fold owns its own
/// disjoint output range).
///
/// No atomics: the CPU interpreter already walks its reduce loop strictly
/// in iteration order, one coordinate at a time, so a colliding write is
/// just another `reduce_op` fold applied to `output[dest]` in place —
/// nothing else can observe or mutate `output` mid-walk. This is the
/// forward half of the worked example `map.rs`'s `IndexMap::Computed` doc
/// and this function's own tests name: `src=[10,20,30,40]`,
/// `idx=[2,0,2,1]`, destination extent 3, body `Add`, `init` `Zero` ->
/// `out=[20,40,40]` (`out[2]` folds `10` then `30`, in iteration order).
///
/// `output` is filled with `init`'s identity *before* the walk (`init ==
/// ReduceInit::FirstElement` is rejected at shape-inference time — see
/// `shape.rs`'s `infer_reduce` — because which source element is "first" at
/// a colliding destination is not well-defined), since which cells a
/// data-dependent write ever touches is unknown until the fetched indices
/// are read.
pub(super) fn run_reduce_scatter<B: Deref<Target = [f32]>>(
    resolved: &BoundOp,
    buffers: &[Option<B>],
    output: &mut [f32],
) -> Result<(), TensorError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        out_layout,
        out_scatter,
        ..
    } = &resolved.kind
    else {
        unreachable!("run_reduce_scatter is only called for a Keep::Reduce fold")
    };
    let Some(target) = out_scatter else {
        unreachable!("run_reduce_scatter is only dispatched when out_scatter is Some")
    };

    output.fill(initial_value(*init).unwrap_or(0.0));

    let raw = operand_buffers(resolved, buffers)?;
    let body = resolved.element_body();
    let shape = body_shape(body);
    let mut operand_values = vec![0.0f32; raw.len()];
    let mut step_values = vec![0.0f32; body.steps.len()];

    let index_buffer = buffer_of(buffers, target.indices)?;
    let mut running: Vec<i64> = vec![0; raw.len()];
    let mut gather_cursors: Vec<Option<GatherCursor>> = (0..raw.len()).map(|_| None).collect();
    let mut coordinate = vec![0u64; resolved.extents.len()];
    let iteration_total = odometer_len(&resolved.extents);

    for flat in 0..iteration_total {
        unflatten_into(flat, &resolved.extents, &mut coordinate);
        fill_running_offsets(resolved, &coordinate, &mut running);
        fill_gather_cursors(resolved, buffers, &coordinate, None, &mut gather_cursors)?;

        for (index, data) in raw.iter().enumerate() {
            let mut offset = running[index];
            if let Some(cursor) = gather_cursors[index].as_mut() {
                offset += cursor.fetch_and_advance(resolved.node)?;
            }
            operand_values[index] = data[offset as usize];
        }
        let value = eval_body_shape(&shape, &operand_values, &mut step_values);

        let mut destination = GatherCursor {
            buffer: index_buffer,
            offset: target.index_layout.offset_of(&coordinate),
            stride: 0,
            element_stride: target.element_stride,
            extent: target.extent,
        };
        let dest_offset =
            out_layout.offset_of(&coordinate) + destination.fetch_and_advance(resolved.node)?;
        let slot = &mut output[dest_offset as usize];
        *slot = apply_scalar_op(*reduce_op, &[*slot, value]);
    }
    Ok(())
}

/// [`ExpertSelectionEntry::record`]'s smoothing factor -- a plain constant,
/// not build-time-configurable: this slice's job is emitting the two raw
/// inputs a per-node precision-allocation rule needs (DynaExq, arXiv
/// 2511.15015's "budget-feasible top-n by EMA hotness"), not tuning or
/// implementing that rule, so there is no consumer yet to size this
/// against.
pub(super) const EXPERT_SELECTION_EMA_ALPHA: f64 = 0.1;
