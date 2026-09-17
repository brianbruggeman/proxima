use super::*;

// ---- width-dim register-tile GEMM kernel (aarch64) ----
//
// `reduce_width_binary`'s FMA axpy path above still round-trips
// `accumulator` through memory every reduction step (load-fma-store per
// width chunk), because `accumulator` is a `&mut [f32]` slice spanning the
// whole output width and can never fit in registers. This tile instead
// keeps a small (rows x columns) block of partial sums in NEON registers
// for the WHOLE `k` reduction, spilling nothing until the block is fully
// accumulated — the same register-budget technique that took the sibling
// dot-path tile from 0.122s to 0.028s at 1024^3.
//
// Applicable to exactly the shape `run_reduce`'s width path already
// specializes for: `out[m, n] += a[m, k] * b[k, n]`, `a` invariant across
// `n` (width stride 0), `b` contiguous along `n` (width stride 1), a
// single leading dim, a single contraction dim.

/// Output rows one call to [`gemm_width_tile_neon`] computes.
#[cfg(target_arch = "aarch64")]
pub(super) use crate::sized::WIDTH_TILE_ROWS;

/// `float32x4_t` vectors of output columns one call to [`gemm_width_tile_neon`]
/// computes — 4 gives `WIDTH_TILE_ROWS * WIDTH_TILE_VECS` = 16 independent
/// accumulators, the measured saturation point for this core's NEON FMA
/// throughput.
#[cfg(target_arch = "aarch64")]
pub(super) use crate::sized::WIDTH_TILE_VECS;

/// Pass/invocation/fallback-element counts for the width tile — mandatory
/// verification, not a runtime feature: a caller (`profile_hot`) reads
/// these after a run to prove the tile actually fired, since a silently-zero
/// invocation count would make any timing number meaningless. Mirrors
/// [`NEON_TILE_GATE_PASSES`]'s family exactly, including the column-tail
/// coverage [`run_width_tile_neon`] below accounts for: `invocations *
/// (WIDTH_TILE_ROWS * WIDTH_TILE_VECS * 4) + row_remainder_elements
/// (`WIDTH_TILE_ROW_REMAINDER_ELEMENTS`, below) + fallback_elements ==
/// leading_total * width` for any shape, not only multiples of the tile.
/// `fallback_elements` alone no longer covers the leading-row remainder —
/// that is now [`gemm_width_tile_neon`] at `ROWS = 2`/`ROWS = 1`, counted
/// separately below — only the column tail still reaches the scalar path.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static WIDTH_TILE_GATE_PASSES: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static WIDTH_TILE_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static WIDTH_TILE_FALLBACK_ELEMENTS: AtomicU64 = AtomicU64::new(0);
/// Row-remainder tile invocations (`ROWS = 2` or `ROWS = 1`,
/// `run_width_tile_neon`'s own greedy 2-then-1 dispatch), tracked apart
/// from [`WIDTH_TILE_INVOCATIONS`] since remainder tiles compute a
/// different, `ROWS`-dependent number of outputs per call than the fixed
/// `WIDTH_TILE_ROWS`-row main tile — exactly why
/// [`NEON_TILE_ROW_REMAINDER_INVOCATIONS`] exists apart from
/// [`NEON_TILE_INVOCATIONS`] for the dot-path tile.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static WIDTH_TILE_ROW_REMAINDER_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
/// Output elements actually covered by row-remainder tiles (`ROWS * (4 *
/// WIDTH_TILE_VECS)` added per invocation) — the width-tile counterpart to
/// [`NEON_TILE_ROW_REMAINDER_ELEMENTS`], usable directly in the coverage
/// identity without knowing which of `ROWS = 2`/`ROWS = 1` fired.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static WIDTH_TILE_ROW_REMAINDER_ELEMENTS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the three `WIDTH_TILE_GATE_PASSES`-family counters:
/// (gate passes, tile invocations, fallback elements) — the width tile's
/// counterpart to [`neon_tile_counters`].
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn width_tile_counters() -> (u64, u64, u64) {
    (
        WIDTH_TILE_GATE_PASSES.load(Ordering::Relaxed),
        WIDTH_TILE_INVOCATIONS.load(Ordering::Relaxed),
        WIDTH_TILE_FALLBACK_ELEMENTS.load(Ordering::Relaxed),
    )
}

/// `WIDTH_TILE_ROW_REMAINDER_INVOCATIONS` snapshot — the row-remainder
/// tiles' own invocation count (`ROWS = 2` or `ROWS = 1`), separate from
/// the main `WIDTH_TILE_ROWS`-row tile's, the width-tile counterpart to
/// [`neon_tile_row_remainder_invocations`].
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn width_tile_row_remainder_invocations() -> u64 {
    WIDTH_TILE_ROW_REMAINDER_INVOCATIONS.load(Ordering::Relaxed)
}

/// `WIDTH_TILE_ROW_REMAINDER_ELEMENTS` snapshot — output elements covered
/// by row-remainder tiles of either width, for the `main*64 +
/// row_remainder + fallback == m*n` coverage identity, the width-tile
/// counterpart to [`neon_tile_row_remainder_elements`].
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn width_tile_row_remainder_elements() -> u64 {
    WIDTH_TILE_ROW_REMAINDER_ELEMENTS.load(Ordering::Relaxed)
}

/// Everything [`try_run_width_tile`] needs, bundled the same way
/// [`OperandSpan`] and [`DotFold`] are — keeps the entry point under
/// clippy's argument-count lint instead of reaching for `#[allow]`.
/// `cfg`-gated to aarch64: the width tile path is compiled out entirely on
/// every other target, so there is nothing left to hand this to.
#[cfg(target_arch = "aarch64")]
pub(super) struct WidthPathContext<'a> {
    pub(super) resolved: &'a BoundOp,
    pub(super) shape: &'a BodyShape<'a>,
    pub(super) strides: &'a [i64],
    pub(super) reduce_op: ScalarOp,
    pub(super) init: ReduceInit,
    pub(super) leading_output_axes: &'a [u16],
    pub(super) reduction_dims: &'a [u16],
    pub(super) last_output_dim: Option<u16>,
    pub(super) width: usize,
    pub(super) out_layout: &'a bind::Layout,
}

/// Everything the tiled loop needs to walk, resolved once per bound op.
/// `outer_extent`/`outer_stride_*` (attention-tile task, 2026-09-01): a
/// second, OUTER leading axis both operands step through together (`heads`,
/// for BGE's `Q@K^T`/`softmax@V` folds) -- `1`/`0`/`0`/`0` for every
/// single-leading-axis node (every node this tile already served), which
/// makes the outer loop [`run_width_tile_neon`] wraps its walk in run
/// exactly once at zero offset: bit-identical to the pre-existing address
/// sequence for those nodes. Mirrors [`ConvGemmTilePlan`]'s own
/// `outer_extent`/`outer_stride_m`/`outer_stride_n` shape, applied to the
/// leading axes instead of the reduction axes.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub(super) struct WidthTilePlan {
    pub(super) a_operand: usize,
    pub(super) b_operand: usize,
    pub(super) row_stride_a: i64,
    pub(super) base_a: i64,
    pub(super) k_stride_a: i64,
    pub(super) base_b: i64,
    pub(super) k_stride_b: i64,
    pub(super) out_base: i64,
    pub(super) out_row_stride: i64,
    pub(super) out_col_stride: i64,
    pub(super) leading_total: usize,
    pub(super) reduction_total: usize,
    pub(super) width: usize,
    pub(super) seed: f32,
    pub(super) outer_extent: usize,
    pub(super) outer_stride_a: i64,
    pub(super) outer_stride_b: i64,
    pub(super) outer_stride_out: i64,
    /// `float32x4_t` vectors [`run_width_tile_neon`] tiles this call at --
    /// `WIDTH_TILE_VECS` (4, 16-wide) for every node this tile already
    /// served, unchanged; `2` (8-wide) or `1` (4-wide) for the narrow-N
    /// nodes `width_tile_vecs_for` admits (narrow-tile task, 2026-09-01) --
    /// `attn_qk`'s `Q@K^T`, `N = seq_len` in `{7, 8, 9}`, always below the
    /// `WIDTH_TILE_VECS * 4 == 16` main-tile floor. Selects which
    /// monomorphisation of [`gemm_width_tile_neon`] the call site
    /// instantiates -- a runtime value driving a compile-time const generic
    /// choice via [`try_run_width_tile`]'s own match, the same shape
    /// `width_row_remainder_tile!`'s `$rows` literal dispatch already uses
    /// for `ROWS`.
    pub(super) vecs: usize,
}

/// Whether `dims` (`resolve_reduce_axis_shape`'s own outer-to-inner order)
/// compose into ONE virtual axis this operand's `layout` can be walked with
/// a single constant stride — `Some((combined_extent, stride))`, `None` the
/// first time the chain breaks. Weaker than [`reduction_is_fully_flat`]:
/// that function additionally demands the innermost stride equal `1` (a
/// literal contiguous span); this only demands each outer dim's stride equal
/// the product of every extent nested inside it, relative to whatever the
/// innermost stride actually is — the general row-major-VIEW condition, not
/// the stronger row-major-STORAGE one. `attn_o`'s weight operand (`[heads,
/// head_dim]` reducing into a `[384,384]` matrix laid out `[in=384,
/// out=384]`) composes with stride `384` (its own `out`-axis width), never
/// `1` — `reduction_is_fully_flat` would wrongly decline it.
#[cfg(target_arch = "aarch64")]
pub(super) fn composed_reduction_stride(
    resolved: &BoundOp,
    dims: &[u16],
    layout: &bind::Layout,
) -> Option<(i64, i64)> {
    let (&innermost, outer_dims) = dims.split_last()?;
    let stride = layout.stride(innermost);
    let mut extent_total = resolved.extents[innermost as usize] as i64;
    for &dim in outer_dims.iter().rev() {
        if layout.stride(dim) != stride.saturating_mul(extent_total) {
            return None;
        }
        extent_total = extent_total.saturating_mul(resolved.extents[dim as usize] as i64);
    }
    Some((extent_total, stride))
}

/// Which [`gemm_width_tile_neon`] `VECS` monomorphisation `width` admits,
/// `None` when even the narrowest (`VECS = 1`, 4-wide) tile does not fit --
/// narrow-tile task (2026-09-01), discharging ROW 216's named residual.
/// `WIDTH_TILE_VECS` (4, 16-wide) is unchanged and checked first so every
/// node the main tile already serves takes the SAME branch it always has;
/// `2` (8-wide) matches BGE's `Q@K^T` `N = 8` sentence exactly, `1` (4-wide)
/// covers `N` in `4..=7` (BGE's `N = 7` sentence) with a 3-column scalar
/// tail [`run_width_tile_neon`]'s existing column-tail loop already walks
/// unchanged -- no new remainder mechanism, the same one `VECS = 4` nodes
/// use when `width` is not itself a multiple of `WIDTH_TILE_VECS * 4`.
#[cfg(target_arch = "aarch64")]
pub(super) const fn width_tile_vecs_for(width: usize) -> Option<usize> {
    if width >= WIDTH_TILE_VECS * 4 {
        Some(WIDTH_TILE_VECS)
    } else if width >= 8 {
        Some(2)
    } else if width >= 4 {
        Some(1)
    } else {
        None
    }
}

/// Resolves [`WidthTilePlan`] once per bound op, or `None` when this node
/// does not match the shape the tile is built for. Every condition here is
/// checked once, never per element.
#[cfg(target_arch = "aarch64")]
pub(super) fn width_tile_plan(context: &WidthPathContext) -> Option<WidthTilePlan> {
    // width-gate-decline task (2026-09-01): every `return None` below also
    // records WHICH condition rejected the node, restricted to the 96
    // gemm-shaped `MatMul` folds (`reduce_is_gemm_shaped`) so the table
    // does not also collect the 74 small LayerNorm/softmax reduces that
    // structurally reach this function too. `-1` marks a shape/stride
    // field not yet resolvable at that decline point -- see
    // `instrument::WidthDeclineTotals`'s own doc for which fields that is,
    // per reason.
    #[cfg(feature = "instrument")]
    let record_decline = |reason: instrument::WidthDeclineReason,
                          m: i64,
                          k: i64,
                          n: i64,
                          stride_a: i64,
                          stride_b: i64| {
        if reduce_is_gemm_shaped(context.resolved) {
            instrument::record_width_tile_decline(
                context.resolved.node,
                reason,
                m,
                k,
                n,
                stride_a,
                stride_b,
            );
        }
    };
    #[cfg(feature = "instrument")]
    let width_i64 = context.width as i64;

    if !FUSED_MULTIPLY_ADD || context.reduce_op != ScalarOp::Add {
        #[cfg(feature = "instrument")]
        record_decline(
            instrument::WidthDeclineReason::NoFusedMultiplyAdd,
            -1,
            -1,
            width_i64,
            -1,
            -1,
        );
        return None;
    }
    let (operand_a, operand_b) = match *context.shape {
        BodyShape::Binary(ScalarOp::Multiply, operand_a, operand_b) => (operand_a, operand_b),
        _ => {
            #[cfg(feature = "instrument")]
            record_decline(
                instrument::WidthDeclineReason::NotMultiplyAddBody,
                -1,
                -1,
                width_i64,
                -1,
                -1,
            );
            return None;
        }
    };
    #[cfg(feature = "instrument")]
    let stride_a_early = context.strides[operand_a as usize];
    #[cfg(feature = "instrument")]
    let stride_b_early = context.strides[operand_b as usize];
    if matches!(context.init, ReduceInit::FirstElement) {
        #[cfg(feature = "instrument")]
        record_decline(
            instrument::WidthDeclineReason::FirstElementInit,
            -1,
            -1,
            width_i64,
            stride_a_early,
            stride_b_early,
        );
        return None;
    }
    // ROW 210 (width-gate-decline task, 2026-09-01): the per-`NodeId` census
    // (`instrument::width_tile_decline_snapshot`) named all 36 of BGE's
    // declined `MatMul` folds as `AxesShape` and split them into two real
    // classes, neither the originally-suspected stride-layout condition:
    // 24 (`Q@K^T`/`softmax@V`) carry a genuine 2-axis LEADING walk
    // (`[heads, seq_q]`, heads never extent 1 -- still declined below,
    // unchanged), and 12 (`attention/output/dense`, `attn_o`) carry a
    // 2-axis REDUCTION (`[heads, head_dim]`, `12 x 32 == 384`) that measured
    // fully row-major-composable for BOTH operands at every one of BGE's 12
    // layers (`stride(heads) == stride(head_dim) * extent(head_dim)`,
    // `docs/discipline.md` ROW 210's own trace) -- `attn_o`'s weight was
    // never algebraically reshaped back to a flat `[384,384]`, but its
    // physical layout already IS that matrix. `composed_reduction_stride`
    // below proves this per-call rather than assuming it: any node whose
    // operands do NOT compose (a non-contiguous view) still declines,
    // unchanged.
    if context.leading_output_axes.is_empty() || context.reduction_dims.is_empty() {
        #[cfg(feature = "instrument")]
        record_decline(
            instrument::WidthDeclineReason::AxesShape,
            -1,
            -1,
            width_i64,
            stride_a_early,
            stride_b_early,
        );
        return None;
    }
    // attention-tile task (2026-09-01) + three-axis-merge task (2026-09-02):
    // up to THREE non-degenerate leading axes now reach this point --
    // `attn_qk`/`attn_v` at batch > 1 (BGE's Q@K^T and softmax@V folds)
    // carry `[batch, heads, seq_q]`, none extent 1 once batch stops being
    // elided. Which axis is the tile's own "row" axis (only the `a` operand
    // varies over it, `b` constant -- the shape every single-axis node
    // already proves) versus the "outer" axis/axes (both operands step by
    // them, e.g. per-batch-per-head K/V) is a physical-stride question,
    // answered below once `layout_a`/`layout_b` are resolved -- not an
    // extent question, so only bound and collect candidates here. More
    // than 3 non-degenerate leading axes has no proven shape and declines
    // now rather than being guessed at further down.
    let non_degenerate_leading: Vec<u16> = context
        .leading_output_axes
        .iter()
        .copied()
        .filter(|&axis| context.resolved.extents[axis as usize] != 1)
        .collect();
    if non_degenerate_leading.len() > 3 {
        #[cfg(feature = "instrument")]
        record_decline(
            instrument::WidthDeclineReason::AxesShape,
            -1,
            -1,
            width_i64,
            stride_a_early,
            stride_b_early,
        );
        return None;
    }
    // batch-widen task (2026-09-02): no longer instrument-only -- the `(0,
    // 0)` two-leading-axis merge below reads this to sanity-check the
    // composed extent unconditionally, not just to log a decline.
    let leading_total_early: i64 = if non_degenerate_leading.is_empty() {
        1
    } else {
        non_degenerate_leading
            .iter()
            .map(|&axis| context.resolved.extents[axis as usize] as i64)
            .product()
    };
    #[cfg(feature = "instrument")]
    let reduction_total_early: i64 = context
        .reduction_dims
        .iter()
        .map(|&dim| context.resolved.extents[dim as usize] as i64)
        .product();
    let Some(vecs) = width_tile_vecs_for(context.width) else {
        #[cfg(feature = "instrument")]
        record_decline(
            instrument::WidthDeclineReason::NarrowWidth,
            leading_total_early,
            reduction_total_early,
            width_i64,
            stride_a_early,
            stride_b_early,
        );
        return None;
    };
    let Some(last_output_dim) = context.last_output_dim else {
        #[cfg(feature = "instrument")]
        record_decline(
            instrument::WidthDeclineReason::NoOutputDim,
            leading_total_early,
            reduction_total_early,
            width_i64,
            stride_a_early,
            stride_b_early,
        );
        return None;
    };

    let operands = context.resolved.operands();
    let (_, layout_a_raw, gather_a) = &operands[operand_a as usize];
    let (_, layout_b_raw, gather_b) = &operands[operand_b as usize];
    if gather_a.is_some() || gather_b.is_some() {
        #[cfg(feature = "instrument")]
        record_decline(
            instrument::WidthDeclineReason::Gathered,
            leading_total_early,
            reduction_total_early,
            width_i64,
            stride_a_early,
            stride_b_early,
        );
        return None;
    }
    let (a_operand, layout_a, b_operand, layout_b) = match (
        context.strides[operand_a as usize],
        context.strides[operand_b as usize],
    ) {
        (0, 1) => (
            operand_a as usize,
            layout_a_raw,
            operand_b as usize,
            layout_b_raw,
        ),
        (1, 0) => (
            operand_b as usize,
            layout_b_raw,
            operand_a as usize,
            layout_a_raw,
        ),
        _ => {
            #[cfg(feature = "instrument")]
            record_decline(
                instrument::WidthDeclineReason::StrideLayout,
                leading_total_early,
                reduction_total_early,
                width_i64,
                stride_a_early,
                stride_b_early,
            );
            return None;
        }
    };

    // final leading-axis resolution: needs `layout_a`/`layout_b`'s physical
    // strides, unavailable until the stride-layout match just above settles
    // which operand is `a` (row-varying, contiguous over `k`) vs `b`
    // (weight-like, constant per row). Zero non-degenerate axes: `leading_dim`
    // stays whatever `leading_output_axes[0]` is (extent 1, a single
    // degenerate row) -- the pre-existing behavior, and safe regardless of
    // `layout_b`'s stride there since the row loop below only ever runs once.
    // One: `gemm_width_tile_neon` has no `row_stride_b` -- it reads `b` from
    // the SAME base for every row, so this is only sound when `b` is truly
    // row-invariant (`layout_b.stride(only) == 0`, the shared-weight GEMM
    // shape). A per-row `b` (a batched/grouped fold, e.g. a per-head weight
    // slice reached only after `bind`'s elementwise-into-reduce fusion)
    // silently reused row 0's `b` slice for every row here before this
    // check existed -- proven by `omega`'s `backend_parity`/`metal_real_forward`
    // real-forward-graph gates, which caught it as every row/head of a
    // fused reduce collapsing to the first row's value. Two (`attn_qk`/
    // `attn_v`): the axis where `layout_b`'s stride is 0 is the row axis
    // (the SAME shape every other node already proves); the other must be a
    // genuine outer axis both operands step by -- `layout_b`'s stride
    // nonzero there by construction (the complement), and `layout_a`'s
    // stride ALSO nonzero (else `a` never changes across it, a shape this
    // tile has not proven and declines rather than guesses).
    let (leading_dim, outer_dims, merged_leading) = match non_degenerate_leading.as_slice() {
        [] => (context.leading_output_axes[0], None, None),
        &[only] if layout_b.stride(only) == 0 => (only, None, None),
        &[_] => {
            #[cfg(feature = "instrument")]
            record_decline(
                instrument::WidthDeclineReason::AxesShape,
                leading_total_early,
                reduction_total_early,
                width_i64,
                stride_a_early,
                stride_b_early,
            );
            return None;
        }
        &[first, second] => match (layout_b.stride(first), layout_b.stride(second)) {
            (0, other) if other != 0 => (first, Some((second, None)), None),
            (other, 0) if other != 0 => (second, Some((first, None)), None),
            // batch-widen task (2026-09-02): `b` (the weight) is invariant
            // over BOTH leading axes at once -- a plain `[batch, seq, k] @
            // [k, n]` GEMM once `batch` stops being elided (extent > 1).
            // There is no genuine "outer" step here (`b` never varies), so
            // `first`/`second` just need to compose row-major on `a` and
            // `out` to become one flat leading axis of extent `first *
            // second` -- the SAME row-major-VIEW proof
            // `composed_reduction_stride` already gives the multi-axis
            // REDUCTION case below (ROW 210's `attn_o`), reused verbatim
            // rather than re-derived (`leading_output_axes`'s own
            // outer-to-inner order, `[first, second]`, is exactly the
            // `dims` order that function expects). Any operand that does
            // NOT compose (a real broadcast or a transposed batch axis)
            // still declines, unchanged.
            (0, 0) => {
                match (
                    composed_reduction_stride(context.resolved, &[first, second], layout_a),
                    composed_reduction_stride(
                        context.resolved,
                        &[first, second],
                        context.out_layout,
                    ),
                ) {
                    (Some((extent_a, stride_a)), Some((extent_out, stride_out)))
                        if extent_a == leading_total_early && extent_out == leading_total_early =>
                    {
                        (
                            first,
                            None,
                            Some((stride_a, stride_out, leading_total_early as usize)),
                        )
                    }
                    _ => {
                        #[cfg(feature = "instrument")]
                        record_decline(
                            instrument::WidthDeclineReason::AxesShape,
                            leading_total_early,
                            reduction_total_early,
                            width_i64,
                            stride_a_early,
                            stride_b_early,
                        );
                        return None;
                    }
                }
            }
            _ => {
                #[cfg(feature = "instrument")]
                record_decline(
                    instrument::WidthDeclineReason::AxesShape,
                    leading_total_early,
                    reduction_total_early,
                    width_i64,
                    stride_a_early,
                    stride_b_early,
                );
                return None;
            }
        },
        // three-axis-merge task (2026-09-02): `attn_qk`/`attn_v` at batch >
        // 1 carry `[batch, heads, seq_q]`, all three non-degenerate --
        // established from `lower_matmul`'s own affine projections
        // (`proxima-onnx/src/lower.rs`), not assumed: `b` (K/V)'s pattern
        // skips `seq_q` (axis 2) on both operands for both folds, so `seq_q`
        // is the row axis (`b` invariant, the same shape every single-axis
        // node already proves) and `batch`/`heads` are the two axes `b`
        // genuinely varies over -- there is no third "outer" role, `batch`
        // and `heads` must instead compose into ONE outer step the same way
        // the two-axis `(0, 0)` arm above composes a row, reusing
        // `composed_reduction_stride` again rather than re-deriving it.
        // Exactly one candidate may have `b`-stride 0 -- zero or two-plus
        // has no proven shape and declines (`AxesShape`) rather than
        // guessing which axis is the row.
        &[first, second, third] => {
            let row_candidates: Vec<u16> = [first, second, third]
                .into_iter()
                .filter(|&axis| layout_b.stride(axis) == 0)
                .collect();
            match *row_candidates.as_slice() {
                [row] if row == first => (first, Some((second, Some(third))), None),
                [row] if row == second => (second, Some((first, Some(third))), None),
                [row] if row == third => (third, Some((first, Some(second))), None),
                _ => {
                    #[cfg(feature = "instrument")]
                    record_decline(
                        instrument::WidthDeclineReason::AxesShape,
                        leading_total_early,
                        reduction_total_early,
                        width_i64,
                        stride_a_early,
                        stride_b_early,
                    );
                    return None;
                }
            }
        }
        _ => unreachable!("non_degenerate_leading.len() > 3 already declined above"),
    };
    let outer_info = match outer_dims {
        Some((axis, None)) if layout_a.stride(axis) != 0 => Some((
            context.resolved.extents[axis as usize] as usize,
            layout_a.stride(axis),
            layout_b.stride(axis),
            context.out_layout.stride(axis),
        )),
        Some((_, None)) => {
            #[cfg(feature = "instrument")]
            record_decline(
                instrument::WidthDeclineReason::AxesShape,
                leading_total_early,
                reduction_total_early,
                width_i64,
                stride_a_early,
                stride_b_early,
            );
            return None;
        }
        // three-axis-merge task (2026-09-02): `batch`/`heads` compose into
        // ONE outer step, the same row-major-VIEW proof the two-axis `(0,
        // 0)` row merge above already gives, applied to `layout_a`,
        // `layout_b`, AND `context.out_layout` this time -- unlike the row
        // merge, `b` genuinely steps here (that is what makes these axes
        // "outer" rather than "row"), so all three operands must
        // independently compose to the SAME extent or this declines rather
        // than guessing.
        Some((first, Some(second))) => {
            let outer_total_early = context.resolved.extents[first as usize] as i64
                * context.resolved.extents[second as usize] as i64;
            match (
                composed_reduction_stride(context.resolved, &[first, second], layout_a),
                composed_reduction_stride(context.resolved, &[first, second], layout_b),
                composed_reduction_stride(context.resolved, &[first, second], context.out_layout),
            ) {
                (
                    Some((extent_a, stride_a)),
                    Some((extent_b, stride_b)),
                    Some((extent_out, stride_out)),
                ) if extent_a == outer_total_early
                    && extent_b == outer_total_early
                    && extent_out == outer_total_early =>
                {
                    Some((outer_total_early as usize, stride_a, stride_b, stride_out))
                }
                _ => {
                    #[cfg(feature = "instrument")]
                    record_decline(
                        instrument::WidthDeclineReason::AxesShape,
                        leading_total_early,
                        reduction_total_early,
                        width_i64,
                        stride_a_early,
                        stride_b_early,
                    );
                    return None;
                }
            }
        }
        None => None,
    };

    // single reduction axis: unchanged, direct `layout.stride`/`extents`
    // read. Multi-axis (`attn_o`'s `[heads, head_dim]`): both operands must
    // independently compose to one constant-stride virtual axis --
    // `composed_reduction_stride`'s own doc has the proof. Declines
    // (`AxesShape`, late) rather than panicking when either operand's
    // combined extent disagrees with the other or either fails to compose,
    // which every OTHER width-fast node with `reduction_dims.len() == 1`
    // structurally cannot reach (this branch is unique to the multi-axis
    // case, gated on the same length check either arm shares).
    let (k_stride_a, k_stride_b, reduction_total) = if let [reduction_dim] = *context.reduction_dims
    {
        (
            layout_a.stride(reduction_dim),
            layout_b.stride(reduction_dim),
            context.resolved.extents[reduction_dim as usize] as usize,
        )
    } else {
        let Some((extent_a, stride_a)) =
            composed_reduction_stride(context.resolved, context.reduction_dims, layout_a)
        else {
            #[cfg(feature = "instrument")]
            record_decline(
                instrument::WidthDeclineReason::AxesShape,
                leading_total_early,
                reduction_total_early,
                width_i64,
                stride_a_early,
                stride_b_early,
            );
            return None;
        };
        let Some((extent_b, stride_b)) =
            composed_reduction_stride(context.resolved, context.reduction_dims, layout_b)
        else {
            #[cfg(feature = "instrument")]
            record_decline(
                instrument::WidthDeclineReason::AxesShape,
                leading_total_early,
                reduction_total_early,
                width_i64,
                stride_a_early,
                stride_b_early,
            );
            return None;
        };
        if extent_a != extent_b {
            #[cfg(feature = "instrument")]
            record_decline(
                instrument::WidthDeclineReason::AxesShape,
                leading_total_early,
                reduction_total_early,
                width_i64,
                stride_a_early,
                stride_b_early,
            );
            return None;
        }
        (stride_a, stride_b, extent_a as usize)
    };

    // `merged_leading` is `Some` only from the `(0, 0)` two-axis case just
    // above -- a single composed row stride/extent standing in for
    // `layout_a.stride(leading_dim)`/`context.out_layout.stride(leading_dim)`/
    // `context.resolved.extents[leading_dim]`, since no single axis index
    // names the merged (batch, seq) pair. `None` is every pre-existing
    // shape (0 or 1 non-degenerate leading axes, or the "one b-invariant,
    // one outer" two-axis case), unchanged.
    let (row_stride_a, out_row_stride, leading_total) = match merged_leading {
        Some((stride_a, stride_out, extent)) => (stride_a, stride_out, extent),
        None => (
            layout_a.stride(leading_dim),
            context.out_layout.stride(leading_dim),
            context.resolved.extents[leading_dim as usize] as usize,
        ),
    };

    Some(WidthTilePlan {
        a_operand,
        b_operand,
        row_stride_a,
        base_a: layout_a.base,
        k_stride_a,
        base_b: layout_b.base,
        k_stride_b,
        out_base: context.out_layout.base,
        out_row_stride,
        out_col_stride: context.out_layout.stride(last_output_dim),
        leading_total,
        reduction_total,
        width: context.width,
        seed: initial_value(context.init).unwrap_or(0.0),
        outer_extent: outer_info.map_or(1, |(extent, ..)| extent),
        outer_stride_a: outer_info.map_or(0, |(_, stride_a, ..)| stride_a),
        outer_stride_b: outer_info.map_or(0, |(_, _, stride_b, _)| stride_b),
        outer_stride_out: outer_info.map_or(0, |(_, _, _, stride_out)| stride_out),
        vecs,
    })
}

/// A flat `f32` operand's addressing bundle: `data` is the physical buffer,
/// `base` the flat offset of this tile's `(row 0, col 0, k 0)` corner (dot
/// path) or `(row, k)` corner (width path), `k_stride` the per-row (dot `a`),
/// per-column (dot `b`), or per-reduction-step (width path, both operands)
/// step between adjacent lanes the kernel reads — whichever axis that
/// caller's kernel actually steps by.
///
/// `i64`, not `usize`: `neon_tile_plan` proves the dot path's strides
/// non-negative, but `width_tile_plan`'s can run negative, so the one type
/// serving both carries the wider constraint. The casts this costs
/// `gemm_tile_neon` are free on aarch64 — both widths are one register, and
/// the kernels emit zero `sxtw`/`uxtw` either way.
///
/// A per-row stride is a parameter, never a field: only `gemm_width_tile_neon`
/// steps rows independently, and a field would make every other caller supply
/// a value its kernel does not read.
///
/// Bundled for the same argument-count-lint reason `OperandSpan`/`DotFold`
/// already document.
#[cfg(target_arch = "aarch64")]
pub struct KStridedTile<'a> {
    pub data: &'a [f32],
    pub base: i64,
    pub k_stride: i64,
}

/// The register-tile microkernel: `ROWS` output rows x `WIDTH_TILE_VECS`
/// `float32x4_t` vectors of output columns, folded over the whole `k`
/// reduction with `acc` living in registers throughout — the vector
/// *type*, not a plain `[f32; 4]` array, is the entire trick: a plain array
/// forces LLVM to put it in memory and spill. `out` already holds the seed
/// value on entry (`vaddq_f32` below folds `acc` into it, rather than
/// overwriting), so a caller may reuse this for a running total if that
/// shape is ever needed.
///
/// `ROWS` and `VECS` are const generics, not fixed at `WIDTH_TILE_ROWS`/
/// `WIDTH_TILE_VECS`, so the same body serves the main tile (`ROWS =
/// WIDTH_TILE_ROWS, VECS = WIDTH_TILE_VECS` in every production call site)
/// and the row-remainder variants (`ROWS = 2`, `ROWS = 1`,
/// `run_width_tile_neon`'s own greedy 2-then-1 dispatch) — the identical
/// precedent `gemm_tile_neon`'s own `const ROWS: usize` already sets for
/// the dot-path tile's row remainder. `VECS` is generic for the same reason
/// ROW 20's accumulator sweep needs it: the live register count is
/// `ROWS * VECS` accumulators + `VECS` b-vectors + 1 broadcast a-value,
/// and only a compile-time `VECS` lets a caller monomorphise that count
/// without spilling. `out` is nested (`[[f32; 4]; VECS]`, not a flat
/// `[f32; VECS * 4]`) purely so the array length depends on `VECS` alone,
/// never `VECS * 4` — a generic const parameter cannot appear in an
/// arithmetic array-length expression on stable Rust without
/// `generic_const_exprs`, so the nesting is a stable-Rust workaround, not a
/// layout change: `[[f32; 4]; VECS]` and `[f32; VECS * 4]` are bit-identical.
/// Fewer row/vec accumulators at smaller `ROWS`/`VECS` is the only
/// behavioural change; the per-`step` load/FMA structure is untouched.
///
/// # Safety
/// Caller guarantees every offset `a.base + i*a_row_stride + step*a.k_stride`
/// for `i in 0..ROWS, step in 0..k` lies within `a.data`, and every offset
/// `b.base + step*b.k_stride + v*4 + lane` for `v in 0..VECS,
/// lane in 0..4` lies within `b.data`.
#[cfg(target_arch = "aarch64")]
pub unsafe fn gemm_width_tile_neon<const ROWS: usize, const VECS: usize>(
    a: KStridedTile,
    a_row_stride: i64,
    b: KStridedTile,
    k: usize,
    out: &mut [[[f32; 4]; VECS]; ROWS],
) {
    // caller-checked: every (row, step, vec) offset below is in bounds.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); VECS]; ROWS];
        for step in 0..k {
            let step = step as i64;
            let mut bv = [vdupq_n_f32(0.0); VECS];
            for (v, lane) in bv.iter_mut().enumerate() {
                let offset = b.base + step * b.k_stride + v as i64 * 4;
                *lane = vld1q_f32(b.data.as_ptr().add(offset as usize));
            }
            for (i, row_acc) in acc.iter_mut().enumerate() {
                let offset = a.base + i as i64 * a_row_stride + step * a.k_stride;
                let value_a = *a.data.get_unchecked(offset as usize);
                for (slot, &vector_b) in row_acc.iter_mut().zip(&bv) {
                    *slot = vfmaq_n_f32(*slot, vector_b, value_a);
                }
            }
        }
        for (i, row_acc) in acc.iter().enumerate() {
            for (v, &value) in row_acc.iter().enumerate() {
                let combined = vaddq_f32(vld1q_f32(out[i][v].as_ptr()), value);
                vst1q_f32(out[i][v].as_mut_ptr(), combined);
            }
        }
    }
}

/// Law 6 (constant staging) composed with law 5 (layout commutation),
/// `docs/rewrite-algebra.md` section 6: a constant 2-D weight operand's
/// panel-packed relay, built once at [`build_static_arena_with_constants`]
/// time and read every step after. The panel width is fixed at
/// [`WIDTH_TILE_VECS`] `* 4` (not a free parameter) because that is the
/// exact stride [`gemm_width_tile_neon`]'s own `b.base + step*b.k_stride +
/// v*4` read (`cpu.rs:7848`) already walks — packing chooses `k_stride ==
/// tile_cols` so that read becomes sequential instead of the unpacked
/// layout's `k_stride == width` (one full weight row away, ROW 203's
/// first-touch-latency mechanism, `docs/discipline.md:17862`).
///
/// `data` is panel-major: panel `p`'s `k_total * tile_cols` block starts at
/// `p * k_total * tile_cols`; row `k` inside panel `p` sits at
/// `p*k_total*tile_cols + k*tile_cols`, `tile_cols` floats wide and
/// contiguous. Only `full_col_tiles = width / tile_cols` panels exist — the
/// column tail (`width % tile_cols` leftover columns) is never packed and
/// [`run_width_tile_neon`] still reads it from the unpacked buffer, exactly
/// as it does today.
///
/// Declared without an `aarch64` gate (unlike every other width-tile type)
/// purely so [`run_reduce`]'s new `packed_width` parameter can have ONE
/// signature on every target instead of a `#[cfg]`-conditional parameter
/// list — every other target's value is always `None`, unused, and this
/// struct is never populated there ([`pack_width_tile_panels`] and its
/// scan stay `aarch64`-gated).
// off-aarch64, nothing in the crate ever constructs this type -- only ever
// passed as `None`, so its fields would otherwise trip `dead_code` under
// `-D warnings`.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(super) struct PackedWidthPanels {
    pub(super) data: Vec<f32>,
    pub(super) tile_cols: usize,
    pub(super) k_total: usize,
    pub(super) full_col_tiles: usize,
    /// The `b`-operand [`Op::Input`] node this panel was packed from --
    /// [`bind_named_inputs_into_arena`]'s own invalidation check reads this
    /// to know which packed panels a rebind of THIS node makes stale. Not a
    /// new soundness mechanism bolted on top of packing: it is what lets
    /// packing stay safe when [`checkout_arena`] derives `constant_inputs`
    /// from `program` structure rather than a caller's explicit promise --
    /// see [`checkout_arena`]'s own doc.
    pub(super) source: NodeId,
}

/// `docs/discipline.md` ROW 207's own paired-bench escape, same shape as
/// [`EPILOGUE_FUSE_ENABLED`]/`set_epilogue_fuse_enabled` (ROW 186): a
/// process-wide switch [`build_packed_width_panels`] consults once per
/// [`build_static_arena_with_constants`] call (plan-build time, never a
/// hot-path branch), defaulting to `true` now that plan-time weight packing
/// is this crate's default `aarch64` build behavior. A paired bench/test
/// needing the pre-ROW-207 unpacked arm for comparison calls
/// [`set_pack_at_plan_time_enabled`] rather than rebuilding the crate under
/// a separate cargo feature.
#[cfg(target_arch = "aarch64")]
pub(super) static PACK_AT_PLAN_TIME_ENABLED: EpilogueFuseAtomicBool =
    EpilogueFuseAtomicBool::new(true);

/// Bench/test-only escape valve (see [`PACK_AT_PLAN_TIME_ENABLED`]'s own
/// doc): flips whether [`build_packed_width_panels`] packs anything,
/// process-wide, for every [`build_static_arena_with_constants`] call from
/// this point forward. Not part of this crate's taught public surface -- a
/// caller composing tensor programs never needs this; a paired bench
/// comparing the packed default against the unpacked arm does.
#[doc(hidden)]
#[cfg(target_arch = "aarch64")]
pub fn set_pack_at_plan_time_enabled(enabled: bool) {
    PACK_AT_PLAN_TIME_ENABLED.store(enabled, EpilogueFuseOrdering::Relaxed);
}

/// Packs `b_data` (the full, unpacked underlying buffer a `WidthTilePlan`'s
/// `b_operand` addresses) into [`PackedWidthPanels`], reading it with the
/// EXACT same addressing [`run_width_tile_neon`]'s unpacked column-tile loop
/// already uses (`base_b + col_tile*tile_cols + k*k_stride_b`) so the packed
/// and unpacked arms are provably reading the same source elements, just
/// writing them out in a different order — the property the bit-identity
/// test in `cpu.rs`'s own test module checks.
#[cfg(target_arch = "aarch64")]
pub(super) fn pack_width_tile_panels(
    b_data: &[f32],
    base_b: i64,
    k_stride_b: i64,
    k_total: usize,
    width: usize,
    source: NodeId,
) -> PackedWidthPanels {
    let tile_cols = WIDTH_TILE_VECS * 4;
    let full_col_tiles = width / tile_cols;
    let mut data = vec![0.0f32; full_col_tiles * k_total * tile_cols];
    for panel in 0..full_col_tiles {
        let col_start = base_b + (panel * tile_cols) as i64;
        for k in 0..k_total {
            let row_base = col_start + k as i64 * k_stride_b;
            let dst_base = panel * k_total * tile_cols + k * tile_cols;
            data[dst_base..dst_base + tile_cols]
                .copy_from_slice(&b_data[row_base as usize..row_base as usize + tile_cols]);
        }
    }
    PackedWidthPanels {
        data,
        tile_cols,
        k_total,
        full_col_tiles,
        source,
    }
}

/// Re-derives [`WidthTilePlan`] eligibility purely from `resolved` — no
/// runtime `raw`/`output` needed, since every field [`WidthPathContext`]
/// bundles ([`resolve_reduce_axis_shape`], [`body_shape`], each operand's
/// stride) is a structural property of the bound op, not its data. This is
/// what makes plan-time (bind-time, before any sentence's activation data
/// exists) eligibility detection possible at all: the SAME gate
/// [`width_tile_plan`] runs per invocation, run once here instead, against
/// the identical [`BoundOp`] the arena will run every step after.
#[cfg(target_arch = "aarch64")]
pub(super) fn width_tile_pack_candidate(resolved: &BoundOp) -> Option<WidthTilePlan> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        keep: Keep::Reduce,
        out_scatter: None,
        output_axes,
        out_layout,
        epilogue_body,
        epilogue_operands,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    // A non-identity epilogue (`bind::bind`'s own `reduce-epilogue-fusion`,
    // e.g. a sigmoid/silu gate folded onto this fold) has no renderer on the
    // packed-panel path: `run_resolved_nodes_in_arena` calls `run_reduce`
    // directly for a packed node, skipping `run_node_into`'s own
    // `apply_reduce_epilogue` call entirely, which would silently strand the
    // fold's raw value where the epilogue's result belongs. Same capability
    // rejection `is_staged_batch_eligible` already gives this shape --
    // decline here so it falls through to the always-correct `run_node_into`
    // path instead.
    if !reduce_epilogue_is_identity(epilogue_body, epilogue_operands) {
        return None;
    }
    let body = resolved.element_body();
    let shape = body_shape(body);
    let ReduceAxisShape {
        reduction_dims,
        leading_output_axes,
        last_output_dim,
        width,
        ..
    } = resolve_reduce_axis_shape(resolved, output_axes.as_slice());
    let leading_output_axes: &[u16] = &leading_output_axes;
    let strides: Vec<i64> = resolved
        .operands()
        .iter()
        .map(|(_, view, _)| last_output_dim.map_or(0, |dim| view.stride(dim)))
        .collect();
    let context = WidthPathContext {
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
    // narrow-tile task (2026-09-01): packing panels are built and sized for
    // the `VECS = WIDTH_TILE_VECS` (16-wide) tile only -- a narrow-`vecs`
    // plan's `run_width_tile_neon::<VECS>` reads `tile_cols = VECS * 4`-wide
    // strided slices from `data_b_unpacked` unconditionally (see
    // `try_run_width_tile`'s own dispatch), so declining here rather than
    // building a mismatched-width panel buffer keeps the existing packing
    // pipeline byte-for-byte unchanged for the 84 nodes it already serves.
    width_tile_plan(&context).filter(|plan| plan.vecs == WIDTH_TILE_VECS)
}

/// Scans every resolved node for a width-tile reduce whose `b` operand is
/// one of `constant_inputs`' bound [`Op::Input`] nodes with rank exactly 2
/// (the MLAS gate, `docs/rewrite-algebra.md`'s admission rule: packing
/// targets a constant 2-D weight operand, never a non-constant or
/// higher-rank one), and packs it once. Called from
/// [`build_static_arena_with_constants`] after `constant_inputs`' data is
/// already bound into `buffers`, so `buffers[b_node]` holds the real weight
/// values, not the zero-filled placeholder [`build_static_arena`] sizes
/// every input to. Training never reaches this: `proxima-autograd::train`
/// calls plain [`build_static_arena`], which always passes an empty
/// `constant_inputs` slice, so `constant_nodes` below is empty and no
/// trainable weight is ever a packing candidate -- the eligibility gate is
/// membership in the caller-named `constant_inputs` set, not a graph-shape
/// heuristic, so there is no path a mutable parameter can slip through.
#[cfg(target_arch = "aarch64")]
pub(super) fn build_packed_width_panels(
    resolved: &[BoundOp],
    shapes: &shape::Shapes,
    buffers: &[Option<Vec<f32>>],
    input_names: &[(NodeId, String)],
    constant_inputs: &[(&str, &[f32])],
) -> BTreeMap<NodeId, PackedWidthPanels> {
    if !PACK_AT_PLAN_TIME_ENABLED.load(EpilogueFuseOrdering::Relaxed) {
        return BTreeMap::new();
    }
    let constant_nodes: BTreeSet<NodeId> = constant_inputs
        .iter()
        .filter_map(|(name, _)| {
            input_names
                .iter()
                .find(|(_, candidate)| candidate == name)
                .map(|(node, _)| *node)
        })
        .collect();
    let mut packed = BTreeMap::new();
    for computed in resolved {
        let Some(plan) = width_tile_pack_candidate(computed) else {
            continue;
        };
        let (b_node, _, _) = computed.operands()[plan.b_operand];
        if !constant_nodes.contains(&b_node) || shapes.of(b_node).len() != 2 {
            continue;
        }
        let Some(b_data) = buffers[b_node.0 as usize].as_deref() else {
            continue;
        };
        packed.insert(
            computed.node,
            pack_width_tile_panels(
                b_data,
                plan.base_b,
                plan.k_stride_b,
                plan.reduction_total,
                plan.width,
                b_node,
            ),
        );
    }
    packed
}

/// A single (row, column) partial sum computed the scalar way — the
/// remainder path for a leading count or width not divisible by the tile
/// shape. Correctness-only: `profile_hot`'s 1024^3 GEMM divides evenly by
/// both `WIDTH_TILE_ROWS` and `WIDTH_TILE_VECS * 4`, so this never fires
/// there (`WIDTH_TILE_FALLBACK_ELEMENTS` proves it), but a caller with an
/// arbitrary shape still gets a correct answer.
#[cfg(target_arch = "aarch64")]
pub(super) fn width_tile_scalar_cell(a: KStridedTile, b: KStridedTile, k: usize, seed: f32) -> f32 {
    let mut acc = seed;
    let mut offset_a = a.base;
    let mut offset_b = b.base;
    for _ in 0..k {
        acc = a.data[offset_a as usize].mul_add(b.data[offset_b as usize], acc);
        offset_a += a.k_stride;
        offset_b += b.k_stride;
    }
    acc
}

/// Walks the full leading x width space in `WIDTH_TILE_ROWS x
/// (WIDTH_TILE_VECS * 4)` blocks via [`gemm_width_tile_neon`]`::<WIDTH_TILE_ROWS, WIDTH_TILE_VECS>`,
/// then covers whatever leading rows are left (`leading_total %
/// WIDTH_TILE_ROWS`, always `0..WIDTH_TILE_ROWS`) with the SAME kernel
/// monomorphised narrower — a 2-row tile, then a 1-row tile, consumed
/// greedily (`row_remainder_tile!(2)` while `>= 2` rows remain, then
/// `row_remainder_tile!(1)` for a last odd row) — mirroring the dot-path
/// tile's own `gemm_tile_neon::<const ROWS: usize>` row-remainder dispatch
/// (`cpu.rs`'s `row_remainder_tile!` macro next to `neon_tile_plan`), which
/// already proved this shape for `TILE_ROWS = 6`. Since `2` and `1` sum to
/// every non-negative integer, no leading-row count ever reaches
/// [`width_tile_scalar_cell`] any more — the true scalar fallback below
/// this function only fires for the COLUMN tail (`width % (WIDTH_TILE_VECS *
/// 4)` leftover columns inside an otherwise-tiled row-block), which no
/// row-count NEON variant can express since [`gemm_width_tile_neon`]'s own
/// output type is a fixed `WIDTH_TILE_VECS`-wide column block. Every
/// remainder loop below increments either [`WIDTH_TILE_FALLBACK_ELEMENTS`]
/// (scalar column-tail cells) or [`WIDTH_TILE_ROW_REMAINDER_ELEMENTS`]
/// (2-row/1-row NEON tile cells), so `invocations * tile_cols *
/// WIDTH_TILE_ROWS + row_remainder_elements + fallback_elements` always
/// equals `leading_total * width`, for any shape (mirrors
/// [`NEON_TILE_ROW_REMAINDER_ELEMENTS`]'s own coverage identity for the
/// dot-path tile).
#[cfg(target_arch = "aarch64")]
pub(super) fn run_width_tile_neon<const VECS: usize>(
    plan: &WidthTilePlan,
    raw: &[&[f32]],
    packed: Option<&PackedWidthPanels>,
    output: &mut [f32],
) {
    #[cfg(feature = "instrument")]
    WIDTH_TILE_GATE_PASSES.fetch_add(1, Ordering::Relaxed);

    // composition-split task (2026-09-01): whole-function wall time, read
    // first thing, committed at the tail alongside `kernel_ticks` below --
    // `record_width_tile_split_ticks`'s own doc names why this pair (fn
    // entry/exit) is the right granularity: cheap enough not to perturb a
    // function this short-lived, coarse enough to bound the surround.
    #[cfg(feature = "instrument")]
    let width_tile_fn_started = instrument::read_ticks();
    // ticks strictly inside `gemm_width_tile_neon`, summed across every call
    // this function makes (main tile loop below, plus the row-remainder
    // macro's own call site) -- accumulated locally, committed once, same
    // discipline as every other counter in this function. Read at the CALL
    // boundary, never inside the kernel's own k-loop: a ~1-2us kernel call
    // can absorb a read-pair's overhead, a per-element inner loop could not.
    #[cfg(feature = "instrument")]
    let mut width_tile_kernel_ticks = 0u64;
    // MACs those same kernel calls computed -- `ROWS * tile_cols *
    // reduction_total` per call, tracked separately from `run_reduce`'s own
    // `MAC_OPS` so the column-tail scalar fallback's MACs never mix in (see
    // `WIDTH_TILE_KERNEL_MACS`'s own doc).
    #[cfg(feature = "instrument")]
    let mut width_tile_kernel_macs = 0u64;

    // accumulated locally across the whole tile walk and committed once at
    // the end, never as a per-element atomic inside the fallback loops.
    #[cfg(feature = "instrument")]
    let mut width_tile_fallback_elements = 0u64;
    // was a `fetch_add(1)` per tile call (`row_tiles * col_tiles` times,
    // the same magnitude as `NEON_TILE_INVOCATIONS`'s historical
    // per-tile atomic) — tallied locally and committed once instead.
    #[cfg(feature = "instrument")]
    let mut width_tile_invocations = 0u64;
    #[cfg(feature = "instrument")]
    let mut width_tile_row_remainder_invocations = 0u64;
    #[cfg(feature = "instrument")]
    let mut width_tile_row_remainder_elements = 0u64;

    let data_a = raw[plan.a_operand];
    let data_b_unpacked = raw[plan.b_operand];
    let tile_cols = VECS * 4;
    let row_tiles = plan.leading_total / WIDTH_TILE_ROWS;
    let col_tiles = plan.width / tile_cols;
    // law 6∘5: only trusted when the packed panel buffer covers every full
    // column tile this call will walk — built once, at plan time, against
    // this exact node's `plan.width`, so `full_col_tiles >= col_tiles`
    // always holds when `Some`; the `filter` is defense against a stale
    // buffer from a differently-shaped node, never expected to reject here.
    let packed = packed.filter(|panels| panels.full_col_tiles >= col_tiles);

    // attention-tile task (2026-09-01): `outer_extent` walks a SECOND
    // leading axis both operands step through together (`heads`, for
    // `attn_qk`/`attn_v`) -- `1` for every node this tile already served,
    // which runs this loop exactly once at zero offset: bit-identical to
    // the pre-existing address sequence for those nodes. `local_plan`
    // shadows the parameter with `base_a`/`base_b`/`out_base` advanced by
    // the current outer step; every other field is untouched, so the walk
    // below is the SAME code, just re-based per outer step.
    for outer_step in 0..plan.outer_extent {
        let step = outer_step as i64;
        let local_plan = WidthTilePlan {
            base_a: plan.base_a + step * plan.outer_stride_a,
            base_b: plan.base_b + step * plan.outer_stride_b,
            out_base: plan.out_base + step * plan.outer_stride_out,
            ..*plan
        };
        let plan = &local_plan;

        for row_tile in 0..row_tiles {
            let row_start = row_tile * WIDTH_TILE_ROWS;
            let base_a = plan.base_a + row_start as i64 * plan.row_stride_a;
            let out_row_prefix = plan.out_base + row_start as i64 * plan.out_row_stride;

            for col_tile in 0..col_tiles {
                let col_start = col_tile * tile_cols;
                // packed: panel `col_tile`'s block, `k_stride == tile_cols`
                // (sequential); unpacked: the original strided read,
                // `k_stride == plan.k_stride_b` (one weight row apart).
                let (b_data, base_b, k_stride_b) = match packed {
                    Some(panels) => (
                        panels.data.as_slice(),
                        (col_tile * panels.k_total * panels.tile_cols) as i64,
                        panels.tile_cols as i64,
                    ),
                    None => (
                        data_b_unpacked,
                        plan.base_b + col_start as i64,
                        plan.k_stride_b,
                    ),
                };
                let mut tile_out = [[[plan.seed; 4]; VECS]; WIDTH_TILE_ROWS];

                // caller-checked: `base_a`/`base_b` plus every stride-scaled
                // offset the kernel touches stay inside `data_a`/`b_data`,
                // guaranteed by `row_tiles`/`col_tiles` only covering whole
                // tiles carved out of `plan.leading_total`/`plan.width` (packed:
                // `full_col_tiles`/`k_total` sized exactly to match).
                #[cfg(feature = "instrument")]
                let kernel_started = instrument::read_ticks();
                unsafe {
                    gemm_width_tile_neon::<WIDTH_TILE_ROWS, VECS>(
                        KStridedTile {
                            data: data_a,
                            base: base_a,
                            k_stride: plan.k_stride_a,
                        },
                        plan.row_stride_a,
                        KStridedTile {
                            data: b_data,
                            base: base_b,
                            k_stride: k_stride_b,
                        },
                        plan.reduction_total,
                        &mut tile_out,
                    );
                }
                #[cfg(feature = "instrument")]
                {
                    width_tile_kernel_ticks += instrument::elapsed_ticks(kernel_started);
                    width_tile_kernel_macs +=
                        WIDTH_TILE_ROWS as u64 * tile_cols as u64 * plan.reduction_total as u64;
                    width_tile_invocations += 1;
                }

                for (i, row) in tile_out.iter().enumerate() {
                    let row_prefix = out_row_prefix + i as i64 * plan.out_row_stride;
                    for (v, quad) in row.iter().enumerate() {
                        for (lane, &value) in quad.iter().enumerate() {
                            let position = row_prefix
                                + (col_start + v * 4 + lane) as i64 * plan.out_col_stride;
                            output[position as usize] = value;
                        }
                    }
                }
            }

            // column tail for these `WIDTH_TILE_ROWS` rows: columns past the
            // last full tile, still inside a tiled row-block.
            for col in col_tiles * tile_cols..plan.width {
                for i in 0..WIDTH_TILE_ROWS {
                    let row = row_start + i;
                    let value = width_tile_scalar_cell(
                        KStridedTile {
                            data: data_a,
                            base: plan.base_a + row as i64 * plan.row_stride_a,
                            k_stride: plan.k_stride_a,
                        },
                        KStridedTile {
                            data: data_b_unpacked,
                            base: plan.base_b + col as i64,
                            k_stride: plan.k_stride_b,
                        },
                        plan.reduction_total,
                        plan.seed,
                    );
                    let position = plan.out_base
                        + row as i64 * plan.out_row_stride
                        + col as i64 * plan.out_col_stride;
                    output[position as usize] = value;
                    #[cfg(feature = "instrument")]
                    {
                        width_tile_fallback_elements += 1;
                    }
                }
            }
        }

        // row tail: leading rows past the last full row-tile — covered by the
        // SAME NEON kernel monomorphised at `ROWS = 2` then `ROWS = 1`, greedily,
        // never the scalar path. `row_remainder_tile!` is a macro (not a
        // generic helper fn) for the identical reason `cpu.rs`'s dot-path
        // `row_remainder_tile!` is: `$rows` must be a literal so `tile_out`'s
        // array length and `gemm_width_tile_neon::<$rows, WIDTH_TILE_VECS>`'s monomorphisation
        // are both resolved at compile time, and a runtime `usize` parameter
        // could not do either. Reset per outer step (`heads`, attention-tile
        // task 2026-09-01): each step's own remainder is independent of every
        // other step's.
        let mut row_start = row_tiles * WIDTH_TILE_ROWS;
        macro_rules! width_row_remainder_tile {
            ($rows:literal) => {{
                let base_a = plan.base_a + row_start as i64 * plan.row_stride_a;
                let out_row_prefix = plan.out_base + row_start as i64 * plan.out_row_stride;

                for col_tile in 0..col_tiles {
                    let col_start = col_tile * tile_cols;
                    // same packed/unpacked selection the main tile above uses.
                    let (b_data, base_b, k_stride_b) = match packed {
                        Some(panels) => (
                            panels.data.as_slice(),
                            (col_tile * panels.k_total * panels.tile_cols) as i64,
                            panels.tile_cols as i64,
                        ),
                        None => (
                            data_b_unpacked,
                            plan.base_b + col_start as i64,
                            plan.k_stride_b,
                        ),
                    };
                    let mut tile_out = [[[plan.seed; 4]; VECS]; $rows];

                    // caller-checked: same argument `run_width_tile_neon`'s main
                    // loop above already proves for `WIDTH_TILE_ROWS` rows, just
                    // `$rows` of them starting at `row_start` — still fully
                    // inside `plan.leading_total`, since `row_start + $rows` is
                    // exactly the greedy 2-then-1 dispatch below never
                    // overshooting `plan.leading_total`.
                    #[cfg(feature = "instrument")]
                    let kernel_started = instrument::read_ticks();
                    unsafe {
                        gemm_width_tile_neon::<$rows, VECS>(
                            KStridedTile {
                                data: data_a,
                                base: base_a,
                                k_stride: plan.k_stride_a,
                            },
                            plan.row_stride_a,
                            KStridedTile {
                                data: b_data,
                                base: base_b,
                                k_stride: k_stride_b,
                            },
                            plan.reduction_total,
                            &mut tile_out,
                        );
                    }
                    #[cfg(feature = "instrument")]
                    {
                        width_tile_kernel_ticks += instrument::elapsed_ticks(kernel_started);
                        width_tile_kernel_macs +=
                            $rows as u64 * tile_cols as u64 * plan.reduction_total as u64;
                        width_tile_row_remainder_invocations += 1;
                        width_tile_row_remainder_elements += ($rows * tile_cols) as u64;
                    }

                    for (i, row) in tile_out.iter().enumerate() {
                        let row_prefix = out_row_prefix + i as i64 * plan.out_row_stride;
                        for (v, quad) in row.iter().enumerate() {
                            for (lane, &value) in quad.iter().enumerate() {
                                let position = row_prefix
                                    + (col_start + v * 4 + lane) as i64 * plan.out_col_stride;
                                output[position as usize] = value;
                            }
                        }
                    }
                }

                // column tail for these `$rows` rows — same shape as the main
                // tile's own column tail above, still genuinely scalar: no
                // row-count NEON variant covers a column count short of a full
                // `WIDTH_TILE_VECS`-wide block.
                for col in col_tiles * tile_cols..plan.width {
                    for i in 0..$rows {
                        let row = row_start + i;
                        let value = width_tile_scalar_cell(
                            KStridedTile {
                                data: data_a,
                                base: plan.base_a + row as i64 * plan.row_stride_a,
                                k_stride: plan.k_stride_a,
                            },
                            KStridedTile {
                                data: data_b_unpacked,
                                base: plan.base_b + col as i64,
                                k_stride: plan.k_stride_b,
                            },
                            plan.reduction_total,
                            plan.seed,
                        );
                        let position = plan.out_base
                            + row as i64 * plan.out_row_stride
                            + col as i64 * plan.out_col_stride;
                        output[position as usize] = value;
                        #[cfg(feature = "instrument")]
                        {
                            width_tile_fallback_elements += 1;
                        }
                    }
                }

                row_start += $rows;
            }};
        }

        // `leading_total - row_tiles * WIDTH_TILE_ROWS` is always `0..
        // WIDTH_TILE_ROWS` by construction (`row_tiles` is the floor division);
        // 2s-then-1 covers every value in that range (and, generally, any
        // non-negative remainder, not only `< WIDTH_TILE_ROWS`), so the loop
        // below always terminates with zero rows left over.
        let mut rows_remaining = plan.leading_total - row_start;
        while rows_remaining >= 2 {
            width_row_remainder_tile!(2);
            rows_remaining -= 2;
        }
        if rows_remaining == 1 {
            width_row_remainder_tile!(1);
            rows_remaining -= 1;
        }
        debug_assert_eq!(
            rows_remaining, 0,
            "width-tile row-remainder dispatch: 2s-then-1 must fully consume the remainder"
        );
        debug_assert_eq!(
            row_start, plan.leading_total,
            "width-tile row-remainder dispatch: row_start must reach leading_total exactly"
        );
    }

    #[cfg(feature = "instrument")]
    {
        WIDTH_TILE_FALLBACK_ELEMENTS.fetch_add(width_tile_fallback_elements, Ordering::Relaxed);
        WIDTH_TILE_INVOCATIONS.fetch_add(width_tile_invocations, Ordering::Relaxed);
        WIDTH_TILE_ROW_REMAINDER_INVOCATIONS
            .fetch_add(width_tile_row_remainder_invocations, Ordering::Relaxed);
        WIDTH_TILE_ROW_REMAINDER_ELEMENTS
            .fetch_add(width_tile_row_remainder_elements, Ordering::Relaxed);
        // computed once from `plan.width`/`tile_cols`, both already in
        // scope — never re-checked per iteration.
        if col_tiles * tile_cols < plan.width {
            counter!(instrument::WIDTH_TILE_COLUMN_TAIL_PRESENT, 1);
        }
        instrument::record_width_tile_split_ticks(
            width_tile_kernel_ticks,
            width_tile_kernel_macs,
            instrument::elapsed_ticks(width_tile_fn_started),
        );
    }
}

/// `run_reduce`'s single entry point into the width tile: resolves the gate
/// once, runs the whole node through [`run_width_tile_neon`] and reports
/// `true` when it applies, or leaves `output` untouched and reports `false`
/// so the caller falls back to the existing per-element width path
/// unchanged. `run_reduce`'s call site is itself aarch64-gated — every other
/// target keeps the per-element width path only, this function never exists
/// there.
#[cfg(target_arch = "aarch64")]
pub(super) fn try_run_width_tile(
    context: &WidthPathContext,
    raw: &[&[f32]],
    packed: Option<&PackedWidthPanels>,
    output: &mut [f32],
) -> bool {
    match width_tile_plan(context) {
        // narrow-tile task (2026-09-01): `plan.vecs` is a runtime value
        // selecting a compile-time `gemm_width_tile_neon` monomorphisation
        // -- resolved here, once, the same way `width_row_remainder_tile!`'s
        // `$rows` literal already resolves `ROWS`. `vecs == WIDTH_TILE_VECS`
        // is the pre-existing path, byte-for-byte: same generic argument,
        // same `packed` panels. The narrow arms never receive `packed` --
        // `width_tile_pack_candidate` never builds a panel buffer for them
        // (see its own doc), so passing one through would only ever be a
        // stale buffer from an unrelated node; `None` states that plainly
        // instead of relying on the runtime filter to catch it.
        Some(plan) if plan.vecs == WIDTH_TILE_VECS => {
            run_width_tile_neon::<WIDTH_TILE_VECS>(&plan, raw, packed, output);
            true
        }
        Some(plan) if plan.vecs == 2 => {
            run_width_tile_neon::<2>(&plan, raw, None, output);
            true
        }
        Some(plan) if plan.vecs == 1 => {
            run_width_tile_neon::<1>(&plan, raw, None, output);
            true
        }
        Some(plan) => unreachable!(
            "width_tile_vecs_for only emits vecs in {{1, 2, WIDTH_TILE_VECS}}, got {}",
            plan.vecs
        ),
        None => false,
    }
}
