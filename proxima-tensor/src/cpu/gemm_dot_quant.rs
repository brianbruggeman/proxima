use super::*;

/// The reduction-dim fast path's fold state, bundled for the same reason
/// [`OperandSpan`] and `ScanState` are: keeps `reduce_dot_binary` under
/// clippy's argument-count lint. `len` is the contraction length (`k`'s
/// extent), `init` is the reduction's identity/seed value, `seeded` mirrors
/// [`run_reduce`]'s own `seeded` flag (whether `init` should be combined
/// into the first term or overwritten by it, per [`ReduceInit::FirstElement`]).
#[derive(Clone, Copy)]
pub(super) struct DotFold {
    pub(super) len: usize,
    pub(super) init: f32,
    pub(super) seeded: bool,
}

/// Partial-accumulator count for the contiguous dot fold
/// ([`dot_fold_multi_accumulator_binary`]/[`dot_fold_multi_accumulator_unary`]).
/// The strict left-to-right fold (`acc = reduce(acc, op(a, b))` once per
/// `k`) is a serial dependency chain LLVM cannot widen, because float
/// `+`/`*` are not associative under IEEE 754 — reordering the sum
/// changes its bit pattern (`proxima-tensor/docs/discipline.md` ROW 11).
/// Splitting the chain into `DOT_LANES` independent partial folds (one
/// per position in a `DOT_LANES`-wide `chunks_exact` block) breaks that
/// dependency: each lane's own chain is still strictly sequential (still
/// no per-lane reassociation), but the lanes run independently, so LLVM
/// can pack the common case into vector `fmul`/`fadd` and pay the
/// horizontal combine once per call instead of once per element —
/// exactly what every BLAS and ggml itself do. 4 and 8 were measured
/// head-to-head (ROW 12, `proxima-tensor/docs/discipline.md`): 8 measured
/// consistently faster (~0.337-0.349s vs ~0.352-0.354s, 1024^3
/// transposed-RHS GEMM, 5 runs each) — more independent lanes hide more
/// of the reduce's latency on this core's issue width. 8 was kept.
pub(super) use crate::sized::DOT_LANES;

/// Whether the target issues a fused multiply-add as one instruction.
/// aarch64 carries `fmla` in the base ISA; x86-64 needs FMA3. Without it
/// `f32::mul_add` becomes a libm call and is far slower than the two-op
/// form, so the specialization below must not fire.
///
/// A structural axis, not a tunable: it belongs in the build-time profile
/// alongside lane width and unroll factor once the microkernel axes land.
pub(super) const FUSED_MULTIPLY_ADD: bool =
    cfg!(target_arch = "aarch64") || cfg!(target_feature = "fma");

/// `DOT_LANES` independent partial accumulators folded with `f32::mul_add`
/// — the multiply-accumulate specialization of
/// [`dot_fold_multi_accumulator_binary`].
///
/// Rust never contracts `a * b + c` into an FMA on its own, and that is a
/// guarantee rather than a missed optimization: contraction rounds once
/// instead of twice, so it changes the result and is not a rewrite the
/// optimiser may make unasked. Measured on the pre-change binary at 1024^3:
/// `fmla.4s` = 0, `fmul.4s` = 467, `fadd.4s` = 458 — the loop vectorised
/// and then issued two instructions per multiply-accumulate. `mul_add` is
/// the explicit request.
///
/// Numerically this moves *toward* the infinitely-precise result (one
/// rounding per term, not two), so it stays inside the 1e-5 relative
/// tolerance ROW 12 already established for this fold.
#[inline(always)]
pub(super) fn dot_fold_fused_multiply_add(slice_a: &[f32], slice_b: &[f32], fold: DotFold) -> f32 {
    let (chunks_a, remainder_a) = slice_a.as_chunks::<DOT_LANES>();
    let (chunks_b, remainder_b) = slice_b.as_chunks::<DOT_LANES>();
    let mut lanes = [0.0f32; DOT_LANES];
    for (chunk_a, chunk_b) in chunks_a.iter().zip(chunks_b) {
        for ((lane, &value_a), &value_b) in lanes.iter_mut().zip(chunk_a).zip(chunk_b) {
            *lane = value_a.mul_add(value_b, *lane);
        }
    }
    let mut acc = fold.init;
    for &lane in &lanes {
        acc += lane;
    }
    for (&value_a, &value_b) in remainder_a.iter().zip(remainder_b) {
        acc = value_a.mul_add(value_b, acc);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
pub(super) use crate::sized::TILE_COLS;
/// Output rows/columns computed per call of [`gemm_tile_neon`] — ggml
/// tinyBLAS's `RM`/`RN`. Vector width (4) is implied by `float32x4_t`. An
/// iso-accumulator shape sweep at 1024^3, single-thread (CoV 0.2-0.44%, 7
/// launches each) measured 6x4 at 86.5 GFLOPS against 4x6 at 49.9 and 3x8 at
/// 48.8 — the row-heavy orientation beats its own transpose by 73% despite
/// identical accumulator count and loads/MAC. Why orientation dominates is
/// still unexplained.
#[cfg(target_arch = "aarch64")]
pub(super) use crate::sized::TILE_ROWS;

/// Bytes of L2 budgeted for a resident `b` column panel in the tiled GEMM
/// pass below. M1 Max: 12 MiB shared L2 per performance cluster of 4 cores —
/// about 3 MiB/core once every worker in the cluster streams its own panel,
/// not 12 MiB as an 8 MiB budget implicitly assumed (one worker owning the
/// whole cluster's L2). ggml's own combined panel footprint never exceeds
/// ~2.5 MiB at any size or thread count, which is also where headroom for
/// the row-strip's `a` tile, the output tile in flight, and set-associativity
/// conflicts remains without the near-fit turning into a thrash.
///
/// Swept 8/4/3/2.5/2 MiB at 512/1024/2048^3, 1/2/4/8 threads, n=9,
/// interleaved round-robin per budget, 2026-08-18, system load 1.8-3.4
/// (mostly under 3.0, one late 8-thread cell drifted to 3.37). Only the
/// 1-thread cells stayed under the 1.5% CoV resolvability bar; every
/// 2+-thread cell exceeded it (up to 20% CoV, this session's shared-host
/// contention) and is not usable for a budget comparison. Within the
/// resolvable 1-thread cells: 512^3 and 1024^3 measured flat across every
/// budget from 8 MiB down to 2 MiB (busy-per-MAC within ~1% of each other,
/// GFLOPS parity vs ggml 89.57-90.17 for 1024^3 across 8/2.5 MiB, no
/// resolvable win despite the panel becoming numerically "active" at
/// 1024^3 below ~2.8 MiB) — the hypothesis that a lower budget would help
/// 1024^3 did NOT hold up. 2048^3/1-thread did show a real, resolvable
/// effect: busy-per-mac dropped ~1.7-2% for every budget at or below 4 MiB
/// versus the 8 MiB control (0.02238 -> ~0.0220), and GFLOPS parity vs ggml
/// rose from 0.999x to 1.026x at 2.5 MiB. 4/3/2.5/2 MiB were statistically
/// indistinguishable from each other at 2048^3/1-thread (within ~0.5%, same
/// order as the noise floor) — no single value in that range measured best.
/// 2.5 MiB is landed here because it matches ggml's own measured combined
/// footprint and never measured worse than the 8 MiB control in any
/// resolvable cell; 4 MiB or 3 MiB would be an equally defensible pick on
/// this data. checksums (135.87619/260.24106/513.10425) and the 1024^3
/// allocation shape were unchanged across every budget tested.
#[cfg(target_arch = "aarch64")]
pub(super) use crate::sized::NEON_COLUMN_PANEL_BUDGET_BYTES;

/// Column-panel width for the tiled GEMM pass: the widest multiple of
/// `TILE_COLS` whose panel of `b` (`panel_cols` columns, each a contiguous
/// run of `reduction_len` `f32`s along the contraction dim) fits inside
/// [`NEON_COLUMN_PANEL_BUDGET_BYTES`]. At `reduction_len = 2048` (2048^3's
/// `k`): `2.5 MiB / (2048 * 4 bytes) = 640 -> 640` columns (rounds to a
/// `TILE_COLS` multiple exactly), five-plus panels across 2048's tiled
/// width — the cell this budget measurably helps. At `reduction_len = 1024`:
/// `2.5 MiB / 4096 bytes = 640` columns against a 1024-wide tiled output,
/// so the panel loop is numerically active (two panels, not the pre-2026-08
/// no-op) but measured flat against every other budget swept, 1-thread,
/// n=9 (`NEON_COLUMN_PANEL_BUDGET_BYTES`'s doc has the full sweep). At
/// `reduction_len = 512` the budget covers 1280 columns, wider than any
/// tiled width a 512^3 call produces, so the `clamp` below still collapses
/// to one panel spanning `tiled_width_cols` — an unconditional no-op there
/// at every budget from 8 MiB down to 2 MiB.
#[cfg(target_arch = "aarch64")]
pub(super) fn neon_column_panel_cols(reduction_len: u64, tiled_width_cols: usize) -> usize {
    let bytes_per_col = reduction_len as usize * 4;
    let budget_cols = NEON_COLUMN_PANEL_BUDGET_BYTES
        .checked_div(bytes_per_col)
        .unwrap_or(tiled_width_cols);
    let rounded = budget_cols - budget_cols % TILE_COLS;
    rounded.clamp(TILE_COLS, tiled_width_cols.max(TILE_COLS))
}

/// One bound op's applicability gate for [`gemm_tile_neon`], resolved once
/// before [`run_reduce`]'s leading-dimension loop rather than per tile. The
/// six conditions mirror attempt 2's (`proxima-tensor/docs/discipline.md`):
/// FMA available, seeded, the fused body is `Multiply` reduced by `Add`,
/// both operands gather-free, both contraction-dim strides `== 1`, and
/// exactly one operand's width-dim stride is `0` (that one is `a`, whose
/// leading-axis stride becomes `row_stride_a`) while the other is nonzero
/// (that one is `b`, whose width-dim stride becomes `col_stride_b`).
#[cfg(target_arch = "aarch64")]
pub(super) struct NeonTilePlan {
    pub(super) index_a: usize,
    pub(super) index_b: usize,
    pub(super) row_stride_a: usize,
    pub(super) col_stride_b: usize,
}

/// Runtime evidence the tile path actually ran, not just compiled: how many
/// bound ops passed [`neon_tile_plan`]'s gate, how many times
/// [`gemm_tile_neon`] was called, and how many output elements fell through
/// to the per-slot [`reduce_dot_fast`] remainder instead (row/column
/// leftovers past the last full `TILE_ROWS`x`TILE_COLS` block). Plain
/// process-wide counters, not a telemetry event, because the only consumer
/// is `profile_hot`'s one-shot report.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static NEON_TILE_GATE_PASSES: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static NEON_TILE_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static NEON_TILE_FALLBACK_ELEMENTS: AtomicU64 = AtomicU64::new(0);
/// Row-remainder tile invocations (any width `1..=5`), tracked apart from
/// [`NEON_TILE_INVOCATIONS`] since remainder tiles compute a different,
/// width-dependent number of outputs per call than the fixed-24 main tile.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static NEON_TILE_ROW_REMAINDER_INVOCATIONS: AtomicU64 = AtomicU64::new(0);
/// Output elements actually covered by row-remainder tiles, summed across
/// every width `1..=5` a run may exercise — `rows * TILE_COLS` added per
/// invocation. Unlike [`NEON_TILE_ROW_REMAINDER_INVOCATIONS`], which just
/// counts calls, this is directly usable in the coverage identity
/// (`main_invocations * 24 + row_remainder_elements + fallback == m*n`)
/// without knowing which width(s) fired.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub(super) static NEON_TILE_ROW_REMAINDER_ELEMENTS: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the three `NEON_TILE_GATE_PASSES`-family counters for the
/// main 6x4 tile: (gate passes, tile invocations, fallback elements).
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn neon_tile_counters() -> (u64, u64, u64) {
    (
        NEON_TILE_GATE_PASSES.load(Ordering::Relaxed),
        NEON_TILE_INVOCATIONS.load(Ordering::Relaxed),
        NEON_TILE_FALLBACK_ELEMENTS.load(Ordering::Relaxed),
    )
}

/// `NEON_TILE_ROW_REMAINDER_INVOCATIONS` snapshot — the row-remainder
/// tiles' own invocation count (any width `1..=5`), separate from the main
/// 6x4 tile's.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn neon_tile_row_remainder_invocations() -> u64 {
    NEON_TILE_ROW_REMAINDER_INVOCATIONS.load(Ordering::Relaxed)
}

/// `NEON_TILE_ROW_REMAINDER_ELEMENTS` snapshot — output elements covered
/// by row-remainder tiles of any width, for the `main*24 + row_remainder +
/// fallback == m*n` coverage identity.
#[cfg(all(target_arch = "aarch64", feature = "instrument"))]
pub fn neon_tile_row_remainder_elements() -> u64 {
    NEON_TILE_ROW_REMAINDER_ELEMENTS.load(Ordering::Relaxed)
}

#[cfg(target_arch = "aarch64")]
pub(super) fn neon_tile_plan(
    resolved: &BoundOp,
    shape: &BodyShape,
    reduce_op: ScalarOp,
    seeded_always: bool,
    reduction_strides: &[i64],
    strides: &[i64],
    leading_output_axes: &[u16],
) -> Option<NeonTilePlan> {
    if !FUSED_MULTIPLY_ADD
        || !seeded_always
        || leading_output_axes.len() != 1
        || reduce_op != ScalarOp::Add
    {
        return None;
    }
    let BodyShape::Binary(op, a, b) = *shape else {
        return None;
    };
    if op != ScalarOp::Multiply {
        return None;
    }
    let index_a = a as usize;
    let index_b = b as usize;
    let (_, _, gather_a) = &resolved.operands()[index_a];
    let (_, _, gather_b) = &resolved.operands()[index_b];
    if gather_a.is_some() || gather_b.is_some() {
        return None;
    }
    if reduction_strides[index_a] != 1 || reduction_strides[index_b] != 1 {
        return None;
    }
    let (index_a, index_b) = match (strides[index_a], strides[index_b]) {
        (0, other) if other != 0 => (index_a, index_b),
        (other, 0) if other != 0 => (index_b, index_a),
        _ => return None,
    };
    let row_stride_a = resolved.operands()[index_a]
        .1
        .stride(leading_output_axes[0]);
    if row_stride_a < 0 {
        return None;
    }
    // GEMM precondition this gate never checked (`docs/discipline.md` ROW
    // 561, found by direct instrumentation on qwen35's own partial-rotary
    // "new key" score term): `b`'s address advance below is `column *
    // col_stride_b` ONLY -- `gemm_tile_neon` never adds a per-row term for
    // `b` at all, because a real GEMM's right-hand operand (`[k,n]`) never
    // varies with the left-hand operand's own row axis (`m`). A grouped
    // broadcast like qwen35's `q_first_grouped`/`q_pass_grouped`
    // (`spec.rs:4847-4872`, `[s,u,g,i]`, genuinely dependent on `u` AND `g`)
    // satisfies every OTHER gate here (both reduction strides `== 1`, exactly
    // one operand's width-dim stride `0`) while still varying along the
    // leading axis -- silently reusing row 0's `base_b` for every later row.
    // Declining whenever `b` is not truly row-invariant falls through to the
    // scalar `reduction_fast_path` loop, which re-derives every operand's
    // base address from `full_coordinate` on every `leading_flat` iteration
    // and was already proven exact for this exact reduce shape (ROW 560).
    let row_stride_b = resolved.operands()[index_b]
        .1
        .stride(leading_output_axes[0]);
    if row_stride_b != 0 {
        return None;
    }
    Some(NeonTilePlan {
        index_a,
        index_b,
        row_stride_a: row_stride_a as usize,
        col_stride_b: strides[index_b] as usize,
    })
}

// docs/discipline.md ROW 188: Apple's Accelerate `cblas_sgemm` -- an AMX
// coprocessor route for exactly the GEMM shape `neon_tile_plan`'s own gate
// already isolates. Wired as a local `extern` block (no new crate
// dependency, per this workspace's `cargo add` rule) rather than a
// `blas`/`cblas-sys` crate: the only symbol this file calls is
// `cblas_sgemm` itself, and Accelerate ships in every macOS SDK, so a full
// BLAS binding crate would add surface this file never touches. Platform
// cfg, not a Cargo feature -- the same category as NEON's own
// `target_arch = "aarch64"` gates elsewhere in this file: a build-time
// execution resource, not an opt-in capability.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) const CBLAS_NO_TRANS: i32 = 111;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) const CBLAS_TRANS: i32 = 112;

/// Calls Accelerate's `cblas_sgemm` for the WHOLE `(m, n, k)` block
/// [`neon_tile_plan`]'s gate already proved gather-free and contraction-
/// contiguous on both operands, replacing the NEON 6x4 microkernel's own
/// per-tile loop below with ONE call -- the AMX coprocessor amortizes
/// tiling internally, so no analogue of `TILE_ROWS`/`TILE_COLS`/the
/// column-panel budget is needed on this route.
///
/// Both operand layouts match `neon_tile_plan`'s doc verbatim: `a` is `m x
/// k` row-major (`lda = row_stride_a`, contraction-dim stride 1 already
/// proved by the caller), `b` is stored `n x k` row-major -- the ggml
/// `mul_mat` transposed-RHS layout `run_reduce`'s own doc names -- so
/// `trans_b = CBLAS_TRANS` reads it as the conceptual `k x n` operand
/// without a repack. Returns `false` (does nothing) when the output block
/// is not a single contiguous-row-major span (`out_col_stride != 1`) or any
/// dimension overflows `i32`: a non-unit column stride would need a
/// scatter this route does not pay for yet, so the caller falls through to
/// the NEON tile unchanged.
///
/// `beta` is threaded through (not hardcoded) so [`try_run_accelerate_conv_gemm`]
/// can accumulate `outer_extent` partial products into the same `c` block --
/// `beta = 0.0` on the first step, `1.0` on every step after. The flat
/// `reduction_fast_path` caller below always passes `0.0`, unchanged from
/// this function's behavior before `beta` existed.
///
/// `transpose_b` is threaded through the same way, so
/// [`try_run_accelerate_width_gemm`] can share this call rather than
/// duplicating it: the dot-path/conv routes' own `b` is stored `[n, k]`
/// (ggml `mul_mat`'s transposed-RHS convention, `CBLAS_TRANS`), while the
/// width-tile route's `b` is already `[k, n]` (`width_tile_plan`'s own doc:
/// "the `[k,n]`-layout twin of the dot-path tile"), so it passes `false` for
/// `CBLAS_NO_TRANS` instead -- both callers below pass `true`, unchanged
/// from this function's behavior before `transpose_b` existed.
///
/// # Safety
/// Caller guarantees `a`/`b` each have at least `base + (rows-1)*row_stride
/// + (k-1)` in-bounds elements for their respective row/col strides, and
/// `c` has at least `base + (m-1)*ldc + (n-1)` in-bounds elements -- the
/// same bound `neon_tile_plan`'s own gate already established for the NEON
/// path's `raw`/`output` slices.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)] // mirrors cblas_sgemm's own C signature one-for-one
pub(super) unsafe fn try_run_accelerate_sgemm(
    a: &[f32],
    base_a: usize,
    lda: usize,
    b: &[f32],
    base_b: usize,
    ldb: usize,
    c: &mut [f32],
    base_c: usize,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
    out_col_stride: i64,
    beta: f32,
    transpose_b: bool,
) -> bool {
    if out_col_stride != 1 || m == 0 || n == 0 || k == 0 {
        return false;
    }
    let Ok(m_i32) = i32::try_from(m) else {
        return false;
    };
    let Ok(n_i32) = i32::try_from(n) else {
        return false;
    };
    let Ok(k_i32) = i32::try_from(k) else {
        return false;
    };
    let Ok(lda_i32) = i32::try_from(lda) else {
        return false;
    };
    let Ok(ldb_i32) = i32::try_from(ldb) else {
        return false;
    };
    let Ok(ldc_i32) = i32::try_from(ldc) else {
        return false;
    };
    let trans_b = if transpose_b {
        CBLAS_TRANS
    } else {
        CBLAS_NO_TRANS
    };
    // SAFETY: caller upholds this function's own `# Safety` bound; the four
    // slice-to-pointer conversions below stay in-bounds of `a`/`b`/`c` by
    // that same contract, and `cblas_sgemm` treats `a`/`b` as read-only and
    // writes only the `m x n` block of `c` starting at `base_c`.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            trans_b,
            m_i32,
            n_i32,
            k_i32,
            1.0,
            a[base_a..].as_ptr(),
            lda_i32,
            b[base_b..].as_ptr(),
            ldb_i32,
            beta,
            c[base_c..].as_mut_ptr(),
            ldc_i32,
        );
    }
    true
}

/// Routes [`WidthTilePlan`] through `cblas_sgemm` instead of
/// [`run_width_tile_neon`], behind the SAME `ACCELERATE_GEMM_ENABLED` toggle
/// ROW 188/189 already gate -- this crate's ROW 209 own pre-flight
/// (`accelerate_gemm_totals()` on a real BGE forward pass, valve ON) proved
/// `(hits, declined) == (0, 0)`: BGE's 96 MatMuls all route through
/// `try_run_width_tile` (`fast_path`), which returns before `run_reduce`
/// ever reaches the dot-tile/conv gates the valve was previously wired to
/// -- so this is the route that actually intercepts them.
///
/// `b` here is already stored `[k, n]` row-major (`width_tile_plan`'s own
/// doc: "the `[k,n]`-layout twin of the dot-path tile"), unlike the dot/conv
/// routes' `[n, k]` -- so this calls [`try_run_accelerate_sgemm`] with
/// `transpose_b = false`, reading `b` as-is.
///
/// `packed_width` (`PackedWidthPanels`) is deliberately never read here: its
/// panel-major layout exists only for `run_width_tile_neon`'s own
/// sequential-read NEON kernel and is not a valid `cblas_sgemm` operand --
/// feeding it in would silently reinterpret packed panel bytes as a `[k,n]`
/// matrix and corrupt the result. This route always reads the ORIGINAL
/// unpacked `raw[plan.b_operand]` buffer, the exact same buffer
/// `run_width_tile_neon` itself falls back to whenever `packed` is `None`
/// or a column tail is hit (`cpu.rs`'s own `pack_width_tile_panels` doc).
/// `cblas_sgemm` does its own internal blocking/packing, so no packing is
/// needed -- or valid -- on this route regardless.
///
/// Declines (returns `false`, `output` untouched) whenever any of:
/// `plan.seed != 0.0` (a non-zero seed needs a `beta = 1.0` pre-fill this
/// route does not pay for, the same residual ROW 188's own flat route
/// carries); `plan.k_stride_a != 1` (`cblas_sgemm`'s `lda` describes only a
/// ROW stride -- it has no way to express a non-unit stride WITHIN a row,
/// so `a`'s own contraction axis must already be contiguous; unlike the
/// dot path, `width_tile_plan`'s own gate never proves this -- its
/// `fast_path` premise is the WIDTH dim's stride, not the reduction dim's
/// -- so it is checked here explicitly); or any base/stride does not fit
/// `usize` (`width_tile_plan`'s own doc: its offsets can run negative,
/// unlike `neon_tile_plan`'s, so every one is validated rather than cast).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn try_run_accelerate_width_gemm(
    plan: &WidthTilePlan,
    raw: &[&[f32]],
    output: &mut [f32],
) -> bool {
    if plan.seed != 0.0 || plan.k_stride_a != 1 {
        return false;
    }
    let (
        Ok(base_a),
        Ok(row_stride_a),
        Ok(base_b),
        Ok(k_stride_b),
        Ok(out_base),
        Ok(out_row_stride),
    ) = (
        usize::try_from(plan.base_a),
        usize::try_from(plan.row_stride_a),
        usize::try_from(plan.base_b),
        usize::try_from(plan.k_stride_b),
        usize::try_from(plan.out_base),
        usize::try_from(plan.out_row_stride),
    )
    else {
        return false;
    };
    // SAFETY: `width_tile_plan`'s own gate already proves `a`/`b`
    // gather-free, `b`'s width-dim stride 0 or 1 (this route reads the
    // unpacked buffer, the same one the width-dim-1 branch of
    // `run_width_tile_neon`'s own column-tail loop reads), and `a`'s
    // contraction-dim stride 1 is checked above -- so `a`'s own
    // `leading_total x reduction_total` block is contiguous per row from
    // `base_a`, and `b`'s own `reduction_total x width` block is contiguous
    // per row from `base_b`. `output` is this function's own `&mut [f32]`
    // parameter, sized by the caller to `out_layout`'s extents, so
    // `out_base + (leading_total-1)*out_row_stride + (width-1)` stays
    // in-bounds given `plan.out_col_stride == 1` (checked inside
    // `try_run_accelerate_sgemm` itself).
    unsafe {
        try_run_accelerate_sgemm(
            raw[plan.a_operand],
            base_a,
            row_stride_a,
            raw[plan.b_operand],
            base_b,
            k_stride_b,
            output,
            out_base,
            out_row_stride,
            plan.leading_total,
            plan.width,
            plan.reduction_total,
            plan.out_col_stride,
            0.0,
            false,
        )
    }
}

/// Routes [`ConvGemmTilePlan`] through `cblas_sgemm` instead of
/// [`run_conv_gemm_tile`]'s NEON tile, behind the SAME `ACCELERATE_GEMM_ENABLED`
/// toggle ROW 188 landed. Unlike the flat `reduction_fast_path` route, neither
/// conv operand is a single contiguous `outer_extent * inner_span` span --
/// `conv_gemm_tile_plan`'s own doc: the windowed operand's `[n,c,oh,ow,kh,kw]`
/// layout puts `oh,ow` between `c` and `kh,kw`. Rather than materializing a
/// NEW packed im2col buffer (ROW 151 found materializing beats streaming, but
/// `windowed` is ALREADY the materialized buffer that finding refers to --
/// packing it a second time would duplicate that copy for no reason), this
/// calls `cblas_sgemm` once per `outer_extent` (`ci`) step directly against
/// `windowed`'s own natural strides -- each step's `K = inner_span` slice is
/// exactly `plan.col_stride_n`-contiguous per `conv_gemm_tile_plan`'s own
/// gate, the identical bound [`conv_gemm_row_block`]'s NEON loop relies on --
/// accumulating with `beta = 1.0` after the first call, mirroring that NEON
/// loop's own `tile_out` accumulation across the same `outer_extent` steps.
/// Zero heap allocation: no scratch buffer, no packing, only stack-resident
/// loop state and pointer arithmetic into the caller's existing buffers.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) fn try_run_accelerate_conv_gemm(
    plan: &ConvGemmTilePlan,
    raw: &[&[f32]],
    output: &mut [f32],
) -> bool {
    if plan.seed != 0.0
        || plan.out_row_stride < 0
        || plan.out_base < 0
        || plan.outer_stride_m < 0
        || plan.outer_stride_n < 0
    {
        return false;
    }
    let (Ok(out_base), Ok(out_row_stride), Ok(col_stride_n), Ok(row_stride_m)) = (
        usize::try_from(plan.out_base),
        usize::try_from(plan.out_row_stride),
        usize::try_from(plan.col_stride_n),
        usize::try_from(plan.row_stride_m),
    ) else {
        return false;
    };
    for step in 0..plan.outer_extent {
        let step = step as i64;
        let (Some(offset_m), Some(offset_n)) = (
            plan.outer_stride_m.checked_mul(step),
            plan.outer_stride_n.checked_mul(step),
        ) else {
            return false;
        };
        let (Some(base_m), Some(base_n)) = (
            plan.base_m.checked_add(offset_m),
            plan.base_n.checked_add(offset_n),
        ) else {
            return false;
        };
        let (Ok(base_m), Ok(base_n)) = (usize::try_from(base_m), usize::try_from(base_n)) else {
            return false;
        };
        // SAFETY: `conv_gemm_tile_plan`'s own gate already proves `inner_span`
        // contiguous elements at every `base_m + row*row_stride_m` and
        // `base_n + row*col_stride_n` this loop forms, for `m_total`/`n_total`
        // rows -- the identical bound `conv_gemm_row_block`'s NEON tile relies
        // on for the SAME `raw[plan.index_m]`/`raw[plan.index_n]` reads;
        // `output` is sized by the caller to `out_layout`'s extents, matching
        // `run_conv_gemm_tile`'s own contract for the SAME `plan`.
        let accelerated = unsafe {
            try_run_accelerate_sgemm(
                raw[plan.index_m],
                base_m,
                row_stride_m,
                raw[plan.index_n],
                base_n,
                col_stride_n,
                output,
                out_base,
                out_row_stride,
                plan.m_total,
                plan.n_total,
                plan.inner_span,
                1,
                if step == 0 { 0.0 } else { 1.0 },
                true,
            )
        };
        if !accelerated {
            return false;
        }
    }
    true
}

/// Everything [`conv_gemm_tile_plan`] needs to decide whether `Conv`'s own
/// disjoint-leading-axis reduce shape (`docs/discipline.md` ROW 148/149)
/// qualifies for the blocked 2D GEMM tile — the same field set
/// [`WidthPathContext`] bundles for its own gate, one context type per tile
/// kind rather than a single context threading fields only some tiles read.
#[cfg(target_arch = "aarch64")]
pub(super) struct ConvGemmContext<'a> {
    pub(super) resolved: &'a BoundOp,
    pub(super) shape: &'a BodyShape<'a>,
    pub(super) reduce_op: ScalarOp,
    pub(super) init: ReduceInit,
    pub(super) leading_output_axes: &'a [u16],
    pub(super) reduction_dims: &'a [u16],
    pub(super) last_output_dim: Option<u16>,
    pub(super) out_layout: &'a bind::Layout,
}

/// Everything [`run_conv_gemm_tile`]'s loop nest needs to walk, resolved
/// once per bound op. `index_m`/`index_n` name the two operands by their
/// role (`M` = the weight-shaped operand that owns exactly one leading axis
/// and nothing else; `N` = the windowed-shaped operand that owns every
/// other leading axis plus the width axis), not by which body-step slot
/// they started in — `conv_gemm_tile_plan` tries both assignments and picks
/// whichever satisfies the shape.
#[cfg(target_arch = "aarch64")]
pub(super) struct ConvGemmTilePlan {
    pub(super) index_m: usize,
    pub(super) index_n: usize,
    pub(super) base_m: i64,
    pub(super) row_stride_m: i64,
    pub(super) outer_stride_m: i64,
    pub(super) base_n: i64,
    pub(super) col_stride_n: i64,
    pub(super) outer_stride_n: i64,
    pub(super) outer_extent: u64,
    pub(super) inner_span: usize,
    pub(super) out_base: i64,
    pub(super) out_row_stride: i64,
    pub(super) m_total: usize,
    pub(super) n_total: usize,
    pub(super) seed: f32,
}

/// Row-major contiguity chain over `axes` (given outer-to-inner, walked
/// innermost-first via `.rev()`, the same convention
/// [`max_flat_reduction_suffix_len`] uses) — `Some(total_elements)` when
/// `view` addresses the WHOLE combined axis space as one contiguous
/// stride-`unit` span, `None` at the first break. An axis whose own extent
/// is `<= 1` is skipped rather than checked: its coordinate is always 0, so
/// its physical stride can never affect an address and is not informative
/// about contiguity — `Conv`'s own `n` axis (always extent 1 for every
/// shape this initiative measured) would otherwise spuriously break the
/// chain purely because `windowed`'s declared `[n,c,oh,ow,kh,kw]` layout
/// (`docs/discipline.md` ROW 148) puts a real, un-skipped `c` between `n`
/// and `oh`/`ow` — a genuine break for `n>1`, which this skip rule
/// correctly still catches (a real, nonzero-extent axis out of chain order
/// fails the `stride(dim) != expected` check exactly as before). Pure
/// stride/extent arithmetic over already-resolved [`bind::Layout`] data, no
/// NEON intrinsics — genuinely architecture-generic despite living beside
/// the `aarch64`-only [`conv_gemm_tile_plan`] that introduced it; NOT
/// `#[cfg(target_arch = "aarch64")]` because [`elementwise_rows_are_flat`]
/// (`docs/discipline.md` ROW 178) reuses it verbatim on every target.
pub(super) fn axes_flat_chain(
    resolved: &BoundOp,
    axes: &[u16],
    view: &bind::Layout,
    unit: i64,
) -> Option<u64> {
    let mut expected = unit;
    let mut count: u64 = 1;
    for &axis in axes.iter().rev() {
        let extent = resolved.extents[axis as usize];
        if extent <= 1 {
            continue;
        }
        if view.stride(axis) != expected {
            return None;
        }
        expected = expected.saturating_mul(extent as i64);
        count = count.saturating_mul(extent);
    }
    Some(count)
}

/// [`run_elementwise_range`]'s own row-flattening precondition (ROW 178):
/// true when EVERY physical operand's width-dim address, walked across the
/// WHOLE `outer_axes` odometer, composes as one contiguous
/// stride-`strides[operand]` span — [`axes_flat_chain`] reused verbatim per
/// operand with `unit = strides[operand] * inner_len`, the address one
/// outer step away landing exactly one row (`inner_len` elements, at that
/// SAME per-element stride) past the current row's own span. A stride-0
/// operand collapses `unit` to 0, which `axes_flat_chain` already treats as
/// "every nonzero-extent axis in the chain must ALSO be stride 0" — the
/// identical predicate covers a genuinely global broadcast scalar with no
/// special case. `resolved.operands()` (every physical operand, not only
/// the ones a `Generic` body's steps actually reference) is checked for
/// simplicity: an operand the body never reads cannot make this call
/// INCORRECT by being conservatively included, only cost this one
/// optimization opportunity on a pathological unread-but-non-flat operand.
pub(super) fn elementwise_rows_are_flat(
    resolved: &BoundOp,
    outer_axes: &[u16],
    strides: &[i64],
    inner_len: usize,
) -> bool {
    resolved
        .operands()
        .iter()
        .enumerate()
        .all(|(index, (_, view, _))| {
            let unit = strides[index].saturating_mul(inner_len as i64);
            axes_flat_chain(resolved, outer_axes, view, unit).is_some()
        })
}

/// Resolves [`ConvGemmTilePlan`] once per bound op, or `None` when this node
/// does not match the shape the tile is built for. `Conv`'s own shape
/// (`docs/discipline.md` ROW 148/149): `leading_output_axes` splits into one
/// axis the `M` operand (weight) alone varies over and everything else the
/// `N` operand (`windowed`) alone varies over, and `reduction_dims` splits
/// into exactly one outer axis (`ci`) neither operand can flatten away and a
/// trailing inner span (`ky,kx`) both operands read contiguously — the
/// "blocked (2-level: outer x contiguous-inner)" shape ROW 148 named as the
/// real next step and left unattempted pending this generalization.
#[cfg(target_arch = "aarch64")]
pub(super) fn conv_gemm_tile_plan(context: &ConvGemmContext) -> Option<ConvGemmTilePlan> {
    if !FUSED_MULTIPLY_ADD
        || context.reduce_op != ScalarOp::Add
        || matches!(context.init, ReduceInit::FirstElement)
    {
        return None;
    }
    let BodyShape::Binary(op, operand_a, operand_b) = *context.shape else {
        return None;
    };
    if op != ScalarOp::Multiply {
        return None;
    }
    // `leading_output_axes.len() == 1` is exactly the shape `neon_tile_plan`
    // already claims (both operands share one row axis); this tile exists
    // for the case that gate can never reach.
    if context.leading_output_axes.len() < 2 || context.reduction_dims.len() < 2 {
        return None;
    }
    let resolved = context.resolved;
    let operands = resolved.operands();
    let index_a = operand_a as usize;
    let index_b = operand_b as usize;
    let (_, view_a, gather_a) = &operands[index_a];
    let (_, view_b, gather_b) = &operands[index_b];
    if gather_a.is_some() || gather_b.is_some() {
        return None;
    }

    let suffix_a = max_flat_reduction_suffix_len(resolved, context.reduction_dims, index_a);
    let suffix_b = max_flat_reduction_suffix_len(resolved, context.reduction_dims, index_b);
    let inner_len = suffix_a.min(suffix_b);
    // `inner_len == reduction_dims.len()` is `reduction_is_fully_flat` for
    // BOTH operands at once — already `reduction_fast_path`'s own case, and
    // this function is only ever tried when that gate declined.
    if inner_len == 0 || inner_len >= context.reduction_dims.len() {
        return None;
    }
    let split = context.reduction_dims.len() - inner_len;
    let outer_dims = &context.reduction_dims[..split];
    // exactly one outer axis (`ci`) is the provably-safe case ROW 148/149
    // measured against every one of `Conv`'s 3 real folds; a wider outer
    // block is a genuinely different, larger piece of surgery (a nested
    // odometer instead of one flat `for outer_idx in 0..outer_extent` loop)
    // left for a future row rather than guessed at here.
    if outer_dims.len() != 1 {
        return None;
    }
    let outer_dim = outer_dims[0];
    let inner_dims = &context.reduction_dims[split..];
    let inner_span: u64 = inner_dims
        .iter()
        .map(|&dim| resolved.extents[dim as usize])
        .product();
    let inner_span_i64 = i64::try_from(inner_span).ok()?;

    try_conv_gemm_assignment(
        context,
        index_a,
        view_a,
        index_b,
        view_b,
        outer_dim,
        inner_span_i64,
    )
    .or_else(|| {
        try_conv_gemm_assignment(
            context,
            index_b,
            view_b,
            index_a,
            view_a,
            outer_dim,
            inner_span_i64,
        )
    })
}

/// One candidate `(M operand, N operand)` assignment for
/// [`conv_gemm_tile_plan`] — tried once per operand ordering, since the
/// fused body's own step order (`windowed * weight` vs `weight * windowed`)
/// is not guaranteed and this tile cares about roles, not slot positions.
#[cfg(target_arch = "aarch64")]
pub(super) fn try_conv_gemm_assignment(
    context: &ConvGemmContext,
    index_m: usize,
    view_m: &bind::Layout,
    index_n: usize,
    view_n: &bind::Layout,
    outer_dim: u16,
    inner_span: i64,
) -> Option<ConvGemmTilePlan> {
    let resolved = context.resolved;
    let mut m_axis = None;
    for &axis in context.leading_output_axes {
        if resolved.extents[axis as usize] <= 1 {
            continue;
        }
        match (view_m.stride(axis), view_n.stride(axis)) {
            (stride_m, 0) if stride_m != 0 => {
                if m_axis.is_some() {
                    // more than one axis the M operand alone owns: outside
                    // this tile's single-row-axis scope.
                    return None;
                }
                m_axis = Some(axis);
            }
            (0, _) => {}
            _ => return None, // shared or N-owned axis the M operand also varies over
        }
    }
    let m_axis = m_axis?;
    let row_stride_m = view_m.stride(m_axis);
    if row_stride_m < 0 {
        return None;
    }
    if let Some(width_dim) = context.last_output_dim
        && resolved.extents[width_dim as usize] > 1
        && view_m.stride(width_dim) != 0
    {
        return None;
    }

    // built once per bound op (this function runs once per `Keep::Reduce`
    // fold, never per element), the same setup-path cost `reduction_strides`
    // already pays in `run_reduce`.
    let n_axes: Vec<u16> = context
        .leading_output_axes
        .iter()
        .copied()
        .filter(|&axis| axis != m_axis)
        .chain(context.last_output_dim)
        .collect();
    if n_axes.is_empty() {
        return None;
    }
    let n_total_n = axes_flat_chain(resolved, &n_axes, view_n, inner_span)?;
    let n_total_out = axes_flat_chain(resolved, &n_axes, context.out_layout, 1)?;
    if n_total_n != n_total_out {
        return None;
    }

    let outer_stride_m = view_m.stride(outer_dim);
    let outer_stride_n = view_n.stride(outer_dim);
    if outer_stride_m < 0 || outer_stride_n < 0 {
        return None;
    }

    Some(ConvGemmTilePlan {
        index_m,
        index_n,
        base_m: view_m.base,
        row_stride_m,
        outer_stride_m,
        base_n: view_n.base,
        col_stride_n: inner_span,
        outer_stride_n,
        outer_extent: resolved.extents[outer_dim as usize],
        inner_span: inner_span as usize,
        out_base: context.out_layout.base,
        out_row_stride: context.out_layout.stride(m_axis),
        m_total: resolved.extents[m_axis as usize] as usize,
        n_total: n_total_n as usize,
        seed: initial_value(context.init).unwrap_or(0.0),
    })
}

/// Runs the whole bound op `plan` describes: an `M x N` output tile, each
/// cell folded over `outer_extent` blocks of `inner_span` contiguous
/// elements — `Conv`'s own `sum over ci of (weight[co,ci,:,:] . windowed[n,
/// ci, oy, ox, :, :])`. Reuses [`gemm_tile_neon`] entirely unchanged, called
/// once per `(M tile, N tile, outer step)` into the SAME `tile_out`
/// register array: that kernel already reads its `out` parameter's existing
/// value and adds to it (`gemm_tile_neon`'s own doc), so accumulating across
/// `outer_extent` ci-blocks needs no new kernel body, only a caller that
/// seeds `tile_out` once before the outer loop and writes it to `output`
/// once after — the exact reuse ROW 148's own "blocked" rejected-alternative
/// named as mechanically sound but blocked on this function's own gate.
#[cfg(target_arch = "aarch64")]
pub(super) fn run_conv_gemm_tile(plan: &ConvGemmTilePlan, raw: &[&[f32]], output: &mut [f32]) {
    let tiled_rows = plan.m_total - plan.m_total % TILE_ROWS;
    let mut row = 0usize;
    while row < tiled_rows {
        conv_gemm_row_block::<TILE_ROWS>(plan, raw, output, row);
        row += TILE_ROWS;
    }
    match plan.m_total - tiled_rows {
        0 => {}
        1 => conv_gemm_row_block::<1>(plan, raw, output, tiled_rows),
        2 => conv_gemm_row_block::<2>(plan, raw, output, tiled_rows),
        3 => conv_gemm_row_block::<3>(plan, raw, output, tiled_rows),
        4 => conv_gemm_row_block::<4>(plan, raw, output, tiled_rows),
        5 => conv_gemm_row_block::<5>(plan, raw, output, tiled_rows),
        _ => unreachable!("m_total - tiled_rows must be < TILE_ROWS (6) after the main tiled pass"),
    }
}

/// One `ROWS`-tall strip of [`run_conv_gemm_tile`]'s own `M x N` output,
/// generic over `ROWS` the same way [`gemm_tile_neon`] itself is: the main
/// pass monomorphises at [`TILE_ROWS`], the row-remainder pass (any leftover
/// `1..=5`) at exactly the width it needs, identical body either way. The
/// column loop mirrors `run_reduce`'s own main tile pass — a `TILE_COLS`-wide
/// NEON tile per step, then a scalar remainder for whatever `n_total %
/// TILE_COLS` leaves over (never fired for any of `Conv`'s 3 real mnist
/// folds, all `n_total` multiples of 4, but not assumed so here).
#[cfg(target_arch = "aarch64")]
pub(super) fn conv_gemm_row_block<const ROWS: usize>(
    plan: &ConvGemmTilePlan,
    raw: &[&[f32]],
    output: &mut [f32],
    row_start: usize,
) {
    let out_row_base = plan.out_base + plan.out_row_stride * row_start as i64;
    let a_row_base = plan.base_m + plan.row_stride_m * row_start as i64;
    let tiled_cols = plan.n_total - plan.n_total % TILE_COLS;

    let mut col = 0usize;
    while col < tiled_cols {
        let mut tile_out = [[plan.seed; TILE_COLS]; ROWS];
        let mut a_base = a_row_base;
        let mut b_base = plan.base_n + plan.col_stride_n * col as i64;
        for _ in 0..plan.outer_extent {
            // `conv_gemm_tile_plan`'s own gate already proved: no gathers,
            // `inner_span` contiguous elements at `a_base`/`b_base` for
            // every row and column this tile visits, `m_total`/`n_total`
            // bound every offset formed below within the source slices.
            unsafe {
                gemm_tile_neon::<ROWS>(
                    KStridedTile {
                        data: raw[plan.index_m],
                        base: a_base,
                        k_stride: plan.row_stride_m,
                    },
                    KStridedTile {
                        data: raw[plan.index_n],
                        base: b_base,
                        k_stride: plan.col_stride_n,
                    },
                    plan.inner_span,
                    &mut tile_out,
                );
            }
            a_base += plan.outer_stride_m;
            b_base += plan.outer_stride_n;
        }
        for (row, tile_row) in tile_out.iter().enumerate() {
            let out_row = out_row_base + plan.out_row_stride * row as i64;
            for (column, &value) in tile_row.iter().enumerate() {
                output[(out_row + (col + column) as i64) as usize] = value;
            }
        }
        col += TILE_COLS;
    }

    for n in tiled_cols..plan.n_total {
        for row in 0..ROWS {
            let mut a_base = a_row_base + plan.row_stride_m * row as i64;
            let mut b_base = plan.base_n + plan.col_stride_n * n as i64;
            let mut total = plan.seed;
            for _ in 0..plan.outer_extent {
                for step in 0..plan.inner_span as i64 {
                    total = raw[plan.index_m][(a_base + step) as usize]
                        .mul_add(raw[plan.index_n][(b_base + step) as usize], total);
                }
                a_base += plan.outer_stride_m;
                b_base += plan.outer_stride_n;
            }
            let out_row = out_row_base + plan.out_row_stride * row as i64;
            output[(out_row + n as i64) as usize] = total;
        }
    }
}

/// Packed bytes per `Q4_K` super-block — re-exported at this crate's own
/// name rather than spelling `proxima_gguf::quant::q4_k::BLOCK_BYTES` at
/// every call site below.
pub(super) const Q4K_BLOCK_BYTES: usize = proxima_gguf::quant::q4_k::BLOCK_BYTES;

/// Decoded `f32` elements per `Q4_K` super-block (`QK_K` in ggml/gguf
/// terms). `Q5_K` and `Q6_K` share this exact per-superblock element count
/// (both codecs' own module docs: `QK_K` is 256 crate-wide) -- only the
/// packed byte count differs per format ([`Q5K_BLOCK_BYTES`]/
/// [`Q6K_BLOCK_BYTES`] below), so this one constant covers all three rather
/// than three identical `_BLOCK_ELEMENTS` constants.
pub(super) const Q4K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_k::QK_K;

/// Packed bytes per `Q5_K` super-block — needed unconditionally (not just
/// under `q5k-int8-dot`) because [`run_reduce_quantized`]'s dispatch reads
/// it regardless of which matmul arm (dequantize-then-fold or packed int8
/// dot) actually runs.
pub(super) const Q5K_BLOCK_BYTES: usize = proxima_gguf::quant::q5_k::BLOCK_BYTES;

/// Packed bytes per `Q3_K` super-block — same reasoning as
/// [`Q5K_BLOCK_BYTES`]; `Q3_K` shares the same 256-element `QK_K` super-block
/// shape ([`Q4K_BLOCK_ELEMENTS`]) as `Q4_K`/`Q5_K`/`Q6_K`, only its packed
/// byte count differs (no per-sub-block min, a 6-bit scale-only field).
pub(super) const Q3K_BLOCK_BYTES: usize = proxima_gguf::quant::q3_k::BLOCK_BYTES;

/// Packed bytes per `Q2_K` super-block — same reasoning as
/// [`Q5K_BLOCK_BYTES`].
pub(super) const Q2K_BLOCK_BYTES: usize = proxima_gguf::quant::q2_k::BLOCK_BYTES;

/// Packed bytes per `Q6_K` super-block — same reasoning as
/// [`Q5K_BLOCK_BYTES`].
pub(super) const Q6K_BLOCK_BYTES: usize = proxima_gguf::quant::q6_k::BLOCK_BYTES;

/// One output row of a `Q4_K`-quantized-weight x `f32`-activation dot
/// product — the scalar counterpart [`reject_non_float32`]'s quantized-weight
/// exemption documents. `weight_row` is one packed weight row's raw bytes
/// (a whole number of `Q4_K` super-blocks, [`Q4K_BLOCK_BYTES`] each — not a
/// [`KStridedTile`], which only ever addresses `f32` data, never the packed
/// `u8` bytes a quantized row is stored as); `activation` is the matching
/// `f32` slice, `Q4K_BLOCK_ELEMENTS` (256) wide per block.
///
/// Dequantizes one super-block at a time into a reused stack buffer
/// (`[f32; 256]`, never a per-row or per-matrix allocation) via
/// [`proxima_gguf::quant::q4_k::dequantize_block`] — the crate's own tested
/// `Q4_K` codec, ported bit-for-bit from `ggml-quants.c` and proven against
/// real GGUF weights — then folds it against the matching activation slice.
/// This reads [`Q4K_BLOCK_BYTES`] (144) bytes per 256 weights from memory
/// rather than the 1024 bytes a pre-expanded `f32` row would cost, which is
/// the whole point: the weight matrix is never materialized as `f32`, only
/// one super-block at a time is. It stops short of ggml's own register-level
/// int4 `vec_dot` (masking/shifting nibbles straight into a SIMD multiply,
/// no `f32` intermediate at all, not even a 256-element one) — see this
/// function's caller for exactly what is and is not NEON-accelerated.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q4K_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4K_BLOCK_ELEMENTS`].
pub(super) fn dot_q4k_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q4K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q4_k block size",
        });
    }
    let block_count = weight_row.len() / Q4K_BLOCK_BYTES;
    if activation.len() != block_count * Q4K_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q4K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4K_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q4_k::dequantize_block(block, &mut scratch);
        // `DOT_LANES` (8) independent partial sums instead of one serial
        // mul_add chain -- reuses the same fold `reduce_dot_binary` already
        // uses for every f32 GEMM contraction (ROW 12, discipline.md);
        // `Q4K_BLOCK_ELEMENTS` (256) is a whole multiple of `DOT_LANES` so
        // every block folds with zero remainder.
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: Q4K_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Q4_K`-quantized weight matrix (`rows` x `k`, row-major packed
/// bytes) times one `f32` activation vector (`k` wide) — batch-1 decode's
/// actual shape, the case the module docs measure at 4.00 bytes/mac. Each
/// output row is independent, so this is the scalar fallback
/// `reject_non_float32`'s quantized-weight exemption routes to when no
/// NEON tile plan claims the node; see `dot_q4k_f32` for the per-row
/// kernel and exactly what it does and does not materialize.
///
/// # Errors
/// Propagates `dot_q4k_f32`'s [`TensorError::QuantizedShapeMismatch`] for
/// the first row that fails its shape check, or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
pub fn matmul_q4k_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q4k_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q4k_f32,
    )
}

/// Shared dispatch shape behind `matmul_q4k_f32`/`matmul_q5k_f32`/
/// `matmul_q6k_f32`: validate `rows`/`weights.len()`, then route the
/// per-row work either through [`matmul_rows_threaded`] (pool dispatch) or
/// a sequential `chunks_exact` fold, depending on
/// [`quantized_matmul_workers`]'s call. The three codecs differ only in
/// which per-row kernel (`dot_row`) they fold with and which `&'static str`
/// reasons their shape errors carry — both isolated as parameters so this
/// is the only copy of the dispatch logic itself.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] with `zero_rows_reason` if
/// `rows == 0`, with `row_length_reason` if `weights.len()` is not a whole
/// multiple of `rows`, or whatever `dot_row` itself reports for the first
/// row that fails its own shape check.
pub(super) fn matmul_quantized_dispatch<Row>(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
    zero_rows_reason: &'static str,
    row_length_reason: &'static str,
    dot_row: Row,
) -> Result<Vec<f32>, TensorError>
where
    Row: Fn(&[u8], &[f32]) -> Result<f32, TensorError> + Sync,
{
    if rows == 0 {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: zero_rows_reason,
        });
    }
    if !weights.len().is_multiple_of(rows) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: row_length_reason,
        });
    }
    let row_bytes = weights.len() / rows;
    match quantized_matmul_workers(rows, activation.len()) {
        // No shared cohort session at this call site — `matmul_quantized_dispatch`
        // backs the dequantize-then-fold codecs (`matmul_q4k_f32`/`q5k_f32`/
        // `q6k_f32`), called standalone by non-matmul consumers and tests, not
        // through `evaluate_quantized`'s per-forward session.
        Some(workers) => {
            matmul_rows_threaded(rows, 1, workers, None, activation.len(), |row, slot| {
                let start = row * row_bytes;
                slot[0] = dot_row(&weights[start..start + row_bytes], activation)?;
                Ok(())
            })
        }
        None => weights
            .chunks_exact(row_bytes)
            .map(|weight_row| dot_row(weight_row, activation))
            .collect(),
    }
}

/// [`dot_q4k_f32`]'s mechanism applied to `Q5_K`: dequantizes one
/// super-block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::q5_k::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. This is [`Codec::Q5K`]'s codec path whenever
/// `q5k-int8-dot` is off, and stays the codec path for non-matmul
/// consumers regardless.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q5K_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4K_BLOCK_ELEMENTS`].
pub(super) fn dot_q5k_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q5K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q5_k block size",
        });
    }
    let block_count = weight_row.len() / Q5K_BLOCK_BYTES;
    if activation.len() != block_count * Q4K_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q5K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4K_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q5_k::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: Q4K_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Q5_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — `dot_q5k_f32`'s per-row kernel, one row at a time
/// (no `matmul_q4k_f32`-style thread split; `Q5_K` has not yet earned that
/// on its own bench).
///
/// # Errors
/// Propagates `dot_q5k_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q5k_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    // proxima-debugger diagnostic: was ALWAYS sequential -- unlike
    // `matmul_q4k_f32`/`matmul_q4k_q8k_f32`, it never called
    // `quantized_matmul_workers`, so it was invisible to every other
    // `MATMUL_*` counter. Now routed through the same
    // `matmul_quantized_dispatch` pool dispatch as `matmul_q4k_f32`; timer
    // kept as a whole-function wrap so this counter stays comparable to the
    // pre-fix baseline it was built to measure.
    #[cfg(feature = "instrument")]
    let diag_q5k_started = instrument::read_ticks();
    let result = matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q5k_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q5k_f32,
    );
    #[cfg(feature = "instrument")]
    {
        counter!(instrument::MATMUL_Q5K_F32_CALLS, 1);
        counter!(
            instrument::MATMUL_Q5K_F32_TICKS,
            instrument::elapsed_ticks(diag_q5k_started)
        );
    }
    result
}

/// [`dot_q4k_f32`]'s mechanism applied to `Q3_K`: dequantizes one
/// super-block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::q3_k::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. `Q3_K` has no packed int8-dot kernel yet (no `q3k-int8-dot`
/// feature), so this dequantize-then-fold path is its only codec path,
/// unconditionally.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q3K_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4K_BLOCK_ELEMENTS`].
pub(super) fn dot_q3k_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q3K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q3_k block size",
        });
    }
    let block_count = weight_row.len() / Q3K_BLOCK_BYTES;
    if activation.len() != block_count * Q4K_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q3K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4K_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q3_k::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: Q4K_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Q3_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — `dot_q3k_f32`'s per-row kernel through the shared
/// `matmul_quantized_dispatch` pool dispatch, same shape as
/// [`matmul_q5k_f32`].
///
/// # Errors
/// Propagates `dot_q3k_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q3k_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q3k_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q3k_f32,
    )
}

/// [`dot_q4k_f32`]'s mechanism applied to `Q2_K`: dequantizes one
/// super-block at a time via [`proxima_gguf::quant::q2_k::dequantize_block`],
/// then folds against the matching activation slice. `Q2_K`'s super-block
/// shares the same 256-element width as every other k-quant, so this reuses
/// [`Q4K_BLOCK_ELEMENTS`] for the activation chunk width -- same shape
/// [`dot_q3k_f32`]/[`dot_q6k_f32`] already share.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q2K_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4K_BLOCK_ELEMENTS`].
pub(super) fn dot_q2k_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q2K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q2_k block size",
        });
    }
    let block_count = weight_row.len() / Q2K_BLOCK_BYTES;
    if activation.len() != block_count * Q4K_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q2K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4K_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q2_k::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: Q4K_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Q2_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — `dot_q2k_f32`'s per-row kernel through the shared
/// `matmul_quantized_dispatch` pool dispatch, same shape as
/// [`matmul_q3k_f32`]. No `q2k-int8-dot` kernel exists yet, unconditionally
/// -- same reasoning as [`matmul_q3k_f32`]'s own doc.
///
/// # Errors
/// Propagates `dot_q2k_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q2k_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q2k_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q2k_f32,
    )
}

/// [`dot_q4k_f32`]'s mechanism applied to `Q6_K`: dequantizes one
/// super-block at a time via [`proxima_gguf::quant::q6_k::dequantize_block`],
/// then folds against the matching activation slice. [`Codec::Q6K`]'s
/// codec path whenever `q6k-int8-dot` is off.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q6K_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4K_BLOCK_ELEMENTS`].
pub(super) fn dot_q6k_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q6K_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q6_k block size",
        });
    }
    let block_count = weight_row.len() / Q6K_BLOCK_BYTES;
    if activation.len() != block_count * Q4K_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4K_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q6K_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4K_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q6_k::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: Q4K_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Q6_K`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector — `dot_q6k_f32`'s per-row kernel.
///
/// # Errors
/// Propagates `dot_q6k_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q6k_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    // proxima-debugger diagnostic: see the matching note on
    // `matmul_q5k_f32` -- same was-always-sequential shape, now routed
    // through the same `matmul_quantized_dispatch` pool dispatch, same
    // whole-function timer for baseline comparability.
    #[cfg(feature = "instrument")]
    let diag_q6k_started = instrument::read_ticks();
    let result = matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q6k_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q6k_f32,
    );
    #[cfg(feature = "instrument")]
    {
        counter!(instrument::MATMUL_Q6K_F32_CALLS, 1);
        counter!(
            instrument::MATMUL_Q6K_F32_TICKS,
            instrument::elapsed_ticks(diag_q6k_started)
        );
    }
    result
}

/// Packed bytes per `Q8_0` block -- needed unconditionally, same reasoning
/// as [`Q5K_BLOCK_BYTES`].
pub(super) const Q8_0_BLOCK_BYTES: usize = proxima_gguf::quant::q8_0::BLOCK_BYTES;

/// Decoded `f32` elements per `Q8_0` block (`QK8_0`, 32) -- unlike the
/// `Q4_K`/`Q5_K`/`Q6_K` family, `Q8_0` has no shared super-block constant
/// with them; see [`Codec::Q8_0`]'s own doc for why this codec's
/// much smaller block is the one that fits the key/value context cache's
/// row width.
pub(super) const Q8_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q8_0::QK8_0;

/// [`dot_q4k_f32`]'s mechanism applied to `Q8_0`: dequantizes one 32-element
/// block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::q8_0::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. `Q8_0`'s block carries no sub-block scale structure at all -- one
/// `f16` delta per 32 elements -- so this is a direct port of `q8_0.rs`'s
/// own `dequantize_block`, not a variant of the K-quant super-block
/// unpacking `dot_q4k_f32`/`dot_q5k_f32`/`dot_q6k_f32` share.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q8_0_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q8_0_BLOCK_ELEMENTS`].
pub(super) fn dot_q8_0_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q8_0 block size",
        });
    }
    let block_count = weight_row.len() / Q8_0_BLOCK_BYTES;
    if activation.len() != block_count * Q8_0_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q8_0_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q8_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q8_0_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q8_0::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: Q8_0_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Q8_0`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- `dot_q8_0_f32`'s per-row kernel, the scalar
/// dequantize-then-fold path only (no packed int8-dot wide fold the way
/// `Q4_K`/`Q5_K`/`Q6_K` earn under their own `*-int8-dot` features): this is
/// the growable key/value context cache's storage codec, appended one call's
/// worth of new rows at a time, so the wide per-call fold those weight
/// codecs use (streamed once, reused across every batch position) does not
/// apply the same way here -- the cache itself IS the thing growing between
/// calls.
///
/// # Errors
/// Propagates `dot_q8_0_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q8_0_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q8_0_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q8_0_f32,
    )
}

/// Packed bytes per `Q4_0` block -- needed unconditionally, same reasoning
/// as [`Q8_0_BLOCK_BYTES`].
pub(super) const Q4_0_BLOCK_BYTES: usize = proxima_gguf::quant::q4_0::BLOCK_BYTES;

/// Decoded `f32` elements per `Q4_0` block (`QK4_0`, 32) -- the same flat
/// 32-element shape as [`Codec::Q8_0`], not [`Q4K_BLOCK_ELEMENTS`]'s
/// 256-wide super-block; see [`Codec::Q4_0`]'s own doc.
pub(super) const Q4_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_0::QK4_0;

/// [`dot_q8_0_f32`]'s mechanism applied to `Q4_0`: dequantizes one
/// 32-element block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::q4_0::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. `Q4_0` has no shared super-block with the K-quant family and no
/// `dot_fn_for` entry (see that function's own doc) -- this scalar
/// dequantize-then-fold path is the only one this codec takes on the CPU
/// backend.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q4_0_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q4_0_BLOCK_ELEMENTS`].
pub(super) fn dot_q4_0_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q4_0 block size",
        });
    }
    let block_count = weight_row.len() / Q4_0_BLOCK_BYTES;
    if activation.len() != block_count * Q4_0_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q4_0_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q4_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q4_0_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q4_0::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: Q4_0_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Q4_0`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- `dot_q4_0_f32`'s per-row kernel, same scalar
/// dequantize-then-fold shape as [`matmul_q8_0_f32`] (no packed int8-dot
/// wide fold exists for this codec either).
///
/// # Errors
/// Propagates `dot_q4_0_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q4_0_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q4_0_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q4_0_f32,
    )
}

/// Packed bytes per `Q5_1` block -- needed unconditionally, same reasoning
/// as [`Q4_0_BLOCK_BYTES`].
pub(super) const Q5_1_BLOCK_BYTES: usize = proxima_gguf::quant::q5_1::BLOCK_BYTES;

/// Decoded `f32` elements per `Q5_1` block (`QK5_1`, 32) -- the same flat
/// 32-element shape as [`Codec::Q4_0`]/[`Codec::Q8_0`];
/// see [`Codec::Q5_1`]'s own doc.
pub(super) const Q5_1_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q5_1::QK5_1;

/// [`dot_q4_0_f32`]'s mechanism applied to `Q5_1`: dequantizes one
/// 32-element block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::q5_1::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. `Q5_1` has no shared super-block with the K-quant family and no
/// `dot_fn_for` entry -- this plain scalar dequantize-then-fold path is the
/// only one this codec takes on the CPU backend (no int8-dot fast path
/// exists for it, same as `Q4_0`/`Q8_0`).
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q5_1_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q5_1_BLOCK_ELEMENTS`].
pub(super) fn dot_q5_1_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q5_1_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q5_1 block size",
        });
    }
    let block_count = weight_row.len() / Q5_1_BLOCK_BYTES;
    if activation.len() != block_count * Q5_1_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q5_1_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q5_1_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q5_1_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q5_1::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: Q5_1_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Q5_1`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- `dot_q5_1_f32`'s per-row kernel, same scalar
/// dequantize-then-fold shape as [`matmul_q4_0_f32`] (no packed int8-dot
/// wide fold exists for this codec either).
///
/// # Errors
/// Propagates `dot_q5_1_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q5_1_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q5_1_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q5_1_f32,
    )
}

/// Packed bytes per `Q5_0` block -- needed unconditionally, same reasoning
/// as [`Q4_0_BLOCK_BYTES`].
pub(super) const Q5_0_BLOCK_BYTES: usize = proxima_gguf::quant::q5_0::BLOCK_BYTES;

/// Decoded `f32` elements per `Q5_0` block (`QK5_0`, 32) -- the same flat
/// 32-element shape as [`Codec::Q4_0`]/[`Codec::Q5_1`];
/// see [`Codec::Q5_0`]'s own doc.
pub(super) const Q5_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q5_0::QK5_0;

/// [`dot_q5_1_f32`]'s mechanism applied to `Q5_0`: dequantizes one
/// 32-element block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::q5_0::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. `Q5_0` has no shared super-block with the K-quant family and no
/// `dot_fn_for` entry -- this plain scalar dequantize-then-fold path is the
/// only one this codec takes on the CPU backend (no int8-dot fast path
/// exists for it, same as `Q4_0`/`Q5_1`).
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`Q5_0_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`Q5_0_BLOCK_ELEMENTS`].
pub(super) fn dot_q5_0_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(Q5_0_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the q5_0 block size",
        });
    }
    let block_count = weight_row.len() / Q5_0_BLOCK_BYTES;
    if activation.len() != block_count * Q5_0_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; Q5_0_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<Q5_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<Q5_0_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::q5_0::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: Q5_0_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Q5_0`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- `dot_q5_0_f32`'s per-row kernel, same scalar
/// dequantize-then-fold shape as [`matmul_q5_1_f32`] (no packed int8-dot
/// wide fold exists for this codec either).
///
/// # Errors
/// Propagates `dot_q5_0_f32`'s [`TensorError::QuantizedShapeMismatch`], or
/// reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_q5_0_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_q5_0_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_q5_0_f32,
    )
}

/// Packed bytes per `IQ4_NL` block -- byte-identical to
/// [`Q4_0_BLOCK_BYTES`], kept as its own named constant rather than reused
/// since the two codecs decode differently (fixed recenter vs codebook
/// lookup) and a future divergence in either's block size should not
/// silently couple to the other's constant.
pub(super) const IQ4_NL_BLOCK_BYTES: usize = proxima_gguf::quant::iq4_nl::BLOCK_BYTES;

/// Decoded `f32` elements per `IQ4_NL` block (`QK4_NL`, 32) -- the same flat
/// 32-element shape as [`Codec::Q4_0`]; see
/// [`Codec::Iq4Nl`]'s own doc.
pub(super) const IQ4_NL_BLOCK_ELEMENTS: usize = proxima_gguf::quant::iq4_nl::QK4_NL;

/// [`dot_q4_0_f32`]'s mechanism applied to `IQ4_NL`: dequantizes one
/// 32-element block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::iq4_nl::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. No `dot_fn_for` entry exists for this codec (no shared int8-wide
/// fold path); this plain scalar dequantize-then-fold path is the only one
/// this codec takes on the CPU backend.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`IQ4_NL_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`IQ4_NL_BLOCK_ELEMENTS`].
pub(super) fn dot_iq4_nl_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(IQ4_NL_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the iq4_nl block size",
        });
    }
    let block_count = weight_row.len() / IQ4_NL_BLOCK_BYTES;
    if activation.len() != block_count * IQ4_NL_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; IQ4_NL_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<IQ4_NL_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<IQ4_NL_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::iq4_nl::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: IQ4_NL_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `IQ4_NL`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- `dot_iq4_nl_f32`'s per-row kernel, same scalar
/// dequantize-then-fold shape as [`matmul_q4_0_f32`] (no packed int8-dot
/// wide fold exists for this codec either).
///
/// # Errors
/// Propagates `dot_iq4_nl_f32`'s [`TensorError::QuantizedShapeMismatch`],
/// or reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_iq4_nl_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_iq4_nl_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_iq4_nl_f32,
    )
}

/// Packed bytes per `IQ2_XS` super-block -- 74 bytes for 256 elements
/// ([`Codec::Iq2Xs`]'s own doc).
pub(super) const IQ2_XS_BLOCK_BYTES: usize = proxima_gguf::quant::iq2_xs::BLOCK_BYTES;

/// Decoded `f32` elements per `IQ2_XS` super-block (`QK_K`, 256) -- the same
/// super-block width as the K-quant family, unlike `Q4_0`/`Q8_0`'s 32.
pub(super) const IQ2_XS_BLOCK_ELEMENTS: usize = proxima_gguf::quant::iq2_xs::QK_K;

/// [`dot_q4_0_f32`]'s mechanism applied to `IQ2_XS`: dequantizes one
/// 256-element super-block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::iq2_xs::dequantize_block`], then folds against the
/// matching activation slice with the same [`dot_fold_fused_multiply_add`]
/// fold. No `dot_fn_for` entry exists for this codec (no shared int8-wide
/// fold path); this plain scalar dequantize-then-fold path is the only one
/// this codec takes on the CPU backend.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`IQ2_XS_BLOCK_BYTES`], or `activation.len()` does not
/// equal the row's block count times [`IQ2_XS_BLOCK_ELEMENTS`].
pub(super) fn dot_iq2_xs_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(IQ2_XS_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the iq2_xs block size",
        });
    }
    let block_count = weight_row.len() / IQ2_XS_BLOCK_BYTES;
    if activation.len() != block_count * IQ2_XS_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; IQ2_XS_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<IQ2_XS_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<IQ2_XS_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::iq2_xs::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: IQ2_XS_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `IQ2_XS`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- `dot_iq2_xs_f32`'s per-row kernel, same scalar
/// dequantize-then-fold shape as [`matmul_q4_0_f32`] (no packed int8-dot
/// wide fold exists for this codec either).
///
/// # Errors
/// Propagates `dot_iq2_xs_f32`'s [`TensorError::QuantizedShapeMismatch`],
/// or reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_iq2_xs_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_iq2_xs_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_iq2_xs_f32,
    )
}

/// Packed bytes per `IQ3_XXS` super-block -- 98 bytes for 256 elements
/// ([`Codec::Iq3Xxs`]'s own doc).
pub(super) const IQ3_XXS_BLOCK_BYTES: usize = proxima_gguf::quant::iq3_xxs::BLOCK_BYTES;

/// Decoded `f32` elements per `IQ3_XXS` super-block (`QK_K`, 256) -- same
/// super-block width as [`IQ2_XS_BLOCK_ELEMENTS`].
pub(super) const IQ3_XXS_BLOCK_ELEMENTS: usize = proxima_gguf::quant::iq3_xxs::QK_K;

/// [`dot_iq2_xs_f32`]'s mechanism applied to `IQ3_XXS`: dequantizes one
/// 256-element super-block at a time into a reused stack buffer via
/// [`proxima_gguf::quant::iq3_xxs::dequantize_block`], then folds against
/// the matching activation slice with the same
/// [`dot_fold_fused_multiply_add`] fold. No `dot_fn_for` entry exists for
/// this codec either; this plain scalar dequantize-then-fold path is the
/// only one this codec takes on the CPU backend.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`IQ3_XXS_BLOCK_BYTES`], or `activation.len()` does
/// not equal the row's block count times [`IQ3_XXS_BLOCK_ELEMENTS`].
pub(super) fn dot_iq3_xxs_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row.len().is_multiple_of(IQ3_XXS_BLOCK_BYTES) {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "weight row length is not a whole multiple of the iq3_xxs block size",
        });
    }
    let block_count = weight_row.len() / IQ3_XXS_BLOCK_BYTES;
    if activation.len() != block_count * IQ3_XXS_BLOCK_ELEMENTS {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the weight row's decoded element count",
        });
    }

    let mut scratch = [0.0f32; IQ3_XXS_BLOCK_ELEMENTS];
    let mut acc = 0.0f32;
    for (block, activation_chunk) in weight_row
        .as_chunks::<IQ3_XXS_BLOCK_BYTES>()
        .0
        .iter()
        .zip(activation.as_chunks::<IQ3_XXS_BLOCK_ELEMENTS>().0)
    {
        proxima_gguf::quant::iq3_xxs::dequantize_block(block, &mut scratch);
        acc = dot_fold_fused_multiply_add(
            &scratch,
            activation_chunk,
            DotFold {
                len: IQ3_XXS_BLOCK_ELEMENTS,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `IQ3_XXS`-quantized weight matrix (`rows` x `k`) times one `f32`
/// activation vector -- `dot_iq3_xxs_f32`'s per-row kernel, same scalar
/// dequantize-then-fold shape as [`matmul_q4_0_f32`] (no packed int8-dot
/// wide fold exists for this codec either).
///
/// # Errors
/// Propagates `dot_iq3_xxs_f32`'s [`TensorError::QuantizedShapeMismatch`],
/// or reports the same error if `weights.len()` is not a whole multiple of
/// `rows`.
pub fn matmul_iq3_xxs_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_iq3_xxs_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_iq3_xxs_f32,
    )
}

/// Bytes per half-precision element -- both [`Codec::Float16`]
/// and [`Codec::BFloat16`] are 2-byte formats
/// ([`DType::size_bytes`] agrees for both).
pub(super) const HALF_PRECISION_ELEMENT_BYTES: usize = 2;

/// Elements converted per stack-buffer chunk in [`dot_f16_f32`]/
/// [`dot_bf16_f32`] -- reuses [`Q4K_BLOCK_ELEMENTS`] (256) for the same
/// cache/register-friendly width the K-quant kernels beside this one were
/// already measured at, not because these two unrelated formats share any
/// structural need to agree on it. A structural axis sizing a stack array,
/// not a runtime tunable -- same reasoning as `sized.rs`'s own
/// `DOT_LANES`/`WIDTH_TILE_ROWS`.
pub(super) const HALF_PRECISION_DOT_CHUNK: usize = Q4K_BLOCK_ELEMENTS;

/// One output row of a half-precision-weight x `f32`-activation dot
/// product -- composes two EXISTING primitives rather than shipping a new
/// kernel (guiding-principles §1's pipe question, answered by writing the
/// expression instead of a paragraph): [`Convert::<f16, f32>`]'s
/// [`SimdConvert::convert_slice`] widens one stack-buffer chunk of packed
/// bytes to `f32`, then [`dot_fold_fused_multiply_add`] folds it against
/// the activation slice -- the exact fold every other codec's dequantize
/// step already reuses. No half-precision-specific SIMD dot was written:
/// unlike `Q4_K`/`Q5_K`/`Q6_K`'s packed nibbles, an `f16` element carries no
/// block or scale structure to unpack, so the composed form is not a
/// stand-in for a missing fused kernel -- it is the whole job. Byte pairs
/// are widened to `f16` by hand (`u16::from_le_bytes` then `f16::from_bits`)
/// rather than an unsafe transmute of `&[u8]` to `&[f16]`. because a
/// `weight_row` sub-slice's 2-byte alignment relative to its backing
/// allocation is not a language guarantee.
///
/// # Errors
/// [`TensorError::QuantizedShapeMismatch`] if `weight_row.len()` is not a
/// whole multiple of [`HALF_PRECISION_ELEMENT_BYTES`], or `activation.len()`
/// does not equal the row's decoded element count.
pub(super) fn dot_f16_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row
        .len()
        .is_multiple_of(HALF_PRECISION_ELEMENT_BYTES)
    {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "f16 weight row length is not a whole multiple of 2 bytes",
        });
    }
    let element_count = weight_row.len() / HALF_PRECISION_ELEMENT_BYTES;
    if activation.len() != element_count {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the f16 weight row's element count",
        });
    }

    let converter = Convert::<f16, f32>::new();
    let mut half_scratch = [f16::from_bits(0); HALF_PRECISION_DOT_CHUNK];
    let mut wide_scratch = [0.0f32; HALF_PRECISION_DOT_CHUNK];
    let mut acc = 0.0f32;
    let byte_chunk_len = HALF_PRECISION_DOT_CHUNK * HALF_PRECISION_ELEMENT_BYTES;
    for (byte_chunk, activation_chunk) in weight_row
        .chunks(byte_chunk_len)
        .zip(activation.chunks(HALF_PRECISION_DOT_CHUNK))
    {
        let chunk_len = activation_chunk.len();
        for (slot, bytes) in half_scratch[..chunk_len]
            .iter_mut()
            .zip(byte_chunk.as_chunks::<HALF_PRECISION_ELEMENT_BYTES>().0)
        {
            *slot = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
        converter.convert_slice(&half_scratch[..chunk_len], &mut wide_scratch[..chunk_len]);
        acc = dot_fold_fused_multiply_add(
            &wide_scratch[..chunk_len],
            activation_chunk,
            DotFold {
                len: chunk_len,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// [`dot_f16_f32`]'s mechanism applied to `bfloat16` -- same composed
/// convert-then-fold shape, [`Convert::<bf16, f32>`] in place of
/// `Convert<f16, f32>`. See that function's doc for why no new kernel was
/// written.
///
/// # Errors
/// Same shape as [`dot_f16_f32`]'s.
pub(super) fn dot_bf16_f32(weight_row: &[u8], activation: &[f32]) -> Result<f32, TensorError> {
    if !weight_row
        .len()
        .is_multiple_of(HALF_PRECISION_ELEMENT_BYTES)
    {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "bf16 weight row length is not a whole multiple of 2 bytes",
        });
    }
    let element_count = weight_row.len() / HALF_PRECISION_ELEMENT_BYTES;
    if activation.len() != element_count {
        return Err(TensorError::QuantizedShapeMismatch {
            reason: "activation length does not match the bf16 weight row's element count",
        });
    }

    let converter = Convert::<bf16, f32>::new();
    let mut half_scratch = [bf16::from_bits(0); HALF_PRECISION_DOT_CHUNK];
    let mut wide_scratch = [0.0f32; HALF_PRECISION_DOT_CHUNK];
    let mut acc = 0.0f32;
    let byte_chunk_len = HALF_PRECISION_DOT_CHUNK * HALF_PRECISION_ELEMENT_BYTES;
    for (byte_chunk, activation_chunk) in weight_row
        .chunks(byte_chunk_len)
        .zip(activation.chunks(HALF_PRECISION_DOT_CHUNK))
    {
        let chunk_len = activation_chunk.len();
        for (slot, bytes) in half_scratch[..chunk_len]
            .iter_mut()
            .zip(byte_chunk.as_chunks::<HALF_PRECISION_ELEMENT_BYTES>().0)
        {
            *slot = bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
        }
        converter.convert_slice(&half_scratch[..chunk_len], &mut wide_scratch[..chunk_len]);
        acc = dot_fold_fused_multiply_add(
            &wide_scratch[..chunk_len],
            activation_chunk,
            DotFold {
                len: chunk_len,
                init: acc,
                seeded: true,
            },
        );
    }
    Ok(acc)
}

/// A full `Float16`-weight matrix (`rows` x `k`, row-major raw bytes) times
/// one `f32` activation vector -- `dot_f16_f32`'s per-row kernel driven
/// through the same `matmul_quantized_dispatch` every other codec shares.
///
/// # Errors
/// Propagates `dot_f16_f32`'s [`TensorError::QuantizedShapeMismatch`] for
/// the first row that fails its shape check, or reports the same error if
/// `weights.len()` is not a whole multiple of `rows`.
pub fn matmul_f16_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_f16_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_f16_f32,
    )
}

/// [`matmul_f16_f32`]'s `bfloat16` counterpart.
///
/// # Errors
/// Same shape as [`matmul_f16_f32`]'s.
pub fn matmul_bf16_f32(
    weights: &[u8],
    rows: usize,
    activation: &[f32],
) -> Result<Vec<f32>, TensorError> {
    matmul_quantized_dispatch(
        weights,
        rows,
        activation,
        "matmul_bf16_f32 called with zero rows",
        "weight byte length is not a whole multiple of the row count",
        dot_bf16_f32,
    )
}
