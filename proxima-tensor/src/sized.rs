//! Build-time constants -- the no_std+alloc floor's only configuration
//! surface (conflaguration's tier-2 pattern: constants ARE the config
//! below `std`; see the workspace `conflag` skill and
//! `proxima-gguf/src/sized.rs`, the pattern this mirrors).
//!
//! This module is visible at the alloc tier ([`bind`](mod@crate::bind) and
//! [`map`](crate::map) consult it directly), matching that pair's own
//! `#[cfg(any(feature = "std", feature = "alloc"))]` gate; the
//! `cpu`-module constants below carry an additional `feature = "std"`
//! gate because [`cpu`](crate::cpu) itself only compiles at that tier.
//!
//! Two families:
//!
//! - **Const-generic-shaped, build-time-only forever**: [`MAX_INLINE_RANK`],
//!   [`READY_BATCH_CAPACITY`], [`MAX_INLINE_TERMS`], [`TILE_ROWS`],
//!   [`TILE_COLS`], [`WIDTH_TILE_ROWS`], [`WIDTH_TILE_VECS`], [`DOT_LANES`]
//!   size a `SmallVec`/`ArrayVec` inline capacity or a fixed-size array /
//!   `const ROWS: usize` kernel parameter directly -- there is no runtime
//!   type to hold an override at any tier, exactly like
//!   `proxima-gguf::sized::MAX_DIMS`. [`GATHER_EXTENT_EXACT_FLOAT_LIMIT`]
//!   is the same shape for a different reason: it is an IEEE-754 mantissa
//!   fact (`2^24`, the largest integer an `f32` represents exactly), not a
//!   policy choice -- no target or profile ever changes it. [`DOT_LANES`]
//!   sizes the portable scalar dot-fold lane array used by every target,
//!   not just aarch64 -- see its own doc for why it carries no
//!   `target_arch` gate. This family stays hand-written literals: nothing
//!   below `std` needs a `proxima-tensor-runtime.toml`, and the values
//!   never vary by target or measurement, only by array shape.
//! - **Execution policy, build-time-configurable**: [`PARALLEL_THRESHOLD`],
//!   [`OVERSUBSCRIBE`], [`ROW_OVERSUBSCRIBE`], [`SPLIT_ALIGNMENT`],
//!   [`MIN_MACS_PER_CHUNK`], [`MIN_QUANTIZE_BLOCKS_FOR_DISPATCH`],
//!   [`MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH`], and (aarch64-only)
//!   [`NEON_COLUMN_PANEL_BUDGET_BYTES`] are plain runtime values (not const
//!   generics) read inside `cpu.rs`'s hot-path functions
//!   (`evaluate_node_parallel`, `run_chunks_threaded`,
//!   `BoundOp::split_aligned`). These now trace to
//!   `proxima-tensor-runtime.toml` via `build.rs`'s `emit_sizing_consts`
//!   (mirrors `prime/build.rs`'s `emit_sizing_consts` over
//!   `prime-runtime.toml`): the TOML holds the number and a one-line
//!   pointer, this module keeps the doc comment carrying the full
//!   measurement record (sweeps, rejected candidates, degenerate
//!   controls) since intra-doc links like `[`MIN_MACS_PER_CHUNK`]` only
//!   resolve in a Rust doc comment, not TOML. `toml` is a
//!   `[build-dependencies]`-only crate -- it runs inside `build.rs` and
//!   never links into the compiled artifact (a *runtime* conflaguration
//!   surface over these same constants was built and reverted: linking
//!   `bon` + `conflaguration` + `serde` + `toml` grew `.text` 18.7% and
//!   cost +31% end-to-end under fat LTO/`codegen-units=1` by displacing
//!   the hot path's codegen; that exposure does not exist at build time).
//!
//! There is no runtime override for these yet (unlike `prime`'s
//! `os::sizing` layer, which additionally lets `std` callers override past
//! the build-time TOML): wiring a per-process override into `cpu.rs`'s
//! call sites is real surgery (new parameters threaded through several
//! private helpers) this session declines to do while another agent is
//! concurrently restructuring that file's tile operands and quantized
//! paths.

/// Inline capacity for one bound op's per-iteration-axis buffers
/// ([`crate::bind::Layout::strides`]). Sizes a `SmallVec` const generic --
/// cannot be runtime config at any tier. See `crate::bind::MAX_INLINE_RANK`
/// (re-exported here) for the headroom rationale.
#[cfg(any(feature = "std", feature = "alloc"))]
pub const MAX_INLINE_RANK: usize = 4;

/// Capacity for one [`crate::bind::BoundOpBuilder::push`] call's ready
/// batch. Sizes an `ArrayVec` const generic
/// ([`crate::bind::ReadyBatch`]) -- cannot be runtime config at any tier.
#[cfg(any(feature = "std", feature = "alloc"))]
pub const READY_BATCH_CAPACITY: usize = 3;

/// Inline capacity for one axis's term list in the index-pattern grammar
/// ([`crate::map::AxisIndex`]). Sizes a `SmallVec` const generic -- cannot
/// be runtime config at any tier.
#[cfg(any(feature = "std", feature = "alloc"))]
pub const MAX_INLINE_TERMS: usize = 2;

/// The largest integer an `f32` can represent exactly -- its 24-bit
/// mantissa's width (`crate::shape::GATHER_EXTENT_EXACT_FLOAT_LIMIT`'s
/// bound). An IEEE-754 fact, not a policy choice -- cannot be runtime
/// config at any tier, and no build target changes it either.
#[cfg(any(feature = "std", feature = "alloc"))]
pub const GATHER_EXTENT_EXACT_FLOAT_LIMIT: u64 = 1 << 24;

/// Raw values emitted by `build.rs`'s `emit_sizing_consts` from
/// `proxima-tensor-runtime.toml` -- the pub consts below this module
/// re-export each one under its doc-commented name (see this module's own
/// doc for why the measurement record stays a Rust doc comment rather than
/// moving into the TOML). Private and `std`-gated: no execution-policy
/// const is consumed below `std` (`cpu` itself is `#[cfg(feature =
/// "std")]`), so an alloc-only build never references this module and it
/// would otherwise be flagged unused.
#[cfg(feature = "std")]
mod generated {
    include!(concat!(env!("OUT_DIR"), "/proxima_tensor_sized.rs"));
}

/// Below this many iteration-space elements, a nest runs the plain
/// sequential path even when `workers > 1`. Execution policy (see this
/// module's doc); currently build-time-only in practice, not by
/// necessity.
#[cfg(feature = "std")]
pub const PARALLEL_THRESHOLD: usize = generated::PARALLEL_THRESHOLD;

/// Chunk-count multiplier over `workers` for `run_chunks_threaded`.
/// Execution policy (see this module's doc); currently build-time-only in
/// practice, not by necessity.
#[cfg(feature = "std")]
pub const OVERSUBSCRIBE: usize = generated::OVERSUBSCRIBE;

/// Chunk-count multiplier over `workers` for `matmul_rows_threaded`'s
/// dynamic-claiming row split. A separate constant from [`OVERSUBSCRIBE`]
/// rather than a shared one: that constant's own doc records a `4` rejected
/// specifically for `run_chunks_threaded`'s `BoundOp`/GEMM-tile chunk shape
/// on an unvalidated measurement (no ambient-load record, 8-worker cells
/// never cleared their own CoV gate) -- that rejection is about the evidence
/// quality of one measurement, not a verdict against oversubscription in
/// general, and the row-loop's per-row quantized-dot chunk cost has a
/// different distribution than a GEMM tile's.
///
/// `4` is measured on this box (`proxima-tensor/src/cpu.rs`'s
/// `bench_row_oversubscribe_picks_the_multiplier`, 10 cores, ambient load
/// via `uptime` recorded with every run, 4096 rows, n=5 samples/arm across
/// two repeated runs): an imbalanced arm (last 1/8 of rows ~8x a normal
/// row's cost, echoing [`OVERSUBSCRIBE`]'s own 2.04x measured spread) went
/// 1 -> 5063-5675us (cov 0.17-0.22, the static split's own straggler-driven
/// noise) -> 2 -> 2592-3262us (cov 0.004-0.036) -> 4 -> 1657-1820us (cov
/// 0.008-0.041); 8/16/32 kept falling (down to ~1290us at 32) but with
/// diminishing, noisier steps (8's cov spiked to 0.025). A degenerate
/// control (uniform per-row cost, nothing to steal around -- isolates
/// atomic/`SyncSender` overhead from any real imbalance) improved
/// 1 -> 2304us -> 4 -> 1672us -> 8/16/32 -> 1624-1688us, i.e. flat past 4,
/// so the gain past 4 in the imbalanced arm is real oversubscription payoff
/// (matches `run_chunks_threaded`'s own `claim_and_run` mechanism) but
/// small relative to `1`'s idle-recv cost, while `chunk_count`'s
/// `clamp(1, rows)` bounds worst case for small-`rows` call sites regardless
/// of how high this constant goes. `4` is chosen over 8/16/32 as the point
/// past which both arms plateau within their own run-to-run noise, leaving
/// headroom before per-chunk `SyncSender`/atomic overhead could matter at
/// the smaller row counts this call site can see.
#[cfg(feature = "std")]
pub const ROW_OVERSUBSCRIBE: usize = generated::ROW_OVERSUBSCRIBE;

/// Row-alignment applied to every non-final chunk boundary via
/// `BoundOp::split_aligned`. Execution policy (see this module's doc);
/// currently build-time-only in practice, not by necessity.
#[cfg(feature = "std")]
pub const SPLIT_ALIGNMENT: u64 = generated::SPLIT_ALIGNMENT;

/// Floor on multiply-adds per chunk for `matmul_rows_threaded`'s row split
/// (`cpu.rs`'s `row_chunk_count`) -- caps `workers * ROW_OVERSUBSCRIBE`
/// chunks down to `(rows * contraction_width) / MIN_MACS_PER_CHUNK` when a
/// call's total work is small, instead of always paying full oversubscribed
/// dispatch. Execution policy (see this module's doc); currently
/// build-time-only in practice, not by necessity.
///
/// Measured on this box (`proxima-model-interop`'s real openchat-3.5
/// forward, `PROXIMA_PREFAULT=1`, `--features std`, 10 workers, token
/// `2651`/`"known"` held fixed every run, `uptime` recorded alongside each
/// number, 3 runs per candidate) against the fixed-40-chunk baseline's own
/// per-shape table (`DIAG q4k_shape_table`): `attn_k`/`attn_v`
/// (`rows=1024 k=4096`, 4.19M macs/call, 104,857 macs/chunk at the old
/// fixed split) was the one shape paying a fixed 40-way dispatch for 14x
/// less work than `ffn_up`/`ffn_gate` (`rows=14336 k=4096`, 58.7M
/// macs/call, 1,468,006 macs/chunk) at the same chunk count.
///
/// `500_000` (8 chunks for `attn_k`/`attn_v`, unchanged 40 for
/// `ffn_up`/`ffn_gate`): `attn_k`/`attn_v` dropped from the pre-fix
/// 0.0244-0.0254 ns/mac to 0.0180-0.0185 ns/mac (n=3), `ffn_up`/`ffn_gate`
/// unmoved at 0.0060-0.0062 ns/mac -- neither run ever left `token_id=2651`
/// (`"known"`). `700_000` (5 chunks for `attn_k`/`attn_v`, still 40 for the
/// wide shapes) was REJECTED: measured 0.0192-0.0194 ns/mac (n=3), worse
/// than `500_000`'s 8 chunks -- 5 chunks under-fills the 10-worker pool
/// (2 workers idle for the whole dispatch) where 8 chunks does not, and
/// that idle cost outweighs the smaller per-chunk overhead. `500_000` is
/// the floor between `attn_k`'s natural per-chunk work (so it drops well
/// below the old fixed 40) and `ffn_up`'s (so wide shapes are untouched,
/// still landing on `workers * ROW_OVERSUBSCRIBE`). See
/// `proxima-model-interop/src/bind.rs`'s `runs_one_real_forward_pass_and_
/// greedy_picks_a_real_token` for the harness this was measured against.
#[cfg(feature = "std")]
pub const MIN_MACS_PER_CHUNK: usize = generated::MIN_MACS_PER_CHUNK;

/// Floor on `Q8_K` super-blocks (256 elements each) before
/// `cpu.rs`'s `quantize_row_q8k_dispatch` splits a call's activation buffer
/// across the cohort instead of running it serially on the leader.
/// Execution policy (see this module's doc); currently build-time-only in
/// practice, not by necessity.
///
/// Derived against the same real openchat-3.5 forward shapes
/// [`MIN_MACS_PER_CHUNK`]'s own doc measures against (`EMBEDDING = 4096`,
/// `FEED_FORWARD = 14_336`, `leading_total = 6`): `attn_q`/`attn_o`/
/// `ffn_gate`/`ffn_up` (`k = 4096`) quantize 96 blocks/call, `ffn_down`
/// (`k = 14_336`) quantizes 336 blocks/call, `attn_k`/`attn_v` (`k = 4096`,
/// narrower `rows`) also land at 96. A cohort round costs ~32us of
/// fixed open/close overhead regardless of chunk count (`cpu.rs`'s own
/// `RowRound`/`ElementwiseRowRound` doc); at this term's measured
/// ~300ns/block (10.603ms / ~29,184 blocks across the forward, `DIAG
/// quantize_activation`), only a call clearing ~121 blocks recovers that
/// overhead across `workers - 1` idle cohort members. `200` sits above
/// every 96-block shape (left serial) and below `ffn_down`'s 336 (dispatched)
/// -- the only shape this term's real-forward measurement showed clearing
/// break-even.
#[cfg(feature = "std")]
pub const MIN_QUANTIZE_BLOCKS_FOR_DISPATCH: usize = generated::MIN_QUANTIZE_BLOCKS_FOR_DISPATCH;

/// Floor on `rows * leading_total` output elements before
/// `cpu.rs`'s `run_reduce_quantized` splits the `Q4_K` wide-fold transpose
/// copy-back across the cohort instead of running it serially on the
/// leader. Execution policy (see this module's doc); currently
/// build-time-only in practice, not by necessity.
///
/// Derived against the same real openchat-3.5 forward shapes
/// [`MIN_QUANTIZE_BLOCKS_FOR_DISPATCH`]'s own doc measures against:
/// `attn_q`/`attn_o`/`ffn_down` (`rows = 4096`, `leading_total = 6`)
/// transpose 24,576 elements/call, `attn_k`/`attn_v` (`rows = 1024`)
/// transpose 6,144, `ffn_gate`/`ffn_up` (`rows = 14_336`) transpose 86,016.
/// This term's measured ~0.635ns/element (5.247ms / ~8,257,536 elements
/// across the forward, `DIAG q4k transpose`) puts break-even (recovering
/// the same ~32us round overhead across `workers - 1` idle members) at
/// ~57,600 elements -- only `ffn_gate`/`ffn_up`'s 86,016 clears it.
/// `64_000` sits above every other shape (left serial) and below
/// `ffn_gate`/`ffn_up`'s 86,016 (dispatched).
#[cfg(feature = "std")]
pub const MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH: usize = generated::MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH;

/// Output rows one call to `gemm_width_tile_neon` computes. Sizes a fixed
/// `[[f32; _]; WIDTH_TILE_ROWS]` output array -- cannot be runtime config
/// at any tier, aarch64-only.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub const WIDTH_TILE_ROWS: usize = 4;

/// `float32x4_t` vectors of output columns one call to
/// `gemm_width_tile_neon` computes. Sizes a fixed array dimension --
/// cannot be runtime config at any tier, aarch64-only.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub const WIDTH_TILE_VECS: usize = 4;

/// Partial-accumulator count for the contiguous dot fold. Sizes a fixed
/// `[f32; DOT_LANES]` lane array -- cannot be runtime config at any tier.
/// Portable, not aarch64-only: `dot_fold_multi_accumulator_binary` and
/// `dot_fold_multi_accumulator_unary` (the functions this sizes) are the
/// scalar fold path every target compiles, carrying no `target_arch`
/// gate themselves, so this constant cannot carry one either.
#[cfg(feature = "std")]
pub const DOT_LANES: usize = 8;

/// Spin budget, in `core::hint::spin_loop()` polls, a cohort member burns
/// waiting for the next round to open before it parks. prime's own default
/// is 2000, sized for a cohort whose rounds are far apart. A forward pass
/// opens rounds back to back -- four call sites (`cpu`'s elementwise,
/// transpose, matmul-rows and quantize dispatches) -- separated only by the
/// leader's serial work, so a member that parks pays a futex wake the spin
/// would have avoided. Held at prime's default until the A/B measures.
#[cfg(feature = "std")]
pub const COHORT_SPIN_POLLS: u32 = generated::COHORT_SPIN_POLLS;

/// Output rows computed per call of `gemm_tile_neon` -- ggml tinyBLAS's
/// `RM`. Instantiates a `const ROWS: usize` kernel generic
/// (`gemm_tile_neon::<TILE_ROWS>`) -- cannot be runtime config at any
/// tier, exactly like `proxima-gguf::sized::MAX_DIMS`.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub const TILE_ROWS: usize = 6;

/// Output columns computed per call of `gemm_tile_neon` -- ggml
/// tinyBLAS's `RN`. Vector width (4) is implied by `float32x4_t`; sizes a
/// fixed array dimension used throughout the tiled GEMM pass -- cannot be
/// runtime config at any tier.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub const TILE_COLS: usize = 4;

/// Bytes of L2 budgeted for a resident `b` column panel in the tiled GEMM
/// pass. Execution policy in shape, but `target_arch = "aarch64"`-only:
/// no other target has a build-time value for this constant to seed a
/// cross-platform runtime config struct from, so it stays `sized`-only.
#[cfg(all(feature = "std", target_arch = "aarch64"))]
pub const NEON_COLUMN_PANEL_BUDGET_BYTES: usize = generated::NEON_COLUMN_PANEL_BUDGET_BYTES;

/// Asserts every execution-policy const still equals the value on record
/// (this module's own doc comments) after `build.rs`'s
/// `emit_sizing_consts` started sourcing them from
/// `proxima-tensor-runtime.toml` -- catches a TOML/doc-comment drift that
/// the type system cannot: nothing stops someone editing one file and not
/// the other.
#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use super::*;

    #[test]
    fn generated_consts_match_the_measurement_record() {
        assert_eq!(PARALLEL_THRESHOLD, 4096);
        assert_eq!(OVERSUBSCRIBE, 1);
        assert_eq!(ROW_OVERSUBSCRIBE, 4);
        assert_eq!(SPLIT_ALIGNMENT, 1);
        assert_eq!(MIN_MACS_PER_CHUNK, 500_000);
        assert_eq!(MIN_QUANTIZE_BLOCKS_FOR_DISPATCH, 200);
        assert_eq!(MIN_TRANSPOSE_ELEMENTS_FOR_DISPATCH, 64_000);
        assert_eq!(COHORT_SPIN_POLLS, 2_000);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_column_panel_budget_matches_the_measurement_record() {
        assert_eq!(NEON_COLUMN_PANEL_BUDGET_BYTES, 2_621_440);
    }
}
