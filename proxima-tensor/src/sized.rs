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
//!   `target_arch` gate.
//! - **Execution policy, currently build-time-only in practice**:
//!   [`PARALLEL_THRESHOLD`], [`OVERSUBSCRIBE`], [`SPLIT_ALIGNMENT`] are
//!   plain runtime values (not const generics) read inside `cpu.rs`'s
//!   hot-path functions (`evaluate_node_parallel`, `run_chunks_threaded`,
//!   `BoundOp::split_aligned`) -- nothing about their *type* forbids a
//!   runtime override. There is no `config.rs` surface for them yet:
//!   wiring a per-process override into those call sites is real cpu.rs
//!   surgery (new parameters threaded through several private helpers),
//!   which this session declines to do while another agent is
//!   concurrently restructuring that file's tile operands and quantized
//!   paths. [`NEON_COLUMN_PANEL_BUDGET_BYTES`] is the same policy
//!   shape but additionally `target_arch = "aarch64"`-only, so it has no
//!   build-time value on any other target to seed a cross-platform config
//!   struct from -- staying `sized`-only is not just deferred here, it is
//!   the only correct shape for an arch-conditional constant.

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

/// Below this many iteration-space elements, a nest runs the plain
/// sequential path even when `workers > 1`. Execution policy (see this
/// module's doc); currently build-time-only in practice, not by
/// necessity.
#[cfg(feature = "std")]
pub const PARALLEL_THRESHOLD: usize = 4096;

/// Chunk-count multiplier over `workers` for `run_chunks_threaded`.
/// Execution policy (see this module's doc); currently build-time-only in
/// practice, not by necessity.
#[cfg(feature = "std")]
pub const OVERSUBSCRIBE: usize = 1;

/// Row-alignment applied to every non-final chunk boundary via
/// `BoundOp::split_aligned`. Execution policy (see this module's doc);
/// currently build-time-only in practice, not by necessity.
#[cfg(feature = "std")]
pub const SPLIT_ALIGNMENT: u64 = 1;

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
pub const NEON_COLUMN_PANEL_BUDGET_BYTES: usize = 2_621_440;
