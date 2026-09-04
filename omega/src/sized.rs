//! Build-time constants -- the no_std+alloc floor's only configuration
//! surface (conflaguration's tier-2 pattern: constants ARE the config
//! below `std`; see the workspace `conflag` skill and
//! `proxima-gguf/src/sized.rs`, the pattern this mirrors).
//!
//! Two families, same split `proxima-tensor/src/sized.rs`'s own module doc
//! draws:
//!
//! - **Hardware-family fact, never a policy knob**: [`SIMD_WIDTH`]. There is
//!   no `config.rs` alongside it: a runtime override would let a caller ask
//!   for a value the hardware does not have, which is not configurability,
//!   it is a footgun.
//! - **Execution policy, build-time-configurable**: [`PACKED_ROW_BLOCK_SIMDGROUPS`]
//!   and [`COOPERATIVE_REDUCE_MIN_LEN`] (both always compiled — the
//!   row-blocked packed path and the cooperative/serial reduce split have no
//!   feature gate), `TILED_GEMM_MIN_TOKENS`, `TILED_GEMM_BLOCK_M`, `TILED_GEMM_BLOCK_N`,
//!   `TILED_GEMM_BLOCK_K` (`metal-tiled-gemm`-only, ROW 109's multi-simdgroup
//!   redesign — ports `ggml-metal.metal:6487-6489`'s `BLOCK_SIZE_M`/
//!   `BLOCK_SIZE_N`/`BLOCK_SIZE_K`). These trace to `omega-runtime.toml` via `build.rs`'s
//!   `emit_sizing_consts` (mirrors `proxima-tensor/build.rs`'s function of
//!   the same name over `proxima-tensor-runtime.toml`) and can be
//!   overridden per-build via an `OMEGA_<SECTION>_<KEY>` env var
//!   (`build.rs`'s `resolve_int`), each override consulted emitting its own
//!   `cargo:rerun-if-env-changed` line. `build.rs`'s
//!   `require_multiple_of_sixteen`/`require_divides_q4k_block`/
//!   `require_multiple_of_eight` enforce the cross-axis constraints
//!   `omega/src/msl.rs`'s `push_tiled_gemm_body` depends on.
//! - `PACKED_ROW_SPLIT_K_TARGET_SIMDGROUPS`/`PACKED_ROW_SPLIT_K_MAX_SPLIT`
//!   (`metal-q4k-split-k`-only) — the row-blocked packed matmul's split-K
//!   knobs; see `msl.rs`'s `packed_row_split_factor` and
//!   `omega-runtime.toml`'s `[packed_row_split_k]` for the measured
//!   rationale.
//! - `PACKED_ROW_SPLIT_K_MAX_ROWS` (`metal-q4k-split-k`-only) — the hard
//!   row-count ceiling above which split-K never engages regardless of the
//!   simdgroup-target arithmetic above; see `msl.rs`'s
//!   `packed_row_split_factor` and `omega-runtime.toml`'s
//!   `[packed_row_block].split_k_max_rows`.
//!
//! `msl` (this module's own crate) is alloc-tier and target-independent --
//! emission never touches a device -- so [`SIMD_WIDTH`] is visible at every
//! tier this crate has, matching `msl.rs`'s own gate. `TILED_GEMM_MIN_TOKENS`
//! is gated to `feature = "metal-tiled-gemm"` alone (no `std` requirement):
//! the tiled-GEMM eligibility check that reads it lives in `msl.rs` itself,
//! which stays alloc-tier.

include!(concat!(env!("OUT_DIR"), "/omega_sized.rs"));

/// Every lane of one Apple GPU SIMD-group -- fixed at 32 on every Apple
/// Silicon/A-series GPU family this crate targets. Not read from the
/// device at emit time: emission has no device handle, only the
/// `BoundOp`'s structure, so the width has to be a compile-time fact the
/// driver's dispatch (`crate::metal::dispatch`) is built to honor
/// unconditionally. Cannot be runtime config at any tier -- there is no
/// device query at emission time to override it against, and a value
/// other than 32 would not match the hardware `dispatch` actually runs
/// on.
pub const SIMD_WIDTH: u64 = 32;

// `COOPERATIVE_REDUCE_MIN_LEN` comes in through the `include!` above --
// `msl::reduce_is_cooperative`'s routing threshold: a `Keep::Reduce` fold
// whose reduced-axis extent is below this many elements takes the serial
// one-thread-per-output route instead of the SIMD-group cooperative fold,
// regardless of whether it would otherwise qualify (no gather, an
// associative/commutative `ScalarOp`). 0 at the `omega-runtime.toml`
// default: every reduce that qualifies otherwise stays cooperative, the
// routing every build before this key existed used. NOT 128 -- see that
// file's `[cooperative_reduce]` doc for the measured NEGATIVE result
// (`perf/short-reduce-serial-route`'s own discipline row) that kept it at
// 0: routing attention's short reduces (34/64-long) to serial made them
// ~3x slower on real hardware, memory-latency-bound kernels losing more
// from 32x-fewer threads in flight than they gained from fewer idle lanes.
// `OMEGA_COOPERATIVE_REDUCE_MIN_LEN=<n>` exercises a non-zero threshold
// per-build without editing the TOML -- the mechanism that same discipline
// row's bake-off used.

// `PACKED_ROW_BLOCK_SIMDGROUPS` comes in through the `include!` above --
// number of independent `SIMD_WIDTH`-lane SIMD-groups Metal packs into one
// threadgroup for the row-blocked packed-Q4_K/Q5_K/Q6_K matmul kernel
// (`crate::msl::push_packed_row_blocked_body`). Raising it changes only
// occupancy, never the kernel body: each simdgroup indexes off
// `[[thread_position_in_grid]]` alone (`gid / SIMD_WIDTH` as its output
// group, `gid % SIMD_WIDTH` as its lane) with no threadgroup-shared memory
// and no barrier, so `dispatchThreads:threadsPerThreadgroup:`'s documented
// global-id semantics keep the math correct at any multiple of
// `SIMD_WIDTH` -- see `crate::msl::tiled_gemm_threadgroup_width`'s doc for
// the specific invariant this constant must preserve. Measured sweep:
// `proxima-tensor/docs/discipline.md` ROW 234 -- measured NEGATIVE: no
// distinguishable win in the production single-command-buffer decode path
// at any of N in {1,2,4,8} on 0.6B (30.6-31.6 ms/token, all within ~2%
// noise), and a directional REGRESSION on 4B at N=2 (75.1 vs 63.98 ms/token,
// n=8 steady steps). Left at the TOML default of 1.
