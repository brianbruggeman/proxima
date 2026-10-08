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
//! - [`UNIFORM_CACHE_ENTRIES`] (always compiled) — capacity of
//!   `metal::UNIFORM_BUFFERS`, the content-keyed uploaded-uniform-buffer
//!   cache; see `omega-runtime.toml`'s `[spans].uniform_cache_entries` for
//!   the measured default and eviction-cost rationale.
//! - `OUTPUT_POOL_MAX_PER_BUCKET` (`metal-buffer-pool`-only) — per-bucket
//!   cap on `metal::OUTPUT_BUFFER_POOL`'s retained buffers; see
//!   `omega-runtime.toml`'s `[output_pool]`.
//! - `ARENA_TRANSIENT_CAP` (`metal-plan-stable-buffers`-only) — CARD 6.5's
//!   MG-3 kill-condition budget, in bytes; see `omega-runtime.toml`'s
//!   `[arena]`.
//! - [`WORKGROUP_SIZE`] (always compiled) — threads per workgroup every
//!   v1 WGSL kernel dispatches with; see `omega-runtime.toml`'s `[wgsl]`.
//! - `PACKED_ROW_NSG` (`metal-packed-row-nsg2`/`metal-q4k-ggml-port`-only)
//!   — simdgroups per threadgroup for the row-blocked packed path; see
//!   `omega-runtime.toml`'s `[packed_row_nsg]`.
//! - [`ATTENTION_CONTEXT_KEYS_PER_CHUNK`]/[`ATTENTION_CONTEXT_CHUNK_CAP`]
//!   (always compiled) — `crate::msl::context_chunks_for`'s divisor and
//!   ceiling for splitting cached-attention's key range across simdgroups;
//!   see `omega-runtime.toml`'s `[attention_context_chunks]`.
//! - [`ATTENTION_BLOCK_WIDTH`] (always compiled) — `crate::msl::
//!   block_width_for`'s in-block staging width for cached attention's
//!   Q·K/softmax/V loop, gated by `NumericRewrite::TreeReduce`; see
//!   `omega-runtime.toml`'s `[attention_block]`.
//! - `ATTENTION_SPLIT_KEYS_PER_SPLIT`/`ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`/
//!   `ATTENTION_SPLIT_MAX` (always compiled) — `crate::msl::splits_for`'s
//!   two divisors (small below the knee, large at or above it) and ceiling
//!   for splitting cached-attention's key range across THREADGROUPS (one
//!   level above `ATTENTION_CONTEXT_KEYS_PER_CHUNK`'s intra-threadgroup
//!   split), gated by `NumericRewrite::ContextSplitMerge`; see
//!   `omega-runtime.toml`'s `[attention_splits]`.
//! - `ATTENTION_SPLIT_KEYS_PER_SPLIT_DECODE` (always compiled) — the two-range
//!   decode split form's keys-per-threadgroup-split divisor
//!   (`crate::msl::decode_splits_for`), read only under
//!   `metal-attn-split-decode`; see `omega-runtime.toml`'s `[attention_splits]`.
//! - `ATTENTION_DECODE_KEYS_IN_FLIGHT`/`ATTENTION_DECODE_KEYS_PER_BATCH`/
//!   `ATTENTION_DECODE_KEYS_PER_SIMDGROUP`/`ATTENTION_DECODE_SIMDGROUPS_MAX`
//!   (always compiled) -- the decode split kernel's lane layout, load batch
//!   and simdgroup count (`crate::msl::decode_lanes_per_key`,
//!   `decode_keys_per_batch`, `decode_chunks_for`, `decode_simdgroup_cap`),
//!   read only under `metal-attn-split-decode`; see `omega-runtime.toml`'s
//!   `[attention_decode]`.
//! - `ATTENTION_ROWS_KEYS_PER_BLOCK`/`ATTENTION_ROWS_KEYS_PER_SPLIT`/
//!   `ATTENTION_ROWS_HEAD_DIMS_PER_SIMDGROUP`/`ATTENTION_ROWS_MIN_SIMDGROUPS`/
//!   `ATTENTION_ROWS_MMA_MIN_QUERY_ROWS`/`ATTENTION_ROWS_VECTOR_BLOCKS_PER_TILE`/
//!   `ATTENTION_ROWS_ACCUMULATOR_FRAGMENTS`/`ATTENTION_ROWS_TARGET_SIMDGROUPS`
//!   (always compiled) -- the row-tiled cached attention form's block width, split
//!   granule, simdgroup rule, smallest K, tile height, per-simdgroup register
//!   budget and split target (`crate::msl::rows_per_threadgroup`, `row_tiled_simdgroups`,
//!   `row_tiled_splits`), read only under `metal-attn-split-rows`; see
//!   `omega-runtime.toml`'s `[attention_rows]`.
//! - `SELECTION_TOP_FRACTION_MIN_ROWS` (always compiled) -- the row count at which the rank-count top-fraction expression lowers to one selection kernel; see `omega-runtime.toml`'s `[selection]`.
//! - `CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES` (always compiled) —
//!   Metal's per-threadgroup `threadgroup` memory ceiling; `crate::msl::
//!   effective_context_chunk_cap` clamps `ATTENTION_CONTEXT_CHUNK_CAP`
//!   against it per-shape so `render_cached_attention`'s
//!   `shared_m`/`shared_l`/`shared_o` declaration never exceeds what the
//!   driver will compile; see `omega-runtime.toml`'s `[cached_attention]`.
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

// `COOPERATIVE_SERIAL_BELOW_LEN` and `COOPERATIVE_SERIAL_MIN_OUTPUTS` come in
// through the `include!` above -- `msl::short_fold_prefers_serial`: a fold
// shorter than the first AND producing at least the second many outputs takes
// the serial body even though it clears `COOPERATIVE_REDUCE_MIN_LEN`. Shape,
// not length alone, decides because the 8-long MoE combine is 24 lanes idle
// across a million prefill outputs but latency-bound across a thousand decode
// outputs. 0 in either disables it. `OMEGA_COOPERATIVE_REDUCE_SERIAL_BELOW_LEN`
// and `OMEGA_COOPERATIVE_REDUCE_SERIAL_MIN_OUTPUTS` override per build.

// `COOPERATIVE_REDUCE_UNROLL` comes in through the `include!` above -- how
// many elements a cooperative fold's lane loads before folding the first of
// them (`msl::push_cooperative_reduce_body`). Fold order is unchanged, so it
// moves latency, not bits. `OMEGA_COOPERATIVE_REDUCE_UNROLL=<n>` exercises
// another value per build; see `omega-runtime.toml`'s `[cooperative_reduce]`.
// `COOPERATIVE_REDUCE_PREFETCH_REGISTERS` is the per-lane register budget a
// broadcast epilogue's operand prefetch shares across its operands.

// `BROADCAST_REDUCE_MAX_WIDTH` (only with `metal-wide-cooperative-reduce`) and
// `COOPERATIVE_REDUCE_BROADCAST_SIMD_FOLD` come in through the `include!`
// above -- the width cap and the partial-combine structure of a reduce that
// carries a broadcast epilogue (a row normalization); see
// `omega-runtime.toml`'s `[wide_cooperative_reduce]` and `[cooperative_reduce]`.
// `OMEGA_WIDE_COOPERATIVE_REDUCE_BROADCAST_MAX_WIDTH=256` and
// `OMEGA_COOPERATIVE_REDUCE_BROADCAST_SIMD_FOLD=0` rebuild the shape before
// these keys, which is the A/B control.

// `GRID_LINEAR_THREAD_LIMIT`/`GRID_MAX_THREADGROUPS_X` come in through the
// `include!` above -- `msl::grid2d_for`'s two grid-shape facts: the widest 1D
// grid a `uint gid [[thread_position_in_grid]]` kernel can address (a wider
// dispatch is silently truncated to `threads mod 2^32` by Metal, so it takes
// the flat 2D form instead), and how many threadgroups that flat form puts on
// its x axis before spilling into y. See `omega-runtime.toml`'s `[grid]`.

// `ELEMENTWISE_RECIPROCAL_MIN_ELEMENTS` comes in through the `include!` above --
// the grid size at which `msl::render_elementwise` swaps its per-axis integer
// divide for the exact float-reciprocal decode; see `omega-runtime.toml`'s
// `[elementwise]` doc for why a large grid is the case it targets.

// `COMMAND_BUFFER_FIRST_CHUNK_OPS` comes in through the `include!` above --
// the op count of a chunked decode step's first command buffer; see
// `omega-runtime.toml`'s `[command_buffer]` doc for the measured rationale.

// `UNIFORM_CACHE_ENTRIES` comes in through the `include!` above -- LRU
// capacity of `crate::metal::UNIFORM_BUFFERS`. See
// `omega-runtime.toml`'s `[spans]` doc for the measured default (57 entries
// on a plan-stable decode, well under 4096) and the eviction-cost
// rationale.

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

// `LOAD_TIME_FIT_OS_HEADROOM_BYTES`/`LOAD_TIME_FIT_ARENA_ALLOWANCE_BYTES`/
// `LOAD_TIME_FIT_ARENA_BYTES_PER_PREFILL_ROW`
// come in through the `include!` above -- `proxima-model-interop`'s
// load-time memory-fit gate's own three byte-budget constants (see
// `omega-runtime.toml`'s `[load_time_fit]` doc for what each measures and
// where the default came from). Read from this crate rather than declared
// in `proxima-model-interop` itself: `omega` is the crate every GPU-backed
// serving build already links for device facts
// (`crate::metal::system_memory_facts`), so its own sizing toml is the one
// source of truth for a byte constant the fit gate compares against those
// facts -- not a parallel default the interop crate would otherwise have to
// keep in sync by hand.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_min_rows_is_the_toml_value() {
        assert_eq!(SELECTION_TOP_FRACTION_MIN_ROWS, 256);
    }
}
