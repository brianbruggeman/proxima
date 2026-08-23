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
//! - **Execution policy, build-time-configurable**: [`TILED_GEMM_MIN_TOKENS`],
//!   `TILED_GEMM_BLOCK_M`, `TILED_GEMM_BLOCK_N`, `TILED_GEMM_BLOCK_K`
//!   (`metal-tiled-gemm`-only, ROW 109's multi-simdgroup redesign — ports
//!   `ggml-metal.metal:6487-6489`'s `BLOCK_SIZE_M`/`BLOCK_SIZE_N`/
//!   `BLOCK_SIZE_K`). These trace to `omega-runtime.toml` via `build.rs`'s
//!   `emit_sizing_consts` (mirrors `proxima-tensor/build.rs`'s function of
//!   the same name over `proxima-tensor-runtime.toml`) and can be
//!   overridden per-build via an `OMEGA_<SECTION>_<KEY>` env var
//!   (`build.rs`'s `resolve_int`), each override consulted emitting its own
//!   `cargo:rerun-if-env-changed` line. `build.rs`'s
//!   `require_multiple_of_sixteen`/`require_divides_q4k_block`/
//!   `require_multiple_of_eight` enforce the cross-axis constraints
//!   `omega/src/msl.rs`'s `push_tiled_gemm_body` depends on.
//!
//! `msl` (this module's own crate) is alloc-tier and target-independent --
//! emission never touches a device -- so [`SIMD_WIDTH`] is visible at every
//! tier this crate has, matching `msl.rs`'s own gate. [`TILED_GEMM_MIN_TOKENS`]
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
