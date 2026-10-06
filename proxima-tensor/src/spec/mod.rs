//! The configuration face: a tensor program as TOML, and the conversion into
//! a `Vec<Op>`.
//!
//! This module exists to *force* a property rather than to claim one. If a
//! program can be written as data, adding an operation to a model is a
//! config edit; if it cannot, the claim that this algebra is describable was
//! never tested. The round-trip test at the bottom is that test — the same
//! matmul built in Rust and parsed from TOML must produce an equal `Vec<Op>`.
//!
//! Index patterns are written in `operand->iteration` notation, which reads
//! like einsum: `ik->ijk` says the operand has axes `i,k` drawn from an
//! iteration space of `i,j,k`. That covers projection, transpose, and
//! broadcast — the overwhelming majority.
//!
//! An operand axis may also be a comma-separated *expression*, one term per
//! axis: `"s,2*i->si"` says axis 0 is plain `s` and axis 1 is `2*i` — the
//! `AxisTerm { axis, coeff }` sum [`map::affine`] already
//! builds in Rust, spelled as data. A term is `[coeff*]letter` (letters stay
//! single ASCII characters, the same alphabet the bare-letter grammar uses),
//! several terms may be summed with `+`/`-`, and a bare integer term
//! contributes to the offset instead of a coefficient: `"2*h+r-1"` is a
//! stride-2, dilation-1 convolution window with padding folded into the
//! offset, `"2*i+1"` is RoPE's odd half of a pair. The comma is the trigger —
//! without one, the operand is still the old bare letter run (`ik->ijk`), so
//! no existing spelling changes meaning. This is parsing only: [`AxisIndex`]
//! and [`AxisTerm`] already expressed every one of these patterns before
//! this module could spell them.
//!
//! A [`NodeSpec::Reduce`]'s `in_map` reads through this same richer grammar
//! (`parse_operand_pattern`) — the asymmetry where only `Elementwise`
//! operands could spell a multi-term axis was an oversight, not a design
//! decision, since a `Reduce`'s operand is windowed exactly the same way a
//! convolution's `Elementwise(Multiply)` operand is (see
//! `specs/conv2d.toml`). `out_map` stays on the older, bare-letter-only
//! `parse_projection` deliberately: `shape::project_output_shape` already
//! rejects any `out_map` axis that is not a pure single-term `coeff == 1`
//! projection (`NotLowerable`, "reduce output maps must be pure projections
//! in v1"), so parsing a richer `out_map` would only ever be thrown away at
//! bind time — `parse_projection`'s narrower grammar gives the same
//! rejection at parse time instead, before a spec that could never lower
//! reaches shape inference at all.
//!
//! A [`NodeSpec::Elementwise`] operand map may instead be a
//! [`MapSpec::Gather`] table: `{ gather = "ids", index_map = "s->sd", map =
//! "d->sd", dim = 0 }`. `index_map` addresses the `gather` node the same
//! einsum way; `map` addresses the operand's *non-gathered* axes only, in
//! operand-axis order, skipping the position `dim` names —
//! `build_base_pattern` splices an empty (gathered) entry back in at that
//! position.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::TensorError;
#[cfg(feature = "instrument")]
use crate::instrument;
use crate::map::{self, AxisIndex, AxisTerm, IndexMap, IndexPattern};
use crate::op::{self, Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};

#[macro_use]
mod primitives;
#[macro_use]
mod mistral_layer_moe;
#[macro_use]
mod mistral_forward_cached;
#[macro_use]
mod hyperconn_qwen35_dense;
#[macro_use]
mod single_range_moe_cached;
#[macro_use]
mod lfm2_qwen35_gdn;
#[macro_use]
mod attention_forward;
#[macro_use]
mod lfm2_single_range_cached;
mod two_block_attention;
mod descriptor;
mod hybrid_forward;
mod layer_runs;
#[cfg(feature = "config")]
mod program_spec;
mod gguf_descriptor;
pub use attention_forward::*;
pub use crate::op::{SLIDING_CACHED_LEN_INPUT, SLIDING_KV_SYMBOL};
pub use descriptor::*;
pub use gguf_descriptor::*;
pub use hybrid_forward::{Qwen35MoeLayerDiagnostics, append_sigmoid_gated_shared_expert};
use hybrid_forward::{hybrid_dense_forward, hybrid_routed_forward};
pub use hyperconn_qwen35_dense::*;
pub use lfm2_qwen35_gdn::*;
pub use lfm2_single_range_cached::*;
pub use mistral_forward_cached::*;
pub use mistral_layer_moe::*;
#[cfg(feature = "config")]
pub use program_spec::*;
pub use primitives::*;
pub use single_range_moe_cached::*;
use two_block_attention::{append_cached_block_scores, append_local_block_and_combine, group_queries};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
