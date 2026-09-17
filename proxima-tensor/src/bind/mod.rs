//! Tensor operations with their addressing resolved — the seam between the
//! symbolic program and any executor.
//!
//! There is no second intermediate representation here. An [`BoundOp`] is an
//! [`Op`] whose addressing has been worked out against one call's symbol
//! bindings — the same elementwise/reduce shape, with a resolved [`Layout`]
//! standing in for a symbolic [`IndexMap`] and resolved iteration extents
//! standing in for a symbolic [`Extent`](crate::Extent) list. Nothing here is a competing
//! tree: [`bind`] still walks the program left to right, one `Op` at a
//! time, and every `BoundOp` it emits corresponds to exactly one `Op` that
//! actually computes (`Op::Input` never does — it is where data enters,
//! not where anything is derived).
//!
//! [`BoundOpBuilder`] does two things in one pass rather than two passes, and
//! that is a property of the algebra, not a corner cut: whether an
//! elementwise operand's layout is ever worth resolving on its own depends
//! entirely on whether the reduce consuming it can absorb that elementwise
//! op's body directly (the fusion decision). Splitting "resolve layout" from
//! "fuse" into separate stages would mean resolving a `Layout` for every
//! elementwise op and then throwing most of them away — real work with no
//! payoff — so the one `BoundOpBuilder` stage decides both at once, exactly the
//! way [`shape::ShapeTable`] decides shape *and* validity in one pass rather
//! than validating first and shaping second.
//!
//! Like [`shape::ShapeTable`], [`BoundOpBuilder`] is a sans-IO push state
//! machine: a program can arrive a step at a time, and op building must not
//! require the whole thing in hand. [`bind`] is the batch driver over it.
//! What *does* require the whole program in hand is liveness
//! ([`live::annotate`]) — computed once, upstream, and handed to
//! `BoundOpBuilder` as a plain kill-flag list it never has to guess at;
//! `BoundOpBuilder::new` takes that list up front for exactly this reason,
//! streamed or not.
//!
//! `BoundOpBuilder` also implements [`Pipe`]
//! (`In = (Op, Shapes)`, `Out = `[`ReadyBatch`]) — the same state machine,
//! not a second type wrapping it. `ReadyBatch` is a fixed-capacity, no-alloc
//! batch rather than a `Vec`: a single `push` readies at most
//! `READY_BATCH_CAPACITY` `BoundOp`s (see `push`'s own doc), so the
//! composition boundary between this stage and [`crate::cpu::Interpreter`]
//! never pays a heap allocation per `Op` pushed through the chain.
//! [`Pipe::call`] takes `&self`, so `held` below is
//! a `RefCell` and the node position a `Cell`, the same interior-mutability
//! idiom [`shape::ShapeTable`] uses for its own `Pipe` impl.
//!
//! The one optimization decided here: when a reduce's operand is an
//! elementwise op whose last use is that reduce (exact liveness, from
//! [`live::annotate`]), the elementwise op is never materialized — its body
//! is composed directly into the reduce's [`BoundOp`], which is the difference
//! between an O(extents) buffer and an O(iteration space) one for something
//! like matmul. An elementwise op whose last use is anything else (another
//! elementwise op, a non-fusable reduce, or nothing — a requested output or
//! dead code) materializes as its own `BoundOp`, emitted the moment that use is
//! seen (or, for dead code and outputs never referenced again, when
//! [`BoundOpBuilder::finish`] flushes it).

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::future::Future;

use arrayvec::ArrayVec;
use proxima_primitives::pipe::Pipe;
use smallvec::SmallVec;

use crate::dtype::DType;
use crate::error::TensorError;
#[cfg(feature = "instrument")]
use crate::instrument;
use crate::live;
#[cfg(feature = "cached-attention-streaming")]
use crate::map;
use crate::map::{AxisIndex, AxisTerm, IndexMap, IndexPattern};
use crate::numeric::{NumericPolicy, NumericRewrite, admit};
use crate::op::{Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use crate::shape::{self, Shapes};
#[cfg(feature = "instrument")]
use proxima_telemetry::debug;

/// Inline capacity for one bound op's per-iteration-axis buffers (`Layout`
/// strides, a reduce's surviving `output_axes`). No rank bound is stated or
/// enforced anywhere in this crate (`iter_rank` is a plain runtime `u16`);
/// the highest rank this crate's own tests and CPU evaluator exercise today
/// is 3 (`cpu.rs`'s `matmul_program`-style fixtures). 4 gives one axis of
/// headroom (e.g. a batch/heads/seq/dim attention iteration space) while
/// staying inline; `SmallVec` spills to the heap past this instead of
/// truncating, so a wider program still binds correctly.
pub use crate::sized::MAX_INLINE_RANK;

#[macro_use]
mod types_layout_boundop;
#[macro_use]
mod builder_compose_window;
#[macro_use]
mod dead_code_cached_attention;
#[macro_use]
mod gdn_moe_fusion_apply;
#[macro_use]
mod cached_attention_epilogue_liveness;
pub use builder_compose_window::*;
pub use cached_attention_epilogue_liveness::*;
pub use dead_code_cached_attention::*;
pub use gdn_moe_fusion_apply::*;
pub use types_layout_boundop::*;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
