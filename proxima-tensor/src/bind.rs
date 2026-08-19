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
use alloc::vec;
use alloc::vec::Vec;
use core::cell::{Cell, RefCell};
use core::future::Future;

use arrayvec::ArrayVec;
use proxima_primitives::pipe::Pipe;
use smallvec::SmallVec;

use crate::dtype::DType;
use crate::error::TensorError;
use crate::live;
use crate::map::{AxisIndex, AxisTerm, IndexMap, IndexPattern};
use crate::op::{Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};
use crate::shape::{self, Shapes};

/// Inline capacity for one bound op's per-iteration-axis buffers (`Layout`
/// strides, a reduce's surviving `output_axes`). No rank bound is stated or
/// enforced anywhere in this crate (`iter_rank` is a plain runtime `u16`);
/// the highest rank this crate's own tests and CPU evaluator exercise today
/// is 3 (`cpu.rs`'s `matmul_program`-style fixtures). 4 gives one axis of
/// headroom (e.g. a batch/heads/seq/dim attention iteration space) while
/// staying inline; `SmallVec` spills to the heap past this instead of
/// truncating, so a wider program still binds correctly.
pub const MAX_INLINE_RANK: usize = 4;

/// One operand's address into its own buffer, expressed directly in an
/// [`BoundOp`]'s iteration-axis space: `strides[axis]` is how far the linear
/// offset moves per step of iteration axis `axis`. An axis this operand
/// never varies along (broadcast) simply has stride 0.
///
/// This is the tensor-domain notion of memory layout (the same concept
/// `torch.Tensor.stride()` names) — the resolved counterpart of an
/// [`IndexPattern`], the same category as `IndexPattern` itself, not a rival
/// record kind, so it stays a small value type with its own accessors
/// rather than being folded away into bare tuple fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub base: i64,
    pub strides: SmallVec<[i64; MAX_INLINE_RANK]>,
}

impl Layout {
    #[must_use]
    pub fn offset_of(&self, coordinate: &[u64]) -> i64 {
        self.base
            + coordinate
                .iter()
                .zip(&self.strides)
                .map(|(index, stride)| stride * (*index as i64))
                .sum::<i64>()
    }

    #[must_use]
    pub fn stride(&self, axis: u16) -> i64 {
        self.strides.get(axis as usize).copied().unwrap_or(0)
    }
}

/// The extra addressing a gathered operand needs on top of its [`Layout`]:
/// where to fetch the index from, and how to turn a fetched index into an
/// offset once it lands — the same shape an embedding table lookup needs.
/// `element_stride` is the operand's own per-element stride along the
/// gathered axis (the table's row stride, for an embedding lookup) — an
/// executor's runtime read offset is
/// `layout.offset_of(coord) + fetched_index * element_stride`. `extent` is
/// the gathered axis's size, carried so an executor can reject an
/// out-of-range fetched index instead of reading past the buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lookup {
    pub indices: NodeId,
    pub index_layout: Layout,
    pub element_stride: i64,
    pub extent: u64,
}

type BoundOperands = Vec<(NodeId, Layout, Option<Lookup>)>;

/// Capacity for one [`BoundOpBuilder::push`] call's ready batch. An
/// elementwise push materializes at most one [`BoundOp`] per operand that
/// fails to fuse, bounded by [`ScalarOp::arity`]'s current maximum
/// (`Select`, 3); a reduce push materializes at most one held predecessor
/// plus the reduce's own op (2). `push` and `materialize_if_held` return
/// [`TensorError::NotLowerable`] rather than overflow this if a future
/// higher-arity `ScalarOp` variant is ever added.
const READY_BATCH_CAPACITY: usize = 3;

/// The batch [`BoundOpBuilder::push`] readies for one `Op`: [`Pipe::Out`]
/// for [`BoundOpBuilder`] and, by the composition law, [`Pipe::In`] for
/// [`crate::cpu::Interpreter`]. Fixed-capacity and stack-resident rather
/// than heap-backed like `BoundOperands` above — one `Vec` allocation per
/// `Op` pushed through the chain was the actual cost this replaces, for a
/// container that only ever holds 0 to `READY_BATCH_CAPACITY` items.
pub type ReadyBatch = ArrayVec<BoundOp, READY_BATCH_CAPACITY>;

/// One argument to a [`BodyStep`]: a fresh read of one of the [`BoundOp`]'s
/// own physical operands, or the result of an earlier step in the same
/// body. Backwards-only, the same rule [`crate::op::Op`]'s own module doc
/// states for a whole program's [`NodeId`] references, recreated here at the
/// scalar granularity a fused body composes at — a plain index into a side
/// table (`ComposedBody::steps`), never a `Box<dyn>` recursive tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepArg {
    Operand(u16),
    Step(u16),
}

/// One scalar computation inside a [`ComposedBody`]: apply `op` to `args`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BodyStep {
    pub op: ScalarOp,
    pub args: Vec<StepArg>,
}

/// A fused elementwise body: an ordered sequence of [`BodyStep`]s whose last
/// entry is the body's result. An unfused op is the one-step case
/// ([`ComposedBody::leaf`]), so every `BoundOp` carries exactly one
/// `ComposedBody` whether or not it absorbed anything — an executor has one
/// shape to walk regardless of how many elementwise ops a chain fused away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedBody {
    pub steps: Vec<BodyStep>,
}

impl ComposedBody {
    /// One scalar op applied directly to consecutive operand slots — the
    /// shape every body had before fusion existed.
    #[must_use]
    pub fn leaf(op: ScalarOp) -> Self {
        let args = (0..op.arity() as u16).map(StepArg::Operand).collect();
        Self {
            steps: vec![BodyStep { op, args }],
        }
    }
}

/// One tensor op with its addressing resolved: which buffers to read, at
/// what layout, combined by which scalar op — and, for a reduce, how the
/// reduction is shaped. Carries only what an executor needs, nothing about
/// *how* to execute it: [`cpu`](crate::cpu) interprets an `BoundOp` with nested
/// loops; a GPU backend could instead emit kernel source from the same
/// descriptor. Neither backend's shape belongs in this module.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundOp {
    pub node: NodeId,
    /// This node's own element type, carried straight from the [`Op`] it
    /// was built from ([`Op::dtype`]) — an executor spells its buffers and
    /// scratch declarations from this rather than assuming `f32`, which is
    /// what lets a GPU backend emit a narrower kernel for a narrower node
    /// without a second, parallel emitter.
    pub dtype: DType,
    /// The iteration-space extents this op's loop walks. Equal to the
    /// output shape for an [`BoundOpKind::Elementwise`] or a `Keep::Scan` reduce
    /// (a scan drops no axis); wider than the output shape for a
    /// `Keep::Reduce` reduce, which walks the full pre-reduction space.
    pub extents: Vec<u64>,
    pub kind: BoundOpKind,
}

/// The resolved counterpart of [`Op`]'s `Elementwise`/`Reduce` variants —
/// `Input` has no counterpart because a leaf never computes anything to
/// resolve.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundOpKind {
    Elementwise {
        body: ComposedBody,
        operands: BoundOperands,
    },
    Reduce {
        /// The per-step combine of `operands` before reducing: the fused
        /// elementwise chain's own body when one or more were absorbed, a
        /// one-step [`ScalarOp::Identity`] body otherwise. Distinct from
        /// `reduce_op`, which is the reduce's own accumulation.
        element_body: ComposedBody,
        reduce_op: ScalarOp,
        init: ReduceInit,
        keep: Keep,
        operands: BoundOperands,
        /// Iteration axes that survive to the output, in the reduce's
        /// `out_map` operand-axis order. The last entry (if any) is the
        /// innermost loop.
        output_axes: SmallVec<[u16; MAX_INLINE_RANK]>,
        out_layout: Layout,
    },
    /// The resolved counterpart of [`Op::Iota`]: no operands, no body — an
    /// executor derives every output value straight from its own position
    /// in `BoundOp::extents`, which is why this variant carries no fields of
    /// its own.
    Iota,
    /// The resolved counterpart of [`Op::Constant`]: no operands, no body,
    /// and unlike [`BoundOpKind::Iota`] not even a dependence on position —
    /// an executor writes `value` to every element of `BoundOp::extents`.
    Constant { value: f32 },
}

/// A fused body with zero steps — [`BoundOp::element_body`]'s answer for
/// [`BoundOpKind::Iota`], which has no combining body at all. Every real
/// caller (`cpu::run_elementwise`/`run_reduce`/`run_scan`,
/// `omega`'s renderers) only reaches `element_body()` from inside a branch
/// already matched on `Elementwise`/`Reduce`, so this is never actually
/// read; it exists so the accessor stays total over every `BoundOpKind`
/// instead of panicking on the one variant that has nothing to answer with.
static EMPTY_BODY: ComposedBody = ComposedBody { steps: Vec::new() };

impl BoundOp {
    #[must_use]
    pub fn operands(&self) -> &[(NodeId, Layout, Option<Lookup>)] {
        match &self.kind {
            BoundOpKind::Elementwise { operands, .. } | BoundOpKind::Reduce { operands, .. } => {
                operands
            }
            BoundOpKind::Iota | BoundOpKind::Constant { .. } => &[],
        }
    }

    /// The composed body applied per step to build one combined value from
    /// `operands()`, before any reduction: an elementwise op's own
    /// (possibly fused) body, or a fused reduce's absorbed body (a one-step
    /// `Identity` body if nothing fused). See `EMPTY_BODY`'s doc for the
    /// [`BoundOpKind::Iota`] case.
    #[must_use]
    pub fn element_body(&self) -> &ComposedBody {
        match &self.kind {
            BoundOpKind::Elementwise { body, .. } => body,
            BoundOpKind::Reduce { element_body, .. } => element_body,
            BoundOpKind::Iota | BoundOpKind::Constant { .. } => &EMPTY_BODY,
        }
    }

    /// Splits this op along its outermost output-iteration axis into
    /// `parts` contiguous chunks so each chunk can execute independently:
    /// for an `Elementwise` op that axis is `extents[0]`; for a
    /// `Keep::Reduce` reduce it is the first entry of `output_axes` (the
    /// reduce's outermost surviving axis, per that field's own doc). This
    /// split is backend-neutral on purpose: a CPU driver runs chunks on
    /// worker threads, a GPU backend would tile the identical axis into
    /// threadgroups — neither belongs in this module, only the geometry
    /// does.
    ///
    /// Returns `None` when splitting is not sound or not useful:
    /// - a scalar reduction (`output_axes` is empty — nothing to split, the
    ///   whole op is one accumulator),
    /// - a `Keep::Scan` scan (each step reads the previous step's output,
    ///   so the extent is a sequential dependency, not parallel work),
    /// - the split axis's extent is smaller than `parts` (some chunk would
    ///   be empty),
    /// - `parts < 2` (nothing to split into).
    ///
    /// Chunk `k`'s output occupies the contiguous range
    /// `[chunk_start * inner_size, chunk_start * inner_size + chunk_len *
    /// inner_size)` of the parent output buffer, where `inner_size` is the
    /// product of the extents after the split axis. A caller relies on this
    /// to hand each chunk its own disjoint `&mut` sub-slice — via successive
    /// `split_at_mut` of one parent buffer — and run every chunk
    /// concurrently with no further coordination.
    ///
    /// The two layout kinds are rebased differently, which is *why* that
    /// works:
    /// - every **operand** layout's base is shifted by
    ///   `chunk_start * stride(split_axis)`, because operands are read from
    ///   the one full, unsplit source buffer shared by every chunk.
    /// - a reduce's `out_layout` is **not** shifted at all: the interpreter
    ///   already derives every write offset from a loop that iterates the
    ///   split axis's own (already-shrunk) extent starting at 0, for every
    ///   chunk alike, so an out_layout carrying the parent's unmodified
    ///   base already produces exactly the 0-based offsets a `split_at_mut`
    ///   sub-slice expects. Rebasing it too would double-count the offset.
    #[must_use]
    pub fn split(&self, parts: usize) -> Option<Vec<BoundOp>> {
        self.split_aligned(parts, 1)
    }

    /// Same contract as [`split`](Self::split), except each of the first
    /// `parts - 1` chunks is rounded down to a multiple of `alignment` rows
    /// (the remainder folds into the last, already-ragged chunk) instead of
    /// always taking `extent / parts` exactly. `alignment == 1` degenerates
    /// to [`split`](Self::split)'s behavior byte-for-byte.
    ///
    /// Exists because equal row counts are not equal wall-clock: a caller
    /// tiling its kernel in `alignment`-row blocks pays a narrower,
    /// measurably slower fallback path for every chunk boundary that does
    /// not land on a tile edge, and that count grows with chunk count (see
    /// `cpu::TILE_ROWS`'s doc for the measured spread).
    #[must_use]
    pub fn split_aligned(&self, parts: usize, alignment: u64) -> Option<Vec<BoundOp>> {
        if parts < 2 {
            return None;
        }
        let split_axis = self.split_axis()?;
        let extent = self.extents[split_axis as usize];
        if extent < parts as u64 {
            return None;
        }

        Some(
            chunk_ranges(extent, parts, alignment)
                .map(|(chunk_start, chunk_len)| {
                    self.rebase_chunk(split_axis, chunk_start, chunk_len)
                })
                .collect(),
        )
    }

    fn split_axis(&self) -> Option<u16> {
        match &self.kind {
            BoundOpKind::Elementwise { .. } => (!self.extents.is_empty()).then_some(0),
            BoundOpKind::Reduce {
                keep, output_axes, ..
            } => match keep {
                Keep::Scan => None,
                Keep::Reduce => output_axes.first().copied(),
            },
            // an `Iota` is cheap enough (one write per element, no operand
            // reads) that splitting it across workers is not worth the
            // bookkeeping; `None` here just means a caller runs it as one
            // chunk, the same as any other unsplittable op.
            BoundOpKind::Iota | BoundOpKind::Constant { .. } => None,
        }
    }

    fn rebase_chunk(&self, split_axis: u16, chunk_start: u64, chunk_len: u64) -> BoundOp {
        let mut extents = self.extents.clone();
        extents[split_axis as usize] = chunk_len;

        let kind = match &self.kind {
            BoundOpKind::Elementwise { body, operands } => BoundOpKind::Elementwise {
                body: body.clone(),
                operands: rebase_operands(operands, split_axis, chunk_start),
            },
            BoundOpKind::Reduce {
                element_body,
                reduce_op,
                init,
                keep,
                operands,
                output_axes,
                out_layout,
            } => BoundOpKind::Reduce {
                element_body: element_body.clone(),
                reduce_op: *reduce_op,
                init: *init,
                keep: *keep,
                operands: rebase_operands(operands, split_axis, chunk_start),
                output_axes: output_axes.clone(),
                // unchanged from the parent: see this method's doc for why
                // an unshifted out_layout already yields 0-based write
                // offsets.
                out_layout: out_layout.clone(),
            },
            // unreachable in practice: `split_axis` returns `None` for
            // `Iota`, so `split`/`split_aligned` never call this for one —
            // kept explicit rather than a catch-all so a future change to
            // `split_axis` cannot silently start routing `Iota` here with no
            // rebase logic to run.
            BoundOpKind::Iota => BoundOpKind::Iota,
            // same reasoning as `Iota` above: `split_axis` returns `None`
            // for a `Constant`, so this arm is never reached in practice.
            BoundOpKind::Constant { value } => BoundOpKind::Constant { value: *value },
        };

        BoundOp {
            node: self.node,
            dtype: self.dtype,
            extents,
            kind,
        }
    }
}

fn rebase_operands(operands: &BoundOperands, split_axis: u16, chunk_start: u64) -> BoundOperands {
    operands
        .iter()
        .map(|(node, layout, lookup)| {
            let rebased_lookup = lookup.as_ref().map(|lookup| Lookup {
                indices: lookup.indices,
                index_layout: rebase_layout(&lookup.index_layout, split_axis, chunk_start),
                element_stride: lookup.element_stride,
                extent: lookup.extent,
            });
            (
                *node,
                rebase_layout(layout, split_axis, chunk_start),
                rebased_lookup,
            )
        })
        .collect()
}

/// `parts` contiguous `(start, len)` ranges covering `0..extent`: the first
/// `parts - 1` ranges are `extent / parts` wide, rounded down to a multiple
/// of `alignment` (unless that would zero them out, in which case the raw
/// unaligned width is kept), and the last absorbs whatever remains — the
/// only one that can be a different (ragged) size. `alignment <= 1` is a
/// no-op: the rounding step is skipped entirely.
fn chunk_ranges(extent: u64, parts: usize, alignment: u64) -> impl Iterator<Item = (u64, u64)> {
    let raw_len = extent / parts as u64;
    let chunk_len = if alignment > 1 && raw_len >= alignment {
        raw_len - (raw_len % alignment)
    } else {
        raw_len
    };
    (0..parts).scan(0u64, move |start, index| {
        let chunk_start = *start;
        let len = if index + 1 == parts {
            extent - chunk_start
        } else {
            chunk_len
        };
        *start += len;
        Some((chunk_start, len))
    })
}

fn rebase_layout(layout: &Layout, split_axis: u16, chunk_start: u64) -> Layout {
    Layout {
        base: layout.base + layout.stride(split_axis) * chunk_start as i64,
        strides: layout.strides.clone(),
    }
}

struct HeldElementwise {
    dtype: DType,
    body: ScalarOp,
    operands: Vec<(NodeId, IndexMap)>,
}

/// The prefix state of op building: elementwise ops seen but not yet
/// materialized.
///
/// `retires` and `position` make [`BoundOpBuilder::push`] a single-argument-per-node
/// step (`expr`, `shapes`) rather than needing `node`/`retires` threaded in
/// by the caller on every call: `retires[i]` is node `i`'s kill-flag list
/// (see [`live::annotate`], computed once over the whole program before the
/// first push), and `position` is the node id the next push resolves to —
/// both pieces this type already needed to know, now carried as its own
/// state instead of repeated arguments.
pub struct BoundOpBuilder {
    held: RefCell<BTreeMap<NodeId, HeldElementwise>>,
    retires: Vec<Vec<NodeId>>,
    position: Cell<u32>,
}

impl BoundOpBuilder {
    /// `retires` is normally [`live::annotate`]`(program, outputs)`.
    #[must_use]
    pub fn new(retires: Vec<Vec<NodeId>>) -> Self {
        Self {
            held: RefCell::new(BTreeMap::new()),
            retires,
            position: Cell::new(0),
        }
    }

    /// Judge one expression: hold an elementwise op, or emit whatever is now
    /// ready.
    ///
    /// May return more than one [`BoundOp`]: consuming a held elementwise op
    /// that turns out not to fuse must materialize it before the current
    /// expression can read it, so a single push can ready both that
    /// standalone op and the current expression's own — and, for an
    /// elementwise expression, one materialization per operand that fails to
    /// fuse, up to [`ScalarOp::arity`]'s current maximum
    /// (`READY_BATCH_CAPACITY`).
    pub fn push(&self, expr: &Op, shapes: &Shapes) -> Result<ReadyBatch, TensorError> {
        let node = NodeId(self.position.get());
        self.position.set(self.position.get() + 1);
        let empty = Vec::new();
        let retires = self.retires.get(node.0 as usize).unwrap_or(&empty);

        let mut emitted = ReadyBatch::new();

        match expr {
            Op::Input { .. } => {}
            Op::Iota { dtype, .. } => {
                let extents = shapes.of(node).to_vec();
                push_ready(
                    &mut emitted,
                    node,
                    BoundOp {
                        node,
                        dtype: *dtype,
                        extents,
                        kind: BoundOpKind::Iota,
                    },
                )?;
            }
            Op::Constant { dtype, value, .. } => {
                let extents = shapes.of(node).to_vec();
                push_ready(
                    &mut emitted,
                    node,
                    BoundOp {
                        node,
                        dtype: *dtype,
                        extents,
                        kind: BoundOpKind::Constant { value: *value },
                    },
                )?;
            }
            Op::Elementwise {
                dtype,
                body,
                operands,
                ..
            } => {
                for (operand_node, map) in operands {
                    let fuses = retires.contains(operand_node)
                        && is_identity_projection(map)
                        && self.held.borrow().contains_key(operand_node);
                    if !fuses {
                        self.materialize_if_held(*operand_node, shapes, &mut emitted)?;
                    }
                }
                self.held.borrow_mut().insert(
                    node,
                    HeldElementwise {
                        dtype: *dtype,
                        body: *body,
                        operands: operands.clone(),
                    },
                );
            }
            Op::Reduce(reduce) => {
                let fuses = retires.contains(&reduce.operand)
                    && is_identity_projection(&reduce.in_map)
                    && self.held.borrow().contains_key(&reduce.operand);

                let (element_body, operands) = if fuses {
                    let reduce_extent: u64 =
                        shape::fold_iteration_extents(reduce, shapes).iter().product();
                    self.quarantine_broadcast_operands(
                        reduce.operand,
                        reduce_extent,
                        shapes,
                        &mut emitted,
                    )?;
                    compose_fused_operands(shapes, &self.held, reduce.operand, &reduce.in_map)
                } else {
                    self.materialize_if_held(reduce.operand, shapes, &mut emitted)?;
                    let operand = build_operand(reduce.operand, &reduce.in_map, shapes);
                    (ComposedBody::leaf(ScalarOp::Identity), vec![operand])
                };

                push_ready(
                    &mut emitted,
                    node,
                    build_reduce_op(node, reduce, shapes, element_body, operands),
                )?;
            }
        }

        Ok(emitted)
    }

    /// Flush every elementwise op still held: each was either a requested
    /// output or dead code, and either way it materializes as its own op.
    /// Processed from the highest [`NodeId`] down: a still-held node can
    /// only ever be fused into a consumer with a *greater* id (references
    /// point backwards only), so visiting consumers first lets
    /// `build_elementwise_op` absorb whatever it still can before an
    /// earlier, now-absorbed node would otherwise be flushed standalone.
    pub fn finish(self, shapes: &Shapes) -> Result<Vec<BoundOp>, TensorError> {
        let mut remaining: Vec<NodeId> = self.held.borrow().keys().copied().collect();
        remaining.sort_unstable_by(|left, right| right.cmp(left));

        let mut built = Vec::new();
        for node in remaining {
            // NOT `if let Some(x) = self.held.borrow_mut()....` — that
            // temporary's `RefMut` lives to the end of the `if let` body
            // under Rust's temporary-lifetime-extension rule, and
            // `build_elementwise_op` below borrows `self.held` itself, so
            // the two would collide. Ending the borrow at this statement's
            // semicolon first avoids the re-entrant panic.
            let removed = self.held.borrow_mut().remove(&node);
            if let Some(held) = removed {
                built.push(build_elementwise_op(
                    node,
                    shapes,
                    &self.held,
                    held.dtype,
                    held.body,
                    &held.operands,
                ));
            }
        }
        built.reverse();
        Ok(built)
    }

    fn materialize_if_held(
        &self,
        node: NodeId,
        shapes: &Shapes,
        emitted: &mut ReadyBatch,
    ) -> Result<(), TensorError> {
        // see `finish`'s comment: the borrow must end before this `if let`
        // body runs, since `build_elementwise_op` borrows `self.held` too.
        let removed = self.held.borrow_mut().remove(&node);
        if let Some(held) = removed {
            let materialized = build_elementwise_op(
                node,
                shapes,
                &self.held,
                held.dtype,
                held.body,
                &held.operands,
            );
            push_ready(emitted, node, materialized)?;
        }
        Ok(())
    }

    /// Walks `node`'s still-held operands and materializes any whose own
    /// natural iteration space (`shapes.of(child)`) is smaller than
    /// `reduce_extent` — composing one through anyway would run its body
    /// once per `reduce_extent` element instead of once per its own, which
    /// is exactly the cost [`is_identity_projection`] cannot see: it only
    /// judges one map's shape, not what fusing recursively absorbs beneath
    /// it (see `compose_operand`'s own doc — it trusts every map it
    /// recurses through was already checked, but that check happened at a
    /// different, earlier `push`, against that op's own — smaller —
    /// iteration space, not against this reduce's). Safe children (same or
    /// larger extent) are walked further, since a broadcast can reappear
    /// several levels down.
    fn quarantine_broadcast_operands(
        &self,
        node: NodeId,
        reduce_extent: u64,
        shapes: &Shapes,
        emitted: &mut ReadyBatch,
    ) -> Result<(), TensorError> {
        let children = self.held.borrow().get(&node).map(|held| held.operands.clone());
        let Some(children) = children else {
            return Ok(());
        };
        for (child, _map) in children {
            if !self.held.borrow().contains_key(&child) {
                continue;
            }
            let child_extent: u64 = shapes.of(child).iter().product();
            if child_extent < reduce_extent {
                self.materialize_if_held(child, shapes, emitted)?;
            } else {
                self.quarantine_broadcast_operands(child, reduce_extent, shapes, emitted)?;
            }
        }
        Ok(())
    }
}

/// Appends one ready [`BoundOp`] to a [`ReadyBatch`], turning an overflow
/// (never observed for today's `ScalarOp` variants — see
/// [`READY_BATCH_CAPACITY`]'s own doc) into a [`TensorError`] instead of a
/// panic.
fn push_ready(emitted: &mut ReadyBatch, node: NodeId, op: BoundOp) -> Result<(), TensorError> {
    emitted.try_push(op).map_err(|_| TensorError::NotLowerable {
        node,
        reason: "one push readied more BoundOps than the no-alloc batch capacity allows",
    })
}

/// `In = (Op, Shapes)` matches [`shape::ShapeTable`]'s own `Pipe::Out`
/// exactly, so `AndThen::new(ShapeTable, BoundOpBuilder)` (or
/// `shapes_instance.and_then(builder_instance)`) composes with no adapter:
/// shape resolution's snapshot travels alongside the `Op` it was resolved
/// for, and [`BoundOpBuilder::push`] reads both straight out of `Self::In`.
impl Pipe for BoundOpBuilder {
    type In = (Op, Shapes);
    type Out = ReadyBatch;
    type Err = TensorError;

    fn call(
        &self,
        (expr, shapes): Self::In,
    ) -> impl Future<Output = Result<ReadyBatch, TensorError>> {
        async move { self.push(&expr, &shapes) }
    }
}

fn build_elementwise_op(
    node: NodeId,
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    dtype: DType,
    body: ScalarOp,
    operands: &[(NodeId, IndexMap)],
) -> BoundOp {
    let extents = shapes.of(node).to_vec();
    let (composed_body, built_operands) = compose(shapes, held, body, operands);
    BoundOp {
        node,
        dtype,
        extents,
        kind: BoundOpKind::Elementwise {
            body: composed_body,
            operands: built_operands,
        },
    }
}

/// One operand's [`Layout`] (and, for a gather, its [`Lookup`]), built
/// directly from its [`IndexMap`] — the one place that decides how an
/// `Affine` vs a `Computed` map turns into what an executor reads.
fn build_operand(
    node: NodeId,
    map: &IndexMap,
    shapes: &Shapes,
) -> (NodeId, Layout, Option<Lookup>) {
    match map {
        IndexMap::Affine(pattern) => (node, layout_of(pattern, shapes.of(node)), None),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => {
            let operand_shape = shapes.of(node);
            let layout = layout_of(base, operand_shape);
            let index_layout = layout_of(index_map, shapes.of(*indices));
            let element_stride = row_major_strides(operand_shape)[*gathered_dim as usize];
            let extent = operand_shape[*gathered_dim as usize];
            let lookup = Lookup {
                indices: *indices,
                index_layout,
                element_stride,
                extent,
            };
            (node, layout, Some(lookup))
        }
    }
}

fn build_reduce_op(
    node: NodeId,
    reduce: &Reduce,
    shapes: &Shapes,
    element_body: ComposedBody,
    operands: BoundOperands,
) -> BoundOp {
    let out_pattern = reduce.out_map.affine();
    let out_layout = layout_of(out_pattern, shapes.of(node));
    let output_axes = pure_projection_axes(out_pattern);
    BoundOp {
        node,
        dtype: reduce.dtype,
        extents: shape::fold_iteration_extents(reduce, shapes),
        kind: BoundOpKind::Reduce {
            element_body,
            reduce_op: reduce.body,
            init: reduce.init,
            keep: reduce.keep,
            operands,
            output_axes,
            out_layout,
        },
    }
}

fn pure_projection_axes(pattern: &IndexPattern) -> SmallVec<[u16; MAX_INLINE_RANK]> {
    pattern
        .axes
        .iter()
        .filter_map(|axis| match axis.terms.as_slice() {
            [term] if term.coeff == 1 => Some(term.axis),
            _ => None,
        })
        .collect()
}

/// A fusion can compose through: every axis a plain, unshifted projection.
/// Anything richer (a window, a slice, a stride) still resolves correctly,
/// it just materializes its operand instead of composing through it.
fn is_identity_projection(map: &IndexMap) -> bool {
    if map.is_data_dependent() {
        return false;
    }
    map.affine()
        .axes
        .iter()
        .all(|axis| axis.offset == 0 && matches!(axis.terms.as_slice(), [term] if term.coeff == 1))
}

/// Composes the single still-held node `node` — reached from its consumer
/// through `map` — into a [`ComposedBody`] plus the flat, fully-addressed
/// operand list an executor reads from: the reduce-fusion entry point.
/// `node` is guaranteed present in `held` by every caller's own `fuses`
/// check, so this always absorbs at least one op; [`compose_operand`]
/// recurses through however many more are held beneath it.
fn compose_fused_operands(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    node: NodeId,
    map: &IndexMap,
) -> (ComposedBody, BoundOperands) {
    let mut steps = Vec::new();
    let mut operands = Vec::new();
    let mut absorbed = Vec::new();
    compose_operand(
        shapes,
        held,
        &mut steps,
        &mut operands,
        &mut absorbed,
        node,
        map,
    );
    drop_absorbed(held, absorbed);
    (ComposedBody { steps }, operands)
}

/// Composes an explicit `body` applied over `operands` — the
/// materialize-a-chain entry point [`build_elementwise_op`] uses, where the
/// top body and its immediate operand list are already in hand (the node
/// itself has already been removed from `held` by its caller).
fn compose(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    body: ScalarOp,
    operands: &[(NodeId, IndexMap)],
) -> (ComposedBody, BoundOperands) {
    let mut steps = Vec::new();
    let mut resolved_operands = Vec::new();
    let mut absorbed = Vec::new();
    compose_body(
        shapes,
        held,
        &mut steps,
        &mut resolved_operands,
        &mut absorbed,
        body,
        operands,
    );
    drop_absorbed(held, absorbed);
    (ComposedBody { steps }, resolved_operands)
}

fn drop_absorbed(held: &RefCell<BTreeMap<NodeId, HeldElementwise>>, absorbed: Vec<NodeId>) {
    let mut held_mut = held.borrow_mut();
    for node in absorbed {
        held_mut.remove(&node);
    }
}

/// Appends one [`BodyStep`] for `body` applied over `body_operands`
/// (expressed in the caller's own iteration space), recursively composing
/// each operand through [`compose_operand`] first. Returns the new step's
/// index — the value a caller reads back via `StepArg::Step`.
fn compose_body(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    steps: &mut Vec<BodyStep>,
    operands: &mut BoundOperands,
    absorbed: &mut Vec<NodeId>,
    body: ScalarOp,
    body_operands: &[(NodeId, IndexMap)],
) -> u16 {
    let args = body_operands
        .iter()
        .map(|(node, map)| compose_operand(shapes, held, steps, operands, absorbed, *node, map))
        .collect();
    steps.push(BodyStep { op: body, args });
    (steps.len() - 1) as u16
}

/// Composes one operand reference `(node, map)` into `steps`/`operands`:
/// reads it directly from its own buffer when `node` is not (or is no
/// longer) held, or — when it is still held, meaning it satisfied the
/// fusion condition at the exact position that made this its last use —
/// absorbs its own body as one more [`BodyStep`], recursing through
/// however many further levels are held beneath it. `map`'s axes are
/// remapped through [`remap_sub_operands`] before recursing, since a held
/// node's own operand maps are expressed in *its* iteration space, not the
/// caller's.
fn compose_operand(
    shapes: &Shapes,
    held: &RefCell<BTreeMap<NodeId, HeldElementwise>>,
    steps: &mut Vec<BodyStep>,
    operands: &mut BoundOperands,
    absorbed: &mut Vec<NodeId>,
    node: NodeId,
    map: &IndexMap,
) -> StepArg {
    let entry = held
        .borrow()
        .get(&node)
        .map(|held_elementwise| (held_elementwise.body, held_elementwise.operands.clone()));

    let Some((body, sub_operands)) = entry else {
        operands.push(build_operand(node, map, shapes));
        return StepArg::Operand((operands.len() - 1) as u16);
    };

    absorbed.push(node);
    let remapped = remap_sub_operands(&sub_operands, map);
    StepArg::Step(compose_body(
        shapes, held, steps, operands, absorbed, body, &remapped,
    ))
}

/// The outer iteration axis each of `map`'s own axes corresponds to — sound
/// only when `map` is [`is_identity_projection`], which every caller here
/// already checked before fusing through it.
fn axis_correspondence(map: &IndexMap) -> Vec<u16> {
    map.affine()
        .axes
        .iter()
        .map(|axis| axis.terms[0].axis)
        .collect()
}

fn remap_pattern(pattern: &IndexPattern, axis_map: &[u16], outer_iter_rank: u16) -> IndexPattern {
    let axes = pattern
        .axes
        .iter()
        .map(|axis_index| AxisIndex {
            terms: axis_index
                .terms
                .iter()
                .map(|term| AxisTerm {
                    axis: axis_map[term.axis as usize],
                    coeff: term.coeff,
                })
                .collect(),
            offset: axis_index.offset,
        })
        .collect();
    IndexPattern {
        iter_rank: outer_iter_rank,
        axes,
    }
}

fn remap_index_map(map: &IndexMap, axis_map: &[u16], outer_iter_rank: u16) -> IndexMap {
    match map {
        IndexMap::Affine(pattern) => {
            IndexMap::Affine(remap_pattern(pattern, axis_map, outer_iter_rank))
        }
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => IndexMap::Computed {
            indices: *indices,
            index_map: remap_pattern(index_map, axis_map, outer_iter_rank),
            base: remap_pattern(base, axis_map, outer_iter_rank),
            gathered_dim: *gathered_dim,
        },
    }
}

/// Composes a held op's own operand maps (expressed in its own iteration
/// space) through `outer_map` — how its consumer reads it, always an
/// identity projection, the fusion precondition — into the consumer's own
/// iteration space. The symbolic counterpart of [`Layout`]-level stride
/// remapping, applied one level per absorbed node so [`compose_operand`]'s
/// recursion composes through as many levels as a chain has.
fn remap_sub_operands(
    sub_operands: &[(NodeId, IndexMap)],
    outer_map: &IndexMap,
) -> Vec<(NodeId, IndexMap)> {
    let axis_map = axis_correspondence(outer_map);
    let outer_iter_rank = outer_map.affine().iter_rank;
    sub_operands
        .iter()
        .map(|(node, map)| (*node, remap_index_map(map, &axis_map, outer_iter_rank)))
        .collect()
}

fn layout_of(pattern: &IndexPattern, operand_shape: &[u64]) -> Layout {
    let element_strides = row_major_strides(operand_shape);
    let mut strides = SmallVec::<[i64; MAX_INLINE_RANK]>::from_elem(0, pattern.iter_rank as usize);
    let mut base = 0i64;
    for (axis_index, axis) in pattern.axes.iter().enumerate() {
        let stride = element_strides[axis_index];
        base += i64::from(axis.offset) * stride;
        for term in &axis.terms {
            strides[term.axis as usize] += i64::from(term.coeff) * stride;
        }
    }
    Layout { base, strides }
}

fn row_major_strides(shape: &[u64]) -> Vec<i64> {
    let mut strides = vec![0i64; shape.len()];
    let mut accumulator = 1i64;
    for (axis_index, extent) in shape.iter().enumerate().rev() {
        strides[axis_index] = accumulator;
        accumulator *= *extent as i64;
    }
    strides
}

/// Batch driver: computes liveness once, then streams every expression
/// through a fresh [`BoundOpBuilder`], flushing whatever remains held at the end.
pub fn bind(
    program: &[Op],
    shapes: &Shapes,
    outputs: &[NodeId],
) -> Result<Vec<BoundOp>, TensorError> {
    let retires = live::annotate(program, outputs);
    let building = BoundOpBuilder::new(retires);
    let mut built = Vec::new();
    for expr in program {
        built.extend(building.push(expr, shapes)?);
    }
    built.extend(building.finish(shapes)?);
    Ok(built)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::map;
    use crate::op::{Extent, append};
    use rstest::rstest;

    fn matmul_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(768)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(768), Extent::Static(3072)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: crate::op::ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        (program, product, sum, lhs)
    }

    /// An `Iota` binds directly to its own ready `BoundOp`, the same way a
    /// `Reduce` always does — never held pending fusion the way an
    /// `Elementwise` op is, since it has no operand to fuse with anything.
    #[test]
    fn an_iota_binds_to_its_own_ready_bound_op_with_no_operands() {
        let mut program = Vec::new();
        let iota = append(
            &mut program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(8),
            },
        );

        let shapes = shape::infer(&program, &[]).expect("iota infers");
        let built = bind(&program, &shapes, &[]).expect("iota builds ops");

        assert_eq!(built.len(), 1, "the iota leaf materializes on its own");
        assert_eq!(built[0].node, iota);
        assert_eq!(built[0].dtype, DType::Float32);
        assert_eq!(built[0].extents, alloc::vec![8]);
        assert!(matches!(built[0].kind, BoundOpKind::Iota));
        assert!(
            built[0].operands().is_empty(),
            "a leaf with no operands binds to none"
        );
    }

    #[test]
    fn matmul_resolves_to_one_fused_op_not_two() {
        let (program, product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let built = bind(&program, &shapes, &[]).expect("matmul builds ops");

        assert_eq!(
            built.len(),
            1,
            "the elementwise op must not materialize separately"
        );
        assert_eq!(built[0].node, sum);
        assert!(matches!(built[0].kind, BoundOpKind::Reduce { .. }));
        assert_eq!(
            built[0].element_body().steps.len(),
            1,
            "one absorbed elementwise op is one composed step"
        );
        assert_ne!(
            built[0].element_body().steps[0].op,
            ScalarOp::Identity,
            "the fused body is the elementwise op's multiply"
        );
        let _ = product;
    }

    #[test]
    fn requesting_the_intermediate_elementwise_op_as_an_output_prevents_fusion() {
        let (program, product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let built =
            bind(&program, &shapes, &[product, sum]).expect("matmul builds ops with two outputs");

        assert_eq!(
            built.len(),
            2,
            "the requested-output elementwise op must materialize"
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == product && matches!(op.kind, BoundOpKind::Elementwise { .. }))
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == sum && matches!(op.kind, BoundOpKind::Reduce { .. }))
        );
    }

    /// `b = a * scale; c = b + bias; d = c * c` — three chained elementwise
    /// ops, each the sole and last use of the one before it, none of them
    /// requested as an output. All three must fuse into `d`'s own `BoundOp`
    /// rather than materializing `b` and `c` along the way.
    fn elementwise_chain_program() -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let scale = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let bias = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(a, identity()), (scale, identity())],
                name: None,
            },
        );
        let c = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(b, identity()), (bias, identity())],
                name: None,
            },
        );
        let d = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(c, identity()), (c, identity())],
                name: None,
            },
        );
        (program, b, c, d)
    }

    #[test]
    fn a_chain_of_elementwise_ops_fuses_into_one_bound_op_not_three() {
        let (program, _b, _c, d) = elementwise_chain_program();
        let shapes = shape::infer(&program, &[]).expect("elementwise chain infers");
        let built = bind(&program, &shapes, &[]).expect("elementwise chain builds ops");

        assert_eq!(
            built.len(),
            1,
            "b and c must absorb into d's own BoundOp instead of materializing"
        );
        assert_eq!(built[0].node, d);
        assert!(matches!(built[0].kind, BoundOpKind::Elementwise { .. }));
        assert!(
            built[0].element_body().steps.len() >= 2,
            "the composed body must carry more than one absorbed op's step"
        );
    }

    #[test]
    fn an_elementwise_intermediate_requested_as_an_output_prevents_fusion() {
        let (program, b, _c, d) = elementwise_chain_program();
        let shapes = shape::infer(&program, &[]).expect("elementwise chain infers");
        let built =
            bind(&program, &shapes, &[b, d]).expect("elementwise chain builds ops with 2 outputs");

        assert_eq!(
            built.len(),
            2,
            "requesting b as an output must force it to materialize on its own"
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == b && matches!(op.kind, BoundOpKind::Elementwise { .. }))
        );
        assert!(built.iter().any(|op| op.node == d));
    }

    #[test]
    fn an_elementwise_intermediate_consumed_by_two_different_ops_is_not_fused() {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: alloc::vec![(a, identity())],
                name: None,
            },
        );
        let c1 = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Negate,
                operands: alloc::vec![(b, identity())],
                name: None,
            },
        );
        let c2 = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Reciprocal,
                operands: alloc::vec![(b, identity())],
                name: None,
            },
        );
        let d = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![(c1, identity()), (c2, identity())],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("diamond chain infers");
        let built = bind(&program, &shapes, &[]).expect("diamond chain builds ops");

        assert_eq!(
            built.len(),
            2,
            "b feeds two different consumers, so it must materialize once on its own, \
             and d (absorbing c1 and c2, whose only use each is d) is the other"
        );
        assert!(
            built
                .iter()
                .any(|op| op.node == b && matches!(op.kind, BoundOpKind::Elementwise { .. })),
            "b must materialize standalone rather than fuse into either consumer"
        );
        assert!(built.iter().any(|op| op.node == d));
        let _ = c1;
        let _ = c2;
    }

    /// `product = a * b; scaled = product * c; sum = reduce(+, scaled)` — two
    /// chained elementwise ops feeding a reduce, mirroring `matmul_program`
    /// but with an extra elementwise hop before the contraction. Both
    /// elementwise ops must absorb into the reduce's own `BoundOp`.
    #[test]
    fn elementwise_into_elementwise_into_reduce_fuses_into_one_bound_op() {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let b = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let c = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(a, identity()), (b, identity())],
                name: None,
            },
        );
        let scaled = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(product, identity()), (c, identity())],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: scaled,
                in_map: identity(),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: Some("weighted_dot".into()),
            }),
        );

        let shapes = shape::infer(&program, &[]).expect("weighted dot infers");
        let built = bind(&program, &shapes, &[]).expect("weighted dot builds ops");

        assert_eq!(
            built.len(),
            1,
            "both elementwise hops must absorb into the reduce's own BoundOp"
        );
        assert_eq!(built[0].node, sum);
        assert!(matches!(built[0].kind, BoundOpKind::Reduce { .. }));
        assert_eq!(
            built[0].element_body().steps.len(),
            2,
            "one step per absorbed elementwise op"
        );
    }

    #[test]
    fn a_broadcast_operand_has_stride_zero_in_the_broadcast_axis() {
        let mut program = Vec::new();
        let matrix = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4), Extent::Static(8)],
                name: None,
            },
        );
        let bias = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (matrix, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (bias, IndexMap::Affine(map::projection(2, &[1]))),
                ],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("broadcast infers");
        let built = bind(&program, &shapes, &[]).expect("broadcast builds ops");
        let op = built.iter().find(|op| op.node == sum).expect("sum emitted");
        assert_eq!(
            op.operands()[1].1.stride(0),
            0,
            "bias never varies over the batch axis"
        );
        assert_ne!(
            op.operands()[0].1.stride(0),
            0,
            "matrix does vary over the batch axis"
        );
    }

    #[test]
    fn a_conv_window_operand_folds_two_terms_into_one_stride_slot() {
        let mut program = Vec::new();
        let anchor = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4), Extent::Static(2)],
                name: None,
            },
        );
        let signal = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        let window = IndexMap::Affine(map::affine(
            2,
            &[(
                &[
                    crate::map::AxisTerm::scaled(0, 2),
                    crate::map::AxisTerm::scaled(1, 1),
                ],
                0,
            )],
        ));
        let touched = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                operands: alloc::vec![
                    (anchor, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (signal, window)
                ],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("conv window infers");
        let built = bind(&program, &shapes, &[]).expect("conv window builds ops");
        let op = built
            .iter()
            .find(|op| op.node == touched)
            .expect("touched emitted");
        let signal_layout = &op.operands()[1].1;
        assert_eq!(
            signal_layout.strides.len(),
            2,
            "one stride slot per iteration axis"
        );
        assert_ne!(signal_layout.stride(0), 0, "stride term contributes");
        assert_ne!(signal_layout.stride(1), 0, "dilation term contributes");
    }

    #[test]
    fn transpose_layout_has_permuted_strides() {
        let mut program = Vec::new();
        let matrix = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(3), Extent::Static(5)],
                name: None,
            },
        );
        let transposed = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(matrix, IndexMap::Affine(map::projection(2, &[1, 0])))],
                name: None,
            },
        );

        let shapes = shape::infer(&program, &[]).expect("transpose infers");
        let built = bind(&program, &shapes, &[]).expect("transpose builds ops");
        let op = built
            .iter()
            .find(|op| op.node == transposed)
            .expect("transposed emitted");
        let layout = &op.operands()[0].1;
        // matrix is row-major [3, 5]: elem strides are [5, 1]. axis 0 of the
        // operand (stride 5) projects iteration axis 1; axis 1 (stride 1)
        // projects iteration axis 0, so the strides land permuted relative
        // to iteration order.
        assert_eq!(layout.stride(0), 1);
        assert_eq!(layout.stride(1), 5);
    }

    fn elementwise_op() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(10), Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(source, IndexMap::Affine(map::projection(2, &[0, 1])))],
                name: None,
            },
        );
        let shapes = shape::infer(&program, &[]).expect("elementwise infers");
        bind(&program, &shapes, &[])
            .expect("elementwise builds ops")
            .into_iter()
            .next()
            .expect("one op emitted")
    }

    fn scalar_reduction_op() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let shapes = shape::infer(&program, &[]).expect("scalar reduction infers");
        bind(&program, &shapes, &[])
            .expect("scalar reduction builds ops")
            .into_iter()
            .next()
            .expect("one op emitted")
    }

    fn scan_op() -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(8)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::Scan,
                name: None,
            }),
        );
        let shapes = shape::infer(&program, &[]).expect("scan infers");
        bind(&program, &shapes, &[])
            .expect("scan builds ops")
            .into_iter()
            .next()
            .expect("one op emitted")
    }

    #[test]
    fn split_of_an_elementwise_op_yields_contiguous_chunks_with_a_ragged_last() {
        let op = elementwise_op();

        let chunks = op.split(3).expect("extent 10 over 3 parts splits");
        assert_eq!(chunks.len(), 3, "one chunk per part");

        let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
        assert_eq!(
            lengths,
            alloc::vec![3, 3, 4],
            "only the last chunk is ragged"
        );

        // axis 0's stride is 4 (row-major over [10, 4]): chunk k's operand
        // layout is rebased by chunk_start * stride(0), which is exactly
        // what lets a caller treat each chunk's output as a disjoint
        // sub-slice.
        let stride = op.operands()[0].1.stride(0);
        assert_eq!(chunks[0].operands()[0].1.base, op.operands()[0].1.base);
        assert_eq!(
            chunks[1].operands()[0].1.base,
            op.operands()[0].1.base + stride * 3
        );
        assert_eq!(
            chunks[2].operands()[0].1.base,
            op.operands()[0].1.base + stride * 6
        );
    }

    #[test]
    fn split_aligned_rounds_non_final_chunks_down_to_the_alignment() {
        let op = elementwise_op();

        // extent 10, 3 parts: raw_len = 10 / 3 = 3, rounded down to the
        // nearest multiple of 2 is 2 — only the final (already-ragged)
        // chunk absorbs what the rounding shaved off the other two.
        let chunks = op
            .split_aligned(3, 2)
            .expect("extent 10 over 3 parts splits");
        let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
        assert_eq!(
            lengths,
            alloc::vec![2, 2, 6],
            "non-final chunks round down to the alignment, final absorbs the rest"
        );
    }

    #[test]
    fn split_aligned_below_the_alignment_falls_back_to_unaligned() {
        let op = elementwise_op();

        // raw_len = 10 / 3 = 3 is already below alignment 4, so rounding
        // down would zero the chunk out — the doc promises the raw
        // unaligned width is kept instead.
        let chunks = op
            .split_aligned(3, 4)
            .expect("extent 10 over 3 parts splits");
        let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.extents[0]).collect();
        assert_eq!(
            lengths,
            alloc::vec![3, 3, 4],
            "falls back to split's own behavior"
        );
    }

    #[test]
    fn split_aligned_with_alignment_one_matches_split_exactly() {
        let op = elementwise_op();

        let aligned = op
            .split_aligned(3, 1)
            .expect("extent 10 over 3 parts splits");
        let plain = op.split(3).expect("extent 10 over 3 parts splits");
        let aligned_lengths: Vec<u64> = aligned.iter().map(|chunk| chunk.extents[0]).collect();
        let plain_lengths: Vec<u64> = plain.iter().map(|chunk| chunk.extents[0]).collect();
        assert_eq!(aligned_lengths, plain_lengths, "alignment 1 is a no-op");
    }

    #[test]
    fn split_of_a_fused_matmul_reduction_rebases_operands_but_not_out_layout() {
        let (program, _product, sum, _lhs) = matmul_program();
        let shapes = shape::infer(&program, &[512]).expect("matmul infers");
        let op = bind(&program, &shapes, &[])
            .expect("matmul builds ops")
            .into_iter()
            .next()
            .expect("one fused op emitted");
        assert_eq!(op.node, sum);
        let BoundOpKind::Reduce { .. } = &op.kind else {
            panic!("the reduction fused with its elementwise op");
        };

        let chunks = op.split(2).expect("512 rows over 2 parts splits");
        assert_eq!(chunks.len(), 2, "one chunk per part");
        for chunk in &chunks {
            assert!(
                matches!(chunk.kind, BoundOpKind::Reduce { .. }),
                "each chunk is still a reduce"
            );
            // the contracted axis (k) is untouched by a split on the output
            // row axis: every chunk still walks the full contraction.
            assert_eq!(chunk.extents[2], op.extents[2]);
        }
        assert_eq!(chunks[0].extents[0], 256, "rows split evenly in half");
        assert_eq!(chunks[1].extents[0], 256);

        // the fused elementwise op's lhs/rhs operand reads are rebased:
        // chunk 1 starts reading lhs at row 256 (row stride = k = 768).
        let lhs_row_stride = op.operands()[0].1.stride(0);
        assert_eq!(
            chunks[1].operands()[0].1.base,
            op.operands()[0].1.base + lhs_row_stride * 256
        );

        // out_layout stays exactly as the parent's: the interpreter's own
        // per-chunk loop already starts each chunk's leading coordinate at
        // 0, so an unshifted out_layout already yields the 0-based write
        // offsets a `split_at_mut` sub-slice expects (see the `split` doc).
        let BoundOpKind::Reduce {
            out_layout: parent_out,
            ..
        } = &op.kind
        else {
            unreachable!("checked above")
        };
        for chunk in &chunks {
            let BoundOpKind::Reduce {
                out_layout: chunk_out,
                ..
            } = &chunk.kind
            else {
                panic!("chunk reduction");
            };
            assert_eq!(chunk_out, parent_out);
        }
    }

    #[rstest]
    #[case::scalar_reduction(scalar_reduction_op(), 2)]
    #[case::keep_scan_scan(scan_op(), 2)]
    #[case::too_few_parts(elementwise_op(), 1)]
    #[case::extent_smaller_than_parts(elementwise_op(), 999)]
    fn split_returns_none_when_unsound_or_unhelpful(#[case] op: BoundOp, #[case] parts: usize) {
        assert!(op.split(parts).is_none());
    }

    /// A ternary `ScalarOp::Select` node (arity 3, the crate's current
    /// maximum) whose three operands are all held, non-fusing elementwise
    /// predecessors: a single `push` must materialize all three in one
    /// call, proving `push` can ready more than the two `BoundOp`s this
    /// module's docs once claimed as its ceiling — the true bound tracks
    /// `ScalarOp::arity()`, which is why `READY_BATCH_CAPACITY` is 3, not 2.
    #[test]
    fn select_push_emits_three_when_all_three_operands_are_held_and_non_fusing() {
        let mut program = Vec::new();
        let a = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let b = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let c = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let identity = || IndexMap::Affine(map::projection(1, &[0]));
        let held_a = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(a, identity())],
                name: None,
            },
        );
        let held_b = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(b, identity())],
                name: None,
            },
        );
        let held_c = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(c, identity())],
                name: None,
            },
        );
        program.push(Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Select,
            operands: alloc::vec![
                (held_a, identity()),
                (held_b, identity()),
                (held_c, identity()),
            ],
            name: None,
        });

        let shapes = shape::infer(&program, &[]).expect("select program infers");
        // an empty retires list (rather than `live::annotate`'s real kill-flags)
        // makes `retires.contains(operand_node)` false for every node, so
        // every held predecessor fails the fuse check regardless of its
        // projection — isolating exactly what a single push can materialize.
        let building = BoundOpBuilder::new(Vec::new());
        let mut last_emitted_len = 0;
        for expr in program.iter() {
            let emitted = building.push(expr, &shapes).expect("push succeeds");
            last_emitted_len = emitted.len();
        }
        assert_eq!(
            last_emitted_len, 3,
            "the select node's push must materialize all three held, \
             non-fusing predecessors in one call: proves the 0/1/2 bound is \
             wrong, true bound tracks ScalarOp::arity() (3, Select)"
        );
    }

    // THE PROOF: `ShapeTable` and `BoundOpBuilder` compose through the real
    // `PipeExt` surface (`.and_then`, not hand-sequenced calls dressed up as
    // composition), and the ops that composed chain produces for a matmul
    // program are byte-for-byte the same ops `shape::infer` + `bind::bind`
    // (the free-function path every other test in this crate trusts)
    // produce for the identical program.
    #[test]
    fn infer_and_then_build_ops_matches_the_free_function_pipeline() {
        use crate::shape::ShapeTable;
        use proxima_primitives::pipe::PipeExt;

        let (program, _product, sum, _lhs) = matmul_program();
        let outputs: Vec<NodeId> = Vec::new();
        let retires = live::annotate(&program, &outputs);

        let shape_table = ShapeTable::new(&[512]);
        let builder = BoundOpBuilder::new(retires);
        let chain = shape_table.and_then(builder);

        let mut built_via_pipe = Vec::new();
        for expr in &program {
            let batch = proxima_primitives::block_on(Pipe::call(&chain, expr.clone()))
                .expect("shape+op pipe step succeeds");
            built_via_pipe.extend(batch);
        }

        let shapes = shape::infer(&program, &[512]).expect("free-function infer succeeds");
        let built_via_free_function =
            bind(&program, &shapes, &outputs).expect("free-function op building succeeds");

        assert_eq!(built_via_pipe, built_via_free_function);
        assert_eq!(built_via_pipe.len(), 1, "matmul fuses into one op");
        assert_eq!(built_via_pipe[0].node, sum);
    }
}
