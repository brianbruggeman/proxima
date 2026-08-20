//! Tensor vocabulary for proxima: the language you write, reified as data.
//!
//! There is no `Tensor` type here, the same way there is no `Http` type in
//! `proxima-http`. "Tensor" names the domain; what writing it produces is a
//! program — a plain, self-contained `Vec<`[`Op`]`>` that an executor runs
//! against a call's symbol bindings. "Graph" is not a construct in this
//! crate: it is a *projection* — shapes ([`shape::infer`]) and bound
//! addressing ([`bind::bind`]) are read off a program, never built into a
//! separate arena a program has to agree with.
//!
//! # Three op forms
//!
//! A composed [`Pipe`](proxima_primitives::pipe::Pipe) chain is already a
//! tree, but it lives in the type system: `AndThen<A, B>` cannot be walked,
//! rewritten against runtime shapes, or written to disk. This crate is that
//! same algebra reified, so fusion and placement can rewrite it after the
//! shapes are known.
//!
//! Everything reduces to three op forms:
//!
//! - [`Op::Input`] — a leaf. Where data enters.
//! - [`Op::Elementwise`] — n-ary elementwise. Arity 1 with a permuting index
//!   pattern is transpose/slice/broadcast; an operand whose index pattern is
//!   data-dependent is a gather.
//! - [`Op::Reduce`] — reduce, scan, scatter, contraction, and argmax, split
//!   by [`Keep`] and by whether the output index pattern is data-dependent.
//!
//! `matmul` is a `Reduce(+)` over an `Elementwise(*)`; `softmax` is two
//! reduces and three elementwise ops. Named operations are an *attribute*
//! (`Op::name`), never a variant — adding `flash_attention` is a table entry
//! in a backend, not a change here.
//!
//! # Where the expressiveness actually lives
//!
//! Not in the three variants — in [`map`]. An [`IndexPattern`] relates an
//! iteration space to an operand's index space, and stride/dilation/offset
//! in that einsum-style grammar is what makes convolution and windowing
//! expressible without new op forms. Read that module first.
//!
//! # Distribution stance
//!
//! The function [`partition::partition_at`] is the first slice of this. A
//! partition is a projection of `&[Op]` (a contiguous or renumbered
//! sub-slice); a cut edge becomes a named [`Op::Input`] on the consuming
//! side; [`NodeId`] is a position and is *not* stable across a
//! partitioning pass while an `Input`'s `name` is; and the wire payload of a
//! cut edge is `dtype + shape + bytes` — exactly what
//! [`cpu::Evaluated::get`] already hands back per requested output.
//! [`cpu::evaluate_named`] is the consumer-side counterpart — it binds
//! `Op::Input`s by name instead of position, which is what a renumbered
//! consumer program needs.
//!
//! Still not built — serializing the cut payload onto an actual wire, and
//! choosing a cut point automatically. `partition_at` takes the cut point
//! as an argument and does not choose one itself.
//!
//! # Stream stance
//!
//! Because references point backwards only, a program is a valid
//! append-only stream: every `Op` is checkable against everything before it
//! the moment it arrives. [`shape::ShapeTable`] and [`bind::BoundOpBuilder`]
//! are the sans-IO cores that do that judging one op at a time; [`infer`]
//! and [`bind::bind`] are three-line batch drivers over them, and both types
//! also implement [`Pipe`](proxima_primitives::pipe::Pipe) directly (`In =
//! Op` on `ShapeTable`, `In = (Op, Shapes)` on `BoundOpBuilder`) —
//! composable with `proxima_primitives::pipe::PipeExt::and_then` into one
//! chain, since `ShapeTable`'s `Out` is exactly `BoundOpBuilder`'s `In`.
//! [`cpu::Interpreter`] is the third stage: `In = Vec<BoundOp>`, `Out = ()`,
//! over a caller-provided buffer table it borrows rather than owns — the
//! same interior-state discipline `ShapeTable` and `BoundOpBuilder` already
//! apply to their own per-record state, not a carve-out for execution.
//! `BoundOpBuilder::Out` is `Vec<BoundOp>` because one pushed `Op` can ready
//! zero, one, or two `BoundOp`s (fusion's own lookahead); `Interpreter::In`
//! matches that batch exactly, so the full
//! `shapes.and_then(bind).and_then(run)` composes into one chain —
//! `Second::In = First::Out` holds at both joins — and the caller drives
//! that one chain once per `Op` record; see `cpu`'s test module for the
//! composed chain and the reasoning.
//!
//! # One record kind, not two
//!
//! A [`bind::BoundOp`] is not a second intermediate representation: it is an
//! [`Op`] whose addressing has been bound against one call's symbol
//! bindings — the same `Elementwise`/`Reduce` shape, with a
//! [`bind::Layout`] standing in for a symbolic [`IndexMap`] and bound
//! iteration extents standing in for a symbolic [`Extent`] list.
//! [`shape::ShapeTable`] resolves shape; [`bind::BoundOpBuilder`] binds
//! layout *and* decides fusion in the same pass, because whether an
//! operand's layout is worth binding on its own depends entirely on whether
//! a consuming reduce can absorb it — splitting that into two stages would
//! mean binding layout for elementwise ops that are about to be thrown away.
//! The stored program (`Vec<Op>`) stays symbolic; layout is an annotation on
//! the in-flight record produced for one call, never part of what gets
//! stored.
//!
//! # Tiers
//!
//! - `alloc` (no_std + alloc): the program representation, [`shape`], and
//!   [`mod@bind`] — everything an out-of-process backend (a GPU kernel emitter,
//!   say) needs to consume a program with bound addressing.
//! - `std` (default): forwards `alloc`, adds `std::error::Error` and the CPU
//!   interpreter ([`cpu`]), which needs floating-point transcendentals.
//! - `config`: the TOML/serde face, std-only, per the layering caveat in
//!   `rust.md`. The `alloc` tier never sees it.
//!
//! # Building `matmul`
//!
//! There is no `matmul` constructor, because `matmul` is not an operation.
//! It is a `Reduce(+)` over an `Elementwise(*)` whose index patterns
//! contract a shared axis, optionally labelled so a backend can reach a
//! tuned kernel:
//!
//! ```
//! use proxima_tensor::{
//!     append, map, DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp,
//! };
//!
//! let mut program = Vec::new();
//! // sequence length is symbolic: it is not known until a request arrives.
//! let lhs = append(
//!     &mut program,
//!     Op::Input {
//!         dtype: DType::Float32,
//!         shape: vec![Extent::Symbolic(0), Extent::Static(768)],
//!         name: None,
//!     },
//! );
//! let rhs = append(
//!     &mut program,
//!     Op::Input {
//!         dtype: DType::Float32,
//!         shape: vec![Extent::Static(768), Extent::Static(3072)],
//!         name: None,
//!     },
//! );
//!
//! // iteration space (i, j, k). k is shared, so it is the contracted axis.
//! let product = append(
//!     &mut program,
//!     Op::Elementwise {
//!         dtype: DType::Float32,
//!         body: ScalarOp::Multiply,
//!         operands: vec![
//!             (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
//!             (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
//!         ],
//!         name: None,
//!     },
//! );
//!
//! let sum = append(
//!     &mut program,
//!     Op::Reduce(Reduce {
//!         dtype: DType::Float32,
//!         body: ScalarOp::Add,
//!         init: ReduceInit::Zero,
//!         operand: product,
//!         in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
//!         // dropping k is the reduction.
//!         out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
//!         keep: Keep::Reduce,
//!         name: Some("matmul".into()),
//!     }),
//! );
//!
//! let shapes = proxima_tensor::infer(&program, &[512])?;
//! assert_eq!(shapes.of(sum), &[512, 3072]);
//! assert_eq!(program[sum.0 as usize].name(), Some("matmul"));
//! # Ok::<(), proxima_tensor::TensorError>(())
//! ```
//!
//! Change `keep` to [`Keep::Scan`] and the same reduce is a scan. Make
//! `out_map` data-dependent and it is a scatter. Neither needs a new op form.

#![cfg_attr(not(feature = "std"), no_std)]

// `Op` (op.rs), `Shapes` (shape.rs), and `BoundOpBuilder` (bind.rs) are all
// `alloc::vec::Vec`/`BTreeMap`-backed today, so the `alloc` feature is not
// yet optional — this is a gap to close, not a permanent boundary. See the
// itemized no-alloc blocker list carried alongside this change for what a
// fixed-capacity/caller-provided-storage version of each would need.
#[cfg(not(any(feature = "std", feature = "alloc")))]
compile_error!(
    "proxima-tensor currently requires the `alloc` feature (or `std`, which implies \
     it); a no-alloc tier is on the roadmap but not yet implemented"
);

#[cfg(feature = "alloc")]
extern crate alloc;

// bind/live/map/op/shape are the `Vec`/`BTreeMap`-backed core (see the
// blocker list); dtype and error stand alone and would compile with neither
// `std` nor `alloc`, but there is nothing useful to build without the core,
// so the whole crate body is gated on the same condition as the
// `compile_error!` above rather than leaving a half-populated crate root.
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod align;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod bind;
// element conversion (`convert::Convert`, a `Pipe`) and decimal-as-conversion
// (`convert::decimal`) need neither `alloc` nor `std` — slices and core
// arithmetic only — so this stays ungated, same as `dtype`.
pub mod convert;
#[cfg(feature = "std")]
pub mod cpu;
pub mod dtype;
// `error::TensorError` names `op::NodeId` on nearly every variant, so it is
// pulled into the same gate as `op` even though most of its own variants
// (post `config`-scoping above) carry no alloc-only data themselves — the
// blocker is the shared module, not the type.
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod error;
#[cfg(feature = "instrument")]
pub mod instrument;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod live;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod map;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod op;
// pure over `&[Op]`/`NodeId`/`shape::infer`, so it lives at the same tier as
// both — see the module's own doc for what it produces and why no new type
// hosts it.
// runtime execution policy over `sized`'s compiled floor. `std`-gated, not
// `config`-gated: `cpu` reads it on every threaded dispatch and must compile
// without a loader, so the struct plus its `COMPILED` value live at `std`
// while the conflaguration file/env layering is `config`-only. The
// `no_std + alloc` tier compiles neither this nor `cpu`, and keeps `sized`
// as its only configuration.
#[cfg(feature = "std")]
pub mod policy;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod partition;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod shape;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod sized;
#[cfg(feature = "config")]
pub mod spec;
// also active under plain `cfg(test)` (no feature flag needed) so this
// crate's own `#[cfg(test)] mod tests` blocks (spec.rs, cpu.rs) can reach it
// via `crate::test_support` without requiring `--features test-support` on
// an ordinary `cargo test`/`nextest run`.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use align::AlignedBuffer;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use bind::{
    BodyStep, BoundOp, BoundOpBuilder, BoundOpKind, ComposedBody, Layout, Lookup, StepArg, bind,
};
pub use convert::{Convert, SimdConvert};
#[cfg(feature = "std")]
pub use cpu::{
    Evaluated, Interpreter, TypedBuffer, evaluate, evaluate_parallel, evaluate_typed,
    evaluate_with_scratch,
};
pub use dtype::DType;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use error::TensorError;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use live::annotate;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use map::{AxisIndex, AxisTerm, IndexMap, IndexPattern, affine, projection};
#[cfg(any(feature = "std", feature = "alloc"))]
pub use op::{Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append};
#[cfg(any(feature = "std", feature = "alloc"))]
pub use shape::{ShapeTable, Shapes, infer};
