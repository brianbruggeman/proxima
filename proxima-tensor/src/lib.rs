//! Tensor vocabulary for proxima: the language you write, reified as data.
//!
//! There is no `Tensor` type here, the same way there is no `Http` type in
//! `proxima-http`. "Tensor" names the domain; what writing it produces is a
//! program — a plain, self-contained `Vec<`[`Expr`]`>` that a backend lowers.
//! "Graph" is not a construct in this crate: it is a *projection* — shapes
//! ([`shape::infer`]) and loop nests ([`nest::lower`]) are read off a program,
//! never built into a separate arena a program has to agree with.
//!
//! # Three expression forms
//!
//! A composed [`Pipe`](proxima_primitives::pipe::Pipe) chain is already a
//! tree, but it lives in the type system: `AndThen<A, B>` cannot be walked,
//! rewritten against runtime shapes, or written to disk. This crate is that
//! same algebra reified, so fusion and placement can rewrite it after the
//! shapes are known.
//!
//! Everything reduces to three expression forms:
//!
//! - [`Expr::Block`] — a leaf. Where data enters.
//! - [`Expr::Zip`] — n-ary elementwise. Arity 1 with a permuting map is
//!   transpose/slice/broadcast; an input whose map is data-dependent is a
//!   gather.
//! - [`Expr::Fold`] — reduce, scan, scatter, contraction, and argmax, split by
//!   [`Keep`] and by whether the output map is data-dependent.
//!
//! `matmul` is a `Fold(+)` over a `Zip(*)`; `softmax` is two folds and three
//! zips. Named operations are an *attribute* (`Expr::name`), never a variant —
//! adding `flash_attention` is a table entry in a backend, not a change here.
//!
//! # Where the expressiveness actually lives
//!
//! Not in the three variants — in [`map`]. An [`AffineMap`] relates an
//! iteration space to an operand's index space, and stride/dilation/offset in
//! that grammar is what makes convolution and windowing expressible without
//! new expression forms. Read that module first.
//!
//! # Distribution stance
//!
//! No partition machinery is built yet, but the representation is shaped for
//! it: a partition is a projection of `&[Expr]` (a contiguous or renumbered
//! sub-slice), a cut edge becomes a named [`Expr::Block`] on the consuming
//! side, [`NodeId`] is a position and is *not* stable across a partitioning
//! pass while a `Block`'s `name` is, and the wire payload of a cut edge is
//! `dtype + shape + bytes` — exactly what [`cpu::Evaluated::get`] already
//! hands back per requested output.
//!
//! # Stream stance
//!
//! Because references point backwards only, a program is a valid
//! append-only stream: every `Expr` is checkable against everything before
//! it the moment it arrives. [`shape::Infer`] and [`nest::Lower`] are the
//! sans-IO cores that do that judging one expression at a time; [`infer`] and
//! [`nest::lower`] are three-line batch drivers over them, and both types
//! also implement [`Pipe`](proxima_primitives::pipe::Pipe) directly (`In =
//! Expr` on `Infer`, `In = (Expr, Shapes)` on `Lower`) — composable with
//! `proxima_primitives::pipe::PipeExt::and_then` into one chain, since
//! `Infer`'s `Out` is exactly `Lower`'s `In`. [`Nest`] execution is not a
//! `Pipe`: it writes into a buffer table it does not own, so [`cpu`]'s
//! `run_nest_into` takes the caller's slice directly instead of moving the
//! whole table through an `In`/`Out` pair — a plain loop over the `Vec<Nest>`
//! a lowering push readies, driven by the composed chain's caller.
//!
//! # Tiers
//!
//! - `alloc` (no_std + alloc): the program representation, [`shape`], and
//!   [`nest`] — everything an out-of-process backend (a GPU kernel emitter,
//!   say) needs to consume a lowered program.
//! - `std` (default): forwards `alloc`, adds `std::error::Error` and the CPU
//!   interpreter ([`cpu`]), which needs floating-point transcendentals.
//! - `config`: the TOML/serde face, std-only, per the layering caveat in
//!   `rust.md`. The `alloc` tier never sees it.
//!
//! # Building `matmul`
//!
//! There is no `matmul` constructor, because `matmul` is not an operation. It
//! is a `Fold(+)` over a `Zip(*)` whose index maps contract a shared
//! dimension, optionally labelled so a backend can reach a tuned kernel:
//!
//! ```
//! use proxima_tensor::{
//!     append, map, DType, Expr, Extent, Fold, FoldInit, IndexMap, Keep, ScalarOp,
//! };
//!
//! let mut program = Vec::new();
//! // sequence length is symbolic: it is not known until a request arrives.
//! let lhs = append(
//!     &mut program,
//!     Expr::Block {
//!         dtype: DType::Float32,
//!         shape: vec![Extent::Symbolic(0), Extent::Static(768)],
//!         name: None,
//!     },
//! );
//! let rhs = append(
//!     &mut program,
//!     Expr::Block {
//!         dtype: DType::Float32,
//!         shape: vec![Extent::Static(768), Extent::Static(3072)],
//!         name: None,
//!     },
//! );
//!
//! // iteration space (i, j, k). k is shared, so it is the contracted dim.
//! let product = append(
//!     &mut program,
//!     Expr::Zip {
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
//!     Expr::Fold(Fold {
//!         dtype: DType::Float32,
//!         body: ScalarOp::Add,
//!         init: FoldInit::Zero,
//!         operand: product,
//!         in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
//!         // dropping k is the reduction.
//!         out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
//!         keep: Keep::Last,
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
//! Change `keep` to [`Keep::All`] and the same fold is a scan. Make `out_map`
//! data-dependent and it is a scatter. Neither needs an expression form.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
pub mod cpu;
pub mod dtype;
pub mod error;
pub mod expr;
pub mod live;
pub mod map;
pub mod nest;
pub mod shape;
#[cfg(feature = "config")]
pub mod spec;

#[cfg(feature = "std")]
pub use cpu::{Evaluated, evaluate, evaluate_parallel};
pub use dtype::DType;
pub use error::TensorError;
pub use expr::{Expr, Extent, Fold, FoldInit, Keep, NodeId, ScalarOp, append};
pub use live::annotate;
pub use map::{AffineMap, AffineTerm, DimExpr, IndexMap, affine, projection};
pub use nest::{GatherAccess, Lower, Nest, Reduction, StridedView, lower};
pub use shape::{Infer, Shapes, infer};
