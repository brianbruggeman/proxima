//! Reverse-mode autodiff over a [`proxima_tensor`] program.
//!
//! There is no `Tensor<B, D>` value type here and no `Optimizer` trait —
//! the graph already IS the value (`proxima_tensor`'s own crate doc:
//! "there is no `Tensor` type... what writing it produces is a program"),
//! and Adam is nine elementwise nodes over existing `Op::Input` leaves,
//! not a method on anything. Four pieces:
//!
//! - [`adjoint::differentiate`] — the adjoint transform, `&[Op] -> Differentiated`,
//!   a pure synchronous function, not a [`proxima_primitives::pipe::Pipe`]
//!   (see that module's own doc for why forcing one would be wrong).
//! - [`activation::relu`] / [`activation::softmax`] — graph-building
//!   functions composing existing [`proxima_tensor::op::ScalarOp`]s, never
//!   a new variant.
//! - [`optimizer::adam_step`] — the Adam update as an elementwise
//!   expression over `(param, grad, m, v, step)`.
//! - [`sparse::dedupe_and_sum_rows`] — the host-side scatter-add a gathered
//!   operand's adjoint (`adjoint::GatheredContribution`) needs applied back
//!   onto its full shape, `O(touched x row_len)` rather than the dense
//!   `O(vocab x touched)` mask composition `proxima-tensor` itself uses for
//!   a statically-known destination.
//!
//! Gradient-to-parameter binding is `Differentiated::gradient_of_named`
//! (dense) or `Differentiated::gathered_gradients_of_named` (a gathered
//! operand, e.g. an embedding table), both lookups over
//! [`proxima_tensor::op::Op::Input::name`] — the same name
//! [`proxima_tensor::cpu::evaluate_named`] already binds by — not a second
//! tree structure next to the program.
//!
//! # Tiers
//!
//! [`adjoint`], [`activation`], `expr` (private), and [`sparse`] touch nothing but
//! `proxima-tensor`'s alloc-tier surface (`op`/`map`/`shape`/`error`) and
//! build under `--no-default-features --features alloc`: pure graph
//! construction and host-side buffer reduction, no evaluation. `sparse`
//! additionally touches nothing but `alloc::collections::BTreeMap`, so it
//! needs no new dependency to stay off the `O(n^2)` linear-scan path a
//! `Vec`-based dedupe would otherwise take.
//! [`optimizer::adam_step`]/[`optimizer::step_input`] are the same tier.
//! `optimizer::AdamConfig`'s `bon`/`conflaguration`/`serde` derive stack is
//! behind the `config` feature (std-gated, mirroring `proxima-tensor`'s own
//! `config` feature); running any of it — [`proxima_tensor::cpu::evaluate_named`]
//! — needs `std`.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(feature = "std", feature = "alloc")))]
compile_error!(
    "proxima-autograd currently requires the `alloc` feature (or `std`, which implies \
     it) -- proxima-tensor's own op/map/shape modules need it"
);

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod activation;
pub mod adjoint;
pub mod error;
pub(crate) mod expr;
pub mod optimizer;
pub mod sparse;

pub use error::AutogradError;
