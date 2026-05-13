//! Compile-time marker traits for dispatch units.
//!
//! Empty traits — zero runtime cost. Each type impls the markers it
//! qualifies for; operators (e.g. `AndThen<A, B>` in proxima-process)
//! propagate via blanket impls.
//!
//! Promoted from `proxima-process-protocol::markers` to `proxima-core`
//! as Phase C.1 of the cliff-extension plan
//! (the we-want-the-dependency-drifting-bear plan) so the
//! whole workspace can use the marker vocabulary without depending on
//! `proxima-process-protocol`. `proxima-process-protocol::markers`
//! continues to re-export from here for backward compatibility.
//!
//! # Categories
//!
//! - **Tier compatibility:** [`NoStd`], [`AllocFree`]
//! - **Effect-absence (negative — prove no effect X):**
//!   [`WithoutFilesystem`], [`WithoutNetwork`], [`WithoutSpawn`],
//!   [`WithoutTime`], [`WithoutRandom`]
//! - **Effect-absence (umbrella):** [`IsPure`] (no effects at all)
//! - **Determinism hierarchy:** [`Deterministic`], [`Reproducible`],
//!   [`IdempotentSideEffectFree`], [`Commutative`]
//!
//! # Why effect markers are NEGATIVE
//!
//! Rust trait-overlap rules make POSITIVE effect propagation
//! (`HasX` for `AndThen<A, B>` if either A or B has it) impossible
//! without specialization — two blanket impls
//! (`impl<A: HasX, B> HasX for AndThen<A, B>` AND
//! `impl<A, B: HasX> HasX for AndThen<A, B>`) would overlap when
//! both sides have the marker. Specialization is unstable.
//!
//! Negative markers compose via AND (`WithoutX` for `AndThen<A, B>`
//! requires BOTH sides). AND semantics work with blanket impls; OR
//! doesn't.
//!
//! The negative form is also the load-bearing one for sandbox use
//! cases: "prove this chain CANNOT touch X" is the security
//! assertion. The positive form ("this chain MAY touch X") isn't
//! useful for users — they already know what grounds they put in.

/// Compiles under `#![no_std]` (uses only `core::*` + optionally
/// `alloc::*`). Orthogonal to [`AllocFree`].
pub trait NoStd {}

/// No heap allocations on the dispatch hot path (no
/// `Box`/`Vec`/`String`/`Arc` allocated per call). Construction-time
/// allocations are fine; per-dispatch allocations are not.
///
/// Stronger than [`NoStd`]: an `AllocFree` type may still link `std`,
/// it just can't allocate per call. Typically used in tandem.
pub trait AllocFree {}

/// No side effects at all. Pure data flow: same input → same output,
/// no external state read, no external state written.
///
/// Stronger than [`Deterministic`] (which only requires same
/// in→same out): `IsPure` also forbids any I/O or kernel
/// interaction. Implies all `Without*` markers (a `IsPure` ground
/// is without filesystem, without network, without spawn, etc.).
pub trait IsPure {}

/// Type does not read or write the host filesystem.
///
/// Assert via `T: WithoutFilesystem` to prove at compile time that a
/// chain has no filesystem access. Grounds that touch the real
/// filesystem (e.g. `host_read`, `host_write`) do NOT impl this; pure
/// grounds (`canned`, `empty`, `deny`) do. Operators propagate via
/// AND: `AndThen<A, B>: WithoutFilesystem` requires both sides.
pub trait WithoutFilesystem {}

/// Type does not open or accept network connections.
pub trait WithoutNetwork {}

/// Type does not spawn subprocesses or send signals.
pub trait WithoutSpawn {}

/// Type does not read the system clock or depend on wall-clock time.
pub trait WithoutTime {}

/// Type does not read OS entropy or depend on randomness.
pub trait WithoutRandom {}

/// Same input → same output, modulo construction-time configuration.
/// Required for any dispatch unit in a replay chain.
///
/// A ground that reads time is `Deterministic` only if its clock is
/// fixed at construction; `Real` clocks disqualify it. Same rule for
/// entropy (`Seeded { seed }` vs `Os`).
pub trait Deterministic {}

/// Deterministic AND stable across versions. A `Reproducible` ground
/// produces the same output today and a year from now. Distinguishes
/// from [`Deterministic`] grounds whose output may shift on internal
/// changes (e.g. a hash function whose algorithm changes).
pub trait Reproducible: Deterministic {}

/// Calling the dispatch twice with the same input is observationally
/// identical to calling it once. Useful for caching, retry-safety,
/// at-least-once delivery semantics.
pub trait IdempotentSideEffectFree {}

/// The order in which two dispatches happen does not affect the
/// observable outcome. Useful for parallel execution and reordering
/// optimizations.
pub trait Commutative {}

/// The dispatch's in-flight future is safe to drop at any await point: a
/// dropped `call` leaves no observable partial state — the peer/store sees a
/// complete operation or nothing, never a torn one.
///
/// This is the cancellation-safety contract the concurrent fan combinators
/// (`Race`, `ScatterGather`) require: they cancel losing branches by *dropping*
/// their futures, so only a `DropSafe` pipe may be raced. The sequential
/// `FanOut` awaits every branch and drops none, so it does NOT require it.
///
/// Justified impls: detached blocking work (`spawn_blocking`/offload completes
/// regardless of the caller's fate), purely computational pipes (codecs,
/// in-memory transforms), datagram-atomic transports. NOT justified: a
/// streaming send whose partial body leaves the peer mid-message.
pub trait DropSafe {}
