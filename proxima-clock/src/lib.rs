//! Hardware-clock source polymorphism, expressed as [`proxima_primitives`]
//! pipes instead of a bespoke clock trait — principle 5 ("source
//! polymorphism", today applied to the reactor's readiness sources)
//! applied to time.
//!
//! # Why this is a pipe, not a trait
//!
//! A monotonic tick source — a `read_volatile` on a memory-mapped
//! register, `mrs CNTVCT_EL0`, `rdtsc`, a DMA'd timestamp a device writes
//! into memory — takes nothing and produces a reading. That is exactly
//! [`proxima_primitives::pipe::primitives::Pipe`]'s source shape (`In =
//! (), Out = Ticks`), already in the workspace, already `no_std` +
//! no-alloc, already box-free. Binding a new hardware backend is
//! "implement `Pipe` for your type" — zero edits to this crate, and not
//! even this crate's own trait: it is the workspace's general pipe
//! algebra, the same one the reactor's readiness sources, HTTP handlers,
//! and codec frame pipelines already speak.
//!
//! This crate therefore does **not** define a `MonotonicClock` or
//! `Sleeper` capability trait. Both shapes were checked against the
//! binary rule ("can this be expressed with a pipe? If yes, it does not
//! get a type") by writing the composition, not by arguing about it:
//!
//! - A tick source is `Pipe<In = (), Out = Ticks, Err = Infallible>` —
//!   the source form from `Pipe`'s own module doc's "four forms" table.
//! - A sleeper is `Pipe<In = Ticks, Out = (), Err = Infallible>` whose
//!   future genuinely pends until the deadline — the sink form, mirrored.
//!   Any type implementing this shape, with a future that registers a
//!   real waker (a timer interrupt, `prime`'s timer wheel), is a sleeper;
//!   there is nothing else to implement. This crate ships no reference
//!   sleeper — a correct one needs an interrupt or a runtime's timer
//!   wheel, which is a runtime-tier concern, not a leaf crate's.
//! - Converting ticks to wall-clock time is `Pipe<In = Ticks, Out =
//!   UnixNanos, Err = Infallible>` — [`anchor::ToUnixNanos`], composed
//!   with a source via `.and_then`. There is no `WallClock<C>` wrapper
//!   type: the transform stage IS the monotonic-to-wall-clock bridge.
//! - Arbitrating between redundant tick sources (a GPS PPS signal, a PTP
//!   grandmaster, a local oscillator, each potentially going stale) is
//!   [`proxima_primitives::pipe::fan_in::FanIn`] — see
//!   `tests/fan_in_multi_source.rs` for a worked example. No bespoke
//!   arbitration machinery needed.
//!
//! What this crate *does* provide: two domain value types
//! ([`ticks::Ticks`], [`unix_nanos::UnixNanos`]) so a caller cannot
//! accidentally mix tick-domain and wall-clock-domain arithmetic, and two
//! small stateful cells ([`coarse::TickCell`], [`anchor::AnchorCell`])
//! that a hardware-facing `Pipe` impl or the wall-clock bridge can hold —
//! plain data-and-synchronization types, not a parallel API to `Pipe`.
//!
//! # Tiers
//!
//! - **Bare `no_std` + no-alloc (default core)**: [`ticks`], [`unix_nanos`],
//!   [`coarse`], [`anchor`] — zero `alloc::`, zero `Box`/`Vec`/`Arc`/
//!   `String`, builds for `thumbv7m-none-eabi` with
//!   `--no-default-features`.
//! - **`config` feature (std, opt-in)**: [`config`] — the principle-4
//!   config + fluent-builder surface for [`anchor::AnchorCell`]'s initial
//!   `(ticks, unix_nanos)` pair. `conflaguration` is std-only, so this
//!   lives at the std composition boundary; the no-alloc core never
//!   depends on it.
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(clippy::pedantic)]

pub mod anchor;
pub mod coarse;
#[cfg(feature = "config")]
pub mod config;
mod seq_u64s;
pub mod ticks;
pub mod unix_nanos;
