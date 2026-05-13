//! Typed emit-control + filtering surface — a first-class replacement for
//! `RUST_LOG` / `tracing-subscriber`'s `EnvFilter` that matches the familiar
//! forms and improves on them.
//!
//! The surface composes existing proxima primitives rather than replacing them:
//! the flat [`crate::level::Level`] is the depth-1 special case of a [`Coord`],
//! and the filter installs as a Pipe alongside the existing
//! [`crate::pipes::FilterByLevelPipe`]. Familiar inputs — a `Level` floor, a
//! `RUST_LOG`-style string, a TOML config — all lower to one typed compiled
//! filter; hierarchical named levels and compile-time discoverability are the
//! additions on top.
//!
//! Tier: the value types ([`Coord`]) are `no_std`/no-alloc (T0-eligible); the
//! module is gated at the `alloc` tier because the compiled rule table (C2)
//! allocates. The config loader lands at the `std` tier.

mod compiled;
mod coord;
mod decision;
mod env_filter;
mod gate;
mod level_tree;

pub use crate::sampler::Decision;
pub use compiled::{CompiledEmit, EmitRule, MatchMode};
pub use coord::{Coord, CoordError, SEG_MAX};
pub use decision::EmitThreshold;
pub use env_filter::EnvFilter;
pub use gate::{CallsiteGate, FilterGeneration};
pub use level_tree::{HierLevel, LevelTree, LevelTreeBuilder, LevelTreeError};

// T2 — std: the conflaguration config + fluent builder.
#[cfg(feature = "std")]
mod config;
#[cfg(feature = "std")]
pub use config::{EmitConfig, EmitConfigError, EmitLayerBuilder, TargetSpec};

// T2 — std: the process-global runtime filter the log macros gate against.
#[cfg(feature = "std")]
pub mod global;

// T2 — std: the proxima-native `error!`/`warn!`/`info!`/`debug!`/`trace!` macros.
// `#[macro_export]` hoists them to the crate root regardless of this module path.
#[cfg(feature = "std")]
mod macros;
