//! Process-global runtime emit filter — the std-tier half of proxima's emit
//! gate. Callsites compile in unconditionally (the compile floor stays at
//! `trace`); whether a given callsite actually records is decided here at
//! runtime, cached per-callsite by a [`CallsiteGate`](crate::emit::CallsiteGate)
//! keyed on [`current_generation`].
//!
//! Default floor is `error`: with no `RUST_LOG` set, only `error!` records.
//! `RUST_LOG` overrides via the [`EnvFilter`] front-end (same grammar as
//! `tracing-subscriber`'s `EnvFilter`), read lazily on first emit. [`install`]
//! swaps a new filter in at runtime and bumps the generation so every cached
//! callsite re-decides.

use std::sync::LazyLock;

use proxima_core::live::{Live, LiveControl, live};

use crate::emit::{CompiledEmit, Coord, Decision, EnvFilter, FilterGeneration};

static GENERATION: FilterGeneration = FilterGeneration::new();

static FILTER: LazyLock<(Live<CompiledEmit>, LiveControl<CompiledEmit>)> =
    LazyLock::new(|| live(EnvFilter::from_default_env()));

/// The current filter generation — a callsite gate caches its decision against
/// this; [`install`] bumps it to invalidate every cache.
#[must_use]
pub fn current_generation() -> u32 {
    GENERATION.current()
}

/// Decide whether `target` (a module path) at `coord` (a level) records under
/// the installed filter. Called only on a callsite-gate cache miss.
#[must_use]
pub fn decide(target: &str, coord: Coord) -> Decision {
    FILTER.0.read(|filter| filter.decide(target, coord))
}

/// Install a compiled filter as the process-global emit filter and invalidate
/// every callsite cache. The replacement for `tracing_subscriber::set_global_default`.
pub fn install(filter: CompiledEmit) {
    FILTER.1.replace(filter);
    GENERATION.bump();
}

/// Install the filter parsed from `RUST_LOG` (default `error` when unset). Idempotent.
pub fn install_from_env() {
    install(EnvFilter::from_default_env());
}

/// Format an emit field value into owned `Bytes` for the `?x` / `%x` field forms.
/// Only runs on a kept record (the gate already passed), so the allocation is cold.
#[must_use]
pub fn fmt_bytes(args: core::fmt::Arguments<'_>) -> bytes::Bytes {
    bytes::Bytes::from(alloc::format!("{args}"))
}
