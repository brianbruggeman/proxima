//! Typed configuration for [`PrimeRuntime`](super::os::runtime::PrimeRuntime).
//! Two equally first-class entry points: a `conflaguration`-derived
//! `Settings` struct (env-driven + file-driven) and a fluent builder. Both
//! compose — a builder can take a `PrimeConfig` as its starting point and
//! override individual fields.

#![cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]

use std::fmt;
use std::str::FromStr;

use conflaguration::{Settings, Validate};

/// Parse error for prime config fields. Required because conflaguration's
/// `resolve_with` plumbing demands a `std::error::Error` impl on the
/// parser's error type — plain `String` does not satisfy that bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(String);

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

/// Number of worker cores prime spawns.
///
/// `Auto` queries `num_cpus::get_physical()` at construction; `Count(n)`
/// pins explicitly. The env-driven shape accepts the literal string
/// `"auto"` or any positive integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreSelection {
    #[default]
    Auto,
    Count(usize),
}

impl CoreSelection {
    /// Resolve to a concrete worker count. `Auto` queries physical
    /// cores; `Count(0)` clamps to 1.
    #[must_use]
    pub fn resolve(self) -> usize {
        match self {
            Self::Auto => num_cpus::get_physical().max(1),
            Self::Count(count) => count.max(1),
        }
    }
}

impl FromStr for CoreSelection {
    type Err = ParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let trimmed = input.trim();
        if trimmed.eq_ignore_ascii_case("auto") || trimmed.is_empty() {
            return Ok(Self::Auto);
        }
        trimmed.parse::<usize>().map(Self::Count).map_err(|err| {
            ParseError(format!(
                "cores: '{trimmed}' is not 'auto' or a positive integer: {err}"
            ))
        })
    }
}

/// WHERE the `cores` workers pin, as opposed to [`CoreSelection`] which is
/// HOW MANY. `Packed` (default) keeps the historical `0..N` layout; `Offset`
/// shifts that window to `start..start+N` (the one-knob fix for two colocated
/// runtimes — give the second an offset past the first); `Cores` pins an
/// explicit physical-core list (NUMA / hand-placed), and its length also fixes
/// the worker count. The env form DWIMs: empty/`"packed"` -> Packed, a bare
/// integer `"4"` -> Offset(4), a list `"4,5,6,7"` or range `"4-7"` -> Cores.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Affinity {
    /// Pin workers to physical cores `0..N`.
    Packed,
    /// Pin workers to `start..start+N`.
    Offset(usize),
    /// Pin workers to an explicit physical-core list (also fixes the count).
    Cores(Vec<usize>),
    /// DON'T pin — workers float, the OS schedules and migrates them. The
    /// DEFAULT: it keeps the per-core shared-nothing architecture (the
    /// throughput win) while letting a worker dodge a noisy neighbour, so it
    /// shows well on a shared / dev box. The pinned variants are the opt-in for
    /// a dedicated box where cache/NUMA locality is worth the set-and-forget.
    #[default]
    Float,
}

impl Affinity {
    /// Physical-core indices for `count` workers under this placement. `Float`
    /// has no pinning, so it reports the packed indices for any caller that
    /// still wants a count-shaped answer; the build path routes it to the
    /// unpinned constructor instead of pinning to these.
    #[must_use]
    pub fn placement(&self, count: usize) -> Vec<usize> {
        match self {
            Self::Packed | Self::Float => (0..count).collect(),
            Self::Offset(start) => (*start..start.saturating_add(count)).collect(),
            Self::Cores(cores) => cores.iter().copied().take(count).collect(),
        }
    }

    /// When the placement itself fixes the worker count (`Cores`), that count;
    /// otherwise `None` and the count comes from [`CoreSelection`].
    #[must_use]
    pub fn fixed_count(&self) -> Option<usize> {
        match self {
            Self::Cores(cores) => Some(cores.len()),
            _ => None,
        }
    }
}

impl FromStr for Affinity {
    type Err = ParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let trimmed = input.trim();
        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("float")
            || trimmed.eq_ignore_ascii_case("unpinned")
        {
            return Ok(Self::Float);
        }
        if trimmed.eq_ignore_ascii_case("packed") {
            return Ok(Self::Packed);
        }
        if let Some((start, end)) = trimmed.split_once('-') {
            let start = parse_core_index(start)?;
            let end = parse_core_index(end)?;
            if end < start {
                return Err(ParseError(format!(
                    "affinity range '{trimmed}': end {end} precedes start {start}"
                )));
            }
            return Ok(Self::Cores((start..=end).collect()));
        }
        if trimmed.contains(',') {
            let cores = trimmed
                .split(',')
                .map(parse_core_index)
                .collect::<Result<Vec<usize>, ParseError>>()?;
            return Ok(Self::Cores(cores));
        }
        Ok(Self::Offset(parse_core_index(trimmed)?))
    }
}

fn parse_core_index(value: &str) -> Result<usize, ParseError> {
    value.trim().parse::<usize>().map_err(|err| {
        ParseError(format!(
            "affinity: '{}' is not a core index: {err}",
            value.trim()
        ))
    })
}

fn parse_affinity(value: &str) -> Result<Affinity, ParseError> {
    Affinity::from_str(value)
}

/// Choice of [`BackgroundPool`](crate::runtime::BackgroundPool)
/// implementation for cross-thread CPU-bound work.
///
/// `Rayon` (default) plugs in `RayonBackgroundPool` (feature `rayon`)
/// — work-stealing across a fixed thread count, the right shape for
/// CPU-bound parallel compute. `Inline` falls back to a one-off
/// `std::thread::spawn` per job — fine for tests and small workloads.
/// `None` attaches no pool; calls to
/// [`Runtime::spawn_background_blocking`](crate::runtime::Runtime::spawn_background_blocking)
/// still work via the inline fallback baked into `PrimeRuntime` itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoolKind {
    #[default]
    Rayon,
    Inline,
    None,
}

impl FromStr for PoolKind {
    type Err = ParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "rayon" => Ok(Self::Rayon),
            "inline" => Ok(Self::Inline),
            "none" | "off" => Ok(Self::None),
            other => Err(ParseError(format!(
                "background_pool: '{other}' is not 'rayon', 'inline', or 'none'"
            ))),
        }
    }
}

fn parse_core_selection(value: &str) -> Result<CoreSelection, ParseError> {
    CoreSelection::from_str(value)
}

fn parse_pool_kind(value: &str) -> Result<PoolKind, ParseError> {
    PoolKind::from_str(value)
}

/// How the prime worker and tokio coexist — "the split" between prime-native
/// execution and the tokio escape hatch.
///
/// `None` (default): pure prime, no tokio runtime context — `tokio::*` APIs
/// from a prime task panic. `Sister`: each prime worker enters a per-core
/// sister tokio current-thread runtime on its OWN thread (classic compat;
/// `tokio::spawn` is a cross-thread hop). `Inverted` (D2): each prime worker
/// owns its sister and ticks the executor inside `sister.block_on`, so raw
/// `tokio::spawn` from a prime task takes tokio's LOCAL fast path. `Sister`
/// needs the `prime-tokio-compat` feature; `Inverted` needs
/// `prime-tokio-compat-inverted` — selecting one without its feature is a
/// build-time error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompatMode {
    #[default]
    None,
    Sister,
    Inverted,
}

impl FromStr for CompatMode {
    type Err = ParseError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "none" | "off" | "" => Ok(Self::None),
            "sister" | "compat" | "thread" => Ok(Self::Sister),
            "inverted" | "d2" => Ok(Self::Inverted),
            other => Err(ParseError(format!(
                "compat: '{other}' is not 'none', 'sister', or 'inverted'"
            ))),
        }
    }
}

fn parse_compat_mode(value: &str) -> Result<CompatMode, ParseError> {
    CompatMode::from_str(value)
}

/// Typed configuration for `PrimeRuntime` — env-driven via
/// `conflaguration`, file-driven via the same `Settings` derive, and
/// composable into a [`PrimeRuntime::builder()`](super::os::runtime::PrimeRuntime::builder).
///
/// ```ignore
/// // env: PRIME_CORES=8 PRIME_BACKGROUND_POOL=rayon
/// use proxima::prime::PrimeConfig;
/// use conflaguration::Settings;
/// let config: PrimeConfig = PrimeConfig::from_env()?;
/// ```
///
/// Env vars are prefixed `PRIME_` by default — override at the
/// builder level if you need a different prefix.
#[derive(Debug, Clone, Settings, Validate)]
#[settings(prefix = "PRIME")]
pub struct PrimeConfig {
    /// Worker core count. `"auto"` (default) resolves to physical
    /// cores at construction; an integer string pins explicitly.
    #[setting(default_str = "auto", resolve_with = "parse_core_selection")]
    pub cores: CoreSelection,

    /// Worker placement: `"float"` (DEFAULT — unpinned, the OS schedules them),
    /// `"packed"` (pin to cores `0..N`), a bare integer `"4"` (pin to the window
    /// `4..4+N`), or a list `"4,5,6,7"` / range `"4-7"` (pin to explicit cores,
    /// which also sets the count). Float shows well on a shared box; the pinned
    /// forms are the opt-in for a dedicated box or to keep two colocated runtimes
    /// off each other's cores.
    #[setting(default_str = "float", resolve_with = "parse_affinity")]
    pub affinity: Affinity,

    /// Background-pool implementation: `"rayon"` (default), `"inline"`,
    /// or `"none"`.
    #[setting(default_str = "rayon", resolve_with = "parse_pool_kind")]
    pub background_pool: PoolKind,

    /// tokio coexistence mode — "the split". `"none"` (default, pure prime),
    /// `"sister"` (per-core sister tokio thread), or `"inverted"` (D2 in-thread,
    /// local `tokio::spawn`). std/alloc runtime control via `PRIME_COMPAT`; the
    /// no_std default is baked from `prime-runtime.toml` `[compat]` by build.rs.
    #[setting(default_str = "none", resolve_with = "parse_compat_mode")]
    pub compat: CompatMode,
}

impl Default for PrimeConfig {
    fn default() -> Self {
        Self {
            cores: CoreSelection::Auto,
            affinity: Affinity::Float,
            background_pool: PoolKind::Rayon,
            compat: CompatMode::None,
        }
    }
}

/// Fluent builder for [`PrimeRuntime`](super::os::runtime::PrimeRuntime).
/// Composes with [`PrimeConfig`]: a builder can take a `PrimeConfig` as
/// its starting point via [`Builder::from_config`] and override individual
/// fields afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Builder {
    cores: CoreSelection,
    affinity: Affinity,
    background_pool: PoolKind,
    /// tokio coexistence mode ("the split"). [`build`](Self::build) routes to
    /// `PrimeRuntime::new` / `new_with_tokio_compat` / `new_with_tokio_compat_inverted`
    /// accordingly. Set via [`compat`](Self::compat),
    /// [`tokio_compat`](Self::tokio_compat), or
    /// [`tokio_compat_inverted`](Self::tokio_compat_inverted).
    compat: CompatMode,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    /// Start with `cores = auto`, `background_pool = rayon`. Match the
    /// defaults `PrimeConfig::default()` produces.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cores: CoreSelection::Auto,
            affinity: Affinity::Float,
            background_pool: PoolKind::Rayon,
            compat: CompatMode::None,
        }
    }

    /// Use the physical core count.
    #[must_use]
    pub fn cores_auto(mut self) -> Self {
        self.cores = CoreSelection::Auto;
        self
    }

    /// Pin to an explicit worker count.
    #[must_use]
    pub fn cores(mut self, count: usize) -> Self {
        self.cores = CoreSelection::Count(count);
        self
    }

    /// Set the core selection from a typed value (config path).
    #[must_use]
    pub fn cores_selection(mut self, selection: CoreSelection) -> Self {
        self.cores = selection;
        self
    }

    /// Pin the `cores` workers starting at physical core `start` (window
    /// `start..start+N`) instead of `0..N`. The one-knob fix for colocating two
    /// prime runtimes: give the second `.affinity(first_core_count)` so they
    /// occupy disjoint cores. Worker COUNT still comes from [`cores`](Self::cores).
    #[must_use]
    pub fn affinity(mut self, start: usize) -> Self {
        self.affinity = Affinity::Offset(start);
        self
    }

    /// Pin workers to an EXPLICIT physical-core list (NUMA / hand-placed). The
    /// list length also sets the worker count, so `.affinities([4, 5, 6, 7])`
    /// is a complete placement — four workers on cores 4-7.
    #[must_use]
    pub fn affinities(mut self, cores: impl Into<Vec<usize>>) -> Self {
        let cores = cores.into();
        self.cores = CoreSelection::Count(cores.len());
        self.affinity = Affinity::Cores(cores);
        self
    }

    /// Float (the DEFAULT): don't pin — workers ride the OS scheduler and dodge
    /// contention. The explicit form, for overriding a config that pinned.
    #[must_use]
    pub fn float(mut self) -> Self {
        self.affinity = Affinity::Float;
        self
    }

    /// Pin workers to cores `0..N` (packed) — opt out of the float default for a
    /// dedicated box where cache/NUMA locality beats the ability to migrate.
    #[must_use]
    pub fn packed(mut self) -> Self {
        self.affinity = Affinity::Packed;
        self
    }

    /// Set the placement from a typed value (config path).
    #[must_use]
    pub fn affinity_selection(mut self, affinity: Affinity) -> Self {
        if let Some(count) = affinity.fixed_count() {
            self.cores = CoreSelection::Count(count);
        }
        self.affinity = affinity;
        self
    }

    /// Use Rayon as the background pool. Requires the `rayon` feature.
    #[must_use]
    pub fn background_rayon(mut self) -> Self {
        self.background_pool = PoolKind::Rayon;
        self
    }

    /// Use the inline `std::thread::spawn` fallback for background
    /// work — fine for tests, small workloads, or when the deployment
    /// does not want a long-lived pool.
    #[must_use]
    pub fn background_inline(mut self) -> Self {
        self.background_pool = PoolKind::Inline;
        self
    }

    /// Attach no explicit background pool. Semantically equivalent
    /// to `inline` today (runtime falls back to `std::thread::spawn`);
    /// flags intent for code review and future-proofs against a pool
    /// becoming non-optional.
    #[must_use]
    pub fn background_none(mut self) -> Self {
        self.background_pool = PoolKind::None;
        self
    }

    /// Choose the background-pool kind directly (config path).
    #[must_use]
    pub fn background_pool(mut self, kind: PoolKind) -> Self {
        self.background_pool = kind;
        self
    }

    /// Apply every field from a typed [`PrimeConfig`]. Subsequent
    /// builder calls still override.
    #[must_use]
    pub fn from_config(mut self, config: &PrimeConfig) -> Self {
        self.cores = config.cores;
        self.affinity = config.affinity.clone();
        self.background_pool = config.background_pool;
        self.compat = config.compat;
        self
    }

    /// Set the tokio coexistence mode directly (config path).
    #[must_use]
    pub fn compat(mut self, mode: CompatMode) -> Self {
        self.compat = mode;
        self
    }

    /// P2 — opt into prime+tokio compat mode. Each prime worker holds a
    /// `tokio::runtime::EnterGuard` into a per-core sister tokio
    /// current-thread runtime, so `tokio::spawn` / `tokio::sync::*` /
    /// `tokio::time::*` API calls from inside a prime task resolve
    /// against the sister runtime.
    ///
    /// Cost picture, ship criteria, and bench matrix live in
    /// `rust/docs/runtime-prime/discipline-prime-tokio-compat.md`. Pure
    /// prime is the recommended default; compat mode exists for
    /// consumers who can't or won't migrate their tokio imports.
    #[cfg(feature = "prime-tokio-compat")]
    #[must_use]
    pub fn tokio_compat(mut self) -> Self {
        self.compat = CompatMode::Sister;
        self
    }

    /// D2 — opt into INVERTED prime+tokio compat mode. Each prime worker
    /// owns its own sister tokio current-thread runtime and ticks the prime
    /// executor inside `sister.block_on(...)`, so raw `tokio::spawn` from a
    /// prime task takes tokio's LOCAL fast path (no per-spawn kevent). Takes
    /// precedence over [`tokio_compat`](Self::tokio_compat) at build time.
    ///
    /// minimal park; full Dekker-park fidelity tracked in
    /// discipline-inverted-compat.md.
    #[cfg(feature = "prime-tokio-compat-inverted")]
    #[must_use]
    pub fn tokio_compat_inverted(mut self) -> Self {
        self.compat = CompatMode::Inverted;
        self
    }

    /// Construct the runtime. Resolves `cores` to a concrete count
    /// and attaches the chosen background pool.
    ///
    /// # Errors
    /// Returns `ProximaError::Body` when an explicit `cores(n)` and an
    /// explicit `Cores` affinity list of a DIFFERENT length were both set —
    /// `affinities`/`affinity_selection` keep `cores` in sync when called
    /// alone, so this only fires when a later `.cores(n)` call diverges from
    /// an already-set explicit core list; silently preferring one over the
    /// other would make that later call a no-op with no diagnostic.
    pub fn build(self) -> Result<super::os::runtime::PrimeRuntime, proxima_core::ProximaError> {
        if let (CoreSelection::Count(count), Some(affinity_count)) =
            (self.cores, self.affinity.fixed_count())
            && count != affinity_count
        {
            return Err(proxima_core::ProximaError::Body(format!(
                "cores({count}) conflicts with an affinity core list of length {affinity_count}"
            )));
        }
        // resolve the affinity knob to the (placement, pin) the one composable
        // runtime constructor takes. an explicit `Cores` list fixes the count;
        // `Float` (default) yields pin = false; every pinned variant pin = true.
        let resolved_cores = self
            .affinity
            .fixed_count()
            .unwrap_or_else(|| self.cores.resolve())
            .max(1);
        let placement = self.affinity.placement(resolved_cores);
        let pin = !matches!(self.affinity, Affinity::Float);
        let runtime = match self.compat {
            CompatMode::None => {
                super::os::runtime::PrimeRuntime::new_inner_placed(placement, pin, false)?
            }
            CompatMode::Sister => {
                #[cfg(feature = "prime-tokio-compat")]
                {
                    super::os::runtime::PrimeRuntime::new_inner_placed(placement, pin, true)?
                }
                #[cfg(not(feature = "prime-tokio-compat"))]
                {
                    let _ = (placement, pin);
                    return Err(proxima_core::ProximaError::Body(
                        "compat = sister requires the 'prime-tokio-compat' feature".into(),
                    ));
                }
            }
            CompatMode::Inverted => {
                #[cfg(feature = "prime-tokio-compat-inverted")]
                {
                    super::os::runtime::PrimeRuntime::new_inverted_placed(placement, pin)?
                }
                #[cfg(not(feature = "prime-tokio-compat-inverted"))]
                {
                    let _ = (placement, pin);
                    return Err(proxima_core::ProximaError::Body(
                        "compat = inverted requires the 'prime-tokio-compat-inverted' feature"
                            .into(),
                    ));
                }
            }
        };
        match self.background_pool {
            PoolKind::Rayon => {
                #[cfg(feature = "rayon")]
                {
                    // sized to this runtime's own core budget, not rayon's
                    // machine-wide default: a `cores(1)` runtime otherwise got
                    // a background pool spanning every CPU on the box, so
                    // Offloaded blocking work never actually serialized on a
                    // single-core runtime the way `cores` implies it should.
                    let pool = std::sync::Arc::new(
                        proxima_runtime::RayonBackgroundPool::with_threads(resolved_cores)?,
                    );
                    Ok(runtime.with_background_pool(pool))
                }
                #[cfg(not(feature = "rayon"))]
                {
                    let _ = runtime;
                    Err(proxima_core::ProximaError::Body(
                        "background_pool = rayon requires the 'rayon' feature".into(),
                    ))
                }
            }
            PoolKind::Inline | PoolKind::None => Ok(runtime),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn core_selection_parses_auto() {
        assert_eq!(
            CoreSelection::from_str("auto").expect("parse"),
            CoreSelection::Auto
        );
        assert_eq!(
            CoreSelection::from_str("AUTO").expect("parse"),
            CoreSelection::Auto
        );
        assert_eq!(
            CoreSelection::from_str("").expect("parse"),
            CoreSelection::Auto
        );
    }

    #[test]
    fn core_selection_parses_count() {
        assert_eq!(
            CoreSelection::from_str("8").expect("parse"),
            CoreSelection::Count(8)
        );
        assert_eq!(
            CoreSelection::from_str(" 4 ").expect("parse"),
            CoreSelection::Count(4)
        );
    }

    #[test]
    fn core_selection_rejects_garbage() {
        let result = CoreSelection::from_str("banana");
        assert!(
            result.is_err(),
            "banana should not parse as a core selection"
        );
    }

    #[test]
    fn core_selection_resolve_count_clamps_to_one() {
        assert_eq!(CoreSelection::Count(0).resolve(), 1);
        assert_eq!(CoreSelection::Count(4).resolve(), 4);
    }

    #[test]
    fn pool_kind_parses_each_variant() {
        assert_eq!(PoolKind::from_str("rayon").expect("parse"), PoolKind::Rayon);
        assert_eq!(
            PoolKind::from_str("Inline").expect("parse"),
            PoolKind::Inline
        );
        assert_eq!(PoolKind::from_str("none").expect("parse"), PoolKind::None);
        assert_eq!(PoolKind::from_str("off").expect("parse"), PoolKind::None);
    }

    #[test]
    fn pool_kind_rejects_garbage() {
        assert!(PoolKind::from_str("monoidal").is_err());
    }

    #[test]
    fn config_default_is_auto_and_rayon() {
        let config = PrimeConfig::default();
        assert_eq!(config.cores, CoreSelection::Auto);
        assert_eq!(config.background_pool, PoolKind::Rayon);
    }

    #[test]
    fn builder_defaults_match_config_defaults() {
        let builder = Builder::new();
        assert_eq!(builder.cores, CoreSelection::Auto);
        assert_eq!(builder.background_pool, PoolKind::Rayon);
    }

    #[test]
    fn builder_cores_override() {
        let builder = Builder::new().cores(4);
        assert_eq!(builder.cores, CoreSelection::Count(4));
    }

    #[test]
    fn builder_from_config_copies_fields() {
        let config = PrimeConfig {
            cores: CoreSelection::Count(8),
            affinity: Affinity::Packed,
            background_pool: PoolKind::Inline,
            compat: CompatMode::Inverted,
        };
        let builder = Builder::new().from_config(&config);
        assert_eq!(builder.cores, CoreSelection::Count(8));
        assert_eq!(builder.background_pool, PoolKind::Inline);
        assert_eq!(builder.compat, CompatMode::Inverted);
    }

    #[test]
    fn builder_subsequent_calls_override_config() {
        let config = PrimeConfig {
            cores: CoreSelection::Count(8),
            affinity: Affinity::Packed,
            background_pool: PoolKind::Inline,
            compat: CompatMode::Sister,
        };
        let builder = Builder::new()
            .from_config(&config)
            .cores(2)
            .background_none()
            .compat(CompatMode::None);
        assert_eq!(builder.cores, CoreSelection::Count(2));
        assert_eq!(builder.background_pool, PoolKind::None);
        assert_eq!(builder.compat, CompatMode::None);
    }

    #[test]
    fn compat_mode_parses_each_variant() {
        assert_eq!(
            CompatMode::from_str("none").expect("parse"),
            CompatMode::None
        );
        assert_eq!(CompatMode::from_str("").expect("parse"), CompatMode::None);
        assert_eq!(
            CompatMode::from_str("Sister").expect("parse"),
            CompatMode::Sister
        );
        assert_eq!(
            CompatMode::from_str("inverted").expect("parse"),
            CompatMode::Inverted
        );
        assert_eq!(
            CompatMode::from_str("d2").expect("parse"),
            CompatMode::Inverted
        );
    }

    #[test]
    fn compat_mode_rejects_garbage() {
        assert!(CompatMode::from_str("hybrid").is_err());
    }

    #[test]
    fn config_default_compat_is_none() {
        assert_eq!(PrimeConfig::default().compat, CompatMode::None);
        assert_eq!(Builder::new().compat, CompatMode::None);
    }

    // gate point 12 (config + API parity): the same intent expressed via the
    // typed `PrimeConfig` (conflaguration path) and via the fluent builder must
    // produce the same builder state, so neither path is a second source of
    // truth that can drift from the other.
    #[test]
    fn fluent_and_config_paths_produce_equal_builders() {
        let config = PrimeConfig {
            cores: CoreSelection::Count(4),
            affinity: Affinity::Packed,
            background_pool: PoolKind::Inline,
            compat: CompatMode::Inverted,
        };
        let from_config = Builder::new().from_config(&config);
        let fluent = Builder::new()
            .cores(4)
            .packed()
            .background_inline()
            .compat(CompatMode::Inverted);
        assert_eq!(
            from_config, fluent,
            "config-derived and fluent builders must agree on every field"
        );
    }

    // both construction paths must also BUILD an equivalent runtime, not just
    // agree on builder state. inline pool keeps this off the rayon feature.
    #[test]
    fn fluent_and_config_paths_build_equivalent_runtimes() {
        use proxima_runtime::Runtime;
        let config = PrimeConfig {
            cores: CoreSelection::Count(2),
            affinity: Affinity::Packed,
            background_pool: PoolKind::Inline,
            compat: CompatMode::None,
        };
        let from_config = Builder::new()
            .from_config(&config)
            .build()
            .expect("config build");
        let fluent = Builder::new()
            .cores(2)
            .background_inline()
            .compat(CompatMode::None)
            .build()
            .expect("fluent build");
        assert_eq!(from_config.num_cores(), fluent.num_cores());
    }

    #[test]
    fn builder_build_constructs_runtime_with_inline_pool() {
        use proxima_runtime::Runtime;
        // inline pool path bypasses the rayon-feature dependency, so
        // this test runs under `runtime-prime-full` alone.
        let runtime = Builder::new()
            .cores(2)
            .background_inline()
            .build()
            .expect("build runtime");
        assert_eq!(runtime.num_cores(), 2);
    }

    #[test]
    fn affinity_default_is_float() {
        assert_eq!(Affinity::default(), Affinity::Float);
        assert_eq!(PrimeConfig::default().affinity, Affinity::Float);
        assert_eq!(Builder::new().affinity, Affinity::Float);
    }

    #[test]
    fn affinity_parses_each_form() {
        assert_eq!(Affinity::from_str("").expect("parse"), Affinity::Float);
        assert_eq!(Affinity::from_str("float").expect("parse"), Affinity::Float);
        assert_eq!(
            Affinity::from_str("unpinned").expect("parse"),
            Affinity::Float
        );
        assert_eq!(
            Affinity::from_str("packed").expect("parse"),
            Affinity::Packed
        );
        assert_eq!(Affinity::from_str("4").expect("parse"), Affinity::Offset(4));
        assert_eq!(
            Affinity::from_str("4,5,6,7").expect("parse"),
            Affinity::Cores(vec![4, 5, 6, 7])
        );
        assert_eq!(
            Affinity::from_str("4-7").expect("parse"),
            Affinity::Cores(vec![4, 5, 6, 7])
        );
    }

    #[test]
    fn affinity_rejects_garbage_and_bad_range() {
        assert!(Affinity::from_str("banana").is_err());
        assert!(
            Affinity::from_str("7-4").is_err(),
            "descending range rejected"
        );
        assert!(Affinity::from_str("4,x,6").is_err());
    }

    #[test]
    fn affinity_placement_shifts_the_window() {
        assert_eq!(Affinity::Packed.placement(4), vec![0, 1, 2, 3]);
        assert_eq!(Affinity::Offset(4).placement(4), vec![4, 5, 6, 7]);
        assert_eq!(Affinity::Cores(vec![2, 5]).placement(2), vec![2, 5]);
    }

    #[test]
    fn affinities_fluent_sets_count_and_placement() {
        let builder = Builder::new().affinities([4, 5, 6, 7]);
        assert_eq!(builder.cores, CoreSelection::Count(4));
        assert_eq!(builder.affinity, Affinity::Cores(vec![4, 5, 6, 7]));
    }

    // P4 parity for the affinity dimension: typed config and fluent agree.
    #[test]
    fn affinity_config_and_fluent_paths_agree() {
        let config = PrimeConfig {
            cores: CoreSelection::Count(4),
            affinity: Affinity::Offset(4),
            background_pool: PoolKind::Inline,
            compat: CompatMode::None,
        };
        let from_config = Builder::new().from_config(&config);
        let fluent = Builder::new().cores(4).affinity(4).background_inline();
        assert_eq!(from_config, fluent, "config and fluent affinity must agree");
    }

    #[test]
    fn offset_placement_builds_runtime_with_the_right_core_count() {
        use proxima_runtime::Runtime;
        // offset placement pins (best-effort) via the one composable constructor;
        // so even if cores 6-7 are out of range the workers launch unpinned.
        let runtime = Builder::new()
            .cores(2)
            .affinity(6)
            .background_inline()
            .build()
            .expect("build offset runtime");
        assert_eq!(runtime.num_cores(), 2);
    }

    // a later `.cores(n)` that diverges from an already-set explicit `Cores`
    // list used to be silently discarded at build time (`fixed_count()`
    // always won) — the caller's last setter call had no effect and no
    // diagnostic said so.
    #[test]
    fn cores_after_affinities_with_different_length_is_a_build_error() {
        let result = Builder::new()
            .affinities([4, 5, 6])
            .cores(4)
            .background_inline()
            .build();
        assert!(
            result.is_err(),
            "cores(4) conflicting with a 3-entry affinity list must error, not silently lose"
        );
    }

    // the ordinary case `affinities` exists for — no explicit `.cores()` call
    // — must keep working: the setter syncs `cores` to the list length itself.
    #[test]
    fn affinities_alone_builds_without_conflict() {
        use proxima_runtime::Runtime;
        let runtime = Builder::new()
            .affinities([4, 5])
            .background_inline()
            .build()
            .expect("affinities alone must not conflict with itself");
        assert_eq!(runtime.num_cores(), 2);
    }
}
