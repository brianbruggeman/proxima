//! `BlacklistTable` — the accept-edge DoS-blacklist a `DenySignature`
//! (`crate::any::deny`) and an unclassifiable-reject hook record strikes
//! against, and `proxima_http::any_listener::AnyListenProtocol` consults
//! BEFORE `ListenerCore::admit` on every accepted connection.
//!
//! Built the identical way [`crate::any::AnyRegistry`] is (`ArcSwap` +
//! copy-on-write mutate, load_full -> clone -> mutate one entry ->
//! compare_and_swap -> retry) — the same lock-free CoW discipline, just
//! keyed by peer address instead of protocol name.
//!
//! Two independent per-peer counters, not one blended weight: a `Strike::Deny`
//! (a connection matched a registered `DenySignature` literal — deliberate
//! malicious traffic) bans in ONE strike by default; a `Strike::Unclassifiable`
//! (the classifier rejected the connection outright, or the prefix bound was
//! exceeded — noisy but not necessarily hostile scanning/misconfiguration)
//! needs many more before it bans. Blending them into a single score would
//! let a burst of noise or a single confirmed-malicious hit fight over the
//! same threshold; keeping them apart means a deny match always bans fast
//! regardless of how much unrelated noise a peer has already accumulated,
//! and vice versa.
//!
//! `now` is an explicit parameter on every method — this module never reads
//! a wall clock itself, so a test drives the ban/expiry lifecycle
//! deterministically with `proxima_core::time::Instant::from_monotonic`, no
//! sleeps.

use std::net::IpAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use hashbrown::HashMap;
use proxima_core::time::Instant;
use serde::{Deserialize, Serialize};

fn default_deny_strike_threshold() -> u32 {
    super::sized::ADMISSION_BLACKLIST_DENY_STRIKE_THRESHOLD_DEFAULT
}

fn default_unclassifiable_strike_threshold() -> u32 {
    super::sized::ADMISSION_BLACKLIST_UNCLASSIFIABLE_STRIKE_THRESHOLD_DEFAULT
}

fn default_strike_window_ms() -> u64 {
    super::sized::ADMISSION_BLACKLIST_STRIKE_WINDOW_MS_DEFAULT
}

fn default_ban_duration_ms() -> u64 {
    super::sized::ADMISSION_BLACKLIST_BAN_DURATION_MS_DEFAULT
}

/// Runtime configuration for [`BlacklistTable`]. Mirrors
/// `crate::config::ListenTuningConfig` verbatim: `#[derive(Builder,
/// Deserialize, Serialize, Settings)]` + [`Validate`], a
/// [`BlacklistLayerBuilder`] with call-order precedence, and defaults
/// sourced from the `sized` floor `build.rs` generates from
/// `proxima-listen-core.toml`'s `[admission.blacklist]` section — there is
/// no double source of truth.
/// Config is first-class in two equivalent forms — the fluent `.builder()`
/// and a TOML file loaded through [`Self::layered`] — and they produce the
/// exact same value:
///
/// ```
/// use std::io::Write;
///
/// use proxima_listen::BlacklistConfig;
///
/// let via_builder = BlacklistConfig::builder()
///     .deny_strike_threshold(2)
///     .ban_duration_ms(600_000)
///     .build();
///
/// let mut file = tempfile::Builder::new().suffix(".toml").tempfile().expect("tempfile");
/// write!(file, "deny_strike_threshold = 2\nban_duration_ms = 600000\n").expect("write toml");
///
/// let via_toml = BlacklistConfig::layered()
///     .from_path(file.path())
///     .expect("load from toml")
///     .build();
///
/// assert_eq!(via_builder, via_toml);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_LISTEN_ADMISSION_BLACKLIST")]
#[builder(derive(Clone, Debug))]
pub struct BlacklistConfig {
    /// Strikes at [`Strike::Deny`] before a peer is banned — deliberately
    /// low (defaults to 1): a connection matching a registered
    /// `DenySignature` literal is not ambiguous noise, it is a positively
    /// identified bad actor.
    #[setting(default = 1)]
    #[serde(default = "default_deny_strike_threshold")]
    #[builder(default = default_deny_strike_threshold())]
    pub deny_strike_threshold: u32,

    /// Strikes at [`Strike::Unclassifiable`] before a peer is banned —
    /// deliberately higher (defaults to 20): a rejected/unresolved
    /// classification is common under normal noise (health-checkers,
    /// misconfigured clients) and should not ban on the first occurrence.
    #[setting(default = 20)]
    #[serde(default = "default_unclassifiable_strike_threshold")]
    #[builder(default = default_unclassifiable_strike_threshold())]
    pub unclassifiable_strike_threshold: u32,

    /// Window (ms) both counters share: a peer's `deny_count` and
    /// `unclassifiable_count` reset TOGETHER once this much time has
    /// elapsed since the window started, so an old, unrelated strike never
    /// contributes to a ban long after the fact.
    #[setting(default = 60_000)]
    #[serde(default = "default_strike_window_ms")]
    #[builder(default = default_strike_window_ms())]
    pub strike_window_ms: u64,

    /// How long a ban lasts (ms) once either threshold trips.
    #[setting(default = 300_000)]
    #[serde(default = "default_ban_duration_ms")]
    #[builder(default = default_ban_duration_ms())]
    pub ban_duration_ms: u64,
}

impl Default for BlacklistConfig {
    fn default() -> Self {
        BlacklistConfig::builder().build()
    }
}

impl Validate for BlacklistConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.deny_strike_threshold == 0 {
            errors.push(ValidationMessage::new(
                "deny_strike_threshold",
                "must be > 0",
            ));
        }
        if self.unclassifiable_strike_threshold == 0 {
            errors.push(ValidationMessage::new(
                "unclassifiable_strike_threshold",
                "must be > 0",
            ));
        }
        if self.strike_window_ms == 0 {
            errors.push(ValidationMessage::new("strike_window_ms", "must be > 0"));
        }
        if self.ban_duration_ms == 0 {
            errors.push(ValidationMessage::new("ban_duration_ms", "must be > 0"));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl BlacklistConfig {
    /// Start a layered builder from the `sized`-seeded defaults.
    #[must_use]
    pub fn layered() -> BlacklistLayerBuilder {
        BlacklistLayerBuilder {
            inner: BlacklistConfig::default(),
            deny_strike_threshold_set: false,
            unclassifiable_strike_threshold_set: false,
            strike_window_ms_set: false,
            ban_duration_ms_set: false,
        }
    }
}

/// Partial view of [`BlacklistConfig`] used by `.from_path`/`.underlay_path` —
/// only fields actually present in the file are applied.
#[derive(Debug, Default, Deserialize)]
struct BlacklistConfigPartial {
    deny_strike_threshold: Option<u32>,
    unclassifiable_strike_threshold: Option<u32>,
    strike_window_ms: Option<u64>,
    ban_duration_ms: Option<u64>,
}

/// Fluent builder for [`BlacklistConfig`] — VERBATIM the shape of
/// `crate::config::ListenTuningLayerBuilder`: every source contributes only
/// the fields it actually specifies; `.from_path`/`.from_env` override
/// (last writer wins per field), `.underlay_path`/`.underlay_env` fill only
/// fields still unset, `.with_*` always overrides at its call position.
pub struct BlacklistLayerBuilder {
    inner: BlacklistConfig,
    deny_strike_threshold_set: bool,
    unclassifiable_strike_threshold_set: bool,
    strike_window_ms_set: bool,
    ban_duration_ms_set: bool,
}

impl BlacklistLayerBuilder {
    /// Merge a TOML/JSON file's fields onto the accumulated config; the file
    /// wins for every field it specifies.
    pub fn from_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: BlacklistConfigPartial = conflaguration::from_file(path.as_ref())?;
        if let Some(value) = partial.deny_strike_threshold {
            self.inner.deny_strike_threshold = value;
            self.deny_strike_threshold_set = true;
        }
        if let Some(value) = partial.unclassifiable_strike_threshold {
            self.inner.unclassifiable_strike_threshold = value;
            self.unclassifiable_strike_threshold_set = true;
        }
        if let Some(value) = partial.strike_window_ms {
            self.inner.strike_window_ms = value;
            self.strike_window_ms_set = true;
        }
        if let Some(value) = partial.ban_duration_ms {
            self.inner.ban_duration_ms = value;
            self.ban_duration_ms_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from a TOML/JSON file; already-set
    /// fields are left untouched.
    pub fn underlay_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let partial: BlacklistConfigPartial = conflaguration::from_file(path.as_ref())?;
        if !self.deny_strike_threshold_set
            && let Some(value) = partial.deny_strike_threshold
        {
            self.inner.deny_strike_threshold = value;
            self.deny_strike_threshold_set = true;
        }
        if !self.unclassifiable_strike_threshold_set
            && let Some(value) = partial.unclassifiable_strike_threshold
        {
            self.inner.unclassifiable_strike_threshold = value;
            self.unclassifiable_strike_threshold_set = true;
        }
        if !self.strike_window_ms_set
            && let Some(value) = partial.strike_window_ms
        {
            self.inner.strike_window_ms = value;
            self.strike_window_ms_set = true;
        }
        if !self.ban_duration_ms_set
            && let Some(value) = partial.ban_duration_ms
        {
            self.inner.ban_duration_ms = value;
            self.ban_duration_ms_set = true;
        }
        Ok(self)
    }

    /// Merge env-set fields onto the accumulated config; env wins for every
    /// field it sets.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = BlacklistConfig::from_env()?;
        if env_is_set("PROXIMA_LISTEN_ADMISSION_BLACKLIST_DENY_STRIKE_THRESHOLD") {
            self.inner.deny_strike_threshold = resolved.deny_strike_threshold;
            self.deny_strike_threshold_set = true;
        }
        if env_is_set("PROXIMA_LISTEN_ADMISSION_BLACKLIST_UNCLASSIFIABLE_STRIKE_THRESHOLD") {
            self.inner.unclassifiable_strike_threshold = resolved.unclassifiable_strike_threshold;
            self.unclassifiable_strike_threshold_set = true;
        }
        if env_is_set("PROXIMA_LISTEN_ADMISSION_BLACKLIST_STRIKE_WINDOW_MS") {
            self.inner.strike_window_ms = resolved.strike_window_ms;
            self.strike_window_ms_set = true;
        }
        if env_is_set("PROXIMA_LISTEN_ADMISSION_BLACKLIST_BAN_DURATION_MS") {
            self.inner.ban_duration_ms = resolved.ban_duration_ms;
            self.ban_duration_ms_set = true;
        }
        Ok(self)
    }

    /// Fill any still-unset fields from env vars; already-set fields are
    /// left untouched even if the matching env var is set.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let resolved = BlacklistConfig::from_env()?;
        if !self.deny_strike_threshold_set
            && env_is_set("PROXIMA_LISTEN_ADMISSION_BLACKLIST_DENY_STRIKE_THRESHOLD")
        {
            self.inner.deny_strike_threshold = resolved.deny_strike_threshold;
            self.deny_strike_threshold_set = true;
        }
        if !self.unclassifiable_strike_threshold_set
            && env_is_set("PROXIMA_LISTEN_ADMISSION_BLACKLIST_UNCLASSIFIABLE_STRIKE_THRESHOLD")
        {
            self.inner.unclassifiable_strike_threshold = resolved.unclassifiable_strike_threshold;
            self.unclassifiable_strike_threshold_set = true;
        }
        if !self.strike_window_ms_set
            && env_is_set("PROXIMA_LISTEN_ADMISSION_BLACKLIST_STRIKE_WINDOW_MS")
        {
            self.inner.strike_window_ms = resolved.strike_window_ms;
            self.strike_window_ms_set = true;
        }
        if !self.ban_duration_ms_set
            && env_is_set("PROXIMA_LISTEN_ADMISSION_BLACKLIST_BAN_DURATION_MS")
        {
            self.inner.ban_duration_ms = resolved.ban_duration_ms;
            self.ban_duration_ms_set = true;
        }
        Ok(self)
    }

    /// Set the deny-strike ban threshold.
    #[must_use]
    pub fn with_deny_strike_threshold(mut self, value: u32) -> Self {
        self.inner.deny_strike_threshold = value;
        self.deny_strike_threshold_set = true;
        self
    }

    /// Set the unclassifiable-strike ban threshold.
    #[must_use]
    pub fn with_unclassifiable_strike_threshold(mut self, value: u32) -> Self {
        self.inner.unclassifiable_strike_threshold = value;
        self.unclassifiable_strike_threshold_set = true;
        self
    }

    /// Set the shared strike window (ms).
    #[must_use]
    pub fn with_strike_window_ms(mut self, value: u64) -> Self {
        self.inner.strike_window_ms = value;
        self.strike_window_ms_set = true;
        self
    }

    /// Set the ban duration (ms).
    #[must_use]
    pub fn with_ban_duration_ms(mut self, value: u64) -> Self {
        self.inner.ban_duration_ms = value;
        self.ban_duration_ms_set = true;
        self
    }

    /// The built immutable config.
    #[must_use]
    pub fn build(self) -> BlacklistConfig {
        self.inner
    }
}

fn env_is_set(name: &str) -> bool {
    std::env::var(name).is_ok()
}

/// Which counter a [`BlacklistTable::record_strike`] call increments — see
/// this module's doc for why the two stay independent rather than blending
/// into one score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strike {
    /// A connection matched a registered `DenySignature` literal.
    Deny,
    /// The classifier rejected the connection (no candidate matched, or the
    /// prefix bound was exceeded) before any candidate resolved.
    Unclassifiable,
}

/// Per-peer strike state. `Copy` — cheap to read out of the map, mutate, and
/// write back inside [`BlacklistTable::record_strike`]'s CoW loop.
#[derive(Debug, Clone, Copy)]
struct StrikeRecord {
    deny_count: u32,
    unclassifiable_count: u32,
    window_start: Instant,
    banned_until: Option<Instant>,
}

impl StrikeRecord {
    fn fresh(now: Instant) -> Self {
        Self {
            deny_count: 0,
            unclassifiable_count: 0,
            window_start: now,
            banned_until: None,
        }
    }

    /// Reset both counters together if the shared window has elapsed, bump
    /// the counter `kind` names, then ban if either threshold is now
    /// reached. A ban already in effect is simply extended/re-evaluated —
    /// this method never shortens a live ban.
    fn apply_strike(&mut self, now: Instant, kind: Strike, config: &BlacklistConfig) {
        let window = Duration::from_millis(config.strike_window_ms);
        if now.duration_since(self.window_start) >= window {
            self.deny_count = 0;
            self.unclassifiable_count = 0;
            self.window_start = now;
        }
        match kind {
            Strike::Deny => self.deny_count = self.deny_count.saturating_add(1),
            Strike::Unclassifiable => {
                self.unclassifiable_count = self.unclassifiable_count.saturating_add(1);
            }
        }
        let tripped = self.deny_count >= config.deny_strike_threshold
            || self.unclassifiable_count >= config.unclassifiable_strike_threshold;
        if tripped {
            self.banned_until = Some(now + Duration::from_millis(config.ban_duration_ms));
        }
    }
}

/// Accept-edge DoS-blacklist: cheap to `Clone` (an `Arc` bump + a small
/// `Copy`-shaped config), shared across every accepted connection the same
/// way [`crate::admission::ConnAdmission`] is. See this module's doc for the
/// two-counter policy and the CoW discipline `record_strike` uses.
#[derive(Clone)]
pub struct BlacklistTable {
    table: Arc<ArcSwap<HashMap<IpAddr, StrikeRecord>>>,
    config: BlacklistConfig,
}

impl BlacklistTable {
    #[must_use]
    pub fn new(config: BlacklistConfig) -> Self {
        Self {
            table: Arc::new(ArcSwap::from_pointee(HashMap::new())),
            config,
        }
    }

    /// `true` if `peer` is currently within a live ban window. Read-only —
    /// never mutates, never allocates a fresh record for a peer it has
    /// never seen.
    #[must_use]
    pub fn is_banned(&self, peer: IpAddr, now: Instant) -> bool {
        self.table
            .load()
            .get(&peer)
            .and_then(|record| record.banned_until)
            .is_some_and(|until| now < until)
    }

    /// Record one strike of `kind` against `peer`, evaluated at `now`.
    /// Copy-on-write, mirroring [`crate::any::AnyRegistry::register`]'s
    /// loop exactly: load the current snapshot, clone it, mutate this ONE
    /// peer's entry, compare-and-swap, retry on a concurrent writer.
    pub fn record_strike(&self, peer: IpAddr, now: Instant, kind: Strike) {
        loop {
            let current = self.table.load_full();
            let mut next: HashMap<IpAddr, StrikeRecord> = (*current).clone();
            let record = next.entry(peer).or_insert_with(|| StrikeRecord::fresh(now));
            record.apply_strike(now, kind, &self.config);
            let next = Arc::new(next);
            let prev = self.table.compare_and_swap(&current, next);
            if Arc::ptr_eq(&prev, &current) {
                return;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const PEER: IpAddr = IpAddr::V4(core::net::Ipv4Addr::new(127, 0, 0, 1));
    const OTHER_PEER: IpAddr = IpAddr::V4(core::net::Ipv4Addr::new(127, 0, 0, 2));

    fn instant_at(millis: u64) -> Instant {
        Instant::from_monotonic(Duration::from_millis(millis))
    }

    // the fluent builder and the conflaguration surface agree on defaults —
    // both seeded by the `sized` floor (build.rs), the single source.
    #[test]
    fn defaults_track_the_sized_floor() {
        let config = BlacklistConfig::default();
        assert_eq!(
            config.deny_strike_threshold,
            super::super::sized::ADMISSION_BLACKLIST_DENY_STRIKE_THRESHOLD_DEFAULT
        );
        assert_eq!(
            config.unclassifiable_strike_threshold,
            super::super::sized::ADMISSION_BLACKLIST_UNCLASSIFIABLE_STRIKE_THRESHOLD_DEFAULT
        );
        assert_eq!(
            config.strike_window_ms,
            super::super::sized::ADMISSION_BLACKLIST_STRIKE_WINDOW_MS_DEFAULT
        );
        assert_eq!(
            config.ban_duration_ms,
            super::super::sized::ADMISSION_BLACKLIST_BAN_DURATION_MS_DEFAULT
        );
        temp_env::with_vars::<&str, &str, _, _>([], || {
            let from_env = BlacklistConfig::from_env().expect("from_env");
            assert_eq!(from_env, config);
        });
    }

    #[test]
    fn default_config_validates() {
        assert!(BlacklistConfig::default().validate().is_ok());
    }

    #[test]
    fn zero_any_field_rejected() {
        let error = BlacklistConfig::builder()
            .deny_strike_threshold(0)
            .build()
            .validate()
            .expect_err("must reject 0");
        assert!(format!("{error:?}").contains("deny_strike_threshold"));
    }

    #[test]
    fn builder_starts_at_default() {
        assert_eq!(
            BlacklistConfig::layered().build(),
            BlacklistConfig::default()
        );
    }

    #[test]
    fn with_overrides_default() {
        let config = BlacklistConfig::layered()
            .with_deny_strike_threshold(5)
            .build();
        assert_eq!(config.deny_strike_threshold, 5);
        assert_eq!(BlacklistConfig::default().deny_strike_threshold, 1);
    }

    #[test]
    fn from_path_overrides_default() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("blacklist.toml");
        std::fs::write(&path, "deny_strike_threshold = 3\n").expect("write toml");
        let config = BlacklistConfig::layered()
            .from_path(&path)
            .expect("from_path")
            .build();
        assert_eq!(config.deny_strike_threshold, 3);
        assert_eq!(
            config.unclassifiable_strike_threshold, 20,
            "untouched field"
        );
    }

    #[test]
    fn env_override_demonstration() {
        temp_env::with_vars(
            [(
                "PROXIMA_LISTEN_ADMISSION_BLACKLIST_DENY_STRIKE_THRESHOLD",
                Some("9"),
            )],
            || {
                let config = BlacklistConfig::from_env().expect("from_env");
                assert_eq!(config.deny_strike_threshold, 9);
            },
        );
    }

    #[test]
    fn underlay_path_fills_only_unset_fields() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("blacklist.toml");
        std::fs::write(
            &path,
            "deny_strike_threshold = 1\nunclassifiable_strike_threshold = 7\n",
        )
        .expect("write toml");
        let config = BlacklistConfig::layered()
            .with_deny_strike_threshold(64)
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(
            config.deny_strike_threshold, 64,
            "already set; file value dropped"
        );
        assert_eq!(
            config.unclassifiable_strike_threshold, 7,
            "unset before underlay; filled"
        );
    }

    #[test]
    fn config_round_trips_through_serde() {
        let built = BlacklistConfig::layered()
            .with_deny_strike_threshold(2)
            .with_ban_duration_ms(1_000)
            .build();
        let json = serde_json::to_string(&built).expect("serialize");
        let from_json: BlacklistConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(from_json, built);
    }

    // --- BlacklistTable behavior: deterministic, no sleeps ---

    #[test]
    fn deny_bans_on_a_single_strike() {
        let table = BlacklistTable::new(BlacklistConfig::default());
        let now = instant_at(0);
        assert!(!table.is_banned(PEER, now));
        table.record_strike(PEER, now, Strike::Deny);
        assert!(
            table.is_banned(PEER, now),
            "one deny strike must ban at the default threshold"
        );
    }

    #[test]
    fn unclassifiable_needs_the_full_threshold_before_banning() {
        let table = BlacklistTable::new(BlacklistConfig::default());
        let now = instant_at(0);
        for _ in 0..19 {
            table.record_strike(PEER, now, Strike::Unclassifiable);
        }
        assert!(
            !table.is_banned(PEER, now),
            "19 strikes must not ban against the default threshold of 20"
        );
        table.record_strike(PEER, now, Strike::Unclassifiable);
        assert!(table.is_banned(PEER, now), "the 20th strike must ban");
    }

    #[test]
    fn ban_expires_after_ban_duration_ms() {
        let config = BlacklistConfig::layered()
            .with_ban_duration_ms(10_000)
            .build();
        let table = BlacklistTable::new(config);
        let strike_time = instant_at(0);
        table.record_strike(PEER, strike_time, Strike::Deny);
        assert!(table.is_banned(PEER, strike_time + Duration::from_millis(9_999)));
        assert!(!table.is_banned(PEER, strike_time + Duration::from_millis(10_000)));
    }

    #[test]
    fn strike_window_resets_both_counters_together() {
        let config = BlacklistConfig::layered()
            .with_unclassifiable_strike_threshold(5)
            .with_strike_window_ms(1_000)
            .build();
        let table = BlacklistTable::new(config);
        for _ in 0..4 {
            table.record_strike(PEER, instant_at(0), Strike::Unclassifiable);
        }
        assert!(!table.is_banned(PEER, instant_at(0)));
        // past the window: the 4 earlier strikes must not carry over.
        table.record_strike(PEER, instant_at(2_000), Strike::Unclassifiable);
        assert!(
            !table.is_banned(PEER, instant_at(2_000)),
            "the window reset must drop the earlier 4 strikes, leaving only 1"
        );
    }

    #[test]
    fn ban_is_independent_per_peer() {
        let table = BlacklistTable::new(BlacklistConfig::default());
        let now = instant_at(0);
        table.record_strike(PEER, now, Strike::Deny);
        assert!(table.is_banned(PEER, now));
        assert!(
            !table.is_banned(OTHER_PEER, now),
            "a different peer must be unaffected"
        );
    }

    // Proves the CoW `record_strike` loop never loses a strike under real
    // concurrent writers — 8 threads x 50 strikes each must land exactly
    // 400 strikes, no more, no fewer: set the threshold to the exact total
    // so a single dropped write (a retry that silently discarded a
    // concurrent mutation) would leave the peer un-banned.
    #[test]
    fn concurrent_record_strike_never_loses_a_strike() {
        const THREADS: usize = 8;
        const STRIKES_PER_THREAD: usize = 50;
        const TOTAL: u32 = (THREADS * STRIKES_PER_THREAD) as u32;

        let config = BlacklistConfig::layered()
            .with_unclassifiable_strike_threshold(TOTAL)
            .build();
        let table = BlacklistTable::new(config);
        let now = instant_at(0);

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let table = table.clone();
                std::thread::spawn(move || {
                    for _ in 0..STRIKES_PER_THREAD {
                        table.record_strike(PEER, now, Strike::Unclassifiable);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("thread join");
        }

        assert!(
            table.is_banned(PEER, now),
            "exactly {TOTAL} strikes must have landed to trip the threshold; a lost \
             write under the CAS race would leave the peer un-banned"
        );
    }

    #[test]
    fn concurrent_record_strike_below_threshold_proves_the_count_is_exact_not_at_least() {
        const THREADS: usize = 8;
        const STRIKES_PER_THREAD: usize = 50;
        const TOTAL: u32 = (THREADS * STRIKES_PER_THREAD) as u32;

        let config = BlacklistConfig::layered()
            .with_unclassifiable_strike_threshold(TOTAL + 1)
            .build();
        let table = BlacklistTable::new(config);
        let now = instant_at(0);

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let table = table.clone();
                std::thread::spawn(move || {
                    for _ in 0..STRIKES_PER_THREAD {
                        table.record_strike(PEER, now, Strike::Unclassifiable);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("thread join");
        }

        assert!(
            !table.is_banned(PEER, now),
            "exactly {TOTAL} strikes landed against a threshold of {}; a duplicated \
             write under the CAS race would over-count and ban early",
            TOTAL + 1
        );
    }
}
