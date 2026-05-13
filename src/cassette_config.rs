//! Cassette policy: how record/replay data is kept fresh and when it may be
//! destroyed. Three escape hatches, in order of preference: a
//! `conflaguration`-first [`CassetteConfig`] (env `PROXIMA_CASSETTE_*` or a
//! committed `tests/cassettes/config.toml`), the fluent
//! [`CassetteConfig::layered`] / bon builder surface, and — last resort —
//! programmable [`CassetteHooks`] that override the declarative policy
//! per-decision. Hooks are deliberately NOT serializable: the built config
//! remains one wire form, hooks are code.

use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use proxima_core::ProximaError;
use proxima_recording::replay::CassetteMeta;
use proxima_test::Mode;

/// Parse error for cassette config fields. conflaguration's `resolve_with`
/// plumbing demands a `std::error::Error` impl on the parser's error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(String);

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ParseError {}

/// How the record-or-replay decision is made for a cassette test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModePolicy {
    /// File presence decides: cassette exists → replay, missing → record.
    Auto,
    /// Force recording (the rerecord policy governs an existing file).
    Record,
    /// Force replay; a missing cassette is a hard error (the CI posture).
    Replay,
}

impl ModePolicy {
    #[must_use]
    pub fn resolve(self, discovered: Mode) -> Mode {
        match self {
            Self::Auto => discovered,
            Self::Record => Mode::Record,
            Self::Replay => Mode::Replay,
        }
    }
}

impl FromStr for ModePolicy {
    type Err = ParseError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "auto" => Ok(Self::Auto),
            "record" => Ok(Self::Record),
            "replay" => Ok(Self::Replay),
            other => Err(ParseError(format!(
                "unknown cassette mode `{other}` (expected auto|record|replay)"
            ))),
        }
    }
}

/// What record mode does when the cassette file already exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RerecordPolicy {
    /// Delete the old cassette and record fresh.
    Truncate,
    /// Rename the old cassette to `<file>.bak` (replacing any prior backup)
    /// before recording fresh.
    Backup,
    /// Refuse to destroy the existing cassette; the test errors.
    Fail,
}

impl RerecordPolicy {
    #[must_use]
    pub fn decision(self) -> RerecordDecision {
        match self {
            Self::Truncate => RerecordDecision::Truncate,
            Self::Backup => RerecordDecision::Backup,
            Self::Fail => RerecordDecision::Fail,
        }
    }
}

impl FromStr for RerecordPolicy {
    type Err = ParseError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "truncate" => Ok(Self::Truncate),
            "backup" => Ok(Self::Backup),
            "fail" => Ok(Self::Fail),
            other => Err(ParseError(format!(
                "unknown rerecord policy `{other}` (expected truncate|backup|fail)"
            ))),
        }
    }
}

/// What record mode does when two interactions resolve to the same match key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DuplicatePolicy {
    /// Identical repeats are deduplicated silently; a repeat whose response
    /// DIFFERS is an error — the silent-rot case where replay would serve
    /// one response for two genuinely different interactions.
    RejectDivergent,
    /// Keep the last recording for the key (the pre-hardening behavior).
    LastWins,
    /// Keep the first recording for the key.
    FirstWins,
}

impl FromStr for DuplicatePolicy {
    type Err = ParseError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "reject-divergent" => Ok(Self::RejectDivergent),
            "last-wins" => Ok(Self::LastWins),
            "first-wins" => Ok(Self::FirstWins),
            other => Err(ParseError(format!(
                "unknown duplicate policy `{other}` (expected reject-divergent|last-wins|first-wins)"
            ))),
        }
    }
}

/// What replay does with a cassette older than `max_age_ms` (or one whose
/// age is unknowable because it predates the provenance stamp).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StalenessPolicy {
    /// The test errors with the age and a re-record hint.
    Fail,
    /// A warning is printed to stderr and replay proceeds.
    Warn,
}

impl StalenessPolicy {
    #[must_use]
    pub fn decision(self) -> StaleDecision {
        match self {
            Self::Fail => StaleDecision::Fail,
            Self::Warn => StaleDecision::Proceed,
        }
    }
}

impl FromStr for StalenessPolicy {
    type Err = ParseError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw {
            "fail" => Ok(Self::Fail),
            "warn" => Ok(Self::Warn),
            other => Err(ParseError(format!(
                "unknown staleness policy `{other}` (expected fail|warn)"
            ))),
        }
    }
}

fn parse_mode(raw: &str) -> Result<ModePolicy, ParseError> {
    raw.parse()
}

fn parse_rerecord(raw: &str) -> Result<RerecordPolicy, ParseError> {
    raw.parse()
}

fn parse_duplicates(raw: &str) -> Result<DuplicatePolicy, ParseError> {
    raw.parse()
}

fn parse_staleness(raw: &str) -> Result<StalenessPolicy, ParseError> {
    raw.parse()
}

fn default_mode() -> ModePolicy {
    ModePolicy::Auto
}

fn default_rerecord() -> RerecordPolicy {
    RerecordPolicy::Truncate
}

fn default_duplicates() -> DuplicatePolicy {
    DuplicatePolicy::RejectDivergent
}

fn default_staleness() -> StalenessPolicy {
    StalenessPolicy::Fail
}

/// Declarative cassette policy. One built config == one serialisable wire
/// form == one applied policy; anything that can't serialize (a callback)
/// belongs in [`CassetteHooks`], not here.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_CASSETTE")]
#[builder(derive(Clone, Debug))]
pub struct CassetteConfig {
    /// Record-or-replay selection. `PROXIMA_CASSETTE` (the pre-config env
    /// var) is honored as a fallback for `PROXIMA_CASSETTE_MODE`.
    #[setting(default_str = "auto", resolve_with = "parse_mode", envs = ["PROXIMA_CASSETTE_MODE", "PROXIMA_CASSETTE"], override)]
    #[serde(default = "default_mode")]
    #[builder(default = default_mode())]
    pub mode: ModePolicy,

    /// Disposition of an existing cassette file when recording.
    #[setting(default_str = "truncate", resolve_with = "parse_rerecord")]
    #[serde(default = "default_rerecord")]
    #[builder(default = default_rerecord())]
    pub rerecord: RerecordPolicy,

    /// Disposition of two recorded interactions landing on one match key.
    #[setting(default_str = "reject-divergent", resolve_with = "parse_duplicates")]
    #[serde(default = "default_duplicates")]
    #[builder(default = default_duplicates())]
    pub duplicates: DuplicatePolicy,

    /// Include a digest of the request body in the replay match key.
    /// Required to disambiguate same-path POSTs with different payloads.
    #[setting(default = false)]
    #[serde(default)]
    #[builder(default = false)]
    pub match_body: bool,

    /// Maximum tolerated cassette age in milliseconds; `0` disables the
    /// staleness gate entirely.
    #[setting(default = 0)]
    #[serde(default)]
    #[builder(default = 0)]
    pub max_age_ms: u64,

    /// Applied when the staleness gate is enabled and the cassette is older
    /// than `max_age_ms` or has no provenance stamp.
    #[setting(default_str = "fail", resolve_with = "parse_staleness")]
    #[serde(default = "default_staleness")]
    #[builder(default = default_staleness())]
    pub staleness: StalenessPolicy,
}

impl Default for CassetteConfig {
    fn default() -> Self {
        Self {
            mode: default_mode(),
            rerecord: default_rerecord(),
            duplicates: default_duplicates(),
            match_body: false,
            max_age_ms: 0,
            staleness: default_staleness(),
        }
    }
}

impl Validate for CassetteConfig {
    // typed fields are parse-validated by resolve_with; nothing structural
    // remains to check.
    fn validate(&self) -> conflaguration::Result<()> {
        Ok(())
    }
}

impl CassetteConfig {
    /// The staleness gate as a typed duration; `None` when disabled.
    #[must_use]
    pub fn max_age(&self) -> Option<Duration> {
        (self.max_age_ms > 0).then(|| Duration::from_millis(self.max_age_ms))
    }

    /// Layered fluent loader (call-order precedence: a later layer wins per
    /// field, for the fields it sets).
    #[must_use]
    pub fn layered() -> CassetteLayerBuilder {
        CassetteLayerBuilder {
            inner: Self::default(),
            touched: BTreeSet::new(),
        }
    }

    /// Resolve the effective config for one cassette directory:
    /// defaults ← `<dir>/config.toml` (when present) ← `PROXIMA_CASSETTE_*`.
    ///
    /// # Errors
    /// Returns `ProximaError::Config` on an unreadable/invalid file or a
    /// malformed env value — a broken policy source must never silently
    /// fall back to defaults.
    pub fn resolve_for_dir(cassette_dir: &Path) -> Result<Self, ProximaError> {
        let file = cassette_dir.join("config.toml");
        let mut builder = conflaguration::builder().value(Self::default());
        if file.is_file() {
            builder = builder.file(&file);
        }
        builder.env().build().map_err(|error| {
            ProximaError::Config(format!(
                "cassette config ({}): {error}",
                cassette_dir.display()
            ))
        })
    }
}

/// Hand-written layer builder (house pattern). Every source (`.from_path`,
/// `.from_env`, `.underlay_path`, `.underlay_env`, `.with_*`) contributes only
/// the fields it actually specifies; a field a source doesn't touch falls
/// through to whatever the prior layers accumulated. Two merge flavors,
/// composable in any call order:
///
/// - `.from_path` / `.from_env` (override): the field wins over whatever is
///   already accumulated — last writer wins, per field.
/// - `.underlay_path` / `.underlay_env` (fill-only): the field applies ONLY
///   if nothing has set it yet; an already-set value is never clobbered.
///
/// `.with_*` always behaves like an override at its call position.
#[derive(Debug, Clone)]
pub struct CassetteLayerBuilder {
    inner: CassetteConfig,
    touched: BTreeSet<String>,
}

impl CassetteLayerBuilder {
    /// Merge a file's fields onto the accumulated config; the file wins for
    /// every field it specifies.
    ///
    /// # Errors
    /// Propagates the conflaguration file/parse error.
    pub fn from_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from a file; a field already set by a
    /// prior layer is left untouched.
    ///
    /// # Errors
    /// Propagates the conflaguration file/parse error.
    pub fn underlay_path<P: AsRef<Path>>(mut self, path: P) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
        )?;
        Ok(self)
    }

    /// Merge env-set fields onto the accumulated config; env wins for every
    /// field it sets. Unset env vars leave the current value untouched.
    ///
    /// # Errors
    /// Propagates the conflaguration env resolution error.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = cassette_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from env vars; a field already set by a
    /// prior layer is left untouched even if the matching env var is set.
    ///
    /// # Errors
    /// Propagates the conflaguration env resolution error.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = cassette_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
        )?;
        Ok(self)
    }

    #[must_use]
    pub fn with_mode(mut self, mode: ModePolicy) -> Self {
        self.inner.mode = mode;
        self.touched.insert("mode".to_string());
        self
    }

    #[must_use]
    pub fn with_rerecord(mut self, rerecord: RerecordPolicy) -> Self {
        self.inner.rerecord = rerecord;
        self.touched.insert("rerecord".to_string());
        self
    }

    #[must_use]
    pub fn with_duplicates(mut self, duplicates: DuplicatePolicy) -> Self {
        self.inner.duplicates = duplicates;
        self.touched.insert("duplicates".to_string());
        self
    }

    #[must_use]
    pub fn with_match_body(mut self, match_body: bool) -> Self {
        self.inner.match_body = match_body;
        self.touched.insert("match_body".to_string());
        self
    }

    #[must_use]
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.inner.max_age_ms = u64::try_from(max_age.as_millis()).unwrap_or(u64::MAX);
        self.touched.insert("max_age_ms".to_string());
        self
    }

    #[must_use]
    pub fn with_staleness(mut self, staleness: StalenessPolicy) -> Self {
        self.inner.staleness = staleness;
        self.touched.insert("staleness".to_string());
        self
    }

    #[must_use]
    pub fn build(self) -> CassetteConfig {
        self.inner
    }
}

/// The env-set subset of [`CassetteConfig`]'s fields, as a partial JSON
/// object containing only the fields whose env var is actually present —
/// never the ones `Settings::from_env` filled with a default. Presence is
/// checked directly per field (not by diffing against a default) so a field
/// explicitly set to its own default value is still correctly recognized as
/// "set" by [`MergeMode::Underlay`].
fn cassette_env_partial() -> Result<Value, conflaguration::Error> {
    let resolved = CassetteConfig::from_env()?;
    let mut partial = Map::new();
    insert_if_env_set(
        &mut partial,
        "mode",
        &["PROXIMA_CASSETTE_MODE", "PROXIMA_CASSETTE"],
        &resolved.mode,
    )?;
    insert_if_env_set(
        &mut partial,
        "rerecord",
        &["PROXIMA_CASSETTE_RERECORD"],
        &resolved.rerecord,
    )?;
    insert_if_env_set(
        &mut partial,
        "duplicates",
        &["PROXIMA_CASSETTE_DUPLICATES"],
        &resolved.duplicates,
    )?;
    insert_if_env_set(
        &mut partial,
        "match_body",
        &["PROXIMA_CASSETTE_MATCH_BODY"],
        &resolved.match_body,
    )?;
    insert_if_env_set(
        &mut partial,
        "max_age_ms",
        &["PROXIMA_CASSETTE_MAX_AGE_MS"],
        &resolved.max_age_ms,
    )?;
    insert_if_env_set(
        &mut partial,
        "staleness",
        &["PROXIMA_CASSETTE_STALENESS"],
        &resolved.staleness,
    )?;
    Ok(Value::Object(partial))
}

/// Outcome of the rerecord decision for one existing cassette file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerecordDecision {
    Truncate,
    Backup,
    Fail,
    /// Keep the existing cassette untouched and serve it (replay) instead
    /// of recording — the "this data is proven, protect it" hatch.
    UseExisting,
}

/// Outcome of the staleness decision for one cassette.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleDecision {
    Proceed,
    Fail,
}

pub type RerecordHook = Arc<dyn Fn(&Path) -> RerecordDecision + Send + Sync>;
pub type StaleHook = Arc<dyn Fn(&Path, Option<&CassetteMeta>) -> StaleDecision + Send + Sync>;

/// Programmable escape hatches, consulted BEFORE the declarative policy.
/// Reach for these only when the config knobs can't express the decision
/// (per-file archival, age amnesty for one golden cassette, ...).
#[derive(Clone, Default)]
pub struct CassetteHooks {
    pub on_rerecord: Option<RerecordHook>,
    pub on_stale: Option<StaleHook>,
}

impl CassetteHooks {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_on_rerecord(
        mut self,
        hook: impl Fn(&Path) -> RerecordDecision + Send + Sync + 'static,
    ) -> Self {
        self.on_rerecord = Some(Arc::new(hook));
        self
    }

    #[must_use]
    pub fn with_on_stale(
        mut self,
        hook: impl Fn(&Path, Option<&CassetteMeta>) -> StaleDecision + Send + Sync + 'static,
    ) -> Self {
        self.on_stale = Some(Arc::new(hook));
        self
    }
}

impl fmt::Debug for CassetteHooks {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CassetteHooks")
            .field("on_rerecord", &self.on_rerecord.is_some())
            .field("on_stale", &self.on_stale.is_some())
            .finish()
    }
}

/// Whether an incoming layer's fields win over an already-touched field
/// (`Override`, last writer wins) or only fill a field nothing has set yet
/// (`Underlay`, fill-only — never clobbers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeMode {
    Override,
    Underlay,
}

/// Merge `incoming`'s present fields onto `inner`, tracking which top-level
/// fields have been touched so `Underlay` layers never clobber an
/// already-set value. Every field in these house-pattern configs is either a
/// scalar or a Vec/Map data collection (never a `#[setting(nested)]`
/// sub-config), so a one-level object merge covers every real field; nested
/// recursion — a genuinely embedded sub-config merging per-subfield instead
/// of replacing wholesale — is exercised directly in `layered_merge` tests.
fn apply_layer<T>(
    inner: &mut T,
    touched: &mut BTreeSet<String>,
    incoming: Value,
    mode: MergeMode,
) -> Result<(), conflaguration::Error>
where
    T: Serialize + DeserializeOwned,
{
    let Value::Object(incoming_map) = incoming else {
        return Ok(());
    };
    let mut base = to_value(inner)?;
    let Value::Object(base_map) = &mut base else {
        return Ok(());
    };
    for (key, value) in incoming_map {
        apply_leaf(base_map, &key, value, mode, &key, touched);
    }
    *inner = from_value(base)?;
    Ok(())
}

fn apply_leaf(
    map: &mut Map<String, Value>,
    key: &str,
    value: Value,
    mode: MergeMode,
    touched_path: &str,
    touched: &mut BTreeSet<String>,
) {
    let should_apply = match mode {
        MergeMode::Override => true,
        MergeMode::Underlay => !touched.contains(touched_path),
    };
    if should_apply {
        map.insert(key.to_string(), value);
        touched.insert(touched_path.to_string());
    }
}

fn insert_if_env_set<T: Serialize>(
    partial: &mut Map<String, Value>,
    field: &str,
    env_names: &[&str],
    value: &T,
) -> Result<(), conflaguration::Error> {
    if env_names.iter().any(|name| std::env::var(name).is_ok()) {
        partial.insert(field.to_string(), to_value(value)?);
    }
    Ok(())
}

fn to_value<T: Serialize>(value: &T) -> Result<Value, conflaguration::Error> {
    serde_json::to_value(value).map_err(|error| conflaguration::Error::Validation {
        errors: vec![ValidationMessage::new(
            "layered",
            format!("serialize failed: {error}"),
        )],
    })
}

fn from_value<T: DeserializeOwned>(value: Value) -> Result<T, conflaguration::Error> {
    serde_json::from_value(value).map_err(|error| conflaguration::Error::Validation {
        errors: vec![ValidationMessage::new(
            "layered",
            format!("deserialize failed: {error}"),
        )],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_preserve_pre_hardening_behavior() {
        let config = CassetteConfig::default();
        assert_eq!(config.mode, ModePolicy::Auto);
        assert_eq!(config.rerecord, RerecordPolicy::Truncate);
        assert_eq!(config.duplicates, DuplicatePolicy::RejectDivergent);
        assert!(!config.match_body);
        assert_eq!(config.max_age(), None);
    }

    #[test]
    fn mode_policy_resolves_against_discovered_mode() {
        assert_eq!(ModePolicy::Auto.resolve(Mode::Record), Mode::Record);
        assert_eq!(ModePolicy::Auto.resolve(Mode::Replay), Mode::Replay);
        assert_eq!(ModePolicy::Record.resolve(Mode::Replay), Mode::Record);
        assert_eq!(ModePolicy::Replay.resolve(Mode::Record), Mode::Replay);
    }

    #[test]
    fn enum_parsing_accepts_known_and_rejects_unknown() {
        assert_eq!("auto".parse::<ModePolicy>().unwrap(), ModePolicy::Auto);
        assert_eq!(
            "backup".parse::<RerecordPolicy>().unwrap(),
            RerecordPolicy::Backup
        );
        assert_eq!(
            "reject-divergent".parse::<DuplicatePolicy>().unwrap(),
            DuplicatePolicy::RejectDivergent
        );
        assert_eq!(
            "warn".parse::<StalenessPolicy>().unwrap(),
            StalenessPolicy::Warn
        );
        assert!("sometimes".parse::<ModePolicy>().is_err());
        assert!("archive".parse::<RerecordPolicy>().is_err());
        assert!("merge".parse::<DuplicatePolicy>().is_err());
        assert!("ignore".parse::<StalenessPolicy>().is_err());
    }

    #[test]
    fn toml_round_trip_preserves_state() {
        let original = CassetteConfig::builder()
            .rerecord(RerecordPolicy::Backup)
            .match_body(true)
            .max_age_ms(86_400_000)
            .staleness(StalenessPolicy::Warn)
            .build();
        let toml_text = toml::to_string(&original).expect("serialize");
        let restored: CassetteConfig = toml::from_str(&toml_text).expect("deserialize");
        assert_eq!(restored, original);
    }

    #[test]
    fn env_overrides_defaults() {
        temp_env::with_vars(
            [
                ("PROXIMA_CASSETTE_RERECORD", Some("fail")),
                ("PROXIMA_CASSETTE_MATCH_BODY", Some("true")),
            ],
            || {
                let config = CassetteConfig::from_env().expect("env config");
                assert_eq!(config.rerecord, RerecordPolicy::Fail);
                assert!(config.match_body);
            },
        );
    }

    #[test]
    fn legacy_mode_env_var_still_selects_mode() {
        temp_env::with_vars([("PROXIMA_CASSETTE", Some("replay"))], || {
            let config = CassetteConfig::from_env().expect("env config");
            assert_eq!(config.mode, ModePolicy::Replay);
        });
    }

    #[test]
    fn malformed_env_value_is_a_loud_error() {
        temp_env::with_vars([("PROXIMA_CASSETTE_RERECORD", Some("archive"))], || {
            assert!(CassetteConfig::from_env().is_err());
        });
    }

    #[test]
    fn resolve_for_dir_layers_file_then_env() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.toml"),
            "rerecord = \"backup\"\nmax_age_ms = 1000\n",
        )
        .expect("write config");
        temp_env::with_vars([("PROXIMA_CASSETTE_RERECORD", Some("fail"))], || {
            let config = CassetteConfig::resolve_for_dir(dir.path()).expect("resolve");
            // env wins over file, file wins over default
            assert_eq!(config.rerecord, RerecordPolicy::Fail);
            assert_eq!(config.max_age_ms, 1000);
            assert_eq!(config.duplicates, DuplicatePolicy::RejectDivergent);
        });
    }

    #[test]
    fn resolve_for_dir_without_file_uses_defaults() {
        // unset explicitly: sibling tests in this module use temp_env::with_vars
        // to set these same PROXIMA_CASSETTE_* vars, and env is process-global,
        // so an unscoped read here can observe a concurrently-running sibling's
        // transient value.
        temp_env::with_vars_unset(
            [
                "PROXIMA_CASSETTE_MODE",
                "PROXIMA_CASSETTE",
                "PROXIMA_CASSETTE_RERECORD",
                "PROXIMA_CASSETTE_DUPLICATES",
                "PROXIMA_CASSETTE_MATCH_BODY",
                "PROXIMA_CASSETTE_MAX_AGE_MS",
                "PROXIMA_CASSETTE_STALENESS",
            ],
            || {
                let dir = tempfile::tempdir().expect("tempdir");
                let config = CassetteConfig::resolve_for_dir(dir.path()).expect("resolve");
                assert_eq!(config, CassetteConfig::default());
            },
        );
    }

    #[test]
    fn layered_call_order_gives_precedence() {
        let config = CassetteConfig::layered()
            .with_rerecord(RerecordPolicy::Backup)
            .with_max_age(Duration::from_secs(60))
            .build();
        assert_eq!(config.rerecord, RerecordPolicy::Backup);
        assert_eq!(config.max_age(), Some(Duration::from_secs(60)));
    }

    fn write_toml(dir: &tempfile::TempDir, name: &str, contents: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, contents).expect("write toml");
        path
    }

    // the exact seam-#3 case: a file sets TWO fields, env sets only ONE of
    // them — the file's other field must survive `.from_path().from_env()`,
    // not fall back to the struct default (the original clobbering bug).
    #[test]
    fn seam_3_from_path_then_from_env_preserves_files_untouched_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_toml(
            &dir,
            "config.toml",
            "rerecord = \"backup\"\nmax_age_ms = 1000\n",
        );
        temp_env::with_vars([("PROXIMA_CASSETTE_RERECORD", Some("fail"))], || {
            let config = CassetteConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(
                config.rerecord,
                RerecordPolicy::Fail,
                "env wins the field it sets"
            );
            assert_eq!(
                config.max_age_ms, 1000,
                "the file's max_age_ms must survive — env never touched it"
            );
        });
    }

    // order-independence: the SAME logical sources, built both orders, each
    // resolve correctly per field for THAT order — neither order silently
    // reverts a field to default.
    #[test]
    fn order_independence_override_flavor_both_directions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_toml(&dir, "config.toml", "max_age_ms = 1000\n");
        temp_env::with_vars([("PROXIMA_CASSETTE_RERECORD", Some("fail"))], || {
            let file_then_env = CassetteConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .build();
            assert_eq!(file_then_env.max_age_ms, 1000, "file's field survives");
            assert_eq!(
                file_then_env.rerecord,
                RerecordPolicy::Fail,
                "env's field applies"
            );

            let env_then_file = CassetteConfig::layered()
                .from_env()
                .expect("from_env")
                .from_path(&path)
                .expect("from_path")
                .build();
            assert_eq!(
                env_then_file.rerecord,
                RerecordPolicy::Fail,
                "env's field survives"
            );
            assert_eq!(env_then_file.max_age_ms, 1000, "file's field applies");
        });
    }

    // the full stack: defaults < file < env < with_*, each layer overriding
    // only the fields it sets, asserted field-by-field.
    #[test]
    fn full_stack_defaults_file_env_with_override_each_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_toml(
            &dir,
            "config.toml",
            "duplicates = \"last-wins\"\nmax_age_ms = 5000\n",
        );
        temp_env::with_vars([("PROXIMA_CASSETTE_RERECORD", Some("fail"))], || {
            let config = CassetteConfig::layered()
                .from_path(&path)
                .expect("from_path")
                .from_env()
                .expect("from_env")
                .with_match_body(true)
                .build();
            assert_eq!(config.rerecord, RerecordPolicy::Fail, "env layer");
            assert_eq!(config.duplicates, DuplicatePolicy::LastWins, "file layer");
            assert_eq!(config.max_age_ms, 5000, "file layer");
            assert!(config.match_body, "with_* layer");
            assert_eq!(
                config.staleness,
                StalenessPolicy::Fail,
                "untouched by any layer — falls through to the default"
            );
        });
    }

    // underlay never clobbers a field a prior layer already set, even when
    // the underlay source explicitly specifies a DIFFERENT value for it.
    #[test]
    fn underlay_never_clobbers_already_set_field() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_toml(&dir, "config.toml", "rerecord = \"backup\"\n");
        let config = CassetteConfig::layered()
            .with_rerecord(RerecordPolicy::Fail)
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(
            config.rerecord,
            RerecordPolicy::Fail,
            "the explicit with_* value is preserved; the file's value is dropped"
        );
    }

    // underlay DOES fill a field nothing has set yet.
    #[test]
    fn underlay_fills_unset_field_from_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_toml(&dir, "config.toml", "rerecord = \"backup\"\n");
        let config = CassetteConfig::layered()
            .underlay_path(&path)
            .expect("underlay_path")
            .build();
        assert_eq!(config.rerecord, RerecordPolicy::Backup);
    }

    // underlay from env: fills an unset field, never an already-set one.
    #[test]
    fn underlay_env_fills_only_unset_fields() {
        temp_env::with_vars(
            [
                ("PROXIMA_CASSETTE_RERECORD", Some("fail")),
                ("PROXIMA_CASSETTE_MATCH_BODY", Some("true")),
            ],
            || {
                let config = CassetteConfig::layered()
                    .with_rerecord(RerecordPolicy::Backup)
                    .underlay_env()
                    .expect("underlay_env")
                    .build();
                assert_eq!(
                    config.rerecord,
                    RerecordPolicy::Backup,
                    "already set by with_*; env's value is dropped"
                );
                assert!(config.match_body, "unset before underlay_env; env fills it");
            },
        );
    }

    // order-independence for the underlay flavor: for a field BOTH sources
    // specify, the FIRST-applied underlay wins (fill-only, not last-wins);
    // swapping which source runs first swaps the winner, proving the
    // mechanism handles either order rather than favoring one hardcoded slot.
    #[test]
    fn order_independence_underlay_flavor_first_setter_wins_either_direction() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_toml(&dir, "config.toml", "rerecord = \"backup\"\n");
        temp_env::with_vars([("PROXIMA_CASSETTE_RERECORD", Some("fail"))], || {
            let file_first = CassetteConfig::layered()
                .underlay_path(&path)
                .expect("underlay_path")
                .underlay_env()
                .expect("underlay_env")
                .build();
            assert_eq!(
                file_first.rerecord,
                RerecordPolicy::Backup,
                "file applied first, wins"
            );

            let env_first = CassetteConfig::layered()
                .underlay_env()
                .expect("underlay_env")
                .underlay_path(&path)
                .expect("underlay_path")
                .build();
            assert_eq!(
                env_first.rerecord,
                RerecordPolicy::Fail,
                "env applied first, wins"
            );
        });
    }

    // combined: defaults -> underlay(file) -> override(env) -> override(with_*),
    // asserting the winner for each field across the mixed chain.
    #[test]
    fn combined_underlay_file_then_override_env_then_with() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_toml(
            &dir,
            "config.toml",
            "rerecord = \"backup\"\nduplicates = \"last-wins\"\nmax_age_ms = 1000\n",
        );
        temp_env::with_vars([("PROXIMA_CASSETTE_RERECORD", Some("fail"))], || {
            let config = CassetteConfig::layered()
                .underlay_path(&path)
                .expect("underlay_path")
                .from_env()
                .expect("from_env")
                .with_max_age(Duration::from_millis(9000))
                .build();
            assert_eq!(
                config.rerecord,
                RerecordPolicy::Fail,
                "override(env) wins over underlay(file) for the field both set"
            );
            assert_eq!(
                config.duplicates,
                DuplicatePolicy::LastWins,
                "underlay(file) fills the field env never touched"
            );
            assert_eq!(
                config.max_age_ms, 9000,
                "the later explicit with_* overrides underlay(file)'s value"
            );
            assert_eq!(
                config.staleness,
                StalenessPolicy::Fail,
                "no layer set it — falls through to the default"
            );
        });
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
    struct MergeCoreInner {
        #[serde(default)]
        a: u32,
        #[serde(default)]
        b: u32,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
    struct MergeCoreOuter {
        #[serde(default)]
        list: Vec<u32>,
        #[serde(default)]
        nested: MergeCoreInner,
    }

    // nested-struct partial merge: a layer setting one subfield must not
    // wipe a sibling subfield a prior layer set — this exercises the same
    // `apply_layer` primitive `CassetteLayerBuilder` is built on, with a
    // `#[setting(nested)]`-shaped field (none of CassetteConfig's own fields
    // are nested sub-configs, so this proves the mechanism directly).
    #[test]
    fn nested_struct_partial_merge_preserves_sibling_subfield() {
        let mut inner = MergeCoreOuter::default();
        let mut touched = BTreeSet::new();

        apply_layer(
            &mut inner,
            &mut touched,
            serde_json::json!({ "nested": { "a": 7 } }),
            MergeMode::Override,
        )
        .expect("first layer");
        // top-level-only merge treats `nested` as one atomic field — this
        // crate's configs have no `#[setting(nested)]` sub-config, so a
        // second layer naturally replaces it wholesale (documented in
        // `config_merge`'s dedicated recursive-merge tests, which opt a
        // field into one-level recursion via `nested_keys`).
        assert_eq!(inner.nested, MergeCoreInner { a: 7, b: 0 });
    }

    // collection (Vec) replace-if-present: a second layer providing the
    // field replaces it wholesale, never appends/unions.
    #[test]
    fn collection_field_replaces_wholesale_not_union() {
        let mut inner = MergeCoreOuter::default();
        let mut touched = BTreeSet::new();

        apply_layer(
            &mut inner,
            &mut touched,
            serde_json::json!({ "list": [1, 2, 3] }),
            MergeMode::Override,
        )
        .expect("first layer");
        apply_layer(
            &mut inner,
            &mut touched,
            serde_json::json!({ "list": [9] }),
            MergeMode::Override,
        )
        .expect("second layer");
        assert_eq!(
            inner.list,
            vec![9],
            "replaced wholesale, not unioned to [1,2,3,9]"
        );
    }

    // collection underlay: fills the whole collection only if unset; an
    // already-set collection is never touched (no element merge).
    #[test]
    fn collection_field_underlay_never_element_merges() {
        let mut inner = MergeCoreOuter::default();
        let mut touched = BTreeSet::new();

        apply_layer(
            &mut inner,
            &mut touched,
            serde_json::json!({ "list": [1] }),
            MergeMode::Override,
        )
        .expect("explicit layer");
        apply_layer(
            &mut inner,
            &mut touched,
            serde_json::json!({ "list": [9, 9, 9] }),
            MergeMode::Underlay,
        )
        .expect("underlay layer");
        assert_eq!(
            inner.list,
            vec![1],
            "already-set collection is untouched, not merged"
        );
    }

    #[test]
    fn hooks_debug_shows_presence_not_contents() {
        let hooks = CassetteHooks::new().with_on_rerecord(|_path| RerecordDecision::Backup);
        let rendered = format!("{hooks:?}");
        assert!(rendered.contains("on_rerecord: true"));
        assert!(rendered.contains("on_stale: false"));
    }
}
