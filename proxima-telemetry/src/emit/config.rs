//! `EmitConfig` — the typed config + fluent builder surface (the "make it
//! better" half: P4 first-class config AND builder).
//!
//! Mirrors [`crate::config::TelemetryConfig`] exactly: `#[derive(Builder,
//! Deserialize, Serialize, Settings)]` + [`Validate`], a `EmitLayerBuilder` with
//! call-order precedence, and a typed env surface. The wire form is strings
//! (`"warn"`, `"17.3"`, a named level); [`EmitConfig::compile`] lowers them to a
//! typed [`CompiledEmit`], and [`Validate`] runs the same parse at load so a typo
//! is a validation error that lists the valid options — never a silent no-op the
//! way a `RUST_LOG` typo is.
//!
//! Tier: T2 (std — conflaguration, fs, env).

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::str::FromStr;
use std::collections::BTreeSet;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::config_merge::{MergeMode, apply_layer, insert_if_env_set};
use crate::emit::{CompiledEmit, Coord, EmitRule, EmitThreshold, LevelTree, MatchMode};
use crate::level::Level;

fn default_emit_default() -> String {
    "warn".to_string()
}
fn default_match_mode() -> String {
    "boundary".to_string()
}

/// Typed emit-control config. One built `EmitConfig` == one serialisable config
/// == one [`CompiledEmit`] (via [`EmitConfig::compile`]).
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "EMIT")]
#[builder(derive(Clone, Debug))]
pub struct EmitConfig {
    /// The NAMED global default threshold — a level name, dotted coord, or `off`.
    /// Never positional: this is the explicit fix for `RUST_LOG`'s "is there a
    /// default and where?" ambiguity.
    #[setting(default = "warn")]
    #[serde(default = "default_emit_default")]
    #[builder(default = default_emit_default())]
    pub default: String,

    /// Prefix match mode: `"boundary"` (native, `::`-aware) or `"raw"` (tracing
    /// `EnvFilter` parity).
    #[setting(default = "boundary")]
    #[serde(default = "default_match_mode")]
    #[builder(default = default_match_mode())]
    pub match_mode: String,

    /// Per-target overrides, keyed by module-path prefix — an OPEN-ENDED map.
    /// TOML: `[targets."proxima::h2"]` (serde fills the map via
    /// `conflaguration::from_file`); env: any `EMIT_TARGET_<key>` var
    /// (`__` → `::`) is collected by a prefix scan in
    /// [`EmitLayerBuilder::from_env`]. `#[setting(skip)]` only because
    /// conflaguration's *derive* can't yet prefix-collect a map from env (the env
    /// scan is done explicitly); the TOML side is fully conflaguration.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub targets: BTreeMap<String, TargetSpec>,

    /// Declared filterable targets, for validation + discovery. When non-empty,
    /// a target not in this set is a validation error listing the known targets
    /// — what `RUST_LOG` structurally cannot do.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub known_targets: Vec<String>,
}

/// A per-target rule's value in the `targets` map (the target is the map key).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TargetSpec {
    /// A level name, dotted coord, or `off`.
    pub level: String,
    /// Optional verbose subtree kept regardless of the floor.
    #[serde(default)]
    pub verbose: Option<String>,
}

/// Why an [`EmitConfig`] failed to compile — structured so the message can list
/// the valid options (discoverability).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitConfigError {
    /// A level/coord string didn't parse.
    BadLevel { field: String, got: String },
    /// An unknown match mode.
    BadMatchMode { got: String },
    /// A target not present in `known_targets`.
    UnknownTarget { got: String, known: Vec<String> },
}

impl core::fmt::Display for EmitConfigError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadLevel { field, got } => write!(
                formatter,
                "{field}: unknown level '{got}'; expected trace|debug|info|warn|error|fatal|off, a dotted coord (e.g. 17.3), or a registered name"
            ),
            Self::BadMatchMode { got } => {
                write!(
                    formatter,
                    "match_mode: unknown '{got}'; expected boundary|raw"
                )
            }
            Self::UnknownTarget { got, known } => {
                write!(
                    formatter,
                    "unknown target '{got}'; known: {}",
                    known.join(", ")
                )
            }
        }
    }
}

impl core::error::Error for EmitConfigError {}

impl EmitConfig {
    /// Lower the config to the typed [`CompiledEmit`]. All string parsing,
    /// validation, and the longest-prefix sort happen here (config time), never
    /// per record.
    pub fn compile(&self) -> Result<CompiledEmit, EmitConfigError> {
        self.compile_with(&LevelTree::empty())
    }

    /// Lower the config, resolving named levels against `levels`. Use this when
    /// the config references named hierarchical levels (`level = "security"`);
    /// [`compile`](Self::compile) is the empty-tree case (flat/dotted/off only).
    pub fn compile_with(&self, levels: &LevelTree) -> Result<CompiledEmit, EmitConfigError> {
        let default = EmitThreshold::at(parse_floor("default", &self.default, levels)?);
        let mode = parse_mode(&self.match_mode)?;
        let mut rules = Vec::with_capacity(self.targets.len());
        for (target, spec) in &self.targets {
            if !self.known_targets.is_empty()
                && !self.known_targets.iter().any(|known| known == target)
            {
                return Err(EmitConfigError::UnknownTarget {
                    got: target.clone(),
                    known: self.known_targets.clone(),
                });
            }
            let floor = parse_floor("targets.level", &spec.level, levels)?;
            let verbose = match &spec.verbose {
                Some(text) => Some(parse_coord("targets.verbose", text, levels)?),
                None => None,
            };
            rules.push(EmitRule::new(
                target.clone(),
                EmitThreshold {
                    floor,
                    verbose_subtree: verbose,
                },
            ));
        }
        Ok(CompiledEmit::build(default, rules, mode))
    }
}

impl Validate for EmitConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        // a config that compiles is valid — reuse the real lowering so the load
        // path catches the same typos compile() would, with the listing message.
        self.compile()
            .map(|_| ())
            .map_err(|err| conflaguration::Error::Validation {
                errors: alloc::vec![ValidationMessage::new("emit", err.to_string())],
            })
    }
}

fn parse_floor(field: &str, text: &str, levels: &LevelTree) -> Result<Coord, EmitConfigError> {
    let text = text.trim();
    if text.eq_ignore_ascii_case("off") {
        return Ok(Coord::from_severity(u8::MAX)); // drop-all floor
    }
    if let Ok(level) = Level::from_str(text) {
        return Ok(Coord::from(level));
    }
    if let Ok(coord) = Coord::parse(text) {
        return Ok(coord);
    }
    // otherwise a named hierarchical level — resolved against the tree.
    levels
        .resolve(text)
        .ok_or_else(|| EmitConfigError::BadLevel {
            field: field.to_string(),
            got: text.to_string(),
        })
}

fn parse_coord(field: &str, text: &str, levels: &LevelTree) -> Result<Coord, EmitConfigError> {
    let text = text.trim();
    if let Ok(coord) = Coord::parse(text) {
        return Ok(coord);
    }
    levels
        .resolve(text)
        .ok_or_else(|| EmitConfigError::BadLevel {
            field: field.to_string(),
            got: text.to_string(),
        })
}

fn parse_mode(text: &str) -> Result<MatchMode, EmitConfigError> {
    match text.trim().to_ascii_lowercase().as_str() {
        "boundary" => Ok(MatchMode::Boundary),
        "raw" => Ok(MatchMode::Raw),
        _ => Err(EmitConfigError::BadMatchMode {
            got: text.to_string(),
        }),
    }
}

/// Fluent builder for [`EmitConfig`] (mirrors `TelemetryLayerBuilder`). Every
/// source contributes only the fields it actually specifies, merged onto the
/// accumulated config. `.from_path`/`.from_env` override (last writer wins
/// per field); `.underlay_path`/`.underlay_env` fill only fields still
/// unset; `.with_*` always acts as an override at its call position.
/// `targets` is a collection like any other: a source that provides it
/// replaces it wholesale, never element-merges with a prior layer's map.
pub struct EmitLayerBuilder {
    inner: EmitConfig,
    touched: BTreeSet<String>,
}

impl EmitConfig {
    /// Start a layered builder from defaults.
    #[must_use]
    pub fn layered() -> EmitLayerBuilder {
        EmitLayerBuilder {
            inner: EmitConfig::builder().build(),
            touched: BTreeSet::new(),
        }
    }
}

impl EmitLayerBuilder {
    /// Merge a TOML/JSON file's fields onto the accumulated config; the file
    /// wins for every field it specifies.
    pub fn from_path<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
            &[],
        )?;
        Ok(self)
    }

    /// Fill any still-unset fields from a TOML/JSON file; already-set fields
    /// are left untouched.
    pub fn underlay_path<P: AsRef<std::path::Path>>(
        mut self,
        path: P,
    ) -> Result<Self, conflaguration::Error> {
        let incoming: Value = conflaguration::from_file(path.as_ref())?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
            &[],
        )?;
        Ok(self)
    }

    /// Merge `EMIT_*` scalars onto the accumulated config (env wins for every
    /// field it sets), then collect per-target rules OPEN-ENDED by scanning
    /// every `EMIT_TARGET_<key>=level[;verbose=coord]` var — no count, no
    /// index. `<key>` uses `__` for `::` (env names can't hold `::`), so
    /// `EMIT_TARGET_proxima__h2=debug` is the rule for `proxima::h2`. The
    /// scanned targets replace `targets` wholesale, like any collection.
    pub fn from_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = emit_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Override,
            &[],
        )?;
        if let Some(targets) = env_target_contribution()? {
            self.inner.targets = targets;
            self.touched.insert("targets".to_string());
        }
        Ok(self)
    }

    /// Fill any still-unset `EMIT_*` scalars from env, and fill `targets`
    /// from the `EMIT_TARGET_*` scan ONLY if nothing has set `targets` yet.
    pub fn underlay_env(mut self) -> Result<Self, conflaguration::Error> {
        let incoming = emit_env_partial()?;
        apply_layer(
            &mut self.inner,
            &mut self.touched,
            incoming,
            MergeMode::Underlay,
            &[],
        )?;
        if !self.touched.contains("targets")
            && let Some(targets) = env_target_contribution()?
        {
            self.inner.targets = targets;
            self.touched.insert("targets".to_string());
        }
        Ok(self)
    }

    /// Set the named default threshold.
    #[must_use]
    pub fn with_default(mut self, level: impl Into<String>) -> Self {
        self.inner.default = level.into();
        self.touched.insert("default".to_string());
        self
    }

    /// Choose the prefix match mode (`"boundary"` | `"raw"`).
    #[must_use]
    pub fn with_match_mode(mut self, mode: impl Into<String>) -> Self {
        self.inner.match_mode = mode.into();
        self.touched.insert("match_mode".to_string());
        self
    }

    /// Add a per-target rule.
    #[must_use]
    pub fn with_target(mut self, target: impl Into<String>, level: impl Into<String>) -> Self {
        self.inner.targets.insert(
            target.into(),
            TargetSpec {
                level: level.into(),
                verbose: None,
            },
        );
        self.touched.insert("targets".to_string());
        self
    }

    /// Add a per-target rule with a verbose subtree.
    #[must_use]
    pub fn with_verbose_target(
        mut self,
        target: impl Into<String>,
        level: impl Into<String>,
        verbose: impl Into<String>,
    ) -> Self {
        self.inner.targets.insert(
            target.into(),
            TargetSpec {
                level: level.into(),
                verbose: Some(verbose.into()),
            },
        );
        self.touched.insert("targets".to_string());
        self
    }

    /// Declare the known filterable targets (enables typo validation).
    #[must_use]
    pub fn with_known_targets(mut self, targets: Vec<String>) -> Self {
        self.inner.known_targets = targets;
        self.touched.insert("known_targets".to_string());
        self
    }

    /// The built immutable config.
    #[must_use]
    pub fn build(self) -> EmitConfig {
        self.inner
    }

    /// Build and compile in one step.
    pub fn compile(self) -> Result<CompiledEmit, EmitConfigError> {
        self.inner.compile()
    }

    /// Build and compile, resolving named levels against `levels`.
    pub fn compile_with(self, levels: &LevelTree) -> Result<CompiledEmit, EmitConfigError> {
        self.inner.compile_with(levels)
    }
}

/// The env-set subset of [`EmitConfig`]'s scalar fields (`targets` is handled
/// separately by [`env_target_contribution`], since it's `#[setting(skip)]`).
fn emit_env_partial() -> Result<Value, conflaguration::Error> {
    let resolved = EmitConfig::from_env()?;
    let mut partial = Map::new();
    insert_if_env_set(
        &mut partial,
        "default",
        &["EMIT_DEFAULT"],
        &resolved.default,
    )?;
    insert_if_env_set(
        &mut partial,
        "match_mode",
        &["EMIT_MATCH_MODE"],
        &resolved.match_mode,
    )?;
    Ok(Value::Object(partial))
}

/// Scan every `EMIT_TARGET_<key>` var in the current process env into a
/// target map — this layer's whole contribution, or `None` if the scan found
/// nothing (so the field falls through like any other unset collection).
fn env_target_contribution() -> Result<Option<BTreeMap<String, TargetSpec>>, conflaguration::Error>
{
    let mut found = BTreeMap::new();
    for (key, value) in std::env::vars() {
        if let Some(suffix) = key.strip_prefix("EMIT_TARGET_") {
            let target = suffix.replace("__", "::");
            found.insert(target, parse_target_threshold_env(&key, &value)?);
        }
    }
    Ok((!found.is_empty()).then_some(found))
}

/// Parse an `EMIT_TARGET_*` value `level[;verbose=coord]` (the target is the env
/// key, not the value).
fn parse_target_threshold_env(key: &str, value: &str) -> Result<TargetSpec, conflaguration::Error> {
    let (level, verbose) = match value.split_once(";verbose=") {
        Some((level, verbose)) => (level.trim(), Some(verbose.trim().to_string())),
        None => (value.trim(), None),
    };
    if level.is_empty() {
        return Err(conflaguration::Error::Validation {
            errors: alloc::vec![ValidationMessage::new(key, "expected a level")],
        });
    }
    Ok(TargetSpec {
        level: level.to_string(),
        verbose,
    })
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use alloc::collections::BTreeMap;

    use super::{EmitConfig, TargetSpec};
    use crate::emit::{Coord, Decision};
    use crate::level::Level;

    // P4 parity: a built config round-trips through serde AND compiles to the
    // same decisions (behavior parity, not just struct equality).
    #[test]
    fn config_round_trips_struct_and_decisions() {
        let built = EmitConfig::layered()
            .with_default("warn")
            .with_target("proxima::h2", "debug")
            .with_verbose_target("proxima::h2::hpack", "warn", "9.2")
            .build();

        // serde round-trip is identity
        let json_text = serde_json::to_string(&built).unwrap();
        let from_json: EmitConfig = serde_json::from_str(&json_text).unwrap();
        assert_eq!(from_json, built);

        // compiling both yields the SAME decisions
        let compiled = built.compile().unwrap();
        let recompiled = from_json.compile().unwrap();
        for (target, coord, want) in [
            (
                "proxima::h2::frame",
                Coord::from(Level::DEBUG),
                Decision::Keep,
            ),
            (
                "proxima::h2::hpack::evict",
                Coord::parse("9.2.4").unwrap(),
                Decision::Keep,
            ),
            ("downstream::store", Coord::from(Level::INFO), Decision::Drop),
        ] {
            assert_eq!(compiled.decide(target, coord), want);
            assert_eq!(recompiled.decide(target, coord), want);
        }
    }

    // a typo'd level is a compile/validation error that LISTS the valid options
    // (the RUST_LOG killer).
    #[test]
    fn typo_is_an_error_that_lists_options() {
        let cfg = EmitConfig::layered().with_target("proxima", "eror").build();
        let err = cfg.compile().unwrap_err();
        let message = err.to_string();
        assert!(message.contains("eror"), "names the typo: {message}");
        assert!(message.contains("warn"), "lists valid levels: {message}");
    }

    // named hierarchical levels resolve through the tree and drive the verbose
    // subtree lever end-to-end (the "names not numbers" surface).
    #[test]
    fn named_level_resolves_via_tree() {
        use crate::emit::{Coord, Decision, LevelTree};
        let tree = LevelTree::builder()
            .family("security", Level::ERROR)
            .child("security", "security.auth")
            .build()
            .unwrap();
        let cfg = EmitConfig::layered()
            .with_default("warn")
            .with_verbose_target("app", "warn", "security") // verbose subtree = a NAME
            .build();

        // compile() can't resolve the name without the tree...
        assert!(cfg.compile().is_err());
        // ...compile_with(tree) does.
        let compiled = cfg.compile_with(&tree).unwrap();

        // a record in the security subtree is kept under "app" despite the warn floor
        let token = tree.resolve("security.auth").unwrap();
        assert_eq!(compiled.decide("app", token), Decision::Keep);
        // a plain info (not in the subtree, below warn) is dropped
        assert_eq!(
            compiled.decide("app", Coord::from(Level::INFO)),
            Decision::Drop
        );
    }

    // an unknown target (when known_targets declared) is rejected with the list.
    #[test]
    fn unknown_target_rejected_when_declared() {
        let cfg = EmitConfig::layered()
            .with_known_targets(vec!["proxima".to_string(), "downstream".to_string()])
            .with_target("proxma", "debug") // typo
            .build();
        let err = cfg.compile().unwrap_err();
        assert!(err.to_string().contains("proxma"));
        assert!(err.to_string().contains("known: proxima, downstream"));
    }

    // Validate (the load-time gate) rejects the same typo.
    #[test]
    fn validate_rejects_typo_at_load() {
        use conflaguration::Validate;
        let mut targets = BTreeMap::new();
        targets.insert(
            "proxima".to_string(),
            TargetSpec {
                level: "bogus".to_string(),
                verbose: None,
            },
        );
        let cfg = EmitConfig {
            default: "warn".to_string(),
            match_mode: "boundary".to_string(),
            targets,
            known_targets: vec![],
        };
        assert!(cfg.validate().is_err());
    }

    // OPEN-ENDED: any number of targets load from a real TOML map through the
    // actual `conflaguration::from_file` loader — no array, no count.
    #[test]
    fn targets_map_is_open_ended_via_conflaguration() {
        let toml = r#"
default = "warn"
match_mode = "boundary"
[targets."proxima::h2"]
level = "debug"
[targets."downstream::store"]
level = "error"
verbose = "9.2"
"#;
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("emit.toml");
        std::fs::write(&path, toml).unwrap();

        let cfg = EmitConfig::layered().from_path(&path).unwrap().build();
        assert_eq!(cfg.targets.len(), 2);
        assert_eq!(cfg.targets["proxima::h2"].level, "debug");
        assert_eq!(cfg.targets["downstream::store"].verbose.as_deref(), Some("9.2"));
    }

    // the exact seam-#3 case: a file sets TWO scalar fields, env sets only
    // ONE — the file's other field must survive `.from_path().from_env()`.
    #[test]
    fn seam_3_from_path_then_from_env_preserves_files_untouched_field() {
        let toml = "default = \"debug\"\nmatch_mode = \"raw\"\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("emit.toml");
        std::fs::write(&path, toml).unwrap();

        temp_env::with_vars([("EMIT_DEFAULT", Some("error"))], || {
            let cfg = EmitConfig::layered()
                .from_path(&path)
                .unwrap()
                .from_env()
                .unwrap()
                .build();
            assert_eq!(cfg.default, "error", "env wins the field it sets");
            assert_eq!(cfg.match_mode, "raw", "the file's field must survive");
        });
    }

    // order-independence: the same two sources, built both orders.
    #[test]
    fn order_independence_file_then_env_vs_env_then_file() {
        let toml = "match_mode = \"raw\"\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("emit.toml");
        std::fs::write(&path, toml).unwrap();

        temp_env::with_vars([("EMIT_DEFAULT", Some("error"))], || {
            let file_then_env = EmitConfig::layered()
                .from_path(&path)
                .unwrap()
                .from_env()
                .unwrap()
                .build();
            assert_eq!(file_then_env.match_mode, "raw", "file's field survives");
            assert_eq!(file_then_env.default, "error", "env's field applies");

            let env_then_file = EmitConfig::layered()
                .from_env()
                .unwrap()
                .from_path(&path)
                .unwrap()
                .build();
            assert_eq!(env_then_file.default, "error", "env's field survives");
            assert_eq!(env_then_file.match_mode, "raw", "file's field applies");
        });
    }

    // full stack: defaults < file < env < with_*.
    #[test]
    fn full_stack_defaults_file_env_with_override_each_field() {
        let toml = "match_mode = \"raw\"\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("emit.toml");
        std::fs::write(&path, toml).unwrap();

        temp_env::with_vars([("EMIT_DEFAULT", Some("error"))], || {
            let cfg = EmitConfig::layered()
                .from_path(&path)
                .unwrap()
                .from_env()
                .unwrap()
                .with_known_targets(vec!["proxima".to_string()])
                .build();
            assert_eq!(cfg.default, "error", "env layer");
            assert_eq!(cfg.match_mode, "raw", "file layer");
            assert_eq!(
                cfg.known_targets,
                vec!["proxima".to_string()],
                "with_* layer"
            );
        });
    }

    // underlay never clobbers an already-set field; it DOES fill an unset one.
    #[test]
    fn underlay_path_fills_only_unset_fields() {
        let toml = "default = \"debug\"\nmatch_mode = \"raw\"\n";
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("emit.toml");
        std::fs::write(&path, toml).unwrap();

        let cfg = EmitConfig::layered()
            .with_default("error")
            .underlay_path(&path)
            .unwrap()
            .build();
        assert_eq!(
            cfg.default, "error",
            "already set by with_*; the file's value is dropped"
        );
        assert_eq!(
            cfg.match_mode, "raw",
            "unset before underlay; the file fills it"
        );
    }

    // targets is a collection: it replaces wholesale on override, and
    // underlay never element-merges it once it's set.
    #[test]
    fn targets_collection_replaces_wholesale_and_underlay_never_element_merges() {
        temp_env::with_vars(
            [
                ("EMIT_TARGET_proxima", Some("debug")),
                ("EMIT_TARGET_downstream", Some("warn")),
            ],
            || {
                let overridden = EmitConfig::layered()
                    .with_target("legacy", "info")
                    .from_env()
                    .unwrap()
                    .build();
                assert_eq!(
                    overridden.targets.len(),
                    2,
                    "env's scan replaces the whole targets map, not unions with with_target's entry"
                );
                assert!(!overridden.targets.contains_key("legacy"));

                let underlaid = EmitConfig::layered()
                    .with_target("legacy", "info")
                    .underlay_env()
                    .unwrap()
                    .build();
                assert_eq!(
                    underlaid.targets.len(),
                    1,
                    "already-set targets map is untouched"
                );
                assert!(underlaid.targets.contains_key("legacy"));
            },
        );
    }
}
