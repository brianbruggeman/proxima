//! `SpeculativeSettings`: the conflaguration-facing (env/TOML) loader and
//! `bon` fluent builder for [`crate::SpeculativeConfig`]'s data, following
//! the house pattern (`proxima-telemetry/src/config.rs`'s `TelemetryConfig`,
//! `.claude/skills/conflag/SKILL.md`) -- a single owned data shape carrying
//! `bon::Builder`, `serde` (file I/O), and `conflaguration::{Settings,
//! Validate}` (env loading).
//!
//! This lives in its own `std`-gated module, separate from `serving.rs`,
//! because [`crate::SpeculativeConfig`] itself must stay available at the
//! no_std+alloc floor (`serving.rs`'s own module doc: no
//! `serde`/`toml`/`bon`/`conflaguration` in that module or its dependency
//! graph, and `ServingConfig`'s own `Copy` derive is load-bearing --
//! `examples/speculative_decode_parity.rs`'s OFF/ON pairs rely on it).
//! [`SpeculativeSettings`] owns the `String` cache-path storage
//! [`crate::SpeculativeConfig`]'s `&'model str` fields borrow from
//! ([`Self::as_speculative_config`]), the same relationship
//! `ServingConfig::model_path` already has to whatever owned path storage a
//! caller keeps alive.

use bon::Builder;
use conflaguration::{Settings, Validate};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::serving::{
    NgramMapParams, NgramModParams, SpeculativeConfig, SpeculativeType, SpeculativeTypeSet,
};

/// llama's `common_speculative_type`, serde/`FromStr`-facing mirror of
/// [`crate::SpeculativeType`] (kept separate so the no_std-safe enum in
/// `serving.rs` never derives `serde`). Variant names round-trip llama's own
/// `--spec-type` strings (`common_speculative_type_to_str`,
/// `common/speculative.cpp:2229-2244`) via `#[serde(rename = ..)]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeculativeTypeName {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "draft-simple")]
    DraftSimple,
    #[serde(rename = "draft-eagle3")]
    DraftEagle3,
    #[serde(rename = "draft-mtp")]
    DraftMtp,
    #[serde(rename = "draft-dflash")]
    DraftDflash,
    #[serde(rename = "draft-dspark")]
    DraftDspark,
    #[serde(rename = "ngram-simple")]
    NgramSimple,
    #[serde(rename = "ngram-map-k")]
    NgramMapK,
    #[serde(rename = "ngram-map-k4v")]
    NgramMapK4v,
    #[serde(rename = "ngram-mod")]
    NgramMod,
    #[serde(rename = "ngram-cache")]
    NgramCache,
}

impl From<SpeculativeTypeName> for SpeculativeType {
    fn from(value: SpeculativeTypeName) -> Self {
        match value {
            SpeculativeTypeName::None => Self::None,
            SpeculativeTypeName::DraftSimple => Self::DraftSimple,
            SpeculativeTypeName::DraftEagle3 => Self::DraftEagle3,
            SpeculativeTypeName::DraftMtp => Self::DraftMtp,
            SpeculativeTypeName::DraftDflash => Self::DraftDflash,
            SpeculativeTypeName::DraftDspark => Self::DraftDspark,
            SpeculativeTypeName::NgramSimple => Self::NgramSimple,
            SpeculativeTypeName::NgramMapK => Self::NgramMapK,
            SpeculativeTypeName::NgramMapK4v => Self::NgramMapK4v,
            SpeculativeTypeName::NgramMod => Self::NgramMod,
            SpeculativeTypeName::NgramCache => Self::NgramCache,
        }
    }
}

impl core::fmt::Display for SpeculativeTypeName {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let type_id: SpeculativeType = (*self).into();
        formatter.write_str(type_id.llama_name())
    }
}

/// `value` was not one of llama's own `--spec-type` strings.
#[derive(Debug, Error)]
#[error("unknown speculative type: {0}")]
pub struct InvalidSpeculativeType(String);

impl core::str::FromStr for SpeculativeTypeName {
    type Err = InvalidSpeculativeType;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "draft-simple" => Ok(Self::DraftSimple),
            "draft-eagle3" => Ok(Self::DraftEagle3),
            "draft-mtp" => Ok(Self::DraftMtp),
            "draft-dflash" => Ok(Self::DraftDflash),
            "draft-dspark" => Ok(Self::DraftDspark),
            "ngram-simple" => Ok(Self::NgramSimple),
            "ngram-map-k" => Ok(Self::NgramMapK),
            "ngram-map-k4v" => Ok(Self::NgramMapK4v),
            "ngram-mod" => Ok(Self::NgramMod),
            "ngram-cache" => Ok(Self::NgramCache),
            other => Err(InvalidSpeculativeType(other.to_string())),
        }
    }
}

/// This crate's own priority-order table restated over
/// [`SpeculativeTypeName`] (mirrors
/// [`crate::serving::SpeculativeTypeSet`]'s private `PRIORITY_ORDER` one to
/// one) -- used only to give [`SpeculativeTypeNameSet::iter`] and its
/// `Display` a deterministic member order; llama's own set semantics never
/// depend on registration order (that type's own doc).
const PRIORITY_ORDER: [SpeculativeTypeName; 10] = [
    SpeculativeTypeName::NgramSimple,
    SpeculativeTypeName::NgramMapK,
    SpeculativeTypeName::NgramMapK4v,
    SpeculativeTypeName::NgramMod,
    SpeculativeTypeName::NgramCache,
    SpeculativeTypeName::DraftSimple,
    SpeculativeTypeName::DraftEagle3,
    SpeculativeTypeName::DraftMtp,
    SpeculativeTypeName::DraftDflash,
    SpeculativeTypeName::DraftDspark,
];

/// llama's `std::vector<common_speculative_type> types` (`common/common.h:373`),
/// serde/`FromStr`/`Display`-facing mirror of
/// [`crate::serving::SpeculativeTypeSet`] (kept separate for the same
/// no_std-isolation reason [`SpeculativeTypeName`] is its own mirror of
/// [`SpeculativeType`]). Round-trips llama's own `--spec-type`
/// comma-separated string form (`common/arg.cpp`'s own handler:
/// `string_split<std::string>(value, ',')` then
/// `common_speculative_types_from_names`) through [`core::str::FromStr`] and
/// [`core::fmt::Display`] -- the SAME string form both the env loader
/// (`parse_speculative_type_set`, below) and TOML (`Serialize`/`Deserialize`,
/// via that `Display`/`FromStr` pair) accept, matching this crate's own
/// convention of one textual form per setting rather than a second TOML-only
/// shape. The empty set displays as llama's own default string, `"none"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpeculativeTypeNameSet(u16);

impl SpeculativeTypeNameSet {
    /// llama's own default: `types = { COMMON_SPECULATIVE_TYPE_NONE }` --
    /// no speculator enabled.
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// This set with `name` added, llama's own `types.push_back` --
    /// `SpeculativeTypeName::None` is a no-op (llama's own `switch` in
    /// `common_speculative_init` never adds an implementation for it).
    #[must_use]
    pub fn insert(self, name: SpeculativeTypeName) -> Self {
        if matches!(name, SpeculativeTypeName::None) {
            return self;
        }
        let type_id: SpeculativeType = name.into();
        Self(self.0 | (1u16 << type_id as u16))
    }

    /// Whether `name` is one of this set's enabled speculators.
    #[must_use]
    pub fn contains(self, name: SpeculativeTypeName) -> bool {
        if matches!(name, SpeculativeTypeName::None) {
            return false;
        }
        let type_id: SpeculativeType = name.into();
        self.0 & (1u16 << type_id as u16) != 0
    }

    /// This crate's shipped default: `ngram-simple` alone, mirroring
    /// [`crate::SpeculativeConfig::ngram_simple`]. [`Self::empty`] (`none`)
    /// is the off switch.
    #[must_use]
    pub fn ngram_simple() -> Self {
        Self::empty().insert(SpeculativeTypeName::NgramSimple)
    }

    /// No speculator enabled.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// This set's members in llama's own fixed priority order
    /// ([`PRIORITY_ORDER`]'s own doc).
    pub fn iter(self) -> impl Iterator<Item = SpeculativeTypeName> {
        PRIORITY_ORDER.into_iter().filter(move |&name| self.contains(name))
    }
}

impl core::fmt::Display for SpeculativeTypeNameSet {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_empty() {
            return formatter.write_str("none");
        }
        for (index, name) in self.iter().enumerate() {
            if index > 0 {
                formatter.write_str(",")?;
            }
            core::fmt::Display::fmt(&name, formatter)?;
        }
        Ok(())
    }
}

impl core::str::FromStr for SpeculativeTypeNameSet {
    type Err = InvalidSpeculativeType;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut set = Self::empty();
        for part in value.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            set = set.insert(part.parse()?);
        }
        Ok(set)
    }
}

impl Serialize for SpeculativeTypeNameSet {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for SpeculativeTypeNameSet {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        raw.parse().map_err(D::Error::custom)
    }
}

fn parse_speculative_type_set(
    value: &str,
) -> Result<SpeculativeTypeNameSet, InvalidSpeculativeType> {
    value.parse()
}

/// The conflaguration/`bon` mirror of [`crate::SpeculativeConfig`] --
/// llama's `common_params_speculative` (`common/common.h:372-389`), env/TOML
/// loadable and fluent-buildable. See the module doc for why this owns the
/// data separately from the no_std-safe, `Copy` struct
/// [`crate::ServingConfig::speculative`] actually reads at decode time.
#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "SPECULATIVE")]
#[builder(derive(Clone, Debug))]
pub struct SpeculativeSettings {
    /// llama's `types` -- see [`crate::SpeculativeConfig::speculative_types`]'s
    /// own doc for the set semantics this mirrors. Defaults to `ngram-simple`
    /// (speculation on); `none` -- builder `SpeculativeTypeNameSet::empty()`,
    /// TOML `speculative_types = "none"`, env `SPECULATIVE_SPECULATIVE_TYPES=none`
    /// -- turns it off.
    #[setting(resolve_with = "parse_speculative_type_set", default_str = "ngram-simple")]
    #[builder(default = SpeculativeTypeNameSet::ngram_simple())]
    pub speculative_types: SpeculativeTypeNameSet,

    /// llama's `common_params_speculative_draft::n_max`.
    #[setting(default = 3)]
    #[builder(default = 3)]
    pub n_max: i32,
    /// llama's `common_params_speculative_draft::n_min`.
    #[setting(default = 0)]
    #[builder(default = 0)]
    pub n_min: i32,
    /// llama's `common_params_speculative_draft::p_min`.
    #[setting(default = 0.0)]
    #[builder(default = 0.0)]
    pub p_min: f32,

    /// llama's `ngram_simple.size_n`.
    #[setting(default = 12)]
    #[builder(default = 12)]
    pub ngram_simple_size_n: u16,
    /// llama's `ngram_simple.size_m`.
    #[setting(default = 48)]
    #[builder(default = 48)]
    pub ngram_simple_size_m: u16,
    /// llama's `ngram_simple.min_hits`.
    #[setting(default = 1)]
    #[builder(default = 1)]
    pub ngram_simple_min_hits: u16,

    /// llama's `ngram_map_k.size_n`.
    #[setting(default = 12)]
    #[builder(default = 12)]
    pub ngram_map_k_size_n: u16,
    /// llama's `ngram_map_k.size_m`.
    #[setting(default = 48)]
    #[builder(default = 48)]
    pub ngram_map_k_size_m: u16,
    /// llama's `ngram_map_k.min_hits`.
    #[setting(default = 1)]
    #[builder(default = 1)]
    pub ngram_map_k_min_hits: u16,

    /// llama's `ngram_map_k4v.size_n`.
    #[setting(default = 12)]
    #[builder(default = 12)]
    pub ngram_map_k4v_size_n: u16,
    /// llama's `ngram_map_k4v.size_m`.
    #[setting(default = 48)]
    #[builder(default = 48)]
    pub ngram_map_k4v_size_m: u16,
    /// llama's `ngram_map_k4v.min_hits`.
    #[setting(default = 1)]
    #[builder(default = 1)]
    pub ngram_map_k4v_min_hits: u16,

    /// llama's `ngram_mod.n_match`.
    #[setting(default = 24)]
    #[builder(default = 24)]
    pub ngram_mod_n_match: u16,
    /// llama's `ngram_mod.n_max`.
    #[setting(default = 64)]
    #[builder(default = 64)]
    pub ngram_mod_n_max: u16,
    /// llama's `ngram_mod.n_min`.
    #[setting(default = 48)]
    #[builder(default = 48)]
    pub ngram_mod_n_min: u16,

    /// llama's `ngram_cache.lookup_cache_static` -- empty string means unset,
    /// matching upstream's own empty-`std::string` default.
    #[serde(default)]
    pub ngram_cache_lookup_static: Option<String>,
    /// llama's `ngram_cache.lookup_cache_dynamic`.
    #[serde(default)]
    pub ngram_cache_lookup_dynamic: Option<String>,
}

impl SpeculativeSettings {
    /// Borrows this settings value's own owned storage into the no_std-safe,
    /// `Copy` shape `ServingConfig::speculative` actually reads -- the same
    /// borrow relationship `ServingConfig::model_path` has to whatever owned
    /// path a caller keeps alive for `'model`.
    #[must_use]
    pub fn as_speculative_config(&self) -> SpeculativeConfig<'_> {
        let speculative_types = self
            .speculative_types
            .iter()
            .fold(SpeculativeTypeSet::empty(), |set, name| {
                set.insert(name.into())
            });
        SpeculativeConfig {
            speculative_types,
            n_max: self.n_max,
            n_min: self.n_min,
            p_min: self.p_min,
            ngram_simple: NgramMapParams {
                size_n: self.ngram_simple_size_n,
                size_m: self.ngram_simple_size_m,
                min_hits: self.ngram_simple_min_hits,
            },
            ngram_map_k: NgramMapParams {
                size_n: self.ngram_map_k_size_n,
                size_m: self.ngram_map_k_size_m,
                min_hits: self.ngram_map_k_min_hits,
            },
            ngram_map_k4v: NgramMapParams {
                size_n: self.ngram_map_k4v_size_n,
                size_m: self.ngram_map_k4v_size_m,
                min_hits: self.ngram_map_k4v_min_hits,
            },
            ngram_mod: NgramModParams {
                n_match: self.ngram_mod_n_match,
                n_max: self.ngram_mod_n_max,
                n_min: self.ngram_mod_n_min,
            },
            ngram_cache_lookup_static: self.ngram_cache_lookup_static.as_deref(),
            ngram_cache_lookup_dynamic: self.ngram_cache_lookup_dynamic.as_deref(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    /// R9: the fluent builder and the conflaguration loader (TOML file, then
    /// env vars) produce identical configs for the same values -- AC12. A
    /// genuine multi-type SET (`ngram-simple,ngram-map-k`), not a single
    /// active type, proving the set (not scalar) semantics round-trip
    /// through both loaders identically to the builder.
    #[test]
    fn speculative_config_builder_matches_loader() {
        let enabled_types = SpeculativeTypeNameSet::empty()
            .insert(SpeculativeTypeName::NgramSimple)
            .insert(SpeculativeTypeName::NgramMapK);
        let via_builder = SpeculativeSettings::builder()
            .speculative_types(enabled_types)
            .ngram_simple_size_n(16)
            .ngram_simple_size_m(32)
            .ngram_simple_min_hits(2)
            .build();

        let mut toml_file = NamedTempFile::with_suffix(".toml")
            .expect("create temp toml file");
        writeln!(
            toml_file,
            r#"
            speculative_types = "ngram-simple,ngram-map-k"
            n_max = 3
            n_min = 0
            p_min = 0.0
            ngram_simple_size_n = 16
            ngram_simple_size_m = 32
            ngram_simple_min_hits = 2
            ngram_map_k_size_n = 12
            ngram_map_k_size_m = 48
            ngram_map_k_min_hits = 1
            ngram_map_k4v_size_n = 12
            ngram_map_k4v_size_m = 48
            ngram_map_k4v_min_hits = 1
            ngram_mod_n_match = 24
            ngram_mod_n_max = 64
            ngram_mod_n_min = 48
            "#
        )
        .expect("write temp toml file");
        let via_file: SpeculativeSettings = conflaguration::from_file(toml_file.path())
            .unwrap_or_else(|err| panic!("from_file failed: {err}"));
        assert_eq!(via_builder, via_file, "builder must match the TOML loader");
        assert!(via_file.speculative_types.contains(SpeculativeTypeName::NgramSimple));
        assert!(via_file.speculative_types.contains(SpeculativeTypeName::NgramMapK));
        assert!(!via_file.speculative_types.contains(SpeculativeTypeName::NgramMod));

        temp_env::with_vars(
            [
                ("SPECULATIVE_SPECULATIVE_TYPES", Some("ngram-simple,ngram-map-k")),
                ("SPECULATIVE_NGRAM_SIMPLE_SIZE_N", Some("16")),
                ("SPECULATIVE_NGRAM_SIMPLE_SIZE_M", Some("32")),
                ("SPECULATIVE_NGRAM_SIMPLE_MIN_HITS", Some("2")),
            ],
            || {
                let via_env = SpeculativeSettings::from_env()
                    .unwrap_or_else(|err| panic!("from_env failed: {err}"));
                assert_eq!(via_builder, via_env, "builder must match the env loader");
            },
        );
    }

    /// Owner directive 2026-09-29: speculation is ON by default with the
    /// `ngram-simple` drafter; its per-type param defaults still match
    /// llama's `common_params_speculative` (`common/common.h:372-389`).
    #[test]
    fn default_speculative_settings_enable_ngram_simple_with_llama_defaults() {
        temp_env::with_vars(
            [
                ("SPECULATIVE_SPECULATIVE_TYPES", None::<&str>),
                ("SPECULATIVE_N_MAX", None::<&str>),
                ("SPECULATIVE_N_MIN", None::<&str>),
                ("SPECULATIVE_P_MIN", None::<&str>),
                ("SPECULATIVE_NGRAM_SIMPLE_SIZE_N", None::<&str>),
                ("SPECULATIVE_NGRAM_SIMPLE_SIZE_M", None::<&str>),
                ("SPECULATIVE_NGRAM_SIMPLE_MIN_HITS", None::<&str>),
                ("SPECULATIVE_NGRAM_MAP_K_SIZE_N", None::<&str>),
                ("SPECULATIVE_NGRAM_MAP_K_SIZE_M", None::<&str>),
                ("SPECULATIVE_NGRAM_MAP_K_MIN_HITS", None::<&str>),
                ("SPECULATIVE_NGRAM_MAP_K4V_SIZE_N", None::<&str>),
                ("SPECULATIVE_NGRAM_MAP_K4V_SIZE_M", None::<&str>),
                ("SPECULATIVE_NGRAM_MAP_K4V_MIN_HITS", None::<&str>),
                ("SPECULATIVE_NGRAM_MOD_N_MATCH", None::<&str>),
                ("SPECULATIVE_NGRAM_MOD_N_MAX", None::<&str>),
                ("SPECULATIVE_NGRAM_MOD_N_MIN", None::<&str>),
            ],
            || {
                let settings = SpeculativeSettings::from_env()
                    .unwrap_or_else(|err| panic!("from_env failed: {err}"));
                assert_eq!(settings.speculative_types, SpeculativeTypeNameSet::ngram_simple());
                assert!(settings.speculative_types.contains(SpeculativeTypeName::NgramSimple));
                assert_eq!(settings.n_max, 3);
                assert_eq!(settings.n_min, 0);
                assert_eq!(settings.p_min, 0.0);
                assert_eq!(settings.ngram_simple_size_n, 12);
                assert_eq!(settings.ngram_simple_size_m, 48);
                assert_eq!(settings.ngram_simple_min_hits, 1);
                assert_eq!(settings.ngram_mod_n_match, 24);
                assert_eq!(settings.ngram_mod_n_max, 64);
                assert_eq!(settings.ngram_mod_n_min, 48);
                assert!(settings.ngram_cache_lookup_static.is_none());
                assert!(settings.ngram_cache_lookup_dynamic.is_none());
            },
        );
    }

    const SPECULATIVE_ENV_KEYS: [&str; 16] = [
        "SPECULATIVE_SPECULATIVE_TYPES",
        "SPECULATIVE_N_MAX",
        "SPECULATIVE_N_MIN",
        "SPECULATIVE_P_MIN",
        "SPECULATIVE_NGRAM_SIMPLE_SIZE_N",
        "SPECULATIVE_NGRAM_SIMPLE_SIZE_M",
        "SPECULATIVE_NGRAM_SIMPLE_MIN_HITS",
        "SPECULATIVE_NGRAM_MAP_K_SIZE_N",
        "SPECULATIVE_NGRAM_MAP_K_SIZE_M",
        "SPECULATIVE_NGRAM_MAP_K_MIN_HITS",
        "SPECULATIVE_NGRAM_MAP_K4V_SIZE_N",
        "SPECULATIVE_NGRAM_MAP_K4V_SIZE_M",
        "SPECULATIVE_NGRAM_MAP_K4V_MIN_HITS",
        "SPECULATIVE_NGRAM_MOD_N_MATCH",
        "SPECULATIVE_NGRAM_MOD_N_MAX",
        "SPECULATIVE_NGRAM_MOD_N_MIN",
    ];

    fn env_with_types(types: Option<&'static str>) -> Vec<(&'static str, Option<&'static str>)> {
        SPECULATIVE_ENV_KEYS
            .into_iter()
            .map(|key| {
                if key == "SPECULATIVE_SPECULATIVE_TYPES" {
                    (key, types)
                } else {
                    (key, None)
                }
            })
            .collect()
    }

    fn toml_with_types(types: &str) -> NamedTempFile {
        let mut toml_file = NamedTempFile::with_suffix(".toml").expect("create temp toml file");
        writeln!(
            toml_file,
            r#"
            speculative_types = "{types}"
            n_max = 3
            n_min = 0
            p_min = 0.0
            ngram_simple_size_n = 12
            ngram_simple_size_m = 48
            ngram_simple_min_hits = 1
            ngram_map_k_size_n = 12
            ngram_map_k_size_m = 48
            ngram_map_k_min_hits = 1
            ngram_map_k4v_size_n = 12
            ngram_map_k4v_size_m = 48
            ngram_map_k4v_min_hits = 1
            ngram_mod_n_match = 24
            ngram_mod_n_max = 64
            ngram_mod_n_min = 48
            "#
        )
        .expect("write temp toml file");
        toml_file
    }

    fn off_settings() -> SpeculativeSettings {
        SpeculativeSettings::builder()
            .speculative_types(SpeculativeTypeNameSet::empty())
            .build()
    }

    #[test]
    fn builder_default_matches_env_loader_default_and_is_on() {
        let via_builder = SpeculativeSettings::builder().build();

        temp_env::with_vars(env_with_types(None), || {
            let via_env = SpeculativeSettings::from_env()
                .unwrap_or_else(|err| panic!("from_env failed: {err}"));
            assert_eq!(via_builder, via_env, "builder default must match the env loader default");
        });
        assert!(via_builder.speculative_types.contains(SpeculativeTypeName::NgramSimple));
    }

    #[test]
    fn default_settings_lower_to_the_same_config_as_serving_config_default() {
        let settings = SpeculativeSettings::builder().build();

        assert_eq!(settings.as_speculative_config(), SpeculativeConfig::default());
        assert_eq!(
            settings.as_speculative_config(),
            crate::serving::ServingConfig::default().speculative
        );
        assert!(
            SpeculativeConfig::default()
                .speculative_types
                .contains(SpeculativeType::NgramSimple)
        );
    }

    #[test]
    fn builder_none_turns_speculation_off() {
        let config_owner = off_settings();
        let config = config_owner.as_speculative_config();

        assert!(config_owner.speculative_types.is_empty());
        assert!(config.speculative_types.is_empty());
        assert_eq!(config, SpeculativeConfig::none());
    }

    #[test]
    fn toml_none_turns_speculation_off_and_matches_builder() {
        let toml_file = toml_with_types("none");

        let via_file: SpeculativeSettings = conflaguration::from_file(toml_file.path())
            .unwrap_or_else(|err| panic!("from_file failed: {err}"));

        assert_eq!(via_file, off_settings());
        assert_eq!(via_file.as_speculative_config(), SpeculativeConfig::none());
    }

    #[test]
    fn toml_ngram_simple_matches_the_default_builder() {
        let toml_file = toml_with_types("ngram-simple");

        let via_file: SpeculativeSettings = conflaguration::from_file(toml_file.path())
            .unwrap_or_else(|err| panic!("from_file failed: {err}"));

        assert_eq!(via_file, SpeculativeSettings::builder().build());
    }

    #[test]
    fn env_none_turns_speculation_off_and_matches_builder() {
        temp_env::with_vars(env_with_types(Some("none")), || {
            let via_env = SpeculativeSettings::from_env()
                .unwrap_or_else(|err| panic!("from_env failed: {err}"));

            assert_eq!(via_env, off_settings());
            assert_eq!(via_env.as_speculative_config(), SpeculativeConfig::none());
        });
    }

    #[test]
    fn env_ngram_simple_matches_the_default_builder() {
        temp_env::with_vars(env_with_types(Some("ngram-simple")), || {
            let via_env = SpeculativeSettings::from_env()
                .unwrap_or_else(|err| panic!("from_env failed: {err}"));

            assert_eq!(via_env, SpeculativeSettings::builder().build());
        });
    }

    #[test]
    fn env_unknown_type_is_rejected_not_silently_defaulted() {
        temp_env::with_vars(env_with_types(Some("quantum-mtp")), || {
            let result = SpeculativeSettings::from_env();

            assert!(result.is_err(), "an unknown type must not fall back to a default");
        });
    }

    /// R9: llama's own `--spec-type` strings round-trip through
    /// [`SpeculativeTypeName`]'s `FromStr`/`Display`.
    #[test]
    fn llama_type_names_round_trip() {
        let names = [
            "none",
            "draft-simple",
            "draft-eagle3",
            "draft-mtp",
            "draft-dflash",
            "draft-dspark",
            "ngram-simple",
            "ngram-map-k",
            "ngram-map-k4v",
            "ngram-mod",
            "ngram-cache",
        ];
        for name in names {
            let parsed: SpeculativeTypeName = name
                .parse()
                .unwrap_or_else(|err| panic!("{name} must parse: {err}"));
            assert_eq!(parsed.to_string(), name, "round trip for {name}");
        }
    }

    #[test]
    fn unknown_type_name_is_rejected() {
        let result: Result<SpeculativeTypeName, _> = "quantum-mtp".parse();
        assert!(result.is_err());
    }

    /// R9: llama's own multi-type `--spec-type` string form
    /// (`ngram-simple,ngram-map-k,ngram-mod`) round-trips through
    /// [`SpeculativeTypeNameSet`]'s `FromStr`/`Display` -- proving the SET
    /// (not a single active type) is what this crate's config carries, and
    /// that member order in the string never changes the parsed set.
    #[test]
    fn llama_type_set_round_trips_multi_type_string() {
        let parsed: SpeculativeTypeNameSet = "ngram-simple,ngram-map-k,ngram-mod"
            .parse()
            .expect("multi-type string parses");
        assert!(parsed.contains(SpeculativeTypeName::NgramSimple));
        assert!(parsed.contains(SpeculativeTypeName::NgramMapK));
        assert!(parsed.contains(SpeculativeTypeName::NgramMod));
        assert!(!parsed.contains(SpeculativeTypeName::NgramCache));
        // llama's own priority order (`common/speculative.cpp:2617-2629`),
        // not the string's own member order (mod was listed last above).
        assert_eq!(parsed.to_string(), "ngram-simple,ngram-map-k,ngram-mod");

        // registration order in the string never matters (this crate's own
        // set semantics, mirroring `common_get_enabled_speculative_configs`
        // folding the caller's list into a bitset before priority order is
        // walked) -- a differently-ordered string parses to the identical set.
        let reordered: SpeculativeTypeNameSet = "ngram-mod,ngram-simple,ngram-map-k"
            .parse()
            .expect("reordered multi-type string parses");
        assert_eq!(parsed, reordered);

        let empty: SpeculativeTypeNameSet = "none".parse().expect("none parses");
        assert!(empty.is_empty());
        assert_eq!(empty.to_string(), "none");
    }

    #[test]
    fn unknown_member_in_type_set_is_rejected() {
        let result: Result<SpeculativeTypeNameSet, _> = "ngram-simple,quantum-mtp".parse();
        assert!(result.is_err());
    }

    /// `as_speculative_config` borrows this settings value's own storage --
    /// the ngram-cache path fields round-trip through the borrow.
    #[test]
    fn as_speculative_config_borrows_cache_paths() {
        let settings = SpeculativeSettings::builder()
            .speculative_types(SpeculativeTypeNameSet::empty().insert(SpeculativeTypeName::NgramCache))
            .ngram_cache_lookup_static("static.cache".to_string())
            .ngram_cache_lookup_dynamic("dynamic.cache".to_string())
            .build();
        let config = settings.as_speculative_config();
        assert!(config.speculative_types.contains(SpeculativeType::NgramCache));
        assert_eq!(config.ngram_cache_lookup_static, Some("static.cache"));
        assert_eq!(config.ngram_cache_lookup_dynamic, Some("dynamic.cache"));
    }
}
