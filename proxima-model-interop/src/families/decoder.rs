//! The decoder-only family: one TOML of tensor-name roles per checkpoint
//! dialect, expanded into [`FamilyTemplate`]s for [`infer`](super::infer), and a
//! generic [`infer_profile`] that fills every [`FamilyProfile`] field a file
//! states (PROXNUM-611).
//!
//! Teaching pointer: this composes [`FamilyTemplate`] probes (tensor presence)
//! and the `config` JSON the inferrer already takes (GGUF `{arch}.*` keys with
//! the prefix stripped by [`gguf_config`], or an HF `config.json`). No name of a
//! model appears here: the architecture string is only a key prefix.
//!
//! A field no file states is a [`Slot::Unresolved`], and one that does not
//! exist for this checkpoint (a routing bias on a dense model) is
//! [`Slot::NotApplicable`]. [`ProfileFacts::unresolved`] is the list a TOML
//! fact or a probe must settle. `ple` and `dense_feed_forward` are not here:
//! the descriptor builders own them per layer.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_tensor::spec::{
    Activation, EmbeddingScale, ExpertGatingFunc, FfnCombination, ParallelDenseMoeConfig,
    RecurrenceProfile, RopeLayout, ScheduleSource, expert_gating_func,
};
use serde::Deserialize;
use serde_json::{Value as Json, json};
use thiserror::Error;

use super::infer::Inferred;
use super::pattern::Pattern;
use super::template::{FamilyTemplate, Probe, Stack, TemplateError, Value};

const ROLE: &str = "block";

/// One checkpoint format's spelling of the family: where the layer stack lives,
/// the non-layer tensors it carries, and its role and metadata names.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Dialect {
    pub layer: Pattern,
    #[serde(default)]
    pub claims: Vec<String>,
    #[serde(default)]
    pub claim_prefixes: Vec<String>,
    #[serde(default)]
    pub roles: BTreeMap<String, String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

/// The decoder-only family as data: shared identity plus a table of dialects.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DecoderFamily {
    pub name: String,
    pub required_stages: Vec<String>,
    #[serde(default)]
    pub requires: Vec<String>,
    pub dialects: BTreeMap<String, Dialect>,
}

/// Why the profile could not be read from a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProfileInferError {
    #[error("inferred family {family:?} names no dialect of this decoder family")]
    UnknownDialect { family: String },
    #[error("expert_gating_func {value} has no routing function")]
    Gating { value: u32 },
}

/// A profile field's state after inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Slot<T> {
    Known(T),
    /// The field has no meaning for this checkpoint.
    NotApplicable,
    /// The file states nothing about it.
    Unresolved,
}

impl<T> Slot<T> {
    fn from_option(found: Option<T>) -> Self {
        found.map_or(Self::Unresolved, Self::Known)
    }

    #[must_use]
    pub fn is_unresolved(&self) -> bool {
        matches!(self, Self::Unresolved)
    }
}

/// Every [`FamilyProfile`](proxima_tensor::spec::FamilyProfile) field a file can
/// state, each a [`Slot`]. `layer_combination` is `profile.layer_ffn(experts)`'s
/// combination, which is where the parallel dense+MoE payload lives.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileFacts {
    pub embedding_scale: Slot<Option<EmbeddingScale>>,
    pub post_attention_norm: Slot<bool>,
    pub layer_combination: Slot<FfnCombination>,
    pub output_scale: Slot<bool>,
    pub routed_gating: Slot<ExpertGatingFunc>,
    pub routed_expert_bias: Slot<bool>,
    pub activation: Slot<Activation>,
    pub exclusive_dense_post_norm: Slot<bool>,
    pub shared_expert_gate: Slot<bool>,
    pub score_scale_inverse_sqrt_head_dim: Slot<bool>,
    pub value_norm: Slot<bool>,
    pub qk_norm: Slot<bool>,
    pub rope_layout: Slot<RopeLayout>,
    pub recurrence: Slot<Option<RecurrenceProfile>>,
    pub schedule_source: Slot<ScheduleSource>,
    pub ssm_v_head_reordered: Slot<bool>,
}

impl ProfileFacts {
    /// The names of the fields the file left unresolved, in field order.
    #[must_use]
    pub fn unresolved(&self) -> Vec<&'static str> {
        let table = [
            ("embedding_scale", self.embedding_scale.is_unresolved()),
            (
                "post_attention_norm",
                self.post_attention_norm.is_unresolved(),
            ),
            ("layer_combination", self.layer_combination.is_unresolved()),
            ("output_scale", self.output_scale.is_unresolved()),
            ("routed_gating", self.routed_gating.is_unresolved()),
            (
                "routed_expert_bias",
                self.routed_expert_bias.is_unresolved(),
            ),
            ("activation", self.activation.is_unresolved()),
            (
                "exclusive_dense_post_norm",
                self.exclusive_dense_post_norm.is_unresolved(),
            ),
            (
                "shared_expert_gate",
                self.shared_expert_gate.is_unresolved(),
            ),
            (
                "score_scale_inverse_sqrt_head_dim",
                self.score_scale_inverse_sqrt_head_dim.is_unresolved(),
            ),
            ("value_norm", self.value_norm.is_unresolved()),
            ("qk_norm", self.qk_norm.is_unresolved()),
            ("rope_layout", self.rope_layout.is_unresolved()),
            ("recurrence", self.recurrence.is_unresolved()),
            ("schedule_source", self.schedule_source.is_unresolved()),
            (
                "ssm_v_head_reordered",
                self.ssm_v_head_reordered.is_unresolved(),
            ),
        ];
        table
            .into_iter()
            .filter_map(|(name, open)| open.then_some(name))
            .collect()
    }
}

impl DecoderFamily {
    /// The embedded decoder-only family.
    ///
    /// # Errors
    ///
    /// [`TemplateError::Parse`] when the embedded TOML does not parse.
    #[cfg(feature = "toml")]
    #[must_use = "a refused family must be handled"]
    pub fn builtin() -> Result<Self, TemplateError> {
        toml::from_str(include_str!("decoder_only.toml")).map_err(|error| TemplateError::Parse {
            reason: error.to_string(),
        })
    }

    /// One [`FamilyTemplate`] per dialect, named `{family}.{dialect}`, every
    /// role a probe that asks "in any layer".
    ///
    /// # Errors
    ///
    /// A [`TemplateError`] when a role suffix makes a bad pattern or a template
    /// fails [`FamilyTemplate::validate`].
    #[must_use = "a refused family must be handled"]
    pub fn templates(&self) -> Result<Vec<FamilyTemplate>, TemplateError> {
        self.dialects
            .iter()
            .map(|(dialect, spelling)| self.template(dialect, spelling))
            .collect()
    }

    fn template(&self, dialect: &str, spelling: &Dialect) -> Result<FamilyTemplate, TemplateError> {
        let probes = spelling
            .roles
            .iter()
            .map(|(key, suffix)| role_probe(&spelling.layer, key, suffix))
            .collect::<Result<Vec<_>, _>>()?;
        FamilyTemplate {
            name: format!("{}.{dialect}", self.name),
            stacks: [Stack {
                role: ROLE.to_string(),
                pattern: spelling.layer.clone(),
            }]
            .into(),
            claims: spelling.claims.clone(),
            claim_prefixes: spelling.claim_prefixes.clone(),
            probes,
            requires: self.requires.clone(),
            forbids: Vec::new(),
            dims: Vec::new(),
            knobs: Vec::new(),
            required_stages: self.required_stages.clone(),
        }
        .validate()
    }

    fn dialect_of(&self, inferred: &Inferred) -> Result<&Dialect, ProfileInferError> {
        inferred
            .family
            .strip_prefix(self.name.as_str())
            .and_then(|rest| rest.strip_prefix('.'))
            .and_then(|dialect| self.dialects.get(dialect))
            .ok_or_else(|| ProfileInferError::UnknownDialect {
                family: inferred.family.clone(),
            })
    }
}

fn role_probe(layer: &Pattern, key: &str, suffix: &str) -> Result<Probe, TemplateError> {
    let pattern =
        Pattern::parse(&format!("{layer}{suffix}")).map_err(|error| TemplateError::Parse {
            reason: error.to_string(),
        })?;
    Ok(Probe {
        key: key.to_string(),
        role: ROLE.to_string(),
        pattern,
        any: true,
    })
}

struct Reading<'file> {
    inferred: &'file Inferred,
    dialect: &'file Dialect,
    config: &'file Json,
}

/// A `post_attention_norm` tensor is a norm after attention only where the FFN has its own
/// `ffn_norm`; a hybrid layer with no `ffn_norm` uses that tensor as the FFN's own pre-norm.
fn post_attention_norm(reading: &Reading<'_>) -> Option<bool> {
    let present = reading.flag("post_attention_norm")?;
    Some(present && reading.flag("ffn_norm").unwrap_or(true))
}

impl Reading<'_> {
    fn flag(&self, role: &str) -> Option<bool> {
        match self.inferred.facts.get(role)?.value {
            Value::Flag(present) => Some(present),
            _ => None,
        }
    }

    fn metadata(&self, name: &str) -> Option<&Json> {
        self.config.get(self.dialect.metadata.get(name)?)
    }

    fn has_metadata_key(&self, name: &str) -> Option<bool> {
        let key = self.dialect.metadata.get(name)?;
        Some(self.config.get(key).is_some())
    }
}

/// Fills a [`ProfileFacts`] from what `inferred` and `config` state.
///
/// # Errors
///
/// [`ProfileInferError::UnknownDialect`] when `inferred` is not this family's;
/// [`ProfileInferError::Gating`] when the header names a routing function this
/// engine has no kernel for (serving the default instead would route wrongly).
#[must_use = "a refused profile must be handled"]
pub fn infer_profile(
    family: &DecoderFamily,
    inferred: &Inferred,
    config: &Json,
) -> Result<ProfileFacts, ProfileInferError> {
    let reading = Reading {
        inferred,
        dialect: family.dialect_of(inferred)?,
        config,
    };
    let layer_combination = combination(&reading);
    Ok(ProfileFacts {
        embedding_scale: embedding_scale(&reading),
        post_attention_norm: Slot::from_option(post_attention_norm(&reading)),
        shared_expert_gate: shared_gate(&reading, layer_combination),
        layer_combination,
        output_scale: Slot::from_option(reading.flag("output_scale")),
        routed_gating: routed_gating(&reading)?,
        routed_expert_bias: routed_bias(&reading),
        activation: activation(&reading),
        exclusive_dense_post_norm: Slot::from_option(dense_post_norm(&reading)),
        score_scale_inverse_sqrt_head_dim: Slot::Unresolved,
        value_norm: Slot::Unresolved,
        qk_norm: Slot::from_option(reading.flag("qk_norm")),
        rope_layout: Slot::Unresolved,
        recurrence: recurrence(&reading),
        schedule_source: schedule_source(&reading),
        ssm_v_head_reordered: ssm_order(&reading),
    })
}

fn embedding_scale(reading: &Reading) -> Slot<Option<EmbeddingScale>> {
    match reading.metadata("embedding_scale").and_then(Json::as_f64) {
        Some(factor) if factor != 0.0 => Slot::Known(Some(EmbeddingScale::Factor(factor as f32))),
        _ => Slot::Unresolved,
    }
}

fn all_present(reading: &Reading, roles: &[&str]) -> Option<bool> {
    let flags: Option<Vec<bool>> = roles.iter().map(|role| reading.flag(role)).collect();
    flags.map(|found| found.into_iter().all(|present| present))
}

fn combination(reading: &Reading) -> Slot<FfnCombination> {
    let shared = all_present(reading, &["shexp_gate", "shexp_up", "shexp_down"]);
    if shared == Some(true) {
        return Slot::Known(FfnCombination::RoutedWithSharedExpert);
    }
    match all_present(reading, &["post_ffw_norm_1", "post_ffw_norm_2"]) {
        Some(true) => parallel(reading).map_or(Slot::Unresolved, |config| {
            Slot::Known(FfnCombination::ParallelDenseMoe(config))
        }),
        Some(false) if shared == Some(false) => Slot::Known(FfnCombination::Exclusive),
        _ => Slot::Unresolved,
    }
}

fn parallel(reading: &Reading) -> Option<ParallelDenseMoeConfig> {
    Some(ParallelDenseMoeConfig {
        dense_post_norm: reading.flag("post_ffw_norm_1")?,
        routed_post_norm: reading.flag("post_ffw_norm_2")?,
        combined_post_norm: reading.flag("post_ffw_norm")?,
        routed_pre_norm: reading.flag("pre_ffw_norm_2")?,
        router_scale: reading.flag("router_scale")?,
        expert_output_scale: reading.flag("expert_output_scale")?,
    })
}

fn shared_gate(reading: &Reading, combination: Slot<FfnCombination>) -> Slot<bool> {
    match combination {
        Slot::Known(FfnCombination::RoutedWithSharedExpert) => {
            Slot::from_option(reading.flag("shexp_router"))
        }
        Slot::Unresolved => Slot::Unresolved,
        _ => Slot::NotApplicable,
    }
}

fn routed_gating(reading: &Reading) -> Result<Slot<ExpertGatingFunc>, ProfileInferError> {
    match reading.flag("router") {
        None => return Ok(Slot::Unresolved),
        Some(false) => return Ok(Slot::NotApplicable),
        Some(true) => {}
    }
    let Some(code) = reading
        .metadata("expert_gating_func")
        .and_then(Json::as_u64)
        .and_then(|raw| u32::try_from(raw).ok())
    else {
        return Ok(Slot::Unresolved);
    };
    let function =
        expert_gating_func(code).map_err(|_| ProfileInferError::Gating { value: code })?;
    Ok(Slot::from_option(function))
}

fn routed_bias(reading: &Reading) -> Slot<bool> {
    match reading.flag("router") {
        None => Slot::Unresolved,
        Some(false) => Slot::NotApplicable,
        Some(true) => Slot::from_option(reading.flag("expert_bias")),
    }
}

fn activation(reading: &Reading) -> Slot<Activation> {
    match reading.metadata("activation").and_then(Json::as_str) {
        Some("silu") => Slot::Known(Activation::Silu),
        Some("gelu_pytorch_tanh") => Slot::Known(Activation::GeluTanh),
        _ => Slot::Unresolved,
    }
}

fn dense_post_norm(reading: &Reading) -> Option<bool> {
    let combined = reading.flag("post_ffw_norm")?;
    Some(combined && !reading.flag("post_ffw_norm_1")? && !reading.flag("post_ffw_norm_2")?)
}

fn recurrence(reading: &Reading) -> Slot<Option<RecurrenceProfile>> {
    match reading.has_metadata_key("loop_count") {
        Some(false) => Slot::Known(None),
        _ => Slot::Unresolved,
    }
}

fn schedule_source(reading: &Reading) -> Slot<ScheduleSource> {
    let (Some(_), Some(interval)) = (
        reading.has_metadata_key("sliding_window_pattern"),
        reading.has_metadata_key("full_attention_interval"),
    ) else {
        return Slot::Unresolved;
    };
    let sliding = matches!(
        reading.metadata("sliding_window_pattern"),
        Some(Json::Array(_))
    );
    match (sliding, interval, reading.flag("router")) {
        (true, _, _) => Slot::Known(ScheduleSource::SlidingPattern),
        (false, true, Some(true)) => Slot::Known(ScheduleSource::RecurrentRoutedInterval),
        (false, true, Some(false)) => Slot::Known(ScheduleSource::RecurrentInterval),
        (false, false, _) => Slot::Known(ScheduleSource::Uniform),
        (false, true, None) => Slot::Unresolved,
    }
}

fn ssm_order(reading: &Reading) -> Slot<bool> {
    match reading.flag("ssm_state") {
        Some(false) => Slot::NotApplicable,
        _ => Slot::from_option(
            reading
                .metadata("ssm_v_head_reordered")
                .and_then(Json::as_bool),
        ),
    }
}

/// The `{arch}.*` metadata of `parsed` as a JSON object with the prefix
/// stripped, and `general.architecture` as `None` when the header has none.
/// The architecture string is only a key prefix: nothing branches on it.
#[must_use]
pub fn gguf_config(parsed: &ParsedGguf) -> Json {
    let Some(MetadataValue::String(arch)) = parsed.metadata_value("general.architecture") else {
        return json!({});
    };
    let prefix = format!("{arch}.");
    let entries = parsed.metadata.iter().filter_map(|(key, value)| {
        let name = key.strip_prefix(prefix.as_str())?;
        Some((name.to_string(), metadata_json(value)))
    });
    Json::Object(entries.collect())
}

fn metadata_json(value: &MetadataValue) -> Json {
    match value {
        MetadataValue::U8(raw) => json!(raw),
        MetadataValue::I8(raw) => json!(raw),
        MetadataValue::U16(raw) => json!(raw),
        MetadataValue::I16(raw) => json!(raw),
        MetadataValue::U32(raw) => json!(raw),
        MetadataValue::I32(raw) => json!(raw),
        MetadataValue::F32(raw) => json!(raw),
        MetadataValue::Bool(raw) => json!(raw),
        MetadataValue::String(raw) => json!(raw),
        MetadataValue::U64(raw) => json!(raw),
        MetadataValue::I64(raw) => json!(raw),
        MetadataValue::F64(raw) => json!(raw),
        MetadataValue::Array(array) => array_json(array),
    }
}

fn array_json(array: &MetadataArray) -> Json {
    match array {
        MetadataArray::U8(raw) => json!(raw),
        MetadataArray::I8(raw) => json!(raw),
        MetadataArray::U16(raw) => json!(raw),
        MetadataArray::I16(raw) => json!(raw),
        MetadataArray::U32(raw) => json!(raw),
        MetadataArray::I32(raw) => json!(raw),
        MetadataArray::F32(raw) => json!(raw),
        MetadataArray::Bool(raw) => json!(raw),
        MetadataArray::String(raw) => json!(raw),
        MetadataArray::U64(raw) => json!(raw),
        MetadataArray::I64(raw) => json!(raw),
        MetadataArray::F64(raw) => json!(raw),
    }
}
