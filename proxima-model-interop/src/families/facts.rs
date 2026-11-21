//! Architecture facts: the values no checkpoint states, as TOML data keyed by
//! the architecture string a file declares (PROXNUM-611).
//!
//! Teaching pointer: [`infer_profile`](super::infer_profile) fills every
//! [`ProfileFacts`] slot a file states; [`ProfileFacts::settle`] fills the
//! slots left [`Slot::Unresolved`] from one [`ArchFacts`] section, and
//! [`ProfileFacts::into_profile`] builds the [`FamilyProfile`] the serving path
//! reads. A fact never overrides the file: one that disagrees with a
//! [`Slot::Known`] value is a [`FactsError::Contradicts`]. A slot still open
//! after the facts is a [`FactsError::Unresolved`] naming the architecture,
//! never a default. The loader parses every section into a map and names no
//! model.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use core::fmt::Debug;

use proxima_tensor::spec::{
    Activation, EmbeddingScale, ExpertGatingFunc, FamilyProfile, FfnCombination, LayerFfnConfig,
    RecurrenceProfile, RopeLayout,
};
use serde::Deserialize;
use thiserror::Error;

use super::decoder::{ProfileFacts, Slot};

/// What a TOML section says an architecture's embedding scale is. The explicit
/// `None` is what lets a section state "no scale" for a header that omits the key.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub enum EmbeddingFact {
    None,
    Sqrt,
    Factor(f32),
}

impl EmbeddingFact {
    fn scale(self) -> Option<EmbeddingScale> {
        match self {
            Self::None => None,
            Self::Sqrt => Some(EmbeddingScale::Sqrt),
            Self::Factor(factor) => Some(EmbeddingScale::Factor(factor)),
        }
    }
}

/// One architecture's section of `facts.toml`: only fields a file cannot state.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchFacts {
    pub activation: Option<Activation>,
    pub rope_layout: Option<RopeLayout>,
    pub value_norm: Option<bool>,
    pub score_scale_inverse_sqrt_head_dim: Option<bool>,
    pub embedding_scale: Option<EmbeddingFact>,
    pub routed_gating: Option<ExpertGatingFunc>,
    pub recurrence: Option<RecurrenceProfile>,
    pub ssm_v_head_reordered: Option<bool>,
}

impl ArchFacts {
    /// How many fields this section sets.
    #[must_use]
    pub fn field_count(&self) -> usize {
        [
            self.activation.is_some(),
            self.rope_layout.is_some(),
            self.value_norm.is_some(),
            self.score_scale_inverse_sqrt_head_dim.is_some(),
            self.embedding_scale.is_some(),
            self.routed_gating.is_some(),
            self.recurrence.is_some(),
            self.ssm_v_head_reordered.is_some(),
        ]
        .into_iter()
        .filter(|set| *set)
        .count()
    }
}

/// Why facts could not complete a profile.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum FactsError {
    #[error("facts table does not parse: {reason}")]
    Parse { reason: String },
    #[error(
        "architecture {architecture:?}: facts say {field} = {fact} but the file shows {file}"
    )]
    Contradicts {
        architecture: String,
        field: &'static str,
        fact: String,
        file: String,
    },
    #[error("architecture {architecture:?}: field(s) neither the file nor the facts state: {fields}")]
    Unresolved {
        architecture: String,
        fields: String,
    },
}

/// Every architecture's [`ArchFacts`], parsed from one TOML document.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(transparent)]
pub struct FactsTable {
    sections: BTreeMap<String, ArchFacts>,
}

impl FactsTable {
    /// Parses a facts document: one `[architecture]` section per key.
    ///
    /// # Errors
    ///
    /// [`FactsError::Parse`] for bad TOML, an unknown field or a bad value.
    #[cfg(feature = "toml")]
    #[must_use = "a refused facts table must be handled"]
    pub fn parse(text: &str) -> Result<Self, FactsError> {
        toml::from_str(text).map_err(|error| FactsError::Parse {
            reason: error.to_string(),
        })
    }

    /// The table embedded in this crate.
    ///
    /// # Errors
    ///
    /// [`FactsError::Parse`] when the embedded document does not parse.
    #[cfg(feature = "toml")]
    #[must_use = "a refused facts table must be handled"]
    pub fn builtin() -> Result<Self, FactsError> {
        Self::parse(include_str!("facts.toml"))
    }

    #[must_use]
    pub fn get(&self, architecture: &str) -> Option<&ArchFacts> {
        self.sections.get(architecture)
    }

    /// Every architecture the table has a section for, with its section.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ArchFacts)> {
        self.sections.iter().map(|(name, facts)| (name.as_str(), facts))
    }
}

fn settle_one<T: PartialEq + Debug>(
    slot: Slot<T>,
    fact: Option<T>,
    field: &'static str,
    architecture: &str,
) -> Result<Slot<T>, FactsError> {
    match (slot, fact) {
        (Slot::Known(file), Some(stated)) if file != stated => Err(FactsError::Contradicts {
            architecture: architecture.to_string(),
            field,
            fact: format!("{stated:?}"),
            file: format!("{file:?}"),
        }),
        (Slot::Unresolved, Some(stated)) => Ok(Slot::Known(stated)),
        (other, _) => Ok(other),
    }
}

impl ProfileFacts {
    /// Fills the open slots `facts` states; refuses one that contradicts a slot
    /// the file already fixed.
    ///
    /// # Errors
    ///
    /// [`FactsError::Contradicts`] naming the field and both values.
    #[must_use = "a contradiction must be handled"]
    pub fn settle(self, facts: &ArchFacts, architecture: &str) -> Result<Self, FactsError> {
        let embedding = facts.embedding_scale.map(EmbeddingFact::scale);
        Ok(Self {
            embedding_scale: settle_one(self.embedding_scale, embedding, "embedding_scale", architecture)?,
            activation: settle_one(self.activation, facts.activation, "activation", architecture)?,
            routed_gating: settle_one(self.routed_gating, facts.routed_gating, "routed_gating", architecture)?,
            score_scale_inverse_sqrt_head_dim: settle_one(
                self.score_scale_inverse_sqrt_head_dim,
                facts.score_scale_inverse_sqrt_head_dim,
                "score_scale_inverse_sqrt_head_dim",
                architecture,
            )?,
            value_norm: settle_one(self.value_norm, facts.value_norm, "value_norm", architecture)?,
            rope_layout: settle_one(self.rope_layout, facts.rope_layout, "rope_layout", architecture)?,
            recurrence: settle_one(self.recurrence, facts.recurrence.map(Some), "recurrence", architecture)?,
            ssm_v_head_reordered: settle_one(
                self.ssm_v_head_reordered,
                facts.ssm_v_head_reordered,
                "ssm_v_head_reordered",
                architecture,
            )?,
            ..self
        })
    }

    /// The [`FamilyProfile`] these facts describe. A slot that does not apply to
    /// this checkpoint (a routing bias on a dense model) takes the neutral value:
    /// no lowering of this checkpoint reads it.
    ///
    /// # Errors
    ///
    /// [`FactsError::Unresolved`] naming every open field.
    #[must_use = "an incomplete profile must be handled"]
    pub fn into_profile(self, architecture: &str) -> Result<FamilyProfile, FactsError> {
        let open = self.unresolved();
        if !open.is_empty() {
            return Err(FactsError::Unresolved {
                architecture: architecture.to_string(),
                fields: open.join(", "),
            });
        }
        let (combination, parallel) = match known(self.layer_combination, FfnCombination::Exclusive) {
            FfnCombination::ParallelDenseMoe(config) => (FfnCombination::Exclusive, Some(config)),
            other => (other, None),
        };
        Ok(FamilyProfile {
            embedding_scale: known(self.embedding_scale, None),
            ffn: LayerFfnConfig {
                post_attention_norm: known(self.post_attention_norm, false),
                combination,
                output_scale: known(self.output_scale, false),
                routed_gating: known(self.routed_gating, ExpertGatingFunc::Softmax),
                routed_expert_bias: known(self.routed_expert_bias, false),
                activation: known(self.activation, Activation::Silu),
                dense_feed_forward: None,
                exclusive_dense_post_norm: known(self.exclusive_dense_post_norm, false),
                ple: false,
                shared_expert_gate: known(self.shared_expert_gate, false),
            },
            parallel_dense_moe: parallel,
            score_scale_inverse_sqrt_head_dim: known(self.score_scale_inverse_sqrt_head_dim, true),
            value_norm: known(self.value_norm, false),
            qk_norm: known(self.qk_norm, false),
            rope_layout: known(self.rope_layout, RopeLayout::SplitHalf),
            recurrence: known(self.recurrence, None),
            schedule_source: known(self.schedule_source, Default::default()),
            ssm_v_head_reordered: known(self.ssm_v_head_reordered, false),
        })
    }
}

fn known<T>(slot: Slot<T>, neutral: T) -> T {
    match slot {
        Slot::Known(value) => value,
        Slot::NotApplicable | Slot::Unresolved => neutral,
    }
}

