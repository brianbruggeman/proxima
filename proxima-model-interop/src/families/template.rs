//! A family template: structure as data, so a new model is a TOML row.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::Deserialize;
use thiserror::Error;

use super::pattern::Pattern;

/// A scalar a template or `config.json` can state.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Flag(bool),
    Int(u64),
    Float(f64),
    Text(String),
}

/// A contiguous `{L}` stack, claimed by name prefix: `role` layers.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Stack {
    pub role: String,
    pub pattern: Pattern,
}

/// A boolean fact: is `pattern` present in every layer of stack `role`?
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Probe {
    pub key: String,
    pub role: String,
    pub pattern: Pattern,
    /// `true` asks "in any layer", so a hybrid stack whose layers differ is not
    /// a [`InferError::MixedProbe`]; `false` demands every layer or none.
    #[serde(default)]
    pub any: bool,
}

/// A dimension bound from one tensor's shape axis; `rank` pins the rank.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Dim {
    pub key: String,
    pub tensor: String,
    pub axis: usize,
    pub rank: Option<usize>,
}

/// A knob weights cannot show: read from `config`, else a cited default.
///
/// `agrees_with` turns the knob into a check against a shape fact.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Knob {
    pub key: String,
    pub config: String,
    pub agrees_with: Option<String>,
    pub default: Option<Value>,
    pub cite: Option<String>,
}

/// One family: what must be present, what must be absent, what is claimed.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FamilyTemplate {
    pub name: String,
    pub stacks: Vec<Stack>,
    #[serde(default)]
    pub claims: Vec<String>,
    /// Name prefixes of another family's tensors that ride in the same file
    /// (a vision tower, a speculative head); claimed so the file is accepted,
    /// and counted in the `companion_tensors` fact so they are not hidden.
    #[serde(default)]
    pub claim_prefixes: Vec<String>,
    #[serde(default)]
    pub probes: Vec<Probe>,
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub forbids: Vec<String>,
    #[serde(default)]
    pub dims: Vec<Dim>,
    #[serde(default)]
    pub knobs: Vec<Knob>,
    /// Golden stages (e.g. `encoder`) a checkpoint must pass before it runs.
    #[serde(default)]
    pub required_stages: Vec<String>,
}

/// Why a template was refused at load.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TemplateError {
    #[error("template does not parse: {reason}")]
    Parse { reason: String },
    #[error("template {family:?} knob {key:?} has a default but no cite")]
    DefaultWithoutCite { family: String, key: String },
    #[error("template {family:?} declares no required golden stages")]
    NoRequiredStages { family: String },
    #[error("template {family:?} names probe {key:?} but declares none by that key")]
    UnknownProbe { family: String, key: String },
    #[error("template {family:?} probe {key:?} names stack role {role:?} it does not declare")]
    UnknownRole {
        family: String,
        key: String,
        role: String,
    },
}

impl FamilyTemplate {
    /// Checks the cross-references a TOML schema cannot express.
    ///
    /// # Errors
    ///
    /// A [`TemplateError`] naming the first dangling probe, role or uncited default.
    #[must_use = "a refused template must be handled"]
    pub fn validate(self) -> Result<Self, TemplateError> {
        self.check_stages()?;
        self.check_cites()?;
        self.check_probes()?;
        Ok(self)
    }

    fn check_stages(&self) -> Result<(), TemplateError> {
        match self.required_stages.is_empty() {
            true => Err(TemplateError::NoRequiredStages {
                family: self.name.clone(),
            }),
            false => Ok(()),
        }
    }

    fn check_cites(&self) -> Result<(), TemplateError> {
        match self
            .knobs
            .iter()
            .find(|knob| knob.default.is_some() && knob.cite.is_none())
        {
            Some(knob) => Err(TemplateError::DefaultWithoutCite {
                family: self.name.clone(),
                key: knob.key.clone(),
            }),
            None => Ok(()),
        }
    }

    fn check_probes(&self) -> Result<(), TemplateError> {
        for probe in &self.probes {
            if !self.stacks.iter().any(|stack| stack.role == probe.role) {
                return Err(self.unknown_role(probe));
            }
        }
        match self.probe_refs().find(|key| self.probe(key).is_none()) {
            Some(key) => Err(TemplateError::UnknownProbe {
                family: self.name.clone(),
                key: key.to_string(),
            }),
            None => Ok(()),
        }
    }

    fn unknown_role(&self, probe: &Probe) -> TemplateError {
        TemplateError::UnknownRole {
            family: self.name.clone(),
            key: probe.key.clone(),
            role: probe.role.clone(),
        }
    }

    fn probe_refs(&self) -> impl Iterator<Item = &str> {
        self.requires
            .iter()
            .chain(&self.forbids)
            .map(String::as_str)
    }

    /// The probe declared under `key`.
    #[must_use]
    pub fn probe(&self, key: &str) -> Option<&Probe> {
        self.probes.iter().find(|probe| probe.key == key)
    }
}

#[cfg(feature = "toml")]
const BUILTIN: &[&str] = &[include_str!("encoder_decoder.toml")];

#[cfg(feature = "toml")]
impl FamilyTemplate {
    /// Parses and validates one template.
    ///
    /// # Errors
    ///
    /// [`TemplateError::Parse`] on malformed TOML or a bad pattern, else [`Self::validate`]'s.
    #[must_use = "a refused template must be handled"]
    pub fn from_toml(text: &str) -> Result<Self, TemplateError> {
        let parsed: Self = toml::from_str(text).map_err(|error| TemplateError::Parse {
            reason: error.to_string(),
        })?;
        parsed.validate()
    }

    /// The embedded templates, like `profiles::FAMILY_PROFILES`.
    ///
    /// # Errors
    ///
    /// The first embedded template that fails [`Self::from_toml`].
    #[must_use = "a refused template must be handled"]
    pub fn builtin() -> Result<Vec<Self>, TemplateError> {
        BUILTIN.iter().map(|text| Self::from_toml(text)).collect()
    }
}
