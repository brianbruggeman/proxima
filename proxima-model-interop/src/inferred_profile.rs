//! The serving seam's profile: built on the fly from a checkpoint's own header
//! plus the architecture facts, in place of a per-family row (PROXNUM-611).
//!
//! Teaching pointer: [`crate::families::infer`] matches the tensor directory to
//! a template, [`crate::families::infer_profile`] reads every field the file
//! states, and [`crate::families::ProfileFacts::settle`] fills the rest from
//! the [`FactsTable`]. [`ProfileSourceSettings`] is the switch, built the three
//! ways the other settings are: fluent, TOML and `PROXIMA_FAMILY_PROFILE_*` env.
//! The old row path stays compiled ([`crate::profiles::family_profile`]) as the
//! differential's incumbent and as the fallback for a checkpoint no template
//! matches or an architecture with no facts section.

use std::fs;

use bon::Builder;
use conflaguration::{Settings, Validate};
use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::FamilyProfile;
use proxima_telemetry::warn;
use serde::{Deserialize, Serialize};

use crate::error::InteropError;
use crate::profiles::family_profile;
use crate::families::{
    CheckpointSha, DecoderFamily, FactsTable, TensorShape, gguf_config, infer,
    infer_profile,
};

const UNHASHED: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// Which source builds the serving profile.
#[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize, Settings, Validate)]
#[settings(prefix = "PROXIMA_FAMILY_PROFILE")]
#[builder(derive(Clone, Debug))]
#[serde(default)]
pub struct ProfileSourceSettings {
    /// `true` builds the profile from the header plus the facts table; `false`
    /// reads the per-family row.
    #[setting(default = true)]
    #[builder(default = true)]
    pub inferred: bool,
    /// A facts TOML replacing the embedded one.
    #[setting(default)]
    #[serde(default)]
    pub facts_file: Option<String>,
}

impl Default for ProfileSourceSettings {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl ProfileSourceSettings {
    fn facts(&self, family: &str) -> Result<FactsTable, InteropError> {
        let invalid = |message: String| InteropError::InvalidFamilyProfile {
            family: family.into(),
            message,
        };
        match &self.facts_file {
            Some(path) => {
                let text = fs::read_to_string(path).map_err(|error| invalid(format!("{path}: {error}")))?;
                Ok(FactsTable::parse(&text)?)
            }
            None => Ok(FactsTable::builtin()?),
        }
    }
}

/// The profile `parsed`'s header and `settings`' facts describe, or `None` when
/// the checkpoint matches no family template or the facts have no section for
/// `architecture`, so the caller keeps the row path.
///
/// # Errors
///
/// [`InteropError::FamilyFacts`] when a fact contradicts the file or a field is
/// still open after the facts; [`InteropError::InvalidFamilyProfile`] when the
/// family, the settings or the facts file cannot be read.
pub fn inferred_profile(
    parsed: &ParsedGguf,
    architecture: &str,
    settings: &ProfileSourceSettings,
) -> Result<Option<FamilyProfile>, InteropError> {
    let invalid = |message: String| InteropError::InvalidFamilyProfile {
        family: architecture.into(),
        message,
    };
    let family = DecoderFamily::builtin().map_err(|error| invalid(error.to_string()))?;
    let templates = family.templates().map_err(|error| invalid(error.to_string()))?;
    let tensors: Vec<TensorShape> = parsed
        .tensors
        .iter()
        .map(|tensor| TensorShape {
            name: tensor.name.clone(),
            shape: tensor.dims.to_vec(),
        })
        .collect();
    let config = gguf_config(parsed);
    let sha = CheckpointSha::parse(UNHASHED).map_err(|error| invalid(error.to_string()))?;
    let Ok(inferred) = infer(&tensors, &config, &templates, sha) else {
        return Ok(None);
    };
    let table = settings.facts(architecture)?;
    let Some(facts) = table.get(architecture) else {
        return Ok(None);
    };
    let read = infer_profile(&family, &inferred, &config).map_err(|error| invalid(error.to_string()))?;
    Ok(Some(read.settle(facts, architecture)?.into_profile(architecture)?))
}

/// The one place a GGUF load's family profile is resolved: the inferred
/// profile of [`inferred_profile`], or, transitionally and only here, the
/// per-family row when the facts refuse or have no section (PROXNUM-624 and
/// PROXNUM-625 delete that arm). Every GGUF header reader takes the result;
/// none looks a row up itself.
///
/// # Errors
///
/// The settings or family cannot be read, a fact contradicts the file, or the
/// fallback row is absent.
pub(crate) fn resolved_profile(parsed: &ParsedGguf, family: &str) -> Result<FamilyProfile, InteropError> {
    let settings = ProfileSourceSettings::from_env().map_err(|error| InteropError::InvalidFamilyProfile {
        family: family.into(),
        message: error.to_string(),
    })?;
    if settings.inferred {
        match inferred_profile(parsed, family, &settings) {
            Ok(Some(profile)) => return Ok(profile),
            Ok(None) => {}
            Err(InteropError::FamilyFacts(refusal)) => {
                warn!(family = ?family, refusal = ?refusal, "architecture facts refused; serving from the family row");
            }
            Err(other) => return Err(other),
        }
    }
    family_profile(family)
}

pub(crate) fn inferred_profile_from_hf(
    config: &serde_json::Value,
    family: &str,
    tensor_names: &[&str],
    settings: &ProfileSourceSettings,
) -> Result<Option<FamilyProfile>, InteropError> {
    let table = settings.facts(family)?;
    let Some(facts) = table.get(family) else {
        return Ok(None);
    };
    let invalid = |message: String| InteropError::InvalidFamilyProfile {
        family: family.into(),
        message,
    };
    let decoder = DecoderFamily::builtin().map_err(|error| invalid(error.to_string()))?;
    let templates = decoder.templates().map_err(|error| invalid(error.to_string()))?;
    let tensors: Vec<TensorShape> = tensor_names
        .iter()
        .map(|name| TensorShape {
            name: (*name).into(),
            shape: Vec::new(),
        })
        .collect();
    let sha = CheckpointSha::parse(UNHASHED).map_err(|error| invalid(error.to_string()))?;
    let Ok(inferred) = infer(&tensors, config, &templates, sha) else {
        return Ok(None);
    };
    let read = infer_profile(&decoder, &inferred, config).map_err(|error| invalid(error.to_string()))?;
    Ok(Some(read.settle(facts, family)?.into_profile(family)?))
}

/// The HF twin of [`resolved_profile`]: the profile a `config.json` and its
/// safetensors tensor names describe plus the `facts.toml` section for
/// `family`, with the same transitional row fallback living only here.
///
/// # Errors
///
/// As [`resolved_profile`].
pub(crate) fn resolved_profile_from_hf(
    config: &serde_json::Value,
    family: &str,
    tensor_names: &[&str],
) -> Result<FamilyProfile, InteropError> {
    let settings = ProfileSourceSettings::from_env().map_err(|error| InteropError::InvalidFamilyProfile {
        family: family.into(),
        message: error.to_string(),
    })?;
    if settings.inferred {
        match inferred_profile_from_hf(config, family, tensor_names, &settings) {
            Ok(Some(profile)) => return Ok(profile),
            Ok(None) => {}
            Err(InteropError::FamilyFacts(refusal)) => {
                warn!(family = ?family, refusal = ?refusal, "architecture facts refused; serving from the family row");
            }
            Err(other) => return Err(other),
        }
    }
    family_profile(family)
}
