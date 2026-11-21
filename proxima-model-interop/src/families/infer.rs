//! The inferrer: a pure function from tensor names, shapes and config to a
//! filled family template.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{self, Write};

use proxima_safetensors::TensorEntry;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::template::{Dim, FamilyTemplate, Knob, Probe, Stack, Value};

/// One tensor's name and shape: all a family needs to know of a weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub name: String,
    pub shape: Vec<u64>,
}

impl From<&TensorEntry> for TensorShape {
    fn from(entry: &TensorEntry) -> Self {
        Self {
            name: entry.name.clone(),
            shape: entry.shape.clone(),
        }
    }
}

/// The sha256 of the whole checkpoint file, as 64 lowercase hex digits.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CheckpointSha(String);

/// A checkpoint sha that is not 64 lowercase hex digits.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("checkpoint sha must be 64 lowercase hex digits, got {0:?}")]
pub struct ShaError(String);

impl CheckpointSha {
    /// # Errors
    ///
    /// [`ShaError`] unless `text` is 64 lowercase hex digits.
    #[must_use = "a refused sha must be handled"]
    pub fn parse(text: &str) -> Result<Self, ShaError> {
        let hex = |byte: u8| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte);
        match text.len() == 64 && text.bytes().all(hex) {
            true => Ok(Self(text.to_string())),
            false => Err(ShaError(text.to_string())),
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Where an inferred value came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Tensor { name: String, axis: Option<usize> },
    Stack { pattern: String },
    Config { key: String },
    Default { cite: String },
}

/// A value and its provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct Fact {
    pub value: Value,
    pub source: Source,
}

/// A template filled from one checkpoint; a hypothesis until accepted.
#[derive(Debug, Clone, PartialEq)]
pub struct Inferred {
    pub family: String,
    pub sha: CheckpointSha,
    pub facts: BTreeMap<String, Fact>,
    /// Golden stages that must each pass before the model runs.
    pub required_stages: Vec<String>,
}

impl Inferred {
    /// A hex sha256 over the family and every fact with its source, so a
    /// changed rule that changes any filled value re-opens the golden.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut text = format!("{}\n{:?}\n", self.family, self.required_stages);
        for (key, fact) in &self.facts {
            let _ = writeln!(text, "{key}={:?}@{:?}", fact.value, fact.source);
        }
        Sha256::digest(text.as_bytes())
            .iter()
            .fold(String::new(), |mut hex, byte| {
                let _ = write!(hex, "{byte:02x}");
                hex
            })
    }
}

/// The first requirement a family failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unmet {
    pub family: String,
    pub pattern: String,
}

impl fmt::Display for Unmet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.family, self.pattern)
    }
}

/// Why a checkpoint was not inferred.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InferError {
    #[error("no family matches the checkpoint; unmet: {}", join(.unmet))]
    NoFamily { unmet: Vec<Unmet> },
    #[error("several families match the checkpoint: {families:?}")]
    AmbiguousFamily { families: Vec<String> },
    #[error("stack {role:?} has no layer {missing} but has a later one")]
    StackGap { role: String, missing: u64 },
    #[error("family {family:?} leaves {} tensors unclaimed, first {:?}", .names.len(), .names.first())]
    UnclaimedTensors { family: String, names: Vec<String> },
    #[error("probe {key:?} holds in some layers and not others")]
    MixedProbe { key: String },
    #[error("tensor {tensor:?} the family needs is absent")]
    MissingTensor { tensor: String },
    #[error("tensor {tensor:?} has shape {shape:?}, which has no axis {axis} at rank {rank:?}")]
    ShapeAxis {
        tensor: String,
        axis: usize,
        rank: Option<usize>,
        shape: Vec<u64>,
    },
    #[error("dim {key:?} is {first} but {second}")]
    DimDisagrees {
        key: String,
        first: String,
        second: String,
    },
    #[error("config {key:?} says {config} but the tensors say {inferred}")]
    ConfigContradicts {
        key: String,
        config: String,
        inferred: String,
    },
    #[error("config {key:?} holds a value that is not a scalar")]
    ConfigType { key: String },
    #[error("knob {key:?} is in neither config {config:?} nor a default")]
    UnfilledKnob { key: String, config: String },
}

fn join(unmet: &[Unmet]) -> String {
    unmet
        .iter()
        .map(Unmet::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

type Shapes<'tensors> = BTreeMap<&'tensors str, &'tensors [u64]>;
type Facts = BTreeMap<String, Fact>;

/// Matches `tensors` to exactly one of `templates` and fills its slots.
///
/// Composes nothing async: a pure function, usable at the alloc tier.
///
/// # Errors
///
/// An [`InferError`]: no or several families, a gapped stack, unclaimed
/// tensors, disagreeing dims, a config that contradicts the shapes, or a knob
/// with no source.
#[must_use = "an inference failure must be handled"]
pub fn infer(
    tensors: &[TensorShape],
    config: &serde_json::Value,
    templates: &[FamilyTemplate],
    sha: CheckpointSha,
) -> Result<Inferred, InferError> {
    let shapes: Shapes = tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice()))
        .collect();
    let family = select(&shapes, templates)?;
    let mut facts = Facts::new();
    fill_stacks(family, &shapes, &mut facts)?;
    let companions = check_claims(family, &shapes)?;
    if !family.claim_prefixes.is_empty() {
        let source = Source::Stack {
            pattern: family.claim_prefixes.join(","),
        };
        facts.insert("companion_tensors".to_string(), fact(Value::Int(companions), source));
    }
    fill_probes(family, &shapes, &mut facts)?;
    fill_dims(family, &shapes, &mut facts)?;
    fill_knobs(family, config, &mut facts)?;
    Ok(Inferred {
        family: family.name.clone(),
        sha,
        facts,
        required_stages: family.required_stages.clone(),
    })
}

fn select<'t>(
    shapes: &Shapes,
    templates: &'t [FamilyTemplate],
) -> Result<&'t FamilyTemplate, InferError> {
    let mut unmet = Vec::new();
    let mut matched = Vec::new();
    for template in templates {
        match first_unmet(template, shapes) {
            None => matched.push(template),
            Some(pattern) => unmet.push(Unmet {
                family: template.name.clone(),
                pattern,
            }),
        }
    }
    match matched.as_slice() {
        [only] => Ok(only),
        [] => Err(InferError::NoFamily { unmet }),
        _ => Err(InferError::AmbiguousFamily {
            families: matched.iter().map(|family| family.name.clone()).collect(),
        }),
    }
}

fn first_unmet(template: &FamilyTemplate, shapes: &Shapes) -> Option<String> {
    let empty_stack = template
        .stacks
        .iter()
        .find(|stack| layer_indices(stack, shapes).is_empty())
        .map(|stack| stack.pattern.to_string());
    empty_stack
        .or_else(|| probe_failure(template, shapes, &template.requires, ProbeState::All))
        .or_else(|| probe_failure(template, shapes, &template.forbids, ProbeState::None))
}

fn probe_failure(
    template: &FamilyTemplate,
    shapes: &Shapes,
    keys: &[String],
    wanted: ProbeState,
) -> Option<String> {
    keys.iter()
        .filter_map(|key| template.probe(key))
        .find(|probe| probe_state(template, probe, shapes) != wanted)
        .map(|probe| match wanted {
            ProbeState::None => format!("absent {}", probe.pattern),
            _ => probe.pattern.to_string(),
        })
}

fn layer_indices(stack: &Stack, shapes: &Shapes) -> BTreeSet<u64> {
    shapes
        .keys()
        .filter_map(|name| stack.pattern.claims(name))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeState {
    All,
    None,
    Mixed,
}

fn probe_state(template: &FamilyTemplate, probe: &Probe, shapes: &Shapes) -> ProbeState {
    let layers = template
        .stacks
        .iter()
        .find(|stack| stack.role == probe.role)
        .map(|stack| layer_indices(stack, shapes))
        .unwrap_or_default();
    let present = layers
        .iter()
        .filter(|layer| {
            core::iter::once(&probe.pattern)
                .chain(&probe.also)
                .any(|spelling| shapes.contains_key(spelling.render(**layer).as_str()))
        })
        .count();
    match present {
        0 => ProbeState::None,
        _ if probe.any => ProbeState::All,
        count if count == layers.len() => ProbeState::All,
        _ => ProbeState::Mixed,
    }
}

fn fill_stacks(
    template: &FamilyTemplate,
    shapes: &Shapes,
    facts: &mut Facts,
) -> Result<(), InferError> {
    for stack in &template.stacks {
        let layers = layer_indices(stack, shapes);
        let count = layers.len() as u64;
        if let Some(missing) = (0..count).find(|index| !layers.contains(index)) {
            return Err(InferError::StackGap {
                role: stack.role.clone(),
                missing,
            });
        }
        let source = Source::Stack {
            pattern: stack.pattern.to_string(),
        };
        facts.insert(
            format!("{}_layers", stack.role),
            fact(Value::Int(count), source),
        );
    }
    Ok(())
}

fn fact(value: Value, source: Source) -> Fact {
    Fact { value, source }
}

/// The count of tensors only a `claim_prefixes` entry claims: another family's
/// weights riding in the same file, counted so they are reported, not hidden.
fn check_claims(template: &FamilyTemplate, shapes: &Shapes) -> Result<u64, InferError> {
    let companion = |name: &str| {
        template
            .claim_prefixes
            .iter()
            .any(|prefix| name.starts_with(prefix.as_str()))
    };
    let claimed = |name: &str| {
        companion(name)
            || template.claims.iter().any(|claim| claim == name)
            || template
                .stacks
                .iter()
                .any(|stack| stack.pattern.claims(name).is_some())
    };
    let names: Vec<String> = shapes
        .keys()
        .filter(|name| !claimed(name))
        .map(ToString::to_string)
        .collect();
    match names.is_empty() {
        true => Ok(shapes.keys().filter(|name| companion(name)).count() as u64),
        false => Err(InferError::UnclaimedTensors {
            family: template.name.clone(),
            names,
        }),
    }
}

fn fill_probes(
    template: &FamilyTemplate,
    shapes: &Shapes,
    facts: &mut Facts,
) -> Result<(), InferError> {
    for probe in &template.probes {
        let present = match probe_state(template, probe, shapes) {
            ProbeState::All => true,
            ProbeState::None => false,
            ProbeState::Mixed => {
                return Err(InferError::MixedProbe {
                    key: probe.key.clone(),
                });
            }
        };
        let source = Source::Stack {
            pattern: probe.pattern.to_string(),
        };
        facts.insert(probe.key.clone(), fact(Value::Flag(present), source));
    }
    Ok(())
}

fn fill_dims(
    template: &FamilyTemplate,
    shapes: &Shapes,
    facts: &mut Facts,
) -> Result<(), InferError> {
    for dim in &template.dims {
        let value = dim_value(dim, shapes)?;
        let source = Source::Tensor {
            name: dim.tensor.clone(),
            axis: Some(dim.axis),
        };
        if let Some(earlier) = facts.get(&dim.key) {
            check_agreement(dim, value, earlier)?;
        }
        facts
            .entry(dim.key.clone())
            .or_insert(fact(Value::Int(value), source));
    }
    Ok(())
}

fn dim_value(dim: &Dim, shapes: &Shapes) -> Result<u64, InferError> {
    let shape = shapes
        .get(dim.tensor.as_str())
        .ok_or_else(|| InferError::MissingTensor {
            tensor: dim.tensor.clone(),
        })?;
    let rank_ok = dim.rank.is_none_or(|rank| rank == shape.len());
    match shape.get(dim.axis) {
        Some(value) if rank_ok => Ok(*value),
        _ => Err(InferError::ShapeAxis {
            tensor: dim.tensor.clone(),
            axis: dim.axis,
            rank: dim.rank,
            shape: shape.to_vec(),
        }),
    }
}

fn check_agreement(dim: &Dim, value: u64, earlier: &Fact) -> Result<(), InferError> {
    match (&earlier.value, &earlier.source) {
        (Value::Int(first), _) if *first == value => Ok(()),
        (first, source) => Err(InferError::DimDisagrees {
            key: dim.key.clone(),
            first: format!("{first:?} from {source:?}"),
            second: format!("{value} from {} axis {}", dim.tensor, dim.axis),
        }),
    }
}

fn fill_knobs(
    template: &FamilyTemplate,
    config: &serde_json::Value,
    facts: &mut Facts,
) -> Result<(), InferError> {
    for knob in &template.knobs {
        let from_config = config_value(config, &knob.config)?;
        match &knob.agrees_with {
            Some(fact_key) => agree(knob, from_config, facts.get(fact_key))?,
            None => {
                let found = knob_fact(knob, from_config)?;
                facts.insert(knob.key.clone(), found);
            }
        }
    }
    Ok(())
}

fn config_value(config: &serde_json::Value, key: &str) -> Result<Option<Value>, InferError> {
    let converted = config.get(key).map(|raw| match raw {
        serde_json::Value::Bool(flag) => Some(Value::Flag(*flag)),
        serde_json::Value::String(text) => Some(Value::Text(text.clone())),
        serde_json::Value::Number(number) => number
            .as_u64()
            .map(Value::Int)
            .or_else(|| number.as_f64().map(Value::Float)),
        _ => None,
    });
    match converted {
        Some(None) => Err(InferError::ConfigType {
            key: key.to_string(),
        }),
        Some(value) => Ok(value),
        None => Ok(None),
    }
}

fn agree(
    knob: &Knob,
    from_config: Option<Value>,
    inferred: Option<&Fact>,
) -> Result<(), InferError> {
    match (from_config, inferred) {
        (Some(config), Some(shape)) if config != shape.value => {
            Err(InferError::ConfigContradicts {
                key: knob.config.clone(),
                config: format!("{config:?}"),
                inferred: format!("{:?} from {:?}", shape.value, shape.source),
            })
        }
        _ => Ok(()),
    }
}

fn knob_fact(knob: &Knob, from_config: Option<Value>) -> Result<Fact, InferError> {
    match (from_config, &knob.default, &knob.cite) {
        (Some(value), _, _) => Ok(fact(
            value,
            Source::Config {
                key: knob.config.clone(),
            },
        )),
        (None, Some(value), Some(cite)) => {
            Ok(fact(value.clone(), Source::Default { cite: cite.clone() }))
        }
        _ => Err(InferError::UnfilledKnob {
            key: knob.key.clone(),
            config: knob.config.clone(),
        }),
    }
}
