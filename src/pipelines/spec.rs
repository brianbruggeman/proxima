use std::collections::BTreeMap;
use std::path::PathBuf;

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::error::ProximaError;

/// A declarative pipeline: a DAG of stages, each one a child process.
/// Stage identity is by user-supplied `name`; `depends_on` edges
/// reference parent stage names. At submission the executor allocates
/// per-stage `InteractionId`s and emits universal `RecordingEvent`s
/// stamped with `parent = pipeline_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelineSpec {
    /// Optional human-readable handle. Lets `proxima pipeline inspect example-search`
    /// resolve to a specific pipeline submission. Not required for execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub stages: Vec<StageSpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageSpec {
    /// Unique within the pipeline. Used by `depends_on` edges and human references.
    pub name: String,
    pub command: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Names of stages that must complete successfully before this stage runs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
}

impl PipelineSpec {
    /// Validate structural integrity: every `depends_on` reference must
    /// resolve to a known stage name, and stage names must be unique.
    /// Cycle detection happens at topo-sort time in `dag::topological_order`.
    pub fn validate(&self) -> Result<(), ProximaError> {
        let mut seen = BTreeMap::new();
        for stage in &self.stages {
            if seen.insert(stage.name.clone(), ()).is_some() {
                return Err(ProximaError::Config(format!(
                    "duplicate stage name `{}` in pipeline spec",
                    stage.name
                )));
            }
        }
        for stage in &self.stages {
            for parent in &stage.depends_on {
                if !seen.contains_key(parent) {
                    return Err(ProximaError::Config(format!(
                        "stage `{}` depends on unknown stage `{}`",
                        stage.name, parent
                    )));
                }
                if parent == &stage.name {
                    return Err(ProximaError::Config(format!(
                        "stage `{}` cannot depend on itself",
                        stage.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// BLAKE3 hash of the canonical serialization. Identical specs
    /// produce identical hashes, regardless of submission timing or
    /// pipeline id. The `name` field is part of the hash — renaming a
    /// pipeline produces a different hash. (If hash stability across
    /// renames matters later, drop `name` from the hashed payload.)
    #[must_use]
    pub fn spec_hash(&self) -> [u8; 32] {
        // serde_json serializes BTreeMap with sorted keys deterministically;
        // we wrap stages in a struct that mirrors the public shape so the
        // hash isn't sensitive to future field additions on PipelineSpec
        // (it'll change anyway, but at least the surface is intentional).
        let payload = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Hasher::new();
        hasher.update(&payload);
        let digest = hasher.finalize();
        let mut output = [0_u8; 32];
        output.copy_from_slice(digest.as_bytes());
        output
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn stage(name: &str, deps: &[&str]) -> StageSpec {
        StageSpec {
            name: name.into(),
            command: "/bin/true".into(),
            args: Vec::new(),
            env: BTreeMap::new(),
            cwd: None,
            depends_on: deps.iter().map(|raw| (*raw).into()).collect(),
        }
    }

    #[test]
    fn validate_accepts_well_formed_dag() {
        let spec = PipelineSpec {
            name: Some("p".into()),
            stages: vec![stage("a", &[]), stage("b", &["a"]), stage("c", &["a", "b"])],
        };
        spec.validate().expect("valid spec");
    }

    #[test]
    fn validate_rejects_duplicate_stage_names() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![stage("a", &[]), stage("a", &[])],
        };
        let err = spec.validate().expect_err("dup");
        assert!(matches!(err, ProximaError::Config(_)));
    }

    #[test]
    fn validate_rejects_unknown_dependency() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![stage("a", &["does-not-exist"])],
        };
        let err = spec.validate().expect_err("unknown dep");
        assert!(matches!(err, ProximaError::Config(_)));
    }

    #[test]
    fn validate_rejects_self_dependency() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![stage("a", &["a"])],
        };
        let err = spec.validate().expect_err("self dep");
        assert!(matches!(err, ProximaError::Config(_)));
    }

    #[test]
    fn spec_hash_is_stable_across_identical_specs() {
        let one = PipelineSpec {
            name: Some("p".into()),
            stages: vec![stage("a", &[]), stage("b", &["a"])],
        };
        let two = one.clone();
        assert_eq!(one.spec_hash(), two.spec_hash());
    }

    #[test]
    fn spec_hash_differs_when_stage_command_changes() {
        let mut one = PipelineSpec {
            name: Some("p".into()),
            stages: vec![stage("a", &[])],
        };
        let two = one.clone();
        one.stages[0].command = "/usr/bin/true".into();
        assert_ne!(one.spec_hash(), two.spec_hash());
    }
}
