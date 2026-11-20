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

/// One DAG node. The body is flattened so a process stage keeps today's
/// JSON shape (`command`, `args`, `env`, `cwd`); a pipe stage carries
/// `pipe` + `request`. Deserialization goes through `RawStageSpec` so a
/// stage with both bodies or neither is refused with the stage name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawStageSpec")]
pub struct StageSpec {
    /// Unique within the pipeline. Used by `depends_on` edges and human references.
    pub name: String,
    #[serde(flatten)]
    pub body: StageBody,
    /// Names of stages that must complete successfully before this stage runs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
    /// CEL guard over `input` and the parents' outputs; false declines the stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum StageBody {
    Process {
        command: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        env: BTreeMap<String, String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<PathBuf>,
    },
    Pipe {
        pipe: String,
        request: StageRequest,
    },
}

/// Request a pipe stage sends; field names mirror `WorkloadSpec`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRequest {
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default = "default_path")]
    pub path: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub query: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
}

fn default_method() -> String {
    "GET".into()
}

fn default_path() -> String {
    "/".into()
}

#[derive(Deserialize)]
struct RawStageSpec {
    name: String,
    command: Option<String>,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    cwd: Option<PathBuf>,
    pipe: Option<String>,
    request: Option<StageRequest>,
    #[serde(default)]
    depends_on: Vec<String>,
    #[serde(default)]
    when: Option<String>,
}

fn refusal(name: &str, has_process: bool, has_pipe: bool) -> String {
    match (has_process, has_pipe) {
        (true, true) => format!("stage `{name}` has both a process body and a pipe body"),
        (false, false) => format!("stage `{name}` has neither a process body nor a pipe body"),
        _ => format!(
            "stage `{name}` is incomplete: a pipe body needs both `pipe` and `request`, a process body needs `command`"
        ),
    }
}

impl TryFrom<RawStageSpec> for StageSpec {
    type Error = String;

    fn try_from(raw: RawStageSpec) -> Result<Self, Self::Error> {
        let RawStageSpec { name, command, args, env, cwd, pipe, request, depends_on, when } = raw;
        let has_process =
            command.is_some() || !args.is_empty() || !env.is_empty() || cwd.is_some();
        let has_pipe = pipe.is_some() || request.is_some();
        let body = match (command, pipe, request) {
            (Some(command), None, None) => StageBody::Process { command, args, env, cwd },
            (None, Some(pipe), Some(request)) if !has_process => StageBody::Pipe { pipe, request },
            _ => return Err(refusal(&name, has_process, has_pipe)),
        };
        Ok(Self { name, body, depends_on, when })
    }
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
            body: StageBody::Process {
                command: "/bin/true".into(),
                args: Vec::new(),
                env: BTreeMap::new(),
                cwd: None,
            },
            depends_on: deps.iter().map(|raw| (*raw).into()).collect(),
            when: None,
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
        one.stages[0].body = StageBody::Process {
            command: "/usr/bin/true".into(),
            args: Vec::new(),
            env: BTreeMap::new(),
            cwd: None,
        };
        assert_ne!(one.spec_hash(), two.spec_hash());
    }

    fn parse_stage(raw: &str) -> Result<StageSpec, serde_json::Error> {
        serde_json::from_str(raw)
    }

    fn parsed_pipe_stage() -> StageSpec {
        StageSpec {
            name: "pr".into(),
            body: StageBody::Pipe {
                pipe: "github".into(),
                request: StageRequest {
                    method: "GET".into(),
                    path: "/repos/x/pulls/1".into(),
                    headers: BTreeMap::new(),
                    query: BTreeMap::new(),
                    body: None,
                },
            },
            depends_on: Vec::new(),
            when: None,
        }
    }

    const PIPE_STAGE_JSON: &str =
        r#"{"name":"pr","pipe":"github","request":{"method":"GET","path":"/repos/x/pulls/1"}}"#;

    #[test]
    fn stage_body_process_only_is_accepted() {
        let raw = r#"{"name":"build","command":"cargo","args":["check"]}"#;
        let stage = parse_stage(raw).expect("process-only stage parses");
        let expected = StageSpec {
            name: "build".into(),
            body: StageBody::Process {
                command: "cargo".into(),
                args: vec!["check".into()],
                env: BTreeMap::new(),
                cwd: None,
            },
            depends_on: Vec::new(),
            when: None,
        };
        assert_eq!(stage, expected);
    }

    #[test]
    fn stage_body_pipe_only_is_accepted() {
        let stage = parse_stage(PIPE_STAGE_JSON).expect("pipe-only stage parses");
        assert_eq!(stage, parsed_pipe_stage());
    }

    #[test]
    fn ctl_parse_round_trips_existing_process_stage() {
        let original = stage("build", &[]);
        let text = serde_json::to_string(&original).expect("stage serializes");
        let parsed = parse_stage(&text).expect("serialized stage parses back");
        assert_eq!(parsed, original);
    }

    #[test]
    fn ctl_parse_empty_pipe_name_is_detected() {
        let parsed = parse_stage(PIPE_STAGE_JSON).expect("pipe-only stage parses");
        let mut blanked = parsed_pipe_stage();
        if let StageBody::Pipe { pipe, .. } = &mut blanked.body {
            pipe.clear();
        }
        assert_ne!(parsed, blanked);
    }

    fn refusal_names(err_text: &str, stage: &str, word: &str) -> bool {
        err_text.contains(&format!("stage `{stage}`")) && err_text.contains(word)
    }

    #[test]
    fn stage_body_both_is_refused_naming_the_stage() {
        let raw = r#"{"name":"fetch-ticket","command":"cargo","pipe":"github","request":{"method":"GET","path":"/"}}"#;
        let err = parse_stage(raw).expect_err("both bodies must be refused");
        assert!(refusal_names(&err.to_string(), "fetch-ticket", "both"), "reason lost the stage name or `both`: {err}");
    }

    #[test]
    fn stage_body_neither_is_refused_naming_the_stage() {
        let raw = r#"{"name":"fetch-ticket"}"#;
        let err = parse_stage(raw).expect_err("no body must be refused");
        assert!(refusal_names(&err.to_string(), "fetch-ticket", "neither"), "reason lost the stage name or `neither`: {err}");
    }

    #[test]
    fn ctl_parse_generic_serde_message_is_rejected() {
        let text = "data did not match any variant of untagged enum StageBody";
        assert!(!refusal_names(text, "fetch-ticket", "both"));
    }

    #[test]
    fn ctl_parse_named_refusal_is_accepted() {
        let text = "stage `fetch-ticket` has both a process body and a pipe body";
        assert!(refusal_names(text, "fetch-ticket", "both"));
    }
}
