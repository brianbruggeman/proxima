use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::error::ProximaError;
use crate::pipelines::spec::PipelineSpec;

/// One row in an `explain` trace. `depth = 0` is the queried stage;
/// each `depends_on` edge increments depth. For a diamond like
/// `bench depends_on build, build depends_on fetch`, explain(bench)
/// returns `[(bench,0), (build,1), (fetch,2)]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplainStep {
    pub stage: String,
    pub depth: usize,
}

/// Walk a stage's `depends_on` closure. Returns the queried stage at
/// `depth = 0` followed by each ancestor, deduplicated, in
/// depth-first order with stable tie-breaking (spec-declaration order
/// among siblings at the same depth). Errors if the stage isn't in
/// the spec.
pub fn explain_stage(
    spec: &PipelineSpec,
    stage_name: &str,
) -> Result<Vec<ExplainStep>, ProximaError> {
    if !spec.stages.iter().any(|stage| stage.name == stage_name) {
        return Err(ProximaError::NotFound(format!(
            "stage `{stage_name}` not in pipeline spec"
        )));
    }
    let mut output: Vec<ExplainStep> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    visit(spec, stage_name, 0, &mut output, &mut visited);
    Ok(output)
}

fn visit(
    spec: &PipelineSpec,
    name: &str,
    depth: usize,
    output: &mut Vec<ExplainStep>,
    visited: &mut HashSet<String>,
) {
    if !visited.insert(name.to_string()) {
        return;
    }
    output.push(ExplainStep {
        stage: name.to_string(),
        depth,
    });
    let Some(stage) = spec.stages.iter().find(|stage| stage.name == name) else {
        return;
    };
    for parent in &stage.depends_on {
        visit(spec, parent, depth + 1, output, visited);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipelines::spec::StageSpec;
    use std::collections::BTreeMap;

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
    fn linear_chain_returns_self_then_ancestors() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![
                stage("fetch", &[]),
                stage("build", &["fetch"]),
                stage("bench", &["build"]),
            ],
        };
        let trace = explain_stage(&spec, "bench").expect("explain");
        assert_eq!(
            trace,
            vec![
                ExplainStep {
                    stage: "bench".into(),
                    depth: 0
                },
                ExplainStep {
                    stage: "build".into(),
                    depth: 1
                },
                ExplainStep {
                    stage: "fetch".into(),
                    depth: 2
                },
            ]
        );
    }

    #[test]
    fn diamond_visits_each_ancestor_once() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![
                stage("a", &[]),
                stage("b", &["a"]),
                stage("c", &["a"]),
                stage("d", &["b", "c"]),
            ],
        };
        let trace = explain_stage(&spec, "d").expect("explain");
        let names: Vec<&str> = trace.iter().map(|step| step.stage.as_str()).collect();
        assert_eq!(names, vec!["d", "b", "a", "c"]);
        // `a` must appear once even though both b and c depend on it
        let a_count = names.iter().filter(|name| **name == "a").count();
        assert_eq!(a_count, 1);
    }

    #[test]
    fn root_stage_explain_is_self_only() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![stage("root", &[])],
        };
        let trace = explain_stage(&spec, "root").expect("explain");
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].stage, "root");
        assert_eq!(trace[0].depth, 0);
    }

    #[test]
    fn unknown_stage_returns_not_found() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![stage("a", &[])],
        };
        let outcome = explain_stage(&spec, "nope");
        assert!(matches!(outcome, Err(ProximaError::NotFound(_))));
    }
}
