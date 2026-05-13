use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::error::ProximaError;
use crate::pipelines::spec::PipelineSpec;

/// Returns stage names in a topological order respecting `depends_on`
/// edges. Errors if the DAG has a cycle. Stable: ties (independent
/// stages with no relative ordering) are returned in spec-declaration
/// order, so deterministic test assertions are possible.
pub fn topological_order(spec: &PipelineSpec) -> Result<Vec<String>, ProximaError> {
    let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
    let mut children: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut declaration_index: BTreeMap<String, usize> = BTreeMap::new();
    for (index, stage) in spec.stages.iter().enumerate() {
        declaration_index.insert(stage.name.clone(), index);
        in_degree.entry(stage.name.clone()).or_insert(0);
        children.entry(stage.name.clone()).or_default();
    }
    for stage in &spec.stages {
        for parent in &stage.depends_on {
            *in_degree.entry(stage.name.clone()).or_insert(0) += 1;
            children
                .entry(parent.clone())
                .or_default()
                .push(stage.name.clone());
        }
    }

    // ready queue: stable across runs by sorting on declaration order
    let mut ready: VecDeque<String> = in_degree
        .iter()
        .filter(|(_, degree)| **degree == 0)
        .map(|(name, _)| name.clone())
        .collect();
    let mut ordered: Vec<String> = Vec::with_capacity(spec.stages.len());
    let mut emitted: BTreeSet<String> = BTreeSet::new();

    while !ready.is_empty() {
        // sort the current ready set by declaration_index for determinism
        let mut current: Vec<String> = ready.drain(..).collect();
        current.sort_by_key(|name| declaration_index.get(name).copied().unwrap_or(usize::MAX));
        for name in current {
            if !emitted.insert(name.clone()) {
                continue;
            }
            ordered.push(name.clone());
            if let Some(child_names) = children.get(&name).cloned() {
                for child in child_names {
                    if let Some(degree) = in_degree.get_mut(&child) {
                        *degree = degree.saturating_sub(1);
                        if *degree == 0 {
                            ready.push_back(child);
                        }
                    }
                }
            }
        }
    }

    if ordered.len() != spec.stages.len() {
        let remaining: Vec<String> = in_degree
            .into_iter()
            .filter(|(name, degree)| *degree > 0 && !emitted.contains(name))
            .map(|(name, _)| name)
            .collect();
        return Err(ProximaError::Config(format!(
            "pipeline DAG has a cycle involving stages: {remaining:?}"
        )));
    }
    Ok(ordered)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipelines::spec::StageSpec;

    fn stage(name: &str, deps: &[&str]) -> StageSpec {
        StageSpec {
            name: name.into(),
            command: "/bin/true".into(),
            args: Vec::new(),
            env: Default::default(),
            cwd: None,
            depends_on: deps.iter().map(|raw| (*raw).into()).collect(),
        }
    }

    #[test]
    fn linear_dag_orders_by_dependency() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![stage("a", &[]), stage("b", &["a"]), stage("c", &["b"])],
        };
        let order = topological_order(&spec).expect("topo");
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn diamond_dag_preserves_dependency_constraints() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![
                stage("a", &[]),
                stage("b", &["a"]),
                stage("c", &["a"]),
                stage("d", &["b", "c"]),
            ],
        };
        let order = topological_order(&spec).expect("topo");
        // a must come before b, c, d
        let position = |needle: &str| order.iter().position(|x| x == needle).expect("present");
        assert!(position("a") < position("b"));
        assert!(position("a") < position("c"));
        assert!(position("b") < position("d"));
        assert!(position("c") < position("d"));
    }

    #[test]
    fn diamond_dag_is_stable_across_runs() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![
                stage("a", &[]),
                stage("b", &["a"]),
                stage("c", &["a"]),
                stage("d", &["b", "c"]),
            ],
        };
        let first = topological_order(&spec).expect("topo");
        for _ in 0..10 {
            let next = topological_order(&spec).expect("topo");
            assert_eq!(first, next, "topo must be stable across runs");
        }
    }

    #[test]
    fn cycle_returns_config_error_listing_involved_stages() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![stage("a", &["b"]), stage("b", &["a"])],
        };
        let err = topological_order(&spec).expect_err("cycle");
        match err {
            ProximaError::Config(message) => {
                assert!(message.contains("cycle"));
                assert!(message.contains("a") || message.contains("b"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn isolated_stages_appear_in_declaration_order() {
        let spec = PipelineSpec {
            name: None,
            stages: vec![stage("c", &[]), stage("a", &[]), stage("b", &[])],
        };
        let order = topological_order(&spec).expect("topo");
        assert_eq!(order, vec!["c", "a", "b"]);
    }
}
