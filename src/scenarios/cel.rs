use std::collections::HashMap;

use cel_interpreter::{Context, Program, Value as CelValue};

use crate::error::ProximaError;
use crate::scenarios::orchestrator::CelBindings;

/// CEL evaluator against scenario runtime state. Bindings:
/// `success_rate` (f64, 1.0 when `completed == 0`), `successes` /
/// `completed` / `failures` (int), `counter[<name>]` (int, summed
/// across label sets), `histogram_p99[<name>]` (double, ms; max across
/// label sets).
pub fn evaluate(expression: &str, bindings: &CelBindings<'_>) -> Result<bool, ProximaError> {
    let program = Program::compile(expression)
        .map_err(|err| ProximaError::Config(format!("cel compile: {err}")))?;
    let mut context = Context::default();
    let success_rate = if bindings.completed > 0 {
        (bindings.successes as f64) / (bindings.completed as f64)
    } else {
        1.0
    };
    context
        .add_variable("success_rate", success_rate)
        .map_err(|err| ProximaError::Config(format!("cel bind success_rate: {err}")))?;
    context
        .add_variable("successes", bindings.successes as i64)
        .map_err(|err| ProximaError::Config(format!("cel bind successes: {err}")))?;
    context
        .add_variable("completed", bindings.completed as i64)
        .map_err(|err| ProximaError::Config(format!("cel bind completed: {err}")))?;
    context
        .add_variable("failures", bindings.failures as i64)
        .map_err(|err| ProximaError::Config(format!("cel bind failures: {err}")))?;

    let mut counter_map: HashMap<String, i64> = HashMap::new();
    for (metric, _, value) in &bindings.snapshot.counters {
        let entry = counter_map.entry(metric.clone()).or_insert(0);
        *entry += *value as i64;
    }
    context
        .add_variable("counter", counter_map)
        .map_err(|err| ProximaError::Config(format!("cel bind counter: {err}")))?;

    let mut histogram_p99_map: HashMap<String, f64> = HashMap::new();
    for (metric, _, summary) in &bindings.snapshot.histograms {
        // largest p99 across label sets — worst-case is the SLO answer.
        histogram_p99_map
            .entry(metric.clone())
            .and_modify(|existing| {
                if summary.p99 > *existing {
                    *existing = summary.p99;
                }
            })
            .or_insert(summary.p99);
    }
    context
        .add_variable("histogram_p99", histogram_p99_map)
        .map_err(|err| ProximaError::Config(format!("cel bind histogram_p99: {err}")))?;

    let result = program
        .execute(&context)
        .map_err(|err| ProximaError::Config(format!("cel evaluate: {err}")))?;
    match result {
        CelValue::Bool(flag) => Ok(flag),
        CelValue::Int(value) => Ok(value != 0),
        CelValue::UInt(value) => Ok(value != 0),
        CelValue::Float(value) => Ok(value != 0.0),
        other => Err(ProximaError::Config(format!(
            "cel expression must yield bool/int/float; got {other:?}"
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::telemetry::{HistogramSummary, Labels, MetricsSnapshot};

    fn snapshot_fixture() -> MetricsSnapshot {
        MetricsSnapshot {
            counters: vec![
                ("hits_total".into(), Labels::empty(), 42),
                ("misses_total".into(), Labels::empty(), 8),
            ],
            gauges: Vec::new(),
            histograms: vec![(
                "latency_ms".into(),
                Labels::empty(),
                HistogramSummary {
                    count: 100,
                    min: 1.0,
                    max: 12.0,
                    mean: 4.5,
                    p50: 4.0,
                    p90: 9.0,
                    p99: 11.0,
                    p99_9: 11.5,
                },
            )],
        }
    }

    fn bindings_with_snapshot(snapshot: &MetricsSnapshot) -> CelBindings<'_> {
        CelBindings {
            successes: 100,
            completed: 100,
            failures: 0,
            snapshot,
        }
    }

    #[test]
    fn truthy_expression_returns_true() {
        let snapshot = snapshot_fixture();
        let bindings = bindings_with_snapshot(&snapshot);
        let outcome = evaluate("success_rate >= 0.95", &bindings).expect("evaluate");
        assert!(outcome);
    }

    #[test]
    fn falsy_expression_returns_false() {
        let snapshot = snapshot_fixture();
        let bindings = bindings_with_snapshot(&snapshot);
        let outcome = evaluate("success_rate >= 1.5", &bindings).expect("evaluate");
        assert!(!outcome);
    }

    #[test]
    fn counter_map_resolves_metric_value() {
        let snapshot = snapshot_fixture();
        let bindings = bindings_with_snapshot(&snapshot);
        let outcome = evaluate("counter['hits_total'] == 42", &bindings).expect("evaluate");
        assert!(outcome);
    }

    #[test]
    fn histogram_p99_map_resolves_value() {
        let snapshot = snapshot_fixture();
        let bindings = bindings_with_snapshot(&snapshot);
        let outcome = evaluate("histogram_p99['latency_ms'] < 50.0", &bindings).expect("evaluate");
        assert!(outcome);
    }

    #[test]
    fn invalid_expression_returns_typed_error() {
        let snapshot = snapshot_fixture();
        let bindings = bindings_with_snapshot(&snapshot);
        let outcome = evaluate("not valid &^&* cel", &bindings);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn non_boolean_result_returns_typed_error() {
        let snapshot = snapshot_fixture();
        let bindings = bindings_with_snapshot(&snapshot);
        let outcome = evaluate("'string-not-bool'", &bindings);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }
}
