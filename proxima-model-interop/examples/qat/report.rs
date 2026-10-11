use std::collections::BTreeMap;

use super::QatError;
use super::census::{
    NodeLine, PlanOutcome, Verdict, classify_nodes, control_int32_input, fused_verdicts,
};
use super::graph::StepGraph;

fn verdict_text(verdict: &Verdict) -> String {
    match verdict {
        Verdict::Accepted => "ok".to_owned(),
        Verdict::Rejected(reason) => format!("reject:{reason}"),
    }
}

fn print_node_lines(lines: &[NodeLine]) {
    for line in lines {
        println!(
            "node id=%{} op={} side={} metal={}",
            line.node.0,
            line.op,
            line.side,
            verdict_text(&line.verdict)
        );
    }
}

fn print_op_counts(layers: u32, lines: &[NodeLine]) {
    let mut counts: BTreeMap<(&str, bool), usize> = BTreeMap::new();
    for line in lines {
        let accepted = line.verdict == Verdict::Accepted;
        *counts.entry((line.op.as_str(), accepted)).or_default() += 1;
    }
    for ((op, accepted), nodes) in counts {
        let status = if accepted { "metal_ok" } else { "metal_reject" };
        println!("op_census layers={layers} op={op} {status}={nodes}");
    }
}

fn print_refusals(layers: u32, graph: &StepGraph) {
    let mut by_op: BTreeMap<&str, usize> = BTreeMap::new();
    for refusal in &graph.refusals {
        *by_op.entry(refusal.op.as_str()).or_default() += 1;
        println!(
            "autograd_refused layers={layers} node=%{} op={} variant={} operand_map={} reason={}",
            refusal.node.0, refusal.op, refusal.variant, refusal.operand_map, refusal.reason
        );
    }
    for (op, nodes) in by_op {
        println!("autograd_refused_by_op layers={layers} op={op} nodes={nodes}");
    }
}

fn print_fused(layers: u32, graph: &StepGraph) -> Result<(), QatError> {
    let fused = fused_verdicts(graph)?;
    for (node, kind, verdict) in fused.iter().filter(|entry| entry.2 != Verdict::Accepted) {
        let text = verdict_text(verdict);
        println!(
            "fused_node layers={layers} id=%{} kind={kind} metal={text}",
            node.0
        );
    }
    let accepted = fused
        .iter()
        .filter(|entry| entry.2 == Verdict::Accepted)
        .count();
    println!(
        "fused_census layers={layers} bound_ops={} emit_ok={accepted} emit_reject={}",
        fused.len(),
        fused.len() - accepted
    );
    Ok(())
}

fn print_plan(layers: u32, outcome: &PlanOutcome) {
    let state = outcome
        .failure
        .as_deref()
        .map_or("ok".to_owned(), |error| format!("reject:{error}"));
    println!(
        "plan_check layers={layers} dtype_rejected={} plan={state}",
        outcome.rejected.len()
    );
}

fn print_summary(layers: u32, graph: &StepGraph, lines: &[NodeLine]) {
    let accepted = lines
        .iter()
        .filter(|line| line.verdict == Verdict::Accepted)
        .count();
    let refused = graph.refusals.len();
    let (prefix, suffix) = match refused {
        0 => ("census", String::new()),
        count => ("census_partial", format!(" autograd_refused={count}")),
    };
    println!(
        "{prefix} layers={layers} nodes={} node_lines={} adjoint_nodes={} metal_ok={accepted} metal_reject={}{suffix}",
        graph.program.len(),
        lines.len(),
        graph.program.len() - graph.forward_nodes,
        lines.len() - accepted
    );
}

pub fn print_cell(layers: u32, graph: &StepGraph) -> Result<(), QatError> {
    let (lines, outcome) = classify_nodes(graph)?;
    print_node_lines(&lines);
    print_op_counts(layers, &lines);
    print_refusals(layers, graph);
    print_fused(layers, graph)?;
    print_plan(layers, &outcome);
    print_summary(layers, graph, &lines);
    Ok(())
}

pub fn print_control() -> Result<(), QatError> {
    let (nodes, accepted, rejected) = control_int32_input()?;
    println!("control census nodes={nodes} metal_ok={accepted} metal_reject={rejected}");
    Ok(())
}
