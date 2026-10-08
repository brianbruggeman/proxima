// Reads one instrumented decode_gbps_baseline telemetry log taken with PROXIMA_DISPATCH=concurrent and
// prints, per decode step, the barriers the hazard tracker fired and why.
//
// build: rustc -O hazard_table.rs -o hazard_table
// usage: hazard_table <label> <telemetry.log> [step ...]    (no steps: summary line for every step)
//
// A step's events are the `hazard_op` / `hazard_internal_barrier` lines that precede that step's
// `token_breakdown_metal` line. An internal barrier belongs to the most recent `hazard_op` (the op whose
// encode fired it). A run is consecutive hazard ops with no barrier before them and none inside them; an op
// with an internal barrier ends the run it is in.

use std::collections::BTreeMap;
use std::env;
use std::fs;

fn field<'line>(line: &'line str, key: &str) -> Option<&'line str> {
    let needle = format!(" {key}=");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find(' ').unwrap_or(rest.len());
    Some(&rest[..end])
}

fn number(line: &str, key: &str) -> Option<u64> {
    field(line, key)?.parse().ok()
}

#[derive(Default)]
struct StepEvents {
    ops: u64,
    op_barriers: BTreeMap<String, u64>,
    internal_barriers: BTreeMap<String, u64>,
    longest_run_ops: u64,
    runs_ge_8: u64,
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let label = &arguments[1];
    let text = fs::read_to_string(&arguments[2]).expect("read telemetry log");
    let wanted: Vec<u64> = arguments[3..].iter().filter_map(|value| value.parse().ok()).collect();

    let mut current = StepEvents::default();
    let mut run: u64 = 0;
    let mut totals: Vec<(u64, StepEvents, u64, u64, u64, u64, u64, u64)> = Vec::new();

    for line in text.lines() {
        if line.contains(" hazard_op site=op") {
            current.ops += 1;
            let reason = field(line, "reason").unwrap_or("?").to_string();
            if reason == "none" {
                run += 1;
            } else {
                *current.op_barriers.entry(reason).or_default() += 1;
                if run >= 8 {
                    current.runs_ge_8 += 1;
                }
                current.longest_run_ops = current.longest_run_ops.max(run);
                run = 1;
            }
            current.longest_run_ops = current.longest_run_ops.max(run);
        } else if line.contains(" hazard_internal_barrier site=") {
            let site = field(line, "site").unwrap_or("?").to_string();
            *current.internal_barriers.entry(site).or_default() += 1;
            if run >= 8 {
                current.runs_ge_8 += 1;
            }
            current.longest_run_ops = current.longest_run_ops.max(run);
            run = 0;
        } else if line.contains("token_breakdown_metal:") {
            let step = number(line, "step").unwrap_or(0);
            let finished = std::mem::take(&mut current);
            totals.push((
                step,
                finished,
                number(line, "barriers").unwrap_or(0),
                number(line, "barriers_raw").unwrap_or(0),
                number(line, "barriers_waw").unwrap_or(0),
                number(line, "barriers_war").unwrap_or(0),
                number(line, "physical_dispatch_calls").unwrap_or(0),
                number(line, "encode_dispatch_calls").unwrap_or(0),
            ));
            run = 0;
        }
    }

    println!("cell {label} steps_recorded={}", totals.len());
    println!("step | hazard_ops | op_barriers | internal_barriers | counter barriers | counter raw/waw/war | physical_dispatch_calls | encode_dispatch_calls | longest_run_ops | runs>=8");
    for (step, events, barriers, raw, waw, war, physical, encode) in &totals {
        if !wanted.is_empty() && !wanted.contains(step) {
            continue;
        }
        let op_total: u64 = events.op_barriers.values().sum();
        let internal_total: u64 = events.internal_barriers.values().sum();
        println!(
            "{step} | {} | {op_total} | {internal_total} | {barriers} | {raw}/{waw}/{war} | {physical} | {encode} | {} | {}",
            events.ops, events.longest_run_ops, events.runs_ge_8
        );
        if !wanted.is_empty() {
            for (reason, count) in &events.op_barriers {
                println!("  op_barrier reason={reason} count={count}");
            }
            for (site, count) in &events.internal_barriers {
                println!("  internal_barrier site={site} count={count}");
            }
        }
    }
}
