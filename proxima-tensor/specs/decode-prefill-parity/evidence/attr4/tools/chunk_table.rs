use std::collections::BTreeMap;
use std::env;
use std::fs;

#[derive(Clone, Default)]
struct ChunkRecord {
    chunk: u64,
    ops: f64,
    encode_ms: f64,
    commit_at_ms: f64,
    gpu_start_ms: f64,
    gpu_end_ms: f64,
    commit_to_gpu_start_ms: f64,
    idle_before_ms: f64,
}

#[derive(Default)]
struct StepRecord {
    new_count: u64,
    wall_ms: f64,
    evaluate_ms: f64,
    chunks: Vec<ChunkRecord>,
}

fn value(line: &str, name: &str) -> f64 {
    let marker = format!(" {name}=");
    let Some(at) = line.find(&marker) else { return 0.0 };
    let rest = &line[at + marker.len()..];
    rest[..rest.find(' ').unwrap_or(rest.len())].parse().unwrap_or(0.0)
}

fn parse(path: &str) -> BTreeMap<u64, StepRecord> {
    let text = fs::read_to_string(path).expect("read telemetry log");
    let mut steps: BTreeMap<u64, StepRecord> = BTreeMap::new();
    for line in text.lines() {
        let step = value(line, "step") as u64;
        if line.contains("chunk_record step=") {
            steps.entry(step).or_default().chunks.push(ChunkRecord {
                chunk: value(line, "chunk") as u64,
                ops: value(line, "op_last") - value(line, "op_first") + 1.0,
                encode_ms: value(line, "encode_end_ms") - value(line, "encode_start_ms"),
                commit_at_ms: value(line, "commit_at_ms"),
                gpu_start_ms: value(line, "gpu_start_ms"),
                gpu_end_ms: value(line, "gpu_end_ms"),
                commit_to_gpu_start_ms: value(line, "commit_to_gpu_start_ms"),
                idle_before_ms: value(line, "gpu_idle_before_ms"),
            });
        } else if line.contains("token_breakdown: per-decode-step") {
            let record = steps.entry(step).or_default();
            record.new_count = value(line, "new_count") as u64;
            record.wall_ms = value(line, "step_wall_ms");
            record.evaluate_ms = value(line, "evaluate_ms");
        }
    }
    steps
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    sorted[sorted.len() / 2]
}

fn cov_percent(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0).max(1.0);
    100.0 * variance.sqrt() / mean
}

fn dispatch_counts(path: &str) -> BTreeMap<u64, u64> {
    let mut counts = BTreeMap::new();
    for line in fs::read_to_string(path).expect("read dispatch csv").lines().skip(1) {
        let chunk: u64 = line.split(',').nth(2).expect("chunk column").parse().expect("chunk integer");
        *counts.entry(chunk).or_insert(0) += 1;
    }
    counts
}

fn encode_bound_idle(chunks: &[ChunkRecord]) -> f64 {
    chunks
        .windows(2)
        .filter(|pair| pair[1].commit_at_ms > pair[0].gpu_end_ms)
        .map(|pair| pair[1].idle_before_ms)
        .sum()
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let steps = parse(&arguments[1]);
    let dispatches = dispatch_counts(&arguments[2]);
    let excluded: Vec<u64> = arguments[3].split(',').filter_map(|token| token.parse().ok()).collect();
    let steady: Vec<&StepRecord> = steps
        .iter()
        .filter(|(step, record)| **step >= 2 && !excluded.contains(step) && record.new_count == 1 && !record.chunks.is_empty())
        .map(|(_, record)| record)
        .collect();
    assert!(!steady.is_empty(), "N==0: no steady decode steps");
    println!("steady decode steps used: {} (steps >= 2, excluding {:?}); medians, CoV% in brackets\n", steady.len(), excluded);
    println!("| chunk | ops | dispatches (capture step) | host encode us | encode us/op | gpu busy us | gpu us/dispatch | commit to gpu start us | gpu idle before us | steps where the chunk was committed after the previous chunk's gpu end |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    let chunk_count = steady[0].chunks.len();
    for position in 0..chunk_count {
        let pick = |extract: &dyn Fn(&ChunkRecord) -> f64| -> Vec<f64> { steady.iter().map(|record| extract(&record.chunks[position])).collect() };
        let ops = median(&pick(&|chunk| chunk.ops));
        let encode = pick(&|chunk| chunk.encode_ms * 1e3);
        let busy = pick(&|chunk| (chunk.gpu_end_ms - chunk.gpu_start_ms) * 1e3);
        let count = dispatches.get(&steady[0].chunks[position].chunk).copied().unwrap_or(0);
        let late = steady.iter().filter(|record| position > 0 && record.chunks[position].commit_at_ms > record.chunks[position - 1].gpu_end_ms).count();
        println!(
            "| {} | {ops:.0} | {count} | {:.0} [{:.1}] | {:.2} | {:.0} [{:.1}] | {:.2} | {:.0} | {:.0} | {late} of {} |",
            position + 1,
            median(&encode), cov_percent(&encode), median(&encode) / ops,
            median(&busy), cov_percent(&busy), median(&busy) / count.max(1) as f64,
            median(&pick(&|chunk| chunk.commit_to_gpu_start_ms * 1e3)),
            median(&pick(&|chunk| chunk.idle_before_ms * 1e3)),
            steady.len()
        );
    }
    let per_step = |extract: &dyn Fn(&StepRecord) -> f64| -> Vec<f64> { steady.iter().map(|record| extract(record)).collect() };
    let rows: Vec<(&str, Vec<f64>)> = vec![
        ("step wall ms", per_step(&|record| record.wall_ms)),
        ("evaluate ms", per_step(&|record| record.evaluate_ms)),
        ("sum of host encode windows ms", per_step(&|record| record.chunks.iter().map(|chunk| chunk.encode_ms).sum())),
        ("sum of chunk gpu busy ms", per_step(&|record| record.chunks.iter().map(|chunk| chunk.gpu_end_ms - chunk.gpu_start_ms).sum())),
        ("lead idle (entry to first gpu start) ms", per_step(&|record| record.chunks[0].idle_before_ms)),
        ("inter-chunk gpu idle ms", per_step(&|record| record.chunks.iter().skip(1).map(|chunk| chunk.idle_before_ms).sum())),
        ("  of which chunk committed after previous gpu end (waiting on encode) ms", per_step(&|record| encode_bound_idle(&record.chunks))),
        ("last gpu end ms", per_step(&|record| record.chunks.last().map_or(0.0, |chunk| chunk.gpu_end_ms))),
        ("evaluate minus last gpu end ms (tail on the host)", per_step(&|record| record.evaluate_ms - record.chunks.last().map_or(0.0, |chunk| chunk.gpu_end_ms))),
        ("wall minus evaluate ms (sampling, token feedback)", per_step(&|record| record.wall_ms - record.evaluate_ms)),
    ];
    println!("\n| per step | median ms | CoV % | min | max |");
    println!("|---|---|---|---|---|");
    for (label, values) in &rows {
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        println!("| {label} | {:.3} | {:.2} | {low:.3} | {high:.3} |", median(values), cov_percent(values));
    }
}
