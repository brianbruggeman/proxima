use std::env;
use std::fs;

#[derive(Default, Clone)]
struct Phase {
    name: String,
    chunk: u64,
    start_s: f64,
    end_s: f64,
}

fn value(line: &str, name: &str) -> f64 {
    let marker = format!(" {name}=");
    let Some(at) = line.find(&marker) else { return 0.0 };
    let rest = &line[at + marker.len()..];
    rest[..rest.find(' ').unwrap_or(rest.len())].parse().unwrap_or(0.0)
}

fn word(line: &str, name: &str) -> String {
    let marker = format!(" {name}=");
    let Some(at) = line.find(&marker) else { return String::new() };
    let rest = &line[at + marker.len()..];
    rest[..rest.find(' ').unwrap_or(rest.len())].to_string()
}

fn find<'a>(phases: &'a [Phase], name: &str, chunk: u64) -> Option<&'a Phase> {
    phases.iter().find(|phase| phase.name == name && phase.chunk == chunk)
}

fn main() {
    let text = fs::read_to_string(env::args().nth(1).expect("telemetry log path")).expect("read log");
    let mut step0: Vec<Phase> = Vec::new();
    let mut compile: Vec<(u64, f64)> = Vec::new();
    let mut new_counts: Vec<u64> = Vec::new();
    for line in text.lines() {
        if (line.contains("step_phase step=0 ") || line.contains("chunk_phase step=0 ")) && line.contains("start_raw_s=") {
            step0.push(Phase {
                name: word(line, "phase"),
                chunk: value(line, "chunk") as u64,
                start_s: value(line, "start_raw_s"),
                end_s: value(line, "end_raw_s"),
            });
        } else if line.contains("token_breakdown_metal: ") && line.contains(" step=0 ") {
            compile.push((value(line, "pipeline_misses") as u64, value(line, "pipeline_compile_ms")));
        } else if line.contains("token_breakdown: per-decode-step") && line.contains(" step=0 ") {
            new_counts.push(value(line, "new_count") as u64);
        }
    }
    let mut prepares: Vec<f64> = step0
        .iter()
        .filter(|phase| phase.name == "prepare")
        .map(|phase| phase.start_s)
        .collect();
    prepares.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    assert!(!prepares.is_empty(), "N==0: no step 0 prepare events");
    println!("| generation | new tokens | pipeline misses (compile ms) | prepare ms | pre_encode ms | encode ms | commit ms | commit end to scheduled ms | scheduled to gpu start ms | gpu ms | commit end to gpu end ms |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for (generation, begin) in prepares.iter().enumerate() {
        let end = prepares.get(generation + 1).copied().unwrap_or(f64::INFINITY);
        let mine: Vec<Phase> = step0.iter().filter(|phase| phase.start_s >= *begin && phase.start_s < end).cloned().collect();
        let span = |name: &str, chunk: u64| find(&mine, name, chunk).map_or(f64::NAN, |phase| (phase.end_s - phase.start_s) * 1e3);
        let gap = |from: &str, to: &str| match (find(&mine, from, 1), find(&mine, to, 1)) {
            (Some(left), Some(right)) => (right.start_s - left.end_s) * 1e3,
            _ => f64::NAN,
        };
        let commit_to_end = match (find(&mine, "commit", 1), find(&mine, "gpu", 1)) {
            (Some(left), Some(right)) => (right.end_s - left.end_s) * 1e3,
            _ => f64::NAN,
        };
        let (misses, compile_ms) = compile.get(generation).copied().unwrap_or((0, f64::NAN));
        println!(
            "| {generation} | {} | {misses} ({compile_ms:.1}) | {:.2} | {:.2} | {:.2} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} |",
            new_counts.get(generation).copied().unwrap_or(0),
            span("prepare", 0), span("pre_encode", 0), span("encode", 1), span("commit", 1),
            gap("commit", "scheduled"), gap("scheduled", "gpu"), span("gpu", 1), commit_to_end
        );
    }
}
