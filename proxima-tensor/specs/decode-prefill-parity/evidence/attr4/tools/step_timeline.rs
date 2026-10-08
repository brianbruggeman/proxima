use std::env;
use std::fs;

#[derive(Default, Clone)]
struct Chunk {
    chunk: u64,
    op_first: u64,
    op_last: u64,
    encode_start_ms: f64,
    encode_end_ms: f64,
    gpu_start_ms: f64,
    gpu_end_ms: f64,
    gpu_idle_before_ms: f64,
}

#[derive(Default, Clone)]
struct Phase {
    chunk: u64,
    name: String,
    start_s: f64,
    end_s: f64,
}

#[derive(Default)]
struct StepRecord {
    step: u64,
    new_count: u64,
    wall_ms: f64,
    evaluate_ms: f64,
    prepare_ms: f64,
    pre_encode_ms: f64,
    pipeline_compile_ms: f64,
    pipeline_misses: u64,
    encode_dispatch_ms: f64,
    gpu_exec_ms: f64,
    chunks: Vec<Chunk>,
    phases: Vec<Phase>,
}

struct DispatchRow {
    index: u64,
    chunk_index: u64,
    class: String,
    entry: String,
    extents: String,
}

fn value(line: &str, name: &str) -> f64 {
    let marker = format!(" {name}=");
    let Some(at) = line.find(&marker) else { return 0.0 };
    let rest = &line[at + marker.len()..];
    let end = rest.find(' ').unwrap_or(rest.len());
    rest[..end].parse().unwrap_or(0.0)
}

fn word(line: &str, name: &str) -> String {
    let marker = format!(" {name}=");
    let Some(at) = line.find(&marker) else { return String::new() };
    let rest = &line[at + marker.len()..];
    rest[..rest.find(' ').unwrap_or(rest.len())].to_string()
}

fn parse_log(path: &str) -> Vec<StepRecord> {
    let text = fs::read_to_string(path).expect("read telemetry log");
    let mut records: Vec<StepRecord> = Vec::new();
    let mut pending: Vec<Chunk> = Vec::new();
    let mut phases_by_step: std::collections::BTreeMap<u64, Vec<Phase>> = std::collections::BTreeMap::new();
    for line in text.lines() {
        if line.contains("step_phase step=") || line.contains("chunk_phase step=") {
            phases_by_step.entry(value(line, "step") as u64).or_default().push(Phase {
                chunk: value(line, "chunk") as u64,
                name: word(line, "phase"),
                start_s: value(line, "start_raw_s"),
                end_s: value(line, "end_raw_s"),
            });
        } else if line.contains("chunk_record step=") {
            pending.push(Chunk {
                chunk: value(line, "chunk") as u64,
                op_first: value(line, "op_first") as u64,
                op_last: value(line, "op_last") as u64,
                encode_start_ms: value(line, "encode_start_ms"),
                encode_end_ms: value(line, "encode_end_ms"),
                gpu_start_ms: value(line, "gpu_start_ms"),
                gpu_end_ms: value(line, "gpu_end_ms"),
                gpu_idle_before_ms: value(line, "gpu_idle_before_ms"),
            });
        } else if line.contains("token_breakdown_metal: ") {
            if let Some(record) = records.last_mut() {
                record.prepare_ms = value(line, "prepare_ms");
                record.pre_encode_ms = value(line, "pre_encode_ms");
                record.pipeline_compile_ms = value(line, "pipeline_compile_ms");
                record.pipeline_misses = value(line, "pipeline_misses") as u64;
                record.encode_dispatch_ms = value(line, "encode_dispatch_ms");
                record.gpu_exec_ms = value(line, "gpu_exec_ms");
            }
        } else if line.contains("token_breakdown: per-decode-step") {
            records.push(StepRecord {
                step: value(line, "step") as u64,
                new_count: value(line, "new_count") as u64,
                wall_ms: value(line, "step_wall_ms"),
                evaluate_ms: value(line, "evaluate_ms"),
                chunks: std::mem::take(&mut pending),
                ..StepRecord::default()
            });
        }
    }
    for record in &mut records {
        record.phases = phases_by_step.remove(&record.step).unwrap_or_default();
    }
    records
}

fn parse_dispatches(path: &str) -> Vec<DispatchRow> {
    let text = fs::read_to_string(path).expect("read dispatch csv");
    text.lines()
        .skip(1)
        .map(|line| {
            let fields: Vec<&str> = line.splitn(10, ',').collect();
            DispatchRow {
                index: fields[0].parse().expect("index"),
                chunk_index: fields[2].parse().expect("chunk_index"),
                class: fields[4].to_string(),
                entry: fields[6].to_string(),
                extents: fields[9].to_string(),
            }
        })
        .collect()
}

fn median(values: &mut Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    values[values.len() / 2]
}

fn busy_sum(record: &StepRecord) -> f64 {
    record.chunks.iter().map(|chunk| chunk.gpu_end_ms - chunk.gpu_start_ms).sum()
}

fn inter_chunk_idle(record: &StepRecord) -> f64 {
    record.chunks.iter().skip(1).map(|chunk| chunk.gpu_idle_before_ms).sum()
}

fn lead_idle(record: &StepRecord) -> f64 {
    record.chunks.first().map(|chunk| chunk.gpu_idle_before_ms).unwrap_or(0.0)
}

fn last_gpu_end(record: &StepRecord) -> f64 {
    record.chunks.last().map(|chunk| chunk.gpu_end_ms).unwrap_or(0.0)
}

fn residual(record: &StepRecord) -> f64 {
    record.evaluate_ms - record.prepare_ms - record.pre_encode_ms - last_gpu_end(record)
}

fn short(text: &str) -> String {
    text.chars().take(70).collect()
}

fn dispatch_label(row: &DispatchRow) -> String {
    format!("#{} {} {} {}", row.index, row.class, short(&row.entry), row.extents)
}

fn print_steps(records: &[StepRecord]) {
    println!("| seq | step | new tokens | wall ms | evaluate ms | prepare ms | pre_encode ms | pipeline compile ms (misses) | encode ms | gpu_exec ms | chunks | lead idle ms | inter-chunk idle ms | chunk busy sum ms | last gpu end ms | residual after last gpu end ms (evaluate - prepare - pre_encode - last gpu end) |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for (seq, record) in records.iter().enumerate() {
        println!(
            "| {seq} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} ({}) | {:.3} | {:.3} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            record.step, record.new_count, record.wall_ms, record.evaluate_ms, record.prepare_ms, record.pre_encode_ms,
            record.pipeline_compile_ms, record.pipeline_misses, record.encode_dispatch_ms,
            record.gpu_exec_ms, record.chunks.len(), lead_idle(record), inter_chunk_idle(record),
            busy_sum(record), last_gpu_end(record), residual(record)
        );
    }
}

fn print_decode_median(records: &[StepRecord]) {
    let steady: Vec<&StepRecord> = records.iter().filter(|record| record.new_count == 1 && record.step >= 2 && !record.chunks.is_empty()).collect();
    if steady.is_empty() {
        println!("\nno decode steps >= 2 in this log (N=0)");
        return;
    }
    let pick = |extract: &dyn Fn(&StepRecord) -> f64| median(&mut steady.iter().map(|record| extract(record)).collect());
    println!("\nmedian over {} decode steps >= 2: wall {:.3} ms, evaluate {:.3}, chunk busy sum {:.3}, lead idle {:.3}, inter-chunk idle {:.3}, last gpu end {:.3}, residual after last gpu end {:.3}",
        steady.len(),
        pick(&|record| record.wall_ms), pick(&|record| record.evaluate_ms), pick(&|record| busy_sum(record)),
        pick(&|record| lead_idle(record)), pick(&|record| inter_chunk_idle(record)),
        pick(&|record| last_gpu_end(record)), pick(&|record| residual(record)));
}

fn print_detail(record: &StepRecord, seq: usize, dispatches: &[DispatchRow]) {
    println!("\ndetail seq {seq} (step {}): wall {:.3} ms, evaluate {:.3} ms, chunk busy sum {:.3} ms, last gpu end {:.3} ms\n", record.step, record.wall_ms, record.evaluate_ms, busy_sum(record), last_gpu_end(record));
    println!("| chunk | ops | dispatches (census index) | encode start-end ms | gpu start-end ms | gpu busy ms | idle before ms | dispatch before the idle | dispatch after the idle |");
    println!("|---|---|---|---|---|---|---|---|---|");
    let mut gaps: Vec<(f64, String)> = Vec::new();
    for (position, chunk) in record.chunks.iter().enumerate() {
        let members: Vec<&DispatchRow> = dispatches.iter().filter(|row| row.chunk_index == chunk.chunk).collect();
        let first = members.first();
        let last_of_previous = if position == 0 {
            None
        } else {
            dispatches.iter().filter(|row| row.chunk_index == record.chunks[position - 1].chunk).last()
        };
        let before = last_of_previous.map(dispatch_label).unwrap_or_else(|| "none (host: prepare, plan, encode)".to_string());
        let after = first.map(|row| dispatch_label(row)).unwrap_or_default();
        println!("| {} | {}-{} | {} ({}-{}) | {:.3}-{:.3} | {:.3}-{:.3} | {:.3} | {:.3} | {before} | {after} |",
            chunk.chunk, chunk.op_first, chunk.op_last, members.len(),
            first.map(|row| row.index).unwrap_or(0), members.last().map(|row| row.index).unwrap_or(0),
            chunk.encode_start_ms, chunk.encode_end_ms, chunk.gpu_start_ms, chunk.gpu_end_ms,
            chunk.gpu_end_ms - chunk.gpu_start_ms, chunk.gpu_idle_before_ms);
        gaps.push((chunk.gpu_idle_before_ms, format!("before chunk {}: after {before}; before {after}", chunk.chunk)));
    }
    gaps.push((residual(record), "residual: evaluate minus prepare, pre_encode and the last chunk's gpu end (readback, kv append, host)".to_string()));
    gaps.sort_by(|left, right| right.0.partial_cmp(&left.0).expect("finite"));
    println!("\nlargest host gaps in this step, ranked:\n");
    for (rank, (gap, label)) in gaps.iter().take(6).enumerate() {
        println!("{}. {:.3} ms {label}", rank + 1, gap);
    }
}

fn phase_edge(record: &StepRecord, chunk: u64, name: &str, end: bool) -> Option<f64> {
    record
        .phases
        .iter()
        .find(|phase| phase.chunk == chunk && phase.name == name)
        .map(|phase| if end { phase.end_s } else { phase.start_s })
}

fn print_phases(record: &StepRecord) {
    if record.phases.is_empty() {
        println!("\nno step_phase or chunk_phase events in this step (N=0); capture with a file-sink exporter on the instrument build");
        return;
    }
    let origin = record.phases.iter().map(|phase| phase.start_s).fold(f64::INFINITY, f64::min);
    let mut ordered = record.phases.clone();
    ordered.sort_by(|left, right| left.start_s.partial_cmp(&right.start_s).expect("finite"));
    println!("\nhost phases, ms from the earliest phase start (step_phase has chunk 0):\n");
    println!("| chunk | phase | start ms | end ms | duration ms |");
    println!("|---|---|---|---|---|");
    for phase in &ordered {
        println!("| {} | {} | {:.3} | {:.3} | {:.3} |", phase.chunk, phase.name, (phase.start_s - origin) * 1e3, (phase.end_s - origin) * 1e3, (phase.end_s - phase.start_s) * 1e3);
    }
    println!("\ncommit-to-GPU-start split per chunk (commit end to scheduled callback, scheduled callback to GPU start):\n");
    println!("| chunk | commit end to scheduled ms | scheduled to gpu start ms |");
    println!("|---|---|---|");
    let chunks: Vec<u64> = record.phases.iter().filter(|phase| phase.name == "commit").map(|phase| phase.chunk).collect();
    for chunk in chunks {
        let queued = phase_edge(record, chunk, "scheduled", false).zip(phase_edge(record, chunk, "commit", true)).map(|(scheduled, committed)| (scheduled - committed) * 1e3);
        let started = phase_edge(record, chunk, "gpu", false).zip(phase_edge(record, chunk, "scheduled", false)).map(|(gpu, scheduled)| (gpu - scheduled) * 1e3);
        println!("| {chunk} | {} | {} |", queued.map_or("missing".to_string(), |ms| format!("{ms:.3}")), started.map_or("missing".to_string(), |ms| format!("{ms:.3}")));
    }
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    assert!(arguments.len() >= 4, "usage: step_timeline <telemetry.log> <census_dispatches.csv> <detail seq>");
    let records = parse_log(&arguments[1]);
    let dispatches = parse_dispatches(&arguments[2]);
    let detail_seq: usize = arguments[3].parse().expect("detail seq");
    assert!(!records.is_empty(), "N==0 step records in {}", arguments[1]);
    println!("telemetry {}: {} step records, {} dispatch rows\n", arguments[1], records.len(), dispatches.len());
    print_steps(&records);
    print_decode_median(&records);
    print_detail(&records[detail_seq], detail_seq, &dispatches);
    print_phases(&records[detail_seq]);
}
