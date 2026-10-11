use std::collections::BTreeMap;
use std::env;
use std::fs;

#[derive(Default, Clone)]
struct Phase {
    step: u64,
    chunk: u64,
    name: String,
    start_s: f64,
    end_s: f64,
}

#[derive(Default, Clone)]
struct ChunkRecord {
    chunk: u64,
    op_first: u64,
    op_last: u64,
}

#[derive(Default, Clone)]
struct Execution {
    step: u64,
    plan_shape: String,
    chunk_count: u64,
    entry_s: f64,
    wait_returned_ms: f64,
    exit_ms: f64,
    new_count: u64,
    wall_ms: f64,
    evaluate_ms: f64,
    outside_ms: [f64; 5],
    named_phases: [f64; 4],
    pipeline_misses: u64,
    pipeline_compile_ms: f64,
    phases: Vec<Phase>,
    records: Vec<ChunkRecord>,
    generation: usize,
}

struct Chunk {
    id: u64,
    op_first: u64,
    op_last: u64,
    encode: (f64, f64),
    commit: (f64, f64),
    scheduled: f64,
    gpu: (f64, f64),
}

fn value(line: &str, name: &str) -> f64 {
    let marker = format!(" {name}=");
    let Some(at) = line.find(&marker) else { return f64::NAN };
    let rest = &line[at + marker.len()..];
    let end = rest.find(' ').unwrap_or(rest.len());
    rest[..end].parse().unwrap_or(f64::NAN)
}

fn word(line: &str, name: &str) -> String {
    let marker = format!(" {name}=");
    let Some(at) = line.find(&marker) else { return String::new() };
    let rest = &line[at + marker.len()..];
    rest[..rest.find(' ').unwrap_or(rest.len())].to_string()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|value| value.is_finite()).collect();
    assert!(!sorted.is_empty(), "median over N==0 values");
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    sorted[sorted.len() / 2]
}

fn cov_percent(values: &[f64]) -> f64 {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    if mean == 0.0 || values.len() < 2 {
        return 0.0;
    }
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (count - 1.0);
    variance.sqrt() / mean.abs() * 100.0
}

fn parse(path: &str) -> Vec<Execution> {
    let text = fs::read_to_string(path).expect("read telemetry log");
    let mut host: Vec<Execution> = Vec::new();
    let mut walls: Vec<(u64, u64, f64, f64, [f64; 5])> = Vec::new();
    let mut pending_named: [f64; 4] = [0.0; 4];
    let mut metals: Vec<(u64, u64, f64)> = Vec::new();
    let mut phases: Vec<Phase> = Vec::new();
    let mut groups: Vec<Vec<ChunkRecord>> = Vec::new();
    let mut groups_steps: Vec<u64> = Vec::new();
    for line in text.lines() {
        if line.contains(" named_blocks_phase") {
            let slot = match word(line, "phase").as_str() {
                "weights_loop" => 0,
                "device_kv_adopt" => 1,
                "push_step_named_blocks" => 2,
                "greedy_program" => 3,
                other => panic!("unknown named_blocks_phase `{other}`"),
            };
            pending_named[slot] = value(line, "ms");
        } else if line.contains(" step_host_timeline step=") {
            host.push(Execution {
                named_phases: std::mem::replace(&mut pending_named, [0.0; 4]),
                step: value(line, "step") as u64,
                plan_shape: word(line, "plan_shape"),
                chunk_count: value(line, "chunks") as u64,
                entry_s: value(line, "entry_raw_s"),
                wait_returned_ms: value(line, "wait_returned_ms"),
                exit_ms: value(line, "exit_ms"),
                ..Execution::default()
            });
        } else if line.contains("token_breakdown: per-decode-step") {
            walls.push((value(line, "step") as u64, value(line, "new_count") as u64, value(line, "step_wall_ms"), value(line, "evaluate_ms"), [value(line, "apply_serving_config_ms"), value(line, "build_position_inputs_ms"), value(line, "named_blocks_weights_ms"), value(line, "layer_cache_append_ms"), value(line, "greedy_pick_ms")]));
        } else if line.contains("token_breakdown_metal: ") {
            metals.push((value(line, "step") as u64, value(line, "pipeline_misses") as u64, value(line, "pipeline_compile_ms")));
        } else if line.contains(" step_phase step=") || line.contains(" chunk_phase step=") {
            let chunk = value(line, "chunk");
            phases.push(Phase {
                step: value(line, "step") as u64,
                chunk: if chunk.is_nan() { 0 } else { chunk as u64 },
                name: word(line, "phase"),
                start_s: value(line, "start_raw_s"),
                end_s: value(line, "end_raw_s"),
            });
        } else if line.contains(" chunk_record step=") {
            let record = ChunkRecord { chunk: value(line, "chunk") as u64, op_first: value(line, "op_first") as u64, op_last: value(line, "op_last") as u64 };
            if record.chunk == 1 {
                groups.push(Vec::new());
                groups_steps.push(value(line, "step") as u64);
            }
            groups.last_mut().expect("chunk_record before chunk 1").push(record);
        }
    }
    assert!(!host.is_empty(), "N==0 step_host_timeline events in {path}");
    assert_eq!(host.len(), walls.len(), "step_host_timeline count {} != token_breakdown count {} in {path}", host.len(), walls.len());
    assert_eq!(host.len(), metals.len(), "token_breakdown_metal count differs in {path}");
    assert_eq!(host.len(), groups.len(), "chunk_record group count {} != executions {} in {path}", groups.len(), host.len());
    let mut generation = 0usize;
    let mut seen_step_zero = false;
    for (index, execution) in host.iter_mut().enumerate() {
        assert_eq!(execution.step, walls[index].0, "step mismatch at execution {index}");
        assert_eq!(execution.step, groups_steps[index], "chunk_record step mismatch at execution {index}");
        if execution.step == 0 {
            if seen_step_zero {
                generation += 1;
            }
            seen_step_zero = true;
        }
        execution.generation = generation;
        execution.new_count = walls[index].1;
        execution.wall_ms = walls[index].2;
        execution.evaluate_ms = walls[index].3;
        execution.outside_ms = walls[index].4;
        execution.pipeline_misses = metals[index].1;
        execution.pipeline_compile_ms = metals[index].2;
        execution.records = groups[index].clone();
    }
    let mut by_step: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for (index, execution) in host.iter().enumerate() {
        by_step.entry(execution.step).or_default().push(index);
    }
    let mut unassigned = 0usize;
    for phase in phases {
        let Some(candidates) = by_step.get(&phase.step) else {
            unassigned += 1;
            continue;
        };
        let target = if phase.name == "prepare" {
            candidates.iter().copied().filter(|index| host[*index].entry_s >= phase.end_s - 1e-4).min_by(|left, right| host[*left].entry_s.partial_cmp(&host[*right].entry_s).expect("finite"))
        } else {
            candidates.iter().copied().filter(|index| host[*index].entry_s <= phase.start_s + 1e-6).max_by(|left, right| host[*left].entry_s.partial_cmp(&host[*right].entry_s).expect("finite"))
        };
        match target {
            Some(index) => host[index].phases.push(phase),
            None => unassigned += 1,
        }
    }
    eprintln!("{path}: {} executions, {} generations, {} phase events unassigned", host.len(), generation + 1, unassigned);
    host
}

fn find(execution: &Execution, chunk: u64, name: &str) -> Option<(f64, f64)> {
    execution.phases.iter().find(|phase| phase.chunk == chunk && phase.name == name).map(|phase| (phase.start_s, phase.end_s))
}

fn chunks_of(execution: &Execution) -> Vec<Chunk> {
    (1..=execution.chunk_count)
        .map(|id| {
            let record = execution.records.iter().find(|record| record.chunk == id).cloned().unwrap_or_default();
            let missing = |name: &str| panic!("step {} gen {} chunk {id}: no {name} phase event (N==0)", execution.step, execution.generation);
            Chunk {
                id,
                op_first: record.op_first,
                op_last: record.op_last,
                encode: find(execution, id, "encode").unwrap_or_else(|| missing("encode")),
                commit: find(execution, id, "commit").unwrap_or_else(|| missing("commit")),
                scheduled: find(execution, id, "scheduled").unwrap_or_else(|| missing("scheduled")).0,
                gpu: find(execution, id, "gpu").unwrap_or_else(|| missing("gpu")),
            }
        })
        .collect()
}

#[derive(Default, Clone)]
struct Row {
    values: BTreeMap<&'static str, f64>,
    largest_gap: String,
}

const COLUMNS: [&str; 33] = [
    "wall", "evaluate", "outside_evaluate", "apply_serving_config", "build_position_inputs", "named_blocks_span", "nb_weights_loop", "nb_device_kv_adopt", "nb_push_named_blocks", "nb_greedy_program", "append_plus_greedy", "outside_unattributed", "prepare", "gap_after_prepare", "pre_encode", "pipeline_compile", "encode_sum", "encode_to_commit", "commit_call_sum", "chunk1_queue",
    "chunk1_sched_to_gpu", "gpu_busy_sum", "inter_chunk_idle", "lead_idle_entry_to_gpu", "lead_idle_prepare_start_to_gpu", "tail", "post_wait_to_exit", "in_evaluate_unattributed",
    "chunk1_ops", "chunks", "busy_over_wall_pct", "lead_unattributed", "pipeline_misses",
];

fn derive(execution: &Execution) -> Row {
    let chunks = chunks_of(execution);
    let first = &chunks[0];
    let (prepare_start, prepare_end) = find(execution, 0, "prepare").unwrap_or((execution.entry_s, execution.entry_s));
    let (pre_encode_start, pre_encode_end) = find(execution, 0, "pre_encode").unwrap_or_else(|| panic!("step {} gen {}: no pre_encode phase event (N==0)", execution.step, execution.generation));
    let ms = |seconds: f64| seconds * 1e3;
    let busy: f64 = chunks.iter().map(|chunk| ms(chunk.gpu.1 - chunk.gpu.0)).sum();
    let inter: f64 = chunks.windows(2).map(|pair| ms(pair[1].gpu.0 - pair[0].gpu.1)).sum();
    let last_end = chunks.last().expect("chunks").gpu.1;
    let encode_sum: f64 = chunks.iter().map(|chunk| ms(chunk.encode.1 - chunk.encode.0)).sum();
    let commit_sum: f64 = chunks.iter().map(|chunk| ms(chunk.commit.1 - chunk.commit.0)).sum();
    let prepare = ms(prepare_end - prepare_start);
    let gap_after_prepare = ms(pre_encode_start - prepare_end);
    let pre_encode = ms(pre_encode_end - pre_encode_start);
    let encode_to_commit = ms(first.commit.0 - first.encode.1);
    let queue = ms(first.scheduled - first.commit.1);
    let sched_to_gpu = ms(first.gpu.0 - first.scheduled);
    let lead_from_prepare = ms(first.gpu.0 - prepare_start);
    let lead_attributed = prepare + gap_after_prepare + ms(first.encode.0 - pre_encode_end) + pre_encode + ms(first.encode.1 - first.encode.0) + encode_to_commit + ms(first.commit.1 - first.commit.0) + queue + sched_to_gpu;
    let tail = ms(execution.entry_s + execution.wait_returned_ms / 1e3 - last_end);
    let post_wait = execution.exit_ms - execution.wait_returned_ms;
    let host_span = ms(execution.entry_s + execution.exit_ms / 1e3 - prepare_start);
    let mut row = Row::default();
    let mut put = |name: &'static str, number: f64| {
        row.values.insert(name, number);
    };
    put("wall", execution.wall_ms);
    put("evaluate", execution.evaluate_ms);
    put("outside_evaluate", execution.wall_ms - execution.evaluate_ms);
    put("apply_serving_config", execution.outside_ms[0]);
    put("build_position_inputs", execution.outside_ms[1]);
    put("named_blocks_span", execution.outside_ms[2]);
    put("nb_weights_loop", execution.named_phases[0]);
    put("nb_device_kv_adopt", execution.named_phases[1]);
    put("nb_push_named_blocks", execution.named_phases[2]);
    put("nb_greedy_program", execution.named_phases[3]);
    put("append_plus_greedy", execution.outside_ms[3] + execution.outside_ms[4]);
    put("outside_unattributed", execution.wall_ms - execution.evaluate_ms - execution.outside_ms.iter().sum::<f64>());
    put("prepare", prepare);
    put("gap_after_prepare", gap_after_prepare);
    put("pre_encode", pre_encode);
    put("pipeline_compile", execution.pipeline_compile_ms);
    put("encode_sum", encode_sum);
    put("encode_to_commit", encode_to_commit);
    put("commit_call_sum", commit_sum);
    put("chunk1_queue", queue);
    put("chunk1_sched_to_gpu", sched_to_gpu);
    put("gpu_busy_sum", busy);
    put("inter_chunk_idle", inter);
    put("lead_idle_entry_to_gpu", ms(first.gpu.0 - execution.entry_s));
    put("lead_idle_prepare_start_to_gpu", lead_from_prepare);
    put("tail", tail);
    put("post_wait_to_exit", post_wait);
    put("in_evaluate_unattributed", execution.evaluate_ms - host_span);
    put("chunk1_ops", (first.op_last - first.op_first + 1) as f64);
    put("chunks", chunks.len() as f64);
    put("busy_over_wall_pct", busy / execution.wall_ms * 100.0);
    put("lead_unattributed", lead_from_prepare - lead_attributed);
    put("pipeline_misses", execution.pipeline_misses as f64);
    row.largest_gap = largest_gap(execution, &chunks, prepare_start, tail);
    row
}

fn largest_gap(execution: &Execution, chunks: &[Chunk], prepare_start: f64, tail: f64) -> String {
    let mut gaps: Vec<(f64, String)> = vec![(
        (chunks[0].gpu.0 - prepare_start) * 1e3,
        format!("lead: prepare start to GPU start of chunk 1 (ops {}-{})", chunks[0].op_first, chunks[0].op_last),
    )];
    for pair in chunks.windows(2) {
        gaps.push((
            (pair[1].gpu.0 - pair[0].gpu.1) * 1e3,
            format!("between chunk {} (ops {}-{}) and chunk {} (ops {}-{})", pair[0].id, pair[0].op_first, pair[0].op_last, pair[1].id, pair[1].op_first, pair[1].op_last),
        ));
    }
    let last = chunks.last().expect("chunks");
    gaps.push((tail, format!("tail: GPU end of chunk {} (ops {}-{}) to wait return", last.id, last.op_first, last.op_last)));
    let _ = execution;
    gaps.sort_by(|left, right| right.0.partial_cmp(&left.0).expect("finite"));
    format!("{:.3} ms {}", gaps[0].0, gaps[0].1)
}

fn print_rows(path: &str, executions: &[Execution]) {
    println!("\n### raw per execution: {path}\n");
    println!("| gen | step | shape | new tokens | {} | largest GPU-idle gap |", COLUMNS.join(" | "));
    println!("|{}|", "---|".repeat(COLUMNS.len() + 5));
    for execution in executions {
        let row = derive(execution);
        let cells: Vec<String> = COLUMNS.iter().map(|name| format!("{:.3}", row.values[name])).collect();
        println!("| {} | {} | {} | {} | {} | {} |", execution.generation + 1, execution.step, execution.plan_shape, execution.new_count, cells.join(" | "), row.largest_gap);
    }
}

fn select<'a>(executions: &'a [Execution], prefill: bool, generations: &dyn Fn(usize) -> bool) -> Vec<&'a Execution> {
    executions
        .iter()
        .filter(|execution| generations(execution.generation))
        .filter(|execution| if prefill { execution.step == 0 } else { execution.step >= 2 && execution.new_count == 1 })
        .collect()
}

fn aggregate(per_process: &[Vec<f64>]) -> String {
    let process_medians: Vec<f64> = per_process.iter().map(|values| median(values)).collect();
    let low = process_medians.iter().copied().fold(f64::INFINITY, f64::min);
    let high = process_medians.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    format!("{:.3} [CoV {:.1}%] ({:.3}-{:.3})", median(&process_medians), cov_percent(&process_medians), low, high)
}

fn print_summary(title: &str, logs: &[Vec<Execution>], prefill: bool, generations: &dyn Fn(usize) -> bool) {
    println!("\n### {title}\n");
    let per_log: Vec<Vec<Row>> = logs.iter().map(|executions| select(executions, prefill, generations).into_iter().map(derive).collect()).collect();
    let counts: Vec<usize> = per_log.iter().map(Vec::len).collect();
    println!("N executions per process: {counts:?} (median of the per-process medians, CoV and range over the {} processes)\n", logs.len());
    assert!(counts.iter().all(|count| *count > 0), "N==0 executions selected for {title}");
    println!("| quantity (ms unless named) | {} |", "median [CoV] (min-max)");
    println!("|---|---|");
    for name in COLUMNS {
        let series: Vec<Vec<f64>> = per_log.iter().map(|rows| rows.iter().map(|row| row.values[name]).collect()).collect();
        println!("| {name} | {} |", aggregate(&series));
    }
}

fn main() {
    let mut paths: Vec<String> = env::args().skip(1).collect();
    if paths.first().is_some_and(|first| first == "--resources-only") {
        paths.remove(0);
        print_resources(&paths);
        return;
    }
    assert!(!paths.is_empty(), "usage: warm_timeline <telemetry.log>...");
    let logs: Vec<Vec<Execution>> = paths.iter().map(|path| parse(path)).collect();
    for (path, executions) in paths.iter().zip(&logs) {
        print_rows(path, executions);
    }
    print_summary("prefill step 0, generation 1 (cold)", &logs, true, &|generation| generation == 0);
    print_summary("prefill step 0, generations 2 to 5 (warm)", &logs, true, &|generation| generation >= 1);
    print_summary("decode steps >= 2, generation 1", &logs, false, &|generation| generation == 0);
    print_summary("decode steps >= 2, generations 2 to 5", &logs, false, &|generation| generation >= 1);
    print_chunks("prefill step 0", &logs, true);
    print_chunks("decode steps >= 2", &logs, false);
    print_resources(&paths);
}

struct RunDone {
    ttft_ms: f64,
    wall_ms: f64,
    cpu_pct: f64,
    steady_rss_mb: f64,
    steady_footprint_mb: f64,
    gpu_peak_mb: f64,
    text_hash: String,
}

fn time_l_field(stderr: &str, label: &str) -> f64 {
    stderr
        .lines()
        .find(|line| line.contains(label))
        .and_then(|line| line.split_whitespace().next())
        .and_then(|field| field.parse::<f64>().ok())
        .map_or(f64::NAN, |bytes| bytes / 1048576.0)
}

fn runs_done(path: &str) -> Vec<RunDone> {
    let text = fs::read_to_string(path).expect("read telemetry log");
    text.lines()
        .filter(|line| line.contains("decode_gbps_baseline run=done"))
        .map(|line| RunDone {
            ttft_ms: value(line, "ttft_ms"),
            wall_ms: value(line, "wall_ms"),
            cpu_pct: value(line, "cpu_pct"),
            steady_rss_mb: value(line, "steady_rss_bytes") / 1048576.0,
            steady_footprint_mb: value(line, "steady_footprint_bytes") / 1048576.0,
            gpu_peak_mb: value(line, "gpu_peak_bytes") / 1048576.0,
            text_hash: word(line, "text_hash"),
        })
        .collect()
}

fn print_resources(paths: &[String]) {
    println!("\n### resources per process (telemetry `run=done` records and `/usr/bin/time -l` of the same process)\n");
    println!("| process | gen 1 ttft ms | gens 2-5 ttft ms median (min-max) | gens 2-5 wall ms median | gens 2-5 cpu % median | steady rss MB | steady footprint MB | gpu peak MB | time -l max rss MB | time -l peak footprint MB | text hashes (distinct) |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    let mut warm_ttft: Vec<Vec<f64>> = Vec::new();
    for path in paths {
        let runs = runs_done(path);
        assert!(runs.len() >= 2, "N<2 run=done records in {path}");
        let warm: Vec<&RunDone> = runs.iter().skip(1).collect();
        let pick = |extract: &dyn Fn(&RunDone) -> f64| median(&warm.iter().map(|run| extract(run)).collect::<Vec<f64>>());
        let ttfts: Vec<f64> = warm.iter().map(|run| run.ttft_ms).collect();
        warm_ttft.push(ttfts.clone());
        let stderr = fs::read_to_string(path.replace("telemetry.log", "stderr.log")).unwrap_or_default();
        let mut hashes: Vec<&str> = runs.iter().map(|run| run.text_hash.as_str()).collect();
        hashes.sort_unstable();
        hashes.dedup();
        println!(
            "| {path} | {:.0} | {:.0} ({:.0}-{:.0}) | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {} |",
            runs[0].ttft_ms,
            median(&ttfts),
            ttfts.iter().copied().fold(f64::INFINITY, f64::min),
            ttfts.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            pick(&|run| run.wall_ms),
            pick(&|run| run.cpu_pct),
            pick(&|run| run.steady_rss_mb),
            pick(&|run| run.steady_footprint_mb),
            pick(&|run| run.gpu_peak_mb),
            time_l_field(&stderr, "maximum resident set size"),
            time_l_field(&stderr, "peak memory footprint"),
            hashes.join(",")
        );
    }
    println!("\nwarm ttft (gens 2-5) pooled over processes: median {:.1} ms, CoV of the per-process medians {:.2}%", median(&warm_ttft.concat()), cov_percent(&warm_ttft.iter().map(|series| median(series)).collect::<Vec<f64>>()));
}

fn print_chunks(title: &str, logs: &[Vec<Execution>], prefill: bool) {
    let selected: Vec<Vec<&Execution>> = logs.iter().map(|executions| select(executions, prefill, &|generation| generation >= 1)).collect();
    let chunk_count = selected[0][0].chunk_count as usize;
    println!("\n### {title}: per chunk, generations 2 to 5, median of the per-process medians [CoV] (min-max), ms\n");
    println!("| chunk | ops | commit end to scheduled | scheduled to GPU start | GPU busy | GPU end of previous chunk to this GPU start | encode |");
    println!("|---|---|---|---|---|---|---|");
    for index in 0..chunk_count {
        let per_process = |extract: &dyn Fn(&[Chunk], usize) -> f64| -> Vec<Vec<f64>> {
            selected.iter().map(|executions| executions.iter().map(|execution| extract(&chunks_of(execution), index)).collect()).collect()
        };
        let ops: Vec<Vec<f64>> = per_process(&|chunks, at| (chunks[at].op_last - chunks[at].op_first + 1) as f64);
        let queue = per_process(&|chunks, at| (chunks[at].scheduled - chunks[at].commit.1) * 1e3);
        let start = per_process(&|chunks, at| (chunks[at].gpu.0 - chunks[at].scheduled) * 1e3);
        let busy = per_process(&|chunks, at| (chunks[at].gpu.1 - chunks[at].gpu.0) * 1e3);
        let gap = per_process(&|chunks, at| if at == 0 { f64::NAN } else { (chunks[at].gpu.0 - chunks[at - 1].gpu.1) * 1e3 });
        let encode = per_process(&|chunks, at| (chunks[at].encode.1 - chunks[at].encode.0) * 1e3);
        let gap_text = if index == 0 { "n/a".to_string() } else { aggregate(&gap) };
        println!("| {} | {} | {} | {} | {} | {} | {} |", index + 1, aggregate(&ops), aggregate(&queue), aggregate(&start), aggregate(&busy), gap_text, aggregate(&encode));
    }
}
