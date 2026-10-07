//! Slice 0 attribution of the decode and prefill speed parity spec: ranks where proxima's
//! GPU time goes against llama.cpp's per-op GPU time on the same checkpoint and prompt.
//!
//! Three inputs, all files an earlier run wrote, so the table re-proves from the artifacts:
//!
//! - a `gemma4_decode_kernel_census` output directory (`census_groups.csv`,
//!   `census_dispatches.csv`): per kernel group, the median GPU span of one dispatch in its own
//!   command buffer (`warm`, floor included, the llama per-op shape) and the batched marginal
//!   span (`marginal`, floor amortised);
//! - a llama.cpp per-op timing file (`PROXIMA_OP_TIMING_OUT` of the patched Metal backend,
//!   `evidence/llama_per_op.patch`): one command buffer per encoded op, GPU span per op key;
//! - `decode_gbps_baseline` telemetry (`token_breakdown*` events) for the host and GPU timeline.
//!
//! ```sh
//! attribution_rank rank --census DIR --llama FILE --ntok 512,455,4 --requests 3
//! attribution_rank steps --events FILE
//! ```
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

type Fields = BTreeMap<String, String>;

const VOCAB_WIDE_ROWS: u64 = 65_536;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    Matmul,
    Head,
    Attention,
    Norm,
    Rope,
    MoeRouting,
    Other,
}

impl Class {
    fn label(&self) -> &'static str {
        match self {
            Class::Matmul => "matmul (weights)",
            Class::Head => "output head",
            Class::Attention => "attention core",
            Class::Norm => "rms norm",
            Class::Rope => "rope",
            Class::MoeRouting => "moe routing",
            Class::Other => "elementwise, copy, other",
        }
    }
}

#[derive(Default, Clone)]
struct Cost {
    ops: f64,
    warm_ms: f64,
    marginal_ms: f64,
    llama_raw_ms: f64,
    llama_net_ms: f64,
}

struct Side {
    by_class: BTreeMap<Class, Cost>,
    by_shape: BTreeMap<String, Cost>,
    labels: BTreeMap<String, String>,
    by_label: BTreeMap<String, Cost>,
}

impl Side {
    fn new() -> Self {
        Self {
            by_class: BTreeMap::new(),
            by_shape: BTreeMap::new(),
            labels: BTreeMap::new(),
            by_label: BTreeMap::new(),
        }
    }

    fn add(&mut self, class: Class, shape: Option<String>, delta: &Cost) {
        accumulate(self.by_class.entry(class).or_default(), delta);
        if let Some(shape) = shape {
            accumulate(self.by_shape.entry(shape).or_default(), delta);
        }
    }
}

fn accumulate(total: &mut Cost, delta: &Cost) {
    total.ops += delta.ops;
    total.warm_ms += delta.warm_ms;
    total.marginal_ms += delta.marginal_ms;
    total.llama_raw_ms += delta.llama_raw_ms;
    total.llama_net_ms += delta.llama_net_ms;
}

fn codec_name(raw: &str) -> String {
    match raw.to_ascii_lowercase().as_str() {
        "float16" | "f16" => "F16".to_string(),
        "float32" | "f32" => "F32".to_string(),
        "q6k" | "q6_k" => "Q6_K".to_string(),
        other => other.to_ascii_uppercase(),
    }
}

fn shape_key(codec: &str, dims: &[u64]) -> String {
    let elements: u64 = dims.iter().product();
    format!("{} {elements}", codec_name(codec))
}

fn shape_label(key: &str, dims: &[u64]) -> String {
    let spelled: Vec<String> = dims.iter().map(u64::to_string).collect();
    format!("{key} ({})", spelled.join("x"))
}

fn parse_extents(text: &str) -> Vec<u64> {
    text.trim_matches(|character| character == '[' || character == ']')
        .split(',')
        .filter_map(|token| token.trim().parse().ok())
        .collect()
}

fn proxima_class(label: &str) -> Class {
    if label == "head" {
        return Class::Head;
    }
    if label.starts_with("matvec ") {
        return Class::Matmul;
    }
    if label.contains("attention") || label.starts_with("softmax") {
        return Class::Attention;
    }
    if label.starts_with("RMSNorm") || label == "norm apply" {
        return Class::Norm;
    }
    if label == "RoPE" {
        return Class::Rope;
    }
    if label.contains("moe_topk") {
        return Class::MoeRouting;
    }
    Class::Other
}

fn proxima_shape(label: &str, extents: &[u64]) -> Option<String> {
    let codec = label.strip_prefix("matvec ")?;
    Some(shape_key(codec, extents.get(1..)?))
}

fn read_proxima(directory: &Path) -> (Side, f64, usize) {
    let dispatches = fs::read_to_string(directory.join("census_dispatches.csv")).expect("dispatch csv");
    let mut extents_by_node: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
    for line in dispatches.lines().skip(1) {
        let parts: Vec<&str> = line.splitn(10, ',').collect();
        extents_by_node.insert(parts[3].parse().expect("node"), parse_extents(parts[9]));
    }
    let groups = fs::read_to_string(directory.join("census_groups.csv")).expect("groups csv");
    let mut side = Side::new();
    let (mut dispatch_total, mut untimed) = (0usize, 0usize);
    for line in groups.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        let count: f64 = parts[1].parse().expect("count");
        dispatch_total += count as usize;
        if !parts[12].is_empty() {
            untimed += count as usize;
        }
        let extents = extents_by_node.get(&parts[7].parse().expect("rep node")).expect("rep extents");
        let warm_ns: f64 = parts[10].parse().expect("warm");
        let marginal_ns: f64 = parts[11].parse().expect("marginal");
        let cost = Cost {
            ops: count,
            warm_ms: count * warm_ns / 1e6,
            marginal_ms: count * marginal_ns / 1e6,
            ..Cost::default()
        };
        accumulate(side.by_label.entry(parts[0].to_string()).or_default(), &cost);
        side.add(proxima_class(parts[0]), proxima_shape(parts[0], extents), &cost);
    }
    let total_warm: f64 = side.by_class.values().map(|cost| cost.warm_ms).sum();
    (side, total_warm, dispatch_total - untimed)
}

fn field_value<'line>(line: &'line str, key: &str) -> &'line str {
    let marker = format!("{key}=");
    line.split('\t')
        .find_map(|part| part.strip_prefix(marker.as_str()))
        .unwrap_or_else(|| panic!("no {key} in {line}"))
}

fn bracket_dims(text: &str) -> Vec<u64> {
    let inner = text.split_once('[').map_or("", |(_, rest)| rest.trim_end_matches(']'));
    inner.split(',').filter_map(|token| token.parse().ok()).collect()
}

fn llama_class(op: &str, name: &str, source_zero: &str) -> Class {
    match op {
        "MUL_MAT" | "MUL_MAT_ID" if name.starts_with("kq") => Class::Attention,
        "MUL_MAT" | "MUL_MAT_ID" => Class::Matmul,
        "FLASH_ATTN_EXT" | "SOFT_MAX" if !source_zero.contains("expert") => Class::Attention,
        "RMS_NORM" => Class::Norm,
        "ROPE" => Class::Rope,
        "ARGSORT" | "TOP_K" => Class::MoeRouting,
        _ => Class::Other,
    }
}

struct LlamaFilter {
    ntoks: Vec<u32>,
    requests: f64,
    floor_us: f64,
}

fn read_llama(path: &Path, filter: &LlamaFilter) -> Side {
    let text = fs::read_to_string(path).expect("llama ops file");
    let mut side = Side::new();
    for line in text.lines().filter(|line| line.starts_with("OP\t")) {
        let ntok: u32 = field_value(line, "ntok").parse().expect("ntok");
        if !filter.ntoks.contains(&ntok) {
            continue;
        }
        let count: f64 = field_value(line, "count").parse().expect("count");
        let sum_us: f64 = field_value(line, "sum_us").parse().expect("sum_us");
        let cost = Cost {
            ops: count / filter.requests,
            llama_raw_ms: sum_us / filter.requests / 1e3,
            llama_net_ms: (sum_us - count * filter.floor_us) / filter.requests / 1e3,
            ..Cost::default()
        };
        let (op, name) = (field_value(line, "op"), field_value(line, "name"));
        let source_zero = field_value(line, "src0");
        let dims = bracket_dims(source_zero);
        let codec = source_zero.split('[').next().unwrap_or("");
        let mut class = llama_class(op, name, source_zero);
        if class == Class::Matmul && dims.get(1).is_some_and(|rows| *rows >= VOCAB_WIDE_ROWS) {
            class = Class::Head;
        }
        let shape = (class == Class::Matmul && dims.len() >= 2).then(|| shape_key(codec, &dims[..2]));
        if let Some(key) = &shape {
            side.labels.entry(key.clone()).or_insert_with(|| shape_label(key, &dims[..2]));
        }
        side.add(class, shape, &cost);
    }
    side
}

fn render_classes(proxima: &Side, llama: &Side) -> String {
    let mut classes: Vec<&Class> = proxima.by_class.keys().chain(llama.by_class.keys()).collect();
    classes.sort();
    classes.dedup();
    let mut table = String::from(
        "| class | proxima ops | proxima ms (own cb) | proxima ms (marginal) | llama ops | llama ms (own cb) | llama ms (net of floor) | gap ms (own cb) | ratio |\n|---|---|---|---|---|---|---|---|---|\n",
    );
    let mut rows: Vec<(f64, String)> = Vec::new();
    for class in classes {
        let (ours, theirs) = (cost_of(&proxima.by_class, class), cost_of(&llama.by_class, class));
        let line = format!(
            "| {} | {:.0} | {:.2} | {:.2} | {:.0} | {:.2} | {:.2} | {:+.2} | {} |\n",
            class.label(),
            ours.ops,
            ours.warm_ms,
            ours.marginal_ms,
            theirs.ops,
            theirs.llama_raw_ms,
            theirs.llama_net_ms,
            ours.warm_ms - theirs.llama_raw_ms,
            ratio_text(ours.warm_ms, theirs.llama_raw_ms)
        );
        rows.push((ours.warm_ms - theirs.llama_raw_ms, line));
    }
    rows.sort_by(|left, right| right.0.partial_cmp(&left.0).expect("finite"));
    for (_, line) in rows {
        table.push_str(&line);
    }
    table
}

fn cost_of(map: &BTreeMap<Class, Cost>, class: &Class) -> Cost {
    map.get(class).cloned().unwrap_or_default()
}

fn ratio_text(numerator: f64, denominator: f64) -> String {
    if denominator <= 0.0 {
        return "n/a".to_string();
    }
    format!("{:.2}x", numerator / denominator)
}

fn render_shapes(proxima: &Side, llama: &Side, top: usize) -> String {
    let mut keys: Vec<&String> = proxima.by_shape.keys().chain(llama.by_shape.keys()).collect();
    keys.sort();
    keys.dedup();
    let mut rows: Vec<(f64, String)> = keys
        .into_iter()
        .map(|key| {
            let ours = proxima.by_shape.get(key).cloned().unwrap_or_default();
            let theirs = llama.by_shape.get(key).cloned().unwrap_or_default();
            let line = format!(
                "| {} | {:.0} | {:.2} | {:.0} | {:.2} | {:+.2} | {} |\n",
                llama.labels.get(key).unwrap_or(key),
                ours.ops,
                ours.warm_ms,
                theirs.ops,
                theirs.llama_raw_ms,
                ours.warm_ms - theirs.llama_raw_ms,
                ratio_text(ours.warm_ms, theirs.llama_raw_ms)
            );
            (ours.warm_ms - theirs.llama_raw_ms, line)
        })
        .collect();
    rows.sort_by(|left, right| right.0.partial_cmp(&left.0).expect("finite"));
    let mut table = String::from(
        "| matmul shape | proxima ops | proxima ms (own cb) | llama ops | llama ms (own cb) | gap ms | ratio |\n|---|---|---|---|---|---|---|\n",
    );
    for (_, line) in rows.into_iter().take(top) {
        table.push_str(&line);
    }
    table
}

fn render_labels(proxima: &Side, top: usize) -> String {
    let mut rows: Vec<(&String, &Cost)> = proxima.by_label.iter().collect();
    rows.sort_by(|left, right| right.1.warm_ms.partial_cmp(&left.1.warm_ms).expect("finite"));
    let mut table = String::from(
        "| proxima census label | ops | ms (own cb) | ms (marginal) | us per op (own cb) |\n|---|---|---|---|---|\n",
    );
    for (label, cost) in rows.into_iter().take(top) {
        table.push_str(&format!(
            "| {label} | {:.0} | {:.2} | {:.2} | {:.1} |\n",
            cost.ops,
            cost.warm_ms,
            cost.marginal_ms,
            cost.warm_ms * 1e3 / cost.ops
        ));
    }
    table
}

fn flag_value(args: &[String], flag: &str) -> String {
    let position = args.iter().position(|argument| argument == flag).unwrap_or_else(|| panic!("{flag} required"));
    args.get(position + 1).unwrap_or_else(|| panic!("{flag} needs a value")).clone()
}

fn rank_text(census: &Path, llama_path: &Path, filter: &LlamaFilter) -> String {
    let (proxima, proxima_total, timed) = read_proxima(census);
    let llama = read_llama(llama_path, filter);
    let llama_total: f64 = llama.by_class.values().map(|cost| cost.llama_raw_ms).sum();
    let llama_ops: f64 = llama.by_class.values().map(|cost| cost.ops).sum();
    let mut text = String::new();
    writeln!(text, "proxima timed dispatches={timed} sum(own cb)={proxima_total:.2} ms").expect("write to string");
    writeln!(text, "llama ops per request={llama_ops:.0} sum(own cb)={llama_total:.2} ms floor_us={}", filter.floor_us).expect("write to string");
    writeln!(text, "\n{}", render_classes(&proxima, &llama)).expect("write to string");
    writeln!(text, "{}", render_shapes(&proxima, &llama, 14)).expect("write to string");
    writeln!(text, "{}", render_labels(&proxima, 12)).expect("write to string");
    text
}

fn run_rank(args: &[String]) {
    let filter = LlamaFilter {
        ntoks: flag_value(args, "--ntok").split(',').map(|token| token.parse().expect("ntok")).collect(),
        requests: flag_value(args, "--requests").parse().expect("requests"),
        floor_us: flag_value(args, "--floor-us").parse().expect("floor"),
    };
    let census = PathBuf::from(flag_value(args, "--census"));
    print!("{}", rank_text(&census, &PathBuf::from(flag_value(args, "--llama")), &filter));
}

fn parse_fields(line: &str) -> Fields {
    line.split_whitespace()
        .filter_map(|token| token.split_once('='))
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn number(fields: &Fields, key: &str) -> f64 {
    fields.get(key).and_then(|value| value.parse().ok()).unwrap_or(f64::NAN)
}

struct StepRecord {
    run: usize,
    wall: Fields,
    metal: Fields,
    gpu: Fields,
}

fn read_steps(path: &Path) -> Vec<StepRecord> {
    let text = fs::read_to_string(path).expect("events file");
    let (mut walls, mut metals, mut gpus) = (Vec::new(), Vec::new(), Vec::new());
    for line in text.lines() {
        if line.contains("token_breakdown: per-decode-step") {
            walls.push(parse_fields(line));
        } else if line.contains("token_breakdown_metal:") {
            metals.push(parse_fields(line));
        } else if line.contains("token_breakdown_gpu ") {
            gpus.push(parse_fields(line));
        }
    }
    assert!(!walls.is_empty(), "N==0: no token_breakdown wall events in {path:?}");
    assert!(walls.len() == metals.len() && metals.len() == gpus.len(), "N mismatch across event kinds");
    let mut run = 0usize;
    walls
        .into_iter()
        .zip(metals)
        .zip(gpus)
        .map(|((wall, metal), gpu)| {
            run += usize::from(number(&wall, "step") == 0.0);
            StepRecord { run: run - 1, wall, metal, gpu }
        })
        .collect()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    values[values.len() / 2]
}

type Column = (&'static str, &'static str);

const HOST_COLUMNS: [Column; 9] = [
    ("op_setup", "op_setup_ms"),
    ("loop_head", "loop_head_ms"),
    ("pre_encode", "pre_encode_ms"),
    ("expert_lookup", "expert_buffers_lookup_ms"),
    ("retire_scan", "retire_scan_ms"),
    ("placement", "placement_resolve_ms"),
    ("resolve_plan", "resolve_plan_ms"),
    ("encoder_finish", "encoder_finish_ms"),
    ("gpu_end_to_return", "gpu_end_to_wait_return_ms"),
];

const TIMELINE_COLUMNS: [Column; 15] = [
    ("wall", "step_wall_ms"),
    ("evaluate", "evaluate_ms"),
    ("gpu_exec", "gpu_exec_ms"),
    ("gpu_busy", "gpu_busy_ms"),
    ("commit_to_start", "commit_to_gpu_start_ms"),
    ("prepare", "prepare_ms"),
    ("encode", "encode_dispatch_ms"),
    ("readback", "readback_ms"),
    ("kv_named", "named_blocks_kv_ms"),
    ("kv_append", "layer_cache_append_ms"),
    ("dispatches", "physical_dispatch_calls"),
    ("chunks", "chunks"),
    ("encode_overlap", "encode_overlap_ms"),
    ("plan_misses", "plan_misses"),
    ("plan_hits", "plan_hits"),
];

fn column_value(record: &StepRecord, key: &str) -> f64 {
    [&record.wall, &record.metal, &record.gpu]
        .iter()
        .find_map(|fields| fields.get(key).and_then(|value| value.parse().ok()))
        .unwrap_or(f64::NAN)
}

fn step_row(label: &str, columns: &[Column], records: &[&StepRecord]) -> String {
    let mut row = format!("| {label} |");
    for (_, key) in columns {
        let mut values: Vec<f64> = records.iter().map(|record| column_value(record, key)).collect();
        write!(row, " {:.2} |", median(&mut values)).expect("write to string");
    }
    row.push('\n');
    row
}

fn step_table(records: &[StepRecord], columns: &[Column]) -> String {
    let runs = records.iter().map(|record| record.run).max().expect("non-empty") + 1;
    let mut table = String::from("| arm |");
    for (name, _) in columns {
        write!(table, " {name} |").expect("write to string");
    }
    table.push_str("\n|---|");
    table.push_str(&"---|".repeat(columns.len()));
    table.push('\n');
    for run in 0..runs {
        let in_run: Vec<&StepRecord> = records.iter().filter(|record| record.run == run).collect();
        let prefill: Vec<&StepRecord> = in_run.iter().copied().filter(|record| number(&record.wall, "step") == 0.0).collect();
        let steady: Vec<&StepRecord> = in_run.iter().copied().filter(|record| number(&record.wall, "step") >= 2.0).collect();
        table.push_str(&step_row(&format!("run {run} step 0 (prefill, n={})", prefill.len()), columns, &prefill));
        table.push_str(&step_row(&format!("run {run} steps >= 2 median (n={})", steady.len()), columns, &steady));
    }
    table
}

fn steps_text(events: &Path) -> String {
    let records = read_steps(events);
    format!("{}\n{}\n", step_table(&records, &TIMELINE_COLUMNS), step_table(&records, &HOST_COLUMNS))
}

fn run_steps(args: &[String]) {
    print!("{}", steps_text(&PathBuf::from(flag_value(args, "--events"))));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("rank") => run_rank(&args),
        Some("steps") => run_steps(&args),
        other => panic!("subcommand rank or steps, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_q4_0_projection_maps_to_the_same_shape_key_on_both_sides() {
        let proxima = proxima_shape("matvec Q4_0", &[971, 12288, 1536]).expect("matmul shape");
        let llama = shape_key("q4_0", &[1536, 12288]);
        assert_eq!(proxima, llama);
    }

    #[test]
    fn the_proxima_head_label_is_the_head_class() {
        assert_eq!(proxima_class("head"), Class::Head);
    }

    #[test]
    fn fused_and_plain_norm_labels_share_one_class() {
        assert_eq!(proxima_class("RMSNorm sumsq + fused epilogue"), Class::Norm);
        assert_eq!(llama_class("RMS_NORM", "norm", "f32[1536,1,1]"), Class::Norm);
    }

    #[test]
    fn extents_with_inner_axes_multiply_into_the_row_count() {
        let shape = proxima_shape("matvec Q4_0", &[971, 8, 256, 1536]).expect("matmul shape");
        assert_eq!(shape, shape_key("q4_0", &[1536, 2048]));
    }

    fn evidence(relative: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../proxima-tensor/specs/decode-prefill-parity/evidence/slice0")
            .join(relative)
    }

    fn assert_rank_matches(census: &str, llama: &str, filter: &LlamaFilter, saved: &str) {
        let text = rank_text(&evidence(census), &evidence(llama), filter);
        assert!(text.contains("proxima timed dispatches="), "N==0: the rank report timed nothing");
        assert_eq!(text, saved, "the committed raw evidence no longer reproduces the saved table");
    }

    fn prefill_filter(ntoks: &[u32]) -> LlamaFilter {
        LlamaFilter { ntoks: ntoks.to_vec(), requests: 3.0, floor_us: 4.0 }
    }

    fn decode_filter() -> LlamaFilter {
        LlamaFilter { ntoks: vec![1], requests: 381.0, floor_us: 4.0 }
    }

    #[test]
    fn the_e2b_prefill_table_reproduces_from_the_committed_census_and_llama_ops() {
        assert_rank_matches(
            "census_e2b_prefill",
            "llama_ops/e2b_ops.tsv",
            &prefill_filter(&[512, 455, 4]),
            include_str!("../../proxima-tensor/specs/decode-prefill-parity/evidence/slice0/rank/e2b_prefill.md"),
        );
    }

    #[test]
    fn the_e2b_decode_table_reproduces_from_the_committed_census_and_llama_ops() {
        assert_rank_matches(
            "census_e2b_decode",
            "llama_ops/e2b_ops.tsv",
            &decode_filter(),
            include_str!("../../proxima-tensor/specs/decode-prefill-parity/evidence/slice0/rank/e2b_decode.md"),
        );
    }

    #[test]
    fn the_granite_prefill_table_reproduces_from_the_committed_census_and_llama_ops() {
        assert_rank_matches(
            "census_granite_prefill",
            "llama_ops/granite_ops.tsv",
            &prefill_filter(&[512, 488]),
            include_str!("../../proxima-tensor/specs/decode-prefill-parity/evidence/slice0/rank/granite_prefill.md"),
        );
    }

    #[test]
    fn the_granite_decode_table_reproduces_from_the_committed_census_and_llama_ops() {
        assert_rank_matches(
            "census_granite_decode",
            "llama_ops/granite_ops.tsv",
            &decode_filter(),
            include_str!("../../proxima-tensor/specs/decode-prefill-parity/evidence/slice0/rank/granite_decode.md"),
        );
    }

    #[test]
    fn a_decode_filter_over_prefill_evidence_does_not_reproduce_the_prefill_table() {
        let text = rank_text(
            &evidence("census_e2b_prefill"),
            &evidence("llama_ops/e2b_ops.tsv"),
            &decode_filter(),
        );
        let saved = include_str!("../../proxima-tensor/specs/decode-prefill-parity/evidence/slice0/rank/e2b_prefill.md");
        assert_ne!(text, saved, "control: the wrong graph selection must change the table");
    }

    #[test]
    fn the_step_timelines_reproduce_from_the_committed_token_breakdown_events() {
        let e2b = steps_text(&evidence("timeline/e2b_token_breakdown.log"));
        let granite = steps_text(&evidence("timeline/granite_token_breakdown.log"));
        assert_eq!(e2b, include_str!("../../proxima-tensor/specs/decode-prefill-parity/evidence/slice0/rank/e2b_steps.md"));
        assert_eq!(granite, include_str!("../../proxima-tensor/specs/decode-prefill-parity/evidence/slice0/rank/granite_steps.md"));
    }
}
