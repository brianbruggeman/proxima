//! Prompt-cache oracle and timing harness for `proxima-tensor/specs/prefix-cache-reuse/SPEC.md`
//! slice S4: AC6 (token ids against llama-server on the same ids), AC7 (time to first token with
//! the cache on and off, plus llama-server's own), and the cost of splitting a prefill at
//! checkpoints.
//!
//! `--mode oracle`: replays the AC2 3-turn extension, the AC3 50-token rewrite and the AC4
//! 2,000-token rewrite through the prompt cache, then sends the same token ids, in the same turn
//! order, to a fresh llama-server per scenario with `cache_prompt` on, and compares the generated ids.
//!
//! `--mode ttft`: a multi-turn transcript (`data/prompt_cache_transcript.jsonl`, or the AC2 one),
//! cache on, cache off and a prefix-only arm (the prefill time of the tokens the cache reuses), as
//! interleaved pairs; then, with `--llama-server-bin`, the same prompt ids against llama-server
//! with `cache_prompt` on and off.
//!
//! `--mode split`: one cold full-prompt prefill with checkpoint splitting off, at the default
//! interval, and at swept intervals.
//!
//! Every request is one JSON line in `--out`; the summary is printed to stdout. Host snapshots
//! between pairs record the load average, GPU utilization, the busiest processes, and the CPU of the
//! process named by `PROMPT_CACHE_BENCH_WATCH_PROCESS`.
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_lines)]

use core::ops::ControlFlow;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{
    CacheReport, GPU_LAYERS_ALL, LoadedModel, PromptCacheConfig, ServingConfig,
    SpeculativeConfig,
};
use proxima_tokenizer::vocab::Vocab;
use serde_json::{Value, json};

const CORPUS: &str = include_str!("data/speculative_corpus.jsonl");
const TRANSCRIPT: &str = include_str!("data/prompt_cache_transcript.jsonl");
const DEFAULT_MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
const LLAMA_PORT_ON: u16 = 18490;
const LLAMA_PORT_OFF: u16 = 18491;
const LLAMA_CONTEXT_HEADROOM_TOKENS: usize = 512;
const REFUTATION_MARGIN: f64 = 0.9;
const REWRITTEN_WITHIN_SLACK: usize = 50;
const REWRITTEN_BEYOND_SLACK: usize = 2000;
const CLEAR_PROBE_TEXT: &str = "ok, reset the session";
const PROBE_ENTRY_BYTES_BOUND: usize = 100 << 20;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Oracle,
    Ttft,
    Split,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TranscriptKind {
    Long,
    Ac2,
}

struct Args {
    mode: Mode,
    model_path: String,
    llama_bin: Option<PathBuf>,
    pairs: usize,
    max_tokens: usize,
    out: PathBuf,
    lengths: Vec<usize>,
    intervals: Vec<u32>,
    transcript: TranscriptKind,
}

fn parse_list<T: std::str::FromStr>(value: &str) -> Vec<T> {
    value
        .split(',')
        .map(|item| {
            item.parse()
                .unwrap_or_else(|_| panic!("cannot parse list item {item:?}"))
        })
        .collect()
}

fn parse_args() -> Args {
    let mut args = Args {
        mode: Mode::Ttft,
        model_path: env::var("PROXIMA_GEMMA4_E2B_GGUF").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string()),
        llama_bin: None,
        pairs: 10,
        max_tokens: 32,
        out: PathBuf::from("prompt_cache_bench.jsonl"),
        lengths: vec![2500, 4300, 8300],
        intervals: vec![1024, 512, 256],
        transcript: TranscriptKind::Long,
    };
    let mut iterator = env::args().skip(1);
    while let Some(flag) = iterator.next() {
        let value = iterator.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--mode" => args.mode = parse_mode(&value),
            "--model" => args.model_path = value,
            "--llama-server-bin" => args.llama_bin = Some(PathBuf::from(value)),
            "--pairs" => args.pairs = value.parse().expect("--pairs is a count"),
            "--max-tokens" => args.max_tokens = value.parse().expect("--max-tokens is a count"),
            "--out" => args.out = PathBuf::from(value),
            "--lengths" => args.lengths = parse_list(&value),
            "--intervals" => args.intervals = parse_list(&value),
            "--transcript" => args.transcript = parse_transcript(&value),
            other => panic!("unknown flag {other}"),
        }
    }
    args
}

fn parse_mode(value: &str) -> Mode {
    match value {
        "oracle" => Mode::Oracle,
        "ttft" => Mode::Ttft,
        "split" => Mode::Split,
        other => panic!("--mode is oracle, ttft or split, got {other}"),
    }
}

fn parse_transcript(value: &str) -> TranscriptKind {
    match value {
        "long" => TranscriptKind::Long,
        "ac2" => TranscriptKind::Ac2,
        other => panic!("--transcript is long or ac2, got {other}"),
    }
}

struct Recorder {
    file: File,
}

impl Recorder {
    fn create(path: &Path) -> Self {
        Self {
            file: File::create(path).unwrap_or_else(|err| panic!("create {}: {err}", path.display())),
        }
    }

    fn write(&mut self, record: &Value) {
        writeln!(self.file, "{record}").expect("write a record");
        self.file.flush().expect("flush a record");
    }
}

fn percentile(sorted: &[f64], percent: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = ((percent / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut copy = values.to_vec();
    copy.sort_by(f64::total_cmp);
    copy
}

fn median(values: &[f64]) -> f64 {
    percentile(&sorted(values), 50.0)
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt() / mean
}

struct Spread {
    count: usize,
    median: f64,
    p10: f64,
    p90: f64,
    cov: f64,
}

fn spread(values: &[f64]) -> Spread {
    let ordered = sorted(values);
    Spread {
        count: values.len(),
        median: percentile(&ordered, 50.0),
        p10: percentile(&ordered, 10.0),
        p90: percentile(&ordered, 90.0),
        cov: coefficient_of_variation(values),
    }
}

impl Spread {
    fn text(&self) -> String {
        format!(
            "n={} median={:.1} p10={:.1} p90={:.1} cov={:.1}%",
            self.count,
            self.median,
            self.p10,
            self.p90,
            self.cov * 100.0
        )
    }
}

fn run_text(command: &str, args: &[&str]) -> String {
    Command::new(command)
        .args(args)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_default()
}

fn gpu_utilization_percent() -> Option<f64> {
    let stdout = run_text("ioreg", &["-r", "-d", "1", "-c", "IOAccelerator"]);
    let needle = "\"Device Utilization %\"=";
    let start = stdout.find(needle)? + needle.len();
    let rest = &stdout[start..];
    let end = rest.find(|character: char| !character.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn top_processes() -> Vec<String> {
    let table = run_text("ps", &["-axo", "%cpu=,pid=,comm="]);
    let mut rows: Vec<(f64, String)> = table
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            let (cpu, rest) = trimmed.split_once(' ')?;
            Some((cpu.parse().ok()?, rest.trim().to_string()))
        })
        .collect();
    rows.sort_by(|left, right| right.0.total_cmp(&left.0));
    rows.iter().take(5).map(|(cpu, rest)| format!("{cpu}% {rest}")).collect()
}

fn watched_process_cpu_percent() -> f64 {
    let Some(name) = env::var("PROMPT_CACHE_BENCH_WATCH_PROCESS").ok() else {
        return f64::NAN;
    };
    run_text("ps", &["-axo", "%cpu=,comm="])
        .lines()
        .filter(|line| line.contains(&name))
        .filter_map(|line| line.split_whitespace().next()?.parse::<f64>().ok())
        .sum()
}

fn host_snapshot(label: &str) -> Value {
    json!({
        "kind": "host",
        "label": label,
        "loadavg": run_text("sysctl", &["-n", "vm.loadavg"]),
        "watched_process_cpu_percent": watched_process_cpu_percent(),
        "gpu_device_utilization_percent": gpu_utilization_percent(),
        "top_processes": top_processes(),
    })
}

fn build_summary() -> Value {
    let features: Vec<&str> = [
        ("std", cfg!(feature = "std")),
        ("metal", cfg!(feature = "metal")),
        ("metal-fuse-attn-decode", cfg!(feature = "metal-fuse-attn-decode")),
        ("identity-copy-alias", cfg!(feature = "identity-copy-alias")),
        ("metal-tiled-gemm", cfg!(feature = "metal-tiled-gemm")),
    ]
    .iter()
    .filter_map(|(name, enabled)| enabled.then_some(*name))
    .collect();
    json!({
        "kind": "build",
        "features": features,
        "debug_assertions": cfg!(debug_assertions),
        "git_head": run_text("git", &["rev-parse", "--short", "HEAD"]),
    })
}

fn base_config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        temperature: 0.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        speculative: SpeculativeConfig::none(),
        ..ServingConfig::default()
    }
}

fn config_with(prompt_cache: PromptCacheConfig) -> ServingConfig<'static> {
    ServingConfig {
        prompt_cache,
        ..base_config()
    }
}

fn uncached() -> ServingConfig<'static> {
    config_with(PromptCacheConfig::off())
}

fn checkpointed(interval: u32, max_checkpoints: u32) -> PromptCacheConfig {
    PromptCacheConfig {
        checkpoint_interval: interval,
        max_checkpoints,
        ..PromptCacheConfig::standard()
    }
}

fn with_model<T>(model_path: &str, body: impl FnOnce(&LoadedModel<'_>, &Vocab) -> T) -> T {
    let file = File::open(model_path).unwrap_or_else(|err| panic!("open model {model_path}: {err}"));
    // SAFETY: read-only mapping of a checkpoint nothing else writes during the run.
    let mapping = unsafe { Mmap::map(&file) }.expect("map the model");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the model header");
    let model = LoadedModel::load(&parsed, bytes).expect("bind the model");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("build the vocab");
    body(&model, &vocab)
}

fn wants_bos(vocab: &Vocab) -> bool {
    vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some())
}

fn encode_opening(vocab: &Vocab, text: &str) -> Vec<u32> {
    proxima_tokenizer::encode_with_bos_eos(text, vocab, wants_bos(vocab), vocab.add_eos_token().unwrap_or(false))
        .expect("tokenize the opening prompt")
}

fn encode_continuation(vocab: &Vocab, text: &str) -> Vec<u32> {
    proxima_tokenizer::encode_with_bos_eos(text, vocab, false, false).expect("tokenize a continuation")
}

fn corpus_document(id: &str) -> String {
    CORPUS
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("corpus line is json"))
        .find(|record| record["id"] == id)
        .and_then(|record| record["prompt"].as_str().map(str::to_owned))
        .unwrap_or_else(|| panic!("corpus has no record with id {id}"))
}

fn excerpt(document: &str, from_char: usize, chars: usize) -> String {
    document.chars().skip(from_char).take(chars).collect()
}

fn opening_turn(user_text: &str) -> String {
    format!("<|turn>user\n{user_text}<turn|>\n<|turn>model\n")
}

fn next_turn(user_text: &str) -> String {
    format!("<turn|>\n<|turn>user\n{user_text}<turn|>\n<|turn>model\n")
}

fn long_document(chars: usize) -> String {
    ["rag011", "rag016", "rag013", "rag008", "rag012", "rag004"]
        .iter()
        .map(|id| corpus_document(id))
        .collect::<Vec<_>>()
        .join("\n\n")
        .chars()
        .take(chars)
        .collect()
}

fn user_texts(kind: TranscriptKind) -> Vec<String> {
    match kind {
        TranscriptKind::Ac2 => {
            let document = corpus_document("rag004");
            vec![
                excerpt(&document, 0, 1200),
                excerpt(&document, 1200, 500),
                excerpt(&document, 1700, 500),
            ]
        }
        TranscriptKind::Long => TRANSCRIPT
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("transcript line is json"))
            .map(|turn| {
                let document = corpus_document(turn["doc_id"].as_str().expect("doc_id"));
                let body = excerpt(
                    &document,
                    turn["from_char"].as_u64().expect("from_char") as usize,
                    turn["chars"].as_u64().expect("chars") as usize,
                );
                format!("{}\n\n{body}", turn["instruction"].as_str().expect("instruction"))
            })
            .collect(),
    }
}

struct Timed {
    ttft_ms: f64,
    total_ms: f64,
    generated: Vec<u32>,
    stopped_by_eos: bool,
    report: Option<CacheReport>,
}

fn timed_request(
    model: &LoadedModel<'_>,
    ids: &[u32],
    max_tokens: usize,
    config: &ServingConfig<'_>,
) -> Timed {
    let start = Instant::now();
    let mut first_token = None;
    let (generated, _text, stopped_by_eos) = model
        .generate_from_ids(ids, max_tokens, config, &mut |_event| {
            first_token.get_or_insert_with(|| start.elapsed());
            ControlFlow::Continue(())
        })
        .expect("generate from ids");
    let total = start.elapsed();
    Timed {
        ttft_ms: first_token.expect("the request emitted a token").as_secs_f64() * 1000.0,
        total_ms: total.as_secs_f64() * 1000.0,
        generated,
        stopped_by_eos,
        report: config
            .prompt_cache
            .is_enabled()
            .then(|| model.last_prompt_cache_report().expect("a cached request records its report")),
    }
}

fn probe_ids(vocab: &Vocab) -> Vec<u32> {
    encode_continuation(vocab, CLEAR_PROBE_TEXT)
}

/// The cache has no clear call. A one-entry request whose first token is not the BOS every real
/// prompt starts with matches nothing, stores itself, and evicts every older entry, so the next
/// real prompt reports `Miss` / `NoCommonPrefix` and the cache holds one probe-sized entry.
fn clear_cache(model: &LoadedModel<'_>, vocab: &Vocab) {
    let probe = config_with(PromptCacheConfig {
        max_entries: 1,
        ..PromptCacheConfig::standard()
    });
    let ids = probe_ids(vocab);
    assert_ne!(ids[0], encode_opening(vocab, "x")[0], "the probe must not start with the BOS token");
    model
        .generate_from_ids(&ids, 1, &probe, &mut |_event| ControlFlow::Continue(()))
        .expect("clearing request");
    let bytes = model.prompt_cache_bytes();
    assert!(bytes < PROBE_ENTRY_BYTES_BOUND, "the clearing request left {bytes} cache bytes behind");
}

fn tokens_held_after(prompt_len: usize, timed: &Timed) -> usize {
    let unforwarded = usize::from(!timed.stopped_by_eos);
    prompt_len + timed.generated.len() - unforwarded.min(timed.generated.len())
}

fn path_label(report: Option<CacheReport>) -> Value {
    report.map_or(Value::Null, |report| {
        json!({
            "path": report.path.as_str(),
            "lcp": report.lcp,
            "reused_tokens": report.reused_tokens,
            "prefilled_tokens": report.prefilled_tokens,
            "miss": report.miss.map(|reason| reason.as_str()),
        })
    })
}

struct TurnPlan {
    prompt_ids: Vec<u32>,
    held_before: usize,
    generated: Vec<u32>,
}

fn build_plans(model: &LoadedModel<'_>, vocab: &Vocab, kind: TranscriptKind, max_tokens: usize) -> Vec<TurnPlan> {
    clear_cache(model, vocab);
    let config = config_with(PromptCacheConfig::standard());
    let mut plans: Vec<TurnPlan> = Vec::new();
    let mut held_before = 0;
    for text in &user_texts(kind) {
        let prompt_ids = match plans.last() {
            None => encode_opening(vocab, &opening_turn(text)),
            Some(previous) => {
                let mut ids = previous.prompt_ids.clone();
                ids.extend_from_slice(&previous.generated);
                ids.extend(encode_continuation(vocab, &next_turn(text)));
                ids
            }
        };
        let timed = timed_request(model, &prompt_ids, max_tokens, &config);
        plans.push(TurnPlan {
            prompt_ids: prompt_ids.clone(),
            held_before,
            generated: timed.generated.clone(),
        });
        held_before = tokens_held_after(prompt_ids.len(), &timed);
    }
    plans
}

const ARM_ORDERS: [[usize; 3]; 6] = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2]];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    CacheOn,
    CacheOff,
    PrefixOnly,
}

impl Arm {
    const ALL: [Self; 3] = [Self::CacheOn, Self::CacheOff, Self::PrefixOnly];

    const fn label(self) -> &'static str {
        match self {
            Self::CacheOn => "cache_on",
            Self::CacheOff => "cache_off",
            Self::PrefixOnly => "prefix_only",
        }
    }
}

#[derive(Default)]
struct TurnSamples {
    on: Vec<f64>,
    off: Vec<f64>,
    prefix: Vec<f64>,
    saving_met: usize,
    identical: usize,
    pairs: usize,
}

fn run_arm(
    model: &LoadedModel<'_>,
    plan: &TurnPlan,
    arm: Arm,
    max_tokens: usize,
) -> (Timed, usize) {
    let cached = config_with(PromptCacheConfig::standard());
    match arm {
        Arm::CacheOn => (timed_request(model, &plan.prompt_ids, max_tokens, &cached), plan.prompt_ids.len()),
        Arm::CacheOff => (timed_request(model, &plan.prompt_ids, max_tokens, &uncached()), plan.prompt_ids.len()),
        Arm::PrefixOnly => (
            timed_request(model, &plan.prompt_ids[..plan.held_before], max_tokens, &uncached()),
            plan.held_before,
        ),
    }
}

fn run_ttft_pairs(
    model: &LoadedModel<'_>,
    vocab: &Vocab,
    args: &Args,
    recorder: &mut Recorder,
) -> (Vec<TurnSamples>, Vec<TurnPlan>, Vec<u32>) {
    let plans = build_plans(model, vocab, args.transcript, args.max_tokens);
    for (turn, plan) in plans.iter().enumerate() {
        recorder.write(&json!({
            "kind": "plan", "turn": turn, "prompt_tokens": plan.prompt_ids.len(),
            "held_before": plan.held_before, "generated": plan.generated,
        }));
    }
    let mut samples: Vec<TurnSamples> = plans.iter().map(|_| TurnSamples::default()).collect();
    for pair in 0..=args.pairs {
        recorder.write(&host_snapshot(&format!("pair_{pair}_start")));
        clear_cache(model, vocab);
        for (turn, plan) in plans.iter().enumerate() {
            let mut measured = [None; 3];
            let order = ARM_ORDERS[(pair * plans.len() + turn) % ARM_ORDERS.len()];
            let mut identical = true;
            for (position, arm_index) in order.into_iter().enumerate() {
                let arm = Arm::ALL[arm_index];
                if arm == Arm::PrefixOnly && plan.held_before == 0 {
                    continue;
                }
                let (timed, prompt_tokens) = run_arm(model, plan, arm, args.max_tokens);
                measured[arm_index] = Some(timed.ttft_ms);
                if arm != Arm::PrefixOnly {
                    identical &= timed.generated == plan.generated;
                }
                recorder.write(&json!({
                    "kind": "request", "mode": "ttft", "pair": pair, "warmup": pair == 0, "turn": turn,
                    "arm": arm.label(), "order_position": position, "prompt_tokens": prompt_tokens,
                    "ttft_ms": timed.ttft_ms, "total_ms": timed.total_ms,
                    "generated": timed.generated.len(), "cache": path_label(timed.report),
                }));
            }
            if pair > 0 {
                record_pair(&mut samples[turn], measured, identical);
            }
        }
        recorder.write(&host_snapshot(&format!("pair_{pair}_end")));
    }
    (samples, plans, probe_ids(vocab))
}

fn record_pair(samples: &mut TurnSamples, measured: [Option<f64>; 3], identical: bool) {
    let on = measured[0].expect("cache-on arm ran");
    let off = measured[1].expect("cache-off arm ran");
    samples.on.push(on);
    samples.off.push(off);
    samples.pairs += 1;
    samples.identical += usize::from(identical);
    if let Some(prefix) = measured[2] {
        samples.prefix.push(prefix);
        samples.saving_met += usize::from(off - on >= REFUTATION_MARGIN * prefix);
    }
}

fn print_ttft_summary(samples: &[TurnSamples], plans: &[TurnPlan]) {
    println!("proxima ttft ms (cache on vs off, prefix_only = prefill time of the tokens the cache reuses)");
    for (turn, (turn_samples, plan)) in samples.iter().zip(plans).enumerate() {
        println!(
            "turn={turn} prompt_tokens={} held_tokens={} ids_identical_on_vs_reference={}/{}",
            plan.prompt_ids.len(),
            plan.held_before,
            turn_samples.identical,
            turn_samples.pairs
        );
        println!("  cache_on   {}", spread(&turn_samples.on).text());
        println!("  cache_off  {}", spread(&turn_samples.off).text());
        let ratio = median(&turn_samples.on) / median(&turn_samples.off);
        println!("  median_on/median_off={ratio:.3}");
        if turn_samples.prefix.is_empty() {
            continue;
        }
        let saving = median(&turn_samples.off) - median(&turn_samples.on);
        let required = REFUTATION_MARGIN * median(&turn_samples.prefix);
        println!("  prefix_only {}", spread(&turn_samples.prefix).text());
        println!(
            "  median_saving_ms={saving:.1} required_ms(0.9*median_prefix)={required:.1} pairs_with_saving_ge_required={}/{}",
            turn_samples.saving_met, turn_samples.pairs
        );
    }
}

struct ChildGuard {
    child: Child,
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct LlamaServer {
    guard: ChildGuard,
    port: u16,
}

fn llama_args(config: &ServingConfig<'_>, context: usize) -> Vec<String> {
    let pairs: [(&str, String); 13] = [
        ("-c", context.to_string()),
        ("-ngl", "all".to_string()),
        ("-np", config.parallel_sequences.to_string()),
        ("--temp", config.temperature.to_string()),
        ("--top-k", config.top_k.to_string()),
        ("--top-p", config.top_p.to_string()),
        ("--min-p", config.min_p.to_string()),
        ("--repeat-last-n", config.repeat_last_n.to_string()),
        ("--repeat-penalty", config.repeat_penalty.to_string()),
        ("--frequency-penalty", config.frequency_penalty.to_string()),
        ("--presence-penalty", config.presence_penalty.to_string()),
        ("--seed", config.seed.to_string()),
        ("--spec-type", "none".to_string()),
    ];
    pairs
        .into_iter()
        .flat_map(|(flag, value)| [flag.to_string(), value])
        .chain(["--no-webui".to_string()])
        .collect()
}

impl LlamaServer {
    fn spawn(bin: &Path, model_path: &str, port: u16, context: usize, extra: &[&str]) -> Self {
        let mut command = Command::new(bin);
        if let Some(directory) = bin.parent() {
            command.env("DYLD_LIBRARY_PATH", directory);
        }
        let arguments: Vec<String> = ["--model", model_path, "--host", "127.0.0.1", "--port", &port.to_string()]
            .iter()
            .map(ToString::to_string)
            .chain(llama_args(&base_config(), context))
            .chain(extra.iter().map(ToString::to_string))
            .collect();
        eprintln!("llama-server command: {} {}", bin.display(), arguments.join(" "));
        let child = command
            .args(&arguments)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|err| panic!("spawn llama-server {}: {err}", bin.display()));
        let server = Self {
            guard: ChildGuard { child },
            port,
        };
        server.wait_until_healthy();
        server
    }

    fn wait_until_healthy(&self) {
        let deadline = Instant::now() + Duration::from_secs(180);
        while Instant::now() < deadline {
            if health_ok(self.port) {
                return;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        panic!("llama-server on port {} never became healthy", self.port);
    }

    fn pid(&self) -> u32 {
        self.guard.child.id()
    }
}

fn health_ok(port: u16) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let request = format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut text = String::new();
    BufReader::new(stream)
        .lines()
        .map_while(Result::ok)
        .for_each(|line| text.push_str(&line));
    text.contains("\"status\":\"ok\"")
}

struct LlamaResult {
    ids: Vec<u32>,
    ttft_wall_ms: f64,
    total_wall_ms: f64,
    timings: Value,
}

fn llama_completion(port: u16, ids: &[u32], max_tokens: usize, cache_prompt: bool) -> LlamaResult {
    let body = serde_json::to_vec(&json!({
        "prompt": ids, "n_predict": max_tokens, "temperature": 0, "cache_prompt": cache_prompt,
        "stream": true, "return_tokens": true,
    }))
    .expect("serialize the request");
    let start = Instant::now();
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect to llama-server");
    let head = format!(
        "POST /completion HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).expect("write the head");
    stream.write_all(&body).expect("write the body");
    read_llama_stream(BufReader::new(stream), start)
}

fn read_llama_stream(mut reader: BufReader<TcpStream>, start: Instant) -> LlamaResult {
    let mut status = String::new();
    reader.read_line(&mut status).expect("read the status line");
    assert!(status.contains("200"), "llama-server answered {status:?}");
    let mut result = LlamaResult {
        ids: Vec::new(),
        ttft_wall_ms: f64::NAN,
        total_wall_ms: 0.0,
        timings: Value::Null,
    };
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).expect("read a stream line") == 0 {
            break;
        }
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        let event: Value = serde_json::from_str(payload.trim()).unwrap_or_else(|err| panic!("stream event {payload:?}: {err}"));
        let tokens: Vec<u32> = event["tokens"]
            .as_array()
            .map(|array| array.iter().map(|token| token.as_u64().expect("token id") as u32).collect())
            .unwrap_or_default();
        if !tokens.is_empty() && result.ttft_wall_ms.is_nan() {
            result.ttft_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
        }
        result.ids.extend(tokens);
        if event.get("timings").is_some() {
            result.timings = event["timings"].clone();
        }
    }
    result.total_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    result
}

fn first_divergence(left: &[u32], right: &[u32]) -> Option<usize> {
    let shared = left.len().min(right.len());
    (0..shared)
        .find(|&index| left[index] != right[index])
        .or_else(|| (left.len() != right.len()).then_some(shared))
}

struct OracleRequest {
    scenario: &'static str,
    index: usize,
    ids: Vec<u32>,
    cached: Vec<u32>,
    fresh: Vec<u32>,
    report: CacheReport,
    expectation: String,
}

fn oracle_request(
    model: &LoadedModel<'_>,
    scenario: &'static str,
    index: usize,
    ids: Vec<u32>,
    config: &ServingConfig<'_>,
    expectation: String,
    max_tokens: usize,
) -> OracleRequest {
    let cached = timed_request(model, &ids, max_tokens, config);
    let fresh = timed_request(model, &ids, max_tokens, &uncached());
    OracleRequest {
        scenario,
        index,
        report: cached.report.expect("cached request report"),
        cached: cached.generated,
        fresh: fresh.generated,
        ids,
        expectation,
    }
}

fn oracle_ac2(model: &LoadedModel<'_>, vocab: &Vocab, max_tokens: usize) -> Vec<OracleRequest> {
    clear_cache(model, vocab);
    let config = config_with(checkpointed(PromptCacheConfig::standard().checkpoint_interval, 8));
    let mut requests: Vec<OracleRequest> = Vec::new();
    let mut held = 0;
    for (turn, text) in user_texts(TranscriptKind::Ac2).iter().enumerate() {
        let ids = match requests.last() {
            None => encode_opening(vocab, &opening_turn(text)),
            Some(previous) => {
                let mut ids = previous.ids.clone();
                ids.extend_from_slice(&previous.cached);
                ids.extend(encode_continuation(vocab, &next_turn(text)));
                ids
            }
        };
        let expectation = if turn == 0 {
            "miss".to_string()
        } else {
            format!("extend lcp={held} prefilled={}", ids.len() - held)
        };
        let request = oracle_request(model, "ac2_three_turn_extension", turn, ids, &config, expectation, max_tokens);
        held = request.ids.len() + request.cached.len() - 1;
        requests.push(request);
    }
    requests
}

fn oracle_rewrite(
    model: &LoadedModel<'_>,
    vocab: &Vocab,
    scenario: &'static str,
    opening_text: &str,
    rewritten: usize,
    config: &ServingConfig<'_>,
    max_tokens: usize,
) -> Vec<OracleRequest> {
    clear_cache(model, vocab);
    let opening = encode_opening(vocab, opening_text);
    let first = oracle_request(model, scenario, 0, opening.clone(), config, "miss".to_string(), max_tokens);
    let held = opening.len() + first.cached.len() - 1;
    let mut stored = opening;
    stored.extend_from_slice(&first.cached);
    stored.truncate(held);
    let kept = held - rewritten;
    let document = corpus_document("rag004");
    let tail_text = format!(
        "Disregard the passage above and answer in one sentence. {}",
        next_turn(&excerpt(&document, 3200, 1500))
    );
    let tail = encode_continuation(vocab, &tail_text);
    assert_ne!(tail[0], stored[kept], "the rewrite must diverge at its first token");
    let mut ids = stored[..kept].to_vec();
    ids.extend_from_slice(&tail);
    let expectation = format!("kept={kept} rewritten={rewritten} tail={} stored={held}", tail.len());
    let second = oracle_request(model, scenario, 1, ids, config, expectation, max_tokens);
    vec![first, second]
}

fn run_oracle_proxima(args: &Args) -> Vec<OracleRequest> {
    with_model(&args.model_path, |model, vocab| {
        let max_tokens = args.max_tokens;
        let mut requests = oracle_ac2(model, vocab, max_tokens);
        let within = config_with(checkpointed(PromptCacheConfig::standard().checkpoint_interval, 8));
        requests.extend(oracle_rewrite(
            model,
            vocab,
            "ac3_rewrite_last_50",
            &opening_turn(&excerpt(&corpus_document("rag004"), 0, 3000)),
            REWRITTEN_WITHIN_SLACK,
            &within,
            max_tokens,
        ));
        let beyond = config_with(checkpointed(512, 8));
        requests.extend(oracle_rewrite(
            model,
            vocab,
            "ac4_rewrite_2000_back",
            &opening_turn(&long_document(12_000)),
            REWRITTEN_BEYOND_SLACK,
            &beyond,
            max_tokens,
        ));
        requests
    })
}

fn run_oracle(args: &Args, recorder: &mut Recorder) {
    let bin = args.llama_bin.as_ref().expect("--llama-server-bin is required for the oracle");
    let requests = run_oracle_proxima(args);
    let context = requests.iter().map(|request| request.ids.len()).max().unwrap_or(0)
        + args.max_tokens
        + LLAMA_CONTEXT_HEADROOM_TOKENS;
    let mut identical = 0;
    let mut cached_equals_fresh = 0;
    let mut llama_is_prefix = 0;
    let mut scenarios: Vec<&'static str> = requests.iter().map(|request| request.scenario).collect();
    scenarios.dedup();
    for scenario in scenarios {
        let server = LlamaServer::spawn(bin, &args.model_path, LLAMA_PORT_ON, context, &[]);
        recorder.write(&json!({"kind": "llama_server", "scenario": scenario, "pid": server.pid()}));
        for request in requests.iter().filter(|request| request.scenario == scenario) {
            let llama = llama_completion(LLAMA_PORT_ON, &request.ids, args.max_tokens, true);
            let divergence = first_divergence(&request.cached, &llama.ids);
            identical += usize::from(divergence.is_none());
            llama_is_prefix += usize::from(request.cached.starts_with(&llama.ids));
            cached_equals_fresh += usize::from(request.cached == request.fresh);
            let record = oracle_record(request, &llama, divergence);
            println!("{record}");
            recorder.write(&record);
        }
    }
    println!(
        "oracle summary: proxima_through_cache_equals_llama={identical}/{total} llama_ids_are_a_prefix_of_proxima_ids={llama_is_prefix}/{total} proxima_cache_equals_proxima_fresh={cached_equals_fresh}/{total}",
        total = requests.len(),
    );
}

fn oracle_record(request: &OracleRequest, llama: &LlamaResult, divergence: Option<usize>) -> Value {
    json!({
        "kind": "oracle", "scenario": request.scenario, "request": request.index,
        "prompt_tokens": request.ids.len(), "expectation": request.expectation,
        "proxima_cache": path_label(Some(request.report)),
        "proxima_cached_ids": request.cached, "proxima_fresh_ids": request.fresh, "llama_ids": llama.ids,
        "identical_to_llama": divergence.is_none(), "first_divergence": divergence,
        "cached_equals_fresh": request.cached == request.fresh,
        "llama_ids_are_prefix_of_proxima": request.cached.starts_with(&llama.ids),
        "proxima_ids_past_llama_stop": request.cached.get(llama.ids.len()..).unwrap_or_default(),
        "llama_prompt_n": llama.timings["prompt_n"], "llama_cache_n": llama.timings["cache_n"],
        "llama_prompt_ms": llama.timings["prompt_ms"],
    })
}

fn run_llama_ttft(args: &Args, plans: &[TurnPlan], flush_ids: &[u32], recorder: &mut Recorder) {
    let bin = args.llama_bin.as_ref().expect("llama binary");
    let context = plans.iter().map(|plan| plan.prompt_ids.len()).max().unwrap_or(0)
        + args.max_tokens
        + LLAMA_CONTEXT_HEADROOM_TOKENS;
    let extra = ["--cache-ram", "0"];
    let on = LlamaServer::spawn(bin, &args.model_path, LLAMA_PORT_ON, context, &extra);
    let off = LlamaServer::spawn(bin, &args.model_path, LLAMA_PORT_OFF, context, &extra);
    recorder.write(&json!({"kind": "llama_server", "on_pid": on.pid(), "off_pid": off.pid()}));
    let mut on_ttft: Vec<Vec<f64>> = vec![Vec::new(); plans.len()];
    let mut off_ttft: Vec<Vec<f64>> = vec![Vec::new(); plans.len()];
    let mut on_prompt_ms: Vec<Vec<f64>> = vec![Vec::new(); plans.len()];
    let mut off_prompt_ms: Vec<Vec<f64>> = vec![Vec::new(); plans.len()];
    for pair in 0..=args.pairs {
        recorder.write(&host_snapshot(&format!("llama_pair_{pair}_start")));
        llama_completion(LLAMA_PORT_ON, flush_ids, 1, true);
        for (turn, plan) in plans.iter().enumerate() {
            let on_first = (pair + turn) % 2 == 0;
            let order = if on_first { [true, false] } else { [false, true] };
            for cache_on in order {
                let port = if cache_on { LLAMA_PORT_ON } else { LLAMA_PORT_OFF };
                let result = llama_completion(port, &plan.prompt_ids, args.max_tokens, cache_on);
                let prompt_ms = result.timings["prompt_ms"].as_f64().unwrap_or(f64::NAN);
                recorder.write(&json!({
                    "kind": "request", "mode": "llama_ttft", "pair": pair, "warmup": pair == 0, "turn": turn,
                    "arm": if cache_on { "llama_cache_on" } else { "llama_cache_off" },
                    "prompt_tokens": plan.prompt_ids.len(), "ttft_wall_ms": result.ttft_wall_ms,
                    "total_wall_ms": result.total_wall_ms, "prompt_ms": prompt_ms,
                    "prompt_n": result.timings["prompt_n"], "cache_n": result.timings["cache_n"],
                    "generated": result.ids.len(),
                    "first_divergence_vs_proxima": first_divergence(&plan.generated, &result.ids),
                }));
                if pair > 0 {
                    let (wall, prompt) = if cache_on { (&mut on_ttft, &mut on_prompt_ms) } else { (&mut off_ttft, &mut off_prompt_ms) };
                    wall[turn].push(result.ttft_wall_ms);
                    prompt[turn].push(prompt_ms);
                }
            }
        }
    }
    println!("llama-server ttft (wall to first streamed token; prompt_ms is llama's own timings.prompt_ms)");
    for turn in 0..plans.len() {
        println!("turn={turn} prompt_tokens={}", plans[turn].prompt_ids.len());
        println!("  cache_on  wall {}", spread(&on_ttft[turn]).text());
        println!("  cache_off wall {}", spread(&off_ttft[turn]).text());
        println!("  cache_on  prompt_ms {}", spread(&on_prompt_ms[turn]).text());
        println!("  cache_off prompt_ms {}", spread(&off_prompt_ms[turn]).text());
        println!("  median_on/median_off wall={:.3}", median(&on_ttft[turn]) / median(&off_ttft[turn]));
    }
}

fn run_ttft(args: &Args, recorder: &mut Recorder) {
    let (samples, plans, probe_ids) = with_model(&args.model_path, |model, vocab| {
        run_ttft_pairs(model, vocab, args, recorder)
    });
    print_ttft_summary(&samples, &plans);
    if args.llama_bin.is_some() {
        run_llama_ttft(args, &plans, &probe_ids, recorder);
    }
}

struct SplitArm {
    label: String,
    config: PromptCacheConfig,
    splits: usize,
}

fn split_arms(length: usize, intervals: &[u32]) -> Vec<SplitArm> {
    let splits_at = |interval: u32| (length - 1) / interval as usize;
    let standard = PromptCacheConfig::standard();
    let mut arms = vec![
        SplitArm { label: "cache_off".into(), config: PromptCacheConfig::off(), splits: 0 },
        SplitArm { label: "cache_on_max_checkpoints_0".into(), config: checkpointed(standard.checkpoint_interval, 0), splits: 0 },
        SplitArm {
            label: format!("cache_on_default_interval_{}", standard.checkpoint_interval),
            config: standard,
            splits: splits_at(standard.checkpoint_interval).min(standard.max_checkpoints as usize),
        },
    ];
    arms.extend(intervals.iter().map(|&interval| SplitArm {
        label: format!("cache_on_interval_{interval}_max_32"),
        config: checkpointed(interval, 32),
        splits: splits_at(interval).min(32),
    }));
    arms
}

fn run_split(args: &Args, recorder: &mut Recorder) {
    with_model(&args.model_path, |model, vocab| {
        let source = encode_opening(vocab, &opening_turn(&long_document(60_000)));
        let longest = args.lengths.iter().copied().max().expect("--lengths is not empty");
        assert!(source.len() >= longest, "the source prompt has {} tokens, need {longest}", source.len());
        for &length in &args.lengths {
            run_split_length(model, vocab, &source[..length], args, recorder);
        }
    });
}

fn run_split_length(model: &LoadedModel<'_>, vocab: &Vocab, ids: &[u32], args: &Args, recorder: &mut Recorder) {
    let arms = split_arms(ids.len(), &args.intervals);
    let mut ttft: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    for pair in 0..=args.pairs {
        recorder.write(&host_snapshot(&format!("split_{}_pair_{pair}_start", ids.len())));
        for step in 0..arms.len() {
            let index = (step + pair) % arms.len();
            let arm = &arms[index];
            clear_cache(model, vocab);
            let timed = timed_request(model, ids, 1, &config_with(arm.config));
            recorder.write(&json!({
                "kind": "request", "mode": "split", "pair": pair, "warmup": pair == 0, "arm": arm.label,
                "prompt_tokens": ids.len(), "planned_splits": arm.splits, "order_position": step,
                "ttft_ms": timed.ttft_ms, "total_ms": timed.total_ms, "cache": path_label(timed.report),
            }));
            if pair > 0 {
                ttft[index].push(timed.ttft_ms);
            }
        }
    }
    print_split_summary(ids.len(), &arms, &ttft);
}

fn print_split_summary(length: usize, arms: &[SplitArm], ttft: &[Vec<f64>]) {
    println!("split overhead, cold cache, prompt_tokens={length}, ttft ms (max_tokens=1)");
    let baseline = median(&ttft[1]);
    for (arm, values) in arms.iter().zip(ttft) {
        let delta = median(values) - baseline;
        let per_split = if arm.splits == 0 { f64::NAN } else { delta / arm.splits as f64 };
        println!(
            "  {:<42} splits={:<2} {}  vs_max_checkpoints_0_ms={delta:+.1} per_split_ms={per_split:.1}",
            arm.label,
            arm.splits,
            spread(values).text()
        );
    }
}

fn main() {
    let args = parse_args();
    let mut recorder = Recorder::create(&args.out);
    recorder.write(&build_summary());
    recorder.write(&host_snapshot("start"));
    match args.mode {
        Mode::Oracle => run_oracle(&args, &mut recorder),
        Mode::Ttft => run_ttft(&args, &mut recorder),
        Mode::Split => run_split(&args, &mut recorder),
    }
    recorder.write(&host_snapshot("end"));
}
