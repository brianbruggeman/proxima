//! End-to-end pipeline comparison, proxima against llama.cpp, apples to apples.
//!
//! Single command (CUDA):
//!
//! ```text
//! CARGO_TARGET_DIR=/mnt/data3/cargo-target-cuda-spec \
//! LD_LIBRARY_PATH=/home/bix/proxima-cuda-runtime/lib \
//! PROXIMA_CUDA_INCLUDE_PATH=/home/bix/proxima-cuda-runtime/include \
//! cargo run -p proxima-model-interop --features cuda --example pipeline_bench -- \
//!     /home/bix/models/qwen3-0.6b-q8_0.gguf "The capital of France is" 8 [repeats]
//! ```
//!
//! Usage: `pipeline_bench <model.gguf> [prompt] [tokens] [repeats]` (repeats
//! default 3). The binary re-runs itself as the proxima child
//! (`PIPELINE_BENCH_CHILD=1`) and runs `llama-completion` and `llama-tokenize`
//! from `PIPELINE_BENCH_LLAMA_DIR` (library path `PIPELINE_BENCH_LLAMA_LD`),
//! each child under `/usr/bin/time -v` (peak host RSS) with an `nvidia-smi
//! --query-compute-apps` sampler at 100 ms filtered to the child's process tree
//! (peak VRAM).
//!
//! Rows: proxima with activation quantization on, proxima with it off, llama
//! with f32 KV (matches proxima's `kv_cache_key_quant`/`kv_cache_value_quant`
//! default) and llama with its default f16 KV. Every row reads the same gguf
//! file, so a format llama cannot load shows `unsupported format` on its rows
//! and is never substituted. The prompt ids come from both tokenizers and are
//! compared before any row is marked comparable; generated ids are compared as
//! leading-id matches (llama's text is re-tokenized with `llama-tokenize
//! --no-bos`, since llama-completion prints no ids). llama gets the same
//! context (`-c`), `-b/-ub`, `-ngl 99`, `-fa` as proxima's `ServingConfig`,
//! printed in the settings table together with `PROXIMA_CUDA_KERNEL_CACHE_DIR`
//! state, GPU compute-app count and load average before and after each row.
//!
//! Warm vs cold: the proxima child loads, runs one discarded 2-token generation
//! (`device_warmup`: weights resident, kernels compiled, prompt cache off), then
//! `repeats` measured generations in-process; llama runs `repeats` processes
//! with its default empty warmup decode and reports each run's own `load time`.
//! `load_ms` for proxima is `load + device_warmup`. Cells read
//! `median [min-max]`.
//!
//! Proxima child stdout carries `pipeline engine=proxima stage=<name> ...`:
//! `load`, `device_warmup`, `config`, `run idx=<i>` (tokenize, ttft, prefill,
//! decode, total), `kv`. `ttft` is call start to step-0 token event (the
//! library tokenizes and builds the runtime inside the call); `prefill` is
//! `ttft - tokenize`, derived because the library fires its prefill event and
//! the first token event together; `decode` is the gap between token events
//! after the first. Llama columns come from `common_perf_print` (`load time`,
//! `prompt eval time` for prefill and ttft, `eval time` mean for decode,
//! `load + total time` for total). llama-completion does not time tokenization
//! (`n/a`) and logs no KV size, so `kv_mib` for llama rows is computed from the
//! checkpoint's `kv_layers` at `-c` positions and the row's KV byte width, and
//! for proxima at prompt + generated positions at 8 bytes per head-dim element.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::ops::ControlFlow;
use std::path::Path;
use std::process::{Child, Command, ExitCode, Stdio};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{
    ContextLength, GPU_LAYERS_ALL, InteropError, LoadedModel, Phase, PromptCacheConfig,
    ServingConfig, TokenEvent, kv_layers,
};
use proxima_tokenizer::gguf::vocab_from_metadata;
use proxima_tokenizer::{TokenizerError, Vocab, encode_with_bos_eos};

const DEFAULT_PROMPT: &str = "The capital of France is";
const DEFAULT_TOKENS: usize = 16;
const DEFAULT_REPEATS: usize = 3;
const MIN_CONTEXT: usize = 512;
const BATCH: u32 = 512;
const CHILD_ENV: &str = "PIPELINE_BENCH_CHILD";
const QUANTIZATION_ENV: &str = "DECODE_COMPARE_ACTIVATION_QUANTIZATION";
const KERNEL_CACHE_ENV: &str = "PROXIMA_CUDA_KERNEL_CACHE_DIR";
const DEFAULT_LLAMA_DIR: &str = "/home/bix/repos/llama.cpp-incumbent/build/bin";
const DEFAULT_LLAMA_LD: &str =
    "/usr/local/lib/ollama.stale-0.5.7.bak:/home/bix/repos/llama.cpp-incumbent/build/bin";
const WARMUP_TOKENS: usize = 2;
const PROMPT_TOKENS_ENV: &str = "PIPELINE_BENCH_PROMPT_TOKENS";
const PARAGRAPH: &str = "The history of computing is a story of steady abstraction. Early machines \
were programmed by rewiring plugboards, then by punched cards, and later by assembly languages that \
gave names to the numeric opcodes. Compilers followed, letting engineers describe what they wanted \
in terms closer to mathematics than to registers. Operating systems arrived to share one expensive \
machine among many people, and networks arrived to share information among many machines. Each \
layer hid the details of the one beneath it, and each hidden detail eventually returned as a \
performance problem that someone had to understand again from first principles. Today a single \
graphics processor can perform trillions of arithmetic operations every second, yet moving the \
numbers to where they are needed often costs more than computing with them.";
const MIB: f64 = 1024.0 * 1024.0;
const HEADER: [&str; 13] = [
    "row",
    "load_ms",
    "tokenize_ms",
    "prompt_tokens",
    "prefill_ms",
    "prefill_tok/s",
    "ttft_ms",
    "decode_ms/tok(mean)",
    "decode_tok/s",
    "total_ms",
    "kv_mib",
    "peak_host_rss_mb",
    "peak_vram_mb",
];
const WALL_HEADER: [&str; 7] = [
    "row",
    "run",
    "cached_kb_before",
    "cached_kb_after",
    "wall_to_ready_ms",
    "wall_to_first_token_ms",
    "self-reported load_ms",
];
const SETTINGS_HEADER: [&str; 11] = [
    "row",
    "kv_precision",
    "ctx",
    "flash_attn",
    "ngl",
    "batch/ubatch",
    "act_quant",
    "gpu_apps_before",
    "loadavg_start",
    "loadavg_end",
    "generated(n/leading_match)",
];

#[derive(Debug, thiserror::Error)]
enum BenchError {
    #[error("usage: pipeline_bench <model.gguf> [prompt] [tokens] [repeats]")]
    Usage,
    #[error("{path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("{path}: gguf parse: {message}")]
    Gguf { path: String, message: String },
    #[error("interop: {0}")]
    Interop(#[from] InteropError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("{0:?} is not a number")]
    BadCount(String),
    #[error("generation produced no token events")]
    NoTokens,
    #[error("vram sampler thread panicked")]
    Sampler,
    #[error("{what}: unparseable output {text:?}")]
    Parse { what: &'static str, text: String },
}

type Samples = HashMap<&'static str, Vec<f64>>;

struct Measured {
    wall: Wall,
    stdout: String,
    stderr: String,
    success: bool,
    peak_rss_mb: Option<f64>,
    peak_vram_mb: u64,
}

#[derive(Clone, Copy, Default)]
struct Wall {
    ready_ms: Option<f64>,
    first_byte_ms: Option<f64>,
    cached_before_kb: u64,
    cached_after_kb: u64,
}

struct Scan {
    text: String,
    first_byte_ms: Option<f64>,
    marker_ms: Option<f64>,
}

struct VramSampler {
    process: Child,
    reader: JoinHandle<u64>,
}

#[derive(Clone, Copy)]
struct RunArgs<'run> {
    path: &'run str,
    prompt: &'run str,
    tokens: usize,
    repeats: usize,
    context: usize,
}

struct RowResult {
    label: String,
    kv_precision: String,
    samples: Samples,
    ids_note: String,
    gpu_apps_before: usize,
    loadavg_start: String,
    loadavg_end: String,
    supported: bool,
    walls: Vec<WallRow>,
}

struct WallRow {
    run: usize,
    wall: Wall,
    self_reported_load_ms: Option<f64>,
    first_token_ms: Option<f64>,
}

fn io_error(path: &str) -> impl Fn(std::io::Error) -> BenchError + '_ {
    move |source| BenchError::Io {
        path: path.to_owned(),
        source,
    }
}

fn millis(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn quantization_enabled() -> bool {
    env::var(QUANTIZATION_ENV).is_ok_and(|value| value != "0")
}

fn config(context: usize) -> ServingConfig<'static> {
    let defaults = ServingConfig::default();
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        context_length: ContextLength::Within(u32::try_from(context).unwrap_or(u32::MAX)),
        batch_size: BATCH,
        ubatch_size: BATCH,
        numeric_policy: defaults
            .numeric_policy
            .with_activation_quantization(quantization_enabled()),
        prompt_cache: PromptCacheConfig::off(),
        ..defaults
    }
}

fn prompt_ids(vocab: &Vocab, prompt: &str) -> Result<Vec<u32>, TokenizerError> {
    let wants_bos = vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some());
    let wants_eos = vocab.add_eos_token().unwrap_or(false);
    encode_with_bos_eos(prompt, vocab, wants_bos, wants_eos)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn emit(stage: &str, fields: &str) {
    println!("pipeline engine=proxima stage={stage} {fields}");
}

fn kv_bytes(layers: &[(u32, u32, Option<u32>)], positions: u64, element_bytes: u64) -> u64 {
    layers
        .iter()
        .map(|&(kv_heads, head_dim, window)| {
            let rows = window.map_or(positions, |rows| positions.min(u64::from(rows)));
            u64::from(kv_heads) * u64::from(head_dim) * 2 * element_bytes * rows
        })
        .sum()
}

struct Generation {
    token_ms: Vec<f64>,
    ids: Vec<u32>,
    text: String,
}

fn generate(
    model: &LoadedModel,
    prompt: &str,
    tokens: usize,
    context: usize,
) -> Result<Generation, BenchError> {
    let mut token_ms = Vec::new();
    let started = Instant::now();
    let mut on_token = |event: TokenEvent<'_>| {
        if matches!(event.phase, Phase::Token) {
            token_ms.push(millis(started));
        }
        ControlFlow::Continue(())
    };
    let (ids, text, _stopped) =
        model.generate_streaming(prompt, tokens, config(context), &mut on_token)?;
    Ok(Generation {
        token_ms,
        ids,
        text,
    })
}

fn report_run(
    index: usize,
    generation: &Generation,
    tokenize_ms: f64,
    prompt_tokens: usize,
    setup_ms: f64,
) -> Result<(), BenchError> {
    let ttft = *generation.token_ms.first().ok_or(BenchError::NoTokens)?;
    let last = *generation.token_ms.last().ok_or(BenchError::NoTokens)?;
    let prefill = (ttft - tokenize_ms).max(f64::EPSILON);
    let steps: Vec<f64> = generation
        .token_ms
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    let decode_mean = mean(&steps).max(f64::EPSILON);
    emit(
        "run",
        &format!(
            "idx={index} tokenize_ms={tokenize_ms:.6} ttft_ms={ttft:.3} prefill_ms={prefill:.3} \
             prefill_tps={:.3} decode_ms={decode_mean:.3} decode_median_ms={:.3} decode_tps={:.3} \
             steps={} total_ms={:.3}",
            prompt_tokens as f64 / prefill * 1000.0,
            median(&steps),
            1000.0 / decode_mean,
            steps.len(),
            setup_ms + tokenize_ms + last
        ),
    );
    Ok(())
}

fn run_child(args: &RunArgs<'_>) -> Result<(), BenchError> {
    let path = args.path;
    let load_started = Instant::now();
    let file = File::open(path).map_err(io_error(path))?;
    // the checkpoint is not written while this process reads it
    let mapping = unsafe { Mmap::map(&file) }.map_err(io_error(path))?;
    let parsed = parse_complete(&mapping).map_err(|error| BenchError::Gguf {
        path: path.to_owned(),
        message: error.to_string(),
    })?;
    let vocab = vocab_from_metadata(&parsed)?;
    let model = LoadedModel::load(&parsed, &mapping)?;
    let load_ms = millis(load_started);
    emit("load", &format!("ms={load_ms:.3}"));
    let warmup_started = Instant::now();
    generate(&model, args.prompt, WARMUP_TOKENS, args.context)?;
    let warmup_ms = millis(warmup_started);
    emit("device_warmup", &format!("ms={warmup_ms:.3}"));
    emit("ready", "");
    let settings = config(args.context);
    emit(
        "config",
        &format!(
            "ctx={} kv_k={:?} kv_v={:?} flash_attention={} batch={} ubatch={} act_quant={}",
            args.context,
            settings.kv_cache_key_quant,
            settings.kv_cache_value_quant,
            settings.flash_attention,
            settings.batch_size,
            settings.ubatch_size,
            u8::from(quantization_enabled())
        ),
    );
    let mut last = None;
    let mut prompt_token_ids = Vec::new();
    for index in 0..args.repeats {
        let tokenize_started = Instant::now();
        prompt_token_ids = prompt_ids(&vocab, args.prompt)?;
        let tokenize_ms = millis(tokenize_started);
        let generation = generate(&model, args.prompt, args.tokens, args.context)?;
        report_run(
            index,
            &generation,
            tokenize_ms,
            prompt_token_ids.len(),
            load_ms + warmup_ms,
        )?;
        last = Some(generation);
    }
    let generation = last.ok_or(BenchError::NoTokens)?;
    let layers = kv_layers(&parsed)?;
    let positions = (prompt_token_ids.len() + generation.ids.len()) as u64;
    emit(
        "kv",
        &format!(
            "bytes={} llama_f32_ctx_bytes={} llama_f16_ctx_bytes={}",
            kv_bytes(&layers, positions, 4),
            kv_bytes(&layers, args.context as u64, 4),
            kv_bytes(&layers, args.context as u64, 2)
        ),
    );
    println!(
        "output prompt_ids={prompt_token_ids:?} ids={:?}",
        generation.ids
    );
    println!("output text={:?}", generation.text);
    Ok(())
}

fn parent_pid(pid: u32) -> Option<u32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_name = stat.rsplit_once(')')?.1;
    after_name.split_whitespace().nth(1)?.parse().ok()
}

fn descends_from(pid: u32, root: u32) -> bool {
    let mut current = pid;
    while current > 1 {
        if current == root {
            return true;
        }
        match parent_pid(current) {
            Some(parent) => current = parent,
            None => return false,
        }
    }
    false
}

fn parse_sample(line: &str) -> Option<(u32, u64)> {
    let (pid, used) = line.split_once(',')?;
    Some((pid.trim().parse().ok()?, used.trim().parse().ok()?))
}

fn start_sampler(root: u32) -> Result<VramSampler, BenchError> {
    let mut process = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
            "-lms",
            "100",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(io_error("nvidia-smi"))?;
    let stdout = process.stdout.take().ok_or(BenchError::Sampler)?;
    let reader = thread::spawn(move || {
        BufReader::new(stdout)
            .lines()
            .map_while(Result::ok)
            .filter_map(|line| parse_sample(&line))
            .filter(|&(pid, _)| descends_from(pid, root))
            .map(|(_, used)| used)
            .max()
            .unwrap_or(0)
    });
    Ok(VramSampler { process, reader })
}

fn finish_sampler(sampler: VramSampler) -> Result<u64, BenchError> {
    let VramSampler {
        mut process,
        reader,
    } = sampler;
    process.kill().map_err(io_error("nvidia-smi"))?;
    process.wait().map_err(io_error("nvidia-smi"))?;
    reader.join().map_err(|_| BenchError::Sampler)
}

fn gpu_app_count() -> usize {
    Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader"])
        .output()
        .map_or(0, |output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count()
        })
}

fn peak_rss_mb(stderr: &str) -> Option<f64> {
    let line = stderr
        .lines()
        .find(|line| line.contains("Maximum resident set size"))?;
    let kilobytes: f64 = line.rsplit(':').next()?.trim().parse().ok()?;
    Some(kilobytes / 1024.0)
}

fn cached_kb() -> u64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("Cached:"))
                .and_then(|rest| rest.split_whitespace().next()?.parse().ok())
        })
        .unwrap_or(0)
}

fn scan_stream<R: Read>(mut stream: R, marker: &str, started: Instant) -> Scan {
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 4096];
    let mut first_byte_ms = None;
    let mut marker_ms = None;
    while let Ok(count) = stream.read(&mut chunk) {
        if count == 0 {
            break;
        }
        first_byte_ms.get_or_insert_with(|| millis(started));
        bytes.extend_from_slice(&chunk[..count]);
        if marker_ms.is_none() && !marker.is_empty() {
            let found = String::from_utf8_lossy(&bytes).contains(marker);
            marker_ms = found.then(|| millis(started));
        }
    }
    Scan {
        text: String::from_utf8_lossy(&bytes).into_owned(),
        first_byte_ms,
        marker_ms,
    }
}

fn measure(
    program: &str,
    args: &[String],
    envs: &[(&str, String)],
    ready_marker: &str,
) -> Result<Measured, BenchError> {
    let cached_before_kb = cached_kb();
    let started = Instant::now();
    let mut child = Command::new("/usr/bin/time")
        .arg("-v")
        .arg(program)
        .args(args)
        .envs(envs.iter().map(|(key, value)| (*key, value)))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(io_error(program))?;
    let sampler = start_sampler(child.id())?;
    let stdout = child.stdout.take().ok_or(BenchError::Sampler)?;
    let stderr = child.stderr.take().ok_or(BenchError::Sampler)?;
    let marker = ready_marker.to_owned();
    let stdout_marker = marker.clone();
    let stdout_reader = thread::spawn(move || scan_stream(stdout, &stdout_marker, started));
    let stderr_reader = thread::spawn(move || scan_stream(stderr, &marker, started));
    let status = child.wait().map_err(io_error(program))?;
    let out = stdout_reader.join().map_err(|_| BenchError::Sampler)?;
    let err = stderr_reader.join().map_err(|_| BenchError::Sampler)?;
    if env::var_os("PIPELINE_BENCH_ECHO_CHILD_STDERR").is_some() {
        eprint!("{}", err.text);
    }
    Ok(Measured {
        wall: Wall {
            ready_ms: out.marker_ms.or(err.marker_ms),
            first_byte_ms: out.first_byte_ms,
            cached_before_kb,
            cached_after_kb: cached_kb(),
        },
        stdout: out.text,
        peak_rss_mb: peak_rss_mb(&err.text),
        stderr: err.text,
        success: status.success(),
        peak_vram_mb: finish_sampler(sampler)?,
    })
}


fn parse_stage_lines(stdout: &str) -> Vec<HashMap<String, String>> {
    stdout
        .lines()
        .filter(|line| line.starts_with("pipeline "))
        .map(|line| {
            line.split_whitespace()
                .filter_map(|token| token.split_once('='))
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .collect()
        })
        .collect()
}

fn stage_number(lines: &[HashMap<String, String>], stage: &str, key: &str) -> Option<f64> {
    lines
        .iter()
        .find(|fields| fields.get("stage").is_some_and(|name| name == stage))?
        .get(key)?
        .parse()
        .ok()
}

fn push(samples: &mut Samples, key: &'static str, value: Option<f64>) {
    if let Some(number) = value {
        samples.entry(key).or_default().push(number);
    }
}

fn parse_id_list(text: &str) -> Option<Vec<u32>> {
    let inner = text.trim().strip_prefix('[')?.strip_suffix(']')?;
    inner
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse().ok())
        .collect()
}

fn output_ids(stdout: &str, key: &str) -> Option<Vec<u32>> {
    let line = stdout
        .lines()
        .find(|line| line.starts_with("output prompt_ids="))?;
    let start = line.find(&format!(" {key}="))? + key.len() + 2;
    let rest = &line[start..];
    parse_id_list(&rest[..=rest.find(']')?])
}

fn proxima_samples(measured: &Measured) -> Samples {
    let lines = parse_stage_lines(&measured.stdout);
    let mut samples = Samples::new();
    let setup = stage_number(&lines, "load", "ms").zip(stage_number(&lines, "device_warmup", "ms"));
    push(
        &mut samples,
        "load",
        setup.map(|(open, warmup)| open + warmup),
    );
    for fields in lines
        .iter()
        .filter(|fields| fields.get("stage").is_some_and(|name| name == "run"))
    {
        for (key, field) in [
            ("tokenize", "tokenize_ms"),
            ("prefill", "prefill_ms"),
            ("prefill_tps", "prefill_tps"),
            ("ttft", "ttft_ms"),
            ("decode", "decode_ms"),
            ("decode_tps", "decode_tps"),
            ("total", "total_ms"),
        ] {
            push(
                &mut samples,
                key,
                fields.get(field).and_then(|text| text.parse().ok()),
            );
        }
    }
    push(&mut samples, "rss", measured.peak_rss_mb);
    push(&mut samples, "vram", Some(measured.peak_vram_mb as f64));
    samples
}

fn perf_line(stderr: &str, label: &str) -> Option<(f64, Option<f64>)> {
    stderr.lines().find_map(|line| {
        let rest = line.split_once("common_perf_print:")?.1.trim();
        let values = rest.strip_prefix(label)?.trim().strip_prefix('=')?;
        let mut tokens = values.split_whitespace();
        let time_ms: f64 = tokens.next()?.parse().ok()?;
        let count = tokens
            .skip_while(|token| *token != "/")
            .nth(1)
            .and_then(|token| token.parse().ok());
        Some((time_ms, count))
    })
}

fn add_llama_run(samples: &mut Samples, measured: &Measured) {
    let stderr = &measured.stderr;
    let load = perf_line(stderr, "load time").map(|(time, _)| time);
    let prompt = perf_line(stderr, "prompt eval time");
    let decode = perf_line(stderr, "eval time");
    let total = perf_line(stderr, "total time").map(|(time, _)| time);
    let decode_mean = decode.and_then(|(time, runs)| runs.map(|count| time / count.max(1.0)));
    push(samples, "load", load);
    push(
        samples,
        "prompt_tokens",
        prompt.and_then(|(_, count)| count),
    );
    push(samples, "prefill", prompt.map(|(time, _)| time));
    push(
        samples,
        "prefill_tps",
        prompt.and_then(|(time, count)| count.map(|tokens| tokens / time * 1000.0)),
    );
    push(samples, "ttft", prompt.map(|(time, _)| time));
    push(samples, "decode", decode_mean);
    push(
        samples,
        "decode_tps",
        decode_mean.map(|mean_ms| 1000.0 / mean_ms),
    );
    push(
        samples,
        "total",
        load.zip(total).map(|(open, run)| open + run),
    );
    push(samples, "rss", measured.peak_rss_mb);
    push(samples, "vram", Some(measured.peak_vram_mb as f64));
}

fn spread(values: Option<&Vec<f64>>, digits: usize) -> String {
    match values {
        Some(found) if !found.is_empty() => {
            let low = found.iter().copied().fold(f64::INFINITY, f64::min);
            let high = found.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            format!(
                "{:.digits$} [{low:.digits$}-{high:.digits$}]",
                median(found)
            )
        }
        _ => "n/a".to_owned(),
    }
}

fn row_cells(row: &RowResult, kv_mib: Option<f64>, prompt_tokens: Option<usize>) -> Vec<String> {
    if !row.supported {
        let mut cells = vec![row.label.clone()];
        cells.extend(HEADER[1..].iter().map(|_| "unsupported format".to_owned()));
        return cells;
    }
    let samples = &row.samples;
    let tokens = prompt_tokens.map_or_else(
        || spread(samples.get("prompt_tokens"), 0),
        |count| count.to_string(),
    );
    vec![
        row.label.clone(),
        spread(samples.get("load"), 1),
        spread(samples.get("tokenize"), 4),
        tokens,
        spread(samples.get("prefill"), 2),
        spread(samples.get("prefill_tps"), 1),
        spread(samples.get("ttft"), 2),
        spread(samples.get("decode"), 3),
        spread(samples.get("decode_tps"), 1),
        spread(samples.get("total"), 1),
        kv_mib.map_or_else(|| "n/a".to_owned(), |mib| format!("{mib:.2}")),
        spread(samples.get("rss"), 0),
        spread(samples.get("vram"), 0),
    ]
}

fn render_table(header: &[&str], rows: &[Vec<String>]) -> String {
    let line = |cells: Vec<&str>| format!("| {} |", cells.join(" | "));
    let mut lines = vec![
        line(header.to_vec()),
        line(header.iter().map(|_| "---").collect()),
    ];
    lines.extend(
        rows.iter()
            .map(|row| line(row.iter().map(String::as_str).collect())),
    );
    lines.join("\n")
}

fn load_average() -> String {
    fs::read_to_string("/proc/loadavg")
        .map(|text| {
            text.split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_else(|_| "unavailable".to_owned())
}

fn last_lines(text: &str, count: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(count)..].join(" / ")
}

fn llama_dir() -> String {
    env::var("PIPELINE_BENCH_LLAMA_DIR").unwrap_or_else(|_| DEFAULT_LLAMA_DIR.to_owned())
}

fn llama_env() -> [(&'static str, String); 1] {
    [(
        "LD_LIBRARY_PATH",
        env::var("PIPELINE_BENCH_LLAMA_LD").unwrap_or_else(|_| DEFAULT_LLAMA_LD.to_owned()),
    )]
}

fn llama_tokenize(path: &str, text: &str, add_bos: bool) -> Result<Vec<u32>, BenchError> {
    let mut args = vec![
        "-m".to_owned(),
        path.to_owned(),
        "-p".to_owned(),
        text.to_owned(),
    ];
    args.extend(["--ids".to_owned(), "--log-disable".to_owned()]);
    if !add_bos {
        args.push("--no-bos".to_owned());
    }
    let measured = measure(
        &format!("{}/llama-tokenize", llama_dir()),
        &args,
        &llama_env(),
        "",
    )?;
    parse_id_list(measured.stdout.trim()).ok_or(BenchError::Parse {
        what: "llama-tokenize",
        text: last_lines(&measured.stdout, 3),
    })
}

fn leading_match(left: &[u32], right: &[u32]) -> usize {
    left.iter().zip(right).take_while(|(a, b)| a == b).count()
}

fn first_difference(left: &[u32], right: &[u32]) -> Option<usize> {
    (left.len() != right.len())
        .then(|| leading_match(left, right))
        .or_else(|| left.iter().zip(right).position(|(a, b)| a != b))
}

fn llama_args(args: &RunArgs<'_>, kv_type: &str, flash_attention: bool) -> Vec<String> {
    let mut command = vec![
        "-m",
        args.path,
        "-p",
        args.prompt,
        "-ngl",
        "99",
        "-fa",
        if flash_attention { "on" } else { "off" },
        "-ctk",
        kv_type,
        "-ctv",
        kv_type,
        "--temp",
        "0",
        "--top-k",
        "1",
        "-no-cnv",
        "--no-display-prompt",
        "--perf",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    for (flag, value) in [
        ("-n", args.tokens.to_string()),
        ("-c", args.context.to_string()),
        ("-b", BATCH.to_string()),
        ("-ub", BATCH.to_string()),
    ] {
        command.push(flag.to_owned());
        command.push(value);
    }
    command
}

struct Reference {
    prompt_ids: Vec<u32>,
    generated_ids: Vec<u32>,
    flash_attention: bool,
}

fn run_llama_row(
    args: &RunArgs<'_>,
    kv_type: &str,
    reference: &Reference,
) -> Result<RowResult, BenchError> {
    let mut row = RowResult {
        label: format!("llama (kv {kv_type})"),
        kv_precision: kv_type.to_owned(),
        samples: Samples::new(),
        ids_note: String::new(),
        gpu_apps_before: gpu_app_count(),
        loadavg_start: load_average(),
        loadavg_end: String::new(),
        supported: true,
        walls: Vec::new(),
    };
    let mut text = String::new();
    for run in 1..=args.repeats {
        let measured = measure(
            &format!("{}/llama-completion", llama_dir()),
            &llama_args(args, kv_type, reference.flash_attention),
            &llama_env(),
            "generate:",
        )?;
        if !measured.success {
            row.supported = false;
            row.ids_note = last_lines(&measured.stderr, 4);
            break;
        }
        add_llama_run(&mut row.samples, &measured);
        row.walls.push(WallRow {
            run,
            wall: measured.wall,
            self_reported_load_ms: perf_line(&measured.stderr, "load time").map(|(time, _)| time),
            first_token_ms: measured.wall.first_byte_ms,
        });
        text = measured.stdout.trim_end_matches('\n').to_owned();
    }
    row.loadavg_end = load_average();
    if row.supported {
        let ids = llama_tokenize(args.path, &text, false)?;
        row.ids_note = format!(
            "{}/{}",
            ids.len(),
            leading_match(&ids, &reference.generated_ids)
        );
    }
    Ok(row)
}

fn run_proxima_row(
    args: &RunArgs<'_>,
    quantization: &str,
) -> Result<(RowResult, Measured), BenchError> {
    let executable = env::current_exe().map_err(io_error("current_exe"))?;
    let mut row = RowResult {
        label: format!("proxima (act_quant {quantization})"),
        kv_precision: "pending".to_owned(),
        samples: Samples::new(),
        ids_note: String::new(),
        gpu_apps_before: gpu_app_count(),
        loadavg_start: load_average(),
        loadavg_end: String::new(),
        supported: true,
        walls: Vec::new(),
    };
    let child_args = [
        args.path.to_owned(),
        args.prompt.to_owned(),
        args.tokens.to_string(),
        args.repeats.to_string(),
        args.context.to_string(),
    ];
    let measured = measure(
        &executable.to_string_lossy(),
        &child_args,
        &[
            (CHILD_ENV, "1".to_owned()),
            (QUANTIZATION_ENV, quantization.to_owned()),
        ],
        "stage=ready",
    )?;
    row.loadavg_end = load_average();
    row.samples = proxima_samples(&measured);
    row.walls.push(proxima_wall_row(&measured));
    Ok((row, measured))
}

fn proxima_wall_row(measured: &Measured) -> WallRow {
    let lines = parse_stage_lines(&measured.stdout);
    let first_run = lines
        .iter()
        .find(|fields| fields.get("stage").is_some_and(|name| name == "run"));
    let run_ms = |key: &str| first_run.and_then(|fields| fields.get(key)?.parse::<f64>().ok());
    let first_token_ms = measured
        .wall
        .ready_ms
        .zip(run_ms("tokenize_ms"))
        .zip(run_ms("ttft_ms"))
        .map(|((ready, tokenize), ttft)| ready + tokenize + ttft);
    let self_reported_load_ms = stage_number(&lines, "load", "ms")
        .zip(stage_number(&lines, "device_warmup", "ms"))
        .map(|(open, warmup)| open + warmup);
    WallRow {
        run: 1,
        wall: measured.wall,
        self_reported_load_ms,
        first_token_ms,
    }
}

fn wall_cells(row: &RowResult) -> Vec<Vec<String>> {
    let cell = |value: Option<f64>| value.map_or_else(|| "n/a".to_owned(), |ms| format!("{ms:.1}"));
    row.walls
        .iter()
        .map(|entry| {
            vec![
                row.label.clone(),
                entry.run.to_string(),
                entry.wall.cached_before_kb.to_string(),
                entry.wall.cached_after_kb.to_string(),
                cell(entry.wall.ready_ms),
                cell(entry.first_token_ms),
                cell(entry.self_reported_load_ms),
            ]
        })
        .collect()
}

fn settings_cells(
    row: &RowResult,
    context: usize,
    flash_attention: bool,
    quant: &str,
) -> Vec<String> {
    vec![
        row.label.clone(),
        row.kv_precision.clone(),
        context.to_string(),
        if flash_attention { "on" } else { "off" }.to_owned(),
        "all (99)".to_owned(),
        format!("{BATCH}/{BATCH}"),
        quant.to_owned(),
        row.gpu_apps_before.to_string(),
        row.loadavg_start.clone(),
        row.loadavg_end.clone(),
        row.ids_note.clone(),
    ]
}

fn kernel_cache_state() -> String {
    let directory = env::var(KERNEL_CACHE_ENV).unwrap_or_else(|_| {
        format!(
            "{}/.cache/proxima/cuda-ptx",
            env::var("HOME").unwrap_or_default()
        )
    });
    let entries = fs::read_dir(&directory).map_or(0, Iterator::count);
    format!("{directory} entries_at_start={entries}")
}

fn words_prompt(words: &[&str], count: usize) -> String {
    words.iter().cycle().take(count).copied().collect::<Vec<_>>().join(" ")
}

fn sized_prompt(path: &str, target: usize) -> Result<String, BenchError> {
    let file = File::open(path).map_err(io_error(path))?;
    // the checkpoint is not written while this process reads it
    let mapping = unsafe { Mmap::map(&file) }.map_err(io_error(path))?;
    let parsed = parse_complete(&mapping).map_err(|error| BenchError::Gguf {
        path: path.to_owned(),
        message: error.to_string(),
    })?;
    let vocab = vocab_from_metadata(&parsed)?;
    let words: Vec<&str> = PARAGRAPH.split_whitespace().collect();
    let (mut low, mut high) = (1, target.max(1) * 2);
    while low < high {
        let mid = (low + high) / 2;
        let ids = prompt_ids(&vocab, &words_prompt(&words, mid))?;
        if ids.len() < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    Ok(words_prompt(&words, low))
}

fn prompt_tokens_target() -> Result<Option<usize>, BenchError> {
    env::var(PROMPT_TOKENS_ENV).ok().map(parse_number).transpose()
}

fn orchestrate(args: &RunArgs<'_>) -> Result<(), BenchError> {
    let cache_state = kernel_cache_state();
    // a format llama cannot open fails here too; its rows then report unsupported
    let llama_prompt_ids = llama_tokenize(args.path, args.prompt, true).unwrap_or_default();
    let context = (llama_prompt_ids.len() + args.tokens).max(MIN_CONTEXT);
    let args = RunArgs { context, ..*args };
    let (quantized, quantized_run) = run_proxima_row(&args, "1")?;
    let (plain, _) = run_proxima_row(&args, "0")?;
    let lines = parse_stage_lines(&quantized_run.stdout);
    let config_line = lines
        .iter()
        .find(|fields| fields.get("stage").is_some_and(|name| name == "config"))
        .cloned()
        .unwrap_or_default();
    let flash_attention = config_line
        .get("flash_attention")
        .is_some_and(|value| value == "true");
    let kv_precision = format!(
        "{}/{}",
        config_line.get("kv_k").map_or("?", String::as_str),
        config_line.get("kv_v").map_or("?", String::as_str)
    );
    let proxima_prompt_ids = output_ids(&quantized_run.stdout, "prompt_ids").unwrap_or_default();
    let proxima_ids = output_ids(&quantized_run.stdout, "ids").unwrap_or_default();
    let comparable = !llama_prompt_ids.is_empty() && proxima_prompt_ids == llama_prompt_ids;
    let reference = Reference {
        prompt_ids: llama_prompt_ids,
        generated_ids: proxima_ids.clone(),
        flash_attention,
    };
    let f32_row = run_llama_row(&args, "f32", &reference)?;
    let f16_row = run_llama_row(&args, "f16", &reference)?;
    let kv_bytes_of = |key: &str| stage_number(&lines, "kv", key).map(|bytes| bytes / MIB);
    let quantized = RowResult {
        kv_precision: kv_precision.clone(),
        ids_note: format!("{}/{}", proxima_ids.len(), proxima_ids.len()),
        ..quantized
    };
    let plain = RowResult {
        kv_precision,
        ids_note: format!("{}/{}", proxima_ids.len(), proxima_ids.len()),
        ..plain
    };
    let tokens = Some(proxima_prompt_ids.len());
    let rows = [
        row_cells(&quantized, kv_bytes_of("bytes"), tokens),
        row_cells(&plain, kv_bytes_of("bytes"), tokens),
        row_cells(&f32_row, kv_bytes_of("llama_f32_ctx_bytes"), None),
        row_cells(&f16_row, kv_bytes_of("llama_f16_ctx_bytes"), None),
    ];
    println!("{}", render_table(&HEADER, &rows));
    let settings = [
        settings_cells(&quantized, context, flash_attention, "1"),
        settings_cells(&plain, context, flash_attention, "0"),
        settings_cells(&f32_row, context, flash_attention, "n/a (q8_1 matvec)"),
        settings_cells(&f16_row, context, flash_attention, "n/a (q8_1 matvec)"),
    ];
    println!("\n{}", render_table(&SETTINGS_HEADER, &settings));
    let wall_rows: Vec<Vec<String>> = [&quantized, &plain, &f32_row, &f16_row]
        .into_iter()
        .flat_map(wall_cells)
        .collect();
    println!("\n{}", render_table(&WALL_HEADER, &wall_rows));
    let model = Path::new(args.path)
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    println!(
        "\nnotes: model={model} (same file for all rows) prompt={:?} n={} repeats={} kernel_cache={cache_state}",
        args.prompt, args.tokens, args.repeats
    );
    println!(
        "prompt_ids comparable={comparable} proxima_count={} llama_count={} first_difference={:?}",
        proxima_prompt_ids.len(),
        reference.prompt_ids.len(),
        first_difference(&proxima_prompt_ids, &reference.prompt_ids)
    );
    println!("proxima generated ids={proxima_ids:?}");
    if !quantized_run.success {
        println!("proxima failed: {}", last_lines(&quantized_run.stderr, 6));
    }
    for line in quantized_run
        .stdout
        .lines()
        .filter(|line| line.starts_with("output text="))
    {
        println!("proxima {line}");
    }
    for row in [&f32_row, &f16_row].iter().filter(|row| !row.supported) {
        println!("{} unsupported format: {}", row.label, row.ids_note);
    }
    Ok(())
}

fn parse_number(text: String) -> Result<usize, BenchError> {
    text.parse().map_err(|_| BenchError::BadCount(text))
}

fn parse_args() -> Result<(String, String, usize, usize, usize), BenchError> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or(BenchError::Usage)?;
    let prompt = args.next().unwrap_or_else(|| DEFAULT_PROMPT.to_owned());
    let tokens = args.next().map_or(Ok(DEFAULT_TOKENS), parse_number)?;
    let repeats = args.next().map_or(Ok(DEFAULT_REPEATS), parse_number)?;
    let context = args.next().map_or(Ok(MIN_CONTEXT), parse_number)?;
    Ok((path, prompt, tokens, repeats, context))
}

fn main() -> ExitCode {
    let outcome = parse_args().and_then(|(path, prompt, tokens, repeats, context)| {
        let sized = prompt_tokens_target()?
            .map(|target| sized_prompt(&path, target))
            .transpose()?;
        let prompt = sized.unwrap_or(prompt);
        let args = RunArgs {
            path: &path,
            prompt: &prompt,
            tokens,
            repeats,
            context,
        };
        if env::var_os(CHILD_ENV).is_some() {
            run_child(&args)
        } else {
            orchestrate(&args)
        }
    });
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("pipeline_bench: {error}");
            ExitCode::FAILURE
        }
    }
}
