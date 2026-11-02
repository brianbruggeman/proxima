//! End-to-end pipeline comparison, proxima against llama.cpp, one markdown table.
//!
//! Single command (CUDA, activation quantization under test):
//!
//! ```text
//! CARGO_TARGET_DIR=/mnt/data3/cargo-target-cuda-spec \
//! LD_LIBRARY_PATH=/home/bix/proxima-cuda-runtime/lib \
//! PROXIMA_CUDA_INCLUDE_PATH=/home/bix/proxima-cuda-runtime/include \
//! cargo run -p proxima-model-interop --features cuda --example pipeline_bench -- \
//!     /home/bix/models/qwen3-0.6b-q8_0.gguf "The capital of France is" 8
//! ```
//!
//! Usage: `pipeline_bench <model.gguf> [prompt] [tokens]`. The binary runs itself
//! as the proxima child (`PIPELINE_BENCH_CHILD=1`) and `llama-simple` as the llama
//! child, each under `/usr/bin/time -v` (peak host RSS) with an `nvidia-smi
//! --query-compute-apps` sampler at 100 ms filtered to the child's process tree
//! (peak VRAM). Both engines get the same file, prompt, token count, greedy
//! sampling and all layers offloaded. A llama child that exits non-zero prints as
//! `unsupported format` (PQ2_0 / PTQ1_0 checkpoints).
//!
//! `DECODE_COMPARE_ACTIVATION_QUANTIZATION` is the setting under test: the
//! orchestrator defaults it to `1` for the proxima child, `0` turns it off, and
//! the value is printed in the notes. `PIPELINE_BENCH_LLAMA_DIR` and
//! `PIPELINE_BENCH_LLAMA_LD` override the llama binary directory and its
//! `LD_LIBRARY_PATH`.
//!
//! Proxima child stdout carries one line per stage,
//! `pipeline engine=proxima stage=<name> ms=...`:
//! - `load`: open + mmap + gguf parse + vocab + `LoadedModel::load`.
//! - `device_warmup`: a 2-token `generate_streaming` with the prompt cache off,
//!   which is what makes weights resident on the device and compiles kernels;
//!   the table's `load_ms` is `load + device_warmup`.
//! - `tokenize`: `encode_with_bos_eos` on the prompt.
//! - `ttft`: call start of the measured `generate_streaming` to the step-0 token
//!   event; the library tokenizes and builds the runtime inside that call.
//! - `prefill`: `ttft - tokenize`, derived because the library emits its prefill
//!   event and the first token event at the same point.
//! - `decode`: gaps between consecutive token events after the first.
//! - `kv`: bytes computed from `kv_layers` at `kv_heads * head_dim * 8` per stored
//!   position (the placed-KV formula in `memory_fit::kv_row_bytes`), at
//!   prompt + generated positions, windowed layers capped at their window.
//! - `total`: in-process wall from child start to generation end.
//!
//! Llama columns come from `llama_perf_context_print`: `load time`, `prompt eval
//! time` (prefill and ttft), `eval time` (decode mean; llama-simple exposes no
//! per-token times), `total time` + load (total), and the `KV buffer size` log
//! lines (kv). llama-simple does not time tokenization, so that cell is `n/a`.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::ops::ControlFlow;
use std::path::Path;
use std::process::{Child, Command, ExitCode, Stdio};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{
    GPU_LAYERS_ALL, InteropError, LoadedModel, Phase, PromptCacheConfig, ServingConfig, TokenEvent,
    kv_layers,
};
use proxima_tokenizer::gguf::vocab_from_metadata;
use proxima_tokenizer::{TokenizerError, Vocab, encode_with_bos_eos};

const DEFAULT_PROMPT: &str = "The capital of France is";
const DEFAULT_TOKENS: usize = 16;
const CHILD_ENV: &str = "PIPELINE_BENCH_CHILD";
const QUANTIZATION_ENV: &str = "DECODE_COMPARE_ACTIVATION_QUANTIZATION";
const DEFAULT_LLAMA_DIR: &str = "/home/bix/repos/llama.cpp-incumbent/build/bin";
const DEFAULT_LLAMA_LD: &str =
    "/usr/local/lib/ollama.stale-0.5.7.bak:/home/bix/repos/llama.cpp-incumbent/build/bin";
const WARMUP_TOKENS: usize = 2;
const MIB: f64 = 1024.0 * 1024.0;
const HEADER: [&str; 14] = [
    "engine",
    "load_ms",
    "tokenize_ms",
    "prompt_tokens",
    "prefill_ms",
    "prefill_tok/s",
    "ttft_ms",
    "decode_ms/tok(mean)",
    "decode_tok/s",
    "total_ms",
    "wall_ms",
    "kv_mib",
    "peak_host_rss_mb",
    "peak_vram_mb",
];

#[derive(Debug, thiserror::Error)]
enum BenchError {
    #[error("usage: pipeline_bench <model.gguf> [prompt] [tokens]")]
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
    #[error("token count {0:?} is not a number")]
    BadCount(String),
    #[error("generation produced no token events")]
    NoTokens,
    #[error("vram sampler thread panicked")]
    Sampler,
}

type Stages = HashMap<String, HashMap<String, String>>;

struct Measured {
    stdout: String,
    stderr: String,
    success: bool,
    wall_ms: f64,
    peak_rss_mb: Option<f64>,
    peak_vram_mb: u64,
}

struct VramSampler {
    process: Child,
    reader: JoinHandle<u64>,
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

fn config() -> ServingConfig<'static> {
    let defaults = ServingConfig::default();
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
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

fn kv_bytes(layers: &[(u32, u32, Option<u32>)], positions: u64) -> u64 {
    layers
        .iter()
        .map(|&(kv_heads, head_dim, window)| {
            let rows = window.map_or(positions, |rows| positions.min(u64::from(rows)));
            u64::from(kv_heads) * u64::from(head_dim) * 8 * rows
        })
        .sum()
}

struct Generation {
    token_ms: Vec<f64>,
    ids: Vec<u32>,
    text: String,
}

fn generate(model: &LoadedModel, prompt: &str, tokens: usize) -> Result<Generation, BenchError> {
    let mut token_ms = Vec::new();
    let started = Instant::now();
    let mut on_token = |event: TokenEvent<'_>| {
        if matches!(event.phase, Phase::Token) {
            token_ms.push(millis(started));
        }
        ControlFlow::Continue(())
    };
    let (ids, text, _stopped) =
        model.generate_streaming(prompt, tokens, config(), &mut on_token)?;
    Ok(Generation {
        token_ms,
        ids,
        text,
    })
}

fn report_generation(
    generation: &Generation,
    tokenize_ms: f64,
    prompt_tokens: usize,
) -> Result<(), BenchError> {
    let ttft = *generation.token_ms.first().ok_or(BenchError::NoTokens)?;
    let prefill = (ttft - tokenize_ms).max(f64::EPSILON);
    let steps: Vec<f64> = generation
        .token_ms
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    emit("ttft", &format!("ms={ttft:.3}"));
    emit(
        "prefill",
        &format!(
            "ms={prefill:.3} tok_s={:.3}",
            prompt_tokens as f64 / prefill * 1000.0
        ),
    );
    emit(
        "decode",
        &format!(
            "ms_mean={:.3} ms_median={:.3} tok_s={:.3} steps={}",
            mean(&steps),
            median(&steps),
            1000.0 / mean(&steps).max(f64::EPSILON),
            steps.len()
        ),
    );
    Ok(())
}

fn run_child(path: &str, prompt: &str, tokens: usize) -> Result<(), BenchError> {
    let process_started = Instant::now();
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
    emit("load", &format!("ms={:.3}", millis(load_started)));
    let warmup_started = Instant::now();
    generate(&model, prompt, WARMUP_TOKENS)?;
    emit(
        "device_warmup",
        &format!("ms={:.3}", millis(warmup_started)),
    );
    let tokenize_started = Instant::now();
    let ids = prompt_ids(&vocab, prompt)?;
    let tokenize_ms = millis(tokenize_started);
    emit(
        "tokenize",
        &format!("ms={tokenize_ms:.6} prompt_tokens={}", ids.len()),
    );
    let generation = generate(&model, prompt, tokens)?;
    report_generation(&generation, tokenize_ms, ids.len())?;
    let positions = (ids.len() + generation.ids.len()) as u64;
    emit(
        "kv",
        &format!("bytes={}", kv_bytes(&kv_layers(&parsed)?, positions)),
    );
    emit("total", &format!("ms={:.3}", millis(process_started)));
    println!(
        "output engine=proxima activation_quantization={} ids={:?} text={:?}",
        u8::from(quantization_enabled()),
        generation.ids,
        generation.text
    );
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

fn peak_rss_mb(stderr: &str) -> Option<f64> {
    let line = stderr
        .lines()
        .find(|line| line.contains("Maximum resident set size"))?;
    let kilobytes: f64 = line.rsplit(':').next()?.trim().parse().ok()?;
    Some(kilobytes / 1024.0)
}

fn measure(
    program: &str,
    args: &[String],
    envs: &[(&str, String)],
) -> Result<Measured, BenchError> {
    let started = Instant::now();
    let child = Command::new("/usr/bin/time")
        .arg("-v")
        .arg(program)
        .args(args)
        .envs(envs.iter().map(|(key, value)| (*key, value)))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(io_error(program))?;
    let sampler = start_sampler(child.id())?;
    let output = child.wait_with_output().map_err(io_error(program))?;
    let wall_ms = millis(started);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    Ok(Measured {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        peak_rss_mb: peak_rss_mb(&stderr),
        stderr,
        success: output.status.success(),
        wall_ms,
        peak_vram_mb: finish_sampler(sampler)?,
    })
}

fn parse_stages(stdout: &str) -> Stages {
    let mut stages = Stages::new();
    for line in stdout.lines().filter(|line| line.starts_with("pipeline ")) {
        let fields: HashMap<String, String> = line
            .split_whitespace()
            .filter_map(|token| token.split_once('='))
            .map(|(key, value)| (key.to_owned(), value.to_owned()))
            .collect();
        if let Some(stage) = fields.get("stage").cloned() {
            stages.insert(stage, fields);
        }
    }
    stages
}

fn field(stages: &Stages, stage: &str, key: &str) -> Option<f64> {
    stages.get(stage)?.get(key)?.parse().ok()
}

fn cell(value: Option<f64>, digits: usize) -> String {
    value.map_or_else(|| "n/a".to_owned(), |number| format!("{number:.digits$}"))
}

fn proxima_row(measured: &Measured) -> Vec<String> {
    let stages = parse_stages(&measured.stdout);
    let load = field(&stages, "load", "ms").zip(field(&stages, "device_warmup", "ms"));
    let kv_mib = field(&stages, "kv", "bytes").map(|bytes| bytes / MIB);
    vec![
        "proxima".to_owned(),
        cell(load.map(|(open, warmup)| open + warmup), 1),
        cell(field(&stages, "tokenize", "ms"), 4),
        cell(field(&stages, "tokenize", "prompt_tokens"), 0),
        cell(field(&stages, "prefill", "ms"), 2),
        cell(field(&stages, "prefill", "tok_s"), 1),
        cell(field(&stages, "ttft", "ms"), 2),
        cell(field(&stages, "decode", "ms_mean"), 3),
        cell(field(&stages, "decode", "tok_s"), 1),
        cell(field(&stages, "total", "ms"), 1),
        cell(Some(measured.wall_ms), 1),
        cell(kv_mib, 2),
        cell(measured.peak_rss_mb, 0),
        cell(Some(measured.peak_vram_mb as f64), 0),
    ]
}

fn perf_line(stderr: &str, label: &str) -> Option<(f64, Option<f64>)> {
    stderr.lines().find_map(|line| {
        let rest = line.split_once("llama_perf_context_print:")?.1.trim();
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

fn llama_kv_mib(stderr: &str) -> Option<f64> {
    let sizes: Vec<f64> = stderr
        .lines()
        .filter_map(|line| {
            line.split_once("KV buffer size =")?
                .1
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
        .collect();
    (!sizes.is_empty()).then(|| sizes.iter().sum())
}

fn llama_row(measured: &Measured) -> Vec<String> {
    let stderr = &measured.stderr;
    let load = perf_line(stderr, "load time").map(|(time, _)| time);
    let prompt = perf_line(stderr, "prompt eval time");
    let decode = perf_line(stderr, "eval time");
    let total = perf_line(stderr, "total time").map(|(time, _)| time);
    let decode_mean = decode.and_then(|(time, runs)| runs.map(|count| time / count.max(1.0)));
    vec![
        "llama".to_owned(),
        cell(load, 1),
        "n/a".to_owned(),
        cell(prompt.and_then(|(_, count)| count), 0),
        cell(prompt.map(|(time, _)| time), 2),
        cell(
            prompt.and_then(|(time, count)| count.map(|tokens| tokens / time * 1000.0)),
            1,
        ),
        cell(prompt.map(|(time, _)| time), 2),
        cell(decode_mean, 3),
        cell(decode_mean.map(|mean_ms| 1000.0 / mean_ms), 1),
        cell(load.zip(total).map(|(open, run)| open + run), 1),
        cell(Some(measured.wall_ms), 1),
        cell(llama_kv_mib(stderr), 2),
        cell(measured.peak_rss_mb, 0),
        cell(Some(measured.peak_vram_mb as f64), 0),
    ]
}

fn unsupported_row() -> Vec<String> {
    let mut row = vec!["llama".to_owned()];
    row.extend(HEADER[1..].iter().map(|_| "unsupported format".to_owned()));
    row
}

fn render_table(rows: &[Vec<String>]) -> String {
    let line = |cells: Vec<&str>| format!("| {} |", cells.join(" | "));
    let mut lines = vec![
        line(HEADER.to_vec()),
        line(HEADER.iter().map(|_| "---").collect()),
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

fn run_llama(path: &str, prompt: &str, tokens: usize) -> Result<Measured, BenchError> {
    let directory =
        env::var("PIPELINE_BENCH_LLAMA_DIR").unwrap_or_else(|_| DEFAULT_LLAMA_DIR.to_owned());
    let library_path =
        env::var("PIPELINE_BENCH_LLAMA_LD").unwrap_or_else(|_| DEFAULT_LLAMA_LD.to_owned());
    let args = [
        "-m".to_owned(),
        path.to_owned(),
        "-n".to_owned(),
        tokens.to_string(),
        "-ngl".to_owned(),
        "99".to_owned(),
        prompt.to_owned(),
    ];
    measure(
        &format!("{directory}/llama-simple"),
        &args,
        &[("LD_LIBRARY_PATH", library_path)],
    )
}

fn run_proxima(path: &str, prompt: &str, tokens: usize) -> Result<Measured, BenchError> {
    let executable = env::current_exe().map_err(io_error("current_exe"))?;
    let quantization = env::var(QUANTIZATION_ENV).unwrap_or_else(|_| "1".to_owned());
    let args = [path.to_owned(), prompt.to_owned(), tokens.to_string()];
    measure(
        &executable.to_string_lossy(),
        &args,
        &[
            (CHILD_ENV, "1".to_owned()),
            (QUANTIZATION_ENV, quantization),
        ],
    )
}

fn output_line(measured: &Measured, prefix: &str) -> String {
    measured
        .stdout
        .lines()
        .find(|line| line.starts_with(prefix))
        .map_or_else(|| format!("{prefix} missing"), str::to_owned)
}

fn orchestrate(path: &str, prompt: &str, tokens: usize) -> Result<(), BenchError> {
    let load_at_start = load_average();
    let proxima = run_proxima(path, prompt, tokens)?;
    let llama = run_llama(path, prompt, tokens)?;
    let llama_row = if llama.success {
        llama_row(&llama)
    } else {
        unsupported_row()
    };
    let rows = [proxima_row(&proxima), llama_row];
    let model = Path::new(path)
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    println!("{}", render_table(&rows));
    println!(
        "\nnotes: model={model} prompt={prompt:?} n={tokens} load_average_at_start={load_at_start} {QUANTIZATION_ENV}={}",
        env::var(QUANTIZATION_ENV).unwrap_or_else(|_| "1".to_owned())
    );
    println!(
        "proxima stages:\n{}",
        proxima
            .stdout
            .lines()
            .filter(|line| line.starts_with("pipeline "))
            .collect::<Vec<_>>()
            .join("\n")
    );
    if !proxima.success {
        println!("proxima failed: {}", last_lines(&proxima.stderr, 6));
    }
    println!("{}", output_line(&proxima, "output engine=proxima"));
    if llama.success {
        println!("llama output: {:?}", llama.stdout.trim());
    } else {
        println!("llama unsupported format: {}", last_lines(&llama.stderr, 6));
    }
    Ok(())
}

fn parse_args() -> Result<(String, String, usize), BenchError> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or(BenchError::Usage)?;
    let prompt = args.next().unwrap_or_else(|| DEFAULT_PROMPT.to_owned());
    let tokens = args.next().map_or(Ok(DEFAULT_TOKENS), |text| {
        text.parse().map_err(|_| BenchError::BadCount(text))
    })?;
    Ok((path, prompt, tokens))
}

fn main() -> ExitCode {
    let outcome = parse_args().and_then(|(path, prompt, tokens)| {
        if env::var_os(CHILD_ENV).is_some() {
            run_child(&path, &prompt, tokens)
        } else {
            orchestrate(&path, &prompt, tokens)
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
