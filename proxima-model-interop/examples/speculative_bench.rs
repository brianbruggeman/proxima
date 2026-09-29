//! slice 20 (`speculative-decode-llama-parity/TASKS.md`): the performance
//! harness the SPEC's own "performance harness" and "measurement protocol"
//! paragraphs name. THIS RUN IS A HARNESS-CORRECTNESS CHECK, NOT A
//! MEASUREMENT -- every number this binary prints while the host is not
//! verified quiet is a shape proof, not a result (`bench-metrics` skill:
//! never trust a run on a loaded box). `--force` is required whenever the
//! quiet-box precheck (`quiet_box_precheck`) finds contention, and every
//! line a `--force` run prints carries the literal word `unmeasured`.
//!
//! Per-pair protocol (SPEC's "performance harness" paragraph): OFF and ON
//! run as an interleaved pair, order swapped every pair, the first pair per
//! prompt discarded as warmup. Per arm this reads
//! [`proxima_model_interop::SpeculativeDecodeStats`] directly off
//! [`LoadedModel::generate_streaming_with_speculative_stats`]'s own return
//! path for verify_steps/drafted/accepted -- NOT the telemetry ring
//! (`speculative_decode_parity.rs`'s own doc on that ring silently dropping
//! events under metal's per-dispatch `debug!` volume).
//!
//! slice 24: `--incumbent llama-server [--llama-server-bin <path>]` (default
//! `llama-server` off `PATH`) spawns TWO llama-server child processes --
//! `--spec-type none` and `--spec-type <drafter>`, parameters read off the
//! SAME [`ServingConfig`]/[`SpeculativeConfig`] proxima's own ON arm uses
//! (`base_llama_server_args`/`ngram_type_args`) -- both alive for the whole
//! run, interleaved the same OFF/ON/swap-order way proxima's own pair is.
//! Each request sends the token ids proxima's own tokenizer produced for
//! that prompt (`/completion`'s `"prompt": [ids...]`), never re-tokenized
//! text, so a tokenizer disagreement cannot silently bias the comparison;
//! [`check_token_parity`] cross-checks llama's own `/tokenize` against those
//! ids once, on the corpus's first prompt, and reports (never hides) a
//! mismatch. llama's own `predicted_per_token_ms`/`draft_n`/
//! `draft_n_accepted` come straight off its `/completion` response
//! `"timings"` object -- never wall-clock around the HTTP call.
#![allow(clippy::expect_used, clippy::unwrap_used)]
#![allow(clippy::too_many_lines)]

use std::env;
use std::fs::File;
use std::io::{BufRead, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use core::ops::ControlFlow;
use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{
    GPU_LAYERS_ALL, LoadedModel, NgramMapParams, NgramModParams, Phase, PrefixState, ServingConfig,
    SpeculativeConfig, SpeculativeDecodeStats, SpeculativeType, SpeculativeTypeSet, TokenEvent,
};
use proxima_tokenizer::vocab::Vocab;

const DEFAULT_MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
const DEFAULT_MAX_TOKENS: usize = 48;
const DEFAULT_PAIRS: usize = 5;
const GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT: f64 = 5.0;
const CPU_IDLE_MINIMUM_PERCENT: f64 = 70.0;
const GPU_SAMPLE_INTERVAL: Duration = Duration::from_millis(100);
/// The owner's recorded protocol for the precheck's idle-GPU baseline: ~5s
/// at [`GPU_SAMPLE_INTERVAL`] (~10 Hz), decided on the median -- a single
/// `ollama ps` wake-the-Electron-app spike inside a 0.5s/5-sample window
/// produced a false "GPU busy" refusal three times running even though 50
/// direct `ioreg` samples taken seconds later all read 0.
const GPU_IDLE_BASELINE_DURATION: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------

/// `--verify-width-sweep` is a distinct mode from the interleaved-pair bench
/// -- both read a `--gpu-layers`, but the sweep needs no corpus/drafter/
/// incumbent flags at all (SPEC AC23's own command line omits them).
enum BenchMode {
    Pairs,
    VerifyWidthSweep { widths: Vec<usize> },
}

struct BenchArgs {
    mode: BenchMode,
    drafter: Option<SpeculativeType>,
    corpus_path: Option<PathBuf>,
    pairs: usize,
    gpu_layers: i32,
    max_tokens: usize,
    incumbent: Option<String>,
    llama_server_bin: PathBuf,
    force: bool,
    model_path: String,
}

const DEFAULT_LLAMA_SERVER_BIN: &str = "llama-server";

fn parse_gpu_layers(value: &str) -> i32 {
    if value.eq_ignore_ascii_case("all") {
        GPU_LAYERS_ALL
    } else {
        value
            .parse()
            .unwrap_or_else(|err| panic!("--gpu-layers {value}: not `all` or an integer: {err}"))
    }
}

fn parse_drafter(value: &str) -> SpeculativeType {
    SpeculativeType::from_llama_name(value).unwrap_or_else(|| {
        panic!(
            "--drafter {value}: not one of llama's own --spec-type names \
             (ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache, \
             draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark)"
        )
    })
}

fn parse_width_list(value: &str) -> Vec<usize> {
    value
        .split(',')
        .map(|piece| {
            piece
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|err| panic!("--verify-width-sweep {value}: {err}"))
        })
        .collect()
}

fn parse_args() -> BenchArgs {
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut drafter = None;
    let mut corpus_path = None;
    let mut pairs = DEFAULT_PAIRS;
    let mut gpu_layers = 0;
    let mut max_tokens = DEFAULT_MAX_TOKENS;
    let mut widths = None;
    let mut incumbent = None;
    let mut llama_server_bin = PathBuf::from(DEFAULT_LLAMA_SERVER_BIN);
    let mut force = false;
    let mut positionals = Vec::new();

    let mut index = 0;
    while index < raw.len() {
        match raw[index].as_str() {
            "--drafter" => {
                index += 1;
                drafter = Some(parse_drafter(&raw[index]));
            }
            "--corpus" => {
                index += 1;
                corpus_path = Some(PathBuf::from(&raw[index]));
            }
            "--pairs" => {
                index += 1;
                pairs = raw[index]
                    .parse()
                    .unwrap_or_else(|err| panic!("--pairs {}: {err}", raw[index]));
            }
            "--gpu-layers" => {
                index += 1;
                gpu_layers = parse_gpu_layers(&raw[index]);
            }
            "--max-tokens" => {
                index += 1;
                max_tokens = raw[index]
                    .parse()
                    .unwrap_or_else(|err| panic!("--max-tokens {}: {err}", raw[index]));
            }
            "--verify-width-sweep" => {
                index += 1;
                widths = Some(parse_width_list(&raw[index]));
            }
            "--incumbent" => {
                index += 1;
                incumbent = Some(raw[index].clone());
            }
            "--llama-server-bin" => {
                index += 1;
                llama_server_bin = PathBuf::from(&raw[index]);
            }
            "--force" => {
                force = true;
            }
            other => positionals.push(other.to_string()),
        }
        index += 1;
    }

    let model_path = positionals
        .into_iter()
        .next()
        .unwrap_or_else(|| DEFAULT_MODEL_PATH.to_string());

    let mode = match widths {
        Some(widths) => BenchMode::VerifyWidthSweep { widths },
        None => BenchMode::Pairs,
    };

    BenchArgs {
        mode,
        drafter,
        corpus_path,
        pairs,
        gpu_layers,
        max_tokens,
        incumbent,
        llama_server_bin,
        force,
        model_path,
    }
}

// ---------------------------------------------------------------------
// quiet-box precheck (SPEC "measurement protocol" paragraph)
// ---------------------------------------------------------------------

struct PrecheckReport {
    busy_reasons: Vec<String>,
    gpu_idle_baseline: Option<GpuIdleBaseline>,
    cpu_idle_baseline: Option<CpuIdleBaseline>,
    load_average_line: Option<String>,
}

impl PrecheckReport {
    fn is_quiet(&self) -> bool {
        self.busy_reasons.is_empty()
    }
}

/// `ollama ps` model names currently loaded. Absence of the `ollama` binary,
/// or the command failing, is treated as "no models loaded", not an error --
/// a dev box without Ollama installed at all is trivially unloaded.
fn ollama_loaded_models() -> Vec<String> {
    let Ok(output) = Command::new("ollama").arg("ps").output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

/// A loaded model competes for the same GPU this harness measures, but on
/// this machine the Ollama app is relaunched by other tooling whenever the
/// `ollama` CLI is invoked, so the process cannot be kept dead -- its mere
/// presence is not contamination. `ollama stop` each loaded model and
/// re-check; fail the precheck only if a model survives the stop.
fn ollama_busy() -> Option<String> {
    let loaded = ollama_loaded_models();
    if loaded.is_empty() {
        return None;
    }
    for name in &loaded {
        let _ = Command::new("ollama").args(["stop", name]).output();
    }
    let still_loaded = ollama_loaded_models();
    if still_loaded.is_empty() {
        None
    } else {
        Some(format!(
            "ollama ps reports {} loaded model(s) surviving `ollama stop`: {}",
            still_loaded.len(),
            still_loaded.join(", ")
        ))
    }
}

/// Host CPU split from one `top` sample. macOS load average counts blocked
/// threads, so contention is decided on sampled idle time instead.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CpuIdleBaseline {
    user: f64,
    sys: f64,
    idle: f64,
}

fn format_cpu_idle_baseline(baseline: &CpuIdleBaseline) -> String {
    format!(
        "cpu_idle_baseline user={:.2}% sys={:.2}% idle={:.2}%",
        baseline.user, baseline.sys, baseline.idle
    )
}

fn parse_top_cpu_field(line: &str, label: &str) -> Option<f64> {
    let before_label = line.split(label).next()?;
    before_label
        .rsplit([',', ':'])
        .next()?
        .trim()
        .trim_end_matches('%')
        .parse()
        .ok()
}

fn parse_top_cpu_line(line: &str) -> Option<CpuIdleBaseline> {
    let usage = line.trim().strip_prefix("CPU usage:")?;
    Some(CpuIdleBaseline {
        user: parse_top_cpu_field(usage, " user")?,
        sys: parse_top_cpu_field(usage, " sys")?,
        idle: parse_top_cpu_field(usage, " idle")?,
    })
}

/// `top -l 2 -s 1` prints a since-boot sample first and a 1s-delta sample
/// second; only the last `CPU usage:` line is a current reading.
fn sample_cpu_idle() -> Option<CpuIdleBaseline> {
    let output = Command::new("top")
        .args(["-l", "2", "-n", "0", "-s", "1"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .rfind(|line| line.trim_start().starts_with("CPU usage:"))
        .and_then(parse_top_cpu_line)
}

fn cpu_idle_decision(baseline: &CpuIdleBaseline) -> Option<String> {
    (baseline.idle < CPU_IDLE_MINIMUM_PERCENT).then(|| {
        format!(
            "host CPU idle {:.2}% is below {CPU_IDLE_MINIMUM_PERCENT}% ({})",
            baseline.idle,
            format_cpu_idle_baseline(baseline)
        )
    })
}

fn cpu_idle_busy() -> (Option<String>, Option<CpuIdleBaseline>) {
    let Some(baseline) = sample_cpu_idle() else {
        return (None, None);
    };
    (cpu_idle_decision(&baseline), Some(baseline))
}

fn load_average_line() -> Option<String> {
    let mut averages = [0.0_f64; 3];
    // SAFETY: `getloadavg` writes at most `averages.len()` doubles into a
    // caller-owned buffer it never retains a pointer to past this call.
    let filled = unsafe { libc::getloadavg(averages.as_mut_ptr(), averages.len() as i32) };
    (filled > 0).then(|| {
        format!(
            "load_average_informational 1m={:.2} 5m={:.2} 15m={:.2}",
            averages[0], averages[1], averages[2]
        )
    })
}

/// Scans `ps -Ao comm` for other cargo/rustc processes by name -- this
/// process's own `argv[0]` is not itself named any of these, so no
/// self-exclusion is needed. `ollama`/`Ollama` are handled separately by
/// [`ollama_busy`], which stops a loaded model rather than failing on the
/// app process merely being present.
fn other_processes_busy() -> Option<String> {
    let output = Command::new("ps").args(["-Ao", "comm"]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let needles = ["cargo", "rustc"];
    let mut hits: Vec<String> = Vec::new();
    for line in stdout.lines() {
        for needle in needles {
            if line.contains(needle) && !hits.iter().any(|hit| hit == line) {
                hits.push(line.trim().to_string());
            }
        }
    }
    if hits.is_empty() {
        None
    } else {
        Some(format!("other processes present: {}", hits.join("; ")))
    }
}

/// Median/mean/p90/max/n over a GPU idle-window sample vector -- the median
/// is what a contamination decision reads (a single spike from another
/// process waking the GPU cannot drag a median the way it drags a mean over
/// a handful of samples), the rest travel along so every refusal message
/// and every `gpu_idle_baseline` line can show its whole shape.
#[derive(Debug, Clone, Copy, PartialEq)]
struct GpuIdleBaseline {
    median: f64,
    mean: f64,
    p90: f64,
    max: f64,
    sample_count: usize,
}

fn summarize_gpu_idle_samples(samples: &[f64]) -> Option<GpuIdleBaseline> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("gpu idle samples are finite"));
    Some(GpuIdleBaseline {
        median: percentile(&sorted, 50.0),
        mean: mean(&sorted),
        p90: percentile(&sorted, 90.0),
        max: sorted[sorted.len() - 1],
        sample_count: sorted.len(),
    })
}

fn format_gpu_idle_baseline(label: &str, baseline: &GpuIdleBaseline) -> String {
    format!(
        "{label} median={:.1}% mean={:.1}% p90={:.1}% max={:.1}% n={}",
        baseline.median, baseline.mean, baseline.p90, baseline.max, baseline.sample_count
    )
}

/// Samples GPU utilization for `duration` at `interval`, deadline-driven
/// rather than a fixed sample count, so the same helper serves both the
/// precheck's 5s baseline and the per-pair contamination window's shorter
/// sample.
fn sample_gpu_idle_window(duration: Duration, interval: Duration) -> Vec<f64> {
    let deadline = Instant::now() + duration;
    let mut samples = Vec::new();
    while Instant::now() < deadline {
        if let Some(sample) = gpu_utilization_sample() {
            samples.push(sample);
        }
        std::thread::sleep(interval);
    }
    samples
}

/// Runs FIRST in [`quiet_box_precheck`], before `ollama_busy`'s `ollama ps`
/// shell-out (which wakes the Ollama Electron app and briefly moves the
/// GPU) -- sampling the idle baseline after that wake produced three false
/// "GPU busy" refusals even though the box was otherwise idle.
fn gpu_idle_busy() -> (Option<String>, Option<GpuIdleBaseline>) {
    if !cfg!(target_os = "macos") {
        return (None, None);
    }
    let samples = sample_gpu_idle_window(GPU_IDLE_BASELINE_DURATION, GPU_SAMPLE_INTERVAL);
    let Some(baseline) = summarize_gpu_idle_samples(&samples) else {
        return (None, None);
    };
    let reason = (baseline.median > GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT).then(|| {
        format!(
            "GPU idle-window utilization median {:.1}% exceeds {GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT}% ({})",
            baseline.median,
            format_gpu_idle_baseline("gpu_idle_baseline", &baseline)
        )
    });
    (reason, Some(baseline))
}

fn quiet_box_precheck() -> PrecheckReport {
    let (gpu_idle_reason, gpu_idle_baseline) = gpu_idle_busy();
    let mut busy_reasons = Vec::new();
    busy_reasons.extend(gpu_idle_reason);
    busy_reasons.extend(ollama_busy());
    let (cpu_idle_reason, cpu_idle_baseline) = cpu_idle_busy();
    busy_reasons.extend(cpu_idle_reason);
    busy_reasons.extend(other_processes_busy());
    PrecheckReport {
        busy_reasons,
        gpu_idle_baseline,
        cpu_idle_baseline,
        load_average_line: load_average_line(),
    }
}

// ---------------------------------------------------------------------
// GPU utilization sampler (macOS `ioreg` -c IOAccelerator, ~10 Hz)
// ---------------------------------------------------------------------

/// One `ioreg -r -d 1 -c "IOAccelerator"` call, parsed for the first
/// `"Device Utilization %"=<value>` field. Returns `None` off macOS or when
/// `ioreg` cannot be run (no accelerator entry, sandboxed environment).
fn gpu_utilization_sample() -> Option<f64> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let output = Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let needle = "\"Device Utilization %\"=";
    let start = stdout.find(needle)? + needle.len();
    let rest = &stdout[start..];
    let end = rest.find(|character: char| !character.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse::<f64>().ok()
}

struct GpuSampler {
    running: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<f64>>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl GpuSampler {
    fn start() -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let samples: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let thread_running = Arc::clone(&running);
        let thread_samples = Arc::clone(&samples);
        let handle = std::thread::Builder::new()
            .name("gpu-utilization-sampler".to_string())
            .spawn(move || {
                while thread_running.load(Ordering::Relaxed) {
                    if let Some(sample) = gpu_utilization_sample() {
                        thread_samples
                            .lock()
                            .expect("gpu sampler mutex poisoned")
                            .push(sample);
                    }
                    std::thread::sleep(GPU_SAMPLE_INTERVAL);
                }
            })
            .expect("spawn gpu-utilization-sampler thread");
        Self {
            running,
            samples,
            handle: Some(handle),
        }
    }

    fn stop(mut self) -> Vec<f64> {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().expect("join gpu-utilization-sampler thread");
        }
        Arc::try_unwrap(self.samples)
            .expect("sampler thread joined, sole owner remains")
            .into_inner()
            .expect("gpu sampler mutex poisoned")
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct GpuUtilizationSummary {
    mean: f64,
    median: f64,
    p10: f64,
    p90: f64,
    sample_count: usize,
}

fn summarize_gpu_samples(mut samples: Vec<f64>) -> Option<GpuUtilizationSummary> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|left, right| left.partial_cmp(right).expect("gpu samples are finite"));
    let sample_count = samples.len();
    let mean = samples.iter().sum::<f64>() / sample_count as f64;
    Some(GpuUtilizationSummary {
        mean,
        median: percentile(&samples, 50.0),
        p10: percentile(&samples, 10.0),
        p90: percentile(&samples, 90.0),
        sample_count,
    })
}

// ---------------------------------------------------------------------
// resource sampling (RSS, CPU%)
// ---------------------------------------------------------------------

/// Darwin reports `ru_maxrss` in bytes; Linux reports KiB
// (`examples/gguf_generate.rs::print_peak_rss` -- same convention).
fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` initializes the caller-owned structure on success.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0;
    }
    // SAFETY: `result == 0` means `getrusage` filled `usage` above.
    let usage = unsafe { usage.assume_init() };
    let scale = if cfg!(target_os = "macos") { 1 } else { 1024 };
    usage.ru_maxrss as u64 * scale
}

/// User + system CPU seconds consumed by this process so far
/// (`getrusage`'s own `ru_utime`/`ru_stime`), for a before/after delta a
/// caller divides by wall-clock seconds to get CPU%.
fn cpu_seconds() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: same contract as `peak_rss_bytes`.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0.0;
    }
    // SAFETY: `result == 0` means `getrusage` filled `usage` above.
    let usage = unsafe { usage.assume_init() };
    let user = usage.ru_utime.tv_sec as f64 + usage.ru_utime.tv_usec as f64 / 1_000_000.0;
    let system = usage.ru_stime.tv_sec as f64 + usage.ru_stime.tv_usec as f64 / 1_000_000.0;
    user + system
}

// ---------------------------------------------------------------------
// stats helpers
// ---------------------------------------------------------------------

/// Nearest-rank percentile over an already-sorted ascending slice.
fn percentile(sorted: &[f64], percent: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((percent / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Coefficient of variation: standard deviation / mean, as a fraction
/// (0.05 == 5%) -- `disciplined-component`'s own "never a point estimate
/// above 5% CoV" rule reads this directly.
fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let average = mean(values);
    if average == 0.0 {
        return 0.0;
    }
    let variance = values.iter().map(|value| (value - average).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt() / average
}

// ---------------------------------------------------------------------
// llama-server incumbent arm (invariants 1-3 in the task brief; SPEC's
// own "incumbent arm" architecture paragraph)
// ---------------------------------------------------------------------

/// Mirrors `proxima_model_interop::generate::residency_caches::wants_bos`,
/// which is crate-private and unreachable from an example binary --
/// `examples/attn_prompt_tokens.rs` already carries the identical copy for
/// the identical reason.
fn wants_bos(vocab: &Vocab) -> bool {
    vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some())
}

/// One TCP round trip: a full HTTP/1.1 request with a JSON body, read to
/// socket close (`Connection: close` on the request, so a non-streaming
/// JSON response never needs chunked-transfer-encoding handling). llama's
/// own client-side documentation for `/completion` and `/tokenize` is the
/// llama-server source read for this arm (`server-context.cpp`,
/// `server-common.cpp`) -- there is no client crate in this workspace's
/// `Cargo.lock` that would save more than this hand-rolled request/response
/// pair costs (invariant 5: `reqwest`/`ureq`/`hyper` are either absent or,
/// for `hyper`, only ever wired as part of `proxima-http`'s full async
/// server/client stack, which would drag a tokio runtime into a synchronous
/// CLI example for one localhost POST).
/// `Err` on ANY failure (connection refused, malformed response, non-200
/// status) -- the caller decides whether that is a readiness-poll retry
/// ([`LlamaServerHandle::wait_until_healthy`]) or a hard failure
/// ([`http_get_json`]/[`http_post_json`]).
fn try_http_request_json(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let payload = body.map(|value| serde_json::to_vec(value).expect("serialize json body"));
    let mut stream =
        TcpStream::connect(("127.0.0.1", port)).map_err(|err| format!("connect: {err}"))?;
    let content_length = payload.as_ref().map_or(0, Vec::len);
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n\
         Content-Length: {content_length}\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(head.as_bytes())
        .map_err(|err| format!("write request head: {err}"))?;
    if let Some(payload) = &payload {
        stream
            .write_all(payload)
            .map_err(|err| format!("write request body: {err}"))?;
    }
    stream.flush().map_err(|err| format!("flush request: {err}"))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|err| format!("read response: {err}"))?;
    let separator = b"\r\n\r\n";
    let split_at = response
        .windows(separator.len())
        .position(|window| window == separator)
        .ok_or_else(|| "response missing header/body separator".to_string())?;
    let status_line = response[..split_at]
        .split(|&byte| byte == b'\r' || byte == b'\n')
        .next()
        .ok_or_else(|| "response missing status line".to_string())?;
    let status_text = String::from_utf8_lossy(status_line).into_owned();
    if !status_text.contains("200") {
        return Err(format!("{method} {path} returned {status_text}"));
    }
    let response_body = &response[split_at + separator.len()..];
    serde_json::from_slice(response_body)
        .map_err(|err| format!("parse response json: {err} (body: {})", String::from_utf8_lossy(response_body)))
}

fn http_request_json(port: u16, method: &str, path: &str, body: Option<&serde_json::Value>) -> serde_json::Value {
    try_http_request_json(port, method, path, body)
        .unwrap_or_else(|err| panic!("llama-server {method} {path}: {err}"))
}

fn http_post_json(port: u16, path: &str, body: &serde_json::Value) -> serde_json::Value {
    http_request_json(port, "POST", path, Some(body))
}

fn llama_gpu_layers_value(gpu_layers: i32) -> String {
    if gpu_layers == GPU_LAYERS_ALL {
        "all".to_string()
    } else {
        gpu_layers.to_string()
    }
}

/// Only the parameters the task brief names as fair to match run off
/// `config`: sampling (`--temp`/`--top-k`/`--top-p`/`--min-p`/
/// `--repeat-last-n`/`--repeat-penalty`/`--frequency-penalty`/
/// `--presence-penalty`), `--seed`, and `-ngl` (`config.gpu_layers`) --
/// plus `-np` (`config.parallel_sequences`), which is `1` on both sides
/// already (llama.cpp's own `n_parallel` default, `common.h:457`), so
/// passing it explicitly never moves llama off its own turf. `batch_size`/
/// `ubatch_size` are deliberately absent for the same reason they always
/// were: proxima's own `0` sentinel has no llama-server equivalent value,
/// left at llama's own CLI default (`-b 2048 -ub 512`) rather than guessed.
///
/// Everything else here previously copied `config`'s OWN serving values
/// onto llama-server -- `-ctk f32 -ctv f32 -fa off --no-kv-offload` --
/// which is not llama-server's home turf: `--no-kv-offload` in particular
/// forces attention onto the CPU on Apple silicon. Those flags are gone;
/// llama-server now runs its own real per-hardware defaults, confirmed
/// against `common/common.h` and `common/arg.cpp` in this host's
/// `llama.cpp` checkout:
/// - KV cache dtype: `GGML_TYPE_F16` for both K and V (`common.h:589-590`,
///   the `-ctk`/`-ctv` default `common_params` never overrides here).
/// - KV offload: enabled (`no_kv_offload = false`, `common.h:580`;
///   `-nkvo`/`--no-kv-offload` is the opt-in to disable it, `arg.cpp:2412-2419`).
/// - Flash Attention: `auto` (`flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO`,
///   `common.h:501`; `-fa`'s own default string reads `"auto"`,
///   `arg.cpp:1751-1764`).
///
/// `-c` is the one exception NOT left at llama's literal default
/// (`n_ctx = 0`, "whatever the model was trained with", `common.h:452`,
/// `arg.cpp:1636-1644`): `llama_context_length` is sized to this run's own
/// corpus (see `run_pairs_mode`'s own comment) so the incumbent is not
/// forced to allocate KV cache for a multi-hundred-thousand-token context
/// it never uses.
fn base_llama_server_args(config: &ServingConfig, llama_context_length: u32) -> Vec<String> {
    vec![
        "-c".to_string(),
        llama_context_length.to_string(),
        "-ngl".to_string(),
        llama_gpu_layers_value(config.gpu_layers),
        "-np".to_string(),
        config.parallel_sequences.to_string(),
        "--temp".to_string(),
        config.temperature.to_string(),
        "--top-k".to_string(),
        config.top_k.to_string(),
        "--top-p".to_string(),
        config.top_p.to_string(),
        "--min-p".to_string(),
        config.min_p.to_string(),
        "--repeat-last-n".to_string(),
        config.repeat_last_n.to_string(),
        "--repeat-penalty".to_string(),
        config.repeat_penalty.to_string(),
        "--frequency-penalty".to_string(),
        config.frequency_penalty.to_string(),
        "--presence-penalty".to_string(),
        config.presence_penalty.to_string(),
        "--seed".to_string(),
        config.seed.to_string(),
        "--no-webui".to_string(),
    ]
}

/// The size/hit-count flags llama's own five n-gram `--spec-type` values
/// read, sourced from the SAME [`SpeculativeConfig`] struct proxima's
/// decode loop reads for the ON arm -- so a parameter drift on either side
/// shows up as a real speedup difference, not a silent mismatch.
/// `ngram-cache` has no size/hit-count CLI flags upstream (it only takes
/// `--lookup-cache-static`/`--lookup-cache-dynamic` file paths, R7's own
/// concern, not wired here); the empty-args branch runs it at llama's own
/// in-memory-dynamic-cache default, same as omitting both cache flags on
/// llama's own CLI.
fn ngram_type_args(spec_type: SpeculativeType, speculative: &SpeculativeConfig) -> Vec<String> {
    fn map_params_args(flag_prefix: &str, params: NgramMapParams) -> Vec<String> {
        vec![
            format!("--spec-{flag_prefix}-size-n"),
            params.size_n.to_string(),
            format!("--spec-{flag_prefix}-size-m"),
            params.size_m.to_string(),
            format!("--spec-{flag_prefix}-min-hits"),
            params.min_hits.to_string(),
        ]
    }
    fn mod_params_args(params: NgramModParams) -> Vec<String> {
        vec![
            "--spec-ngram-mod-n-match".to_string(),
            params.n_match.to_string(),
            "--spec-ngram-mod-n-max".to_string(),
            params.n_max.to_string(),
            "--spec-ngram-mod-n-min".to_string(),
            params.n_min.to_string(),
        ]
    }
    match spec_type {
        SpeculativeType::NgramSimple => map_params_args("ngram-simple", speculative.ngram_simple),
        SpeculativeType::NgramMapK => map_params_args("ngram-map-k", speculative.ngram_map_k),
        SpeculativeType::NgramMapK4v => map_params_args("ngram-map-k4v", speculative.ngram_map_k4v),
        SpeculativeType::NgramMod => mod_params_args(speculative.ngram_mod),
        SpeculativeType::NgramCache => Vec::new(),
        other => panic!(
            "llama-server incumbent arm only maps the five n-gram --spec-type \
             values; {} is not one of them",
            other.llama_name()
        ),
    }
}

/// PIDs of every child [`ChildGuard`] this run has spawned, so
/// [`install_orphan_reaping_panic_hook`]'s hook can kill them even when
/// [`ChildGuard::drop`] itself never runs -- the release profile's own
/// `panic = "abort"` (workspace `Cargo.toml`) skips unwinding entirely, so
/// no destructor on the panicking thread's stack executes. Plain `u32`
/// PIDs, not `Child` handles: the hook needs to reach these from a context
/// that does not (and must not) own the `Child` itself -- the normal drop
/// path already owns and reaps it directly.
static REAPABLE_PIDS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// Kills and waits the wrapped child on drop. `std::process::Child`'s own
/// `Drop` deliberately does NOT kill the process (the stdlib's documented
/// behavior) -- a bench that only ever calls `.spawn()` leaks a running
/// llama-server on any early return, and leaked it on every panic in this
/// file's own real-run history (two orphaned llama-server processes after a
/// mid-run panic, neither killed nor waited).
struct ChildGuard {
    child: Child,
    pid: u32,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        let pid = child.id();
        REAPABLE_PIDS.lock().unwrap_or_else(PoisonError::into_inner).push(pid);
        Self { child, pid }
    }

    fn pid(&self) -> u32 {
        self.pid
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        REAPABLE_PIDS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .retain(|&candidate| candidate != self.pid);
    }
}

/// The `panic`-hook half of orphan reaping -- [`ChildGuard::drop`] alone
/// does not run under the release profile's `panic = "abort"` (no
/// unwinding, so no destructor on the panicking thread's stack executes),
/// and never runs at all past a bare `std::process::exit`. A panic hook
/// runs BEFORE the abort/unwind decision, on every profile, so it is the
/// one mechanism that reaches both cases. Chains the previous hook rather
/// than replacing it, so the default panic message (thread name, location,
/// the `RUST_BACKTRACE` hint) still prints exactly as before. Installed
/// once, at the top of `main`.
fn install_orphan_reaping_panic_hook() {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        reap_orphaned_llama_servers();
        previous_hook(panic_info);
    }));
}

fn reap_orphaned_llama_servers() {
    let pids = REAPABLE_PIDS.lock().unwrap_or_else(PoisonError::into_inner).clone();
    for pid in pids {
        eprintln!("speculative_bench: panic hook reaping orphaned llama-server pid={pid}");
        let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
    }
}

/// One llama-server child process bound to one port, killed on
/// [`Self::stop`] (and on drop regardless -- see [`ChildGuard`]). `label` is
/// `"off"`/`"on"`, printed alongside the pid so a caller reading stderr can
/// match this run's own process-management instructions to a concrete pid.
struct LlamaServerHandle {
    child: ChildGuard,
    port: u16,
    label: String,
}

impl LlamaServerHandle {
    fn spawn(bin: &Path, model_path: &str, port: u16, label: &str, extra_args: &[String]) -> Self {
        let mut command = Command::new(bin);
        if let Some(lib_dir) = bin.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            // llama-server's own dylibs (libggml*, libllama*) live alongside
            // the binary, not on the default dyld search path.
            command.env("DYLD_LIBRARY_PATH", lib_dir);
        }
        let fixed_args = [
            "--model".to_string(),
            model_path.to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            port.to_string(),
        ];
        // The exact command line this arm's incumbent ran under -- so a
        // parity dispute (was `-c`/`-ctk`/`-fa` really what the log claims?)
        // is settled by grepping this run's own log, not by trusting a
        // doc-comment (principle 16: re-provable from the artifact alone).
        eprintln!(
            "speculative_bench: llama-server ({label}) command: {} {} {}",
            bin.display(),
            fixed_args.join(" "),
            extra_args.join(" "),
        );
        command
            .args(fixed_args)
            .args(extra_args)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let child = command
            .spawn()
            .unwrap_or_else(|err| panic!("spawn llama-server ({label}) from {}: {err}", bin.display()));
        let child = ChildGuard::new(child);
        let pid = child.pid();
        eprintln!("speculative_bench: started llama-server ({label}) pid={pid} port={port}");
        let handle = Self {
            child,
            port,
            label: label.to_string(),
        };
        handle.wait_until_healthy();
        handle
    }

    fn wait_until_healthy(&self) {
        let deadline = Instant::now() + Duration::from_secs(180);
        loop {
            if let Ok(body) = try_http_request_json(self.port, "GET", "/health", None)
                && body.get("status").and_then(serde_json::Value::as_str) == Some("ok")
            {
                return;
            }
            assert!(
                Instant::now() <= deadline,
                "llama-server ({}) on port {} never reported healthy within 180s",
                self.label,
                self.port
            );
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    fn stop(self) {
        let pid = self.child.pid();
        let label = self.label.clone();
        drop(self); // drops `child` (a `ChildGuard`) -- kills and waits it.
        eprintln!("speculative_bench: stopped llama-server ({label}) pid={pid}");
    }
}

/// `timings.prompt_n` above this many tokens on a repeat request means the
/// slot's KV cache missed the common prefix and llama-server re-prefilled
/// the whole prompt (`server-context.cpp:3198`'s `get_common_prefix`
/// returned far short of the full length) -- the exact cost class
/// [`warm_llama_prompt_cache`] exists to keep off the timed pairs. A cache
/// hit still costs a handful of tokens (the freshly generated ones from the
/// prior pair plus any per-call rounding), never the whole prompt.
const LLAMA_REPREFILL_PROMPT_N_BOUND: u64 = 8;

#[derive(Debug, Clone, Copy, Default)]
struct LlamaArmResult {
    ms_per_token: f64,
    ttft_ms: f64,
    predicted_n: u64,
    draft_n: u64,
    draft_n_accepted: u64,
    prompt_n: u64,
    llama_reprefilled: bool,
}

/// Parses one llama-server `/completion` response body into
/// [`LlamaArmResult`], reading llama's OWN timing fields
/// (`predicted_per_token_ms`, `prompt_ms`, `prompt_n`, `predicted_n`,
/// `draft_n`, `draft_n_accepted`) straight off `timings`
/// (`server-common.cpp:84-105`'s `server_slot_stats::to_json`) -- never
/// wall-clock around the HTTP call (invariant 3).
fn parse_llama_completion_response(response: &serde_json::Value) -> LlamaArmResult {
    let timings = response
        .get("timings")
        .unwrap_or_else(|| panic!("llama-server /completion response missing \"timings\": {response}"));
    let get_f64 = |key: &str| timings.get(key).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
    let get_u64 = |key: &str| timings.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0);
    let prompt_n = get_u64("prompt_n");
    LlamaArmResult {
        ms_per_token: get_f64("predicted_per_token_ms"),
        ttft_ms: get_f64("prompt_ms"),
        predicted_n: get_u64("predicted_n"),
        draft_n: get_u64("draft_n"),
        draft_n_accepted: get_u64("draft_n_accepted"),
        prompt_n,
        llama_reprefilled: prompt_n > LLAMA_REPREFILL_PROMPT_N_BOUND,
    }
}

/// One non-streaming `/completion` call against token ids (invariant 2:
/// "prefer sending token ids ... produced by the proxima tokenizer +
/// template"). `cache_prompt: true` (`server-task.h:53`, llama-server's own
/// default) matches proxima's own arm: [`prefill_prompt_once`] prefills a
/// prompt exactly once and every pair below resumes from that cached state,
/// so llama-server must resume from its own per-slot KV cache too rather
/// than re-prefilling on every pair -- [`warm_llama_prompt_cache`] primes
/// that cache once per prompt, discarded, before the timed pairs begin.
fn run_llama_completion(port: u16, token_ids: &[u32], max_tokens: usize) -> LlamaArmResult {
    let body = serde_json::json!({
        "prompt": token_ids,
        "n_predict": max_tokens,
        "cache_prompt": true,
        "stream": false,
    });
    let response = http_post_json(port, "/completion", &body);
    parse_llama_completion_response(&response)
}

/// Discarded warm call issued once per prompt, before this prompt's timed
/// pairs begin: primes `handle`'s per-slot KV cache with `token_ids' full
/// prefix so the first TIMED pair does not eat the one-time prefill cost --
/// the same cost proxima's own [`prefill_prompt_once`] pays exactly once
/// and excludes from the pair loop.
fn warm_llama_prompt_cache(handle: &LlamaServerHandle, token_ids: &[u32]) {
    let _ = run_llama_completion(handle.port, token_ids, 1);
}

struct LlamaPairResult {
    off: LlamaArmResult,
    on: LlamaArmResult,
}

fn run_llama_pair(
    off_handle: &LlamaServerHandle,
    on_handle: &LlamaServerHandle,
    token_ids: &[u32],
    max_tokens: usize,
    swap_order: bool,
) -> LlamaPairResult {
    let (off, on) = if swap_order {
        let on = run_llama_completion(on_handle.port, token_ids, max_tokens);
        let off = run_llama_completion(off_handle.port, token_ids, max_tokens);
        (off, on)
    } else {
        let off = run_llama_completion(off_handle.port, token_ids, max_tokens);
        let on = run_llama_completion(on_handle.port, token_ids, max_tokens);
        (off, on)
    };
    LlamaPairResult { off, on }
}

fn format_llama_arm(label: &str, arm: &LlamaArmResult) -> String {
    format!(
        "{label}_llama_ms_per_token={:.3} {label}_llama_ttft_ms={:.3} {label}_llama_predicted_n={} \
         {label}_llama_draft_n={} {label}_llama_draft_n_accepted={} {label}_llama_prompt_n={} \
         {label}_llama_reprefilled={}",
        arm.ms_per_token,
        arm.ttft_ms,
        arm.predicted_n,
        arm.draft_n,
        arm.draft_n_accepted,
        arm.prompt_n,
        arm.llama_reprefilled,
    )
}

/// Invariant 2's token-equality check, run once on the corpus's first
/// prompt: proxima's own tokenization (`wants_bos`/`add_eos_token`, the
/// exact convention [`run_one_arm`]'s decode call uses) vs llama-server's
/// `/tokenize` on the same raw text. Prints the result either way --
/// divergence is reported, not treated as fatal, per the task brief.
fn check_token_parity(off_handle: &LlamaServerHandle, prompt: &str, proxima_ids: &[u32], add_bos: bool) {
    let body = serde_json::json!({
        "content": prompt,
        "add_special": add_bos,
        "parse_special": true,
    });
    let response = http_post_json(off_handle.port, "/tokenize", &body);
    let llama_ids: Vec<u64> = response
        .get("tokens")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("llama-server /tokenize response missing \"tokens\" array: {response}"))
        .iter()
        .map(|value| value.as_u64().unwrap_or_else(|| panic!("non-integer token id in {value}")))
        .collect();
    let proxima_ids_u64: Vec<u64> = proxima_ids.iter().map(|&id| u64::from(id)).collect();
    let identical = llama_ids == proxima_ids_u64;
    println!(
        "token_parity proxima_len={} llama_len={} identical={identical} \
         proxima_first_six={:?} llama_first_six={:?}",
        proxima_ids_u64.len(),
        llama_ids.len(),
        proxima_ids_u64.iter().take(6).collect::<Vec<_>>(),
        llama_ids.iter().take(6).collect::<Vec<_>>(),
    );
}

/// Spawns the OFF (`--spec-type none`) and ON (`--spec-type <drafter>`,
/// parameters read off `on_config.speculative`) llama-server processes on
/// two fixed ports, both alive for the whole bench run (invariant 1: "one
/// server process per configuration ... loaded once per config"). A caller
/// interleaves requests against the SAME two handles for every pair.
///
/// `llama_context_length` is the ONE non-default value `base_llama_server_args`
/// passes for `-c` -- everything else it emits either matches proxima's arm
/// on purpose (sampling, seed, `-ngl`) or is llama-server's own real
/// per-hardware default (KV cache dtype, KV offload, Flash Attention; see
/// that function's own doc for the `common.h`/`arg.cpp` citations).
fn spawn_incumbent_servers(
    llama_server_bin: &Path,
    model_path: &str,
    drafter: SpeculativeType,
    off_config: &ServingConfig,
    on_config: &ServingConfig,
    llama_context_length: u32,
) -> (LlamaServerHandle, LlamaServerHandle) {
    const OFF_PORT: u16 = 18_080;
    const ON_PORT: u16 = 18_081;
    let mut off_args = base_llama_server_args(off_config, llama_context_length);
    off_args.push("--spec-type".to_string());
    off_args.push("none".to_string());
    let mut on_args = base_llama_server_args(on_config, llama_context_length);
    on_args.push("--spec-type".to_string());
    on_args.push(drafter.llama_name().to_string());
    on_args.extend(ngram_type_args(drafter, &on_config.speculative));
    let off_handle = LlamaServerHandle::spawn(llama_server_bin, model_path, OFF_PORT, "off", &off_args);
    let on_handle = LlamaServerHandle::spawn(llama_server_bin, model_path, ON_PORT, "on", &on_args);
    (off_handle, on_handle)
}

// ---------------------------------------------------------------------
// corpus loading
// ---------------------------------------------------------------------

/// Any jsonl of `{"prompt": ...}` -- SPEC's own invariant on this harness's
/// corpus contract. A malformed or prompt-less line is skipped, not fatal:
/// a corpus assembled by a separate slice (18) may carry other fields this
/// harness does not need.
fn load_corpus(path: &Path) -> Vec<String> {
    let file = File::open(path)
        .unwrap_or_else(|err| panic!("open corpus {}: {err}", path.display()));
    let reader = std::io::BufReader::new(file);
    let mut prompts = Vec::new();
    for line in reader.lines() {
        let line = line.expect("read corpus line");
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if let Some(prompt) = value.get("prompt").and_then(serde_json::Value::as_str) {
            prompts.push(prompt.to_string());
        }
    }
    prompts
}

// ---------------------------------------------------------------------
// one arm / one pair
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
struct ArmResult {
    ms_per_token: f64,
    ttft_ms: f64,
    p50_token_ms: f64,
    p99_token_ms: f64,
    verify_steps: u64,
    accepted_total: u64,
    drafted_total: u64,
    rss_bytes: u64,
    cpu_percent: f64,
    gpu: Option<GpuUtilizationSummary>,
}

/// Every arm's own cached context -- a [`PrefixState`] a single per-prompt
/// [`LoadedModel::prefill_prefix`] call already produced, plus the small
/// suffix text still needing a fresh forward pass. Every OFF/ON arm and
/// every pair for a prompt decodes from the SAME `DecodeSource` instead of
/// re-prefilling per arm (the performance-harness invariant: each prompt
/// is prefilled exactly once per bench run). Plain references, so this is
/// `Copy` -- cheap to pass by value into every arm call.
#[derive(Clone, Copy)]
struct DecodeSource<'source> {
    prefix: &'source PrefixState,
    suffix: &'source str,
}

/// `forced_draft_width` reaches
/// [`LoadedModel::generate_streaming_with_speculative_stats`]'s own
/// argument of the same name, never `serving_config` -- see that method's
/// doc for why the knob stays off the config surface. `None` for every
/// caller except [`run_verify_width_sweep_mode`].
fn run_one_arm(
    model: &LoadedModel,
    source: DecodeSource<'_>,
    max_tokens: usize,
    serving_config: ServingConfig,
    sample_gpu: bool,
    forced_draft_width: Option<u16>,
) -> ArmResult {
    let mut prefill_elapsed_ms: u64 = 0;
    let cpu_start = cpu_seconds();
    let wall_start = Instant::now();
    let sampler = if sample_gpu {
        Some(GpuSampler::start())
    } else {
        None
    };

    let mut stats = SpeculativeDecodeStats::default();
    // `TokenEvent::text_piece` borrows from the loop's own scratch buffer
    // (`'piece` tied to one `on_token` call) -- only `elapsed_ms`/`step`/
    // `phase` survive past the callback, so this harness copies exactly
    // those three fields instead of trying to retain the borrowed str.
    let mut elapsed_by_step: Vec<(usize, Phase, u64)> = Vec::new();
    let mut on_token = |event: TokenEvent<'_>| {
        elapsed_by_step.push((event.step, event.phase, event.elapsed_ms));
        ControlFlow::Continue(())
    };
    let (token_ids, _text, _stopped_by_eos) = model
        .generate_from_prefix_with_speculative_stats(
            source.prefix,
            source.suffix,
            max_tokens,
            serving_config,
            &mut on_token,
            &mut stats,
            forced_draft_width,
        )
        .expect("decode arm");
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let cpu_delta = cpu_seconds() - cpu_start;
    let cpu_percent = if wall_ms > 0.0 {
        (cpu_delta / (wall_ms / 1000.0)) * 100.0
    } else {
        0.0
    };
    for (_step, phase, elapsed_ms) in &elapsed_by_step {
        if matches!(phase, Phase::Prefill { .. }) {
            prefill_elapsed_ms = *elapsed_ms;
        }
    }

    let token_elapsed_ms: Vec<u64> = elapsed_by_step
        .iter()
        .filter(|(_, phase, _)| matches!(phase, Phase::Token))
        .map(|(_, _, elapsed_ms)| *elapsed_ms)
        .collect();
    let ttft_ms = token_elapsed_ms.first().copied().unwrap_or(0) as f64;
    let mut per_token_deltas: Vec<f64> = token_elapsed_ms
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as f64)
        .collect();
    per_token_deltas.sort_by(|left, right| left.partial_cmp(right).expect("finite latencies"));

    let tokens_generated = token_ids.len();
    let ms_per_token = if tokens_generated > 1 {
        (wall_ms - prefill_elapsed_ms as f64) / (tokens_generated - 1) as f64
    } else {
        0.0
    };

    let gpu = sampler.map(|sampler| sampler.stop()).and_then(summarize_gpu_samples);

    ArmResult {
        ms_per_token,
        ttft_ms,
        p50_token_ms: percentile(&per_token_deltas, 50.0),
        p99_token_ms: percentile(&per_token_deltas, 99.0),
        verify_steps: stats.verify_steps,
        accepted_total: stats.accepted_total,
        drafted_total: stats.drafted_total,
        rss_bytes: peak_rss_bytes(),
        cpu_percent,
        gpu,
    }
}

struct PairResult {
    off: ArmResult,
    on: ArmResult,
    contaminated: bool,
}

/// Prints `ollama ps`'s loaded-model set under `label` so a model that
/// loaded mid-arm is visible in the run's log, not just in the precheck.
fn log_ollama_ps(label: &str) -> Vec<String> {
    let loaded = ollama_loaded_models();
    println!(
        "ollama_ps label={label} loaded_count={} loaded={}",
        loaded.len(),
        if loaded.is_empty() {
            "none".to_string()
        } else {
            loaded.join(",")
        }
    );
    loaded
}

/// Runs one arm bracketed by `ollama ps` snapshots; the arm is flagged
/// loaded-contaminated if a model shows up loaded once the arm has finished,
/// even though none was loaded going in.
fn run_arm_logged(
    model: &LoadedModel,
    source: DecodeSource<'_>,
    max_tokens: usize,
    config: ServingConfig,
    sample_gpu: bool,
    label: &str,
) -> (ArmResult, bool) {
    log_ollama_ps(&format!("before_{label}"));
    let result = run_one_arm(model, source, max_tokens, config, sample_gpu, None);
    let after = log_ollama_ps(&format!("after_{label}"));
    (result, !after.is_empty())
}

fn run_pair(
    model: &LoadedModel,
    source: DecodeSource<'_>,
    max_tokens: usize,
    off_config: ServingConfig,
    on_config: ServingConfig,
    swap_order: bool,
    sample_gpu: bool,
) -> PairResult {
    let idle_samples: Vec<f64> = if sample_gpu {
        (0..3)
            .filter_map(|_| {
                let sample = gpu_utilization_sample();
                std::thread::sleep(GPU_SAMPLE_INTERVAL);
                sample
            })
            .collect()
    } else {
        Vec::new()
    };
    let idle_baseline = summarize_gpu_idle_samples(&idle_samples);
    if let Some(baseline) = &idle_baseline {
        println!("{}", format_gpu_idle_baseline("pair_gpu_idle_baseline", baseline));
    }
    let gpu_idle_contaminated = idle_baseline
        .is_some_and(|baseline| baseline.median > GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT);

    let (off, on, ollama_contaminated) = if swap_order {
        let (on, on_loaded) = run_arm_logged(model, source, max_tokens, on_config, sample_gpu, "on");
        let (off, off_loaded) = run_arm_logged(model, source, max_tokens, off_config, sample_gpu, "off");
        (off, on, on_loaded || off_loaded)
    } else {
        let (off, off_loaded) = run_arm_logged(model, source, max_tokens, off_config, sample_gpu, "off");
        let (on, on_loaded) = run_arm_logged(model, source, max_tokens, on_config, sample_gpu, "on");
        (off, on, off_loaded || on_loaded)
    };

    PairResult {
        off,
        on,
        contaminated: gpu_idle_contaminated || ollama_contaminated,
    }
}

// ---------------------------------------------------------------------
// prefill-once: one forward pass per prompt, every arm resumes from it
// ---------------------------------------------------------------------

/// gemma4-E2B's own real chat template (`tokenizer.chat_template`
/// metadata) -- the exact rendering `tests/gemma4_correctness_gate.rs`'s
/// own `chat_prompt` uses and confirms against this checkpoint's real
/// vocab (`<|turn>` is id 105, `<turn|>` is id 106, NOT the older
/// gemma2/3 `<start_of_turn>`/`<end_of_turn>` pair). Every corpus prompt is
/// wrapped in this template before tokenization here -- a completion-style
/// RAG passage and a bare one-line chat question both become a real "user
/// turn", so neither depends on the raw corpus text happening to contain a
/// newline for [`split_prompt_at_hard_boundary`] to find (the bug this
/// fixes: 17 of the 51 corpus prompts are one-line and panicked with no
/// template applied). The model's own BOS is prepended by the tokenizer
/// (`tokenizer.ggml.add_bos_token`), so this template never spells `<bos>`
/// itself.
fn chat_prompt(user_turn: &str) -> String {
    format!("<|turn>user\n{user_turn}<turn|>\n<|turn>model\n")
}

/// Splits a [`chat_prompt`]-rendered prompt at its own trailing newline --
/// the one literal newline [`chat_prompt`] always appends after the
/// model-turn opener (`<|turn>model`), regardless of whether `user_turn`
/// itself carries embedded newlines. `rfind` therefore always lands on this
/// template-guaranteed boundary, never on a newline inside the RAG passage
/// or chat question: nothing follows the template's own trailing newline,
/// so it is unconditionally the LAST one in the string.
///
/// The newline stays in the SUFFIX half, not the prefix: keeping it in the
/// prefix (an earlier version of this function did) makes the prefix the
/// entire templated text and the suffix empty whenever `prompt` ends in
/// that newline -- and an empty suffix leaves no token for
/// [`LoadedModel::generate_from_prefix`] to forward-evaluate before it can
/// sample (see that function's own doc). Splitting on the newline itself
/// keeps `prefix` ending exactly at the model-turn opener (this crate's own
/// "prefix is the templated prompt through the opener" invariant) and
/// leaves `suffix` a single real token -- the newline -- to seed
/// generation from. A prompt with no newline at all has no verified-safe
/// cut point, so this refuses loudly rather than guessing at an arbitrary
/// byte offset that could split a multi-byte character or a BPE merge.
///
/// Still a generic last-newline split, not a [`chat_prompt`]-only one:
/// [`run_verify_width_sweep_mode`]'s own multi-paragraph prompt (no
/// template, several embedded newlines with real text after the last one)
/// splits the same way and keeps working.
fn split_prompt_at_hard_boundary(prompt: &str) -> (&str, &str) {
    let newline_index = prompt.rfind('\n').unwrap_or_else(|| {
        panic!(
            "prompt has no newline to split on -- no verified-safe prefix/suffix boundary \
             (prompt starts: {:?})",
            prompt.chars().take(60).collect::<String>()
        )
    });
    prompt.split_at(newline_index)
}

/// One prefill per prompt, kept alive across every OFF/ON arm and every
/// pair a caller runs against it -- the performance-harness invariant that
/// re-prefilling per arm would otherwise violate. `suffix` is a borrow of
/// `prompt` itself (from [`split_prompt_at_hard_boundary`]), never a copy.
struct CachedPrompt<'prompt> {
    prefix_state: PrefixState,
    suffix: &'prompt str,
    prefill_ttft_ms: f64,
}

fn prefill_prompt_once<'prompt>(
    model: &LoadedModel,
    prompt: &'prompt str,
    serving_config: &ServingConfig,
) -> CachedPrompt<'prompt> {
    let (prefix_text, suffix) = split_prompt_at_hard_boundary(prompt);
    let prefill_start = Instant::now();
    let prefix_state = model
        .prefill_prefix(prefix_text, serving_config)
        .expect("prefill this prompt's shared prefix exactly once");
    let prefill_ttft_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    CachedPrompt {
        prefix_state,
        suffix,
        prefill_ttft_ms,
    }
}

/// Invariant 1's own proof: a fresh full-prompt decode and a resumed
/// decode off [`prefill_prompt_once`]'s cached prefix must sample the
/// IDENTICAL first 32 greedy token ids. Run once per bench (prompt index
/// 0), never per pair -- this is a correctness gate, not a measurement.
fn verify_prefix_resume_matches_full_decode(
    model: &LoadedModel,
    full_prompt: &str,
    cached: &CachedPrompt<'_>,
    serving_config: ServingConfig,
) {
    const VERIFY_TOKENS: usize = 32;
    let mut fresh_stats = SpeculativeDecodeStats::default();
    let (fresh_ids, ..) = model
        .generate_streaming_with_speculative_stats(
            full_prompt,
            VERIFY_TOKENS,
            serving_config,
            &mut |_event| ControlFlow::Continue(()),
            &mut fresh_stats,
            None,
        )
        .expect("fresh full-prompt decode for the prefix-resume parity check");
    let mut resumed_stats = SpeculativeDecodeStats::default();
    let (resumed_ids, ..) = model
        .generate_from_prefix_with_speculative_stats(
            &cached.prefix_state,
            cached.suffix,
            VERIFY_TOKENS,
            serving_config,
            &mut |_event| ControlFlow::Continue(()),
            &mut resumed_stats,
            None,
        )
        .expect("resumed decode for the prefix-resume parity check");
    println!(
        "prefix_resume_parity fresh_first32={fresh_ids:?} resumed_first32={resumed_ids:?} \
         identical={}",
        fresh_ids == resumed_ids,
    );
    assert_eq!(
        resumed_ids, fresh_ids,
        "resuming decode from the cached prefix must produce identical greedy token ids to a \
         fresh full-prompt decode -- prefix/suffix split is not a tokenizer-safe boundary"
    );
}

// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------

fn base_serving_config(gpu_layers: i32) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers,
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
        ..ServingConfig::default()
    }
}

/// Every field of one arm (invariants 2/3): ms/token, TTFT, p50/p99
/// per-token latency, verify/accept/draft counts, RSS, CPU%, and GPU
/// utilization summary when sampled -- none blank.
fn format_arm(label: &str, arm: &ArmResult) -> String {
    let gpu = arm.gpu.map_or_else(
        || {
            format!(
                "{label}_gpu_mean=n/a {label}_gpu_median=n/a {label}_gpu_p10=n/a \
                 {label}_gpu_p90=n/a {label}_gpu_n=0"
            )
        },
        |summary| {
            format!(
                "{label}_gpu_mean={:.2} {label}_gpu_median={:.2} {label}_gpu_p10={:.2} \
                 {label}_gpu_p90={:.2} {label}_gpu_n={}",
                summary.mean, summary.median, summary.p10, summary.p90, summary.sample_count
            )
        },
    );
    format!(
        "{label}_ms_per_token={:.3} {label}_ttft_ms={:.3} {label}_p50_token_ms={:.3} \
         {label}_p99_token_ms={:.3} {label}_verify_steps={} {label}_accepted_total={} \
         {label}_drafted_total={} {label}_rss_bytes={} {label}_cpu_percent={:.2} {gpu}",
        arm.ms_per_token,
        arm.ttft_ms,
        arm.p50_token_ms,
        arm.p99_token_ms,
        arm.verify_steps,
        arm.accepted_total,
        arm.drafted_total,
        arm.rss_bytes,
        arm.cpu_percent,
    )
}

fn run_pairs_mode(model: &LoadedModel, vocab: &Vocab, args: &BenchArgs, unmeasured_label: &str) {
    let drafter = args.drafter.unwrap_or(SpeculativeType::NgramSimple);
    if drafter != SpeculativeType::NgramSimple {
        eprintln!(
            "speculative_bench: --drafter {}: not wired in the decode loop yet -- \
             `speculative-decode-llama-parity/TASKS.md` slice 9 (Drafter enum + \
             --drafter flag) wires every type besides ngram-simple",
            drafter.llama_name()
        );
        std::process::exit(2);
    }

    if let Some(incumbent) = &args.incumbent
        && incumbent != "llama-server"
    {
        eprintln!("speculative_bench: --incumbent {incumbent}: only \"llama-server\" is wired");
        std::process::exit(2);
    }

    let corpus_path = args
        .corpus_path
        .clone()
        .unwrap_or_else(|| panic!("--corpus is required in pairs mode"));
    let raw_prompts = load_corpus(&corpus_path);
    assert!(
        !raw_prompts.is_empty(),
        "corpus {} contained no {{\"prompt\": ...}} lines",
        corpus_path.display()
    );
    // Every corpus prompt -- RAG passage or bare one-line chat question
    // alike -- is a USER TURN, wrapped in `chat_prompt` so
    // `split_prompt_at_hard_boundary` always has its template-guaranteed
    // trailing newline instead of depending on the raw corpus text's own.
    let prompts: Vec<String> = raw_prompts.iter().map(|raw| chat_prompt(raw)).collect();

    let sample_gpu = args.gpu_layers != 0 && cfg!(target_os = "macos");
    let off_config = base_serving_config(args.gpu_layers);
    let on_config = off_config.with_speculative(SpeculativeConfig {
        speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramSimple),
        ..SpeculativeConfig::none()
    });

    let add_bos = wants_bos(vocab);
    let add_eos = vocab.add_eos_token().unwrap_or(false);
    // Tokenized once, up front, for the whole corpus -- reused by every
    // prompt's own loop iteration below (never re-tokenized per pair), AND
    // to size the incumbent's own `-c` before it spawns: llama.cpp's real
    // per-hardware default is `-c 0` ("whatever the model was trained
    // with"), which this harness deliberately does NOT hand it (see
    // `spawn_incumbent_servers`'s own doc) in favor of sizing to what the
    // corpus + this run's own `--max-tokens` actually need.
    let token_ids_per_prompt: Vec<Vec<u32>> = prompts
        .iter()
        .map(|prompt| {
            proxima_tokenizer::encode_with_bos_eos(prompt, vocab, add_bos, add_eos)
                .expect("tokenize prompt under the cached_len convention run_one_arm's decode call uses")
        })
        .collect();
    let longest_prompt_tokens = token_ids_per_prompt
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or_else(|| panic!("corpus produced no tokenized prompts"));
    // Headroom absorbs the handful of extra tokens `cache_prompt`
    // bookkeeping and the speculative draft buffer can append past
    // `max_tokens` worth of predicted tokens. Generous, not tight:
    // oversizing `-c` costs KV-cache memory, undersizing it truncates or
    // forces a mid-run reprocess -- the wrong failure mode for a
    // correctness-sensitive comparison run.
    const LLAMA_CONTEXT_HEADROOM_TOKENS: usize = 256;
    let llama_context_length = u32::try_from(longest_prompt_tokens + args.max_tokens + LLAMA_CONTEXT_HEADROOM_TOKENS)
        .expect("corpus's longest prompt + max_tokens + headroom fits in a u32 context size");

    let incumbent_handles = args.incumbent.as_deref().map(|_| {
        spawn_incumbent_servers(
            &args.llama_server_bin,
            &args.model_path,
            drafter,
            &off_config,
            &on_config,
            llama_context_length,
        )
    });

    let mut pair_ratios: Vec<f64> = Vec::new();
    let mut llama_pair_ratios: Vec<f64> = Vec::new();
    let mut contaminated_pairs = 0usize;
    let mut verify_steps_total = 0u64;
    let mut accepted_total = 0u64;
    let mut drafted_total = 0u64;
    let mut wins = 0usize;
    let mut llama_wins = 0usize;

    for (prompt_index, prompt) in prompts.iter().enumerate() {
        let token_ids = &token_ids_per_prompt[prompt_index];
        if prompt_index == 0
            && let Some((off_handle, _)) = &incumbent_handles
        {
            check_token_parity(off_handle, prompt, token_ids, add_bos);
        }
        if let Some((off_handle, on_handle)) = &incumbent_handles {
            // discarded per-prompt warm call: primes each llama-server's
            // own KV cache so the first TIMED pair below does not pay this
            // prompt's one-time prefill cost -- mirrors prefill_prompt_once
            // below, which does the same for proxima's own arm.
            warm_llama_prompt_cache(off_handle, token_ids);
            warm_llama_prompt_cache(on_handle, token_ids);
        }

        // Exactly one forward pass over this prompt's own tokens, kept
        // alive across every pair below -- the performance-harness
        // invariant (`speculative_bench` no longer re-prefills per arm).
        let cached = prefill_prompt_once(model, prompt, &off_config);
        println!(
            "{unmeasured_label} prefill prompt_index={prompt_index} \
             prefill_ttft_ms={:.3} prefix_tokens={} suffix_chars={}",
            cached.prefill_ttft_ms,
            cached.prefix_state.len(),
            cached.suffix.len(),
        );
        // Every prompt, not just prompt 0 -- a split-boundary bug specific
        // to one prompt's own template rendering (e.g. an embedded
        // newline confusing the tokenizer at the cut point) must never
        // pass silently just because prompt 0 happened to be safe.
        verify_prefix_resume_matches_full_decode(model, prompt, &cached, off_config);
        let source = DecodeSource {
            prefix: &cached.prefix_state,
            suffix: cached.suffix,
        };

        for pair_index in 0..args.pairs {
            let swap_order = pair_index % 2 == 1;
            let pair = run_pair(
                model,
                source,
                args.max_tokens,
                off_config,
                on_config,
                swap_order,
                sample_gpu,
            );
            verify_steps_total += pair.on.verify_steps;
            accepted_total += pair.on.accepted_total;
            drafted_total += pair.on.drafted_total;
            let ratio = if pair.on.ms_per_token > 0.0 {
                pair.off.ms_per_token / pair.on.ms_per_token
            } else {
                0.0
            };
            let is_warmup = pair_index == 0;

            let llama_pair = incumbent_handles.as_ref().map(|(off_handle, on_handle)| {
                run_llama_pair(off_handle, on_handle, token_ids, args.max_tokens, swap_order)
            });
            let llama_ratio = llama_pair.as_ref().map(|llama_pair| {
                if llama_pair.on.ms_per_token > 0.0 {
                    llama_pair.off.ms_per_token / llama_pair.on.ms_per_token
                } else {
                    0.0
                }
            });

            let llama_fields = llama_pair.as_ref().map_or_else(String::new, |llama_pair| {
                format!(
                    " llama_ratio={:.4} {} {}",
                    llama_ratio.unwrap_or(0.0),
                    format_llama_arm("off", &llama_pair.off),
                    format_llama_arm("on", &llama_pair.on),
                )
            });
            println!(
                "{unmeasured_label} pair prompt_prefix={:?} pair_index={pair_index} \
                 warmup={is_warmup} contaminated={} ratio={ratio:.4} {} {}{llama_fields}",
                prompt.chars().take(24).collect::<String>(),
                pair.contaminated,
                format_arm("off", &pair.off),
                format_arm("on", &pair.on),
            );
            if is_warmup {
                // warmup pair: discarded from ratio stats, per SPEC's own
                // "warmup pairs discarded" clause -- pipeline/plan-cache
                // compilation on the first pair for this prompt would
                // otherwise inflate whichever arm ran first. Still printed
                // above (invariant 2/3: no field silently dropped), just
                // excluded from the aggregate below.
                continue;
            }
            if pair.contaminated {
                contaminated_pairs += 1;
                continue;
            }
            if ratio > 1.0 {
                wins += 1;
            }
            pair_ratios.push(ratio);
            if let Some(llama_ratio) = llama_ratio {
                if llama_ratio > 1.0 {
                    llama_wins += 1;
                }
                llama_pair_ratios.push(llama_ratio);
            }
        }
    }

    if let Some((off_handle, on_handle)) = incumbent_handles {
        off_handle.stop();
        on_handle.stop();
    }

    pair_ratios.sort_by(|left, right| left.partial_cmp(right).expect("finite ratios"));
    let median_speedup = percentile(&pair_ratios, 50.0);
    let p90_speedup = percentile(&pair_ratios, 90.0);
    let pair_cov = coefficient_of_variation(&pair_ratios);
    let win_fraction = if pair_ratios.is_empty() {
        0.0
    } else {
        wins as f64 / pair_ratios.len() as f64
    };

    let llama_summary_fields = if llama_pair_ratios.is_empty() {
        String::new()
    } else {
        llama_pair_ratios.sort_by(|left, right| left.partial_cmp(right).expect("finite ratios"));
        let llama_median_speedup = percentile(&llama_pair_ratios, 50.0);
        let llama_win_fraction = llama_wins as f64 / llama_pair_ratios.len() as f64;
        format!(
            " proxima_speedup={median_speedup:.4} llama_speedup={llama_median_speedup:.4} \
             llama_win_fraction={llama_win_fraction:.4}"
        )
    };

    println!(
        "{unmeasured_label} summary drafter={} prompts={} pairs_per_prompt={} \
         median_speedup={:.4} p90_speedup={:.4} pair_cov={:.4} win_fraction={:.4} \
         contaminated_pairs={contaminated_pairs} verify_steps_total={verify_steps_total} \
         accepted_total={accepted_total} drafted_total={drafted_total}{llama_summary_fields}",
        drafter.llama_name(),
        prompts.len(),
        args.pairs,
        median_speedup,
        p90_speedup,
        pair_cov,
        win_fraction,
    );
}

/// Per SPEC's own architecture paragraph -- "ms per verify forward at width
/// k+1... via the verify program with forced drafts". Genuinely forces an
/// exact width via `run_one_arm`'s own `forced_draft_width` argument, which
/// reaches [`LoadedModel::generate_streaming_with_speculative_stats`]
/// directly rather than `ServingConfig`: `decode.rs`'s own speculative
/// branch fills the draft buffer with `k` copies of the just-sampled token
/// instead of calling a real drafter, then runs the SAME
/// [`LoadedModel::speculative_verify_program`] forward every natural draft
/// uses -- no forward-pass code is duplicated here. `k=0` degrades to plain
/// non-speculative decode (speculation off), matching every other width-0
/// arm in this harness.
fn run_verify_width_sweep_mode(model: &LoadedModel, args: &BenchArgs, widths: &[usize], unmeasured_label: &str) {
    const REPEATED_PARAGRAPH: &str = "The quick brown fox jumps over the lazy dog while a curious cat \
         watches quietly from the garden wall. Pack my box with five dozen liquor jugs before \
         the delivery truck arrives at noon.";
    // Joined by newline, never `.repeat` -- `split_prompt_at_hard_boundary`
    // needs a newline with real suffix text after it, not one at the very
    // end (an empty suffix has no tokens left to forward-evaluate).
    let prompt = [REPEATED_PARAGRAPH; 6].join("\n");
    let sample_gpu = args.gpu_layers != 0 && cfg!(target_os = "macos");
    let serving_config = base_serving_config(args.gpu_layers);

    // Same prefill-once path `run_pairs_mode` uses (invariant 3): one
    // forward pass over this fixed prompt, reused across every width and
    // every one of the 3 runs per width below.
    let cached = prefill_prompt_once(model, &prompt, &serving_config);
    println!(
        "{unmeasured_label} prefill prompt_index=0 prefill_ttft_ms={:.3} prefix_tokens={} \
         suffix_chars={}",
        cached.prefill_ttft_ms,
        cached.prefix_state.len(),
        cached.suffix.len(),
    );
    verify_prefix_resume_matches_full_decode(model, &prompt, &cached, serving_config);
    let source = DecodeSource {
        prefix: &cached.prefix_state,
        suffix: cached.suffix,
    };

    let mut ms_per_verify_at_zero = 0.0;
    let mut rows: Vec<(usize, f64, f64, f64)> = Vec::new();

    for &width in widths {
        let mut samples_ms_per_verify = Vec::new();
        for _run in 0..3 {
            let forced_draft_width = if width == 0 { None } else { Some(width as u16) };
            let arm = run_one_arm(
                model,
                source,
                args.max_tokens,
                serving_config,
                sample_gpu,
                forced_draft_width,
            );
            let steps = arm.verify_steps.max(1);
            let wall_over_decode = arm.ms_per_token * (steps as f64).max(1.0);
            let ms_per_verify = if width == 0 {
                arm.ms_per_token
            } else {
                wall_over_decode / steps as f64
            };
            samples_ms_per_verify.push(ms_per_verify);
        }
        samples_ms_per_verify.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
        let cov = coefficient_of_variation(&samples_ms_per_verify);
        let median = percentile(&samples_ms_per_verify, 50.0);
        if width == 0 {
            ms_per_verify_at_zero = median;
        }
        println!(
            "{unmeasured_label} verify_width_sweep k={width} width_plus_one={} \
             ms_per_verify={median:.4} cov={cov:.4}",
            width + 1,
        );
        rows.push((width, median, cov, 0.0));
    }

    let largest = rows.last().copied();
    if let Some((k, ms_per_verify_k, _cov, _)) = largest
        && ms_per_verify_at_zero > 0.0
    {
        let break_even_accepted_per_step = ms_per_verify_k / ms_per_verify_at_zero - 1.0;
        println!(
            "{unmeasured_label} break_even_accepted_per_step={break_even_accepted_per_step:.4} \
             at k={k} (ms_per_verify_k1={ms_per_verify_at_zero:.4})"
        );
    }
}

/// One env lever this binary's own gemma4-on-Metal prefill/decode path
/// reads (`omega/src/msl/kernel_types_identity.rs`'s own A/B switches) --
/// `value` is `None` when the caller left the var unset, matching every
/// switch's own "unset default" posture.
struct LeverVar {
    name: &'static str,
    value: Option<String>,
}

impl LeverVar {
    fn read(name: &'static str) -> Self {
        LeverVar {
            name,
            value: env::var(name).ok(),
        }
    }

    fn is_one(&self) -> bool {
        matches!(self.value.as_deref(), Some(value) if value.trim() == "1")
    }

    fn display(&self) -> String {
        format!("{}={}", self.name, self.value.as_deref().unwrap_or("unset"))
    }
}

/// The compiled-in cargo features that move timing numbers for this
/// binary (`proxima-model-interop/Cargo.toml` lines 128-227), rendered
/// `key=true/false` via `cfg!` so the printed line reflects THIS binary's
/// own compilation, never an assumption about what the caller meant to
/// build with.
fn compiled_perf_features_summary() -> String {
    format!(
        "metal_feature={} metal_fuse_attn_decode_feature={} identity_copy_alias_feature={} metal_tiled_gemm_feature={}",
        cfg!(feature = "metal"),
        cfg!(feature = "metal-fuse-attn-decode"),
        cfg!(feature = "identity-copy-alias"),
        cfg!(feature = "metal-tiled-gemm"),
    )
}

/// The short commit this binary was built from, read at startup rather
/// than baked in by a build script (no build.rs exists in this crate) --
/// `unknown` when `git` is unavailable or this tree is not a git checkout.
fn git_commit_at_startup() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |commit| commit.trim().to_string())
}

/// Prints every lever this bench's prefill/verify/decode path reads,
/// per run (never assumed from a prior run -- SPEC's own "a result can
/// never be read without its config"), and refuses outright when
/// `PROXIMA_TILED_GEMM_DENSE=1` is set: on this branch's base (`df3766dd`)
/// `classify_dense_batched_gemm`'s `grid_threads` arm has no dense-batched
/// dispatch shape yet, so admitting a dense op onto the tiled-GEMM path
/// over-dispatches by roughly 4080x and a 510-token prefill hangs past
/// 180s -- the fix for that gap is not on this branch.
fn print_lever_config_and_refuse_if_unsafe() {
    let levers = [
        LeverVar::read("PROXIMA_MULTI_ROW_UNROLL"),
        LeverVar::read("PROXIMA_MULTI_ROW_INDEX32"),
        LeverVar::read("PROXIMA_COORD_INDEX32"),
        LeverVar::read("PROXIMA_TILED_GEMM_Q4_0"),
        LeverVar::read("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD"),
        LeverVar::read("PROXIMA_TILED_GEMM_SLIM_TGMEM"),
        LeverVar::read("PROXIMA_TILED_GEMM_DENSE"),
    ];
    println!(
        "lever_config {} {} git_commit={}",
        levers
            .iter()
            .map(LeverVar::display)
            .collect::<Vec<_>>()
            .join(" "),
        compiled_perf_features_summary(),
        git_commit_at_startup(),
    );
    let dense = &levers[6];
    if dense.is_one() {
        eprintln!(
            "speculative_bench: refusing to run with {} -- on this branch's base \
             (df3766dd) the dense-batched tiled-GEMM arm over-dispatches roughly 4080x \
             and hangs a 510-token prefill past 180s; unset PROXIMA_TILED_GEMM_DENSE and \
             retry",
            dense.display()
        );
        std::process::exit(4);
    }
}

fn main() {
    install_orphan_reaping_panic_hook();
    print_lever_config_and_refuse_if_unsafe();
    let args = parse_args();
    let precheck = quiet_box_precheck();

    if let Some(baseline) = &precheck.gpu_idle_baseline {
        eprintln!(
            "speculative_bench: {}",
            format_gpu_idle_baseline("gpu_idle_baseline", baseline)
        );
    }
    if let Some(baseline) = &precheck.cpu_idle_baseline {
        eprintln!("speculative_bench: {}", format_cpu_idle_baseline(baseline));
    }
    if let Some(line) = &precheck.load_average_line {
        eprintln!("speculative_bench: {line}");
    }
    if !precheck.is_quiet() && !args.force {
        eprintln!("speculative_bench: quiet-box precheck failed, refusing to run without --force:");
        for reason in &precheck.busy_reasons {
            eprintln!("  - {reason}");
        }
        std::process::exit(3);
    }
    if !precheck.is_quiet() {
        eprintln!("speculative_bench: --force set; every printed line below is `unmeasured` (host busy):");
        for reason in &precheck.busy_reasons {
            eprintln!("  - {reason}");
        }
    }
    let unmeasured_label = if args.force { "unmeasured" } else { "measured" };

    let file = File::open(&args.model_path)
        .unwrap_or_else(|err| panic!("open model {}: {err}", args.model_path));
    // SAFETY: `file` stays alive for the duration of this mapping's use --
    // it is dropped only at the end of `main`, after every decode call.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map model");
    let parsed = parse_complete(&bytes).expect("parse model");
    let model = LoadedModel::load(&parsed, &bytes).expect("bind model");
    // SAME tokenization `run_one_arm`'s decode call performs internally
    // (`generate/residency_caches.rs::wants_bos`, crate-private) -- this
    // example rebuilds the identical `Vocab` from the identical metadata to
    // hand llama-server the SAME token ids proxima decodes, rather than
    // trusting the two tokenizers to agree from a shared prompt string
    // (invariant 2).
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("builds vocab via the current gemma4 dispatch");

    match &args.mode {
        BenchMode::Pairs => run_pairs_mode(&model, &vocab, &args, unmeasured_label),
        BenchMode::VerifyWidthSweep { widths } => {
            run_verify_width_sweep_mode(&model, &args, widths, unmeasured_label);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::{
        ChildGuard, CpuIdleBaseline, GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT,
        LLAMA_REPREFILL_PROMPT_N_BOUND, REAPABLE_PIDS, chat_prompt, compiled_perf_features_summary,
        cpu_idle_decision, git_commit_at_startup, parse_llama_completion_response,
        parse_top_cpu_line, split_prompt_at_hard_boundary, summarize_gpu_idle_samples,
    };

    const REAL_TOP_CPU_LINE: &str = "CPU usage: 8.97% user, 5.40% sys, 85.61% idle ";

    #[test]
    fn top_cpu_line_from_this_mac_parses_user_sys_idle() {
        let baseline = parse_top_cpu_line(REAL_TOP_CPU_LINE).expect("real top line parses");

        assert_eq!(
            baseline,
            CpuIdleBaseline {
                user: 8.97,
                sys: 5.40,
                idle: 85.61
            }
        );
    }

    #[test]
    fn top_line_without_cpu_usage_prefix_is_rejected() {
        assert_eq!(parse_top_cpu_line("Processes: 512 total, 2 running"), None);
    }

    #[test]
    fn cpu_idle_85_percent_passes_and_40_percent_refuses() {
        let quiet = parse_top_cpu_line(REAL_TOP_CPU_LINE).expect("real top line parses");
        let contended = CpuIdleBaseline {
            user: 45.0,
            sys: 15.0,
            idle: 40.0,
        };

        assert_eq!(cpu_idle_decision(&quiet), None);
        let reason = cpu_idle_decision(&contended).expect("40% idle must refuse");
        assert!(reason.contains("idle=40.00%"), "reason carries the sample: {reason}");
    }

    /// A single `ollama ps` wake-the-Electron-app spike inside an otherwise
    /// idle 50-sample/5s window (the exact shape that produced three false
    /// refusals): the mean crosses [`GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT`]
    /// but the median stays at 0, so the median-based decision passes.
    fn idle_window_with_one_spike(spike_percent: f64, sample_count: usize) -> Vec<f64> {
        let mut samples = vec![0.0_f64; sample_count];
        samples[sample_count / 2] = spike_percent;
        samples
    }

    #[test]
    fn all_zero_idle_window_is_not_contaminated_on_median() {
        let samples = vec![0.0_f64; 50];

        let baseline = summarize_gpu_idle_samples(&samples).expect("50 samples summarize");

        assert!(
            baseline.median <= GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT,
            "an all-idle window must not read contaminated, got median={}",
            baseline.median
        );
    }

    #[test]
    fn one_spike_in_fifty_passes_on_median() {
        let samples = idle_window_with_one_spike(88.0, 50);

        let baseline = summarize_gpu_idle_samples(&samples).expect("50 samples summarize");

        assert!(
            baseline.median <= GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT,
            "the median-based rule must pass a single spike in an otherwise idle window, got median={}",
            baseline.median
        );
        assert!(
            (baseline.max - 88.0).abs() < 1e-9,
            "the spike must still be visible in max, got {}",
            baseline.max
        );
    }

    /// The reported bug's own shape: a 5-sample/0.5s window (the pre-fix
    /// sample size) where a single `ollama ps` wake-spike drags the mean
    /// past [`GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT`] even though four of
    /// the five samples are genuinely idle -- the mean-based rule this fix
    /// replaces would have refused this run; the median-based rule does not.
    #[test]
    fn one_spike_in_five_fails_on_mean_but_passes_on_median() {
        let samples = idle_window_with_one_spike(30.0, 5);

        let baseline = summarize_gpu_idle_samples(&samples).expect("5 samples summarize");

        assert!(
            baseline.mean > GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT,
            "this is the exact shape that produced three false refusals -- mean={}",
            baseline.mean
        );
        assert!(
            baseline.median <= GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT,
            "the median-based rule must pass this window, got median={}",
            baseline.median
        );
    }

    #[test]
    fn sustained_high_utilization_fails_on_median() {
        let samples = vec![30.0_f64; 50];

        let baseline = summarize_gpu_idle_samples(&samples).expect("50 samples summarize");

        assert!(
            baseline.median > GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT,
            "sustained non-idle utilization must fail the median-based rule, got median={}",
            baseline.median
        );
    }

    #[test]
    fn empty_idle_window_summarizes_to_none() {
        assert!(
            summarize_gpu_idle_samples(&[]).is_none(),
            "an empty sample vector (ioreg unavailable) must not fabricate a baseline"
        );
    }

    /// Field layout taken from llama.cpp's own non-streaming `/completion`
    /// response (`server-task.cpp:340-358`'s `to_json_non_oaicompat`), with
    /// `timings` built off `server_slot_stats::to_json`
    /// (`server-common.cpp:84-105`) -- a cache HIT shape: `prompt_n` covers
    /// only the tokens past the previously cached common prefix, well under
    /// [`LLAMA_REPREFILL_PROMPT_N_BOUND`].
    fn cache_hit_response_body() -> serde_json::Value {
        serde_json::json!({
            "index": 0,
            "content": " a friendly island.",
            "tokens": [],
            "id_slot": 0,
            "stop": true,
            "model": "gemma-3-4b-it",
            "tokens_predicted": 6,
            "tokens_evaluated": 512,
            "prompt": "",
            "has_new_line": false,
            "truncated": false,
            "stop_type": "eos",
            "stopping_word": "",
            "tokens_cached": 517,
            "timings": {
                "cache_n": 512,
                "prompt_n": 1,
                "prompt_ms": 4.221,
                "prompt_per_token_ms": 4.221,
                "prompt_per_second": 236.912,
                "predicted_n": 6,
                "predicted_ms": 71.883,
                "predicted_per_token_ms": 11.980,
                "predicted_per_second": 83.470,
                "draft_n": 24,
                "draft_n_accepted": 18,
            },
        })
    }

    /// Same field layout, a cache MISS shape: `prompt_n` covers the entire
    /// 512-token prompt because the slot's common prefix lookup
    /// (`server-context.cpp:3198`'s `get_common_prefix`) found nothing to
    /// reuse -- the exact silent-cost-regression case
    /// [`crate::warm_llama_prompt_cache`] exists to keep off timed pairs.
    fn cache_miss_response_body() -> serde_json::Value {
        serde_json::json!({
            "index": 0,
            "content": " a friendly island.",
            "tokens": [],
            "id_slot": 0,
            "stop": true,
            "model": "gemma-3-4b-it",
            "tokens_predicted": 6,
            "tokens_evaluated": 512,
            "prompt": "",
            "has_new_line": false,
            "truncated": false,
            "stop_type": "eos",
            "stopping_word": "",
            "tokens_cached": 517,
            "timings": {
                "cache_n": 0,
                "prompt_n": 512,
                "prompt_ms": 612.44,
                "prompt_per_token_ms": 1.196,
                "prompt_per_second": 836.115,
                "predicted_n": 6,
                "predicted_ms": 71.883,
                "predicted_per_token_ms": 11.980,
                "predicted_per_second": 83.470,
            },
        })
    }

    #[test]
    fn parse_llama_completion_response_reads_timing_fields_on_cache_hit() {
        let arm = parse_llama_completion_response(&cache_hit_response_body());

        assert!(
            (arm.ms_per_token - 11.980).abs() < 1e-9,
            "ms_per_token should come straight off timings.predicted_per_token_ms, got {}",
            arm.ms_per_token
        );
        assert!(
            (arm.ttft_ms - 4.221).abs() < 1e-9,
            "ttft_ms should come straight off timings.prompt_ms, got {}",
            arm.ttft_ms
        );
        assert_eq!(arm.predicted_n, 6, "predicted_n should come off timings.predicted_n");
        assert_eq!(arm.draft_n, 24, "draft_n should come off timings.draft_n");
        assert_eq!(
            arm.draft_n_accepted, 18,
            "draft_n_accepted should come off timings.draft_n_accepted"
        );
        assert_eq!(arm.prompt_n, 1, "prompt_n should come off timings.prompt_n");
    }

    #[test]
    fn parse_llama_completion_response_flags_a_cache_hit_as_not_reprefilled() {
        let arm = parse_llama_completion_response(&cache_hit_response_body());

        assert!(
            !arm.llama_reprefilled,
            "prompt_n=1 is under LLAMA_REPREFILL_PROMPT_N_BOUND={LLAMA_REPREFILL_PROMPT_N_BOUND}; \
             a resumed prompt must not be flagged as re-prefilled"
        );
    }

    #[test]
    fn parse_llama_completion_response_flags_a_cache_miss_as_reprefilled() {
        let arm = parse_llama_completion_response(&cache_miss_response_body());

        assert_eq!(arm.prompt_n, 512, "a cache miss processes the whole prompt");
        assert!(
            arm.llama_reprefilled,
            "prompt_n=512 exceeds LLAMA_REPREFILL_PROMPT_N_BOUND={LLAMA_REPREFILL_PROMPT_N_BOUND}; \
             a cache miss must be visible, never silent"
        );
    }

    #[test]
    fn parse_llama_completion_response_defaults_missing_draft_fields_to_zero() {
        // llama-server only emits draft_n/draft_n_accepted when
        // n_draft_tokens > 0 (server-common.cpp:99-102) -- a non-speculative
        // ("off") arm's response omits them entirely.
        let arm = parse_llama_completion_response(&cache_miss_response_body());

        assert_eq!(arm.draft_n, 0, "missing draft_n must default to 0, not panic");
        assert_eq!(arm.draft_n_accepted, 0, "missing draft_n_accepted must default to 0, not panic");
    }

    #[test]
    #[should_panic(expected = "missing \"timings\"")]
    fn parse_llama_completion_response_panics_on_missing_timings() {
        let response = serde_json::json!({ "content": "no timings field at all" });
        let _ = parse_llama_completion_response(&response);
    }

    /// Regression for the real-run panic: a one-line corpus prompt ("What's
    /// the difference between a Roth IRA...") has no newline of its own, so
    /// the split must rely entirely on `chat_prompt`'s own trailing newline.
    #[test]
    fn splits_a_one_line_chat_template_prompt_at_the_model_turn_opener() {
        let templated =
            chat_prompt("What's the difference between a Roth IRA and a traditional IRA?");

        let (prefix, suffix) = split_prompt_at_hard_boundary(&templated);

        assert!(
            prefix.ends_with("<|turn>model"),
            "prefix must end exactly at the model-turn opener, got {prefix:?}"
        );
        assert_eq!(suffix, "\n", "suffix must be the template's own trailing newline only");
        assert_eq!(
            format!("{prefix}{suffix}"),
            templated,
            "prefix and suffix must reassemble the original templated prompt exactly"
        );
    }

    /// A multi-line user turn (embedded newlines inside the RAG-style
    /// passage) must still split on the TEMPLATE's own trailing newline,
    /// not on one of the embedded ones -- `chat_prompt` always appends its
    /// own newline last, so `rfind` never sees the embedded ones.
    #[test]
    fn splits_a_multi_line_chat_template_prompt_at_the_model_turn_opener() {
        let templated = chat_prompt(
            "# Section one\n\nSome context spanning\nseveral lines.\n\nWhat does this say?",
        );

        let (prefix, suffix) = split_prompt_at_hard_boundary(&templated);

        assert!(
            prefix.ends_with("<|turn>model"),
            "prefix must end exactly at the model-turn opener, got {prefix:?}"
        );
        assert_eq!(suffix, "\n", "suffix must be the template's own trailing newline only");
        assert_eq!(
            format!("{prefix}{suffix}"),
            templated,
            "prefix and suffix must reassemble the original templated prompt exactly"
        );
    }

    /// Regression for the real-run orphan bug: two llama-server processes
    /// survived a mid-run panic because nothing ever killed them.
    /// `sleep 60` stands in for llama-server here -- any long-lived,
    /// harmless child proves the same claim (`ChildGuard::drop` kills AND
    /// waits it) without needing the real binary or a model on this host.
    #[test]
    fn child_guard_kills_and_reaps_a_real_child_on_drop() {
        let child = Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn a harmless `sleep 60` child for the kill-on-drop test");
        let pid = child.id();

        {
            let guard = ChildGuard::new(child);
            assert_eq!(guard.pid(), pid);
            assert!(
                REAPABLE_PIDS.lock().expect("lock REAPABLE_PIDS").contains(&pid),
                "ChildGuard::new must register its pid for the panic-hook fallback"
            );
        } // `guard` drops here -- kills and waits `sleep 60`.

        assert!(
            !REAPABLE_PIDS.lock().expect("lock REAPABLE_PIDS").contains(&pid),
            "ChildGuard::drop must deregister its pid once it has reaped the child"
        );
        // `wait()` inside `Drop::drop` already reaped the process, so a
        // liveness probe must find nothing -- a lingering zombie would still
        // answer `kill -0` on some platforms, so this also confirms `wait()`
        // ran, not just that a signal was sent.
        let status = Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .status()
            .expect("run `kill -0` to probe the child's liveness");
        assert!(
            !status.success(),
            "pid {pid} still answers a liveness probe after ChildGuard was dropped"
        );
    }

    #[test]
    fn compiled_perf_features_summary_names_every_performance_relevant_feature() {
        let summary = compiled_perf_features_summary();

        for key in [
            "metal_feature=",
            "metal_fuse_attn_decode_feature=",
            "identity_copy_alias_feature=",
            "metal_tiled_gemm_feature=",
        ] {
            assert!(
                summary.contains(key),
                "lever_config must attribute {key} on the same line, got: {summary}"
            );
        }
        assert!(
            summary.split(' ').all(|field| field.ends_with("=true") || field.ends_with("=false")),
            "every feature field must render a bool, got: {summary}"
        );
    }

    #[test]
    fn git_commit_at_startup_never_panics_and_is_never_empty() {
        let commit = git_commit_at_startup();

        assert!(
            !commit.is_empty(),
            "git_commit_at_startup must fall back to \"unknown\", never an empty string"
        );
        assert!(
            !commit.contains(char::is_whitespace),
            "git_commit_at_startup must trim to a bare token, got: {commit:?}"
        );
    }
}
