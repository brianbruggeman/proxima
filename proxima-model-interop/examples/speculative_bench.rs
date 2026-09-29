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
#![allow(clippy::expect_used, clippy::unwrap_used)]
#![allow(clippy::too_many_lines)]

use std::env;
use std::fs::File;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use core::ops::ControlFlow;
use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{
    GPU_LAYERS_ALL, LoadedModel, Phase, ServingConfig, SpeculativeConfig, SpeculativeDecodeStats,
    SpeculativeType, SpeculativeTypeSet, TokenEvent,
};

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
    force: bool,
    model_path: String,
}

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

fn run_one_arm(
    model: &LoadedModel,
    prompt: &str,
    max_tokens: usize,
    serving_config: ServingConfig,
    sample_gpu: bool,
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
        .generate_streaming_with_speculative_stats(
            prompt,
            max_tokens,
            serving_config,
            &mut on_token,
            &mut stats,
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
    prompt: &str,
    max_tokens: usize,
    config: ServingConfig,
    sample_gpu: bool,
    label: &str,
) -> (ArmResult, bool) {
    log_ollama_ps(&format!("before_{label}"));
    let result = run_one_arm(model, prompt, max_tokens, config, sample_gpu);
    let after = log_ollama_ps(&format!("after_{label}"));
    (result, !after.is_empty())
}

fn run_pair(
    model: &LoadedModel,
    prompt: &str,
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
        let (on, on_loaded) = run_arm_logged(model, prompt, max_tokens, on_config, sample_gpu, "on");
        let (off, off_loaded) = run_arm_logged(model, prompt, max_tokens, off_config, sample_gpu, "off");
        (off, on, on_loaded || off_loaded)
    } else {
        let (off, off_loaded) = run_arm_logged(model, prompt, max_tokens, off_config, sample_gpu, "off");
        let (on, on_loaded) = run_arm_logged(model, prompt, max_tokens, on_config, sample_gpu, "on");
        (off, on, off_loaded || on_loaded)
    };

    PairResult {
        off,
        on,
        contaminated: gpu_idle_contaminated || ollama_contaminated,
    }
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

fn run_pairs_mode(model: &LoadedModel, args: &BenchArgs, unmeasured_label: &str) {
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

    if let Some(incumbent) = &args.incumbent {
        eprintln!(
            "speculative_bench: --incumbent {incumbent}: not yet wired -- \
             `speculative-decode-llama-parity/TASKS.md` slice 24 (llama-server \
             incumbent arm) wires this"
        );
        std::process::exit(2);
    }

    let corpus_path = args
        .corpus_path
        .clone()
        .unwrap_or_else(|| panic!("--corpus is required in pairs mode"));
    let prompts = load_corpus(&corpus_path);
    assert!(
        !prompts.is_empty(),
        "corpus {} contained no {{\"prompt\": ...}} lines",
        corpus_path.display()
    );

    let sample_gpu = args.gpu_layers != 0 && cfg!(target_os = "macos");
    let off_config = base_serving_config(args.gpu_layers);
    let on_config = off_config.with_speculative(SpeculativeConfig {
        speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramSimple),
        ..SpeculativeConfig::none()
    });

    let mut pair_ratios: Vec<f64> = Vec::new();
    let mut contaminated_pairs = 0usize;
    let mut verify_steps_total = 0u64;
    let mut accepted_total = 0u64;
    let mut drafted_total = 0u64;
    let mut wins = 0usize;

    for prompt in &prompts {
        for pair_index in 0..args.pairs {
            let swap_order = pair_index % 2 == 1;
            let pair = run_pair(
                model,
                prompt,
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
            println!(
                "{unmeasured_label} pair prompt_prefix={:?} pair_index={pair_index} \
                 warmup={is_warmup} contaminated={} ratio={ratio:.4} {} {}",
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
        }
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

    println!(
        "{unmeasured_label} summary drafter={} prompts={} pairs_per_prompt={} \
         median_speedup={:.4} p90_speedup={:.4} pair_cov={:.4} win_fraction={:.4} \
         contaminated_pairs={contaminated_pairs} verify_steps_total={verify_steps_total} \
         accepted_total={accepted_total} drafted_total={drafted_total}",
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
/// k+1... via the verify program with forced drafts". Genuinely forcing an
/// exact width bypasses the drafter entirely and calls
/// [`LoadedModel::speculative_verify_program`] directly with a synthetic
/// draft of length `k`; that entry point does not exist yet (it is
/// `pub(crate)`, unexposed past the decode loop, per `architecture.rs:
/// 292-309`'s own doc that only `ngram-simple`'s natural draft path is
/// wired end to end today). This sweep is therefore an ENGINEERED
/// approximation, not the literal forced call the SPEC names -- documented
/// here rather than silently narrowed (guiding-principles principle 15):
/// `ngram_simple.size_m` is pinned to `k` on the repeated-paragraph prompt
/// (`speculative_decode_parity.rs`'s own `default_prompt`, chosen there
/// because its token period clears `ngram_simple_draft`'s match-recency
/// floor), so nearly every verify step's draft length is `k` once the
/// prompt has looped once; `mean_drafted_per_step` is printed alongside so
/// a reader can see how closely `k` was actually achieved, and `k=0`
/// degrades to plain non-speculative decode (speculation off). Wiring a
/// literal forced-width call is `speculative-decode-llama-parity`'s own
/// slice 9 (`Drafter` enum) work, not this harness's.
fn run_verify_width_sweep_mode(model: &LoadedModel, args: &BenchArgs, widths: &[usize], unmeasured_label: &str) {
    const REPEATED_PARAGRAPH: &str = "The quick brown fox jumps over the lazy dog while a curious cat \
         watches quietly from the garden wall. Pack my box with five dozen liquor jugs before \
         the delivery truck arrives at noon. ";
    let prompt = REPEATED_PARAGRAPH.repeat(6);
    let sample_gpu = args.gpu_layers != 0 && cfg!(target_os = "macos");

    let mut ms_per_verify_at_zero = 0.0;
    let mut rows: Vec<(usize, f64, f64, f64)> = Vec::new();

    for &width in widths {
        let mut samples_ms_per_verify = Vec::new();
        for _run in 0..3 {
            let serving_config = if width == 0 {
                base_serving_config(args.gpu_layers)
            } else {
                base_serving_config(args.gpu_layers).with_speculative(SpeculativeConfig {
                    speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramSimple),
                    ngram_simple: proxima_model_interop::NgramMapParams {
                        size_n: 6,
                        size_m: width as u16,
                        min_hits: 1,
                    },
                    ..SpeculativeConfig::none()
                })
            };
            let arm = run_one_arm(model, &prompt, args.max_tokens, serving_config, sample_gpu);
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

fn main() {
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

    match &args.mode {
        BenchMode::Pairs => run_pairs_mode(&model, &args, unmeasured_label),
        BenchMode::VerifyWidthSweep { widths } => {
            run_verify_width_sweep_mode(&model, &args, widths, unmeasured_label);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CpuIdleBaseline, GPU_IDLE_CONTAMINATION_THRESHOLD_PERCENT, cpu_idle_decision,
        parse_top_cpu_line, summarize_gpu_idle_samples,
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
}
