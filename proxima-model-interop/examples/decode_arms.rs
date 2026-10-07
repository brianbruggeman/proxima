//! Interleaved decode ms/token arms: proxima builds, llama-server, Ollama.
//!
//! One driver, no scripts: every arm is launched here under a cleared
//! environment, the GPU-peer gate is checked before every launch, every launch
//! is logged with its UTC time, and every per-run number is printed raw before
//! any summary. Proxima arms are `decode_gbps_baseline` binaries (one per
//! build under comparison); they run as `processes` rounds of
//! `warmup + runs` generations each, rounds interleaved A B A B so a drifting
//! GPU clock or background load lands on every arm alike. llama-server and
//! Ollama are driven over HTTP with `std::net`, temperature 0, the same prompt
//! text, `n_predict`/`num_predict` 128 unless `--new-tokens N` says otherwise.
//!
//! Outlier rule, fixed before any run: a run is an outlier when its distance
//! from its arm's pooled median exceeds `3 * 1.4826 * MAD` of that pool. Raw
//! runs, the outliers, and both summaries (all runs, outliers removed) print.
//!
//! Every proxima child runs under `/usr/bin/time -l`, whose `maximum resident
//! set size` and `peak memory footprint` lines are the per-process memory
//! read; a child that prints `gpu_peak_bytes` (the Metal device's allocated
//! size sampled at every token) adds the peak GPU allocation. `--model` names
//! the checkpoint for children that read `PROXIMA_DECODE_MODEL_GGUF`.
//!
//! `--dump-llama-ids DIR` writes, per case, the first llama-server process's last run as one
//! record (`prompt`, llama's `prompt_ids`, greedy `generated_ids`) in the shape
//! `tests/arch_data_baseline.rs` reads, so a long-prompt oracle fixture is a run of this driver.
//!
//! ```sh
//! cargo run --release -p proxima-model-interop --example decode_arms -- \
//!   --prompt-file prompt1k.txt --log launches.log --processes 2 --runs 7 \
//!   --arm base=/path/decode_gbps_baseline_base --arm tip=/path/decode_gbps_baseline_tip \
//!   --llama-server /path/llama-server --ollama gemma4:e2b-it-qat
//! ```
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
const PEER_NAMES: [&str; 7] = [
    "census",
    "decode_gbps",
    "llama-server",
    "xctrace",
    "nextest",
    "gguf_generate",
    "speculative_bench",
];
const PEER_WAIT: Duration = Duration::from_secs(3600);
const LLAMA_ENGINE: &str = "llama-server";
const OLLAMA_ENGINE: &str = "ollama";
const LLAMA_PORT: u16 = 8097;
const OLLAMA_PORT: u16 = 11434;
const DEFAULT_NEW_TOKENS: usize = 128;
const CONTEXT_TOKENS: usize = 4096;
const OUTLIER_MAD_SCALE: f64 = 3.0 * 1.4826;

static SERVER_PID: AtomicU32 = AtomicU32::new(0);
static IGNORE_OLLAMA: AtomicBool = AtomicBool::new(false);
static NEW_TOKENS: AtomicUsize = AtomicUsize::new(DEFAULT_NEW_TOKENS);

// the release profile aborts on panic, so Drop never runs and a failed request
// would orphan the server; the hook kills the registered pid before the abort
fn install_server_cleanup_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        kill_registered_server();
        previous(info);
    }));
}

fn new_tokens() -> usize {
    NEW_TOKENS.load(Ordering::SeqCst)
}

fn kill_registered_server() {
    let pid = SERVER_PID.swap(0, Ordering::SeqCst);
    if pid != 0 {
        let _ = Command::new("kill").arg(pid.to_string()).status();
    }
}

struct Case {
    name: String,
    model: String,
    ollama_tag: Option<String>,
}

struct Arguments {
    prompt_file: PathBuf,
    log: PathBuf,
    processes: usize,
    runs: usize,
    arms: Vec<(String, PathBuf)>,
    llama_server: Option<PathBuf>,
    ollama_tag: Option<String>,
    model: String,
    cases: Vec<Case>,
    dump_llama_ids: Option<PathBuf>,
}

fn parse_arguments() -> Arguments {
    let mut arguments = Arguments {
        prompt_file: PathBuf::new(),
        log: PathBuf::new(),
        processes: 2,
        runs: 7,
        arms: Vec::new(),
        llama_server: None,
        ollama_tag: None,
        model: MODEL_PATH.to_string(),
        cases: Vec::new(),
        dump_llama_ids: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--prompt-file" => arguments.prompt_file = PathBuf::from(value()),
            "--log" => arguments.log = PathBuf::from(value()),
            "--processes" => arguments.processes = value().parse().expect("integer"),
            "--runs" => arguments.runs = value().parse().expect("integer"),
            "--model" => arguments.model = value(),
            "--case" => arguments.cases.push(parse_case(&value())),
            "--llama-server" => arguments.llama_server = Some(PathBuf::from(value())),
            "--dump-llama-ids" => arguments.dump_llama_ids = Some(PathBuf::from(value())),
            "--ollama" => arguments.ollama_tag = Some(value()),
            "--new-tokens" => NEW_TOKENS.store(value().parse().expect("integer"), Ordering::SeqCst),
            "--ignore-ollama" => {
                // a token-id correctness run, not a timing run, so an idle
                // resident Ollama is not a GPU peer
                IGNORE_OLLAMA.store(true, Ordering::SeqCst);
            }
            "--arm" => {
                let spec = value();
                let (label, path) = spec.split_once('=').expect("--arm label=path");
                arguments
                    .arms
                    .push((label.to_string(), PathBuf::from(path)));
            }
            other => panic!("unknown flag {other}"),
        }
    }
    assert!(
        !arguments.prompt_file.as_os_str().is_empty(),
        "--prompt-file required"
    );
    assert!(!arguments.log.as_os_str().is_empty(), "--log required");
    if arguments.cases.is_empty() {
        arguments.cases.push(Case {
            name: String::new(),
            model: arguments.model.clone(),
            ollama_tag: arguments.ollama_tag.clone(),
        });
    }
    arguments
}

fn parse_case(spec: &str) -> Case {
    let mut parts = spec.splitn(3, '=');
    let name = parts.next().expect("--case name=gguf[=ollama_tag]");
    let model = parts.next().expect("--case name=gguf[=ollama_tag]");
    Case {
        name: name.to_string(),
        model: model.to_string(),
        ollama_tag: parts.next().map(str::to_string),
    }
}

fn utc_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_secs() as i64;
    let days = seconds.div_euclid(86_400);
    let remainder = seconds.rem_euclid(86_400);
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        remainder / 3600,
        remainder % 3600 / 60,
        remainder % 60
    )
}

fn log_line(arguments: &Arguments, text: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&arguments.log)
        .expect("open launches log");
    writeln!(file, "{} {text}", utc_now()).expect("write launches log");
}

fn running_peers(ollama_is_a_peer: bool) -> Vec<String> {
    let output = Command::new("ps")
        .args(["-axo", "comm"])
        .output()
        .expect("ps runs");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| {
            PEER_NAMES.iter().any(|name| line.contains(name))
                || (ollama_is_a_peer
                    && !IGNORE_OLLAMA.load(Ordering::SeqCst)
                    && line.to_ascii_lowercase().contains("ollama"))
        })
        .map(str::to_string)
        .collect()
}

fn require_quiet_gpu(what: &str) {
    let started = Instant::now();
    loop {
        let peers = running_peers(what != "ollama");
        if peers.is_empty() {
            return;
        }
        assert!(
            started.elapsed() < PEER_WAIT,
            "refusing to launch {what}: GPU peers still present after {PEER_WAIT:?}: {peers:?}"
        );
        std::thread::sleep(Duration::from_secs(5));
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn cov_percent(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean * 100.0
}

fn outlier_flags(values: &[f64]) -> Vec<bool> {
    if values.is_empty() {
        return Vec::new();
    }
    let center = median(values);
    let deviations: Vec<f64> = values.iter().map(|value| (value - center).abs()).collect();
    let mad = median(&deviations);
    values
        .iter()
        .map(|value| mad > 0.0 && (value - center).abs() > OUTLIER_MAD_SCALE * mad)
        .collect()
}

struct ArmRuns {
    case: String,
    engine: String,
    label: String,
    runs: Vec<(usize, usize, f64)>,
    prefill_runs: Vec<(usize, usize, f64)>,
    ttft_runs: Vec<(usize, usize, f64)>,
    gpu_peak_runs: Vec<(usize, usize, f64)>,
    rss_by_process: Vec<(usize, usize, f64)>,
    footprint_by_process: Vec<(usize, usize, f64)>,
    ids_by_run: Vec<Vec<u64>>,
    extra: Vec<String>,
}

impl ArmRuns {
    fn new(case: &str, engine: &str, qualified: bool) -> Self {
        let label = if qualified {
            format!("{case}.{engine}")
        } else {
            engine.to_string()
        };
        Self {
            case: case.to_string(),
            engine: engine.to_string(),
            label,
            runs: Vec::new(),
            prefill_runs: Vec::new(),
            ttft_runs: Vec::new(),
            gpu_peak_runs: Vec::new(),
            rss_by_process: Vec::new(),
            footprint_by_process: Vec::new(),
            ids_by_run: Vec::new(),
            extra: Vec::new(),
        }
    }
}

fn mad(values: &[f64]) -> f64 {
    let center = median(values);
    let deviations: Vec<f64> = values.iter().map(|value| (value - center).abs()).collect();
    median(&deviations)
}

fn kept_values(runs: &[(usize, usize, f64)]) -> Vec<f64> {
    let values: Vec<f64> = runs.iter().map(|run| run.2).collect();
    let flags = outlier_flags(&values);
    values
        .iter()
        .zip(&flags)
        .filter_map(|(value, flagged)| (!flagged).then_some(*value))
        .collect()
}

fn print_metric(label: &str, metric: &str, runs: &[(usize, usize, f64)]) {
    let values: Vec<f64> = runs.iter().map(|run| run.2).collect();
    let flags = outlier_flags(&values);
    for ((process, run, value), flagged) in runs.iter().zip(&flags) {
        println!("arm={label} metric={metric} process={process} run={run} value={value:.4} outlier={flagged}");
    }
    let kept = kept_values(runs);
    println!(
        "summary arm={label} metric={metric} n_all={} median_all={:.4} cov_all_pct={:.2} min_all={:.4} max_all={:.4} | n_kept={} median_kept={:.4} mad_kept={:.4} cov_kept_pct={:.2} min_kept={:.4} max_kept={:.4}",
        values.len(),
        median(&values),
        cov_percent(&values),
        values.iter().copied().fold(f64::INFINITY, f64::min),
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        kept.len(),
        median(&kept),
        mad(&kept),
        cov_percent(&kept),
        kept.iter().copied().fold(f64::INFINITY, f64::min),
        kept.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
}

fn print_summary(arm: &ArmRuns) {
    print_metric(&arm.label, "ms_per_token", &arm.runs);
    if !arm.prefill_runs.is_empty() {
        print_metric(&arm.label, "prefill_ms", &arm.prefill_runs);
    }
    if !arm.ttft_runs.is_empty() {
        print_metric(&arm.label, "ttft_ms", &arm.ttft_runs);
    }
    print_memory(arm);
    for line in &arm.extra {
        println!("extra arm={} {line}", arm.label);
    }
    std::io::stdout().flush().expect("flush stdout");
}

fn print_memory(arm: &ArmRuns) {
    for (name, series) in [
        ("peak_rss_bytes", &arm.rss_by_process),
        ("peak_footprint_bytes", &arm.footprint_by_process),
        ("peak_gpu_bytes", &arm.gpu_peak_runs),
    ] {
        for (process, _, value) in series {
            println!("memory arm={} metric={name} process={process} value={value:.0}", arm.label);
        }
        let values: Vec<f64> = series.iter().map(|entry| entry.2).collect();
        if !values.is_empty() {
            println!(
                "summary arm={} metric={name} n={} median={:.0} max={:.0}",
                arm.label,
                values.len(),
                median(&values),
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            );
        }
    }
}

// bound: each arm against every earlier arm: median <= reference median + max(reference MAD, 2% of reference median), outliers removed
fn is_server_arm(arm: &ArmRuns) -> bool {
    arm.engine == LLAMA_ENGINE || arm.engine == OLLAMA_ENGINE
}

fn bound_references(arms: &[ArmRuns], index: usize) -> Vec<&ArmRuns> {
    let arm = &arms[index];
    if is_server_arm(arm) {
        return Vec::new();
    }
    arms.iter()
        .enumerate()
        .filter(|(other, reference)| {
            *other != index
                && reference.case == arm.case
                && (is_server_arm(reference) || *other < index)
        })
        .map(|(_, reference)| reference)
        .collect()
}

fn print_bounds(arms: &[ArmRuns]) {
    for (index, arm) in arms.iter().enumerate() {
        for reference in bound_references(arms, index) {
            bound_line(arm, reference, "ms_per_token", |each| &each.runs);
            bound_line(arm, reference, "prefill_ms", |each| &each.prefill_runs);
            bound_line(arm, reference, "ttft_ms", |each| &each.ttft_runs);
            memory_bound_line(arm, reference, "peak_rss_bytes", |each| &each.rss_by_process);
            memory_bound_line(arm, reference, "peak_footprint_bytes", |each| &each.footprint_by_process);
            memory_bound_line(arm, reference, "peak_gpu_bytes", |each| &each.gpu_peak_runs);
        }
    }
}

// bound: arm median <= reference median + 2% of reference median; one value per process, no outlier removal
fn memory_bound_line(
    arm: &ArmRuns,
    reference: &ArmRuns,
    metric: &str,
    select: impl Fn(&ArmRuns) -> &Vec<(usize, usize, f64)>,
) {
    let arm_values: Vec<f64> = select(arm).iter().map(|entry| entry.2).collect();
    let reference_values: Vec<f64> = select(reference).iter().map(|entry| entry.2).collect();
    if arm_values.is_empty() || reference_values.is_empty() {
        println!("bound metric={metric} arm={} vs={} n=0", arm.label, reference.label);
        return;
    }
    let reference_median = median(&reference_values);
    let slack = reference_median * 0.02;
    let arm_median = median(&arm_values);
    println!(
        "bound metric={metric} arm={} vs={} arm_median={arm_median:.0} reference_median={reference_median:.0} delta={:.0} limit={slack:.0} within={}",
        arm.label,
        reference.label,
        arm_median - reference_median,
        arm_median <= reference_median + slack
    );
}

fn bound_line(
    arm: &ArmRuns,
    reference: &ArmRuns,
    metric: &str,
    select: impl Fn(&ArmRuns) -> &Vec<(usize, usize, f64)>,
) {
    let arm_kept = kept_values(select(arm));
    let reference_kept = kept_values(select(reference));
    if arm_kept.is_empty() || reference_kept.is_empty() {
        println!("bound metric={metric} arm={} vs={} n=0", arm.label, reference.label);
        return;
    }
    let reference_median = median(&reference_kept);
    let slack = mad(&reference_kept).max(reference_median * 0.02);
    let arm_median = median(&arm_kept);
    println!(
        "bound metric={metric} arm={} vs={} arm_median={arm_median:.4} reference_median={reference_median:.4} delta={:.4} limit={slack:.4} within={}",
        arm.label,
        reference.label,
        arm_median - reference_median,
        arm_median <= reference_median + slack
    );
}

fn field_after<'text>(line: &'text str, key: &str) -> Option<&'text str> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest.find(' ').unwrap_or(rest.len());
    Some(&rest[..end])
}

fn parse_ids(line: &str) -> Vec<u64> {
    let start = line.find("ids=[").map_or(0, |index| index + 5);
    let end = line.rfind(']').unwrap_or(line.len());
    line[start..end]
        .split(',')
        .filter_map(|token| token.trim().parse().ok())
        .collect()
}

fn run_proxima_process(
    arguments: &Arguments,
    model: &str,
    arm: &mut ArmRuns,
    binary: &PathBuf,
    prompt: &str,
    process: usize,
) {
    require_quiet_gpu(&arm.label);
    log_line(
        arguments,
        &format!(
            "LAUNCH decode_arms proxima arm={} process={process} bin={}",
            arm.label,
            binary.display()
        ),
    );
    let output = Command::new("/usr/bin/time")
        .arg("-l")
        .arg(binary)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", std::env::var("HOME").expect("HOME"))
        .env("PROXIMA_GEMMA4_E2B_GGUF", model)
        .env("PROXIMA_DECODE_MODEL_GGUF", model)
        .env("PROXIMA_SPECULATIVE_TYPES", "none")
        .env("PROXIMA_PROMPT", prompt)
        .env("PROXIMA_MAX_TOKENS", new_tokens().to_string())
        .env("PROXIMA_RUNS", (arguments.runs + 1).to_string())
        .output()
        .expect("spawn decode_gbps_baseline");
    log_line(
        arguments,
        &format!(
            "EXIT decode_arms proxima arm={} process={process} status={}",
            arm.label, output.status
        ),
    );
    let peers_after = running_peers(true);
    arm.extra.push(format!(
        "process={process} peers_present_at_exit={peers_after:?}"
    ));
    assert!(
        output.status.success(),
        "{} failed: {}",
        arm.label,
        String::from_utf8_lossy(&output.stderr)
    );
    let raw_path = arguments
        .log
        .with_extension(format!("raw.{}.p{process}.err", arm.label));
    std::fs::write(&raw_path, &output.stderr).expect("persist raw child stderr");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut parsed = 0usize;
    let mut gpu_peak = 0.0f64;
    for line in stderr.lines() {
        if let Some(bytes) = time_report_bytes(line, "maximum resident set size") {
            arm.rss_by_process.push((process, 0, bytes));
        }
        if let Some(bytes) = time_report_bytes(line, "peak memory footprint") {
            arm.footprint_by_process.push((process, 0, bytes));
        }
        if line.contains("token_ids run_index=") {
            arm.ids_by_run.push(parse_ids(line));
        }
        if line.contains("run=done") {
            let run_index: usize = field_after(line, "run_index=")
                .expect("run_index")
                .parse()
                .expect("int");
            let value: f64 = field_after(line, "decode_ms_per_token=")
                .expect("decode_ms_per_token")
                .parse()
                .expect("float");
            let wall_ms: f64 = field_after(line, "wall_ms=")
                .expect("wall_ms")
                .parse()
                .expect("float");
            let tokens: f64 = field_after(line, "tokens_generated=")
                .expect("tokens_generated")
                .parse()
                .expect("float");
            let prefill_ms = wall_ms - value * (tokens - 1.0);
            let ttft = field_after(line, "ttft_ms=").and_then(|text| text.parse::<f64>().ok());
            if let Some(ttft) = ttft.filter(|_| run_index > 0) {
                arm.ttft_runs.push((process, run_index, ttft));
            }
            if let Some(bytes) = field_after(line, "gpu_peak_bytes=").and_then(|text| text.parse::<f64>().ok()) {
                gpu_peak = gpu_peak.max(bytes);
            }
            if run_index > 0 {
                arm.runs.push((process, run_index, value));
                arm.prefill_runs.push((process, run_index, prefill_ms));
            }
            println!(
                "raw arm={} process={process} run={run_index} ms_per_token={value:.4}",
                arm.label
            );
            std::io::stdout().flush().expect("flush stdout");
            parsed += 1;
        }
    }
    assert_eq!(
        parsed,
        arguments.runs + 1,
        "N mismatch: run=done lines parsed"
    );
    if gpu_peak > 0.0 {
        arm.gpu_peak_runs.push((process, 0, gpu_peak));
    }
    assert!(
        arm.rss_by_process.iter().any(|entry| entry.0 == process),
        "no maximum resident set size line from /usr/bin/time -l for arm {} process {process}",
        arm.label
    );
}

fn time_report_bytes(line: &str, label: &str) -> Option<f64> {
    let trimmed = line.trim();
    let value = trimmed.strip_suffix(label)?.trim();
    value.parse::<f64>().ok()
}

struct HttpExchange {
    body: String,
    first_chunk: Duration,
}

fn http_request(port: u16, method: &str, path: &str, body: &str, timeout: Duration) -> String {
    http_exchange(port, method, path, body, timeout).body
}

fn body_has_data(raw: &[u8]) -> bool {
    let Some(separator) = raw.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let head = String::from_utf8_lossy(&raw[..separator]).to_ascii_lowercase();
    let payload = &raw[separator + 4..];
    if head.contains("transfer-encoding: chunked") {
        !dechunk(payload).is_empty()
    } else {
        !payload.is_empty()
    }
}

// first_chunk is the client clock from the request write to the first read that carries a complete body chunk
fn http_exchange(
    port: u16,
    method: &str,
    path: &str,
    body: &str,
    timeout: Duration,
) -> HttpExchange {
    let address: SocketAddr = format!("127.0.0.1:{port}").parse().expect("address");
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(5)).expect("connect");
    stream.set_read_timeout(Some(timeout)).expect("timeout");
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let started = Instant::now();
    stream.write_all(request.as_bytes()).expect("write request");
    let mut raw = Vec::new();
    let mut first_chunk = None;
    let mut buffer = vec![0u8; 65_536];
    loop {
        let read = stream.read(&mut buffer).expect("read response");
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..read]);
        if first_chunk.is_none() && body_has_data(&raw) {
            first_chunk = Some(started.elapsed());
        }
    }
    let separator = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("http response head");
    let head = String::from_utf8_lossy(&raw[..separator]).to_ascii_lowercase();
    let payload = &raw[separator + 4..];
    let bytes = if head.contains("transfer-encoding: chunked") {
        dechunk(payload)
    } else {
        payload.to_vec()
    };
    HttpExchange {
        body: String::from_utf8(bytes).expect("utf-8 response body"),
        first_chunk: first_chunk.unwrap_or_else(|| started.elapsed()),
    }
}

fn dechunk(payload: &[u8]) -> Vec<u8> {
    let mut rest = payload;
    let mut decoded = Vec::new();
    while let Some(line_end) = rest.windows(2).position(|window| window == b"\r\n") {
        let Ok(size_text) = std::str::from_utf8(&rest[..line_end]) else {
            break;
        };
        let Ok(size) = usize::from_str_radix(size_text.trim(), 16) else {
            break;
        };
        let start = line_end + 2;
        let Some(chunk) = rest.get(start..start + size).filter(|_| size > 0) else {
            break;
        };
        decoded.extend_from_slice(chunk);
        rest = rest.get(start + size + 2..).unwrap_or(&[]);
    }
    decoded
}

fn wait_for_http(port: u16, path: &str, seconds: u64) {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(seconds) {
        let address: SocketAddr = format!("127.0.0.1:{port}").parse().expect("address");
        if TcpStream::connect_timeout(&address, Duration::from_secs(1)).is_ok() {
            let answer = http_request(port, "GET", path, "", Duration::from_secs(5));
            let refused = answer.to_ascii_lowercase().contains("\"error\"");
            if !answer.is_empty() && !refused {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    panic!("port {port} never answered {path}");
}

struct ServerRun {
    prefill_ms: f64,
    per_token_ms: f64,
    ttft_ms: f64,
    ids: Vec<u64>,
    note: String,
}

fn json_lines(body: &str, prefix: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix(prefix))
        .filter_map(|json| serde_json::from_str(json).ok())
        .collect()
}

// one record in the shape `arch_data_baseline`'s `llama_cases` reads: the prompt, the ids llama
// tokenized it to, and the ids it generated greedily
fn dump_llama_record(directory: &PathBuf, case: &str, prompt: &str, ids_by_run: &[Vec<u64>]) {
    let tokenized: Value = serde_json::from_str(&http_request(
        LLAMA_PORT,
        "POST",
        "/tokenize",
        &json!({"content": prompt, "add_special": true}).to_string(),
        Duration::from_secs(30),
    ))
    .expect("llama tokenize json");
    let generated = ids_by_run.last().expect("a llama run produced ids");
    let record = json!([{
        "prompt": prompt,
        "prompt_ids": tokenized["tokens"],
        "generated_ids": generated,
    }]);
    std::fs::create_dir_all(directory).expect("create llama ids directory");
    let name = if case.is_empty() { "case" } else { case };
    let path = directory.join(format!("{name}_llama_ids.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&record).expect("serialize record"))
        .expect("write llama ids record");
    println!("llama ids record case={case} path={}", path.display());
}

fn llama_request(prompt: &str) -> ServerRun {
    let body = json!({"prompt": prompt, "n_predict": new_tokens(), "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": false, "ignore_eos": true, "stream": true, "return_tokens": true}).to_string();
    let exchange = http_exchange(LLAMA_PORT, "POST", "/completion", &body, Duration::from_secs(120));
    let events = json_lines(&exchange.body, "data: ");
    let timings = events
        .iter()
        .rev()
        .find_map(|event| event.get("timings"))
        .unwrap_or_else(|| {
            panic!(
                "llama stream without timings: {}",
                &exchange.body[..exchange.body.len().min(400)]
            )
        });
    let ids: Vec<u64> = events
        .iter()
        .filter_map(|event| event["tokens"].as_array())
        .flatten()
        .filter_map(Value::as_u64)
        .collect();
    ServerRun {
        prefill_ms: timings["prompt_ms"].as_f64().expect("prompt_ms"),
        per_token_ms: timings["predicted_per_token_ms"]
            .as_f64()
            .expect("predicted_per_token_ms"),
        ttft_ms: exchange.first_chunk.as_secs_f64() * 1000.0,
        note: format!(
            "prompt_n={} predicted_n={} ids_len={} events={}",
            timings["prompt_n"],
            timings["predicted_n"],
            ids.len(),
            events.len()
        ),
        ids,
    }
}

// the server runs under /usr/bin/time -l so its peak rss is the kernel's, as for the proxima arms
struct TimedServer {
    timer: Child,
    stderr_path: PathBuf,
}

impl TimedServer {
    fn spawn(binary: &PathBuf, model: &str, stderr_path: PathBuf) -> Self {
        let stderr_file = std::fs::File::create(&stderr_path).expect("create server stderr file");
        let timer = Command::new("/usr/bin/time")
            .arg("-l")
            .arg(binary)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", std::env::var("HOME").expect("HOME"))
            .args([
                "--model",
                model,
                "-c",
                &CONTEXT_TOKENS.to_string(),
                "-ngl",
                "99",
                "-np",
                "1",
                "--temp",
                "0",
                "--top-k",
                "1",
                "--seed",
                "1",
                "--host",
                "127.0.0.1",
                "--port",
                &LLAMA_PORT.to_string(),
            ])
            .stdout(Stdio::null())
            .stderr(stderr_file)
            .spawn()
            .expect("spawn llama-server under time");
        SERVER_PID.store(timer.id(), Ordering::SeqCst);
        let server_pid = Self::child_of(timer.id());
        SERVER_PID.store(server_pid, Ordering::SeqCst);
        Self { timer, stderr_path }
    }

    fn child_of(parent: u32) -> u32 {
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(10) {
            let output = Command::new("pgrep")
                .args(["-P", &parent.to_string()])
                .output()
                .expect("pgrep runs");
            let pid = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .and_then(|line| line.trim().parse::<u32>().ok());
            if let Some(pid) = pid {
                return pid;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        panic!("time pid {parent} never started a child");
    }

    fn stop(mut self) -> String {
        kill_registered_server();
        let started = Instant::now();
        while self.timer.try_wait().expect("poll timer").is_none() {
            assert!(
                started.elapsed() < Duration::from_secs(60),
                "llama-server did not exit within 60 s of SIGTERM"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
        std::fs::read_to_string(&self.stderr_path).expect("read server stderr")
    }
}

impl Drop for TimedServer {
    fn drop(&mut self) {
        kill_registered_server();
        let _ = self.timer.wait();
    }
}

fn push_time_report(arm: &mut ArmRuns, process: usize, report: &str) {
    for line in report.lines() {
        if let Some(bytes) = time_report_bytes(line, "maximum resident set size") {
            arm.rss_by_process.push((process, 0, bytes));
        }
        if let Some(bytes) = time_report_bytes(line, "peak memory footprint") {
            arm.footprint_by_process.push((process, 0, bytes));
        }
    }
}

fn record_server_run(arm: &mut ArmRuns, process: usize, request: usize, run: ServerRun) {
    arm.extra.push(format!(
        "process={process} request={request} {} prefill_ms={:.4} ttft_ms={:.4} ms_per_token={:.4}",
        run.note, run.prefill_ms, run.ttft_ms, run.per_token_ms
    ));
    println!(
        "raw arm={} process={process} run={request} ms_per_token={:.4}",
        arm.label, run.per_token_ms
    );
    std::io::stdout().flush().expect("flush stdout");
    if request > 0 {
        arm.runs.push((process, request, run.per_token_ms));
        arm.prefill_runs.push((process, request, run.prefill_ms));
        arm.ttft_runs.push((process, request, run.ttft_ms));
    }
    arm.ids_by_run.push(run.ids);
}

fn run_llama_round(
    arguments: &Arguments,
    model: &str,
    binary: &PathBuf,
    prompt: &str,
    process: usize,
    arm: &mut ArmRuns,
) {
    require_quiet_gpu(&arm.label);
    log_line(
        arguments,
        &format!(
            "LAUNCH decode_arms llama-server arm={} process={process} port={LLAMA_PORT} requests={}",
            arm.label,
            arguments.runs + 1
        ),
    );
    let stderr_path = arguments
        .log
        .with_extension(format!("raw.{}.p{process}.err", arm.label));
    let server = TimedServer::spawn(binary, model, stderr_path);
    wait_for_http(LLAMA_PORT, "/health", 120);
    for request in 0..=arguments.runs {
        record_server_run(arm, process, request, llama_request(prompt));
    }
    if let (Some(directory), 0) = (&arguments.dump_llama_ids, process) {
        dump_llama_record(directory, &arm.case, prompt, &arm.ids_by_run);
    }
    let report = server.stop();
    push_time_report(arm, process, &report);
    log_line(
        arguments,
        &format!("EXIT decode_arms llama-server arm={} process={process}", arm.label),
    );
}

fn quit_ollama(arguments: &Arguments) {
    let _ = Command::new("osascript")
        .args(["-e", "tell application \"Ollama\" to quit"])
        .output();
    std::thread::sleep(Duration::from_secs(2));
    let _ = Command::new("pkill")
        .args(["-TERM", "-x", "Ollama"])
        .output();
    let _ = Command::new("pkill")
        .args(["-TERM", "-f", "Ollama.app/Contents/Resources/ollama"])
        .output();
    log_line(arguments, "ollama quit requested");
}

// sums the rss of every process whose executable path contains ollama, kept as the running maximum
struct OllamaRssSampler {
    stop: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<f64>,
}

fn ollama_rss_bytes() -> f64 {
    let output = Command::new("ps")
        .args(["-axo", "rss=,comm="])
        .output()
        .expect("ps runs");
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| line.to_ascii_lowercase().contains("ollama"))
        .filter_map(|line| line.split_whitespace().next()?.parse::<f64>().ok())
        .sum::<f64>()
        * 1024.0
}

impl OllamaRssSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            let mut peak = 0.0f64;
            while !flag.load(Ordering::SeqCst) {
                peak = peak.max(ollama_rss_bytes());
                std::thread::sleep(Duration::from_millis(50));
            }
            peak
        });
        Self { stop, handle }
    }

    fn finish(self) -> f64 {
        self.stop.store(true, Ordering::SeqCst);
        self.handle.join().expect("rss sampler thread")
    }
}

// keep_alive 0 drops the runner and with it Ollama's prompt prefix cache, so every request prefills the whole prompt
fn ollama_unload(tag: &str) {
    let body = json!({"model": tag, "keep_alive": 0}).to_string();
    http_request(OLLAMA_PORT, "POST", "/api/generate", &body, Duration::from_secs(60));
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(60) {
        let listing: Value = serde_json::from_str(&http_request(
            OLLAMA_PORT,
            "GET",
            "/api/ps",
            "",
            Duration::from_secs(5),
        ))
        .expect("ollama ps json");
        if listing["models"].as_array().is_some_and(Vec::is_empty) {
            return;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    panic!("ollama still holds a model 60 s after keep_alive 0");
}

// ttft is the client clock to the first streamed chunk minus the server's own load_duration, so a reload is not billed as prefill
fn ollama_request(tag: &str, prompt: &str) -> ServerRun {
    let body = json!({"model": tag, "prompt": prompt, "raw": true, "stream": true, "options": {"temperature": 0, "top_k": 1, "num_predict": new_tokens(), "seed": 1, "num_ctx": CONTEXT_TOKENS}}).to_string();
    let exchange = http_exchange(OLLAMA_PORT, "POST", "/api/generate", &body, Duration::from_secs(300));
    let events = json_lines(&exchange.body, "");
    let last = events.last().expect("ollama stream had no events");
    assert!(
        last["done"].as_bool().unwrap_or(false),
        "ollama stream ended without done: {last}"
    );
    let eval_count = last["eval_count"].as_u64().expect("eval_count");
    let eval_duration = last["eval_duration"].as_u64().expect("eval_duration");
    let load_ms = last["load_duration"].as_f64().unwrap_or(0.0) / 1e6;
    ServerRun {
        prefill_ms: last["prompt_eval_duration"].as_f64().expect("prompt_eval_duration") / 1e6,
        per_token_ms: eval_duration as f64 / eval_count as f64 / 1e6,
        ttft_ms: exchange.first_chunk.as_secs_f64() * 1000.0 - load_ms,
        note: format!(
            "eval_count={eval_count} prompt_eval_count={} load_ms={load_ms:.1} events={}",
            last["prompt_eval_count"],
            events.len()
        ),
        ids: Vec::new(),
    }
}

fn run_ollama_round(
    arguments: &Arguments,
    tag: &str,
    prompt: &str,
    process: usize,
    arm: &mut ArmRuns,
) {
    require_quiet_gpu("ollama");
    log_line(
        arguments,
        &format!(
            "LAUNCH decode_arms ollama arm={} tag={tag} process={process} requests={}",
            arm.label,
            arguments.runs + 1
        ),
    );
    Command::new("open")
        .args(["-a", "Ollama"])
        .status()
        .expect("open Ollama");
    wait_for_http(OLLAMA_PORT, "/api/tags", 120);
    let sampler = OllamaRssSampler::start();
    for request in 0..=arguments.runs {
        ollama_unload(tag);
        record_server_run(arm, process, request, ollama_request(tag, prompt));
    }
    arm.rss_by_process.push((process, 0, sampler.finish()));
    quit_ollama(arguments);
    log_line(
        arguments,
        &format!("EXIT decode_arms ollama arm={} process={process}", arm.label),
    );
}

fn compare_ids(arms: &[ArmRuns]) {
    for reference_arm in arms.iter().filter(|arm| arm.engine == LLAMA_ENGINE) {
        let Some(reference) = reference_arm.ids_by_run.last() else {
            continue;
        };
        for arm in arms
            .iter()
            .filter(|arm| arm.case == reference_arm.case && !is_server_arm(arm))
        {
            for (run, ids) in arm.ids_by_run.iter().enumerate() {
                let common = ids
                    .iter()
                    .zip(reference.iter())
                    .take_while(|(left, right)| left == right)
                    .count();
                println!(
                    "ids arm={} run={run} len={} llama_len={} common_prefix={common} equal={}",
                    arm.label,
                    ids.len(),
                    reference.len(),
                    ids == reference
                );
            }
        }
    }
}

fn case_arms(arguments: &Arguments, case: &Case, qualified: bool) -> Vec<ArmRuns> {
    let mut arms: Vec<ArmRuns> = arguments
        .arms
        .iter()
        .map(|(label, _)| ArmRuns::new(&case.name, label, qualified))
        .collect();
    if arguments.llama_server.is_some() {
        arms.push(ArmRuns::new(&case.name, LLAMA_ENGINE, qualified));
    }
    if case.ollama_tag.is_some() {
        arms.push(ArmRuns::new(&case.name, OLLAMA_ENGINE, qualified));
    }
    arms
}

fn run_arm(arguments: &Arguments, case: &Case, prompt: &str, process: usize, arm: &mut ArmRuns) {
    match (arm.engine.as_str(), &arguments.llama_server, &case.ollama_tag) {
        (LLAMA_ENGINE, Some(binary), _) => {
            run_llama_round(arguments, &case.model, binary, prompt, process, arm);
        }
        (OLLAMA_ENGINE, _, Some(tag)) => run_ollama_round(arguments, tag, prompt, process, arm),
        (label, _, _) => {
            let binary = &arguments
                .arms
                .iter()
                .find(|(arm_label, _)| arm_label == label)
                .expect("proxima arm label")
                .1;
            run_proxima_process(arguments, &case.model, arm, binary, prompt, process);
        }
    }
}

fn main() {
    install_server_cleanup_hook();
    let arguments = parse_arguments();
    let prompt = std::fs::read_to_string(&arguments.prompt_file).expect("read prompt file");
    let qualified = arguments.cases.len() > 1;
    let mut per_case: Vec<Vec<ArmRuns>> = arguments
        .cases
        .iter()
        .map(|case| case_arms(&arguments, case, qualified))
        .collect();
    for process in 0..arguments.processes {
        for (case, arms) in arguments.cases.iter().zip(per_case.iter_mut()) {
            let count = arms.len();
            for slot in 0..count {
                let rotated = (slot + process) % count;
                run_arm(&arguments, case, &prompt, process, &mut arms[rotated]);
            }
        }
    }
    let arms: Vec<ArmRuns> = per_case.into_iter().flatten().collect();
    for arm in &arms {
        print_summary(arm);
    }
    print_bounds(&arms);
    compare_ids(&arms);
}
