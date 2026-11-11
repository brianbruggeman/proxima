//! Interleaved KV-cache scoreboard: proxima builds against `llama-completion`.
//!
//! One driver, no scripts. Cells are {each proxima binary, llama cold, llama warm} x
//! {26-token short prompt, long prompt file} x {F32, F16, Q4_0-K + F16-V}. Every round visits
//! every cell once in the same order, so drift in GPU clock or background load lands on every
//! cell alike. Every child runs under `/usr/bin/time -l`; its `peak memory footprint` line is
//! the RAM read, and `getloadavg` is sampled before each launch.
//!
//! Metric definitions: proxima cold TTFT is the `phase=prefill` step of run 0 in a fresh
//! process, warm TTFT is the prefill of runs 1 and 2 (`PROXIMA_RUNS=3`,
//! `PROXIMA_STEP_TIMES=1`); proxima tok/s is `1000 / decode_ms_per_token` of the `run=done`
//! lines. Runs 1 and 2 are separate samples (`ttft_warm_run1_ms`, `tok_s_run1`, ...), so a
//! proxima cell has 2 x rounds warm samples; run 0's decode stays its own `tok_s_run0` column.
//! RAM and GPU are MB (1e6 bytes); GPU is the max `gpu_peak_bytes`, `steady_gpu_bytes` is as
//! logged. Cell order alternates forward and reverse per round so no cell always runs just
//! before another. Llama TTFT is
//! `prompt eval time` (cold with `--no-warmup`, warm without), llama tok/s is the
//! `eval time` line's `tokens per second`.
//!
//! Output directory gets `raw.tsv` (one row per process, appended as it completes) and
//! `scoreboard.md` (n, mean, CoV% per cell). A nonzero child exit is a row and a table column.
//!
//! This is a sibling of `decode_arms.rs`, which drives llama-server over HTTP with MAD outlier
//! removal; this driver reproduces the one-shot `llama-completion` cold/warm protocol instead.
//!
//! ```sh
//! cargo run --release -p proxima-model-interop --example kv_scoreboard --features std -- \
//!   --out DIR --model BLOB --llama PATH/llama-completion --long-prompt-file FILE \
//!   --prox new=PATH_A --prox old=PATH_B --rounds 5
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

const SHORT_PROMPT: &str = "<|turn>user\nWhich of these is smaller in size: a hippopotamus or a large office building?<turn|>\n<|turn>model\n";
const DEFAULT_ROUNDS: usize = 5;
const DEFAULT_TIME_BIN: &str = "/usr/bin/time";
const MEGABYTE: f64 = 1_000_000.0;
const TSV_HEADER: &str = "round\tengine\tlabel\tprompt\tkv\tkind\tload1\tfootprint_bytes\tgpu_peak_bytes\tgpu_steady_bytes\tttft_cold_ms\tttft_warm_run1_ms\tttft_warm_run2_ms\ttok_s_run0\ttok_s_run1\ttok_s_run2\texit_code\tnote";

#[derive(Debug, Error)]
enum HarnessError {
    #[error("missing required flag {0}")]
    MissingFlag(&'static str),
    #[error("flag {0} needs a value")]
    MissingValue(String),
    #[error("unknown flag {0}")]
    UnknownFlag(String),
    #[error("--prox expects LABEL=PATH, got {0}")]
    BadProx(String),
    #[error("--rounds expects a positive integer, got {0}")]
    BadRounds(String),
    #[error("log has no {0}")]
    MissingLine(&'static str),
    #[error("could not parse {field} from line: {line}")]
    BadNumber { field: &'static str, line: String },
    #[error("io on {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Prompt {
    Short,
    Long,
}

impl Prompt {
    fn name(self) -> &'static str {
        match self {
            Prompt::Short => "short",
            Prompt::Long => "long",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kv {
    F32,
    F16,
    Q4K,
}

impl Kv {
    fn name(self) -> &'static str {
        match self {
            Kv::F32 => "F32",
            Kv::F16 => "F16",
            Kv::Q4K => "Q4K",
        }
    }

    fn proxima_env(self) -> Vec<(&'static str, &'static str)> {
        match self {
            Kv::F32 => vec![("PROXIMA_KV_CACHE_TYPE", "f32")],
            Kv::F16 => vec![("PROXIMA_KV_CACHE_TYPE", "f16")],
            Kv::Q4K => vec![
                ("PROXIMA_KV_CACHE_TYPE", "f16"),
                ("PROXIMA_KV_CACHE_KEY_TYPE", "q4_0"),
            ],
        }
    }

    fn llama_flags(self) -> [&'static str; 4] {
        match self {
            Kv::F32 => ["-ctk", "f32", "-ctv", "f32"],
            Kv::F16 => ["-ctk", "f16", "-ctv", "f16"],
            Kv::Q4K => ["-ctk", "q4_0", "-ctv", "f16"],
        }
    }
}

const ALL_KV: [Kv; 3] = [Kv::F32, Kv::F16, Kv::Q4K];
const ALL_PROMPTS: [Prompt; 2] = [Prompt::Short, Prompt::Long];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Both,
    Cold,
    Warm,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Both => "both",
            Kind::Cold => "cold",
            Kind::Warm => "warm",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Engine {
    Proxima { binary: PathBuf },
    Llama,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Cell {
    engine: Engine,
    label: String,
    prompt: Prompt,
    kv: Kv,
    kind: Kind,
}

impl Cell {
    fn engine_name(&self) -> &'static str {
        match self.engine {
            Engine::Proxima { .. } => "proxima",
            Engine::Llama => "llama",
        }
    }
}

/// Cell order is fixed: prompt, then kv, then each proxima binary, then llama cold, llama warm.
fn cell_order(proxima: &[(String, PathBuf)]) -> Vec<Cell> {
    let mut cells = Vec::new();
    for prompt in ALL_PROMPTS {
        for kv in ALL_KV {
            for (label, binary) in proxima {
                cells.push(Cell {
                    engine: Engine::Proxima {
                        binary: binary.clone(),
                    },
                    label: label.clone(),
                    prompt,
                    kv,
                    kind: Kind::Both,
                });
            }
            for kind in [Kind::Cold, Kind::Warm] {
                cells.push(Cell {
                    engine: Engine::Llama,
                    label: "llama".to_owned(),
                    prompt,
                    kv,
                    kind,
                });
            }
        }
    }
    cells
}

/// Every round visits every cell once: odd rounds forward, even rounds reversed, so any two
/// cells swap relative order between rounds 1 and 2 and neither always runs first.
fn round_plan(rounds: usize, cell_count: usize) -> Vec<(usize, usize)> {
    (1..=rounds)
        .flat_map(|round| {
            (0..cell_count).map(move |position| {
                let index = if round % 2 == 1 {
                    position
                } else {
                    cell_count - 1 - position
                };
                (round, index)
            })
        })
        .collect()
}

#[derive(Debug, Default, Clone, PartialEq)]
struct Metrics {
    ttft_cold_ms: Option<f64>,
    /// warm prefill per run: proxima runs 1 and 2 as separate samples, llama one sample
    ttft_warm_ms: Vec<f64>,
    /// decode tok/s per run: proxima runs 1 and 2 as separate samples, llama one sample
    tok_s: Vec<f64>,
    tok_s_run0: Option<f64>,
    /// max `gpu_peak_bytes` over the run=done lines
    gpu_peak_bytes: Option<u64>,
    /// `steady_gpu_bytes` as logged on the last run=done line
    gpu_steady_bytes: Option<u64>,
}

fn number_after(line: &str, key: &str, field: &'static str) -> Result<f64, HarnessError> {
    let bad = || HarnessError::BadNumber {
        field,
        line: line.to_owned(),
    };
    let start = line.find(key).ok_or_else(bad)? + key.len();
    let rest = line[start..].trim_start();
    let end = rest
        .find(|character: char| {
            !(character.is_ascii_digit() || matches!(character, '.' | '-' | 'e' | 'E' | '+'))
        })
        .unwrap_or(rest.len());
    rest[..end].parse().map_err(|_| bad())
}

/// Reads `/usr/bin/time -l`'s `peak memory footprint` line (bytes).
fn parse_footprint_bytes(text: &str) -> Result<u64, HarnessError> {
    let line = text
        .lines()
        .find(|line| line.trim_end().ends_with("peak memory footprint"))
        .ok_or(HarnessError::MissingLine("peak memory footprint line"))?;
    let digits = line.split_whitespace().next().unwrap_or("");
    digits.parse().map_err(|_| HarnessError::BadNumber {
        field: "peak memory footprint",
        line: line.to_owned(),
    })
}

fn run_index_of(line: &str) -> Option<usize> {
    let start = line.find("run_index=")? + "run_index=".len();
    let digits: String = line[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Proxima child log: prefill `step_ns` per run and `decode_ms_per_token` per `run=done`.
fn parse_proxima(text: &str) -> Result<Metrics, HarnessError> {
    let mut prefill_ms: BTreeMap<usize, f64> = BTreeMap::new();
    let mut decode_ms: BTreeMap<usize, f64> = BTreeMap::new();
    let mut gpu_peak: Option<u64> = None;
    let mut gpu_steady: Option<u64> = None;
    for line in text.lines() {
        let Some(run) = run_index_of(line) else {
            continue;
        };
        if line.contains("step_time") && line.contains("phase=prefill") {
            let nanos = number_after(line, "step_ns=", "step_ns")?;
            prefill_ms.insert(run, nanos / 1e6);
        } else if line.contains("run=done") {
            decode_ms.insert(
                run,
                number_after(line, "decode_ms_per_token=", "decode_ms_per_token")?,
            );
            let peak = number_after(line, "gpu_peak_bytes=", "gpu_peak_bytes")?.round() as u64;
            gpu_peak = Some(gpu_peak.map_or(peak, |seen| seen.max(peak)));
            gpu_steady =
                Some(number_after(line, "steady_gpu_bytes=", "steady_gpu_bytes")?.round() as u64);
        }
    }
    let cold = *prefill_ms
        .get(&0)
        .ok_or(HarnessError::MissingLine("run 0 prefill step_time line"))?;
    let run_value = |map: &BTreeMap<usize, f64>, run: usize, what: &'static str| {
        map.get(&run).copied().ok_or(HarnessError::MissingLine(what))
    };
    let warm = vec![
        run_value(&prefill_ms, 1, "run 1 prefill step_time line")?,
        run_value(&prefill_ms, 2, "run 2 prefill step_time line")?,
    ];
    let run0 = run_value(&decode_ms, 0, "run 0 run=done line")?;
    let steady = [
        run_value(&decode_ms, 1, "run 1 run=done line")?,
        run_value(&decode_ms, 2, "run 2 run=done line")?,
    ];
    Ok(Metrics {
        ttft_cold_ms: Some(cold),
        ttft_warm_ms: warm,
        tok_s: steady.iter().map(|millis| 1000.0 / millis).collect(),
        tok_s_run0: Some(1000.0 / run0),
        gpu_peak_bytes: gpu_peak,
        gpu_steady_bytes: gpu_steady,
    })
}

/// Llama stderr: `prompt eval time = X ms` and the `eval time` line's tokens per second.
fn parse_llama(text: &str, kind: Kind) -> Result<Metrics, HarnessError> {
    let prompt_line = text
        .lines()
        .find(|line| line.contains("prompt eval time ="))
        .ok_or(HarnessError::MissingLine("prompt eval time line"))?;
    let eval_line = text
        .lines()
        .find(|line| line.contains("eval time =") && !line.contains("prompt eval time ="))
        .ok_or(HarnessError::MissingLine("eval time line"))?;
    let ttft = number_after(prompt_line, "prompt eval time =", "prompt eval time")?;
    let per_second_end = eval_line
        .find(" tokens per second")
        .ok_or(HarnessError::MissingLine("tokens per second"))?;
    let head = &eval_line[..per_second_end];
    let tok_s = number_after(
        &head[head.rfind(',').map_or(0, |comma| comma + 1)..],
        "",
        "tokens per second",
    )?;
    let mut metrics = Metrics {
        tok_s: vec![tok_s],
        ..Metrics::default()
    };
    match kind {
        Kind::Cold => metrics.ttft_cold_ms = Some(ttft),
        _ => metrics.ttft_warm_ms = vec![ttft],
    }
    Ok(metrics)
}

fn sample_mean_cov(values: &[f64]) -> Option<(f64, Option<f64>)> {
    if values.is_empty() {
        return None;
    }
    let center = mean(values);
    if values.len() < 2 || center == 0.0 {
        return Some((center, None));
    }
    let variance =
        values.iter().map(|value| (value - center).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    Some((center, Some(variance.sqrt() / center * 100.0)))
}

#[derive(Debug, Clone)]
struct Row {
    round: usize,
    cell: Cell,
    load1: f64,
    footprint_bytes: Option<u64>,
    metrics: Metrics,
    exit_code: i32,
    note: String,
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"))
}

fn fmt_bytes(value: Option<u64>) -> String {
    value.map_or_else(|| "-".to_owned(), |bytes| bytes.to_string())
}

fn tsv_line(row: &Row) -> String {
    let sample = |samples: &[f64], index: usize| fmt_opt(samples.get(index).copied());
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        row.round,
        row.cell.engine_name(),
        row.cell.label,
        row.cell.prompt.name(),
        row.cell.kv.name(),
        row.cell.kind.name(),
        row.load1,
        fmt_bytes(row.footprint_bytes),
        fmt_bytes(row.metrics.gpu_peak_bytes),
        fmt_bytes(row.metrics.gpu_steady_bytes),
        fmt_opt(row.metrics.ttft_cold_ms),
        sample(&row.metrics.ttft_warm_ms, 0),
        sample(&row.metrics.ttft_warm_ms, 1),
        fmt_opt(row.metrics.tok_s_run0),
        sample(&row.metrics.tok_s, 0),
        sample(&row.metrics.tok_s, 1),
        row.exit_code,
        row.note.replace(['\t', '\n'], " "),
    )
}

fn stat_cell(values: &[f64]) -> String {
    match sample_mean_cov(values) {
        None => "-".to_owned(),
        Some((center, Some(cov))) => format!("{center:.2} ({cov:.1}%) n={}", values.len()),
        Some((center, None)) => format!("{center:.2} (n/a) n={}", values.len()),
    }
}

fn collect(rows: &[&Row], pick: impl Fn(&Row) -> Vec<f64>) -> Vec<f64> {
    rows.iter()
        .filter(|row| row.exit_code == 0)
        .flat_map(|row| pick(row))
        .collect()
}

fn bytes_to_megabytes(bytes: Option<u64>) -> Vec<f64> {
    bytes.map(|bytes| bytes as f64 / MEGABYTE).into_iter().collect()
}

/// One markdown table per (prompt, kv); one table row per cell; each metric as
/// `mean (CoV%) n=samples`. Warm TTFT and tok/s count runs 1 and 2 as separate samples.
fn scoreboard_markdown(rows: &[Row], cells: &[Cell]) -> String {
    let mut out = String::from("# kv scoreboard\n\nmetric cells are `mean (CoV%) n=samples` over exit-0 processes; CoV is sample sd / mean x 100; MB is 1e6 bytes.\n");
    for prompt in ALL_PROMPTS {
        for kv in ALL_KV {
            let _ = write!(
                out,
                "\n## {} prompt, {}\n\n| engine | label | kind | n | nonzero exits | RAM MB | GPU MB | TTFT cold ms | TTFT warm ms | tok/s | tok/s run 0 |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
                prompt.name(),
                kv.name()
            );
            for cell in cells.iter().filter(|cell| cell.prompt == prompt && cell.kv == kv) {
                let mine: Vec<&Row> = rows.iter().filter(|row| row.cell == *cell).collect();
                let failed = mine.iter().filter(|row| row.exit_code != 0).count();
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                    cell.engine_name(),
                    cell.label,
                    cell.kind.name(),
                    mine.len(),
                    failed,
                    stat_cell(&collect(&mine, |row| bytes_to_megabytes(row.footprint_bytes))),
                    stat_cell(&collect(&mine, |row| bytes_to_megabytes(row.metrics.gpu_peak_bytes))),
                    stat_cell(&collect(&mine, |row| row.metrics.ttft_cold_ms.into_iter().collect())),
                    stat_cell(&collect(&mine, |row| row.metrics.ttft_warm_ms.clone())),
                    stat_cell(&collect(&mine, |row| row.metrics.tok_s.clone())),
                    stat_cell(&collect(&mine, |row| row.metrics.tok_s_run0.into_iter().collect())),
                );
            }
        }
    }
    out
}

struct Config {
    out: PathBuf,
    model: PathBuf,
    llama: PathBuf,
    long_prompt_file: PathBuf,
    proxima: Vec<(String, PathBuf)>,
    rounds: usize,
    time_bin: PathBuf,
}

fn parse_config(arguments: impl IntoIterator<Item = String>) -> Result<Config, HarnessError> {
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    let mut proxima = Vec::new();
    let mut tokens = arguments.into_iter();
    while let Some(flag) = tokens.next() {
        let value = tokens
            .next()
            .ok_or_else(|| HarnessError::MissingValue(flag.clone()))?;
        match flag.as_str() {
            "--prox" => {
                let (label, path) = value
                    .split_once('=')
                    .ok_or_else(|| HarnessError::BadProx(value.clone()))?;
                proxima.push((label.to_owned(), PathBuf::from(path)));
            }
            "--out" | "--model" | "--llama" | "--long-prompt-file" | "--rounds" | "--time-bin" => {
                values.insert(flag, value);
            }
            _ => return Err(HarnessError::UnknownFlag(flag)),
        }
    }
    let required = |flag: &'static str| {
        values
            .get(flag)
            .map(PathBuf::from)
            .ok_or(HarnessError::MissingFlag(flag))
    };
    if proxima.is_empty() {
        return Err(HarnessError::MissingFlag("--prox"));
    }
    let rounds = match values.get("--rounds") {
        None => DEFAULT_ROUNDS,
        Some(text) => text
            .parse::<usize>()
            .ok()
            .filter(|rounds| *rounds > 0)
            .ok_or_else(|| HarnessError::BadRounds(text.clone()))?,
    };
    Ok(Config {
        out: required("--out")?,
        model: required("--model")?,
        llama: required("--llama")?,
        long_prompt_file: required("--long-prompt-file")?,
        proxima,
        rounds,
        time_bin: values
            .get("--time-bin")
            .map_or_else(|| PathBuf::from(DEFAULT_TIME_BIN), PathBuf::from),
    })
}

fn load1() -> f64 {
    let mut loads = [0.0_f64; 3];
    // SAFETY: the buffer holds exactly the 3 doubles getloadavg is told it may write.
    let written = unsafe { libc::getloadavg(loads.as_mut_ptr(), 3) };
    if written >= 1 { loads[0] } else { f64::NAN }
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> HarnessError + '_ {
    move |source| HarnessError::Io {
        path: path.display().to_string(),
        source,
    }
}

fn prompt_text(prompt: Prompt, long_prompt: &str) -> String {
    match prompt {
        Prompt::Short => SHORT_PROMPT.to_owned(),
        Prompt::Long => long_prompt.to_owned(),
    }
}

fn build_command(config: &Config, cell: &Cell, prompt: &str) -> Command {
    let mut command = Command::new(&config.time_bin);
    command.arg("-l");
    match &cell.engine {
        Engine::Proxima { binary } => {
            command.arg(binary);
            command.env("PROXIMA_SPECULATIVE_TYPES", "none");
            command.env("PROXIMA_STEP_TIMES", "1");
            command.env("PROXIMA_MAX_TOKENS", "128");
            command.env("PROXIMA_RUNS", "3");
            for (key, value) in cell.kv.proxima_env() {
                command.env(key, value);
            }
            if cell.prompt == Prompt::Long {
                command.env("PROXIMA_PROMPT", prompt);
            }
        }
        Engine::Llama => {
            command.arg(&config.llama).arg("-m").arg(&config.model);
            command.args(["-p", prompt, "-n", "128", "-ngl", "99", "--temp", "0"]);
            command.args(["--top-k", "1", "-s", "1", "-no-cnv", "--ignore-eos"]);
            command.arg("--no-display-prompt");
            command.args(cell.kv.llama_flags());
            if cell.kind == Kind::Cold {
                command.arg("--no-warmup");
            }
        }
    }
    command
}

fn run_cell(config: &Config, cell: &Cell, prompt: &str, round: usize) -> Row {
    let load = load1();
    let mut row = Row {
        round,
        cell: cell.clone(),
        load1: load,
        footprint_bytes: None,
        metrics: Metrics::default(),
        exit_code: -1,
        note: String::new(),
    };
    let output = match build_command(config, cell, prompt).output() {
        Ok(output) => output,
        Err(source) => {
            row.note = format!("spawn failed: {source}");
            return row;
        }
    };
    row.exit_code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut notes = Vec::new();
    match parse_footprint_bytes(&stderr) {
        Ok(bytes) => row.footprint_bytes = Some(bytes),
        Err(error) => notes.push(error.to_string()),
    }
    let parsed = match cell.engine {
        Engine::Proxima { .. } => parse_proxima(&format!("{stdout}\n{stderr}")),
        Engine::Llama => parse_llama(&stderr, cell.kind),
    };
    match parsed {
        Ok(metrics) => row.metrics = metrics,
        Err(error) => notes.push(error.to_string()),
    }
    row.note = notes.join("; ");
    row
}

fn append_line(path: &Path, line: &str) -> Result<(), HarnessError> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(io_error(path))?;
    writeln!(file, "{line}").map_err(io_error(path))
}

fn run(config: &Config) -> Result<(), HarnessError> {
    fs::create_dir_all(&config.out).map_err(io_error(&config.out))?;
    let long_prompt = fs::read_to_string(&config.long_prompt_file)
        .map_err(io_error(&config.long_prompt_file))?
        .trim_end_matches('\n')
        .to_owned();
    let cells = cell_order(&config.proxima);
    let tsv_path = config.out.join("raw.tsv");
    fs::write(&tsv_path, format!("{TSV_HEADER}\n")).map_err(io_error(&tsv_path))?;
    let mut rows = Vec::new();
    for (round, index) in round_plan(config.rounds, cells.len()) {
        let cell = &cells[index];
        let row = run_cell(config, cell, &prompt_text(cell.prompt, &long_prompt), round);
        let line = tsv_line(&row);
        append_line(&tsv_path, &line)?;
        println!("{line}");
        rows.push(row);
    }
    let table_path = config.out.join("scoreboard.md");
    let table = scoreboard_markdown(&rows, &cells);
    fs::write(&table_path, &table).map_err(io_error(&table_path))?;
    println!("{table}");
    Ok(())
}

fn main() {
    let outcome = parse_config(std::env::args().skip(1)).and_then(|config| run(&config));
    if let Err(error) = outcome {
        eprintln!("kv_scoreboard: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROX_LOG: &str = "\
2026-11-11T16:17:32.639669333Z INFO decode_gbps_baseline: step_time run_index=0 step=0 step_ms=337 step_ns=338091250 phase=prefill
2026-11-11T16:17:32.639670417Z INFO decode_gbps_baseline: step_time run_index=0 step=0 step_ms=0 step_ns=1917 phase=decode
2026-11-11T16:17:33.875203875Z INFO decode_gbps_baseline: decode_gbps_baseline run=done run_index=0 wall_ms=1573.234917 cpu_user_ms=336.2160000000001 cpu_sys_ms=172.93399999999997 cpu_ms=509.1500000000001 cpu_pct=32.36325322418458 tokens_generated=128 prompt_token_count=26 text_hash=16f789b7871d97d1 stopped_by_eos=false decode_ms_per_token=9.73413320472441 ttft_ms=337 gpu_peak_bytes=3375562752 steady_rss_bytes=3766009856 steady_footprint_bytes=342485376 steady_gpu_bytes=3375562752 steady_samples=119
2026-11-11T16:17:33.965175625Z INFO decode_gbps_baseline: step_time run_index=1 step=0 step_ms=89 step_ns=89921292 phase=prefill
2026-11-11T16:17:35.884058750Z INFO decode_gbps_baseline: decode_gbps_baseline run=done run_index=1 wall_ms=2007.6836660000001 cpu_user_ms=636.8980000000001 cpu_sys_ms=202.1980000000001 cpu_ms=839.0960000000002 cpu_pct=41.794233534397854 tokens_generated=128 prompt_token_count=26 text_hash=16f789b7871d97d1 stopped_by_eos=false decode_ms_per_token=15.107745401574805 ttft_ms=89 gpu_peak_bytes=3376611328 steady_rss_bytes=3773202432 steady_footprint_bytes=346155712 steady_gpu_bytes=3375955968 steady_samples=119
2026-11-11T16:17:36.006986042Z INFO decode_gbps_baseline: step_time run_index=2 step=0 step_ms=121 step_ns=122763750 phase=prefill
2026-11-11T16:17:37.894311708Z INFO decode_gbps_baseline: decode_gbps_baseline run=done run_index=2 wall_ms=2008.9052920000001 cpu_user_ms=581.6600000000003 cpu_sys_ms=192.05600000000004 cpu_ms=773.7160000000003 cpu_pct=38.51430941424392 tokens_generated=128 prompt_token_count=26 text_hash=16f789b7871d97d1 stopped_by_eos=false decode_ms_per_token=14.865396 ttft_ms=121 gpu_peak_bytes=3377004544 steady_rss_bytes=3791863808 steady_footprint_bytes=352889536 steady_gpu_bytes=3375693824 steady_samples=119
        7.53 real         2.90 user         1.85 sys
          3792928768  maximum resident set size
           355363520  peak memory footprint
";

    const LLAMA_COLD: &str = "\
0.02.469.352 I common_perf_print:        load time =      76.48 ms
0.02.469.353 I common_perf_print: prompt eval time =      76.32 ms /    27 tokens (    2.83 ms per token,   353.77 tokens per second)
0.02.469.354 I common_perf_print:        eval time =    1137.03 ms /   127 runs   (    8.95 ms per token,   111.69 tokens per second)
0.02.469.354 I common_perf_print:       total time =    1238.60 ms /   154 tokens
           952119104  peak memory footprint
";

    fn close(left: Option<f64>, right: f64) -> bool {
        left.is_some_and(|value| (value - right).abs() < 1e-6)
    }

    #[test]
    fn proxima_cold_is_run0_prefill_and_warm_keeps_runs_1_and_2_as_separate_samples() {
        let metrics = parse_proxima(PROX_LOG).expect("real proxima log parses");
        assert!(close(metrics.ttft_cold_ms, 338.09125));
        assert_eq!(metrics.ttft_warm_ms.len(), 2);
        assert!((metrics.ttft_warm_ms[0] - 89.921292).abs() < 1e-6);
        assert!((metrics.ttft_warm_ms[1] - 122.76375).abs() < 1e-6);
    }

    #[test]
    fn proxima_tok_s_is_1000_over_decode_ms_per_token_per_run() {
        let metrics = parse_proxima(PROX_LOG).expect("real proxima log parses");
        assert!(close(metrics.tok_s_run0, 1000.0 / 9.73413320472441));
        assert_eq!(metrics.tok_s.len(), 2);
        assert!((metrics.tok_s[0] - 1000.0 / 15.107745401574805).abs() < 1e-9);
        assert!((metrics.tok_s[1] - 1000.0 / 14.865396).abs() < 1e-9);
    }

    #[test]
    fn proxima_gpu_peak_is_max_over_runs_and_steady_is_the_last_logged() {
        let metrics = parse_proxima(PROX_LOG).expect("real proxima log parses");
        assert_eq!(metrics.gpu_peak_bytes, Some(3_377_004_544));
        assert_eq!(metrics.gpu_steady_bytes, Some(3_375_693_824));
    }

    #[test]
    fn proxima_log_missing_run_2_is_an_error_not_a_short_sample() {
        let without_run2: String = PROX_LOG
            .lines()
            .filter(|line| !line.contains("run_index=2"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(parse_proxima(&without_run2).is_err());
    }

    #[test]
    fn rows_of_the_same_cell_pool_per_run_samples_and_report_megabytes() {
        let cell = Cell {
            engine: Engine::Proxima {
                binary: PathBuf::from("/bin/a"),
            },
            label: "new".to_owned(),
            prompt: Prompt::Short,
            kv: Kv::F16,
            kind: Kind::Both,
        };
        let metrics = parse_proxima(PROX_LOG).expect("real proxima log parses");
        let rows: Vec<Row> = (1..=3)
            .map(|round| Row {
                round,
                cell: cell.clone(),
                load1: 1.0,
                footprint_bytes: Some(355_363_520),
                metrics: metrics.clone(),
                exit_code: 0,
                note: String::new(),
            })
            .collect();
        let table = scoreboard_markdown(&rows, &[cell]);
        assert!(table.contains("RAM MB | GPU MB"));
        assert!(table.contains("| proxima | new | both | 3 | 0 | 355.36 (0.0%) n=3 | 3377.00 (0.0%) n=3 |"));
        assert!(table.contains("n=6 |"));
        let tsv = tsv_line(&rows[0]);
        assert_eq!(tsv.split('\t').count(), TSV_HEADER.split('\t').count());
    }

    #[test]
    fn proxima_log_without_run_done_lines_is_an_error_not_zero() {
        let truncated: String = PROX_LOG
            .lines()
            .filter(|line| !line.contains("run=done"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(parse_proxima(&truncated).is_err());
        assert!(parse_proxima("panicked at src/main.rs").is_err());
    }

    #[test]
    fn llama_cold_reads_prompt_eval_and_eval_tokens_per_second() {
        let metrics = parse_llama(LLAMA_COLD, Kind::Cold).expect("real llama log parses");
        assert!(close(metrics.ttft_cold_ms, 76.32));
        assert!(metrics.ttft_warm_ms.is_empty());
        assert_eq!(metrics.tok_s, vec![111.69]);
        assert_eq!(metrics.gpu_peak_bytes, None);
    }

    #[test]
    fn llama_warm_fills_the_warm_ttft_slot() {
        let metrics = parse_llama(LLAMA_COLD, Kind::Warm).expect("real llama log parses");
        assert_eq!(metrics.ttft_warm_ms, vec![76.32]);
        assert!(metrics.ttft_cold_ms.is_none());
    }

    #[test]
    fn llama_log_with_no_eval_line_is_an_error_not_zero() {
        let without_eval: String = LLAMA_COLD
            .lines()
            .filter(|line| !line.contains(" eval time ="))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(parse_llama(&without_eval, Kind::Cold).is_err());
        let only_prompt = "0.02 I common_perf_print: prompt eval time =      76.32 ms /    27 tokens";
        assert!(parse_llama(only_prompt, Kind::Cold).is_err());
    }

    #[test]
    fn footprint_line_from_time_l_is_bytes() {
        assert_eq!(parse_footprint_bytes(PROX_LOG).expect("footprint"), 355_363_520);
        assert_eq!(parse_footprint_bytes(LLAMA_COLD).expect("footprint"), 952_119_104);
    }

    #[test]
    fn missing_footprint_line_is_an_error() {
        assert!(parse_footprint_bytes("7.53 real 2.90 user").is_err());
    }

    #[test]
    fn cov_of_8_10_12_is_20_percent_mean_10() {
        let (center, cov) = sample_mean_cov(&[8.0, 10.0, 12.0]).expect("three values");
        assert!((center - 10.0).abs() < 1e-12);
        assert!((cov.expect("n=3") - 20.0).abs() < 1e-9);
    }

    #[test]
    fn cov_needs_two_samples_and_nothing_is_not_zero() {
        assert_eq!(sample_mean_cov(&[5.0]), Some((5.0, None)));
        assert_eq!(sample_mean_cov(&[]), None);
    }

    #[test]
    fn every_round_visits_every_cell_once_and_no_pair_keeps_one_relative_order() {
        let proxima = vec![
            ("new".to_owned(), PathBuf::from("/bin/a")),
            ("old".to_owned(), PathBuf::from("/bin/b")),
        ];
        let cells = cell_order(&proxima);
        assert_eq!(cells.len(), 2 * 3 * (2 + 2));
        let plan = round_plan(5, cells.len());
        assert_eq!(plan.len(), 5 * cells.len());
        let order_of = |round: usize| -> Vec<usize> {
            plan.iter()
                .filter(|(entry, _)| *entry == round)
                .map(|(_, index)| *index)
                .collect()
        };
        for round in 1..=5 {
            let mut sorted = order_of(round);
            sorted.sort_unstable();
            assert_eq!(sorted, (0..cells.len()).collect::<Vec<_>>());
        }
        let position = |round: usize, cell: usize| order_of(round).iter().position(|index| *index == cell);
        for first in 0..cells.len() {
            for second in (first + 1)..cells.len() {
                let before_in_round = |round: usize| position(round, first) < position(round, second);
                let always_same = (1..=5).all(|round| before_in_round(round) == before_in_round(1));
                assert!(!always_same, "cells {first} and {second} keep one relative order");
            }
        }
    }

    #[test]
    fn config_requires_a_proxima_arm_and_rejects_bad_rounds() {
        let base = ["--out", "o", "--model", "m", "--llama", "l", "--long-prompt-file", "p"];
        let mut arguments: Vec<String> = base.iter().map(|text| (*text).to_owned()).collect();
        assert!(parse_config(arguments.clone()).is_err());
        arguments.extend(["--prox", "new=/bin/a", "--rounds", "0"].map(str::to_owned));
        assert!(parse_config(arguments.clone()).is_err());
        arguments.truncate(arguments.len() - 2);
        let config = parse_config(arguments).expect("valid arguments");
        assert_eq!(config.rounds, DEFAULT_ROUNDS);
    }

    #[test]
    fn nonzero_exit_rows_stay_in_the_table() {
        let cell = Cell {
            engine: Engine::Llama,
            label: "llama".to_owned(),
            prompt: Prompt::Short,
            kv: Kv::F16,
            kind: Kind::Cold,
        };
        let row = Row {
            round: 1,
            cell: cell.clone(),
            load1: 1.0,
            footprint_bytes: None,
            metrics: Metrics::default(),
            exit_code: 134,
            note: "log has no prompt eval time line".to_owned(),
        };
        let table = scoreboard_markdown(&[row], &[cell]);
        assert!(table.contains("| llama | llama | cold | 1 | 1 | - | - | - | - | - | - |"));
    }
}
