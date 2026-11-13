//! Interleaved KV-cache scoreboard: proxima builds against `llama-server`, with a parity record
//! on every row.
//!
//! One driver, no scripts. Cells are {each proxima binary, llama-server} x {short prompt, long
//! prompt file} x {F32, F16, BF16, Q4_0-K + F16-V}; the Q4_0-K row has no llama arm because llama
//! flash attention needs K type = V type (`ggml-metal-device.m:1754`), so it is proxima-only.
//! Every round visits every cell once, alternating forward and reverse, so drift lands on every
//! cell alike. Every child runs under `/usr/bin/time -l`; its `peak memory footprint` line is the
//! RAM read.
//!
//! Llama arm: one `llama-server` process per (prompt, KV config, round) with `-fa on`, `-c` set
//! from the same rule as proxima's `kv_extent` (`generate/load_model.rs`: positions needed =
//! prompt ids + max tokens, rounded up to the 32-token bucket, capped at the context length),
//! `-ctk/-ctv`, `--no-warmup`, `--seed 1`. Request 1 is the cold TTFT; request 2 is the warm TTFT
//! and the decode. Both send the prompt as the token-id array proxima's tokenizer produced
//! (`encode_with_bos_eos` under `wants_bos`, mirrored here because `wants_bos` is crate-private),
//! `n_predict` 128, temperature 0, `top_k` 1, seed 1, `cache_prompt` false, `ignore_eos` true,
//! `return_tokens` true. Metrics come from the response `timings` object.
//!
//! The HTTP client below is a minimal copy of the one in `decode_arms.rs`; examples cannot share
//! code.
//!
//! Parity record (I1..I11 of the kv-epic architecture, part A, plus I12 incumbent build: the
//! server's startup `build NNNNN (commit)` line must equal `--llama-commit`): a row whose record has any
//! unequal invariant is INVALID, is listed in `scoreboard.md` with the failing invariants, and is
//! excluded from every mean. Quiet gate: before each process, no `cargo`/`rustc`/`nextest`/`kv_scoreboard`
//! /`llama-server` process other than this one, and load1 below the limit for the hold window; if
//! it never clears within the wait limit the row is not run and is INVALID on I10. Run the built
//! binary directly, not under `cargo run`, whose own process is a gate hit.
//!
//! llama-inproc arm (`--llama-inproc PATH/inproc_arm`, built from `tools/llama-arm`, never a
//! dependency of this crate): one process per cell, the pinned libllama taken from the server's own
//! install directory behind the arm's sha256 + build-number gate, the same prompt ids, `n_predict`,
//! KV types and padded `-c`. Proxima's `step_time` lines and the arm's are both parsed into
//! `TokenRecord`s and reduced by the one `decode_span_ms_per_token`; the server's
//! `timings.predicted_per_token_ms` stays its own `server_predicted_per_token_ms` column, so the
//! difference between the two is the server's overhead. `--prompts` / `--kvs` select a subset of
//! cells; `--quiet-gate off` runs without waiting and marks every row INVALID on I10.
//!
//! Stage-matrix mode (`--stages DIR`, PROXNUM-184): every pipeline stage of both engines on
//! `DIR/wp_{84,510,2024,4069}.txt`, one TSV row per (round, engine, prompt, metric) in
//! `stages.tsv` and an owner-ordered markdown table in `stages.md` (tokenization, prefill,
//! decode, attention, kv, detokenization, ete, then model load / first-request compile / RSS /
//! GPU peak). The `Metric` consts carry each number's definition into the TSV and the markdown.
//! Cells a side cannot produce are `unavailable` with the reason, never filled: llama has no
//! per-op timer, so its attention cells are unavailable and `--fa-off-arm on` adds the
//! `llama-fa-delta` rows (flash attention off minus on) as the only proxy. Uses the first
//! `--prox`, `--stage-kv` (default F16) and `--rounds`.
//!
//! ```sh
//! kv_scoreboard --stages ~/.cache/kvgate/proxnum97 --out DIR --model BLOB --llama PATH/llama-server \
//!   --llama-commit ba3c05f71 --prox new=PATH_A --rounds 3 --fa-off-arm on
//! ```
//!
//! Output directory gets `raw.tsv` (one row per process with the parity columns), `logs/` (each
//! child's stderr) and `scoreboard.md`.
//!
//! ```sh
//! kv_scoreboard --out DIR --model BLOB --llama PATH/llama-server --llama-commit ba3c05f71 --long-prompt-file FILE \
//!   --prox new=PATH_A --prox old=PATH_B --rounds 5
//! ```

use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{Read, Write as _};
use std::net::{SocketAddr, TcpStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, UNIX_EPOCH};

use memmap2::MmapOptions;
use proxima_gguf::parse_complete;
use proxima_tokenizer::vocab::Vocab;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use thiserror::Error;

const SHORT_PROMPT: &str = "<|turn>user\nWhich of these is smaller in size: a hippopotamus or a large office building?<turn|>\n<|turn>model\n";
const DEFAULT_ROUNDS: usize = 5;
const DEFAULT_TIME_BIN: &str = "/usr/bin/time";
const DEFAULT_LOAD_LIMIT: f64 = 4.0;
const DEFAULT_HOLD_SECS: u64 = 30;
const DEFAULT_WAIT_LIMIT_SECS: u64 = 900;
const DEFAULT_CONTEXT_LENGTH: usize = 131_072;
const DEFAULT_KV_BUCKET: usize = 32;
const GATE_POLL: Duration = Duration::from_secs(2);
const STOP_GRACE: Duration = Duration::from_secs(2);
const MAX_TOKENS: usize = 128;
const LLAMA_KV_PAD: usize = 256;
const BASE_PORT: u16 = 18_200;
const BUSY_PATTERN: &str = "cargo|rustc|nextest|kv_scoreboard|llama-server";
const INVARIANT_COUNT: usize = 12;
const INVARIANT_NAMES: [&str; INVARIANT_COUNT] = [
    "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12",
];
const PROXIMA_SPAN_DEFINITION: &str =
    "PROXIMA ttft=run=done ttft_ms_precise (float ms from loop start); decode=decode_span_ms_per_token over step_time records (sum decode elapsed_ns, index>=1, over those records)";
const PROXIMA_LEGACY_DEFINITION: &str =
    "LEGACY ttft=prefill step_ns incl tokenize+setup; decode=decode_ms_per_token (wall-prefill)/(N-1) incl polling";
const LLAMA_METRIC_DEFINITION: &str =
    "SERVER ttft=timings.prompt_ms of one llama_decode prefill; decode=timings.predicted_per_second (slot t_gen, includes server overhead); server_predicted_per_token_ms=timings.predicted_per_token_ms";
const INPROC_METRIC_DEFINITION: &str =
    "INPROC ttft=prefill record elapsed_ns (llama_decode+llama_synchronize); decode=decode_span_ms_per_token over step_time records (same reducer as PROXIMA, no server)";
/// the incumbent build number the in-process arm's load gate must report (I12).
const PINNED_LLAMA_BUILD: u32 = 11_671;
const TSV_HEADER: &str = "round\tengine\tlabel\tprompt\tkv\tload1\tfootprint_bytes\tgpu_peak_bytes\tgpu_steady_bytes\tttft_cold_ms\tttft_warm_run1_ms\tttft_warm_run2_ms\ttok_s_run0\ttok_s_run1\ttok_s_run2\texit_code\tllama_kv_buffer_mib\tllama_n_ctx\tcapacity\tI1\tI2\tI3\tI4\tI5\tI6\tI7\tI8\tI9\tI10\tI11\tI12\tvalid\tfirst_divergence\tgen_llama\tgen_proxima\tmodel_sha256\tbinary_sha256\tfeatures\tllama_build\tlegacy_ttft_cold_ms\tlegacy_ttft_warm_run1_ms\tlegacy_ttft_warm_run2_ms\tlegacy_tok_s_run0\tlegacy_tok_s_run1\tlegacy_tok_s_run2\tdecode_ms_per_token_run1\tserver_predicted_per_token_ms\tmetric_definition\tquiet_gate\tnote";

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
    #[error("{flag} expects {expected}, got {got}")]
    BadValue {
        flag: &'static str,
        expected: &'static str,
        got: String,
    },
    #[error("log has no {0}")]
    MissingLine(&'static str),
    #[error("could not parse {field} from line: {line}")]
    BadNumber { field: &'static str, line: String },
    #[error("llama-server response: {0}")]
    BadResponse(String),
    #[error("llama-server: {0}")]
    Server(String),
    #[error("{engine} counted {reported} prompt tokens, the tokenizer sent {sent}: it ran another model or prompt, not a measurement")]
    PromptCountMismatch { engine: &'static str, reported: usize, sent: usize },
    #[error("tokenizer: {0}")]
    Tokenize(String),
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

    fn parse(text: &str) -> Option<Prompt> {
        ALL_PROMPTS.into_iter().find(|prompt| prompt.name() == text)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kv {
    F32,
    F16,
    Bf16,
    Q4K,
}

impl Kv {
    fn name(self) -> &'static str {
        match self {
            Kv::F32 => "F32",
            Kv::F16 => "F16",
            Kv::Bf16 => "BF16",
            Kv::Q4K => "Q4K",
        }
    }

    fn parse(text: &str) -> Option<Kv> {
        ALL_KV.into_iter().find(|kv| kv.name() == text)
    }

    fn proxima_env(self) -> Vec<(&'static str, &'static str)> {
        match self {
            Kv::F32 => vec![("PROXIMA_KV_CACHE_TYPE", "f32")],
            Kv::F16 => vec![("PROXIMA_KV_CACHE_TYPE", "f16")],
            Kv::Bf16 => vec![("PROXIMA_KV_CACHE_TYPE", "bf16")],
            Kv::Q4K => vec![
                ("PROXIMA_KV_CACHE_TYPE", "f16"),
                ("PROXIMA_KV_CACHE_KEY_TYPE", "q4_0"),
            ],
        }
    }

    /// llama cache types as (K, V); `None` where llama flash attention has no such pair.
    fn llama_types(self) -> Option<(&'static str, &'static str)> {
        match self {
            Kv::F32 => Some(("f32", "f32")),
            Kv::F16 => Some(("f16", "f16")),
            Kv::Bf16 => Some(("bf16", "bf16")),
            Kv::Q4K => None,
        }
    }
}

const ALL_KV: [Kv; 4] = [Kv::F32, Kv::F16, Kv::Bf16, Kv::Q4K];
const ALL_PROMPTS: [Prompt; 2] = [Prompt::Short, Prompt::Long];

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Engine {
    Proxima { binary: PathBuf },
    /// `llama-server`, the HTTP arm
    Llama,
    /// the `llama-arm` `inproc_arm` bin: the same libllama build, no server
    LlamaInproc { binary: PathBuf },
}

impl Engine {
    /// the arms whose rows are compared in an invariant: proxima and the server meet in
    /// both directions; the in-process arm is judged against the server only.
    fn pairs_with(&self, other: &Engine) -> bool {
        matches!(
            (self, other),
            (Engine::Proxima { .. } | Engine::LlamaInproc { .. }, Engine::Llama) | (Engine::Llama, Engine::Proxima { .. })
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Cell {
    engine: Engine,
    label: String,
    prompt: Prompt,
    kv: Kv,
}

impl Cell {
    fn engine_name(&self) -> &'static str {
        match self.engine {
            Engine::Proxima { .. } => "proxima",
            Engine::Llama => "llama",
            Engine::LlamaInproc { .. } => "llama-inproc",
        }
    }

    fn group(&self) -> (Prompt, Kv) {
        (self.prompt, self.kv)
    }
}

/// Cell order is fixed: prompt, then kv, then each proxima binary, then llama-server and the
/// in-process llama arm when llama has an arm for that KV pair.
fn cell_order(proxima: &[(String, PathBuf)], inproc: Option<&Path>) -> Vec<Cell> {
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
                });
            }
            if kv.llama_types().is_some() {
                cells.push(Cell {
                    engine: Engine::Llama,
                    label: "llama".to_owned(),
                    prompt,
                    kv,
                });
                cells.extend(inproc.map(|binary| Cell {
                    engine: Engine::LlamaInproc { binary: binary.to_path_buf() },
                    label: "llama-inproc".to_owned(),
                    prompt,
                    kv,
                }));
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

/// The KV capacity proxima allocates for a prompt: `kv_extent(positions_needed, context, bucket)`
/// from `generate/load_model.rs`, with `positions_needed = prompt ids + max tokens` and no
/// speculative draft (`decode.rs:3718-3731`). This becomes llama's `-c`.
fn kv_capacity(prompt_ids: usize, max_tokens: usize, bucket: usize, context: usize) -> usize {
    (prompt_ids + max_tokens)
        .div_ceil(bucket)
        .saturating_mul(bucket)
        .min(context)
}

/// llama pads its KV cells to a multiple of 256, so its `n_ctx_seq` is the nearest value it can
/// report to a requested `-c`.
fn llama_padded_capacity(capacity: usize) -> usize {
    capacity.div_ceil(LLAMA_KV_PAD) * LLAMA_KV_PAD
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

fn bad_number(line: &str, field: &'static str) -> HarnessError {
    HarnessError::BadNumber {
        field,
        line: line.to_owned(),
    }
}

/// The numeric token right after `key` in `line`.
fn numeric_token<'a>(line: &'a str, key: &str, field: &'static str) -> Result<&'a str, HarnessError> {
    let start = line.find(key).ok_or_else(|| bad_number(line, field))? + key.len();
    let rest = line[start..].trim_start();
    let end = rest
        .find(|character: char| {
            !(character.is_ascii_digit() || matches!(character, '.' | '-' | 'e' | 'E' | '+'))
        })
        .unwrap_or(rest.len());
    Ok(&rest[..end])
}

fn number_after(line: &str, key: &str, field: &'static str) -> Result<f64, HarnessError> {
    numeric_token(line, key, field)?.parse().map_err(|_| bad_number(line, field))
}

/// A whole-number field (byte or token counts) read as an integer, never through a float.
fn count_after(line: &str, key: &str, field: &'static str) -> Result<u64, HarnessError> {
    numeric_token(line, key, field)?.parse().map_err(|_| bad_number(line, field))
}

/// `value / 1e6` as float: nanoseconds to milliseconds, bytes to megabytes.
fn per_million(value: u64) -> f64 {
    Duration::from_nanos(value).as_secs_f64() * 1e3
}

fn count_as_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// Exact below 2^53: the high and low u32 halves recombined, so byte counts above 4 GiB keep their value.
fn bytes_as_f64(bytes: u64) -> f64 {
    let high = u32::try_from(bytes >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(bytes & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
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
    values.iter().sum::<f64>() / count_as_f64(values.len())
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
            let peak = count_after(line, "gpu_peak_bytes=", "gpu_peak_bytes")?;
            gpu_peak = Some(gpu_peak.map_or(peak, |seen| seen.max(peak)));
            gpu_steady = Some(count_after(line, "steady_gpu_bytes=", "steady_gpu_bytes")?);
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpanPhase {
    Prefill,
    Decode,
}

/// One generated token as an arm reports it: its step index, the phase that produced it, and the
/// wall ns of the evaluation behind it. Proxima's `step_time` lines and the `inproc_arm` bin's
/// lines are both this record; nothing downstream knows which arm wrote them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TokenRecord {
    index: usize,
    phase: SpanPhase,
    elapsed_ns: u64,
}

impl TokenRecord {
    /// llama-server's `predicted_ms` span: one `llama_decode` per generated token after the first.
    /// The prefill record and decode index 0 (the prompt's own evaluation) are outside it.
    fn counts_toward_decode_span(self) -> bool {
        self.phase == SpanPhase::Decode && self.index >= 1
    }
}

/// THE decode-span definition, written once: summed `elapsed_ns` of the records
/// `counts_toward_decode_span` admits, over their count, in float ms. `None` when no record is
/// admitted, because a run with no decode steps has no ms/token and must not read as 0.
fn decode_span_ms_per_token(records: &[TokenRecord]) -> Option<f64> {
    let (total_ns, count) = records
        .iter()
        .filter(|record| record.counts_toward_decode_span())
        .fold((0_u64, 0_usize), |(total, count), record| (total + record.elapsed_ns, count + 1));
    (count > 0).then(|| per_million(total_ns) / count_as_f64(count))
}

fn prefill_ms(records: &[TokenRecord]) -> Option<f64> {
    records
        .iter()
        .find(|record| record.phase == SpanPhase::Prefill)
        .map(|record| per_million(record.elapsed_ns))
}

fn field_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split_whitespace().find_map(|token| token.strip_prefix(key))
}

/// Every `step_time run_index= step= step_ns= phase=` line of an arm's log, grouped by run.
fn parse_token_records(text: &str) -> BTreeMap<usize, Vec<TokenRecord>> {
    let mut runs: BTreeMap<usize, Vec<TokenRecord>> = BTreeMap::new();
    for line in text.lines().filter(|line| line.contains("step_time")) {
        let parsed = (|| {
            let run = field_value(line, "run_index=")?.parse().ok()?;
            let index = field_value(line, "step=")?.parse().ok()?;
            let elapsed_ns = field_value(line, "step_ns=")?.parse().ok()?;
            let phase = match field_value(line, "phase=")? {
                "prefill" => SpanPhase::Prefill,
                "decode" => SpanPhase::Decode,
                _ => return None,
            };
            Some((run, TokenRecord { index, phase, elapsed_ns }))
        })();
        if let Some((run, record)) = parsed {
            runs.entry(run).or_default().push(record);
        }
    }
    runs
}

/// Decode ms/token for runs 0, 1 and 2 from the shared reducer; `None` unless all three reduce.
fn decode_by_run(records: &BTreeMap<usize, Vec<TokenRecord>>) -> Option<[f64; 3]> {
    let run = |index: usize| decode_span_ms_per_token(records.get(&index)?);
    Some([run(0)?, run(1)?, run(2)?])
}

fn tok_s_from(decode_ms: [f64; 3]) -> (Option<f64>, Vec<f64>) {
    (Some(1000.0 / decode_ms[0]), vec![1000.0 / decode_ms[1], 1000.0 / decode_ms[2]])
}

/// Proxima's span metrics: `ttft_ms_precise` (float ms from loop start) from each `run=done`
/// line, decode from the shared reducer over the run's `step_time` records. `Some` only when runs
/// 0, 1 and 2 each have both; GPU columns are carried over from the legacy parse.
fn parse_proxima_span(text: &str, legacy: &Metrics) -> Option<Metrics> {
    let mut ttft: BTreeMap<usize, f64> = BTreeMap::new();
    for line in text.lines().filter(|line| line.contains("run=done")) {
        let run = run_index_of(line)?;
        ttft.insert(run, number_after(line, "ttft_ms_precise=", "ttft_ms_precise").ok()?);
    }
    let (tok_s_run0, tok_s) = tok_s_from(decode_by_run(&parse_token_records(text))?);
    Some(Metrics {
        ttft_cold_ms: Some(*ttft.get(&0)?),
        ttft_warm_ms: vec![*ttft.get(&1)?, *ttft.get(&2)?],
        tok_s,
        tok_s_run0,
        gpu_peak_bytes: legacy.gpu_peak_bytes,
        gpu_steady_bytes: legacy.gpu_steady_bytes,
    })
}

/// The in-process llama arm's metrics: prefill record as TTFT, decode from the same reducer.
fn parse_inproc_span(text: &str) -> Option<Metrics> {
    let records = parse_token_records(text);
    let prefill = |run: usize| prefill_ms(records.get(&run)?);
    let (tok_s_run0, tok_s) = tok_s_from(decode_by_run(&records)?);
    Some(Metrics {
        ttft_cold_ms: Some(prefill(0)?),
        ttft_warm_ms: vec![prefill(1)?, prefill(2)?],
        tok_s,
        tok_s_run0,
        ..Metrics::default()
    })
}

/// What the in-process arm's `gate` line reports: the load gate's identity and the settings the
/// context was created with.
#[derive(Debug, Clone, PartialEq)]
struct InprocGate {
    sha256: String,
    build: u32,
    type_k: String,
    type_v: String,
    flash_attn: bool,
    /// the `n_ctx` the arm requested; the bindings expose no read-back of llama's own value
    n_ctx_requested: usize,
}

fn parse_inproc_gate(text: &str) -> Option<InprocGate> {
    let line = text.lines().find(|line| line.contains("llama_arm gate "))?;
    let value = |key: &str| field_value(line, key);
    Some(InprocGate {
        sha256: value("sha256=")?.to_owned(),
        build: value("build=")?.parse().ok()?,
        type_k: value("type_k=")?.to_owned(),
        type_v: value("type_v=")?.to_owned(),
        flash_attn: value("flash_attn=")? == "enabled",
        n_ctx_requested: value("n_ctx=")?.parse().ok()?,
    })
}

/// What the parity record needs from a proxima child: the last run's generated ids (from the
/// `token_ids ... ids=[...]` line), its `tokens_generated`, and the `prompt_token_count` it
/// displays.
#[derive(Debug, Clone, PartialEq)]
struct ProximaDetail {
    ids: Vec<u64>,
    tokens_generated: usize,
    prompt_token_count: usize,
}

fn parse_id_list(text: &str) -> Vec<u64> {
    text.trim_matches(|character| character == '[' || character == ']')
        .split(',')
        .filter_map(|token| token.trim().parse().ok())
        .collect()
}

fn parse_proxima_detail(text: &str) -> Result<ProximaDetail, HarnessError> {
    let mut ids: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    let mut done: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for line in text.lines() {
        let Some(run) = run_index_of(line) else {
            continue;
        };
        if let Some(start) = line.find(" ids=[") {
            ids.insert(run, parse_id_list(&line[start + " ids=".len()..]));
        } else if line.contains("run=done") {
            let generated = usize::try_from(count_after(line, "tokens_generated=", "tokens_generated")?)
                .map_err(|_| bad_number(line, "tokens_generated"))?;
            let prompt = usize::try_from(count_after(line, "prompt_token_count=", "prompt_token_count")?)
                .map_err(|_| bad_number(line, "prompt_token_count"))?;
            done.insert(run, (generated, prompt));
        }
    }
    let (run, ids) = ids
        .into_iter()
        .next_back()
        .ok_or(HarnessError::MissingLine("token_ids ids= line"))?;
    let (tokens_generated, prompt_token_count) = done
        .get(&run)
        .copied()
        .ok_or(HarnessError::MissingLine("run=done line for the ids run"))?;
    Ok(ProximaDetail {
        ids,
        tokens_generated,
        prompt_token_count,
    })
}

/// One `/completion` response (non-streaming): the `timings` object, the returned token ids
/// (`return_tokens`), the echoed sampler settings and the served model path.
/// Server source `ba3c05f71`: `server_task_result_cmpl_final::to_json_non_oaicompat`
/// (`server-task.cpp:333-350`) and `server_slot_stats::to_json` (`server-common.cpp:83-100`).
#[derive(Debug, Clone, PartialEq)]
struct LlamaResponse {
    ids: Vec<u64>,
    model: String,
    prompt_n: usize,
    cache_n: usize,
    prompt_ms: f64,
    predicted_n: usize,
    predicted_ms: f64,
    predicted_per_second: f64,
    predicted_per_token_ms: f64,
    top_k: i64,
    temperature: f64,
    seed: i64,
}

fn parse_llama_response(body: &str) -> Result<LlamaResponse, HarnessError> {
    let bad = |what: &str| HarnessError::BadResponse(format!("missing {what}"));
    let value: Value =
        serde_json::from_str(body).map_err(|source| HarnessError::BadResponse(source.to_string()))?;
    let timings = value.get("timings").ok_or_else(|| bad("timings"))?;
    let float = |object: &Value, key: &str| object.get(key).and_then(Value::as_f64).ok_or_else(|| bad(key));
    let count = |object: &Value, key: &str| {
        object
            .get(key)
            .and_then(Value::as_u64)
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| bad(key))
    };
    let ids = value
        .get("tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("tokens"))?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    let settings = value.get("generation_settings").ok_or_else(|| bad("generation_settings"))?;
    Ok(LlamaResponse {
        ids,
        model: value
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("model"))?
            .to_owned(),
        prompt_n: count(timings, "prompt_n")?,
        cache_n: count(timings, "cache_n")?,
        prompt_ms: float(timings, "prompt_ms")?,
        predicted_n: count(timings, "predicted_n")?,
        predicted_ms: float(timings, "predicted_ms")?,
        predicted_per_second: float(timings, "predicted_per_second")?,
        predicted_per_token_ms: float(timings, "predicted_per_token_ms")?,
        top_k: settings.get("top_k").and_then(Value::as_i64).ok_or_else(|| bad("top_k"))?,
        temperature: float(settings, "temperature")?,
        seed: settings.get("seed").and_then(Value::as_i64).ok_or_else(|| bad("seed"))?,
    })
}

/// The lines of llama-server's startup log the invariants read (needs `-lv 4`).
#[derive(Debug, Clone, Default, PartialEq)]
struct LlamaLog {
    n_ctx_seq: Option<usize>,
    flash_attn: Option<bool>,
    kv_buffer_mib: f64,
    /// (K, V) types from each `llama_kv_cache: size = ... K (x): ... V (y):` line
    kv_types: Vec<(String, String)>,
    /// (build number, commit) from `common_params_print_info: build 11671 (ba3c05f71)`
    build: Option<(u32, String)>,
    /// ids from the `print_info: EOG token = 106 '<turn|>'` lines; llama's `--ignore-eos`
    /// masks exactly this set (`common.cpp:1343-1355`)
    eog_ids: Vec<u64>,
}

fn parse_build(line: &str) -> Option<(u32, String)> {
    let rest = line.split("common_params_print_info: build ").nth(1)?;
    let (number, tail) = rest.split_once(" (")?;
    Some((number.trim().parse().ok()?, tail.split(')').next()?.to_owned()))
}

fn type_in_parens(line: &str, marker: &str) -> Option<String> {
    let start = line.find(marker)? + marker.len();
    let rest = &line[start..];
    Some(rest[..rest.find(')')?].to_owned())
}

fn parse_llama_log(text: &str) -> LlamaLog {
    let mut log = LlamaLog::default();
    for line in text.lines() {
        if let Some(value) = line.split("llama_context: n_ctx_seq").nth(1).filter(|_| line.contains("= ")) {
            log.n_ctx_seq = value.split('=').nth(1).and_then(|digits| digits.trim().parse().ok());
        } else if line.contains("common_params_print_info: build ") {
            log.build = parse_build(line);
        } else if line.contains("print_info: EOG token") {
            let id: Option<u64> =
                line.split_once('=').and_then(|(_, rest)| rest.split_whitespace().next()?.parse().ok());
            log.eog_ids.extend(id);
        } else if line.contains("llama_context: flash_attn") {
            log.flash_attn = Some(line.trim_end().ends_with("enabled"));
        } else if line.contains("KV buffer size =") {
            log.kv_buffer_mib += number_after(line, "KV buffer size =", "KV buffer size").unwrap_or(0.0);
        } else if let (Some(key), Some(value)) = (type_in_parens(line, "K ("), type_in_parens(line, "V (")) {
            log.kv_types.push((key, value));
        }
    }
    log
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
        values.iter().map(|value| (value - center).powi(2)).sum::<f64>() / count_as_f64(values.len() - 1);
    Some((center, Some(variance.sqrt() / center * 100.0)))
}

/// Quiet-gate configuration; every field is a flag.
#[derive(Debug, Clone, Copy, PartialEq)]
struct GateConfig {
    /// load1 must stay below this
    load_limit: f64,
    /// seconds of continuous quiet required before a process may start
    hold_secs: u64,
    /// seconds to wait for quiet before the row is refused
    wait_limit_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
enum GateStep {
    Clear,
    Waiting,
    GaveUp(String),
}

/// Quiet-gate state: when the current quiet stretch began and when waiting began (both seconds).
#[derive(Debug, Clone, Copy, PartialEq)]
struct QuietState {
    started: u64,
    quiet_since: Option<u64>,
}

/// One gate decision from one observation. Pure, so the policy is testable on fixed inputs.
fn gate_step(
    state: &mut QuietState,
    now: u64,
    load: f64,
    busy: &[String],
    config: &GateConfig,
) -> GateStep {
    let hot = !load.is_finite() || load >= config.load_limit;
    if hot || !busy.is_empty() {
        state.quiet_since = None;
    } else {
        let since = *state.quiet_since.get_or_insert(now);
        if now - since >= config.hold_secs {
            return GateStep::Clear;
        }
    }
    if now - state.started >= config.wait_limit_secs {
        let reason = if busy.is_empty() {
            format!("load1 {load:.2} did not hold below {} for {} s", config.load_limit, config.hold_secs)
        } else {
            format!("busy: {}", busy.join(" | "))
        };
        return GateStep::GaveUp(reason);
    }
    GateStep::Waiting
}

/// Lines of `pgrep -lf` output that are other people's work: everything except this process,
/// its parent and the pgrep itself.
fn busy_from_pgrep(output: &str, own_pid: u32, parent_pid: u32) -> Vec<String> {
    output
        .lines()
        .filter(|line| {
            let pid = line.split_whitespace().next().and_then(|digits| digits.parse::<u32>().ok());
            pid.is_some_and(|pid| pid != own_pid && pid != parent_pid) && !line.contains("pgrep")
        })
        .map(|line| line.chars().take(120).collect())
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
enum GateOutcome {
    Clear { waited_secs: u64 },
    Refused(String),
    /// `--quiet-gate off`: the process ran without waiting; I10 fails, so the row is INVALID
    Skipped,
}

impl GateOutcome {
    fn text(&self) -> String {
        match self {
            GateOutcome::Clear { waited_secs } => format!("clear after {waited_secs}s"),
            GateOutcome::Refused(reason) => format!("refused: {reason}"),
            GateOutcome::Skipped => "skipped: --quiet-gate off".to_owned(),
        }
    }

    fn may_run(&self) -> bool {
        !matches!(self, GateOutcome::Refused(_))
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Check {
    Held,
    NotApplicable(String),
    Failed(String),
}

impl Check {
    fn text(&self) -> String {
        match self {
            Check::Held => "ok".to_owned(),
            Check::NotApplicable(reason) => format!("n/a:{reason}"),
            Check::Failed(reason) => format!("FAIL:{reason}"),
        }
    }

    fn require(held: bool, failure: impl FnOnce() -> String) -> Check {
        if held { Check::Held } else { Check::Failed(failure()) }
    }
}

/// The parity record: one `Check` per invariant I1..I11.
#[derive(Debug, Clone, PartialEq)]
struct Parity {
    checks: [Check; INVARIANT_COUNT],
}

impl Parity {
    fn failing(&self) -> Vec<String> {
        self.checks
            .iter()
            .zip(INVARIANT_NAMES)
            .filter_map(|(check, name)| match check {
                Check::Failed(reason) => Some(format!("{name} {reason}")),
                _ => None,
            })
            .collect()
    }

    fn valid(&self) -> bool {
        self.failing().is_empty()
    }

    fn unevaluated() -> Parity {
        Parity {
            checks: std::array::from_fn(|_| Check::Failed("not evaluated".to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LlamaEvidence {
    cold: LlamaResponse,
    warm: LlamaResponse,
    log: LlamaLog,
}

/// What an `inproc_arm` process reported: its ids and counts (same lines proxima prints) and the
/// load gate it ran behind.
#[derive(Debug, Clone, PartialEq)]
struct InprocEvidence {
    detail: ProximaDetail,
    gate: InprocGate,
}

#[derive(Debug, Clone, PartialEq)]
enum Evidence {
    None,
    Proxima(ProximaDetail),
    Llama(Box<LlamaEvidence>),
    Inproc(Box<InprocEvidence>),
}

fn inproc_evidence(row: Option<&Row>) -> Option<&InprocEvidence> {
    match &row?.evidence {
        Evidence::Inproc(evidence) => Some(evidence),
        _ => None,
    }
}

/// Run-wide facts the parity record compares rows against.
#[derive(Debug, Clone)]
struct Facts {
    model_path: String,
    model_sha256: String,
    load_limit: f64,
    /// commit the llama-server must report, e.g. `ba3c05f71` (the pinned incumbent)
    llama_commit: String,
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
    evidence: Evidence,
    /// proxima only: the legacy columns (`decode_ms_per_token`, prefill `step_ns`), kept apart
    legacy: Option<Metrics>,
    /// whether `metrics` came from the llama-span fields (`ttft_ms_precise`, `decode_step_ms_per_token`)
    span_metrics: bool,
    sent_ids: usize,
    capacity: usize,
    binary_sha256: String,
    debug_build: bool,
    features: String,
    gate: GateOutcome,
    parity: Parity,
}

fn first_divergence(left: &[u64], right: &[u64]) -> Option<usize> {
    let common = left.iter().zip(right).position(|(first, second)| first != second);
    common.or_else(|| (left.len() != right.len()).then(|| left.len().min(right.len())))
}

fn llama_evidence(row: Option<&Row>) -> Option<&LlamaEvidence> {
    match &row?.evidence {
        Evidence::Llama(evidence) => Some(evidence),
        _ => None,
    }
}

fn proxima_detail(row: Option<&Row>) -> Option<&ProximaDetail> {
    match &row?.evidence {
        Evidence::Proxima(detail) => Some(detail),
        _ => None,
    }
}

/// Evaluates I1..I11 for `row`; `peer` is the other engine's row of the same (round, prompt, kv),
/// absent for the proxima-only Q4_0-K row. Pair invariants (I2 to I4, I6, I7, I9) are decided on
/// the pair so both rows of a pair carry the same verdict.
fn evaluate(row: &Row, peer: Option<&Row>, facts: &Facts) -> Parity {
    let (proxima_row, llama_row) = match row.cell.engine {
        Engine::Proxima { .. } => (Some(row), peer),
        Engine::Llama => (peer, Some(row)),
        Engine::LlamaInproc { .. } => return evaluate_inproc(row, peer, facts),
    };
    let llama = llama_evidence(llama_row);
    let proxima = proxima_detail(proxima_row);
    let paired = peer.is_some();
    let no_peer = || Check::NotApplicable("proxima-only row".to_owned());
    let missing = |what: &str| Check::Failed(format!("no {what}"));

    let model_ok = !facts.model_sha256.is_empty()
        && llama.is_none_or(|evidence| evidence.warm.model == facts.model_path);
    let i1 = Check::require(model_ok, || "model sha missing or llama served another path".to_owned());

    let i2 = if let Some(detail) = proxima.filter(|_| row.cell.engine != Engine::Llama) {
        Check::require(detail.prompt_token_count == row.sent_ids, || {
            format!("proxima reports {} prompt ids, {} sent", detail.prompt_token_count, row.sent_ids)
        })
    } else if let Some(evidence) = llama {
        let counted = evidence.cold.prompt_n + evidence.cold.cache_n;
        let warm = evidence.warm.prompt_n + evidence.warm.cache_n;
        Check::require(counted == row.sent_ids && warm == row.sent_ids, || {
            format!("llama counted {counted}/{warm} prompt ids, {} sent", row.sent_ids)
        })
    } else {
        missing("evidence")
    };

    let pair = |judge: &dyn Fn(&ProximaDetail, &LlamaEvidence) -> Check| match (proxima, llama, paired) {
        (_, _, false) => no_peer(),
        (Some(detail), Some(evidence), true) => judge(detail, evidence),
        _ => missing("evidence from both engines"),
    };
    let i3 = pair(&|detail, evidence| {
        if evidence.log.eog_ids.is_empty() {
            return Check::Failed("llama log lists no EOG ids".to_owned());
        }
        if detail.tokens_generated != evidence.warm.predicted_n || detail.tokens_generated != MAX_TOKENS {
            return Check::Failed(format!(
                "proxima {} tokens, llama {}, required {MAX_TOKENS}",
                detail.tokens_generated, evidence.warm.predicted_n
            ));
        }
        let emitted: Vec<u64> = detail.ids.iter().copied().filter(|id| evidence.log.eog_ids.contains(id)).collect();
        Check::require(emitted.is_empty(), || format!("proxima emitted EOG ids {emitted:?}"))
    });
    let i4 = pair(&|detail, evidence| match first_divergence(&detail.ids, &evidence.warm.ids) {
        None => Check::Held,
        Some(position) => Check::Failed(format!("first divergence at {position}")),
    });

    let expected_types = row.cell.kv.llama_types();
    let i5 = match (llama, expected_types) {
        (Some(evidence), Some((key, value))) => Check::require(
            !evidence.log.kv_types.is_empty()
                && evidence.log.kv_types.iter().all(|(k, v)| k == key && v == value),
            || format!("llama KV types {:?}, expected {key}/{value}", evidence.log.kv_types),
        ),
        (Some(_), None) => Check::Failed("llama row for a pair it cannot flash".to_owned()),
        _ => Check::Held,
    };

    let on_pair = |judge: &dyn Fn(&LlamaEvidence) -> Check| {
        if paired { llama.map_or_else(|| missing("llama evidence"), judge) } else { no_peer() }
    };
    let i6 = on_pair(&|evidence| {
        Check::require(evidence.log.flash_attn == Some(true), || "llama flash_attn not enabled".to_owned())
    });
    let i7 = on_pair(&|evidence| {
        let expected = llama_padded_capacity(row.capacity);
        Check::require(evidence.log.n_ctx_seq == Some(expected), || {
            format!("llama n_ctx {:?}, expected {expected} (ours {})", evidence.log.n_ctx_seq, row.capacity)
        })
    });
    let i8 = Check::require(!row.binary_sha256.is_empty() && !row.debug_build, || {
        "binary sha missing or debug build".to_owned()
    });
    let i9 = proxima_row.map_or(Check::Held, |proxima_row| {
        Check::require(proxima_row.span_metrics, || "binary predates llama-span metrics".to_owned())
    });
    let i10 = quiet_check(row, facts);
    let i11 = llama.map_or(Check::Held, |evidence| {
        let warm = &evidence.warm;
        Check::require(warm.top_k == 1 && warm.temperature == 0.0 && warm.seed == 1, || {
            format!("llama sampler top_k {} temperature {} seed {}", warm.top_k, warm.temperature, warm.seed)
        })
    });
    let i12 = on_pair(&|evidence| {
        Check::require(
            evidence.log.build.as_ref().is_some_and(|(_, commit)| *commit == facts.llama_commit),
            || format!("llama build {:?}, required commit {}", evidence.log.build, facts.llama_commit),
        )
    });
    Parity {
        checks: [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12],
    }
}

fn quiet_check(row: &Row, facts: &Facts) -> Check {
    match &row.gate {
        GateOutcome::Clear { .. } => Check::require(row.load1 < facts.load_limit, || {
            format!("load1 {:.2} at launch", row.load1)
        }),
        GateOutcome::Refused(reason) => Check::Failed(reason.clone()),
        GateOutcome::Skipped => Check::Failed("quiet gate skipped".to_owned()),
    }
}

/// I1..I12 for the in-process llama arm. Its identity checks read the arm's own `gate` line (I12:
/// the pinned build number; I8: a recorded sha256, here libllama's) and its context settings (I5,
/// I6, I7); the pair checks (I3 EOG, I4) compare against the same cell's `llama-server` row, whose
/// absence marks them not applicable. The sampler is greedy by construction (I11).
fn evaluate_inproc(row: &Row, server_peer: Option<&Row>, facts: &Facts) -> Parity {
    let evidence = inproc_evidence(Some(row));
    let server = llama_evidence(server_peer);
    let without_peer = || Check::NotApplicable("no llama-server row in this cell".to_owned());
    let judge = |check: &dyn Fn(&InprocEvidence) -> Check| {
        evidence.map_or_else(|| Check::Failed("no llama-arm output".to_owned()), check)
    };
    let i1 = Check::require(!facts.model_sha256.is_empty(), || "model sha missing".to_owned());
    let i2 = judge(&|found| {
        Check::require(found.detail.prompt_token_count == row.sent_ids, || {
            format!("llama-arm reports {} prompt ids, {} sent", found.detail.prompt_token_count, row.sent_ids)
        })
    });
    let i3 = judge(&|found| {
        let emitted: Vec<u64> = server
            .map(|peer| found.detail.ids.iter().copied().filter(|id| peer.log.eog_ids.contains(id)).collect())
            .unwrap_or_default();
        Check::require(found.detail.tokens_generated == MAX_TOKENS && emitted.is_empty(), || {
            format!("{} tokens (required {MAX_TOKENS}), EOG ids emitted {emitted:?}", found.detail.tokens_generated)
        })
    });
    let i4 = match (evidence, server) {
        (Some(found), Some(peer)) => first_divergence(&found.detail.ids, &peer.warm.ids)
            .map_or(Check::Held, |position| Check::Failed(format!("first divergence at {position}"))),
        (None, _) => Check::Failed("no llama-arm output".to_owned()),
        (_, None) => without_peer(),
    };
    let i5 = judge(&|found| match row.cell.kv.llama_types() {
        Some((key, value)) => Check::require(found.gate.type_k == key && found.gate.type_v == value, || {
            format!("llama-arm KV types {}/{}, expected {key}/{value}", found.gate.type_k, found.gate.type_v)
        }),
        None => Check::Failed("llama-inproc row for a pair it cannot flash".to_owned()),
    });
    let i6 = judge(&|found| Check::require(found.gate.flash_attn, || "llama-arm flash_attn not enabled".to_owned()));
    let i7 = judge(&|found| {
        let expected = llama_padded_capacity(row.capacity);
        Check::require(found.gate.n_ctx_requested == expected, || {
            format!("llama-arm n_ctx {}, expected {expected} (ours {})", found.gate.n_ctx_requested, row.capacity)
        })
    });
    let i8 = Check::require(!row.binary_sha256.is_empty() && !row.debug_build, || {
        "gate sha missing or debug llama-arm build".to_owned()
    });
    let i9 = Check::require(row.span_metrics, || "no step_time records for runs 0..2".to_owned());
    let i10 = quiet_check(row, facts);
    let i11 = Check::Held;
    let i12 = judge(&|found| {
        Check::require(found.gate.build == PINNED_LLAMA_BUILD && found.gate.sha256 == row.binary_sha256, || {
            format!("llama-arm gate build {} (required {PINNED_LLAMA_BUILD})", found.gate.build)
        })
    });
    Parity {
        checks: [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12],
    }
}

/// The row an invariant compares `row` against in the same (round, prompt, kv) cell group.
fn find_peer<'a>(rows: &'a [Row], row: &Row) -> Option<&'a Row> {
    rows.iter().find(|other| {
        other.round == row.round
            && other.cell.group() == row.cell.group()
            && row.cell.engine.pairs_with(&other.cell.engine)
    })
}

fn evaluate_all(rows: &mut [Row], facts: &Facts) {
    let parities: Vec<Parity> = rows
        .iter()
        .map(|row| evaluate(row, find_peer(rows, row), facts))
        .collect();
    for (row, parity) in rows.iter_mut().zip(parities) {
        row.parity = parity;
    }
}

fn fmt_opt(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{value:.3}"))
}

fn fmt_bytes(value: Option<u64>) -> String {
    value.map_or_else(|| "-".to_owned(), |bytes| bytes.to_string())
}

fn divergence_text(row: &Row, peer: Option<&Row>) -> String {
    let (ours, server) = match row.cell.engine {
        Engine::Proxima { .. } => (proxima_detail(Some(row)), llama_evidence(peer)),
        Engine::Llama => (proxima_detail(peer), llama_evidence(Some(row))),
        Engine::LlamaInproc { .. } => (inproc_evidence(Some(row)).map(|found| &found.detail), llama_evidence(peer)),
    };
    match (ours, server) {
        (Some(detail), Some(evidence)) => first_divergence(&detail.ids, &evidence.warm.ids)
            .map_or_else(|| "none".to_owned(), |position| position.to_string()),
        _ => "-".to_owned(),
    }
}

/// The llama-side evidence a row's llama columns show: the server's own, or the in-process arm's.
fn llama_columns(row: &Row, peer: Option<&Row>) -> (String, String, String) {
    if let Some(found) = inproc_evidence(Some(row)) {
        let build = format!("build {} (inproc gate)", found.gate.build);
        return (found.detail.tokens_generated.to_string(), build, found.gate.n_ctx_requested.to_string());
    }
    let server = llama_evidence(Some(row)).or_else(|| llama_evidence(peer));
    let generated = server.map_or_else(|| "-".to_owned(), |evidence| evidence.warm.predicted_n.to_string());
    let build = server
        .and_then(|evidence| evidence.log.build.as_ref())
        .map_or_else(|| "-".to_owned(), |(number, commit)| format!("build {number} ({commit})"));
    let n_ctx = llama_evidence(Some(row))
        .and_then(|evidence| evidence.log.n_ctx_seq)
        .map_or_else(|| "-".to_owned(), |ctx| ctx.to_string());
    (generated, build, n_ctx)
}

fn tsv_line(row: &Row, peer: Option<&Row>, facts: &Facts) -> String {
    let sample = |samples: &[f64], index: usize| fmt_opt(samples.get(index).copied());
    let proxima = proxima_detail(Some(row)).or_else(|| proxima_detail(peer));
    let (gen_llama, llama_build, llama_n_ctx) = llama_columns(row, peer);
    let definition = match (&row.cell.engine, row.span_metrics) {
        (Engine::Proxima { .. }, true) => PROXIMA_SPAN_DEFINITION,
        (Engine::Proxima { .. }, false) => PROXIMA_LEGACY_DEFINITION,
        (Engine::Llama, _) => LLAMA_METRIC_DEFINITION,
        (Engine::LlamaInproc { .. }, _) => INPROC_METRIC_DEFINITION,
    };
    let reduced_ms = match row.cell.engine {
        Engine::Llama => None,
        _ if row.span_metrics => row.metrics.tok_s.first().map(|tok_s| 1000.0 / tok_s),
        _ => None,
    };
    let server_ms = llama_evidence(Some(row)).map(|evidence| evidence.warm.predicted_per_token_ms);
    let legacy = row.legacy.clone().unwrap_or_default();
    let checks: Vec<String> = row.parity.checks.iter().map(Check::text).collect();
    let fields = [
        row.round.to_string(),
        row.cell.engine_name().to_owned(),
        row.cell.label.clone(),
        row.cell.prompt.name().to_owned(),
        row.cell.kv.name().to_owned(),
        format!("{:.2}", row.load1),
        fmt_bytes(row.footprint_bytes),
        fmt_bytes(row.metrics.gpu_peak_bytes),
        fmt_bytes(row.metrics.gpu_steady_bytes),
        fmt_opt(row.metrics.ttft_cold_ms),
        sample(&row.metrics.ttft_warm_ms, 0),
        sample(&row.metrics.ttft_warm_ms, 1),
        fmt_opt(row.metrics.tok_s_run0),
        sample(&row.metrics.tok_s, 0),
        sample(&row.metrics.tok_s, 1),
        row.exit_code.to_string(),
        llama_evidence(Some(row)).map_or_else(|| "-".to_owned(), |evidence| format!("{:.2}", evidence.log.kv_buffer_mib)),
        llama_n_ctx,
        row.capacity.to_string(),
        checks[0].clone(),
        checks[1].clone(),
        checks[2].clone(),
        checks[3].clone(),
        checks[4].clone(),
        checks[5].clone(),
        checks[6].clone(),
        checks[7].clone(),
        checks[8].clone(),
        checks[9].clone(),
        checks[10].clone(),
        checks[11].clone(),
        if row.parity.valid() { "valid" } else { "INVALID" }.to_owned(),
        divergence_text(row, peer),
        gen_llama,
        proxima.map_or_else(|| "-".to_owned(), |detail| detail.tokens_generated.to_string()),
        facts.model_sha256.clone(),
        row.binary_sha256.clone(),
        row.features.clone(),
        llama_build,
        fmt_opt(legacy.ttft_cold_ms),
        sample(&legacy.ttft_warm_ms, 0),
        sample(&legacy.ttft_warm_ms, 1),
        fmt_opt(legacy.tok_s_run0),
        sample(&legacy.tok_s, 0),
        sample(&legacy.tok_s, 1),
        fmt_opt(reduced_ms),
        fmt_opt(server_ms),
        definition.to_owned(),
        row.gate.text(),
        row.note.clone(),
    ];
    fields
        .iter()
        .map(|field| field.replace(['\t', '\n'], " "))
        .collect::<Vec<_>>()
        .join("\t")
}

fn stat_cell(values: &[f64]) -> String {
    match sample_mean_cov(values) {
        None => "-".to_owned(),
        Some((center, Some(cov))) => format!("{center:.2} ({cov:.1}%) n={}", values.len()),
        Some((center, None)) => format!("{center:.2} (n/a) n={}", values.len()),
    }
}

/// Means pool only rows that exited 0 and whose parity record holds 11/11 (or n/a).
fn collect(rows: &[&Row], pick: impl Fn(&Row) -> Vec<f64>) -> Vec<f64> {
    rows.iter()
        .filter(|row| row.exit_code == 0 && row.parity.valid())
        .flat_map(|row| pick(row))
        .collect()
}

fn bytes_to_megabytes(bytes: Option<u64>) -> Vec<f64> {
    bytes.map(per_million).into_iter().collect()
}

/// One markdown table per (prompt, kv); one table row per cell; each metric as
/// `mean (CoV%) n=samples` over valid exit-0 processes. INVALID rows are listed after the tables
/// with their failing invariants and never reach a mean.
fn scoreboard_markdown(rows: &[Row], cells: &[Cell]) -> String {
    let mut out = String::from("# kv scoreboard\n\nmetric cells are `mean (CoV%) n=samples` over exit-0 processes whose parity record has no failing invariant; CoV is sample sd / mean x 100; MB is 1e6 bytes.\n");
    for prompt in ALL_PROMPTS {
        for kv in ALL_KV {
            let _ = write!(
                out,
                "\n## {} prompt, {}\n\n| engine | label | n | invalid | nonzero exits | RAM MB | GPU MB | TTFT cold ms | TTFT warm ms | tok/s | tok/s run 0 |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
                prompt.name(),
                kv.name()
            );
            for cell in cells.iter().filter(|cell| cell.prompt == prompt && cell.kv == kv) {
                let mine: Vec<&Row> = rows.iter().filter(|row| row.cell == *cell).collect();
                let failed = mine.iter().filter(|row| row.exit_code != 0).count();
                let invalid = mine.iter().filter(|row| !row.parity.valid()).count();
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                    cell.engine_name(),
                    cell.label,
                    mine.len(),
                    invalid,
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
    out.push_str(&invalid_section(rows));
    out
}

fn invalid_section(rows: &[Row]) -> String {
    let invalid: Vec<&Row> = rows.iter().filter(|row| !row.parity.valid()).collect();
    let mut out = format!("\n## INVALID rows\n\n{} of {} rows are INVALID and excluded from every mean.\n\n", invalid.len(), rows.len());
    for row in invalid {
        let _ = writeln!(
            out,
            "- round {} {} {} {} {}: {}",
            row.round,
            row.cell.engine_name(),
            row.cell.label,
            row.cell.prompt.name(),
            row.cell.kv.name(),
            row.parity.failing().join("; ")
        );
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
    gate: GateConfig,
    context_length: usize,
    kv_bucket: usize,
    features: String,
    llama_commit: String,
    /// path of the `llama-arm` `inproc_arm` bin; adds the llama-inproc arm
    llama_inproc: Option<PathBuf>,
    prompts: Vec<Prompt>,
    kvs: Vec<Kv>,
    /// false runs every process without waiting; every such row is INVALID on I10
    quiet_gate: bool,
    /// directory holding `wp_{84,510,2024,4069}.txt`; its presence selects the stage-matrix mode
    stages: Option<PathBuf>,
    /// KV configuration of the stage-matrix run, one of the llama-comparable kinds
    stage_kv: Kv,
    /// adds the optional `llama-fa-off` arm and the `llama-fa-delta` attention proxy rows
    fa_off_arm: bool,
    /// runs only tokenize and detokenize of the stage prompts; llama stays on the cpu with a tiny context so the weights are mapped, never faulted in
    text_edges: bool,
    /// directory of cached llama-server incumbent samples, one tsv per (model, commit, ids, flash, engine, round); absent means every arm runs
    incumbent_cache: Option<PathBuf>,
    /// ignores cache hits and overwrites the cached samples with fresh runs
    refresh_incumbent: bool,
}

const VALUE_FLAGS: [&str; 20] = [
    "--incumbent-cache",
    "--stages",
    "--stage-kv",
    "--fa-off-arm",
    "--quiet-gate",
    "--llama-inproc",
    "--prompts",
    "--kvs",
    "--out",
    "--model",
    "--llama",
    "--long-prompt-file",
    "--rounds",
    "--time-bin",
    "--load-limit",
    "--hold-secs",
    "--wait-limit-secs",
    "--context-length",
    "--features",
    "--llama-commit",
];

fn parse_value<T: std::str::FromStr>(
    values: &BTreeMap<String, String>,
    flag: &'static str,
    expected: &'static str,
    default: T,
) -> Result<T, HarnessError> {
    values.get(flag).map_or(Ok(default), |text| {
        text.parse().map_err(|_| HarnessError::BadValue {
            flag,
            expected,
            got: text.clone(),
        })
    })
}

fn parse_config(arguments: impl IntoIterator<Item = String>) -> Result<Config, HarnessError> {
    let mut values: BTreeMap<String, String> = BTreeMap::new();
    let mut proxima = Vec::new();
    let mut tokens = arguments.into_iter();
    let mut text_edges = false;
    let mut refresh_incumbent = false;
    while let Some(flag) = tokens.next() {
        if flag == "--text-edges" {
            text_edges = true;
            continue;
        }
        if flag == "--refresh-incumbent" {
            refresh_incumbent = true;
            continue;
        }
        let value = tokens
            .next()
            .ok_or_else(|| HarnessError::MissingValue(flag.clone()))?;
        if flag == "--prox" {
            let (label, path) = value
                .split_once('=')
                .ok_or_else(|| HarnessError::BadProx(value.clone()))?;
            proxima.push((label.to_owned(), PathBuf::from(path)));
        } else if VALUE_FLAGS.contains(&flag.as_str()) {
            values.insert(flag, value);
        } else {
            return Err(HarnessError::UnknownFlag(flag));
        }
    }
    let required = |flag: &'static str| {
        values
            .get(flag)
            .map(PathBuf::from)
            .ok_or(HarnessError::MissingFlag(flag))
    };
    let llama_inproc = values.get("--llama-inproc").map(PathBuf::from);
    let stages = values.get("--stages").map(PathBuf::from);
    let long_prompt_file = match (&stages, values.get("--long-prompt-file")) {
        (_, Some(path)) => PathBuf::from(path),
        (Some(_), None) => PathBuf::new(),
        (None, None) => return Err(HarnessError::MissingFlag("--long-prompt-file")),
    };
    let stage_kv = match values.get("--stage-kv") {
        None => Kv::F16,
        Some(text) => Kv::parse(text).filter(|kv| kv.llama_types().is_some()).ok_or_else(|| HarnessError::BadValue {
            flag: "--stage-kv",
            expected: "F32|F16|BF16",
            got: text.clone(),
        })?,
    };
    let fa_off_arm = match values.get("--fa-off-arm").map(String::as_str) {
        None | Some("off") => false,
        Some("on") => true,
        Some(other) => {
            return Err(HarnessError::BadValue { flag: "--fa-off-arm", expected: "on|off", got: other.to_owned() });
        }
    };
    if text_edges && stages.is_none() {
        return Err(HarnessError::MissingFlag("--stages"));
    }
    if proxima.is_empty() && llama_inproc.is_none() {
        return Err(HarnessError::MissingFlag("--prox"));
    }
    let prompts = parse_names(&values, "--prompts", "short|long", ALL_PROMPTS, Prompt::parse)?;
    let kvs = parse_names(&values, "--kvs", "F32|F16|BF16|Q4K", ALL_KV, Kv::parse)?;
    let rounds: usize = parse_value(&values, "--rounds", "a positive integer", DEFAULT_ROUNDS)?;
    if rounds == 0 {
        return Err(HarnessError::BadValue {
            flag: "--rounds",
            expected: "a positive integer",
            got: "0".to_owned(),
        });
    }
    Ok(Config {
        out: required("--out")?,
        model: required("--model")?,
        llama: required("--llama")?,
        long_prompt_file,
        stages,
        stage_kv,
        fa_off_arm,
        text_edges,
        incumbent_cache: values.get("--incumbent-cache").map(PathBuf::from),
        refresh_incumbent,
        proxima,
        rounds,
        time_bin: values
            .get("--time-bin")
            .map_or_else(|| PathBuf::from(DEFAULT_TIME_BIN), PathBuf::from),
        gate: GateConfig {
            load_limit: parse_value(&values, "--load-limit", "a number", DEFAULT_LOAD_LIMIT)?,
            hold_secs: parse_value(&values, "--hold-secs", "seconds", DEFAULT_HOLD_SECS)?,
            wait_limit_secs: parse_value(&values, "--wait-limit-secs", "seconds", DEFAULT_WAIT_LIMIT_SECS)?,
        },
        context_length: parse_value(&values, "--context-length", "tokens", DEFAULT_CONTEXT_LENGTH)?,
        kv_bucket: DEFAULT_KV_BUCKET,
        llama_commit: values
            .get("--llama-commit")
            .cloned()
            .ok_or(HarnessError::MissingFlag("--llama-commit"))?,
        features: values
            .get("--features")
            .cloned()
            .unwrap_or_else(|| "unrecorded".to_owned()),
        llama_inproc,
        prompts,
        kvs,
        quiet_gate: match values.get("--quiet-gate").map(String::as_str) {
            None | Some("on") => true,
            Some("off") => false,
            Some(other) => {
                return Err(HarnessError::BadValue { flag: "--quiet-gate", expected: "on|off", got: other.to_owned() });
            }
        },
    })
}

/// A comma-separated subset of a closed set (`--prompts short`); absent means the whole set.
fn parse_names<T: Copy, const COUNT: usize>(
    values: &BTreeMap<String, String>,
    flag: &'static str,
    expected: &'static str,
    all: [T; COUNT],
    parse: fn(&str) -> Option<T>,
) -> Result<Vec<T>, HarnessError> {
    let Some(text) = values.get(flag) else {
        return Ok(all.to_vec());
    };
    text.split(',')
        .map(|name| {
            parse(name.trim()).ok_or_else(|| HarnessError::BadValue { flag, expected, got: name.to_owned() })
        })
        .collect()
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

/// A prompt as both engines see it: text for proxima's env, ids for llama's request.
struct PromptData {
    text: String,
    ids: Vec<u32>,
    capacity: usize,
}

/// Mirrors `generate::residency_caches::wants_bos` and the `encode_with_bos_eos` call at
/// `decode.rs:3420-3428`; both are unreachable from an example.
fn tokenize_like_proxima(text: &str, vocab: &Vocab) -> Result<Vec<u32>, HarnessError> {
    let add_bos = vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some());
    let wants_eos = vocab.add_eos_token().unwrap_or(false);
    proxima_tokenizer::encode_with_bos_eos(text, vocab, add_bos, wants_eos)
        .map_err(|source| HarnessError::Tokenize(format!("{source:?}")))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

fn file_sha256(path: &Path) -> Result<String, HarnessError> {
    fs::read(path).map(|bytes| sha256_hex(&bytes)).map_err(io_error(path))
}

fn own_busy_processes() -> Vec<String> {
    let output = Command::new("pgrep").args(["-lf", BUSY_PATTERN]).output();
    let text = output.map(|output| String::from_utf8_lossy(&output.stdout).into_owned()).unwrap_or_default();
    // SAFETY: getppid takes no arguments and cannot fail.
    let parent = unsafe { libc::getppid() };
    busy_from_pgrep(&text, std::process::id(), u32::try_from(parent).unwrap_or(0))
}

fn wait_for_quiet(config: &GateConfig) -> GateOutcome {
    let started = Instant::now();
    let mut state = QuietState {
        started: 0,
        quiet_since: None,
    };
    loop {
        let now = started.elapsed().as_secs();
        match gate_step(&mut state, now, load1(), &own_busy_processes(), config) {
            GateStep::Clear => return GateOutcome::Clear { waited_secs: now },
            GateStep::GaveUp(reason) => return GateOutcome::Refused(reason),
            GateStep::Waiting => std::thread::sleep(GATE_POLL),
        }
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

/// Minimal copy of `decode_arms.rs`'s `http_exchange`, non-streaming.
fn http_exchange(port: u16, method: &str, path: &str, body: &str, timeout: Duration) -> Result<String, HarnessError> {
    let failed = |source: std::io::Error| HarnessError::Server(format!("{method} {path}: {source}"));
    let address: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .map_err(|_| HarnessError::Server("bad address".to_owned()))?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(5)).map_err(failed)?;
    stream.set_read_timeout(Some(timeout)).map_err(failed)?;
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).map_err(failed)?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(failed)?;
    let separator = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| HarnessError::Server("no http response head".to_owned()))?;
    let head = String::from_utf8_lossy(&raw[..separator]).to_ascii_lowercase();
    let payload = &raw[separator + 4..];
    let bytes = if head.contains("transfer-encoding: chunked") {
        dechunk(payload)
    } else {
        payload.to_vec()
    };
    String::from_utf8(bytes).map_err(|_| HarnessError::Server("response body is not utf-8".to_owned()))
}

fn completion_body(ids: &[u32]) -> String {
    json!({
        "prompt": ids,
        "n_predict": MAX_TOKENS,
        "temperature": 0,
        "top_k": 1,
        "seed": 1,
        "cache_prompt": false,
        "ignore_eos": true,
        "return_tokens": true,
    })
    .to_string()
}

/// llama-server under `/usr/bin/time -l`; `stop` signals the server so `time` reports its peak.
struct TimedServer {
    timer: Child,
    server_pid: Option<i32>,
}

impl TimedServer {
    fn spawn(
        config: &Config,
        capacity: usize,
        kv: Kv,
        port: u16,
        stderr_path: &Path,
        flash: &str,
    ) -> Result<Self, HarnessError> {
        let (key, value) = kv.llama_types().ok_or_else(|| HarnessError::Server("no llama arm for this KV".to_owned()))?;
        let stderr_file = File::create(stderr_path).map_err(io_error(stderr_path))?;
        let timer = Command::new(&config.time_bin)
            .arg("-l")
            .arg(&config.llama)
            .arg("-m")
            .arg(&config.model)
            .args(["-ngl", "99", "-fa", flash, "-np", "1", "--no-warmup", "--seed", "1"])
            .args(["-c", &capacity.to_string(), "-ctk", key, "-ctv", value])
            .args(["--host", "127.0.0.1", "--port", &port.to_string(), "-lv", "4"])
            .stdout(Stdio::null())
            .stderr(stderr_file)
            .process_group(0)
            .spawn()
            .map_err(|source| HarnessError::Server(format!("spawn: {source}")))?;
        let server_pid = Self::child_of(timer.id());
        Ok(Self { timer, server_pid })
    }

    fn child_of(parent: u32) -> Option<i32> {
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(10) {
            let output = Command::new("pgrep").args(["-P", &parent.to_string()]).output().ok()?;
            let pid = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .and_then(|line| line.trim().parse::<i32>().ok());
            if pid.is_some() {
                return pid;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        None
    }

    fn wait_ready(port: u16, seconds: u64) -> bool {
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(seconds) {
            let answer = http_exchange(port, "GET", "/health", "", Duration::from_secs(5)).unwrap_or_default();
            if answer.contains("\"ok\"") {
                return true;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        false
    }

    /// the server pid when pgrep found it, else the timer's whole process group, so stop never depends on the lookup.
    fn signal_target(&self) -> i32 {
        self.server_pid.unwrap_or_else(|| -i32::try_from(self.timer.id()).unwrap_or(i32::MAX))
    }

    fn stop(mut self) -> i32 {
        let target = self.signal_target();
        // SAFETY: target is the pgrep-found server pid, or the negated pid of the process group this harness created for the timer; SIGTERM only.
        unsafe { libc::kill(target, libc::SIGTERM) };
        let asked = Instant::now();
        while asked.elapsed() < STOP_GRACE && matches!(self.timer.try_wait(), Ok(None)) {
            std::thread::sleep(Duration::from_millis(50));
        }
        // llama-server ignores SIGTERM until the tensor load finishes, so a load in progress needs the kill.
        if matches!(self.timer.try_wait(), Ok(None)) {
            // SAFETY: same target as above; SIGKILL cannot be handled, so the load stops now.
            unsafe { libc::kill(target, libc::SIGKILL) };
        }
        self.timer.wait().ok().and_then(|status| status.code()).unwrap_or(-1)
    }
}

struct RunContext<'a> {
    config: &'a Config,
    prompt: &'a PromptData,
    round: usize,
    port: u16,
    log_path: PathBuf,
}

fn run_llama(context: &RunContext, cell: &Cell, row: &mut Row) -> Result<(), HarnessError> {
    let server = TimedServer::spawn(context.config, context.prompt.capacity, cell.kv, context.port, &context.log_path, "on")?;
    let outcome = (|| {
        if !TimedServer::wait_ready(context.port, 180) {
            return Err(HarnessError::Server("never became healthy".to_owned()));
        }
        let body = completion_body(&context.prompt.ids);
        let ask = || {
            http_exchange(context.port, "POST", "/completion", &body, Duration::from_secs(300))
                .and_then(|text| parse_llama_response(&text))
        };
        Ok((ask()?, ask()?))
    })();
    row.exit_code = server.stop();
    let (cold, warm) = outcome?;
    let log_text = fs::read_to_string(&context.log_path).map_err(io_error(&context.log_path))?;
    row.footprint_bytes = parse_footprint_bytes(&log_text).ok();
    row.metrics = Metrics {
        ttft_cold_ms: Some(cold.prompt_ms),
        ttft_warm_ms: vec![warm.prompt_ms],
        tok_s: vec![warm.predicted_per_second],
        tok_s_run0: Some(cold.predicted_per_second),
        ..Metrics::default()
    };
    row.evidence = Evidence::Llama(Box::new(LlamaEvidence {
        cold,
        warm,
        log: parse_llama_log(&log_text),
    }));
    Ok(())
}

fn run_proxima(context: &RunContext, cell: &Cell, binary: &Path, row: &mut Row) -> Result<(), HarnessError> {
    let mut command = Command::new(&context.config.time_bin);
    command.arg("-l").arg(binary);
    command.env("PROXIMA_SPECULATIVE_TYPES", "none");
    command.env("PROXIMA_STEP_TIMES", "1");
    command.env("PROXIMA_MAX_TOKENS", MAX_TOKENS.to_string());
    command.env("PROXIMA_RUNS", "3");
    for (key, value) in cell.kv.proxima_env() {
        command.env(key, value);
    }
    if cell.prompt == Prompt::Long {
        command.env("PROXIMA_PROMPT", &context.prompt.text);
    }
    let output = command
        .output()
        .map_err(|source| HarnessError::Server(format!("spawn failed: {source}")))?;
    row.exit_code = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let combined = format!("{}\n{stderr}", String::from_utf8_lossy(&output.stdout));
    fs::write(&context.log_path, &combined).map_err(io_error(&context.log_path))?;
    row.footprint_bytes = parse_footprint_bytes(&stderr).ok();
    let legacy = parse_proxima(&combined)?;
    let span = parse_proxima_span(&combined, &legacy);
    row.span_metrics = span.is_some();
    row.metrics = span.unwrap_or_else(|| legacy.clone());
    row.legacy = Some(legacy);
    row.evidence = Evidence::Proxima(parse_proxima_detail(&combined)?);
    Ok(())
}

/// One `inproc_arm` process under `/usr/bin/time -l`: same prompt ids, `n_predict`, KV types and
/// padded `-c` as the server arm, libllama taken from the server's own install directory so both
/// arms load the same build. Its row's `binary_sha256` is the gate's libllama sha256 (I12); the
/// arm bin's own sha256 goes in the note.
fn run_inproc(context: &RunContext, cell: &Cell, binary: &Path, row: &mut Row) -> Result<(), HarnessError> {
    let (type_k, type_v) = cell.kv.llama_types().ok_or_else(|| HarnessError::Server("no llama arm for this KV".to_owned()))?;
    let ids: Vec<String> = context.prompt.ids.iter().map(u32::to_string).collect();
    let library_dir = context.config.llama.parent().unwrap_or(Path::new("."));
    let output = Command::new(&context.config.time_bin)
        .arg("-l")
        .arg(binary)
        .arg("--model")
        .arg(&context.config.model)
        .args(["--ids", &ids.join(",")])
        .args(["--n-ctx", &llama_padded_capacity(context.prompt.capacity).to_string()])
        .args(["--n-predict", &MAX_TOKENS.to_string(), "--runs", "3"])
        .args(["--type-k", type_k, "--type-v", type_v])
        .env("LLAMA_LIB_DIR", library_dir)
        .env("GGML_METAL_RESIDENCY_KEEP_ALIVE_S", "1")
        .output()
        .map_err(|source| HarnessError::Server(format!("spawn failed: {source}")))?;
    row.exit_code = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    fs::write(&context.log_path, &stderr).map_err(io_error(&context.log_path))?;
    row.footprint_bytes = parse_footprint_bytes(&stderr).ok();
    let gate = parse_inproc_gate(&stderr).ok_or(HarnessError::MissingLine("llama_arm gate line"))?;
    row.binary_sha256.clone_from(&gate.sha256);
    let metrics = parse_inproc_span(&stderr);
    row.span_metrics = metrics.is_some();
    row.metrics = metrics.unwrap_or_default();
    let detail = parse_proxima_detail(&stderr)?;
    row.evidence = Evidence::Inproc(Box::new(InprocEvidence { detail, gate }));
    Ok(())
}

struct ArmIdentity {
    sha256: String,
    debug_build: bool,
}

fn arm_identity(path: &Path) -> Result<ArmIdentity, HarnessError> {
    Ok(ArmIdentity {
        sha256: file_sha256(path)?,
        debug_build: path.components().any(|part| part.as_os_str() == "debug"),
    })
}

fn blank_row(round: usize, cell: &Cell, prompt: &PromptData, identity: &ArmIdentity, config: &Config, gate: GateOutcome) -> Row {
    Row {
        round,
        cell: cell.clone(),
        load1: load1(),
        footprint_bytes: None,
        metrics: Metrics::default(),
        exit_code: -1,
        note: String::new(),
        evidence: Evidence::None,
        legacy: None,
        span_metrics: false,
        sent_ids: prompt.ids.len(),
        capacity: prompt.capacity,
        binary_sha256: identity.sha256.clone(),
        debug_build: identity.debug_build,
        features: config.features.clone(),
        gate,
        parity: Parity::unevaluated(),
    }
}

fn load_prompts(config: &Config) -> Result<(String, BTreeMap<Prompt, PromptData>), HarnessError> {
    let file = File::open(&config.model).map_err(io_error(&config.model))?;
    // SAFETY: the mapping is read-only and the blob is not modified while the harness runs.
    let bytes = unsafe { MmapOptions::new().map(&file) }.map_err(io_error(&config.model))?;
    let model_sha256 = sha256_hex(&bytes);
    let parsed = parse_complete(&bytes).map_err(|source| HarnessError::Tokenize(format!("{source:?}")))?;
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .map_err(|source| HarnessError::Tokenize(format!("{source:?}")))?;
    let long_text = fs::read_to_string(&config.long_prompt_file)
        .map_err(io_error(&config.long_prompt_file))?
        .trim_end_matches('\n')
        .to_owned();
    let mut prompts = BTreeMap::new();
    for (prompt, text) in [(Prompt::Short, SHORT_PROMPT.to_owned()), (Prompt::Long, long_text)] {
        let ids = tokenize_like_proxima(&text, &vocab)?;
        let capacity = kv_capacity(ids.len(), MAX_TOKENS, config.kv_bucket, config.context_length);
        prompts.insert(prompt, PromptData { text, ids, capacity });
    }
    Ok((model_sha256, prompts))
}

fn run(config: &Config) -> Result<(), HarnessError> {
    let logs = config.out.join("logs");
    fs::create_dir_all(&logs).map_err(io_error(&logs))?;
    let (model_sha256, prompts) = load_prompts(config)?;
    let facts = Facts {
        model_path: config.model.display().to_string(),
        model_sha256,
        load_limit: config.gate.load_limit,
        llama_commit: config.llama_commit.clone(),
    };
    let llama_identity = arm_identity(&config.llama)?;
    let mut cells = cell_order(&config.proxima, config.llama_inproc.as_deref());
    cells.retain(|cell| config.prompts.contains(&cell.prompt) && config.kvs.contains(&cell.kv));
    let mut rows = Vec::new();
    for (step, (round, index)) in round_plan(config.rounds, cells.len()).into_iter().enumerate() {
        let cell = &cells[index];
        let prompt = &prompts[&cell.prompt];
        let identity = match &cell.engine {
            Engine::Proxima { binary } | Engine::LlamaInproc { binary } => arm_identity(binary)?,
            Engine::Llama => ArmIdentity { sha256: llama_identity.sha256.clone(), debug_build: false },
        };
        let gate = if config.quiet_gate { wait_for_quiet(&config.gate) } else { GateOutcome::Skipped };
        let mut row = blank_row(round, cell, prompt, &identity, config, gate.clone());
        let context = RunContext {
            config,
            prompt,
            round,
            port: BASE_PORT + u16::try_from(step % 1000).unwrap_or(0),
            log_path: logs.join(format!("r{round}-{}-{}-{}-{}.err", cell.engine_name(), cell.label, cell.prompt.name(), cell.kv.name())),
        };
        if gate.may_run() {
            let outcome = match &cell.engine {
                Engine::Proxima { binary } => run_proxima(&context, cell, binary, &mut row),
                Engine::Llama => run_llama(&context, cell, &mut row),
                Engine::LlamaInproc { binary } => {
                    row.note = format!("llama-arm bin sha256 {}", identity.sha256);
                    run_inproc(&context, cell, binary, &mut row)
                }
            };
            if let Err(error) = outcome {
                row.note = format!("{}; {error}", row.note);
            }
        } else {
            "not run: quiet gate refused".clone_into(&mut row.note);
        }
        println!("round {} {} {} {} {} exit {} {}", context.round, cell.engine_name(), cell.label, cell.prompt.name(), cell.kv.name(), row.exit_code, row.note);
        rows.push(row);
    }
    evaluate_all(&mut rows, &facts);
    write_outputs(config, &cells, &rows, &facts)
}

fn write_outputs(config: &Config, cells: &[Cell], rows: &[Row], facts: &Facts) -> Result<(), HarnessError> {
    let mut tsv = format!("{TSV_HEADER}\n");
    for row in rows {
        let _ = writeln!(tsv, "{}", tsv_line(row, find_peer(rows, row), facts));
    }
    let tsv_path = config.out.join("raw.tsv");
    fs::write(&tsv_path, &tsv).map_err(io_error(&tsv_path))?;
    let table_path = config.out.join("scoreboard.md");
    let table = scoreboard_markdown(rows, cells);
    fs::write(&table_path, &table).map_err(io_error(&table_path))?;
    println!("{table}");
    Ok(())
}

// ---- stage matrix: every pipeline stage of both engines on the same prompt files ----

const STAGE_PROMPT_LABELS: [&str; 4] = ["84", "510", "2024", "4069"];
const TEXT_EDGE_REPS: usize = 20;
const MEBIBYTE: f64 = 1_048_576.0;
const STAGE_BASE_PORT: u16 = BASE_PORT + 500;
const PROFILE_PREFILL_STEP: &str = "0";
const PROFILE_DECODE_STEP: &str = "3";
const PROFILE_DECODE_TOKENS: usize = 5;
const STAGE_HEADER: &str = "round\tengine\tprompt\tstage\tmetric\tvalue\tunit\tstatus\tload1\tdefinition\tnote";
const NO_LLAMA_OP_TIMER: &str = "llama-server exposes no per-op timer and none is invented here; the closest real proxy is the llama-fa-delta rows (--fa-off-arm on)";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Tokenization,
    Prefill,
    Decode,
    Attention,
    Kv,
    Detokenization,
    EndToEnd,
    Extra,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Tokenization => "tokenization",
            Stage::Prefill => "prefill",
            Stage::Decode => "decode",
            Stage::Attention => "attention",
            Stage::Kv => "kv",
            Stage::Detokenization => "detokenization",
            Stage::EndToEnd => "ete",
            Stage::Extra => "extra",
        }
    }
}

/// One measured quantity with the definition that both engines' values share.
struct Metric {
    stage: Stage,
    name: &'static str,
    unit: &'static str,
    definition: &'static str,
}

const fn metric(stage: Stage, name: &'static str, unit: &'static str, definition: &'static str) -> Metric {
    Metric { stage, name, unit, definition }
}

const TOKENIZE_MS: Metric = metric(Stage::Tokenization, "tokenize_ms", "ms", "text to ids for the prompt file. proxima: mean of 20 in-process encode_with_bos_eos calls on the GGUF vocab after 1 discarded call, no transport. llama: client wall of POST /tokenize {content, add_special:true} minus tokenize_http_floor_ms");
const TOKENIZE_WALL_MS: Metric = metric(Stage::Tokenization, "tokenize_wall_ms", "ms", "llama: client wall of POST /tokenize on the prompt text, HTTP included. proxima: the in-process call, equal to tokenize_ms");
const TOKENIZE_FLOOR_MS: Metric = metric(Stage::Tokenization, "tokenize_http_floor_ms", "ms", "llama only: client wall of POST /tokenize on an empty string against the local server, the HTTP round trip with no tokenizer work, subtracted from tokenize_wall_ms");
const TOKENIZE_TOKENS: Metric = metric(Stage::Tokenization, "tokenize_tokens", "count", "ids the tokenizer returned for the prompt text (proxima: encode_with_bos_eos count; llama: /tokenize tokens length)");
const PREFILL_MS: Metric = metric(Stage::Prefill, "prefill_ms_per_prompt_token", "ms/token", "warm prefill ms / prompt tokens. proxima: mean over runs 1 and 2 of the prefill step_time step_ns / prompt tokens. llama: timings.prompt_ms of request 2 / timings.prompt_n");
const PREFILL_COLD_MS: Metric = metric(Stage::Prefill, "prefill_cold_ms_per_prompt_token", "ms/token", "first-request prefill ms / prompt tokens. proxima: run 0 prefill step_ns. llama: timings.prompt_ms of request 1");
const DECODE_MS: Metric = metric(Stage::Decode, "decode_ms_per_token", "ms/token", "proxima: mean over runs 1 and 2 of decode_span_ms_per_token over step_time records (decode index>=1). llama: timings.predicted_per_token_ms of request 2 (server timing, includes server overhead)");
const ATTN_PREFILL_MS: Metric = metric(Stage::Attention, "attention_prefill_gpu_ms", "ms", "proxima: sum of gpu_ms over op_profile_kind kinds containing 'attention' in a separate process with PROXIMA_METAL_OP_PROFILE_STEP=0 (one command buffer per op, so slower than the unprofiled step; kinds not named attention are not counted)");
const ATTN_PREFILL_SHARE: Metric = metric(Stage::Attention, "attention_prefill_share", "fraction", "attention_prefill_gpu_ms / sum of gpu_ms over every op_profile_kind of the same profiled step");
const ATTN_DECODE_MS: Metric = metric(Stage::Attention, "attention_decode_gpu_ms", "ms", "as attention_prefill_gpu_ms with PROXIMA_METAL_OP_PROFILE_STEP=3 and 5 tokens");
const ATTN_DECODE_SHARE: Metric = metric(Stage::Attention, "attention_decode_share", "fraction", "attention_decode_gpu_ms / sum of gpu_ms over every op_profile_kind of the same profiled step");
const FA_PREFILL_PROXY: Metric = metric(Stage::Attention, "fa_off_minus_on_prefill_ms_per_token", "ms/token", "llama proxy, not attention time: warm prefill ms/token with -fa off minus -fa on; the graph also changes KV layout and softmax materialisation, so this bounds attention cost only loosely");
const FA_DECODE_PROXY: Metric = metric(Stage::Attention, "fa_off_minus_on_decode_ms_per_token", "ms/token", "llama proxy, not attention time: timings.predicted_per_token_ms with -fa off minus -fa on");
const KV_BYTES_PER_TOKEN: Metric = metric(Stage::Kv, "kv_bytes_per_token", "bytes", "KV bytes per token position across all layers. proxima: mean layer_cache_append_bytes over token_breakdown decode steps. llama: load-log 'KV buffer size' bytes / n_ctx_seq");
const KV_PEAK_BYTES: Metric = metric(Stage::Kv, "kv_peak_bytes", "bytes", "llama: sum of the load-log 'KV buffer size' lines, allocated up front for the padded -c so it is also the peak. proxima: no KV-only peak is logged");
const KV_WRITE_MS: Metric = metric(Stage::Kv, "kv_write_ms_per_step", "ms", "proxima: mean over token_breakdown decode steps (step>=1) of named_blocks_kv_ms + layer_cache_append_ms + that step's device_kv_narrow narrow_ms when logged, the encode-side KV timing. llama: no per-step KV timer");
const DETOKENIZE_MS: Metric = metric(Stage::Detokenization, "detokenize_ms", "ms", "the 128 generated ids to text. proxima: mean of 20 in-process proxima_tokenizer::decode calls after 1 discarded call. llama: client wall of POST /detokenize minus detokenize_http_floor_ms");
const DETOKENIZE_WALL_MS: Metric = metric(Stage::Detokenization, "detokenize_wall_ms", "ms", "llama: client wall of POST /detokenize on the generated ids, HTTP included. proxima: the in-process call, equal to detokenize_ms");
const DETOKENIZE_FLOOR_MS: Metric = metric(Stage::Detokenization, "detokenize_http_floor_ms", "ms", "llama only: client wall of POST /detokenize with an empty token array, subtracted from detokenize_wall_ms");
const TOKENIZE_INPROC_MS: Metric = metric(Stage::Tokenization, "tokenize_inproc_ms", "ms", "text to ids for the prompt file, no transport. mean of 20 in-process calls after 1 discarded: proxima_tokenizer on the GGUF vocab / llama b12110 llama_tokenize on a vocab_only model");
const TOKENIZE_INPROC_TOKENS: Metric = metric(Stage::Tokenization, "tokenize_inproc_tokens", "count", "ids returned by the in-process tokenize call on the prompt text: proxima_tokenizer encode_with_bos_eos on the GGUF vocab / llama b12110 llama_tokenize on a vocab_only model");
const DETOKENIZE_PROMPT_INPROC_MS: Metric = metric(Stage::Detokenization, "detokenize_prompt_inproc_ms", "ms", "the prompt's own ids to text, never merged into detokenize_ms (the 128 generated ids). mean of 20 in-process calls after 1 discarded: proxima_tokenizer on the GGUF vocab / llama b12110 llama_detokenize on a vocab_only model");
const VOCAB_ONLY_RSS_BYTES: Metric = metric(Stage::Extra, "vocab_only_process_rss_bytes", "bytes", "resident set (ps -o rss=) of this harness process after the llama b12110 vocab_only model load and its in-process calls; no tensors are loaded, so it must stay under the text-edge limit");
const E2E_WALL_MS: Metric = metric(Stage::EndToEnd, "e2e_wall_ms", "ms", "full request, 128 tokens out. proxima: mean over runs 1 and 2 of run=done wall_ms from the decode_gbps_baseline loop (text edges not proven inside it, see e2e_with_text_edges_ms). llama: client wall of warm POST /completion with the prompt as text, n_predict 128, ignore_eos, text returned");
const E2E_EDGES_MS: Metric = metric(Stage::EndToEnd, "e2e_with_text_edges_ms", "ms", "proxima DERIVED: e2e_wall_ms + tokenize_ms + detokenize_ms, the composed text-in text-out path; llama's /completion already includes both edges");
const LOAD_MS: Metric = metric(Stage::Extra, "model_load_ms", "ms", "proxima DERIVED: /usr/bin/time real minus the sum of run wall_ms (process start, GGUF parse, weight load, teardown). llama: spawn to /health ok, 500 ms poll granularity");
const FIRST_REQUEST_MS: Metric = metric(Stage::Extra, "first_request_compile_ms", "ms", "DERIVED: first-request prefill ms minus warm prefill ms, the pipeline-compile and first-use cost (llama runs --no-warmup)");
const FOOTPRINT_BYTES: Metric = metric(Stage::Extra, "peak_footprint_bytes", "bytes", "/usr/bin/time -l 'peak memory footprint' of the engine process (proxima binary, or llama-server)");
const RSS_BYTES: Metric = metric(Stage::Extra, "max_rss_bytes", "bytes", "/usr/bin/time -l 'maximum resident set size' of the engine process");
const GPU_PEAK_BYTES: Metric = metric(Stage::Extra, "gpu_peak_bytes", "bytes", "proxima: max gpu_peak_bytes over run=done lines. llama: not reported by llama-server");
const ARM_ERROR: Metric = metric(Stage::Extra, "arm_error", "none", "an arm that failed to run; the reason is in the note");

/// (column title, metric name) in the owner's order: tokenization, prefill, decode, attention, kv, detokenization, ete, extras.
const HEADLINES: [(&str, &str); 20] = [
    ("tokenize ms", "tokenize_ms"),
    ("prefill ms/tok", "prefill_ms_per_prompt_token"),
    ("decode ms/tok", "decode_ms_per_token"),
    ("attn prefill gpu ms", "attention_prefill_gpu_ms"),
    ("attn prefill share", "attention_prefill_share"),
    ("attn decode gpu ms", "attention_decode_gpu_ms"),
    ("attn decode share", "attention_decode_share"),
    ("kv bytes/tok", "kv_bytes_per_token"),
    ("kv peak bytes", "kv_peak_bytes"),
    ("kv write ms/step", "kv_write_ms_per_step"),
    ("detokenize ms", "detokenize_ms"),
    ("detokenize prompt inproc ms", "detokenize_prompt_inproc_ms"),
    ("ete wall ms", "e2e_wall_ms"),
    ("load ms", "model_load_ms"),
    ("first-req compile ms", "first_request_compile_ms"),
    ("footprint bytes", "peak_footprint_bytes"),
    ("rss bytes", "max_rss_bytes"),
    ("gpu peak bytes", "gpu_peak_bytes"),
    ("vocab-only process rss bytes", "vocab_only_process_rss_bytes"),
    ("tokenize inproc ms", "tokenize_inproc_ms"),
];

#[derive(Clone, Debug, PartialEq)]
struct StageRow {
    round: usize,
    engine: String,
    prompt: String,
    stage: Stage,
    metric: &'static str,
    value: Option<f64>,
    unit: &'static str,
    definition: &'static str,
    load1: f64,
    note: String,
}

struct RowSink {
    round: usize,
    engine: String,
    prompt: String,
    load1: f64,
    rows: Vec<StageRow>,
}

impl RowSink {
    fn new(round: usize, engine: &str, prompt: &str, load1: f64) -> Self {
        Self { round, engine: engine.to_owned(), prompt: prompt.to_owned(), load1, rows: Vec::new() }
    }

    fn push(&mut self, spec: &Metric, value: Option<f64>, note: &str) {
        self.rows.push(StageRow {
            round: self.round,
            engine: self.engine.clone(),
            prompt: self.prompt.clone(),
            stage: spec.stage,
            metric: spec.name,
            value,
            unit: spec.unit,
            definition: spec.definition,
            load1: self.load1,
            note: note.to_owned(),
        });
    }

    fn measured(&mut self, spec: &Metric, value: f64, note: &str) {
        self.push(spec, Some(value), note);
    }

    fn unavailable(&mut self, spec: &Metric, reason: &str) {
        self.push(spec, None, reason);
    }

    /// A value when the source line existed, else an unavailable cell naming what was missing.
    fn optional(&mut self, spec: &Metric, value: Option<f64>, missing: &str) {
        self.push(spec, value, if value.is_some() { "" } else { missing });
    }
}

/// Everything one llama-server arm observed for one prompt.
#[derive(Debug, Clone, PartialEq)]
struct LlamaStage {
    cold: LlamaResponse,
    warm: LlamaResponse,
    tokenize_wall_ms: f64,
    tokenize_floor_ms: f64,
    tokenize_tokens: usize,
    detokenize_wall_ms: f64,
    detokenize_floor_ms: f64,
    e2e_wall_ms: f64,
    ready_ms: f64,
    log: LlamaLog,
    footprint_bytes: Option<u64>,
    rss_bytes: Option<u64>,
}

fn bytes_value(bytes: Option<u64>) -> Option<f64> {
    bytes.map(bytes_as_f64)
}

fn llama_rows(sink: &mut RowSink, input: &LlamaStage) {
    sink.measured(&TOKENIZE_WALL_MS, input.tokenize_wall_ms, "");
    sink.measured(&TOKENIZE_FLOOR_MS, input.tokenize_floor_ms, "");
    sink.measured(&TOKENIZE_MS, input.tokenize_wall_ms - input.tokenize_floor_ms, "wall minus http floor");
    sink.measured(&TOKENIZE_TOKENS, count_as_f64(input.tokenize_tokens), "");
    sink.measured(&PREFILL_MS, input.warm.prompt_ms / count_as_f64(input.warm.prompt_n), "request 2");
    sink.measured(&PREFILL_COLD_MS, input.cold.prompt_ms / count_as_f64(input.cold.prompt_n), "request 1");
    sink.measured(&DECODE_MS, input.warm.predicted_per_token_ms, "request 2");
    for spec in [&ATTN_PREFILL_MS, &ATTN_PREFILL_SHARE, &ATTN_DECODE_MS, &ATTN_DECODE_SHARE] {
        sink.unavailable(spec, NO_LLAMA_OP_TIMER);
    }
    let kv_bytes = input.log.kv_buffer_mib * MEBIBYTE;
    sink.optional(
        &KV_BYTES_PER_TOKEN,
        input.log.n_ctx_seq.filter(|&context| context > 0).map(|context| kv_bytes / count_as_f64(context)),
        "load log has no n_ctx_seq line",
    );
    sink.optional(&KV_PEAK_BYTES, Some(kv_bytes).filter(|&bytes| bytes > 0.0), "load log has no KV buffer size line");
    sink.unavailable(&KV_WRITE_MS, "llama-server has no per-step KV timer");
    sink.measured(&DETOKENIZE_WALL_MS, input.detokenize_wall_ms, "");
    sink.measured(&DETOKENIZE_FLOOR_MS, input.detokenize_floor_ms, "");
    sink.measured(&DETOKENIZE_MS, input.detokenize_wall_ms - input.detokenize_floor_ms, "wall minus http floor");
    sink.measured(&E2E_WALL_MS, input.e2e_wall_ms, "warm, text prompt");
    sink.unavailable(&E2E_EDGES_MS, "/completion already includes server-side tokenize and detokenize");
    sink.measured(&LOAD_MS, input.ready_ms, "");
    sink.measured(&FIRST_REQUEST_MS, input.cold.prompt_ms - input.warm.prompt_ms, "request 1 minus request 2 prompt_ms");
    sink.optional(&FOOTPRINT_BYTES, bytes_value(input.footprint_bytes), "no peak memory footprint line");
    sink.optional(&RSS_BYTES, bytes_value(input.rss_bytes), "no maximum resident set size line");
    sink.unavailable(&GPU_PEAK_BYTES, "llama-server does not report GPU memory");
}

/// Attention gpu time and the whole profiled step's gpu time from `op_profile_kind` lines.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OpProfile {
    attention_ms: f64,
    total_ms: f64,
}

fn parse_op_profile(text: &str) -> Option<OpProfile> {
    let mut profile = OpProfile { attention_ms: 0.0, total_ms: 0.0 };
    let mut lines = 0_usize;
    // the engine prints the profile once per profiled evaluation (a real log carries two blocks); the first block is one step
    for line in text.lines().filter(|line| line.starts_with("op_profile step=") || line.contains("op_profile_kind step=")) {
        if line.starts_with("op_profile step=") {
            if lines > 0 {
                break;
            }
            continue;
        }
        let (Some(kind), Some(gpu_ms)) =
            (field_value(line, "kind="), field_value(line, "gpu_ms=").and_then(|token| token.parse::<f64>().ok()))
        else {
            continue;
        };
        profile.total_ms += gpu_ms;
        if kind.contains("attention") {
            profile.attention_ms += gpu_ms;
        }
        lines += 1;
    }
    (lines > 0).then_some(profile)
}

/// Per-decode-step means of the `token_breakdown` event.
#[derive(Debug, Clone, Copy, PartialEq)]
struct KvBreakdown {
    write_ms: f64,
    append_bytes: f64,
}

fn parse_kv_breakdown(text: &str) -> Option<KvBreakdown> {
    let (mut write_ms, mut append_bytes, mut steps) = (0.0, 0.0, 0_usize);
    // the device KV path logs `device_kv_narrow step= narrow_ms=` just before that step's token_breakdown line
    let mut pending_narrow: Option<(usize, f64)> = None;
    for line in text.lines().filter(|line| line.contains("token_breakdown") || line.contains("device_kv_narrow step=")) {
        if line.contains("device_kv_narrow step=") {
            pending_narrow = (|| Some((field_value(line, "step=")?.parse().ok()?, field_value(line, "narrow_ms=")?.parse().ok()?)))();
            continue;
        }
        let parsed = (|| {
            let step: usize = field_value(line, "step=")?.parse().ok()?;
            let encode: f64 = field_value(line, "named_blocks_kv_ms=")?.parse().ok()?;
            let append: f64 = field_value(line, "layer_cache_append_ms=")?.parse().ok()?;
            let bytes: f64 = field_value(line, "layer_cache_append_bytes=")?.parse().ok()?;
            let narrow = pending_narrow.filter(|(narrow_step, _)| *narrow_step == step).map_or(0.0, |(_, millis)| millis);
            (step >= 1).then_some((encode + append + narrow, bytes))
        })();
        if let Some((millis, bytes)) = parsed {
            write_ms += millis;
            append_bytes += bytes;
            steps += 1;
        }
    }
    (steps > 0).then(|| KvBreakdown {
        write_ms: write_ms / count_as_f64(steps),
        append_bytes: append_bytes / count_as_f64(steps),
    })
}

fn parse_run_walls(text: &str) -> BTreeMap<usize, f64> {
    text.lines()
        .filter(|line| line.contains("run=done"))
        .filter_map(|line| Some((run_index_of(line)?, field_value(line, "wall_ms=")?.parse().ok()?)))
        .collect()
}

/// `/usr/bin/time -l` first line: `7.53 real 2.90 user 1.85 sys`, in milliseconds.
fn parse_real_ms(text: &str) -> Option<f64> {
    let line = text.lines().find(|line| line.contains(" real "))?;
    Some(line.split_whitespace().next()?.parse::<f64>().ok()? * 1e3)
}

fn parse_rss_bytes(text: &str) -> Option<u64> {
    let line = text.lines().find(|line| line.trim_end().ends_with("maximum resident set size"))?;
    line.split_whitespace().next()?.parse().ok()
}

/// Everything one proxima run set observed for one prompt.
#[derive(Debug, Clone, PartialEq)]
struct ProximaStage {
    prompt_tokens: usize,
    tokenize_ms: f64,
    detokenize_ms: f64,
    ttft_cold_ms: f64,
    ttft_warm_ms: Vec<f64>,
    decode_ms: Vec<f64>,
    run_walls_ms: BTreeMap<usize, f64>,
    process_real_ms: Option<f64>,
    gpu_peak_bytes: Option<u64>,
    footprint_bytes: Option<u64>,
    rss_bytes: Option<u64>,
    kv: Option<KvBreakdown>,
    attention_prefill: Option<OpProfile>,
    attention_decode: Option<OpProfile>,
}

fn warm_runs_mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| mean(values))
}

fn attention_rows(sink: &mut RowSink, profile: Option<OpProfile>, millis: &Metric, share: &Metric, missing: &str) {
    sink.optional(millis, profile.map(|found| found.attention_ms), missing);
    sink.optional(
        share,
        profile.filter(|found| found.total_ms > 0.0).map(|found| found.attention_ms / found.total_ms),
        missing,
    );
}

fn proxima_rows(sink: &mut RowSink, input: &ProximaStage) {
    sink.measured(&TOKENIZE_WALL_MS, input.tokenize_ms, "in-process, no transport");
    sink.unavailable(&TOKENIZE_FLOOR_MS, "in-process call, no transport to subtract");
    sink.measured(&TOKENIZE_MS, input.tokenize_ms, "");
    sink.measured(&TOKENIZE_TOKENS, count_as_f64(input.prompt_tokens), "");
    let tokens = count_as_f64(input.prompt_tokens.max(1));
    let warm = warm_runs_mean(&input.ttft_warm_ms);
    sink.optional(&PREFILL_MS, warm.map(|millis| millis / tokens), "no warm prefill step_time lines");
    sink.measured(&PREFILL_COLD_MS, input.ttft_cold_ms / tokens, "run 0");
    sink.optional(&DECODE_MS, warm_runs_mean(&input.decode_ms), "no decode span records");
    attention_rows(sink, input.attention_prefill, &ATTN_PREFILL_MS, &ATTN_PREFILL_SHARE, "no op_profile_kind lines from the step 0 profile run");
    attention_rows(sink, input.attention_decode, &ATTN_DECODE_MS, &ATTN_DECODE_SHARE, "no op_profile_kind lines from the step 3 profile run");
    let no_breakdown = "no token_breakdown lines in the run's output (event not emitted at this log level)";
    sink.optional(&KV_BYTES_PER_TOKEN, input.kv.map(|found| found.append_bytes), no_breakdown);
    sink.unavailable(&KV_PEAK_BYTES, "decode_gbps_baseline logs no KV-only peak; gpu_peak_bytes is whole-process");
    sink.optional(&KV_WRITE_MS, input.kv.map(|found| found.write_ms), no_breakdown);
    sink.measured(&DETOKENIZE_WALL_MS, input.detokenize_ms, "in-process, no transport");
    sink.unavailable(&DETOKENIZE_FLOOR_MS, "in-process call, no transport to subtract");
    sink.measured(&DETOKENIZE_MS, input.detokenize_ms, "");
    let walls: Vec<f64> = [1, 2].iter().filter_map(|run| input.run_walls_ms.get(run).copied()).collect();
    let wall = warm_runs_mean(&walls);
    sink.optional(&E2E_WALL_MS, wall, "no run=done wall_ms for runs 1 and 2");
    sink.optional(&E2E_EDGES_MS, wall.map(|millis| millis + input.tokenize_ms + input.detokenize_ms), "needs e2e_wall_ms");
    let loop_ms: f64 = input.run_walls_ms.values().sum();
    sink.optional(&LOAD_MS, input.process_real_ms.map(|real| real - loop_ms), "no /usr/bin/time real line");
    sink.optional(&FIRST_REQUEST_MS, warm.map(|millis| input.ttft_cold_ms - millis), "needs warm prefill");
    sink.optional(&FOOTPRINT_BYTES, bytes_value(input.footprint_bytes), "no peak memory footprint line");
    sink.optional(&RSS_BYTES, bytes_value(input.rss_bytes), "no maximum resident set size line");
    sink.optional(&GPU_PEAK_BYTES, bytes_value(input.gpu_peak_bytes), "no gpu_peak_bytes in run=done");
}

/// The llama flash-attention on/off delta as the only available attention proxy.
fn fa_delta_rows(sink: &mut RowSink, on: &LlamaStage, off: &LlamaStage) {
    let prefill = |stage: &LlamaStage| stage.warm.prompt_ms / count_as_f64(stage.warm.prompt_n);
    sink.measured(&FA_PREFILL_PROXY, prefill(off) - prefill(on), "request 2 of each arm");
    sink.measured(&FA_DECODE_PROXY, off.warm.predicted_per_token_ms - on.warm.predicted_per_token_ms, "request 2 of each arm");
}

fn sanitize(text: &str) -> String {
    text.replace(['\t', '\n'], " ")
}

fn stage_tsv(rows: &[StageRow]) -> String {
    let mut tsv = format!("{STAGE_HEADER}\n");
    for row in rows {
        let (value, status) = row.value.map_or((String::new(), "unavailable"), |found| (format!("{found:.6}"), "measured"));
        let _ = writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{value}\t{}\t{status}\t{:.2}\t{}\t{}",
            row.round, row.engine, row.prompt, row.stage.name(), row.metric, row.unit, row.load1, row.definition, sanitize(&row.note)
        );
    }
    tsv
}

fn unique_in_order<T: PartialEq>(items: impl Iterator<Item = T>) -> Vec<T> {
    let mut seen: Vec<T> = Vec::new();
    for item in items {
        if !seen.contains(&item) {
            seen.push(item);
        }
    }
    seen
}

fn stage_markdown(rows: &[StageRow]) -> String {
    let mut table = String::from("# stage matrix\n\nCells are `mean (CoV%) n=rounds`; `-` means no measured value (see Unavailable).\n\n| engine | prompt |");
    for (title, _) in HEADLINES {
        let _ = write!(table, " {title} |");
    }
    let _ = write!(table, "\n|{}\n", "---|".repeat(HEADLINES.len() + 2));
    for (engine, prompt) in unique_in_order(rows.iter().map(|row| (row.engine.as_str(), row.prompt.as_str()))) {
        let _ = write!(table, "| {engine} | {prompt} |");
        for (_, name) in HEADLINES {
            let values: Vec<f64> = rows
                .iter()
                .filter(|row| row.engine == engine && row.prompt == prompt && row.metric == name)
                .filter_map(|row| row.value)
                .collect();
            let _ = write!(table, " {} |", stat_cell(&values));
        }
        table.push('\n');
    }
    table.push_str("\n## Definitions\n\n");
    for (stage, name, unit, definition) in unique_in_order(rows.iter().map(|row| (row.stage, row.metric, row.unit, row.definition))) {
        let _ = writeln!(table, "- `{name}` ({}, {unit}): {definition}", stage.name());
    }
    table.push_str("\n## Unavailable\n\n");
    for (engine, name, reason) in unique_in_order(rows.iter().filter(|row| row.value.is_none()).map(|row| (row.engine.as_str(), row.metric, row.note.as_str()))) {
        let _ = writeln!(table, "- {engine} `{name}`: {}", sanitize(reason));
    }
    table
}

struct StagePrompt {
    label: String,
    text: String,
    ids: Vec<u32>,
    capacity: usize,
}

fn load_stage_prompts(config: &Config, directory: &Path) -> Result<(Vocab, Vec<StagePrompt>), HarnessError> {
    let file = File::open(&config.model).map_err(io_error(&config.model))?;
    // SAFETY: the mapping is read-only and the blob is not modified while the harness runs.
    let bytes = unsafe { MmapOptions::new().map(&file) }.map_err(io_error(&config.model))?;
    let parsed = parse_complete(&bytes).map_err(|source| HarnessError::Tokenize(format!("{source:?}")))?;
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .map_err(|source| HarnessError::Tokenize(format!("{source:?}")))?;
    let mut prompts = Vec::new();
    for label in STAGE_PROMPT_LABELS {
        let path = directory.join(format!("wp_{label}.txt"));
        let text = fs::read_to_string(&path).map_err(io_error(&path))?.trim_end_matches('\n').to_owned();
        let ids = tokenize_like_proxima(&text, &vocab)?;
        let capacity = kv_capacity(ids.len(), MAX_TOKENS, config.kv_bucket, config.context_length);
        prompts.push(StagePrompt { label: label.to_owned(), text, ids, capacity });
    }
    Ok((vocab, prompts))
}

fn mean_call_ms<T>(mut action: impl FnMut() -> T) -> f64 {
    action();
    let started = Instant::now();
    for _ in 0..TEXT_EDGE_REPS {
        std::hint::black_box(action());
    }
    started.elapsed().as_secs_f64() * 1e3 / count_as_f64(TEXT_EDGE_REPS)
}

fn timed_exchange(port: u16, path: &str, body: &str) -> Result<(String, f64), HarnessError> {
    let started = Instant::now();
    let answer = http_exchange(port, "POST", path, body, Duration::from_secs(300))?;
    Ok((answer, started.elapsed().as_secs_f64() * 1e3))
}

fn json_array_len(body: &str, key: &str) -> Result<usize, HarnessError> {
    let value: Value = serde_json::from_str(body).map_err(|source| HarnessError::BadResponse(source.to_string()))?;
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| HarnessError::BadResponse(format!("missing {key}")))
}

fn text_completion_body(text: &str) -> String {
    json!({
        "prompt": text,
        "n_predict": MAX_TOKENS,
        "temperature": 0,
        "top_k": 1,
        "seed": 1,
        "cache_prompt": false,
        "ignore_eos": true,
    })
    .to_string()
}

fn llama_stage_requests(port: u16, prompt: &StagePrompt, ready_ms: f64) -> Result<LlamaStage, HarnessError> {
    let (answer, tokenize_wall_ms) = timed_exchange(port, "/tokenize", &json!({"content": prompt.text, "add_special": true}).to_string())?;
    let tokenize_tokens = json_array_len(&answer, "tokens")?;
    let (_, tokenize_floor_ms) = timed_exchange(port, "/tokenize", &json!({"content": ""}).to_string())?;
    let body = completion_body(&prompt.ids);
    let ask = || timed_exchange(port, "/completion", &body).and_then(|(text, _)| parse_llama_response(&text));
    let (cold, warm) = (ask()?, ask()?);
    require_prompt_count("llama cold", cold.prompt_n + cold.cache_n, prompt.ids.len())?;
    require_prompt_count("llama warm", warm.prompt_n + warm.cache_n, prompt.ids.len())?;
    let (_, detokenize_wall_ms) = timed_exchange(port, "/detokenize", &json!({"tokens": warm.ids}).to_string())?;
    let (_, detokenize_floor_ms) = timed_exchange(port, "/detokenize", &json!({"tokens": []}).to_string())?;
    let (answer, e2e_wall_ms) = timed_exchange(port, "/completion", &text_completion_body(&prompt.text))?;
    parse_llama_response(&answer)?;
    Ok(LlamaStage {
        cold,
        warm,
        tokenize_wall_ms,
        tokenize_floor_ms,
        tokenize_tokens,
        detokenize_wall_ms,
        detokenize_floor_ms,
        e2e_wall_ms,
        ready_ms,
        log: LlamaLog::default(),
        footprint_bytes: None,
        rss_bytes: None,
    })
}

fn run_llama_stage(config: &Config, prompt: &StagePrompt, port: u16, flash: &str, log_path: &Path) -> Result<LlamaStage, HarnessError> {
    let started = Instant::now();
    let server = TimedServer::spawn(config, prompt.capacity, config.stage_kv, port, log_path, flash)?;
    let outcome = if TimedServer::wait_ready(port, 180) {
        llama_stage_requests(port, prompt, started.elapsed().as_secs_f64() * 1e3)
    } else {
        Err(HarnessError::Server("never became healthy".to_owned()))
    };
    server.stop();
    let log_text = fs::read_to_string(log_path).map_err(io_error(log_path))?;
    Ok(LlamaStage {
        log: parse_llama_log(&log_text),
        footprint_bytes: parse_footprint_bytes(&log_text).ok(),
        rss_bytes: parse_rss_bytes(&log_text),
        ..outcome?
    })
}

fn require_prompt_count(engine: &'static str, reported: usize, sent: usize) -> Result<(), HarnessError> {
    if reported == sent {
        Ok(())
    } else {
        Err(HarnessError::PromptCountMismatch { engine, reported, sent })
    }
}

fn proxima_command(config: &Config, binary: &Path, text: &str, profile: Option<&str>) -> Command {
    let (runs, tokens) = match profile {
        Some(PROFILE_PREFILL_STEP) => ("1", 1),
        Some(_) => ("1", PROFILE_DECODE_TOKENS),
        None => ("3", MAX_TOKENS),
    };
    let mut command = Command::new(&config.time_bin);
    command.arg("-l").arg(binary);
    command.env("PROXIMA_SPECULATIVE_TYPES", "none");
    command.env("PROXIMA_STEP_TIMES", "1");
    command.env("PROXIMA_IGNORE_EOS", "1");
    command.env("PROXIMA_RUNS", runs);
    command.env("PROXIMA_MAX_TOKENS", tokens.to_string());
    command.env("PROXIMA_PROMPT", text);
    command.env("PROXIMA_DECODE_MODEL_GGUF", &config.model);
    for (key, value) in config.stage_kv.proxima_env() {
        command.env(key, value);
    }
    if let Some(step) = profile {
        command.env("PROXIMA_METAL_OP_PROFILE_STEP", step);
    }
    command
}

fn capture_proxima(config: &Config, binary: &Path, text: &str, profile: Option<&str>, log_path: &Path) -> Result<String, HarnessError> {
    let mut command = proxima_command(config, binary, text, profile);
    let output = command.output().map_err(|source| HarnessError::Server(format!("spawn failed: {source}")))?;
    let combined = format!("{}\n{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    fs::write(log_path, &combined).map_err(io_error(log_path))?;
    if output.status.success() {
        Ok(combined)
    } else {
        Err(HarnessError::Server(format!("{} exited {:?}", binary.display(), output.status.code())))
    }
}

fn run_proxima_stage(config: &Config, binary: &Path, vocab: &Vocab, prompt: &StagePrompt, logs: &Path, round: usize) -> Result<ProximaStage, HarnessError> {
    let log_for = |kind: &str| logs.join(format!("stage-r{round}-proxima-{}-{kind}.err", prompt.label));
    let main_log = capture_proxima(config, binary, &prompt.text, None, &log_for("main"))?;
    let profile = |step: &str, kind: &str| {
        capture_proxima(config, binary, &prompt.text, Some(step), &log_for(kind)).ok().and_then(|text| parse_op_profile(&text))
    };
    let attention_prefill = profile(PROFILE_PREFILL_STEP, "profile-prefill");
    let attention_decode = profile(PROFILE_DECODE_STEP, "profile-decode");
    let legacy = parse_proxima(&main_log)?;
    let span = parse_proxima_span(&main_log, &legacy).unwrap_or(legacy);
    let detail = parse_proxima_detail(&main_log)?;
    require_prompt_count("proxima", detail.prompt_token_count, prompt.ids.len())?;
    let generated: Vec<u32> = detail.ids.iter().filter_map(|&id| u32::try_from(id).ok()).collect();
    let tokenize_ms = mean_call_ms(|| tokenize_like_proxima(&prompt.text, vocab));
    let detokenize_ms = mean_call_ms(|| proxima_tokenizer::decode(&generated, vocab));
    Ok(ProximaStage {
        prompt_tokens: prompt.ids.len(),
        tokenize_ms,
        detokenize_ms,
        ttft_cold_ms: span.ttft_cold_ms.ok_or(HarnessError::MissingLine("cold prefill"))?,
        ttft_warm_ms: span.ttft_warm_ms.clone(),
        decode_ms: span.tok_s.iter().map(|rate| 1000.0 / rate).collect(),
        run_walls_ms: parse_run_walls(&main_log),
        process_real_ms: parse_real_ms(&main_log),
        gpu_peak_bytes: span.gpu_peak_bytes,
        footprint_bytes: parse_footprint_bytes(&main_log).ok(),
        rss_bytes: parse_rss_bytes(&main_log),
        kv: parse_kv_breakdown(&main_log),
        attention_prefill,
        attention_decode,
    })
}

const LLAMA_LIBRARY: &str = "/opt/homebrew/lib/libllama.dylib";
const TEXT_EDGE_RSS_LIMIT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// What the in-process vocab-only llama observed for the text edges of one prompt.
#[derive(Debug, Clone, PartialEq)]
struct LlamaTextEdges {
    tokenize_ms: f64,
    tokenize_tokens: usize,
    detokenize_ms: f64,
    rss_bytes: u64,
}

/// `struct llama_model_params` of llama b12110 (`llama.h:321-358`); C enums are `int` sized.
#[repr(C)]
struct LlamaModelParams {
    devices: *const c_void,
    tensor_buft_overrides: *const c_void,
    n_gpu_layers: i32,
    split_mode: i32,
    load_mode: i32,
    lazy_mode: i32,
    main_gpu: i32,
    tensor_split: *const f32,
    progress_callback: *const c_void,
    progress_callback_user_data: *mut c_void,
    kv_overrides: *const c_void,
    vocab_only: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
    load_mtp: bool,
}

type LlamaBackendInit = unsafe extern "C" fn();
type LlamaDefaultParams = unsafe extern "C" fn() -> LlamaModelParams;
type LlamaLoad = unsafe extern "C" fn(*const c_char, LlamaModelParams) -> *mut c_void;
type LlamaFree = unsafe extern "C" fn(*mut c_void);
type LlamaGetVocab = unsafe extern "C" fn(*const c_void) -> *const c_void;
type LlamaTokenize = unsafe extern "C" fn(*const c_void, *const c_char, i32, *mut i32, i32, bool, bool) -> i32;
type LlamaDetokenize = unsafe extern "C" fn(*const c_void, *const i32, i32, *mut c_char, i32, bool, bool) -> i32;

/// libllama taken with `dlopen` at run time: bench-only, no build or link dependency.
struct LlamaApi {
    backend_init: LlamaBackendInit,
    default_params: LlamaDefaultParams,
    load: LlamaLoad,
    free: LlamaFree,
    get_vocab: LlamaGetVocab,
    tokenize: LlamaTokenize,
    detokenize: LlamaDetokenize,
}

fn llama_symbol<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, HarnessError> {
    let c_name = CString::new(name).map_err(|source| HarnessError::Server(source.to_string()))?;
    // SAFETY: handle came from a successful dlopen and c_name is NUL terminated.
    let address = unsafe { libc::dlsym(handle, c_name.as_ptr()) };
    if address.is_null() {
        return Err(HarnessError::Server(format!("libllama has no symbol {name}")));
    }
    // SAFETY: T is one of the extern "C" fn pointer aliases above, pointer sized, matching llama.h for this build.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&address) })
}

impl LlamaApi {
    fn open(library: &str) -> Result<Self, HarnessError> {
        let c_path = CString::new(library).map_err(|source| HarnessError::Server(source.to_string()))?;
        // SAFETY: c_path is NUL terminated; RTLD_NOW resolves every symbol before returning.
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW) };
        if handle.is_null() {
            return Err(HarnessError::Server(format!("dlopen {library} failed")));
        }
        Ok(Self {
            backend_init: llama_symbol(handle, "llama_backend_init")?,
            default_params: llama_symbol(handle, "llama_model_default_params")?,
            load: llama_symbol(handle, "llama_model_load_from_file")?,
            free: llama_symbol(handle, "llama_model_free")?,
            get_vocab: llama_symbol(handle, "llama_model_get_vocab")?,
            tokenize: llama_symbol(handle, "llama_tokenize")?,
            detokenize: llama_symbol(handle, "llama_detokenize")?,
        })
    }

    fn load_vocab_only(&self, model: &Path) -> Result<LlamaVocabModel<'_>, HarnessError> {
        let c_path = CString::new(model.as_os_str().as_encoded_bytes()).map_err(|source| HarnessError::Server(source.to_string()))?;
        // SAFETY: symbols were resolved from this build's llama.h; the params struct mirrors it.
        let handle = unsafe {
            (self.backend_init)();
            let mut params = (self.default_params)();
            params.vocab_only = true;
            params.n_gpu_layers = 0;
            (self.load)(c_path.as_ptr(), params)
        };
        if handle.is_null() {
            return Err(HarnessError::Server(format!("llama_model_load_from_file failed for {}", model.display())));
        }
        // SAFETY: handle is the live model just loaded.
        let vocab = unsafe { (self.get_vocab)(handle) };
        Ok(LlamaVocabModel { api: self, handle, vocab })
    }
}

struct LlamaVocabModel<'api> {
    api: &'api LlamaApi,
    handle: *mut c_void,
    vocab: *const c_void,
}

impl Drop for LlamaVocabModel<'_> {
    fn drop(&mut self) {
        // SAFETY: handle was returned by llama_model_load_from_file and is freed exactly once.
        unsafe { (self.api.free)(self.handle) };
    }
}

impl LlamaVocabModel<'_> {
    /// a negative return is the needed size: grow once, then call again
    fn tokenize(&self, text: &str, tokens: &mut Vec<i32>) -> Result<usize, HarnessError> {
        let length = i32::try_from(text.len()).map_err(|source| HarnessError::Tokenize(source.to_string()))?;
        for _ in 0..2 {
            let capacity = i32::try_from(tokens.len()).unwrap_or(i32::MAX);
            // SAFETY: text and tokens are live for the call, lengths passed are their real lengths.
            let count = unsafe { (self.api.tokenize)(self.vocab, text.as_ptr().cast(), length, tokens.as_mut_ptr(), capacity, true, false) };
            if count >= 0 {
                return Ok(count as usize);
            }
            if count == i32::MIN {
                return Err(HarnessError::Tokenize("llama_tokenize overflow".to_owned()));
            }
            tokens.resize(count.unsigned_abs() as usize, 0);
        }
        Err(HarnessError::Tokenize("llama_tokenize size query did not converge".to_owned()))
    }

    fn detokenize(&self, ids: &[i32], text: &mut Vec<u8>) -> Result<usize, HarnessError> {
        let count = i32::try_from(ids.len()).map_err(|source| HarnessError::Tokenize(source.to_string()))?;
        for _ in 0..2 {
            let capacity = i32::try_from(text.len()).unwrap_or(i32::MAX);
            // SAFETY: ids and text are live for the call, lengths passed are their real lengths.
            let written = unsafe { (self.api.detokenize)(self.vocab, ids.as_ptr(), count, text.as_mut_ptr().cast(), capacity, false, false) };
            if written >= 0 {
                return Ok(written as usize);
            }
            text.resize(written.unsigned_abs() as usize, 0);
        }
        Err(HarnessError::Tokenize("llama_detokenize size query did not converge".to_owned()))
    }
}

fn own_rss_bytes() -> Option<u64> {
    let output = Command::new("ps").args(["-o", "rss=", "-p", &std::process::id().to_string()]).output().ok()?;
    let kib: u64 = String::from_utf8_lossy(&output.stdout).trim().parse().ok()?;
    Some(kib * 1024)
}

fn run_llama_text_edges(config: &Config, prompt: &StagePrompt) -> Result<LlamaTextEdges, HarnessError> {
    let api = LlamaApi::open(LLAMA_LIBRARY)?;
    let model = api.load_vocab_only(&config.model)?;
    let rss_bytes = own_rss_bytes().ok_or_else(|| HarnessError::Server("could not read the harness rss".to_owned()))?;
    if rss_bytes > TEXT_EDGE_RSS_LIMIT_BYTES {
        return Err(rss_error("after the vocab-only load", rss_bytes));
    }
    let ids: Vec<i32> = prompt.ids.iter().map(|&id| id as i32).collect();
    let mut tokens = Vec::new();
    let tokenize_tokens = model.tokenize(&prompt.text, &mut tokens)?;
    let mut text = Vec::new();
    model.detokenize(&ids, &mut text)?;
    let mut failure = None;
    let tokenize_ms = mean_call_ms(|| model.tokenize(&prompt.text, &mut tokens).map_err(|error| failure = Some(error)));
    let detokenize_ms = mean_call_ms(|| model.detokenize(&ids, &mut text).map_err(|error| failure = Some(error)));
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(LlamaTextEdges { tokenize_ms, tokenize_tokens, detokenize_ms, rss_bytes })
}

fn llama_text_edge_rows(sink: &mut RowSink, input: &LlamaTextEdges) {
    sink.measured(&TOKENIZE_INPROC_MS, input.tokenize_ms, "vocab_only model, no transport");
    sink.measured(&TOKENIZE_INPROC_TOKENS, count_as_f64(input.tokenize_tokens), "");
    sink.measured(&DETOKENIZE_PROMPT_INPROC_MS, input.detokenize_ms, "vocab_only model, prompt ids");
    sink.measured(&VOCAB_ONLY_RSS_BYTES, bytes_as_f64(input.rss_bytes), "ps rss of the harness after the vocab-only load");
}

fn rss_error(stage: &str, bytes: u64) -> HarnessError {
    HarnessError::Server(format!("text-edge server rss {bytes} bytes exceeds {TEXT_EDGE_RSS_LIMIT_BYTES} {stage}; weights were faulted in"))
}

/// proxima's decode into a caller-owned `String` (`decode_into`), allocated before the timed closure, matching the llama arm's reused output buffer.
fn detokenize_prompt_ms(ids: &[u32], vocab: &Vocab) -> f64 {
    let mut text = String::with_capacity(ids.len() * 8);
    mean_call_ms(|| proxima_tokenizer::decode_into(ids, vocab, &mut text))
}

fn proxima_text_edge_rows(sink: &mut RowSink, tokenize_ms: f64, detokenize_prompt_ms: f64, prompt_tokens: usize) {
    sink.measured(&TOKENIZE_INPROC_MS, tokenize_ms, "GGUF vocab, no transport");
    sink.measured(&TOKENIZE_INPROC_TOKENS, count_as_f64(prompt_tokens), "");
    sink.measured(&DETOKENIZE_PROMPT_INPROC_MS, detokenize_prompt_ms, "GGUF vocab, prompt ids");
}

fn text_edge_round(config: &Config, binary: &(String, PathBuf), vocab: &Vocab, prompt: &StagePrompt, round: usize) -> Vec<StageRow> {
    let load = load1();
    let outcome = if quiet_enough(config) {
        run_llama_text_edges(config, prompt)
    } else {
        Err(HarnessError::Server("quiet gate refused".to_owned()))
    };
    let mut rows = match outcome {
        Ok(edges) => {
            let mut sink = RowSink::new(round, "llama", &prompt.label, load);
            llama_text_edge_rows(&mut sink, &edges);
            sink.rows
        }
        Err(error) => error_rows(round, "llama", &prompt.label, load, &error),
    };
    let mut sink = RowSink::new(round, &format!("proxima-{}", binary.0), &prompt.label, load);
    let tokenize_ms = mean_call_ms(|| tokenize_like_proxima(&prompt.text, vocab));
    let detokenize_ms = detokenize_prompt_ms(&prompt.ids, vocab);
    proxima_text_edge_rows(&mut sink, tokenize_ms, detokenize_ms, prompt.ids.len());
    rows.extend(sink.rows);
    rows
}

fn error_rows(round: usize, engine: &str, prompt: &str, load: f64, error: &HarnessError) -> Vec<StageRow> {
    let mut sink = RowSink::new(round, engine, prompt, load);
    sink.unavailable(&ARM_ERROR, &error.to_string());
    sink.rows
}

fn quiet_enough(config: &Config) -> bool {
    !config.quiet_gate || wait_for_quiet(&config.gate).may_run()
}

type ArmKeys = Vec<(&'static str, String)>;

const CACHED_NOTE: &str = "cached";

const KNOWN_METRICS: &[&Metric] = &[
    &TOKENIZE_MS, &TOKENIZE_WALL_MS, &TOKENIZE_FLOOR_MS, &TOKENIZE_TOKENS, &PREFILL_MS, &PREFILL_COLD_MS, &DECODE_MS,
    &ATTN_PREFILL_MS, &ATTN_PREFILL_SHARE, &ATTN_DECODE_MS, &ATTN_DECODE_SHARE, &FA_PREFILL_PROXY, &FA_DECODE_PROXY,
    &KV_BYTES_PER_TOKEN, &KV_PEAK_BYTES, &KV_WRITE_MS, &DETOKENIZE_MS, &DETOKENIZE_WALL_MS, &DETOKENIZE_FLOOR_MS,
    &TOKENIZE_INPROC_MS, &TOKENIZE_INPROC_TOKENS, &DETOKENIZE_PROMPT_INPROC_MS, &VOCAB_ONLY_RSS_BYTES, &E2E_WALL_MS,
    &E2E_EDGES_MS, &LOAD_MS, &FIRST_REQUEST_MS, &FOOTPRINT_BYTES, &RSS_BYTES, &GPU_PEAK_BYTES, &ARM_ERROR,
];

fn metric_by_name(name: &str) -> Option<&'static Metric> {
    KNOWN_METRICS.iter().copied().find(|known| known.name == name)
}

/// Model identity without hashing 25 GB: canonical path, length and mtime in nanoseconds, plus the stage KV kind.
fn model_stamp(model: &Path, kv: Kv) -> Result<String, HarnessError> {
    let canonical = fs::canonicalize(model).map_err(io_error(model))?;
    let metadata = fs::metadata(&canonical).map_err(io_error(&canonical))?;
    let modified = metadata.modified().map_err(io_error(&canonical))?;
    let nanos = modified.duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_nanos());
    Ok(format!("{}|{}|{nanos}|{}", canonical.display(), metadata.len(), kv.name()))
}

fn cache_key(stamp: &str, commit: &str, ids: &[u32], flash: &str, engine: &str, round: usize) -> String {
    let id_text = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    sha256_hex(format!("{stamp}\n{commit}\n{id_text}\n{flash}\n{engine}\n{MAX_TOKENS}\n{round}").as_bytes())
}

fn incumbent_arms(config: &Config) -> Vec<(&'static str, &'static str)> {
    if config.fa_off_arm {
        vec![("llama", "on"), ("llama-fa-off", "off"), ("llama-fa-delta", "on+off")]
    } else {
        vec![("llama", "on")]
    }
}

fn incumbent_keys(config: &Config, prompt: &StagePrompt, round: usize) -> ArmKeys {
    if config.incumbent_cache.is_none() {
        return Vec::new();
    }
    match model_stamp(&config.model, config.stage_kv) {
        Ok(stamp) => incumbent_arms(config)
            .into_iter()
            .map(|(engine, flash)| (engine, cache_key(&stamp, &config.llama_commit, &prompt.ids, flash, engine, round)))
            .collect(),
        Err(error) => {
            eprintln!("kv_scoreboard: incumbent cache disabled: {error}");
            Vec::new()
        }
    }
}

fn cache_tsv(rows: &[StageRow]) -> String {
    let mut tsv = format!("{STAGE_HEADER}\n");
    for row in rows {
        let (value, status) = row.value.map_or((String::new(), "unavailable"), |found| (found.to_string(), "measured"));
        let _ = writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{value}\t{}\t{status}\t{}\t{}\t{}",
            row.round, row.engine, row.prompt, row.stage.name(), row.metric, row.unit, row.load1, row.definition, sanitize(&row.note)
        );
    }
    tsv
}

fn cached_row(line: &str, round: usize, prompt: &str, file: &str) -> Option<StageRow> {
    let columns: Vec<&str> = line.splitn(11, '\t').collect();
    let [_, engine, _, _, name, value, _, _, load1, _, note] = columns.as_slice() else {
        return None;
    };
    let spec = metric_by_name(name)?;
    let value = if value.is_empty() { None } else { Some(value.parse().ok()?) };
    let note = if note.is_empty() { format!("{CACHED_NOTE} {file}") } else { format!("{CACHED_NOTE} {file}: {note}") };
    Some(StageRow {
        round,
        engine: (*engine).to_owned(),
        prompt: prompt.to_owned(),
        stage: spec.stage,
        metric: spec.name,
        value,
        unit: spec.unit,
        definition: spec.definition,
        load1: load1.parse().ok()?,
        note,
    })
}

fn rows_from_cache_tsv(text: &str, round: usize, prompt: &str, file: &str) -> Option<Vec<StageRow>> {
    let mut lines = text.lines();
    if lines.next()? != STAGE_HEADER {
        return None;
    }
    let rows: Option<Vec<StageRow>> = lines.map(|line| cached_row(line, round, prompt, file)).collect();
    rows.filter(|found| !found.is_empty())
}

fn lookup_cached(directory: &Path, key: &str, round: usize, prompt: &str) -> Option<Vec<StageRow>> {
    let file = format!("{key}.tsv");
    let text = fs::read_to_string(directory.join(&file)).ok()?;
    rows_from_cache_tsv(&text, round, prompt, &file)
}

fn store_cached(directory: &Path, key: &str, rows: &[StageRow]) -> Result<(), HarnessError> {
    fs::create_dir_all(directory).map_err(io_error(directory))?;
    let path = directory.join(format!("{key}.tsv"));
    fs::write(&path, cache_tsv(rows)).map_err(io_error(&path))
}

fn cached_incumbent(config: &Config, prompt: &StagePrompt, round: usize, keys: &ArmKeys) -> Option<Vec<StageRow>> {
    let directory = config.incumbent_cache.as_deref().filter(|_| !config.refresh_incumbent && !keys.is_empty())?;
    let mut rows = Vec::new();
    for (_, key) in keys {
        rows.extend(lookup_cached(directory, key, round, &prompt.label)?);
    }
    Some(rows)
}

fn remember_incumbent(config: &Config, keys: &ArmKeys, engine: &str, rows: &[StageRow]) {
    let Some(directory) = config.incumbent_cache.as_deref() else {
        return;
    };
    for (_, key) in keys.iter().filter(|(name, _)| *name == engine) {
        if let Err(error) = store_cached(directory, key, rows) {
            eprintln!("kv_scoreboard: incumbent cache write failed: {error}");
        }
    }
}

fn fresh_incumbent(config: &Config, prompt: &StagePrompt, logs: &Path, round: usize, keys: &ArmKeys) -> Vec<StageRow> {
    let mut rows = Vec::new();
    let load = load1();
    let slot = STAGE_PROMPT_LABELS.iter().position(|label| *label == prompt.label).unwrap_or(0);
    let port = STAGE_BASE_PORT + u16::try_from(round * 8 + slot).unwrap_or(0);
    let flashes: &[(&str, &str)] = if config.fa_off_arm { &[("llama", "on"), ("llama-fa-off", "off")] } else { &[("llama", "on")] };
    let mut arms: Vec<(&str, Result<LlamaStage, HarnessError>)> = Vec::new();
    for &(engine, flash) in flashes {
        let log_path = logs.join(format!("stage-r{round}-{engine}-{}.err", prompt.label));
        let outcome = if quiet_enough(config) {
            run_llama_stage(config, prompt, port, flash, &log_path)
        } else {
            Err(HarnessError::Server("quiet gate refused".to_owned()))
        };
        arms.push((engine, outcome));
    }
    for (engine, outcome) in &arms {
        match outcome {
            Ok(stage) => {
                let mut sink = RowSink::new(round, engine, &prompt.label, load);
                llama_rows(&mut sink, stage);
                remember_incumbent(config, keys, engine, &sink.rows);
                rows.extend(sink.rows);
            }
            Err(error) => rows.extend(error_rows(round, engine, &prompt.label, load, error)),
        }
    }
    if let [(_, Ok(on)), (_, Ok(off))] = arms.as_slice() {
        let mut sink = RowSink::new(round, "llama-fa-delta", &prompt.label, load);
        fa_delta_rows(&mut sink, on, off);
        remember_incumbent(config, keys, "llama-fa-delta", &sink.rows);
        rows.extend(sink.rows);
    }
    rows
}

fn stage_round(config: &Config, binary: &(String, PathBuf), vocab: &Vocab, prompt: &StagePrompt, logs: &Path, round: usize) -> Vec<StageRow> {
    let load = load1();
    let keys = incumbent_keys(config, prompt, round);
    let mut rows = cached_incumbent(config, prompt, round, &keys).unwrap_or_else(|| fresh_incumbent(config, prompt, logs, round, &keys));
    let engine = format!("proxima-{}", binary.0);
    let outcome = if quiet_enough(config) {
        run_proxima_stage(config, &binary.1, vocab, prompt, logs, round)
    } else {
        Err(HarnessError::Server("quiet gate refused".to_owned()))
    };
    match outcome {
        Ok(stage) => {
            let mut sink = RowSink::new(round, &engine, &prompt.label, load);
            proxima_rows(&mut sink, &stage);
            rows.extend(sink.rows);
        }
        Err(error) => rows.extend(error_rows(round, &engine, &prompt.label, load, &error)),
    }
    rows
}

fn run_stages(config: &Config, directory: &Path) -> Result<(), HarnessError> {
    let logs = config.out.join("logs");
    fs::create_dir_all(&logs).map_err(io_error(&logs))?;
    let (vocab, prompts) = load_stage_prompts(config, directory)?;
    let binary = config.proxima.first().ok_or(HarnessError::MissingFlag("--prox"))?;
    let mut rows = Vec::new();
    for round in 0..config.rounds {
        for prompt in &prompts {
            rows.extend(if config.text_edges {
                text_edge_round(config, binary, &vocab, prompt, round)
            } else {
                stage_round(config, binary, &vocab, prompt, &logs, round)
            });
            println!("stage round {round} prompt {} rows {}", prompt.label, rows.len());
        }
    }
    let tsv_path = config.out.join("stages.tsv");
    fs::write(&tsv_path, stage_tsv(&rows)).map_err(io_error(&tsv_path))?;
    let table_path = config.out.join("stages.md");
    let table = stage_markdown(&rows);
    fs::write(&table_path, &table).map_err(io_error(&table_path))?;
    println!("{table}");
    Ok(())
}

fn main() {
    let outcome = parse_config(std::env::args().skip(1)).and_then(|config| match &config.stages {
        Some(directory) => run_stages(&config, directory),
        None => run(&config),
    });
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

    const PROX_IDS_LINE: &str = "decode_gbps_baseline token_ids run_index=2 ids=[236776, 5213, 20619, 4408, 3788, 1018, 563, 7100, 528, 2425, 1082, 496, 50205, 64981, 55569, 236761]";

    const LLAMA_RESPONSE: &str = include_str!("fixtures/llama_server_completion_short.json");
    const LLAMA_START_LOG: &str = include_str!("fixtures/llama_server_start_bf16.log");
    const LLAMA_PEAK: &str = "           952119104  peak memory footprint\n";

    fn close(left: Option<f64>, right: f64) -> bool {
        left.is_some_and(|value| (value - right).abs() < 1e-6)
    }

    const LLAMA_F32_84_RESPONSE: &str = include_str!("fixtures/llama_server_completion_f32_84.json");
    const OP_PROFILE_LINES: &str = "\
op_profile_kind step=3 kind=cached_attention op_count=35 gpu_ms=1.500 operand_bytes=1024
op_profile_kind step=3 kind=matmul op_count=210 gpu_ms=6.000 operand_bytes=4096
op_profile_kind step=3 kind=cached_attention_merge op_count=35 gpu_ms=0.500 operand_bytes=64
";

    fn llama_stage_from_real_response() -> LlamaStage {
        let response = parse_llama_response(LLAMA_F32_84_RESPONSE).expect("real llama completion parses");
        LlamaStage {
            cold: response.clone(),
            warm: response,
            tokenize_wall_ms: 2.5,
            tokenize_floor_ms: 0.5,
            tokenize_tokens: 84,
            detokenize_wall_ms: 3.0,
            detokenize_floor_ms: 0.75,
            e2e_wall_ms: 3300.0,
            ready_ms: 4000.0,
            log: parse_llama_log(LLAMA_START_LOG),
            footprint_bytes: Some(952_119_104),
            rss_bytes: None,
        }
    }

    fn rows_of(sink: &RowSink, metric: &str) -> Vec<StageRow> {
        sink.rows.iter().filter(|row| row.metric == metric).cloned().collect()
    }

    #[test]
    fn llama_stage_rows_use_the_server_timings_definitions_and_subtract_the_http_floor() {
        let mut sink = RowSink::new(0, "llama", "84", 1.5);
        llama_rows(&mut sink, &llama_stage_from_real_response());
        let prefill = rows_of(&sink, "prefill_ms_per_prompt_token");
        assert!(close(prefill[0].value, 374.526 / 84.0));
        let decode = rows_of(&sink, "decode_ms_per_token");
        assert!(close(decode[0].value, 25.15844094488189));
        assert!(close(rows_of(&sink, "tokenize_ms")[0].value, 2.0));
        assert!(close(rows_of(&sink, "detokenize_ms")[0].value, 2.25));
        assert!(close(rows_of(&sink, "first_request_compile_ms")[0].value, 0.0));
    }

    #[test]
    fn llama_attention_cells_are_unavailable_and_say_why() {
        let mut sink = RowSink::new(0, "llama", "84", 1.5);
        llama_rows(&mut sink, &llama_stage_from_real_response());
        let attention: Vec<&StageRow> = sink.rows.iter().filter(|row| row.stage == Stage::Attention).collect();
        assert_eq!(attention.len(), 4);
        assert!(attention.iter().all(|row| row.value.is_none() && row.note.contains("no per-op timer")));
    }

    #[test]
    fn stage_tsv_has_one_row_per_metric_with_header_status_and_definition() {
        let mut sink = RowSink::new(2, "llama", "84", 1.5);
        llama_rows(&mut sink, &llama_stage_from_real_response());
        let tsv = stage_tsv(&sink.rows);
        let lines: Vec<&str> = tsv.lines().collect();
        assert_eq!(lines[0], STAGE_HEADER);
        assert_eq!(lines.len(), sink.rows.len() + 1);
        let columns = STAGE_HEADER.split('\t').count();
        assert!(lines.iter().all(|line| line.split('\t').count() == columns));
        let unavailable = lines.iter().filter(|line| line.contains("\tunavailable\t")).count();
        assert!(unavailable >= 6, "attention x4, kv write and gpu peak must be unavailable, got {unavailable}");
        assert!(lines.iter().skip(1).all(|line| line.split('\t').nth(9).is_some_and(|definition| !definition.is_empty())));
    }

    #[test]
    fn stage_markdown_orders_columns_like_the_owner_and_lists_definitions() {
        let mut sink = RowSink::new(0, "llama", "84", 1.5);
        llama_rows(&mut sink, &llama_stage_from_real_response());
        let table = stage_markdown(&sink.rows);
        let header = table.lines().find(|line| line.starts_with("| engine")).expect("table header");
        let position = |title: &str| header.find(title).expect("column present");
        assert!(position("tokenize ms") < position("prefill ms/tok"));
        assert!(position("prefill ms/tok") < position("decode ms/tok"));
        assert!(position("decode ms/tok") < position("attn prefill gpu ms"));
        assert!(position("attn decode share") < position("kv bytes/tok"));
        assert!(position("kv write ms/step") < position("detokenize ms"));
        assert!(position("detokenize ms") < position("ete wall ms"));
        assert!(position("ete wall ms") < position("load ms"));
        assert!(table.contains("## Definitions"));
        assert!(table.contains("`tokenize_ms` (tokenization, ms)"));
        assert!(table.contains("llama `attention_prefill_gpu_ms`"));
    }

    const FUSED_2024: &str = include_str!("fixtures/op_profile_flint_fused_2024.err");
    const UNFUSED_2024: &str = include_str!("fixtures/op_profile_flint_unfused_2024.err");
    const KV_EVENTS: &str = include_str!("fixtures/kv_write_events_flint_device_kv.log");

    fn kind_line(text: &str, kind: &str) -> (usize, f64) {
        let line = text
            .lines()
            .find(|line| line.contains(&format!("kind={kind} ")))
            .expect("kind present in the first profile block");
        (
            field_value(line, "op_count=").and_then(|token| token.parse().ok()).expect("op_count"),
            field_value(line, "gpu_ms=").and_then(|token| token.parse().ok()).expect("gpu_ms"),
        )
    }

    #[test]
    fn real_fused_2024_prefill_profile_reads_the_first_block_only() {
        let profile = parse_op_profile(FUSED_2024).expect("real flint profile parses");
        assert!((profile.total_ms - 2678.585).abs() < 0.1, "total {}", profile.total_ms);
        let (op_count, attention_ms) = kind_line(FUSED_2024, "cached_attention");
        assert_eq!(op_count, 49);
        assert!((profile.attention_ms - attention_ms).abs() < 1e-9);
        assert!((353.0..355.0).contains(&profile.attention_ms), "attention {}", profile.attention_ms);
        let share = profile.attention_ms / profile.total_ms;
        assert!((share - 0.13).abs() < 0.01, "share {share}");
    }

    #[test]
    fn real_unfused_2024_prefill_profile_reads_the_first_block_only() {
        let profile = parse_op_profile(UNFUSED_2024).expect("real unfused profile parses");
        assert!((profile.total_ms - 12981.844).abs() < 1.0, "total {}", profile.total_ms);
        assert!(profile.attention_ms > 0.0 && profile.attention_ms < profile.total_ms);
    }

    #[test]
    fn kv_write_adds_device_kv_narrow_to_the_step_it_precedes() {
        let breakdown = parse_kv_breakdown(KV_EVENTS).expect("real events parse");
        let expected = [
            5.17525 + 0.001 + 0.226875,
            0.10975 + 0.001208 + 0.050458,
            0.107875 + 0.001541 + 0.530291,
            3.307916 + 0.000833 + 0.062291,
            0.104208 + 0.034416 + 0.1425,
            0.326375 + 0.001125 + 0.048125,
        ];
        let mean = expected.iter().sum::<f64>() / 6.0;
        assert!((breakdown.write_ms - mean).abs() < 1e-6, "write {}", breakdown.write_ms);
    }

    #[test]
    fn llama_b12110_tokenize_and_detokenize_bodies_match_server_field_names() {
        let tokenize = json!({"tokens": [2, 651, 6037]}).to_string();
        assert_eq!(json_array_len(&tokenize, "tokens").expect("tokens array"), 3);
        let empty = json!({"tokens": []}).to_string();
        assert_eq!(json_array_len(&empty, "tokens").expect("empty tokens array"), 0);
        assert!(json_array_len(&json!({"content": "hello"}).to_string(), "tokens").is_err());
    }

    #[test]
    fn op_profile_counts_every_attention_kind_and_shares_over_the_whole_step() {
        let profile = parse_op_profile(OP_PROFILE_LINES).expect("op profile lines parse");
        assert!((profile.attention_ms - 2.0).abs() < 1e-9);
        assert!((profile.total_ms - 8.0).abs() < 1e-9);
        assert!(parse_op_profile("no profile lines here").is_none());
    }

    #[test]
    fn proxima_rows_derive_load_and_first_request_from_the_real_run_log() {
        let legacy = parse_proxima(PROX_LOG).expect("real proxima log parses");
        let span = parse_proxima_span(PROX_LOG, &legacy).unwrap_or(legacy);
        let input = ProximaStage {
            prompt_tokens: 26,
            tokenize_ms: 0.1,
            detokenize_ms: 0.05,
            ttft_cold_ms: span.ttft_cold_ms.expect("cold prefill"),
            ttft_warm_ms: span.ttft_warm_ms.clone(),
            decode_ms: span.tok_s.iter().map(|rate| 1000.0 / rate).collect(),
            run_walls_ms: parse_run_walls(PROX_LOG),
            process_real_ms: parse_real_ms(PROX_LOG),
            gpu_peak_bytes: span.gpu_peak_bytes,
            footprint_bytes: parse_footprint_bytes(PROX_LOG).ok(),
            rss_bytes: parse_rss_bytes(PROX_LOG),
            kv: None,
            attention_prefill: None,
            attention_decode: parse_op_profile(OP_PROFILE_LINES),
        };
        let mut sink = RowSink::new(0, "proxima-new", "84", 1.5);
        proxima_rows(&mut sink, &input);
        let load = rows_of(&sink, "model_load_ms")[0].value.expect("load derived");
        let walls: f64 = input.run_walls_ms.values().sum();
        assert!((load - (7530.0 - walls)).abs() < 1e-6);
        assert!(rows_of(&sink, "attention_prefill_share")[0].value.is_none());
        assert!(close(rows_of(&sink, "attention_decode_share")[0].value, 0.25));
        assert!(rows_of(&sink, "kv_write_ms_per_step")[0].note.contains("token_breakdown"));
        assert_eq!(input.run_walls_ms.len(), 3);
    }

    #[test]
    fn flash_attention_delta_is_off_minus_on() {
        let on = llama_stage_from_real_response();
        let mut off = on.clone();
        off.warm.predicted_per_token_ms += 4.0;
        let mut sink = RowSink::new(0, "llama-fa-delta", "84", 1.5);
        fa_delta_rows(&mut sink, &on, &off);
        assert!(close(rows_of(&sink, "fa_off_minus_on_decode_ms_per_token")[0].value, 4.0));
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
    fn proxnum250_run_done_line_reports_85_ids_for_an_84_token_prompt_and_is_refused() {
        let log = include_str!("../tests/fixtures/proxnum250_proxima_84_run_done.log");
        let detail = parse_proxima_detail(log).expect("proxnum250 log parses");
        assert_eq!(detail.prompt_token_count, 85);
        let refusal = require_prompt_count("proxima", detail.prompt_token_count, 84).expect_err("85 != 84 must be refused");
        assert_eq!(refusal.to_string(), "proxima counted 85 prompt tokens, the tokenizer sent 84: it ran another model or prompt, not a measurement");
        assert!(require_prompt_count("proxima", 84, 84).is_ok());
    }

    #[test]
    fn proxima_detail_reads_the_last_runs_ids_count_and_displayed_prompt_count() {
        let log = format!("{PROX_LOG}\n{PROX_IDS_LINE}");
        let detail = parse_proxima_detail(&log).expect("ids line and run=done parse");
        assert_eq!(detail.ids.len(), 16);
        assert_eq!(detail.ids[0], 236_776);
        assert_eq!(detail.tokens_generated, 128);
        assert_eq!(detail.prompt_token_count, 26);
        assert!(parse_proxima_detail(PROX_LOG).is_err());
    }

    #[test]
    fn llama_server_response_yields_timings_ids_and_sampler_settings() {
        let response = parse_llama_response(LLAMA_RESPONSE).expect("real llama-server response parses");
        assert_eq!(response.prompt_n, 27);
        assert_eq!(response.cache_n, 0);
        assert!((response.prompt_ms - 114.585).abs() < 1e-9);
        assert_eq!(response.predicted_n, 16);
        assert!((response.predicted_per_second - 75.22152739818766).abs() < 1e-9);
        assert_eq!(response.ids.len(), 16);
        assert_eq!(response.ids[0], 236_776);
        assert_eq!((response.top_k, response.seed), (1, 1));
        assert_eq!(response.temperature, 0.0);
        assert!(response.model.ends_with("3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd"));
    }

    #[test]
    fn llama_server_response_missing_timings_is_an_error() {
        assert!(parse_llama_response("{\"content\":\"x\",\"tokens\":[1]}").is_err());
        assert!(parse_llama_response("not json").is_err());
    }

    #[test]
    fn llama_start_log_yields_n_ctx_flash_attention_kv_types_and_buffer_size() {
        let log = parse_llama_log(LLAMA_START_LOG);
        assert_eq!(log.n_ctx_seq, Some(256));
        assert_eq!(log.flash_attn, Some(true));
        assert_eq!(log.build, Some((11_671, "ba3c05f71".to_owned())));
        assert_eq!(log.eog_ids, vec![1, 50, 106]);
        assert!((log.kv_buffer_mib - 4.5).abs() < 1e-9);
        assert_eq!(log.kv_types.len(), 2);
        assert!(log.kv_types.iter().all(|(key, value)| key == "bf16" && value == "bf16"));
    }

    #[test]
    fn footprint_line_from_time_l_is_bytes() {
        assert_eq!(parse_footprint_bytes(PROX_LOG).expect("footprint"), 355_363_520);
        assert_eq!(parse_footprint_bytes(LLAMA_PEAK).expect("footprint"), 952_119_104);
        assert!(parse_footprint_bytes("7.53 real 2.90 user").is_err());
    }

    #[test]
    fn kv_capacity_matches_the_logged_decode_kv_bound_values() {
        let events = include_str!("fixtures/decode_kv_bound_events.txt");
        let logged: Vec<usize> = events
            .lines()
            .filter_map(|line| line.split("kv_capacity=").nth(1))
            .filter_map(|rest| rest.split_whitespace().next()?.parse().ok())
            .collect();
        assert!(logged.contains(&2720), "long-prompt kv_capacity is in the log");
        assert_eq!(kv_capacity(2565, MAX_TOKENS, DEFAULT_KV_BUCKET, DEFAULT_CONTEXT_LENGTH), 2720);
        assert_eq!(kv_capacity(27, MAX_TOKENS, DEFAULT_KV_BUCKET, DEFAULT_CONTEXT_LENGTH), 160);
    }

    #[test]
    fn kv_capacity_is_capped_by_the_context_length_and_llama_pads_to_256() {
        assert_eq!(kv_capacity(27, MAX_TOKENS, DEFAULT_KV_BUCKET, 128), 128);
        assert_eq!(llama_padded_capacity(160), 256);
        assert_eq!(llama_padded_capacity(2720), 2816);
        assert_eq!(llama_padded_capacity(256), 256);
    }

    fn gate_config() -> GateConfig {
        GateConfig { load_limit: 4.0, hold_secs: 30, wait_limit_secs: 120 }
    }

    #[test]
    fn gate_clears_only_after_the_hold_window_of_continuous_quiet() {
        let mut state = QuietState { started: 0, quiet_since: None };
        assert_eq!(gate_step(&mut state, 0, 1.0, &[], &gate_config()), GateStep::Waiting);
        assert_eq!(gate_step(&mut state, 29, 1.0, &[], &gate_config()), GateStep::Waiting);
        assert_eq!(gate_step(&mut state, 30, 1.0, &[], &gate_config()), GateStep::Clear);
    }

    #[test]
    fn gate_restarts_the_hold_when_load_spikes_or_a_busy_process_appears() {
        let mut state = QuietState { started: 0, quiet_since: None };
        gate_step(&mut state, 0, 1.0, &[], &gate_config());
        assert_eq!(gate_step(&mut state, 20, 4.0, &[], &gate_config()), GateStep::Waiting);
        assert_eq!(state.quiet_since, None);
        let busy = vec!["41 cargo nextest run".to_owned()];
        assert_eq!(gate_step(&mut state, 25, 1.0, &busy, &gate_config()), GateStep::Waiting);
        assert_eq!(gate_step(&mut state, 40, 1.0, &[], &gate_config()), GateStep::Waiting);
        assert_eq!(gate_step(&mut state, 70, 1.0, &[], &gate_config()), GateStep::Clear);
    }

    #[test]
    fn gate_gives_up_at_the_wait_limit_naming_the_busy_process() {
        let mut state = QuietState { started: 0, quiet_since: None };
        let busy = vec!["77 rustc --crate-name x".to_owned()];
        assert_eq!(gate_step(&mut state, 60, 1.0, &busy, &gate_config()), GateStep::Waiting);
        match gate_step(&mut state, 120, 90.0, &busy, &gate_config()) {
            GateStep::GaveUp(reason) => assert!(reason.contains("rustc")),
            other => panic!("expected give-up, got {other:?}"),
        }
    }

    #[test]
    fn gate_treats_nan_load_as_not_quiet() {
        let mut state = QuietState { started: 0, quiet_since: None };
        assert_eq!(gate_step(&mut state, 500, f64::NAN, &[], &gate_config()), GateStep::GaveUp("load1 NaN did not hold below 4 for 30 s".to_owned()));
    }

    #[test]
    fn pgrep_output_drops_this_process_its_parent_and_pgrep() {
        let output = "100 /bin/kv_scoreboard --out x\n90 zsh -c kv_scoreboard\n201 cargo nextest run\n305 pgrep -lf cargo\n411 llama-server -m blob\n";
        let busy = busy_from_pgrep(output, 100, 90);
        assert_eq!(busy.len(), 2);
        assert!(busy[0].starts_with("201 cargo"));
        assert!(busy[1].starts_with("411 llama-server"));
    }

    fn facts() -> Facts {
        Facts {
            model_path: "/models/blob".to_owned(),
            model_sha256: "abc".to_owned(),
            load_limit: 4.0,
            llama_commit: "ba3c05f71".to_owned(),
        }
    }

    fn llama_response(ids: Vec<u64>) -> LlamaResponse {
        LlamaResponse {
            predicted_n: ids.len(),
            ids,
            model: "/models/blob".to_owned(),
            prompt_n: 27,
            cache_n: 0,
            prompt_ms: 80.0,
            predicted_ms: 1000.0,
            predicted_per_second: 128.0,
            predicted_per_token_ms: 1000.0 / 128.0,
            top_k: 1,
            temperature: 0.0,
            seed: 1,
        }
    }

    fn llama_log() -> LlamaLog {
        LlamaLog {
            n_ctx_seq: Some(256),
            flash_attn: Some(true),
            kv_buffer_mib: 4.5,
            kv_types: vec![("f16".to_owned(), "f16".to_owned()), ("f16".to_owned(), "f16".to_owned())],
            build: Some((11_671, "ba3c05f71".to_owned())),
            eog_ids: vec![1, 50, 106],
        }
    }

    fn row(engine: Engine, evidence: Evidence) -> Row {
        let label = if engine == Engine::Llama { "llama" } else { "new" };
        Row {
            round: 1,
            cell: Cell { engine, label: label.to_owned(), prompt: Prompt::Short, kv: Kv::F16 },
            load1: 1.0,
            footprint_bytes: Some(355_363_520),
            metrics: parse_proxima(PROX_LOG).expect("real proxima log parses"),
            exit_code: 0,
            note: String::new(),
            evidence,
            legacy: None,
            span_metrics: true,
            sent_ids: 27,
            capacity: 160,
            binary_sha256: "deadbeef".to_owned(),
            debug_build: false,
            features: "std,metal".to_owned(),
            gate: GateOutcome::Clear { waited_secs: 30 },
            parity: Parity::unevaluated(),
        }
    }

    fn proxima_row(ids: Vec<u64>, prompt_count: usize) -> Row {
        let detail = ProximaDetail { tokens_generated: ids.len(), ids, prompt_token_count: prompt_count };
        row(Engine::Proxima { binary: PathBuf::from("/bin/a") }, Evidence::Proxima(detail))
    }

    fn llama_row(ids: Vec<u64>) -> Row {
        let evidence = LlamaEvidence { cold: llama_response(ids.clone()), warm: llama_response(ids), log: llama_log() };
        row(Engine::Llama, Evidence::Llama(Box::new(evidence)))
    }

    fn pair_rows() -> (Row, Row) {
        let ids: Vec<u64> = (1000..1128).collect();
        (proxima_row(ids.clone(), 27), llama_row(ids))
    }

    fn failing_names(proxima: &Row, llama: &Row, facts: &Facts) -> Vec<String> {
        let mut names: Vec<String> = evaluate(proxima, Some(llama), facts)
            .failing()
            .iter()
            .filter_map(|text| text.split_whitespace().next().map(str::to_owned))
            .collect();
        names.dedup();
        names
    }

    #[test]
    fn a_pair_with_every_invariant_equal_has_an_empty_failing_list_on_both_rows() {
        let (proxima, llama) = pair_rows();
        assert!(evaluate(&proxima, Some(&llama), &facts()).valid());
        assert!(evaluate(&llama, Some(&proxima), &facts()).valid());
    }

    #[test]
    fn i1_fails_when_llama_served_another_model_path_or_no_sha_was_taken() {
        let (proxima, mut llama) = pair_rows();
        if let Evidence::Llama(evidence) = &mut llama.evidence {
            evidence.warm.model = "/models/other".to_owned();
        }
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I1"]);
        let (proxima, llama) = pair_rows();
        let unhashed = Facts { model_sha256: String::new(), ..facts() };
        assert_eq!(failing_names(&proxima, &llama, &unhashed), vec!["I1"]);
    }

    #[test]
    fn i2_fails_when_proxima_displays_26_ids_and_27_were_sent() {
        let ids: Vec<u64> = (1000..1128).collect();
        let proxima = proxima_row(ids.clone(), 26);
        let llama = llama_row(ids);
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I2"]);
        assert!(evaluate(&llama, Some(&proxima), &facts()).valid());
    }

    #[test]
    fn i2_fails_when_llama_counts_fewer_prompt_ids_than_were_sent() {
        let (proxima, mut llama) = pair_rows();
        if let Evidence::Llama(evidence) = &mut llama.evidence {
            evidence.warm.prompt_n = 26;
        }
        assert!(evaluate(&llama, Some(&proxima), &facts()).failing()[0].starts_with("I2"));
    }

    #[test]
    fn i3_fails_when_counts_differ_from_llama_and_128() {
        let (_, llama) = pair_rows();
        let short = proxima_row((1000..1052).collect(), 27);
        let reasons = evaluate(&short, Some(&llama), &facts()).failing();
        assert!(reasons.iter().any(|reason| reason.contains("proxima 52 tokens, llama 128")), "{reasons:?}");
    }

    #[test]
    fn i3_fails_when_proxima_ids_contain_the_turn_end_id_106() {
        let (_, llama) = pair_rows();
        let mut ids: Vec<u64> = (1000..1128).collect();
        ids[70] = 106;
        let proxima = proxima_row(ids, 27);
        let reasons = evaluate(&proxima, Some(&llama), &facts()).failing();
        assert!(reasons.contains(&"I3 proxima emitted EOG ids [106]".to_owned()), "{reasons:?}");
    }

    #[test]
    fn i3_fails_when_the_llama_log_lists_no_eog_ids() {
        let (proxima, mut llama) = pair_rows();
        if let Evidence::Llama(evidence) = &mut llama.evidence {
            evidence.log.eog_ids.clear();
        }
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I3"]);
    }

    #[test]
    fn i4_names_the_first_divergence_position() {
        let mut ids: Vec<u64> = (1000..1128).collect();
        let llama = llama_row(ids.clone());
        ids[40] = 9_999;
        let proxima = proxima_row(ids, 27);
        let failing = evaluate(&proxima, Some(&llama), &facts()).failing();
        assert_eq!(failing, vec!["I4 first divergence at 40".to_owned()]);
        assert_eq!(divergence_text(&proxima, Some(&llama)), "40");
    }

    #[test]
    fn i5_fails_when_llama_allocated_a_different_kv_type() {
        let (proxima, mut llama) = pair_rows();
        if let Evidence::Llama(evidence) = &mut llama.evidence {
            evidence.log.kv_types[1] = ("f16".to_owned(), "q8_0".to_owned());
        }
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I5"]);
    }

    #[test]
    fn i6_fails_when_llama_flash_attention_is_off() {
        let (proxima, mut llama) = pair_rows();
        if let Evidence::Llama(evidence) = &mut llama.evidence {
            evidence.log.flash_attn = Some(false);
        }
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I6"]);
    }

    #[test]
    fn i7_fails_when_llama_n_ctx_is_not_our_capacity_padded_to_256() {
        let (proxima, mut llama) = pair_rows();
        if let Evidence::Llama(evidence) = &mut llama.evidence {
            evidence.log.n_ctx_seq = Some(131_072);
        }
        let failing = evaluate(&proxima, Some(&llama), &facts()).failing();
        assert_eq!(failing.len(), 1);
        assert!(failing[0].starts_with("I7 llama n_ctx Some(131072), expected 256 (ours 160)"));
    }

    #[test]
    fn i12_fails_when_llama_reports_another_build_than_the_required_commit() {
        let homebrew = "0.00.247.973 I cmn  common_param: common_params_print_info: build 12110 (cd959bae9) with AppleClang 17.0.0.17000604 for Darwin arm64";
        assert_eq!(parse_llama_log(homebrew).build, Some((12_110, "cd959bae9".to_owned())));
        let (proxima, mut llama) = pair_rows();
        if let Evidence::Llama(evidence) = &mut llama.evidence {
            evidence.log.build = parse_llama_log(homebrew).build;
        }
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I12"]);
        if let Evidence::Llama(evidence) = &mut llama.evidence {
            evidence.log.build = None;
        }
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I12"]);
    }

    #[test]
    fn i8_fails_for_a_debug_binary_or_a_missing_binary_hash() {
        let (mut proxima, llama) = pair_rows();
        proxima.debug_build = true;
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I8"]);
        let (mut proxima, llama) = pair_rows();
        proxima.binary_sha256.clear();
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I8"]);
    }

    #[test]
    fn i9_fails_and_the_row_is_invalid_for_a_binary_without_llama_span_fields() {
        let (mut proxima, llama) = pair_rows();
        proxima.span_metrics = parse_proxima_span(PROX_LOG, &Metrics::default()).is_some();
        assert!(!proxima.span_metrics, "PROX_LOG run=done lines carry no span fields");
        let failing = evaluate(&proxima, Some(&llama), &facts()).failing();
        assert_eq!(failing, vec!["I9 binary predates llama-span metrics".to_owned()]);
        let mut rows = vec![proxima, llama];
        evaluate_all(&mut rows, &facts());
        assert!(!rows[0].parity.valid());
    }

    // built from real run=done lines: `ttft_ms_precise` appended to each, plus two decode
    // `step_time` records per run whose step_ns are exactly 8.0, 10.0 and 12.5 ms
    fn span_log() -> String {
        let done = PROX_LOG.lines().map(|line| {
            let extra = match run_index_of(line).filter(|_| line.contains("run=done")) {
                Some(0) => " ttft_ms_precise=300.5",
                Some(1) => " ttft_ms_precise=80.25",
                Some(2) => " ttft_ms_precise=90.75",
                _ => "",
            };
            format!("{line}{extra}")
        });
        let records = [(0, 8_000_000_u64), (1, 10_000_000), (2, 12_500_000)].into_iter().flat_map(|(run, nanos)| {
            (1..=2).map(move |step| format!("INFO decode_gbps_baseline: step_time run_index={run} step={step} step_ms=0 step_ns={nanos} phase=decode"))
        });
        done.chain(records).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn llama_span_fields_give_the_expected_ttft_and_tok_s_and_keep_the_legacy_apart() {
        let log = span_log();
        let legacy = parse_proxima(&log).expect("legacy fields still parse");
        let span = parse_proxima_span(&log, &legacy).expect("span fields present on every run=done line");
        assert!(close(span.ttft_cold_ms, 300.5));
        assert_eq!(span.ttft_warm_ms, vec![80.25, 90.75]);
        assert!(close(span.tok_s_run0, 125.0));
        assert_eq!(span.tok_s, vec![100.0, 80.0]);
        assert_eq!(span.gpu_peak_bytes, legacy.gpu_peak_bytes);
        assert!(close(legacy.ttft_cold_ms, 338.09125));
    }

    #[test]
    fn one_run_done_line_missing_a_span_field_rejects_the_whole_span_parse() {
        let partial = span_log().replacen(" ttft_ms_precise=90.75", "", 1);
        let legacy = parse_proxima(&partial).expect("legacy parses");
        assert!(parse_proxima_span(&partial, &legacy).is_none());
    }

    #[test]
    fn a_run_without_decode_step_records_rejects_the_whole_span_parse() {
        let partial: String = span_log().lines().filter(|line| !(line.contains("run_index=2 step=") && line.contains("phase=decode"))).collect::<Vec<_>>().join("\n");
        let legacy = parse_proxima(&partial).expect("legacy parses");
        assert!(partial.contains("run_index=2 ") && parse_proxima_span(&partial, &legacy).is_none());
    }

    const STEP_TIMES_FIXTURE: &str = include_str!("../tests/fixtures/decode_gbps_baseline_step_times.log");

    fn record(index: usize, phase: SpanPhase, elapsed_ns: u64) -> TokenRecord {
        TokenRecord { index, phase, elapsed_ns }
    }

    #[test]
    fn the_reducer_gives_the_same_ms_per_token_for_the_same_records_whichever_arm_wrote_them() {
        let llama_arm_text = STEP_TIMES_FIXTURE.replace("INFO decode_gbps_baseline:", "llama_arm");
        assert_ne!(llama_arm_text, STEP_TIMES_FIXTURE, "the arm prefix must differ for the control to mean anything");
        let from_proxima = decode_span_ms_per_token(&parse_token_records(STEP_TIMES_FIXTURE)[&0]).expect("127 decode records");
        let from_llama = decode_span_ms_per_token(&parse_token_records(&llama_arm_text)[&0]).expect("127 decode records");
        assert_eq!(from_proxima.to_bits(), from_llama.to_bits());
        assert!((from_proxima - 9.851_366_795).abs() < 1e-9, "real 128-token run: got {from_proxima}");
    }

    #[test]
    fn prefill_records_are_excluded_from_the_decode_span() {
        let decode = [record(1, SpanPhase::Decode, 9_000_000), record(2, SpanPhase::Decode, 11_000_000)];
        let with_prefill = [record(0, SpanPhase::Prefill, 400_000_000), decode[0], decode[1]];
        assert_eq!(decode_span_ms_per_token(&with_prefill), Some(10.0));
        assert_eq!(decode_span_ms_per_token(&decode), Some(10.0));
        assert_eq!(prefill_ms(&with_prefill), Some(400.0));
    }

    #[test]
    fn a_prefill_record_mislabelled_decode_changes_the_ms_per_token() {
        let honest = [
            record(0, SpanPhase::Prefill, 400_000_000),
            record(1, SpanPhase::Decode, 9_000_000),
            record(2, SpanPhase::Prefill, 300_000_000),
            record(3, SpanPhase::Decode, 11_000_000),
        ];
        let mut mislabelled = honest;
        mislabelled[2].phase = SpanPhase::Decode;
        assert_eq!(decode_span_ms_per_token(&honest), Some(10.0));
        assert_eq!(decode_span_ms_per_token(&mislabelled), Some((9.0 + 300.0 + 11.0) / 3.0));
    }

    #[test]
    fn a_step_zero_prefill_record_is_excluded_by_index_even_when_mislabelled_decode() {
        let mut records = [record(0, SpanPhase::Prefill, 400_000_000), record(1, SpanPhase::Decode, 9_000_000)];
        let honest = decode_span_ms_per_token(&records);
        records[0].phase = SpanPhase::Decode;
        assert_eq!(decode_span_ms_per_token(&records), honest);
    }

    #[test]
    fn no_admitted_record_is_none_not_zero() {
        assert_eq!(decode_span_ms_per_token(&[]), None);
        assert_eq!(decode_span_ms_per_token(&[record(0, SpanPhase::Prefill, 5), record(0, SpanPhase::Decode, 5)]), None);
    }

    const INPROC_GATE_LINE: &str = "llama_arm gate sha256=d54034a90777fb94ce2b8c6638b71a7cde7beb29ed7e160c0622ca894f36cabc build=11671 type_k=f16 type_v=f16 flash_attn=enabled n_ctx=256 n_predict=128 runs=3";

    fn inproc_log() -> String {
        let steps = (0..3).flat_map(|run| {
            let prefill = format!("llama_arm step_time run_index={run} step=0 step_ms=80 step_ns=80500000 phase=prefill");
            let decode = (1..=3).map(move |step| format!("llama_arm step_time run_index={run} step={step} step_ms=9 step_ns=9250000 phase=decode"));
            std::iter::once(prefill).chain(decode)
        });
        std::iter::once(INPROC_GATE_LINE.to_owned()).chain(steps).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn inproc_log_gives_prefill_ttft_and_reducer_tok_s_for_all_three_runs() {
        let metrics = parse_inproc_span(&inproc_log()).expect("three runs of records");
        assert!(close(metrics.ttft_cold_ms, 80.5));
        assert_eq!(metrics.ttft_warm_ms, vec![80.5, 80.5]);
        assert!(close(metrics.tok_s_run0, 1000.0 / 9.25));
        assert_eq!(metrics.tok_s, vec![1000.0 / 9.25, 1000.0 / 9.25]);
        let gate = parse_inproc_gate(&inproc_log()).expect("gate line");
        assert_eq!((gate.build, gate.type_k.as_str(), gate.flash_attn, gate.n_ctx_requested), (11_671, "f16", true, 256));
        assert!(parse_inproc_span("llama_arm gate sha256=x").is_none());
    }

    const LLAMA_ARM_FIXTURE: &str = include_str!("fixtures/llama_arm_step_times.log");

    #[test]
    fn a_real_inproc_arm_log_reduces_to_the_sum_of_its_127_decode_records_over_127() {
        let records = parse_token_records(LLAMA_ARM_FIXTURE);
        assert_eq!(records.len(), 3);
        for run in records.values() {
            assert_eq!(run.iter().filter(|found| found.phase == SpanPhase::Prefill).count(), 1);
            assert_eq!(run.iter().filter(|found| found.counts_toward_decode_span()).count(), 127);
        }
        let by_hand: u64 = records[&1].iter().filter(|found| found.phase == SpanPhase::Decode && found.index >= 1).map(|found| found.elapsed_ns).sum();
        let reduced = decode_span_ms_per_token(&records[&1]).expect("127 decode records");
        assert!((reduced - by_hand as f64 / 127.0 / 1e6).abs() < 1e-12);
        let gate = parse_inproc_gate(LLAMA_ARM_FIXTURE).expect("gate line");
        assert_eq!((gate.build, gate.flash_attn), (PINNED_LLAMA_BUILD, true));
        assert!(parse_inproc_span(LLAMA_ARM_FIXTURE).is_some());
    }

    fn inproc_row(ids: Vec<u64>, build: u32) -> Row {
        let gate = InprocGate {
            sha256: "deadbeef".to_owned(),
            build,
            type_k: "f16".to_owned(),
            type_v: "f16".to_owned(),
            flash_attn: true,
            n_ctx_requested: 256,
        };
        let detail = ProximaDetail { tokens_generated: ids.len(), ids, prompt_token_count: 27 };
        let evidence = Evidence::Inproc(Box::new(InprocEvidence { detail, gate }));
        row(Engine::LlamaInproc { binary: PathBuf::from("/bin/inproc_arm") }, evidence)
    }

    #[test]
    fn an_inproc_row_with_the_same_ids_as_the_server_has_no_failing_invariant() {
        let ids: Vec<u64> = (1000..1128).collect();
        let server = llama_row(ids.clone());
        let inproc = inproc_row(ids, PINNED_LLAMA_BUILD);
        assert_eq!(evaluate(&inproc, Some(&server), &facts()).failing(), Vec::<String>::new());
        assert_eq!(divergence_text(&inproc, Some(&server)), "none");
    }

    #[test]
    fn an_inproc_row_names_the_first_divergence_from_the_server_ids() {
        let ids: Vec<u64> = (1000..1128).collect();
        let mut other = ids.clone();
        other[40] = 7;
        let server = llama_row(ids);
        let inproc = inproc_row(other, PINNED_LLAMA_BUILD);
        assert_eq!(evaluate(&inproc, Some(&server), &facts()).failing(), vec!["I4 first divergence at 40".to_owned()]);
        assert_eq!(divergence_text(&inproc, Some(&server)), "40");
    }

    #[test]
    fn an_inproc_gate_reporting_another_build_fails_i12() {
        let inproc = inproc_row((1000..1128).collect(), 11_670);
        let failing = evaluate(&inproc, None, &facts()).failing();
        assert_eq!(failing, vec!["I12 llama-arm gate build 11670 (required 11671)".to_owned()]);
    }

    #[test]
    fn an_inproc_row_pairs_with_the_server_row_and_the_tsv_carries_gate_sha_and_build() {
        let ids: Vec<u64> = (1000..1128).collect();
        let mut rows = vec![inproc_row(ids.clone(), PINNED_LLAMA_BUILD), llama_row(ids), proxima_row((1000..1128).collect(), 27)];
        evaluate_all(&mut rows, &facts());
        assert_eq!(find_peer(&rows, &rows[0]).map(|peer| peer.cell.engine_name()), Some("llama"));
        assert_eq!(find_peer(&rows, &rows[2]).map(|peer| peer.cell.engine_name()), Some("llama"));
        let line = tsv_line(&rows[0], find_peer(&rows, &rows[0]), &facts());
        assert_eq!(line.split('\t').count(), TSV_HEADER.split('\t').count());
        assert!(line.contains("\tdeadbeef\t") && line.contains("\tbuild 11671 (inproc gate)\t"), "{line}");
        assert!(line.contains("INPROC ttft="));
    }

    #[test]
    fn i10_fails_when_the_gate_refused_or_load_was_high_at_launch() {
        let (mut proxima, llama) = pair_rows();
        proxima.gate = GateOutcome::Refused("busy: 201 cargo".to_owned());
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I10"]);
        let (mut proxima, llama) = pair_rows();
        proxima.load1 = 39.0;
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I10"]);
    }

    #[test]
    fn i11_fails_when_llama_echoes_a_sampler_other_than_greedy_seed_1() {
        let (proxima, mut llama) = pair_rows();
        if let Evidence::Llama(evidence) = &mut llama.evidence {
            evidence.warm.top_k = 40;
        }
        assert_eq!(failing_names(&proxima, &llama, &facts()), vec!["I11"]);
    }

    #[test]
    fn the_proxima_only_row_marks_pair_invariants_not_applicable_not_failed() {
        let mut proxima = proxima_row((1000..1128).collect(), 27);
        proxima.cell.kv = Kv::Q4K;
        let parity = evaluate(&proxima, None, &facts());
        assert!(parity.valid());
        assert!(matches!(parity.checks[2], Check::NotApplicable(_)));
        assert!(matches!(parity.checks[3], Check::NotApplicable(_)));
    }

    #[test]
    fn invalid_rows_are_listed_with_failing_invariants_and_excluded_from_the_means() {
        let (mut proxima, llama) = pair_rows();
        let mut slow = proxima.clone();
        slow.metrics.ttft_cold_ms = Some(9_000.0);
        slow.load1 = 39.0;
        proxima.round = 1;
        slow.round = 2;
        let mut llama_two = llama.clone();
        llama_two.round = 2;
        let mut rows = vec![proxima.clone(), llama, slow, llama_two];
        evaluate_all(&mut rows, &facts());
        assert!(rows[0].parity.valid());
        assert!(!rows[2].parity.valid());
        let cells = vec![proxima.cell.clone(), rows[1].cell.clone()];
        let table = scoreboard_markdown(&rows, &cells);
        assert!(table.contains("| proxima | new | 2 | 1 | 0 |"), "{table}");
        assert!(table.contains("338.09 (n/a) n=1"), "the 9000 ms cold TTFT of the INVALID row must not enter the mean: {table}");
        assert!(table.contains("1 of 4 rows are INVALID"));
        assert!(table.contains("round 2 proxima new short F16: I10 load1 39.00 at launch"));
    }

    #[test]
    fn tsv_line_has_one_field_per_header_column_and_carries_the_parity_record() {
        let (proxima, llama) = pair_rows();
        let mut rows = vec![proxima, llama];
        evaluate_all(&mut rows, &facts());
        for (index, tsv_row) in rows.iter().enumerate() {
            let peer = &rows[1 - index];
            let line = tsv_line(tsv_row, Some(peer), &facts());
            assert_eq!(line.split('\t').count(), TSV_HEADER.split('\t').count());
            assert!(line.contains("\tvalid\tnone\t128\t128\t"));
        }
    }

    #[test]
    fn llama_has_no_arm_for_the_q4_key_row_and_cells_follow_the_row_space() {
        let proxima = vec![("new".to_owned(), PathBuf::from("/bin/a"))];
        let cells = cell_order(&proxima, None);
        assert_eq!(cells.len(), 2 * (4 + 3));
        assert!(!cells.iter().any(|cell| cell.engine == Engine::Llama && cell.kv == Kv::Q4K));
    }

    #[test]
    fn the_inproc_arm_gets_a_cell_wherever_the_server_has_one_and_flags_select_a_subset() {
        let inproc = PathBuf::from("/bin/inproc_arm");
        let cells = cell_order(&[], Some(&inproc));
        assert_eq!(cells.len(), 2 * 3 * 2);
        assert!(cells.iter().all(|cell| cell.kv != Kv::Q4K));
        let base = ["--out", "o", "--model", "m", "--llama", "l", "--long-prompt-file", "p", "--llama-commit", "ba3c05f71"];
        let mut arguments: Vec<String> = base.iter().map(|text| (*text).to_owned()).collect();
        arguments.extend(["--llama-inproc", "/bin/inproc_arm", "--prompts", "short", "--kvs", "F16"].map(str::to_owned));
        let config = parse_config(arguments.clone()).expect("an inproc arm alone is an arm");
        assert_eq!((config.prompts, config.kvs), (vec![Prompt::Short], vec![Kv::F16]));
        arguments.extend(["--kvs", "F8"].map(str::to_owned));
        assert!(parse_config(arguments).is_err());
    }

    #[test]
    fn every_round_visits_every_cell_once_and_no_pair_keeps_one_relative_order() {
        let proxima = vec![
            ("new".to_owned(), PathBuf::from("/bin/a")),
            ("old".to_owned(), PathBuf::from("/bin/b")),
        ];
        let cells = cell_order(&proxima, None);
        assert_eq!(cells.len(), 2 * (4 * 2 + 3));
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
    fn cov_of_8_10_12_is_20_percent_mean_10() {
        let (center, cov) = sample_mean_cov(&[8.0, 10.0, 12.0]).expect("three values");
        assert!((center - 10.0).abs() < 1e-12);
        assert!((cov.expect("n=3") - 20.0).abs() < 1e-9);
        assert_eq!(sample_mean_cov(&[5.0]), Some((5.0, None)));
        assert_eq!(sample_mean_cov(&[]), None);
    }

    #[test]
    fn proxima_command_sets_ignore_eos_for_main_and_profile_runs() {
        let arguments = ["--out", "o", "--model", "m", "--llama", "l", "--long-prompt-file", "p", "--llama-commit", "ba3c05f71", "--prox", "new=/bin/a"]
            .map(str::to_owned)
            .to_vec();
        let config = parse_config(arguments).expect("valid arguments");
        for profile in [None, Some(PROFILE_PREFILL_STEP), Some("3")] {
            let command = proxima_command(&config, Path::new("/bin/a"), "hello", profile);
            let ignore_eos = command
                .get_envs()
                .find(|(key, _)| *key == "PROXIMA_IGNORE_EOS")
                .and_then(|(_, value)| value);
            assert_eq!(ignore_eos, Some(std::ffi::OsStr::new("1")), "profile {profile:?} must mirror llama ignore_eos");
        }
    }

    #[test]
    fn config_requires_a_proxima_arm_and_rejects_bad_rounds_and_flags() {
        let base = ["--out", "o", "--model", "m", "--llama", "l", "--long-prompt-file", "p", "--llama-commit", "ba3c05f71"];
        let mut arguments: Vec<String> = base.iter().map(|text| (*text).to_owned()).collect();
        assert!(parse_config(arguments.clone()).is_err());
        arguments.extend(["--prox", "new=/bin/a", "--rounds", "0"].map(str::to_owned));
        assert!(parse_config(arguments.clone()).is_err());
        arguments.truncate(arguments.len() - 2);
        let config = parse_config(arguments.clone()).expect("valid arguments");
        assert_eq!(config.rounds, DEFAULT_ROUNDS);
        assert_eq!(config.gate, GateConfig { load_limit: 4.0, hold_secs: 30, wait_limit_secs: 900 });
        arguments.extend(["--load-limit", "2.5", "--hold-secs", "60"].map(str::to_owned));
        let tuned = parse_config(arguments.clone()).expect("tuned gate");
        assert_eq!((tuned.gate.load_limit, tuned.gate.hold_secs), (2.5, 60));
        arguments.extend(["--bogus", "1"].map(str::to_owned));
        assert!(parse_config(arguments).is_err());
    }

    #[test]
    fn detokenize_prompt_ms_times_a_decode_that_succeeds_on_the_prompt_ids() {
        let tokens = vec!["<unk>".to_owned(), "Hel".to_owned(), "lo".to_owned(), " world".to_owned()];
        let vocab = Vocab::new(tokens, &[], None, None, None).expect("plain vocab");
        let mut text = String::new();
        proxima_tokenizer::decode_into(&[1, 2, 3], &vocab, &mut text).expect("decodes");
        assert_eq!(text, "Hello world");
        assert!(detokenize_prompt_ms(&[1, 2, 3], &vocab) >= 0.0);
    }

    #[test]
    fn text_edge_arms_report_prompt_detokenize_apart_from_generated_and_carry_rss() {
        let response = parse_llama_response(LLAMA_F32_84_RESPONSE).expect("real llama completion parses");
        let edges = LlamaTextEdges { tokenize_ms: 2.0, tokenize_tokens: response.ids.len(), detokenize_ms: 2.25, rss_bytes: 600 * 1024 * 1024 };
        let mut llama = RowSink::new(0, "llama", "84", 1.5);
        llama_text_edge_rows(&mut llama, &edges);
        assert!(close(rows_of(&llama, "tokenize_inproc_ms")[0].value, 2.0));
        assert!(close(rows_of(&llama, "detokenize_prompt_inproc_ms")[0].value, 2.25));
        assert!(rows_of(&llama, "detokenize_ms").is_empty() && rows_of(&llama, "tokenize_ms").is_empty());
        assert!(close(rows_of(&llama, "vocab_only_process_rss_bytes")[0].value, 629_145_600.0));
        let mut proxima = RowSink::new(0, "proxima-new", "84", 1.5);
        proxima_text_edge_rows(&mut proxima, 0.1, 0.05, response.ids.len());
        assert!(close(rows_of(&proxima, "detokenize_prompt_inproc_ms")[0].value, 0.05));
        assert!(rows_of(&proxima, "detokenize_ms").is_empty() && rows_of(&proxima, "prefill_ms_per_prompt_token").is_empty());
        assert!(close(rows_of(&proxima, "tokenize_inproc_tokens")[0].value, count_as_f64(response.ids.len())));
        assert!(rows_of(&proxima, "tokenize_ms").is_empty());
    }

    #[test]
    fn llama_model_params_matches_the_b12110_header_layout() {
        // llama.h:321-358 on 64-bit: devices 0, tensor_buft_overrides 8, five i32 at 16,20,24,28,32 (end 36),
        // tensor_split pads to 40, progress_callback 48, user_data 56, kv_overrides 64, vocab_only 72,
        // six bools 72..78, struct align 8 so size rounds 78 up to 80
        assert_eq!(std::mem::offset_of!(LlamaModelParams, n_gpu_layers), 16);
        assert_eq!(std::mem::offset_of!(LlamaModelParams, tensor_split), 40);
        assert_eq!(std::mem::offset_of!(LlamaModelParams, kv_overrides), 64);
        assert_eq!(std::mem::offset_of!(LlamaModelParams, vocab_only), 72);
        assert_eq!(std::mem::size_of::<LlamaModelParams>(), 80);
    }

    #[test]
    fn rss_over_the_limit_becomes_an_error_row_naming_the_rss() {
        let error = rss_error("after the exchanges", 25_000_000_000);
        let rows = error_rows(0, "llama", "84", 1.5, &error);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].value.is_none() && rows[0].note.contains("25000000000"));
    }

    #[test]
    fn text_edges_flag_needs_stages_and_takes_no_value() {
        let base = ["--out", "o", "--model", "m", "--llama", "l", "--llama-commit", "x", "--prox", "new=/bin/true"];
        let mut arguments: Vec<String> = base.iter().map(|text| (*text).to_owned()).collect();
        arguments.push("--text-edges".to_owned());
        assert!(parse_config(arguments.clone()).is_err());
        arguments.extend(["--stages", "d"].map(str::to_owned));
        assert!(parse_config(arguments).expect("stages with text edges").text_edges);
    }

    #[test]
    fn nonzero_exit_rows_stay_in_the_table() {
        let (mut proxima, _) = pair_rows();
        proxima.exit_code = 134;
        proxima.parity = Parity { checks: std::array::from_fn(|_| Check::Held) };
        let cell = proxima.cell.clone();
        let table = scoreboard_markdown(&[proxima], &[cell]);
        assert!(table.contains("| proxima | new | 1 | 0 | 1 | - | - | - | - | - | - |"), "{table}");
    }

    const REAL_IDS: [u32; 5] = [2, 236776, 5213, 20619, 1];
    const REAL_STAMP: &str = "/models/gemma.gguf|25666650112|1762900000000000000|F16";

    fn llama_arm_rows() -> Vec<StageRow> {
        let mut sink = RowSink::new(0, "llama", "84", 1.37);
        llama_rows(&mut sink, &llama_stage_from_real_response());
        sink.rows
    }

    #[test]
    fn bytes_above_four_gib_convert_exactly() {
        assert_eq!(bytes_as_f64(25_666_650_112).to_bits(), 25_666_650_112.0_f64.to_bits());
        assert_eq!(bytes_as_f64(4_294_967_296).to_bits(), 4_294_967_296.0_f64.to_bits());
        assert_eq!(bytes_as_f64(u64::from(u32::MAX)).to_bits(), 4_294_967_295.0_f64.to_bits());
        assert_eq!(bytes_value(Some(25_666_650_112)).map(f64::to_bits), Some(25_666_650_112.0_f64.to_bits()));
    }

    #[test]
    fn cache_key_changes_with_commit_ids_flash_and_round_and_is_otherwise_stable() {
        let base = cache_key(REAL_STAMP, "ba3c05f71", &REAL_IDS, "on", "llama", 0);
        assert_eq!(base, cache_key(REAL_STAMP, "ba3c05f71", &REAL_IDS, "on", "llama", 0));
        assert_eq!(base.len(), 64);
        assert_ne!(base, cache_key(REAL_STAMP, "b12110", &REAL_IDS, "on", "llama", 0));
        assert_ne!(base, cache_key(REAL_STAMP, "ba3c05f71", &REAL_IDS[..4], "on", "llama", 0));
        assert_ne!(base, cache_key(REAL_STAMP, "ba3c05f71", &REAL_IDS, "off", "llama", 0));
        assert_ne!(base, cache_key(REAL_STAMP, "ba3c05f71", &REAL_IDS, "on", "llama", 1));
        assert_ne!(base, cache_key("/models/gemma.gguf|25666650112|1762900000000000001|F16", "ba3c05f71", &REAL_IDS, "on", "llama", 0));
    }

    #[test]
    fn cached_rows_read_back_with_value_unit_load_and_a_cached_note() {
        let directory = tempfile::tempdir().expect("tempdir for the incumbent cache");
        let rows = llama_arm_rows();
        store_cached(directory.path(), "abc", &rows).expect("cache file written");
        let back = lookup_cached(directory.path(), "abc", 2, "510").expect("cache file hits");
        assert_eq!(back.len(), rows.len());
        for (stored, found) in rows.iter().zip(&back) {
            assert_eq!((found.metric, found.value, found.unit, found.definition), (stored.metric, stored.value, stored.unit, stored.definition));
            assert_eq!(found.load1, stored.load1);
            assert_eq!((found.round, found.prompt.as_str()), (2, "510"));
            assert!(found.note.starts_with("cached abc.tsv"), "{}", found.note);
        }
    }

    #[test]
    fn a_directory_without_the_file_is_a_miss() {
        let directory = tempfile::tempdir().expect("tempdir for the incumbent cache");
        assert!(lookup_cached(directory.path(), "absent", 0, "84").is_none());
    }

    #[test]
    fn a_file_with_an_unknown_metric_name_is_a_miss() {
        let directory = tempfile::tempdir().expect("tempdir for the incumbent cache");
        let text = cache_tsv(&llama_arm_rows()).replace("decode_ms_per_token", "decode_ms_renamed");
        fs::write(directory.path().join("stale.tsv"), text).expect("stale cache file written");
        assert!(lookup_cached(directory.path(), "stale", 0, "84").is_none());
    }

    #[test]
    fn incumbent_cache_flags_parse_and_default_to_off() {
        let base = ["--out", "o", "--model", "m", "--llama", "l", "--long-prompt-file", "p", "--llama-commit", "ba3c05f71", "--prox", "new=/bin/a"];
        let mut arguments: Vec<String> = base.iter().map(|text| (*text).to_owned()).collect();
        let plain = parse_config(arguments.clone()).expect("valid arguments");
        assert!(plain.incumbent_cache.is_none() && !plain.refresh_incumbent);
        arguments.extend(["--incumbent-cache", "/cache", "--refresh-incumbent"].map(str::to_owned));
        let cached = parse_config(arguments).expect("cache flags");
        assert_eq!(cached.incumbent_cache, Some(PathBuf::from("/cache")));
        assert!(cached.refresh_incumbent);
    }
}
