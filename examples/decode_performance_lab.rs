//! Reproducible process-level capture for literal-prompt GGUF decode arms.

use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, exit};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const METHODS: [&str; 5] = [
    "baseline",
    "ngram_speculation",
    "loopcd",
    "gemma_mtp",
    "diffusion",
];
const FAILURE_MARKERS: [&str; 4] = [
    "GGUF PARSE FAILED",
    "WEIGHT LOAD FAILED",
    "METAL RUN FAILED",
    "GENERATION FAILED",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    max_tokens: usize,
    pairs_per_prompt: usize,
    models: Vec<Model>,
    prompts: Vec<Prompt>,
    references: Vec<Reference>,
    method_eligibility: Vec<MethodEligibility>,
    primary_sources: Vec<PrimarySource>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    key: String,
    model_id: String,
    checkpoint: PathBuf,
    expected_architecture: String,
    expected_general_name: String,
    speculative_verification: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Prompt {
    key: String,
    text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Reference {
    method: String,
    model_id: String,
    model_type: String,
    path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MethodEligibility {
    method: String,
    eligible: bool,
    reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrimarySource {
    topic: String,
    url: String,
}

#[derive(Debug, Serialize)]
struct TokenEvent {
    kind: String,
    token_id: Option<u32>,
    text_piece: Option<String>,
    step: u64,
    elapsed_ms: f64,
    prompt_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
struct RunRecord {
    record_type: &'static str,
    model_key: String,
    model_id: String,
    prompt_key: String,
    pair_index: usize,
    arm: String,
    configured_method: String,
    algorithm_eligible: bool,
    algorithm_eligibility_reason: String,
    algorithm_execution_observed: bool,
    model_fidelity_unassessed: bool,
    command: Vec<String>,
    environment: BTreeMap<String, String>,
    host_load_snapshot: BTreeMap<String, String>,
    status: String,
    exit_code: Option<i32>,
    wall_ms: f64,
    user_cpu_seconds: Option<f64>,
    system_cpu_seconds: Option<f64>,
    peak_rss_bytes: Option<u64>,
    generate_ms: Option<f64>,
    reported_tokens_per_second: Option<f64>,
    cpu_percent_of_wall: Option<f64>,
    measurement_sources: BTreeMap<String, String>,
    generated_ids: Vec<u32>,
    payload_parse_error: Option<String>,
    generated_text: Option<String>,
    prefill_prompt_tokens: Option<u64>,
    ttft_ms: Option<f64>,
    decode_intervals_ms: Vec<f64>,
    token_events: Vec<TokenEvent>,
    metal_counter_rows: Vec<String>,
    speculative_counter_rows: Vec<String>,
    draft_tokens: Option<u64>,
    accepted_draft_tokens: Option<u64>,
    speculative_counters: BTreeMap<String, u64>,
    stdout_path: String,
    stderr_path: String,
    time_path: String,
    stdout: String,
    stderr: String,
    time_output: String,
}

fn cli_value(args: &[OsString], flag: &str) -> Option<PathBuf> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| PathBuf::from(&pair[1]))
}

fn has_flag(args: &[OsString], flag: &str) -> bool {
    args.iter().any(|argument| argument == flag)
}

fn read_manifest(path: &Path) -> Result<Manifest, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("read manifest {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse manifest {}: {error}", path.display()))
}

fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.version != 1 || manifest.max_tokens == 0 || manifest.max_tokens > 24 {
        return Err("manifest version or max_tokens is outside the supported bound".to_owned());
    }
    if manifest.models.len() != 2
        || manifest.prompts.len() != 2
        || manifest.pairs_per_prompt != 4
        || manifest.references.len() != 3
    {
        return Err("manifest must contain two models, two prompts, three references, and four pairs per prompt".to_owned());
    }
    if manifest.models[0].key == manifest.models[1].key
        || manifest.prompts[0].key == manifest.prompts[1].key
    {
        return Err("model and prompt keys must be unique".to_owned());
    }
    if manifest.models.iter().any(|model| {
        model.key.is_empty()
            || model.model_id.is_empty()
            || model.expected_architecture.is_empty()
            || model.expected_general_name.is_empty()
    }) || manifest
        .prompts
        .iter()
        .any(|prompt| prompt.key.is_empty() || prompt.text.is_empty())
    {
        return Err("manifest contains an empty model or prompt field".to_owned());
    }
    if manifest.method_eligibility.len() != METHODS.len()
        || METHODS.iter().any(|method| {
            !manifest
                .method_eligibility
                .iter()
                .any(|entry| entry.method == *method)
        })
    {
        return Err("manifest must describe all five inference methods".to_owned());
    }
    if manifest.primary_sources.len() != 4
        || manifest
            .primary_sources
            .iter()
            .any(|source| source.topic.is_empty() || !source.url.starts_with("https://"))
    {
        return Err("manifest must retain four named primary-source URLs".to_owned());
    }
    Ok(())
}

fn quoted_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    let json_value: Value = serde_json::from_str(trimmed).ok()?;
    json_value.as_str().map(ToOwned::to_owned)
}

fn extract_ids(text: &str) -> Result<Vec<u32>, String> {
    let value = text
        .lines()
        .find_map(|line| line.strip_prefix("generated_ids = "))
        .ok_or_else(|| "generated_ids line is absent".to_owned())?;
    serde_json::from_str::<Vec<u32>>(value.trim())
        .map_err(|error| format!("generated_ids is malformed: {error}"))
}

fn extract_number(text: &str, key: &str) -> Option<f64> {
    text.lines().find_map(|line| {
        let (name, value) = line.split_once('=')?;
        if name.trim() == key {
            value.trim().parse::<f64>().ok()
        } else {
            None
        }
    })
}

fn extract_token_count(text: &str) -> Option<usize> {
    text.lines().find_map(|line| {
        line.strip_prefix("tokens_generated = ")?
            .trim()
            .parse()
            .ok()
    })
}

fn extract_events(text: &str) -> (Vec<TokenEvent>, Option<u64>, Option<f64>, Vec<f64>) {
    let mut events = Vec::new();
    let mut prefill_tokens = None;
    let mut ttft = None;
    let mut decode_intervals = Vec::new();
    let mut token_times = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("prefill_event ") {
            let fields = key_values(rest);
            let prompt_tokens = fields
                .get("prompt_tokens")
                .and_then(|value| value.parse::<u64>().ok());
            let step = fields
                .get("step")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            let elapsed_ms = fields
                .get("elapsed_ms")
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.0);
            prefill_tokens = prompt_tokens;
            events.push(TokenEvent {
                kind: "prefill".to_owned(),
                token_id: None,
                text_piece: None,
                step,
                elapsed_ms,
                prompt_tokens,
            });
        } else if let Some(rest) = line.strip_prefix("token_event ") {
            let fields = key_values(rest);
            let token_id = fields
                .get("token_id")
                .and_then(|value| value.parse::<u32>().ok());
            let step = fields
                .get("step")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            let elapsed_ms = fields
                .get("elapsed_ms")
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.0);
            let text_piece = line
                .split_once("text_piece=")
                .and_then(|(_, tail)| tail.split_once(" step=").map(|(value, _)| value))
                .and_then(quoted_string);
            token_times.push(elapsed_ms);
            events.push(TokenEvent {
                kind: "token".to_owned(),
                token_id,
                text_piece,
                step,
                elapsed_ms,
                prompt_tokens: None,
            });
        }
    }
    if let Some(first) = token_times.first() {
        ttft = Some(*first);
    }
    for window in token_times.windows(2) {
        decode_intervals.push(window[1] - window[0]);
    }
    (events, prefill_tokens, ttft, decode_intervals)
}

fn key_values(text: &str) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    for segment in text.split_whitespace() {
        if let Some((key, value)) = segment.split_once('=') {
            result.insert(key.to_owned(), value.to_owned());
        }
    }
    result
}

fn parse_time_output(text: &str) -> (Option<f64>, Option<f64>, Option<u64>) {
    let mut user_cpu = None;
    let mut system_cpu = None;
    let mut peak = None;
    for line in text.lines() {
        if let Some(value) = line.strip_prefix("\tUser time (seconds): ") {
            user_cpu = value.trim().parse().ok();
        }
        if let Some(value) = line.strip_prefix("\tSystem time (seconds): ") {
            system_cpu = value.trim().parse().ok();
        }
        if let Some(value) = line.strip_prefix("\tMaximum resident set size (kbytes): ") {
            peak = value.trim().parse::<u64>().ok().map(|bytes| bytes * 1024);
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        for window in fields.windows(2) {
            if window[1] == "user" {
                user_cpu = window[0].parse().ok();
            }
            if window[1] == "sys" {
                system_cpu = window[0].parse().ok();
            }
        }
        if peak.is_none()
            && !line.contains("kbytes")
            && let Some(index) = fields
                .windows(4)
                .position(|window| window == ["maximum", "resident", "set", "size"])
                .and_then(|index| index.checked_sub(1))
        {
            peak = fields
                .get(index)
                .and_then(|value| value.parse::<u64>().ok());
        }
    }
    (user_cpu, system_cpu, peak)
}

fn parse_counter(text: &str, key: &str) -> Option<u64> {
    let prefix = format!("{key}=");
    text.lines().find_map(|line| {
        let fields: Vec<&str> = line.split_whitespace().collect();
        for (index, field) in fields.iter().enumerate() {
            if let Some(value) = field
                .strip_prefix(&prefix)
                .and_then(|value| value.parse::<u64>().ok())
            {
                return Some(value);
            }
            if *field == key
                && fields.get(index + 1) == Some(&"=")
                && let Some(value) = fields
                    .get(index + 2)
                    .and_then(|value| value.trim_end_matches(',').parse::<u64>().ok())
            {
                return Some(value);
            }
            if field.strip_suffix('=').is_some_and(|name| name == key)
                && let Some(value) = fields
                    .get(index + 1)
                    .and_then(|value| value.trim_end_matches(',').parse::<u64>().ok())
            {
                return Some(value);
            }
        }
        None
    })
}

fn speculative_counters(text: &str) -> BTreeMap<String, u64> {
    [
        "ngram_simple_drafted",
        "ngram_simple_accepted",
        "ngram_map_k_drafted",
        "ngram_map_k_accepted",
        "ngram_map_k4v_drafted",
        "ngram_map_k4v_accepted",
        "ngram_mod_drafted",
        "ngram_mod_accepted",
        "ngram_cache_drafted",
        "ngram_cache_accepted",
    ]
    .into_iter()
    .filter_map(|key| parse_counter(text, key).map(|value| (key.to_owned(), value)))
    .collect()
}

fn host_load_snapshot() -> BTreeMap<String, String> {
    let mut snapshot = BTreeMap::new();
    for (key, command, arguments) in [
        ("uptime", "/usr/bin/uptime", Vec::<&str>::new()),
        ("processes", "/bin/ps", vec!["-A", "-o", "pid,comm"]),
    ] {
        let result = Command::new(command).args(arguments).output();
        let value = match result {
            Ok(output) => format!(
                "exit={:?}\n{}\n{}",
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
            Err(error) => format!("unavailable: {error}"),
        };
        snapshot.insert(key.to_owned(), value);
    }
    snapshot
}

#[derive(Clone, Copy, Default)]
struct RunPayload<'run> {
    ids: &'run [u32],
    generated_text: Option<&'run str>,
    printed_count: Option<usize>,
    parse_error: Option<&'run str>,
    token_events: &'run [TokenEvent],
}

fn validate_run(
    output: &Output,
    stdout: &str,
    stderr: &str,
    payload: &RunPayload<'_>,
    model: &Model,
) -> (String, bool, String) {
    let RunPayload {
        ids,
        generated_text,
        printed_count,
        parse_error: payload_parse_error,
        token_events,
    } = *payload;
    let combined = format!("{stdout}\n{stderr}");
    if output.status.code() != Some(0) {
        return (
            "process_failed".to_owned(),
            false,
            "child exit status was nonzero".to_owned(),
        );
    }
    if let Some(marker) = FAILURE_MARKERS
        .iter()
        .find(|marker| combined.contains(**marker))
    {
        return (
            "failure_marker".to_owned(),
            false,
            format!("captured output contains {marker}"),
        );
    }
    if combined.contains("falling back to CPU")
        || combined.contains("backend_requested = CPU (fallback:")
    {
        return (
            "cpu_fallback".to_owned(),
            false,
            "GPU request fell back to CPU".to_owned(),
        );
    }
    let gpu_execution = combined.lines().any(|line| {
        line.split_whitespace()
            .filter_map(|field| field.strip_prefix("gpu_exec_calls="))
            .any(|value| value.parse::<u64>().is_ok_and(|calls| calls > 0))
    });
    if !combined.contains("backend_requested = GPU/METAL (requested") || !gpu_execution {
        return (
            "no_gpu_execution_evidence".to_owned(),
            false,
            "requested GPU run has no token-step Metal execution counter".to_owned(),
        );
    }
    if let Some(error) = payload_parse_error {
        return (
            "malformed_generated_ids".to_owned(),
            false,
            error.to_owned(),
        );
    }
    let token_ids: Vec<u32> = token_events
        .iter()
        .filter(|event| event.kind == "token")
        .filter_map(|event| event.token_id)
        .collect();
    if ids.is_empty()
        || generated_text.is_none()
        || printed_count != Some(ids.len())
        || token_ids != ids
    {
        return ("invalid_token_count".to_owned(), false, "generated ID, text, printed-token, and token-event payload counts disagree or are empty".to_owned());
    }
    let architecture = combined
        .lines()
        .find_map(|line| line.strip_prefix("general.architecture = "));
    let general_name = combined
        .lines()
        .find_map(|line| line.strip_prefix("general.name = "));
    if architecture != Some(model.expected_architecture.as_str())
        || general_name != Some(model.expected_general_name.as_str())
    {
        return (
            "model_identity_mismatch".to_owned(),
            false,
            "reported architecture or general name differs from manifest identity".to_owned(),
        );
    }
    (
        "captured".to_owned(),
        true,
        "all process, payload, model, and Metal checks passed".to_owned(),
    )
}

fn run_arm(
    binary: &Path,
    model: &Model,
    prompt: &Prompt,
    pair_index: usize,
    arm: &str,
    max_tokens: usize,
    output_dir: &Path,
) -> Result<RunRecord, String> {
    let configured_method = if arm == "off" {
        "baseline"
    } else {
        "ngram_speculation"
    };
    let algorithm_eligible = arm == "off" || model.speculative_verification;
    let eligibility_reason = if arm == "off" {
        "baseline path".to_owned()
    } else if algorithm_eligible {
        "model has configured verification capability".to_owned()
    } else {
        "configured drafter has no verifier path for this checkpoint; no acceleration attribution"
            .to_owned()
    };
    let run_id = format!("{}-{}-pair-{pair_index:02}-{arm}", model.key, prompt.key);
    let stdout_path = output_dir.join(format!("{run_id}.stdout"));
    let stderr_path = output_dir.join(format!("{run_id}.stderr"));
    let time_path = output_dir.join(format!("{run_id}.time"));
    let model_path = model.checkpoint.to_string_lossy().into_owned();
    let prompt_text = prompt.text.clone();
    let max_tokens = max_tokens.to_string();
    let mut child = Command::new("/usr/bin/time");
    child
        .arg("-l")
        .arg("-o")
        .arg(&time_path)
        .arg(binary)
        .arg(&model_path)
        .arg(&prompt_text)
        .arg(&max_tokens)
        .arg("gpu");
    for (key, _) in env::vars_os().filter(|(key, _)| key.to_string_lossy().starts_with("PROXIMA_"))
    {
        child.env_remove(key);
    }
    child.env("PROXIMA_TEMPERATURE", "0");
    child.env("PROXIMA_REPEAT_PENALTY", "1");
    child.env("PROXIMA_SPECULATIVE_NGRAM_SIMPLE_SIZE_N", "1");
    child.env("PROXIMA_SPECULATIVE_NGRAM_SIMPLE_SIZE_M", "3");
    child.env(
        "PROXIMA_SPECULATIVE_TYPES",
        if arm == "off" { "none" } else { "ngram-simple" },
    );
    child.env("RUST_LOG", "info");
    let environment = BTreeMap::from([
        ("PROXIMA_TEMPERATURE".to_owned(), "0".to_owned()),
        ("PROXIMA_REPEAT_PENALTY".to_owned(), "1".to_owned()),
        (
            "PROXIMA_SPECULATIVE_NGRAM_SIMPLE_SIZE_N".to_owned(),
            "1".to_owned(),
        ),
        (
            "PROXIMA_SPECULATIVE_NGRAM_SIMPLE_SIZE_M".to_owned(),
            "3".to_owned(),
        ),
        (
            "PROXIMA_SPECULATIVE_TYPES".to_owned(),
            if arm == "off" {
                "none".to_owned()
            } else {
                "ngram-simple".to_owned()
            },
        ),
        ("RUST_LOG".to_owned(), "info".to_owned()),
    ]);
    let command = vec![
        "/usr/bin/time".to_owned(),
        "-l".to_owned(),
        "-o".to_owned(),
        time_path.display().to_string(),
        binary.display().to_string(),
        model_path,
        prompt_text,
        max_tokens.to_owned(),
        "gpu".to_owned(),
    ];
    let host_load_snapshot = host_load_snapshot();
    let started = Instant::now();
    let child_output = match child.output() {
        Ok(output) => output,
        Err(error) => {
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            let stderr = format!("failed to launch timed child: {error}");
            fs::write(&stdout_path, "").map_err(|write_error| write_error.to_string())?;
            fs::write(&stderr_path, &stderr).map_err(|write_error| write_error.to_string())?;
            fs::write(&time_path, "").map_err(|write_error| write_error.to_string())?;
            return Ok(RunRecord {
                record_type: "run",
                model_key: model.key.clone(),
                model_id: model.model_id.clone(),
                prompt_key: prompt.key.clone(),
                pair_index,
                arm: arm.to_owned(),
                configured_method: configured_method.to_owned(),
                algorithm_eligible: false,
                algorithm_eligibility_reason: "timed child failed to launch".to_owned(),
                algorithm_execution_observed: false,
                model_fidelity_unassessed: true,
                command,
                environment,
                host_load_snapshot,
                status: "launch_failed".to_owned(),
                exit_code: None,
                wall_ms,
                user_cpu_seconds: None,
                system_cpu_seconds: None,
                peak_rss_bytes: None,
                generate_ms: None,
                reported_tokens_per_second: None,
                cpu_percent_of_wall: None,
                measurement_sources: BTreeMap::new(),
                generated_ids: Vec::new(),
                payload_parse_error: None,
                generated_text: None,
                prefill_prompt_tokens: None,
                ttft_ms: None,
                decode_intervals_ms: Vec::new(),
                token_events: Vec::new(),
                metal_counter_rows: Vec::new(),
                speculative_counter_rows: Vec::new(),
                draft_tokens: None,
                accepted_draft_tokens: None,
                speculative_counters: BTreeMap::new(),
                stdout_path: stdout_path.display().to_string(),
                stderr_path: stderr_path.display().to_string(),
                time_path: time_path.display().to_string(),
                stdout: String::new(),
                stderr,
                time_output: String::new(),
            });
        }
    };
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&child_output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&child_output.stderr).into_owned();
    let time_output = fs::read_to_string(&time_path).unwrap_or_default();
    fs::write(&stdout_path, &child_output.stdout).map_err(|error| error.to_string())?;
    fs::write(&stderr_path, &child_output.stderr).map_err(|error| error.to_string())?;
    let (ids, payload_parse_error) = match extract_ids(&stdout) {
        Ok(ids) => (ids, None),
        Err(error) => (Vec::new(), Some(error)),
    };
    let generated_text = stdout.lines().find_map(|line| {
        line.strip_prefix("generated_text = ")
            .and_then(quoted_string)
    });
    let printed_count = extract_token_count(&stdout);
    let (token_events, prefill_prompt_tokens, ttft_ms, decode_intervals_ms) =
        extract_events(&stdout);
    let complete_output = format!("{stdout}\n{stderr}");
    let metal_counter_rows = complete_output
        .lines()
        .filter(|line| line.contains("token_breakdown_metal:"))
        .map(ToOwned::to_owned)
        .collect();
    let speculative_counter_rows = complete_output
        .lines()
        .filter(|line| line.contains("draft_n=") && line.contains("draft_n_accepted="))
        .map(ToOwned::to_owned)
        .collect();
    let draft_tokens = parse_counter(&complete_output, "draft_n");
    let accepted_draft_tokens = parse_counter(&complete_output, "draft_n_accepted");
    let algorithm_execution_observed =
        algorithm_eligible && draft_tokens.is_some_and(|count| count > 0);
    let speculative_counters = speculative_counters(&complete_output);
    let (status, _, status_reason) = validate_run(
        &child_output,
        &stdout,
        &stderr,
        &RunPayload {
            ids: &ids,
            generated_text: generated_text.as_deref(),
            printed_count,
            parse_error: payload_parse_error.as_deref(),
            token_events: &token_events,
        },
        model,
    );
    let (user_cpu_seconds, system_cpu_seconds, peak_rss_bytes) = parse_time_output(&time_output);
    let (status, _, status_reason) = if status == "captured"
        && (user_cpu_seconds.is_none() || system_cpu_seconds.is_none() || peak_rss_bytes.is_none())
    {
        (
            "instrumentation_incomplete".to_owned(),
            false,
            "timed child output omitted CPU time or peak RSS".to_owned(),
        )
    } else {
        (status, true, status_reason)
    };
    let generate_ms = extract_number(&stdout, "generate_ms");
    let reported_tokens_per_second = extract_number(&stdout, "tokens_per_sec");
    let cpu_percent_of_wall = user_cpu_seconds
        .zip(system_cpu_seconds)
        .filter(|_| wall_ms > 0.0)
        .map(|(user, system)| (user + system) / (wall_ms / 1000.0) * 100.0);
    let measurement_sources = BTreeMap::from([
        (
            "wall_ms".to_owned(),
            "std::time::Instant around the timed child process".to_owned(),
        ),
        (
            "generate_ms".to_owned(),
            "gguf_generate stdout generate_ms field".to_owned(),
        ),
        (
            "cpu_seconds".to_owned(),
            "/usr/bin/time -l raw output".to_owned(),
        ),
        (
            "peak_rss_bytes".to_owned(),
            "/usr/bin/time -l raw output, converted from its reported unit".to_owned(),
        ),
        (
            "reported_tokens_per_second".to_owned(),
            "gguf_generate stdout tokens_per_sec field".to_owned(),
        ),
        (
            "cpu_percent_of_wall".to_owned(),
            "derived (user CPU seconds + system CPU seconds) / child wall seconds * 100".to_owned(),
        ),
        (
            "token_events".to_owned(),
            "gguf_generate callback stdout token_event and prefill_event rows".to_owned(),
        ),
    ]);
    Ok(RunRecord {
        record_type: "run",
        model_key: model.key.clone(),
        model_id: model.model_id.clone(),
        prompt_key: prompt.key.clone(),
        pair_index,
        arm: arm.to_owned(),
        configured_method: configured_method.to_owned(),
        algorithm_eligible,
        algorithm_eligibility_reason: if status == "captured" {
            eligibility_reason
        } else {
            status_reason
        },
        algorithm_execution_observed,
        model_fidelity_unassessed: true,
        command,
        environment,
        host_load_snapshot,
        status,
        exit_code: child_output.status.code(),
        wall_ms,
        user_cpu_seconds,
        system_cpu_seconds,
        peak_rss_bytes,
        generate_ms,
        reported_tokens_per_second,
        cpu_percent_of_wall,
        measurement_sources,
        generated_ids: ids,
        payload_parse_error,
        generated_text,
        prefill_prompt_tokens,
        ttft_ms,
        decode_intervals_ms,
        token_events,
        metal_counter_rows,
        speculative_counter_rows,
        draft_tokens,
        accepted_draft_tokens,
        speculative_counters,
        stdout_path: stdout_path.display().to_string(),
        stderr_path: stderr_path.display().to_string(),
        time_path: time_path.display().to_string(),
        stdout,
        stderr,
        time_output,
    })
}

fn write_json_line(file: &mut File, value: &impl Serialize) -> Result<(), String> {
    serde_json::to_writer(&mut *file, value).map_err(|error| error.to_string())?;
    file.write_all(b"\n").map_err(|error| error.to_string())
}

fn run_campaign(manifest: &Manifest, binary: &Path, output: &Path) -> Result<(), String> {
    if !binary.is_file() {
        return Err(format!("binary does not exist: {}", binary.display()));
    }
    for model in &manifest.models {
        if !model.checkpoint.is_file() {
            return Err(format!(
                "checkpoint does not exist: {}",
                model.checkpoint.display()
            ));
        }
    }
    fs::create_dir_all(output).map_err(|error| format!("create output directory: {error}"))?;
    let records_path = output.join("runs.jsonl");
    let mut records = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&records_path)
        .map_err(|error| error.to_string())?;
    let pairs_path = output.join("pairs.jsonl");
    let mut pairs = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&pairs_path)
        .map_err(|error| error.to_string())?;
    let mut attempted_runs = 0usize;
    let mut pair_records = 0usize;
    for model in &manifest.models {
        for prompt in &manifest.prompts {
            for pair_index in 0..manifest.pairs_per_prompt {
                let order = if pair_index % 2 == 0 {
                    ["off", "on"]
                } else {
                    ["on", "off"]
                };
                let mut pair_records_local = Vec::new();
                for arm in order {
                    let record = run_arm(
                        binary,
                        model,
                        prompt,
                        pair_index,
                        arm,
                        manifest.max_tokens,
                        output,
                    )?;
                    println!(
                        "completed model={} prompt={} pair={} arm={} status={}",
                        record.model_key,
                        record.prompt_key,
                        record.pair_index,
                        record.arm,
                        record.status
                    );
                    write_json_line(&mut records, &record)?;
                    attempted_runs += 1;
                    pair_records_local.push(record);
                }
                let off = pair_records_local.iter().find(|record| record.arm == "off");
                let on = pair_records_local.iter().find(|record| record.arm == "on");
                let payload_match = off.zip(on).map(|(off, on)| {
                    off.generated_ids == on.generated_ids && off.generated_text == on.generated_text
                });
                let usable = off.zip(on).is_some_and(|(off, on)| {
                    off.status == "captured"
                        && on.status == "captured"
                        && payload_match == Some(true)
                        && on.algorithm_execution_observed
                });
                let pair = json!({"record_type":"pair","model_key":model.key,"prompt_key":prompt.key,"pair_index":pair_index,
                    "warmup":pair_index == 0,"off_status":off.map(|record| record.status.as_str()),"on_status":on.map(|record| record.status.as_str()),
                    "payload_match":payload_match,"usable_for_timing_summary":usable,
                    "off_wall_ms":off.map(|record| record.wall_ms),"on_wall_ms":on.map(|record| record.wall_ms),
                    "on_minus_off_wall_ms":off.zip(on).map(|(off,on)|on.wall_ms-off.wall_ms),
                    "on_draft_tokens":on.and_then(|record|record.draft_tokens),"on_accepted_draft_tokens":on.and_then(|record|record.accepted_draft_tokens),
                    "off_eligible":off.is_some_and(|record|record.algorithm_eligible),"on_eligible":on.is_some_and(|record|record.algorithm_eligible),
                    "on_algorithm_execution_observed":on.is_some_and(|record|record.algorithm_execution_observed),
                    "on_speculative_counter_rows":on.map(|record|&record.speculative_counter_rows)});
                write_json_line(&mut pairs, &pair)?;
                pair_records += 1;
            }
        }
    }
    let summary = derive_summary(&records_path, &pairs_path)?;
    fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    println!("attempted_runs={attempted_runs} pair_records={pair_records}");
    Ok(())
}

fn read_json_lines(path: &Path) -> Result<Vec<Value>, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    BufReader::new(file)
        .lines()
        .map(|line| {
            let line = line.map_err(|error| error.to_string())?;
            serde_json::from_str(&line).map_err(|error| error.to_string())
        })
        .collect()
}

fn stats(values: &[f64], signed_values: bool) -> Value {
    if values.is_empty() {
        return json!({"sample_count":0,"min":null,"max":null,"mean":null,"standard_deviation":null,"cov_percent":null,"cov_ratio":null,"p50":null,"p95":null,"p99":null});
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let standard_deviation = if values.len() < 2 {
        None
    } else {
        Some(
            (values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (values.len() - 1) as f64)
                .sqrt(),
        )
    };
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |percent: usize| {
        let rank = (percent * sorted.len()).div_ceil(100).max(1);
        sorted[rank - 1]
    };
    json!({"sample_count":values.len(),"min":values.iter().copied().fold(f64::INFINITY,f64::min),
        "max":values.iter().copied().fold(f64::NEG_INFINITY,f64::max),"mean":mean,"standard_deviation":standard_deviation,
        "p50":percentile(50),"p95":percentile(95),"p99":percentile(99),
        "cov_percent":if signed_values || values.len() < 2 || mean <= 0.0 { None } else { standard_deviation.map(|stddev|stddev/mean*100.0) },
        "cov_ratio":if signed_values || values.len() < 2 || mean <= 0.0 { None } else { standard_deviation.map(|stddev|stddev/mean) }})
}

#[derive(Default)]
struct SummaryGroup {
    off_wall_ms: Vec<f64>,
    on_wall_ms: Vec<f64>,
    wall_delta_ms: Vec<f64>,
    off_reported_tps: Vec<f64>,
    on_reported_tps: Vec<f64>,
    off_decode_reciprocal_tps: Vec<f64>,
    on_decode_reciprocal_tps: Vec<f64>,
    off_ttft_ms: Vec<f64>,
    on_ttft_ms: Vec<f64>,
    off_intervals_ms: Vec<f64>,
    on_intervals_ms: Vec<f64>,
}

#[derive(Default)]
struct ObservedArmGroup {
    wall_ms: Vec<f64>,
    generate_ms: Vec<f64>,
    ttft_ms: Vec<f64>,
    reported_tps: Vec<f64>,
    reciprocal_decode_tps: Vec<f64>,
    decode_intervals_ms: Vec<f64>,
    cpu_percent: Vec<f64>,
    peak_rss_bytes: Vec<f64>,
    draft_tokens: Vec<f64>,
    accepted_draft_tokens: Vec<f64>,
    algorithm_eligible: bool,
    algorithm_execution_observed_count: usize,
}

fn reciprocal_mean_tps(intervals_ms: &[f64]) -> Option<f64> {
    if intervals_ms.is_empty() {
        return None;
    }
    let mean_ms = intervals_ms.iter().sum::<f64>() / intervals_ms.len() as f64;
    (mean_ms > 0.0).then_some(1000.0 / mean_ms)
}

fn derive_summary(records_path: &Path, pairs_path: &Path) -> Result<Value, String> {
    let records = read_json_lines(records_path)?;
    let pairs = read_json_lines(pairs_path)?;
    let mut groups: BTreeMap<(String, String), SummaryGroup> = BTreeMap::new();
    let mut observed_groups: BTreeMap<(String, String, String), ObservedArmGroup> = BTreeMap::new();
    for record in &records {
        if record.get("status").and_then(Value::as_str) != Some("captured")
            || record.get("pair_index").and_then(Value::as_u64) == Some(0)
        {
            continue;
        }
        let Some(model_key) = record.get("model_key").and_then(Value::as_str) else {
            continue;
        };
        let Some(prompt_key) = record.get("prompt_key").and_then(Value::as_str) else {
            continue;
        };
        let Some(arm) = record.get("arm").and_then(Value::as_str) else {
            continue;
        };
        let group = observed_groups
            .entry((model_key.to_owned(), prompt_key.to_owned(), arm.to_owned()))
            .or_default();
        group
            .wall_ms
            .extend(record.get("wall_ms").and_then(Value::as_f64));
        group
            .generate_ms
            .extend(record.get("generate_ms").and_then(Value::as_f64));
        group
            .ttft_ms
            .extend(record.get("ttft_ms").and_then(Value::as_f64));
        group.reported_tps.extend(
            record
                .get("reported_tokens_per_second")
                .and_then(Value::as_f64),
        );
        group
            .cpu_percent
            .extend(record.get("cpu_percent_of_wall").and_then(Value::as_f64));
        group.peak_rss_bytes.extend(
            record
                .get("peak_rss_bytes")
                .and_then(Value::as_u64)
                .map(|value| value as f64),
        );
        group.draft_tokens.extend(
            record
                .get("draft_tokens")
                .and_then(Value::as_u64)
                .map(|value| value as f64),
        );
        group.accepted_draft_tokens.extend(
            record
                .get("accepted_draft_tokens")
                .and_then(Value::as_u64)
                .map(|value| value as f64),
        );
        group.algorithm_eligible = record
            .get("algorithm_eligible")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if record
            .get("algorithm_execution_observed")
            .and_then(Value::as_bool)
            == Some(true)
        {
            group.algorithm_execution_observed_count += 1;
        }
        let intervals = record
            .get("decode_intervals_ms")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
            .unwrap_or_default();
        group
            .reciprocal_decode_tps
            .extend(reciprocal_mean_tps(&intervals));
        group.decode_intervals_ms.extend(intervals);
    }
    for pair in pairs {
        if pair.get("warmup").and_then(Value::as_bool) != Some(false)
            || pair
                .get("usable_for_timing_summary")
                .and_then(Value::as_bool)
                != Some(true)
            || pair.get("on_eligible").and_then(Value::as_bool) != Some(true)
        {
            continue;
        }
        let Some(model_key) = pair.get("model_key").and_then(Value::as_str) else {
            continue;
        };
        let Some(prompt_key) = pair.get("prompt_key").and_then(Value::as_str) else {
            continue;
        };
        let Some(delta) = pair.get("on_minus_off_wall_ms").and_then(Value::as_f64) else {
            continue;
        };
        let get_arm = |arm: &str| {
            records.iter().find(|record| {
                record.get("model_key").and_then(Value::as_str) == Some(model_key)
                    && record.get("prompt_key").and_then(Value::as_str) == Some(prompt_key)
                    && record.get("pair_index").and_then(Value::as_u64)
                        == pair.get("pair_index").and_then(Value::as_u64)
                    && record.get("arm").and_then(Value::as_str) == Some(arm)
            })
        };
        let off_record = get_arm("off");
        let on_record = get_arm("on");
        let collect_intervals = |record: Option<&Value>| {
            record
                .and_then(|value| value.get("decode_intervals_ms"))
                .and_then(Value::as_array)
                .map(|items| items.iter().filter_map(Value::as_f64).collect::<Vec<_>>())
                .unwrap_or_default()
        };
        let off_intervals = collect_intervals(off_record);
        let on_intervals = collect_intervals(on_record);
        let group = groups
            .entry((model_key.to_owned(), prompt_key.to_owned()))
            .or_default();
        group.wall_delta_ms.push(delta);
        group.off_wall_ms.extend(
            off_record
                .and_then(|record| record.get("wall_ms"))
                .and_then(Value::as_f64),
        );
        group.on_wall_ms.extend(
            on_record
                .and_then(|record| record.get("wall_ms"))
                .and_then(Value::as_f64),
        );
        group.off_reported_tps.extend(
            off_record
                .and_then(|record| record.get("reported_tokens_per_second"))
                .and_then(Value::as_f64),
        );
        group.on_reported_tps.extend(
            on_record
                .and_then(|record| record.get("reported_tokens_per_second"))
                .and_then(Value::as_f64),
        );
        group.off_ttft_ms.extend(
            off_record
                .and_then(|record| record.get("ttft_ms"))
                .and_then(Value::as_f64),
        );
        group.on_ttft_ms.extend(
            on_record
                .and_then(|record| record.get("ttft_ms"))
                .and_then(Value::as_f64),
        );
        group
            .off_decode_reciprocal_tps
            .extend(reciprocal_mean_tps(&off_intervals));
        group
            .on_decode_reciprocal_tps
            .extend(reciprocal_mean_tps(&on_intervals));
        group.off_intervals_ms.extend(off_intervals);
        group.on_intervals_ms.extend(on_intervals);
    }
    let grouped = groups.into_iter().map(|((model_key,prompt_key),group)| {
        json!({"model_key":model_key,"prompt_key":prompt_key,
            "off_wall_ms":stats(&group.off_wall_ms,false),"on_wall_ms":stats(&group.on_wall_ms,false),
            "matched_pair_on_minus_off_wall_ms":stats(&group.wall_delta_ms,true),
            "off_reported_tokens_per_second":stats(&group.off_reported_tps,false),
            "on_reported_tokens_per_second":stats(&group.on_reported_tps,false),
            "off_reciprocal_mean_decode_interval_tokens_per_second":stats(&group.off_decode_reciprocal_tps,false),
            "on_reciprocal_mean_decode_interval_tokens_per_second":stats(&group.on_decode_reciprocal_tps,false),
            "off_ttft_ms":stats(&group.off_ttft_ms,false),"off_decode_interval_ms":stats(&group.off_intervals_ms,false),
            "on_ttft_ms":stats(&group.on_ttft_ms,false),"on_decode_interval_ms":stats(&group.on_intervals_ms,false)})
    }).collect::<Vec<_>>();
    let observed = observed_groups.into_iter().map(|((model_key,prompt_key,arm),group)| {
        json!({"model_key":model_key,"prompt_key":prompt_key,"arm":arm,"observational_only":true,"model_fidelity":"unassessed",
            "algorithm_eligible":group.algorithm_eligible,"algorithm_execution_observed_count":group.algorithm_execution_observed_count,
            "wall_ms":stats(&group.wall_ms,false),"generate_ms":stats(&group.generate_ms,false),"ttft_ms":stats(&group.ttft_ms,false),
            "reported_tokens_per_second":stats(&group.reported_tps,false),
            "reciprocal_mean_decode_interval_tokens_per_second":stats(&group.reciprocal_decode_tps,false),
            "decode_interval_ms":stats(&group.decode_intervals_ms,false),"cpu_percent_of_wall":stats(&group.cpu_percent,false),
            "peak_rss_bytes":stats(&group.peak_rss_bytes,false),"draft_tokens":stats(&group.draft_tokens,false),
            "accepted_draft_tokens":stats(&group.accepted_draft_tokens,false)})
    }).collect::<Vec<_>>();
    Ok(
        json!({"derived_only":true,"statistic_definition":{"standard_deviation":"sample standard deviation with denominator n-1; null for n<2","cov_percent":"100 * sample standard deviation / mean; null for n<2, mean<=0, and signed wall deltas","percentiles":"nearest-rank p50/p95/p99","decode_reciprocal_tps":"1000 divided by the per-run arithmetic mean of token-event intervals in milliseconds"},"observation_exclusions":["pair 0 warmup","failed or invalid arms"],"matched_pair_exclusions":["pair 0 warmup","failed or invalid arms","payload-mismatched pairs","ON arm without a positive draft count","ON method not eligible for algorithm attribution"],"observed_arm_distributions":observed,"matched_payload_draft_groups":grouped}),
    )
}

fn inventory(manifest: &Manifest) -> Result<(), String> {
    validate_manifest(manifest)?;
    for model in &manifest.models {
        let exists = model.checkpoint.is_file();
        println!(
            "model={} id={} checkpoint={} checkpoint_exists={} architecture={} expected_name={}",
            model.key,
            model.model_id,
            model.checkpoint.display(),
            exists,
            model.expected_architecture,
            model.expected_general_name
        );
    }
    println!(
        "models={} prompts={} planned_runs={} methods={}",
        manifest.models.len(),
        manifest.prompts.len(),
        manifest.models.len() * manifest.prompts.len() * manifest.pairs_per_prompt * 2,
        METHODS.len()
    );
    for method in &manifest.method_eligibility {
        println!(
            "method={} eligible={} reason={}",
            method.method, method.eligible, method.reason
        );
    }
    for source in &manifest.primary_sources {
        println!("primary_source topic={} url={}", source.topic, source.url);
    }
    Ok(())
}

fn check_references(manifest_path: &Path, manifest: &Manifest) -> Result<(), String> {
    let mut identities = 0;
    for reference in &manifest.references {
        println!(
            "reference method={} model_id={} model_type={}",
            reference.method, reference.model_id, reference.model_type
        );
        let reference_path = manifest_path
            .parent()
            .ok_or("manifest parent missing")?
            .join(&reference.path);
        let bytes = fs::read(&reference_path)
            .map_err(|error| format!("read reference {}: {error}", reference_path.display()))?;
        let config: Value =
            serde_json::from_slice(&bytes).map_err(|error| format!("parse reference: {error}"))?;
        let actual_type = config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("");
        if actual_type != reference.model_type {
            return Err(format!(
                "reference type mismatch for {}",
                reference.model_id
            ));
        }
        identities += 1;
    }
    for model in &manifest.models {
        let exists = model.checkpoint.is_file();
        println!(
            "model_identity key={} id={} exists={}",
            model.key, model.model_id, exists
        );
    }
    println!(
        "reference_configs={} reference_model_type_checks={} models={} local_runtime_eligibility={} methods",
        manifest.references.len(),
        identities,
        manifest.models.len(),
        METHODS.len()
    );
    Ok(())
}

fn check_fixtures(manifest_path: &Path, manifest: &Manifest) -> Result<(), String> {
    let fixture_path = manifest_path
        .parent()
        .ok_or("manifest parent missing")?
        .join("fixtures/expected.json");
    let fixtures: Value =
        serde_json::from_slice(&fs::read(&fixture_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let entries = fixtures
        .get("fixtures")
        .and_then(Value::as_array)
        .ok_or("fixtures array missing")?;
    let mut mismatches = 0;
    for fixture in entries {
        let path = fixture
            .get("stdout_path")
            .and_then(Value::as_str)
            .ok_or("fixture stdout path missing")?;
        let stdout_path = manifest_path
            .parent()
            .ok_or("manifest parent missing")?
            .join(path);
        let stdout = fs::read_to_string(&stdout_path)
            .map_err(|error| format!("read captured fixture {}: {error}", stdout_path.display()))?;
        let (ids, payload_parse_error) = match extract_ids(&stdout) {
            Ok(ids) => (ids, None),
            Err(error) => (Vec::new(), Some(error)),
        };
        let expected_ids: Vec<u32> = fixture
            .get("generated_ids")
            .and_then(Value::as_array)
            .ok_or("fixture IDs missing")?
            .iter()
            .filter_map(Value::as_u64)
            .map(|value| value as u32)
            .collect();
        let expected_text = fixture
            .get("generated_text")
            .and_then(Value::as_str)
            .ok_or("fixture text missing")?;
        let actual_text = stdout.lines().find_map(|line| {
            line.strip_prefix("generated_text = ")
                .and_then(quoted_string)
        });
        let token_count = extract_token_count(&stdout);
        let output = Output {
            status: ExitStatus::from_raw(0),
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
        };
        let model = manifest
            .models
            .iter()
            .find(|model| {
                model.key
                    == fixture
                        .get("model_key")
                        .and_then(Value::as_str)
                        .unwrap_or("")
            })
            .ok_or("fixture model missing")?;
        let (actual_status, _, _) = validate_run(
            &output,
            &stdout,
            "",
            &RunPayload {
                ids: &ids,
                generated_text: actual_text.as_deref(),
                printed_count: token_count,
                parse_error: payload_parse_error.as_deref(),
                token_events: &extract_events(&stdout).0,
            },
            model,
        );
        let expected_exit = fixture.get("exit_code").and_then(Value::as_i64);
        let has_failure_marker = FAILURE_MARKERS.iter().any(|marker| stdout.contains(marker));
        if ids != expected_ids
            || actual_text.as_deref() != Some(expected_text)
            || actual_status != fixture.get("status").and_then(Value::as_str).unwrap_or("")
            || expected_exit != Some(0)
            || fixture.get("failure_marker").and_then(Value::as_bool) != Some(has_failure_marker)
            || fixture
                .get("gpu_execution_observed")
                .and_then(Value::as_bool)
                != Some(stdout.lines().any(|line| {
                    line.split_whitespace().any(|field| {
                        field
                            .strip_prefix("gpu_exec_calls=")
                            .and_then(|value| value.parse::<u64>().ok())
                            .is_some_and(|calls| calls > 0)
                    })
                }))
        {
            mismatches += 1;
        }
    }
    println!(
        "incumbent_fixtures={} parser_mismatches={mismatches}",
        entries.len()
    );
    if mismatches != 0 {
        return Err("captured incumbent fixture mismatch".to_owned());
    }
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<OsString> = env::args_os().skip(1).collect();
    let manifest_path = cli_value(&args, "--manifest").ok_or("required: --manifest PATH")?;
    let manifest = read_manifest(&manifest_path)?;
    validate_manifest(&manifest)?;
    if has_flag(&args, "--inventory") {
        inventory(&manifest)?;
    }
    if has_flag(&args, "--check-references") {
        check_references(&manifest_path, &manifest)?;
    }
    if has_flag(&args, "--check-fixtures") {
        check_fixtures(&manifest_path, &manifest)?;
    }
    if let Some(binary) = cli_value(&args, "--binary") {
        let output = cli_value(&args, "--output").ok_or("--output PATH required with --binary")?;
        run_campaign(&manifest, &binary, &output)?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("decode performance lab: {error}");
        exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model(architecture: &str, general_name: &str) -> Model {
        Model {
            key: "fixture".to_owned(),
            model_id: "fixture/model".to_owned(),
            checkpoint: PathBuf::from("fixture.gguf"),
            expected_architecture: architecture.to_owned(),
            expected_general_name: general_name.to_owned(),
            speculative_verification: false,
        }
    }

    fn manifest() -> Manifest {
        Manifest {
            version: 1,
            max_tokens: 24,
            pairs_per_prompt: 4,
            models: vec![test_model("granitemoe", "Granite 3.1 1b A400M Instruct"), test_model("gemma4", "2B_dequant_qat_it_hf")].into_iter().enumerate().map(|(index,mut model)| {
                model.key = if index == 0 { "granite".to_owned() } else { "gemma".to_owned() };
                model
            }).collect(),
            prompts: vec![Prompt { key: "a".to_owned(), text: "A".to_owned() }, Prompt { key: "b".to_owned(), text: "B".to_owned() }],
            references: vec![
                Reference { method: "gemma_mtp".to_owned(), model_id: "fixture/mtp".to_owned(), model_type: "gemma4_assistant".to_owned(), path: PathBuf::from("mtp.json") },
                Reference { method: "diffusion".to_owned(), model_id: "fixture/diffusion".to_owned(), model_type: "diffusion_gemma".to_owned(), path: PathBuf::from("diffusion.json") },
                Reference { method: "ouro_recurrent_depth".to_owned(), model_id: "fixture/ouro".to_owned(), model_type: "ouro".to_owned(), path: PathBuf::from("ouro.json") },
            ],
            method_eligibility: METHODS.iter().map(|method| MethodEligibility { method: (*method).to_owned(), eligible: true, reason: "test".to_owned() }).collect(),
            primary_sources: vec![
                PrimarySource { topic: "loopcd".to_owned(), url: "https://arxiv.org/html/2610.02185v1".to_owned() },
                PrimarySource { topic: "gemma_mtp".to_owned(), url: "https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/".to_owned() },
                PrimarySource { topic: "gemma_qat".to_owned(), url: "https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/".to_owned() },
                PrimarySource { topic: "diffusion_gemma".to_owned(), url: "https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/".to_owned() },
            ],
        }
    }

    fn fixture_output(text: &str) -> Output {
        Output {
            status: ExitStatus::from_raw(0),
            stdout: text.as_bytes().to_vec(),
            stderr: Vec::new(),
        }
    }

    #[test]
    fn valid_manifest_has_two_by_two_by_four_plan() {
        let fixture = manifest();
        assert!(validate_manifest(&fixture).is_ok());
        assert_eq!(
            fixture.models.len() * fixture.prompts.len() * fixture.pairs_per_prompt * 2,
            32
        );
    }

    #[test]
    fn empty_prompt_is_rejected() {
        let mut fixture = manifest();
        fixture.prompts[1].text.clear();
        assert!(validate_manifest(&fixture).is_err());
        fixture.prompts[1].text = "B".to_owned();
        assert!(validate_manifest(&fixture).is_ok());
        let temporary = tempfile::tempdir().expect("temporary directory should be created");
        let binary = temporary.path().join("not-run-binary");
        File::create(&binary).expect("placeholder binary file should be created");
        fixture.models[0].checkpoint = temporary.path().join("missing.gguf");
        assert!(run_campaign(&fixture, &binary, &temporary.path().join("results")).is_err());
        let malformed_path = temporary.path().join("malformed.json");
        fs::write(&malformed_path, "{").expect("malformed JSON fixture should be written");
        assert!(read_manifest(&malformed_path).is_err());
        fs::write(&malformed_path, r#"{"version":1}"#)
            .expect("missing-field JSON fixture should be written");
        assert!(read_manifest(&malformed_path).is_err());
        let valid_manifest =
            include_str!("../proxima-tensor/specs/decode-performance-lab/manifest.json");
        let unknown_field_manifest = valid_manifest.replacen(
            "\"version\": 1,",
            "\"version\": 1, \"unexpected\": true,",
            1,
        );
        fs::write(&malformed_path, unknown_field_manifest)
            .expect("unknown-field JSON fixture should be written");
        assert!(read_manifest(&malformed_path).is_err());
    }

    #[test]
    fn captured_literal_fixtures_match_transcribed_payloads() {
        let granite =
            include_str!("../proxima-tensor/specs/decode-performance-lab/fixtures/granite.log");
        let gemma =
            include_str!("../proxima-tensor/specs/decode-performance-lab/fixtures/gemma.log");
        assert_eq!(extract_ids(granite), Ok(vec![318]));
        assert_eq!(extract_ids(gemma), Ok(vec![9079, 236761]));
        assert!(extract_ids("generated_ids = [1, nope]").is_err());
        assert_eq!(
            granite
                .lines()
                .find_map(|line| line
                    .strip_prefix("generated_text = ")
                    .and_then(quoted_string))
                .as_deref(),
            Some("ed")
        );
        assert_eq!(
            gemma
                .lines()
                .find_map(|line| line
                    .strip_prefix("generated_text = ")
                    .and_then(quoted_string))
                .as_deref(),
            Some(" Paris.")
        );
    }

    #[test]
    fn event_parser_keeps_text_piece_spaces_and_decode_intervals() {
        let capture = "prefill_event prompt_tokens=6 step=0 elapsed_ms=20\ntoken_event token_id=9079 text_piece=\" Paris\" step=0 elapsed_ms=20\ntoken_event token_id=236761 text_piece=\".\" step=1 elapsed_ms=35";
        let (events, prompt_tokens, ttft, intervals) = extract_events(capture);
        assert_eq!(prompt_tokens, Some(6));
        assert_eq!(ttft, Some(20.0));
        assert_eq!(intervals, vec![15.0]);
        assert_eq!(events[1].text_piece.as_deref(), Some(" Paris"));
    }

    #[test]
    fn mac_time_payload_parses_cpu_and_peak_rss() {
        let payload = "\tUser time (seconds): 1.25\n\tSystem time (seconds): 0.5\n\tMaximum resident set size (kbytes): 2048\n";
        assert_eq!(
            parse_time_output(payload),
            (Some(1.25), Some(0.5), Some(2_097_152))
        );
        let native_payload = "1.25 real 0.25 user 0.125 sys\n2097152 maximum resident set size\n";
        assert_eq!(
            parse_time_output(native_payload),
            (Some(0.25), Some(0.125), Some(2_097_152))
        );
        let statistics = stats(&[10.0, 20.0, 30.0], false);
        assert_eq!(statistics["sample_count"], 3);
        assert_eq!(statistics["mean"], 20.0);
        assert_eq!(statistics["standard_deviation"], 10.0);
        assert_eq!(statistics["cov_percent"], 50.0);
        assert_eq!(statistics["p50"], 20.0);
        assert_eq!(statistics["p95"], 30.0);
        assert_eq!(statistics["p99"], 30.0);
        assert_eq!(stats(&[4.0], false)["cov_percent"], Value::Null);
        assert_eq!(stats(&[-2.0, 2.0], true)["cov_percent"], Value::Null);
        summary_excludes_warmup_failed_and_payload_mismatch_records();
    }

    fn summary_excludes_warmup_failed_and_payload_mismatch_records() {
        let temporary = tempfile::tempdir().expect("temporary directory should be created");
        let records_path = temporary.path().join("runs.jsonl");
        let pairs_path = temporary.path().join("pairs.jsonl");
        let runs = [
            json!({"model_key":"model","prompt_key":"prompt","pair_index":0,"arm":"off","status":"captured","wall_ms":10.0,"ttft_ms":1.0,"reported_tokens_per_second":10.0,"decode_intervals_ms":[100.0]}),
            json!({"model_key":"model","prompt_key":"prompt","pair_index":0,"arm":"on","status":"captured","wall_ms":8.0,"ttft_ms":0.8,"reported_tokens_per_second":12.0,"decode_intervals_ms":[80.0]}),
            json!({"model_key":"model","prompt_key":"prompt","pair_index":1,"arm":"off","status":"captured","wall_ms":30.0,"ttft_ms":3.0,"reported_tokens_per_second":5.0,"decode_intervals_ms":[200.0]}),
            json!({"model_key":"model","prompt_key":"prompt","pair_index":1,"arm":"on","status":"captured","wall_ms":20.0,"ttft_ms":2.0,"reported_tokens_per_second":6.0,"decode_intervals_ms":[150.0]}),
            json!({"model_key":"model","prompt_key":"prompt","pair_index":2,"arm":"off","status":"captured","wall_ms":40.0,"decode_intervals_ms":[300.0]}),
            json!({"model_key":"model","prompt_key":"prompt","pair_index":2,"arm":"on","status":"captured","wall_ms":5.0,"decode_intervals_ms":[50.0]}),
            json!({"model_key":"model","prompt_key":"prompt","pair_index":3,"arm":"on","status":"process_failed","wall_ms":1.0}),
        ];
        let pairs = [
            json!({"model_key":"model","prompt_key":"prompt","pair_index":0,"warmup":true,"usable_for_timing_summary":true,"on_eligible":true,"on_minus_off_wall_ms":-2.0}),
            json!({"model_key":"model","prompt_key":"prompt","pair_index":1,"warmup":false,"usable_for_timing_summary":true,"on_eligible":true,"on_minus_off_wall_ms":-10.0}),
            json!({"model_key":"model","prompt_key":"prompt","pair_index":2,"warmup":false,"usable_for_timing_summary":false,"on_eligible":true,"payload_match":false,"on_minus_off_wall_ms":-35.0}),
            json!({"model_key":"model","prompt_key":"prompt","pair_index":3,"warmup":false,"usable_for_timing_summary":false,"on_eligible":false,"on_status":"process_failed","on_minus_off_wall_ms":1.0}),
        ];
        let runs_text = runs
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        let pairs_text = pairs
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&records_path, runs_text).expect("run fixture should be written");
        fs::write(&pairs_path, pairs_text).expect("pair fixture should be written");
        let summary =
            derive_summary(&records_path, &pairs_path).expect("summary should be derived");
        assert_eq!(
            summary["matched_payload_draft_groups"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        assert_eq!(
            summary["matched_payload_draft_groups"][0]["off_wall_ms"]["sample_count"],
            1
        );
        assert_eq!(
            summary["matched_payload_draft_groups"][0]["on_wall_ms"]["sample_count"],
            1
        );
        assert_eq!(
            summary["matched_payload_draft_groups"][0]["matched_pair_on_minus_off_wall_ms"]["mean"],
            -10.0
        );
        assert_eq!(
            summary["matched_payload_draft_groups"][0]["on_decode_interval_ms"]["sample_count"],
            1
        );
        let observations = summary["observed_arm_distributions"]
            .as_array()
            .expect("observational groups should exist");
        assert_eq!(observations.len(), 2);
        assert!(
            observations
                .iter()
                .all(|group| group["wall_ms"]["sample_count"] == 2)
        );
        assert!(
            observations
                .iter()
                .all(|group| group["observational_only"] == true
                    && group["model_fidelity"] == "unassessed")
        );
    }

    #[test]
    fn zero_exit_with_failure_marker_is_rejected() {
        let model = test_model("granitemoe", "Granite 3.1 1b A400M Instruct");
        let (status, eligible, _) = validate_run(
            &fixture_output("GGUF PARSE FAILED: fixture\n"),
            "GGUF PARSE FAILED: fixture",
            "",
            &RunPayload::default(),
            &model,
        );
        assert_eq!(status, "failure_marker");
        assert!(!eligible);
    }

    #[test]
    fn cpu_fallback_is_rejected_even_when_output_contains_tokens() {
        let model = test_model("granitemoe", "Granite 3.1 1b A400M Instruct");
        let (status, eligible, _) = validate_run(
            &fixture_output("falling back to CPU\n"),
            "falling back to CPU",
            "",
            &RunPayload {
                ids: &[318],
                ..RunPayload::default()
            },
            &model,
        );
        assert_eq!(status, "cpu_fallback");
        assert!(!eligible);
    }

    #[test]
    fn missing_gpu_counter_and_empty_ids_are_not_captured() {
        let model = test_model("granitemoe", "Granite 3.1 1b A400M Instruct");
        let basic = "general.architecture = granitemoe\ngeneral.name = Granite 3.1 1b A400M Instruct\nbackend_requested = GPU/METAL (requested, gpu_layers = GPU_LAYERS_ALL)\n";
        let (counter_status, _, _) = validate_run(
            &fixture_output(basic),
            basic,
            "",
            &RunPayload::default(),
            &model,
        );
        assert_eq!(counter_status, "no_gpu_execution_evidence");
        let with_counter = format!("{basic}INFO token_breakdown_metal: gpu_exec_calls=1\n");
        let (empty_status, _, _) = validate_run(
            &fixture_output(&with_counter),
            &with_counter,
            "",
            &RunPayload::default(),
            &model,
        );
        assert_eq!(empty_status, "invalid_token_count");
    }
}
