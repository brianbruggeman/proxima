//! Slice M0: per-kernel GPU-time census of ONE warm gemma4-E2B decode evaluation.
//!
//! Method: isolated replay. The decode runs through the normal path
//! (`LoadedModel::generate_streaming`, same `ServingConfig` as
//! `decode_gbps_baseline`, speculation at the production default). The last
//! evaluation's dispatches (a verify evaluation is several rows wide; its width
//! is printed) are captured LIVE by
//! `omega`'s `PROXIMA_CAPTURE_LIVE` hook (`omega::take_captured_dispatches`:
//! real pipeline, real resolved buffers, uniform bytes, grid), grouped by
//! kernel identity plus launch geometry, and each group's representative is
//! replayed alone in its own command buffer, timed with
//! `GPUEndTime - GPUStartTime`.
//!
//! Three replay arms per group, all reported:
//! - cold: a blit-fill flush streams `M0_FLUSH_MIB` through memory first,
//!   then one command buffer holding one dispatch. Includes the
//!   per-command-buffer floor. This is the isolated-replay number the brief asks for.
//! - warm: the same command buffer back to back, no flush (weights may sit in the
//!   system-level cache; reported so the cache effect is visible).
//! - marginal: (time of `M0_BATCH` dispatches in one serial encoder minus
//!   time of one) / (`M0_BATCH` - 1): the in-situ shape, floor amortized.
//!
//! The per-command-buffer floor is measured with an empty kernel and reported
//! separately. The whole-step reference (`gpu_exec_ms`, host clock around
//! commit-to-wait; `gpu_busy_ms`, GPU clock first-start to last-end) is
//! parsed from this run's own `token_breakdown_metal` / `token_breakdown_gpu`
//! telemetry events, written by a file exporter under `M0_OUT_DIR`.
//!
//! # Run
//! ```sh
//! cd /Users/brianbruggeman/repos/slot-0/proxima && \
//! CARGO_TARGET_DIR=/private/tmp/cargo_target_lc_m0 cargo run --release -p proxima-model-interop \
//!   --features std,metal,instrument,metal-fuse-attn-decode --example gemma4_decode_kernel_census
//! ```
//! Knobs: `M0_MAX_TOKENS` (24), `M0_ITERS` (50), `M0_BATCH` (16),
//! `M0_FLUSH_MIB` (256), `M0_OUT_DIR`, `PROXIMA_PROMPT`, or
//! `PROXIMA_PROMPT_FILE`.
#![allow(clippy::expect_used, clippy::unwrap_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
use std::collections::{BTreeMap, HashMap};
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::fmt::Write as _;
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::fs::File;
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::io::Write as _;
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::path::{Path, PathBuf};

#[cfg(all(feature = "metal", target_os = "macos"))]
const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
#[cfg(all(feature = "metal", target_os = "macos"))]
const DEFAULT_OUT_DIR: &str = "/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/m0";
#[cfg(all(feature = "metal", target_os = "macos"))]
const DEFAULT_PROMPT: &str = "<|turn>user\nWhich of these is smaller in size: a hippopotamus or a large office building?<turn|>\n<|turn>model\n";
#[cfg(all(feature = "metal", target_os = "macos"))]
const HEAD_MIN_OUTPUT_EXTENT: u64 = 65536;
#[cfg(all(feature = "metal", target_os = "macos"))]
const PROJECTION_MIN_REDUCTION: u64 = 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg(all(feature = "metal", target_os = "macos"))]
enum Class {
    Matvec(String),
    Head,
    RmsSumsq,
    RmsSumsqEpilogue,
    NormApply,
    Rope,
    SoftmaxMax,
    SoftmaxExp,
    SoftmaxSum,
    AttentionDot,
    AttentionAv,
    CachedAttentionPartial,
    CachedAttentionMerge,
    CandidateB,
    IdentityCopy,
    Elementwise(String),
    Constant,
    Iota,
    Unclassified(String),
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Class {
    fn label(&self) -> String {
        match self {
            Class::Matvec(codec) => format!("matvec {codec}"),
            Class::Head => "head".to_string(),
            Class::RmsSumsq => "RMSNorm sumsq".to_string(),
            Class::RmsSumsqEpilogue => "RMSNorm sumsq + fused epilogue".to_string(),
            Class::NormApply => "norm apply".to_string(),
            Class::Rope => "RoPE".to_string(),
            Class::SoftmaxMax => "softmax max".to_string(),
            Class::SoftmaxExp => "softmax exp".to_string(),
            Class::SoftmaxSum => "softmax sum".to_string(),
            Class::AttentionDot => "attention dot".to_string(),
            Class::AttentionAv => "attention AV".to_string(),
            Class::CachedAttentionPartial => "cached attention partial".to_string(),
            Class::CachedAttentionMerge => "cached attention merge".to_string(),
            Class::CandidateB => "Candidate B (cached_softmax_weights)".to_string(),
            Class::IdentityCopy => "identity copy".to_string(),
            Class::Elementwise(body) => format!("elementwise {body}"),
            Class::Constant => "constant".to_string(),
            Class::Iota => "iota".to_string(),
            Class::Unclassified(entry) => format!("UNCLASSIFIED {entry}"),
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
struct Facts<'a> {
    entry: &'a str,
    kind_name: &'a str,
    operands: &'a [(u32, String)],
    extents: &'a [u64],
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn elementwise_body(entry: &str) -> Option<&str> {
    let rest = entry.strip_prefix("omega_elementwise_")?;
    rest.splitn(3, '_').nth(2)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn reduce_parts(entry: &str) -> Option<(&str, &str)> {
    let rest = entry.strip_prefix("omega_reduce_")?.splitn(4, '_').nth(3)?;
    if let Some(index) = rest.find("_maximum_") {
        return Some((&rest[..index], "maximum"));
    }
    rest.find("_add_zero").map(|index| (&rest[..index], "add"))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn codec_label(codec: &str) -> String {
    let lowered = codec.to_ascii_lowercase();
    match lowered.as_str() {
        text if text.contains("q4_0") || text.contains("q40") => "Q4_0".to_string(),
        text if text.contains("q6") => "Q6_K".to_string(),
        text if text.contains("q8_0") || text.contains("q80") => "Q8_0".to_string(),
        _ => codec.to_string(),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn packed_codec(operands: &[(u32, String)]) -> Option<String> {
    operands
        .iter()
        .find(|(_, codec)| codec != "unpacked")
        .map(|(_, codec)| codec_label(codec))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn produced_by(facts: &Facts<'_>, producers: &HashMap<u32, Class>, wanted: &[Class]) -> bool {
    facts.operands.iter().any(|(node, _)| {
        producers
            .get(node)
            .is_some_and(|class| wanted.contains(class))
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn classify_reduce(facts: &Facts<'_>, producers: &HashMap<u32, Class>) -> Class {
    let Some((body, reduce_op)) = reduce_parts(facts.entry) else {
        return Class::Unclassified(facts.entry.to_string());
    };
    if reduce_op == "maximum" {
        return Class::SoftmaxMax;
    }
    let same_operand = facts.operands.len() == 2 && facts.operands[0].0 == facts.operands[1].0;
    let output_extent = facts.extents.iter().copied().max().unwrap_or(0);
    let reduction_extent = facts.extents.last().copied().unwrap_or(0);
    if body.contains("multiply") && same_operand {
        return match facts.entry.contains("_epi") {
            true => Class::RmsSumsqEpilogue,
            false => Class::RmsSumsq,
        };
    }
    if let Some(codec) = packed_codec(facts.operands) {
        return match output_extent >= HEAD_MIN_OUTPUT_EXTENT {
            true => Class::Head,
            false => Class::Matvec(codec),
        };
    }
    if body == "identity" || body.contains("exponential") {
        return Class::SoftmaxSum;
    }
    if !body.contains("multiply") {
        return Class::Unclassified(facts.entry.to_string());
    }
    if output_extent >= HEAD_MIN_OUTPUT_EXTENT {
        return Class::Head;
    }
    let softmax_weights = [Class::CandidateB, Class::SoftmaxExp, Class::SoftmaxSum];
    let attention_shaped = facts.extents.len() >= 4 && reduction_extent < PROJECTION_MIN_REDUCTION;
    match (
        attention_shaped,
        produced_by(facts, producers, &softmax_weights),
    ) {
        (true, true) => Class::AttentionAv,
        (true, false) => Class::AttentionDot,
        (false, _) => Class::Matvec("f32".to_string()),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn classify_elementwise(facts: &Facts<'_>) -> Class {
    let Some(body) = elementwise_body(facts.entry) else {
        return Class::Unclassified(facts.entry.to_string());
    };
    if body == "identity" {
        return Class::IdentityCopy;
    }
    if body.contains("exponential") {
        return Class::SoftmaxExp;
    }
    if body.contains("square_root") {
        return Class::NormApply;
    }
    let rotates = body.contains("multiply")
        && (body.contains("subtract") || body.contains("add"))
        && facts.operands.len() >= 4;
    match rotates {
        true => Class::Rope,
        false => Class::Elementwise(body.to_string()),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn classify(facts: &Facts<'_>, producers: &HashMap<u32, Class>) -> Class {
    match facts.kind_name {
        "cached_attention" if facts.entry.ends_with("_merge") => Class::CachedAttentionMerge,
        "cached_attention" => Class::CachedAttentionPartial,
        "cached_softmax_weights" => Class::CandidateB,
        "constant" => Class::Constant,
        "iota" => Class::Iota,
        "elementwise" => classify_elementwise(facts),
        "keep::reduce fold" => classify_reduce(facts, producers),
        _ => Class::Unclassified(facts.entry.to_string()),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).expect("timings are finite"));
    match sorted.len() {
        0 => 0.0,
        length if length % 2 == 1 => sorted[length / 2],
        length => (sorted[length / 2 - 1] + sorted[length / 2]) / 2.0,
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn coefficient_of_variation(values: &[f64]) -> f64 {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count.max(1.0);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count.max(1.0);
    match mean > 0.0 {
        true => variance.sqrt() / mean * 100.0,
        false => 0.0,
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn field_after<'line>(line: &'line str, key: &str) -> Option<&'line str> {
    let marker = format!(" {key}=");
    let start = line.find(&marker)? + marker.len();
    let rest = &line[start..];
    Some(rest.split_whitespace().next().unwrap_or(rest))
}

#[derive(Debug, Clone, Default)]
#[cfg(all(feature = "metal", target_os = "macos"))]
struct StepStats {
    gpu_exec_ms: Option<f64>,
    physical_dispatch_calls: Option<u64>,
    gpu_busy_ms: Option<f64>,
    rows_evaluated: Option<usize>,
    tokens_committed: Option<usize>,
}

/// A `step` is the decode loop's token index, but only a step that is not a
/// queued-token pop runs an evaluation, so with speculation on a verify
/// evaluation at step `s` commits several tokens and steps `s+1..` evaluate
/// nothing. Each evaluation emits one `token_breakdown_metal` and one
/// `token_breakdown_gpu` line; the k-th gpu line belongs to the k-th
/// evaluated step. An evaluation with no `speculative_verify` line is a plain
/// one-row, one-token step.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn parse_step_stats(log: &str) -> BTreeMap<usize, StepStats> {
    let mut steps: BTreeMap<usize, StepStats> = BTreeMap::new();
    let mut gpu_lines: Vec<f64> = Vec::new();
    let mut verifies: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for line in log.lines() {
        if line.contains("token_breakdown_metal:") {
            let Some(step) = field_after(line, "step").and_then(|text| text.parse::<usize>().ok())
            else {
                continue;
            };
            let stats = steps.entry(step).or_default();
            stats.gpu_exec_ms = field_after(line, "gpu_exec_ms").and_then(|text| text.parse().ok());
            stats.physical_dispatch_calls =
                field_after(line, "physical_dispatch_calls").and_then(|text| text.parse().ok());
        } else if line.contains("token_breakdown_gpu")
            && let Some(busy) = field_after(line, "gpu_busy_ms").and_then(|text| text.parse().ok())
        {
            gpu_lines.push(busy);
        } else if line.contains("speculative_verify")
            && let Some(step) = field_after(line, "step").and_then(|text| text.parse().ok())
            && let Some(draft_len) =
                field_after(line, "draft_len").and_then(|text| text.parse::<usize>().ok())
            && let Some(emitted) =
                field_after(line, "emitted").and_then(|text| text.parse::<usize>().ok())
        {
            verifies.insert(step, (draft_len + 1, emitted));
        }
    }
    for (stats, busy) in steps.values_mut().zip(gpu_lines) {
        stats.gpu_busy_ms = Some(busy);
    }
    for (step, stats) in &mut steps {
        let (rows, committed) = verifies.get(step).copied().unwrap_or((1, 1));
        stats.rows_evaluated = Some(rows);
        stats.tokens_committed = Some(committed);
    }
    steps
}

/// Every decode step from 1 up: step 0 is the prompt prefill, and which later
/// steps run an evaluation is only known after the run.
#[cfg(all(feature = "metal", target_os = "macos"))]
fn capture_steps_arg(max_tokens: usize) -> String {
    (1..max_tokens)
        .map(|step| step.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod harness {
    use core::ops::ControlFlow;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use memmap2::{Mmap, MmapOptions};
    use omega::CapturedDispatch;
    use proxima_gguf::parse_complete;
    use proxima_gguf::types::GgmlType;
    use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig, TokenEvent};
    use proxima_telemetry::export::Exporter;
    use proxima_telemetry::recorder::Recorder;

    use super::*;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn unix_time_ns() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after the unix epoch")
            .as_nanos()
    }

    struct Config {
        max_tokens: usize,
        iterations: usize,
        batch: usize,
        flush_bytes: usize,
        out_dir: PathBuf,
    }

    impl Config {
        fn from_env() -> Self {
            Self {
                max_tokens: env_usize("M0_MAX_TOKENS", 24),
                iterations: env_usize("M0_ITERS", 50),
                batch: env_usize("M0_BATCH", 16),
                flush_bytes: env_usize("M0_FLUSH_MIB", 256) * 1024 * 1024,
                out_dir: PathBuf::from(
                    std::env::var("M0_OUT_DIR").unwrap_or_else(|_| DEFAULT_OUT_DIR.to_string()),
                ),
            }
        }
    }

    struct ReplaySample {
        gpu_span_ns: f64,
        host_start_offset_ns: u128,
        host_end_offset_ns: u128,
    }

    struct GroupMeasure {
        cold_ns: f64,
        cold_cov_percent: f64,
        warm_ns: f64,
        marginal_ns: f64,
        started_unix_ns: u128,
        completed_unix_ns: u128,
        cold_samples: Vec<ReplaySample>,
        warm_samples: Vec<ReplaySample>,
        batched_samples: Vec<ReplaySample>,
        failure: Option<String>,
    }

    struct Group {
        representative: usize,
        members: Vec<usize>,
        class: Class,
        measure: GroupMeasure,
    }

    fn install_telemetry(path: &Path) -> Arc<Recorder<proxima_telemetry::clock::GlobalClock>> {
        proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse(
            "debug",
        ));
        let recorder = Recorder::builder()
            .ring_capacity(262_144)
            .export(Exporter::file(path))
            .expect("file exporter installs")
            .install()
            .expect("telemetry recorder installs");
        let pump_recorder = Arc::clone(&recorder);
        let pumped = Arc::new(AtomicUsize::new(0));
        thread::Builder::new()
            .name("m0-telemetry-drain".to_string())
            .spawn(move || {
                loop {
                    pumped.fetch_add(pump_recorder.drain(), Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(5));
                }
            })
            .expect("spawn telemetry drain thread");
        recorder
    }

    fn arm_capture(max_tokens: usize) {
        let steps = capture_steps_arg(max_tokens);
        // SAFETY: called from `main` before any thread is spawned, so no
        // concurrent environment reader exists.
        unsafe {
            std::env::set_var("PROXIMA_CAPTURE_NODES", "all");
            std::env::set_var("PROXIMA_CAPTURE_STEPS", steps);
            std::env::set_var("PROXIMA_CAPTURE_LIVE", "1");
        }
    }

    fn decode(config: &Config) -> (usize, String) {
        let file = File::open(MODEL_PATH).expect("open gemma4-E2B blob");
        // SAFETY: read-only mapping of a checkpoint no other process writes.
        let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map gemma4-E2B blob");
        let parsed = parse_complete(&bytes).expect("parse gemma4-E2B header");
        let model = LoadedModel::load(&parsed, &bytes).expect("bind gemma4-E2B");
        let numeric_policy = match std::env::var("PROXIMA_EPILOGUE_SOURCES").as_deref() {
            Ok("0") => ServingConfig::default().numeric_policy.with_epilogue_sources(false),
            Ok("1") => ServingConfig::default().numeric_policy.with_epilogue_sources(true),
            Ok(other) => panic!("PROXIMA_EPILOGUE_SOURCES={other}: expected `0` or `1`"),
            Err(_) => ServingConfig::default().numeric_policy,
        };
        let serving_config = ServingConfig {
            gpu_layers: GPU_LAYERS_ALL,
            numeric_policy,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            reasoning_budget: 0,
            dispatch_type: omega::DispatchType::Serial,
            ..ServingConfig::default()
        };
        let prompt = match std::env::var("PROXIMA_PROMPT_FILE") {
            Ok(path) => std::fs::read_to_string(path).expect("read PROXIMA_PROMPT_FILE"),
            Err(_) => {
                std::env::var("PROXIMA_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string())
            }
        };
        let mut on_token = |_event: TokenEvent<'_>| ControlFlow::Continue(());
        let (token_ids, text, _stopped_by_eos) = model
            .generate_streaming(&prompt, config.max_tokens, serving_config, &mut on_token)
            .expect("greedy decode on the real gemma4-E2B checkpoint");
        (token_ids.len(), text)
    }

    fn classify_in_program_order(dispatches: &[CapturedDispatch]) -> Vec<Class> {
        let mut producers: HashMap<u32, Class> = HashMap::new();
        dispatches
            .iter()
            .map(|dispatch| {
                let facts = Facts {
                    entry: &dispatch.entry,
                    kind_name: dispatch.kind_name,
                    operands: &dispatch.operands,
                    extents: &dispatch.extents,
                };
                let class = classify(&facts, &producers);
                producers.insert(dispatch.node, class.clone());
                class
            })
            .collect()
    }

    fn group_by_identity(dispatches: &[CapturedDispatch], classes: &[Class]) -> Vec<Group> {
        let mut index_of: HashMap<(String, String, u64), usize> = HashMap::new();
        let mut groups: Vec<Group> = Vec::new();
        for (position, dispatch) in dispatches.iter().enumerate() {
            let key = (
                dispatch.msl_sha256.clone(),
                format!("{:?}", dispatch.grid),
                fnv64(&dispatch.uniform_bytes),
            );
            match index_of.get(&key) {
                Some(&existing) => groups[existing].members.push(position),
                None => {
                    index_of.insert(key, groups.len());
                    groups.push(Group {
                        representative: position,
                        members: vec![position],
                        class: classes[position].clone(),
                        measure: GroupMeasure {
                            cold_ns: 0.0,
                            cold_cov_percent: 0.0,
                            warm_ns: 0.0,
                            marginal_ns: 0.0,
                            started_unix_ns: 0,
                            completed_unix_ns: 0,
                            cold_samples: Vec::new(),
                            warm_samples: Vec::new(),
                            batched_samples: Vec::new(),
                            failure: None,
                        },
                    });
                }
            }
        }
        groups
    }

    fn timed_gpu_sample(
        dispatch: &CapturedDispatch,
        batch_dispatches: usize,
        group_origin: Instant,
    ) -> Result<ReplaySample, String> {
        let host_start_offset_ns = group_origin.elapsed().as_nanos();
        let gpu_span_ns = dispatch
            .time_gpu_ns(batch_dispatches)
            .map_err(|error| error.to_string())?;
        let host_end_offset_ns = group_origin.elapsed().as_nanos();
        Ok(ReplaySample {
            gpu_span_ns,
            host_start_offset_ns,
            host_end_offset_ns,
        })
    }

    fn sample_cold(
        dispatch: &CapturedDispatch,
        config: &Config,
        group_origin: Instant,
    ) -> Result<Vec<ReplaySample>, String> {
        (0..config.iterations)
            .map(|_| {
                omega::flush_gpu_caches(config.flush_bytes).map_err(|error| error.to_string())?;
                timed_gpu_sample(dispatch, 1, group_origin)
            })
            .collect()
    }

    fn sample_warm(
        dispatch: &CapturedDispatch,
        config: &Config,
        group_origin: Instant,
    ) -> Result<Vec<ReplaySample>, String> {
        (0..config.iterations)
            .map(|_| timed_gpu_sample(dispatch, 1, group_origin))
            .collect()
    }

    fn sample_batched(
        dispatch: &CapturedDispatch,
        config: &Config,
        group_origin: Instant,
    ) -> Result<Vec<ReplaySample>, String> {
        let rounds = (config.iterations / 5).max(9);
        (0..rounds)
            .map(|_| timed_gpu_sample(dispatch, config.batch, group_origin))
            .collect()
    }

    fn measure_group(dispatch: &CapturedDispatch, config: &Config) -> GroupMeasure {
        let started_unix_ns = unix_time_ns();
        let group_origin = Instant::now();
        let arms = (|| -> Result<GroupMeasure, String> {
            let cold = sample_cold(dispatch, config, group_origin)?;
            let warm = sample_warm(dispatch, config, group_origin)?;
            let batched = sample_batched(dispatch, config, group_origin)?;
            let cold_spans: Vec<f64> = cold.iter().map(|sample| sample.gpu_span_ns).collect();
            let warm_spans: Vec<f64> = warm.iter().map(|sample| sample.gpu_span_ns).collect();
            let batched_spans: Vec<f64> = batched.iter().map(|sample| sample.gpu_span_ns).collect();
            let warm_ns = median(&warm_spans);
            let marginal_ns =
                ((median(&batched_spans) - warm_ns) / (config.batch as f64 - 1.0)).max(0.0);
            Ok(GroupMeasure {
                cold_ns: median(&cold_spans),
                cold_cov_percent: coefficient_of_variation(&cold_spans),
                warm_ns,
                marginal_ns,
                started_unix_ns,
                completed_unix_ns: unix_time_ns(),
                cold_samples: cold,
                warm_samples: warm,
                batched_samples: batched,
                failure: None,
            })
        })();
        arms.unwrap_or_else(|reason| GroupMeasure {
            cold_ns: 0.0,
            cold_cov_percent: 0.0,
            warm_ns: 0.0,
            marginal_ns: 0.0,
            started_unix_ns: 0,
            completed_unix_ns: 0,
            cold_samples: Vec::new(),
            warm_samples: Vec::new(),
            batched_samples: Vec::new(),
            failure: Some(reason),
        })
    }

    fn measure_floor(samples: usize) -> Vec<f64> {
        (0..samples)
            .map(|_| {
                omega::time_empty_command_buffer_gpu_ns().expect("empty command buffer replays")
            })
            .collect()
    }

    #[derive(Default)]
    struct ClassTotals {
        dispatches: usize,
        cold_ns: f64,
        floor_subtracted_ns: f64,
        warm_ns: f64,
        marginal_ns: f64,
    }

    fn totals_by_class(groups: &[Group], floor_ns: f64) -> BTreeMap<Class, ClassTotals> {
        let mut totals: BTreeMap<Class, ClassTotals> = BTreeMap::new();
        for group in groups {
            let count = group.members.len() as f64;
            let entry = totals.entry(group.class.clone()).or_default();
            entry.dispatches += group.members.len();
            entry.cold_ns += group.measure.cold_ns * count;
            entry.floor_subtracted_ns += (group.measure.cold_ns - floor_ns).max(0.0) * count;
            entry.warm_ns += group.measure.warm_ns * count;
            entry.marginal_ns += group.measure.marginal_ns * count;
        }
        totals
    }

    fn render_table(totals: &BTreeMap<Class, ClassTotals>, step_reference_us: f64) -> String {
        let mut rows: Vec<(&Class, &ClassTotals)> = totals.iter().collect();
        rows.sort_by(|left, right| {
            right
                .1
                .cold_ns
                .partial_cmp(&left.1.cold_ns)
                .expect("finite")
        });
        let mut table = String::new();
        writeln!(
            table,
            "{:<40} {:>10} {:>12} {:>14} {:>8} | {:>12} {:>12} | {:>12} {:>12}",
            "class",
            "disp/step",
            "us/disp cold",
            "total us cold",
            "% step",
            "us/disp -flr",
            "total -flr",
            "us/disp marg",
            "total marg"
        )
        .expect("write to string");
        for (class, row) in rows {
            let count = row.dispatches as f64;
            writeln!(
                table,
                "{:<40} {:>10} {:>12.2} {:>14.1} {:>7.1}% | {:>12.2} {:>12.1} | {:>12.2} {:>12.1}",
                class.label(),
                row.dispatches,
                row.cold_ns / count / 1e3,
                row.cold_ns / 1e3,
                row.cold_ns / 1e3 / step_reference_us * 100.0,
                row.floor_subtracted_ns / count / 1e3,
                row.floor_subtracted_ns / 1e3,
                row.marginal_ns / count / 1e3,
                row.marginal_ns / 1e3,
            )
            .expect("write to string");
        }
        table
    }

    fn write_group_csv(path: &Path, dispatches: &[CapturedDispatch], groups: &[Group]) {
        let mut csv = File::create(path).expect("create groups csv");
        writeln!(
            csv,
            "class,count,entry,msl_sha256,grid_threads,threadgroup_width,uniform_fnv,rep_node,cold_ns,cold_cov_pct,warm_ns,marginal_ns,failure"
        )
        .expect("write csv header");
        for group in groups {
            let rep = &dispatches[group.representative];
            writeln!(
                csv,
                "{},{},{},{},{},{:?},{:016x},{},{:.1},{:.2},{:.1},{:.1},{}",
                group.class.label(),
                group.members.len(),
                rep.entry,
                rep.msl_sha256,
                rep.grid.threads,
                rep.grid.threadgroup_width,
                fnv64(&rep.uniform_bytes),
                rep.node,
                group.measure.cold_ns,
                group.measure.cold_cov_percent,
                group.measure.warm_ns,
                group.measure.marginal_ns,
                group
                    .measure
                    .failure
                    .clone()
                    .unwrap_or_default()
                    .replace(',', ";"),
            )
            .expect("write csv row");
        }
    }

    fn write_sample_trace_csv(
        path: &Path,
        dispatches: &[CapturedDispatch],
        groups: &[Group],
        batch_dispatches: usize,
    ) {
        let mut csv = File::create(path).expect("create timing sample trace csv");
        writeln!(
            csv,
            "group_index,step,chunk_index,node,class,entry,group_started_unix_ns,group_completed_unix_ns,sample_started_unix_ns,sample_completed_unix_ns,arm,sample_index,batch_dispatches,gpu_span_ns"
        )
        .expect("write timing sample trace header");
        for (group_index, group) in groups.iter().enumerate() {
            let dispatch = &dispatches[group.representative];
            for (arm, batch_dispatches, samples) in [
                ("cold", 1, &group.measure.cold_samples),
                ("warm", 1, &group.measure.warm_samples),
                ("batched", batch_dispatches, &group.measure.batched_samples),
            ] {
                for (sample_index, sample) in samples.iter().enumerate() {
                    writeln!(
                        csv,
                        "{group_index},{},{},{},{},{},{},{},{},{},{arm},{sample_index},{batch_dispatches},{:.1}",
                        dispatch.step,
                        dispatch.chunk_index,
                        dispatch.node,
                        group.class.label(),
                        dispatch.entry,
                        group.measure.started_unix_ns,
                        group.measure.completed_unix_ns,
                        group.measure.started_unix_ns + sample.host_start_offset_ns,
                        group.measure.started_unix_ns + sample.host_end_offset_ns,
                        sample.gpu_span_ns,
                    )
                    .expect("write timing sample trace row");
                }
            }
        }
    }

    fn write_dispatch_manifest(path: &Path, dispatches: &[CapturedDispatch], classes: &[Class]) {
        let mut csv = File::create(path).expect("create dispatch manifest");
        writeln!(
            csv,
            "index,step,chunk_index,node,class,kind,entry,grid_threads,operand_codecs,extents"
        )
        .expect("write manifest header");
        for (index, (dispatch, class)) in dispatches.iter().zip(classes).enumerate() {
            let codecs: Vec<&str> = dispatch
                .operands
                .iter()
                .map(|(_, codec)| codec.as_str())
                .collect();
            writeln!(
                csv,
                "{index},{},{},{},{},{},{},{},{},{:?}",
                dispatch.step,
                dispatch.chunk_index,
                dispatch.node,
                class.label(),
                dispatch.kind_name,
                dispatch.entry,
                dispatch.grid.threads,
                codecs.join("|"),
                dispatch.extents,
            )
            .expect("write manifest row");
        }
    }

    fn warm_median(steps: &BTreeMap<usize, StepStats>, pick: fn(&StepStats) -> Option<f64>) -> f64 {
        let values: Vec<f64> = steps
            .iter()
            .filter(|(step, _)| **step >= 2)
            .filter_map(|(_, stats)| pick(stats))
            .collect();
        median(&values)
    }

    pub fn run() {
        let config = Config::from_env();
        std::fs::create_dir_all(&config.out_dir).expect("create M0_OUT_DIR");
        let telemetry_path = config.out_dir.join("decode_telemetry.log");
        let _ = std::fs::remove_file(&telemetry_path);
        arm_capture(config.max_tokens);
        let recorder = install_telemetry(&telemetry_path);

        let (generated, text) = decode(&config);
        while recorder.drain() > 0 {}
        println!("m0 decode: generated_tokens={generated} text={text:?}");

        let mut dispatches = omega::take_captured_dispatches();
        assert!(
            !dispatches.is_empty(),
            "N==0: no dispatch captured (capture step never ran on this thread)"
        );
        let capture_step = dispatches
            .iter()
            .map(|dispatch| dispatch.step)
            .max()
            .expect("non-empty");
        dispatches.retain(|dispatch| dispatch.step == capture_step);
        let launched: Vec<CapturedDispatch> = dispatches
            .into_iter()
            .filter(|dispatch| dispatch.grid.threads > 0)
            .collect();
        println!(
            "m0 capture: step={capture_step} dispatches_with_threads={}",
            launched.len()
        );

        let telemetry = std::fs::read_to_string(&telemetry_path).expect("read decode telemetry");
        let steps = parse_step_stats(&telemetry);
        assert!(
            !steps.is_empty(),
            "N==0: no token_breakdown_metal step parsed from {telemetry_path:?}"
        );
        let last_logged_step = *steps.keys().max().expect("non-empty");
        let capture_stats = steps
            .get(&(capture_step as usize))
            .cloned()
            .unwrap_or_default();

        let classes = classify_in_program_order(&launched);
        let mut groups = group_by_identity(&launched, &classes);
        println!(
            "m0 groups: {} kernels-by-identity-and-geometry over {} dispatches",
            groups.len(),
            launched.len()
        );
        let floor_samples = measure_floor(200);
        let floor_ns = median(&floor_samples);
        let group_total = groups.len();
        let sample_trace_path = config.out_dir.join("census_timing_samples.csv");
        for (position, group) in groups.iter_mut().enumerate() {
            group.measure = measure_group(&launched[group.representative], &config);
            if position % 50 == 0 {
                println!("m0 replay: group {position}/{group_total}");
            }
        }
        let sequence_replay_ns: Vec<f64> = (0..3)
            .map(|_| CapturedDispatch::time_gpu_sequence_ns(&launched))
            .collect::<Result<_, _>>()
            .expect("replay the captured dispatch sequence");
        println!(
            "m0 sequence replay: one command buffer, dispatches={}, gpu_ms={:?}",
            launched.len(),
            sequence_replay_ns
                .iter()
                .map(|elapsed_ns| elapsed_ns / 1e6)
                .collect::<Vec<_>>()
        );
        let chunk_sequence_replay_ns: Vec<Vec<f64>> = (0..3)
            .map(|_| CapturedDispatch::time_gpu_chunk_sequences_ns(&launched))
            .collect::<Result<_, _>>()
            .expect("replay the captured command-buffer chunks");
        println!(
            "m0 chunk sequence replay: chunks={}, gpu_ms={:?}",
            chunk_sequence_replay_ns.first().map_or(0, Vec::len),
            chunk_sequence_replay_ns
                .iter()
                .map(|chunks| {
                    chunks
                        .iter()
                        .map(|elapsed_ns| elapsed_ns / 1e6)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        );
        write_sample_trace_csv(&sample_trace_path, &launched, &groups, config.batch);
        report(&Census {
            config: &config,
            launched: &launched,
            classes: &classes,
            groups: &groups,
            floor_ns,
            floor_samples: &floor_samples,
            steps: &steps,
            capture_step,
            capture_stats,
            last_logged_step,
        });
    }

    struct Census<'a> {
        config: &'a Config,
        launched: &'a [CapturedDispatch],
        classes: &'a [Class],
        groups: &'a [Group],
        floor_ns: f64,
        floor_samples: &'a [f64],
        steps: &'a BTreeMap<usize, StepStats>,
        capture_step: u64,
        capture_stats: StepStats,
        last_logged_step: usize,
    }

    fn report(census: &Census<'_>) {
        let Census {
            config,
            launched,
            classes,
            groups,
            floor_ns,
            floor_samples,
            steps,
            capture_step,
            capture_stats,
            last_logged_step,
        } = census;
        let (floor_ns, capture_step, last_logged_step) =
            (*floor_ns, *capture_step, *last_logged_step);
        write_group_csv(&config.out_dir.join("census_groups.csv"), launched, groups);
        write_dispatch_manifest(
            &config.out_dir.join("census_dispatches.csv"),
            launched,
            classes,
        );
        let failed: Vec<&Group> = groups
            .iter()
            .filter(|group| group.measure.failure.is_some())
            .collect();
        let totals = totals_by_class(groups, floor_ns);
        let reference_ms = capture_stats
            .gpu_busy_ms
            .or(capture_stats.gpu_exec_ms)
            .expect("capture step has a whole-step reference");
        let step_reference_us = reference_ms * 1e3;
        println!("\n{}", render_table(&totals, step_reference_us));

        let sum_cold: f64 = totals.values().map(|row| row.cold_ns).sum::<f64>() / 1e3;
        let sum_floor_sub: f64 = totals
            .values()
            .map(|row| row.floor_subtracted_ns)
            .sum::<f64>()
            / 1e3;
        let sum_warm: f64 = totals.values().map(|row| row.warm_ns).sum::<f64>() / 1e3;
        let sum_marginal: f64 = totals.values().map(|row| row.marginal_ns).sum::<f64>() / 1e3;
        let counted: usize = totals.values().map(|row| row.dispatches).sum();
        println!(
            "m0 floor: empty command buffer GPU span median={:.2} us min={:.2} us max={:.2} us n={}",
            floor_ns / 1e3,
            floor_samples.iter().copied().fold(f64::INFINITY, f64::min) / 1e3,
            floor_samples.iter().copied().fold(0.0, f64::max) / 1e3,
            floor_samples.len()
        );
        println!(
            "m0 step (captured step {capture_step}, last evaluation that ran): rows_evaluated={:?} tokens_committed={:?} gpu_busy_ms={:?} gpu_exec_ms={:?} physical_dispatch_calls={:?}",
            capture_stats.rows_evaluated,
            capture_stats.tokens_committed,
            capture_stats.gpu_busy_ms,
            capture_stats.gpu_exec_ms,
            capture_stats.physical_dispatch_calls
        );
        println!(
            "m0 step (median over logged evaluations at step >= 2, mixed widths): gpu_busy_ms={:.3} gpu_exec_ms={:.3}",
            warm_median(steps, |stats| stats.gpu_busy_ms),
            warm_median(steps, |stats| stats.gpu_exec_ms)
        );
        for (label, total_us) in [
            ("cold", sum_cold),
            ("cold-minus-floor", sum_floor_sub),
            ("warm", sum_warm),
            ("marginal", sum_marginal),
        ] {
            println!(
                "m0 reconcile: sum(median x count) [{label}]={total_us:.1} us; ratio to gpu_busy_ms={:.3}; ratio to gpu_exec_ms={:.3}",
                total_us / (capture_stats.gpu_busy_ms.unwrap_or(f64::NAN) * 1e3),
                total_us / (capture_stats.gpu_exec_ms.unwrap_or(f64::NAN) * 1e3)
            );
        }
        println!(
            "m0 invariant 1: sum(group members)={counted} captured_dispatches={} physical_dispatch_calls_logged={:?} capture_step_is_last_logged_step={}",
            launched.len(),
            capture_stats.physical_dispatch_calls,
            capture_step as usize == last_logged_step
        );
        println!("m0 replay failures: {}", failed.len());
        for group in &failed {
            println!(
                "m0 replay failure: class={} entry={} reason={}",
                group.class.label(),
                launched[group.representative].entry,
                group.measure.failure.clone().unwrap_or_default()
            );
        }
        assert_eq!(
            counted,
            launched.len(),
            "invariant 1: distinct kernels x counts must equal captured dispatches"
        );
        assert_eq!(
            Some(launched.len() as u64),
            capture_stats.physical_dispatch_calls,
            "invariant 1: captured dispatches must equal the step's physical_dispatch_calls"
        );
        assert!(
            failed.is_empty(),
            "{} kernel groups failed to replay",
            failed.len()
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    harness::run();
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    println!("gemma4_decode_kernel_census requires --features metal on macOS");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts<'a>(
        entry: &'a str,
        kind_name: &'a str,
        operands: &'a [(u32, String)],
        extents: &'a [u64],
    ) -> Facts<'a> {
        Facts {
            entry,
            kind_name,
            operands,
            extents,
        }
    }

    fn unpacked(nodes: &[u32]) -> Vec<(u32, String)> {
        nodes
            .iter()
            .map(|node| (*node, "unpacked".to_string()))
            .collect()
    }

    #[test]
    fn packed_q4_0_reduce_is_a_q4_0_matvec() {
        let operands = vec![(10, "Q4_0".to_string()), (11, "unpacked".to_string())];
        let class = classify(
            &facts(
                "omega_reduce_r3_o2_n2_multiply_add_zero",
                "keep::reduce fold",
                &operands,
                &[1, 6144, 1536],
            ),
            &HashMap::new(),
        );
        assert_eq!(class, Class::Matvec("Q4_0".to_string()));
    }

    #[test]
    fn packed_q6_k_vocab_wide_reduce_is_the_head() {
        let operands = vec![(10, "Q6K".to_string()), (11, "unpacked".to_string())];
        let class = classify(
            &facts(
                "omega_reduce_r3_o2_n2_multiply_add_zero",
                "keep::reduce fold",
                &operands,
                &[1, 262144, 1536],
            ),
            &HashMap::new(),
        );
        assert_eq!(class, Class::Head);
    }

    #[test]
    fn packed_q8_0_reduce_is_a_q8_0_matvec() {
        let operands = vec![(10, "Q8_0".to_string()), (11, "unpacked".to_string())];
        let class = classify(
            &facts(
                "omega_reduce_r3_o2_n2_multiply_add_zero",
                "keep::reduce fold",
                &operands,
                &[1, 1536, 6144],
            ),
            &HashMap::new(),
        );
        assert_eq!(class, Class::Matvec("Q8_0".to_string()));
    }

    #[test]
    fn reduce_of_one_operand_squared_is_rms_sumsq() {
        let operands = unpacked(&[876, 876]);
        let class = classify(
            &facts(
                "omega_reduce_r3_o2_n2_multiply_add_zero",
                "keep::reduce fold",
                &operands,
                &[1, 1, 512],
            ),
            &HashMap::new(),
        );
        assert_eq!(class, Class::RmsSumsq);
    }

    #[test]
    fn maximum_reduce_is_softmax_max() {
        let operands = unpacked(&[5]);
        let class = classify(
            &facts(
                "omega_reduce_r3_o2_n1_identity_maximum_negative_infinity",
                "keep::reduce fold",
                &operands,
                &[1, 8, 32],
            ),
            &HashMap::new(),
        );
        assert_eq!(class, Class::SoftmaxMax);
    }

    #[test]
    fn unpacked_rank5_reduce_is_dot_until_it_reads_candidate_b_output_then_av() {
        let entry = "omega_reduce_r5_o4_n2_multiply_add_zero";
        let dot_operands = unpacked(&[837, 891]);
        let av_operands = unpacked(&[900, 837]);
        let producers = HashMap::from([(900, Class::CandidateB)]);
        let extents = [1, 32, 1, 8, 256];
        assert_eq!(
            classify(
                &facts(entry, "keep::reduce fold", &dot_operands, &extents),
                &producers
            ),
            Class::AttentionDot
        );
        assert_eq!(
            classify(
                &facts(entry, "keep::reduce fold", &av_operands, &extents),
                &producers
            ),
            Class::AttentionAv
        );
    }

    #[test]
    fn cached_softmax_weights_kind_is_candidate_b() {
        let operands = unpacked(&[1, 2, 3]);
        let class = classify(
            &facts(
                "omega_cached_softmax_weights_x",
                "cached_softmax_weights",
                &operands,
                &[8, 32],
            ),
            &HashMap::new(),
        );
        assert_eq!(class, Class::CandidateB);
    }

    #[test]
    fn rotation_body_with_four_operands_is_rope_not_norm_apply() {
        let operands = unpacked(&[1, 2, 3, 4]);
        let class = classify(
            &facts(
                "omega_elementwise_r3_n4_fused_multiply_o0_o1__multiply_o2_o3__subtract_s0_s1",
                "elementwise",
                &operands,
                &[1, 8, 128],
            ),
            &HashMap::new(),
        );
        assert_eq!(class, Class::Rope);
    }

    #[test]
    fn fused_rsqrt_apply_body_is_norm_apply() {
        let operands = unpacked(&[1, 2, 3, 4, 5, 6]);
        let entry = "omega_elementwise_r2_n6_fused_multiply_o4_o5__add_s0_o3__square_root_s1__reciprocal_s2__multiply_s3_o2__multiply_s4_o1__add_s5_o0";
        let class = classify(
            &facts(entry, "elementwise", &operands, &[1, 1536]),
            &HashMap::new(),
        );
        assert_eq!(class, Class::NormApply);
    }

    #[test]
    fn projection_shaped_unpacked_rank5_reduce_is_an_f32_matvec_not_attention() {
        let operands = unpacked(&[10, 11]);
        let class = classify(
            &facts(
                "omega_reduce_r5_o2_n2_multiply_add_zero",
                "keep::reduce fold",
                &operands,
                &[1, 1, 8, 256, 1536],
            ),
            &HashMap::new(),
        );
        assert_eq!(class, Class::Matvec("f32".to_string()));
    }

    #[test]
    fn sumsq_with_a_fused_epilogue_is_reported_separately() {
        let operands = unpacked(&[7, 7]);
        let entry =
            "omega_reduce_r2_o1_n2_multiply_add_zero_epi4_fused_identity_o4__multiply_s0_o3";
        let class = classify(
            &facts(entry, "keep::reduce fold", &operands, &[1, 1536]),
            &HashMap::new(),
        );
        assert_eq!(class, Class::RmsSumsqEpilogue);
    }

    #[test]
    fn unknown_entry_is_reported_unclassified_never_dropped() {
        let operands = unpacked(&[1]);
        let class = classify(
            &facts("omega_something_new", "keep::scan fold", &operands, &[4]),
            &HashMap::new(),
        );
        assert_eq!(
            class,
            Class::Unclassified("omega_something_new".to_string())
        );
    }

    #[test]
    fn step_stats_parse_metal_and_gpu_lines_by_evaluation_order() {
        let log = "t INFO m: token_breakdown_metal: per-decode-step metal stage attribution step=3 prepare_calls=1 physical_dispatch_calls=1154 gpu_exec_ms=14.25 readback_ms=0.1\n\
                   t DEBUG o: token_breakdown_gpu commit_call_ms=0.1 gpu_busy_ms=13.5 chunks=1\n";
        let steps = parse_step_stats(log);
        assert_eq!(steps[&3].physical_dispatch_calls, Some(1154));
        assert_eq!(steps[&3].gpu_exec_ms, Some(14.25));
        assert_eq!(steps[&3].gpu_busy_ms, Some(13.5));
    }

    #[test]
    fn step_stats_pair_gpu_lines_with_evaluated_steps_and_read_verify_width() {
        let log = "t INFO m: token_breakdown_metal: x step=1 physical_dispatch_calls=900 gpu_exec_ms=20.0\n\
                   t DEBUG o: token_breakdown_gpu gpu_busy_ms=11.0 chunks=1\n\
                   t DEBUG d: speculative_verify step=4 draft_len=3 accepted=2 emitted=3 drafter=Some(NgramSimple)\n\
                   t INFO m: token_breakdown_metal: x step=4 physical_dispatch_calls=1000 gpu_exec_ms=25.0\n\
                   t DEBUG o: token_breakdown_gpu gpu_busy_ms=14.0 chunks=1\n";
        let steps = parse_step_stats(log);
        assert_eq!(steps.keys().copied().collect::<Vec<_>>(), vec![1, 4]);
        assert_eq!(steps[&1].gpu_busy_ms, Some(11.0));
        assert_eq!(steps[&4].gpu_busy_ms, Some(14.0));
        assert_eq!(steps[&1].rows_evaluated, Some(1));
        assert_eq!(steps[&1].tokens_committed, Some(1));
        assert_eq!(steps[&4].rows_evaluated, Some(4));
        assert_eq!(steps[&4].tokens_committed, Some(3));
    }

    #[test]
    fn capture_steps_cover_every_decode_step_after_prefill() {
        assert_eq!(capture_steps_arg(5), "1,2,3,4");
        assert_eq!(capture_steps_arg(2), "1");
        assert_eq!(capture_steps_arg(1), "");
    }

    #[test]
    fn step_stats_of_a_log_with_no_step_lines_is_empty() {
        assert!(parse_step_stats("nothing relevant here\n").is_empty());
    }

    #[test]
    fn median_of_even_and_odd_counts() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod real_program_names {
    use memmap2::Mmap;
    use omega::PackedOperands;
    use proxima_gguf::parse_complete;
    use proxima_model_interop::{Architecture, GEMMA4, bind_symbols};
    use proxima_tensor::spec::Qwen35LayerRoots;
    use proxima_tensor::{NodeId, NumericPolicy, bind_with_fusion, infer, prune_dead};

    use super::*;

    const NEW_COUNT: usize = 1;
    const KV_BUCKET_EXTENT: usize = 32;

    fn production_outputs(logits_root: NodeId, layer_roots: &[Qwen35LayerRoots]) -> Vec<NodeId> {
        let mut outputs = vec![logits_root];
        for roots in layer_roots {
            if let Qwen35LayerRoots::Attention((even, odd, value)) = roots {
                outputs.extend([*even, *odd, *value]);
            }
        }
        outputs
    }

    fn emitted_facts(
        fused: bool,
    ) -> Vec<(u32, String, &'static str, Vec<(u32, String)>, Vec<u64>)> {
        let file = File::open(MODEL_PATH).expect("real gemma4-E2B blob must exist for this test");
        // SAFETY: read-only mapping of a checkpoint no other process writes.
        let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real checkpoint");
        let parsed = parse_complete(&mapping).expect("parse the real checkpoint header");
        let bound_program = GEMMA4
            .bind(&parsed, &mapping)
            .expect("bind gemma4 production program");
        let outputs = production_outputs(bound_program.logits_root, &bound_program.layer_roots);
        let symbols = bind_symbols(
            NEW_COUNT,
            KV_BUCKET_EXTENT,
            &[],
            bound_program.single_position_step,
        )
        .expect("bind symbols");
        let shapes = infer(&bound_program.program, &symbols).expect("infer shapes");
        let policy = NumericPolicy::llama_relaxed();
        let bound_ops = bind_with_fusion(&bound_program.program, &shapes, &outputs, fused, policy)
            .expect("bind with fusion");
        let bound_ops = prune_dead(bound_ops, &outputs);
        let packed = PackedOperands::new();
        bound_ops
            .iter()
            .map(|bound| {
                let kernel = omega::emit(bound, &packed, policy).expect("emit kernel");
                let operands = bound
                    .operands()
                    .iter()
                    .map(|(node, _layout, _lookup)| (node.0, "unpacked".to_string()))
                    .collect();
                (
                    bound.node.0,
                    kernel.entry,
                    bound.kind.name(),
                    operands,
                    bound.extents.clone(),
                )
            })
            .collect()
    }

    fn histogram(fused: bool) -> BTreeMap<String, usize> {
        let mut producers: HashMap<u32, Class> = HashMap::new();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (node, entry, kind_name, operands, extents) in emitted_facts(fused) {
            let class = classify(
                &Facts {
                    entry: &entry,
                    kind_name,
                    operands: &operands,
                    extents: &extents,
                },
                &producers,
            );
            *counts.entry(class.label()).or_default() += 1;
            producers.insert(node, class);
        }
        counts
    }

    fn detail_histogram(fused: bool) -> BTreeMap<String, usize> {
        let mut producers: HashMap<u32, Class> = HashMap::new();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (node, entry, kind_name, operands, extents) in emitted_facts(fused) {
            let class = classify(
                &Facts {
                    entry: &entry,
                    kind_name,
                    operands: &operands,
                    extents: &extents,
                },
                &producers,
            );
            let distinct_operands = operands
                .iter()
                .map(|(operand, _)| *operand)
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            *counts
                .entry(format!(
                    "{} | {entry} | operands={} distinct={distinct_operands} | extents={extents:?}",
                    class.label(),
                    operands.len()
                ))
                .or_default() += 1;
            producers.insert(node, class);
        }
        counts
    }

    #[test]
    fn real_fused_decode_program_detail_is_recorded_per_entry_and_extents() {
        let counts = detail_histogram(true);
        write_histogram("classifier_cpu_detail_fused.txt", &counts);
        assert!(
            counts.len() > 20,
            "only {} distinct (class, entry, extents) rows",
            counts.len()
        );
    }

    fn write_histogram(name: &str, counts: &BTreeMap<String, usize>) {
        let out_dir = std::env::var("M0_OUT_DIR").unwrap_or_else(|_| DEFAULT_OUT_DIR.to_string());
        std::fs::create_dir_all(&out_dir).expect("create out dir");
        let mut text = String::new();
        for (label, count) in counts {
            writeln!(text, "{count:6} {label}").expect("write to string");
        }
        std::fs::write(Path::new(&out_dir).join(name), &text).expect("write histogram");
        println!("{name}\n{text}");
    }

    #[test]
    fn every_kernel_of_the_real_fused_decode_program_gets_a_class() {
        let counts = histogram(true);
        write_histogram("classifier_cpu_histogram_fused.txt", &counts);
        let total: usize = counts.values().sum();
        let unclassified: Vec<&String> = counts
            .keys()
            .filter(|label| label.starts_with("UNCLASSIFIED"))
            .collect();
        assert!(total > 1000, "N: only {total} bound ops classified");
        assert!(
            unclassified.is_empty(),
            "unclassified entries: {unclassified:?}"
        );
    }

    #[test]
    fn real_fused_decode_program_exposes_every_structural_class() {
        let counts = histogram(true);
        for label in [
            "RMSNorm sumsq",
            "RMSNorm sumsq + fused epilogue",
            "cached attention partial",
            "norm apply",
            "RoPE",
            "matvec f32",
            "head",
            "identity copy",
        ] {
            assert!(
                counts.get(label).copied().unwrap_or(0) > 0,
                "no kernel classified as {label}: {counts:?}"
            );
        }
    }

    #[test]
    fn real_unfused_decode_program_exposes_the_softmax_chain() {
        let counts = histogram(false);
        write_histogram("classifier_cpu_histogram_unfused.txt", &counts);
        for label in [
            "softmax max",
            "softmax exp",
            "softmax sum",
            "attention dot",
            "attention AV",
        ] {
            assert!(
                counts.get(label).copied().unwrap_or(0) > 0,
                "no kernel classified as {label}: {counts:?}"
            );
        }
        let unclassified: Vec<&String> = counts
            .keys()
            .filter(|label| label.starts_with("UNCLASSIFIED"))
            .collect();
        assert!(
            unclassified.is_empty(),
            "unclassified entries: {unclassified:?}"
        );
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("unsupported target for this benchmark or example");
    std::process::exit(1);
}
