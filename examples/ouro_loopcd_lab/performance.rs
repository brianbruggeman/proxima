use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::mem::{MaybeUninit, size_of};
use std::ops::ControlFlow;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use memmap2::MmapOptions;
use omega::backend;
use proxima_model_interop::{
    ContextLength, LoadedModel, Phase, PromptCacheConfig, ServingConfig, SpeculativeConfig,
};
use proxima_telemetry::debug;
use proxima_telemetry::emit::EnvFilter;
use proxima_telemetry::emit::global::install as install_emit_filter;
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::recorder::Recorder;
use proxima_tensor::instrument::ticks_to_nanos;
use proxima_tensor::spec::input_leaf;
use proxima_tensor::{DType, Extent, QuantizedBlock};
use proxima_tokenizer::bpe::decode_ids;
use serde_json::{Value, json};

use super::checkpoint::{Checkpoint, DIGEST, REVISION, WeightMappingMode};
use super::controls;
use super::engine::Executor;
use super::graph::{self, Graph};
use super::guidance::Guidance;
use super::model::{Arm, Model, Readouts, pick};
use super::runner::{Arguments, record, stats};

const PANEL_PATH: &str = "proxima-tensor/specs/omega-programmable/frozen-prefixes.json";
const RUN_COUNT: usize = 5;

fn cached_panel() -> io::Result<Value> {
    let original: Value = serde_json::from_str(include_str!(
        "../../proxima-tensor/specs/omega-programmable/frozen-prefixes.json"
    ))
    .map_err(io::Error::other)?;
    let short: Vec<u32> = serde_json::from_value(original["cases"][1]["prefix_ids"].clone())
        .map_err(io::Error::other)?;
    if short.len() != 24 {
        return Err(io::Error::other("cached short prefix requires 24 IDs"));
    }
    let long: Vec<u32> = short.iter().copied().cycle().take(256).collect();
    Ok(
        json!({"checkpoint_sha256":DIGEST,"checkpoint_revision":REVISION,
        "source":"omega-programmable/frozen-prefixes.json ball case",
        "corpus":"synthetic length-control; not a quality corpus",
        "cases":[{"id":"ball24","prefix_ids":short},{"id":"repeated-ball256","prefix_ids":long}]}),
    )
}

pub(super) fn cached(arguments: &Arguments) -> io::Result<()> {
    let _lock = ResidentLock::acquire()?;
    let profile = arguments
        .gpu_profile
        .then(|| GpuProfile::install_cached(&arguments.output))
        .transpose()?;
    let panel = cached_panel()?;
    serde_json::to_writer_pretty(
        File::create(arguments.output.join("cached-panel.json"))?,
        &panel,
    )
    .map_err(io::Error::other)?;
    let mut records = BufWriter::new(File::create(arguments.output.join("cached.jsonl"))?);
    let path = "/private/tmp/ouro-llama-converted-v1/ouro-bf16.gguf";
    let digest = Command::new("/usr/bin/openssl")
        .args(["dgst", "-sha256", path])
        .output()?;
    let digest_text = String::from_utf8(digest.stdout).map_err(io::Error::other)?;
    let observed_digest = digest_text.split_whitespace().last();
    if !digest.status.success()
        || observed_digest
            != Some("d3cdf276005d768c5def9cfb6679eb5600a424c5cd23b45157e65204708ae478")
    {
        return Err(io::Error::other(format!(
            "cached GGUF digest verification failed: {digest_text}; {}",
            String::from_utf8_lossy(&digest.stderr)
        )));
    }
    let file = File::open(path)?;
    // safety: the immutable mapping outlives the borrowed model and every request.
    let mapping = unsafe { MmapOptions::new().map(&file)? };
    let setup_started = Instant::now();
    let parsed = proxima_gguf::pipe::parse_complete(&mapping).map_err(io::Error::other)?;
    let loaded = LoadedModel::load(&parsed, &mapping).map_err(io::Error::other)?;
    let config = ServingConfig {
        model_path: path,
        context_length: ContextLength::Within(320),
        batch_size: 256,
        ubatch_size: 256,
        gpu_layers: -1,
        temperature: 0.0,
        repeat_penalty: 1.0,
        prompt_cache: PromptCacheConfig::off(),
        speculative: SpeculativeConfig::none(),
        omega_runtime: arguments.omega_config.as_ref(),
        numeric_policy: arguments
            .omega_config
            .as_ref()
            .map_or(proxima_tensor::NumericPolicy::llama_relaxed(), |config| {
                config.numeric
            }),
        ..ServingConfig::default()
    };
    record(
        &mut records,
        &json!({"kind":"cached_setup","model_path":path,"checkpoint_source_sha256":DIGEST,
        "observed_gguf_sha256":observed_digest,"digest_source":"native openssl dgst -sha256",
        "omega_config":arguments.omega_config,
        "batch_size":config.batch_size,"ubatch_size":config.ubatch_size,"context_size":config.context_length.length(),
        "gpu_profile":arguments.gpu_profile,
        "timing_condition":if arguments.gpu_profile {"instrumentation-on; symmetric production arms"} else {"instrumentation-off"},
        "load_ms":setup_started.elapsed().as_secs_f64()*1000.0,"settings":format!("{config:?}"),
        "execution_owner":"LoadedModel::generate_from_ids","panel":panel,
        "build_dense_float_storage":omega::sized::DENSE_FLOAT_STORAGE_ENABLED,
        "build_tiled_min_tokens":omega::sized::DENSE_TILED_MIN_TOKENS}),
    )?;
    let mut request_count = 0;
    for case in panel["cases"]
        .as_array()
        .ok_or_else(|| io::Error::other("cached cases absent"))?
    {
        let prefix: Vec<u32> =
            serde_json::from_value(case["prefix_ids"].clone()).map_err(io::Error::other)?;
        let root = loaded.logits_root();
        let diagnostic = loaded
            .forward_node_values_from_ids(&prefix, &[root], &config)
            .map_err(io::Error::other)?;
        record(
            &mut records,
            &json!({"kind":"cached_diagnostic","case_id":case["id"],"prefix_ids":prefix,
            "requested_nodes":[root.0],"logits":diagnostic,"diagnostic_only":true}),
        )?;
        let mut walls = Vec::new();
        let mut first_tokens = Vec::new();
        let mut decode_latencies = Vec::new();
        for repetition in 0..=RUN_COUNT {
            let profile_fields = json!({"case_id":case["id"],"repetition":repetition,
                "warmup":repetition==0,"prefix_ids":prefix,"batch_size":config.batch_size,"ubatch_size":config.ubatch_size});
            let profile_begin = profile
                .as_ref()
                .map(|profile| profile.marker("ouro_gpu_profile_begin", profile_fields.clone()))
                .transpose()?;
            let before = ResourceSnapshot::capture();
            let started = Instant::now();
            let mut previous = 0.0;
            let mut callback_ids = Vec::new();
            let mut latencies = Vec::new();
            let mut events = Vec::new();
            let result = loaded.generate_from_ids(&prefix, arguments.max_new_tokens, &config, &mut |event| {
                let elapsed = started.elapsed().as_secs_f64()*1000.0;
                if event.phase == Phase::Token {
                    callback_ids.push(event.token_id);
                    latencies.push(elapsed - previous);
                    previous = elapsed;
                }
                events.push(json!({"id":event.token_id,"step":event.step,"phase":format!("{:?}",event.phase),"caller_elapsed_ms":elapsed,"loop_elapsed_ms":event.elapsed_ms}));
                ControlFlow::Continue(())
            });
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            let after = ResourceSnapshot::capture();
            let profile_end = profile
                .as_ref()
                .map(|profile| profile.marker("ouro_gpu_profile_end", profile_fields))
                .transpose()?;
            let (generated, text, stopped_by_eos) = match result {
                Ok(result) => result,
                Err(error) => {
                    record(
                        &mut records,
                        &json!({"kind":"cached_error","case_id":case["id"],"repetition":repetition,
                        "error":error.to_string(),"wall_ms":wall_ms,"events":events,"before":before.json(),"after":after.json()}),
                    )?;
                    records.flush()?;
                    return Err(io::Error::other(error));
                }
            };
            if generated != callback_ids || generated.len() != latencies.len() {
                return Err(io::Error::other(
                    "production returned IDs differ from actual callbacks",
                ));
            }
            record(
                &mut records,
                &json!({"kind":if repetition==0 {"cached_warmup"} else {"cached_sample"},
                "runtime":"proxima","case_id":case["id"],"repetition":repetition.saturating_sub(1),
                "prefix_ids":prefix,"generated_ids":generated,"generated_count":generated.len(),"text":text,
                "stopped_by_eos":stopped_by_eos,"max_new_tokens":arguments.max_new_tokens,
                "token_latencies_ms":latencies,"events":events,"wall_ms":wall_ms,"first_token_ms":latencies.first(),
                "gpu_profile":profile_begin.zip(profile_end).map(|(begin,end)| json!({"begin_drain_records":begin.0,"end_drain_records":end.0,"dropped_before":begin.1,"dropped_after":end.1,"outside_timer":true})),
                "response_tokens_per_second":generated.len() as f64*1000.0/wall_ms,
                "cpu_percent_one_core":before.cpu.seconds.zip(after.cpu.seconds).map(|(before,after)|100000.0*(after-before)/wall_ms),
                "before":before.json(),"after":after.json(),"kv_state_policy":"empty_per_request_then_cached_append",
                "type_k":"F32","type_v":"F32","depth":4,"error_count":0,
                "timing_provenance":"measured native production callback; setup and diagnostics separate"}),
            )?;
            records.flush()?;
            request_count += 1;
            if repetition != 0 {
                walls.push(wall_ms);
                if let Some(first) = latencies.first() {
                    first_tokens.push(*first);
                }
                decode_latencies.extend(latencies.into_iter().skip(1));
            }
            eprintln!(
                "cached request {} repetition {repetition}: {} generated tokens",
                case["id"],
                generated.len()
            );
        }
        record(
            &mut records,
            &json!({"kind":"cached_summary","case_id":case["id"],"wall":stats(&walls,0),
            "first_token":stats(&first_tokens,0),"decode_latency":stats(&decode_latencies,0),
            "cov_convention":"sample standard deviation with N-1 denominator","error_count":0}),
        )?;
    }
    if request_count != 12 {
        return Err(io::Error::other("cached cohort requires twelve requests"));
    }
    record(
        &mut records,
        &json!({"kind":"cached_complete","request_count":request_count,"case_count":2,"measured_samples":10,"warmups":2,"error_count":0}),
    )?;
    records.flush()
}

#[cfg(test)]
mod mainline_tests {
    use super::*;
    #[test]
    fn mainline_caller_cached_panel() {
        let panel = cached_panel().expect("frozen panel");
        let short = panel["cases"][0]["prefix_ids"]
            .as_array()
            .expect("short IDs");
        let long = panel["cases"][1]["prefix_ids"]
            .as_array()
            .expect("long IDs");
        assert_eq!(short.len(), 24);
        assert_eq!(long.len(), 256);
        for (index, token) in long.iter().enumerate() {
            assert_eq!(token, &short[index % 24]);
        }
        assert_ne!(long.len(), short.len());
        println!("{panel}");
    }
    #[test]
    fn mainline_caller_sample_statistics() {
        let actual = stats(&[1.0, 2.0, 3.0], 0);
        assert_eq!(actual["sample_count"], 3);
        assert_eq!(actual["mean_ms"], 2.0);
        assert_eq!(actual["sample_standard_deviation_ms"], 1.0);
        assert_eq!(actual["cov_percent"], 50.0);
        assert_eq!(actual["p50_ms"], 2.0);
        assert_eq!(actual["p99_ms"], 3.0);
        assert_ne!(actual["cov_percent"], 0.0);
        assert!(stats(&[0.0, 0.0], 0)["cov_percent"].is_null());
        println!("{actual}");
    }
}
const METHODS: [(&str, Guidance); 4] = [
    ("baseline4", Guidance::Baseline),
    ("fixed-logits-0.5", Guidance::FixedLogits { strength: 0.5 }),
    (
        "adaptive-logits-cap-1",
        Guidance::AdaptiveLogits { max_strength: 1.0 },
    ),
    ("fixed-hidden-0.5", Guidance::FixedHidden { strength: 0.5 }),
];

struct Case {
    identifier: String,
    prefix: Vec<u32>,
}

struct Selection {
    case_id: String,
    repetition: usize,
    method_index: usize,
    sample_id: String,
    selected_id: Value,
    selected_bytes: Value,
    selected_spelling: Value,
}

struct GpuProfile {
    recorder: Arc<Recorder>,
}

impl GpuProfile {
    fn install(output: &Path) -> io::Result<Self> {
        Self::install_with_filter(
            output,
            "warn,omega::metal::placements_execute_named=debug,ouro_loopcd_lab::performance=debug",
        )
    }

    fn install_cached(output: &Path) -> io::Result<Self> {
        Self::install_with_filter(
            output,
            "warn,proxima_model_interop::generate::decode=debug,omega::metal::arena_encode_dispatch_finish=debug,omega::metal::pipeline_buffers_upload=debug,ouro_loopcd_lab::performance=debug",
        )
    }

    fn install_with_filter(output: &Path, filter: &str) -> io::Result<Self> {
        install_emit_filter(EnvFilter::parse(filter));
        let file_path = output.join("gpu-events.jsonl");
        let recorder = Recorder::builder()
            .ring_capacity(65_536)
            .export(
                Exporter::fan(vec![
                    Exporter::std().format(Formatter::Json),
                    Exporter::file(file_path).format(Formatter::Json),
                ])
                .map_err(io::Error::other)?
                .format(Formatter::Json),
            )
            .map_err(io::Error::other)?
            .core_count(1)
            .install()
            .map_err(io::Error::other)?;
        Ok(Self { recorder })
    }

    fn drain(&self) -> io::Result<(usize, u64)> {
        let mut total_drained = 0;
        loop {
            let batch_drained = self.recorder.drain();
            total_drained += batch_drained;
            if batch_drained == 0 {
                break;
            }
        }
        let dropped = self.recorder.dropped();
        if dropped != 0 {
            return Err(io::Error::other(format!(
                "GPU profile recorder dropped {dropped} telemetry records"
            )));
        }
        Ok((total_drained, dropped))
    }

    fn marker(&self, marker: &str, fields: Value) -> io::Result<(usize, u64)> {
        match marker {
            "ouro_gpu_profile_begin" => debug!(
                marker = "ouro_gpu_profile_begin",
                fields = %fields,
                "gpu profile bracket marker recorded"
            ),
            "ouro_gpu_profile_end" => debug!(
                marker = "ouro_gpu_profile_end",
                fields = %fields,
                "gpu profile bracket marker recorded"
            ),
            _ => debug!(
                marker = "ouro_gpu_profile_setup",
                fields = %fields,
                "gpu profile setup marker recorded"
            ),
        }
        self.drain()
    }
}

struct ResidentLock(File);

impl ResidentLock {
    fn acquire() -> io::Result<Self> {
        let cache_directory = PathBuf::from(env::var_os("HOME").ok_or_else(|| {
            io::Error::other("HOME is unavailable for the native resident-model lock")
        })?)
        .join(".cache/proxima");
        fs::create_dir_all(&cache_directory)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(cache_directory.join("resident-model.lock"))?;
        // safety: flock only observes the live file descriptor and stores no pointers.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(io::Error::other(format!(
                "resident-model lock is held (errno {})",
                io::Error::last_os_error()
            )));
        }
        Ok(Self(file))
    }
}

impl Drop for ResidentLock {
    fn drop(&mut self) {
        // safety: the descriptor remains open for the lifetime of this guard.
        unsafe {
            libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[derive(Clone)]
struct CpuResource {
    seconds: Option<f64>,
    peak_rss_bytes: Option<u64>,
    error: Option<String>,
}

#[derive(Clone)]
struct TaskResource {
    current_rss_bytes: Option<u64>,
    physical_footprint_bytes: Option<u64>,
    error: Option<String>,
}

#[derive(Clone)]
struct HostResource {
    load_1m: Option<f64>,
    load_5m: Option<f64>,
    load_15m: Option<f64>,
    error: Option<String>,
}

#[derive(Clone)]
struct ResourceSnapshot {
    cpu: CpuResource,
    task: TaskResource,
    host: HostResource,
}

impl ResourceSnapshot {
    fn capture() -> Self {
        Self {
            cpu: cpu_resource(),
            task: task_resource(),
            host: host_resource(),
        }
    }

    fn json(&self) -> Value {
        json!({
            "process_cpu_seconds": self.cpu.seconds,
            "process_peak_rss_bytes": self.cpu.peak_rss_bytes,
            "current_rss_bytes": self.task.current_rss_bytes,
            "physical_footprint_bytes": self.task.physical_footprint_bytes,
            "host_load_1m": self.host.load_1m,
            "host_load_5m": self.host.load_5m,
            "host_load_15m": self.host.load_15m,
            "sampler_errors": {
                "getrusage": self.cpu.error,
                "task_info": self.task.error,
                "getloadavg": self.host.error,
            }
        })
    }
}

#[cfg(target_os = "macos")]
fn cpu_resource() -> CpuResource {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    // safety: getrusage initializes the caller-owned structure on success.
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return CpuResource {
            seconds: None,
            peak_rss_bytes: None,
            error: Some(io::Error::last_os_error().to_string()),
        };
    }
    // safety: a zero result means getrusage initialized the value.
    let usage = unsafe { usage.assume_init() };
    let user = usage.ru_utime.tv_sec as f64 + usage.ru_utime.tv_usec as f64 / 1_000_000.0;
    let system = usage.ru_stime.tv_sec as f64 + usage.ru_stime.tv_usec as f64 / 1_000_000.0;
    CpuResource {
        seconds: Some(user + system),
        peak_rss_bytes: Some(usage.ru_maxrss as u64),
        error: None,
    }
}

#[cfg(not(target_os = "macos"))]
fn cpu_resource() -> CpuResource {
    CpuResource {
        seconds: None,
        peak_rss_bytes: None,
        error: Some("native performance resource sampler requires macOS".to_owned()),
    }
}

#[cfg(target_os = "macos")]
fn task_resource() -> TaskResource {
    #[repr(C)]
    #[derive(Default)]
    struct TaskVmInfo {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
    }
    unsafe extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: i32,
            output: *mut TaskVmInfo,
            count: *mut u32,
        ) -> i32;
    }
    let mut info = TaskVmInfo::default();
    let mut count = (size_of::<TaskVmInfo>() / size_of::<u32>()) as u32;
    // safety: the output buffer mirrors the versioned TASK_VM_INFO prefix and count.
    let result = unsafe { task_info(mach_task_self(), 22, &mut info, &mut count) };
    if result == 0 {
        TaskResource {
            current_rss_bytes: Some(info.resident_size),
            physical_footprint_bytes: Some(info.phys_footprint),
            error: None,
        }
    } else {
        TaskResource {
            current_rss_bytes: None,
            physical_footprint_bytes: None,
            error: Some(format!("task_info returned {result}")),
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn task_resource() -> TaskResource {
    TaskResource {
        current_rss_bytes: None,
        physical_footprint_bytes: None,
        error: Some("native performance resource sampler requires macOS".to_owned()),
    }
}

#[cfg(target_os = "macos")]
fn host_resource() -> HostResource {
    let mut values = [0.0; 3];
    // safety: getloadavg writes at most three doubles into the supplied array.
    let count = unsafe { libc::getloadavg(values.as_mut_ptr(), values.len() as i32) };
    if count == 3 {
        HostResource {
            load_1m: Some(values[0]),
            load_5m: Some(values[1]),
            load_15m: Some(values[2]),
            error: None,
        }
    } else {
        HostResource {
            load_1m: None,
            load_5m: None,
            load_15m: None,
            error: Some(format!("getloadavg returned {count} values")),
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn host_resource() -> HostResource {
    HostResource {
        load_1m: None,
        load_5m: None,
        load_15m: None,
        error: Some("native performance resource sampler requires macOS".to_owned()),
    }
}

fn read_cases(path: &Path, checkpoint: &Checkpoint) -> io::Result<Vec<Case>> {
    let manifest: Value = serde_json::from_slice(&fs::read(path)?).map_err(io::Error::other)?;
    if manifest["checkpoint_revision"] != REVISION || manifest["checkpoint_sha256"] != DIGEST {
        return Err(io::Error::other(
            "frozen prefixes belong to another checkpoint",
        ));
    }
    let records = manifest["cases"]
        .as_array()
        .ok_or_else(|| io::Error::other("frozen panel cases absent"))?;
    let mut cases = Vec::new();
    for wanted in ["france", "ball"] {
        let value = records
            .iter()
            .find(|record| record["id"].as_str() == Some(wanted))
            .ok_or_else(|| io::Error::other(format!("frozen panel case {wanted} absent")))?;
        let prefix = value["prefix_ids"]
            .as_array()
            .ok_or_else(|| io::Error::other("frozen prefix ids absent"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| io::Error::other("frozen token id is not a u32"))
            })
            .collect::<io::Result<Vec<_>>>()?;
        let text = value["text"]
            .as_str()
            .ok_or_else(|| io::Error::other("frozen prefix text absent"))?;
        if decode_ids(&prefix, &checkpoint.vocab).map_err(io::Error::other)? != text.as_bytes() {
            return Err(io::Error::other(format!(
                "frozen prefix bytes differ for {wanted}"
            )));
        }
        cases.push(Case {
            identifier: wanted.to_owned(),
            prefix,
        });
    }
    Ok(cases)
}

fn arm(index: usize) -> Arm {
    Arm {
        name: METHODS[index].0,
        depth: 4,
        guidance: METHODS[index].1,
    }
}

fn payload(readouts: &Readouts) -> Value {
    json!({
        "logits": readouts.logits,
        "reference": readouts.reference,
        "head_logits": readouts.head_logits,
        "raw_reference_hidden": readouts.raw_reference_hidden,
        "raw_terminal_hidden": readouts.raw_terminal_hidden,
        "contrasted_hidden": readouts.contrasted_hidden,
        "normalized_hidden": readouts.normalized_hidden,
    })
}

fn compare_payload(old: &Value, new: &Value) -> io::Result<Vec<Value>> {
    let fields = [
        "logits",
        "reference",
        "head_logits",
        "raw_reference_hidden",
        "raw_terminal_hidden",
        "contrasted_hidden",
        "normalized_hidden",
    ];
    let mut differences = Vec::new();
    for field in fields {
        let previous = old
            .get(field)
            .ok_or_else(|| io::Error::other(format!("old payload field missing: {field}")))?;
        let current = new
            .get(field)
            .ok_or_else(|| io::Error::other(format!("new payload field missing: {field}")))?;
        if previous.is_null() || current.is_null() {
            if previous.is_null() != current.is_null() {
                return Err(io::Error::other(format!(
                    "payload field presence differs for {field}"
                )));
            }
            continue;
        }
        let previous = previous
            .as_array()
            .ok_or_else(|| io::Error::other(format!("payload {field} is not an array")))?;
        let current = current
            .as_array()
            .ok_or_else(|| io::Error::other(format!("payload {field} is not an array")))?;
        if previous.len() != current.len() {
            return Err(io::Error::other(format!(
                "payload field length differs for {field}"
            )));
        }
        for (coordinate, (old_value, new_value)) in previous.iter().zip(current).enumerate() {
            let old_number = old_value
                .as_f64()
                .ok_or_else(|| io::Error::other("non-numeric old payload value"))?;
            let new_number = new_value
                .as_f64()
                .ok_or_else(|| io::Error::other("non-numeric new payload value"))?;
            if old_number != new_number {
                differences.push(json!({
                    "field": field,
                    "coordinate": coordinate,
                    "old": old_number,
                    "new": new_number,
                    "difference": new_number - old_number,
                }));
            }
        }
    }
    Ok(differences)
}

fn expected_payload_rejection(old: &Value, new: &Value, label: &str) -> io::Result<String> {
    match compare_payload(old, new) {
        Ok(_) => Err(io::Error::other(format!(
            "payload comparator failed to reject {label}"
        ))),
        Err(error) => Ok(error.to_string()),
    }
}

fn metal_snapshot() -> Value {
    let totals = omega::metal::metal_stage_totals();
    json!({
        "raw_debug": format!("{totals:?}"),
        "physical_dispatch_calls": totals.physical_dispatch_calls,
        "gpu_exec_calls": totals.gpu_exec_calls,
        "gpu_exec_ticks": totals.gpu_exec_ticks,
        "gpu_exec_host_ms": ticks_to_nanos(totals.gpu_exec_ticks) as f64 / 1_000_000.0,
        "readback_calls": totals.readback_calls,
        "readback_ticks": totals.readback_ticks,
        "readback_bytes": totals.readback_bytes,
        "encode_dispatch_calls": totals.encode_dispatch_calls,
        "encode_dispatch_ticks": totals.encode_dispatch_ticks,
        "block_upload_calls": totals.block_upload_calls,
        "block_upload_ticks": totals.block_upload_ticks,
        "block_offered_bytes": totals.block_offered_bytes,
        "block_copied_bytes": totals.block_copied_bytes,
        "block_nocopy_bound_bytes": totals.block_nocopy_bound_bytes,
        "block_offset_bound_bytes": totals.block_offset_bound_bytes,
        "output_buffer_allocations": totals.output_buffer_allocations,
        "output_buffer_allocated_bytes": totals.output_buffer_allocated_bytes,
        "gpu_exec_ticks_meaning": "host commit-through-wait duration; not GPU busy time",
    })
}

fn invocation(
    writer: &mut impl Write,
    model: &mut Model<'_>,
    case: &Case,
    method_index: usize,
    kind: &str,
    position: (usize, usize),
    profile: Option<&GpuProfile>,
) -> io::Result<(Value, f64)> {
    let checkpoint = model.checkpoint;
    let (repetition, order) = position;
    let selected_arm = arm(method_index);
    let metal_before = metal_snapshot();
    let sample_id = format!(
        "{}:{}:{}:{}",
        case.identifier, METHODS[method_index].0, kind, repetition
    );
    let mut profile_begin_drained = 0;
    let mut profile_dropped_before = 0;
    let expected_gpu_events = if method_index == 0 {
        101
    } else if method_index == 3 {
        102
    } else {
        103
    };
    if let Some(profile) = profile {
        (profile_begin_drained, profile_dropped_before) = profile.drain()?;
        profile.marker(
            "ouro_gpu_profile_begin",
            json!({"sample_id":sample_id,"case_id":case.identifier,"method":METHODS[method_index].0,"kind":kind,"repetition":repetition,"method_order_position":order,"expected_gpu_events":expected_gpu_events}),
        )?;
    }
    let resource_before = ResourceSnapshot::capture();
    let started = Instant::now();
    let readouts = model.forward_readouts(selected_arm, false)?;
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    let resource_after = ResourceSnapshot::capture();
    let mut profile_end_drained = 0;
    let mut profile_dropped_after = profile_dropped_before;
    if let Some(profile) = profile {
        (profile_end_drained, profile_dropped_after) = profile.marker(
            "ouro_gpu_profile_end",
            json!({"sample_id":sample_id,"case_id":case.identifier,"method":METHODS[method_index].0,"kind":kind,"repetition":repetition,"method_order_position":order,"expected_gpu_events":expected_gpu_events}),
        )?;
    }
    let metal = metal_snapshot();
    let output = payload(&readouts);
    let chosen = pick(&readouts.logits)?;
    let duration = wall_ms;
    let process_cpu_delta = match (resource_before.cpu.seconds, resource_after.cpu.seconds) {
        (Some(before), Some(after)) => Some(after - before),
        _ => None,
    };
    let mut row = json!({
        "kind": kind,
        "case_id": case.identifier,
        "method": METHODS[method_index].0,
        "repetition": repetition,
        "method_order_position": order,
        "context": model.context(selected_arm),
        "wall_ms": wall_ms,
        "process_cpu_delta_seconds": process_cpu_delta,
        "process_cpu_percent_one_core": process_cpu_delta.map(|seconds| 100.0 * seconds * 1000.0 / wall_ms),
        "resource_before": resource_before.json(),
        "resource_after": resource_after.json(),
        "resource_bracket_note": "resource reads bracket the measured forward; the wall timer surrounds only forward_readouts",
        "metal_stage_totals_before": metal_before,
        "metal_stage_totals_after": metal,
        "selected_id": chosen,
        "selected_bytes": checkpoint.vocab.token_bytes(chosen),
        "selected_spelling": checkpoint.vocab.token_str(chosen),
        "payload": output,
        "payload_difference_count": 0,
        "complete_payload_differences": [],
        "head_executions": readouts.head_executions,
        "layer_executions": readouts.layer_executions,
        "backend": "metal",
        "cpu_fallback_executed": false,
        "gpu_profile": profile.map(|_| json!({"expected_gpu_events":expected_gpu_events,"begin_drain_records":profile_begin_drained,"end_drain_records":profile_end_drained,"dropped_before":profile_dropped_before,"dropped_after":profile_dropped_after,"outside_timer":true})),
    });
    row["sample_id"] = json!(sample_id);
    record(
        writer,
        &json!({"kind":"performance_invocation_index","sample_id":row["sample_id"],"case_id":case.identifier,"method":METHODS[method_index].0,"payload_source":"samples.jsonl","context":row["context"]}),
    )?;
    Ok((row, duration))
}

fn projection_error_coordinates(
    expected: &[f64],
    actual: &[f32],
    bounds: &[f64],
) -> io::Result<Vec<usize>> {
    if expected.is_empty() || expected.len() != actual.len() || expected.len() != bounds.len() {
        return Err(io::Error::other(
            "projection comparator requires equal nonempty coordinate arrays",
        ));
    }
    Ok(expected
        .iter()
        .zip(actual)
        .zip(bounds)
        .enumerate()
        .filter_map(|(coordinate, ((expected, actual), bound))| {
            let delta = f64::from(*actual) - expected;
            (!delta.is_finite() || !bound.is_finite() || *bound < 0.0 || delta.abs() > *bound)
                .then_some(coordinate)
        })
        .collect())
}

pub(super) fn microbench(arguments: &Arguments) -> io::Result<()> {
    let _resident_lock = ResidentLock::acquire()?;
    let config = arguments.omega_config.as_ref().ok_or_else(|| {
        io::Error::other("projection microbench requires an explicit runtime config")
    })?;
    let checkpoint = Checkpoint::open(&arguments.directory, WeightMappingMode::WholeCheckpoint)?;
    let cases = read_cases(Path::new(PANEL_PATH), &checkpoint)?;
    let weight_name = "model.layers.0.self_attn.q_proj.weight";
    let entry = checkpoint
        .entries
        .get(weight_name)
        .ok_or_else(|| io::Error::other("projection microbench checkpoint query matrix absent"))?;
    let [columns, reduction] = entry.shape.as_slice() else {
        return Err(io::Error::other(
            "projection microbench requires a rank-two checkpoint weight",
        ));
    };
    let columns = *columns;
    let reduction = *reduction;
    let weight_bytes = checkpoint.bytes(weight_name)?;
    let decoded: Vec<f64> = weight_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|bytes| {
            f64::from(f32::from_bits(
                u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16,
            ))
        })
        .collect();
    if decoded.len() != columns * reduction || decoded.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::other(
            "projection microbench weight payload is malformed",
        ));
    }
    let mut writer = BufWriter::new(File::create(arguments.output.join("microbench.jsonl"))?);
    for rows in [1usize, 5, 24] {
        let prefix = if rows == 1 {
            cases
                .first()
                .and_then(|case| case.prefix.first())
                .copied()
                .map(|token| vec![token])
        } else {
            cases
                .iter()
                .find(|case| case.prefix.len() == rows)
                .map(|case| case.prefix.clone())
        }
        .ok_or_else(|| io::Error::other("frozen projection prefix length absent"))?;
        let activations = checkpoint.embedding_rows(&prefix)?;
        if activations.len() != rows * reduction {
            return Err(io::Error::other(
                "embedding width disagrees with query projection",
            ));
        }
        let mut expected = Vec::with_capacity(rows * columns);
        let mut absolute_bounds = Vec::with_capacity(rows * columns);
        let roundoff = reduction as f64 * f64::from(f32::EPSILON);
        if roundoff >= 1.0 {
            return Err(io::Error::other(
                "projection extent exceeds floating error-bound domain",
            ));
        }
        for activation in activations.chunks_exact(reduction) {
            for weight in decoded.chunks_exact(reduction) {
                let (sum, magnitude) = activation.iter().zip(weight).fold(
                    (0.0_f64, 0.0_f64),
                    |(sum, magnitude), (activation, weight)| {
                        let product = f64::from(*activation) * weight;
                        (sum + product, magnitude + product.abs())
                    },
                );
                expected.push(sum);
                absolute_bounds.push(roundoff / (1.0 - roundoff) * magnitude);
            }
        }
        let oracle_control: Vec<f32> = expected.iter().map(|value| *value as f32).collect();
        let mut corrupt_control = oracle_control.clone();
        corrupt_control[0] += (2.0 * absolute_bounds[0] + 1.0) as f32;
        let positive_violations =
            projection_error_coordinates(&expected, &oracle_control, &absolute_bounds)?;
        let negative_violations =
            projection_error_coordinates(&expected, &corrupt_control, &absolute_bounds)?;
        record(
            &mut writer,
            &json!({
                "kind":"projection_comparator_control", "rows":rows,
                "expected":expected, "bounds":absolute_bounds,
                "positive_payload":oracle_control, "negative_payload":corrupt_control,
                "positive_violations":positive_violations, "negative_violations":negative_violations,
            }),
        )?;
        writer.flush()?;
        if !positive_violations.is_empty() || negative_violations != [0] {
            return Err(io::Error::other(
                "projection comparator control disagrees with literal corruption location",
            ));
        }
        let mut program = Vec::new();
        let activation = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(u32::try_from(rows).map_err(io::Error::other)?),
                Extent::Static(u32::try_from(reduction).map_err(io::Error::other)?),
            ],
            "activation",
        );
        let weight = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(u32::try_from(reduction).map_err(io::Error::other)?),
                Extent::Static(u32::try_from(columns).map_err(io::Error::other)?),
            ],
            "weight",
        );
        let root = graph::projection(&mut program, activation, weight)?;
        let bindings = [
            ("activation", QuantizedBlock::Float32(&activations)),
            ("weight", checkpoint.packed(weight_name)?),
        ];
        let setup_started = Instant::now();
        let mut executor = Executor::prepare_with_config(
            Graph {
                program,
                root,
                taps: Vec::new(),
            },
            "metal",
            rows,
            &bindings,
            false,
            Some(config),
        )?;
        let setup_id = format!("projection-{rows}-setup");
        record(
            &mut writer,
            &json!({
                "kind":"projection_setup", "setup_id":setup_id,
                "rows":rows, "columns":columns,
                "reduction":reduction, "prefix_ids":prefix, "weight_name":weight_name,
                "checkpoint_revision":REVISION,
                "checkpoint_sha256":DIGEST,
                "weight_absolute_byte_range":[entry.start,entry.end],
                "weight_codec":"BFloat16", "activation_codec":"Float32",
                "requested_output_nodes":[root.0],
                "build_dense_float_storage":omega::sized::DENSE_FLOAT_STORAGE_ENABLED,
                "build_tiled_min_tokens":omega::sized::DENSE_TILED_MIN_TOKENS,
                "config":config, "planning":executor.planning_records,
                "cold_plan_ms":setup_started.elapsed().as_secs_f64()*1000.0,
                "activations":activations, "independent_f64_output":expected,
                "absolute_error_bounds":absolute_bounds,
                "bound_formula":"gamma_n * sum(abs(products)); gamma_n=n*f32::EPSILON/(1-n*f32::EPSILON); BF16 inputs promoted exactly",
                "reference_timing":"outside measured execution",
            }),
        )?;
        let mut durations = Vec::with_capacity(RUN_COUNT);
        for repetition in 0..=RUN_COUNT {
            let metal_before = metal_snapshot();
            let before = ResourceSnapshot::capture();
            let started = Instant::now();
            let evaluated = backend::execute_plan_named(&mut executor.plan, &bindings)
                .map_err(io::Error::other)?;
            let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
            let after = ResourceSnapshot::capture();
            let metal_after = metal_snapshot();
            let (actual, shape) = evaluated
                .get(root)
                .ok_or_else(|| io::Error::other("projection microbench output root absent"))?;
            if actual.len() != expected.len() {
                return Err(io::Error::other(
                    "projection microbench output size changed",
                ));
            }
            let deltas: Vec<f64> = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| f64::from(*actual) - expected)
                .collect();
            let violations = projection_error_coordinates(&expected, actual, &absolute_bounds)?;
            let cpu_seconds = after
                .cpu
                .seconds
                .zip(before.cpu.seconds)
                .map(|(after, before)| after - before);
            record(
                &mut writer,
                &json!({
                    "kind":if repetition==0 {"projection_warmup"} else {"projection_sample"},
                    "rows":rows,"repetition":repetition,"wall_ms":wall_ms,
                    "sample_id":format!("projection-{rows}-{repetition}"),
                    "setup_source":{"file":"microbench.jsonl","setup_id":setup_id},
                    "process_cpu_seconds":cpu_seconds,
                    "cpu_percent":cpu_seconds.filter(|_| wall_ms>0.0).map(|seconds|seconds*100_000.0/wall_ms),
                    "resources_before":before.json(),"resources_after":after.json(),
                    "metal_counters_before":metal_before,"metal_counters_after":metal_after,
                    "actual":actual,"shape":shape,"signed_coordinate_deltas":deltas,
                    "error_bound_violation_coordinates":violations,
                    "timing_scope":"existing plan execute through synchronized device readback; artifact serialization outside timer",
                }),
            )?;
            writer.flush()?;
            if !violations.is_empty() {
                return Err(io::Error::other(
                    "projection output exceeds recorded arithmetic error envelope",
                ));
            }
            if repetition != 0 {
                durations.push(wall_ms);
            }
        }
        record(
            &mut writer,
            &json!({
                "kind":"projection_summary","rows":rows,"cost":stats(&durations,rows),
                "minimum_ms":durations.iter().copied().reduce(f64::min),
                "maximum_ms":durations.iter().copied().reduce(f64::max),
                "sample_count":durations.len(),"error_count":0,
                "sources":(1..=RUN_COUNT).map(|repetition| json!({"file":"microbench.jsonl","sample_id":format!("projection-{rows}-{repetition}")})).collect::<Vec<_>>(),
                "cov_formula":"100 * sqrt(sum((sample-mean)^2)/(n-1)) / mean",
                "gpu_utilization_percent":null,
                "gpu_utilization_reason":"device occupancy is not supplied by this execution surface",
            }),
        )?;
    }
    writer.flush()
}

pub(super) fn run(arguments: &Arguments) -> io::Result<()> {
    let _resident_lock = ResidentLock::acquire()?;
    let profile = if arguments.gpu_profile {
        Some(GpuProfile::install(&arguments.output)?)
    } else {
        None
    };
    if let Some(profile) = &profile {
        setup_marker(profile, "checkpoint", "begin")?;
    }
    let setup_started = Instant::now();
    let panel_path = PathBuf::from(PANEL_PATH);
    let mapping_mode = WeightMappingMode::for_run("metal", false);
    let checkpoint = Checkpoint::open(&arguments.directory, mapping_mode)?;
    let cases = read_cases(&panel_path, &checkpoint)?;
    let checkpoint_setup_ms = setup_started.elapsed().as_secs_f64() * 1000.0;
    if let Some(profile) = &profile {
        setup_marker(profile, "checkpoint", "end")?;
    }
    let mut writer = BufWriter::new(File::create(arguments.output.join("records.jsonl"))?);
    if let Some(config) = &arguments.omega_config {
        let cpu_config = config
            .clone()
            .into_builder()
            .with_engine("cpu")
            .with_gpu_driver("auto")
            .build()
            .map_err(io::Error::other)?;
        for (backend, control_config) in [("cpu", &cpu_config), ("metal", config)] {
            record(
                &mut writer,
                &json!({
                    "kind":"configured_controls", "config":control_config,
                    "model_controls":controls::run_with_config(backend, Some(control_config))?,
                    "paper_controls":controls::paper_with_config(backend, Some(control_config))?,
                }),
            )?;
            writer.flush()?;
        }
    }
    let mut samples = BufWriter::new(File::create(arguments.output.join("samples.jsonl"))?);
    let mut comparisons = BufWriter::new(File::create(arguments.output.join("comparisons.jsonl"))?);
    let mut summaries = Vec::new();
    let mut selections = Vec::new();
    let mut negative_control = None;
    for case in &cases {
        if let Some(profile) = &profile {
            setup_marker(profile, &format!("{}:model", case.identifier), "begin")?;
        }
        let model_setup_started = Instant::now();
        let mut model = Model::prepare_with_config(
            &checkpoint,
            case.prefix.clone(),
            "metal",
            arguments.omega_config.as_ref(),
        )?;
        let model_setup_ms = model_setup_started.elapsed().as_secs_f64() * 1000.0;
        if let Some(profile) = &profile {
            setup_marker(profile, &format!("{}:model", case.identifier), "end")?;
        }
        let mut warmups = Vec::new();
        for (method_index, method) in METHODS.iter().enumerate() {
            let (row, _) = invocation(
                &mut writer,
                &mut model,
                case,
                method_index,
                "warmup",
                (0, method_index),
                profile.as_ref(),
            )?;
            if negative_control.is_none() {
                negative_control = Some((row["sample_id"].clone(), row["payload"].clone()));
            }
            record(&mut samples, &row)?;
            samples.flush()?;
            eprintln!(
                "completed performance warmup {} {}",
                case.identifier, method.0
            );
            warmups.push(row);
        }
        let mut durations: Vec<Vec<f64>> = (0..METHODS.len())
            .map(|_| Vec::with_capacity(RUN_COUNT))
            .collect();
        let mut sample_ids: Vec<Vec<String>> = (0..METHODS.len())
            .map(|_| Vec::with_capacity(RUN_COUNT))
            .collect();
        for repetition in 0..RUN_COUNT {
            for order_position in 0..METHODS.len() {
                let method_index = (repetition + order_position) % METHODS.len();
                let warm_payload = &warmups[method_index]["payload"];
                let (row, wall_ms) = invocation(
                    &mut writer,
                    &mut model,
                    case,
                    method_index,
                    "sample",
                    (repetition, order_position),
                    profile.as_ref(),
                )?;
                let differences = compare_payload(warm_payload, &row["payload"])?;
                let difference_count = differences.len();
                let mut row = row;
                row["warmup_sample_id"] = warmups[method_index]["sample_id"].clone();
                row["payload_difference_count"] = json!(difference_count);
                row["complete_payload_differences"] = json!(differences);
                record(&mut samples, &row)?;
                samples.flush()?;
                record(
                    &mut writer,
                    &json!({"kind":"payload_difference_reconciliation","sample_id":row["sample_id"],"payload_difference_count":difference_count,"complete_payload_differences":differences}),
                )?;
                durations[method_index].push(wall_ms);
                let sample_id = row["sample_id"]
                    .as_str()
                    .ok_or_else(|| io::Error::other("sample ID absent"))?
                    .to_owned();
                sample_ids[method_index].push(sample_id.clone());
                selections.push(Selection {
                    case_id: case.identifier.clone(),
                    repetition,
                    method_index,
                    sample_id,
                    selected_id: row["selected_id"].clone(),
                    selected_bytes: row["selected_bytes"].clone(),
                    selected_spelling: row["selected_spelling"].clone(),
                });
                eprintln!(
                    "completed performance sample {} {} {}",
                    case.identifier,
                    METHODS[method_index].0,
                    repetition + 1
                );
            }
        }
        for (method_index, method) in METHODS.iter().enumerate() {
            summaries.push(json!({
                "case_id": case.identifier,
                "method": method.0,
                "context": model.context(arm(method_index)),
                "raw_durations_ms": durations[method_index],
                "sample_ids": sample_ids[method_index],
                "sources": sample_ids[method_index].iter().map(|sample_id| json!({"file":"samples.jsonl","sample_id":sample_id})).collect::<Vec<_>>(),
                "cost": stats(&durations[method_index], case.prefix.len()),
                "forward_readouts_per_second": 1000.0 / (durations[method_index].iter().sum::<f64>() / RUN_COUNT as f64),
                "duration_order_statistics_ms": duration_order_statistics(&durations[method_index]),
                "warmup_sample_id": warmups[method_index]["sample_id"],
                "cold_model_setup_ms": model_setup_ms,
            }));
        }
        // each comparison row uses one explicit repetition cohort.
        for repetition in 0..RUN_COUNT {
            let cohort: Vec<&Selection> = selections
                .iter()
                .filter(|row| row.case_id == case.identifier && row.repetition == repetition)
                .collect();
            for (left, left_method) in METHODS.iter().enumerate() {
                for (right, right_method) in METHODS.iter().enumerate().skip(left + 1) {
                    let left_row = cohort
                        .iter()
                        .find(|row| row.method_index == left)
                        .ok_or_else(|| io::Error::other("comparison left sample absent"))?;
                    let right_row = cohort
                        .iter()
                        .find(|row| row.method_index == right)
                        .ok_or_else(|| io::Error::other("comparison right sample absent"))?;
                    record(
                        &mut comparisons,
                        &json!({
                            "case_id":case.identifier,"repetition":repetition,
                            "left_method":left_method.0,"right_method":right_method.0,
                            "left_selected_id":left_row.selected_id,"left_selected_bytes":left_row.selected_bytes,"left_selected_spelling":left_row.selected_spelling,
                            "right_selected_id":right_row.selected_id,"right_selected_bytes":right_row.selected_bytes,"right_selected_spelling":right_row.selected_spelling,
                            "selected_id_changed":left_row.selected_id != right_row.selected_id,
                            "left_source":left_row.sample_id,"right_source":right_row.sample_id,
                        }),
                    )?;
                }
            }
        }
    }
    let (control_sample_id, unchanged_control) =
        negative_control.ok_or_else(|| io::Error::other("warmup payload control source absent"))?;
    let mut changed_control = unchanged_control.clone();
    let initial_logit = changed_control["logits"][0]
        .as_f64()
        .ok_or_else(|| io::Error::other("control first logit absent"))?;
    changed_control["logits"][0] = json!(initial_logit + 1.0);
    let control_changes = compare_payload(&unchanged_control, &changed_control)?;
    let unchanged_changes = compare_payload(&unchanged_control, &unchanged_control)?;
    let mut missing_key_old = unchanged_control.clone();
    let mut missing_key_new = unchanged_control.clone();
    missing_key_old
        .as_object_mut()
        .ok_or_else(|| io::Error::other("control payload is not an object"))?
        .remove("logits");
    missing_key_new
        .as_object_mut()
        .ok_or_else(|| io::Error::other("control payload is not an object"))?
        .remove("logits");
    let missing_key_error =
        expected_payload_rejection(&missing_key_old, &missing_key_new, "symmetric missing key")?;
    let mut short_payload = unchanged_control.clone();
    short_payload["logits"]
        .as_array_mut()
        .ok_or_else(|| io::Error::other("control logits are not an array"))?
        .remove(0);
    let length_mismatch_error = expected_payload_rejection(
        &unchanged_control,
        &short_payload,
        "coordinate length mismatch",
    )?;
    record(
        &mut writer,
        &json!({
            "kind":"payload_coordinate_control",
            "source_warmup_sample_id":control_sample_id,
            "unchanged_payload":unchanged_control,
            "changed_payload":changed_control,
            "unchanged_difference_count":unchanged_changes.len(),
            "changed_difference_count":control_changes.len(),
            "changed_coordinate":control_changes,
            "missing_key_old_payload":missing_key_old,
            "missing_key_new_payload":missing_key_new,
            "missing_key_error":missing_key_error,
            "length_mismatch_old_payload":unchanged_control,
            "length_mismatch_new_payload":short_payload,
            "length_mismatch_error":length_mismatch_error,
        }),
    )?;
    record(
        &mut writer,
        &json!({"kind":"performance_setup","checkpoint_setup_ms":checkpoint_setup_ms,"panel_path":panel_path,"panel_cases":cases.iter().map(|case| json!({"case_id":case.identifier,"prefix_ids":case.prefix})).collect::<Vec<_>>(),"total_setup_before_cases_ms":checkpoint_setup_ms,"timing_scope":"setup and cold warmups are recorded outside timed sample durations"}),
    )?;
    if !unchanged_changes.is_empty() || control_changes.len() != 1 {
        return Err(io::Error::other(
            "payload coordinate comparator control failed",
        ));
    }
    serde_json::to_writer_pretty(
        File::create(arguments.output.join("summary.json"))?,
        &summaries,
    )
    .map_err(io::Error::other)?;
    writer.flush()?;
    samples.flush()?;
    comparisons.flush()?;
    if let Some(profile) = &profile {
        let _ = profile.drain()?;
    }
    Ok(())
}

fn setup_marker(profile: &GpuProfile, setup_group: &str, phase: &str) -> io::Result<()> {
    let _ = profile.drain()?;
    profile.marker(
        "ouro_gpu_profile_setup",
        json!({"setup_group":setup_group,"phase":phase}),
    )?;
    Ok(())
}

fn duration_order_statistics(durations: &[f64]) -> Value {
    let mut ordered = durations.to_vec();
    ordered.sort_by(f64::total_cmp);
    let nearest_rank = |fraction: f64| {
        ordered[((fraction * ordered.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(ordered.len() - 1)]
    };
    json!({
        "min_ms": ordered[0],
        "p90_ms": nearest_rank(0.90),
        "p99_ms": nearest_rank(0.99),
        "p99_9_ms": nearest_rank(0.999),
        "max_ms": ordered[ordered.len() - 1],
        "nearest_rank_definition": "ceil(p*n), one-based rank",
    })
}
