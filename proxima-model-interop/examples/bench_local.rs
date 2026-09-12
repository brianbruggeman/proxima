//! Measure one local checkpoint through the public streaming API.
//!
//! Usage:
//! `cargo run -p proxima-model-interop --example bench_local --features std -- \
//!   /path/to/model.gguf cpu "prompt" 16 [batch_size] [ubatch_size]`
//!
//! Use `--features std,vulkan` and `vulkan` for the wgpu/Vulkan route, or
//! `--features std,cuda` and `cuda` for CUDA. The
//! callback timestamps are wall-clock observations from the same client call:
//! TTFT is the prefill event boundary and TTNT is the mean interval between
//! generated-token events.  No derived number is presented as device time.
//!
//! Set `PROXIMA_VERIFY_GPU=1` to run the same prompt through CPU after the
//! measured GPU call and fail if greedy token IDs differ. This is an explicit
//! correctness gate, not part of the reported GPU timing.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::GgmlType;
use proxima_gguf::parse_complete;
use proxima_model_interop::{
    Control, GPU_LAYERS_ALL, LoadedModel, Phase, ServingConfig, classify_task,
};

struct TtntJson(Option<f64>);

impl fmt::Display for TtntJson {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(value) => write!(formatter, "{value:.3}"),
            None => formatter.write_str("null"),
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: bench_local <model.gguf> <cpu|cuda|vulkan> <prompt> [max_tokens] \
         [batch_size] [ubatch_size] [--gpu-memory-bytes N]\n\
         cuda requires --features std,cuda; vulkan requires --features std,vulkan"
    );
    std::process::exit(2)
}

fn rss_kib() -> Option<u64> {
    let mut status = String::new();
    File::open("/proc/self/status")
        .ok()?
        .read_to_string(&mut status)
        .ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmRSS:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

/// Process CPU time sampled from `/proc`, in seconds. This is intentionally
/// process CPU rather than host CPU: it lets one compare serialized and
/// chunked runs as compute-bound or wait-bound without a second sampler.
#[cfg(target_os = "linux")]
fn process_cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let fields = stat
        .rsplit_once(')')?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    let user_ticks: f64 = fields.get(11)?.parse::<u64>().ok()? as f64;
    let system_ticks: f64 = fields.get(12)?.parse::<u64>().ok()? as f64;
    // SAFETY: `_SC_CLK_TCK` is a constant query with no pointer arguments.
    let ticks_per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    (ticks_per_second > 0).then_some((user_ticks + system_ticks) / ticks_per_second as f64)
}

#[cfg(not(target_os = "linux"))]
fn process_cpu_seconds() -> Option<f64> {
    None
}

#[cfg(feature = "cuda")]
fn cuda_memory_kib() -> Option<(u64, u64)> {
    let driver = omega::CudaDriver::new(0).ok()?;
    let (free, total) = driver.memory_info().ok()?;
    Some(((free / 1024) as u64, (total / 1024) as u64))
}

#[cfg(not(feature = "cuda"))]
fn cuda_memory_kib() -> Option<(u64, u64)> {
    None
}

fn memory_json(snapshot: Option<(u64, u64)>) -> String {
    snapshot.map_or_else(
        || "null".to_owned(),
        |(free_kib, total_kib)| format!("{{\"free_kib\":{free_kib},\"total_kib\":{total_kib}}}"),
    )
}

#[cfg(feature = "vulkan")]
fn require_vulkan_adapter() {
    if let Err(error) = omega::probe_wgpu() {
        eprintln!("bench_local: vulkan adapter unavailable: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "vulkan"))]
fn require_vulkan_adapter() {}

#[cfg(feature = "cuda")]
fn require_cuda_device() {
    let driver = match omega::CudaDriver::new(0) {
        Ok(driver) => driver,
        Err(error) => {
            eprintln!("bench_local: cuda device unavailable: {error}");
            std::process::exit(1);
        }
    };
    if let Err(error) = driver.memory_info() {
        eprintln!("bench_local: cuda memory query unavailable: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
fn require_cuda_device() {}

fn main() {
    let mut args = env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| usage());
    let backend = args.next().unwrap_or_else(|| usage());
    let prompt = args.next().unwrap_or_else(|| usage());
    let max_tokens = args
        .next()
        .map(|value| value.parse::<usize>().unwrap_or_else(|_| usage()))
        .unwrap_or(16);
    let batch_size = args
        .next()
        .map(|value| value.parse::<u32>().unwrap_or_else(|_| usage()));
    let ubatch_size = args
        .next()
        .map(|value| value.parse::<u32>().unwrap_or_else(|_| usage()));
    let gpu_memory_limit_bytes = match args.next() {
        None => None,
        Some(flag) if flag == "--gpu-memory-bytes" => Some(
            args.next()
                .unwrap_or_else(|| usage())
                .parse::<u64>()
                .unwrap_or_else(|_| usage()),
        ),
        Some(_) => usage(),
    };
    if args.next().is_some() || !matches!(backend.as_str(), "cpu" | "cuda" | "vulkan") {
        usage();
    }
    if backend == "cuda" && !cfg!(feature = "cuda") {
        eprintln!("bench_local: cuda mode requires --features cuda");
        std::process::exit(2);
    }
    if backend == "vulkan" && !cfg!(feature = "vulkan") {
        eprintln!("bench_local: vulkan mode requires --features vulkan");
        std::process::exit(2);
    }
    if backend != "cpu" && cfg!(all(feature = "cuda", feature = "vulkan")) {
        eprintln!(
            "bench_local: build exactly one of cuda or vulkan; both features select wgpu by default"
        );
        std::process::exit(2);
    }
    if backend == "vulkan" {
        require_vulkan_adapter();
    }
    if backend == "cuda" {
        require_cuda_device();
    }

    let file = File::open(&model_path).expect("open GGUF checkpoint");
    // SAFETY: `file` remains alive for the lifetime of `file_bytes`, and the
    // mapping is read-only; LoadedModel borrows this view for the generation
    // call, avoiding a second allocation the size of the checkpoint.
    let file_bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map GGUF checkpoint");
    let rss_before_load_kib = rss_kib();
    let parsed = parse_complete(&file_bytes).expect("parse GGUF checkpoint");
    let task_profile = classify_task(&parsed);
    eprintln!(
        "bench_local: task={} generation_supported={} architecture={:?} evidence={:?}",
        task_profile.task.name(),
        task_profile.generation_supported,
        task_profile.architecture,
        task_profile.evidence,
    );
    if !task_profile.generation_supported {
        eprintln!(
            "bench_local: refusing decoder-only generation for task {}; an encoder/task-specific graph is required",
            task_profile.task.name()
        );
        std::process::exit(3);
    }
    let loaded_model = LoadedModel::load(&parsed, &file_bytes).expect("bind GGUF checkpoint");
    let rss_after_load_kib = rss_kib();
    let gpu_memory_before = if backend == "cuda" {
        cuda_memory_kib()
    } else {
        None
    };
    let serving_config = ServingConfig {
        model_path: &model_path,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        reasoning_budget: 0,
        batch_size: batch_size.unwrap_or_else(|| ServingConfig::default().batch_size),
        ubatch_size: ubatch_size.unwrap_or_else(|| ServingConfig::default().ubatch_size),
        gpu_layers: if backend != "cpu" { GPU_LAYERS_ALL } else { 0 },
        gpu_memory_limit_bytes,
        gpu_correctness_fallback: env::var_os("PROXIMA_GPU_CORRECTNESS_FALLBACK").is_some(),
        ..ServingConfig::default()
    };

    let started = Instant::now();
    let cpu_before = process_cpu_seconds();
    let mut ttft_ms = None;
    let mut token_times_ms = Vec::with_capacity(max_tokens);
    let mut events = 0_usize;
    let result =
        loaded_model.generate_streaming(&prompt, max_tokens, serving_config, &mut |event| {
            events += 1;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            match event.phase {
                Phase::Prefill { prompt_tokens } => {
                    ttft_ms = Some((prompt_tokens, elapsed_ms));
                }
                Phase::Token => token_times_ms.push(elapsed_ms),
            }
            Control::Continue
        });
    let (ids, text, stopped_by_eos) = result.expect("local generation");
    let verification = if backend != "cpu" && env::var_os("PROXIMA_VERIFY_GPU").is_some() {
        let cpu_config = ServingConfig {
            gpu_layers: 0,
            ..serving_config
        };
        let (cpu_ids, _cpu_text, _cpu_stopped_by_eos) = loaded_model
            .generate_streaming(&prompt, max_tokens, cpu_config, &mut |_| Control::Continue)
            .expect("CPU verification generation");
        let matches = ids == cpu_ids;
        eprintln!(
            "bench_local: gpu_cpu_verification match={matches} gpu_ids={ids:?} cpu_ids={cpu_ids:?}"
        );
        Some((matches, cpu_ids))
    } else {
        None
    };
    let gpu_memory_after = if backend == "cuda" {
        cuda_memory_kib()
    } else {
        None
    };

    // A one-token run has no inter-token interval. Do not report a fake
    // zero-cost decode as a measured TTNT value.
    let ttnt_ms = if token_times_ms.len() >= 2 {
        Some(
            token_times_ms
                .windows(2)
                .map(|pair| pair[1] - pair[0])
                .sum::<f64>()
                / (token_times_ms.len() - 1) as f64,
        )
    } else {
        None
    };
    let total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let ttnt_ms = TtntJson(ttnt_ms);
    let cpu_percent = match (cpu_before, process_cpu_seconds()) {
        (Some(before), Some(after)) if total_ms > 0.0 => {
            Some((after - before) * 100_000.0 / total_ms)
        }
        _ => None,
    };
    let cpu_percent = cpu_percent.map_or_else(|| "null".to_owned(), |value| format!("{value:.2}"));
    println!(
        "{{\"model\":{model_path:?},\"model_bytes\":{},\"backend\":{backend:?},\
         \"rss_before_load_kib\":{},\"rss_after_load_kib\":{},\"prompt_tokens\":{},\
         \"gpu_memory_before\":{},\"gpu_memory_after\":{},\
         \"batch_size\":{},\"ubatch_size\":{},\"cpu_percent\":{},\
         \"generated_tokens\":{},\"ttft_ms\":{},\"ttnt_ms\":{ttnt_ms:.3},\
         \"total_ms\":{total_ms:.3},\"events\":{events},\"stopped_by_eos\":{},\
         \"ids\":{ids:?},\"text\":{text:?},\"verification\":{}}}",
        file_bytes.len(),
        rss_before_load_kib.unwrap_or(0),
        rss_after_load_kib.unwrap_or(0),
        ttft_ms.map_or(0, |(tokens, _)| tokens),
        memory_json(gpu_memory_before),
        memory_json(gpu_memory_after),
        serving_config.batch_size,
        serving_config.ubatch_size,
        cpu_percent,
        ids.len(),
        ttft_ms.map_or(0.0, |(_, elapsed)| elapsed),
        stopped_by_eos,
        verification.as_ref().map_or_else(
            || "null".to_owned(),
            |(matches, cpu_ids)| format!("{{\"matches\":{matches},\"cpu_ids\":{cpu_ids:?}}}"),
        ),
    );
    if verification.is_some_and(|(matches, _)| !matches) {
        std::process::exit(4);
    }
}
