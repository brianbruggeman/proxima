//! HOBBIT expert-precision driver: gate-magnitude high/low selection on a
//! real MoE checkpoint.
//!
//! ```sh
//! hobbit_experts prepare --config moe.toml --model checkpoint.gguf --sidecar sidecar.pxex
//! hobbit_experts run     --config moe.toml --model checkpoint.gguf --sidecar sidecar.pxex \
//!                        [--pairs 3] [--max-tokens 8] [--llama-bench /opt/homebrew/bin/llama-bench]
//! ```
//!
//! `prepare` writes a low-codec copy of every `blk.<n>.ffn_{gate,up,down}_exps.weight`
//! stack with `write_expert_sidecar`; the stack list comes from the checkpoint
//! directory, never a family table. `run` measures two arms interleaved:
//! the checkpoint's high bytes on both (high-only) and the same model with
//! the sidecar attached, the configured residency budget and the gate-magnitude
//! threshold (HOBBIT). The HOBBIT arm prints how many routed experts were
//! served high and low, the quality delta against the high-only arm, and the
//! same five metrics on both sides. The decode configuration is the TOML's
//! `ServingSettings`, and `--llama-bench` mirrors the same prompt-token and
//! generation counts.
//!
//! The policy is the paper's first mechanism: selected experts are ordered by
//! the router gate, the cumulative normalized gate mass `s_i` decides high
//! (`s_i <= T1` or resident) versus low. The paper's skip tier and LHU cache
//! are named residuals, not silently implied.
//!
//! The per-token arm needs a model whose pre-gather plan binds. qwen35moe is
//! the only checkpoint in this host's zoo that exposes routed boundaries, and
//! its pre-gather plan currently fails at metal node `%0` (CPU node `3`), so
//! `run` prints `per_token_arm=blocked` and exits nonzero instead of measuring
//! a different arm. `prepare` is independent and works on any MoE checkpoint.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs::File;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use memmap2::Mmap;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::expert_sidecar::write_expert_sidecar;
use proxima_model_interop::hobbit::{
    calibration_ids, check_quality_bound, quality_delta,
};
use proxima_model_interop::{
    ExpertStackSpec, LoadedModel, ServingConfig, ServingSettings, TokenEvent,
};
use proxima_numeric::NumericFormat;

const USAGE: &str = "usage: hobbit_experts <prepare|run> --config PATH --model PATH --sidecar PATH \
[--pairs N] [--max-tokens N] [--llama-bench PATH]";

type DriverResult<T> = Result<T, String>;

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("hobbit_experts: {message}");
            ExitCode::FAILURE
        }
    }
}

struct Flags {
    command: String,
    config: PathBuf,
    model: PathBuf,
    sidecar: PathBuf,
    pairs: usize,
    max_tokens: usize,
    llama_bench: Option<PathBuf>,
}

fn parse_flags(arguments: Vec<String>) -> DriverResult<Flags> {
    let mut arguments = arguments.into_iter();
    let command = arguments.next().ok_or_else(|| USAGE.to_owned())?;
    let mut config = None;
    let mut model = None;
    let mut sidecar = None;
    let mut pairs = 3usize;
    let mut max_tokens = 8usize;
    let mut llama_bench = None;
    while let Some(flag) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--config" => config = Some(PathBuf::from(value()?)),
            "--model" => model = Some(PathBuf::from(value()?)),
            "--sidecar" => sidecar = Some(PathBuf::from(value()?)),
            "--pairs" => {
                pairs = value()?
                    .parse::<usize>()
                    .map_err(|error| format!("--pairs: {error}"))?;
            }
            "--max-tokens" => {
                max_tokens = value()?
                    .parse::<usize>()
                    .map_err(|error| format!("--max-tokens: {error}"))?;
            }
            "--llama-bench" => llama_bench = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown flag {other}; {USAGE}")),
        }
    }
    if pairs < 3 {
        return Err("--pairs must be >= 3 for an interleaved measurement".to_owned());
    }
    if max_tokens == 0 {
        return Err("--max-tokens must be nonzero".to_owned());
    }
    Ok(Flags {
        command,
        config: config.ok_or("--config is required")?,
        model: model.ok_or("--model is required")?,
        sidecar: sidecar.ok_or("--sidecar is required")?,
        pairs,
        max_tokens,
        llama_bench,
    })
}

fn run(arguments: Vec<String>) -> DriverResult<()> {
    let flags = parse_flags(arguments)?;
    let settings: ServingSettings = conflaguration::from_file(&flags.config)
        .map_err(|error| format!("{}: {error}", flags.config.display()))?;
    let file = File::open(&flags.model)
        .map_err(|error| format!("{}: {error}", flags.model.display()))?;
    let map = unsafe { Mmap::map(&file) }
        .map_err(|error| format!("{}: {error}", flags.model.display()))?;
    let bytes: &[u8] = &map;
    let parsed = proxima_gguf::pipe::parse_complete(bytes)
        .map_err(|error| format!("{}: {error}", flags.model.display()))?;
    let base = base_config(&settings);

    match flags.command.as_str() {
        "prepare" => prepare(&parsed, bytes, &settings, &flags.sidecar),
        "run" => run_arms(&parsed, bytes, &settings, &base, &flags),
        other => Err(format!("unknown command {other}; {USAGE}")),
    }
}

fn base_config(settings: &ServingSettings) -> ServingConfig<'_> {
    ServingConfig {
        moe_pre_gather: true,
        ..settings.as_serving_config(&[])
    }
}

fn expert_stack_specs<'a>(
    parsed: &'a ParsedGguf,
    codec: NumericFormat,
) -> Vec<ExpertStackSpec<'a>> {
    parsed
        .tensors
        .iter()
        .filter_map(|tensor| {
            let rest = tensor.name.strip_prefix("blk.")?;
            let (layer, rest) = rest.split_once('.')?;
            let (projection, _) = rest.split_once("_exps.weight")?;
            if !projection.starts_with("ffn_") || tensor.dims.len() != 3 {
                return None;
            }
            Some(ExpertStackSpec {
                layer: layer.parse().ok()?,
                projection,
                tensor_name: tensor.name.as_str(),
                expert_count: u32::try_from(tensor.dims[2]).ok()?,
                out_dim: u32::try_from(tensor.dims[1]).ok()?,
                in_dim: u32::try_from(tensor.dims[0]).ok()?,
                target_codec: codec,
            })
        })
        .collect()
}

fn codec_bytes(codec: NumericFormat, elements: usize) -> usize {
    let layout = GgmlType::from_numeric_format(&codec)
        .expect("the configured low codec has a ggml wire type")
        .wire_shape();
    let block_elements = layout.block_elements as usize;
    let block_bytes = layout.block_bytes as usize;
    elements.div_ceil(block_elements) * block_bytes
}

fn prepare(
    parsed: &ParsedGguf,
    bytes: &[u8],
    settings: &ServingSettings,
    sidecar_path: &Path,
) -> DriverResult<()> {
    let codec = settings.moe_expert_precision.low_codec;
    if GgmlType::from_numeric_format(&codec).is_none() {
        return Err(format!(
            "low_codec {} has no ggml wire type",
            codec.name().unwrap_or("unknown")
        ));
    }
    let stacks = expert_stack_specs(parsed, codec);
    if stacks.is_empty() {
        return Err("the checkpoint has no blk.<n>.ffn_*_exps.weight stacks".to_owned());
    }
    let max_elements = stacks
        .iter()
        .map(|stack| stack.out_dim as usize * stack.in_dim as usize)
        .max()
        .unwrap_or(0);
    let max_bytes = codec_bytes(codec, max_elements);
    let mut scratch = vec![0.0f32; max_elements];
    let mut output = vec![0u8; max_bytes];
    let mut file = File::create(sidecar_path)
        .map_err(|error| format!("{}: {error}", sidecar_path.display()))?;
    let sidecar = write_expert_sidecar(
        parsed,
        bytes,
        &stacks,
        codec,
        &mut scratch,
        &mut output,
        &mut file,
    )
    .map_err(|error| error.to_string())?;
    file.sync_all()
        .map_err(|error| format!("{}: {error}", sidecar_path.display()))?;
    println!(
        "sidecar_bytes={} descriptors={} codec={} stacks={}",
        sidecar.total_bytes,
        sidecar.descriptors.len(),
        codec.name().unwrap_or("unknown"),
        stacks.len()
    );
    Ok(())
}

fn run_arms(
    parsed: &ParsedGguf,
    bytes: &[u8],
    settings: &ServingSettings,
    base: &ServingConfig<'_>,
    flags: &Flags,
) -> DriverResult<()> {
    let threshold = settings.moe_expert_precision.checked_threshold()?;
    let Some(threshold) = threshold else {
        return Err("moe_expert_precision.enabled is false; no HOBBIT arm to run".to_owned());
    };
    if settings.moe_residency_budget_bytes == 0 {
        return Err("moe_residency_budget_bytes is 0; the policy needs a high-precision pool".to_owned());
    }
    let sidecar_bytes = std::fs::metadata(&flags.sidecar)
        .map_err(|error| format!("{}: {error}", flags.sidecar.display()))?
        .len();
    let prompt_ids = calibration_ids(
        parsed,
        &settings.hobbit.calibration_prompts,
        settings.hobbit.max_calibration_tokens,
    )
    .map_err(|error| error.to_string())?;
    let high_config = *base;
    let hobbit_config = ServingConfig {
        moe_residency_budget_bytes: settings.moe_residency_budget_bytes,
        moe_high_precision_threshold: Some(threshold),
        ..*base
    };
    {
        let mut probe = LoadedModel::load_with_serving(parsed, bytes, &hobbit_config)
            .map_err(|error| error.to_string())?;
        let sidecar = File::open(&flags.sidecar)
            .map_err(|error| format!("{}: {error}", flags.sidecar.display()))?;
        probe
            .attach_expert_sidecar_file(sidecar)
            .map_err(|error| error.to_string())?;
        let mut on_token = |_event: TokenEvent<'_>| ControlFlow::Continue(());
        if let Err(error) =
            probe.generate_from_ids(&prompt_ids[0], 1, &hobbit_config, &mut on_token)
        {
            println!("per_token_arm=blocked error={error}");
            return Err(format!("per-token high/low arm blocked: {error}"));
        }
        println!("per_token_arm=ready");
    }

    quality_gate(
        parsed,
        bytes,
        &flags.sidecar,
        &high_config,
        &hobbit_config,
        &prompt_ids,
        settings,
    )?;

    let mut runs: BTreeMap<&'static str, Vec<ArmRun>> = BTreeMap::new();
    for pair in 0..flags.pairs {
        let arms: [(&'static str, &ServingConfig<'_>); 2] = if pair % 2 == 0 {
            [("high_only", &high_config), ("hobbit", &hobbit_config)]
        } else {
            [("hobbit", &hobbit_config), ("high_only", &high_config)]
        };
        for (name, config) in arms {
            let mut model = LoadedModel::load_with_serving(parsed, bytes, config)
                .map_err(|error| error.to_string())?;
            if name == "hobbit" {
                let sidecar = File::open(&flags.sidecar)
                    .map_err(|error| format!("{}: {error}", flags.sidecar.display()))?;
                model
                    .attach_expert_sidecar_file(sidecar)
                    .map_err(|error| error.to_string())?;
            }
            let run = bench_arm(&model, config, &prompt_ids, flags.max_tokens)?;
            let (high, low) = model.moe_precision_counts();
            println!(
                "pair={pair} arm={name} moe_high_decisions={high} moe_low_decisions={low}"
            );
            print_run(pair, name, &run);
            runs.entry(name).or_default().push(run);
        }
    }
    for (name, samples) in &runs {
        print_summary(name, samples);
    }
    println!(
        "pairing pairs={} high_only_runs={} hobbit_runs={}",
        flags.pairs,
        runs.get("high_only").map_or(0, Vec::len),
        runs.get("hobbit").map_or(0, Vec::len)
    );
    println!(
        "hobbit_config threshold={threshold} budget={} sidecar_bytes={sidecar_bytes}",
        settings.moe_residency_budget_bytes
    );
    if let Some(llama_bench) = &flags.llama_bench {
        run_llama_bench(
            llama_bench,
            &flags.model,
            &prompt_ids,
            flags.max_tokens,
            flags.pairs,
        )?;
    }
    Ok(())
}

fn quality_gate(
    parsed: &ParsedGguf,
    bytes: &[u8],
    sidecar_path: &Path,
    high_config: &ServingConfig<'_>,
    hobbit_config: &ServingConfig<'_>,
    prompt_ids: &[Vec<u32>],
    settings: &ServingSettings,
) -> DriverResult<()> {
    let bound = settings
        .moe_expert_precision
        .quality_bound()
        .ok_or("moe_expert_precision.quality sets no bound; no bound, no apply")?;
    let reference = teacher_forced(parsed, bytes, high_config, prompt_ids)?;
    let mut model = LoadedModel::load_with_serving(parsed, bytes, hobbit_config)
        .map_err(|error| error.to_string())?;
    let sidecar = File::open(sidecar_path)
        .map_err(|error| format!("{}: {error}", sidecar_path.display()))?;
    model
        .attach_expert_sidecar_file(sidecar)
        .map_err(|error| error.to_string())?;
    let candidate = teacher_forced_model(&model, hobbit_config, prompt_ids)?;
    let delta = quality_delta(&reference, &candidate).map_err(|error| error.to_string())?;
    check_quality_bound(&delta, Some(bound)).map_err(|error| error.to_string())?;
    println!(
        "quality positions={} top1_agreement={:.6} mean_kl={:.9} bound_max_mean_kl={:?} bound_min_top1={:?} holds=true",
        delta.positions, delta.top1_agreement, delta.mean_kl, bound.0, bound.1
    );
    Ok(())
}

fn teacher_forced(
    parsed: &ParsedGguf,
    bytes: &[u8],
    config: &ServingConfig<'_>,
    prompt_ids: &[Vec<u32>],
) -> DriverResult<Vec<Vec<f32>>> {
    let model =
        LoadedModel::load_with_serving(parsed, bytes, config).map_err(|error| error.to_string())?;
    teacher_forced_model(&model, config, prompt_ids)
}

fn teacher_forced_model(
    model: &LoadedModel<'_>,
    config: &ServingConfig<'_>,
    prompt_ids: &[Vec<u32>],
) -> DriverResult<Vec<Vec<f32>>> {
    let mut rows = Vec::new();
    for ids in prompt_ids {
        if ids.len() < 2 {
            continue;
        }
        rows.extend(
            model
                .teacher_forced_logits_from_ids(&ids[..1], &ids[1..], config)
                .map_err(|error| error.to_string())?,
        );
    }
    Ok(rows)
}

#[derive(Clone, Copy, Debug, Default)]
struct ArmRun {
    ttft_cold_ns: u64,
    ttft_warm_ns: u64,
    tokens: usize,
    tok_s: f64,
    peak_rss_bytes: u64,
    steady_rss_bytes: u64,
    peak_footprint_bytes: u64,
    steady_footprint_bytes: u64,
    text_hash: u64,
}

fn bench_arm(
    model: &LoadedModel<'_>,
    config: &ServingConfig<'_>,
    prompt_ids: &[Vec<u32>],
    max_tokens: usize,
) -> DriverResult<ArmRun> {
    let mut run = ArmRun {
        steady_footprint_bytes: current_footprint_bytes(),
        steady_rss_bytes: current_rss_bytes(),
        ..ArmRun::default()
    };
    let cold_prompt = prompt_ids.first().ok_or("the calibration set is empty")?;
    let warm_prompt = prompt_ids.get(1).unwrap_or(cold_prompt);
    let (cold_ns, _, cold_tokens, cold_hash) =
        generate_measured(model, config, cold_prompt, max_tokens, &mut run)?;
    let (warm_ns, warm_span_ns, warm_tokens, warm_hash) =
        generate_measured(model, config, warm_prompt, max_tokens, &mut run)?;
    run.ttft_cold_ns = cold_ns;
    run.ttft_warm_ns = warm_ns;
    run.tokens = cold_tokens + warm_tokens;
    run.tok_s = if warm_tokens > 1 && warm_span_ns > 0 {
        (warm_tokens.saturating_sub(1)) as f64 * 1e9 / warm_span_ns as f64
    } else {
        0.0
    };
    run.text_hash = cold_hash ^ warm_hash.rotate_left(17);
    run.steady_rss_bytes = current_rss_bytes();
    run.steady_footprint_bytes = current_footprint_bytes();
    Ok(run)
}

fn generate_measured(
    model: &LoadedModel<'_>,
    config: &ServingConfig<'_>,
    prompt: &[u32],
    max_tokens: usize,
    run: &mut ArmRun,
) -> DriverResult<(u64, u64, usize, u64)> {
    let mut first = None;
    let mut last = 0u64;
    let mut tokens = 0usize;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut on_token = |event: TokenEvent<'_>| {
        if first.is_none() {
            first = Some(event.elapsed_ns);
        }
        last = event.elapsed_ns;
        tokens += 1;
        hash ^= u64::from(event.token_id);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        run.peak_rss_bytes = run.peak_rss_bytes.max(peak_rss_bytes());
        run.peak_footprint_bytes = run
            .peak_footprint_bytes
            .max(current_footprint_bytes());
        ControlFlow::Continue(())
    };
    let _ = model
        .generate_from_ids(prompt, max_tokens, config, &mut on_token)
        .map_err(|error| error.to_string())?;
    let first = first.unwrap_or(last);
    Ok((first, last.saturating_sub(first), tokens, hash))
}

fn print_run(pair: usize, arm: &str, run: &ArmRun) {
    println!(
        "pair={pair} arm={arm} ttft_cold_ms={:.3} ttft_warm_ms={:.3} tok_s={:.3} peak_rss_bytes={} steady_rss_bytes={} peak_footprint_bytes={} steady_footprint_bytes={} tokens={} text_hash={:016x}",
        run.ttft_cold_ns as f64 / 1e6,
        run.ttft_warm_ns as f64 / 1e6,
        run.tok_s,
        run.peak_rss_bytes,
        run.steady_rss_bytes,
        run.peak_footprint_bytes,
        run.steady_footprint_bytes,
        run.tokens,
        run.text_hash
    );
}

type ArmMetric = (&'static str, fn(&ArmRun) -> f64);

fn print_summary(arm: &str, samples: &[ArmRun]) {
    let metrics: [ArmMetric; 7] = [
        ("ttft_cold_ms", |run| run.ttft_cold_ns as f64 / 1e6),
        ("ttft_warm_ms", |run| run.ttft_warm_ns as f64 / 1e6),
        ("tok_s", |run| run.tok_s),
        ("peak_rss_bytes", |run| run.peak_rss_bytes as f64),
        ("steady_rss_bytes", |run| run.steady_rss_bytes as f64),
        ("peak_footprint_bytes", |run| run.peak_footprint_bytes as f64),
        ("steady_footprint_bytes", |run| {
            run.steady_footprint_bytes as f64
        }),
    ];
    for (name, read) in metrics {
        let values = samples.iter().map(read).collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / values.len() as f64;
        let cov = if mean == 0.0 {
            0.0
        } else {
            variance.sqrt() / mean * 100.0
        };
        println!(
            "summary arm={arm} metric={name} mean={mean:.6} cov_percent={cov:.3} n={}",
            values.len()
        );
    }
}

fn run_llama_bench(
    binary: &Path,
    model: &Path,
    prompt_ids: &[Vec<u32>],
    max_tokens: usize,
    pairs: usize,
) -> DriverResult<()> {
    let prompt_tokens = prompt_ids.first().map_or(1, Vec::len).to_string();
    let output = Command::new(binary)
        .arg("-m")
        .arg(model)
        .arg("-p")
        .arg(&prompt_tokens)
        .arg("-n")
        .arg(max_tokens.to_string())
        .arg("-r")
        .arg(pairs.to_string())
        .arg("-o")
        .arg("json")
        .output()
        .map_err(|error| format!("{}: {error}", binary.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} exited {}: {}",
            binary.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    println!(
        "llama_bench pairs={pairs} mirror_prompt_tokens={prompt_tokens} mirror_max_tokens={max_tokens}"
    );
    println!("llama_bench_json={}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

#[cfg(target_os = "macos")]
fn task_vm_info() -> Option<(u64, u64, u64)> {
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
    const TASK_VM_INFO: i32 = 22;
    unsafe extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: i32,
            task_info_out: *mut TaskVmInfo,
            task_info_out_count: *mut u32,
        ) -> i32;
    }
    let mut info = TaskVmInfo::default();
    let mut count = (core::mem::size_of::<TaskVmInfo>() / core::mem::size_of::<u32>()) as u32;
    // SAFETY: TASK_VM_INFO_REV1's struct and a matching word count.
    let result = unsafe { task_info(mach_task_self(), TASK_VM_INFO, &mut info, &mut count) };
    (result == 0).then_some((
        info.resident_size,
        info.resident_size_peak,
        info.phys_footprint,
    ))
}

#[cfg(target_os = "macos")]
fn current_rss_bytes() -> u64 {
    task_vm_info().map_or(0, |(resident, _, _)| resident)
}

#[cfg(target_os = "macos")]
fn peak_rss_bytes() -> u64 {
    task_vm_info().map_or(0, |(_, peak, _)| peak)
}

#[cfg(target_os = "macos")]
fn current_footprint_bytes() -> u64 {
    task_vm_info().map_or(0, |(_, _, footprint)| footprint)
}

#[cfg(target_os = "linux")]
fn proc_status(key: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == key).then(|| {
            value
                .split_whitespace()
                .next()
                .and_then(|raw| raw.parse::<u64>().ok())
                .map(|kib| kib * 1024)
        })?
    })
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> u64 {
    proc_status("VmRSS").unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> u64 {
    proc_status("VmHWM").unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn current_footprint_bytes() -> u64 {
    current_rss_bytes()
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn current_rss_bytes() -> u64 {
    0
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn peak_rss_bytes() -> u64 {
    0
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn current_footprint_bytes() -> u64 {
    0
}
