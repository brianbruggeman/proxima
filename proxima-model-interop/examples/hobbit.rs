//! HOBBIT driver: per-layer mixed precision on any GGUF checkpoint.
//!
//! ```sh
//! hobbit sensitivity --config hobbit.toml --model checkpoint.gguf [--json-out ranking.json]
//! hobbit allocate    --config hobbit.toml --model checkpoint.gguf [--ranking ranking.json] [--budget BYTES]
//! hobbit run         --config hobbit.toml --model checkpoint.gguf [--ranking ranking.json] \
//!                    [--pairs 3] [--max-tokens 32] [--llama-bench /opt/homebrew/bin/llama-bench]
//! ```
//!
//! `sensitivity` measures each layer's output delta against the unrecoded
//! run (top-1 agreement and mean KL on the configured calibration set) and
//! prints a deterministic ranking hash. `allocate` spends the configured
//! byte budget on the least sensitive layers and either prints
//! `shortfall=0` or refuses with the shortfall. `run` applies the quality
//! bound, then measures two arms — unrecoded and recoded — interleaved over
//! `--pairs` pairs and prints the five metrics with CoV. The unrecoded arm
//! sits beside every recoded number.
//!
//! The decode configuration is the same one `llama.cpp` uses: the harness
//! reads `ServingSettings` (`-c`, `-b`, `-ub`, greedy `--temp 0`, `--top-k 0`,
//! fixed seed) and prints its own mirrored values; `--llama-bench` adds the
//! incumbent arm with the same prompt-token and generation counts.
//!
//! Nothing here names a model family or a layer: prefixes and tensor names
//! come from the checkpoint directory.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs::File;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use memmap2::Mmap;
use proxima_gguf::pipe::ParsedGguf;
use proxima_model_interop::hobbit::{
    HobbitAllocation, HobbitQualityDelta, HobbitSensitivityReport, allocate, calibration_ids,
    check_quality_bound, layer_prefixes, measure_sensitivity, quality_delta,
    recode_candidate_tensors,
};
use proxima_model_interop::{
    LoadedModel, NamePattern, ServingConfig, ServingSettings, WeightPrecisionRule,
};

const USAGE: &str = "usage: hobbit <sensitivity|allocate|probe|run> --config PATH --model PATH \
[--json-out PATH] [--ranking PATH] [--budget BYTES] [--pairs N] [--max-tokens N] \
[--layer blk.N.] [--llama-bench PATH]";

type DriverResult<T> = Result<T, String>;

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("hobbit: {message}");
            ExitCode::FAILURE
        }
    }
}

struct Flags {
    command: String,
    config: PathBuf,
    model: PathBuf,
    json_out: Option<PathBuf>,
    ranking: Option<PathBuf>,
    budget: Option<u64>,
    pairs: usize,
    max_tokens: usize,
    layer: Option<String>,
    llama_bench: Option<PathBuf>,
}

fn parse_flags(arguments: Vec<String>) -> DriverResult<Flags> {
    let mut arguments = arguments.into_iter();
    let command = arguments.next().ok_or_else(|| USAGE.to_owned())?;
    let mut config = None;
    let mut model = None;
    let mut json_out = None;
    let mut ranking = None;
    let mut budget = None;
    let mut pairs = 3usize;
    let mut max_tokens = 32usize;
    let mut layer = None;
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
            "--json-out" => json_out = Some(PathBuf::from(value()?)),
            "--ranking" => ranking = Some(PathBuf::from(value()?)),
            "--budget" => {
                budget = Some(
                    value()?
                        .parse::<u64>()
                        .map_err(|error| format!("--budget: {error}"))?,
                );
            }
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
            "--layer" => layer = Some(value()?),
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
        json_out,
        ranking,
        budget,
        pairs,
        max_tokens,
        layer,
        llama_bench,
    })
}

fn run(arguments: Vec<String>) -> DriverResult<()> {
    let flags = parse_flags(arguments)?;
    let settings: ServingSettings = conflaguration::from_file(&flags.config)
        .map_err(|error| format!("{}: {error}", flags.config.display()))?;
    let file =
        File::open(&flags.model).map_err(|error| format!("{}: {error}", flags.model.display()))?;
    let map = unsafe { Mmap::map(&file) }
        .map_err(|error| format!("{}: {error}", flags.model.display()))?;
    let bytes: &[u8] = &map;
    let parsed = proxima_gguf::pipe::parse_complete(bytes)
        .map_err(|error| format!("{}: {error}", flags.model.display()))?;
    let base = base_config(&settings);
    let prompt_ids = calibration_ids(
        &parsed,
        &settings.hobbit.calibration_prompts,
        settings.hobbit.max_calibration_tokens,
    )
    .map_err(|error| error.to_string())?;

    match flags.command.as_str() {
        "sensitivity" => {
            let report = sensitivity(&parsed, bytes, &settings, &base)?;
            print_sensitivity(&report);
            if let Some(path) = &flags.json_out {
                let json =
                    serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
                std::fs::write(path, json)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
            }
            Ok(())
        }
        "allocate" => {
            let report = load_or_measure(&flags, &parsed, bytes, &settings, &base)?;
            let budget = flags.budget.or(settings.hobbit.budget_bytes);
            match allocate(&report, budget, settings.hobbit.min_layers) {
                Ok(allocation) => {
                    print_allocation(&allocation);
                    verify_allocation(&parsed, bytes, &base, &allocation, &report)?;
                    Ok(())
                }
                Err(error) => Err(format!("refuse {error}")),
            }
        }
        "probe" => probe(&parsed, bytes, &settings, &base, flags.layer.as_deref()),
        "run" => {
            let report = load_or_measure(&flags, &parsed, bytes, &settings, &base)?;
            let budget = flags.budget.or(settings.hobbit.budget_bytes);
            let allocation = allocate(&report, budget, settings.hobbit.min_layers)
                .map_err(|error| format!("refuse {error}"))?;
            print_allocation(&allocation);
            quality_gate(&parsed, bytes, &base, &prompt_ids, &allocation, &settings)?;
            benchmark(
                &parsed,
                bytes,
                &base,
                &prompt_ids,
                &allocation,
                flags.pairs,
                flags.max_tokens,
                &report,
            )?;
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
        other => Err(format!("unknown command {other}; {USAGE}")),
    }
}

fn probe(
    parsed: &ParsedGguf,
    bytes: &[u8],
    settings: &ServingSettings,
    base: &ServingConfig<'_>,
    requested_layer: Option<&str>,
) -> DriverResult<()> {
    let prefixes = layer_prefixes(parsed);
    let prefix = requested_layer
        .map(str::to_owned)
        .or_else(|| prefixes.first().cloned())
        .ok_or("the checkpoint names no blk.<n>. layer")?;
    if !prefixes.iter().any(|candidate| candidate == &prefix) {
        return Err(format!("{prefix} is not a layer this checkpoint names"));
    }
    let target = settings
        .hobbit
        .candidate_targets()
        .map_err(|error| error.to_string())?
        .first()
        .copied()
        .ok_or("hobbit.candidate_codecs is empty")?;
    let rules = recode_candidate_tensors(parsed, &prefix, target)
        .into_iter()
        .map(|tensor| WeightPrecisionRule {
            pattern: NamePattern::Exact(tensor.name.as_str()),
            target,
        })
        .collect::<Vec<_>>();
    if rules.is_empty() {
        return Err(format!("{prefix} has no recode candidate tensor"));
    }
    let config = base.with_weight_precision(&rules);
    let model = LoadedModel::load_with_serving(parsed, bytes, &config)
        .map_err(|error| error.to_string())?;
    let recoded = model.recoded_weights();
    println!("layer={prefix} rules={} recoded_count={}", rules.len(), recoded.len());
    for (name, bytes) in &recoded {
        println!("bound_name={name} bytes={bytes}");
    }
    println!("program_hash={:016x}", model.program_hash());
    Ok(())
}

fn base_config(settings: &ServingSettings) -> ServingConfig<'_> {
    ServingConfig {
        gpu_layers: 0,
        ..settings.as_serving_config(&[])
    }
}

fn sensitivity(
    parsed: &ParsedGguf,
    bytes: &[u8],
    settings: &ServingSettings,
    base: &ServingConfig<'_>,
) -> DriverResult<HobbitSensitivityReport> {
    settings
        .hobbit
        .candidate_targets()
        .map_err(|error| error.to_string())?;
    let candidate = settings
        .hobbit
        .candidate_codecs
        .first()
        .copied()
        .ok_or("hobbit.candidate_codecs is empty")?;
    if settings.hobbit.candidate_codecs.len() > 1 {
        eprintln!(
            "hobbit: {} candidate codecs configured; measuring the first ({})",
            settings.hobbit.candidate_codecs.len(),
            candidate.name().unwrap_or("unknown")
        );
    }
    measure_sensitivity(
        parsed,
        bytes,
        &settings.hobbit.calibration_prompts,
        candidate,
        settings.hobbit.max_calibration_tokens,
        base,
    )
    .map_err(|error| error.to_string())
}

fn load_or_measure(
    flags: &Flags,
    parsed: &ParsedGguf,
    bytes: &[u8],
    settings: &ServingSettings,
    base: &ServingConfig<'_>,
) -> DriverResult<HobbitSensitivityReport> {
    match &flags.ranking {
        Some(path) => {
            let json = std::fs::read_to_string(path)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            serde_json::from_str(&json).map_err(|error| format!("{}: {error}", path.display()))
        }
        None => sensitivity(parsed, bytes, settings, base),
    }
}

fn print_sensitivity(report: &HobbitSensitivityReport) {
    for layer in &report.layers {
        println!(
            "layer={} mean_kl={:.9} top1_agreement={:.6} positions={} saved_bytes={} recoded_bytes={} tensors={}",
            layer.layer,
            layer.mean_kl,
            layer.top1_agreement,
            layer.positions,
            layer.saved_bytes,
            layer.recoded_bytes,
            layer.tensors.len()
        );
    }
    println!(
        "ranking={}",
        report
            .layers
            .iter()
            .map(|layer| layer.layer.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "ranking_hash={:016x} candidate={} prompts={} positions={} base_bytes={}",
        report.ranking_hash,
        report.candidate_codec,
        report.prompts,
        report.positions,
        report.base_bytes
    );
}

fn print_allocation(allocation: &HobbitAllocation) {
    println!("budget={}", allocation.budget_bytes);
    println!("base_bytes={}", allocation.base_bytes);
    println!("assigned={}", allocation.assigned_bytes);
    println!("recoded_bytes={}", allocation.recoded_bytes);
    println!("saved_bytes={}", allocation.saved_bytes);
    println!("shortfall={}", allocation.shortfall);
    println!("rules={}", allocation.rules.len());
    for rule in &allocation.rules {
        println!("rule={}:{}", rule.tensor, rule.codec);
    }
}

fn verify_allocation(
    parsed: &ParsedGguf,
    bytes: &[u8],
    base: &ServingConfig<'_>,
    allocation: &HobbitAllocation,
    report: &HobbitSensitivityReport,
) -> DriverResult<()> {
    let rules = allocation
        .weight_precision_rules()
        .map_err(|error| error.to_string())?;
    let config = base.with_weight_precision(&rules);
    let model = LoadedModel::load_with_serving(parsed, bytes, &config)
        .map_err(|error| error.to_string())?;
    let bound = model
        .recoded_weights()
        .iter()
        .map(|(_, bytes)| *bytes as u64)
        .sum::<u64>();
    println!("bound_recoded_bytes={bound}");
    if bound != allocation.recoded_bytes {
        return Err(format!(
            "allocator predicted {} recoded bytes, the bind produced {bound}",
            allocation.recoded_bytes
        ));
    }
    if allocation.assigned_bytes > report.base_bytes {
        return Err("assigned bytes exceed the checkpoint's own tensor bytes".to_owned());
    }
    Ok(())
}

fn quality_gate(
    parsed: &ParsedGguf,
    bytes: &[u8],
    base: &ServingConfig<'_>,
    prompt_ids: &[Vec<u32>],
    allocation: &HobbitAllocation,
    settings: &ServingSettings,
) -> DriverResult<()> {
    let bound = settings
        .hobbit
        .quality_bound()
        .ok_or("hobbit.quality sets no bound; no bound, no apply")?;
    let rules = allocation
        .weight_precision_rules()
        .map_err(|error| error.to_string())?;
    let recoded_config = base.with_weight_precision(&rules);
    let reference = teacher_forced(parsed, bytes, base, prompt_ids)?;
    let candidate = teacher_forced(parsed, bytes, &recoded_config, prompt_ids)?;
    let delta = quality_delta(&reference, &candidate).map_err(|error| error.to_string())?;
    check_quality_bound(&delta, Some(bound)).map_err(|error| error.to_string())?;
    print_quality(&delta, bound);
    Ok(())
}

fn print_quality(delta: &HobbitQualityDelta, bound: (Option<f64>, Option<f64>)) {
    println!(
        "quality positions={} top1_agreement={:.6} mean_kl={:.9} bound_max_mean_kl={:?} bound_min_top1={:?} holds=true",
        delta.positions, delta.top1_agreement, delta.mean_kl, bound.0, bound.1
    );
}

fn teacher_forced(
    parsed: &ParsedGguf,
    bytes: &[u8],
    config: &ServingConfig<'_>,
    prompt_ids: &[Vec<u32>],
) -> DriverResult<Vec<Vec<f32>>> {
    let model =
        LoadedModel::load_with_serving(parsed, bytes, config).map_err(|error| error.to_string())?;
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

// the driver's measurement tuple: every input the interleaved arm needs,
// mirroring the fields a scorer reads; grouping them would mint a type for a
// single call site.
#[allow(clippy::too_many_arguments)]
fn benchmark(
    parsed: &ParsedGguf,
    bytes: &[u8],
    base: &ServingConfig<'_>,
    prompt_ids: &[Vec<u32>],
    allocation: &HobbitAllocation,
    pairs: usize,
    max_tokens: usize,
    report: &HobbitSensitivityReport,
) -> DriverResult<()> {
    let rules = allocation
        .weight_precision_rules()
        .map_err(|error| error.to_string())?;
    let recoded = base.with_weight_precision(&rules);
    let mut runs: BTreeMap<&'static str, Vec<ArmRun>> = BTreeMap::new();
    for pair in 0..pairs {
        let arms: [(&'static str, &ServingConfig<'_>); 2] = if pair % 2 == 0 {
            [("unrecoded", base), ("recoded", &recoded)]
        } else {
            [("recoded", &recoded), ("unrecoded", base)]
        };
        for (name, config) in arms {
            let run = bench_arm(parsed, bytes, config, prompt_ids, max_tokens)?;
            print_run(pair, name, &run);
            runs.entry(name).or_default().push(run);
        }
    }
    for (name, samples) in &runs {
        print_summary(name, samples);
    }
    println!(
        "pairing pairs={} unrecoded_runs={} recoded_runs={}",
        pairs,
        runs.get("unrecoded").map_or(0, Vec::len),
        runs.get("recoded").map_or(0, Vec::len)
    );
    println!(
        "ranking_hash={:016x} candidate={} budget={} assigned={} recoded_bytes={}",
        report.ranking_hash,
        report.candidate_codec,
        allocation.budget_bytes,
        allocation.assigned_bytes,
        allocation.recoded_bytes
    );
    Ok(())
}

fn bench_arm(
    parsed: &ParsedGguf,
    bytes: &[u8],
    config: &ServingConfig<'_>,
    prompt_ids: &[Vec<u32>],
    max_tokens: usize,
) -> DriverResult<ArmRun> {
    let model =
        LoadedModel::load_with_serving(parsed, bytes, config).map_err(|error| error.to_string())?;
    let mut run = ArmRun {
        steady_footprint_bytes: current_footprint_bytes(),
        steady_rss_bytes: current_rss_bytes(),
        ..ArmRun::default()
    };
    let cold_prompt = prompt_ids.first().ok_or("the calibration set is empty")?;
    let warm_prompt = prompt_ids.get(1).unwrap_or(cold_prompt);
    let (cold_ns, _, cold_tokens, cold_hash) =
        generate_measured(&model, cold_prompt, max_tokens, config, &mut run)?;
    let (warm_ns, warm_span_ns, warm_tokens, warm_hash) =
        generate_measured(&model, warm_prompt, max_tokens, config, &mut run)?;
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
    prompt: &[u32],
    max_tokens: usize,
    config: &ServingConfig<'_>,
    run: &mut ArmRun,
) -> DriverResult<(u64, u64, usize, u64)> {
    let mut first = None;
    let mut last = 0u64;
    let mut tokens = 0usize;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut on_token = |event: proxima_model_interop::TokenEvent<'_>| {
        if first.is_none() {
            first = Some(event.elapsed_ns);
        }
        last = event.elapsed_ns;
        tokens += 1;
        hash ^= u64::from(event.token_id);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        run.peak_rss_bytes = run.peak_rss_bytes.max(peak_rss_bytes());
        run.peak_footprint_bytes = run.peak_footprint_bytes.max(current_footprint_bytes());
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
        ("peak_footprint_bytes", |run| {
            run.peak_footprint_bytes as f64
        }),
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
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!(
        "llama_bench pairs={pairs} mirror_prompt_tokens={prompt_tokens} mirror_max_tokens={max_tokens}"
    );
    println!("llama_bench_json={stdout}");
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
