//! Universal HOBBIT driver: a per-expert high/low plan on the slab table
//! every MoE family reads, with no pre-gather dependency.
//!
//! ```sh
//! hobbit_universal plan    --config moe.toml --model checkpoint.gguf [--json-out plan.json]
//! hobbit_universal prepare --config moe.toml --model checkpoint.gguf --sidecar sidecar.pxex
//! hobbit_universal run     --config moe.toml --model checkpoint.gguf --sidecar sidecar.pxex \
//!                          [--plan plan.json] [--pairs 3] [--max-tokens 8] [--llama-bench PATH]
//! ```
//!
//! `plan` runs one CPU calibration generation, reads the model's own routed
//! expert selection counter, and ranks each layer's experts by `(count, ema,
//! expert)`; the coldest `low_expert_ppm` parts per million are low.
//! `prepare` writes the low-codec copies. `run` installs only the plan's low
//! experts into the slab and measures high-only versus mixed arms
//! interleaved; the slab's own `sources_for_step` is the mixed table, so the
//! same code path works for the uniform, sliding-pattern and recurrent MoE
//! families on CPU and GPU.
//!
//! `run` prints one raw JSON record per arm run (integer nanoseconds, token
//! counts and byte sizes; no rates, no statistics). Collect them with
//! `grep '^{'` into a file and run `scripts/scoreboard/scoreboard summarize`
//! on it for the percentiles, mean and CoV.

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
    HobbitQualityDelta, calibration_ids, check_quality_bound, quality_delta,
};
use proxima_model_interop::{
    ExpertStackSpec, HobbitLayerPlan, HobbitLayerScore, LoadedModel, PPM, ServingConfig,
    ServingSettings, TokenEvent, fixed_point,
};
use proxima_numeric::NumericFormat;

const USAGE: &str = "usage: hobbit_universal <plan|prepare|run> --config PATH --model PATH \
[--sidecar PATH] [--plan PATH] [--json-out PATH] [--pairs N] [--max-tokens N] [--llama-bench PATH]";

type DriverResult<T> = Result<T, String>;

/// A plan to measure and the sidecar its low stacks come from.
#[derive(Clone, Copy)]
struct Candidate<'a> {
    sidecar_path: &'a Path,
    plan: &'a HobbitLayerPlan,
}

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("hobbit_universal: {message}");
            ExitCode::FAILURE
        }
    }
}

struct Flags {
    command: String,
    config: PathBuf,
    model: PathBuf,
    sidecar: Option<PathBuf>,
    plan: Option<PathBuf>,
    json_out: Option<PathBuf>,
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
    let mut plan = None;
    let mut json_out = None;
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
            "--plan" => plan = Some(PathBuf::from(value()?)),
            "--json-out" => json_out = Some(PathBuf::from(value()?)),
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
        sidecar,
        plan,
        json_out,
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

    match flags.command.as_str() {
        "plan" => {
            let sidecar = flags.sidecar.as_deref().ok_or("--sidecar is required for plan")?;
            let plan = calibrate(&parsed, bytes, &settings, sidecar)?;
            println!("plan={}", plan.describe());
            if let Some(path) = &flags.json_out {
                let json =
                    serde_json::to_string_pretty(&plan).map_err(|error| error.to_string())?;
                std::fs::write(path, json)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
            }
            Ok(())
        }
        "prepare" => {
            let sidecar = flags.sidecar.as_deref().ok_or("--sidecar is required")?;
            prepare(&parsed, bytes, &settings, sidecar)
        }
        "run" => {
            let sidecar = flags.sidecar.as_deref().ok_or("--sidecar is required")?;
            let plan = match &flags.plan {
                Some(path) => {
                    let json = std::fs::read_to_string(path)
                        .map_err(|error| format!("{}: {error}", path.display()))?;
                    serde_json::from_str(&json)
                        .map_err(|error| format!("{}: {error}", path.display()))?
                }
                None => calibrate(&parsed, bytes, &settings, sidecar)?,
            };
            println!("plan={}", plan.describe());
            run_arms(&parsed, bytes, &settings, &plan, sidecar, &flags)
        }
        other => Err(format!("unknown command {other}; {USAGE}")),
    }
}

fn base_config(settings: &ServingSettings) -> ServingConfig<'_> {
    ServingConfig {
        moe_pre_gather: false,
        ..settings.as_serving_config(&[])
    }
}

fn calibrate(
    parsed: &ParsedGguf,
    bytes: &[u8],
    settings: &ServingSettings,
    sidecar_path: &Path,
) -> DriverResult<HobbitLayerPlan> {
    let config = base_config(settings);
    let held_out = calibration_ids(
        parsed,
        &settings.moe_expert_precision.held_out_prompts,
        settings.hobbit.max_calibration_tokens,
    )
    .map_err(|error| error.to_string())?;
    with_reference(parsed, bytes, &config, &held_out, |delta_for| {
        search_plan(parsed, settings, sidecar_path, delta_for)
    })
}

fn search_plan(
    parsed: &ParsedGguf,
    settings: &ServingSettings,
    sidecar_path: &Path,
    delta_for: &dyn Fn(Option<Candidate<'_>>) -> DriverResult<HobbitQualityDelta>,
) -> DriverResult<HobbitLayerPlan> {
    let share = settings.moe_expert_precision.checked_low_layer_ppm()?;
    let bound = settings
        .moe_expert_precision
        .quality_bound()
        .ok_or("moe_expert_precision.quality sets no bound; no bound, no apply")?;
    let layers = parsed
        .metadata_value("general.architecture")
        .and_then(|value| match value {
            proxima_gguf::value::MetadataValue::String(family) => {
                parsed.metadata_value(&format!("{family}.block_count"))
            }
            _ => None,
        })
        .and_then(|value| match value {
            proxima_gguf::value::MetadataValue::U32(count) => Some(*count),
            _ => None,
        })
        .ok_or("the checkpoint has no block_count")?;
    let mut scores = Vec::with_capacity(layers as usize);
    for layer in 0..layers {
        let plan = HobbitLayerPlan::from_low_layers(layers, PPM, vec![layer], Vec::new());
        let delta = delta_for(Some(Candidate { sidecar_path, plan: &plan }))?;
        eprintln!(
            "layer={layer} mean_kl={:.9} top1_agreement={:.6}",
            delta.mean_kl, delta.top1_agreement
        );
        scores.push(HobbitLayerScore {
            layer,
            mean_kl_fixed: fixed_point!(delta.mean_kl),
            top1_agreement_fixed: fixed_point!(delta.top1_agreement),
        });
    }
    let mut plan = HobbitLayerPlan::from_measured(scores, layers, share);
    loop {
        let delta = delta_for(Some(Candidate { sidecar_path, plan: &plan }))?;
        match check_quality_bound(&delta, Some(bound)) {
            Ok(()) => {
                println!(
                    "combined_quality positions={} top1_agreement={:.6} mean_kl={:.9} holds=true",
                    delta.positions, delta.top1_agreement, delta.mean_kl
                );
                return Ok(plan);
            }
            Err(error) => {
                let Some(worst) = plan
                    .low_layers
                    .iter()
                    .copied()
                    .max_by(|left, right| {
                        let score = |layer: u32| {
                            plan.layer_scores
                                .iter()
                                .find(|score| score.layer == layer)
                                .map_or(0, |score| score.mean_kl_fixed)
                        };
                        score(*left).cmp(&score(*right))
                    })
                else {
                    return Err(format!(
                        "no low-layer set holds the quality bound: {error}"
                    ));
                };
                let remaining = plan
                    .low_layers
                    .iter()
                    .copied()
                    .filter(|layer| *layer != worst)
                    .collect::<Vec<_>>();
                plan = HobbitLayerPlan::from_low_layers(
                    layers,
                    plan.low_layer_ppm,
                    remaining,
                    plan.layer_scores.clone(),
                );
            }
        }
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
    elements.div_ceil(layout.block_elements as usize) * layout.block_bytes as usize
}

fn prepare(
    parsed: &ParsedGguf,
    bytes: &[u8],
    settings: &ServingSettings,
    sidecar_path: &Path,
) -> DriverResult<()> {
    let codec = settings.moe_expert_precision.low_codec;
    let stacks = expert_stack_specs(parsed, codec);
    if stacks.is_empty() {
        return Err("the checkpoint has no blk.<n>.ffn_*_exps.weight stacks".to_owned());
    }
    let max_elements = stacks
        .iter()
        .map(|stack| stack.out_dim as usize * stack.in_dim as usize)
        .max()
        .unwrap_or(0);
    let mut scratch = vec![Default::default(); max_elements];
    let mut output = vec![0u8; codec_bytes(codec, max_elements)];
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
    plan: &HobbitLayerPlan,
    sidecar_path: &Path,
    flags: &Flags,
) -> DriverResult<()> {
    let config = base_config(settings);
    let held_out = calibration_ids(
        parsed,
        &settings.moe_expert_precision.held_out_prompts,
        settings.hobbit.max_calibration_tokens,
    )
    .map_err(|error| error.to_string())?;
    quality_gate(parsed, bytes, &config, sidecar_path, plan, &held_out, settings)?;

    let mut runs: BTreeMap<&'static str, Vec<ArmRun>> = BTreeMap::new();
    for pair in 0..flags.pairs {
        let arms: [(&'static str, bool); 2] = if pair % 2 == 0 {
            [("high_only", false), ("mixed", true)]
        } else {
            [("mixed", true), ("high_only", false)]
        };
        for (name, mixed) in arms {
            let mut model = LoadedModel::load_with_serving(parsed, bytes, &config)
                .map_err(|error| error.to_string())?;
            if mixed {
                let sidecar = File::open(sidecar_path)
                    .map_err(|error| format!("{}: {error}", sidecar_path.display()))?;
                model
                    .attach_expert_sidecar_file(sidecar)
                    .map_err(|error| error.to_string())?;
                model
                    .apply_hobbit_layer_plan(plan)
                    .map_err(|error| error.to_string())?;
                println!("pair={pair} arm={name} low_stacks={}", model.hobbit_low_stack_count());
            }
            let run = bench_arm(&model, &config, &held_out, flags.max_tokens)?;
            println!("pair={pair} arm={name} low_layers={:?}", plan.low_layers);
            print_run(pair, name, &run);
            runs.entry(name).or_default().push(run);
        }
    }
    println!("statistics: keep the lines that start with a brace and run scripts/scoreboard/scoreboard summarize on them");
    println!(
        "pairing pairs={} high_only_runs={} mixed_runs={}",
        flags.pairs,
        runs.get("high_only").map_or(0, Vec::len),
        runs.get("mixed").map_or(0, Vec::len)
    );
    println!(
        "hobbit_plan={} sidecar_bytes={}",
        plan.describe(),
        std::fs::metadata(sidecar_path)
            .map_err(|error| format!("{}: {error}", sidecar_path.display()))?
            .len()
    );
    if let Some(llama_bench) = &flags.llama_bench {
        run_llama_bench(
            llama_bench,
            &flags.model,
            &held_out,
            flags.max_tokens,
            flags.pairs,
        )?;
    }
    Ok(())
}

fn quality_gate(
    parsed: &ParsedGguf,
    bytes: &[u8],
    config: &ServingConfig<'_>,
    sidecar_path: &Path,
    plan: &HobbitLayerPlan,
    held_out: &[Vec<u32>],
    settings: &ServingSettings,
) -> DriverResult<()> {
    let bound = settings
        .moe_expert_precision
        .quality_bound()
        .ok_or("moe_expert_precision.quality sets no bound; no bound, no apply")?;
    let delta = with_reference(parsed, bytes, config, held_out, |delta_for| {
        delta_for(Some(Candidate { sidecar_path, plan }))
    })?;
    check_quality_bound(&delta, Some(bound)).map_err(|error| error.to_string())?;
    println!(
        "quality positions={} top1_agreement={:.6} mean_kl={:.9} bound_max_mean_kl={:?} bound_min_top1={:?} holds=true",
        delta.positions, delta.top1_agreement, delta.mean_kl, bound.0, bound.1
    );
    Ok(())
}

/// Runs the unmodified model once as the reference, then hands `body` a
/// function that measures any candidate plan against it. The logits row
/// type is whatever the engine returns; it never leaves this function.
fn with_reference<T>(
    parsed: &ParsedGguf,
    bytes: &[u8],
    config: &ServingConfig<'_>,
    held_out: &[Vec<u32>],
    body: impl FnOnce(
        &dyn Fn(Option<Candidate<'_>>) -> DriverResult<HobbitQualityDelta>,
    ) -> DriverResult<T>,
) -> DriverResult<T> {
    let logits = |plan: Option<Candidate<'_>>| -> DriverResult<_> {
        let mut model = LoadedModel::load_with_serving(parsed, bytes, config)
            .map_err(|error| error.to_string())?;
        if let Some(Candidate { sidecar_path, plan }) = plan {
            let sidecar = File::open(sidecar_path)
                .map_err(|error| format!("{}: {error}", sidecar_path.display()))?;
            model
                .attach_expert_sidecar_file(sidecar)
                .map_err(|error| error.to_string())?;
            model
                .apply_hobbit_layer_plan(plan)
                .map_err(|error| error.to_string())?;
        }
        let last = held_out
            .first()
            .and_then(|ids| ids.last())
            .copied()
            .ok_or("the held-out prompt set is empty")?;
        let values = model
            .forward_node_values_from_ids(&[last], &[model.logits_root()], config)
            .map_err(|error| error.to_string())?;
        values
            .into_iter()
            .next()
            .ok_or_else(|| "the logits root produced no row".to_owned())
    };
    let reference = logits(None)?;
    body(&|plan| {
        let candidate = logits(plan)?;
        quality_delta(
            std::slice::from_ref(&reference),
            std::slice::from_ref(&candidate),
        )
        .map_err(|error| error.to_string())
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct ArmRun {
    ttft_cold_ns: u64,
    ttft_warm_ns: u64,
    warm_tokens: usize,
    warm_span_ns: u64,
    tokens: usize,
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
    let cold_prompt = prompt_ids.first().ok_or("the held-out set is empty")?;
    let warm_prompt = prompt_ids.get(1).unwrap_or(cold_prompt);
    let (cold_ns, _, cold_tokens, cold_hash) =
        generate_measured(model, config, cold_prompt, max_tokens, &mut run)?;
    let (warm_ns, warm_span_ns, warm_tokens, warm_hash) =
        generate_measured(model, config, warm_prompt, max_tokens, &mut run)?;
    run.ttft_cold_ns = cold_ns;
    run.ttft_warm_ns = warm_ns;
    run.warm_tokens = warm_tokens;
    run.warm_span_ns = warm_span_ns;
    run.tokens = cold_tokens + warm_tokens;
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
    let record = serde_json::json!({
        "arm": arm,
        "pair": pair,
        "timings_ns": {
            "ttft_cold": run.ttft_cold_ns,
            "ttft_warm": run.ttft_warm_ns,
            "warm_span": run.warm_span_ns,
        },
        "tokens": run.tokens,
        "warm_tokens": run.warm_tokens,
        "resources": {
            "peak_rss_bytes": run.peak_rss_bytes,
            "steady_rss_bytes": run.steady_rss_bytes,
            "peak_footprint_bytes": run.peak_footprint_bytes,
            "steady_footprint_bytes": run.steady_footprint_bytes,
        },
        "text_hash": format!("{:016x}", run.text_hash),
    });
    println!("{record}");
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
