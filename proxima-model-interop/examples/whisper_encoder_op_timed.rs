//! Per-op time of the real whisper encoder, through the serving path.
//!
//! Loads the checkpoint with `Model::load` and the serving settings from the environment, exactly as
//! the `whisper` example does, so `PROXIMA_SERVING_WEIGHT_PRECISION` applies and the projections are
//! the recoded weights the serving encoder binds (`Bound::bind`, which reads `weight_precision_rules`
//! and calls `ModelWeights::bind`).
//!
//! On Metal (`metal` feature, macos, `PROXIMA_SERVING_GPU_LAYERS` not 0) `Model::with_encode_op_timings`
//! runs the encode program one command buffer per op through `BackendRuntime::evaluate_segment_op_timed`,
//! the wrapper of `omega::metal::execute_plan_named_op_timed`, and the table prints `gpu_ns` per op.
//!
//! On the CPU (`PROXIMA_SERVING_GPU_LAYERS=0`, or a build with no GPU feature) the serving encode runs
//! `CpuStepPlan::execute` -> `run_prepared_step`, which records every node into the
//! `PROXIMA_ARENA_PER_NODE=1` table (`proxima_tensor::instrument::record_arena_node_ticks`); the table
//! prints `cpu_ns` per node with its op kind (the fused `cached_attention` included), extents and the
//! dot-tile outcome recorded for it. The process must run with `PROXIMA_ARENA_PER_NODE=1`.
//!
//! `--wav <file>` and `--model <dir>` are the same inputs the `whisper` example takes. `--passes <n>`
//! (alias `--repeat`, default 5) runs the timed encode n times in this process and reports each op's
//! min and median over them, the first pass included.

#[cfg(feature = "instrument")]
mod run {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use anyhow::{Context, Result, bail};
    use conflaguration::Settings;
    use proxima_model_interop::ServingSettings;
    use proxima_model_interop::audio::wav;
    use proxima_model_interop::families::encoder_decoder_host::Model;
    use proxima_tensor::instrument::ticks_to_nanos;

    fn median(sorted: &[u64]) -> u64 {
        let middle = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[middle - 1] + sorted[middle]) / 2
        } else {
            sorted[middle]
        }
    }

    pub fn main() -> Result<()> {
        let mut wav_path: Option<PathBuf> = None;
        let mut model_path: Option<PathBuf> = None;
        let mut passes: usize = 5;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--wav" => wav_path = Some(args.next().context("--wav wants a path")?.into()),
                "--model" => model_path = Some(args.next().context("--model wants a path")?.into()),
                "--passes" | "--repeat" => {
                    passes = args
                        .next()
                        .context("--passes wants a count")?
                        .parse()
                        .context("--passes wants a positive integer")?;
                }
                other => bail!("unknown argument {other}"),
            }
        }
        if passes == 0 {
            bail!("--passes must be at least 1");
        }
        let audio = wav::read(&wav_path.context("--wav is required")?)?;
        let settings =
            <ServingSettings as Settings>::from_env().context("serving settings from env")?;
        let on_gpu = settings.gpu_layers != 0;
        let model =
            Model::load(&model_path.context("--model is required")?)?.with_settings(settings);
        let mut transcript = None;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let model = if on_gpu {
            model.with_encode_op_timings()
        } else {
            model
        };
        for _ in 0..passes {
            transcript = Some(model.transcribe(&audio.samples, "en")?);
        }
        println!("passes={passes} engine={}", engine_name(on_gpu));
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if on_gpu {
            gpu_table(&model, passes)?;
        }
        if !on_gpu || !cfg!(all(feature = "metal", target_os = "macos")) {
            cpu_table(&model, passes)?;
        }
        let ids: Vec<String> = transcript
            .context("the transcription ran no pass")?
            .tokens
            .iter()
            .map(ToString::to_string)
            .collect();
        println!("ids={}", ids.join(","));
        Ok(())
    }

    fn engine_name(on_gpu: bool) -> &'static str {
        if on_gpu && cfg!(all(feature = "metal", target_os = "macos")) {
            "metal"
        } else {
            "cpu"
        }
    }

    fn cpu_table(model: &Model, passes: usize) -> Result<()> {
        let raw = model.encode_cpu_node_passes();
        let first = raw.first().context("the transcription ran no encode")?;
        if first.0.is_empty() {
            bail!(
                "the CPU encode recorded no per-node timings; run with PROXIMA_ARENA_PER_NODE=1 and PROXIMA_SERVING_GPU_LAYERS=0"
            );
        }
        let mut per_node: BTreeMap<u32, Vec<u64>> = BTreeMap::new();
        for (nodes, _) in &raw {
            for (node, stat) in nodes {
                per_node
                    .entry(*node)
                    .or_default()
                    .push(ticks_to_nanos(stat.ticks));
            }
        }
        let mut kind_sums: BTreeMap<&'static str, (usize, u64, u64)> = BTreeMap::new();
        let mut min_total = 0_u64;
        for (node, stat) in &first.0 {
            let mut times = per_node.remove(node).unwrap_or_default();
            times.sort_unstable();
            let (min_ns, median_ns) = (times[0], median(&times));
            let outcomes: Vec<String> = first
                .1
                .iter()
                .filter(|(asked, _, _)| asked == node)
                .map(|(_, reason, calls)| format!("{reason}x{calls}"))
                .collect();
            let dot_tile = if outcomes.is_empty() {
                "not_asked".to_string()
            } else {
                outcomes.join("+")
            };
            println!(
                "op node={node} kind={} extents={:?} hits={} passes={} cpu_ns_min={min_ns} cpu_ns_median={median_ns} dot_tile={dot_tile}",
                stat.kind_label,
                stat.extents,
                stat.count,
                times.len()
            );
            let entry = kind_sums.entry(stat.kind_label).or_insert((0, 0, 0));
            *entry = (entry.0 + 1, entry.1 + min_ns, entry.2 + median_ns);
            min_total += min_ns;
        }
        for (kind, (count, min_sum, median_sum)) in &kind_sums {
            println!(
                "kind={kind} ops={count} passes={passes} cpu_ns_min_total={min_sum} cpu_ns_median_total={median_sum}"
            );
        }
        println!(
            "ops={} passes={passes} cpu_ns_min_total={min_total}",
            first.0.len()
        );
        Ok(())
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn gpu_table(model: &Model, passes: usize) -> Result<()> {
        use proxima_model_interop::families::encoder_decoder_host::{
            OpSample, kind_totals, reduce_op_passes,
        };

        let raw = model.encode_op_timing_passes();
        let first_pass = raw
            .first()
            .context("the transcription ran no profiled encode")?;
        if first_pass.is_empty() {
            bail!("the timed run returned no per-op timings");
        }
        let sampled: Vec<Vec<OpSample>> = raw
            .iter()
            .map(|pass| pass.iter().map(OpSample::from).collect())
            .collect();
        let stats = reduce_op_passes(&sampled)?;
        let first_total: u64 = first_pass.iter().map(|timing| timing.gpu_ns).sum();
        println!("first_pass_gpu_ns_total={first_total}");
        for stat in &stats {
            println!(
                "op node={} kind={} extents={:?} passes={} gpu_ns_min={} gpu_ns_median={}",
                stat.node.0, stat.kind, stat.extents, stat.passes, stat.min_ns, stat.median_ns
            );
        }
        for (kind, (count, min_total, median_total)) in kind_totals(&stats) {
            println!(
                "kind={kind} ops={count} passes={passes} gpu_ns_min_total={min_total} gpu_ns_median_total={median_total}"
            );
        }
        let min_total: u64 = stats.iter().map(|stat| stat.min_ns).sum();
        println!(
            "ops={} passes={passes} gpu_ns_min_total={min_total}",
            stats.len()
        );
        Ok(())
    }
}

#[cfg(feature = "instrument")]
fn main() -> anyhow::Result<()> {
    run::main()
}

#[cfg(not(feature = "instrument"))]
fn main() {
    eprintln!("whisper_encoder_op_timed needs the instrument feature");
}
