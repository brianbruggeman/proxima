//! Per-op time of the real whisper stages, through the serving path.
//!
//! Loads the checkpoint with `Model::load` and the serving settings from the environment, exactly as
//! the `whisper` example does, so `PROXIMA_SERVING_WEIGHT_PRECISION` applies and the projections are
//! the recoded weights the serving programs bind (`Bound::bind`, which reads `weight_precision_rules`
//! and calls `ModelWeights::bind`).
//!
//! On a GPU build (`metal` on macos, or `cuda`; `PROXIMA_SERVING_GPU_LAYERS` not 0) `Model::with_op_timings`
//! times the mel front end, the encoder and every decode step through
//! `BackendRuntime::evaluate_segment_op_timed`: Metal runs one command buffer per op
//! (`omega::metal::execute_plan_named_op_timed`), CUDA brackets each launch with an event pair
//! (`omega::backend::execute_plan_named_cuda_op_timed`). Both return `omega::OpGpuTiming`. Per stage the
//! table prints a per-kind total and the top-10 ops by `gpu_ns`; mel and encode are reduced over
//! `--passes`, decode is one steady step (the middle step of the last transcription), unreduced.
//!
//! On the CPU (`PROXIMA_SERVING_GPU_LAYERS=0`, or a build with no GPU feature) the serving encode runs
//! `CpuStepPlan::execute` -> `run_prepared_step`, which records every node into the
//! `PROXIMA_ARENA_PER_NODE=1` table (`proxima_tensor::instrument::record_arena_node_ticks`); the table
//! prints `cpu_ns` per node with its op kind (the fused `cached_attention` included), extents and the
//! dot-tile outcome recorded for it. The process must run with `PROXIMA_ARENA_PER_NODE=1`.
//!
//! `--wav <file>` and `--model <dir>` are the same inputs the `whisper` example takes. `--passes <n>`
//! (alias `--repeat`, default 5) runs the timed transcription n times in this process and reports each
//! op's min and median over them, the first pass included.

#[cfg(feature = "instrument")]
mod run {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use anyhow::{Context, Result, bail};
    use conflaguration::Settings;
    use proxima_model_interop::ServingSettings;
    use proxima_model_interop::audio::wav;
    use proxima_model_interop::families::encoder_decoder_host::Model;
    use proxima_numeric::NumericPolicy;
    use proxima_tensor::instrument::ticks_to_nanos;

    const GPU_TIMED: bool = cfg!(any(
        feature = "cuda",
        all(feature = "metal", target_os = "macos")
    ));

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
        let mut cpu_policy: Option<String> = None;
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
                "--cpu-policy" => {
                    cpu_policy = Some(
                        args.next()
                            .context("--cpu-policy wants relaxed or approx")?,
                    );
                }
                other => bail!("unknown argument {other}"),
            }
        }
        if passes == 0 {
            bail!("--passes must be at least 1");
        }
        let audio = wav::read(&wav_path.context("--wav is required")?)?;
        let mut settings =
            <ServingSettings as Settings>::from_env().context("serving settings from env")?;
        match cpu_policy.as_deref() {
            None => {}
            Some("relaxed") => settings.numeric_policy = NumericPolicy::llama_relaxed(),
            Some("approx") => {
                settings.numeric_policy =
                    NumericPolicy::llama_relaxed().with_approx_functions(true);
            }
            Some(other) => bail!("--cpu-policy wants relaxed or approx, not {other}"),
        }
        let on_gpu = settings.gpu_layers != 0 && GPU_TIMED;
        let model =
            Model::load(&model_path.context("--model is required")?)?.with_settings(settings);
        #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
        let model = if on_gpu {
            model.with_op_timings()
        } else {
            model
        };
        let mut transcript = None;
        #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
        let mut last_run_steps = 0;
        for _ in 0..passes {
            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
            let before = model.op_timing_passes("decode").len();
            transcript = Some(model.transcribe(&audio.samples, "en")?);
            #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
            {
                last_run_steps = model.op_timing_passes("decode").len() - before;
            }
        }
        println!("passes={passes} engine={}", engine_name(on_gpu));
        #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
        if on_gpu {
            gpu_tables(&model, last_run_steps)?;
        }
        if !on_gpu {
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
        match (on_gpu, cfg!(feature = "cuda")) {
            (true, true) => "cuda",
            (true, false) => "metal",
            (false, _) => "cpu",
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

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    const TOP_OPS: usize = 10;

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    fn reduce(
        raw: &[Vec<omega::OpGpuTiming>],
    ) -> Result<Vec<proxima_model_interop::families::encoder_decoder_host::OpPassStats>> {
        use proxima_model_interop::families::encoder_decoder_host::{OpSample, reduce_op_passes};

        let sampled: Vec<Vec<OpSample>> = raw
            .iter()
            .map(|pass| pass.iter().map(OpSample::from).collect())
            .collect();
        reduce_op_passes(&sampled)
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    fn report(
        stage: &str,
        passes: usize,
        stats: &[proxima_model_interop::families::encoder_decoder_host::OpPassStats],
    ) {
        use proxima_model_interop::families::encoder_decoder_host::kind_totals;

        let min_total: u64 = stats.iter().map(|stat| stat.min_ns).sum();
        println!(
            "stage={stage} ops={} passes={passes} gpu_ns_min_total={min_total}",
            stats.len()
        );
        let mut kinds: Vec<_> = kind_totals(stats).into_iter().collect();
        kinds.sort_by_key(|(_, (_, min_ns, _))| std::cmp::Reverse(*min_ns));
        for (kind, (count, min_ns, median_ns)) in kinds {
            println!(
                "stage={stage} kind={kind} ops={count} gpu_ns_min_total={min_ns} gpu_ns_median_total={median_ns}"
            );
        }
        let mut ranked: Vec<_> = stats.iter().collect();
        ranked.sort_by_key(|stat| std::cmp::Reverse(stat.min_ns));
        for stat in ranked.into_iter().take(TOP_OPS) {
            println!(
                "stage={stage} top node={} kind={} extents={:?} gpu_ns_min={} gpu_ns_median={}",
                stat.node.0, stat.kind, stat.extents, stat.min_ns, stat.median_ns
            );
        }
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    fn gpu_tables(model: &Model, last_run_steps: usize) -> Result<()> {
        for stage in ["mel", "encode"] {
            let raw = model.op_timing_passes(stage);
            if raw.first().is_none_or(Vec::is_empty) {
                bail!("the {stage} stage returned no per-op timings");
            }
            report(stage, raw.len(), &reduce(&raw)?);
        }
        let decode = model.op_timing_passes("decode");
        let last_run = &decode[decode.len() - last_run_steps.min(decode.len())..];
        let steady = last_run
            .get(last_run.len() / 2)
            .filter(|step| !step.is_empty())
            .context("the transcription ran no timed decode step")?;
        report("decode", 1, &reduce(std::slice::from_ref(steady))?);
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
