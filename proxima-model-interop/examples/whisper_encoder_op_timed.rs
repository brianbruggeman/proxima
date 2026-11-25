//! Per-op GPU time of the real whisper encoder on Metal, through the serving path.
//!
//! Loads the checkpoint with `Model::load` and the serving settings from the environment, exactly as
//! the `whisper` example does, so `PROXIMA_SERVING_WEIGHT_PRECISION` applies and the projections are
//! the recoded weights the serving encoder binds (`Bound::bind`, which reads `weight_precision_rules`
//! and calls `ModelWeights::bind`). `Model::with_encode_op_timings` runs the encode program one
//! command buffer per op through `BackendRuntime::evaluate_segment_op_timed`, the wrapper of
//! `omega::metal::execute_plan_named_op_timed`. Prints `gpu_ns` per op with its kind and node, then a
//! total per kind, then the ids the transcription produced.
//!
//! `--wav <file>` and `--model <dir>` are the same inputs the `whisper` example takes. `--passes <n>`
//! (default 5) runs the timed encode n times in this process and reports each op's min and median
//! `gpu_ns` over them, the first pass included; its total prints apart as `first_pass_gpu_ns_total`.

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
mod run {
    use std::path::PathBuf;

    use anyhow::{Context, Result, bail};
    use conflaguration::Settings;
    use proxima_model_interop::ServingSettings;
    use proxima_model_interop::audio::wav;
    use proxima_model_interop::families::encoder_decoder_host::{
        Model, OpSample, kind_totals, reduce_op_passes,
    };

    pub fn main() -> Result<()> {
        let mut wav_path: Option<PathBuf> = None;
        let mut model_path: Option<PathBuf> = None;
        let mut passes: usize = 5;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--wav" => wav_path = Some(args.next().context("--wav wants a path")?.into()),
                "--model" => model_path = Some(args.next().context("--model wants a path")?.into()),
                "--passes" => {
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
        let model = Model::load(&model_path.context("--model is required")?)?
            .with_settings(settings)
            .with_encode_op_timings();
        let mut transcript = model.transcribe(&audio.samples, "en")?;
        for _ in 1..passes {
            transcript = model.transcribe(&audio.samples, "en")?;
        }
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

        println!("passes={}", sampled.len());
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
        let ids: Vec<String> = transcript.tokens.iter().map(ToString::to_string).collect();
        println!("ids={}", ids.join(","));
        Ok(())
    }
}

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    run::main()
}

#[cfg(not(all(feature = "metal", feature = "instrument", target_os = "macos")))]
fn main() {
    eprintln!("whisper_encoder_op_timed needs the metal and instrument features on macos");
}
