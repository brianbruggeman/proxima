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
//! `--wav <file>` and `--model <dir>` are the same inputs the `whisper` example takes.

#[cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
mod run {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use anyhow::{Context, Result, bail};
    use conflaguration::Settings;
    use proxima_model_interop::ServingSettings;
    use proxima_model_interop::audio::wav;
    use proxima_model_interop::families::encoder_decoder_host::Model;

    pub fn main() -> Result<()> {
        let mut wav_path: Option<PathBuf> = None;
        let mut model_path: Option<PathBuf> = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--wav" => wav_path = Some(args.next().context("--wav wants a path")?.into()),
                "--model" => model_path = Some(args.next().context("--model wants a path")?.into()),
                other => bail!("unknown argument {other}"),
            }
        }
        let audio = wav::read(&wav_path.context("--wav is required")?)?;
        let settings =
            <ServingSettings as Settings>::from_env().context("serving settings from env")?;
        let model = Model::load(&model_path.context("--model is required")?)?
            .with_settings(settings)
            .with_encode_op_timings();
        let transcript = model.transcribe(&audio.samples, "en")?;
        let timings = model
            .encode_op_timings()
            .context("the transcription ran no profiled encode")?;
        if timings.is_empty() {
            bail!("the timed run returned no per-op timings");
        }

        let mut per_kind: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
        for timing in timings {
            println!(
                "op node={} kind={} extents={:?} gpu_ns={}",
                timing.node.0, timing.kind, timing.extents, timing.gpu_ns
            );
            let entry = per_kind.entry(timing.kind).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += timing.gpu_ns;
        }
        for (kind, (count, nanos)) in &per_kind {
            println!("kind={kind} ops={count} gpu_ns_total={nanos}");
        }
        let total: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
        println!("ops={} gpu_ns_total={total}", timings.len());
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
