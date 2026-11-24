//! End-to-end speech transcription through the generic encoder-decoder path.
//!
//! Reads a PCM WAV, builds the mel/encoder/cross-KV/decoder graphs from the
//! checkpoint's own tensors and config, runs the greedy prompt from
//! `decode_policy`, and prints the generated token ids and text. It never reads
//! the oracle: the reference ids are the caller's comparison, not an input.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use conflaguration::Settings;
use proxima_model_interop::ServingSettings;
use proxima_model_interop::audio::wav;
use proxima_model_interop::families::encoder_decoder_host::Model;

fn main() -> Result<()> {
    let process_start = Instant::now();
    let mut repeats = 1_usize;
    let mut wav_path: Option<PathBuf> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut language = String::from("en");
    let mut noise = 0.0_f32;
    let mut gpu_layers: Option<i32> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wav" => wav_path = Some(args.next().context("--wav wants a path")?.into()),
            "--model" => model_path = Some(args.next().context("--model wants a path")?.into()),
            "--lang" => language = args.next().context("--lang wants a name")?,
            "--gpu-layers" => {
                gpu_layers = Some(
                    args.next()
                        .context("--gpu-layers wants a count")?
                        .parse()
                        .context("--gpu-layers is an i32")?,
                )
            }
            "--repeat" => {
                repeats = args
                    .next()
                    .context("--repeat wants a count")?
                    .parse()
                    .context("--repeat is a count")?
            }
            "--noise" => {
                noise = args
                    .next()
                    .context("--noise wants an amplitude")?
                    .parse()
                    .context("--noise is an f32")?
            }
            other => bail!("unknown argument {other}"),
        }
    }
    let wav_path = wav_path.context("--wav is required")?;
    let model_path = model_path.context("--model is required")?;

    let audio = wav::read(&wav_path)?;
    if audio.sample_rate != 16_000 {
        bail!("expected 16 kHz audio, got {} Hz", audio.sample_rate);
    }
    let mut samples = audio.samples;
    if noise != 0.0 {
        // A deterministic perturbation: it must change the transcript, so a
        // run that merely echoes a stored answer cannot pass this control.
        let mut state = 0x2545_f491_u32;
        for sample in &mut samples {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            *sample = (*sample + value * noise).clamp(-1.0, 1.0);
        }
    }

    let load_start = std::time::Instant::now();
    let mut settings =
        <ServingSettings as Settings>::from_env().context("serving settings from env")?;
    if let Some(layers) = gpu_layers {
        settings.gpu_layers = layers;
    }
    let gpu_layers = settings.gpu_layers;
    let model = Model::load(&model_path)?.with_settings(settings);
    let load = load_start.elapsed();
    println!("models=1 noise={noise} gpu_layers={gpu_layers}");
    for repeat in 0..repeats {
        let begin = Instant::now();
        let transcript = model.transcribe(&samples, &language)?;
        let cold_offset = if repeat == 0 {
            begin.duration_since(process_start)
        } else {
            Duration::ZERO
        };
        let ids: Vec<String> = transcript.tokens.iter().map(ToString::to_string).collect();
        println!("ids={}", ids.join(","));
        println!("eot={}", transcript.ended_on_eot);
        let steps: Vec<String> = transcript
            .step_times
            .iter()
            .map(|step| step.as_micros().to_string())
            .collect();
        print!(
            "repeat={repeat} load_us={} mel_us={} encode_us={} decode_us={} decode_us_per_token={} ttft_us={} ttlt_us={} step_us={}",
            load.as_micros(),
            transcript.mel.as_micros(),
            transcript.encode.as_micros(),
            transcript.decode.as_micros(),
            transcript
                .decode
                .checked_div(transcript.tokens.len().try_into()?)
                .unwrap_or_default()
                .as_micros(),
            (cold_offset + transcript.first_token).as_micros(),
            (cold_offset + transcript.last_token).as_micros(),
            steps.join(","),
        );
        if let (Some(bytes), Some(peak)) = (
            transcript.gpu_allocated_bytes,
            transcript.gpu_allocated_peak_bytes,
        ) {
            print!(" gpu_allocated_bytes={bytes} gpu_allocated_peak_bytes={peak}");
        }
        println!(
            " prompt_tokens={} generated_tokens={}",
            transcript.prompt_tokens,
            transcript.tokens.len(),
        );
        println!("text={}", transcript.text);
    }
    Ok(())
}
