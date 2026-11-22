//! End-to-end speech transcription through the generic encoder-decoder path.
//!
//! Reads a PCM WAV, builds the mel/encoder/cross-KV/decoder graphs from the
//! checkpoint's own tensors and config, runs the greedy prompt from
//! `decode_policy`, and prints the generated token ids and text. It never reads
//! the oracle: the reference ids are the caller's comparison, not an input.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use proxima_model_interop::audio::wav;
use proxima_model_interop::families::encoder_decoder_host::{Device, Model};

fn main() -> Result<()> {
    let mut wav_path: Option<PathBuf> = None;
    let mut model_path: Option<PathBuf> = None;
    let mut language = String::from("en");
    let mut noise = 0.0_f32;
    let mut device = Device::Cpu;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wav" => wav_path = Some(args.next().context("--wav wants a path")?.into()),
            "--model" => model_path = Some(args.next().context("--model wants a path")?.into()),
            "--lang" => language = args.next().context("--lang wants a name")?,
            "--device" => {
                device = match args.next().context("--device wants cpu|metal")?.as_str() {
                    "cpu" => Device::Cpu,
                    "metal" => Device::Metal,
                    other => bail!("unknown device {other}"),
                }
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
    let model = Model::load(&model_path)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let transcript = model.transcribe_with(&samples, &language, device)?;
    let ids: Vec<String> = transcript.tokens.iter().map(ToString::to_string).collect();
    println!("models=1 ids={} noise={noise} device={device:?}", ids.len());
    println!("ids={}", ids.join(","));
    println!("eot={}", transcript.ended_on_eot);
    println!(
        "load_ms={load_ms:.3} mel_ms={:.3} encode_ms={:.3} decode_ms={:.3} decode_ms_per_token={:.3}",
        transcript.mel_ms,
        transcript.encode_ms,
        transcript.decode_ms,
        transcript.decode_ms / transcript.tokens.len() as f64,
    );
    println!("text={}", transcript.text);
    Ok(())
}
