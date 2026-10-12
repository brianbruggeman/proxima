//! Builds the controls `codec_token_agreement --teacher-forced` compares fp6 against:
//! `requantize_noise --bits <0|7|6|5> --policy <default|class,class> --job <source>=<output> ...`
//! and measures a converted file's weight noise: `requantize_noise --rms --policy <...> --job
//! <source>=<other> ...`.
//!
//! Composes `parse_complete`, `gguf_tensor_as_f32` (dequantize whatever the source stores),
//! `policy_selects` (the tensors fp6 converts), `proxima_gguf::quant::q8_0::quantize` and
//! `write_complete`. `--bits 0` re-encodes the selected tensors to Q8_0 unchanged (the identity
//! control). `--bits N` first rounds every block of 32 weights to a symmetric N-bit integer grid
//! (`scale = absmax / (2^(N-1) - 1)`) and then stores the rounded values as Q8_0, so the Q8_0
//! kernel runs the noise. Each job prints `requantized bits=<N> tensors=<n> rel_rms=<r>` where
//! `rel_rms` is `sqrt(sum (written - source)^2 / sum source^2)` over the selected tensors;
//! `--rms` prints the same figure between a source and any other checkpoint (an fp6 file).

use std::error::Error;
use std::fs::{File, create_dir_all, write};
use std::path::PathBuf;

use memmap2::Mmap;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::q8_0::{self, BLOCK_BYTES, QK8_0};
use proxima_gguf::quant::scaled_float as fp6;
use proxima_gguf::{GgmlType, GgufModel, TensorPayload, parse_complete, write_complete};
use proxima_model_interop::{CodecPolicySettings, gguf_tensor_as_f32, policy_selects};
use proxima_primitives::float_layout::FloatLayout;

const USAGE: &str = "usage: requantize_noise [--rms | --bits <0|7|6|5>] --policy <default|class,class> --job <source>=<output> ...";

#[derive(Default)]
struct Flags {
    bits: Option<u32>,
    rms: bool,
    policy: Option<String>,
    jobs: Vec<(PathBuf, PathBuf)>,
}

fn read_flags(mut arguments: impl Iterator<Item = String>) -> Result<Flags, Box<dyn Error>> {
    let mut flags = Flags::default();
    while let Some(flag) = arguments.next() {
        if flag == "--rms" {
            flags.rms = true;
            continue;
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} has no value"))?;
        match flag.as_str() {
            "--bits" => flags.bits = Some(value.parse()?),
            "--policy" => flags.policy = Some(value),
            "--job" => {
                let (source, output) = value
                    .split_once('=')
                    .ok_or_else(|| format!("job {value:?} is not <source>=<output>"))?;
                flags
                    .jobs
                    .push((PathBuf::from(source), PathBuf::from(output)));
            }
            other => return Err(format!("unknown flag {other}\n{USAGE}").into()),
        }
    }
    Ok(flags)
}

fn policy_of(raw: &str) -> CodecPolicySettings {
    let builder = CodecPolicySettings::builder();
    match raw {
        "default" => builder.build(),
        classes => builder
            .classes(classes.split(',').map(str::to_owned).collect())
            .build(),
    }
}

fn round_block(block: &mut [f32], bits: u32) {
    let levels = ((1u32 << (bits - 1)) - 1) as f32;
    let peak = block
        .iter()
        .fold(0.0f32, |peak, value| peak.max(value.abs()));
    if peak == 0.0 {
        return;
    }
    let scale = peak / levels;
    block
        .iter_mut()
        .for_each(|value| *value = (*value / scale).round().clamp(-levels, levels) * scale);
}

struct Energy {
    error: f64,
    signal: f64,
}

impl Energy {
    fn add(&mut self, source: &[f32], written: &[f32]) {
        for (before, after) in source.iter().zip(written) {
            let delta = f64::from(*after) - f64::from(*before);
            self.error += delta * delta;
            self.signal += f64::from(*before) * f64::from(*before);
        }
    }

    fn rel_rms(&self) -> f64 {
        (self.error / self.signal).sqrt()
    }
}

fn open(path: &PathBuf) -> Result<Mmap, Box<dyn Error>> {
    let file = File::open(path)?;
    // the checkpoint is not written while this process reads it
    Ok(unsafe { Mmap::map(&file)? })
}

fn selected<'a>(
    parsed: &'a ParsedGguf,
    policy: &'a CodecPolicySettings,
) -> impl Iterator<Item = &'a str> {
    parsed
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .filter(|name| policy_selects(policy, name))
}

fn requantize(
    source: &PathBuf,
    output: &PathBuf,
    bits: u32,
    policy: &CodecPolicySettings,
) -> Result<String, Box<dyn Error>> {
    let mapping = open(source)?;
    let parsed = parse_complete(&mapping)?;
    let mut energy = Energy {
        error: 0.0,
        signal: 0.0,
    };
    let mut written_blocks: Vec<Option<Vec<u8>>> = Vec::new();
    for tensor in &parsed.tensors {
        if !policy_selects(policy, &tensor.name) {
            written_blocks.push(None);
            continue;
        }
        let weights = gguf_tensor_as_f32(&parsed, &mapping, &tensor.name)?;
        let mut noised = weights.clone();
        if bits > 0 {
            noised
                .chunks_mut(QK8_0)
                .for_each(|block| round_block(block, bits));
        }
        let mut blocks = vec![0u8; noised.len() / QK8_0 * BLOCK_BYTES];
        q8_0::quantize(&noised, &mut blocks)?;
        let mut decoded = vec![0f32; noised.len()];
        q8_0::dequantize(&blocks, &mut decoded)?;
        energy.add(&weights, &decoded);
        written_blocks.push(Some(blocks));
    }
    let tensors = parsed
        .tensors
        .iter()
        .zip(&written_blocks)
        .map(|(tensor, blocks)| {
            let range = parsed.tensor_data_range(tensor, mapping.len() as u64)?;
            let (ggml_type, data) = match blocks {
                Some(blocks) => (GgmlType::Q8_0, blocks.as_slice()),
                None => (
                    tensor.ggml_type,
                    &mapping[range.start as usize..range.end as usize],
                ),
            };
            Ok(TensorPayload {
                name: tensor.name.clone(),
                dims: tensor.dims.clone(),
                ggml_type,
                data,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let model = GgufModel {
        version: parsed.version,
        metadata: parsed.metadata.clone(),
        tensors,
    };
    let bytes = write_complete(&model)?;
    if let Some(directory) = output.parent() {
        create_dir_all(directory)?;
    }
    write(output, &bytes)?;
    let count = written_blocks.iter().flatten().count();
    Ok(format!(
        "requantized bits={bits} tensors={count} rel_rms={:.6} bytes={}",
        energy.rel_rms(),
        bytes.len()
    ))
}

fn weights_of(parsed: &ParsedGguf, mapping: &Mmap, name: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or("tensor missing")?;
    if tensor.ggml_type != GgmlType::Fp6E3M2 {
        return Ok(gguf_tensor_as_f32(parsed, mapping, name)?);
    }
    let range = parsed.tensor_data_range(tensor, mapping.len() as u64)?;
    let data = &mapping[range.start as usize..range.end as usize];
    let mut weights = vec![0f32; data.len() / fp6::BLOCK_BYTES * fp6::BLOCK_ELEMENTS];
    for (block, out) in data
        .as_chunks::<{ fp6::BLOCK_BYTES }>()
        .0
        .iter()
        .zip(weights.as_chunks_mut::<{ fp6::BLOCK_ELEMENTS }>().0)
    {
        fp6::dequantize_block(block, FloatLayout::E3M2, out)?;
    }
    Ok(weights)
}

fn measure(
    source: &PathBuf,
    other: &PathBuf,
    policy: &CodecPolicySettings,
) -> Result<String, Box<dyn Error>> {
    let (source_map, other_map) = (open(source)?, open(other)?);
    let (source_parsed, other_parsed) = (parse_complete(&source_map)?, parse_complete(&other_map)?);
    let mut energy = Energy {
        error: 0.0,
        signal: 0.0,
    };
    let mut count = 0usize;
    for name in selected(&source_parsed, policy) {
        let before = weights_of(&source_parsed, &source_map, name)?;
        let after = weights_of(&other_parsed, &other_map, name)?;
        energy.add(&before, &after);
        count += 1;
    }
    Ok(format!(
        "measured tensors={count} rel_rms={:.6}",
        energy.rel_rms()
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let flags = read_flags(std::env::args().skip(1))?;
    let policy = policy_of(flags.policy.as_deref().ok_or(USAGE)?);
    if flags.jobs.is_empty() {
        return Err(USAGE.into());
    }
    for (source, output) in &flags.jobs {
        let line = if flags.rms {
            measure(source, output, &policy)
        } else {
            requantize(source, output, flags.bits.ok_or(USAGE)?, &policy)
        };
        println!(
            "{}: {}",
            output.display(),
            line.map_err(|error| format!("{}: {error}", source.display()))?
        );
    }
    Ok(())
}
