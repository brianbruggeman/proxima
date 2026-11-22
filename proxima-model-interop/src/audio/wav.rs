//! A minimal RIFF/WAVE reader: PCM integer samples to f32, mixed down to mono.
//!
//! Deliberately small: the speech examples read 16-bit or 24-bit PCM WAV, and
//! every other container stays out. A file that needs a decoder is refused by
//! name rather than guessed at.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// One decoded sound: mono f32 samples in `[-1, 1]` and the file's rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Read a PCM WAV file at `path`.
///
/// # Errors
///
/// A non-RIFF file, a missing `fmt `/`data` chunk, a compressed format, or a
/// bit depth this reader does not decode.
pub fn read(path: &Path) -> Result<Audio> {
    let bytes = fs::read(path).with_context(|| path.display().to_string())?;
    parse(&bytes).with_context(|| path.display().to_string())
}

fn parse(bytes: &[u8]) -> Result<Audio> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("not a RIFF/WAVE file");
    }
    let mut position = 12;
    let mut format = None;
    let mut samples = None;
    while position + 8 <= bytes.len() {
        let id = &bytes[position..position + 4];
        let length = u32::from_le_bytes(
            bytes[position + 4..position + 8]
                .try_into()
                .expect("chunk length"),
        ) as usize;
        let body = position + 8;
        if body + length > bytes.len() {
            bail!("truncated chunk");
        }
        match id {
            b"fmt " => {
                let tag = u16_at(bytes, body);
                let channels = u16_at(bytes, body + 2) as usize;
                let sample_rate = u32_at(bytes, body + 4);
                let bits = u16_at(bytes, body + 14);
                if tag != 1 {
                    bail!("compressed WAV format {tag}; only PCM is read");
                }
                format = Some((channels, sample_rate, bits));
            }
            b"data" => samples = Some(body..body + length),
            _ => {}
        }
        position = body + length + (length & 1);
    }
    let (channels, sample_rate, bits) = format.context("no fmt chunk")?;
    if channels == 0 {
        bail!("zero channels");
    }
    let range = samples.context("no data chunk")?;
    let data = &bytes[range];
    let values: Vec<f32> = match bits {
        16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| f32::from(i16::from_le_bytes(*word)) / 32768.0)
            .collect(),
        24 => data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|word| {
                let value = i32::from_le_bytes([word[0], word[1], word[2], 0]) >> 8;
                value as f32 / 8_388_608.0
            })
            .collect(),
        32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect(),
        other => bail!("unsupported PCM bit depth {other}"),
    };
    let samples = if channels == 1 {
        values
    } else {
        values
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok(Audio {
        samples,
        sample_rate,
    })
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("two bytes"))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}
