//! Rewrite a ggml whisper checkpoint with every F16 tensor stored as F32.
//!
//! The published `ggml-tiny.bin` is ftype 1 (F16 weights); the HF safetensors
//! this crate reads are F32. The two cannot be compared at an F32-derived
//! ceiling, so this tool widens the published file without changing a value.
//! The F16 decode is `proxima_numeric`'s codec, not a local half implementation.
//!
//! usage: ggml_rewrite <in.ggml> <out.ggml>

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use proxima_gguf::GgmlType;
use proxima_numeric::{NumericFormat, decode_blocks};

const MAGIC: u32 = 0x6767_6D6C;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let source = args.next().context("usage: ggml_rewrite <in> <out>")?;
    let target = args.next().context("usage: ggml_rewrite <in> <out>")?;
    convert(Path::new(&source), Path::new(&target))
}

fn convert(source: &Path, target: &Path) -> Result<()> {
    let mut input = File::open(source).with_context(|| source.display().to_string())?;
    let mut output = File::create(target).with_context(|| target.display().to_string())?;

    let magic = read_i32(&mut input)? as u32;
    if magic != MAGIC {
        bail!("bad ggml magic {magic:#x}");
    }
    write_i32(&mut output, magic as i32)?;

    let mut hparams = [0_i32; 11];
    for slot in &mut hparams {
        *slot = read_i32(&mut input)?;
    }
    hparams[10] = GgmlType::F32.to_wire();
    for value in hparams {
        write_i32(&mut output, value)?;
    }

    let n_mel = read_i32(&mut input)?;
    let n_fft = read_i32(&mut input)?;
    write_i32(&mut output, n_mel)?;
    write_i32(&mut output, n_fft)?;
    copy(&mut input, &mut output, (n_mel * n_fft * 4) as usize)?;

    let n_vocab = read_i32(&mut input)?;
    write_i32(&mut output, n_vocab)?;
    for _ in 0..n_vocab {
        let length = read_i32(&mut input)?;
        write_i32(&mut output, length)?;
        copy(&mut input, &mut output, length as usize)?;
    }

    let mut converted = 0_u32;
    loop {
        let mut head = [0_u8; 12];
        let mut first = [0_u8; 4];
        match input.read(&mut first)? {
            0 => break,
            4 => head[..4].copy_from_slice(&first),
            got => bail!("truncated tensor header ({got} bytes)"),
        }
        input.read_exact(&mut head[4..])?;
        let [dims_bytes, length_bytes, type_bytes] = head.as_chunks::<4>().0 else {
            bail!("tensor header is not three words");
        };
        let n_dims = i32::from_le_bytes(*dims_bytes);
        let length = i32::from_le_bytes(*length_bytes);
        let ttype = i32::from_le_bytes(*type_bytes);
        if !(0..=4).contains(&n_dims) {
            bail!("tensor rank {n_dims} out of range");
        }
        let mut dims = vec![0_i32; n_dims as usize];
        for dim in &mut dims {
            *dim = read_i32(&mut input)?;
        }
        let mut name = vec![0_u8; length as usize];
        input.read_exact(&mut name)?;
        let elements: usize = dims.iter().map(|&dim| dim as usize).product();

        let source = GgmlType::from_wire(ttype)
            .with_context(|| format!("tensor {:?} has unknown ggml type {ttype}", String::from_utf8_lossy(&name)))?;
        let format = source
            .numeric_format()
            .with_context(|| format!("tensor {:?} has no numeric format", String::from_utf8_lossy(&name)))?;
        let bytes_per_element = format.bytes_per_block() as usize / format.elements_per_block() as usize;
        let mut raw = vec![0_u8; elements * bytes_per_element];
        input.read_exact(&mut raw)?;
        let payload: Vec<u8> = if format == NumericFormat::Float32 {
            raw
        } else {
            let mut decoded = vec![0.0_f32; elements];
            decode_blocks(&format, &raw, &mut decoded)
                .with_context(|| format!("decoding {format:?}"))?;
            let mut bytes = Vec::with_capacity(elements * 4);
            for value in decoded {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            converted += 1;
            bytes
        };

        write_i32(&mut output, n_dims)?;
        write_i32(&mut output, length)?;
        write_i32(&mut output, GgmlType::F32.to_wire())?;
        for dim in dims {
            write_i32(&mut output, dim)?;
        }
        output.write_all(&name)?;
        output.write_all(&payload)?;
    }

    println!("converted={converted} tensors; wrote {}", target.display());
    Ok(())
}

fn read_i32(input: &mut File) -> Result<i32> {
    let mut bytes = [0_u8; 4];
    input.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn write_i32(output: &mut File, value: i32) -> Result<()> {
    output.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn copy(input: &mut File, output: &mut File, count: usize) -> Result<()> {
    let mut bytes = vec![0_u8; count];
    input.read_exact(&mut bytes)?;
    output.write_all(&bytes)?;
    Ok(())
}
