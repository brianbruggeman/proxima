//! Gate harness for the layout-rendered Q8_0 matvec: runs one real `Q8_0` tensor of a GGUF at the
//! decode shape (`[rows, k] x [k, 1]`) under `llama_relaxed`, then prints one JSON line with the
//! output's FNV-1a checksum, its error against an f64 host reference, the error of a deliberately
//! mis-laid host reference (the control: it must be far larger, so the comparison can fail), and the
//! GPU time of each timed dispatch.
//!
//! The same source is built in two worktrees (old body, new body) and the binaries are run
//! interleaved by the caller; `Q8_OUT` dumps the raw output bits so two runs can be diffed.
//!
//! Environment: `Q8_GGUF` (path), `Q8_TENSOR` (name), `Q8_ITERS`, `Q8_WARM`, `Q8_IDLE_MS`
//! (idle before the timed window; 0 keeps the clocks warm), `Q8_OUT` (optional raw-bits path),
//! `Q8_POLICY=bit_exact` (default `llama_relaxed`), `Q8_CMP` (raw output of another run to diff), `Q8_LIST` (list the Q8_0 tensors and exit).

// allow: a gate harness reads fixed-size records; `as_chunks` buys nothing over `chunks_exact` here.
#![allow(clippy::chunks_exact_to_as_chunks)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::Duration;

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::q8_0;
use proxima_gguf::types::GgmlType;
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, map};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn parse_header(file: &mut File) -> Result<(ParsedGguf, u64), String> {
    let file_len = file.metadata().map_err(|error| error.to_string())?.len();
    let mut prefix_len = 1usize << 22;
    loop {
        let mut buf = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).map_err(|error| error.to_string())?;
        let read = file.read(&mut buf).map_err(|error| error.to_string())?;
        buf.truncate(read);
        if let Ok((parser, events)) = GgufParser::new().push(&buf) {
            let (mut version, mut metadata, mut tensors, mut completion) = (None, Vec::new(), Vec::new(), None);
            for event in events {
                match event {
                    GgufEvent::Header { version: found, .. } => version = Some(found),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete { data_offset, alignment } => completion = Some((data_offset, alignment)),
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().map_err(|error| format!("{error:?}"))?;
                let parsed = ParsedGguf {
                    version,
                    tensor_count: tensors.len() as u64,
                    kv_count: metadata.len() as u64,
                    metadata,
                    tensors,
                    data_offset,
                    alignment,
                };
                return Ok((parsed, file_len));
            }
        }
        if prefix_len as u64 >= file_len {
            return Err("gguf header not found".to_owned());
        }
        prefix_len *= 2;
    }
}

fn matmul_program(rows: u32, k: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input { dtype: DType::UINT8, shape: vec![Extent::Static(rows), Extent::Static(k)], name: None },
    );
    let activation = append(
        &mut program,
        Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(k), Extent::Static(1)], name: None },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("q8_0_flat_gate".into()),
        }),
    );
    (program, sum)
}

/// `sum_j d * code_j * act_j` in f64 for each row, reading the scale at `scale_byte` and the codes
/// from `code_byte`; the layout control passes a code offset one byte off the real one.
fn host_reference(weight: &[u8], activation: &[f32], k: usize, scale_byte: usize, code_byte: usize) -> Vec<f64> {
    let blocks = k / q8_0::QK8_0;
    weight
        .chunks_exact(blocks * q8_0::BLOCK_BYTES)
        .map(|row| {
            row.chunks_exact(q8_0::BLOCK_BYTES)
                .zip(activation.chunks_exact(q8_0::QK8_0))
                .map(|(block, acts)| {
                    let bits = u16::from_le_bytes([block[scale_byte], block[scale_byte + 1]]);
                    let scale = f64::from(half::f16::from_bits(bits).to_f32());
                    let codes = &block[code_byte..code_byte + q8_0::QK8_0];
                    codes.iter().zip(acts).map(|(&code, &act)| scale * f64::from(code as i8) * f64::from(act)).sum::<f64>()
                })
                .sum::<f64>()
        })
        .collect()
}

fn relative_error(got: &[f32], want: &[f64]) -> f64 {
    let magnitude = want.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
    got.iter().zip(want).fold(0.0f64, |acc, (&g, &w)| acc.max((f64::from(g) - w).abs())) / magnitude
}

/// Mean and max absolute error against the f64 reference, plus (when `Q8_CMP` names another run's
/// raw output) the largest difference between the two runs relative to the largest output.
fn error_summary(got: &[f32], want: &[f64]) -> (f64, f64) {
    let magnitude = want.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
    let sum: f64 = got.iter().zip(want).map(|(&g, &w)| (f64::from(g) - w).abs()).sum();
    (sum / got.len() as f64 / magnitude, magnitude)
}

fn main() -> Result<(), String> {
    let path = env_or("Q8_GGUF", "/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf");
    let mut file = File::open(&path).map_err(|error| error.to_string())?;
    let (parsed, file_len) = parse_header(&mut file)?;
    if std::env::var("Q8_LIST").is_ok() {
        for tensor in parsed.tensors.iter().filter(|tensor| tensor.ggml_type == GgmlType::Q8_0) {
            println!("{} dims={:?}", tensor.name, tensor.dims);
        }
        return Ok(());
    }
    let name = env_or("Q8_TENSOR", "blk.3.attn_q.weight");
    let tensor = parsed.tensors.iter().find(|tensor| tensor.name == name).ok_or("tensor not found")?;
    if tensor.ggml_type != GgmlType::Q8_0 {
        return Err(format!("{name} is {:?}, not Q8_0", tensor.ggml_type));
    }
    let (k, rows) = (tensor.dims[0] as usize, tensor.dims[1] as usize);
    let range = parsed.tensor_data_range(tensor, file_len).map_err(|error| format!("{error:?}"))?;
    let mut weight = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start)).map_err(|error| error.to_string())?;
    file.read_exact(&mut weight).map_err(|error| error.to_string())?;

    let mut lcg = Lcg(2026);
    let activation: Vec<f32> = (0..k).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();
    let (program, output) = matmul_program(rows as u32, k as u32);
    let blocks = [QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: &weight }, QuantizedBlock::Float32(&activation)];
    let plan = omega::plan(&program, &[], &blocks, &[output], if std::env::var("Q8_POLICY").as_deref() == Ok("bit_exact") { NumericPolicy::bit_exact() } else { NumericPolicy::llama_relaxed() }).map_err(|error| format!("{error:?}"))?;

    let (iters, warm, idle_ms): (usize, usize, u64) = (
        env_or("Q8_ITERS", "50").parse().map_err(|_| "Q8_ITERS")?,
        env_or("Q8_WARM", "3").parse().map_err(|_| "Q8_WARM")?,
        env_or("Q8_IDLE_MS", "0").parse().map_err(|_| "Q8_IDLE_MS")?,
    );
    for _ in 0..warm {
        omega::execute_plan_timed(&plan, &blocks).map_err(|error| format!("{error:?}"))?;
    }
    std::thread::sleep(Duration::from_millis(idle_ms));
    let mut gpu_ns = Vec::with_capacity(iters);
    let mut last = Vec::new();
    for _ in 0..iters {
        let (evaluated, ns) = omega::execute_plan_timed(&plan, &blocks).map_err(|error| format!("{error:?}"))?;
        gpu_ns.push(ns);
        last = evaluated.root().to_vec();
    }

    let checksum = last.iter().fold(FNV_OFFSET, |hash, value| {
        value.to_bits().to_le_bytes().iter().fold(hash, |inner, &byte| (inner ^ u64::from(byte)).wrapping_mul(FNV_PRIME))
    });
    if let Ok(out) = std::env::var("Q8_OUT") {
        let bytes: Vec<u8> = last.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect();
        File::create(out).and_then(|mut dump| dump.write_all(&bytes)).map_err(|error| error.to_string())?;
    }
    let reference = host_reference(&weight, &activation, k, 0, 2);
    let true_error = relative_error(&last, &reference);
    let (mean_error, magnitude) = error_summary(&last, &reference);
    let versus_other = std::env::var("Q8_CMP").ok().map(|other| {
        let raw = std::fs::read(other).map_err(|error| error.to_string())?;
        let theirs: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
        let differing = last.iter().zip(&theirs).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        let max_diff = last.iter().zip(&theirs).fold(0.0f64, |acc, (&a, &b)| acc.max((f64::from(a) - f64::from(b)).abs()));
        Ok::<_, String>((differing, max_diff / magnitude))
    });
    if let Some(compare) = versus_other {
        let (differing, relative) = compare?;
        println!("{{\"vs_other_differing_words\":{differing},\"vs_other_max_rel_diff\":{relative:e},\"words\":{}}}", last.len());
    }
    let control_error = relative_error(&last, &host_reference(&weight, &activation, k, 2, 2));
    println!(
        "{{\"tensor\":\"{name}\",\"rows\":{rows},\"k\":{k},\"fnv\":\"{checksum:016x}\",\"rel_err_f64\":{true_error:e},\"mean_rel_err_f64\":{mean_error:e},\"control_rel_err\":{control_error:e},\"gpu_ns\":{gpu_ns:?}}}"
    );
    Ok(())
}
