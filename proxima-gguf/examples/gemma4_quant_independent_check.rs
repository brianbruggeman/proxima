//! Non-circular parity check for Q3_K and Q5_1 dequantization against an
//! independent inline transcription of ggml's `dequantize_row_q3_K` /
//! `dequantize_row_q5_1`, run directly on raw bytes read from a real gguf
//! checkpoint. Does not go through the model loader and does not call
//! `proxima_gguf::quant` on the independent side.
//!
//! Usage: `GEMMA4_GGUF=<path> cargo run --example gemma4_quant_independent_check`

use std::env;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use proxima_gguf::pipe::parse_complete;
use proxima_gguf::quant::dispatch::dequantize as proxima_dequantize;
use proxima_gguf::tensor::TensorInfo;
use proxima_gguf::types::GgmlType;

const Q3_K_BLOCK_BYTES: usize = 110;
const Q3_K_BLOCK_ELEMENTS: usize = 256;
const Q5_1_BLOCK_BYTES: usize = 24;
const Q5_1_BLOCK_ELEMENTS: usize = 32;

fn parse_header(file: &mut std::fs::File) -> proxima_gguf::pipe::ParsedGguf {
    let mut header_buf = Vec::new();
    for cap in [4usize << 20, 16 << 20, 64 << 20, 128 << 20, 512 << 20] {
        header_buf.resize(cap, 0);
        file.seek(SeekFrom::Start(0)).expect("seek to file start");
        let read = file.read(&mut header_buf).expect("read gguf header region");
        header_buf.truncate(read);
        if let Ok(parsed) = parse_complete(&header_buf) {
            return parsed;
        }
    }
    panic!("gguf metadata region did not fit in 512 MiB");
}

fn read_range(file: &mut std::fs::File, range: core::ops::Range<u64>) -> Vec<u8> {
    let mut buffer = vec![0u8; (range.end - range.start) as usize];
    file.seek(SeekFrom::Start(range.start))
        .expect("seek to tensor data range start");
    file.read_exact(&mut buffer)
        .expect("read exact tensor data range");
    buffer
}

fn find_tensor<'parsed>(
    parsed: &'parsed proxima_gguf::pipe::ParsedGguf,
    name: &str,
) -> &'parsed TensorInfo {
    parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("{name} not present in checkpoint"))
}

/// Independent inline transcription of ggml's `dequantize_row_q3_K`
/// (`ggml-quants.c:1050-1098`). Does not call any `proxima_gguf::quant` code.
/// `block` is exactly 110 bytes (`hmask[32]`, `qs[64]`, `scales[12]`, `d`
/// f16 trailing); `output` is exactly 256 f32s.
fn independent_dequantize_q3_k_block(block: &[u8], output: &mut [f32; Q3_K_BLOCK_ELEMENTS]) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;

    let hmask = &block[0..32];
    let qs = &block[32..32 + 64];
    let scales_raw = &block[96..96 + 12];
    let d_bytes: [u8; 2] = [block[108], block[109]];
    let d_all = half::f16::from_le_bytes(d_bytes).to_f32();

    let mut aux = [0u32; 4];
    for (word, chunk) in aux.iter_mut().zip(scales_raw.chunks_exact(4)) {
        *word = u32::from_le_bytes(chunk.try_into().unwrap());
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scale_bytes = [0i8; 16];
    for (chunk_index, word) in aux.iter().enumerate() {
        let bytes = word.to_le_bytes();
        for (local, &byte) in bytes.iter().enumerate() {
            scale_bytes[chunk_index * 4 + local] = byte as i8;
        }
    }

    let mut mask: u8 = 1;
    let mut is_bit: usize = 0;
    let mut out_offset = 0usize;
    for chunk in 0..(Q3_K_BLOCK_ELEMENTS / 128) {
        let q = &qs[chunk * 32..chunk * 32 + 32];
        let mut shift: u32 = 0;
        for _ in 0..4 {
            let dl = d_all * (scale_bytes[is_bit] - 32) as f32;
            is_bit += 1;
            for local in 0..16 {
                let raw = ((q[local] >> shift) & 3) as i32;
                let bit_set = (hmask[local] & mask) != 0;
                let q3 = if bit_set { raw } else { raw - 4 };
                output[out_offset] = dl * q3 as f32;
                out_offset += 1;
            }

            let dh = d_all * (scale_bytes[is_bit] - 32) as f32;
            is_bit += 1;
            for local in 0..16 {
                let index = local + 16;
                let raw = ((q[index] >> shift) & 3) as i32;
                let bit_set = (hmask[index] & mask) != 0;
                let q3 = if bit_set { raw } else { raw - 4 };
                output[out_offset] = dh * q3 as f32;
                out_offset += 1;
            }

            shift += 2;
            mask <<= 1;
        }
    }
}

/// Independent inline transcription of ggml's `dequantize_row_q5_1`
/// (`ggml-quants.c:316-341`). Does not call any `proxima_gguf::quant` code.
/// `block` is exactly 24 bytes (`d` f16, `m` f16, `qh[4]`, `qs[16]`);
/// `output` is exactly 32 f32s.
fn independent_dequantize_q5_1_block(block: &[u8], output: &mut [f32; Q5_1_BLOCK_ELEMENTS]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let qs = &block[8..8 + 16];

    for j in 0..16 {
        let xh_0 = ((qh >> j) << 4) & 0x10;
        let xh_1 = (qh >> (j + 12)) & 0x10;
        let x0 = (u32::from(qs[j] & 0x0F) | xh_0) as f32;
        let x1 = (u32::from(qs[j] >> 4) | xh_1) as f32;
        output[j] = x0 * d + m;
        output[16 + j] = x1 * d + m;
    }
}

fn compare_q3_k(
    file: &mut std::fs::File,
    parsed: &proxima_gguf::pipe::ParsedGguf,
    file_len: u64,
    tensor_name: &str,
) {
    let tensor = find_tensor(parsed, tensor_name);
    assert_eq!(tensor.ggml_type, GgmlType::Q3_K, "{tensor_name} must be Q3_K");
    let full_range = parsed
        .tensor_data_range(tensor, file_len)
        .expect("tensor data range within checkpoint");
    let block_range = full_range.start..full_range.start + Q3_K_BLOCK_BYTES as u64;
    let raw = read_range(file, block_range);

    let mut proxima_out = [0.0f32; Q3_K_BLOCK_ELEMENTS];
    proxima_dequantize(GgmlType::Q3_K, &raw, &mut proxima_out).expect("proxima q3_k decode");

    let mut independent_out = [0.0f32; Q3_K_BLOCK_ELEMENTS];
    independent_dequantize_q3_k_block(&raw, &mut independent_out);

    let mut max_abs_diff = 0.0f32;
    let mut first_diffs: Vec<(usize, f32, f32)> = Vec::new();
    for (index, (&proxima_value, &independent_value)) in
        proxima_out.iter().zip(independent_out.iter()).enumerate()
    {
        let diff = (proxima_value - independent_value).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        if diff != 0.0 && first_diffs.len() < 5 {
            first_diffs.push((index, proxima_value, independent_value));
        }
    }
    println!(
        "Q3_K tensor={tensor_name} block=0 max_abs_diff={max_abs_diff:e} first_five_nonzero_diffs={first_diffs:?}"
    );
    println!("  raw_d_bytes={:02x?} first_raw_bytes={:02x?}", &raw[108..110], &raw[0..8]);
    println!("  proxima[0..8]={:?}", &proxima_out[0..8]);
    println!("  independent[0..8]={:?}", &independent_out[0..8]);
}

fn compare_q5_1(
    file: &mut std::fs::File,
    parsed: &proxima_gguf::pipe::ParsedGguf,
    file_len: u64,
    tensor_name: &str,
    block_index: u64,
) {
    let tensor = find_tensor(parsed, tensor_name);
    assert_eq!(tensor.ggml_type, GgmlType::Q5_1, "{tensor_name} must be Q5_1");
    let full_range = parsed
        .tensor_data_range(tensor, file_len)
        .expect("tensor data range within checkpoint");
    let block_start = full_range.start + block_index * Q5_1_BLOCK_BYTES as u64;
    let block_range = block_start..block_start + Q5_1_BLOCK_BYTES as u64;
    let raw = read_range(file, block_range);

    let mut proxima_out = [0.0f32; Q5_1_BLOCK_ELEMENTS];
    proxima_dequantize(GgmlType::Q5_1, &raw, &mut proxima_out).expect("proxima q5_1 decode");

    let mut independent_out = [0.0f32; Q5_1_BLOCK_ELEMENTS];
    independent_dequantize_q5_1_block(&raw, &mut independent_out);

    let mut max_abs_diff = 0.0f32;
    let mut first_diffs: Vec<(usize, f32, f32)> = Vec::new();
    for (index, (&proxima_value, &independent_value)) in
        proxima_out.iter().zip(independent_out.iter()).enumerate()
    {
        let diff = (proxima_value - independent_value).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        if diff != 0.0 && first_diffs.len() < 5 {
            first_diffs.push((index, proxima_value, independent_value));
        }
    }
    println!(
        "Q5_1 tensor={tensor_name} block={block_index} max_abs_diff={max_abs_diff:e} first_five_nonzero_diffs={first_diffs:?}"
    );
    println!("  d_bytes={:02x?} m_bytes={:02x?}", &raw[0..2], &raw[2..4]);
    println!("  proxima={proxima_out:?}");
    println!("  independent={independent_out:?}");
}

fn main() {
    let gguf_path = env::var("GEMMA4_GGUF").unwrap_or_else(|_| {
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129"
            .to_string()
    });
    let gguf_path = Path::new(&gguf_path);
    assert!(gguf_path.exists(), "gguf path {gguf_path:?} does not exist");

    let mut file = std::fs::File::open(gguf_path).expect("open gguf checkpoint");
    let file_len = file.metadata().expect("stat gguf checkpoint").len();
    let parsed = parse_header(&mut file);

    println!(
        "parsed {} tensors, data_offset={}",
        parsed.tensors.len(),
        parsed.data_offset
    );

    compare_q3_k(&mut file, &parsed, file_len, "blk.0.attn_q.weight");
    compare_q3_k(&mut file, &parsed, file_len, "blk.0.attn_k.weight");
    compare_q5_1(&mut file, &parsed, file_len, "blk.0.ffn_down_exps.weight", 0);
    compare_q5_1(&mut file, &parsed, file_len, "blk.0.ffn_down_exps.weight", 1);
}
