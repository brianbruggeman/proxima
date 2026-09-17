// scratch diagnostic: per-channel dump of gemma4 norm weight tensors to
// resolve the output_norm RMS/max anomaly. Not library surface.
#![allow(clippy::expect_used)]

use std::env;
use std::fs::File;

fn dequantize_tensor(
    file_bytes: &[u8],
    data_offset: u64,
    tensor: &proxima_gguf::tensor::TensorInfo,
) -> Vec<f32> {
    let element_count = tensor.element_count() as usize;
    let nbytes = tensor.nbytes().expect("tensor byte footprint") as usize;
    let start = (data_offset + tensor.offset) as usize;
    let raw = &file_bytes[start..start + nbytes];

    if tensor.ggml_type == proxima_gguf::types::GgmlType::F32 {
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect()
    } else {
        let mut output = vec![0.0_f32; element_count];
        proxima_gguf::quant::dispatch::dequantize(tensor.ggml_type, raw, &mut output)
            .expect("dequantize tensor");
        output
    }
}

fn report_channel_stats(name: &str, values: &[f32]) {
    let count = values.len();
    let sum_sq: f64 = values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum();
    let rms = (sum_sq / count as f64).sqrt();
    let max_abs = values
        .iter()
        .fold(0.0_f32, |acc, value| acc.max(value.abs()));

    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|left, right| {
        right
            .1
            .abs()
            .total_cmp(&left.1.abs())
            .then(left.0.cmp(&right.0))
    });

    let top20: Vec<(usize, f32)> = indexed.iter().take(20).copied().collect();
    let top20_sum_sq: f64 = top20
        .iter()
        .map(|(_, value)| f64::from(*value) * f64::from(*value))
        .sum();
    let concentration_pct = 100.0 * top20_sum_sq / sum_sq;

    println!("=== {name} ===");
    println!("count={count} rms={rms:.4} max_abs={max_abs:.4}");
    println!("top20 energy concentration: {concentration_pct:.2}% of sum-of-squares");
    println!("top20 (index, value):");
    for (index, value) in &top20 {
        println!("  [{index}] = {value:.4}");
    }

    let mut sorted_abs: Vec<f32> = values.iter().map(|value| value.abs()).collect();
    sorted_abs.sort_by(f32::total_cmp);
    let percentile = |fraction: f64| -> f32 {
        let index = ((count as f64 - 1.0) * fraction).round() as usize;
        sorted_abs[index]
    };
    println!(
        "abs percentiles: p50={:.4} p90={:.4} p99={:.4} p99.9={:.4} max={:.4}",
        percentile(0.50),
        percentile(0.90),
        percentile(0.99),
        percentile(0.999),
        percentile(1.0),
    );
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129".to_string());
    let file = File::open(&path).expect("open gemma4 gguf");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap gemma4 gguf");
    let file_bytes: &[u8] = &mapping;
    let parsed = proxima_gguf::parse_complete(file_bytes).expect("parse gemma4 header");

    let data_offset = parsed.data_offset;

    for name in [
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.29.post_ffw_norm.weight",
    ] {
        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("tensor {name} present"));
        let values = dequantize_tensor(file_bytes, data_offset, tensor);
        report_channel_stats(name, &values);
    }
}
