//! Multi-row quantized-activation matmuls on real weights sliced from a GGUF
//! checkpoint (`PROXIMA_BENCH_GGUF_PATH`): `matmul_q8_0_activation_f32` on Q5_0 /
//! Q8_0 tensors and the Q4_K / Q6_K wide folds on `ffn_down`, at
//! `PROXIMA_BENCH_POSITIONS` activation rows (default 512). Prints min-of-N wall
//! time, GMAC/s, and an FNV hash of the output bits so two builds can be
//! compared for bit-identity.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::Instant;

use proxima_gguf::pipe::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_primitives::Codec;
use proxima_tensor::cpu::{
    matmul_q4k_q8k_f32_wide, matmul_q6k_q8k_f32_wide, matmul_q8_0_activation_f32,
};
use proxima_tensor::test_support::Lcg;

fn output_hash(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf29ce484222325u64, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3)
    })
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn run(ggml_type: GgmlType, weights: &[u8], rows: usize, activation: &[f32], positions: usize) -> Vec<f32> {
    match ggml_type {
        GgmlType::Q5_0 => matmul_q8_0_activation_f32(Codec::Q5_0, weights, rows, activation, positions),
        GgmlType::Q8_0 => matmul_q8_0_activation_f32(Codec::Q8_0, weights, rows, activation, positions),
        GgmlType::Q4_K => matmul_q4k_q8k_f32_wide(weights, rows, activation, positions),
        GgmlType::Q6_K => matmul_q6k_q8k_f32_wide(weights, rows, activation, positions),
        other => panic!("bench has no kernel for {other:?}"),
    }
    .expect("matmul")
}

fn main() {
    let path = std::env::var("PROXIMA_BENCH_GGUF_PATH").expect("PROXIMA_BENCH_GGUF_PATH names a qwen2 gguf");
    let positions = env_usize("PROXIMA_BENCH_POSITIONS", 512);
    let iterations = env_usize("PROXIMA_BENCH_ITERATIONS", 7);
    let bytes = std::fs::read(&path).expect("read gguf");
    let parsed = parse_complete(&bytes).expect("parse gguf");
    let file_len = bytes.len() as u64;
    let down_of = |wanted: GgmlType| {
        parsed
            .tensors
            .iter()
            .find(|tensor| tensor.name.contains("ffn_down") && tensor.ggml_type == wanted)
            .map(|tensor| tensor.name.clone())
            .expect("a ffn_down tensor of the wanted type")
    };
    let names = [
        "blk.0.ffn_gate.weight".to_string(),
        "blk.0.attn_q.weight".to_string(),
        "blk.0.attn_v.weight".to_string(),
        down_of(GgmlType::Q4_K),
        down_of(GgmlType::Q6_K),
    ];
    for name in names {
        let tensor = parsed.tensors.iter().find(|tensor| tensor.name == name).expect("tensor present");
        let in_dim = tensor.dims[0] as usize;
        let out_dim = tensor.dims[1] as usize;
        let range = parsed.tensor_data_range(tensor, file_len).expect("tensor range");
        let weights = &bytes[range.start as usize..range.end as usize];
        let mut lcg = Lcg(0x5eed);
        let activation: Vec<f32> = (0..in_dim * positions).map(|_| lcg.next_unit() * 0.5).collect();
        let mut best = f64::MAX;
        let mut hash = 0;
        for _ in 0..iterations {
            let started = Instant::now();
            let output = run(tensor.ggml_type, black_box(weights), out_dim, black_box(&activation), positions);
            best = best.min(started.elapsed().as_secs_f64());
            hash = output_hash(&output);
        }
        let macs = (out_dim * in_dim * positions) as f64;
        println!(
            "kernel name={name} type={:?} rows={out_dim} k={in_dim} positions={positions} min_ms={:.3} gmac_s={:.2} gflop_s={:.2} hash={hash:016x}",
            tensor.ggml_type,
            best * 1e3,
            macs / best / 1e9,
            2.0 * macs / best / 1e9,
        );
    }
}
