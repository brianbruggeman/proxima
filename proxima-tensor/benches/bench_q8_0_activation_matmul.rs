//! `proxima_tensor::cpu::matmul_q8_0_activation_f32` on real Q5_0 / Q8_0 weights
//! sliced from a GGUF checkpoint (`PROXIMA_BENCH_GGUF_PATH`), multi-row
//! activations (`PROXIMA_BENCH_POSITIONS`, default 512). Prints min-of-N wall
//! time, GMAC/s, and an FNV hash of the output bits so two builds can be
//! compared for bit-identity.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::Instant;

use proxima_gguf::pipe::parse_complete;
use proxima_primitives::Codec;
use proxima_tensor::cpu::matmul_q8_0_activation_f32;
use proxima_tensor::test_support::Lcg;

fn output_hash(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf29ce484222325u64, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3)
    })
}

fn main() {
    let path = std::env::var("PROXIMA_BENCH_GGUF_PATH").expect("PROXIMA_BENCH_GGUF_PATH names a qwen2 gguf");
    let positions: usize = std::env::var("PROXIMA_BENCH_POSITIONS").ok().and_then(|value| value.parse().ok()).unwrap_or(512);
    let iterations: usize = std::env::var("PROXIMA_BENCH_ITERATIONS").ok().and_then(|value| value.parse().ok()).unwrap_or(7);
    let bytes = std::fs::read(&path).expect("read gguf");
    let parsed = parse_complete(&bytes).expect("parse gguf");
    let file_len = bytes.len() as u64;
    for (name, codec) in [
        ("blk.0.ffn_gate.weight", Codec::Q5_0),
        ("blk.0.attn_q.weight", Codec::Q5_0),
        ("blk.0.attn_v.weight", Codec::Q8_0),
    ] {
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
            let output = matmul_q8_0_activation_f32(codec, black_box(weights), out_dim, black_box(&activation), positions).expect("matmul");
            best = best.min(started.elapsed().as_secs_f64());
            hash = output_hash(&output);
        }
        let macs = (out_dim * in_dim * positions) as f64;
        println!(
            "kernel name={name} codec={codec:?} rows={out_dim} k={in_dim} positions={positions} min_ms={:.3} gmac_s={:.2} gflop_s={:.2} hash={hash:016x}",
            best * 1e3,
            macs / best / 1e9,
            2.0 * macs / best / 1e9,
        );
    }
}
