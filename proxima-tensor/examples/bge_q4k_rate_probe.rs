//! Probe: does `proxima_tensor::cpu::matmul_q4k_q8k_f32` (ROW 116's
//! 147.72-150.60 GMAC/s int8-dot kernel, measured at LLM decode GEMV
//! shapes) hold that rate at BGE-small's OWN weight-matrix shapes --
//! QKVO (384x384), FFN-up (384->1536), FFN-down (1536->384) -- at
//! M in {8, 128, 512} positions? Answers "can Q4_K beat AMX outright" at
//! the shapes that actually matter for the S=8 loss (6.2642 vs
//! onnxruntime's 4.7706 ms), independent of BGE fidelity (fidelity is
//! `bge_q4k_fidelity_probe.rs`, a separate probe in `proxima-onnx`; this
//! one runs unconditionally regardless of what that one found).
//!
//! Pipe question, in writing: throwaway measurement binary, no new type.
//! Wires two already-public `cpu` functions
//! (`quantize_row_q8k`/`matmul_q4k_q8k_f32`) plus `proxima_gguf::quant::q4_k`
//! at the call site, exactly the way `benches/bench_q4k_matmul.rs` already
//! does for a different shape set.
//!
//! **Structural finding, load-bearing for every number below:** `Q4_K`'s
//! super-block is `QK_K=256` elements; `matmul_q4k_q8k_f32`'s row
//! addressing (`row_bytes = weights.len() / rows`) requires each row's
//! reduction length `k` to be a whole `QK_K` multiple. BGE's hidden size
//! is 384 -- QKVO (`k=384`) and FFN-up (`k=384`) are NOT multiples of 256
//! and cannot be packed in the real per-row `Q4_K` layout without padding
//! `k` up to 512 (the next `QK_K` multiple, a 33.3% compute-padding
//! overhead paid on every call). FFN-down (`k=1536`) IS a clean multiple
//! (6 super-blocks/row) and needs no padding. Every QKVO/FFN-up cell below
//! is measured at the PADDED `k=512` shape and labeled as such; FFN-down
//! is native.
//!
//! Kernel-only, not in-path: this calls `matmul_q4k_q8k_f32` directly, M
//! independent single-vector calls per cell (the kernel's only public
//! multi-position-free entry point -- the internal `leading_total>1`/
//! `CohortSession` batched path is private, reached only from inside a
//! full forward-pass evaluation). There is no in-path number to report
//! alongside it: BGE's graph currently routes every `MatMul` through the
//! f32 width-tile/Accelerate path, never through a quantized kernel --
//! producing an in-path number would mean wiring a quantized BGE
//! execution path, which this task's own brief says not to build. ROW
//! 116's own kernel-only (147.72-150.60) vs in-path (39.54, a 3.74x
//! orchestration tax) gap is the reason to read every number below as an
//! upper bound, not a deployable rate.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use proxima_gguf::quant::q4_k;
use proxima_tensor::cpu::matmul_q4k_q8k_f32;
use proxima_tensor::test_support::Lcg;

const REPEATS: usize = 7;
const BYTES_PER_WEIGHT: f64 = q4_k::BLOCK_BYTES as f64 / q4_k::QK_K as f64;

struct Shape {
    label: &'static str,
    rows: usize,
    k_true: usize,
    k_padded: usize,
}

fn shapes() -> [Shape; 3] {
    [
        Shape {
            label: "QKVO (384x384, k padded 384->512)",
            rows: 384,
            k_true: 384,
            k_padded: 512,
        },
        Shape {
            label: "FFN-up (384->1536, k padded 384->512)",
            rows: 1536,
            k_true: 384,
            k_padded: 512,
        },
        Shape {
            label: "FFN-down (1536->384, k native 1536)",
            rows: 384,
            k_true: 1536,
            k_padded: 1536,
        },
    ]
}

fn packed_q4k_weights(rows: usize, k_padded: usize, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let element_count = rows * k_padded;
    assert!(
        element_count.is_multiple_of(q4_k::QK_K),
        "N==0 guard: rows*k_padded must be a QK_K multiple, got {element_count}"
    );
    let weights: Vec<f32> = (0..element_count).map(|_| lcg.next_unit()).collect();
    let block_count = element_count / q4_k::QK_K;
    let mut packed = vec![0u8; q4_k::bytes_for_blocks(block_count)];
    q4_k::quantize(&weights, &mut packed).expect("quantize synthetic BGE-shaped weights");
    packed
}

fn synthetic_activation(k_padded: usize, seed: u64) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..k_padded).map(|_| lcg.next_unit()).collect()
}

fn mean(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len() as f64
}

fn coefficient_of_variation(samples: &[f64], mean_value: f64) -> f64 {
    if samples.len() < 2 || mean_value == 0.0 {
        return 0.0;
    }
    let variance = samples
        .iter()
        .map(|&value| (value - mean_value).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    variance.sqrt() / mean_value * 100.0
}

fn main() {
    println!("=== Q4_K int8-dot kernel-only rate at BGE's own weight shapes ===");
    println!(
        "BLOCK_BYTES={} QK_K={} bytes/weight={BYTES_PER_WEIGHT:.4}",
        q4_k::BLOCK_BYTES,
        q4_k::QK_K
    );

    for shape in shapes() {
        println!(
            "\n--- {} rows={} k_true={} k_padded={} padding_overhead={:.1}% ---",
            shape.label,
            shape.rows,
            shape.k_true,
            shape.k_padded,
            (shape.k_padded as f64 / shape.k_true as f64 - 1.0) * 100.0
        );
        let weights = packed_q4k_weights(shape.rows, shape.k_padded, 0xBBEE_0001);
        let row_bytes = weights.len() / shape.rows;
        assert!(
            row_bytes.is_multiple_of(q4_k::BLOCK_BYTES),
            "N==0 guard: row_bytes must be a whole block-count multiple"
        );

        for &positions in &[8usize, 128, 512] {
            let activations: Vec<Vec<f32>> = (0..positions)
                .map(|index| synthetic_activation(shape.k_padded, 0xACE0_0000 + index as u64))
                .collect();
            assert!(
                !activations.is_empty(),
                "N==0: zero activation positions — RED"
            );

            // one untimed warm-up pass over the real activations, discarded,
            // before the timed reps below
            for activation in &activations {
                let warm = matmul_q4k_q8k_f32(&weights, shape.rows, activation)
                    .expect("warm-up matmul_q4k_q8k_f32 on BGE-shaped weights");
                std::hint::black_box(&warm);
            }

            let mut wall_seconds = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let started = Instant::now();
                for activation in &activations {
                    let result = matmul_q4k_q8k_f32(&weights, shape.rows, activation)
                        .expect("matmul_q4k_q8k_f32 on BGE-shaped weights");
                    std::hint::black_box(&result);
                }
                wall_seconds.push(started.elapsed().as_secs_f64());
            }

            let mean_seconds = mean(&wall_seconds);
            let cov = coefficient_of_variation(&wall_seconds, mean_seconds);
            let total_macs = shape.rows as f64 * shape.k_padded as f64 * positions as f64;
            let total_bytes = shape.rows as f64 * row_bytes as f64 * positions as f64;
            let gmac_s = total_macs / mean_seconds / 1e9;
            let gb_s = total_bytes / mean_seconds / 1e9;
            let ns_per_call = mean_seconds * 1e9 / positions as f64;
            let true_bge_equivalent_gmac_s = gmac_s * (shape.k_true as f64 / shape.k_padded as f64);

            println!(
                "  M={positions:>3}: mean={mean_seconds:.6}s ns/call={ns_per_call:.1} GMAC/s={gmac_s:.2} (true-k-equivalent={true_bge_equivalent_gmac_s:.2}) GB/s={gb_s:.2} CoV%={cov:.2}{}",
                if cov > 5.0 { " [RANGE: CoV>5%]" } else { "" }
            );
        }
    }
}
