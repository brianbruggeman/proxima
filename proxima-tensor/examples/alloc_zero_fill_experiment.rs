#![allow(clippy::expect_used)]
//! Side experiment, NOT on the production path: at 1024x1024 f32 (4 MB), is
//! `vec![0.0f32; n]`'s cost dominated by the allocator or by the zero-fill?
//! Each measurement repeats the same size in a loop so the allocator's
//! free-list/size-class caching behaves the way it does inside
//! `evaluate_node_parallel`'s real per-node loop (repeated same-size
//! alloc/dealloc across many `evaluate_parallel` calls), not a cold
//! first-touch mmap.
//!
//! Three costs, isolated:
//! 1. `Vec::with_capacity(n)` alone — allocator only, memory left
//!    uninitialized.
//! 2. `Vec::with_capacity(n)` plus an explicit manual zero-fill loop —
//!    allocator cost plus a memset this code controls, to see the fill cost
//!    on its own.
//! 3. `vec![0.0f32; n]` — the actual production expression (`cpu.rs`'s
//!    `node_output_len` allocation site), whatever path the standard
//!    library picks for a zeroed fill.
//!
//! Not part of the crate's public surface; throwaway for this measurement.

use std::hint::black_box;
use std::time::{Duration, Instant};

const ELEMENT_COUNT: usize = 1024 * 1024;
const ITERATIONS: usize = 200;
const WARMUP: usize = 20;

fn mean_and_cov(samples: &[Duration]) -> (f64, f64) {
    let nanos: Vec<f64> = samples
        .iter()
        .map(|sample| sample.as_nanos() as f64)
        .collect();
    let mean = nanos.iter().sum::<f64>() / nanos.len() as f64;
    let variance = nanos
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / nanos.len() as f64;
    let cov_percent = if mean == 0.0 {
        0.0
    } else {
        variance.sqrt() / mean * 100.0
    };
    (mean, cov_percent)
}

fn time_with_capacity_only(iterations: usize) -> Vec<Duration> {
    (0..iterations)
        .map(|_| {
            let start = Instant::now();
            let buffer: Vec<f32> = Vec::with_capacity(ELEMENT_COUNT);
            let elapsed = start.elapsed();
            black_box(&buffer);
            elapsed
        })
        .collect()
}

fn time_with_capacity_then_manual_fill(iterations: usize) -> Vec<Duration> {
    (0..iterations)
        .map(|_| {
            let start = Instant::now();
            // `resize` fills every new slot before extending `len`, so
            // this is the manual-fill cost without `set_len` over
            // uninitialized memory: capacity is already `ELEMENT_COUNT`
            // from `with_capacity`, so this never reallocates, only fills.
            let mut buffer: Vec<f32> = Vec::with_capacity(ELEMENT_COUNT);
            buffer.resize(ELEMENT_COUNT, 0.0);
            let elapsed = start.elapsed();
            black_box(&buffer);
            elapsed
        })
        .collect()
}

fn time_vec_zeroed_macro(iterations: usize) -> Vec<Duration> {
    (0..iterations)
        .map(|_| {
            let start = Instant::now();
            let buffer = vec![0.0f32; ELEMENT_COUNT];
            let elapsed = start.elapsed();
            black_box(&buffer);
            elapsed
        })
        .collect()
}

fn report(label: &str, samples: &[Duration]) {
    let (mean_ns, cov_percent) = mean_and_cov(samples);
    println!(
        "{label}: mean_ns={mean_ns:.1} cov_percent={cov_percent:.2} n={}",
        samples.len()
    );
}

fn main() {
    let _ = time_with_capacity_only(WARMUP);
    let _ = time_with_capacity_then_manual_fill(WARMUP);
    let _ = time_vec_zeroed_macro(WARMUP);

    let with_capacity_only = time_with_capacity_only(ITERATIONS);
    let with_capacity_manual_fill = time_with_capacity_then_manual_fill(ITERATIONS);
    let vec_zeroed_macro = time_vec_zeroed_macro(ITERATIONS);

    report("with_capacity_only", &with_capacity_only);
    report("with_capacity_then_manual_fill", &with_capacity_manual_fill);
    report("vec_zeroed_macro", &vec_zeroed_macro);
}
