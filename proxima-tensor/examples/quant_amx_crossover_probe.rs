#![allow(clippy::expect_used)]
//! Three-arm crossover probe: at what `M` (prefill width) does
//! dequantize-a-tile-then-`cblas_sgemm` beat our native `Q4_K` int8-dot NEON
//! kernel?
//!
//! - **arm A**: [`proxima_tensor::cpu::matmul_q4k_q8k_f32_wide`] -- the same
//!   wide fold `run_reduce_quantized` dispatches to for real prefill
//!   (`cpu.rs:7093`), not a `leading_total`-times loop over the batch-1 entry
//!   point.
//! - **arm B**: [`proxima_gguf::quant::q4_k::dequantize`] into a staging
//!   buffer, THEN `cblas_sgemm` -- dequant cost paid inside the timed region
//!   every call, the whole point of this arm.
//! - **arm C** (control): `cblas_sgemm` on weights already dequantized once,
//!   outside the timed region -- the ceiling arm B is chasing.
//!
//! Real packed `Q4_K` bytes come straight off a real GGUF checkpoint on
//! disk, same convention as `benches/bench_q4k_matmul.rs`. `cblas_sgemm` is
//! declared as a local `extern` block, same reasoning `cpu.rs` itself gives
//! (`cpu.rs:11052-11059`): the only symbol needed is `cblas_sgemm`, Accelerate
//! ships in every macOS SDK, no new crate dependency.
//!
//! Accelerate/`cblas_sgemm` is macOS/aarch64-only, so the whole probe body
//! lives in [`mac`] and `main` falls back to a plain message elsewhere.

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod mac {
    use std::fs::File;
    use std::hint::black_box;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::Path;
    use std::time::Instant;

    use proxima_gguf::pipe::ParsedGguf;
    use proxima_gguf::quant::q4_k;
    use proxima_gguf::tensor::TensorInfo;
    use proxima_gguf::types::GgmlType;
    use proxima_tensor::cpu::matmul_q4k_q8k_f32_wide;
    use proxima_tensor::test_support::Lcg;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;

    const REPEATS: usize = 7;
    const M_VALUES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];
    const Q4K_TOLERANCE: f32 = 0.5; // matches bench_q4k_matmul.rs's own tolerance
    const FLOAT_TOLERANCE: f32 = 0.1; // sgemm accumulation order vs scalar reference

    pub fn gguf_path() -> String {
        std::env::var("PROXIMA_BENCH_GGUF_PATH").unwrap_or_else(|_| {
            "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf"
                .to_string()
        })
    }

    fn parse_header(path: &Path) -> (ParsedGguf, u64) {
        let mut file = File::open(path).expect("open real gguf file");
        let file_len = file.metadata().expect("stat gguf file").len();
        let mut prefix_len = 1usize << 20;
        loop {
            let mut buf = vec![0u8; prefix_len];
            file.seek(SeekFrom::Start(0)).expect("seek to start");
            let read = file.read(&mut buf).expect("read gguf prefix");
            buf.truncate(read);
            match proxima_gguf::pipe::parse_complete(&buf) {
                Ok(parsed) => return (parsed, file_len),
                Err(_) => {
                    assert!(
                        prefix_len < (1 << 26),
                        "gguf header/directory exceeded 64 MiB prefix budget"
                    );
                    prefix_len *= 2;
                }
            }
        }
    }

    fn find_tensor<'a>(parsed: &'a ParsedGguf, name: &str) -> &'a TensorInfo {
        parsed
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("tensor {name} not found in real gguf file"))
    }

    fn read_tensor_bytes(
        file: &mut File,
        parsed: &ParsedGguf,
        tensor: &TensorInfo,
        file_len: u64,
    ) -> Vec<u8> {
        let range = parsed
            .tensor_data_range(tensor, file_len)
            .expect("tensor byte range within file bounds");
        let mut buf = vec![0u8; (range.end - range.start) as usize];
        file.seek(SeekFrom::Start(range.start))
            .expect("seek to tensor data");
        file.read_exact(&mut buf).expect("read exact tensor bytes");
        buf
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "shape mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    struct Stats {
        mean_ns: f64,
        cov_pct: f64,
    }

    fn stats(samples_ns: &[f64]) -> Stats {
        let n = samples_ns.len() as f64;
        let mean = samples_ns.iter().sum::<f64>() / n;
        let variance = samples_ns
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / n;
        let stddev = variance.sqrt();
        Stats {
            mean_ns: mean,
            cov_pct: if mean > 0.0 {
                100.0 * stddev / mean
            } else {
                0.0
            },
        }
    }

    /// One weight row is `k` elements = `k / q4_k::QK_K` blocks of
    /// `q4_k::BLOCK_BYTES` each.
    fn row_bytes(k: usize) -> usize {
        (k / q4_k::QK_K) * q4_k::BLOCK_BYTES
    }

    #[allow(clippy::too_many_arguments)]
    fn sgemm_rows_by_m(
        weight_f32: &[f32],
        activation: &[f32],
        output: &mut [f32],
        rows: usize,
        k: usize,
        m: usize,
    ) {
        let rows_i32 = i32::try_from(rows).expect("rows fits i32");
        let m_i32 = i32::try_from(m).expect("m fits i32");
        let k_i32 = i32::try_from(k).expect("k fits i32");
        // SAFETY: `weight_f32` is `rows*k` f32 (row-major, lda=k), `activation`
        // is `m*k` f32 (row-major, ldb=k, transposed by CBLAS_TRANS), `output`
        // is caller-sized to `rows*m` f32 (row-major, ldc=m) -- all three
        // lengths checked by the caller before this function is reached.
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                rows_i32,
                m_i32,
                k_i32,
                1.0,
                weight_f32.as_ptr(),
                k_i32,
                activation.as_ptr(),
                k_i32,
                0.0,
                output.as_mut_ptr(),
                m_i32,
            );
        }
    }

    struct Shape {
        label: &'static str,
        tensor_name: &'static str,
        seed: u64,
    }

    const SHAPES: &[Shape] = &[
        Shape {
            label: "attn_q_4096x4096",
            tensor_name: "blk.0.attn_q.weight",
            seed: 200,
        },
        Shape {
            label: "ffn_gate_4096x14336",
            tensor_name: "blk.0.ffn_gate.weight",
            seed: 201,
        },
    ];

    fn probe_shape(file: &mut File, parsed: &ParsedGguf, file_len: u64, shape: &Shape) {
        let tensor = find_tensor(parsed, shape.tensor_name);
        if tensor.ggml_type != GgmlType::Q4_K {
            println!(
                "{} BLOCKED: real file stores this tensor as {:?}, not Q4_K -- skipped",
                shape.label, tensor.ggml_type
            );
            return;
        }
        let k = tensor.dims[0] as usize;
        let rows = tensor.dims[1] as usize;
        let weight_bytes = read_tensor_bytes(file, parsed, tensor, file_len);
        assert_eq!(
            weight_bytes.len(),
            rows * row_bytes(k),
            "{}: packed byte length mismatch",
            shape.label
        );

        // one-time dequant for arm C's ceiling -- NOT timed, this is the
        // "weights already f32" precondition arm C measures against.
        let mut f32_weights_once = vec![0f32; rows * k];
        q4_k::dequantize(&weight_bytes, &mut f32_weights_once)
            .expect("dequantize real q4_k weight matrix once");

        // per-call staging buffer for arm B, reused (overwritten) every call --
        // realistic per-tile staging, not re-allocated each iteration.
        let mut f32_scratch = vec![0f32; rows * k];

        println!(
            "\n=== {} k={k} rows={rows} macs_per_row_per_position={k} packed_bytes={} ===",
            shape.label,
            weight_bytes.len()
        );

        for &m in M_VALUES {
            let mut lcg = Lcg(shape.seed.wrapping_add(m as u64));
            let activation: Vec<f32> = (0..m * k).map(|_| lcg.next_unit() * 0.5).collect();
            let macs = (rows as u64) * (k as u64) * (m as u64);

            // independent scalar reference, position 0 only -- bounded cost
            // regardless of m, built from the same dequantized weights every
            // arm ultimately traces back to (dequantize's own fidelity against
            // ggml's decoder is separately proven by
            // `examples/q4k_ggml_fidelity.rs`).
            let mut reference = vec![0f32; rows];
            for (row, slot) in reference.iter_mut().enumerate() {
                let weight_row = &f32_weights_once[row * k..(row + 1) * k];
                let activation_row = &activation[0..k];
                *slot = weight_row
                    .iter()
                    .zip(activation_row)
                    .map(|(w, a)| w * a)
                    .sum();
            }

            // --- arm A: native Q4_K int8-dot wide kernel ---
            let arm_a_first = matmul_q4k_q8k_f32_wide(&weight_bytes, rows, &activation, m)
                .expect("arm a computes");
            let arm_a_pos0: Vec<f32> = (0..rows).map(|row| arm_a_first[row * m]).collect();
            let arm_a_diff = max_abs_diff(&arm_a_pos0, &reference);
            assert!(
                arm_a_diff < Q4K_TOLERANCE,
                "{}: arm A m={m} correctness fail: diff={arm_a_diff}",
                shape.label
            );

            let mut arm_a_samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let start = Instant::now();
                let result = matmul_q4k_q8k_f32_wide(&weight_bytes, rows, &activation, m)
                    .expect("arm a computes");
                let elapsed = start.elapsed();
                black_box(&result);
                arm_a_samples.push(elapsed.as_nanos() as f64);
            }
            let arm_a_stats = stats(&arm_a_samples);
            let arm_a_bytes = (weight_bytes.len() + activation.len() * 4 + rows * m * 4) as f64;

            // --- arm B: dequant-in-timing + sgemm ---
            let mut arm_b_output = vec![0f32; rows * m];
            q4_k::dequantize(&weight_bytes, &mut f32_scratch).expect("arm b first dequant");
            sgemm_rows_by_m(&f32_scratch, &activation, &mut arm_b_output, rows, k, m);
            let arm_b_pos0: Vec<f32> = (0..rows).map(|row| arm_b_output[row * m]).collect();
            let arm_b_diff = max_abs_diff(&arm_b_pos0, &reference);
            assert!(
                arm_b_diff < FLOAT_TOLERANCE,
                "{}: arm B m={m} correctness fail: diff={arm_b_diff}",
                shape.label
            );

            let mut arm_b_samples = Vec::with_capacity(REPEATS);
            let mut arm_b_dequant_only_samples = Vec::with_capacity(REPEATS);
            let mut arm_b_gemm_only_samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let start = Instant::now();
                q4_k::dequantize(&weight_bytes, &mut f32_scratch).expect("arm b dequant");
                let dequant_done = Instant::now();
                sgemm_rows_by_m(&f32_scratch, &activation, &mut arm_b_output, rows, k, m);
                let elapsed = start.elapsed();
                black_box(&arm_b_output);
                arm_b_samples.push(elapsed.as_nanos() as f64);
                arm_b_dequant_only_samples.push((dequant_done - start).as_nanos() as f64);
                arm_b_gemm_only_samples.push((Instant::now() - dequant_done).as_nanos() as f64);
            }
            let arm_b_stats = stats(&arm_b_samples);
            let arm_b_dequant_stats = stats(&arm_b_dequant_only_samples);
            let arm_b_gemm_stats = stats(&arm_b_gemm_only_samples);
            let arm_b_bytes =
                (weight_bytes.len() + rows * k * 4 * 2 + activation.len() * 4 + rows * m * 4)
                    as f64;

            // --- arm C: sgemm on already-f32 weights, no dequant (control) ---
            let mut arm_c_output = vec![0f32; rows * m];
            sgemm_rows_by_m(
                &f32_weights_once,
                &activation,
                &mut arm_c_output,
                rows,
                k,
                m,
            );
            let arm_c_pos0: Vec<f32> = (0..rows).map(|row| arm_c_output[row * m]).collect();
            let arm_c_diff = max_abs_diff(&arm_c_pos0, &reference);
            assert!(
                arm_c_diff < FLOAT_TOLERANCE,
                "{}: arm C m={m} correctness fail: diff={arm_c_diff}",
                shape.label
            );
            let arm_bc_diff = max_abs_diff(&arm_b_pos0, &arm_c_pos0);

            let mut arm_c_samples = Vec::with_capacity(REPEATS);
            for _ in 0..REPEATS {
                let start = Instant::now();
                sgemm_rows_by_m(
                    &f32_weights_once,
                    &activation,
                    &mut arm_c_output,
                    rows,
                    k,
                    m,
                );
                let elapsed = start.elapsed();
                black_box(&arm_c_output);
                arm_c_samples.push(elapsed.as_nanos() as f64);
            }
            let arm_c_stats = stats(&arm_c_samples);
            let arm_c_bytes = (rows * k * 4 + activation.len() * 4 + rows * m * 4) as f64;

            let gmacs = |ns: f64| macs as f64 / ns;
            let gbs = |bytes: f64, ns: f64| bytes / ns;

            println!(
                "{} m={m:4} arm=A  ns={:>12.1} cov%={:>5.2} gmac/s={:>7.3} gb/s={:>7.3} macs={macs} n={REPEATS} diff={arm_a_diff:e}",
                shape.label,
                arm_a_stats.mean_ns,
                arm_a_stats.cov_pct,
                gmacs(arm_a_stats.mean_ns),
                gbs(arm_a_bytes, arm_a_stats.mean_ns)
            );
            println!(
                "{} m={m:4} arm=B  ns={:>12.1} cov%={:>5.2} gmac/s={:>7.3} gb/s={:>7.3} macs={macs} n={REPEATS} diff={arm_b_diff:e} dequant_ns={:>10.1} dequant_cov%={:>5.2} gemm_ns={:>10.1} gemm_cov%={:>5.2}",
                shape.label,
                arm_b_stats.mean_ns,
                arm_b_stats.cov_pct,
                gmacs(arm_b_stats.mean_ns),
                gbs(arm_b_bytes, arm_b_stats.mean_ns),
                arm_b_dequant_stats.mean_ns,
                arm_b_dequant_stats.cov_pct,
                arm_b_gemm_stats.mean_ns,
                arm_b_gemm_stats.cov_pct
            );
            println!(
                "{} m={m:4} arm=C  ns={:>12.1} cov%={:>5.2} gmac/s={:>7.3} gb/s={:>7.3} macs={macs} n={REPEATS} diff={arm_c_diff:e} bc_diff={arm_bc_diff:e}",
                shape.label,
                arm_c_stats.mean_ns,
                arm_c_stats.cov_pct,
                gmacs(arm_c_stats.mean_ns),
                gbs(arm_c_bytes, arm_c_stats.mean_ns)
            );
        }
    }

    /// Thread count for BOTH arms comes from the launching shell's
    /// environment, read (never set) here -- `PROXIMA_MATMUL_WORKERS` for
    /// arm A's `OnceLock` (`cpu.rs:12585-12601`, fixed at the FIRST call in
    /// the process, so a shell-level `export` before `exec` is the only
    /// reliable way to control it) and `VECLIB_MAXIMUM_THREADS` for
    /// Accelerate's own internal pool (arms B/C's `cblas_sgemm`). Printed,
    /// not asserted -- Accelerate exposes no readback API, so this prints
    /// what the process saw in its environment, not proof the framework
    /// honored it; the sweep's own arm-C throughput is the behavioral check.
    pub fn run(path: &str) {
        let workers =
            std::env::var("PROXIMA_MATMUL_WORKERS").unwrap_or_else(|_| "unset".to_string());
        let veclib =
            std::env::var("VECLIB_MAXIMUM_THREADS").unwrap_or_else(|_| "unset".to_string());
        println!("PROXIMA_MATMUL_WORKERS={workers} VECLIB_MAXIMUM_THREADS={veclib}");
        let (parsed, file_len) = parse_header(Path::new(path));
        let mut file = File::open(path).expect("reopen real gguf file for tensor data");
        for shape in SHAPES {
            probe_shape(&mut file, &parsed, file_len, shape);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    let path = mac::gguf_path();
    if !std::path::Path::new(&path).exists() {
        println!("real gguf file not found at {path}; nothing to probe");
        return;
    }
    mac::run(&path);
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    println!(
        "this probe is macos/aarch64-only (Accelerate cblas_sgemm); nothing to run on this host"
    );
}
