//! Isolated GPU speed of the staging-loop switches (see
//! `docs/model-interop/discipline.md` ROWs C4.9/C4.10)
//! against the phase-1 baseline (`PROXIMA_TILED_GEMM_Q4_0=1`, nothing
//! else). `PROXIMA_TILED_GEMM_PTR_HOIST` (item 3b) and `PROXIMA_TILED_
//! GEMM_DECODE_SPREAD`/`_INTERIOR_STORE` (items 3a/3d) were measured here
//! and found noise-level or a regression on this shape (see
//! `docs/model-interop/discipline.md` ROW C4.9) -- ROLLBACK; this probe
//! now covers only the two switches that shipped, `_WIDE_ACT_LOAD` (item
//! 3c) and `_SLIM_TGMEM` (phase 2), both default ON as of ROWs C4.9/C4.10
//! (unset admits; explicit `"0"` disables) -- every combo below forces the
//! OTHER switch to `"0"` explicitly so the isolated arms stay meaningful.
//! Same `execute_plan_op_timed`
//! real-`GPUStartTime`/`GPUEndTime` isolated-timing technique as
//! `q4_0_tiled_gemm_run8_speed_probe.rs`.
//!
//! Two shapes: ROW C4.9's own target Q4_0 weight matmul
//! `[K=1536, M=12288] x [N=510, K=1536]`, and node 162's dense-batched
//! "value" shape (this repo's own admitted census: feature=256,
//! token=510, batch=8, K=512) run through `PROXIMA_TILED_GEMM_DENSE=1` --
//! included for completeness even though neither switch touches
//! `push_dense_batched_gemm_body`'s weight-decode loop, so this arm is
//! expected to show no attributable delta; that null result is itself
//! part of the evidence.
//!
//! ISOLATED, not end-to-end: one synthetic op run in a loop, nothing else
//! touching the GPU. Weight bytes are synthetic (random `Q4_0` blocks).

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos"))]
    run();
    #[cfg(not(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos")))]
    println!(
        "staging_switches_isolated_probe requires --features metal,metal-tiled-gemm,instrument on macOS"
    );
}

// phase-2 occupancy investigation: reads back each compiled
// tiled-GEMM pipeline's `staticThreadgroupMemoryLength` via the
// `pipeline_footprint` debug event this session added at
// `pipeline_buffers_upload.rs`'s `pipeline_for` -- installed here rather than
// in the library because this is the one call site that needs a console sink
// at all; `omega` itself stays a no-op emitter without an installed recorder.
#[cfg(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos"))]
fn install_footprint_telemetry() -> std::sync::Arc<proxima_telemetry::recorder::Recorder<proxima_telemetry::clock::GlobalClock>> {
    proxima_telemetry::emit::global::install(proxima_telemetry::emit::EnvFilter::parse("debug"));
    proxima_telemetry::recorder::Recorder::builder()
        .ring_capacity(4096)
        .export(proxima_telemetry::export::Exporter::stderr())
        .expect("stderr exporter installs")
        .install()
        .expect("telemetry recorder installs")
}

#[cfg(all(feature = "metal", feature = "metal-tiled-gemm", feature = "instrument", target_os = "macos"))]
fn run() {
    use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, quantize};
    use proxima_primitives::Codec;
    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce,
        ReduceInit, ScalarOp, append, projection,
    };

    let telemetry_recorder = install_footprint_telemetry();

    const IN_DIM: usize = 1536;
    const OUT_DIM: usize = 12288;
    const TOKENS: usize = 510;
    const RUNS: usize = 12;

    fn random_vec(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    fn pack_rows(rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
        let blocks_per_row = in_dim / QK4_0;
        let mut packed = vec![0u8; rows.len() * blocks_per_row * BLOCK_BYTES];
        for (row, row_packed) in rows.iter().zip(packed.chunks_exact_mut(blocks_per_row * BLOCK_BYTES)) {
            quantize(row, row_packed).expect("in_dim is a whole multiple of QK4_0");
        }
        packed
    }

    fn matmul_program(tokens: u32, in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(in_dim), Extent::Static(out_dim)],
                name: Some("weight".into()),
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(tokens), Extent::Static(in_dim)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                    (activation, IndexMap::Affine(projection(3, &[0, 1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(projection(3, &[0, 2])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    fn dense_batched_value_program() -> (Vec<Op>, NodeId, usize, usize, usize, usize) {
        // node 162's own admitted shape: feature=256,
        // token=510, batch=8, K=512.
        let (token, feature, batch, reduce_len) = (510u32, 256u32, 8u32, 512u32);
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(feature), Extent::Static(batch), Extent::Static(reduce_len)],
                name: None,
            },
        );
        let other = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(token), Extent::Static(batch), Extent::Static(reduce_len)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(4, &[1, 2, 3]))),
                    (other, IndexMap::Affine(projection(4, &[0, 2, 3]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(projection(4, &[0, 1, 2])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum, token as usize, feature as usize, batch as usize, reduce_len as usize)
    }

    fn median_gpu_ns(program: &[Op], root: NodeId, blocks: &[QuantizedBlock<'_>], runs: usize) -> u64 {
        let plan = omega::plan(program, &[], blocks, &[root], NumericPolicy::default()).expect("plan compiles");
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let (_, timings) = omega::metal::execute_plan_op_timed(&plan, blocks, None)
                .expect("metal executes on a real device");
            let total: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
            samples.push(total);
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    if let Ok(output) = std::process::Command::new("pgrep").args(["-fl", "ollama|llama-server|Ollama"]).output()
        && !output.stdout.is_empty()
    {
        eprintln!(
            "WARNING: ollama/llama-server/Ollama still running -- GPU is not quiet:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }

    // Q4_0 weight-matmul shape (STAGING.md's own target).
    let rows: Vec<Vec<f32>> = (0..OUT_DIM).map(|row| random_vec(11 + row as u64, IN_DIM)).collect();
    let packed = pack_rows(&rows, IN_DIM);
    let activation = random_vec(97, TOKENS * IN_DIM);
    let (program, sum) = matmul_program(TOKENS as u32, IN_DIM as u32, OUT_DIM as u32);
    let blocks = [
        QuantizedBlock::Packed { codec: Codec::Q4_0, bytes: &packed },
        QuantizedBlock::Float32(&activation),
    ];

    // both staging switches default ON now (unset admits); to keep this
    // probe's isolated A/B comparisons meaningful, "baseline" and the
    // single-switch arms explicitly force the OTHER switch off with "0"
    // rather than relying on unset, which no longer means off.
    let combos: &[(&str, &[(&str, &str)])] = &[
        (
            "baseline (Q4_0 tiled only, staging switches off)",
            &[
                ("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", "0"),
                ("PROXIMA_TILED_GEMM_SLIM_TGMEM", "0"),
            ],
        ),
        ("wide_act_load alone", &[("PROXIMA_TILED_GEMM_SLIM_TGMEM", "0")]),
        // phase 2 (see `docs/model-interop/discipline.md` ROW C4.10):
        // `out_tile` aliased onto
        // `weight_tile`/`act_tile`'s backing bytes.
        ("slim_tgmem alone", &[("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", "0")]),
        ("slim_tgmem + wide_act_load (new default, unset)", &[]),
    ];

    println!("=== Q4_0 weight matmul [K={IN_DIM}, M={OUT_DIM}] x [N={TOKENS}, K={IN_DIM}] ===");
    let mut baseline_ns = 0u64;
    for &(label, extra) in combos {
        let mut vars: Vec<(&str, Option<&str>)> = vec![("PROXIMA_TILED_GEMM_Q4_0", Some("1"))];
        for &(key, value) in extra {
            vars.push((key, Some(value)));
        }
        let ns = temp_env::with_vars(vars, || median_gpu_ns(&program, sum, &blocks, RUNS));
        telemetry_recorder.drain();
        if label.starts_with("baseline") {
            baseline_ns = ns;
        }
        println!(
            "  [{label}] median={ns} ns = {:.4} ms  ratio-vs-baseline={:.3}x",
            ns as f64 / 1e6,
            baseline_ns as f64 / ns as f64
        );
    }

    // node 162's dense-batched shape -- none of these switches touch
    // `push_dense_batched_gemm_body` this session; expected null result.
    let (dense_program, dense_sum, token, feature, batch, reduce_len) = dense_batched_value_program();
    let weight_data = random_vec(2026, feature * batch * reduce_len);
    let other_data = random_vec(2027, token * batch * reduce_len);
    let dense_blocks = [QuantizedBlock::Float32(&weight_data), QuantizedBlock::Float32(&other_data)];

    println!(
        "=== node162-shaped dense batched value [token={token} feature={feature} batch={batch} K={reduce_len}] ==="
    );
    let mut dense_baseline_ns = 0u64;
    for &(label, extra) in combos {
        let mut vars: Vec<(&str, Option<&str>)> = vec![("PROXIMA_TILED_GEMM_DENSE", Some("1"))];
        for &(key, value) in extra {
            vars.push((key, Some(value)));
        }
        let ns = temp_env::with_vars(vars, || median_gpu_ns(&dense_program, dense_sum, &dense_blocks, RUNS));
        telemetry_recorder.drain();
        if label.starts_with("baseline") {
            dense_baseline_ns = ns;
        }
        println!(
            "  [{label}] median={ns} ns = {:.4} ms  ratio-vs-baseline={:.3}x",
            ns as f64 / 1e6,
            dense_baseline_ns as f64 / ns as f64
        );
    }

    // phase 2 (see `docs/model-interop/discipline.md` ROW C4.10): interleaved baseline / wide_act_load
    // / slim_tgmem comparison, 12 samples per round x 3 rounds, round-robin
    // across configs so a clock-ramp or thermal drift within one round
    // cannot land entirely on one config's numbers.
    println!("=== interleaved baseline vs wide_act_load vs slim_tgmem (Q4_0 shape, 12 samples x 3 rounds) ===");
    let interleave_configs: &[(&str, &[(&str, &str)])] = &[
        (
            "baseline",
            &[
                ("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", "0"),
                ("PROXIMA_TILED_GEMM_SLIM_TGMEM", "0"),
            ],
        ),
        ("wide_act_load", &[("PROXIMA_TILED_GEMM_SLIM_TGMEM", "0")]),
        ("slim_tgmem", &[("PROXIMA_TILED_GEMM_WIDE_ACT_LOAD", "0")]),
    ];
    // the kernel source (and which switch fires) is decided at first Metal
    // pipeline compile, not at `omega::plan` -- warm each plan with one
    // real dispatch INSIDE its own env scope so the right variant lands in
    // `PIPELINE_CACHE` before the interleaved loop below reads the env
    // var's REVERTED (unset) state on every subsequent call.
    let plans: Vec<_> = interleave_configs
        .iter()
        .map(|&(label, extra)| {
            let mut vars: Vec<(&str, Option<&str>)> = vec![("PROXIMA_TILED_GEMM_Q4_0", Some("1"))];
            for &(key, value) in extra {
                vars.push((key, Some(value)));
            }
            let plan = temp_env::with_vars(vars, || {
                let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default())
                    .expect("plan compiles");
                omega::metal::execute_plan_op_timed(&plan, &blocks, None)
                    .expect("warm-up dispatch executes on a real device");
                plan
            });
            (label, plan)
        })
        .collect();
    let mut samples: Vec<Vec<u64>> = vec![Vec::new(); interleave_configs.len()];
    for _round in 0..3 {
        for _sample in 0..RUNS {
            for (index, (_, plan)) in plans.iter().enumerate() {
                let (_, timings) = omega::metal::execute_plan_op_timed(plan, &blocks, None)
                    .expect("metal executes on a real device");
                let total: u64 = timings.iter().map(|timing| timing.gpu_ns).sum();
                samples[index].push(total);
            }
        }
        telemetry_recorder.drain();
    }
    let interleaved_baseline_median = {
        let mut sorted = samples[0].clone();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    };
    for (index, (label, _)) in plans.iter().enumerate() {
        let mut sorted = samples[index].clone();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2];
        let min = *sorted.first().expect("at least one sample");
        let max = *sorted.last().expect("at least one sample");
        println!(
            "  [{label}] n={} median={median} ns = {:.4} ms  min={} ns max={} ns  ratio-vs-baseline={:.3}x",
            sorted.len(),
            median as f64 / 1e6,
            min,
            max,
            interleaved_baseline_median as f64 / median as f64
        );
    }
}
