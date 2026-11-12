//! Isolated GPU time of flint's 2024-token prefill GEMMs with the activation
//! tile staged in binary32, binary16 and bfloat16 (`PROXIMA_TILED_GEMM_STAGING`),
//! plus a binary32 A/A control. One op per dispatch, nothing else on the device,
//! `GPUStartTime`/`GPUEndTime` per op ([`omega::metal::execute_plan_named_op_timed`]).
//!
//! Shapes are flint's (`proxima-model-interop/tests/fixtures/flint/tensor_dir.txt`):
//! `attn_q` Q8_0 dims (2048, 4096), expert stacks of 256 experts, gate/up Q5_K
//! 2048 -> 512 and down Q5_K 512 -> 2048, top-k 6, stacked into one dispatch.
//! The staging environment variable is read when the kernel is emitted, so each
//! arm sets it around its own plan build; the output hash and non-finite count
//! per arm prove which kernel ran. Arms run in rotated order each round and a
//! round is skipped (and recorded) when a named competing process is alive.
//!
//! ```sh
//! cargo run --release -p proxima-omega --example staging_format_gemm_probe \
//!   --features metal,metal-tiled-gemm,metal-grouped-gemm,instrument
//! ```

fn main() -> anyhow::Result<()> {
    #[cfg(all(
        feature = "metal",
        feature = "metal-tiled-gemm",
        feature = "metal-grouped-gemm",
        feature = "instrument",
        target_os = "macos"
    ))]
    return probe::run();
    #[cfg(not(all(
        feature = "metal",
        feature = "metal-tiled-gemm",
        feature = "metal-grouped-gemm",
        feature = "instrument",
        target_os = "macos"
    )))]
    {
        println!("staging_format_gemm_probe requires --features metal,metal-tiled-gemm,metal-grouped-gemm,instrument on macOS");
        Ok(())
    }
}

#[cfg(all(
    feature = "metal",
    feature = "metal-tiled-gemm",
    feature = "metal-grouped-gemm",
    feature = "instrument",
    target_os = "macos"
))]
mod probe {
    use std::process::Command;

    use anyhow::Context;
    use proxima_gguf::quant::{q5_k, q8_0};
    use proxima_numeric::codec::Codec;
    use proxima_numeric::{DType, NumericPolicy};
    use proxima_tensor::map::{self, AxisIndex, AxisTerm};
    use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append};

    const TOKENS: usize = 2024;
    const EXPERTS: usize = 256;
    const SELECTED: usize = 6;
    const ROUNDS: usize = 7;
    const SAMPLES: usize = 15;
    const WARMUP: usize = 4;
    const OVERFLOW_VALUE: f32 = 8.4e4;
    const ARMS: [(&str, &str); 4] = [
        ("f32_a", "float32"),
        ("float16", "float16"),
        ("bfloat16", "bfloat16"),
        ("f32_b", "float32"),
    ];

    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Dense,
        Grouped,
    }

    struct Case {
        name: &'static str,
        kind: Kind,
        rows: usize,
        k: usize,
        per_selected_activation: bool,
        overflow: bool,
    }

    const CASES: [Case; 4] = [
        Case { name: "dense_q8_attn_q_2048to4096", kind: Kind::Dense, rows: 4096, k: 2048, per_selected_activation: false, overflow: false },
        Case { name: "grouped_q5k_gate_up_2048to512", kind: Kind::Grouped, rows: 512, k: 2048, per_selected_activation: false, overflow: false },
        Case { name: "grouped_q5k_down_512to2048", kind: Kind::Grouped, rows: 2048, k: 512, per_selected_activation: true, overflow: false },
        Case { name: "grouped_q5k_down_overflow1pct", kind: Kind::Grouped, rows: 2048, k: 512, per_selected_activation: true, overflow: true },
    ];

    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn unit(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f32 {
            let (first, second) = (self.unit(), self.unit());
            ((-2.0 * first.ln()).sqrt() * (2.0 * std::f64::consts::PI * second).cos()) as f32
        }
    }

    fn normal_rms_one(seed: u64, count: usize) -> Vec<f32> {
        let mut rng = Rng(seed);
        let mut values: Vec<f32> = (0..count).map(|_| rng.normal()).collect();
        let rms = (values.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / count as f64).sqrt() as f32;
        values.iter_mut().for_each(|v| *v /= rms);
        values
    }

    fn with_overflow(mut values: Vec<f32>) -> (Vec<f32>, usize) {
        let mut rng = Rng(77);
        let mut hits = 0;
        for value in values.iter_mut() {
            if rng.unit() < 0.01 {
                *value = if rng.next_u64() & 1 == 0 { OVERFLOW_VALUE } else { -OVERFLOW_VALUE };
                hits += 1;
            }
        }
        (values, hits)
    }

    fn weight_bytes(case: &Case) -> anyhow::Result<Vec<u8>> {
        let rows = normal_rms_one(11, case.rows * case.k).into_iter().map(|v| v * 0.05).collect::<Vec<_>>();
        let (block_bytes, block_elements) = match case.kind {
            Kind::Dense => (q8_0::BLOCK_BYTES, q8_0::QK8_0),
            Kind::Grouped => (q5_k::BLOCK_BYTES, q5_k::QK_K),
        };
        let row_bytes = case.k / block_elements * block_bytes;
        let mut one = vec![0u8; case.rows * row_bytes];
        for (row, out) in rows.chunks_exact(case.k).zip(one.chunks_exact_mut(row_bytes)) {
            match case.kind {
                Kind::Dense => q8_0::quantize(row, out).context("q8_0 quantize")?,
                Kind::Grouped => q5_k::quantize(row, out).context("q5_k quantize")?,
            }
        }
        if case.kind == Kind::Dense {
            return Ok(one);
        }
        Ok(one.repeat(EXPERTS))
    }

    fn input(program: &mut Vec<Op>, dtype: DType, extents: &[usize], name: &str) -> NodeId {
        let shape = extents.iter().map(|&extent| Extent::Static(extent as u32)).collect();
        append(program, Op::Input { dtype, shape, name: Some(name.into()) })
    }

    fn dense_program(case: &Case) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = input(&mut program, DType::UINT8, &[case.k, case.rows], "weight");
        let activation = input(&mut program, DType::FLOAT32, &[TOKENS, case.k], "activation");
        let product = append(&mut program, Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[0, 1]))),
            ],
            name: None,
        });
        let sum = append(&mut program, Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: None,
        }));
        (program, sum)
    }

    fn grouped_program(case: &Case) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = input(&mut program, DType::UINT8, &[EXPERTS, case.rows, case.k], "weight");
        let route = input(&mut program, DType::FLOAT32, &[TOKENS, SELECTED], "route");
        let (activation, activation_map) = if case.per_selected_activation {
            (input(&mut program, DType::FLOAT32, &[TOKENS, SELECTED, case.k], "activation"), map::projection(4, &[0, 1, 3]))
        } else {
            (input(&mut program, DType::FLOAT32, &[TOKENS, case.k], "activation"), map::projection(4, &[0, 3]))
        };
        let axis = |dim| AxisIndex { terms: core::iter::once(AxisTerm::projection(dim)).collect(), offset: 0, len: None };
        let gather = IndexMap::Computed {
            indices: route,
            index_map: map::projection(4, &[0, 1]),
            base: map::IndexPattern { iter_rank: 4, axes: vec![AxisIndex::default(), axis(2), axis(3)] },
            gathered_dim: 0,
        };
        let product = append(&mut program, Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![(weight, gather), (activation, IndexMap::Affine(activation_map))],
            name: None,
        });
        let sum = append(&mut program, Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
            out_map: IndexMap::Affine(map::projection(4, &[0, 1, 2])),
            keep: Keep::Reduce,
            name: None,
        }));
        (program, sum)
    }

    fn top_k_route() -> Vec<f32> {
        let mut rng = Rng(5);
        let mut route = Vec::with_capacity(TOKENS * SELECTED);
        for _ in 0..TOKENS {
            let mut pool: Vec<usize> = (0..EXPERTS).collect();
            for slot in 0..SELECTED {
                let draw = (rng.unit() * (EXPERTS - slot) as f64) as usize;
                route.push(pool.remove(draw.min(pool.len() - 1)) as f32);
            }
        }
        route
    }

    struct Prepared {
        weights: Vec<u8>,
        route: Vec<f32>,
        activation: Vec<f32>,
        overflow_fraction: f64,
    }

    fn prepare(case: &Case) -> anyhow::Result<Prepared> {
        let count = match (case.kind, case.per_selected_activation) {
            (Kind::Grouped, true) => TOKENS * SELECTED * case.k,
            _ => TOKENS * case.k,
        };
        let base = normal_rms_one(23, count);
        let (activation, hits) = if case.overflow { with_overflow(base) } else { (base, 0) };
        Ok(Prepared { weights: weight_bytes(case)?, route: top_k_route(), activation, overflow_fraction: hits as f64 / count as f64 })
    }

    struct ArmResult {
        samples_ns: Vec<u64>,
        output_hash: u64,
        non_finite: usize,
        outputs: usize,
    }

    fn fnv1a(bits: impl Iterator<Item = u32>) -> u64 {
        bits.fold(0xcbf2_9ce4_8422_2325u64, |hash, word| (hash ^ u64::from(word)).wrapping_mul(0x0000_0100_0000_01b3))
    }

    fn run_arm(case: &Case, prepared: &Prepared, layout: &str) -> anyhow::Result<ArmResult> {
        temp_env::with_var("PROXIMA_TILED_GEMM_STAGING", Some(layout), || -> anyhow::Result<ArmResult> {
            let (program, root) = if case.kind == Kind::Dense { dense_program(case) } else { grouped_program(case) };
            let codec = if case.kind == Kind::Dense { Codec::Q8_0 } else { Codec::Q5K };
            let mut named = vec![
                ("weight", QuantizedBlock::Packed { codec, bytes: &prepared.weights }),
                ("activation", QuantizedBlock::Float32(&prepared.activation)),
            ];
            if case.kind == Kind::Grouped {
                named.push(("route", QuantizedBlock::Float32(&prepared.route)));
            }
            let plan = omega::plan_named(&program, &[], &named, &[root], NumericPolicy::default()).context("plan compiles")?;
            for _ in 0..WARMUP {
                omega::metal::execute_plan_named_op_timed(&plan, &named, None).context("warm-up")?;
            }
            let mut samples_ns = Vec::with_capacity(SAMPLES);
            let (mut output_hash, mut non_finite, mut outputs) = (0, 0, 0);
            for _ in 0..SAMPLES {
                let (evaluated, timings) = omega::metal::execute_plan_named_op_timed(&plan, &named, None).context("timed run")?;
                samples_ns.push(timings.iter().map(|timing| timing.gpu_ns).sum::<u64>());
                let root_values = evaluated.root();
                outputs = root_values.len();
                non_finite = root_values.iter().filter(|value| !value.is_finite()).count();
                output_hash = fnv1a(root_values.iter().map(|value| value.to_bits()));
            }
            Ok(ArmResult { samples_ns, output_hash, non_finite, outputs })
        })
    }

    fn process_alive(pattern: &str) -> bool {
        Command::new("pgrep").args(["-f", pattern]).output().map(|out| out.status.success()).unwrap_or(true)
    }

    fn loadavg() -> String {
        Command::new("sysctl").args(["-n", "vm.loadavg"]).output().map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string()).unwrap_or_default()
    }

    fn median(values: &[u64]) -> u64 {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }

    #[derive(Default, Clone)]
    struct Pool {
        samples: Vec<u64>,
        round_medians: Vec<u64>,
        hash: u64,
        non_finite: usize,
        outputs: usize,
    }

    pub fn run() -> anyhow::Result<()> {
        println!("probe=staging_format_gemm tokens={TOKENS} experts={EXPERTS} top_k={SELECTED} rounds={ROUNDS} samples_per_block={SAMPLES} unit=us profile=release");
        let prepared: Vec<Prepared> = CASES.iter().map(prepare).collect::<anyhow::Result<_>>()?;
        let mut pools = vec![vec![Pool::default(); ARMS.len()]; CASES.len()];
        let mut skipped = Vec::new();
        for round in 0..ROUNDS {
            let load = loadavg();
            let busy = [("decode_gbps_baseline", "[d]ecode_gbps_baseline"), ("llama-server", "[l]lama-server")]
                .into_iter()
                .find(|(_, pattern)| process_alive(pattern));
            if let Some((name, _)) = busy {
                println!("round={round} SKIPPED process={name} loadavg={load}");
                skipped.push(round);
                continue;
            }
            println!("round={round} ran loadavg={load}");
            for (case_index, case) in CASES.iter().enumerate() {
                for step in 0..ARMS.len() {
                    let arm_index = (step + round) % ARMS.len();
                    let result = run_arm(case, &prepared[case_index], ARMS[arm_index].1)?;
                    let pool = &mut pools[case_index][arm_index];
                    pool.round_medians.push(median(&result.samples_ns));
                    pool.samples.extend(&result.samples_ns);
                    (pool.hash, pool.non_finite, pool.outputs) = (result.output_hash, result.non_finite, result.outputs);
                }
            }
        }
        println!("skipped_rounds={skipped:?}");
        for (case_index, case) in CASES.iter().enumerate() {
            let f32_median = median(&pools[case_index][0].samples).max(1) as f64;
            println!("case={} overflow_fraction={:.4}", case.name, prepared[case_index].overflow_fraction);
            for (arm_index, (arm, _)) in ARMS.iter().enumerate() {
                let pool = &pools[case_index][arm_index];
                let mid = median(&pool.samples) as f64;
                let (low, high) = (pool.samples.iter().min().copied().unwrap_or(0), pool.samples.iter().max().copied().unwrap_or(0));
                let (round_low, round_high) = (pool.round_medians.iter().min().copied().unwrap_or(0), pool.round_medians.iter().max().copied().unwrap_or(0));
                println!(
                    "  arm={arm} median_us={:.1} min_us={:.1} max_us={:.1} round_median_range_us={:.1}..{:.1} ratio_vs_f32_a={:.3} non_finite={}/{} output_fnv={:016x} n={}",
                    mid / 1e3, low as f64 / 1e3, high as f64 / 1e3, round_low as f64 / 1e3, round_high as f64 / 1e3,
                    mid / f32_median, pool.non_finite, pool.outputs, pool.hash, pool.samples.len()
                );
            }
        }
        Ok(())
    }
}
