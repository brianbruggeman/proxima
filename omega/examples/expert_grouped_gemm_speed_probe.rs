//! Isolated GPU time of one gathered `Q8_0` expert matmul at prefill width on
//! the expert-grouped tiled path (`metal-grouped-gemm`), at the granite moe
//! 1b expert shapes: gate/up `[1024 -> 512]`, down `[512 -> 1024]`, 32
//! experts, one route index per token (one top-k slot per dispatch, which is
//! what the per-route program issues).
//!
//! Timing is `GPUStartTime`/`GPUEndTime` per op ([`omega::metal::execute_plan_named_op_timed`],
//! `instrument` feature): the command buffer's own GPU occupancy, one op in a
//! loop with nothing else on the device. Weight bytes are synthetic (random
//! `Q8_0` blocks); correctness is `expert_grouped_gemm_parity.rs`'s job.
//!
//! Five routing shapes, because the kernel's cost depends on how tokens fall
//! on experts, not only on how many there are: `balanced` (token t to expert
//! t mod E, every expert one near-full tile), `uniform` (each token draws an
//! expert uniformly at random, the distribution llama.cpp's own
//! `test-backend-ops` MUL_MAT_ID perf cases use), `zipf` (expert e gets
//! weight 1/(e+1), a few experts own many tiles and most own a partial one),
//! `single` (one expert owns every token, the dense-equivalent worst case for
//! the scan) and `chain` (token blocks of 32 to consecutive experts, so every
//! busy threadgroup runs exactly one tile). Each arm runs five untimed
//! dispatches first so the GPU clock has ramped; a dispatch this small still
//! runs at a lower clock than the seconds-long prefill it belongs to, so the
//! absolute microseconds are a lower-load figure and the ratios between arms
//! and between builds are the readable part.
//!
//! ```sh
//! cargo run --release -p omega --example expert_grouped_gemm_speed_probe \
//!   --features metal,metal-grouped-gemm,instrument
//! ```

#[cfg(all(
    feature = "metal",
    feature = "metal-grouped-gemm",
    feature = "instrument",
    target_os = "macos"
))]
use anyhow::Context;
#[cfg(all(
    feature = "metal",
    feature = "metal-grouped-gemm",
    feature = "instrument",
    target_os = "macos"
))]
use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, quantize};
#[cfg(all(
    feature = "metal",
    feature = "metal-grouped-gemm",
    feature = "instrument",
    target_os = "macos"
))]
use proxima_primitives::Codec;
#[cfg(all(
    feature = "metal",
    feature = "metal-grouped-gemm",
    feature = "instrument",
    target_os = "macos"
))]
use proxima_tensor::map::{self, AxisIndex, AxisTerm};
#[cfg(all(
    feature = "metal",
    feature = "metal-grouped-gemm",
    feature = "instrument",
    target_os = "macos"
))]
use proxima_tensor::test_support::Lcg;
#[cfg(all(
    feature = "metal",
    feature = "metal-grouped-gemm",
    feature = "instrument",
    target_os = "macos"
))]
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit,
    ScalarOp, append,
};

fn main() -> anyhow::Result<()> {
    #[cfg(all(
        feature = "metal",
        feature = "metal-grouped-gemm",
        feature = "instrument",
        target_os = "macos"
    ))]
    return run();
    #[cfg(not(all(
        feature = "metal",
        feature = "metal-grouped-gemm",
        feature = "instrument",
        target_os = "macos"
    )))]
    {
        println!(
            "expert_grouped_gemm_speed_probe requires --features metal,metal-grouped-gemm,instrument on macOS"
        );
        Ok(())
    }
}

#[cfg(all(
    feature = "metal",
    feature = "metal-grouped-gemm",
    feature = "instrument",
    target_os = "macos"
))]
fn run() -> anyhow::Result<()> {
    const EXPERTS: usize = 32;
    const RUNS: usize = 21;
    const WARMUP_RUNS: usize = 5;
    const DEFAULT_TOKEN_COUNTS: [usize; 4] = [160, 512, 1000, 2048];
    const SHAPES: [(&str, usize, usize); 2] = [("gate_up", 512, 1024), ("down", 1024, 512)];
    const ROUTINGS: [&str; 5] = ["balanced", "uniform", "zipf", "single", "chain"];

    fn env_list<T: std::str::FromStr>(name: &str, default: Vec<T>) -> Vec<T> {
        match std::env::var(name) {
            Ok(value) => value.split(',').filter_map(|item| item.trim().parse().ok()).collect(),
            Err(_) => default,
        }
    }

    struct Shape {
        tokens: usize,
        selected: usize,
        per_selected_activation: bool,
        rows: usize,
        k: usize,
    }

    fn unit_values(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit() * 2.0 - 1.0).collect()
    }

    fn expert_stack(shape: &Shape) -> anyhow::Result<Vec<u8>> {
        let row_bytes = shape.k / QK8_0 * BLOCK_BYTES;
        let mut stack = vec![0u8; EXPERTS * shape.rows * row_bytes];
        let weights = unit_values(11, shape.rows * shape.k);
        for expert_bytes in stack.chunks_exact_mut(shape.rows * row_bytes) {
            for (row, row_bytes_out) in weights
                .chunks_exact(shape.k)
                .zip(expert_bytes.chunks_exact_mut(row_bytes))
            {
                quantize(row, row_bytes_out).context("k is a whole number of q8_0 blocks")?;
            }
        }
        Ok(stack)
    }

    fn route(shape_name: &str, tokens: usize) -> Vec<f32> {
        let mut lcg = Lcg(5);
        let mut unit_draw = move || (lcg.next_unit() + 1.0) * 0.5;
        let weights: Vec<f32> = (0..EXPERTS).map(|expert| 1.0 / (expert as f32 + 1.0)).collect();
        let total: f32 = weights.iter().sum();
        (0..tokens)
            .map(|token| match shape_name {
                "balanced" => (token % EXPERTS) as f32,
                "uniform" => (unit_draw() * EXPERTS as f32).min(EXPERTS as f32 - 1.0).floor(),
                "single" => 7.0,
                "chain" => ((token / 32) % EXPERTS) as f32,
                _ => {
                    let mut draw = unit_draw() * total;
                    let mut chosen = EXPERTS - 1;
                    for (expert, weight) in weights.iter().enumerate() {
                        if draw < *weight {
                            chosen = expert;
                            break;
                        }
                        draw -= weight;
                    }
                    chosen as f32
                }
            })
            .collect()
    }

    fn top_k_route(tokens: usize, selected: usize) -> Vec<f32> {
        let mut lcg = Lcg(5);
        let mut route = Vec::with_capacity(tokens * selected);
        for _ in 0..tokens {
            let mut pool: Vec<usize> = (0..EXPERTS).collect();
            for slot in 0..selected {
                let draw = ((lcg.next_unit() + 1.0) * 0.5 * (EXPERTS - slot) as f32) as usize;
                route.push(pool.remove(draw.min(pool.len() - 1)) as f32);
            }
        }
        route
    }

    fn stacked_program(shape: &Shape) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let input = |program: &mut Vec<Op>, dtype: DType, extents: &[usize], name: &str| {
            append(
                program,
                Op::Input {
                    dtype,
                    shape: extents.iter().map(|&extent| Extent::Static(extent as u32)).collect(),
                    name: Some(name.into()),
                },
            )
        };
        let weight = input(&mut program, DType::UInt8, &[EXPERTS, shape.rows, shape.k], "weight");
        let route_node = input(&mut program, DType::Float32, &[shape.tokens, shape.selected], "route");
        let (activation, activation_map) = if shape.per_selected_activation {
            (
                input(&mut program, DType::Float32, &[shape.tokens, shape.selected, shape.k], "activation"),
                map::projection(4, &[0, 1, 3]),
            )
        } else {
            (
                input(&mut program, DType::Float32, &[shape.tokens, shape.k], "activation"),
                map::projection(4, &[0, 3]),
            )
        };
        let gather = IndexMap::Computed {
            indices: route_node,
            index_map: map::projection(4, &[0, 1]),
            base: map::IndexPattern {
                iter_rank: 4,
                axes: vec![
                    AxisIndex::default(),
                    AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(2)).collect(),
                        offset: 0,
                        len: None,
                    },
                    AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(3)).collect(),
                        offset: 0,
                        len: None,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![(weight, gather), (activation, IndexMap::Affine(activation_map))],
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
                in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(map::projection(4, &[0, 1, 2])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    fn build_program(shape: &Shape) -> (Vec<Op>, NodeId) {
        if shape.selected > 1 {
            return stacked_program(shape);
        }
        let mut program = Vec::new();
        let input = |program: &mut Vec<Op>, dtype: DType, extents: &[usize], name: &str| {
            append(
                program,
                Op::Input {
                    dtype,
                    shape: extents.iter().map(|&extent| Extent::Static(extent as u32)).collect(),
                    name: Some(name.into()),
                },
            )
        };
        let weight = input(&mut program, DType::UInt8, &[EXPERTS, shape.rows, shape.k], "weight");
        let route_node = input(&mut program, DType::Int32, &[shape.tokens], "route");
        let activation = input(&mut program, DType::Float32, &[shape.tokens, shape.k], "activation");
        let gather = IndexMap::Computed {
            indices: route_node,
            index_map: map::projection(3, &[0]),
            base: map::IndexPattern {
                iter_rank: 3,
                axes: vec![
                    AxisIndex::default(),
                    AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(1)).collect(),
                        offset: 0,
                        len: None,
                    },
                    AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(2)).collect(),
                        offset: 0,
                        len: None,
                    },
                ],
            },
            gathered_dim: 0,
        };
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, gather),
                    (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
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
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    fn median(mut samples: Vec<u64>) -> u64 {
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    fn coefficient_of_variation_percent(samples: &[u64]) -> f64 {
        let mean = samples.iter().sum::<u64>() as f64 / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|&sample| (sample as f64 - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        variance.sqrt() / mean * 100.0
    }

    println!("probe=expert_grouped_gemm experts={EXPERTS} runs={RUNS} unit=us gpu_clock=GPUStartTime..GPUEndTime");
    let token_counts = env_list("PROBE_TOKEN_COUNTS", DEFAULT_TOKEN_COUNTS.to_vec());
    let routings = env_list("PROBE_ROUTINGS", ROUTINGS.iter().map(|routing| (*routing).to_string()).collect());
    let shape_names = env_list("PROBE_SHAPES", SHAPES.iter().map(|shape| shape.0.to_string()).collect());
    let selected = env_list("PROBE_SELECTED", vec![1_usize]).first().copied().unwrap_or(1).max(1);
    let interleave_per_slot = std::env::var_os("PROBE_AB").is_some();
    for (name, rows, k) in SHAPES.into_iter().filter(|shape| shape_names.iter().any(|wanted| wanted == shape.0)) {
        for &tokens in &token_counts {
            let shape = Shape {
                tokens,
                selected,
                per_selected_activation: selected > 1 && name == "down",
                rows,
                k,
            };
            let stack = expert_stack(&shape)?;
            let activation_len = if shape.per_selected_activation {
                tokens * selected * k
            } else {
                tokens * k
            };
            let activation = unit_values(23, activation_len);
            let (program, root) = build_program(&shape);
            for routing in routings.iter().map(String::as_str) {
                let route_values = if selected > 1 {
                    top_k_route(tokens, selected)
                } else {
                    route(routing, tokens)
                };
                let named = [
                    ("weight", QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: &stack }),
                    ("route", QuantizedBlock::Float32(&route_values)),
                    ("activation", QuantizedBlock::Float32(&activation)),
                ];
                let plan = omega::plan_named(&program, &[], &named, &[root], NumericPolicy::default())
                    .context("plan compiles")?;
                for _ in 0..WARMUP_RUNS {
                    omega::metal::execute_plan_named_op_timed(&plan, &named, None)
                        .context("metal warm-up executes on a real device")?;
                }
                let mut samples = Vec::with_capacity(RUNS);
                let mut per_slot_samples = Vec::with_capacity(RUNS);
                let per_slot_case = if interleave_per_slot && selected > 1 {
                    let single = Shape {
                        tokens,
                        selected: 1,
                        per_selected_activation: false,
                        rows,
                        k,
                    };
                    let single_activation = unit_values(23, tokens * k);
                    let single_route = route(routing, tokens);
                    let (single_program, single_root) = build_program(&single);
                    Some((single_program, single_root, single_activation, single_route))
                } else {
                    None
                };
                let per_slot_plan = match &per_slot_case {
                    Some((single_program, single_root, single_activation, single_route)) => {
                        let single_named = [
                            ("weight", QuantizedBlock::Packed { codec: Codec::Q8_0, bytes: &stack }),
                            ("route", QuantizedBlock::Float32(single_route)),
                            ("activation", QuantizedBlock::Float32(single_activation)),
                        ];
                        let single_plan = omega::plan_named(
                            single_program,
                            &[],
                            &single_named,
                            &[*single_root],
                            NumericPolicy::default(),
                        )
                        .context("per-slot plan compiles")?;
                        for _ in 0..WARMUP_RUNS {
                            omega::metal::execute_plan_named_op_timed(&single_plan, &single_named, None)
                                .context("per-slot warm-up executes")?;
                        }
                        Some((single_plan, single_named))
                    }
                    None => None,
                };
                for _ in 0..RUNS {
                    if let Some((single_plan, single_named)) = &per_slot_plan {
                        let (_, timings) =
                            omega::metal::execute_plan_named_op_timed(single_plan, single_named, None)
                                .context("per-slot executes")?;
                        per_slot_samples.push(timings.iter().map(|timing| timing.gpu_ns).sum::<u64>());
                    }
                    let (_, timings) = omega::metal::execute_plan_named_op_timed(&plan, &named, None)
                        .context("metal executes on a real device")?;
                    samples.push(timings.iter().map(|timing| timing.gpu_ns).sum::<u64>());
                }
                if !per_slot_samples.is_empty() {
                    let per_slot_us = median(per_slot_samples.clone()) as f64 / 1000.0;
                    let stacked_us = median(samples.clone()) as f64 / 1000.0;
                    println!(
                        "ab shape={name} tokens={tokens} selected={selected} per_slot_us={per_slot_us:.1} per_slot_cov_pct={:.2} stacked_us={stacked_us:.1} stacked_cov_pct={:.2} stacked_over_{selected}x_per_slot={:.3}",
                        coefficient_of_variation_percent(&per_slot_samples),
                        coefficient_of_variation_percent(&samples),
                        stacked_us / (per_slot_us * selected as f64)
                    );
                }
                let gflop = 2.0 * (tokens * selected) as f64 * rows as f64 * k as f64 / 1.0e9;
                let microseconds = median(samples.clone()) as f64 / 1000.0;
                println!(
                    "shape={name} tokens={tokens} selected={selected} routing={routing} median_us={microseconds:.1} cov_pct={:.2} useful_tflops={:.3}",
                    coefficient_of_variation_percent(&samples),
                    gflop / microseconds * 1.0e3
                );
            }
        }
    }
    Ok(())
}
