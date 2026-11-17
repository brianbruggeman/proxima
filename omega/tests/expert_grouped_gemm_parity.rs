//! Correctness gate for the expert-grouped tiled GEMM (`metal-grouped-gemm`):
//! a `Q8_0` expert slab gathered per token by a route index, reduced over the
//! input axis, at prefill width. The oracle is `proxima_tensor::cpu`'s own
//! evaluator on the same packed bytes, which dequantizes in f32 and folds in
//! source order; the Metal kernel stages the dequantized weight tile as
//! `half`, so the two agree to half-precision weight rounding and the
//! tolerance below is that rounding, not a reduction-order excuse.
//!
//! Every case routes with a distribution a real router produces and a unit
//! test usually does not: experts that receive no token, one expert that
//! receives far more than one tile of tokens, and ragged row and token
//! extents that leave partial tiles on both edges.

#![cfg(all(feature = "metal", feature = "metal-grouped-gemm", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::MetalError;
use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, quantize};
use proxima_numeric::codec::Codec;
use proxima_tensor::map::{self, AxisIndex, AxisTerm};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, TensorError, append};
use proxima_numeric::{DType, NumericPolicy};

const MIN_TOKENS_FOR_TILES: usize = omega::sized::TILED_GEMM_MIN_TOKENS as usize;
const RELATIVE_TOLERANCE: f32 = 2.0e-3;

struct Shape {
    tokens: usize,
    rows: usize,
    k: usize,
    experts: usize,
}

struct Case {
    shape: Shape,
    route: Vec<f32>,
    fuse_gate: bool,
}

fn input(program: &mut Vec<Op>, dtype: DType, shape: &[usize], name: &str) -> NodeId {
    append(
        program,
        Op::Input {
            dtype,
            shape: shape
                .iter()
                .map(|&extent| Extent::Static(extent as u32))
                .collect(),
            name: Some(name.into()),
        },
    )
}

fn expert_gather_map(route: NodeId) -> IndexMap {
    IndexMap::Computed {
        indices: route,
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
    }
}

fn program(shape: &Shape, fuse_gate: bool) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = input(
        &mut program,
        DType::UINT8,
        &[shape.experts, shape.rows, shape.k],
        "weight",
    );
    let route = input(&mut program, DType::INT32, &[shape.tokens], "route");
    let activation = input(
        &mut program,
        DType::FLOAT32,
        &[shape.tokens, shape.k],
        "activation",
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, expert_gather_map(route)),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("expert_product".into()),
        }),
    );
    if !fuse_gate {
        return (program, sum);
    }
    let gate = input(
        &mut program,
        DType::FLOAT32,
        &[shape.tokens, shape.rows],
        "gate",
    );
    let gated = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (sum, IndexMap::Affine(map::projection(2, &[0, 1]))),
                (gate, IndexMap::Affine(map::projection(2, &[0, 1]))),
            ],
            name: None,
        },
    );
    (program, gated)
}

fn unit_values(seed: u64, count: usize, low: f32, high: f32) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count)
        .map(|_| low + (lcg.next_unit() + 1.0) * 0.5 * (high - low))
        .collect()
}

fn quantized_expert_stack(shape: &Shape) -> Vec<u8> {
    let row_bytes = shape.k / QK8_0 * BLOCK_BYTES;
    let mut stack = vec![0u8; shape.experts * shape.rows * row_bytes];
    for expert in 0..shape.experts {
        let magnitude = 0.5 + expert as f32;
        let weights = unit_values(
            1000 + expert as u64,
            shape.rows * shape.k,
            -magnitude,
            magnitude,
        );
        let expert_bytes = shape.rows * row_bytes;
        let target = &mut stack[expert * expert_bytes..(expert + 1) * expert_bytes];
        for (row, row_target) in weights
            .chunks_exact(shape.k)
            .zip(target.chunks_exact_mut(row_bytes))
        {
            quantize(row, row_target).expect("a row is a whole number of q8_0 blocks");
        }
    }
    stack
}

fn heavy_expert_route(tokens: usize, experts: usize) -> Vec<f32> {
    let mut lcg = Lcg(77);
    let populated = [0usize, 1, 5, experts - 2];
    (0..tokens)
        .map(|_| {
            let draw = (lcg.next_unit() + 1.0) * 0.5;
            if draw < 0.6 {
                3.0
            } else {
                populated[((draw * 1000.0) as usize) % populated.len()] as f32
            }
        })
        .collect()
}

struct Outputs {
    oracle: Vec<f32>,
    metal: Vec<f32>,
    kernel_keys: Vec<String>,
}

fn run(case: &Case) -> Outputs {
    let Case {
        shape,
        route,
        fuse_gate,
    } = case;
    let (program, output) = program(shape, *fuse_gate);
    let stack = quantized_expert_stack(shape);
    let activation = unit_values(2000, shape.tokens * shape.k, -1.0, 1.0);
    let gate = unit_values(3000, shape.tokens * shape.rows, 0.5, 1.5);
    let mut named = vec![
        (
            "weight",
            QuantizedBlock::Packed {
                codec: Codec::Q8_0,
                bytes: &stack,
            },
        ),
        ("route", QuantizedBlock::Float32(route)),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    if *fuse_gate {
        named.push(("gate", QuantizedBlock::Float32(&gate)));
    }
    let outputs = [output];
    let mut free_buffers = proxima_tensor::cpu::BufferPools::default();
    let mut validated = None;
    let oracle = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &program,
        &[],
        &named,
        &outputs,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu evaluates the gathered q8_0 expert product");
    let plan = omega::plan_named(&program, &[], &named, &outputs, NumericPolicy::default())
        .expect("metal plans the gathered q8_0 expert product");
    let kernel_keys = plan.kernel_keys().expect("plan exposes its kernel keys");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the gathered q8_0 expert product on a real device");
    Outputs {
        oracle: oracle.root().to_vec(),
        metal: metal.root().to_vec(),
        kernel_keys,
    }
}

fn grouped_kernel_ran(keys: &[String]) -> bool {
    keys.iter()
        .any(|key| key.contains("_g10") && key.contains("E_w128"))
}

fn worst_relative_row_error(oracle: &[f32], metal: &[f32], row_width: usize) -> f32 {
    oracle
        .chunks_exact(row_width)
        .zip(metal.chunks_exact(row_width))
        .map(|(want_row, got_row)| {
            let norm = want_row
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let worst = want_row
                .iter()
                .zip(got_row)
                .map(|(want, got)| (want - got).abs())
                .fold(0.0f32, f32::max);
            worst / norm.max(1.0e-6)
        })
        .fold(0.0f32, f32::max)
}

fn assert_grouped_matches_oracle(case: &Case) {
    let outputs = run(case);
    let element_count = case.shape.tokens * case.shape.rows;
    assert_eq!(
        outputs.oracle.len(),
        element_count,
        "degenerate: oracle produced no output"
    );
    assert_eq!(
        outputs.metal.len(),
        element_count,
        "degenerate: metal produced no output"
    );
    assert!(
        grouped_kernel_ran(&outputs.kernel_keys),
        "the expert-grouped kernel was never selected: {:?}",
        outputs.kernel_keys
    );
    let worst = worst_relative_row_error(&outputs.oracle, &outputs.metal, case.shape.rows);
    eprintln!(
        "tokens={} rows={} k={} experts={} fuse_gate={} worst_relative_row_error={worst:e}",
        case.shape.tokens, case.shape.rows, case.shape.k, case.shape.experts, case.fuse_gate
    );
    assert!(
        worst <= RELATIVE_TOLERANCE,
        "grouped result drifted from the f32 oracle by {worst:e} of a row's norm"
    );
}

#[test]
fn heavy_expert_and_empty_experts_match_the_f32_oracle() {
    let shape = Shape {
        tokens: 300,
        rows: 192,
        k: 512,
        experts: 8,
    };
    let route = heavy_expert_route(shape.tokens, shape.experts);
    assert_grouped_matches_oracle(&Case {
        shape,
        route,
        fuse_gate: false,
    });
}

#[test]
fn ragged_rows_and_tokens_match_the_f32_oracle() {
    let shape = Shape {
        tokens: 161,
        rows: 100,
        k: 256,
        experts: 6,
    };
    let route = heavy_expert_route(shape.tokens, shape.experts);
    assert_grouped_matches_oracle(&Case {
        shape,
        route,
        fuse_gate: false,
    });
}

#[test]
fn one_expert_receiving_every_token_spans_every_token_tile() {
    let shape = Shape {
        tokens: 200,
        rows: 128,
        k: 256,
        experts: 4,
    };
    let route = vec![2.0; shape.tokens];
    assert_grouped_matches_oracle(&Case {
        shape,
        route,
        fuse_gate: false,
    });
}

#[test]
fn tokens_cycling_through_every_expert_match_the_f32_oracle() {
    let shape = Shape {
        tokens: 256,
        rows: 128,
        k: 512,
        experts: 8,
    };
    let route = (0..shape.tokens)
        .map(|token| (token % shape.experts) as f32)
        .collect();
    assert_grouped_matches_oracle(&Case {
        shape,
        route,
        fuse_gate: false,
    });
}

#[test]
fn a_fused_epilogue_reads_its_operands_at_the_scattered_token() {
    let shape = Shape {
        tokens: 300,
        rows: 192,
        k: 512,
        experts: 8,
    };
    let route = heavy_expert_route(shape.tokens, shape.experts);
    assert_grouped_matches_oracle(&Case {
        shape,
        route,
        fuse_gate: true,
    });
}

#[test]
fn below_the_tiled_token_minimum_the_dense_gather_kernel_runs() {
    let shape = Shape {
        tokens: MIN_TOKENS_FOR_TILES - 1,
        rows: 128,
        k: 256,
        experts: 8,
    };
    let route = heavy_expert_route(shape.tokens, shape.experts);
    let outputs = run(&Case {
        shape,
        route,
        fuse_gate: false,
    });
    assert!(
        !grouped_kernel_ran(&outputs.kernel_keys),
        "a prefill shorter than the tiled minimum must not take the grouped kernel: {:?}",
        outputs.kernel_keys
    );
}

#[test]
fn an_expert_index_past_the_slab_raises_the_same_error_as_cpu() {
    let shape = Shape {
        tokens: 200,
        rows: 64,
        k: 256,
        experts: 8,
    };
    let mut route = heavy_expert_route(shape.tokens, shape.experts);
    route[137] = shape.experts as f32 + 5.0;
    let (program, output) = program(&shape, false);
    let stack = quantized_expert_stack(&shape);
    let activation = unit_values(2000, shape.tokens * shape.k, -1.0, 1.0);
    let named = [
        (
            "weight",
            QuantizedBlock::Packed {
                codec: Codec::Q8_0,
                bytes: &stack,
            },
        ),
        ("route", QuantizedBlock::Float32(&route)),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let outputs = [output];
    let mut free_buffers = proxima_tensor::cpu::BufferPools::default();
    let mut validated = None;
    let cpu_error = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
        &program,
        &[],
        &named,
        &outputs,
        &mut free_buffers,
        &mut validated,
    )
    .expect_err("cpu rejects the out-of-range expert");
    let plan = omega::plan_named(&program, &[], &named, &outputs, NumericPolicy::default())
        .expect("metal plans the gathered q8_0 expert product");
    let metal_error = omega::execute_plan_named(&plan, &named)
        .expect_err("metal rejects the out-of-range expert instead of clamping it away");
    let TensorError::GatherIndexOutOfRange {
        node: cpu_node,
        extent: cpu_extent,
        ..
    } = cpu_error
    else {
        panic!("cpu raised {cpu_error} instead of GatherIndexOutOfRange");
    };
    let MetalError::Tensor(TensorError::GatherIndexOutOfRange {
        node,
        index,
        extent,
    }) = metal_error
    else {
        panic!("metal raised {metal_error:?} instead of GatherIndexOutOfRange");
    };
    assert_eq!((node, extent), (cpu_node, cpu_extent));
    assert_eq!(
        index,
        shape.experts as i64 + 5,
        "the fault carries the raw fetched index"
    );
}

#[test]
fn control_an_oracle_routed_to_other_experts_disagrees_by_a_gross_factor() {
    let shape = Shape {
        tokens: 300,
        rows: 192,
        k: 512,
        experts: 8,
    };
    let route = heavy_expert_route(shape.tokens, shape.experts);
    let matching = run(&Case {
        shape: Shape { ..shape },
        route: route.clone(),
        fuse_gate: false,
    });
    let rotated: Vec<f32> = route
        .iter()
        .map(|expert| (expert + 1.0) % shape.experts as f32)
        .collect();
    let misrouted = run(&Case {
        shape,
        route: rotated,
        fuse_gate: false,
    });
    let gross = worst_relative_row_error(&misrouted.oracle, &matching.metal, 192);
    assert!(
        gross > RELATIVE_TOLERANCE,
        "a metal result compared against an oracle routed elsewhere must not pass: {gross:e}"
    );
}

/// The bytes of threadgroup memory a compacted expert-grouped kernel declares:
/// exactly the `tg_shared` staging tile (64 rows x 32 depth of `half` weights
/// plus 32 tokens x 32 depth of `float` activations = 8192), the same figure
/// llama.cpp's `kernel_mul_mm_id` declares. Read off the compiled
/// `MTLComputePipelineState`, not derived from the source.
#[cfg(feature = "instrument")]
#[test]
fn the_compacted_pipeline_declares_exactly_the_staging_tile_of_threadgroup_memory() {
    const STAGING_TILE_BYTES: usize = 8192;
    if !omega::sized::GROUPED_GEMM_ROUTE_COMPACTED {
        return;
    }
    let shape = Shape {
        tokens: 300,
        rows: 192,
        k: 512,
        experts: 8,
    };
    let route = heavy_expert_route(shape.tokens, shape.experts);
    let outputs = run(&Case {
        shape,
        route,
        fuse_gate: false,
    });
    assert!(grouped_kernel_ran(&outputs.kernel_keys), "{:?}", outputs.kernel_keys);
    let lengths: Vec<(String, usize)> = omega::pipeline_cache_keys()
        .into_iter()
        .filter(|key| key.contains("_g10") && key.contains("E_w128") && !key.contains("prepass"))
        .filter_map(|key| {
            omega::pipeline_static_threadgroup_memory(&key).map(|length| (key, length))
        })
        .collect();
    assert!(!lengths.is_empty(), "degenerate: no grouped pipeline was compiled");
    for (key, length) in &lengths {
        assert_eq!(*length, STAGING_TILE_BYTES, "{key}");
    }
}

/// Words the IQ4_XS decode may read with a plain `uint` load only when the weight binding's byte origin
/// is a multiple of 4. `GroupedTail` renders both tile loops behind one `if (weight_words_aligned)`;
/// every fixture above binds at an aligned origin, so the guarded loop (the `else`) never ran. These
/// tests bind the same IQ4_XS bytes at origin 0 and at origin 2 of a registered checkpoint mapping
/// (`checkpoint_mapping_offset` binds `address - mapping_base` as the `setBuffer` offset).
///
/// proxima-numeric has no IQ4_XS encoder (`encode.rs:28`). Every bit pattern of scales and nibbles is a
/// valid IQ4_XS block, so the blocks are synthesized from real row values the way
/// `expert_gemm_types_vs_llama.rs::synthesize_iq4_xs` does, with a fixed finite f16 super-scale. The gap:
/// the bytes are structurally valid IQ4_XS, not the output of a production quantizer on a trained tensor.
mod iq4_xs_binding_origin {
    use super::*;
    use proxima_gguf::quant::iq4_xs;
    use proxima_tensor::AlignedBuffer;

    const MISALIGNED_ORIGIN: usize = 2;

    fn synthesize_iq4_xs(row: &[f32], out: &mut [u8]) {
        const HALF_SCALE: [u8; 2] = 0x10AFu16.to_le_bytes();
        for (values, block) in row.as_chunks::<{ iq4_xs::QK_K }>().0.iter().zip(out.as_chunks_mut::<{ iq4_xs::BLOCK_BYTES }>().0.iter_mut()) {
            let byte_of = |value: f32| ((value + 1.0) * 127.5) as u8;
            block[..2].copy_from_slice(&HALF_SCALE);
            let scales_h = u16::from_le_bytes([byte_of(values[0]), byte_of(values[1])]);
            block[2..4].copy_from_slice(&scales_h.to_le_bytes());
            for (slot, value) in block[4..8].iter_mut().zip(&values[2..6]) {
                *slot = byte_of(*value);
            }
            for (slot, pair) in block[8..].iter_mut().zip(values.as_chunks::<2>().0) {
                *slot = (byte_of(pair[0]) & 0x0f) | (byte_of(pair[1]) << 4);
            }
        }
    }

    fn iq4_xs_expert_stack(shape: &Shape) -> Vec<u8> {
        let row_bytes = shape.k / iq4_xs::QK_K * iq4_xs::BLOCK_BYTES;
        let mut stack = vec![0u8; shape.experts * shape.rows * row_bytes];
        let values = unit_values(1000, shape.experts * shape.rows * shape.k, -1.0, 1.0);
        for (row, target) in values.chunks_exact(shape.k).zip(stack.chunks_exact_mut(row_bytes)) {
            synthesize_iq4_xs(row, target);
        }
        stack
    }

    struct Mapping {
        buffer: AlignedBuffer,
    }

    impl Mapping {
        fn holding(regions: &[(usize, &[u8])], total: usize) -> Self {
            let mut buffer = AlignedBuffer::new(total.div_ceil(4), omega::page_size()).expect("page-aligned mapping");
            let base = buffer.as_mut_ptr().cast::<u8>();
            for (origin, bytes) in regions {
                assert!(origin + bytes.len() <= buffer.len() * 4, "region fits the mapping");
                // SAFETY: the region lies inside the allocation (asserted above) and `bytes` is a distinct borrow, so the ranges cannot overlap.
                unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(*origin), bytes.len()) };
            }
            Self { buffer }
        }

        fn all_bytes(&self) -> &[u8] {
            // SAFETY: the allocation is `len * 4` initialized bytes, alive as long as `self`.
            unsafe { core::slice::from_raw_parts(self.buffer.as_ptr().cast::<u8>(), self.buffer.len() * 4) }
        }

        fn region(&self, origin: usize, length: usize) -> &[u8] {
            &self.all_bytes()[origin..origin + length]
        }
    }

    struct Bound {
        output: Vec<f32>,
        oracle: Vec<f32>,
        origin: usize,
        kernel_keys: Vec<String>,
    }

    fn run_at(case: &Case, mapping: &Mapping, origin: usize, length: usize) -> Bound {
        let (program, output) = program(&case.shape, false);
        let weight = mapping.region(origin, length);
        let base = mapping.all_bytes().as_ptr() as usize;
        let activation = unit_values(2000, case.shape.tokens * case.shape.k, -1.0, 1.0);
        let named = [
            ("weight", QuantizedBlock::Packed { codec: Codec::Iq4Xs, bytes: weight }),
            ("route", QuantizedBlock::Float32(&case.route)),
            ("activation", QuantizedBlock::Float32(&activation)),
        ];
        let outputs = [output];
        let mut free_buffers = proxima_tensor::cpu::BufferPools::default();
        let mut validated = None;
        let oracle = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(&program, &[], &named, &outputs, &mut free_buffers, &mut validated)
            .expect("cpu evaluates the gathered iq4_xs expert product");
        let plan = omega::plan_named(&program, &[], &named, &outputs, NumericPolicy::default()).expect("metal plans the gathered iq4_xs expert product");
        let kernel_keys = plan.kernel_keys().expect("plan exposes its kernel keys");
        let metal = omega::execute_plan_named(&plan, &named).expect("metal runs the gathered iq4_xs expert product");
        Bound {
            output: metal.root().to_vec(),
            oracle: oracle.root().to_vec(),
            origin: weight.as_ptr() as usize - base,
            kernel_keys,
        }
    }

    fn emitted_source(shape: &Shape) -> String {
        let (program, output) = program(shape, false);
        let shapes = proxima_tensor::infer(&program, &[]).expect("the program infers");
        let packed: omega::PackedOperands = [(NodeId(0), Codec::Iq4Xs)].into_iter().collect();
        let bound = proxima_tensor::bind(&program, &shapes, &[output], NumericPolicy::default()).expect("the program binds");
        let resolved = bound.iter().find(|op| op.node == output).expect("the reduce node is bound");
        omega::emit(resolved, &packed, NumericPolicy::default()).expect("the grouped kernel emits").source
    }

    fn assert_source_has_both_tile_loops(source: &str) {
        assert!(source.contains("bool weight_words_aligned = (((ulong)weight_bytes) & 3ul) == 0ul;"), "the kernel derives the flag from the bound address");
        assert!(source.contains("if (weight_words_aligned) {"), "the aligned branch is rendered");
        assert!(source.contains("    } else {\n"), "the guarded branch is rendered");
    }

    #[test]
    fn iq4_xs_experts_bound_at_a_non_word_origin_take_the_guarded_loop_and_match_the_aligned_run_bit_for_bit() {
        let shape = Shape { tokens: 161, rows: 100, k: 512, experts: 6 };
        let route = heavy_expert_route(shape.tokens, shape.experts);
        let stack = iq4_xs_expert_stack(&shape);
        let misaligned_origin = stack.len().next_multiple_of(4) + MISALIGNED_ORIGIN;
        let mapping = Mapping::holding(&[(0, &stack), (misaligned_origin, &stack)], misaligned_origin + stack.len());
        omega::backend::register_checkpoint_mapping(mapping.all_bytes());
        let case = Case { shape, route, fuse_gate: false };

        let aligned = run_at(&case, &mapping, 0, stack.len());
        let misaligned = run_at(&case, &mapping, misaligned_origin, stack.len());
        omega::backend::unregister_checkpoint_mapping(mapping.all_bytes());

        assert_eq!(aligned.origin % 4, 0, "the control binds the weights at a word origin");
        assert_eq!(misaligned.origin % 4, MISALIGNED_ORIGIN, "the weights are bound {MISALIGNED_ORIGIN} bytes past a word");
        assert_source_has_both_tile_loops(&emitted_source(&case.shape));
        for run in [&aligned, &misaligned] {
            assert!(grouped_kernel_ran(&run.kernel_keys), "the expert-grouped tiled kernel was never selected: {:?}", run.kernel_keys);
            assert_eq!(run.output.len(), case.shape.tokens * case.shape.rows, "degenerate: metal produced no output");
            let worst = worst_relative_row_error(&run.oracle, &run.output, case.shape.rows);
            assert!(worst <= RELATIVE_TOLERANCE, "origin {} drifted from the f32 oracle by {worst:e} of a row's norm", run.origin);
        }
        let differing = aligned.output.iter().zip(&misaligned.output).filter(|(left, right)| left.to_bits() != right.to_bits()).count();
        assert_eq!(differing, 0, "{differing} of {} outputs differ between the aligned and the misaligned binding", aligned.output.len());
    }
}
