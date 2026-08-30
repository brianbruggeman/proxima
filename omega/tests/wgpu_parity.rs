//! GPU-vs-CPU parity gate for the portable `wgpu`/WGSL backend
//! (`omega::wgpu_driver`, reached through `omega::backend` exactly the way
//! `backend_parity.rs` reaches Metal), run through the SAME
//! `plan_named`/`execute_plan_named` wrapper.
//!
//! # Why this is not `backend_parity.rs`'s own `real_forward_fixture`
//!
//! That fixture (`support::real_forward_fixture`) is a full cached-attention
//! transformer forward: it needs `Op::Elementwise`'s gather form
//! (`embedding_lookup`'s `IndexMap::Computed`) to bind the token embedding
//! table, RoPE, and a causal mask. `omega::wgsl`'s v1 scope is explicitly
//! elementwise + `Keep::Reduce` + `Keep::Scan` with **no gather** (see that
//! module's own doc) — running the real fixture through the wgpu backend
//! would fail on the embedding lookup before ever reaching a matmul.
//!
//! This test instead builds a standalone two-layer MLP
//! (`matmul -> erf -> matmul -> tanh -> cumsum`) that stays entirely inside
//! v1's op set while still exercising every kernel shape that set covers:
//! `Keep::Reduce` (both matmuls, `Add`-reduce over a `Multiply` body — the
//! same shape a real matmul takes), an elementwise `Erf` (the ported
//! polynomial) and `Tanh`, and `Keep::Scan` (a per-row cumulative sum).

#![cfg(all(feature = "cpu", feature = "wgpu-backend"))]
// every expect below runs against data this test just built or a real
// device call; a failure there IS the test failing, not a case to recover.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use omega::backend::{Backend, execute_plan_named, plan_named};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    AxisIndex, AxisTerm, DType, Extent, IndexMap, IndexPattern, Keep, NodeId, Op, QuantizedBlock, Reduce,
    ReduceInit, ScalarOp, TensorError, append, map, projection,
};

const BATCH: u32 = 4;
const IN_FEATURES: u32 = 8;
const HIDDEN: u32 = 16;
const OUT_FEATURES: u32 = 8;

/// `m`/`k`/`n` name the matmul shape for callers' readability; the
/// projections below encode it structurally, so the function itself never
/// reads them.
fn append_matmul(program: &mut Vec<Op>, lhs: NodeId, rhs: NodeId, _m: u32, _k: u32, _n: u32) -> NodeId {
    let product = append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    )
}

/// `matmul(x, w1) -> erf -> matmul(_, w2) -> tanh -> cumsum(last axis)`,
/// entirely within `omega::wgsl`'s v1 op set — see the module doc.
fn two_layer_mlp_fixture() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(BATCH), Extent::Static(IN_FEATURES)],
            name: Some("x".into()),
        },
    );
    let w1 = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(IN_FEATURES), Extent::Static(HIDDEN)],
            name: Some("w1".into()),
        },
    );
    let hidden = append_matmul(&mut program, x, w1, BATCH, IN_FEATURES, HIDDEN);
    let hidden_act = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Erf,
            operands: vec![(hidden, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let w2 = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(HIDDEN), Extent::Static(OUT_FEATURES)],
            name: Some("w2".into()),
        },
    );
    let output = append_matmul(&mut program, hidden_act, w2, BATCH, HIDDEN, OUT_FEATURES);
    let output_act = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Tanh,
            operands: vec![(output, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let cumsum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: output_act,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            keep: Keep::Scan,
            name: None,
        }),
    );
    (program, cumsum)
}

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

#[test]
fn the_two_layer_mlp_runs_on_wgpu_at_cpu_parity() {
    let (program, _root) = two_layer_mlp_fixture();

    let x = random_vec(1, (BATCH * IN_FEATURES) as usize);
    let w1 = random_vec(2, (IN_FEATURES * HIDDEN) as usize);
    let w2 = random_vec(3, (HIDDEN * OUT_FEATURES) as usize);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("x", QuantizedBlock::Float32(&x)),
        ("w1", QuantizedBlock::Float32(&w1)),
        ("w2", QuantizedBlock::Float32(&w2)),
    ];

    let mut cpu_plan = plan_named(Backend::Cpu, &program, &[], &named, &[])
        .expect("omega::backend plans the mlp on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named).expect("omega::backend runs the mlp on cpu");

    let mut wgpu_plan = plan_named(Backend::Wgpu, &program, &[], &named, &[])
        .expect("omega::backend plans the mlp on wgpu");
    let wgpu = execute_plan_named(&mut wgpu_plan, &named).expect("omega::backend runs the mlp on a real device");

    let expected = cpu.root();
    let actual = wgpu.root();
    assert_eq!(
        actual.len(),
        (BATCH * OUT_FEATURES) as usize,
        "degenerate gate: the cumsum output must be one row per batch element"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "wgpu, via the wrapper, produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!("wgpu mlp parity: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}");
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and wgpu arms disagree on the mlp: relative={relative} max_diff={max_diff}"
    );
}

/// `table[ids[s], d]` over iteration space `(s, d)` — the same program shape
/// `omega/tests/metal_parity.rs::embedding_lookup_program` runs against
/// Metal, now exercising [`crate::wgsl`]'s gather bindings (`Indices`/
/// `Fault`) through the portable wgpu driver.
fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(vocab), Extent::Static(dim)],
            name: Some("table".into()),
        },
    );
    let ids = append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: vec![Extent::Static(seq)],
            name: Some("ids".into()),
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: projection(2, &[0]),
        base: IndexPattern {
            iter_rank: 2,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: vec![AxisTerm::projection(1)].into(),
                    offset: 0,
                },
            ],
        },
        gathered_dim: 0,
    };
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: vec![(table, gathered_map)],
            name: None,
        },
    );
    program
}

#[test]
fn embedding_lookup_runs_on_wgpu_at_cpu_parity_for_integer_valued_inputs() {
    let (vocab, dim, seq) = (256usize, 8usize, 4usize);
    let program = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| (value % 97) as f32).collect();
    let ids_data = [3.0f32, 255.0, 12.0, 0.0];
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("table", QuantizedBlock::Float32(&table_data)),
        ("ids", QuantizedBlock::Float32(&ids_data)),
    ];

    let mut cpu_plan =
        plan_named(Backend::Cpu, &program, &[], &named, &[]).expect("omega::backend plans the gather on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named).expect("omega::backend runs the gather on cpu");

    let mut wgpu_plan =
        plan_named(Backend::Wgpu, &program, &[], &named, &[]).expect("omega::backend plans the gather on wgpu");
    let wgpu =
        execute_plan_named(&mut wgpu_plan, &named).expect("omega::backend runs the gather on a real device");

    let expected = cpu.root();
    let actual = wgpu.root();
    assert_eq!(actual.len(), expected.len());
    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    eprintln!("wgpu gather parity: max_diff={max_diff}");
    assert_eq!(
        max_diff, 0.0,
        "gathering integer-valued table rows must round-trip exactly (max abs diff was {max_diff})"
    );
}

#[test]
fn an_out_of_range_gather_index_faults_on_wgpu_the_same_way_it_faults_on_cpu() {
    let (vocab, dim, seq) = (16usize, 4usize, 2usize);
    let program = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| value as f32).collect();
    let ids_data = [0.0f32, 999.0];
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("table", QuantizedBlock::Float32(&table_data)),
        ("ids", QuantizedBlock::Float32(&ids_data)),
    ];

    let mut wgpu_plan =
        plan_named(Backend::Wgpu, &program, &[], &named, &[]).expect("omega::backend plans the gather on wgpu");
    let error =
        execute_plan_named(&mut wgpu_plan, &named).expect_err("an out-of-range gather index must fault, not clamp silently");
    assert!(
        matches!(error, omega::backend::BackendError::Wgpu(omega::WgpuError::Tensor(
            TensorError::GatherIndexOutOfRange { extent, .. }
        )) if extent == vocab as u64),
        "expected a GatherIndexOutOfRange fault against extent {vocab}, got {error:?}"
    );
}

/// `matmul(lhs, rhs)` at `DType::Float16` — mirrors
/// `omega/tests/metal_parity.rs::matmul_parity_is_within_f16_epsilon_of_the_f32_cpu_oracle`'s
/// program shape, now exercising [`crate::wgsl`]'s `enable f16;` compute
/// path (or its named rejection) through the portable wgpu driver.
fn f16_matmul_program(m: u32, k: u32, n: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: Some("lhs".into()),
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: Some("rhs".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float16,
            body: ScalarOp::Multiply,
            operands: vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float16,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("f16_matmul".into()),
        }),
    );
    program
}

/// `EPSILON` mirrors `metal_parity.rs`'s own 5e-3 f16-rounding convention
/// (see that test's doc for the measured-error derivation) — the same
/// order of relative error a 10-bit-mantissa half accumulates over a
/// handful of terms, independent of which backend computes it.
#[test]
fn f16_matmul_runs_on_wgpu_within_the_metal_parity_f16_epsilon_or_names_its_rejection() {
    const EPSILON: f32 = 5e-3;
    let (m, k, n) = (17usize, 23usize, 13usize);
    let f32_program = {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(m as u32), Extent::Static(k as u32)],
                name: Some("lhs".into()),
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k as u32), Extent::Static(n as u32)],
                name: Some("rhs".into()),
            },
        );
        append_matmul(&mut program, lhs, rhs, m as u32, k as u32, n as u32);
        program
    };
    let f16_program = f16_matmul_program(m as u32, k as u32, n as u32);

    let lhs = random_vec(11, m * k);
    let rhs = random_vec(12, k * n);
    let named: Vec<(&str, QuantizedBlock<'_>)> =
        vec![("lhs", QuantizedBlock::Float32(&lhs)), ("rhs", QuantizedBlock::Float32(&rhs))];

    let mut cpu_plan =
        plan_named(Backend::Cpu, &f32_program, &[], &named, &[]).expect("omega::backend plans the f32 oracle on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named).expect("omega::backend runs the f32 oracle on cpu");

    let mut wgpu_plan = plan_named(Backend::Wgpu, &f16_program, &[], &named, &[])
        .expect("omega::backend plans the f16 matmul on wgpu");
    match execute_plan_named(&mut wgpu_plan, &named) {
        Ok(wgpu) => {
            eprintln!("wgpu f16 parity: adapter offers SHADER_F16, computed in half precision");
            let expected = cpu.root();
            let actual = wgpu.root();
            assert_eq!(actual.len(), expected.len());
            assert_eq!(actual.len(), m * n);
            let mut max_diff = 0.0f32;
            for (&got, &want) in actual.iter().zip(expected.iter()) {
                assert!(got.is_finite(), "wgpu f16 matmul produced a non-finite value: {got}");
                max_diff = max_diff.max((got - want).abs() / want.abs().max(f32::MIN_POSITIVE));
            }
            eprintln!("wgpu f16 parity: max_relative_diff={max_diff}");
            assert!(
                max_diff < EPSILON,
                "omega::backend's cpu f32 oracle and wgpu f16 compute disagree beyond the f16 epsilon: \
                 max_relative_diff={max_diff} epsilon={EPSILON}"
            );
        }
        Err(error) => {
            eprintln!(
                "wgpu f16 parity: adapter has no SHADER_F16, named rejection instead of a silent f32 fallback: {error}"
            );
            assert!(
                matches!(
                    error,
                    omega::backend::BackendError::Wgpu(omega::WgpuError::Emit(
                        omega::EmitError::UnsupportedDType { .. }
                    ))
                ),
                "an unsupported adapter must reject with UnsupportedDType, not some other failure: {error:?}"
            );
        }
    }
}

/// `weight[rows, k] * activation[k, 1]` summed over `k` — the same program
/// shape `omega/tests/metal_parity.rs::q4k_matmul_program` runs against a
/// packed Metal weight, now exercising [`crate::wgsl`]'s packed-operand
/// codec table through the portable wgpu driver. `weight_dtype` is
/// `DType::UInt8` for the packed arm (an opaque byte stream `crate::wgsl`
/// never itself interprets as a float) and `DType::Float32` for the
/// dequantized CPU oracle.
fn packed_matmul_program(rows: u32, k: u32, weight_dtype: DType) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: Some("weight".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(k), Extent::Static(1)],
            name: Some("activation".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(projection(3, &[2, 1]))),
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
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("packed_matmul".into()),
        }),
    );
    (program, sum)
}

/// Runs `weight (packed as `block_of`) x activation` on wgpu and against a
/// dequantized-`f32` CPU oracle, asserting relative parity within
/// `tolerance` — one gate shape shared by every `#[test]` below so the five
/// codecs cannot drift on how the claim is checked. `quantize`/`dequantize`
/// are `proxima_gguf::quant::<codec>::{quantize, dequantize}`, the same real
/// codec `crate::wgpu_driver`'s upload path and `crate::wgsl`'s unpack
/// functions are checked against.
#[allow(clippy::too_many_arguments)]
fn assert_packed_codec_parity(
    codec_name: &str,
    block_bytes: usize,
    block_elements: usize,
    quantize: fn(&[f32], &mut [u8]) -> Result<(), proxima_gguf::quant::QuantError>,
    dequantize: fn(&[u8], &mut [f32]) -> Result<(), proxima_gguf::quant::QuantError>,
    to_block: fn(&[u8]) -> QuantizedBlock<'_>,
    tolerance: f32,
) {
    let rows: u32 = 3;
    let blocks_per_row = 2usize;
    let k = (block_elements * blocks_per_row) as u32;

    let activation: Vec<f32> = random_vec(31, k as usize).into_iter().map(|value| value * 4.0 - 2.0).collect();
    let weight_f32: Vec<f32> = random_vec(37, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * block_bytes];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * block_bytes))
    {
        quantize(row_f32, row_blocks).expect("row length is a whole multiple of the codec's block width");
    }
    let mut dequantized: Vec<f32> = vec![0.0; rows as usize * k as usize];
    for (row_blocks, row_f32) in weight_blocks
        .chunks_exact(blocks_per_row * block_bytes)
        .zip(dequantized.chunks_exact_mut(k as usize))
    {
        dequantize(row_blocks, row_f32).expect("a whole number of packed blocks");
    }

    let (packed_program, packed_sum) = packed_matmul_program(rows, k, DType::UInt8);
    let named: Vec<(&str, QuantizedBlock<'_>)> =
        vec![("weight", to_block(&weight_blocks)), ("activation", QuantizedBlock::Float32(&activation))];
    let mut wgpu_plan = plan_named(Backend::Wgpu, &packed_program, &[], &named, &[packed_sum])
        .expect("omega::backend plans the packed matmul on wgpu");
    let wgpu =
        execute_plan_named(&mut wgpu_plan, &named).expect("omega::backend runs the packed matmul on a real device");

    let (f32_program, f32_sum) = packed_matmul_program(rows, k, DType::Float32);
    let f32_named: Vec<(&str, QuantizedBlock<'_>)> =
        vec![("weight", QuantizedBlock::Float32(&dequantized)), ("activation", QuantizedBlock::Float32(&activation))];
    let mut cpu_plan = plan_named(Backend::Cpu, &f32_program, &[], &f32_named, &[f32_sum])
        .expect("omega::backend plans the dequantized oracle on cpu");
    let cpu =
        execute_plan_named(&mut cpu_plan, &f32_named).expect("omega::backend runs the dequantized oracle on cpu");

    let actual = wgpu.root();
    let expected = cpu.root();
    assert_eq!(actual.len(), rows as usize, "degenerate gate: no outputs compared");
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "wgpu {codec_name} matmul produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!("wgpu packed-{codec_name} parity: rows={rows} k={k} max_diff={max_diff} relative={relative}");
    assert!(
        relative < tolerance,
        "wgpu packed-{codec_name} unpack disagrees with the dequantized reference: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn packed_q4k_matmul_matches_the_dequantized_f32_cpu_path_on_wgpu() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_packed_codec_parity("q4_k", BLOCK_BYTES, QK_K, quantize, dequantize, |bytes| QuantizedBlock::Q4K(bytes), 1e-5);
}

#[test]
fn packed_q5k_matmul_matches_the_dequantized_f32_cpu_path_on_wgpu() {
    use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_packed_codec_parity("q5_k", BLOCK_BYTES, QK_K, quantize, dequantize, |bytes| QuantizedBlock::Q5K(bytes), 1e-5);
}

#[test]
fn packed_q6k_matmul_matches_the_dequantized_f32_cpu_path_on_wgpu() {
    use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_packed_codec_parity("q6_k", BLOCK_BYTES, QK_K, quantize, dequantize, |bytes| QuantizedBlock::Q6K(bytes), 1e-5);
}

#[test]
fn packed_q8_0_matmul_matches_the_dequantized_f32_cpu_path_on_wgpu() {
    use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, dequantize, quantize};
    assert_packed_codec_parity("q8_0", BLOCK_BYTES, QK8_0, quantize, dequantize, |bytes| QuantizedBlock::Q8_0(bytes), 1e-5);
}

#[test]
fn packed_q4_0_matmul_matches_the_dequantized_f32_cpu_path_on_wgpu() {
    use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, dequantize, quantize};
    assert_packed_codec_parity("q4_0", BLOCK_BYTES, QK4_0, quantize, dequantize, |bytes| QuantizedBlock::Q4_0(bytes), 1e-5);
}

/// `weight[m, k] * activation[k, n]` summed over `k`, `Add`-reduce over a
/// `Multiply` body — the exact shape [`crate::wgsl::reduce_is_cooperative`]
/// selects for whichever path the local adapter supports (subgroup or
/// serial), run directly through [`omega::wgpu_driver::plan_named`] (not
/// [`omega::backend`]'s wrapper) so this test can read
/// [`omega::wgpu_driver::WgpuPlan::caps`] and report which path actually
/// ran — a silent divergence between the two paths must show up as a
/// numeric disagreement here, not just as an unread code path.
#[test]
fn matmul_runs_on_wgpu_at_cpu_parity_whichever_reduce_path_the_adapter_takes() {
    let (m, k, n) = (11usize, 37usize, 5usize);
    let (program, _root) = {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(m as u32), Extent::Static(k as u32)],
                name: Some("lhs".into()),
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k as u32), Extent::Static(n as u32)],
                name: Some("rhs".into()),
            },
        );
        let root = append_matmul(&mut program, lhs, rhs, m as u32, k as u32, n as u32);
        (program, root)
    };

    let lhs = random_vec(41, m * k);
    let rhs = random_vec(43, k * n);
    let named: Vec<(&str, QuantizedBlock<'_>)> =
        vec![("lhs", QuantizedBlock::Float32(&lhs)), ("rhs", QuantizedBlock::Float32(&rhs))];

    let mut cpu_plan = plan_named(Backend::Cpu, &program, &[], &named, &[]).expect("cpu plans the matmul");
    let cpu = execute_plan_named(&mut cpu_plan, &named).expect("cpu runs the matmul");

    let mut wgpu_plan = omega::wgpu_driver::plan_named(&program, &[], &named, &[])
        .expect("omega::wgpu_driver plans the matmul directly");
    let path = match wgpu_plan.caps().subgroup_size {
        Some(width) => format!("cooperative (subgroup width {width})"),
        None => "serial (no fixed-width subgroup reported)".to_string(),
    };
    eprintln!("wgpu reduce path taken: {path}");
    let wgpu = omega::wgpu_driver::execute_plan_named(&mut wgpu_plan, &named)
        .expect("omega::wgpu_driver runs the matmul on a real device");

    let expected = cpu.root();
    let actual = wgpu.root();
    assert_eq!(actual.len(), m * n);
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "wgpu matmul ({path}) produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!("wgpu matmul parity ({path}): max_diff={max_diff} relative={relative}");
    assert!(
        relative < 1e-4,
        "wgpu matmul ({path}) disagrees with the cpu oracle: relative={relative} max_diff={max_diff}"
    );
}
