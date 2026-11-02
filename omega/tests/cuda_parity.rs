//! GPU-vs-CPU parity gate for the PTX-native CUDA driver
//! (`omega::cuda_driver`, reached through `omega::backend` exactly the way
//! `wgpu_parity.rs` reaches wgpu), run through the SAME
//! `plan_named`/`execute_plan_named` wrapper.
//!
//! Needs a real NVIDIA device and NVRTC on `LD_LIBRARY_PATH` -- see
//! `omega::cuda_driver`'s module doc for both requirements. Every test here
//! mirrors a `wgpu_parity.rs` case one-for-one so the two portable GPU
//! drivers can never silently drift on what "parity" means.

#![cfg(all(feature = "cpu", feature = "cuda-driver"))]
// every expect below runs against data this test just built or a real
// device call; a failure there IS the test failing, not a case to recover.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;
use support::{
    as_named_blocks, qwen3_gqa_qk_norm_forward_fixture, qwen3_gqa_qk_norm_padded_cache_fixture,
    qwen35_partial_rotary_forward_fixture,
    real_forward_fixture,
};

use std::env;

use omega::Codec;
use proxima_gguf::quant::block_layout;
use proxima_primitives::block_layout::BlockLayout;
use omega::backend::{
    BackendError, Engine, GpuDriver, Plan, cuda_kernel_identities, execute_plan_named, plan_named,
};
use proxima_tensor::NumericPolicy;
use proxima_tensor::infer;
use proxima_tensor::spec::{
    input_leaf, qk_norm_cached_forward_program, rmsnorm, rmsnorm_per_head, scalar_constant,
};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, map, projection,
};

const BATCH: u32 = 4;
const IN_FEATURES: u32 = 8;
const HIDDEN: u32 = 16;
const OUT_FEATURES: u32 = 8;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// `m`/`k`/`n` name the matmul shape for callers' readability; the
/// projections below encode it structurally, so the function itself never
/// reads them.
fn append_matmul(
    program: &mut Vec<Op>,
    lhs: NodeId,
    rhs: NodeId,
    _m: u32,
    _k: u32,
    _n: u32,
) -> NodeId {
    let product = append(
        program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
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
            dtype: DType::FLOAT32,
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

fn elementwise_tanh_program(extent: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(extent)],
            name: Some("x".into()),
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Tanh,
            operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
            name: None,
        },
    );
    program
}

#[test]
fn elementwise_tanh_runs_on_cuda_at_cpu_parity() {
    let program = elementwise_tanh_program(97);
    let x = random_vec(5, 97);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![("x", QuantizedBlock::Float32(&x))];

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the elementwise chain on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the elementwise chain on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the elementwise chain on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the elementwise chain on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), expected.len());
    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "cuda produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    eprintln!("cuda elementwise tanh parity: max_diff={max_diff}");
    assert!(
        max_diff < 1e-5,
        "cuda and cpu disagree on tanh: max_diff={max_diff}"
    );
}

fn sum_reduce_program(rows: u32, cols: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(rows), Extent::Static(cols)],
            name: Some("x".into()),
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("row_sum".into()),
        }),
    );
    program
}

#[test]
fn reduce_add_runs_on_cuda_at_cpu_parity_via_the_cooperative_warp_shuffle_path() {
    let (rows, cols) = (13usize, 200usize);
    let program = sum_reduce_program(rows as u32, cols as u32);
    let x = random_vec(7, rows * cols);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![("x", QuantizedBlock::Float32(&x))];

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the row sum on cpu");
    let cpu =
        execute_plan_named(&mut cpu_plan, &named).expect("omega::backend runs the row sum on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the row sum on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the row sum on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), rows);
    assert_eq!(actual.len(), expected.len());
    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "cuda produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs() / want.abs().max(1e-6));
    }
    eprintln!("cuda reduce parity: max_relative_diff={max_diff}");
    assert!(
        max_diff < 1e-4,
        "cuda's warp-shuffle reduce disagrees with the cpu oracle: max_relative_diff={max_diff}"
    );
}

fn cumsum_program(rows: u32, cols: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(rows), Extent::Static(cols)],
            name: Some("x".into()),
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: source,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            keep: Keep::Scan,
            name: Some("cumsum".into()),
        }),
    );
    program
}

#[test]
fn scan_cumsum_runs_on_cuda_at_cpu_parity() {
    let (rows, cols) = (5usize, 41usize);
    let program = cumsum_program(rows as u32, cols as u32);
    let x = random_vec(11, rows * cols);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![("x", QuantizedBlock::Float32(&x))];

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the cumsum on cpu");
    let cpu =
        execute_plan_named(&mut cpu_plan, &named).expect("omega::backend runs the cumsum on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the cumsum on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the cumsum on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), rows * cols);
    assert_eq!(actual.len(), expected.len());
    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(got.is_finite(), "cuda produced a non-finite value: {got}");
        max_diff = max_diff.max((got - want).abs());
    }
    eprintln!("cuda scan parity: max_diff={max_diff}");
    assert!(
        max_diff < 1e-3,
        "cuda's serial scan disagrees with the cpu oracle: max_diff={max_diff}"
    );
}

/// `matmul(x, w1) -> erf -> matmul(_, w2) -> tanh -> cumsum(last axis)` --
/// the same fixture `wgpu_parity.rs::two_layer_mlp_fixture` runs, exercising
/// every kernel shape cuda.rs's v1 op set covers in one program.
fn two_layer_mlp_fixture() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(BATCH), Extent::Static(IN_FEATURES)],
            name: Some("x".into()),
        },
    );
    let w1 = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(IN_FEATURES), Extent::Static(HIDDEN)],
            name: Some("w1".into()),
        },
    );
    let hidden = append_matmul(&mut program, x, w1, BATCH, IN_FEATURES, HIDDEN);
    let hidden_act = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Erf,
            operands: vec![(hidden, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let w2 = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(HIDDEN), Extent::Static(OUT_FEATURES)],
            name: Some("w2".into()),
        },
    );
    let output = append_matmul(&mut program, hidden_act, w2, BATCH, HIDDEN, OUT_FEATURES);
    let output_act = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Tanh,
            operands: vec![(output, IndexMap::Affine(map::projection(2, &[0, 1])))],
            name: None,
        },
    );
    let cumsum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
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

#[test]
fn the_two_layer_mlp_runs_on_cuda_at_cpu_parity() {
    let (program, _root) = two_layer_mlp_fixture();

    let x = random_vec(1, (BATCH * IN_FEATURES) as usize);
    let w1 = random_vec(2, (IN_FEATURES * HIDDEN) as usize);
    let w2 = random_vec(3, (HIDDEN * OUT_FEATURES) as usize);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("x", QuantizedBlock::Float32(&x)),
        ("w1", QuantizedBlock::Float32(&w1)),
        ("w2", QuantizedBlock::Float32(&w2)),
    ];

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the mlp on cpu");
    let cpu =
        execute_plan_named(&mut cpu_plan, &named).expect("omega::backend runs the mlp on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the mlp on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the mlp on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(
        actual.len(),
        (BATCH * OUT_FEATURES) as usize,
        "degenerate gate: the cumsum output must be one row per batch element"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "cuda, via the wrapper, produced a non-finite value: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "cuda mlp parity: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and cuda arms disagree on the mlp: relative={relative} max_diff={max_diff}"
    );
}

/// `table[ids[s], d]` over iteration space `(s, d)` -- the same program shape
/// `wgpu_parity.rs::embedding_lookup_program` runs against wgpu, now
/// exercising `cuda.rs`'s gather bindings (`Indices`/`Fault`) through the
/// PTX-native cuda driver.
fn embedding_lookup_program(vocab: u32, dim: u32, seq: u32) -> Vec<Op> {
    use proxima_tensor::{AxisIndex, AxisTerm, IndexPattern};

    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(vocab), Extent::Static(dim)],
            name: Some("table".into()),
        },
    );
    let ids = append(
        &mut program,
        Op::Input {
            dtype: DType::INT32,
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
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Identity,
            operands: vec![(table, gathered_map)],
            name: None,
        },
    );
    program
}

#[test]
fn embedding_lookup_runs_on_cuda_at_cpu_parity_for_integer_valued_inputs() {
    let (vocab, dim, seq) = (256usize, 8usize, 4usize);
    let program = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| (value % 97) as f32).collect();
    let ids_data = [3.0f32, 255.0, 12.0, 0.0];
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("table", QuantizedBlock::Float32(&table_data)),
        ("ids", QuantizedBlock::Float32(&ids_data)),
    ];

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the gather on cpu");
    let cpu =
        execute_plan_named(&mut cpu_plan, &named).expect("omega::backend runs the gather on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the gather on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the gather on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), expected.len());
    let max_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(&got, &want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    eprintln!("cuda gather parity: max_diff={max_diff}");
    assert_eq!(
        max_diff, 0.0,
        "gathering integer-valued table rows must round-trip exactly (max abs diff was {max_diff})"
    );
}

#[test]
fn an_out_of_range_gather_index_faults_on_cuda_the_same_way_it_faults_on_cpu() {
    let (vocab, dim, seq) = (16usize, 4usize, 2usize);
    let program = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| value as f32).collect();
    let ids_data = [0.0f32, 999.0];
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("table", QuantizedBlock::Float32(&table_data)),
        ("ids", QuantizedBlock::Float32(&ids_data)),
    ];

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the gather on cuda");
    let error = execute_plan_named(&mut cuda_plan, &named)
        .expect_err("an out-of-range gather index must fault, not clamp silently");
    assert!(
        matches!(
            error,
            omega::backend::BackendError::Cuda(omega::CudaDriverError::GatherIndexOutOfRange {
                index: 999,
                ..
            })
        ),
        "expected a GatherIndexOutOfRange fault at index 999, got {error:?}"
    );
}

fn append_embedding_gather(program: &mut Vec<Op>, table: NodeId, ids: NodeId) -> NodeId {
    use proxima_tensor::{AxisIndex, AxisTerm, IndexPattern};

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
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    append(
        program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Identity,
            operands: vec![(table, gathered_map)],
            name: None,
        },
    )
}

fn append_input(program: &mut Vec<Op>, dtype: DType, shape: Vec<Extent>, name: &str) -> NodeId {
    append(
        program,
        Op::Input {
            dtype,
            shape,
            name: Some(name.into()),
        },
    )
}

/// two embedding gathers (`ids_a`, `ids_b` over one `table`), each row-summed by its own reduce node, added
/// into the root; returns the program and the two kernel nodes that own the gathers.
fn two_gather_program(vocab: u32, dim: u32, seq: u32) -> (Vec<Op>, NodeId, NodeId) {
    let mut program = Vec::new();
    let table = append_input(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(vocab), Extent::Static(dim)],
        "table",
    );
    let ids_a = append_input(
        &mut program,
        DType::INT32,
        vec![Extent::Static(seq)],
        "ids_a",
    );
    let ids_b = append_input(
        &mut program,
        DType::INT32,
        vec![Extent::Static(seq)],
        "ids_b",
    );
    let first_gather = append_embedding_gather(&mut program, table, ids_a);
    let first = append_row_sum(&mut program, first_gather);
    let second_gather = append_embedding_gather(&mut program, table, ids_b);
    let second = append_row_sum(&mut program, second_gather);
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            operands: vec![
                (first, IndexMap::Affine(map::projection(1, &[0]))),
                (second, IndexMap::Affine(map::projection(1, &[0]))),
            ],
            name: None,
        },
    );
    (program, first, second)
}

fn append_row_sum(program: &mut Vec<Op>, gathered: NodeId) -> NodeId {
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: gathered,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("row_sum".into()),
        }),
    )
}

fn faulting_node_for_two_gathers(
    engine: Engine,
    driver: Option<GpuDriver>,
    ids_a: &[f32],
    ids_b: &[f32],
) -> (NodeId, NodeId, NodeId) {
    let (vocab, dim, seq) = (16usize, 4usize, 2usize);
    let (program, first, second) = two_gather_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| value as f32).collect();
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("table", QuantizedBlock::Float32(&table_data)),
        ("ids_a", QuantizedBlock::Float32(ids_a)),
        ("ids_b", QuantizedBlock::Float32(ids_b)),
    ];
    let mut plan = plan_named(
        engine,
        driver,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans two gathers");
    let error =
        execute_plan_named(&mut plan, &named).expect_err("an out-of-range gather index must fault");
    match error {
        BackendError::Cuda(omega::CudaDriverError::GatherIndexOutOfRange { node, .. })
        | BackendError::Tensor(proxima_tensor::TensorError::GatherIndexOutOfRange {
            node, ..
        }) => (node, first, second),
        other => panic!("expected GatherIndexOutOfRange, got {other:?}"),
    }
}

#[test]
fn a_fault_in_only_the_second_of_two_gathers_names_the_second_node() {
    let (named_node, first, second) =
        faulting_node_for_two_gathers(Engine::Gpu, Some(GpuDriver::Cuda), &[0.0, 3.0], &[1.0, 999.0]);
    assert_ne!(first, second);
    assert_eq!(named_node, second, "only the second gather faulted");
}

#[test]
fn faults_in_both_gathers_name_the_same_node_on_cuda_as_on_cpu() {
    let (ids_a, ids_b) = ([777.0, 3.0], [1.0, 999.0]);
    let (cpu_node, _, _) = faulting_node_for_two_gathers(Engine::Cpu, None, &ids_a, &ids_b);
    let (cuda_node, _, _) =
        faulting_node_for_two_gathers(Engine::Gpu, Some(GpuDriver::Cuda), &ids_a, &ids_b);
    assert_eq!(
        cuda_node, cpu_node,
        "the node a multi-fault graph names must match the cpu oracle"
    );
}

#[test]
fn a_single_gather_fault_names_the_gather_node() {
    let (vocab, dim, seq) = (16usize, 4usize, 2usize);
    let mut program = Vec::new();
    let table = append_input(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(vocab as u32), Extent::Static(dim as u32)],
        "table",
    );
    let ids = append_input(
        &mut program,
        DType::INT32,
        vec![Extent::Static(seq as u32)],
        "ids",
    );
    let gather = append_embedding_gather(&mut program, table, ids);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| value as f32).collect();
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("table", QuantizedBlock::Float32(&table_data)),
        ("ids", QuantizedBlock::Float32(&[0.0, 999.0])),
    ];
    let mut plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the gather on cuda");
    let error = execute_plan_named(&mut plan, &named).expect_err("must fault");
    assert!(
        matches!(
            error,
            BackendError::Cuda(omega::CudaDriverError::GatherIndexOutOfRange { node, index: 999, .. })
                if node == gather
        ),
        "expected a fault naming {gather:?} at index 999, got {error:?}"
    );
}

/// `weight[rows, k] * activation[k, 1]` summed over `k` -- the same program
/// shape `wgpu_parity.rs::packed_matmul_program` runs against a packed wgpu
/// weight, now exercising `cuda.rs`'s packed-operand codec table through the
/// PTX-native cuda driver.
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
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(k), Extent::Static(1)],
            name: Some("activation".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
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
            dtype: DType::FLOAT32,
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

/// Runs `weight (packed as `block_of`) x activation` on cuda and against a
/// dequantized-`f32` CPU oracle, asserting relative parity within
/// `tolerance` -- one gate shape shared by every `#[test]` below so the five
/// codecs cannot drift on how the claim is checked. `quantize`/`dequantize`
/// are `proxima_gguf::quant::<codec>::{quantize, dequantize}`, the same real
/// codec `cuda.rs`'s unpack functions are checked against.
#[allow(clippy::too_many_arguments)]
fn assert_packed_codec_parity(
    codec_name: &str,
    block_bytes: usize,
    block_elements: usize,
    quantize: fn(&[f32], &mut [u8]) -> Result<(), proxima_gguf::quant::QuantError>,
    dequantize: fn(&[u8], &mut [f32]) -> Result<(), proxima_gguf::quant::QuantError>,
    to_block: fn(&[u8]) -> QuantizedBlock<'_>,
    rows: u32,
    blocks_per_row: usize,
    cuda_policy: NumericPolicy,
    tolerance: f32,
) {
    let k = (block_elements * blocks_per_row) as u32;

    let activation: Vec<f32> = random_vec(31, k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let weight_f32: Vec<f32> = random_vec(37, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();

    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * block_bytes];
    for (row_f32, row_blocks) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * block_bytes))
    {
        quantize(row_f32, row_blocks)
            .expect("row length is a whole multiple of the codec's block width");
    }
    let mut dequantized: Vec<f32> = vec![0.0; rows as usize * k as usize];
    for (row_blocks, row_f32) in weight_blocks
        .chunks_exact(blocks_per_row * block_bytes)
        .zip(dequantized.chunks_exact_mut(k as usize))
    {
        dequantize(row_blocks, row_f32).expect("a whole number of packed blocks");
    }

    let (packed_program, packed_sum) = packed_matmul_program(rows, k, DType::UINT8);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("weight", to_block(&weight_blocks)),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &packed_program,
        &[],
        &named,
        &[packed_sum],
        cuda_policy,
    )
    .expect("omega::backend plans the packed matmul on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the packed matmul on a real cuda device");

    let (f32_program, f32_sum) = packed_matmul_program(rows, k, DType::FLOAT32);
    let f32_named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("weight", QuantizedBlock::Float32(&dequantized)),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &f32_program,
        &[],
        &f32_named,
        &[f32_sum],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the dequantized oracle on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &f32_named)
        .expect("omega::backend runs the dequantized oracle on cpu");

    let actual = cuda.root();
    let expected = cpu.root();
    assert_eq!(
        actual.len(),
        rows as usize,
        "degenerate gate: no outputs compared"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "cuda {codec_name} matmul produced a non-finite value: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "cuda packed-{codec_name} parity: rows={rows} k={k} max_diff={max_diff} relative={relative}"
    );
    assert!(
        relative < tolerance,
        "cuda packed-{codec_name} unpack disagrees with the dequantized reference: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn packed_q4k_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_packed_codec_parity(
        "q4_k",
        BLOCK_BYTES,
        QK_K,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Q4K,
            bytes,
        },
        3,
        2,
        NumericPolicy::default(),
        1e-5,
    );
}

#[test]
fn packed_q5k_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_packed_codec_parity(
        "q5_k",
        BLOCK_BYTES,
        QK_K,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Q5K,
            bytes,
        },
        3,
        2,
        NumericPolicy::default(),
        1e-5,
    );
}

#[test]
fn packed_q6k_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_packed_codec_parity(
        "q6_k",
        BLOCK_BYTES,
        QK_K,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Q6K,
            bytes,
        },
        3,
        2,
        NumericPolicy::default(),
        1e-5,
    );
}

#[test]
fn packed_q6k_matmul_cooperative_reduce_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_packed_codec_parity(
        "q6_k cooperative",
        BLOCK_BYTES,
        QK_K,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Q6K,
            bytes,
        },
        33,
        6,
        NumericPolicy::bit_exact().with_reassociation(true),
        1e-5,
    );
}

#[test]
fn packed_f16_matmul_cooperative_reduce_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::f16::{BLOCK_BYTES, QK_F16, dequantize, quantize};
    assert_packed_codec_parity(
        "f16 cooperative",
        BLOCK_BYTES,
        QK_F16,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Float16,
            bytes,
        },
        33,
        1536,
        NumericPolicy::bit_exact().with_reassociation(true),
        1e-5,
    );
}

#[test]
fn packed_q8_0_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, dequantize, quantize};
    assert_packed_codec_parity(
        "q8_0",
        BLOCK_BYTES,
        QK8_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Q8_0,
            bytes,
        },
        3,
        2,
        NumericPolicy::default(),
        1e-5,
    );
}

#[test]
fn packed_q4_0_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, dequantize, quantize};
    assert_packed_codec_parity(
        "q4_0",
        BLOCK_BYTES,
        QK4_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Q4_0,
            bytes,
        },
        3,
        2,
        NumericPolicy::default(),
        1e-5,
    );
}

#[test]
fn packed_q5_0_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::q5_0::{BLOCK_BYTES, QK5_0, dequantize, quantize};
    assert_packed_codec_parity(
        "q5_0",
        BLOCK_BYTES,
        QK5_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Q5_0,
            bytes,
        },
        3,
        2,
        NumericPolicy::default(),
        1e-5,
    );
}

#[test]
fn packed_q1_0_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::q1_0::{BLOCK_BYTES, QK_Q1_0, dequantize, quantize};
    assert_packed_codec_parity(
        "q1_0",
        BLOCK_BYTES,
        QK_Q1_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Q1_0,
            bytes,
        },
        3,
        2,
        NumericPolicy::default(),
        1e-5,
    );
}

#[test]
fn packed_q2_0_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::q2_0::{BLOCK_BYTES, QK_Q2_0, dequantize, quantize};
    assert_packed_codec_parity(
        "q2_0",
        BLOCK_BYTES,
        QK_Q2_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Q2_0,
            bytes,
        },
        3,
        4,
        NumericPolicy::default(),
        1e-5,
    );
}

#[test]
fn packed_pq2_0_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::pq2_0::{BLOCK_BYTES, QK_PQ2_0, dequantize, quantize};
    assert_packed_codec_parity(
        "pq2_0",
        BLOCK_BYTES,
        QK_PQ2_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Pq2_0,
            bytes,
        },
        3,
        2,
        NumericPolicy::default(),
        1e-5,
    );
}

fn packed_batched_program(
    rows: u32,
    k: u32,
    tokens: Option<u32>,
    weight_dtype: DType,
) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(rows), Extent::Static(k)],
            name: Some("weight".into()),
        },
    );
    let activation_shape = match tokens {
        Some(tokens) => vec![Extent::Static(k), Extent::Static(tokens)],
        None => vec![Extent::Static(k)],
    };
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: activation_shape,
            name: Some("activation".into()),
        },
    );
    let rank = if tokens.is_some() { 3 } else { 2 };
    let (activation_dims, output_dims): (&[u16], &[u16]) = match tokens {
        Some(_) => (&[2, 1], &[0, 1]),
        None => (&[1], &[0]),
    };
    let weight_dims: &[u16] = if tokens.is_some() { &[0, 2] } else { &[0, 1] };
    let all_dims: &[u16] = if tokens.is_some() { &[0, 1, 2] } else { &[0, 1] };
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(rank, weight_dims))),
                (
                    activation,
                    IndexMap::Affine(projection(rank, activation_dims)),
                ),
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
            in_map: IndexMap::Affine(projection(rank, all_dims)),
            out_map: IndexMap::Affine(projection(rank, output_dims)),
            keep: Keep::Reduce,
            name: Some("packed_batched_matmul".into()),
        }),
    );
    (program, sum)
}

#[allow(clippy::too_many_arguments)]
fn run_batched_program(
    engine: Engine,
    driver: Option<GpuDriver>,
    weight: QuantizedBlock<'_>,
    weight_dtype: DType,
    activation: &[f32],
    rows: u32,
    k: u32,
    tokens: Option<u32>,
) -> (Vec<f32>, Option<Vec<String>>) {
    let (program, root) = packed_batched_program(rows, k, tokens, weight_dtype);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("weight", weight),
        ("activation", QuantizedBlock::Float32(activation)),
    ];
    let mut plan = plan_named(
        engine,
        driver,
        &program,
        &[],
        &named,
        &[root],
        NumericPolicy::bit_exact().with_reassociation(true),
    )
    .expect("omega::backend plans the batched packed matmul");
    let identities = cuda_kernel_identities(&plan);
    let output = execute_plan_named(&mut plan, &named)
        .expect("omega::backend runs the batched packed matmul")
        .root()
        .to_vec();
    (output, identities)
}

const BATCHED_TOKEN_COUNTS: [u32; 6] = [1, 2, 7, 8, 66, 130];

/// A prefill of `tokens` rows must equal, row for row, the single-row matvec
/// kernel (bit for bit: the tiled body keeps each row's accumulation order),
/// and the dequantized f32 CPU path within `tolerance`.
#[allow(clippy::too_many_arguments)]
fn assert_batched_prefill_parity(
    codec_name: &str,
    block_bytes: usize,
    block_elements: usize,
    quantize: fn(&[f32], &mut [u8]) -> Result<(), proxima_gguf::quant::QuantError>,
    dequantize: fn(&[u8], &mut [f32]) -> Result<(), proxima_gguf::quant::QuantError>,
    to_block: fn(&[u8]) -> QuantizedBlock<'_>,
    rows: u32,
    blocks_per_row: usize,
) {
    let k = (block_elements * blocks_per_row) as u32;
    let weight_f32: Vec<f32> = random_vec(37, rows as usize * k as usize)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let mut weight_blocks = vec![0u8; rows as usize * blocks_per_row * block_bytes];
    let mut dequantized: Vec<f32> = vec![0.0; rows as usize * k as usize];
    for ((row_f32, row_blocks), row_dequantized) in weight_f32
        .chunks_exact(k as usize)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * block_bytes))
        .zip(dequantized.chunks_exact_mut(k as usize))
    {
        quantize(row_f32, row_blocks).expect("row length is a whole number of blocks");
        dequantize(row_blocks, row_dequantized).expect("a whole number of packed blocks");
    }

    for tokens in BATCHED_TOKEN_COUNTS {
        let activation: Vec<f32> = random_vec(41 + u64::from(tokens), (k * tokens) as usize)
            .into_iter()
            .map(|value| value * 4.0 - 2.0)
            .collect();
        let (tiled, identities) = run_batched_program(
            Engine::Gpu,
            Some(GpuDriver::Cuda),
            to_block(&weight_blocks),
            DType::UINT8,
            &activation,
            rows,
            k,
            Some(tokens),
        );
        let identities = identities.expect("cuda plan exposes its kernels");
        assert_eq!(
            identities.iter().any(|identity| identity.contains("_mma") || identity.contains("_rt8")),
            tokens > 1,
            "{codec_name} at {tokens} tokens: kernel choice is wrong: {identities:?}"
        );
        let (reference, _) = run_batched_program(
            Engine::Cpu,
            None,
            QuantizedBlock::Float32(&dequantized),
            DType::FLOAT32,
            &activation,
            rows,
            k,
            Some(tokens),
        );
        assert_eq!(tiled.len(), rows as usize * tokens as usize);
        let max_magnitude = reference.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let max_diff = tiled
            .iter()
            .zip(&reference)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(
            tiled.iter().all(|value| value.is_finite())
                && max_diff / max_magnitude.max(f32::MIN_POSITIVE) < 1e-5,
            "cuda {codec_name} prefill at {tokens} tokens disagrees with the f32 oracle: max_diff={max_diff}"
        );

        let probe_columns: Vec<u32> = [0, 1, tokens / 2, tokens.saturating_sub(1)]
            .into_iter()
            .filter(|column| *column < tokens)
            .collect();
        for column in probe_columns {
            let column_activation: Vec<f32> = (0..k as usize)
                .map(|kk| activation[kk * tokens as usize + column as usize])
                .collect();
            let (matvec, matvec_identities) = run_batched_program(
                Engine::Gpu,
                Some(GpuDriver::Cuda),
                to_block(&weight_blocks),
                DType::UINT8,
                &column_activation,
                rows,
                k,
                None,
            );
            let matvec_identities = matvec_identities.expect("cuda plan exposes its kernels");
            assert!(
                matvec_identities
                    .iter()
                    .all(|identity| !identity.contains("_mma") && !identity.contains("_rt8")),
                "degenerate control: the matvec took the row-tiled kernel: {matvec_identities:?}"
            );
            for row in 0..rows as usize {
                let got = tiled[row * tokens as usize + column as usize];
                if tokens == 1 {
                    assert_eq!(
                        got.to_bits(),
                        matvec[row].to_bits(),
                        "{codec_name}: row {row} of a single-token prefill differs from the matvec kernel"
                    );
                } else {
                    assert!(
                        (got - matvec[row]).abs() <= 2e-5 * max_magnitude,
                        "{codec_name}: row {row} token {column} of {tokens} is {got}, single-row kernel gives {}",
                        matvec[row]
                    );
                }
            }
        }
    }
}

#[test]
fn batched_q4k_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_batched_prefill_parity(
        "q4_k",
        BLOCK_BYTES,
        QK_K,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Q4K, bytes },
        5,
        2,
    );
}

#[test]
fn batched_q5k_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::q5_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_batched_prefill_parity(
        "q5_k",
        BLOCK_BYTES,
        QK_K,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Q5K, bytes },
        5,
        2,
    );
}

#[test]
fn batched_q6k_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    assert_batched_prefill_parity(
        "q6_k",
        BLOCK_BYTES,
        QK_K,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Q6K, bytes },
        5,
        2,
    );
}

#[test]
fn batched_q8_0_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, dequantize, quantize};
    assert_batched_prefill_parity(
        "q8_0",
        BLOCK_BYTES,
        QK8_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Q8_0, bytes },
        5,
        9,
    );
}

#[test]
fn batched_q4_0_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, dequantize, quantize};
    assert_batched_prefill_parity(
        "q4_0",
        BLOCK_BYTES,
        QK4_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Q4_0, bytes },
        5,
        9,
    );
}

#[test]
fn batched_q5_0_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::q5_0::{BLOCK_BYTES, QK5_0, dequantize, quantize};
    assert_batched_prefill_parity(
        "q5_0",
        BLOCK_BYTES,
        QK5_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Q5_0, bytes },
        5,
        9,
    );
}

#[test]
fn batched_q1_0_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::q1_0::{BLOCK_BYTES, QK_Q1_0, dequantize, quantize};
    assert_batched_prefill_parity(
        "q1_0",
        BLOCK_BYTES,
        QK_Q1_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Q1_0, bytes },
        5,
        5,
    );
}

#[test]
fn batched_q2_0_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::q2_0::{BLOCK_BYTES, QK_Q2_0, dequantize, quantize};
    assert_batched_prefill_parity(
        "q2_0",
        BLOCK_BYTES,
        QK_Q2_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Q2_0, bytes },
        5,
        5,
    );
}

#[test]
fn batched_pq2_0_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::pq2_0::{BLOCK_BYTES, QK_PQ2_0, dequantize, quantize};
    assert_batched_prefill_parity(
        "pq2_0",
        BLOCK_BYTES,
        QK_PQ2_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Pq2_0, bytes },
        5,
        5,
    );
}

#[test]
fn batched_f16_prefill_matches_the_matvec_and_the_f32_oracle_on_cuda() {
    use proxima_gguf::quant::f16::{BLOCK_BYTES, QK_F16, dequantize, quantize};
    assert_batched_prefill_parity(
        "f16",
        BLOCK_BYTES,
        QK_F16,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed { codec: Codec::Float16, bytes },
        5,
        200,
    );
}

#[test]
fn packed_ptq1_0_matmul_matches_the_dequantized_f32_cpu_path_on_cuda() {
    use proxima_gguf::quant::ptq1_0::{BLOCK_BYTES, QK_PTQ1_0, dequantize, quantize};
    assert_packed_codec_parity(
        "ptq1_0",
        BLOCK_BYTES,
        QK_PTQ1_0,
        quantize,
        dequantize,
        |bytes| QuantizedBlock::Packed {
            codec: Codec::Ptq1_0,
            bytes,
        },
        3,
        3,
        NumericPolicy::default(),
        1e-5,
    );
}

/// Snaps each group of 32 onto the levels `amax / 127 * level` that an int8 activation round
/// trip reproduces exactly, so the f32 oracle can hold the quantized kernel to a tight bound.
fn on_the_int8_grid(values: &[f32]) -> Vec<f32> {
    values
        .chunks(32)
        .flat_map(|group| {
            let scale = group.iter().fold(0.0f32, |amax, value| amax.max(value.abs())) / 127.0;
            group.iter().map(move |value| (value / scale).round() * scale)
        })
        .collect()
}

/// Random bytes in every code position, so every digit pattern a stored byte can hold is read,
/// and a finite power-of-two f16 in the scale field so no NaN hides a wrong code.
fn arbitrary_blocks(layout: &BlockLayout, blocks: usize, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg(seed);
    let block_bytes = usize::from(layout.block_bytes);
    let scale_at = usize::from(layout.scale.byte_offset);
    let mut bytes: Vec<u8> = (0..blocks * block_bytes).map(|_| (lcg.next_unit() * 256.0) as u8).collect();
    for block in bytes.chunks_exact_mut(block_bytes) {
        let exponent = 9 + (lcg.next_unit() * 4.0) as u16;
        block[scale_at..scale_at + 2].copy_from_slice(&(exponent << 10).to_le_bytes());
    }
    bytes
}

/// The contract of the int8 lowering: for a layout it accepts, the dp4a matvec over any block
/// bytes equals the CPU block-layout walker's decode dotted with the same int8-grid activation.
#[proxima::test]
#[case::ptq1_0_radix_digits_in_three_regions(Codec::Ptq1_0)]
#[case::pq2_0_two_bit_runs(Codec::Pq2_0)]
#[case::q2_0_two_bit_runs(Codec::Q2_0)]
#[case::q1_0_one_bit_runs(Codec::Q1_0)]
#[case::q8_0_signed_bytes(Codec::Q8_0)]
async fn staged_int8_matvec_over_arbitrary_block_bytes_equals_the_block_layout_walker_on_cuda(#[case] codec: Codec) {
    let layout = codec.block_layout().expect("a block codec");
    assert_eq!(layout.scale.layout.width(), 16, "arbitrary_blocks writes an f16 scale");
    let (rows, blocks_per_row) = (40usize, 10usize);
    let k = usize::from(layout.elements) * blocks_per_row;
    let weight = arbitrary_blocks(&layout, rows * blocks_per_row, 41);
    let activation = on_the_int8_grid(&random_vec(43, k).into_iter().map(|value| value * 4.0 - 2.0).collect::<Vec<_>>());
    let mut decoded = vec![0.0f32; rows * k];
    block_layout::dequantize(&layout, &weight, &mut decoded).expect("whole blocks");
    let expected: Vec<f64> = decoded
        .chunks_exact(k)
        .map(|row| row.iter().zip(&activation).map(|(code, value)| f64::from(*code) * f64::from(*value)).sum())
        .collect();

    let (program, sum) = packed_matmul_program(rows as u32, k as u32, DType::UINT8);
    let named = vec![
        ("weight", QuantizedBlock::Packed { codec, bytes: &weight }),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let policy = NumericPolicy::default().with_activation_quantization(true);
    let mut plan = plan_named(Engine::Gpu, Some(GpuDriver::Cuda), &program, &[], &named, &[sum], policy)
        .expect("the packed matvec plans on cuda");
    let identities = cuda_kernel_identities(&plan).expect("a cuda plan");
    assert!(
        identities.iter().any(|identity| identity.contains("_smv")),
        "{codec:?} must lower to the staged dp4a matvec, got {:?}",
        identities.iter().map(|identity| identity.lines().next()).collect::<Vec<_>>()
    );
    let actual = execute_plan_named(&mut plan, &named).expect("the matvec runs on cuda");

    let magnitude = expected.iter().fold(0.0f64, |largest, value| largest.max(value.abs()));
    let worst = actual
        .root()
        .iter()
        .zip(&expected)
        .map(|(got, want)| (f64::from(*got) - want).abs())
        .fold(0.0f64, f64::max);
    assert_eq!(actual.root().len(), rows, "every row is compared");
    assert!(worst / magnitude < 1e-5, "{codec:?} staged dp4a disagrees with the walker: worst={worst} magnitude={magnitude}");
}

/// `cuda.rs`'s `kernel_signature` declares the OUTPUT pointer itself as
/// `__half*` for a `DType::FLOAT16` op (unlike `crate::wgsl`, which always
/// casts an f16 compute result back to an `array<f32>` storage write) -- see
/// `omega::cuda_driver`'s module doc. v1 does not allocate/read back that
/// narrower buffer, so this proves the documented rejection actually fires
/// rather than silently misreading a `__half`-packed buffer as `f32`.
fn f16_dtype_reduce_program(m: u32, k: u32, n: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let lhs = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(m), Extent::Static(k)],
            name: Some("lhs".into()),
        },
    );
    let rhs = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(k), Extent::Static(n)],
            name: Some("rhs".into()),
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT16,
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
            dtype: DType::FLOAT16,
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

#[test]
fn a_float16_dtyped_op_names_its_rejection_on_cuda_instead_of_misreading_a_half_buffer() {
    let (m, k, n) = (5usize, 7usize, 3usize);
    let program = f16_dtype_reduce_program(m as u32, k as u32, n as u32);
    let lhs = random_vec(51, m * k);
    let rhs = random_vec(53, k * n);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("lhs", QuantizedBlock::Float32(&lhs)),
        ("rhs", QuantizedBlock::Float32(&rhs)),
    ];

    let error = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .err()
    .expect("a Float16-dtyped op must be rejected at plan time, not misread through an f32 buffer");
    assert!(
        matches!(
            error,
            BackendError::Cuda(omega::CudaDriverError::UnsupportedDtype {
                dtype: DType::FLOAT16,
                ..
            })
        ),
        "expected UnsupportedDtype naming Float16, got {error:?}"
    );
}

/// `support::real_forward_fixture` -- the same full cached-attention
/// transformer forward `wgpu_parity.rs::the_full_mistral_cached_forward_runs_on_wgpu_at_cpu_parity`
/// runs against wgpu, now run against the PTX-native cuda driver through the
/// same `omega::backend` wrapper. `cuda_driver::plan`'s own
/// `fuse_cached_attention: false` (mirroring `wgpu_driver::plan`) means this
/// fixture's attention ops lower to plain elementwise/reduce nodes long
/// before `emit_cuda` ever sees them, so `BoundOpKind::CachedAttention`'s
/// `CudaUnsupportedOpKind` rejection is never reached by this test.
#[test]
fn the_full_mistral_cached_forward_runs_on_cuda_at_cpu_parity() {
    const VOCAB: usize = 64;

    let (program, symbols, roots, owned) = real_forward_fixture();
    let named = as_named_blocks(&owned);

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the real forward on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the real forward on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the real forward on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the real forward on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(
        actual.len(),
        VOCAB,
        "degenerate gate: logits must be one row of the vocabulary"
    );
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "cuda, via the wrapper, produced a non-finite logit: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "cuda real forward parity: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and cuda arms disagree on the real forward: relative={relative} max_diff={max_diff}"
    );
}

/// `support::qwen3_gqa_qk_norm_forward_fixture` -- the real Qwen3 shape's
/// two distinguishing features over Mistral: GQA (already exercised by
/// `real_forward_fixture`) plus per-head `q_norm`/`k_norm`, a SECOND
/// broadcast-reduce epilogue site (RMSNorm's own row-mean-square fold is the
/// first, already covered by [`the_full_mistral_cached_forward_runs_on_cuda_at_cpu_parity`])
/// with a narrower per-head reduction extent than the row-wide RMSNorm case.
/// Needs `reduce-epilogue-fusion` to actually bind a non-identity
/// `epilogue_broadcast_axes` at all -- run with
/// `--features cpu,cuda-backend,reduce-epilogue-fusion`.
#[test]
fn the_qwen3_gqa_qk_norm_forward_runs_on_cuda_at_cpu_parity() {
    let (program, symbols, roots, owned) = qwen3_gqa_qk_norm_forward_fixture(1, 0);
    let named = as_named_blocks(&owned);

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the qk-norm forward on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the qk-norm forward on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the qk-norm forward on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the qk-norm forward on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "cuda, via the wrapper, produced a non-finite value: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "cuda qk-norm forward parity: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and cuda arms disagree on the qk-norm forward: relative={relative} max_diff={max_diff}"
    );
}

/// [`the_qwen3_gqa_qk_norm_forward_runs_on_cuda_at_cpu_parity`]'s own fixture
/// pins `HEAD_DIM = 16`, SMALLER than [`omega::WARP_SIZE`] (32) -- every lane
/// of the cooperative reduce's warp never loops more than once, so this is
/// the ONE shape neither that test nor `the_full_mistral_cached_forward_runs_
/// on_cuda_at_cpu_parity` (a per-ROW `[batch, embedding=64]` RMSNorm, not a
/// per-HEAD one) actually exercises for the broadcast-write loop:
/// `reduction_total > WARP_SIZE` combined with the per-head (not per-row)
/// axis topology real Qwen3 checkpoints use (`query_heads`/`kv_heads` as
/// EXTRA, non-reduced axes alongside the reduced `head_dim`). Built directly
/// from `qk_norm_cached_forward_program` (bypassing `support`'s fixed-width
/// fixture) at `HEAD_DIM = 64`, matching a real checkpoint's own per-head
/// width, so a bug that only appears once a lane strides through the
/// reduction more than once cannot hide behind the narrower fixture.
#[test]
fn the_qwen3_gqa_qk_norm_forward_runs_on_cuda_at_cpu_parity_with_head_dim_wider_than_a_warp() {
    const VOCAB: u32 = 64;
    const EMBEDDING: u32 = 256;
    const FEED_FORWARD: u32 = 512;
    const QUERY_HEADS: u32 = 4;
    const KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 64;
    const LAYERS: u32 = 2;

    let (program, logits_root, cache_roots) = qk_norm_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
    )
    .expect("the wide-head-dim qwen3 gqa+qk_norm forward program builds");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    let symbols = vec![1u64, 0u64];
    let shapes =
        infer(&program, &symbols).expect("the wide-head-dim qwen3 gqa+qk_norm forward infers");

    let mut owned: Vec<(String, Vec<f32>)> = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let node = NodeId(position as u32);
        let count: usize = shapes
            .of(node)
            .iter()
            .map(|extent| *extent as usize)
            .product();
        let name = name
            .clone()
            .expect("every block input in this program is named");
        let data = if name == "ids" {
            vec![3.0f32; count]
        } else if name == "eps" {
            vec![1e-5f32; count]
        } else if name == "cached_len" {
            vec![0.0f32]
        } else {
            random_vec(position as u64 + 1, count)
        };
        owned.push((name, data));
    }
    let named = as_named_blocks(&owned);

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the wide-head-dim qk-norm forward on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the wide-head-dim qk-norm forward on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the wide-head-dim qk-norm forward on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the wide-head-dim qk-norm forward on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "cuda, via the wrapper, produced a non-finite value: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "cuda wide-head-dim qk-norm forward parity: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and cuda arms disagree on the wide-head-dim qk-norm forward: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn the_qwen3_multi_token_prefill_runs_on_cuda_at_cpu_parity() {
    const VOCAB: u32 = 64;
    const EMBEDDING: u32 = 256;
    const FEED_FORWARD: u32 = 512;
    const QUERY_HEADS: u32 = 4;
    const KV_HEADS: u32 = 2;
    const HEAD_DIM: u32 = 64;
    const LAYERS: u32 = 2;
    const NEW_COUNT: u64 = 5;

    let (program, logits_root, cache_roots) = qk_norm_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
    )
    .expect("the multi-token prefill qwen3 program builds");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    let symbols = vec![NEW_COUNT, 0u64];
    let shapes = infer(&program, &symbols).expect("the multi-token prefill qwen3 program infers");

    let mut owned: Vec<(String, Vec<f32>)> = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let node = NodeId(position as u32);
        let count: usize = shapes
            .of(node)
            .iter()
            .map(|extent| *extent as usize)
            .product();
        let name = name
            .clone()
            .expect("every block input in this program is named");
        let data = if name == "ids" {
            (0..count)
                .map(|index| (index % VOCAB as usize) as f32)
                .collect()
        } else if name == "eps" {
            vec![1e-5f32; count]
        } else if name == "cached_len" {
            vec![0.0f32]
        } else {
            random_vec(position as u64 + 1, count)
        };
        owned.push((name, data));
    }
    let named = as_named_blocks(&owned);

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the multi-token prefill on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the multi-token prefill on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the multi-token prefill on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the multi-token prefill on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "cuda, via the wrapper, produced a non-finite value: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "cuda multi-token prefill parity: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and cuda arms disagree on the multi-token prefill: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn the_qwen3_real_scale_forward_runs_on_cuda_at_cpu_parity() {
    const VOCAB: u32 = 151936;
    const EMBEDDING: u32 = 1024;
    const FEED_FORWARD: u32 = 3072;
    const QUERY_HEADS: u32 = 16;
    const KV_HEADS: u32 = 8;
    const HEAD_DIM: u32 = 128;
    const LAYERS: u32 = 28;
    const NEW_COUNT: u64 = 5;

    let (program, logits_root, cache_roots) = qk_norm_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        LAYERS,
    )
    .expect("the real-scale qwen3 program builds");

    let mut roots = vec![logits_root];
    for (even, odd, value) in &cache_roots {
        roots.push(*even);
        roots.push(*odd);
        roots.push(*value);
    }

    let symbols = vec![NEW_COUNT, 0u64];
    let shapes = infer(&program, &symbols).expect("the real-scale qwen3 program infers");

    let mut owned: Vec<(String, Vec<f32>)> = Vec::new();
    for (position, op) in program.iter().enumerate() {
        let Op::Input { name, .. } = op else { continue };
        let node = NodeId(position as u32);
        let count: usize = shapes
            .of(node)
            .iter()
            .map(|extent| *extent as usize)
            .product();
        let name = name
            .clone()
            .expect("every block input in this program is named");
        let data = if name == "ids" {
            (0..count)
                .map(|index| (index % VOCAB as usize) as f32)
                .collect()
        } else if name == "eps" {
            vec![1e-6f32; count]
        } else if name == "cached_len" {
            vec![0.0f32]
        } else {
            random_vec(position as u64 + 1, count)
        };
        owned.push((name, data));
    }
    let named = as_named_blocks(&owned);

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the real-scale forward on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the real-scale forward on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the real-scale forward on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the real-scale forward on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    let mut worst_index = 0usize;
    for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            got.is_finite(),
            "cuda, via the wrapper, produced a non-finite value at index {index}: {got}"
        );
        if (got - want).abs() > max_diff {
            max_diff = (got - want).abs();
            worst_index = index;
        }
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "cuda real-scale forward parity: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative} worst_index={worst_index} cpu_at_worst={} cuda_at_worst={}",
        expected[worst_index], actual[worst_index]
    );
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and cuda arms disagree on the real-scale forward: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn the_qwen3_gqa_qk_norm_decode_step_after_a_prefill_runs_on_cuda_at_cpu_parity() {
    let (program, symbols, roots, owned) = qwen3_gqa_qk_norm_forward_fixture(1, 5);
    let named = as_named_blocks(&owned);

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the post-prefill decode step on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the post-prefill decode step on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the post-prefill decode step on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the post-prefill decode step on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), expected.len());

    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "cuda, via the wrapper, produced a non-finite value: {got}"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "cuda post-prefill decode step parity: max_diff={max_diff} max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "omega::backend's cpu and cuda arms disagree on the post-prefill decode step: relative={relative} max_diff={max_diff}"
    );
}

#[test]
fn embedding_lookup_runs_on_cuda_at_cpu_parity_for_int32_token_blocks() {
    let (vocab, dim, seq) = (256usize, 8usize, 4usize);
    let program = embedding_lookup_program(vocab as u32, dim as u32, seq as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| (value % 97) as f32).collect();
    let ids_data = [3i32, 255, 12, 0];
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("table", QuantizedBlock::Float32(&table_data)),
        ("ids", QuantizedBlock::Int32(&ids_data)),
    ];

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the int32 gather on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the int32 gather on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the int32 gather on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend uploads an int32 token block to a real cuda device");

    assert_eq!(cuda.root().len(), seq * dim);
    assert_eq!(
        cuda.root(),
        cpu.root(),
        "int32 token ids must gather the same table rows on cuda as on cpu"
    );
}

fn int32_route_gather_program(vocab: u32, dim: u32, seq: u32, candidates: u32) -> Vec<Op> {
    use proxima_tensor::{AxisIndex, AxisTerm, IndexPattern};

    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(vocab), Extent::Static(dim)],
            name: Some("table".into()),
        },
    );
    let scored = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(seq), Extent::Static(candidates)],
            name: Some("candidates".into()),
        },
    );
    let route = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::INT32,
            body: ScalarOp::Maximum,
            init: ReduceInit::NegativeInfinity,
            operand: scored,
            in_map: IndexMap::Affine(map::projection(2, &[0, 1])),
            out_map: IndexMap::Affine(map::projection(2, &[0])),
            keep: Keep::Reduce,
            name: Some("route".into()),
        }),
    );
    let gathered_map = IndexMap::Computed {
        indices: route,
        index_map: projection(2, &[0]),
        base: IndexPattern {
            iter_rank: 2,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: vec![AxisTerm::projection(1)].into(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Identity,
            operands: vec![(table, gathered_map)],
            name: None,
        },
    );
    program
}

#[test]
fn int32_typed_route_reduce_runs_on_cuda_with_gathered_rows_equal_to_cpu() {
    let (vocab, dim, seq, candidates) = (256usize, 8usize, 5usize, 8usize);
    let program =
        int32_route_gather_program(vocab as u32, dim as u32, seq as u32, candidates as u32);
    let table_data: Vec<f32> = (0..vocab * dim).map(|value| (value % 97) as f32).collect();
    let candidate_data: Vec<f32> = (0..seq * candidates)
        .map(|value| ((value * 37 + 11) % vocab) as f32)
        .collect();
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("table", QuantizedBlock::Float32(&table_data)),
        ("candidates", QuantizedBlock::Float32(&candidate_data)),
    ];

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the int32 route gather on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the int32 route gather on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans an int32-typed route reduce on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs an int32-typed route reduce on a real cuda device");

    assert_eq!(cuda.root().len(), seq * dim);
    assert_eq!(
        cuda.root(),
        cpu.root(),
        "rows gathered by an int32-typed route reduce must be element-equal on cuda and cpu"
    );
}

fn gathered_matvec_reduce_program(experts: u32, rows: u32, width: u32, tokens: u32) -> Vec<Op> {
    use proxima_tensor::{AxisIndex, AxisTerm, IndexPattern};

    let mut program = Vec::new();
    let table = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![
                Extent::Static(experts),
                Extent::Static(rows),
                Extent::Static(width),
            ],
            name: Some("experts".into()),
        },
    );
    let route = append(
        &mut program,
        Op::Input {
            dtype: DType::INT32,
            shape: vec![Extent::Static(tokens)],
            name: Some("route".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(tokens), Extent::Static(width)],
            name: Some("activation".into()),
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: route,
        index_map: projection(3, &[0]),
        base: IndexPattern {
            iter_rank: 3,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: vec![AxisTerm::projection(1)].into(),
                    offset: 0,
                    len: None,
                },
                AxisIndex {
                    terms: vec![AxisTerm::projection(2)].into(),
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
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (table, gathered_map),
                (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
            ],
            name: None,
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("gathered_matvec".into()),
        }),
    );
    program
}

#[test]
fn gathered_matvec_reduce_runs_on_cuda_cooperative_at_cpu_parity() {
    let (experts, rows, width, tokens) = (8usize, 64usize, 256usize, 5usize);
    let program =
        gathered_matvec_reduce_program(experts as u32, rows as u32, width as u32, tokens as u32);
    let expert_data: Vec<f32> = (0..experts * rows * width)
        .map(|value| (value % 7) as f32)
        .collect();
    let route_data = [3i32, 0, 7, 2, 5];
    let activation_data: Vec<f32> = (0..tokens * width)
        .map(|value| (value % 5) as f32)
        .collect();
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("experts", QuantizedBlock::Float32(&expert_data)),
        ("route", QuantizedBlock::Int32(&route_data)),
        ("activation", QuantizedBlock::Float32(&activation_data)),
    ];

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::llama_relaxed(),
    )
    .expect("omega::backend plans the gathered matvec reduce on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the gathered matvec reduce on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::llama_relaxed(),
    )
    .expect("omega::backend plans the gathered matvec reduce on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the gathered matvec reduce on a real cuda device");

    let expected = cpu.root();
    assert_eq!(
        expected.len(),
        tokens * rows,
        "the gathered matvec must produce one output per (token, row)"
    );
    assert!(
        expected.iter().any(|&value| value != 0.0),
        "an all-zero cpu result would make parity vacuous"
    );
    assert_eq!(
        cuda.root(),
        expected,
        "a gathered matvec reduced cooperatively on cuda must equal the cpu result for integer-valued inputs"
    );
}

#[test]
fn an_out_of_range_gathered_matvec_index_faults_on_cuda_cooperative_path() {
    let (experts, rows, width, tokens) = (8usize, 64usize, 256usize, 5usize);
    let program =
        gathered_matvec_reduce_program(experts as u32, rows as u32, width as u32, tokens as u32);
    let expert_data: Vec<f32> = (0..experts * rows * width)
        .map(|value| (value % 7) as f32)
        .collect();
    let route_data = [3i32, 0, 999, 2, 5];
    let activation_data: Vec<f32> = (0..tokens * width)
        .map(|value| (value % 5) as f32)
        .collect();
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("experts", QuantizedBlock::Float32(&expert_data)),
        ("route", QuantizedBlock::Int32(&route_data)),
        ("activation", QuantizedBlock::Float32(&activation_data)),
    ];

    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::llama_relaxed(),
    )
    .expect("omega::backend plans the gathered matvec reduce on cpu");
    let cpu_error = execute_plan_named(&mut cpu_plan, &named)
        .expect_err("cpu must fault on an out-of-range gathered matvec index");
    let cpu_node = match cpu_error {
        BackendError::Tensor(proxima_tensor::TensorError::GatherIndexOutOfRange {
            node,
            index: 999,
            ..
        }) => node,
        other => panic!("expected a cpu GatherIndexOutOfRange at index 999, got {other:?}"),
    };

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::llama_relaxed(),
    )
    .expect("omega::backend plans the gathered matvec reduce on cuda");
    let cuda_error = execute_plan_named(&mut cuda_plan, &named)
        .expect_err("an out-of-range gathered matvec index must fault on cuda, not clamp");
    match cuda_error {
        BackendError::Cuda(omega::CudaDriverError::GatherIndexOutOfRange {
            node,
            index: 999,
            ..
        }) => assert_eq!(
            node, cpu_node,
            "cuda must name the same gathered node as the cpu oracle"
        ),
        other => panic!("expected a cuda GatherIndexOutOfRange at index 999, got {other:?}"),
    }
}

#[cfg(feature = "moe-topk-fusion")]
mod moe_topk {
    use super::*;
    use moe_test_json as serde_json;
    use proxima_tensor::spec::{
        Activation, ExpertGatingFunc, MoeFfnSpec, MoeProjectionStrategy, MoeRouter, append_moe_ffn,
        input_leaf, scalar_constant,
    };

    const GRANITE_EXPERT_COUNT: u32 = 32;
    const GRANITE_TOP_K: u32 = 8;
    const GRANITE_ROUTER_LOGITS: &str = include_str!("fixtures/granite_router_logits.jsonl");

    struct MoeRoutingFixture {
        program: Vec<Op>,
        outputs: Vec<NodeId>,
        named: Vec<(&'static str, Vec<f32>)>,
    }

    fn moe_routing_fixture(
        logits: &[f32],
        seq: u32,
        expert_count: u32,
        top_k: u32,
    ) -> MoeRoutingFixture {
        let embedding = 16u32;
        let expert_hidden = 8u32;
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(seq), Extent::Static(embedding)],
            "x",
        );
        let router_logits = input_leaf(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(seq), Extent::Static(expert_count)],
            "router_logits",
        );
        let slab = |program: &mut Vec<Op>, name: &str, rows: u32, cols: u32| {
            input_leaf(
                program,
                DType::FLOAT32,
                vec![
                    Extent::Static(expert_count),
                    Extent::Static(rows),
                    Extent::Static(cols),
                ],
                name,
            )
        };
        let expert_w_gate = slab(&mut program, "expert_w_gate", embedding, expert_hidden);
        let expert_w_up = slab(&mut program, "expert_w_up", embedding, expert_hidden);
        let expert_w_down = slab(&mut program, "expert_w_down", expert_hidden, embedding);
        let ones = scalar_constant(&mut program, 1.0);
        let spec = MoeFfnSpec {
            router: MoeRouter::Logits(router_logits),
            expert_w_gate,
            expert_w_up,
            expert_w_down,
            expert_count,
            expert_used_count: top_k,
            ones,
            gating: ExpertGatingFunc::Softmax,
            expert_bias: None,
            expert_scale: None,
            activation: Activation::Silu,
            strategy: MoeProjectionStrategy::PerRoute,
        };
        let (routed, site) =
            append_moe_ffn(&mut program, 0, x, &spec).expect("routed moe ffn lowers");
        let _ = routed;
        let mut outputs = Vec::new();
        outputs.extend(site.selected.iter().copied());
        outputs.extend(site.weights.iter().copied());
        let experts = expert_count as usize;
        let named = vec![
            ("x", random_vec(31, seq as usize * embedding as usize)),
            ("router_logits", logits.to_vec()),
            (
                "expert_w_gate",
                random_vec(32, experts * (embedding * expert_hidden) as usize),
            ),
            (
                "expert_w_up",
                random_vec(33, experts * (embedding * expert_hidden) as usize),
            ),
            (
                "expert_w_down",
                random_vec(34, experts * (expert_hidden * embedding) as usize),
            ),
        ];
        MoeRoutingFixture {
            program,
            outputs,
            named,
        }
    }

    fn run_moe_routing(
        engine: Engine,
        driver: Option<GpuDriver>,
        fixture: &MoeRoutingFixture,
    ) -> Vec<Vec<f32>> {
        let named: Vec<(&str, QuantizedBlock<'_>)> = fixture
            .named
            .iter()
            .map(|(name, values)| (*name, QuantizedBlock::Float32(values.as_slice())))
            .collect();
        let mut plan = plan_named(
            engine,
            driver,
            &fixture.program,
            &[],
            &named,
            &fixture.outputs,
            NumericPolicy::default(),
        )
        .expect("omega::backend plans the routed moe ffn");
        let evaluated =
            execute_plan_named(&mut plan, &named).expect("omega::backend runs the routed moe ffn");
        fixture
            .outputs
            .iter()
            .map(|node| {
                evaluated
                    .get(*node)
                    .expect("every requested output is materialized")
                    .0
                    .to_vec()
            })
            .collect()
    }

    fn assert_moe_routing_parity(
        label: &str,
        logits: &[f32],
        seq: u32,
        expert_count: u32,
        top_k: u32,
    ) {
        let fixture = moe_routing_fixture(logits, seq, expert_count, top_k);
        let cpu = run_moe_routing(Engine::Cpu, None, &fixture);
        let cuda = run_moe_routing(Engine::Gpu, Some(GpuDriver::Cuda), &fixture);
        let weights = top_k as usize;
        let routes = cpu.len() - weights;
        assert!(
            routes >= weights,
            "{label}: degenerate gate: at least top_k routes and top_k weights are compared"
        );
        for slot in 0..routes {
            assert_eq!(
                cuda[slot], cpu[slot],
                "{label}: route of round {slot} differs between cuda and cpu"
            );
        }
        let mut max_weight_diff = 0.0f32;
        for slot in routes..cpu.len() {
            for (&got, &want) in cuda[slot].iter().zip(cpu[slot].iter()) {
                max_weight_diff = max_weight_diff.max((got - want).abs());
                assert!(
                    (got.is_nan() && want.is_nan())
                        || (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                    "{label}: weight slot {slot} cuda={got} cpu={want}"
                );
            }
        }
        eprintln!("cuda moe_topk {label}: routes identical, max weight diff={max_weight_diff}");
    }

    #[test]
    fn granite_router_logits_route_on_cuda_exactly_as_on_cpu_at_layers_0_10_23() {
        let mut layers_checked = 0;
        for line in GRANITE_ROUTER_LOGITS
            .lines()
            .filter(|line| !line.is_empty())
        {
            let record: serde_json::Value =
                serde_json::from_str(line).expect("fixture line is json");
            let values: Vec<f32> = record["values"]
                .as_array()
                .expect("fixture carries a values array")
                .iter()
                .map(|value| value.as_f64().expect("router logit is a number") as f32)
                .collect();
            assert_eq!(values.len(), 5 * GRANITE_EXPERT_COUNT as usize);
            let label = format!("granite node {}", record["node"]);
            assert_moe_routing_parity(&label, &values, 5, GRANITE_EXPERT_COUNT, GRANITE_TOP_K);
            layers_checked += 1;
        }
        assert_eq!(layers_checked, 3, "fixture holds layers 0, 10 and 23");
    }

    #[test]
    fn moe_topk_ties_resolve_to_the_higher_expert_and_exclude_every_tied_expert_on_cuda() {
        let experts = GRANITE_EXPERT_COUNT as usize;
        let mut logits = vec![0.0f32; 3 * experts];
        for (index, value) in logits.iter_mut().enumerate().take(experts) {
            *value = index as f32 * 0.01;
        }
        logits[5] = 9.0;
        logits[20] = 9.0;
        let second = experts;
        for (offset, value) in logits[second..second + experts].iter_mut().enumerate() {
            *value = -(offset as f32) * 0.1;
        }
        logits[second + 3] = 4.0;
        logits[second + 11] = 2.5;
        logits[second + 12] = 2.5;
        logits[second + 30] = 2.5;
        assert_moe_routing_parity("ties", &logits, 3, GRANITE_EXPERT_COUNT, GRANITE_TOP_K);
    }

    #[test]
    fn moe_topk_over_all_negative_infinity_scores_never_selects_a_padding_lane_on_cuda() {
        let experts = GRANITE_EXPERT_COUNT as usize;
        let mut logits = vec![f32::NEG_INFINITY; 2 * experts];
        for (offset, value) in logits[experts..].iter_mut().enumerate() {
            *value = offset as f32;
        }
        let fixture = moe_routing_fixture(&logits, 2, GRANITE_EXPERT_COUNT, GRANITE_TOP_K);
        let cpu = run_moe_routing(Engine::Cpu, None, &fixture);
        let cuda = run_moe_routing(Engine::Gpu, Some(GpuDriver::Cuda), &fixture);
        for slot in 0..GRANITE_TOP_K as usize {
            assert_eq!(
                cuda[slot], cpu[slot],
                "all -inf row: route of round {} differs",
                slot
            );
            assert!(
                cuda[slot][0] < GRANITE_EXPERT_COUNT as f32,
                "all -inf row selected padding lane {}",
                cuda[slot][0]
            );
        }
    }

    #[test]
    fn moe_topk_with_a_non_warp_multiple_expert_count_matches_cpu_on_cuda() {
        let experts = 40usize;
        let logits = random_vec(77, 4 * experts);
        assert_moe_routing_parity("40 experts", &logits, 4, 40, 6);
    }
}

mod resident_weights {
    use std::collections::BTreeSet;

    use omega::{Codec, CudaDriver, CudaPlan};
    use proxima_gguf::quant::q4_0::{BLOCK_BYTES, QK4_0, quantize};
    use proxima_tensor::{NumericPolicy, QuantizedBlock};

    use super::{DType, packed_matmul_program, random_vec};

    const ROWS: u32 = 3;
    const BLOCKS_PER_ROW: usize = 2;
    const K: usize = QK4_0 * BLOCKS_PER_ROW;

    fn q4_0_weight(seed: u64) -> Vec<u8> {
        let row_len = std::hint::black_box(K);
        let row_bytes = BLOCKS_PER_ROW * BLOCK_BYTES;
        let values: Vec<f32> = random_vec(seed, ROWS as usize * row_len)
            .into_iter()
            .map(|value| value * 4.0 - 2.0)
            .collect();
        let mut blocks = vec![0u8; ROWS as usize * row_bytes];
        for (row_values, row_blocks) in values
            .chunks_exact(row_len)
            .zip(blocks.chunks_exact_mut(row_bytes))
        {
            quantize(row_values, row_blocks).expect("row is whole q4_0 blocks");
        }
        blocks
    }

    fn named_inputs<'a>(
        weight: &'a [u8],
        activation: &'a [f32],
    ) -> Vec<(&'static str, QuantizedBlock<'a>)> {
        vec![
            (
                "weight",
                QuantizedBlock::Packed {
                    codec: Codec::Q4_0,
                    bytes: weight,
                },
            ),
            ("activation", QuantizedBlock::Float32(activation)),
        ]
    }

    fn fresh_resident_plan(driver: &CudaDriver, weight: &[u8], activation: &[f32]) -> CudaPlan {
        let (program, sum) = packed_matmul_program(ROWS, K as u32, DType::UINT8);
        let named = named_inputs(weight, activation);
        let mut plan = driver
            .plan(&program, &[], &[sum], NumericPolicy::default(), &named)
            .expect("cuda driver plans the packed matmul");
        plan.mark_resident(&BTreeSet::from(["weight"]));
        plan
    }

    fn run_call(driver: &CudaDriver, weight: &[u8], activation: &[f32]) -> (Vec<f32>, u64) {
        let before = driver.packed_upload_bytes();
        let mut plan = fresh_resident_plan(driver, weight, activation);
        let named = named_inputs(weight, activation);
        let output = plan
            .execute_named(&named, None)
            .expect("cuda runs the packed matmul")
            .root()
            .to_vec();
        (output, driver.packed_upload_bytes() - before)
    }

    #[test]
    fn a_resident_weight_uploads_once_across_generate_calls_and_decode_steps() {
        let driver = CudaDriver::new(0).expect("a real cuda device");
        let weight = q4_0_weight(37);

        let uploads: Vec<u64> = (0..5)
            .map(|step| run_call(&driver, &weight, &random_vec(31 + step, K)).1)
            .collect();

        assert_eq!(
            uploads[0],
            weight.len() as u64,
            "call 1 uploads the full weight set"
        );
        assert_eq!(
            &uploads[1..],
            &[0, 0, 0, 0],
            "call 2 and each decode step upload 0 weight bytes"
        );
    }

    #[test]
    fn a_resident_weight_gives_the_same_output_as_a_cold_upload() {
        let driver = CudaDriver::new(0).expect("a real cuda device");
        let weight = q4_0_weight(37);
        let activation = random_vec(31, K);

        let (cold, _) = run_call(&driver, &weight, &activation);
        let (warm, warm_upload) = run_call(&driver, &weight, &activation);

        assert_eq!(
            warm_upload, 0,
            "degenerate gate: the warm call must hit the cache"
        );
        assert_eq!(
            cold.len(),
            ROWS as usize,
            "degenerate gate: no outputs compared"
        );
        assert_eq!(
            cold, warm,
            "a cached device buffer must compute the same bytes"
        );
    }

    #[test]
    fn a_changed_weight_source_is_reuploaded_and_never_served_stale() {
        let driver = CudaDriver::new(0).expect("a real cuda device");
        let original = q4_0_weight(37);
        let changed = q4_0_weight(41);
        let activation = random_vec(31, K);

        let (original_output, _) = run_call(&driver, &original, &activation);
        let (changed_output, changed_upload) = run_call(&driver, &changed, &activation);
        let (original_again, original_upload) = run_call(&driver, &original, &activation);

        assert_eq!(
            changed_upload,
            changed.len() as u64,
            "a new source must upload"
        );
        assert_ne!(
            original_output, changed_output,
            "degenerate gate: the two weights must differ"
        );
        assert_eq!(original_upload, 0, "the original source is still cached");
        assert_eq!(original_output, original_again);
    }

    #[test]
    fn releasing_a_weight_name_evicts_its_device_buffer() {
        let driver = CudaDriver::new(0).expect("a real cuda device");
        let weight = q4_0_weight(37);
        let activation = random_vec(31, K);

        let _ = run_call(&driver, &weight, &activation);
        driver.release_resident_names(["weight"]);
        let (_, upload_after_release) = run_call(&driver, &weight, &activation);

        assert_eq!(upload_after_release, weight.len() as u64);
    }
}

fn two_layer_mlp_cuda_plan() -> (omega::backend::Plan, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (program, _root) = two_layer_mlp_fixture();
    let x = random_vec(1, (BATCH * IN_FEATURES) as usize);
    let w1 = random_vec(2, (IN_FEATURES * HIDDEN) as usize);
    let w2 = random_vec(3, (HIDDEN * OUT_FEATURES) as usize);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("x", QuantizedBlock::Float32(&x)),
        ("w1", QuantizedBlock::Float32(&w1)),
        ("w2", QuantizedBlock::Float32(&w2)),
    ];
    let plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the mlp on cuda");
    (plan, x, w1, w2)
}

#[test]
fn refit_symbols_increments_refit_calls_once_per_call() {
    let (mut cuda_plan, _x, _w1, _w2) = two_layer_mlp_cuda_plan();
    let before = omega::backend::cuda_refit_counters(&cuda_plan)
        .expect("a cuda plan exposes its refit counters");
    assert_eq!(
        before,
        [0, 0, 0, 0],
        "a freshly built plan has no refit history"
    );

    let refitted =
        omega::backend::refit_cuda_symbols(&mut cuda_plan, &[], NumericPolicy::default())
            .expect("refit at unchanged symbols succeeds on cuda");
    assert!(refitted, "the mlp plan refits in place");

    let after = omega::backend::cuda_refit_counters(&cuda_plan)
        .expect("a cuda plan exposes its refit counters");
    let op_count = after[2];
    assert!(
        op_count > 0,
        "a refit keeps at least one resolved op, got {op_count}"
    );
    assert_eq!(
        after,
        [1, 0, op_count, 0],
        "unchanged symbols re-emit and repack nothing: every op is reused as it stands"
    );
}

#[test]
fn plan_execution_leaves_refit_counters_unchanged() {
    let (mut cuda_plan, x, w1, w2) = two_layer_mlp_cuda_plan();
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("x", QuantizedBlock::Float32(&x)),
        ("w1", QuantizedBlock::Float32(&w1)),
        ("w2", QuantizedBlock::Float32(&w2)),
    ];
    let before = omega::backend::cuda_refit_counters(&cuda_plan)
        .expect("a cuda plan exposes its refit counters");
    execute_plan_named(&mut cuda_plan, &named).expect("cuda runs the mlp plan");
    execute_plan_named(&mut cuda_plan, &named).expect("cuda reruns the mlp plan");
    let after = omega::backend::cuda_refit_counters(&cuda_plan)
        .expect("a cuda plan exposes its refit counters");
    assert_eq!(
        after, before,
        "executing a plan is not a refit, re-emit or repack"
    );
}

fn padded_cache_relative_error(
    cached_len: u64,
    pad_value: f32,
    wrong_cached_len: Option<f32>,
) -> (f32, Vec<String>) {
    let (reference, mut padded) = qwen3_gqa_qk_norm_padded_cache_fixture(cached_len, 64, pad_value);
    if let Some(wrong) = wrong_cached_len {
        for (name, data) in padded.3.iter_mut() {
            if name == "cached_len" {
                *data = vec![wrong];
            }
        }
    }
    let reference_named = as_named_blocks(&reference.3);
    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &reference.0,
        &reference.1,
        &reference_named,
        &reference.2,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the live reference on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &reference_named)
        .expect("omega::backend runs the live reference on cpu");

    let padded_named = as_named_blocks(&padded.3);
    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &padded.0,
        &padded.1,
        &padded_named,
        &padded.2,
        NumericPolicy::bit_exact().with_reassociation(true),
    )
    .expect("omega::backend plans the padded bucket on cuda");
    let identities = cuda_kernel_identities(&cuda_plan).expect("cuda plan exposes its kernels");
    let cuda = execute_plan_named(&mut cuda_plan, &padded_named)
        .expect("omega::backend runs the padded bucket on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), expected.len());
    let mut max_diff = 0.0f32;
    for (&got, &want) in actual.iter().zip(expected.iter()) {
        assert!(
            got.is_finite(),
            "cuda read a non-finite value from the padded bucket: {got} (a pad row leaked)"
        );
        max_diff = max_diff.max((got - want).abs());
    }
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    (max_diff / max_magnitude.max(f32::MIN_POSITIVE), identities)
}

#[test]
fn cuda_cached_attention_matches_cpu_at_two_cached_lens_in_one_bucket() {
    for cached_len in [31_u64, 33] {
        let (relative, identities) = padded_cache_relative_error(cached_len, f32::NAN, None);
        eprintln!("cached_attention cached_len={cached_len} bucket=64 relative={relative}");
        assert!(
            identities
                .iter()
                .any(|identity| identity.contains("omega_cuda_cached_attention_")),
            "fused cached attention absent from the cuda plan: {identities:?}"
        );
        assert!(
            relative < 1e-4,
            "cached_len={cached_len}: cuda vs cpu relative={relative}, tolerance 1e-4"
        );
    }
}

#[test]
fn cuda_cached_attention_diverges_when_cached_len_is_wrong() {
    let (relative, _) = padded_cache_relative_error(31, 0.5, Some(32.0));
    eprintln!("cached_attention control: wrong cached_len=32 vs live 31, relative={relative}");
    assert!(
        relative >= 1e-4,
        "a wrong cached_len must diverge beyond the tolerance, relative={relative}"
    );
}

/// Wall time of one planned prefill matmul per call at the shapes a 8B
/// ternary checkpoint runs: asserts the tensor-core kernel agrees with the
/// f32 oracle and prints the mean over `iterations` executes.
fn time_packed_prefill(rows: u32, k: u32, tokens: u32, iterations: u32) -> f64 {
    use proxima_gguf::quant::pq2_0::{BLOCK_BYTES, QK_PQ2_0};
    let blocks_per_row = k as usize / QK_PQ2_0;
    let mut lcg = Lcg(53);
    let mut weight_blocks: Vec<u8> = (0..rows as usize * blocks_per_row * BLOCK_BYTES)
        .map(|_| (lcg.next_unit() * 255.0) as u8)
        .collect();
    for block in weight_blocks.as_chunks_mut::<BLOCK_BYTES>().0 {
        block[0] = 0x00;
        block[1] = 0x3C;
    }
    let activation = random_vec(59, (k * tokens) as usize);
    let (program, root) = packed_batched_program(rows, k, Some(tokens), DType::UINT8);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        (
            "weight",
            QuantizedBlock::Packed {
                codec: Codec::Pq2_0,
                bytes: &weight_blocks,
            },
        ),
        ("activation", QuantizedBlock::Float32(&activation)),
    ];
    let mut plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[root],
        NumericPolicy::bit_exact().with_reassociation(true),
    )
    .expect("omega::backend plans the timed matmul");
    execute_plan_named(&mut plan, &named).expect("warm-up execute");
    let started = std::time::Instant::now();
    for _ in 0..iterations {
        let output = execute_plan_named(&mut plan, &named).expect("timed execute");
        assert!(output.root().iter().all(|value| value.is_finite()));
    }
    started.elapsed().as_secs_f64() * 1_000.0 / f64::from(iterations)
}

#[test]
fn prefill_matmul_timing_at_8b_gate_shape_reports_ms_per_call() {
    let per_call: Vec<f64> = [5u32, 22, 66]
        .iter()
        .map(|tokens| time_packed_prefill(12288, 4096, *tokens, 20))
        .collect();
    eprintln!("prefill_matmul_timing ms_per_call at 5, 22, 66 tokens: {per_call:?}");
    assert!(
        per_call.iter().all(|millis| *millis > 0.0),
        "timed executes must take measurable time: {per_call:?}"
    );
    assert!(
        per_call[2] < per_call[0] * 66.0,
        "66 tokens must cost less than 66 single-row executes: {per_call:?}"
    );
}

#[cfg(feature = "cached-attention-streaming")]
fn partial_rotary_relative_error(cached_extent: u64, cached_len: u64) -> (f32, Vec<String>) {
    let (program, symbols, roots, owned) =
        qwen35_partial_rotary_forward_fixture(1, cached_extent, cached_len);
    let named = as_named_blocks(&owned);
    let mut cpu_plan = plan_named(
        Engine::Cpu,
        None,
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the partial-rotary fixture on cpu");
    let cpu = execute_plan_named(&mut cpu_plan, &named)
        .expect("omega::backend runs the partial-rotary fixture on cpu");

    let mut cuda_plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &symbols,
        &named,
        &roots,
        NumericPolicy::bit_exact().with_reassociation(true),
    )
    .expect("omega::backend plans the partial-rotary fixture on cuda");
    let identities = cuda_kernel_identities(&cuda_plan).expect("cuda plan exposes its kernels");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the partial-rotary fixture on a real cuda device");

    let expected = cpu.root();
    let actual = cuda.root();
    assert_eq!(actual.len(), expected.len());
    let max_diff = expected
        .iter()
        .zip(actual.iter())
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    (max_diff / max_magnitude.max(f32::MIN_POSITIVE), identities)
}

#[cfg(feature = "cached-attention-streaming")]
#[test]
fn cuda_partial_rotary_cached_attention_matches_cpu_with_padded_rows() {
    for (cached_extent, cached_len) in [(40_u64, 37_u64), (512, 512)] {
        let (relative, identities) = partial_rotary_relative_error(cached_extent, cached_len);
        eprintln!(
            "partial_rotary cached_extent={cached_extent} cached_len={cached_len} relative={relative}"
        );
        assert!(
            identities
                .iter()
                .any(|identity| identity.contains("omega_cuda_cached_attention_") && identity.contains("_r64")),
            "partial-rotary fused attention absent from the cuda plan: {identities:?}"
        );
        assert!(
            relative < 1e-4,
            "cached_extent={cached_extent} cached_len={cached_len}: cuda vs cpu relative={relative}, tolerance 1e-4"
        );
    }
}

#[test]
fn packed_q1_0_matmul_matches_at_the_27b_contraction_widths_on_cuda() {
    use proxima_gguf::quant::q1_0::{BLOCK_BYTES, QK_Q1_0, dequantize, quantize};
    for blocks_per_row in [40_usize, 48, 136] {
        assert_packed_codec_parity(
            "q1_0",
            BLOCK_BYTES,
            QK_Q1_0,
            quantize,
            dequantize,
            |bytes| QuantizedBlock::Packed {
                codec: Codec::Q1_0,
                bytes,
            },
            96,
            blocks_per_row,
            NumericPolicy::default(),
            1e-4,
        );
    }
}

const APPEND_WIDTH: usize = 1024;
const APPEND_CACHE: NodeId = NodeId(0);
const APPENDED_ROWS: NodeId = NodeId(2);
const APPEND_COLUMN_SUM: NodeId = NodeId(3);

fn appended_rows_program() -> Vec<Op> {
    appended_rows_program_with(Extent::Static(1))
}

fn appended_rows_program_with(row_extent: Extent) -> Vec<Op> {
    let mut program = Vec::new();
    let cache = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Symbolic(0), Extent::Static(APPEND_WIDTH as u32)],
            name: Some("cache".into()),
        },
    );
    let row = append(
        &mut program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![row_extent, Extent::Static(APPEND_WIDTH as u32)],
            name: Some("row".into()),
        },
    );
    let both = IndexMap::Affine(map::projection(2, &[0, 1]));
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            operands: vec![(row, both.clone()), (row, both.clone())],
            name: Some("appended".into()),
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: cache,
            in_map: both,
            out_map: IndexMap::Affine(map::projection(2, &[1])),
            keep: Keep::Reduce,
            name: Some("column_sum".into()),
        }),
    );
    program
}

fn appended_row_values(step: usize) -> Vec<f32> {
    (0..APPEND_WIDTH)
        .map(|column| ((step * APPEND_WIDTH + column + 13) % 97) as f32 * 0.25)
        .collect()
}

fn padded_cache(valid: &[f32], extent: usize) -> Vec<f32> {
    let mut padded = valid.to_vec();
    padded.resize(extent * APPEND_WIDTH, 0.0);
    padded
}

fn appended_rows_plan(
    program: &[Op],
    extent: usize,
    cache: &[f32],
    row: &[f32],
) -> Plan {
    appended_rows_plan_with_symbols(program, &[extent as u64], cache, row)
}

fn appended_rows_plan_with_symbols(
    program: &[Op],
    symbols: &[u64],
    cache: &[f32],
    row: &[f32],
) -> Plan {
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("cache", QuantizedBlock::Float32(cache)),
        ("row", QuantizedBlock::Float32(row)),
    ];
    plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        program,
        symbols,
        &named,
        &[APPENDED_ROWS, APPEND_COLUMN_SUM],
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the appended-rows program on cuda")
}

#[proxima::test]
#[case::within_the_planned_extent(16, 2, 5)]
#[case::crossing_an_extent_growth(4, 2, 7)]
#[case::from_an_empty_cache(4, 0, 6)]
async fn appended_rows_stay_on_device_and_drain_equal_to_the_host_append(
    #[case] initial_extent: usize,
    #[case] initial_rows: usize,
    #[case] steps: usize,
) {
    let program = appended_rows_program();
    let seed: Vec<f32> = (0..initial_rows * APPEND_WIDTH)
        .map(|index| (index % 97) as f32 * 0.25)
        .collect();
    let first_row = appended_row_values(0);
    let policy = NumericPolicy::default();

    let mut host_plan = appended_rows_plan(&program, initial_extent, &padded_cache(&seed, initial_extent), &first_row);
    let mut carry_plan = appended_rows_plan(&program, initial_extent, &padded_cache(&seed, initial_extent), &first_row);
    assert!(
        omega::backend::carry_cuda_rows(&mut carry_plan, &[(APPEND_CACHE, APPENDED_ROWS)], initial_rows as u64),
        "the cache and row outputs share a row width"
    );

    let mut host_cache = seed.clone();
    let mut extent = initial_extent;
    let first_upload = (initial_extent * APPEND_WIDTH * std::mem::size_of::<f32>()) as u64;
    for step in 0..steps {
        let row = appended_row_values(step);
        if host_cache.len() / APPEND_WIDTH + 1 > extent {
            extent *= 2;
            for plan in [&mut host_plan, &mut carry_plan] {
                let refitted = omega::backend::refit_cuda_symbols(plan, &[extent as u64], policy)
                    .expect("refit to the grown cache extent succeeds");
                assert!(refitted, "step {step}: the plan refits in place");
            }
        }
        let host_cache_block = padded_cache(&host_cache, extent);
        let host_named: Vec<(&str, QuantizedBlock<'_>)> = vec![
            ("cache", QuantizedBlock::Float32(&host_cache_block)),
            ("row", QuantizedBlock::Float32(&row)),
        ];
        let host_step = execute_plan_named(&mut host_plan, &host_named)
            .expect("host round trip step executes");
        let stale: &[f32] = &[];
        let carry_named: Vec<(&str, QuantizedBlock<'_>)> = vec![
            ("cache", QuantizedBlock::Float32(if step == 0 { &host_cache_block } else { stale })),
            ("row", QuantizedBlock::Float32(&row)),
        ];
        let carry_step = execute_plan_named(&mut carry_plan, &carry_named)
            .expect("carried step executes");

        assert_eq!(
            carry_step.get(APPEND_COLUMN_SUM).expect("carried step returns the column sum").0,
            host_step.get(APPEND_COLUMN_SUM).expect("host step returns the column sum").0,
            "step {step}: the reader sees the same cache rows on both paths"
        );
        assert!(carry_step.get(APPENDED_ROWS).is_none(), "step {step}: appended rows are not read back");
        host_cache.extend_from_slice(host_step.get(APPENDED_ROWS).expect("host step returns the new row").0);
        assert_eq!(
            omega::backend::cuda_carried_host_bytes(&carry_plan),
            Some(first_upload),
            "step {step}: nothing but the first upload crosses the host"
        );
    }

    let drained = omega::backend::drain_cuda_appended_rows(&mut carry_plan)
        .expect("drain reads the valid rows from the device");
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].0, APPEND_CACHE, "drained rows are keyed by the cache input");
    assert_eq!(drained[0].1, host_cache, "drained cache equals the host append");
    let drained_bytes = (host_cache.len() * std::mem::size_of::<f32>()) as u64;
    assert_eq!(
        omega::backend::cuda_carried_host_bytes(&carry_plan),
        Some(first_upload + drained_bytes),
        "carried bytes are the first upload plus one drain"
    );
}

fn appended_row_block(first_row: usize, count: usize) -> Vec<f32> {
    (first_row..first_row + count).flat_map(appended_row_values).collect()
}

#[proxima::test]
#[case::cache_extent_stays_zero(false)]
#[case::cache_extent_refits_to_the_valid_rows(true)]
async fn seven_rows_into_an_exact_empty_cache_then_single_rows_grow_the_device_cache(
    #[case] refit_cache_extent: bool,
) {
    const FIRST_APPEND_ROWS: usize = 7;
    const STEPS: usize = 10;
    let program = appended_rows_program_with(Extent::Symbolic(1));
    let first_block = appended_row_block(0, FIRST_APPEND_ROWS);
    let policy = NumericPolicy::default();
    let symbols = [0, FIRST_APPEND_ROWS as u64];
    let mut host_plan = appended_rows_plan_with_symbols(&program, &symbols, &[], &first_block);
    let mut carry_plan = appended_rows_plan_with_symbols(&program, &symbols, &[], &first_block);
    assert!(
        omega::backend::carry_cuda_rows(&mut carry_plan, &[(APPEND_CACHE, APPENDED_ROWS)], 0),
        "the cache and row outputs share a row width"
    );

    let mut host_cache: Vec<f32> = Vec::new();
    let mut extent = 0usize;
    let mut appended_rows = 0usize;
    for step in 0..STEPS {
        let new_rows = if step == 0 { FIRST_APPEND_ROWS } else { 1 };
        let block = appended_row_block(appended_rows, new_rows);
        if step > 0 {
            if refit_cache_extent {
                extent = extent.max((appended_rows + new_rows).next_power_of_two());
            }
            for plan in [&mut host_plan, &mut carry_plan] {
                let refitted = omega::backend::refit_cuda_symbols(plan, &[extent as u64, new_rows as u64], policy)
                    .expect("refit to the step's cache extent and row count succeeds");
                assert!(refitted, "step {step}: the plan refits in place");
            }
        }
        let host_cache_block = padded_cache(&host_cache, extent);
        let host_named: Vec<(&str, QuantizedBlock<'_>)> = vec![
            ("cache", QuantizedBlock::Float32(&host_cache_block)),
            ("row", QuantizedBlock::Float32(&block)),
        ];
        let host_step = execute_plan_named(&mut host_plan, &host_named).expect("host step executes");
        let stale: &[f32] = &[];
        let carry_named: Vec<(&str, QuantizedBlock<'_>)> = vec![
            ("cache", QuantizedBlock::Float32(if step == 0 { &host_cache_block } else { stale })),
            ("row", QuantizedBlock::Float32(&block)),
        ];
        let carry_step = execute_plan_named(&mut carry_plan, &carry_named).expect("carried step executes");

        assert_eq!(
            carry_step.get(APPEND_COLUMN_SUM).expect("carried column sum").0,
            host_step.get(APPEND_COLUMN_SUM).expect("host column sum").0,
            "step {step}: the reader sees the same cache rows on both paths"
        );
        host_cache.extend_from_slice(host_step.get(APPENDED_ROWS).expect("host appended rows").0);
        appended_rows += new_rows;
        assert_eq!(
            omega::backend::cuda_carried_host_bytes(&carry_plan),
            Some(0),
            "step {step}: an empty first cache uploads nothing and nothing crosses the host"
        );
    }

    let drained = omega::backend::drain_cuda_appended_rows(&mut carry_plan)
        .expect("drain reads the valid rows from the device");
    assert_eq!(drained.len(), 1);
    assert_eq!(drained[0].0, APPEND_CACHE, "drained rows are keyed by the cache input");
    assert_eq!(drained[0].1.len(), (FIRST_APPEND_ROWS + STEPS - 1) * APPEND_WIDTH);
    assert_eq!(drained[0].1, host_cache, "drained cache equals the host append");
}

#[test]
fn carry_rows_rejects_a_pair_whose_row_widths_differ() {
    let program = appended_rows_program();
    let cache = padded_cache(&[], 4);
    let row = appended_row_values(0);
    let mut plan = appended_rows_plan(&program, 4, &cache, &row);

    let carried = omega::backend::carry_cuda_rows(&mut plan, &[(APPEND_CACHE, APPEND_COLUMN_SUM)], 0);

    assert!(!carried, "the column sum has row width 1, the cache row width {APPEND_WIDTH}");
    assert_eq!(omega::backend::cuda_carried_host_bytes(&plan), Some(0));
}

const NORM_EPSILON: f32 = 1e-5;
const NORM_RUNS: usize = 3;
const NORM_CHAIN: usize = 24;

// the backlog only makes event deltas device time; a correctness run needs neither it nor repeats,
// and paying for it on a shared gpu pushed the chain cases past the harness timeout
fn timing_requested() -> bool {
    env::var_os("PROXIMA_CUDA_EVENT_TIMING").is_some()
}

fn busy_extent() -> u32 {
    if timing_requested() { 1 << 26 } else { 1 << 10 }
}

fn busy_depth() -> usize {
    if timing_requested() { 40 } else { 1 }
}

fn timed_runs(runs: usize) -> usize {
    if timing_requested() { runs } else { 1 }
}

/// Tanh passes over a large buffer keep the stream backlogged while the host
/// enqueues the norm chain, so event deltas between the norms are device time and
/// not host launch latency.
fn append_busy_chain(program: &mut Vec<Op>) -> NodeId {
    let mut node = input_leaf(
        program,
        DType::FLOAT32,
        vec![Extent::Static(busy_extent())],
        "busy",
    );
    for _ in 0..busy_depth() {
        node = append(
            program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Tanh,
                operands: vec![(node, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
    }
    node
}

fn norm_program(rows: u32, width: u32) -> (Vec<Op>, Vec<NodeId>) {
    let mut program = Vec::new();
    let busy = append_busy_chain(&mut program);
    let eps_scalar = scalar_constant(&mut program, NORM_EPSILON);
    let inv_width = scalar_constant(&mut program, 1.0 / width as f32);
    let gamma = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Static(width)], "gamma");
    let eps_row = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Static(1)], "eps");
    let shape = if rows == 1 {
        vec![Extent::Static(1), Extent::Static(width)]
    } else {
        vec![Extent::Static(1), Extent::Static(rows), Extent::Static(width)]
    };
    let mut node = input_leaf(&mut program, DType::FLOAT32, shape, "x");
    for _ in 0..NORM_CHAIN {
        node = if rows == 1 {
            rmsnorm(&mut program, node, gamma, inv_width, eps_scalar).expect("rmsnorm lowers")
        } else {
            rmsnorm_per_head(&mut program, node, gamma, inv_width, eps_row, "h")
                .expect("per-head rmsnorm lowers")
        };
    }
    (program, vec![node, busy])
}

fn norm_oracle(x: &[f32], gamma: &[f32], width: usize) -> Vec<f64> {
    let mut current: Vec<f64> = x.iter().map(|value| f64::from(*value)).collect();
    for _ in 0..NORM_CHAIN {
        current = current
            .chunks_exact(width)
            .flat_map(|row| {
                let mean_square = row.iter().map(|value| value * value).sum::<f64>() / width as f64;
                let scale = 1.0 / (mean_square + f64::from(NORM_EPSILON)).sqrt();
                row.iter().zip(gamma).map(move |(value, weight)| value * scale * f64::from(*weight))
            })
            .collect();
    }
    current
}

#[proxima::test]
#[case::decode_hidden_norm_one_row_of_4096(1, 4096)]
#[case::per_head_q_norm_32_rows_of_128(32, 128)]
#[case::per_head_k_norm_8_rows_of_128(8, 128)]
async fn rmsnorm_broadcast_epilogue_reduce_matches_the_f64_oracle_on_cuda(
    #[case] rows: u32,
    #[case] width: u32,
) {
    let (program, roots) = norm_program(rows, width);
    let x = random_vec(11, (rows * width) as usize);
    let gamma: Vec<f32> = random_vec(13, width as usize).iter().map(|value| 1.0 + value * 0.5).collect();
    let eps = [NORM_EPSILON];
    let busy = random_vec(17, busy_extent() as usize);
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("busy", QuantizedBlock::Float32(&busy)),
        ("x", QuantizedBlock::Float32(&x)),
        ("gamma", QuantizedBlock::Float32(&gamma)),
        ("eps", QuantizedBlock::Float32(&eps)),
    ];
    let mut plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &roots,
        NumericPolicy::bit_exact().with_reassociation(true),
    )
    .expect("the rmsnorm chain plans on cuda");
    let identities = cuda_kernel_identities(&plan).expect("a cuda plan");
    assert!(
        identities.iter().any(|identity| {
            identity.contains("_reduce_r") && identity.contains("_epi") && identity.contains("square_root")
        }),
        "the norm must lower to the broadcast-epilogue reduce, got {:?}",
        identities.iter().map(|identity| identity.lines().next()).collect::<Vec<_>>()
    );
    let mut actual = Vec::new();
    for _ in 0..timed_runs(NORM_RUNS) {
        actual = execute_plan_named(&mut plan, &named)
            .expect("the rmsnorm chain runs on cuda")
            .root()
            .to_vec();
    }
    let expected = norm_oracle(&x, &gamma, width as usize);
    assert_eq!(actual.len(), expected.len());
    let worst = actual
        .iter()
        .zip(&expected)
        .map(|(got, want)| (f64::from(*got) - want).abs() / want.abs().max(1e-3))
        .fold(0.0f64, f64::max);
    assert!(worst < 1e-5, "cuda rmsnorm disagrees with the f64 oracle: worst relative={worst}");
}

const MATVEC_CHAIN: usize = 24;
const MATVEC_RUNS: usize = 8;

fn matvec_chain_program(rows: u32, k: u32) -> (Vec<Op>, Vec<NodeId>, Vec<String>) {
    let mut program = Vec::new();
    let busy = append_busy_chain(&mut program);
    let weight = input_leaf(&mut program, DType::UINT8, vec![Extent::Static(rows), Extent::Static(k)], "weight");
    let names: Vec<String> = (0..MATVEC_CHAIN).map(|index| format!("activation{index}")).collect();
    let mut roots = Vec::new();
    for name in &names {
        let activation = input_leaf(&mut program, DType::FLOAT32, vec![Extent::Static(k), Extent::Static(1)], name);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(3, &[0, 2]))),
                    (activation, IndexMap::Affine(projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        roots.push(append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::FLOAT32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matvec".into()),
            }),
        ));
    }
    roots.push(busy);
    (program, roots, names)
}

#[proxima::test]
#[case::pq2_0_qkv_4096_by_4096(Codec::Pq2_0, 4096, 4096)]
#[case::pq2_0_gate_up_12288_by_4096(Codec::Pq2_0, 12288, 4096)]
#[case::pq2_0_down_4096_by_12288(Codec::Pq2_0, 4096, 12288)]
#[case::ptq1_0_5120_by_17408(Codec::Ptq1_0, 5120, 17408)]
#[case::q1_0_5120_by_17408(Codec::Q1_0, 5120, 17408)]
#[case::q1_0_gate_17408_by_5120(Codec::Q1_0, 17408, 5120)]
async fn staged_matvec_chain_matches_the_f64_oracle_and_reports_event_time_on_cuda(
    #[case] codec: Codec,
    #[case] rows: usize,
    #[case] k: usize,
) {
    let layout = codec.block_layout().expect("a block codec");
    let weight = arbitrary_blocks(&layout, rows * k / usize::from(layout.elements), 41);
    let activations: Vec<Vec<f32>> = (0..MATVEC_CHAIN)
        .map(|index| on_the_int8_grid(&random_vec(43 + index as u64, k).iter().map(|value| value * 4.0 - 2.0).collect::<Vec<_>>()))
        .collect();
    let busy = random_vec(17, busy_extent() as usize);
    let (program, roots, names) = matvec_chain_program(rows as u32, k as u32);
    let mut named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("busy", QuantizedBlock::Float32(&busy)),
        ("weight", QuantizedBlock::Packed { codec, bytes: &weight }),
    ];
    named.extend(names.iter().zip(&activations).map(|(name, values)| (name.as_str(), QuantizedBlock::Float32(values))));
    let policy = NumericPolicy::default().with_activation_quantization(true);
    let mut plan = plan_named(Engine::Gpu, Some(GpuDriver::Cuda), &program, &[], &named, &roots, policy)
        .expect("the matvec chain plans on cuda");
    let actual = (0..timed_runs(MATVEC_RUNS))
        .map(|_| {
            let evaluated = execute_plan_named(&mut plan, &named).expect("the matvec chain runs on cuda");
            evaluated.get(roots[0]).expect("the first matvec is a requested output").0.to_vec()
        })
        .last()
        .expect("at least one run");

    let mut decoded = vec![0.0f32; rows * k];
    block_layout::dequantize(&layout, &weight, &mut decoded).expect("whole blocks");
    let expected: Vec<f64> = decoded
        .chunks_exact(k)
        .map(|row| row.iter().zip(&activations[0]).map(|(code, value)| f64::from(*code) * f64::from(*value)).sum())
        .collect();
    let magnitude = expected.iter().fold(0.0f64, |largest, value| largest.max(value.abs()));
    let worst = actual.iter().zip(&expected).map(|(got, want)| (f64::from(*got) - want).abs()).fold(0.0f64, f64::max);
    assert_eq!(actual.len(), rows, "every row is compared");
    assert!(worst / magnitude < 1e-5, "{codec:?} {rows}x{k} disagrees with the walker: worst={worst} magnitude={magnitude}");
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, rank: u16, operands: Vec<(NodeId, &[u16])>) -> NodeId {
    let operands = operands
        .into_iter()
        .map(|(node, axes)| (node, IndexMap::Affine(map::projection(rank, axes))))
        .collect();
    append(program, Op::Elementwise { dtype: DType::FLOAT32, body, operands, name: None })
}

fn state_update_program(rows: u32, columns: u32, heads: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let state = append_input(&mut program, DType::FLOAT32, vec![Extent::Static(rows), Extent::Static(columns), Extent::Static(heads)], "state");
    let key = append_input(&mut program, DType::FLOAT32, vec![Extent::Static(rows), Extent::Static(heads)], "key");
    let value = append_input(&mut program, DType::FLOAT32, vec![Extent::Static(columns), Extent::Static(heads)], "value");
    let decay = append_input(&mut program, DType::FLOAT32, vec![Extent::Static(heads)], "decay");
    let product = elementwise(&mut program, ScalarOp::Multiply, 3, vec![(state, &[0, 1, 2]), (key, &[0, 2])]);
    let projected = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[1, 2])),
            keep: Keep::Reduce,
            name: Some("projected".into()),
        }),
    );
    let residual = elementwise(&mut program, ScalarOp::Subtract, 2, vec![(value, &[0, 1]), (projected, &[0, 1])]);
    let decayed = elementwise(&mut program, ScalarOp::Multiply, 3, vec![(state, &[0, 1, 2]), (decay, &[2])]);
    let outer = elementwise(&mut program, ScalarOp::Multiply, 3, vec![(key, &[0, 2]), (residual, &[1, 2])]);
    let next = elementwise(&mut program, ScalarOp::Add, 3, vec![(decayed, &[0, 1, 2]), (outer, &[0, 1, 2])]);
    (program, next)
}

fn state_update_oracle(rows: usize, columns: usize, heads: usize, inputs: &[Vec<f32>; 4]) -> Vec<f64> {
    let [state, key, value, decay] = inputs;
    let at = |row: usize, column: usize, head: usize| (row * columns + column) * heads + head;
    let mut next = vec![0.0f64; rows * columns * heads];
    for column in 0..columns {
        for head in 0..heads {
            let projected: f64 = (0..rows)
                .map(|row| f64::from(state[at(row, column, head)]) * f64::from(key[row * heads + head]))
                .sum();
            let residual = f64::from(value[column * heads + head]) - projected;
            for row in 0..rows {
                next[at(row, column, head)] = f64::from(state[at(row, column, head)]) * f64::from(decay[head])
                    + f64::from(key[row * heads + head]) * residual;
            }
        }
    }
    next
}

/// A reduce over the slow axis of a state whose epilogue broadcasts the reduced vector back
/// over the whole state: the lane-major lowering must equal the f64 host oracle, including
/// when the output rows do not fill the last warp and when the reduction is shorter than the
/// block's warp count.
#[proxima::test]
#[case::gated_delta_net_state_tail_warp(128, 33, 3)]
#[case::state_rows_fewer_than_block_warps(3, 40, 2)]
#[case::state_rows_not_a_warp_multiple(75, 16, 4)]
async fn broadcast_epilogue_state_update_over_a_strided_reduction_equals_the_f64_oracle_on_cuda(
    #[case] rows: usize,
    #[case] columns: usize,
    #[case] heads: usize,
) {
    let (program, next) = state_update_program(rows as u32, columns as u32, heads as u32);
    let inputs = [
        random_vec(51, rows * columns * heads),
        random_vec(53, rows * heads),
        random_vec(57, columns * heads),
        random_vec(59, heads),
    ];
    let names = ["state", "key", "value", "decay"];
    let named: Vec<(&str, QuantizedBlock<'_>)> = names
        .iter()
        .zip(&inputs)
        .map(|(name, data)| (*name, QuantizedBlock::Float32(data)))
        .collect();
    let mut plan = plan_named(Engine::Gpu, Some(GpuDriver::Cuda), &program, &[], &named, &[next], NumericPolicy::llama_relaxed())
        .expect("the state update plans on cuda");
    let identities = cuda_kernel_identities(&plan).expect("a cuda plan");
    assert!(
        identities.iter().any(|identity| identity.contains("_lm")),
        "the strided reduction must lower lane-major, got {:?}",
        identities.iter().map(|identity| identity.lines().next()).collect::<Vec<_>>()
    );
    let actual = execute_plan_named(&mut plan, &named).expect("the state update runs on cuda");

    let expected = state_update_oracle(rows, columns, heads, &inputs);
    let magnitude = expected.iter().fold(0.0f64, |largest, value| largest.max(value.abs()));
    let worst = actual
        .root()
        .iter()
        .zip(&expected)
        .map(|(got, want)| (f64::from(*got) - want).abs())
        .fold(0.0f64, f64::max);
    assert_eq!(actual.root().len(), expected.len(), "every state element is compared");
    assert!(worst / magnitude < 1e-4, "lane-major state update disagrees with the oracle: worst={worst} magnitude={magnitude}");
}

mod pooled_intermediates {
    use omega::CudaDriver;
    use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, QuantizedBlock};

    use super::{append_input, append_matmul, random_vec};
    use omega::backend::{Engine, execute_plan_named, plan_named};

    const ROWS: u32 = 4;
    const WIDTH: u32 = 8;
    const CHAIN: usize = 12;

    fn chain_program() -> (Vec<proxima_tensor::Op>, NodeId) {
        let mut program = Vec::new();
        let square = vec![Extent::Static(WIDTH), Extent::Static(WIDTH)];
        let mut activation = append_input(
            &mut program,
            DType::FLOAT32,
            vec![Extent::Static(ROWS), Extent::Static(WIDTH)],
            "x",
        );
        for link in 0..CHAIN {
            let weight = append_input(&mut program, DType::FLOAT32, square.clone(), &format!("w{link}"));
            activation = append_matmul(&mut program, activation, weight, ROWS, WIDTH, WIDTH);
        }
        (program, activation)
    }

    fn weights() -> Vec<Vec<f32>> {
        (0..CHAIN)
            .map(|link| {
                random_vec(100 + link as u64, (WIDTH * WIDTH) as usize)
                    .into_iter()
                    .map(|value| value * 0.25)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_long_chain_of_intermediates_allocates_fewer_buffers_than_it_has_nodes_and_matches_cpu() {
        let (program, root) = chain_program();
        let activation = random_vec(7, (ROWS * WIDTH) as usize);
        let weights = weights();
        let names: Vec<String> = (0..CHAIN).map(|link| format!("w{link}")).collect();
        let mut named: Vec<(&str, QuantizedBlock<'_>)> =
            vec![("x", QuantizedBlock::Float32(&activation))];
        named.extend(
            names
                .iter()
                .zip(&weights)
                .map(|(name, values)| (name.as_str(), QuantizedBlock::Float32(values))),
        );

        let mut cpu_plan = plan_named(Engine::Cpu, None, &program, &[], &named, &[root], NumericPolicy::default())
            .expect("cpu plans the matmul chain");
        let expected = execute_plan_named(&mut cpu_plan, &named).expect("cpu runs the matmul chain");

        let driver = CudaDriver::new(0).expect("a real cuda device");
        let mut plan = driver
            .plan(&program, &[], &[root], NumericPolicy::default(), &named)
            .expect("cuda plans the matmul chain");
        let actual = plan.execute_named(&named, None).expect("cuda runs the matmul chain");

        let inputs = 1 + CHAIN as u64;
        let uniforms_and_unpooled_intermediates = 2 * CHAIN as u64;
        assert!(
            plan.allocations() < inputs + uniforms_and_unpooled_intermediates,
            "degenerate gate: allocations {} must undercut inputs plus one uniform and one buffer per link {}",
            plan.allocations(),
            inputs + uniforms_and_unpooled_intermediates
        );
        let worst = actual
            .root()
            .iter()
            .zip(expected.root())
            .map(|(got, want)| (got - want).abs() / want.abs().max(1e-6))
            .fold(0.0f32, f32::max);
        assert_eq!(actual.root().len(), (ROWS * WIDTH) as usize, "degenerate gate: output compared");
        assert!(worst < 1e-4, "pooled chain diverges from cpu: worst relative {worst}");
    }
}

fn attention_oracle(
    rows: usize,
    heads: usize,
    groups: usize,
    head_dim: usize,
    planes: &[Vec<f32>; 5],
) -> Vec<f64> {
    let pair = head_dim / 2;
    let query_heads = heads * groups;
    let [query_low, query_high, key_low, key_high, values] = planes;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut out = vec![0.0; rows * query_heads * head_dim];
    for row in 0..rows {
        for head in 0..query_heads {
            let kv = head / groups;
            let dot = |key: usize| -> f64 {
                (0..pair)
                    .map(|d| {
                        let q = row * query_heads * pair + head * pair + d;
                        let k = key * heads * pair + kv * pair + d;
                        f64::from(query_low[q]) * f64::from(key_low[k])
                            + f64::from(query_high[q]) * f64::from(key_high[k])
                    })
                    .sum::<f64>()
                    * scale
            };
            let scores: Vec<f64> = (0..=row).map(dot).collect();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - peak).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..head_dim {
                let mixed: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(key, weight)| weight * f64::from(values[key * heads * head_dim + kv * head_dim + d]))
                    .sum();
                out[(row * query_heads + head) * head_dim + d] = mixed / total;
            }
        }
    }
    out
}

fn causal_attention_kind(rows: u64, heads: u64, groups: u64, head_dim: u64) -> proxima_tensor::BoundOpKind {
    proxima_tensor::BoundOpKind::CachedAttention {
        operands: (0..8)
            .map(|index| {
                (
                    NodeId(index),
                    proxima_tensor::Layout { base: 0, strides: vec![1_i64].into() },
                    None,
                )
            })
            .collect(),
        query_rows: rows,
        cached_key_rows: 0,
        new_key_rows: rows,
        kv_heads: heads,
        query_groups: groups,
        head_dim,
        rotary_dim: head_dim,
        scale: (1.0 / (head_dim as f64).sqrt()) as f32,
        cached_lower_inclusive: i64::MIN,
        new_upper_inclusive: 0,
    }
}

#[proxima::test]
#[case::tiled_prefill_512_rows_16q_8kv(512, 8, 2)]
#[case::tiled_ragged_37_rows_16q_8kv(37, 8, 2)]
#[case::tiled_ragged_37_rows_8q_2kv(37, 2, 4)]
#[case::untiled_15_rows_16q_8kv(15, 8, 2)]
#[case::untiled_single_row_decode(1, 8, 2)]
async fn cuda_cached_attention_matches_f64_oracle_with_causal_mask(
    #[case] rows: usize,
    #[case] heads: usize,
    #[case] groups: usize,
) {
    let head_dim = 128;
    let pair = head_dim / 2;
    let query_heads = heads * groups;
    let planes = [
        random_vec(11, rows * query_heads * pair),
        random_vec(12, rows * query_heads * pair),
        random_vec(13, rows * heads * pair),
        random_vec(14, rows * heads * pair),
        random_vec(15, rows * heads * head_dim),
    ];
    let op = proxima_tensor::BoundOp {
        node: NodeId(8),
        dtype: DType::FLOAT32,
        extents: vec![rows as u64, heads as u64, groups as u64, head_dim as u64],
        kind: causal_attention_kind(rows as u64, heads as u64, groups as u64, head_dim as u64),
    };
    let kernel = omega::emit_cuda(&op, &std::collections::BTreeMap::new()).expect("attention emits");
    let driver = omega::CudaDriver::new(0).expect("a real cuda device");
    let mut arena = driver.new_arena();
    let dummy = [0.0_f32];
    let order: [&[f32]; 8] = [
        &planes[0], &planes[1], &dummy, &dummy, &planes[2], &planes[3], &dummy, &planes[4],
    ];
    for (index, values) in order.iter().enumerate() {
        driver.upload_f32(&mut arena, NodeId(index as u32), values).expect("input uploads");
    }
    let vectors = (rows * query_heads) as i64;
    driver
        .launch_f32_persistent_sized(&mut arena, &kernel, &vectors.to_ne_bytes(), rows * query_heads * head_dim)
        .expect("attention launches");
    let actual = driver.read_f32(&arena, NodeId(8)).expect("output reads back");

    let expected = attention_oracle(rows, heads, groups, head_dim, &planes);
    let worst = actual
        .iter()
        .zip(&expected)
        .map(|(got, want)| (f64::from(*got) - want).abs())
        .fold(0.0, f64::max);
    eprintln!("attention parity rows={rows} heads={heads} groups={groups} worst_abs={worst}");
    assert_eq!(actual.len(), expected.len());
    assert!(worst < 1e-4, "rows={rows}: worst abs error {worst} vs f64 oracle");
}

fn decode_attention_kind(
    cached: u64,
    heads: u64,
    groups: u64,
    head_dim: u64,
) -> proxima_tensor::BoundOpKind {
    let proxima_tensor::BoundOpKind::CachedAttention {
        cached_lower_inclusive,
        new_upper_inclusive,
        scale,
        rotary_dim,
        ..
    } = causal_attention_kind(1, heads, groups, head_dim)
    else {
        unreachable!("causal_attention_kind builds a CachedAttention kind");
    };
    proxima_tensor::BoundOpKind::CachedAttention {
        operands: (0..9)
            .map(|index| {
                (
                    NodeId(index),
                    proxima_tensor::Layout { base: 0, strides: vec![1_i64].into() },
                    None,
                )
            })
            .collect(),
        query_rows: 1,
        cached_key_rows: cached,
        new_key_rows: 1,
        kv_heads: heads,
        query_groups: groups,
        head_dim,
        rotary_dim,
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
    }
}

fn decode_attention_oracle(
    cached: usize,
    heads: usize,
    groups: usize,
    head_dim: usize,
    planes: &[Vec<f32>; 7],
) -> Vec<f64> {
    let pair = head_dim / 2;
    let [query_low, query_high, cached_low, cached_high, new_low, new_high, cached_values] = planes;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut out = vec![0.0; heads * groups * head_dim];
    for head in 0..heads * groups {
        let kv = head / groups;
        let score = |key: usize| -> f64 {
            (0..pair)
                .map(|d| {
                    let q = head * pair + d;
                    let (low, high) = if key < cached {
                        (cached_low[key * heads * pair + kv * pair + d], cached_high[key * heads * pair + kv * pair + d])
                    } else {
                        (new_low[kv * pair + d], new_high[kv * pair + d])
                    };
                    f64::from(query_low[q]) * f64::from(low) + f64::from(query_high[q]) * f64::from(high)
                })
                .sum::<f64>()
                * scale
        };
        let scores: Vec<f64> = (0..=cached).map(score).collect();
        let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|value| (value - peak).exp()).collect();
        let total: f64 = weights.iter().sum();
        for d in 0..head_dim {
            let mixed: f64 = weights
                .iter()
                .enumerate()
                .map(|(key, weight)| {
                    let value = if key < cached {
                        cached_values[key * heads * head_dim + kv * head_dim + d]
                    } else {
                        cached_values[cached * heads * head_dim + kv * head_dim + d]
                    };
                    weight * f64::from(value)
                })
                .sum();
            out[head * head_dim + d] = mixed / total;
        }
    }
    out
}

#[proxima::test]
#[case::decode_cached_1(1)]
#[case::decode_cached_37(37)]
#[case::decode_cached_512(512)]
#[case::decode_cached_2048(2048)]
async fn cuda_split_decode_attention_matches_f64_oracle(#[case] cached: usize) {
    let (heads, groups, head_dim) = (8_usize, 2_usize, 128_usize);
    let pair = head_dim / 2;
    let query_heads = heads * groups;
    let cached_values_len = (cached + 1) * heads * head_dim;
    let planes = [
        random_vec(21, query_heads * pair),
        random_vec(22, query_heads * pair),
        random_vec(23, cached * heads * pair),
        random_vec(24, cached * heads * pair),
        random_vec(25, heads * pair),
        random_vec(26, heads * pair),
        random_vec(27, cached_values_len),
    ];
    let op = proxima_tensor::BoundOp {
        node: NodeId(20),
        dtype: DType::FLOAT32,
        extents: vec![1, heads as u64, groups as u64, head_dim as u64],
        kind: decode_attention_kind(cached as u64, heads as u64, groups as u64, head_dim as u64),
    };
    let packed = std::collections::BTreeMap::new();
    let device = omega::EmitDevice { multiprocessors: 36 };
    let kernel = omega::emit_cuda_on_device(&op, &packed, NumericPolicy::default(), device)
        .expect("decode attention emits for a known device");
    let unsplit = omega::emit_cuda(&op, &packed).expect("decode attention emits for an unknown device");
    assert_eq!(kernel.prologue.len(), 1, "a known device splits the key range");
    assert!(unsplit.prologue.is_empty(), "an unknown device keeps one block per vector");
    let driver = omega::CudaDriver::new(0).expect("a real cuda device");
    let mut arena = driver.new_arena();
    let new_value = &planes[6][cached * heads * head_dim..];
    let bound = [cached as f32];
    let dummy = [0.0_f32];
    let order: [&[f32]; 9] = [
        &planes[0], &planes[1], &planes[2], &planes[3], &planes[4], &planes[5],
        &planes[6][..cached.max(1) * heads * head_dim], new_value, &bound,
    ];
    for (index, values) in order.iter().enumerate() {
        let values = if values.is_empty() { &dummy[..] } else { values };
        driver.upload_f32(&mut arena, NodeId(index as u32), values).expect("input uploads");
    }
    let vectors = query_heads as i64;
    driver
        .launch_f32_persistent_sized(&mut arena, &kernel, &vectors.to_ne_bytes(), query_heads * head_dim)
        .expect("split attention launches");
    let actual = driver.read_f32(&arena, NodeId(20)).expect("output reads back");

    let expected = decode_attention_oracle(cached, heads, groups, head_dim, &planes);
    let worst = actual
        .iter()
        .zip(&expected)
        .map(|(got, want)| (f64::from(*got) - want).abs())
        .fold(0.0, f64::max);
    eprintln!("split decode attention cached={cached} worst_abs={worst}");
    assert_eq!(actual.len(), expected.len());
    assert!(worst < 1e-4, "cached={cached}: worst abs error {worst} vs f64 oracle");
}

const POLICY_PREFILL_TOKENS: usize = 512;
const POLICY_PREFILL_FEATURES: usize = 1024;
const POLICY_PREFILL_DEPTH: usize = 1024;
const POLICY_PREFILL_PROBES: [usize; 6] = [0, 1, 127, 128, 300, POLICY_PREFILL_TOKENS - 1];

fn round_through_fp16(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| half::f16::from_f32(*value).to_f32())
        .collect()
}

fn probed_oracle_max_diff(
    output: &[f32],
    weight: &[f32],
    activation: &[f32],
    max_magnitude: f32,
) -> f32 {
    let mut worst = 0.0f32;
    for &token in &POLICY_PREFILL_PROBES {
        for feature in 0..POLICY_PREFILL_FEATURES {
            let weight_row = &weight[feature * POLICY_PREFILL_DEPTH..][..POLICY_PREFILL_DEPTH];
            let exact: f64 = weight_row
                .iter()
                .enumerate()
                .map(|(depth, value)| {
                    f64::from(*value) * f64::from(activation[depth * POLICY_PREFILL_TOKENS + token])
                })
                .sum();
            let got = output[feature * POLICY_PREFILL_TOKENS + token];
            worst = worst.max((f64::from(got) - exact).abs() as f32);
        }
    }
    worst / max_magnitude
}

fn run_policy_prefill(
    weight_blocks: &[u8],
    codec: Codec,
    activation: &[f32],
    activation_quantization: bool,
) -> (Vec<f32>, Vec<String>) {
    let (program, root) = packed_batched_program(
        POLICY_PREFILL_FEATURES as u32,
        POLICY_PREFILL_DEPTH as u32,
        Some(POLICY_PREFILL_TOKENS as u32),
        DType::UINT8,
    );
    let named: Vec<(&str, QuantizedBlock<'_>)> = vec![
        ("weight", QuantizedBlock::Packed { codec, bytes: weight_blocks }),
        ("activation", QuantizedBlock::Float32(activation)),
    ];
    let mut plan = plan_named(
        Engine::Gpu,
        Some(GpuDriver::Cuda),
        &program,
        &[],
        &named,
        &[root],
        NumericPolicy::bit_exact()
            .with_reassociation(true)
            .with_activation_quantization(activation_quantization),
    )
    .expect("omega::backend plans the policy prefill matmul");
    let identities = cuda_kernel_identities(&plan).expect("cuda plan exposes its kernels");
    let output = execute_plan_named(&mut plan, &named)
        .expect("omega::backend runs the policy prefill matmul")
        .root()
        .to_vec();
    (output, identities)
}

fn assert_policy_prefill_parity(
    codec_name: &str,
    codec: Codec,
    block_bytes: usize,
    block_elements: usize,
    quantize: fn(&[f32], &mut [u8]) -> Result<(), proxima_gguf::quant::QuantError>,
    dequantize: fn(&[u8], &mut [f32]) -> Result<(), proxima_gguf::quant::QuantError>,
) {
    let blocks_per_row = POLICY_PREFILL_DEPTH / block_elements;
    let weight_f32: Vec<f32> = random_vec(53, POLICY_PREFILL_FEATURES * POLICY_PREFILL_DEPTH)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let mut weight_blocks = vec![0u8; POLICY_PREFILL_FEATURES * blocks_per_row * block_bytes];
    let mut dequantized = vec![0.0f32; weight_f32.len()];
    let depth = weight_f32.len() / POLICY_PREFILL_FEATURES;
    for ((row_f32, row_blocks), row_dequantized) in weight_f32
        .chunks_exact(depth)
        .zip(weight_blocks.chunks_exact_mut(blocks_per_row * block_bytes))
        .zip(dequantized.chunks_exact_mut(depth))
    {
        quantize(row_f32, row_blocks).expect("row length is a whole number of blocks");
        dequantize(row_blocks, row_dequantized).expect("a whole number of packed blocks");
    }
    let activation: Vec<f32> = random_vec(59, POLICY_PREFILL_DEPTH * POLICY_PREFILL_TOKENS)
        .into_iter()
        .map(|value| value * 4.0 - 2.0)
        .collect();
    let rounded = round_through_fp16(&activation);

    let (single, single_identities) = run_policy_prefill(&weight_blocks, codec, &activation, true);
    assert!(
        single_identities.iter().any(|identity| identity.contains("_mma16x8x8h_")),
        "{codec_name}: policy-on prefill did not take the single-fp16 mma kernel: {single_identities:?}"
    );
    let (split, split_identities) = run_policy_prefill(&weight_blocks, codec, &activation, false);
    assert!(
        split_identities
            .iter()
            .all(|identity| !identity.contains("_mma16x8x8h_"))
            && split_identities.iter().any(|identity| identity.contains("_mma16x8x8_")),
        "{codec_name}: policy-off prefill must keep the hi/lo mma kernel: {split_identities:?}"
    );

    let max_magnitude = single.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
    let against_rounded = probed_oracle_max_diff(&single, &dequantized, &rounded, max_magnitude);
    let against_unrounded = probed_oracle_max_diff(&single, &dequantized, &activation, max_magnitude);
    assert!(
        single.iter().all(|value| value.is_finite()) && against_rounded < 2e-5,
        "{codec_name}: single-fp16 prefill disagrees with the fp16-rounded f64 oracle by {against_rounded} of peak"
    );
    assert!(
        against_unrounded > 2.0 * against_rounded,
        "{codec_name}: degenerate control: output matches the unrounded activations ({against_unrounded}) as tightly as the rounded ones ({against_rounded})"
    );
    let split_against_exact = probed_oracle_max_diff(&split, &dequantized, &activation, max_magnitude);
    assert!(
        split_against_exact < 1e-5,
        "{codec_name}: policy-off prefill lost f32 activation precision: {split_against_exact} of peak"
    );
}

#[test]
fn policy_on_q8_0_prefill_matches_the_fp16_rounded_f64_oracle_on_cuda() {
    use proxima_gguf::quant::q8_0::{BLOCK_BYTES, QK8_0, dequantize, quantize};
    assert_policy_prefill_parity("q8_0", Codec::Q8_0, BLOCK_BYTES, QK8_0, quantize, dequantize);
}

#[test]
fn policy_on_q2_0_prefill_matches_the_fp16_rounded_f64_oracle_on_cuda() {
    use proxima_gguf::quant::q2_0::{BLOCK_BYTES, QK_Q2_0, dequantize, quantize};
    assert_policy_prefill_parity("q2_0", Codec::Q2_0, BLOCK_BYTES, QK_Q2_0, quantize, dequantize);
}
