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
use support::{as_named_blocks, qwen3_gqa_qk_norm_forward_fixture, real_forward_fixture};

use omega::Codec;
use omega::backend::{BackendError, Engine, GpuDriver, execute_plan_named, plan_named};
use proxima_tensor::NumericPolicy;
use proxima_tensor::infer;
use proxima_tensor::spec::qk_norm_cached_forward_program;
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

fn elementwise_tanh_program(extent: u32) -> Vec<Op> {
    let mut program = Vec::new();
    let source = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: Some("x".into()),
        },
    );
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
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
            dtype: DType::Float32,
            shape: vec![Extent::Static(rows), Extent::Static(cols)],
            name: Some("x".into()),
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
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
            dtype: DType::Float32,
            shape: vec![Extent::Static(rows), Extent::Static(cols)],
            name: Some("x".into()),
        },
    );
    append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
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
                    len: None,
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
            dtype: DType::Float32,
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
        DType::Float32,
        vec![Extent::Static(vocab), Extent::Static(dim)],
        "table",
    );
    let ids_a = append_input(
        &mut program,
        DType::Int32,
        vec![Extent::Static(seq)],
        "ids_a",
    );
    let ids_b = append_input(
        &mut program,
        DType::Int32,
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
            dtype: DType::Float32,
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
            dtype: DType::Float32,
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
        DType::Float32,
        vec![Extent::Static(vocab as u32), Extent::Static(dim as u32)],
        "table",
    );
    let ids = append_input(
        &mut program,
        DType::Int32,
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
    tolerance: f32,
) {
    let rows: u32 = 3;
    let blocks_per_row = 2usize;
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

    let (packed_program, packed_sum) = packed_matmul_program(rows, k, DType::UInt8);
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
        NumericPolicy::default(),
    )
    .expect("omega::backend plans the packed matmul on cuda");
    let cuda = execute_plan_named(&mut cuda_plan, &named)
        .expect("omega::backend runs the packed matmul on a real cuda device");

    let (f32_program, f32_sum) = packed_matmul_program(rows, k, DType::Float32);
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
        1e-5,
    );
}

/// `cuda.rs`'s `kernel_signature` declares the OUTPUT pointer itself as
/// `__half*` for a `DType::Float16` op (unlike `crate::wgsl`, which always
/// casts an f16 compute result back to an `array<f32>` storage write) -- see
/// `omega::cuda_driver`'s module doc. v1 does not allocate/read back that
/// narrower buffer, so this proves the documented rejection actually fires
/// rather than silently misreading a `__half`-packed buffer as `f32`.
fn f16_dtype_reduce_program(m: u32, k: u32, n: u32) -> Vec<Op> {
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
                dtype: DType::Float16,
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
            dtype: DType::Float32,
            shape: vec![Extent::Static(vocab), Extent::Static(dim)],
            name: Some("table".into()),
        },
    );
    let scored = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(seq), Extent::Static(candidates)],
            name: Some("candidates".into()),
        },
    );
    let route = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Int32,
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
            dtype: DType::Float32,
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
            dtype: DType::Float32,
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
            dtype: DType::Int32,
            shape: vec![Extent::Static(tokens)],
            name: Some("route".into()),
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
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
            dtype: DType::Float32,
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
            dtype: DType::Float32,
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
            DType::Float32,
            vec![Extent::Static(seq), Extent::Static(embedding)],
            "x",
        );
        let router_logits = input_leaf(
            &mut program,
            DType::Float32,
            vec![Extent::Static(seq), Extent::Static(expert_count)],
            "router_logits",
        );
        let slab = |program: &mut Vec<Op>, name: &str, rows: u32, cols: u32| {
            input_leaf(
                program,
                DType::Float32,
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
        let (program, sum) = packed_matmul_program(ROWS, K as u32, DType::UInt8);
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
