//! First-ever execution of an autograd-produced backward graph + optimizer
//! step on a GPU backend, run through the SAME `plan_named`/
//! `execute_plan_named` wrapper `backend_parity.rs`/`wgpu_parity.rs` use for
//! forward-only graphs.
//!
//! # Known boundary
//!
//! [`proxima_autograd::adjoint::Differentiated`]'s `gathered` contributions
//! are applied HOST-side via sparse scatter-add (that field's own doc,
//! `proxima-autograd/src/adjoint.rs:132`) -- outside `Differentiated.program`
//! entirely, so no backend ever sees them. This test uses a small dense MLP
//! (matmul + bias + relu, matmul + bias + softmax-cross-entropy, the same
//! network `proxima-autograd/tests/train_fit.rs` builds), which has no
//! gather anywhere in its forward pass, so its backward has no `gathered`
//! contribution and the entire backward + Adam step is in-graph.
//!
//! # Three arms
//!
//! CPU is the oracle every GPU backend is compared against, never the other
//! way around (guiding principle 14). Metal and wgpu are each gated exactly
//! like `backend_parity.rs`/`wgpu_parity.rs` gate them, so a build missing
//! either feature or platform simply does not compile that arm's test.

#![cfg(feature = "cpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use omega::backend::{Backend, execute_plan_named, plan_named};
use proxima_autograd::activation::relu;
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::loss::softmax_cross_entropy;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, map};

const IN_DIM: usize = 3;
const HIDDEN_DIM: usize = 4;
const OUT_DIM: usize = 2;

fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    append(program, Op::Elementwise { dtype: DType::Float32, body, operands, name: None })
}

fn reduce_add(program: &mut Vec<Op>, operand: NodeId, in_map: IndexMap, out_map: IndexMap) -> NodeId {
    append(
        program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map,
            out_map,
            keep: Keep::Reduce,
            name: None,
        }),
    )
}

fn identity(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

fn dense(program: &mut Vec<Op>, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let product = elementwise(program, ScalarOp::Multiply, vec![(w, identity(2)), (x, IndexMap::Affine(map::projection(2, &[0])))]);
    let matmul = reduce_add(program, product, identity(2), IndexMap::Affine(map::projection(2, &[1])));
    elementwise(program, ScalarOp::Add, vec![(matmul, identity(1)), (b, identity(1))])
}

fn counter_pattern(seed: usize, count: usize) -> Vec<f32> {
    (0..count).map(|index| (((seed + index) * 7 % 13) as f32 - 6.0) / 12.0).collect()
}

/// One MLP training step: forward, `differentiate`, then an Adam update for
/// every parameter -- structurally identical to
/// `proxima-autograd/tests/train_fit.rs`'s `build_network` plus its
/// hand-built Adam wiring, assembled once here so both the CPU oracle and
/// every GPU arm plan the exact same program.
struct TrainingStep {
    program: Vec<Op>,
    loss: NodeId,
    /// `(output_node, rebind_name)` for every `new_param`/`new_m`/`new_v` --
    /// the 12 outputs a caller requests and rebinds under the ORIGINAL
    /// names for the next step, mirroring `train_fit.rs`'s `rebind` list.
    rebind: Vec<(NodeId, &'static str)>,
}

fn build_training_step() -> TrainingStep {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", vec![Extent::Static(IN_DIM as u32)]);
    let y = leaf(&mut program, "y", vec![Extent::Static(OUT_DIM as u32)]);
    let w1 = leaf(&mut program, "w1", vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let b1 = leaf(&mut program, "b1", vec![Extent::Static(HIDDEN_DIM as u32)]);
    let w2 = leaf(&mut program, "w2", vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let b2 = leaf(&mut program, "b2", vec![Extent::Static(OUT_DIM as u32)]);

    let h_pre = dense(&mut program, x, w1, b1);
    let h = relu(&mut program, DType::Float32, h_pre, 1);
    let out_pre = dense(&mut program, h, w2, b2);
    let loss = softmax_cross_entropy(&mut program, DType::Float32, out_pre, y, 1, 0);

    let differentiated = differentiate(&program, loss).expect("scalar loss differentiates");
    let grad_w1 = differentiated.gradient_of_named("w1").expect("w1 feeds the loss");
    let grad_b1 = differentiated.gradient_of_named("b1").expect("b1 feeds the loss");
    let grad_w2 = differentiated.gradient_of_named("w2").expect("w2 feeds the loss");
    let grad_b2 = differentiated.gradient_of_named("b2").expect("b2 feeds the loss");
    let mut program = differentiated.program;

    let config = AdamConfig { learning_rate: 0.05, ..AdamConfig::default() };
    let step_node = step_input(&mut program, "step");

    let m_w1 = leaf(&mut program, "m_w1", vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let v_w1 = leaf(&mut program, "v_w1", vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let m_b1 = leaf(&mut program, "m_b1", vec![Extent::Static(HIDDEN_DIM as u32)]);
    let v_b1 = leaf(&mut program, "v_b1", vec![Extent::Static(HIDDEN_DIM as u32)]);
    let m_w2 = leaf(&mut program, "m_w2", vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let v_w2 = leaf(&mut program, "v_w2", vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let m_b2 = leaf(&mut program, "m_b2", vec![Extent::Static(OUT_DIM as u32)]);
    let v_b2 = leaf(&mut program, "v_b2", vec![Extent::Static(OUT_DIM as u32)]);

    let (new_w1, new_m_w1, new_v_w1) = adam_step(&mut program, &config, 2, AdamOperands { param: w1, grad: grad_w1, m: m_w1, v: v_w1 }, step_node);
    let (new_b1, new_m_b1, new_v_b1) = adam_step(&mut program, &config, 1, AdamOperands { param: b1, grad: grad_b1, m: m_b1, v: v_b1 }, step_node);
    let (new_w2, new_m_w2, new_v_w2) = adam_step(&mut program, &config, 2, AdamOperands { param: w2, grad: grad_w2, m: m_w2, v: v_w2 }, step_node);
    let (new_b2, new_m_b2, new_v_b2) = adam_step(&mut program, &config, 1, AdamOperands { param: b2, grad: grad_b2, m: m_b2, v: v_b2 }, step_node);

    let rebind = vec![
        (new_w1, "w1"),
        (new_m_w1, "m_w1"),
        (new_v_w1, "v_w1"),
        (new_b1, "b1"),
        (new_m_b1, "m_b1"),
        (new_v_b1, "v_b1"),
        (new_w2, "w2"),
        (new_m_w2, "m_w2"),
        (new_v_w2, "v_w2"),
        (new_b2, "b2"),
        (new_m_b2, "m_b2"),
        (new_v_b2, "v_b2"),
    ];

    TrainingStep { program, loss, rebind }
}

fn initial_state() -> BTreeMap<String, Vec<f32>> {
    let mut named = BTreeMap::new();
    named.insert("w1".into(), counter_pattern(1, IN_DIM * HIDDEN_DIM));
    named.insert("b1".into(), counter_pattern(2, HIDDEN_DIM));
    named.insert("w2".into(), counter_pattern(3, HIDDEN_DIM * OUT_DIM));
    named.insert("b2".into(), counter_pattern(4, OUT_DIM));
    named.insert("m_w1".into(), vec![0.0f32; IN_DIM * HIDDEN_DIM]);
    named.insert("v_w1".into(), vec![0.0f32; IN_DIM * HIDDEN_DIM]);
    named.insert("m_b1".into(), vec![0.0f32; HIDDEN_DIM]);
    named.insert("v_b1".into(), vec![0.0f32; HIDDEN_DIM]);
    named.insert("m_w2".into(), vec![0.0f32; HIDDEN_DIM * OUT_DIM]);
    named.insert("v_w2".into(), vec![0.0f32; HIDDEN_DIM * OUT_DIM]);
    named.insert("m_b2".into(), vec![0.0f32; OUT_DIM]);
    named.insert("v_b2".into(), vec![0.0f32; OUT_DIM]);
    named
}

fn as_named_blocks(owned: &[(String, Vec<f32>)]) -> Vec<(&str, QuantizedBlock<'_>)> {
    owned.iter().map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data.as_slice()))).collect()
}

fn outputs_of(step: &TrainingStep) -> Vec<NodeId> {
    step.rebind.iter().map(|(node, _)| *node).collect()
}

/// Runs one training step on `backend` and returns `(new_param, new_m,
/// new_v)` for every output node, in `step.rebind`'s order.
fn run_one_step(backend: Backend, step: &TrainingStep, batch: &[(String, Vec<f32>)]) -> Vec<Vec<f32>> {
    let named_blocks = as_named_blocks(batch);
    let outputs = outputs_of(step);
    let mut plan = plan_named(backend, &step.program, &[], &named_blocks, &outputs)
        .unwrap_or_else(|error| panic!("{} plans the training step: {error}", backend.name()));
    let evaluated = execute_plan_named(&mut plan, &named_blocks)
        .unwrap_or_else(|error| panic!("{} executes the training step: {error}", backend.name()));
    outputs
        .iter()
        .map(|node| evaluated.get(*node).unwrap_or_else(|| panic!("{} produced no output for {node:?}", backend.name())).0.to_vec())
        .collect()
}

fn one_step_batch() -> Vec<(String, Vec<f32>)> {
    let mut lcg = Lcg(7);
    let example: Vec<f32> = (0..IN_DIM).map(|_| lcg.next_unit()).collect();
    let mut label = vec![0.0f32; OUT_DIM];
    label[0] = 1.0;

    let mut batch = initial_state();
    batch.insert("x".into(), example);
    batch.insert("y".into(), label);
    batch.insert("step".into(), vec![1.0]);
    batch.into_iter().collect()
}

fn assert_parity(backend_name: &str, cpu: &[Vec<f32>], gpu: &[Vec<f32>], tolerance: f32) {
    assert_eq!(cpu.len(), gpu.len(), "{backend_name}: output count mismatch");
    let mut worst_diff = 0.0f32;
    for (cpu_values, gpu_values) in cpu.iter().zip(gpu.iter()) {
        assert_eq!(cpu_values.len(), gpu_values.len(), "{backend_name}: buffer length mismatch");
        for (&want, &got) in cpu_values.iter().zip(gpu_values.iter()) {
            assert!(got.is_finite(), "{backend_name} produced a non-finite value: {got}");
            worst_diff = worst_diff.max((want - got).abs());
        }
    }
    eprintln!("{backend_name} training-step parity: worst_diff={worst_diff} tolerance={tolerance}");
    assert!(worst_diff < tolerance, "{backend_name} disagrees with cpu on the training step: worst_diff={worst_diff} tolerance={tolerance}");
}

/// Proves the entire forward + backward + Adam-step graph
/// [`differentiate`]/[`adam_step`] emit plans and executes on Metal, and
/// that every `new_param`/`new_m`/`new_v` buffer agrees with the CPU
/// evaluator -- the same wrapper `backend_parity.rs` uses for a
/// forward-only graph, with an absolute tolerance (these are small
/// optimizer-state values, not logits, so a relative tolerance would divide
/// by a near-zero magnitude).
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn a_training_step_runs_on_metal_at_cpu_parity() {
    let step = build_training_step();
    let batch = one_step_batch();
    let cpu = run_one_step(Backend::Cpu, &step, &batch);
    let metal = run_one_step(Backend::Metal, &step, &batch);
    assert_parity("metal", &cpu, &metal, 1e-4);
}

/// Proves the same fused-backward+Adam graph
/// `a_training_step_runs_on_metal_at_cpu_parity` proves on Metal also runs on
/// the portable wgpu/WGSL backend, at the same CPU-oracle parity tolerance.
/// `differentiate` + `adam_step` fuse the whole backward-plus-optimizer
/// update into ONE `Op::Elementwise` per parameter with 13 storage-buffer
/// operands; that used to exceed `wgpu::Limits::default()`'s
/// `max_storage_buffers_per_shader_stage` (8) and surface as an uncaught
/// validation panic (see `wgpu_driver::acquire_device`'s doc, and the git
/// history of this test for the prior `#[should_panic]` shape). Fixed by
/// requesting `adapter.limits()` at device-acquisition time rather than the
/// portable-safe default.
#[cfg(feature = "wgpu-backend")]
#[test]
fn a_training_step_runs_on_wgpu_at_cpu_parity() {
    let step = build_training_step();
    let batch = one_step_batch();
    let cpu = run_one_step(Backend::Cpu, &step, &batch);
    let wgpu = run_one_step(Backend::Wgpu, &step, &batch);
    assert_parity("wgpu", &cpu, &wgpu, 1e-4);
}

/// Constructs a graph that exceeds even the (now-requested) adapter's real
/// `max_storage_buffers_per_shader_stage` -- a chain of pairwise
/// `ScalarOp::Add` nodes (`ScalarOp::Add` is fixed-arity 2, so N leaf inputs
/// takes a chain, not one N-ary node), each intermediate consumed exactly
/// once so `proxima_tensor::bind` fuses the whole chain into ONE
/// `BoundOpKind::Elementwise` (see that field's `ComposedBody` doc) carrying
/// every leaf as a distinct storage-buffer operand -- and asserts
/// the driver returns [`omega::wgpu_driver::WgpuError::TooManyStorageBuffers`]
/// as a named `Result::Err`, never a panic (see that variant's own doc: the
/// binding count is pre-validated before `create_compute_pipeline`, which has
/// no error scope around it).
#[cfg(feature = "wgpu-backend")]
#[test]
fn a_graph_past_the_adapter_storage_buffer_limit_is_a_named_error_on_wgpu() {
    use omega::wgpu_driver::WgpuError;

    let device_limit = {
        let probe_program = vec![Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(1)],
            name: Some("probe".into()),
        }];
        let probe_data = vec![0.0f32];
        let probe_named: Vec<(&str, QuantizedBlock<'_>)> = vec![("probe", QuantizedBlock::Float32(&probe_data))];
        let probe_plan = omega::wgpu_driver::plan_named(&probe_program, &[], &probe_named, &[NodeId(0)])
            .expect("a single-input identity program plans on wgpu");
        probe_plan.limits().max_storage_buffers_per_shader_stage
    };
    // 1 output + 1 uniforms binding are always present, so this many leaf
    // inputs pushes the fused elementwise node's binding count one past the
    // limit.
    let addend_count = device_limit as usize;

    let mut program = Vec::new();
    let identity = IndexMap::Affine(map::projection(1, &[0]));
    let addends: Vec<NodeId> = (0..addend_count)
        .map(|index| leaf(&mut program, &format!("addend{index}"), vec![Extent::Static(1)]))
        .collect();
    let sum = addends
        .iter()
        .skip(1)
        .fold(addends[0], |accumulator, addend| elementwise(&mut program, ScalarOp::Add, vec![(accumulator, identity.clone()), (*addend, identity.clone())]));

    let owned: Vec<(String, Vec<f32>)> = addends.iter().enumerate().map(|(index, _)| (format!("addend{index}"), vec![1.0f32])).collect();
    let named_blocks = as_named_blocks(&owned);

    let mut plan = plan_named(Backend::Wgpu, &program, &[], &named_blocks, &[sum]).expect("this program plans (limit is checked at dispatch, not plan)");
    let error = execute_plan_named(&mut plan, &named_blocks).expect_err("a graph past the adapter's storage-buffer limit is a named error, not a panic");
    match error {
        omega::backend::BackendError::Wgpu(WgpuError::TooManyStorageBuffers { needed, limit, .. }) => {
            assert_eq!(limit, device_limit, "the named error reports this device's actual limit");
            assert!(needed > limit, "needed ({needed}) must exceed limit ({limit}) for this to be the right error");
        }
        other => panic!("expected WgpuError::TooManyStorageBuffers, got {other:?}"),
    }
}

/// Proves state REBINDING across steps works on-device, not just one kernel
/// launch: ten training steps in a row on `backend`, each rebinding
/// `new_param`/`new_m`/`new_v` under the original names for the next
/// step's `plan_named` call -- the multi-step shape
/// `proxima-autograd/src/train.rs`'s `fit` already proves on CPU, now
/// proven on-device. Returns the per-step loss curve.
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "wgpu-backend"))]
fn run_multi_step_on(backend: Backend) -> Vec<f32> {
    let step = build_training_step();
    let mut outputs = outputs_of(&step);
    outputs.push(step.loss);

    let examples: [[f32; IN_DIM]; 4] = [[1.0, 0.5, 0.2], [-1.0, -0.5, 0.3], [0.8, -0.2, 0.1], [-0.3, 0.1, 0.9]];
    let labels: [[f32; OUT_DIM]; 4] = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];

    let mut state = initial_state();
    const STEPS: u32 = 10;
    let mut loss_curve: Vec<f32> = Vec::new();

    for step_number in 1..=STEPS {
        let batch_index = ((step_number - 1) as usize) % examples.len();
        state.insert("x".into(), examples[batch_index].to_vec());
        state.insert("y".into(), labels[batch_index].to_vec());
        state.insert("step".into(), vec![f32::from(step_number as u16)]);

        let owned: Vec<(String, Vec<f32>)> = state.iter().map(|(name, data)| (name.clone(), data.clone())).collect();
        let named_blocks = as_named_blocks(&owned);

        let mut plan = plan_named(backend, &step.program, &[], &named_blocks, &outputs)
            .unwrap_or_else(|error| panic!("{} plans training step {step_number}: {error}", backend.name()));
        let evaluated = execute_plan_named(&mut plan, &named_blocks)
            .unwrap_or_else(|error| panic!("{} executes training step {step_number}: {error}", backend.name()));

        let step_loss = evaluated.get(step.loss).unwrap_or_else(|| panic!("{} produced no loss output", backend.name())).0[0];
        loss_curve.push(step_loss);

        for (node, name) in &step.rebind {
            let (data, _shape) = evaluated.get(*node).unwrap_or_else(|| panic!("{} produced no output for {node:?}", backend.name()));
            state.insert((*name).into(), data.to_vec());
        }
    }

    loss_curve
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn ten_training_steps_rebind_state_and_the_loss_drops_on_metal() {
    let loss_curve = run_multi_step_on(Backend::Metal);
    eprintln!("metal multi-step loss curve: {loss_curve:?}");
    assert!(loss_curve.iter().all(|value| value.is_finite()), "loss went non-finite on metal: {loss_curve:?}");
    assert!(
        loss_curve.last().expect("at least one step ran") < &loss_curve[0],
        "expected the loss to drop across 10 rebound steps on metal, got {loss_curve:?}"
    );
}

/// The wgpu counterpart of `ten_training_steps_rebind_state_and_the_loss_drops_on_metal`
/// -- the multi-step loop hits the identical 13-storage-buffer fused pipeline
/// on its very first iteration, now within `adapter.limits()`'s real ceiling
/// (see `wgpu_driver::acquire_device`'s doc) rather than rejected by it.
#[cfg(feature = "wgpu-backend")]
#[test]
fn ten_training_steps_rebind_state_and_the_loss_drops_on_wgpu() {
    let loss_curve = run_multi_step_on(Backend::Wgpu);
    eprintln!("wgpu multi-step loss curve: {loss_curve:?}");
    assert!(loss_curve.iter().all(|value| value.is_finite()), "loss went non-finite on wgpu: {loss_curve:?}");
    assert!(
        loss_curve.last().expect("at least one step ran") < &loss_curve[0],
        "expected the loss to drop across 10 rebound steps on wgpu, got {loss_curve:?}"
    );
}
