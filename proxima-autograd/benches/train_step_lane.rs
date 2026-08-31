//! Sealed baseline for `proxima-autograd`'s `train::train_step` (proxima
//! discipline-log campaign, `proxima-tensor/docs/discipline.md` ROW 159+).
//! Measures per-step latency of one real training step -- forward, backward,
//! and one Adam update for every parameter -- on the same `784-128-10` MLP,
//! batch 32, real MNIST digits `proxima-autograd/tests/real_mnist_training.rs`
//! trains, against the PyTorch incumbent recorded in ROW 157
//! (`proxima-onnx/scripts/torch_reference/train_bench.py --threads 1`).
//!
//! `train-step-bench`, default-off: presence-guarded on
//! `~/.cache/burn-dataset/mnist`, clean skip (prints and returns) when
//! absent. Deliberately does NOT enable `proxima-tensor/instrument` -- see
//! `mnist_f32_lane.rs`'s own doc for why (30-40% overhead on every
//! `run_reduce`/`run_elementwise` call); the per-node-kind breakdown for
//! this same program is captured separately (discipline log row, not this
//! file) with `instrument` on for one untimed run.
//!
//! **Convergence gate, on the benched run itself, not a separate one:** the
//! manual sweep below threads real Adam state (`m`/`v`/`step`) forward one
//! real batch at a time -- not the same batch repeated -- and asserts the
//! loss actually drops over the run. A step whose timing loop stopped being
//! a real training step (e.g. reusing stale state) would still show a flat
//! or non-finite loss curve and trip this gate.
//!
//! Re-prove with (host must be quiet; see the discipline log row this bench
//! seeds for the loadout it was actually measured under):
//! `CARGO_TARGET_DIR=<scratch> cargo bench -p proxima-autograd --bench train_step_lane --features train-step-bench -- --save-baseline train-step-lane`
//!
//! **The `scratch` arm is a sealed NEGATIVE result, not a library API.**
//! `train_step_scratch` below composes `evaluate_quantized_named_with_scratch`
//! directly (bench-local, never landed in `proxima-autograd::train`) --
//! ROW 159 measured a caller-carried `free_buffers` pool 15-30% SLOWER than
//! `train::train_step`'s own fresh-`Vec::new()`-per-call baseline on this
//! exact program, and rolled the library change back (`git diff` on
//! `proxima-autograd/src/train.rs` is empty against the pre-session tree).
//! This arm stays in the sealed bench so the negative number re-proves from
//! the artifact alone (principle 16) without carrying dead, unbeneficial
//! surface into the library (principle 1).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use criterion::Criterion;
use proxima_autograd::activation::relu;
use proxima_autograd::adjoint::differentiate_wanted;
use proxima_autograd::loss::softmax_cross_entropy;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_autograd::train::{State, train_step};
use proxima_tensor::cpu::{QuantizedBlock, StaticArena, build_static_arena, evaluate_named_with_arena, evaluate_quantized_named_with_scratch};
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

/// [`train_step_scratch`]'s two threaded-across-the-run pieces, named once so
/// the criterion closure below doesn't repeat the tuple type at both the
/// `RefCell` declaration and the destructure.
type ScratchPool = (Vec<Vec<f32>>, Option<BTreeSet<NodeId>>);

/// Bench-local twin of [`train_step`], composing
/// [`evaluate_quantized_named_with_scratch`] directly instead of
/// `train::train_step`'s own `evaluate_named` (fresh `free_buffers` every
/// call) -- see this file's own top-level doc for why this stays
/// bench-local rather than a `train::train_step_with_scratch` landing.
fn train_step_scratch(
    program: &[Op],
    loss: NodeId,
    rebind: &[(NodeId, &str)],
    named: &[(&str, &[f32])],
    free_buffers: &mut Vec<Vec<f32>>,
    validated_weight_nodes: &mut Option<BTreeSet<NodeId>>,
) -> (f32, State) {
    let mut outputs = Vec::with_capacity(rebind.len() + 1);
    outputs.push(loss);
    outputs.extend(rebind.iter().map(|(node, _)| *node));
    let wrapped: Vec<(&str, QuantizedBlock)> = named.iter().map(|(name, data)| (*name, QuantizedBlock::Float32(data))).collect();
    let evaluated = evaluate_quantized_named_with_scratch(program, &[], &wrapped, &outputs, free_buffers, validated_weight_nodes)
        .expect("train_step_scratch evaluates");
    let loss_value = evaluated.get(loss).and_then(|(data, _)| data.first().copied()).unwrap_or(0.0);
    let next_state = rebind
        .iter()
        .map(|(node, name)| {
            let values = evaluated.get(*node).map_or_else(Vec::new, |(data, _)| data.to_vec());
            (String::from(*name), values)
        })
        .collect();
    (loss_value, next_state)
}

/// Bench-local twin of [`train_step`], composing [`build_static_arena`] +
/// [`evaluate_named_with_arena`] instead of `train::train_step`'s own
/// `evaluate_named` -- ROW 164's static-arena lever: `arena` is built ONCE
/// outside the sweep loop, so every call here reuses the SAME per-node
/// buffers `build_static_arena` sized once, with no `shape::infer`, no
/// `bind::bind`, and no per-node `Vec` allocation on this call's own path.
fn train_step_arena(arena: &mut StaticArena, named: &[(&str, &[f32])], loss: NodeId, rebind: &[(NodeId, &str)]) -> (f32, State) {
    let evaluated = evaluate_named_with_arena(arena, named).expect("train_step_arena evaluates");
    let loss_value = evaluated.get(loss).and_then(|(data, _)| data.first().copied()).unwrap_or(0.0);
    let next_state = rebind
        .iter()
        .map(|(node, name)| {
            let values = evaluated.get(*node).map_or_else(Vec::new, |(data, _)| data.to_vec());
            (String::from(*name), values)
        })
        .collect();
    (loss_value, next_state)
}

const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const IN_DIM: usize = 28 * 28;
const HIDDEN_DIM: usize = 128;
const OUT_DIM: usize = 10;
const BATCH: usize = 32;
// enough real examples for WARMUP+MEASURED distinct batches, no cycling.
const TRAIN_EXAMPLES: usize = 4096;
const WARMUP_STEPS: usize = 20;
const MEASURED_STEPS: usize = 100;

fn train_images_path() -> PathBuf {
    Path::new(DATASET_DIR).join("train/train-images-idx3-ubyte")
}
fn train_labels_path() -> PathBuf {
    Path::new(DATASET_DIR).join("train/train-labels-idx1-ubyte")
}

fn dataset_present() -> bool {
    train_images_path().exists() && train_labels_path().exists()
}

/// Verbatim of `real_mnist_training.rs`'s own idx3/idx1 header parse.
fn idx_header(bytes: &[u8]) -> (usize, Vec<usize>) {
    let dimension_count = bytes[3] as usize;
    let item_count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut extents = Vec::with_capacity(dimension_count - 1);
    for axis in 1..dimension_count {
        let offset = 4 + axis * 4;
        extents.push(u32::from_be_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as usize);
    }
    (item_count, extents)
}

fn load_normalized_images(path: &Path, limit: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read idx3 image file");
    let (item_count, extents) = idx_header(&bytes);
    let pixel_count = extents.iter().product::<usize>();
    let take = item_count.min(limit);
    let header_length = 4 + extents.len() * 4 + 4;
    bytes[header_length..header_length + take * pixel_count].iter().map(|&pixel| ((pixel as f32 / 255.0) - 0.1307) / 0.3081).collect()
}

fn load_one_hot_labels(path: &Path, limit: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read idx1 label file");
    let (item_count, _extents) = idx_header(&bytes);
    let take = item_count.min(limit);
    let raw = &bytes[8..8 + take];
    let mut one_hot = vec![0.0f32; take * OUT_DIM];
    for (index, &label) in raw.iter().enumerate() {
        one_hot[index * OUT_DIM + label as usize] = 1.0;
    }
    one_hot
}

fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
}

fn identity(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

fn axes(rank: u16, selected: &[u16]) -> IndexMap {
    IndexMap::Affine(map::projection(rank, selected))
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(program, Op::Elementwise { dtype: DType::Float32, body, operands, name: None })
}

fn reduce_add(program: &mut Vec<Op>, operand: NodeId, in_map: IndexMap, out_map: IndexMap) -> NodeId {
    op::append(
        program,
        Op::Reduce(op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map,
            out_map,
            keep: op::Keep::Reduce,
            name: None,
        }),
    )
}

/// `real_mnist_training.rs`'s own `batched_dense`, verbatim -- a bench
/// binary cannot depend on a sibling test binary's helpers.
fn batched_dense(program: &mut Vec<Op>, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let product = elementwise(program, ScalarOp::Multiply, vec![(w, axes(3, &[1, 2])), (x, axes(3, &[0, 1]))]);
    let matmul = reduce_add(program, product, identity(3), axes(3, &[0, 2]));
    elementwise(program, ScalarOp::Add, vec![(matmul, identity(2)), (b, axes(2, &[1]))])
}

struct Network {
    program: Vec<Op>,
    w1: NodeId,
    b1: NodeId,
    w2: NodeId,
    b2: NodeId,
    loss: NodeId,
}

fn build_network() -> Network {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", vec![Extent::Static(BATCH as u32), Extent::Static(IN_DIM as u32)]);
    let y = leaf(&mut program, "y", vec![Extent::Static(BATCH as u32), Extent::Static(OUT_DIM as u32)]);
    let w1 = leaf(&mut program, "w1", vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let b1 = leaf(&mut program, "b1", vec![Extent::Static(HIDDEN_DIM as u32)]);
    let w2 = leaf(&mut program, "w2", vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let b2 = leaf(&mut program, "b2", vec![Extent::Static(OUT_DIM as u32)]);

    let h_pre = batched_dense(&mut program, x, w1, b1);
    let h = relu(&mut program, DType::Float32, h_pre, 2);
    let logits = batched_dense(&mut program, h, w2, b2);
    let summed_loss = softmax_cross_entropy(&mut program, DType::Float32, logits, y, 2, 1);

    let inverse_batch = op::append(&mut program, Op::Constant { dtype: DType::Float32, shape: Vec::new(), value: 1.0 / BATCH as f32 });
    let loss = elementwise(&mut program, ScalarOp::Multiply, vec![(summed_loss, identity(0)), (inverse_batch, identity(0))]);

    Network { program, w1, b1, w2, b2, loss }
}

fn he_init(seed: u64, count: usize, fan_in: usize) -> Vec<f32> {
    let scale = (2.0f32 / fan_in as f32).sqrt();
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let uniform = ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            uniform * scale
        })
        .collect()
}

fn zeros(count: usize) -> Vec<f32> {
    vec![0.0f32; count]
}

/// Everything the sweep below needs to run `train_step` on the real,
/// differentiated + Adam-composed program: the training program, `loss`,
/// the `rebind` list, and the real MNIST batches to feed it.
struct TrainingLane {
    program: Vec<Op>,
    loss: NodeId,
    rebind: Vec<(NodeId, &'static str)>,
    initial_state: State,
    batches: Vec<Vec<(&'static str, Vec<f32>)>>,
}

fn build_training_lane() -> TrainingLane {
    let network = build_network();
    // ROW 162: scoped to exactly the params `rebind` below threads forward --
    // `differentiate`'s own `x`/`y` gradients would be computed and then
    // never read, matching ROW 161's `grad_x`=8.58%-of-step finding.
    let wanted = [network.w1, network.b1, network.w2, network.b2];
    let differentiated = differentiate_wanted(&network.program, network.loss, &wanted).expect("scalar loss differentiates");
    let grad_w1 = differentiated.gradient_of_named("w1").expect("w1 feeds the loss");
    let grad_b1 = differentiated.gradient_of_named("b1").expect("b1 feeds the loss");
    let grad_w2 = differentiated.gradient_of_named("w2").expect("w2 feeds the loss");
    let grad_b2 = differentiated.gradient_of_named("b2").expect("b2 feeds the loss");
    let mut program = differentiated.program;
    let config = AdamConfig { learning_rate: 0.001, ..AdamConfig::default() };
    let step_node = step_input(&mut program, "step");

    let m_w1 = leaf(&mut program, "m_w1", vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let v_w1 = leaf(&mut program, "v_w1", vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let m_b1 = leaf(&mut program, "m_b1", vec![Extent::Static(HIDDEN_DIM as u32)]);
    let v_b1 = leaf(&mut program, "v_b1", vec![Extent::Static(HIDDEN_DIM as u32)]);
    let m_w2 = leaf(&mut program, "m_w2", vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let v_w2 = leaf(&mut program, "v_w2", vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let m_b2 = leaf(&mut program, "m_b2", vec![Extent::Static(OUT_DIM as u32)]);
    let v_b2 = leaf(&mut program, "v_b2", vec![Extent::Static(OUT_DIM as u32)]);

    let (new_w1, new_m_w1, new_v_w1) = adam_step(&mut program, &config, 2, AdamOperands { param: network.w1, grad: grad_w1, m: m_w1, v: v_w1 }, step_node);
    let (new_b1, new_m_b1, new_v_b1) = adam_step(&mut program, &config, 1, AdamOperands { param: network.b1, grad: grad_b1, m: m_b1, v: v_b1 }, step_node);
    let (new_w2, new_m_w2, new_v_w2) = adam_step(&mut program, &config, 2, AdamOperands { param: network.w2, grad: grad_w2, m: m_w2, v: v_w2 }, step_node);
    let (new_b2, new_m_b2, new_v_b2) = adam_step(&mut program, &config, 1, AdamOperands { param: network.b2, grad: grad_b2, m: m_b2, v: v_b2 }, step_node);

    let rebind: Vec<(NodeId, &'static str)> = vec![
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

    let initial_state: State = vec![
        ("w1".into(), he_init(0x9E37_79B9, IN_DIM * HIDDEN_DIM, IN_DIM)),
        ("m_w1".into(), zeros(IN_DIM * HIDDEN_DIM)),
        ("v_w1".into(), zeros(IN_DIM * HIDDEN_DIM)),
        ("b1".into(), zeros(HIDDEN_DIM)),
        ("m_b1".into(), zeros(HIDDEN_DIM)),
        ("v_b1".into(), zeros(HIDDEN_DIM)),
        ("w2".into(), he_init(0x8542_D2C3, HIDDEN_DIM * OUT_DIM, HIDDEN_DIM)),
        ("m_w2".into(), zeros(HIDDEN_DIM * OUT_DIM)),
        ("v_w2".into(), zeros(HIDDEN_DIM * OUT_DIM)),
        ("b2".into(), zeros(OUT_DIM)),
        ("m_b2".into(), zeros(OUT_DIM)),
        ("v_b2".into(), zeros(OUT_DIM)),
    ];

    let train_images = load_normalized_images(&train_images_path(), TRAIN_EXAMPLES);
    let train_one_hot = load_one_hot_labels(&train_labels_path(), TRAIN_EXAMPLES);
    let example_count = TRAIN_EXAMPLES - (TRAIN_EXAMPLES % BATCH);
    let batch_count = example_count / BATCH;
    assert!(batch_count >= WARMUP_STEPS + MEASURED_STEPS, "need {} distinct real batches, got {batch_count}", WARMUP_STEPS + MEASURED_STEPS);

    let batches: Vec<Vec<(&'static str, Vec<f32>)>> = (0..batch_count)
        .map(|batch_index| {
            let image_start = batch_index * BATCH * IN_DIM;
            let label_start = batch_index * BATCH * OUT_DIM;
            let step_value = vec![(batch_index + 1) as f32];
            vec![
                ("x", train_images[image_start..image_start + BATCH * IN_DIM].to_vec()),
                ("y", train_one_hot[label_start..label_start + BATCH * OUT_DIM].to_vec()),
                ("step", step_value),
            ]
        })
        .collect();

    TrainingLane { program, loss: network.loss, rebind, initial_state, batches }
}

fn named_for_step<'a>(batch: &'a [(&'static str, Vec<f32>)], state: &'a State) -> Vec<(&'a str, &'a [f32])> {
    batch
        .iter()
        .map(|(name, values)| (*name, values.as_slice()))
        .chain(state.iter().map(|(name, values)| (name.as_str(), values.as_slice())))
        .collect()
}

fn percentile(sorted_ns: &[u64], fraction: f64) -> u64 {
    let index = ((sorted_ns.len() as f64 - 1.0) * fraction).round() as usize;
    sorted_ns[index]
}

struct SweepResult {
    per_step_ns: Vec<u64>,
    loss_curve: Vec<f32>,
    final_state: State,
}

/// One real training run, `WARMUP_STEPS` unmeasured then `MEASURED_STEPS`
/// timed, over the given `lane`'s real batches, calling [`train_step`] fresh
/// every step -- the sealed ROW 159 baseline arm, `evaluate_named`'s own
/// `free_buffers: Vec::new()` per call, never threaded across steps.
fn sweep_baseline(lane: &TrainingLane) -> SweepResult {
    let mut state = lane.initial_state.clone();
    for batch in &lane.batches[..WARMUP_STEPS] {
        let named = named_for_step(batch, &state);
        let (_loss, next_state) = train_step(&lane.program, lane.loss, &lane.rebind, &named).expect("warm-up train_step evaluates");
        state = next_state;
    }

    let mut per_step_ns: Vec<u64> = Vec::with_capacity(MEASURED_STEPS);
    let mut loss_curve: Vec<f32> = Vec::with_capacity(MEASURED_STEPS);
    for batch in &lane.batches[WARMUP_STEPS..WARMUP_STEPS + MEASURED_STEPS] {
        let named = named_for_step(batch, &state);
        let start = Instant::now();
        let (loss_value, next_state) = train_step(&lane.program, lane.loss, &lane.rebind, &named).expect("measured train_step evaluates");
        per_step_ns.push(start.elapsed().as_nanos() as u64);
        loss_curve.push(loss_value);
        state = next_state;
    }
    SweepResult { per_step_ns, loss_curve, final_state: state }
}

/// Same real training run as [`sweep_baseline`], same batches, same initial
/// state, but every step (warm-up included, matching [`fit`](proxima_autograd::train::fit)'s
/// own threading) calls [`train_step_scratch`] with ONE `free_buffers`
/// pool and ONE `validated_weight_nodes` cache carried across the whole run
/// -- the ROW 159 attacked-and-ROLLED-BACK lever arm (see this file's own
/// top-level doc).
fn sweep_scratch(lane: &TrainingLane) -> SweepResult {
    let mut state = lane.initial_state.clone();
    let mut free_buffers: Vec<Vec<f32>> = Vec::new();
    let mut validated_weight_nodes: Option<BTreeSet<NodeId>> = None;
    for batch in &lane.batches[..WARMUP_STEPS] {
        let named = named_for_step(batch, &state);
        let (_loss, next_state) = train_step_scratch(&lane.program, lane.loss, &lane.rebind, &named, &mut free_buffers, &mut validated_weight_nodes);
        state = next_state;
    }

    let mut per_step_ns: Vec<u64> = Vec::with_capacity(MEASURED_STEPS);
    let mut loss_curve: Vec<f32> = Vec::with_capacity(MEASURED_STEPS);
    for batch in &lane.batches[WARMUP_STEPS..WARMUP_STEPS + MEASURED_STEPS] {
        let named = named_for_step(batch, &state);
        let start = Instant::now();
        let (loss_value, next_state) = train_step_scratch(&lane.program, lane.loss, &lane.rebind, &named, &mut free_buffers, &mut validated_weight_nodes);
        per_step_ns.push(start.elapsed().as_nanos() as u64);
        loss_curve.push(loss_value);
        state = next_state;
    }
    SweepResult { per_step_ns, loss_curve, final_state: state }
}

/// Same real training run as [`sweep_baseline`], same batches, same initial
/// state, but the graph is bound and every node's output buffer sized ONCE
/// (`build_static_arena`, before the loop) and reused unchanged in size for
/// every step -- ROW 164's static-arena lever.
fn sweep_arena(lane: &TrainingLane) -> SweepResult {
    let mut outputs = Vec::with_capacity(lane.rebind.len() + 1);
    outputs.push(lane.loss);
    outputs.extend(lane.rebind.iter().map(|(node, _)| *node));
    let mut arena = build_static_arena(&lane.program, &[], &outputs).expect("build_static_arena builds the training lane");

    let mut state = lane.initial_state.clone();
    for batch in &lane.batches[..WARMUP_STEPS] {
        let named = named_for_step(batch, &state);
        let (_loss, next_state) = train_step_arena(&mut arena, &named, lane.loss, &lane.rebind);
        state = next_state;
    }

    let mut per_step_ns: Vec<u64> = Vec::with_capacity(MEASURED_STEPS);
    let mut loss_curve: Vec<f32> = Vec::with_capacity(MEASURED_STEPS);
    for batch in &lane.batches[WARMUP_STEPS..WARMUP_STEPS + MEASURED_STEPS] {
        let named = named_for_step(batch, &state);
        let start = Instant::now();
        let (loss_value, next_state) = train_step_arena(&mut arena, &named, lane.loss, &lane.rebind);
        per_step_ns.push(start.elapsed().as_nanos() as u64);
        loss_curve.push(loss_value);
        state = next_state;
    }
    SweepResult { per_step_ns, loss_curve, final_state: state }
}

/// Correctness gate for ROW 164: runs [`train_step`] (fresh-alloc baseline)
/// and [`train_step_arena`] (static-arena) side by side, from the SAME
/// initial state, over 3 consecutive real steps, and asserts every one of
/// the 12 `rebind` outputs plus the loss is bit-identical between the two
/// paths at every step -- not just at the end, so a divergence introduced
/// at step 1 and silently corrected by step 3 (or vice versa) cannot hide.
fn assert_arena_bit_identical_to_baseline(lane: &TrainingLane) {
    let mut outputs = Vec::with_capacity(lane.rebind.len() + 1);
    outputs.push(lane.loss);
    outputs.extend(lane.rebind.iter().map(|(node, _)| *node));
    let mut arena = build_static_arena(&lane.program, &[], &outputs).expect("build_static_arena builds the training lane");

    let mut baseline_state = lane.initial_state.clone();
    let mut arena_state = lane.initial_state.clone();
    for (step_index, batch) in lane.batches[..3].iter().enumerate() {
        let baseline_named = named_for_step(batch, &baseline_state);
        let (baseline_loss, next_baseline_state) = train_step(&lane.program, lane.loss, &lane.rebind, &baseline_named).expect("baseline train_step evaluates");

        let arena_named = named_for_step(batch, &arena_state);
        let (arena_loss, next_arena_state) = train_step_arena(&mut arena, &arena_named, lane.loss, &lane.rebind);

        assert_eq!(baseline_loss.to_bits(), arena_loss.to_bits(), "step {step_index}: loss diverged between baseline and arena paths");
        assert_eq!(next_baseline_state.len(), 12, "step {step_index}: expected exactly 12 rebind outputs");
        assert_eq!(next_arena_state.len(), 12, "step {step_index}: expected exactly 12 rebind outputs");
        for ((baseline_name, baseline_values), (arena_name, arena_values)) in next_baseline_state.iter().zip(next_arena_state.iter()) {
            assert_eq!(baseline_name, arena_name, "step {step_index}: rebind name ordering diverged");
            assert_eq!(
                baseline_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                arena_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "step {step_index}: rebind output {baseline_name} diverged between baseline and arena paths"
            );
        }
        baseline_state = next_baseline_state;
        arena_state = next_arena_state;
    }
    eprintln!("train_step_lane: arena vs baseline bit-identical over 3 consecutive real steps (loss + all 12 rebind outputs)");
}

fn report_sweep(label: &str, result: &SweepResult) {
    assert!(
        result.loss_curve.iter().all(|value| value.is_finite()),
        "{label}: loss went non-finite during the benched run: {:?}",
        result.loss_curve
    );
    let first_quarter = MEASURED_STEPS / 4;
    let first_avg = result.loss_curve[..first_quarter].iter().sum::<f32>() / first_quarter as f32;
    let last_avg = result.loss_curve[MEASURED_STEPS - first_quarter..].iter().sum::<f32>() / first_quarter as f32;
    eprintln!("train_step_lane[{label}]: convergence gate: first-quarter-avg-loss={first_avg:.4} last-quarter-avg-loss={last_avg:.4}");
    assert!(
        last_avg < first_avg,
        "{label}: expected loss to decrease over the {MEASURED_STEPS}-step benched run (real training step, not a cost-only replay), got {first_avg:.4} -> {last_avg:.4}"
    );

    let mut sorted_ns = result.per_step_ns.clone();
    sorted_ns.sort_unstable();
    let mean_ns = sorted_ns.iter().sum::<u64>() as f64 / sorted_ns.len() as f64;
    let variance = sorted_ns.iter().map(|&value| (value as f64 - mean_ns).powi(2)).sum::<f64>() / sorted_ns.len() as f64;
    let cov = variance.sqrt() / mean_ns * 100.0;
    let p50_ns = percentile(&sorted_ns, 0.50);
    let p95_ns = percentile(&sorted_ns, 0.95);
    eprintln!(
        "train_step_lane[{label}]: manual sweep over {} real training steps: mean={:.4}ms p50={:.4}ms p95={:.4}ms CoV={cov:.2}%",
        sorted_ns.len(),
        mean_ns / 1e6,
        p50_ns as f64 / 1e6,
        p95_ns as f64 / 1e6,
    );
}

fn main() {
    if !dataset_present() {
        eprintln!("train_step_lane: skipping, no host-local MNIST idx dataset under {DATASET_DIR}");
        return;
    }

    let lane = build_training_lane();
    eprintln!(
        "train_step_lane: MLP {IN_DIM}-{HIDDEN_DIM}-{OUT_DIM}, batch={BATCH}, warmup={WARMUP_STEPS}, measured={MEASURED_STEPS}, program_len={}",
        lane.program.len()
    );

    assert_arena_bit_identical_to_baseline(&lane);

    let baseline = sweep_baseline(&lane);
    report_sweep("baseline train_step (fresh alloc every call)", &baseline);

    let scratch = sweep_scratch(&lane);
    report_sweep("train_step_with_scratch (pool threaded across the run)", &scratch);

    let arena = sweep_arena(&lane);
    report_sweep("train_step_arena (static arena, bind+size once)", &arena);

    // criterion groups: repeated calls against each arm's own post-warm-up
    // state, fixed (not threaded forward per-iteration -- criterion's own
    // iteration count is not under this file's control, so the convergence
    // gates above, on the manual sweeps, are the load-bearing proof both are
    // real steps; these groups exist for criterion's own outlier/CI
    // reporting on top of the manual p50/p95/CoV already computed above).
    let fixed_named_batch = lane.batches[WARMUP_STEPS].clone();
    let mut criterion = Criterion::default();
    let mut group = criterion.benchmark_group("train_step_lane");
    group.sample_size(30);

    let baseline_state = baseline.final_state.clone();
    group.bench_function("train_step_mlp_784_128_10_batch32_adam_baseline", |bencher| {
        bencher.iter(|| {
            let named = named_for_step(&fixed_named_batch, &baseline_state);
            train_step(&lane.program, lane.loss, &lane.rebind, &named).expect("train_step evaluates")
        });
    });

    let scratch_state = scratch.final_state.clone();
    let scratch_pool: RefCell<ScratchPool> = RefCell::new((Vec::new(), None));
    group.bench_function("train_step_mlp_784_128_10_batch32_adam_scratch", |bencher| {
        bencher.iter(|| {
            let named = named_for_step(&fixed_named_batch, &scratch_state);
            let mut pool = scratch_pool.borrow_mut();
            let (free_buffers, validated_weight_nodes) = &mut *pool;
            train_step_scratch(&lane.program, lane.loss, &lane.rebind, &named, free_buffers, validated_weight_nodes)
        });
    });
    group.finish();
    criterion.final_summary();
}
