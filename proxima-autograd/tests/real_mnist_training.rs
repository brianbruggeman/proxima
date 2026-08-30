//! Trains a `784 -> 128 -> 10` MLP (relu hidden,
//! [`proxima_autograd::loss::softmax_cross_entropy`]) through
//! [`proxima_autograd::train::fit`] and [`proxima_autograd::optimizer::adam_step`]
//! on real MNIST training images, then evaluates it on real held-out test
//! images -- the same `fit`/`optimizer`/`activation`/`loss` composition
//! `tests/train_fit.rs` exercises on a synthetic 3-4-2 toy network, now
//! carrying a batch axis (`tests/train_fit.rs`'s own `dense` widened by one
//! leading axis, addressed the same way `proxima-autograd/src/expr.rs`'s
//! `identity`/`projection` idiom already addresses every other axis in this
//! crate) and real pixels instead of a synthetic pattern.
//!
//! Real data: the same `~/.cache/burn-dataset/mnist` idx files
//! `proxima-onnx/tests/real_mnist_accuracy.rs` reads, normalized the same
//! way (`(pixel / 255 - 0.1307) / 0.3081`, confirmed against
//! `~/repos/others/burn/examples/onnx-inference`'s own inference example).
//! `#[cfg(test)]`-runtime presence-guarded, not `#[ignore]`d -- the
//! `checkpoint_present()` convention `tests/language_model.rs` uses for its
//! own host-local tokenizer fixture, so this test runs wherever the data
//! exists instead of requiring `--ignored`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use proxima_autograd::activation::relu;
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::loss::softmax_cross_entropy;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_autograd::train::fit;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const IN_DIM: usize = 28 * 28;
const HIDDEN_DIM: usize = 128;
const OUT_DIM: usize = 10;
const BATCH: usize = 32;
const TRAIN_EXAMPLES: usize = 8000;
const TEST_EXAMPLES: usize = 1000;
const EPOCHS: u32 = 4;

fn checkpoint_present() -> bool {
    train_images_path().exists() && train_labels_path().exists() && test_images_path().exists() && test_labels_path().exists()
}

fn train_images_path() -> std::path::PathBuf {
    std::path::Path::new(DATASET_DIR).join("train/train-images-idx3-ubyte")
}
fn train_labels_path() -> std::path::PathBuf {
    std::path::Path::new(DATASET_DIR).join("train/train-labels-idx1-ubyte")
}
fn test_images_path() -> std::path::PathBuf {
    std::path::Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}
fn test_labels_path() -> std::path::PathBuf {
    std::path::Path::new(DATASET_DIR).join("test/t10k-labels-idx1-ubyte")
}

/// Same idx3/idx1 big-endian header `proxima-onnx/tests/real_mnist_accuracy.rs`
/// parses, restated here rather than shared across the crate boundary --
/// this crate's own DE-CISC posture keeps it a plain inline fn, not a
/// dependency.
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

fn load_normalized_images(path: &std::path::Path, limit: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read idx3 image file");
    let (item_count, extents) = idx_header(&bytes);
    let pixel_count = extents.iter().product::<usize>();
    let take = item_count.min(limit);
    let header_length = 4 + extents.len() * 4 + 4;
    bytes[header_length..header_length + take * pixel_count].iter().map(|&pixel| ((pixel as f32 / 255.0) - 0.1307) / 0.3081).collect()
}

fn load_one_hot_labels(path: &std::path::Path, limit: usize) -> (Vec<f32>, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read idx1 label file");
    let (item_count, _extents) = idx_header(&bytes);
    let take = item_count.min(limit);
    let raw = &bytes[8..8 + take];
    let mut one_hot = alloc::vec![0.0f32; take * OUT_DIM];
    for (index, &label) in raw.iter().enumerate() {
        one_hot[index * OUT_DIM + label as usize] = 1.0;
    }
    (one_hot, raw.to_vec())
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

/// `tests/train_fit.rs`'s own `dense` (`x @ w + b`, no batch axis) widened
/// by one leading batch axis: the 3-axis iteration space is
/// `[batch, in, out]`; `w` addresses axes `[in, out]` (broadcasting over
/// `batch`), `x` addresses axes `[batch, in]` (broadcasting over `out`),
/// the product reduces out the `in` axis (axis 1) keeping `[batch, out]`,
/// and `b` (shape `[out]`) broadcasts over `batch` onto that result --
/// the exact axis-selection idiom `expr::identity`/`expr::broadcast`
/// already generalizes, spelled out here since this integration test
/// cannot reach that `pub(crate)` module.
fn batched_dense(program: &mut Vec<Op>, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let product = elementwise(program, ScalarOp::Multiply, alloc::vec![(w, axes(3, &[1, 2])), (x, axes(3, &[0, 1]))]);
    let matmul = reduce_add(program, product, identity(3), axes(3, &[0, 2]));
    elementwise(program, ScalarOp::Add, alloc::vec![(matmul, identity(2)), (b, axes(2, &[1]))])
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
    let x = leaf(&mut program, "x", alloc::vec![Extent::Static(BATCH as u32), Extent::Static(IN_DIM as u32)]);
    let y = leaf(&mut program, "y", alloc::vec![Extent::Static(BATCH as u32), Extent::Static(OUT_DIM as u32)]);
    let w1 = leaf(&mut program, "w1", alloc::vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let b1 = leaf(&mut program, "b1", alloc::vec![Extent::Static(HIDDEN_DIM as u32)]);
    let w2 = leaf(&mut program, "w2", alloc::vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let b2 = leaf(&mut program, "b2", alloc::vec![Extent::Static(OUT_DIM as u32)]);

    let h_pre = batched_dense(&mut program, x, w1, b1);
    let h = relu(&mut program, DType::Float32, h_pre, 2);
    let logits = batched_dense(&mut program, h, w2, b2);
    let summed_loss = softmax_cross_entropy(&mut program, DType::Float32, logits, y, 2, 1);

    let inverse_batch = op::append(&mut program, Op::Constant { dtype: DType::Float32, shape: Vec::new(), value: 1.0 / BATCH as f32 });
    let loss = elementwise(&mut program, ScalarOp::Multiply, alloc::vec![(summed_loss, identity(0)), (inverse_batch, identity(0))]);

    Network { program, w1, b1, w2, b2, loss }
}

/// He-scaled pseudo-random init from a splitmix-style counter (no external
/// RNG dependency, deterministic across runs) -- zero init would leave
/// every hidden unit identical (symmetric gradients never break the tie).
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
    alloc::vec![0.0f32; count]
}

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|left, right| left.1.total_cmp(right.1)).map(|(index, _)| index).expect("nonempty logits")
}

/// Trains [`build_network`] on real MNIST digits through
/// [`proxima_autograd::train::fit`] and evaluates the trained weights on
/// real held-out test digits, asserting top-1 accuracy -- the training
/// counterpart to `proxima-onnx/tests/real_mnist_accuracy.rs`'s inference
/// proof, now proving the training stack (`fit`/`loss`/`optimizer`) itself
/// against a real dataset rather than a synthetic pattern.
#[test]
fn real_mnist_mlp_trains_and_classifies_at_reference_accuracy() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local MNIST idx dataset under {DATASET_DIR}");
        return;
    }

    let network = build_network();
    let differentiated = differentiate(&network.program, network.loss).expect("scalar loss differentiates");
    let grad_w1 = differentiated.gradient_of_named("w1").expect("w1 feeds the loss");
    let grad_b1 = differentiated.gradient_of_named("b1").expect("b1 feeds the loss");
    let grad_w2 = differentiated.gradient_of_named("w2").expect("w2 feeds the loss");
    let grad_b2 = differentiated.gradient_of_named("b2").expect("b2 feeds the loss");
    let mut program = differentiated.program;
    let config = AdamConfig { learning_rate: 0.001, ..AdamConfig::default() };
    let step_node = step_input(&mut program, "step");

    let m_w1 = leaf(&mut program, "m_w1", alloc::vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let v_w1 = leaf(&mut program, "v_w1", alloc::vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let m_b1 = leaf(&mut program, "m_b1", alloc::vec![Extent::Static(HIDDEN_DIM as u32)]);
    let v_b1 = leaf(&mut program, "v_b1", alloc::vec![Extent::Static(HIDDEN_DIM as u32)]);
    let m_w2 = leaf(&mut program, "m_w2", alloc::vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let v_w2 = leaf(&mut program, "v_w2", alloc::vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let m_b2 = leaf(&mut program, "m_b2", alloc::vec![Extent::Static(OUT_DIM as u32)]);
    let v_b2 = leaf(&mut program, "v_b2", alloc::vec![Extent::Static(OUT_DIM as u32)]);

    let (new_w1, new_m_w1, new_v_w1) = adam_step(&mut program, &config, 2, AdamOperands { param: network.w1, grad: grad_w1, m: m_w1, v: v_w1 }, step_node);
    let (new_b1, new_m_b1, new_v_b1) = adam_step(&mut program, &config, 1, AdamOperands { param: network.b1, grad: grad_b1, m: m_b1, v: v_b1 }, step_node);
    let (new_w2, new_m_w2, new_v_w2) = adam_step(&mut program, &config, 2, AdamOperands { param: network.w2, grad: grad_w2, m: m_w2, v: v_w2 }, step_node);
    let (new_b2, new_m_b2, new_v_b2) = adam_step(&mut program, &config, 1, AdamOperands { param: network.b2, grad: grad_b2, m: m_b2, v: v_b2 }, step_node);

    let rebind: Vec<(NodeId, &str)> = alloc::vec![
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

    let initial_state: Vec<(String, Vec<f32>)> = alloc::vec![
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
    let (train_one_hot, _train_labels) = load_one_hot_labels(&train_labels_path(), TRAIN_EXAMPLES);
    let example_count = TRAIN_EXAMPLES - (TRAIN_EXAMPLES % BATCH);
    let batch_count = example_count / BATCH;

    let steps: Vec<[f32; 1]> = (1..=batch_count as u32).map(|value| [value as f32]).collect();
    let batches: Vec<Vec<(&str, &[f32])>> = (0..batch_count)
        .map(|batch_index| {
            let image_start = batch_index * BATCH * IN_DIM;
            let label_start = batch_index * BATCH * OUT_DIM;
            alloc::vec![
                ("x", &train_images[image_start..image_start + BATCH * IN_DIM]),
                ("y", &train_one_hot[label_start..label_start + BATCH * OUT_DIM]),
                ("step", steps[batch_index].as_slice()),
            ]
        })
        .collect();

    std::eprintln!("real_mnist_training: {batch_count} batches/epoch x {EPOCHS} epochs, batch={BATCH}, train_examples={example_count}");
    let start = std::time::Instant::now();
    let (final_state, loss_curve) = fit(&program, network.loss, &rebind, initial_state, EPOCHS, &batches).expect("fit runs to completion on real mnist data");
    let elapsed = start.elapsed();

    let first_epoch_average = loss_curve[..batch_count].iter().sum::<f32>() / batch_count as f32;
    let last_epoch_average = loss_curve[loss_curve.len() - batch_count..].iter().sum::<f32>() / batch_count as f32;
    std::eprintln!(
        "real_mnist_training loss curve: first-epoch-avg={first_epoch_average:.4} last-epoch-avg={last_epoch_average:.4} wall_clock={elapsed:?}"
    );
    assert!(loss_curve.iter().all(|value| value.is_finite()), "loss went non-finite: first 10 = {:?}", &loss_curve[..10.min(loss_curve.len())]);
    assert!(
        last_epoch_average < first_epoch_average * 0.5,
        "expected training loss to more than halve over {EPOCHS} epochs, got {first_epoch_average:.4} -> {last_epoch_average:.4}"
    );

    let final_w1 = &final_state.iter().find(|(name, _)| name == "w1").expect("trained w1 present").1;
    let final_b1 = &final_state.iter().find(|(name, _)| name == "b1").expect("trained b1 present").1;
    let final_w2 = &final_state.iter().find(|(name, _)| name == "w2").expect("trained w2 present").1;
    let final_b2 = &final_state.iter().find(|(name, _)| name == "b2").expect("trained b2 present").1;

    let test_images = load_normalized_images(&test_images_path(), TEST_EXAMPLES);
    let (_test_one_hot, test_labels) = load_one_hot_labels(&test_labels_path(), TEST_EXAMPLES);
    let test_count = test_labels.len();

    // A fresh forward-only program, not `network.program.clone()`: that
    // program's own `x`/`y` leaves are fixed at `Extent::Static(BATCH)`,
    // and `evaluate_named` validates every declared `Op::Input` is bound
    // regardless of whether the requested output actually depends on it
    // (`resolve_named_blocks`, `proxima-tensor/src/cpu.rs:1025-1043`) --
    // so a training-sized `x` left dangling in a cloned program fails to
    // evaluate even when the eval subgraph never reads it.
    let mut eval_program = Vec::new();
    let eval_x = leaf(&mut eval_program, "x", alloc::vec![Extent::Static(test_count as u32), Extent::Static(IN_DIM as u32)]);
    let eval_w1 = leaf(&mut eval_program, "w1", alloc::vec![Extent::Static(IN_DIM as u32), Extent::Static(HIDDEN_DIM as u32)]);
    let eval_b1 = leaf(&mut eval_program, "b1", alloc::vec![Extent::Static(HIDDEN_DIM as u32)]);
    let eval_w2 = leaf(&mut eval_program, "w2", alloc::vec![Extent::Static(HIDDEN_DIM as u32), Extent::Static(OUT_DIM as u32)]);
    let eval_b2 = leaf(&mut eval_program, "b2", alloc::vec![Extent::Static(OUT_DIM as u32)]);
    let eval_h_pre = batched_dense(&mut eval_program, eval_x, eval_w1, eval_b1);
    let eval_h = relu(&mut eval_program, DType::Float32, eval_h_pre, 2);
    let eval_logits = batched_dense(&mut eval_program, eval_h, eval_w2, eval_b2);

    let eval_named: Vec<(&str, &[f32])> = alloc::vec![
        ("x", test_images.as_slice()),
        ("w1", final_w1.as_slice()),
        ("b1", final_b1.as_slice()),
        ("w2", final_w2.as_slice()),
        ("b2", final_b2.as_slice()),
    ];
    let evaluated = proxima_tensor::cpu::evaluate_named(&eval_program, &[], &eval_named, &[eval_logits]).expect("evaluate the trained mlp on real held-out mnist test images");
    let (logits, shape) = evaluated.get(eval_logits).expect("eval logits present");
    assert_eq!(shape, &alloc::vec![test_count as u64, OUT_DIM as u64], "one 10-way logit row per test image");

    let mut correct = 0_usize;
    for (index, &label) in test_labels.iter().enumerate() {
        let row = &logits[index * OUT_DIM..(index + 1) * OUT_DIM];
        if argmax(row) == label as usize {
            correct += 1;
        }
    }
    let accuracy = correct as f64 / test_count as f64;
    std::eprintln!("real_mnist_training test accuracy: {accuracy:.4} ({correct}/{test_count} images), total wall_clock={:?}", start.elapsed());

    assert!(accuracy >= 0.90, "expected the trained mlp to classify at least 90% of {test_count} real held-out mnist test images, got {accuracy:.4}");
}
