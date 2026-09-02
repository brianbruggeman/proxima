//! Closes train -> checkpoint -> serve on real data: trains the exact
//! `784-128-10` MLP `tests/real_mnist_training.rs` trains, [`save_state`]s
//! the trained weights to a [`tempfile`] checkpoint, evaluates test
//! accuracy from the in-memory `final_state` (pre-save), then -- in a
//! FRESH `eval_program`/`state` binding, never touching `final_state` again
//! -- [`load_state`]s that same checkpoint and re-evaluates accuracy,
//! asserting the two numbers are bit-identical. A second, independent
//! check parses the written file with [`proxima_safetensors::SafetensorsParser`]
//! directly (not through [`load_state`]) and asserts every name/shape/dtype
//! round-tripped.
//!
//! Real data, same convention as `tests/real_mnist_training.rs`:
//! `#[cfg(test)]`-runtime presence-guarded on the host-local
//! `~/.cache/burn-dataset/mnist` idx files, not `#[ignore]`d.

#![cfg(feature = "safetensors")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use proxima_autograd::activation::relu;
use proxima_autograd::adjoint::differentiate;
use proxima_autograd::loss::softmax_cross_entropy;
use proxima_autograd::optimizer::{AdamConfig, AdamOperands, adam_step, step_input};
use proxima_autograd::persist::{load_state, save_state};
use proxima_autograd::train::fit;
use proxima_tensor::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const IN_DIM: usize = 28 * 28;
const HIDDEN_DIM: usize = 128;
const OUT_DIM: usize = 10;
const BATCH: usize = 32;
const TRAIN_EXAMPLES: usize = 2000;
const TEST_EXAMPLES: usize = 500;
const EPOCHS: u32 = 2;

fn checkpoint_present() -> bool {
    train_images_path().exists()
        && train_labels_path().exists()
        && test_images_path().exists()
        && test_labels_path().exists()
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

/// Same idx3/idx1 big-endian header `tests/real_mnist_training.rs` and
/// `proxima-onnx/tests/real_mnist_accuracy.rs` each restate inline -- this
/// crate's own DE-CISC posture keeps it a plain inline fn, not a shared
/// dependency.
fn idx_header(bytes: &[u8]) -> (usize, Vec<usize>) {
    let dimension_count = bytes[3] as usize;
    let item_count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut extents = Vec::with_capacity(dimension_count - 1);
    for axis in 1..dimension_count {
        let offset = 4 + axis * 4;
        extents.push(u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize);
    }
    (item_count, extents)
}

fn load_normalized_images(path: &std::path::Path, limit: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read idx3 image file");
    let (item_count, extents) = idx_header(&bytes);
    let pixel_count = extents.iter().product::<usize>();
    let take = item_count.min(limit);
    let header_length = 4 + extents.len() * 4 + 4;
    bytes[header_length..header_length + take * pixel_count]
        .iter()
        .map(|&pixel| ((pixel as f32 / 255.0) - 0.1307) / 0.3081)
        .collect()
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
    op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape,
            name: Some(name.into()),
        },
    )
}

fn identity(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

fn axes(rank: u16, selected: &[u16]) -> IndexMap {
    IndexMap::Affine(map::projection(rank, selected))
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

fn reduce_add(
    program: &mut Vec<Op>,
    operand: NodeId,
    in_map: IndexMap,
    out_map: IndexMap,
) -> NodeId {
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

/// `tests/real_mnist_training.rs`'s own `batched_dense`: `[batch, in, out]`
/// iteration space, `w` broadcasting over `batch`, `x` broadcasting over
/// `out`, reduced over `in`, `b` broadcasting the bias onto `[batch, out]`.
fn batched_dense(program: &mut Vec<Op>, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let product = elementwise(
        program,
        ScalarOp::Multiply,
        alloc::vec![(w, axes(3, &[1, 2])), (x, axes(3, &[0, 1]))],
    );
    let matmul = reduce_add(program, product, identity(3), axes(3, &[0, 2]));
    elementwise(
        program,
        ScalarOp::Add,
        alloc::vec![(matmul, identity(2)), (b, axes(2, &[1]))],
    )
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
    let x = leaf(
        &mut program,
        "x",
        alloc::vec![Extent::Static(BATCH as u32), Extent::Static(IN_DIM as u32)],
    );
    let y = leaf(
        &mut program,
        "y",
        alloc::vec![Extent::Static(BATCH as u32), Extent::Static(OUT_DIM as u32)],
    );
    let w1 = leaf(
        &mut program,
        "w1",
        alloc::vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32)
        ],
    );
    let b1 = leaf(
        &mut program,
        "b1",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    let w2 = leaf(
        &mut program,
        "w2",
        alloc::vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32)
        ],
    );
    let b2 = leaf(
        &mut program,
        "b2",
        alloc::vec![Extent::Static(OUT_DIM as u32)],
    );

    let h_pre = batched_dense(&mut program, x, w1, b1);
    let h = relu(&mut program, DType::Float32, h_pre, 2);
    let logits = batched_dense(&mut program, h, w2, b2);
    let summed_loss = softmax_cross_entropy(&mut program, DType::Float32, logits, y, 2, 1);

    let inverse_batch = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 1.0 / BATCH as f32,
        },
    );
    let loss = elementwise(
        &mut program,
        ScalarOp::Multiply,
        alloc::vec![(summed_loss, identity(0)), (inverse_batch, identity(0))],
    );

    Network {
        program,
        w1,
        b1,
        w2,
        b2,
        loss,
    }
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
    alloc::vec![0.0f32; count]
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("nonempty logits")
}

/// Builds the forward-only (no batch-axis grad state) evaluation program
/// `tests/real_mnist_training.rs` itself uses to score a trained network,
/// and returns its accuracy over `test_images`/`test_labels` given
/// `w1`/`b1`/`w2`/`b2` host buffers -- reused for both the pre-save
/// (`final_state`) and post-load (fresh, from `load_state`) evaluation so
/// the two calls are provably the same code path over different buffers.
fn evaluate_accuracy(
    test_images: &[f32],
    test_labels: &[u8],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
) -> f64 {
    let test_count = test_labels.len();
    let mut eval_program = Vec::new();
    let eval_x = leaf(
        &mut eval_program,
        "x",
        alloc::vec![
            Extent::Static(test_count as u32),
            Extent::Static(IN_DIM as u32)
        ],
    );
    let eval_w1 = leaf(
        &mut eval_program,
        "w1",
        alloc::vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32)
        ],
    );
    let eval_b1 = leaf(
        &mut eval_program,
        "b1",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    let eval_w2 = leaf(
        &mut eval_program,
        "w2",
        alloc::vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32)
        ],
    );
    let eval_b2 = leaf(
        &mut eval_program,
        "b2",
        alloc::vec![Extent::Static(OUT_DIM as u32)],
    );
    let eval_h_pre = batched_dense(&mut eval_program, eval_x, eval_w1, eval_b1);
    let eval_h = relu(&mut eval_program, DType::Float32, eval_h_pre, 2);
    let eval_logits = batched_dense(&mut eval_program, eval_h, eval_w2, eval_b2);

    let eval_named: Vec<(&str, &[f32])> = alloc::vec![
        ("x", test_images),
        ("w1", w1),
        ("b1", b1),
        ("w2", w2),
        ("b2", b2)
    ];
    let evaluated =
        proxima_tensor::cpu::evaluate_named(&eval_program, &[], &eval_named, &[eval_logits])
            .expect("evaluate the mlp on real held-out mnist test images");
    let (logits, shape) = evaluated.get(eval_logits).expect("eval logits present");
    assert_eq!(
        shape,
        &alloc::vec![test_count as u64, OUT_DIM as u64],
        "one 10-way logit row per test image"
    );

    let mut correct = 0_usize;
    for (index, &label) in test_labels.iter().enumerate() {
        let row = &logits[index * OUT_DIM..(index + 1) * OUT_DIM];
        if argmax(row) == label as usize {
            correct += 1;
        }
    }
    correct as f64 / test_count as f64
}

/// Trains, checkpoints through [`save_state`], scores the trained weights
/// straight out of `final_state`, then -- in a fresh program/state binding
/// that never sees `final_state` again -- [`load_state`]s the checkpoint
/// back and scores it a second time, asserting the two accuracy numbers
/// are bit-identical (same buffers, same forward graph). A third,
/// independent check parses the written file with
/// [`proxima_safetensors::SafetensorsParser`] directly and asserts every
/// name/shape/dtype survived.
#[test]
fn trained_mlp_survives_a_save_load_round_trip_at_identical_accuracy() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local MNIST idx dataset under {DATASET_DIR}");
        return;
    }

    let network = build_network();
    let differentiated =
        differentiate(&network.program, network.loss).expect("scalar loss differentiates");
    let grad_w1 = differentiated
        .gradient_of_named("w1")
        .expect("w1 feeds the loss");
    let grad_b1 = differentiated
        .gradient_of_named("b1")
        .expect("b1 feeds the loss");
    let grad_w2 = differentiated
        .gradient_of_named("w2")
        .expect("w2 feeds the loss");
    let grad_b2 = differentiated
        .gradient_of_named("b2")
        .expect("b2 feeds the loss");
    let mut program = differentiated.program;
    let config = AdamConfig {
        learning_rate: 0.001,
        ..AdamConfig::default()
    };
    let step_node = step_input(&mut program, "step");

    let m_w1 = leaf(
        &mut program,
        "m_w1",
        alloc::vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32)
        ],
    );
    let v_w1 = leaf(
        &mut program,
        "v_w1",
        alloc::vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32)
        ],
    );
    let m_b1 = leaf(
        &mut program,
        "m_b1",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    let v_b1 = leaf(
        &mut program,
        "v_b1",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    let m_w2 = leaf(
        &mut program,
        "m_w2",
        alloc::vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32)
        ],
    );
    let v_w2 = leaf(
        &mut program,
        "v_w2",
        alloc::vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32)
        ],
    );
    let m_b2 = leaf(
        &mut program,
        "m_b2",
        alloc::vec![Extent::Static(OUT_DIM as u32)],
    );
    let v_b2 = leaf(
        &mut program,
        "v_b2",
        alloc::vec![Extent::Static(OUT_DIM as u32)],
    );

    let (new_w1, new_m_w1, new_v_w1) = adam_step(
        &mut program,
        &config,
        2,
        AdamOperands {
            param: network.w1,
            grad: grad_w1,
            m: m_w1,
            v: v_w1,
        },
        step_node,
    );
    let (new_b1, new_m_b1, new_v_b1) = adam_step(
        &mut program,
        &config,
        1,
        AdamOperands {
            param: network.b1,
            grad: grad_b1,
            m: m_b1,
            v: v_b1,
        },
        step_node,
    );
    let (new_w2, new_m_w2, new_v_w2) = adam_step(
        &mut program,
        &config,
        2,
        AdamOperands {
            param: network.w2,
            grad: grad_w2,
            m: m_w2,
            v: v_w2,
        },
        step_node,
    );
    let (new_b2, new_m_b2, new_v_b2) = adam_step(
        &mut program,
        &config,
        1,
        AdamOperands {
            param: network.b2,
            grad: grad_b2,
            m: m_b2,
            v: v_b2,
        },
        step_node,
    );

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
        (
            "w1".into(),
            he_init(0x9E37_79B9, IN_DIM * HIDDEN_DIM, IN_DIM)
        ),
        ("m_w1".into(), zeros(IN_DIM * HIDDEN_DIM)),
        ("v_w1".into(), zeros(IN_DIM * HIDDEN_DIM)),
        ("b1".into(), zeros(HIDDEN_DIM)),
        ("m_b1".into(), zeros(HIDDEN_DIM)),
        ("v_b1".into(), zeros(HIDDEN_DIM)),
        (
            "w2".into(),
            he_init(0x8542_D2C3, HIDDEN_DIM * OUT_DIM, HIDDEN_DIM)
        ),
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

    let steps: Vec<[f32; 1]> = (1..=batch_count as u32)
        .map(|value| [value as f32])
        .collect();
    let batches: Vec<Vec<(&str, &[f32])>> = (0..batch_count)
        .map(|batch_index| {
            let image_start = batch_index * BATCH * IN_DIM;
            let label_start = batch_index * BATCH * OUT_DIM;
            alloc::vec![
                (
                    "x",
                    &train_images[image_start..image_start + BATCH * IN_DIM]
                ),
                (
                    "y",
                    &train_one_hot[label_start..label_start + BATCH * OUT_DIM]
                ),
                ("step", steps[batch_index].as_slice()),
            ]
        })
        .collect();

    let (final_state, loss_curve) = fit(
        &program,
        network.loss,
        &rebind,
        initial_state,
        EPOCHS,
        &batches,
    )
    .expect("fit runs to completion on real mnist data");
    assert!(
        loss_curve.iter().all(|value| value.is_finite()),
        "loss went non-finite: first 10 = {:?}",
        &loss_curve[..10.min(loss_curve.len())]
    );

    let checkpoint = tempfile::NamedTempFile::new().expect("create temp checkpoint file");
    save_state(&program, &final_state, checkpoint.path())
        .expect("save_state writes the trained checkpoint");

    let final_w1 = &final_state
        .iter()
        .find(|(name, _)| name == "w1")
        .expect("trained w1 present")
        .1;
    let final_b1 = &final_state
        .iter()
        .find(|(name, _)| name == "b1")
        .expect("trained b1 present")
        .1;
    let final_w2 = &final_state
        .iter()
        .find(|(name, _)| name == "w2")
        .expect("trained w2 present")
        .1;
    let final_b2 = &final_state
        .iter()
        .find(|(name, _)| name == "b2")
        .expect("trained b2 present")
        .1;

    let test_images = load_normalized_images(&test_images_path(), TEST_EXAMPLES);
    let (_test_one_hot, test_labels) = load_one_hot_labels(&test_labels_path(), TEST_EXAMPLES);

    let pre_save_accuracy = evaluate_accuracy(
        &test_images,
        &test_labels,
        final_w1,
        final_b1,
        final_w2,
        final_b2,
    );
    std::eprintln!("pre-save accuracy: {pre_save_accuracy:.6}");
    assert!(
        pre_save_accuracy > 0.0,
        "sanity: the trained network must classify at least one test digit correctly"
    );

    // A fresh program, built from scratch in this "new session" -- the
    // same declared shapes `program` carries for its trained parameters,
    // never `program.clone()`, so `load_state` is proven against a
    // genuinely independent `&[Op]`, not the exact object it was saved
    // from.
    let mut load_program = Vec::new();
    leaf(
        &mut load_program,
        "w1",
        alloc::vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32)
        ],
    );
    leaf(
        &mut load_program,
        "b1",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    leaf(
        &mut load_program,
        "w2",
        alloc::vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32)
        ],
    );
    leaf(
        &mut load_program,
        "b2",
        alloc::vec![Extent::Static(OUT_DIM as u32)],
    );
    leaf(
        &mut load_program,
        "m_w1",
        alloc::vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32)
        ],
    );
    leaf(
        &mut load_program,
        "v_w1",
        alloc::vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32)
        ],
    );
    leaf(
        &mut load_program,
        "m_b1",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    leaf(
        &mut load_program,
        "v_b1",
        alloc::vec![Extent::Static(HIDDEN_DIM as u32)],
    );
    leaf(
        &mut load_program,
        "m_w2",
        alloc::vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32)
        ],
    );
    leaf(
        &mut load_program,
        "v_w2",
        alloc::vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32)
        ],
    );
    leaf(
        &mut load_program,
        "m_b2",
        alloc::vec![Extent::Static(OUT_DIM as u32)],
    );
    leaf(
        &mut load_program,
        "v_b2",
        alloc::vec![Extent::Static(OUT_DIM as u32)],
    );

    let loaded_state =
        load_state(&load_program, checkpoint.path()).expect("load_state reads the checkpoint back");
    assert_eq!(
        loaded_state.len(),
        final_state.len(),
        "every rebind name must round-trip through the checkpoint"
    );

    let loaded_w1 = &loaded_state
        .iter()
        .find(|(name, _)| name == "w1")
        .expect("loaded w1 present")
        .1;
    let loaded_b1 = &loaded_state
        .iter()
        .find(|(name, _)| name == "b1")
        .expect("loaded b1 present")
        .1;
    let loaded_w2 = &loaded_state
        .iter()
        .find(|(name, _)| name == "w2")
        .expect("loaded w2 present")
        .1;
    let loaded_b2 = &loaded_state
        .iter()
        .find(|(name, _)| name == "b2")
        .expect("loaded b2 present")
        .1;

    assert_eq!(
        loaded_w1, final_w1,
        "w1 must round-trip bit-identical through save_state/load_state"
    );
    assert_eq!(
        loaded_b1, final_b1,
        "b1 must round-trip bit-identical through save_state/load_state"
    );
    assert_eq!(
        loaded_w2, final_w2,
        "w2 must round-trip bit-identical through save_state/load_state"
    );
    assert_eq!(
        loaded_b2, final_b2,
        "b2 must round-trip bit-identical through save_state/load_state"
    );

    let post_load_accuracy = evaluate_accuracy(
        &test_images,
        &test_labels,
        loaded_w1,
        loaded_b1,
        loaded_w2,
        loaded_b2,
    );
    std::eprintln!("post-load accuracy: {post_load_accuracy:.6}");
    assert_eq!(
        post_load_accuracy, pre_save_accuracy,
        "accuracy must be identical before save and after a fresh-session load"
    );

    // Independent cross-tool check: parse the written file with the
    // safetensors crate's own parser directly (not through `load_state`)
    // and assert names/shapes/dtypes round-tripped.
    let written_bytes = std::fs::read(checkpoint.path()).expect("read the written checkpoint file");
    let manifest = proxima_safetensors::SafetensorsParser::new()
        .push(&written_bytes)
        .expect("parser accepts the written checkpoint")
        .finish()
        .expect("checkpoint parses as a well-formed safetensors manifest");

    for (name, shape) in [
        ("w1", alloc::vec![IN_DIM as u64, HIDDEN_DIM as u64]),
        ("b1", alloc::vec![HIDDEN_DIM as u64]),
        ("w2", alloc::vec![HIDDEN_DIM as u64, OUT_DIM as u64]),
        ("b2", alloc::vec![OUT_DIM as u64]),
    ] {
        let entry = manifest
            .tensor(name)
            .unwrap_or_else(|| panic!("{name} present in the parsed manifest"));
        assert_eq!(
            entry.dtype,
            proxima_tensor::DType::Float32,
            "{name} dtype must round-trip as Float32"
        );
        assert_eq!(entry.shape, shape, "{name} shape must round-trip exactly");
    }
}
