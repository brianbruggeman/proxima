//! One real omega training step, benchmarked under llama.cpp's finetune
//! configuration.
//!
//! The configuration is not benchmark-local: `benches/llama-finetune.toml` is a
//! `conflaguration` `TrainConfig` whose values are the incumbent's defaults at
//! tag `b11671` (`common/common.h:438,442,443,458,459,619,620`,
//! `ggml/src/ggml-opt.cpp:229-231`, `common/common.cpp:2108-2109`), and
//! `omega/tests/training_api.rs::llama_finetune_config_is_the_benchmark_config`
//! pins every one of them. AdamW, lr 1e-5, betas 0.9/0.999, eps 1e-8,
//! weight decay 0, batch 2048 split into ubatch 512: the step measured here is
//! the ubatch llama.cpp actually feeds one optimizer call.
//!
//! The network is the repo's existing batched `784-128-10` training lane. That
//! is the model shape, not the configuration: no llama checkpoint has a
//! differentiable training path here, so inventing a llama-shaped graph would
//! special-case the benchmark in a different direction. The config file is the
//! surface under test.
//!
//! Run with:
//! `cargo bench -p proxima-omega --features train,cpu --bench train_step_llama`
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use omega::train::{
    TrainConfig, adam_first_moment_name, adam_second_moment_name, build_optimizer_step,
};
use proxima_autograd::activation::relu;
use proxima_autograd::loss::softmax_cross_entropy;
use proxima_numeric::DType;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};

const IN_DIM: usize = 28 * 28;
const HIDDEN_DIM: usize = 128;
const OUT_DIM: usize = 10;
const PARAMETERS: [&str; 4] = ["w1", "b1", "w2", "b2"];
const WARMUP_STEPS: usize = 3;

fn config_path() -> String {
    format!("{}/benches/llama-finetune.toml", env!("CARGO_MANIFEST_DIR"))
}

fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape,
            name: Some(name.to_owned()),
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
            dtype: DType::FLOAT32,
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
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
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

fn batched_dense(program: &mut Vec<Op>, x: NodeId, w: NodeId, b: NodeId) -> NodeId {
    let product = elementwise(
        program,
        ScalarOp::Multiply,
        vec![(w, axes(3, &[1, 2])), (x, axes(3, &[0, 1]))],
    );
    let matmul = reduce_add(program, product, identity(3), axes(3, &[0, 2]));
    elementwise(
        program,
        ScalarOp::Add,
        vec![(matmul, identity(2)), (b, axes(2, &[1]))],
    )
}

fn build_forward(batch: usize) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let x = leaf(
        &mut program,
        "x",
        vec![Extent::Static(batch as u32), Extent::Static(IN_DIM as u32)],
    );
    let y = leaf(
        &mut program,
        "y",
        vec![Extent::Static(batch as u32), Extent::Static(OUT_DIM as u32)],
    );
    let w1 = leaf(
        &mut program,
        "w1",
        vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(HIDDEN_DIM as u32),
        ],
    );
    let b1 = leaf(&mut program, "b1", vec![Extent::Static(HIDDEN_DIM as u32)]);
    let w2 = leaf(
        &mut program,
        "w2",
        vec![
            Extent::Static(HIDDEN_DIM as u32),
            Extent::Static(OUT_DIM as u32),
        ],
    );
    let b2 = leaf(&mut program, "b2", vec![Extent::Static(OUT_DIM as u32)]);

    let h_pre = batched_dense(&mut program, x, w1, b1);
    let hidden = relu(&mut program, DType::FLOAT32, h_pre, 2);
    let logits = batched_dense(&mut program, hidden, w2, b2);
    let loss = softmax_cross_entropy(&mut program, DType::FLOAT32, logits, y, 2, 1);
    (program, loss)
}

fn counter_pattern(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn one_hot_labels(batch: usize) -> Vec<f32> {
    let mut labels = vec![0.0f32; batch * OUT_DIM];
    for row in 0..batch {
        labels[row * OUT_DIM + (row * 7 % OUT_DIM)] = 1.0;
    }
    labels
}

struct Lane {
    step: omega::train::TrainingStep,
    x: Vec<f32>,
    y: Vec<f32>,
}

fn build_lane(config: &TrainConfig) -> Lane {
    let batch = config.schedule.ubatch as usize;
    let (program, loss) = build_forward(batch);
    let built = build_optimizer_step(&program, loss, &PARAMETERS, config)
        .expect("the batched MLP differentiates");

    let mut named: Vec<(String, Vec<f32>)> = vec![
        (String::from("x"), counter_pattern(0x1234, batch * IN_DIM)),
        (String::from("y"), one_hot_labels(batch)),
        (String::from("step"), vec![1.0f32]),
        (
            String::from("w1"),
            counter_pattern(0x9E37, IN_DIM * HIDDEN_DIM)
                .into_iter()
                .map(|value| value * (2.0f32 / IN_DIM as f32).sqrt())
                .collect(),
        ),
        (String::from("b1"), vec![0.0f32; HIDDEN_DIM]),
        (
            String::from("w2"),
            counter_pattern(0xBEEF, HIDDEN_DIM * OUT_DIM)
                .into_iter()
                .map(|value| value * (2.0f32 / HIDDEN_DIM as f32).sqrt())
                .collect(),
        ),
        (String::from("b2"), vec![0.0f32; OUT_DIM]),
    ];
    for parameter in PARAMETERS {
        let count = match parameter {
            "w1" => IN_DIM * HIDDEN_DIM,
            "b1" => HIDDEN_DIM,
            "w2" => HIDDEN_DIM * OUT_DIM,
            _ => OUT_DIM,
        };
        named.push((adam_first_moment_name(parameter), vec![0.0f32; count]));
        named.push((adam_second_moment_name(parameter), vec![0.0f32; count]));
    }

    let named_blocks: Vec<(&str, QuantizedBlock<'_>)> = named
        .iter()
        .map(|(name, values)| (name.as_str(), QuantizedBlock::Float32(values.as_slice())))
        .collect();
    let step = built
        .plan(&named_blocks)
        .expect("the configured engine plans the training step");
    let x = named
        .iter()
        .find(|(name, _)| name == "x")
        .map(|(_, values)| values.clone())
        .expect("x is bound");
    let y = named
        .iter()
        .find(|(name, _)| name == "y")
        .map(|(_, values)| values.clone())
        .expect("y is bound");
    Lane { step, x, y }
}

fn run_step(lane: &mut Lane, step_number: u32) -> f32 {
    let step_value = [step_number as f32];
    lane.step
        .step(&[
            ("x", QuantizedBlock::Float32(&lane.x)),
            ("y", QuantizedBlock::Float32(&lane.y)),
            ("step", QuantizedBlock::Float32(&step_value)),
        ])
        .expect("one training step executes")
}

fn bench_training_step(criterion: &mut Criterion) {
    let config = TrainConfig::from_file(config_path()).expect("the llama finetune config loads");
    assert_eq!(
        config.optimizer, "adamw",
        "the benchmark measures llama.cpp's optimizer, not a benchmark-only one"
    );
    println!(
        "train_step_llama: optimizer={} lr={} beta1={} beta2={} eps={} wd={} batch={} ubatch={} model={IN_DIM}-{HIDDEN_DIM}-{OUT_DIM}",
        config.optimizer,
        config.adamw.learning_rate,
        config.adamw.beta1,
        config.adamw.beta2,
        config.adamw.epsilon,
        config.adamw.weight_decay,
        config.schedule.batch,
        config.schedule.ubatch,
    );

    let mut lane = build_lane(&config);
    let initial_w1 = lane.step.state_value("w1").expect("w1 is owned").to_vec();
    let mut losses = Vec::with_capacity(WARMUP_STEPS);
    for index in 1..=WARMUP_STEPS as u32 {
        losses.push(run_step(&mut lane, index));
    }
    assert!(
        losses.iter().all(|loss| loss.is_finite()),
        "the benched step must stay finite, got {losses:?}"
    );
    assert_ne!(
        lane.step.state_value("w1").expect("w1 is owned"),
        initial_w1.as_slice(),
        "the benched step must still update the parameter"
    );

    let mut step_number = WARMUP_STEPS as u32 + 1;
    criterion.bench_function("omega_train_step_llama_adamw", |bencher| {
        bencher.iter(|| {
            black_box(run_step(&mut lane, step_number));
            step_number = step_number.saturating_add(1);
        });
    });
}

criterion_group!(benches, bench_training_step);
criterion_main!(benches);
