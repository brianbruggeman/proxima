//! The public training surface: one `TrainConfig` drives graph construction and
//! planning, and a [`TrainingStep`] owns the parameter/optimizer state across
//! steps on `Engine::Cpu` and `Engine::Gpu`.
//!
//! The CPU arm is the oracle the Metal arm is compared against, the same
//! contract `training_step_parity.rs` proves at the plan/execute level. The
//! llama test pins the benchmark's configuration to the incumbent's defaults
//! (`llama.cpp` tag `b11671`: `common/common.h:438,442,443,458,459,619,620`,
//! `ggml/src/ggml-opt.cpp:229-231`, `common/common.cpp:2108-2109`) so the bench
//! cannot drift into a benchmark-only optimizer.
#![cfg(all(feature = "train", feature = "cpu"))]
// assertions name the failure; the workspace denies clippy::expect_used
#![allow(clippy::expect_used)]

use std::io::Write;

use omega::config::RuntimeConfig;
use omega::train::{
    AdamConfig, AdamwConfig, LLAMA_FINETUNE_CONFIG, TrainConfig, TrainError, TrainSchedule,
    adam_first_moment_name, adam_second_moment_name, build_optimizer_step,
};
use proxima_autograd::loss::softmax_cross_entropy;
use proxima_numeric::DType;
use proxima_numeric::float_layout::FloatLayout;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append};
use proxima_autograd::optimizer::store_moment;

const IN_DIM: usize = 2;
const OUT_DIM: usize = 3;
const PARAMETERS: [&str; 2] = ["w", "b"];

fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    append(
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

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    append(
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
    append(
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

fn build_forward() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", vec![Extent::Static(IN_DIM as u32)]);
    let y = leaf(&mut program, "y", vec![Extent::Static(OUT_DIM as u32)]);
    let w = leaf(
        &mut program,
        "w",
        vec![
            Extent::Static(IN_DIM as u32),
            Extent::Static(OUT_DIM as u32),
        ],
    );
    let b = leaf(&mut program, "b", vec![Extent::Static(OUT_DIM as u32)]);

    let product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![
            (w, identity(2)),
            (x, IndexMap::Affine(map::projection(2, &[0]))),
        ],
    );
    let matmul = reduce_add(
        &mut program,
        product,
        identity(2),
        IndexMap::Affine(map::projection(2, &[1])),
    );
    let logits = elementwise(
        &mut program,
        ScalarOp::Add,
        vec![(matmul, identity(1)), (b, identity(1))],
    );
    let loss = softmax_cross_entropy(&mut program, DType::FLOAT32, logits, y, 1, 0);
    (program, loss)
}

fn counter_pattern(seed: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| (((seed + index) * 7 % 13) as f32 - 6.0) / 12.0)
        .collect()
}

fn initial_state() -> Vec<(String, Vec<f32>)> {
    let mut named = vec![
        (String::from("x"), vec![0.5f32, -0.25f32]),
        (String::from("y"), vec![0.0f32, 1.0, 0.0]),
        (String::from("step"), vec![1.0f32]),
        (String::from("w"), counter_pattern(1, IN_DIM * OUT_DIM)),
        (String::from("b"), vec![0.0f32; OUT_DIM]),
    ];
    for parameter in PARAMETERS {
        named.push((
            adam_first_moment_name(parameter),
            vec![
                0.0f32;
                if parameter == "w" {
                    IN_DIM * OUT_DIM
                } else {
                    OUT_DIM
                }
            ],
        ));
        named.push((
            adam_second_moment_name(parameter),
            vec![
                0.0f32;
                if parameter == "w" {
                    IN_DIM * OUT_DIM
                } else {
                    OUT_DIM
                }
            ],
        ));
    }
    named
}

fn as_named(owned: &[(String, Vec<f32>)]) -> Vec<(&str, QuantizedBlock<'_>)> {
    owned
        .iter()
        .map(|(name, values)| (name.as_str(), QuantizedBlock::Float32(values.as_slice())))
        .collect()
}

fn batch(step_number: &[f32]) -> Vec<(&'static str, QuantizedBlock<'_>)> {
    static X: [f32; IN_DIM] = [0.5, -0.25];
    static Y: [f32; OUT_DIM] = [0.0, 1.0, 0.0];
    vec![
        ("x", QuantizedBlock::Float32(&X)),
        ("y", QuantizedBlock::Float32(&Y)),
        ("step", QuantizedBlock::Float32(step_number)),
    ]
}

fn cpu_config() -> TrainConfig {
    let runtime = RuntimeConfig {
        engine: String::from("cpu"),
        ..RuntimeConfig::default()
    };
    TrainConfig::builder()
        .runtime(runtime)
        .adam(AdamConfig {
            learning_rate: 0.05,
            ..AdamConfig::default()
        })
        .build()
        .expect("the CPU config validates")
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_config() -> TrainConfig {
    let runtime = RuntimeConfig {
        engine: String::from("gpu"),
        gpu_driver: String::from("metal"),
        ..RuntimeConfig::default()
    };
    TrainConfig::builder()
        .runtime(runtime)
        .adam(AdamConfig {
            learning_rate: 0.05,
            ..AdamConfig::default()
        })
        .build()
        .expect("the Metal config validates")
}

#[test]
fn llama_finetune_config_is_the_benchmark_config() {
    let path = LLAMA_FINETUNE_CONFIG;
    let config =
        TrainConfig::from_file(path).expect("the checked-in llama config loads and validates");

    assert_eq!(config.optimizer, "adamw");
    assert_eq!(config.adamw.learning_rate, 1.0e-5);
    assert_eq!(config.adamw.beta1, 0.9);
    assert_eq!(config.adamw.beta2, 0.999);
    assert_eq!(config.adamw.epsilon, 1.0e-8);
    assert_eq!(config.adamw.weight_decay, 0.0);
    assert_eq!(config.schedule.epochs, 2);
    assert_eq!(config.schedule.val_split, 0.05);
    assert_eq!(config.schedule.batch, 2048);
    assert_eq!(config.schedule.ubatch, 512);
    assert_eq!(config.adamw.hyper_layout, FloatLayout::BF32);
    assert_eq!(config.adamw.first_moment_layout, FloatLayout::BF32);
    assert_eq!(config.adamw.second_moment_layout, FloatLayout::BF32);
}

#[test]
fn environment_overlay_reaches_every_nested_layer() {
    temp_env::with_vars(
        [
            ("OMEGA_TRAIN_OPTIMIZER", Some("adamw")),
            ("OMEGA_TRAIN_RUNTIME_ENGINE", Some("cpu")),
            ("OMEGA_TRAIN_SCHEDULE_UBATCH", Some("128")),
            ("OMEGA_TRAIN_ADAMW_LEARNING_RATE", Some("0.25")),
        ],
        || {
            let config = TrainConfig::from_env().expect("the environment layer validates");
            assert_eq!(config.optimizer, "adamw");
            assert_eq!(config.runtime.engine, "cpu");
            assert_eq!(config.schedule.ubatch, 128);
            assert_eq!(config.adamw.learning_rate, 0.25);
        },
    );
}

#[test]
fn builder_overrides_every_group_and_validates() {
    let runtime = RuntimeConfig {
        engine: String::from("cpu"),
        ..RuntimeConfig::default()
    };
    let schedule = TrainSchedule {
        ubatch: 128,
        ..TrainSchedule::default()
    };
    let config = TrainConfig::builder()
        .runtime(runtime)
        .optimizer("adamw")
        .schedule(schedule)
        .adam(AdamConfig {
            learning_rate: 0.5,
            ..AdamConfig::default()
        })
        .adamw(AdamwConfig {
            learning_rate: 0.25,
            weight_decay: 0.01,
            ..AdamwConfig::default()
        })
        .build()
        .expect("every group is explicit and valid");

    assert_eq!(config.runtime.engine, "cpu");
    assert_eq!(config.optimizer, "adamw");
    assert_eq!(config.schedule.ubatch, 128);
    assert_eq!(config.adam.learning_rate, 0.5);
    assert_eq!(config.adamw.learning_rate, 0.25);
    assert_eq!(config.adamw.weight_decay, 0.01);
}

#[test]
fn the_environment_overrides_the_file_not_the_other_way_round() {
    let mut file = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("a temporary config file");
    file.write_all(b"optimizer = \"adam\"\n[adam]\nlearning_rate = 0.5\n")
        .expect("the file is writable");
    temp_env::with_vars(
        [
            ("OMEGA_TRAIN_OPTIMIZER", Some("adamw")),
            ("OMEGA_TRAIN_ADAM_LEARNING_RATE", Some("0.25")),
        ],
        || {
            let config = TrainConfig::from_file_then_env(file.path())
                .expect("the file plus environment layer validates");
            assert_eq!(config.optimizer, "adamw");
            assert_eq!(config.adam.learning_rate, 0.25);
        },
    );
}

#[test]
fn a_nested_unknown_field_is_refused_by_name() {
    let mut file = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("a temporary config file");
    file.write_all(b"optimizer = \"adamw\"\n[adamw]\nlearnign_rate = 0.1\n")
        .expect("the file is writable");
    let error = TrainConfig::from_file(file.path())
        .expect_err("a misspelled nested field must not be silently dropped");
    let message = error.to_string();
    assert!(
        message.contains("learnign_rate"),
        "the error must name the unknown nested field, got {message}"
    );
}

#[test]
fn an_unknown_field_is_refused_by_name() {
    let mut file = tempfile::Builder::new()
        .suffix(".toml")
        .tempfile()
        .expect("a temporary config file");
    file.write_all(b"optimizer = \"adamw\"\nbogus = 1\n")
        .expect("the file is writable");
    let error = TrainConfig::from_file(file.path())
        .expect_err("an unknown field must not be silently ignored");
    let message = error.to_string();
    assert!(
        message.contains("bogus"),
        "the error must name the unknown field, got {message}"
    );
}

#[test]
fn an_unknown_optimizer_is_refused_by_name() {
    let error = TrainConfig::builder()
        .optimizer("lbfgs")
        .build()
        .expect_err("lbfgs is not implemented");
    let message = error.to_string();
    assert!(
        message.contains("optimizer") && message.contains("lbfgs"),
        "the error must name the field and the value, got {message}"
    );
}

#[test]
fn ten_steps_through_one_plan_drop_the_loss_and_move_every_parameter() {
    let (program, loss) = build_forward();
    let built = build_optimizer_step(&program, loss, &PARAMETERS, &cpu_config())
        .expect("the MLP differentiates");
    let state = initial_state();
    let initial_w = state
        .iter()
        .find(|(name, _)| name == "w")
        .map(|(_, values)| values.clone())
        .expect("w is bound");

    let mut training = built
        .plan(&as_named(&state))
        .expect("the CPU plans the training step");

    let mut losses = Vec::with_capacity(10);
    for step_number in 1..=10u8 {
        let step_value = [f32::from(step_number)];
        losses.push(
            training
                .step(&batch(&step_value))
                .expect("the CPU executes the training step"),
        );
    }

    assert!(
        losses.last().expect("ten steps ran") < losses.first().expect("ten steps ran"),
        "loss must drop, got {losses:?}"
    );
    assert!(
        losses.iter().all(|value| value.is_finite()),
        "loss went non-finite: {losses:?}"
    );
    assert_ne!(
        training.state_value("w").expect("w is owned"),
        initial_w.as_slice(),
        "the parameter must move"
    );
    for parameter in PARAMETERS {
        assert!(
            training
                .state_value(&adam_first_moment_name(parameter))
                .expect("the first moment is owned")
                .iter()
                .any(|value| *value != 0.0),
            "the first moment {parameter} never left zero"
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_and_cpu_run_the_same_training_step_at_parity() {
    let (program, loss) = build_forward();
    let cpu_built = build_optimizer_step(&program, loss, &PARAMETERS, &cpu_config())
        .expect("the MLP differentiates for the CPU");
    let metal_built = build_optimizer_step(&program, loss, &PARAMETERS, &metal_config())
        .expect("the MLP differentiates for Metal");
    let state = initial_state();
    let named = as_named(&state);

    let mut cpu = cpu_built
        .plan(&named)
        .expect("the CPU plans the training step");
    let mut metal = metal_built
        .plan(&named)
        .expect("Metal plans the training step");

    for step_number in 1..=3u8 {
        let step_value = [f32::from(step_number)];
        let cpu_loss = cpu.step(&batch(&step_value)).expect("CPU executes");
        let metal_loss = metal.step(&batch(&step_value)).expect("Metal executes");
        assert!(
            (cpu_loss - metal_loss).abs() < 1e-4,
            "step {step_number}: loss cpu={cpu_loss} metal={metal_loss}"
        );
    }

    let mut worst = 0.0f32;
    for (name, values) in cpu.state() {
        let metal_values = metal
            .state_value(name)
            .expect("Metal owns the same state names");
        assert_eq!(values.len(), metal_values.len(), "{name} length");
        for (&want, &got) in values.iter().zip(metal_values.iter()) {
            assert!(got.is_finite(), "{name} produced a non-finite value");
            worst = worst.max((want - got).abs());
        }
    }
    assert!(worst < 1e-4, "training-state parity worst_diff={worst}");
}

#[test]
fn a_state_buffer_cannot_be_overridden_by_the_batch() {
    let (program, loss) = build_forward();
    let built = build_optimizer_step(&program, loss, &PARAMETERS, &cpu_config())
        .expect("the MLP differentiates");
    let state = initial_state();
    let mut training = built
        .plan(&as_named(&state))
        .expect("the CPU plans the training step");

    let overridden = [0.0f32; 3];
    let error = training
        .step(&[("b", QuantizedBlock::Float32(&overridden))])
        .expect_err("a batch may not replace owned state");
    assert!(matches!(
        error,
        TrainError::BatchOverridesState { ref name } if name == "b"
    ));
}

#[test]
fn a_missing_state_binding_is_refused_by_name() {
    let (program, loss) = build_forward();
    let built = build_optimizer_step(&program, loss, &PARAMETERS, &cpu_config())
        .expect("the MLP differentiates");
    let error = built
        .plan(&[])
        .err()
        .expect("a plan with no state must be refused");
    assert!(matches!(
        error,
        TrainError::MissingStateBinding { ref name } if name == "w"
    ));
}

#[test]
fn a_state_buffer_bound_with_another_dtype_is_refused() {
    let (program, loss) = build_forward();
    let built = build_optimizer_step(&program, loss, &PARAMETERS, &cpu_config())
        .expect("the MLP differentiates");
    let state = initial_state();
    let mut named = as_named(&state);
    for entry in &mut named {
        if entry.0 == "w" {
            *entry = ("w", QuantizedBlock::Int32(&[0i32; 4]));
        }
    }
    let error = built
        .plan(&named)
        .err()
        .expect("a non-float state buffer is refused");
    assert!(matches!(
        error,
        TrainError::StateBinding { ref name } if name == "w"
    ));
}

fn assert_moment_layouts(wide: &TrainConfig, narrow: &TrainConfig) {
    let (program, loss) = build_forward();
    let wide_built =
        build_optimizer_step(&program, loss, &PARAMETERS, wide).expect("the MLP differentiates");
    let narrow_built = build_optimizer_step(&program, loss, &PARAMETERS, narrow)
        .expect("the same graph with narrow moment storage differentiates");
    let state = initial_state();
    let mut wide_step = wide_built.plan(&as_named(&state)).expect("wide plans");
    let mut narrow_step = narrow_built.plan(&as_named(&state)).expect("narrow plans");

    let step_value = [1.0f32];
    wide_step.step(&batch(&step_value)).expect("wide executes");
    narrow_step
        .step(&batch(&step_value))
        .expect("narrow executes");

    let first = adam_first_moment_name("w");
    let second = adam_second_moment_name("w");
    let mut expected_first = wide_step
        .state_value(&first)
        .expect("the wide step owns m_w")
        .to_vec();
    store_moment(&mut expected_first, FloatLayout::BF16, "m_w")
        .expect("bf16 narrows the wide first moment");
    assert_eq!(
        narrow_step
            .state_value(&first)
            .expect("the narrow step owns m_w"),
        expected_first.as_slice(),
        "the step must store m through first_moment_layout"
    );
    assert_ne!(
        narrow_step
            .state_value(&first)
            .expect("the narrow step owns m_w"),
        wide_step
            .state_value(&first)
            .expect("the wide step owns m_w"),
        "bf16 storage must differ from float32 storage for this moment"
    );

    let mut expected_second = wide_step
        .state_value(&second)
        .expect("the wide step owns v_w")
        .to_vec();
    store_moment(&mut expected_second, FloatLayout::F16, "v_w")
        .expect("f16 narrows the wide second moment");
    assert_eq!(
        narrow_step
            .state_value(&second)
            .expect("the narrow step owns v_w"),
        expected_second.as_slice(),
        "the step must store v through second_moment_layout"
    );
    assert_eq!(
        narrow_step.state_value("w").expect("w is owned"),
        wide_step.state_value("w").expect("w is owned"),
        "the parameter itself is not narrowed"
    );
}

#[test]
fn adam_moment_layouts_narrow_the_owned_state() {
    let wide = cpu_config();
    let mut narrow = cpu_config();
    narrow.adam.first_moment_layout = FloatLayout::BF16;
    narrow.adam.second_moment_layout = FloatLayout::F16;
    assert_moment_layouts(&wide, &narrow);
}

#[test]
fn adamw_moment_layouts_narrow_the_owned_state() {
    let mut wide = cpu_config();
    wide.optimizer = String::from("adamw");
    wide.adamw = AdamwConfig {
        learning_rate: 0.05,
        weight_decay: 0.0,
        ..AdamwConfig::default()
    };
    let mut narrow = wide.clone();
    narrow.adamw.first_moment_layout = FloatLayout::BF16;
    narrow.adamw.second_moment_layout = FloatLayout::F16;
    assert_moment_layouts(&wide, &narrow);
}

#[test]
fn a_parameter_with_a_gathered_read_is_refused() {
    use proxima_tensor::map::{AxisIndex, AxisTerm};

    let mut program = Vec::new();
    let table = leaf(
        &mut program,
        "table",
        vec![Extent::Static(2), Extent::Static(2)],
    );
    let ids = append(
        &mut program,
        Op::Input {
            dtype: DType::INT32,
            shape: vec![Extent::Static(1)],
            name: Some(String::from("ids")),
        },
    );
    let dense = elementwise(&mut program, ScalarOp::Identity, vec![(table, identity(2))]);
    let dense_sum = reduce_add(
        &mut program,
        dense,
        identity(2),
        IndexMap::Affine(map::projection(2, &[])),
    );
    let gather_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: map::IndexPattern {
            iter_rank: 2,
            axes: vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: std::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let gathered = elementwise(&mut program, ScalarOp::Identity, vec![(table, gather_map)]);
    let gathered_sum = reduce_add(
        &mut program,
        gathered,
        identity(2),
        IndexMap::Affine(map::projection(2, &[])),
    );
    let loss = elementwise(
        &mut program,
        ScalarOp::Add,
        vec![(dense_sum, identity(0)), (gathered_sum, identity(0))],
    );

    let error = build_optimizer_step(&program, loss, &["table"], &cpu_config())
        .expect_err("a parameter read both densely and through a gather must be refused");
    assert!(matches!(error, TrainError::GatheredGradient { ref name } if name == "table"));
}

#[test]
fn adamw_is_a_selectable_optimizer_in_the_same_config() {
    let (program, loss) = build_forward();
    let mut config = cpu_config();
    config.optimizer = String::from("adamw");
    config.adamw = AdamwConfig {
        learning_rate: 0.05,
        weight_decay: 0.0,
        ..AdamwConfig::default()
    };

    let built = build_optimizer_step(&program, loss, &PARAMETERS, &config)
        .expect("adamw builds the same graph shape");
    let mut training = built
        .plan(&as_named(&initial_state()))
        .expect("adamw plans");

    let before = training.state_value("w").expect("w is owned").to_vec();
    let step_value = [1.0f32];
    training.step(&batch(&step_value)).expect("adamw executes");
    assert_ne!(
        training.state_value("w").expect("w is owned"),
        before.as_slice(),
        "adamw must move the parameter"
    );
}
