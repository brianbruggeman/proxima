// test-only module: expect is the failure message a test wants, as in optimizer.rs tests
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use conflaguration::{Settings, Validate};
use proxima_numeric::{DType, FloatLayout};
use proxima_tensor::op::{Extent, NodeId, Op};

use crate::optimizer::{AdamOperands, AdamwConfig, OptimizerError, adamw_step, step_input};

const LANES: usize = 8;

const OLD_PARAM_BITS: [u32; LANES] = [
    3211540828, 3198886941, 3196750137, 3186819362, 3211270963, 3196391991, 3207408940, 3210407390,
];
const OLD_FIRST_BITS: [u32; LANES] = [
    1030908960, 3176261028, 3168834814, 1015121352, 1001483944, 3164845027, 1026938164, 1011371446,
];
const OLD_SECOND_BITS: [u32; LANES] = [
    976456382, 1002565717, 1008960869, 985841740, 996246356, 986172074, 1004012530, 1007695115,
];

fn splitmix(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^= mixed >> 31;
    (mixed >> 40) as f32 / (1u64 << 24) as f32
}

struct Scenario {
    param: Vec<f32>,
    grad: Vec<f32>,
    first: Vec<f32>,
    second: Vec<f32>,
}

fn scenario() -> Scenario {
    let mut state = 0x5EED_u64;
    let mut draw = |scale: f32, shift: f32| -> Vec<f32> {
        (0..LANES)
            .map(|_| (splitmix(&mut state) - shift) * scale)
            .collect()
    };
    Scenario {
        param: draw(2.0, 0.5),
        grad: draw(0.5, 0.5),
        first: draw(0.1, 0.5),
        second: draw(0.01, 0.0),
    }
}

fn leaf(program: &mut Vec<Op>, name: &str) -> NodeId {
    proxima_tensor::op::append(
        program,
        Op::Input {
            dtype: DType::FLOAT32,
            shape: vec![Extent::Static(LANES as u32)],
            name: Some(String::from(name)),
        },
    )
}

fn run_step(config: &AdamwConfig, data: &Scenario) -> [Vec<f32>; 3] {
    let mut program = Vec::new();
    let param = leaf(&mut program, "param");
    let grad = leaf(&mut program, "grad");
    let first = leaf(&mut program, "m");
    let second = leaf(&mut program, "v");
    let step = step_input(&mut program, "step");
    let operands = AdamOperands {
        param,
        grad,
        m: first,
        v: second,
    };
    let (new_param, new_m, new_v) = adamw_step(&mut program, config, 1, operands, step)
        .expect("the layouts are supported");
    let evaluated = proxima_tensor::cpu::evaluate_named(
        &program,
        &[],
        &[
            ("param", &data.param),
            ("grad", &data.grad),
            ("m", &data.first),
            ("v", &data.second),
            ("step", &[3.0f32]),
        ],
        &[new_param, new_m, new_v],
    )
    .expect("the adamw program lowers and evaluates");
    [new_param, new_m, new_v].map(|node| evaluated.get(node).expect("node requested").0.to_vec())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn non_default() -> AdamwConfig {
    AdamwConfig::builder()
        .learning_rate(0.002)
        .beta1(0.8)
        .beta2(0.99)
        .epsilon(1e-6)
        .weight_decay(0.05)
        .hyper_layout(FloatLayout::F16)
        .first_moment_layout(FloatLayout::BF16)
        .second_moment_layout(FloatLayout::BF16)
        .build()
}

const NON_DEFAULT_TOML: &str = r#"
learning_rate = 0.002
beta1 = 0.8
beta2 = 0.99
epsilon = 1e-6
weight_decay = 0.05
hyper_layout = "float16"
first_moment_layout = "bfloat16"
second_moment_layout = "bfloat16"
"#;

#[test]
fn builder_toml_and_env_agree() {
    let from_builder = non_default();
    let directory = tempfile::tempdir().expect("scratch directory");
    let path = directory.path().join("adamw.toml");
    std::fs::write(&path, NON_DEFAULT_TOML).expect("toml written");
    let from_toml: AdamwConfig = conflaguration::from_file(&path).expect("toml parses");
    let from_env = temp_env::with_vars(
        [
            ("AUTOGRAD_ADAMW_LEARNING_RATE", Some("0.002")),
            ("AUTOGRAD_ADAMW_BETA1", Some("0.8")),
            ("AUTOGRAD_ADAMW_BETA2", Some("0.99")),
            ("AUTOGRAD_ADAMW_EPSILON", Some("1e-6")),
            ("AUTOGRAD_ADAMW_WEIGHT_DECAY", Some("0.05")),
            ("AUTOGRAD_ADAMW_HYPER_LAYOUT", Some("float16")),
            ("AUTOGRAD_ADAMW_FIRST_MOMENT_LAYOUT", Some("bfloat16")),
            ("AUTOGRAD_ADAMW_SECOND_MOMENT_LAYOUT", Some("bfloat16")),
        ],
        AdamwConfig::from_env,
    )
    .expect("env parses");

    assert_eq!(from_toml, from_builder, "toml and builder disagree");
    assert_eq!(from_env, from_builder, "env and builder disagree");
    assert_eq!(AdamwConfig::builder().build(), AdamwConfig::default());
    assert!(from_builder.validate().is_ok());
    let wide = AdamwConfig::builder()
        .first_moment_layout(FloatLayout::F64)
        .build();
    assert!(wide.validate().is_err(), "a 64-bit layout must be refused");
}

#[test]
fn f32_moments_step_is_bit_identical_to_the_captured_old_result() {
    let [param, first, second] = run_step(&AdamwConfig::default(), &scenario());

    assert_eq!(bits(&param), OLD_PARAM_BITS, "param");
    assert_eq!(bits(&first), OLD_FIRST_BITS, "first moment");
    assert_eq!(bits(&second), OLD_SECOND_BITS, "second moment");
}

#[test]
fn bf16_moments_step_agrees_with_an_f64_reference_within_derived_u() {
    // u = 2^-(mantissa_bits+1) is bf16's unit roundoff; the stored m is the input rounded once (b1*u*|m|) plus the output rounded once (u*|m_new|), v likewise with b2, and param inherits the same relative error through m_hat/(sqrt(v_hat)+eps); 2 margin covers the f32 compute and the f64 reference's own f32 inputs
    let layout = FloatLayout::BF16;
    let unit = 0.5f64.powi(i32::from(layout.mantissa_bits) + 1);
    let config = AdamwConfig::builder()
        .first_moment_layout(layout)
        .second_moment_layout(layout)
        .build();
    let data = scenario();
    let mut stored = scenario();
    config.store_first_moment(&mut stored.first).expect("bf16 runs");
    config.store_second_moment(&mut stored.second).expect("bf16 runs");
    let [param, mut first, mut second] = run_step(&config, &stored);
    config.store_first_moment(&mut first).expect("bf16 runs");
    config.store_second_moment(&mut second).expect("bf16 runs");

    let step = 3.0f64;
    let slack = 2.0;
    for lane in 0..LANES {
        let (p, g, m, v) = (
            f64::from(data.param[lane]),
            f64::from(data.grad[lane]),
            f64::from(data.first[lane]),
            f64::from(data.second[lane]),
        );
        let m_ref = config.beta1 * m + (1.0 - config.beta1) * g;
        let v_ref = config.beta2 * v + (1.0 - config.beta2) * g * g;
        let m_hat = m_ref / (1.0 - config.beta1.powf(step));
        let v_hat = v_ref / (1.0 - config.beta2.powf(step));
        let rate = 1.0 / (v_hat.sqrt() + config.epsilon);
        let param_ref = p - config.learning_rate * (m_hat * rate + config.weight_decay * p);

        let m_bound = slack * unit * (config.beta1 * m.abs() + m_ref.abs()) + 1e-9;
        let v_bound = slack * unit * (config.beta2 * v.abs() + v_ref.abs()) + 1e-12;
        let bias_one = 1.0 - config.beta1.powf(step);
        let param_bound = slack
            * config.learning_rate
            * rate
            * unit
            * (config.beta1 * m.abs() / bias_one + m_hat.abs())
            + 1e-6;
        assert!((f64::from(first[lane]) - m_ref).abs() <= m_bound, "m lane {lane}");
        assert!((f64::from(second[lane]) - v_ref).abs() <= v_bound, "v lane {lane}");
        assert!(
            (f64::from(param[lane]) - param_ref).abs() <= param_bound,
            "param lane {lane}: {} vs {param_ref}",
            param[lane]
        );
    }
}

#[test]
fn optimizer_source_has_no_f32_typed_public_field() {
    let source = include_str!("optimizer.rs");
    let offenders: Vec<&str> = source
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("pub ") && trimmed.contains(": f32")
        })
        .collect();

    assert!(offenders.is_empty(), "f32-typed public items: {offenders:?}");
}

#[test]
fn unsupported_layouts_are_refused_by_name_instead_of_falling_back() {
    let wide = AdamwConfig::builder()
        .hyper_layout(FloatLayout::F64)
        .first_moment_layout(FloatLayout::INT8)
        .build();
    let mut program = Vec::new();
    let param = leaf(&mut program, "param");
    let step = step_input(&mut program, "step");
    let operands = AdamOperands {
        param,
        grad: param,
        m: param,
        v: param,
    };

    let stepped = adamw_step(&mut program, &wide, 1, operands, step);
    let stored = wide.store_first_moment(&mut [0.5f32]);

    assert!(
        matches!(
            stepped,
            Err(OptimizerError::UnsupportedLayout { field: "hyper_layout", layout, .. })
                if layout == FloatLayout::F64
        ),
        "a 64-bit hyper layout must name itself: {stepped:?}"
    );
    assert!(
        matches!(
            stored,
            Err(OptimizerError::NotFloat { field: "first_moment_layout", .. })
        ),
        "an integer moment layout must be refused: {stored:?}"
    );
}
