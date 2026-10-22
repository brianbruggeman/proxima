//! Single-row projection on real Ouro 1.4B bytes, through the configured Metal path.
//!
//! The fixture is `omega/tests/fixtures/ouro_1_4b_q_proj_row0_and_token504_embedding.bf16`: 4096
//! bytes of `model.layers.0.self_attn.q_proj.weight` (BF16, absolute file bytes 488687986..488692082,
//! the first 2048 elements) followed by 4096 bytes of `model.embed_tokens.weight` row 504 (the first
//! frozen prefix token), both copied unchanged from the checkpoint with sha256
//! `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af` at revision
//! `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. Output `o` reads weight elements `o * reduction ..`,
//! the packed `[out, in]` convention. The reference is an independent f64 dot product of the decoded
//! bytes, compared coordinate by coordinate against `gamma_n * sum(abs(products))`, the same
//! envelope the AC10 microbench records.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::bf16;
use omega::backend::{execute_plan_named, plan_named_with_config};
use omega::config::RuntimeConfig;
use proxima_primitives::Codec;
use proxima_tensor::spec::{elementwise, input_leaf, reduce};
use proxima_tensor::{
    DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock, ReduceInit, ScalarOp,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/ouro_1_4b_q_proj_row0_and_token504_embedding.bf16");
const WEIGHT_BYTES: usize = 4096;

struct Reference {
    expected: Vec<f64>,
    bounds: Vec<f64>,
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16::from_le_bytes(*pair).to_f32())
        .collect()
}

fn reference(weight: &[f32], activation: &[f32], reduction: usize) -> Reference {
    let roundoff = reduction as f64 * f64::from(f32::EPSILON);
    let (expected, bounds) = weight
        .chunks_exact(reduction)
        .map(|row| {
            let (sum, magnitude) = row.iter().zip(activation).fold(
                (0.0_f64, 0.0_f64),
                |(sum, magnitude), (weight, activation)| {
                    let product = f64::from(*weight) * f64::from(*activation);
                    (sum + product, magnitude + product.abs())
                },
            );
            (sum, roundoff / (1.0 - roundoff) * magnitude)
        })
        .unzip();
    Reference { expected, bounds }
}

fn violations(reference: &Reference, actual: &[f32]) -> Vec<usize> {
    assert_eq!(
        reference.expected.len(),
        actual.len(),
        "projection output must have one coordinate per column"
    );
    reference
        .expected
        .iter()
        .zip(actual)
        .zip(&reference.bounds)
        .enumerate()
        .filter_map(|(coordinate, ((expected, actual), bound))| {
            let delta = f64::from(*actual) - expected;
            (!delta.is_finite() || delta.abs() > *bound).then_some(coordinate)
        })
        .collect()
}

fn projection_program(reduction: u32, columns: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let activation = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(1), Extent::Static(reduction)],
        "activation",
    );
    let weight = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(reduction), Extent::Static(columns)],
        "weight",
    );
    let product = elementwise(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Multiply,
        &[(activation, "si->sio"), (weight, "io->sio")],
    )
    .expect("projection product has compatible axes");
    let output = reduce(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "sio->sio",
        "so->sio",
    )
    .expect("projection reduction keeps row and column axes");
    (program, output)
}

fn execute_projection(weight_bytes: &[u8], activation: &[f32], columns: usize) -> Vec<f32> {
    let reduction = activation.len();
    let (program, output) = projection_program(reduction as u32, columns as u32);
    let bindings = [
        ("activation", QuantizedBlock::Float32(activation)),
        (
            "weight",
            QuantizedBlock::Packed {
                codec: Codec::BFloat16,
                bytes: weight_bytes,
            },
        ),
    ];
    let config = RuntimeConfig {
        engine: "gpu".into(),
        gpu_driver: "metal".into(),
        allow_fusion: true,
        numeric: NumericPolicy::llama_relaxed(),
        ..RuntimeConfig::default()
    };
    let (mut plan, records) = plan_named_with_config(&config, &program, &[], &bindings, &[output])
        .expect("metal prepares the single-row projection");
    assert!(records[0].selected, "the configured candidate is selected");
    if omega::sized::DENSE_TILED_MIN_TOKENS.is_some_and(|minimum| minimum <= 1) {
        assert!(
            records[0]
                .kernel_identities
                .iter()
                .any(|identity| identity.contains("_dbg")),
            "single-row projection must run the admitted dense tiled kernel: {:?}",
            records[0].kernel_identities,
        );
    }
    let evaluated =
        execute_plan_named(&mut plan, &bindings).expect("metal executes the projection");
    evaluated
        .get(output)
        .expect("requested projection root is present")
        .0
        .to_vec()
}

fn check_prefix_projection(columns: usize, reduction: usize) {
    let weight_bytes = &FIXTURE[..columns * reduction * 2];
    let activation = decode(&FIXTURE[WEIGHT_BYTES..WEIGHT_BYTES + reduction * 2]);
    let reference = reference(&decode(weight_bytes), &activation, reduction);
    assert_eq!(
        reference.expected.len(),
        columns,
        "reference covers every column"
    );

    let actual = execute_projection(weight_bytes, &activation, columns);

    assert_eq!(
        violations(&reference, &actual),
        Vec::<usize>::new(),
        "columns={columns} reduction={reduction} expected[..3]={:?} actual[..3]={:?}",
        &reference.expected[..3.min(columns)],
        &actual[..3.min(columns)],
    );
}

#[test]
fn single_row_projection_matches_f64_reference_across_k_tiles_and_columns() {
    check_prefix_projection(4, 128);
}

#[test]
fn single_row_projection_matches_f64_reference_with_ragged_reduction() {
    check_prefix_projection(7, 100);
}

#[test]
fn single_row_projection_matches_f64_reference_for_one_wide_column() {
    check_prefix_projection(1, 2048);
}

#[test]
fn single_row_projection_matches_f64_reference_for_two_long_columns() {
    check_prefix_projection(2, 1024);
}

#[test]
fn comparator_flags_a_corrupted_coordinate() {
    let reduction = 128;
    let activation = decode(&FIXTURE[WEIGHT_BYTES..WEIGHT_BYTES + reduction * 2]);
    let reference = reference(
        &decode(&FIXTURE[..4 * reduction * 2]),
        &activation,
        reduction,
    );
    let mut corrupted: Vec<f32> = reference
        .expected
        .iter()
        .map(|value| *value as f32)
        .collect();
    assert_eq!(violations(&reference, &corrupted), Vec::<usize>::new());

    corrupted[2] += (2.0 * reference.bounds[2] + 1.0) as f32;

    assert_eq!(violations(&reference, &corrupted), vec![2]);
}
