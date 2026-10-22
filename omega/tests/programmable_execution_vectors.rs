#![cfg(all(feature = "cpu", feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use half::{bf16, f16};
use omega::backend::{execute_plan_named, plan_named_with_config};
use omega::config::RuntimeConfig;
use proxima_primitives::Codec;
use proxima_tensor::spec::{elementwise, input_leaf, reduce};
use proxima_tensor::{
    DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock, ReduceInit, ScalarOp,
};

fn matrix_program(rows: u32, reduction: u32, columns: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let activation = input_leaf(
        &mut program,
        DType::FLOAT32,
        vec![Extent::Static(rows), Extent::Static(reduction)],
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
    .expect("matrix fixture product must have compatible axes");
    let output = reduce(
        &mut program,
        DType::FLOAT32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "sio->sio",
        "so->sio",
    )
    .expect("matrix fixture reduction must preserve row and column axes");
    (program, output)
}

fn mismatches(expected: &[f32], actual: &[f32]) -> Vec<usize> {
    assert!(
        !expected.is_empty(),
        "oracle must compare actual coordinates"
    );
    assert_eq!(expected.len(), actual.len(), "coordinate counts must match");
    expected
        .iter()
        .zip(actual)
        .enumerate()
        .filter_map(|(index, (expected, actual))| {
            (!actual.is_finite() || expected.to_bits() != actual.to_bits()).then_some(index)
        })
        .collect()
}

fn packed(values: &[f32], codec: Codec) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| match codec {
            Codec::BFloat16 => bf16::from_f32(*value).to_bits().to_le_bytes(),
            Codec::Float16 => f16::from_f32(*value).to_bits().to_le_bytes(),
            _ => panic!("fixture supports only the two float storage formats"),
        })
        .collect()
}

fn execute_literal(
    backend: &str,
    rows: u32,
    weight_codec: Option<Codec>,
    activation_codec: Option<Codec>,
    ragged: bool,
) {
    let (reduction, columns) = if ragged { (35, 5) } else { (2, 2) };
    let (program, output) = matrix_program(rows, reduction, columns);
    let weights: Vec<f32> = if ragged && weight_codec.is_some() {
        [1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .flat_map(|value| vec![value; reduction as usize])
            .collect()
    } else if ragged {
        (0..reduction)
            .flat_map(|_| [1.0, 2.0, 3.0, 4.0, 5.0])
            .collect()
    } else if weight_codec.is_some() {
        vec![1.0, 2.0, 3.0, 4.0]
    } else {
        vec![1.0, 3.0, 2.0, 4.0]
    };
    let activations: Vec<f32> = if ragged {
        (0..rows)
            .flat_map(|row| vec![if row % 2 == 0 { 1.0 } else { 2.0 }; reduction as usize])
            .collect()
    } else {
        (0..rows)
            .flat_map(|row| if row % 2 == 0 { [5.0, 6.0] } else { [7.0, 8.0] })
            .collect()
    };
    let expected: Vec<f32> = if ragged {
        (0..rows)
            .flat_map(|row| {
                if row % 2 == 0 {
                    [35.0, 70.0, 105.0, 140.0, 175.0]
                } else {
                    [70.0, 140.0, 210.0, 280.0, 350.0]
                }
            })
            .collect()
    } else {
        (0..rows)
            .flat_map(|row| {
                if row % 2 == 0 {
                    [17.0, 39.0]
                } else {
                    [23.0, 53.0]
                }
            })
            .collect()
    };
    let weight_bytes = weight_codec.map(|codec| packed(&weights, codec));
    let activation_bytes = activation_codec.map(|codec| packed(&activations, codec));
    let weight_block = match (weight_codec, weight_bytes.as_deref()) {
        (Some(codec), Some(bytes)) => QuantizedBlock::Packed { codec, bytes },
        _ => QuantizedBlock::Float32(&weights),
    };
    let activation_block = match (activation_codec, activation_bytes.as_deref()) {
        (Some(codec), Some(bytes)) => QuantizedBlock::Packed { codec, bytes },
        _ => QuantizedBlock::Float32(&activations),
    };
    let bindings = [("activation", activation_block), ("weight", weight_block)];
    let config = RuntimeConfig {
        engine: if backend == "cpu" { "cpu" } else { "gpu" }.into(),
        gpu_driver: if backend == "cpu" { "auto" } else { "metal" }.into(),
        allow_fusion: true,
        numeric: NumericPolicy::llama_relaxed(),
        ..RuntimeConfig::default()
    };
    let (mut plan, records) = plan_named_with_config(&config, &program, &[], &bindings, &[output])
        .expect("configured backend must prepare literal matrix fixture");
    assert_eq!(
        records.len(),
        1,
        "configured planning must record one candidate"
    );
    assert!(
        records[0].selected,
        "configured candidate must be identified"
    );
    if backend == "metal" && rows >= 160 {
        assert!(
            records[0]
                .kernel_identities
                .iter()
                .any(|identity| identity.contains("_dbg")),
            "float storage fixture rows={rows} must execute the admitted dense tiled kernel: {:?}",
            records[0].kernel_identities,
        );
    }
    let evaluated = execute_plan_named(&mut plan, &bindings)
        .expect("configured backend must execute literal matrix fixture");
    #[cfg(feature = "instrument")]
    if backend == "metal" {
        let captured = omega::metal::take_captured_dispatches();
        let dispatch = captured
            .iter()
            .find(|dispatch| dispatch.node == output.0)
            .expect("the selected literal matrix dispatch is captured");
        println!(
            "captured_dispatch node={} entry={} source_sha256={} extents={:?} grid={:?} operands={:?}",
            dispatch.node,
            dispatch.entry,
            dispatch.msl_sha256,
            dispatch.extents,
            dispatch.grid,
            dispatch.operands,
        );
    }
    let (actual, shape) = evaluated
        .get(output)
        .expect("requested matrix root must be present");
    assert_eq!(shape, &[u64::from(rows), u64::from(columns)]);
    assert_eq!(mismatches(&expected, actual), Vec::<usize>::new());
    println!(
        "literal_matrix backend={backend} rows={rows} reduction={reduction} columns={columns} weight_codec={weight_codec:?} activation_codec={activation_codec:?} weight_bytes={weight_bytes:?} activation_bytes={activation_bytes:?} expected={expected:?} actual={actual:?} kernels={:?}",
        records[0].kernel_identities,
    );
}

#[test]
fn programmable_vector_cpu_float() {
    execute_literal("cpu", 2, None, None, false);
}

#[test]
fn programmable_vector_cpu_bf16() {
    execute_literal("cpu", 2, Some(Codec::BFloat16), None, false);
}

#[test]
fn programmable_vector_metal_bf16() {
    execute_literal("metal", 160, Some(Codec::BFloat16), None, false);
}

#[test]
fn programmable_vector_metal_f16() {
    execute_literal("metal", 160, Some(Codec::Float16), None, false);
}

#[test]
fn programmable_vector_metal_two_bf16_operands() {
    execute_literal(
        "metal",
        160,
        Some(Codec::BFloat16),
        Some(Codec::BFloat16),
        false,
    );
}

#[test]
fn programmable_vector_metal_ragged_mixed_storage() {
    execute_literal(
        "metal",
        160,
        Some(Codec::BFloat16),
        Some(Codec::Float16),
        true,
    );
}

#[test]
fn programmable_vector_oracle_self_control() {
    assert_eq!(
        mismatches(&[17.0, 39.0, 23.0, 53.0], &[17.0, 39.0, 23.0, 53.0]),
        Vec::<usize>::new()
    );
}

#[test]
fn programmable_vector_oracle_corruption_control() {
    assert_eq!(
        mismatches(&[17.0, 39.0, 23.0, 53.0], &[17.0, 39.0, 24.0, 53.0]),
        vec![2]
    );
}
