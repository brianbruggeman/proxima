#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::expect_used)]

use proxima_gguf::quant::q2_k;
use proxima_tensor::{
    DType, Extent, IndexMap, Keep, NumericPolicy, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
    append, evaluate, map,
};

fn matmul_program(rows: u32, width: u32, weight_dtype: DType) -> Vec<Op> {
    let mut program = Vec::new();
    let weight = append(
        &mut program,
        Op::Input {
            dtype: weight_dtype,
            shape: vec![Extent::Static(rows), Extent::Static(width)],
            name: None,
        },
    );
    let activation = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(width), Extent::Static(1)],
            name: None,
        },
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (activation, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
            name: Some("q2k_device_parity".into()),
        }),
    );
    program
}

#[test]
fn q2k_metal_matmul_matches_dequantized_cpu() {
    const ROWS: usize = 4;
    const WIDTH: usize = q2_k::QK_K;

    let weights: Vec<f32> = (0..ROWS * WIDTH)
        .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.01)
        .collect();
    let activation: Vec<f32> = (0..WIDTH)
        .map(|index| ((index * 13 % 67) as f32 - 33.0) * 0.02)
        .collect();
    let mut packed = vec![0u8; ROWS * q2_k::BLOCK_BYTES];
    q2_k::quantize(&weights, &mut packed).expect("quantizes four Q2_K rows");
    let mut dequantized = vec![0.0f32; weights.len()];
    q2_k::dequantize(&packed, &mut dequantized).expect("dequantizes the Q2_K device fixture");

    let packed_program = matmul_program(ROWS as u32, WIDTH as u32, DType::UInt8);
    let packed_root = proxima_tensor::NodeId((packed_program.len() - 1) as u32);
    let metal = omega::execute(
        &packed_program,
        &[],
        &[
            QuantizedBlock::Q2K(&packed),
            QuantizedBlock::Float32(&activation),
        ],
        &[packed_root],
        NumericPolicy::default(),
    )
    .expect("Metal executes the Q2_K matmul");

    let f32_program = matmul_program(ROWS as u32, WIDTH as u32, DType::Float32);
    let f32_root = proxima_tensor::NodeId((f32_program.len() - 1) as u32);
    let cpu = evaluate(&f32_program, &[], &[&dequantized, &activation], &[f32_root])
        .expect("CPU evaluates the dequantized reference");
    let actual = metal.root();
    let expected = cpu.root();
    let max_abs_diff = actual
        .iter()
        .zip(expected)
        .map(|(metal_value, cpu_value)| (metal_value - cpu_value).abs())
        .fold(0.0f32, f32::max);
    eprintln!("q2k_device_parity rows={ROWS} width={WIDTH} max_abs_diff={max_abs_diff:e}");
    assert!(max_abs_diff <= 1e-5, "max_abs_diff={max_abs_diff:e}");
}
