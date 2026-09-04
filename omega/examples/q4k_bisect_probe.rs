//! Quick bisection: at what OUT_DIM does
//! `q4k_matmul_layout.rs`'s proven-correct 2D shape start disagreeing
//! between CPU and Metal? Throwaway diagnostic, not part of the discipline
//! gate -- exists only to hand the other agent's `fix/metal-cpu-
//! disagreement` branch exact repro data (task brief: "report the exact
//! shape, node, and payload values; that is data the other agent needs").

#![allow(clippy::unwrap_used, clippy::expect_used)]

fn main() {
    #[cfg(all(feature = "metal", feature = "cpu", target_os = "macos"))]
    run();
    #[cfg(not(all(feature = "metal", feature = "cpu", target_os = "macos")))]
    println!("q4k_bisect_probe requires --features metal,cpu on macOS");
}

#[cfg(all(feature = "metal", feature = "cpu", target_os = "macos"))]
fn run() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, dequantize, quantize};
    use proxima_tensor::cpu::evaluate_quantized;
    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp,
        append, projection,
    };

    fn random_vec(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    fn matmul_program(in_dim: u32, out_dim: u32) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::UInt8,
                shape: vec![Extent::Static(in_dim), Extent::Static(out_dim)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(in_dim)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(projection(2, &[1, 0]))),
                    (activation, IndexMap::Affine(projection(2, &[1]))),
                ],
                name: None,
            },
        );
        let sum = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(projection(2, &[0, 1])),
                out_map: IndexMap::Affine(projection(2, &[0])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    fn pack_rows(rows: &[Vec<f32>], in_dim: usize) -> Vec<u8> {
        let blocks_per_row = in_dim / QK_K;
        let mut packed = vec![0u8; rows.len() * blocks_per_row * BLOCK_BYTES];
        for (row, row_packed) in rows
            .iter()
            .zip(packed.chunks_exact_mut(blocks_per_row * BLOCK_BYTES))
        {
            quantize(row, row_packed).expect("in_dim is a whole multiple of QK_K");
        }
        packed
    }

    fn expected_output(packed: &[u8], in_dim: usize, out_dim: usize, activation: &[f32]) -> Vec<f32> {
        let blocks_per_row = in_dim / QK_K;
        let mut expected = Vec::with_capacity(out_dim);
        for row_packed in packed.chunks_exact(blocks_per_row * BLOCK_BYTES) {
            let mut row = vec![0.0f32; in_dim];
            dequantize(row_packed, &mut row).expect("packed row dequantizes");
            let dot: f32 = row.iter().zip(activation.iter()).map(|(w, a)| w * a).sum();
            expected.push(dot);
        }
        expected
    }

    const IN_DIM: usize = 512;
    for out_dim in [3usize, 4, 8, 32, 64, 128, 512, 1024, 4096] {
        let rows: Vec<Vec<f32>> = (0..out_dim)
            .map(|row| random_vec(17 + row as u64, IN_DIM))
            .collect();
        let packed = pack_rows(&rows, IN_DIM);
        let activation = random_vec(97, IN_DIM);
        let expected = expected_output(&packed, IN_DIM, out_dim, &activation);

        let (program, sum) = matmul_program(IN_DIM as u32, out_dim as u32);
        let blocks = [
            QuantizedBlock::Q4K(&packed),
            QuantizedBlock::Float32(&activation),
        ];
        let cpu = evaluate_quantized(&program, &[], &blocks, &[sum]).expect("cpu runs");
        let plan = omega::plan(&program, &[], &blocks, &[sum]).expect("metal plans");
        let metal = omega::execute_plan(&plan, &blocks).expect("metal runs");

        let cpu_root = cpu.root();
        let metal_root = metal.root();
        let mut max_cpu_rel = 0.0f32;
        let mut max_metal_rel = 0.0f32;
        let mut first_bad_row: Option<usize> = None;
        for (index, ((&cpu_value, &metal_value), &reference)) in cpu_root
            .iter()
            .zip(metal_root.iter())
            .zip(expected.iter())
            .enumerate()
        {
            let scale = reference.abs().max(f32::MIN_POSITIVE);
            let cpu_rel = (cpu_value - reference).abs() / scale;
            let metal_rel = (metal_value - reference).abs() / scale;
            if metal_rel > 1e-2 && first_bad_row.is_none() {
                first_bad_row = Some(index);
                println!(
                    "    first bad row {index}: cpu={cpu_value} metal={metal_value} reference={reference}"
                );
            }
            max_cpu_rel = max_cpu_rel.max(cpu_rel);
            max_metal_rel = max_metal_rel.max(metal_rel);
        }
        println!(
            "out_dim={out_dim:5} in_dim={IN_DIM}: max_cpu_rel={max_cpu_rel:e} max_metal_rel={max_metal_rel:e} \
             groups={} first_bad_row={first_bad_row:?}",
            out_dim.div_ceil(4)
        );
    }
}
