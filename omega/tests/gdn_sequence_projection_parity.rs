#![cfg(all(feature = "metal", target_os = "macos"))]

use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};
use proxima_tensor::spec::{
    Qwen35GdnSequenceTail, append_qwen35_gdn_sequence_tail_with_taps, input_leaf, scalar_constant,
};
use proxima_tensor::{DType, Extent, NumericPolicy, Op, QuantizedBlock};

#[test]
fn qwen35_sequence_tail_projection_matches_cpu_on_packed_weight() {
    const POSITIONS: u32 = 2;
    const HEAD_DIM: u32 = 128;
    const KV_HEADS: u32 = 16;
    const GROUP: u32 = 1;
    const EMBED: u32 = 4096;
    const CONTRACTION: usize = (HEAD_DIM * KV_HEADS * GROUP) as usize;
    assert_eq!(CONTRACTION % QK_K, 0);

    let mut program: Vec<Op> = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(POSITIONS), Extent::Static(EMBED)],
        "x",
    );
    let delta = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(POSITIONS),
            Extent::Static(HEAD_DIM),
            Extent::Static(KV_HEADS),
            Extent::Static(GROUP),
        ],
        "delta",
    );
    let z = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(POSITIONS),
            Extent::Static(KV_HEADS),
            Extent::Static(GROUP),
            Extent::Static(HEAD_DIM),
        ],
        "z",
    );
    let head_eps = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(KV_HEADS), Extent::Static(GROUP)],
        "head_eps",
    );
    let inv_head_v_dim = scalar_constant(&mut program, 1.0 / HEAD_DIM as f32);
    let norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(HEAD_DIM)],
        "norm_weight",
    );
    let out_weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(CONTRACTION as u32), Extent::Static(EMBED)],
        "out_weight",
    );
    let taps = append_qwen35_gdn_sequence_tail_with_taps(
        &mut program,
        Qwen35GdnSequenceTail {
            x,
            delta_out: delta,
            z,
            head_eps,
            inv_head_v_dim,
            norm_weight,
            out_weight,
            head_v_dim: HEAD_DIM,
            kv_heads: KV_HEADS,
            group: GROUP,
        },
    )
    .expect("sequence tail lowers");

    let x_values = vec![0.0_f32; (POSITIONS * EMBED) as usize];
    let delta_values: Vec<f32> = (0..POSITIONS as usize * CONTRACTION)
        .map(|index| ((index * 37 % 211) as f32 - 105.0) / 53.0)
        .collect();
    let z_values: Vec<f32> = (0..POSITIONS as usize * CONTRACTION)
        .map(|index| ((index * 19 % 101) as f32 - 50.0) / 47.0)
        .collect();
    let head_eps_values = vec![1.0e-6_f32; (KV_HEADS * GROUP) as usize];
    let norm_values: Vec<f32> = (0..HEAD_DIM)
        .map(|index| 0.5 + (index % 11) as f32 / 17.0)
        .collect();
    let weight_values: Vec<f32> = (0..EMBED as usize * CONTRACTION)
        .map(|index| ((index * 23 % 181) as f32 - 90.0) / 61.0)
        .collect();
    let packed_row_bytes = CONTRACTION / QK_K * BLOCK_BYTES;
    let mut weight_blocks = vec![0_u8; EMBED as usize * packed_row_bytes];
    for (row, packed) in weight_values
        .chunks_exact(CONTRACTION)
        .zip(weight_blocks.chunks_exact_mut(packed_row_bytes))
    {
        quantize(row, packed).expect("each output row is one q4_k block");
    }
    let blocks = [
        QuantizedBlock::Float32(x_values.as_slice()),
        QuantizedBlock::Float32(delta_values.as_slice()),
        QuantizedBlock::Float32(z_values.as_slice()),
        QuantizedBlock::Float32(head_eps_values.as_slice()),
        QuantizedBlock::Float32(norm_values.as_slice()),
        QuantizedBlock::Q4K(weight_blocks.as_slice()),
    ];

    let cpu =
        proxima_tensor::cpu::evaluate_quantized_exact(&program, &[], &blocks, &[taps.projected])
            .expect("cpu evaluates the packed sequence projection");
    let metal = omega::execute(
        &program,
        &[],
        &blocks,
        &[taps.projected],
        NumericPolicy::default(),
    )
    .expect("metal evaluates the packed sequence projection");

    let expected = cpu
        .get(taps.projected)
        .expect("cpu retains the requested projection")
        .0;
    let actual = metal
        .get(taps.projected)
        .expect("metal retains the requested projection")
        .0;
    assert!(
        !actual.is_empty(),
        "projection comparison must not be vacuous"
    );
    assert_eq!(actual.len(), expected.len());
    let max_abs_diff = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "qwen35 sequence projection max_abs_diff={max_abs_diff} cpu={expected:?} metal={actual:?}"
    );
    assert!(
        max_abs_diff < 1.0e-3,
        "packed sequence projection diverged by {max_abs_diff}"
    );
}
