//! The model-free lowerings of `proxima_tensor::spec::generic_lowering`, run on
//! the host evaluator against naive references on shapes no diffusion model uses.

use alloc::vec::Vec;

use proxima_numeric::DType;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::op::{Extent, NodeId, Op};
use proxima_tensor::spec::{
    RopePairing, fused_rope_pair, input_leaf, layer_norm_no_affine, modulate,
    multi_axis_rope_tables, non_causal_attention, scalar_constant, tap_layers,
};

const F32_TOLERANCE: f32 = 1e-5;

fn leaf(program: &mut Vec<Op>, name: &str, shape: &[u32]) -> NodeId {
    let extents = shape.iter().map(|&size| Extent::Static(size)).collect();
    input_leaf(program, DType::FLOAT32, extents, name)
}

fn run(program: &[Op], named: &[(&str, &[f32])], output: NodeId) -> Vec<f32> {
    let evaluated = evaluate_named(program, &[], named, &[output]).expect("program evaluates");
    let (data, _) = evaluated.get(output).expect("requested output is present");
    data.to_vec()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "output length");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= F32_TOLERANCE * want.abs().max(1.0),
            "element {index}: got {got}, want {want}"
        );
    }
}

fn naive_layer_norm(row: &[f32], eps: f32) -> Vec<f32> {
    let width = row.len() as f32;
    let mean = row.iter().sum::<f32>() / width;
    let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / width;
    row.iter()
        .map(|value| (value - mean) / (variance + eps).sqrt())
        .collect()
}

#[test]
fn layer_norm_no_affine_then_modulate_matches_the_naive_row_formula() {
    let (rows, width, eps) = (3usize, 5usize, 1e-6f32);
    let x_data: Vec<f32> = (0..rows * width)
        .map(|index| (index as f32 * 0.7).sin() * 3.0 + 0.5)
        .collect();
    let shift_data = [0.1f32, -0.2, 0.3, 0.0, 0.4];
    let scale_data = [0.5f32, -0.25, 0.0, 1.0, 0.125];
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", &[rows as u32, width as u32]);
    let shift = leaf(&mut program, "shift", &[width as u32]);
    let scale = leaf(&mut program, "scale", &[width as u32]);
    let normed = layer_norm_no_affine(&mut program, x, width as u32, eps).expect("norm builds");
    let out = modulate(&mut program, normed, shift, scale).expect("modulate builds");
    let actual = run(
        &program,
        &[
            ("x", &x_data),
            ("shift", &shift_data),
            ("scale", &scale_data),
        ],
        out,
    );
    let expected: Vec<f32> = x_data
        .chunks(width)
        .flat_map(|row| {
            naive_layer_norm(row, eps)
                .into_iter()
                .enumerate()
                .map(|(column, value)| value * (1.0 + scale_data[column]) + shift_data[column])
        })
        .collect();
    assert_close(&actual, &expected);
}

#[test]
fn non_causal_attention_lets_every_query_see_every_key() {
    let (queries, keys, heads, head_dim, value_dim) = (3usize, 4usize, 2usize, 2usize, 3usize);
    let fill = |count: usize, phase: f32| -> Vec<f32> {
        (0..count)
            .map(|index| (index as f32 * 0.9 + phase).cos())
            .collect()
    };
    let (q_data, k_data, v_data) = (
        fill(queries * heads * head_dim, 0.1),
        fill(keys * heads * head_dim, 0.7),
        fill(keys * heads * value_dim, 1.3),
    );
    let mut program = Vec::new();
    let q = leaf(
        &mut program,
        "q",
        &[queries as u32, heads as u32, head_dim as u32],
    );
    let k = leaf(
        &mut program,
        "k",
        &[keys as u32, heads as u32, head_dim as u32],
    );
    let v = leaf(
        &mut program,
        "v",
        &[keys as u32, heads as u32, value_dim as u32],
    );
    let inv_sqrt = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    let out = non_causal_attention(&mut program, q, k, v, inv_sqrt).expect("attention builds");
    let actual = run(
        &program,
        &[("q", &q_data), ("k", &k_data), ("v", &v_data)],
        out,
    );
    let mut expected = Vec::new();
    for query in 0..queries {
        for head in 0..heads {
            let score = |key: usize| -> f32 {
                (0..head_dim)
                    .map(|axis| {
                        q_data[(query * heads + head) * head_dim + axis]
                            * k_data[(key * heads + head) * head_dim + axis]
                    })
                    .sum::<f32>()
                    / (head_dim as f32).sqrt()
            };
            let weights: Vec<f32> = (0..keys).map(|key| score(key).exp()).collect();
            let total: f32 = weights.iter().sum();
            for column in 0..value_dim {
                expected.push(
                    (0..keys)
                        .map(|key| {
                            weights[key] / total * v_data[(key * heads + head) * value_dim + column]
                        })
                        .sum::<f32>(),
                );
            }
        }
    }
    assert_close(&actual, &expected);
}

#[test]
fn multi_axis_rope_rotates_each_section_by_its_own_position_stream() {
    let (rows, heads, sections, thetas) = (3usize, 2usize, [2u32, 4], [100.0f32, 50.0]);
    let head_dim: usize = sections.iter().sum::<u32>() as usize;
    let pairs = head_dim / 2;
    let positions = [0.0f32, 2.0, 1.0, 5.0, 3.0, 1.0];
    let (cos_bytes, sin_bytes) =
        multi_axis_rope_tables(&positions, &sections, &thetas, DType::FLOAT32)
            .expect("tables build");
    let decode = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    };
    let (cos, sin) = (decode(&cos_bytes), decode(&sin_bytes));
    let x_data: Vec<f32> = (0..rows * heads * head_dim)
        .map(|index| (index as f32 * 0.37).sin())
        .collect();
    let mut program = Vec::new();
    let x = leaf(
        &mut program,
        "x",
        &[rows as u32, heads as u32, head_dim as u32],
    );
    let cos_leaf = leaf(&mut program, "cos", &[rows as u32, pairs as u32]);
    let sin_leaf = leaf(&mut program, "sin", &[rows as u32, pairs as u32]);
    let (even, odd) = fused_rope_pair(
        &mut program,
        x,
        'h',
        cos_leaf,
        sin_leaf,
        RopePairing::Interleaved,
    )
    .expect("rope builds");
    let named: [(&str, &[f32]); 3] = [("x", &x_data), ("cos", &cos), ("sin", &sin)];
    let (got_even, got_odd) = (run(&program, &named, even), run(&program, &named, odd));
    let (mut want_even, mut want_odd) = (Vec::new(), Vec::new());
    for row in 0..rows {
        for head in 0..heads {
            let (mut pair, mut start) = (0usize, 0usize);
            for (axis, &section) in sections.iter().enumerate() {
                for local in 0..section as usize / 2 {
                    let angle = positions[row * 2 + axis]
                        * thetas[axis].powf(-((2 * local) as f32) / section as f32);
                    let base = (row * heads + head) * head_dim + start + 2 * local;
                    want_even.push(x_data[base] * angle.cos() - x_data[base + 1] * angle.sin());
                    want_odd.push(x_data[base + 1] * angle.cos() + x_data[base] * angle.sin());
                    pair += 1;
                }
                start += section as usize;
            }
            assert_eq!(pair, pairs);
        }
    }
    assert_close(&got_even, &want_even);
    assert_close(&got_odd, &want_odd);
}

#[test]
fn tap_layers_stacks_the_chosen_layer_outputs_in_tap_order() {
    let (rows, width) = (2usize, 3usize);
    let layers: Vec<Vec<f32>> = (0..4)
        .map(|layer| {
            (0..rows * width)
                .map(|index| (layer * 10 + index) as f32 + 0.25)
                .collect()
        })
        .collect();
    let mut program = Vec::new();
    let names = ["l0", "l1", "l2", "l3"];
    let nodes: Vec<NodeId> = names
        .iter()
        .map(|name| leaf(&mut program, name, &[rows as u32, width as u32]))
        .collect();
    let stacked = tap_layers(&mut program, &nodes, &[2, 0, 3]).expect("taps build");
    let named: Vec<(&str, &[f32])> = names
        .iter()
        .zip(&layers)
        .map(|(name, data)| (*name, data.as_slice()))
        .collect();
    let actual = run(&program, &named, stacked);
    let mut expected = Vec::new();
    for row in 0..rows {
        for tap in [2usize, 0, 3] {
            expected.extend_from_slice(&layers[tap][row * width..(row + 1) * width]);
        }
    }
    assert_close(&actual, &expected);
}
