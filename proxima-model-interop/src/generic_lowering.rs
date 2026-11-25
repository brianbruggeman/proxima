//! The model-free lowerings of `proxima_tensor::spec::generic_lowering`, run on
//! the host evaluator against naive references on shapes no diffusion model uses.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};

use proxima_numeric::float_layout::NAMED_LAYOUTS;
use proxima_numeric::{DType, Philox};
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::op::{Extent, NodeId, Op};
use proxima_tensor::spec::{
    RopePairing, fused_rope_pair, input_leaf, layer_norm_no_affine, modulate,
    multi_axis_rope_tables, non_causal_attention_heads, scalar_constant, tap_layers,
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
    let out = non_causal_attention_heads(&mut program, &[(q, k)], v, inv_sqrt, "u", "i", "d")
        .expect("attention builds");
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

fn index_dtype() -> DType {
    NAMED_LAYOUTS
        .iter()
        .map(|(layout, _)| DType::Layout(*layout))
        .find(|dtype| {
            dtype.is_signed_integer() && dtype.size_bytes() == core::mem::size_of::<i32>()
        })
        .expect("a signed integer layout holds an index")
}

fn integer_leaf(program: &mut Vec<Op>, name: &str, extent: u32) -> NodeId {
    input_leaf(
        program,
        index_dtype(),
        alloc::vec![Extent::Static(extent)],
        name,
    )
}

#[test]
fn project_rows_matches_the_row_major_outer_product_on_a_non_model_shape() {
    let (tokens, into, out) = (3usize, 5usize, 4usize);
    let input = Philox::new(31).normal(tokens * into);
    let weight = Philox::new(17).normal(out * into);
    let bias = Philox::new(10).normal(out);
    let mut program = Vec::new();
    let input_node = leaf(&mut program, "x", &[tokens as u32, into as u32]);
    let weight_node = leaf(&mut program, "weight", &[out as u32, into as u32]);
    let bias_node = leaf(&mut program, "bias", &[out as u32]);
    let out_node =
        proxima_tensor::spec::project_rows(&mut program, input_node, weight_node, Some(bias_node))
            .expect("linear builds");
    let actual = run(
        &program,
        &[("x", &input), ("weight", &weight), ("bias", &bias)],
        out_node,
    );
    let mut expected = Vec::new();
    for token in 0..tokens {
        for column in 0..out {
            let dot = (0..into).fold(0.0, |total, inner| {
                total + input[token * into + inner] * weight[column * into + inner]
            });
            expected.push(dot + bias[column]);
        }
    }
    assert_close(&actual, &expected);
}

#[test]
fn gelu_tanh_matches_its_closed_form() {
    let values = Philox::new(5).normal(6);
    let mut program = Vec::new();
    let input_node = leaf(&mut program, "x", &[values.len() as u32]);
    let out =
        proxima_tensor::spec::gelu_tanh(&mut program, input_node, "s->s").expect("gelu builds");
    let actual = run(&program, &[("x", &values)], out);
    let unit = values[0].signum().abs();
    let scale = (2.0 / (-unit).acos()).sqrt();
    let expected: Vec<_> = values
        .iter()
        .map(|value| {
            let inner = scale * (value + 0.044_715 * value.powi(3));
            0.5 * value * (1.0 + inner.tanh())
        })
        .collect();
    assert_close(&actual, &expected);
}

#[test]
fn group_norm_matches_the_naive_per_group_mean_and_variance() {
    let (rows, groups, group_width) = (4u32, 2u32, 2u32);
    let channels = groups * group_width;
    let source = Philox::new(23).normal((rows * channels) as usize);
    let weight = Philox::new(25).normal(channels as usize);
    let bias = Philox::new(12).normal(channels as usize);
    let mut program = Vec::new();
    let input_node = leaf(&mut program, "x", &[rows, groups, group_width]);
    let weight_node = leaf(&mut program, "weight", &[groups, group_width]);
    let bias_node = leaf(&mut program, "bias", &[groups, group_width]);
    let eps = proxima_tensor::spec::scalar_constant_like(&mut program, 1e-6, input_node)
        .expect("eps builds");
    let out = proxima_tensor::spec::group_norm(
        &mut program,
        input_node,
        weight_node,
        bias_node,
        rows * group_width,
        eps,
    )
    .expect("group norm builds");
    let actual = run(
        &program,
        &[("x", &source), ("weight", &weight), ("bias", &bias)],
        out,
    );
    let mut expected = vec![0.0; source.len()];
    for group in 0..groups as usize {
        let mut values = Vec::new();
        for row in 0..rows as usize {
            for within in 0..group_width as usize {
                values.push(source[row * channels as usize + group * group_width as usize + within]);
            }
        }
        let count = values.iter().fold(0.0, |total, _| total + 1.0);
        let mean = values.iter().fold(0.0, |total, &value| total + value) / count;
        let variance = values
            .iter()
            .fold(0.0, |total, &value| total + (value - mean).powi(2))
            / count;
        let deviation = (variance + 1e-6).sqrt();
        for (index, value) in values.iter().enumerate() {
            let row = index / group_width as usize;
            let within = index % group_width as usize;
            let channel = group * group_width as usize + within;
            expected[row * channels as usize + channel] =
                (value - mean) / deviation * weight[channel] + bias[channel];
        }
    }
    assert_close(&actual, &expected);
}

#[test]
fn nearest_upsample_x2_repeats_each_source_cell_into_a_two_by_two_block() {
    let (rows, columns, group_width) = (2u32, 3u32, 2u32);
    let source = Philox::new(7).normal((rows * columns * group_width) as usize);
    let mut program = Vec::new();
    let input_node = leaf(&mut program, "x", &[rows * columns, 1, group_width]);
    let index = integer_leaf(&mut program, "index", 4 * rows * columns);
    let out = proxima_tensor::spec::nearest_upsample_x2(&mut program, input_node, index)
        .expect("upsample builds");
    let encoded: Vec<_> = proxima_tensor::spec::upsample_x2_index(rows, columns)
        .iter()
        .map(|value| {
            i16::try_from(*value)
                .expect("an upsample index is small")
                .into()
        })
        .collect();
    let actual = run(
        &program,
        &[("x", &source), ("index", encoded.as_slice())],
        out,
    );
    let mut expected = Vec::new();
    for row in 0..2 * rows {
        for column in 0..2 * columns {
            for within in 0..group_width {
                let source_cell = ((row / 2) * columns + (column / 2)) * group_width + within;
                expected.push(source[source_cell as usize]);
            }
        }
    }
    assert_close(&actual, &expected);
}

#[test]
fn window_same_matches_a_direct_loop_on_a_two_by_two_image() {
    let (rows, columns, groups, in_width, out_width) = (2u32, 2u32, 2u32, 2u32, 3u32);
    let source = Philox::new(41).normal((rows * columns * groups * in_width) as usize);
    let weight = Philox::new(19).normal((groups * out_width * 9 * groups * in_width) as usize);
    let mut program = Vec::new();
    let input_node = leaf(&mut program, "x", &[rows * columns, groups, in_width]);
    let mut taps = [proxima_tensor::op::NodeId(0); 9];
    for tap_row in 0..3u32 {
        for tap_column in 0..3u32 {
            taps[(tap_row * 3 + tap_column) as usize] = leaf(
                &mut program,
                &format!("tap{tap_row}{tap_column}"),
                &[groups, out_width, groups, in_width],
            );
        }
    }
    let pad_rows = proxima_tensor::spec::window_pad_rows(rows, columns, 3);
    let pad_index = integer_leaf(&mut program, "pad_index", pad_rows);
    let pad_mask = leaf(&mut program, "pad_mask", &[pad_rows]);
    let crop_index = integer_leaf(&mut program, "crop_index", rows * columns);
    let out = proxima_tensor::spec::window_same(
        &mut program,
        input_node,
        &taps,
        pad_index,
        pad_mask,
        crop_index,
        rows,
        columns,
        3,
    )
    .expect("conv builds");
    let pad: Vec<_> = proxima_tensor::spec::window_pad_index(rows, columns, 3)
        .iter()
        .map(|value| {
            i16::try_from(*value)
                .expect("a window index is small")
                .into()
        })
        .collect();
    let mask: Vec<_> = proxima_tensor::spec::window_pad_mask(rows, columns, 3)
        .iter()
        .map(|value| i16::try_from(*value).expect("a mask entry is small").into())
        .collect();
    let crop: Vec<_> = proxima_tensor::spec::window_crop_index(rows, columns, 3)
        .iter()
        .map(|value| {
            i16::try_from(*value)
                .expect("a window index is small")
                .into()
        })
        .collect();
    let mut named = vec![
        ("x", source.as_slice()),
        ("pad_index", pad.as_slice()),
        ("pad_mask", mask.as_slice()),
        ("crop_index", crop.as_slice()),
    ];
    let names: Vec<String> = (0..9).map(|tap| format!("tap{}{}", tap / 3, tap % 3)).collect();
    for (tap, name) in names.iter().enumerate() {
        let row = tap / 3;
        let column = tap % 3;
        let tap_values: Vec<_> = (0..(groups * out_width * groups * in_width) as usize)
            .map(|index| {
                let out_group = index / (out_width * groups * in_width) as usize;
                let rest = index % (out_width * groups * in_width) as usize;
                let out_within = rest / (groups * in_width) as usize;
                let inner = rest % (groups * in_width) as usize;
                let in_group = inner / in_width as usize;
                let in_within = inner % in_width as usize;
                let kernel = ((((out_group as u32 * out_width + out_within as u32) * 3
                    + row as u32)
                    * 3
                    + column as u32)
                    * groups
                    + in_group as u32)
                    * in_width
                    + in_within as u32;
                weight[kernel as usize]
            })
            .collect();
        named.push((name.as_str(), Box::leak(tap_values.into_boxed_slice())));
    }
    let actual = run(&program, &named, out);
    let mut expected = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            for out_group in 0..groups {
                for out_within in 0..out_width {
                    let mut total = 0.0;
                    for tap_row in 0..3i32 {
                        for tap_column in 0..3i32 {
                            let source_row = row as i32 + tap_row - 1;
                            let source_column = column as i32 + tap_column - 1;
                            if source_row < 0
                                || source_column < 0
                                || source_row >= rows as i32
                                || source_column >= columns as i32
                            {
                                continue;
                            }
                            for in_group in 0..groups {
                                for in_within in 0..in_width {
                                    let pixel = (source_row as u32 * columns
                                        + source_column as u32)
                                        * groups
                                        * in_width
                                        + in_group * in_width
                                        + in_within;
                                    let kernel = ((((out_group * out_width + out_within) * 3
                                        + tap_row as u32)
                                        * 3
                                        + tap_column as u32)
                                        * groups
                                        + in_group)
                                        * in_width
                                        + in_within;
                                    total += source[pixel as usize] * weight[kernel as usize];
                                }
                            }
                        }
                    }
                    expected.push(total);
                }
            }
        }
    }
    assert_close(&actual, &expected);
}
