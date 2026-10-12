use std::io;

use omega::config::RuntimeConfig;
use proxima_primitives::Codec;
use proxima_tensor::spec::{RopePairing, fused_rope_pair, input_leaf};
use proxima_tensor::{DType, Extent, QuantizedBlock};
use serde_json::{Value, json};

use super::engine::Executor;
use super::graph::{self, Graph, Shape};
use super::guidance;

pub fn mismatch(observed: &[f32], expected: &[f32], tolerance: f32) -> usize {
    observed
        .iter()
        .zip(expected)
        .filter(|(observed, expected)| {
            (*observed - *expected).abs() > tolerance || !observed.is_finite()
        })
        .count()
        + observed.len().abs_diff(expected.len())
}

fn normalized(values: &[f32]) -> Vec<f32> {
    let rms = (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32).sqrt();
    values.iter().map(|value| value / rms).collect()
}

struct Miniature<'weights> {
    epsilon: &'weights [f32],
    input_scale: &'weights [f32],
    output_scale: &'weights [f32],
    zero: &'weights [f32],
    identity: &'weights [f32],
    gate: &'weights [f32],
    cosine: &'weights [f32],
    sine: &'weights [f32],
}

impl Miniature<'_> {
    fn layer<'call>(
        &'call self,
        values: &'call [f32],
        post_attention: &'call [f32],
    ) -> Vec<(&'static str, QuantizedBlock<'call>)> {
        vec![
            ("hidden", QuantizedBlock::Float32(values)),
            ("epsilon", QuantizedBlock::Float32(self.epsilon)),
            ("input_norm", QuantizedBlock::Float32(self.input_scale)),
            ("attention_norm", QuantizedBlock::Float32(post_attention)),
            ("ffn_norm", QuantizedBlock::Float32(self.input_scale)),
            ("output_norm", QuantizedBlock::Float32(self.output_scale)),
            ("query", QuantizedBlock::Float32(self.zero)),
            ("key", QuantizedBlock::Float32(self.zero)),
            ("value", QuantizedBlock::Float32(self.identity)),
            ("attention", QuantizedBlock::Float32(self.identity)),
            ("gate", QuantizedBlock::Float32(self.gate)),
            ("up", QuantizedBlock::Float32(self.gate)),
            ("down", QuantizedBlock::Float32(self.identity)),
            ("cosine", QuantizedBlock::Float32(self.cosine)),
            ("sine", QuantizedBlock::Float32(self.sine)),
        ]
    }

    fn norm<'call>(
        &'call self,
        values: &'call [f32],
    ) -> Vec<(&'static str, QuantizedBlock<'call>)> {
        vec![
            ("hidden", QuantizedBlock::Float32(values)),
            ("scale", QuantizedBlock::Float32(self.input_scale)),
            ("epsilon", QuantizedBlock::Float32(self.epsilon)),
        ]
    }
}

pub fn run(backend_name: &str) -> io::Result<Value> {
    run_with_config(backend_name, None)
}

pub fn run_with_config(backend_name: &str, config: Option<&RuntimeConfig>) -> io::Result<Value> {
    let shape = Shape {
        hidden: 2,
        intermediate: 2,
        heads: 1,
        head_width: 2,
        vocab: 2,
    };
    let identity = [1.0, 0.0, 0.0, 1.0];
    let zero = [0.0; 4];
    let gate = [1.0, 0.0, 0.0, 0.0];
    let input_scale = [1.0; 2];
    let attention_scale = [2.0; 2];
    let output_scale = [1.0 / 2.0f32.sqrt(); 2];
    let epsilon = [0.0];
    let cosine = [1.0];
    let sine = [0.0];
    let mut hidden = vec![3.0, 4.0];
    let weights = Miniature {
        epsilon: &epsilon,
        input_scale: &input_scale,
        output_scale: &output_scale,
        zero: &zero,
        identity: &identity,
        gate: &gate,
        cosine: &cosine,
        sine: &sine,
    };
    let mut layer = Executor::prepare_with_config(
        graph::layer(shape)?,
        backend_name,
        1,
        &weights.layer(&hidden, &attention_scale),
        false,
        config,
    )?;
    let initial = hidden.clone();
    let mut normalization = Executor::prepare_with_config(
        graph::normalization(2)?,
        backend_name,
        1,
        &weights.norm(&hidden),
        false,
        config,
    )?;
    let mut observed = Vec::new();
    let mut first_raw = Vec::new();
    let mut layer_rms_sites = 0;
    let mut loop_rms_sites = 0;
    for iteration in 0..4 {
        hidden = layer
            .run(&weights.layer(&hidden, &attention_scale), false)?
            .0;
        layer_rms_sites += layer.graph.norm_sites();
        if iteration == 0 {
            first_raw = hidden.clone();
        }
        hidden = normalization.run(&weights.norm(&hidden), false)?.0;
        loop_rms_sites += normalization.graph.norm_sites();
        observed.push(hidden.clone());
    }
    let first_literal = [
        4.0 + 6.0 * 2.0f32.sqrt() / 5.0,
        4.0 + 8.0 * 2.0f32.sqrt() / 5.0,
    ];
    let mut expected = vec![normalized(&first_literal)];
    for iteration in 1..4 {
        let previous = &expected[iteration - 1];
        expected.push(normalized(&[3.0 * previous[0] + 1.0, 3.0 * previous[1]]));
    }
    let state_mismatches: usize = observed
        .iter()
        .zip(&expected)
        .map(|(observed, expected)| mismatch(observed, expected, 0.00001))
        .sum();
    let wrong_raw = layer
        .run(&weights.layer(&first_raw, &attention_scale), false)?
        .0;
    let wrong_feed = normalization.run(&weights.norm(&wrong_raw), false)?.0;
    let wrong_feed_mismatches = mismatch(&wrong_feed, &expected[1], 0.00001);
    let wrong_branch_raw = layer.run(&weights.layer(&initial, &input_scale), false)?.0;
    let wrong_branch = normalization
        .run(&weights.norm(&wrong_branch_raw), false)?
        .0;
    let wrong_branch_mismatches = mismatch(&wrong_branch, &expected[0], 0.00001);
    if state_mismatches != 0 || wrong_feed_mismatches == 0 || wrong_branch_mismatches == 0 {
        return Err(io::Error::other(format!(
            "miniature control counts differ: {}",
            json!({"state_mismatches":state_mismatches,"observed":observed,"expected":expected,"wrong_feed":wrong_feed,"wrong_feed_mismatches":wrong_feed_mismatches,"wrong_branch":wrong_branch,"wrong_branch_mismatches":wrong_branch_mismatches})
        )));
    }

    let mut program = Vec::new();
    let activation = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(1), Extent::Static(2)],
        "activation",
    );
    let weight = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(2), Extent::Static(3)],
        "weight",
    );
    let root = graph::projection(&mut program, activation, weight)?;
    let mut bytes = Vec::new();
    for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    let activation_values = [1.0, 2.0];
    let bindings = [
        ("activation", QuantizedBlock::Float32(&activation_values)),
        (
            "weight",
            QuantizedBlock::Packed {
                codec: Codec::BFloat16,
                bytes: &bytes,
            },
        ),
    ];
    let mut projection = Executor::prepare_with_config(
        Graph {
            program,
            root,
            taps: Vec::new(),
        },
        backend_name,
        1,
        &bindings,
        false,
        config,
    )?;
    let projected = projection.run(&bindings, false)?.0;
    let projection_mismatches = mismatch(&projected, &[5.0, 11.0, 17.0], 0.0);
    let mut transposed_bytes = Vec::new();
    for value in [1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0] {
        transposed_bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    let wrong_bindings = [
        ("activation", QuantizedBlock::Float32(&activation_values)),
        (
            "weight",
            QuantizedBlock::Packed {
                codec: Codec::BFloat16,
                bytes: &transposed_bytes,
            },
        ),
    ];
    let transposed = projection.run(&wrong_bindings, false)?.0;
    let transposed_mismatches = mismatch(&transposed, &[5.0, 11.0, 17.0], 0.0);

    let mut program = Vec::new();
    let source = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(1), Extent::Static(1), Extent::Static(4)],
        "source",
    );
    let cosine_node = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(1), Extent::Static(2)],
        "cosine",
    );
    let sine_node = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(1), Extent::Static(2)],
        "sine",
    );
    let (first, second) = fused_rope_pair(
        &mut program,
        source,
        'h',
        cosine_node,
        sine_node,
        RopePairing::SplitHalf { pairs: 2 },
    )
    .map_err(io::Error::other)?;
    let rope_bindings = [
        ("source", QuantizedBlock::Float32(&[1.0, 2.0, 3.0, 4.0])),
        ("cosine", QuantizedBlock::Float32(&[0.0, 1.0])),
        ("sine", QuantizedBlock::Float32(&[1.0, 0.0])),
    ];
    let mut rope = Executor::prepare_with_config(
        Graph {
            program,
            root: first,
            taps: vec![("second", second)],
        },
        backend_name,
        1,
        &rope_bindings,
        true,
        config,
    )?;
    let (mut rotated, taps) = rope.run(&rope_bindings, true)?;
    rotated.extend_from_slice(&taps[0].1);
    let rope_mismatches = mismatch(&rotated, &[-3.0, 2.0, 1.0, 4.0], 0.0);
    if projection_mismatches != 0 || transposed_mismatches == 0 || rope_mismatches != 0 {
        return Err(io::Error::other(format!(
            "projection or RoPE literal control failed: {}",
            json!({"projection":projected,"projection_expected":[5.0,11.0,17.0],"projection_mismatches":projection_mismatches,"transposed_projection":transposed,"transposed_mismatches":transposed_mismatches,"rope":rotated,"rope_expected":[-3.0,2.0,1.0,4.0],"rope_mismatches":rope_mismatches})
        )));
    }
    Ok(
        json!({"kind":"controls","backend":backend_name,"layer_program_node_56":format!("{:?}",layer.graph.program.get(56)),"layer_program_nodes":layer.graph.program.len(),"miniature_loop_states":observed.len(),"compared_state_elements":observed.iter().map(Vec::len).sum::<usize>(),"layer_rms_sites":layer_rms_sites,"loop_rms_sites":loop_rms_sites,"absolute_tolerance":0.00001,"state_mismatches":state_mismatches,"observed_states":observed,"expected_states":expected,"unnormalized_next_loop":wrong_feed,"unnormalized_next_loop_mismatches":wrong_feed_mismatches,"omitted_post_attention_norm":wrong_branch,"omitted_post_attention_norm_mismatches":wrong_branch_mismatches,"projection_compared_elements":projected.len(),"projection":projected,"projection_expected":[5.0,11.0,17.0],"projection_mismatches":projection_mismatches,"transposed_projection":transposed,"transposed_direction_mismatches":transposed_mismatches,"rope_compared_elements":rotated.len(),"rope":rotated,"rope_expected":[-3.0,2.0,1.0,4.0],"rope_mismatches":rope_mismatches}),
    )
}

fn contrasted(
    backend_name: &str,
    reference: &[f32],
    terminal: &[f32],
    strength: f32,
    config: Option<&RuntimeConfig>,
) -> io::Result<Vec<f32>> {
    let strengths = vec![strength; terminal.len()];
    let bindings = [
        ("reference", QuantizedBlock::Float32(reference)),
        ("terminal", QuantizedBlock::Float32(terminal)),
        ("strength", QuantizedBlock::Float32(&strengths)),
    ];
    let mut executor = Executor::prepare_with_config(
        graph::contrast(u32::try_from(terminal.len()).map_err(io::Error::other)?)?,
        backend_name,
        1,
        &bindings,
        false,
        config,
    )?;
    Ok(executor.run(&bindings, false)?.0)
}

pub fn paper(backend_name: &str) -> io::Result<Value> {
    paper_with_config(backend_name, None)
}

pub fn paper_with_config(backend_name: &str, config: Option<&RuntimeConfig>) -> io::Result<Value> {
    let tolerance = 0.00001;
    let reference = [1.0, 4.0, 2.0];
    let terminal = [2.0, 1.0, 2.0];
    let mut fixed = Vec::new();
    let mut fixed_mismatches = 0;
    for (strength, expected) in [
        (0.0, [2.0, 1.0, 2.0]),
        (0.5, [2.5, -0.5, 2.0]),
        (1.0, [3.0, -2.0, 2.0]),
    ] {
        let observed = contrasted(backend_name, &reference, &terminal, strength, config)?;
        let mismatches = mismatch(&observed, &expected, tolerance);
        fixed_mismatches += mismatches;
        fixed.push(json!({"strength":strength,"reference":reference,"terminal":terminal,"expected":expected,"observed":observed,"compared_elements":3,"mismatches":mismatches}));
    }
    let wrong_sign = contrasted(backend_name, &reference, &terminal, -0.5, config)?;
    let wrong_sign_mismatches = mismatch(&wrong_sign, &[2.5, -0.5, 2.0], tolerance);

    let logarithm = 3.0f32.ln();
    let mut adaptive_fixtures = Vec::new();
    let mut adaptive_mismatches = 0;
    for (name, terminal, expected_probabilities, expected_strength, expected_logits) in [
        (
            "original",
            [logarithm, 0.0],
            [0.75, 0.25],
            0.4,
            [7.0 * logarithm / 5.0, 0.0],
        ),
        (
            "shifted",
            [logarithm + 10.0, 10.0],
            [0.75, 0.25],
            0.4,
            [1.4 * (logarithm + 10.0), 14.0],
        ),
        ("tie", [2.0, 2.0], [0.5, 0.5], 0.8, [3.6, 3.6]),
    ] {
        let margin = guidance::adaptive(&terminal, 0.8)?;
        let probabilities = [margin.top_probability, margin.second_probability];
        let observed = contrasted(
            backend_name,
            &[0.0, 0.0],
            &terminal,
            margin.effective_strength,
            config,
        )?;
        let mismatches = mismatch(&probabilities, &expected_probabilities, tolerance)
            + mismatch(
                &[margin.effective_strength],
                &[expected_strength],
                tolerance,
            )
            + mismatch(&observed, &expected_logits, tolerance);
        adaptive_mismatches += mismatches;
        adaptive_fixtures.push(json!({"fixture":name,"reference":[0.0,0.0],"terminal":terminal,"probabilities":probabilities,"expected_probabilities":expected_probabilities,"margin":margin.margin,"effective_strength":margin.effective_strength,"expected_strength":expected_strength,"observed":observed,"expected":expected_logits,"compared_elements":5,"mismatches":mismatches}));
    }
    let expected_adaptive = [7.0 * logarithm / 5.0, 0.0];
    let wrong_reference_margin = guidance::adaptive(&[0.0, 0.0], 0.8)?;
    let wrong_reference_logits = contrasted(
        backend_name,
        &[0.0, 0.0],
        &[logarithm, 0.0],
        wrong_reference_margin.effective_strength,
        config,
    )?;
    let wrong_reference_mismatches =
        mismatch(&wrong_reference_logits, &expected_adaptive, tolerance);
    let wrong_guided_margin = guidance::adaptive(&expected_adaptive, 0.8)?;
    let wrong_guided_logits = contrasted(
        backend_name,
        &[0.0, 0.0],
        &[logarithm, 0.0],
        wrong_guided_margin.effective_strength,
        config,
    )?;
    let wrong_guided_mismatches = mismatch(&wrong_guided_logits, &expected_adaptive, tolerance);

    let raw_first = [1.0, 0.0];
    let raw_final = [0.0, 2.0];
    let hidden = contrasted(backend_name, &raw_first, &raw_final, 0.5, config)?;
    let scale = [1.0, 2.0];
    let epsilon = [0.0];
    let norm_bindings = [
        ("hidden", QuantizedBlock::Float32(&hidden)),
        ("scale", QuantizedBlock::Float32(&scale)),
        ("epsilon", QuantizedBlock::Float32(&epsilon)),
    ];
    let mut normalization = Executor::prepare_with_config(
        graph::normalization(2)?,
        backend_name,
        1,
        &norm_bindings,
        false,
        config,
    )?;
    let normalized_hidden = normalization.run(&norm_bindings, false)?.0;
    let head = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
    let head_bindings = [
        ("hidden", QuantizedBlock::Float32(&normalized_hidden)),
        ("head", QuantizedBlock::Float32(&head)),
    ];
    let shape = Shape {
        hidden: 2,
        intermediate: 2,
        heads: 1,
        head_width: 2,
        vocab: 3,
    };
    let mut readout = Executor::prepare_with_config(
        graph::readout(shape)?,
        backend_name,
        1,
        &head_bindings,
        false,
        config,
    )?;
    let hidden_logits = readout.run(&head_bindings, false)?.0;
    let scalar = (2.0f32 / 37.0).sqrt();
    let expected_hidden = [-scalar, 12.0 * scalar, 11.0 * scalar];
    let hidden_mismatches = mismatch(&hidden_logits, &expected_hidden, tolerance);
    let first_bindings = [
        ("hidden", QuantizedBlock::Float32(&raw_first)),
        ("scale", QuantizedBlock::Float32(&scale)),
        ("epsilon", QuantizedBlock::Float32(&epsilon)),
    ];
    let final_bindings = [
        ("hidden", QuantizedBlock::Float32(&raw_final)),
        ("scale", QuantizedBlock::Float32(&scale)),
        ("epsilon", QuantizedBlock::Float32(&epsilon)),
    ];
    let normalized_first = normalization.run(&first_bindings, false)?.0;
    let normalized_final = normalization.run(&final_bindings, false)?.0;
    let wrong_hidden = contrasted(
        backend_name,
        &normalized_first,
        &normalized_final,
        0.5,
        config,
    )?;
    let wrong_head_bindings = [
        ("hidden", QuantizedBlock::Float32(&wrong_hidden)),
        ("head", QuantizedBlock::Float32(&head)),
    ];
    let wrong_hidden_logits = readout.run(&wrong_head_bindings, false)?.0;
    let wrong_boundary_mismatches = mismatch(&wrong_hidden_logits, &expected_hidden, tolerance);
    let value = json!({
        "kind":"paper_controls", "backend":backend_name,
        "source":"PAPER-METHODS-SPEC.md worked examples before code", "absolute_tolerance":tolerance,
        "fixed_fixtures":fixed, "fixed_compared_elements":9, "fixed_mismatches":fixed_mismatches,
        "adaptive_fixtures":adaptive_fixtures, "adaptive_compared_elements":15, "adaptive_mismatches":adaptive_mismatches,
        "wrong_sign":wrong_sign, "wrong_sign_mismatches":wrong_sign_mismatches,
        "wrong_reference_probability_logits":wrong_reference_logits, "wrong_reference_probability_mismatches":wrong_reference_mismatches,
        "wrong_guided_probability_logits":wrong_guided_logits, "wrong_guided_probability_mismatches":wrong_guided_mismatches,
        "raw_reference_hidden":raw_first, "raw_terminal_hidden":raw_final, "contrasted_hidden":hidden,
        "normalized_hidden":normalized_hidden, "hidden_logits":hidden_logits, "expected_hidden_logits":expected_hidden,
        "hidden_compared_elements":3, "hidden_mismatches":hidden_mismatches,
        "wrong_boundary_logits":wrong_hidden_logits, "wrong_boundary_mismatches":wrong_boundary_mismatches,
    });
    if fixed_mismatches != 0
        || adaptive_mismatches != 0
        || hidden_mismatches != 0
        || wrong_sign_mismatches == 0
        || wrong_boundary_mismatches == 0
        || wrong_reference_mismatches == 0
        || wrong_guided_mismatches == 0
    {
        return Err(io::Error::other(format!("paper controls differ: {value}")));
    }
    Ok(value)
}
