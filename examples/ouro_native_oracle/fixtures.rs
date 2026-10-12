use std::io;
use std::path::Path;

use serde_json::{Value, json};

use super::compare::{self, JsonLines};
use super::math::{self, Mode};

fn fixture(
    writer: &mut JsonLines,
    name: &str,
    actual: &[f32],
    expected: &[f32],
    exact: bool,
) -> io::Result<()> {
    let tolerance = if exact { 0.0 } else { 0.000001 };
    let counts = compare::counts(actual, expected, tolerance, tolerance)?;
    writer.emit(&json!({"kind":"fixture","name":name,"spec":"proxima-tensor/specs/loopcd-panel/ORACLE-SPEC.md","actual":actual,"expected":expected,"coordinates":counts.compared,"exceeding":counts.exceeding,"absolute_tolerance":tolerance,"relative_tolerance":tolerance}))?;
    if counts.exceeding != 0 || (exact && counts.exact != counts.compared) {
        return Err(io::Error::other(format!("hand fixture {name} differs")));
    }
    Ok(())
}

pub fn run(directory: &Path, negative: bool) -> io::Result<Value> {
    let mut records = JsonLines::create(&directory.join("fixtures.jsonl"))?;
    let input = [1.0, 2.0, 3.0, -1.0, 0.0, 2.0];
    let weights = [1.0, 0.0, 2.0, 0.0, -1.0, 1.0, 2.0, 1.0, 0.0, -1.0, 2.0, 1.0];
    let projection = math::linear(&input, &weights, 2, 3, 4, Mode::F32)?;
    fixture(
        &mut records,
        "projection",
        &projection,
        &[7.0, 1.0, 4.0, 6.0, 3.0, 2.0, -2.0, 3.0],
        true,
    )?;
    let rotated = math::rope(
        &[1.0, 2.0, 3.0, 4.0],
        &[0.0, 1.0, 0.0, 1.0],
        &[1.0, 0.0, 1.0, 0.0],
        1,
        1,
        4,
        Mode::F32,
    )?;
    fixture(&mut records, "rope", &rotated, &[-3.0, 2.0, 1.0, 4.0], true)?;
    let normalized = math::norm(&[3.0, 4.0], &[2.0, 0.5], 12.5, Mode::F32)?;
    let normalized_bf16 = math::norm(&[3.0, 4.0], &[2.0, 0.5], 12.5, Mode::Bf16)?;
    let norm_values: Vec<f32> = normalized.iter().chain(&normalized_bf16).copied().collect();
    fixture(
        &mut records,
        "norm",
        &norm_values,
        &[1.2, 0.4, 1.203125, 0.400_390_63],
        false,
    )?;
    let first = math::residual(&[1.0, 2.0], &[1.0, 0.0], Mode::F32)?;
    let feedback = math::norm(&first, &[1.0, 1.0], 0.0, Mode::F32)?;
    let second = math::residual(&feedback, &[1.0, 0.0], Mode::F32)?;
    let result = math::norm(&second, &[1.0, 1.0], 0.0, Mode::F32)?;
    let feedback_values: Vec<f32> = feedback
        .iter()
        .chain(&second)
        .chain(&result)
        .copied()
        .collect();
    fixture(
        &mut records,
        "feedback",
        &feedback_values,
        &[
            1.0,
            1.0,
            2.0,
            1.0,
            2.0 * (2.0f32 / 5.0).sqrt(),
            (2.0f32 / 5.0).sqrt(),
        ],
        false,
    )?;
    let attended = math::attention(&[0.0, 0.0], &[0.0, 0.0], &[2.0, 6.0], 2, 1, 1, Mode::F32)?;
    fixture(
        &mut records,
        "causal_attention",
        &attended,
        &[2.0, 4.0],
        true,
    )?;
    let bits = [
        0x3f80_8000,
        0x3f81_8000,
        0xbf80_8000,
        0xbf81_8000,
        0,
        0x8000_0000,
    ];
    let rounding: Vec<f32> = bits
        .iter()
        .map(|bits| Mode::Bf16.round(f32::from_bits(*bits)))
        .collect::<io::Result<_>>()?;
    let expected_rounding: Vec<f32> = [
        0x3f80_0000,
        0x3f82_0000,
        0xbf80_0000,
        0xbf82_0000,
        0,
        0x8000_0000,
    ]
    .iter()
    .map(|bits| f32::from_bits(*bits))
    .collect();
    fixture(&mut records, "rne", &rounding, &expected_rounding, true)?;
    let guided = math::contrast(&[1.0, 4.0, 2.0], &[2.0, 1.0, 2.0], 0.5)?;
    fixture(&mut records, "contrast", &guided, &[2.5, -0.5, 2.0], true)?;
    let words: [u16; 6] = [0x0000, 0x8000, 0x3f80, 0xc000, 0x3fc0, 0x7f7f];
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let decoded = math::decode(&bytes)?;
    for (actual, word) in decoded.iter().zip(words) {
        if actual.to_bits() != u32::from(word) << 16 {
            return Err(io::Error::other("independent decoder fixture differs"));
        }
    }
    records.emit(&json!({"kind":"decoder_fixture","input_words":words,"output_bits":decoded.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),"coordinates":6,"mismatches":0}))?;
    let original_difference =
        f64::from(f32::from_bits(3118744224)) - f64::from(f32::from_bits(3118744944));
    let encoded = serde_json::to_string(&original_difference).map_err(io::Error::other)?;
    let parsed: f64 = serde_json::from_str(&encoded).map_err(io::Error::other)?;
    records.emit(&json!({"kind":"json_f64_transport","original_bits":original_difference.to_bits(),"parsed_bits":parsed.to_bits(),"encoded":encoded,"bit_equal":original_difference.to_bits()==parsed.to_bits(),"comparator_policy":"canonical F32 values are checked by bits; derived F64 fields are checked after the same serializer/parser transform"}))?;
    let mut rejected = 0;
    if negative {
        let wrong_projection = [1.0, 7.0, 4.0, 6.0, 2.0, 3.0, -2.0, 3.0];
        let wrong_rotary = [-2.0, 2.0, -4.0, 4.0];
        let wrong_feedback = [3.0, 2.0];
        let wrong_mask = [4.0, 4.0];
        let wrong_ties: Vec<f32> = bits
            .iter()
            .map(|bits| f32::from_bits(*bits & 0xffff_0000))
            .collect();
        let wrong_contrast = [1.5, 2.5, 2.0];
        for (name, wrong, expected) in [
            (
                "projection",
                wrong_projection.as_slice(),
                projection.as_slice(),
            ),
            ("adjacent_rope", wrong_rotary.as_slice(), rotated.as_slice()),
            (
                "missing_feedback_norm",
                wrong_feedback.as_slice(),
                second.as_slice(),
            ),
            (
                "missing_causal_mask",
                wrong_mask.as_slice(),
                attended.as_slice(),
            ),
            (
                "bf16_ties",
                wrong_ties.as_slice(),
                expected_rounding.as_slice(),
            ),
            (
                "wrong_contrast_sign",
                wrong_contrast.as_slice(),
                guided.as_slice(),
            ),
        ] {
            let result = compare::counts(wrong, expected, 0.0, 0.0)?;
            let differs = result.exact != result.compared;
            records.emit(&json!({"kind":"negative_control","name":name,"actual":wrong,"expected":expected,"differing_coordinates":result.compared-result.exact,"rejected":differs}))?;
            if !differs {
                return Err(io::Error::other(format!(
                    "negative control {name} unexpectedly accepted"
                )));
            }
            rejected += 1;
        }
        let positive = compare::counts(&[1.0, 2.0], &[1.0, 2.0], 0.0, 0.0)?;
        if positive.exact != 2 {
            return Err(io::Error::other("positive comparator fixture differs"));
        }
        let missing = compare::counts(&[1.0], &[1.0, 2.0], 0.0, 0.0).is_err();
        if !missing {
            return Err(io::Error::other("missing-boundary control accepted"));
        }
        records.emit(&json!({"kind":"comparator_controls","positive_literal_accept":1,"missing_boundary_structural_rejection":1}))?;
    }
    records.flush()?;
    Ok(
        json!({"fixture_records":7,"fixture_expected_coordinate_mismatches":0,"decoder_values_checked":6,"sgemm_projection_coordinates":8,"negative_controls":rejected,"negative_controls_rejected":rejected,"unexpected_control_acceptances":0}),
    )
}

pub fn adaptive(terminal: &[f32], maximum_strength: f32) -> io::Result<(f32, Value)> {
    if terminal.len() < 2
        || terminal.iter().any(|value| !value.is_finite())
        || !maximum_strength.is_finite()
        || maximum_strength < 0.0
    {
        return Err(io::Error::other(
            "adaptive logits or maximum strength differ",
        ));
    }
    let mut first = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for value in terminal {
        if *value > first {
            second = first;
            first = *value;
        } else if *value > second {
            second = *value;
        }
    }
    let mut denominator = 0.0f32;
    for value in terminal {
        denominator += (*value - first).exp();
    }
    let top_probability = 1.0 / denominator;
    let second_probability = (second - first).exp() / denominator;
    let margin = top_probability - second_probability;
    let effective = maximum_strength * (1.0 - margin);
    Ok((
        effective,
        json!({"top_probability":top_probability,"second_probability":second_probability,"margin":margin,"effective_strength":effective,"probability_source":"full softmax of unguided terminal logits"}),
    ))
}

pub fn method_fixtures(directory: &Path) -> io::Result<Value> {
    let mut writer = JsonLines::create(&directory.join("method-fixtures.jsonl"))?;
    let terminal = [3.0f32.ln(), 0.0];
    let (effective, probabilities) = adaptive(&terminal, 0.8)?;
    let output = math::contrast(&[0.0, 0.0], &terminal, effective)?;
    fixture(
        &mut writer,
        "adaptive_ln3",
        &[effective, output[0], output[1]],
        &[0.4, 1.4 * 3.0f32.ln(), 0.0],
        false,
    )?;
    writer.emit(&json!({"kind":"adaptive_fixture_payload","adaptive":probabilities,"terminal":terminal,"output":output}))?;
    let (shifted, _) = adaptive(&[3.0f32.ln() + 10.0, 10.0], 0.8)?;
    fixture(&mut writer, "adaptive_shift", &[shifted], &[0.4], false)?;
    let (tied, _) = adaptive(&[2.0, 2.0], 0.8)?;
    fixture(&mut writer, "adaptive_tie", &[tied], &[0.8], true)?;
    writer.flush()?;
    Ok(
        json!({"method_fixture_records":3,"method_fixture_mismatches":0,"probabilities_from_unguided_full_softmax":true}),
    )
}
