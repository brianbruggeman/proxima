#[path = "../examples/low_precision_moe_training_pilot.rs"]
pub mod pilot;

use proxima_autograd::low_precision::WeightFormat;
use proxima_autograd::sparse::dedupe_and_sum_rows;
use proxima_tensor::test_support::Lcg;

fn assert_close(actual: &[f32], expected: &[f32], field: &str) {
    assert_eq!(actual.len(), expected.len(), "{field} lengths differ");
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual_value - expected_value).abs() <= 1e-6,
            "{field}[{index}] differs: {actual_value} vs {expected_value}"
        );
    }
}

#[test]
fn packed_pilot_step_matches_scalar_reference() {
    let model = pilot::build_model().expect("tiny pilot graph builds");
    let optimizer = pilot::optimizer_program();
    for format in [
        WeightFormat::Bf8E5M2,
        WeightFormat::Bf4E2M1,
        WeightFormat::Fp32,
    ] {
        let format_name = match format {
            WeightFormat::Bf8E5M2 => "bf8_e5m2",
            WeightFormat::Bf4E2M1 => "bf4_e2m1",
            WeightFormat::Fp32 => "fp32",
        };
        let mut lcg = Lcg(17);
        let masters: Vec<f32> = (0..32).map(|_| lcg.next_unit() * 0.5).collect();
        let packed_bytes = match format {
            WeightFormat::Fp32 => Vec::new(),
            _ => pilot::encode_packed_weights(&masters, format)
                .expect("finite expert rows encode into the selected packed format"),
        };
        let mut compact_values = Vec::with_capacity(8 * 16);
        let mut scalar_compact_values = Vec::with_capacity(8 * 16);
        let mut gradient_ids = Vec::with_capacity(8);
        let mut actual_logits = Vec::with_capacity(8 * 4);
        let mut scalar_logits = Vec::with_capacity(8 * 4);
        let mut actual_losses = Vec::with_capacity(8);
        let mut scalar_losses = Vec::with_capacity(8);
        for (token_id, target_id) in pilot::TRAIN_INPUTS.into_iter().zip(pilot::TRAIN_TARGETS) {
            let (logits, loss, gradient) = match format {
                WeightFormat::Fp32 => pilot::evaluate_token(&model, &masters, token_id, target_id)
                    .expect("FP32 control token evaluates"),
                _ => pilot::evaluate_packed_token(
                    &model,
                    &packed_bytes,
                    format,
                    token_id,
                    target_id,
                    1,
                )
                .expect("packed BF token evaluates without a decoded expert table"),
            };
            let (reference_logits, reference_loss, reference_gradient) =
                pilot::scalar_token(&masters, &packed_bytes, format, token_id, target_id);
            actual_logits.extend(logits);
            scalar_logits.extend(reference_logits);
            actual_losses.push(loss);
            scalar_losses.push(reference_loss);
            compact_values.extend(gradient);
            scalar_compact_values.extend(reference_gradient);
            gradient_ids.push((token_id % 2) as f32);
        }
        assert_close(&actual_logits, &scalar_logits, format_name);
        assert_close(&actual_losses, &scalar_losses, format_name);
        assert_close(&compact_values, &scalar_compact_values, format_name);

        let (expert_ids, coalesced_rows) = dedupe_and_sum_rows(&gradient_ids, &compact_values, 16)
            .expect("compact token gradients coalesce by selected expert");
        let (scalar_expert_ids, scalar_coalesced_rows) =
            dedupe_and_sum_rows(&gradient_ids, &scalar_compact_values, 16)
                .expect("scalar token gradients coalesce by selected expert");
        assert_eq!(expert_ids, [0, 1]);
        assert_eq!(expert_ids, scalar_expert_ids);
        assert_close(&coalesced_rows, &scalar_coalesced_rows, format_name);
        let averaged_gradients: Vec<f32> = coalesced_rows
            .iter()
            .map(|gradient| gradient / 8.0)
            .collect();
        let scalar_coalesced: Vec<f32> = scalar_coalesced_rows
            .chunks_exact(16)
            .flat_map(|row| row.iter().copied())
            .collect();
        let zeros = vec![0.0; 32];
        let (updated_masters, updated_first, updated_second) = pilot::run_optimizer(
            &optimizer.0,
            optimizer.1,
            optimizer.2,
            optimizer.3,
            &masters,
            &averaged_gradients,
            &zeros,
            &zeros,
            1.0,
        )
        .expect("one FP32 Adam update evaluates");
        let (reference_masters, reference_first, reference_second) =
            pilot::scalar_adam_step(&masters, &zeros, &zeros, &scalar_coalesced, 1);
        assert_close(&updated_masters, &reference_masters, format_name);
        assert_close(&updated_first, &reference_first, format_name);
        assert_close(&updated_second, &reference_second, format_name);

        let held_out_bytes = match format {
            WeightFormat::Fp32 => Vec::new(),
            _ => pilot::encode_packed_weights(&updated_masters, format)
                .expect("updated FP32 masters re-encode for held-out evaluation"),
        };
        for (token_id, target_id) in pilot::HELD_OUT_INPUTS
            .into_iter()
            .zip(pilot::HELD_OUT_TARGETS)
        {
            let (logits, loss, _) = match format {
                WeightFormat::Fp32 => {
                    pilot::evaluate_token(&model, &updated_masters, token_id, target_id)
                        .expect("FP32 held-out token evaluates")
                }
                _ => pilot::evaluate_packed_token(
                    &model,
                    &held_out_bytes,
                    format,
                    token_id,
                    target_id,
                    1,
                )
                .expect("held-out token consumes newly packed updated masters"),
            };
            let (reference_logits, reference_loss, _) = pilot::scalar_token(
                &updated_masters,
                &held_out_bytes,
                format,
                token_id,
                target_id,
            );
            assert_close(&logits, &reference_logits, format_name);
            assert_close(&[loss], &[reference_loss], format_name);
        }
    }
}
