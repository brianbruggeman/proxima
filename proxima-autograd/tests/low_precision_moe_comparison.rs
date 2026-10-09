use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use pilot_json::Value;

const FORMATS: [&str; 3] = ["bf8_e5m2", "bf4_e2m1", "fp32"];
const SEEDS: [u64; 3] = [17, 29, 43];
const TRAIN_INPUTS: [u64; 8] = [0, 1, 2, 3, 0, 1, 2, 3];
const TRAIN_TARGETS: [u64; 8] = [1, 2, 3, 0, 1, 2, 3, 0];
const HELD_OUT_INPUTS: [u64; 8] = [0, 0, 1, 1, 2, 2, 3, 3];
const HELD_OUT_TARGETS: [u64; 8] = [0, 1, 1, 2, 2, 3, 3, 0];

fn integer_array(value: &Value, field: &str) -> Vec<u64> {
    value[field]
        .as_array()
        .expect("payload field is an array")
        .iter()
        .map(|element| element.as_u64().expect("payload element is an integer"))
        .collect()
}

fn float_array(value: &Value, field: &str) -> Vec<f64> {
    value[field]
        .as_array()
        .expect("parameter field is an array")
        .iter()
        .map(|element| element.as_f64().expect("parameter is numeric"))
        .collect()
}

#[test]
fn low_precision_moe_fixed_payload_comparison() {
    let artifact_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../proxima-tensor/specs/low_precision_moe_training/results/tiny-pilot.json");
    let artifact = fs::read_to_string(artifact_path).expect("pilot report exists");
    let report: Value = pilot_json::from_str(&artifact).expect("pilot report is JSON");

    assert_eq!(report["fixture"], "tiny");
    assert_eq!(report["scale_cost"], "unmeasured");
    assert_eq!(report["training_steps"], 64);
    let arms = report["arms"].as_array().expect("report contains arms");
    assert_eq!(arms.len(), 9, "three formats by three seeds");

    let mut initial_parameters_by_seed = BTreeMap::new();
    let mut step_count = 0;
    let mut wall_time_count = 0;
    let mut throughput_count = 0;
    let mut rss_count = 0;
    for (format, seed) in arms.iter().map(|arm| {
        (
            arm["format"].as_str().expect("arm format is text"),
            arm["seed"].as_u64().expect("arm seed is integer"),
        )
    }) {
        assert!(FORMATS.contains(&format));
        assert!(SEEDS.contains(&seed));
    }

    for format in FORMATS {
        for seed in SEEDS {
            let matching_arms: Vec<_> = arms
                .iter()
                .filter(|arm| arm["format"] == format && arm["seed"] == seed)
                .collect();
            assert_eq!(matching_arms.len(), 1, "one record for {format}/{seed}");
            let arm = matching_arms[0];
            assert_eq!(arm["parameters"], 32);
            assert_eq!(arm["hyperparameter_search_trials"], 0);
            assert_eq!(arm["checkpoint_step"], 64);
            let initial = float_array(arm, "initial_parameters");
            let final_parameters = float_array(arm, "final_parameters");
            assert_eq!(initial.len(), 32);
            assert_eq!(final_parameters.len(), 32);
            if format == FORMATS[0] {
                initial_parameters_by_seed.insert(seed, initial.clone());
            } else {
                assert_eq!(initial, initial_parameters_by_seed[&seed]);
            }

            let steps = arm["steps"].as_array().expect("arm contains steps");
            assert_eq!(steps.len(), 64);
            step_count += steps.len();
            for (step_index, step) in steps.iter().enumerate() {
                assert_eq!(step["step"], step_index as u64 + 1);
                assert_eq!(integer_array(step, "input_ids"), TRAIN_INPUTS);
                assert_eq!(
                    integer_array(step, "expert_ids"),
                    vec![0, 1, 0, 1, 0, 1, 0, 1]
                );
                assert_eq!(integer_array(step, "target_ids"), TRAIN_TARGETS);
                assert_eq!(integer_array(step, "held_out_input_ids"), HELD_OUT_INPUTS);
                assert_eq!(
                    integer_array(step, "held_out_expert_ids"),
                    vec![0, 0, 1, 1, 0, 0, 1, 1]
                );
                assert_eq!(integer_array(step, "held_out_target_ids"), HELD_OUT_TARGETS);
                assert!(step["train_loss"].as_f64().is_some());
                assert!(step["held_out_loss"].as_f64().is_some());
                assert!(step["non_finite_count"].as_u64().is_some());
                assert!(step["wall_time_ns"].as_u64().is_some());
                assert!(step["tokens_per_second"].as_f64().is_some());
                wall_time_count += 1;
                throughput_count += 1;
            }
            assert!(arm["peak_rss_bytes"].as_u64().is_some());
            rss_count += 1;
        }
    }

    assert_eq!(step_count, 576);
    assert_eq!(wall_time_count, 576);
    assert_eq!(throughput_count, 576);
    assert_eq!(rss_count, 9);
}
