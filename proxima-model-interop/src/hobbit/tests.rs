use std::path::PathBuf;

use proxima_gguf::tensor::TensorInfo;
use proxima_gguf::types::GgmlType;
use proxima_numeric::NumericFormat;

use super::*;

fn layer(
    name: &str,
    mean_kl: f64,
    saved_bytes: u64,
    recoded_bytes: u64,
    base_bytes: u64,
) -> HobbitLayerSensitivity {
    HobbitLayerSensitivity {
        layer: name.to_owned(),
        mean_kl,
        top1_agreement: 1.0 - mean_kl,
        positions: 4,
        recoded_bytes,
        base_bytes,
        saved_bytes,
        tensors: vec![format!("{name}ffn_down.weight")],
    }
}

fn report(base_bytes: u64, layers: Vec<HobbitLayerSensitivity>) -> HobbitSensitivityReport {
    let hash = ranking_hash("q3_k", 4, &layers);
    HobbitSensitivityReport {
        candidate_codec: "q3_k".to_owned(),
        prompts: 2,
        positions: 4,
        base_bytes,
        layers,
        ranking_hash: hash,
    }
}

#[test]
fn hobbit_quality_delta_identity_is_perfect_and_kl_is_zero() {
    let rows = vec![vec![3.0f32, 1.0, -1.0, 0.5], vec![0.0f32, 2.0, 1.0, -2.0]];
    let delta = quality_delta(&rows, &rows).expect("identical runs compare");
    assert_eq!(delta.positions, 2);
    assert_eq!(delta.top1_agreement, 1.0);
    assert!(delta.mean_kl.abs() < 1e-12, "{}", delta.mean_kl);
}

#[test]
fn hobbit_quality_delta_flip_and_kl_are_measured_not_proxied() {
    let reference = vec![vec![4.0f32, 1.0, 0.0]];
    let candidate = vec![vec![1.0f32, 4.0, 0.0]];
    let delta = quality_delta(&reference, &candidate).expect("same shape");
    assert_eq!(delta.positions, 1);
    assert_eq!(delta.top1_agreement, 0.0);
    assert!(delta.mean_kl > 0.0);
    let log_softmax_reference = log_softmax(&reference[0]);
    let log_softmax_candidate = log_softmax(&candidate[0]);
    let expected = log_softmax_reference
        .iter()
        .zip(log_softmax_candidate.iter())
        .map(|(log_p, log_q)| log_p.exp() * (log_p - log_q))
        .sum::<f64>();
    assert!((delta.mean_kl - expected).abs() < 1e-12);
}

#[test]
fn hobbit_quality_delta_rejects_mismatched_rows() {
    let error = quality_delta(&[vec![0.0, 1.0]], &[]).expect_err("row count differs");
    assert!(matches!(
        error,
        HobbitError::QualityShapeMismatch {
            reference_rows: 1,
            candidate_rows: 0
        }
    ));
}

#[test]
fn hobbit_allocator_spends_on_least_sensitive_first_and_meets_budget() {
    let report = report(
        1000,
        vec![
            layer("blk.0.", 0.5, 100, 250, 350),
            layer("blk.1.", 0.1, 200, 150, 350),
            layer("blk.2.", 0.3, 50, 100, 150),
        ],
    );
    let allocation = allocate(&report, Some(750), 1).expect("750 is reachable");
    assert_eq!(allocation.budget_bytes, 750);
    assert_eq!(allocation.assigned_bytes, 750);
    assert_eq!(allocation.shortfall, 0);
    assert_eq!(allocation.saved_bytes, 250);
    assert_eq!(
        allocation.rules,
        vec![
            HobbitRule {
                tensor: "blk.1.ffn_down.weight".to_owned(),
                codec: "q3_k".to_owned(),
            },
            HobbitRule {
                tensor: "blk.2.ffn_down.weight".to_owned(),
                codec: "q3_k".to_owned(),
            },
        ],
        "0.1 then 0.3 sensitivity; blk.0 is the most sensitive and stays high"
    );
}

#[test]
fn hobbit_allocator_is_deterministic_across_shuffled_rankings() {
    let layers = vec![
        layer("blk.2.", 0.3, 50, 100, 150),
        layer("blk.0.", 0.5, 100, 250, 350),
        layer("blk.1.", 0.1, 200, 150, 350),
    ];
    let first = allocate(&report(1000, layers.clone()), Some(700), 1).expect("feasible");
    let mut shuffled = layers;
    shuffled.reverse();
    let second = allocate(&report(1000, shuffled), Some(700), 1).expect("feasible");
    assert_eq!(first, second);
}

#[test]
fn hobbit_allocator_honors_min_layers_even_when_budget_is_met() {
    let report = report(
        500,
        vec![
            layer("blk.0.", 0.1, 100, 50, 150),
            layer("blk.1.", 0.2, 100, 50, 150),
            layer("blk.2.", 0.3, 100, 50, 150),
        ],
    );
    let allocation = allocate(&report, Some(500), 2).expect("two layers fit");
    assert_eq!(allocation.rules.len(), 2);
    assert_eq!(allocation.assigned_bytes, 300);
}

#[test]
fn hobbit_allocator_refuses_an_unreachable_budget_with_the_shortfall() {
    let report = report(
        1000,
        vec![
            layer("blk.0.", 0.1, 100, 250, 350),
            layer("blk.1.", 0.2, 100, 250, 350),
        ],
    );
    let error = allocate(&report, Some(700), 1).expect_err("minimum is 800");
    match error {
        HobbitError::BudgetInfeasible {
            budget,
            minimum,
            shortfall,
        } => {
            assert_eq!(budget, 700);
            assert_eq!(minimum, 800);
            assert_eq!(shortfall, 100);
        }
        other => panic!("expected BudgetInfeasible, got {other}"),
    }
}

#[test]
fn hobbit_allocator_refuses_unset_budget_and_impossible_min_layers() {
    let report = report(100, vec![layer("blk.0.", 0.1, 10, 5, 15)]);
    assert!(matches!(
        allocate(&report, None, 0).expect_err("budget missing"),
        HobbitError::BudgetUnset
    ));
    assert!(matches!(
        allocate(&report, Some(100), 2).expect_err("only one candidate"),
        HobbitError::InsufficientCandidateLayers {
            requested: 2,
            available: 1
        }
    ));
}

#[test]
fn hobbit_ranking_hash_is_stable_and_order_sensitive() {
    let one = layer("blk.0.", 0.1, 10, 5, 15);
    let two = layer("blk.1.", 0.2, 10, 5, 15);
    let first = ranking_hash("q3_k", 4, &[one.clone(), two.clone()]);
    let second = ranking_hash("q3_k", 4, &[one.clone(), two.clone()]);
    let reversed = ranking_hash("q3_k", 4, &[two, one]);
    assert_eq!(first, second);
    assert_ne!(first, reversed);
}

#[test]
fn hobbit_layer_prefixes_come_from_the_directory_not_a_list() {
    let parsed = prose_checkpoint(&[
        "blk.7.attn_q.weight",
        "blk.2.ffn_up.weight",
        "token_embd.weight",
    ]);
    assert_eq!(
        layer_prefixes(&parsed),
        vec!["blk.2.".to_owned(), "blk.7.".to_owned()]
    );
}

#[test]
fn hobbit_settings_section_loads_through_serving_settings() {
    let toml_text = r#"
model_path = "checkpoint.gguf"

[hobbit]
budget_bytes = 1234
candidate_codecs = ["q4_k"]
calibration_prompts = ["calibrate"]
min_layers = 3
max_calibration_tokens = 8

[hobbit.quality]
max_mean_kl = 0.02
"#;
    let settings: crate::ServingSettings =
        conflaguration::from_toml_str(toml_text).expect("the serving settings load");
    assert_eq!(settings.hobbit.budget_bytes, Some(1234));
    assert_eq!(settings.hobbit.min_layers, 3);
    assert_eq!(
        settings.hobbit.candidate_targets().expect("q4_k encodes"),
        vec![GgmlType::Q4_K]
    );
    assert_eq!(settings.hobbit.quality_bound(), Some((Some(0.02), None)));
}

/// The control: the same quality check that passes a measured delta must
/// fail when the rule is mutated into a lossy one. This proves the gate can
/// reject, not only accept.
#[test]
fn control_hobbit_quality_gate_rejects_a_mutated_lossy_rule() {
    let passing = HobbitQualityDelta {
        positions: 4,
        top1_agreement: 0.98,
        mean_kl: 0.01,
    };
    check_quality_bound(&passing, Some((Some(0.05), Some(0.95))))
        .expect("the measured delta holds the configured bound");
    let mutated = HobbitQualityDelta {
        mean_kl: 0.30,
        ..passing
    };
    let error = check_quality_bound(&mutated, Some((Some(0.05), Some(0.95))))
        .expect_err("the mutated rule must fail");
    assert!(matches!(error, HobbitError::QualityBoundExceeded { .. }));
    check_quality_bound(&passing, None).expect_err("no bound, no apply");
}

#[test]
fn control_hobbit_allocation_check_rejects_an_overspent_allocation() {
    let report = report(1000, vec![layer("blk.0.", 0.1, 100, 250, 350)]);
    // The same allocator that refuses 700 must accept the budget where the
    // minimum fits; a mutated budget one byte lower must fail.
    allocate(&report, Some(900), 1).expect("900 is reachable");
    let error = allocate(&report, Some(899), 1).expect_err("899 is below the 900 minimum");
    assert!(matches!(error, HobbitError::BudgetInfeasible { .. }));
}

fn prose_checkpoint(names: &[&str]) -> ParsedGguf {
    let tensors = names
        .iter()
        .map(|name| {
            let mut dims = arrayvec::ArrayVec::new();
            dims.push(64u64);
            dims.push(2u64);
            TensorInfo {
                name: (*name).to_owned(),
                dims,
                ggml_type: GgmlType::Q4_K,
                offset: 0,
            }
        })
        .collect();
    ParsedGguf {
        version: 3,
        tensor_count: names.len() as u64,
        kv_count: 0,
        metadata: Vec::new(),
        tensors,
        data_offset: 0,
        alignment: 32,
    }
}

fn real_checkpoint(env_key: &str, default: &str) -> PathBuf {
    let path = std::env::var_os(env_key)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default));
    assert!(
        path.is_file(),
        "HOBBIT's real-checkpoint gate needs a GGUF at {env_key} or {default}; \
         set {env_key} to a real checkpoint. This gate fails loudly rather than skipping."
    );
    path
}

fn mapped(path: &std::path::Path) -> (std::fs::File, memmap2::Mmap) {
    let file = std::fs::File::open(path).expect("open real checkpoint");
    let map = unsafe { memmap2::Mmap::map(&file) }.expect("map real checkpoint");
    (file, map)
}

/// G5's real gate: a configured rule reaches the real `LoadedModel` load
/// path, the bound census names `<tensor>@<codec>`, and a non-matching rule
/// leaves the program hash and the recoded census untouched.
#[test]
fn hobbit_real_dense_checkpoint_rule_changes_bound_names_and_program() {
    let path = real_checkpoint(
        "PROXIMA_HOBBIT_GGUF",
        "/Users/brianbruggeman/models/granite-4.0-350m/granite-4.0-350m-Q4_K_M.gguf",
    );
    let (_file, map) = mapped(&path);
    let bytes: &[u8] = &map;
    let parsed = proxima_gguf::pipe::parse_complete(bytes).expect("parse real dense checkpoint");
    let config = ServingConfig {
        gpu_layers: 0,
        ..ServingConfig::default()
    };
    let baseline = LoadedModel::load_with_serving(&parsed, bytes, &config)
        .expect("load the unrecoded checkpoint");
    assert!(
        baseline.recoded_weights().is_empty(),
        "no rule must be byte-identical to today"
    );

    let prefix = layer_prefixes(&parsed)
        .into_iter()
        .next()
        .expect("a real transformer names layers");
    let target = GgmlType::Q3_K;
    let candidates = recode_candidate_tensors(&parsed, &prefix, target);
    let rules = candidates
        .iter()
        .map(|tensor| WeightPrecisionRule {
            pattern: NamePattern::Exact(tensor.name.as_str()),
            target,
        })
        .collect::<Vec<_>>();
    assert!(!rules.is_empty(), "the first layer has a rank-2 weight");
    let recoded_config = config.with_weight_precision(&rules);
    let recoded = LoadedModel::load_with_serving(&parsed, bytes, &recoded_config)
        .expect("load the recoded checkpoint");
    let census = recoded.recoded_weights();
    assert!(!census.is_empty(), "the rule reached the real bind");
    for tensor in &candidates {
        let expected = format!("{}@q3_k", tensor.name);
        assert!(
            census
                .iter()
                .any(|(name, bytes)| name == &expected && *bytes > 0),
            "expected {expected} in {census:?}"
        );
    }
    assert_ne!(
        baseline.program_hash(),
        recoded.program_hash(),
        "the lowered program must be retargeted to the recoded leaf names"
    );

    let non_matching = [WeightPrecisionRule {
        pattern: NamePattern::Exact("blk.999.no_such_tensor.weight"),
        target,
    }];
    let unchanged = LoadedModel::load_with_serving(
        &parsed,
        bytes,
        &config.with_weight_precision(&non_matching),
    )
    .expect("a non-matching rule loads");
    assert!(unchanged.recoded_weights().is_empty());
    assert_eq!(baseline.program_hash(), unchanged.program_hash());
}

/// G7's real gate: one binary path, one rule shape, an MoE checkpoint. The
/// layer prefixes and candidate tensors come from the directory; there is no
/// family branch anywhere in this test.
#[test]
fn hobbit_real_moe_checkpoint_rule_changes_bound_names() {
    let path = real_checkpoint(
        "PROXIMA_HOBBIT_MOE_GGUF",
        "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80",
    );
    let (_file, map) = mapped(&path);
    let bytes: &[u8] = &map;
    let parsed = proxima_gguf::pipe::parse_complete(bytes).expect("parse real MoE checkpoint");
    let config = ServingConfig {
        gpu_layers: 0,
        ..ServingConfig::default()
    };
    let prefix = layer_prefixes(&parsed)
        .into_iter()
        .next()
        .expect("a real MoE checkpoint names layers");
    let target = GgmlType::Q4_K;
    let candidates = recode_candidate_tensors(&parsed, &prefix, target);
    assert!(
        !candidates.is_empty(),
        "a MoE layer still has rank-2 attention/shared tensors"
    );
    let rules = candidates
        .iter()
        .map(|tensor| WeightPrecisionRule {
            pattern: NamePattern::Exact(tensor.name.as_str()),
            target,
        })
        .collect::<Vec<_>>();
    let recoded =
        LoadedModel::load_with_serving(&parsed, bytes, &config.with_weight_precision(&rules))
            .expect("load the recoded MoE checkpoint");
    let census = recoded.recoded_weights();
    assert!(!census.is_empty(), "the rule reached the MoE bind");
    assert!(
        census
            .iter()
            .all(|(name, bytes)| name.contains('@') && *bytes > 0)
    );
}

#[test]
fn hobbit_real_checkpoint_sensitivity_rankings_match_twice() {
    let path = real_checkpoint(
        "PROXIMA_HOBBIT_GGUF",
        "/Users/brianbruggeman/models/granite-4.0-350m/granite-4.0-350m-Q4_K_M.gguf",
    );
    let (_file, map) = mapped(&path);
    let bytes: &[u8] = &map;
    let parsed = proxima_gguf::pipe::parse_complete(bytes).expect("parse real dense checkpoint");
    let config = ServingConfig {
        gpu_layers: 0,
        ..ServingConfig::default()
    };
    let prompts = vec!["The quick brown fox jumps over the lazy dog.".to_owned()];
    let prefixes = layer_prefixes(&parsed)
        .into_iter()
        .take(2)
        .collect::<Vec<_>>();
    let first = measure_sensitivity_for_layers(
        &parsed,
        bytes,
        &prompts,
        NumericFormat::Q3K,
        4,
        &config,
        &prefixes,
    )
    .expect("the first sensitivity pass runs");
    let second = measure_sensitivity_for_layers(
        &parsed,
        bytes,
        &prompts,
        NumericFormat::Q3K,
        4,
        &config,
        &prefixes,
    )
    .expect("the second sensitivity pass runs");
    assert_eq!(first.ranking_hash, second.ranking_hash);
    assert_eq!(first.layers, second.layers);
}
