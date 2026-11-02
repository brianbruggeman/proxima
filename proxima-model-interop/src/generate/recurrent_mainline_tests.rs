// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

//! A recurrent family loaded through the ordinary GGUF path: the family
//! profile names the recurrence, `build_forward` lowers it, and `LoadedModel`
//! serves it with no per-family code. Every expected value is computed by hand
//! from [`super::ouro_fixture`]'s checkpoint: embeddings `[3, 4]`, branches that
//! add nothing, unit gammas and epsilon one, so a pass ends at `[3, 4]` and each
//! output norm scales by `1 / sqrt(mean_square + 1)`.

use core::ops::ControlFlow;

use proxima_gguf::parse_complete;
use proxima_tensor::LoopCdMode;
use proxima_tensor::spec::{CacheMask, LayerCacheRoots, RecurrenceConfig, RopeLayout, RopePairing};

use super::LoadedModel;
use super::ouro_fixture::{STATE_VOCABULARY, state_sensitive_ouro_gguf, tiny_ouro_gguf};
use crate::error::InteropError;
use crate::lowering::{KvLayout, bind_checkpoint, header_descriptor};
use crate::profiles::family_profile;
use crate::recurrent_config::RecurrentConfig;
use crate::serving::ServingConfig;

fn near(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 0.000_001 + 0.000_001 * expected.abs(),
            "coordinate {index}: actual={actual} expected={expected}"
        );
    }
}

fn guided(iterations: Option<u32>, mode: LoopCdMode, strength: f32) -> RecurrentConfig {
    RecurrentConfig {
        iterations,
        mode,
        strength,
        reference_iteration: 1,
    }
}

fn first_pass() -> [f32; 2] {
    let scale = (2.0f32 / 27.0).sqrt();
    [3.0 * scale, 4.0 * scale]
}

fn second_pass() -> [f32; 2] {
    let scale = (2.0f32 / 52.0).sqrt();
    [3.0 * scale, 4.0 * scale]
}

fn logits_of(model: &LoadedModel<'_>, config: &ServingConfig) -> Vec<f32> {
    model
        .forward_node_values_from_ids(&[0], &[model.logits_root()], config)
        .expect("evaluate the recurrent graph")
        .remove(0)
}

#[test]
fn recurrent_mainline_profile_is_data() {
    let profile = family_profile("ouro").expect("the ouro profile is embedded");

    assert_eq!(
        profile.recurrence.map(|r| r.normalize_each_iteration),
        Some(true)
    );
    assert!(!profile.qk_norm);
    assert_eq!(profile.rope_layout, RopeLayout::Adjacent);
    assert_eq!(profile.rope_pairing(2), RopePairing::Interleaved);
    assert!(profile.ffn.post_attention_norm && profile.ffn.exclusive_dense_post_norm);
    assert_eq!(
        profile.kv_cache_shape,
        proxima_tensor::spec::KvCacheShape::Custom
    );
}

#[test]
fn recurrent_mainline_header_descriptor_repeats_the_physical_schedule() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");

    let descriptor = header_descriptor(&parsed, KvLayout::Full).expect("read the header");

    assert_eq!(descriptor.block_count, 1, "block_count stays physical");
    assert_eq!(descriptor.layers.len(), 1);
    assert_eq!(
        descriptor.recurrence,
        Some(RecurrenceConfig {
            iterations: 2,
            normalize_each_iteration: true
        })
    );
    assert_eq!(descriptor.cache_mask, CacheMask::Padded);
    assert!(!descriptor.layers[0].attention.qk_norm);
    assert_eq!(
        descriptor.layers[0].attention.rope_pairing,
        RopePairing::Interleaved
    );
}

#[test]
fn recurrent_mainline_tiny_gguf_literal() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
    let model = LoadedModel::load(&parsed, &bytes).expect("load through the family profile");

    let logits = logits_of(&model, &ServingConfig::default());

    near(&logits, &second_pass());
    let roots = model
        .recurrent_readout_roots()
        .expect("a recurrent graph has readouts");
    let rows = model
        .forward_node_values_from_ids(
            &[0],
            &[
                roots.reference_raw.expect("first pass raw"),
                roots.reference_normalized.expect("first pass normalized"),
                roots.terminal_raw.expect("terminal raw"),
            ],
            &ServingConfig::default(),
        )
        .expect("read the completed-pass roots");
    near(&rows[0], &[3.0, 4.0]);
    near(&rows[1], &first_pass());
    near(&rows[2], &first_pass());
}

#[test]
fn recurrent_mainline_depth_follows_the_header() {
    let one = tiny_ouro_gguf(1, false);
    let parsed = parse_complete(&one).expect("parse the one-pass GGUF");
    let model = LoadedModel::load(&parsed, &one).expect("load one pass");

    near(&logits_of(&model, &ServingConfig::default()), &first_pass());
    assert_eq!(model.recurrent_config().iterations, Some(1));
}

#[test]
fn recurrent_mainline_physical_weights_once() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");

    let bound = bind_checkpoint(&parsed, &bytes).expect("bind one block for two passes");

    let mut names: Vec<&str> = bound
        .weights
        .owned
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    names.extend(bound.weights.packed.iter().map(|(name, _)| name.as_str()));
    assert_eq!(
        names
            .iter()
            .filter(|name| name.starts_with("blk.0."))
            .count(),
        11
    );
    assert!(!names.iter().any(|name| name.starts_with("blk.1.")));
    assert_eq!(
        bound.layer_roots.len(),
        2,
        "one cache owner per logical pass"
    );
    assert_eq!(bound.architecture.block_count, 2, "the logical depth");
    assert_eq!(bound.architecture.kv_heads_by_layer, [1, 1]);
    let LayerCacheRoots::Attention(first) = bound.layer_roots[0] else {
        panic!("the first pass owns a cache");
    };
    let LayerCacheRoots::Attention(second) = bound.layer_roots[1] else {
        panic!("the second pass owns a cache");
    };
    assert_ne!(first, second, "weight sharing is not history sharing");
}

#[test]
fn recurrent_mainline_loader_fastpath_gate() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
    let model = LoadedModel::load(&parsed, &bytes).expect("load");

    assert_eq!(model.kv_layers.len(), model.layer_roots.len());
    assert_eq!(model.kv_layers, [(1, 2, None), (1, 2, None)]);
    assert_eq!(model.layer_count(), 2);
    let names: Vec<String> = model
        .program
        .iter()
        .filter_map(|operation| match operation {
            proxima_tensor::op::Op::Input {
                name: Some(name), ..
            } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        names.contains(&"kv_cache.0.v".to_owned()) && names.contains(&"kv_cache.1.v".to_owned())
    );
    assert!(!names.iter().any(|name| name.starts_with("blk.1.")));
}

#[test]
fn recurrent_mainline_serving_identity_before_work() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
    let model = LoadedModel::load(&parsed, &bytes).expect("load at the header's depth");
    let nodes = [model.logits_root()];
    let ask = |requested: RecurrentConfig| {
        model.forward_node_values_from_ids(
            &[0],
            &nodes,
            &ServingConfig::default().with_recurrent(requested),
        )
    };

    assert!(ask(guided(None, LoopCdMode::Disabled, 0.0)).is_ok());
    assert!(ask(guided(Some(2), LoopCdMode::Disabled, 0.0)).is_ok());
    for wrong in [
        guided(Some(1), LoopCdMode::Disabled, 0.0),
        guided(Some(2), LoopCdMode::FixedLogits, 0.5),
        guided(None, LoopCdMode::FixedHidden, 0.5),
    ] {
        let outcome = ask(wrong);
        assert!(
            matches!(outcome, Err(InteropError::InvalidRecurrentConfig { .. })),
            "{wrong:?}: {outcome:?}"
        );
    }
    let generated = model.generate_from_ids(
        &[0],
        1,
        &ServingConfig::default().with_recurrent(guided(Some(1), LoopCdMode::Disabled, 0.0)),
        &mut |_| ControlFlow::Continue(()),
    );
    assert!(
        matches!(generated, Err(InteropError::InvalidRecurrentConfig { .. })),
        "{generated:?}"
    );
}

#[test]
fn recurrent_mainline_policy_chosen_at_load_is_the_graph() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");

    let shallow = guided(Some(1), LoopCdMode::Disabled, 0.0);
    let model = LoadedModel::load_with_recurrent_config(&parsed, &bytes, shallow)
        .expect("load one pass of a two-pass checkpoint");
    near(
        &logits_of(&model, &ServingConfig::default().with_recurrent(shallow)),
        &first_pass(),
    );
    let refused = logits_of_result(
        &model,
        &ServingConfig::default().with_recurrent(guided(Some(2), LoopCdMode::Disabled, 0.0)),
    );
    assert!(
        matches!(refused, Err(InteropError::InvalidRecurrentConfig { .. })),
        "{refused:?}"
    );
    drop(model);

    let contrast = guided(None, LoopCdMode::FixedLogits, 0.5);
    let guided_model = LoadedModel::load_with_recurrent_config(&parsed, &bytes, contrast)
        .expect("load with fixed-logit guidance");
    let coefficient = 1.5 * (2.0f32 / 52.0).sqrt() - 0.5 * (2.0f32 / 27.0).sqrt();
    near(
        &logits_of(
            &guided_model,
            &ServingConfig::default().with_recurrent(contrast),
        ),
        &[3.0 * coefficient, 4.0 * coefficient],
    );

    let unusable = LoadedModel::load_with_recurrent_config(
        &parsed,
        &bytes,
        guided(Some(2), LoopCdMode::FixedHidden, f32::NAN),
    );
    assert!(matches!(
        unusable,
        Err(InteropError::InvalidRecurrentConfig { .. })
    ));
}

fn logits_of_result(
    model: &LoadedModel<'_>,
    config: &ServingConfig,
) -> Result<Vec<Vec<f32>>, InteropError> {
    model.forward_node_values_from_ids(&[0], &[model.logits_root()], config)
}

#[test]
fn recurrent_mainline_header_that_skips_the_loop_norm_is_refused() {
    let bytes = tiny_ouro_gguf(2, true);
    let parsed = parse_complete(&bytes).expect("parse the skip-norm GGUF");

    let outcome = LoadedModel::load(&parsed, &bytes);

    assert!(
        matches!(outcome, Err(InteropError::InvalidRecurrentConfig { .. })),
        "{:?}",
        outcome.err()
    );
}

#[test]
fn recurrent_mainline_production_decode_literal() {
    let bytes = tiny_ouro_gguf(2, false);
    let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
    let model = LoadedModel::load(&parsed, &bytes).expect("load");
    let mut events = Vec::new();

    let (generated, _text, _stopped) = model
        .generate_from_ids(&[0], 2, &ServingConfig::default(), &mut |event| {
            events.push((event.step, event.token_id));
            ControlFlow::Continue(())
        })
        .expect("prefill then decode through the shared loop");

    assert_eq!(
        generated,
        [1, 1],
        "4 > 3 at the first row, 5 > 0 at the second"
    );
    assert_eq!(events.len(), 3, "one prefill event, then one per token");

    let uncached = ServingConfig {
        prompt_cache: crate::serving::PromptCacheConfig::off(),
        ..ServingConfig::default()
    };
    let mut runtime = model.backend_runtime(&uncached);
    let (_, _, _, state) = model
        .run_decode_loop_through_cache(
            vec![0],
            &[],
            2,
            &uncached,
            &mut runtime,
            None,
            &mut super::LogitsSink::Discard,
            &mut super::NodeValuesSink::Discard,
            &mut |_| ControlFlow::Continue(()),
            None,
            true,
            None,
            None,
        )
        .expect("rerun keeping the cache state");
    assert_eq!(state.layer_caches.len(), 2, "one cache per logical pass");
    assert_eq!(
        state.len(),
        2,
        "the prompt token and the first generated token"
    );
}

fn last_position_logits(model: &LoadedModel<'_>, ids: &[u32]) -> Vec<f32> {
    let flat = model
        .forward_node_values_from_ids(ids, &[model.logits_root()], &ServingConfig::default())
        .expect("evaluate the state-sensitive graph")
        .remove(0);
    let vocabulary = STATE_VOCABULARY as usize;
    assert_eq!(
        flat.len() % vocabulary,
        0,
        "logits are whole vocabulary rows"
    );
    flat[flat.len() - vocabulary..].to_vec()
}

#[test]
fn recurrent_state_sensitive_fixture_last_logits_depend_on_earlier_tokens() {
    let bytes = state_sensitive_ouro_gguf(2);
    let parsed = parse_complete(&bytes).expect("parse the state-sensitive GGUF");
    let model = LoadedModel::load(&parsed, &bytes).expect("load the state-sensitive model");

    let baseline = last_position_logits(&model, &[3, 9, 5]);
    let changed_first = last_position_logits(&model, &[4, 9, 5]);
    let changed_middle = last_position_logits(&model, &[3, 10, 5]);
    let repeat = last_position_logits(&model, &[3, 9, 5]);

    let spread = |left: &[f32], right: &[f32]| {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    assert_eq!(
        spread(&baseline, &repeat),
        0.0,
        "the forward is deterministic"
    );
    assert!(
        spread(&baseline, &changed_first) > 1e-3,
        "an earlier prompt token must change the last logits, spread {}",
        spread(&baseline, &changed_first)
    );
    assert!(
        spread(&baseline, &changed_middle) > 1e-3,
        "a middle prompt token must change the last logits, spread {}",
        spread(&baseline, &changed_middle)
    );
}

#[cfg(feature = "proxima-storage")]
mod served_weights {
    use core::ops::ControlFlow;
    use core::sync::atomic::{AtomicBool, Ordering};
    use core::time::Duration;

    use super::{LoadedModel, ServingConfig, tiny_ouro_gguf};
    use crate::mapped_residency::{MappedResidency, WEIGHTS_REGION};
    use crate::memory_reclaim::{ReclaimSettings, monotonic_now};
    use crate::serving_settings::ServingSettings;
    use crate::source::CheckpointMapping;

    fn generate_ids(model: &LoadedModel<'_>) -> Vec<u32> {
        model
            .generate_from_ids(&[0], 3, &ServingConfig::default(), &mut |_| {
                ControlFlow::Continue(())
            })
            .expect("generate from the tiny recurrent checkpoint")
            .0
    }

    #[test]
    fn served_weights_unpin_release_after_idle_and_refault_identical_ids() {
        let directory = tempfile::tempdir().expect("checkpoint directory");
        let path = directory.path().join("tiny-ouro.gguf");
        std::fs::write(&path, tiny_ouro_gguf(2, false)).expect("write the tiny checkpoint");
        let cold = directory.path().join("cold");
        let reclaim = ReclaimSettings::builder()
            .cold_dir(cold.to_string_lossy().into_owned())
            .build();
        let settings = ServingSettings::builder().reclaim(reclaim).build();
        let source = CheckpointMapping::open_file(&path).expect("map the tiny checkpoint");
        let model = source
            .load_serving(&settings)
            .expect("load through serving settings");
        let residency = model
            .residency
            .as_ref()
            .expect("load_serving tracks the weights");

        let first = generate_ids(&model);

        assert_eq!(
            residency.pins(WEIGHTS_REGION),
            0,
            "an evaluation leaves nothing pinned"
        );
        assert!(residency.resident_bytes() > 0);
        let idle = monotonic_now() + Duration::from_secs(301);
        let released = residency
            .reclaim(idle, || 0.1)
            .expect("reclaim past the idle limit");
        assert_eq!(
            released
                .iter()
                .map(|entry| entry.region)
                .collect::<Vec<_>>(),
            [WEIGHTS_REGION]
        );
        assert_eq!(residency.resident_bytes(), 0);
        assert_eq!(
            generate_ids(&model),
            first,
            "released pages refault to the same ids"
        );
        assert!(residency.resident_bytes() > 0);
        assert!(
            cold.is_dir(),
            "the cold tier directory was installed from the settings"
        );
    }

    fn tracked_model(directory: &std::path::Path) -> CheckpointMapping {
        let path = directory.join("tiny-ouro.gguf");
        std::fs::write(&path, tiny_ouro_gguf(2, false)).expect("write the tiny checkpoint");
        CheckpointMapping::open_file(&path).expect("map the tiny checkpoint")
    }

    #[test]
    fn forward_entry_points_hold_the_pin_for_their_whole_call() {
        let directory = tempfile::tempdir().expect("checkpoint directory");
        let source = tracked_model(directory.path());
        let model = source
            .load_serving(&ServingSettings::builder().build())
            .expect("load through serving settings");
        let residency = model.residency.as_ref().expect("weights are tracked");

        let (_runtime, pinned) = model.pinned_runtime(&ServingConfig::default());

        assert!(pinned.is_some());
        assert_eq!(residency.pins(WEIGHTS_REGION), 1);
        drop(pinned);
        assert_eq!(residency.pins(WEIGHTS_REGION), 0);
    }

    fn pins_observed_during(
        residency: &MappedResidency<'_>,
        forward: impl Fn() -> Result<(), crate::InteropError>,
    ) {
        let observed = AtomicBool::new(false);
        let finished = AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !finished.load(Ordering::Acquire) {
                    if residency.pins(WEIGHTS_REGION) > 0 {
                        observed.store(true, Ordering::Release);
                    }
                    std::thread::yield_now();
                }
            });
            (0..100_000)
                .take_while(|_| !observed.load(Ordering::Acquire))
                .for_each(|_| forward().expect("forward on the tiny recurrent checkpoint"));
            finished.store(true, Ordering::Release);
        });
        assert!(
            observed.load(Ordering::Acquire),
            "a sampler saw the weights pinned during a forward call"
        );
        assert_eq!(residency.pins(WEIGHTS_REGION), 0);
    }

    #[test]
    fn forward_node_values_from_ids_pins_while_it_runs() {
        let directory = tempfile::tempdir().expect("checkpoint directory");
        let source = tracked_model(directory.path());
        let model = source
            .load_serving(&ServingSettings::builder().build())
            .expect("load through serving settings");
        let residency = model.residency.as_ref().expect("weights are tracked");

        pins_observed_during(residency, || {
            model
                .forward_node_values_from_ids(&[0, 1], &[], &ServingConfig::default())
                .map(|_| ())
        });
    }

    #[test]
    fn forward_logits_on_backend_pins_while_it_runs() {
        let directory = tempfile::tempdir().expect("checkpoint directory");
        let source = tracked_model(directory.path());
        let model = source
            .load_serving(&ServingSettings::builder().build())
            .expect("load through serving settings");
        let residency = model.residency.as_ref().expect("weights are tracked");

        pins_observed_during(residency, || {
            model.forward_logits_on_backend("ab", 0).map(|_| ())
        });
    }

    #[test]
    fn decode_holds_the_pin_while_tokens_are_produced() {
        let directory = tempfile::tempdir().expect("checkpoint directory");
        let source = tracked_model(directory.path());
        let model = source
            .load_serving(&ServingSettings::builder().build())
            .expect("load through serving settings");
        let residency = model.residency.as_ref().expect("weights are tracked");
        let mut pins_seen = Vec::new();

        model
            .generate_from_ids(&[0], 3, &ServingConfig::default(), &mut |_| {
                pins_seen.push(residency.pins(WEIGHTS_REGION));
                ControlFlow::Continue(())
            })
            .expect("generate from the tiny recurrent checkpoint");

        assert!(!pins_seen.is_empty() && pins_seen.iter().all(|pins| *pins > 0));
        assert_eq!(residency.pins(WEIGHTS_REGION), 0);
    }
}

#[cfg(feature = "cuda")]
mod resident_cuda_plans {
    use core::ops::ControlFlow;

    use proxima_gguf::parse_complete;

    use super::{LoadedModel, ServingConfig, state_sensitive_ouro_gguf};
    use crate::serving::{GPU_LAYERS_ALL, PromptCacheConfig};

    fn gpu_config() -> ServingConfig<'static> {
        ServingConfig {
            gpu_layers: GPU_LAYERS_ALL,
            prompt_cache: PromptCacheConfig::off(),
            ..ServingConfig::default()
        }
    }

    fn generate(model: &LoadedModel<'_>, prompt: &[u32]) -> Vec<u32> {
        model
            .generate_from_ids(prompt, 4, &gpu_config(), &mut |_| ControlFlow::Continue(()))
            .expect("generate on the tiny recurrent checkpoint through cuda")
            .0
    }

    fn fresh_ids(bytes: &[u8], prompt: &[u32]) -> Vec<u32> {
        let parsed = parse_complete(bytes).expect("parse the tiny recurrent GGUF");
        let model = LoadedModel::load(&parsed, bytes).expect("load a fresh model");
        generate(&model, prompt)
    }

    struct Decoded {
        ids: Vec<u32>,
        logits: Vec<Vec<f32>>,
        plan_hits: usize,
        plan_misses: usize,
    }

    fn decode(model: &LoadedModel<'_>, prompt: &[u32]) -> Decoded {
        let config = gpu_config();
        let mut runtime = model.backend_runtime(&config);
        let mut logits = Vec::new();
        let (ids, _, _, _) = model
            .run_decode_loop_through_cache(
                prompt.to_vec(),
                &[],
                4,
                &config,
                &mut runtime,
                None,
                &mut super::super::LogitsSink::Collect(&mut logits),
                &mut super::super::NodeValuesSink::Discard,
                &mut |_| ControlFlow::Continue(()),
                None,
                true,
                None,
                None,
            )
            .expect("decode on the tiny recurrent checkpoint through cuda");
        Decoded {
            ids,
            logits,
            plan_hits: runtime.plan_hits,
            plan_misses: runtime.plan_misses,
        }
    }

    fn assert_logits_close(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
        assert_eq!(actual.len(), expected.len(), "logit row count");
        for (row, (got, want)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(got.len(), want.len(), "row {row} width");
            for (column, (left, right)) in got.iter().zip(want).enumerate() {
                assert!(
                    (left - right).abs() <= 1e-6,
                    "logit row {row} column {column}: reused {left} vs fresh {right}"
                );
            }
        }
    }

    #[test]
    fn second_request_reuses_the_resident_plans_and_matches_a_fresh_model() {
        let bytes = state_sensitive_ouro_gguf(2);
        let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
        let model = LoadedModel::load(&parsed, &bytes).expect("load");
        let fresh_parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
        let fresh_model = LoadedModel::load(&fresh_parsed, &bytes).expect("load a fresh model");
        let long = [3_u32, 9, 5, 12, 1, 7];
        let expected = decode(&fresh_model, &long);

        let first = generate(&model, &long);
        let resident = model.cuda_resident.len();
        let second = decode(&model, &long);

        assert_eq!(first, expected.ids, "first request equals a fresh model");
        assert!(
            resident >= 1,
            "the first request left plans resident, got {resident}"
        );
        assert_eq!(second.plan_misses, 0, "the second request built no plan");
        assert!(
            second.plan_hits > 0,
            "the second request hit the resident plans"
        );
        assert!(
            !expected.logits.is_empty(),
            "the fresh model produced logits"
        );
        assert_eq!(
            second.ids, expected.ids,
            "reused plans change no sampled id"
        );
        assert_logits_close(&second.logits, &expected.logits);
    }

    #[test]
    fn a_different_prompt_length_on_the_reused_model_matches_a_fresh_model() {
        let bytes = state_sensitive_ouro_gguf(2);
        let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
        let model = LoadedModel::load(&parsed, &bytes).expect("load");
        generate(&model, &[2]);

        for prompt in [
            &[6_u32, 11, 1][..],
            &[14, 0][..],
            &[8][..],
            &[13, 4, 10, 15, 2][..],
        ] {
            assert_eq!(
                generate(&model, prompt),
                fresh_ids(&bytes, prompt),
                "prompt {prompt:?} after earlier requests"
            );
        }
    }

    #[test]
    fn a_request_after_a_carried_kv_request_does_not_see_its_rows() {
        let bytes = state_sensitive_ouro_gguf(2);
        let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
        let model = LoadedModel::load(&parsed, &bytes).expect("load");
        let long = [3_u32, 9, 5, 12, 1, 7];
        generate(&model, &long);

        assert_eq!(generate(&model, &[11]), fresh_ids(&bytes, &[11]));
        assert_eq!(generate(&model, &long), fresh_ids(&bytes, &long));
    }

    #[test]
    fn a_different_prompt_after_a_long_one_matches_a_fresh_model_ids_and_logits() {
        let bytes = state_sensitive_ouro_gguf(2);
        let parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
        let model = LoadedModel::load(&parsed, &bytes).expect("load");
        let fresh_parsed = parse_complete(&bytes).expect("parse the tiny recurrent GGUF");
        let fresh_model = LoadedModel::load(&fresh_parsed, &bytes).expect("load a fresh model");
        let first_prompt = [3_u32, 9, 5, 12, 1, 7];
        let second_prompt = [14_u32, 2, 8];
        let expected = decode(&fresh_model, &second_prompt);

        generate(&model, &first_prompt);
        let reused = decode(&model, &second_prompt);

        assert_eq!(
            reused.ids, expected.ids,
            "second prompt ids after the first"
        );
        assert_logits_close(&reused.logits, &expected.logits);
    }
}
