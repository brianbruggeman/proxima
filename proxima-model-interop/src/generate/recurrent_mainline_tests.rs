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
use super::ouro_fixture::tiny_ouro_gguf;
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
