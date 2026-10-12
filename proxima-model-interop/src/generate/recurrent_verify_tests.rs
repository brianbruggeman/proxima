// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

use alloc::vec::Vec;
use core::ops::ControlFlow;

use proxima_gguf::GgmlType;
use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::ModelDescriptor;

use crate::error::InteropError;
use crate::lowering::{KvLayout, header_descriptor};
use crate::serving::{NgramMapParams, SpeculativeType, SpeculativeTypeSet};
use crate::{ContextLength, SpeculativeConfig};

use super::drafter::DrafterSet;
use super::recurrent_checkpoint_fixture::{
    HEAD_DIM, LAYERS, QUERY_HEADS, recurrent_checkpoint, recurrent_checkpoint_shaped,
    recurrent_checkpoint_with_eos,
};
use super::{
    BackendRuntime, LayerCacheState, LoadedModel, LogitsSink, NodeValuesSink, Phase, PrefixState,
    ServingConfig, SpeculativeDecodeStats, TokenEvent,
};

const PROMPT: [u32; 3] = [49, 50, 51];
const COMPARISON_ROUNDOFF: f32 = 64.0 * f32::EPSILON;

fn load_armed<'file>(
    parsed: &ParsedGguf,
    bytes: &'file [u8],
) -> Result<LoadedModel<'file>, InteropError> {
    let decode = header_descriptor(parsed, KvLayout::Full)?;
    let armed = ModelDescriptor {
        speculative_verify: true,
        ..decode
    };
    LoadedModel::load_with_descriptor(parsed, bytes, &armed)
}

fn config() -> ServingConfig<'static> {
    ServingConfig {
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        context_length: ContextLength::Within(32),
        gpu_layers: 0,
        reasoning_budget: 0,
        speculative: SpeculativeConfig::none(),
        ..ServingConfig::default()
    }
}

struct DecodeCapture {
    generated: Vec<u32>,
    state: PrefixState,
    logits: Vec<Vec<f32>>,
    stats: SpeculativeDecodeStats,
}

fn decode_boundary(
    model: &LoadedModel<'_>,
    serving: &ServingConfig<'_>,
    tokens: &[u32],
    speculative: bool,
) -> DecodeCapture {
    decode_with_stop(model, serving, tokens, speculative, tokens.len())
}

fn decode_with_stop(
    model: &LoadedModel<'_>,
    serving: &ServingConfig<'_>,
    tokens: &[u32],
    speculative: bool,
    stop_after: usize,
) -> DecodeCapture {
    let mut runtime = BackendRuntime::new(serving);
    let mut logits = Vec::new();
    let mut stats = SpeculativeDecodeStats::default();
    let mut callback = |event: TokenEvent<'_>| {
        if matches!(event.phase, Phase::Token) && event.step + 1 == stop_after {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    };
    let (generated, _, _, state) = model
        .run_decode_loop_from_ids(
            PROMPT.to_vec(),
            tokens.len().max(4),
            serving,
            &mut runtime,
            Some(tokens),
            &mut LogitsSink::Collect(&mut logits),
            &mut NodeValuesSink::Discard,
            &mut callback,
            None,
            false,
            Some(&mut stats),
            speculative.then_some(2),
        )
        .unwrap_or_else(|error| {
            panic!("the fixture reaches the requested verify boundary: {error:?}")
        });
    DecodeCapture {
        generated,
        state,
        logits,
        stats,
    }
}

fn assert_payload(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len(), "{label}: payload lengths");
    assert!(!reference.is_empty(), "{label}: payload must be observed");
    for (index, (&expected, &found)) in reference.iter().zip(candidate).enumerate() {
        let tolerance = COMPARISON_ROUNDOFF * (1.0 + expected.abs());
        assert!(
            expected.is_finite() && found.is_finite() && (expected - found).abs() <= tolerance,
            "{label}[{index}]: expected={expected} found={found} tolerance={tolerance}",
        );
    }
}

fn assert_caches(reference: &PrefixState, candidate: &PrefixState) {
    assert_eq!(reference.ids, candidate.ids);
    assert_eq!(reference.cached_len, candidate.cached_len);
    assert_eq!(reference.layer_caches.len(), LAYERS as usize);
    assert_eq!(candidate.layer_caches.len(), LAYERS as usize);
    for (layer, (expected, found)) in reference
        .layer_caches
        .iter()
        .zip(&candidate.layer_caches)
        .enumerate()
    {
        match (expected, found) {
            (LayerCacheState::Ssm(expected), LayerCacheState::Ssm(found)) => {
                assert!(
                    expected.state.iter().any(|value| *value != 0.0),
                    "recurrence must advance"
                );
                assert_payload(
                    &alloc::format!("layer {layer} recurrence"),
                    &expected.state,
                    &found.state,
                );
                assert_payload(
                    &alloc::format!("layer {layer} convolution"),
                    &expected.conv_history,
                    &found.conv_history,
                );
            }
            (LayerCacheState::DenseAttention(expected), LayerCacheState::DenseAttention(found)) => {
                assert_payload(
                    &alloc::format!("layer {layer} first key"),
                    &expected.k_first,
                    &found.k_first,
                );
                assert_payload(
                    &alloc::format!("layer {layer} second key"),
                    &expected.k_second,
                    &found.k_second,
                );
                assert_payload(
                    &alloc::format!("layer {layer} pass key"),
                    &expected.k_pass,
                    &found.k_pass,
                );
                assert_payload(
                    &alloc::format!("layer {layer} value"),
                    &expected.v,
                    &found.v,
                );
            }
            _ => panic!("layer {layer} must preserve its fixture cache kind"),
        }
    }
}

fn resume_one(
    model: &LoadedModel<'_>,
    serving: &ServingConfig<'_>,
    state: PrefixState,
    token: u32,
) -> DecodeCapture {
    let mut runtime = BackendRuntime::new(serving);
    let mut logits = Vec::new();
    let mut stats = SpeculativeDecodeStats::default();
    let (generated, _, _, state) = model
        .run_decode_loop_from_ids(
            alloc::vec![token],
            1,
            serving,
            &mut runtime,
            None,
            &mut LogitsSink::Collect(&mut logits),
            &mut NodeValuesSink::Discard,
            &mut |_| ControlFlow::Continue(()),
            Some(state),
            false,
            Some(&mut stats),
            None,
        )
        .unwrap_or_else(|error| {
            panic!("the retained prefix resumes for one real sampled token: {error:?}")
        });
    DecodeCapture {
        generated,
        state,
        logits,
        stats,
    }
}

fn check_recurrent_prefix(bytes: &[u8], accepted: usize) {
    let bytes = bytes.to_vec();
    let parsed = proxima_gguf::pipe::parse_complete(&bytes)
        .unwrap_or_else(|error| panic!("the recurrent fixture parses: {error:?}"));
    let model = load_armed(&parsed, &bytes)
        .unwrap_or_else(|error| panic!("the recurrent fixture loads: {error:?}"));
    let serving = config();
    let mut tokens = alloc::vec![65; accepted + 1];
    tokens.push(66);
    let reference = decode_boundary(&model, &serving, &tokens, false);
    let candidate = decode_boundary(&model, &serving, &tokens, true);
    assert_eq!(reference.generated, tokens);
    assert_eq!(candidate.generated, tokens);
    assert_eq!(reference.stats.verify_steps, 0);
    assert_eq!(candidate.stats.verify_steps, 1);
    assert_eq!(candidate.stats.drafted_total, 2);
    assert_eq!(candidate.stats.accepted_total, accepted as u64);
    assert_eq!(candidate.state.cached_len, PROMPT.len() + accepted + 1);
    assert_caches(&reference.state, &candidate.state);

    let reference_next = resume_one(&model, &serving, reference.state, 66);
    let candidate_next = resume_one(&model, &serving, candidate.state, 66);
    assert_eq!(reference_next.logits.len(), 1);
    assert_eq!(candidate_next.logits.len(), 1);
    assert_payload(
        "next decode logits",
        &reference_next.logits[0],
        &candidate_next.logits[0],
    );
    assert_eq!(reference_next.generated, candidate_next.generated);
    assert_caches(&reference_next.state, &candidate_next.state);

    let mut continued_tokens = tokens.clone();
    while continued_tokens.len() < 5 {
        continued_tokens.push(66 + (continued_tokens.len() - accepted - 1) as u32);
    }
    let reference_continued = decode_boundary(&model, &serving, &continued_tokens, false);
    let candidate_continued = decode_boundary(&model, &serving, &continued_tokens, true);
    assert_eq!(reference_continued.generated, continued_tokens);
    assert_eq!(candidate_continued.generated, continued_tokens);
    assert_eq!(
        candidate_continued.stats.verify_steps,
        (3 - accepted) as u64
    );
    assert_eq!(candidate_continued.stats.accepted_total, accepted as u64);
    assert_eq!(reference_continued.logits.len(), 5);
    assert_eq!(candidate_continued.logits.len(), 2);
    assert_payload(
        "uninterrupted final decode logits",
        &reference_continued.logits[4],
        &candidate_continued.logits[1],
    );
    assert_caches(&reference_continued.state, &candidate_continued.state);

    let mut changed_prompt = PROMPT;
    changed_prompt[0] = 52;
    let mut control_runtime = BackendRuntime::new(&serving);
    let mut control_logits = Vec::new();
    model
        .run_decode_loop_from_ids(
            changed_prompt.to_vec(),
            1,
            &serving,
            &mut control_runtime,
            None,
            &mut LogitsSink::Collect(&mut control_logits),
            &mut NodeValuesSink::Discard,
            &mut |_| ControlFlow::Continue(()),
            None,
            false,
            None,
            None,
        )
        .unwrap_or_else(|error| panic!("the changed-input control evaluates: {error:?}"));
    assert_eq!(control_logits.len(), 1);
    assert!(
        reference.logits[0]
            .iter()
            .zip(&control_logits[0])
            .any(|(&expected, &found)| {
                (expected - found).abs() > COMPARISON_ROUNDOFF * (1.0 + expected.abs())
            }),
        "changing the prompt must move observed logits beyond the comparison tolerance"
    );
}

#[proxima::test]
#[case::qwen35_first_rejection(false, 0)]
#[case::qwen35_partial_acceptance(false, 1)]
#[case::qwen35_full_acceptance(false, 2)]
#[case::qwen35moe_first_rejection(true, 0)]
#[case::qwen35moe_partial_acceptance(true, 1)]
#[case::qwen35moe_full_acceptance(true, 2)]
async fn rollback_recurrent_retains_accepted_prefix(#[case] routed: bool, #[case] accepted: usize) {
    check_recurrent_prefix(&recurrent_checkpoint(routed), accepted);
}

fn check_cancelled_prefix(routed: bool, delivered: usize) {
    let bytes = recurrent_checkpoint(routed);
    let parsed = proxima_gguf::pipe::parse_complete(&bytes)
        .unwrap_or_else(|error| panic!("the recurrent fixture parses: {error:?}"));
    let model = load_armed(&parsed, &bytes)
        .unwrap_or_else(|error| panic!("the recurrent fixture loads: {error:?}"));
    let serving = config();
    let tokens = [65, 65, 65, 66];
    let reference = decode_with_stop(&model, &serving, &tokens, false, delivered);
    let candidate = decode_with_stop(&model, &serving, &tokens, true, delivered);
    assert_eq!(reference.generated, &tokens[..delivered]);
    assert_eq!(candidate.generated, reference.generated);
    assert_eq!(candidate.stats.verify_steps, 1);
    assert_eq!(candidate.stats.accepted_total, 2);
    assert_eq!(candidate.state.cached_len, PROMPT.len() + delivered - 1);
    assert_eq!(candidate.state.ids.len(), candidate.state.cached_len);
    assert_caches(&reference.state, &candidate.state);

    let last_delivered = tokens[delivered - 1];
    let reference_next = resume_one(&model, &serving, reference.state, last_delivered);
    let candidate_next = resume_one(&model, &serving, candidate.state, last_delivered);
    assert_payload(
        "cancelled next logits",
        &reference_next.logits[0],
        &candidate_next.logits[0],
    );
    assert_eq!(reference_next.generated, candidate_next.generated);
    assert_caches(&reference_next.state, &candidate_next.state);
}

#[proxima::test]
#[case::qwen35_first(false, 2)]
#[case::qwen35_middle(false, 3)]
#[case::qwen35moe_first(true, 2)]
#[case::qwen35moe_middle(true, 3)]
async fn rollback_recurrent_callback_retains_delivered_prefix(
    #[case] routed: bool,
    #[case] delivered: usize,
) {
    check_cancelled_prefix(routed, delivered);
}

fn check_matched_eos(routed: bool) {
    let bytes = recurrent_checkpoint_with_eos(routed, Some(65));
    let parsed = proxima_gguf::pipe::parse_complete(&bytes)
        .unwrap_or_else(|error| panic!("the eos fixture parses: {error:?}"));
    let model = load_armed(&parsed, &bytes)
        .unwrap_or_else(|error| panic!("the eos fixture loads: {error:?}"));
    let prompt = [48, 49, 50, 51, 65, 66, 49, 50];
    let overrides = [51, 65, 67, 68];
    let mut outcomes = Vec::new();
    for speculative in [false, true] {
        let mut serving = config();
        if speculative {
            serving.speculative = SpeculativeConfig {
                speculative_types: SpeculativeTypeSet::single(SpeculativeType::NgramSimple),
                ngram_simple: NgramMapParams {
                    size_n: 2,
                    size_m: 2,
                    min_hits: 1,
                },
                ..SpeculativeConfig::none()
            };
            let mut drafter = DrafterSet::build(&serving.speculative, 32);
            drafter.begin(&prompt);
            let mut drafts = Vec::new();
            drafter.draft(&prompt, overrides[0], 2, &mut drafts);
            assert_eq!(
                drafts,
                [65, 66],
                "the first draft is the selected eos token"
            );
        }
        let mut runtime = BackendRuntime::new(&serving);
        let mut stats = SpeculativeDecodeStats::default();
        let (generated, _, stopped, state) = model
            .run_decode_loop_from_ids(
                prompt.to_vec(),
                overrides.len(),
                &serving,
                &mut runtime,
                Some(&overrides),
                &mut LogitsSink::Discard,
                &mut NodeValuesSink::Discard,
                &mut |_| ControlFlow::Continue(()),
                None,
                false,
                Some(&mut stats),
                None,
            )
            .unwrap_or_else(|error| {
                panic!("the matching eos draft terminates decoding: {error:?}")
            });
        assert!(stopped);
        assert_eq!(generated, [51]);
        assert_eq!(state.cached_len, prompt.len() + 1);
        assert_eq!(state.ids.len(), state.cached_len);
        assert_eq!(stats.verify_steps, u64::from(speculative));
        assert_eq!(stats.drafted_total, if speculative { 2 } else { 0 });
        assert_eq!(
            stats.accepted_total,
            u64::from(speculative),
            "the matched eos draft counts as accepted; the cache keeps only the rows delivered before it"
        );
        outcomes.push(state);
    }
    let candidate = outcomes
        .pop()
        .unwrap_or_else(|| panic!("speculative eos state"));
    let reference = outcomes
        .pop()
        .unwrap_or_else(|| panic!("ordinary eos state"));
    assert_caches(&reference, &candidate);
    let serving = config();
    let reference_next = resume_one(&model, &serving, reference, 66);
    let candidate_next = resume_one(&model, &serving, candidate, 66);
    assert_payload(
        "eos next logits",
        &reference_next.logits[0],
        &candidate_next.logits[0],
    );
    assert_caches(&reference_next.state, &candidate_next.state);
}

#[proxima::test]
#[case::qwen35(false)]
#[case::qwen35moe(true)]
async fn rollback_recurrent_matched_eos_retains_forwarded_prefix(#[case] routed: bool) {
    check_matched_eos(routed);
}

#[proxima::test]
#[case::qwen35(false)]
#[case::qwen35moe(true)]
async fn a_descriptor_load_keeps_the_rotated_width_of_a_partial_rotary_hybrid(
    #[case] routed: bool,
) {
    let bytes = recurrent_checkpoint(routed);
    let parsed = proxima_gguf::pipe::parse_complete(&bytes)
        .unwrap_or_else(|error| panic!("the recurrent fixture parses: {error:?}"));

    let model = load_armed(&parsed, &bytes)
        .unwrap_or_else(|error| panic!("the recurrent fixture loads: {error:?}"));

    assert_eq!(
        model.architecture.head_dim,
        (HEAD_DIM / 2) as u32,
        "the fixture rotates half of its head channels, and the rope table is built that wide"
    );
}

#[proxima::test]
#[case::first_rejection(0)]
#[case::partial_acceptance(1)]
#[case::full_acceptance(2)]
async fn rollback_recurrent_with_one_value_head_per_key_head(#[case] accepted: usize) {
    check_recurrent_prefix(
        &recurrent_checkpoint_shaped(false, None, 2, 4, false, (QUERY_HEADS, 1)),
        accepted,
    );
}

#[proxima::test]
#[case::first_rejection(0)]
#[case::partial_acceptance(1)]
#[case::full_acceptance(2)]
async fn rollback_recurrent_over_q8_0_weights(#[case] accepted: usize) {
    check_recurrent_prefix(
        &recurrent_checkpoint_shaped(false, None, 1, 3, true, (QUERY_HEADS, 1)),
        accepted,
    );
}

#[proxima::test]
#[case::first_rejection(0)]
#[case::partial_acceptance(1)]
#[case::full_acceptance(2)]
async fn rollback_recurrent_over_grouped_query_q8_0_weights(#[case] accepted: usize) {
    check_recurrent_prefix(
        &recurrent_checkpoint_shaped(false, None, 1, 3, true, (4, 2)),
        accepted,
    );
}
