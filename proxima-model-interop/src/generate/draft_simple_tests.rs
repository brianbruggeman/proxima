use core::mem::size_of;
use std::io::Write;

use proxima_gguf::GgmlType;

use crate::ContextLength;
use crate::lowering::{KvLayout, header_descriptor};
use crate::serving::{PromptCacheConfig, SpeculativeConfig, SpeculativeType, SpeculativeTypeSet};

use super::chunked_prefill_tests::{dense_checkpoint, dense_moe_checkpoint, gemma4_checkpoint};
use super::{
    BackendRuntime, LayerCacheState, LoadedModel, LogitsSink, NodeValuesSink, Phase, PrefixState,
    ServingConfig, SpeculativeDecodeStats, TokenEvent,
};

const PROMPT: [u32; 3] = [49, 50, 51];
const MAX_TOKENS: usize = 8;
const COMPARISON_ROUNDOFF: f32 = 64.0 * f32::EPSILON;
const DENSE_MULTI_ROW_ROUNDOFF: f32 = 0.05;

struct DecodeResult {
    generated: Vec<u32>,
    state: PrefixState,
    stats: SpeculativeDecodeStats,
    residuals: Vec<Vec<Vec<f32>>>,
    logits: Vec<Vec<f32>>,
}

fn write_draft_checkpoint(checkpoint: &[u8]) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new()
        .unwrap_or_else(|error| panic!("creates a temporary draft checkpoint: {error:?}"));
    file.write_all(checkpoint)
        .unwrap_or_else(|error| panic!("writes the draft checkpoint bytes: {error:?}"));
    file
}

fn serving_config(speculative: SpeculativeConfig<'_>) -> ServingConfig<'_> {
    ServingConfig {
        gpu_layers: 0,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        temperature: 0.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        speculative,
        ..ServingConfig::default()
    }
}

fn draft_simple_config(draft_path: &str) -> ServingConfig<'_> {
    serving_config(SpeculativeConfig {
        speculative_types: SpeculativeTypeSet::single(SpeculativeType::DraftSimple),
        draft_model_path: Some(draft_path),
        n_max: 3,
        n_min: 1,
        ..SpeculativeConfig::none()
    })
}

fn decode(
    model: &LoadedModel<'_>,
    serving: &ServingConfig<'_>,
    max_tokens: usize,
    token_override: Option<&[u32]>,
) -> DecodeResult {
    decode_prompt(model, serving, &PROMPT, max_tokens, token_override, None)
}

fn decode_prompt(
    model: &LoadedModel<'_>,
    serving: &ServingConfig<'_>,
    prompt: &[u32],
    max_tokens: usize,
    token_override: Option<&[u32]>,
    stop_after: Option<usize>,
) -> DecodeResult {
    let mut runtime = BackendRuntime::new(serving);
    let mut stats = SpeculativeDecodeStats::default();
    let mut residuals = Vec::new();
    let mut logits = Vec::new();
    let mut callback = |event: TokenEvent<'_>| {
        if matches!(event.phase, Phase::Token) && stop_after == Some(event.step + 1) {
            core::ops::ControlFlow::Break(())
        } else {
            core::ops::ControlFlow::Continue(())
        }
    };
    let (generated, _, _, state) = model
        .run_decode_loop_from_ids(
            prompt.to_vec(),
            max_tokens,
            serving,
            &mut runtime,
            token_override,
            &mut LogitsSink::Collect(&mut logits),
            &mut NodeValuesSink::Collect {
                nodes: model.layer_residual_roots(),
                steps: &mut residuals,
            },
            &mut callback,
            None,
            true,
            Some(&mut stats),
            None,
        )
        .unwrap_or_else(|error| {
            panic!("the deterministic draft-simple fixture decodes: {error:?}")
        });
    DecodeResult {
        generated,
        state,
        stats,
        residuals,
        logits,
    }
}

fn propose_rows(
    model: &LoadedModel<'_>,
    configuration: &ServingConfig<'_>,
    committed: &[u32],
    limits: &[usize],
) -> Vec<Vec<u32>> {
    model
        .with_draft_model(configuration, |proposer| {
            let propose = proposer.unwrap_or_else(|| panic!("draft-simple selects a model"));
            limits
                .iter()
                .map(|&limit| {
                    let mut proposals = vec![256];
                    propose(committed, limit, &mut proposals)?;
                    Ok(proposals)
                })
                .collect()
        })
        .unwrap_or_else(|error| panic!("the draft proposes rows: {error:?}"))
}

fn assert_payload(roundoff: f32, label: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{label}: payload lengths");
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let tolerance = roundoff * (1.0 + expected.abs());
        assert!(
            expected.is_finite() && actual.is_finite() && (expected - actual).abs() <= tolerance,
            "{label}[{index}]: expected={expected} actual={actual} tolerance={tolerance}"
        );
    }
}

fn assert_cache_state_equal(roundoff: f32, expected: &PrefixState, actual: &PrefixState) {
    assert_eq!(expected.ids, actual.ids);
    assert_eq!(expected.cached_len, actual.cached_len);
    assert_eq!(expected.layer_caches.len(), actual.layer_caches.len());
    let reference_cached_len = expected.cached_len;
    for (layer, (expected_cache, actual_cache)) in expected
        .layer_caches
        .iter()
        .zip(&actual.layer_caches)
        .enumerate()
    {
        let label = |field: &str| alloc::format!("layer {layer} {field}");
        match (expected_cache, actual_cache) {
            (LayerCacheState::Attention(expected), LayerCacheState::Attention(actual)) => {
                if let (Some(expected_ring), Some(actual_ring)) =
                    (expected.ring_geometry(), actual.ring_geometry())
                {
                    let rows = expected_ring.live_rows(reference_cached_len);
                    assert_eq!(rows, actual_ring.live_rows(reference_cached_len));
                    let mut expected_even = vec![0.0; rows * expected_ring.even_odd_row];
                    let mut expected_odd = expected_even.clone();
                    let mut expected_value = vec![0.0; rows * expected_ring.v_row];
                    let mut actual_even = expected_even.clone();
                    let mut actual_odd = expected_odd.clone();
                    let mut actual_value = expected_value.clone();
                    expected
                        .unroll_live_rows(
                            reference_cached_len,
                            &mut expected_even,
                            &mut expected_odd,
                            &mut expected_value,
                            layer,
                        )
                        .unwrap_or_else(|error| panic!("reference ring rows unroll: {error:?}"));
                    actual
                        .unroll_live_rows(
                            reference_cached_len,
                            &mut actual_even,
                            &mut actual_odd,
                            &mut actual_value,
                            layer,
                        )
                        .unwrap_or_else(|error| panic!("candidate ring rows unroll: {error:?}"));
                    assert_payload(
                        roundoff,
                        &label("ring key even"),
                        &expected_even,
                        &actual_even,
                    );
                    assert_payload(roundoff, &label("ring key odd"), &expected_odd, &actual_odd);
                    assert_payload(
                        roundoff,
                        &label("ring value"),
                        &expected_value,
                        &actual_value,
                    );
                    continue;
                }
                assert_payload(
                    roundoff,
                    &label("key even"),
                    &expected.k_even,
                    &actual.k_even,
                );
                assert_payload(roundoff, &label("key odd"), &expected.k_odd, &actual.k_odd);
                assert_payload(roundoff, &label("value"), &expected.v, &actual.v);
            }
            (
                LayerCacheState::DenseAttention(expected),
                LayerCacheState::DenseAttention(actual),
            ) => {
                assert_payload(
                    roundoff,
                    &label("first key"),
                    &expected.k_first,
                    &actual.k_first,
                );
                assert_payload(
                    roundoff,
                    &label("second key"),
                    &expected.k_second,
                    &actual.k_second,
                );
                assert_payload(
                    roundoff,
                    &label("pass key"),
                    &expected.k_pass,
                    &actual.k_pass,
                );
                assert_payload(roundoff, &label("value"), &expected.v, &actual.v);
            }
            (LayerCacheState::Ssm(expected), LayerCacheState::Ssm(actual)) => {
                assert_payload(
                    roundoff,
                    &label("convolution"),
                    &expected.conv_history,
                    &actual.conv_history,
                );
                assert_payload(
                    roundoff,
                    &label("recurrent state"),
                    &expected.state,
                    &actual.state,
                );
            }
            (LayerCacheState::ShortConv(expected), LayerCacheState::ShortConv(actual)) => {
                assert_payload(
                    roundoff,
                    &label("history"),
                    &expected.history,
                    &actual.history,
                );
                assert_eq!(expected.roll_indices, actual.roll_indices);
            }
            (LayerCacheState::SharedFromLayer, LayerCacheState::SharedFromLayer) => {}
            _ => panic!("layer {layer} cache kind changed across speculative decode"),
        }
    }
}

fn armed_target<'file>(
    parsed: &proxima_gguf::ParsedGguf,
    checkpoint: &'file [u8],
) -> LoadedModel<'file> {
    let mut descriptor = header_descriptor(parsed, KvLayout::SlidingRing)
        .unwrap_or_else(|error| panic!("the tiny target header derives a descriptor: {error:?}"));
    descriptor.speculative_verify = true;
    LoadedModel::load_with_descriptor(parsed, checkpoint, &descriptor)
        .unwrap_or_else(|error| panic!("loads the tiny target model: {error:?}"))
}

fn with_identical_checkpoint(checkpoint: Vec<u8>, run: impl FnOnce(&LoadedModel<'_>, &str)) {
    let parsed = proxima_gguf::pipe::parse_complete(&checkpoint)
        .unwrap_or_else(|error| panic!("the tiny target checkpoint parses: {error:?}"));
    let model = armed_target(&parsed, &checkpoint);
    let draft_file = write_draft_checkpoint(&checkpoint);
    let draft_path = draft_file
        .path()
        .to_str()
        .unwrap_or_else(|| panic!("temporary draft path is valid UTF-8"));
    run(&model, draft_path);
}

fn with_identical_target_and_draft(run: impl FnOnce(&LoadedModel<'_>, &str)) {
    with_identical_checkpoint(dense_checkpoint(), run);
}

fn with_zero_output_draft(run: impl FnOnce(&LoadedModel<'_>, &str)) {
    let target_checkpoint = dense_checkpoint();
    let parsed = proxima_gguf::pipe::parse_complete(&target_checkpoint)
        .unwrap_or_else(|error| panic!("the tiny target checkpoint parses: {error:?}"));
    let model = armed_target(&parsed, &target_checkpoint);
    let tensor = crate::bind::find_tensor(&parsed, "output.weight")
        .unwrap_or_else(|error| panic!("the dense fixture stores output.weight: {error:?}"));
    assert_eq!(tensor.ggml_type, GgmlType::F32);
    let range = parsed
        .tensor_data_range(tensor, target_checkpoint.len() as u64)
        .unwrap_or_else(|error| panic!("output.weight byte range fits the fixture: {error:?}"));
    let mut draft_checkpoint = target_checkpoint.clone();
    let (weights, remainder) = draft_checkpoint[range.start as usize..range.end as usize]
        .as_chunks_mut::<{ size_of::<f32>() }>();
    assert!(
        remainder.is_empty(),
        "output.weight byte range is f32 aligned"
    );
    for value in weights {
        value.copy_from_slice(&0.0f32.to_le_bytes());
    }
    let draft_file = write_draft_checkpoint(&draft_checkpoint);
    let draft_path = draft_file
        .path()
        .to_str()
        .unwrap_or_else(|| panic!("temporary draft path is valid UTF-8"));
    run(&model, draft_path);
}

#[test]
fn draft_simple_identical_model_accepts_every_draft_and_preserves_cache() {
    check_identical_checkpoint(dense_checkpoint(), DENSE_MULTI_ROW_ROUNDOFF);
}

#[test]
fn draft_simple_dense_moe_preserves_padded_cache_and_all_layer_residuals() {
    check_identical_checkpoint(dense_moe_checkpoint(), DENSE_MULTI_ROW_ROUNDOFF);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[proxima::test]
#[case::dense(false)]
#[case::gemma4(true)]
async fn draft_simple_placed_models_keep_separate_resident_weights(#[case] gemma: bool) {
    let (checkpoint, roundoff) = if gemma {
        (gemma4_checkpoint(), COMPARISON_ROUNDOFF)
    } else {
        (dense_checkpoint(), DENSE_MULTI_ROW_ROUNDOFF)
    };
    check_identical_checkpoint_on(checkpoint, crate::GPU_LAYERS_ALL, roundoff);
}

#[proxima::test]
#[case::first_accepted_token(2)]
#[case::middle_accepted_token(3)]
async fn draft_simple_gemma_ring_callback_retains_delivered_prefix(#[case] stop_after: usize) {
    with_identical_checkpoint(gemma4_checkpoint(), |model, draft_path| {
        let prompt: Vec<u32> = (49..60).collect();
        let mut plain_config = serving_config(SpeculativeConfig::none());
        plain_config.context_length = ContextLength::Within(32);
        plain_config.prompt_cache = PromptCacheConfig::off();
        let mut speculative_config = draft_simple_config(draft_path);
        speculative_config.context_length = ContextLength::Within(32);
        speculative_config.prompt_cache = PromptCacheConfig::off();
        let plain = decode_prompt(
            model,
            &plain_config,
            &prompt,
            MAX_TOKENS,
            None,
            Some(stop_after),
        );
        let speculative = decode_prompt(
            model,
            &speculative_config,
            &prompt,
            MAX_TOKENS,
            None,
            Some(stop_after),
        );
        assert_eq!(plain.generated.len(), stop_after);
        assert_eq!(speculative.generated, plain.generated);
        assert_eq!(speculative.stats.verify_steps, 1);
        assert_eq!(speculative.stats.drafted_total, 3);
        assert_eq!(speculative.stats.accepted_total, 3);
        assert_eq!(speculative.state.cached_len, prompt.len() + stop_after - 1);
        assert_eq!(plain.logits.len(), stop_after);
        assert_caches_have_live_ring(&speculative.state);
        assert_cache_state_equal(COMPARISON_ROUNDOFF, &plain.state, &speculative.state);
        let token = *plain
            .generated
            .last()
            .unwrap_or_else(|| panic!("callback delivered a token"));
        let reference_logits = resume_logits(model, &plain_config, plain.state, token);
        let candidate_logits = resume_logits(model, &plain_config, speculative.state, token);
        assert_payload(
            COMPARISON_ROUNDOFF,
            "resumed gemma logits",
            &reference_logits,
            &candidate_logits,
        );
    });
}

fn assert_caches_have_live_ring(state: &PrefixState) {
    let rings: Vec<_> = state
        .layer_caches
        .iter()
        .filter_map(|cache| match cache {
            LayerCacheState::Attention(cache) => cache.ring_geometry(),
            _ => None,
        })
        .collect();
    assert_eq!(rings.len(), 2);
    for ring in rings {
        assert_eq!(ring.window, 8);
        assert!(state.cached_len > ring.window);
        assert_eq!(ring.capacity, ring.window + 3);
    }
}

fn resume_logits(
    model: &LoadedModel<'_>,
    serving: &ServingConfig<'_>,
    state: PrefixState,
    token: u32,
) -> Vec<f32> {
    let mut runtime = BackendRuntime::new(serving);
    let mut logits = Vec::new();
    model
        .run_decode_loop_from_ids(
            vec![token],
            1,
            serving,
            &mut runtime,
            None,
            &mut LogitsSink::Collect(&mut logits),
            &mut NodeValuesSink::Discard,
            &mut |_| core::ops::ControlFlow::Continue(()),
            Some(state),
            false,
            None,
            None,
        )
        .unwrap_or_else(|error| panic!("the delivered ring prefix resumes: {error:?}"));
    assert_eq!(logits.len(), 1);
    logits
        .pop()
        .unwrap_or_else(|| panic!("the resumed row is captured"))
}

fn check_identical_checkpoint(checkpoint: Vec<u8>, roundoff: f32) {
    check_identical_checkpoint_on(checkpoint, 0, roundoff);
}

fn check_identical_checkpoint_on(checkpoint: Vec<u8>, gpu_layers: i32, roundoff: f32) {
    with_identical_checkpoint(checkpoint, |model, draft_path| {
        let mut plain_config = serving_config(SpeculativeConfig::none());
        plain_config.gpu_layers = gpu_layers;
        let mut speculative_config = draft_simple_config(draft_path);
        speculative_config.gpu_layers = gpu_layers;
        let plain = decode(model, &plain_config, MAX_TOKENS, None);
        let speculative = decode(model, &speculative_config, MAX_TOKENS, None);

        assert_eq!(speculative.generated, plain.generated);
        for layer in 0..model.layer_residual_roots().len() {
            let plain_rows: Vec<_> = plain
                .residuals
                .iter()
                .flat_map(|evaluation| evaluation[layer].iter().copied())
                .collect();
            let verified_rows: Vec<_> = speculative
                .residuals
                .iter()
                .flat_map(|evaluation| evaluation[layer].iter().copied())
                .collect();
            assert_payload(
                roundoff,
                &format!("layer {layer} residual"),
                &plain_rows,
                &verified_rows,
            );
        }
        assert_cache_state_equal(roundoff, &plain.state, &speculative.state);
        assert!(speculative.stats.verify_steps > 0);
        assert!(speculative.stats.drafted_total > 0);
        assert_eq!(
            speculative.stats.draft_simple.drafted,
            speculative.stats.drafted_total
        );
        assert_eq!(
            speculative.stats.draft_simple.accepted,
            speculative.stats.accepted_total
        );
        assert_eq!(
            speculative.stats.draft_simple.accepted,
            speculative.stats.draft_simple.drafted
        );
    });
}

#[test]
fn draft_simple_uses_distinct_output_weights_and_restores_target_cache() {
    with_zero_output_draft(|model, draft_path| {
        let plain = decode(
            model,
            &serving_config(SpeculativeConfig::none()),
            MAX_TOKENS,
            None,
        );
        let mut forced_correction = (plain.generated[1] + 1) % 257;
        if forced_correction == 0 {
            forced_correction = 1;
        }
        let overrides = [plain.generated[0], forced_correction];
        let configuration = draft_simple_config(draft_path);
        let mut committed = PROMPT.to_vec();
        committed.push(plain.generated[0]);
        let proposals = propose_rows(model, &configuration, &committed, &[3]).remove(0);
        assert_eq!(
            proposals,
            [0, 0, 0],
            "the draft uses its own zero output head"
        );
        assert_ne!(
            plain.generated[1], 0,
            "the target head distinguishes the control"
        );
        let plain = decode(
            model,
            &serving_config(SpeculativeConfig::none()),
            MAX_TOKENS,
            Some(&overrides),
        );
        let speculative = decode(
            model,
            &draft_simple_config(draft_path),
            MAX_TOKENS,
            Some(&overrides),
        );

        assert_eq!(speculative.generated, plain.generated);
        assert_eq!(speculative.generated.get(1), Some(&forced_correction));
        assert_cache_state_equal(DENSE_MULTI_ROW_ROUNDOFF, &plain.state, &speculative.state);
        assert!(speculative.stats.draft_simple.drafted > 0);
        assert!(speculative.stats.draft_simple.drafted > speculative.stats.draft_simple.accepted);
        assert_eq!(
            speculative.stats.drafted_total,
            speculative.stats.draft_simple.drafted
        );
        assert_eq!(
            speculative.stats.accepted_total,
            speculative.stats.draft_simple.accepted
        );
    });
}

#[test]
fn draft_simple_rejection_restores_branch_before_the_next_real_token() {
    with_identical_target_and_draft(|model, draft_path| {
        let plain = decode(
            model,
            &serving_config(SpeculativeConfig::none()),
            MAX_TOKENS,
            None,
        );
        let forced_correction = (plain.generated[1] + 1) % 257;
        let overrides = [plain.generated[0], forced_correction];
        let plain = decode(
            model,
            &serving_config(SpeculativeConfig::none()),
            MAX_TOKENS,
            Some(&overrides),
        );
        let mut speculative_config = draft_simple_config(draft_path);
        speculative_config.speculative.n_max = 3;
        let speculative = decode(model, &speculative_config, MAX_TOKENS, Some(&overrides));

        assert_eq!(speculative.generated, plain.generated);
        assert_eq!(speculative.generated.get(1), Some(&forced_correction));
        assert_cache_state_equal(DENSE_MULTI_ROW_ROUNDOFF, &plain.state, &speculative.state);
        assert!(speculative.stats.verify_steps >= 2);
        assert!(speculative.stats.drafted_total > speculative.stats.accepted_total);
        assert_eq!(
            speculative.stats.draft_simple.drafted,
            speculative.stats.drafted_total
        );
    });
}

#[test]
fn draft_simple_p_min_one_proposes_no_tokens() {
    with_identical_target_and_draft(|model, draft_path| {
        let plain = decode(
            model,
            &serving_config(SpeculativeConfig::none()),
            MAX_TOKENS,
            None,
        );
        let mut speculative_config = draft_simple_config(draft_path);
        speculative_config.speculative.p_min = 1.0;
        let speculative = decode(model, &speculative_config, MAX_TOKENS, None);

        assert_eq!(speculative.generated, plain.generated);
        assert_cache_state_equal(DENSE_MULTI_ROW_ROUNDOFF, &plain.state, &speculative.state);
        assert_eq!(speculative.stats.verify_steps, 0);
        assert_eq!(speculative.stats.draft_simple.drafted, 0);
    });
}

#[proxima::test]
#[case::zero(0)]
#[case::negative(-1)]
#[case::minimum(i32::MIN)]
async fn draft_simple_nonpositive_maximum_keeps_first_candidate(#[case] maximum: i32) {
    with_identical_checkpoint(gemma4_checkpoint(), |model, draft_path| {
        let prompt: Vec<u32> = (49..60).collect();
        let mut plain_config = serving_config(SpeculativeConfig::none());
        plain_config.context_length = ContextLength::Within(32);
        plain_config.prompt_cache = PromptCacheConfig::off();
        let plain = decode_prompt(model, &plain_config, &prompt, MAX_TOKENS, None, Some(2));
        let mut configuration = draft_simple_config(draft_path);
        configuration.context_length = ContextLength::Within(32);
        configuration.prompt_cache = PromptCacheConfig::off();
        configuration.speculative.n_max = maximum;
        configuration.speculative.n_min = 0;

        let mut committed = prompt.clone();
        committed.push(plain.generated[0]);
        let proposals = propose_rows(model, &configuration, &committed, &[0, 3]);
        assert!(
            proposals[0].is_empty(),
            "zero request capacity skips drafting"
        );
        assert_eq!(
            proposals[1],
            [plain.generated[1]],
            "a nonpositive configured cap still keeps the first candidate"
        );

        let speculative = decode_prompt(model, &configuration, &prompt, MAX_TOKENS, None, Some(2));
        assert_eq!(speculative.generated, plain.generated);
        assert_eq!(speculative.stats.verify_steps, 1);
        assert_eq!(speculative.stats.draft_simple.drafted, 1);
        assert_eq!(speculative.stats.draft_simple.accepted, 1);
        assert_cache_state_equal(COMPARISON_ROUNDOFF, &plain.state, &speculative.state);
        let rings: Vec<_> = speculative
            .state
            .layer_caches
            .iter()
            .filter_map(|cache| match cache {
                LayerCacheState::Attention(cache) => cache.ring_geometry(),
                _ => None,
            })
            .collect();
        assert_eq!(rings.len(), 2);
        for ring in rings {
            assert_eq!(ring.window, 8);
            assert_eq!(ring.capacity, 9);
        }
    });
}

#[test]
fn draft_simple_negative_minimum_clears_bounded_proposals() {
    with_identical_target_and_draft(|model, draft_path| {
        let plain = decode(
            model,
            &serving_config(SpeculativeConfig::none()),
            MAX_TOKENS,
            None,
        );
        let mut configuration = draft_simple_config(draft_path);
        configuration.speculative.n_min = -1;
        let mut committed = PROMPT.to_vec();
        committed.push(plain.generated[0]);
        let proposals = propose_rows(model, &configuration, &committed, &[3]);
        assert!(
            proposals[0].is_empty(),
            "a negative minimum clears the bounded proposal"
        );
        let speculative = decode(model, &configuration, MAX_TOKENS, None);
        assert_eq!(speculative.generated, plain.generated);
        assert_eq!(speculative.stats.verify_steps, 0);
        assert_eq!(speculative.stats.draft_simple.drafted, 0);
        assert_cache_state_equal(DENSE_MULTI_ROW_ROUNDOFF, &plain.state, &speculative.state);
    });
}

#[test]
fn draft_simple_clears_proposals_when_n_min_exceeds_n_max() {
    with_identical_target_and_draft(|model, draft_path| {
        let plain = decode(
            model,
            &serving_config(SpeculativeConfig::none()),
            MAX_TOKENS,
            None,
        );
        let mut speculative_config = draft_simple_config(draft_path);
        speculative_config.speculative.n_max = 2;
        speculative_config.speculative.n_min = 3;
        let speculative = decode(model, &speculative_config, MAX_TOKENS, None);

        assert_eq!(speculative.generated, plain.generated);
        assert_cache_state_equal(DENSE_MULTI_ROW_ROUNDOFF, &plain.state, &speculative.state);
        assert_eq!(speculative.stats.verify_steps, 0);
        assert_eq!(speculative.stats.draft_simple.drafted, 0);
    });
}
