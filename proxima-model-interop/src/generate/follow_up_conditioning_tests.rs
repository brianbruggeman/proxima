#![allow(clippy::expect_used)]

use core::ops::ControlFlow;

use super::prompt_cache_real_model_tests::{
    cached_config, encode_continuation, encode_opening, run_cached, uncached_config, with_model,
};
use super::*;
use crate::serving::SpeculativeConfig;

const USER_TURN_OPENER: &str = "<turn|>\n<|turn>user\n";
const QUESTION: &str = "What's the difference between a mutex and a semaphore?";

fn first_step_logits(
    model: &LoadedModel<'_>,
    ids: Vec<u32>,
    seed: Option<PrefixState>,
) -> Vec<f32> {
    let config = uncached_config(SpeculativeConfig::none());
    let effective = model
        .effective_serving_config(&config)
        .expect("the uncached config resolves");
    let mut runtime = BackendRuntime::new(&effective);
    let mut collected: Vec<Vec<f32>> = Vec::new();
    model
        .run_decode_loop_from_ids(
            ids,
            1,
            &effective,
            &mut runtime,
            None,
            &mut LogitsSink::Collect(&mut collected),
            &mut NodeValuesSink::Discard,
            &mut |_event| ControlFlow::Continue(()),
            seed,
            true,
            None,
            None,
        )
        .expect("decode one token");
    collected
        .into_iter()
        .next()
        .expect("one step produces one row of logits")
}

/// The drafter's seed is the entry the answer left, and it forwards only the
/// last token of `prompt + answer + end-of-turn + user-turn opener`. What it
/// samples from must be what a full prefill of that whole sequence samples
/// from: the draft is conditioned on the turn boundary, not on the answer
/// alone.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn the_drafter_samples_from_the_distribution_a_full_prefill_of_the_boundary_gives() {
    with_model(|model| {
        let config = cached_config(SpeculativeConfig::none());
        let prompt = encode_opening(
            model,
            &format!("<|turn>user\n{QUESTION}<turn|>\n<|turn>model\n"),
        );
        let answer = run_cached(model, config, &prompt);
        let opener = encode_continuation(model, USER_TURN_OPENER);
        let base: Vec<u32> = prompt
            .iter()
            .chain(&answer.generated)
            .chain(&opener)
            .copied()
            .collect();
        model.prewarm(&base, &config).expect("prewarm the boundary");
        let (_, widths) = model
            .declared_layer_cache_names_and_widths()
            .expect("layer widths");
        let effective = model
            .effective_serving_config(&config)
            .expect("the config resolves");
        let key = model.cache_key(&effective, &BackendRuntime::new(&effective), None);
        let (entry, _) = model
            .prompt_cache
            .lock()
            .take_for_prewarm(&base, &key, &widths, 0);
        let seed = entry.expect("the boundary entry is cached").state;
        assert_eq!(seed.cached_len + 1, base.len());

        let seeded = first_step_logits(model, vec![*base.last().expect("a base")], Some(seed));
        let fresh = first_step_logits(model, base, None);

        let largest_difference = seeded
            .iter()
            .zip(&fresh)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        println!("CONDITIONING seeded_vs_full_prefill_max_abs_logit_diff={largest_difference}");
        assert_eq!(seeded.len(), fresh.len());
        assert!(
            largest_difference < 1e-2,
            "the seeded draft step differs from a full prefill by {largest_difference}"
        );
    });
}
