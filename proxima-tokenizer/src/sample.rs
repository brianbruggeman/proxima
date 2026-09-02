//! Decoding: turns a model's raw logits into the token id the model
//! selects. Pairs with [`crate::decode`] to complete the logits -> id ->
//! text path. Every step here -- [`greedy_pick`]'s argmax fold, each
//! [`sample_next_token`] filter, the softmax, the weighted draw -- is a
//! free function over a slice (`&[f32]`, `&mut Vec<(u32, f32)>`,
//! `&[u32]`), no pipe wrapper: a sampler is a pure reduction, and the pipe
//! algebra already expresses that as a function, not a form (this
//! module's own long-standing convention -- [`greedy_pick`] never needed
//! one either). [`SamplingConfig`] is the one exception, and it earns
//! its existence a different way: it is plain data grouping the filter
//! chain's seven knobs so [`sample_next_token`]'s signature does not trip
//! `clippy::too_many_arguments`, not a behavior-carrying type -- see its
//! own doc for why it cannot be [`crate::TokenizerError`]'s caller's own
//! `ServingConfig` instead. Lives here rather than a dedicated inference
//! crate because it is the other half of the same boundary
//! [`crate::decode`] already owns: [`crate::decode`] turns ids into text,
//! this module turns logits into an id.
//!
//! # Filter chain order
//!
//! [`sample_next_token`] applies, in order: repetition penalty, top-k,
//! top-p (nucleus), min-p, temperature, softmax, weighted sample. This is
//! upstream llama.cpp's own default chain order
//! (`common/common.h:162-172`'s `samplers` default vector -- penalties,
//! dry, top_n_sigma, top_k, typical_p, top_p, min_p, xtc, temperature,
//! then `llama_sampler_init_dist` appended after the loop at
//! `common/sampling.cpp:276`), read down to exactly the five filters this
//! module implements (dry/top_n_sigma/typical_p/xtc are not implemented
//! here) with every other filter's relative order preserved.
//!
//! Each filter's formula is upstream's own, cited at its own function:
//! penalties (`src/llama-sampling.cpp:1650-1681`), top-k
//! (`src/llama-sampling.cpp:228-267`), top-p (`:708-733`), min-p
//! (`:775-806`), temperature (`:179-201`), softmax (`:203-226`), the
//! weighted draw (`:576-582` -- `llama_sample_dist` over
//! `std::discrete_distribution`, reimplemented here as an explicit
//! cumulative-probability walk against a uniform `[0, 1)` draw, which is
//! the same algorithm `std::discrete_distribution` documents itself as
//! using).
//!
//! # Determinism
//!
//! [`sample_next_token`] takes the caller's [`fastrand::Rng`] by `&mut
//! reference` rather than owning or seeding one itself -- the same seeded
//! [`fastrand::Rng`] this workspace already uses for every other
//! deterministic-by-seed pipe (`proxima-primitives/src/pipe/mutate.rs:40`,
//! `retry.rs:408`, `when.rs:54`), reused here rather than adding a second
//! PRNG dependency. A caller constructs one `fastrand::Rng::with_seed`
//! once per generation call and threads it through every step, mirroring
//! upstream's own `llama_sampler_dist`, which seeds one `std::mt19937`
//! once per sampler chain and draws from it across the whole decode
//! (`src/llama-sampling.cpp:617-627`) rather than reseeding per token.
//! Same seed, same logits, same recent-token window -> same token, every
//! time -- proved in this module's `determinism` test.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use fastrand::Rng;

/// Returns the index of the largest value in `logits`, or `None` if
/// `logits` is empty. Ties resolve to the lowest index (the first
/// occurrence of the maximum) -- deterministic, and matching the
/// convention most reference decoders use for greedy argmax.
#[must_use]
pub fn greedy_pick(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .fold(None, |best, (index, &value)| match best {
            Some((_, best_value)) if value <= best_value => best,
            _ => Some((index, value)),
        })
        .map(|(index, _)| index as u32)
}

/// The token-sampling filter chain's seven knobs, grouped so
/// [`sample_next_token`] takes one argument instead of seven. Not a
/// duplicate of a caller's own richer config (e.g.
/// `proxima-model-interop::serving::ServingConfig`, which also carries
/// `-c`/`-np`/`-ctk`/... knobs this crate has no business knowing about)
/// -- this crate cannot depend on that crate (it would be a reverse
/// dependency, `proxima-model-interop` already depends on
/// `proxima-tokenizer`), so a caller destructures its own config into
/// this one at the call site instead.
///
/// Each field's own disabling value is upstream's own convention:
/// `top_k <= 0` disables the filter (`src/llama-sampling.cpp:234-236`);
/// `top_p >= 1.0` disables nucleus filtering (`:711-713`); `min_p <= 0.0`
/// disables the min-p filter (`:778`); a neutral penalty triple
/// (`repeat_penalty` at `1.0`, `frequency_penalty` and `presence_penalty`
/// both at `0.0`) disables the penalty filter (`:1653-1655`);
/// `temperature <= 0.0` collapses the chain to plain argmax rather than
/// dividing by (or sampling from) a distribution (`:180-196`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    /// `--temp`. `<= 0.0` samples greedily (exact argmax over whatever
    /// survives the earlier filters); otherwise every surviving logit is
    /// divided by this value before the softmax.
    pub temperature: f32,
    /// `--top-k`. `<= 0` keeps every candidate (upstream: "use vocab
    /// size"); otherwise keeps only the `top_k` highest-logit candidates.
    pub top_k: i32,
    /// `--top-p` (nucleus sampling). `>= 1.0` disables the filter;
    /// otherwise keeps the smallest prefix (by descending probability)
    /// whose cumulative probability reaches `top_p`.
    pub top_p: f32,
    /// `--min-p`. `<= 0.0` disables the filter; otherwise drops every
    /// candidate whose logit is below `max_logit + ln(min_p)` (the logit
    /// whose probability, relative to the top candidate, is under
    /// `min_p`).
    pub min_p: f32,
    /// `--repeat-penalty`. `1.0` disables the multiplicative half of the
    /// penalty filter (recently-seen tokens' logits are multiplied by
    /// this value when negative, divided when positive).
    pub repeat_penalty: f32,
    /// `--frequency-penalty`. `0.0` disables the per-occurrence-count
    /// subtractive penalty.
    pub frequency_penalty: f32,
    /// `--presence-penalty`. `0.0` disables the flat once-per-distinct-
    /// recent-token subtractive penalty.
    pub presence_penalty: f32,
}

impl Default for SamplingConfig {
    /// Every filter disabled -- [`sample_next_token`] against this config
    /// collapses to exactly [`greedy_pick`]'s own argmax fold over the raw
    /// logits (proved in this module's
    /// `default_config_matches_greedy_pick_exactly` test), because every
    /// filter's no-op guard above is satisfied and the candidate list
    /// never gets reordered before the `temperature <= 0.0` argmax
    /// collapse runs over it in original (vocab-index) order -- the same
    /// order [`greedy_pick`]'s own fold walks.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

fn apply_repetition_penalty(
    candidates: &mut [(u32, f32)],
    recent_tokens: &[u32],
    config: SamplingConfig,
) {
    if recent_tokens.is_empty()
        || (config.repeat_penalty == 1.0
            && config.frequency_penalty == 0.0
            && config.presence_penalty == 0.0)
    {
        return;
    }
    let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
    for &token in recent_tokens {
        *counts.entry(token).or_insert(0) += 1;
    }
    for candidate in candidates.iter_mut() {
        let Some(&count) = counts.get(&candidate.0) else {
            continue;
        };
        if candidate.1 <= 0.0 {
            candidate.1 *= config.repeat_penalty;
        } else {
            candidate.1 /= config.repeat_penalty;
        }
        candidate.1 -= (count as f32) * config.frequency_penalty
            + if count > 0 {
                config.presence_penalty
            } else {
                0.0
            };
    }
}

fn sort_candidates_descending(candidates: &mut [(u32, f32)]) {
    candidates.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
}

fn apply_top_k(candidates: &mut Vec<(u32, f32)>, top_k: i32) {
    if top_k <= 0 {
        return;
    }
    let keep = (top_k as usize).min(candidates.len());
    sort_candidates_descending(candidates);
    candidates.truncate(keep);
}

fn softmax_probabilities(candidates: &[(u32, f32)]) -> Vec<f32> {
    let max_logit = candidates
        .iter()
        .map(|candidate| candidate.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let exponentiated: Vec<f32> = candidates
        .iter()
        .map(|candidate| libm::expf(candidate.1 - max_logit))
        .collect();
    let sum: f32 = exponentiated.iter().sum();
    exponentiated.into_iter().map(|value| value / sum).collect()
}

fn apply_top_p(candidates: &mut Vec<(u32, f32)>, top_p: f32) {
    if top_p >= 1.0 || candidates.is_empty() {
        return;
    }
    sort_candidates_descending(candidates);
    let probabilities = softmax_probabilities(candidates);
    let mut keep = candidates.len();
    let mut cumulative = 0.0f32;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if cumulative >= top_p {
            keep = index + 1;
            break;
        }
    }
    candidates.truncate(keep);
}

fn apply_min_p(candidates: &mut Vec<(u32, f32)>, min_p: f32) {
    if min_p <= 0.0 || candidates.is_empty() {
        return;
    }
    let max_logit = candidates
        .iter()
        .map(|candidate| candidate.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let threshold = max_logit + libm::logf(min_p);
    candidates.retain(|candidate| candidate.1 >= threshold);
}

fn collapse_to_argmax(candidates: &[(u32, f32)]) -> Option<u32> {
    candidates
        .iter()
        .fold(None, |best, &(id, logit)| match best {
            Some((_, best_logit)) if logit <= best_logit => best,
            _ => Some((id, logit)),
        })
        .map(|(id, _)| id)
}

fn sample_from_distribution(
    candidates: &[(u32, f32)],
    probabilities: &[f32],
    rng: &mut Rng,
) -> Option<u32> {
    let draw = rng.f32();
    let mut cumulative = 0.0f32;
    for (candidate, probability) in candidates.iter().zip(probabilities) {
        cumulative += probability;
        if draw < cumulative {
            return Some(candidate.0);
        }
    }
    candidates.last().map(|candidate| candidate.0)
}

/// Turns one step's raw logits into a token id through the filter chain
/// this module's own doc names: repetition penalty, top-k, top-p, min-p,
/// temperature, softmax, weighted sample. `recent_tokens` is the
/// caller-owned tail of already-emitted ids (prompt included, matching
/// upstream -- `tools/main/main.cpp:725` feeds prompt tokens through the
/// same `common_sampler_accept` generated tokens use) the caller has
/// already sliced to its own `repeat_last_n` window; this function does
/// no windowing of its own. Returns `None` only if `logits` is empty.
#[must_use]
pub fn sample_next_token(
    logits: &[f32],
    recent_tokens: &[u32],
    config: SamplingConfig,
    rng: &mut Rng,
) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| (index as u32, logit))
        .collect();

    apply_repetition_penalty(&mut candidates, recent_tokens, config);
    apply_top_k(&mut candidates, config.top_k);
    apply_top_p(&mut candidates, config.top_p);
    apply_min_p(&mut candidates, config.min_p);

    if config.temperature <= 0.0 {
        return collapse_to_argmax(&candidates);
    }
    for candidate in &mut candidates {
        candidate.1 /= config.temperature;
    }

    let probabilities = softmax_probabilities(&candidates);
    sample_from_distribution(&candidates, &probabilities, rng)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use fastrand::Rng;

    use super::{
        SamplingConfig, apply_min_p, apply_repetition_penalty, apply_top_k, apply_top_p,
        greedy_pick, sample_next_token,
    };

    #[test]
    fn empty_logits_pick_nothing() {
        assert_eq!(greedy_pick(&[]), None);
    }

    #[test]
    fn picks_the_single_peak() {
        let logits = vec![0.1, 0.9, -3.0, 0.2];
        assert_eq!(greedy_pick(&logits), Some(1));
    }

    #[test]
    fn ties_resolve_to_the_lowest_index() {
        let logits = vec![0.5, 0.9, 0.9, 0.1];
        assert_eq!(greedy_pick(&logits), Some(1));
    }

    #[test]
    fn constant_logits_do_not_leak_a_wrong_peak() {
        // degenerate control: a broken argmax that always returns a
        // fixed non-zero index would pass a naive "returns something"
        // assertion. asserting the exact deterministic tie-break (lowest
        // index) catches that class of bug.
        let logits = vec![3.0; 16];
        assert_eq!(greedy_pick(&logits), Some(0));
    }

    /// Hand-computed: logits `[1.0, 5.0, 3.0, 2.0]`, `k=2` keeps exactly the
    /// two highest (`5.0` at id 1, `3.0` at id 2), sorted descending -- the
    /// one right answer for this vector, not merely "returns 2 items".
    #[test]
    fn top_k_keeps_exactly_the_k_highest_logits_sorted_descending() {
        let mut candidates = vec![(0u32, 1.0f32), (1, 5.0), (2, 3.0), (3, 2.0)];
        apply_top_k(&mut candidates, 2);
        assert_eq!(candidates, vec![(1, 5.0), (2, 3.0)]);
    }

    /// `k <= 0` is upstream's own "use vocab size" convention
    /// (`src/llama-sampling.cpp:234-236`) -- a no-op, not an error.
    #[test]
    fn top_k_zero_or_negative_disables_the_filter() {
        let original = vec![(0u32, 1.0f32), (1, 5.0), (2, 3.0)];
        let mut zero = original.clone();
        apply_top_k(&mut zero, 0);
        assert_eq!(zero, original);
        let mut negative = original.clone();
        apply_top_k(&mut negative, -1);
        assert_eq!(negative, original);
    }

    /// Hand-computed: logits chosen as `ln(p)` for `p = [0.5, 0.3, 0.2]` so
    /// the softmax reproduces exactly those probabilities (softmax of
    /// `ln(p)` over an already-normalized `p` is `p` itself). `top_p = 0.7`
    /// keeps the smallest prefix whose cumulative probability reaches
    /// `0.7`: `0.5` alone is short, `0.5 + 0.3 = 0.8 >= 0.7`, so exactly the
    /// first two survive (`src/llama-sampling.cpp:708-733`).
    #[test]
    fn top_p_keeps_the_smallest_prefix_reaching_the_cumulative_threshold() {
        let mut candidates = vec![
            (0u32, libm::logf(0.5)),
            (1, libm::logf(0.3)),
            (2, libm::logf(0.2)),
        ];
        apply_top_p(&mut candidates, 0.7);
        assert_eq!(
            candidates.len(),
            2,
            "0.5 + 0.3 = 0.8 is the first prefix to reach 0.7"
        );
        assert_eq!(candidates[0].0, 0);
        assert_eq!(candidates[1].0, 1);
    }

    /// `top_p >= 1.0` is upstream's own disabling sentinel
    /// (`src/llama-sampling.cpp:711-713`).
    #[test]
    fn top_p_of_one_disables_the_filter() {
        let original = vec![(0u32, 2.0f32), (1, 1.0), (2, 0.5)];
        let mut candidates = original.clone();
        apply_top_p(&mut candidates, 1.0);
        assert_eq!(candidates, original);
    }

    /// Hand-computed: `max_logit = 2.0`, `min_p = 0.5`, threshold `= 2.0 +
    /// ln(0.5) ~= 1.3069`. Only the `2.0` candidate clears it; `1.0` and
    /// `-1.0` are both below it (`src/llama-sampling.cpp:775-806`).
    #[test]
    fn min_p_drops_candidates_below_the_relative_probability_threshold() {
        let mut candidates = vec![(0u32, 2.0f32), (1, 1.0), (2, -1.0)];
        apply_min_p(&mut candidates, 0.5);
        assert_eq!(candidates, vec![(0, 2.0)]);
    }

    /// Hand-computed against `src/llama-sampling.cpp:1650-1681`'s own
    /// formula: `recent_tokens = [0, 0, 1]` gives id 0 a count of 2, id 1 a
    /// count of 1. `repeat_penalty = 2.0`: id 0's logit `1.0 > 0` divides
    /// (`1.0 / 2.0 = 0.5`); id 1's logit `-1.0 <= 0` multiplies (`-1.0 * 2.0
    /// = -2.0`). Then both subtract `count * frequency_penalty +
    /// (count > 0 ? presence_penalty : 0)`: id 0 subtracts `2 * 0.1 + 0.05 =
    /// 0.25` -> `0.25`; id 1 subtracts `1 * 0.1 + 0.05 = 0.15` -> `-2.15`.
    #[test]
    fn repetition_penalty_matches_llama_cpp_formula_by_hand() {
        let config = SamplingConfig {
            repeat_penalty: 2.0,
            frequency_penalty: 0.1,
            presence_penalty: 0.05,
            ..SamplingConfig::default()
        };
        let mut candidates = vec![(0u32, 1.0f32), (1, -1.0)];
        apply_repetition_penalty(&mut candidates, &[0, 0, 1], config);
        assert_eq!(candidates, vec![(0, 0.25), (1, -2.15)]);
    }

    /// A token absent from `recent_tokens` is untouched by the penalty --
    /// the filter only ever modifies candidates it has a count for.
    #[test]
    fn repetition_penalty_leaves_unseen_tokens_untouched() {
        let config = SamplingConfig {
            repeat_penalty: 5.0,
            frequency_penalty: 1.0,
            ..SamplingConfig::default()
        };
        let mut candidates = vec![(7u32, 3.0f32)];
        apply_repetition_penalty(&mut candidates, &[1, 2, 3], config);
        assert_eq!(candidates, vec![(7, 3.0)]);
    }

    /// The order this module's chain applies penalty and temperature in --
    /// penalty first, matching `common/common.h:162-172`'s default chain
    /// (`PENALTIES` before `TEMPERATURE`) -- is not interchangeable with the
    /// reverse. Penalty's frequency/presence terms are an ABSOLUTE
    /// subtraction in raw-logit units; temperature's divide rescales
    /// whatever came before it. Applying penalty then dividing by
    /// `temperature = 2.0` on logits `[4.0, 4.0]` (id 0 seen once,
    /// `frequency_penalty = 2.0`) gives `[(4.0 - 2.0) / 2.0, 4.0 / 2.0] =
    /// [1.0, 2.0]`; dividing first then applying the same penalty gives
    /// `[(4.0 / 2.0) - 2.0, 4.0 / 2.0] = [0.0, 2.0]` -- a different id-0
    /// value, proving the order is load-bearing, not cosmetic.
    #[test]
    fn repetition_penalty_before_temperature_differs_from_after() {
        let config = SamplingConfig {
            repeat_penalty: 1.0,
            frequency_penalty: 2.0,
            ..SamplingConfig::default()
        };
        let temperature = 2.0f32;
        let recent_tokens = [0u32];

        let mut penalty_then_temperature = vec![(0u32, 4.0f32), (1, 4.0)];
        apply_repetition_penalty(&mut penalty_then_temperature, &recent_tokens, config);
        for candidate in &mut penalty_then_temperature {
            candidate.1 /= temperature;
        }

        let mut temperature_then_penalty = vec![(0u32, 4.0f32), (1, 4.0)];
        for candidate in &mut temperature_then_penalty {
            candidate.1 /= temperature;
        }
        apply_repetition_penalty(&mut temperature_then_penalty, &recent_tokens, config);

        assert_eq!(
            penalty_then_temperature,
            vec![(0, 1.0), (1, 2.0)],
            "this module's own order"
        );
        assert_eq!(
            temperature_then_penalty,
            vec![(0, 0.0), (1, 2.0)],
            "the reversed order"
        );
        assert_ne!(
            penalty_then_temperature, temperature_then_penalty,
            "swapping penalty/temperature order changes id 0's logit -- llama.cpp applies \
             penalties before temperature (common/common.h:162-172), never after"
        );
    }

    /// Degenerate control: `temperature <= 0.0` must converge to plain
    /// argmax, matching [`greedy_pick`] exactly over the same logits, for
    /// every seed -- the rng is never consulted once the chain collapses.
    #[test]
    fn temperature_zero_converges_to_argmax_matching_greedy_pick() {
        let logits = vec![0.1, 0.9, -3.0, 0.2];
        let config = SamplingConfig::default();
        for seed in 0..8u64 {
            let mut rng = Rng::with_seed(seed);
            assert_eq!(
                sample_next_token(&logits, &[], config, &mut rng),
                greedy_pick(&logits)
            );
        }
    }

    /// [`SamplingConfig::default`]'s whole point: every filter disabled
    /// collapses [`sample_next_token`] to exactly [`greedy_pick`], proved
    /// directly against the same fixture [`crate::sample`]'s own
    /// `greedy_pick` tests use, not just a synthetic vector picked to make
    /// this test easy to pass.
    #[test]
    fn default_config_matches_greedy_pick_exactly() {
        let logits = vec![0.5, 0.9, 0.9, 0.1];
        let mut rng = Rng::with_seed(42);
        assert_eq!(
            sample_next_token(&logits, &[], SamplingConfig::default(), &mut rng),
            greedy_pick(&logits)
        );
    }

    /// Determinism: the same seed against the same logits (and the same
    /// empty recent-token window) must reproduce the same token on every
    /// one of 100 independent draws -- each draw reconstructs a fresh `Rng`
    /// from the same seed, so this is not merely "the same `Rng` object
    /// wasn't consumed twice".
    #[test]
    fn same_seed_and_logits_always_pick_the_same_token_across_100_draws() {
        let config = SamplingConfig {
            temperature: 1.0,
            top_p: 0.9,
            min_p: 0.05,
            ..SamplingConfig::default()
        };
        let logits = vec![0.1, 2.5, -1.0, 0.4, 1.8];
        let seed = 0x00C0_FFEE_u64;

        let first_draw = sample_next_token(&logits, &[], config, &mut Rng::with_seed(seed));
        for _ in 0..100 {
            let draw = sample_next_token(&logits, &[], config, &mut Rng::with_seed(seed));
            assert_eq!(
                draw, first_draw,
                "same seed + same logits must reproduce the same token"
            );
        }
    }

    /// The only test that catches a sampler that is deterministic when it
    /// should not be: logits `[1.0, 0.0]` at `temperature = 1.0` give an
    /// exact softmax split (`sigmoid(1.0) ~= 0.7310586` for id 0, the
    /// complement for id 1`), and drawing many times from ONE advancing
    /// `Rng` must land near that split, not always pick the same token.
    #[test]
    fn temperature_softmax_distribution_matches_known_probabilities() {
        let config = SamplingConfig {
            temperature: 1.0,
            ..SamplingConfig::default()
        };
        let logits = vec![1.0f32, 0.0f32];
        let mut rng = Rng::with_seed(0x5EED);
        let draws = 20_000u32;
        let mut token_zero_count = 0u32;
        for _ in 0..draws {
            match sample_next_token(&logits, &[], config, &mut rng) {
                Some(0) => token_zero_count += 1,
                Some(1) => {}
                other => panic!("logits are non-empty, expected Some(0 | 1), got {other:?}"),
            }
        }
        let empirical = f64::from(token_zero_count) / f64::from(draws);
        let expected = 0.7310586;
        assert!(
            (empirical - expected).abs() < 0.02,
            "empirical id-0 rate {empirical} too far from the known softmax probability {expected}"
        );
    }

    /// `logits` empty is the one documented `None` case, regardless of
    /// config or rng state.
    #[test]
    fn empty_logits_sample_nothing() {
        let mut rng = Rng::with_seed(7);
        let empty: Vec<f32> = Vec::new();
        assert_eq!(
            sample_next_token(&empty, &[], SamplingConfig::default(), &mut rng),
            None
        );
    }
}
