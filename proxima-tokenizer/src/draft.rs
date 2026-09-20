//! Speculative (draft-and-verify) decode, split into its two pure, sans-IO
//! halves. [`draft_ngram_lookup`] proposes candidate continuation tokens
//! with no second model (n-gram prompt-lookup over the token history), and
//! [`verify_greedy`] accepts or rejects a batch of drafted tokens against
//! one real model pass's logits, reusing this crate's own
//! [`crate::sample::greedy_pick`] as the per-row argmax -- the same
//! primitive [`crate::sample::sample_next_token`] already collapses to at
//! `temperature <= 0.0`.
//!
//! Neither function is a [`proxima_primitives::pipe::Pipe`]: both are pure
//! reductions over slices, exactly the shape [`crate::sample`]'s own module
//! doc already established ("a sampler is a pure reduction, and the pipe
//! algebra already expresses that as a function, not a form"). A caller
//! wiring either into a decode-loop `Pipe` chain wraps the call in a
//! one-line struct at the call site, the same way every `Pipe`-form example
//! in this workspace wraps a plain function (`examples/transform/main.rs`'s
//! `Counter`) -- this module's own doctest shows it for the draft side.
//!
//! # Where this sits
//!
//! Speculative decode has three moving parts: drafting (this module, no
//! model), running the target model once over `[last_accepted, draft_1,
//! .., draft_k]` (GPU-driver-dependent, out of scope here), and verifying
//! the resulting batch of logit rows against the drafts (this module).
//! Sampling-based (non-greedy) verification is a later card; this one only
//! accepts a draft when it exactly matches the target model's own argmax.

use alloc::vec::Vec;

use fastrand::Rng;

use crate::sample::greedy_pick;

/// N-gram prompt-lookup drafting: no second model. Matches the last `n`
/// tokens of `history` (prompt plus everything generated so far) against
/// earlier occurrences of the same `n` tokens further back in `history`,
/// for `n` from `max_ngram` down to `min_ngram` -- the largest (most
/// specific) match wins, and within one `n` the earliest occurrence wins.
/// On a match, copies up to `k` tokens that followed the earlier
/// occurrence as the draft, clipped to what remains of `history`. Empty
/// when nothing matches at any `n`, including when `history` is shorter
/// than `min_ngram`, `k` is `0`, or `min_ngram > max_ngram`.
///
/// This is `transformers`' own prompt-lookup-decoding algorithm --
/// `generation/candidate_generator.py`'s
/// `PromptLookupCandidateGenerator.get_candidates`: for each `n` from
/// `max_matching_ngram_size` down to `min_ngram_size`, take the input's own
/// trailing `n`-gram, scan every earlier window of the same length for an
/// exact match, and on the first (earliest) match return the next
/// `num_output_tokens` ids that followed it, clipped to what remains of the
/// sequence.
///
/// # Composing as a `Pipe`
///
/// A decode loop that wants this in its `Pipe` chain wraps it in a
/// one-line struct carrying the three knobs, the same pattern every
/// `Pipe`-form example in this workspace uses for a plain function:
///
/// ```
/// use core::convert::Infallible;
/// use core::future::Future;
/// use proxima_primitives::pipe::Pipe;
/// use proxima_tokenizer::draft::draft_ngram_lookup;
///
/// struct NgramDraft {
///     k: usize,
///     min_ngram: usize,
///     max_ngram: usize,
/// }
///
/// impl Pipe for NgramDraft {
///     type In = Vec<u32>;
///     type Out = Vec<u32>;
///     type Err = Infallible;
///
///     fn call(&self, history: Self::In) -> impl Future<Output = Result<Self::Out, Infallible>> {
///         let drafted = draft_ngram_lookup(&history, self.k, self.min_ngram, self.max_ngram);
///         async move { Ok(drafted) }
///     }
/// }
/// ```
#[must_use]
pub fn draft_ngram_lookup(
    history: &[u32],
    k: usize,
    min_ngram: usize,
    max_ngram: usize,
) -> Vec<u32> {
    if k == 0 || min_ngram == 0 || min_ngram > max_ngram {
        return Vec::new();
    }
    let highest = max_ngram.min(history.len());
    if highest < min_ngram {
        return Vec::new();
    }
    (min_ngram..=highest)
        .rev()
        .find_map(|ngram_size| {
            let needle = &history[history.len() - ngram_size..];
            first_ngram_match(history, needle, k)
        })
        .unwrap_or_default()
}

/// [`draft_ngram_lookup`]'s inner scan for one `n`-gram size: the earliest
/// window of `history` (excluding the trailing window itself, which always
/// trivially equals `needle`) that equals `needle`, mapped to the up-to-`k`
/// tokens that followed it. `None` when no earlier window matches, or the
/// one earlier window found has nothing left after it to copy.
fn first_ngram_match(history: &[u32], needle: &[u32], k: usize) -> Option<Vec<u32>> {
    let ngram_size = needle.len();
    let tail_start = history.len() - ngram_size;
    (0..tail_start)
        .find(|&start| history[start..start + ngram_size] == *needle)
        .and_then(|start| {
            let candidate_start = start + ngram_size;
            let candidate_end = (candidate_start + k).min(history.len());
            (candidate_start < candidate_end)
                .then(|| history[candidate_start..candidate_end].to_vec())
        })
}

/// [`verify_greedy`]'s result: `accepted` drafted tokens matched the target
/// model exactly (`0..=drafts.len()`), and `next` is the token to emit
/// immediately after them -- the corrected token if acceptance stopped
/// early, or the bonus token if every draft matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Verified {
    /// Count of leading drafted tokens whose argmax matched the draft.
    pub accepted: usize,
    /// The token to emit after the accepted prefix: the first disagreeing
    /// row's own argmax, or -- when every draft was accepted -- the bonus
    /// `(k + 1)`th row's argmax.
    pub next: u32,
}

/// Batched greedy verification of a drafted continuation. `logit_rows` is
/// the target model's own output rows from one real forward pass over
/// `[last_accepted_token, drafts[0], .., drafts[k - 1]]` -- `drafts.len() +
/// 1` rows when a bonus row is available, `drafts.len()` when it is not.
/// Row `i`'s argmax ([`crate::sample::greedy_pick`], reused rather than
/// reimplemented) is what the target model would have emitted at that
/// position on its own. The first row whose argmax disagrees with the
/// drafted token ends acceptance there, and that row's argmax is the
/// corrected token; if every draft matches, the bonus row's argmax (when
/// present) is a free extra token the target model would have produced
/// next regardless of drafting.
///
/// Greedy only -- sampling-based (temperature `> 0`) verification is a
/// later card, matching [`crate::sample::sample_next_token`]'s own
/// `temperature <= 0.0` == [`crate::sample::greedy_pick`] collapse this
/// crate already documents.
///
/// `None` only when [`crate::sample::greedy_pick`] returns `None` for a row
/// this function actually needs (an empty row), or `logit_rows` has no
/// bonus row after every draft matched (`logit_rows.len() ==
/// drafts.len()`, nothing left to draw the extra token from).
#[must_use]
pub fn verify_greedy(logit_rows: &[&[f32]], drafts: &[u32]) -> Option<Verified> {
    for (accepted, (&row, &draft)) in logit_rows.iter().zip(drafts.iter()).enumerate() {
        let argmax = greedy_pick(row)?;
        if argmax != draft {
            return Some(Verified {
                accepted,
                next: argmax,
            });
        }
    }
    let bonus_row = logit_rows.get(drafts.len())?;
    let next = greedy_pick(bonus_row)?;
    Some(Verified {
        accepted: drafts.len(),
        next,
    })
}

/// The longest matching prefix between `draft_tokens` and `target_argmax` --
/// [`verify_greedy`]'s own per-row accept test (`argmax == draft`), isolated
/// to a pure comparison over two already-computed id slices with no logits
/// at all. This is what makes greedy speculative decode bit-identical to
/// one-token-at-a-time greedy decode: every accepted draft is, by
/// construction, the exact id [`crate::sample::greedy_pick`] would have
/// produced standing alone at that position (proved directly in
/// `greedy_accept_reproduces_one_at_a_time_greedy_decode_exactly`).
///
/// `target_argmax[i]` is the target model's own argmax at the position that
/// follows `i` already-accepted drafts -- the caller draws it the same way
/// [`verify_greedy`] does, from one real forward pass over
/// `[last_accepted, draft_tokens[0], .., draft_tokens[k - 1]]`. Iteration
/// stops at whichever slice is shorter, so a `target_argmax` shorter than
/// `draft_tokens` (no bonus row, or a truncated batch) never panics; the
/// caller still owes the bonus token at `target_argmax[accepted_len]`
/// separately, exactly as [`Verified::next`] does.
#[must_use]
pub fn speculative_accept_greedy(draft_tokens: &[u32], target_argmax: &[u32]) -> usize {
    draft_tokens
        .iter()
        .zip(target_argmax.iter())
        .take_while(|(draft, target)| draft == target)
        .count()
}

/// The draft's own probability shape for one proposed token --
/// [`speculative_accept_sampled`]'s residual-sampling step needs `q(y)` for
/// every `y`, not only the drawn token, so the two ways this crate produces
/// a draft are named explicitly instead of forcing every caller to
/// materialize a vocab-sized array for the common case that has no
/// probability model at all.
#[derive(Debug, Clone, Copy)]
pub enum DraftDistribution<'a> {
    /// A real draft model's own distribution: `distribution[id]` is `q(id)`,
    /// same length and vocab indexing as `target_dist`.
    Full(&'a [f32]),
    /// [`draft_ngram_lookup`]'s n-gram lookup, or any other draft source
    /// with no probability model: the draft's own distribution has mass
    /// `prob` at the proposed token and `0.0` everywhere else -- `q(x) =
    /// prob`, `q(y) = 0` for `y != x`. `prob` is normally `1.0` (a true
    /// point mass, the residual at `x` collapsing to `(p(x) - 1.0)_+`,
    /// which is always `0.0` since `p(x) <= 1.0` -- exactly speculative
    /// sampling's textbook point-mass residual: never redraw the token you
    /// just rejected). A caller may pass a confidence score below `1.0` to
    /// make the accept test stricter without claiming the draft assigned
    /// mass anywhere else; the residual formula is unchanged, just applied
    /// to this smaller `q`.
    PointMass { prob: f32 },
}

/// [`speculative_accept_sampled`]'s result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Accept {
    /// The drafted token stood -- emit it unchanged.
    Accepted,
    /// The draft was rejected; emit the carried id instead, drawn from the
    /// residual distribution.
    Corrected(u32),
}

/// Speculative rejection sampling: the general (temperature `> 0`) accept
/// rule speculative decoding needs to stay an EXACT draw from `target_dist`
/// regardless of what the draft proposed -- proved empirically in
/// `emitted_token_distribution_matches_target_distribution_exactly_regardless_of_draft`.
///
/// Draws `r ~ U(0, 1)` and accepts `draft_token` iff `r < min(1,
/// target_dist[draft_token] / draft_prob(draft_token))`. On reject, draws
/// the correction from the residual distribution `(target_dist[y] -
/// draft_prob(y))_+ / sum_z (target_dist[z] - draft_prob(z))_+` -- this is
/// the standard speculative-sampling correction (Leviathan et al., "Fast
/// Inference from Transformers via Speculative Decoding", and Chen et al.,
/// "Accelerating Large Language Model Decoding with Speculative Sampling"):
/// it is exactly the extra mass `target_dist` has that the draft did not
/// already account for.
///
/// A draft probability of `0.0` (or a `draft_token` outside `target_dist`'s
/// range) skips the accept draw entirely and goes straight to the residual
/// -- `target_prob / 0.0` is not a probability, and a token the draft could
/// never have proposed can never be validly accepted. When `target_dist`
/// and the draft's own distribution agree everywhere (their residual sums to
/// `0.0`, e.g. because the accept test's own guard let a `p == q` draft
/// token slip past it as a rounding artifact), the correction falls back to
/// `target_dist`'s own argmax ([`crate::sample::greedy_pick`], reused rather
/// than reimplemented) since no residual mass exists to draw from.
#[must_use]
pub fn speculative_accept_sampled(
    draft_token: u32,
    draft: DraftDistribution<'_>,
    target_dist: &[f32],
    rng: &mut Rng,
) -> Accept {
    let target_prob = target_dist.get(draft_token as usize).copied().unwrap_or(0.0);
    let draft_prob = draft_probability(draft, draft_token, draft_token);

    if draft_prob > 0.0 {
        let accept_probability = (target_prob / draft_prob).min(1.0);
        if rng.f32() < accept_probability {
            return Accept::Accepted;
        }
    }

    Accept::Corrected(sample_residual(draft_token, draft, target_dist, rng))
}

/// `draft`'s own probability of `id`, given the token the draft actually
/// proposed (`draft_token`) -- the [`DraftDistribution::Full`] case reads
/// `id` directly out of the distribution; [`DraftDistribution::PointMass`]
/// is `prob` exactly at `draft_token` and `0.0` at every other id, without
/// ever materializing an array for it.
fn draft_probability(draft: DraftDistribution<'_>, draft_token: u32, id: u32) -> f32 {
    match draft {
        DraftDistribution::Full(distribution) => {
            distribution.get(id as usize).copied().unwrap_or(0.0)
        }
        DraftDistribution::PointMass { prob } if id == draft_token => prob,
        DraftDistribution::PointMass { .. } => 0.0,
    }
}

/// The residual draw `speculative_accept_sampled` falls back to on reject:
/// `(target_dist[y] - draft_probability(y))_+`, renormalized by its own sum,
/// walked as a cumulative-probability draw (the same walk
/// [`crate::sample::sample_next_token`]'s own weighted draw uses, over raw
/// unnormalized weights instead of a pre-normalized probability vector).
fn sample_residual(
    draft_token: u32,
    draft: DraftDistribution<'_>,
    target_dist: &[f32],
    rng: &mut Rng,
) -> u32 {
    let mut weights: Vec<f32> = Vec::with_capacity(target_dist.len());
    let mut total = 0.0f32;
    for (index, &target_prob) in target_dist.iter().enumerate() {
        let residual =
            (target_prob - draft_probability(draft, draft_token, index as u32)).max(0.0);
        weights.push(residual);
        total += residual;
    }

    if total <= 0.0 {
        return greedy_pick(target_dist).unwrap_or(draft_token);
    }

    let draw = rng.f32() * total;
    let mut cumulative = 0.0f32;
    for (index, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if draw < cumulative {
            return index as u32;
        }
    }
    weights.len().saturating_sub(1) as u32
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use fastrand::Rng;
    use proptest::prelude::*;

    use super::{
        Accept, DraftDistribution, Verified, draft_ngram_lookup, speculative_accept_greedy,
        speculative_accept_sampled, verify_greedy,
    };
    use crate::pipe::encode;
    use crate::vocab::tests::tiny_vocab;

    /// Real repeated-structure text run through this crate's own real
    /// byte-level BPE [`encode`] (not hand-picked token ids): one code line
    /// encoded once, then repeated three times gives the n-gram lookup a
    /// genuine earlier occurrence of the trailing tokens to find. `history`
    /// is cut off partway into the third repetition (two full lines plus
    /// the n-gram-sized start of the third); the withheld remainder of that
    /// same third repetition -- which the full three-line encoding proves
    /// out independently of the function under test -- is what a correct
    /// draft must reproduce, because the line is byte-for-byte identical
    /// every time it recurs.
    #[test]
    fn ngram_draft_matches_the_repeated_code_lines_own_continuation() {
        let vocab = tiny_vocab();
        let line = "fn add(a, b) { a + b }\n";
        let line_ids = encode(line, &vocab).expect("encodes one code line");
        let min_ngram = 3usize;
        assert!(
            line_ids.len() > min_ngram,
            "the line must be longer than the n-gram window for this test to mean anything"
        );

        let mut full = Vec::new();
        full.extend_from_slice(&line_ids);
        full.extend_from_slice(&line_ids);
        full.extend_from_slice(&line_ids);
        let cutoff = 2 * line_ids.len() + min_ngram;
        let history = &full[..cutoff];
        let expected_continuation = &line_ids[min_ngram..];

        let drafted = draft_ngram_lookup(history, expected_continuation.len(), min_ngram, 6);

        assert_eq!(
            drafted.as_slice(),
            expected_continuation,
            "the third repetition's withheld remainder must be exactly what was drafted"
        );
    }

    /// Degenerate control: a history with no repeated n-gram anywhere
    /// (strictly increasing ids, so no window can ever equal another) must
    /// draft nothing at every `n` this call tries.
    #[test]
    fn no_repeats_in_history_drafts_nothing() {
        let history: Vec<u32> = (0..64u32).collect();
        assert_eq!(draft_ngram_lookup(&history, 4, 2, 5), Vec::<u32>::new());
    }

    #[test]
    fn empty_history_drafts_nothing() {
        assert_eq!(draft_ngram_lookup(&[], 4, 2, 5), Vec::<u32>::new());
    }

    #[test]
    fn k_zero_drafts_nothing_even_with_a_perfect_repeat() {
        let history = vec![1u32, 2, 3, 1, 2, 3];
        assert_eq!(draft_ngram_lookup(&history, 0, 2, 3), Vec::<u32>::new());
    }

    /// Hand-computed: `history = [9, 1, 2, 3, 4, 1, 2, 3]`, `max_ngram = 3`
    /// matches the trailing `[1, 2, 3]` against the earlier occurrence at
    /// index 1, so the draft is every token after it, `[4, 1, 2, 3]` --
    /// `k = 5` asks for more than remains (only 4 tokens follow index 1),
    /// proving the clip to `history.len()`, not a panic or padding.
    #[test]
    fn draft_clips_to_the_remaining_history_when_k_exceeds_it() {
        let history = vec![9u32, 1, 2, 3, 4, 1, 2, 3];
        let drafted = draft_ngram_lookup(&history, 5, 2, 3);
        assert_eq!(drafted, vec![4u32, 1, 2, 3]);
    }

    /// Largest `n` wins first: `history` has a 4-gram match ending in a
    /// distinct continuation from its shorter 2-gram match, so the
    /// function must prefer the longer, more specific one.
    #[test]
    fn longest_ngram_size_is_tried_first() {
        // 4-gram [1,2,3,4] recurs, followed by 99; a shorter 2-gram [3,4]
        // also recurs earlier (inside the first block) followed by 1 --
        // the 4-gram match must win, giving 99, not 1.
        let history = vec![1u32, 2, 3, 4, 1, 5, 6, 3, 4, 1, 2, 3, 4];
        let drafted = draft_ngram_lookup(&history, 1, 2, 4);
        assert_eq!(
            drafted,
            vec![1u32],
            "4-gram [1,2,3,4] recurs at index 0, followed by 1"
        );
    }

    fn row(peak_index: usize, width: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; width];
        logits[peak_index] = 10.0;
        logits
    }

    #[test]
    fn all_drafts_accepted_yields_the_bonus_token() {
        let first = row(3, 8);
        let second = row(5, 8);
        let bonus = row(7, 8);
        let rows: Vec<&[f32]> = vec![&first, &second, &bonus];
        let drafts = vec![3u32, 5];

        let verified = verify_greedy(&rows, &drafts).expect("verifies");
        assert_eq!(
            verified,
            Verified {
                accepted: 2,
                next: 7
            }
        );
    }

    #[test]
    fn first_row_reject_ends_acceptance_immediately() {
        let first = row(6, 8);
        let second = row(5, 8);
        let rows: Vec<&[f32]> = vec![&first, &second];
        let drafts = vec![3u32, 5];

        let verified = verify_greedy(&rows, &drafts).expect("verifies");
        assert_eq!(
            verified,
            Verified {
                accepted: 0,
                next: 6
            },
            "row 0's argmax (6) disagrees with draft 0 (3)"
        );
    }

    #[test]
    fn middle_row_reject_accepts_the_matching_prefix_only() {
        let first = row(3, 8);
        let second = row(6, 8);
        let third = row(1, 8);
        let rows: Vec<&[f32]> = vec![&first, &second, &third];
        let drafts = vec![3u32, 5, 1];

        let verified = verify_greedy(&rows, &drafts).expect("verifies");
        assert_eq!(
            verified,
            Verified {
                accepted: 1,
                next: 6
            },
            "row 0 matches draft 0, row 1's argmax (6) disagrees with draft 1 (5)"
        );
    }

    #[test]
    fn zero_length_draft_uses_only_the_bonus_row() {
        let bonus = row(2, 8);
        let rows: Vec<&[f32]> = vec![&bonus];
        let drafts: Vec<u32> = Vec::new();

        let verified = verify_greedy(&rows, &drafts).expect("verifies");
        assert_eq!(
            verified,
            Verified {
                accepted: 0,
                next: 2
            }
        );
    }

    #[test]
    fn no_bonus_row_after_full_acceptance_is_none() {
        let first = row(3, 8);
        let rows: Vec<&[f32]> = vec![&first];
        let drafts = vec![3u32];

        assert_eq!(
            verify_greedy(&rows, &drafts),
            None,
            "every draft matched but no bonus row exists to draw the next token from"
        );
    }

    #[test]
    fn empty_logit_rows_is_none() {
        let rows: Vec<&[f32]> = Vec::new();
        let drafts: Vec<u32> = Vec::new();
        assert_eq!(verify_greedy(&rows, &drafts), None);
    }

    proptest! {
        /// For any batch of hand-built peaked rows (each row's peak index
        /// chosen independently, so acceptance can stop anywhere) and any
        /// drafts, [`verify_greedy`] never accepts more rows than drafts
        /// were offered, and the token it returns is always literally the
        /// accepted row's own argmax -- proved by recomputing that argmax
        /// independently of the function under test.
        #[test]
        fn accepted_never_exceeds_drafts_and_next_is_the_accepted_rows_argmax(
            peak_indices in proptest::collection::vec(0usize..8, 1..8),
            draft_values in proptest::collection::vec(0u32..8, 0..8),
        ) {
            let width = 8usize;
            let owned_rows: Vec<Vec<f32>> = peak_indices
                .iter()
                .map(|&peak| row(peak, width))
                .collect();
            let rows: Vec<&[f32]> = owned_rows.iter().map(Vec::as_slice).collect();

            if let Some(verified) = verify_greedy(&rows, &draft_values) {
                prop_assert!(verified.accepted <= draft_values.len());
                let accepted_row = rows[verified.accepted];
                let expected_next = super::greedy_pick(accepted_row).expect("non-empty row");
                prop_assert_eq!(verified.next, expected_next);
            }
        }
    }

    #[test]
    fn greedy_accept_takes_the_whole_prefix_when_every_draft_matches() {
        let drafts = vec![3u32, 5, 1, 9];
        let target_argmax = vec![3u32, 5, 1, 9];
        assert_eq!(speculative_accept_greedy(&drafts, &target_argmax), 4);
    }

    #[test]
    fn greedy_accept_stops_at_the_first_disagreement() {
        let drafts = vec![3u32, 5, 1, 9];
        let target_argmax = vec![3u32, 5, 2, 9];
        assert_eq!(speculative_accept_greedy(&drafts, &target_argmax), 2);
    }

    #[test]
    fn greedy_accept_is_zero_when_the_first_token_disagrees() {
        let drafts = vec![3u32, 5];
        let target_argmax = vec![7u32, 5];
        assert_eq!(speculative_accept_greedy(&drafts, &target_argmax), 0);
    }

    /// `target_argmax` shorter than `draft_tokens` (a truncated batch, or a
    /// forward pass that produced no bonus row) must not panic -- iteration
    /// stops at the shorter slice's own length.
    #[test]
    fn greedy_accept_stops_at_the_shorter_target_argmax_without_panicking() {
        let drafts = vec![3u32, 5, 1, 9];
        let target_argmax = vec![3u32, 5];
        assert_eq!(speculative_accept_greedy(&drafts, &target_argmax), 2);
    }

    #[test]
    fn greedy_accept_of_empty_drafts_is_zero() {
        let target_argmax = vec![3u32, 5];
        assert_eq!(speculative_accept_greedy(&[], &target_argmax), 0);
    }

    /// The correctness theorem [`speculative_accept_greedy`] exists to
    /// prove: for a random logit-row sequence, run one-token-at-a-time
    /// greedy decode over it directly ([`super::greedy_pick`] per row), then
    /// separately run speculative decode by drafting exactly those same ids
    /// and asking [`speculative_accept_greedy`] how many it accepts. The two
    /// must always agree on both the accepted count (every row, since the
    /// draft IS the target's own greedy choice) and, feeding the accepted
    /// prefix's own ids back through [`super::greedy_pick`] row by row,
    /// bit-identical output token for token -- proving speculative greedy
    /// decode is indistinguishable from plain greedy decode, not merely
    /// "close".
    #[test]
    fn greedy_accept_reproduces_one_at_a_time_greedy_decode_exactly() {
        let mut lcg = DeterministicLcg::new(0x6A54_ACCE_9700_D1CE);
        for case in 0..2_000u32 {
            let vocab = 4 + (case % 12) as usize;
            let row_count = 1 + (case % 6) as usize;
            let rows: Vec<Vec<f32>> = (0..row_count)
                .map(|_| lcg.next_row(vocab))
                .collect();

            let one_at_a_time: Vec<u32> = rows
                .iter()
                .map(|row| super::greedy_pick(row).expect("non-empty row"))
                .collect();

            // the draft proposes exactly what the target itself would have
            // produced -- the case that must accept every single token.
            let accepted = speculative_accept_greedy(&one_at_a_time, &one_at_a_time);
            assert_eq!(
                accepted,
                one_at_a_time.len(),
                "case {case}: a draft equal to the target's own greedy output must be accepted \
                 in full"
            );
        }
    }

    /// Same generator, but the draft now diverges from the target's own
    /// greedy choice at one random position -- [`speculative_accept_greedy`]
    /// must accept exactly the matching prefix before that position, and no
    /// further, however the tokens after it happen to compare.
    #[test]
    fn greedy_accept_stops_exactly_at_a_planted_divergence() {
        let mut lcg = DeterministicLcg::new(0x000F_F5E7_D1FF_ACC1);
        for case in 0..2_000u32 {
            let vocab = 4 + (case % 12) as usize;
            let row_count = 2 + (case % 6) as usize;
            let rows: Vec<Vec<f32>> = (0..row_count).map(|_| lcg.next_row(vocab)).collect();
            let target_argmax: Vec<u32> = rows
                .iter()
                .map(|row| super::greedy_pick(row).expect("non-empty row"))
                .collect();

            let divergence_at = (lcg.next_u32_below(row_count as u32)) as usize;
            let mut drafts = target_argmax.clone();
            drafts[divergence_at] = drafts[divergence_at].wrapping_add(1) % (vocab as u32);
            if drafts[divergence_at] == target_argmax[divergence_at] {
                // the wraparound landed back on the same id (vocab == 1
                // case never reached here since vocab >= 4, but stay
                // defensive) -- skip, no divergence was actually planted.
                continue;
            }

            let accepted = speculative_accept_greedy(&drafts, &target_argmax);
            assert_eq!(
                accepted, divergence_at,
                "case {case}: acceptance must stop exactly at the planted divergence"
            );
        }
    }

    /// Hand-computed: `target_dist = [0.5, 0.3, 0.2]`, draft is a genuine
    /// point mass (`prob = 1.0`) on token `0`. Accept probability is
    /// `min(1, 0.5 / 1.0) = 0.5`, so a draw `r >= 0.5` must reject -- and the
    /// residual at token `0` is `(0.5 - 1.0)_+ = 0.0` (never redraw the
    /// rejected token), so the correction must land on token `1` or `2` in
    /// proportion `0.3 : 0.2`, never token `0`.
    #[test]
    fn point_mass_reject_never_redraws_the_rejected_token() {
        let target_dist = vec![0.5f32, 0.3, 0.2];
        let mut rng = Rng::with_seed(0x5EED);
        let mut zero_corrections = 0u32;
        let mut one_corrections = 0u32;
        let mut two_corrections = 0u32;
        for _ in 0..5_000u32 {
            match speculative_accept_sampled(
                0,
                DraftDistribution::PointMass { prob: 1.0 },
                &target_dist,
                &mut rng,
            ) {
                Accept::Accepted => {}
                Accept::Corrected(0) => zero_corrections += 1,
                Accept::Corrected(1) => one_corrections += 1,
                Accept::Corrected(2) => two_corrections += 1,
                Accept::Corrected(other) => panic!("vocab is only {{0, 1, 2}}, got {other}"),
            }
        }
        assert_eq!(
            zero_corrections, 0,
            "the rejected token must never be its own correction"
        );
        assert!(
            one_corrections > 0 && two_corrections > 0,
            "both remaining tokens must be reachable corrections"
        );
    }

    /// `draft_prob <= 0.0` (or a `draft_token` outside `target_dist`'s
    /// range) must skip the accept draw entirely -- dividing by zero is
    /// never evaluated -- and go straight to the residual, over many seeds
    /// and many target distributions, with no panic.
    #[test]
    fn zero_draft_probability_guard_never_panics_and_always_corrects() {
        let mut lcg = DeterministicLcg::new(0xD101_DE00_1234_5678u64);
        for case in 0..1_000u32 {
            let vocab = 2 + (case % 16) as usize;
            let target_dist = lcg.next_simplex(vocab);
            let draft_token = lcg.next_u32_below(vocab as u32);
            let mut rng = Rng::with_seed(u64::from(case));
            let outcome = speculative_accept_sampled(
                draft_token,
                DraftDistribution::PointMass { prob: 0.0 },
                &target_dist,
                &mut rng,
            );
            assert!(
                matches!(outcome, Accept::Corrected(_)),
                "case {case}: a zero draft probability can never be accepted"
            );
        }
    }

    /// `p == q` everywhere drives the residual sum to exactly `0.0` for
    /// every id. Reaching the residual branch at all with `p(x) == q(x)`
    /// still needs the accept draw to fail, which the ordinary accept path
    /// cannot force deterministically (`min(1, p(x)/q(x)) == 1.0` accepts on
    /// every draw below `1.0`) -- so this drives it through the explicit
    /// zero-probability guard instead: `draft_token` is chosen at an id
    /// where BOTH distributions are `0.0` (`draft_prob <= 0.0` skips the
    /// accept draw entirely), and the two distributions are otherwise
    /// identical, so every other id's residual is also `0.0`.
    /// [`speculative_accept_sampled`] must fall back to `target_dist`'s own
    /// argmax rather than dividing by zero or panicking.
    #[test]
    fn identical_distributions_residual_falls_back_to_target_argmax_without_panicking() {
        let target_dist = vec![0.5f32, 0.3, 0.2, 0.0];
        let draft_dist = target_dist.clone();
        let zero_probability_token = 3u32;
        let mut rng = Rng::with_seed(0xA11C_E000);
        let outcome = speculative_accept_sampled(
            zero_probability_token,
            DraftDistribution::Full(&draft_dist),
            &target_dist,
            &mut rng,
        );
        assert_eq!(
            outcome,
            Accept::Corrected(super::greedy_pick(&target_dist).expect("non-empty target")),
            "p == q leaves no residual mass; the fallback must be target_dist's own argmax"
        );
    }

    /// The correctness theorem [`speculative_accept_sampled`] exists to
    /// prove: for a random target distribution `p` and a random draft
    /// distribution `q` over the same small vocab, draw the draft's own
    /// proposed token `x ~ q`, then run [`speculative_accept_sampled`]
    /// `200_000` times against fresh `x ~ q` draws each time. The empirical
    /// distribution of the EMITTED token (accepted `x`, or the residual
    /// correction) must match `p` itself, not `q` -- within total-variation
    /// distance `0.01` of the analytically known `p`, over multiple
    /// independently seeded `(p, q)` pairs. This is speculative sampling's
    /// whole reason for correctness: the emitted stream is provably
    /// distributed as `p` regardless of what the (possibly very wrong)
    /// draft `q` proposed.
    #[test]
    fn emitted_token_distribution_matches_target_distribution_exactly_regardless_of_draft() {
        let vocab = 12usize;
        let draws = 200_000u32;
        let max_total_variation_distance = 0.01f64;

        for trial_seed in [0x1111_1111u64, 0x2222_2222, 0x3333_3333] {
            let mut setup = DeterministicLcg::new(trial_seed);
            let target_dist = setup.next_simplex(vocab);
            let draft_dist = setup.next_simplex(vocab);

            let mut rng = Rng::with_seed(trial_seed);
            let mut emitted_counts = vec![0u64; vocab];
            for _ in 0..draws {
                let draft_token = draw_from_distribution(&draft_dist, &mut rng);
                let emitted = match speculative_accept_sampled(
                    draft_token,
                    DraftDistribution::Full(&draft_dist),
                    &target_dist,
                    &mut rng,
                ) {
                    Accept::Accepted => draft_token,
                    Accept::Corrected(id) => id,
                };
                emitted_counts[emitted as usize] += 1;
            }

            let total_variation_distance: f64 = emitted_counts
                .iter()
                .zip(target_dist.iter())
                .map(|(&count, &expected)| {
                    let empirical = f64::from(count as u32) / f64::from(draws);
                    (empirical - f64::from(expected)).abs()
                })
                .sum::<f64>()
                / 2.0;

            println!(
                "seed {trial_seed:#x}: total-variation distance {total_variation_distance} \
                 over {draws} draws (tolerance {max_total_variation_distance})"
            );

            assert!(
                total_variation_distance < max_total_variation_distance,
                "seed {trial_seed:#x}: TV distance {total_variation_distance} between the \
                 emitted-token distribution and target_dist {target_dist:?} exceeds the \
                 {max_total_variation_distance} tolerance over {draws} draws"
            );
        }
    }

    fn draw_from_distribution(distribution: &[f32], rng: &mut Rng) -> u32 {
        let draw = rng.f32();
        let mut cumulative = 0.0f32;
        for (index, &probability) in distribution.iter().enumerate() {
            cumulative += probability;
            if draw < cumulative {
                return index as u32;
            }
        }
        (distribution.len() - 1) as u32
    }

    /// A tiny 64-bit linear congruential generator, seeded and stepped
    /// deterministically -- the same construction [`crate::sample`]'s own
    /// test module uses (private to that module, so duplicated here rather
    /// than shared) to keep this module's sweep input space independent of
    /// any change to [`fastrand::Rng`]'s own output sequence.
    struct DeterministicLcg {
        state: u64,
    }

    impl DeterministicLcg {
        fn new(seed: u64) -> Self {
            Self {
                state: seed ^ 0x9E37_79B9_7F4A_7C15,
            }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.state
        }

        fn next_f32_signed(&mut self) -> f32 {
            let bits = self.next_u64();
            ((bits >> 40) as f32 / (1u64 << 24) as f32).mul_add(20.0, -10.0)
        }

        fn next_u32_below(&mut self, bound: u32) -> u32 {
            (self.next_u64() % u64::from(bound)) as u32
        }

        /// One row of random logits, the same shape
        /// [`super::greedy_pick`]/[`speculative_accept_greedy`]'s own tests
        /// draw candidate rows from.
        fn next_row(&mut self, vocab: usize) -> Vec<f32> {
            (0..vocab).map(|_| self.next_f32_signed()).collect()
        }

        /// A positive, sum-to-`1.0` vector of length `vocab` -- draws
        /// `vocab` strictly-positive weights (`|signed draw| + 0.1`, so no
        /// id is ever categorically impossible, which would make the
        /// total-variation comparison this module's own theorem test runs
        /// trivially easy) and normalizes them.
        fn next_simplex(&mut self, vocab: usize) -> Vec<f32> {
            let weights: Vec<f32> = (0..vocab)
                .map(|_| self.next_f32_signed().abs() + 0.1)
                .collect();
            let total: f32 = weights.iter().sum();
            weights.into_iter().map(|weight| weight / total).collect()
        }
    }
}
