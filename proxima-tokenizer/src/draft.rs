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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::{Verified, draft_ngram_lookup, verify_greedy};
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
}
