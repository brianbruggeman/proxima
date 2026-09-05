//! Speculative (draft-and-verify) decode's pure drafting half.
//! [`draft_ngram_lookup`] proposes candidate continuation tokens with no
//! second model (n-gram prompt-lookup over the token history).
//!
//! Not a [`proxima_primitives::pipe::Pipe`]: it is a pure reduction over a
//! slice, exactly the shape [`crate::sample`]'s own module doc already
//! established ("a sampler is a pure reduction, and the pipe algebra
//! already expresses that as a function, not a form"). A caller wiring it
//! into a decode-loop `Pipe` chain wraps the call in a one-line struct at
//! the call site, the same way every `Pipe`-form example in this workspace
//! wraps a plain function (`examples/transform/main.rs`'s `Counter`) --
//! this module's own doctest shows it.
//!
//! # Where this sits
//!
//! Speculative decode has three moving parts: drafting (this module, no
//! model), running the target model once over `[last_accepted, draft_1,
//! .., draft_k]` (GPU-driver-dependent, out of scope here), and verifying
//! the resulting batch of logit rows against the drafts (a sibling
//! function landing separately in this same module).

use alloc::vec::Vec;

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::draft_ngram_lookup;
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
        // 4-gram [1,2,3,4] recurs, followed by 1; a shorter 2-gram [3,4]
        // also recurs earlier (inside the first block) -- the 4-gram match
        // must win.
        let history = vec![1u32, 2, 3, 4, 1, 5, 6, 3, 4, 1, 2, 3, 4];
        let drafted = draft_ngram_lookup(&history, 1, 2, 4);
        assert_eq!(
            drafted,
            vec![1u32],
            "4-gram [1,2,3,4] recurs at index 0, followed by 1"
        );
    }
}
