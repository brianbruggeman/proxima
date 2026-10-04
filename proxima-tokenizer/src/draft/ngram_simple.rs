//! A faithful port of llama.cpp's `common_ngram_simple_draft`
//! (`common/ngram-map.cpp:49`, config at `common/ngram-map.h:24`): the one
//! self-speculative n-gram drafter every llama.cpp speculation type traces
//! back to, and the drafter in this module's
//! now-deleted `draft_ngram_lookup` duplicated with different tie-break and
//! size semantics (RISC debt this port retires -- proxima's own guiding
//! principle 1).
//!
//! Pure reduction over an id slice, exactly the shape this crate's `sample`
//! module already established for `sample.rs`'s samplers: no
//! [`proxima_primitives::pipe::Pipe`] here either, a caller wires it into a
//! decode-loop `Pipe` chain the same way `examples/transform/main.rs`'s
//! `Counter` wraps a plain function.
//!
//! # The algorithm, traced to the incumbent
//!
//! Given `history` (every token generated so far, NOT including the token
//! just sampled) and `sampled` (the token the caller's own sampler just
//! produced for the position after `history`), build one `size_n`-token
//! pattern: the trailing `size_n - 1` tokens of `history` followed by
//! `sampled`. Scan `history` from its most recent possible start position
//! backwards (never the trailing window itself, which trivially equals the
//! pattern) for the MOST RECENT earlier occurrence of that exact pattern --
//! llama.cpp's own loop counts down, so a later match always wins over an
//! earlier one, the opposite tie-break from `transformers`'
//! `PromptLookupCandidateGenerator` (which this crate's deleted
//! `draft_ngram_lookup` mirrored instead). On a match, the draft is up to
//! `size_m` tokens that followed that earlier occurrence, clipped to what
//! remains of `history` -- but only when at least `size_n` tokens remain to
//! copy; anything shorter drafts nothing at all, not a short draft.
//!
//! No draft is attempted at all unless `history.len() > size_n + size_m +
//! 1` -- llama.cpp's own guard, preserved exactly (a strict `>`, not
//! `>=`).

use alloc::vec::Vec;

/// llama.cpp's `common_ngram_simple_config` (`common/ngram-map.h:24`),
/// renamed to the field names `speculative-decode-llama-parity/SPEC.md`
/// (requirement R4) uses: `size_n` is llama's `size_ngram` (the n-gram
/// looked up in the token history), `size_m` is llama's `size_mgram` (how
/// many tokens after a match are drafted). llama.cpp's own default for
/// `ngram-simple` is `size_n = 12, size_m = 48`
/// (`common_params_speculative_ngram_map`, `common/common.h:361-365`, this
/// crate's own copy at [`DEFAULT_SIZE_N`]/[`DEFAULT_SIZE_M`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramSimpleConfig {
    /// Size of the n-gram looked up in `history` -- llama's `size_ngram`.
    pub size_n: u16,
    /// Size of the m-gram drafted after a match -- llama's `size_mgram`.
    pub size_m: u16,
}

/// llama.cpp's `ngram-simple` default `size_n`
/// (`common_params_speculative_ngram_map::size_n`, `common/common.h:361`).
pub const DEFAULT_SIZE_N: u16 = 12;

/// llama.cpp's `ngram-simple` default `size_m`
/// (`common_params_speculative_ngram_map::size_m`, `common/common.h:362`).
pub const DEFAULT_SIZE_M: u16 = 48;

impl Default for NgramSimpleConfig {
    fn default() -> Self {
        Self {
            size_n: DEFAULT_SIZE_N,
            size_m: DEFAULT_SIZE_M,
        }
    }
}

/// A faithful port of `common_ngram_simple_draft` (`common/ngram-map.cpp:49`)
/// -- see this module's own doc for the algorithm, traced line for line to
/// the incumbent. `history` is `tokens` in the C++ signature (every token
/// generated so far, not including `sampled`); `sampled` is the token the
/// caller's own sampler just drew for the position immediately after
/// `history`.
///
/// `out` is cleared then filled with the draft (empty when nothing
/// matched) -- the caller owns the allocation and is expected to keep one
/// buffer alive across decode steps and reuse it here, the same
/// clear-and-refill discipline `crate::sample`'s own scratch buffers use.
/// `clear` drops no capacity, so a buffer that has already grown to this
/// call's own draft length allocates nothing; this function itself performs
/// no allocation on any path (the match scan compares `history`'s own
/// slices directly rather than materializing a separate pattern buffer).
///
/// Stateless otherwise: every call recomputes the pattern and rescans
/// `history` from scratch, matching the incumbent, which carries no state
/// between calls either (unlike `ngram-map`/`ngram-mod`/`ngram-cache`,
/// whose ports build up an index across calls).
///
/// # Composing as a `Pipe`
///
/// A decode loop that wants this in its `Pipe` chain wraps it in a
/// one-line struct carrying `sampled` alongside the config, the same
/// pattern every `Pipe`-form example in this workspace uses for a plain
/// function (`examples/transform/main.rs`'s `Counter`). `call` takes
/// `&self`, so a caller wanting the reused-buffer discipline this
/// function's own doc describes keeps that buffer OUTSIDE the `Pipe`
/// (owned by the decode loop driving it, exactly how
/// `proxima-model-interop`'s own production loop does it) rather than
/// inside a field `call` cannot mutate through a shared reference:
///
/// ```
/// use core::convert::Infallible;
/// use core::future::Future;
/// use proxima_primitives::pipe::Pipe;
/// use proxima_tokenizer::draft::{NgramSimpleConfig, ngram_simple_draft};
///
/// struct NgramSimpleDraft {
///     config: NgramSimpleConfig,
///     sampled: u32,
/// }
///
/// impl Pipe for NgramSimpleDraft {
///     type In = Vec<u32>;
///     type Out = Vec<u32>;
///     type Err = Infallible;
///
///     fn call(&self, history: Self::In) -> impl Future<Output = Result<Self::Out, Infallible>> {
///         let mut drafted = Vec::new();
///         ngram_simple_draft(&self.config, &history, self.sampled, &mut drafted);
///         async move { Ok(drafted) }
///     }
/// }
/// ```
pub fn ngram_simple_draft(
    config: &NgramSimpleConfig,
    history: &[u32],
    sampled: u32,
    out: &mut Vec<u32>,
) {
    out.clear();

    let cur_len = history.len();
    let size_n = usize::from(config.size_n);
    let size_m = usize::from(config.size_m);

    // llama.cpp: `if (cur_len <= n_draft_min + n_draft_max + 1) return {};`
    if cur_len <= size_n + size_m + 1 {
        return;
    }

    // llama.cpp builds `pattern` as `tokens[cur_len - n_draft_min + 1 ..
    // cur_len]` (the trailing `size_n - 1` tokens) then pushes `sampled`.
    // Rather than materializing that pattern into its own buffer, the scan
    // below compares `history`'s own trailing window directly against each
    // candidate start position, and checks `sampled` against the position
    // immediately following it -- same comparison, zero allocation.
    // `saturating_sub` keeps `prefix_len` well-defined at `size_n == 0` too
    // (an empty prefix), matching the C++ loop bound `j < cur_len` never
    // firing when `cur_len - n_draft_min + 1 > cur_len`.
    let prefix_len = size_n.saturating_sub(1);
    let pattern_prefix = &history[cur_len - prefix_len..cur_len];

    // llama.cpp: `for (j = cur_len - n_draft_min - 1; j > 0; --j)` -- scans
    // from the most recent possible start position DOWN to `1`, so the
    // MOST RECENT match wins, not the earliest (the opposite tie-break from
    // `transformers`' prompt-lookup decoding).
    let scan_start = cur_len - size_n - 1;
    let match_pos = (1..=scan_start).rev().find(|&start| {
        history[start..start + prefix_len] == *pattern_prefix
            && history[start + prefix_len] == sampled
    });

    let Some(match_pos) = match_pos else {
        return;
    };

    // llama.cpp: `copy_max = min(n_draft_max, cur_len - (match_pos +
    // n_draft_min))`, then `if (copy_max < n_draft_min) return {};` --
    // note the guard is against `size_n`, not `0`: a short remainder drafts
    // nothing at all rather than a partial draft.
    let copy_max = size_m.min(cur_len - (match_pos + size_n));
    if copy_max < size_n {
        return;
    }

    out.extend_from_slice(&history[match_pos + size_n..match_pos + size_n + copy_max]);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use serde::Deserialize;

    use super::{NgramSimpleConfig, ngram_simple_draft};

    #[derive(Debug, Deserialize)]
    struct StreamsFile {
        streams: Vec<Stream>,
    }

    #[derive(Debug, Deserialize)]
    struct Stream {
        id: u32,
        tokens: Vec<u32>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureFile {
        params: FixtureParams,
        cases: Vec<FixtureCase>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureParams {
        size_ngram: u16,
        size_mgram: u16,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureCase {
        stream_id: u32,
        position: usize,
        sampled: u32,
        draft: Vec<u32>,
    }

    const STREAMS_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama-ngram/fixtures/streams.json"
    ));
    const NGRAM_SIMPLE_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama-ngram/fixtures/ngram_simple.json"
    ));

    /// [`ngram_simple_draft`] against every case in
    /// `tests/fixtures/llama-ngram/fixtures/ngram_simple.json`, whose
    /// `draft` field was produced by calling llama.cpp's own
    /// `common_ngram_simple_draft` directly (`README.md`,
    /// `speculative-decode-llama-parity/SPEC.md` AC5) -- never by
    /// re-deriving the expected output from reading the C++ (guiding
    /// principle 14). Each case replays one decode step: `history =
    /// stream[..position]`, `sampled = stream[position]`.
    #[test]
    fn ngram_simple_matches_llama_fixture() {
        let streams: StreamsFile = serde_json::from_str(STREAMS_JSON).expect("streams.json parses");
        let fixture: FixtureFile =
            serde_json::from_str(NGRAM_SIMPLE_JSON).expect("ngram_simple.json parses");

        let config = NgramSimpleConfig {
            size_n: fixture.params.size_ngram,
            size_m: fixture.params.size_mgram,
        };

        let mut non_empty = 0usize;
        let mut drafted: Vec<u32> = Vec::new();
        for (case_index, case) in fixture.cases.iter().enumerate() {
            let stream = streams
                .streams
                .iter()
                .find(|stream| stream.id == case.stream_id)
                .unwrap_or_else(|| panic!("stream {} exists in streams.json", case.stream_id));
            let history = &stream.tokens[..case.position];

            ngram_simple_draft(&config, history, case.sampled, &mut drafted);
            if !case.draft.is_empty() {
                non_empty += 1;
            }
            assert_eq!(
                drafted, case.draft,
                "case {case_index} (stream {}, position {}): got {drafted:?}, want {:?}",
                case.stream_id, case.position, case.draft
            );
        }

        println!("cases = {} non_empty = {non_empty}", fixture.cases.len());
        assert!(
            fixture.cases.len() >= 200,
            "fixture must carry at least 200 cases per SPEC.md AC5"
        );
    }

    /// Happy path, hand-computed: `history` repeats `[1, 2, 3, 4, 5, 6]`
    /// twice, `sampled` continues the pattern the same way the earlier
    /// occurrence did, and the earlier occurrence has `size_m` tokens left
    /// to copy -- the draft must be exactly that continuation.
    #[test]
    fn happy_path_drafts_the_earlier_occurrences_continuation() {
        let config = NgramSimpleConfig {
            size_n: 3,
            size_m: 3,
        };
        // `history`'s trailing `size_n - 1 = 2` tokens are `[1, 2]`;
        // `sampled = 3` completes the pattern `[1, 2, 3]`. That pattern
        // recurs earlier at index 1 (`history[1..4] == [1, 2, 3]`),
        // followed by exactly `size_m = 3` real tokens, `[4, 5, 9]` --
        // `copy_max = min(3, 10 - (1 + 3)) = 3 >= size_n`, so the full
        // continuation drafts.
        let history = vec![0u32, 1, 2, 3, 4, 5, 9, 9, 1, 2];
        let sampled = 3u32;
        let mut drafted = Vec::new();
        ngram_simple_draft(&config, &history, sampled, &mut drafted);
        assert_eq!(drafted, vec![4u32, 5, 9]);
    }

    /// Sad path: `history` shorter than `size_n + size_m + 1` must draft
    /// nothing, regardless of what it contains.
    #[test]
    fn history_too_short_drafts_nothing() {
        let config = NgramSimpleConfig {
            size_n: 12,
            size_m: 48,
        };
        let history: Vec<u32> = (0..60u32).collect();
        assert!(history.len() <= 12 + 48 + 1);
        let mut drafted = Vec::new();
        ngram_simple_draft(&config, &history, 999, &mut drafted);
        assert_eq!(drafted, Vec::<u32>::new());
    }

    /// Sad path: `history` long enough but with no repeated n-gram anywhere
    /// (strictly increasing ids never repeat a window) must draft nothing.
    #[test]
    fn no_matching_pattern_drafts_nothing() {
        let config = NgramSimpleConfig {
            size_n: 3,
            size_m: 3,
        };
        let history: Vec<u32> = (0..64u32).collect();
        let mut drafted = Vec::new();
        ngram_simple_draft(&config, &history, 9_999, &mut drafted);
        assert_eq!(drafted, Vec::<u32>::new());
    }

    /// The buffer-reuse contract [`ngram_simple_draft`]'s own doc claims:
    /// calling it a second time with a buffer that already holds a PRIOR
    /// draft (not the empty starting state every other test in this module
    /// uses) must leave `out` holding exactly the new draft, nothing carried
    /// over from the first call -- `clear` before every write, not append.
    #[test]
    fn a_reused_buffer_holding_a_stale_draft_is_fully_overwritten() {
        let config = NgramSimpleConfig {
            size_n: 3,
            size_m: 3,
        };
        let history = vec![0u32, 1, 2, 3, 4, 5, 9, 9, 1, 2];
        let mut drafted: Vec<u32> = vec![111, 222, 333, 444, 555];

        ngram_simple_draft(&config, &history, 3, &mut drafted);
        assert_eq!(
            drafted,
            vec![4u32, 5, 9],
            "a real match must fully overwrite whatever the buffer held before"
        );

        ngram_simple_draft(&config, &history, 9_999, &mut drafted);
        assert_eq!(
            drafted,
            Vec::<u32>::new(),
            "a non-match must leave the buffer empty, not the previous call's draft"
        );
    }
}
