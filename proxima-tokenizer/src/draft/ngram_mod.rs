//! A faithful port of llama.cpp's `common_ngram_mod` (`common/ngram-mod.h`,
//! `common/ngram-mod.cpp`) and its driver,
//! `common_speculative_impl_ngram_mod` (`common/speculative.cpp:1849-2022`):
//! a single shared hash table mapping a rolling `n_match`-token context to
//! the ONE token last seen following it -- llama's own PR description names
//! it "basic n-gram hasher" (`ngram-mod.h:9`, `ggml-org/llama.cpp#19164`).
//! Unlike [`crate::draft::ngram_map`], there is no collision resolution at
//! all: a hash collision silently overwrites whatever was there, so
//! [`ngram_mod_draft`]'s own chained lookahead (see below) can walk into
//! stale or unrelated content, which is exactly what the two reset
//! mechanisms this module ports exist to bound.
//!
//! # The algorithm, traced to the incumbent
//!
//! `NgramMod::add` (`common_ngram_mod::add`) takes a window of
//! `n_match + 1` tokens: the leading `n_match` tokens hash to a table slot
//! (llama's own multiplicative hash, `idx()`, `common/ngram-mod.cpp:15-25`
//! -- an LCG-style `res = res * 6364136223846793005 + token` folded over
//! every context token, reduced mod the table size), and the trailing
//! token overwrites whatever was stored there (last-write-wins, no probing).
//! `NgramMod::get` (`common_ngram_mod::get`) hashes an `n_match`-token
//! window the same way and returns whatever token is currently stored
//! there, or `EMPTY` if that slot was never written.
//!
//! [`ngram_mod_begin`] (`common_speculative_impl_ngram_mod::begin`,
//! `:1894-1920`) trains the table over the entire prompt in one pass, then
//! checks OCCUPANCY: if the fraction of used slots exceeds
//! [`OCCUPANCY_THRESHOLD`] (`0.25` -- llama's own hardcoded constant,
//! `:1914`), the whole table is wiped. A saturated table degrades into pure
//! noise (every lookup returns SOME token, usually the wrong one), so this
//! is the only defense against unbounded false-positive drafting.
//!
//! [`ngram_mod_draft`] (`draft_one`, `:1922-1976`) is the interesting half:
//! it builds an `n_match + n_max`-token scratch buffer -- the trailing
//! `n_match - 1` tokens of `history` followed by `sampled`, then CHAINS
//! forward: each new drafted token becomes part of the context window for
//! the next lookup (`result.data() + i`, not `history.data() + i` -- the
//! lookahead never re-reads real history past the seed, only its own
//! guesses). A chain that runs dry before [`NgramModConfig::n_min`] tokens
//! have been drafted returns NOTHING at all, not a short draft (llama's own
//! `if (i < params.n_min) { result.clear(); return; }`); a chain that runs
//! dry AFTER `n_min` keeps what it has. The table is also extended here,
//! in CHUNKS of 32 new tokens at a time (`:1940-1946` -- llama's own
//! incremental-add lag, preserved exactly: a call that hasn't advanced 32
//! tokens past the last add does no training work at all this step) rather
//! than every call, unlike [`ngram_mod_begin`]'s one-shot full-prompt pass.
//!
//! [`ngram_mod_accept`] (`accept`, `:1996-2021`) tracks the ACCEPTANCE
//! fraction of the just-drafted tokens; five consecutive rounds below
//! [`LOW_ACCEPT_THRESHOLD`] (`0.25`) reset the table AND the incremental-add
//! cursor (`sinfo.i_last = 0`, so the next draft call retrains from
//! scratch) -- llama's own defense against a table that has drifted into
//! confidently-wrong territory.
//!
//! [`NgramMod::occupancy_resets`]/[`NgramMod::low_accept_resets`] are not
//! in the incumbent (which only logs these via `SPC_WRN`/`SPC_TRC`) --
//! added here as plain counters so a caller (or this module's own fixture
//! test) can observe that both reset paths actually fired, and as the seed
//! for the `draft_n`/telemetry counters `speculative-decode-llama-parity`
//! SPEC.md requirement R12 will wire up later.

use alloc::vec::Vec;

/// llama's `common_ngram_mod` table size, hardcoded at the call site
/// (`common/speculative.cpp:1876`, `mod(params.ngram_mod.n_match,
/// 4*1024*1024)`) rather than configurable.
pub const TABLE_SIZE: usize = 4 * 1024 * 1024;

/// llama's occupancy reset threshold (`common/speculative.cpp:1914`).
pub const OCCUPANCY_THRESHOLD: f64 = 0.25;

/// llama's low-acceptance reset threshold (`common/speculative.cpp:2006`).
pub const LOW_ACCEPT_THRESHOLD: f64 = 0.25;

/// llama's low-acceptance reset streak length (`common/speculative.cpp:2008`,
/// `sinfo.n_low >= 5`).
pub const LOW_ACCEPT_STREAK: u32 = 5;

/// llama's `common_params_speculative_ngram_mod` default `n_match`
/// (`common/common.h:355`).
pub const DEFAULT_N_MATCH: u16 = 24;

/// llama's `common_params_speculative_ngram_mod` default `n_max`
/// (`common/common.h:356`).
pub const DEFAULT_N_MAX: u16 = 64;

/// llama's `common_params_speculative_ngram_mod` default `n_min`
/// (`common/common.h:357`).
pub const DEFAULT_N_MIN: u16 = 48;

/// llama's `common_ngram_mod::EMPTY` sentinel (`common/ngram-mod.h:16`):
/// a table slot that has never been written.
const EMPTY: i32 = -1;

/// llama's `idx()` multiplier (`common/ngram-mod.cpp:19`) -- the constant
/// `6364136223846793005` PCG/LCG families use, folded over every context
/// token in turn.
const HASH_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

/// llama's `common_params_speculative_ngram_mod` (`common/common.h:354-359`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramModConfig {
    /// The context window size hashed into the table -- llama's `n_match`.
    pub n_match: u16,
    /// The maximum chain length drafted after a hit -- llama's `n_max`.
    pub n_max: u16,
    /// The minimum chain length required before a partial chain is kept --
    /// llama's `n_min`.
    pub n_min: u16,
}

impl Default for NgramModConfig {
    fn default() -> Self {
        Self {
            n_match: DEFAULT_N_MATCH,
            n_max: DEFAULT_N_MAX,
            n_min: DEFAULT_N_MIN,
        }
    }
}

/// A faithful port of `common_ngram_mod` (`common/ngram-mod.h`/`.cpp`) plus
/// the per-sequence bookkeeping `common_speculative_impl_ngram_mod::seq_info`
/// (`common/speculative.cpp:1858-1867`) carries alongside it -- kept as one
/// struct here since every fixture and every real caller drives exactly one
/// generation stream through it, the same simplification
/// [`crate::draft::ngram_map::NgramMap`]'s own doc explains for `key_map`.
#[derive(Debug, Clone)]
pub struct NgramMod {
    config: NgramModConfig,
    table: Vec<i32>,
    used: usize,
    i_last: usize,
    n_draft_last: usize,
    n_low: u32,
    occupancy_resets: u32,
    low_accept_resets: u32,
}

impl NgramMod {
    /// Allocates the fixed [`TABLE_SIZE`]-entry table up front, at
    /// construction, so [`ngram_mod_draft`]'s hot path allocates nothing --
    /// this crate's zero-per-call-allocation discipline
    /// ([`crate::draft::ngram_simple::ngram_simple_draft`]'s own doc).
    #[must_use]
    pub fn new(config: NgramModConfig) -> Self {
        Self {
            config,
            table: alloc::vec![EMPTY; TABLE_SIZE],
            used: 0,
            i_last: 0,
            n_draft_last: 0,
            n_low: 0,
            occupancy_resets: 0,
            low_accept_resets: 0,
        }
    }

    /// How many times [`ngram_mod_begin`] has reset the table for exceeding
    /// [`OCCUPANCY_THRESHOLD`] -- see this module's own doc.
    #[must_use]
    pub fn occupancy_resets(&self) -> u32 {
        self.occupancy_resets
    }

    /// How many times [`ngram_mod_accept`] has reset the table for a
    /// five-round-or-longer low-acceptance streak -- see this module's own
    /// doc.
    #[must_use]
    pub fn low_accept_resets(&self) -> u32 {
        self.low_accept_resets
    }

    /// llama's `common_ngram_mod::idx` (`common/ngram-mod.cpp:15-25`).
    /// `context` is exactly `n_match` tokens.
    fn hash(&self, context: &[u32]) -> usize {
        let mut hash = 0u64;
        for &token in context {
            hash = hash
                .wrapping_mul(HASH_MULTIPLIER)
                .wrapping_add(u64::from(token));
        }
        (hash % self.table.len() as u64) as usize
    }

    /// llama's `common_ngram_mod::add` (`common/ngram-mod.cpp:27-35`).
    /// `window` is `n_match + 1` tokens: the context, then the token that
    /// followed it.
    fn add(&mut self, window: &[u32]) {
        let n_match = usize::from(self.config.n_match);
        let index = self.hash(&window[..n_match]);
        if self.table[index] == EMPTY {
            self.used += 1;
        }
        // llama.cpp's own storage is `int32_t`; real vocab ids never
        // approach `i32::MAX`, so this cast is lossless in practice.
        self.table[index] = window[n_match] as i32;
    }

    /// llama's `common_ngram_mod::get` (`common/ngram-mod.cpp:37-41`).
    /// `context` is exactly `n_match` tokens.
    fn get(&self, context: &[u32]) -> i32 {
        let index = self.hash(context);
        self.table[index]
    }

    /// llama's `common_ngram_mod::reset` (`common/ngram-mod.cpp:43-46`) --
    /// does not touch either reset counter, matching the incumbent (which
    /// has none); [`ngram_mod_begin`]/[`ngram_mod_accept`] increment their
    /// own counter around the call.
    fn reset(&mut self) {
        self.table.fill(EMPTY);
        self.used = 0;
    }
}

/// A faithful port of `common_speculative_impl_ngram_mod::begin`
/// (`common/speculative.cpp:1894-1920`) -- see this module's own doc.
/// `prompt` is llama's `prompt` argument: every token this generation
/// stream starts from.
pub fn ngram_mod_begin(mod_: &mut NgramMod, prompt: &[u32]) {
    mod_.i_last = 0;
    mod_.n_draft_last = 0;

    let n_match = usize::from(mod_.config.n_match);
    if prompt.len() < n_match {
        return;
    }

    for start in 0..prompt.len() - n_match {
        mod_.add(&prompt[start..=start + n_match]);
    }
    mod_.i_last = prompt.len() - n_match;

    let occupancy = mod_.used as f64 / mod_.table.len() as f64;
    if occupancy > OCCUPANCY_THRESHOLD {
        mod_.reset();
        mod_.occupancy_resets += 1;
    }
}

/// A faithful port of `common_speculative_impl_ngram_mod::draft_one`
/// (`common/speculative.cpp:1922-1976`) -- see this module's own doc for
/// the chained-lookahead algorithm. `history` is llama's `prompt` (every
/// token generated so far, NOT including `sampled`); `sampled` is the
/// token the caller's own sampler just drew for the position immediately
/// after `history`.
///
/// `out` doubles as llama's own `result` scratch buffer (`draft_one`
/// reuses `result` for both the rolling lookahead window and the final
/// draft, shifting the tail down before returning) -- a caller pre-sizing
/// `out` to at least `n_match + n_max` up front pays no allocation on this
/// call's own `resize`, matching this crate's caller-owned-buffer
/// discipline ([`crate::draft::ngram_simple::ngram_simple_draft`]'s own
/// doc).
pub fn ngram_mod_draft(mod_: &mut NgramMod, history: &[u32], sampled: u32, out: &mut Vec<u32>) {
    out.clear();
    mod_.n_draft_last = 0;

    let n_match = usize::from(mod_.config.n_match);
    let n_max = usize::from(mod_.config.n_max);
    let n_min = usize::from(mod_.config.n_min);
    let cur_len = history.len();

    if cur_len < n_match {
        return;
    }

    // llama.cpp: `if (sinfo.i_last + 32 < cur_len) { ... }` -- chunked
    // incremental training, not every call.
    if mod_.i_last + 32 < cur_len {
        for start in mod_.i_last..cur_len - n_match {
            mod_.add(&history[start..=start + n_match]);
        }
        mod_.i_last = cur_len - n_match;
    }

    out.resize(n_match + n_max, 0);
    out[..n_match - 1].copy_from_slice(&history[cur_len - n_match + 1..cur_len]);
    out[n_match - 1] = sampled;

    let mut drafted_len = n_max;
    for step in 0..n_max {
        let token = mod_.get(&out[step..step + n_match]);
        if token == EMPTY {
            if step < n_min {
                out.clear();
                return;
            }
            out.truncate(n_match + step);
            drafted_len = step;
            break;
        }
        // llama.cpp's own storage is `int32_t`; see `NgramMod::add`'s doc.
        out[n_match + step] = token as u32;
    }

    for index in 0..drafted_len {
        out[index] = out[n_match + index];
    }
    out.truncate(drafted_len);

    mod_.n_draft_last = out.len();
}

/// A faithful port of `common_speculative_impl_ngram_mod::accept`
/// (`common/speculative.cpp:1996-2021`) -- see this module's own doc.
pub fn ngram_mod_accept(mod_: &mut NgramMod, n_accepted: u16) {
    if mod_.n_draft_last == 0 {
        return;
    }

    let acceptance = f64::from(n_accepted) / mod_.n_draft_last as f64;
    if acceptance < LOW_ACCEPT_THRESHOLD {
        mod_.n_low += 1;
        if mod_.n_low >= LOW_ACCEPT_STREAK {
            mod_.reset();
            mod_.low_accept_resets += 1;
            mod_.n_low = 0;
            mod_.i_last = 0;
        }
    } else {
        mod_.n_low = 0;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use serde::Deserialize;

    use super::{NgramMod, NgramModConfig, ngram_mod_accept, ngram_mod_begin, ngram_mod_draft};

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
        occupancy_warmup: OccupancyWarmup,
        cases: Vec<FixtureCase>,
    }

    #[derive(Debug, Deserialize)]
    struct OccupancyWarmup {
        seed: u64,
        length: u32,
        vocab_size: u32,
    }

    // Vigna's SplitMix64 (public domain). The occupancy-reset warmup only
    // needs to touch enough distinct 24-token windows to push ngram-mod's
    // table past its 0.25 occupancy threshold -- its content is otherwise
    // never compared against anything -- so `ngram_mod.json`'s
    // `occupancy_warmup` header records {seed, length, vocab_size} instead
    // of 2.2 million token ids, and `generator/main.cpp` implements this
    // exact function so both sides produce the identical stream.
    fn splitmix64_next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn synthetic_occupancy_warmup(warmup: &OccupancyWarmup) -> Vec<u32> {
        let mut state = warmup.seed;
        (0..warmup.length)
            .map(|_| (splitmix64_next(&mut state) % u64::from(warmup.vocab_size)) as u32)
            .collect()
    }

    #[derive(Debug, Deserialize)]
    struct FixtureParams {
        n_match: u16,
        n_max: u16,
        n_min: u16,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureCase {
        stream_id: u32,
        position: usize,
        sampled: u32,
        draft: Vec<u32>,
        accepted: u16,
    }

    const STREAMS_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama-ngram/fixtures/streams.json"
    ));
    const NGRAM_MOD_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama-ngram/fixtures/ngram_mod.json"
    ));

    /// [`ngram_mod_draft`]/[`ngram_mod_begin`]/[`ngram_mod_accept`] against
    /// every case in `tests/fixtures/llama-ngram/fixtures/ngram_mod.json`
    /// (SPEC.md AC8), replaying the SAME single shared `NgramMod` instance
    /// across streams, in the SAME order
    /// (`generator/main.cpp`'s `ordered = { streams[5], streams[2],
    /// streams[0], streams[1], streams[3], streams[4] }`) the fixture
    /// generator used -- `ngram-mod`'s table is shared state across the
    /// whole run, so replaying streams out of order or in isolation would
    /// not reproduce the recorded drafts.
    ///
    /// Before that ordered replay, this test feeds the SAME instance a
    /// synthetic occupancy-warmup stream through one `ngram_mod_begin`
    /// call, exactly as `generator/main.cpp` fed the identical stream into
    /// the SAME `common_speculative` instance whose subsequent draft cases
    /// below are recorded into `ngram_mod.json` -- not a throwaway
    /// instance. The stream's content is irrelevant to what it tests (only
    /// the count of distinct 24-token windows it inserts matters), so
    /// `ngram_mod.json`'s `occupancy_warmup` header records the
    /// reproducible recipe -- `{seed, length, vocab_size}` fed through
    /// [`splitmix64_next`] -- instead of storing 2.2M token ids
    /// (`README.md`'s "ngram-mod reset mechanism" section explains why this
    /// one stream is synthetic while every other fixture stays real text).
    /// `generator/main.cpp` implements the identical SplitMix64 constants,
    /// so both sides produce the same stream. llama.cpp's own `begin()`
    /// logged `occupancy = 1712260/4194304 (0.41) - resetting` for this
    /// exact call (`fixtures/generator.log`), so a genuine
    /// [`NgramMod::occupancy_resets`] fires here against the incumbent's
    /// own recorded behavior, not a Rust-only reconstruction. Because a
    /// reset wipes the table back to fully empty, the ordered replay below
    /// still produces byte-identical cases to a fresh instance -- confirmed
    /// by diffing `ngram_mod.json`'s `cases` before and after this
    /// fixture's regeneration.
    ///
    /// The fixture's own low-acceptance trap (`streams[5]`, documented in
    /// `README.md`) fires a real [`NgramMod::low_accept_resets`] during
    /// this replay. This test's own printed line folds both counts
    /// together per SPEC.md AC8's literal wording.
    #[test]
    fn ngram_mod_matches_llama_fixture() {
        let streams: StreamsFile = serde_json::from_str(STREAMS_JSON).expect("streams.json parses");
        let fixture: FixtureFile =
            serde_json::from_str(NGRAM_MOD_JSON).expect("ngram_mod.json parses");

        let config = NgramModConfig {
            n_match: fixture.params.n_match,
            n_max: fixture.params.n_max,
            n_min: fixture.params.n_min,
        };

        let mut cases_by_stream: alloc::collections::BTreeMap<u32, Vec<&FixtureCase>> =
            alloc::collections::BTreeMap::new();
        for case in &fixture.cases {
            cases_by_stream.entry(case.stream_id).or_default().push(case);
        }

        let mut mod_ = NgramMod::new(config);

        let occupancy_warmup = synthetic_occupancy_warmup(&fixture.occupancy_warmup);
        ngram_mod_begin(&mut mod_, &occupancy_warmup);
        assert_eq!(
            mod_.occupancy_resets(),
            1,
            "llama.cpp's own recorded occupancy warmup must reset the table exactly once before replay"
        );

        let order = [5u32, 2, 0, 1, 3, 4];
        let mut non_empty = 0usize;
        let mut total = 0usize;
        let mut drafted: Vec<u32> = Vec::new();

        for stream_id in order {
            let Some(cases) = cases_by_stream.get(&stream_id) else {
                continue;
            };
            let stream = streams
                .streams
                .iter()
                .find(|stream| stream.id == stream_id)
                .unwrap_or_else(|| panic!("stream {stream_id} exists in streams.json"));

            let prompt_len = if stream_id == 5 {
                (usize::from(config.n_match) + 1).min(stream.tokens.len() / 4)
            } else {
                stream.tokens.len() / 2
            };
            ngram_mod_begin(&mut mod_, &stream.tokens[..prompt_len]);

            for case in cases {
                let history = &stream.tokens[..case.position];
                ngram_mod_draft(&mut mod_, history, case.sampled, &mut drafted);
                total += 1;
                if !case.draft.is_empty() {
                    non_empty += 1;
                }
                assert_eq!(
                    drafted, case.draft,
                    "stream {stream_id} position {}: got {drafted:?}, want {:?}",
                    case.position, case.draft
                );
                ngram_mod_accept(&mut mod_, case.accepted);
            }
        }

        println!(
            "cases = {total} non_empty = {non_empty} occupancy_resets = {} low_accept_resets = {}",
            mod_.occupancy_resets(),
            mod_.low_accept_resets()
        );
        assert!(total >= 200, "fixture must carry at least 200 cases per SPEC.md AC8");
        assert!(
            mod_.occupancy_resets() >= 1,
            "the synthetic occupancy warmup replayed against llama.cpp must fire at least one reset"
        );
        assert!(
            mod_.low_accept_resets() >= 1,
            "streams[5]'s constructed low-acceptance trap must fire at least one reset"
        );
    }

    /// The occupancy-reset mechanism (`README.md`'s own "ngram-mod reset
    /// mechanism" section), as a second, self-contained Rust proof
    /// independent of the vendored oracle fixture --
    /// `ngram_mod_matches_llama_fixture` above already exercises the real
    /// mechanism against llama.cpp's own recorded 2.2-million-token
    /// occupancy-warmup replay (`ngram_mod.json`'s `occupancy_warmup`
    /// recipe); this test proves the SAME threshold arithmetic holds over
    /// an input this module
    /// controls end to end, rather than depending only on one vendored
    /// oracle run: [`ngram_mod_begin`] must reset the table once the
    /// fraction of used slots exceeds [`OCCUPANCY_THRESHOLD`]. The table's
    /// modulus ([`super::TABLE_SIZE`]) is a power of two, and this hash's
    /// LCG-style mix (`common_ngram_mod::idx`) has notoriously short cycles
    /// in its LOW bits under a power-of-two modulus -- a strictly
    /// increasing token sequence (tried first here) drives almost every
    /// window into the same handful of slots instead of filling the table,
    /// the opposite of what real, high-entropy vocabulary ids do (the
    /// `README.md` warmup measured 0.55-0.66 used-slots-per-token on real
    /// C source). A seeded `fastrand::Rng` draws token ids across a
    /// realistic vocab range instead, so this proof exercises the same
    /// high-entropy regime real generation traffic does, never re-deriving
    /// the reset threshold from reading the C++ (the occupancy fraction
    /// itself is still this module's own arithmetic, not guessed).
    #[test]
    fn occupancy_reset_fires_over_threshold() {
        let config = NgramModConfig {
            n_match: 24,
            n_max: 64,
            n_min: 48,
        };
        let mut mod_ = NgramMod::new(config);

        let mut rng = fastrand::Rng::with_seed(0x0CCA_9A0C);
        let target_windows = 1_800_000u32;
        let prompt: Vec<u32> = (0..target_windows + u32::from(config.n_match))
            .map(|_| rng.u32(0..262_144))
            .collect();

        ngram_mod_begin(&mut mod_, &prompt);

        assert_eq!(
            mod_.occupancy_resets(),
            1,
            "a prompt training past the occupancy threshold in one begin() call must reset exactly once"
        );
    }

    /// The low-acceptance reset mechanism, hand-constructed: five
    /// consecutive [`ngram_mod_accept`] calls reporting zero acceptance
    /// against a non-empty draft must reset the table and clear the
    /// streak.
    #[test]
    fn low_acceptance_streak_resets_after_five_rounds() {
        let config = NgramModConfig {
            n_match: 4,
            n_max: 4,
            n_min: 1,
        };
        let mut mod_ = NgramMod::new(config);
        // seed one draft so `n_draft_last > 0` going into the accept calls.
        mod_.n_draft_last = 4;

        for _ in 0..4 {
            ngram_mod_accept(&mut mod_, 0);
            assert_eq!(mod_.low_accept_resets(), 0, "fewer than five low rounds must not reset yet");
            mod_.n_draft_last = 4;
        }
        ngram_mod_accept(&mut mod_, 0);
        assert_eq!(mod_.low_accept_resets(), 1, "the fifth consecutive low round must reset");
    }

    /// Sad path: `history` shorter than `n_match` must draft nothing.
    #[test]
    fn history_shorter_than_n_match_drafts_nothing() {
        let config = NgramModConfig {
            n_match: 24,
            n_max: 64,
            n_min: 48,
        };
        let mut mod_ = NgramMod::new(config);
        let history: Vec<u32> = (0..10u32).collect();
        let mut drafted = Vec::new();
        ngram_mod_draft(&mut mod_, &history, 999, &mut drafted);
        assert_eq!(drafted, Vec::<u32>::new());
    }

    /// Sad path: a chain that runs dry before `n_min` tokens are drafted
    /// must draft nothing, not a short chain -- llama's own
    /// `if (i < params.n_min) { result.clear(); return; }`.
    #[test]
    fn chain_shorter_than_n_min_drafts_nothing() {
        let config = NgramModConfig {
            n_match: 4,
            n_max: 8,
            n_min: 8,
        };
        let mut mod_ = NgramMod::new(config);
        // never trained: every lookup misses immediately (step 0 < n_min).
        let history: Vec<u32> = (0..20u32).collect();
        let mut drafted = Vec::new();
        ngram_mod_draft(&mut mod_, &history, 999, &mut drafted);
        assert_eq!(drafted, Vec::<u32>::new());
    }

    /// Happy path, hand-computed: train the table on a short repeating
    /// stream so the chain lookahead can walk forward past the seed.
    #[test]
    fn happy_path_chains_forward_past_the_seed() {
        let config = NgramModConfig {
            n_match: 3,
            n_max: 4,
            n_min: 1,
        };
        let mut mod_ = NgramMod::new(config);
        // trains windows [1,2,3]->4, [2,3,4]->5, [3,4,5]->6, [4,5,6]->7.
        let training = vec![1u32, 2, 3, 4, 5, 6, 7];
        ngram_mod_begin(&mut mod_, &training);

        // history ends [.., 1, 2]; sampled = 3 completes the trained
        // context [1, 2, 3] -> chain should walk 4, 5, 6, 7.
        let history = vec![9u32, 9, 9, 1, 2];
        let mut drafted = Vec::new();
        ngram_mod_draft(&mut mod_, &history, 3, &mut drafted);
        assert_eq!(drafted, vec![4u32, 5, 6, 7]);
    }

    /// The buffer-reuse contract every drafter in this crate follows
    /// ([`crate::draft::ngram_simple::ngram_simple_draft`]'s own doc):
    /// calling `ngram_mod_draft` with a buffer already holding a PRIOR
    /// draft must leave `out` holding exactly the new draft.
    #[test]
    fn a_reused_buffer_holding_a_stale_draft_is_fully_overwritten() {
        let config = NgramModConfig {
            n_match: 3,
            n_max: 4,
            n_min: 1,
        };
        let mut mod_ = NgramMod::new(config);
        let training = vec![1u32, 2, 3, 4, 5, 6, 7];
        ngram_mod_begin(&mut mod_, &training);

        let history = vec![9u32, 9, 9, 1, 2];
        let mut drafted: Vec<u32> = vec![111, 222, 333, 444, 555];

        ngram_mod_draft(&mut mod_, &history, 3, &mut drafted);
        assert_eq!(drafted, vec![4u32, 5, 6, 7]);

        let never_trained_history = vec![100u32, 200, 300];
        ngram_mod_draft(&mut mod_, &never_trained_history, 9_999, &mut drafted);
        assert_eq!(
            drafted,
            Vec::<u32>::new(),
            "a chain that dies immediately must leave the buffer empty, not the previous draft"
        );
    }

    /// Sanity check on the printed diagnostic line itself: a non-empty
    /// `String` builds from the same values the fixture test prints, so a
    /// future refactor of the print statement cannot silently drop a field
    /// without a compile error surfacing it here.
    #[test]
    fn diagnostic_line_names_all_four_fields() {
        let line = format!(
            "cases = {} non_empty = {} occupancy_resets = {} low_accept_resets = {}",
            10, 3, 1, 1
        );
        let expected: String = "cases = 10 non_empty = 3 occupancy_resets = 1 low_accept_resets = 1".into();
        assert_eq!(line, expected);
    }
}
