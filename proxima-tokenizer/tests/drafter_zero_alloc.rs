//! `speculative-decode-llama-parity/SPEC.md` AC22 (R18): a drafter's
//! `draft()` call allocates nothing once constructed. [`proxima_test`]'s
//! [`CountingAllocator`] is the workspace's own shared zero-allocation proof
//! gate (`proxima-test/src/alloc_count.rs`) -- reused here rather than
//! re-minting a per-crate counting allocator, per guiding principle 1.
//!
//! One case per n-gram TYPE, per `TASKS.md` slice 21's own note: ngram-simple,
//! ngram-map key-only, ngram-map four-value, ngram-mod, and ngram-cache each
//! land their own test here (five in total) -- `key_only`/`key4v` are the
//! SAME `ngram_map_draft` function ([`ngram_map`]'s own module doc), but
//! they walk different branches (`draft_key_only` vs `draft_k4v`) with
//! independent allocation profiles worth proving separately, matching
//! `README.md`'s own "ngram-map key_only vs k4v divergence" framing of the
//! two as distinct configurations of one port. The `Drafter` enum dispatch
//! itself (slice 9) joins once it exists.
//!
//! `drafter_zero_alloc_ngram_cache` measures the PER-STEP path a real
//! decode loop drives: [`NgramCacheState`], presized via
//! [`NgramCacheState::new`]'s own `max_context_len` bound (`ngram_cache`'s
//! own module doc, "Zero allocation" section), so
//! [`ngram_cache_state_draft`]'s own internal `ngram_cache_update_delta`
//! call -- the maintenance step every earlier version of this test skipped
//! by measuring only a cache built once outside the window -- is now inside
//! the measured 100 000-call loop alongside the draft itself.
//!
//! `#[global_allocator]` is process-wide, so it lives once, at the top of
//! this binary -- `cargo nextest` gives every `tests/*.rs` target its own
//! process, so this counter is never shared with the crate's own
//! `#[cfg(test)]` unit tests.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_test::alloc_count::{CountingAllocator, allocations};
use proxima_tokenizer::draft::{
    LLAMA_NGRAM_MAX, NgramCacheState, NgramMapConfig, NgramModConfig, NgramSimpleConfig,
    ngram_cache_state_draft, ngram_map_accept, ngram_map_begin, ngram_map_draft, ngram_mod_accept,
    ngram_mod_begin, ngram_mod_draft, ngram_simple_draft,
};
use serde::Deserialize;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const ITERATIONS: usize = 100_000;

#[derive(Debug, Deserialize)]
struct StreamsFile {
    streams: Vec<Stream>,
}

#[derive(Debug, Deserialize)]
struct Stream {
    id: u32,
    tokens: Vec<u32>,
}

const STREAMS_JSON: &str = include_str!(
    "fixtures/llama-ngram/fixtures/streams.json"
);

/// Stream 4 (`streams.json`): 10 502 real gemma4-tokenized ids, the longest
/// stream the fixture generator vendored -- long enough that cycling
/// `position` across it for 100 000 calls exercises many distinct history
/// windows rather than replaying one fixed slice.
fn real_stream() -> Vec<u32> {
    let streams: StreamsFile = serde_json::from_str(STREAMS_JSON).expect("streams.json parses");
    streams
        .streams
        .iter()
        .find(|stream| stream.id == 4)
        .expect("stream 4 exists in streams.json")
        .tokens
        .clone()
}

/// [`ngram_simple_draft`]: stateless, so the proof is a straight loop over
/// real positions in [`real_stream`], `out` pre-sized to `size_m` once
/// before the measured window so every call's `extend_from_slice` writes
/// into existing capacity.
#[test]
fn drafter_zero_alloc_ngram_simple() {
    let tokens = real_stream();
    let config = NgramSimpleConfig {
        size_n: 12,
        size_m: 48,
    };
    let min_len = usize::from(config.size_n) + usize::from(config.size_m) + 2;
    let span = tokens.len() - min_len - 1;

    let mut drafted: Vec<u32> = Vec::with_capacity(usize::from(config.size_m));

    // warm-up outside the measured window: first call into a freshly grown
    // process may still pay one-time allocator bookkeeping (arena carve-out)
    // that is not this function's own allocation.
    let warmup_position = min_len + 1;
    ngram_simple_draft(
        &config,
        &tokens[..warmup_position],
        tokens[warmup_position],
        &mut drafted,
    );

    let before = allocations();
    for step in 0..ITERATIONS {
        let position = min_len + 1 + (step % span);
        let sampled = tokens[position];
        ngram_simple_draft(&config, &tokens[..position], sampled, &mut drafted);
    }
    let after = allocations();

    println!("allocs = {} over {ITERATIONS} calls", after - before);
    assert_eq!(after, before, "ngram_simple_draft must not allocate on its hot path");
}

/// Shared setup + measured loop for both [`drafter_zero_alloc_ngram_map_key_only`]
/// and [`drafter_zero_alloc_ngram_map_four_value`]: stateful, so construction
/// (`NgramMap::new`) and one `ngram_map_begin` call are the setup path,
/// outside the measured window -- both allocate once, per
/// `NgramMap::new`'s own doc, and that is the allocation budget each test
/// proves is paid exactly once, not per call. `key_only` selects
/// `draft_key_only` vs `draft_k4v` ([`ngram_map`]'s own module doc) --
/// distinct branches with their own allocation profile, so each gets its
/// own proof rather than one test standing in for both.
fn run_ngram_map_zero_alloc(key_only: bool) -> usize {
    let tokens = real_stream();
    let config = NgramMapConfig {
        size_key: 12,
        size_value: 48,
        key_only,
        min_hits: 1,
    };
    let min_len = 2 * usize::from(config.size_key) + usize::from(config.size_value) + 2;
    let span = tokens.len() - min_len - 1;

    let mut map = proxima_tokenizer::draft::NgramMap::new(config, tokens.len());
    let prompt_len = min_len.min(tokens.len() / 2);
    ngram_map_begin(&mut map, &tokens[..prompt_len]);

    let mut drafted: Vec<u32> = Vec::with_capacity(usize::from(config.size_value));

    let before = allocations();
    for step in 0..ITERATIONS {
        let position = min_len + 1 + (step % span);
        let sampled = tokens[position];
        ngram_map_draft(&mut map, &tokens[..position], sampled, &mut drafted);
        ngram_map_accept(&mut map, drafted.len() as u16);
    }
    let after = allocations();
    after - before
}

/// `ngram-map-k`'s own allocation profile: `draft_key_only` always drafts
/// from `values[0]` with no tally loop (`ngram_map`'s own module doc).
#[test]
fn drafter_zero_alloc_ngram_map_key_only() {
    let allocs = run_ngram_map_zero_alloc(true);
    println!("allocs = {allocs} over {ITERATIONS} calls");
    assert_eq!(allocs, 0, "ngram_map_draft/ngram_map_accept (key_only) must not allocate on their hot path");
}

/// `ngram-map-k4v`'s own allocation profile: `draft_k4v` tallies up to
/// `MAX_VALUES` continuations per key and applies the dominance guard
/// (`ngram_map`'s own module doc) -- a different code path from `key_only`,
/// so it gets its own zero-alloc proof.
#[test]
fn drafter_zero_alloc_ngram_map_four_value() {
    let allocs = run_ngram_map_zero_alloc(false);
    println!("allocs = {allocs} over {ITERATIONS} calls");
    assert_eq!(allocs, 0, "ngram_map_draft/ngram_map_accept (key4v) must not allocate on their hot path");
}

/// [`ngram_mod_draft`]/[`ngram_mod_begin`]/[`ngram_mod_accept`]: stateful,
/// so construction (`NgramMod::new`, which allocates the fixed
/// `TABLE_SIZE`-entry table once) and one `ngram_mod_begin` call are the
/// setup path, outside the measured window. `drafted` is pre-sized to
/// `n_match + n_max`, the scratch capacity `ngram_mod_draft`'s own doc
/// says it reuses for both the rolling lookahead window and the final
/// draft.
#[test]
fn drafter_zero_alloc_ngram_mod() {
    let tokens = real_stream();
    let config = NgramModConfig {
        n_match: 24,
        n_max: 64,
        n_min: 48,
    };
    let min_len = usize::from(config.n_match);
    let span = tokens.len() - min_len - 1;

    let mut mod_ = proxima_tokenizer::draft::NgramMod::new(config);
    let prompt_len = tokens.len() / 2;
    ngram_mod_begin(&mut mod_, &tokens[..prompt_len]);

    let mut drafted: Vec<u32> =
        Vec::with_capacity(usize::from(config.n_match) + usize::from(config.n_max));

    let before = allocations();
    for step in 0..ITERATIONS {
        let position = min_len + 1 + (step % span);
        let sampled = tokens[position];
        ngram_mod_draft(&mut mod_, &tokens[..position], sampled, &mut drafted);
        ngram_mod_accept(&mut mod_, drafted.len() as u16);
    }
    let after = allocations();

    println!("allocs = {} over {ITERATIONS} calls", after - before);
    assert_eq!(after, before, "ngram_mod_draft/ngram_mod_accept must not allocate on their hot path");
}

/// [`ngram_cache_state_draft`]'s PER-STEP path: update-then-draft, exactly
/// what a real decode loop drives, not just [`ngram_cache_draft`]'s pure
/// query path against a cache built once outside the window. `history`
/// walks STRICTLY FORWARD across a real token stream cycled out to
/// `ITERATIONS` positions -- `NgramCacheState`'s own `cache_size`
/// bookkeeping requires monotonically growing history (unlike the other
/// drafters' own modulo-cycled position, which would make `cache_size`
/// overtake `history.len()` and underflow
/// [`ngram_cache_update_delta`](proxima_tokenizer::draft::ngram_cache)'s
/// internal subtraction). [`NgramCacheState::new`] presizes the context
/// cache from `tokens.len()`, the exact upper bound on tokens this state
/// will ever see across the measured window (`ngram_cache`'s own module
/// doc, "Zero allocation" section).
#[test]
fn drafter_zero_alloc_ngram_cache() {
    let source = real_stream();
    let min_len = LLAMA_NGRAM_MAX;
    let total_len = min_len + ITERATIONS + 4;
    let tokens: Vec<u32> = source.iter().copied().cycle().take(total_len).collect();

    let mut state = NgramCacheState::new(tokens.len());
    let mut drafted: Vec<u32> = Vec::with_capacity(8);

    // warm-up outside the measured window: first call into a freshly grown
    // process may still pay one-time allocator bookkeeping (arena carve-out)
    // that is not this function's own allocation.
    let warmup_position = min_len + 1;
    ngram_cache_state_draft(&mut state, &tokens[..warmup_position], tokens[warmup_position], 8, &mut drafted);

    let before = allocations();
    for step in 0..ITERATIONS {
        let position = warmup_position + 1 + step;
        let sampled = tokens[position];
        ngram_cache_state_draft(&mut state, &tokens[..position], sampled, 8, &mut drafted);
    }
    let after = allocations();

    println!("allocs = {} over {ITERATIONS} calls", after - before);
    assert_eq!(after, before, "ngram_cache_state_draft's update+draft per-step path must not allocate");
}

/// Degenerate control: proves [`CountingAllocator`] is actually live and
/// wired to this binary, not silently no-op'd out -- without this, the two
/// tests above asserting `after == before` would pass identically whether
/// the allocator were counting or dead.
#[test]
fn drafter_zero_alloc_control_detects_a_deliberate_allocation() {
    let before = allocations();
    let boxed: Vec<u8> = Vec::with_capacity(1);
    let after = allocations();
    assert!(
        after > before,
        "the counting allocator must observe a deliberate Vec::with_capacity(1) allocation"
    );
    drop(boxed);
}
