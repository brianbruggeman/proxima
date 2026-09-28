//! `speculative-decode-llama-parity/SPEC.md` AC22 (R18): a drafter's
//! `draft()` call allocates nothing once constructed. [`proxima_test`]'s
//! [`CountingAllocator`] is the workspace's own shared zero-allocation proof
//! gate (`proxima-test/src/alloc_count.rs`) -- reused here rather than
//! re-minting a per-crate counting allocator, per guiding principle 1.
//!
//! One drafter per test, per `TASKS.md` slice 21's own note: ngram-simple,
//! ngram-map, ngram-mod, and ngram-cache land here; the fifth type (the
//! `Drafter` enum dispatch itself, slice 9) joins once it exists.
//!
//! `ngram_cache`'s own module doc names an exception this file's
//! `drafter_zero_alloc_ngram_cache` case honors rather than hides
//! (SPEC.md invariant 3): only `ngram_cache_draft`'s PURE query path is
//! zero-alloc -- cache MAINTENANCE (`ngram_cache_update`) genuinely grows
//! an `alloc::collections::BTreeMap` per newly-observed n-gram, unboundedly,
//! by the incumbent's own design. This case proves the query path directly
//! against a cache built once, outside the measured window, rather than
//! going through the stateful `NgramCacheState` wrapper (whose own
//! per-call maintenance step is exactly the part that cannot make a
//! zero-alloc claim).
//!
//! `#[global_allocator]` is process-wide, so it lives once, at the top of
//! this binary -- `cargo nextest` gives every `tests/*.rs` target its own
//! process, so this counter is never shared with the crate's own
//! `#[cfg(test)]` unit tests.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_test::alloc_count::{CountingAllocator, allocations};
use proxima_tokenizer::draft::{
    LLAMA_NGRAM_MAX, NgramCache, NgramMapConfig, NgramModConfig, NgramSimpleConfig,
    ngram_cache_draft, ngram_cache_update, ngram_map_accept, ngram_map_begin, ngram_map_draft,
    ngram_mod_accept, ngram_mod_begin, ngram_mod_draft, ngram_simple_draft,
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

/// [`ngram_map_draft`]/[`ngram_map_accept`]: stateful, so construction
/// (`NgramMap::new`) and one `ngram_map_begin` call are the setup path,
/// outside the measured window -- both allocate once, per
/// `NgramMap::new`'s own doc, and that is the allocation budget this test
/// proves is paid exactly once, not per call.
#[test]
fn drafter_zero_alloc_ngram_map() {
    let tokens = real_stream();
    let config = NgramMapConfig {
        size_key: 12,
        size_value: 48,
        key_only: true,
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

    println!("allocs = {} over {ITERATIONS} calls", after - before);
    assert_eq!(after, before, "ngram_map_draft/ngram_map_accept must not allocate on their hot path");
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

/// [`ngram_cache_draft`]'s pure query path: the context cache is built ONCE
/// via [`ngram_cache_update`] over the whole real stream, outside the
/// measured window (cache maintenance is the part this module's own doc
/// names as genuinely unbounded, per SPEC.md invariant 3) -- every
/// measured call then only READS that already-built cache and writes into
/// the caller-owned `drafted` buffer.
#[test]
fn drafter_zero_alloc_ngram_cache() {
    let tokens = real_stream();
    let mut context = NgramCache::new();
    ngram_cache_update(&mut context, &tokens, tokens.len());
    let dynamic = NgramCache::new();
    let static_cache = NgramCache::new();

    let min_len = LLAMA_NGRAM_MAX;
    let span = tokens.len() - min_len - 2;

    let mut drafted: Vec<u32> = Vec::with_capacity(64);

    let warmup_position = min_len + 1;
    ngram_cache_draft(
        &tokens[..warmup_position],
        tokens[warmup_position],
        8,
        &context,
        &dynamic,
        &static_cache,
        &mut drafted,
    );

    let before = allocations();
    for step in 0..ITERATIONS {
        let position = min_len + 1 + (step % span);
        let sampled = tokens[position];
        ngram_cache_draft(
            &tokens[..position],
            sampled,
            8,
            &context,
            &dynamic,
            &static_cache,
            &mut drafted,
        );
    }
    let after = allocations();

    println!("allocs = {} over {ITERATIONS} calls", after - before);
    assert_eq!(after, before, "ngram_cache_draft's pure query path must not allocate");
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
