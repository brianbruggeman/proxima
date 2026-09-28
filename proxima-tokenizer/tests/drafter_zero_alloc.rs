//! `speculative-decode-llama-parity/SPEC.md` AC22 (R18): a drafter's
//! `draft()` call allocates nothing once constructed. [`proxima_test`]'s
//! [`CountingAllocator`] is the workspace's own shared zero-allocation proof
//! gate (`proxima-test/src/alloc_count.rs`) -- reused here rather than
//! re-minting a per-crate counting allocator, per guiding principle 1.
//!
//! One drafter per test, per `TASKS.md` slice 21's own note: ngram-simple
//! and ngram-map land here; ngram-mod and ngram-cache join once their own
//! slices (6, 7) land, and the fifth type (the `Drafter` enum dispatch
//! itself, slice 9) once it exists.
//!
//! `#[global_allocator]` is process-wide, so it lives once, at the top of
//! this binary -- `cargo nextest` gives every `tests/*.rs` target its own
//! process, so this counter is never shared with the crate's own
//! `#[cfg(test)]` unit tests.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_test::alloc_count::{CountingAllocator, allocations};
use proxima_tokenizer::draft::{
    NgramMapConfig, NgramSimpleConfig, ngram_map_accept, ngram_map_begin, ngram_map_draft,
    ngram_simple_draft,
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
