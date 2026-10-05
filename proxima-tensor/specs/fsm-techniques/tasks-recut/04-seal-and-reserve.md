# slice 4 (re-cut): the seal hook for kv blocks and the placement hook for device kv (cards FT4.0 - FT4.14 and FT4.16 - FT4.23)

anchors read at main e4cf9beb (full sha e4cf9beb8342a80447342f0813ada9c84fc3a519) with `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`. Every `path::symbol (~line N)` below is the line at that sha; the executor re-locates by symbol. Paths are relative to the proxima repo root. Every card: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_<n>` (n is the card number), removed when the card is done.

Governing direction (owner, 2026-10-04): the hooks are built so techniques can be vetted later; a technique is not built here. This slice builds two hooks and proves each with a test:
- the seal hook (stage "seal and tier"): a pure decision in proxima-core says which blocks of a layer are sealed, a sealed row cannot be rewound by an in-flight decode, and each sealed block can be folded into a per-block summary record by a function the caller supplies. Nothing in the library names a technique; "key min and max" appears only as the test that proves the summary slot holds it.
- the placement hook (stage "place"): where the device-resident kv buffers live is chosen by a buffer source the model carries. The default source is today's allocator, so default behaviour is unchanged; a test supplies a source that places the buffers in memory it owns (an mmap arena), which is how the lazily-backed reservation technique is expressed without a library type.
Governing spec: `proxima-windows/proxima-tensor/specs/pipeline-as-data/SPEC.md` (rows for seal and tier, and place), `sketches/08-rectified-sparse-attention.md` gaps for the seal storage and the rewind commit, `sketches/10-tiered-chunk-cache.md` (block size is one fact: `prompt_cache.block_tokens`), `sketches/03-moe-offload.md` (the placement callback is already a sink; the slot is data, not a new type).

Binding (decisions in core): the seal rule is a pure tier-1 function in proxima-core (FT4.0 adds `proxima_core::kv_decision::seal_target`, FT4.20 adds `proxima_core::kv_decision::sealed_blocks`); interop cards call it and hold only data and IO. Slice 17's tests and slices 5, 6, 10 and 15 name these two functions.

## old to new id map

Sibling files still cite the old ids; this table is the key.

| old card | new card | verdict |
|---|---|---|
| FT4.0 | FT4.0, FT4.20 | split by the one-public-item rule: FT4.0 adds `seal_target`, FT4.20 adds `sealed_blocks`; anchor fix: `ring` is a directory module. A sibling file that cites FT4.0 for `sealed_blocks` reads FT4.20 |
| FT4.1 | FT4.1, FT4.21, FT4.22 | keep, premise fixed, split by the one-public-item rule: four callers of `truncate` exist on main, not two; FT4.1 adds `sealed_end` (lowered by the infallible `truncate` the three whole-state callers keep), FT4.21 adds the `RewindIntoSealed` error, FT4.22 adds `try_truncate` for the one in-flight decode rewind. A sibling file that cites FT4.1 for `try_truncate` or `RewindIntoSealed` reads FT4.22 or FT4.21 |
| none | FT4.2 | new: the seal horizon is a config field beside the one block size (the old cards read it from a slice 2 section) |
| FT4.2 | FT4.3, FT4.23, FT4.16 | keep, split by the one-public-item and file-count rules: FT4.3 adds `seal_attention_layers` (the seal reads the layer's own row count; device-resident layers hold no host rows) with unit tests; FT4.23 is the decode-level test on the synthetic gemma4 checkpoint (a sibling file that cites FT4.3 for that test reads FT4.23); FT4.16 extracts `LayerCache::seal`, which is the item a sibling file that cites FT4.3 for `seal` means |
| FT4.3 | FT4.17, FT4.18, FT4.4, FT4.19, FT4.5, FT4.6 | recut: FT4.17 re-seal when the block size changes; FT4.18 the per-block record field counted in held memory; FT4.4 `BlockSummarizer` with the fill inside `seal_attention_layers`; FT4.19 the `LayerCache::summarize_sealed` method (what a sibling file that cites FT4.4 for `summarize_sealed` means); FT4.5 the carrier that hands the summarizer to decode; FT4.6 the proof that key min and max runs through the slot |
| FT4.4 | FT4.7 | keep, asserts re-pointed at the summary slot |
| FT4.5 | FT4.8 | keep, property restated (see drift item 8) |
| FT4.10 | FT4.9 to FT4.11 (hook), FT4.12 to FT4.14 (proof) | recut: the library `KvReservation` is gone; the mmap reservation is the arena in the proof tests |
| FT4.11 | FT4.9 | keep, null and zero-length refusals added |
| FT4.13 | FT4.10, FT4.11 | recut: the buffer-source parameter on device kv (FT4.10) and the carrier on the model (FT4.11); `Allocation::{Grow, Reserve}`, host `reserve_rows`, `mapped_pages` and `row_copies` are not built |
| FT4.15 | FT4.12, FT4.13, FT4.14 | recut: one model-loading run per card (gemma4 E2B, gemma4 26B, granite); the qwen2, qwen3 and openchat arms are removed |
| FT4.6, FT4.7, FT4.8, FT4.9, FT4.12, FT4.14 (old) | none | ids were never used |

## dropped (content removed from old cards; no whole card is dropped)

- Old FT4.10 `proxima-model-interop/src/kv_reserve.rs` (`KvReservation`, `reserve`, `commit_rows`, `mapped_groups`), the `ReserveFailed` and `ReserveMisaligned` errors and the `"dep:libc"` edit to the `std` feature: that is a paged-kv technique built as public library API with no hook gap naming it. The same mmap reservation appears as about 20 test-side lines in FT4.12.
- Old FT4.13 `Allocation::{Grow, Reserve { page_blocks }}` selection, `DeviceKvLayer.reservations`, `mapped_pages`, `row_copies` and their telemetry event, host `LayerCache::reserve_rows`, and the skipped tail zeroing in `DeviceKvLayer::seed`: the placement hook is a buffer source; zero filling stays exactly as it is because a source returns memory in unspecified state, like today's allocator.
- Old FT4.3 `block_key_min` and `block_key_max` as library fields and `summarize_blocks` computing min and max: replaced by a generic record slot and a summarizer the caller supplies (FT4.4); key min and max is test code in FT4.6.
- Old FT4.15 `kv_reserve_counters_4096` (mapped-page and row-copy counters) and every qwen arm: no counter exists any more, and qwen is barred.
- The slice 2 dependency of the old FT4.2, FT4.3 and FT4.13 (`serving.kv.*` fields): not needed. Choosing a summarizer by name from a list in configuration needs the list-capable serving config that slice 2 owns (`ServingConfig` is `Copy`, `serving.rs` ~line 720); until then the slot is reached from code through FT4.5.

## the two questions for every new item, and the designs abandoned

- Can it be a pipe? The seal decision and the sealed-block range are pure integer functions called synchronously inside the decode step; a pipe would add an async boundary around a division (sketch 08 section 6 reaches the same answer for the summary). The summarizer and the buffer source are `fn` pointers a caller passes, and a configured-length chain of them cannot be a static `and_then` chain; the closed set of today's behaviour is the default value of each slot.
- What can a caller do that they could not before? Rewind a decode into rows a seal protects: refused with a typed error (FT4.22). Keep a per-block record next to sealed rows and have it counted in held memory (FT4.4, FT4.5). Place the device-resident kv in memory the caller owns (FT4.9 to FT4.11). The production call site of the buffer source is the same line as today's allocator by design, because the default reproduces today; the capability is the replacement.
- Abandoned: a public `KvReservation` module plus an `Allocation` config enum (a technique built in the library, and a list-valued stage in a `Copy` config); `block_key_min` and `block_key_max` as library fields with min and max computed in the library; a fallible rewind for every caller (it would stop the prompt cache from reusing an entry it diverges from); a `fn` pointer field in `ServingConfig` (a cache key must be a proved name, never an address); a provisional trie insert at seal (no production caller).

## prerequisites outside this slice

- The granite cards of slice 0 (`tasks-recut/00-oracles-and-worked-examples.md`, FT0.31 to FT0.43): FT0.31 adds `const GRANITE_MOE: Checkpoint` (name `"granite_moe"`, `architecture: "granitemoe"`) in `proxima-model-interop/tests/arch_data_baseline.rs` and `tests/fixtures/llama-parity/granite_moe/gguf_kv.txt` (`granitemoe.block_count = 24`); FT0.41 vendors `tests/fixtures/llama-parity/granite_moe/llama_ids.json` (3 records, ids recorded once from the incumbent, never re-queried); FT0.40 loads granitemoe by family profile plus descriptor; FT0.43 adds `llama_parity_granite_moe`. Only FT4.14 needs them, through FT0.43 (which needs FT0.39, FT0.41 and FT0.42; FT0.42 needs FT0.40, which needs FT0.31 and FT0.37). At a7c08c4c `git grep -n -i granite main -- proxima-model-interop` prints 0 lines, so these are outputs of those cards, not anchors: FT4.14 cites only anchors that exist on main and names the granite pieces as dependencies.
- Nothing from slice 1 or slice 2. `proxima-core` is reachable from `proxima-model-interop` under the `std` feature already (`interop-bgpool = ["dep:prime", "dep:proxima-core"]` in `proxima-model-interop/Cargo.toml`, ~line 58).

Test models (owner, 2026-10-04): gemma4 for dense or MoE, granite for MoE. No qwen of any kind appears in any card of this file. Local checkpoints: gemma4 E2B (ollama `gemma4:e2b-it-qat`, dense, `tests/arch_data_baseline.rs::GEMMA4_E2B`), gemma4 26B (ollama `batiai/gemma4-26b:latest`, MoE, `GEMMA4_26B`), granite (ollama `granite3.1-moe:1b`, MoE, 1.4 GB; `ollama show --modelfile granite3.1-moe:1b` names its blob on the `FROM` line, `sha256-cd60b3e8...`). Oracles run once and are recorded (`llama_ids.json` is the recording); no card re-queries Ollama or llama.cpp. The no-model tests use the synthetic gemma4 checkpoint that `proxima-model-interop/src/generate/chunked_prefill_tests.rs::gemma4_checkpoint` builds (three layers: two sliding with an 8-row window, one full attention).

Test-name rule: no new test in this file contains `sansio_`, `tier_round_trip_`, `serving_state_` or `action_speculation_` (other slices count those prefixes).

## shared fixture for FT4.3 to FT4.8 (the seal trace values)

Test-local helper `fn row_at(position: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>)` returns `even = [p, p + 0.5]`, `odd = [-p, -p - 0.5]`, `value = [10 p]` with `p = position as f32`. Widths: `even_odd_row = 2`, `v_row = 1`. Seal arguments: `block_tokens = 4`, `horizon_rows = 1`.

Sealed-end rule (one formula; it lives in `proxima_core::kv_decision::seal_target` and `sealed_blocks`, FT4.0 and FT4.20, and every interop card calls it): `target = ((cached_rows.saturating_sub(horizon_rows)) / block_tokens) * block_tokens`. A block is sealed iff its last row is at least `horizon_rows` behind the newest row at the moment a seal call runs; a call never lowers the sealed end.

The worked trace (10 appends, 3 rewinds). `rows` is the committed row count after the op, `sealed_end` after calling `seal(2, 4, 1)`:

| op | rows | sealed_end | result |
|---|---|---|---|
| append pos 0 | 1 | 0 | ok |
| append pos 1 | 2 | 0 | ok |
| append pos 2 | 3 | 0 | ok |
| append pos 3 | 4 | 0 | ok (block full, age 0 < 1) |
| append pos 4 | 5 | 4 | block 0 sealed (target `(4/4)*4`) |
| rewind to 4 | 4 | 4 | ok (4 == sealed_end) |
| append pos 4 | 5 | 4 | ok |
| append pos 5 | 6 | 4 | ok |
| append pos 6 | 7 | 4 | ok |
| append pos 7 | 8 | 4 | ok (target `(7/4)*4 = 4`) |
| append pos 8 | 9 | 8 | block 1 sealed (target `(8/4)*4`) |
| rewind to 8 | 8 | 8 | ok (8 == sealed_end) |
| rewind to 7 | 8 | 8 | Err `RewindIntoSealed { keep_positions: 7, sealed_end: 8 }`, rows still 8 |

That is 10 appends (rows 1-5, then 5-9) and 3 rewinds (to 4, to 8, to 7). The test summarizer (FT4.6 `key_minmax`) returns the per-dimension minimum then the per-dimension maximum over the block's rows of the concatenated key `[even row, odd row]`, so each record has 8 values. Block 0 (positions 0-3) is `[0, 0.5, -3, -3.5, 3, 3.5, 0, -0.5]`; block 1 (positions 4-7) is `[4, 4.5, -7, -7.5, 7, 7.5, -4, -4.5]`.

## cards

### 4.0 The seal target as a pure proxima-core function

- id: FT4.0
- needs: none
- budget: 20 min
- crate(s): proxima-core (features: none; builds bare `--no-default-features`, `alloc`, and `std`)
- read first:
  - `proxima-core/src/lib.rs (~line 44 at e4cf9beb)`: where `pub mod ring;` and the other ungated tier-1 modules are declared (`ring` is the directory `proxima-core/src/ring/mod.rs`, not a file);
  - `proxima-core/src/per_core.rs (tests at ~line 116)`: a pure tier-1 module with an in-file `#[cfg(test)] mod tests`, the layout to copy;
  - the fixture section of this file: the sealed-end rule and the trace table.
- change:
  1. `proxima-core/src/kv_decision.rs` (new), declared `pub mod kv_decision;` in `proxima-core/src/lib.rs` beside `pub mod ring;`, no cfg gate (no allocation, no IO). Slice 5 adds its own decisions to the same file later. One item, `#[must_use]`, with a one-line English doc: `pub const fn seal_target(cached_rows: usize, block_tokens: usize, horizon_rows: usize) -> usize`: `0` when `block_tokens == 0`, else `cached_rows.saturating_sub(horizon_rows) / block_tokens * block_tokens`. This is the sealed end in rows for a layer that holds `cached_rows` rows: a block that is not full is never sealed, and a full block seals only when every row is at least `horizon_rows` old.
     Why pure and in proxima-core: every stage's decision is a tier-1 function; `proxima-model-interop` keeps only the data it applies the decision to (`LayerCache`), so the sealing rule is provable with no IO.
- test: in-file `#[cfg(test)] mod tests` (every name carries `kv_decision_`, none carries `sansio_`):
  - `kv_decision_seal_target_worked`: with `block_tokens = 4`, `horizon_rows = 1`, rows 0..=9 give targets `[0, 0, 0, 0, 0, 4, 4, 4, 4, 8]`;
  - `kv_decision_seal_target_edges`: `seal_target(3, 4, 0) == 0`; `seal_target(4, 4, 0) == 4`; `seal_target(8, 4, 8) == 0`; `seal_target(100, 0, 1) == 0`; `seal_target(usize::MAX, 4, 0) == usize::MAX - (usize::MAX % 4)`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_0 cargo nextest run -p proxima-core -E 'test(/kv_decision_/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-core --all-targets`; `cargo check -p proxima-core --no-default-features --features alloc`; `cargo check -p proxima-core --no-default-features`
- stage: `proxima-core/src/kv_decision.rs`, `proxima-core/src/lib.rs`
- commit: `feat(core): add pure seal target for kv blocks`
- done when: the expect line printed, clippy clean, the two cargo checks pass, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: add an eviction, tier or read decision (other slices); add the sealed-block range (the next card); add a serde or config derive; touch `proxima-model-interop`.
- gpu: none

### 4.20 The newly sealed block range as a pure proxima-core function

- id: FT4.20
- needs: FT4.0
- budget: 20 min
- crate(s): proxima-core (features: none; builds bare `--no-default-features`, `alloc`, and `std`)
- read first:
  - `proxima-core/src/kv_decision.rs::seal_target` and its `tests` module (FT4.0): the function this card calls and the test layout to extend;
  - the fixture section of this file: the sealed-end rule and the trace table.
- change:
  1. `proxima-core/src/kv_decision.rs`: add one item, `#[must_use]`, with a one-line English doc: `pub const fn sealed_blocks(previous_sealed_end: usize, cached_rows: usize, block_tokens: usize, horizon_rows: usize) -> core::ops::Range<usize>`: `0..0` when `block_tokens == 0`; otherwise `start = previous_sealed_end / block_tokens`, `end = seal_target(cached_rows, block_tokens, horizon_rows) / block_tokens`, and `start..end` when `end > start`, else the empty `start..start`. These are the block indexes a seal call newly seals; the sealed end never decreases.
- test: in the same `tests` module: `kv_decision_sealed_blocks_is_monotone`: `sealed_blocks(4, 5, 4, 1) == 1..1`, `sealed_blocks(4, 9, 4, 1) == 1..2`, `sealed_blocks(8, 5, 4, 1) == 2..2` (rows rewound below the sealed end: nothing new, the end does not move), `sealed_blocks(0, 9, 4, 1) == 0..2`, `sealed_blocks(0, 9, 0, 1) == 0..0`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_20 cargo nextest run -p proxima-core -E 'test(/kv_decision_/)'`
- expect: `3 passed` (the two tests of the seal target card plus the new one; confirm the names with `cargo nextest list` first)
- also green: `cargo clippy -p proxima-core --all-targets`; `cargo check -p proxima-core --no-default-features --features alloc`; `cargo check -p proxima-core --no-default-features`
- stage: `proxima-core/src/kv_decision.rs`
- commit: `feat(core): add pure newly sealed block range for kv blocks`
- done when: the expect line printed, clippy clean, the two cargo checks pass, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: change `seal_target`; add an eviction, tier or read decision (other slices); touch `proxima-model-interop`.
- gpu: none

### 4.1 Track the sealed end on a layer cache

- id: FT4.1
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::LayerCache (~line 112)`, `::truncate (~line 155)`, `layer_cache_truncate_tests (~line 167)`: the struct this card gives a sealed end, the rewind that lowers it, and the three tests that must keep passing;
  - the whole-state callers of `truncate`: `proxima-model-interop/src/generate/chunk_shift.rs::PrefixState::append_moved (~line 317)`, `proxima-model-interop/src/generate/prompt_cache.rs::PrefixState::rewind_to (~line 322)`, `proxima-model-interop/src/generate/ring_checkpoint.rs::PrefixState::restore_checkpoint (~line 129)`: they discard rows they rebuild, so they keep the infallible `truncate`.
- change:
  1. `proxima-model-interop/src/generate/residency_caches.rs`: `LayerCache` gains `pub(super) sealed_end: usize` (doc: rows below this are sealed; a ring layer never seals; `0` until a seal call raises it), `0` in `new()`. In `truncate`, after the ring early return, add `self.sealed_end = self.sealed_end.min(keep_positions);` and extend its doc: the whole-state rewinds (prompt cache lookup, ring checkpoint restore, chunk shift) discard rows they rebuild, so they lower the sealed end with the rows. Add the test module `layer_cache_sealing_tests` (`#[cfg(test)]`, `#[allow(clippy::unwrap_used, clippy::expect_used)]` like its neighbour) with the helpers `row_at(position)` and `cache_with_rows(rows) -> LayerCache` from the fixture section (`cache_with_rows` appends `row_at(0..rows)` one row at a time).
- test: in `layer_cache_sealing_tests`: `a_whole_state_rewind_lowers_the_sealed_end_with_the_rows`: `cache_with_rows(6)`, set `sealed_end = 4` directly; `truncate(2, 2, 1)` leaves `k_even.len() == 4` and `sealed_end == 2`; a following `truncate(5, 2, 1)` changes neither. The three `layer_cache_truncate_tests` keep passing unchanged.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_1 cargo nextest run -p proxima-model-interop --features std -E 'test(/a_whole_state_rewind_lowers_the_sealed_end_with_the_rows|layer_cache_truncate_tests/)'`
- expect: `4 passed` (1 new plus the 3 `layer_cache_truncate_tests`; confirm the module's count of 3 with `cargo nextest list` first)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`
- commit: `feat(interop): track the sealed end on a layer cache`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: set `sealed_end` outside tests (the seal card does); edit `chunk_shift.rs`, `prompt_cache.rs`, `ring_checkpoint.rs`, `decode.rs` or `error.rs`.
- gpu: none

### 4.21 Add the error for an in-flight rewind into sealed rows

- id: FT4.21
- needs: FT4.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/error.rs::InteropError (~line 13)`: variant style, `UnknownArchitecture` and its `#[error(..)]` attribute;
  - `proxima-model-interop/src/generate/residency_caches.rs::layer_cache_sealing_tests` (FT4.1): the module the message test joins.
- change:
  1. `proxima-model-interop/src/error.rs`: add the variant `RewindIntoSealed { keep_positions: usize, sealed_end: usize }` with `#[error("cannot rewind to {keep_positions} rows: rows below {sealed_end} are sealed")]` and a one-line doc ("an in-flight rewind reached rows a seal made immutable"), beside `UnknownArchitecture`. Not feature-gated. It is a public variant of a public error enum, so it is reachable and needs no production caller in this card; the next card constructs it.
- test: in `layer_cache_sealing_tests`: `rewind_error_names_both_lengths`: the message of `InteropError::RewindIntoSealed { keep_positions: 7, sealed_end: 8 }` (`to_string()`) equals `"cannot rewind to 7 rows: rows below 8 are sealed"`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_21 cargo nextest run -p proxima-model-interop --features std -E 'test(/rewind_error_names_both_lengths/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/error.rs`, `proxima-model-interop/src/generate/residency_caches.rs`
- commit: `feat(interop): add the error for a rewind into sealed rows`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: construct the variant outside the test (the next card does); touch `decode.rs`.
- gpu: none

### 4.22 Refuse in-flight kv rewinds that cut into sealed rows

- id: FT4.22
- needs: FT4.21
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::LayerCache::truncate (~line 155)`, `::sealed_end` (FT4.1) and `layer_cache_sealing_tests` (FT4.1, FT4.21): the infallible rewind this card wraps and the module the test joins;
  - `proxima-model-interop/src/generate/decode.rs (~line 5818)`: the one in-flight caller, inside the speculative verify commit; the closure returns `Result<u32, InteropError>` (its `return Err(InteropError::..)` at ~line 5748 shows it);
  - `proxima-model-interop/src/error.rs::InteropError::RewindIntoSealed` (FT4.21).
- change:
  1. `proxima-model-interop/src/generate/residency_caches.rs`: add `pub(super) fn try_truncate(&mut self, keep_positions: usize, even_odd_row: usize, v_row: usize) -> Result<(), InteropError>`: returns `Err(InteropError::RewindIntoSealed { keep_positions, sealed_end: self.sealed_end })` with nothing changed when `keep_positions < self.sealed_end`; otherwise calls `self.truncate(..)` and returns `Ok(())`. One-line doc: the in-flight rewind; the whole-state rewinds keep the infallible `truncate`.
  2. `proxima-model-interop/src/generate/decode.rs`: the call at ~line 5818 becomes `cache.try_truncate(keep_positions, *even_odd_row, *v_row)?;`. The three whole-state callers are not edited.
- test: in `layer_cache_sealing_tests`: `rewind_into_sealed_rows_is_refused`: `cache_with_rows(6)`, set `sealed_end = 4` directly; `try_truncate(3, 2, 1)` matches `Err(InteropError::RewindIntoSealed { keep_positions: 3, sealed_end: 4 })` and `k_even.len() == 12`; `try_truncate(4, 2, 1)` is `Ok(())` with `k_even.len() == 8` and `v.len() == 4`; `try_truncate(9, 2, 1)` is `Ok(())` with `k_even.len() == 8`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_22 cargo nextest run -p proxima-model-interop --features std -E 'test(/layer_cache_sealing_tests|layer_cache_truncate_tests/)'`; then the replay guard on the speculative commit path: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_22 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_22 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^llama_parity_gemma4_e2b$/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_22/run.log`
- expect: `6 passed` (3 in `layer_cache_sealing_tests`: the FT4.1 test, the FT4.21 test and the new one; 3 in `layer_cache_truncate_tests`; confirm the 6 names with `cargo nextest list` first); the replay guard `1 passed` (it replays the vendored llama ids for gemma4 E2B)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --features std,metal`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`, `proxima-model-interop/src/generate/decode.rs`
- commit: `feat(interop): refuse in-flight kv rewinds below the sealed end`
- done when: both expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: set `sealed_end` outside tests (the seal card does); edit `chunk_shift.rs`, `prompt_cache.rs` or `ring_checkpoint.rs`; change speculative-decode semantics.
- gpu: one run (the replay guard), waiting for a quiet box (the peer-gate check in SPEC "machine safety")

### 4.2 Add the seal horizon to the prompt cache config

- id: FT4.2
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/serving.rs::PromptCacheConfig (~line 472)`, its `block_tokens` field (~line 550), `::standard (~line 564)` and `::off (~line 587)`: the one block size every tier shares, and the two constructors;
  - `proxima-model-interop/src/prompt_cache_settings.rs::PromptCacheSettings (~line 17)`, `::as_prompt_cache_config`, and the tests `prompt_cache_builder_matches_toml_and_env_loaders (~line 139)`, `default_settings_lower_to_the_standard_config (~line 212)`, `a_zero_byte_budget_from_the_environment_turns_the_cache_off (~line 229)`: the settings mirror and its parity tests.
- change:
  1. `proxima-model-interop/src/serving.rs`: `PromptCacheConfig` gains `pub seal_horizon_rows: u32` after `block_tokens`, doc (semantic only): "How many rows behind the newest row a full block's last row must be before the block is sealed. A sealed block is never rewound by an in-flight decode, so this is the deepest rewind a decode may make. `0` seals a block as soon as it is full." `standard()` sets `256`; `off()` inherits it through `..Self::standard()`. `ServingConfig::prompt_cache` is already `_` in the prompt-cache key, and sealing changes no row, so the key is unchanged.
  2. `proxima-model-interop/src/prompt_cache_settings.rs`: `PromptCacheSettings` gains `#[setting(default = 256)] #[builder(default = 256)] pub seal_horizon_rows: u32` (doc: "See [`crate::PromptCacheConfig::seal_horizon_rows`]."), and `as_prompt_cache_config` copies it. In the tests: `PROMPT_CACHE_ENV_KEYS` grows to `[&str; 15]` with `"PROXIMA_PROMPT_CACHE_SEAL_HORIZON_ROWS"` last; `prompt_cache_builder_matches_toml_and_env_loaders` gains `.seal_horizon_rows(512)` on the builder, `\nseal_horizon_rows = 512` in the TOML text, the env arm `"PROXIMA_PROMPT_CACHE_SEAL_HORIZON_ROWS" => Some("512")` (without it the wildcard arm would set `700`), and `assert_eq!(lowered.seal_horizon_rows, 512)`.
- test: add `seal_horizon_defaults_to_256_rows` in the `prompt_cache_settings.rs` tests, asserting `PromptCacheConfig::standard().seal_horizon_rows == 256`, `PromptCacheConfig::off().seal_horizon_rows == 256`, `PromptCacheSettings::builder().build().seal_horizon_rows == 256`, and `PromptCacheSettings::builder().build().as_prompt_cache_config() == PromptCacheConfig::standard()`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_2 cargo nextest run -p proxima-model-interop --features std -E 'test(/seal_horizon_defaults_to_256_rows|prompt_cache_builder_matches_toml_and_env_loaders|default_settings_lower_to_the_standard_config|a_zero_byte_budget_from_the_environment_turns_the_cache_off/)'`
- expect: `4 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --features std,metal --examples`
- stage: `proxima-model-interop/src/serving.rs`, `proxima-model-interop/src/prompt_cache_settings.rs`
- commit: `feat(interop): add the seal horizon to the prompt cache config`
- done when: the expect line printed, clippy and the examples check clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: add a `kv` section or a second block size (the block size is `block_tokens`); edit `generate/prompt_cache_key.rs`; read the field anywhere (the seal card does).
- gpu: none

### 4.3 Seal full layer blocks as decode commits rows

- id: FT4.3
- needs: FT4.20, FT4.22, FT4.2
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first (anchors re-read at a7c08c4c):
  - `proxima-model-interop/src/generate/residency_caches.rs::LayerCache`, `layer_cache_sealing_tests` (FT4.1), `LayerCacheState (~line 644)` and `LayerPadRowWidths (~line 956)`;
  - `proxima-core/src/kv_decision.rs::sealed_blocks` (FT4.20);
  - `proxima-model-interop/src/generate/decode.rs::LoadedModel::run_decode_loop_from_ids (~line 2988 at a7c08c4c)`: the step closure passed to `decode_until_stop_or_budget (~line 3596)`: the non-speculative commit `cached_len += new_count;` (~line 5705, inside `if !active_layer_roots.is_empty() && !speculative_step`) and the speculative commit `cached_len = keep_positions;` (~line 5821); `layer_caches` (~line 3151) and `layer_row_widths` (~line 3118, bound by `declared_layer_cache_names_and_widths`) are in scope for both. `run_decode_loop_through_cache` is in `prompt_cache.rs` and is not edited.
- change:
  1. `proxima-model-interop/src/generate/residency_caches.rs`: add one item, `pub(super) fn seal_attention_layers(layer_caches: &mut [LayerCacheState], widths: &[LayerPadRowWidths], block_tokens: usize, horizon_rows: usize)`. It zips the two slices and acts only on a `(LayerCacheState::Attention(cache), LayerPadRowWidths::Attention { even_odd_row, .. })` pair; every other pairing is skipped. A pair is also skipped for a ring layer (`cache.ring_geometry().is_some()`; one lowercase comment line: ring slots are overwritten in place, so a ring row is never immutable; rings restore through checkpoints) and when `*even_odd_row == 0`. Otherwise: `let rows = cache.k_even.len() / *even_odd_row; let blocks = proxima_core::kv_decision::sealed_blocks(cache.sealed_end, rows, block_tokens, horizon_rows); cache.sealed_end = cache.sealed_end.max(blocks.end * block_tokens);`. The row count comes from the layer's own planes because a device-resident layer holds no host rows during decode (`DeviceKv::adopt` empties them), so it seals nothing. No seal rule of its own: the formula lives in `proxima-core`.
  2. `proxima-model-interop/src/generate/decode.rs`: inside `run_decode_loop_from_ids` (`run_decode_loop_observed_seeded` only tokenizes and forwards to it through `run_decode_loop_through_cache`, and is not edited), before `decode_until_stop_or_budget` bind `let block_tokens = serving_config.prompt_cache.block_tokens as usize;` and `let seal_horizon_rows = serving_config.prompt_cache.seal_horizon_rows as usize;`. Call `seal_attention_layers(&mut layer_caches, &layer_row_widths, block_tokens, seal_horizon_rows);` once right after `cached_len += new_count;` (inside the same `if`) and once right after `cached_len = keep_positions;`. Allocation budget: none (one division per attention layer per step). `run_decode_loop_placed_kv (~line 6327)` owns its own device buffers and holds no host rows; it is not edited.
- test: in `layer_cache_sealing_tests`, with two test-local helpers: `fn widths_for(layers: &[LayerCacheState]) -> Vec<LayerPadRowWidths>` (an `Attention { even_odd_row: 2, v_row: 1 }` for every `LayerCacheState::Attention`, `LayerPadRowWidths::SharedFromLayer` for every other state) and `fn attention(layers: &mut [LayerCacheState], index: usize) -> &mut LayerCache` (the cache inside an `Attention` state, `expect` otherwise); `seal_attention_layers(&mut layers, &widths_for(&layers), 4, 1)` below is written `seal(layers, 4, 1)`:
    - `seal_full_block_inside_horizon_not_sealed`: one layer `Attention(cache_with_rows(4))`, `seal(layers, 4, 1)` leaves `sealed_end == 0`;
    - `seal_partial_block_never_sealed`: one layer `Attention(cache_with_rows(3))`, `seal(layers, 4, 0)` leaves `sealed_end == 0`;
    - `seal_bytes_unchanged_after_later_appends`: one layer `Attention(cache_with_rows(5))`, `seal(layers, 4, 1)` leaves `sealed_end == 4`; clone `k_even[..8]`, `k_odd[..8]`, `v[..4]`; append rows for positions 5..10 through `attention(..)`; `seal(layers, 4, 1)` leaves `sealed_end == 8`; the first block's bytes equal the clones (compare with `to_bits`);
    - `ring_layer_never_reports_sealed_rows`: one layer `Attention(LayerCache::ring(KvRing::new(4, 0, 2, 1, 0), 8))`, write 8 rows through `append_at(position, ..)` from `row_at`, `seal(layers, 4, 0)` leaves `sealed_end == 0`;
    - `decode_commit_seals_attention_layers_only`: `seal(layers, 4, 0)` over `[Attention(cache_with_rows(8)), Attention(ring of 8 rows as above), SharedFromLayer]` leaves the first layer at `sealed_end == 8` and the ring layer at `0`.
  The decode-level proof on a real checkpoint is the next card.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_3 cargo nextest run -p proxima-model-interop --features std -E 'test(/seal_full_block_inside_horizon_not_sealed|seal_partial_block_never_sealed|seal_bytes_unchanged_after_later_appends|ring_layer_never_reports_sealed_rows|decode_commit_seals_attention_layers_only/)'`; then the replay guard: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_3 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_3 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^llama_parity_gemma4_e2b$/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_3/run.log`
- expect: `5 passed`; the replay guard `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`, `proxima-model-interop/src/generate/decode.rs`
- commit: `feat(interop): seal full layer blocks as decode commits rows`
- done when: both expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: write the seal formula in interop (call `kv_decision`); add a per-layer `seal` method (a later card extracts it); add summaries; edit `run_decode_loop_placed_kv` or `run_decode_loop_through_cache`; add `kv_seal_tests.rs` (the next card).
- gpu: one run (the replay guard), waiting for a quiet box (the peer-gate check in SPEC "machine safety")

### 4.23 Prove decode seals the full attention layer of a gemma4 checkpoint

- id: FT4.23
- needs: FT4.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first (anchors re-read at a7c08c4c):
  - `proxima-model-interop/src/generate/chunked_prefill_tests.rs::gemma4_checkpoint (~line 95)`, `config (~line 187)`, `Prefilled (~line 200)`, `prefill (~line 206)`, `prompt_of (~line 316)`: the synthetic gemma4 fixture (three layers: two sliding with an 8-row window, one full attention) and its decode helper, all private today;
  - `proxima-model-interop/src/generate/chunked_prefill_tests.rs::chunked_prefill_matches_one_evaluation_on_two_range_gemma4`: how the checkpoint is loaded (`gemma4_checkpoint()`, `proxima_gguf::pipe::parse_complete`, `LoadedModel::load`);
  - `proxima-model-interop/src/generate/mod.rs::chunked_prefill_tests (~line 231)`: the `#[cfg(test)] mod` declaration this card adds a neighbour to.
- change:
  1. `proxima-model-interop/src/generate/chunked_prefill_tests.rs`: make `gemma4_checkpoint`, `config`, `prompt_of`, `Prefilled` and its `state` field `pub(super)`; split `prefill(model, prompt, ubatch_size)` into `pub(super) fn prefill_with(model, prompt, serving_config: &ServingConfig<'_>) -> Prefilled` (the current body) plus a `prefill` that calls it with `&config(ubatch_size)`. The three existing assertions and tests are not edited.
  2. `proxima-model-interop/src/generate/mod.rs`: add `#[cfg(test)] mod kv_seal_tests;` after the `chunked_prefill_tests` declaration.
  3. `proxima-model-interop/src/generate/kv_seal_tests.rs` (new): the test below. Its first lines are the anchor's file-level `#![allow(clippy::expect_used)]` with its one-line reason comment (the workspace denies `expect_used` and `unwrap_used`, and an inner attribute is not inherited by a new module file); the tests in it use `expect` with a message, never `unwrap`.
- test: add `decode_seals_full_attention_layers_at_commit` in `kv_seal_tests.rs`: load the synthetic checkpoint as the anchor above does, run `prefill_with` on `prompt_of(40, '3')` under `ServingConfig { prompt_cache: PromptCacheConfig { block_tokens: 8, seal_horizon_rows: 4, ..PromptCacheConfig::standard() }, ..config(7) }`. Measured on this fixture (a throwaway probe printing the prefill result, at a7c08c4c): `state.len() == 40` and the six prefill commits are `[7, 7, 7, 7, 7, 5]` rows. Assert `state.len() == 40`; the layer caches hold exactly 1 full-attention layer (`ring_geometry().is_none()`) and 2 ring layers; the full layer has `sealed_end == 32` (hand value: `(40 - 4) / 8 * 8`); both ring layers have `sealed_end == 0`. If `state.len()` is not 40, stop and report the printed length.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_23 cargo nextest run -p proxima-model-interop --features std -E 'test(/decode_seals_full_attention_layers_at_commit/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/generate/chunked_prefill_tests.rs`, `proxima-model-interop/src/generate/mod.rs`, `proxima-model-interop/src/generate/kv_seal_tests.rs`
- commit: `test(interop): prove decode seals full layers on a gemma4 checkpoint`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: change non-test code; edit the three assertions of the existing chunked-prefill tests; add a summarizer.
- gpu: none

### 4.16 Expose the per-layer seal as a method that reports the new blocks

- id: FT4.16
- needs: FT4.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::seal_attention_layers` (FT4.3): the loop body this card moves into a method;
  - `proxima-model-interop/src/generate/residency_caches.rs::layer_cache_sealing_tests` (`row_at`, `cache_with_rows`, FT4.1).
- change:
  1. `proxima-model-interop/src/generate/residency_caches.rs`: add one item, `pub(super) fn seal(&mut self, even_odd_row: usize, block_tokens: usize, horizon_rows: usize) -> core::ops::Range<usize>` on `LayerCache`, holding the per-layer body of `seal_attention_layers` unchanged: it returns `0..0` for a ring layer (keep the one-line comment) and when `even_odd_row == 0`; otherwise it computes `rows`, calls `proxima_core::kv_decision::sealed_blocks`, raises `self.sealed_end` and returns the blocks range. `seal_attention_layers` now calls `cache.seal(*even_odd_row, block_tokens, horizon_rows);` for each attention pair and discards the range; its own checks for a ring layer and a zero width are removed (the method holds them). Behaviour of every FT4.3 test is unchanged.
- test: in `layer_cache_sealing_tests`:
  - `seal_returns_the_newly_sealed_block_range`: `cache_with_rows(5)`, `seal(2, 4, 1)` returns `0..1` and `sealed_end == 4`; append rows for positions 5..10; `seal(2, 4, 1)` returns `1..2` and `sealed_end == 8`; a second `seal(2, 4, 1)` returns `2..2`;
  - `seal_returns_an_empty_range_for_a_ring_layer_and_a_partial_block`: the ring cache of the FT4.3 ring test with 8 rows: `seal(2, 4, 0)` returns `0..0`; `cache_with_rows(3)`: `seal(2, 4, 0)` returns `0..0`; `cache_with_rows(8)`: `seal(0, 4, 0)` returns `0..0`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_16 cargo nextest run -p proxima-model-interop --features std -E 'test(/layer_cache_sealing_tests/)'`
- expect: `10 passed` (3 from the rewind cards, 5 from the seal card, 2 new; confirm the 10 names with `cargo nextest list` before the run)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`
- commit: `refactor(interop): expose the per-layer seal as a method`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: edit `decode.rs`; change what is sealed; add summaries.
- gpu: none

### 4.17 Re-seal from scratch when the block size changes

- id: FT4.17
- needs: FT4.16
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::LayerCache`, `::seal` (FT4.16), `::new`: the struct, the method this card changes, and the constructor that gets the new field;
  - `proxima-model-interop/src/serving.rs::PromptCacheConfig::block_tokens (~line 550)`: the one block size, a runtime configuration value, so a cached layer can meet a different size on a later request.
- change:
  1. `proxima-model-interop/src/generate/residency_caches.rs`: `LayerCache` gains `pub(super) block_tokens: usize` (doc: rows per block the last seal call used; `0` before the first), `0` in `new()`. In `seal`, after the ring and zero-width early return and before the formula: `if self.block_tokens != block_tokens { self.sealed_end = 0; self.block_tokens = block_tokens; }` (a changed block size re-seals from scratch: a sealed end counted in the old blocks indexes other blocks).
- test: in `layer_cache_sealing_tests`: `a_new_block_size_re_seals_from_scratch`: `cache_with_rows(9)`, `seal(2, 4, 1)` returns `0..2` with `sealed_end == 8` and `block_tokens == 4`; `seal(2, 2, 1)` returns `0..4` (without the reset it would return `4..4`), `sealed_end == 8` and `block_tokens == 2`; a repeated `seal(2, 2, 1)` returns `4..4`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_17 cargo nextest run -p proxima-model-interop --features std -E 'test(/layer_cache_sealing_tests/)'`
- expect: `12 passed` (the 11 tests the filter matched on main before this card, including `seal_proptest_never_seals_a_rewindable_row`, plus the new one)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`
- commit: `fix(interop): re-seal a layer from scratch when the block size changes`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: add a summary field (the next card); edit `decode.rs` or `serving.rs`.
- gpu: none

### 4.18 Hold a record per sealed block and count it as held memory

- id: FT4.18
- needs: FT4.17
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::LayerCache`, `::seal`, `::truncate` (FT4.1, FT4.17);
  - `proxima-model-interop/src/generate/prompt_cache.rs::PrefixState::byte_len (~line 332)`: the byte accounting the prompt cache budgets against; the records are held memory and must be counted here (the field's production reader);
  - `proxima-model-interop/src/generate/prompt_cache.rs` tests: `gemma_like_state (~line 1769)` builds a `PrefixState` with one full layer.
- change:
  1. `proxima-model-interop/src/generate/residency_caches.rs`: `LayerCache` gains `pub(super) block_summaries: Vec<Vec<f32>>` (doc: one record per sealed block in block order, written by the caller's summarizer; empty when none ran), empty in `new()`. In `seal`'s block-size reset branch (FT4.17) add `self.block_summaries.clear();`. In `truncate`, after the `min` clamp: `if self.block_tokens > 0 { self.sealed_end -= self.sealed_end % self.block_tokens; self.block_summaries.truncate(self.sealed_end / self.block_tokens); }` (a whole-state rewind keeps only whole sealed blocks and their records).
  2. `proxima-model-interop/src/generate/prompt_cache.rs::PrefixState::byte_len`: in the `LayerCacheState::Attention(cache)` arm add `cache.block_summaries.capacity() * size_of::<Vec<f32>>() + cache.block_summaries.iter().map(|record| record.capacity() * size_of::<f32>()).sum::<usize>()` to the layer's bytes.
- test:
  - in `layer_cache_sealing_tests`:
    - `a_whole_state_rewind_keeps_only_whole_sealed_blocks_and_their_records`: `cache_with_rows(9)`, `seal(2, 4, 1)` (sealed end 8), set `block_summaries = vec![vec![60.0f32], vec![220.0f32]]`; `truncate(5, 2, 1)` leaves `sealed_end == 4` and `block_summaries == [[60.0]]`;
    - `a_new_block_size_drops_the_records`: the same setup; `seal(2, 2, 1)` leaves `block_summaries` empty and `sealed_end == 8`;
  - in the `prompt_cache.rs` tests: `prefix_state_byte_len_counts_block_summaries`: build `baseline = gemma_like_state(8)` and `with_records = gemma_like_state(8)` (two builds of the same helper, so every plane has the same capacity on both sides; `PrefixState` is not `Clone`), set `with_records`'s full layer (layer index 1) `block_summaries = vec![vec![0.0f32; 8], vec![0.0f32; 8]]` (capacity equals length for a `vec!` literal, and the baseline's `block_summaries` is empty with capacity 0); `with_records.byte_len() - baseline.byte_len() == 2 * size_of::<Vec<f32>>() + 2 * 8 * size_of::<f32>()` (`2 * 24 + 2 * 8 * 4 = 112` on a 64-bit target; assert the expression, not the literal).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_18 cargo nextest run -p proxima-model-interop --features std -E 'test(/a_whole_state_rewind_keeps_only_whole_sealed_blocks_and_their_records|a_new_block_size_drops_the_records|prefix_state_byte_len_counts_block_summaries|a_whole_state_rewind_lowers_the_sealed_end_with_the_rows/)'`
- expect: `4 passed` (3 new, plus the FT4.1 whole-state rewind test, which must keep passing because its `block_tokens` is `0`)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`, `proxima-model-interop/src/generate/prompt_cache.rs`
- commit: `feat(interop): hold a record per sealed block and count its bytes`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: compute any record in non-test code (the next card); add a field to `ServingConfig` or the prompt-cache key; edit `decode.rs`.
- gpu: none

### 4.4 Fill the per-block records from a caller-supplied summarizer

- id: FT4.4
- needs: FT4.18
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::seal_attention_layers`, `LayerCache::seal`, `::block_summaries`, `::block_tokens` (FT4.3, FT4.16 to FT4.18);
  - `proxima-model-interop/src/generate/decode.rs`: the two `seal_attention_layers` call sites (FT4.3);
  - `proxima-model-interop/src/generate/residency_caches.rs::layer_cache_sealing_tests` (`row_at`, `cache_with_rows`, `widths_for`, `attention`).
- change:
  1. `proxima-model-interop/src/generate/residency_caches.rs`: add one item, `pub type BlockSummarizer = fn(k_even: &[f32], k_odd: &[f32], value: &[f32], even_odd_row: usize) -> Vec<f32>;` with a teaching doc: it folds one sealed block's rows (the K even plane rows, the K odd plane rows and the V rows, each row-major; K rows are `even_odd_row` wide, V rows are `value.len() / (k_even.len() / even_odd_row)` wide) into the record kept for that block; it is called from [`seal_attention_layers`], which a seal call feeds. `seal_attention_layers` gains `summarize: Option<BlockSummarizer>` as its last parameter. After a pair is sealed, and only when `summarize` is `Some(summarizer)` and `cache.block_tokens > 0` and the layer is not a ring layer: `let sealed_blocks = cache.sealed_end / cache.block_tokens;` truncate `cache.block_summaries` to `sealed_blocks`, then push `summarizer(..)` for each missing block index from `block_summaries.len()` up to `sealed_blocks`, over the slices `k_even[block * block_tokens * even_odd_row..(block + 1) * block_tokens * even_odd_row]` (same for `k_odd`) and `v[block * block_tokens * v_row..(block + 1) * block_tokens * v_row]` (`v_row` from the width pair). It is idempotent and catches up a summarizer that arrived after blocks were sealed. Allocation: one record per sealed block (once per `block_tokens` rows per layer), none on a step that seals nothing.
  2. `proxima-model-interop/src/generate/decode.rs`: both `seal_attention_layers` calls pass `None` as the new argument (no summarizer is carried yet). The FT4.3 tests pass `None` too.
- test: in `layer_cache_sealing_tests`, with the test-local summarizer `fn value_sum(_k_even: &[f32], _k_odd: &[f32], value: &[f32], _even_odd_row: usize) -> Vec<f32> { vec![value.iter().sum()] }` and `seal(layers, block_tokens, horizon_rows, summarize)` meaning `seal_attention_layers(&mut layers, &widths_for(&layers), block_tokens, horizon_rows, summarize)`:
  - `commit_summarizes_sealed_blocks_only_when_a_summarizer_is_given`: one layer `Attention(cache_with_rows(8))` with `seal(layers, 4, 0, Some(value_sum))` gives `block_summaries == [[60.0], [220.0]]` (block 0 values `0 + 10 + 20 + 30`, block 1 `40 + 50 + 60 + 70`); the same call with `None` on a fresh copy leaves it empty while `sealed_end == 8`;
  - `a_summarizer_that_arrives_late_catches_up_and_is_idempotent`: one layer `Attention(cache_with_rows(9))`, `seal(layers, 4, 1, None)` (sealed end 8, no records), then `seal(layers, 4, 1, Some(value_sum))` gives `[[60.0], [220.0]]`; one more identical call leaves the same two records.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_4 cargo nextest run -p proxima-model-interop --features std -E 'test(/commit_summarizes_sealed_blocks_only_when_a_summarizer_is_given|a_summarizer_that_arrives_late_catches_up_and_is_idempotent/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`, `proxima-model-interop/src/generate/decode.rs`
- commit: `feat(interop): fill per-block records from a caller-supplied summarizer`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: compute any summary in non-test code; add a per-layer `summarize_sealed` method (the next card extracts it); add a field to `ServingConfig` or the prompt-cache key.
- gpu: none

### 4.19 Expose the per-layer summary fill as a method

- id: FT4.19
- needs: FT4.4
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::seal_attention_layers`, `BlockSummarizer` (FT4.4): the summary loop this card moves into a method;
  - `proxima-model-interop/src/generate/residency_caches.rs::layer_cache_sealing_tests` (`value_sum`, FT4.4).
- change:
  1. `proxima-model-interop/src/generate/residency_caches.rs`: add one item, `pub(super) fn summarize_sealed(&mut self, even_odd_row: usize, v_row: usize, summarize: BlockSummarizer)` on `LayerCache`, holding the summary loop of `seal_attention_layers` unchanged (returns at once for a ring layer or `block_tokens == 0`). `seal_attention_layers` calls `cache.summarize_sealed(*even_odd_row, *v_row, summarizer)` when `summarize` is `Some`. The doc of `BlockSummarizer` now names [`LayerCache::summarize_sealed`] as its caller. Behaviour of every FT4.4 test is unchanged.
- test: in `layer_cache_sealing_tests`: `summarize_sealed_fills_missing_blocks_and_is_idempotent`: `cache_with_rows(9)`, `seal(2, 4, 1)` (sealed end 8), `summarize_sealed(2, 1, value_sum)` gives `block_summaries == [[60.0], [220.0]]`; a second call leaves the same two records; after `truncate(5, 2, 1)` (one whole block left) a third call leaves `block_summaries == [[60.0]]`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_19 cargo nextest run -p proxima-model-interop --features std -E 'test(/layer_cache_sealing_tests/)'`
- expect: `16 passed` (the 11 of the block-size card, 2 from the record card, 2 from the summarizer card, 1 new; confirm the 16 names with `cargo nextest list` before the run)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`
- commit: `refactor(interop): expose the per-layer summary fill as a method`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: edit `decode.rs`; change what is summarized.
- gpu: none

### 4.5 Let a model carry the sealed block summarizer

- id: FT4.5
- needs: FT4.19, FT4.23
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/load_model.rs::LoadedModel (~line 803)`: the struct and its `ring_write_offset` field (~line 835), the precedent for a model-level value kept off `ServingConfig`;
  - `proxima-model-interop/src/generate/pregather.rs::LoadedModel::with_ring_write_offset_for_parity_control (~line 2489)` and the three `LoadedModel { .. }` literals (~lines 2713, 2844, 2965): the builder precedent and every construction site;
  - `proxima-model-interop/src/generate/tests_all.rs::memory_fit_gate_tests::model_with (~line 2558)`: a fourth struct literal, under `#[cfg(all(feature = "metal", target_os = "macos"))]`;
  - `proxima-model-interop/src/generate/prompt_cache.rs::LoadedModel::clear_prompt_cache (~line 999)` and `::prompt_cache_bytes (~line 1007)`.
- change:
  1. `proxima-model-interop/src/generate/load_model.rs`: `LoadedModel` gains `pub(super) block_summarizer: Option<BlockSummarizer>` with the doc "The fold each sealed block of a full-attention layer is summarized by; `None` keeps no summaries." (import `BlockSummarizer` from `residency_caches`).
  2. `proxima-model-interop/src/generate/pregather.rs`: each of the three literals gains `block_summarizer: None,`. Beside `with_ring_write_offset_for_parity_control` add `#[must_use] pub fn with_block_summarizer(mut self, summarizer: BlockSummarizer) -> Self` that calls `self.clear_prompt_cache();`, sets `self.block_summarizer = Some(summarizer);` and returns `self`. Doc (teaching surface): this is the seal hook's summary slot, [`LayerCache::summarize_sealed`] in `residency_caches.rs`; entries stored before it are dropped because their sealed blocks carry no summaries; a model built without it keeps none. If the compiler reports the private `BlockSummarizer` alias in a public signature, spell the `fn(&[f32], &[f32], &[f32], usize) -> Vec<f32>` type in this builder's parameter.
  3. `proxima-model-interop/src/generate/decode.rs`: bind `let summarizer = self.block_summarizer;` beside the FT4.3 bindings and pass `summarizer` instead of `None` at both `seal_attention_layers` calls.
- test: in `proxima-model-interop/src/generate/tests_all.rs`, the one test file of this card (its `memory_fit_gate_tests::model_with` literal also gains `block_summarizer: None,`). Add a new module at the end of the file, `#[cfg(all(test, feature = "std"))] #[allow(clippy::unwrap_used, clippy::expect_used)] mod block_summarizer_tests`, importing the synthetic checkpoint helpers `crate::generate::chunked_prefill_tests::{config, gemma4_checkpoint, prefill_with, prompt_of}` (made `pub(super)` by FT4.23) and loading the checkpoint as FT4.23's test does:
  - `decode_keeps_one_summary_record_per_sealed_block`: with the test-local `fn block_row_count(k_even: &[f32], _k_odd: &[f32], _value: &[f32], even_odd_row: usize) -> Vec<f32> { vec![(k_even.len() / even_odd_row) as f32] }`, load the synthetic checkpoint, `.with_block_summarizer(block_row_count)`, run `prefill_with` on `prompt_of(40, '3')` under the FT4.23 config (block 8, horizon 4, ubatch 7). Assert `state.len() == 40` (measured, see FT4.23) and the full layer has `sealed_end == 32`; its `block_summaries == vec![vec![8.0f32]; 4]` (four sealed blocks of eight rows, each record the block row count); both ring layers have empty `block_summaries`. Control: the same run on a model without a summarizer has the same `sealed_end` and empty `block_summaries`;
  - `with_block_summarizer_drops_entries_stored_before_it`: load the synthetic checkpoint, `generate_with_serving_config(&prompt_of(40, '3'), 2, config(0))` (prompt cache on by default) and assert `prompt_cache_bytes() > 0`; after `with_block_summarizer(block_row_count)` assert `prompt_cache_bytes() == 0`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_5 cargo nextest run -p proxima-model-interop --features std -E 'test(/decode_keeps_one_summary_record_per_sealed_block|with_block_summarizer_drops_entries_stored_before_it/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` (this compiles the `tests_all.rs` literal)
- stage: `proxima-model-interop/src/generate/load_model.rs`, `proxima-model-interop/src/generate/pregather.rs`, `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/src/generate/tests_all.rs`
- commit: `feat(interop): let a model carry the sealed block summarizer`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: add a field to `ServingConfig` or to the prompt-cache key; add a registry of named summarizers (that needs the list-capable config of slice 2); change what `seal` decides; edit `kv_seal_tests.rs` (its tests belong to FT4.23 and FT4.6).
- gpu: none

### 4.6 Prove key min and max runs through the summary slot

- id: FT4.6
- needs: FT4.5
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/kv_seal_tests.rs` (FT4.23): the loader lines and the config this card reuses; `LoadedModel::with_block_summarizer` (FT4.5);
  - `proxima-model-interop/src/generate/residency_caches.rs::BlockSummarizer`, `LayerCache::summarize_sealed` (FT4.4);
  - this file's fixture section: the record layout (per-dimension minimum then maximum over the concatenated key).
- change: none to non-test code. This is the proof card for the summary hook: a block-sparse-read technique needs one fold of each sealed block's keys, and the slot expresses it with about 15 lines of test code and no library edit.
- test: in `kv_seal_tests.rs` add the `pub(super)` summarizer
  ```rust
  pub(super) fn key_minmax(k_even: &[f32], k_odd: &[f32], _value: &[f32], even_odd_row: usize) -> Vec<f32> {
      let width = 2 * even_odd_row;
      let mut low = vec![f32::INFINITY; width];
      let mut high = vec![f32::NEG_INFINITY; width];
      for (even_row, odd_row) in k_even.chunks_exact(even_odd_row).zip(k_odd.chunks_exact(even_odd_row)) {
          for (slot, key) in even_row.iter().chain(odd_row).enumerate() {
              low[slot] = low[slot].min(*key);
              high[slot] = high[slot].max(*key);
          }
      }
      [low, high].concat()
  }
  ```
  and two tests:
  - `key_minmax_summaries_equal_an_independent_fold_of_the_decoded_rows`: `.with_block_summarizer(key_minmax)` on the synthetic checkpoint, `prefill_with` on `prompt_of(40, '3')` under the FT4.23 config. Assert `state.len() == 40` (measured, see FT4.23); with `rows = 40`: the full layer's `k_even.len()` divides evenly by `rows` (derive `even_odd_row = k_even.len() / rows`); `block_summaries.len() == 4` (`sealed_end == 32`, four blocks of eight rows); every record has `4 * even_odd_row` values; for each block `b` the record equals, bit for bit, a second fold written differently in the test (per key slot, `rows_of_the_block.iter().map(|row| row[slot]).fold(f32::INFINITY, f32::min)` and the same with `f32::NEG_INFINITY` and `f32::max`, over rows rebuilt from `k_even[b * 8 * even_odd_row..(b + 1) * 8 * even_odd_row]` and `k_odd`); at least one slot has `low < high` (the rows are not all equal, so the fold is not degenerate);
  - `a_horizon_past_the_prompt_seals_and_summarizes_nothing`: the control that must fail to summarize: the same run with `seal_horizon_rows: 1000` gives the full layer `sealed_end == 0` and empty `block_summaries`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_6 cargo nextest run -p proxima-model-interop --features std -E 'test(/key_minmax_summaries_equal_an_independent_fold_of_the_decoded_rows|a_horizon_past_the_prompt_seals_and_summarizes_nothing/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/generate/kv_seal_tests.rs`
- commit: `test(interop): drive key min and max through the summary slot`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: change non-test code; move `key_minmax` into non-test code.
- gpu: none

### 4.7 The worked sealed-end trace

- id: FT4.7
- needs: FT4.6, FT4.22, FT4.16, FT4.19
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - this file's fixture section: the 13-row trace table and the two block records;
  - `proxima-model-interop/src/generate/residency_caches.rs::layer_cache_sealing_tests` (`row_at`, `cache_with_rows`) and `LayerCache::seal`, `summarize_sealed`, `try_truncate`;
  - `proxima-model-interop/src/generate/kv_seal_tests.rs::key_minmax` (FT4.6), reached from the test as `crate::generate::kv_seal_tests::key_minmax`.
- change: none to non-test code (a test-only card; the test is the use of the earlier cards together).
- test: add `seal_worked_trace` in `layer_cache_sealing_tests`: run the 13-row trace table exactly, with an empty `LayerCache::new()`; after each append call `seal(2, 4, 1)` then `summarize_sealed(2, 1, key_minmax)` (call these directly, not `seal_attention_layers`); `try_truncate` for the three rewinds. Assert `rows` (`k_even.len() / 2`) and `sealed_end` after every table row; the last rewind returns `Err(InteropError::RewindIntoSealed { keep_positions: 7, sealed_end: 8 })` and `k_even.len() == 16`; at the end `block_summaries` has two records equal to the two block records of the fixture section (compare with `to_bits`); the test counts 10 appends and 3 rewind attempts and asserts both counts.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_7 cargo nextest run -p proxima-model-interop --features std -E 'test(/seal_worked_trace/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`
- commit: `test(interop): walk ten appends and three rewinds through the seal`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: change non-test code.
- gpu: none

### 4.8 Property test: no rewindable row is ever sealed

- id: FT4.8
- needs: FT4.16, FT4.22
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::layer_cache_sealing_tests` (FT4.22, FT4.16): `LayerCache::seal` and `try_truncate`;
  - `proxima-model-interop/src/generate/arena.rs` tests (~line 277, `the_arena_agrees_with_a_map_over_10000_generated_operation_sequences`): the `TestRunner::new(Config { cases, failure_persistence: None, ..Config::default() })` style, the `vec(..)` strategy and the executed-cases counter to copy (`proptest` is a dev-dependency of the crate, `proxima-model-interop/Cargo.toml` ~line 360).
- change: none to non-test code.
- test: add `seal_proptest_never_seals_a_rewindable_row` in `layer_cache_sealing_tests`: a test-local `enum Op { Append(usize), Rewind(usize) }`; `cases: 256`; strategy: `(0usize..=3, vec(op, 1..60))` where `op` is `Append(1..=5)` or `Rewind(0..40)`. State: a `LayerCache::new()` and a model `rows`. Per op, with `before = cache.sealed_end`: an `Append(count)` appends `row_at(rows..rows + count)`, adds `count` to `rows`, calls `seal(2, 4, horizon)` and asserts `cache.sealed_end == before.max(rows.saturating_sub(horizon) / 4 * 4)` (the formula written out, not `seal_target`); a `Rewind(keep)` calls `try_truncate(keep, 2, 1)`: when `keep < before` it matches `Err(RewindIntoSealed { keep_positions, sealed_end })` with `keep_positions == keep` and `sealed_end == before` and changes nothing, otherwise it is `Ok` and `rows = rows.min(keep)`. After every op assert `cache.k_even.len() == rows * 2`, `cache.sealed_end % 4 == 0`, `cache.sealed_end >= before` and `cache.sealed_end <= rows`. After the run assert the counter of executed cases equals 256.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_8 cargo nextest run -p proxima-model-interop --features std -E 'test(/seal_proptest_never_seals_a_rewindable_row/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/generate/residency_caches.rs`
- commit: `test(interop): property check that no rewindable row ever seals`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: assert `sealed_end <= rows - horizon` after a rewind (it does not hold there; see drift item 8); change non-test code.
- gpu: none

### 4.9 A placed buffer over caller memory

- id: FT4.9
- needs: none
- budget: 20 min
- crate(s): omega (features: metal, metal-output-placement)
- read first:
  - `omega/src/metal/execute_and_hazards.rs::allocate_placed_buffer (~line 1038)`: the owning form this card sits beside, and its `device_and_queue()` call;
  - `omega/src/metal/resident_nocopy_cache.rs::create_no_copy_buffer (~line 205)`: the existing `newBufferWithBytesNoCopy` call with `None` deallocator; page alignment precondition `omega/src/metal/pipeline_buffers_upload.rs::is_page_aligned (~line 1185)` and `page_size (~line 1175)`; both reach `execute_and_hazards.rs` through `use super::*` (the `pub use` globs in `omega/src/metal/mod.rs`);
  - `omega/src/lib.rs (~line 84)`: the `pub use metal::{ PlacedBuffer, allocate_placed_buffer, .. }` list;
  - `omega/src/metal/device_buffers_arena_plan.rs::MetalError (~line 274)`: `CompileFailed { log }` is the variant `allocate_placed_buffer` uses for refusal.
- change:
  1. `omega/src/metal/execute_and_hazards.rs`: add (cfg `metal-output-placement`) `pub unsafe fn allocate_placed_buffer_over(base: *mut u8, byte_len: usize) -> Result<PlacedBuffer, MetalError>`: returns `Err(MetalError::CompileFailed { log: "a placed buffer over caller memory needs a page-aligned, non-null pointer and a non-zero page-aligned length".to_string() })` when `base.is_null()`, `byte_len == 0` or `is_page_aligned(base.cast(), byte_len)` is false (reuse the variant `allocate_placed_buffer` uses; no new variant); otherwise `let (device, _queue) = device_and_queue()?; create_no_copy_buffer(&device, base.cast(), byte_len)` (it returns the `MetalBuffer` alias, the same type as `PlacedBuffer`; if `create_no_copy_buffer` or `device_and_queue` is not in scope through `use super::*`, import it by its module path under `omega/src/metal/`). Docs name the primitives it composes (`allocate_placed_buffer` for the owning form, `create_no_copy_buffer` for the call) and say why it exists: the placement of a kv buffer is the caller's decision, and this is how a caller places it in memory it owns. `# Safety`: the memory outlives every use of the returned buffer and is not freed, remapped or reused while a command buffer is in flight; the buffer never owns it (deallocator `None`). `# Errors`: the refusal above and a device refusal.
  2. `omega/src/lib.rs`: add `allocate_placed_buffer_over` to the re-export list that carries `allocate_placed_buffer`.
- test: add `placed_buffer_over_mmap_range_round_trips` in `omega/tests/placed_buffer_over.rs` (inner attributes `#![cfg(all(target_os = "macos", feature = "metal", feature = "metal-output-placement"))]` and `#![allow(clippy::unwrap_used, clippy::expect_used)]`, the pair the other omega test files carry because `omega` sets `[lints] workspace = true`): `page = omega::page_size()`; `libc::mmap` an anonymous private range of `2 * page`; `allocate_placed_buffer_over` on it is `Ok`; `omega::write_placed_buffer_f32(&buffer, 0, &[1.0, 2.0, 3.0])`, then reading the three `f32`s through the raw pointer gives `[1.0, 2.0, 3.0]` (the buffer aliases the range, no copy); writing `[4.0, 5.0, 6.0]` through the raw pointer then `omega::read_placed_buffer_f32(&buffer, 0, 3)` gives `[4.0, 5.0, 6.0]`; four refusals, each `Err(MetalError::CompileFailed { .. })`: the pointer plus 1 with length `page`, length `page + 1`, a null pointer with length `page`, and length 0; drop the buffer, then `munmap` last.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_9 cargo nextest run -p omega --features metal -j 1 -E 'test(/placed_buffer_over_mmap_range_round_trips/)'`
- expect: `1 passed`
- also green: `cargo clippy -p omega --features metal --all-targets`
- stage: `omega/src/metal/execute_and_hazards.rs`, `omega/src/lib.rs`, `omega/tests/placed_buffer_over.rs`
- commit: `feat(omega): wrap caller memory as a placed buffer without copying`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: change `allocate_placed_buffer`; add an error variant.
- gpu: one run (one buffer creation), waiting for a quiet box (the peer-gate check in SPEC "machine safety")

### 4.10 Take device kv buffers from a given source

- id: FT4.10
- needs: FT4.9
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/src/generate/device_kv.rs::DeviceKvLayer::allocate (~line 67)` (the three `allocate_placed_buffer` calls) and `DeviceKv::adopt (~line 208)`: the only places the kv buffers are created;
  - `proxima-model-interop/src/generate/device_kv.rs` tests (`mod tests (~line 391)`): `adopted (~line 420)`, `seeding_zeroes_every_row_past_the_kept_ones_whatever_the_buffer_held (~line 553)` and `a_cache_that_misses_positions_is_not_adopted` call `adopt` or `allocate` and must pass the new parameter;
  - `proxima-model-interop/src/generate/decode.rs (~line 3944)`: the single production call of `DeviceKv::adopt`;
  - `omega::allocate_placed_buffer` (`omega/src/metal/execute_and_hazards.rs ~line 1038`): the default source.
- change:
  1. `proxima-model-interop/src/generate/device_kv.rs`: add `pub(super) type KvBufferSource = fn(usize) -> Result<PlacedBuffer, omega::MetalError>;` with a doc naming the primitives ("the placement hook of the device kv: the allocator the three buffers of each layer come from; the default is [`omega::allocate_placed_buffer`]; [`omega::allocate_placed_buffer_over`] wraps memory a caller owns"). `DeviceKvLayer::allocate` and `DeviceKv::adopt` take `source: KvBufferSource` as their last parameter and create each buffer with `source(capacity_rows * row_bytes)?` where they call `allocate_placed_buffer(..)?` today. Zero filling, sizing and everything else are unchanged.
  2. `proxima-model-interop/src/generate/decode.rs`: the `DeviceKv::adopt` call passes `allocate_placed_buffer` as the new last argument.
  The three existing test call sites in `device_kv.rs` pass `allocate_placed_buffer` too.
- test: add `adopt_takes_every_buffer_from_the_given_source` in the `device_kv.rs` tests: a test-local `static REQUESTED: std::sync::Mutex<Vec<usize>>` and `fn recording_source(byte_len: usize) -> Result<PlacedBuffer, omega::MetalError>` that pushes `byte_len` and returns `allocate_placed_buffer(byte_len)`. Adopt two layers (`host_cache(None, 37)` and `host_cache(Some(8), 37)`, widths `Attention { even_odd_row: EVEN_ODD_ROW, v_row: V_ROW }` twice, `cached_len = 37`, `positions_needed = 80`, bucket 4, step rows 3) with `recording_source`: the result is `Some`; `REQUESTED` holds exactly 6 sizes; for each layer's three sizes `[a, b, c]`: `a == b`, `a * V_ROW == c * EVEN_ODD_ROW` and `a > 0`.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_10 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_10 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/generate::device_kv::tests/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_10/run.log`
- expect: `7 passed` (the 6 existing tests of that module plus the new one; confirm the 6 with `cargo nextest list` first). The three existing tests that call `adopt` or `allocate` now pass `allocate_placed_buffer`, so they run the default source on real placed buffers. This card loads no model: the default source is held to the recorded llama ids by the E2B arm of the proof card, which compares the default run, the hooked run and the recorded ids in one process
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/device_kv.rs`, `proxima-model-interop/src/generate/decode.rs`
- commit: `refactor(interop): take device kv buffers from a given source`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: skip the tail zeroing in `seed`; add a counter or a telemetry field; change buffer sizes; run a model-loading test.
- gpu: one run (placed buffer creation under `-j 1`), waiting for a quiet box (the peer-gate check in SPEC "machine safety")

### 4.11 Let a model carry the device kv buffer source

- id: FT4.11
- needs: FT4.9, FT4.10
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/src/generate/load_model.rs::LoadedModel (~line 803)`: the struct, gated fields use `#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]` (see `single_range`, `plan_life`);
  - `proxima-model-interop/src/generate/pregather.rs::with_ring_write_offset_for_parity_control (~line 2489)` and the three literals (~lines 2713, 2844, 2965);
  - `proxima-model-interop/src/generate/tests_all.rs::memory_fit_gate_tests::model_with (~line 2558)`: the fourth literal;
  - `proxima-model-interop/src/generate/decode.rs (~line 3944)`: the `DeviceKv::adopt` call FT4.10 left passing `allocate_placed_buffer`.
- change:
  1. `proxima-model-interop/src/generate/load_model.rs`: `LoadedModel` gains, under the cfg above, `pub(super) kv_buffer_source: KvBufferSource` (import it from `device_kv` under the same cfg; doc: where the device-resident kv buffers of a decode come from; `allocate_placed_buffer` unless a caller replaced it; placement only, so no cache key changes: the same values land in other memory).
  2. `proxima-model-interop/src/generate/pregather.rs`: each of the three literals gains, under the same cfg, `kv_buffer_source: allocate_placed_buffer,`. Beside `with_ring_write_offset_for_parity_control` add, under the same cfg, `#[must_use] pub fn with_kv_buffer_source(mut self, source: KvBufferSource) -> Self` that sets the field and returns `self`. Doc (teaching surface): the placement hook of the device kv, [`DeviceKv::adopt`] in `device_kv.rs`; compose it with [`omega::allocate_placed_buffer_over`] to put the kv in memory you own; the default reproduces today's allocation. If the compiler reports the private alias in a public signature, spell the `fn(usize) -> Result<omega::PlacedBuffer, omega::MetalError>` type in the parameter.
  3. `proxima-model-interop/src/generate/decode.rs`: the `DeviceKv::adopt` call passes `self.kv_buffer_source` instead of `allocate_placed_buffer`.
- test: in `tests_all.rs` `memory_fit_gate_tests` (the `model_with` literal also gains `kv_buffer_source: allocate_placed_buffer,` under the cfg, with `use omega::allocate_placed_buffer;` in that module): add `with_kv_buffer_source_replaces_the_default_source`: a test-local `static REQUESTS: AtomicUsize` and `fn counting_source(byte_len: usize) -> Result<omega::PlacedBuffer, omega::MetalError>` that increments it and returns `allocate_placed_buffer(byte_len)`. The control comes first: `(model_with(1_000_000).kv_buffer_source)(4096)` is `Ok` and leaves `REQUESTS == 0`. Then `let model = model_with(1_000_000).with_kv_buffer_source(counting_source);` and `(model.kv_buffer_source)(4096)` is `Ok` with `REQUESTS == 1`.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_11 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/with_kv_buffer_source_replaces_the_default_source/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`; `cargo check -p proxima-model-interop --features std`
- stage: `proxima-model-interop/src/generate/load_model.rs`, `proxima-model-interop/src/generate/pregather.rs`, `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/src/generate/tests_all.rs`
- commit: `feat(interop): let a model carry the device kv buffer source`
- done when: the expect line printed, clippy clean, both checks pass, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: add a field to `ServingConfig` or to the prompt-cache key; add an `Allocation` enum; wire any other `allocate_placed_buffer` call in `decode.rs` (the logits, dense-attention and state buffers keep their allocator).
- gpu: one run (one placed buffer creation), waiting for a quiet box (the peer-gate check in SPEC "machine safety")

### 4.12 Place gemma4 E2B kv in caller memory and match the recorded ids

- id: FT4.12
- needs: FT4.9, FT4.11
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity (~line 668)`, `llama_cases (~line 617)`, `first_divergence (~line 661)`, `LLAMA_GENERATED_TOKENS`, `GEMMA4_E2B (~line 54)`: the replay of the vendored `tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json` (3 prompts); `llama_parity` itself is left untouched;
  - `omega::allocate_placed_buffer_over` (FT4.9), `omega::page_size` (`omega/src/metal/pipeline_buffers_upload.rs ~line 1175`) and `LoadedModel::with_kv_buffer_source` (FT4.11);
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/gguf_kv.txt`: `gemma4.block_count = 35` and `gemma4.attention.shared_kv_layers = 20`, so 15 layers own a kv cache.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs` (test file; add `use std::sync::OnceLock;` and `use std::sync::atomic::{AtomicUsize, Ordering};` under the cfg `all(feature = "metal", target_os = "macos")`, as are all items below): the arena, which is the mmap reservation expressed as a placement source:
     ```rust
     const ARENA_BYTES: usize = 4 << 30;
     static ARENA_BASE: OnceLock<usize> = OnceLock::new();
     static ARENA_USED: AtomicUsize = AtomicUsize::new(0);
     static ARENA_REQUESTS: AtomicUsize = AtomicUsize::new(0);

     fn map_arena() -> usize {
         let pointer = unsafe {
             libc::mmap(std::ptr::null_mut(), ARENA_BYTES, libc::PROT_READ | libc::PROT_WRITE,
                 libc::MAP_ANON | libc::MAP_PRIVATE | libc::MAP_NORESERVE, -1, 0)
         };
         assert_ne!(pointer, libc::MAP_FAILED, "the kv arena could not be reserved");
         pointer as usize
     }

     fn arena_source(byte_len: usize) -> Result<omega::PlacedBuffer, omega::MetalError> {
         let page = omega::page_size();
         let length = byte_len.max(1).div_ceil(page) * page;
         let offset = ARENA_USED.fetch_add(length, Ordering::SeqCst);
         assert!(offset + length <= ARENA_BYTES, "the kv arena is exhausted at {offset} bytes");
         ARENA_REQUESTS.fetch_add(1, Ordering::SeqCst);
         let base = *ARENA_BASE.get_or_init(map_arena);
         unsafe { omega::allocate_placed_buffer_over((base + offset) as *mut u8, length) }
     }
     ```
     and `fn generated_ids(model: &LoadedModel<'_>, cases: &[LlamaCase], config: &ServingConfig<'_>) -> Vec<Vec<u32>>` (one `generate_from_ids(&case.prompt_ids, LLAMA_GENERATED_TOKENS, config, &mut |_event| ControlFlow::Continue(()))` per case, keeping the ids) and `fn kv_in_caller_memory(checkpoint: &Checkpoint, kv_owning_layers: usize, matches_llama: bool)`:
     - load the cases, map the checkpoint, build `config = ServingConfig { prompt_cache: PromptCacheConfig::off(), ..ServingConfig::default() }` exactly as `llama_parity` does;
     - `default_ids = generated_ids(&model, ..)`; assert `ARENA_REQUESTS.load(..) == 0` after it (control: the default source never touches the arena); store 0 into `ARENA_USED` and `ARENA_REQUESTS`;
     - `let hooked = model.with_kv_buffer_source(arena_source);` `hooked_ids = generated_ids(&hooked, ..)`; `assert_eq!(hooked_ids, default_ids)` for the whole vectors;
     - when `matches_llama`: for each case, with `compared = hooked_ids[i].len().min(case.generated_ids.len())`, `first_divergence(&case.generated_ids, &hooked_ids[i][..compared]) == None`;
     - `assert_eq!(ARENA_REQUESTS.load(..), 3 * kv_owning_layers * cases.len())` (3 buffers per kv-owning layer, one adoption per generate call) and `ARENA_USED.load(..) > 0`.
- test: add `kv_in_caller_memory_gemma4_e2b` calling `kv_in_caller_memory(&GEMMA4_E2B, 15, true)`: the hooked ids equal the default ids for all 3 prompts and match the recorded llama ids; the arena served exactly `3 * 15 * 3 = 135` buffer requests. If the run prints another count, stop and report that count: either the device kv path declined adoption for a prompt or the layer count premise is false.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_12 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_12 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^kv_in_caller_memory_gemma4_e2b$/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_12/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(interop): place gemma e2b kv in caller memory and match llama`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, the commit landed with that message, and the target dir is removed
- do not: edit `llama_parity` or any `llama_parity_*` test (their count is a guard); run another model-loading process; edit the fixtures; add a page-residency or copy-count assertion.
- gpu: one run (one model load, 6 generations), waiting for a quiet box (the peer-gate check in SPEC "machine safety")

### 4.13 Place gemma4 26B MoE kv in caller memory and match the recorded llama ids

- id: FT4.13
- needs: FT4.12
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::kv_in_caller_memory` (FT4.12) and `GEMMA4_26B (~line 48)`;
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/gguf_kv.txt`: `gemma4.block_count = 30`, `gemma4.attention.shared_kv_layers = 0`, `gemma4.expert_count = 128`: 30 layers own a kv cache;
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/llama_ids.json` (3 records, chat-templated prompts on which llama is confident) and `proxima-tensor/specs/architecture-as-data/SPEC.md` line 224 (the 26B replay `llama_parity_gemma4_26b` passed on main): the 26B has a live llama oracle, so this arm holds the hooked run to those recorded ids as well as to the default run.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add the test below. Nothing else.
- test: add `kv_in_caller_memory_gemma4_26b` calling `kv_in_caller_memory(&GEMMA4_26B, 30, true)`: the hooked ids equal the default ids for all 3 prompts and match the recorded llama ids (the control in the helper shows the default run touched no arena); the arena served exactly `3 * 30 * 3 = 270` buffer requests. If the run prints another count, stop and report it.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_13 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_13 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^kv_in_caller_memory_gemma4_26b$/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_13/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(interop): place gemma moe kv in caller memory and match llama`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`, the commit landed with that message, and the target dir is removed
- do not: run another model-loading process (this checkpoint is 13.3 GB); edit the fixtures; change `kv_in_caller_memory` or `llama_parity_gemma4_26b`.
- gpu: one run, waiting for a quiet box (the peer-gate check in SPEC "machine safety")

### 4.14 Place granite MoE kv in caller memory and match the recorded ids

- id: FT4.14
- needs: FT4.12, FT0.43
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::kv_in_caller_memory` (FT4.12);
  - `proxima-model-interop/tests/arch_data_baseline.rs::Checkpoint (~line 41)`, `::GEMMA4_E2B (~line 54)` and `::llama_cases (~line 617)`: the checkpoint row shape, the replay twin whose const the granite row copies, and the reader of `tests/fixtures/llama-parity/<name>/llama_ids.json` (3 records) that the granite checkpoint is read through;
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/gguf_kv.txt` and `gemma4_e2b/llama_ids.json` (on main): the layout the granite fixtures under `granite_moe/` repeat; the granite 1B-A400M header declares `granitemoe.block_count = 24` with no shared kv layers, so 24 layers own a kv cache, and the arena request count printed by the run is the check on that number.
  - Not anchors: the granite checkpoint const `GRANITE_MOE` (name `"granite_moe"`, `architecture: "granitemoe"`), the fixtures under `granite_moe/` and the test `llama_parity_granite_moe` are outputs of the cards FT0.31, FT0.41 and FT0.43 named in `needs`, and exist only after those cards land; a card in `needs` is a dependency, not a path to read at the commit the anchors were taken.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add the test below. Nothing else. Precondition (stop and report if it prints anything but `1`; `cargo nextest list` prints test names and no total, so the count is taken with `grep -c`): `cargo nextest list -p proxima-model-interop --features std,metal -E 'binary(arch_data_baseline) & test(/^llama_parity_granite_moe$/)' 2>/dev/null | grep -c 'llama_parity_granite_moe$'` prints `1`, which is the witness that the granite cards in `needs` landed. This card does not run the replay: loading the model twice would be a second model-loading run.
- test: add `kv_in_caller_memory_granite_moe` calling `kv_in_caller_memory(&GRANITE_MOE, 24, true)`: the hooked ids equal the default ids for all 3 recorded prompts and match the recorded llama ids; the arena served exactly `3 * 24 * 3 = 216` buffer requests (3 buffers per kv-owning layer, 24 layers, 3 prompts, one adoption per generate call). If the run prints another count, stop and report that count: either the device kv path declined adoption for a prompt or the layer count premise is false. No granite-specific code path exists in the library; the model reaches the placement hook through its profile and descriptor.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_14 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_4_14 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^kv_in_caller_memory_granite_moe$/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_4_14/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(interop): place granite moe kv in caller memory and match llama`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`, the commit landed with that message, and the target dir is removed
- do not: run another model-loading process; edit the fixtures; edit `llama_parity_granite_moe`; add a granite branch anywhere in `src/`.
- gpu: one run (one model load, 6 generations), waiting for a quiet box (the peer-gate check in SPEC "machine safety")

## spec drift

1. SPEC/TASKS: "`libc::mmap` per layer cache" and "`newBufferWithBytesNoCopy` over the range on the placed path". On main the host `LayerCache` planes are `Vec<f32>` read at about 50 call sites, and the placed path allocates each layer at full capacity (`DeviceKvLayer::allocate`), so it never grew by copying. The re-cut builds no reservation type: FT4.9 is the primitive (a placed buffer over caller memory), FT4.10 and FT4.11 are the hook (a buffer source the model carries), FT4.12 to FT4.14 express the lazily-backed reservation as an mmap arena in a test. No claim is made about page residency, mapped-page counts or row copies; the proof is byte-identical ids through memory the caller owns.
2. `libc` is already a dev-dependency of `proxima-model-interop` (`Cargo.toml` ~line 345) and of `omega` through its `metal` feature; no card edits `Cargo.toml`.
3. SPEC/TASKS: "`kv_ring.rs:121`: ring layers seal only rows outside the ring". A row outside the ring has been overwritten and does not exist in the layer. Cards: ring layers never seal (`seal` returns an empty range, `sealed_end` stays 0). Consequence for gemma4's sliding layers (restore through `RingCheckpoint`s, not blocks) is carried by slice 5. Also, `LayerCache::ring` builds through `Self::new()` and assignment, not a struct literal, so `kv_ring.rs` needs no edit (the earlier cut said it did).
4. SPEC/TASKS: "`arena.rs` and `prefix_trie.rs`: sealed blocks carry the content key". `arena.rs` is a generic slab (`Arena<Item>`), not a block store. The content key is slice 5's `chained_keys`. `arena.rs` is untouched.
5. Not built: "trie insert moves from request end to seal". With one request in flight per `LoadedModel` (`&mut self`) there is no production caller for a provisional trie insert, so a card for it would add dead code. The owner decides whether a cancel path justifies it.
6. Device-resident layers hold no host rows during decode: `DeviceKv::adopt` empties the host planes and `flush` rebuilds them after the call. `seal` reads the row count from the layer's own planes, so on gemma4 under Metal the seal point seals nothing for the full-attention layers (and never for ring layers); it fires for host-resident layers (the CPU path, which FT4.23's decode test uses on the synthetic checkpoint). Sealing device-resident rows is a placed-buffer fold the hook does not yet cover (sketch 08, device residency gap).
7. `seal` and the summary slot take `block_tokens` as an argument, so unit tests use 4 (not a multiple of 16). A config validity rule on the block size belongs to configuration (slice 2), not to the primitive. The block size is `PromptCacheConfig::block_tokens`; there is no second block size.
8. The seal-trace table is defined in this file (`worked-examples.md` does not exist at e4cf9beb); slice 0 must copy it verbatim. The earlier property asserted `sealed_end <= rows - horizon` after every op, which is false after a rewind (the trace itself has `sealed_end = 8` at `rows = 8` with `horizon = 1`); the invariant that holds is `sealed_end <= rows`, plus the exact formula right after an append.
9. `LayerCache::truncate` has four production callers on main, not the two the earlier cut read: the in-flight speculative commit (`decode.rs ~5818`) and three whole-state rewinds (`chunk_shift.rs ~317`, `prompt_cache.rs ~323`, `ring_checkpoint.rs ~129`). Only the first can reach rows a seal protects from an in-flight decode; the other three discard rows they rebuild, so refusing there would stop the prompt cache from reusing an entry it diverges from. They keep the infallible `truncate`, which lowers the sealed end (and, from FT4.18, drops the summaries of blocks no longer whole). The rewind that rectification needs is truncate-then-append through `try_truncate` (sketch 08 abandons a separate overwrite primitive).
10. The summary slot is reached from code (`with_block_summarizer`), not from configuration: a list of named summarizers in configuration needs the list-capable serving config of slice 2, because `ServingConfig` is `Copy` (`serving.rs ~line 720`). Entries stored before a summarizer arrives are dropped by the builder, and a block size change re-seals a layer from scratch, so a summary never indexes the wrong block.
11. SPEC rule (decisions in core): the seal decision is `proxima_core::kv_decision::seal_target` (FT4.0) and `sealed_blocks` (FT4.20). TASKS slice 4 lists no proxima-core file; FT4.0 adds `proxima-core/src/kv_decision.rs` and one `pub mod` line.
12. The earlier cut ran a four-checkpoint llama parity count after two cards. This cut runs the single-checkpoint replay `llama_parity_gemma4_e2b` after the cards that touch the decode loop or the device kv (FT4.22, FT4.3); FT4.10 loads no model, so the default source of the device kv is held to the recorded ids by FT4.12, which compares the hooked run with the default run and with the recorded ids in one process. The 26B and granite arms are held to the recorded llama ids the same way.

## slice exit

- `cargo nextest run -p proxima-core -E 'test(/kv_decision_/)'` prints `3 passed`.
- `cargo nextest run -p proxima-model-interop --features std -E 'test(/layer_cache_sealing_tests|layer_cache_truncate_tests|kv_seal_tests|decode_keeps_one_summary_record_per_sealed_block|with_block_summarizer_drops_entries_stored_before_it|prefix_state_byte_len_counts_block_summaries/)'` prints `27 passed` (18 in `layer_cache_sealing_tests`: 3 rewind, 5 seal, 2 seal range, 1 block size, 2 records, 2 summarizer, 1 summary method, 1 worked trace, 1 property; 3 in `layer_cache_truncate_tests`; 3 in `kv_seal_tests`: 1 decode seal, 2 key min and max; 2 in `tests_all.rs` `block_summarizer_tests`; 1 byte-accounting test).
- `cargo nextest run -p proxima-model-interop --features std -E 'test(/seal_horizon_defaults_to_256_rows|prompt_cache_builder_matches_toml_and_env_loaders|default_settings_lower_to_the_standard_config|a_zero_byte_budget_from_the_environment_turns_the_cache_off/)'` prints `4 passed`.
- `cargo nextest run -p omega --features metal -j 1 -E 'test(/placed_buffer_over_mmap_range_round_trips/)'` prints `1 passed`; `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/generate::device_kv::tests|with_kv_buffer_source_replaces_the_default_source/)'` prints `8 passed`.
- three separate model-loading runs under `-j 1` (one process at a time): `kv_in_caller_memory_gemma4_e2b`, `kv_in_caller_memory_gemma4_26b`, `kv_in_caller_memory_granite_moe`, each `1 passed`; the replay `llama_parity_gemma4_e2b` prints `1 passed` after them.
- Premises to check live (each is a stop-and-report if false): `proxima_core` resolves from `proxima-model-interop` under `--features std` (FT4.3); `chunked_prefill_tests::gemma4_checkpoint` builds one full-attention layer and two ring layers (FT4.23 asserts it); the device kv path adopts for every prompt of the E2B and 26B replays, giving the stated request counts (FT4.12, FT4.13); the granite cards landed through FT0.43 (FT4.14).
- Not cuttable: FT4.3, FT4.4 and FT4.5 each touch `decode.rs`, `residency_caches.rs` or the model literals and must land in order; FT4.11 changes four struct literals because a new field must reach every construction site in the same commit.
- Ids removed by the re-cut (never reused): none; ids FT4.5 to FT4.14 name different cards than in the earlier cut (see the map). FT4.16 to FT4.19, FT4.20 to FT4.22 and FT4.23 are additions made by the one-public-item and file-count splits; FT4.20 is a new card, not the old FT4.15 (the earlier cut's parity-count card, mapped above to FT4.12 to FT4.14). Execution order follows `needs`, not the order the cards appear in this file: FT4.0, FT4.20, FT4.1, FT4.21, FT4.22, FT4.2, FT4.3, FT4.23, FT4.16, FT4.17, FT4.18, FT4.4, FT4.19, FT4.5, FT4.6, FT4.7, FT4.8, FT4.9 to FT4.14.
