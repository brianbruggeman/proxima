# slice 16 (re-cut): the hook ledger and the last residual hooks (cards FT16.1 - FT16.5)

anchors read at main e4cf9beb (full sha e4cf9beb8342a80447342f0813ada9c84fc3a519) with `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`. Every `path::symbol (~line N)` below is the line at that sha; the executor re-locates by symbol. Paths are relative to the proxima repo root. Every card uses `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_16_<n>` (n is the card number) and removes it when done. Logs of a model run go under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_16_<n>/`.

Governing direction (owner, 2026-10-04): the hooks are built so techniques can be vetted later; a technique is not built. The earlier cut of this slice wrote fifteen technique-named doctests into the rustdoc of `ServingSettings` and `SpeculationSettings`, which puts technique names in library docs and counts them as shipped configuration. That is the opposite direction, so no card here writes a doctest. This slice follows every other slice, so its job is the residue: every hook gap named by the paper sketches (`pipeline-as-data/sketches/08` to `14`) and by the research changes (`pipeline-as-data/research.md`, catalog changes) is looked up in the other slices' cards, and a card exists here only for a gap that no card holds and that main shows to be real.

Result of that lookup: ledger A below lists the gaps and changes the sketches and the research name, each with the card that owns it, or the evidence that no card is needed, or its place in ledger C (parts not built). Five cards in this file cover the gaps no other card held: FT16.1 and FT16.2 (the read hook's cache binding, sketch 08 GAP-6: FT2.1 puts the read spec in the prompt cache key, FT16.1 proves the store separates entries by it, FT16.2 drops entries when the rule changes), FT16.3 (the idle list refusal, sketch 12 GAP-6), FT16.4 (selection grain, research change on block size) and FT16.5 (the stop proposal, research change on the fork and the stop).

Test models (owner, 2026-10-04): gemma4 for dense or MoE, granite for MoE. No qwen of any kind appears in any card of this file. No card here has a MoE arm. FT16.1 loads no model, FT16.2, FT16.3 and FT16.4 run the synthetic gemma4 checkpoint, and FT16.5 loads gemma4 E2B and exercises the stop path of the decode loop, which is the same function whatever the feed-forward kind; none of the five edits expert code. Oracles run once and are recorded: FT16.5 compares against the vendored `llama_ids.json`, and nothing queries Ollama or llama-server.

## old to new id map

| old card | new cards | verdict |
|---|---|---|
| FT16.1 | FT16.1, FT16.2, FT16.3, FT16.4 | recut: the rectified-attention doctest becomes the cache binding proofs of the read hook (FT16.1, FT16.2) and the summary grain proof (FT16.4); the sleep-time doctest becomes the idle list refusal (FT16.3); the cartridge doctest has no card here (FT14.2 to FT14.6 own the file, the key and the load stage) |
| FT16.2 | none | recut, no residual gap: cache blending is FT9.1 to FT9.11, segments are FT5.7 to FT5.16 and FT10.9, the threshold cascade is FT12.8 |
| FT16.3 | none | recut, no residual gap: cluster routing is FT12.11, the conformal cascade is FT12.7 and FT13.5, the isotonic cascade is FT12.9, validation is FT12.2 and FT13.1, calibration keyed by digest is FT13.2, the readouts judges read are FT11.1 to FT11.3 |
| FT16.4 | FT16.1, FT16.2 | recut: the tier and reserve proofs are FT5.12 to FT5.16 and FT4.12 to FT4.14, the sampled read proof is FT6.8; the parts no card held are the store-level proof of the read hook's cache binding (FT16.1) and the cache clear on a rule change (FT16.2) |
| FT16.5 | FT16.5 | recut: the nameable state and the accept rules are FT1.2 to FT1.12; the part no card held is the stop signal driven by a probe |
| FT16.6 | none | dropped |

Counts: 6 old cards, 0 kept, 5 recut, 1 dropped; 5 new cards.

## dropped

- old FT16.6 (a table in `docs/configuration.md` listing 15 techniques as shipped configuration, plus the final guards): the table advertises techniques as library configuration, which the pipeline-as-data direction forbids (a technique appears only as the proof that a hook expresses it), and its row count depended on the doctests this cut removes. The guard it carried, that no library type is named after a technique, moves into the slice exit below, with a control that proves the pattern matches real declarations.

## designs abandoned (what each constraint ruled out)

- One doctest per technique in rustdoc: abandoned for the proofs the other slices already hold in tests, because a doctest is library text and a library may not name a technique.
- A `summary_tokens` config field next to the block size (research change on selection grain): abandoned for FT16.4. The summary record is an opaque list of floats per sealed block (FT4.4), so the selection grain is the summarizer's choice, and the seal card forbids a second block size.
- A refusal row "seal horizon at least the widest verify shape": not written, because it can never fire for drafted verification. Evidence: the seal rule keeps a block only when its last row is at least the horizon behind the newest committed row (FT4.3), and a verify step rewinds to `keep_positions = cached_len_before_step + emitted.len()`, never below the committed length (`generate/decode.rs`, `let keep_positions` ~line 5808 at e4cf9beb). The replay pass is the only step that rewinds committed rows, and its row is FT7.7.
- A library `fork` transition on the serving state: abandoned, `state.clone()` per branch is the same call site, and FT11.4 already proves N seeded calls.
- Keying the cache by the address or description of a read rule: abandoned for dropping the entries when the rule is set (FT16.2), the same answer the expert sidecar and the block summarizer already take.

## ledger A: every named gap and its owner

Gaps are cited as sketch number and gap id; research changes as CC-n. "No card needed" always carries its evidence.

| gap | owner |
|---|---|
| 08 GAP-1 enter rule | FT7.1, FT7.3, FT7.4 |
| 08 GAP-2 verify program only for gemma4 | refused with a typed error by FT7.6; verify programs for other families belong to architecture-as-data |
| 08 GAP-3 verify carries `last`, one resume transition | FT7.2 |
| 08 GAP-4 commit by rewind then append, device step buffer | FT7.6, FT7.5, with the fallible rewind of FT4.1 |
| 08 GAP-5 replay rows in the ring slack | FT7.5 |
| 08 GAP-6 read field defaulting to dense | the serving-settings slice owns the field; FT6.10 binds it to the resident plan identity; FT2.1 binds it to the prompt cache key (`CacheKey.read`); FT16.1 adds the store-level proof that entries separate by it, and FT16.2 the clear on a rule change |
| 08 GAP-7 visibility term in the cached mask | FT6.3, FT6.4 (two-range engines); single-range engines are an engine change (FT6 file, decided section) |
| 08 GAP-8 selection kernel, GAP-9 fusion | FT3.1 to FT3.9 (the top-fraction cards); FT6.15 to FT6.18 |
| 08 GAP-10 summary storage and seal point | FT4.3, FT4.4, FT4.5 |
| 08 GAP-11 device-resident seal | no card, ledger C item 3 |
| 08 GAP-12 validation rows | FT7.7 (replay against the horizon); the verify-width half has no row to add (designs abandoned) |
| CC-3 selection grain separate from block size | FT16.4 |
| CC-4 validation rows | FT7.7 and the note above |
| CC-5 tie semantics of rank-count selection | FT3.2 (the lower index wins) |
| 09 GAP-1 more than one source entry, GAP-2 host halves, GAP-6 seam trim | no card, ledger C item 1 |
| 09 GAP-2 graph half, GAP-3 residual tap | FT9.5, FT9.3, FT9.4 |
| 09 GAP-4 chunk against ring capacity | no card needed: the run is refused at runtime by `generate/chunk_shift.rs::ring_rows_live` (~line 160), and no card makes the chunk size a configuration value, so no config row can state the bound |
| 09 GAP-5 provenance | FT9.7 |
| CC-1 key as a hash chain over layer groups | FT9.7 binds the whole assemble list as one digest; the per-layer-group chain is not built and the single digest errs only toward less sharing |
| 10 G1 eviction rule | FT5.7 |
| 10 G2 victim hand-off, G3 index larger than the resident set | FT5.10 |
| 10 G4 serialization | FT5.1 to FT5.9 |
| 10 G5 promote outside the lock | FT5.11 |
| 10 G6 codec parameter in the key | decided not built in the tiers file (identity codec, asserted byte for byte by FT5.12 and FT5.13) |
| 10 G7 only whole entries tier on hybrid models, CC-8 tier grain | entry grain stands, ledger C item 2 |
| 11 G1 stage list | FT8.2, FT8.4, FT8.7 |
| 11 G2 fresh entry shape, G3 placeholder ids | FT14.4 (the load stage) |
| 11 G4 file format and loader | FT14.2 (write), FT14.4 (load), on the block file of FT5.1 to FT5.9 |
| 11 G5 key identity of the assembled prefix | FT14.3 (content, never the path) |
| 11 G6 loading outside the cache | recorded as a limit by the sketch; the load stage runs inside `run_decode_loop_through_cache` (FT14.4), so no bypass route is added |
| 12 G1 job list | FT10.4, FT10.5 |
| 12 G2 lead tokens | FT10.2 |
| 12 G3 draft output | FT10.6 |
| 12 G4 queue policy | carried by the idle file, no card there; a caller drives many contexts through FT10.7 |
| 12 G5 keep marker | FT10.3 |
| 12 G6 validation, entry count half | FT16.3 |
| 12 G6 validation, byte half | no card, ledger C item 4 |
| 13 G1 home of the decision | FT12.1 (`proxima-core`, no serving state module needed) |
| 13 G2 no N-samples-per-step mode | not a gap: N samples are N seeded calls (FT11.4) |
| 13 G3 tier list validation | FT12.2, FT13.1 |
| 13 G4 the weaker pipe entry of the model | no card needed: the sketch states the technique does not use it, and FT12.12 holds the one-tier cascade to the direct call |
| 13 G5 readouts judges need | FT11.1 to FT11.3 |
| 13 G6 raw text in, raw text out (no template stage) | outside this slice family: the template stage is a separate hook with its own sketch |
| CC-10 calibration keyed by configuration digest | FT13.2 |
| 14 G1 nameable state | FT1.2 to FT1.6 |
| 14 G2 nothing reads the accept settings | FT1.10 to FT1.12 |
| 14 G3 unread snapshot, G4 rows verified | behaviour kept by FT1.3 and FT1.4 |
| CC-6 fork | FT11.4 (N seeded calls; the library has nothing to add) |
| CC-6 stop proposal | FT16.5 |

Research changes CC-2, CC-7, CC-9 and CC-11 are classified as active research in `research.md`, not engineering, and are not cut. CC-12 is an edit to the pipeline-as-data SPEC, not a card.

## ledger B: each technique and where its proof lives

| technique | hooks it uses | proof |
|---|---|---|
| rectified sparse attention | seal summaries FT4.3 to FT4.5, read FT6.3 to FT6.6, periodic replay FT7.1 to FT7.6, cache key FT2.1, store proof FT16.1, rule clear FT16.2, summary grain FT16.4 | FT4.6, FT6.6, FT7.8; FT15.1 to FT15.3 run the arms end to end |
| cartridges | block file FT5.1 to FT5.9, assemble list FT8.2 to FT8.7, key FT14.3, load stage FT14.4 | FT14.6, FT14.7, FT14.8 |
| sleep-time compute | draft step FT10.1 to FT10.3, list FT10.4, FT10.5, observer FT10.6, refusal FT16.3 | FT10.8 |
| cache blending | tap and edit FT9.3, FT9.4, positions FT9.5, key FT9.7 | FT9.9 |
| growing and sealed segments | seal FT4.3, tiers FT5.7 to FT5.11, idle jobs FT10.7 | FT10.9, FT5.12 |
| threshold, classifier and isotonic cascades, cluster routing | settle decision FT12.1, pipe FT12.3, router FT12.4, readouts FT11.1 to FT11.3 | FT12.8, FT12.10, FT12.9, FT12.11 |
| conformal cascade | settle FT12.1, FT12.5, validation FT12.2, FT13.1, calibration key FT13.2 | FT12.7, FT13.5 |
| action speculation, verifier speculation, macro commit | generic entry FT1.2 to FT1.8, accept rule FT1.10, FT1.11 | FT1.9, FT1.12 |
| tiered chunk cache | eviction FT5.7, spill FT5.10, restore FT5.11 | FT5.12 to FT5.16 |
| reserved key-value memory | buffer source FT4.9 to FT4.11 | FT4.12 to FT4.14 |
| verified sparse attention | read FT6.3, FT6.4, FT16.1 | FT6.8 |
| stop by probe, best of N | stop signal on the token callback, readouts FT11.1 to FT11.3 | FT16.5, FT11.4 |

## ledger C: parts not built here (the exact part, and why)

1. Request-scoped attachment of a caller's assemble stage, and the host halves of sparse-position prefill (09 GAP-1, the host halves of GAP-2, GAP-6; items 1, 3, 4 and 5 of the not-built list in the cache-blending file). Exact parts: a cache write at a position list (`generate/kv_ring.rs::LayerCache::append_at` takes one start position), RoPE inputs from a position list (`generate/residency_caches.rs::build_position_inputs` loops `start_position + offset`), per-request row-aligned inputs in `decode.rs::push_step_named_blocks`, a hook for a caller's stage to run in `prompt_cache.rs::run_decode_loop_through_cache` after the configured stages, and a trim over `Vec<ChunkRun>`. Why not cut now: the stage kind that would carry a source scope is a variant of `AssembleStep`, which the serving-settings recut owns and which is absent on main (`git grep -n AssembleStep main` prints nothing); and each of the host halves has no caller until the attachment exists, so a card for it would add dead code (CARDS.md coherence rule). Assumption for the card that closes it, chosen because three hooks already took it (FT4.5 summarizer, FT6.11 read rule, FT4.11 buffer source): the attachment is a carrier on the loaded model set by a setter, and lifting from a second entry needs no change to `generate/chunk_shift.rs::lift_chunks`, which already takes the source entry as a parameter.
2. Block-grain tiers (CC-8, 10 G7). The tiers file chose the entry as the unit (an entry at rest is immutable, and ring layers hold only a window), so a block that left the host while a read still attends it needs an edge from the read set to the tier that no sketch specifies. Nothing in this file depends on it.
3. Device-resident seal (08 GAP-11). Read on main: the device-resident path adopts the host caches once, when the last prefill batch runs (`generate/decode.rs`, `device_kv_attempted` block ~line 3937), and `generate/device_kv.rs::DeviceKv::adopt` empties the host row vectors. The seal card calls seal on host commits (FT4.3), so prompt blocks are sealed before adoption; blocks that fill during one device-resident decode stay unsealed until the next call's first commit and are read densely as part of the unsealed tail. Not measured: how many generated blocks a long generation leaves unsealed. No card, because the effect on the proofs (prompt blocks sealed, tail dense) is the safe direction and the number is unmeasured.
4. The byte half of the idle list refusal (12 G6): an entry's size is unknown until the first draft is stored, so the bytes bound can only be observed at store time inside `follow_up_branches`, which the idle file rewrites; the entry-count half is FT16.3.

## prerequisites outside this file (an executor whose premise is false stops and reports)

- The serving-settings slice: `ServingConfig.attention: AttentionConfig { read: ReadSpec }` exists, `CacheKey` holds `read: ReadSpec`, and `CacheKey::of` destructures `attention: AttentionConfig { read }` (FT2.1, change item 3; `attention: _` appears only in `PlanIdentity::of`). At main `git grep -n ReadSpec main -- proxima-model-interop/src` prints nothing. Needed by FT16.1.
- The read-hook slice: `LoadedModel::set_read_rule` (FT6.11). Needed by FT16.2.
- The seal slice: FT4.3 makes `config`, `gemma4_checkpoint` and `prompt_of` in `generate/chunked_prefill_tests.rs` visible to sibling test modules, and FT4.5 adds the summarizer carrier; FT4.6 adds the test-local `key_minmax` this file's FT16.4 builds on.
- The idle slice: `IdleStep`, `DraftStep`, `set_idle_schedule` and the list-driven `prewarm_queued` (FT10.4, FT10.5), `with_prewarm_worker_observed` (FT10.6). Needed by FT16.3.
- The readouts slice: `decode.readouts`, `TokenEvent.readout` and `TokenReadout` (FT11.1 to FT11.3). Needed by FT16.5.
- Main today holds none of these: `git ls-tree main proxima-model-interop/src` lists no `serving_settings.rs` and no `serving_grammar.rs`, `proxima-core/src` holds no serving module, `ServingConfig` is `Copy` (`serving.rs` ~line 719), `serving_fsm.rs` is dead code, and `git grep -n -i granite main -- proxima-model-interop proxima-tensor/src` prints nothing.

---

### 16.1 Prove the prompt cache separates entries by read spec

- id: FT16.1
- needs: FT2.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/prompt_cache_key.rs::CacheKey (~line 32 at e4cf9beb)` and `::CacheKey::of (~line 82)`: after FT2.1 (its change item 3) `CacheKey` holds `pub(super) read: ReadSpec` and `CacheKey::of` destructures `attention: AttentionConfig { read }` and stores it. This card changes none of it; it proves the store honours it;
  - `proxima-model-interop/src/generate/prompt_cache.rs`, tests: `key_under (~line 2801)`, `a_request_differing_in_a_row_affecting_config_field_misses (~line 2827)` with its 13 `#[case::...]` lines (~2814 to 2826), `a_request_differing_only_in_a_row_independent_field_shares_the_entry (~line 2893)` with its 6 cases, and `entries_under_two_configs_coexist_and_each_serves_its_own (~line 2916)`: the cases and the pattern this card copies;
  - `proxima-model-interop/src/generate/prompt_cache_key.rs`, tests: FT2.1's `cache_key_separates_read_specs` proves the two keys differ as values. This card proves the other half, that the cache store and `take_best` act on that difference.
- premises (stop and report if any is false):
  - `git grep -n "pub enum ReadSpec" -- proxima-model-interop/src` prints 1 line, and the derive above it lists `Copy` and `PartialEq`;
  - `git grep -n "Operand" -- proxima-model-interop/src/serving_grammar.rs` prints at least 1 line;
  - `git grep -n "pub(super) read: ReadSpec" -- proxima-model-interop/src/generate/prompt_cache_key.rs` prints 1 line, and `git grep -n "attention: _" -- proxima-model-interop/src/generate/prompt_cache_key.rs` prints 0 lines (a `attention: _` line means FT2.1 is absent and this card stops);
  - the before-run of the validate command below prints `19 passed` (13 cases of the affecting test plus 6 of the independent test, listed by `cargo nextest list` at a7c08c4c, whose test file holds the same case lines as main at e4cf9beb). Any other count, including `0 passed`, means a card between main and this one changed those tests: stop and report the count.
- why this card exists: rows a decode step stores depend on what the step attended. A step that skips cache rows computes different hidden states from a dense step, so the key and value rows it appends differ. The request's finished state, generated rows included, is stored back (`prompt_cache.rs::run_decode_loop_through_cache`, `self.prompt_cache_store(entry, &config)` ~line 1280), and a later turn's prompt contains the answer, so a dense request would resume from rows a dense run never computed. FT2.1 puts the read in the key; no test holds that the store refuses a dense entry to an operand request and serves each its own, and the row-affecting case list has no row for the read.
- change: none to non-test code. The key field exists after FT2.1, so this is a proof card.
- test: in the `tests` module of `prompt_cache.rs` (add `use crate::serving::AttentionConfig;` and `use crate::serving_grammar::ReadSpec;` if absent):
  - add `#[case::attention_read_operand(|config| ServingConfig { attention: AttentionConfig { read: ReadSpec::Operand }, ..config })]` to `a_request_differing_in_a_row_affecting_config_field_misses`: the stored default entry is not offered to the operand request, which reports `ConfigMismatch`;
  - add `an_entry_built_under_the_operand_read_serves_only_operand_requests`: `operand = ServingConfig { attention: AttentionConfig { read: ReadSpec::Operand }, ..ServingConfig::default() }`, `operand_key = key_under(&operand, RopeScaling::None, 0, 0)`; store `state_with_ids(&[2, 105, 2364, 107])` under the base key and a second entry with the same ids whose `key` is set to `operand_key` (the pattern of `entries_under_two_configs_coexist_and_each_serves_its_own`); with prompt `[2, 105, 2364, 107, 9259]`, `take_best` under `operand_key` returns an entry whose `key == operand_key` with path `CachePath::Extend`, `take_best` under `base_key()` returns an entry whose `key == base_key()`, and `assert_ne!(base_key(), operand_key)`.
- validate: run `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_16_1 cargo nextest run -p proxima-model-interop --features std -E 'test(/a_request_differing_in_a_row_affecting_config_field_misses|an_entry_built_under_the_operand_read_serves_only_operand_requests|a_request_differing_only_in_a_row_independent_field_shares_the_entry/)'` BEFORE editing and confirm it prints `19 passed` (the premise above); after editing run the same command
- expect: before: `19 passed`; after: `21 passed` (the new case and the new test; the 19 existing cases stay green unchanged)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`
- commit: `test(interop): prove the prompt cache separates entries by read spec`
- done when: the expect lines printed, clippy and the check clean, `git diff --cached --stat` touches only `generate/prompt_cache.rs` with every added line inside `mod tests`, and the commit landed with that message
- do not: edit `prompt_cache_key.rs` (FT2.1 owns the field); add the read spec to the resident plan identity (that card exists); hash or name a closure; touch `prewarm_queue.rs`; validate the floats of `ReadSpec` here (it holds none).
- gpu: none

### 16.2 Drop cached entries when the read rule changes

- id: FT16.2
- needs: FT6.11, FT4.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/load_model.rs::LoadedModel::set_read_rule` (FT6.11): the setter, taking `&mut self` and a closure;
  - `proxima-model-interop/src/generate/prompt_cache.rs::LoadedModel::clear_prompt_cache (~line 999 at e4cf9beb)`: its doc names the precedent, the expert sidecar, an input the key cannot carry, so setting it clears;
  - `proxima-model-interop/src/generate/pregather.rs::LoadedModel::with_block_summarizer` (FT4.5): the same one-line call for the same reason;
  - `proxima-model-interop/src/generate/kv_seal_tests.rs::with_block_summarizer_drops_entries_stored_before_it` (FT4.5): the model loading and the generate call to copy.
- premises (stop and report if any is false):
  - `git grep -n "pub fn set_read_rule" -- proxima-model-interop/src/generate/load_model.rs` prints 1 line;
  - `git grep -n "clear_prompt_cache" -- proxima-model-interop/src/generate/pregather.rs` prints at least 1 line.
- why this card exists: the read rule is a closure, so no key can name it (a cache key is a proved name, never an address). Two requests under the same read spec but different rules store different rows, and the key from FT16.1 cannot tell them apart. The rows a rule shaped are dropped when the rule changes, which is the answer the model already takes for the sidecar and the summarizer.
- change:
  1. `proxima-model-interop/src/generate/load_model.rs`: the first statement of `set_read_rule` becomes `self.clear_prompt_cache();`. Its doc gains one sentence: "Entries stored before it are dropped: their rows were read through another rule, and a rule is a closure with no name the cache key could hold."
  2. `proxima-model-interop/src/generate/mod.rs`: add `#[cfg(test)] mod read_rule_cache_tests;` after the `kv_seal_tests` declaration.
- test: add `proxima-model-interop/src/generate/read_rule_cache_tests.rs` (new; `use super::chunked_prefill_tests::{config, gemma4_checkpoint, prompt_of};`; load the synthetic checkpoint as the FT4.5 test does, with `let mut model`), two tests, each calling `generate_with_serving_config(&prompt_of(40, '3'), 2, config(0))` as FT4.5's test does (the prompt cache is on by default):
  - `set_read_rule_drops_entries_stored_before_it`: after the generate call `prompt_cache_bytes() > 0`; after `model.set_read_rule(|_layer, _rows, _skip| {})` it is `0`;
  - `a_read_rule_set_before_any_request_keeps_the_entry_it_builds`: set the rule first, then generate: `prompt_cache_bytes() > 0` (the clear happens when the rule is set, not at every request).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_16_2 cargo nextest run -p proxima-model-interop --features std -E 'test(/set_read_rule_drops_entries_stored_before_it|a_read_rule_set_before_any_request_keeps_the_entry_it_builds/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/load_model.rs`, `proxima-model-interop/src/generate/mod.rs`, `proxima-model-interop/src/generate/read_rule_cache_tests.rs`
- commit: `fix(interop): drop cached entries when the read rule changes`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: put the rule or its address in the key; clear the cache on any call but the setter; add a field to `LoadedModel`.
- gpu: none

### 16.3 Refuse an idle step list whose drafts cannot fit the cache

- id: FT16.3
- needs: FT10.5, FT10.6, FT4.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/serving.rs::PromptCacheConfig::follow_up_branches (~line 530 at e4cf9beb)`: its doc says every draft counts against `max_entries`, and `::is_enabled (~line 596)`; nothing checks it today: `git grep -n max_entries main -- proxima-model-interop/src/generate/prewarm_follow_up.rs proxima-model-interop/src/generate/prewarm.rs` prints nothing, and the draft loop reads only `follow_up_branches`;
  - `proxima-model-interop/src/generate/prewarm_follow_up.rs::IdleStep`, `::DraftStep` (FT10.4): the list and its element;
  - `proxima-model-interop/src/generate/prewarm.rs::LoadedModel::prewarm_queued (~line 336 at e4cf9beb)`, `::run_pending_prewarm (~line 317)` and `::with_prewarm_worker_observed` (FT10.6; `with_prewarm_worker` at ~line 393 on main): `prewarm_queued` resolves the stored list or the default list (FT10.5), and the two public entries each compute `effective` from the caller's config before any job runs;
  - `proxima-model-interop/src/generate/prompt_cache.rs::PromptCache::eviction_victim (~line 820)`: an unused follow-up branch is evicted first, so a pass whose drafts fill the cap evicts its own earlier drafts when the answer entry is stored back.
- premises (stop and report if any is false):
  - `git grep -n "default_list(" -- proxima-model-interop/src/generate/prewarm.rs` prints 1 line, inside `prewarm_queued`;
  - `git grep -n "fn with_prewarm_worker_observed" -- proxima-model-interop/src/generate/prewarm.rs` prints 1 line.
- change:
  1. `proxima-model-interop/src/generate/prewarm_follow_up.rs`: add `pub(super) fn idle_steps_fit(steps: &[IdleStep], config: &PromptCacheConfig) -> Result<(), InteropError>`. Body: `drafted` is the sum over `steps` of `u64::from(draft.branches)` for `IdleStep::Draft(draft)` and `0` for `IdleStep::Prewarm` (an exhaustive match, no wildcard); return `Ok(())` when `!config.is_enabled() || drafted == 0 || drafted < u64::from(config.max_entries)`; otherwise `Err(InteropError::UnsupportedServingConfig(format!("idle steps draft {drafted} follow-up branches, but the prompt cache keeps {} entries and the answer entry needs one of them", config.max_entries)))`. Doc names the primitives it composes, `IdleStep` and `PromptCacheConfig::max_entries`, and says why: the drafts and the answer entry share the cap, and a draft that does not fit evicts an earlier one.
  2. `proxima-model-interop/src/generate/prewarm.rs`: add private `fn resolved_idle_steps(&self, config: &PromptCacheConfig) -> Result<Vec<IdleStep>, InteropError>`: the stored list cloned, or `IdleStep::default_list(config)` when it is empty (the expression `prewarm_queued` holds today), then `idle_steps_fit(&steps, config)?`, then `Ok(steps)`. In `prewarm_queued` replace the inline resolution with `let steps = self.resolved_idle_steps(&effective.prompt_cache)?;`. In `run_pending_prewarm` and in `with_prewarm_worker_observed`, add `self.resolved_idle_steps(&effective.prompt_cache)?;` directly after `effective` is computed, so a list that cannot fit is refused at the first call that carries a config, before any request is served.
  3. `proxima-model-interop/src/generate/mod.rs`: add `#[cfg(test)] mod idle_steps_fit_tests;` after the `kv_seal_tests` declaration.
- test: five tests.
  - in the `tests` module of `prewarm_follow_up.rs`:
    - `idle_steps_fit_accepts_drafts_that_leave_room_for_the_answer_entry`: with `max_entries = 4`, `[Prewarm, Draft(3 branches)]` is `Ok(())`, `[Prewarm]` is `Ok(())`, and `IdleStep::default_list(&PromptCacheConfig::standard())` (zero drafts) is `Ok(())`;
    - `idle_steps_fit_refuses_drafts_that_would_evict_each_other`: with `max_entries = 4`, `[Draft(4 branches)]` is `Err(InteropError::UnsupportedServingConfig(message))` where `message.contains("4 follow-up branches")` and `message.contains("keeps 4 entries")`; `[Draft(2), Draft(2)]` is the same refusal; with `max_entries = 5` the same two lists are `Ok(())`;
    - `idle_steps_fit_ignores_a_disabled_cache`: `PromptCacheConfig::off()` with `[Draft(9 branches)]` is `Ok(())`.
    Draft elements are built `DraftStep { branches, max_tokens: 48, temperature_milli: 800, lead: Vec::new(), keep: false }`.
  - in the new `generate/idle_steps_fit_tests.rs` (`use super::chunked_prefill_tests::{config, gemma4_checkpoint};`; synthetic gemma4 checkpoint loaded as the FT4.5 test does; a local `fn draft_of(branches: u32) -> DraftStep` building the element above; `serving = ServingConfig { prompt_cache: PromptCacheConfig { max_entries: 8, ..PromptCacheConfig::standard() }, ..config(0) }`):
    - `run_pending_prewarm_refuses_an_idle_list_that_cannot_fit`: after `model.set_idle_schedule(&[IdleStep::Draft(draft_of(8))])`, `run_pending_prewarm(&serving)` is `Err(InteropError::UnsupportedServingConfig(message))` with `message.contains("8 follow-up branches")`; after `set_idle_schedule(&[IdleStep::Draft(draft_of(7))])` it is `Ok(None)` (nothing queued); after `set_idle_schedule(&[])` it is `Ok(None)`;
    - `with_prewarm_worker_refuses_an_idle_list_that_cannot_fit`: with the 8-branch list set, `with_prewarm_worker(&serving, || ())` is `Err(InteropError::UnsupportedServingConfig(_))`; with the 7-branch list it is `Ok(())`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_16_3 cargo nextest run -p proxima-model-interop --features std -E 'test(/idle_steps_fit_|run_pending_prewarm_refuses_an_idle_list|with_prewarm_worker_refuses_an_idle_list/)'`
- expect: `5 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`; `cargo check -p proxima-model-interop --features std,metal --examples` (the benches set `follow_up_branches: 3` with `max_entries: 8`, which fits)
- stage: `proxima-model-interop/src/generate/prewarm_follow_up.rs`, `proxima-model-interop/src/generate/prewarm.rs`, `proxima-model-interop/src/generate/mod.rs`, `proxima-model-interop/src/generate/idle_steps_fit_tests.rs`
- commit: `feat(prewarm): refuse idle drafts that cannot fit the cache`
- done when: the expect line printed, clippy and the examples check clean, `git diff --cached --stat` equals the stage list (three source files plus the new test file), and the commit landed with that message
- do not: refuse inside `set_idle_schedule` (the cap is part of the request config, which a setter never sees); add a type or an error variant; add a bytes bound (ledger C item 4); change what a valid list does.
- gpu: none

### 16.4 Prove a summary record holds finer bounds than its seal block

- id: FT16.4
- needs: FT4.6
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/kv_seal_tests.rs::key_minmax` and the test `key_minmax_summaries_equal_an_independent_fold_of_the_decoded_rows` (FT4.6): the summarizer, the setup (synthetic checkpoint, `prompt_of(40, '3')`, block 8, horizon 4) and the independent fold this card reuses;
  - `proxima-model-interop/src/generate/residency_caches.rs::BlockSummarizer` and `LayerCache::summarize_sealed` (FT4.4): the record is a `Vec<f32>` of any length, one per sealed block, and the slot passes the summarizer exactly that block's rows;
  - `pipeline-as-data/research.md`, row A-3 and catalog change CC-3: a selection block should be able to differ from the seal block. The seal card forbids a second block size (FT4.2, "do not"), so this card shows the grain is the summarizer's.
- change: none to non-test code. This is a proof card: the grain of selection is a property of the record a summarizer returns, so no config field and no library item is needed.
- test: in `kv_seal_tests.rs` add the test-local summarizer
  ```rust
  fn half_block_minmax(k_even: &[f32], k_odd: &[f32], _value: &[f32], even_odd_row: usize) -> Vec<f32> {
      let half = k_even.len() / 2;
      [
          key_minmax(&k_even[..half], &k_odd[..half], &[], even_odd_row),
          key_minmax(&k_even[half..], &k_odd[half..], &[], even_odd_row),
      ]
      .concat()
  }
  ```
  and two tests, each loading the synthetic checkpoint with `.with_block_summarizer(half_block_minmax)` and running `prefill_with` on `prompt_of(40, '3')` under the seal config of FT4.3 (block 8, horizon 4). With `rows = state.len()`, the one full-attention layer has `even_odd_row = k_even.len() / rows`, and `sealed = seal_target(rows, 8, 4) / 8` is at least 3:
  - `a_summary_record_holds_one_bound_pair_per_half_block`: `block_summaries.len() == sealed`; every record has `8 * even_odd_row` values; for every block `b`, the first `4 * even_odd_row` values equal, bit for bit, an independent per-slot fold (written as in FT4.6's second fold) over rows `8b .. 8b + 4` of the concatenated key `[even row, odd row]`, and the last `4 * even_odd_row` values equal the same fold over rows `8b + 4 .. 8b + 8`; for at least one block the two halves differ (the halves carry information the whole block hides);
  - `half_block_bounds_combine_into_the_whole_block_bounds`: for every block, taking the per-slot minimum of the two halves' low values and the per-slot maximum of their high values equals, bit for bit, the independent fold over all 8 rows of the block (the finer grain refines the coarse one and never disagrees with it).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_16_4 cargo nextest run -p proxima-model-interop --features std -E 'test(/a_summary_record_holds_one_bound_pair_per_half_block|half_block_bounds_combine_into_the_whole_block_bounds/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: `proxima-model-interop/src/generate/kv_seal_tests.rs`
- commit: `test(interop): hold finer than block bounds in a summary record`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` touches only `generate/kv_seal_tests.rs`, and the commit landed with that message
- do not: change non-test code; add a `summary_tokens` field or a second block size; move `half_block_minmax` out of the test module.
- gpu: none

### 16.5 Prove a probe-driven stop through the token callback on gemma4 E2B

- id: FT16.5
- needs: FT11.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::decode_until_stop_or_budget (~line 3433 at e4cf9beb)`: the callback returns `ControlFlow`; a `Break` on a `Phase::Token` event stops generation after that token has been pushed (the token is in the result), while a `Break` on the step-0 `Phase::Prefill` event stops before any token;
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity (~line 668)`, `::llama_cases (~line 617)`, `::first_divergence (~line 661)`, `LLAMA_GENERATED_TOKENS (~line 609)`: the loader, the recorded ids and the divergence policy this card reuses;
  - the test `readout_entry_events_carry_finite_readouts_gemma4_e2b` (FT11.3, same file): the readouts configuration and the event collection;
  - `pipeline-as-data/research.md` rows C25 (force or forbid the end of thinking) and C45 (a probe as a reasoning stop signal) and catalog change CC-6: the stop proposal is a signal that reaches the existing stop, not a new stage.
- premises (stop and report if any is false):
  - `git grep -n "pub readout" -- proxima-model-interop/src/generate/residency_caches.rs` prints 1 line and `git grep -n "pub readouts: bool" -- proxima-model-interop/src/serving.rs` prints 1 line;
  - the first line of `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/llama_ids.json` is `[` and the file holds records with `prompt_ids` and `generated_ids` (read at e4cf9beb: 3 records).
- change: none to non-test code. The H15 hook (what ends a generation) is the callback's `ControlFlow` that `generate_from_ids` takes and the readouts of FT11.3; a probe that proposes a stop is a closure over them, a test of about 40 lines.
- test: add `stop_signal_ends_generation_at_the_least_certain_token_gemma4_e2b` in `tests/arch_data_baseline.rs`. Load `GEMMA4_E2B` once as `llama_parity` does. `config = ServingConfig { prompt_cache: PromptCacheConfig::off(), decode: DecodeConfig { readouts: true, ..DecodeConfig::default() }, ..ServingConfig::default() }`. A local closure `run(case, threshold) -> (Vec<u32>, Vec<f32>)` calls `model.generate_from_ids(&case.prompt_ids, LLAMA_GENERATED_TOKENS, &config, ..)` with a callback that, on `Phase::Token` events only, pushes `event.readout.expect("readouts are on").top2_margin` to a list and returns `ControlFlow::Break(())` when that margin is less than `threshold`, otherwise `ControlFlow::Continue(())`; it returns the generated ids and the list. Then, over the records of `llama_cases(&GEMMA4_E2B)` in file order:
  1. `baseline = run(case, f32::NEG_INFINITY)` (the control: a threshold no margin is below never stops). Assert `baseline_margins.len() == baseline_ids.len()`, and `first_divergence(&case.generated_ids, &baseline_ids[..compared])` is `None` for `compared = min(len)` with `compared > 0`, so the run the stop is cut from is the recorded llama run.
  2. `sorted` is the margins sorted with `f32::total_cmp`. The record qualifies when `sorted.len() >= 8`, `sorted[1] - sorted[0] > 1e-3`, and the first index `stop_index` with `baseline_margins[stop_index] == sorted[0]` has `stop_index + 1 < baseline_ids.len()`. A record that does not qualify is skipped and noted in a list.
  3. For the first qualifying record, `threshold = (sorted[0] + sorted[1]) / 2.0` and `stopped = run(case, threshold)`: assert `stopped_ids == baseline_ids[..=stop_index]`, `stopped_margins.len() == stop_index + 1`, `stopped_ids.len() < baseline_ids.len()`, and, when `stop_index < case.generated_ids.len()`, `stopped_ids == case.generated_ids[..=stop_index]`. Return.
  4. If no record qualified, `panic!` with the list of skipped records (their lengths and their two smallest margins): zero qualifying records is RED, not a skip.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_16_5 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_16_5 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/stop_signal_ends_generation_at_the_least_certain_token_gemma4_e2b/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_16_5/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(generate): stop generation from a token readout probe`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`, and the commit landed with that message
- do not: add a function, field or type to `src`; query Ollama or llama-server; compare against a run of proxima that was not first held to the recorded llama ids; load a second model; break on the step-0 `Phase::Prefill` event.
- gpu: one run (`-j 1`, one model-loading process), waiting for a quiet box (the peer-gate check above)

## slice exit

Run, in order, with `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_16_exit` (remove it after). Every count is asserted; `0 passed` is red.

1. `cargo nextest run -p proxima-model-interop --features std -E 'test(/a_request_differing_in_a_row_affecting_config_field_misses|an_entry_built_under_the_operand_read_serves_only_operand_requests|a_request_differing_only_in_a_row_independent_field_shares_the_entry/)'`: expect `21 passed` (the 19 listed on main plus the new case and the new test of card FT16.1); `0 passed` is red.
2. `cargo nextest run -p proxima-model-interop --features std -E 'test(/set_read_rule_drops_entries_stored_before_it|a_read_rule_set_before_any_request_keeps_the_entry_it_builds/)'`: expect `2 passed`.
3. `cargo nextest run -p proxima-model-interop --features std -E 'test(/idle_steps_fit_|run_pending_prewarm_refuses_an_idle_list|with_prewarm_worker_refuses_an_idle_list/)'`: expect `5 passed`.
4. `cargo nextest run -p proxima-model-interop --features std -E 'test(/a_summary_record_holds_one_bound_pair_per_half_block|half_block_bounds_combine_into_the_whole_block_bounds/)'`: expect `2 passed`.
5. The model run of card FT16.5, only when `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` prints nothing: expect `1 passed`.
6. No library type is named after a technique: `git grep -nP '\b(struct|enum|trait)\s+\w*(Resa|ReSA|Cartridge|SleepTime|CacheBlend|Milvus|SealedSegment|GrowingSegment|Cascadia|Conformal|Ucci|UCCI|AoSpec|Sherlock|SpeculativeMacro|LmCache|VAttention)' -- proxima-model-interop/src proxima-tensor/src proxima-core/src omega/src | wc -l`: expect `0`. A nonzero count is not cut: print the lines without `wc -l` and report each one.
7. The control for step 6, which must be nonzero or the pattern form matches nothing: `git grep -nP '\b(struct|enum|trait)\s+\w*(ServingConfig|PromptCache|ServingState)' -- proxima-model-interop/src proxima-core/src | wc -l`: expect a number of at least `1`; print it. `0` here means the pattern form is blind, and step 6 proves nothing.
8. No qwen in the files this slice created: `git grep -n -i qwen -- proxima-model-interop/src/generate/read_rule_cache_tests.rs proxima-model-interop/src/generate/idle_steps_fit_tests.rs | wc -l`: expect `0`.
9. `cargo clippy -p proxima-model-interop --features std,metal --all-targets`: clean; `cargo check -p proxima-model-interop --no-default-features`: passes.
10. Remove `/private/tmp/cargo_target_ft_16_*`; logs stay under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_16_5/`.

The slice is complete when steps 1 to 9 printed the counts above. The hook ledgers above are the record of which gap each other card owns and which parts are not built.
