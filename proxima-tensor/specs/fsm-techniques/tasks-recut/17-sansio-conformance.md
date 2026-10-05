# slice 17 (re-cut): the sans-IO conformance suite, the end-state gate

anchors read at main e4cf9beb (full sha e4cf9beb8342a80447342f0813ada9c84fc3a519) with `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`. A location written `path::symbol (~line N at e4cf9beb)` exists on main and the executor re-locates it by symbol. A location written `path::symbol (created by FT<a>.<b>)` does not exist on main (`git ls-tree -r main --name-only | grep -E 'proxima-core/src/serving'` prints nothing) and is created by the named card; every card that reads one carries a premise check and stops when the symbol is absent. Paths are relative to the proxima repo root. "SPECDIR" is `proxima-tensor/specs/fsm-techniques/`; every `SPECDIR/...` anchor is absent on main until the spec directory is committed to main, which CARDS.md "Where the spec lives" requires before the first card executes (`git ls-tree -r main --name-only | grep fsm-techniques` prints nothing before that commit and lists the directory after it); the owner's commit of the directory is the precondition of every card in this file.

Rules: CARDS.md is binding. Every card uses `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_<n>` (n is the card number) and removes it when done. Logs go under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards17/`.

Governing direction (owner, 2026-10-04): "I did not ask the original agent to implement the techniques. I asked the original agent to build the hooks so they could be implemented. if the hooks are mostly programmable, the surface becomes very hackable and future techniques can be quickly vetted." This slice builds no hook and no technique. It is the test-only gate that proves, with no GPU, no file or network IO and no weights, that the control flow of every hook the earlier slices landed is correct. A technique appears here only as configuration values or as a test-local pipe of at most about 40 lines, never in a library file. The non-test code this slice adds is exactly three items and none is a hook or a technique: the example (the no-IO guard, FT17.35); the `sansio-script` Cargo feature with `proxima-core/src/serving_state/scripted.rs` (FT17.8: a scripted-readout function behind `cfg(any(test, feature = "sansio-script"))`, built as non-test code by the `alloc,sansio-script` check); and the visibility change of `rule_victim` to `pub(super)` in `proxima-model-interop/src/generate/prompt_cache.rs` (FT17.30). Every other file this slice edits is a test file, `Cargo.toml` or a spec file.

Test models (owner, 2026-10-04): gemma4 for dense or MoE, granite for MoE. This file loads no model, so no model arm exists; the numerics of gemma4 and granite are proved by the oracle cards of their own slices. No qwen of any kind appears in any card of this file.

## old to new id map

The earlier cut is `tasks/17-sansio-conformance.md`. Card numbers 17.3 to 17.6 of that cut were already withdrawn.

| old card | new card | verdict |
|---|---|---|
| 17.1 | FT17.1 | kept; the rectify proposal became the replay proposal (`replay_range`) |
| 17.2 | none | dropped (see "dropped") |
| 17.7 | FT17.3 | kept; the rectify overwrite became the replay rewrite and the `accept_replay` placement |
| 17.8 | FT17.4 | kept; the proposer-index trace became the ordered rule list trace |
| 17.9 | FT17.5 | kept; rewritten over the idle step list and the request gate |
| none | FT17.6 | new derivation: eviction by an ordered rule list (the earlier cut cited a least-recently-used trace that the tier slice's re-cut replaced) |
| none | FT17.7 | new derivation: one tier settles by vote (the earlier cut cited five judge traces; the vote rule is now a test-local function, because the cascade slice keeps no vote rule in a library) |
| 17.10 | FT17.8 | kept |
| 17.11 | FT17.9 | kept; the runner now draws from a fixed seed |
| 17.12 | FT17.10 | kept; 18 legal rows (the replay transition is one more) |
| 17.13 | FT17.11 | kept; 49 refused rows (40 illegal pairs and 9 guards) |
| 17.14 | FT17.12 | kept |
| 17.15 | none | dropped (see "dropped") |
| 17.16 | FT17.14 | kept |
| 17.17 | FT17.15 | kept; the accept rule is `AcceptRule`, no `config` gate |
| 17.18 | FT17.16 | kept; same |
| 17.19 | FT17.17 | kept; same |
| 17.20 | FT17.18 | kept |
| 17.21 | FT17.19 | kept; rewritten over `replay_range` |
| 17.22 | FT17.34 | kept; moved to `proxima-model-interop` and rewritten over the idle step list |
| 17.23 | FT17.20 | kept |
| 17.24 | FT17.21 | kept |
| 17.25 | FT17.22 | kept |
| 17.26 | FT17.23 | kept |
| 17.27 | FT17.24 | kept |
| 17.28 | FT17.25 | kept |
| 17.29 | FT17.26 | kept |
| 17.30 | FT17.27 | kept |
| 17.31 | FT17.28 | kept |
| 17.32 | FT17.29 | kept; the scripted dev-dependency is gone |
| 17.33 | FT17.30 | kept; rewritten over the eviction rule list |
| 17.34 | FT17.31, FT17.32 | recut: FT17.31 the assemble order rows, FT17.32 the proof that selective recompute runs through the assemble place |
| 17.35 | FT17.33 | kept; rewritten over the block read decisions in `proxima-core` |
| 17.36 | FT17.35 | recut: the shell script became a Rust example |
| 17.37 | FT17.36 | kept; counts re-derived |

Counts: 39 card ids, 37 cards in this file; 29 kept (FT17.1, FT17.3 and FT17.6 each narrowed to one worked example), 2 recut, 6 new (FT17.6, FT17.7, FT17.32, and FT17.37 to FT17.39 which come from splitting the derivation cards), and 2 dropped (FT17.2 and FT17.13; their ids stay unused, as slice 0 leaves FT0.38 unused, so no other card is renumbered).

## dropped

Two cards, then parts of kept cards that were removed, each with its reason:
- FT17.2 (the tree shape and visibility derivation, old 17.2) and FT17.13 (the shape place test, old 17.15): the shape functions `Shape`, `chain_shape` and `tree_shape` belong to the host shape pipes slice of the decode-as-data spec (`proxima-tensor/specs/decode-as-data/`, untracked in the proxima-windows checkout and absent from main: `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:proxima-tensor/specs/decode-as-data/SPEC.md` fails). Its `TASKS.md` lists that slice as one line with no card file, and no file in `tasks-recut/` other than this one names `tree_shape` or `chain_shape`, so no card creates them. A test over a symbol no card creates cannot run, and a test-local copy of the arrays would test itself. The shape place is proved by that slice's own tests, not by this suite. Effect: 13 `proxima-core` tests (transitions 2, places 5, properties 6) instead of 14, and FT17.36 edits the spec to match;
- the assemble rows `Load`, `Blend`, `BlendNotLast`, `BlendWithoutRows` (old 17.34): the assemble decision now knows only `Prefix` and `Shift` (FT8.2); a stage the library does not define composes by list order and is proved by FT8.8 and by FT17.32;
- the threshold, classifier and isotonic judge rows and the router and chain rows (old 17.23, 17.30): the cascade slice re-cut drops `Agreement`, `settle_votes` and the `Judge` enum from the library (`tasks-recut/12-cascade.md` drops FT12.1 and FT12.2 and lists them under designs abandoned); the vote rule is a technique, so FT17.20 holds one vote case set over a test-local function and no other judge has a row;
- the scripted readout use from `proxima-model-interop` and its dev-dependency (old 17.32): an interop layer cache holds `f32` planes, not entries, so a readout table has no input there;
- the tensor-side top-fraction operator (old 17.35): the host half of the read hook is test-local block read arithmetic (the keep count and the row count, which `tasks-recut/06-read-sets.md` drops as FT6.1 and FT6.2) plus a test-local selection pipe;
- `sansio_place_schedule` in `proxima-core` (old 17.22): no pure schedule decision exists; see "what the recut changes".

## what the recut changes, in English

- The enter place. The earlier cut tested a proposer index over `[rectify, ngram]`. The enter hook is now a configured list of rules (`EnterRule::{DraftNonempty, Replay { max_rows, decide }}`, where `decide` is a function the caller supplies; the periodic cadence is a test-local function in this file, never a library item) and a pure `replay_range` that names the committed rows a step re-encodes first. The place test covers that list, and the state machine gains one transition for it (`accept_replay`), so the legal table grows from 17 to 18 rows and the illegal pairs from 35 to 40.
- The commit place. A replay commits by rewinding the cache to the start of the range and appending the re-encoded rows. The test checks that only rows inside the range change, the length does not, and `accept_replay` resumes from the waiting entry.
- The accept place. The accept decision is the data rule `AcceptRule` (`similarity_floor`, `min_run`, `max_rows`). Equality, a verifier and an anchor run are three settings of it, written as struct literals in the test.
- The schedule place. Slice 10 dropped the pure `next` function: the next idle job is `steps.get(completed)` behind one atomic load of the request gate. So the place test is model-free and runs in `proxima-model-interop` over `IdleStep` and `PrewarmGate::run_idle`. This moves one test from `proxima-core` to `proxima-model-interop`, and the shape place test is dropped (see "dropped"): 13 and 5 tests instead of 15 and 4. Card FT17.36 edits the spec and task list to match.
- The tier place. Eviction is an ordered rule list `[Branch, Oldest]` (the default reproduces today's rule), and the disk file format round-trips in memory. There is no `Tier` enum, no `demotion_target` and no `lookup_tier` any more.
- The settle place. The vote rule is a test-local function over a test-local `Agreement` record (no library holds it); the place test and the property both drive it, and the property adds a test-local tier loop.
- The guard. The shell script became a Rust example, because the owner's rule is Rust only, and its source lives outside the guarded pathspec so it cannot flag itself.

## designs abandoned

- A `ScheduleNext` function in `proxima-core` so that the schedule place stays in core: its body would be `steps.get(completed)` behind a boolean, and the call site both ways is the same line. Abandoned for a test over the real gate.
- A scripted backend with a readout struct, a trait, or a `ScriptedBackend` struct with `new` and `readouts`: the readout is the tuple `(next, logprob, margin)`; a trait would have one impl; the struct is the plain function's table and function-pointer arguments held between calls (see the design section), and it made one card add more than one public item that later cards consume.
- Gating the accept rows behind `config`: `AcceptRule` has three public scalar fields, so a struct literal is the same value a TOML table loads; slice 1's own tests prove the load.
- A `proptest!` macro and the default runner: the macro reads its case count from the environment, and the default runner seeds from the clock, so a control could pass by luck. The runner here fixes the seed.

## cross-file contract (a card whose premise is false stops and reports)

| symbol | made by | read by |
|---|---|---|
| `proxima_core::serving_state::{ServingState, ServingFsmError}`, `ServingState::{start, advance_prefill, advance_decode, enter_verify, accept, resume, rollback, finish}` | FT1.2, FT1.3 | FT17.10 on |
| `ServingState::accept_rows(accepted, &row_tokens, row_caches)` | FT1.7 | FT17.10, FT17.11, FT17.14 |
| `proxima_core::serving_state::action_fixture::{Action, run_sequential, run_speculative}` (test-only, `pub(crate)`) | FT1.8, FT1.9 | FT17.15 to FT17.17 |
| `proxima_core::accept_rule::AcceptRule { similarity_floor: f32, min_run: u16, max_rows: u16 }` and `AcceptRule::accepted_rows(&self, draft, choices, similarity: impl Fn(&Entry, &Entry) -> f32) -> usize` | FT1.10 | FT17.15 to FT17.17, FT17.23, FT17.24 |
| `proxima_core::serving_state::enter::EnterRule { DraftNonempty, Replay { max_rows: u32, decide: fn(usize, usize) -> Option<usize> } }` and `replay_range(rules, committed_len, last_replay_end) -> Option<Range<usize>>` | FT7.0 (`EnterRule`), FT7.1 (`replay_range`) | FT17.12, FT17.18, FT17.19, FT17.25 |
| `Verify { draft, last, snapshot, cache }` (gains `last`) and `ServingState::accept_replay(row_caches)` | FT7.2 | FT17.10, FT17.11, FT17.18 |
| `Prefill { positions, start_position, cache }` and `ServingState::start_at(positions, start_position, cache)` | FT8.1 | FT17.10 |
| `proxima_core::serving_state::assemble::{AssembleKind, AssembleDecision, AssembleOrderError, assemble_decision}` | FT8.2 | FT17.31 |
| `proxima_core::kv_decision::seal_target` | FT4.0 | FT17.26, FT17.29, FT17.33 |
| `proxima_core::kv_decision::sealed_blocks(previous_sealed_end, cached_rows, block_tokens, horizon_rows) -> Range<usize>` | FT4.20 (FT4.0 adds only `seal_target`) | FT17.26 |
| the field `LayerCache::sealed_end` | FT4.1 | FT17.29 |
| `InteropError::RewindIntoSealed { keep_positions, sealed_end }` | FT4.21 | FT17.29 |
| `LayerCache::try_truncate(keep_positions, even_odd_row, v_row)` | FT4.22 | FT17.29 |
| `LayerCache::seal(even_odd_row, block_tokens, horizon_rows) -> Range<usize>` | FT4.16 (FT4.3 adds only `seal_attention_layers`) | FT17.29 |
| `LayerCache::{new, append}` | exist on main | FT17.29 |
| `proxima_model_interop::block_file::{encoded_len, BlockFileHeader, BlockFileLayer}` | FT5.1 | FT17.30 |
| `block_file::encode_block`, `InteropError::BlockFileMalformed { reason }` | FT5.17 | FT17.30 |
| `block_file::decode_block`, `BlockFileView` | FT5.2 | FT17.30 |
| `BlockFileView::plane_f32` | FT5.18 | FT17.30 |
| `generate::prompt_cache::{EvictionRule, rule_victim}` (`rule_victim` is private until FT17.30 widens it) | FT5.7 | FT17.30 |
| `generate::prewarm_follow_up::{IdleStep, DraftStep}` with `IdleStep::default_list`, `DraftStep { branches, max_tokens, temperature_milli, lead, keep }` | FT10.1 to FT10.4 | FT17.34 |
| `generate::prewarm_gate::PrewarmGate::{new, enter_request, run_idle}` | FT10.7 | FT17.34 |
none: the vote rule, the blend keep count and mask index, and the block read counts are test-local functions in `sansio_tests.rs` (FT17.20, FT17.32, FT17.33); `tasks-recut/12-cascade.md`, `09-cacheblend.md` and `06-read-sets.md` drop the library forms | none | FT17.20, FT17.27, FT17.32, FT17.33 |

## common to every card (not repeated)

- gpu: none for every card. No model is loaded and no Metal feature is built;
- `proxima-core` cards: also green is `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets` clean and `cargo check -p proxima-core --no-default-features --features alloc`;
- `proxima-model-interop` cards: also green is `cargo clippy -p proxima-model-interop --features std --all-targets` clean;
- everything a card has the executor write (code, doc comments, assert messages, error text, derivation sections, commit messages) is plain English with no slice, stage, card, AC, R, W, FT or D id and no pointer into the spec files; derivation sections get English titles; each place test carries one doc comment naming its trace in plain English;
- test modules that use `unwrap` or `expect` carry `#[allow(clippy::unwrap_used, clippy::expect_used)]` with the one lowercase comment line `// a failed transition in a test is a broken test; expect names it` (precedent: `proxima-core/src/batch.rs` tests);
- no sleeps, no `unwrap` outside tests, no comments except a lowercase why, no variable name of two characters or fewer, no `cargo fmt` or rustfmt on a `src` file;
- one card is one commit: stage only the `stage:` list, commit with the exact `commit:` subject, no attribution trailer.

## design

### the scripted readouts, and why they are one function and not a trait or a struct

File `proxima-core/src/serving_state/scripted.rs`, declared `#[cfg(any(test, feature = "sansio-script"))] pub mod scripted;` (feature `sansio-script = ["alloc"]`, default off). `alloc` and `core` only. A row's readout is the tuple `(next, logprob, top2_margin)`; no readout type exists.

```rust
pub enum ScriptError {
    LengthMismatch { entries: usize, positions: usize },
    MissingRow { position: usize },
    CacheBehindPositions { cache_rows: usize, first_position: usize },
}

pub fn scripted_readouts<Entry: Ord + Clone, Cache>(
    table: &BTreeMap<(usize, Entry), (Entry, f32, f32)>,
    rows_of: fn(&Cache) -> usize,
    entries: &[Entry],
    positions: &[usize],
    cache: &Cache,
) -> Result<Vec<(Entry, f32, f32)>, ScriptError>;
```

The public surface is these two items and no more: `scripted_readouts` is the only one another card consumes, and no other card names `ScriptError` (they call `expect` on the result). The table is the caller's `BTreeMap`, so no table type exists. `scripted_readouts` is pure (no clock, no interior mutability, no IO). Row `i` is `table[(positions[i], entries[i])]`. Checks in this order: lengths (`LengthMismatch`), then `rows_of(cache) < positions[0]` (`CacheBehindPositions`; a verify pass scored against a cache that lacks rows below its first position is a control-flow bug, and this is the only place the suite sees it), then each row (`MissingRow`). `rows_of(cache) >= positions[0]` is accepted, which covers a replay pass whose positions lie inside rows the cache already holds. `ScriptError` derives `Debug, PartialEq, Eq` and `thiserror::Error` with lowercase messages. The table is position-keyed and Markov in the input entry, so a K-row verify pass returns what K single-row decodes would, which is what makes the equality property meaningful.

Call it three ways. Pipe form: `block_on(backend.call((entries.to_vec(), positions.to_vec(), cache.clone())))` needs `proxima-primitives`, which depends on `proxima-core` (`proxima-primitives/Cargo.toml:123`), so it cannot be a dependency here, plus an executor and three clones. Struct form: `ScriptedBackend::new(rows, rows_of).readouts(&entries, &positions, &cache)` would hold a map and a function pointer between calls and add a type, a constructor and a method, three public items that later cards consume. Plain form: `scripted_readouts(&table, rows_of, &entries, &positions, &cache)`. The struct form and the plain form return the same `Vec` or the same error, with no state kept between calls, so the struct is a relocation of two arguments and is abandoned. The plain form is the transform-form body as a free function: no new trait, no struct and no `Pipe` impl.

### the test cache

`LogCache { rows: Vec<u32> }` in `sansio_tests.rs`. `rows` are the entries consumed so far (the kv cache is the fold over the log). Placement after consuming `inputs[..=i]` from `pre` is `pre.rows ++ inputs[..=i]`. A `Decode` state's `last` is not yet in `cache.rows` (it is the next input); a verify pass over draft `d0..d(k-1)` consumes the inputs `[last, d0, ..., d(k-2)]` at positions `rows.len()..`, and its readout `i` is the prediction after input `i`.

### where each place test lives (the placement rule)

| place | test | crate | pure body |
|---|---|---|---|
| propose | `sansio_place_propose` | proxima-core | `ngram_draft` (test-local to `sansio_tests.rs`, FT17.12), `replay_range` |
| accept | `sansio_place_accept` | proxima-core | `accept`, `accept_rows`, `AcceptRule::accepted_rows` |
| commit | `sansio_place_commit` | proxima-core | `accept_rows` placement, `accept_replay` |
| enter | `sansio_place_enter` | proxima-core | `replay_range` |
| settle | `sansio_place_settle` | proxima-core | test-local `settle_votes` |
| seal | `sansio_place_seal` | proxima-model-interop | `seal_target` applied by `LayerCache` |
| place | `sansio_place_place` | proxima-model-interop | `rule_victim`, `encode_block`, `decode_block` |
| assemble | `sansio_place_assemble` | proxima-model-interop | `assemble_decision`, test-local `blend_keep_count` and `blend_mask_index` |
| read (host) | `sansio_place_read_host` | proxima-model-interop | `seal_target`, test-local `block_read_keep_count` and `block_read_row_count` |
| schedule | `sansio_place_schedule` | proxima-model-interop | `IdleStep` list behind `PrewarmGate::run_idle` |

### the state machine tables (after the move, `start_at` and the replay transition)

States: Prefill, Decode, Verify, Accept, Rollback, Done. Events: `advance_prefill`, `advance_decode`, `enter_verify`, `accept`, `accept_rows`, `accept_replay`, `resume`, `rollback` (eight methods that can be refused), plus `start`, `start_at` and `finish` (which cannot).

Legal rows, 18:

| # | state | event | next |
|---|---|---|---|
| 1 | none | `start` | Prefill with start position 0 |
| 2 | none | `start_at` | Prefill with the given start position |
| 3 | Prefill | `advance_prefill` | Decode |
| 4 | Decode | `advance_decode` | Decode |
| 5 | Decode | `enter_verify` | Verify |
| 6 | Verify | `accept`, every row equal | Accept |
| 7 | Verify | `accept`, first mismatch inside the draft | Rollback |
| 8 | Verify | `accept_rows`, count equals draft length | Accept |
| 9 | Verify | `accept_rows`, count below draft length | Rollback |
| 10 | Verify | `accept_replay` | Decode with the waiting entry |
| 11 | Accept | `resume` | Decode |
| 12 | Rollback | `rollback` | Decode |
| 13 to 18 | each of the six states | `finish` | Done |

Refused: 8 events against 6 states is 48 pairs, 8 legal (one per event), so 40 illegal pairs, each `Err(ServingFsmError::IllegalTransition { attempted: "<method>" })`. Guards, all from `Verify`, all `IllegalTransition` naming the called method:

| guard | calls | rows |
|---|---|---|
| draft empty | `accept(&[], vec![])`, `accept_rows(0, &[], vec![])`, `accept_replay(vec![])` | 3 |
| `row_tokens.len() != draft.len()` | `accept` and `accept_rows` | 2 |
| `row_caches.len() != draft.len()` | `accept`, `accept_rows`, `accept_replay` | 3 |
| `accepted > draft.len()` | `accept_rows(3, ..)` on a 2-draft | 1 |

Total refused rows: 40 + 9 = 49. Finding: a wrong state and bad lengths are the same variant, so a caller cannot tell them apart, and every consuming transition drops the state on `Err`; the tests `clone()` before each refused call. Decode-as-data's `refine` (a `Verify -> Verify` step) is in no card file; if it lands the tables gain events and FT17.10 and FT17.11 stop and report (with one more event: 9 x 6 = 54 pairs).

### the properties, proptest, exact

Harness for all six (FT17.9): `try_256` builds `TestRunner::new_with_rng(Config { cases: 256, failure_persistence: None, ..Config::default() }, TestRng::deterministic_rng(RngAlgorithm::ChaCha))` and counts body invocations; `run_256` requires `Ok` and asserts the count is exactly 256. `failure_persistence: None` because the default writes a `proptest-regressions` file on failure, which is file IO; a fixed seed so a failure reproduces on every run and a control cannot pass by luck. `proptest` is not a dev-dependency of `proxima-core` on main (`git grep -n proptest main -- proxima-core/Cargo.toml` prints nothing; the workspace has `proptest = "1.11"` at `Cargo.toml:208`): FT17.9 runs `cargo add --dev proptest -p proxima-core`.

| prop | test | strategy | exact property |
|---|---|---|---|
| 1 | `sansio_prop_log_append_only` | `vec(step, 0..48)`; `step` is `Decode`, `Verify { draft_len 1..5, accepted 0..=draft_len }` or `Replay { rows 1..8 }` | after every step `log_after.starts_with(log_before)`, except `Replay` where `log_after.len() == log_before.len()` and the first `len - rows` entries (rows clamped to len) are unchanged |
| 2 | `sansio_prop_rollback_restores` | `(prefix vec(0..6, 0..24), last 0..6, draft vec(0..6, 1..6), mismatch 0..draft.len())`, verifier choices equal to the draft before `mismatch` and different at it | after `accept` yields Rollback and `rollback()`: the pre-Verify rows are a prefix of the post-Rollback rows and `post.rows.len() == pre.rows.len() + 1 + mismatch` |
| 3 | `sansio_prop_exact_accept_equals_sequential` | `(policy 0..4, table vec(vec(0..6, 6), 64), prompt vec(0..6, 1..6), max_tokens 1..40, min_skip 1..4)`; policy 0 chain `accept`, 1 equality, 2 verifier cap, 3 anchor run, the last three through `accept_rows` and `AcceptRule::accepted_rows` | the speculative driver's first `max_tokens` generated entries equal the sequential driver's |
| 4 | `sansio_prop_terminates` | `(max_tokens 1..64, every_rows in {1, 2, 3, 4, 8, 15}, draft_flags vec(bool, 1..8))` | the driver reaches `Done`; iterations `<= max_tokens + max_tokens / every_rows` (a replay round commits nothing, every other round commits at least one entry) |
| 5 | `sansio_prop_sealed_immutable` | `(ops vec(op, 0..64), block_tokens in {4, 16, 32}, horizon_rows 0..4)`, `op` is `Append(1..8)` or `Rewind(0..40)` | with `sealed_end` the running maximum of `seal_target(rows, block_tokens, horizon_rows)`, a rewind to `keep < sealed_end` is refused and leaves the rows unchanged; after every op `rows[..sealed_end]` equals the copy taken when those rows first sealed; every block `sealed_blocks` yields lies at least `horizon_rows` rows behind the newest row |
| 6 | `sansio_prop_one_tier_settles` | `vec(bool, 1..7)`: tier `t` draws votes `[15, 1]` when its flag is true and `[8, 8]` when false; every tier but the last holds the test-local `Agreement { samples: 16, max_disagree: 4 }`; the last tier holds `Agreement { samples: 1, max_disagree: 0 }` and draws the single vote `[1]` whatever its flag | the answering tier is the first true flag among the non-last tiers, else the last tier; the generation counters are 1 for every tier up to and including it and 0 after |

### the no-IO guard

AC commands, verbatim, run by the gate card:
- `git ls-files proxima-core/src/serving_state.rs 'proxima-core/src/serving_state/*' | wc -l` must be at least 3, so that an empty pathspec cannot pass as 0;
- `git grep -nE 'std::(fs|net|io|env|thread|time)|File::|TcpStream|tokio|libc::|println!' -- proxima-core/src/serving_state.rs 'proxima-core/src/serving_state/*' | wc -l` must be 0;
- `cargo check -p proxima-core --no-default-features --features alloc,sansio-script --message-format=json > <log>.json` and then `grep -c '"kind":\["lib"\].*"name":"proxima_core".*"features":\[[^]]*"sansio-script"' <log>.json` must print `1`: cargo emits one compiler-artifact record for the library, and its feature list names `sansio-script` only when that feature was built. A check that failed, or that never built the library with the feature, prints `0`. Measured on the proxima-windows checkout with the `alloc` feature in place of `sansio-script` (the new feature does not exist yet): the same grep over the library record printed `1`, and the build-script record, which carries the same feature list, did not match because its kind is `custom-build`.

Both git commands see tracked files only; an untracked file hides from them. The example of FT17.35 prints the tracked count beside the count of `.rs` files on disk and fails when they differ.

---

## phase 1: worked derivations (docs, `worked-examples.md`)

Each card holds one example, derived by hand before any test code. Each goes under a section titled `## sans-IO control-flow traces` in `worked-examples.md` (add the heading at the end of the file when absent), under an English `###` title.

Every derivation card carries its assertion as lines. The `change` step lists the exact lines the section body holds, one derived fact per line, each starting `- `. The `validate` command cuts the section out of the file (`sed -n '/^### <title>$/,/^##/p'`) and counts how many of those exact lines it holds (`grep -cxF -e <line> ...`). The expected value is the number of lines, so a missing heading, an empty body, a changed value or a dropped line prints less than that number.

### 17.1 derive the replay proposal trace

- id: FT17.1
- needs: FT0.14 (creates `worked-examples.md`)
- budget: 20 min
- crate(s): none (docs)
- read first: `SPECDIR/tasks-recut/07-rectify.md` section "shared worked values" and cards FT7.0 and FT7.1 (the `Replay { max_rows, decide }` rule and the bounds `replay_range` applies to the decision's answer; the periodic cadence is only a decision this file writes test-locally)
- change:
  1. `SPECDIR/worked-examples.md`: add the heading `### replay proposal` and, under it, exactly these 5 lines (derived by hand: the committed log is `10 + p` at position p, the periodic rule is a caller-supplied decision that answers `replay_rows` once `committed_len - last_replay_end >= every_rows`, and `replay_range` accepts the answer when `0 < replay_rows <= max_rows` (here `max_rows = replay_rows`) and `replay_rows <= committed_len`, giving `committed_len - replay_rows..committed_len`):
     ```
     - committed log: the entry at position p is 10 + p, so positions 0 to 7 hold 10, 11, 12, 13, 14, 15, 16, 17
     - rule every 3 rows replaying 3 rows, last replay ended at 5, committed length 8: range 5..8, proposal 15, 16, 17 at positions 5, 6, 7
     - same rule at committed length 7: no proposal (2 rows since the last pass, 3 needed)
     - rule every 3 rows replaying 8 rows, committed length 8: range 0..8, proposal is the whole log
     - rule every 3 rows replaying 9 rows, committed length 8: no proposal (more rows than committed)
     ```
- test: the section holds all 5 listed lines verbatim; the assertion is the count of exact line matches inside the section, which is 5 only when every derived value is present
- validate: `sed -n '/^### replay proposal$/,/^##/p' proxima-tensor/specs/fsm-techniques/worked-examples.md | grep -cxF -e '- committed log: the entry at position p is 10 + p, so positions 0 to 7 hold 10, 11, 12, 13, 14, 15, 16, 17' -e '- rule every 3 rows replaying 3 rows, last replay ended at 5, committed length 8: range 5..8, proposal 15, 16, 17 at positions 5, 6, 7' -e '- same rule at committed length 7: no proposal (2 rows since the last pass, 3 needed)' -e '- rule every 3 rows replaying 8 rows, committed length 8: range 0..8, proposal is the whole log' -e '- rule every 3 rows replaying 9 rows, committed length 8: no proposal (more rows than committed)'`
- expect: `5`
- also green: no `.rs` file changed
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm-techniques): add replay proposal worked trace`
- done when: the expect line printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any `.rs` file or SPEC.md
- gpu: none

### 17.37 derive the n-gram draft trace

- id: FT17.37
- needs: FT17.1
- budget: 20 min
- crate(s): none (docs)
- read first: `proxima-model-interop/src/serving_fsm.rs::draft_prompt_lookup (~line 97 at e4cf9beb)` (after FT1.2 a private copy lives in `proxima-core/src/serving_state.rs` `mod tests` and is invisible to every other module, so no sans-IO test calls it; FT17.12 writes the test-local `ngram_draft` in `sansio_tests.rs` with this same body; the in-source example, history `[5,1,2,3,9,1,2,3]`, is cited, not rewritten)
- change:
  1. `SPECDIR/worked-examples.md`: add the heading `### n-gram draft from history` and, under it, exactly these 4 lines (derived by hand: the n-gram draft is what followed the last earlier occurrence of the final `ngram` ids, capped at max_k):
     ```
     - n-gram 2 over 5, 1, 2, 3, 9, 1, 2, 3 with max 5: the last two ids 2, 3 occurred earlier starting at index 2, draft 9, 1, 2, 3
     - same history with max 2: draft 9, 1
     - same history with max 0: empty draft
     - n-gram 2 over 1, 2: empty draft, no earlier room
     ```
- test: the section holds all 4 listed lines verbatim; the assertion is the count of exact line matches inside the section, which is 4 only when every derived value is present
- validate: `sed -n '/^### n-gram draft from history$/,/^##/p' proxima-tensor/specs/fsm-techniques/worked-examples.md | grep -cxF -e '- n-gram 2 over 5, 1, 2, 3, 9, 1, 2, 3 with max 5: the last two ids 2, 3 occurred earlier starting at index 2, draft 9, 1, 2, 3' -e '- same history with max 2: draft 9, 1' -e '- same history with max 0: empty draft' -e '- n-gram 2 over 1, 2: empty draft, no earlier room'`
- expect: `4`
- also green: no `.rs` file changed
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm-techniques): add n-gram draft worked trace`
- done when: the expect line printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any `.rs` file or SPEC.md
- gpu: none

### 17.3 derive the replay rewrite trace

- id: FT17.3
- needs: FT17.1
- budget: 20 min
- crate(s): none (docs)
- read first: `SPECDIR/worked-examples.md` section "replay proposal" (FT17.1); `SPECDIR/tasks-recut/07-rectify.md` card FT7.2 (the `accept_replay` resume value)
- change:
  1. `SPECDIR/worked-examples.md`: add the heading `### replay rewrite over a stale cache` and, under it, exactly these 7 lines (derived by hand: a replay rewinds the cache to the start of the range and appends the dense rows, where the dense row for an entry is the entry):
     ```
     - replay rewrite: the committed entries at positions 0 to 7 are 10 to 17; the cache holds 10, 11, 12, 13, 14, 15, 116, 117 because a sparse step wrote positions 6 and 7
     - rule every 3 rows replaying 3 rows, last replay ended at 5, length 8: range 5..8
     - the pass rewinds the cache to 10, 11, 12, 13, 14 and re-encodes 15, 16, 17 at positions 5, 6, 7 with the dense row function
     - after appending, the cache is 10 to 17: length 8 unchanged, rows before position 5 untouched, 3 rows rewritten, 2 changed (positions 6 and 7)
     - the same pass over an already dense cache leaves the rows byte-equal and changes 0 rows
     - state machine form: last 18, cache 10 to 14, enter verify with 15, 16, 17, placements 10 to 15, 10 to 16 and 10 to 17, accept replay gives last 18 with cache 10 to 17
     - accept replay with two placements for the three entries is refused
     ```
- test: the section holds all 7 listed lines verbatim; the assertion is the count of exact line matches inside the section, which is 7 only when every derived value is present
- validate: `sed -n '/^### replay rewrite over a stale cache$/,/^##/p' proxima-tensor/specs/fsm-techniques/worked-examples.md | grep -cxF -e '- replay rewrite: the committed entries at positions 0 to 7 are 10 to 17; the cache holds 10, 11, 12, 13, 14, 15, 116, 117 because a sparse step wrote positions 6 and 7' -e '- rule every 3 rows replaying 3 rows, last replay ended at 5, length 8: range 5..8' -e '- the pass rewinds the cache to 10, 11, 12, 13, 14 and re-encodes 15, 16, 17 at positions 5, 6, 7 with the dense row function' -e '- after appending, the cache is 10 to 17: length 8 unchanged, rows before position 5 untouched, 3 rows rewritten, 2 changed (positions 6 and 7)' -e '- the same pass over an already dense cache leaves the rows byte-equal and changes 0 rows' -e '- state machine form: last 18, cache 10 to 14, enter verify with 15, 16, 17, placements 10 to 15, 10 to 16 and 10 to 17, accept replay gives last 18 with cache 10 to 17' -e '- accept replay with two placements for the three entries is refused'`
- expect: `7`
- also green: no `.rs` file changed
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm-techniques): add replay rewrite worked trace`
- done when: the expect line printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any `.rs` file
- gpu: none

### 17.38 derive the commit placement trace

- id: FT17.38
- needs: FT17.1
- budget: 20 min
- crate(s): none (docs)
- read first: `SPECDIR/tasks-recut/01-fsm-generic-entry.md` card FT1.7 (the `accept_rows` values: caches `c1`, `c2`)
- change:
  1. `SPECDIR/worked-examples.md`: add the heading `### commit placement` and, under it, exactly these 3 lines (derived by hand: a log cache holds the entries consumed, so the placement after input i is the pre-verify rows plus the inputs up to i):
     ```
     - placements: prefix rows 1, state last 2, draft 3, 4: the verify pass consumes inputs 2, 3 at positions 1 and 2, so the placements are 1, 2 and 1, 2, 3, and the verifier readouts are 3, 4
     - accepting 2 rows commits the second placement: accepted 2, next 4, cache 1, 2, 3; resume gives last 4 with cache 1, 2, 3
     - accepting 0 rows commits the first placement: rollback snapshot 1, 2 to 3; rollback gives last 3 with cache 1, 2
     ```
- test: the section holds all 3 listed lines verbatim; the assertion is the count of exact line matches inside the section, which is 3 only when every derived value is present
- validate: `sed -n '/^### commit placement$/,/^##/p' proxima-tensor/specs/fsm-techniques/worked-examples.md | grep -cxF -e '- placements: prefix rows 1, state last 2, draft 3, 4: the verify pass consumes inputs 2, 3 at positions 1 and 2, so the placements are 1, 2 and 1, 2, 3, and the verifier readouts are 3, 4' -e '- accepting 2 rows commits the second placement: accepted 2, next 4, cache 1, 2, 3; resume gives last 4 with cache 1, 2, 3' -e '- accepting 0 rows commits the first placement: rollback snapshot 1, 2 to 3; rollback gives last 3 with cache 1, 2'`
- expect: `3`
- also green: no `.rs` file changed
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm-techniques): add commit placement worked trace`
- done when: the expect line printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any `.rs` file
- gpu: none

### 17.4 derive the replay passes under an ordered rule list trace

- id: FT17.4
- needs: FT17.1
- budget: 20 min
- crate(s): none (docs)
- read first: `SPECDIR/worked-examples.md` section "replay proposal" (FT17.1); `SPECDIR/tasks-recut/07-rectify.md` cards FT7.1 and FT7.3 (the rule semantics and the default list `[DraftNonempty]`)
- change:
  1. `SPECDIR/worked-examples.md`: add the heading `### replay passes under an ordered rule list` and, under it, exactly these 10 lines (derived by hand from the firing rule of the first line; `last_replay_end` is reset to the length after each pass):
     ```
     - a rule fires when its decision answers rows with rows > 0, rows <= max_rows and rows <= committed_len; the first firing rule in list order wins and returns committed_len - rows..committed_len; the periodic decision used below answers replay_rows when every_rows > 0 and committed_len - last_replay_end >= every_rows, and answers nothing when committed_len < last_replay_end
     - DraftNonempty never fires a replay; a caller reads it as: the list contains it
     - single rule every 4 rows replaying 2, last replay ended at 10, lengths 10 to 19 walked in order: passes at 14 (range 12..14) and 18 (range 16..18)
     - every other length from 10 to 19 returns no range
     - rules every 4 rows replaying 3 then every 2 rows replaying 1, last replay ended at 10: length 11 gives no range (1 row since the last pass)
     - same two rules, length 12: range 11..12 (the second rule fires, the first needs 4 rows)
     - same two rules, length 14: range 11..14 (the first rule wins)
     - edges, all no range at last replay ended at 10: every_rows 0 at length 14; replay_rows 0 at length 14; replay_rows 20 at length 14 (more rows than committed); any rule at length 8 (before the last pass)
     - the list holding only DraftNonempty gives no range at every length from 0 to 40
     - the list every 4 rows replaying 2 then DraftNonempty contains DraftNonempty; the empty list does not
     ```
- test: the section holds all 10 listed lines verbatim; the assertion is the count of exact line matches inside the section, which is 10 only when every derived value is present
- validate: `sed -n '/^### replay passes under an ordered rule list$/,/^##/p' proxima-tensor/specs/fsm-techniques/worked-examples.md | grep -cxF -e '- a rule fires when its decision answers rows with rows > 0, rows <= max_rows and rows <= committed_len; the first firing rule in list order wins and returns committed_len - rows..committed_len; the periodic decision used below answers replay_rows when every_rows > 0 and committed_len - last_replay_end >= every_rows, and answers nothing when committed_len < last_replay_end' -e '- DraftNonempty never fires a replay; a caller reads it as: the list contains it' -e '- single rule every 4 rows replaying 2, last replay ended at 10, lengths 10 to 19 walked in order: passes at 14 (range 12..14) and 18 (range 16..18)' -e '- every other length from 10 to 19 returns no range' -e '- rules every 4 rows replaying 3 then every 2 rows replaying 1, last replay ended at 10: length 11 gives no range (1 row since the last pass)' -e '- same two rules, length 12: range 11..12 (the second rule fires, the first needs 4 rows)' -e '- same two rules, length 14: range 11..14 (the first rule wins)' -e '- edges, all no range at last replay ended at 10: every_rows 0 at length 14; replay_rows 0 at length 14; replay_rows 20 at length 14 (more rows than committed); any rule at length 8 (before the last pass)' -e '- the list holding only DraftNonempty gives no range at every length from 0 to 40' -e '- the list every 4 rows replaying 2 then DraftNonempty contains DraftNonempty; the empty list does not'`
- expect: `10`
- also green: no `.rs` file changed
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm-techniques): add replay rule list worked trace`
- done when: the expect line printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any `.rs` file
- gpu: none

### 17.5 derive the idle steps around a waiting request trace

- id: FT17.5
- needs: FT17.1
- budget: 20 min
- crate(s): none (docs)
- read first: `SPECDIR/tasks-recut/10-idle-jobs.md` cards FT10.4 (the step list and its default) and FT10.7 (the request gate: a job runs only while no request waits, and polls for one); `proxima-model-interop/src/generate/prewarm_gate.rs::PrewarmGate (~line 22 at e4cf9beb)`
- change:
  1. `SPECDIR/worked-examples.md`: add the heading `### idle steps around a waiting request` and, under it, exactly these 10 lines (derived by hand; steps are counted from 1 in prose and list positions from 0):
     ```
     - default list: Prewarm, then Draft with branches 0, max_tokens 48, temperature_milli 800
     - the next step is the one after the count of finished steps, and it starts only while no request waits
     - idle gate, default list: both steps run in list order, 2 finished
     - list Draft with max_tokens 16, Prewarm, Draft with max_tokens 64: all three run in that order, each keeping its own parameters, 3 finished
     - empty list: 0 steps run, 0 finished
     - a request is waiting when the pass starts: 0 steps start, 0 finished
     - once that request leaves, the same list runs both steps, 2 finished
     - a request arrives while step 1 runs: step 1 sees it at its own poll and yields, 1 step finished, step 2 does not start
     - once the request leaves, a pass over the rest of the list from list position 1 runs step 2, 1 finished
     - across the two passes the steps ran as Prewarm once and Draft once
     ```
- test: the section holds all 10 listed lines verbatim; the assertion is the count of exact line matches inside the section, which is 10 only when every derived value is present
- validate: `sed -n '/^### idle steps around a waiting request$/,/^##/p' proxima-tensor/specs/fsm-techniques/worked-examples.md | grep -cxF -e '- default list: Prewarm, then Draft with branches 0, max_tokens 48, temperature_milli 800' -e '- the next step is the one after the count of finished steps, and it starts only while no request waits' -e '- idle gate, default list: both steps run in list order, 2 finished' -e '- list Draft with max_tokens 16, Prewarm, Draft with max_tokens 64: all three run in that order, each keeping its own parameters, 3 finished' -e '- empty list: 0 steps run, 0 finished' -e '- a request is waiting when the pass starts: 0 steps start, 0 finished' -e '- once that request leaves, the same list runs both steps, 2 finished' -e '- a request arrives while step 1 runs: step 1 sees it at its own poll and yields, 1 step finished, step 2 does not start' -e '- once the request leaves, a pass over the rest of the list from list position 1 runs step 2, 1 finished' -e '- across the two passes the steps ran as Prewarm once and Draft once'`
- expect: `10`
- also green: no `.rs` file changed
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm-techniques): add idle steps worked trace`
- done when: the expect line printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any `.rs` file
- gpu: none

### 17.6 derive the eviction by an ordered rule list trace

- id: FT17.6
- needs: FT17.1
- budget: 20 min
- crate(s): none (docs)
- read first: `SPECDIR/tasks-recut/05-tiers.md` sections "worked trace of the tiered cache" and card FT5.7 (the rule list `[Branch, Oldest]`); `proxima-model-interop/src/generate/prompt_cache.rs::PromptCache::eviction_victim (~line 819 at e4cf9beb)` (today's rule: an unused follow-up branch first, then the lowest stamp)
- change:
  1. `SPECDIR/worked-examples.md`: add the heading `### eviction by an ordered rule list` and, under it, exactly these 8 lines (derived by hand; candidates are `(stamp, is_branch)` in ascending stamp order):
     ```
     - candidates are (stamp, is_branch) in ascending stamp order; Branch selects the first branch entry, Oldest the lowest stamp; the first rule in list order that selects an entry decides; the default list is Branch, Oldest
     - candidates (3,no) (5,yes) (7,yes) (9,no): victim 5 under the default list, victim 3 under Oldest alone
     - candidates (3,no) (9,no): victim 3 under the default list, no victim under Branch alone (so a list must end with Oldest)
     - no candidates: no victim
     - tiered cache, hot limit 2, stamps issued 0, 1, 2 and so on: after storing stamps 0, 1, 2 the hot set is 0, 1, 2 and the victim is 0
     - after the restore stores stamp 3 the hot set is 1, 2, 3 and the victim is 1
     - then the hot set 2, 3, 4 gives victim 2, and 3, 4, 5 gives victim 3 (no branch entry exists, so Oldest decides each time)
     - hot entries (0,no) (1,yes) (2,no): the default list evicts stamp 1 (the branch), Oldest alone evicts stamp 0
     ```
- test: the section holds all 8 listed lines verbatim; the assertion is the count of exact line matches inside the section, which is 8 only when every derived value is present
- validate: `sed -n '/^### eviction by an ordered rule list$/,/^##/p' proxima-tensor/specs/fsm-techniques/worked-examples.md | grep -cxF -e '- candidates are (stamp, is_branch) in ascending stamp order; Branch selects the first branch entry, Oldest the lowest stamp; the first rule in list order that selects an entry decides; the default list is Branch, Oldest' -e '- candidates (3,no) (5,yes) (7,yes) (9,no): victim 5 under the default list, victim 3 under Oldest alone' -e '- candidates (3,no) (9,no): victim 3 under the default list, no victim under Branch alone (so a list must end with Oldest)' -e '- no candidates: no victim' -e '- tiered cache, hot limit 2, stamps issued 0, 1, 2 and so on: after storing stamps 0, 1, 2 the hot set is 0, 1, 2 and the victim is 0' -e '- after the restore stores stamp 3 the hot set is 1, 2, 3 and the victim is 1' -e '- then the hot set 2, 3, 4 gives victim 2, and 3, 4, 5 gives victim 3 (no branch entry exists, so Oldest decides each time)' -e '- hot entries (0,no) (1,yes) (2,no): the default list evicts stamp 1 (the branch), Oldest alone evicts stamp 0'`
- expect: `8`
- also green: no `.rs` file changed
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm-techniques): add eviction rule list worked trace`
- done when: the expect line printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any `.rs` file
- gpu: none

### 17.39 derive the block file byte layout trace

- id: FT17.39
- needs: FT17.1
- budget: 20 min
- crate(s): none (docs)
- read first: `SPECDIR/tasks-recut/05-tiers.md` section "block file format" (the header size `48 + 24 * layers`) and cards FT5.1 (`encoded_len`) and FT5.17 (`encode_block`)
- change:
  1. `SPECDIR/worked-examples.md`: add the heading `### block file byte layout` and, under it, exactly these 4 lines (derived by hand; the block size is `48 + 24 * layers` header bytes plus the payload bytes):
     ```
     - block bytes: a block with 2 layers; layer 0 holds 4 rows of widths 8, 8 and 4 bytes for key even, key odd and value; layer 1 holds no rows
     - encoded size: 48 + 24 * 2 = 96 header bytes plus 4 * (8 + 8 + 4) = 80 payload bytes, 176 bytes
     - decoding then encoding again gives the same 176 bytes
     - dropping the last 4 bytes is refused because the payload length disagrees with the header
     ```
- test: the section holds all 4 listed lines verbatim; the assertion is the count of exact line matches inside the section, which is 4 only when every derived value is present
- validate: `sed -n '/^### block file byte layout$/,/^##/p' proxima-tensor/specs/fsm-techniques/worked-examples.md | grep -cxF -e '- block bytes: a block with 2 layers; layer 0 holds 4 rows of widths 8, 8 and 4 bytes for key even, key odd and value; layer 1 holds no rows' -e '- encoded size: 48 + 24 * 2 = 96 header bytes plus 4 * (8 + 8 + 4) = 80 payload bytes, 176 bytes' -e '- decoding then encoding again gives the same 176 bytes' -e '- dropping the last 4 bytes is refused because the payload length disagrees with the header'`
- expect: `4`
- also green: no `.rs` file changed
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm-techniques): add block file layout worked trace`
- done when: the expect line printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any `.rs` file
- gpu: none

### 17.7 derive the one tier settles by vote trace

- id: FT17.7
- needs: FT17.1
- budget: 20 min
- crate(s): none (docs)
- read first: `/Users/brianbruggeman/repos/slot-0/proxima-windows/proxima-tensor/specs/pipeline-as-data/sketches/13-conformal-cascade.md` sections 2 and 3 (the draw, tally, settle loop and the vote cases; the file is on main: `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows ls-tree main proxima-tensor/specs/pipeline-as-data/sketches/13-conformal-cascade.md` prints it); `SPECDIR/tasks-recut/12-cascade.md` header (the vote rule is not a library function, so FT17.20 and FT17.27 hold it test-locally)
- change:
  1. `SPECDIR/worked-examples.md`: add the heading `### one tier settles by vote` and, under it, exactly these 15 lines (derived by hand from the first three lines; the last tier holds 1 sample and 0 allowed disagreement, so it always settles):
     ```
     - a tier draws samples answers and tallies one count per distinct answer
     - an answer is inside when its disagreement, samples minus count, is at most max_disagree
     - the tier settles when exactly one answer is inside, and escalates when zero or two or more are
     - samples 16, max_disagree 4: votes 13, 2, 1 settle on answer 0 (disagreement 3)
     - samples 16, max_disagree 4: votes 12, 4 settle on answer 0 (disagreement 4 is inside, 12 is not)
     - samples 16, max_disagree 4: votes 9, 7 escalate (7 and 9 both exceed 4)
     - samples 16, max_disagree 4: votes 8, 8 escalate
     - samples 16, max_disagree 4: votes 16 settle on answer 0
     - samples 16, max_disagree 10: votes 8, 8 escalate (both inside, a set of two)
     - samples 16, max_disagree 10: votes 13, 3 settle on answer 0 (disagreement 3 is inside, 13 is not)
     - the last tier always settles
     - tier loop over 3 tiers with flags false, true, false: tier 0 draws 8, 8 and escalates, tier 1 draws 15, 1 and settles (disagreement 1), tier 2 never runs; generation counters 1, 1, 0
     - flags false, false: tier 0 escalates, tier 1 is last and settles; counters 1, 1
     - flags true: one tier, it is last, counters 1
     - flags true, false, false: tier 0 settles, counters 1, 0, 0
     ```
- test: the section holds all 15 listed lines verbatim; the assertion is the count of exact line matches inside the section, which is 15 only when every derived value is present
- validate: `sed -n '/^### one tier settles by vote$/,/^##/p' proxima-tensor/specs/fsm-techniques/worked-examples.md | grep -cxF -e '- a tier draws samples answers and tallies one count per distinct answer' -e '- an answer is inside when its disagreement, samples minus count, is at most max_disagree' -e '- the tier settles when exactly one answer is inside, and escalates when zero or two or more are' -e '- samples 16, max_disagree 4: votes 13, 2, 1 settle on answer 0 (disagreement 3)' -e '- samples 16, max_disagree 4: votes 12, 4 settle on answer 0 (disagreement 4 is inside, 12 is not)' -e '- samples 16, max_disagree 4: votes 9, 7 escalate (7 and 9 both exceed 4)' -e '- samples 16, max_disagree 4: votes 8, 8 escalate' -e '- samples 16, max_disagree 4: votes 16 settle on answer 0' -e '- samples 16, max_disagree 10: votes 8, 8 escalate (both inside, a set of two)' -e '- samples 16, max_disagree 10: votes 13, 3 settle on answer 0 (disagreement 3 is inside, 13 is not)' -e '- the last tier always settles' -e '- tier loop over 3 tiers with flags false, true, false: tier 0 draws 8, 8 and escalates, tier 1 draws 15, 1 and settles (disagreement 1), tier 2 never runs; generation counters 1, 1, 0' -e '- flags false, false: tier 0 escalates, tier 1 is last and settles; counters 1, 1' -e '- flags true: one tier, it is last, counters 1' -e '- flags true, false, false: tier 0 settles, counters 1, 0, 0'`
- expect: `15`
- also green: no `.rs` file changed
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- commit: `docs(fsm-techniques): add vote settle worked trace`
- done when: the expect line printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit any `.rs` file
- gpu: none

## phase 2: proxima-core scaffolding

### 17.8 add the scripted readouts function and feature

- id: FT17.8
- needs: FT1.3
- budget: 20 min
- crate(s): proxima-core (features: alloc, sansio-script)
- read first: `proxima-core/src/serving_state.rs` (created by FT1.2; the module root and its `mod` lines; premise check: `git ls-files proxima-core/src/serving_state.rs` prints that path, else stop and report that FT1.2 has not landed); `proxima-core/Cargo.toml::[features]` (~line 12 at e4cf9beb; `alloc = ["time-alloc"]` at ~line 28 is the shape to copy); design "the scripted readouts" in this file
- change: the card adds two public items, `ScriptError` and `scripted_readouts`; no other card names `ScriptError`, so one public item is consumed by later cards.
  1. `proxima-core/Cargo.toml`: add the feature line `sansio-script = ["alloc"]` under `[features]` (a feature line, not a dependency edit).
  2. `proxima-core/src/serving_state.rs`: add `#[cfg(any(test, feature = "sansio-script"))] pub mod scripted;`.
  3. `proxima-core/src/serving_state/scripted.rs` (new): exactly the two items of the design (`ScriptError` and `scripted_readouts`, no struct, no trait), `alloc::{collections::BTreeMap, vec::Vec}` imports at the top, `ScriptError` deriving `Debug, PartialEq, Eq` with `thiserror::Error` and the messages `scripted readouts: {entries} entries but {positions} positions`, `scripted readouts: no row for position {position}` and `scripted readouts: cache holds {cache_rows} rows, behind first position {first_position}`.
- test: add three tests in `scripted.rs` `#[cfg(test)] mod tests`, using a local `struct Rows(Vec<u32>)` and `fn rows_len(cache: &Rows) -> usize { cache.0.len() }`:
  - `scripted_readouts_follow_the_table`: table `{(0,3) -> (4, -0.5, 0.25), (1,4) -> (5, -0.75, 0.5)}`; `scripted_readouts(&table, rows_len, &[3,4], &[0,1], &Rows(vec![9, 9]))` equals `Ok(vec![(4, -0.5, 0.25), (5, -0.75, 0.5)])`;
  - `scripted_refuses_unknown_rows_and_mismatched_lengths`: `scripted_readouts(&table, rows_len, &[3], &[7], &Rows(vec![9, 9]))` is `Err(MissingRow { position: 7 })`; `scripted_readouts(&table, rows_len, &[3,4], &[0], ..)` is `Err(LengthMismatch { entries: 2, positions: 1 })`;
  - `scripted_cache_behind_positions_is_refused`: a cache of 1 row with positions `[3]` is `Err(CacheBehindPositions { cache_rows: 1, first_position: 3 })`, and a cache of 5 rows with positions `[3]` over a table holding `(3,3)` is `Ok` (a replay pass lies inside rows already cached).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_8 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/scripted_/)'`
- expect: `3 passed`, names `scripted_readouts_follow_the_table`, `scripted_refuses_unknown_rows_and_mismatched_lengths`, `scripted_cache_behind_positions_is_refused`
- also green: clippy line and no_std alloc check, plus the non-test feature build: `cargo check -p proxima-core --no-default-features --features alloc,sansio-script --message-format=json > /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards17/17.8-check.json` and then `grep -c '"kind":\["lib"\].*"name":"proxima_core".*"features":\[[^]]*"sansio-script"' /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards17/17.8-check.json` printing `1`
- stage: `proxima-core/Cargo.toml`, `proxima-core/src/serving_state.rs`, `proxima-core/src/serving_state/scripted.rs`
- commit: `test(core): add the scripted readouts for the sans-io suite`
- done when: the expect line printed, clippy clean, the no_std check clean, the feature build grep printed `1`, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a trait; add a struct; add a `Pipe` impl; add a readout struct; add a third public item; touch `proxima-model-interop`
- gpu: none

### 17.9 add the 256-case property runner and its controls

- id: FT17.9
- needs: FT17.8
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/serving_state.rs` (where the `mod` lines go); the "properties" section of this file (the harness paragraph); `proxima-core/Cargo.toml::[dev-dependencies]` (~line 93 at e4cf9beb)
- change:
  1. `proxima-core/Cargo.toml`: run `cargo add --dev proptest -p proxima-core`. Expected result in `[dev-dependencies]`: `proptest.workspace = true`. If cargo writes a version string instead, stop and report (workspace inheritance did not apply).
  2. `proxima-core/src/serving_state.rs`: add `#[cfg(test)] mod sansio_tests;`.
  3. `proxima-core/src/serving_state/sansio_tests.rs` (new), only the runner (the cache and state helpers land with their first use in FT17.10): `fn try_256<Strategy: proptest::strategy::Strategy>(strategy: Strategy, body: impl Fn(Strategy::Value) -> Result<(), proptest::test_runner::TestCaseError>) -> (Result<(), proptest::test_runner::TestError<Strategy::Value>>, u32)` that builds `TestRunner::new_with_rng(Config { cases: 256, failure_persistence: None, ..Config::default() }, TestRng::deterministic_rng(RngAlgorithm::ChaCha))` and counts invocations in a `core::cell::Cell<u32>`; and `fn run_256(strategy, body)` that calls `try_256`, requires `Ok` (the assert message prints the error) and `assert_eq!(count, 256)`.
- test: add `scripted_control_runner_counts_256` (`run_256` over `0u32..1000` with an always-`Ok` body reports exactly 256 invocations, counted by the test's own `Cell`) and `scripted_control_runner_reports_failure` (`try_256` over `0u32..1000` with a body returning `Err(TestCaseError::fail("control"))` for every value returns `Err`; this is the control that must fail)
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_9 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/scripted_control_/)'`
- expect: `2 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/Cargo.toml`, `Cargo.lock`, `proxima-core/src/serving_state.rs`, `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): add the 256-case property runner with controls`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add any test named `sansio_*` (the count is exact and they land one per card); use the `proptest!` macro; edit the `proptest` line by hand; improvise if `TestRunner::new_with_rng` or `TestRng::deterministic_rng` has another signature in the locked proptest (stop and report it)
- gpu: none

## phase 3: transitions (2 tests)

### 17.10 `sansio_transitions_legal`

- id: FT17.10
- needs: FT17.9, FT1.7, FT7.2, FT8.1
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/serving_state.rs::ServingState` (created by FT1.2, FT1.3, FT7.2, FT8.1: every method at its landed signature; premise check: `accept_rows`, `accept_replay`, `start_at` and `Verify.last` exist, else stop); `proxima-model-interop/src/serving_fsm.rs::tests::walkthrough_drives_every_legal_transition (~line 318 at e4cf9beb; `serving_state_walkthrough_drives_every_legal_transition` after FT1.3)`; the legal table in this file
- change:
  1. `sansio_tests.rs`: add the helpers this test and the later ones use, each exercised here:
     - `#[derive(Debug, Clone, PartialEq)] struct LogCache { rows: Vec<u32> }`, `fn log(rows: &[u32]) -> LogCache`, `fn rows_of(cache: &LogCache) -> usize`;
     - `fn placements(pre: &LogCache, inputs: &[u32]) -> Vec<LogCache>` (placement `i` is `pre.rows ++ inputs[..=i]`);
     - `type Table = BTreeMap<(usize, u32), (u32, f32, f32)>`, `fn script(rows: &[(usize, u32, u32)]) -> Table` (each `(position, entry, next)` becomes `((position, entry), (next, -0.5, 0.25))`) and `fn readouts(table: &Table, entries: &[u32], positions: &[usize], cache: &LogCache) -> Vec<(u32, f32, f32)>` (calls `scripted_readouts(table, rows_of, entries, positions, cache)` and `expect`s the result with the message `every scripted row exists`);
     - `fn check<Value: PartialEq + core::fmt::Debug>(rows_checked: &mut u32, label: &str, actual: Value, expected: Value)` (`assert_eq!(actual, expected, "{label}")` then `*rows_checked += 1`);
     - `enum StateKind { Prefill, Decode, Verify, Accept, Rollback, Done }` and `fn state_in(kind: StateKind) -> ServingState<u32, LogCache>` built by driving the real transitions: Prefill is `start(vec![1,2,3], log(&[]))`; Decode is Prefill then `advance_prefill(4, log(&[1,2,3]))`; Verify is Decode then `enter_verify(vec![5,6])`; Accept is Verify then `accept_rows(2, &[5,6], placements(&log(&[1,2,3]), &[4,5]))`; Rollback is the same with count 0; Done is Decode then `finish()`; each step `.expect("legal")`.
  2. `sansio_tests.rs`: add `fn sansio_transitions_legal()` with a doc comment saying it drives every legal transition once. It checks these 18 rows with `check` (the verify rows take their readouts from `readouts(&script(&[(3,4,5),(4,5,6)]), &[4,5], &[3,4], &log(&[1,2,3]))`, and `placements(&log(&[1,2,3]), &[4,5])`):
     - 1: `ServingState::start(vec![1,2,3], log(&[]))` equals `Prefill { positions: vec![1,2,3], start_position: 0, cache: log(&[]) }`;
     - 2: `ServingState::start_at(vec![7,8,9], 5, log(&[0,1,2,3,4]))` equals `Prefill { positions: vec![7,8,9], start_position: 5, cache: log(&[0,1,2,3,4]) }`;
     - 3: `state_in(Prefill).advance_prefill(4, log(&[1,2,3]))` is `Ok(Decode { last: 4, cache: log(&[1,2,3]) })`;
     - 4: `state_in(Decode).advance_decode(5, log(&[1,2,3,4]))` is `Ok(Decode { last: 5, cache: log(&[1,2,3,4]) })`;
     - 5: `state_in(Decode).enter_verify(vec![5,6])` is `Ok(Verify { draft: vec![5,6], last: 4, snapshot: log(&[1,2,3]), cache: log(&[1,2,3]) })`;
     - 6: `state_in(Verify).accept(&[5,6], placements)` is `Ok(Accept { n: 2, next: 6, cache: log(&[1,2,3,4,5]) })`;
     - 7: the same with the readouts of `readouts(&script(&[(3,4,5),(4,5,9)]), &[4,5], &[3,4], &log(&[1,2,3]))` (`[5,9]`) is `Ok(Rollback { snapshot: log(&[1,2,3,4,5]), to: 9 })`;
     - 8: `accept_rows(2, &[5,6], placements)` is `Ok(Accept { n: 2, next: 6, cache: log(&[1,2,3,4,5]) })`;
     - 9: `accept_rows(0, &[5,6], placements)` is `Ok(Rollback { snapshot: log(&[1,2,3,4]), to: 5 })`;
     - 10: `state_in(Decode).advance_decode(4, log(&[1]))` then `enter_verify(vec![2,3])` then `accept_replay(placements(&log(&[1]), &[2,3]))` is `Ok(Decode { last: 4, cache: log(&[1,2,3]) })`;
     - 11: `state_in(Accept).resume()` is `Ok(Decode { last: 6, cache: log(&[1,2,3,4,5]) })`;
     - 12: `state_in(Rollback).rollback()` is `Ok(Decode { last: 5, cache: log(&[1,2,3,4]) })`;
     - 13 to 18: `finish` from Prefill, Decode, Verify, Accept, Rollback and Done equals `Done { cache: log(&[]) }`, `Done { cache: log(&[1,2,3]) }`, `Done { cache: log(&[1,2,3]) }`, `Done { cache: log(&[1,2,3,4,5]) }`, `Done { cache: log(&[1,2,3,4]) }` (the rollback snapshot) and `Done { cache: log(&[1,2,3]) }`.
     End with `assert_eq!(rows_checked, 18)`.
- test: `sansio_transitions_legal` asserts all 18 rows by value and `rows_checked == 18`
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_10 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_transitions_legal/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove every legal serving transition`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change any `ServingState` method; if a method named above is absent or has a different signature, stop and report
- gpu: none

### 17.11 `sansio_transitions_illegal`

- id: FT17.11
- needs: FT17.10
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/serving_state.rs::ServingFsmError` (`IllegalTransition { attempted }`) and the private `settle` guard (FT1.7; premise check: `grep -n "ServingFsmError\|fn settle" proxima-core/src/serving_state.rs` prints at least 2 lines, else stop and report); `sansio_tests.rs::state_in`, `placements`, `check`; the refused tables in this file
- change:
  1. `sansio_tests.rs`: add `enum Event { AdvancePrefill, AdvanceDecode, EnterVerify, Accept, AcceptRows, AcceptReplay, Resume, Rollback }`, `fn call(event: Event, state: ServingState<u32, LogCache>) -> Result<ServingState<u32, LogCache>, ServingFsmError>` (fixed arguments: `advance_prefill(4, log(&[1,2,3]))`, `advance_decode(5, log(&[1,2,3,4]))`, `enter_verify(vec![5,6])`, `accept(&[5,6], p)`, `accept_rows(2, &[5,6], p)`, `accept_replay(p)` with `p = placements(&log(&[1,2,3]), &[4,5])`), `fn attempted(event: Event) -> &'static str` (the method name) and `fn is_legal(event: Event, kind: StateKind) -> bool` (the 8 legal pairs: `AdvancePrefill` at Prefill, `AdvanceDecode` and `EnterVerify` at Decode, `Accept`, `AcceptRows` and `AcceptReplay` at Verify, `Resume` at Accept, `Rollback` at Rollback).
  2. `sansio_tests.rs`: add `fn sansio_transitions_illegal()`. Loop the 8 events over the 6 kinds, skip the 8 legal pairs, and for each of the other 40 pairs assert `call(event, state_in(kind))` equals `Err(ServingFsmError::IllegalTransition { attempted: attempted(event) })` through `check`. Then the 9 guards from a Verify state (`state_in(Verify)`; the empty-draft rows use `state_in(Decode).enter_verify(vec![])`): empty draft with `accept(&[], vec![])`, `accept_rows(0, &[], vec![])` and `accept_replay(vec![])`; `row_tokens` of length 1 against a 2-draft for `accept` and `accept_rows(1, ..)`; one placement for a 2-draft for `accept`, `accept_rows(1, ..)` and `accept_replay`; `accept_rows(3, &[5,6], two placements)`. Each is `Err(IllegalTransition { attempted: "<the called method>" })`. Each call runs on a clone. `assert_eq!(rows_checked, 49)`.
- test: `sansio_transitions_illegal`, exactly 49 refused rows (40 pairs and 9 guards)
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_11 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_transitions_/)'`
- expect: `2 passed`, names `sansio_transitions_legal` and `sansio_transitions_illegal`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove every illegal serving transition`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add an error variant; if the pair count differs on the landed state machine (a new method or state such as `refine`), stop and report the new count
- gpu: none

## phase 4: the five core place tests

Each place test asserts its `rows_checked` count through `check` and names its worked trace in a doc comment on the test (the one allowed comment).

### 17.12 `sansio_place_propose`

- id: FT17.12
- needs: FT17.11, FT17.1, FT17.37, FT7.1
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-model-interop/src/serving_fsm.rs::draft_prompt_lookup (~line 97 at e4cf9beb)` (the body `ngram_draft` copies; after FT1.2 the only copy in proxima-core is private to `serving_state.rs` `mod tests` and cannot be called from `sansio_tests.rs`); `proxima-core/src/serving_state/enter.rs::{EnterRule, replay_range}` (`EnterRule { DraftNonempty, Replay { max_rows, decide } }` from FT7.0, `replay_range` from FT7.1; premise check: `git grep -n "pub fn replay_range" -- proxima-core/src/serving_state/enter.rs` prints 1 line, else stop and report; neither symbol is on main at e4cf9beb); `SPECDIR/worked-examples.md` sections "replay proposal" (FT17.1) and "n-gram draft from history" (FT17.37); `scripted.rs::scripted_readouts` (FT17.8) through the `readouts` helper of `sansio_tests.rs`
- change:
  1. `sansio_tests.rs`: add `fn ngram_draft(history: &[u32], ngram: usize, max_k: usize) -> Vec<u32>` (test-local, private, body copied verbatim from `draft_prompt_lookup` at `serving_fsm.rs` ~lines 91-106 at e4cf9beb with `Entry` fixed to `u32`; FT17.23 and later cards call it) and `fn sansio_place_propose()` reproducing the replay proposal and n-gram draft trace, 9 rows through `check`: `ngram_draft` over `[5,1,2,3,9,1,2,3]` with ngram 2 and max_k 5 is `[9,1,2,3]`, max_k 2 is `[9,1]`, max_k 0 is `[]`; history `[1,2]` with ngram 2 is `[]` (4 rows). The periodic cadence is test-local, because the library holds no cadence: add `fn periodic_decision<const EVERY_ROWS: usize, const REPLAY_ROWS: usize>(committed_len: usize, last_replay_end: usize) -> Option<usize>` (`None` when `EVERY_ROWS == 0` or `committed_len < last_replay_end` or `committed_len - last_replay_end < EVERY_ROWS`, else `Some(REPLAY_ROWS)`) and `fn periodic_rule<const EVERY_ROWS: usize, const REPLAY_ROWS: usize>() -> EnterRule` (`EnterRule::Replay { max_rows: u32::try_from(REPLAY_ROWS).expect("replay rows fit u32"), decide: periodic_decision::<EVERY_ROWS, REPLAY_ROWS> }`); later cards in this file reuse both. Replay proposals over the committed log `(10..=17)` with `last_replay_end = 5`: `[periodic_rule::<3, 3>()]` at length 8 is `Some(5..8)` and the proposed entries `log[5..8]` are `[15,16,17]`; at length 7 it is `None`; `[periodic_rule::<3, 8>()]` at length 8 is `Some(0..8)`; `[periodic_rule::<3, 9>()]` at length 8 is `None` (4 rows). Then feed the proposal to the scripted readouts: `readouts(&script(&[(5,15,16),(6,16,17),(7,17,18)]), &[15,16,17], &[5,6,7], &log(&[10,11,12,13,14]))` has `.0` values `[16,17,18]` (1 row). `assert_eq!(rows_checked, 9)`.
- test: `sansio_place_propose`, traces "replay proposal" and "n-gram draft from history"
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_12 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_place_propose/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove the propose place`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a proposal function; make `ngram_draft` `pub` or move it out of `sansio_tests.rs`; put `periodic_decision` or `periodic_rule` in `enter.rs` or any `src/` file outside `sansio_tests.rs`; stop and report if `replay_range` is absent
- gpu: none

### 17.14 `sansio_place_accept`: chain and accept_rows rows

- id: FT17.14
- needs: FT17.12, FT1.7
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/serving_state.rs::accept` and `::accept_rows` (FT1.7; premise check: `grep -n "fn accept\b\|fn accept_rows" proxima-core/src/serving_state.rs` prints 2 lines, else stop and report); `proxima-core/src/serving_state.rs` test `serving_state_walkthrough_drives_every_legal_transition` (FT1.3: the chain values); `sansio_tests.rs::placements`, `script`
- change:
  1. `sansio_tests.rs`: add `fn sansio_place_accept()` with 5 rows over the state `start(vec![1], log(&[]))`, `advance_prefill(2, log(&[1]))`, `enter_verify(vec![3,4])`; the verifier's choices are the `.0` values of `readouts(&script(&[(1,2,3),(2,3,4)]), &[2,3], &[1,2], &log(&[1]))`, placements `placements(&log(&[1]), &[2,3])` (`c1 = [1,2]`, `c2 = [1,2,3]`):
     - chain `accept` with choices `[3,4]` is `Ok(Accept { n: 2, next: 4, cache: log(&[1,2,3]) })`;
     - chain `accept` with the `.0` values of `readouts(&script(&[(1,2,3),(2,3,55)]), &[2,3], &[1,2], &log(&[1]))` (`[3,55]`) is `Ok(Rollback { snapshot: log(&[1,2,3]), to: 55 })` (the placement at index 1);
     - `accept_rows(0, &[3,4], placements)` is `Ok(Rollback { snapshot: log(&[1,2]), to: 3 })`;
     - `accept_rows(2, &[3,4], placements)` is `Ok(Accept { n: 2, next: 4, cache: log(&[1,2,3]) })`;
     - `accept_rows(3, &[3,4], placements)` is `Err(IllegalTransition { attempted: "accept_rows" })`.
     `assert_eq!(rows_checked, 5)`. The next three cards add rows to this one test and raise the count.
- test: `sansio_place_accept` (5 rows): the in-source walkthrough values and the `accept_rows` values
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_14 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_place_accept/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove the accept place on chains`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `accept` or `accept_rows`
- gpu: none

### 17.15 `sansio_place_accept`: equality rule row

- id: FT17.15
- needs: FT17.14, FT1.9, FT1.10
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/serving_state/action_fixture.rs::{run_sequential, run_speculative, Trace}` (FT1.8, FT1.9; `pub(crate)` and test-only); `proxima-core/src/accept_rule.rs::AcceptRule::accepted_rows` (FT1.10); premise check: `git ls-files proxima-core/src/serving_state/action_fixture.rs proxima-core/src/accept_rule.rs` prints 2 paths and `grep -n "fn accepted_rows" proxima-core/src/accept_rule.rs` prints 1 line, else stop and report which card has not landed; `SPECDIR/tasks-recut/01-fsm-generic-entry.md` card FT1.12 (the same three rules as TOML tables, with the counts); `sansio_tests.rs::sansio_place_accept`
- change:
  1. `sansio_tests.rs`: add `fn field_agreement(drafted: &Action, chosen: &Action) -> f32` (the number of equal fields among `key`, `delta` and `version`, as `f32`, divided by 3.0; 1.0 only when all three agree) and extend `sansio_place_accept` with the equality row: `rule = AcceptRule { similarity_floor: 1.0, min_run: 0, max_rows: 8 }` (a struct literal; the same value FT1.12 loads from `similarity_floor = 1.0`, `min_run = 0`, `max_rows = 8`); `run_speculative(|draft, choices| rule.accepted_rows(draft, choices, field_agreement))` has the same `history` and `environment` as `run_sequential()`, and `(rounds, rejected_rounds, accepted_rows)` equals `(4, 4, 8)`. The row counts as 1 `check`; raise the count: `assert_eq!(rows_checked, 6)`.
- test: `sansio_place_accept` (6 rows), trace "action speculation", equality
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_15 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_place_accept/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove the accept place under equality`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change a worked count to make the row pass (a mismatch is a false premise: stop and report); gate the row behind a feature
- gpu: none

### 17.16 `sansio_place_accept`: verifier rule row

- id: FT17.16
- needs: FT17.15
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/accept_rule.rs::AcceptRule::accepted_rows` (FT1.10; premise check: `grep -n "fn accepted_rows" proxima-core/src/accept_rule.rs` prints 1 line, else stop and report; the cap applies before the minimum run); `SPECDIR/tasks-recut/01-fsm-generic-entry.md` card FT1.12 (`configured_rule_verifier_exact_matches_sequential`); `sansio_tests.rs::sansio_place_accept`
- change:
  1. `sansio_tests.rs`: extend `sansio_place_accept` with the verifier row: `rule = AcceptRule { similarity_floor: 1.0, min_run: 0, max_rows: u16::MAX }` (the verifier's own choice is the true entry, so exact rollback keeps only the equal prefix; the cap is off); `run_speculative` with that rule equals `run_sequential()` on `history` and `environment`, and the counts are `(4, 4, 8)`. `assert_eq!(rows_checked, 7)`.
- test: `sansio_place_accept` (7 rows), trace "action speculation", verifier with exact rollback
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_16 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_place_accept/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove the accept place under a verifier`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the rule type or a worked count
- gpu: none

### 17.17 `sansio_place_accept`: anchor run rule row

- id: FT17.17
- needs: FT17.16
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/accept_rule.rs::AcceptRule::accepted_rows` (FT1.10; premise check: `grep -n "fn accepted_rows" proxima-core/src/accept_rule.rs` prints 1 line, else stop and report; a run shorter than `min_run` is discarded, so the count is 0); `SPECDIR/tasks-recut/01-fsm-generic-entry.md` card FT1.12 (`configured_rule_anchor_run_matches_sequential`); `sansio_tests.rs::sansio_place_accept`
- change:
  1. `sansio_tests.rs`: extend `sansio_place_accept` with the anchor run row: `rule = AcceptRule { similarity_floor: 1.0, min_run: 3, max_rows: u16::MAX }`; `run_speculative` with that rule equals `run_sequential()` on `history` and `environment`, and the counts are `(6, 6, 6)`. `assert_eq!(rows_checked, 8)`.
- test: `sansio_place_accept` (8 rows), trace "action speculation", anchor run
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_17 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_place_accept/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove the accept place under an anchor run`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the rule type or a worked count
- gpu: none

### 17.18 `sansio_place_commit`

- id: FT17.18
- needs: FT17.17, FT17.3, FT17.38, FT7.1, FT7.2
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/serving_state.rs::accept_rows` (the placement it chooses is the commit) and `::accept_replay` (FT7.2); `proxima-core/src/serving_state/enter.rs::replay_range` (FT7.1; premise check: absent means stop and report); `sansio_tests.rs::periodic_rule` (FT17.12); `SPECDIR/worked-examples.md` sections "commit placement" (FT17.38) and "replay rewrite over a stale cache" (FT17.3)
- change:
  1. `sansio_tests.rs`: add `fn sansio_place_commit()` with 6 rows through `check`, over the state `start(vec![1], log(&[]))`, `advance_prefill(2, log(&[1]))`, `enter_verify(vec![3,4])`, `c1 = log(&[1,2])`, `c2 = log(&[1,2,3])`:
     - `accept_rows(2, &[3,4], vec![c1, c2])` then `resume()` is `Ok(Decode { last: 4, cache: log(&[1,2,3]) })`;
     - `accept_rows(0, ..)` then `rollback()` is `Ok(Decode { last: 3, cache: log(&[1,2]) })`;
     - replay over stale rows: `stale = log(&[10,11,12,13,14,15,116,117])`, `replay_range(&[periodic_rule::<3, 3>()], 8, 5)` is `Some(5..8)` (`periodic_rule` is the test-local helper FT17.12 added); the pass rewinds to `stale.rows[..5]`, takes the dense rows from `readouts(&script(&[(5,15,15),(6,16,16),(7,17,17)]), &[15,16,17], &[5,6,7], &rewound)` (`.0` of each readout is the dense row for that entry; the table is the identity), and appends them: the cache is `log(&(10..=17).collect::<Vec<u32>>())`, its length is 8 and the number of rows that changed against `stale` is 2;
     - replay over an already dense cache `log(&[10..=17])`: the same pass gives the byte-equal cache and 0 changed rows;
     - the state form: `start(vec![0], log(&[]))`, `advance_prefill(18, log(&[10,11,12,13,14]))`, `enter_verify(vec![15,16,17])`, placements `placements(&log(&[10,11,12,13,14]), &[15,16,17])`, `accept_replay(placements)` is `Ok(Decode { last: 18, cache: log(&[10,11,12,13,14,15,16,17]) })`;
     - `accept_replay` with two placements for the 3-entry draft is `Err(IllegalTransition { attempted: "accept_replay" })`.
     `assert_eq!(rows_checked, 6)`.
- test: `sansio_place_commit`, traces "commit placement" and "replay rewrite over a stale cache"
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_18 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_place_commit/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove the commit place`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a commit or overwrite function; stop and report if `accept_replay` or `replay_range` is absent
- gpu: none

### 17.19 `sansio_place_enter`

- id: FT17.19
- needs: FT17.18, FT17.4, FT7.1
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/serving_state/enter.rs::{EnterRule, replay_range}` (FT7.0, FT7.1; premise check: `git grep -n "pub fn replay_range" -- proxima-core/src/serving_state/enter.rs` prints 1 line, else stop and report); `sansio_tests.rs::{periodic_decision, periodic_rule}` (FT17.12); `SPECDIR/worked-examples.md` section "replay passes under an ordered rule list"
- change:
  1. `sansio_tests.rs`: add `fn sansio_place_enter()` with 4 rows through `check`: single rule `[periodic_rule::<4, 2>()]` walked over committed lengths `10..=19` from `last_replay_end = 10`, resetting `last_replay_end` after each `Some`, collects exactly `[12..14, 16..18]` (row 1); rule order `[periodic_rule::<4, 3>(), periodic_rule::<2, 1>()]` at `last_replay_end = 10` gives `None` at 11, `Some(11..12)` at 12 and `Some(11..14)` at 14 (row 2); the edge list, all `None` at `last_replay_end = 10`: `[periodic_rule::<0, 2>()]` at 14, `[periodic_rule::<4, 0>()]` at 14, `[periodic_rule::<4, 20>()]` at 14, `[periodic_rule::<4, 2>()]` at length 8 (row 3); `[DraftNonempty]` is `None` at every length in `0..=40`, `[periodic_rule::<4, 2>(), EnterRule::DraftNonempty].contains(&EnterRule::DraftNonempty)` is true and `[].contains(..)` is false (row 4). `assert_eq!(rows_checked, 4)`.
- test: `sansio_place_enter`, trace "replay passes under an ordered rule list"
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_19 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_place_enter/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove the enter place`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a rule type or a function; stop and report if `replay_range` is absent
- gpu: none

### 17.20 `sansio_place_settle`

- id: FT17.20
- needs: FT17.19, FT17.7
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `SPECDIR/worked-examples.md` section "one tier settles by vote" (FT17.7: the rule and the seven vote cases); `proxima-core/src/serving_state/sansio_tests.rs::check` (created by FT17.10: the row counter this card calls); `SPECDIR/tasks-recut/12-cascade.md` header (no `Agreement` or `settle_votes` exists in any library crate, so this card writes them in the test file)
- change:
  1. `sansio_tests.rs`: add the test-local record `#[derive(Clone, Copy)] struct Agreement { samples: u32, max_disagree: u32 }` and the test-local function `fn settle_votes(votes: &[u32], agreement: Agreement) -> Option<usize>`: an answer is inside when `agreement.samples - votes[index] <= agreement.max_disagree` (use `saturating_sub`); the result is `Some(index)` when exactly one answer is inside and `None` when zero or two or more are (about 10 lines, built from `iter().enumerate().filter(..)`, no allocation). Both stay private to the test module; this card is their first use, so nothing is dead.
  2. `sansio_tests.rs`: add `fn sansio_place_settle()` with 7 rows through `check`, each a vote case from the trace, with `tight = Agreement { samples: 16, max_disagree: 4 }` and `loose = Agreement { samples: 16, max_disagree: 10 }`: `settle_votes(&[13,2,1], tight)` is `Some(0)`, `settle_votes(&[12,4], tight)` is `Some(0)`, `settle_votes(&[9,7], tight)` is `None`, `settle_votes(&[8,8], tight)` is `None`, `settle_votes(&[16], tight)` is `Some(0)`, `settle_votes(&[8,8], loose)` is `None` and `settle_votes(&[13,3], loose)` is `Some(0)`. `assert_eq!(rows_checked, 7)`.
- test: `sansio_place_settle`, trace "one tier settles by vote"
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_20 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_place_settle/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove the settle place`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: put `Agreement` or `settle_votes` in any file outside `sansio_tests.rs` (the vote rule is a technique, and no library holds a technique); add a `pub` modifier to either
- gpu: none

## phase 5: the six property tests

### 17.21 `sansio_prop_log_append_only`

- id: FT17.21
- needs: FT17.20, FT7.1, FT7.2
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `sansio_tests.rs::run_256`, `LogCache`, `placements`, `script`; property 1 in this file; `proxima-core/src/serving_state/enter.rs` and `ServingState::accept_replay` (the replay form; FT7.1 and FT7.2; premise check: `grep -n "fn replay_range" proxima-core/src/serving_state/enter.rs` and `grep -n "fn accept_replay" proxima-core/src/serving_state.rs` each print 1 line, else stop and report)
- change:
  1. `sansio_tests.rs`: add `enum Step { Decode, Verify { draft_len: u8, accepted: u8 }, Replay { rows: u8 } }`, `enum StepKind { Grow, Replay { rows: usize } }`, `fn log_only_grows(before: &[u32], after: &[u32], kind: StepKind) -> bool` (`Grow` is `after.starts_with(before)`; `Replay { rows }` is `after.len() == before.len()` and the first `len - min(rows, len)` entries are equal), and `fn run_log_driver(steps: &[Step]) -> Vec<(Vec<u32>, Vec<u32>, StepKind)>` (for each step, the committed log before and after it, where the committed log is exactly the `rows` field of the state's `LogCache` and nothing else: `cache.rows` read from the `Decode { last, cache }` state before the step and again after the step settles back to `Decode`; `last` is never part of the log). The driver: scripted table `next = (entry + 1) % 11` at positions `0..200` for entries `0..11`; start `start(vec![0,1,2], log(&[]))` then `advance_prefill(3, log(&[0,1,2]))`; between steps the state is `Decode { last, cache }`. `Decode`: next from `readouts(&table, &[last], &[cache.rows.len()], &cache)`, cache `rows ++ [last]`, `advance_decode(next, cache)`. `Verify`: the true chain `t0 = next(last)`, `t(i) = next(t(i-1))`; `draft` is `draft_len` entries of that chain with the entry at index `accepted` replaced by `(truth + 1) % 11` when `accepted < draft_len`; inputs `[last] ++ draft[..draft_len - 1]`; choices from `readouts(&table, &inputs, &positions, &cache)`; `enter_verify(draft)`, `accept_rows(accepted, &choices, placements(&cache, &inputs))`, then `resume()` on `Accept` or `rollback()` on `Rollback`. `Replay`: `f = rows.min(cache.rows.len())`, nothing when `f == 0`; otherwise rewind to `rows[..len - f]`, `advance_decode(last, rewound)`, `enter_verify(entries)` with the entries `rows[len - f..]`, then `accept_replay(placements(&rewound, &entries))` (dense rows equal the committed rows). Add `fn sansio_prop_log_append_only()`: `run_256` over the step strategy of property 1; the body requires every recorded step to satisfy `log_only_grows` with `prop_assert!`.
- test: `sansio_prop_log_append_only`, 256 cases
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_21 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_prop_log_append_only/)'`
- expect: `1 passed` (the in-runner count assertion proves 256 cases ran)
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove the committed log only grows`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: use `proptest!` macros (they read the case count from the environment); skip `run_256`
- gpu: none

### 17.22 `sansio_prop_rollback_restores`

- id: FT17.22
- needs: FT17.21
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/serving_state.rs::accept` and `::rollback` (premise check: `grep -n "fn accept\b\|fn rollback" proxima-core/src/serving_state.rs` prints 2 lines, else stop and report; `snapshot` is the chosen placement; `Verify.snapshot` is written and read by nothing, kept by FT1.2); `sansio_tests.rs::placements`; property 2 in this file
- change:
  1. `sansio_tests.rs`: add `fn sansio_prop_rollback_restores()` through `run_256`. Strategy: `(vec(0u32..6, 0..24), 0u32..6, vec(0u32..6, 1..6)).prop_flat_map(|(prefix, last, draft)| { let len = draft.len(); (Just(prefix), Just(last), Just(draft), 0..len) })`. Body: `pre = LogCache { rows: prefix }`; `choices` equal to `draft` before `mismatch` and `(draft[mismatch] + 1) % 6` at it, and `draft` after it; inputs `[last] ++ draft[..draft.len() - 1]`; state `start(vec![0], pre.clone())`, `advance_prefill(last, pre.clone())`, `enter_verify(draft)`, `accept(&choices, placements(&pre, &inputs))`, require `Rollback`, then `rollback()`. Require `post.rows[..pre.rows.len()] == pre.rows` and `post.rows.len() == pre.rows.len() + 1 + mismatch` with `prop_assert!`.
- test: `sansio_prop_rollback_restores`, 256 cases
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_22 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_prop_rollback_restores/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove rollback restores the cache prefix`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: assert `post == pre`
- gpu: none

### 17.23 `sansio_prop_exact_accept_equals_sequential` (chain accept)

- id: FT17.23
- needs: FT17.22, FT17.12
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `sansio_tests.rs::run_256`, `placements`, `log`; `proxima-model-interop/src/serving_fsm.rs::tests::speculation_matches_plain_greedy_for_thirty_two_tokens (~line 527 at e4cf9beb; `serving_state_speculation_matches_plain_greedy_for_thirty_two_tokens` after FT1.4)`: the chain driver this generalizes; `scripted.rs::scripted_readouts` through the `readouts` helper of `sansio_tests.rs` (premise check: `grep -n "fn scripted_readouts" proxima-core/src/serving_state/scripted.rs` and `grep -n "fn ngram_draft" proxima-core/src/serving_state/sansio_tests.rs` each print 1 line, else stop and report that FT17.8 or FT17.12 has not landed)
- change:
  1. `sansio_tests.rs`: add `fn script_from(table: &[Vec<u32>]) -> Table` (each `(position, entry)` with `entry` in `0..6` maps to `(table[position][entry], -0.5, 0.25)`), `fn run_sequential_log(table: &[Vec<u32>], prompt: &[u32], max_tokens: usize) -> Vec<u32>` (`last = table[P-1][prompt[P-1]]`; for each of `max_tokens` steps push `last` as generated entry `i` at position `P + i`, then `last = table[P + i][last]`) and `fn run_speculative_log(table, prompt, max_tokens) -> (Vec<u32>, u32)` returning the first `max_tokens` generated entries and the number of verify rounds. The speculative driver keeps the state `Decode { last, cache }` with `cache.rows = prompt ++ generated[..g-1]` and `last = generated[g-1]`. Each round: `draft = ngram_draft(&(prompt ++ generated), 2, 4)`; empty draft is one plain decode step committing `table[rows.len()][last]`; otherwise inputs `[last] ++ draft[..k-1]`, positions from `rows.len()`, choices from the scripted `readouts` over `script_from(table)`, `enter_verify(draft)`, `accept(&choices, placements)`, then `Accept` appends all `k` draft entries and `resume()`, `Rollback { to }` appends `draft[..n]` and `to` and `rollback()`. Each helper fits one screen. Add `fn sansio_prop_exact_accept_equals_sequential()` through `run_256`; this card has no policy or rule parameter, because the only reader of such a parameter lands in the next card, and a parameter nothing reads fails clippy: `table` is `vec(vec(0u32..6, 6..=6), 64..=64)`, prompt `vec(0u32..6, 1..6)`, `max_tokens 1..40`; assert the two logs are equal with `prop_assert_eq!`; sum the verify rounds in a `Cell<u32>` and, after `run_256`, assert the sum is at least 1 (guards the empty-draft degeneracy where speculation never runs).
- test: `sansio_prop_exact_accept_equals_sequential`, chain accept
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_23 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_prop_exact_accept_equals_sequential/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove chain accept equals sequential decode`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a `policy` or `min_skip` parameter or policies 1 to 3 here (an unused parameter makes the non-test clippy build fail on this commit)
- gpu: none

### 17.24 `sansio_prop_exact_accept_equals_sequential`: the three configured rules

- id: FT17.24
- needs: FT17.23, FT1.10
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `sansio_tests.rs::run_speculative_log`; `proxima-core/src/accept_rule.rs::AcceptRule::accepted_rows` (FT1.10; premise check: `grep -n "fn accepted_rows" proxima-core/src/accept_rule.rs` prints 1 line, else stop and report; generic over the entry, so `u32` works with the similarity `|drafted, chosen| if drafted == chosen { 1.0 } else { 0.0 }`)
- change:
  1. `sansio_tests.rs`: change `run_speculative_log` to `fn run_speculative_log(table, prompt, max_tokens, policy: u8, min_skip: u16) -> (Vec<u32>, u32)` (policy 0 is the chain `accept` of the previous card; the `min_skip` parameter is read only by policy 3 below, so this commit leaves no unused parameter) and extend the same test, whose only caller this is, to `policy in 0..4` and `min_skip in 1..4`: policy 1 uses `AcceptRule { similarity_floor: 1.0, min_run: 0, max_rows: u16::MAX }` (equality), policy 2 uses `AcceptRule { similarity_floor: 1.0, min_run: 0, max_rows: 8 }` (a verifier with a row cap above the draft length), policy 3 uses `AcceptRule { similarity_floor: 1.0, min_run: min_skip, max_rows: u16::MAX }` with the `min_skip` argument. The rule arms call `accepted_rows` for the count and `accept_rows(count, &choices, placements)` to commit, with the same append and resume logic as policy 0. Keep the verify-round guard (sum at least 1).
- test: same test, now 4 policies
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_24 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_prop_exact_accept_equals_sequential/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove every exact accept policy equals sequential`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: weaken policy 0; add a tree accept (it joins when decode-as-data's row-index accept lands)
- gpu: none

### 17.25 `sansio_prop_terminates`

- id: FT17.25
- needs: FT17.24, FT7.1, FT7.2
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `sansio_tests.rs::run_log_driver` (the replay form) and `script`; `proxima-core/src/serving_state/enter.rs::replay_range` (FT7.1; premise check: `git grep -n "pub fn replay_range" -- proxima-core/src/serving_state/enter.rs` prints 1 line, else stop and report); `sansio_tests.rs::periodic_rule` (FT17.12); property 4 in this file
- change:
  1. `sansio_tests.rs`: add `fn sansio_prop_terminates()` through `run_256`, strategy `(1usize..64, prop_oneof![Just(1usize), Just(2), Just(3), Just(4), Just(8), Just(15)], vec(any::<bool>(), 1..8))` (a decision function cannot capture `every`, so the cadence is chosen by a test-local `fn periodic_for(every: usize) -> EnterRule` matching those six values to `periodic_rule::<1, 1>()`, `periodic_rule::<2, 2>()`, `periodic_rule::<3, 3>()`, `periodic_rule::<4, 4>()`, `periodic_rule::<8, 8>()` and `periodic_rule::<15, 15>()`, with `other => panic!("every {other} is outside the strategy")`). The driver: prompt rows `log(&(0..16).map(|index| index % 11).collect::<Vec<u32>>())`, table `next = (entry + 1) % 11` at positions `0..200`, rules `[periodic_for(every)]`, `last_replay_end` starting at 16 and set to the row count after each replay pass, `generated = 1` after the prefill. Loop while `generated < max_tokens`: one iteration is a replay round when `replay_range` fires (the replay form of the log driver: no entry is committed), else a verify round when `draft_flags[iteration % draft_flags.len()]` is true (draft of the next 2 table entries, `accept_rows(2, ..)`, commits 2) else a decode round (commits 1). A hard cap of `4 * max_tokens + 8` iterations fails the case instead of hanging. After the loop call `finish()` and require `Done { .. }`, and require `iterations <= max_tokens + max_tokens / every` with `prop_assert!`.
- test: `sansio_prop_terminates`, 256 cases
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_25 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_prop_terminates/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove decoding terminates within its budget`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: loop without the iteration cap
- gpu: none

### 17.26 `sansio_prop_sealed_immutable`

- id: FT17.26
- needs: FT17.25, FT4.0, FT4.20
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/kv_decision.rs::seal_target` (FT4.0) and `::sealed_blocks` (FT4.20, signature `sealed_blocks(previous_sealed_end: usize, cached_rows: usize, block_tokens: usize, horizon_rows: usize) -> Range<usize>`) (premise check: both exist, else stop; FT4.0 alone does not create `sealed_blocks`); `SPECDIR/tasks-recut/04-seal-and-reserve.md` card FT4.1 (a rewind to `keep < sealed_end` is refused) and card FT4.8 (the interop property this mirrors); property 5 in this file
- change:
  1. `sansio_tests.rs`: add `fn sansio_prop_sealed_immutable()` through `run_256`, strategy `(vec(op, 0..64), prop_oneof![Just(4usize), Just(16), Just(32)], 0usize..4)` with `op` an `Append(1..8)` or a `Rewind(0..40)`. A model drives the rows: `rows: Vec<u32>` where each appended row takes the next value of a running counter (so no value repeats), `sealed_end`, and `sealed_copy: Vec<u32>`. `Append(count)` appends then sets `sealed_end = sealed_end.max(seal_target(rows.len(), block_tokens, horizon_rows))` and `sealed_copy = rows[..sealed_end].to_vec()`. `Rewind(keep)`: when `keep < sealed_end` the rewind is refused and `rows` is unchanged, otherwise `rows.truncate(keep)`. After every op assert `rows[..sealed_end] == sealed_copy[..sealed_end]`, and for every block index `block` in `sealed_blocks(previous_sealed_end, rows.len(), block_tokens, horizon_rows)` assert `(block + 1) * block_tokens + horizon_rows <= rows.len()`.
- test: `sansio_prop_sealed_immutable`, 256 cases
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_26 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_prop_sealed_immutable/)'`
- expect: `1 passed`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove sealed rows never change`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: re-implement `seal_target` in the test; the refusal guard `keep < sealed_end` is a test-side mirror of interop's `try_truncate`, whose own refusal is covered by FT4.1, FT4.8 and FT17.29
- gpu: none

### 17.27 `sansio_prop_one_tier_settles`

- id: FT17.27
- needs: FT17.26, FT17.20
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `sansio_tests.rs::{Agreement, settle_votes}` (FT17.20: the test-local vote rule); `SPECDIR/worked-examples.md` section "one tier settles by vote" (the tier loop cases); `sansio_tests.rs::run_256`; property 6 in this file
- change:
  1. `sansio_tests.rs`: add `fn sansio_prop_one_tier_settles()` through `run_256`, strategy `vec(any::<bool>(), 1..7)`. The tier list is `count - 1` agreements `Agreement { samples: 16, max_disagree: 4 }` then `Agreement { samples: 1, max_disagree: 0 }`. The tier loop is test-local (about 12 lines): a `Vec<Cell<u32>>` of generation counters; for each tier in order bump its counter, then the tier draws its votes (the last tier draws `[1]` whatever its flag; any other tier draws `[15, 1]` when its flag is true and `[8, 8]` when false) and answers when `settle_votes(&votes, agreement)` is `Some(_)`; the last tier always answers because one draw has zero disagreement. Require the answering tier equals the first true flag among the non-last tiers, else the last tier, and the counters are 1 for every tier up to and including it and 0 after, with `prop_assert_eq!`.
- test: `sansio_prop_one_tier_settles`, 256 cases
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_27 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_prop_/)'`
- expect: `6 passed`, names `sansio_prop_log_append_only`, `sansio_prop_rollback_restores`, `sansio_prop_exact_accept_equals_sequential`, `sansio_prop_terminates`, `sansio_prop_sealed_immutable`, `sansio_prop_one_tier_settles`
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): prove exactly one cascade tier settles`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a seventh property; add a tier loop or a vote rule to `src/`
- gpu: none

### 17.28 controls: mutated drivers must fail the checkers

- id: FT17.28
- needs: FT17.27
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `sansio_tests.rs::log_only_grows`, `run_log_driver` (FT17.21), `run_speculative_log`, `try_256`
- change:
  1. `sansio_tests.rs`: change `run_log_driver` to `fn run_log_driver(steps: &[Step], truncate_on_verify: bool)` (parameter added here: when true, after each `Verify` step settles back to `Decode` the driver truncates `cache.rows` to `before.len() - 1` entries, that is it removes a row that was already committed before the step, so `after.starts_with(before)` is false; `before` is never empty here because prefill committed 3 rows, so the first `Verify` step of any generated sequence violates the checker and `Verify` has weight 1 in the step strategy of property 1; the caller in `sansio_prop_log_append_only` passes `false`) and add `scripted_control_truncating_driver_breaks_append_only` (`try_256` over the property 1 strategy with a body that runs `run_log_driver(&steps, true)` and requires every step to satisfy `log_only_grows` returns `Err`) and `scripted_control_skipping_accept_breaks_equality` (a speculative driver variant, built from `run_speculative_log` with a `commit_rejected: bool` parameter added here and false in every existing call: on a `Rollback { to }` with `n` accepted entries it appends `draft[..n]` and then `draft[n]` (the rejected draft entry) in place of `to`; because `draft[n] != to` by the definition of a mismatch, the generated log differs from `run_sequential_log` at that index whenever it falls inside the first `max_tokens` entries; the control runs property 3's strategy and comparison with `commit_rejected = true` and the comparison returns `Err` under `try_256`). These are the controls the gate lacks; named `scripted_control_` so the `sansio_` count stays exact.
- test: the two tests above, each asserting `try_256(..).0.is_err()`
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_28 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/scripted_/)'`
- expect: `7 passed` (3 scripted readout tests, the 2 runner controls and these 2)
- also green: clippy line and no_std alloc check
- stage: `proxima-core/src/serving_state/sansio_tests.rs`
- commit: `test(core): add mutated drivers that the checkers must reject`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: name these `sansio_*`; change a checker to make a control fail
- gpu: none

## phase 6: proxima-model-interop (5 tests, in-memory)

### 17.29 interop scaffold and `sansio_place_seal`

- id: FT17.29
- needs: FT17.28, FT4.0, FT4.1, FT4.21, FT4.22, FT4.3, FT4.16
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first: `proxima-model-interop/src/generate/residency_caches.rs::LayerCache (~line 112 at e4cf9beb)` and its created-by-FT4 items: the field `sealed_end` (FT4.1), `try_truncate` (FT4.22), `seal` (FT4.16, signature `seal(&mut self, even_odd_row: usize, block_tokens: usize, horizon_rows: usize) -> Range<usize>`) and `proxima-model-interop/src/error.rs::InteropError::RewindIntoSealed` (FT4.21) (premise check: all four exist, else stop); `proxima-model-interop/src/generate/mod.rs` (`mod chunked_prefill_tests;` ~line 231 at e4cf9beb: where the new test module is declared); `SPECDIR/tasks-recut/04-seal-and-reserve.md` fixture section (the 13-row trace); `proxima-core/src/kv_decision.rs::seal_target`
- change:
  1. `proxima-model-interop/src/generate/mod.rs`: add `#[cfg(test)] mod sansio_tests;` after the `chunked_prefill_tests` declaration.
  2. `proxima-model-interop/src/generate/sansio_tests.rs` (new): `#[allow(clippy::unwrap_used, clippy::expect_used)]` with the shared comment line, `use proxima_core::kv_decision::seal_target;`, `fn row_at(position: usize) -> ([f32; 2], [f32; 2], [f32; 1])` (even `[p, p + 0.5]`, odd `[-p, -p - 0.5]`, value `[10p]` with `p = position as f32`), and `fn check<Value: PartialEq + core::fmt::Debug>(rows_checked: &mut u32, label: &str, actual: Value, expected: Value)` (`assert_eq!` then count).
- test: add `sansio_place_seal` with a doc comment saying it replays ten appends and three rewinds through a layer cache. On an empty `LayerCache::new()`, after each append call `seal(2, 4, 1)` and after each rewind call `try_truncate(keep, 2, 1)`; `rows = k_even.len() / 2`. The 13 operations and the `(rows, sealed_end)` after each: append position 0 `(1,0)`; 1 `(2,0)`; 2 `(3,0)`; 3 `(4,0)`; 4 `(5,4)`; rewind to 4 `(4,4)`; append position 4 `(5,4)`; 5 `(6,4)`; 6 `(7,4)`; 7 `(8,4)`; 8 `(9,8)`; rewind to 8 `(8,8)`; rewind to 7 refused, `(8,8)`. Each row also asserts `sealed_end` equals the running maximum of `seal_target(rows, 4, 1)` taken after appends only. The refused rewind matches `Err(InteropError::RewindIntoSealed { keep_positions: 7, sealed_end: 8 })` (use `matches!`: `InteropError` has no `PartialEq`) and leaves `k_even.len() == 16`. Count 10 appends and 3 rewinds separately and assert both; `assert_eq!(rows_checked, 13)`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_29 cargo nextest run -p proxima-model-interop --features std -E 'test(/sansio_tests::sansio_place_seal/)'`
- expect: `1 passed`
- also green: interop clippy line
- stage: `proxima-model-interop/src/generate/mod.rs`, `proxima-model-interop/src/generate/sansio_tests.rs`
- commit: `test(interop): prove the seal place on a layer cache`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: enable `metal`; build any example; add a dev-dependency; add a `Cargo.toml` edit
- gpu: none

### 17.30 `sansio_place_place`

- id: FT17.30
- needs: FT17.29, FT17.6, FT17.39, FT5.1, FT5.17, FT5.2, FT5.18, FT5.7
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first: `proxima-model-interop/src/generate/prompt_cache.rs::rule_victim` and `::EvictionRule` (created by FT5.7; `rule_victim` is private there; premise check: its signature is `(candidates: impl Iterator<Item = (u64, bool)> + Clone, rules: &[EvictionRule]) -> Option<u64>`, else stop); `proxima-model-interop/src/block_file.rs::{encoded_len, BlockFileHeader, BlockFileLayer}` (FT5.1), `::encode_block` and `InteropError::BlockFileMalformed` (FT5.17), `::decode_block` (FT5.2) and `::BlockFileView::plane_f32` (FT5.18); `SPECDIR/worked-examples.md` sections "eviction by an ordered rule list" (FT17.6) and "block file byte layout" (FT17.39)
- change:
  1. `proxima-model-interop/src/generate/prompt_cache.rs`: change `fn rule_victim` to `pub(super) fn rule_victim` (visibility only; it is already called by `eviction_victim`, so nothing becomes dead).
  2. `sansio_tests.rs`: add `fn sansio_place_place()` with a doc comment saying it follows eviction by an ordered rule list and the block bytes round trip. 14 rows through `check`, `rule_victim(candidates.iter().copied(), rules)` with `default = [EvictionRule::Branch, EvictionRule::Oldest]` and `oldest = [EvictionRule::Oldest]`: `[(3,false),(5,true),(7,true),(9,false)]` gives `Some(5)` under default and `Some(3)` under oldest; `[(3,false),(9,false)]` gives `Some(3)` under default and `None` under `[EvictionRule::Branch]`; no candidates give `None` (5 rows); hot stamps `{0,1,2}`, `{1,2,3}`, `{2,3,4}`, `{3,4,5}` with no branch give `Some(0)`, `Some(1)`, `Some(2)`, `Some(3)` under default (4 rows); `[(0,false),(1,true),(2,false)]` gives `Some(1)` under default and `Some(0)` under oldest (2 rows). Block bytes (3 rows): `BlockFileHeader { descriptor_digest: [0x22; 16], content_key: 7, base_position: 0, layers: vec![BlockFileLayer { rows: 4, k_even_row_bytes: 8, k_odd_row_bytes: 8, v_row_bytes: 4, ring_window: 0, ring_capacity: 0 }, BlockFileLayer::default()] }` with planes layer 0 `k_even` 8 floats `start + index * 0.5` from 0.0, `k_odd` 8 floats from 100.0, `v` 4 floats from 200.0 and three empty planes for layer 1: `encode_block` gives 176 bytes and `encoded_len(&header) == 176`; `decode_block` then reading every plane of both layers through `plane_f32` and encoding again gives the same 176 bytes and `view.header == header`; dropping the last 4 bytes gives `Err(InteropError::BlockFileMalformed { reason: "payload length disagrees with the header" })` (`matches!`). `assert_eq!(rows_checked, 14)`.
- test: `sansio_place_place`, traces "eviction by an ordered rule list" and "block file byte layout"
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_30 cargo nextest run -p proxima-model-interop --features std -E 'test(/sansio_tests::sansio_place_place/)'`
- expect: `1 passed`
- also green: interop clippy line
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`, `proxima-model-interop/src/generate/sansio_tests.rs`
- commit: `test(interop): prove the tier place decisions`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: write a file; call a function that touches the disk; add a rule that reads a field `CacheEntry` lacks
- gpu: none

### 17.31 `sansio_place_assemble`: assemble order rows

- id: FT17.31
- needs: FT17.30, FT8.2
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first: `proxima-core/src/serving_state/assemble.rs::{assemble_decision, AssembleKind, AssembleDecision, AssembleOrderError}` (created by FT8.2; premise check: the stage kinds are exactly `Prefix` and `Shift`, else stop) and `SPECDIR/tasks-recut/08-assembled-prefill.md` FT8.2's `assemble_plan_` tests (the same rows); `sansio_tests.rs` (FT17.29)
- change:
  1. `sansio_tests.rs`: add `fn sansio_place_assemble()` with a doc comment saying it covers assembly order decisions and selective recompute across check layers. This card adds 7 rows through `check`, calling the core function, not interop-side assembly code: `assemble_decision(&[Prefix, Shift], false, true, true)` is `Ok(Configured)`; `(&[Shift], true, true, true)` is `Err(ShiftWithoutPrefix)`; `(&[], true, true, true)` is `Ok(Legacy { shift: true })`; `(&[Prefix, Shift], true, true, false)` is `Ok(Bypass)`; `(&[Prefix, Prefix], true, true, true)` is `Err(Repeated { kind: AssembleKind::Prefix })`; `(&[Prefix, Shift], true, false, true)` is `Err(ShiftNeedsReuse)`; `(&[], false, false, true)` is `Ok(Bypass)`. `assert_eq!(rows_checked, 7)`; the next card adds rows and raises the count.
- test: `sansio_place_assemble` (7 rows), the assemble list order decisions
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_31 cargo nextest run -p proxima-model-interop --features std -E 'test(/sansio_tests::sansio_place_assemble/)'`
- expect: `1 passed`
- also green: interop clippy line
- stage: `proxima-model-interop/src/generate/sansio_tests.rs`
- commit: `test(interop): prove the assemble place decisions`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: call interop-side assembly code or `rotate_rows` (private, interop side); if a landed name differs, use the landed name and keep the row's meaning
- gpu: none

### 17.32 `sansio_place_assemble`: selective recompute through the assemble place

- id: FT17.32
- needs: FT17.31
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first: `SPECDIR/tasks-recut/09-cacheblend.md` section "dropped" (FT9.1 and FT9.2 are not built: the keep count and the mask index are the technique's own schedule, so this card writes them in the test file) and card FT9.8 (`blend_selection_worked_three_chunks`: the same deviations and sets through a graph; this card runs them through plain Rust); `sansio_tests.rs::sansio_place_assemble`
- change: this card is the proof that the assemble hook plus a test-local selection express selective recompute with no library code; the technique is the test-local functions below.
  1. `sansio_tests.rs`: add three test-local functions (about 20 lines in all). `fn blend_keep_count(ratio_milli: u32, check_index: u32, rows: u32) -> usize`: `ceil(ratio_milli^check_index * rows / 1000^check_index)` capped at `rows`, computed in `u128` with `div_ceil` and converted with `usize::try_from(..).expect("a keep count fits in usize")`. `fn blend_mask_index(layer: u32, check_layers: &[u32]) -> Option<usize>`: `check_layers.iter().filter(|check| **check < layer).count().checked_sub(1)`. `fn recompute_rows(deviation: &[f32], keep_count: usize, last_row: usize) -> Vec<usize>` (the `keep_count` rows of largest deviation, ties to the lower index, plus `last_row`, in ascending order; about 8 lines).
  2. `sansio_tests.rs`: extend `sansio_place_assemble` with 4 rows over 12 rows in 3 chunks, ratio 500 per thousand, check layers `[1, 2, 3]`, deviations `d1 = [0.05,0.90,0.10,0.40,0.02,0.30,0.75,0.08,0.60,0.15,0.04,0.50]`, `d2 = [0.0,0.50,0.0,0.80,0.0,0.20,0.70,0.0,0.10,0.0,0.0,0.60]`, `d3 = [0.0,0.0,0.0,0.30,0.0,0.0,0.90,0.0,0.0,0.0,0.0,0.50]`: `blend_keep_count(500, j, 12)` for `j = 1, 2, 3` is `[6, 3, 2]` (row 8); `recompute_rows(dj, keep_j, 11)` gives `[1,3,5,6,8,11]`, `[3,6,11]`, `[6,11]` (row 9); `blend_mask_index(layer, &[1,2,3])` for layers `0..=4` is `[None, None, Some(0), Some(1), Some(2)]` (row 10); the rows recomputed per layer, `None` meaning all 12 rows and `Some(index)` the length of the selection at that index, are `[12, 12, 6, 3, 2]` (row 11). `assert_eq!(rows_checked, 11)`.
- test: `sansio_place_assemble` (11 rows), trace "selective recompute across check layers" (the shared 12-row, 3-chunk selection)
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_32 cargo nextest run -p proxima-model-interop --features std -E 'test(/sansio_tests::sansio_place_assemble/)'`
- expect: `1 passed`
- also green: interop clippy line
- stage: `proxima-model-interop/src/generate/sansio_tests.rs`
- commit: `test(interop): drive selective recompute through the assemble place`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: put `recompute_rows`, `blend_keep_count`, `blend_mask_index` or any other blend arithmetic in `src/` or in `proxima-core`; change a deviation or a set to make a row pass (a mismatch is a false premise: stop and report)
- gpu: none

### 17.33 `sansio_place_read_host`

- id: FT17.33
- needs: FT17.32, FT4.0
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first: `SPECDIR/tasks-recut/06-read-sets.md` item 11 of its drift list (FT6.1 and FT6.2 are dropped: the keep count and the row count are block read selection arithmetic, so this card writes them in the test file); `proxima-core/src/kv_decision.rs::seal_target` (FT4.0; premise check: `grep -n "fn seal_target" proxima-core/src/kv_decision.rs` prints 1 line, else stop and report; the symbol is absent on main at e4cf9beb); `SPECDIR/tasks-recut/06-read-sets.md` section "shared worked values" (the block scoring, the block selection and the row count instance); `sansio_tests.rs`
- change:
  1. `sansio_tests.rs`: add the test-local `fn block_read_keep_count(sealed_blocks: usize, local_blocks: usize, keep_ratio_milli: u32, min_blocks: u32) -> usize` (`local = local_blocks.min(sealed_blocks)`, `nonlocal = sealed_blocks - local`, `proportional = (keep_ratio_milli as usize * nonlocal).div_ceil(1000)`, result `(min_blocks as usize).max(proportional).min(nonlocal)`; write the two casts with `usize::try_from(..).expect(..)`) and `fn block_read_row_count(sealed_blocks: usize, local_blocks: usize, keep_count: usize, block_tokens: usize, tail_rows: usize) -> usize` (`(keep_count + local_blocks.min(sealed_blocks)) * block_tokens + tail_rows`), and add `fn block_scores(query: [i32; 2], mins: &[[i32; 2]], maxes: &[[i32; 2]]) -> Vec<i32>` (per block, the sum over the two dimensions of `max(query_j * max_j, query_j * min_j)`) and `fn attended_blocks(scores: &[i32], keep_count: usize, local_blocks: usize) -> Vec<usize>` (the `keep_count` highest-scoring blocks among the non-local ones, ties to the lower index, plus the last `local_blocks` blocks, ascending). Add `fn sansio_place_read_host()` with a doc comment saying it follows the block read selection over four sealed blocks and a five-row tail. 8 rows through `check`: `block_scores([2,-1], mins [[0,0],[-2,-1],[1,-3],[-1,-1]], maxes [[1,1],[0,3],[2,-1],[1,1]])` is `[2,1,7,3]` (row 1); `block_read_keep_count(4, 1, 500, 1)` is 2 (row 2); `attended_blocks(&[6,9,4,1], 2, 1)` is `[0,1,3]` (row 3); the attended rows, 16 per block over `[0,1,3]` plus the tail rows `64..69`, equal `(0..32).chain(48..69)` (row 4); `block_read_row_count(4, 1, 2, 16, 5)` is 53 and equals that list's length (row 5); `seal_target(69, 16, 5)` is 64, so the tail is 5 rows (row 6); for block size 64, horizon 64, minimum 16 blocks, 1 local block, prompt 4096, steps of length 4097 and 4098, 36 layers: the total over both steps of `block_read_row_count(sealed / 64, 1, block_read_keep_count(sealed / 64, 1, keep, 16), 64, len - sealed)` with `sealed = seal_target(len, 64, 64)` times 36 is 83052 at keep 100 per thousand (row 7) and 267372 at keep 900 per thousand (row 8). `assert_eq!(rows_checked, 8)`.
- test: `sansio_place_read_host`, traces "block read selection" and the row count instance
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_33 cargo nextest run -p proxima-model-interop --features std -E 'test(/sansio_tests::sansio_place_read_host/)'`
- expect: `1 passed`
- also green: interop clippy line
- stage: `proxima-model-interop/src/generate/sansio_tests.rs`
- commit: `test(interop): prove the host read selection`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: run a GPU kernel arm; add selection code or the two read counts to `src/` or to `proxima-core`; change a worked number to make a row pass
- gpu: none

### 17.34 `sansio_place_schedule`

- id: FT17.34
- needs: FT17.33, FT17.5, FT10.4, FT10.7
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first: `proxima-model-interop/src/generate/prewarm_gate.rs::PrewarmGate (~line 22 at e4cf9beb)` with `enter_request (~line 57)` and the created-by-FT10.7 `run_idle` (premise check: `run_idle(job: impl FnOnce(&dyn Fn() -> bool) -> T) -> Option<T>` exists, else stop); `proxima-model-interop/src/generate/prewarm_follow_up.rs::{IdleStep, DraftStep}` (created by FT10.1 to FT10.4: `IdleStep::default_list`, `DraftStep { branches, max_tokens, temperature_milli, lead, keep }`); `proxima-model-interop/src/generate/prewarm_gate.rs` test `a_request_waits_for_the_prewarm_to_yield_and_the_prewarm_sees_it (~line 118 at e4cf9beb)` (the spin-on-an-atomic pattern, no sleeps); `SPECDIR/worked-examples.md` section "idle steps around a waiting request"
- change:
  1. `sansio_tests.rs`: add `fn run_from(gate: &PrewarmGate, steps: &[IdleStep], ran: &mut Vec<IdleStep>) -> usize` (runs the steps in order, each inside `gate.run_idle(|_poll| ran.push(step.clone()))`, stops at the first `None`, returns the number finished). Add `fn sansio_place_schedule()` with a doc comment saying it follows the idle steps around a waiting request. 5 rows through `check`, each on a fresh `PrewarmGate::new()`:
     - the default list `IdleStep::default_list(&PromptCacheConfig::standard())`: `run_from` finishes 2 and `ran` equals the list (row 1);
     - the list `[Draft(a), Prewarm, Draft(b)]` with `a = DraftStep { branches: 3, max_tokens: 16, temperature_milli: 700, lead: Vec::new(), keep: false }` and `b` the same with `max_tokens: 64`: finishes 3 and `ran` equals the list by value (row 2);
     - the empty list: finishes 0 and `ran` is empty (row 3);
     - `let request = gate.enter_request();` before the pass: `run_from` over the default list finishes 0 and `ran` is empty; after `drop(request)` the same list finishes 2 (row 4);
     - a request arrives during step 1: inside `std::thread::scope`, a worker walks the default list with `enumerate`, each step in `gate.run_idle`, and in the first step's job stores `true` in a shared `AtomicBool` then spins on `poll()` (`core::hint::spin_loop`) until it is true, then pushes the step to its own `Vec`; the worker stops at the first `None` and returns `(ran, stopped_at)`; the main thread spins until the flag is set, takes `gate.enter_request()` (it returns once the worker's job releases the slot), joins the worker, then drops the request. The worker returned `ran == [Prewarm]` and `stopped_at == Some(1)`; then `run_from(&gate, &steps[1..], &mut resumed)` finishes 1 and `resumed` equals `[Draft(..)]` (row 5).
     `assert_eq!(rows_checked, 5)`.
- test: `sansio_place_schedule`, trace "idle steps around a waiting request"
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_34 cargo nextest run -p proxima-model-interop --features std -E 'test(/sansio_tests::sansio_/)'`
- expect: `5 passed`, names `sansio_place_seal`, `sansio_place_place`, `sansio_place_assemble`, `sansio_place_read_host`, `sansio_place_schedule`
- also green: interop clippy line
- stage: `proxima-model-interop/src/generate/sansio_tests.rs`
- commit: `test(interop): prove the idle schedule place`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a `schedule` module or a `Next` type to `proxima-core`; change `PrewarmGate`; add a sleep; load a model
- gpu: none

## phase 7: the guard and the gate

### 17.35 add the sans-IO no-IO guard example

- id: FT17.35
- needs: FT17.34
- budget: 20 min
- crate(s): proxima-core (features: none; an example, std only)
- read first: the "no-IO guard" section of this file (the three commands, verbatim); `ls proxima-core/src/serving_state/` and `git ls-files proxima-core/src/serving_state.rs 'proxima-core/src/serving_state/*'` (the two counts the example must reconcile); `proxima-core/Cargo.toml` (no `[[example]]` entry is needed: cargo finds `examples/*.rs`)
- change:
  1. `proxima-core/examples/sansio_guard.rs` (new; the owner's rule is Rust only, so this replaces a shell script). `fn main() -> std::process::ExitCode`, no `unwrap` or `expect` (the workspace denies both). It reads the optional arguments `--pathspec <path>` (repeatable; replaces the default list `proxima-core/src/serving_state.rs` and `proxima-core/src/serving_state/*`), runs `git ls-files -- <pathspecs>` through `std::process::Command` from the repository root, counts the tracked files, reads each one and counts the lines that contain any of these 11 substrings: `std::fs`, `std::net`, `std::io`, `std::env`, `std::thread`, `std::time`, `File::`, `TcpStream`, `tokio`, `libc::`, `println!` (the six `std::` forms are the alternation of the guard pattern); it also counts the `.rs` files on disk under `proxima-core/src/serving_state/` (recursive walk) plus `proxima-core/src/serving_state.rs`. It prints `files=<tracked> on_disk=<disk> io_hits=<hits>` and exits success only when `tracked >= 3`, `tracked == on_disk` and `hits == 0`; on failure it prints a message naming both counts and exits with failure. A function `fn count_hits(text: &str) -> usize` holds the matching.
- test: add two tests in the example file (`cargo nextest` runs example tests): `guard_flags_every_forbidden_token` (a text with one line per each of the 11 substrings gives `count_hits == 11`) and `guard_ignores_core_and_alloc_paths` (a text with the lines `use core::cell::Cell;`, `use alloc::vec::Vec;` and `let readout = scripted_readouts(&table, rows_of, &entries, &positions, &cache);` gives 0).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_17_35 cargo nextest run -p proxima-core --example sansio_guard`
- expect: `2 passed`
- also green: clippy line and no_std alloc check; then from the repository root `cargo run -p proxima-core --example sansio_guard` prints `files=<N> on_disk=<N> io_hits=0` with N at least 3 and exits 0, logged to `.long_ctx_backups/fsm/cards17/17.35-run.txt`; and the control `cargo run -p proxima-core --example sansio_guard -- --pathspec proxima-core/src/serving_state_missing.rs` prints `files=0` with the count message and exits 1 (a zero cannot pass for an empty pathspec), logged to `17.35-control.txt`
- stage: `proxima-core/examples/sansio_guard.rs`
- commit: `test(core): add the sans-io no-io guard example`
- done when: the expect line printed, the run printed `io_hits=0` with N at least 3, the control exited 1, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: write the guard as a shell script; loosen the substring list; edit a `serving_state` file; if `io_hits` is nonzero, report the file and line list and stop
- gpu: none

### 17.36 run the conformance gate and record the result

- id: FT17.36
- needs: FT17.35 and every other slice (TASKS.md slice 17 "follows every other slice here")
- budget: 20 min
- crate(s): proxima-core (features: alloc), proxima-model-interop (features: std)
- read first: `SPECDIR/SPEC.md` AC31 and R23b (the commands and expected line this card records, and the two places this card edits); `SPECDIR/TASKS.md` slice 17 (lines ~368-385: the file list, `Expected:` and `Done:`); this file's "spec drift" items 1, 3 and 4
- change:
  1. Run, in order, logging each to `.long_ctx_backups/fsm/cards17/17.36-<n>.txt` (n counts the five bullets from 1; bullet 3 writes its check log to `17.36-3.json` and its grep count to `17.36-3.txt`; the two commands of step 2 write `17.36-6.txt` and `17.36-7.txt`):
     - `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_/)'`
     - `cargo nextest run -p proxima-model-interop --features std -E 'test(/sansio_tests::sansio_/)'`
     - `cargo check -p proxima-core --no-default-features --features alloc,sansio-script --message-format=json > /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards17/17.36-3.json`, then `grep -c '"kind":\["lib"\].*"name":"proxima_core".*"features":\[[^]]*"sansio-script"' /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards17/17.36-3.json` (command 3 is the pair; the count the grep prints is the result)
     - `git ls-files proxima-core/src/serving_state.rs 'proxima-core/src/serving_state/*' | wc -l`
     - `git grep -nE 'std::(fs|net|io|env|thread|time)|File::|TcpStream|tokio|libc::|println!' -- proxima-core/src/serving_state.rs 'proxima-core/src/serving_state/*' | wc -l`
  2. Then, beyond the gate: `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/scripted_/)'` and `cargo run -p proxima-core --example sansio_guard`.
  3. `proxima-tensor/specs/fsm-techniques/TASKS.md`, slice 17: in the file list change the `scripted.rs` line to say the module is `#[cfg(any(test, feature = "sansio-script"))]` and holds the scripted readouts function, the `sansio_tests.rs` line to `13 tests (transitions 2, places 5, properties 6)`, and the interop line to `5 tests (seal, place, assemble, host read, schedule)`; set `Expected: 13 passed; 5 passed; 0.`; set `Done: [x]` and append the printed result lines of step 1 and step 2 (command, then the count line it printed), plain text.
  4. `proxima-tensor/specs/fsm-techniques/SPEC.md`: in R23b replace "propose, shape, accept, commit, enter, schedule and settle in `proxima-core`," with "propose, accept, commit, enter and settle in `proxima-core` (the shape place belongs to the host shape pipes slice of the decode-as-data spec, which proves it with its own tests)," and "and seal, place, assemble and the host half of read in `proxima-model-interop` over" with "and seal, place, assemble, schedule and the host half of read in `proxima-model-interop` over"; in the placement rule replace "the enter rule, the schedule choice and every settle" with "the enter rule and every settle" and add after "judge." the sentence "The idle schedule is the one exception: its decision is the configured step list indexed by the count of finished steps, behind the request gate's waiting check, and lives in interop with the gate."; in AC31 replace "then `cargo check -p proxima-core --no-default-features --features alloc,sansio-script`" with "then `cargo check -p proxima-core --no-default-features --features alloc,sansio-script --message-format=json` written to a log, and a count of the log's library compiler-artifact records whose feature list names `sansio-script`", "Expected: 15 passed, then 4 passed" with "Expected: 13 passed, then 5 passed", "then exit 0, then a file count" with "then 1, then a file count", "proxima-core, 15 tests:" with "proxima-core, 13 tests:", "7 place tests: `sansio_place_propose`, `_shape`, `_accept`, `_commit`, `_enter`," with "5 place tests: `sansio_place_propose`, `_accept`, `_commit`, `_enter`,", "`_schedule`, `_settle`;" with "`_settle`;", "proxima-model-interop, 4 tests:" with "proxima-model-interop, 5 tests:" and "`sansio_place_assemble`, `sansio_place_read_host`, over in-memory tiers and the" with "`sansio_place_assemble`, `sansio_place_read_host`, `sansio_place_schedule`, over in-memory tiers and the".
  5. Then check that the two spec-file edits landed, from the repository root, each logged to `.long_ctx_backups/fsm/cards17/17.36-<n>.txt` (n = 8 to 12 in this order). The file text is first normalised so a line wrap cannot split a pattern: `tr -s ' \n' ' ' < <file>`. The expected counts are derived from the step 3 and step 4 edit lists (11 SPEC.md replacements, 5 TASKS.md edits, 7 result lines), not from a run.
     - 8: `tr -s ' \n' ' ' < proxima-tensor/specs/fsm-techniques/SPEC.md | grep -o -F -e 'propose, accept, commit, enter and settle in `proxima-core` (the shape place belongs' -e 'and seal, place, assemble, schedule and the host half of read in `proxima-model-interop` over' -e 'the enter rule and every settle' -e 'The idle schedule is the one exception:' -e '--features alloc,sansio-script --message-format=json` written to a log' -e 'Expected: 13 passed, then 5 passed' -e 'then 1, then a file count' -e 'proxima-core, 13 tests:' -e '5 place tests: `sansio_place_propose`, `_accept`, `_commit`, `_enter`, `_settle`;' -e 'proxima-model-interop, 5 tests:' -e '`sansio_place_read_host`, `sansio_place_schedule`, over in-memory tiers' | wc -l`
     - 9: `tr -s ' \n' ' ' < proxima-tensor/specs/fsm-techniques/SPEC.md | grep -o -F -e 'propose, shape, accept, commit, enter, schedule and settle' -e 'the schedule choice' -e 'Expected: 15 passed, then 4 passed' -e 'then exit 0, then a file count' -e 'proxima-core, 15 tests:' -e '7 place tests:' -e '`_schedule`, `_settle`;' -e 'proxima-model-interop, 4 tests:' | wc -l`
     - 10: `sed -n '/^### slice 17:/,/^Cards:/p' proxima-tensor/specs/fsm-techniques/TASKS.md | tr -s ' \n' ' ' | grep -o -F -e 'feature = "sansio-script"' -e '13 tests (transitions 2, places 5, properties 6)' -e '5 tests (seal, place, assemble, host read, schedule)' -e 'Expected: 13 passed; 5 passed; 0.' -e 'Done: [x]' | wc -l`
     - 11: `sed -n '/^### slice 17:/,/^Cards:/p' proxima-tensor/specs/fsm-techniques/TASKS.md | grep -c -F -e 'Done: [ ]' -e 'Expected: 15 passed; 4 passed; 0.' -e '15 tests (transitions 2' -e '4 tests (seal, place, assemble, host'`
     - 12: `sed -n '/^### slice 17:/,/^Cards:/p' proxima-tensor/specs/fsm-techniques/TASKS.md | grep -c '^Result: '` (step 3 writes each appended result as its own line beginning `Result: `: the five step 1 commands, then the two step 2 commands, 7 lines)
- test: none new; the assertions are the step 5 counts (8: 11, 9: 0, 10: 5, 11: 0, 12: 7), which fail when any replacement is missing or an old string remains, plus the recorded run counts of steps 1 and 2
- validate: the five bullets of step 1, the two commands of step 2, and the five commands of step 5
- expect: step 5 prints `11`, `0`, `5`, `0`, `7` in order (command 11 exits 1 because grep prints 0 for no match; that exit is the expected one), and `13 passed`, then `5 passed`, then `1` (the count of library compiler-artifact records naming `sansio-script`; `0` means the check failed or never built the library with the feature), then a file count of at least 3, then `0` IO matches. Step 2: `7 passed` (3 scripted readout tests and 4 controls), and `files=<N> on_disk=<N> io_hits=0` where both N equal the tracked file count that bullet 4 printed (at least 3)
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`, `cargo clippy -p proxima-model-interop --features std --all-targets`, `cargo check -p proxima-core --no-default-features --features alloc`
- stage: `proxima-tensor/specs/fsm-techniques/TASKS.md`, `proxima-tensor/specs/fsm-techniques/SPEC.md`
- commit: `docs(fsm-techniques): record sans-io conformance gate results`
- done when: 13, 5, 1, a file count of at least 3 and 0 matches printed, the step 2 counts printed, the step 5 counts 11, 0, 5, 0 and 7 printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change a test to make a count match; if any count differs, stop and report the names
- gpu: none (the llama parity counts are run by every slice's validation per TASKS.md and are not part of this card)

## spec drift

Each item names the artifact that showed it, the effect, and the card that carries it.

1. The counts moved from 15 and 4 to 13 and 5. Artifacts: `tasks-recut/10-idle-jobs.md` header ("this slice adds no module to `proxima-core`"; the next idle job is `steps.get(completed)` behind `PrewarmGate::request_waiting`) and item 4 below. Effect: the schedule place test moved to interop (core 14, interop 5); SPEC R23b's placement rule listed "the schedule choice" as a core function; dropping the shape place takes core to 13. Carried by FT17.34 (the test) and FT17.36 (the SPEC and TASKS edits, exact strings).
2. `refine` (decode-as-data, a `Verify -> Verify` step) is in no card file and has no fixed method name. The tables and counts here (18 legal rows, 40 illegal pairs, 9 guards) are the machine as slices 1, 7 and 8 land it; FT17.10 and FT17.11 stop and report if the landed machine differs.
3. Anchors. On main e4cf9beb the machine is `proxima-model-interop/src/serving_fsm.rs`, `#![allow(dead_code)]` at line 38, and `ServingFsmError::NotSupported` is still defined at ~line 87; FT1.2 does not copy it, so after slice 1 the only variant is `IllegalTransition`. `proxima-core/src/serving_state*`, `kv_decision.rs` and `examples/` do not exist on main, and no card creates `read_decision.rs`, `settle.rs` or `serving_state/blend.rs`. The original test `speculation_matches_plain_greedy_for_thirty_two_tokens` is at ~line 527; FT1.4 renames it.
4. The shape place has no test in this suite. Artifacts: `proxima-tensor/specs/decode-as-data/TASKS.md` slice 2 (one line, "Host shape pipes ...", no card file) and `grep -l 'tree_shape\|chain_shape' tasks-recut/*.md` (this file only). `Shape`, `chain_shape` and `tree_shape` are created by no card, so the earlier FT17.2 and FT17.13 could not run; both are dropped (see "dropped"). If a card file for that slice creating them lands, a derivation and a place test join this suite then, and a tree row joins FT17.14 and the exact-accept property when decode-as-data's row-index accept lands.
5. The cascade slice's re-cut (`tasks-recut/12-cascade.md`) drops FT12.1, FT12.2 and FT12.8 and changes no `proxima-core` file, so no `proxima-core/src/settle.rs`, `Agreement` or `settle_votes` exists in any library. FT17.20 writes the vote rule as a test-local function over a test-local `Agreement` record and FT17.27 reuses it, so neither card depends on a symbol another card must create. The earlier wording of this file (`Judge::{Always, Conformal}` and `settle` in `serving_state/settle.rs`, `validate_tier_list`, then `Agreement::check`) named symbols no card creates. The tier list validation (`Agreement::check`) has no row. The threshold, classifier and isotonic judges and the router are techniques and have no row here. The same holds for the blend keep count and mask index (`tasks-recut/09-cacheblend.md` drops FT9.1 and FT9.2, FT17.32) and the block read counts (`tasks-recut/06-read-sets.md` drops FT6.1 and FT6.2, FT17.33): each is a test-local function in `sansio_tests.rs`. `tasks-recut/15-niah-read-arms.md` still names FT6.1 and FT6.2; this file does not edit it.
6. `Verify.snapshot` is written by `enter_verify` and read by nothing (FT1.2 keeps it). Property 2 is stated against the chosen placement, not the snapshot; a later slice that makes Rollback restore the snapshot would change that property.
7. Property 1's replay re-encodes dense rows equal to the committed rows, so no row inside the range changes in the driver; the checker is written for the general case (the rows inside the range may change) and FT17.18 proves the changing case. FT17.28 shows the checker rejects a truncating driver.
8. The seal property's refusal guard (`keep < sealed_end`) is mirrored in the test; the real refusal is interop's `try_truncate`, covered by FT4.1, FT4.8 and by FT17.29.
9. The gate's controls: FT17.9 (a runner that asserts exactly 256 cases and a failing body that must fail) and FT17.28 (two mutated drivers that must fail their checkers), named `scripted_control_` so the `sansio_` count stays exact; and FT17.35's missing-pathspec run that must exit 1.
10. The feature `sansio-script` has no consumer outside `proxima-core`: the interop tests do not use the scripted backend (see "dropped"). Its reason to exist is the non-test build the spec requires (`cargo check -p proxima-core --no-default-features --features alloc,sansio-script`), which proves the module cannot touch std IO.
11. Test models. Nothing here loads a model: the places are proved against a scripted table and in-memory structures, so there is no gemma4 or granite arm in this slice. The numerics of the dense and MoE checkpoints are proved by their slices' oracle cards.

## slice exit

Slice 17 is complete when all of these printed, from a tree where slices 0-16 are merged:
- `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/sansio_tests::sansio_/)'` prints `13 passed`;
- `cargo nextest run -p proxima-model-interop --features std -E 'test(/sansio_tests::sansio_/)'` prints `5 passed`;
- `cargo check -p proxima-core --no-default-features --features alloc,sansio-script --message-format=json` written to a log, then `grep -c '"kind":\["lib"\].*"name":"proxima_core".*"features":\[[^]]*"sansio-script"'` over the log prints `1`;
- `git ls-files proxima-core/src/serving_state.rs 'proxima-core/src/serving_state/*' | wc -l` prints at least `3`, and the `git grep` over the same pathspec (pattern in the no-IO guard section) prints `0`;
- `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/scripted_/)'` prints `7 passed` (3 scripted readouts, 4 controls);
- `cargo run -p proxima-core --example sansio_guard` prints `io_hits=0` with equal `files` and `on_disk` counts of at least 3, and its missing-pathspec control exits 1;
- each of the 6 property tests ran exactly 256 cases (the in-runner count assertion);
- `worked-examples.md` holds the nine English-titled sections of cards FT17.1, FT17.3 to FT17.7 and FT17.37 to FT17.39;
- TASKS.md slice 17 carries `Done: [x]` with the printed result lines, and SPEC.md R23b and AC31 carry the 13 and 5 counts (card FT17.36).
