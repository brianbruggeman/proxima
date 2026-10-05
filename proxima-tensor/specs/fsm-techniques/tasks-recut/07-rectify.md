# slice 7 cards (re-cut): the verify entry as configuration, and the periodic replay hook

anchors read at main 4b4be6cf (full sha 4b4be6cf78bcd536fecdd4e4620a9769b9b7364c). Read each with
`git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`; the working tree is not the source.
Paths are relative to the proxima repo root. Every `~line N` is a hint; re-locate by symbol. Line hints were read at 4b4be6cf; the existence of
every path below was re-checked against main a7c08c4c with `git cat-file -e main:<path>`.

A path or symbol marked `(created by FTx.y; absent on main)` does not exist on main, so `git show main:<path>` cannot read it. The executor
reads it in its own checkout after the card named there has landed, by symbol, and stops and reports if it is still absent.

Expected counts: where a card records a count `N` before editing, `N` must be greater than 0. A filter that matches nothing prints no tests
(a count of 0): the executor stops and reports, it does not proceed to the edit.

Rules: CARDS.md applies to every card. Every card uses `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_<n>` and removes it
when done. Logs go under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards07/`.

## what this slice builds now

The earlier cut built the technique (a periodic dense re-encode of the last rows). Owner direction: build the hooks, not the technique.
This cut builds three hook changes on the step stage. The library holds no periodic rule, no cadence and no row count of its own: the
technique is a small decision function that a proof test writes and hands to the hook.

- The enter rule list: a configured list of rules says when a step does something other than plain decoding. The rules are
  `DraftNonempty` (verify the drafters' draft, today's behaviour, loadable by name) and `Replay { max_rows, decide }`, where `decide` is a
  caller-supplied function `fn(committed_len, last_replay_end) -> Option<usize>` that names how many of the newest committed rows to
  re-encode, and `max_rows` bounds that answer (the kv slack and the sealed-rows refusal read the bound, never the function). A function
  cannot be named in TOML or an environment variable, so a `Replay` rule is composed in Rust and configuration cannot load one. A list-valued stage cannot
  be a `Vec` field of the `Copy` serving config, so the list is a borrowed slice, `&'model [EnterRule]`, exactly as the existing
  `weight_precision` list (`proxima-model-interop/src/serving.rs::ServingConfig.weight_precision`, `&'model [WeightPrecisionRule<'model>]`,
  ~line 909 at 4b4be6cf) and the `model_path` borrow (`::ServingConfig.model_path`, `&'model str`). No new container type.
- The replay pass: a pure function in `proxima-core` (`replay_range`) runs the configured decisions and names which committed rows a step
  re-encodes first; the decode loop re-encodes whatever range it names through the verify program, rewinds then appends the same rows,
  emits no token, and goes on with the step. The pass knows nothing about cadence: it asks the rule list.
- The verify state carries the entry that is waiting, and one transition resumes from it (the sans-IO twin of the replay pass, for the
  conformance suite).

Technique proof (FT7.8, and the real-model test of FT7.6): a decision function written in the test, "once 16 rows are committed since the last
pass, re-encode the newest 8", composed into `decode.enter` as a `Replay` rule, and a row comparison against a dense prefill. The periodic
rule lives in those tests and nowhere in the library.

Test models (owner, 2026-10-04): gemma4 E2B (`gemma4:e2b-it-qat`, dense) for every model-loading test here. No qwen of any kind. A `Replay`
rule needs a verify program; a model that has none is refused by FT7.6 with a typed error (a check on the capability, not on a model name)
rather than silently ignored; no model without one is loaded here.

## old-to-new id map

| earlier id | verdict | new id | note |
|---|---|---|---|
| FT7.1 `rectify_proposal` | recut | FT7.1 | the proposer is `replay_range`, a pure function that runs the decisions of a configured rule list |
| FT7.2 `choose_proposer` / `EnterRule::Rectify` | recut | FT7.0, FT7.3, FT7.3a | generic rules `DraftNonempty` and `Replay { max_rows, decide }` (the `EnterRule` enum is FT7.0, so FT7.1 adds one public item); the list is configuration with a default that reproduces today, and the periodic rule is a decision a test supplies |
| FT7.3 `rectify_overwrite_range` | dropped | none | see "dropped" |
| FT7.4 live rectify wiring | recut | FT7.2, FT7.4, FT7.5, FT7.6, FT7.7 | one card per hook change |
| FT7.5 rectified rows equal dense | recut | FT7.8 | gemma4 E2B only; no other checkpoint |

## dropped

- earlier FT7.3 (`rectify_overwrite_range`): the range it checks is the range `replay_range` returns, so the function is a tautology; the
  commit is rewind-then-append through the kv rewind that already exists (`LayerCache::truncate`, which FT4.22 wraps as `try_truncate`), so
  no `overwrite_rows` is added either.

## cross-file contract (the executor of a card stops when its premise is false)

- Slice 1: `proxima_core::serving_state` exists with `ServingState<Entry, Cache>` (FT1.2, FT1.3, FT1.7), and interop depends on
  `proxima-core` in its `std` feature (FT1.5).
- Slice 2 (`tasks-recut/02-serving-settings.md`): `ServingConfig` has the `attention` and `prefill` sections (FT2.1, FT2.2) and no `decode`
  field; `ServingSettings` (FT2.15, its `as_serving_config` literal names every field) has `from_json`, `round_trip::assert_three_ways` and
  the `kv` section (FT2.9), and `refusals()` with `ServingRefusal` (FT2.16); there is no `decode` section, no `kv.seal`, and no card FT2.11.
  This file creates `DecodeConfig` and `ServingConfig.decode` (FT7.3) and `DecodeSettings` and `ServingSettings.decode` (FT7.4), and owns
  `decode.enter` (FT7.3, FT7.3a, FT7.4, FT7.7). If a card still carries a rectify key, a `samples` field or a `kv.seal` read, the executor
  stops and reports; that card is what gets re-cut, not this one.
- Slice 4: `LayerCache::try_truncate` (FT4.22) and `InteropError::RewindIntoSealed` (FT4.21), and `PromptCacheConfig::seal_horizon_rows` with
  `PromptCacheSettings::seal_horizon_rows` (FT4.2), the one home of the seal horizon.
- Slice 0 (`tasks-recut/00-oracles-and-worked-examples.md`): `proxima-tensor/specs/fsm-techniques/worked-examples.md` is created by FT0.14
  and holds section `## row tolerance` from FT0.20 (`RESULT row tolerance: gemma4_e2b=5.841e-05 gemma4_26b=4.649e-05 granite_moe=2.861e-05`).
  Neither exists on main; FT7.8 `needs` both.

## shared worked values

The decisions below are functions the tests write; the library holds none of them.

Decision `every_32_replay_32(committed_len, last_replay_end)` is `Some(32)` when `committed_len - last_replay_end >= 32` and `None` otherwise;
rule list `[Replay { max_rows: 32, decide: every_32_replay_32 }]`, prompt of 4096 rows, one row committed per call, `last_replay_end`
starting at 4096 and set to the committed length after each pass. Walking `committed_len` over `4096..=4224`:
- `replay_range` fires at exactly 4128, 4160, 4192 and 4224, returning `4096..4128`, `4128..4160`, `4160..4192`, `4192..4224`;
- every other committed length returns `None`.

Rule order, `[Replay { max_rows: 8, decide: every_32_replay_8 }, Replay { max_rows: 4, decide: every_16_replay_4 }]`, where `every_32_replay_8` is
`Some(8)` once 32 rows are committed since the last pass and `every_16_replay_4` is `Some(4)` once 16 are, `last_replay_end = 4096`:
- committed 4112: the first rule has 16 rows since the last pass (needs 32), the second fires: `Some(4108..4112)`;
- committed 4128: the first rule fires: `Some(4120..4128)`.

A decision outside its bounds does not fire: `Some(0)`, `Some(100000)` at committed 4128, and `Some(40)` under `max_rows: 32` all give `None`, and the
next rule in the list is tried.

Step limit, one number for the kv slack and the step buffer: the wider of the widest drafter width and the widest `max_rows`:
- speculation off, `[Replay { max_rows: 32, .. }]`: `Some(32)`;
- ngram-mod `n_max = 64` and `max_rows` 32: `Some(64)`; `max_rows` 128: `Some(128)`; nothing configured: `None`.

Resume after a replay: committed log `[10, 11, 12, 13]`, entry waiting `14`, replay draft `[12, 13]`, row caches after rows 12 and 13 are
`cache_3` and `cache_4`: the state resumes as `Decode { last: 14, cache: cache_4 }`.

## cards

### 7.0 the enter rule enum with a caller-supplied replay decision

- id: FT7.0
- needs: FT1.2
- budget: 20 min
- crate(s): proxima-core (features: alloc; serde for the json test)
- read first:
  - `proxima-core/src/serving_state.rs` (created by FT1.2; absent on main): the module the new file joins, and how its `pub mod` lines are declared;
  - `proxima-core/Cargo.toml`: the optional `serde` dependency (`dep:serde`, declared with `derive` and `alloc`) that only the `config` feature enables today, and `[dev-dependencies]`;
  - `proxima-model-interop/src/generate/drafter.rs::DrafterSet` (~line 123 at 4b4be6cf): the one working configured-list stage, the shape this rule list follows.
- change:
  1. `proxima-core/src/serving_state/enter.rs` (new): `#[derive(Debug, Clone, Copy)] #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(tag = "kind", rename_all = "snake_case"))] pub enum EnterRule { DraftNonempty, #[cfg_attr(feature = "serde", serde(skip))] Replay { max_rows: u32, decide: fn(usize, usize) -> Option<usize> } }`, with `impl PartialEq for EnterRule` written by hand (two `DraftNonempty` are equal; two `Replay` are equal when `max_rows` is equal and `core::ptr::fn_addr_eq(left_decide, right_decide)`; any other pair is unequal) and `impl Eq for EnterRule {}`. The comparison is by hand because a derived comparison of a function pointer trips the function-pointer comparison lint, and `fn_addr_eq` is the lint's own answer: two functions with identical bodies may share an address, which only ever makes two rules that behave the same compare equal.
     Docs (English): `DraftNonempty` means a step verifies the draft the drafters made. `Replay` means that before the next step the decision is asked, with the committed length and the committed length at the end of the previous replay pass, how many of the newest committed rows to re-encode through the verify pass (`None` means none); `max_rows` is the most rows the decision may name, which the kv slack and the sealed-rows check read. The decision is code, so configuration cannot name it: a caller composes it in Rust, and the library ships no decision of its own. Name the primitives: the verify pass and the kv rewind.
  2. `proxima-core/src/serving_state.rs`: add `pub mod enter;`.
  3. `proxima-core/Cargo.toml`: add the feature line `serde = ["dep:serde"]` (manual: a feature line), and run `cargo add serde_json --dev -p proxima-core` (workspace dependency).
- test: two tests in `enter.rs` `tests` (names avoid `serving_state_`), with two test-local functions `fn never_replays(_committed_len: usize, _last_replay_end: usize) -> Option<usize> { None }` and `fn always_replays_one(_committed_len: usize, _last_replay_end: usize) -> Option<usize> { Some(1) }` (bodies differ, so the compiler cannot merge them):
  - `enter_rule_names_round_trip_through_json` (`#[cfg(feature = "serde")]`, `serde_json`): `{"kind":"draft_nonempty"}` equals `EnterRule::DraftNonempty` and `to_string` returns the same text; sad: `{"kind":"replay"}` is `Err` (a function is not loadable by name), `{"kind":"periodic","every_rows":32}` is `Err`, `{"kind":"never"}` is `Err`, and `serde_json::to_string(&EnterRule::Replay { max_rows: 8, decide: never_replays })` is `Err`;
  - `enter_rule_equality_compares_the_decision_and_its_bound`: `Replay { max_rows: 8, decide: never_replays }` equals itself; differs from `Replay { max_rows: 9, decide: never_replays }`, from `Replay { max_rows: 8, decide: always_replays_one }` and from `DraftNonempty`; `DraftNonempty == DraftNonempty`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_0 cargo nextest run -p proxima-core --no-default-features --features alloc,serde -E 'test(/enter_rule_/)'`
- expect: `2 passed` (`enter_rule_names_round_trip_through_json` and `enter_rule_equality_compares_the_decision_and_its_bound`; with `--features alloc` alone only the second exists, so the count line is the serde one)
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc,serde --all-targets`; `cargo check -p proxima-core --no-default-features --features alloc`
- stage: `proxima-core/src/serving_state/enter.rs`, `proxima-core/src/serving_state.rs`, `proxima-core/Cargo.toml`, `Cargo.lock`
- commit: `feat(core): add the enter rule enum for decode steps`
- done when: the expect line printed, clippy and check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a function over the rules (FT7.1); add a decision function, a cadence or a row count to the library (every decision belongs to a caller); add a `ServingState` variant or transition; add a type for the rule list; touch `proxima-model-interop`.
- gpu: none

### 7.1 `replay_range`, which committed rows a step re-encodes first

- id: FT7.1
- needs: FT7.0
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-core/src/serving_state/enter.rs::EnterRule` (created by FT7.0; absent on main): the enum the function reads, and the file it joins;
  - this file's shared worked values.
- change:
  1. `proxima-core/src/serving_state/enter.rs`: add
     `pub fn replay_range(rules: &[EnterRule], committed_len: usize, last_replay_end: usize) -> Option<core::ops::Range<usize>>`: walk `rules` in list order; for each `Replay { max_rows, decide }`, call `decide(committed_len, last_replay_end)`; the first answer `Some(rows)` with `rows > 0`, `rows <= max_rows as usize` and `rows <= committed_len` gives `Some(committed_len - rows..committed_len)`; an answer outside those bounds, or `None`, makes that rule not fire and the walk goes on; no rule fires gives `None`; `DraftNonempty` never fires here (a caller reads it as `rules.contains(&EnterRule::DraftNonempty)`).
     Doc (English): names the primitives it feeds, the verify pass and the kv rewind, and says it holds no rule of its own: it runs the decisions the caller composed, so the same rule list gives the same rows on every run, and the rows named are always the newest committed rows because a replay is a rewind then an append.
- test: in `enter.rs` `tests` (names avoid `serving_state_`), with test-local decisions `fn every_32_replay_32`, `fn every_32_replay_8`, `fn every_16_replay_4` (each returns `Some(rows)` once `committed_len.saturating_sub(last_replay_end)` reaches its first number, otherwise `None`), `fn names_too_many` (always `Some(100000)`) and `fn names_zero` (always `Some(0)`):
  - `enter_replay_range_runs_the_supplied_decision`: rules `[Replay { max_rows: 32, decide: every_32_replay_32 }]`, `last_replay_end = 4096`, walk `committed_len` over `4096..=4224` resetting `last_replay_end = committed_len` after each `Some`; the collected ranges equal `[4096..4128, 4128..4160, 4160..4192, 4192..4224]` and no other length returned `Some`;
  - `enter_replay_range_bounds_and_rule_order`: `names_zero` never fires over `4096..=4300`; `names_too_many` never fires at `committed_len = 4128`; `Replay { max_rows: 4, decide: every_32_replay_32 }` (answers 32, over its bound of 4) is `None` at 4128 with `last_replay_end = 4096`; the empty list is `None`; `[DraftNonempty]` is `None` at every length in `0..=4300`; for `[Replay { max_rows: 8, decide: every_32_replay_8 }, Replay { max_rows: 4, decide: every_16_replay_4 }]` with `last_replay_end = 4096`: `replay_range(.., 4112, 4096) == Some(4108..4112)` and `replay_range(.., 4128, 4096) == Some(4120..4128)`; a rule that answers out of bounds is skipped: `[Replay { max_rows: 32, decide: names_too_many }, Replay { max_rows: 8, decide: every_32_replay_8 }]` at `4128` gives `Some(4120..4128)`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_1 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/enter_replay_range_/)'`
- expect: `2 passed` (the two names above)
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc,serde --all-targets`; `cargo check -p proxima-core --no-default-features --features alloc`
- stage: `proxima-core/src/serving_state/enter.rs`
- commit: `feat(core): choose committed rows to re-encode from a rule list`
- done when: the expect line printed, clippy and check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a decision function, a cadence or a periodic rule to the library; add a `ServingState` variant or transition; add a type for the rule list; change `EnterRule`; touch `proxima-model-interop`.
- gpu: none

### 7.2 the verify state carries the waiting entry, and one transition resumes from it

- id: FT7.2
- needs: FT1.7
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-core/src/serving_state.rs::ServingState::Verify` and `::enter_verify` (created by FT1.2; absent on main; at main these are `proxima-model-interop/src/serving_fsm.rs::ServingState::Verify` ~line 59 and `enter_verify` ~line 158, which discards `last`);
  - `proxima-core/src/serving_state.rs::accept_rows` (created by FT1.7; absent on main): the sibling transition, which resumes from `draft[n - 1]` and so cannot serve a replay;
  - the test fake cache used by FT1.7's `accept_rows_takes_declared_count_not_prefix` (`FakeCache::empty().advanced_by(n)`);
  - this file's resume worked value.
- change:
  1. `proxima-core/src/serving_state.rs`: the `Verify` variant gains `last: Entry` (the entry that was waiting when verification was entered); `enter_verify` fills it from the `Decode { last, .. }` it consumes; every existing `Verify { .. }` pattern and the `Verify` literals in the tests gain `last` or `..` (the compiler lists them).
     Add `pub fn accept_replay(self, row_caches: Vec<Cache>) -> Result<Self, ServingFsmError>`: `Verify { draft, last, .. }` with `!draft.is_empty()` and `row_caches.len() == draft.len()` gives `Ok(Decode { last, cache })` where `cache` is the last element of `row_caches` (taken with `into_iter().next_back()`, no index); any other input, or any other state, is `Err(IllegalTransition { attempted: "accept_replay" })`.
     Doc (English): a replayed draft re-encodes entries the cache already holds, so no entry is chosen and decoding resumes from the entry that was waiting; it is the state machine's form of the periodic replay pass.
- test: add `accept_replay_resumes_from_the_waiting_entry` in `serving_state.rs` `tests` (the name avoids `serving_state_`): `start(vec![10_u32, 11, 12, 13], FakeCache::empty())`, `advance_prefill(14, FakeCache::empty().advanced_by(4))`, `enter_verify(vec![12, 13])`; with `cache_3 = FakeCache::empty().advanced_by(3)` and `cache_4 = FakeCache::empty().advanced_by(4)`:
  - `accept_replay(vec![cache_3, cache_4])` is `Ok(Decode { last: 14, cache: cache_4 })`;
  - with `vec![cache_3]` only (one cache for two draft entries) it is `Err(IllegalTransition { attempted: "accept_replay" })`;
  - after `enter_verify(vec![])`, `accept_replay(vec![])` is `Err(IllegalTransition { attempted: "accept_replay" })`;
  - `accept_replay` on a `Decode` state is `Err(IllegalTransition { attempted: "accept_replay" })`.
- validate: record N, the count printed by `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_2 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/serving_state_|accept_rows_|accept_replay_/)'` BEFORE editing; after editing run the same command
- expect: before: `5 passed` (derived by reading `tasks-recut/01-fsm-generic-entry.md`, not run: four `serving_state_*` tests, 01-fsm-generic-entry.md:73, plus `accept_rows_takes_declared_count_not_prefix`, which FT1.7 states as `5 passed` for the filter `serving_state_|accept_rows_` at 01-fsm-generic-entry.md:441; if the printed N is not 5, or is 0, stop and report); after: `6 passed` (the new test is the only addition)
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`; `cargo check -p proxima-core --no-default-features --features alloc`
- stage: `proxima-core/src/serving_state.rs`
- commit: `feat(core): resume decoding from the waiting entry after a replay`
- done when: the expect lines printed, clippy and check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the result of any existing transition test; add a state variant; touch other crates.
- gpu: none

### 7.3 add the decode section with the enter rule list to the serving config

- id: FT7.3
- needs: FT7.0, FT1.5, FT2.15
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/serving.rs::ServingConfig` (~line 720 at 4b4be6cf; `attention` and `prefill` are its last fields after FT2.1 and FT2.2) and `impl Default for ServingConfig<'static>` (~line 1136): the struct that gains a field, the existing borrowed-slice field `weight_precision` (~line 909) as the shape for a list inside the `Copy` config, and the six full struct literals with no `..` (the default impl, `tests::fully_supported_config_applies_without_error` and the four `via_full_literal` literals);
  - `proxima-model-interop/src/serving_settings.rs::ServingSettings::as_serving_config` (file created by FT2.3, literal completed by FT2.15; absent on main): its literal names every field with no base, so a new field is `E0063` there until it is named;
  - `proxima-model-interop/src/generate/prompt_cache_key.rs::CacheKey::of` (~line 82) and `proxima-model-interop/src/generate/resident_plans.rs::PlanIdentity::of` (~line 80): the two exhaustive destructures of `ServingConfig`, each of which names a field either bound or `_` with the reason.
- change:
  1. `proxima-model-interop/Cargo.toml`: `EnterRule` lives in `proxima-core`, which interop reaches only through `std` today, and `serving.rs` is alloc-tier. Run `cargo add proxima-core -p proxima-model-interop --no-optional --features alloc` (then `git diff` must show the line still reads `workspace = true`; if cargo rewrote it to a version, restore `workspace = true`), and remove `"dep:proxima-core"` from the `interop-bgpool` feature line (a feature line). If `cargo check -p proxima-model-interop --no-default-features` fails because `proxima-core` pulls a std-only default, the executor stops and reports.
  2. `proxima-model-interop/src/serving.rs`: `use proxima_core::serving_state::enter::EnterRule;`; add `#[derive(Debug, Clone, Copy, PartialEq)] pub struct DecodeConfig<'model> { pub enter: &'model [EnterRule] }` (field doc, semantic only: "ordered rules that say when a step does something other than plain decoding; the default list verifies the draft the drafters made, which is today's behaviour"), `impl Default for DecodeConfig<'static>` with `enter: &[EnterRule::DraftNonempty]`; append `pub decode: DecodeConfig<'model>` to `ServingConfig` after `prefill`; add `decode: DecodeConfig::default()` to `impl Default` and to the other five full literals (the compiler lists them).
  3. `proxima-model-interop/src/serving_settings.rs`: name `decode: DecodeConfig::default()` in the `as_serving_config` literal (FT7.4 replaces this line with the loaded value).
  4. `proxima-model-interop/src/generate/prompt_cache_key.rs`: add `decode: _,` to the `CacheKey::of` destructure of `ServingConfig`, with the line `// the enter rules choose which steps run and no step changes what a row holds`.
  5. `proxima-model-interop/src/generate/resident_plans.rs`: add `decode: _,` to the `PlanIdentity::of` destructure, with the line `// the enter rules choose which steps run, not how a plan is lowered`.
- test: add `decode_enter_defaults_to_drafting_only` in `serving.rs` `tests`: `DecodeConfig::default().enter == [EnterRule::DraftNonempty]` and `ServingConfig::default().decode == DecodeConfig::default()`; with `ServingConfig { decode: DecodeConfig { enter: &[] }, ..ServingConfig::default() }` the two are not equal (the field participates in `PartialEq`).
- validate: record N, the count printed by `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_3 cargo nextest run -p proxima-model-interop --features std -E 'test(/^serving::tests::/)'` BEFORE editing; after editing run the same command
- expect: before: `N passed` with N greater than 0 (a 0 means the filter matched nothing: stop and report); after: `N+1 passed`, and `serving::tests::decode_enter_defaults_to_drafting_only` appears in the output
- also green: `cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_settings_default_parity|serving_settings_round_trip/)'` prints `2 passed` (FT2.15's two parity tests still agree with the new field); `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --no-default-features`; `cargo check -p proxima-model-interop --features std,metal --all-targets` (compiles every `ServingConfig { .. }` literal in examples and benches)
- stage: `proxima-model-interop/src/serving.rs`, `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/generate/prompt_cache_key.rs`, `proxima-model-interop/src/generate/resident_plans.rs`, `proxima-model-interop/Cargo.toml`, `Cargo.lock` (only if it changed)
- commit: `feat(interop): add the decode section with the enter rule list`
- done when: the expect lines printed, clippy and both checks clean, `git diff --cached --stat` equals the stage list (four source files, one over the limit of three: FT7.3 is an admitted exception in CARDS.md; see "size-rule exception" below), and the commit landed with that message
- do not: add a `Vec` field to `ServingConfig`; add a `samples` field; export `DecodeConfig` or `EnterRule` from `lib.rs` (FT7.4 does); gate or refuse any value (FT7.3a); touch the drafters or `decode.rs`.
- gpu: none
### 7.3a configure when decoding verifies a draft

- id: FT7.3a
- needs: FT7.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std; metal for the real-model test)
- read first:
  - `proxima-model-interop/src/serving.rs::DecodeConfig` (FT7.3) and `::apply_serving_config` (~line 1236 at 4b4be6cf): the place that refuses a configuration value this build does not run yet, with `InteropError::UnsupportedServingConfig(String)`;
  - `proxima-model-interop/src/generate/decode.rs`, the `speculative_enabled` binding (~line 3535 at 4b4be6cf): the one inline condition that decides whether a step may draft;
  - `proxima-model-interop/src/generate/speculative_default_on_tests.rs::assert_default_matches_off` (~line 83): the real-model parity guard the default must keep passing.
- change:
  1. `proxima-model-interop/src/serving.rs`: in `apply_serving_config`, refuse a configuration whose `decode.enter` contains a `Replay` rule with `InteropError::UnsupportedServingConfig(..)`: the message is English and names `decode.enter`, saying a replay rule needs the decode loop to re-encode committed rows, which this build does not run yet.
  2. `proxima-model-interop/src/generate/decode.rs`: next to `speculative_enabled`, `let drafting_enabled = serving_config.decode.enter.contains(&EnterRule::DraftNonempty);` (import `EnterRule` from `proxima_core::serving_state::enter`) and `speculative_enabled = (...) && drafting_enabled && rings_cover_speculation(..)`. A list without `DraftNonempty` therefore never drafts or verifies.
- test: two tests.
  - `a_replay_rule_is_refused_until_the_loop_runs_it` in `serving.rs` `tests`, with a test-local `fn never_replays(_committed_len: usize, _last_replay_end: usize) -> Option<usize> { None }`: with `supported_default()`, setting `decode: DecodeConfig { enter: &[EnterRule::Replay { max_rows: 8, decide: never_replays }] }`, `apply_serving_config(&config, 1)` is `Err(InteropError::UnsupportedServingConfig(message))` with `message.contains("decode.enter")`; with the default decode section it is `Ok(())`;
  - `an_enter_list_without_drafting_never_verifies_on_real_gemma4_e2b` in `generate/speculative_default_on_tests.rs` (`#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]`, like its neighbours): it opens, maps, parses and loads the checkpoint once, exactly as `assert_default_matches_off` does (~lines 83-93), then with `greedy_config()` and `decode: DecodeConfig { enter: &[] }` the `decode(&model, config)` helper returns ids equal to the ids of `decode(&model, config.with_speculative(SpeculativeConfig::none()))` and stats equal to `SpeculativeDecodeStats::default()`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_3a cargo nextest run -p proxima-model-interop --features std,metal -j 1 --run-ignored all -E 'test(/a_replay_rule_is_refused_until_the_loop_runs_it|an_enter_list_without_drafting_never_verifies_on_real_gemma4_e2b/)'`
- expect: `2 passed` (the two new tests; only `an_enter_list_without_drafting_never_verifies_on_real_gemma4_e2b` loads a model, once; the default list keeping today's drafting is re-run by the slice exit, not here)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`; `cargo check -p proxima-model-interop --no-default-features`
- stage: `proxima-model-interop/src/serving.rs`, `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/src/generate/speculative_default_on_tests.rs`
- commit: `feat(interop): configure when decoding verifies a draft`
- done when: the expect line printed, clippy and the check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a `Vec` field to `ServingConfig`; touch the drafters; make the loop run a `Replay` rule (the refusal is what keeps it honest until the replay pass lands); touch the cache key.
- gpu: one run (`-j 1`, one model load: the one no-drafting test), waiting for a quiet box (CARDS.md machine safety: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` prints nothing).

### 7.4 load the decode entry rules from serving settings

- id: FT7.4
- needs: FT7.3, FT2.9
- budget: 20 min
- crate(s): proxima-model-interop (features: std, conflaguration)
- read first:
  - `proxima-model-interop/src/serving_settings/kv.rs::KvSettings` (file created by FT2.9; absent on main): the section file to copy (derives, `#[serde(default)]`, `impl Default` from the builder), whose `eviction: Vec<EvictionRule>` with `resolve_with = "from_json"` is the shape for a list field; and `proxima-model-interop/src/serving_settings/prefill.rs::PrefillSettings::as_prefill_config` (created by FT2.12; absent on main), the borrow of a list into a `Copy` config section;
  - `proxima-model-interop/src/serving_settings.rs::round_trip::assert_three_ways` (file created by FT2.3, helper by FT2.27; absent on main): the helper that proves TOML, environment and builder agree;
  - `proxima-model-interop/src/serving.rs::DecodeConfig` (FT7.3).
- change:
  1. `proxima-model-interop/src/serving_settings/decode.rs` (new; `mod decode;` plus `pub use decode::DecodeSettings;` in `serving_settings.rs`; `use super::from_json;` and `use proxima_core::serving_state::enter::EnterRule;`): `DecodeSettings { enter: Vec<EnterRule> }` with the section derives of `KvSettings` (`#[derive(Debug, Clone, PartialEq, Builder, Deserialize, Serialize, Settings)]`, `#[builder(derive(Clone, Debug))]`, `#[serde(default)]`, `impl Default` = `Self::builder().build()`); the field carries `#[setting(resolve_with = "from_json", default_str = "[{\"kind\":\"draft_nonempty\"}]")]`, builder default `vec![EnterRule::DraftNonempty]`; doc: "the ordered list of rules for when a step does more than plain decoding". `pub(super) fn as_decode_config(&self) -> DecodeConfig<'_> { DecodeConfig { enter: &self.enter } }` (`pub(super)`: only the parent module `serving_settings.rs` calls it, and a private method in a child module is not callable from its parent; its first production caller is the `as_serving_config` line in step 2; add `use crate::serving::DecodeConfig;` to the file). Environment key `PROXIMA_SERVING_DECODE_ENTER` (a JSON list).
  2. `proxima-model-interop/src/serving_settings.rs`: append `#[setting(nested)] #[builder(default)] pub decode: DecodeSettings` to `ServingSettings`, and in `as_serving_config` replace the `decode: DecodeConfig::default()` line (FT7.3) with `decode: self.decode.as_decode_config()`.
  3. `proxima-model-interop/src/lib.rs`: `pub use proxima_core::serving_state::enter::EnterRule;`, `DecodeConfig` added to the `pub use serving::{..}` list, and `DecodeSettings` added to the `pub use serving_settings::{..}` line.
  4. `proxima-model-interop/Cargo.toml`: add `"proxima-core/serde"` to the `std` feature list (a feature line), so `EnterRule` derives serde where settings load it.
- test: add `serving_settings_decode_enter_variants` in `serving_settings/decode.rs` `tests`: `assert_three_ways` twice. (1) TOML `[decode]` with `enter = []`; environment `PROXIMA_SERVING_DECODE_ENTER=[]`; the builder sets `.enter(vec![])`; then assert `as_serving_config(&[]).decode.enter.is_empty()`. (2) TOML `[[decode.enter]] kind = "draft_nonempty"` equals `ServingSettings::default()`, and `ServingSettings::default().as_serving_config(&[]).decode.enter == [EnterRule::DraftNonempty]`. Sad: `kind = "replay"` is `Err` (a replay decision is a function, composed in Rust, so configuration cannot name it); `kind = "periodic"` is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_4 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_decode_enter_variants/)'`
- expect: `1 passed` (`serving_settings_decode_enter_variants`)
- also green: `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_default_parity|serving_settings_round_trip/)'` prints `2 passed` (the default list loads as `[DraftNonempty]`, equal to `DecodeConfig::default()`); `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/decode.rs`, `proxima-model-interop/src/serving_settings.rs`, `proxima-model-interop/src/lib.rs`, `proxima-model-interop/Cargo.toml`
- commit: `feat(interop): load the decode entry rules from serving settings`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list (one new file, two small edits to existing source files, and a feature line), and the commit landed with that message
- do not: add a refusal (the sealed-rows refusal is its own card); add a `samples` or rectify key; add a registry of named decisions so configuration can load a `Replay` rule (a function is composed in Rust); derive serde on `DecodeConfig`.
- gpu: none

### 7.5 count the replay rows in the kv slack and the step buffer

- id: FT7.5
- needs: FT7.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/kv_ring.rs::speculative_draft_limit` (~line 222) and `::ring_slack_rows` (~line 243): the one reader of how many rows a step may write past the last committed row, and the slack built from it;
  - `proxima-model-interop/src/generate/decode.rs`: the two call sites of `speculative_draft_limit`, `draft_slack` (~line 2542, the memory fit) and `draft_limit` (~line 3149, which also sizes `device_kv_step_rows` at ~line 3565 and feeds `rings_cover_speculation`);
  - `proxima-model-interop/src/generate/prompt_cache_key.rs::CacheKey::of` (~line 82): the exhaustive destructure whose `decode: _` line (FT7.3) carries a comment this card replaces.
- change:
  1. `proxima-model-interop/src/generate/kv_ring.rs`: add `pub(super) fn step_row_limit(config: &ServingConfig<'_>, forced_draft_width: Option<u16>) -> Option<usize>`: the larger of `speculative_draft_limit(&config.speculative, forced_draft_width)` and the largest `max_rows` among the `Replay` rules of `config.decode.enter`; `None` when neither exists. `ring_slack_rows` calls it in place of `speculative_draft_limit`.
  2. `proxima-model-interop/src/generate/decode.rs`: the two call sites call `step_row_limit(serving_config, ..)` instead of `speculative_draft_limit(&serving_config.speculative, ..)` (same other arguments).
  3. `proxima-model-interop/src/generate/prompt_cache_key.rs`: change the comment above `decode: _` to the lowercase line `// the enter rules reach the key as ring slack rows, through step_row_limit`.
- test: add two tests in `kv_ring.rs` `tests`:
  - `step_row_limit_is_the_wider_of_draft_and_replay_rows`: with a test-local `fn never_replays(_committed_len: usize, _last_replay_end: usize) -> Option<usize> { None }`, `SpeculativeConfig::none()` and `[Replay { max_rows: 32, decide: never_replays }]` it is `Some(32)`; with the ngram-mod config of `draft_limit_is_the_widest_enabled_drafters_bound` (`n_max = 64`) and `max_rows` 32 it is `Some(64)`, and with `max_rows` 128 it is `Some(128)`; with `SpeculativeConfig::none()` and the default enter list it is `None`; with `forced_draft_width = Some(6)` and `max_rows` 32 it is `Some(32)`;
  - `ring_slack_covers_the_replay_rows`: with `prompt_cache: PromptCacheConfig::off()`, speculation none and `[Replay { max_rows: 64, decide: never_replays }]`, `ring_slack_rows(&config, None) == 64`; with the same config and the default enter list it is `0`.
- validate: record N, the count printed by `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_5 cargo nextest run -p proxima-model-interop --features std -E 'test(/kv_ring::tests::/)'` BEFORE editing; after editing run the same command
- expect: before: `N passed` with N at least 12 (derived, not run: `kv_ring.rs` holds 12 `#[test]` functions in its `tests` module at main a7c08c4c, and an earlier card may have added more; a printed 0, or fewer than 12, means stop and report); after: `N+2 passed`, with `step_row_limit_is_the_wider_of_draft_and_replay_rows` and `ring_slack_covers_the_replay_rows` in the output (the existing draft-limit tests keep passing unchanged)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `cargo check -p proxima-model-interop --features std,metal`
- stage: `proxima-model-interop/src/generate/kv_ring.rs`, `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/src/generate/prompt_cache_key.rs`
- commit: `feat(interop): count replay rows in the kv slack and step buffer`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `speculative_draft_limit` or its tests; add a field to the cache key; run a replay (the loop card does).
- gpu: none

### 7.6 re-encode recent committed rows on a periodic rule

- id: FT7.6
- needs: FT7.1, FT7.3a, FT7.5, FT4.21, FT4.22
- budget: 20 min (see "not cuttable")
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/src/generate/decode.rs`: the step closure passed to `decode_until_stop_or_budget` (~line 3596 at 4b4be6cf): the `pending` pop at its top, the `speculative_step` and `speculative_ids` bindings (~line 3680), the batch loop that picks `ids_for_step` and the verify program (~line 3700-3800), the K/V append (`append_at` ~line 5522, `append` ~line 5562) and the verify commit that follows (`cache.truncate(keep_positions, ..)` ~line 5818; on main only `LayerCache::truncate` exists, at `generate/residency_caches.rs` ~line 155, and `LayerCache::try_truncate` is created by FT4.22 (its error variant by FT4.21) and absent on main, so the card reads it in its own checkout);
  - `proxima-model-interop/src/generate/residency_caches.rs::SpeculativeDecodeStats` (~line 3206, `pub`, derives `Default` and `PartialEq`, exported from `lib.rs`, built only through `Default::default()` at every call site on main) and `::record_verify_step`: the counters a test reads;
  - `proxima-model-interop/src/serving.rs::apply_serving_config`: the periodic-rule refusal FT7.3a added (absent on main), which this card replaces;
  - `proxima-model-interop/src/generate/decode.rs::draft_limit` (FT7.5): the slack that already covers the replay rows.
- change:
  1. `generate/residency_caches.rs`: `SpeculativeDecodeStats` gains the one public field `pub replay_rows: u64` (zero by `Default`; doc in English: rows re-encoded by replay passes, summed over one decode call). No method is added: the loop adds to the field directly, as `record_verify_step` does for its counters.
  2. `serving.rs`: delete the `Replay` refusal from `apply_serving_config` and its test (`a_replay_rule_is_refused_until_the_loop_runs_it`); add `pub(crate) fn replay_refusal(enter: &[EnterRule], has_verify_program: bool) -> Result<(), InteropError>`: `Err(UnsupportedServingConfig(..))` (English, naming `decode.enter` and saying a replay rule needs a model with a verify program) when `enter` has a `Replay` rule and `!has_verify_program`, otherwise `Ok(())`.
  3. `generate/decode.rs`: (a) before the loop, `replay_refusal(serving_config.decode.enter, self.speculative_verify_program.is_some())?;` and `let mut last_replay_end: Option<usize> = None;` beside `pending`; (b) at the top of the step closure, after the `pending` pop, `let mut replayed = false;` and `'step: loop {` around the rest of the body (the loop adds no re-indentation; rustfmt is not run on src files); (c) when `!replayed && next_ids.len() == 1 && cached_len > 0 && self.speculative_verify_program.is_some() && rings_cover_speculation(&layer_caches, draft_limit)`, set `let since_base = *last_replay_end.get_or_insert(cached_len);` and `let replay = replay_range(serving_config.decode.enter, cached_len, since_base);`; (d) when `replay` is `Some(range)`: rewind every attention layer with `cache.try_truncate(range.start, even_odd_row, v_row)?`, set `cached_len = range.start`, clear `speculative_draft`, and run this iteration as a verify step whose `speculative_ids` are `token_history[range.clone()]` only (no waiting entry in front), so the verify program re-encodes exactly `range.len()` rows at positions `range.start..range.end`; (e) right after the K/V append for that iteration and before the verify selection, `if let Some(range) = replay { cached_len = range.end; last_replay_end = Some(range.end); replayed = true; debug!(start = range.start as u64, rows = range.len() as u64, "replay_pass"); continue 'step; }`, with one more line inside that block: when stats are requested, `speculative_stats.replay_rows += range.len() as u64`. No token is sampled or emitted by the pass; `next_ids` is untouched, so the second iteration of the same step decodes the waiting entry as usual.
     If the replay iteration needs the evaluation to run on a path where the host never sees the rows (the device-resident decode path), the rewind is `cached_len` going back and the verify evaluation overwrites the same device rows; if that path cannot be driven this way, the executor stops and reports.
- test: two tests (the new `replay_rows` field is read by the real-model test below, which is its first reader; `residency_caches.rs` has no `tests` module on main and none is created).
  - `replay_without_a_verify_program_is_refused` in `serving.rs` `tests`: `replay_refusal(&[EnterRule::Replay { max_rows: 8, decide: never_replays }], false)` (a test-local `never_replays`, as in the card that added the earlier refusal test) is `Err(UnsupportedServingConfig(message))` with `message.contains("decode.enter")`; the same rule with `true` is `Ok(())`; `replay_refusal(&[EnterRule::DraftNonempty], false)` is `Ok(())`; the empty list with `false` is `Ok(())`;
  - `a_periodic_replay_keeps_greedy_ids_and_counts_its_rows_on_real_gemma4_e2b` in `generate/speculative_default_on_tests.rs` (`#[ignore = ..]` as its neighbours; it loads the checkpoint once, as `assert_default_matches_off` does, and calls `decode` twice): `greedy_config()` with `speculative: SpeculativeConfig::none()` and `decode.enter = &[EnterRule::Replay { max_rows: 8, decide: every_sixteen_rows_replay_eight }]`, where the test-local `fn every_sixteen_rows_replay_eight(committed_len: usize, last_replay_end: usize) -> Option<usize> { (committed_len.saturating_sub(last_replay_end) >= 16).then_some(8) }` is the whole technique, against the same config with `decode.enter = &[]`, 40 tokens, repeated paragraph prompt: the two id lists are equal, the replay run's `replay_rows == 16` (one pass of 8 rows at committed length prompt + 16 and one at prompt + 32), and the other run's stats equal `SpeculativeDecodeStats::default()`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_6 cargo nextest run -p proxima-model-interop --features std,metal -j 1 --run-ignored all -E 'test(/replay_without_a_verify_program_is_refused|a_periodic_replay_keeps_greedy_ids_and_counts_its_rows_on_real_gemma4_e2b/)'`
- expect: `2 passed` (the two new tests; only `a_periodic_replay_keeps_greedy_ids_and_counts_its_rows_on_real_gemma4_e2b` loads a model, once, and decodes twice on it; the default-speculation tests and the no-drafting test are re-run by the slice exit, not here)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`; `cargo check -p proxima-model-interop --no-default-features`
- stage: `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/src/serving.rs`, `proxima-model-interop/src/generate/residency_caches.rs`, `proxima-model-interop/src/generate/speculative_default_on_tests.rs`
- commit: `feat(interop): re-encode recent committed rows on a periodic rule`
- done when: the expect line printed, clippy and the check clean, `git diff --cached --stat` equals the stage list (three source files plus the test file), and the commit landed with that message
- do not: add a `ServingState` variant; sample or emit a token in the pass; add a field to the plan key or the cache key; add a model-specific path (a model without a verify program is refused by `replay_refusal`); put a cadence or a row count in the library (the decision is the test's).
- gpu: one run (`-j 1`, one model load: the one periodic-replay test), waiting for a quiet box (CARDS.md machine safety).

### 7.7 refuse a replay that reaches sealed rows

- id: FT7.7
- needs: FT7.4, FT2.16, FT4.2
- budget: 20 min
- crate(s): proxima-model-interop (features: std, conflaguration)
- read first:
  - `proxima-model-interop/src/serving_settings/refusal.rs::ServingRefusal` and `::field_path` (file created by FT2.16; absent on main): the variant style, and `field_path` per variant;
  - `proxima-model-interop/src/serving_settings/refusals.rs::ServingSettings::refusals` (file created by FT2.16; absent on main): the `check_*` method style (`check_block`);
  - `proxima-model-interop/src/prompt_cache_settings.rs::PromptCacheSettings.seal_horizon_rows` (the file exists on main, the field does not: FT4.2 adds it, a `u32`, default 256), reached as `ServingSettings.prompt_cache.seal_horizon_rows`: how many rows behind the newest row a full block must be before it is sealed, so the deepest rewind a replay may make.
- change:
  1. `refusal.rs`: add `#[error("decode.enter replays {replay_rows} rows, but only the last {horizon_rows} rows stay unsealed")] ReplayReachesSealedRows { replay_rows: u32, horizon_rows: u32 }`; `field_path` returns `"decode.enter"` for it.
  2. `refusals.rs`: add `check_enter(&self, out)` pushing `ReplayReachesSealedRows { replay_rows, horizon_rows: self.prompt_cache.seal_horizon_rows }` once, where `replay_rows` is the largest `max_rows` among the `Replay` rules of `self.decode.enter`, when that value is greater than `self.prompt_cache.seal_horizon_rows`; call it from `refusals()`.
- test: add `serving_settings_refuses_replay_reaching_sealed_rows` in `refusals.rs` `tests`, with a test-local `fn never_replays(_committed_len: usize, _last_replay_end: usize) -> Option<usize> { None }` and `ServingSettings.decode.enter` set from rules `Replay { max_rows, decide: never_replays }`: `max_rows = 32` with `prompt_cache.seal_horizon_rows = 256`: `refusals()` is empty; `max_rows = 256` with horizon 256: empty; `max_rows = 300` with horizon 256: `vec![ReplayReachesSealedRows { replay_rows: 300, horizon_rows: 256 }]` and `validate()` is `Err(Validation { errors })` with `errors.len() == 1` and `errors[0].path == "decode.enter"`; two rules with `max_rows` 8 and 300: one refusal naming 300; `ServingSettings::default().refusals()` is empty.
- validate: record N, the count printed by `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_7 cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_/)'` BEFORE editing; after editing run the same command
- expect: before: `N passed` with N greater than 0 (slice 2 leaves many `serving_settings_*` tests; a printed 0 means stop and report); after: `N+1 passed`, with `serving_settings_refuses_replay_reaching_sealed_rows` in the output
- also green: `cargo clippy -p proxima-model-interop --features std,conflaguration --all-targets`
- stage: `proxima-model-interop/src/serving_settings/refusal.rs`, `proxima-model-interop/src/serving_settings/refusals.rs`
- commit: `feat(interop): refuse a replay that reaches sealed rows`
- done when: the expect lines printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: refuse by attention read or by model (those are not properties of the hook); add a descriptor-dependent row; add a `kv.seal` section (the horizon is `prompt_cache.seal_horizon_rows`).
- gpu: none

### 7.8 the technique through the hooks: replayed rows match a dense prefill

- id: FT7.8
- needs: FT7.6, FT7.5, FT0.14, FT0.20
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/src/generate/prompt_cache_chunk_shift_real_model_tests.rs::the_squash_stores_the_lifted_rows_in_every_layer_not_the_fresh_ones` (~line 725), `::prefill_state` (~line 51) and `::rows_of` (~line 569): how a real-model test reads one layer's stored rows after a run and builds a dense prefill state to compare with;
  - `proxima-model-interop/src/generate/prefix_resume_long_prompt_tests.rs::corpus_document` and `::chat_prompt`: the real-text prompt helpers;
  - `proxima-tensor/specs/fsm-techniques/worked-examples.md`, section "row tolerance" (file created by FT0.14, section by FT0.20; absent on main, so in the executor's checkout a missing file or section means stop): the `RESULT row tolerance:` line, whose gemma4 E2B tolerance `5.841e-05` and the comparison `max_i |a_i - b_i| <= tau * max_i |b_i|` per row;
  - `proxima-model-interop/src/generate/decode.rs::generate_streaming_with_speculative_stats` (~line 2889): the generate entry that returns `SpeculativeDecodeStats` (FT7.6's `replay_rows`).
- change:
  1. `generate/prompt_cache_chunk_shift_real_model_tests.rs` (the test module that already owns the row readers): add a private `fn row_within_tolerance(replayed: &[f32], dense: &[f32], tolerance: f32) -> bool` (`largest gap <= tolerance * largest dense magnitude`) and the two tests below. The technique is one test-local decision function, `fn every_sixteen_rows_replay_eight(committed_len: usize, last_replay_end: usize) -> Option<usize> { (committed_len.saturating_sub(last_replay_end) >= 16).then_some(8) }`, composed through the hook as `decode.enter = &[EnterRule::Replay { max_rows: 8, decide: every_sixteen_rows_replay_eight }]`, with `speculative: SpeculativeConfig::none()`, greedy, standard prompt cache so the finished request stores its entry; that is the whole of it (about 15 lines of setup), and nothing is added to the library: the decision lives in this test module.
- test: two tests.
  - `a_periodic_replay_rewrites_rows_to_match_a_dense_prefill_on_real_gemma4_e2b` (`#[ignore = ..]` as its neighbours): generate 40 tokens from a chat prompt over the opening of the corpus document with the configuration above, through `generate_streaming_with_speculative_stats`; assert `stats.replay_rows == 16` and take `start = prompt_ids.len() + 24` (the second pass: with the decision above the passes fire at committed lengths `prompt_ids.len() + 16` and `prompt_ids.len() + 32`, so the second re-encodes rows `prompt_ids.len() + 24..prompt_ids.len() + 32`); read the stored entry's rows `start..start + 8` for every attention layer that is not a ring (`cache.ring_geometry().is_none()`) with `rows_of`; build the dense reference with `prefill_state(model, &ids[..start + 8])` (prompt ids followed by the generated ids); for each such layer and each of the 8 rows, `row_within_tolerance` holds for `k_even`, `k_odd` and `v` with tolerance `5.841e-05`; assert the number of layers compared is greater than 0 and print each layer's largest gap. If the entry the request stores does not hold those rows, stop and report.
  - `row_tolerance_rejects_a_perturbed_row` (not ignored, no model; the control): a row `dense = [0.5, -1.25, 3.0, 0.0]` and `replayed = dense` passes at `5.841e-05`; `replayed` with its first element shifted by `2.0 * 5.841e-05 * 3.0` fails the same comparison; the all-zero row against a nonzero perturbation fails.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_7_8 cargo nextest run -p proxima-model-interop --features std,metal -j 1 --run-ignored all -E 'test(/a_periodic_replay_rewrites_rows_to_match_a_dense_prefill_on_real_gemma4_e2b|row_tolerance_rejects_a_perturbed_row/)'`
- expect: `2 passed` (the model test and the control; the per-layer gaps print under `--no-capture`)
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: `proxima-model-interop/src/generate/prompt_cache_chunk_shift_real_model_tests.rs`
- commit: `test(interop): check replayed rows against a dense prefill`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: edit `src/` outside the test module; widen the tolerance when a row misses it (report the measured gap: the tolerance is derived, not tuned); add another checkpoint (gemma4 E2B only; no qwen of any kind).
- gpu: one run (`-j 1`, one model at a time), waiting for a quiet box.

## size-rule exception

FT7.3 touches four source files (`serving.rs`, `serving_settings.rs`, `prompt_cache_key.rs`, `resident_plans.rs`), one over the limit of three, and is listed as an admitted exception in CARDS.md for the reason already admitted for FT2.1 and FT2.2: a new `ServingConfig` field must reach the settings literal and both exhaustive destructures in the commit that adds it. A cut to three files was tried and is not green: the commit that adds `ServingConfig.decode` is `E0063` in `as_serving_config`, `CacheKey::of` and `PlanIdentity::of` until each names the field, and a commit that names the field before it exists does not compile; hosting the list inside an existing section would only move the fan-out to that section's literals. If CARDS.md does not list FT7.3 when the executor starts, it stops and reports; it does not split the card or edit CARDS.md.

## spec drift

1. The earlier cut fixed the enter order (rectify first, then drafters) and named the rule `Rectify`. The recut keeps the order as the order of the configured list, and the rules are generic: `DraftNonempty` (today's behaviour, the default list) and `Replay { max_rows, decide }`, whose decision is supplied by the caller. The periodic cadence and the row count appear only in the decision functions of the proof tests (FT7.6's real-model test and FT7.8), never in the library; an earlier draft shipped `Periodic { every_rows, replay_rows }` as a variant and a cadence check in `replay_range`, which is the technique built in, and was cut.
2. The earlier cut counted steps (`step % every == 0`). A call served from the queue of already-accepted draft tokens is not an entry point, so a step count can skip a pass; the recut counts committed rows since the last pass (`last_replay_end`), which catches up on the next evaluated call.
3. The earlier cut added `overwrite_rows` and a `RewindIntoSealed` check of its own. The recut commits by rewinding with the kv rewind that exists and appending the re-encoded rows; the sealed-row refusal is `try_truncate`'s (slice 4), plus a load-time refusal (FT7.7) so the failure is not discovered at the step where the first pass runs.
4. The pass is not a state of the serving state machine in the live loop (the machine is not what drives the loop). FT7.2 gives the machine its replay transition for the conformance suite; the live loop runs the same shape as a pre-pass inside the step.
5. A function cannot be named in TOML or an environment variable, so a `Replay` rule is composed in Rust and its serde variant is skipped: configuration loads only `DraftNonempty`, and the settings tests assert that `kind = "replay"` is refused. The decision is a function pointer; a derived `PartialEq` would trip the function-pointer comparison lint, so `EnterRule` compares by hand with `fn_addr_eq`. `max_rows` is data beside the function so the kv slack (FT7.5) and the sealed-rows refusal (FT7.7) can read a bound without calling it.
6. The dense reference in the proof is a dense prefill by this crate, a consistency check between two paths of one implementation, not an oracle. Token parity with the incumbent stays with the existing default-speculation tests and the vendored llama ids; no new oracle is queried.
7. The cache key sees the enter rules only through the ring slack rows (FT7.5). A replay rewrites recent rows with values that a dense prefill reproduces within the derived tolerance; if the proof shows a gap above the tolerance, the rules would have to join the key, and that is a re-cut, not a quiet edit.
8. Anchors: `ServingState` is `proxima-model-interop/src/serving_fsm.rs` at main 4b4be6cf and `#![allow(dead_code)]` there (line 38); the cards in slice 7 run after slice 1 has moved it to `proxima-core/src/serving_state.rs`.
9. Cards in other slice files cite the earlier shape and are re-cut by their own writers, not here: `tasks-recut/15-niah-read-arms.md` (the `REPLAY_ENTER` constant and the `EnterRule::{DraftNonempty, Periodic { every_rows, replay_rows }}` description) must define its periodic decision as a function of its own and use `Replay { max_rows, decide }`; `tasks-recut/17-sansio-conformance.md` (its `replay_range` worked example and tests written over `Periodic { .. }` literals) must write its periodic decisions as test-local functions; `tasks-recut/11-readouts.md` cites the test `a_periodic_rule_is_refused_until_the_loop_runs_it` as a `DecodeConfig` literal site (it is now `a_replay_rule_is_refused_until_the_loop_runs_it`, deleted by FT7.6).

## slice exit

- Commands, in order, each at the stated count:
  - `cargo nextest run -p proxima-core --no-default-features --features alloc,serde -E 'test(/enter_replay_range_runs_the_supplied_decision|enter_replay_range_bounds_and_rule_order|enter_rule_names_round_trip_through_json|enter_rule_equality_compares_the_decision_and_its_bound|accept_replay_resumes_from_the_waiting_entry/)'` prints `5 passed` (FT7.0's 2 + FT7.1's 2 + FT7.2's 1);
  - `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/decode_enter_defaults_to_drafting_only|step_row_limit_is_the_wider_of_draft_and_replay_rows|ring_slack_covers_the_replay_rows|replay_without_a_verify_program_is_refused|serving_settings_decode_enter_variants|serving_settings_refuses_replay_reaching_sealed_rows/)'` prints `6 passed` (derived by counting the names, not run; `row_tolerance_rejects_a_perturbed_row` is not in this run: its module is `#[cfg(all(test, feature = "metal", target_os = "macos"))]`, so without `metal` it is absent);
  - `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/row_tolerance_rejects_a_perturbed_row/)'` prints `1 passed` (loads no model);
  - `cargo nextest run -p proxima-model-interop --features std,metal -j 1 --run-ignored only -E 'test(/an_enter_list_without_drafting_never_verifies_on_real_gemma4_e2b|a_periodic_replay_keeps_greedy_ids_and_counts_its_rows_on_real_gemma4_e2b|a_periodic_replay_rewrites_rows_to_match_a_dense_prefill_on_real_gemma4_e2b|default_speculation_matches_off_/)'` prints `5 passed` (3 + the 2 default-speculation tests): five model loads, one per test, run one at a time under `-j 1`; this is the slice exit, not a card.
- Feature-off builds: `cargo check -p proxima-core --no-default-features --features alloc` and `cargo check -p proxima-model-interop --features std` are clean.
- Not cuttable (and why): FT7.6 holds the first production caller of `replay_range`, the `replay_rows` counter and `replay_refusal`, and the loop edit that makes them run. `generate` is a private module, so a test alone does not use a private item; splitting the card would leave an unused function in the first half. It touches three source files plus the test file, inside the file-count rule, and its validation loads one model (one real-model test, two decodes on that load), inside the one-model-run rule.
- Decide-later items with owners: a verify program for models other than gemma4 (architecture-as-data); until it lands a `Replay` rule on a model without a verify program is refused by FT7.6, never silently off.
