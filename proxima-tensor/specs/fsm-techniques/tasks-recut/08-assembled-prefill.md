# slice 8: Prefill from an assembled cache (re-cut; cards FT8.1 - FT8.17)

anchors read at main 4b4be6cf (full sha 4b4be6cf78bcd536fecdd4e4620a9769b9b7364c) with `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`. Every `path::symbol (~line N)` below is the line at that sha; the executor re-locates by symbol.

Governing direction (owner, 2026-10-04): the hooks are built so techniques can be vetted later; a technique is not built here. This slice builds the assemble hook (the request-to-starting-cache stage): its stage list is configuration, its decision is a pure function in proxima-core, its default list reproduces today's lookup byte for byte, and one test proves a stage the library does not define composes through it. The positions hook (ids to positions) is the second hook here. Governing spec: `proxima-windows/proxima-tensor/specs/pipeline-as-data/SPEC.md` (assemble and position rows), `sketches/11-cartridges-load.md` gap G1, `sketches/09-cacheblend.md` section 1.

## old to new id map

| old card | new card | verdict |
|---|---|---|
| FT8.1 | FT8.1 | keep |
| FT8.2 | FT8.2 | keep, trimmed to the stages bound in this slice; recut for the size rule: its three types land first in FT8.12, FT8.13 and FT8.14, and FT8.2 adds the function |
| FT8.3 | FT8.3 | keep, now also asserts non-contiguous positions |
| FT8.4 | FT8.4 | keep, maps to the payload-free kind list instead of a step list; `kind_of` is an exhaustive match of the two variants the grammar declares |
| FT8.6 | FT8.15, FT8.5 | recut for the size rule: FT8.15 takes the entry out before its rewind, FT8.5 marks the taken entry with its stamp and length |
| FT8.7 | FT8.6 | keep, shift stage is a no-op without a taken prefix |
| FT8.9 | FT8.16, FT8.17, FT8.7, FT8.8 | recut for the size rule: FT8.16 the ordered assemble over the cache with `take_best_shifting` delegating to it, FT8.17 the runner, FT8.7 the live caller that replaces the old lookup, FT8.8 the proof test |
| FT8.10 | FT8.9, FT8.10, FT8.11 | recut: one model-loading run per card (gemma4 e2b, gemma4 26b, granite) |
| FT8.5, FT8.8 (old) | none | ids were never used |

## dropped (content removed from old cards; no whole card is dropped)

- Old FT8.9 load arm, `assemble_load`, `assemble_load_missing_source`: a refusing stub with no production source; its first caller is the card that binds a production load source (cartridge or tier blocks), which also adds the load kind to the decision.
- Old FT8.9 blend arm and old FT8.2 `Blend` kind with `BlendNotLast` and `BlendWithoutRows`: they name a technique and no pipe here can run it.
- Old FT8.9 `CacheEntry::unmatched_rows` and `request_tail_start`: the per-entry assembled-prefix field is the design sketch 11 abandoned (placeholder ids keep the trie, lcp and rewind on ids). A load card that needs rows without token ids uses placeholder ids.
- Old FT8.10 openchat, qwen2 and qwen3 arms: qwen is banned; openchat is not a model the owner named for this work. The dense arm is gemma4 e2b, the MoE arms are gemma4 26b and granite.

## carried to a later card (not a card here)

- Fresh-entry layer state for a load stage (sketch 11 gap G2: `CacheEntry::empty` has `layer_caches: Vec::new()`, so `PrefixState::append_moved` refuses the first run on a fresh entry; the model-side builder is `LoadedModel::fresh_layer_caches`, `generate/decode.rs` ~line 1549). It is not a card here because its only caller is a production load source, which this slice does not bind; building it here would be dead code under the CARDS coherence rule, and calling it on the miss path would allocate every sliding-window ring on every miss (`attention_layer_cache` allocates the ring) with no gain. The card that binds the first production load source owns it. FT8.8 proves the hook with a hand-built empty layer inside the test.

## prerequisites outside this slice (each card names the one it needs)

- FT1 (all cards): `ServingState` lives in `proxima-core/src/serving_state.rs` (at main it is `proxima-model-interop/src/serving_fsm.rs::ServingState`, enum at ~line 48) and `proxima-core` is a non-optional dependency of `proxima-model-interop` with feature `alloc` (at main `proxima-model-interop/Cargo.toml` ~line 314 declares it `optional = true`). The executor checks `proxima_core::serving_state` resolves from `proxima-model-interop` before FT8.4 and stops if it does not.
- FT2 (all cards): `ServingConfig.prefill.assemble`, an ordered list of `AssembleStep` read with `.iter()`, empty when unset. FT2.2 declares exactly two variants, `Prefix` and `Shift`. FT8.4 maps both by name in an exhaustive `match` (no wildcard, no refusal arm), so a variant the grammar gains without a pipe stops compiling there. Premise check, run before FT8.4: `git grep -n -A4 "pub enum AssembleStep" -- proxima-model-interop/src/serving_grammar.rs` shows `Prefix` and `Shift` and no other variant; if it shows another, stop and report. In this slice the list does not enter the prompt-cache key: with Prefix and Shift only, the rows an entry holds are the same whatever the order. A stage that changes what a row means must bind its identity into the key in its own card.
- DAD C7 (decode-as-data): `build_position_inputs_at(new_ids: &[u32], positions: &[usize], head_dim, rope_freq_base, rms_epsilon, rope_freqs, scaling)` in `proxima-model-interop/src/generate/residency_caches.rs`, with `build_position_inputs` kept as a wrapper (absent at main: `git grep -n build_position_inputs_at main` finds nothing).
- FT0 (all cards): vendors `tests/fixtures/llama-parity/<name>/followup_ids.json` for gemma4_e2b, gemma4_26b and the granite checkpoint (one-element JSON array; the element holds `turn1.prompt_ids`, `turn2.prompt_ids`, `turn2.generated_ids`, arrays of unsigned integers; recorded once from the incumbent, never re-queried), and makes granite3.1-moe load by family profile plus descriptor with a `Checkpoint` static in `tests/arch_data_baseline.rs` whose `architecture` is `"granitemoe"`.

Test-name rule for this slice: no new test in `proxima-core/src/serving_state.rs` or `proxima-core/src/serving_state/` may contain `serving_state_` or `action_speculation_` or start with `sansio_` (other slices count those prefixes). The proxima-core tests of this slice start with `assemble_kind_`, `assemble_refusal_`, `assemble_decision_` or `assemble_plan_`. The proxima-model-interop tests of this slice in `prompt_cache.rs` start with `prefix_take_`, `assemble_shift_`, `assemble_order_`, `assemble_stops_`, `assemble_empty_`, `assemble_default_`, `assemble_without_`, `assemble_miss_`, `assemble_refused_`, `assemble_config_` or `assemble_kind`, or are named `a_stage_the_library_does_not_define_composes_by_list_order`. Each card's validate filter names only its own prefixes, and no other card of the plan may start a test name with `prefix_take_`, `assemble_shift_` or `assemble_config_` (the cache-reuse oracle tests of the later cache-reuse slice start with `assemble_prefix_`, which no filter here matches).

Size rule for this slice: every card adds at most one item (type, function, method or field) that another card consumes. The assemble module in proxima-core is therefore four cards (kind FT8.12, error FT8.13, decision FT8.14, function FT8.2), and the prompt-cache work is six (take FT8.15, mark FT8.5, shift FT8.6, ordered assemble FT8.16, runner FT8.17, live caller FT8.7). The card order in this file is the dependency order, not the numeric order.

Test models (owner, 2026-10-04): gemma4 for dense or MoE, granite for MoE. No qwen of any kind appears in any card of this file.

---

### 8.1 Add `start_position` to the `Prefill` variant

- id: FT8.1
- needs: FT1 (all cards)
- budget: 20 min
- crate(s): proxima-core (features: alloc), proxima-model-interop (features: std)
- read first:
  - `proxima-core/src/serving_state.rs::ServingState` (moved from `proxima-model-interop/src/serving_fsm.rs::ServingState`, enum ~line 48 and `Prefill { positions: Vec<u32>, cache }` ~line 52 at 4b4be6cf): the `Prefill` variant and its doc;
  - `ServingState::start` (serving_fsm.rs ~line 116) and `ServingState::advance_prefill` (~line 123);
  - `proxima-model-interop/src/generate/serving_backend.rs`, the test `serving_backend_drives_serving_state_through_prefill_and_decode` (~line 182), whose pattern `ServingState::Prefill { positions, cache }` is at ~line 193;
  - the test that builds `ServingState::Prefill { positions: vec![1, 2, 3], cache: FakeCache::empty() }` (at main `walkthrough_drives_every_legal_transition`, serving_fsm.rs ~line 318; FT1 may have renamed it) and the test module's `FakeCache` (~line 292, fields `conv_history_len`, `kv_len`).
- change:
  1. `proxima-core/src/serving_state.rs`: the variant becomes `Prefill { positions: Vec<Entry>, start_position: usize, cache: Cache }` (`Entry` is whatever element type FT1 gave `positions`). Rewrite its doc to: "`positions`: the entries to place in one program evaluation. `start_position`: the absolute position of `positions[0]`. `cache` may already hold the rows of every position before `start_position`: an assembled prefix, loaded blocks, or a cartridge."
  2. Same file: `start(positions: Vec<Entry>, cache: Cache) -> Self` keeps its signature and sets `start_position: 0`. Add `pub fn start_at(positions: Vec<Entry>, start_position: usize, cache: Cache) -> Self`. `advance_prefill` keeps matching `Self::Prefill { .. }`.
  3. Same file, test module: in the walkthrough test, the expected value becomes `ServingState::Prefill { positions: vec![1, 2, 3], start_position: 0, cache: FakeCache::empty() }`.
  4. `proxima-model-interop/src/generate/serving_backend.rs`: the pattern becomes `ServingState::Prefill { positions, cache, .. }`.
- test: add `fsm_prefill_start_at_keeps_start_position` in `proxima-core/src/serving_state.rs` (test module, using its existing `FakeCache`). It asserts:
  - `ServingState::start_at(vec![7_u32, 8, 9], 5, FakeCache { conv_history_len: 5, kv_len: 5 })` equals `ServingState::Prefill { positions: vec![7, 8, 9], start_position: 5, cache: FakeCache { conv_history_len: 5, kv_len: 5 } }`;
  - `.advance_prefill(10, FakeCache { conv_history_len: 8, kv_len: 8 })` gives `Ok(ServingState::Decode { last: 10, cache: FakeCache { conv_history_len: 8, kv_len: 8 } })`;
  - `ServingState::start(vec![1_u32], FakeCache::empty())` has `start_position: 0`;
  - `ServingState::<u32, FakeCache>::start_at(vec![1], 0, FakeCache::empty()).advance_decode(2, FakeCache::empty())` is `Err(ServingFsmError::IllegalTransition { attempted: "advance_decode" })` (sad path).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_1 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/fsm_prefill_/)'`, then the same command with `-E 'test(/walkthrough_drives_every_legal_transition/)'`, then `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_1 cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_backend_drives_serving_state_through_prefill_and_decode/)'`
- expect: `1 passed`, then `1 passed`, then `1 passed`
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`, `cargo clippy -p proxima-model-interop --features std --all-targets`, `cargo check -p proxima-core --no-default-features --features alloc`
- stage: proxima-core/src/serving_state.rs, proxima-model-interop/src/generate/serving_backend.rs
- commit: `feat(core): start prefill from an existing cache position`
- done when: the three expect lines printed, clippy clean, and `git diff --cached --stat` touches only `proxima-core/src/serving_state.rs` and `proxima-model-interop/src/generate/serving_backend.rs`
- do not: change `advance_prefill`, `enter_verify`, `accept` or any other transition; touch `proxima-core/src/lib.rs`
- gpu: none

### 8.12 Name the stages the prompt cache runs to build a starting cache

- id: FT8.12
- needs: FT1 (all cards)
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-core/src/serving_state.rs` (after FT1): where its child modules are declared, and the in-file test layout to copy;
  - `proxima-model-interop/src/generate/prompt_cache.rs::LoadedModel::run_decode_loop_through_cache` (~line 1202 at 4b4be6cf): the `let cacheable = config.is_enabled() && seed.is_none() && ..` expression (~line 1220), whose "run the cache" outcome the stages below name;
  - `prompt_cache.rs::LoadedModel::prompt_cache_lookup` (~line 1047): `let shifting = config.cache_reuse_min > 0 && config.ring_rewind_slack > 0;` (~line 1059), the other half of what runs today;
  - SPEC (pipeline-as-data) structural finding: "decision in proxima-core" has no precedent; `git ls-tree main proxima-core/src` lists no serving-decision module at main, so this slice's first three cards create the first one.
- change:
  1. File `proxima-core/src/serving_state/assemble.rs` (new; uses `core` only, no allocation): `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum AssembleKind { Prefix, Shift }`. Doc: "a stage that builds the starting cache of a request. `Prefix` starts from the stored entry that shares the longest prefix with the prompt. `Shift` also reuses stored chunks that moved position." Why payload-free: the stage list's payloads stay in proxima-model-interop, so core needs no config type from interop and no allocation.
  2. `proxima-core/src/serving_state.rs`: declare `pub mod assemble;` (no cfg gate beyond the file's own). Interop reaches the item as `proxima_core::serving_state::assemble::AssembleKind`.
- test: add in `proxima-core/src/serving_state/assemble.rs` (`#[cfg(test)] mod tests`, with `use alloc::format;` when the crate is `no_std` there) `assemble_kind_debug_names_are_the_stage_names`, asserting `format!("{:?}", AssembleKind::Prefix) == "Prefix"`, `format!("{:?}", AssembleKind::Shift) == "Shift"`, `AssembleKind::Prefix != AssembleKind::Shift`, and a copy equals its source (`let kind = AssembleKind::Shift; let copy = kind; assert_eq!(kind, copy)`). The next card's error message prints a kind through `{kind:?}`, which is why the debug text is asserted.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_12 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/assemble_kind_/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`, `cargo check -p proxima-core --no-default-features --features alloc`, and `git grep -nE 'std::|println!|tokio' -- proxima-core/src/serving_state/assemble.rs` prints nothing
- stage: proxima-core/src/serving_state.rs, proxima-core/src/serving_state/assemble.rs
- commit: `feat(core): name the stages that build a starting cache`
- done when: expect printed, clippy and the alloc check clean, the grep empty, and `git diff --cached --stat` touches only `proxima-core/src/serving_state.rs` and `proxima-core/src/serving_state/assemble.rs`
- do not: add a config type, a trait or a `Box`; depend on `proxima-model-interop`; add a stage that no production code runs
- gpu: none

### 8.13 Refuse a malformed assemble order with a named error

- id: FT8.13
- needs: FT8.12
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-core/src/serving_state/assemble.rs::AssembleKind` (FT8.12) and its test module;
  - `proxima-core/src/ring/mod.rs` (~line 12 at 4b4be6cf): the `thiserror::Error` derive style `proxima-core` already uses on a `no_std` module (`thiserror` is an unconditional dependency of the crate: `proxima-core/Cargo.toml` ~line 77).
- change:
  1. `assemble.rs`: add `#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)] pub enum AssembleOrderError { #[error("assemble: shift needs an earlier prefix")] ShiftWithoutPrefix, #[error("assemble: {kind:?} appears more than once")] Repeated { kind: AssembleKind }, #[error("assemble: shift needs cache_reuse_min and ring_rewind_slack above zero")] ShiftNeedsReuse }`. Doc on the enum: "why a configured stage list was refused; order is never rewritten, the configured order is the rule".
- test: add `assemble_refusal_messages_name_the_rule` in `assemble.rs` tests (`use alloc::string::ToString;` when the crate is `no_std` there), asserting `AssembleOrderError::ShiftWithoutPrefix.to_string() == "assemble: shift needs an earlier prefix"`, `AssembleOrderError::Repeated { kind: AssembleKind::Prefix }.to_string() == "assemble: Prefix appears more than once"`, `AssembleOrderError::ShiftNeedsReuse.to_string() == "assemble: shift needs cache_reuse_min and ring_rewind_slack above zero"`, and `AssembleOrderError::Repeated { kind: AssembleKind::Prefix } != AssembleOrderError::Repeated { kind: AssembleKind::Shift }`. Interop puts this text into its serving-config error, and a later test there matches on `shift needs an earlier prefix`, so the wording is behaviour.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_13 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/assemble_refusal_/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`, `cargo check -p proxima-core --no-default-features --features alloc`, and `git grep -nE 'std::|println!|tokio' -- proxima-core/src/serving_state/assemble.rs` prints nothing
- stage: proxima-core/src/serving_state/assemble.rs
- commit: `feat(core): name why an assemble order is refused`
- done when: expect printed, clippy and the alloc check clean, the grep empty, and `git diff --cached --stat` touches only `proxima-core/src/serving_state/assemble.rs`
- do not: add a variant that no check of this slice returns; add a `Box`, a `String` field or any message built at runtime
- gpu: none

### 8.14 Name what the assemble decision can be

- id: FT8.14
- needs: FT8.12
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-core/src/serving_state/assemble.rs::AssembleKind` (FT8.12) and its test module;
  - `prompt_cache.rs::LoadedModel::run_decode_loop_through_cache` (~line 1202 at 4b4be6cf), the `cacheable` expression and the `if !cacheable { return self.run_decode_loop_from_ids(..) }` branch (~line 1226): the "bypass" outcome;
  - `prompt_cache.rs::LoadedModel::prompt_cache_lookup` (~line 1047), the `shifting` expression: the two stages that run when no list is configured.
- change:
  1. `assemble.rs`: add `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum AssembleDecision { Bypass, Legacy { shift: bool }, Configured }`. Doc: `Bypass` "run no stage: the request prefills from an empty cache"; `Legacy` "no list is configured: run the prefix stage, and the shift stage when `shift`"; `Configured` "run the configured list, in its own order". Why a decision enum and not the stage list: payloads stay in interop.
- test: add `assemble_decision_variants_differ_by_what_they_run` in `assemble.rs` tests, asserting `AssembleDecision::Legacy { shift: true } != AssembleDecision::Legacy { shift: false }`, `AssembleDecision::Bypass != AssembleDecision::Configured`, and a copy of `AssembleDecision::Legacy { shift: true }` equals its source. The next card returns these values from a pure function.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_14 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/assemble_decision_/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`, `cargo check -p proxima-core --no-default-features --features alloc`, and `git grep -nE 'std::|println!|tokio' -- proxima-core/src/serving_state/assemble.rs` prints nothing
- stage: proxima-core/src/serving_state/assemble.rs
- commit: `feat(core): name the outcomes of the assemble decision`
- done when: expect printed, clippy and the alloc check clean, the grep empty, and `git diff --cached --stat` touches only `proxima-core/src/serving_state/assemble.rs`
- do not: add a variant that carries a stage list or a `Vec`; add a variant that no later card in this slice returns
- gpu: none

### 8.2 Decide the assemble stage list as a pure function in proxima-core

- id: FT8.2
- needs: FT1 (all cards), FT8.12, FT8.13, FT8.14
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-core/src/serving_state/assemble.rs`: `AssembleKind` (FT8.12), `AssembleOrderError` (FT8.13) and `AssembleDecision` (FT8.14), and the in-file test layout to copy;
  - `proxima-model-interop/src/generate/prompt_cache.rs::LoadedModel::run_decode_loop_through_cache` (~line 1202 at 4b4be6cf): the `let cacheable = config.is_enabled() && seed.is_none() && token_override.is_none() && !ids.is_empty() && Discard sinks` expression (~line 1220) this card turns into a pure function;
  - `prompt_cache.rs::LoadedModel::prompt_cache_lookup` (~line 1047): `let shifting = config.cache_reuse_min > 0 && config.ring_rewind_slack > 0;` (~line 1059).
- change:
  1. `assemble.rs`: add `pub fn assemble_decision(configured: &[AssembleKind], cache_enabled: bool, shift_enabled: bool, plain_request: bool) -> Result<AssembleDecision, AssembleOrderError>`. Body, in this order:
     1. `!plain_request` gives `Ok(Bypass)` (a caller-owned seed, a forced token stream or an observing sink needs the full prefill);
     2. `configured` empty: `cache_enabled` gives `Ok(Legacy { shift: shift_enabled })`, else `Ok(Bypass)`;
     3. else validate `configured` in list order, per element first `Repeated` (the kind was already seen), then `ShiftWithoutPrefix` (a `Shift` with no `Prefix` before it); after the walk, a list holding `Shift` while `!shift_enabled` is `Err(ShiftNeedsReuse)`; otherwise `Ok(Configured)`. Order is never rewritten: config order is the rule.
     The function uses `core` only and allocates nothing.
- test: add in `assemble.rs` (`#[cfg(test)] mod tests`), names prefixed `assemble_plan_`:
  - `assemble_plan_matches_the_old_cacheable_expression`: all 2^5 combinations of `(cache_enabled, seed_none, override_none, ids_nonempty, sinks_discard)` with `configured = []`, `shift_enabled = false`, `plain_request = seed_none && override_none && ids_nonempty && sinks_discard`: the result is `Ok(Legacy { shift: false })` exactly when `cache_enabled && plain_request`, else `Ok(Bypass)`. Count asserted: `assert_eq!(checked, 32)`;
  - `assemble_plan_legacy_shift_follows_the_reuse_flag`: `([], true, true, true)` is `Ok(Legacy { shift: true })`; `([], true, false, true)` is `Ok(Legacy { shift: false })`;
  - `assemble_plan_configured_list_wins_over_the_cache_flag`: `([Prefix, Shift], false, true, true)` is `Ok(Configured)`;
  - `assemble_plan_refuses_shift_without_prefix`: `([Shift], true, true, true)` and `([Shift, Prefix], true, true, true)` are `Err(ShiftWithoutPrefix)`;
  - `assemble_plan_refuses_repeats`: `([Prefix, Prefix], true, true, true)` is `Err(Repeated { kind: AssembleKind::Prefix })`; `([Prefix, Shift, Shift], true, true, true)` is `Err(Repeated { kind: AssembleKind::Shift })`;
  - `assemble_plan_refuses_shift_when_reuse_is_off`: `([Prefix, Shift], true, false, true)` is `Err(ShiftNeedsReuse)`; `([Prefix], true, false, true)` is `Ok(Configured)`;
  - `assemble_plan_bypass_ignores_the_list`: `([Prefix, Shift], true, true, false)` is `Ok(Bypass)`.
  That is 7 tests; the expect line counts them.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_2 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/assemble_plan_/)'`, then the same command with `-E 'test(/assemble_kind_|assemble_refusal_|assemble_decision_/)'`
- expect: `7 passed`, then `3 passed`
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`, `cargo check -p proxima-core --no-default-features --features alloc`, and `git grep -nE 'std::|println!|tokio' -- proxima-core/src/serving_state/assemble.rs` prints nothing
- stage: proxima-core/src/serving_state/assemble.rs
- commit: `feat(core): decide the assemble pipe order as a pure function`
- done when: both expect lines printed, clippy and the alloc check clean, the grep empty, and `git diff --cached --stat` touches only `proxima-core/src/serving_state/assemble.rs`
- do not: put `#[must_use]` on it (the `Result` return is already must_use and `clippy::double_must_use` is denied); add a config type, a trait or a `Box`; depend on `proxima-model-interop`; reorder `configured`; add a kind that no production code runs
- gpu: none

### 8.3 Route the live prefill positions through `build_position_inputs_at`

- id: FT8.3
- needs: DAD C7
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/decode.rs::LoadedModel::run_decode_loop_from_ids` (~line 2988 at 4b4be6cf): the local `seed_cached_len` (~line 3011) and the per-step call `build_position_inputs(ids_for_step, cached_len, ..)` (~line 3829);
  - `proxima-model-interop/src/generate/residency_caches.rs::build_position_inputs` (~line 1206) and `build_position_inputs_at` (added beside it by DAD C7, absent at main);
  - `residency_caches.rs::rope_freqs_tests` (~line 1258): the gemma4 head_dim 512 fixture (`rope_freqs` = 64 values `1.0` then 192 values `1.0e30`, `rope_freq_base` `1.0e6`).
- change:
  1. `decode.rs`: before the step loop in `run_decode_loop_from_ids`, add `let mut position_scratch: Vec<usize> = Vec::new();`. At the call site (~line 3829), replace `build_position_inputs(ids_for_step, cached_len, ..)` by `position_scratch.clear(); position_scratch.extend(cached_len..cached_len + ids_for_step.len());` then `build_position_inputs_at(ids_for_step, &position_scratch, ..)` with the identical remaining arguments. The scratch `Vec` is reused every step, so there is no per-step allocation after the first. DAD C7 leaves this site on the wrapper; this card is the one that moves it.
  2. One lowercase why-comment above the `position_scratch` declaration: `// a positions slice, not a start offset: any prefill start or position scheme is one slice`.
- test: add `prefill_positions_at_match_start_offset_table` in the `rope_freqs_tests` module of `residency_caches.rs`. Fixture: head_dim 512, `rope_freq_base` `1.0e6_f32`, `rms_epsilon` `1.0e-5_f32`, `rope_freqs` as above passed as `Some(..)`, `RopeScaling::None`. For each `start in [0_usize, 5, 512]` with ids `[11_u32, 12, 13]`: `build_position_inputs_at(&ids, &[start, start + 1, start + 2], ..)` has `cos` and `sin` equal (`==` on `Vec<f32>`) to `build_position_inputs(&ids, start, ..)`, and `ids_i32` is `[11, 12, 13]`. A further assertion for a non-contiguous slice: `build_position_inputs_at(&[11, 12], &[3, 700], ..)` has the first row (`cos[..256]`) equal to `build_position_inputs(&[11], 3, ..).cos` and the second row (`cos[256..]`) equal to `build_position_inputs(&[12], 700, ..).cos`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_3 cargo nextest run -p proxima-model-interop --features std -E 'test(/prefill_positions_at_match_start_offset_table/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: proxima-model-interop/src/generate/decode.rs, proxima-model-interop/src/generate/residency_caches.rs
- commit: `refactor(interop): build prefill positions from a slice`
- done when: expect printed, clippy clean, `git diff --cached --stat` touches only `generate/decode.rs` and `generate/residency_caches.rs`
- do not: change the verify call site (`decode.rs` ~line 6447, owned by DAD C7), `chunk_shift.rs` ~line 397 or `lfm2.rs` ~line 586; run any model
- gpu: none

### 8.4 Derive the cache bypass from the assemble decision

- id: FT8.4
- needs: FT8.2, FT2 (all cards)
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-core/src/serving_state/assemble.rs::assemble_decision` (FT8.2): the decision this card applies;
  - `proxima-model-interop/src/generate/prompt_cache.rs::LoadedModel::run_decode_loop_through_cache` (~line 1202 at 4b4be6cf), the `let cacheable = ..` expression (~line 1220);
  - `prompt_cache.rs::LoadedModel::prompt_cache_lookup` (~line 1047), the `shifting` expression (~line 1059);
  - FT2's `AssembleStep` (module `serving_grammar`, exactly `Prefix` and `Shift`) and `ServingConfig.prefill.assemble` (`PrefillConfig`); at main neither exists (`git ls-tree main proxima-model-interop/src` lists no `serving_grammar.rs`) and `ServingConfig` is `Copy` at `serving.rs` ~line 719, which is why FT2 owns the list's container type. Premise check: `git grep -n -A4 "pub enum AssembleStep" -- proxima-model-interop/src/serving_grammar.rs` shows no variant besides `Prefix` and `Shift`; if it shows another, stop and report.
- change:
  1. `prompt_cache.rs`: add private `fn kind_of(step: &AssembleStep) -> AssembleKind`: an exhaustive `match step` with the arms `AssembleStep::Prefix => AssembleKind::Prefix` and `AssembleStep::Shift => AssembleKind::Shift`. No wildcard arm and no refusal arm: a variant the grammar gains without a pipe behind it stops compiling here, which is the refusal. It is infallible (a `Result` that is always `Ok` would trip `clippy::unnecessary_wraps`).
  2. Same file: add `pub(super) fn assemble_kinds_for(serving_config: &ServingConfig, plain_request: bool) -> Result<Vec<AssembleKind>, InteropError>`. Body: `let cache = serving_config.prompt_cache;`; `let configured: Vec<AssembleKind> = serving_config.prefill.assemble.iter().map(kind_of).collect();`; `let shift_enabled = cache.cache_reuse_min > 0 && cache.ring_rewind_slack > 0;`; call `assemble_decision(&configured, cache.is_enabled(), shift_enabled, plain_request)` and map `AssembleOrderError` to `InteropError::UnsupportedServingConfig(error.to_string())`; then `Bypass` gives `Vec::new()`, `Legacy { shift }` gives `[Prefix]` plus `Shift` when `shift`, `Configured` gives `configured`. No decision is made here: only the mapping.
  3. `run_decode_loop_through_cache`: compute `let plain_request = seed.is_none() && token_override.is_none() && !ids.is_empty() && matches!(logits_sink, LogitsSink::Discard) && matches!(node_values_sink, NodeValuesSink::Discard);` (facts, not a decision), then `let kinds = assemble_kinds_for(serving_config, plain_request)?;` and `let cacheable = !kinds.is_empty();`. The `if !cacheable { return self.run_decode_loop_from_ids(..) }` branch is unchanged.
- test: add in the `tests` module of `prompt_cache.rs`, `assemble_kinds_for_maps_the_core_decision`, with `ServingConfig { prompt_cache: PromptCacheConfig { byte_budget: 1 << 20, cache_reuse_min: 4, ring_rewind_slack: 256, ..PromptCacheConfig::off() }, ..ServingConfig::default() }` (call it the reuse config) and `plain_request = true`:
  - the result is `[Prefix, Shift]`; with `cache_reuse_min: 0` it is `[Prefix]`; with `byte_budget: 0` it is `[]`; with the default config and `plain_request = false` it is `[]`;
  - the reuse config with `prefill: PrefillConfig { assemble: &[AssembleStep::Prefix] }` is `[Prefix]` (the configured list wins over the legacy shift), and with `assemble: &[AssembleStep::Prefix, AssembleStep::Shift]` it is `[Prefix, Shift]` (both variants map through `kind_of`, order kept);
  - the reuse config with `prefill: PrefillConfig { assemble: &[AssembleStep::Shift] }` is `Err(InteropError::UnsupportedServingConfig(_))` whose text contains `shift needs an earlier prefix`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_4 cargo nextest run -p proxima-model-interop --features std -E 'test(/assemble_kind/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; run `cargo nextest run -p proxima-model-interop --features std -E 'test(/prompt_cache::tests::/)'` before and after: the after count is the before count plus 1; `git grep -n "has no bound pipe" -- proxima-model-interop/src` prints nothing
- stage: proxima-model-interop/src/generate/prompt_cache.rs
- commit: `refactor(interop): derive the cache bypass from the assemble decision`
- done when: expect printed, the before and after counts differ by exactly 1, clippy clean, `git diff --cached --stat` touches only `generate/prompt_cache.rs`
- do not: re-implement any condition of `assemble_decision` in interop; remove `PromptCacheConfig::is_enabled`; add the list to the prompt-cache key; add a wildcard or refusal arm to `kind_of`
- gpu: none

### 8.15 Take the best prefix entry out of the cache before its rewind

- id: FT8.15
- needs: FT8.4 (no symbol of it is used; this card edits the same file, so it runs after FT8.4)
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `prompt_cache.rs::PromptCache::take_best_shifting` (~line 644 at 4b4be6cf): the `best` and `resume` computation and the `outcome` match; `PromptCache::resume_at` (~line 745): its first statement drops the entry from the cache, and on a refused rewind it calls `self.keep(stamp, entry)` with the stamp it was given;
  - `prompt_cache.rs::PromptCache::best_candidate` (~line 696), `miss_reason` (~line 724), `drop_entry` (~line 806) and `keep` (~line 775);
  - the tests `a_lifted_run_turns_the_request_into_a_shift_that_counts_the_moved_tokens_as_reused` (~line 1371, the 40-token example: stored ids `1..=40`, prompt of 30, shared prefix 10) and `a_refused_rewind_keeps_the_entry_cached` (~line 1913), with the helpers `state_with_ids` (~line 1319), `shared_widths` (~line 1330), `enabled_config`, `stored_conversation_and_squashed_prompt` and `base_key`.
- change:
  1. `prompt_cache.rs`, `impl PromptCache`: add private `fn assemble_prefix(&mut self, prompt_ids: &[u32], key: &CacheKey, min_similarity_milli: u32) -> Result<(CacheEntry, u64, usize), MissReason>`. Body: `let Some((stamp, lcp)) = self.best_candidate(prompt_ids, key, min_similarity_milli) else { return Err(self.miss_reason(prompt_ids, key)) };` then `let resume_len = lcp.min(prompt_ids.len().saturating_sub(1));`, then `if resume_len == 0 { return Err(MissReason::NoCommonPrefix) }`, then `let entry = self.drop_entry(stamp).ok_or(MissReason::UnrewindableLayer)?;` and `Ok((entry, stamp, resume_len))`. It takes the entry out of the cache and leaves it whole: the rows past the shared prefix are still there.
  2. Same file: `resume_at` changes signature to `fn resume_at(&mut self, entry: CacheEntry, stamp: u64, resume_len: usize, widths: &[LayerPadRowWidths], lift: Option<Lift<'_>>) -> Result<(CacheEntry, CachePath), MissReason>` (the entry is now an argument as `mut entry`). Its first statement, the `drop_entry` call, is deleted; every other line is unchanged.
  3. `take_best_shifting`: replace the `best`, `resume` and `outcome` lines by `let outcome = self.assemble_prefix(prompt_ids, key, min_similarity_milli).and_then(|(entry, stamp, resume_len)| self.resume_at(entry, stamp, resume_len, widths, lift).map(|(entry, path)| (entry, path, resume_len)));`. The report arm and everything after it are unchanged; the signature is unchanged.
- test: add in the `tests` module of `prompt_cache.rs`, on the 40/30 example (`state_with_ids(&(1..=40).collect::<Vec<u32>>())` stored with `enabled_config()`, prompt `stored_conversation_and_squashed_prompt().1`, `ANY_OVERLAP`):
  - `prefix_take_returns_the_entry_unrewound`: `cache.assemble_prefix(&prompt, &base_key(), ANY_OVERLAP)` is `Ok((entry, _, 10))` with `entry.state.cached_len == 40`, `entry.moved.is_empty()`, and `cache.entries.len() == 0` (the entry left the cache);
  - `prefix_take_miss_leaves_the_cache_intact` (sad path): the prompt `vec![900_u32; 30]` gives `Err(MissReason::NoCommonPrefix)` and `cache.entries.len() == 1`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_15 cargo nextest run -p proxima-model-interop --features std -E 'test(/prefix_take_/)'`
- expect: `2 passed`; then the `test(/prompt_cache::tests::/)` filter passes the baseline count (recorded before the edit) plus 2, 0 failed (the existing shift, rewind and refused-rewind tests stay green)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: proxima-model-interop/src/generate/prompt_cache.rs
- commit: `refactor(interop): take the prefix entry out before its rewind`
- done when: both expect lines printed, clippy clean, `git diff --cached --stat` touches only `generate/prompt_cache.rs`
- do not: change the signatures of `take_best` or `take_best_shifting`; change any existing test; touch `chunk_shift.rs`
- gpu: none

### 8.5 Prefix stage: mark the taken entry with its stamp and resume length

- id: FT8.5
- needs: FT8.15
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `prompt_cache.rs::PromptCache::assemble_prefix` and `PromptCache::resume_at` (FT8.15): the two functions whose signatures this card changes, and `take_best_shifting`'s `outcome` line;
  - `prompt_cache.rs::CacheEntry` (~line 358 at 4b4be6cf), `CacheEntry::new` and `CacheEntry::resume` (~line 487): the rewind;
  - the tests `prefix_take_returns_the_entry_unrewound` (FT8.15) and `a_refused_rewind_keeps_the_entry_cached` (~line 1913), with `gemma_like_state` (~line 1775) and `gemma_like_widths`;
  - `prompt_cache.rs::CacheReport::miss` (~line 195) and `MissReason::RingSlackExceeded`.
- change:
  1. `prompt_cache.rs`, `CacheEntry`: add the private field `resume_from: Option<(u64, usize)>`, doc: "set while the entry is taken out of the cache but not yet rewound: the stamp it was stored under, and the length it must be rewound to once the stages that need its whole rows have run; `None` otherwise". `CacheEntry::new` sets `resume_from: None`; any other `CacheEntry { .. }` literal the compiler lists gets the same line.
  2. `PromptCache::assemble_prefix` returns `Result<CacheEntry, MissReason>`: its last lines become `let mut entry = self.drop_entry(stamp).ok_or(MissReason::UnrewindableLayer)?;`, `entry.resume_from = Some((stamp, resume_len));`, `Ok(entry)`.
  3. `PromptCache::resume_at` becomes `fn resume_at(&mut self, mut entry: CacheEntry, widths: &[LayerPadRowWidths], lift: Option<Lift<'_>>) -> Result<(CacheEntry, CachePath), MissReason>`. Its first statement is `let Some((stamp, resume_len)) = entry.resume_from.take() else { return Err(MissReason::NoCommonPrefix) };` (the take clears the mark, so a rewound entry and a kept entry both carry `None`); every other line is unchanged.
  4. `take_best_shifting`: the `outcome` line becomes `let outcome = self.assemble_prefix(prompt_ids, key, min_similarity_milli).and_then(|entry| { let resume_len = entry.resume_from.map_or(0, |(_, len)| len); self.resume_at(entry, widths, lift).map(|(entry, path)| (entry, path, resume_len)) });`.
- test: in the `tests` module of `prompt_cache.rs`, on the 40/30 example of FT8.15:
  - change `prefix_take_returns_the_entry_unrewound` (added by FT8.15; this card owns the edit): `cache.assemble_prefix(..)` is now `Ok(entry)` with `entry.resume_from.map(|(_, len)| len) == Some(10)`, `entry.state.cached_len == 40`, `entry.moved.is_empty()` and `cache.entries.len() == 0`;
  - add `prefix_take_rewind_clears_the_mark`: `cache.take_best(&prompt, &base_key(), &shared_widths(), ANY_OVERLAP)` is `(Some(entry), report)` with `entry.resume_from == None`, `entry.state.cached_len == 10` and `report.path == CachePath::Rewind`;
  - add `prefix_take_refused_rewind_keeps_the_entry_unmarked` (sad path): store `CacheEntry::new(gemma_like_state(40), base_key())` the way `a_refused_rewind_keeps_the_entry_cached` stores its entry; for the prompt `(0..35).chain([900, 901])` `cache.take_best(.., &gemma_like_widths(), ANY_OVERLAP)` is `(None, report)` with `report.miss == Some(MissReason::RingSlackExceeded { rewind_rows: 5, slack_rows: SLACK })`, `cache.entries.len() == 1`, and every held entry has `resume_from == None` (`cache.entries.values().all(|held| held.resume_from.is_none())`).
  Total tests matching the filter after this card: the 2 of FT8.15 plus 2 new.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_5 cargo nextest run -p proxima-model-interop --features std -E 'test(/prefix_take_/)'`
- expect: `4 passed`; then the `test(/prompt_cache::tests::/)` filter passes the baseline count (recorded before the edit) plus 2, 0 failed
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: proxima-model-interop/src/generate/prompt_cache.rs
- commit: `refactor(interop): mark the taken prefix entry with its stamp`
- done when: both expect lines printed, clippy clean, `git diff --cached --stat` touches only `generate/prompt_cache.rs`
- do not: change the signatures of `take_best` or `take_best_shifting`; change any test other than `prefix_take_returns_the_entry_unrewound`; touch `chunk_shift.rs`
- gpu: none

### 8.6 Shift stage: lift chunks from the un-rewound entry

- id: FT8.6
- needs: FT8.5
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/chunk_shift.rs::LoadedModel::lift_chunks` (~line 459 at 4b4be6cf): the lift the shift stage calls; `MovedRun` (~line 67);
  - `prompt_cache.rs::Lift` (~line 531, `&mut dyn FnMut(&CacheEntry, usize) -> Vec<MovedRun>`), `PromptCache::resume_at` (FT8.5), whose `lift` argument this card removes, and `take_best_shifting`'s `outcome` line;
  - `prompt_cache.rs::CacheEntry::resume_from` (FT8.5), the mark the shift stage reads;
  - the tests `a_lifted_run_turns_the_request_into_a_shift_that_counts_the_moved_tokens_as_reused` (~line 1371) and `a_lift_that_finds_nothing_leaves_the_path_a_rewind` (~line 1398), which must stay green unchanged.
- change:
  1. `prompt_cache.rs`: add `fn assemble_shift(entry: &mut CacheEntry, lift: impl FnOnce(&CacheEntry, usize) -> Vec<MovedRun>)`. Body: `if let Some((_, resume_len)) = entry.resume_from { entry.moved = lift(entry, resume_len); }`. With no taken prefix there is nothing to shift: the entry is untouched and `lift` is not called. Why a closure: the real lift needs `&LoadedModel`, a test does not.
  2. `PromptCache::resume_at` becomes `fn resume_at(&mut self, mut entry: CacheEntry, widths: &[LayerPadRowWidths]) -> Result<(CacheEntry, CachePath), MissReason>` (the `lift` argument and the local `moved` go). On a successful rewind the path is `CachePath::Shift` when `entry.moved` is non-empty, else the rewind's own path (the rule it applies today); on a refused rewind it sets `entry.moved = Vec::new()` before `self.keep(stamp, entry)`, so a kept entry never carries lifted runs.
  3. `take_best_shifting`: the `outcome` line becomes `self.assemble_prefix(prompt_ids, key, min_similarity_milli).and_then(|mut entry| { if let Some(lift) = lift { assemble_shift(&mut entry, |held, from| lift(held, from)); } let resume_len = entry.resume_from.map_or(0, |(_, len)| len); self.resume_at(entry, widths).map(|(entry, path)| (entry, path, resume_len)) })`; behaviour identical.
- test: add in the `tests` module of `prompt_cache.rs`, reusing the 40/30 example and `lifted_run()` (one 15-token run, `old_start` 20, `new_start` 14):
  - `assemble_shift_lifts_before_the_rewind`: after `let mut entry = cache.assemble_prefix(..).expect(..)`, `assemble_shift(&mut entry, |held, from| { assert_eq!(from, 10); assert_eq!(held.state.cached_len, 40); vec![lifted_run()] })` leaves `entry.moved.len() == 1` and `entry.state.cached_len == 40`;
  - `assemble_shift_resume_reports_a_shift`: repeat the setup of the previous test (each test builds its own cache), then `cache.resume_at(entry, &shared_widths())` is `Ok((entry, CachePath::Shift))` with `entry.state.cached_len == 10`;
  - `assemble_shift_without_a_prefix_does_nothing` (sad path): `assemble_shift` on `state_with_ids(&[1, 2, 3])` (a `CacheEntry` whose `resume_from` is `None`) with a closure that increments a `Cell<usize>` leaves the cell at 0 and `entry.moved.is_empty()`;
  - `assemble_shift_refused_rewind_keeps_the_entry_without_lifted_runs` (sad path): store `CacheEntry::new(gemma_like_state(40), base_key())` the way `a_refused_rewind_keeps_the_entry_cached` stores its entry; for the prompt `(0..35).chain([900, 901])`, `assemble_prefix(..)` is `Ok(entry)`, `assemble_shift(&mut entry, |_, _| vec![lifted_run()])` sets one moved run, `cache.resume_at(entry, &gemma_like_widths())` is `Err(MissReason::RingSlackExceeded { rewind_rows: 5, slack_rows: SLACK })`, `cache.entries.len() == 1` and every held entry has `moved.is_empty()`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_6 cargo nextest run -p proxima-model-interop --features std -E 'test(/assemble_shift_/)'`
- expect: `4 passed`; the `test(/prompt_cache::tests::/)` filter is the baseline plus 4 with 0 failed
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: proxima-model-interop/src/generate/prompt_cache.rs
- commit: `refactor(interop): lift shifted chunks before the rewind`
- done when: both expect lines printed, clippy clean, `git diff --cached --stat` touches only `generate/prompt_cache.rs`
- do not: change `lift_chunks`, `plan_runs` or `rotate_rows`; change any existing test
- gpu: none

### 8.16 Assemble the starting entry from an ordered stage list

- id: FT8.16
- needs: FT8.4, FT8.6
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `prompt_cache.rs::PromptCache::take_best_shifting` (after FT8.6): its `Ok((mut entry, path, lcp))` report arm (`entry.clamp_prewarmed()`, the `shifted`, `held` and `reused` sums, the `CacheReport { .. }` literal) moves into the function this card adds;
  - `prompt_cache.rs::PromptCache::assemble_prefix`, `assemble_shift` and `resume_at` (FT8.15, FT8.5, FT8.6): the three stage pieces the new function calls; `CacheEntry::empty` and `CacheReport::miss` (~line 195);
  - `prompt_cache.rs::PromptCache::take_best` (~line 628) and `take_for_prewarm` (~line 597): the production callers of `take_best_shifting` after this card;
  - the tests `a_lifted_run_turns_the_request_into_a_shift_that_counts_the_moved_tokens_as_reused` (~line 1371), `a_lift_that_finds_nothing_leaves_the_path_a_rewind` and `a_refused_rewind_keeps_the_entry_cached` (~line 1913): they stay unchanged and now run through the new function.
- change:
  1. `prompt_cache.rs`: add `fn assemble(&mut self, kinds: &[AssembleKind], prompt_ids: &[u32], key: &CacheKey, widths: &[LayerPadRowWidths], min_similarity_milli: u32, lift: Lift<'_>) -> (CacheEntry, CacheReport)` on `PromptCache`. Body:
     - `let mut miss: Option<MissReason> = None;`
     - `let entry = kinds.iter().fold(CacheEntry::empty(*key), |held, kind| match kind { .. });` where the arms are exhaustive: `AssembleKind::Prefix`: `self.assemble_prefix(prompt_ids, key, min_similarity_milli)`, `Ok(taken)` gives `taken`, `Err(reason)` sets `miss = Some(reason)` and gives `held`; `AssembleKind::Shift`: `let mut shifted = held;` `assemble_shift(&mut shifted, |taken, from| lift(taken, from));` then `shifted`;
     - `let lcp = entry.resume_from.map_or(0, |(_, len)| len);`
     - `let outcome = match entry.resume_from { None => Err(miss.unwrap_or(MissReason::NoCommonPrefix)), Some(_) => self.resume_at(entry, widths).map(|(rewound, path)| (rewound, path, lcp)) };`
     - `Ok((mut rewound, path, lcp))`: the report arm moved verbatim from `take_best_shifting` (with `entry` renamed `rewound`), giving `(rewound, report)`; `Err(reason)` gives `(CacheEntry::empty(*key), CacheReport::miss(prompt_ids.len(), reason))`;
     - `self.last_report = Some(report)` and return `(entry, report)`.
     Import `AssembleKind` from `proxima_core::serving_state::assemble` at the module top if FT8.4 has not already.
  2. `take_best_shifting`: keep its doc and signature and replace its body by `let mut no_lift = |_: &CacheEntry, _: usize| Vec::new();` `let lift = lift.unwrap_or(&mut no_lift);` `let (entry, report) = self.assemble(&[AssembleKind::Prefix, AssembleKind::Shift], prompt_ids, key, widths, min_similarity_milli, lift);` `(report.miss.is_none().then_some(entry), report)`. A request with no lift runs the shift stage with a lift that finds nothing, which is a rewind, so `take_best` keeps today's behaviour. The old prefix and shift logic of this function is gone: the one implementation is `assemble`, used by the prewarm path (`take_best`), by the existing tests, and by the request path in a later card.
- test: add in the `tests` module of `prompt_cache.rs` (the 40/30 example, `shared_widths()`, `ANY_OVERLAP`, a `Cell<usize>` counting lift calls), numbers taken from `a_lifted_run_turns_the_request_into_a_shift_that_counts_the_moved_tokens_as_reused`:
  - `assemble_default_list_reproduces_the_worked_shift_example`: store the 40-token entry; `cache.assemble(&[AssembleKind::Prefix, AssembleKind::Shift], &prompt, &base_key(), &shared_widths(), ANY_OVERLAP, &mut lift)` with a lift that asserts `from == 10` and `held.state.cached_len == 40` and returns `vec![lifted_run()]` gives `report.path == CachePath::Shift`, `report.lcp == 10`, `report.shifted_tokens == 15`, `report.reused_tokens == 25`, `report.prefilled_tokens == 5`, `entry.state.cached_len == 10`, `entry.moved.len() == 1`;
  - `assemble_without_shift_keeps_the_rewind_path`: the same store with `[AssembleKind::Prefix]` and a lift that increments the cell: the cell is 0, `report.path == CachePath::Rewind`, `report.shifted_tokens == 0`, `report.reused_tokens == 10`, `entry.moved.is_empty()`;
  - `assemble_miss_returns_an_empty_entry_with_the_reason` (sad path): prompt `vec![900_u32; 30]`, `[Prefix, Shift]`: `entry.state.cached_len == 0`, `entry.state.layer_caches.is_empty()`, `report.miss == Some(MissReason::NoCommonPrefix)`, `report.path == CachePath::Miss`, `report.prefilled_tokens == 30`, `cache.entries.len() == 1`, the lift cell is 0;
  - `assemble_refused_rewind_returns_an_empty_entry` (sad path): store `CacheEntry::new(gemma_like_state(40), base_key())` the way `a_refused_rewind_keeps_the_entry_cached` stores its entry, prompt `(0..35).chain([900, 901])`, `[Prefix]`, `gemma_like_widths()`: `report.miss == Some(MissReason::RingSlackExceeded { rewind_rows: 5, slack_rows: SLACK })`, `entry.state.cached_len == 0`, `cache.entries.len() == 1`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_16 cargo nextest run -p proxima-model-interop --features std -E 'test(/assemble_default_|assemble_without_|assemble_miss_|assemble_refused_/)'`, then the same command with `-E 'test(/prompt_cache/)'`
- expect: `4 passed`, then the `prompt_cache` filter count equals its baseline (recorded before the edit) plus 4, 0 failed
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: proxima-model-interop/src/generate/prompt_cache.rs
- commit: `refactor(interop): assemble the cache entry from an ordered stage list`
- done when: both expect lines printed, clippy clean, `git diff --cached --stat` touches only `generate/prompt_cache.rs`
- do not: change the signature of `take_best` or `take_best_shifting`; change `prompt_cache_lookup` (a later card replaces it); change any existing test; add a stage kind
- gpu: none

### 8.17 Fold the assemble stages through one ordered runner

- id: FT8.17
- needs: FT8.16
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `prompt_cache.rs::PromptCache::assemble` (FT8.16): the `kinds.iter().fold(..)` this card replaces;
  - `proxima-core/src/serving_state/assemble.rs::AssembleKind` (FT8.12): the step type the live fold runs over;
  - the module's `use` block at the top of `prompt_cache.rs` (~line 33 at 4b4be6cf), where `core::convert::Infallible` joins if absent.
- change:
  1. `prompt_cache.rs`: add `fn run_assemble<Step, Entry, Failure>(steps: &[Step], entry: Entry, mut apply: impl FnMut(&Step, Entry) -> Result<Entry, Failure>) -> Result<Entry, Failure>` with the body `steps.iter().try_fold(entry, |held, step| apply(step, held))`. It is the one place the config order is applied; it is generic over the step so a stage the library does not define can be driven through it (the proof card).
  2. `PromptCache::assemble`: replace the `fold` by `let Ok(entry) = run_assemble(kinds, CacheEntry::empty(*key), |kind, held| -> Result<CacheEntry, Infallible> { Ok(match kind { .. }) });` with the same two arms. Add `use core::convert::Infallible;` at the module top if it is not already imported.
- test: add in the `tests` module of `prompt_cache.rs`:
  - `assemble_order_follows_config_order`: with `Step = &'static str`, `Entry = Vec<&'static str>`, `apply = |step, mut trace| { trace.push(*step); Ok::<_, ()>(trace) }`: `run_assemble(&["prefix", "shift", "load"], vec![], apply)` is `Ok(vec!["prefix", "shift", "load"])`, and `run_assemble(&["load", "prefix"], vec![], apply)` is `Ok(vec!["load", "prefix"])`;
  - `assemble_stops_on_first_error`: `apply` fails with `"shift refused"` at `"shift"`; over `["prefix", "shift", "load"]` the result is `Err("shift refused")` and the call count (a `Cell<usize>`) is 2;
  - `assemble_empty_list_runs_no_pipe`: `run_assemble(&[] as &[&str], entry, ..)` returns the entry untouched and calls the closure 0 times.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_17 cargo nextest run -p proxima-model-interop --features std -E 'test(/assemble_order_|assemble_stops_|assemble_empty_/)'`, then the same command with `-E 'test(/prompt_cache/)'`
- expect: `3 passed`, then the `prompt_cache` filter count equals its baseline (recorded before the edit) plus 3, 0 failed
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: proxima-model-interop/src/generate/prompt_cache.rs
- commit: `refactor(interop): fold the assemble stages through one runner`
- done when: both expect lines printed, clippy clean, `git diff --cached --stat` touches only `generate/prompt_cache.rs`
- do not: change what `assemble` returns; add a stage kind; change any existing test
- gpu: none

### 8.7 Run the live request through the assemble stages

- id: FT8.7
- needs: FT8.4, FT8.17
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `prompt_cache.rs::LoadedModel::run_decode_loop_through_cache` (~line 1202 at 4b4be6cf) from `let key = ..` (~line 1243) to the `prefill_through_runs` call: the lookup this card replaces; `found.unwrap_or_else(|| CacheEntry::empty(key))` is at ~line 1248;
  - `prompt_cache.rs::LoadedModel::prompt_cache_lookup` (~line 1047) as it stands when this card runs (an earlier slice may have added lines to it): its lock idiom (`let mut cache = self.prompt_cache.lock();`, ~lines 1056-1057 at main; the guard is `drop(cache)`d before the `debug!` event), its `cache.last_report = Some(report);` (~line 1064) and `cache.bloom_candidates(..)` (~line 1068) calls and its `debug!` event, all kept; its only caller is `run_decode_loop_through_cache` (`git grep -n prompt_cache_lookup main` finds the definition and that one call);
  - `prompt_cache.rs::PromptCache::assemble` (FT8.16, runner from FT8.17) and `assemble_kinds_for` (FT8.4);
  - `chunk_shift.rs::LoadedModel::prefill_through_runs` (~line 538): consumes `entry.moved` and the entry's cached length unchanged.
- change:
  1. `prompt_cache.rs`, `impl LoadedModel`: replace `prompt_cache_lookup` by `fn assemble_request(&self, kinds: &[AssembleKind], prompt_ids: &[u32], key: &CacheKey, widths: &[LayerPadRowWidths], waited: Duration, serving_config: &ServingConfig) -> CacheEntry`. Its body is the body of `prompt_cache_lookup` as it stands, every line kept, with only these changes: the `shifting` local is deleted; the lift closure stays (`let mut lift = |entry: &CacheEntry, from: usize| self.lift_chunks(entry, prompt_ids, from, widths, serving_config);`); the `take_best_shifting(..)` call becomes `cache.assemble(kinds, prompt_ids, key, widths, config.min_similarity_milli, &mut lift)` returning `(mut entry, mut report)`; the tail that cleared `prewarmed` and `branch_base` on an `Option` becomes `entry.prewarmed = None; entry.branch_base = None;`; the function returns `entry`. The lock is held across the whole assemble, exactly as `prompt_cache_lookup` held it across the lift; the `debug!` event keeps its field names; its doc comment moves with it.
  2. `run_decode_loop_through_cache`: replace the `prompt_cache_lookup` call and the `found.unwrap_or_else(..)` line by `let entry = self.assemble_request(&kinds, &ids, &key, &widths, pending.waited(), serving_config);`.
  3. `take_best_shifting` stays as FT8.16 left it: the prewarm path calls it through `take_best`, it holds no prefix or shift logic of its own, and its `lift` argument reaches the same `assemble` the request path runs. The lookup this card removes was its only production caller that passed a lift; nothing else is left to reroute.
- test: add in the `tests` module of `prompt_cache.rs` (the 40/30 example, `shared_widths()`, `ANY_OVERLAP`; the model-side method needs a loaded model, so these two tests drive the pieces it composes: the config to stage list mapping and the cache assemble, with the same lift as the live path):
  - `assemble_config_with_reuse_runs_the_shift_pipe`: with `ServingConfig { prompt_cache: PromptCacheConfig { byte_budget: 1 << 20, cache_reuse_min: 4, ring_rewind_slack: 256, ..PromptCacheConfig::off() }, ..ServingConfig::default() }`, `assemble_kinds_for(&config, true)` is `Ok(kinds)`; `cache.assemble(&kinds, &prompt, &base_key(), &shared_widths(), ANY_OVERLAP, &mut lift)` with a lift that asserts `from == 10` and returns `vec![lifted_run()]` gives `report.path == CachePath::Shift`, `report.shifted_tokens == 15`, `entry.moved.len() == 1`;
  - `assemble_config_without_reuse_keeps_the_rewind_pipe`: the same with `cache_reuse_min: 0`: `kinds` is `[Prefix]`, the lift cell (a `Cell<usize>`) stays 0, `report.path == CachePath::Rewind`, `report.reused_tokens == 10`, `entry.moved.is_empty()`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_7 cargo nextest run -p proxima-model-interop --features std -E 'test(/assemble_config_/)'`, then the same command with `-E 'test(/prompt_cache/)'`
- expect: `2 passed`, then the `prompt_cache` filter count equals its baseline (recorded before the edit) plus 2, 0 failed
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`; `git grep -n "prompt_cache_lookup\|cacheable = config.is_enabled()" -- proxima-model-interop/src` prints nothing
- stage: proxima-model-interop/src/generate/prompt_cache.rs
- commit: `refactor(interop): run the cache assemble pipes in config order`
- done when: both expect lines printed, the grep empty, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the decode loop or `prefill_through_runs`; add a stage kind, a load arm or a blend arm; change the `debug!` field names; drop any line an earlier slice added to the lookup
- gpu: none (the live path runs against real models in FT8.9 to FT8.11)

### 8.8 Proof: a stage the library does not define composes by list order

- id: FT8.8
- needs: FT8.7
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `prompt_cache.rs::run_assemble` (FT8.17) and `PromptCache::assemble_prefix` (FT8.15, FT8.5): the fold and the one library stage this test uses;
  - `chunk_shift.rs::CacheEntry::apply_moved` (~line 341 at 4b4be6cf) and `PrefixState::append_moved` (~line 299): a run is written onto an entry whose state is at the run's `new_start`, one `LayerRows` per attention layer, `LayerCache::new()` being a full (non-ring) layer;
  - `chunk_shift.rs::MovedRun` (~line 67), `ChunkRun` (~line 46) and `ring_checkpoint.rs::LayerRows` (~line 25): the shape of the run the test builds;
  - the `tests` module helpers `state_with_ids`, `base_key`, `enabled_config`, `stored_conversation_and_squashed_prompt` (~line 1319 to 1351).
- change:
  1. none in library code. The card adds one test (the stage is test code: a technique appears only as the proof that the hook expresses it, never in the library).
- test: add `a_stage_the_library_does_not_define_composes_by_list_order` in the `tests` module of `prompt_cache.rs` (about 40 lines). Setup, all in the test module:
  - `enum Stage { Prefix, Preload }`;
  - `preload_run()`: `MovedRun { run: ChunkRun { old_start: 0, new_start: 0, len: 8 }, ids: (500..508).collect(), layers: vec![LayerRows { layer: 0, k_even: vec![0.5; 32], k_odd: vec![0.25; 32], v: vec![0.125; 32] }] }`;
  - `layered_empty_entry()`: `CacheEntry::new(PrefixState { ids: Vec::new(), layer_caches: vec![LayerCacheState::Attention(LayerCache::new())], cached_len: 0 }, base_key())` (the hand-built stand-in for a fresh entry that has layers);
  - `assemble_with(order: &[Stage], cache: &mut PromptCache) -> (CacheEntry, usize)`: `run_assemble(order, layered_empty_entry(), apply)` where `apply` matches the stage: `Stage::Prefix` is `Ok(cache.assemble_prefix(&prompt, &base_key(), ANY_OVERLAP).unwrap_or(held))`; `Stage::Preload`, only when `held.state.cached_len == 0` (the increment sits inside the same guard, so a skipped stage counts 0), increments a `Cell<usize>` and calls `held.apply_moved(&preload_run())?`, then returns `Ok(held)` either way; the result unwraps with `expect`, and the second tuple field is the cell's count of placements.
  Assertions (prompt `stored_conversation_and_squashed_prompt().1`):
  - cache holding `state_with_ids(&(1..=40).collect::<Vec<u32>>())`, order `[Prefix, Preload]`: `entry.state.cached_len == 40`, `entry.resume_from.map(|(_, len)| len) == Some(10)`, placements 0;
  - empty cache, order `[Prefix, Preload]`: `entry.state.cached_len == 8`, `entry.state.ids == (500..508).collect::<Vec<u32>>()`, placements 1, `entry.restored_at == 8`, and layer 0's `k_even.len() == 32`;
  - cache holding the 40-token entry, order `[Preload, Prefix]` (the control that must differ): placements 1 and `entry.state.cached_len == 40` (the prefix stage ran last and replaced the preloaded rows), so the starting entry is decided by list order and by nothing else.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_8 cargo nextest run -p proxima-model-interop --features std -E 'test(/a_stage_the_library_does_not_define_composes_by_list_order/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`
- stage: proxima-model-interop/src/generate/prompt_cache.rs
- commit: `test(interop): compose an assemble stage the library lacks by order`
- done when: expect printed, clippy clean, `git diff --cached --stat` touches only `generate/prompt_cache.rs` and every added line is inside `mod tests`
- do not: add any non-test item; add a stage kind to proxima-core; run a model
- gpu: none

### 8.9 Oracle: prefix reuse at a start position, gemma4 e2b (dense)

- id: FT8.9
- needs: FT8.7, FT0 (all cards)
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity` (~line 667 at 4b4be6cf) and `llama_cases` (~line 616): the loader (`parse_complete`, `LoadedModel::load`), `generate_from_ids`, `ids_of` (~line 651), `first_divergence` (~line 659) and the `LLAMA_GENERATED_TOKENS` constant (~line 608) this card reuses;
  - `tests/fixtures/llama-parity/gemma4_e2b/followup_ids.json` (vendored by FT0; main has only `llama_ids.json` next to it);
  - `arch_data_baseline.rs` static `GEMMA4_E2B` (~line 54) and `Checkpoint::open`;
  - `prompt_cache.rs::LoadedModel::last_prompt_cache_report` (~line 1020, public): the report of the last cached request.
- change:
  1. `tests/arch_data_baseline.rs` (the file that owns the `Checkpoint` statics, which are private to it): add `struct FollowupCase { turn1_prompt_ids: Vec<u32>, turn2_prompt_ids: Vec<u32>, turn2_generated_ids: Vec<u32> }` and `fn followup_cases(checkpoint: &Checkpoint) -> Vec<FollowupCase>` reading `tests/fixtures/llama-parity/<name>/followup_ids.json` (a JSON array; each element holds `turn1.prompt_ids`, `turn2.prompt_ids`, `turn2.generated_ids`; read through the existing `ids_of`). A missing file or key panics with the path (fails, never skips); zero elements panics.
  2. Add `fn prefill_start_position_oracle(checkpoint: &Checkpoint)`: load the model as `llama_parity` does, with `ServingConfig { prompt_cache: PromptCacheConfig { byte_budget: 1 << 30, ..PromptCacheConfig::off() }, ..ServingConfig::default() }`; per case: `model.generate_from_ids(&case.turn1_prompt_ids, 8, &config, ..)` to store the entry (bind its returned `Vec<u32>` as `turn1_generated`; the call is `let (turn1_generated, _text, _stopped) = ..`), then `model.generate_from_ids(&case.turn2_prompt_ids, LLAMA_GENERATED_TOKENS, &config, &mut |event| ..)` whose callback records the `prompt_tokens` of the first `Phase::Prefill` event (add `Phase` to the `proxima_model_interop` import).
  3. Add `fn common_prefix_len(left: &[u32], right: &[u32]) -> usize` (the count of leading equal ids) beside `first_divergence`. Assertions per case, every expected value computed from the fixture and the turn-1 call's own return, never a bound:
     - `stored_ids` = `case.turn1_prompt_ids` followed by `turn1_generated[..turn1_generated.len() - 1]`, where `turn1_generated` is the `Vec<u32>` the turn-1 `generate_from_ids` returned (the entry holds every generated token except the last, which is sampled but never forwarded: `decode.rs` `final_ids` / `forwarded_generated`, ~line 6285 at main);
     - `expected_lcp = common_prefix_len(&stored_ids, &case.turn2_prompt_ids).min(case.turn2_prompt_ids.len() - 1)` (the cap is `take_best_shifting`'s `resume`, `prompt_cache.rs` ~line 650);
     - `model.last_prompt_cache_report()` is `Some(report)` with `report.miss == None`, `report.lcp == expected_lcp`, `report.reused_tokens == expected_lcp` and `report.prefilled_tokens == case.turn2_prompt_ids.len() - expected_lcp` (cache reuse is off in this config, so `path` is `CachePath::Rewind` and nothing is shifted);
     - the recorded first-`Phase::Prefill` `prompt_tokens == case.turn2_prompt_ids.len() - expected_lcp`, which equals `report.prefilled_tokens` (`run_decode_loop_through_cache` hands `ids[resumed_at..]` to the decode loop, whose event reports that slice's length);
     - `expected_generated = case.turn2_generated_ids.len().min(LLAMA_GENERATED_TOKENS)`; the fixture panics with the case index if it is 0; `generated.len() == expected_generated` and `first_divergence(&case.turn2_generated_ids[..expected_generated], &generated) == None`.
     Control that must fail: a second `generate_from_ids` of the same turn-2 prompt under a config with `PromptCacheConfig::off()` records a first-`Phase::Prefill` `prompt_tokens == case.turn2_prompt_ids.len()` exactly, and that value differs from the reused run's `case.turn2_prompt_ids.len() - expected_lcp` because `expected_lcp > 0` (the test asserts `expected_lcp > 0` as the fixture precondition, panicking with the case index if the vendored turn-2 prompt shares nothing with the stored turn 1). The expected values are derived from reading the code and the fixture, not from a run; if a recorded value differs from its derivation, the premise is false: stop and report the case index, the derived value and the recorded value.
  4. One test, `#[test]`: `prefill_start_position_oracle_gemma4_e2b` calling `prefill_start_position_oracle(&GEMMA4_E2B)`.
- test: as above; the assertions are the five exact equalities of item 3 (`report.lcp`, `report.reused_tokens`, `report.prefilled_tokens`, the first-prefill `prompt_tokens`, `generated.len()` with `first_divergence(..) == None`) and the cache-off control equal to the full turn-2 length.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_8_9 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_9 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/prefill_start_position_oracle_gemma4_e2b/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_8_9/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: proxima-model-interop/tests/arch_data_baseline.rs
- commit: `test(interop): prefix reuse at a start position matches recorded ids`
- done when: `1 passed` printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`
- do not: run another model-loading process; edit the fixtures; change `llama_parity`; query Ollama or llama.cpp (the ids are replayed from the vendored file)
- gpu: one run, waiting for a quiet box (the peer-gate check in SPEC "machine safety")

### 8.10 Oracle: prefix reuse at a start position, gemma4 26b (MoE)

- id: FT8.10
- needs: FT8.9
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `tests/arch_data_baseline.rs::prefill_start_position_oracle` (FT8.9) and the static `GEMMA4_26B` (~line 48 at 4b4be6cf; `batiai/gemma4-26b:latest`, expert_count 128, blob `sha256-ea549b76...`);
  - `tests/fixtures/llama-parity/gemma4_26b/followup_ids.json` (vendored by FT0).
- change:
  1. `tests/arch_data_baseline.rs`: add the test `prefill_start_position_oracle_gemma4_26b` calling `prefill_start_position_oracle(&GEMMA4_26B)`. Nothing else.
- test: `prefill_start_position_oracle_gemma4_26b`, asserting what `prefill_start_position_oracle` asserts (FT8.9 item 3) on the 26b MoE checkpoint, with the same exact values: `report.miss == None`, `report.lcp == expected_lcp`, `report.reused_tokens == expected_lcp`, `report.prefilled_tokens == turn2_len - expected_lcp`, first-prefill `prompt_tokens == turn2_len - expected_lcp`, `generated.len() == expected_generated` with `first_divergence(..) == None`, and the cache-off control `prompt_tokens == turn2_len`. `expected_lcp` and `expected_generated` are computed by the shared function from this checkpoint's `followup_ids.json`; nothing here is a bound.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_8_10 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_10 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/prefill_start_position_oracle_gemma4_26b/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_8_10/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: proxima-model-interop/tests/arch_data_baseline.rs
- commit: `test(interop): gemma4 moe prefix reuse matches recorded ids`
- done when: `1 passed` printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`
- do not: run another model-loading process (this checkpoint is 13.3 GB); edit the fixtures; change `prefill_start_position_oracle`
- gpu: one run, waiting for a quiet box (the peer-gate check in SPEC "machine safety")

### 8.11 Oracle: prefix reuse at a start position, granite 3.1 MoE

- id: FT8.11
- needs: FT8.9, FT0 (all cards: the granite family profile plus descriptor and the recorded follow-up ids)
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `tests/arch_data_baseline.rs::prefill_start_position_oracle` (FT8.9);
  - the `Checkpoint` static whose `architecture` is `"granitemoe"` (added by FT0's granite card; locate it with `git grep -n 'architecture: "granitemoe"' -- proxima-model-interop/tests/arch_data_baseline.rs`; it points at the blob `ollama show --modelfile granite3.1-moe:1b` names on its `FROM` line, `sha256-cd60b3e8...`, 1.4 GB);
  - `tests/fixtures/llama-parity/<that static's name>/followup_ids.json` (vendored by FT0).
- change:
  1. `tests/arch_data_baseline.rs`: add the test `prefill_start_position_oracle_granite_moe` calling `prefill_start_position_oracle(&<the granitemoe static>)`. Nothing else. If the grep finds no such static, or the fixture file is absent, stop and report: the granite profile card is not done.
- test: `prefill_start_position_oracle_granite_moe`, the same exact assertions as FT8.9 item 3 on the granite checkpoint (`report.lcp == expected_lcp`, `report.reused_tokens == expected_lcp`, `report.prefilled_tokens == turn2_len - expected_lcp`, first-prefill `prompt_tokens == turn2_len - expected_lcp`, `generated.len() == expected_generated` with `first_divergence(..) == None`, cache-off control `prompt_tokens == turn2_len`), with `expected_lcp` and `expected_generated` computed from that checkpoint's `followup_ids.json` by the shared function. No granite-specific code path exists in the library; the model reaches the generic prefix reuse through its profile and descriptor.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_8_11 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_8_11 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/prefill_start_position_oracle_granite_moe/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_8_11/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets`
- stage: proxima-model-interop/tests/arch_data_baseline.rs
- commit: `test(interop): granite moe prefix reuse matches recorded ids`
- done when: `1 passed` printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`
- do not: run another model-loading process; edit the fixtures; add a granite branch anywhere in `src/`
- gpu: one run, waiting for a quiet box (the peer-gate check in SPEC "machine safety")

---

## spec drift

- D1. TASKS slice 8 says `prefill.assemble` lowers prefix, shift, load in config order. At main, `take_best_shifting` fuses prefix and shift: `resume_at` lifts chunks from the entry BEFORE `entry.resume` rewinds it, because the rows past the shared prefix are gone after the rewind. A free-order list needs the rewind deferred to the end of the list, so FT8.15 splits the take out of `resume_at`, FT8.5 adds the field `CacheEntry::resume_from` (carrying the stamp as well as the length so a refused rewind still puts the entry back under its own stamp; the earlier cut passed no stamp and could not), and FT8.6 moves the lift out of `resume_at`.
- D2. `start_position` is independent of the cache in the spec. At main every flow has `start_position == PrefixState::len()` (`seed_cached_len`, `decode.rs` ~line 3011): prefix reuse, chunk shift (`append_moved` refuses `cached_len != new_start`) and loaded rows all leave the cache holding exactly the rows before the start. FT8.1 keeps the field as the spec requires; no card asserts the two differ.
- D3. The live fold runs over core's payload-free `AssembleKind`, not interop's `AssembleStep`: FT8.4 maps the configured list once through `kind_of`, an exhaustive `match` of the two variants FT2.2 declares, so the fold has no wildcard arm and no refusal arm to leave untested. A variant the grammar gains without a pipe behind it fails to compile in `kind_of`; the refusal text for a stage with no pipe returns with the card that adds such a variant.
- D4. `prompt_cache_lookup` has exactly one caller at main (`git grep -n prompt_cache_lookup main`), so FT8.7 deletes it; the earlier cut kept it on the claim that the prewarm path calls it, which it does not (`take_for_prewarm` calls `take_best`). Its lift was the only production use of `take_best_shifting`'s `Some(lift)` arm, so FT8.16 turns `take_best_shifting` into a one-call wrapper over `PromptCache::assemble` before FT8.7 removes the lookup: the prefix and shift logic exists once, in `assemble`, and the prewarm path, the existing shift tests and the request path all run it.
- D5. `DAD C7` lists `decode.rs` ~line 3831 as a call site that stays on the wrapper. FT8.3 moves the prefill site (~line 3829 at 4b4be6cf) onto `build_position_inputs_at` on purpose: it is the live consumer that makes a start position a positions slice. If DAD C7 lands a different parameter list, FT8.3 changes that one call.
- D6. Fixture keys follow FT0: nested `turn1` and `turn2` in a one-element array.
- D7. A configured list that holds shift while `cache_reuse_min` or `ring_rewind_slack` is zero is refused (`ShiftNeedsReuse`), not silently run as a no-op.
- D8. Every place decision is a pure function in proxima-core: FT8.2 holds the assemble decision. FT8.4 and later in interop only map facts in and apply the decision out.
- D9. The size rule (at most one consumed item per card) cuts the assemble module into FT8.12 (kind), FT8.13 (error), FT8.14 (decision) and FT8.2 (function), and the prompt-cache work into FT8.15 (take), FT8.5 (mark), FT8.6 (shift), FT8.16 (ordered assemble), FT8.17 (runner) and FT8.7 (request). The report code the earlier cut extracted into a free function moves straight into `assemble` instead, and the earlier give-back and finish helpers fold into `resume_at`, so no card adds an item that only a later card uses.

## slice exit

- Every card FT8.1 to FT8.17 printed its expect line.
- `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/assemble_plan_/)'` prints `7 passed`.
- `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/assemble_kind_|assemble_refusal_|assemble_decision_/)'` prints `3 passed`.
- `cargo nextest run -p proxima-model-interop --features std -E 'test(/prefix_take_/)'` prints `4 passed`, and the same command with `test(/assemble_shift_/)` prints `4 passed`.
- `cargo nextest run -p proxima-model-interop --features std -E 'test(/assemble_kind/)'` prints `1 passed`.
- `cargo nextest run -p proxima-model-interop --features std -E 'test(/assemble_order_|assemble_stops_|assemble_empty_|assemble_default_|assemble_without_|assemble_miss_|assemble_refused_|assemble_config_|a_stage_the_library_does_not_define_composes_by_list_order/)'` prints `10 passed`.
- `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/prefill_start_position_oracle_/)'` prints `3 passed` (checked once, on a quiet box, one model at a time).
- `cargo check -p proxima-core --no-default-features --features alloc` is clean.
- `git grep -n "prompt_cache_lookup\|cacheable = config.is_enabled()" -- proxima-model-interop/src` prints nothing.
- `git grep -n "prefill_start_position_oracle_qwen\|prefill_start_position_oracle_openchat" -- proxima-model-interop/tests/arch_data_baseline.rs` prints nothing.
