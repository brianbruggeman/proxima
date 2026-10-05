# slice 1 (re-cut): FSM generic over its entry, at tier 1; accept rule as data; action worked example

anchors read at main a7c08c4c

Rules: CARDS.md (binding), the pipeline-as-data SPEC (hook 11, the step: enter, propose, shape,
accept, commit; the generic entry), `sketches/14-action-speculation.md`, and the owner direction
that a hook is configured data and a technique appears only as the proof that a hook expresses
it. The spec directory is `proxima-tensor/specs/fsm-techniques/`, committed to main; every path in
this file, spec files included, is relative to the main checkout. A card that writes a spec file
writes it under that directory, stages it with `git add`, checks `git diff --cached --stat` and
commits on main with `git commit`; every command that reads a spec file names it by its
repo-relative path. Cards that run cargo run from the proxima repo root (the checkout holding
main). Cross-file `needs` use the ids as they stand in
`tasks/`.

Test models (owner): gemma4 (dense or MoE), granite (MoE). No qwen model of any kind appears in a
test, fixture, oracle or validation of this file. Nothing in this slice loads a model except
slice exit row 7, which names gemma4 E2B only.

## id map (old card to new card)

| old | new | verdict |
|---|---|---|
| FT1.1 | FT1.1 | kept; the derivation text gains the accept-rule values of each policy, the RESULT line is unchanged |
| FT1.2 | FT1.2 | kept; module doc no longer carries the model-specific blocked-on paragraph; the module doc and `ServingFsmError` moved out into the new card FT1.13 so FT1.2 adds one public item (the size rule) |
| (new) | FT1.13 | new: creates `serving_state.rs` with the module doc and `ServingFsmError`; FT1.2 needs it, so it sits before FT1.2 in this file |
| FT1.3 | FT1.3 | kept |
| FT1.4 | FT1.4 | kept; the moved test's doc comment no longer names a model |
| FT1.5 | FT1.5 | kept |
| FT1.6 | FT1.6 | kept |
| FT1.7 | FT1.7 | kept |
| FT1.8 | FT1.8 | kept; the fixture module declaration carries the test-only lint allowance |
| FT1.9 | FT1.9 | kept |
| FT1.10 | FT1.10 | recut: the closed enum of technique names is replaced by the generic accept rule (similarity floor, minimum run, row cap) and its pure function |
| FT1.11 | FT1.11 | recut: the nine technique-named settings fields are replaced by the three fields of that rule; builder, TOML and env round trips |
| FT1.12 | FT1.12 | recut: the library lowering of three named techniques is replaced by three proof tests that run the three techniques as configuration values |

## dropped

None. Every old card is kept or recut; no card body was removed.

## what the recut changes, in English

- The hook is "the accept rule is data". `ServingState::accept_rows` (FT1.7) takes the count; the
  rule that produces the count is `AcceptRule` (FT1.10), three scalars, loadable from builder, TOML
  and env (FT1.11). Nothing in the library names a technique.
- A technique appears only in tests, as configuration values driven through the hook against the
  action fixture (FT1.9 with closures, FT1.12 with loaded values). The counts come from the worked
  example (FT1.1) and are unchanged, so the worked example is not re-derived.
- The old `speculation` section clashed in name with the `speculative` section (the token n-gram
  set). The new section is `accept_rule`, with env prefix `PROXIMA_ACCEPT_RULE`.
- `AcceptRule` is all scalars and `Copy`. `ServingConfig` is `Copy` (`proxima-model-interop/src/serving.rs`
  ~line 719), so a rule can ride in it later without a list type; a list of rules, if a later slice
  needs one, needs a config type that holds lists, and is not built here.
- Sketch 14 gaps. G1 (the FSM cannot be named from outside the crate): FT1.13, FT1.2-FT1.6. G2 (nothing in
  proxima reads the consumer's accept settings): FT1.10-FT1.12 give proxima one generic, readable
  rule. G3 (the verify snapshot is cloned and never read) and G4 (the FSM verifies `draft.len()` rows
  and drops the last accepted row where the live loop evaluates one more) are behaviour the move
  preserves byte for byte; FT1.3 and FT1.4 are the proof of that.

Green rule applied: the new `proxima-core::serving_state` is built beside the old
`proxima-model-interop/src/serving_fsm.rs` (cards 1.13, 1.2-1.4); interop switches its one consumer to the
new module (1.5); only then is the old module deleted and the re-export added (1.6).

Behaviour carried over unchanged from main:
- `accept` returns `Rollback { snapshot: row_caches[accepted], to: row_tokens[accepted] }` on a
  partial match, and `Accept { n, next: draft[n-1], cache: row_caches[n-1] }` on a full match;
- `Verify.snapshot` is written by `enter_verify` and never read by any transition. Keep both.

Test module paths: every test lives in `proxima-core/src/serving_state.rs` `mod tests` (path
`serving_state::tests::...`) or `proxima-core/src/accept_rule.rs` (`accept_rule::tests::...` and
`accept_rule::config_tests::...`), because nextest filters match the module-qualified name. A later
slice adds `serving_state/sansio_tests.rs`.

Test-name families (each filter below is an unanchored regex over the qualified name, so a name
must never contain another family's text):
- `serving_fsm_error_` (1 test, card 1.13);
- `serving_state_` (4 tests, after card 1.4);
- `action_speculation_` (3 tests, card 1.9);
- `accept_rule_` (4 tests in card 1.10, 4 more in card 1.11);
- `configured_rule_` (3 tests, card 1.12).

Card list: 1.1 action worked example; 1.13 (placed before 1.2, which needs it) the serving fsm
error and the new module; 1.2-1.4 move the FSM into proxima-core; 1.5-1.6 switch
interop and delete the old module; 1.7 `accept_rows`; 1.8 action fixture; 1.9 the three
`action_speculation_` tests; 1.10 the accept rule and its pure function; 1.11 the accept rule's
config surface; 1.12 the three techniques as configuration values.

### 1.1 Worked example action speculation (the action run, by hand)

- id: FT1.1
- needs: FT0.14 (creates `worked-examples.md`)
- budget: 20 min
- crate(s): none (docs)
- read first: `proxima-tensor/specs/fsm-techniques/worked-examples.md`
  (conventions at the top of the worked-example block in
  `tasks/00-oracles-and-worked-examples.md`); every input the derivation needs is in this card
- change: append `## action speculation` to
  `proxima-tensor/specs/fsm-techniques/worked-examples.md`, with the inputs, the derivation and the RESULT line below.
  Inputs:
  - environment `E`: `values: BTreeMap<u8, i64>` (default empty, a missing key reads 0) and
    `version: u64`; `apply(action)`: `values[key] += delta; version += 1`;
  - action `A(key, delta, version)`, the entry type;
  - true policy `true_action(E) = A(key = E.version % 3, delta = (E.values[key] rem 5) + 1, version = E.version)`;
  - guessing drafter `guessed_action(E) = true_action(E)` except `delta + 1` when `E.version % 4 == 3`;
    the draft runs on a clone, applying its own guesses;
  - run: 12 steps, rows per draft = min(4, remaining); a round whose index `r` satisfies `r % 3 == 2`
    drafts from the environment as it was at the start of the previous round (a stale snapshot);
  - verify: observed row i = `true_action` on `E_i`, where `E_0` is the real environment and
    `E_(i+1) = E_i.apply(draft[i])`; row cache i = `E_i.apply(observed[i])`.
  Sequential run, 12 actions `A(key,delta,version)`: S0 (0,1,0), S1 (1,1,1), S2 (2,1,2), S3 (0,2,3),
  S4 (1,2,4), S5 (2,2,5), S6 (0,4,6), S7 (1,4,7), S8 (2,4,8), S9 (0,3,9), S10 (1,3,10), S11
  (2,3,11); final environment `{0:10, 1:10, 2:10}`, version 12. Show the arithmetic of each delta
  (for example S3: key 0 holds 1, 1 rem 5 = 1, delta 2).
  Each policy below is also stated as the three values of one accept rule: similarity floor,
  minimum run, row cap. The accepted count of a round is the length of the
  leading run of rows whose similarity to the observed row is at least the floor, cut at the row
  cap, and zero when that run is shorter than the minimum run.
  Equality policy (row accepted iff action and version equal; rule values: floor 1.0 under exact
  equality, minimum run 0, row cap 8, which never binds at 4 rows), rounds:
  - r0: the four drafted rows = S0, S1, S2, A(0,3,3) (version 3 guesses wrong); observed S0..S2 equal,
    observed[3] = A(0,2,3) differs; accepted 3, Rollback; commits S0..S2 and S3 (4 steps);
  - r1: from E4, the drafted rows = S4, S5, S6, A(1,5,7); accepted 3, Rollback; commits S4..S7 (8 steps);
  - r2 (stale: `2 % 3 == 2`): drafts from E4 while the real environment is E8; the first drafted row
    S4 differs from the first observed row S8; accepted 0, Rollback; commits S8 (9 steps);
  - r3: from E9, rows = min(4, 3) = 3: the drafted rows = S9, S10, A(2,4,11); accepted 2, Rollback; commits S9,
    S10, S11 (12 steps).
  - totals: rounds 4, rejected rounds 4, accepted rows 3 + 3 + 0 + 2 = 8.
  Anchor+macro policy (rule values: floor 1.0, minimum run 3, no row cap; the aligned prefix must
  reach 3, else accepted 0):
  - r0: 3; r1: 3; r2: 0 (stale); r3: aligned 2 < 3 so accepted 0, commits S9 (10 steps);
  - r4 (from E10, rows 2): the first drafted row S10 is aligned, the second A(2,4,11) is not; aligned 1 < 3, accepted 0, commits S10 (11);
  - r5 (stale, `5 % 3 == 2`, from the start of r4 = E10, rows 1): the only drafted row S10 differs from observed S11; accepted 0, commits S11 (12);
  - totals: rounds 6, rejected rounds 6, accepted rows 6.
  Verifier-exact policy (row accepted iff the whole entry is equal; rule values: floor 1.0 under a
  similarity that is the fraction of the three fields key, delta, version that agree, minimum run
  0, no row cap): same as equality, totals rounds 4, rejected 4, accepted rows 8.
  The RESULT line is verbatim, one line:
  `RESULT action speculation: sequential_final={0:10,1:10,2:10} v12; equality rounds=4 rejected=4 accepted_rows=8; anchor_macro(min_skip=3) rounds=6 rejected=6 accepted_rows=6; verifier_exact rounds=4 rejected=4 accepted_rows=8`
- test: none
- validate: `grep -c -F 'RESULT action speculation: sequential_final={0:10,1:10,2:10} v12; equality rounds=4 rejected=4 accepted_rows=8; anchor_macro(min_skip=3) rounds=6 rejected=6 accepted_rows=6; verifier_exact rounds=4 rejected=4 accepted_rows=8' proxima-tensor/specs/fsm-techniques/worked-examples.md`
- expect: `1`
- also green: n/a
- done when: expect printed, and `git diff --cached --stat` lists only `proxima-tensor/specs/fsm-techniques/worked-examples.md`
- do not: write code
- stage: `proxima-tensor/specs/fsm-techniques/worked-examples.md`, staged with `git add`
- commit: `docs(fsm): derive the action speculation worked example`
- gpu: none

### 1.13 Create proxima-core serving_state with the serving fsm error

- id: FT1.13
- needs: none
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-model-interop/src/serving_fsm.rs` (the file being moved: `ServingFsmError` ~line 77,
    at a7c08c4c);
  - `proxima-core/src/lib.rs` (`pub mod ring;` ~line 44 and `pub use factory::{Composition, Factory,
    FactorySpec};` ~line 87, the two insertion points).
- change: why this card exists apart from the state type: the error is the return type of every
  transition, so it lands first and the state type lands in the next card with one public item.
  1. `proxima-core/src/serving_state.rs` (new). Write the `//!` module doc as exactly two
     paragraphs and do not copy `serving_fsm.rs` lines 1-34 (they describe items that do not exist in
     the library at this commit: `draft_prompt_lookup` stays out of the library, and
     `ServingState` and its transitions land in later commits, so any intra-doc link to them would
     be broken). Paragraph one: "The serving control plane as one sans-IO state machine over the
     algebra: every transition performs exactly one program evaluation, or placement copies only, and
     the per-layer cache lives inside the variant a transition returns rather than in a cache object
     patched from outside." Paragraph two: "The tests below prove the control flow (drafting,
     acceptance counting, cache placement, rollback-to-resume) against a fake target function instead
     of a model program." Use no intra-doc links (plain backticks only) and no reference to any
     other file or step. Do
     not copy the `#![allow(dead_code)]` block (the comment and attribute, ~lines 36-38). Then
     `use thiserror::Error;` and `pub enum ServingFsmError` (derives `Debug, Error, PartialEq, Eq`)
     with only `IllegalTransition { attempted: &'static str }` (message `serving fsm: {attempted} is
     not legal from the current state`). The variant doc is copied from `serving_fsm.rs` with one
     edit: the intra-doc link ``[`ServingState::advance_decode`]`` becomes plain backticks
     (``(for example, `ServingState::advance_decode` called on `Prefill`)``), because `ServingState`
     does not exist in proxima-core until the next card. `NotSupported` is not
     copied (a `git grep -n NotSupported` over `proxima-model-interop/src` finds only its
     definition, ~line 87, so nothing constructs it).
  2. `proxima-core/src/lib.rs`: after `pub mod ring;` add `#[cfg(feature = "alloc")] pub mod serving_state;`
     (two lines, matching the style of `pub mod arena;`). After the
     `pub use factory::{Composition, Factory, FactorySpec};` line add
     `#[cfg(feature = "alloc")] pub use serving_state::ServingFsmError;` as two lines.
  3. Test module at the end of `serving_state.rs`: `#[cfg(test)] mod tests { use super::*; ... }`
     with one test, `serving_fsm_error_names_the_attempted_transition`. proxima-core is `no_std`
     without the `std` feature (`proxima-core/src/lib.rs` line 36), so the message is built with
     `alloc::format!` (`extern crate alloc;` is at `lib.rs` line 39), never a bare `format!`.
- test: `serving_fsm_error_names_the_attempted_transition` asserts
  `alloc::format!("{}", ServingFsmError::IllegalTransition { attempted: "accept" })` equals
  `"serving fsm: accept is not legal from the current state"`, and that
  `ServingFsmError::IllegalTransition { attempted: "accept" }` is not equal to
  `ServingFsmError::IllegalTransition { attempted: "resume" }`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_13 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/serving_fsm_error_/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc`
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
- do not: touch `proxima-model-interop`; run rustfmt or `cargo fmt` on any `src` file
- stage: `proxima-core/src/serving_state.rs`, `proxima-core/src/lib.rs`
- commit: `feat(core): add the serving fsm error type`
- gpu: none

### 1.2 Add the serving state generic over its entry (types and simple transitions)

- id: FT1.2
- needs: FT1.13
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-model-interop/src/serving_fsm.rs` (the file being moved: `ServingState` ~line 48,
    `impl<Cache> ServingState<Cache>` ~114, tests from `mod tests` ~285, at a7c08c4c);
  - `proxima-core/src/serving_state.rs` (from card 1.13: the module doc, `ServingFsmError`, the test
    module) and `proxima-core/src/lib.rs` (the `pub use serving_state::ServingFsmError;` line).
- change:
  1. `proxima-core/src/serving_state.rs`: below `ServingFsmError`, add `use alloc::vec::Vec;` at the
     module top with the other import, then:
     - `pub enum ServingState<Entry, Cache>` with the same six variants and variant docs as
       `serving_fsm.rs`, derives `Debug, Clone, PartialEq`. The doc on the enum itself is not copied
       (the original says "the one control-plane type this plan admits", a pointer into a plan);
       write instead: "Which evaluation shape is legal right now. An enum whose transitions
       consume `self` states that directly; no existing proxima primitive (pipe, cache or config)
       expresses it more cheaply."; `Prefill { positions: Vec<Entry>, cache: Cache }`,
       `Decode { last: Entry, cache: Cache }`, `Verify { draft: Vec<Entry>, snapshot: Cache, cache: Cache }`,
       `Accept { n: usize, next: Entry, cache: Cache }`, `Rollback { snapshot: Cache, to: Entry }`,
       `Done { cache: Cache }`;
     - `impl<Entry, Cache> ServingState<Entry, Cache>` holding, copied with `Entry` in place of
       `u32` and `pub` in place of `pub(crate)`: `start`, `advance_prefill`, `advance_decode`,
       `enter_verify` (still `where Cache: Clone`), `finish`. `accept`, `resume` and `rollback`
       come in card 1.3.
  2. `proxima-core/src/lib.rs`: change the line `pub use serving_state::ServingFsmError;` to
     `pub use serving_state::{ServingFsmError, ServingState};` (its `#[cfg(feature = "alloc")]` line
     stays). The prompt lookup drafter is not copied into the library at all; card 1.4 moves it into
     `mod tests` as a test-only proof.
  3. Test module in `serving_state.rs`: above the existing `mod tests`, copy the three lines that
     sit above `mod tests` in `serving_fsm.rs` after `#[cfg(test)]` (the three-line comment and
     `#[allow(clippy::unwrap_used, clippy::expect_used)]`). `use super::*` brings in `Vec` but not the
     `vec!` macro, so every vec literal in a test of this crate is written `alloc::vec![...]`, never
     a bare `vec![...]`. Add one test in that module, `walk_prefill_decode_verify_finish` (no
     `serving_state_` prefix, so the four-name count is reached only at the end of the move):
     `ServingState::start(alloc::vec![1_u32, 2], 0_u8)`,
     then `.advance_prefill(3, 1_u8)` and `.advance_decode(4, 2_u8)` (each returns a `Result`, so
     `.expect("legal")` each), then `.enter_verify(alloc::vec![5])` (also `.expect`) equals
     `Verify { draft: alloc::vec![5], snapshot: 2, cache: 2 }`; its `.finish()` equals `Done { cache: 2 }`;
     and that `Done` state's `.advance_decode(6, 3)` equals
     `Err(ServingFsmError::IllegalTransition { attempted: "advance_decode" })`. This exercises every
     item the card adds.
- test: `walk_prefill_decode_verify_finish` asserts the walk above.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_2 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/walk_prefill_/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc`
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
- do not: touch `proxima-model-interop`; run rustfmt or `cargo fmt` on any `src` file
- stage: `proxima-core/src/serving_state.rs`, `proxima-core/src/lib.rs`
- commit: `feat(core): add the serving state generic over its entry`
- gpu: none

### 1.3 Move accept, resume and rollback into serving_state (with 2 tests)

- id: FT1.3
- needs: FT1.2
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-model-interop/src/serving_fsm.rs` (`accept` ~line 186-237, `resume` ~241, `rollback`
    ~253; tests ~288-489, at a7c08c4c);
  - `proxima-core/src/serving_state.rs` (from card 1.2).
- change:
  1. `serving_state.rs`: add `impl<Entry: PartialEq + Clone, Cache> ServingState<Entry, Cache>`
     holding `pub fn accept(self, row_tokens: &[Entry], row_caches: Vec<Cache>) -> Result<Self, ServingFsmError>`:
     the body of `serving_fsm.rs::accept` with two changes, `next: draft[accepted - 1].clone()` and
     `to: row_tokens[accepted].clone()`. Add `resume` and `rollback` copied from `serving_fsm.rs`
     into the existing `impl<Entry, Cache>` block (they need no bounds).
  2. Tests in the same `mod tests`: copy `FakeCache` (its doc comment, derive, struct and `impl` with
     `empty` and `advanced_by`, ~lines 288-311), then `walkthrough_drives_every_legal_transition`
     (~line 318, with its doc comment) as `serving_state_walkthrough_drives_every_legal_transition`
     and `accept_rejects_mismatched_or_empty_draft` (~line 461, with its doc comment) as
     `serving_state_accept_rejects_mismatched_or_empty_draft`, bodies verbatim. If type inference
     fails on an untyped integer literal, add a `_u32` suffix to the first literal of that test,
     with no other change.
- test: the walkthrough asserts every legal transition `Prefill -> Decode -> Decode -> Verify ->
  Accept -> Decode -> Verify -> Rollback -> Decode -> Done` and the three illegal-call errors
  (`advance_prefill`, `resume`, `advance_decode`, each `IllegalTransition`); the
  accept test asserts `Err(IllegalTransition { attempted: "accept" })` for mismatched lengths and an
  empty draft.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_3 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/serving_state_/)'`
- expect: `2 passed` (the two `serving_state_` test names added here)
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc`
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
- do not: change behaviour of `accept` (the snapshot choice `row_caches[accepted]` stays);
  touch `proxima-model-interop`
- stage: `proxima-core/src/serving_state.rs`
- commit: `feat(core): add accept, resume and rollback to the serving state`
- gpu: none

### 1.4 Move the speculation-equals-greedy test with a test-only prompt lookup drafter

- id: FT1.4
- needs: FT1.3
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-model-interop/src/serving_fsm.rs` (`draft_prompt_lookup` ~line 97,
  its test `draft_prompt_lookup_finds_the_earlier_occurrence` ~491-506, `fake_greedy_next` ~line 514,
  `speculation_matches_plain_greedy_for_thirty_two_tokens` ~527-630, doc comment ~518-525, at
  a7c08c4c); `proxima-core/src/serving_state.rs` `mod tests`
- change: the n-gram prompt-lookup drafter is a technique, and the library ships hooks, not
  techniques. It therefore moves as a private function inside `mod tests`, where it is the proof that
  the draft-and-verify hook expresses prompt lookup. Nothing technique-named is added outside
  `mod tests`, and `lib.rs` is not touched.
  0. `serving_state.rs` `mod tests`: copy `draft_prompt_lookup` (~lines 97-112, body verbatim) as the
     private function `fn draft_prompt_lookup<Entry: PartialEq + Clone>(history: &[Entry], ngram: usize, max_k: usize) -> Vec<Entry>`
     (no `pub`; it is called by the speculation test of item 1, so it is not dead code). Copy
     `draft_prompt_lookup_finds_the_earlier_occurrence` (doc comment ~lines 491-497, body to ~506)
     verbatim under the name `serving_state_draft_prompt_lookup_finds_the_earlier_occurrence`.
  1. `serving_state.rs` `mod tests`: copy `fake_greedy_next` (~lines 507-516, with its doc comment)
     and the test (~lines 518-630, with its doc comment) as
     `serving_state_speculation_matches_plain_greedy_for_thirty_two_tokens`. Body verbatim, calling
     the test-only `draft_prompt_lookup`. In the
     test's doc comment make two edits. Line ~518 "The oracle this step asks for, at the FSM
     level: plain greedy decode" becomes "The oracle at the FSM level: plain greedy decode" (the
     phrase "this step asks for" points into a plan and is not committed). The last two lines
     (~524-525, the ones that name one specific model program and a bug in it, starting
     "independent of the real") become the single line "independent of any model program."
- test: `serving_state_draft_prompt_lookup_finds_the_earlier_occurrence` asserts the five
  `draft_prompt_lookup` results of the copied body: `[9,1,2,3]`, `[9,1]`, empty, empty, `[9]`;
  the copied speculation test asserts `spec_history == plain_history` over 32 tokens with `K = 5`,
  `NGRAM = 2`, `VOCAB = 11`, and that `verify_rounds > 0 && draft_tokens_accepted > 0`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_4 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/serving_state_/)'`
- expect: `4 passed`; the four names are exactly:
  `serving_state_walkthrough_drives_every_legal_transition`,
  `serving_state_accept_rejects_mismatched_or_empty_draft`,
  `serving_state_draft_prompt_lookup_finds_the_earlier_occurrence`,
  `serving_state_speculation_matches_plain_greedy_for_thirty_two_tokens`
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc`
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
- do not: touch `proxima-model-interop`; edit `lib.rs`; make `draft_prompt_lookup` `pub` or move it
  outside `mod tests`
- stage: `proxima-core/src/serving_state.rs`
- commit: `test(core): check speculation against plain greedy decoding`
- gpu: none

### 1.5 Switch interop's one consumer to proxima_core::serving_state

- id: FT1.5
- needs: FT1.4
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/serving_backend.rs` (the test module's
    `use crate::serving_fsm::ServingState;` at ~line 121 at a7c08c4c, and the test
    `serving_backend_drives_serving_state_through_prefill_and_decode` ~line 182);
  - `proxima-model-interop/Cargo.toml` (`std = [...]` ~line 37 includes `interop-bgpool`, which is
    `["dep:prime", "dep:proxima-core"]` ~line 58; `proxima-core` ~line 317 with `features = ["alloc"]`).
- change:
  1. `serving_backend.rs`: replace the single line `use crate::serving_fsm::ServingState;` with
     `use proxima_core::serving_state::ServingState;`.
  1b. `serving_backend.rs`: this card touches the file, so it rewrites, in plain English, the four
     comments that point into a plan. Locate each by its text; re-wrap the comment lines after the edit.
     - module doc, `//!`: replace `(a separate, later step: the live loop's` with `(the live loop's`;
     - module doc, `//!`: replace `which is exactly the live-loop migration this step does not attempt.`
       with `which is exactly the live-loop migration this module does not attempt.`;
     - the `//` comment above `#![allow(dead_code)]`: replace all of it with the single line
       `// not yet called from decode.rs's live loop; this module's own tests below are its only caller.`
       (the dropped clause named another file as also carrying the allow, which stops being true
       when that file is deleted next);
     - the doc on `MetalPlacementResources`: replace `out of scope for this step (`serving_fsm.rs`'s
       own doc on why that oracle is separately blocked).` with `out of scope for this module.`.
     No code line other than the import changes in this card.
  2. No `Cargo.toml` edit: `dep:proxima-core` is already reachable under interop's `std` feature
     through `interop-bgpool`; the card's `cargo check` below proves it.
- test: the existing `serving_backend_drives_serving_state_through_prefill_and_decode` (1 test),
  now compiled against `proxima_core::serving_state::ServingState` (its Entry infers to `u32`).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_5 cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_backend_/)'`
- expect: `1 passed` (`serving_backend_drives_serving_state_through_prefill_and_decode`)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`;
  `cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_fsm/)'` still prints
  `4 passed` (the old module stays until card 1.6)
- done when: expect printed, clippy clean, `git diff --cached --stat` equals the stage list
- do not: delete `serving_fsm.rs`; edit `lib.rs`; change any code line of `serving_backend.rs` except
  the import
- stage: `proxima-model-interop/src/generate/serving_backend.rs`
- commit: `refactor(interop): drive the serving backend test with the core state`
- gpu: none

### 1.6 Delete serving_fsm, re-export from proxima_core

- id: FT1.6
- needs: FT1.5
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/lib.rs` (`mod serving_fsm;` with its `#[cfg(feature = "std")]` at
    ~lines 71-72; the `pub use serving::{...};` block ~lines 149-153, the insertion point);
  - `proxima-model-interop/src/generate/serving_backend.rs` (doc and comment mentions of
    `serving_fsm` at ~lines 1, 5, 36, 40, 63, 72, 121, 127, 177 at a7c08c4c, before card 1.5 rewrites
    the comments at ~56 and ~109).
- change:
  1. Delete `proxima-model-interop/src/serving_fsm.rs` (`git rm`).
  2. `lib.rs`: delete the two lines `#[cfg(feature = "std")]` and `mod serving_fsm;`. Add, after the
     `pub use serving::{...};` block, as one gated pair of lines:
     `#[cfg(feature = "std")]` then
     `pub use proxima_core::ServingState;`
     (one physical line; the error item stays reachable as `proxima_core::serving_state::...`,
     so no unused re-export is added). The token path is the instantiation `Entry = u32`, which every
     existing call site infers; no alias type is added.
  3. `serving_backend.rs`: change the test module import `use proxima_core::serving_state::ServingState;`
     to `use crate::ServingState;`, so the test exercises the re-export.
  4. `serving_backend.rs`: in comments only, replace `crate::serving_fsm` with
     `proxima_core::serving_state`, `serving_fsm.rs` with `proxima-core/src/serving_state.rs`,
     `serving_fsm::` with `serving_state::`, the bare name `serving_fsm` (line ~36) with
     `serving_state`, and `ServingState<Cache>` with `ServingState<Entry, Cache>`. One further
     rewrite, because the module doc of the new file no longer carries the quoted words: in the
     module doc paragraph on `ServingBackend`, replace
     ``the module doc on `serving_fsm` names ("evaluate program at positions, advance the logical cache")``
     with ``what every evaluating transition of `serving_state::ServingState` performs (evaluate the
     program at positions, advance the logical cache)``, and re-wrap. Afterwards
     `git grep -n 'serving_fsm' -- proxima-model-interop/src` prints nothing. No code line changes.
- test: `serving_backend_drives_serving_state_through_prefill_and_decode`, compiled against the
  re-export path.
- validate: `echo "reexport=$(git grep -nP 'pub use proxima_core::.*\bServingState\b' -- proxima-model-interop/src/lib.rs | wc -l) oldmod=$(git grep -nP '^\s*(pub\(crate\) )?mod serving_fsm\b' -- proxima-model-interop/src/lib.rs | wc -l)" && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_6 cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_backend_/)'`
- expect: `reexport=1 oldmod=0`, then `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets`;
  `out=$(RUSTDOCFLAGS='-D warnings' cargo doc -p proxima-model-interop --features std --no-deps 2>&1); echo "finished=$(echo "$out" | grep -c '^ *Finished') diagnostics=$(echo "$out" | grep -c -E '^(warning|error)')"`
  prints `finished=1 diagnostics=0`
- done when: expect printed, clippy clean, the doc line printed its counts, `git diff --cached --stat` equals the stage list
- do not: add an `Entry = u32` alias type; edit `proxima-core`
- stage: `proxima-model-interop/src/lib.rs`, `proxima-model-interop/src/generate/serving_backend.rs`, `proxima-model-interop/src/serving_fsm.rs` (deleted)
- commit: `refactor(interop): replace the serving fsm module with a core re-export`
- gpu: none

### 1.7 Add accept_rows (the policy-chosen row count)

- id: FT1.7
- needs: FT1.4
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-core/src/serving_state.rs::accept` (from card 1.3)
- change: why the new method exists: `accept` hardcodes "leading equal rows", so a rule that
  accepts fewer rows than that prefix (a minimum run, a row cap) or judges by a similarity cannot
  be expressed; the existing primitive (`accept`) cannot carry it. The two call sites differ in
  what a caller can do: `accept` cannot return a smaller count than the equal prefix,
  `accept_rows(count, ..)` can. The method is the row-index accept the frame table names.
  1. `serving_state.rs`, in `impl<Entry: PartialEq + Clone, Cache> ServingState<Entry, Cache>`:
     - private `fn settle(draft: Vec<Entry>, accepted: usize, row_tokens: &[Entry], row_caches: Vec<Cache>, attempted: &'static str) -> Result<Self, ServingFsmError>`:
       returns `Err(IllegalTransition { attempted })` when `draft.is_empty()`, or
       `row_tokens.len() != draft.len()`, or `row_caches.len() != draft.len()`, or
       `accepted > draft.len()`; otherwise the body of today's `accept` second arm with
       `accepted` given instead of counted (`placement_index = accepted - 1` when
       `accepted == draft.len()`, else `accepted`; `Accept { n: accepted, next: draft[accepted - 1].clone(), cache }`
       or `Rollback { snapshot: cache, to: row_tokens[accepted].clone() }`);
     - `pub fn accept_rows(self, accepted: usize, row_tokens: &[Entry], row_caches: Vec<Cache>) -> Result<Self, ServingFsmError>`:
       `Self::Verify { draft, .. } => Self::settle(draft, accepted, row_tokens, row_caches, "accept_rows")`,
       any other variant `Err(IllegalTransition { attempted: "accept_rows" })`;
     - `accept` becomes: `Self::Verify { draft, .. } => { let accepted = draft.iter().zip(row_tokens.iter()).take_while(|(drafted, predicted)| drafted == predicted).count(); Self::settle(draft, accepted, row_tokens, row_caches, "accept") }`,
       other variants `Err(IllegalTransition { attempted: "accept" })`.
- test: add `accept_rows_takes_declared_count_not_prefix` in `proxima-core/src/serving_state.rs`
  `mod tests` (name deliberately without `serving_state_`, so the four-name count stays 4). State:
  `start(alloc::vec![1_u32], FakeCache::empty())` -> `advance_prefill(2, FakeCache::empty().advanced_by(2))`
  -> `enter_verify(alloc::vec![3, 4])` (each step `.expect`ed; every vec literal is `alloc::vec!`, the
  macro is not in scope under `no_std`); bind the result as `state`.
  `FakeCache` is not `Copy` (it derives `Debug, Clone, PartialEq` only), so every cache is cloned
  where it is used twice. Bind `cache_one = FakeCache::empty().advanced_by(1)` and
  `cache_two = FakeCache::empty().advanced_by(2)`. Each assertion runs on `state.clone()` and passes
  `alloc::vec![cache_one.clone(), cache_two.clone()]` as the row caches, with `row_tokens = &[3, 4]`
  (equal to the draft). Assert:
  - `state.clone().accept_rows(0, &[3, 4], alloc::vec![cache_one.clone(), cache_two.clone()])` == `Ok(ServingState::Rollback { snapshot: cache_one.clone(), to: 3 })`;
  - `state.clone().accept_rows(2, &[3, 4], alloc::vec![cache_one.clone(), cache_two.clone()])` == `Ok(ServingState::Accept { n: 2, next: 4, cache: cache_two.clone() })`;
  - `state.clone().accept_rows(3, &[3, 4], alloc::vec![cache_one.clone(), cache_two.clone()])` == `Err(ServingFsmError::IllegalTransition { attempted: "accept_rows" })`;
  - `ServingState::start(alloc::vec![1_u32], FakeCache::empty()).accept_rows(0, &[3, 4], alloc::vec![cache_one.clone(), cache_two.clone()])`
    (a `Prefill` state, not `Verify`) == `Err(ServingFsmError::IllegalTransition { attempted: "accept_rows" })`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_7 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/serving_state_|accept_rows_/)'`
- expect: `5 passed` (the 4 `serving_state_` names plus `accept_rows_takes_declared_count_not_prefix`)
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc`
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
- do not: change the result of any existing `accept` test
- stage: `proxima-core/src/serving_state.rs`
- commit: `feat(core): let the serving state accept a caller-chosen row count`
- gpu: none

### 1.8 Add the action fixture (test-only)

- id: FT1.8
- needs: FT1.7, FT1.1
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-tensor/specs/fsm-techniques/worked-examples.md` section "action speculation" (the values this card encodes);
  `proxima-core/src/serving_state.rs` (types, `accept_rows`)
- change:
  1. `serving_state.rs`: add, below the non-test items, the three lines
     `#[cfg(test)]`, `#[allow(clippy::unwrap_used, clippy::expect_used)]`, `pub(crate) mod action_fixture;`
     with this comment above them: `// test-only environment; a failed transition here is a broken fixture, so expect names it`.
     The allowance is needed because the fixture is its own module, outside `mod tests`.
     No new public type: the accept rule is a pure fn over `(draft, choices)` slices of `Entry`,
     where `choices` are the verifier's per-row entries (`row_tokens` in `accept_rows`).
  2. `proxima-core/src/serving_state/action_fixture.rs` (new, `cfg(test)` by its declaration;
     `use alloc::collections::BTreeMap; use alloc::vec::Vec;`):
     - `#[derive(Debug, Clone, PartialEq)] pub(crate) struct Action { pub key: u8, pub delta: i64, pub version: u64 }`;
     - `#[derive(Debug, Clone, PartialEq, Default)] pub(crate) struct Environment { pub values: BTreeMap<u8, i64>, pub version: u64 }`
       with `pub(crate) fn apply(&self, action: &Action) -> Self` (clone; `*values.entry(action.key).or_insert(0) += action.delta; version += 1`);
     - `pub(crate) const TARGET_STEPS: usize = 12;`
     - `pub(crate) fn true_action(environment: &Environment) -> Action`:
       `key = (version % 3) as u8`, `delta = values.get(&key).copied().unwrap_or(0).rem_euclid(5) + 1`, `version = environment.version`;
     - `pub(crate) struct Trace { pub history: Vec<Action>, pub environment: Environment }`
       (test-only record for comparing two runs; derives `Debug`);
     - `pub(crate) fn run_sequential() -> Trace`: 12 times `true_action` then `apply`;
  3. `serving_state.rs` `mod tests`: add `action_fixture_sequential_matches_hand_derivation`.
- test: `action_fixture_sequential_matches_hand_derivation` asserts `run_sequential().history` equals
  the 12 actions of the action speculation section, in `(key, delta, version)`: (0,1,0) (1,1,1) (2,1,2) (0,2,3)
  (1,2,4) (2,2,5) (0,4,6) (1,4,7) (2,4,8) (0,3,9) (1,3,10) (2,3,11), and the final environment has
  `values == {0:10, 1:10, 2:10}` and `version == 12`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_8 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/action_fixture_|serving_state_|accept_rows_/)'`
- expect: `6 passed` (4 `serving_state_` names, `accept_rows_takes_declared_count_not_prefix`, `action_fixture_sequential_matches_hand_derivation`)
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc` (the fixture is `cfg(test)`, so
  the check build is unchanged)
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
- do not: name any test `action_speculation_` here; add Cargo dependencies
- stage: `proxima-core/src/serving_state.rs`, `proxima-core/src/serving_state/action_fixture.rs`
- commit: `test(core): add a versioned action environment fixture`
- gpu: none

### 1.9 Add the three action_speculation_ tests

- id: FT1.9
- needs: FT1.8
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first: `proxima-tensor/specs/fsm-techniques/worked-examples.md` the action speculation section (round counts); `proxima-core/src/serving_state/action_fixture.rs`
  (`run_sequential`, `Trace`, from card 1.8)
- change:
  1. `proxima-core/src/serving_state/action_fixture.rs`: add `use super::ServingState;`, the constant
     `pub(crate) const DRAFT_ROWS: usize = 4;`, extend `Trace` with
     `pub rounds: usize, pub rejected_rounds: usize, pub accepted_rows: usize` (`run_sequential` sets
     `rounds = 12` and the others 0), and add:
     - `pub(crate) fn guessed_action(environment: &Environment) -> Action`: `true_action` with
       `delta + 1` when `environment.version % 4 == 3`;
     - `pub(crate) fn draft_from(base: &Environment, rows: usize) -> Vec<Action>`: a cursor clone;
       `rows` times push `guessed_action(&cursor)` and `cursor = cursor.apply(&that)`;
     - `fn verify_rows(environment: &Environment, draft: &[Action]) -> (Vec<Action>, Vec<Environment>)`:
       cursor = environment clone; per drafted action: `truth = true_action(&cursor)`, push
       `cursor.apply(&truth)` to the caches, push `truth` to observed, then `cursor = cursor.apply(drafted)`;
     - `pub(crate) fn run_speculative(accept_policy: impl Fn(&[Action], &[Action]) -> usize) -> Trace`:
       a `genesis = Action { key: 0, delta: 0, version: 0 }` no-op marker;
       `state = ServingState::start(Vec::new(), Environment::default()).advance_prefill(genesis, Environment::default())`;
       `previous = Environment::default()`; loop while `history.len() < TARGET_STEPS`:
       `environment` = the `Decode` state's cache clone; `rows = (TARGET_STEPS - history.len()).min(DRAFT_ROWS)`;
       `base = if rounds % 3 == 2 { &previous } else { &environment }`; `draft = draft_from(base, rows)`;
       `(observed, row_caches) = verify_rows(&environment, &draft)`;
       `accepted = accept_policy(&draft, &observed)`; `settled = state.enter_verify(draft.clone())` then
       `.accept_rows(accepted, &observed, row_caches)`; on `Accept { .. }`: extend history with all of
       `draft`, `accepted_rows += accepted`, `state = settled.resume()`; on `Rollback { .. }`: extend
       history with `draft[..accepted]` then push `observed[accepted].clone()`, `accepted_rows += accepted`,
       `rejected_rounds += 1`, `state = settled.rollback()`; any other variant panics with a message.
       After the round: `previous = environment`, `rounds += 1`. Return a `Trace` whose `environment`
       is the final `Decode` cache. Every `Result` is unwrapped with `expect` and a message.
  2. `serving_state.rs` `mod tests`: add three policy fns over `(draft: &[Action], choices: &[Action])` and three tests.
     A helper `fn leading_equal(draft: &[Action], choices: &[Action]) -> usize` is
     `draft.iter().zip(choices).take_while(|(drafted, chosen)| drafted == chosen).count()`.
     - `fn equality_policy(draft, choices) -> usize`: `leading_equal(draft, choices)`;
     - `fn verifier_exact_policy(draft, choices) -> usize`: `leading_equal(draft, choices)` (the verifier's
       choice is the true entry; exact rollback keeps only the equal prefix);
     - `fn anchor_macro_policy(draft, choices) -> usize`: `aligned = leading_equal(draft, choices)`;
       returns `aligned` iff `aligned >= 3`, else `0`.
  3. Tests, each `let sequential = run_sequential(); let speculative = run_speculative(<policy>);`
     then asserts `speculative.history == sequential.history`,
     `speculative.environment == sequential.environment`, and
     `(rounds, rejected_rounds, accepted_rows)`:
     - `action_speculation_equality_matches_sequential`: `(4, 4, 8)`;
     - `action_speculation_verifier_exact_matches_sequential`: `(4, 4, 8)`;
     - `action_speculation_anchor_macro_matches_sequential`: `(6, 6, 6)`.
- test: the three tests above (values from the action speculation worked example).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_9 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/action_speculation_/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc`
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
- do not: change worked values; widen a count to make a test pass (a mismatch is a false premise: stop and report)
- stage: `proxima-core/src/serving_state.rs`, `proxima-core/src/serving_state/action_fixture.rs`
- commit: `test(core): check action speculation policies against sequential runs`
- gpu: none

### 1.10 Add the accept rule and its pure function

- id: FT1.10
- needs: FT1.7
- budget: 20 min
- crate(s): proxima-core (features: alloc)
- read first:
  - `proxima-core/src/serving_state.rs::accept_rows` and `::accept` (from cards 1.3 and 1.7: the
    count this rule produces is the `accepted` argument of `accept_rows`);
  - `proxima-core/src/lib.rs` (`#[cfg(feature = "alloc")] pub mod serving_state;` from card 1.2, the
    insertion point);
  - `proxima-model-interop/src/speculative_settings.rs::SpeculativeSettings` (~line 253): the token
    n-gram section is named `speculative`; this card's section is named `accept_rule` so the two do
    not clash.
- change: why the type exists. The accept decision must be loadable data: a closure passed to
  `accept_rows` cannot be written in TOML or env and cannot enter a cache key, `AcceptRule` can.
  The call sites differ in what a caller can do: `state.accept_rows(some_closure(..), ..)` is Rust
  only, `state.accept_rows(rule.accepted_rows(..), ..)` with `rule` read from configuration is not.
  Not a pipe: `proxima_primitives::Pipe` lives in a crate above `proxima-core` (`proxima-primitives`
  depends on `proxima-core`), so the decision is the pure method and a pipe adaptor over it belongs
  to a consumer. The type names no technique: equality, a verifier's similarity, and a minimum
  aligned run are three settings of the same three values.
  1. `proxima-core/src/accept_rule.rs` (new). Module doc, short: names `ServingState::accept_rows`
     as the consumer of the count and `AcceptRule::accepted_rows` as the pure decision; one sentence
     that a similarity floor of 1.0 under exact equality, minimum run 0 and no cap reproduce
     `ServingState::accept`. Contents:
     - `#[derive(Debug, Clone, Copy, PartialEq)] pub struct AcceptRule` with three public fields, each
       with a one-line doc comment giving the meaning and one real value:
       `similarity_floor: f32` (a drafted row counts as accepted when its similarity to the verifier's
       row is at least this; 1.0 admits only rows the caller's similarity scores 1.0),
       `min_run: u16` (a leading accepted run shorter than this is discarded, so the count is 0; 3 for
       a rule that only commits runs of three), `max_rows: u16` (the leading run is cut at this many
       rows; 8);
     - `impl Default for AcceptRule` returning `similarity_floor: 1.0, min_run: 0, max_rows: u16::MAX`;
     - `impl AcceptRule { #[must_use] pub fn accepted_rows<Entry>(&self, draft: &[Entry], choices: &[Entry], similarity: impl Fn(&Entry, &Entry) -> f32) -> usize }`:
       `run = draft.iter().zip(choices).take(usize::from(self.max_rows)).take_while(|(drafted, chosen)| similarity(drafted, chosen) >= self.similarity_floor).count()`;
       returns `0` when `run < usize::from(self.min_run)`, else `run`. The cap applies before the
       minimum-run check. Doc comment on the method names `ServingState::accept_rows` as the caller
       and says why the similarity is an argument: the entry type is generic, so only the caller can
       score two entries.
  2. `proxima-core/src/lib.rs`: after the `serving_state` pair add
     `#[cfg(feature = "alloc")] pub mod accept_rule;` (two lines).
  3. Test module in `accept_rule.rs`: `#[cfg(test)]`, the comment
     `// a failed transition in a test is a broken test; expect names it`,
     `#[allow(clippy::unwrap_used, clippy::expect_used)]`, `mod tests { use super::*; use crate::serving_state::ServingState; ... }`.
     A helper `fn exact(drafted: &u32, chosen: &u32) -> f32 { if drafted == chosen { 1.0 } else { 0.0 } }`.
- test: four tests in `accept_rule.rs` `mod tests`:
  - `accept_rule_default_matches_leading_equal_accept`: for each of three cases (draft, choices) =
    `([3,4,5],[3,4,5])`, `([3,4,5],[3,9,5])`, `([3,4,5],[8,4,5])` with state
    `ServingState::start(alloc::vec![1_u32], 0_usize)` -> `advance_prefill(2, 1)` -> `enter_verify(draft.to_vec())` (each
    `.expect`ed; every vec literal is `alloc::vec!`, the macro is not in scope under `no_std`) and caches `alloc::vec![10, 11, 12]`: assert
    `state.clone().accept(&choices, caches.clone()) == state.accept_rows(AcceptRule::default().accepted_rows(&draft, &choices, exact), &choices, caches)`,
    and that the three results equal `Ok(Accept { n: 3, next: 5, cache: 12 })`,
    `Ok(Rollback { snapshot: 11, to: 9 })` and `Ok(Rollback { snapshot: 10, to: 8 })` in order;
  - `accept_rule_similarity_floor_admits_near_matches`: entries `u8`, draft `[10, 20, 30]`, choices
    `[10, 21, 35]`, similarity `1.0 - f32::from(a.abs_diff(*b)) / 10.0` (rows score 1.0, 0.9, 0.5);
    floor 1.0 gives 1, floor 0.85 gives 2, floor 0.4 gives 3 (`min_run` 0, `max_rows` `u16::MAX`);
  - `accept_rule_row_cap_limits_the_accepted_run`: draft equal to choices `[1,2,3,4,5]`, `exact`
    over `u32`: `max_rows` 3 gives 3, `max_rows` 9 gives 5, `max_rows` 0 gives 0;
  - `accept_rule_minimum_run_refuses_short_runs`: draft `[1,2,3,9]`, choices `[1,2,3,4]`: `min_run` 0
    gives 3, `min_run` 3 gives 3, `min_run` 4 gives 0; and `min_run` 3 with `max_rows` 2 gives 0 (the
    cap applies first).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_10 cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/accept_rule_/)'`
- expect: `4 passed` (the four `accept_rule_` names above)
- also green: `cargo clippy -p proxima-core --no-default-features --features alloc --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc`
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
- do not: add a field or variant that names a technique; add a `Pipe` impl; touch `ServingState`;
  add Cargo dependencies or features
- stage: `proxima-core/src/accept_rule.rs`, `proxima-core/src/lib.rs`
- commit: `feat(core): add a data-driven accept rule for speculation rows`
- gpu: none

### 1.11 Load the accept rule from builder, TOML and env

- id: FT1.11
- needs: FT1.10
- budget: 20 min
- crate(s): proxima-core (features: config)
- read first:
  - `proxima-model-interop/src/speculative_settings.rs` (struct and attributes ~lines 253-338,
    including `#[setting(default = 0.0)]` on an `f32` and `#[setting(default = 3)]` on an integer;
    the builder, TOML and env round-trip test ~lines 398-457: the pattern to follow, with
    `temp_env::with_vars` and `Settings::from_env`);
  - `proxima-core/src/accept_rule.rs` (from card 1.10);
  - `proxima-core/Cargo.toml` (`config = [...]` ~line 65, which enables `bon`, `conflaguration` with
    `derive` and `toml`, and `serde`; `[dev-dependencies]`).
- change: the rule is one type, so the config surface is derives on it, gated by `config`; no
  mirror struct. The derive `cfg_attr` line must come before the helper attribute lines (a helper
  attribute placed before its derive is rejected under the workspace's deny-warnings).
  0. From the repo root run `cargo add temp-env --dev -p proxima-core` (cargo add; `temp-env` is in
     `[workspace.dependencies]`, so cargo records it as a workspace member; never edit the dependency
     line by hand). The same card is its first use.
  1. `accept_rule.rs`: on `AcceptRule`, after the existing `#[derive(Debug, Clone, Copy, PartialEq)]`,
     add in this order:
     `#[cfg_attr(feature = "config", derive(bon::Builder, serde::Deserialize, serde::Serialize, conflaguration::Settings, conflaguration::Validate))]`,
     `#[cfg_attr(feature = "config", settings(prefix = "PROXIMA_ACCEPT_RULE"))]`,
     `#[cfg_attr(feature = "config", builder(derive(Clone, Debug)))]`.
     On each field add a pair of `cfg_attr(feature = "config", ...)` lines carrying `setting(default = V)`
     and `builder(default = V)` with the same value `V`:
     - `similarity_floor`: `1.0`;
     - `min_run`: `0`;
     - `max_rows`: `65535` in `setting`, `u16::MAX` in `builder`.
     The env names the loader reads are `PROXIMA_ACCEPT_RULE_SIMILARITY_FLOOR`,
     `PROXIMA_ACCEPT_RULE_MIN_RUN`, `PROXIMA_ACCEPT_RULE_MAX_ROWS`. Update the module doc by one
     sentence: under `config` the rule loads from a builder, a TOML table and env.
  2. Test module `#[cfg(all(test, feature = "config"))] mod config_tests` in `accept_rule.rs`, with
     the same lint allowance and comment as `mod tests`, `use super::*; use conflaguration::Settings;`
     and a helper
     `fn assert_loaders_agree(expected: &AcceptRule, floor: &str, run: &str, rows: &str)`:
     builds the TOML `similarity_floor = {floor}\nmin_run = {run}\nmax_rows = {rows}\n`, loads it with
     `conflaguration::from_toml_str::<AcceptRule>`, asserts equal to `expected`; then
     `temp_env::with_vars` the three `PROXIMA_ACCEPT_RULE_*` vars to the same strings and asserts
     `AcceptRule::from_env()` equals `expected`.
- test: four tests in `config_tests`:
  - `accept_rule_builder_defaults_match_default_impl`: `AcceptRule::builder().build() == AcceptRule::default()`,
    and with the three env vars unset (`None::<&str>` in `with_vars`) `AcceptRule::from_env()` equals
    `AcceptRule::default()`;
  - `accept_rule_round_trip_similarity_floor`: `AcceptRule::builder().similarity_floor(0.75).build()`
    through `assert_loaders_agree(&rule, "0.75", "0", "65535")`;
  - `accept_rule_round_trip_run_and_cap`: `AcceptRule::builder().min_run(3).max_rows(6).build()`
    through `assert_loaders_agree(&rule, "1.0", "3", "6")`;
  - `accept_rule_rejects_malformed_values`: `from_toml_str::<AcceptRule>` is `Err` for
    `similarity_floor = "high"` (with valid other keys) and for `max_rows = 70000` (outside `u16`);
    `AcceptRule::from_env()` is `Err` with `PROXIMA_ACCEPT_RULE_MAX_ROWS=lots`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_11 cargo nextest run -p proxima-core --features config -E 'test(/accept_rule_/)'`
- expect: `8 passed` (the four from card 1.10 plus the four above)
- also green: `cargo clippy -p proxima-core --features config --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc` (the derives are `config`-gated,
  so the alloc build is unchanged)
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
  (`Cargo.toml` and `Cargo.lock` as the dependency tool rewrites them)
- do not: add fields beyond the three; add a settings mirror struct; add a `Pipe` impl; add
  dependencies other than `temp-env`; edit dependency lines by hand
- stage: `proxima-core/src/accept_rule.rs`, `proxima-core/Cargo.toml`, `Cargo.lock`
- commit: `feat(core): load the accept rule from toml, env and builder`
- gpu: none

### 1.12 Drive the three speculation policies through configured accept rules

- id: FT1.12
- needs: FT1.11, FT1.9, FT1.1
- budget: 20 min
- crate(s): proxima-core (features: config)
- read first:
  - `proxima-core/src/serving_state/action_fixture.rs` (`run_sequential`, `run_speculative`,
    `Action`, from cards 1.8 and 1.9);
  - `proxima-tensor/specs/fsm-techniques/worked-examples.md` the action speculation section (the rule values and the counts
    of each policy);
  - `proxima-core/src/accept_rule.rs` (`AcceptRule::accepted_rows`, `config_tests`).
- change: this card is the proof that the hook expresses three techniques, and nothing else. No
  library code changes: the techniques exist only as three TOML tables in tests, and the driving
  code is one closure per test. The call site against the fixture, in each test, is
  `run_speculative(|draft, choices| rule.accepted_rows(draft, choices, field_agreement))`.
  1. `accept_rule.rs` `config_tests`: add `use crate::serving_state::action_fixture::{Action, run_sequential, run_speculative};`
     and a helper
     `fn field_agreement(drafted: &Action, chosen: &Action) -> f32`:
     `f32::from(u8::from(drafted.key == chosen.key) + u8::from(drafted.delta == chosen.delta) + u8::from(drafted.version == chosen.version)) / 3.0`
     (1.0 only when all three fields agree), and a helper
     `fn rule_from(toml: &str) -> AcceptRule` (`conflaguration::from_toml_str`, `.expect("the rule
     table parses")`).
  2. Three tests, each `let rule = rule_from(<table>); let sequential = run_sequential(); let speculative = run_speculative(<closure above>);`
     then asserts `speculative.history == sequential.history`,
     `speculative.environment == sequential.environment`, and
     `(speculative.rounds, speculative.rejected_rounds, speculative.accepted_rows)`:
     - `configured_rule_equality_matches_sequential`: table
       `similarity_floor = 1.0\nmin_run = 0\nmax_rows = 8\n`; `(4, 4, 8)`;
     - `configured_rule_verifier_exact_matches_sequential`: table
       `similarity_floor = 1.0\nmin_run = 0\nmax_rows = 65535\n`; `(4, 4, 8)`;
     - `configured_rule_anchor_run_matches_sequential`: table
       `similarity_floor = 1.0\nmin_run = 3\nmax_rows = 65535\n`; `(6, 6, 6)`.
- test: the three tests above; the counts are the worked example's RESULT line (equality 4/4/8,
  verifier-exact 4/4/8, anchor+macro 6/6/6).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_1_12 cargo nextest run -p proxima-core --features config -E 'test(/accept_rule_|configured_rule_/)'`
- expect: `11 passed` (the 8 `accept_rule_` tests plus the 3 `configured_rule_` tests)
- also green: `cargo clippy -p proxima-core --features config --all-targets`;
  `cargo check -p proxima-core --no-default-features --features alloc`
- done when: expect printed, clippy and check clean, `git diff --cached --stat` equals the stage list
- do not: change the fixture, the rule or the worked counts to make a test pass (a mismatch is a
  false premise: stop and report); add a technique-named item to the library; name a test
  `action_speculation_` or `accept_rule_`
- stage: `proxima-core/src/accept_rule.rs`
- commit: `test(core): drive speculation through configured accept rules`
- gpu: none

## spec drift

1. Feature wiring: the card to "add `dep:proxima-core` to `std`" is not needed.
   `proxima-model-interop/Cargo.toml` `std` already lists `interop-bgpool`, and
   `interop-bgpool = ["dep:prime", "dep:proxima-core"]`. Card 1.5 proves reachability with the
   compile; no Cargo edit.
2. `ServingFsmError::NotSupported` is still on main (`serving_fsm.rs` ~line 87, defined and never
   constructed). Card 1.13 omits it when copying, so the move does not depend on deleting it first.
3. No `RowReadout` type. The verifier's per-row choices are `&[Entry]` beside the draft, and the
   accept rule is a pure fn over (draft, choices, similarity, rule values). `ServingState::accept_rows`
   (card 1.7) takes the count; the rule is `AcceptRule::accepted_rows` (card 1.10).
4. `Pipe` lives in `proxima-primitives`, which depends on `proxima-core`. The accept rule is
   therefore a pure method here; a `Pipe` form is a consumer's adaptor over it.
5. The earlier accept grammar (a closed enum of three technique names, a rollback kind, nine
   technique-named fields, a `speculation` section) is replaced by three scalars under an
   `accept_rule` section. The old `fields` selector is gone because a generic `Entry` has no field
   projection; the similarity function is the argument that carries any field choice. The old
   "tau" is the similarity floor, the old "min_skip" is the minimum run, and the old fork budget and
   macro depth both become the row cap. Forking is a separate hook and is not built here.
6. The action worked run gets its staleness from the stale-snapshot rounds (`rounds % 3 == 2`),
   which make both action and version differ. A run in which the action matches and only the
   version differs is not exercised.
7. The token path is the instantiation `Entry = u32`; no alias type is added (card 1.6).
8. `draft_prompt_lookup` is a technique, so it is never public: card 1.4 keeps it as a private
   function inside `serving_state.rs` `mod tests`, used by the speculation test as the proof that
   the hook expresses prompt lookup. Any card elsewhere that cites FT1.2 for the lookup means
   FT1.4, and none may call it from outside that test module.
9. Sketch 14 argued that interning actions to ids avoids a generic entry. The generic entry is kept
   because the owner keep rule names it; FT1.8-FT1.9 and FT1.12 run it with a non-token entry
   (`Action`) and no interning table.
10. Retired: the spec's `SpeculationSettings` requirement (its `accept` grammar and its lowering to
    the accept pipe) and the acceptance check that counts six `speculation_settings_` tests. This
    slice replaces them with `AcceptRule` (FT1.10, FT1.11); the lowered-policy runs are the three
    `configured_rule_` tests of FT1.12, and the slice exit rows 5 and 6 are the commands that
    replace the old check. No card of this file cites the retired spec entries, and the spec text
    names a type no card builds; it is rewritten to name `AcceptRule` when it lands.

## slice exit

Run after card 1.12, from the checkout holding main. Row 7 needs `env.sh`, which card FT0.4
(`tasks/00-oracles-and-worked-examples.md`) creates and no card of this slice does; FT0.4 must have
landed, then `source proxima-tensor/specs/fsm-techniques/env.sh`:

| order | command | expect |
|---|---|---|
| 1 | `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/serving_state_/)'` | 4 passed |
| 2 | `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/action_speculation_/)'` | 3 passed |
| 3 (a) | `git grep -nP 'pub use proxima_core::.*\bServingState\b' -- proxima-model-interop/src/lib.rs \| wc -l` | 1 |
| 3 (b) | `git grep -nP '^\s*(pub\(crate\) )?mod serving_fsm\b' -- proxima-model-interop/src/lib.rs \| wc -l` | 0 |
| 3 (c) | `cargo nextest run -p proxima-model-interop --features std -E 'test(/serving_backend_/)'` | 1 passed |
| 4 | `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/accept_rule_/)'` | 4 passed |
| 5 | `cargo nextest run -p proxima-core --features config -E 'test(/accept_rule_/)'` | 8 passed |
| 6 | `cargo nextest run -p proxima-core --features config -E 'test(/configured_rule_/)'` | 3 passed |
| 7 | `cargo nextest run -p proxima-model-interop --features std,metal -E 'binary(arch_data_baseline) & test(/^llama_parity_gemma4_e2b$/)'` | 1 passed (gemma4 E2B only, a recorded-oracle replay; one model-loading run, wait for a quiet box first) |
| 8 | `out=$(cargo check -p proxima-core --no-default-features --features alloc 2>&1); echo "finished=$(echo "$out" \| grep -c '^ *Finished') diagnostics=$(echo "$out" \| grep -c -E '^(warning\|error)')"` | finished=1 diagnostics=0 |
| 9 | `cargo nextest run -p proxima-core --no-default-features --features alloc -E 'test(/serving_fsm_error_/)'` | 1 passed |

The four `serving_fsm` tests that slice 0 counted are the four `serving_state_` tests after card 1.6.
