# slice 10: Idle jobs (re-cut; cards FT10.1 - FT10.9)

anchors read at main 2a81f54d (full sha 2a81f54dea3ef5499479477ca6f2bc1d10a66fbf) with `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`. Every `path::symbol (~line N)` below is the line at that sha; the executor re-locates by symbol. `interop` means `proxima-model-interop/src`; `generate` means `proxima-model-interop/src/generate`.

Governing direction (owner, 2026-10-04): the hooks are built so techniques can be vetted later; a technique is not built here. This slice builds the idle-job hook: what the device does while no request is waiting is a configured list of steps (`prewarm`, or `draft` with its own parameters), the default list reproduces today's two-step sequence byte for byte, and two tests prove techniques the library does not define run through the hook (sleep-time derived context, and compaction of stored bytes as a caller-supplied job). Governing spec: `proxima-windows/proxima-tensor/specs/pipeline-as-data/SPEC.md` (idle schedule rows), `sketches/12-sleep-time-prewarm.md` (gaps G1, G2, G3, G5; section 2 says no pure decision function is needed, `jobs.get(completed)` guarded by `request_waiting()` is the whole decision).

Placement result: this slice adds no module to `proxima-core`. The decision "which step runs next, and does a waiting request preempt it" is `steps.get(completed)` guarded by the gate's `request_waiting()` (one atomic load, `generate/prewarm_gate.rs::PrewarmGate::request_waiting`, ~line 77), identical to the body any named function would have; the preemption points (per chunk in `prewarm_ids`, per decoded token in `draft_branch`) already hold on main.

Designs abandoned: a core `IdleKind`/`IdleJob`/`Next` schedule module (same body as `get` plus an atomic load; call sites identical); an `IdleKind::{Sleep, Compact}` job kind (hard-codes techniques; a sleep job is a `draft` element with a lead and a keep flag); a compaction job or file format in the library (compaction appears only as a test job); a pipe wrapping the draft job (`follow_up_branches(..)` against `draft_job.call(entry)` is the same work plus a future around a synchronous loop).

## old to new id map

| old card | new card | verdict |
|---|---|---|
| FT10.1 | none | dropped |
| FT10.2 | none | dropped |
| FT10.3 | FT10.2, FT10.3, FT10.5, FT10.8 | recut: lead tokens, keep marker, draft observer, and the sleep-time proof |
| FT10.4 | FT10.7, FT10.9 | recut: caller-supplied idle job under the gate, and the compaction proof |
| FT10.5 | none | dropped |
| FT10.6 | FT10.1, FT10.4, FT10.6 | recut: draft step parameters, the step list type and default, the stored list driving the queued job |

## dropped

- FT10.1 (pure `schedule::next` and `IdleKind`/`IdleJob`/`Next` in proxima-core): the next job is `steps.get(completed)` behind one atomic load, the same body; the types fail the call-site-both-ways test and `IdleKind::{Sleep, Compact}` hard-codes technique kinds. No serving-decision module exists on main (`git show main:proxima-core/src/serving_state.rs` fails). The list-valued config it carried is FT10.4 and FT10.6.
- FT10.2 (prewarm forwards `request_waiting()` to the core schedule): a pure relocation, behaviour unchanged by its own statement; the preemption points already hold on main (`generate/prewarm.rs::prewarm_ids`, per chunk; `generate/prewarm_follow_up.rs::draft_branch`, per token); it depended on the dropped core module.
- FT10.5 (round preemption to a block boundary): the request-waiting check already runs at chunk and token granularity on main; "block boundary" is not a hook, and at the default chunk of 256 against block of 64 the rounding is a no-op.

## carried, outside this file's verdict list

- Queue policy for many sleep jobs on different contexts (sketch gap G4): the queue replace-on-submit slot (`generate/prewarm_queue.rs::PrewarmQueue::submit`, ~line 94) stays, since an older end-of-answer prefix is obsolete. A caller with several contexts drives them through FT10.7's caller-supplied job entry, which does not queue. No card changes the queue.
- Validation that `branches + 1 <= max_entries` (sketch gap G6) and the sketch's side defects (a silent no-run when the suffix or closing is unregistered; `PrewarmReport` carrying no draft result) are not in the triage verdicts for this file and have no card here. FT10.5 hands the drafts to the caller, which covers the last one.

## prerequisites outside this slice

None. Every card below uses only symbols present at main 2a81f54d. The old order constraint (slice 10 follows slices 1, 2, 4 and 5) no longer applies: no card uses the serving state, the serving config grammar, the seal rule or a block file.

Test models (owner, 2026-10-04): gemma4 for dense or MoE, granite for MoE. No qwen of any kind appears in any card of this file. Cards that load a model use gemma4 E2B (dense) through `with_model`; the idle path runs the same prefill and decode loops whatever the layer kind, and no card here touches expert code, so no MoE arm is added; a card that changes what a row computes owns the MoE arms.

Real-model tests in this crate are `#[ignore]`d because they need the host-local gemma4 E2B blob; every validate line that loads a model passes `--run-ignored all` with a name filter so only the named tests run, and the expected count includes them. `PROXIMA_GEMMA4_E2B_GGUF` names the blob (`ollama show --modelfile gemma4:e2b-it-qat`).

Test-name rule for this slice: new test names say what they check in English; none contains a card, slice, requirement or worked-example id.

---

### 10.1 The follow-up draft job reads its parameters from a draft step

- id: FT10.1
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prewarm_follow_up.rs::LoadedModel::follow_up_branches (~line 78 at 2a81f54d)`: reads `config.follow_up_branches` at the zero guard (~line 94) and in the loop bound (~line 124), and `config.follow_up_branches` again in the closing debug event (~line 172);
  - `interop/generate/prewarm_follow_up.rs::LoadedModel::draft_branch (~line 183)`: reads `config.follow_up_temperature_milli` (~line 197) and `config.follow_up_max_tokens` (~line 205);
  - `interop/generate/prewarm.rs::LoadedModel::prewarm_queued (~line 340)`: the one non-public caller of `follow_up_branches`;
  - `interop/serving.rs::PromptCacheConfig (~line 471)`: the three `follow_up_*` fields (~lines 530 to 540) and their defaults 0, 48, 800 in `standard` (~line 573).
- change:
  1. `interop/generate/prewarm_follow_up.rs`: add, above the `impl LoadedModel` block:
     ```rust
     #[derive(Debug, Clone, Copy, PartialEq, Eq)]
     pub struct DraftStep {
         /// Likely next user turns to draft; 3 drafts three branch entries.
         pub branches: u32,
         /// Most tokens one draft runs to; 48.
         pub max_tokens: u32,
         /// Sampling temperature in thousandths; 800 is 0.8.
         pub temperature_milli: u32,
     }

     impl DraftStep {
         pub(super) const fn from_config(config: &PromptCacheConfig) -> Self
     }
     ```
     `from_config` copies `follow_up_branches`, `follow_up_max_tokens` and `follow_up_temperature_milli` in that field order. Add `PromptCacheConfig` to the `use` lines at the top if `use super::*` does not already bring it in.
  2. Same file: `follow_up_branches` gains a parameter `step: &DraftStep`, placed between `forced_draft_width` and `on_branch`. The zero guard becomes `step.branches == 0`; the loop `for index in 0..step.branches`; the debug event field `follow_up_requested = u64::from(step.branches)`. `draft_branch` gains `step: &DraftStep` (placed before `index`), and reads `step.temperature_milli` and `step.max_tokens` where it read the config fields. Every other use of `config` in both functions stays.
  3. Same file: `prewarm_follow_ups_with_progress` passes `&DraftStep::from_config(&effective.prompt_cache)`.
  4. `interop/generate/prewarm.rs`: `prewarm_queued` passes `&DraftStep::from_config(&effective.prompt_cache)` to `follow_up_branches`; add `DraftStep` to the `use super::prewarm_follow_up::...` line (add the line if absent).
- test: add a `#[cfg(test)] mod tests` to `prewarm_follow_up.rs` with two tests:
  - `draft_step_from_the_standard_config_is_the_drafting_off_default`: `DraftStep::from_config(&PromptCacheConfig::standard())` equals `DraftStep { branches: 0, max_tokens: 48, temperature_milli: 800 }`;
  - `draft_step_copies_the_follow_up_fields_of_a_configured_cache`: with `PromptCacheConfig { follow_up_branches: 3, follow_up_max_tokens: 64, follow_up_temperature_milli: 700, ..PromptCacheConfig::standard() }` the step equals `DraftStep { branches: 3, max_tokens: 64, temperature_milli: 700 }`.
  The existing real-model test `a_user_turn_that_begins_like_a_drafted_follow_up_reuses_the_branch_past_the_answer` (`generate/prewarm_real_model_tests.rs`, ~line 316) is the byte-for-byte check: it passes before and after this change, with the same assertions.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_10_1 cargo nextest run -p proxima-model-interop --features std --run-ignored all -j 1 -E 'test(/draft_step_|a_user_turn_that_begins_like_a_drafted_follow_up/)'`
- expect: `3 passed` (the two `draft_step_` tests and the real-model test among them)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/prewarm_follow_up.rs`, `proxima-model-interop/src/generate/prewarm.rs`
- commit: `refactor(prewarm): give the draft job its own parameters`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: touch `PromptCacheConfig`, its settings loader, or the cache key; do not change a default; do not re-export `DraftStep` (FT10.4 does) or add any other public item; `from_config` stays `pub(super)`
- gpu: one run, waiting for a quiet box (the peer-gate check in CARDS "machine safety"); one model load

### 10.2 Draft step carries lead tokens forwarded before the draft

- id: FT10.2
- needs: FT10.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prewarm_follow_up.rs::LoadedModel::draft_branch (~line 183 at 2a81f54d)`: the first decode input `vec![base_ids[base_len - 1]]` (~line 203), the `branch_ids` build (~line 230), `keep` (~line 237) and the check `state.ids != branch_ids[..keep]` (~line 244);
  - `interop/generate/prewarm_follow_up.rs::DraftStep` (FT10.1): the struct this card extends;
  - `interop/generate/prompt_cache.rs::PromptCache::take_best (~line 653)`: confirms the base entry is handed back one row short of `base_ids` (`FU` guard `base.state.cached_len + 1 != base_ids.len()`, ~line 117), which is why the first decode input is the last base id.
- change:
  1. `interop/generate/prewarm_follow_up.rs`: `DraftStep` gains `pub lead: Vec<u32>` (doc: "Ids forwarded after the context and before the first sampled token, an instruction such as the encoded text `Rethink what the user may ask next.`; empty forwards nothing"). It loses `Copy` (keep `Debug, Clone, PartialEq, Eq`); `from_config` sets `lead: Vec::new()` and is no longer `const fn`.
  2. Same file: add two private pure functions:
     ```rust
     fn draft_decode_input(base_ids: &[u32], lead: &[u32]) -> Vec<u32>   // [last base id] then lead
     fn branch_ids(base_ids: &[u32], lead: &[u32], draft: &[u32], closing: &[u32]) -> Vec<u32>   // base, lead, draft, closing, in that order
     ```
  3. Same file, `draft_branch`: the first decode input becomes `draft_decode_input(base_ids, &step.lead)`; the build of `branch_ids` becomes a call to `branch_ids(base_ids, &step.lead, draft, closing)`; `keep` becomes `state.cached_len.min(base_len + step.lead.len() + draft.len())`. Nothing else in the function changes.
  4. Same file: in the two `draft_step_` tests of FT10.1, add `lead: Vec::new()` to the expected values.
  If `run_decode_loop_from_ids` (`generate/decode.rs`) rejects or mishandles an input of more than one id from a seeded state, stop and report: the premise that it forwards a multi-id first input is false.
- test: add in the `tests` module of `prewarm_follow_up.rs`, by value on real gemma4 ids (`[2, 105, 2364, 107]` is the gemma4 prompt used in `generate/prewarm.rs` tests; `[818, 5279]` are the ids of "capital of" in the vendored gemma4 E2B parity recording):
  - `draft_lead_decode_input_is_the_last_context_id_then_the_lead`: `draft_decode_input(&[2, 105, 2364, 107], &[818, 5279])` equals `[107, 818, 5279]`, and with an empty lead equals `[107]`;
  - `draft_lead_sits_between_the_context_and_the_draft`: `branch_ids(&[2, 105, 2364, 107], &[818, 5279], &[9079, 236761], &[106, 107, 105, 2364, 107])` equals `[2, 105, 2364, 107, 818, 5279, 9079, 236761, 106, 107, 105, 2364, 107]`, and with an empty lead equals the context, the draft and the closing joined (today's branch ids).
  The default path (empty lead) is checked end to end by the existing real-model test `a_user_turn_that_begins_like_a_drafted_follow_up_reuses_the_branch_past_the_answer`, same assertions as before.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_10_2 cargo nextest run -p proxima-model-interop --features std --run-ignored all -j 1 -E 'test(/draft_step_|draft_lead_|a_user_turn_that_begins_like_a_drafted_follow_up/)'`
- expect: `5 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/prewarm_follow_up.rs`
- commit: `feat(prewarm): forward lead tokens before a drafted follow-up`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change the closing handling or `follow_up_closing`; do not touch `prompt_cache.rs`; do not add an item beyond the `lead` field and the two private functions
- gpu: one run, waiting for a quiet box; one model load

### 10.3 Draft step can keep its branch as an ordinary entry

- id: FT10.3
- needs: FT10.2
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prewarm_follow_up.rs::LoadedModel::draft_branch (~line 183 at 2a81f54d)`: `entry.branch_base = Some(base_len);` (~line 253);
  - `interop/generate/prompt_cache.rs::PromptCache::eviction_victim (~line 820)`: an entry with `branch_base.is_some()` is evicted before any other; and `take_best_shifting`'s `follow_up_hit_tokens` (~line 675), which counts only entries with `branch_base` set;
  - `interop/generate/prewarm_real_model_tests.rs::a_user_turn_that_begins_like_a_drafted_follow_up_reuses_the_branch_past_the_answer (~line 316)`: the setup to copy;
  - `interop/generate/prewarm_real_model_tests.rs::answered_turn (~line 301)` and `follow_up_config (~line 283)`.
- change:
  1. `interop/generate/prewarm_follow_up.rs`: `DraftStep` gains `pub keep: bool` (doc: "Store the branch as an ordinary entry that survives cache pressure like one a request produced, instead of an unused branch evicted first; a derived context a later request is expected to use is kept"). `from_config` sets `keep: false`. In `draft_branch` replace `entry.branch_base = Some(base_len);` with `entry.branch_base = (!step.keep).then_some(base_len);`. In the `draft_step_` tests add `keep: false` to the expected values.
- test: add `a_kept_draft_is_reused_without_counting_as_a_follow_up_hit` in `interop/generate/prewarm_real_model_tests.rs` (`#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]`, `use` `DraftStep` through `super::prewarm_follow_up::DraftStep`). One `with_model`, one model load: `let turn = answered_turn(model)`, `model.set_follow_up_closing(&turn.closing)`, `let config = follow_up_config()`, `model.prewarm(&turn.base, &config)`, then `let effective = model.effective_serving_config(&config).expect(..)`, `let mut runtime = BackendRuntime::new(&effective)` (find the import path with `git grep -n "BackendRuntime" proxima-model-interop/src/generate`; stop and report if it is not reachable from this test module), `let drafts = model.follow_up_branches(&turn.base, &effective, &mut runtime, None, &DraftStep { branches: 1, max_tokens: 48, temperature_milli: 800, lead: Vec::new(), keep: true }, &mut |_| {}).expect(..)`. Then a request of `turn.base` plus the first `half = drafts[0].len().div_ceil(2)` tokens of `drafts[0]` plus `encode_continuation(model, " and then what about the weather?")`, run with `run_cached(model, cached_config(SpeculativeConfig::none()), &request)`. Asserts:
  - `drafts.len() == 1`;
  - `follow_up_hit_tokens == 0` (the kept branch is an ordinary entry, so the reuse is not counted as a follow-up hit);
  - `reused_tokens == turn.base.len() + half` (the kept branch still served the request past the answer).
  The control, which asserts the opposite, is the existing test `a_user_turn_that_begins_like_a_drafted_follow_up_reuses_the_branch_past_the_answer`, whose default drafts (`keep: false`) assert `follow_up_hit_tokens == half` for the same request shape; it is not run in this card (one model-loading run), and its unchanged assertions are checked by the slice exit.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_10_3 cargo nextest run -p proxima-model-interop --features std --run-ignored all -j 1 -E 'test(/draft_step_|a_kept_draft_is_reused_without_counting/)'`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/prewarm_follow_up.rs`, `proxima-model-interop/src/generate/prewarm_real_model_tests.rs`
- commit: `feat(prewarm): let a drafted branch be stored as an ordinary entry`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `eviction_victim`, `take_best_shifting` or `CacheEntry`; do not change what `branch_base` means for entries with `keep = false`
- gpu: one run, waiting for a quiet box; one model load

### 10.4 The idle step list type and its default list

- id: FT10.4
- needs: FT10.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prewarm_follow_up.rs::DraftStep` (FT10.1 to FT10.3): the element payload;
  - `interop/generate/prewarm.rs::LoadedModel::prewarm_queued (~line 340 at 2a81f54d)`: the fixed sequence today, `prewarm_ids` then (when neither skipped nor preempted) `follow_up_branches` (~lines 353 to 373);
  - `interop/lib.rs` (~lines 115 to 118) and `interop/generate/mod.rs::pub use prewarm::{PrewarmReport, PrewarmSkip}` (~line 263): where public report types are re-exported.
- change:
  1. `interop/generate/prewarm_follow_up.rs`: `DraftStep` is already `pub` with `pub` fields (FT10.1 to FT10.3); this card only re-exports it (changes 2 and 3). The one new public item is `IdleStep`, with its associated function `default_list`. Add beside `DraftStep`:
     ```rust
     #[derive(Debug, Clone, PartialEq, Eq)]
     pub enum IdleStep {
         /// Prefill the queued prefix into the prompt cache.
         Prewarm,
         /// Draft likely next user turns behind it, each with its own parameters.
         Draft(DraftStep),
     }

     impl IdleStep {
         #[must_use]
         pub fn default_list(config: &PromptCacheConfig) -> Vec<Self>
     }
     ```
     `default_list` returns `vec![Self::Prewarm, Self::Draft(DraftStep::from_config(config))]`: the sequence `prewarm_queued` runs today, with the existing `follow_up_*` fields still the single source of the draft's numbers. `from_config` stays `pub(super)`. The doc comment names the primitive it composes: it is the list form of the two steps `prewarm_queued` runs, whose gate is `PrewarmGate`.
  2. `interop/generate/mod.rs`: `pub use prewarm_follow_up::{DraftStep, IdleStep};` beside the `prewarm` re-export.
  3. `interop/lib.rs`: add `DraftStep, IdleStep,` to the `pub use generate::{...}` list.
  Why a type: a list holding two drafts with different `max_tokens` or lead cannot be a `Copy` field of `PromptCacheConfig` (`ServingConfig` is `Copy`, `serving.rs` ~line 719); the enum is the smallest element that holds `prewarm` and a parameterised `draft` in one list.
- test: add in the `tests` module of `prewarm_follow_up.rs`, reaching the type through the crate root (`crate::IdleStep`, `crate::DraftStep`) so the re-export is exercised:
  - `idle_steps_default_list_is_prewarm_then_a_draft_taken_from_the_config`: `IdleStep::default_list(&PromptCacheConfig::standard())` equals `vec![IdleStep::Prewarm, IdleStep::Draft(DraftStep { branches: 0, max_tokens: 48, temperature_milli: 800, lead: Vec::new(), keep: false })]`;
  - `idle_steps_default_list_follows_the_configured_follow_up_fields`: with `follow_up_branches: 3, follow_up_max_tokens: 64, follow_up_temperature_milli: 700` the second element is `IdleStep::Draft(DraftStep { branches: 3, max_tokens: 64, temperature_milli: 700, lead: Vec::new(), keep: false })`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_10_4 cargo nextest run -p proxima-model-interop --features std -E 'test(/idle_steps_default_list_/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/prewarm_follow_up.rs`, `proxima-model-interop/src/generate/mod.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(prewarm): add the idle step list type with the default list`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `prewarm_queued` or any behaviour; do not add a field to `PromptCacheConfig`; do not make `from_config` `pub` or add any public item besides `IdleStep`
- gpu: none

### 10.5 The prewarm worker hands its drafts to the caller

- id: FT10.5
- needs: FT10.4
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prewarm.rs::LoadedModel::with_prewarm_worker (~line 397 at 2a81f54d)`, `serve_queued_prewarms (~line 413)`, `run_queued_prewarm (~line 330)`, `prewarm_queued (~line 340)`, `run_pending_prewarm (~line 321)`: the chain the observer passes through; the worker path today discards the drafts `follow_up_branches` returns;
  - `interop/generate/prewarm_follow_up.rs::LoadedModel::prewarm_follow_ups (~line 53)`: the caller-driven path already returns `Vec<Vec<u32>>`, the shape the observer receives;
  - `interop/generate/prewarm_real_model_tests.rs::an_answer_returns_with_its_end_of_answer_prefill_still_queued (~line 468)`: the queued prefix is the opening, the generated ids and the registered suffix, in that order;
  - `interop/examples/prompt_cache_bench.rs` (~lines 1228 and 1409, 1561): existing callers of `with_prewarm_worker` and `run_pending_prewarm`, which keep their signatures.
- change:
  1. `interop/generate/prewarm.rs`: add
     ```rust
     pub fn with_prewarm_worker_observed<T>(
         &self,
         serving_config: &ServingConfig,
         observer: impl FnMut(&[u32], &[Vec<u32>]) + Send,
         body: impl FnOnce() -> T,
     ) -> Result<T, InteropError>
     ```
     moving the current body of `with_prewarm_worker` into it, the observer moving into the scoped worker thread. `with_prewarm_worker` becomes `self.with_prewarm_worker_observed(serving_config, |_source, _drafts| {}, body)`. Doc on the new method: `observer` is called on the worker thread with the prefix the drafting ran behind and the drafts it returned, once per queued job that drafted at least one branch; it is the output channel of an idle job, and the same pair `prewarm_follow_ups` returns on the caller-driven path.
  2. Same file: `serve_queued_prewarms`, `run_queued_prewarm` and `prewarm_queued` each take `observer: &mut dyn FnMut(&[u32], &[Vec<u32>])`. In `prewarm_queued`, the guarded `if let Err(error) = self.follow_up_branches(..)` becomes a `match` on that call under the same guard (`report.skipped.is_none() && !report.preempted`): `Ok(drafts)` with `!drafts.is_empty()` calls `observer(&job.prefix, &drafts)`, `Ok(_)` does nothing, `Err(error)` logs the same `warn!` as today. `run_pending_prewarm` passes `&mut |_source, _drafts| {}`.
- test: add `a_worker_hands_the_drafts_and_their_source_ids_to_the_observer` in `prewarm_real_model_tests.rs` (`#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]`): one `with_model`, one model load; `opening = encode_opening(model, &chat_prompt(USER_FROM_CHAR))`, `suffix = turn_boundary_suffix(model)`, `set_prewarm_suffix(&suffix)`, `set_follow_up_closing(&encode_continuation(model, MODEL_TURN_CLOSING))`, `config = follow_up_config()` with `config.prompt_cache.follow_up_branches = 2`; an `observed: Mutex<Vec<(Vec<u32>, Vec<Vec<u32>>)>>`; `outcome = with_prewarm_worker_observed(&config, |source, drafts| observed.lock().expect("..").push((source.to_vec(), drafts.to_vec())), || { let outcome = run_cached(model, config, &opening); model.wait_for_prewarm(); outcome }).expect("..")`. Asserts:
  - `observed.len() == 1`;
  - the observed source ids equal `opening`, then `outcome.generated`, then `suffix`, joined in that order;
  - the observed drafts number exactly 2;
  - each draft is non-empty and holds at most 48 ids (the default `follow_up_max_tokens`).
  If the draft count differs from 2, stop and report the observation; do not change the assertion. The unchanged-behaviour check for the worker without an observer is the existing test `a_worker_prewarms_the_answer_so_the_next_turn_prefills_only_the_user_tokens`, run by the slice exit and not in this card (one model-loading run).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_10_5 cargo nextest run -p proxima-model-interop --features std --run-ignored all -j 1 -E 'test(/a_worker_hands_the_drafts_and_their_source_ids_to_the_observer/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/prewarm.rs`, `proxima-model-interop/src/generate/prewarm_real_model_tests.rs`
- commit: `feat(prewarm): hand the worker's drafts to a caller observer`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `with_prewarm_worker`'s or `run_pending_prewarm`'s signature; do not store the observer in the model; do not add a trait or a type for it
- gpu: one run, waiting for a quiet box; one model load

### 10.6 The idle step list runs the queued job

- id: FT10.6
- needs: FT10.5
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prewarm.rs::LoadedModel::prewarm_queued (~line 340 at 2a81f54d)`: the key check, then the two-step sequence (with the observer call from FT10.5) this card turns into a loop over the list;
  - `interop/generate/prewarm.rs::LoadedModel::set_prewarm_suffix (~line 141)`: the setter pattern, a `&self` method locking `prompt_cache`;
  - `interop/generate/prompt_cache.rs::PromptCache (~line 539)` and `set_follow_up_closing` / `follow_up_closing` (~lines 579 to 585): the storage pattern for a token list that `Copy` config cannot hold;
  - `interop/generate/prewarm_real_model_tests.rs::a_worker_hands_the_drafts_and_their_source_ids_to_the_observer` (FT10.5): the worker and observer setup to copy.
- change:
  1. `interop/generate/prompt_cache.rs`: `PromptCache` gains `idle_steps: Vec<IdleStep>` (empty in `new`), `pub(super) fn set_idle_steps(&mut self, steps: &[IdleStep])` and `pub(super) fn idle_steps(&self) -> &[IdleStep]`, beside the closing accessors. Import `IdleStep` from `super::prewarm_follow_up`.
  2. `interop/generate/prewarm.rs`: add `pub fn set_idle_schedule(&self, steps: &[IdleStep])` beside `set_prewarm_suffix`, locking and calling `set_idle_steps`. Doc: an empty list restores today's behaviour (`prewarm`, then drafting from the `follow_up_*` fields); a non-empty list replaces both, runs in order for each queued end-of-answer job, and is a setter and not a config field because the config is `Copy`. It names `IdleStep` and `PrewarmGate` as the primitives it composes.
  3. Same file, `prewarm_queued`: after the key check, `let steps = <the stored list, cloned>`; when it is empty use `IdleStep::default_list(&effective.prompt_cache)`. Then `let mut report = PrewarmReport::idle(None, false);` and loop over `&steps`:
     - `IdleStep::Prewarm`: `report = self.prewarm_ids(&job.prefix, effective, runtime, job.forced_draft_width, &mut |_position| {})?;` then `break` when `report.skipped.is_some() || report.preempted`;
     - `IdleStep::Draft(draft)`: call `self.follow_up_branches(&job.prefix, effective, runtime, job.forced_draft_width, draft, &mut |_kept| {})`; on `Ok(drafts)` with `!drafts.is_empty()` call `observer(&job.prefix, &drafts)`; on `Err(error)` log the same `warn!` as today (`follow_up_error = %error`, message `follow-up prewarm failed after the request succeeded`) and `break`.
     End with `Ok(report)`. With no stored list this is today's sequence: a skipped or preempted prewarm stops before drafting, and a draft error only warns.
- test:
  - add `idle_steps_set_on_the_cache_are_returned_in_order` in the `tests` module of `prompt_cache.rs`: a new `PromptCache` returns an empty slice; after `set_idle_steps(&[draft_a, IdleStep::Prewarm, draft_b])` (two `IdleStep::Draft` values differing in `max_tokens`, 16 and 64) it returns those three in that order; after `set_idle_steps(&[])` it returns an empty slice;
  - add `idle_job_list_drives_the_end_of_answer_prewarm` in `prewarm_real_model_tests.rs` (`#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]`): one `with_model`, one model load, set up as the observer test of FT10.5 with `follow_up_config()` left at its 3 configured branches, then `set_idle_schedule(&[IdleStep::Prewarm, IdleStep::Draft(DraftStep { branches: 1, max_tokens: 16, temperature_milli: 800, lead: Vec::new(), keep: false })])`, observed through `with_prewarm_worker_observed`. Asserts:
    - `observed.len() == 1`;
    - the observed drafts number exactly 1 (the list, not the config's 3, supplied the draft count);
    - that draft is non-empty and holds at most 16 ids (the list's `max_tokens`, not the config's 48).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_10_6 cargo nextest run -p proxima-model-interop --features std --run-ignored all -j 1 -E 'test(/idle_steps_set_on_the_cache|idle_job_list_drives_the_end_of_answer_prewarm/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/prompt_cache.rs`, `proxima-model-interop/src/generate/prewarm.rs`, `proxima-model-interop/src/generate/prewarm_real_model_tests.rs`
- commit: `feat(prewarm): run a configured step list for the queued prewarm job`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `PrewarmQueue`, `PrewarmGate`, the cache key, or `prewarm_ids`; do not read the list anywhere else
- gpu: one run, waiting for a quiet box; one model load

### 10.7 A caller-supplied idle job runs under the request-waiting gate

- id: FT10.7
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prewarm_gate.rs::PrewarmGate (~line 21 at 2a81f54d)`, `try_begin (~line 68)`, `request_waiting (~line 77)`: the slot and the pending-request count; the field `pending` is reachable from the gate's own test module;
  - `interop/generate/prewarm_gate.rs` tests (`a_request_waits_for_the_prewarm_to_yield_and_the_prewarm_sees_it`, ~line 118): the spin-until-a-request-shows pattern, no sleeps;
  - `interop/generate/prewarm.rs::LoadedModel::prewarm_ids (~line 183)`: how the model takes the slot and returns idle when a request is waiting;
  - `interop/generate/prewarm_real_model_tests.rs::with_model` import (~line 8).
- change:
  1. `interop/generate/prewarm_gate.rs`: add to `impl PrewarmGate`
     ```rust
     pub(super) fn run_idle<T>(&self, job: impl FnOnce(&dyn Fn() -> bool) -> T) -> Option<T>
     ```
     Body: take `self.try_begin()?`, return `None` when `self.request_waiting()`, otherwise `Some(job(&|| self.request_waiting()))`, holding the slot for the whole call.
  2. `interop/generate/prewarm.rs`: add
     ```rust
     pub fn run_idle_job<T>(&self, job: impl FnOnce(&dyn Fn() -> bool) -> T) -> Option<T>
     ```
     forwarding to `self.prewarm_gate.run_idle(job)`. Doc: runs `job` on the calling thread holding the device slot a prewarm holds ([`PrewarmGate`]); `None` when a request is waiting or another idle job holds the slot; the argument to `job` reports whether a request has arrived since, which the job polls at its own boundaries and returns early on, the same contract `prewarm` keeps per chunk. Say why it exists beside `prewarm` and `prewarm_follow_ups`: those are the jobs the library defines; this is the entry for a job it does not.
- test: in the `tests` module of `prewarm_gate.rs`, model-free, with `gate.pending.fetch_add(1, Ordering::SeqCst)` and `fetch_sub` standing in for a request arriving (so the test holds no slot-waiting thread):
  - `idle_job_runs_and_returns_its_value_while_no_request_waits`: `gate.run_idle(|_poll| 7)` is `Some(7)`;
  - `idle_job_does_not_start_while_a_request_is_waiting`: with `pending` raised, `gate.run_idle` is `None` and a counter the job would bump stays 0;
  - `idle_job_does_not_start_while_a_prewarm_holds_the_slot`: with `let slot = gate.try_begin()` held, `gate.run_idle(|_| 1)` is `None`; after `drop(slot)` it is `Some(1)`;
  - `idle_job_sees_a_request_arrive_through_its_poll`: the job records `poll()`, raises `pending` itself, records `poll()` again; the recorded pair is `(false, true)`.
  and in `prewarm_real_model_tests.rs` (`#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]`) add `idle_job_through_the_model_runs_when_the_device_is_idle`: one `with_model`; `model.run_idle_job(|poll| poll())` is `Some(false)`; after a `run_cached` request finishes, the same call is `Some(false)` again (the request released the gate).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_10_7 cargo nextest run -p proxima-model-interop --features std --run-ignored all -j 1 -E 'test(/idle_job_runs_and_returns|idle_job_does_not_start|idle_job_sees_a_request|idle_job_through_the_model/)'`
- expect: `5 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/prewarm_gate.rs`, `proxima-model-interop/src/generate/prewarm.rs`, `proxima-model-interop/src/generate/prewarm_real_model_tests.rs`
- commit: `feat(prewarm): run a caller-supplied idle job under the request gate`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: queue the job; change `try_begin`, `enter_request` or `PrewarmQueue`; make `PrewarmGate` public; add a sleep to any test
- gpu: one run, waiting for a quiet box; one model load

### 10.8 Proof: a draft with a lead and keep expresses sleep-time derived context

- id: FT10.8
- needs: FT10.3, FT10.6
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prewarm_real_model_tests.rs::a_worker_hands_the_drafts_and_their_source_ids_to_the_observer` (FT10.5): the worker and observer setup to extend;
  - `interop/generate/prewarm_real_model_tests.rs::a_user_turn_that_begins_like_a_drafted_follow_up_reuses_the_branch_past_the_answer (~line 316 at 2a81f54d)`: the request-against-a-fresh-run comparison to copy;
  - `interop/generate/prewarm_follow_up.rs::draft_branch`: the lead and keep behaviour from FT10.2 and FT10.3 this test drives.
  No library change in this card. The technique appears only here, in a test of about 30 lines.
- change:
  1. `interop/generate/prewarm_real_model_tests.rs`: add the test below and nothing else.
- test: add `a_draft_with_a_lead_and_keep_stands_in_for_derived_context_the_next_request_reuses` (`#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]`). One `with_model`:
  - `lead = encode_continuation(model, "Summarize what the user is likely to ask next.\n")`, `closing = encode_continuation(model, MODEL_TURN_CLOSING)`;
  - `set_prewarm_suffix(&turn_boundary_suffix(model))`, `set_follow_up_closing(&closing)`, `set_idle_schedule(&[IdleStep::Prewarm, IdleStep::Draft(DraftStep { branches: 1, max_tokens: 16, temperature_milli: 800, lead: lead.clone(), keep: true })])`;
  - run `with_prewarm_worker_observed` as in the observer test, collecting `(source, drafts)`; assert `drafts.len() == 1`;
  - `derived = source ++ lead ++ drafts[0] ++ closing`; `request = derived ++ encode_continuation(model, "And why?")`;
  - `outcome = run_cached(model, cached_config(SpeculativeConfig::none()), &request)`; `fresh = run_fresh(model, uncached_config(SpeculativeConfig::none()), &request)`.
  Asserts: `outcome.report.reused_tokens == derived.len()` (the idle job's derived context, not the original context, served the request); `outcome.report.follow_up_hit_tokens == 0` (it was kept as an ordinary entry); `outcome.generated == fresh`. The control that must fail is the request `source ++ encode_continuation(model, "And why?")` against the same cache: assert its `reused_tokens` is `source.len()`, strictly less than `derived.len()`.
  If `reused_tokens` differs from `derived.len()`, stop and report the printed `CacheReport`; do not adjust the assertion.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_10_8 cargo nextest run -p proxima-model-interop --features std --run-ignored all -j 1 -E 'test(/a_draft_with_a_lead_and_keep_stands_in_for_derived_context/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/prewarm_real_model_tests.rs`
- commit: `test(prewarm): derive context in an idle draft and reuse it next turn`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add any non-test code; do not name a technique, paper or card id in the test or its comments
- gpu: one run, waiting for a quiet box; one model load

### 10.9 Proof: compaction of stored bytes as a caller-supplied idle job

- id: FT10.9
- needs: FT10.7
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/prewarm_gate.rs::PrewarmGate::run_idle` and `interop/generate/prewarm.rs::LoadedModel::run_idle_job` (FT10.7): the entry the job runs under;
  - `interop/generate/prewarm_real_model_tests.rs::a_request_arriving_mid_prewarm_waits_one_chunk_and_reuses_the_partial_prewarm (~line 189 at 2a81f54d)`: the scoped-thread, channel-signal and request-while-the-job-runs pattern, with no sleeps;
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/` (`gguf_kv.txt`, `swa_layers.txt`, `llama_ids.json`): the real recorded bytes the job merges; `tempfile` is a dev-dependency (`proxima-model-interop/Cargo.toml` ~line 356).
  No library change in this card. Compaction appears only in this test, as a job of about 25 lines.
- change:
  1. `interop/generate/prewarm_real_model_tests.rs`: add the test below and nothing else.
- test: add `a_compaction_job_merges_stored_files_and_yields_to_a_waiting_request` (`#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]`). The job, a local closure over `files: [PathBuf; 3]` (the three fixture files, joined to `env!("CARGO_MANIFEST_DIR")`) and `segment: PathBuf` (`tempfile::tempdir()` joined with `segment-000000.seg`): for each file, return the count merged so far when `poll()` is true; otherwise append the file's bytes to the segment and, after the first file only, `sender.send(())` then spin on `poll()` (`std::hint::spin_loop`) until it is true. One `with_model`, a `std::thread::scope`: the job runs through `model.run_idle_job` on a spawned thread; the scope's own thread receives the signal, then runs `run_cached` of a short prompt (`encode_opening(model, &chat_prompt(USER_FROM_CHAR))`) and `run_fresh` of the same. Asserts:
  - the interrupted job returned `Some(1)` and the segment holds exactly the bytes of the first file;
  - the request's generated ids equal `run_fresh`'s (the job did not disturb it);
  - after the request finished, a second `model.run_idle_job` over the same three files with a fresh segment path returns `Some(3)` and the segment equals the three files' bytes concatenated in order.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_10_9 cargo nextest run -p proxima-model-interop --features std --run-ignored all -j 1 -E 'test(/a_compaction_job_merges_stored_files_and_yields_to_a_waiting_request/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/prewarm_real_model_tests.rs`
- commit: `test(prewarm): run a compaction job through the idle job entry`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add any non-test code, a file format, or a segment type to the library; do not add a sleep
- gpu: one run, waiting for a quiet box; one model load

## slice exit

- Every card FT10.1 to FT10.9 printed its expect line.
- `cargo nextest run -p proxima-model-interop --features std -E 'test(/draft_step_|draft_lead_|idle_steps_default_list_|idle_steps_set_on_the_cache|idle_job_runs_and_returns|idle_job_does_not_start|idle_job_sees_a_request/)'` prints `11 passed` (2 + 2 + 2 + 1 + 4; the model-free tests).
- `cargo nextest run -p proxima-model-interop --features std --run-ignored all -j 1 -E 'test(/a_kept_draft_is_reused_without_counting|idle_job_list_drives_the_end_of_answer_prewarm|a_worker_hands_the_drafts_and_their_source_ids|idle_job_through_the_model|a_draft_with_a_lead_and_keep_stands_in|a_compaction_job_merges_stored_files/)'` prints `6 passed` (checked once, on a quiet box, one model at a time).
- The existing real-model follow-up and worker tests still pass with the same assertions: `a_user_turn_that_begins_like_a_drafted_follow_up_reuses_the_branch_past_the_answer` and `a_worker_prewarms_the_answer_so_the_next_turn_prefills_only_the_user_tokens`, `2 passed` under `--run-ignored all -j 1`.
- `cargo check -p proxima-model-interop --features std,metal --example prompt_cache_bench --message-format json > <log> 2>/dev/null`, then `grep -c '"reason":"compiler-artifact".*"name":"prompt_cache_bench"' <log>` prints `1` (the example that calls `with_prewarm_worker` and `run_pending_prewarm`; it needs the `metal` feature, so a `--features std` check does not build it).
- `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'binary(arch_data_baseline) & test(/^llama_parity_/)'` prints `4 passed` of 7, the count recorded in TASKS.md (token selection and random-number order are untouched by this slice).
- N == 0 on any line above is RED.
