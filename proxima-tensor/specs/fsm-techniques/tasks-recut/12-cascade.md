# slice 12: Cascade (re-cut; cards FT12.3 - FT12.7 and FT12.9 - FT12.17; ids FT12.1, FT12.2 and FT12.8 are dropped)

anchors read at main a7c08c4c (full sha a7c08c4c95836f112742f0e30e4ddd672087cda0) with `git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`. Every `path::symbol (~line N)` below is the line at that sha; the executor re-locates by symbol. `interop` means `proxima-model-interop/src`; `generate` means `proxima-model-interop/src/generate`. Integration test files are always written out in full as `proxima-model-interop/tests/<name>.rs`; they are never under `interop`. Between bb84492c and a7c08c4c, `git diff` changed three files this slice reads: `proxima-model-interop/Cargo.toml` (the `std` feature gained `proxima-primitives/std`), `generate/mod.rs` (a new `use proxima_primitives::sync::blocking::Mutex;`) and `generate/decode.rs` (`lock_expert_slab(&self.expert_slab)` became `self.expert_slab.lock()`). None of the three changes a symbol or signature anchored below; every line hint was re-read at a7c08c4c, and the two that moved (`real_gemma4_moe_gguf_path`, the last `*_real_model_tests` line in `generate/mod.rs`) carry their new values.

Governing direction (owner, 2026-10-04): the hooks are built so techniques can be vetted later; a technique is not built here. This slice builds the settle hook: after a tier answers, a pure judge decides "settle this answer, or escalate to the next tier", over a configured list of tiers; the default (one tier, one draw, a judge closure that settles) is the same single `generate_with_serving_config` call a caller makes today. Every technique (threshold, classifier, conformal vote, isotonic, nearest-centroid router) appears only as a test that expresses it through the hook; the library holds the hook and no decision rule. Governing spec: `proxima-windows/proxima-tensor/specs/pipeline-as-data/SPEC.md` (hook catalog row H17, structural findings) and `sketches/13-conformal-cascade.md` (the shape: configuration, one pure function, one pipe of about 40 lines; gaps G1 to G6).

Placement result: this slice changes no `proxima-core` file. The loop over tiers is one pipe in `interop/cascade.rs`, beside its single caller (sketch gap G1: the decision has one consumer); the vote decision of sketch 13 is not in the library at all, it is a function in the conformal proof test (FT12.11), because the only library hook is the `judge` closure on `Tier` and a vote rule is one closure. The tier list lives in `CascadeSettings`, a `Vec` outside `ServingConfig`, because `ServingConfig` is `Copy` (`interop/serving.rs::ServingConfig (~line 720)`, SPEC structural findings) and cannot hold a list. No cache key changes: every sampling field is excluded from the prompt-cache key (`interop/generate/prompt_cache_key.rs`, the destructure with `temperature: _` through `seed: _`, ~lines 126-134) and each tier is a separate `LoadedModel` with its own prompt cache, so the cascade changes no cached row's meaning.

Public-item rule for this slice: a card adds at most one public item that a later card consumes, where a type together with its inherent methods and trait impls is one item. The cuts follow from that: the cascade error (FT12.3), the draw record (FT12.4), the tier and its answer method (FT12.5), the cascade pipe (FT12.6), the tier settings (FT12.9), the cascade settings (FT12.10). FT12.7 adds a method and a field to the existing cascade, not a new item. Every other card adds a test file or a test only.

Designs abandoned, and why:
- A per-judge enum in core (`Judge::{Always, Threshold, Classifier, Conformal, Isotonic}`) with `settle_tier`: the five judges were five techniques in the library. The judge is a borrowed pure function over the draws and the library names no judge kind.
- A data judge in the library (an `Agreement { samples, max_disagree }` value in `proxima-core/src/settle.rs` with a `settle` method, an `AgreementError::check` validating its tuple, a `vote_judge` over it in `interop/cascade.rs`, and `max_disagree` on `TierSettings`): this is the conformal prediction-set vote (an answer stays in the set when its disagreeing draws are at most `max_disagree`, the tier settles iff exactly one answer is in the set) built into the library. Writing the call site both ways gives the same line: `Tier { judge: &vote_judge(agreement), .. }` against `Tier { judge: &vote_judge_from_the_test, .. }`, so the library type bought the caller nothing the closure did not; the always-settle default is the closure `|_: &[Draw]| Ok(Some(0))`. The vote rule, its tally and its calibration all moved into the conformal proof test (FT12.11), as the other four techniques are.
- Validation rows for the vote tuple (zero samples, `max_disagree` above `samples`, the last tier drawing several samples, several samples without sampling): they validated one technique's tuple. The hook needs none of them, because a last tier whose judge declines is already `CascadeError::Unsettled` and a judge reports its own failure as `CascadeError::Judge`.
- `draw: &'model dyn Fn(&str, usize, ServingConfig<'model>) -> ..` on `Tier`, with test drawers annotated `ServingConfig<'static>`: derived from reading, not compiled (this card file forbids builds). A closure whose parameter is `ServingConfig<'static>` implements only `Fn(.., ServingConfig<'static>)`, and the field type needs `Fn(.., ServingConfig<'model>)`; coercing `&drawer` to the field therefore forces `'model = 'static`, which needs a `&'static` borrow of a local closure and its `Cell` and `RefCell` captures, a borrow-does-not-live-long-enough error. The drawer's config parameter is independent of the tier's lifetime (`LoadedModel::generate_with_serving_config` takes `serving_config: ServingConfig` with an elided lifetime, `interop/generate/decode.rs (~line 2107)`), so the field is higher-ranked over it: `dyn Fn(&str, usize, ServingConfig<'_>)`.
- Public type aliases `Drawer`, `Judge` and `Router` for the three closure types: each is a public item a later card consumes, and a caller who writes the closure never names the alias. The `draw` and `route` closure types are written inline on the field; the judge closure type trips `clippy::type_complexity` inline (measured on a scratch crate: `&'model dyn Fn(&[Draw]) -> Result<Option<usize>, CascadeError>` as a field type warns, and the build denies warnings), so FT12.5 declares it as a private alias, which a public field may use and which is not a public item.
- `AndThen` over `Result<Answer, Request>` with a pass-through wrapper pipe per tier (the old Router, Tier and Settle pipes, plus an enum of arms 1 to 4 over the type-level chain): the wrappers exist only to make one pipe composable and the chain type grows with the tier count (sketch section 6, section 7).
- `Fallback<primary, secondary>`: it falls back on any `Err` including a failed forward, so a failed generation would silently escalate (`proxima-primitives/src/pipe/resilience/fallback.rs`, `Fallback (~line 12)`); the cascade returns a draw failure as an error and never escalates on it.
- Carrying a `ServingConfig` inside `LoadedModel`'s `Pipe` impl so a tier is that pipe (sketch gap G4): `LoadedModel`'s impl (`interop/generate/residency_caches.rs::impl Pipe for LoadedModel (~line 3146)`) calls `generate`, which fixes the config. A tier's draw function is `|prompt, max_tokens, serving| model.generate_with_serving_config(prompt, max_tokens, serving)`, which is the public call that already takes a config; making `LoadedModel` hold a config adds a field and a constructor for the same call. That impl stays as it is.
- A q-hat table keyed by prompt class and the offline calibration fitter in the library: the order statistic is five lines and has no runtime caller, so it lives in the conformal proof test.
- A `samples` knob in the decode loop: N samples are N calls with `seed + index` (sketch gap G2); sampling fields are public on `ServingConfig::seed (~line 829)`.

Closure rule for every card of this slice (derived from reading, not compiled): a closure bound with `let` and later borrowed into a `dyn Fn` field is higher-ranked over its reference or config parameters only when those parameters are annotated, because with no expected type an unannotated parameter infers one fixed region and the coercion to the field then fails with "not general enough". So test drawers are written `|prompt: &str, max_tokens: usize, serving: ServingConfig<'_>| -> Result<Draw, InteropError>`, test judges `|draws: &[Draw]|` (an unused parameter is still annotated, `|_: &[Draw]|`), and test routers `|prompt: &str|`. A closure written inline in the struct literal field or in the argument of `with_router` takes its signature from the field type and needs no annotation. Helper functions that return `impl Fn(&[Draw]) -> ..` take the higher-ranked signature from the return type and need none.

## old to new id map

| old card | new card | verdict |
|---|---|---|
| FT12.1 | none | dropped: the settle decision with always-settle as the default was the vote rule in the library; the always-settle default is the judge closure `|_: &[Draw]| Ok(Some(0))` and the vote rule is a function in FT12.11's test; the `Candidate` readout means, `Request`, `Answer`, `Draft` and `Outcome` types are dropped (they existed only for the chain of wrapper pipes) |
| FT12.2 | FT12.4, FT12.5, FT12.6, FT12.12 | recut: the judge input shape with per-token readout columns (the draw record, the tier, the cascade pipe), and the readout-threshold proof |
| FT12.3 | FT12.14 | recut: the classifier proof; the classifier model call is the caller's |
| FT12.4 | FT12.11 | recut: the vote tally decision, its calibration and its proof, all in one test file; no library item remains |
| FT12.5 | FT12.13 | recut: the isotonic proof; the table is a caller-supplied value |
| FT12.6 | FT12.7, FT12.15 | recut: the starting-tier hook, and the nearest-centroid router proof |
| FT12.7 | FT12.3, FT12.5, FT12.6, FT12.7 | keep in intent: the settle order (skip below the routed tier, the last tier must settle, exactly one tier answers) is the loop of one pipe over a draw function; it lives in `interop/cascade.rs` and runs against fake draws without a model, so the sans-IO suite drives it the same way |
| FT12.8 | none | dropped |
| FT12.9 | FT12.5, FT12.6, FT12.9, FT12.10, FT12.16, FT12.17 | recut: one loop pipe over a tier slice, tier list from settings (FT12.9, FT12.10), and two real-model cards with gemma4 |

## dropped

- FT12.8 (the three stage pipes `Router`, `Tier`, `Settle` with `&dyn Fn` aliases): superseded by the single loop pipe (sketch 13 sections 6 and 7 weighed and abandoned the wrapper-per-tier shape). Nothing on main depends on them.
- FT12.1, FT12.2 and the vote judge card of the previous cut of this file (the `settle.rs` agreement with its settle method, its validation rows, and the vote judge in `cascade.rs`): the vote rule is a technique, so it left the library; see "Designs abandoned". The old FT12.1 and FT12.2 ids were not reused, so a reference to them in another file names a card that no longer exists.

## carried, outside this file's verdict list

- Per-token readouts: the per-token logprob and top-two margin producer on `TokenEvent` (`interop/generate/residency_caches.rs::TokenEvent (~line 3169)` has `token_id`, `text_piece`, `phase`, `step` and `elapsed_ms`, and no probability; sketch gap G5) is the readouts slice's hook. This file adds the consumer side only: `Draw` carries two plain readout columns that a caller's draw function fills, and the threshold and isotonic proofs fill them with scripted values. No card here fills the columns from a real decode.
- Calibration tables keyed by a serving-configuration digest (research CC-10): no digest function exists on main (SPEC structural findings), so the isotonic proof's table is a value its judge closure captures; keying is the caller's.
- The classifier proof uses scripted class probabilities. No Ollama or llama run produced them, so no oracle backs them; they are inputs to the argmax decision, which is the thing under test.
- The calibration fitter and the cluster centroids are training-side and out of scope; the proofs carry their literal values.
- No card in this file loads a granite checkpoint. The two real-model cards use gemma4 only, the dense gemma4 E2B and the gemma4 26B mixture-of-experts model; granite reachability is a configuration card in another file.

## prerequisites outside this slice

None. Every card below uses only symbols present at main a7c08c4c. No card uses the serving state, the serving config grammar, the seal rule, or the readouts slice.

Test models (owner, 2026-10-04): gemma4 for dense or MoE, granite for MoE. No qwen of any kind appears in any card of this file. The sketch's example config named a qwen checkpoint for the second tier; every card here names the gemma4 26B MoE blob instead. Local blobs, from `ollama show --modelfile`: gemma4 E2B (`gemma4:e2b-it-qat`) `/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`; gemma4 26B MoE (`batiai/gemma4-26b:latest`) `/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129`. Cards other than FT12.16 and FT12.17 load no model; their draws are scripted, so they run on any box.

Test-name rule for this slice: new test names say what they check in English; none contains a card, slice, requirement or worked-example id. Everything a card tells the executor to write (code, doc comments, test names, error text) is plain English with no id. Every card uses `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_<n>` and removes it when done; logs go under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/cards12/`. Test modules in `src` carry `#[allow(clippy::unwrap_used, clippy::expect_used)]` where they use those (precedent `proxima-core/src/batch.rs`, the `mod tests` header ~line 135); integration test files carry the same attribute at the top. No sleeps. No comments in code except a lowercase why.

Real prompts in every test are needle questions: `"What is the special magic number for marmot? Answer with the number only.\nAnswer:"`, with `marmot` replaced by `lynx`, `heron` or `tapir` where a second prompt is needed. Scripted answers are seven-digit numbers such as `"4830912"`; scripted token ids are the code points of the answer's digits (`'4' as u32` and so on), stated in each test as a fake tokenizer.

---

### 12.3 the cascade error type

- id: FT12.3
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/error.rs::InteropError (~line 13), InteropError::UnknownTensor (~line 40)`: the `thiserror` derive this enum copies and the variant the tests use as a real draw failure;
  - `interop/lib.rs::pub mod profiles (~line 66), pub use error::InteropError (~line 102)`: where `#[cfg(feature = "std")] pub mod cascade;` goes, and the path `crate::InteropError` the new file imports.
- change:
  1. `interop/cascade.rs` (new, std; import at the module top: `crate::InteropError`):
     ```rust
     #[derive(Debug, thiserror::Error)]
     pub enum CascadeError {
         #[error("cascade has no tiers")]
         NoTiers,
         #[error("tier {tier} draw failed: {source}")]
         Draw { tier: usize, #[source] source: InteropError },
         #[error("tier {tier} is the last tier and its judge declined")]
         Unsettled { tier: usize },
         #[error("tier {tier} judge chose draw {chosen} of {drawn}")]
         JudgeIndex { tier: usize, chosen: usize, drawn: usize },
         #[error("judge failed: {reason}")]
         Judge { reason: String },
     }
     ```
     One variant per way a cascade request stops: no tiers configured, a draw function failed (the cause is kept as the error source, the cascade never escalates on it), the last tier's judge declined, a judge named a draw that does not exist, and a judge reported its own failure.
  2. `interop/lib.rs`: add `#[cfg(feature = "std")] pub mod cascade;`.
- test: in `cascade.rs` (`#[cfg(test)] mod tests`, `#[allow(clippy::unwrap_used, clippy::expect_used)]`; `use std::error::Error as _;` at the module top):
  - `each_cascade_error_names_its_tier_and_numbers`: `CascadeError::NoTiers.to_string() == "cascade has no tiers"`; `CascadeError::Unsettled { tier: 0 }.to_string() == "tier 0 is the last tier and its judge declined"`; `CascadeError::JudgeIndex { tier: 0, chosen: 5, drawn: 2 }.to_string() == "tier 0 judge chose draw 5 of 2"`; `CascadeError::Judge { reason: "classifier returned no classes".to_owned() }.to_string() == "judge failed: classifier returned no classes"`;
  - `a_draw_failure_names_its_tier_and_keeps_its_cause_as_the_source`: with `cause = InteropError::UnknownTensor { name: "token_embd.weight".to_owned() }` and `expected = format!("tier 1 draw failed: {cause}")` computed first, `CascadeError::Draw { tier: 1, source: cause }.to_string() == expected` and `.source().is_some()`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_3 cargo nextest run -p proxima-model-interop --features std -E 'test(/cascade::tests::/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/cascade.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(cascade): add the cascade error type`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a variant other than these five here (later cards add their own); do not touch `InteropError`
- gpu: none

### 12.4 the draw record a tier produces

- id: FT12.4
- needs: FT12.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/cascade.rs::CascadeError` (FT12.3): the file this card extends;
  - `interop/generate/decode.rs::LoadedModel::generate_with_serving_config (~line 2107 at a7c08c4c)`: returns `(Vec<u32>, String, bool)`; the ids and the text are the first two fields of a draw;
  - `interop/generate/residency_caches.rs::TokenEvent (~line 3169)`: has no probability field on main, so the two readout columns are filled by a caller's draw function, never by the cascade.
- change:
  1. `interop/cascade.rs`: add
     ```rust
     #[derive(Debug, Clone, Default, PartialEq)]
     pub struct Draw {
         /// token ids of one generated completion, for example the ids of "4830912"
         pub ids: Vec<u32>,
         /// the completion decoded, for example "4830912"
         pub text: String,
         /// one log probability per generated token, for example [-0.25, -0.75]; empty when the draw function reports none
         pub logprobs: Vec<f32>,
         /// one top-two probability margin per generated token, for example [0.875, 0.625]; empty when the draw function reports none
         pub margins: Vec<f32>,
     }
     ```
     `Default` is the draw with no tokens and no readouts, so a fixture writes `Draw { ids, text, ..Draw::default() }` and a judge that needs a column tests it for empty.
- test: in `cascade.rs` tests (fake tokenizer: ids are `u32::from(digit)` of each char of `"4830912"`):
  - `a_default_draw_has_no_tokens_and_no_readouts`: `Draw::default()` equals `Draw { ids: Vec::new(), text: String::new(), logprobs: Vec::new(), margins: Vec::new() }`;
  - `a_cloned_draw_equals_its_source_and_a_changed_readout_differs`: a draw with the digit ids, text `"4830912"`, `logprobs: vec![-0.25, -0.75]`, `margins: vec![0.875, 0.625]`; its clone equals it; the clone with `margins: vec![0.875, 0.5]` does not.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_4 cargo nextest run -p proxima-model-interop --features std -E 'test(/cascade::tests::/)'`
- expect: `4 passed` (2 from the previous card, 2 new)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/cascade.rs`
- commit: `feat(cascade): add the draw record a tier produces`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a constructor, an enum of readout kinds or a third column; do not change `CascadeError`
- gpu: none

### 12.5 the tier that draws and asks its judge

- id: FT12.5
- needs: FT12.3, FT12.4
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/generate/decode.rs::LoadedModel::generate_with_serving_config (~line 2107 at a7c08c4c)`: `(&self, prompt: &str, max_tokens: usize, serving_config: ServingConfig) -> Result<(Vec<u32>, String, bool), InteropError>`; a tier's draw function wraps this public call;
  - `interop/serving.rs::ServingConfig (~line 720), impl Default for ServingConfig<'static> (~line 1119)`: public `seed: u64` (~line 829) and `temperature: f32` (~line 796); `ServingConfig` is `Copy`, so `ServingConfig { seed, ..tier.serving }` is the per-draw config;
  - `interop/cascade.rs::Draw, CascadeError` (FT12.3, FT12.4): the draw a drawer returns and the errors `answer` returns;
  - `interop/serving.rs`, the `parallel_sequences` check (~line 1261): draws run one after another because `parallel_sequences != 1` is refused.
- change:
  1. `interop/cascade.rs` (import added at the module top: `crate::serving::ServingConfig`):
     ```rust
     type JudgeFn<'model> = &'model dyn Fn(&[Draw]) -> Result<Option<usize>, CascadeError>;

     pub struct Tier<'model> {
         pub serving: ServingConfig<'model>,
         pub samples: u32,
         pub draw: &'model dyn Fn(&str, usize, ServingConfig<'_>) -> Result<Draw, InteropError>,
         pub judge: JudgeFn<'model>,
     }

     impl Tier<'_> {
         pub fn answer(&self, tier: usize, last: bool, prompt: &str, max_tokens: usize) -> Result<Option<String>, CascadeError>
     }
     ```
     `JudgeFn` is private on purpose: clippy's `type_complexity` rejects the inline closure type and the build denies warnings, and a private alias used in a public field is not a public item. `tier` is the tier's index in its cascade, used only to label errors.
     `answer`: for `index` in `0..self.samples`, build `ServingConfig { seed: self.serving.seed.wrapping_add(u64::from(index)), ..self.serving }`, call `(self.draw)(prompt, max_tokens, config)` and map its error to `CascadeError::Draw { tier, source }` with `?`, pushing each `Draw` on a `Vec`; then `let drawn = draws.len();` and `match (self.judge)(&draws)?`: `Some(chosen)` takes `draws.into_iter().nth(chosen)` and returns `Ok(Some(draw.text))`, or `Err(JudgeIndex { tier, chosen, drawn })` when absent; `None` returns `Err(Unsettled { tier })` when `last`, else `Ok(None)`.
- test: in `cascade.rs` tests, calling `tier.answer(..)` directly (it is synchronous). Fixture: a `Draw` built from a text and fake digit-code-point ids; drawers are local closures `|_: &str, _: usize, serving: ServingConfig<'_>| -> Result<Draw, InteropError>` (annotated per the closure rule, so they are higher-ranked over the config) that bump a `Cell<usize>` counter and push `format!("{serving:?}")` on a `RefCell<Vec<String>>` (a received config cannot be stored as a `ServingConfig`, because its lifetime is the caller's; its derived `Debug` text names every field, so comparing the text compares the whole value); a test's expected record for a config `config` is `format!("{config:?}")`; judges are local closures annotated `|draws: &[Draw]|`; a tier is `Tier { serving: ServingConfig::default(), samples, draw: &drawer, judge: &judge }`. Results are compared after `.expect("..")` because `CascadeError` has no `PartialEq`:
  - `a_single_settling_draw_uses_the_tier_config_unchanged`: `serving = ServingConfig { seed: 7, ..ServingConfig::default() }`, `samples: 1`, judge `|_: &[Draw]| Ok(Some(0))`: `answer(0, true, prompt, 8)` is `Some("4830912".to_owned())`; the recorded configs equal `vec![format!("{serving:?}")]` (whole-value `assert_eq!`);
  - `draws_are_seeded_with_the_tier_seed_plus_the_draw_index`: `samples: 4`, `base = ServingConfig { seed: 10, ..ServingConfig::default() }`: the recorded configs equal `[10, 11, 12, 13].map(|seed| format!("{:?}", ServingConfig { seed, ..base }))` (so only the seed differs between draws); a second tier with seed `u64::MAX` and `samples: 2` records the same form over seeds `[u64::MAX, 0]`;
  - `the_settled_text_is_the_draw_the_judge_chose`: `samples: 3` drawing texts `"4830912"`, `"4830913"`, `"4830914"` in order, judge `Ok(Some(2))`: `Some("4830914".to_owned())`;
  - `a_declining_judge_on_a_tier_that_is_not_last_returns_none`: `samples: 3`, judge `Ok(None)`, `answer(0, false, ..)` is `None` and the drawer ran 3 times;
  - `a_declining_judge_on_the_last_tier_is_an_error_naming_the_tier`: judge `Ok(None)`, `answer(2, true, ..)` matches `Err(CascadeError::Unsettled { tier: 2 })`;
  - `a_failed_draw_names_its_tier_and_stops_drawing`: `samples: 3`, drawer returns `Err(InteropError::UnknownTensor { name: "token_embd.weight".to_owned() })` on its first call: `answer(1, false, ..)` matches `Err(CascadeError::Draw { tier: 1, .. })` and the drawer ran once;
  - `a_judge_choosing_a_missing_draw_is_refused`: `samples: 2`, judge `Ok(Some(5))`: matches `Err(CascadeError::JudgeIndex { tier: 0, chosen: 5, drawn: 2 })`;
  - `a_judge_can_read_the_readout_columns_of_its_draws`: a drawer returning one `Draw` with `margins: vec![0.25, 0.25]`; the judge settles only when the mean of `draws[0].margins` is at least 0.5 (it returns `Ok(None)` for an empty column): `answer(0, false, ..)` is `None`; with a drawer returning `margins: vec![0.75, 0.75]` it is `Some(text)`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_5 cargo nextest run -p proxima-model-interop --features std -E 'test(/cascade::tests::/)'`
- expect: `12 passed` (4 from the previous cards, 8 new)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/cascade.rs`
- commit: `feat(cascade): add the tier that draws and asks its judge`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a judge decision other than the closure the tier carries; do not use `Box`, `Fallback` or `and_then`; do not make `JudgeFn` public; do not annotate a drawer's config parameter `ServingConfig<'static>`; do not touch `InteropError`
- gpu: none

### 12.6 the tier cascade pipe

- id: FT12.6
- needs: FT12.5
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/cascade.rs::Tier::answer, CascadeError` (FT12.5, FT12.3): the per-tier step the loop calls and the error it returns;
  - `interop/generate/residency_caches.rs::impl Pipe for LoadedModel (~line 3146)`: the precedent for a `Pipe` whose `call` is an `async move` block with a synchronous body, and the `In`, `Out`, `Err` associated types;
  - `proxima_primitives::pipe::Pipe` and `proxima_primitives::block_on`: the import and the test driver.
- change:
  1. `interop/cascade.rs` (import added at the module top: `proxima_primitives::pipe::Pipe`):
     ```rust
     pub struct Cascade<'model> {
         tiers: &'model [Tier<'model>],
     }

     impl<'model> Cascade<'model> {
         pub fn new(tiers: &'model [Tier<'model>]) -> Result<Self, CascadeError>
     }
     impl Pipe for Cascade<'_> { type In = (String, usize); type Out = (usize, String); type Err = CascadeError; .. }
     ```
     `new` returns `NoTiers` for an empty slice, else `Ok(Self { tiers })`. `Out` is the answering tier's index and its text.
     `call`: `async move { let (prompt, max_tokens) = input; let last = self.tiers.len().saturating_sub(1); for (index, tier) in self.tiers.iter().enumerate() { if let Some(text) = tier.answer(index, index == last, &prompt, max_tokens)? { return Ok((index, text)); } } Err(CascadeError::NoTiers) }`. The final `Err` is unreachable for a cascade built by `new`; it is a value, never a panic.
     Why a pipe and a slice loop: a caller can hold one value that stands for "ask tier 1, escalate on doubt" and hand it to anything that takes a `Pipe`, which a free function over a slice cannot be; `Tier` is a plain struct because `[Tier]` cannot implement a foreign trait.
- test: in `cascade.rs` tests, drive each case with `proxima_primitives::block_on(cascade.call((prompt, 8)))` over the drawer and judge fixtures of the previous card; two-tier cascades have per-tier draw counters:
  - `a_settling_judge_stops_before_later_tiers`: two tiers, tier 0 judge `Some(0)`: output `(0, "4830912".to_owned())`, counters `[1, 0]`;
  - `a_declining_judge_escalates_to_the_next_tier`: tier 0 `samples: 3`, judge `Ok(None)`; tier 1 `samples: 1`, judge `Ok(Some(0))`, its drawer returns text `"2236418"`: output `(1, "2236418".to_owned())`, counters `[3, 1]`;
  - `a_failed_draw_stops_the_cascade_without_escalating`: tier 0 drawer returns `Err(InteropError::UnknownTensor { name: "token_embd.weight".to_owned() })`: the result matches `Err(CascadeError::Draw { tier: 0, .. })` and tier 1's counter is 0;
  - `a_cascade_without_tiers_is_refused`: `Cascade::new(&[])` matches `Err(CascadeError::NoTiers)`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_6 cargo nextest run -p proxima-model-interop --features std -E 'test(/cascade::tests::/)'`
- expect: `16 passed` (12 from the previous cards, 4 new)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/cascade.rs`
- commit: `feat(cascade): add the tier cascade pipe`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: change `Tier` or its `answer`; do not use `Box`, `Fallback` or `and_then`
- gpu: none

### 12.7 let a request choose the tier the cascade starts at

- id: FT12.7
- needs: FT12.6
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/cascade.rs::Cascade, impl Pipe for Cascade, CascadeError` (FT12.3, FT12.6): the struct, the loop and the error enum this card extends;
  - `interop/cascade.rs::mod tests` (FT12.5, FT12.6): the drawer and judge fixtures reused here.
- change:
  1. `interop/cascade.rs`: a field `route: &'model dyn Fn(&str) -> Result<usize, CascadeError>` on `Cascade`, and the error variants `#[error("router chose tier {chosen} of {tiers}")] RouterTier { chosen: usize, tiers: usize }` and `#[error("router failed: {reason}")] Route { reason: String }`. A router picks the first tier a request may use; here it is a caller-supplied pure function of the prompt.
  2. `interop/cascade.rs`: a private `fn first_tier(_prompt: &str) -> Result<usize, CascadeError> { Ok(0) }`; `Cascade::new` sets `route: &first_tier`, which is the default and reproduces the cascade of the previous card exactly; `pub fn with_router(self, route: &'model dyn Fn(&str) -> Result<usize, CascadeError>) -> Self` replaces it. In `call`, before the loop: `let start = (self.route)(&prompt)?;`, return `Err(CascadeError::RouterTier { chosen: start, tiers: self.tiers.len() })` when `start >= self.tiers.len()`, and iterate `self.tiers.iter().enumerate().skip(start)`; `last` stays the index of the final tier.
- test: in `cascade.rs` tests, three tiers whose judges all return `Ok(Some(0))`, per-tier draw counters:
  - `the_default_router_starts_at_the_first_tier`: counters `[1, 0, 0]`, answering tier 0;
  - `a_router_choice_skips_the_lower_tiers_without_drawing`: router `|_: &str| Ok(2)`: counters `[0, 0, 1]`, answering tier 2;
  - `a_router_choice_past_the_last_tier_is_refused`: router `|_: &str| Ok(3)`: matches `Err(CascadeError::RouterTier { chosen: 3, tiers: 3 })`, counters `[0, 0, 0]`;
  - `a_failing_router_stops_the_request_before_any_draw`: router `|_: &str| Err(CascadeError::Route { reason: "embedding failed".to_owned() })`: matches `Err(CascadeError::Route { .. })`, counters `[0, 0, 0]`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_7 cargo nextest run -p proxima-model-interop --features std -E 'test(/cascade::tests::/)'`
- expect: `20 passed` (16 from the previous cards, 4 new)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/cascade.rs`
- commit: `feat(cascade): let a request choose the tier the cascade starts at`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: embed or cluster anything here (the router is the caller's function); do not change the draw or judge path; do not add a public type alias for the router
- gpu: none

### 12.9 describe one cascade tier as settings

- id: FT12.9
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/prompt_cache_settings.rs::PromptCacheSettings`: the house pattern for an owned settings mirror (`bon::Builder`, `serde`, a lowering method to the `Copy` config) in its own std-gated module; this card copies the pattern without `conflaguration`'s `Settings` derive, because an environment variable cannot express a list of tiers (stated limit; the TOML file and the builder are the two surfaces);
  - `interop/lib.rs::mod prompt_cache_settings (~line 74)`: where `#[cfg(feature = "std")] pub mod cascade_settings;` goes;
  - `interop/cascade.rs::Tier (samples)` (FT12.5): the tier a caller builds from these settings sets `samples` from the settings' `samples`; the settings name no judge, because a judge is the caller's closure and not data;
  - `interop/serving.rs::ServingConfig::temperature (~line 796)`: an `f32`; the settings hold milli-units so the value round-trips through text exactly.
- change:
  1. `interop/cascade_settings.rs` (new, std; imports at the top: `bon::Builder`, `serde::{Deserialize, Serialize}`, `crate::serving::ServingConfig`):
     ```rust
     #[derive(Debug, Clone, PartialEq, Eq, Builder, Deserialize, Serialize)]
     pub struct TierSettings {
         /// path of the GGUF checkpoint this tier loads, for example "/models/gemma-4-e2b-it-Q4_0.gguf"
         pub model: String,
         /// sampling temperature in thousandths, for example 700 for 0.7; 0 samples greedily
         #[serde(default)]
         #[builder(default = 0)]
         pub temperature_milli: u16,
         /// completions drawn before the judge decides, for example 16
         #[serde(default = "one_sample")]
         #[builder(default = 1)]
         pub samples: u32,
     }
     ```
     with the private `fn one_sample() -> u32 { 1 }`; `impl TierSettings`: `pub fn serving<'model>(&self, base: ServingConfig<'model>) -> ServingConfig<'model>` (`ServingConfig { temperature: f32::from(self.temperature_milli) / 1000.0, ..base }`).
  2. `interop/lib.rs`: add `#[cfg(feature = "std")] pub mod cascade_settings;`.
- test: in `cascade_settings.rs` tests (`toml::from_str`, `toml::to_string`; the full tier is the first tier of the example configuration of the design sketch, without its judge sub-table):
  - `a_toml_tier_loads_every_field`: input `model = "/models/gemma-4-e2b-it-Q4_0.gguf"`, `temperature_milli = 700`, `samples = 16` equals `TierSettings { model: "/models/gemma-4-e2b-it-Q4_0.gguf".to_owned(), temperature_milli: 700, samples: 16 }`;
  - `a_toml_tier_with_only_a_model_takes_the_defaults_the_builder_takes`: input `model = "/models/gemma-4-26b-Q4_0.gguf"` has `temperature_milli 0`, `samples 1` and equals `TierSettings::builder().model("/models/gemma-4-26b-Q4_0.gguf".to_owned()).build()`;
  - `a_tier_round_trips_through_toml`: `toml::from_str(&toml::to_string(&tier)?)` equals `tier` for the full tier;
  - `a_tier_lowers_its_temperature_into_the_serving_config`: the full tier's `serving(base)` equals `ServingConfig { temperature: 0.7, ..base }` with `base = ServingConfig::default()`; the model-only tier's `serving(base) == base` (temperature 0 changes nothing).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_9 cargo nextest run -p proxima-model-interop --features std -E 'test(/cascade_settings::tests::/)'`
- expect: `4 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/cascade_settings.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(cascade): describe one cascade tier as toml or a builder`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a field to `ServingConfig`; do not load a model or a file here; do not add `CascadeSettings` yet
- gpu: none

### 12.10 load the cascade tiers from toml or a builder

- id: FT12.10
- needs: FT12.9
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/cascade_settings.rs::TierSettings` (FT12.9): the element of the list this card adds;
  - sketch 13 section 1, the example configuration: the two-tier TOML the tests load, with the gemma4 26B checkpoint named for the second tier.
- change:
  1. `interop/cascade_settings.rs`:
     ```rust
     #[derive(Debug, Clone, PartialEq, Eq, Default, Builder, Deserialize, Serialize)]
     pub struct CascadeSettings {
         /// tiers in escalation order; empty means no cascade, the request takes today's single-model path
         #[serde(default)]
         #[builder(default)]
         pub tiers: Vec<TierSettings>,
     }
     ```
     The type is data only: it names no judge and validates no tuple, because a judge and any rule over `samples` belong to the technique that supplies the judge.
- test: in `cascade_settings.rs` tests (`toml::from_str`, `toml::to_string`). The two-tier TOML is `[[tiers]]` with `model = "/models/gemma-4-e2b-it-Q4_0.gguf"`, `temperature_milli = 700`, `samples = 16`, then `[[tiers]]` with `model = "/models/gemma-4-26b-Q4_0.gguf"`:
  - `a_toml_cascade_loads_two_tiers`: tier 0 equals `TierSettings { model: "/models/gemma-4-e2b-it-Q4_0.gguf".to_owned(), temperature_milli: 700, samples: 16 }`, tier 1 has `temperature_milli 0` and `samples 1`;
  - `the_builder_produces_the_same_settings_as_toml`: `CascadeSettings::builder().tiers(vec![..the same two tiers via TierSettings::builder()..]).build()` equals the parsed value;
  - `settings_round_trip_through_toml`: `toml::from_str(&toml::to_string(&settings)?)` equals `settings`;
  - `an_absent_cascade_is_no_tiers_and_passes`: `toml::from_str::<CascadeSettings>("")` equals `CascadeSettings::default()` and `tiers` is empty.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_10 cargo nextest run -p proxima-model-interop --features std -E 'test(/cascade_settings::tests::/)'`
- expect: `8 passed` (4 from the previous card, 4 new)
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/cascade_settings.rs`
- commit: `feat(cascade): load the cascade tiers from toml or a builder`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a field to `ServingConfig`; do not load a model or a file here; do not derive `conflaguration::Settings` on a type holding a list
- gpu: none

### 12.11 express a conformal cascade through the judge hook

- id: FT12.11
- needs: FT12.6, FT12.9
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/cascade.rs::Tier, Cascade, CascadeError::Judge` (FT12.3, FT12.5, FT12.6): the hook this test drives, from outside the crate, through its public path `proxima_model_interop::cascade`; a vote rule is one `judge` closure over the draws, and a draw count that the rule cannot use is reported as `CascadeError::Judge`;
  - `interop/cascade_settings.rs::TierSettings` (FT12.9): the `samples` field the test copies into each `Tier`;
  - sketch 13 sections 2 and 3: the vote key is the generated token ids, exact equality (answers that differ only in wording count as different answers, a stated limit of the technique); an answer stays in the prediction set when its disagreeing draws (`samples - votes`) are at most `max_disagree`, and the tier settles iff exactly one answer is in the set; the worked example: calibration votes for the true answer `[16, 16, 15, 14, 16, 12, 9, 16, 13]`, disagreements sorted `[0, 0, 0, 0, 1, 2, 3, 4, 7]`, rank `ceil((1000 - 200) * (9 + 1) / 1000) = 8` in integers, 8th smallest is 4, so `max_disagree = 4`;
  - `proxima-model-interop/tests/arch_data_baseline.rs` (the header, `#![cfg(feature = "std")]` and `#![allow(clippy::unwrap_used, clippy::expect_used)]`): the integration test header to copy.
- change:
  1. `proxima-model-interop/tests/cascade_conformal.rs` (new). The technique lives here and nowhere in the library: a local `#[derive(Clone, Copy)] struct Agreement { samples: u32, max_disagree: u32 }`, the vote rule over a tally, the judge that tallies draws and calls it, and the calibration function, about thirty lines in all:
     ```rust
     fn settle(agreement: Agreement, votes: &[u32]) -> Option<usize> {
         let mut inside = votes
             .iter()
             .enumerate()
             .filter(|&(_, &count)| agreement.samples.saturating_sub(count) <= agreement.max_disagree);
         match (inside.next(), inside.next()) {
             (Some((index, _)), None) => Some(index),
             _ => None,
         }
     }

     fn vote_judge(agreement: Agreement) -> impl Fn(&[Draw]) -> Result<Option<usize>, CascadeError> {
         move |draws| {
             if draws.len() != usize::try_from(agreement.samples).unwrap_or(usize::MAX) {
                 return Err(CascadeError::Judge { reason: format!("agreement expects {} draws, the tier drew {}", agreement.samples, draws.len()) });
             }
             let (mut firsts, mut votes): (Vec<usize>, Vec<u32>) = (Vec::new(), Vec::new());
             for (position, draw) in draws.iter().enumerate() {
                 match firsts.iter().position(|&first| draws[first].ids == draw.ids) {
                     Some(group) => votes[group] += 1,
                     None => {
                         firsts.push(position);
                         votes.push(1);
                     }
                 }
             }
             Ok(settle(agreement, &votes).map(|group| firsts[group]))
         }
     }

     fn max_disagree_for(true_answer_votes: &[u32], samples: u32, alpha_milli: u32) -> u32 {
         let mut disagreements: Vec<u32> = true_answer_votes.iter().map(|votes| samples - votes).collect();
         disagreements.sort_unstable();
         let items = u32::try_from(disagreements.len()).expect("calibration set fits u32");
         let rank = ((1000 - alpha_milli) * (items + 1)).div_ceil(1000);
         disagreements[usize::try_from(rank).expect("rank fits usize") - 1]
     }
     ```
     `votes` holds one count per distinct answer in first-seen order; `settle` returns the position of the one answer inside the set, `None` (escalate) when the set has no member or more than one; `saturating_sub` keeps a malformed tally from underflowing; `vote_judge` maps the settled group position to the draw index of that group's first draw, which is the index the cascade's `Tier` expects. The always-settle default of a tier is the closure `|_: &[Draw]| Ok(Some(0))`, not a vote.
     The test also holds a scripted drawer: `fn answer_of_draw(votes: &[u32], index: u32) -> usize` returns the answer whose cumulative vote count first exceeds `index`; a prompt containing `marmot` scripts votes `[13, 2, 1]`, a prompt containing `lynx` scripts `[9, 7]`; the drawer is annotated per the closure rule, reads the draw index from `serving.seed` (base seed 0, through `u32::try_from`), builds `Draw { ids: vec![1000 + answer_id], text: ["4830912", "4830913", "4830914"][answer], .. }` with `answer_id = u32::try_from(answer)`, and counts calls per tier; tier 1 answers `"2236418"`. The cascade is two tiers, each built as `Tier { serving: ServingConfig::default(), samples, draw: &drawer, judge: &judge }` with `samples` read from a `TierSettings::builder()` value: tier 0 `samples: 16` with `judge: &vote_judge(Agreement { samples: 16, max_disagree })` where `max_disagree = max_disagree_for(..)`, tier 1 `samples: 1` with `judge: &|_: &[Draw]| Ok(Some(0))`.
- test: in `cascade_conformal.rs` (`#![cfg(feature = "std")]`), `agreement = Agreement { samples: 16, max_disagree: 4 }` unless stated:
  - `calibrating_nine_items_at_twenty_percent_gives_four_disagreeing_draws`: `max_disagree_for(&[16, 16, 15, 14, 16, 12, 9, 16, 13], 16, 200) == 4`;
  - `the_vote_rule_settles_one_answer_inside_the_set_and_escalates_otherwise`: `settle(agreement, &[13, 2, 1]) == Some(0)`, `settle(agreement, &[12, 4]) == Some(0)`, `settle(agreement, &[16]) == Some(0)`, `settle(agreement, &[1, 15]) == Some(1)`, `settle(agreement, &[20]) == Some(0)` (a count above the sample total does not underflow); `settle(agreement, &[9, 7]) == None`, `settle(agreement, &[8, 8]) == None`, `settle(agreement, &[]) == None`; with `Agreement { samples: 1, max_disagree: 0 }`, `settle(.., &[1]) == Some(0)`;
  - `votes_key_on_ids_and_the_judge_names_the_winners_first_draw`: call `vote_judge(..)(&draws)` directly. `Agreement { samples: 2, max_disagree: 0 }`: two draws with equal text and different ids give `Ok(None)`; two draws with equal ids give `Ok(Some(0))`. `Agreement { samples: 3, max_disagree: 1 }` over draws with ids of answer B, answer A, answer A (votes `[1, 2]`, disagreeing draws `[2, 1]`): `Ok(Some(1))`, the index of the first draw of the winning answer;
  - `a_draw_count_that_differs_from_the_agreement_is_an_error_not_a_settle`: a tier with `samples: 15` and `vote_judge(agreement)` matches `Err(CascadeError::Judge { .. })`;
  - `a_calibrated_cascade_settles_a_confident_prompt_and_escalates_a_split_one`: prompt `marmot` (votes `[13, 2, 1]`) gives `(0, "4830912".to_owned())` with tier counters `[16, 0]`; prompt `lynx` (votes `[9, 7]`) gives `(1, "2236418".to_owned())` with tier counters `[32, 1]` cumulative;
  - `a_looser_set_still_escalates_a_two_way_tie`: `Agreement { samples: 16, max_disagree: 10 }`, scripted votes `[8, 8]` escalate (answering tier 1) and `[13, 3]` settle at tier 0;
  - `a_drawer_that_ignores_the_seed_never_escalates` (the control that must fail the claim): a drawer that ignores the seed and always returns answer 0 makes both prompts settle at tier 0 with tier 1's counter 0, because sixteen identical draws are one answer with zero disagreement.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_11 cargo nextest run -p proxima-model-interop --features std --test cascade_conformal`
- expect: `7 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/tests/cascade_conformal.rs`
- commit: `test(cascade): express a conformal cascade through the judge hook`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add `settle`, `vote_judge`, `Agreement`, the calibration function or any conformal name to a library file; do not add `max_disagree` to `TierSettings`; do not key votes on text; do not load a model
- gpu: none

### 12.12 express a readout threshold judge through the hook

- id: FT12.12
- needs: FT12.6
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/cascade.rs::Draw (logprobs, margins), Tier, Cascade` (FT12.4 to FT12.6): the readout columns a caller's draw function fills and a judge reads;
  - `interop/generate/residency_caches.rs::TokenEvent (~line 3169)`: confirms no probability exists on a token event on main, so the columns are filled by the test;
  - `proxima-model-interop/tests/arch_data_baseline.rs` (the header, `#![cfg(feature = "std")]` and `#![allow(clippy::unwrap_used, clippy::expect_used)]`): the integration test header to copy.
- change:
  1. `proxima-model-interop/tests/cascade_threshold.rs` (new). The technique is a judge function of about fifteen lines:
     ```rust
     fn floor_judge(column: fn(&Draw) -> &[f32], floor: f32) -> impl Fn(&[Draw]) -> Result<Option<usize>, CascadeError> {
         move |draws| {
             let readouts = draws.first().map(column).unwrap_or_default();
             let count = f32::from(u8::try_from(readouts.len()).unwrap_or(u8::MAX));
             let settles = !readouts.is_empty() && readouts.iter().sum::<f32>() / count >= floor;
             Ok(settles.then_some(0))
         }
     }
     ```
     (the settle test is inclusive, so the boundary value settles). A helper `fn answering_tier(first_draw: Draw, judge: &dyn Fn(&[Draw]) -> Result<Option<usize>, CascadeError>) -> usize` builds a two-tier cascade (tier 0 drawer returns `first_draw.clone()` with `samples: 1` and the given judge; tier 1 drawer returns a draw with empty columns, judge `|_| Ok(Some(0))`) and returns the answering tier from `block_on(cascade.call((prompt, 8)))`.
- test: in `cascade_threshold.rs` (`#![cfg(feature = "std")]`):
  - `a_logprob_floor_settles_at_and_above_the_floor`: floor -0.5 over the `logprobs` column: single-token values -0.2 settles (tier 0), -0.9 escalates (tier 1), -0.5 settles (tier 0); two tokens `[-0.25, -0.75]` (mean -0.5, exact in `f32`) settles;
  - `a_margin_floor_escalates_below_the_floor`: floor 0.5 over the `margins` column: 0.8 settles, 0.3 escalates, 0.5 settles;
  - `a_draw_without_readouts_is_declined_by_the_floor_judge`: empty columns escalate to tier 1 under both columns.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_12 cargo nextest run -p proxima-model-interop --features std --test cascade_threshold`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/tests/cascade_threshold.rs`
- commit: `test(cascade): express a readout threshold judge through the hook`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a threshold, a readout enum or a floor field to a library file; do not decode with a model
- gpu: none

### 12.13 express an isotonic error judge through the hook

- id: FT12.13
- needs: FT12.12
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/cascade.rs::Draw (margins), Tier` (FT12.4, FT12.5): the column the judge reads;
  - `proxima-model-interop/tests/cascade_threshold.rs` (FT12.12): the two-tier helper and the integration test header to copy;
  - `proxima-tensor/specs/pipeline-as-data/research.md`, rows E-7 and CC-10: the calibration table is a fitter's output keyed by a serving digest; no digest function exists on main, so the table is a value the judge closure captures.
- change:
  1. `proxima-model-interop/tests/cascade_isotonic.rs` (new). The technique is a table lookup and a judge of about twenty lines:
     ```rust
     fn isotonic_lookup(knots: &[f64], values: &[f64], at: f64) -> f64 {
         let index = knots.partition_point(|knot| *knot <= at);
         values[index.saturating_sub(1)]
     }
     ```
     (indexing is safe because the table has three equal-length entries in this test) and `fn isotonic_judge<'table>(knots: &'table [f64], values: &'table [f64], theta: f64) -> impl Fn(&[Draw]) -> Result<Option<usize>, CascadeError> + 'table`, which computes `u = 1 - mean(margins of the first draw)` in `f64`, settles `Ok(Some(0))` iff `isotonic_lookup(knots, values, u) <= theta`, and returns `Ok(None)` for an empty column. The table is `knots = [0.125, 0.375, 0.875]`, `values = [0.0, 0.5, 1.0]`, `theta = 0.5`. Reuse the two-tier helper shape of the previous card (copied into this file; test files do not share modules).
- test: in `cascade_isotonic.rs` (`#![cfg(feature = "std")]`):
  - `the_isotonic_lookup_steps_at_the_knots`: `at` 0.0, 0.125 and 0.25 give 0.0; 0.375 and 0.5 give 0.5; 0.875 and 9.0 give 1.0;
  - `the_isotonic_judge_settles_at_or_below_theta`: margins `[0.875, 0.625, 0.75]` (mean 0.75, `u` 0.25, value 0.0) settle at tier 0; `[0.5, 0.5, 0.5]` (`u` 0.5, value 0.5, the boundary) settle at tier 0; `[0.25, 0.0, 0.125]` (mean 0.125, `u` 0.875, value 1.0) escalate to tier 1.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_13 cargo nextest run -p proxima-model-interop --features std --test cascade_isotonic`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/tests/cascade_isotonic.rs`
- commit: `test(cascade): express an isotonic error judge through the hook`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a lookup function, a table type or a fitter to a library file; do not invent a digest
- gpu: none

### 12.14 express a classifier judge through the hook

- id: FT12.14
- needs: FT12.12
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/cascade.rs::Tier, CascadeError::Judge` (FT12.5, FT12.3): a judge may call anything it captures and reports its own failure as `CascadeError::Judge { reason }`;
  - `proxima-model-interop/tests/cascade_threshold.rs` (FT12.12): the integration test header and the two-tier helper;
  - the old classifier trace (class probabilities `[0.7, 0.3]` accept, `[0.4, 0.6]` escalate, ties to the lowest class): the model call that produces probabilities is the caller's, a deployment would use the dense gemma4 E2B as the judge model, and the test supplies scripted probabilities (no oracle run backs them; stated in the file preamble).
- change:
  1. `proxima-model-interop/tests/cascade_classifier.rs` (new). The technique is an argmax and a judge of about twenty lines:
     ```rust
     fn argmax_class(probabilities: &[f32]) -> Option<usize> {
         probabilities.iter().enumerate().fold(None, |best, (class, &probability)| match best {
             Some((_, held)) if probability <= held => best,
             _ => Some((class, probability)),
         }).map(|(class, _)| class)
     }
     ```
     (strict greater-than replaces, so the lowest class wins ties) and `fn classifier_judge<'classify>(classify: &'classify dyn Fn(&Draw) -> Vec<f32>, accept_class: usize) -> impl Fn(&[Draw]) -> Result<Option<usize>, CascadeError> + 'classify`, which calls `classify` on `draws[0]` (via `draws.first()`), returns `Err(CascadeError::Judge { reason: "classifier returned no classes".to_owned() })` when `argmax_class` is `None`, and `Ok(Some(0))` iff the class equals `accept_class`, else `Ok(None)`. The scripted `classify` returns `[0.7, 0.3]` for text `"4830912"` and `[0.4, 0.6]` for `"4830913"`.
- test: in `cascade_classifier.rs` (`#![cfg(feature = "std")]`), two-tier cascade, tier 0 drawing the scripted text:
  - `a_classifier_judge_settles_on_the_accept_class_and_escalates_otherwise`: `"4830912"` is answered by tier 0; `"4830913"` by tier 1;
  - `argmax_ties_go_to_the_lowest_class`: `argmax_class(&[0.1, 0.8, 0.1]) == Some(1)`, `argmax_class(&[0.5, 0.5]) == Some(0)`, `argmax_class(&[]) == None`;
  - `a_classifier_with_no_output_is_an_error_not_a_settle`: `classify` returns `vec![]`: `block_on(cascade.call(..))` matches `Err(CascadeError::Judge { .. })` and tier 1's draw counter is 0.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_14 cargo nextest run -p proxima-model-interop --features std --test cascade_classifier`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/tests/cascade_classifier.rs`
- commit: `test(cascade): express a classifier judge through the hook`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add `argmax_class`, a classifier trait or a class field to a library file; do not load a model
- gpu: none

### 12.15 express a nearest centroid router through the hook

- id: FT12.15
- needs: FT12.7, FT12.12
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `interop/cascade.rs::Cascade::with_router, CascadeError::{RouterTier, Route}` (FT12.7): the starting-tier hook;
  - `proxima-model-interop/tests/cascade_threshold.rs` (FT12.12): the integration test header and the scripted-drawer style;
  - the old router trace: flat centroids `[0,0, 10,10, 20,0]` with `dim = 2`, `table = [2, 0, 1]`; embedding `[1,1]` goes to tier 2, `[9,9]` to tier 0, `[19,1]` to tier 1, and `[5,5]` (equidistant from the first two centroids) to tier 2 because the lowest centroid wins ties. The centroids come from the offline calibration fitter, which is out of scope; the test carries the literal table, and the embedding is a scripted lookup keyed by the prompt's animal name (a model's embed call is the caller's).
- change:
  1. `proxima-model-interop/tests/cascade_router.rs` (new). The technique is a function of about ten lines:
     ```rust
     fn nearest_tier(centroids: &[f64], dim: usize, table: &[usize], embedding: &[f64]) -> usize {
         let distance = |row: &[f64]| row.iter().zip(embedding).map(|(left, right)| (left - right).powi(2)).sum::<f64>();
         let mut best = 0;
         for (index, row) in centroids.chunks(dim).enumerate() {
             if distance(row) < distance(&centroids[best * dim..(best + 1) * dim]) {
                 best = index;
             }
         }
         table[best]
     }
     ```
     and `fn embedding_of(prompt: &str) -> Result<Vec<f64>, CascadeError>` mapping `marmot` to `[1.0, 1.0]`, `lynx` to `[9.0, 9.0]`, `heron` to `[19.0, 1.0]`, `tapir` to `[5.0, 5.0]`, and any other prompt to `Err(CascadeError::Route { reason: "unknown prompt".to_owned() })`. The router closure is `|prompt| Ok(nearest_tier(&CENTROIDS, 2, &TABLE, &embedding_of(prompt)?))`. Three tiers, all judges `Ok(Some(0))`, per-tier draw counters.
- test: in `cascade_router.rs` (`#![cfg(feature = "std")]`):
  - `the_nearest_centroid_routes_each_prompt_to_its_table_tier`: `marmot`, `lynx`, `heron`, `tapir` are answered by tiers 2, 0, 1, 2; after each request exactly the routed tier's counter is 1 and the others 0 (counters reset between requests);
  - `an_unembeddable_prompt_stops_before_any_draw`: prompt with `capybara` matches `Err(CascadeError::Route { .. })`, counters `[0, 0, 0]`;
  - `a_table_naming_a_missing_tier_is_refused_at_call_time`: `table = [0, 3]` over two centroids `[0,0, 10,10]` and two tiers, prompt `lynx` (embedding `[9,9]`, centroid 1, tier 3): matches `Err(CascadeError::RouterTier { chosen: 3, tiers: 2 })`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_15 cargo nextest run -p proxima-model-interop --features std --test cascade_router`
- expect: `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/tests/cascade_router.rs`
- commit: `test(cascade): express a nearest centroid router through the hook`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: add a centroid, distance or embedding function to a library file; do not load a model
- gpu: none

### 12.16 hold a one-tier cascade to the direct generate call

- id: FT12.16
- needs: FT12.6
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `interop/generate/prompt_cache_real_model_tests.rs::with_model (~line 54)`: `pub(super) fn with_model<T>(body: impl FnOnce(&LoadedModel<'_>) -> T) -> T` loads gemma4 E2B from `PROXIMA_GEMMA4_E2B_GGUF` (default blob `sha256-3646b4c1...`); reach it as `super::prompt_cache_real_model_tests::with_model`;
  - `interop/generate/prefix_resume_long_prompt_tests.rs::greedy_config (~line 23)`: `pub(super) fn greedy_config() -> ServingConfig<'static>`, the config the other real-model tests use; reach it as `super::prefix_resume_long_prompt_tests::greedy_config`;
  - `interop/generate/mod.rs` (the block of `#[cfg(all(test, feature = "metal", target_os = "macos"))] mod ..._real_model_tests;` lines, last at ~line 253): where the new module line goes;
  - `interop/cascade.rs::Tier, Cascade` (FT12.5, FT12.6): the default one-tier cascade under test, whose judge is the always-settle closure `|_: &[Draw]| Ok(Some(0))`.
- change:
  1. `interop/generate/cascade_real_model_tests.rs` (new): helpers none beyond the test below.
  2. `interop/generate/mod.rs`: add `#[cfg(all(test, feature = "metal", target_os = "macos"))] mod cascade_real_model_tests;` after the last such line.
- test: add `a_one_tier_default_cascade_returns_the_direct_generate_call` in `cascade_real_model_tests.rs` (`#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]`, `#![allow(clippy::expect_used)]` at the top). Inside one `with_model`: `let config = greedy_config(); let prompt = "What is the special magic number for marmot? Answer with the number only.\nAnswer:";` first `let (direct_ids, direct_text, _) = model.generate_with_serving_config(prompt, 8, config).expect("direct generation");`; then a drawer closure `|prompt: &str, max_tokens: usize, serving: ServingConfig<'_>| -> Result<Draw, InteropError> { .. }` whose body calls `model.generate_with_serving_config(prompt, max_tokens, serving)?`, pushes a clone of the returned `ids` on a `RefCell<Vec<Vec<u32>>>` and returns `Draw { ids, text, logprobs: Vec::new(), margins: Vec::new() }` (annotated per the closure rule); a one-tier cascade `Tier { serving: config, samples: 1, draw: &drawer, judge: &|_: &[Draw]| Ok(Some(0)) }`; `proxima_primitives::block_on(cascade.call((prompt.to_owned(), 8)))`. Asserts: the output equals `(0, direct_text)`; the recorded ids equal `vec![direct_ids]` (token for token); the drawer ran once.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_16 cargo nextest run -p proxima-model-interop --features std,metal --run-ignored all -j 1 -E 'test(/cascade_real_model_tests::a_one_tier_default_cascade_returns_the_direct_generate_call/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/src/generate/cascade_real_model_tests.rs`, `proxima-model-interop/src/generate/mod.rs`
- commit: `test(cascade): hold a one-tier cascade to the direct generate call`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message. If the direct and cascade ids differ, stop and report both id lists: that is a finding about the prompt cache path, not a reason to edit the assertion
- do not: touch `cascade.rs` or any non-test source; do not load a second model
- gpu: one run, waiting for a quiet box (the peer-gate check in CARDS "machine safety"); one model load

### 12.17 hold an escalated answer to the direct mixture model call

- id: FT12.17
- needs: FT12.16
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `interop/generate/cascade_real_model_tests.rs` (FT12.16): the file and imports this card extends;
  - `interop/gemma4/bind.rs::real_gemma4_moe_gguf_path (~line 1335 at a7c08c4c)`: reads `PROXIMA_GEMMA4_MOE_GGUF`, default blob `/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129` (gemma4 26B MoE); it is private to that module, so this card carries its own copy of the two lines;
  - `interop/generate/prompt_cache_real_model_tests.rs::with_model (~line 54)`: the open, mmap, `parse_complete`, `LoadedModel::load` sequence to copy for a second path;
  - `interop/cascade.rs::Tier, Cascade` (FT12.5, FT12.6): tier 0 declines through a judge closure, tier 1 settles through the always-settle closure.
- change:
  1. `interop/generate/cascade_real_model_tests.rs`: add a private `fn with_moe_model<T>(body: impl FnOnce(&LoadedModel<'_>) -> T) -> T` that resolves the path from `PROXIMA_GEMMA4_MOE_GGUF` (default blob above), calls `crate::test_support::require_fixture(&path, Some("PROXIMA_GEMMA4_MOE_GGUF"))`, then maps, parses and binds the file exactly as `with_model` does for the E2B blob and calls `body`.
- test: add `an_escalated_request_is_answered_by_the_moe_tier_exactly_as_a_direct_call` (`#[ignore = "depends on host-local gemma4 E2B and gemma4 26B gguf blobs outside this repo, and a real Metal device"]`). Premise check first, inside the test's first line of setup: loading the 26B blob through `LoadedModel::load` must succeed; if it returns an error, stop and report the error text. Body: `with_model(|dense| with_moe_model(|moe| ..))`; `config = greedy_config()`, the prompt of the previous card, 8 tokens; `let (direct_ids, direct_text, _) = moe.generate_with_serving_config(prompt, 8, config).expect(..)`; tier 0 is a drawer over `dense` with `samples: 1` and judge `&|_: &[Draw]| Ok(None)` (always escalates), tier 1 a drawer over `moe` with `samples: 1` and judge `&|_: &[Draw]| Ok(Some(0))`; each drawer is annotated `|prompt: &str, max_tokens: usize, serving: ServingConfig<'_>| -> Result<Draw, InteropError>` like the previous card's, counts its calls in a `Cell<usize>` and records its ids. Asserts: the output equals `(1, direct_text)`; tier 1's recorded ids equal `vec![direct_ids]`; counters equal `[1, 1]` (tier 0 drew once and declined).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_12_17 cargo nextest run -p proxima-model-interop --features std,metal --run-ignored all -j 1 -E 'test(/cascade_real_model_tests::an_escalated_request_is_answered_by_the_moe_tier_exactly_as_a_direct_call/)'`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/src/generate/cascade_real_model_tests.rs`
- commit: `test(cascade): hold an escalated answer to the direct mixture model call`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: touch `cascade.rs` or any non-test source; do not run any other model-loading process alongside this one (two models are resident in this one test; stop and report if the load is refused for memory)
- gpu: one run, waiting for a quiet box; one test, two model loads (gemma4 E2B and gemma4 26B MoE)

## slice exit

1. No `proxima-core` file changes in this slice; `git diff main --stat -- proxima-core` over the slice's commits prints nothing.
2. The cascade half runs `cargo nextest run -p proxima-model-interop --features std -E 'test(/cascade::tests::/)'` (`20 passed`), `-E 'test(/cascade_settings::tests::/)'` (`8 passed`), and the five test files `--test cascade_conformal` (`7 passed`), `--test cascade_threshold` (`3 passed`), `--test cascade_isotonic` (`2 passed`), `--test cascade_classifier` (`3 passed`), `--test cascade_router` (`3 passed`). N == 0 on any line is RED. These counts are derived from reading the cards, not from a run.
3. The two real-model cards run one at a time on a quiet box with `--features std,metal --run-ignored all -j 1`, each `1 passed`.
4. Total new tests: 20 + 8 in interop `src`, 18 in the five integration files, 2 real-model: 48.
5. Spec drift this slice found: SPEC row H17 says the cascade is "configured by serving config `cascade`", but `ServingConfig` is `Copy` and cannot hold a list; the tier list is `CascadeSettings` (FT12.10), outside `ServingConfig`, and no cache key changes. The sketch's `[cascade.tiers.judge]` sub-table is not in the settings: a judge is the caller's closure, so the tier states `samples` and the vote tuple (`max_disagree`) exists only inside the conformal proof test (FT12.11).
6. Everything one slice later (the sans-IO conformance suite) can drive `Cascade` with scripted draws and no model: FT12.5 to FT12.7 already do; a vote rule it needs is a test-local function like FT12.11's.
