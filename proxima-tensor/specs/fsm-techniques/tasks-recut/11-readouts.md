# slice 11 (re-cut): token readouts as an opt-in part of every decode entry (cards FT11.1 - FT11.7, with FT11.2a)

anchors read at main a7c08c4c (full sha a7c08c4c95836f112742f0e30e4ddd672087cda0). Read each with
`git -C /Users/brianbruggeman/repos/slot-0/proxima-windows show main:<path>`; the working tree is not the source.
Paths are relative to the proxima repo root. Every `~line N` is a hint; re-locate by symbol.

Governing spec: `proxima-tensor/specs/pipeline-as-data/SPEC.md` (the sample hook: logits row to entry; the step hook's fork), `research.md` (CC-6, C20), `sketches/13-conformal-cascade.md` (ground facts, gaps G2 and G5). CARDS.md template and rules are binding.

Owner direction this re-cut obeys: the hooks are built so techniques can be vetted later; a technique is never library code. Here the hook is "a decode entry may carry readouts of the logits row it was chosen from" (the judges of a cascade read them). A technique appears only as a test of at most about 40 lines (FT11.4).

Test models (owner, 2026-10-04): gemma4 for dense or MoE, granite for MoE. No qwen of any kind appears in any card, fixture, oracle or validation of this file. Local checkpoints: gemma4 E2B (ollama `gemma4:e2b-it-qat`, dense, static `GEMMA4_E2B` in `proxima-model-interop/tests/arch_data_baseline.rs`), gemma4 26B (ollama `batiai/gemma4-26b:latest`, MoE, static `GEMMA4_26B`), granite (ollama `granite3.1-moe:1b`, MoE, 1.4 GB; main has no granite support, `git grep -n -i granite main -- proxima-model-interop proxima-tensor/src` finds nothing, so it arrives as a profile plus descriptor from the slice-0 granite card and a `Checkpoint` static whose `architecture` is `"granitemoe"`). Oracles are recorded once and replayed; no test queries Ollama or llama-server.

## decisions taken in this cut (assumptions stated, work proceeds under them)

1. Readouts are opted into through the decode section of the serving configuration (`decode.readouts`, default off), not through a new entry point and not through a new `LogitsSink` variant. Reason: `LogitsSink` is `pub(crate)` (`proxima-model-interop/src/generate/pregather.rs::LogitsSink`, ~line 3105), so a sink variant is reachable from outside the crate only through a second public entry that copies `generate_from_ids_with_turn_ends`. Call site both ways: `generate_from_ids_with_readouts(ids, n, &config, cb)` against `generate_from_ids(ids, n, &config_with_readouts_on, cb)`; the second needs no new item. The decode section is built by the step-hook slice (`DecodeConfig` in `serving.rs`, FT7.3; `DecodeSettings` in `serving_settings/decode.rs`, FT7.4, which round-trips through TOML, environment and builder), and FT11.2a adds one field to each.
2. A step without readouts carries `None`, not a pair of NaN floats. `Option<TokenReadout>` keeps "not requested" and "undefined" apart; a NaN pair loses that and forces every reader to test `is_nan`.
3. Pure functions here take no new named type beyond `TokenReadout`: the two fields are the entry field set that the judges of a cascade read (sketch 13, gap G5). A tuple `(f32, f32)` would hide which float is which at every call site.

## old-to-new id map

| old | new | what happened |
|---|---|---|
| FT11.1 | FT11.1 | kept, re-anchored; returns `Option`, `ABSENT` removed; the finite-row refusal now also covers an infinite maximum (the old body returned a number for a row holding `+inf`) |
| FT11.2 | FT11.2 | kept, re-anchored; the event field is `readout: Option<TokenReadout>` instead of two floats |
| none | FT11.2a | new: the `decode.readouts` field on `DecodeConfig` and `DecodeSettings`, split out of FT11.3 so no commit leaves a full `DecodeConfig` literal without the field |
| FT11.3 | FT11.3 | kept and reshaped: the sink variant and the extra entry point are replaced by `decode.readouts` (config both ways) and the loop fill, with the entry test on gemma4 E2B instead of qwen3 |
| FT11.4 | FT11.4 | recut: the library function `generate_samples_from_ids` and the rule `sample_seed` are dropped (see "dropped"); the card is now the technique proof, best-of-N through the seed field and the readouts, on gemma4 E2B |
| FT11.5 | FT11.5, FT11.6, FT11.7 | kept, retargeted: one card per checkpoint (gemma4 E2B, gemma4 26B, granite), each one model load; qwen2, qwen3 and openchat arms removed |
| FT11.6 | FT11.5, FT11.6, FT11.7 | folded: the margin is asserted next to the logprob in each checkpoint's test, from the independent `top[0]` and `top[1]` fields, so no model is loaded twice |

## dropped

- Old FT11.3 `LogitsSink::Readout` and `LoadedModel::generate_from_ids_with_readouts`: the sink variant needs a second public entry that duplicates the turn-ends body; `decode.readouts` reaches the same loop through the existing entry.
- Old FT11.4 `LoadedModel::generate_samples_from_ids` and `sample_seed`: call site with the function is a loop over `generate_from_ids` with `ServingConfig { seed: base.wrapping_add(index), ..config }`; call site without it is the same loop written in the caller. Identical, so no item. The prompt cache already makes the N calls share one prefill (the seed is excluded from the cache key, `generate/prompt_cache_key.rs::CacheKey::of`, ~line 126; the later calls rewind the earlier call's generated rows), which is what FT11.4 measures. The same seed rule already lives inline in the follow-up drafts (`generate/prewarm_follow_up.rs::draft_branch`, ~line 197).
- Old FT11.5 and FT11.6 qwen2, qwen3 and openchat tests, their per-checkpoint tolerances and the old exit count of 9.

## cross-file contract (a card whose premise is false stops and reports)

- Step-hook slice (`tasks-recut/07-rectify.md`): FT7.3 creates `DecodeConfig<'model> { enter: &'model [EnterRule] }` in `proxima-model-interop/src/serving.rs`, `impl Default for DecodeConfig<'static>` and the `ServingConfig.decode` field; FT7.4 creates `DecodeSettings { enter }` in `proxima-model-interop/src/serving_settings/decode.rs` with `as_decode_config` returning `DecodeConfig { enter: &self.enter }`, and exports `DecodeConfig` and `DecodeSettings` from `lib.rs`. There is no `samples` field anywhere: FT2.11 is dropped (`tasks-recut/02-serving-settings.md`, the id map row `FT2.11 | none | dropped` and its "dropped" bullet). Neither `serving_settings/decode.rs` nor `DecodeConfig` exists on main (`git cat-file -e main:proxima-model-interop/src/serving_settings/decode.rs` fails, `git grep -n 'pub struct DecodeConfig' main` finds nothing), so FT11.2a stops and reports if FT7.3 or FT7.4 has not landed. FT11.2a also rewrites the full `DecodeConfig { enter: .. }` literals that FT7.3 (`decode_enter_defaults_to_drafting_only`), FT7.3a (`a_periodic_rule_is_refused_until_the_loop_runs_it`, `an_enter_list_without_drafting_never_verifies_on_real_gemma4_e2b`) and FT7.5 (the `kv_ring.rs` tests) write, so it needs those cards.
- Oracle slice (`tasks-recut/00-oracles-and-worked-examples.md`): the vendored `proxima-model-interop/tests/fixtures/llama-parity/<name>/n_probs.json`, written by FT0.48 (gemma4_e2b), FT0.49 (gemma4_26b) and FT0.6 (granite_moe), each an array of 3 records `{ "prompt_ids": [u32], "generated_ids": [u32], "steps": [ { "id": u32, "logprob": f64, "top": [ { "id": u32, "logprob": f64 }, { "id": u32, "logprob": f64 } ] } ], "llama_commit": "f1ea20621", "command": str }`, captured once with `n_predict 32, temperature 0, n_probs 2, cache_prompt false, return_tokens true` and `post_sampling_probs` unset (raw-softmax log probabilities, which is what the readout computes). Main holds only `gguf_kv.txt`, `llama_ids.json` and `swa_layers.txt` per checkpoint (`git ls-tree main proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/`), so each `n_probs.json` exists only after its FT0 card has landed.
- Oracle slice: `proxima-tensor/specs/fsm-techniques/worked-examples.md` is created by FT0.14 (absent on main until then) and gains its `RESULT readout tolerance:` line from FT0.21; that line lists `tol_logprob` for gemma4_e2b, gemma4_26b and granite_moe (at the cut: `gemma4_e2b=4.673e-03 gemma4_26b=3.719e-03 granite_moe=2.289e-03`, per the FT0.21 validate line in `tasks-recut/00-oracles-and-worked-examples.md`); the margin tolerance equals the logprob tolerance. Each card takes its value from the printed line, never from this file. A checkpoint missing from that line stops its card.
- Oracle slice: the granite `Checkpoint` static (FT0.31; locate with `git grep -n 'architecture: "granitemoe"' -- proxima-model-interop/tests/arch_data_baseline.rs`), pointing at the blob `ollama show --modelfile granite3.1-moe:1b` names on its `FROM` line, and a granite entry in `llama_ids.json` (FT0.41), with granite loading through its family profile plus descriptor (FT0.40) and no granite-specific code path.

## what the readout is (definition, fixed before any code)

For one decode step with logits row `z` (length V, f32) and chosen token `c`:
- `m = max z`; `S = sum_i exp(z_i - m)` (f64 accumulation);
- `logprob = z[c] - m - ln S` (log of the raw softmax probability of the chosen token; no repetition penalty, top-k, top-p or temperature applied);
- `p1 = 1 / S` (top-1 probability), `p2 = exp(top2 - m) / S` (second-highest logit's probability); `top2_margin = p1 - p2 = (1 - exp(top2 - m)) / S`.

Worked values (f64 reference from this definition, rounded to f32; reproduced by a scratch run of the body in FT11.1 before this card was written):

| row `z` | `c` | `logprob` | `top2_margin` | `S` |
|---|---|---|---|---|
| `[2.0, 1.0, 0.0, -1.0]` | 0 | `-0.44018970` | `0.40703144` | `1.553001792775919` |
| `[2.0, 1.0, 0.0, -1.0]` | 1 | `-1.44018970` | `0.40703144` | same |
| `[0.0, 0.0, 0.0, 0.0]` | 2 | `-1.38629436` (`-ln 4`) | `0.0` exactly | 4.0 |
| `[5.0]` | 0 | `0.0` exactly | `1.0` exactly | 1.0 |
| `[1.0, 3.0, 3.0, -2.0]` | 0 | `-2.76177407` | `0.0` exactly (tie) | `2.1420732302356984` |

Direct check of row 1: softmax is `[0.6439142598879724, 0.23688281808991013, 0.08714431874203257, 0.03205860328008499]`; `ln 0.6439142598879724 = -0.4401896985611953`, `p1 - p2 = 0.4070314417980622`.

Allocation: the function takes `&[f32]` and returns an 8-byte `Copy` struct inside an `Option`; the loop keeps `top1`, `top2`, `sum` in stack locals. No `Vec`, `String`, `collect` or `Box`. Proved by FT11.1's `readout_row_allocates_nothing` with the existing counting allocator (`generate/alloc_probe.rs::allocations_during`).

Cost control: the readout is computed only when `decode.readouts` is true (FT11.3). Every other run carries `None`, and the default path pays one bool test per token.

## test-name rule

The slice's model oracle filter is `test(/token_readout_/)`; nextest matches the module-qualified name. Only the three checkpoint oracle tests (FT11.5 to FT11.7) and the best-of-N test (FT11.4) contain `token_readout_`. Unit tests here use the prefixes `readout_row_`, `readout_entry_`, `decode_readouts_` and `token_event_`; the new module is `readout_row`, not `token_readout*`.

Rules: every card uses `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_11_<n>` and removes it when done; logs go under `/Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_<n>/`. Test modules in `src` carry `#[allow(clippy::unwrap_used, clippy::expect_used)]` as their neighbours do (the workspace denies `expect_used`; precedent `generate/residency_caches.rs::decode_control_suppression_tests`).

---

## cards

### 11.1 compute the logprob and top-2 margin of a logits row

- id: FT11.1
- needs: none
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-tokenizer/src/sample.rs::sample_next_token` (~line 486 at a7c08c4c): the chosen token is picked there from the same row; the readout reads the row after the pick and never reuses its filtered candidates;
  - `proxima-model-interop/src/generate/alloc_probe.rs::allocations_during` (~line 41): the zero-allocation assertion (the module is `#[cfg(test)]`, `pub(super)`, reachable from a sibling module of `generate`);
  - `proxima-model-interop/src/generate/mod.rs` (`mod block_bloom;` ~line 221, `pub use residency_caches::*;` ~line 271): where the new module and its re-export go;
  - `proxima-model-interop/src/lib.rs` (`pub use generate::{ .. }` ~line 115): where the public re-export goes.
- change:
  1. `proxima-model-interop/src/generate/readout_row.rs` (new):
     ```rust
     #[derive(Debug, Clone, Copy, PartialEq)]
     pub struct TokenReadout {
         /// ln of the chosen token's raw softmax probability, e.g. -0.4402 for a 64% token.
         pub logprob: f32,
         /// top-1 probability minus top-2 probability of the raw softmax, in 0.0..=1.0, e.g. 0.4070.
         pub top2_margin: f32,
     }
     impl TokenReadout {
         #[must_use]
         pub fn from_row(row: &[f32], chosen: u32) -> Option<Self>
     }
     ```
     One public item: the struct, with its one associated function (no free function, so FT11.2 and FT11.3 consume a single item). Body of `from_row`: `chosen_logit = f64::from(*row.get(chosen as usize)?)`; one pass with locals `top1 = f64::NEG_INFINITY`, `top2 = f64::NEG_INFINITY`, `sum = 0.0_f64`: for each `value` (as `f64`): if `value > top1` then `sum = sum * (top1 - value).exp() + 1.0; top2 = top1; top1 = value`, else `sum += (value - top1).exp(); if value > top2 { top2 = value }`. After the loop: `None` unless `top1.is_finite() && sum.is_finite() && chosen_logit.is_finite()` (a NaN anywhere makes `sum` NaN; a `+inf` leaves `sum` finite but `top1` infinite, which is why `top1` is tested too); else `Some(TokenReadout { logprob: (chosen_logit - top1 - sum.ln()) as f32, top2_margin: ((1.0 - (top2 - top1).exp()) / sum) as f32 })`.
     Why `Option`: an empty row, an out-of-range token or a non-finite row has no readout, and the loop must never print a number for one; a caller that did not ask for readouts carries `None` too. The doc comment names the primitive it reads: the raw softmax of the model's logits row, the same quantity llama-server reports with `post_sampling_probs` unset.
  2. `proxima-model-interop/src/generate/mod.rs`: add `mod readout_row;` after `mod block_bloom;` and `pub use readout_row::TokenReadout;` after the `pub use residency_caches::*;` line; `proxima-model-interop/src/lib.rs`: add `TokenReadout` to the `pub use generate::{ .. }` list.
- test: in `readout_row.rs` (`#[cfg(test)] mod tests`):
  - `readout_row_matches_the_softmax_reference`: the row `[2.0, 1.0, 0.0, -1.0]` with `chosen = 0` and `chosen = 1`: `(actual - expected).abs() < 1e-6` for `logprob` (`-0.44018970`, `-1.44018970`) and for `top2_margin` (`0.40703144` both);
  - `readout_row_uniform_and_single_element_are_exact`: `[0.0; 4]`, `chosen = 2`: `top2_margin == 0.0` (`assert_eq!`), `logprob` within 1e-6 of `-1.38629436`; `[5.0]`, `chosen = 0`: `logprob == 0.0` and `top2_margin == 1.0` exactly;
  - `readout_row_tie_has_zero_margin`: `[1.0, 3.0, 3.0, -2.0]`, `chosen = 0`: `top2_margin == 0.0` exactly, `logprob` within 1e-6 of `-2.76177407`;
  - `readout_row_refuses_what_has_no_readout`: an empty row gives `None`; `TokenReadout::from_row(&[1.0, 2.0], 2)` gives `None`; rows `[1.0, f32::NAN]`, `[1.0, f32::INFINITY]`, `[f32::INFINITY, 1.0]` and `[f32::NEG_INFINITY, 1.0]` (chosen `0` for the first three, `1` for the last) give `None`;
  - `readout_row_allocates_nothing`: `row = (0..4096).map(|index| (index as f32).sin() * 8.0).collect::<Vec<f32>>()` built once outside the probe; `allocations_during(|| for token in 0..1000 { core::hint::black_box(TokenReadout::from_row(&row, token % 4096)); })` is `0`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_11_1 cargo nextest run -p proxima-model-interop --features std -E 'test(/readout_row_/)'`
- expect: `5 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean
- stage: `proxima-model-interop/src/generate/readout_row.rs`, `proxima-model-interop/src/generate/mod.rs`, `proxima-model-interop/src/lib.rs`
- commit: `feat(generate): compute token logprob and margin from a logits row`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: use `sample_next_token`'s filtered candidates; add a free function beside `TokenReadout::from_row` (the struct is the one public item); add a `Vec`; add an `ABSENT` constant or a NaN sentinel; name any test `token_readout_*`
- gpu: none

### 11.2 every token event carries an optional readout

- id: FT11.2
- needs: FT11.1
- budget: 20 min
- crate(s): proxima-model-interop (features: std)
- read first:
  - `proxima-model-interop/src/generate/residency_caches.rs::TokenEvent` (~line 3169 at a7c08c4c): derives `Debug, Clone, Copy, PartialEq, Eq`; a float inside forces dropping `Eq`;
  - `proxima-model-interop/src/generate/residency_caches.rs::decode_until_stop_or_budget` (~line 3433): `produce_next_token: impl FnMut(usize) -> Result<u32, InteropError>` called at ~line 3446; two `TokenEvent { .. }` literals at ~3471 (the step-0 prefill event) and ~3489 (the token event);
  - `proxima-model-interop/src/generate/decode.rs` (closure at ~3596 with `pending: VecDeque<u32>` at ~3537, its exits at ~3605, ~5823, ~5829, ~6246; the second closure at ~6425 with its exit at ~6875);
  - `proxima-model-interop/src/generate/tests_all.rs` (7 callers of `decode_until_stop_or_budget` at ~999, ~1035, ~1952, ~1992, ~2068, ~2116, ~2159; the three `TokenEvent` literals in `decode_metrics_uses_cumulative_token_event_time`, ~lines 719-740) and `residency_caches.rs::decode_until_stop_or_budget_suppresses_control_text_but_keeps_the_id` (~line 3537).
- change:
  1. `proxima-model-interop/src/generate/residency_caches.rs`: `TokenEvent` drops `Eq` from its derive and gains `pub readout: Option<TokenReadout>` (doc: the logprob and top-2 margin of the logits row this token was chosen from, for example `Some(TokenReadout { logprob: -0.44, top2_margin: 0.41 })`; `None` when the run did not ask for readouts); `decode_until_stop_or_budget` takes `produce_next_token: impl FnMut(usize) -> Result<(u32, Option<TokenReadout>), InteropError>`, destructures `let (token_id, readout) = produce_next_token(step)?;` and puts `readout` in both `TokenEvent` literals (import `TokenReadout` from `super::readout_row` if the file's imports do not already provide it).
  2. `proxima-model-interop/src/generate/decode.rs`: in the two closures passed to `decode_until_stop_or_budget`, wrap every returned token as `(token, None)`: `pending` becomes `VecDeque<(u32, Option<TokenReadout>)>` (~3537) and `pending.push_back(extra)` (~5823) pushes `(extra, None)`; the exits `Ok(emitted[0])` (~5829), `Ok(token_id)` (~6246) and `Ok(token_id)` in the second closure (~6875) become `Ok((emitted[0], None))` and `Ok((token_id, None))`; the `pop_front` exit (~3605) already returns the tuple. The compiler lists every missing site (E0308); there are no others. The next card replaces these `None`s.
  3. `proxima-model-interop/src/generate/tests_all.rs`: each of the 7 callers returns `(token, None)` from its closure; the 3 `TokenEvent` literals gain `readout: None`; the same edit in the inline test in `residency_caches.rs` (~3537-3560).
- test: add `token_event_carries_the_readout_the_producer_returned` in the `decode_control_suppression_tests` module of `residency_caches.rs` (not `token_readout_`): a scripted producer returns `[(h_id, Some(TokenReadout { logprob: -0.25, top2_margin: 0.5 })), (i_id, None)]` over a 2-step budget with the stub `Vocab` the neighbouring test builds (`vocab_with_control_marker`); collect `(event.phase, event.readout)` in `on_token`; assert `assert_eq!` on the collected list equals `[(Phase::Prefill { prompt_tokens: 0 }, Some(first)), (Phase::Token, Some(first)), (Phase::Token, None)]` (step 0 fires the prefill event and the token event, both with the first readout).
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_11_2 cargo nextest run -p proxima-model-interop --features std -E 'test(/token_event_carries_the_readout|decode_until_stop|decode_metrics_uses/)'`
- expect: before the edit the same command prints `2 passed` (`decode_until_stop_or_budget_suppresses_control_text_but_keeps_the_id` and `decode_metrics_uses_cumulative_token_event_time`, the two names `cargo nextest list -p proxima-model-interop --features std -E 'test(~decode_until_stop) + test(~decode_metrics_uses)'` prints at a7c08c4c); a different count, or `0`, stops the card and reports the names listed; after the edit it prints `3 passed`
- also green: `cargo clippy -p proxima-model-interop --features std --all-targets` clean and `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean; `cargo check --examples -p proxima-model-interop --features std,metal` clean (examples read `TokenEvent` fields; none constructs one)
- stage: `proxima-model-interop/src/generate/residency_caches.rs`, `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/src/generate/tests_all.rs`
- commit: `feat(generate): carry an optional readout on every token event`
- done when: the expect line printed with `before` in the report, clippy and the examples check clean, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: compute a readout here (every production return is `None` until the next card); add a NaN field; touch the sampler
- gpu: none

### 11.2a add the readouts switch to the decode configuration

- id: FT11.2a
- needs: FT7.3, FT7.3a, FT7.4, FT7.5
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal, conflaguration)
- read first:
  - `proxima-model-interop/src/serving.rs::DecodeConfig` (FT7.3): the `Copy` decode section whose `Default` is the incumbent behaviour, and its full literals in `serving.rs` `tests` (`decode_enter_defaults_to_drafting_only`, FT7.3, and `a_periodic_rule_is_refused_until_the_loop_runs_it`, FT7.3a), which name every field and carry no `..` base;
  - `proxima-model-interop/src/serving_settings/decode.rs::DecodeSettings::as_decode_config` (FT7.4) and `serving_settings.rs::round_trip::assert_three_ways` (FT2.9's helper, defined by FT2.27): the settings section and the TOML, environment and builder comparison;
  - the attribute shapes for a bool setting, both on main: `examples/gguf_generate.rs` (`#[setting(default = false)]`, ~line 87 at a7c08c4c) and `src/cassette_config.rs` (`#[builder(default = false)]`, ~line 246 at a7c08c4c);
  - the other full `DecodeConfig { enter: .. }` literals: `proxima-model-interop/src/generate/kv_ring.rs` `tests` (the `step_row_limit` and `ring_slack` tests, FT7.5) and `generate/speculative_default_on_tests.rs::an_enter_list_without_drafting_never_verifies_on_real_gemma4_e2b` (FT7.3a).
- change:
  1. `proxima-model-interop/src/serving.rs`: `DecodeConfig` gains `pub readouts: bool` (doc: "fill each token event's logprob and top-2 margin from the raw logits row; reading the row costs a host copy per token and keeps the greedy pick off the device"); `impl Default for DecodeConfig<'static>` adds `readouts: false` beside `enter`; `ServingConfig::default().decode` stays `DecodeConfig::default()`. The two full literals in `serving.rs` `tests` become `DecodeConfig { enter: <same list>, ..DecodeConfig::default() }`.
  2. `proxima-model-interop/src/serving_settings/decode.rs`: `DecodeSettings` gains `readouts: bool` with `#[setting(default = false)]` and `#[builder(default = false)]` (doc: the sentence above); `as_decode_config` becomes `DecodeConfig { enter: &self.enter, readouts: self.readouts }`. Environment key `PROXIMA_SERVING_DECODE_READOUTS`.
  3. `proxima-model-interop/src/generate/kv_ring.rs` and `proxima-model-interop/src/generate/speculative_default_on_tests.rs`: every full `DecodeConfig { enter: .. }` literal in their tests becomes `DecodeConfig { enter: <same list>, ..DecodeConfig::default() }`. Nothing else in these two files changes. The compiler lists every missed literal (E0063); if it lists one in a file outside the stage list, stop and report the path.
- test: two tests, one command:
  - `decode_readouts_default_to_off` in `serving.rs` `tests`: `DecodeConfig::default().readouts == false` and `ServingConfig::default().decode.readouts == false`;
  - `serving_settings_decode_readouts_variants` in `serving_settings/decode.rs` `tests`: `assert_three_ways` with TOML `[decode]` `readouts = true`, environment `PROXIMA_SERVING_DECODE_READOUTS=true`, builder `ServingSettings::builder().decode(DecodeSettings::builder().readouts(true).build())`; then `as_serving_config(&[]).decode.readouts == true`, and `ServingSettings::default().as_serving_config(&[]).decode.readouts == false`. Sad: `readouts = "yes"` in TOML is `Err`.
- validate: `CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_11_2a cargo nextest run -p proxima-model-interop --features std,metal,conflaguration -E 'test(/decode_readouts_default_to_off|serving_settings_decode_readouts_variants/)'`
- expect: `2 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal,conflaguration --all-targets` clean; `cargo check -p proxima-model-interop --no-default-features` clean; `cargo check -p proxima-model-interop --features std,metal --all-targets` clean (compiles every `DecodeConfig` and `ServingConfig` literal in examples and benches); two counts recorded BEFORE editing and compared after, both run as `cargo nextest run -p proxima-model-interop --features std,metal,conflaguration -E '<filter>'`: `N`, the count for `test(/serving_settings_|serving::tests::/)`, prints `N + 2 passed` after (the two new tests match it: `decode_readouts_default_to_off` through `serving::tests::`, `serving_settings_decode_readouts_variants` through `serving_settings_`); `K`, the count for `test(/kv_ring::tests::/)`, prints `K passed` after (this run builds the `kv_ring.rs` and `speculative_default_on_tests.rs` test modules, so a missed literal fails to compile); `N` or `K` of 0 is RED
- stage: `proxima-model-interop/src/serving.rs`, `proxima-model-interop/src/serving_settings/decode.rs`, `proxima-model-interop/src/generate/kv_ring.rs`, `proxima-model-interop/src/generate/speculative_default_on_tests.rs` (`kv_ring.rs` and the test-only file `speculative_default_on_tests.rs` take one-line literal edits only: three source files plus one test file)
- commit: `feat(interop): add a readouts switch to the decode config`
- done when: the expect line printed, clippy and both checks clean, `N + 2` and `K` printed, `git diff --cached --stat` equals the stage list, and the commit landed with that message
- do not: read the field anywhere in `generate/decode.rs` (the next card does); add a `LogitsSink` variant or a new public entry point; touch the prompt-cache key or the resident plan identity; add any key other than `readouts`
- gpu: none (the filters above load no model)

### 11.3 fill the readouts when the decode configuration asks

- id: FT11.3
- needs: FT11.2, FT11.2a
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/src/serving.rs::DecodeConfig` (FT11.2a): the `Copy` decode section whose `readouts: bool` field this card reads;
  - `proxima-model-interop/src/generate/decode.rs` (at a7c08c4c): `greedy_on_device` (~line 3581) requires `matches!(logits_sink, LogitsSink::Discard)` and plain greedy, because the device argmax never materialises the row; the three selection sites are the verify rows (`select_decoded_token(_step + row_index, row, ..)`, ~line 5758), the single-row pick (`token_id = match greedy_device_token { .. }`, ~line 6086) and the sequential loop (`sample_next_token(last_position, ..)`, ~line 6803);
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity` (~line 668) and `::llama_cases` (~line 617): the load-then-generate pattern and the recorded ids this card compares against;
  - `proxima-model-interop/src/generate/prompt_cache_key.rs::CacheKey::of` (~line 82): readouts change no row a cache holds, so this destructure stays untouched.
- change:
  1. `proxima-model-interop/src/generate/decode.rs`: at the top of each of the two functions that call `decode_until_stop_or_budget` add `let readouts_enabled = serving_config.decode.readouts;`. In the closure at ~3596: `greedy_on_device` gains `&& !readouts_enabled`; in the verify loop `emitted` becomes `Vec<(u32, Option<TokenReadout>)>`, each pick pushes `(selected, readouts_enabled.then(|| TokenReadout::from_row(row, selected)).flatten())` (the accept test `speculative_draft.get(row_index) != Some(&selected)` still compares tokens; `emitted.len()`, `emitted[1..]` pushes the tuples to `pending`, the next `next_ids` takes `emitted[emitted.len() - 1].0`, `Ok(emitted[0])` returns the tuple); the single-row pick becomes `(token_id, readout) = match greedy_device_token { Some(picked) => (picked, None), None => { let picked = select_decoded_token(..)?; (picked, readouts_enabled.then(|| TokenReadout::from_row(last_position, picked)).flatten()) } }` and the exit returns `Ok((token_id, readout))`. In the closure at ~6425 the same expression follows the `sample_next_token` pick (~6803) and the exit (~6875) returns `Ok((token_id, readout))`. `TokenReadout` reaches `decode.rs` through its `use super::*` and the `pub use` added by FT11.1. If a path that produces a token has no host row (a `Phase::Token` event under `readouts: true` carries `None`), stop and report which path produced it.
  2. `proxima-model-interop/tests/arch_data_baseline.rs`: add `DecodeConfig` and `Phase` to the `proxima_model_interop` import list and the test below.
- test: one test:
  - `readout_entry_events_carry_finite_readouts_gemma4_e2b` in `tests/arch_data_baseline.rs` (name has no `token_readout_` prefix): load `GEMMA4_E2B` once, take the first record of its `llama_cases`, `config_off = ServingConfig { prompt_cache: PromptCacheConfig::off(), ..ServingConfig::default() }`, `config_on = ServingConfig { decode: DecodeConfig { readouts: true, ..DecodeConfig::default() }, ..config_off }`, `generate_from_ids(&case.prompt_ids, LLAMA_GENERATED_TOKENS, .., ..)` collecting `(event.token_id, event.readout)` of every `Phase::Token` event. With `config_off`: every readout is `None` and the ids equal llama's recorded `generated_ids` over `min(len)` (`first_divergence(..) == None`, compared length greater than 0). With `config_on`: the ids equal the same recorded ids over the same length; the number of token events equals the number of generated ids and is `>= 1` (zero events is RED); every readout is `Some(readout)` with `readout.logprob.is_finite() && readout.logprob <= 0.0` and `0.0 <= readout.top2_margin && readout.top2_margin <= 1.0`. One model load serves both runs.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_3 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_11_3 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/readout_entry_events_carry_finite_readouts_gemma4_e2b/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_3/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean; `cargo check -p proxima-model-interop --no-default-features` clean; `cargo check -p proxima-model-interop --features std,metal --all-targets` clean (compiles every `DecodeConfig` and `ServingConfig` literal in examples and benches); the model-free tests that existed before still pass: record `K`, the count printed by `cargo nextest run -p proxima-model-interop --features std,metal -E 'test(/decode_until_stop|decode_metrics_uses|token_event_carries_the_readout|kv_ring::tests::|serving::tests::/)'` before editing (`K` of 0 is RED), and `K passed` after (no test is added to those modules)
- stage: `proxima-model-interop/src/generate/decode.rs`, `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `feat(generate): fill token readouts when the decode config asks`
- done when: the expect line printed, clippy and both checks clean, `K` printed before and after, `git diff --cached --stat` equals the stage list (one source file plus the test file), and the commit landed with that message
- do not: compute a readout when `decode.readouts` is false; change token selection, the order in which the random generator is consumed, or the `greedy_on_device` condition for any run with `readouts` false; add a `LogitsSink` variant or a new public entry point; touch the prompt-cache key or the resident plan identity
- gpu: one run (`-j 1`, one model-loading process), waiting for a quiet box (the peer-gate check above)

### 11.4 the technique through the hooks: best-of-N by logprob from seeded calls

- id: FT11.4
- needs: FT11.3
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/src/serving.rs::ServingConfig` (`seed: u64` ~line 829, `temperature: f32` ~line 796): `Copy`, so one sample's configuration is `ServingConfig { seed: base.seed.wrapping_add(index), ..*base }`;
  - `proxima-model-interop/src/generate/decode.rs::generate_from_ids` (~line 2277): the id-in entry each sample calls, going through the prompt cache;
  - `proxima-model-interop/src/generate/prompt_cache.rs::LoadedModel::last_prompt_cache_report` (~line 1020, public) and `CacheReport` (~line 161: `reused_tokens`, `prefilled_tokens`, `miss`);
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity` (~line 668): the loader, and the test from FT11.3 (same file) for the readouts configuration.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs` only; nothing is added to the library (see "dropped"). Add two private test helpers of about 20 lines in total, which are the technique and live nowhere else:
     - `struct Sample { ids: Vec<u32>, mean_logprob: f32 }` and `fn draw_samples(model: &LoadedModel<'_>, prompt_ids: &[u32], base: &ServingConfig<'_>, count: usize) -> Vec<Sample>`: for `index` in `0..count`, `config = ServingConfig { seed: base.seed.wrapping_add(index as u64), ..*base }`; one `generate_from_ids(prompt_ids, 16, &config, ..)` whose callback pushes `event.readout.expect("readouts are on").logprob` of each `Phase::Token` event; `mean_logprob` is their sum divided by their count (assert the count is at least 1);
     - `fn best_sample(samples: &[Sample]) -> usize`: the index of the largest `mean_logprob` (`max_by` with `f32::total_cmp`).
- test: add `token_readout_best_of_samples_gemma4_e2b` (the one test in this card), loading `GEMMA4_E2B` once and taking the first record of `llama_cases`; `sampled = ServingConfig { prompt_cache: PromptCacheConfig::off(), temperature: 0.7, seed: 7, decode: DecodeConfig { readouts: true, ..DecodeConfig::default() }, ..ServingConfig::default() }`; `count = 4`:
  - reproducible: two `draw_samples` calls return the same `ids` for all 4 samples (`assert_eq!`) and `mean_logprob` values within 1e-4 of each other;
  - the seed rule: sample 0's ids equal one plain `generate_from_ids` with the same configuration but `decode: DecodeConfig::default()` (seed 7), and sample 3's ids equal one plain call with seed 10; this also shows readouts change no token;
  - the selection: every `mean_logprob` is finite and `<= 0.0`; `samples[best_sample(&samples)].mean_logprob >= samples[i].mean_logprob` for every `i`;
  - the fork is cheap through the existing machinery: with `PromptCacheConfig { byte_budget: 1 << 30, ..PromptCacheConfig::off() }` in place of the off configuration, draw the same 4 samples; each sample's ids equal the cache-off sample's ids with the same index, and after the draw `model.last_prompt_cache_report()` is `Some(report)` with `report.miss == None` and `report.reused_tokens > 0`;
  - control that must fail to distinguish the samples: with `temperature: 0.0` (greedy) the 4 samples have identical ids (the seed does nothing there, so best-of-N over a greedy run measures nothing); it does not assert that the sampled ids differ from each other (a model can repeat).
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_4 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_11_4 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/token_readout_best_of_samples_gemma4_e2b/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_4/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(generate): draw seeded samples and keep the best by logprob`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`, and the commit landed with that message
- do not: add a function, field or type to `src`; share one random generator across samples; query Ollama or llama-server; load a second model
- gpu: one run (`-j 1`), waiting for a quiet box

### 11.5 hold gemma4 E2B readouts to the recorded llama-server probabilities

- id: FT11.5
- needs: FT11.3, FT0.48 (`n_probs.json` for gemma4_e2b), FT0.21 (the readout tolerance line; FT0.14 creates `worked-examples.md`)
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::llama_parity` (~line 668), `::llama_cases` (~line 617), `::ids_of` (~line 649), `::first_divergence` (~line 661) and `LLAMA_GENERATED_TOKENS` (~line 609): the loader and the first-divergence policy this card reuses;
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b/n_probs.json` (vendored by the oracle slice; shape in the cross-file contract above): `steps[i].logprob` is llama's log probability of generated token `i`, `steps[i].top[0]` and `top[1]` its two highest;
  - `proxima-tensor/specs/fsm-techniques/worked-examples.md`, the `RESULT readout tolerance:` line: take the value printed for `gemma4_e2b` (`grep -o 'gemma4_e2b=[0-9.e-]*'` on that line; it must read `4.673e-03`, the value FT0.21 derives, and any other value stops the card and reports), margin tolerance equal;
  - the test from FT11.3 (`readout_entry_events_carry_finite_readouts_gemma4_e2b`, same file): the readouts configuration and event collection.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add `struct NProbsStep { logprob: f64, top_logprobs: [f64; 2] }` and `struct NProbsCase { prompt_ids: Vec<u32>, generated_ids: Vec<u32>, steps: Vec<NProbsStep> }`; `fn n_probs_cases(checkpoint: &Checkpoint) -> Vec<NProbsCase>` reading `tests/fixtures/llama-parity/<name>/n_probs.json` through `serde_json::Value` as `llama_cases` does (a missing file panics with its path and never skips; zero records panics); `const READOUT_TOLERANCE_GEMMA4_E2B: f32 = 4.673e-3;`; and `fn readout_oracle(checkpoint: &Checkpoint, tolerance: f32)`: load the model as `llama_parity` does with `ServingConfig { prompt_cache: PromptCacheConfig::off(), decode: DecodeConfig { readouts: true, ..DecodeConfig::default() }, ..ServingConfig::default() }`; per record call `generate_from_ids(&record.prompt_ids, LLAMA_GENERATED_TOKENS, ..)` collecting `(token_id, readout)` of `Phase::Token` events; compare over the leading steps while the token id equals `record.generated_ids[index]` (stop at the first differing id: the first-divergence policy); for each compared step assert the readout is `Some`, `(readout.logprob - step.logprob as f32).abs() <= tolerance`, and `(readout.top2_margin - llama_margin).abs() <= tolerance` with `llama_margin = (step.top_logprobs[0].exp() - step.top_logprobs[1].exp()) as f32` (taken from the two top fields, never from `step.logprob`, so the two checks are independent evidence). Failures are collected and reported with checkpoint, record, step, token id, proxima value and llama value. Guards: exactly 3 records ran, at least 1 step was compared per record (zero compared is RED).
  2. Add the test `token_readout_oracle_gemma4_e2b` calling `readout_oracle(&GEMMA4_E2B, READOUT_TOLERANCE_GEMMA4_E2B)`.
- test: `token_readout_oracle_gemma4_e2b`: per compared step the logprob and the top-2 margin each within `4.673e-3` of llama-server's recorded values, over 3 records, at least 1 compared step per record.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_5 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_11_5 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/token_readout_oracle_gemma4_e2b/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_5/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(generate): hold gemma4 e2b readouts to llama-server n_probs`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`, and the commit landed with that message
- do not: compare against proxima's own earlier output; edit a fixture; skip when the fixture is missing (the test fails); loosen the tolerance (a failure is reported with the first divergent step); derive the margin from `step.logprob`
- gpu: one run (`-j 1`), waiting for a quiet box

### 11.6 hold gemma4 26B (MoE) readouts to the recorded llama-server probabilities

- id: FT11.6
- needs: FT11.5, FT0.49 (`n_probs.json` for gemma4_26b), FT0.21 (the readout tolerance line)
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::readout_oracle` (FT11.5) and the static `GEMMA4_26B` (~line 48 at a7c08c4c; `batiai/gemma4-26b:latest`, expert_count 128, 13.3 GB blob);
  - `proxima-model-interop/tests/fixtures/llama-parity/gemma4_26b/n_probs.json` (vendored by the oracle slice);
  - the `RESULT readout tolerance:` line of `proxima-tensor/specs/fsm-techniques/worked-examples.md`: take the value printed for `gemma4_26b` (`grep -o 'gemma4_26b=[0-9.e-]*'` on that line); if the line has no entry for it, stop and report.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add `const READOUT_TOLERANCE_GEMMA4_26B: f32` with the value read above, and the test `token_readout_oracle_gemma4_26b` calling `readout_oracle(&GEMMA4_26B, READOUT_TOLERANCE_GEMMA4_26B)`. Nothing else.
- test: `token_readout_oracle_gemma4_26b`, the assertions of FT11.5 on the MoE checkpoint: logprob and top-2 margin each within the tolerance, 3 records, at least 1 compared step per record.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_6 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_11_6 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/token_readout_oracle_gemma4_26b/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_6/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(generate): hold gemma4 moe readouts to llama-server n_probs`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`, and the commit landed with that message
- do not: run another model-loading process (this blob is 13.3 GB); edit a fixture; change `readout_oracle`; loosen the tolerance
- gpu: one run (`-j 1`, one model-loading process), waiting for a quiet box

### 11.7 hold granite (MoE) readouts to the recorded llama-server probabilities

- id: FT11.7
- needs: FT11.5, FT0.31 (the granite `Checkpoint` static), FT0.40 (profile plus descriptor), FT0.41 (recorded ids), FT0.6 (`n_probs.json` for granite_moe), FT0.21 (the readout tolerance line)
- budget: 20 min
- crate(s): proxima-model-interop (features: std, metal)
- read first:
  - `proxima-model-interop/tests/arch_data_baseline.rs::readout_oracle` (FT11.5);
  - the `Checkpoint` static whose `architecture` is `"granitemoe"` (locate with `git grep -n 'architecture: "granitemoe"' -- proxima-model-interop/tests/arch_data_baseline.rs`; its blob is the `FROM` line of `ollama show --modelfile granite3.1-moe:1b`, 1.4 GB);
  - `proxima-model-interop/tests/fixtures/llama-parity/<that static's name>/n_probs.json` (vendored by the oracle slice);
  - the `RESULT readout tolerance:` line of `proxima-tensor/specs/fsm-techniques/worked-examples.md`: the value printed for that static's name; if the line has no entry for it, stop and report.
- change:
  1. `proxima-model-interop/tests/arch_data_baseline.rs`: add `const READOUT_TOLERANCE_GRANITE_MOE: f32` with the value read above, and the test `token_readout_oracle_granite_moe` calling `readout_oracle(&<the granitemoe static>, READOUT_TOLERANCE_GRANITE_MOE)`. Nothing else. If the grep finds no such static, or the fixture file is absent, stop and report: the granite profile card is not done.
- test: `token_readout_oracle_granite_moe`, the assertions of FT11.5 on the granite checkpoint. No granite-specific code path exists in the library; the model reaches the generic readout fill through its profile and descriptor. If a `Phase::Token` event carries `None` under `decode.readouts`, the failing path is reported (FT11.3's stop rule), not worked around here.
- validate: `ps -axo comm | grep -E "decode_gbps|census|llama-server|decode_arms|norm_variant"` (must print nothing), then `mkdir -p /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_7 && CARGO_TARGET_DIR=/private/tmp/cargo_target_ft_11_7 cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/token_readout_oracle_granite_moe/)' 2>&1 | tee /Users/brianbruggeman/repos/slot-0/.long_ctx_backups/fsm/ft_11_7/run.log`
- expect: `1 passed`
- also green: `cargo clippy -p proxima-model-interop --features std,metal --all-targets` clean
- stage: `proxima-model-interop/tests/arch_data_baseline.rs`
- commit: `test(generate): hold granite moe readouts to llama-server n_probs`
- done when: the expect line printed, clippy clean, `git diff --cached --stat` touches only `tests/arch_data_baseline.rs`, and the commit landed with that message
- do not: add a granite branch anywhere in `src`; edit a fixture; change `readout_oracle`; loosen the tolerance
- gpu: one run (`-j 1`), waiting for a quiet box

---

## spec drift

1. The old slice named 9 oracle tests (4 logprob, 4 margin, 1 samples). The count is now 4: three checkpoint oracle tests (each asserts logprob and margin) and the best-of-N test. The pipeline-as-data and fsm-techniques SPEC and TASKS lines that quote 9 describe the earlier cut.
2. The old `decode.samples` completions rule ("N completions seeded seed plus index") has no library implementation in this cut: the cascade pipe (sketch 13, gap G2) and FT11.4 both draw N samples as N seeded calls. No `samples` field exists in `DecodeConfig` or `DecodeSettings` (FT2.11 is dropped and FT7.4 forbids it), so there is nothing for the decode loop to read or refuse.
3. On the plain greedy Metal path the argmax runs on the device (`greedy_on_device`, `decode.rs` ~line 3581) and no host row exists. `decode.readouts` forces the host row by joining that gate; the cost is the host logits fetch per token, for readout runs only. Runs without `readouts` keep the gate unchanged, which FT11.3's entry test checks against llama's recorded ids.
4. `TokenEvent` loses `Eq` (a float inside `Option<TokenReadout>`). `git grep -n "TokenEvent" main` shows only field reads, closures and the five literals FT11.2 edits.

## slice exit

Commands, each with its count (N == 0 is RED):
- model-free: `cargo nextest run -p proxima-model-interop --features std -E 'test(/readout_row_|token_event_carries_the_readout|decode_readouts_default_to_off/)'` prints `7 passed` (5 + 1 + 1);
- settings: `cargo nextest run -p proxima-model-interop --features std,conflaguration -E 'test(/serving_settings_decode_readouts_variants/)'` prints `1 passed`;
- model, one process at a time: `cargo nextest run -p proxima-model-interop --features std,metal -j 1 -E 'test(/token_readout_/)'` prints `4 passed` (gemma4 E2B oracle, gemma4 26B oracle, granite oracle, best-of-N), and `-E 'test(/readout_entry_/)'` prints `1 passed`.
