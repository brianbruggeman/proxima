# sketch 6: min-p plus repetition-penalty sampler (H14 sample; N1 shape logits)

status: paper test. Read at main 4b4be6cf (`git show main:<path>`). Nothing was built or run; every
"hand-derived" value is derived from cited lines, not executed. The token ids are real (llama oracle
fixture); the logit values are illustrative, because real logits cannot be had without running a model.
Paths relative to the proxima repo root; `interop` = `proxima-model-interop/src`,
`tok` = `proxima-tokenizer/src`.

## shape chosen, and the contested decision

Shape: one config section, `[sampling]`, holding an ORDERED chain of stages. Each stage is a name plus its
numbers; a stage whose numbers are neutral is absent from the list. The default list is today's chain, in
today's order: penalties, top_k, top_p, min_p, temperature, then the weighted draw
(`tok/sample.rs:19-21`, `sample_general` body :457-470).

Contested decision: is the chain a list of stages, or the flat seven-number struct it is today
(`SamplingConfig`, `tok/sample.rs:96-123`)? Chosen: a list, because min-p and the repetition penalty are
both already implemented and fixed-order, so the configuration this task asks for is ALREADY expressible as
numbers (section 2), and the only thing a list adds is ORDER. Order is the one thing the flat struct cannot
say, and llama.cpp, the incumbent, can: `--samplers` (`common/arg.cpp:1982`, llama.cpp checkout at commit
f1ea20621, file read directly) and `--sampler-seq` (:1998). So the catalog claim "sampler chain order is
fixed" (SPEC "how much composes today") is a real gap against the oracle, not an extension.

No new pipe type. Second gate, call site both ways: a `Stage: Pipe<In = Candidates, Out = Candidates>`
would replace `apply_min_p(&mut candidates, config.min_p)` (`tok/sample.rs:460`) with
`min_p_stage.call(candidates)`. Same operands, same effect; and `Pipe::call(&self)` returns a future where
the sampler is a synchronous reduction (`tok/sample.rs:3-9`, the module's own long-standing convention:
"a sampler is a pure reduction ... a function, not a form"). So the stage list is a closed enum of
functions, not pipes.

## 1. ground: what a user writes today

There is no library TOML for sampling. `ServingConfig` derives `Debug, Clone, Copy, PartialEq`
(`interop/serving.rs:719`) and carries the knobs as loose fields: `temperature` (:796), `top_k` (:799),
`top_p` (:801), `min_p` (:807), `repeat_last_n` (:812), `repeat_penalty` (:815), `frequency_penalty`
(:818), `presence_penalty` (:821), `seed` (:829). Defaults are greedy and neutral
(`serving.rs:1153-1161`: `temperature 0.0, top_k 0, top_p 1.0, min_p 0.0, repeat_last_n 64,
repeat_penalty 1.0`). The only env/TOML surface for them is example-local: `GenerateConfig`
(`examples/gguf_generate.rs:72-116`), prefix `PROXIMA`, with DIFFERENT defaults (`temperature 0.8,
top_k 40, top_p 0.95, min_p 0.15, repeat_penalty 1.1`, :99-110). So "min-p plus repetition penalty" is today
`PROXIMA_MIN_P=0.05 PROXIMA_REPEAT_PENALTY=1.3 PROXIMA_TEMPERATURE=0.7` on one example binary.

The decode path (`interop/generate/decode.rs`): `sample_config` is built once from `serving_config`
(:3030-3038), `rng = fastrand::Rng::with_seed(serving_config.seed)` (:3039), `repeat_window =
repeat_last_n.max(0)` (:3029). Each step calls `select_decoded_token` (:1319-1337), which slices the last
`repeat_window` ids of `token_history` (prompt included) and calls `sample_next_token`
(`tok/sample.rs:486-499`): empty logits give `None`; `temperature <= 0` and `greedy_fast_path_is_safe`
take the allocation-free argmax; anything else takes `sample_general`. Validation: `min_p` must be in
`0.0..=1.0` (`serving.rs:1356-1365`); `repeat_last_n < 0` (llama's "-1 = context size") is REJECTED
(`serving.rs:1367-1375`).

Incumbent default differs from ours and is deliberate: llama.cpp's own `min_p` default is `0.05`
(`common/common.h:234`); ours is `0.0` because the owner's invocation is `--min-p 0`
(`serving.rs:1120-1122`). The config below keeps `0.0`.

## 2. the config a user would write

Real data: the qwen3 8B oracle fixture
(`proxima-model-interop/tests/fixtures/llama-parity/qwen3/llama_ids.json`, llama commit f1ea20621) records
the prompt `"The capital of France is"` as ids `[785, 6722, 315, 9625, 374]` and the greedy continuation
`[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689, 374, 24081, 13, 576, 6722, 315,
9856, 374, 19846, 13, ...]`: the same frame `[576, 6722, 315, X, 374, Y, 13]` recurs. Greedy output on a
base-style prompt loops; that is the data a repetition penalty exists for.

```toml
[sampling]
seed = 424242
chain = ["penalties", "min_p", "temperature"]

[sampling.penalties]
window = 64
repeat = 1.3
frequency = 0.0
presence = 0.0

[sampling.min_p]
p = 0.05

[sampling.temperature]
value = 0.7
```

Defaults equal today: with no `[sampling]` section the chain is `["penalties", "top_k", "top_p", "min_p",
"temperature"]` with every stage neutral (`window 64, repeat 1.0, frequency 0.0, presence 0.0, top_k 0,
top_p 1.0, min_p 0.0, temperature 0.0`), which `SamplingConfig::default()` already proves collapses to
exactly `greedy_pick` (`tok/sample.rs:125-134` doc; the named test is
`default_config_matches_greedy_pick_exactly`).

## 3. field map: every key to a consumer, or a gap

| key | consumer | status |
|---|---|---|
| `seed` | `serving.rs:829` -> `decode.rs:3039` | EXISTS |
| `penalties.window` | `repeat_last_n` -> `repeat_window` (`decode.rs:3029`) | EXISTS (`-1` rejected, `serving.rs:1367`) |
| `penalties.repeat/frequency/presence` | `SamplingConfig` fields -> `apply_repetition_penalty` (`tok/sample.rs:147-179`) | EXISTS |
| `min_p.p` | `apply_min_p` (`tok/sample.rs:252-262`) | EXISTS |
| `temperature.value` | `sample_general` (:462-467) | EXISTS |
| `chain` (order, membership) | none: `sample_general` calls the five filters in fixed order (:457-460) | GAP-1 |
| `[sampling]` as TOML/env on the library | none: `ServingConfig` has no loader (`serving.rs:719`); only `examples/gguf_generate.rs:72-116` | GAP-2 |
| per-request override | none: the example server parses only `model` and `messages` (`examples/openai_serve_gguf.rs:76-87`) and builds `ServingConfig` with defaults (:121-134) | GAP-3 |

## 4. can the decision be a pure function in core plus a pipe?

Decision half: yes, and it already is one. Every filter is a free function over `&mut [(u32, f32)]` or
`&mut Vec<(u32, f32)>` plus numbers (`tok/sample.rs:147-262`), in a `no_std + alloc` crate
(`tok/lib.rs:43-45`). It lives in `proxima-tokenizer`, not `proxima-core`; the SPEC catalog column should
say so (see sketch 7, GAP-5).

Placement half: `select_decoded_token` is the single call site for the ordinary decode and for each
speculative verify row (`decode.rs:5758`), which is why a stage added here composes with speculation for
free and keeps verify bit-identical to plain decode (doc at `decode.rs:1307-1318`). That property is what
any chain-as-data change must preserve, so the chain list must be read inside `sample_next_token`, not at
a new outer call site.

## 5. worked example (doubles as the test)

Ids are real (qwen3 fixture above); logits are illustrative. Step: after history
`[785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 15344, 374]` (the prompt plus the first seven
generated ids; the oracle's next id is `21718`). `window = 64` covers all 12. Counts in the window:
`6722: 2`, `315: 2`, `374: 2`, `13: 1`, `576: 1`; `21718: 0`.

Illustrative raw logits: `6722: 9.7` (the loop continuation), `21718: 9.5` (the oracle's choice),
`374: 8.0`, every other id `2.0`. Config: `repeat = 1.3`, `frequency = presence = 0`, `min_p = 0.05`,
`ln(0.05) = -2.995732`.

Default order (penalties then min_p; the order in the TOML above):
- penalties, `apply_repetition_penalty` (`tok/sample.rs:163-177`): a positive logit is divided by `1.3`:
  `6722: 9.7/1.3 = 7.4615`, `374: 8.0/1.3 = 6.1538`; `21718` and the tail are unchanged (count 0).
- min_p: `max = 9.5` (`21718`), threshold `9.5 - 2.9957 = 6.5043`. Survivors: `21718 (9.5)`,
  `6722 (7.4615)`. `374 (6.1538)` and the tail are dropped.
- greedy readout: `21718`. Without the penalty stage the raw argmax is `6722` (9.7 > 9.5), the loop.

Swapped order (`chain = ["min_p", "penalties", "temperature"]`), the case the flat struct cannot express:
- min_p first: `max = 9.7`, threshold `9.7 - 2.9957 = 6.7043`. Survivors: `6722 (9.7)`, `21718 (9.5)`,
  `374 (8.0)`; the tail is dropped.
- penalties then: `6722 -> 7.4615`, `374 -> 6.1538`, `21718 = 9.5`.
- survivor set `{21718, 6722, 374}` versus `{21718, 6722}` in the default order. With
  `temperature > 0` the draw then differs (the third candidate exists); with `temperature = 0` both orders
  return `21718`. Hand-derived from the cited lines; not executed.

Test shape: both orders over this row, asserting the two survivor sets above. The same row with
`repeat_penalty < 1.0` (a reward) is the adversarial case in section 6, GAP-4.

Config parity (P4): `SamplingSettings` would load the TOML to a value equal to the builder's, by the same
triple `PromptCacheSettings` ships (`interop/prompt_cache_settings.rs:17-100`). Not built here.

## 6. HOOK GAPS (stage, missing input, smallest generic change)

GAP-1. H14 sample / N1. Missing input: the chain is five hard-coded calls in fixed order
(`tok/sample.rs:457-460`), and the config is seven numbers where a disabled value means "stage absent"
(`:87-94`). Order and membership are data in the incumbent (`--samplers`). Smallest change: a `chain:
&'model [Stage]` field beside the numbers, where `Stage` is the closed enum
`{Penalties, TopK, TopP, MinP, Temperature}` (a plain enum already needed to name the stages; the
numbers stay where they are), and `sample_general` folds the list instead of calling five functions.
Default list = today's order, so the default is bit-identical (the existing test
`greedy_fast_path_matches_general_path_over_random_inputs`, `tok/sample.rs`, is the guard). Stages the
module doc already lists as not implemented (`dry`, `top_n_sigma`, `typical_p`, `xtc`, `:25-29`) become
one enum variant plus one function each: a core addition per technique, correct for a closed set.

GAP-2. H14. Missing input: no library loader for any sampling knob (see section 1). `SpeculativeSettings`
and `PromptCacheSettings` are the working pattern. Smallest change: `SamplingSettings` with
`Settings, Validate, Builder` and an `as_sampling` lowering, as in `speculative_settings.rs:253-381`.
Delete the example-local duplicate (`gguf_generate.rs:99-116`) in the same change, because it carries
different defaults (0.8 / 40 / 0.95 / 0.15 / 1.1) from the library's greedy ones, so two "defaults" exist
for the same fields.

GAP-3. H14 / H1. Missing input: sampling is per request in every serving API, and here it is not at all:
the example server ignores any request field except `model` and `messages`
(`openai_serve_gguf.rs:76-87`) and rebuilds the same default `ServingConfig` per request (:167-168).
Smallest change: the request layer maps `temperature`, `top_p`, `seed`, `frequency_penalty`,
`presence_penalty` onto the `[sampling]` values for that call; the `CacheKey` is unaffected because it
names every sampling field `_` with the reason "sampling picks which token comes next"
(`interop/generate/prompt_cache_key.rs:124-134`). Read from code, not measured.

GAP-4. H14. Missing input: the fast-path eligibility rules duplicate what the chain does, by hand, in
two places, and both are order-dependent. (a) `greedy_fast_path_is_safe` (`tok/sample.rs:358-362`) and its
argument (:346-357): top-k, top-p and min-p cannot drop the running maximum, true for `min_p <= 1.0` in
the penalties-first order. If min-p runs before a penalty and `repeat_penalty < 1.0` (a reward raising a
positive logit), a token min-p dropped could finish above the survivor set's maximum, so skipping the
filters is no longer equivalent. This is hand reasoning, labelled plausible: no test exists for it.
(b) `greedy_on_device` (`decode.rs:3581-3590`) is a hand-written conjunction `temperature <= 0 && min_p <=
1.0 && repeat_penalty == 1.0 && frequency_penalty == 0.0 && presence_penalty == 0.0`; a new stage added
to the chain is silently ignored by the device-argmax path until someone edits that line. Smallest
change: one pure function in the tokenizer crate, `fn preserves_argmax(chain, config) -> bool`, read by
both sites. Call site both ways: today two inline conjunctions; after, one call each; the two differ (the
function owns the order-dependent reasoning), so it is a fix, not a relocation.

GAP-5. N1 shape logits. Missing input: nothing mutable exists before the sampler. `select_decoded_token`
takes `logits_row: &[f32]` and returns a token id (`decode.rs:1319-1337`); `sample_next_token` copies the
row into a `Vec<(u32, f32)>` of the whole vocabulary (`tok/sample.rs:451-455`). A grammar or JSON mask
(the N1 inhabitants in the SPEC) needs (i) a mutable row or candidate list before the filters, and (ii)
its state recomputed from `token_history` per row, because the verify loop pushes `selected` into
`token_history` one row at a time (`decode.rs:5768`) and a stateful mask would need rewinding on a
rejected draft. Smallest change: a `Mask` stage variant whose payload is a caller-supplied per-step
bitset slice, applied before `Penalties`; the grammar state machine stays outside and is a pure function
of the history. Status: plausible; nothing in this sketch needs it.

## 7. what is not a pipe, and why

- Every stage: a synchronous reduction over a candidate slice (`tok/sample.rs:3-9`), no I/O, no await.
  Wrapping them in `Pipe` would add a future per filter per token on a path that is allocation-sensitive
  (`greedy_fast_path`'s doc, :364-390: zero vocab-sized allocation).
- The chain list: a closed enum of functions; a runtime-length `and_then` chain is not expressible
  (`proxima-primitives/src/pipe/ext.rs:48-54`).
- `SamplingConfig`: plain data, as its own doc says (`tok/sample.rs:9-14`).

## 8. designs abandoned

- `Stage` as a `Pipe` composed from config: identical call site; a future per token per stage.
- `Vec<Box<dyn Fn(&mut Candidates)>>`: ruled out by the box-free default; the stage set is closed.
- Folding min-p into top-p: they differ (probability mass versus a relative threshold,
  `tok/sample.rs:234-262`).
- Making the chain order a build-time constant (principle 12): order is a per-request preference, not a
  per-system tunable, and request-level override (GAP-3) rules it out.

## defects found in passing

- Two default sets for the same nine knobs (library greedy, `serving.rs:1153-1161`; example sampling,
  `gguf_generate.rs:99-116`).
- `repeat_last_n = -1` rejected with a message that names `generate.rs`'s loop
  (`serving.rs:1367-1375`); that loop is `generate/decode.rs` now (stale pointer).
- The error text at `serving.rs:1345-1353` says "every token greedy_pick sees", while sampling is not
  greedy_pick-only (stale wording).
- The comment-free hand-maintained device-greedy gate (GAP-4b).

## verdict

fits after 5 named hook changes: GAP-1 (chain as ordered list), GAP-2 (library `SamplingSettings`, delete
the example copy), GAP-3 (per-request sampling at the request layer), GAP-4 (one `preserves_argmax`
function instead of two inline conjunctions), GAP-5 (mask stage before the filters). The task as stated,
min-p plus a repetition penalty at today's order, needs GAP-2 only (the numbers exist); order is what the
hook is missing. Not claimed: that the swapped order gives better output, or that any of this preserves
token ids or speed; none of it was run.
