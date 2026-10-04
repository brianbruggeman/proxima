# sketch 13: conformal cascade at H17 (settle)

status: paper test. Read at main 4b4be6cf with `git show main:<path>`. Nothing was built or run. `interop` =
`proxima-model-interop/src`; `DEC` = `interop/generate/decode.rs`; `SRV` = `interop/serving.rs`. `FT` = the draft
under `/Users/brianbruggeman/repos/slot-0/proxima-fsm-techniques/proxima-tensor/specs/fsm-techniques/` (read-only,
one opinion, not main).

Technique as the draft states it (`FT SPEC.md:77`, row T8): draw N samples; the score of an answer is
s = 1 - votes/N; q-hat is the ceil((1-alpha)(n+1))-th order statistic of the calibration scores of the true answers;
accept iff the prediction set C = {answers with s <= q-hat} has exactly one member; defaults N = 16, T = 0.7,
n >= 200 calibration items. I did not read arXiv:2607.25018; the mechanics below are the draft's, labelled as such.

## ground facts on main (what exists, read)

- One request is one call: `generate_with_serving_config(prompt, max_tokens, serving_config) -> (ids, text,
  stopped_by_eos)` is public (`DEC:2107-2125`), tokenizes the text itself, and goes through the prompt cache
  (`run_decode_loop` -> `run_decode_loop_observed` -> `run_decode_loop_through_cache`, `DEC:2652-2668`,
  `DEC:2957-2985`; the cached path needs `seed.is_none()`, which a text call satisfies, `prompt_cache.rs:1221`).
- Sampling is a field of the call: `ServingConfig.temperature`, `top_k`, `seed` are public
  (`SRV:796`, `SRV:799`, `SRV:829`); the sampler's RNG is seeded once per call from `seed`
  (`DEC:3039`, `fastrand::Rng::with_seed(serving_config.seed)`). So N samples are N calls with `seed + index`. The
  follow-up job does exactly this to draw branches (`prewarm_follow_up.rs:197-198`, `seed.wrapping_add(index + 1)`).
- The sampling fields are not in the cache key (`prompt_cache_key.rs:126-134`: `temperature: _` through `seed: _`,
  reason: sampling picks the next token, the entry stores the ids actually forwarded). N sampled completions of one
  prompt therefore share one cache entry; the second call finds the prompt prefix and rewinds the first call's
  generated rows (`CachePath::Rewind`, `prompt_cache.rs:74-76`).
- `LoadedModel` implements `Pipe` (`In = (String, usize)`, `Out = (Vec<u32>, String, bool)`,
  `interop/generate/residency_caches.rs:3146-3159`) but through `generate`, which fixes the config to
  `supported_serving_config(0, ..)` (`DEC:1381-1392`): `gpu_layers = 0`, CPU, default seed and greedy sampling. A
  tier cannot be that pipe.
- `Fallback<P, S>` exists (`proxima-primitives/src/pipe/resilience/fallback.rs:12-35`) and routes to `secondary` on
  any `Err` from `primary`, replaying the input. `AndThen` requires `Second::In = First::Out` and `Second::Err:
  From<First::Err>` (`proxima-primitives/src/pipe/primitives.rs:203-208`).
- `TokenEvent` carries `token_id`, `text_piece`, `phase`, `step`, `elapsed_ms` and no probability
  (`residency_caches.rs:3169-3194`). Not needed for a vote-based judge; needed for T7 to T9 judges.
- Batching samples is not available: `parallel_sequences != 1` is refused (`SRV:1261-1267`). Samples run one after
  another.
- No cascade code exists: `git grep -n -i -E "cascade|conformal|escalat" main -- proxima-model-interop/src proxima-tensor/src proxima-core/src` returned 0 lines when I ran it.

## shape chosen, and the contested decision

Shape: configuration, one pure function (`settle`), one config enum (`Judge`), and ONE pipe (`Cascade`) of about 40
lines. Contested decision: how to compose tiers, three candidates.
- `Fallback{primary, secondary}`: the judge declines by returning `Err`. Rejected: `Fallback` falls back on every
  error (`fallback.rs:29-31`), including `InteropError` from the engine, so a failed forward silently escalates;
  a decline is also a unit "no" with the votes thrown away.
- `AndThen` over `Result<Answer, Request>` carried in `Ok` (draft R17a, `FT SPEC.md:290`): each later tier's input is
  the previous tier's output, so every later tier needs a pass-through wrapper, and the chain type grows with the
  tier count (the draft's own refusal `CascadeTooManyTiers`, `FT SPEC.md:373`, arms 1 to 4). N wrappers exist to make one
  pipe composable; the defect is the shape, not the count.
- A loop over a slice of tiers. Chosen: every tier is the same Rust type (`LoadedModel`, one value per loaded
  checkpoint), so the list is homogeneous and a slice iterates it; no `dyn`, no type-level chain.

## 1. the configuration

```toml
[[cascade.tiers]]
model = "/models/gemma-4-e2b-it-Q4_0.gguf"
temperature_milli = 700

[cascade.tiers.judge]
kind = "conformal"
samples = 16
max_disagree = 4

[[cascade.tiers]]
model = "/models/qwen3-8b-Q4_0.gguf"

[cascade.tiers.judge]
kind = "always"
```

`max_disagree` is q-hat in units of samples: an answer is in the set iff at most `max_disagree` of the `samples`
draws disagree with it. It is an integer, so it round-trips through env text and compares exactly. The draft's
milli form (`FT SPEC.md:217`, a q-hat table) needs a division by N that is not exact (one disagreeing draw of 16 is 62.5 per thousand, not an integer).

Default reproduces today: an absent `cascade` is one tier with judge `always` and `samples = 1`, which is one call
of `generate_with_serving_config` with `seed + 0`: the same call a caller makes today (section 2 shows both).

## 2. the pure function and the pipe

```rust
pub fn settle(votes: &[u32], samples: u32, max_disagree: u32) -> Option<usize> {
    let mut inside = votes.iter().enumerate().filter(|&(_, &count)| samples - count <= max_disagree);
    match (inside.next(), inside.next()) {
        (Some((index, _)), None) => Some(index),
        _ => None,
    }
}
```

`votes` holds one count per distinct answer; the counts sum to `samples`, so `samples - count` cannot underflow.

```rust
pub enum Judge { Always, Conformal { samples: u32, max_disagree: u32 } }

pub struct Tier<'model> { pub model: &'model LoadedModel<'model>, pub serving: ServingConfig<'model>, pub judge: Judge }
pub struct Cascade<'model> { pub tiers: &'model [Tier<'model>] }

impl Tier<'_> {
    fn settle_answer(&self, prompt: &str, max_tokens: usize) -> Result<Option<String>, InteropError> {
        let samples = match self.judge { Judge::Always => 1, Judge::Conformal { samples, .. } => samples };
        let mut tally: Vec<(Vec<u32>, String, u32)> = Vec::new();
        for index in 0..samples {
            let serving = ServingConfig { seed: self.serving.seed.wrapping_add(u64::from(index)), ..self.serving };
            let (ids, text, _) = self.model.generate_with_serving_config(prompt, max_tokens, serving)?;
            match tally.iter_mut().find(|entry| entry.0 == ids) {
                Some(entry) => entry.2 += 1,
                None => tally.push((ids, text, 1)),
            }
        }
        let votes: Vec<u32> = tally.iter().map(|entry| entry.2).collect();
        let picked = match self.judge {
            Judge::Always => Some(0),
            Judge::Conformal { samples, max_disagree } => settle(&votes, samples, max_disagree),
        };
        Ok(picked.map(|index| tally.swap_remove(index).1))
    }
}

impl Pipe for Cascade<'_> {
    type In = (String, usize);
    type Out = String;
    type Err = InteropError;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<String, InteropError>> {
        async move {
            let (prompt, max_tokens) = input;
            for tier in self.tiers {
                if let Some(answer) = tier.settle_answer(&prompt, max_tokens)? {
                    return Ok(answer);
                }
            }
            Err(InteropError::CascadeUnsettled)
        }
    }
}
```

About 38 lines of body. The async block with a synchronous body is the precedent at `residency_caches.rs:3151-3158`.
`Judge::Always` runs one call and returns it: with one tier, `Cascade::call((prompt, n))` and
`model.generate_with_serving_config(&prompt, n, serving)` are the same call with the same seed, so the default adds no
behaviour. The vote key is the generated token ids, exact equality; answers that differ only in wording count as
different answers, so the technique suits constrained answers; stated limit, unmeasured.

## 3. worked example (hand-derived, not executed)

Calibration, n = 9 items, N = 16 samples each. Votes for the true answer: `[16, 16, 15, 14, 16, 12, 9, 16, 13]`.
- Disagreements `16 - votes`: `[0, 0, 1, 2, 0, 4, 7, 0, 3]`; sorted: `[0, 0, 0, 0, 1, 2, 3, 4, 7]`.
- alpha = 0.2: rank = ceil((1000 - 200) * (9 + 1) / 1000) = ceil(8.000) = 8, computed in integers because the float
  product 0.8 * 10 is not guaranteed to be exactly 8. The 8th smallest is 4. So `max_disagree = 4`
  (`q-hat = 4/16 = 0.25`).

Serving with `samples = 16`, `max_disagree = 4`:
- votes `[13, 2, 1]`: disagree `[3, 14, 15]`; inside = {0}; `settle = Some(0)`; accept the first answer.
- votes `[12, 4]`: disagree `[4, 12]`; inside = {0}; `Some(0)`.
- votes `[9, 7]`: disagree `[7, 9]`; inside = {}; `None`; escalate.
- votes `[8, 8]`: `None`; escalate.
- votes `[16]`: disagree `[0]`; `Some(0)`.
With a looser `max_disagree = 10`: votes `[8, 8]` gives disagree `[8, 8]`, both inside, size 2, `None`; votes `[13, 3]`
gives `[3, 13]`, inside = {0}, `Some(0)`. Two answers can both be inside only when `2 * (samples - max_disagree) <=
samples`, i.e. `max_disagree >= 8` here; at `max_disagree = 4` the set is never larger than 1, so "size 0 or 1" is
the only outcome and escalation means size 0.

Cascade trace, two tiers, a prompt whose tier-1 votes are `[9, 7]`: tier 1 runs 16 calls, `settle` is `None`, tier 2
(`Always`) runs 1 call and its text is returned; the 16 tier-1 completions are discarded (that cost is real and
unmeasured). A prompt whose tier-1 votes are `[13, 3]` returns after tier 1 and tier 2 runs 0 calls (the draft's
R17d).

Control that must fail: calibrate q-hat on tier 1 at temperature 0.7 and serve with temperature 0.0. Greedy samples
agree with themselves, every prompt returns `[16]`, and the cascade never escalates: the measurement would be of
the sampler, not of the model's uncertainty. The binding is that `samples`, `temperature_milli` and `max_disagree`
are one calibrated tuple and must not be configured apart (G3).

## 4. cache-key binding (cross-cutting rule, checked against `generate/prompt_cache_key.rs`)

Nothing about what a cached row means changes: a conformal draw is an ordinary request whose sampling fields are
excluded from the key by design (`prompt_cache_key.rs:126-134`). Each tier is a separate `LoadedModel` with its own
`prompt_cache` (`self.prompt_cache`, used at `prompt_cache.rs:1060-1065`), so tiers never share entries; the key
omits weight identity because an entry lives on one model (`prompt_cache_key.rs:14-18`), which stays true here.
The sketch does not touch H9 or H13.

One cost interaction, not a key issue: sample 2 onward rewinds the previous sample's generated rows. A sliding-window
layer rewinds only within `ring_rewind_slack` (default 256, `SRV:491`, `prompt_cache_settings.rs:30-32`); beyond it the
entry falls to a checkpoint or a full prefill. Answers shorter than 256 tokens stay inside the slack; longer ones:
unread, depends on checkpoint placement.

## 5. HOOK GAPS (every place the sketch edits core)

G1. H17 home for `settle`, `Judge`, `Cascade`. Missing input: a crate. SPEC puts decisions in `proxima-core`; the
decision has one consumer, this pipe, and `proxima-core/src` has no serving module (sketch 10, D5). They can sit in
the consuming crate (`proxima-model-interop` or the server example). Not a hook defect: nothing existing is edited,
code is added next to its one caller.

G2. Not a gap, recorded because the draft asserts one: "no N-samples-per-step mode" (`FT SPEC.md:98`, A9) and the
draft's `decode.samples` knob (R10e, R16b). N samples are N calls with `seed + index` (ground facts); the draft's
decode-loop change is not required for this technique.

G3. Validation. Missing input: a check that ties the calibrated tuple together and that the last tier cannot decline.
Rows to add: last tier's judge is `always` (otherwise `InteropError::CascadeUnsettled`, a variant that does not
exist on main; the error enum is read-only to me, so this is a guess about where it goes); `max_disagree <=
samples`; `temperature_milli > 0` for a conformal judge (the control above). These are config validation, the same
class as sketch 8's GAP-12.

G4. `LoadedModel`'s `Pipe` impl (`residency_caches.rs:3146-3159`) is CPU-only, fixed-config and single-sample, so the
sketch bypasses it and calls the public method. That impl is a second, weaker entry point to the same loop, and
nothing on main composes it (SPEC: "only tests call it"). Smallest change if the SPEC wants tiers composed through
`Pipe`: let the impl carry a `ServingConfig`, so `LoadedModel`-plus-config is the pipe. Not needed for this sketch.

G5. Judges beyond votes need a readout. `threshold`, `classifier` and `isotonic` judges read a per-token top-2 margin
or a logprob, and `TokenEvent` has neither (`residency_caches.rs:3169-3194`). That is an N2 (tap and edit) or H14
(sample) gap, not an H17 one; H17 only needs "answer in, answer or escalate out".

G6. Raw text in, raw text out. A chat request needs H2 (template), which is absent on main (SPEC); the prompt
tokenizes with BOS only (`DEC:2957-2965`, `wants_bos`). The cascade passes the request unchanged to each tier, so
each tier's own tokenizer runs on the same text.

## 6. is any of it a pipe? each candidate, against the two gates

- `Cascade` as a pipe: passes the pipe question (`In = (String, usize)`, `Out = String`). Second gate: a caller can
  now hold one value that stands for "ask tier 1, escalate on doubt" and hand it to anything that takes a `Pipe`,
  which they could not do with a free function over a slice; the two call sites (`cascade.call(request)` and
  `ask_tiers(&tiers, request)`) differ only by the trait, so the pipe earns its place only if a caller composes it.
  No caller composes `LoadedModel` as a `Pipe` today (SPEC), so this is stated, not proven.
- `settle`: a pure function; a pipe would add a future around a 5-line filter.
- Tier choice by `Fallback` or `AndThen`: rejected above (silent escalation; wrapper per tier).
Therefore one pipe, about 40 lines, at H17, and it is the weakest of the sketch's claims.

## 7. designs abandoned

- `Fallback` over errors (shape chosen).
- `AndThen` over `Result<Answer, Request>`, with per-tier pass-through wrappers.
- A fixed `samples` and `max_disagree` shared by every tier: abandoned, each tier is a separate calibration.
- A q-hat table keyed by prompt class (the draft's `qhat table`): abandoned for one integer; a table is a second
  calibration surface and has no consumer on main.
- Batched sampling through `parallel_sequences`: unavailable (refused, `SRV:1261-1267`).

## defects found in passing

- D1. The draft's A9 (`FT SPEC.md:98`) and `decode.samples` addition are not needed (G2).
- D2. `Judge` thresholds in the draft's grammar (`FT SPEC.md:217`) mix three unrelated inputs under one `judge` key
  (a margin readout, a classifier model, a vote tally); the vote tally needs only the answers, the others need
  readouts that do not exist (G5). One section is carrying two hooks.
- D3. `LoadedModel`'s `Pipe` impl ignores every serving field, including `gpu_layers` (G4).
- D4. The draft's calibration fitter (R18, `FT SPEC.md:297`) is an offline example binary; the order statistic it
  needs is 4 lines (`sorted[rank - 1]` with the integer rank above) and has no runtime caller, so it belongs in the
  example, not in core.

## verdict

fits as is, for a vote-based judge over caller-supplied text: configuration, one pure function, one config enum and
one pipe of about 40 lines, all additions beside their single caller and no edit to a core file; the gaps G3 to G6 are
validation, an unused weaker entry point, a readout that other judges (not this one) need, and template absence.
Not claimed: that the coverage guarantee holds for this model and corpus (no calibration was run), that the tier
pipe composes with anything today, or any cost or latency.
