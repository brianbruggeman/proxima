# Card 26 — four Granite semantic checks

## Question

Do the four user-facing Granite 3.1 generation cases pass their expected-answer checks on the pinned Q8_0 model, under both Legacy and the selected SharedK attention configuration?

## Test contract

The cases are Paris, soliloquy, ant versus briefcase, and hippo versus building. They run against the pinned Granite 3.1 1B A400M Instruct Q8_0 checkpoint with F32 K/V. For each case, the test compares Legacy and selected generated token IDs and text, then checks the expected answer and, for the two comparisons, the expected relation.

The helper previously printed the semantic check results without asserting them. It now fails when the expected answer or relation check fails in `proxima-model-interop/tests/granite_attention_variant_prefill.rs:759-812`.

## Command

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 RUSTC_WRAPPER= CARGO_BUILD_JOBS=1 nice -n 10 cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~card_24_granite_viability_)' -j 1 --success-output immediate --no-fail-fast
```

Raw output: `evidence/card24-semantic-four-cases-2026-10-10.log`.

## Results

Nextest ran all four cases serially: Paris passed; soliloquy, ant/briefcase, and hippo/building failed the current acceptance checks. The tests assert Legacy and selected text equality before checking semantics, so the same generated text applies to both attention arms in each case.

| Case | Observed generated text | Check result |
|---|---|---|
| Paris | `The capital of France is Paris.` | Passes the expected `paris` check for both arms. |
| Soliloquy | `Answer: A monologue. Explanation: A monologue is a speech in which a character, alone on stage, speaks their inner thoughts aloud.` | Fails the exact `soliloquy` substring check; whether `monologue` should count as semantically acceptable is unresolved by this lexical oracle. |
| Ant/briefcase | The generated text repeats the prompt; it contains `briefcase` but does not affirm the expected size relation. | Fails the relation check. |
| Hippo/building | `A hippopotamus is significantly larger than a large office building.` | Fails: the response states the opposite of the expected relation. |

The test log records `4 tests run: 1 passed, 3 failed`. The observed outputs are from the Q8_0 Granite 3.1 baseline. These cases do not run the Granite 4 Micro cold-converted artifacts.

## Fine-tuning and ablation scope

The measured four-case run took 213.927 seconds on this Mac, including model load and generation. This is a measured evaluation duration, not a fine-tuning duration. No training run or fine-tuning ablation has been captured in this card.
