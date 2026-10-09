# Card 24: Granite continuation, viability prompts, and request completion

**Owner:** GPT-6 Sol
**Dependency:** 23
**Commit:** `test(interop): check selected granite continuation and request`
**Budget:** at most 30 minutes front-to-back, including the acceptance run and TASKS.md update. Parse the checkpoint once in the fixture test; use a fresh `LoadedModel` per independent prompt pair so resident plans do not cross prompt cases.

## Purpose

Card 23 captured a selected multi-row prefill dispatch and compared one generated token. This card checks three separate behaviors on the same real Granite checkpoint: a full recorded continuation, four established viability prompts, and completion of a caller's public `LoadedModel::generate_with_serving_config` API request. Each check keeps `kv_reuse=shared_k` as the only changed attention axis. Each viability pair runs in a separate test process because sequential cases on one loaded model returned a Metal route-compaction mismatch; this card does not establish same-instance sequential-request reliability.

## Read

- `proxima-model-interop/tests/granite_attention_variant_prefill.rs:16-24,93-177`: checkpoint path, F32 cache/whole-prompt serving config, and selected `SharedK` value.
- `omega/src/msl/emit_and_classify.rs:5-27,200-207`: selected KV reuse falls back to legacy when the bound operation cannot use the row-tiled attention form.
- `proxima-model-interop/tests/fixtures/llama-parity/granite_moe/long_prompt_llama_ids.json`: one recorded Proxima README request with 1,000 prompt IDs and 128 generated IDs. The generated prefix is `[203, 433, 19482, 1236, 47615, 8558, 12011, 2783]`.
- `proxima-model-interop/tests/gemma4_correctness_gate.rs:73-117` and `proxima-model-interop/examples/gemma4_ring_parity.rs:130-157`: the four established prompt strings and answer substrings. Their Gemma4 token IDs and raw-completion framing are not Granite oracles; Granite uses its own chat template.
- `proxima-model-interop/tests/arch_data_baseline.rs:947-990,1049-1130`: fixture parsing, tokenizer-ID comparison, and the recorded llama IDs.
- `proxima-model-interop/src/generate/decode.rs:2089-2112`: public `generate_with_serving_config` request surface.
- `proxima-model-interop/src/serving.rs:842-880,935,1241-1262`: greedy sampling defaults and optional attention selection.

## Scope and controls

Use the real Granite GGUF from `PROXIMA_ARCH_GRANITE_MOE_GGUF` or Card 23's documented local path. Use its own tokenizer and the checked-in llama fixture; do not regenerate expected IDs from the current implementation. Keep both continuation arms on the same loaded model and fixture prompt, greedy sampling, `ubatch_size=0`, F32 K/V cache, `GPU_LAYERS_ALL`, disabled prompt cache, and the same numeric policy. Use `None` for legacy and `Some(AttentionVariant { kv_reuse: SharedK, ..Default::default() })` for selected. A shape that cannot use row-tiled MMA must retain request completion by emitting the legacy KV-reuse schedule for that op; it must not fail the request or claim the SharedK kernel ran there.

The fixture prompt and the four viability prompts serve different checks. Do not use the README fixture as a semantic answer oracle, and do not use Gemma4's recorded token IDs as a Granite oracle. Here, request means an in-process public API call, not an HTTP server request.

## Edit

- `proxima-model-interop/tests/granite_attention_variant_prefill.rs`: add one fixture test, four isolated real-model viability tests (two public generation calls each), and one pure rejection control, reusing the Card 23 helpers where possible.
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md`: mark this new row only after its acceptance run and update the three resume lines in the same commit.

## Steps

1. Parse the single `long_prompt_llama_ids.json` record and the Granite checkpoint once. Assert exactly one case, exactly 1,000 recorded prompt IDs, and exactly 128 recorded generated IDs. Encode its prompt with the checkpoint vocabulary and the same BOS/EOS policy used by `arch_data_baseline.rs`; assert the resulting IDs equal all recorded prompt IDs before generating. Use one fresh loaded model for the fixture pair and make two public `LoadedModel::generate_with_serving_config(&fixture.prompt, 128, config)` calls, one legacy and one selected. Assert each returned vector has length 128 and equals the full recorded `generated_ids` vector, element by element. Also assert legacy IDs equal selected IDs and report the first divergent position and both IDs if any comparison fails. The selected call itself is the request-completion check: it must return `Ok`, non-whitespace text, and the exact eight-ID prefix `[203, 433, 19482, 1236, 47615, 8558, 12011, 2783]` as part of its 128-ID result. This is a continuation and public API check on one recorded llama.cpp case, not a statement about other prompts or numeric policies.
2. Use these four established prompts in separate test processes, each with a fresh loaded model and a 48-token greedy limit per call:

| case | exact prompt body | framing | expected substring |
|---|---|---|---|
| paris | `The capital of France is` | Granite chat turn | `paris` |
| soliloquy | `In drama, what is a speech in which a character, alone on stage, speaks their inner thoughts aloud called?` | Granite chat turn | `soliloquy` |
| ant_vs_briefcase | `Which is bigger, an ant or a briefcase?` | Granite chat turn | `briefcase` |
| hippo_vs_building | `Which of these is smaller in size: a hippopotamus or a large office building?` | Granite chat turn | `hippopotamus` |

   Wrap all four bodies exactly as `<|turn>user\n{body}<turn|>\n<|turn>model\n`, matching Granite's own chat template and the long-prompt fixture markers. Put each prompt pair in a separate real-model test process because request results changed or failed when several prompts shared one process. For each case, make one legacy and one selected public `generate_with_serving_config` call with identical prompt and serving settings; that is eight calls across four cases, in addition to the two fixture calls. Assert each request returns nonempty IDs and text, and assert full legacy/selected IDs and texts are equal. Run the expected-answer predicate for both arms; ant and hippo also run directional relation predicates with reversed/negated controls. Print each predicate result and both full outputs. A false semantic predicate is evidence to report, not a passing answer: this card records whether Granite answered each probe coherently and makes no aggregate quality claim.
3. Add one pure failability control for the comparison helpers: change the first expected fixture ID from `203` to another value and require the 128-ID comparison to reject at position 0; check that the expected-substring assertion rejects a response missing its case's answer (for example, a Paris case with no `paris`), and that comparison checks reject negated and reversed relations. Add a source-level dispatch control for one query row: `SharedK` must emit the same legacy entry, source, and grid as the legacy variant because row-tiled MMA does not serve that shape. Its dispatch manifest must report effective `kv_reuse=Legacy`, so it does not label a legacy kernel as SharedK. The real model request then checks this fallback is executable on the four viability prompts. Keep all expected failures visible rather than swallowing errors or accepting a zero-test filter.

## Acceptance criteria

| id | command | exact tests and expected output |
|---|---|---|
| AC24 | `PROXIMA_TEST_TIMEOUT_MS=900000 cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~card_24_)' -j 1 --no-fail-fast --success-output immediate` | filter selects exactly 6 tests and reports 6 passed: one fixture test checks 1,000 prompt IDs, 128/128 llama IDs in each continuation arm, selected non-whitespace request text and prefix `[203, 433, 19482, 1236, 47615, 8558, 12011, 2783]`; four isolated Granite-chat tests each complete legacy and selected public requests, return nonempty full IDs/text, and match IDs/text across arms while recording both answer and relation predicate results; one pure control rejects six false claims. A false semantic predicate is retained in output and must not be described as a good answer. |
| AC24b | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants --lib -E 'test(~shared_k_variant_falls_back_for_single_query_row_dispatch)'` | filter selects exactly 1 test and reports 1 passed; `SharedK` on a one-query-row operation emits the same legacy entry, source, and grid. |

Read the assertions and the per-case output, including any first divergent token and answer text. The two named tests are the count gate; a zero-match filter or a short vector fails. No timing, throughput, aggregate model-quality score, or benchmark replay belongs to this card.

## Residual

The 128-ID oracle is one README prompt under one checkpoint, greedy config, and F32 cache. The four viability checks compare legacy and selected on four established prompt forms. Their captured semantic results were: Paris accepted; ant-vs-briefcase rejected because Granite echoed the prompt; hippo-vs-building rejected because the response stated the hippopotamus was larger; soliloquy rejected because the response called it a monologue. Each arm produced identical token IDs and text within each prompt pair. These four prompts do not establish broader coherence or long-context grounded question answering. Earlier sequential prompt pairs on one loaded model returned `RouteCompactionMismatch`; fresh processes isolate this card's prompt checks and do not establish same-instance repeated-request reliability. The public API request check is the selected 128-ID call and its eight-token prefix; it does not exercise an HTTP server. Other attention axes, cache codecs, prompts, sampling policies, and serving concurrency remain unmeasured by this card.
