# Card 25: key grouped route prepasses by their baked shape

**Owner:** GPT-6 Sol
**Dependency:** 24
**Commit:** `fix(omega): key route prepasses by token shape`
**Budget:** at most 30 minutes front-to-back, including the Metal request run and TASKS.md update.

## Purpose

Card 24 used a fresh process for each established Granite prompt because a prior combined run returned `RouteCompactionMismatch`. The fixture-first sequence reproduced it at the first selected Paris request. Captured Metal payloads showed the grouped GEMM source had `scratch_base = 1 + 2 * 32 + 8000`, while the current 168-token route allocated 1,060 bytes; the kernel identity omitted the exact token and expert counts baked into that source. This card adds those counts to grouped route pipeline identity, pins equal-grid/different-token identities, and reruns the fixture plus all four prompt pairs sequentially through one `LoadedModel`.

## Read

- `proxima-model-interop/tests/granite_attention_variant_prefill.rs`: `granite_viability_checks`, `serving_config`, `shared_k_variant`, and `run_granite_viability_case` define the checkpoint, prompt text, serving options, and comparison checks.
- `proxima-model-interop/src/generate/decode.rs:2089-2112`: public `LoadedModel::generate_with_serving_config` request API.
- Card 24's residual records the earlier `RouteCompactionMismatch` and its isolated per-prompt process layout.
- `omega/src/msl/emit_and_classify.rs`: `kernel_cache_key_for_grid` builds the pipeline identity.
- `omega/src/msl/expert_grouped_gemm.rs`: `route_compaction_key`, `grouped_token_total`, and `grouped_expert_count` define the shape values embedded in grouped prepass source and storage.
- `omega/src/metal/arena_encode_dispatch_finish.rs`: `resolve_route_prepass` and `encode_route_prepass` compile and allocate the grouped prepass.

## Edit

- `omega/src/msl/emit_and_classify.rs`: include the routed prepass's exact token and expert counts in the key, because its MSL source embeds both in compaction layout constants.
- `omega/src/msl/tests.rs`: assert token counts 300 and 301 with the same launch grid produce distinct pipeline identities.
- `omega/src/metal/arena_encode_dispatch_finish.rs`: expose route-prepass grid/cap data through the instrumented capture selector `PROXIMA_CAPTURE_NODES=route-prepass`.
- `proxima-model-interop/tests/granite_attention_variant_prefill.rs`: add a single request test that loads the Granite checkpoint once, then submits all four prompt pairs on that same model. For every prompt, call legacy and selected `SharedK`, require successful nonempty payloads and exact token/text equality, and print answer/relation predicates plus both full outputs.
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md`: add Card 25 with its exact test count, observed outputs, and result of the earlier error reproduction attempt.

## Steps

1. Load one model and first issue the recorded 1,000-token fixture prompt at a 128-token limit, legacy then selected `SharedK`. Require both generated ID vectors to equal the fixture's full recorded 128 IDs and both texts to match.
2. Reuse the four exact Granite chat prompts from `granite_viability_checks`; do not alter prompt framing, generation length, sampling, cache, or numeric policy. In prompt order Paris, soliloquy, ant-vs-briefcase, hippo-vs-building, issue one legacy public request then one selected `SharedK` public request on that same model.
3. Require each arm to return nonempty IDs and text. Require full token IDs and text to agree for each pair. Print the expected-answer predicate, directional comparison predicate where applicable, and both complete outputs, including false semantic predicates.
4. Run only the named acceptance test with one nextest job and immediate success output. Keep any request error, mismatch, or route fault visible; do not switch to separate processes to satisfy this card.

## Acceptance criteria

| id | command | exact tests and expected output |
|---|---|---|
| AC25a | `cargo nextest run -p omega --features metal-attn-split-rows,metal-attn-variants,metal-grouped-gemm --lib -E 'test(~the_route_prepass_and_the_locating_gemm_render_for_the_compacted_mode_only)'` | Filter selects exactly 1 test and reports 1 passed. Token counts 300 and 301 have identical launch grids but distinct pipeline identities. |
| AC25b | `PROXIMA_TEST_TIMEOUT_MS=900000 cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~card_25_granite_sequential_public_requests_complete_on_one_model)' -j 1 --success-output immediate` | Filter selects exactly 1 test and reports 1 passed. One loaded Granite model completes ten sequential public calls: the 128-token fixture legacy/selected pair matches all recorded IDs, followed by four legacy/selected viability pairs with identical nonempty IDs and text. Output includes semantic predicates and full per-prompt payloads. A failure retains its exact prompt, arm, and error. |

Read each printed response and predicate. Passing request completion is not a semantic-quality verdict; each predicate result is reported independently.

## Residual

This is one ordered four-prompt sequence on Granite 3.1 1B A400M Instruct, one Metal device, one serving configuration, and one loaded model. It does not establish reliability under other prompt orders, concurrency, longer contexts, cache formats, or sampling configurations. The observed semantic predicates remain separate from request completion: Paris passes; soliloquy does not contain the expected term despite answering “A monologue”; the ant prompt is echoed; and the hippo/building comparison is reversed.
