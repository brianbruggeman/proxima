# Card 23: model-backed Granite prefill selector

**Owner:** GPT-6 Luna
**Dependency:** 22
**Commit:** `feat(interop): carry attention variant into granite prefill`
**Budget:** at most 30 minutes front-to-back, including the two acceptance tests and TASKS.md update. Keep this to forwarding the request selector into the existing plan builders and one model-backed test fixture.

## Purpose

Prove that one non-legacy attention selection reaches an executed, model-backed, multi-query Granite prefill dispatch while the generated first-token payload stays equal to the legacy control.

## Scope

One explicit, default-off `AttentionVariant` value travels from `ServingConfig` through `BackendRuntime` to the Metal `Plan` used for the first Granite prefill evaluation. The control uses the same model, prompt, numeric policy, cache storage, and execution path with the all-legacy selector. The test checks the captured selected dispatch and the generated first token. No timing, throughput, model-quality, CUDA, benchmark replay, or default activation.

## Read

- `proxima-model-interop/examples/norm_variant_ab.rs:63-125,128-179,654-669`: seven-field parser and describe-only guard; this path does not execute a selected model dispatch.
- `proxima-model-interop/src/serving.rs:760-801`: request settings and prefill width.
- `proxima-model-interop/src/generate/residency_caches.rs:2275-2311,2413-2460,3059-3093`: request settings reach plan construction; placed plans use one shared builder.
- `proxima-model-interop/src/generate/resident_plans.rs:68-111,111-166`: resident plan identity is derived from the request config and must distinguish selectors.
- `proxima-model-interop/src/generate/prompt_cache_key.rs:27-62,85-165`: cached prefix rows are keyed by the numeric and attention settings that produced them.
- `omega/src/backend.rs:263-310,665-790`: backend plan errors and runtime plan setters preserve Metal-only selection.
- `omega/src/metal/device_buffers_arena_plan.rs:1761-1806,2329-2343`: `Plan::set_attention_variant` and selected kernel keys.
- `omega/src/metal/arena_encode_dispatch_finish.rs:1540-1575`: live capture exposes executed node, kind, entry, source SHA, extents, and grid.
- `proxima-model-interop/src/generate/decode.rs:4900-5075`: a generation's first multi-row forward is the real prompt prefill and uses the backend plan path under test.
- `proxima-model-interop/tests/serving_default_ubatch_prefill_parity.rs:30-40,49-105,119-122`: local Granite checkpoint, real prompt, and single-pass prefill fixture.

## Edit

- `proxima-model-interop/src/serving.rs` (feature-gated optional selector, default `None`)
- `omega/src/backend.rs` (backend-plan selector forwarding and explicit non-Metal refusal)
- `proxima-model-interop/src/generate/residency_caches.rs` (copy the selector into `BackendRuntime` and apply it to the newly built Metal plan before resolution; cover the actual prefill plan path, including the placed-plan builder when enabled)
- `proxima-model-interop/src/generate/resident_plans.rs` (include the selector in `PlanIdentity::of` so a resident decode plan cannot cross selectors)
- `proxima-model-interop/src/generate/prompt_cache_key.rs` (include the selector so cached prefill rows cannot cross selectors)
- `proxima-model-interop/tests/granite_attention_variant_prefill.rs` (two model-backed checks)
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md` (row and resume only)

## Steps

1. Add `ServingConfig::attention_variant: Option<omega::AttentionVariant>` behind `metal-attn-variants`. `None` retains the current all-legacy plan. Copy the value into `BackendRuntime`; on newly constructed Metal plans with more than one query row, call `Plan::set_attention_variant` before the first pipeline resolves. Decode-shaped plans remain on their existing path because `shared_k` is admitted only for row-tiled attention. Preserve the typed admission error and refuse a non-Metal plan rather than silently ignoring a selected variant. `BackendRuntime` is built per request, so its ordinary cached and refitted plans carry one selector; include that selector in `PlanIdentity::of` and `CacheKey::of` so resident plans and cached prefill rows cannot cross requests with different selectors. Do not read `AB_ATTENTION_VARIANT` inside Omega or the serving library.
2. Use the real Granite MoE GGUF selected by `PROXIMA_ARCH_GRANITE_MOE_GGUF` or the fixture's documented default path, with the checkpoint's own tokenizer and the existing Sherlock passage. Set `ubatch_size=0`, F32 K/V cache, `GPU_LAYERS_ALL`, and the same numeric policy in both arms. With `PROXIMA_CAPTURE_LIVE=1`, `PROXIMA_CAPTURE_NODES=all`, and `PROXIMA_CAPTURE_STEPS=0`, run one generated token with `None` and with the explicit one-factor `kv_reuse=shared_k` selection; all six other fields are legacy. Drain captures after each arm. Match corresponding row-tiled `CachedAttention` dispatches by node and extents; require at least one `[rows,8,2,64]` dispatch with `rows > 1`. Assert the selected entry and source SHA differ from legacy, with grid recorded and node/extents unchanged. This must inspect a dispatch captured from execution, not a requested config, synthetic `BoundOp`, or dry manifest.
3. On the same model and prompt, request one generated token with each config and assert the two nonempty token-id payloads are equal. This is an end-to-end output guard; do not call `CapturedDispatch::replay_output`, `time_gpu_ns`, or any timed replay, and do not read pooled output buffers after execution. The existing Card 15 Metal payload test remains the kernel-level numeric control and is not rerun by AC23.
4. The second test exercises the captured-dispatch assertion helper with a legacy record labeled as the selected arm and requires a rejection. It uses constructed metadata and loads no model; this degenerate control proves the identity check can fail. The real-model test fails on a missing checkpoint, unavailable Metal, zero captured matches, or an empty generated token payload.

## Acceptance criteria

| id | command | expected |
|---|---|---|
| AC23 | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~card_23_)' -j 1` | filter selects exactly 2 tests; 2 passed; the positive test opens the real Granite checkpoint, captures a selected multi-row prefill attention dispatch, and compares generated token ids; the pure negative control rejects a false selected-dispatch assertion |

Read the test assertions and captured dispatch records, not only the runner summary. Count matched multi-row dispatches and generated token ids in the test output. No benchmark or timed replay runs in this card.

## Residual

This card establishes model-backed dispatch identity and first-token equality for one F32-storage `shared_k` selection. It does not compare the full attention tensor on the model payload; Card 15 supplies the kernel-level numeric fixture. BF16/BF8 storage and the other selector axes retain their own per-card evidence; model-quality and performance claims require separate measurements.
