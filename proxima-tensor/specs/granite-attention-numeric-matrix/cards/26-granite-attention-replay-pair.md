# Card 26: replay one captured Granite attention pair

**Owner:** GPT-6 Luna
**Dependency:** 25
**Commit:** `test(interop): compare captured granite attention outputs`
**Budget:** at most 30 minutes front-to-back, including AC26, review, and the TASKS.md update. If capture buffers cannot reproduce a full output after their request, preserve the mismatch payload and repair capture ownership in a separate card before timing.

## Purpose

Give the next timing slice a real, replayable F32-cache Granite prefill dispatch for each arm. The same checkpoint and prompt run through the public serving path once with all-legacy attention and once with `kv_reuse=shared_k`; the test compares one matched attention output byte for byte and refuses missing, unreplayable, incomplete, or falsely matched records.

## Source boundary

- `proxima-model-interop/tests/granite_attention_variant_prefill.rs:23-26,99-110,113-154,427-464`: checkpoint, 971-token prompt, F32 cache, selected serving config, and live capture setup.
- `omega/src/metal/arena_encode_dispatch_finish.rs:1278-1355,1555-1601`: a capture retains live buffer references and identifies node, extents, entry, source SHA, and grid. It is not an immutable input snapshot.
- `omega/src/metal/arena_encode_dispatch_finish.rs:2283-2342`: `replay_output_elements` poisons and reads the requested F32 output span after a replay. It does not inspect the replay fault buffer; do not describe a successful return as a fault check.
- `proxima-model-interop/examples/norm_variant_ab.rs:355-402,419-487,654-669,691,802-812`: the current timed harness loads Gemma4, admits a typed selector only in describe mode, and requires external `.metal` body files for timed arms. This card uses two genuinely selected live Granite captures; it does not pass a typed selector into that timed loop.

## Edit

- `proxima-model-interop/tests/granite_attention_variant_prefill.rs`: add a small capture helper and the two AC26 tests; retain the existing Card 23–25 assertions and prompt text.
- `proxima-tensor/specs/granite-attention-numeric-matrix/TASKS.md`: tick Card 26 only after the count-bearing command and payload inspection.

## Steps

1. Load the exact Granite 3.1 1B A400M Instruct GGUF Q8_0 checkpoint from `PROXIMA_ARCH_GRANITE_MOE_GGUF` or the existing fixture path. Build the existing 971-token Sherlock prompt using the checkpoint tokenizer. Set `PROXIMA_CAPTURE_LIVE=1`, `PROXIMA_CAPTURE_NODES=all`, and `PROXIMA_CAPTURE_STEPS=0`; use the existing `serving_config` with F32 K/V, no prompt cache, `ubatch_size=0`, and a one-token request.
2. Drain captures, request the all-legacy arm, then take its captures. Select exactly one `cached_attention` record with extents `[rows,8,2,64]`, `rows>1`, a nonzero grid, `unreplayable=None`, and **no `Binding::Fault`**; fail if the selected record has a fault binding. Choose it by a deterministic `(node, extents)` order and print that identity. Immediately replay it with `replay_output_elements(Some(extents.product()))`, before the next request can reuse its buffers. Require exactly `extents.product()*4` bytes and at least one non-poison F32 output word. Keep the returned bytes and the selected node/extents/entry/SHA/grid.
3. On the same `LoadedModel` and prompt, request `Some(shared_k_variant())`, take captures, and find exactly one record with the same node and extents. Require nonzero grid, `unreplayable=None`, **no `Binding::Fault`**, and changed entry and source SHA. Immediately replay its full output span and require the same byte count. Require its output bytes to equal the legacy bytes exactly; on failure print the first differing byte/word and both values, node, extents, SHA, and grid. Require both one-token public requests to return the same nonempty token IDs. Label the replay comparison as one dispatch on two separately captured requests, not a whole-model tensor proof.
4. Add a pure comparator control that flips one byte in a constructed nonempty output and requires the exact-byte comparison to reject it at that offset. The same test rejects an absent selected record and a constructed selected-record descriptor whose bindings include `Binding::Fault`. This test loads no model; the descriptor exposes the binding-presence field without fabricating a `CapturedDispatch` with private Metal state. The helper returns a typed `Result`, so no false match is hidden by logging.

## Acceptance criteria

| id | command | exact expected observation |
|---|---|---|
| AC26 | `PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~card_26_)' -j 1 --success-output immediate` | Filter selects exactly 2 tests and reports 2 passed: one real-model pair prints 1 matched multi-row attention dispatch, 2 complete replay output spans, exact output bytes, equal nonempty generated ID vectors, and zero `Binding::Fault` on both matched records; one pure control rejects exactly 3 false cases: a changed output byte, an absent selected record, and a selected descriptor containing `Binding::Fault`. |

Read the printed node/extents, both entry/SHA/grid identities, output byte count, and both generated ID payloads. A zero-byte span, an unreplayable record, or no selected match fails AC26. Do not time a benchmark or infer performance from the two correctness replays.

## Residual

The captures retain live Metal buffers, and `replay_output_elements` does not validate fault-buffer contents. The two admitted records have no `Binding::Fault`; if that condition changes, AC26 fails instead of treating replay success as fault-free execution. The test replays each arm directly after its request, but retained buffers can still have been reused within that request. If exact output comparison fails, inspect captured input ownership and payload before treating the difference as a kernel arithmetic error. This one F32-cache SharedK dispatch pair does not establish whole-layer output parity, BF16/BF8 behavior, other axes, other prompts, or timing.
