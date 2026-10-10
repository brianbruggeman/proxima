# Card 11: select row-tiled simdgroup count coherently

**Owner:** GPT-6 Luna

**Dependency:** Cards 03 and 09; existing `AttentionVariant` row-tiled selectors

**Commit:** `perf(omega): select row-tiled simdgroup count`

**Budget:** at most 30 minutes front-to-back, including compile and focused CPU acceptance checks.

## Purpose

Expose the row-tiled kernel's simdgroup count as an independent dispatch choice for Granite prefill. This card establishes a selectable and internally consistent dispatch shape; it makes no latency or quality claim.

The existing row-tiled form already derives its split count and threadgroup shape from `AttentionRowSchedule`. The selected count must also determine entry identity, emitted source, dispatch grid width, partial-kernel split uniforms, merge split uniforms, and scratch allocation. The previous design considered was a literal 32-query-row llama transplant or a new Granite kernel. Pinned llama.cpp Metal uses Q=8 for its dense path and Q=1/2/4 for its alternate half4x4 path; this card instead exposes the existing Proxima row-tiled schedule. The pinned source is `/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal/ggml-metal-impl.h:126-130` and `/Users/brianbruggeman/repos/others/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:3421-3425,3574-3592`.

## Scope

Add a typed `AttentionSimdgroupCount` selector (`Legacy`, `Groups2`, `Groups4`, `Groups8`) behind the existing `metal-attn-variants` feature. Thread it through the schedule and dispatch identity. For every planned or unplaced dispatch, use that same schedule to compute the partial and merge split counts and the corresponding scratch length. Keep default selection on `Legacy`.

Use the captured Granite shape `extents=[1000,8,2,64]` for the dispatch test. For the nondegenerate split-consistency test, use an eight-row, eight-KV-head, two-query-group, 128-wide shape with cached capacity 4096 and eight new rows. Four query groups at 128-wide do not admit the Legacy schedule under the current 16-accumulator-fragment limit (`omega/omega-runtime.toml:531`, `omega/src/msl/signature_tokens_prelude.rs:2836-2844`); reducing to two makes both compared schedules legal without changing the production sizing rule. Assert that the selected and legacy forms produce different split counts, then check both partial and merge uniform words and scratch length against the selected count.

Do not add a Granite-specific kernel, change the production default, run a GPU replay, or state a performance outcome. A separate one-variable replay card can use the toggle after this dispatch contract has run on the intended device.

## Discipline and pipe review

The baseline is the existing `Legacy` schedule. The abandoned design is a literal Q=32 llama port or another Granite-specific kernel; this card extends the existing typed schedule. It adds no new kernel primitive and leaves the default unchanged. No latency or hardware-resource experiment is part of this dispatch contract; later GPU replay must compare `Legacy` against one explicit count at the same captured shape and retain the output and dispatch payload.

The call site is synchronous: `AttentionMmaSelection::from_variant` takes `AttentionVariant` and returns `Result<(AttentionMmaSelection, AttentionRowSchedule), (&'static str, &'static str)>` (`omega/src/msl/signature_tokens_prelude.rs:1870-1873`). It does not return a `Future`, which `Pipe::call` requires (`proxima-primitives/src/pipe/primitives.rs:48-64`), and no loop, channel, or async dataflow is introduced. `proxima-pipe` catalog review against `ai_docs/examples-index.jsonl`: fan_out—no N sink pipes; fan_in—no pollable sources; filter—no `Decide` admission guard; gate—no readiness or demand state; backpressure—no producer/consumer queue; best-effort—no lossy queue policy; rate_limit—no clocked admission; retry—no failed attempt loop; transform—sync `Result`, not a `Future`; fallback—no alternate async backend; circuit_breaker—no dependency health state; codec—no frame boundary; deadline—no clock-based timeout; delivery—no send guarantee; cache—no upstream lookup/writeback; record—no live pipe traffic; replay—no upstream cassette; plugin-skeleton—no plugin registration; load-balance—no backend selection; multi_runtime—no executor boundary; signal—no async completion wait; cancellation—no cancellation tree; chaos—no fault injection. No pipe code or type is needed for this dispatch selector.

## Acceptance

Run the feature build and exactly five focused CPU tests:

```sh
CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo check -p omega --features metal,metal-attn-variants,metal-attn-split-rows
CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p omega --lib --features std,metal,metal-attn-split-rows,metal-attn-variants -E 'test(explicit_four_simdgroup_dispatch_preserves_row_and_key_tiles) | test(explicit_simdgroup_count_rejects_a_non_row_tiled_shape) | test(explicit_simdgroup_count_updates_merge_admission_at_one_split) | test(explicit_simdgroup_count_keeps_partial_scratch_and_both_uniforms_aligned) | test(card_20_dispatch_matrix_exposes_legacy_and_eight_one_factor_flips)' -j 1 --success-output immediate
```

| criterion | required observation |
|---|---|
| Feature build | `cargo check` exits 0 with the four selectors enabled. |
| Main dispatch | Exactly 1 matching test passes. The Granite row-tiled form reports the selected simdgroup count; its row height and key block stay fixed, grid threadgroup width follows the count, emitted source and dispatch identity differ from Legacy. The same test also admits the composed F16 + SharedKv + Rows16 + SimdgroupRows + Groups4 variant and checks its row owner, shared K/V source flags, and four-simdgroup source. |
| Unsupported form | Exactly 1 matching test passes. Applying explicit count to a non-row-tiled shape returns the `simdgroup_count` axis-specific unsupported error. |
| Split and scratch agreement | Exactly 1 matching test passes on the nondegenerate 128-wide shape. Legacy and selected split counts differ; selected scratch length, partial uniform split word, and merge uniform split word all equal the selected form's split count. A separate shape crosses from two splits to one; the selected kernel writes output directly and omits the merge, while Legacy binds scratch and emits its merge. |
| One-axis matrix | Exactly 1 matching test passes. The manifest lists Legacy plus eight one-factor flips, and each candidate changes exactly one selector axis. |
| GPU scope | No GPU replay or benchmark is part of this card. Record no speed, semantic-quality, layer-latency, or parity conclusion. |

If a focused test cannot start, retain its process state and loader/runtime evidence; do not replace the execution result with a compile-only claim.
