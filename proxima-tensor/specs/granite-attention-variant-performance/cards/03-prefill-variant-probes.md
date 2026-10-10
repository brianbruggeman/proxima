# Card 03: compare Granite prefill attention variants

**Owner:** GPT-6 Luna
**Dependency:** Cards 00–02
**Commit:** `perf(interop): record granite prefill variant probes`
**Budget:** at most 30 minutes front-to-back, including the acceptance run and task update.

## Purpose

Retain matched real Granite attention replay data for F16 MMA and the K/V reuse, query scheduling, and prefetch combinations already exposed by `AttentionVariant`.

## Scope

Add four named probes to the existing Granite prefill test. Each compares the legacy F32 path with one selected variant at tokenizer-derived 256- and 971-token prompt lengths, with F32 K/V storage in both arms. Add three one-axis controls at 971 tokens, each comparing against the F16-MMA/F32-KV dispatch: query-parallel rows, SharedK, and Rows16. Record exact prompt lengths, selected dispatch identity/grid, all 20 alternating replay samples, per-sample signed deltas, output tensor difference summary, and generated IDs. Require generated IDs to match before collecting timing samples. Do not change kernel implementation or defaults in this card.

The four legacy comparisons are: F16 MMA only; F16 + SharedK + Rows8 + simdgroup row scheduling + per-head topology; the same selector with SharedKv; and the SharedKv selector with next-block prefetch enabled. The SharedK/SharedKv pair differs by the reuse field alone. The SharedKv prefetch-on/off pair differs by the prefetch field alone. The three same-dtype controls change exactly one selector from F16 MMA: query parallelism, K reuse, or tile height.

## Acceptance

Run:

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~perf_probe_granite_) | test(~perf_card_04_)' -j 1 --success-output immediate
```

Acceptance requires exactly 7 selected tests, 7 passed, and 13 filtered out. The four legacy comparisons each emit two shape records, one for each actual tokenizer count 256 and 971; the three controls each emit one 971-token record. Every record includes 20 samples per arm and 20 signed deltas, equal generated token IDs checked before timing, output difference counts, captured entry identities, and grids. Save the complete runner output under `evidence/card03-acceptance-2026-10-10.log`.

Run the negative control separately:

```sh
PROXIMA_TEST_TIMEOUT_MS=900000 PROXIMA_CAPTURE_LIVE=1 PROXIMA_CAPTURE_NODES=all PROXIMA_CAPTURE_STEPS=0 CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo nextest run -p proxima-model-interop --features std,metal,instrument,metal-attn-split-rows,metal-attn-variants --test granite_attention_variant_prefill -E 'test(~card_24_wrong_oracle_and_wrong_fact_are_rejected)' -j 1 --success-output immediate
```

It must select 1 test, pass 1, and skip 19; the comparator must reject the mutated token ID. This negative test calls the comparator directly; the accepted probe source checks ID equality at lines 1195-1198 before its timing loop starts at line 1203. Retain `evidence/card03-id-mismatch-control-2026-10-10.log`.

This acceptance records isolated dispatch timing only. It does not establish whole-layer/request latency, model-quality parity, or a root cause by itself. F16 K/V is excluded from these prompt-prefill probes: serving binds F32 cache leaves during initial prefill, while device-resident F16 K/V adoption occurs after prefill in `proxima-model-interop/src/generate/decode.rs:4103-4115`; the failed selector attempt is retained in `evidence/f16-kv-live-no-speculation-2026-10-10.log`.

## Observations

| probe / condition | captured result | what the record supports |
|---|---|---|
| F16 MMA, F32 K/V, 971 actual prompt tokens, Granite 3.1 1B A400M Instruct Q8_0, local Metal replay | Acceptance p50: F32 MMA 1,065,125 ns; F16 MMA 1,085,542 ns. Three additional independent process records: F32/F16 p50 1,075,958/1,085,500 ns; 1,075,333/1,088,375 ns; 1,067,500/1,085,750 ns. | F16 MMA's selected p50 is higher in each recorded process at this 971-token dispatch. In the acceptance run, 994,276 of 994,304 output elements changed bits; max absolute difference 11.7567 and RMS 0.05445; generated IDs were `[322]` in both arms. Identical token IDs do not establish semantic output parity. Raw samples: `evidence/card03-acceptance-2026-10-10.log:37-47`, `evidence/f16-mma-run1-2026-10-10.log:5-6`, `evidence/f16-mma-run2-2026-10-10.log:5-6`, `evidence/f16-mma-run3-2026-10-10.log:5-6`. |
| Rows16 only vs F16 MMA baseline, same F32 K/V and F16 MMA, 971 tokens | p50 1,086,708 -> 1,080,292 ns; selected CoV 5.71%; output bits identical; IDs `[322]`. | The selected distribution exceeds the skill's ~5% repeat threshold, so the p50 delta is not isolated as a repeatable scheduling effect. Raw samples and CoV: `evidence/card03-acceptance-2026-10-10.log:7-16`. |
| SharedK only vs F16 MMA baseline, same F32 K/V, 971 tokens | p50 1,089,042 -> 1,945,750 ns; output bits identical; IDs `[322]`. | This same-shape selector's replay was higher. Kernel source has each SIMD group's K fragments stored to threadgroup memory before a barrier and reloaded (`omega/src/msl/cached_attention_row_tiled.rs:578-600`); this source path identifies extra staging operations, while the timing record alone does not attribute all elapsed time to them. Raw samples: `evidence/card03-acceptance-2026-10-10.log:17-26`. |
| simdgroup-row query scheduling only vs F16 MMA baseline, same F32 K/V, 971 tokens | p50 1,085,750 -> 1,993,750 ns; output bits identical; IDs `[322]`. | Retained as a one-axis control; see `evidence/card03-acceptance-2026-10-10.log:27-36`. |
| SharedKv + Rows8 + simdgroup rows + per-head, prefetch off/on, F32 K/V, 971 tokens | p50: legacy F32 MMA 1,076,000 ns; selected prefetch-off 2,405,833 ns and prefetch-on 5,962,917 ns. Same selector at 256 tokens: 186,875 ns vs 396,250 ns (off), and 186,250 ns vs 837,833 ns (on). | Both compared prompts record higher selected replay times. Prefetch-on is an especially large observed difference; MSL source shows next-block copies into scratch/threadgroup operands and barriers (`omega/src/msl/cached_attention_row_tiled.rs:888-940`), but no GPU counter was collected to split the added copy, barrier, occupancy, or cache effects. Raw paired samples: `evidence/card03-acceptance-2026-10-10.log:59-69`; prefetch-off: `:70-80`. |
| F16 K/V selector on initial Granite prefill | The request reached binding and returned `CachedAttentionVariantStorageMismatch { selected: "f16", bound: "f32" }`. | The selected initial prefill cache leaf is F32, so this selector cannot be benchmarked through the captured initial-prefill path. The source adopts F16 device KV only after cached length is nonzero (`proxima-model-interop/src/generate/decode.rs:4103-4115`). Failure record: `evidence/f16-kv-live-no-speculation-2026-10-10.log:17-23`. |
| wrong-ID negative control | 1 test passed; comparator rejected the mutated token ID. | This test exercises the comparator directly, not the replay/timing helper. The accepted probe code performs the same comparison at `proxima-model-interop/tests/granite_attention_variant_prefill.rs:1195-1198` before entering the timing loop at `:1203`. Raw control: `evidence/card03-id-mismatch-control-2026-10-10.log:1-12`. |

The probe loop is a test-only synchronous loop over two owned `CapturedDispatch` values (`proxima-model-interop/tests/granite_attention_variant_prefill.rs:1194-1220`). `time_gpu_ns(1)` synchronously returns each replay duration; there is no producer/consumer flow to compose. The `Pipe` contract at `proxima-primitives/src/pipe/primitives.rs:20-24` requires an async `In -> Result<Out, Err>` call, so wrapping the timing call would only relocate the harness and would not alter kernel data reuse. No pipe, type, or runtime primitive was added. Attention variants remain caller-selected data via `AttentionVariant` in the already-gated Metal path.

## Design constraints applied

- Reused `AttentionVariant` and the existing row-tiled MSL generator; did not add a second tuning API or a Granite-specific kernel.
- Kept the selectors explicit and test-only probes opt-in through the existing `metal-attn-variants` feature.
- The captured Rows16 result crossed the ~5% CoV threshold, so that one-process observation is retained without treating it as a stable gain.
- SharedK and prefetch candidates remain in the evidence even though their paired replay times were higher; neither result is hidden or converted into a whole-request claim.
- F16 K/V was removed from this initial-prefill probe after the real request rejected its F32-bound cache. No selected F16-KV implementation remains in the kernel source.

## Reproduction

Run the acceptance command and negative control in this card; the independent-process F16-MMA commands are recorded in the three `f16-mma-run*.log` files. The logs are raw output from `cargo nextest`; each `granite_variant_probe` record contains both ordered sample arrays and the signed paired deltas.
