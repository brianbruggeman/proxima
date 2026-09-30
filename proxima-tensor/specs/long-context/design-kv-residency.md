# KV residency and storage dtype: replaces slices 8/8b/9 (REJECTED v1, 2026-09-29)

## Why v1 is rejected (proxima-critic pass, same day)

Blockers:
1. **Ownership.** `LayerCacheState::Placed` contradicts `serving_backend.rs:59-64,79-80`:
   `ServingCache` never holds a device handle, and `Verify` clones it. Whichever way it
   goes, it breaks: a handle aliases device memory on clone, and a marker makes a cloned
   `PrefixState` or `Verify` snapshot hold zero rows. The lockstep matches end in
   `unreachable!` (`residency_caches.rs:1077-1079,1135-1137`), so a new variant panics at
   runtime.
2. **Speculative decoding (default-on at main 3d0696d2) is unaddressed.** Rewind of placed
   rows is not designed. Under narrow storage, a verify batch attends its draft rows at f32
   while plain decode re-reads them narrowed, which breaks main SPEC R1 byte-identity.
3. **Prefill is unaddressed.** gemma4 prefill arena peak fits
   `134,015*L + 107.06*L^2` (DERIVED from 4 measured points, 1,618 to 7,898 tokens), which
   extrapolates to about 119 GB at 32,768. Every 8.x validation context above about 8K
   cannot run until prefill is fixed. A prefill fix may change the reader, and with it the
   operands this design stores into.

Major issues:
- Path-2 narrow storage needs a read-back that violates `qwen35_synth_hybrid.rs:225-230`.
- `PlacedKv` re-creates a residency bool and erases named leaf roles.
- The deletions break AC23 (`before.txt:494-495,502`).
- Slice 8.1's validation test is `#[ignore]`d and blind to the defect.
- Default F16 has no admission site that knows the arch or backend, and breaks the CPU path.
- It duplicates `GgmlType::block_layout` and `proxima_gguf::quant::q8_0::quantize`.
- It collides with slot-0-ea's dirty omega files beyond `cached_attention_render.rs`.

Kept from v1 as findings, not as design:
- The three-path table.
- The per-token host copy of `12288*C + 12582912` B for gemma4.
- The observation that f16-in-host-cache does not cut peak.
- The suspected qwen3.6 dense-KV write gap, which is still to be settled by a test that
  can see it.

Next: the prefill root cause (in flight) comes first. KV residency v2 is re-derived after
it, and must answer ownership, rewind and verify numerics explicitly.

---

(v1 text follows, kept for the record)

Source: a proxima-architect pass on 2026-09-29, read-only at df3766dd plus uncommitted work.
All byte figures are DERIVED from code arithmetic, not measured.

## Finding: three KV paths, not one

| | path 1 Uniform (qwen3-8b) | path 2 Monolithic (qwen3.6) | path 3 Custom (gemma4) |
|---|---|---|---|
| store | 3 f32 PlacedBuffers/layer (`decode.rs:5632-5644`) | 4 f32 PlacedBuffers per dense layer (`decode.rs:2840-2888`) | host `LayerCache` Vec<f32> (`residency_caches.rs:103-111`), sliding layers `KvRing` |
| per-step upload | none (input placement, `decode.rs:5819`); an 805 MB host placeholder scratch at 131072 | none for placed leaves, but a 10.7 GB host pad scratch at 262144 (`residency_caches.rs:1059-1074`) | memcpy into `KvPadScratch`: `12288*C + 12582912` B/token (1.62 GB at 131072) |
| write | output placement in-graph | output placement, pre-gather arm only | host append from read-back |
| reader | fused `CachedAttention` (`render_cached_attention`) | two-range; fused if recognized | generic reduce/elementwise. `metal-fuse-attn-decode` is default-off (`Cargo.toml:169-179`), so 0 fused ops (`proxima-tensor/src/spec/tests.rs:13902-13916`) |

The generic reduce path already reads `Codec::Float16` / `Codec::Q8_0` per operand
(`omega/src/msl/signature_tokens_prelude.rs:878-901`).

Suspected defect (read, not run): on qwen3.6's default full-graph arm (`qwen35moe_pre_gather`
defaults to false, `serving.rs:671`), dense placed buffers are allocated and the host append
is skipped (`decode.rs:4881`). But `evaluate_with_placements` receives only SSM placements
(`decode.rs:4216-4225`), so dense KV may never be written. Slice 8.1 settles it.

`KvCacheShape` is where information is destroyed. `Monolithic` doubles as a residency
identity test (`decode.rs:2800-2802`).

## Shape

- Generalize path 2's `Qwen35DenseAttentionBuffers` (4 hard-named leaves) into a per-layer
  `PlacedKv { layers: Vec<Option<Vec<PlacedKvLeaf>>>, positions }`.
- Placement is driven by per-layer state `LayerCacheState::Placed`, not by `KvCacheShape`.
- Storage dtype is the existing `GgmlType`. There is no new config field.
- Path 1's single-range stays for Uniform. gemma4 cannot use it: the builder has no
  shared-KV concept (`pregather.rs:2628-2658`).
- Sliding rings stay host f32 (12,582,912 B, O(1)). A device ring needs in-kernel modulo,
  and gemma4's unfused default graph has no hook for it.
- Narrow storage on the two-range paths is written by host encode from f32 staging.
  Only path 1 needs a device store kernel.
- F16 in `LayerCache` plus an f32 expansion scratch does NOT reduce peak, because all
  scratch blocks are live at once (`residency_caches.rs:1082-1094`). Only a device-resident
  buffer reduces peak.

Deleted: `Qwen35DenseAttentionBuffers`, `Qwen35DenseAttentionPlacement`,
`qwen35_dense_attention_placement_enabled`, the `dense_attention_is_placed` bools
(`decode.rs:3546-3558`, `:4870-4883`), and the pad scratch for placed layers.

## Slices

| # | slice | validation | expected |
|---|---|---|---|
| 8.0 | measured `KV_HOST_COPY_BYTES` counter (in `copy_into_padded` and `unroll_live_rows`) plus `TokenBreakdown.kv_host_copy_bytes`; harness `--decode-steps` prints it | `niah --model "$GEMMA4" --ctx 8192 --needles 1 --kv f32 --decode-steps 16` | 16/16 steps == `12288*cached_len+12582912`; `cached_attention_ops=0` |
| 8.1 | `PlacedKv` (f32) replaces the qwen35 types; full-graph arm passes the placements; no pad scratch for placed layers | qwen35 full-graph metal test plus a new test (layer-3 dense rows nonzero after step 1; device == host logits); AC23 comm | all pass; `0` |
| 8.2 | gemma4 full layers placed; hidden `--host-kv` control | ring parity vs `gemma4_base_tokens.txt`; niah `--decode-steps 16` at 8192 and 32768 | 256/256 twice; 16/16 steps `kv_host_copy_bytes == 12582912` at both contexts; the control grows by 301,989,888 B |
| 8.3 | `GgmlType::row_bytes`; dtype-aware `kv_row_bytes`, `MemoryBudget`, `PlacedKv`; harness `kv_bytes` from `device_bytes()` | `kv_row_bytes_by_storage`, `memory_budget_gemma4_storage` | 9 + 3 passed |
| 8.4 | omega: a placed Packed{F16,Q8_0} input reduces identically to f32 widened; typed EmitError if fused `CachedAttention` gets a packed operand | `cargo nextest run -p omega --features metal placed_packed_kv` | 4 passed |
| 8.5 | F16 storage end to end; default F16; qwen3.6 narrow storage rejected until 10.1 | AC16; `default_kv_is_f16 serving_default_admission` | B == A, B >= Y, Y > 0; 3 passed |
| 9.0 | `quantize_row_q8_0` in proxima_tensor::cpu | `q8_0_kv_roundtrip` | 4 passed (error <= amax/254) |
| 9.1 | Q8_0 for gemma4 | AC18 | C >= Y, Y > 0 |
| 10.0 | real qwen3.6 program fused-op count | one decode step | `cached_attention_ops = 10` (unverified) |
| 10.1 | `render_cached_attention` reads F16 K/V via per-operand codecs | `cached_attention_f16_kv` | 3 passed |
| 10.2 | prefill must be chunked: today the split loop is one evaluation per position, and `prefill_chunk_positions` only applies to the defective opt-in path | AC20 | kv_bytes=5368709120, X >= Y |
| AC21 path 1 | sizes and offsets from storage; remove the 805 MB placeholder; device store kernel (F16, then Q8_0); fused-kernel codec reads (float4 loads at `cached_attention_render.rs:413,419`); chunk the placed loop's whole-prompt evaluation (`decode.rs:5692,5707`) | AC21 | as spec |

## SPEC values to amend

- AC19: f16 = 817,889,280 and q8_0 = 440,401,920, because the ring stays f32.
- R13/R15: name the reader per model.

## Abandoned

- gemma4 on single-range.
- F16 `LayerCache` plus f32 scratch.
- A device ring with modulo.
- A new `KvStorage` type or `KvResidency` field.
- A device store kernel for the two-range paths.
