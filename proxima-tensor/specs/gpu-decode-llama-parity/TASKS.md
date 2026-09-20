# gpu-decode-llama-parity -- slices (components)

Each slice is a whole disciplined component with its own `specs/<slug>` and its
own commit series. This parent TASKS tracks their landing + the one e2e claim.
Validation here is each component's headline op-collapse AC (see its own spec for
the full AC set).

| # | slice (component) | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | `moe-mul-mat-id` — batched indexed expert product (packed-row bucket) | R1 | `reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.allon "$BLOB" "$Q" 32 gpu 2>&1|grep 'reduce-packed-row-blocked'|grep -oE 'op_count=[0-9]+'` | `op_count` ≤ 200 (from ≈1211) | [~] | spec eaa754403; scaffold landed 85ac70da5; kernel crux is BIND/ARENA (contiguity+stride), corrected a1f04dc3 |
| 2 | `gdn-attention-fusion` — fused GatedDeltaNet mixer + attention (cooperative bucket) | R2 | `reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.allon "$BLOB" "$Q" 32 gpu 2>&1|grep 'reduce-cooperative'|grep -oE 'op_count=[0-9]+'` | `op_count` ≤ 150 (from ≈551) | [ ] | NOT spec'd yet — `/spec-first` after component 1's e2e number is in |
| 3 | `elementwise-stage-fusion` — stage-fused rmsnorm/silu/softplus/conv (elementwise bucket) | R3 | `reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.allon "$BLOB" "$Q" 32 gpu 2>&1|grep -w 'elementwise'|grep -oE 'op_count=[0-9]+'` | `op_count` ≤ 250 (from ≈503) | [ ] | NOT spec'd yet |
| 4 | e2e parity claim (all three flags on) | R4, R5 | `reclaim; PROXIMA_METAL_OP_PROFILE_STEP=1 /tmp/ggf.allon "$BLOB" "$Q" 32 gpu >/tmp/on.txt 2>&1; reclaim; /tmp/ggf.alloff "$BLOB" "$Q" 32 gpu >/tmp/off.txt 2>&1; echo on=$(grep -oE 'mean[^0-9]*[0-9.]+' /tmp/on.txt|grep -oE '[0-9.]+'|head -1); diff <(grep generated_text /tmp/on.txt) <(grep generated_text /tmp/off.txt)|grep -c '^[<>]'` | on-mean ≤ 1.1 × llama-mean (compare to `ollama run qwen3.6:35b-a3b --verbose` rate); `0` differing lines | [ ] | the refutation gate — if buckets collapse but TTNT does not move, RETRACT |

## resume

Last landed slice: 0 of the campaign (component 1 scaffold landed; kernel crux corrected + re-scoped to bind/arena)
Next action: land component 1's bind/arena fix (moe-mul-mat-id slice 2), get its packed-row op_count ≤200 + parity, then measure component-1-only TTNT before starting component 2's spec
Open question, if any: does closing packed-row alone move TTNT measurably, or is the e2e win gated on all three together? (R4 measures all-on; component-1-only TTNT is the early read)

## 2026-09-19 decode-dense session (proxima-wt-hoist)

- Component 1 (batched-reduce) CORRECTNESS verified at KB/MB scale (Float32 bit-exact + Q4K within the codebase's quant-noise budget vs an f32 oracle; gpu-decode-diagnosis). BUT its op_count≤200 + TTNT AC (slice 1 / slice 4) is STILL unmeasured — parked on the single 24GB `gguf_generate` run, which needs owner RAM authorization ("don't consume RAM without permission").
- Small-dense-decode scoping (qwen3:0.6b, MEASURED): LAUNCH-bound — 513 dispatches/tok, GPU busy only 46% of wall, 54% CPU dispatch overhead. NOT a proxy for the 8b's bandwidth-bound (36.8% util) gap. The dominant small-model lever = components 2 (R2 cooperative/gdn-attention-fusion — HIGHEST leverage) + 3 (R3 elementwise), the SAME buckets as the MoE; est ~40% dispatch cut (513→~300). So R2/R3 are the levers for both regimes; extend THIS spec, don't open a new one.
- Landed this session (uncommitted, proxima-wt-hoist), both CORRECT but mis-targeted for their measured symptom (see [[feedback_verify_fix_target_against_model_bytes]]): (a) Q8_0 packed-row fast-path (emit_and_classify.rs:1731 + packed_row_blocked_ggml.rs Q8_0 arm; 2.1e-6 vs f32 oracle, 26/26 green) — helps Q8_0-weight models (likely qwen35moe attn_v), NOT qwen3:0.6b (its attn_v is F16, zero Q8_0). (b) CPU-path prefill guard drop (run_reduce_scan.rs leading_total==1; 669 green) — CPU engine only; gpu prefill runs omega Metal (already token-batched), so it doesn't touch the gpu 769ms.
- Also found: gemma4-E2B doesn't load (matformer per-layer-array feed_forward_length, hparams.rs:74) — enablement in flight.
- REFUTATION GATE HOLDS: measure component-1-only TTNT (the 24GB run) BEFORE building R2/R3 — do not spend fusion tooling until dispatch-collapse is proven to move wall-clock. So the decode-dense fusion campaign's next step IS the 24GB component-1 measurement (owner RAM go).

## struck

-
