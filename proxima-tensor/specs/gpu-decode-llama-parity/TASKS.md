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

## struck

-
