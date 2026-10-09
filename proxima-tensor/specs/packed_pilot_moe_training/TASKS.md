# Packed-byte BF8/BF4 tiny MoE pilot -- slices

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Admit packed-byte pilot contract | AC1-AC2 | `rg -c '^VERDICT: ADMIT$' proxima-tensor/specs/packed_pilot_moe_training/ADMISSION.md` | one admitted revision | [x] | Spec auditor admitted the `ExpertSource` CPU path, exact train/held-out byte-state contract, and scalar-reference parity validator. |
| 2 | Bind packed BF8/BF4 rows into one-step pilot graph and match independent scalar reference | AC1 | `cargo test -p proxima-autograd --test packed_low_precision_moe_training packed_pilot_step_matches_scalar_reference -- --exact` | 1 test passes across BF8, BF4, and FP32 control with full per-token and optimizer state parity | [x] | AC1: 1 passed, 0 failed; BF8, BF4, and FP32 control logits, token losses, compact/coalesced gradients, FP32 master updates, Adam moments, and held-out outputs matched the independent scalar reference within `1e-6`. |
| 3 | Run and retain packed-byte 64-step pilot payloads for all nine format/seed arms | AC2 | `cargo run -p proxima-autograd --example low_precision_moe_training_pilot -- --fixture tiny --steps 64 --seeds 3 --report json > /tmp/packed-tiny-pilot.json && python3 proxima-tensor/specs/packed_pilot_moe_training/validate_report.py --input /tmp/packed-tiny-pilot.json --output proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json && jq -e '(.arms|length)==9 and ([.arms[].steps[]]|length)==576' proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json` | exact 3×3 arm set and 576 complete step records pass schema, exact train/held-out byte re-encoding, span-offset, and scalar-reference parity checks | [x] | AC2: cargo/example completed, validator wrote the durable report, and `jq` returned `true` for 9 arms/576 steps. The first validator pass exposed an FP32 empty-span expectation mismatch; the validator was corrected and the exact command rerun successfully. |

## resume

Last completed slice: 3
Next action: build the packed cross-entropy graph on Metal, starting from a spec that preserves this CPU report as its oracle
Open question, if any: none; this contract retains the synthetic pilot boundary
