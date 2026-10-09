# Metal packed pilot cross-entropy -- slices

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Admit the exact tiny-pilot Metal graph and payload contract | AC1-AC2 | `rg -c '^VERDICT: ADMIT$' proxima-tensor/specs/metal_packed_pilot_moe_training/ADMISSION.md` | one admitted revision | [x] | ADMIT recorded in ADMISSION.md |
| 2 | Execute eight-token packed cross-entropy and compare all CPU/scalar payloads | AC1 | `test "$(uname -s)" = Darwin && cargo test -p omega --test packed_brain_float_moe_training_metal packed_pilot_cross_entropy_matches_cpu_and_scalar -- --exact` | one Metal test exercises BF8 and BF4 with logits, loss, compact/coalesced gradients, and optimizer-state parity | [x] | 1 passed, 0 failed; BF8/BF4 matched CPU and scalar payloads, exact command run 2026-10-09 |
| 3 | Retain both codec records from the Metal test output | AC2 | `test "$(uname -s)" = Darwin && RUSTC_WRAPPER= cargo test -p omega --test packed_brain_float_moe_training_metal packed_pilot_cross_entropy_records_payload -- --exact --nocapture > /tmp/metal-packed-pilot-training.log && python3 -c 'from pathlib import Path; import json; log=Path("/tmp/metal-packed-pilot-training.log").read_text(); marker="METAL_PACKED_PILOT_JSON="; start=log.index(marker)+len(marker); records,_=json.JSONDecoder().raw_decode(log[start:].lstrip()); assert len(records)==2; out=Path("proxima-tensor/specs/metal_packed_pilot_moe_training/results/metal-packed-pilot-training.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(records,indent=2)+"\n")' && python3 proxima-tensor/specs/metal_packed_pilot_moe_training/validate_report.py proxima-tensor/specs/metal_packed_pilot_moe_training/results/metal-packed-pilot-training.json` | exactly two BF8/BF4 records satisfy validator schema and nested dimensions | [x] | retained result JSON; validator passed |

## resume

Last completed slice: all three slices completed
Result: AC1 and AC2 ran; the retained BF8/BF4 payload passed schema validation
Open question, if any: none; this step is a backend parity fixture only
