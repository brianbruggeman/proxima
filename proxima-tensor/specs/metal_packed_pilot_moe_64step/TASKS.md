# Metal packed pilot 64-step loop -- slices

Each slice: one behavior change, one validation command, and an asserted count.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Admit the 64-step state-carry and retained-payload contract | AC1 | `rg -c '^VERDICT: ADMIT$' proxima-tensor/specs/metal_packed_pilot_moe_64step/ADMISSION.md` | one admitted revision | [x] | |
| 2 | Run both packed codecs through all 64 Metal steps and retain the compared payloads | AC1 | `test "$(uname -s)" = Darwin && RUSTC_WRAPPER= cargo test -p omega --test packed_brain_float_moe_training_metal packed_pilot_cross_entropy_64step_records_payload -- --exact --nocapture > /tmp/metal-packed-pilot-64step.log && python3 -c 'from pathlib import Path; import json; log=Path("/tmp/metal-packed-pilot-64step.log").read_text(); marker="METAL_PACKED_PILOT_64STEP_JSON="; start=log.index(marker)+len(marker); records,_=json.JSONDecoder().raw_decode(log[start:].lstrip()); assert len(records)==128; out=Path("proxima-tensor/specs/metal_packed_pilot_moe_64step/results/metal-packed-pilot-64step.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(records,indent=2)+"\n")' && python3 proxima-tensor/specs/metal_packed_pilot_moe_64step/validate_report.py proxima-tensor/specs/metal_packed_pilot_moe_64step/results/metal-packed-pilot-64step.json` | AC command emitted 128 records; validator printed `validated_records=128`; test reported `1 passed, 0 failed` | [x] | `results/metal-packed-pilot-64step.json` |

## resume

Last landed slice: both codec loops emitted validated 64-step payloads; AC1 recorded
Open question, if any: none; this remains a 32-parameter synthetic fixture

## struck

-
