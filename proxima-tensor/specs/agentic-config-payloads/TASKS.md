# agentic-config-payloads -- slices

Each slice: one commit, one behaviour change, one validation command. Update the checkbox and the
note IN THE SAME COMMIT as the slice. Every cargo command runs as `timeout -k 30s 25m git wt cargo
...`, `CARGO_BUILD_JOBS=2`, one build at a time.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | baseline counts for AC7 (a)-(e) at `285b519e5`, before any change | AC7 | AC7 (a), (b), (c), (d), (e) | five nonzero counts, 0 failed, recorded here | [x] | 2026-11-20, source = `285b519e5`, `<tickets>/PROXIMA-49/evidence/baseline-{a,b,c,d,e}.log`: (a) 108 passed 0 failed; (b) 24 passed 0 failed; (c) 5 passed 0 failed; (d) 10 passed 0 failed; (e) 13 passed 0 failed (measured at `e10d54950`, whose source is still `285b519e5`) |
| 1 | `expand_value` / `expand_value_strict` in `proxima-http/src/templates.rs`; `StageRequest.body: Option<Value>`; `build_request` structured arm; executor `Probe.bodies`, `check_received`; call sites | AC1, AC6 lib (3) | `cargo val -p proxima-lib --features tokio --lib -E 'test(stage_payload_) or test(ctl_payload_)'` | 8 passed (AC1 7 + AC6 lib (3)) | [ ] | |
| 2 | `ProcessRpcInitFrame.send: Value`; `run_init` structured arm; `check_initialized` | AC2, AC6 lib (1), (2) | `cargo val -p proxima-lib --features tokio --lib -E 'test(process_rpc_payload_) or test(ctl_payload_)'` | 4 passed (AC2 1 + AC6 lib (1)-(3)) | [ ] | |
| 3 | `SynthConfig.body` / `body_template: Option<Value>`, structured arms, env as text, `#[builder(into)]`; `check_body_json` | AC3, AC6 lib (4)-(6) | `cargo val -p proxima-lib --features tokio --lib -E 'test(synth_payload_) or test(ctl_payload_)'` | 10 passed (AC3 4 + AC6 lib 6) | [ ] | |
| 4 | `triage.toml` twin, `check_seen`, `json_text_leaves`, the twin tests and their JSON controls | AC4, AC6 pipe | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_payload_) or test(ctl_payload_)'` | 7 passed (AC4 3 + AC6 pipe 4) | [ ] | |
| 5 | CLI `review_ticket.toml` twin and its session test | AC5, AC6 cli | `cargo val -p proxima-cli --test integration -E 'test(mcp_config_payload_) or test(ctl_payload_)'` | 3 passed (AC5 2 + AC6 cli 1) | [ ] | |
| 6 | incumbents and lints | AC7, AC8 | AC7 (a)-(e); AC8 (C8, C8x) | 126, 31, 5, 10, 13 passed, 0 failed; C8 prints `exit=0`, `0`, `1`; C8x prints `2` | [ ] | |

## resume

Last landed slice: none
Next action: slice 1
Open question, if any: none

## struck

-
