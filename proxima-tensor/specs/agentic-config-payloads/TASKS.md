# agentic-config-payloads -- slices

Each slice: one commit, one behaviour change, one validation command. Update the checkbox and the
note IN THE SAME COMMIT as the slice. Every cargo command runs as `timeout -k 30s 25m git wt cargo
...`, `CARGO_BUILD_JOBS=2`, one build at a time.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | baseline counts for AC7 (a)-(d) at `285b519e5`, before any change | AC7 | AC7 (a), (b), (c), (d) | four nonzero counts, 0 failed, recorded here | [ ] | |
| 1 | `expand_value` / `expand_value_strict` in `proxima-http/src/templates.rs`; `StageRequest.body: Option<Value>`; `build_request` structured arm; executor `Probe.bodies`; call sites | AC1, AC6 lib (2) | `cargo val -p proxima-lib --features tokio --lib -E 'test(stage_payload_) or test(ctl_payload_)'` | 9 passed (AC1 8 + AC6 (2); AC6 (1) lands in slice 2) | [ ] | |
| 2 | `ProcessRpcInitFrame.send: Value`; `run_init` structured arm | AC2, AC6 lib (1) | `cargo val -p proxima-lib --features tokio --lib -E 'test(process_rpc_payload_) or test(ctl_payload_)'` | 3 passed (AC2 1 + AC6 lib 2) | [ ] | |
| 3 | `SynthConfig.body` / `body_template: Option<Value>`, structured arms, env as text, `#[builder(into)]` | AC3 | `cargo val -p proxima-lib --lib -E 'test(synth_payload_)'` | 4 passed | [ ] | |
| 4 | `triage.toml` twin, `json_text_leaves`, the twin tests | AC4, AC6 pipeline_pipe | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_payload_) or test(ctl_payload_)'` | 4 passed | [ ] | |
| 5 | CLI `review_ticket.toml` twin and its session test | AC5, AC6 CLI | `cargo val -p proxima-cli --test integration -E 'test(mcp_config_payload_) or test(ctl_payload_)'` | 3 passed | [ ] | |
| 6 | incumbents and lints | AC7, AC8 | AC7 (a)-(d); AC8 | slice-0 + 15, + 4, + 0, + 0; 0 | [ ] | |

## resume

Last landed slice: none
Next action: audit round 1
Open question, if any: none

## struck

-
