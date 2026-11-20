# agentic-config-payloads -- slices

Each slice: one commit, one behaviour change, one validation command. Update the checkbox and the
note IN THE SAME COMMIT as the slice. Every cargo command runs as `timeout -k 30s 25m git wt cargo
...`, `CARGO_BUILD_JOBS=2`, one build at a time.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | baseline counts for AC7 (a)-(e) at `285b519e5`, before any change | AC7 | AC7 (a), (b), (c), (d), (e) | five nonzero counts, 0 failed, recorded here | [x] | 2026-11-20, source = `285b519e5`, `<tickets>/PROXIMA-49/evidence/baseline-{a,b,c,d,e}.log`: (a) 108 passed 0 failed; (b) 24 passed 0 failed; (c) 5 passed 0 failed; (d) 10 passed 0 failed; (e) 13 passed 0 failed (measured at `e10d54950`, whose source is still `285b519e5`) |
| 1 | `expand_value` / `expand_value_strict` in `proxima-http/src/templates.rs`; `StageRequest.body: Option<Value>`; `build_request` structured arm; executor `Probe.bodies`, `check_received`; call sites | AC1, AC6 lib (3) | `cargo val -p proxima-lib --features tokio --lib -E 'test(stage_payload_) or test(ctl_payload_)'` | 8 passed (AC1 7 + AC6 lib (3)) | [x] | 2026-11-20: 8 passed (worker `slice1.log`, independent `rerun-slice1.log`); incumbents `templates::` 13, `pipelines::` 91 (83 + 8), `load::` 10, `pipeline_pipe` 24; clippy clean on `proxima-http` (http1) and `proxima-lib` (tokio) lib + tests; no `StageRequest` call site outside `spec.rs` / `executor.rs` |
| 2 | `ProcessRpcInitFrame.send: Value`; `run_init` structured arm; `check_initialized` | AC2, AC6 lib (1), (2) | `cargo val -p proxima-lib --features tokio --lib -E 'test(process_rpc_payload_) or test(ctl_payload_)'` | 4 passed (AC2 1 + AC6 lib (1)-(3)) | [x] | 2026-11-20: 4 passed (worker `slice2.log`, independent `rerun-slice2.log`); `upstreams::process_rpc::` 21 (18 + 3); clippy clean on `proxima-lib` (tokio) lib + tests; no other `ProcessRpcInitFrame` construction in the workspace (JSON fixtures with a string `send` deserialize to `Value::String` unchanged) |
| 3 | `SynthConfig.body` / `body_template: Option<Value>`, structured arms, env as text, `#[builder(into)]`; `check_body_json` | AC3, AC6 lib (4)-(6) | `cargo val -p proxima-lib --features tokio --lib -E 'test(synth_payload_) or test(ctl_payload_)'` | 10 passed (AC3 4 + AC6 lib 6) | [x] | 2026-11-20: 10 passed (worker `slice3.log`, after review `slice3-fix.log`); `upstreams::synth::` 14 (7 + 7); clippy clean; env text through `#[setting(default, resolve_with = "parse_env_text")]` (the derive accepts it on `Option<Value>`, so the `skip` fallback was not needed); review moved inline `crate::templates::` paths to a gated import; no other `SynthConfig` construction in the workspace |
| 4 | `triage.toml` twin, `check_seen`, `json_text_leaves`, the twin tests and their JSON controls | AC4, AC6 pipe | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_payload_) or test(ctl_payload_)'` | 7 passed (AC4 3 + AC6 pipe 4) | [ ] | |
| 5 | CLI `review_ticket.toml` twin and its session test | AC5, AC6 cli | `cargo val -p proxima-cli --test integration -E 'test(mcp_config_payload_) or test(ctl_payload_)'` | 3 passed (AC5 2 + AC6 cli 1) | [ ] | |
| 6 | incumbents and lints | AC7, AC8 | AC7 (a)-(e); AC8 (C8, C8x) | 126, 31, 5, 10, 13 passed, 0 failed; C8 prints `exit=0`, `0`, `1`; C8x prints `2` | [ ] | |

## resume

Last landed slice: 3 (structured synth bodies)
Next action: slice 4 (triage TOML twin)
Open question, if any: none

## struck

-
