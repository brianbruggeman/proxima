# agentic-mcp-serve -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice. Every cargo command runs as
`timeout 1500 git wt cargo ...` in the slice's worktree, `CARGO_BUILD_JOBS=2`, load under 30.
`C` below is `/Users/brianbruggeman/.cache/proxima/agentic-mcp-serve`, which holds
`mcp-echo.json` = `{"pipe":[{"name":"echo","synth":{"status":200,"body":"ok"}}],"listen":[{"type":"mcp","transport":"stdio","bind":"127.0.0.1:0","pipe":"echo"}]}`
and `requests.jsonl` = two lines, `{"jsonrpc":"2.0","id":1,"method":"initialize"}` and
`{"jsonrpc":"2.0","id":2,"method":"tools/list"}` (written 2026-11-20).

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0a | baseline, MCP listener + `process_rpc` (PROXIMA-34) | AC11 | `cargo val -p proxima-lib --features tokio --lib -E 'test(listeners::mcp::) or test(upstreams::process_rpc::)' --output plain` | 19 passed, 0 failed | [x] | 2026-11-20 at `53d972c3`: mcp 12, process_rpc 7, wall 235 s |
| 0b | baseline, `proxima-listen` lib (PROXIMA-34) | AC14(a) | `cargo val -p proxima-listen --lib --output plain` | 124 passed, 0 failed | [x] | 2026-11-20 at `53d972c3` in clean worktree `.wt/card/proxima-35`, wall 81 s; first attempts refused by the heavy-command guard (load 82, 50) |
| 0c | today's `proxima serve` with an `mcp` stdio listener, and the L21 snapshot (PROXIMA-35) | L21 for AC5, AC8, AC12, AC13; R7's refutation condition | (1) `git wt cargo build -p proxima-cli --bins`; (2) `timeout 20 target/debug/proxima serve $C/mcp-echo.json < $C/requests.jsonl > $C/serve.out 2> $C/serve.err; echo EXIT=$?`; (3) `timeout 20 target/debug/proximad serve --mcp-stdio --state-dir $(mktemp -d) < $C/requests.jsonl > $C/proximad.out`, then line 2's `result.tools` written to `tests/fixtures/mcp_stub/incumbent_tools.json` | (2): the exit code, the line count of `serve.out`, and how many of its lines parse as JSON-RPC; (3): `jq length` of the fixture = 17 | [x] | 2026-11-20, base binaries built in `.wt/card/proxima-35` (2m 17s): (2) EXIT 1, 2 stdout lines, both JSON-RPC answers, stderr `listener did not come up at 127.0.0.1:63971 within 5s` (the probe, not stdout corruption; see the refutation condition); (3) 17 descriptors, identical to (2)'s `tools/list`; logs in `<tickets>/PROXIMA-35/evidence/` |
| 1 | `process_rpc` `init` frames, label set in `RpcState` | AC1, AC2, AC4 | `cargo val -p proxima-lib --features tokio --lib -E 'test(process_rpc_init_) or test(ctl_rpc_)'` | 11 passed (AC1 3 + AC2 3 + AC4 5); AC11 19; AC17 0 | [x] | 2026-11-20 PROXIMA-36: 3/3/5 in gate1.log (run with slice 2, 31 total, wall 15 s), AC11 19, clippy `-D warnings` clean, AC17 0; logs `<tickets>/PROXIMA-37/evidence/` |
| 2 | MCP listener `tools` from its spec: list, call, refusals | AC5-AC8 | `cargo val -p proxima-lib --features tokio --lib -E 'test(mcp_tools_) or test(ctl_tools_)'` | 20 passed (AC5 1 + AC6 3 + AC7 8 + AC8 8); AC11 19; AC17 0 | [x] | 2026-11-20 PROXIMA-37: 1/3/8/8 in gate1.log (run with slice 1, 31 total), AC11 19, clippy `-D warnings` clean, AC17 0; L21 = `tests/fixtures/mcp_stub/incumbent_tools.json` from base `proximad` (PROXIMA-35) |
| 3 | an `mcp` listener runs one lane and reports no bind address | AC9, AC10 | `cargo val -p proxima-listen --lib -E 'test(mcp_lane_) or test(ctl_lanes_)'` | 5 passed (AC9 2 + AC10 3); AC14(a) unchanged; AC17 0 | [x] | 2026-11-20 PROXIMA-38: 5 passed (gate-a.log, wall 28 s), AC14(a) 124 (gate-b.log), clippy `-D warnings` clean, AC17 0 over 106 new fns; logs `<tickets>/PROXIMA-39/evidence/` |
| 4 | the review-ticket MCP fixture served end to end by `proxima serve` over stdio | AC12, AC13 | `cargo val -p proxima-cli --test integration -E 'test(mcp_config_tool_) or test(ctl_serve_)'` | 5 passed (AC12 1 + AC13 4); AC14(b) 1; AC15 0; AC16 5 and 1; AC17 0 | [ ] | |

## resume

Last landed slice: none (spec round 4; PROXIMA-33 rounds 1-3 REFUSE, amended)
Next action: 0b and 0c when load is under 30, write their numbers here and into AC14(a), audit round 3
Open question, if any: none

## struck

- slice-1 AC3 / R3 (labeled spawn error): unproducible on unix, `proxima-process/src/spawn.rs:327-328,746-748`; round 2
