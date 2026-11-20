# agentic-mcp-guards -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice. Every cargo command runs as
`timeout -k 30s 25m git wt cargo ...` in the slice worktree, `CARGO_BUILD_JOBS=2`, admitted by the
heavy-command guard (load at most 30).

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | baselines at `ca128dd3` (PROXIMA-41) | AC8, AC9 | `cargo val -p proxima-lib --features tokio --lib -E 'test(pipelines::)' --output plain`; `cargo val -p proxima-lib --features tokio --test pipeline_pipe --output plain`; `cargo val -p proxima-lib --test pipeline_pipe --output plain` | 66, 16, 3 passed; 0 failed | [x] | 2026-11-20 PROXIMA-41 at `ca128dd3`; logs `<tickets>/PROXIMA-41/evidence/` |
| 1 | `when` parsed on stages, compiled once, `Declined` state, executor evaluation (CEL `compile` + bool eval in `src/scenarios/cel.rs`) | AC1, AC2, AC3, AC12, AC7 (lib) | `cargo val -p proxima-lib --features tokio --lib -E 'test(guard_) or test(ctl_branch_)' --output plain` | 17 passed (AC1 2 + AC2 7 + AC3 3 + AC12 2 + AC7 lib 3); AC8 66; AC11 0 | [x] | 2026-11-20 PROXIMA-43: 2/7/3/2/3 (slice1-rename.log, after AC11 caught `guard_compiled_supplied_guard_runs_...` matching `guard_runs_`; renamed), AC8 66, clippy `-D warnings` clean with and without tokio, AC11 0; logs `<tickets>/PROXIMA-43/evidence/` |
| 2 | loader compiles guards at build, `output` list, the triage fixture | AC4, AC5, AC6, AC7 (integration), AC13 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_guard_) or test(ctl_branch_)' --output plain`, then AC13's six checks (a1), (a2), (b), (c), (d), (e) | 8 passed (AC4 2 + AC5 1 + AC6 3 + AC7 integration 2); AC9 16 and 3; AC10 0; AC11 0; AC13 (a1) 1 line naming `fn compile_expression`, (a2) 0 lines, (b) 2 lines naming `fn compile_guards` and `fn evaluate`, (c) 1 line naming `fn build_pipeline`, (d) 0, (e) 2 | [x] | 2026-11-20 PROXIMA-44 / PROXIMA-46: refuses 2, branch 1, output 3, ctl_branch 2 (whole `pipeline_pipe` 24 with tokio, 3 without; final-pipe*.log), lib 113 (scenarios 30 + pipelines 66 + guards 17), clippy both builds clean, AC13 (a1) `fn compile_expression`, (a2) 0, (b) `fn compile_guards` + `fn evaluate`, (c) `fn build_pipeline`, (d) 0, (e) 2, AC11 0. G10 first panicked inside cel-interpreter 0.10's parser; `catch_unwind` was rejected (release profile aborts) and the workspace moved to `cel` 0.16 in its own commit (4e1b3588, PROXIMA-45 probe) |

## resume

Last landed slice: 2 (all slices green on card/proxima-42, 2026-11-20)
Next action: rebase onto main, landing gate, ff-merge, push
Open question, if any: none

## struck

-
