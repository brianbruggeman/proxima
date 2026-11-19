# agentic-mcp -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice. Card ids start at
PROXIMA-24 (the store's max is 19, but `/Users/brianbruggeman/repos/slot-0/pm/topics/mcp-daemon.md`
already uses PROXIMA-20..23). Package is `proxima-lib`. Base `dd60e28c` (`src/` identical
at `f42b714a`). Every cargo command carries a target flag (cargo-val runs doc-tests
otherwise), `src/pipelines` needs `--features tokio`, and filtersets use `or` (a hook
blocks `|` inside cargo commands; nextest treats them alike). Worktrees per the repo's
AGENTS.md: `git wt add`, `git wt cargo`; check `git for-each-ref refs/heads | wc -l`
before and after (`git wt cargo` exports GIT_DIR to tests).

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 0 | record the AC7 baseline, no code (PROXIMA-25) | AC7, AC9 | the AC7 command (with its exclusions); the AC9 commands | 43 passed, 0 failed, 0 doc-tests; AC9 prints 5, 1, 1 | [x] | 43 passed / 0 failed / 0 doc-tests at `f42b714a` (8 modules). Measured with `-E 'test(pipelines::)'`; at base that equals the AC7 command because 0 of the 43 names contain an exclusion substring and 0 new-prefix names exist in `src/` (audit 4). AC9 = 5 / 1 / 1. An earlier run without `--features tokio --lib` counted 21 doc-tests and 0 pipelines tests: invalid, kept as `evidence/baseline.log` |
| 1 | `StageBody` enum + `StageRequest`; parsing refuses both/neither (`try_from = "RawStageSpec"`); update the 8 construction sites and the field reads; AC1 asserts full parsed values and literal refusals; `ctl_parse_` controls (PROXIMA-26) | AC1, AC7, AC8, AC16 | `cargo val -p proxima-lib --features tokio --lib -E 'test(stage_body_) or test(pipelines::)'`; the AC7 command; the AC8 command | 51 passed (43 + 4 `stage_body_` + 4 `ctl_parse_`, all inside `pipelines::spec`), 0 failed; AC7 43 passed; AC8 prints 0 | [x] | LANDED on local main 2026-11-19: `40d41bc9 feat(pipelines): stage body is a process or a registry pipe` (main fast-forwarded from `f42b714a`, read back equal; 8 files +274/-59). `git push origin main` refused by the auto-mode permission classifier; origin still `f42b714a`. worktree `.wt/card/proxima-26` (uncommitted). First measurement 2026-11-19, shared main target, before audit 8's additions: `passed=49 failed=0`, AC7 `passed=43 failed=0`, AC8 0, clippy EXIT 0 (logs `22.1 - Tickets/PROXIMA-26/evidence/`). Revalidated 2026-11-19 under `git wt cargo` after review fixes (`StageRequest` `Default` derive removed: it yielded `""` vs serde `GET`/`/` and had no caller; AC1 refusal literals `fetch-ticket`; 4 `ctl_parse_`): `passed=51 failed=0`, AC7 `passed=43 failed=0`, check and clippy EXIT 0, refs 179 -> 179, branch tip `f42b714a` unmoved, +274/-59 (logs `val-1b.log`, `val-ac7b.log`). VALIDATED, UNCOMMITTED: the commit waits on owner authorization |
| 2 | `expand_strict` in `proxima_http::templates`, same resolver as `expand`, plus the incumbent controls (PROXIMA-27) | AC14, AC15 | `cargo val -p proxima-http --lib -E 'test(expand_strict_) or test(ctl_expand_)'` | 5 passed (3 + 2), 0 failed | [x] | `passed=5 failed=0`; all of proxima-http `passed=111 failed=0`; clippy EXIT 0 (`22.1 - Tickets/PROXIMA-27/evidence/val-2.log`, `val-full-2.log`, `clippy-2.log`). `expand` and `expand_strict` share one scanning loop (`expand_with` 9 + `expand_placeholders` 19 + `opens_placeholder` 7 lines; every fn in the file is 19 lines or fewer). `expand` behaviour is unchanged: its 8 pre-existing tests pass |
| 3 | executor runs pipe stages: `with_pipes`, `run_with_input`, scope from parent outputs, output capture, non-2xx -> failed, plus the controls | AC2, AC3, AC10, AC11 | `cargo val -p proxima-lib --features tokio --lib -E 'test(stage_template_) or test(stage_output_) or test(ctl_template_) or test(ctl_output_)'` | 15 passed (3 + 3 + 3 + 6), 0 failed | [x] | `passed=15 failed=0`; AC7 `passed=43`; all `pipelines::` `passed=66 failed=0`; clippy EXIT 0; `rustfmt --check` EXIT 0 (`22.1 - Tickets/PROXIMA-28/evidence/val-2.log`, `val-ac7-2.log`, `val-all-2.log`). Outputs ride a per-stage `watch` beside the unchanged `Copy` completion channel; failures land in `PipelineRunReport.failures`. The first pass had `execute` at 195 lines; the split brought every non-test fn to 20 or fewer, using two private parameter bundles (`RunPlan`, `StageTask`). Under AGENTS.md's second type question they are relocations; they were kept to satisfy the 20-line rule (owner's call) |
| 4 | create `tests/fixtures/pipeline_pipe/review_ticket.json` and `tests/pipeline_pipe.rs` with its counting stub factories (AC4's stub replies); loader key `pipeline` -> `build_pipeline` under `tokio` (pipes via `build_pipe`, validate at load, `PipeHandle` 200/502); named refusal without `tokio`; the load and tokio controls | AC5, AC6, AC13, AC17, AC18 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_pipe_failure_) or test(pipeline_pipe_load_refuses_) or test(ctl_load_)'`; `cargo val -p proxima-lib --test pipeline_pipe -E 'test(pipeline_pipe_refused_without_tokio) or test(ctl_tokio_)'` | 10 passed (1 + 3 + 6), 0 failed; 3 passed (1 + 2), 0 failed | [ ] | |
| 5 | the review-ticket fixture end to end over the slice-4 stubs, plus its controls | AC4, AC7, AC8, AC12 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_pipe_review_ticket) or test(ctl_review_) or test(ctl_failure_)'`; the AC7 command; the AC8 command | 6 passed (1 + 5), 0 failed; AC7 43 passed; AC8 prints 0 | [ ] | |

## resume

Last landed slice: 0 (baseline, no code). Slice 1 code exists uncommitted in `.wt/card/proxima-26`.
Audits 1-8 (PROXIMA-24) refused, each amended; the history is in this file's struck list and SPEC.md. Audit 8 (R 8 / AC 16): G1-G3 and G5-G7 pass; G4 refused AC1 cases 3-4 (no input literal or name) and AC6 (offenders unnamed, no control). Amended: AC1 refusal literals (`fetch-ticket`); AC16 grows to 4 (refusal-helper controls); AC6 gives three named fixture mutations with the incumbent formats (`spec.rs:56-57`, `dag.rs:66-67`); new AC17 `ctl_load_` (incumbent errors -> Ok, name-nothing / name-everything -> Err); AC13 asserts the literal refusal text. Audit 9 (R 8 / AC 17) confirmed the cycle case by hand (`["pr", "review"]`) and every slice sum, and listed all remaining gaps at once: AC17(d) could not test absence; AC13 had no control (my claim that it did was false); AC14 had no known-bad; AC4 and AC10(b) lacked literals; AC2(3) had no known-good; slice 4 depended on a fixture slice 5 created. Amended: AC17 has 6 controls (absence known-bad (d)(f), known-good (e)); new AC18 `ctl_tokio_` (named refusal -> Ok, the incumbent `load.rs:671` fallthrough -> Err); AC15 adds the lenient `Ok("")` known-bad; AC4 gives every stub reply and the expected bytes; AC10 gives `/repos//pulls` and a known-good (c); slice 4 now creates the fixture and stubs. Sums: slice 2 = 5, slice 3 = 15, slice 4 = 10 + 3.
Audit 11: ADMIT (G1-G7 pass, R 8/8, AC 18/18). Its two non-gate notes were closed: R4 now names compact `serde_json::to_vec` (AC4's bytes depend on it), and AC5's github stub fails only the `pr` path, so the reported stage is `pr` whatever the declaration order.
Slice 1 (PROXIMA-26): landed on local main `40d41bc9`; push refused by the permission classifier. Slice 2 (PROXIMA-27): validated, split of `expand_with` (23 -> <=20 lines) in flight.
Audit 10 (R 8 / AC 18): per-AC sweep found no missing literal or control except AC4, whose body literal was unproducible. This serde_json has no `preserve_order` (`Cargo.toml:225,515`), so a parsed-then-serialized reply comes out key-sorted, and AC12's known-good would have accepted raw forwarding. Amended: AC4 expects the sorted bytes `{"content":[{"text":"LGTM with one nit","type":"text"}]}` plus the parsed value; AC12 grows to 5 (known-good through parse + serialize, raw-forward bytes -> Err); the refutation names the slice-4 fixture. Slice 5 = 6.
Next action: spec-auditor round 11
Open question, if any: none

## struck

- (from the capture) a new JSON step format, CSR plan tables, a new scheduler, new step
  kinds `call`/`extract`/`each`/`sample`/`when`: each duplicated an existing piece; owner
  2026-11-19 "you shouldn't be building a new fucking thing"
- a registry factory kind `pipeline`: violates R7 and cannot build nested specs without
  `LoadContext` (see SPEC decisions)
- AC8 as a count of `fn name(&self)` impls (147 at `dd60e28c`): spans ~10 traits, so a
  test stub changes it without breaking R7 and an add+remove leaves it equal; replaced by
  the diff grep + AC9 control
- AC8 pathspec `src 'proxima-*/src'`: the second pathspec matched 0 files (audit 2)
- every AC command without `--features tokio` and a target flag: measured 0 pipelines
  tests and counted doc-tests instead
- AC10 as `expand_strict` agreeing with `expand`: true by construction (same resolver),
  so it controlled nothing (audit 3)
- R1's "`PipelineSpec::validate` refuses both/neither": the enum cannot represent them;
  the refusal is at deserialize
- AC checks that assert only a type, an `Ok`, or "contains the name" without a literal:
  audits 5-8 refused each; every AC now names its exact expected value
