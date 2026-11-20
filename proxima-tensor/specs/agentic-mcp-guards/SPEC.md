# agentic-mcp-guards -- a branch predicate on pipeline stages

status: audited -- ADMIT by spec-auditor round 7, 2026-11-20 (PROXIMA-42; rounds 1-6 refused 10, 5, 5, 3, 1, 1 reasons, each amended)
owner: brian
created: 2026-11-20

Gap 3 of `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md`, inside the owner's
boundary of 2026-11-20 (same file, section "boundary"): proxima carries one-shot ops; the agent
cycle (pick the next tool, loop, keep state) is ragd's. A guard is an op: a pure predicate over one
run's stage scope that decides whether a stage runs. Every `file:line` below is in
`~/repos/mine/proxima` at `ca128dd3` (main, on origin), read 2026-11-20 (PROXIMA-40 inventory).
`<tickets>` is
`/Users/brianbruggeman/repos/obsidian/slot-0/slot-0/20 - Collaboration and Records/22 - Investigations/22.1 - Tickets`.
The cargo package is `proxima-lib`; `src/pipelines` is `#[cfg(feature = "tokio")]`.

## problem

A pipeline built from config cannot take one path or another on an earlier stage's answer:
every stage runs once its parents complete, a stage that does not run is `Skipped`, and
`failure_reason` turns any `Skipped` stage into a failed run (`src/pipelines/executor.rs:147-165,
233-245`), so "fetch the PR only when the ticket has one, else summarize the ticket" cannot be
written -- measured by the triage fixture answering 200 with the review on a ticket with a PR and
200 with the summary on a ticket without one, the PR pipe called 0 times on the second.

## refutation condition

If the triage fixture cannot branch without a new stage kind, a new engine, a loop, or state
carried between calls (the cycle the boundary assigns to ragd), this op is the wrong shape and the
spec is struck.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | a stage (pipe or process) accepts `when`: a CEL expression. The executor never compiles CEL: it evaluates only compiled guards supplied by `with_guards`, and a run whose spec has a stage with `when` and no supplied guard is refused at run start, naming the stage and containing `guard`, before any stage runs. The compile chain is closed: `Program::compile` appears once in `src/`, inside `compile_expression` (`src/scenarios/cel.rs`); `compile_expression` is called only by the loader's `compile_guards` and by the scenarios' incumbent `evaluate`; `compile_guards` is called only by `build_pipeline`; the executor evaluates guards only through `evaluate_guard(&Program, ..)` and never calls the scenarios' `evaluate`. So each pipe compiles its guards once, when it is built, and reuses them on every call. A compile error refuses the load, naming the stage and containing `when` | yes |
| R2 | after a stage's parents settle and before its pipe or process runs, its `when` is evaluated with variables `input` (the pipeline input; CEL reserves `in`) and one variable per parent that produced an output, named by the parent's stage name and valued by that output. `true` runs the stage; `false` makes it `Declined`, and its pipe or process receives 0 calls | yes |
| R3 | `Declined` is not a failure: `failure_reason` gives it none, so a run whose only non-completed stages are `Declined` is `Completed`. A stage with a `Declined` parent is `Declined`, decided before its own `when`. A stage with a failed, skipped or nonzero parent stays `Skipped` as today, and that wins over a `Declined` parent | yes |
| R4 | a `when` that evaluates to an error or to a non-bool fails its stage (`Failed`) with a reason naming the stage and containing `when`, and the run fails as today | yes |
| R5 | the pipeline `output` is a stage name (as today, with today's answers, `load.rs:974-981`) or a list of names. For a list: a failed run answers as today, 502 with its failure reason (`load.rs:978`), whatever any listed stage produced; a completed run answers with the output of the first listed stage, in list order, that produced one, or, when none did, 502 with body exactly `no output stage ran: <names joined by ", ">`; a listed name that is not a stage refuses the load with an error containing that name | yes |
| R6 | a guard keeps no state between calls: one loaded pipe answers a with-PR input and then a without-PR input each with its own branch | yes |
| R7 | incumbents unchanged: every `pipelines::` unit test and every `pipeline_pipe` test passes with its slice-0 count; pipelines without `when` behave exactly as before; no new `impl ... PipeFactory for` outside `tests/` | yes |

## literals

Each line is an id, one space, then the literal to the end of the line.

```
G1 {"repo":"brianbruggeman/proxima","pr":42}
G2 {"repo":"brianbruggeman/proxima","pr":null}
G3 {"title":"split stage body","body":"adds StageBody"}
G4 {"content":[{"type":"text","text":"LGTM with one nit"}]}
G5 {"content":[{"type":"text","text":"no PR yet; ticket only"}]}
G6 ticket.pr != null
G7 ticket.pr == null
G8 {"content":[{"text":"LGTM with one nit","type":"text"}]}
G9 {"content":[{"text":"no PR yet; ticket only","type":"text"}]}
G10 ticket.pr !=
G11 ticket.missing == 1
G12 ticket.repo
G13 no output stage ran: review, summary
G14 stage `pr` failed: stage `pr` pipe `github`: returned status 404
```

G14 is `failure_reason`'s `Failed` arm (`executor.rs:240-243`) around the stage's own reason from
`pipe_stage_output` (`:594-596`) and `call_pipe` (`:642-643`).

G8 / G9 are G4 / G5 re-serialized by `load.rs:976` (`serde_json` without `preserve_order`,
`Cargo.toml:225,515`), keys sorted.

## architecture

Reuse map: the DAG executor and its strict-mode skip (`executor.rs:28-33,546-562`); the stage
scope (`build_scope`, `:599-608`); load-time checks (`check_pipeline`, `load.rs:899-936`); CEL via
`cel-interpreter` 0.10.0, already a non-optional dependency (`Cargo.toml:325`), used by
`src/scenarios/cel.rs:13`; JSON values bind through its `impl<T: Serialize> TryIntoValue`
(`cel-interpreter-0.10.0/src/objects.rs:147-151`); counting stub pipes and the loader rig in
`tests/pipeline_pipe.rs`; executor test helpers `pipe_stage`, `stub`, `run_pipes`
(`executor.rs:960-1009`).

```rust
// src/pipelines/spec.rs: StageSpec and RawStageSpec gain
#[serde(default, skip_serializing_if = "Option::is_none")] pub when: Option<String>,

// src/pipelines/executor.rs
enum StageCompletion { Pending, Skipped, Failed, Declined, Completed { exit_code: Option<i32> } }
impl PipelineExecutor { pub fn with_guards(self, guards: BTreeMap<String, Arc<Program>>) -> Self; }
// the executor has no compile path: a `when` without a supplied guard refuses the run (R1), so
// compile-once is structural -- the only compile call is in build_pipeline

// src/scenarios/cel.rs stays the one CEL module: `compile_expression(&str) -> Result<Program, _>`
// (the only `Program::compile`; the incumbent `evaluate` calls it) and
// `evaluate_guard(&Program, &[(&str, &serde_json::Value)]) -> Result<bool, String>`

// src/load.rs: build_pipeline compiles each `when` once (refusing errors by stage) and passes
// with_guards; `output` parses as a string or a list; render_pipeline_report picks the first
// listed output or answers with G13's form
```

Fixture `tests/fixtures/pipeline_pipe/triage.json`: pipes `tracker`, `github`, `llm`, `brief`,
each a counting stub of the kind `tests/pipeline_pipe.rs` already registers (fixed routes matched
by exact path, 404 otherwise, `tests/pipeline_pipe.rs:47-65`), with routes: `tracker`
`/tickets/PROXIMA-24` -> G1 and `/tickets/PROXIMA-25` -> G2; `github`
`/repos/brianbruggeman/proxima/pulls/42` -> G3; `llm` `/v1/messages` -> G4; `brief` `/v1/summary`
-> G5. Stages: `ticket` (tracker, GET `/tickets/{{body.in.ticket}}`); `pr` (github, depends
`ticket`, `when` G6, GET `/repos/{{body.ticket.repo}}/pulls/{{body.ticket.pr}}`); `review` (llm,
depends `ticket` and `pr`, POST `/v1/messages`, body `{"pr":"{{body.pr.body}}"}`); `summary`
(brief, depends `ticket`, `when` G7, POST `/v1/summary`, body `{"ticket":"{{body.in.ticket}}"}`);
`output` `["review", "summary"]`. Request body `{"ticket":"PROXIMA-24"}` takes the PR branch,
`{"ticket":"PROXIMA-25"}` the summary branch.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| guard shape | a `when` field on a stage | a predicate pipe kind adds a registry kind, and every predicate pipe today answers false with an error or a rejection response (`proxima-primitives/src/pipe/filter.rs:50-56`, `validate.rs:200`), which the executor already turns into a failed run |
| new completion state | `Declined`, separate from `Skipped` | `Skipped` means "a parent failed" and counts as failure (`executor.rs:239`); reusing it would either hide real failures or keep branches failing |
| a declined parent's child | `Declined` too | running it without the parent's output fails its strict templates (`{{body.pr.body}}`) as a missing placeholder |
| variable for the pipeline input | `input` | CEL reserves `in` as an operator |
| where a guard is compiled | once, in `build_pipeline`, kept as `Arc<Program>` (`Program` is not `Clone`) | compiling per call is the defect gap 3 names (`src/scenarios/cel.rs:14`) |
| boundary | a guard sees only this run's scope and returns a bool | looping, choosing a next tool, or keeping state across calls is ragd's (owner, 2026-11-20) |
| test names | positive prefixes `guard_runs_`, `guard_declines_`, `guard_eval_fails_`, `guard_compiled_` (executor unit tests); `pipeline_guard_branch_`, `pipeline_guard_output_`, `pipeline_guard_refuses_` (`tests/pipeline_pipe.rs`); control prefix `ctl_branch_` in both files. No prefix contains another and no new fn name contains two (AC11 counts violations) | nextest's `test()` matches substrings |

## acceptance criteria

Runner `cargo val`, always with `--lib` or `--test`, `--features tokio` unless stated, each run
wrapped `timeout -k 30s 25m`, `CARGO_BUILD_JOBS=2`, one build at a time. Each check helper is
called by its AC and by a control expecting Ok and one expecting Err.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R2 | `cargo val -p proxima-lib --features tokio --lib -E 'test(guard_runs_)'` | 2 passed, through `check_branch(report, calls, expected)` (expected per-stage completion, per-stub call count, run outcome): (1) stub `ticket` answers G1, stage `pr` (`when` G6) -> `pr` Completed, its stub 1 call, run Completed; (2) a process stage with `when` `input.go == true` and input `{"go":true}` -> Completed exit 0 |
| AC2 | R2, R3 | `cargo val -p proxima-lib --features tokio --lib -E 'test(guard_declines_)'` | 7 passed, through `check_branch`: (1) `ticket` answers G2, `pr` `when` G6 -> `pr` Declined, its stub 0 calls, run Completed; (2) the same plus `review` depending on `pr` with its own `when` G11 (which would fail if evaluated) -> `review` Declined, not Failed, its stub 0 calls, run Completed; (3) stage `join` with `depends_on` `["pr", "broken"]` where `pr` is Declined (G2 + G6) and `broken` Failed (its stub answers 500) -> `join` Skipped, run Failed; (4) the same with `depends_on` `["broken", "pr"]` -> `join` Skipped, run Failed; (5) a process stage `touch <tempdir>/ran` with `when` `input.go == true` and input `{"go":false}` -> Declined, `<tempdir>/ran` absent afterwards, run Completed; (6) `join` with `depends_on` `["pr", "after"]` where `pr` is Declined and `after` is Skipped (it depends on `broken`, which fails) -> `join` Skipped, run Failed; (7) `join` with `depends_on` `["pr", "exit3"]` where `pr` is Declined and `exit3` is a process stage `exit 3` -> `join` Skipped, run Failed |
| AC3 | R4 | `cargo val -p proxima-lib --features tokio --lib -E 'test(guard_eval_fails_)'` | 3 passed: (1) `ticket` = G1, `pr` `when` G11 -> `pr` Failed and `report.failures["pr"]` (the stage's own reason, not the run outcome) starts with ``stage `pr` when``; run Failed; (2) `when` G12 -> the same, and that reason also contains `bool`; (3) a process parent `prep` (`true`, exit 0, no output) and a child with `when` `prep == null` -> the child Failed with a reason starting ``stage `child` when`` (a parent without an output is not bound, so `prep` is an undeclared reference; bound as null, the guard would be true and the child would run) |
| AC12 | R1 | `cargo val -p proxima-lib --features tokio --lib -E 'test(guard_compiled_)'` | 2 passed: (1) `ticket` + `pr` (`when` G6) run with no guards supplied -> the run is refused, the error contains ``stage `pr` `` and `guard`, every stub 0 calls; (2) `pr`'s `when` string set to G10 (unparseable) with a compiled G7 guard supplied for `pr`, `ticket` answering G1 -> `pr` Declined (the supplied guard ran and the string was never compiled) |
| AC4 | R1, R5 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_guard_refuses_)'` | 2 passed, every stub's call count 0: (1) the triage fixture with `pr`'s `when` set to G10 -> the load fails, the error contains ``stage `pr` `` and `when`; (2) the triage fixture with `output` `["review", "sumary"]` -> the load fails, the error contains `sumary` |
| AC5 | R2, R3, R5, R6 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_guard_branch_)'` | 1 passed, one loaded triage pipe called twice, through `check_triage(response, calls, expected)`: (a) body `{"ticket":"PROXIMA-24"}` -> 200, body bytes exactly G8, calls tracker 1, github 1, llm 1, brief 0; (b) the same pipe, body `{"ticket":"PROXIMA-25"}` -> 200, body bytes exactly G9, cumulative calls tracker 2, github 1, llm 1, brief 1 |
| AC6 | R5 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_guard_output_)'` | 3 passed: (1) the triage fixture with `summary`'s `when` set to `false`, body `{"ticket":"PROXIMA-25"}` -> 502, body exactly G13; (2) the triage fixture with `summary`'s `when` removed (both branches produce on `{"ticket":"PROXIMA-24"}`) -> 200, body bytes exactly G8 (`review` is listed first); (3) the same with `output` `["summary", "review"]` -> 200, body bytes exactly G9 (list order, not stage order, decides) |
| AC13 | R1 (closed compile chain) | in the slice worktree, with `hdr() { git grep -n -p -E "$1" -- src/ \| awk -v def="fn $2" '/=[0-9]+=/{h=$0; next} index($0, def){next} {print h}' \| grep -v '=mod tests {'; }` printing the enclosing-function header of every NON-TEST line matching `$1`, except the definition of `$2` (each file's test module is a column-0 `mod tests {`, so a call inside it reports that header and is dropped; `src/load.rs:1212`, `src/pipelines/executor.rs:725`, `src/scenarios/cel.rs:76`). cel-interpreter 0.10.0 builds a `Program` only by `Program::compile` (`lib.rs:153`) or `TryFrom<&str>` (`lib.rs:180`). In this cell `\|` is only the markdown escape for a shell pipe `|`; no regex here contains `|`. (a1) `hdr 'Program::compile' compile_expression`; (a2) `hdr 'Program::try_from' compile_expression`; (b) `hdr 'compile_expression\(' compile_expression`; (c) `hdr 'compile_guards\(' compile_guards`; (d) `git grep -c 'cel::evaluate(' -- src/pipelines src/load.rs \| wc -l`; (e) `git grep -n -p -e try_into -e try_from -- src/pipelines src/load.rs \| awk '/=[0-9]+=/{h=$0; next} {print h}' \| grep -v '=mod tests {' \| wc -l` | (a1) exactly 1 line, containing `fn compile_expression`; (a2) 0 lines; (b) exactly 2 lines, one containing `fn compile_guards`, one containing `fn evaluate`; (c) exactly 1 line, containing `fn build_pipeline`; (d) `0`; (e) `2` (unchanged from base: both are `RawStageSpec`'s serde conversion, `src/pipelines/spec.rs:28,108`; a `try_into()` into a `Program` in non-test pipeline or loader code would make it 3). Controls, measured 2026-11-20 on the base tree: `hdr 'check_output_stage\(' check_output_stage` -> 1 line containing `fn check_pipeline` (Ok); `hdr 'run_pipes\(' run_pipes` -> 0 lines (a test-only helper is dropped, Ok); (a1) on base -> 1 line containing `fn evaluate`, not `fn compile_expression` (Err); (c) on base -> 0 lines (Err); `git grep -c 'cel::evaluate(' -- src/scenarios` -> `src/scenarios/orchestrator.rs:1` (Ok, the pattern finds a real caller); `printf 'let guard: Program = text.try_into()?;\n' \| grep -c -e try_into -e try_from` -> 1 (Err, (e)'s patterns catch the `TryFrom` route; measured 2026-11-20, and the same two patterns count 2 in `src/pipelines/spec.rs` at `ca128dd3`) |
| AC7 | R2-R5 controls | `cargo val -p proxima-lib --features tokio --lib -E 'test(ctl_branch_)'` and `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(ctl_branch_)'` | lib 3 passed: `check_branch` on a hand-built report matching AC2(1) exactly -> Ok; on the same report with `pr` Skipped instead of Declined -> Err; on it with outcome Failed -> Err. Integration 2 passed, with the counting stubs `tests/pipeline_pipe.rs` already uses (they route by exact path and answer 404 otherwise, `tests/pipeline_pipe.rs:58-65`; `github` answers G3 only on `/repos/brianbruggeman/proxima/pulls/42`): (a) the triage fixture with both `when` fields removed, one pipe called with AC5's two request bodies in AC5's order -> the second call is 502 with body exactly G14 (R5: a failed run answers with its first failure in topological order, `executor.rs:147-165`, and `pr` precedes `review`; a null `pr` renders as an empty path segment, `proxima-http/src/templates.rs:89,141,147`, so `pr` requests `/repos/brianbruggeman/proxima/pulls/`, gets 404 and fails), cumulative calls tracker 2, github 2, llm 1, brief 2, and `check_triage` with AC5(b)'s expectation on it -> Err (only the routing differs from AC5); (b) `check_triage` on a hand-built 200 / G9 / AC5(b)-count response -> Ok |
| AC8 | R7 | `cargo val -p proxima-lib --features tokio --lib -E 'test(pipelines::) and not (test(guard_) or test(ctl_branch_))'` | 66 passed, 0 failed (PROXIMA-41 at `ca128dd3`, 2026-11-20: control_plane 5, dag 5, executor 20, explain 4, fs_control_plane 7, http_routes 9, replay 2, spec 14; `<tickets>/PROXIMA-41/evidence/baseline-pipelines.log`) |
| AC9 | R7 | (a) `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'not (test(pipeline_guard_) or test(ctl_branch_))'`; (b) `cargo val -p proxima-lib --test pipeline_pipe` (default features) | (a) 16 passed; (b) 3 passed; 0 failed (PROXIMA-41 at `ca128dd3`, `baseline-pipeline-pipe.log`, `baseline-pipeline-pipe-notokio.log`) |
| AC10 | R7 | in the slice worktree: `git add -A -N && git diff ca128dd3 -- '*.rs' ':!:tests/*' ':!:*/tests/*' \| grep -c -E '^\+.*impl.*PipeFactory for'; git reset -q` | 0. Control: `git show 3a8118db -- '*.rs' ':!:tests/*' ':!:*/tests/*' \| grep -c -E '^\+.*impl.*PipeFactory for'` -> 5 (measured 2026-11-20) |
| AC11 | R7 (each count selects only its AC) | in the slice worktree: `git add -A -N && git diff ca128dd3 -- src/pipelines/executor.rs tests/pipeline_pipe.rs \| grep -oE '^\+[[:space:]]*(async )?fn [a-z0-9_]+' \| awk '{print $NF}' \| while read -r name; do hits=0; for prefix in guard_runs_ guard_declines_ guard_eval_fails_ guard_compiled_ pipeline_guard_branch_ pipeline_guard_output_ pipeline_guard_refuses_ ctl_branch_; do case "$name" in *"$prefix"*) hits=$((hits+1));; esac; done; [ "$hits" -gt 1 ] && echo "$name"; done \| wc -l; git reset -q` | 0. Control: the loop fed `ctl_branch_guard_runs_x` prints 1 line; fed `ctl_branch_x` prints 0 |

## out of scope

- loops, retries driven by results, choosing the next tool, memory across calls: ragd's cycle
- per-item fan-out over a list result
- a `when` that reads stages other than the stage's parents (it binds what the template scope binds)
- `settle` turning a `run_stage` error into `Skipped` and dropping its reason (`executor.rs:519`,
  PROXIMA-40): a defect with no current input that produces it; its own card

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| JSON null binds through cel's `TryIntoValue` as something other than CEL `null` | medium | G6 / G7 never match | AC2(1) and AC5(b) assert the null branch directly |
| `Program` is not `Send` / `Sync` | low (auto traits over an AST of Box/Vec/String, PROXIMA-32) | `Arc<Program>` cannot cross the executor's tasks | the first slice compiles it; compile once per run as the fallback |
| shared-machine load trips the heavy-command guard | high | runs refused | coordinate quiet windows with peer sessions, as on 2026-11-20 |
| a spec submitted through the control plane (`pipelines_submit`) carries `when` | low | refused at run start (no compiled guard, R1) | intended: guards are supported on loader-built pipelines; the control plane compiling guards is a later op if wanted |

## context

- `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md` -- owner intent, gaps, the boundary
- `proxima-tensor/specs/agentic-mcp/SPEC.md`, `proxima-tensor/specs/agentic-mcp-serve/SPEC.md` -- the
  pipeline-as-a-pipe and MCP-serving specs this builds on
- `src/pipelines/executor.rs` (completion states, `fold_outcome`, `failure_reason`,
  `parents_succeeded`, test helpers), `src/pipelines/spec.rs` (`RawStageSpec`, `TryFrom`),
  `src/load.rs:850-981` (`build_pipeline`, `check_pipeline`, `render_pipeline_report`)
- `src/scenarios/cel.rs` -- the existing CEL module; `cel-interpreter-0.10.0` sources for binding
- PROXIMA-40 inventory, `<tickets>/PROXIMA-40/`
