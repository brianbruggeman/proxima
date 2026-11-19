# agentic-mcp -- a multi-step tool built from pure config

status: audited -- ADMIT by spec-auditor round 11, 2026-11-19 (PROXIMA-24; rounds 1-10 refused, each amended)
owner: brian
created: 2026-11-19

Capture and owner intent: `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md`.
Spine `[agentic-mcp]`, goal `agentic-control-plane:c5`. All other `file:line` below are
in this repo (`~/repos/mine/proxima`, base `dd60e28c`), read 2026-11-19. The cargo
package is `proxima-lib` (`Cargo.toml:279`); `proxima` is only its lib name (`:291`).

## problem

Someone who wants a multi-step tool (fetch a ticket, then its PR, files and notes in
parallel, then one model call writes a review) cannot define it from config alone,
because a pipeline stage runs only a child process (`src/pipelines/spec.rs:24-36`), a
stage's output goes to a log buffer instead of to its dependents
(`src/pipelines/executor.rs:300`), and a pipeline is not a pipe that can be mounted or
served -- measured by one config file, loaded by the existing loader with no Rust per
workflow, answering a request with the review stage's output at exactly vault 2,
github 2, llm 1 calls.

## refutation condition

If the review-ticket fixture (created in slice 4, run end to end in slice 5) cannot express the review-ticket workflow without a new registry
factory kind, a new engine, or Rust code beyond loading the config, then "pure config on
existing pieces" was the wrong frame and this spec is struck.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | a stage body is either a process (today's fields, unchanged) or a pipe stage: `pipe` (a name) + `request` (`method`, `path`, `headers`, `query`, `body`); parsing a stage with both or neither is refused with a reason naming the stage (~~`PipelineSpec::validate` refuses~~: struck 2026-11-19, the `StageBody` enum cannot represent both/neither, so the refusal happens at deserialize via `#[serde(try_from = "RawStageSpec")]`) | yes |
| R2 | a pipe stage's request strings are expanded by `proxima_http::templates` against one JSON scope `{ "in": <pipeline input>, "<dep>": <output of each dependency> }`, read as `{{body.<path>}}`; a placeholder whose path is absent fails the stage with a reason naming the placeholder, and that stage's pipe receives 0 calls | yes |
| R3 | a pipe stage's output is its 2xx response body as JSON when it parses, else a JSON string; a non-2xx response fails the stage with the status in the reason, and dependents are skipped (existing strict-mode semantics, `executor.rs:22-26`) | yes |
| R4 | a `pipeline` key in a pipe spec is a loader key like `upstreams`: its `pipes` map is built through `build_pipe` (so each carries its own `retry` / `rate_limit` / `client-auth` / `transform`), and the result is a `PipeHandle` whose request body is `in` and whose response is 200 with the `output` stage's value serialized by `serde_json::to_vec` (compact), or 502 with the first failing stage's reason | yes |
| R5 | load refuses, before anything is served, a pipeline with an unknown stage pipe, an unknown dependency, or a cycle; the error names the offender | yes |
| R6 | process stages behave exactly as before: the existing `pipelines::` tests pass with an unchanged count | yes |
| R7 | no new registry factory kind, engine or config format: the diff adds no `impl ... PipeFactory for` outside `tests/` | yes |
| R8 | the `pipeline` loader key exists only in builds with the `tokio` feature (the executor is tokio-only: `pub mod pipelines` is `#[cfg(feature = "tokio")]`, `src/lib.rs:264-265`, while `load` and `app` are ungated, `:99,179`); a build without `tokio` refuses a `pipeline` spec at load with a reason naming the `tokio` feature | yes |

## architecture

Reuse map (nothing in this list is built):

| piece | existing | where |
|---|---|---|
| DAG, parallel stages, skip on failed parent, recording | `PipelineExecutor`, `topological_order`, `PipelineSpec::validate` | `src/pipelines/executor.rs:27-200`; `dag.rs:10`; `spec.rs:43-70` |
| nested pipe specs built by the loader | `build_pipe` recursion, as `build_composed` does for `upstreams` | `src/load.rs:563,660-663,760-800` |
| per-pipe middleware | `apply_middleware_stack` (`retry`, `transform`, `rate_limit`, `middleware: [...]`) | `src/load.rs:822-846` |
| leaves | `http`, `process_rpc`, `fs`, `synth`, ... | `src/load.rs:581-631` |
| templating | `TemplateContext { body }` + `expand`, `{{body.X.Y}}` path lookup | `proxima-http/src/templates.rs:14-24,91-108` |
| strict vs lenient expansion | conflaguration `render` / `render_lenient` split | `~/repos/mine/conflaguration/src/template.rs:90,124` |
| recording sink | `BoundedRecordingSink` | `proxima-recording/src/pipe/cap.rs:124` |
| test stub with call counter | test-registered counting factory | `src/app_builder.rs:520` |

Changes, each an extension of the piece it touches:

```rust
// src/pipelines/spec.rs
pub struct StageSpec {
    pub name: String,
    #[serde(flatten)]
    pub body: StageBody,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
}

#[serde(untagged)]
pub enum StageBody {
    Process { command: String, args: Vec<String>, env: BTreeMap<String, String>, cwd: Option<PathBuf> },
    Pipe { pipe: String, request: StageRequest },
}

pub struct StageRequest {           // field names as WorkloadSpec, src/scenarios/spec.rs:17-28
    pub method: String,
    pub path: String,
    pub headers: BTreeMap<String, String>,
    pub query: BTreeMap<String, String>,
    pub body: Option<String>,
}

// proxima-http/src/templates.rs -- strict sibling of `expand`, same resolver
pub fn expand_strict(input: &str, context: &TemplateContext<'_>) -> Result<String, String>; // Err = the placeholder

// src/pipelines/executor.rs
impl PipelineExecutor {
    pub fn with_pipes(self, pipes: BTreeMap<String, PipeHandle>) -> Self;
    pub async fn run_with_input(&self, spec: PipelineSpec, input: Value) -> Result<PipelineRunReport, ProximaError>;
}
// StageCompletion carries Option<Arc<Value>> output (Copy -> Clone); PipelineRunReport exposes outputs by stage

// src/load.rs -- next to the `upstreams` arm
} else if value.get("pipeline").is_some() {
    // #[cfg(feature = "tokio")]: build_pipeline(value, context) builds `pipes` via build_pipe,
    //   validates, and returns the PipeHandle; apply_middleware_stack wraps it as for any pipe
    // #[cfg(not(feature = "tokio"))]: Err(ProximaError::Config("`pipeline` requires the `tokio` feature ..."))
}
```

Config shape (JSON form; TOML is the same tables):

```json
{ "name": "review_ticket",
  "pipeline": {
    "pipes":  { "vault": { "fs": {...} }, "github": { "http": "...", "retry": {...}, "rate_limit": {...} } },
    "stages": [ { "name": "ticket", "pipe": "vault", "request": { "method": "GET", "path": "/{{body.in.ticket}}/state.json" } },
                { "name": "pr", "pipe": "github", "depends_on": ["ticket"],
                  "request": { "method": "GET", "path": "/repos/{{body.ticket.repo}}/pulls/{{body.ticket.pr}}" } } ],
    "output": "pr" } }
```

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| how a pipeline becomes a pipe | loader key `pipeline`, built in `load.rs` like `upstreams` | a registry factory `pipeline` adds a factory kind (R7) and cannot build nested specs: `PipeFactory::build` gets no `LoadContext`, and the one factory that builds nested specs itself (`record`'s `resolve_inner`, `src/upstreams/record.rs:1072`) is the hand-matcher `substrate-shape:c1` is counting down |
| where a stage's pipes come from | the pipeline's own `pipes` map, built once at load | `LoadContext` holds no named-pipe map (`src/load.rs:111-134`); `App::lookup_pipe` (`src/app.rs:605`) is not reachable from a factory build, and adding it would order-couple `[[pipe]]` tables |
| stage body shape | `StageBody` enum | `Option` fields allow both/neither states the type could rule out (rust rules: type-driven design); cost is 8 construction sites, all test helpers (`spec.rs:99`, `dag.rs:80`, `executor.rs:360`, `control_plane.rs:565`, `fs_control_plane.rs:817`, `replay.rs:395`, `http_routes.rs:374`, `explain.rs:68`), plus field reads at `executor.rs:120,132,289-292`, `replay.rs:227,297`, `spec.rs:165` |
| template engine | `proxima_http::templates` | already on the pipe path (http1 header injection, synth bodies, `src/upstreams/synth.rs:210`); conflaguration's `template.rs` is a byte-level config preprocessor |
| missing template path | `expand_strict` fails the stage | `expand` returns an empty string (`templates.rs:91-108`), which turns `/repos/{{body.ticket.repo}}` into `/repos/` and sends a wrong request |
| where the new tests live | `stage_body_*`, `ctl_parse_*` in `src/pipelines/spec.rs` tests; `stage_template_*`, `stage_output_*`, `ctl_template_*`, `ctl_output_*` in `src/pipelines/executor.rs` tests; `expand_strict_*`, `ctl_expand_*` in `proxima-http/src/templates.rs` tests; `pipeline_pipe_*`, `ctl_review_*`, `ctl_failure_*`, `ctl_load_*`, `ctl_tokio_*` (the last cfg'd `not(feature = "tokio")`) and their counting stub factories in `tests/pipeline_pipe.rs`, with the fixture at `tests/fixtures/pipeline_pipe/review_ticket.json`, both created in slice 4 | stubs inside `src/` would add `impl PipeFactory` lines that AC8 counts; AC7 excludes the new names explicitly so the incumbent count stays comparable; controls carry a `ctl_` prefix so no AC's substring filter selects another AC's control (0 `fn ctl_` at `dd60e28c`) |
| scope naming | `{{body.<stage>.<path>}}`, input under `in` | reuses the existing `body.` resolver unchanged; no new placeholder grammar |
| per-call cost | executor re-runs `validate` + topo sort per call (`executor.rs:71-72`) | caching them is a measured decision for later, not part of making it work |
| runtime | `pipeline` is `tokio`-only; builds without it refuse the key by name | the executor uses tokio `JoinSet` + `watch` and `pipelines` is `#[cfg(feature = "tokio")]` (`src/lib.rs:262-265`: "tokio::spawn capability with no prime equivalent today"); porting the executor to prime is a separate change, out of scope |

## acceptance criteria

Test runner is `cargo val` (nextest filtersets). One cargo build at a time,
`CARGO_BUILD_JOBS=2` (machine memory limits). Every command names a target flag:
cargo-val runs doc-tests regardless of `-E` unless `--lib` / `--test <name>` is given
(the first baseline run counted 21 passed, all doc-tests, 0 naming `pipelines::`:
`22.1 - Tickets/PROXIMA-25/evidence/baseline.log`). `src/pipelines` tests need
`--features tokio`.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo val -p proxima-lib --features tokio --lib -E 'test(stage_body_)'` | 4 passed, 0 failed: (1) `{"name":"build","command":"cargo","args":["check"]}` parses to exactly `StageSpec { name: "build", body: Process { command: "cargo", args: ["check"], env: {}, cwd: None }, depends_on: [] }`; (2) `{"name":"pr","pipe":"github","request":{"method":"GET","path":"/repos/x/pulls/1"}}` parses to exactly `StageSpec { name: "pr", body: Pipe { pipe: "github", request: { method: "GET", path: "/repos/x/pulls/1", headers: {}, query: {}, body: None } }, depends_on: [] }`; (3) `{"name":"fetch-ticket","command":"cargo","pipe":"github","request":{"method":"GET","path":"/"}}` is refused, the error text contains "stage `fetch-ticket`" and "both"; (4) `{"name":"fetch-ticket"}` is refused, the error text contains "stage `fetch-ticket`" and "neither" |
| AC2 | R2 | `cargo val -p proxima-lib --features tokio --lib -E 'test(stage_template_)'` | 3 passed, 0 failed, each asserted against a hand-derived literal (input `{"ticket":"PROXIMA-24"}`; the `ticket` stub returns `{"repo":"brianbruggeman/proxima","pr":42}`; strings render raw and numbers via `to_string`, `templates.rs:114-121`): (1) a stage with path `/{{body.in.ticket}}/state.json` -> its stub receives exactly `/PROXIMA-24/state.json`; (2) a dependent stage with path `/repos/{{body.ticket.repo}}/pulls/{{body.ticket.pr}}` -> its stub receives exactly `/repos/brianbruggeman/proxima/pulls/42`; (3) a stage with path `/repos/{{body.ticket.missing}}/pulls` fails, reason contains `body.ticket.missing`, its stub call count 0 |
| AC3 | R3 | `cargo val -p proxima-lib --features tokio --lib -E 'test(stage_output_)'` | 3 passed, 0 failed, each against a hand-written literal: (1) a stub answering 200 with body bytes `{"repo":"brianbruggeman/proxima","pr":42}` -> the stage output equals exactly `json!({"repo":"brianbruggeman/proxima","pr":42})`; (2) a stub answering 200 with body bytes `plain text review` -> the output equals exactly `Value::String("plain text review")`; (3) a stub answering 503 -> the stage fails with `503` in the reason and its dependent reports skipped |
| AC4 | R4 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_pipe_review_ticket)'` | 1 passed: the fixture `tests/fixtures/pipeline_pipe/review_ticket.json` loaded by the existing loader, request body `{"ticket":"PROXIMA-24"}`, stub replies: vault `state.json` -> `{"repo":"brianbruggeman/proxima","pr":42}`, vault notes dir -> `["design.md"]`, github pr -> `{"title":"split stage body","body":"adds StageBody"}`, github files -> `[{"filename":"src/pipelines/spec.rs"}]`, llm -> `{"content":[{"type":"text","text":"LGTM with one nit"}]}`. Expected: per-stub call counts exactly vault 2, github 2, llm 1; response 200 whose body is exactly the bytes `{"content":[{"text":"LGTM with one nit","type":"text"}]}` (the llm reply parsed per R3 and serialized per R4; this workspace's serde_json has no `preserve_order`, `Cargo.toml:225,515`, so object keys come out sorted) and parses to exactly `json!({"content":[{"type":"text","text":"LGTM with one nit"}]})`; the llm request body contains `adds StageBody`, `src/pipelines/spec.rs` and `design.md` |
| AC5 | R4 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_pipe_failure_)'` | 1 passed: the github stub returns 500 only for path `/repos/brianbruggeman/proxima/pulls/42` (the `pr` stage) and AC4's replies otherwise, so `pr` is the only failing stage -> pipeline response 502, body contains ``stage `pr` ``, llm stub call count 0 |
| AC6 | R5 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_pipe_load_refuses_)'` | 3 passed, 0 failed, each a one-edit mutation of the review-ticket fixture, checked by one helper (required substrings present, a named bystander absent): (i) stage `pr` given `"pipe": "gitlab"` (absent from `pipes`) -> the load error contains "stage `pr`" and "`gitlab`", and not "`notes`"; (ii) stage `review` given an extra dependency `summary` -> contains exactly "stage `review` depends on unknown stage `summary`" (the incumbent format, `spec.rs:56-57`); (iii) stage `pr` given a dependency on `review` -> contains "pipeline DAG has a cycle involving stages:", `"pr"` and `"review"`, and not `"ticket"` (`dag.rs:66-67`) |
| AC17 | R5 control | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(ctl_load_)'` | 6 passed, through AC6's helper: (a) the incumbent `PipelineSpec::validate` error for mutation (ii), produced directly -> Ok; (b) the incumbent `topological_order` error for mutation (iii), produced directly -> Ok; (c) "invalid pipeline" (names nothing) against (i)'s expectation -> Err; (d) "stage `pr` uses unknown pipe `gitlab`; check `notes` too" (every required substring present, plus the bystander) against (i) -> Err; (e) "stage `pr` uses unknown pipe `gitlab`" against (i) -> Ok (known-good for (i)); (f) `pipeline DAG has a cycle involving stages: ["pr", "review", "ticket"]` against (iii) -> Err (the absence check bites) |
| AC18 | R8 control | `cargo val -p proxima-lib --test pipeline_pipe -E 'test(ctl_tokio_)'` (default features, no `tokio`) | 2 passed, through AC13's helper: (a) "`pipeline` requires the `tokio` feature (pipe `review_ticket`)" -> Ok; (b) the incumbent fallthrough text "spec must include `upstreams`, `http`, `kv`, or a `type` discriminator" (`src/load.rs:671`, what a default build says today) -> Err |
| AC7 | R6 | `cargo val -p proxima-lib --features tokio --lib -E 'test(pipelines::) and not (test(stage_body_) or test(stage_template_) or test(stage_output_) or test(pipeline_pipe_) or test(ctl_))'` | 43 passed, 0 failed, 0 doc-tests (PROXIMA-25 at `f42b714a`, whose `src/` is identical to `dd60e28c`: control_plane 5, dag 5, executor 5, explain 4, fs_control_plane 7, http_routes 9, replay 2, spec 6; log `22.1 - Tickets/PROXIMA-25/evidence/baseline-tokio-lib.log`) |
| AC8 | R7 | in the slice worktree: `git add -A -N && git diff dd60e28c -- '*.rs' ':!:tests/*' ':!:*/tests/*' \| grep -c -E '^\+.*impl.*PipeFactory for'` | 0 (scope: 2228 tracked `.rs` files at `f42b714a`; `add -N` brings untracked new files into the diff) |
| AC9 | R7 control | (a) `git show 3a8118db -- '*.rs' ':!:tests/*' ':!:*/tests/*' \| grep -c -E '^\+.*impl.*PipeFactory for'`; (b) the same on `f0aab1d9`; (c) `d=$(mktemp -d) && cd "$d" && git init -q && mkdir -p src tests && printf 'impl PipeFactory for Stub {}\n' > src/a.rs && printf 'impl PipeFactory for TestStub {}\n' > tests/b.rs && git add -A -N && git diff -- '*.rs' ':!:tests/*' ':!:*/tests/*' \| grep -c -E '^\+.*impl.*PipeFactory for'; cd / && rm -rf "$d"` | (a) 5, root `src/` half; (b) 1, `proxima-patterns/src` half; (c) 1: the untracked `src` impl counted, the `tests/` one excluded (0 without `add -N`, 2 without the exclusions). Measured 2026-11-19 |
| AC10 | R2 control | `cargo val -p proxima-lib --features tokio --lib -E 'test(ctl_template_)'` | 3 passed, through the same assertion helper AC2 uses: (a) for AC2's two present-path cases, the request path produced by the incumbent `proxima_http::templates::expand` (called directly in the test with the same scope, not through the new stage path) is fed to the helper -> Ok (the incumbent's own correct output passes AC2's check); (b) case 3's stage wired to the incumbent lenient `expand` -> its stub receives exactly `/repos//pulls` (call count 1) and the helper returns Err (the defect R2 prevents is detected); (c) a hand-built correct case-3 outcome (stage failed, reason `body.ticket.missing`, stub call count 0) fed to the helper -> Ok |
| AC11 | R3 control | `cargo val -p proxima-lib --features tokio --lib -E 'test(ctl_output_)'` | 6 passed, all through the same assertion helper AC3 uses, which takes the expected reason substring as an argument: (a) a process stage exiting 1 (incumbent path, `executor.rs:22-26`), checked against "stage failed, reason contains `exited with code Some(1)` (`executor.rs:176`), dependent skipped" -> Ok; (a') a two-stage process pipeline where both stages exit 0, checked against the same expectation -> Err (the helper does not pass a run with no failure); (b) the literal `json!({"repo":"brianbruggeman/proxima","pr":42})` against case 1's expectation -> Ok; (c) `json!({})` against case 1 -> Err; (d) `Value::String("plain text review")` against case 2 -> Ok; (e) `Value::String("")` against case 2 -> Err (the helper catches the empty-output defect) |
| AC12 | R4 control | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(ctl_review_) or test(ctl_failure_)'` | 5 passed: the AC4 helper accepts a hand-wired Rust call sequence of the same stubs whose final body is the llm reply passed through `serde_json::from_slice` then `serde_json::to_vec`, the same capture R3 and R4 specify (known-good -> Ok); rejects the raw llm stub bytes forwarded unparsed, `{"content":[{"type":"text",...` (known-bad: a build that bypasses R3 is detected); rejects a run where the llm stub was never called; the AC5 helper accepts a hand-built known-good failure (502, body naming ``stage `pr` ``, llm count 0) and rejects a 200 success response (not misread as a failure) |
| AC16 | R1 control | `cargo val -p proxima-lib --features tokio --lib -E 'test(ctl_parse_)'` | 4 passed: (a) the process `StageSpec` the existing `dag.rs:80` helper builds, serialized with `serde_json::to_string` and parsed back through AC1's equality check -> equal (known-good, the incumbent shape); (b) AC1 case 2's parsed value compared with the same value but `pipe: ""` -> not equal (the check catches the empty-name defect); (c) AC1's refusal helper on serde's generic "data did not match any variant of untagged enum StageBody" with stage `fetch-ticket` / word `both` -> false; (d) the same helper on "stage `fetch-ticket` has both a process body and a pipe body" -> true |
| AC14 | R2 (resolver) | `cargo val -p proxima-http --lib -E 'test(expand_strict_)'` | 3 passed, scope `{"in":{"ticket":"PROXIMA-24"},"ticket":{"repo":"brianbruggeman/proxima","pr":42}}`: `{{body.in.ticket}}` -> exactly `Ok("PROXIMA-24")`; `/repos/{{body.ticket.repo}}/pulls/{{body.ticket.pr}}` -> exactly `Ok("/repos/brianbruggeman/proxima/pulls/42")`; `{{body.ticket.missing}}` -> `Err` whose text is exactly `body.ticket.missing` (`templates` rides proxima-http's default `http1`, `proxima-http/src/lib.rs:37-38`) |
| AC15 | R2 (resolver) control | `cargo val -p proxima-http --lib -E 'test(ctl_expand_)'` | 2 passed, through AC14's helper: (a) the incumbent `expand` on AC14's two present-path templates and scope returns exactly `PROXIMA-24` and `/repos/brianbruggeman/proxima/pulls/42` -> Ok (the literals are the incumbent's semantics, not the new function's); (b) the incumbent `expand` on `{{body.ticket.missing}}` yields `""`; `Ok("")` against AC14's expected `Err("body.ticket.missing")` -> the helper rejects it (known-bad detected) |
| AC13 | R8 | `cargo val -p proxima-lib --test pipeline_pipe -E 'test(pipeline_pipe_refused_without_tokio)'` (default features, no `tokio`) | 1 passed: loading the review-ticket fixture fails, the error contains exactly "`pipeline` requires the `tokio` feature" and "review_ticket" (the pipe's name), 0 pipes registered. `tests/pipeline_pipe.rs` compiles in both builds: tokio-only tests, and every helper, stub factory and import they alone use, carry `#[cfg(feature = "tokio")]`; this test carries `#[cfg(not(feature = "tokio"))]` (the workspace denies warnings, `Cargo.toml:85-86`, so an unused tokio-only helper fails the default build) |

## out of scope

- serving the pipeline as an MCP tool: the configured MCP registry is not in git
  (`pm/topics/agentic-mcp.md` gap 5); until it is re-landed, the pipeline pipe is served
  on an HTTP listener from config
- MCP `initialize` handshake for `process_rpc` stdio upstreams (gap 4)
- CEL branch guards between stages (gap 3)
- per-item stages (map over a list result via `ScatterGather`)
- caching `validate` / topo order across calls; JIT; build-time codegen
- streaming stage outputs
- running pipelines on the prime runtime (default builds); this spec only names the refusal
- commits and pushes: each slice ends green in a worktree; landing is the owner's call

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| `#[serde(untagged)]` + `flatten` turns a bad stage into "data did not match any variant", losing the stage name | high | R1's reason requirement fails | AC1 asserts the stage name in the reason; deserialize through a raw form and validate if serde's message loses it |
| `StageCompletion` is `Copy` and travels on `watch` channels; adding an output forces `Clone` and touches every send | medium | slice 3 grows | keep the output in a separate `BTreeMap<String, watch::Receiver<Option<Arc<Value>>>>` beside the completion channel if the `Copy` change spreads |
| stage outputs are buffered whole in memory | medium for large PRs | memory per call | bounded by the leaf pipes' existing response limits; streaming is out of scope |
| machine froze on parallel cargo jobs (2026-10-31) | medium | lost session | one cargo job at a time, `-j 2`, check `vm_stat` before a build |

## context

- `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md` -- owner quotes, element -> primitive table, gaps
- `src/pipelines/{spec,dag,executor}.rs` -- the engine being extended
- `src/load.rs:563-680` (`build_pipe`), `:760-846` (`build_composed`, `apply_middleware_stack`)
- `proxima-http/src/templates.rs` -- the resolver, `{{body.X.Y}}` paths
- `src/client/handle.rs:1240-1248` -- the in-tree `ragd-llm` `http` + `retry` spec shape
