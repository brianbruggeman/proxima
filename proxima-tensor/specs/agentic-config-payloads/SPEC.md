# agentic-config-payloads -- every step's payload written in the config's own format

status: draft, round 1
owner: brian
created: 2026-11-20

The owner, 2026-11-20: "we need to be able to configure the entire thing _within_ toml. like every
step." (`/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md`, section "any format, every
step"). This spec is the TOML half: the four step payloads that today can only be JSON text inside
the config. XML, and routing every format through conflaguration, is a separate spec
(`agentic-config-formats`, PROXIMA-50) gated on a conflaguration release. Every `file:line` below is in
`~/repos/mine/proxima` at `285b519e5` (main), read 2026-11-20 (PROXIMA-49 citation sweeps).
`<tickets>` is
`/Users/brianbruggeman/repos/obsidian/slot-0/slot-0/20 - Collaboration and Records/22 - Investigations/22.1 - Tickets`.
The cargo package is `proxima-lib`; `src/pipelines` is `#[cfg(feature = "tokio")]`.

## problem

A pipeline served as an MCP tool can be written in TOML, YAML, JSON5 or RON except for its step
payloads, which are strings: a stage request body (`StageRequest.body: Option<String>`,
`src/pipelines/spec.rs:72`), a process_rpc init frame (`ProcessRpcInitFrame.send: String`,
`src/upstreams/process_rpc.rs:54`), and a synth body and body template (`SynthConfig.body`,
`body_template: Option<String>`, `src/upstreams/synth.rs:262,267`). A TOML config therefore carries
JSON text inside it: the CLI review fixture holds 6 such strings
(`tools/proxima-cli/tests/fixtures/mcp_config_tool/review_ticket.json` lines 5, 6, 7, 8, 11, 15)
and the triage fixture 2 (`tests/fixtures/pipeline_pipe/triage.json` lines 27, 34) -- measured by
both fixtures rewritten in TOML with every payload a TOML table (0 JSON-text strings), answering
the same as their JSON forms.

## refutation condition

If a payload written as a TOML table cannot reach the same receiver bytes' meaning (the same JSON
value) as its JSON-text form -- including a value taken whole from an earlier stage, as an
unquoted text splice `{"pr": {{body.ticket.pr}}}` gives today -- without a new stage kind, a new
template language, or a per-payload format flag, the "payload is a value" frame is wrong and this
spec is struck.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | `StageRequest.body` holds a JSON value. A string keeps today's meaning byte for byte (expanded by `expand_strict`, `src/pipelines/executor.rs:704-705`). A map or list is a structured payload: each string leaf is expanded by the same strict engine and context; a string leaf that is exactly one placeholder over the body scope (`{{body}}` or `{{body.<path>}}`, whitespace inside the braces trimmed as `expand_with` trims it, `proxima-http/src/templates.rs:64`, nothing outside the braces) takes the referenced JSON value with its type; non-string leaves pass unchanged; the result is sent as compact JSON. A placeholder that does not resolve fails the stage exactly as a text body's does (0 pipe calls, the reason naming the placeholder) | yes |
| R2 | `ProcessRpcInitFrame.send` holds a JSON value: a string is written as today (`process_rpc.rs:205`); a map or list is written as one line of compact JSON. Init frames are not templated, as today | yes |
| R3 | `SynthConfig.body` and `body_template` hold JSON values. A string keeps today's meaning. A structured `body` is answered as compact JSON. A structured `body_template` is expanded per request by R1's rule with synth's non-strict engine (`expand`, as `expand_template_body` uses today, `synth.rs:183-210`; an absent path becomes `""`, `synth_template_missing_body_field_renders_empty`), then answered as compact JSON. The environment (`PROXIMA_SYNTH_BODY`, `PROXIMA_SYNTH_BODY_TEMPLATE`) sets either field as text, as today. `SynthConfig::builder().body("text")` keeps compiling | yes |
| R4 | the triage fixture and the CLI review fixture each have a TOML twin in which no string leaf is JSON text; each twin, loaded through proxima's TOML format (`proxima-config/src/lib.rs:272-276`), answers as its JSON fixture does, and every receiver gets the same JSON value | yes |
| R5 | incumbents unchanged: every existing test in the touched modules and files passes with its slice-0 count; a payload given as a string serializes and behaves exactly as before | yes |

## literals

Each line is an id, one space, then the literal to the end of the line.

```
S1 {"pr":"{{body.ticket.pr}}"}
E1 {"pr":42}
S2 {"title":"#{{body.ticket.pr}} in {{body.ticket.repo}}"}
E2 {"title":"#42 in brianbruggeman/proxima"}
S3 {"ticket":"{{body.ticket}}"}
E3 {"ticket":{"pr":42,"repo":"brianbruggeman/proxima"}}
S4 {"n":3,"ok":true,"none":null}
S5 ["{{body.in.ticket}}",{"deep":["{{body.ticket.repo}}"]}]
E5 ["PROXIMA-24",{"deep":["brianbruggeman/proxima"]}]
S6 {"x":"{{body.missing.path}}"}
S7 {"pr":"{{body.ticket.pr}}"}
E7 {"pr":"42"}
S8 {"a":"{{ body.ticket.pr }}","b":" {{body.ticket.pr}}"}
E8 {"a":42,"b":" 42"}
P1 {"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"proxima","version":"0"}}}
P2 {"jsonrpc":"2.0","method":"notifications/initialized"}
P3 {"jsonrpc":"2.0","id":0,"method":"initialise","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"proxima","version":"0"}}}
P4 {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_ticket","arguments":{"id":"PROXIMA-24"}}}
Y1 {"title":"split stage body","body":"adds StageBody"}
Y2 {"content":[{"type":"text","text":"review of {{body.prompt}}"}]}
Y3 {"prompt":"PROXIMA-24: adds StageBody"}
Y4 {"content":[{"text":"review of PROXIMA-24: adds StageBody","type":"text"}]}
Y5 {"echo":"{{body.n}}","miss":"{{body.absent}}"}
Y6 {"n":5}
Y7 {"echo":5,"miss":""}
Y8 plain text
T1 {"pr":"adds StageBody"}
T2 {"ticket":"PROXIMA-25"}
```

S1-S8 run as the body of stage `pr` in the executor tests' two-stage rig: stage `ticket` is a stub
answering `TICKET_BODY` `{"repo":"brianbruggeman/proxima","pr":42}` (`executor.rs:1038`), the
pipeline input is `{"ticket":"PROXIMA-24"}`, so the scope is `{"in":{"ticket":"PROXIMA-24"},
"ticket":{"repo":"brianbruggeman/proxima","pr":42}}`. S4's expected value is S4. S7 is given as a
string (the incumbent text path); every other S is given as a structure. P3 misspells `initialize`.
Y1 and Y2 are the CLI review fixture's synth payloads (lines 7, 8) as structures. T1 and T2 are the
triage fixture's `review` and `summary` bodies (lines 27, 34) after expansion for PROXIMA-24 and
PROXIMA-25.

## architecture

Reuse map: the strict and non-strict template engine (`proxima-http/src/templates.rs:27-35`, the
body lookup `lookup_body_path`, `:130-140`), the stage request builder (`executor.rs:689-708`),
init frames (`process_rpc.rs:51-58,202-206`), synth's body resolution (`synth.rs:314-332`) and
template expansion (`:183-210`), proxima's TOML format (`proxima-config/src/lib.rs:272-276`), the
executor test rig (`executor.rs:1009-1096`), the triage rig (`tests/pipeline_pipe.rs:500-600`), and
the CLI session harness (`tools/proxima-cli/tests/integration/mcp_config_tool.rs:59-160`).

```rust
// proxima-http/src/templates.rs: the one engine gains its structured form
pub fn expand_value(value: &Value, context: &TemplateContext<'_>) -> Value;
pub fn expand_value_strict(value: &Value, context: &TemplateContext<'_>) -> Result<Value, String>;
// a string leaf that is exactly one `{{body...}}` placeholder takes the looked-up value
// (lookup_body_path's walk, returning the &Value instead of its display string); any other
// string leaf goes through expand / expand_strict; other leaves are cloned

// src/pipelines/spec.rs
pub struct StageRequest { /* method, path, headers, query unchanged */ pub body: Option<Value> }
// src/pipelines/executor.rs build_request: Value::String(text) -> expand_strict(text) as today;
// a map or list -> expand_value_strict, then serde_json::to_vec

// src/upstreams/process_rpc.rs
pub struct ProcessRpcInitFrame { pub send: Value, pub reply: bool }
// run_init: Value::String(text) -> its bytes as today; otherwise serde_json::to_vec

// src/upstreams/synth.rs
// SynthConfig.body / body_template: Option<Value>, `#[builder(into)]`, and a
// `#[setting(resolve_with = ...)]` that wraps the raw env text as Value::String
```

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| payload type | `serde_json::Value`, a string meaning text | a `Payload { Text, Structured }` enum restates what `Value::String` versus map/list already says; a caller can do nothing with it they cannot do with `Value` |
| a value taken whole from an earlier stage | a leaf that is exactly one body placeholder keeps the value's JSON type | without it a structured payload cannot carry a number or object from an earlier stage, which a text body does today with an unquoted splice (`{"pr": {{body.ticket.pr}}}`), so the TOML form would be strictly weaker (the refutation condition) |
| only body placeholders keep a type | yes | `trace_id`, `uuid`, `env.*` and the rest resolve to strings already (`templates.rs:96-125`) |
| where structured expansion lives | `proxima-http/src/templates.rs`, beside `expand` / `expand_strict` | one engine for text and structure; the executor and synth both call it |
| serialization | compact `serde_json::to_vec`, keys in `serde_json`'s map order (sorted: no `preserve_order`, `Cargo.toml:223,512`) | receivers compare JSON values, not key order: the tracker stub matches substrings that compact output keeps (`"method":"initialize"`, `"arguments":{"id":"`, `tests/fixtures/mcp_stub/tracker.sh`), and the CLI test compares parsed values (`check_tool_text`) |
| synth from the environment | text, as today | `Value: FromStr` parses JSON, so `PROXIMA_SYNTH_BODY=plain text` would stop loading |
| XML | out of scope here | XML gives every leaf as text and a one-element list as a bare element; that is PROXIMA-50's census (14 hand-read lists, 5 hand-read scalars, 8 config types with numbers behind serde's buffer) |

## acceptance criteria

Runner `cargo val`, always with `--lib` or `--test`, each run wrapped `timeout -k 30s 25m git wt
cargo ...`, `CARGO_BUILD_JOBS=2`, one build at a time. Test-name prefixes: `stage_payload_`
(`src/pipelines/executor.rs` tests), `process_rpc_payload_` (`src/upstreams/process_rpc.rs` tests),
`synth_payload_` (`src/upstreams/synth.rs` tests), `pipeline_payload_` (`tests/pipeline_pipe.rs`),
`mcp_config_payload_` (`tools/proxima-cli/tests/integration/mcp_config_tool.rs`), and control prefix
`ctl_payload_` in each of those files. No new prefix is a substring of another or of an existing
prefix in those files (`stage_template_`, `stage_output_`, `process_rpc_init_`, `synth_template_`,
`pipeline_pipe_`, `pipeline_guard_`, `mcp_config_tool_`, `ctl_template_`, `ctl_output_`, `ctl_rpc_`,
`ctl_load_`, `ctl_branch_`, `ctl_tokio_`, `ctl_serve_`), and no existing test name contains a new
prefix. The executor rig's `Probe` gains `bodies: Arc<StdMutex<Vec<Bytes>>>`, filled by
`StubPipe::call` beside `paths` (test-only). `json_text_leaves(&Value) -> usize` counts string leaves
that `serde_json::from_str::<Value>` parses to an object or array; it is defined in
`tests/pipeline_pipe.rs` and in the CLI test file.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo val -p proxima-lib --features tokio --lib -E 'test(stage_payload_)'` | 8 passed, one per S1-S8 as stage `pr`'s body: S1 -> the stub's one received body parses to E1; S2 -> E2; S3 -> E3; S4 -> S4; S5 -> E5; S6 -> `check_template(observe(..), &Expect::Missing("body.missing.path"))` -> Ok (stage failed, 0 calls, the reason names the placeholder); S7 (a string) -> the received bytes are exactly E7; S8 -> E8 |
| AC2 | R2 | `cargo val -p proxima-lib --features tokio --lib -E 'test(process_rpc_payload_)'` | 1 passed: a `ProcessRpcConfig` running `/bin/sh <tracker.sh>` (`tests/fixtures/mcp_stub/tracker.sh`) with init frames P1 and P2 (`reply` false) given as structures; one call with body P4 (text) answers JSON whose `result.content[0].text` is `inits=1` |
| AC3 | R3 | `cargo val -p proxima-lib --lib -E 'test(synth_payload_)'` | 4 passed: (1) `body` Y1 as a structure -> the response bytes parse to Y1; (2) `body_template` Y2 as a structure, request body Y3 -> the response parses to Y4; (3) `body_template` Y5 as a structure, request body Y6 -> the response parses to Y7; (4) under `temp_env::with_vars([("PROXIMA_SYNTH_BODY", Some("plain text"))])`, `SynthConfig::from_env()` -> `body` is `Value::String` Y8 and the response bytes are exactly Y8 |
| AC4 | R4 | `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(pipeline_payload_)'` | 3 passed. `tests/fixtures/pipeline_pipe/triage.toml` is the triage fixture written in TOML with both bodies as tables, parsed with `default_config_format_registry()?.parse_with_hint(text, Some("toml"))` and loaded in a fresh `triage_rig()`: (1) PROXIMA-24 -> `check_triage` with the incumbent PR-branch expectation (200, G8 of `agentic-mcp-guards`, calls `[1, 1, 1, 0]`) -> Ok, and the `llm` stub's `seen` body parses to T1; (2) PROXIMA-25 -> `check_triage` with the incumbent summary-branch expectation (200, G9, calls `[1, 0, 0, 1]`) -> Ok, and the `brief` stub's `seen` body parses to T2; (3) `json_text_leaves(twin)` -> 0 |
| AC5 | R4 | `cargo val -p proxima-cli --test integration -E 'test(mcp_config_payload_)'` | 2 passed: (1) `tools/proxima-cli/tests/fixtures/mcp_config_tool/review_ticket.toml`, the CLI fixture in TOML with all 6 payloads as tables, `<dir>` replaced as `write_config` does, served by `proxima serve <tempdir>/review_ticket.toml` through the incumbent session: line 1 `result.protocolVersion` = `2024-11-05`; line 2 `check_tool_list(.., &[configured_tool()])` -> Ok; line 3 `check_tool_text(line, TOOL_TEXT_OK)` -> Ok; stdout holds exactly those 3 lines and the exit status is success, as `mcp_config_tool_review_ticket_served_over_stdio` asserts; (2) `json_text_leaves(twin)` -> 0 |
| AC6 | R1-R4 controls | `cargo val -p proxima-lib --features tokio --lib -E 'test(ctl_payload_)'`; `cargo val -p proxima-lib --features tokio --test pipeline_pipe -E 'test(ctl_payload_)'`; `cargo val -p proxima-cli --test integration -E 'test(ctl_payload_)'` | lib 2 passed: (1) the AC2 rig with P3 in place of P1 -> the call answers JSON whose `error.code` is `-32002` (the tracker saw no `initialize`, so the AC2 check can fail); (2) S1 given as a string (text path) -> the received bytes are exactly `{"pr":"42"}`, not E1 (the whole-leaf rule is what makes AC1 (1) pass). pipeline_pipe 1 passed: `json_text_leaves` of `triage.json` -> 2. CLI 1 passed: `json_text_leaves` of `review_ticket.json` -> 6 |
| AC7 | R5 | (a) `cargo val -p proxima-lib --features tokio --lib -E 'test(pipelines::) or test(upstreams::process_rpc::) or test(upstreams::synth::)'`; (b) `cargo val -p proxima-lib --features tokio --test pipeline_pipe`; (c) `cargo val -p proxima-cli --test integration -E 'test(mcp_config_tool_) or test(ctl_serve_)'`; (d) `cargo val -p proxima-lib --features tokio --lib -E 'test(load::)'` | each passes with its slice-0 count plus the tests this spec adds to that filter, 0 failed: (a) slice-0 + 15 (AC1 8, AC2 1, AC3 4, AC6 lib 2); (b) slice-0 + 4 (AC4 3, AC6 1); (c) slice-0 + 0; (d) slice-0 + 0 |
| AC8 | R5 | `cargo clippy -p proxima-lib -p proxima-http -p proxima-cli --all-targets --features tokio -- -D warnings 2>&1 \| grep -cE '^(error\|warning)'` | 0 |

Counts: 8 + 1 + 4 + 3 + 2 + 4 controls = 22 new tests.

## out of scope

- XML, the conflaguration format layer, the hand-read and buffered-field census: PROXIMA-50 (`agentic-config-formats`)
- templating init frames (not templated today)
- key order in serialized payloads (sorted; no receiver depends on order)
- payloads other than the four named (none other carries JSON text in the two fixtures; PROXIMA-50's census covers every config type)

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| a call site constructs `StageRequest` / `ProcessRpcInitFrame` / `SynthConfig` with a `String` payload | high | compile errors in tests and examples | AC8 compiles every target of the three crates; call sites move to `Value` (`.into()` from `&str` / `String` already exists) |
| `#[setting(resolve_with)]` cannot target `Option<Value>` | medium | R3's env text path needs another attribute | AC3 (4) is the measurement; the derive's `skip` plus a hand-written env read is the fallback, named in the slice note |
| the shared machine's load guard refuses runs | high | delay | one build at a time, coordinated with peer sessions |

## context

- `<tickets>/PROXIMA-47`, `<tickets>/PROXIMA-48` (format inventories), PROXIMA-49 sweeps (this spec's citations)
- `proxima-tensor/specs/agentic-mcp-serve/`, `agentic-mcp-guards/` (the fixtures and rigs this spec reuses; G8 / G9)
- `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md` (goal, boundary, design)
