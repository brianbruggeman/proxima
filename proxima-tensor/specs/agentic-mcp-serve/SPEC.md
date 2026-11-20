# agentic-mcp-serve -- serve a config-built pipeline as an MCP tool

status: audited -- ADMIT by spec-auditor round 4, 2026-11-20 (PROXIMA-33); its four conditions bound the same day (AC14(a) 124, TASKS 0b, TASKS 0c, the L21 file) (rounds 1-3 refused 12, 11 and 7 reasons, each amended)
owner: brian
created: 2026-11-20

Follows `proxima-tensor/specs/agentic-mcp` (landed `828e8cd1`, on origin). Capture and owner
intent: `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md` (gaps 4 and 5). Spine
`[agentic-mcp]`, goal `agentic-control-plane:c5`. Every `file:line` below is in this repo
(`~/repos/mine/proxima`) at base `53d972c3`, read 2026-11-20. The cargo package is
`proxima-lib` (`Cargo.toml:279`); `proxima` is only its lib name. Ticket evidence lives under
`/Users/brianbruggeman/repos/obsidian/slot-0/slot-0/20 - Collaboration and Records/22 - Investigations/22.1 - Tickets/`
(written `<tickets>` below).

## problem

An MCP client (Claude Code's `.mcp.json` entry `proxima serve review_ticket.json`) cannot call
a pipeline defined only in config, because the MCP listener's tool list is fixed at 17
control-plane tools (`src/listeners/mcp.rs:268-471`, dispatch at `:473-551`), a `[[listen]]`
MCP listener runs one serve lane per core all reading the same stdin and reports a TCP bind
address that `proxima serve` probes, failing after 5 s and exiting 1 (measured on the base binary,
see the refutation condition; `proxima-listen/src/handle.rs:195-211,314-330`,
`tools/proxima-cli/src/main.rs:951-958`), and a
`process_rpc` stage cannot talk to an MCP stdio server, which refuses `tools/call` before
`initialize` + `notifications/initialized` (nothing is written to a fresh child before its
first request, `src/upstreams/process_rpc.rs:150-178`) -- measured by one config file served
by the real `proxima` binary over stdio answering `tools/call` with the exact review text,
its first stage fetched from a stub MCP server over `process_rpc`.

## refutation condition

If the review-ticket MCP fixture (slice 4) cannot be served without a new registry factory
kind, a new engine, a new config format, or Rust code beyond loading the config, then
"serve config-built pipelines through the existing listener" was the wrong frame and this
spec is struck. ~~If slice 0's run of `proxima serve` with an `mcp` stdio listener answers both
requests with exactly 2 JSON-RPC lines and nothing else on stdout, R7 is struck.~~ Measured
2026-11-20 (PROXIMA-35, base binary, `<tickets>/PROXIMA-35/evidence/serve-*.txt`): stdout carried
exactly the 2 JSON-RPC answers, so the predicted `READY` corruption did not occur (`READY` prints
only after a successful probe, `main.rs:951-958`), but the process exited 1 after 5 s with
`listener did not come up at 127.0.0.1:63971 within 5s`: `proxima serve` TCP-probes the `mcp`
listener's reported bind address, which nothing listens on, and gives up. That is the defect R7
removes (no bind address -> no probe), so R7 stands on the measured failure and AC12 now asserts
it (exit status 0 after SIGTERM, which the base binary cannot produce).

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | `process_rpc` accepts `init`: an ordered list of frames `{ "send": "<one line>", "reply": <bool, default true> }`. After every spawn of the child (first call, and every respawn), before that call's own request, each frame's `send` is written as one newline-terminated line, and when `reply` is true exactly one reply line is read and discarded. Frames run once per child, never per call | yes |
| R2 | a frame that cannot complete (the child closes stdout, or the call's `request_timeout_ms` elapses while that frame runs) fails the call with an error naming the upstream's configured label and `init frame <index>` (0-based), and drops the child, so the next call respawns it and runs `init` again. The label comes from `RpcState.label`, which is `String::new()` today (`src/upstreams/process_rpc.rs:94`) and is set to the configured label | yes |
| ~~R3~~ | ~~every `process_rpc` error that names the upstream carries its configured label~~ struck round 2: the only labeled errors besides R2's are `spawn_child`'s (`:193-198`), which fires on unix only for a command or argument holding an interior NUL (`proxima-process/src/command.rs:529-535,592-594`), never for a missing program, because `proxima_process` forks and a failed exec exits the child with 127 while `spawn` returns Ok (`proxima-process/src/spawn.rs:327-328` always calls `spawn_fork`; `execvp` then `_exit(127)` at `:746-748`), and `ensure_alive`'s restart=false (`:157-161`), which needs a kept child to exit between calls (a race). The label assignment is R2's mechanism and AC2 asserts it | -- |
| R4 | an `mcp` listener spec accepts `tools`: a list of `{ "name", "description", "inputSchema" (a JSON object), "method" (default `POST`), "path" (default `/`) }`. `tools/list` returns the incumbent 17 descriptors unchanged and in order, followed by each configured tool, in config order, as exactly `{ "name", "description", "inputSchema" }` | yes |
| R5 | `tools/call` naming a configured tool sends exactly one request through the listener's existing `dispatch` (the router of the `[[listen]]`'s `pipe`, `src/app.rs:1334-1346`): the tool's method and path, metadata `content-type: application/json`, payload the call's `arguments` serialized compact by `serde_json::to_vec` (`{}` when absent). The response goes through the incumbent mapping (`mcp.rs:536-550`): status 200-399 -> `{"content":[{"type":"text","text":<body>}]}`, else JSON-RPC error `-32603` with the incumbent message form `tool <name in backticks> returned <status>: <body>` (`mcp.rs:545`) | yes |
| R6 | `serve` refuses, before binding or reading any request, a `tools` value that is not an array, or a tool that repeats an earlier tool's name, uses a built-in tool's name, lacks `name`, lacks `description`, has a non-object `inputSchema`, or carries an unknown field. The error is `ProximaError::Config`; for a non-array `tools` it contains `mcp`, `tools` and `array`, and for a tool it contains `mcp tools[<index>]` plus the offending name or field. The parse is one function, `configured_tools(spec: &Value) -> Result<.., ProximaError>`, called by `serve` before its transport match (`mcp.rs:76`) | yes |
| R7 | a listener whose protocol name is `mcp`, started through `Listener::run_with_runtime` (`proxima-listen/src/handle.rs:157`, reached via `ListenerSpec::attach`, `:123`), runs exactly 1 serve lane regardless of core count and its `ListenerHandle::bind_addr()` is `None`, so `proxima serve` neither probes it over TCP nor prints a `READY` line for it (`tools/proxima-cli/src/main.rs:951-958` already skips `None`). Every other protocol keeps its incumbent lane count and bind address | yes |
| R8 | the real `proxima` binary, given only the slice-4 fixture config (`[[pipe]]` with a `pipeline` whose first stage is a `process_rpc` stub MCP server with `init`, then `synth` stages; one `[[listen]]` of type `mcp`, transport `stdio`, with one configured tool), answers `initialize`, `tools/list` and `tools/call` over its stdin/stdout with the exact results in AC12, and its whole stdout, from start to exit on SIGTERM, is exactly those 3 lines, each a JSON-RPC 2.0 message | yes |
| R9 | incumbents unchanged: the MCP listener and `process_rpc` unit tests (19, slice 0), the `proxima-listen` lib tests (slice 0 count), and `proximad`'s `mcp_stdio` smoke all pass with unchanged counts; the diff adds no `impl ... PipeFactory for` outside `tests/`; and every count in this spec selects only its own AC's tests (no new fn name contains two of the 11 test prefixes) | yes |

## literals

Each line below is an id, one space, then the literal, which runs to the end of the line.
ACs refer to these by id.

```
L1 {"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"proxima","version":"0"}}}
L2 {"jsonrpc":"2.0","method":"notifications/initialized"}
L3 {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_ticket","arguments":{"id":"PROXIMA-24"}}}
L4 {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"inits=1"}],"structuredContent":{"ticket":"PROXIMA-24","repo":"brianbruggeman/proxima","pr":42}}}
L5 {"jsonrpc":"2.0","id":1,"error":{"code":-32002,"message":"server not initialized"}}
L6 {"jsonrpc":"2.0","id":9,"method":"exit"}
L7 {"jsonrpc":"2.0","id":5,"method":"hang"}
L10 process_rpc `tracker` init frame 0: process_rpc subprocess closed stdout
L11 process_rpc `` init frame 0: process_rpc subprocess closed stdout
L12 {"name":"review_ticket","description":"Review the PR linked from a ticket","inputSchema":{"type":"object","properties":{"ticket":{"type":"string"}},"required":["ticket"]}}
L13 {"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"review_ticket","arguments":{"ticket":"PROXIMA-24"}}}
L14 {"jsonrpc":"2.0","id":7,"result":{"content":[{"type":"text","text":"{\"content\":[{\"text\":\"LGTM\",\"type\":\"text\"}]}"}]}}
L15 stage `pr` pipe `github`: returned status 500
L16 tool `review_ticket` returned 502: stage `pr` pipe `github`: returned status 500
L17 mcp tools[1] `review_ticket` duplicates tools[0]
L18 mcp tools[0] duplicate
L19 {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"content\":[{\"text\":\"review of PROXIMA-24: adds StageBody\",\"type\":\"text\"}]}"}]}}
L20 tool `review_ticket` returned 502: stage `pr` failed: stage `pr` placeholder `body.ticket.result.structuredContent.repo` did not resolve
L22 {"name":"triage_ticket","description":"Triage a ticket","inputSchema":{"type":"object"}}
```

L8 and L9 struck in round 2 with R3. L21 is a file, not a line: `tests/fixtures/mcp_stub/incumbent_tools.json`,
the `result.tools` array (17 entries) of a `tools/list` answered by `proximad serve --mcp-stdio` built
at `53d972c3`, captured in slice 0c. It is the oracle for "the incumbent 17, unchanged" in AC5,
AC8 and AC12: a snapshot of the base binary's output, independent of the code being changed.
The stub `tracker.sh` is `tests/fixtures/mcp_stub/tracker.sh` (written and byte-checked against
L4 and L5 under PROXIMA-33, shellcheck clean).

L20 is the chain `executor.rs:240-243` ("stage `<name>` failed: <reason>") -> `load.rs:978` (502
body) -> `mcp.rs:545` ("tool ... returned"). L14 and L19 inner text is key-sorted: the
workspace's `serde_json` has no `preserve_order` (`Cargo.toml:225,515`; `load.rs:976`).

## architecture

Reuse map (nothing in this list is built):

| piece | existing | where |
|---|---|---|
| a pipeline built from config | `[[pipe]]` + the `pipeline` loader key | `src/app.rs:1194-1208`; `src/load.rs:680-681,851` |
| MCP listener bound from config, its dispatch the router of its `pipe` | `[[listen]] type = "mcp"`, `bind_listener`, `build_listener_router` | `src/app.rs:1214-1265,1334-1346` |
| the binary that serves a multi-block config | `proxima serve` -> `run_serve_full` -> `App::load_full` | `tools/proxima-cli/src/main.rs:826-831,938-965` |
| JSON-RPC line loop, `initialize`, notifications | `serve_jsonrpc_loop`, `handle_jsonrpc`, `dispatch_method` | `src/listeners/mcp.rs:172-258` |
| tool response mapping | the incumbent tail of `tool_call` | `src/listeners/mcp.rs:525-550` |
| request to a dispatch path | `build_request` | `src/listeners/mcp.rs:808-834` |
| newline-framed child RPC, respawn, timeout | `ProcessRpcUpstream`, `ensure_alive`, `dispatch_one` | `src/upstreams/process_rpc.rs:111-246` |
| stubs | `synth` (expands `{{body.<field>}}` from a JSON request body) and `/bin/sh` scripts | `src/upstreams/synth.rs:182-215` |
| the lane rule being extended | `is_http` name check | `proxima-listen/src/handle.rs:195-211` |
| lane test doubles | `ReadySignalProtocol`, `ThreadRuntime` | `proxima-listen/src/handle.rs:854-935` |

Changes, each an extension of the piece it touches:

```rust
// src/upstreams/process_rpc.rs
pub struct ProcessRpcInitFrame { pub send: String, #[serde(default = "default_true")] pub reply: bool }
// ProcessRpcConfig gains `#[setting(skip)] #[serde(default)] pub init: Vec<ProcessRpcInitFrame>`,
// ProcessRpcSpec gains `pub init: Vec<ProcessRpcInitFrame>`; `call` runs the frames, each under
// the call's timeout, on the child `ensure_alive` just spawned; RpcState.label = the label.

// src/listeners/mcp.rs
// `serve` parses spec["tools"] once (serde, deny_unknown_fields) into an Arc'd list shared
// by every connection; `tools/list` appends the configured descriptors after the incumbent
// array; `tool_call` checks configured names before the built-in match and, for a hit,
// builds (method, path, Some(arguments bytes)) and falls into the incumbent tail at :525.

// proxima-listen/src/handle.rs
// next to `is_http`: an `mcp` listener gets num_lanes = 1 and bind_addr = None.
```

Fixture (slice 4, JSON; `<dir>` is replaced by the test's tempdir path):

```json
{ "pipe": [ { "name": "review_ticket", "pipeline": {
    "pipes": {
      "tracker": { "process_rpc": { "name": "tracker", "command": "/bin/sh", "args": ["<dir>/tracker.sh"],
                   "init": [ { "send": "<L1>" }, { "send": "<L2>", "reply": false } ] } },
      "github": { "synth": { "status": 200, "body": "{\"title\":\"split stage body\",\"body\":\"adds StageBody\"}" } },
      "llm":    { "synth": { "status": 200, "body": "{\"content\":[{\"type\":\"text\",\"text\":\"review of {{body.prompt}}\"}]}" } } },
    "stages": [
      { "name": "ticket", "pipe": "tracker", "request": { "method": "POST", "path": "/",
        "body": "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"get_ticket\",\"arguments\":{\"id\":\"{{body.in.ticket}}\"}}}" } },
      { "name": "pr", "pipe": "github", "depends_on": ["ticket"], "request": { "method": "GET",
        "path": "/repos/{{body.ticket.result.structuredContent.repo}}/pulls/{{body.ticket.result.structuredContent.pr}}" } },
      { "name": "review", "pipe": "llm", "depends_on": ["ticket", "pr"], "request": { "method": "POST", "path": "/v1/messages",
        "body": "{\"prompt\":\"{{body.ticket.result.structuredContent.ticket}}: {{body.pr.body}}\"}" } } ],
    "output": "review" } } ],
  "listen": [ { "type": "mcp", "transport": "stdio", "bind": "127.0.0.1:0", "pipe": "review_ticket",
    "tools": [ <L12> ] } ] }
```

`<L1>` / `<L2>` stand for L1 / L2 as JSON strings (quotes escaped); `<L12>` is L12 as a JSON value;
`<dir>/tracker.sh` is a copy of `tests/fixtures/mcp_stub/tracker.sh` the test places in its tempdir.

The stub MCP server `tracker.sh` (POSIX sh, one `while IFS= read -r line` loop, used by slices 1
and 4): a line containing `"method":"initialize"` -> increments `inits`, prints
`{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"tracker","version":"1"}}}`;
`"method":"notifications/initialized"` -> sets `ready=1`, prints nothing; `"method":"tools/call"`
with `ready=1` -> prints L4 with `inits=<inits>` and `"ticket":"<id>"`, where `<id>` is the text
after `"arguments":{"id":"` up to the next `"`; with `ready` unset -> prints L5;
`"method":"exit"` -> exits 0 without printing; any other line -> prints nothing.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| where a configured tool's target comes from | the listener's own `dispatch` (the router of its `[[listen]].pipe`), method + path per tool | building a pipe inside the listener needs a `LoadContext` the listener does not hold (`serve` gets `dispatch` + spec only, `mcp.rs:59-66`); the router is already the incumbent target of every built-in tool |
| re-land the orphaned configured registry (`slot-0/proxima-mcp-tools`, 1607-line `mcp.rs`) | no | its `McpToolRoute::Http` also routes only through `dispatch` (PROXIMA-31 inventory, `<tickets>/PROXIMA-31/`), so it adds a route enum and 19 data-driven built-ins without reaching anything `method` + `path` does not |
| MCP protocol revision | keep `2024-11-05` (`mcp.rs:27`) | the modern `2026-07-28` HTTP transport is the mcp-daemon product (`pm/topics/mcp-daemon.md`), whose commits are not in git; Claude Code speaks stdio `2024-11-05` |
| tool failure shape | incumbent JSON-RPC `-32603` | `isError: true` results would make configured tools disagree with the 17 built-ins on the same listener |
| one or many pipes per MCP listener | one (`pipe`), as every non-http listener today | `mount` for `mcp` is the next change if a second workflow per server is wanted; a second `[[listen]]` serves a second pipeline today |
| init frame reply rule | explicit `reply` flag per frame, default true | inferring "request vs notification" from a JSON `id` makes `process_rpc` parse JSON, and its contract is protocol-agnostic lines (`process_rpc.rs:27-28`) |
| when init runs | on the child `ensure_alive` just spawned, each frame under the call's timeout | at build time would spawn children for pipes never called (spawning is lazy, `:150-178`); per call would re-initialize a live MCP session, which servers refuse; one timeout around all frames would lose the failing frame's index (R2) |
| how the lane rule knows `mcp` | protocol name check beside the incumbent `is_http` (`handle.rs:195`) | a `ListenProtocol` method widens a trait every protocol implements for one listener's property |
| test names and filters | positive prefixes `process_rpc_init_runs_`, `process_rpc_init_fails_`, `mcp_tools_list_`, `mcp_tools_call_`, `mcp_tools_refuses_`, `mcp_lane_`, `mcp_config_tool_`; control prefixes `ctl_rpc_`, `ctl_tools_`, `ctl_lanes_`, `ctl_serve_`. No prefix contains another, and every new test name starts with exactly one prefix and contains no second one anywhere (AC17 counts violations, expected 0), so nextest's substring `test()` cannot pull a control or a sibling AC into a count | the round-1 names (`ctl_` + the positive prefix) let `test(mcp_tools_list_)` select `ctl_mcp_tools_list_*`; a starts-with rule alone does not stop a mid-name match (round 2) |
| where the new tests live | `src/upstreams/process_rpc.rs` tests (`process_rpc_init_*`, `ctl_rpc_*`); `src/listeners/mcp.rs` tests (`mcp_tools_*`, `ctl_tools_*`); `proxima-listen/src/handle.rs` tests (`mcp_lane_*`, `ctl_lanes_*`); `tools/proxima-cli/tests/integration/mcp_config_tool.rs`, a `mod` in `tests/integration.rs`, with the review-ticket fixture under `tools/proxima-cli/tests/fixtures/mcp_config_tool/` (`mcp_config_tool_*`, `ctl_serve_*`). `tracker.sh` and the L21 file live once in `tests/fixtures/mcp_stub/` and are read through `CARGO_MANIFEST_DIR` by both crates | unit tests in the file they test (proxima testing rules); the CLI end to end is the only cross-binary test |

## acceptance criteria

Runner `cargo val` (nextest filtersets), always with a target flag (`--lib` / `--test`) so
doc-tests are not counted, wrapped `timeout 1500` (heavy-command guard), one cargo build at a
time, `CARGO_BUILD_JOBS=2`. Every check helper named below is called by its AC and by at least
one control expecting Ok (a hand-built known-good value) and one expecting Err (a known-bad
value), so each helper is shown able to pass and to fail.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo val -p proxima-lib --features tokio --lib -E 'test(process_rpc_init_runs_)'` | 3 passed, 0 failed, each through `check_reply(body, expected)` (byte equality): (1) upstream `name` `tracker`, `/bin/sh` running `tracker.sh`, `init` = [L1 reply true, L2 reply false], built from the loader spec form (`{"process_rpc":{...}}` through `crate::load::load`), one call with body L3 -> reply exactly L4; (2) two calls with L3 on one upstream -> both replies exactly L4 (`inits=1` twice: init ran once for the child); (3) a call with L6 returns an error, then a call with L3 -> exactly L4 (the respawned child was initialized; without init it prints L5) |
| AC2 | R2 | `cargo val -p proxima-lib --features tokio --lib -E 'test(process_rpc_init_fails_)'` | 3 passed, through `check_init_error(error, label, index)`, which requires two substrings: `process_rpc`, one space, and the label wrapped in backticks (for `tracker`, the first 21 bytes of L10); and `init frame <index>`: (1) command `/bin/sh` with args `["-c","exit 0"]`, `name` `tracker`, `init` = [L1] -> the call fails, `check_init_error(.., "tracker", 0)` -> Ok; (2) `tracker.sh`, `name` `tracker`, `init` = [L1, L7] (the stub prints nothing for `hang`), `request_timeout_ms` 200 -> the call fails within 2 s, `check_init_error(.., "tracker", 1)` -> Ok; (3) command `/bin/sh` with args `["-c", "if [ -e \"$0\" ]; then exec /bin/sh \"$1\"; fi; : > \"$0\"", "<tempdir>/marker", "<tracker.sh>"]` (the first child creates the marker and exits, later children exec the stub), `name` `tracker`, `init` = [L1 reply true, L2 reply false]: the first call with L3 fails and `check_init_error(.., "tracker", 0)` -> Ok; the second call with L3 -> reply exactly L4 (`check_reply`), proving the failed child was dropped and the respawned one was initialized |
| ~~AC3~~ | ~~R3~~ | struck round 2 with R3 | -- |
| AC4 | R1-R2 controls | `cargo val -p proxima-lib --features tokio --lib -E 'test(ctl_rpc_)'` | 5 passed: (a) `tracker.sh` with no `init`, a call with L3 -> reply exactly L5, and `check_reply(L5, L4)` -> Err (stub and helper both catch a missing handshake); (b) `check_reply(L4, L4)` -> Ok; (c) `check_init_error(L10, "tracker", 0)` -> Ok; (d) `check_init_error(L10, "tracker", 1)` -> Err (the index is checked); (e) `check_init_error(L11, "tracker", 0)` -> Err (the empty label of `process_rpc.rs:94` is caught) |
| AC5 | R4 | `cargo val -p proxima-lib --features tokio --lib -E 'test(mcp_tools_list_)'` | 1 passed: spec `{"tools":[L12 plus "method":"PUT","path":"/review", L22]}`, `tools/list` id 2 -> `check_tool_list(result.tools, [L12, L22])` -> Ok, where `check_tool_list(tools, configured)` requires exactly 17 + `configured.len()` entries, entries 0-16 equal to the L21 file's array element for element, and the rest equal to `configured` in order (so L12's `method` / `path` are absent from its listed descriptor) |
| AC6 | R5 | `cargo val -p proxima-lib --features tokio --lib -E 'test(mcp_tools_call_)'` | 3 passed, through `check_dispatch(recorded, expected)` (equality of method, path, `content-type` value, payload bytes, and request count) on a test `SendPipe` that records every request: (1) L12 plus `"method":"PUT","path":"/review"`, call L13, stub answers 200 `{"content":[{"text":"LGTM","type":"text"}]}` -> 1 request, `PUT`, `/review`, `application/json`, payload exactly `{"ticket":"PROXIMA-24"}`; the response parses to exactly the value of L14; (2) L12 (no `method` / `path`), a call with no `arguments` -> 1 request, `POST`, `/`, payload exactly `{}`; (3) stub answers 502 with body L15 -> error code `-32603`, message exactly L16 |
| AC7 | R6 | `cargo val -p proxima-lib --features tokio --lib -E 'test(mcp_tools_refuses_)'` | 8 passed, each through `check_refusal(error, required)` (the error is `ProximaError::Config` and every required substring is present). Cases (i)-(vii) call `configured_tools` with spec `{"tools": <value>}`; case (viii) awaits `McpListenProtocol::new().serve(..)` itself with spec `{"transport":"unix","path":"<tempdir>/mcp.sock","tools":[L12, L12]}`, the recording dispatch stub, and a fresh shutdown channel, and requires the future to resolve to the (ii) error with `<tempdir>/mcp.sock` absent afterwards and the stub's count 0 (refused before binding, so before any request can arrive): (i) `tools` = `{}` -> required `mcp`, `tools`, `array`; (ii) two copies of L12 -> `mcp tools[1]`, `review_ticket`; (iii) L12 with `name` `list_pipes` -> `mcp tools[0]`, `list_pipes`; (iv) L12 with `inputSchema` `[]` -> `mcp tools[0]`, `inputSchema`; (v) L12 without `description` -> `mcp tools[0]`, `description`; (vi) L12 plus `"pipe":"x"` -> `mcp tools[0]`, `pipe`; (vii) L12 without `name` -> `mcp tools[0]`, `name`; (viii) `serve` itself, as above -> `mcp tools[1]`, `review_ticket` |
| AC8 | R4-R6 controls | `cargo val -p proxima-lib --features tokio --lib -E 'test(ctl_tools_)'` | 8 passed: (a) a spec with no `tools` -> `tools/list` equals the L21 array exactly (17 entries; the incumbent list is unchanged), and `check_tool_list(it, [L12])` -> Err; (b) `check_tool_list(L21 array + [L12, L22], [L12, L22])` -> Ok, and the same with the last two swapped -> Err (order is checked) -- one test; (h) `check_refusal(ProximaError::Upstream(L17), ["mcp tools[1]","review_ticket"])` -> Err (the variant is checked); (c) `check_dispatch` with AC6(1)'s expectation on a hand-built recording whose payload is `{"ticket":"PROXIMA-24","extra":1}` -> Err; (d) the same with payload `{"ticket":"PROXIMA-24"}` -> Ok; (e) `check_refusal(L18, ["mcp tools[1]","review_ticket"])` -> Err; (f) `check_refusal(L17, ["mcp tools[1]","review_ticket"])` -> Ok; (g) with spec `{"tools":[L12]}`, the incumbent unknown-tool call (`mcp.rs:1078-1092`) still returns code `-32602` |
| AC9 | R7 | `cargo val -p proxima-listen --lib -E 'test(mcp_lane_)'` | 2 passed, using `ReadySignalProtocol` and `ThreadRuntime` extended so the protocol's name is a field, it counts `serve` calls, and the runtime reports 4 cores; a `ListenerSpec::protocol(bind, ..).attach(..)` listener started with `run_with_runtime`: (1) protocol named `mcp` -> `check_lanes(count, 1)` -> Ok; (2) its `bind_addr()` is `None` |
| AC10 | R7 control | `cargo val -p proxima-listen --lib -E 'test(ctl_lanes_)'` | 3 passed, same doubles: (a) protocol named `stream` -> `check_lanes(count, 4)` -> Ok on unix and `check_lanes(count, 1)` -> Ok on windows (the incumbent rule: one lane per core, except one lane on windows, `handle.rs:207`) and `bind_addr()` is `Some`; (b) `check_lanes(4, 1)` -> Err (a multi-lane `mcp` is caught); (c) `check_lanes(1, 1)` -> Ok |
| AC11 | R9 | `cargo val -p proxima-lib --features tokio --lib -E '(test(listeners::mcp::) or test(upstreams::process_rpc::)) and not (test(process_rpc_init_) or test(mcp_tools_) or test(ctl_))'` | 19 passed, 0 failed (PROXIMA-34 at `53d972c3`, wall 235 s: `listeners::mcp::` 12, `upstreams::process_rpc::` 7; log `<tickets>/PROXIMA-34/evidence/baseline-lib.log`) |
| AC12 | R8 | `cargo val -p proxima-cli --test integration -E 'test(mcp_config_tool_)'` | 1 passed: `proxima serve <tempdir>/review_ticket.json` spawned with piped stdio; the test writes `initialize` id 1, `notifications/initialized`, `tools/list` id 2, and `tools/call` id 3 with `{"name":"review_ticket","arguments":{"ticket":"PROXIMA-24"}}`, then reads exactly 3 lines within a 30 s `tokio::time::timeout` (a bound, not a sleep): line 1 `result.protocolVersion` = `2024-11-05`; line 2 `check_tool_list(result.tools, [L12])` -> Ok, where this `check_tool_list` is the integration file's own helper with AC5's definition, reading L21 from `tests/fixtures/mcp_stub/incumbent_tools.json` (the lib's `cfg(test)` helper is not visible to `tools/proxima-cli/tests`); line 3 `check_tool_text(line, L19)` (parse both, compare values) -> Ok; then the test closes stdin, sends SIGTERM (`kill -TERM <pid>`), waits for the child to exit and reads stdout to EOF within the same timeout bound: the whole stdout is exactly those 3 lines, each parsing as JSON with `"jsonrpc":"2.0"` (a stray line before, between or after them fails the count), and the exit status is success (0): the base binary exits 1 when its 5 s TCP probe of the `mcp` listener fails, or dies by the signal if SIGTERM lands before its shutdown handler exists (`main.rs:951-965`) |
| AC13 | R8 control | `cargo val -p proxima-cli --test integration -E 'test(ctl_serve_)'` | 4 passed: (a) the same fixture with the `init` list removed -> line 3 is a JSON-RPC error, code `-32603`, message exactly L20 (the stub refused `tools/call`, so `ticket`'s output is L5's envelope and `pr`'s placeholder does not resolve), and `check_tool_text(line, L19)` -> Err; (b) `check_tool_text(L19, L19)` -> Ok; (c) the integration file's `check_tool_list(L21 array + [L12], [L12])` -> Ok; (d) its `check_tool_list(L21 array, [L12])` -> Err |
| AC14 | R9 | (a) `cargo val -p proxima-listen --lib -E 'not (test(mcp_lane_) or test(ctl_lanes_))'`; (b) `cargo val -p proxima-cli --test integration -E 'test(proximad_serves_pipeline_tools_over_mcp_stdio)'` | (a) 124 passed, 0 failed (PROXIMA-34 at `53d972c3`, 2026-11-20, wall 81 s, `<tickets>/PROXIMA-34/evidence/baseline-listen.log`); (b) 1 passed (`tools/proxima-cli/tests/integration/mcp_stdio.rs:27`) |
| AC15 | R9 | in the slice worktree: `git add -A -N && git diff 53d972c3 -- '*.rs' ':!:tests/*' ':!:*/tests/*' \| grep -c -E '^\+.*impl.*PipeFactory for'` | 0 |
| AC16 | R9 control | (a) `git show 3a8118db -- '*.rs' ':!:tests/*' ':!:*/tests/*' \| grep -c -E '^\+.*impl.*PipeFactory for'`; (b) `d=$(mktemp -d) && cd "$d" && git init -q && mkdir -p src tests && printf 'impl PipeFactory for Stub {}\n' > src/a.rs && printf 'impl PipeFactory for TestStub {}\n' > tests/b.rs && git add -A -N && git diff -- '*.rs' ':!:tests/*' ':!:*/tests/*' \| grep -c -E '^\+.*impl.*PipeFactory for'; cd / && rm -rf "$d"` | (a) 5 (a commit that did add factories is counted); (b) 1 (an untracked `src` impl is counted, the `tests/` one excluded) |
| AC17 | R9 (each count selects only its AC) | in the slice worktree (the prefix list is written inline because zsh does not word-split an unquoted variable; measured 2026-11-20, a `$P` form printed 0 for the control): `git add -A -N && git diff 53d972c3 -- src/upstreams/process_rpc.rs src/listeners/mcp.rs proxima-listen/src/handle.rs tools/proxima-cli/tests/integration/ \| grep -oE '^\+[[:space:]]*(async )?fn [a-z0-9_]+' \| awk '{print $NF}' \| while read -r name; do hits=0; for prefix in process_rpc_init_runs_ process_rpc_init_fails_ mcp_tools_list_ mcp_tools_call_ mcp_tools_refuses_ mcp_lane_ mcp_config_tool_ ctl_rpc_ ctl_tools_ ctl_lanes_ ctl_serve_; do case "$name" in *"$prefix"*) hits=$((hits+1));; esac; done; [ "$hits" -gt 1 ] && echo "$name"; done \| wc -l` | 0 (no new fn name contains two prefixes anywhere in it, so no `test()` substring filter counts a test that belongs to another AC; the module paths `upstreams::process_rpc::tests`, `listeners::mcp::tests`, `handle::tests`, `mcp_config_tool::` contain none of the 11 prefixes). Control: the same `while` loop fed the single name `ctl_tools_mcp_tools_list_x` prints 1 line; fed `ctl_tools_list_x` prints 0 (measured 2026-11-20 in zsh and bash: 1 and 0) |

## out of scope

- the MCP `2026-07-28` HTTP transport, server discovery and per-surface routes (mcp-daemon)
- `mount` arrays on `mcp` listeners (several pipes behind one MCP listener)
- JSON-RPC id correlation in `process_rpc` and handling server-initiated notifications on a
  child's stdout; a child that prints unsolicited lines desynchronizes, as today
- CEL branch guards between stages (gap 3): not required by c5; next spec `agentic-mcp-guards`
- making `bind` optional for stdio listeners (`bind_listener` requires it, `src/app.rs:1233-1236`)

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| `proxima serve` writes logs to stdout | medium | corrupts the MCP stream (R8) | AC12 asserts every stdout line is JSON-RPC; slice 0 records today's stdout |
| `canonical_process` or `build_pipeline_pipes` drops `init` on the way to `ProcessRpcConfig` | medium | AC13's failure shows up in AC12 | AC1 builds through the loader spec form, not only the Rust constructor |
| the `proxima-cli` integration build is heavy | high | slice 4 exceeds 20 minutes | build the test target in its own ticket before the slice's run |
| shared-machine load (82 on 2026-11-20) trips the heavy-command guard | high | runs refused | runs wait for load under 30; one cargo job at a time |

## context

- `/Users/brianbruggeman/repos/slot-0/pm/topics/agentic-mcp.md` -- owner quotes, gaps
- `proxima-tensor/specs/agentic-mcp/SPEC.md` -- the pipeline-as-a-pipe spec this builds on
- PROXIMA-31 / PROXIMA-32 inventories, `<tickets>/PROXIMA-31/`, `<tickets>/PROXIMA-32/`
- `tools/proxima-cli/tests/integration/mcp_stdio.rs` -- the incumbent MCP stdio client flow
