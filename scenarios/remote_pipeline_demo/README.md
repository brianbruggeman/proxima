# remote pipeline runner — anchor demo

A runnable end-to-end demo of proxima's anchor capability: a local
`proxima` CLI submits, inspects, replays, and mutates pipelines (DAG
of stages, each one a child process) on a remote `proximad` over
SSH-stdio HTTP/1.1. The same surface works locally over a UDS, which
is what `run.sh` defaults to.

The demo passes the four threshold criteria for the anchor:

1. **structured introspection** beats `tail -f` / `journalctl` —
   `inspect` returns per-stage records, `tail` reproduces the
   recorded event stream
2. **deterministic replay + mutate** — `replay` reproduces the
   recording bit-for-bit (modulo `ts_ms`); `replay --substitute`
   propagates a fresh failure
3. **`explain` walks a causal chain** — the stage DAG's ancestors
   render bottom-up, derived from the recorded spec
4. **recording is the source of truth** — explain / inspect / replay
   all derive from the on-disk recording, never inferred

## tl;dr

```bash
cd proxima
bash scenarios/remote_pipeline_demo/run.sh
```

The script builds `proximad` + `proxima-cli` if needed, spawns a
local UDS proximad in the background, runs through 8 verbs, prints
`PASS / FAIL` per criterion, and exits non-zero if anything fails.
Expected output ends with:

```
summary
PASS: 10
  ✓ submit returned id …
  ✓ list surfaces the pipeline by name
  ✓ criterion #1: structured introspection (per-stage record + spec)
  ✓ tail emitted 11 recorded events
  ✓ criterion #3: explain walks bench → build → fetch
  ✓ artifact streams bytes from the bench workspace
  ✓ replay returned new id …
  ✓ criterion #2: replay event count matches original (deterministic replay)
  ✓ criterion #4: replay events derived from recording (no inference)
  ✓ criterion #2: substitute propagates failure downstream (bench skipped)

all pass criteria satisfied.
```

## running it remotely

Set `PROXIMA_REMOTE_HOST=<host>` to drive the SSH-stdio transport
against a remote daemon. The CLI spawns
`ssh <host> proximad serve --stdio` per invocation and frames HTTP/1.1
over the pipe; each invocation runs a fresh proximad on the remote.
The daemon's state lives at `~/.local/share/proximad/pipelines/` on
the remote host, so submissions persist across SSH sessions.

```bash
# pre-req: proximad installed on <host> at $PATH; SSH keys set up
PROXIMA_REMOTE_HOST=host-b bash scenarios/remote_pipeline_demo/run.sh
```

The artifact step is currently skipped on the SSH transport — the
one-shot SSH client doesn't decode chunked encoding incrementally yet
(local UDS path doesn't have this restriction). See the discipline
log for the follow-up.

## the spec

[`pipeline.toml`](pipeline.toml) declares a 3-stage DAG:

```toml
name = "remote-pipeline-demo"

[[stages]]
name = "fetch"
command = "/bin/sh"
args = ["-c", "echo 'fetched sources' > sources.txt; sleep 0.1; echo fetch ok"]

[[stages]]
name = "build"
command = "/bin/sh"
args = ["-c", "echo 'built artifact' > artifact.bin; sleep 0.1; echo build ok"]
depends_on = ["fetch"]

[[stages]]
name = "bench"
command = "/bin/sh"
args = ["-c", "echo 'bench report' > criterion.html; sleep 0.1; echo bench ok"]
depends_on = ["build"]
```

Each stage runs `/bin/sh -c …` for portability; swap in `cargo bench`
or any real workload for the actual host-b run. Each stage writes
a predictable artifact to its workspace so `proxima pipeline artifact`
has something to retrieve.

[`build-failure.toml`](build-failure.toml) is a drop-in failing
replacement for the `build` stage, used by the `--substitute` leg:

```toml
name = "build"
command = "/bin/sh"
args = ["-c", "echo 'simulated build failure' 1>&2; exit 3"]
depends_on = ["fetch"]
```

## what the script exercises

| step | command | proves |
|---|---|---|
| 1 | `proxima pipeline submit pipeline.toml` | submission round-trip |
| 2 | `proxima pipeline list` | submission discoverable by name |
| 3 | `proxima pipeline inspect remote-pipeline-demo` | **bar #1**: structured introspection |
| 4 | `proxima pipeline tail remote-pipeline-demo` | terminal pipeline replays its recording |
| 5 | `proxima pipeline explain remote-pipeline-demo --stage bench` | **bar #3**: DAG ancestors `bench → build → fetch` |
| 6 | `proxima pipeline artifact remote-pipeline-demo --stage bench --path criterion.html` | per-stage workspace retrieval |
| 7 | `proxima pipeline replay remote-pipeline-demo` | **bar #2 + bar #4**: replay event count matches original |
| 8 | `proxima pipeline replay remote-pipeline-demo --substitute build=build-failure.toml` | **bar #2**: substituted failure propagates (bench skipped) |

## driving the daemon by hand

```bash
# terminal A: start a daemon on a UDS
proximad serve --unix /tmp/proximad.sock --state-dir /tmp/proximad-state

# terminal B: every pipeline verb
proxima pipeline --socket /tmp/proximad.sock submit pipeline.toml
proxima pipeline --socket /tmp/proximad.sock list
proxima pipeline --socket /tmp/proximad.sock inspect remote-pipeline-demo
proxima pipeline --socket /tmp/proximad.sock tail remote-pipeline-demo
proxima pipeline --socket /tmp/proximad.sock events    # all events, all pipelines
proxima pipeline --socket /tmp/proximad.sock explain remote-pipeline-demo --stage bench
proxima pipeline --socket /tmp/proximad.sock artifact remote-pipeline-demo --stage bench --path criterion.html --output ./report.html
proxima pipeline --socket /tmp/proximad.sock replay remote-pipeline-demo
proxima pipeline --socket /tmp/proximad.sock replay remote-pipeline-demo --substitute build=build-failure.toml
```

Same surface over SSH (replace `--socket /tmp/…` with `--host host-b`).

## the same surface from an agent (MCP)

The pipeline tools are also reachable over MCP JSON-RPC 2.0. Opt in
on the daemon side with `--mcp-stdio` (single duplex over
stdin/stdout) or `--mcp-unix <path>` (multi-connection UDS):

```bash
proximad serve --mcp-stdio --state-dir /tmp/proximad-state
```

Tool surface:

- `pipelines_submit { spec }` → returns the pipeline id
- `pipelines_list { name?, spec_hash_hex? }` → newest-first summaries
- `pipelines_resolve { query }` → canonical id from name / prefix
- `pipelines_inspect { id }` → full record
- `pipelines_explain { id, stage }` → ancestor chain
- `pipelines_replay { id, substitutes? }` → new pipeline id

Streaming tools (`tail` / `events`) aren't on MCP yet — they need
JSON-RPC notification plumbing that's a follow-up. The HTTP path
already supports them.

## further reading

- **Discipline log** — per-gap landing record, rollbacks, deferred
  follow-ups: [`docs/discipline-log/remote-pipeline-runner.md`](../../docs/discipline-log/remote-pipeline-runner.md)
- **Plan file** (with mid-sprint pivot note): the cozy-waddling-wall plan
- **Universal RecordingEvent design** — the recording layer that
  makes deterministic replay possible: [`rust/src/recording/event.rs`](../../rust/src/recording/event.rs)
- **Pipeline executor** — strict failure semantics + diamond
  concurrency: [`rust/src/pipelines/executor.rs`](../../rust/src/pipelines/executor.rs)
