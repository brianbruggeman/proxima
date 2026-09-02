# onnx_reference

Committed incumbent-arm harness for the BGE-small lane: onnxruntime CPU EP
and torch/transformers, benchmarked on the same three real sentences and
hardcoded token-id arrays `proxima-onnx/examples/bge_eval.rs::sentences()`
uses. Mirrors `scripts/burn_reference`'s convention (pinned external deps,
never assume a default, real-data-only, print what actually ran).

This replaces `docs/discipline.md` ROW 195's `ort_bench.py` / `torch_bench.py`
-- those produced real numbers (ort 5.6296 ms/sentence, torch 10.0360
ms/sentence) but were thrown away after the session, leaving the incumbent
arms un-reproducible. An uncommitted incumbent arm is a phantom cell; this
directory is the fix.

## What it measures

Two incumbent arms, same protocol as `bge_eval.rs`:

- **onnxruntime CPU EP**: `InferenceSession` over a `model.onnx`, single
  session reused across sentences, `intra_op_num_threads=1`,
  `inter_op_num_threads=1`, `ExecutionMode.ORT_SEQUENTIAL`.
- **torch/transformers eager**: `AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")`
  from the local HF cache (`HF_HUB_OFFLINE=1`), `torch.set_num_threads(1)`.

Both arms: CLS pooling (first token's hidden state) + L2-normalize, exactly
`bge_eval.rs::embed`'s pooling. Default 5 runs (`ONNX_REF_RUNS`, matches
`bge_eval.rs`'s own `BGE_EVAL_RUNS` default), each run embeds all three
sentences once; reported mean/CoV are computed across the N per-run means
using the same population-variance CoV formula
`bge_eval.rs::coefficient_of_variation` uses (`sqrt(variance)/mean`, not
sample variance). torch gets a 3-pass warmup before timing starts (ROW 195's
own finding: 1-pass warmup left first-call allocator/kernel-selection noise
in the timed window, 16.14% CoV vs 2.87% after 3 passes); onnxruntime gets a
1-pass warmup.

Both arms print the thread settings read back after configuring the
runtime, but **the two readbacks are not equally strong**. torch's
(`torch.get_num_threads()`) is an independent live read of global
interpreter state -- it can and does disagree with what was requested (see
`torch.set_num_interop_threads` below), so it is a real assertion.
onnxruntime's Python `SessionOptions` exposes no such independent readback:
`session_options.intra_op_num_threads` after configuring the session
returns the same attribute this script just set, not a value read from the
C++ session's actual thread pool. The `RuntimeError` guard on it is kept
for symmetry and documentation, but it is a tautology, not proof ORT's
thread pool honored the request -- do not cite it as one.
`torch.set_num_interop_threads` itself raises if parallel work already
started in-process; that failure is caught and printed rather than
swallowed.

Same sanity check `bge_eval.rs` runs: cosine(A,B similar) must exceed both
cosine(A,C) and cosine(B,C) (A/B are paraphrases of "a cat on a mat", C is
unrelated quantum-physics text).

## Arm isolation is mandatory, not an optimization

Each arm runs in its **own subprocess** by default (`ONNX_REF_MODE=isolated`,
the default `run.sh` and `bench.py` invocation). This is load-bearing, not
a nicety -- do not "simplify" it back into one process.

A follow-up investigation into ROW 217's torch instability found the cause
was not thread config (`torch.get_num_threads()` was verified live at 1/1
in every condition below, quiet box, single-threaded, no concurrent
contention):

| condition (all thread-verified 1/1) | torch ms/sentence | CoV% |
|---|---|---|
| torch run AFTER the ORT arm, same process (the old convention) | 9.5136 | 4.23% |
| torch run in ISOLATION, own process | 16.7208 | 4.39% |
| isolation + 2s single-threaded busy-loop before timing (control) | 17.9403 | 4.37% -- no effect, refutes CPU pre-warming as the mechanism |
| isolation, `taskpolicy -c utility` | 10.7182 | 11.00% -- unquotable |
| isolation, 8x `yes` contention | 14.6597 | 12.71% -- unquotable |

ORT was unaffected throughout (5.6408ms, CoV 1.89%, within 1.7% of three
prior independent takes) -- **the artifact is 1.76x on torch alone**, driven
purely by whether ORT ran first in the same process. The mechanism is OS
scheduling/QoS placement on this M1 Max hybrid chip (8 P-core + 2 E-core);
the exact kernel knob is unresolved (`powermetrics`/`spindump` need root).

Decision: an incumbent arm must measure the incumbent, not the incumbent
warmed by a different framework's process state. Per-arm process isolation
removes the variable instead of picking a side of it -- **the isolated
number is canonical**. Consequence: `docs/discipline.md` ROW 195's torch
cell (10.0360 ms/sentence) is retracted -- it sits between the two
escalated readings above and ROW 195's own script used the same
ORT-then-torch-in-one-process convention, so it was almost certainly a
torch-after-ORT reading, not torch's steady state.

`ONNX_REF_MODE=same-process` reproduces the old one-process convention on
demand, strictly as a labeled control to keep the artifact reproducible in
this harness's own committed output -- never report a `same-process` torch
number as the incumbent number.

The parent process in isolated mode asserts each child's PID via a result
JSON handoff (`ONNX_REF_RESULT_JSON`) and refuses to report a result whose
`worker_pid` doesn't match the PID it spawned -- a harness that *requests*
isolation and one that *gets* isolation must not emit identical output; this
is that assertion.

## BGE_MODEL_PATH convention

Same convention as `bge_eval.rs`: the model path is never hardcoded into a
tracked file, machine-specific paths are never written into source, and the
harness skips cleanly (exit 0 for `run.sh`, an early `sys.exit(1)` from
`bench.py` when invoked directly) when `BGE_MODEL_PATH` is unset or missing.

**`BGE_MODEL_PATH` MUST point at the exact same `model.onnx` the Rust
harness loads (`proxima-onnx/examples/bge_eval.rs`'s own `BGE_MODEL_PATH`).
If the two arms load different ONNX exports, the cell is invalid** -- a
different export toolchain/opset changes the node graph ONNX Runtime sees,
which changes which fusions its optimizer fires, which changes the number
this lane is trying to measure. Verified concretely: the file this repo's
own Rust example was run against (133,093,490 bytes, matching
`docs/discipline.md` ROW 195's own cited size) has **1,244** raw graph
nodes; a from-scratch `torch.onnx.export` of the same HF checkout (below)
has **1,504** -- +260 nodes, concentrated in `Constant` (+132), `Sqrt`
(+36), `Cast` (+26), `Mul` (+24), `Shape` (+15), `Slice` (+12), plus five
op types (`ConstantOfShape`, `Equal`, `Expand`, `Gemm`, `Tanh`) the shared
file never emits at all -- a structurally different graph, not a
byte-identical re-export. Under ONNX Runtime's own graph optimizer
(`GraphOptimizationLevel.ORT_ENABLE_ALL`), the shared file's raw 1,244
nodes collapse to **351** -- exactly ROW 195's cited "executes 351 nodes on
this graph" -- via `LayerNormalization` (25, fusing the `ReduceMean`/`Sub`/
`Pow`/`Sqrt`/`Div` chain), `BiasGelu` (12, fusing the bias-add into `Erf`'s
GELU), and `FusedMatMul` (12); a from-scratch export is not guaranteed to
hit the same fusion patterns and was not used for any reported number in
this README.

Locate the shared file the same way you'd resolve any other
machine-cached, per-host asset this repo depends on but never vendors: the
Rust harness already names its own resolution (`BGE_MODEL_PATH` env var,
no default, see `bge_eval.rs`'s own doc comment on `MODEL_PATH_ENV`) --
point this harness's `BGE_MODEL_PATH` at whatever `model.onnx` you already
point that at. Do not hardcode that path here; it lives outside this repo
and is host-specific.

**Fallback only, when no shared file is reachable on this machine:**
`export_model.py` produces a `model.onnx` offline from the HF cache's
`model.safetensors` (the HF cache snapshot,
`~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/<sha>/`,
holds only `model.safetensors` -- no `model.onnx` as of this writing):

```
$ .venv/bin/python export_model.py
exported model.onnx to <this dir>/model/model.onnx (133718479 bytes)
```

It exports via `torch.onnx.export` (opset 14, dynamic batch/sequence axes),
tracing with sentence A's token ids as the example input shape. The output
goes under this directory's `model/` (gitignored), never into the HF cache
and never into a tracked file. **This is a fallback for a machine that
lacks the shared file, not the default arm** -- per the node-count finding
above, a fallback-export cell must be labeled as such and never compared
directly against a number produced from the shared file.

## Venv and pinned versions

```
$ uv venv --python 3.12 .venv     # or: python3.12 -m venv .venv
$ uv pip install --python .venv/bin/python \
    torch==2.5.1 transformers==4.46.3 onnxruntime==1.20.1 "numpy<2" onnx==1.17.0
```

`python3.12` is pinned deliberately: torch/onnxruntime wheel availability
lags the newest CPython release (3.14 has neither as of this writing).
`run.sh` uses `uv` if present, else falls back to stdlib `venv` + `pip`.

Resolved transitive versions (from a real install, informational only --
only the direct pins above are enforced): `numpy==1.26.4`,
`tokenizers==0.20.3`, `safetensors==0.8.0`, `huggingface-hub==0.36.2`.

## Exact invocation

```
$ BGE_MODEL_PATH=/path/to/model.onnx ./run.sh
```

Optional env vars: `ONNX_REF_VENV_DIR` (venv location, default `./.venv`),
`ONNX_REF_PYTHON` (interpreter, default `python3.12`), `ONNX_REF_RUNS`
(runs per arm, default `5`).

Re-running `bench.py` directly against an already-built venv:

```
$ BGE_MODEL_PATH=/path/to/model.onnx .venv/bin/python bench.py
```

## Output

Final table: `arm | ms/sentence | CoV% | n_runs | pid | threads` in isolated
mode (the `pid` column is the isolation proof -- two distinct PIDs, one per
arm), or without the `pid` column in `same-process` mode, plus the printed
thread settings actually in effect and the cosine sanity-check lines for
both arms, to stderr/stdout as `bge_eval.rs` does.

**Every measurement below is labeled loaded-host.** This box had a sibling
agent (`proxima-wt-narrowtile`) building intermittently throughout this
session; the quiet gate (`pgrep`/`ps` for live `cargo`/`rustc`/`nextest`
processes, two checks 60s+ apart) read QUIET before and after every run
below, but 1-minute load average never fully decayed between bursts (a new
build kept starting within the polling window) -- so these are the honest
numbers this contended box produced, not a clean-box reproduction of the
1.76x investigation's numbers. Every run taken is reported; none discarded.

Isolated mode (`ONNX_REF_MODE=isolated`, canonical), three takes, same
model, same box, same session:

| take | onnxruntime ms/sentence | CoV% | onnxruntime pid | torch ms/sentence | CoV% | torch pid | load condition |
|---|---|---|---|---|---|---|---|
| 1 | 5.5639 | 0.63% | 45249 | 21.6177 | 29.13% | 45253 | quiet at start, a sibling `cargo`/`rustc` build started mid-run (visible in the process gate immediately after) -- torch run 2 has a 68.68ms outlier |
| 2 | 5.5817 | 0.82% | 45717 | 12.9132 | 16.05% | 45728 | process-gate quiet both ends, 1m loadavg 8.21 decaying from take 1's build; torch's 5 per-run means decline monotonically 15.28ms->10.40ms (decay artifact, not steady-state noise) |
| 3 | 5.6097 | 0.52% | 48116 | 10.0582 | 6.11% | 48117 | process-gate quiet both ends, 1m loadavg 9.62 rising to 10.93 (another sibling burst started) |

onnxruntime is stable across all three takes (5.56-5.61ms, CoV <1%) --
consistent with the original investigation's finding that ORT is unaffected
by process contention. torch ranges **10.06-21.62 ms/sentence across takes,
CoV 6.11%-29.13%** on this contended box; report as this range, not a point
estimate. Take 3's torch CoV (6.11%) is the closest of the three to the
clean-box isolated finding (4.39%) and its mean (10.0582ms) is closest to
the clean-box same-process reading (9.5136ms) -- both consistent with the
mechanism finding (OS scheduling/QoS placement, not thread config: threads
were verified 1/1 in every take above).

Same-process control (`ONNX_REF_MODE=same-process`), one take, run
immediately after isolated take 3, same load conditions (1m loadavg
11.75->10.97, process-gate quiet):

| arm | ms/sentence | CoV% | n_runs | threads |
|---|---|---|---|---|
| onnxruntime | 5.5837 | 2.10% | 5 | intra_op=1 inter_op=1 sequential (1-pass warmup) |
| torch | 8.0756 | 15.09% | 5 | num_threads=1 num_interop_threads=1 (3-pass warmup) |

Direction reproduces the artifact even under contention: same-process
torch (8.0756ms) < isolated take 3's torch (10.0582ms) -- same-process
still reads faster than isolated, consistent with the clean-box finding
that ORT-then-torch-in-one-process warms torch's later reads. The
magnitude is attenuated by contention noise (both cells' CoV >5%), which is
exactly why `same-process` is retained only as a control, never as the
reported incumbent number.

All four runs above reproduced the exact ROW 195 cosine triple:
`cosine(A,B)=0.936311 cosine(A,C)=0.378777 cosine(B,C)=0.334176`.

Re-run on a genuinely quiet box (verify with `uptime` -- 1m/5m/15m load
averages all near the machine's idle baseline, not just the process gate)
for a clean-box number:

```
$ BGE_MODEL_PATH=/path/to/model.onnx ./run.sh
```
