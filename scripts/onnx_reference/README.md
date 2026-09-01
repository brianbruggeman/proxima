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
1-pass warmup. Both arms print the thread settings actually read back from
the runtime, not the requested values -- `torch.set_num_interop_threads`
raises if parallel work already started in-process, and that failure is
caught and printed rather than swallowed.

Same sanity check `bge_eval.rs` runs: cosine(A,B similar) must exceed both
cosine(A,C) and cosine(B,C) (A/B are paraphrases of "a cat on a mat", C is
unrelated quantum-physics text).

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

Final table: `arm | ms/sentence | CoV% | n_runs | threads`, plus the printed
thread settings actually in effect and the cosine sanity-check lines for
both arms, to stderr/stdout as `bge_eval.rs` does.

Measured against the shared file (loaded-host -- three other agents were
building concurrently on this box; torch's CoV in particular reflects that,
not the harness):

| arm | ms/sentence | CoV% | n_runs | threads |
|---|---|---|---|---|
| onnxruntime | 5.6682 | 1.76% | 5 | intra_op=1 inter_op=1 sequential (1-pass warmup) |
| torch | 14.8845 | 16.79% (loaded-host) | 5 | num_threads=1 num_interop_threads=1 (3-pass warmup) |

Both arms reproduced the exact ROW 195 cosine triple:
`cosine(A,B)=0.936311 cosine(A,C)=0.378777 cosine(B,C)=0.334176`.
