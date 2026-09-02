# torch reference harness for mnist.onnx

Reference implementation for the incumbent (PyTorch) evaluated on the exact
`mnist.onnx` checkpoint `proxima-onnx/tests/real_mnist_checkpoint.rs` and
`tests/real_mnist_accuracy.rs` test against
(`~/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx`), so
proxima-onnx's parity and performance claims against PyTorch have a
re-runnable oracle instead of a number pasted into a doc row. See guiding
principle 16 (execution must not outrun proof) and principle 14 (the
incumbent wins on correctness) — this harness exists so both hold
mechanically, not by memory of a prior run.

This directory is a bench/reference fixture, not production code. The
workspace's no-python rule applies to everything else in this repo; this is
the named, scoped exception, in service of measuring the named incumbent
(PyTorch) on its own terms.

## What each script re-proves

- `model.py` — ports `mnist.onnx`'s 14-node graph (3x `Conv`+`Relu`,
  `BatchNormalization`, `Flatten`, `Gemm`+`Relu`, `Gemm`,
  `BatchNormalization`, `LogSoftmax`; dumped with `onnx.load` against the
  same checkpoint the rust tests read) into an equivalent `torch.nn.Module`,
  loading the model's 18 initializers verbatim.
- `data.py` — idx3/idx1 reader for `~/.cache/burn-dataset/mnist`, same
  normalization `(pixel/255 - 0.1307) / 0.3081` as
  `real_mnist_accuracy.rs`'s own `load_normalized_images`.
- `diagnostics.py` — shared by `inference_bench.py`/`train_bench.py`: prints
  `torch.get_num_threads()`/`torch.get_num_interop_threads()`/
  `torch.__config__.parallel_info()`/the `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` env vars as seen by the
  process, and the load average, immediately before the timed loop — and
  raises `SystemExit` if a single-thread request did not actually land on
  one thread, so a mis-configured run cannot silently report a number under
  a config it never had.
- `mac_count_check.py` — computes `MnistNet`'s (this file's own module, not
  a hand re-derivation) per-layer MAC count and cross-checks it against
  `proxima-tensor/docs/discipline.md`'s cited figures (conv1 48,672 / conv2
  663,552 / conv3 1,672,704 / fc1 371,712 / fc2 320, total 2,756,960
  MACs/image); nonzero exit on mismatch.
- `accuracy_check.py` — evaluates the first 1000 `t10k` test images and
  reports top-1 accuracy, plus a zero-input logit-parity check against the
  logits `real_mnist_checkpoint.rs` printed on its own last run
  (`real_mnist evaluation SUCCEEDED: ... values=[...]`, hardcoded here as
  `REFERENCE_ZERO_INPUT_LOGITS`). Fails (nonzero exit) if the max abs diff
  exceeds `1e-3`.
- `inference_bench.py` — batch=1 inference latency: `--threads N` sets
  `torch.set_num_threads`, 50-image warmup, `--runs K` measured runs,
  reports p50/p95/p99/mean/CoV, bracketed by `diagnostics.py`'s thread/load
  reporting.
- `train_bench.py` — a 784-128-10 MLP (Adam, `CrossEntropyLoss`, batch 32,
  not `mnist.onnx` itself — that checkpoint carries no training graph),
  20 warmup steps then `--steps K` measured steps, reports
  p50/p95/mean/CoV per step, bracketed by `diagnostics.py`'s thread/load
  reporting.

## Reference numbers

Measured 2026-08-31, on this repo's own M1 Max mac, Accelerate BLAS backend
(torch's default on macOS/arm64, confirmed no NNPACK/OpenMP compiled in on
this build), **host was loaded** (background load average 20-94 during the
run — read `p50` as trustworthy, `mean` as noisy, per guiding principle 19's
evidence ladder):

| measurement | value | provenance |
|---|---|---|
| 1-thread inference p50 | 0.1193 ms/image | `inference_bench.py --threads 1` |
| 8-thread (default) inference p50 | 0.6802 ms/image | `inference_bench.py --threads 8`, fork-join overhead dominates on ops this small — a negative result, not a bug |
| accuracy (first 1000 t10k images) | 0.9900 (990/1000) | `accuracy_check.py` |
| logit parity vs `real_mnist_checkpoint.rs` zero-input reference | 3.86e-6 max abs diff | `accuracy_check.py` |
| 1-thread train-step (MLP, batch 32) p50 | 0.3406 ms/step | `train_bench.py --threads 1` |

The 8-thread number is a genuine finding, not noise: PyTorch's default
thread pool loses to single-threaded execution on this model because every
op (784x128 matmul, 3x3 convs on 26x26/24x24/22x22 feature maps) is too
small to amortize fork-join dispatch. Do not read the default thread count
as PyTorch's best case.

## Re-verification session (2026-09-01)

Re-run with the given, pinned venv (`torch==2.5.1`, `onnx==1.17.0`, NOT this
directory's own `requirements.txt` pin of `torch==2.13.0`/`onnx==1.22.0` --
a real version delta, named as a plausible mechanism for any drift below,
not tuned away). Host: Apple M1 Max, macOS, arm64. **Host loadout: LOADED**
throughout this session -- three sibling worktree sessions' own `cargo`/
`rustc`/`nextest` processes were visible via `pgrep` on every check (24-46
matching processes, never this session's own `cdb-daemon`/`sccache`),
`uptime` load average 5-19 during the python arms, higher during the rust
arms; every number below is LOADED-HOST, not clean.

`mac_count_check.py`: **PASSED** -- `MnistNet`'s own per-layer MAC counts
(computed from the live module, not hand re-derived) match this doc's cited
figures exactly: conv1 48,672, conv2 663,552, conv3 1,672,704, fc1 371,712,
fc2 320, total 2,756,960.

`accuracy_check.py`: accuracy 0.9900 (990/1000), exact match to the prior
session. Zero-input logit parity: max abs diff 1.431e-06 (prior session:
3.86e-6, this session's own torch 2.5.1 run is tighter, both single-ULP-
class float32 noise).

5 separate process invocations each, `--runs 200` / `--warmup 20 --steps
200`, thread config verified via `diagnostics.py` before every timed loop
(`torch.get_num_threads()=1`, confirmed, every run -- the fail-loud gate
never fired):

| run | inference p50 (ms) | inference mean (ms) | CoV% | train p50 (ms) | train mean (ms) | CoV% |
|---|---|---|---|---|---|---|
| 1 | 0.1723 | 0.1816 | 18.36% | 0.3250 | 0.3086 | 13.02% |
| 2 | 0.1523 | 0.1528 | 1.03% | 0.2870 | 0.3327 | 26.82% |
| 3 | 0.1569 | 0.1575 | 2.31% | 0.3313 | 0.3126 | 13.48% |
| 4 | 0.1572 | 0.1614 | 6.58% | 0.2676 | 0.2725 | 9.53% |
| 5 | 0.1539 | 0.1552 | 2.73% | 0.3234 | 0.3089 | 12.69% |

Inference across-run p50: **0.1523-0.1723ms** (CoV>5% on 2/5 runs -- reported
as a range, not a point estimate). Train across-run p50: **0.2676-0.3313ms**
(CoV>5% on all 5 runs -- reported as a range).

**ROW 189's 0.119ms inference citation**: did NOT reproduce at point value --
this session's p50 range (0.1523-0.1723ms) sits 28-45% above it. **ROW
159's/157's 0.380/0.3406ms train-step citation**: did NOT reproduce exactly
either, but landed BELOW it this time (0.2676-0.3313ms, 3-30% under) --
opposite direction from the inference gap. Named mechanisms, neither
independently isolated this session: (1) torch 2.5.1 (this venv) vs 2.13.0
(the citation's own venv) is a real, confirmed version delta; (2) this
session's host was loaded the entire time (5-19 background load average
during the python arms) vs the citations' own quiet/loaded mix; (3) 200-run
per-invocation sample size is smaller than some prior sessions' 200-run
sweeps but comparable, not the likely driver. The inference and train gaps
moving in OPPOSITE directions versus their own citations is inconsistent
with "host load alone" as a single explanation (load should bias both the
same way) -- most consistent with the torch-version delta affecting the two
op mixes (conv-heavy vs GEMM-heavy) differently, but this is a hypothesis,
not measured directly this session.

## Re-proving these numbers

```sh
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python accuracy_check.py           # expect 0.9900 (990/1000), diff <= 1e-3
./venv/bin/python inference_bench.py --threads 1 --runs 200
./venv/bin/python inference_bench.py --threads 8 --runs 200
./venv/bin/python train_bench.py --threads 1
```

`accuracy_check.py` and `inference_bench.py` skip cleanly (print and return)
if `~/.cache/burn-dataset/mnist`'s `t10k` idx files are absent, the same
`#[ignore]`-and-skip convention `real_mnist_accuracy.rs` uses. All three
require the real, on-disk `mnist.onnx` checkout at
`~/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx` (`model.py`
hardcodes the same host-local path the rust tests use, for the same reason:
this is a host-local fixture, not something the repo ships).

Numbers on a different host, thread count, or under different background
load will differ, particularly `mean` (noise-sensitive) and the 8-thread
arm (contention-sensitive). `p50` and the qualitative single-thread-beats-
default-threads result are the load-bearing claims; re-run before trusting
absolute numbers on a busy machine.
