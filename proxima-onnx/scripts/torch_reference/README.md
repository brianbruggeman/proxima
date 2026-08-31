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
- `accuracy_check.py` — evaluates the first 1000 `t10k` test images and
  reports top-1 accuracy, plus a zero-input logit-parity check against the
  logits `real_mnist_checkpoint.rs` printed on its own last run
  (`real_mnist evaluation SUCCEEDED: ... values=[...]`, hardcoded here as
  `REFERENCE_ZERO_INPUT_LOGITS`). Fails (nonzero exit) if the max abs diff
  exceeds `1e-3`.
- `inference_bench.py` — batch=1 inference latency: `--threads N` sets
  `torch.set_num_threads`, 50-image warmup, `--runs K` measured runs,
  reports p50/p95/p99/mean/CoV.
- `train_bench.py` — a 784-128-10 MLP (Adam, `CrossEntropyLoss`, batch 32,
  not `mnist.onnx` itself — that checkpoint carries no training graph),
  20 warmup steps then `--steps K` measured steps, reports
  p50/p95/mean/CoV per step.

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
