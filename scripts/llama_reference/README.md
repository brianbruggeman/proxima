# llama_reference

Committed incumbent-arm harness for the Metal GPU-decode lane (ROW 116/193's
own cited numbers -- 17.086 ms/token, 234.1 GB/s, 416.1 GMAC/s -- were
produced by a manually-built external checkout, invoked by hand, with no
committed script anywhere in the repo). Mirrors `scripts/burn_reference` /
`scripts/onnx_reference`'s convention: pinned external checkout, never
assume a default, real-data-only, print what actually ran.

## What it measures

llama.cpp's OWN `llama-bench` binary (never our FFI -- our FFI links a
CPU-only ggml sibling tree, confirmed by grep: no `Metal` symbol resolves
through it) against the SAME `openchat-3.5-1210.Q4_K_S.gguf` checkpoint
`proxima-model-interop/src/serving.rs::DEFAULT_MODEL_PATH` and
`proxima-tensor/benches/bench_q4k_matmul.rs`'s own `gguf_path()` default
already point at.

## External checkout

`~/repos/others/llama.cpp`, pinned SHA `b25346221dadb9101aa9dda55431dde4d3596943`
(`b2534622` short form matches ROW 193's own citation -- **verified this
session**, `git rev-parse HEAD` on the checkout returns this exact SHA, so
the checkout has not drifted since ROW 193). Not vendored, not fetched by
this script -- clone/checkout it yourself first.

## Build (already built on this host; flags recorded from `CMakeCache.txt`)

```sh
cd ~/repos/others/llama.cpp
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=Apple
cmake --build build --config Release -j
```

Produces `build/bin/llama-bench` (354456 bytes) and `build/bin/libggml-metal.dylib`
(607168 bytes) alongside it, both Mach-O arm64, backend row prints
`Metal,BLAS`.

## Model

`~/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf`
-- **4,140,385,376 bytes** on disk (7.24B params, Mistral architecture, 32
layers, Q4_K_S quant), verified present via `ls -la` this session. Same file
every arm in this lane loads -- no separate copy for the incumbent.

## Exact invocation

```sh
./run.sh
```

Env overrides: `LLAMA_CPP_CHECKOUT` (default `~/repos/others/llama.cpp`),
`LLAMA_BENCH` (default `$LLAMA_CPP_CHECKOUT/build/bin/llama-bench`),
`MODEL_PATH` (default the openchat file above), `LLAMA_REF_N_GEN` (default
32), `LLAMA_REF_REPETITIONS` (default 5), `LLAMA_REF_THREADS` (default 8).

Underlying command: `llama-bench -m <model> -n 32 -r 5 -t 8` (defaults
`-ngl 99` full GPU offload, `-b 2048 -ub 512`). `llama-bench` was preferred
over `llama-cli` (an older ROW's citation used `llama-cli -ngl 33 -t 8 -n 24`)
because it reports its own repeated-run mean +/- stddev directly, giving a
CoV without a wrapper script summing over hand-parsed `llama-cli` runs.

## Thread-count asymmetry -- stated loudly, not buried

**`-t 8` gives llama-bench 8 CPU threads for its own host-side orchestration
(tokenization, sampler, graph build/schedule) even though `-ngl 99` puts
every layer on the GPU.** proxima's own Metal decode loop
(`proxima-model-interop/src/bind.rs::real_openchat_file::runs_the_cached_decode_loop_on_the_metal_backend_and_reports_the_plan_cache`)
spawns **zero** worker threads for its CPU-side orchestration (`prepare`,
`emit`, `pipeline_lookup`, `op_setup`, `block_upload`, `encode_dispatch`) --
confirmed by `grep -n "thread::spawn\|rayon\|std::thread"` over
`omega/src/metal.rs` and `proxima-model-interop/src/{generate,bind}.rs`
returning nothing. Every orchestration tick in `token_breakdown_metal`'s own
counters runs on the single calling thread. **This is not a like-for-like
cell on CPU-thread count: llama-bench gets 8, proxima's decode loop gets 1.**
The gap this lane measures conflates GPU-kernel throughput with this
thread-count asymmetry on the orchestration side; STEP 3's own attribution
(`proxima-tensor/docs/discipline.md`, this row) shows GPU-kernel time (not
orchestration) is the dominant term regardless, so the asymmetry does not
overturn the headline ratio -- but it does mean the orchestration slice of
the gap (the ~11-13 ms/token term) is partly attributable to thread count,
not purely to CPU-side algorithm cost, and neither ROW 116 nor ROW 193 nor
this row isolates that split.

## CoV, both arms (this session, loaded host)

| arm | measured | n | CoV |
|---|---|---|---|
| llama.cpp Metal (`tg32`, `-t 8`) | 48.57 +/- 1.26 t/s | 5 reps (harness-internal) | 2.59% |
| proxima Metal, GPU-kernel-only (`gpu_exec_ms`, steady-state steps 1-7) | mean 56.527 ms/token | 7 | 0.41% |
| proxima Metal, full decode step (`evaluate_ms`, steady-state steps 1-7) | mean 67.917 ms/token | 7 | 1.71% |

Neither ROW 116 nor ROW 193 recorded a CoV for the llama.cpp arm; this run's
2.59% (`llama-bench`'s own internal repeated-run stddev) closes that gap.
