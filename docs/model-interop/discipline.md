# model interop discipline log

Correctness is the first gate for this initiative; performance is second and
RAM is third. The active component is the WGPU broadcast-reduce execution path.

## Semantic smoke prompts

Every generative model/backend cell should include these fixed prompts in
addition to numerical parity checks:

1. `What is the capital of France?` — expected answer contains `Paris`.
2. `Give the first five lines of Hamlet's "To be, or not to be" soliloquy.`
   — expected output begins with the canonical opening and contains five
   verse lines, allowing tokenizer and formatting variation.

These are behavioral smoke tests, not substitutes for reference-logit or
gold-text tests; a pass only proves the model did not obviously lose task
semantics.

## C2 — CUDA precompiled-PTX driver boundary

| Build | Tests | Clippy | Micro-bench | Compare-bench | E2E | Opt | SIMD/SM/no-Box | O(1) | Cfg/API | Home-turf | Δ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cargo check -p omega --features cuda-driver` = 0; `cuda_smoke` = 0 | Precompiled PTX add-one returns exactly `[2.0, 3.0, 4.0]`; generated NVRTC path compiles and executes with temporary CUDA runtime/header packages | Not run | Not a performance measurement; one driver smoke dispatch | Real Qwen2.5 0.5B and 1.5B CUDA launches complete; CPU/CUDA logits are not yet parity-correct | CUDA context, memory query, PTX load, launch, synchronize, and readback are proven; full emitted graph executes through NVRTC on the local host | Correctness/deployment boundary only; no speed claim | PTX targets `sm_52`; runtime driver JITs it on the local RTX 2070 | Model run reports GPU memory before/after; no persistent arena conclusion yet | Added `CudaDriver::precompiled_function`; existing launch ABI reused | CUDA driver is the home-turf arm; toolkit-free smoke and generated-graph paths are both executable | `+correctness`: gather ABI and broadcast-reduce warp reduction defects fixed; parity remains open | Generated CUDA C still depends on an NVRTC deployment; `PROXIMA_CUDA_INCLUDE_PATH` supplies headers when the host toolkit is absent. |

## C1 — WGPU broadcast-reduce execution

| Build | Tests | Clippy | Micro-bench | Compare-bench | E2E | Opt | SIMD/SM/no-Box | O(1) | Cfg/API | Home-turf | Δ | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cargo check -p omega --features wgpu-backend` = 0; `std,vulkan` example check = 0 | Runtime parity RED: Vulkan embedding output is finite but `cpu_max_abs_diff=61.843845`; first residual divergence is layer 0 at `2.4511533` | Not run | Existing `embed_local` captures forward time; no isolated WGPU kernel micro-bench yet | No meaningful incumbent arm yet; correctness comparison is CPU evaluator vs WGPU for the same bound graph | Vulkan Qwen3 Embedding 0.6B Q4_K_M runs end-to-end, but parity is not sealed | Correctness fix only: broadcast outputs now use full-rank coordinates and contiguous output strides; no speed claim | WGSL subgroup path retained; no dynamic dispatch or boxed path added; serial broadcast path remains explicitly rejected | Per-dispatch coordinate arrays and fixed binding tables; no new steady-state map | Existing `WgpuPlan`/`emit_wgsl` surfaces reused; no new public API | Home-turf arm pending until CPU parity is green | `+correctness`: one-element output defect removed; parity still RED | Reuse existing `BoundOp`/`Shapes` and the Metal broadcast contract. Correctness > performance > RAM. |

**Measured baseline before C1 change:** Vulkan `embed_local` on
`Qwen3-Embedding-0.6B-q4_k_m.gguf`, `last`, `hello`: `forward_ms=1127.493`,
`cpu_max_abs_diff=63.206184`, normalized first values were `[1.0, 0.0, ...]`;
layer trace localized the first mismatch to layer 0 (`2.56273`).

**Measured C1 result:** the normalized vector is no longer one-hot and the
forward remains in the same range (`1121.065 ms`), but parity is still RED:
`cpu_max_abs_diff=61.843845`, first residual mismatch `2.4511533`. The change
fixed output coverage, not the remaining numerical cause.

**Negative isolation result:** the named generic incumbent/reference arm for
packed Q4_K matmul passes on this same WGPU/Vulkan host (`cargo test ...
packed_q4k_matmul_matches_the_dequantized_f32_cpu_path_on_wgpu`: `1 passed`),
and the complete WGPU parity fixture passes (`11 passed`). The real-model
failure therefore is not yet attributed to the generic Q4_K decoder. A new
packed-Q4_K embedding gather fixture also passes (`1 passed`), narrowing the
remaining cause further toward the real model's fused early-layer graph.

**Allocation budget:** correctness-first diagnostic path may allocate the
existing output/readback buffers; no additional persistent model copy is
accepted until parity is green.

**Re-prove commands:**

```text
cargo check -p omega --features wgpu-backend
cargo check -p proxima-model-interop --example embed_local --features 'std,vulkan'
cargo run -p proxima-model-interop --example embed_local --features 'std,vulkan' -- /home/bix/models/Qwen3-Embedding-0.6B-q4_k_m.gguf vulkan last hello
PROXIMA_COMPARE_LAYERS=1 cargo run -p proxima-model-interop --example embed_local --features 'std,vulkan' -- /home/bix/models/Qwen3-Embedding-0.6B-q4_k_m.gguf vulkan last hello
```

### Changelog

| Date | Change | Δ vs prior | CoV / runs | Host loadout |
|---|---|---|---|---|
| 2026-09-11 | baseline: corrected WGPU output allocation, then measured layer-0 divergence | baseline; correctness RED | single diagnostic run; no perf claim | local Vulkan adapter |
| 2026-09-11 | port broadcast-reduce full-rank coordinate/output-stride handling from the existing Metal contract | output coverage corrected; `63.206184 → 61.843845` max error; one-hot output removed; `1127.493 → 1121.065 ms` is not a performance claim | one before/one after; CoV not established | same local Vulkan adapter |
| 2026-09-11 | isolated generic Q4_K matmul and full WGPU parity suite | kept; Q4_K arm `1/1`, complete WGPU parity `11/11`; no causal claim for the real-model failure | deterministic test counts | same local Vulkan adapter |
| 2026-09-11 | added and ran packed-Q4_K embedding-gather parity arm | kept; `1/1`, no mismatch | deterministic | same local Vulkan adapter |
| 2026-09-11 | added CUDA residual-tap comparator, fixed broadcast-reduce warp-sum propagation, and corrected packed matmul row addressing | node 40 `0.170126021 → 0.000000075`; node 83 `0.32524231 → 0.000000194`; node 96 `4.872974395 → 0.000004053`; CUDA 0.5B generation completes in `TTFT=1064.242 ms`, `TTNT=1021.128 ms`; full logits still require parity investigation | deterministic test `101/101`; one real checkpoint comparison and one generation run | local RTX 2070, driver 610.57.04 |
| 2026-09-11 | instrumented output-set parity and found cached-attention bind fusion changes node 85 from 9 operands to 4; removed that fusion from default serving features; added `omega::config::RuntimeConfig` with conflaguration-backed correctness-first policy | first-layer CPU/CUDA max error `0.4930767119 → 0.0132417679` with fusion disabled; final logits remain RED at `12.4743958` | one controlled before/after comparison; `cargo check` and `101/101` CUDA tests pass | local RTX 2070, driver 610.57.04 |
| 2026-09-11 | added semantic smoke prompt contract and classified/reran local task variants; forced CUDA quantized reductions to serial accumulation as a correctness control; ran Paris prompt and Vulkan embedding | CUDA generation still fails semantic smoke (`"atoiчатtery..."`); Qwen3 Vulkan embedding runs but parity is RED (`cpu_max_abs_diff=73.8947601`); reranker CPU returns `yes_probability=0.8167788` | task probe `2/2`; reranker single run; CUDA `101/101`; one Vulkan run | RTX 2070 Vulkan/CUDA; Qwen3 0.6B and Qwen2.5 0.5B |
| 2026-09-11 | traced CUDA node 109 to the CPU relaxed Q6_K activation-dot oracle rather than a CUDA address/decode defect; made exact activation arithmetic the default serving policy | with `PROXIMA_EXACT_ACTIVATIONS=1`, node 109 max error `0.0241808 → 0.00000536`, first-layer max error `0.0241808 → 0.00000536`, final logits `6.0719 → 0.0006912`; exact-default semantic generation is still wrong against Ollama | controlled real Qwen2.5 0.5B comparison and semantic reference; model-interop `147 passed, 19 ignored`; omega CUDA `101/101` | local RTX 2070, driver 610.57.04 |
| 2026-09-11 | measured one-shot versus serialized CUDA prefill, implemented zero-length CUDA input/intermediate buffers and skipped zero-work launches, then bound the two-range cache at exact logical length | batch/ubatch `32/32` and `1/1` produced identical IDs `[75960, 29540, 2439, 119534]`; TTFT `1148.550 ms` vs `7342.595 ms`, TTNT `1082.851 ms` vs `1083.498 ms`; semantic output remains incorrect against Ollama (`Paris`) | two controlled CUDA runs; omega `101/101`; tensor `579/579`; interop `147/147` | local RTX 2070, driver 610.57.04 |
| 2026-09-11 | ran an all-F32 weight diagnostic (`PROXIMA_FORCE_F32_WEIGHTS=1`) to separate quantization from graph/model semantics | first token remained `75960`, identical to the packed-weight run; quantization is not the primary cause of the Paris semantic failure | one controlled F32-vs-packed CUDA comparison; build check passed | local RTX 2070, driver 610.57.04 |
| 2026-09-11 | froze the Qwen2.5 0.5B tokenizer/reference contract and ran the CPU semantic baseline | checkpoint SHA `74a4da8c…d7a9db`; Proxima raw-token IDs exactly match Ollama `[3838,374,279,6722,315,9625,30]`; CPU still emits `75960` instead of reference `576` (` The`) | tokenizer probe build/run passed; one CPU generation run: `TTFT=29694.123 ms`, `ids=[75960]` | local CPU; exact raw template, no BOS |
| 2026-09-11 | bound metadata-present Qwen2 Q/K/V projection biases and added them before Q/K RoPE and V caching | position-zero discriminator What changed to reference `11414` (` percentage`); France first token changed to reference `576` (` The`); 8-token continuation exactly matches `[576,6722,315,9625,374,12095,13,576]` and contains Paris | controlled CPU runs: What TTFT `4371.738 ms`; France-8 TTFT `30524.576 ms`, TTNT `4395.682 ms`; no GPU claim yet | local CPU; Qwen2.5 0.5B SHA `74a4da8c…d7a9db` |
| 2026-09-11 | measured the metadata-driven bias graph on CUDA and serialized prefill | CUDA `32/32` and `1/1` match the CPU/reference prefix `[576,6722,315,9625]`; CUDA `32/32` reaches Paris for 8 tokens with identical IDs; post-bias node/logit comparator still needs to be run | CUDA `32/32`: TTFT `1655.440 ms`, TTNT `1052.441 ms`; serialized `1/1`: TTFT `7481.088 ms`, TTNT `1046.587 ms`; GPU free memory `6963648 → 7023040 KiB` | local RTX 2070, driver `610.57.04`, NVRTC runtime/header packages |
| 2026-09-11 | corrected one-shot diagnostic cache binding and replaced stale numeric probes with layer-residual probes | fresh-cache one-shot now binds `KV_BOUND=0`; CPU/CUDA direct logits select reference `576`; layer-0 max error `1.3828e-5`, final-logit max error `1.8930e-4` | controlled CUDA comparator after bias/cache fixes; exact activations; `compare_local` exit 0 | local RTX 2070, driver `610.57.04`, NVRTC runtime/header packages |
| 2026-09-11 | threaded validated Q/K/V projection-bias capability through the single-range dense builder while preserving the bias-free API | single-range graphs can now bind `blk.{layer}.attn_{q,k,v}.bias`; existing bias-free graph callers remain unchanged | tensor `579/579`; interop `147/147`; CUDA example compile passed; runtime single-range parity requires the macOS/Metal placement path | local CPU test host; CUDA compile host |
| 2026-09-11 | made WGPU/Vulkan dispatch 2D-limit aware and linearized WGSL global IDs across x/y workgroups | Qwen2.5 0.5B Vulkan no longer aborts at `151936 > 65535` workgroups; it completes, but parity remains RED: layer-0 max `3.6462116` at index `490`, final max `19.006298`, CPU top `576` vs Vulkan top `76779` | WGPU parity `12/12`; one real Vulkan comparator and one generation run; TTFT `1759.956 ms`, TTNT `1023.252 ms`; 8 finite but incorrect tokens | local Vulkan adapter |
| 2026-09-11 | hardened topological Vulkan bisection to exclude packed input leaves, zero-sized nodes, and unaligned mapped-byte readbacks | actual first nonempty divergent computed node is `70` (`Reduce`), `max_abs=0.001953125` at index `407`, CPU `6712.6143` vs Vulkan `6712.6123`; final logits remain `19.006298`, top `576` vs `76779` | two controlled diagnostic runs with `ulimit -c 0`; no core dump created | local Vulkan adapter |
| 2026-09-11 | added targeted node-70 lineage reporting | node `70` reduces node `69`; node `69` (`Elementwise`) already differs by `0.0003662109375` at index `21551` (CPU `1709.8334`, Vulkan `1709.833`), while node `70` reaches `0.001953125`; this rules out a reduction-only explanation | one controlled lineage run with `ulimit -c 0`; no core dump created | local Vulkan adapter |
| 2026-09-11 | bisected node-69’s computed lineage through nodes `56`, `43`, and `41` | node `41` (`Reduce` over node `40`) is only `1.4305115e-6` off; node `40` is `3.5762787e-7` off; node `43` carries `7.6293945e-6` at its worst coordinate, and node `21` is an input leaf rather than a computed activation. The earliest tested drift is therefore in the upstream reduction/product path, with coordinate-dependent accumulation, not bias decoding | three controlled probes with `ulimit -c 0`; no core dump created | local Vulkan adapter |
| 2026-09-11 | added a default-off WGPU serial `Reduce(Add)` control and ran it on the Qwen2.5 0.5B Vulkan path | hypothesis falsified: serializing plain Add reductions worsened node `69` to `0.0010986328` and node `70` to `0.0029296875`; final logits stayed `19.006313`, top `576` vs `76779` | one controlled ablation with `ulimit -c 0`; default WGPU parity remains the reference path | local Vulkan adapter |
| 2026-09-11 | continued upstream lineage through nodes `50`, `42`, `39`, and `38` | node `38` product max `2.3841858e-7`; node `39` reduction max `2.3841858e-6`; node `42` add max `3.8146973e-6`. Drift begins before the reduction under test and compounds across coordinate-dependent activation/product operations | one controlled node-39 probe with `ulimit -c 0`; no core dump created | local Vulkan adapter |
| 2026-09-11 | traced the first material amplification to reciprocal node `35` and tested one Newton refinement | node `34` square-root drift is `3.7252903e-9`; direct Vulkan reciprocal differs from host reciprocal of the Vulkan input by `3.8146973e-6`. Newton refinement produced no change in node-35 or final-logit results, so the control was reverted | one provenance probe and one controlled refinement run; WGPU parity `12/12`; no core dump created | local Vulkan adapter |
| 2026-09-11 | threaded `NumericPolicy` through WGPU and CUDA planning/emission and added a backend capability boundary for WGPU reduction epilogues | `bit_exact` now forbids cooperative reduction reassociation; relaxed policy explicitly enables the shuffle-tree path. WGPU parity passed `12/12`, CUDA cooperative-emission test passed, and combined CPU/CUDA/WGPU compilation passed | explicit policy propagation; WGPU keeps serial lowering for unsupported broadcast epilogues; `git diff --check`; no temporary Proxima trees or coredumps | local Vulkan adapter and RTX 2070 |
| 2026-09-11 | replayed real Vulkan node-34 values through a minimal `Input -> Reciprocal` plan | three replay runs were deterministic and bit-identical to full-graph node `35`; the `7.6293945e-6` difference from host `1.0/f32` is inherited from node `34`'s Vulkan value, not reciprocal graph wiring or buffer binding | `PROXIMA_RECIPROCAL_REPLAY=1`, three runs; full-graph node-35 max error `0`; no core dump created | local Vulkan adapter |
| 2026-09-11 | replayed real CUDA node-34 values through the same minimal `Input -> Reciprocal` plan | all three replay runs were deterministic and bit-identical to both host `1.0/f32` and full-graph node `35`; the CUDA reciprocal is not the source of the CUDA model mismatch | `PROXIMA_RECIPROCAL_REPLAY=1`, three runs; NVRTC include/library paths supplied from the existing local cache; no core dump created | local RTX 2070 |
| 2026-09-11 | replayed real node-33 values through a minimal `Input -> SquareRoot` plan | Vulkan and CUDA replay outputs were deterministic for three runs and bit-identical to their respective full-graph node `34`; Vulkan differed from host `sqrt()` by at most `9.3132257e-10`, while CUDA was bit-identical. The node-34 operation is therefore not the first backend-specific divergence; the difference enters through node `33`'s producer/input | `PROXIMA_SQRT_REPLAY=1`; explicit Vulkan/CUDA drivers; no core dump created | local Vulkan adapter and RTX 2070 |
| 2026-09-11 | replayed real node-32 values through a minimal epsilon-add plan using the checkpoint's declared `1e-6` RMS epsilon | Vulkan and CUDA replay outputs were deterministic for three runs and bit-identical to both host `value + epsilon` and full-graph node `33`; the node-33/34/35 unary tail is not the source of the model mismatch | `PROXIMA_EPSILON_REPLAY=1`; explicit Vulkan/CUDA drivers; no core dump created | local Vulkan adapter and RTX 2070 |
| 2026-09-11 | replayed real node-31 values through a minimal mean-square multiply plan using captured node-3 scalar `0.0011160715` | Vulkan and CUDA replay outputs were deterministic for three runs and bit-identical to both host multiplication and full-graph node `32`; the first backend-specific divergence is therefore at node `31` or earlier, not the multiply or later RMSNorm tail | `PROXIMA_MEAN_SQUARE_REPLAY=1`; explicit Vulkan/CUDA drivers; no core dump created | local Vulkan adapter and RTX 2070 |
| 2026-09-11 | compared node-31 replay against a host 32-lane CUDA-style reduction tree (`16, 8, 4, 2, 1`) | CUDA was bit-identical to the canonical tree across three runs; Vulkan was deterministic and matched the full graph but differed from that tree by at most `1.4901161e-8`. WGSL subgroup reduction therefore uses a different order than CUDA, while both differ from CPU left-to-right accumulation | `PROXIMA_SUM_SQUARES_REPLAY=1`, three runs per backend; no production kernel changed; no core dump created | local Vulkan adapter and RTX 2070 |
| 2026-09-11 | replaced WGSL `subgroupAdd` for cooperative `Reduce(Add)` with an explicit width-derived shuffle tree | Vulkan node-31 replay now matches the CUDA canonical tree bit-for-bit across three runs; WGPU parity remains `12/12`. Full-model first divergence moved/holds at node `71`; final max changed only from `19.006298` to `19.006294`, with the same incorrect top token, so this fix is correct but not sufficient to seal the model | `omega/src/wgsl.rs` production change; real Vulkan replay; no core dump created | local Vulkan adapter |
| 2026-09-11 | replayed real node-68 and node-70 values through the node-71 elementwise `Add` | Vulkan and CUDA replay outputs were deterministic for three runs and bit-identical to both host `f32` addition and full-graph node `71`; node 71 only exposes upstream reduction drift | `PROXIMA_RESIDUAL_ADD_REPLAY=1`; explicit Vulkan/CUDA drivers; no core dump created | local Vulkan adapter and RTX 2070 |
| 2026-09-11 | froze the CPU node-69 tensor and replayed node 70's rank-5 `Reduce(Add)` on CPU, Vulkan, and CUDA | all three runs per backend were deterministic; Vulkan and CUDA agreed with each other but differed from the CPU replay by `0.0009765625`. The shared GPU reduction order, not input plumbing or node-71 addition, is the remaining parity boundary | `PROXIMA_NODE70_REPLAY=1`; explicit Vulkan/CUDA drivers; no core dump created | local Vulkan adapter and RTX 2070 |
| 2026-09-11 | replayed real node-30 values through the exact `Reduce(Add, Zero)` maps for node `31` | Vulkan and CUDA replay outputs were deterministic for three runs and bit-identical to full-graph node `31`, but differed from host left-to-right accumulation by `2.8312206e-7` and `2.682209e-7` respectively. The first backend-specific divergence is reduction lowering/order; the earlier global serial WGPU control remains unsuitable because it worsened later model output | `PROXIMA_SUM_SQUARES_REPLAY=1`; explicit Vulkan/CUDA drivers; no core dump created | local Vulkan adapter and RTX 2070 |

**Honest read:** the change corrected a real output-coverage bug, but the WGPU
path is not correct yet; the remaining first-layer mismatch must be isolated
before any performance or memory conclusion.

**Frozen reference:** `fixtures/reference_contract.json` records the exact
Qwen2.5 0.5B checkpoint SHA, raw-template policy, prompt IDs, and semantic
oracles. `examples/tokenize_local.rs` is the executable probe used to produce
the token-ID result. Tokenization is therefore not the current explanation
for the CPU/CUDA Paris failure; the next boundary is model execution/logits.

**Qwen2 bias correction:** Qwen2 checkpoints carrying
`blk.0.attn_q.bias` now cause all per-layer Q/K/V bias tensors to be bound and
added at the projection boundary. Bias-free checkpoints retain the prior
graph. The real Qwen2.5 0.5B CPU smoke now matches the frozen eight-token
reference continuation and reaches `Paris`; backend parity is still measured
separately.

**Implication:** C1 is not sealed. The next correctness experiment must compare
the first layer's packed-weight/activation subgraph against CPU, not optimize
dispatch or RAM.

**C2 host measurement:** local `RTX 2070`, `driver 610.57.04`,
`free_bytes=7340752896`, `total_bytes=8172404736`; precompiled PTX returns
`[2.0, 3.0, 4.0]`. Temporary `nvidia-cuda-nvrtc-cu12` and
`nvidia-cuda-runtime-cu12` packages supplied NVRTC and headers, allowing the
generated CUDA-C graph to execute Qwen2.5 0.5B and 1.5B. The 0.5B comparator
found layer-0 probe divergence at node 96 (`4.872974395`) and final logit
divergence (`18.175670623` in the exact-activation diagnostic), so CUDA is
executable but not yet correctness-sealed.

| 2026-09-11 | traced the first material amplification to reciprocal node `35` and tested one Newton refinement | node `34` square-root drift is `3.7252903e-9`; direct Vulkan reciprocal differs from host reciprocal of the Vulkan input by `3.8146973e-6`. Newton refinement produced no change in node-35 or final-logit results, so the control was reverted | one provenance probe and one controlled refinement run; WGPU parity `12/12`; no core dump created | local Vulkan adapter |
