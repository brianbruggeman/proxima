# Execution worker command register

All final commands below ran from `/Users/brianbruggeman/repos/slot-0/proxima`.
Cargo builds used `CARGO_TARGET_DIR=/private/tmp/proxima-omega-execution-target`.
Tests are debug executions; these assertion runs are not performance benchmark cells.
No commits were made. The source freeze records the four files edited after root integration.

## Main assertion commands

| Command (with the target environment above) | Opened raw log | Observed rows |
|---|---|---|
| `cargo nextest run -p omega --lib --no-default-features --features std,cpu -E 'test(runtime_config_) \| test(config_plan_) \| test(config_cpu_budget_) \| test(config_selection_)' --no-capture` | `omega-main-config-cpu-selection.txt` | 34 run / 34 passed: runtime config 12, common planner 12, CPU resource 5, selection 5. The existing selection case captures a real CPU warmup and five raw samples, and retains an unavailable preferred wgpu candidate. |
| `cargo nextest run -p proxima-tensor --lib -E 'test(programmable_binding_) \| test(bind::tests::) \| test(evaluate_quantized_matmul_matches_dequantize_then_f32_evaluate) \| test(evaluate_quantized_applies_the_composed_gate_over_a_multi_axis_packed_contraction) \| test(qwen35_batched_packed_ssm_projection_matches_repeated_rows) \| test(evaluate_quantized_exact_matches_dequantize_then_f32_evaluate_near_exactly)' --no-capture` | `omega-main-binding-legacy-packed.txt` | 45 run / 45 passed: new binding 8, existing binder 33, existing packed CPU 4. The binding projection asserts BF16 literal outputs in both fusion modes and the broadcast stride. |
| `cargo nextest run -p omega --lib --features instrument,wgpu-backend -E 'test(config_metal_) \| test(config_wgpu_)' --test-threads 1 --no-capture` | `omega-main-metal-wgpu.txt` | 8 run / 8 passed: Metal 4, wgpu 4, serial GPU execution. Raw reports contain the real heterogeneous CPU/Metal/wgpu samples; the winner is executed against literal vectors. |
| `cargo nextest run -p omega --lib --no-default-features --features std,cpu,cuda-driver -E 'test(config_plan_explicit_unavailable_driver_does_not_fallback)' --no-capture` | `omega-main-cuda-availability-probe.txt` | 1 run / 1 passed; actual `Driver("cuda driver library unavailable")` on this Mac. CUDA hardware execution cases: 0. |
| `cargo nextest list -p omega --lib --features instrument,wgpu-backend,cuda-driver -E 'test(config_metal_) \| test(config_wgpu_) \| test(config_cuda_)'` | `omega-main-final-gpu-list.txt` | Four CUDA, four Metal, four wgpu cases compiled and listed after import cleanup. This listing does not execute CUDA hardware cases. |
| `cargo test -p omega --doc --no-default-features --features std,cpu backend::plan_named_with_config` | `omega-main-configured-doctest.txt` | 1 doctest passed; literal output `[-1,-2]`. |

The Metal raw census has six distinct device objects with capacities `[16,16,16,48,16,32]`. Its bounded refusal records zero uploads, output allocations and dispatches. Wgpu census has twelve physical objects with capacities `[16,16,16,16,24,16,16,16,24,16,16,16]`. Three wgpu refusal rows record `(allocations,uploads,dispatches,submits)=(0,0,0,0)`. Its first admitted execution records `(12,4,2,1)` and repeat records `(8,4,2,1)`. Census identities for wgpu are per-execution ordinals assigned by actual `Buffer` equality, with hash collisions resolved using equality; the capture retains no device objects. These scoped capacities exclude compiler/driver metadata and total RSS.

## Tier and lint commands

| Command | Opened raw log | Observed compiler output |
|---|---|---|
| `cargo check -p omega --no-default-features --features alloc` | `omega-main-alloc-check.txt` | Finished; no error diagnostics. |
| `cargo check -p omega --no-default-features --features std` | `omega-main-std-check-r2.txt` | Finished; no error diagnostics. The earlier `omega-main-std-check.txt` preserves the irrefutable-pattern diagnostic repaired with `Result::err`. |
| `cargo check -p omega --no-default-features --features std,wgpu-backend` | `omega-main-gpu-only-check.txt` | Finished; no error diagnostics, CPU feature absent. |
| `cargo clippy -p omega -p proxima-tensor --lib --no-default-features --features omega/std,omega/cpu,proxima-tensor/config` | `omega-main-final-cpu-clippy.txt` | Finished; no diagnostics. |
| `cargo clippy -p omega -p proxima-tensor --lib --features omega/instrument,omega/wgpu-backend,omega/cuda-driver,omega/metal-tiled-gemm` | `omega-main-final-clippy-r4.txt` | Finished; no diagnostics. `omega-main-final-clippy.txt` preserves two missing CUDA diagnostic arms; the source now returns typed unsupported diagnostic errors for those arms. |

Luna separately ran main-tree AC4/AC5/AC6: `omega-final-ac4.txt` printed 8 run / 8 passed; `omega-final-ac5.txt` printed 1 run / 1 passed; `omega-final-ac6.txt` printed 8 run / 8 passed / 0 skipped. Those runs followed this worker's released GPU window.

## Native independent recount

Target environment: `CARGO_TARGET_DIR=/private/tmp/proxima-omega-recount-check-target`.

```text
cargo run --example omega_programmable_recount -- \
  /private/tmp/proxima-omega-unified-config/proxima-tensor/specs/omega-programmable/results/ouro-configured-control \
  /private/tmp/proxima-omega-unified-config/proxima-tensor/specs/omega-programmable/results/ouro-configured \
  /Users/brianbruggeman/repos/slot-0/proxima/proxima-tensor/specs/ouro-llama-oracle/results/llama-v1/llama.jsonl \
  /private/tmp/proxima-omega-unified-config/proxima-tensor/specs/omega-programmable/results/recount
```

`omega-main-recount-r2.txt` records this invocation. Its generated `recount/timing-recount.jsonl` has 25 rows: 16 model timing cells, 6 micro timing cells, 2 llama cells and 1 oracle register. `recount/coordinate-comparisons.jsonl` has 187 rows: 168 paired payload fields, 18 paired projection vectors and 1 comparator control. The control at line 19 has an empty positive difference array and one negative-control coordinate `(reference=2, actual=3, signed_delta=1)`. Every changed signed coordinate remains in the generated artifact; there is no aggregate performance verdict here.

The first invocation used an absent oracle path. Its log is `omega-main-recount.txt`; its partial outputs were moved to `recount-failed-path` before the canonical invocation. The missing file is not treated as an oracle result.
