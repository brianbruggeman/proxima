# Omega 06: diffusion

**Prerequisites:** [page 02](./02-programs-as-data.md) and [page 05](./05-speech-whisper.md), which established that these lowerings are graph functions with finite-difference tests.

**You will:** see the four graph shapes an image diffusion transformer needs that a text transformer does not, run two of their tests, and read what the image path covers and what it does not.

**New concepts (in order):** adaLN modulation · norm without affine · taps for a conditioned stack · why the timed step is not a lowering · what omega does not do yet.

## 1. What exists beyond page 05

Page 05 already covered non-causal attention, multi-axis rotary tables and layer taps; a diffusion transformer uses the same three, so they are not repeated here. What it adds is the conditioning path: a timestep embedding becomes a per-sample scale and shift, and those two vectors replace the affine weights inside the norm.

| piece | function | what it does |
|---|---|---|
| adaLN modulation | `modulate`, `proxima-tensor/src/spec/generic_lowering.rs:128` | `normed * (1 + scale) + shift`, with one `[d]` scale and shift per sample |
| norm without affine | `layer_norm_no_affine`, `generic_lowering.rs:51` | normalise, then let the caller supply the affine as two nodes |

`modulate` is four lines of graph and the reason it is a function rather than a layer type: `shift` and `scale` are caller-supplied nodes (`generic_lowering.rs:122-124`). In a diffusion model they come from the timestep embedding; in another model they could come from anywhere. The lowering never needs to know which.

The test composes `layer_norm_no_affine` and `modulate` and checks the result against the row formula written out by hand:

```sh
cargo val -p proxima-model-interop --features std -E 'test(layer_norm_no_affine_then_modulate_matches_the_naive_row_formula)'
```

```text
PASS proxima-model-interop generic_lowering::layer_norm_no_affine_then_modulate_matches_the_naive_row_formula
SUMMARY  passed=2 failed=0 ignored=0 not_run=0  wall=68.283s discovered=1220 run=2 filtered=1218 binaries=54
```

## 2. Differentiable for the same reason

Diffusion training backpropagates through the conditioning path, so the finite-difference check has to perturb `shift` and `scale`, not only the activations:

```sh
cargo val -p proxima-autograd \
  -E 'test(modulate_gradient_matches_finite_differences_for_input_shift_and_scale)'
```

```text
PASS proxima-autograd generic_lowering_grad::modulate_gradient_matches_finite_differences_for_input_shift_and_scale
SUMMARY  passed=3 failed=0 ignored=0 not_run=0  wall=2.361s discovered=203 run=3 filtered=200 binaries=24
```

The same file checks `non_causal_attention_heads` (one builder, head-first) for `q`, `k` and `v` (`proxima-autograd/src/generic_lowering_grad.rs:222`) and both rope and tap lowerings, so a diffusion graph built from these pieces is differentiable end to end without a second implementation.

## 3. What omega does not do yet

- The FLUX.2 klein image path exists for Metal. `proxima-model-interop/src/diffusion/` holds the transformer (`dit.rs`), the Euler-step host (`host.rs`), the VAE decode (`vae.rs`) and a PNG writer (`png.rs`); `proxima-model-interop/examples/flux2_klein.rs` runs it end to end and `examples/image_psnr.rs` scores the result against the reference image. `proxima-model-interop/specs/flux2-klein/SPEC.md:3` lists S1-S3 landed and S4 onward open.
- The numbers against `stable-diffusion.cpp` (`master-1052-f5df5f8`) are below, with the raw logs they come from. They are logs, not scoreboard-contract records (n=3, one prompt; the contract needs 20 runs per arm), so treat them as direction. `proxima-model-interop/specs/flux2-klein/SCOREBOARD.md` keeps the earlier dated sets as history.
- This page's earlier claim that there was no scheduler driver, VAE decode or checkpoint reader was true on 2026-11-21 and has been false since the image path landed. The checkpoint at `~/models/image_gen/flux2-klein-4b` is read by that example; `~/models/diffusion_pixel/ddpm-cifar10-32` is still opened by nothing.
- CPU runs the text path only; there is no CPU end-to-end run. CUDA is blocked on a driver mismatch on the one NVIDIA host, and wgpu has no run.

## 4. FLUX.2 klein 4B on Metal, measured 2026-11-24

M1 Max 64 GB, 1024x1024, seed 42, 4 Euler steps, wall from `/usr/bin/time -l`. Parity was last scored on 2026-11-22 (32.58 dB with our text, 49.16 dB with reference text) and not re-scored on 2026-11-24; the 2026-11-24 images sit unscored next to the logs.

**Interleaved rebench, 11:42-11:53.** Per round: `u1` (today's code), `t1` (control binary), sd.cpp, back to back, three rounds. Raw: `~/.cache/kvgate/proxnum699/rebench/{u1,t1,sd}-r{1,2,3}.log`. Medians of three, range in brackets.

| stage | u1 (today) | t1 (control) | sd.cpp | u1 / sd.cpp |
|---|---|---|---|---|
| text | 4.69 s (4.69-4.70) | 4.68 s (3.95-4.75) | 5.09 s (4.85-5.95) | 0.92x |
| DiT, 4 steps | 25.59 s (25.57-25.62) | 25.76 s (25.59-25.99) | 41.79 s (41.67-41.81) | 0.61x |
| VAE decode | 3.66 s (3.57-3.91) | 3.73 s (3.61-3.81) | 12.28 s (12.20-12.29) | 0.30x |
| wall | 48.87 s (44.43-51.89) | 43.99 s (40.84-45.38) | 59.80 s (59.66-76.71) | 0.82x |
| peak footprint | 12.32 GB | 12.31-12.32 GB | 14.80 GB | 0.83x |

Per-run wall: `u1` 51.89 / 48.87 / 44.43 s, `t1` 45.38 / 43.99 / 40.84 s, sd.cpp 76.71 / 59.66 / 59.80 s. The sd.cpp round 1 wall is unexplained. sd.cpp loads: qwen tensors 5.22 / 4.14 / 4.35 s (read 5.06 / 3.85 / 4.08 s), DiT tensors 1.87 / 1.87 / 1.74 s (read 1.66 / 1.53 / 1.52 s). Order per round was `u1`, `t1`, sd.cpp, so `u1` rounds 2 and 3 ran straight after an sd.cpp run.

**Load against file-backed pages.** `vm_stat` was taken at the start of each `u1` (A), `t1` (B) and sd.cpp (C) run (`{A,B,C}-r{1,2,3}.vm`, 16 KiB pages). The wall spread inside our arms is the DiT load.

| round | u1 DiT load | file-backed pages (A) | t1 DiT load | file-backed pages (B) |
|---|---|---|---|---|
| 1 | 16.1 s | 567,751 | 9.5 s | 856,404 |
| 2 | 13.3 s | 664,476 | 7.9 s | 937,722 |
| 3 | 9.2 s | 851,597 | 6.3 s | 1,123,457 |

Load falls as the file-backed count rises, in all six runs. sd.cpp's loads in the same window do not vary that way. Three rounds, one prompt: a direction, not a fit. Why `u1` loads slower than `t1` at a similar page count is not established.

**Warm steady state at `5d5167b864`.** Three warm runs outside the window. Raw: `~/.cache/kvgate/proxnum698/t1-f{2,3,5}.log`.

| stage | runs 2 / 3 / 5 | median |
|---|---|---|
| text | 4.18 / 4.23 / 4.22 s | 4.22 s |
| DiT, 4 steps | 24.97 / 25.46 / 25.47 s | 25.46 s (mean 25.30 s) |
| VAE decode | 3.58 / 3.61 / 3.56 s | 3.58 s |
| DiT load | 6.2 / 6.2 / 6.2 s | 6.2 s |
| wall | 40.33 / 40.85 / 40.82 s | 40.82 s (mean 40.67 s) |
| peak footprint | 12.31 / 12.31 / 12.32 GB | 12.31 GB |
| max RSS | 17.11 / 17.11 / 17.12 GB | 17.11 GB |

Text-only runs (`~/.cache/kvgate/proxnum693/e4-tc-{1,2,3}.log`, binary not recorded) took 1.92 / 1.91 / 1.90 s against 4.2 s for the text stage inside a full run; the difference is not resolved.

**Cause of the earlier cold runs (PROXNUM-698, `5d5167b864`).** Dropping the model released the mapping residency sets but not the warm residency set, which covers every cached buffer, the text mapping's included. `release_warm_residency` now runs next to `release_all_mapping_residency` in the context teardown (`omega/src/metal/metal_context.rs`, `omega/src/metal/warm_resident_buffers.rs`). The warm table above was taken with that fix; no before/after pair on an otherwise identical box is recorded.

Next: [07: comparisons](./07-comparisons.md) is where the measured numbers in this series live, with the command, the hardware, the run count and the variation.
