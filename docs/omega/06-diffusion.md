# Omega 06: diffusion

**Prerequisites:** [page 02](./02-programs-as-data.md) and [page 05](./05-speech-whisper.md), which established that these lowerings are graph functions with finite-difference tests.

**You will:** see the four graph shapes an image diffusion transformer needs that a text transformer does not, run two of their tests, and read what still stands between those pieces and a PNG.

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

The same file checks `non_causal_attention` for `q`, `k` and `v` (`proxima-autograd/src/generic_lowering_grad.rs:222`) and both rope and tap lowerings, so a diffusion graph built from these pieces is differentiable end to end without a second implementation.

## 3. What omega does not do yet

- No image path. `proxima-model-interop/specs/flux2-klein/SPEC.md:3` is a draft with no code and no runs. Its requirements name what a real path owes: latents reproducible from `(seed, shape)`, text conditioning that matches the reference, a flow-matching noise schedule matched step for step, the diffusion transformer's predicted velocity (how much to move the latent this step) and the latent after each step, a VAE decode, and a final image compared by PSNR (peak signal-to-noise ratio). None of that exists as code.
- No scheduler driver. A flow-matching or Euler schedule is a data structure plus a loop over it. What exists in `proxima-tensor/src/spec/generic_lowering.rs` is lowerings; nothing there advances a schedule.
- No VAE decode. There is no decoder stack and no image writer. `proxima-autograd/src/conv.rs:223` is a 1-D convolution; an image VAE needs 2-D.
- The checkpoints are on disk and unread. `~/models/image_gen/flux2-klein-4b` (diffusers layout plus one safetensors) and `~/models/diffusion_pixel/ddpm-cifar10-32` both exist; nothing opens either.
- The comparison has nothing to compare. `~/repos/others/stable-diffusion.cpp/build/bin/sd-cli` exists (`master-1052-f5df5f8`), but with no proxima image path there is no wall-time row to measure. It needs a proxima-side path plus a checkpoint both sides load.

Next: [07: comparisons](./07-comparisons.md) is where the measured numbers in this series live, with the command, the hardware, the run count and the variation.
