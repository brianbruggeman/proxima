# Omega 06: diffusion

**Prerequisites:** [page 02](./02-programs-as-data.md) and [page 03](./03-train.md) (the same two facts as page 05: a graph over named buffers, and a lowering is trainable when autograd differentiates it).

**You will:** see which diffusion-image-generation pieces exist as generic lowerings, run two of their tests, and read precisely what is missing between those pieces and an image. Like page 05, this page is short because the honest state is short: the lowerings exist, the model path does not.

**New concepts (in order):** adaLN modulation · why modulation is a lowering and not a layer type · non-causal attention as the image-attention shape · layer norm without affine · layer taps for a conditioned stack · the flux.2 spec's requirements and status · what omega does not do yet.

## 1. What exists

A diffusion transformer needs four shapes that are not transformer-encoder shapes: a per-sample scale/shift from the timestep embedding (adaLN), attention with no causal mask, a normalisation whose affine comes from that modulation rather than from weights, and the ability to stack chosen intermediate layers. All four are graph-building functions here, and none of them is diffusion-specific code:

| piece | signature | unit test |
|---|---|---|
| adaLN modulation | `modulate`, `proxima-tensor/src/spec/generic_lowering.rs:128` | `layer_norm_no_affine_then_modulate_matches_the_naive_row_formula` (`proxima-model-interop/src/generic_lowering.rs:47`) |
| non-causal attention | `non_causal_attention`, `proxima-tensor/src/spec/generic_lowering.rs:166` | `non_causal_attention_lets_every_query_see_every_key` (`proxima-model-interop/src/generic_lowering.rs:82`) |
| norm without affine | `layer_norm_no_affine`, `proxima-tensor/src/spec/generic_lowering.rs:51` | same test as `modulate` (they compose) |
| layer taps | `tap_layers`, `proxima-tensor/src/spec/generic_lowering.rs:337` | `tap_layers_stacks_the_chosen_layer_outputs_in_tap_order` (`proxima-model-interop/src/generic_lowering.rs:208`) |
| multi-axis rope tables | `multi_axis_rope_tables`, `proxima-tensor/src/spec/generic_lowering.rs:295` | `multi_axis_rope_rotates_each_section_by_its_own_position_stream` (`:146`) |

`modulate` is `normed * (1 + scale) + shift` with `shift` and `scale` one `[d]` vector each, broadcast over the sequence axis (`proxima-tensor/src/spec/generic_lowering.rs:122-124`). That is the whole reason it is a function and not a layer: the modulation values are graph nodes supplied by the caller — a timestep embedding, in a diffusion model; anything else, in a different model — so the lowering never needs to know which.

Run the pair test, which checks the composed `layer_norm_no_affine` → `modulate` against the row formula written out by hand:

```sh
cargo val -p proxima-model-interop --features std -E 'test(layer_norm_no_affine_then_modulate_matches_the_naive_row_formula)'
```

```text
PASS proxima-model-interop generic_lowering::layer_norm_no_affine_then_modulate_matches_the_naive_row_formula
SUMMARY  passed=2 failed=0 ignored=0 not_run=0  wall=68.283s discovered=1220 run=2 filtered=1218 binaries=54
```

## 2. Trainable, not just runnable

Diffusion training backpropagates through the modulation path, so the finite-difference check has to perturb `shift` and `scale`, not only the activations:

```sh
cargo val -p proxima-autograd \
  -E 'test(modulate_gradient_matches_finite_differences_for_input_shift_and_scale)'
```

```text
PASS proxima-autograd generic_lowering_grad::modulate_gradient_matches_finite_differences_for_input_shift_and_scale
SUMMARY  passed=3 failed=0 ignored=0 not_run=0  wall=2.361s discovered=203 run=3 filtered=200 binaries=24
```

The same file checks `non_causal_attention` gradients for `q`, `k` and `v` (`generic_lowering_grad.rs:222`) and the rope and tap lowerings, so a diffusion graph assembled from these pieces is differentiable end to end without a second implementation.

## 3. What omega does not do yet

- **No image path.** `proxima-model-interop/specs/flux2-klein/SPEC.md:3` is a draft with no code and no runs. The requirements name what a real path owes: bit-reproducible initial latents from `(seed, shape)` (R2), matched text conditioning (R3), a sigma-for-sigma flow-match schedule (R4), the DiT velocity and post-step latents (R5), a VAE decode (R6) and an image-level PSNR (R7). None of those exists as code.
- **No scheduler driver.** The flow-match/Euler schedule is a data structure plus a loop over it; nothing in `proxima-tensor` or `proxima-model-interop` implements either. The `generic_lowering.rs` additions from `b7c2d6b22` are lowerings, not a sampler.
- **No VAE decode.** There is no convolution stack driver for a decoder, and no image writer; `proxima-autograd/src/conv.rs:223` is a 1-D convolution, and an image VAE is 2-D.
- **The checkpoints are on disk and unread.** `~/models/image_gen/flux2-klein-4b` (diffusers layout plus one safetensors) and `~/models/diffusion_pixel/ddpm-cifar10-32` exist; nothing opens either.
- **The incumbent comparison has no shared artifact.** `~/repos/others/stable-diffusion.cpp/build/bin/sd-cli` exists (`master-1052-f5df5f8`), but omega has no image path to run against it, so the R8 wall-time row (`proxima-model-interop/specs/flux2-klein/SPEC.md:25`) cannot be a measured comparison yet. What it needs is a proxima-side image path and a checkpoint both sides load.

Next: [07: comparisons](./07-comparisons.md) is where every measured number in this series lives, with the command, the hardware, the run count and the coefficient of variation.
