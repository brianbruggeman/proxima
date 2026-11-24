# TASKS: FLUX.2 [klein] 4B through omega (PROXNUM-588)

One slice = one green commit = one validation command. Update the checkbox and the one-line
note in the SAME commit as the slice. Fixture env: `PROXIMA_FLUX2_KLEIN_DIR=~/models/image_gen/flux2-klein-4b`,
`PROXIMA_FLUX2_KLEIN_REF=~/.cache/kvgate/proxnum588/ref`. Commands run from the worktree root.
A missing fixture is a failure; `0 passed` is red.

- [x] **S0 oracle dumps and measured floor.**
  Built a patched copy of sd.cpp outside the repo (`~/.cache/kvgate/proxnum588/sdcpp-dump/`), ran the reference command, dumped noise, text embedding, 4 step latents, velocity-1 and the VAE output; Metal vs Metal is bit-identical, CPU vs Metal conditioner floor measured (`rms_rel` 0.00353, cos 0.99999376, maxabs_rel 1.009 -> metric changed to `rms_rel`); Metal confirmed from the log; reference-image RMS measured 0.7326 -> PSNR ceilings re-derived 26.4 dB (VAE) and 22.6 dB (e2e); thresholds written into SPEC.md.
  Validate: `for f in noise text step1 step2 step3 step4 velocity1 vae_out; do test -f ~/.cache/kvgate/proxnum588/ref/$f.f32 || echo missing $f; done` prints nothing; `grep -c "mu=2.29" ~/.cache/kvgate/proxnum588/ref/ref.log` prints 1.
- [x] **S1 Philox noise.** Philox4x32-10 + Box-Muller generator, config key `rng`; `std_default` and `cpu` refused with a named reason.
  Validate: `cargo val -p proxima-model-interop --test flux2_klein philox::` -> `3 passed` (AC2).
- [x] **S2 flow-match scheduler + Euler step.** Config `mu` coefficients, `steps`, `image_seq_len`; the Euler update is one Elementwise.
  Validate: `cargo val -p proxima-model-interop --test flux2_klein scheduler::` -> `3 passed` (AC4).
- [x] **S3 generic lowerings, no model attached.** `layer_norm_no_affine` + `modulate`, `non_causal_attention_heads` (one builder, head-first), `multi_axis_rope`, `tap_layers`, each with a non-FLUX unit test.
  Validate: `cargo val -p proxima-model-interop generic_lowering::` -> `4 passed` (4 of the 6 in AC10; the other 2 land in S7).
- [ ] **S4 text encoder.** `arch/flux2.toml` text section + Qwen3 bind through the existing HF path; tokenizer and template from config; taps 9/18/27 concatenated. First task: read sd.cpp's pad-mask behaviour and the safetensors tensor names, record both here.
  Validate: `cargo val -p proxima-model-interop --test flux2_klein text_embedding::` -> `4 passed` (AC3).
- [ ] **S5 DiT single-stream block.** Bind `single_blocks.0`, feed the dumped block input, compare the block output.
  Validate: `cargo val -p proxima-model-interop --test flux2_klein dit::single_block` -> `1 passed`, `maxabs_rel` printed and within the effective threshold.
- [ ] **S6 DiT double-stream block, then the full DiT.** Double block first (`dit::double_block` -> `1 passed`); then pre-blocks (img_in, time_in, txt_in, shared modulations), 5 double + 20 single blocks, final layer; velocity at step 1; stage ordering drops the text encoder before the DiT is resident.
  Validate: `cargo val -p proxima-model-interop --test flux2_klein dit::` -> `6 passed` (AC5; the 4 latent checks use the S2 Euler step).
- [ ] **S7 VAE decoder.** Generic `group_norm` and `nearest_upsample_x2`, conv2d wired into the lowering, VAE attention, unpatchify, de-normalise from checkpoint statistics, `post_quant_conv`.
  Validate: `cargo val -p proxima-model-interop --test flux2_klein vae::` -> `4 passed` (AC6), and `cargo val -p proxima-model-interop generic_lowering::` -> `6 passed` (AC10).
- [ ] **S8 end-to-end example + PSNR tool.** `examples/flux2_klein.rs` composes noise, text pipe, scheduler loop, DiT, VAE and a PNG writer; `examples/image_psnr.rs` (Rust).
  Validate: `cargo run --release -p proxima-model-interop --example flux2_klein -- --seed 42 --out ~/.cache/kvgate/proxnum588/omega.png`, then `cargo run --release -p proxima-model-interop --example image_psnr -- ~/.cache/kvgate/proxnum588/ref.png ~/.cache/kvgate/proxnum588/omega.png` -> 1024 x 1024 and `PSNR >= 22.0 dB` (AC7).
- [ ] **S9 genericity audit.** Grep gate for model names in `src/` and omega (AC9).
  Validate: `git grep -n -i "flux\|klein" -- omega/src proxima-model-interop/src ':!*.toml' ':!*.md'` prints 0 lines; `ls proxima-model-interop/src/arch` lists 3 files.
- [ ] **S10 timed comparison.** `scripts/flux2_klein_ab.sh`: 3 alternating runs per side, quiet window, per-stage and RSS tables recorded in this file as measurements (the verdict on the 5% bar belongs to the owner).
  Validate: `scripts/flux2_klein_ab.sh` prints `runs=3 per side`, 5 per-stage rows per side and one wall-time ratio line; CoV <= 5% for each side (AC8).

Order: S0 first (everything compares to it). S1, S2, S3 are independent after S0. S4 needs S3. S5 and S6 need S2 and S3. S7 is independent of S4 to S6. S8 needs S1, S2, S4, S6, S7. S9 after S8. S10 last.

Notes log (one line per landed slice, appended in the slice's own commit):
- 2026-11-20 S3: generic layer_norm_no_affine, modulate, non_causal_attention_heads (one builder, head-first; shares the mixer softmax-mix core), multi_axis_rope tables (codec-encoded), tap_layers; `generic_lowering::` 4 passed.
- 2026-11-21 S0: reference dumps (8 raw + sidecars), Metal wall 84.58 s / 16.19 GiB RSS, Metal bit-repeatable, CPU-vs-Metal conditioner `rms_rel` 0.00353 (maxabs_rel 1.009 -> gate is `rms_rel`), image RMS 0.7326, ceilings 26.4/22.6 dB.
