# SPEC: FLUX.2 [klein] 4B through proxima / omega (PROXNUM-588)

Status: S0 landed 2026-11-21 (reference run, dumps, measured floors and thresholds below);
S1-S3 landed; S4 onward open. Every sd.cpp fact below is read from tag `master-1052-f5df5f8`
(`~/repos/others/stable-diffusion.cpp`, written `SD:`) and every proxima fact from this worktree.
Numbers tagged DERIVED are arithmetic on those facts; the S0 MEASURED block replaces them where
it applies.

## Problem

`black-forest-labs/FLUX.2-klein-4B` (Apache-2.0, step-distilled) has no path through
proxima. Done means: the fixed prompt and seed give an image within a stated PSNR of
stable-diffusion.cpp's, every intermediate matches its dump within a bf16-derived bound,
and wall time per image is within 5% of sd.cpp on this M1 Max, with a model that differs
from every other omega model only by config values (PROXNUM-586: no bespoke model path).

## Requirements

- R1. A fixed reference run exists (section Reference run) and its intermediates are dumpable.
- R2. Initial latent noise is bit-reproducible from `(seed=42, shape)` with no dump needed at run time.
- R3. Text conditioning (3 Qwen3 hidden layers concatenated, 7680 wide) matches the reference.
- R4. The flow-match sigma schedule matches the reference sigma-for-sigma.
- R5. The DiT velocity and the latent after each of the 4 Euler steps match the reference.
- R6. The VAE decode of a given latent matches the reference image.
- R7. The end-to-end image from the fixed prompt/seed matches the reference image by PSNR.
- R8. Wall time per image, per-stage times and peak RSS are measured against sd.cpp; the bar is within 5%.
- R9. No model-name branch in omega or interop code: all model specifics are config (arch TOML or the checkpoint's own `config.json`).
- R10. New lowerings are generic ops (norm, conv, group norm, upsample, multi-axis rope, non-causal attention), each usable by a model that is not FLUX.

## Reference run

Weights (all in `~/models/image_gen/flux2-klein-4b`, 16.13 GB for the three files, byte counts matched to HF headers per the card):
`flux-2-klein-4b.safetensors` (diffusion, 7.75 GB), `comfy/qwen_3_4b.safetensors` (text, 8.04 GB), `comfy/flux2-vae.safetensors` (VAE, 0.34 GB).

```
~/repos/others/stable-diffusion.cpp/build/bin/sd-cli -v \
  --diffusion-model ~/models/image_gen/flux2-klein-4b/flux-2-klein-4b.safetensors \
  --llm             ~/models/image_gen/flux2-klein-4b/comfy/qwen_3_4b.safetensors \
  --vae             ~/models/image_gen/flux2-klein-4b/comfy/flux2-vae.safetensors --vae-format flux2 \
  -p "a red fox sitting in fresh snow, morning light, photograph" \
  -s 42 -W 1024 -H 1024 --steps 4 --cfg-scale 1.0 \
  -o ~/.cache/kvgate/proxnum588/ref.png
```

No `--rng`, `--sampling-method` or `--scheduler` flags: the defaults are the reference. They resolve to:
Philox "cuda" RNG (`SD:src/stable-diffusion.cpp:335`, `rng_type = CUDA_RNG`; `get_rng` at `SD:src/pipeline/diffusion_engine.cpp:619-625`),
sampler Euler and scheduler `FLUX2_SCHEDULER` (`SD:src/model/integrations/flux2.cpp:44-46`, the `VERSION_FLUX2_KLEIN` block; resolution at `SD:src/pipeline/request.cpp:49-70`).
`-v` prints sigmas and stage timings. The card says Metal at runtime is unconfirmed; S0 confirms it from the log.
Flags checked present: `-p`, `-s`, `-W`, `--steps`, `--cfg-scale`, `--llm`, `--diffusion-model`, `--vae`, `--vae-format` (`SD:examples/common/common.cpp:426-476,1106,1215-1281,1761`); `-o` is taken from the card's flag list, not re-read.

### Measured reference (S0, 2026-11-20, M1 Max 64 GB, macOS 15.8)

Unpatched `sd-cli`, `-v`, wall clock from `/usr/bin/time -l`. Log lines cited as `ref.log:N`.

- Metal confirmed: components and params all on `MTL0` (`ref.log:73-75`); bf16 weights with `kernel_mul_mm_bf16_f32` and f16 flash attention (`kernel_flash_attn_ext_f16_dk128_dv128`, `ref.log:135-143`). The card's "Metal unconfirmed" is closed: it is Metal.
- Wall: **84.58 s real**; 3.08 s user, 15.66 s sys; maximum resident set size 17,380,147,200 B (16.19 GiB); peak memory footprint 14,639,417,280 B.
- Stages: conditioning 6.51 s (`ref.log:149`), sampling 47.04 s for 4 steps (`:182`), VAE decode 12.22 s (`:212`), `generate_image` 65.79 s (`:215`); the rest is model load (qwen 5.69 s, DiT 7.04 s, VAE 0.32 GB) and runtime pipeline compilation.
- Sigmas in the log: `[1, 0.967384, 0.908144, 0.7672, 0]`, `mu=2.291` (`ref.log:117`) -- equal to the DERIVED values in the table below.
- Dumps in `$K/ref/`: `noise.f32` (524,288), `text.f32` (512x7680), `step1.f32` to `step4.f32`, `velocity1.f32` (524,288 each), `vae_out.f32` (3x1024x1024), one `.txt` shape sidecar per dump, `ref.log`; `ref.png` is 1024x1024.
- Reference image: `vae_out` values in [0,1], RMS 0.732612 (= 186.8 of 255 levels), range [0,1].
- Owner bar (tightens AC8): omega wall time per image **<= 84 s**.

Floors measured:

- Metal repeat (same binary, same inputs): bit-identical on noise, text, velocity1, step1, step4, vae_out (max abs diff 0). The Metal path is deterministic.
- sd.cpp CPU vs Metal on the conditioner: `rms_rel` 0.00353, cosine 0.99999376, maxabs 15.06 on a reference with RMS 14.92 and range [-4595, 16360]. `maxabs_rel` is 1.009 -- the massive-activation channels make a max-over-elements metric unusable here (see Metric).
- Full CPU-vs-Metal DiT/VAE floor: `$K/dump-cpu-floor/` (same patched binary, `--backend cpu --params-backend diffusion=cpu,te=cpu,vae=cpu`). It is not needed for the gates below, because the conditioner floor is at least 7x under every ceiling even after doubling.

## What the reference does (read from source)

| Item | Finding | Source |
|---|---|---|
| Qwen3 layers used | `out_layers = {9, 18, 27}`: hidden state after block 9, 18 and 27 (1-based; the test is `out_layers.find(i + 1)`), raw, not the final norm (index 37 is not in the set, so `norm` is not applied). Concatenated along the hidden axis (ggml dim 0) in layer order: `[L9 / L18 / L27]` = 3 x 2560 = 7680. | `SD:src/conditioning/conditioner.hpp:3151-3154`; `SD:src/model/te/llm.hpp:2068-2103` |
| Prompt template | `"<\|im_start\|>user\n" + text + "<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n\n</think>\n\n"`; `prompt_template_encode_start_idx = 0`; `min_length = 512` (tokens padded up to 512 with pad id 151643; Qwen2 BPE tokenizer). | `SD:src/conditioning/conditioner.hpp:3151-3161`; pad id at `:2038-2040`; pad loop `SD:src/tokenizers/tokenizer.cpp:44-` |
| Text arch | `LLM::LLMArch::QWEN3` for `VERSION_FLUX2_KLEIN` | `SD:src/conditioning/conditioner.hpp:2022-2023` |
| RNG | Default `--rng cuda` = `PhiloxRNG` (Philox4x32-10, an AUTOMATIC1111 port of torch cuda randn). `randn(n)`: per element `i` counter = `[offset, 0, i, 0]`, key = `(seed low 32, seed high 32)`, 10 rounds, multipliers `0xD2511F53`, `0xCD9E8D57`, key bumps `0x9E3779B9`, `0xBB67AE85`; value = Box-Muller sine branch of output words 0 and 1: `u = x*2^-32 + 2^-33`, `v = y*2^-32*2pi + 2^-33*2pi`, `sqrt(-2 ln u) * sin(v)`. `offset` is 0 after `manual_seed` and advances by 1 per `randn` call. Other `--rng` values: `std_default` (`std::default_random_engine` + `normal_distribution`) and `cpu` (`CPU_RNG`). | `SD:src/core/rng_philox.hpp:11-88`; `SD:src/core/rng.hpp:11-40`; `SD:examples/common/common.cpp:812-818`; `SD:src/pipeline/diffusion_engine.cpp:619-625` |
| Noise call | `manual_seed(seed)` then ONE `randn_like(init_latent)` of shape `[W/16, H/16, 128, 1]`; element `i` of the flat buffer is Philox element `i` (W fastest). | `SD:src/pipeline/image.cpp:969-971`; `SD:src/core/tensor.hpp:276-286` |
| Latent shape 1024x1024 | VAE scale factor 16 for the flux2 VAE (`SD:src/model/vae/vae.hpp:185-186`), 128 latent channels (`SD:src/model/integrations/flux2.cpp:36`, `geometry = {1, 128, 3}`) gives `[64, 64, 128, 1]` = 524,288 floats; DiT patch size 1 so 4096 image tokens; text 512 tokens; joint sequence 4608. Initial latent buffer is zeros of that shape: `SD:src/pipeline/diffusion_engine.cpp:2932-2943`. | as cited |
| Sigma schedule | `Flux2Scheduler`: `mu = compute_empirical_mu(image_seq_len, steps)` with `a1=8.73809524e-5, b1=1.89833333, a2=1.6927e-4, b2=0.45666666`; for `seq_len <= 4300`: `m200 = a2*L+b2`, `m10 = a1*L+b1`, `a = (m200-m10)/190`, `b = m200-200a`, `mu = a*steps+b`. `sigma_i = e^mu / (e^mu + (1/t - 1))` with `t = 1 - i/steps`; `t>=1 -> 1`, `t<=0 -> 0`, `sigma[n] = 0`. For L=4096, steps=4: mu = 2.29118 and sigmas = `[1, 0.967384, 0.908144, 0.7672, 0]` (DERIVED by evaluating the formula; S0 checks it against the `-v` log). The model's own `scheduler_config.json` says `use_dynamic_shifting`, `shift 3.0`, `base_shift 0.5`, `max_shift 1.15`; sd.cpp does NOT read it for klein (it uses the empirical mu above), so the config-driven route carries the `mu` coefficients as config values and sd.cpp's formula is the oracle (principle 14). | `SD:src/runtime/denoiser.hpp:725-727` (`flux_time_shift`), `:853-903` (`Flux2Scheduler`), `:1185-1187` |
| Denoiser | `FluxFlowDenoiser`: the DiT is fed `t = sigma` (`sigma_to_t` is identity); the timestep embedding multiplies by `time_factor = 1000` and builds a 256-wide sinusoid (`max_period 10000`). | `SD:src/runtime/denoiser.hpp:1442-1453`; `SD:src/model/diffusion/flux.hpp:1060`; `SD:src/core/ggml_extend.cpp:893-901` |
| Euler step | `d = (x - denoised) / sigma; x += d * (sigma_next - sigma)`. The model output is the denoised x0 from the flow denoiser, so `x - denoised = sigma * v` and the update is `x += (sigma_next - sigma) * v`. No noise injection. | `SD:src/runtime/denoiser.hpp:1840-1856` |
| DiT config | in/out channels 128, patch 1, `mlp_ratio 3`, `theta 2000`, `axes_dim {32,32,32,32}`, no qkv/linear bias (`disable_bias`), `share_modulation`, `use_mlp_silu_act`, no guidance embed, `vec_in_dim 0`; `ref_index_scale 10` only matters with reference images (out of scope). | `SD:src/model/diffusion/flux.hpp:90-101` |
| Pre-blocks | `img = img_in(x)` (Linear 128 to 3072); `vec = time_in(sinusoid_256(t*1000))` where `time_in` is `MLPEmbedder`: Linear, SiLU, Linear; `txt = txt_in(context)` (Linear 7680 to 3072); three shared modulation linears computed once from `silu(vec)`: `double_stream_modulation_img`, `double_stream_modulation_txt` (each 3072 to 6x3072) and `single_stream_modulation` (3072 to 3x3072). | `SD:src/model/diffusion/flux.hpp:1018-1094`, `:197-212` (MLPEmbedder), `:384-410` (Modulation) |
| Double-stream block (x5) | For each stream `s` in {img, txt}: `h = LayerNorm(s)` (no affine, eps 1e-6); `h = h + h*scale1 + shift1` (modulate is `x*(1+scale)+shift`); `qkv = Linear(h)` (3072 to 9216, no bias) split into q,k,v of `[24 heads, 128]`; q and k RMS-normed per head with a learned scale (eps 1e-6). Concatenate txt then img along tokens (txt FIRST); RoPE on q and k (interleaved adjacent pairs); full bidirectional attention, scale `1/sqrt(128)`, no mask; split back. `s = s + gate1 * proj(attn_s)`; `s = s + gate2 * MLP(modulate(LayerNorm(s), shift2, scale2))`, `MLP = Linear(3072 to 2x9216)`, SwiGLU (`silu(first half) * second half`), `Linear(9216 to 3072)`. | `SD:src/model/diffusion/flux.hpp:431-593` (block), `:241-319` (QKNorm, SelfAttention), `:321-345` (MLP), `SD:src/model/common/rope.hpp:1128-1146` (rope on q,k then attend), `SD:src/core/ggml_extend.cpp:136-145,180-197` (silu_act, modulate) |
| Single-stream block (x20) | Input is `concat(txt, img)` along tokens. `x_mod = modulate(LayerNorm(x), shift, scale)`; ONE `Linear(3072 to 3x3072 + 2x9216)` (27,648 outputs, no bias) yields q, k, v and the MLP gate/up in parallel; q,k RMS-normed per head; RoPE; attention over 4608 tokens; MLP half through SwiGLU; `concat(attn_out, mlp_out)` (3072 + 9216 = 12,288) to `Linear(12288 to 3072)`; `x = x + gate * out`. | `SD:src/model/diffusion/flux.hpp:595-701` |
| RoPE ids | 4 axes x 32 dims = 128 head_dim, theta 2000. Text ids `(0,0,0,l)` (`txt_arange_dims = {3}`), image ids `(0,h,w,0)`, text tokens first. Table layout `[L, 64, 2, 2]` holding `[[cos,-sin],[sin,cos]]`, built on host and cached. | `SD:src/model/diffusion/flux.hpp:1542-1565`; `SD:src/model/common/rope.hpp:143,165,463-490,1070-1095` |
| Final layer | `norm_final` LayerNorm (no affine), `modulate(shift, scale)` from `adaLN_modulation.1(silu(vec))` (Linear 3072 to 2x3072, chunked shift then scale), `Linear(3072 to 128)`; image tokens only (text dropped). | `SD:src/model/diffusion/flux.hpp:703-757`, `:1173-1175` |
| Latent normalisation | Before decode the 128-channel latent is de-normalised `x*std + mean` per channel (128 mean/std pairs hard-coded in sd.cpp; the checkpoint carries them as VAE batch-norm statistics), then unpatchified `[128, 64, 64]` to `[32, 128, 128]` (2x2 pixel-shuffle), then `post_quant_conv` (1x1). | `SD:src/model/vae/auto_encoder_kl.hpp:816-836` (stats, `diffusion_to_vae_latents`), `:593-625` (`decode`) |
| VAE decoder ops | `conv_in` 3x3 (32 to 512); mid: ResnetBlock, AttnBlock (GroupNorm, 1x1 q/k/v/proj_out convs, single-head attention over `h*w` tokens), ResnetBlock; then levels `i = 3..0`, each with `num_res_blocks + 1 = 3` ResnetBlocks (channels 512, 512, 256, 128 from `ch_mult {1,2,4,4}`, `ch = 128`), and for `i != 0` an Upsample (nearest x2 then 3x3 conv); then `norm_out` GroupNorm, SiLU, `conv_out` 3x3 (128 to 3). ResnetBlock: `GroupNorm, SiLU, conv3x3, GroupNorm, SiLU, conv3x3` plus residual. GroupNorm is 32 groups, eps 1e-6. | `SD:src/model/vae/auto_encoder_kl.hpp:12-62` (Resnet), `:65-160` (Attn), `:371-497` (Decoder); `SD:src/model/common/block.hpp:46-67` (Upsample: `ggml_upscale ... NEAREST` then conv); `SD:src/model/common/ggml_block.hpp:938-941` (`GroupNorm32`, eps 1e-6) |

Open facts this spec does not guess (each is the first task of the slice that needs it, read from source):
sd.cpp's attention mask for padded text tokens (the Qwen3 decoder is causal; whether pad positions are masked changes the embedding);
the exact ResnetBlock shortcut wiring (`auto_encoder_kl.hpp:12-62`); the checkpoint tensor names (read from the safetensors headers).

## Parity design

### Identical initial noise: implement Philox generically (decision)

Options: (A) port sd.cpp's Philox4x32-10 + Box-Muller as a seeded noise source in proxima, or
(B) patch a scratch copy of sd.cpp to dump the noise and feed it to proxima as an input file.

Chosen: A. Why: Philox is counter-based; element `i` of `randn(n)` is a pure function of `(seed, offset, i)` (`SD:src/core/rng_philox.hpp:69-88`), so it is stateless, parallel by construction and fixed-cost per element. It makes the omega example self-contained (seed in, image out), which B can never be, and "reproduce seed 42" stays a statement about the engine instead of about a file.
Cost: bit-compat of f32 `sin`/`log`/`sqrt` against the libm sd.cpp used: expected within a few ulp, immaterial next to a bf16 bound; AC2 pins the tolerance.
`ScalarOp` has no sine (`proxima-tensor/src/op.rs:77-95`: Identity, Add, Subtract, Multiply, Divide, Maximum, Minimum, Negate, Reciprocal, Exponential, Logarithm, SquareRoot, Tanh, Erf, Greater, Equal, Select), so noise is a host-built `Input` tensor, the same way rope tables are host-built today (`proxima-model-interop/src/generate/residency_caches.rs`, `yarn_rope_table_tests`). It is a cold once-per-image 524,288-element path, not a hot-path op.
The generator is selected by a config value (`rng = "philox4x32_10_box_muller"`); `std_default` and `cpu` are named but unimplemented and refused with an error that names them.
B remains a debugging fallback: S0's patched sd.cpp dumps its noise so AC2 verifies A against it once.

### Stage dumps need a patched incumbent (out of tree)

sd-cli has no flag that dumps text embeddings, per-step latents or the pre-VAE latent (a grep of `SD:examples/common/common.cpp` for dump options found none beyond JSON writers).
S0 therefore builds a scratch copy of sd.cpp OUTSIDE the proxima repo (`~/.cache/kvgate/proxnum588/sdcpp-dump/`) that writes raw f32 at: the conditioner output (after `encode_prompt`), the initial noise (`image.cpp:971`), `x` after each `sample_euler` step (`denoiser.hpp:1840-1856`), and the VAE input and output.
Reference timings (AC8) use the UNPATCHED binary.

### Metric and thresholds (S0-measured)

Gate metric for a pair `(a, b)`, `b` the sd.cpp dump: **`rms_rel = rms(a - b) / rms(b)`**, with cosine, maxabs and maxabs_rel reported alongside and never gated. Cosine catches a direction collapse an RMS number can hide.

Why not maxabs_rel: the S0 floor gives `maxabs_rel` 1.009 while `cos` is 0.99999376 and `rms_rel` is 0.0035. Qwen3 carries a handful of massive-activation channels (reference range [-4595, 16360] against RMS 14.9), so dividing the largest element error by the whole-tensor RMS measures outlier channels, not model agreement. `rms_rel` is exactly the quantity the precision model predicts.

Precision model: bf16 has an 8-bit significand, one rounding relative error `u = 2^-8 = 3.906e-3`.
`n` independent roundings random-walk to `u*sqrt(n)`; two independent implementations contribute `u*sqrt(2n)`. Under `rms_rel` that RMS already carries the tail, so no extra factor is applied (the earlier 4x was the max-vs-RMS factor of a max-statistic, which the gate no longer uses). Accumulation is f32 on both sides -- S0 confirms sd.cpp uses bf16 weights with f32 activations (`kernel_mul_mm_bf16_f32`, `ref.log:135-140`).
`n` is counted from the recipes above: ~10 rounded ops per Qwen3 layer (norm, qkv, attention out, o-proj, residual, norm, gate, up, act*gate, down, residual), ~25 per double-stream block, ~10 per single-stream block, ~8 per VAE ResnetBlock.

| Stage | n per side | ceiling `rms_rel` = u*sqrt(2n) |
|---|---|---|
| text tap 9 | 90 | 0.052 |
| text tap 18 | 180 | 0.074 |
| text tap 27 | 270 | 0.091 |
| DiT velocity, one step | 5x25 + 20x10 + 6 = 331 | 0.101 |
| latent after Euler step 1/2/3/4 | -- | 0.003 / 0.007 / 0.022 / 0.137 |
| VAE decode (f32 weights both sides) | -- | 0.065 (loose bf16-derived bound) |

Latent after Euler step k accumulates `Delta_sigma_j * e_v` (first order; error feedback through the model ignored):
`Delta_sigma = [0.032616, 0.059240, 0.140944, 0.767200]` (from the sigmas above), cumulative root-sum-square `[0.0326, 0.0676, 0.1563, 0.7830]`, times `u*sqrt(2*331) = 0.1005`.

PSNR ceilings, re-derived from the MEASURED image RMS 0.732612 (not an assumed 100/255):

- VAE decode alone (given the reference latent): relative RMS 0.0654, absolute 0.0479, `PSNR = 20 log10(1/0.0479) = 26.4 dB`: gate `PSNR >= 26.0 dB`.
- End to end adds the step-4 latent error (0.078 relative): `sqrt(0.0654^2 + 0.078^2) = 0.102`, absolute 0.0746, `PSNR = 22.6 dB`: gate `PSNR >= 22.0 dB`.

These are a-priori CEILINGS from precision, not targets. S0 also measured the incumbent's own
cross-backend spread (sd.cpp Metal vs sd.cpp CPU, same inputs, `$K/dump-cpu-floor/`):

| stage | measured `rms_rel` | cosine / PSNR |
|---|---|---|
| conditioner (text) | 0.00353 | cos 0.99999376 |
| DiT velocity, step 1 | 0.0472 | cos 0.99889002 |
| latent after step 1 | 0.00176 | cos 0.99999845 |
| latent after step 2 | 0.00548 | cos 0.99998497 |
| latent after step 3 | 0.0223 | cos 0.99975083 |
| latent after step 4 | 0.1371 | cos 0.99059928 |
| VAE decode | 0.0322 | 32.54 dB, cos 0.99948137 |

Effective gate = `max(ceiling, measured floor)`: a gate tighter than the incumbent's own
backend-to-backend spread is not an achievable parity claim, and one looser than the precision
ceiling hides a real defect. This replaces the earlier `min(ceiling, 2 x floor)`, whose direction
was wrong once the measured floor exceeded the ceiling (step 4), and which the conditioner floor
made 15x too strict. Final gates: text 0.052, velocity 0.101, latents 0.003 / 0.007 / 0.022 /
0.137, VAE PSNR >= 26.0 dB, end-to-end PSNR >= 22.0 dB. A result outside its gate is a defect in
proxima (principle 14), never a reason to raise a gate.

## Composition rules (binding, owner 2026-11-21)

- No dtype is named by hand inside a function. `f16`/`f32`/`bf16`/`FLOAT32`/`BFloat16`/`Float16` never appear as a hardcoded choice in model, lowering, bind or scheduler code; the dtype comes from `proxima-numeric` (`DType`, `NumericFormat`, `NumericPolicy`) as a parameter or config value. A concrete dtype may appear only in the outermost harness that reads or writes a real file.
- No GPU leaks upward. `omega` is named only at the execution boundary; model and lowering code emit `proxima_tensor::op::Op` graphs and know nothing about Metal or any other driver.
- Sans-IO / FSM composition: the graph is built as data (`Vec<Op>` plus named inputs), the existing lowering/fusion compiles it, then one plan executes it. Model code never drives per-op GPU calls, never branches on a backend, and never fuses by hand in Rust control flow.
- If the pipe/op algebra cannot express a piece, that is a finding to report, not a reason to add a bespoke type or a model branch.

## Architecture: reuse first

No new top-level abstraction. The pipeline is the existing Pipe composition (noise source, text pipe, scheduler loop, DiT pipe, VAE pipe); each component maps to existing pieces plus generic lowerings selected by config.

| Component | Reuse (cited) | Genuinely missing (generic, config-driven) |
|---|---|---|
| IR | 5 ops: Input, Elementwise, Reduce, Iota, Constant (`proxima-tensor/src/op.rs:192-`); `ScalarOp` at `op.rs:77-95` | nothing: every item below lowers to Elementwise/Reduce with index maps |
| Weights | safetensors and GGUF binding in interop (`proxima-model-interop/src/bind.rs`, `convert.rs`); BF16 dtype (`src/dtype.rs:22`); HF `config.json` to hparams (`src/hf_config.rs:1-`); arch-as-TOML (`src/arch_config.rs:278`, `src/arch/{llama,granite}.toml`) | an arch TOML naming the DiT tensors and a VAE tensor map; tensor names come from the safetensors headers (S4/S6 first task) |
| Text encoder | Qwen3 dense decoder: q/k per-head RMSNorm already bound (`bind.rs:632-645`), HF checkpoint bind (`src/lowering.rs:532`, `bind_hf_checkpoint`), a real Qwen3 GGUF drives `examples/op_profile_step.rs:9,125`; rope layout is a config key (`src/arch/llama.toml`, `rope_layout = "adjacent"`) | (1) a hidden-state TAP: no existing lowering returns an intermediate layer (a grep for `output_hidden`, `hidden_states` in `proxima-model-interop/src` and `omega/src` finds none outside tests); config `tap_layers = [9, 18, 27]`, output concatenated on the hidden axis, no final norm; (2) tokenizer + template: whether interop tokenizes Qwen2-BPE is settled in S4's first task, the template string is config; (3) prefill-only, batch 1, no KV write |
| DiT linears, SwiGLU, SiLU | matmul lowering; `silu` builder `proxima-autograd/src/activation.rs:156`; SwiGLU = `Multiply(Silu(a), b)` over two slices of one linear (slice = non-zero `offset` index map, `proxima-tensor/src/map.rs` table) | none |
| Per-head q/k RMSNorm | Qwen3 q/k norm lowering above | none (head_dim 128) |
| LayerNorm (no affine) + adaLN modulate | a layer-norm cluster exists in the CPU epilogue (`proxima-tensor/src/cpu/epilogue.rs:738`); Reduce + Elementwise | a generic no-affine LayerNorm lowering and `modulate = x*(1+scale)+shift` with `[N,1,C]` broadcast (broadcast = projecting fewer axes, `map.rs` table). Config `norm = "layer_norm_no_affine"`, eps 1e-6 |
| Multi-axis RoPE | host-built rope tables (`generate/residency_caches.rs`), adjacent-pair layout key | tables from 4 axis ids x 32 dims, theta 2000 (config `rope_axes = [32,32,32,32]`, `rope_theta = 2000`) and a config-selected id generator (text `(0,0,0,l)`, image `(0,h,w,0)`); today's rope is single-axis |
| Attention | omega attention lowering and goldens (`omega/src/msl/attn_golden_tests.rs:24,68`), online softmax | NON-CAUSAL, mask-free, no-KV-cache joint attention over 4608 tokens x 24 heads (config `causal = false`); the existing path is the causal form |
| Timestep embedding | Iota, Exponential, Multiply | sin/cos are not `ScalarOp`s: the 256-wide sinusoid for a scalar `t` is a host-built `Input` per step (4 per image), same pattern as rope tables |
| Sigma schedule + Euler | scalar code for 5 sigmas; `x += (s1-s0)*v` is one Elementwise over 524,288 elements | a small scheduler step (config: `mu` coefficients, `steps`, `image_seq_len`); no new IR |
| Noise | host-built `Input` (decision above) | Philox4x32-10 + Box-Muller generator, config `rng` |
| VAE conv2d | convolution as an index map with two terms per axis (`proxima-tensor/src/map.rs:14` table row "convolution"); `proxima-autograd/src/conv.rs:121` `conv2d`; `proxima-onnx/src/lower.rs:5590` `conv2d_core` | wire conv2d into the omega/interop lowering (today it lives in the autograd/onnx path); the 128-channel 3x3 conv at 1024x1024 is the tiling/perf risk |
| GroupNorm | Reduce over `(channels/32, h, w)` + Elementwise | no GroupNorm in any proxima lowering (a `git grep -i "groupnorm\|group_norm" -- '*.rs'` finds only `tools/llama-arm/src/bindings.rs`): add a generic one, config `groups = 32`, eps 1e-6 |
| Nearest x2 upsample | an index map reading a halved source axis (stride/dilation row, `map.rs`) | the explicit nearest-x2 lowering (ONNX `Resize` nearest equivalent) |
| VAE attention | same non-causal attention as the DiT with 1 head; 1x1 convs are matmuls | head_dim 512 and 16,384 tokens (`128*128`) at the 1024 decode: check attention tile limits (config); memory risk below |
| Latent (de)normalise + unpatchify | Elementwise with per-channel broadcast; reshape/permute as index maps (`map.rs` transpose row) | mean/std come from the VAE checkpoint's batch-norm statistics (config/weights), not hard-coded as in sd.cpp |

Designs abandoned under principle 1 and R9: a bespoke `Flux2Pipeline` type; a `Box<dyn DenoiserModel>` for the scheduler; an `if arch == "flux2"` branch in omega. Each missing piece is instead a generic lowering selected by a config key (`norm`, `rope_axes`, `causal`, `groups`, `tap_layers`, `rng`).
Tier: new lowering code is no_std + alloc like the tensor crate; the example and bench harness are std and additive.

## Acceptance criteria

Commands run from the worktree root. `$K = ~/.cache/kvgate/proxnum588`, `$M = ~/models/image_gen/flux2-klein-4b`.
S0 dumps live in `$K/ref/` as raw little-endian f32: `noise.f32` (524,288 values), `text.f32` (512 x 7680 = 3,932,160), `step1.f32` to `step4.f32` (524,288 each), `vae_out.f32` (3 x 1024 x 1024), plus `$K/ref.png`.
Test target `flux2_klein` in `proxima-model-interop`, gated on `PROXIMA_FLUX2_KLEIN_DIR=$M` and `PROXIMA_FLUX2_KLEIN_REF=$K/ref`; a missing fixture is a failure, not a skip, so `0 passed` is red.

| AC | Discharges | Command | Expected |
|---|---|---|---|
| AC1 reference run + dumps | R1 | the patched `sd-cli` run, then the existence check | the 8 raw dumps `noise.f32`, `text.f32`, `step1-4.f32`, `velocity1.f32`, `vae_out.f32` all present in `$K/ref` together with `ref.log`; `ref.png` is 1024 x 1024 (`sips -g pixelWidth -g pixelHeight`); `grep -c "mu=2.29" $K/ref/ref.log` = 1; `grep -c "MTL0" $K/ref/ref.log` >= 1 |
| AC2 noise | R2 | `cargo val -p proxima-model-interop --test flux2_klein philox::` | `3 passed`: (a) first 16 values vs `noise.f32` max abs diff <= 1e-5, (b) all 524,288 values max abs diff <= 1e-5, (c) over the 524,288 values mean in [-0.01, 0.01] and std in [0.99, 1.01] |
| AC3 text embedding | R3 | `cargo val -p proxima-model-interop --test flux2_klein text_embedding::` | `4 passed`: tap 9 / 18 / 27 `rms_rel` <= 0.052 each (S0 effective gate) with cosine >= 0.999 and maxabs_rel printed as a diagnostic, and the concatenation has shape 512 x 7680 (3,932,160 values) |
| AC4 sigmas | R4 | `cargo val -p proxima-model-interop --test flux2_klein scheduler::` | `3 passed`: sigma vector of length 5 equals `[1, 0.967384, 0.908144, 0.7672, 0]` within 1e-5; our `mu` equals the log's within 1e-4; 4 Euler steps on a constant velocity `v` equal `x0 - v` within 1e-6 |
| AC5 DiT | R5 | `cargo val -p proxima-model-interop --test flux2_klein dit::` | `6 passed`: single-stream block, double-stream block, full velocity at step 1 (`rms_rel` <= 0.101), latents after steps 1,2,3,4 `rms_rel` <= 0.003 / 0.007 / 0.022 / 0.137, all with cosine >= 0.999; 4 latent dumps compared, 4 verdict lines printed |
| AC6 VAE | R6 | `cargo val -p proxima-model-interop --test flux2_klein vae::` | `4 passed`: GroupNorm vs a hand-computed 32-group case, nearest-x2 unit, conv3x3 vs a direct loop, and the full decode of `step4.f32` vs `vae_out.f32` with PSNR >= 26.0 dB at shape 3 x 1024 x 1024 |
| AC7 end to end | R7 | `cargo run --release -p proxima-model-interop --example flux2_klein -- --seed 42 --out $K/omega.png`, then `cargo run --release -p proxima-model-interop --example image_psnr -- $K/ref.png $K/omega.png` (Rust, no Python) | PNG is 1024 x 1024; printed PSNR >= 22.0 dB; the example prints 3 stage timers (text, DiT steps, VAE) |
| AC8 speed | R8 | `scripts/flux2_klein_ab.sh` (3 alternating pairs, quiet window, release builds) | prints `runs=3 per side`; `omega_median_ms / sdcpp_median_ms <= 1.05`; **omega wall <= 84 s** (owner bar, the measured incumbent); per-stage rows (text encode, DiT step 1 to 4, VAE decode) for both sides with the ratio; peak RSS (`/usr/bin/time -l`, "maximum resident set size") for both sides (report-only, no bar); CoV <= 5% per side |
| AC9 genericity | R9 | `git grep -n -i "flux\|klein" -- omega/src proxima-model-interop/src ':!*.toml' ':!*.md'` | 0 lines; `ls proxima-model-interop/src/arch` lists 3 files (llama, granite, flux2 TOML) |
| AC10 generic lowerings | R10 | `cargo val -p proxima-model-interop generic_lowering::` | `6 passed`: one test per new lowering (layer_norm_no_affine, group_norm, nearest_upsample_x2, multi_axis_rope, non_causal_attention, tap_layers), each on a non-FLUX shape |

## Speed methodology (AC8)

Same box (M1 Max), quiet window (no other cargo or agents; free memory checked first), release builds, 3 alternating runs per side (sd, omega, sd, omega, sd, omega), median reported with CoV; CoV > 5% invalidates the set and it is rerun.
Per-stage times: sd.cpp from its `-v` log lines (conditioning, sampling per step, VAE decode); omega from the example's stage timers. Peak RSS from `/usr/bin/time -l`.
Wall time is whole process launch to PNG written, model load included on both sides (both read the same files); load is also reported separately so a loading gap cannot hide inside the total.
The bar is the owner's: within 5% of sd.cpp. The ratio is a measurement; the mechanism for any gap (which stage, which op) is traced before it is called a result (principles 18, 19).

## Out of scope

Reference-image editing (`ref_index_scale`), the 9B and base (cfg > 1, 20-step) variants, LoRA, img2img, batch > 1, the VAE encoder, VAE tiling, `--offload-to-cpu`, non-Metal backends, Python anywhere (owner rule).

## Risks

- Memory: DiT 7.75 GB bf16, Qwen3-4B about 8 GB, VAE 168 MB. The text encoder is dropped after conditioning, so stage ordering is part of S6. Machine freezes on builds are on record: one cargo job at a time, `-j 2`.
- VAE at 1024^2: the 128-channel 3x3 conv at 1024 x 1024 and attention over 16,384 tokens of width 512 may force row tiling, a perf and memory risk for AC8.
- The text padding mask (open fact above) decides whether AC3 can pass at all.
- sd.cpp's sigma path ignores `scheduler_config.json`; following the checkpoint config instead gives different sigmas (the config's `max_shift` 1.15 vs sd.cpp's mu 2.29). The incumbent wins: config carries sd.cpp's coefficients.
- Metal at runtime in sd.cpp is unconfirmed (card); S0 reads it from the log before any timing is called a comparison.
