# Omega 05: speech, whisper

**Prerequisites:** [page 02](./02-programs-as-data.md) (a program is a graph over named buffers) and [page 03](./03-train.md) (a lowering is trainable if autograd differentiates it).

**You will:** see exactly which pieces of a whisper pipeline exist today, run two of them, and read what is missing before a `.wav` becomes text. This page is short on purpose: the honest state is "the lowerings exist and are differentiable; the model path does not".

**New concepts (in order):** a lowering is a graph-building function · `stft_log_mel` and `StftLogMelConfig` · the log-mel clamp and rescale · `non_causal_attention` for an encoder-decoder cross-attention · `conv1d` as a trainable front end · why "no per-model profile" is a requirement, not a preference · what omega does not do yet.

## 1. What exists

Every piece below is a graph-building function in `proxima-tensor` or `proxima-autograd`, carries a non-whisper unit test, and is a generic operation rather than a whisper-shaped one.

| piece | signature | unit test |
|---|---|---|
| log-mel front end | `stft_log_mel`, `proxima-tensor/src/spec/generic_lowering.rs:525` | `stft_log_mel_matches_a_direct_dft_reference_on_a_64_sample_signal` (`:867`) |
| front-end config | `StftLogMelConfig`, `proxima-tensor/src/spec/generic_lowering.rs:397` | the three `stft_log_mel_*` tests (`:867`, `:882`, `:907`) |
| cross-attention | `non_causal_attention`, `proxima-tensor/src/spec/generic_lowering.rs:166` | `non_causal_attention_lets_every_query_see_every_key` (`proxima-model-interop/src/generic_lowering.rs:82`) |
| encoder 1-D convolution | `conv1d`, `proxima-autograd/src/conv.rs:223` | `conv1d_forward_matches_brute_force_across_stride_padding_and_dilation` (`conv.rs:1904`) |
| affine layer norm | `layer_norm`, `proxima-tensor/src/spec/generic_lowering.rs:99` | `affine_layer_norm_matches_naive_reference_on_a_non_model_shape` (`:695`) |
| multi-axis rope tables | `multi_axis_rope_tables`, `proxima-tensor/src/spec/generic_lowering.rs:295` | `fused_rope_pair_gradient_matches_finite_differences_with_multi_axis_tables` (`proxima-autograd/src/generic_lowering_grad.rs:250`) |
| layer taps | `tap_layers`, `proxima-tensor/src/spec/generic_lowering.rs:337` | `tap_layers_gradient_matches_finite_differences_and_routes_only_to_tapped_layers` (`proxima-autograd/src/generic_lowering_grad.rs:336`) |

The front end is the interesting one, because it is the part of whisper that is not a transformer. `StftLogMelConfig` (`proxima-tensor/src/spec/generic_lowering.rs:397`) is: `window` (`:399`), `n_fft` default 400 (`:401`), `hop` default 160 (`:403`), `floor` default `1e-10` (`:405`), `dynamic_range` default 8 decades (`:407`), and the final affine `scale`/`offset` defaulting to `(x + 4) / 4` (`:409`, `:411`). Those five numbers are the whole "family default" the spec allows: everything else is read from the checkpoint's tensor names and `config.json`.

`stft_log_mel` frames the signal through the two-term window map `hop*frame + tap`, runs a DFT against precomputed cos/sin tables, clamps and rescales, and returns one node. The tables come from `stft_tables` (`proxima-tensor/src/spec/generic_lowering.rs:487`), and the doc states the constraint plainly: padding the signal is the caller's job.

Run the reference test — it compares the lowering against a direct DFT over a 64-sample signal, which is the smallest input where a framing or bin error shows up:

```sh
cargo val -p proxima-tensor -E 'test(stft_log_mel_matches_a_direct_dft_reference_on_a_64_sample_signal)'
```

```text
PASS proxima-tensor spec::generic_lowering::stft_log_mel_tests::stft_log_mel_matches_a_direct_dft_reference_on_a_64_sample_signal
SUMMARY  passed=2 failed=0 ignored=0 not_run=0  wall=3.919s discovered=1141 run=2 filtered=1139 binaries=8
```

## 2. Trainable, not just runnable

A lowering that cannot be differentiated is useless here: whisper is the first family template that must emit both an inference graph and a training graph from the same ops (R9, `proxima-model-interop/specs/whisper/SPEC.md:33`). The proof is a finite-difference test per lowering, in `proxima-autograd/src/generic_lowering_grad.rs`:

```sh
cargo val -p proxima-autograd \
  -E 'test(non_causal_attention_gradient_matches_finite_differences_for_q_k_and_v)'
```

```text
PASS proxima-autograd generic_lowering_grad::non_causal_attention_gradient_matches_finite_differences_for_q_k_and_v
SUMMARY  passed=3 failed=0 ignored=0 not_run=0  wall=2.406s discovered=203 run=3 filtered=200 binaries=24
```

The test perturbs each of `q`, `k` and `v` by a small delta, recomputes the loss, and compares the central difference against the graph's gradient — the same method you would use to check a hand-written backward pass, applied here to a composed lowering.

## 3. What omega does not do yet

- **No whisper model path.** `proxima-model-interop/specs/whisper/SPEC.md:3` says it plainly: the spec is a draft, written from whisper.cpp's source, with no code, no builds and no model runs. Nothing in `proxima-model-interop/src` mentions whisper, and by design: R2 (`proxima-model-interop/specs/whisper/SPEC.md:26`) forbids a per-model profile, so the work is an *inference template* that fills its slots from `model.safetensors` names plus `config.json`, not a `whisper.toml`.
- **The checkpoint is on disk but unused.** `~/models/asr/whisper-tiny` is an HF safetensors checkpoint with `config.json`, `tokenizer.json`, `preprocessor_config.json` and `model.safetensors`; no code path opens it today.
- **No incumbent build to compare against.** `~/repos/others/whisper.cpp` is source only; `build/bin` is empty, and there is no `ggml-*.bin` on this machine. Converting `whisper-tiny` to ggml needs whisper.cpp's Python `convert-h5-to-ggml.py`, which is out of bounds for this workspace, so the R13 wall-time comparison (`proxima-model-interop/specs/whisper/SPEC.md:37`) has no measurable row.
- **No decoder loop.** Even with every lowering present, nothing assembles an encoder pass, a cross-KV cache and a token loop into a transcription; that is the driver the spec's slice list still owes.

Next: [06: diffusion](./06-diffusion.md) does the same for image generation: the adaLN and attention pieces exist and are differentiable; the scheduler and the VAE decode do not.
