# Omega 05: speech, whisper

**Prerequisites:** [page 02](./02-programs-as-data.md) (a program is a graph over named buffers) and [page 03](./03-train.md) (a lowering is trainable if autograd can differentiate it).

**You will:** see which pieces of a whisper pipeline exist as ordinary graph operations, run two of them, and see where the end-to-end transcription path sits and what it costs against whisper.cpp.

**New concepts (in order):** a lowering is a graph-building function · the log-mel front end · cross-attention · conv1d · why "no per-model profile" is a requirement · what omega does not do yet.

## 1. What exists

Two files in this crate pair are both named `generic_lowering.rs` — one in `proxima-tensor`, one in `proxima-model-interop`. Rows below name the crate once and then the bare filename. Whisper needs four kinds of graph that a transformer encoder does not: a log-mel front end, a convolution over audio frames, cross-attention whose keys and values are computed once per utterance, and multi-axis rotary tables. All four exist here as generic functions — nothing in them is whisper-specific — together with the normalisation and layer-tap helpers a stack of blocks also needs:

| piece | function | what it does |
|---|---|---|
| log-mel front end | `stft_log_mel`, `proxima-tensor/src/spec/generic_lowering.rs:525` | frames the signal, runs a DFT against precomputed tables, clamps and rescales |
| its configuration | `StftLogMelConfig`, `generic_lowering.rs:397` | window, `n_fft` (400), `hop` (160), floor, dynamic range, final scale/offset |
| cross-attention | `non_causal_attention`, `generic_lowering.rs:166` | attention with no causal mask, for a decoder reading an encoder |
| encoder convolution | `conv1d`, `proxima-autograd/src/conv.rs:223` | 1-D convolution with stride, padding and dilation, differentiable |
| layer norm | `layer_norm`, `generic_lowering.rs:99` | affine normalisation |
| rotary tables | `multi_axis_rope_tables`, `generic_lowering.rs:295` | per-axis rotary position tables |
| layer taps | `tap_layers`, `generic_lowering.rs:337` | stack chosen intermediate layers as one tensor |

Each row's test lives beside its function in the same file, on a shape that is not a whisper model — for the front end, `stft_log_mel_matches_a_direct_dft_reference_on_a_64_sample_signal` (`generic_lowering.rs:867`). The cross-attention test is in the other crate (`proxima-model-interop/src/generic_lowering.rs:82`), and the convolution's is `conv.rs:1904`.

The front end is the part that is not a transformer at all, so here it is. `StftLogMelConfig` (`generic_lowering.rs:397`) carries exactly five numbers: `n_fft` 400 (`:401`), `hop` 160 (`:403`), a floor of `1e-10` (`:405`), a dynamic range of 8 decades (`:407`), and the final affine rescale `(x + 4) / 4` (`:409`, `:411`). Those are the whole "family default" the spec allows; everything else comes from the checkpoint's tensor names and `config.json`. The tables are built by `stft_tables` (`:487`), and padding the signal is the caller's job.

Run the front end's reference test. It compares the lowering against a direct DFT on 64 samples, the smallest input where a framing or bin error shows up:

```sh
cargo val -p proxima-tensor -E 'test(stft_log_mel_matches_a_direct_dft_reference_on_a_64_sample_signal)'
```

```text
PASS proxima-tensor spec::generic_lowering::stft_log_mel_tests::stft_log_mel_matches_a_direct_dft_reference_on_a_64_sample_signal
SUMMARY  passed=2 failed=0 ignored=0 not_run=0  wall=3.919s discovered=1141 run=2 filtered=1139 binaries=8
```

## 2. Trainable, not just runnable

A lowering that cannot be differentiated is useless for whisper, because the model has to produce both an inference graph and a training graph from the same operations. That requirement is `proxima-model-interop/specs/whisper/SPEC.md:33`: the template emits an inference graph and a training graph from the same lowerings. The proof is a finite-difference test per lowering in `proxima-autograd/src/generic_lowering_grad.rs`.

The attention test perturbs each of `q`, `k` and `v`, recomputes the loss, and compares the central difference with the graph's gradient — the same method you would use on a hand-written backward pass:

```sh
cargo val -p proxima-autograd \
  -E 'test(non_causal_attention_gradient_matches_finite_differences_for_q_k_and_v)'
```

```text
PASS proxima-autograd generic_lowering_grad::non_causal_attention_gradient_matches_finite_differences_for_q_k_and_v
SUMMARY  passed=3 failed=0 ignored=0 not_run=0  wall=2.406s discovered=203 run=3 filtered=200 binaries=24
```

## 3. What exists end to end, and what omega does not do yet

Whisper transcribes today. `Model::load` and `Model::transcribe` (`proxima-model-interop/src/families/encoder_decoder_host.rs:687`) run the front end, the encoder, the cross-KV cache and the token loop on a checkpoint directory, on CPU or, when the serving settings passed to `with_settings` set `gpu_layers` to `GPU_LAYERS_ALL`, on Metal. The test `transcript::ids_and_text_match_the_reference` (`proxima-model-interop/tests/whisper.rs:1216`) transcribes a wav and asserts the token ids, the end-of-text stop and the text equal the reference. `proxima-model-interop/specs/whisper/SPEC.md:3` lists S8 (two modes from one graph set) and S11-S13 (training overfit, genericity audit, timed comparison) as open.

- A per-model profile is ruled out. The spec's second requirement (`proxima-model-interop/specs/whisper/SPEC.md:26`) forbids a per-model profile — no `whisper.toml`, no "whisper" branch in omega or interop. The intended path is one encoder-decoder template that fills its slots from `model.safetensors` names plus `config.json`. A template is more work than a profile, which is why it is not done.
- It is slower than whisper.cpp on every measured row. Source: slot-0 `kv-epic/SCOREBOARD.md`, section "whisper (speech) vs whisper.cpp v1.9.5", read 2026-11-24, jfk.wav (11 s) on an M1 Max, token ids exact on both models.

  | model | device | omega | whisper.cpp v1.9.5 | ratio |
  |---|---|---|---|---|
  | tiny | Metal | 1.24 s (n=3 median, 2026-11-24, same window) | 0.34 s (n=3 median) | about 3.6x |
  | tiny | CPU | 20.92 s | 0.506 s | 41.3x |
  | large-v3-turbo | Metal | 62.14 s (not re-measured) | 1.318 s | 47.2x |
  | large-v3-turbo | CPU | 150.92 s | 1.318 s | 114.5x |

  On tiny/Metal the gap is encode (720-737 ms against 30-37 ms, about 22x) and mel (117-123 ms against 4.5-4.7 ms, about 26x); decode is 8.7-8.8 ms/token, down from 63.8 before the decode reuse landed. The owner's bar is about 300 ms for tiny; the 5% bar is 0.357 s. [Page 07 section 5](./07-comparisons.md#5-whispercpp) has the commits, the load under each run and the caveats.

Next: [06: diffusion](./06-diffusion.md) does the same for image generation: the adaLN and attention pieces exist and are differentiable; the scheduler and the VAE decode do not.
