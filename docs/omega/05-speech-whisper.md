# Omega 05: speech, whisper

**Prerequisites:** [page 02](./02-programs-as-data.md) (a program is a graph over named buffers) and [page 03](./03-train.md) (a lowering is trainable if autograd can differentiate it).

**You will:** see which pieces of a whisper pipeline exist as ordinary graph operations, run two of them, and read the exact point where a `.wav` would fall off the end.

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

## 3. What omega does not do yet

- No whisper model path. `proxima-model-interop/specs/whisper/SPEC.md:3` is a draft: written from whisper.cpp's source, with no code, no build and no model run. Nothing in `proxima-model-interop/src` mentions whisper.
- A per-model profile is ruled out. The spec's second requirement (`proxima-model-interop/specs/whisper/SPEC.md:26`) forbids a per-model profile — no `whisper.toml`, no "whisper" branch in omega or interop. The intended path is one encoder-decoder template that fills its slots from `model.safetensors` names plus `config.json`. A template is more work than a profile, which is why it is not done.
- The checkpoint is on disk and unread. `~/models/asr/whisper-tiny` is a complete HF checkpoint (`config.json`, `tokenizer.json`, `preprocessor_config.json`, `model.safetensors`); no code path opens it.
- There is no decoder loop. Even with every lowering present, nothing assembles the encoder pass, the cross-KV cache and the token loop into a transcription. The spec's slice list still owes that driver.
- Nothing to compare against yet. `~/repos/others/whisper.cpp` is source only (`build` does not exist) and there is no `ggml-*.bin` here. Converting the tiny checkpoint to ggml needs whisper.cpp's Python converter, which this workspace does not run, so the spec's wall-time requirement (`proxima-model-interop/specs/whisper/SPEC.md:37`) has no measurable row.

Next: [06: diffusion](./06-diffusion.md) does the same for image generation: the adaLN and attention pieces exist and are differentiable; the scheduler and the VAE decode do not.
