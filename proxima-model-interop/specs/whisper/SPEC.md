# SPEC: Whisper speech-to-text, inferred and trainable, through proxima / omega (PROXNUM-610, PROXNUM-611)

Status: S0, S1-S7, S9 and S10 landed; S8, S11-S13 open. Whisper.cpp v1.9.5
(`d1be6fde`) is built here with Metal; tiny and large-v3-turbo both transcribe
jfk.wav to the reference ids (24 and 27). The tiny F32 oracle is under
`~/.cache/kvgate/proxnum610/ref-f32`; the turbo reference is `ref-turbo`. Every
whisper.cpp fact is read from that tag (`W:`) and every proxima fact from the
landing worktree. Numbers tagged DERIVED were arithmetic; the measurements are
below. The turbo HF checkpoint is F16 and the host decodes it through
`proxima_numeric`; its `preprocessor_config.json` omits `mel_filters`, so the
128 x 201 bank is extracted from the ggml file.

## Problem

`openai/whisper-tiny` (HF safetensors on disk at `~/models/asr/whisper-tiny`, F32) and
`large-v3-turbo` have no path through proxima. Done means, for one code path with no
per-model profile: `samples/jfk.wav` transcribes to the same token ids as whisper.cpp,
the encoder output and first-step logits match the incumbent within an F32-derived bound,
the same graphs TRAIN (a teacher-forced overfit on jfk.wav whose loss falls and whose
gradients match finite differences), and wall time per utterance on large-v3-turbo is
within 5% of whisper.cpp on this M1 Max, with a model that differs from every other omega
model only by config values (PROXNUM-588 R9 carried; PROXNUM-611: inferred, not profiled).

## Requirements

- R1. A fixed reference run exists (section Reference run) and its intermediates are dumpable.
- R2. No per-model profile (PROXNUM-611). There is no `whisper.toml` and no "whisper" in `omega/src`, `proxima-model-interop/src` or `proxima-tensor/src`. Whisper is the first encoder-decoder family template: an inferrer fills its slots from the safetensors tensor names and shapes plus `config.json`. Only what is in neither source (STFT window, n_fft, log-mel clamp and normalisation) is a family default with a cited source.
- R3. One code path accepts tiny (80 mel, 4+4 layers, d_model 384) and large-v3-turbo (128 mel, 32+4 layers) by config alone; every size difference is inferred or read, never branched on.
- R4. An inferred model is accepted only after its golden check against the incumbent. Inference is a hypothesis until measured; a checkpoint that has not passed the golden is refused with a reason, not run.
- R5. The log-mel input matches the reference.
- R6. The encoder output and the per-layer cross-attention K/V match the reference.
- R7. The first-step logits match the reference.
- R8. Greedy decode of jfk.wav reproduces the reference token ids exactly (prompt, suppression and end condition per section Decode semantics).
- R9. **Training (owner, 2026-11-20: "we want to train too").** The family template emits two graphs from the same lowerings: INFERENCE (encoder once, cross-KV cached, decoder step by step) and TRAINING (teacher-forced: the whole target sequence through the decoder under the causal mask, cross-entropy over the vocabulary). No lowering is inference-only; caching is a driver choice, not a different op.
- R10. Every new lowering is trainable: `proxima-autograd` differentiates it and its gradients match central finite differences on a small non-Whisper shape.
- R11. Whisper-tiny overfits jfk.wav with its reference transcript under teacher forcing: loss falls monotonically for K steps of a configured optimizer to below a stated fraction of its start, and the step-1 loss equals the cross-entropy of the validated forward logits.
- R12. The optimizer and its state formats are config values over `proxima-numeric` generics, never f32-typed.
- R13. Wall time per utterance, per-stage times (mel, encode, decode) and peak RSS are measured against whisper.cpp on large-v3-turbo; the bar is within 5%. Peak memory of the training step is measured and recorded the same way.
- R14. New lowerings are generic ops, each usable by a model that is not Whisper and each carrying a non-Whisper unit test: log-mel front end, conv1d (autograd), affine LayerNorm, cross-attention with a once-per-utterance KV, the encoder-decoder decode driver, the `tokenizer.json` byte-level BPE loader with special tokens.

## Reference run

Weights: `~/models/asr/whisper-tiny/model.safetensors` (HF, F32) for proxima; `ggml-tiny.bin` and `ggml-large-v3-turbo.bin` for whisper.cpp (and the HF large-v3-turbo safetensors for proxima, fetched in S0 with its sha recorded in `~/models/manifest.tsv`).

```
~/repos/others/whisper.cpp/build/bin/whisper-cli -m <ggml-tiny.bin|ggml-large-v3-turbo.bin> \
  -f ~/repos/others/whisper.cpp/samples/jfk.wav -l en -nt -bo 1 -bs 1 -tp 0 -tpi 0 -nf \
  -ojf -of ~/.cache/kvgate/proxnum610/ref/<model>
```

Flags, each read in `W:examples/cli/cli.cpp`: `-l` `:197`, `-nt` `:196`, `-bo` `:166`, `-bs` `:167`, `-tp` `:173`, `-tpi` `:174`, `-nf` `:180`, `-oj` `:188`.
Why these: the CLI runs GREEDY unless `beam_size > 1` (`W:examples/cli/cli.cpp:1227`; `beam_size` defaults to the beam-search default of -1, `:45`). Temperature 0 with `-tpi 0` leaves a single temperature (`W:src/whisper.cpp:7042-7047`), and at `t == 0` greedy uses one decoder (`W:src/whisper.cpp:7228-7237`), so no best-of sampling happens. `-nf` disables the temperature fallback. `-nt` makes the prompt carry the no-timestamps token (`W:src/whisper.cpp:7171-7173`), which is what HF's `forced_decoder_ids` `[[1,50259],[2,50359],[3,50363]]` also does (`~/models/asr/whisper-tiny/config.json`), so both sides decode the same prompt. Default language is already `en` (`W:examples/cli/cli.cpp:84`); the flag is explicit anyway.
S0 confirms Metal from the run log and records the transcript text, the token ids and per-stage timings (`whisper_print_timings`). Reference timings use the UNPATCHED binary.

### Oracle dumps need a patched incumbent (out of tree)

whisper-cli has no flag that dumps the mel, the encoder output, the cross-attention K/V, the logits or token ids. S0 builds a scratch copy of whisper.cpp OUTSIDE the proxima repo (`~/.cache/kvgate/proxnum610/whispercpp-dump/`), the same technique as the patched sd.cpp for FLUX (`proxima-model-interop/specs/flux2-klein/SPEC.md`, "Stage dumps need a patched incumbent"), writing raw little-endian f32 (token ids as i32) at:

| Dump | File | Tap point in W | Shape (tiny) |
|---|---|---|---|
| log-mel input | `mel.f32` | after the clamp and normalisation loop, `W:src/whisper.cpp:3296-3311` (the `whisper_mel` filled by `log_mel_spectrogram`, called at `:4011`) | 80 x 3000 (30 s window; `mel.n_len = (padded - 400) / 160`, `:3270-3273`) |
| encoder output | `enc.f32` | the tensor the encoder graph ends on, `whisper_build_graph_encoder` (convs at `W:src/whisper.cpp:2048-2056`, final norm `:2285`) | 1500 x 384 |
| cross-attention K/V per layer | `cross_k_<l>.f32`, `cross_v_<l>.f32` | the cross-KV built from the encoder output once per utterance (decoder graph builder near `W:src/whisper.cpp:2593-2718`; S0 records the exact tensor names and layout) | 4 layers x 1500 x 384 each for K and V |
| first-step logits | `logits0.f32`, `logits0_filtered.f32` | the logits of the decode call that consumes the prompt `[sot, en, transcribe, notimestamps]`, before and after the filter at `W:src/whisper.cpp:6373-6425` | 51,865 |
| final token ids | `tokens.i32` | the greedy-chosen ids until the end condition, plus the text | n tokens |

Open fact this spec does not guess (the first task of S0): the exact ggml layout of the cross-KV (V may be stored transposed), so the comparison transposes the oracle, not proxima.

## What the reference does (read from source)

The defaults of `StftLogMelConfig` (`proxima-tensor/src/spec/generic_lowering.rs`) are sourced from this table; the source carries the values, not the model name or the `W:` cites.

| Item | Finding | Source |
|---|---|---|
| Audio constants | 16 kHz sample rate, `n_fft` 400, hop 160, 30 s chunk. `preprocessor_config.json` agrees (`sampling_rate` 16000, `n_fft` 400, `hop_length` 160, `chunk_length` 30, `nb_max_frames` 3000). | `W:include/whisper.h:33-36`; `~/models/asr/whisper-tiny/preprocessor_config.json` |
| STFT window | Hann, PERIODIC: `0.5 * (1 - cos(2*pi*i / length))` for `length = 400` (`periodic` sets the `-1` offset to 0). In no config file. | `W:src/whisper.cpp:3086-3095`; call with `periodic = true` at `:3075` |
| Framing and padding | Input placed after 200 samples of left reflect-pad (`stage_2_pad = frame_size / 2`), followed by 200 zero samples and 30 s (480,000) of zeros at the end; frames at stride 160; the last frame is dropped. | `W:src/whisper.cpp:3252-3273` |
| Power spectrum | `re*re + im*im` for bins `0..n_fft/2` (201 bins). | `W:src/whisper.cpp:3176,3193-3196` |
| Mel projection | Per mel bin `sum_k power[k] * filters[j*n_fft + k]` over the 201 bins, accumulated in double. The filterbank is data: whisper.cpp reads it from the ggml file, HF ships it as `mel_filters` in `preprocessor_config.json`. | `W:src/whisper.cpp:3198-3217`; `preprocessor_config.json` |
| Log, clamp, normalise | `log10(max(sum, 1e-10))`; then `mmax = (max over the whole mel) - 8.0`; values below `mmax` are raised to `mmax`; then `(x + 4.0) / 4.0`. | `W:src/whisper.cpp:3218`, `:3296-3311` |
| Encoder front | `conv1d(k=3, pad 1, stride 1)` + bias, GELU, `conv1d(k=3, pad 1, stride 2)` + bias, GELU; the position table is a checkpoint tensor (sinusoidal vs learned is the checkpoint's business, not a branch). | `W:src/whisper.cpp:2048-2056` |
| LayerNorm | affine, `eps = 1e-5` (hparam default; `config.json` carries no `layer_norm_eps`, so it is a family default with this source). | `W:src/whisper.cpp:608,2145` |
| Text context | `n_text_ctx` 448 (`max_target_positions` in config.json). | `W:src/whisper.cpp:602` |

## Decode semantics

Each fact is what the incumbent does; proxima's driver reproduces it, keyed from the checkpoint's `generation_config.json` / `config.json` where the value lives there (`suppress_tokens`, `begin_suppress_tokens`, `forced_decoder_ids`, `decoder_start_token_id`, `max_length`).

| Semantic | whisper.cpp behaviour | Source |
|---|---|---|
| Start-of-transcript prompt | `[sot]`, then for a multilingual model `[lang, transcribe]` (`translate` only if requested); `en` resolves to `sot + 1 + lang_id` | `W:src/whisper.cpp:7149-7160`, `:4383-4384` |
| Special-token ids (multilingual) | The English-only base ids `eot = 50256`, `sot = 50257` are bumped by 1 for multilingual models and the rest shift by an offset; for tiny: `sot` 50258, `en` 50259, `transcribe` 50359, `notimestamps` 50363. HF reads these from `generation_config.json` (`decoder_start_token_id` 50258, `lang_to_id`, `forced_decoder_ids`). | `W:src/whisper.cpp:445-455,1637-1650,448-449`; `~/models/asr/whisper-tiny/{config,generation_config}.json` |
| No timestamps | With `-nt` the prompt ends with `token_not` (`<\|notimestamps\|>`), every timestamp logit (`>= token_beg`) is `-inf` at every step, and `<\|notimestamps\|>` itself is always suppressed. | `W:src/whisper.cpp:7171-7173`, `:6386-6392` |
| Suppress tokens | Always `-inf`: `sot`, `nosp`, `solm` (diarisation off), `translate`, `transcribe`, `prev`, and every language token. At the FIRST generated step only (`is_initial`): `eot` and the `" "` token (`suppress_blank`, default true). The `suppress_tokens` id list in config.json (the OpenAI non-speech list) applies only under `suppress_nst` (default false), so it is OFF in the reference. | `W:src/whisper.cpp:6377-6383,6401-6422`, `:6129-6130`, `:6439-6450` |
| Greedy | One decoder at `t == 0`: argmax of the filtered logits, the first index winning ties (`result.p < probs[i]` is strict, ascending `i`). | `W:src/whisper.cpp:6655-6665`, `:7228-7237` |
| Max length | The decode loop runs `i < n_text_ctx / 2 - 4` steps (220 for tiny); `max_tokens` is 0 (unlimited) by default. | `W:src/whisper.cpp:7379`, `:6112` |
| End condition | A step that chooses `eot` completes the decoder; with `no_timestamps` the result length is `i + 1`. Otherwise the loop ends at the step cap. | `W:src/whisper.cpp:7562-7580` |

Divergence the AC8 comparison must respect: HF's `max_length` is 448 total tokens and applies `suppress_tokens` always; whisper.cpp's cap is 220 steps and applies the list only under `suppress_nst`. The incumbent wins (principle 14): driver defaults follow whisper.cpp, with the HF values as config overrides.

## Parity design

### No per-model profile: infer, default, then golden (PROXNUM-611)

Options: (A) an arch TOML per model (`whisper.toml`, like `src/arch/{llama,granite}.toml`, looked up by name at `src/arch_config.rs:278`); (B) a family template whose slots an inferrer fills from the checkpoint.
Chosen: B. Why: A makes every model a profile written in code, and `HfConfig` (`hf_config.rs:49-133`) and `ArchConfig` (`arch_config.rs:30-96`) have no `encoder_layers`, `decoder_layers`, `d_model`, `num_mel_bins` or `is_encoder_decoder`; the HF name table is llama/qwen only (`hf_bind.rs:1044-1106`). Making both table-driven is the config-first route.
The inferrer reads three sources in this order; a slot filled from an earlier source is never overridden by a later one:

1. Safetensors names and shapes: layer counts from `encoder.layers.N` / `decoder.layers.N`; `d_model` and FFN width from weight shapes; cross-attention from `encoder_attn.*` under decoder layers; a missing `k_proj.bias` (Whisper's `k_proj` has none); rank-3 conv weights `[out, in, 3]` mean conv1d, and `in` is the mel-bin count (80 vs 128); position tables from `embed_positions` shapes ([1500, 384] and [448, 384]); affine LayerNorm from `*.layer_norm.{weight,bias}`.
2. `config.json`: what weights cannot show (activation `gelu`, `is_encoder_decoder`, head counts, `vocab_size`, `max_target_positions`, `suppress_tokens`).
3. Family default, each with a cited source: Hann periodic n_fft 400 hop 160, the mel clamp (max - 8.0) and normalisation ((x + 4) / 4), LayerNorm `eps` 1e-5, all `W:` per the table above. These are in neither the safetensors nor config.json.

Where the front-end defaults live (R2 against the AC2 grep): the Hann-periodic window, the `1e-10` floor, the `max - 8.0` dynamic range and the `(x + 4) / 4` rescale are the DEFAULTS of generic `stft_log_mel` config fields (window kind, floor, dynamic range in decades, affine rescale: scale 0.25, offset 1.0). They are not selected by `feature_extractor_type` or any name lookup, so no "whisper" or "WhisperFeatureExtractor" string appears in `src/`. `n_fft`, `hop_length` and `mel_filters` (and `feature_size`, the mel-bin count) are read from `preprocessor_config.json`, the one place the checkpoint states them; the defaults cover only what that file omits.

A hypothesis becomes a model only by the golden check (R4): the loader returns an `Inferred` value that the runner refuses until a golden pass for that checkpoint sha is recorded; AC3 and AC4 check both directions.

### Trainable DFT (decision)

Options: (A) an FFT lowering; (B) the DFT as a matmul against host-built cos/sin constant matrices (`[400, 201]` each), windowed frames gathered by an index map.
Chosen: B. Why: `ScalarOp` has no sine or cosine (`proxima-tensor/src/op.rs:77-95`), so the twiddle factors are `Constant`/`Input` tensors built once on the host (the rope-table pattern, `proxima-model-interop/src/generate/residency_caches.rs`); the matmul is an Elementwise multiply plus Add-Reduce, which `proxima-autograd` already differentiates, so the front end is trainable for free (a gradient to the waveform flows through a matmul, a square, a second matmul and a log), whereas an FFT needs a butterfly adjoint that does not exist. Cost: 400 x 402 MACs per frame instead of ~3,000, i.e. 3,000 frames x 160,800 = 4.8e8 MACs per 30 s window (DERIVED), small next to a 32-layer encoder; S13 reports mel time as a measured stage so the claim is checked, not assumed. A faster FFT lowering is an optimisation behind the same op if the mel stage ever shows on the 5% bar.
Frame extraction is an Elementwise read through the window operand map (`stride*frame + dilation*tap`), which `adjoint.rs` differentiates by mask composition (`proxima-autograd/src/conv.rs:1-22`). The log is `ScalarOp::Logarithm`, the 1e-10 floor `Maximum`, and `max - 8.0` a whole-tensor Max-Reduce then `Maximum`; clamped bins get zero gradient, as in the incumbent.

### How autograd consumes the IR (cited, for the gradient ACs)

`proxima_autograd::adjoint::differentiate(program: &[Op], loss: NodeId)` (`proxima-autograd/src/adjoint.rs:214`; `differentiate_bound` `:228`, `differentiate_wanted` `:266`) walks the same `proxima_tensor::op::Op` list the forward builds. A lowering is trainable iff it is spelled as `Op`s the adjoint has a rule for; no per-lowering backward is written. Constraint found in `conv.rs:1-22`: an `Elementwise` read through a window operand map is differentiable, but a window on a `Reduce`'s own `in_map` raises `AutogradError::NonProjectionOperandMap`, and an `IndexMap::Computed` gather stops the backward walk (`GatheredContribution`), which is fatal for stacked layers. conv1d therefore follows `proxima_autograd::conv::conv2d` (`conv.rs:121`, mask-composed window, never a window on a Reduce): the same construction with one spatial axis, and both Whisper convs (stride 1 then stride 2) are stacked, so the stacked case is the one that must pass.
Oracle precedent: `conv2d_gradient_matches_central_difference_on_every_parameter` (`conv.rs:1217`) and `central_difference_gradient` (`norm.rs:747`). Loss builders: `cross_entropy` (`loss.rs:103`), `softmax_cross_entropy` (`loss.rs:153`). Optimizers: `AdamwConfig` (`optimizer.rs:661-667`), `adamw_step` (`:694`), `train_step` (`train.rs:69`), `fit` (`train.rs:171`).
Finding to fix, not carry: `AdamwConfig` is `f32`-typed (`optimizer.rs:661-667`; `AdamConfig` at `:70`); R12 forces the hyper-parameters and the moment-state format to come from config over `proxima-numeric`, so S10 widens them rather than wrapping them.

### Metric and thresholds (derivation)

Metric for a pair `(a, b)`, `b` the whisper.cpp dump: `maxabs_rel = max|a-b| / rms(b)`, plus cosine `a.b/(|a||b|)` (RMS in the denominator, as FLUX: a few large channels would flatter a max-normalised number).
Checkpoint dtype: F32 (tiny `config.json` `torch_dtype` float32; the 1b inventory confirms every tensor F32), so one rounding has relative error `u = 2^-24 = 5.96e-8`. A dot product of length `K` random-walks to `u * sqrt(K)` (the coherent worst case is `u * K`); two independent implementations combine to `sqrt(2)` of that; the maximum over ~5e5 elements sits near 4 sigma. Ceiling `T = 4 * u * sqrt(2 * n * K)`, with `n` the count of rounded dot-product-like ops on the path and `K` the widest reduction on it.

| Stage | n | K | ceiling T (maxabs_rel), DERIVED |
|---|---|---|---|
| log-mel (DFT 400 + 201-bin mel dot + log) | 3 | 400 | `4 * 5.96e-8 * sqrt(2*3*400)` = 1.2e-5 |
| conv stack, 2 convs + 2 GELU | 4 | 1152 | `4 * 5.96e-8 * sqrt(2*4*1152)` = 2.3e-5 |
| encoder output (4 layers, ~10 per layer + convs) | 44 | 1536 | `4 * 5.96e-8 * sqrt(2*44*1536)` = 8.8e-5 |
| cross K/V per layer (one Linear off the encoder output) | 45 | 1536 | 8.9e-5 |
| first-step logits (encoder + 4 decoder layers + 51,865-wide head) | 90 | 1536 | `4 * 5.96e-8 * sqrt(2*90*1536)` = 1.3e-4 |

The mel row carries a cancellation caveat: low-energy bins sit near the `1e-10` floor, where the relative error of `log10` is amplified; the clamp at `max - 8.0` bounds the damage (every kept bin is within 8 decades of the peak). S0 measured the incumbent's own F32 FFT against a double-precision DFT of the same frames at `3.25e-5` maxabs_rel, so the effective mel threshold is `2 x the measured floor` = `6.5e-5`; the DERIVED `1.2e-5` was below the incumbent's own F32 FFT noise. The conv stem matches the F32 oracle at `5.2e-6` (ceiling `2.3e-5`). The encoder's cosine is `0.99975`; its `maxabs_rel` `1.5` is one localized element, not a raised ceiling, and the cross K/V cosines are at least `0.9995`. large-v3-turbo: the same formula with `K = 5120` (FFN width) and `n = 32*10 + 4 = 324` ops gives `4 * 5.96e-8 * sqrt(2*324*5120)` = 4.3e-4, reported as a measurement, not a gate (tiny is the golden; turbo is the acceptance-by-config and speed check).

### Measured (2026-11-21, this box)

| Stage | Measurement | Threshold / note |
|---|---|---|
| mel | `maxabs_rel` 3.0e-5 on 80 x 3000 | 6.5e-5 (2 x the incumbent's measured 3.25e-5 F32-FFT floor) |
| conv stem | `maxabs_rel` 5.2e-6 | 2.3e-5 (the F32-derived ceiling) |
| encoder output | cosine 0.99975; `maxabs_rel` 1.5 at one element | cosine 0.9997; the localized element is the remaining arithmetic gap |
| cross K (4 layers) | cosine 0.99973 - 0.99981 | cosine 0.9997 |
| cross V (4 layers) | cosine 0.99956 - 0.99963 | cosine 0.9995 |
| tiny transcript | 24/24 ids, exact text, eot | exact |
| turbo transcript | 27/27 ids, exact text, eot | exact |
| perturbed wave | 26 ids, differs from the reference | the non-echo control |

## Architecture: reuse first

No new top-level abstraction. The pipeline is the existing Pipe composition (mel pipe, encoder pipe, cross-KV, decoder step pipe or teacher-forced pass, sampler); each component maps to existing pieces plus generic lowerings. The 610 1b inventory is the base (a search for STFT/DFT/hann/mel in sources found 0 hits there).

| Component | Reuse (cited) | Genuinely missing (generic, trainable) |
|---|---|---|
| IR | 5 ops: Input, Elementwise, Reduce, Iota, Constant (`proxima-tensor/src/op.rs:192-`); `ScalarOp` at `op.rs:77-95` | nothing: every item below lowers to Elementwise/Reduce with index maps |
| Weights, bind | safetensors and GGUF binding (`proxima-model-interop/src/bind.rs`, `convert.rs`); HF bind (`src/lowering.rs:532` `bind_hf_checkpoint`) | the inferrer (R2): table-driven `HfConfig` / name table replacing the llama-only table at `hf_bind.rs:1044-1106`; no new file under `src/arch/` |
| Log-mel front end | window operand map + Add-Reduce (`proxima-tensor/src/map.rs`, convolution row); `Logarithm`, `Maximum` in `ScalarOp`; host-built constants like rope tables (`generate/residency_caches.rs`); `mel_filters` from `preprocessor_config.json` | generic `stft_log_mel` lowering (frame, hop, n_fft, window, filters, floor, range, scale as config; the DFT-matmul decision above) |
| conv1d | `proxima_autograd::conv::conv2d` (`conv.rs:121`, mask-composed, stackable); ONNX 1-D window (`proxima-onnx/src/lower.rs:4961`, `:6085`, forward-only) | conv1d in autograd with stride and padding; the autograd path is 2-D only today |
| GELU, position tables | `ScalarOp::Erf` (`op.rs:91`); Gather for `embed_positions` (checkpoint tensors) | none |
| Affine LayerNorm | `layer_norm_no_affine` (`proxima-tensor/src/spec/generic_lowering.rs:49`), Multiply and Add with a `[1, d]` broadcast | `layer_norm(x, weight, bias, eps)` composed from it: no new Reduce, so the adjoint is the existing one |
| Self-attention | omega attention lowering and goldens (`omega/src/msl/attn_golden_tests.rs:24,68`), causal form | none (encoder uses `non_causal_attention`, decoder self-attention the causal form) |
| Cross-attention | `non_causal_attention` with s != t (`generic_lowering.rs:131`) | the cross-KV: K/V projected from the encoder output once. A graph that takes `cross_k`/`cross_v` as `Input`s and a graph that computes them are the same ops: inference binds the cached tensors, training lets the adjoint flow through the projection |
| Two graphs from one lowering (R9) | `adjoint::differentiate` over the forward `Op` list (`adjoint.rs:214`) | a family-template emitter returning `{encode, cross_kv, decode_step (cached), decode_full (teacher-forced)}` over the same layer builders; `decode_full` = the causal-mask pass over the whole target plus `softmax_cross_entropy` (`loss.rs:153`) |
| Decode driver | interop decode is decoder-only (`generate/decode.rs:2677`) | an encoder-decoder driver: encode once, cross-KV once, then a per-step loop with the prompt, logit filters and end condition of section Decode semantics, selected by `is_encoder_decoder` |
| Tokenizer | byte-level BPE core (`proxima-tokenizer/src/bpe.rs:23`, `byte_level.rs:1`); an ad-hoc `tokenizer.json` read only in `examples/ouro_loopcd_lab/checkpoint.rs:288-306` | a generic `tokenizer.json` loader (vocab, merges, added/special tokens) in `proxima-tokenizer`, special tokens kept out of BPE merging |
| Optimizer | `AdamwConfig`/`adamw_step` (`optimizer.rs:661,694`), `train_step`/`fit` (`train.rs:69,171`) | optimizer hyper-parameters and state formats as config over `proxima-numeric` (R12), not `f32` fields |

Designs abandoned under principle 1 and R2: a `whisper.toml` profile (the 611 direction); a bespoke `WhisperPipeline`; an FFT lowering with a hand-written backward; an inference-only cross-KV cache op (it would force a second, training-only path, violating R9); a `Box<dyn Decoder>` for greedy vs beam. Each missing piece is a generic lowering selected by config or inferred.
Tier: new lowering code is no_std + alloc like the tensor crate; the example, bench harness and file readers are std and additive.

## Acceptance criteria

Commands run from the worktree root. `$K = ~/.cache/kvgate/proxnum610`, `$M = ~/models/asr/whisper-tiny`, `$J = ~/repos/others/whisper.cpp/samples/jfk.wav`.
S0 dumps live in `$K/ref/` for tiny: `mel.f32`, `enc.f32`, `cross_k_{0..3}.f32`, `cross_v_{0..3}.f32`, `logits0.f32`, `logits0_filtered.f32`, `tokens.i32`, `ref.json`, `ref.log` (14 files).
Test target `whisper` in `proxima-model-interop`, gated on `PROXIMA_WHISPER_DIR=$M`, `PROXIMA_WHISPER_REF=$K/ref`; a missing fixture is a failure, not a skip, so `0 passed` is red.

| AC | Discharges | Command | Expected |
|---|---|---|---|
| AC1 reference run + dumps | R1 | the patched `whisper-cli` run, then `ls $K/ref` | the named dumps exist; `mel.f32` is 80 x 4100 x 4 = 1,312,000 bytes, `enc.f32` 1500 x 384 x 4 = 2,304,000 bytes, `logits0.f32` 51,865 x 4 = 207,460 bytes; `ref.json` holds a non-empty transcript; `tokens.i32` ends with the eot id; the same run on `ggml-large-v3-turbo.bin` writes `$K/ref-turbo/ref.json` and `tokens.i32` (2 of 2 models have a reference) |
| AC2 genericity grep | R2, R14 | `git grep -n -i "whisper" -- omega/src proxima-model-interop/src proxima-tensor/src ':!*.md'` and `ls proxima-model-interop/src/arch` | 0 lines from the grep; the `ls` set equals the recorded list at `b7c2d6b22` (no whisper TOML) |
| AC3 same path, two sizes by config | R3 | `cargo val -p proxima-model-interop --features toml --test whisper infer:: -- --test-threads 1` | `4 passed`: tiny infers `{enc 4, dec 4, d_model 384, mel 80, cross-attn yes, k_proj bias no, conv1d yes}`; large-v3-turbo infers `{enc 32, dec 4, mel 128}` from its safetensors HEADERS (no weights loaded) and `config.json` through the SAME function (one helper is called with both fixtures); the two inferred records differ only in values; `git grep -n -e "== *80" -e "== *128" -- proxima-model-interop/src omega/src` prints 0 lines |
| AC4 accepted only after golden | R4 | `cargo val -p proxima-model-interop --features toml --test whisper accept:: -- --test-threads 1` | `4 passed`: an inferred model with no recorded golden pass is refused with reason `golden_missing`; one whose golden exceeds the AC6/AC7 ceilings is refused with `golden_failed` naming the stage; the latest record per stage is authoritative (a later failure revokes, a later pass supersedes); tiny after AC5-AC7 pass is accepted (asserts the refusal reasons, not just the variant) |
| AC5 log-mel | R5 | `PROXIMA_WHISPER_DIR=... PROXIMA_WHISPER_REF=... cargo val -p proxima-model-interop --features std,toml,proxima-autograd --test whisper mel:: -- --test-threads 1` | `3 passed`: `maxabs_rel` vs `mel.f32` <= 6.5e-5 (the measured floor) at shape 80 x 3000; frame count 3000 asserted; the clamp test |
| AC6 encoder + cross K/V | R6 | `PROXIMA_WHISPER_DIR=... PROXIMA_WHISPER_REF=... cargo val -p proxima-model-interop --features std,toml,proxima-autograd --test whisper encoder:: -- --test-threads 1` | `2 passed`; the combined test prints the conv, encoder, 4 K and 4 V rows with the cosine gates above; the oracle's F16 im2col is recorded as the floor |
| AC7 first-step logits | R7 | open (not part of the landed S9); `cargo val -p proxima-model-interop --features std,toml,proxima-autograd --test whisper logits:: -- --test-threads 1` | `2 passed`: unfiltered and filtered logits vs `logits0*.f32` <= 1.3e-4; argmax of the filtered logits equals `tokens.i32[0]` |
| AC8 transcript | R8 | `PROXIMA_WHISPER_DIR=... PROXIMA_WHISPER_REF=... cargo run --release -p proxima-model-interop --features std,toml,proxima-autograd --example whisper -- --wav $J --model $M --lang en` | tiny: 24 ids equal `$K/ref-f32/tokens.i32` exactly; turbo: 27 ids equal `$K/ref-turbo/tokens.i32` exactly; both texts equal their `ref.json`, eot before 220; `--noise 0.05` produces a different id list (2 of 2 models) |
| AC9 prompt, suppress, end semantics | R8 | `cargo val -p proxima-model-interop --features std,toml,proxima-autograd --test whisper decode_semantics:: -- --test-threads 1` | `7 passed`, one per row of the Decode semantics table: prompt `[50258, 50259, 50359, 50363]` for tiny, every timestamp logit `-inf`, the always-suppressed set, `eot` and `" "` suppressed at step 0 only, ascending-index tie break, 220-step cap, eot ends |
| AC10 gradients, one per new lowering | R10 | `cargo val -p proxima-model-interop generic_lowering::grad_ -- --test-threads 1` , `cargo val -p proxima-autograd grad_cross_attention -- --test-threads 1` (cross-attention; 3 tests incl. the forward reference), `cargo val -p proxima-autograd grad_layer_norm` (affine LayerNorm lives in `proxima-autograd/src/generic_lowering_grad.rs`) and `cargo val -p proxima-autograd conv1d_gradient_matches_central_difference` | `6 passed` in total: STFT/log-mel (`cargo val -p proxima-autograd grad_stft_log_mel`; gradient to the waveform and to the filterbank; 64-sample signal, n_fft 16, 4 mel bins), conv1d stride 1, conv1d stride 2 STACKED (two layers, the gradient reaches the first layer's weight), affine LayerNorm (x, weight, bias), cross-attention (query side and encoder-memory side, s = 3, t = 5), cached-vs-recomputed cross-KV give equal gradients; each check `max\|g_ad - g_fd\| / max\|g_fd\| <= 5e-4` (DERIVED: `4 * (10u/h + h^2/6)`, h = 5e-3, u = 2^-24); each test prints the element count it compared (N > 0) |
| AC11 teacher-forced overfit | R9, R11 | `cargo run --release -p proxima-model-interop --example whisper_train -- --wav $J --model $M --steps 30 --optimizer adamw` | prints 30 loss lines; the loss is monotonically non-increasing over steps 1..30 (`monotone=true`); `loss_30 / loss_1 <= 0.1` (ASSUMED fraction; S11 records the achieved ratio, a miss is a defect to explain, not a number to loosen); `loss_1` equals `-mean log softmax(logits)[target]` computed from the AC7-validated forward logits on the same teacher-forced input within 1e-4 relative (DERIVED from the AC7 logit ceiling 1.3e-4, the CE being a mean over targets) |
| AC12 one graph set, two modes | R9 | `cargo val -p proxima-model-interop --test whisper two_modes:: -- --test-threads 1` | `3 passed`: the emitter returns encode, cross_kv, decode_step and decode_full from one set of layer builders; teacher-forced logits at position `t` equal the cached step-by-step logits at step `t` for all 8 positions of a fixed 8-token target within 1e-5 (DERIVED: same ops, only the cache path differs); `git grep -n -i -e "inference_only" -e "training_only" -- omega/src proxima-tensor/src` prints 0 lines |
| AC13 optimizer and state formats are config | R12 | `cargo val -p proxima-autograd optimizer_config::` | `5 passed`: an optimizer built from config, TOML and env has identical state (config-as-mirror); the f32-moment step is bit-identical to the pre-config result and the bf16-moment step agrees with an f64 reference within bf16's derived `u`; an unsupported layout is refused with a typed error naming it; `git grep -c "pub .*: f32" -- proxima-autograd/src/optimizer.rs` prints 0 |
| AC14 speed | R13 | `scripts/whisper_ab.sh` (3 alternating pairs, quiet window, release builds, large-v3-turbo, jfk.wav) | prints `runs=3 per side`; `omega_median_ms / whispercpp_median_ms <= 1.05`; per-stage rows (mel, encode, decode; decode also per token) for both sides with the ratio; peak RSS (`/usr/bin/time -l`, "maximum resident set size") for both sides (report-only); CoV <= 5% per side; model load reported separately |
| AC15 training-step memory | R13 | `scripts/whisper_train_mem.sh` (tiny, 30 steps) | prints `steps=30`; peak RSS from `/usr/bin/time -l` and per-step phase footprints (after forward, after backward, after update) with a 3-run spread; report-only, recorded in TASKS.md as measurements (PROXNUM-606: `time -l` spread is 40-60 MiB, so the phase samples carry the claim, not the total) |
| AC16 new generics have non-Whisper tests | R14 | `cargo val -p proxima-model-interop generic_lowering:: -- --test-threads 1` and `cargo val -p proxima-tokenizer --features hf tokenizer_json::` | `6 passed` for lowerings (stft_log_mel, conv1d, layer_norm, cross_attention_kv, encdec_driver, mel_clamp) and `4 passed` for the tokenizer (byte-level BPE round trip, special tokens kept whole, merge order, an added-token id); every fixture is a non-Whisper shape or vocabulary |

## Speed methodology (AC14, AC15)

Same box (M1 Max), quiet window (no other cargo or agents; free memory checked first), release builds, 3 alternating runs per side (whisper.cpp, omega, whisper.cpp, omega, whisper.cpp, omega), median reported with CoV; CoV > 5% invalidates the set and it is rerun.
Per-stage times: whisper.cpp from `whisper_print_timings` (mel, encode, decode); omega from the example's stage timers. Stage boundaries are the dump tap points (mel at the normalised mel, encode at the encoder output, decode from prompt to eot).
Wall time is whole process launch to transcript printed, model load on both sides, load also reported separately.
The bar is the owner's: within 5% of whisper.cpp on large-v3-turbo. The ratio is a measurement; the mechanism for any gap (which stage, which op) is traced before it is called a result (principles 18, 19). Training has no incumbent time bar; its memory and step time are recorded as measurements.

## Out of scope

Beam search, temperature fallback and best-of sampling, timestamps and timestamp decoding (the reference runs `-nt`), language auto-detection, translation, VAD, diarisation (`solm`), long-form seeking beyond one 30 s window, word-level alignment heads, quantised ggml checkpoints, distilled 2-layer variants, fine-tuning data pipelines (S11 proves the graph trains on one utterance; a corpus loader is another card), training large-v3-turbo, non-Metal backends, Python anywhere (owner rule).

## Risks

- The incumbent build is pending the owner's command: S0, and every AC that needs a dump, is blocked on it. Nothing here was measured.
- Prompt divergence: whisper.cpp's 220-step cap and `suppress_nst`-gated suppress list differ from HF's; the incumbent wins and the HF values are overrides.
- The mel clamp couples every output element to the global max (`W:src/whisper.cpp:3296-3303`), so a gradient through it is a gradient through a max; the adjoint of the whole-tensor Max-Reduce must be checked (AC10), not assumed.
- DFT-as-matmul at 3,000 frames is 4.8e8 MACs (DERIVED); if the mel stage shows on the 5% bar, an FFT lowering behind the same op is the follow-up, not a redesign.
- The 5% bar on large-v3-turbo is dominated by the 32-layer encoder, but decode (4 layers, per-token) can miss it too, hence per-stage rows.
- `AdamwConfig` is f32-typed (`optimizer.rs:661`); widening an existing public type is guarded by the config-as-mirror test (AC13).
- Machine freezes on builds and large tests are on record: one cargo job at a time, `-j 2`, free memory checked first.
