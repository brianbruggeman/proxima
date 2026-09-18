# technique taxonomy — the research index behind the model descriptor

Draft. This is the vocabulary the composable model descriptor is built from:
every model is a *composition of named techniques*, and every technique here
becomes a config variant (`AttentionKind`, `RopeKind`, `FfnKind`, `NormKind`,
`CacheStrategy`, ...) whose rustdoc cites the row below. You should never again
have to recall a paper name — you read the descriptor, it names the technique,
the technique names the paper.

Legend: **variant** = the future config enum variant · *paper* = source ·
`where` = where it lives in proxima today.

## attention

| variant | paper (keyword) | idea | where |
|---|---|---|---|
| `Gqa { kv_heads }` | Ainslie et al. 2023, **GQA** | fewer KV heads than Q heads, shared across a group | `LayerAttentionConfig.kv_heads`, qwen35/gemma4 |
| `SlidingWindow { window }` | Beltagy 2020 **Longformer**; Jiang 2023 **Mistral** | attend only the last W tokens | `causal_mask_windowed`, gemma4 `sliding_window_pattern` |
| `FlashOnlineSoftmax` | Milakov & Gimelshein 2018 **online softmax**; Dao 2022 **FlashAttention** | combine score blocks without materializing the full matrix | the two-range combine `append_lfm2_two_range_cached_attention` |
| `ScoreScale::Unscaled` | **Gemma** (scale folded / =1.0) | no `1/sqrt(d)` term | `AttentionScoreScale::Unscaled` |
| `ValueNorm` | Gemma / QK-norm lineage | per-kv-head RMSNorm on V (or Q/K) post-projection | `value_norm` |
| `ValueSource::SharedWithKey` | **Gemma** (full layers) | V derived from K, no separate `attn_v` | `ValueSourceKind::SharedWithKey` |

## positional

| variant | paper | idea | where |
|---|---|---|---|
| `Rope { base }` | Su et al. 2021 **RoFormer / RoPE** | rotate Q/K by position-dependent angle | `fused_rope`, `rope_cos/sin` |
| `DualBaseRope { base, base_swa }` | **Gemma** local-global | different `freq_base` on SWA (1e4) vs full (1e6) layers | `RopeTableSel` per-layer |

## ffn / experts

| variant | paper | idea | where |
|---|---|---|---|
| `Activation::SwiGLU` / `GeGLU(GeluTanh)` | Shazeer 2020 **GLU Variants** | gated FFN; SwiGLU = Silu gate, GeGLU = Gelu gate (gemma4 uses GeluTanh) | `Activation`, `FfnCombination` |
| `Moe { experts, used }` | Shazeer 2017 **Outrageously Large NNs**; Fedus 2021 **Switch**; Jiang 2024 **Mixtral** | route each token to top-k of N experts | `expert_count/used`, `append_moe_ffn`, discipline ROW ~421-425 |
| `ParallelDenseMoe` | **Gemma**-style hybrid | a dense branch + a routed-MoE branch, summed and re-normed | `FfnCombination::ParallelDenseMoe` |

## normalization / output

| variant | paper | idea | where |
|---|---|---|---|
| `Norm::RmsNorm` | Zhang & Sennrich 2019 **RMSNorm** | normalize by RMS, no mean subtraction | `rmsnorm` |
| `LogitSoftcap { cap }` | **Gemma 2** 2024 | `cap * tanh(logits / cap)` to bound logits/attention | gemma4 `final_logit_softcapping=30`, the tanh composition |

## sequence mixer (non-attention)

| variant | paper | idea | where |
|---|---|---|---|
| `GatedDeltaNet` | Yang et al. 2024/25 **Gated DeltaNet** | delta-rule + gating linear-attention SSM mixer | qwen35 `GatedDeltaNet`, ROW ~427 (single-position step) |

## cache strategy — the axis that must become a *parameter*

| variant | source | idea |
|---|---|---|
| `CacheStrategy::Cacheless` | — | recompute the whole sequence each step (state ≡ ∅). the gemma4 default that made it 0.24 tok/s |
| `CacheStrategy::TwoRange` | proxima-internal (Flash-style combine) | local block = this step's own K/V, cache block = history via online-softmax. **the general form** — cacheless and single-range are its degenerate cases |

*(single-range is a redundant third shape; it should not survive the collapse.)*

## quantization codecs

k-quants **Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q4_0/Q5_0/Q5_1/Q8_0** — Gerganov et al.
**GGML k-quants** (block-wise formats). proxima re-implements the *codec/format*
(packed reduce, ROW ~428-431) and never uses llama.cpp as a runtime/oracle
(per the no-llama.cpp rule). These are a decode axis, orthogonal to architecture.

## training

| variant | paper | where |
|---|---|---|
| Adam / AdamW | Kingma & Ba 2014 **Adam**; Loshchilov & Hutter 2017 **AdamW** | `proxima-autograd/optimizer.rs` |
| reverse-mode autodiff | (graph-transform adjoint) | `proxima-autograd/adjoint.rs` |

---

## how this drives the design

- **(a) research** — a model descriptor is a row of these variant names; each
  variant's doc links here; the discipline log becomes the *changelog* of this
  registry (technique landed → paper → parity ROW). Adding research = a cited
  variant, never a new builder.
- **(b) fsm** — a layer is a **transform pipe** over these configs; the cache is
  the **`ServingState` FSM's** carried state; `CacheStrategy` is its state
  representation; the whole composes up to the `LoadedModel` pipe.
- **(c) conflaguration** — the descriptor is a `#[derive(Settings, Validate)]`
  type; a high-level spec (`{blocks, attention, ffn, cache, softcap}`) expands
  via layered defaults into `[LayerSpec]`; **gguf metadata is this config in
  another dialect**, so a new model = a validated config, zero new Rust
  (conflaguration §4 config-as-composition).

ROW numbers marked `~` need one pass against `proxima-tensor/docs/discipline.md`
to pin the exact entry; the papers are firm.
