---
status: draft (paper test, not audited)
sketch: 2 of 14, pipeline-as-data
hooks exercised: H3 H4 H5 H6 H7 H8 H11 H12 H13 H14 H15 H16
checkpoint: qwen3.5 0.8B (general.architecture = qwen35), the ollama blob in checkpoints.toml
code read at: main (git show main:...); nothing was built or run, so every behavioural claim is "read", never "measured"
---

# sketch 2: qwen3.5 (GDN + attention) as configuration

Verdict field first (full evidence in section e):

    verdict = "fits after 12 named hook changes (the 12 rows of section e, covering gaps G1-G14),
               on top of the descriptor-config slice architecture-as-data already specifies (G0,
               not counted). No new pipe is needed. The byte-identical-digest claim (R7) is
               unmeasured."

The one contested decision: **the cached hybrid engine (G3).** `build_forward` today is a
three-arm dispatcher over whole-model builders (`descriptor.rs:303-426`), and no arm runs a
state-cached recurrent layer. Taken: lift the layer loop already living at
`attention_forward.rs:3046-3525` into a schedule-driven engine, replacing its
`(layer + 1) % full_attention_interval` predicate (`:3069`) with `descriptor.layers[layer].kind`.
Abandoned: a fourth arm `CacheStrategy::Qwen35` that forwards to the bespoke function with a
`full_attention_interval` argument. That arm keeps the bespoke op order by construction and still
fails R9 (a new model variant is a config file with zero new Rust), because a variant with a different layer pattern would need new Rust.

## 0. sources and notation

Paths are relative to `proxima-tensor/src/spec/` (T), `proxima-model-interop/src/` (I),
`proxima-tokenizer/src/` (K), or are the llama.cpp oracle at commit f1ea20621
(`/Users/brianbruggeman/repos/others/llama.cpp`, L). `kv:N` is a line of
`proxima-model-interop/tests/fixtures/llama-parity/qwen35/gguf_kv.txt`. `bound:` and `digest:`
are `llama-parity/qwen35.bound` and `qwen35.digest`.

Status tags: READ (I opened the lines), DERIVED (arithmetic on READ values, not a measurement of
what ran), UNMEASURED (nothing was executed). Every `DERIVED` byte count below is arithmetic over
config values; no allocation was observed.

Fixture limit: `kv:14` records `head_count_kv` as `[0, 0, 0, 2, 0, 0, ...]` (24 entries, 6
shown). Attention layers 3,7,11,15,19,23 come from `full_attention_interval = 4` (`kv:22`) and
are corroborated by the real bound names: `bound:` holds exactly `blk.{3,7,11,15,19,23}.attn_q.weight`
and 18 `ssm_a` tensors.

## a. the configuration

Field names follow `ModelDescriptor`/`LayerSchedule` (T/descriptor.rs:54-124,
T/attention_forward.rs:305-459) wherever a field exists. A name marked `# GAP Gn` is a field the
config needs and no descriptor has. Values not in the GGUF are marked `# not GGUF`.

```toml
# ---- [model]: ModelDescriptor scalars ------------------------------------------------------
[model]
family_label            = "qwen35"        # kv:9 general.architecture. A label for logs, never a branch key (R6)
vocab                   = 248320          # kv:60 tokens length; DERIVED-equal to token_embd rows (bound: 270172160 B Q8_0 = 248320 x 1024 x 34/32)
embedding               = 1024            # kv:20
feed_forward            = 3584            # kv:21
query_heads             = 8               # kv:13
block_count             = 24              # kv:18
expert_count            = 0               # no expert_count key; qwen35.cpp:471 asserts dense (T/attention_forward.rs:2876-2879)
expert_used_count       = 0
leading_dense_block_count = 24
expert_feed_forward     = 3584            # equals feed_forward, the precedent at descriptor.rs:205
rms_epsilon             = 9.999999974752427e-07   # kv:16.  GAP G4
context_length          = 262144          # kv:19
embedding_scale         = "none"
logit_softcap           = "none"
ple_dim                 = "none"
sliding_kv_ring         = false
cache_strategy          = "state_cached"  # GAP G3: not a CacheStrategy variant today

# ---- [layers]: the schedule, as a repeating group (config-as-composition) ------------------
[layers]
repeat = 6                                # block_count / len(unit)
unit   = ["gdn", "gdn", "gdn", "attention"]
# expansion rule = what kv:22 means: attention at i iff (i + 1) % 4 == 0 (T/attention_forward.rs:3069).
# validate: kv_heads[i] != 0  <=>  unit[i % 4] == "attention"  (cross-checks kv:14 against the pattern)

# ---- [layer.attention]: LayerAttentionConfig (T/attention_forward.rs:411-445) --------------
[layer.attention]
head_dim          = 256                   # kv:15 key_length == kv:17 value_length (NOT embedding / query_heads = 128)
kv_heads          = 2                     # kv:14 non-zero entries
mask_window       = "none"                # no sliding_window key
value_source      = "projected"
key_source        = "projected"
rope_table        = { cos = "rope_cos", sin = "rope_sin" }
rope_pairing      = "interleaved"         # T/hyperconn_qwen35_dense.rs:698-704 (see U2)
score_scale       = { inverse_sqrt = 256 }   # T/attention_forward.rs:3000 (1/sqrt(attn_head_dim))
value_norm        = false
rotary_dim        = 64                    # kv:25 rope.dimension_count.  GAP G2
output_gate       = "sigmoid"             # attn_q is [1024, 2 x 8 x 256]; bound: 4456448 B Q8_0 = 1024 x 4096.  GAP G2
qk_norm           = true                  # attn_q_norm / attn_k_norm, 256 floats (bound: 1024 B).  GAP G2

# ---- [layer.gdn]: new LayerKind::Gdn(GdnConfig) ---------------------------------------------
[layer.gdn]                               # GAP G1. Mapped onto append_qwen35_ssm_mixer's own arguments
key_heads         = 16                    # kv:31 ssm.group_count        -> kv_heads
key_head_dim      = 128                   # kv:33 ssm.state_size         -> key_dim = 128 x 16 = 2048
value_heads       = 16                    # kv:34 ssm.time_step_rank     -> group = value_heads / key_heads = 1
inner_size        = 2048                  # kv:32 ssm.inner_size         -> value_dim; head_v_dim = 2048 / 16 = 128
conv_kernel       = 4                     # kv:30 ssm.conv_kernel        -> l_cache
v_head_reordered  = true                  # kv:35.  See U1: the bespoke program bakes false
output_gate       = "silu"                # T/attention_forward.rs:3426 GdnOutputGate::Silu (T/lfm2_qwen35_gdn.rs:1069-1078)
# fixed by the primitive, not fields (T/lfm2_qwen35_gdn.rs): silu after the causal conv (:1754), l2norm on q and k
# (:1787-1788), beta = sigmoid(.) (:1678), gate = softplus(alpha + ssm_dt) x ssm_a (:1695-1707), gated rmsnorm (:2135-2185)

# ---- [layer.ffn]: LayerFfnConfig (T/attention_forward.rs:305-361), every layer ------------
[layer.ffn]
combination                = "exclusive"
activation                 = "silu"
post_attention_norm        = false        # TRAP: this flag means Gemma's sandwich norm (:306-310), not qwen35's tensor of that name
pre_norm_leaf              = "post_attention_norm.weight"   # GAP G7: the FFN pre-norm leaf name; generic builders declare "ffn_norm.weight" (:1819-1823), qwen35 declares this (:3059-3064)
output_scale               = false
routed_expert_bias         = false
dense_feed_forward         = "none"
exclusive_dense_post_norm  = false
ple                        = false

# ---- [norm] / [head] -------------------------------------------------------------------------
[head]
output_norm      = "output_norm.weight"
tied_embeddings  = true                   # no GGUF key. bound: output.weight and token_embd.weight share sha256 093f7e90...; rule is qwen35.rs:466-479
last_row_only    = "per call"             # T/descriptor.rs:276-280, not config

# ---- [rope]: the step-input table, one flat [positions, 32] table -----------------------------
[rope]                                    # GAP G8: no descriptor field holds any of these; base lives in ModelArchitecture (I/qwen35.rs:682)
base                    = 10000000.0      # kv:27
rotary_dim              = 64              # kv:25
dimension_sections      = [11, 11, 10]    # kv:26; sums to rotary_dim / 2 = 32
mrope_sections          = [11, 11, 10]    # kv:24 (duplicate key)
mrope_section           = [11, 11, 10]    # kv:29 (duplicate key)
mrope_interleaved       = true            # kv:28
scaling                 = "none"          # no rope.scaling.* keys
freq_factors            = "none"          # no rope_freqs tensor in bound:

# ---- [bind] ------------------------------------------------------------------------------------
[bind]
file_type = 7                             # kv:10 (Q8_0 mostly). Codec is read per tensor from the directory, never from config
[bind.alias]                              # GAP G7: program leaf name -> file tensor name (I/qwen35.rs:362-375, :431-437)
"ssm_in.weight"   = "attn_qkv.weight"
"ssm_gate.weight" = "attn_gate.weight"
"ssm_dt.bias"     = "ssm_dt"

# ---- [cache.*]: state per layer kind ------------------------------------------------------------
[cache.attention]                         # 6 layers. Leaves kv_cache.{l}.{k_first,k_second,k_pass,v} (T/attention_forward.rs:3249-3288)
growth        = "append"                  # host Vec, grows by position (I/generate/residency_caches.rs:418-441)
rewind        = "truncate"                # G12: capability exists for 3-wide LayerCache only (:155-162), not for this 4-wide cache
placement     = "host"                    # G11
# DERIVED row width: 64 + 64 + 384 + 512 = 1024 f32 = 4096 B per position per layer (24576 B per position over 6 layers)

[cache.gdn]                               # 18 layers. Leaves ssm_cache.{l}.conv_history [3, 6144] and .state [128, 128, 16, 1]
growth            = "fixed"               # shapes derive from [layer.gdn] (T/attention_forward.rs:3346-3398); no field
rewind            = "snapshot"            # G12/G13
rollback_snapshots = 0                    # not GGUF. L/include/llama.h:370 n_rs_seq (0 = no rollback); default 0 at L/src/llama-context.cpp:3718
placement         = "host"                # G11
# DERIVED size: (3 x 6144 + 128 x 128 x 16) x 4 B = 1,122,304 B per layer; x 18 = 20,201,472 B

# ---- tokenizer (H3), detokenizer (H16) ---------------------------------------------------------
[tokenizer]
model            = "gpt2"                 # kv:55 -> byte-level BPE (K/gguf.rs:84-101)
pre              = "qwen35"               # kv:57.  GAP G9: never read (K/gguf.rs:40-49 has no PRE key)
tokens           = 248320                 # kv:60
merges           = 247587                 # kv:54
token_type       = "present"              # kv:59 -> Vocab::with_token_types (K/gguf.rs:114-117) feeds added-token markers (K/pipe.rs:44)
scores           = "present, unused"      # kv:58, 248320 f32. The "gpt2" arm never reads it (K/gguf.rs:91-101)
bos_token_id     = "absent"
unknown_token_id = "absent"
eos_token_id     = 248046                 # kv:53
padding_token_id = 248044                 # kv:56.  GAP G10: no reader
add_bos_token    = "absent"               # None -> Vocab::with_bos_eos_policy keeps it unset (K/gguf.rs:81,118)
add_eos_token    = false                  # kv:51
add_padding_token = false                 # kv:52.  GAP G10: no reader
[tokenizer.presplit]                      # GAP G9: values taken from L/src/llama-vocab.cpp:392-396 (pre type QWEN35)
digit_group             = 1               # \p{N}      (LLAMA3 is \p{N}{1,3}: L/src/llama-vocab.cpp:284-290, K/pretokenize.rs:10)
letters_include_marks   = true            # [\p{L}\p{M}]+ and a punctuation class that excludes \p{M}
[template]                                # H2, outside this sketch's hook range; recorded so the key has an owner
chat_template = "gguf"                    # kv:50 tokenizer.chat_template (a Jinja string). Consumer today: none (examples/gguf_generate.rs:179-195 hard-codes <|im_start|> framing behind architecture.starts_with("qwen35"))

[detokenize]
hold_incomplete_utf8 = true               # I/generate/residency_caches.rs:3395-3409 (drain_lossy_utf8 leaves an unfinished tail pending)
suppress_control     = true               # :3453-3458, TokenType::Control ids stay in the id list, empty text

# ---- sampler (H14), stop (H15): not in the GGUF ---------------------------------------------------
[sampling]                                # not GGUF: no sampling keys exist in kv:1-60
temperature = 0.0
top_k = 0
top_p = 1.0
min_p = 0.0
repeat_penalty = 1.0
frequency_penalty = 0.0
presence_penalty = 0.0
repeat_last_n = 64                        # serving-only default, I/serving.rs:1157 (field doc :808-812)
seed = 0                                  # I/serving.rs:1161
[stop]
ids = [248046]                            # eos only (owner policy 2026-09-17; I/generate/residency_caches.rs:3448)

# ---- placement (H8), serving (H11, H13) ----------------------------------------------------------
[placement]
weights = "gpu"                           # not GGUF
[serving]
speculative = "none"                      # qwen35 has no verify program (I/architecture.rs:311-335 default Ok(None))
prefill_rows_per_evaluation = "derived"   # G6: 1 while a Gdn layer exists and no width-pinned program is built
[serving.prompt_cache]                    # I/serving.rs:472-557; defaults per its doc at :559-562; consumed, but inert for this model until G12/G13
checkpoint_interval = 2048
max_checkpoints     = 4
```

Everything in `[sampling]` is `SamplingConfig::default()` (`K/sample.rs:134-144`) plus the serving-only
`repeat_last_n` and `seed` (`I/serving.rs:1153-1161`).

Excluded on purpose: 15 multimodal keys (`kv:23`, `kv:36-49`: `image_token_id`, `vision_start_token_id`,
`vision_end_token_id`, `vision.*`). Their only consumer is the rejection at `I/serving.rs:1335-1343`
(`multimodal_projector = true` returns `UnsupportedServingConfig`). The catalog has no multimodal
intake stage, so no H# owns them; the text path needs no change.

## b. field to consumer map, and the gap list

Status: C = consumed by an existing primitive or descriptor field; G = HOOK GAP.

### b1. every field

| field | value source | consumer today (file:line) | status |
|---|---|---|---|
| model.vocab | kv:60, bound: | `ModelDescriptor.vocab` T/descriptor.rs:55; leaf `token_embd.weight` T/attention_forward.rs:2982-2987; bespoke read from the directory I/qwen35.rs:185 | C |
| model.embedding | kv:20 | descriptor.rs:56 | C |
| model.feed_forward | kv:21 | descriptor.rs:60; leaves T/attention_forward.rs:3221-3235 | C |
| model.query_heads | kv:13 | descriptor.rs:68 | C |
| model.block_count | kv:18 | descriptor.rs:69 | C |
| model.expert_*, leading_dense_block_count, expert_feed_forward | 0 | descriptor.rs:63-72 | C (inert) |
| model.embedding_scale, logit_softcap, ple_dim, sliding_kv_ring | none | descriptor.rs:80-99 | C (inert) |
| model.rms_epsilon | kv:16 | not on the descriptor. Read at bind into `ModelArchitecture.rms_epsilon` (I/qwen35.rs:683) and passed as a loose argument (I/qwen35.rs:619) | **G4** |
| model.context_length | kv:19 | `Architecture::trained_context_length` I/architecture.rs:493-496 | C |
| model.cache_strategy = state_cached | none | no such variant (descriptor.rs:14-37) | **G3** |
| layers.unit / repeat | kv:22 | `ModelDescriptor.layers` descriptor.rs:82 holds the expanded list; the only expansion rule is `Qwen35LayerKind::from_interval` (I/qwen35.rs:107-120) and the inline predicate (T/attention_forward.rs:3069) | G0 + **G1** + **G3** |
| layers: Gdn kind | kv:25-35 | `LayerKind { Attention, ShortConv }` T/single_range_moe_cached.rs:1452-1455; two more private enums exist: `Qwen35LayerKind` I/qwen35.rs:96-105 and `qwen35moe::hparams::LayerKind` I/qwen35moe/hparams.rs:13 | **G1** |
| attention.head_dim | kv:15 | `LayerAttentionConfig.head_dim` T/attention_forward.rs:412 (the bespoke builder names it `attn_head_dim`, `:2952`) | C |
| attention.kv_heads | kv:14 | attention_forward.rs:413 | C |
| attention.mask_window, value_source, key_source, rope_table, score_scale, value_norm | none | attention_forward.rs:416-444 | C |
| attention.rope_pairing | none; see U2 | `RopePairing` T/primitives.rs:574-577, hard-coded `Interleaved` at T/hyperconn_qwen35_dense.rs:702,704 | C (value disputed, U2) |
| attention.rotary_dim | kv:25 | passed in the slot named `head_dim` (T/attention_forward.rs:2951, :2970, :3248); generic builders compute `pairs = head_dim / 2` (:1464, :2386) | **G2** |
| attention.output_gate | none (derived from `attn_q` shape) | only inside `append_qwen35_dense_attention_only_with_taps` T/hyperconn_qwen35_dense.rs:630-653, :1078-1116 | **G2** |
| attention.qk_norm | tensors | `ModelDescriptor.qk_norm`, model-global, read only by the SingleRange arm (descriptor.rs:100-109, :410) | **G2** |
| gdn.* (6 fields) | kv:30-35 | `append_qwen35_ssm_mixer` arguments T/lfm2_qwen35_gdn.rs:1294-1321, `GdnOutputGate` :1069-1078, `v_head_reordered` :1590. No descriptor field | **G1** |
| ffn.combination, activation, ... | none | `LayerFfnConfig` T/attention_forward.rs:305-361; dense SwiGLU `append_dense_swiglu_ffn` :875-956 | C |
| head.output_norm | tensor | leaf T/attention_forward.rs:3527-3533 | C |
| head.tied_embeddings | directory | I/qwen35.rs:466-479 (alias bind); `ModelArchitecture.tied_embeddings = false` at :684 is a second, unread statement of the same fact | C (H5) |
| rope.base | kv:27 | `ModelArchitecture.rope_freq_base` I/qwen35.rs:682 -> `build_position_inputs` I/generate/residency_caches.rs:1206-1254 | C, outside descriptor (**G8**) |
| rope.rotary_dim | kv:25 | `build_position_inputs(head_dim = rotary)` residency_caches.rs:1209,1216 | C (name collision, **G2**) |
| rope.dimension_sections, mrope_sections, mrope_section, mrope_interleaved | kv:24,26,28,29 | none. `qwen35.rs` never reads them; `qwen35moe` parses two into hparams (I/qwen35moe/hparams.rs:46-47,128-137) and its own doc says unconsumed (I/qwen35moe/program.rs:25-31) | **G8** |
| rope.scaling, freq_factors | none | `RopeScaling` and `rope_freq_factors` (I/architecture.rs:411-429) | C (inert) |
| bind.alias (3 entries) | tensor names | hand-written in `bind_qwen35_weights` I/qwen35.rs:362-375, :431-437 | **G7** |
| layer.ffn.pre_norm_leaf | tensor name | none: the leaf name is fixed in each builder (T/attention_forward.rs:1819-1823, :3059-3064) | **G7** |
| bind.file_type | kv:10 | none needed; codec read per tensor (I/bind.rs:1244-1260) | C |
| cache.attention.{growth,rewind,placement} | none | `Qwen35DenseAttentionCache` residency_caches.rs:418-441 (no `truncate`); placement flags decode.rs:3242 | **G11, G12** |
| cache.gdn.{shapes} | derived | program-declared leaves, `declared_layer_cache_names_and_widths` I/generate/decode.rs:1452-1530 | C |
| cache.gdn.rewind, rollback_snapshots | none | none (`SsmLayerCache` has `advance` only, residency_caches.rs:598-621) | **G12, G13** |
| cache.*.placement | none | `device_kv_eligible` decode.rs:3552-3563 (all layers must be Attention or SharedFromLayer); ssm placement decode.rs:4406-4426 | **G11** |
| tokenizer.model | kv:55 | K/gguf.rs:78-112 | C |
| tokenizer.pre | kv:57 | none (`git grep 'ggml.pre'` over `*.rs` hits only doc comments) | **G9** |
| tokenizer.tokens, merges, token_type | kv:54,59,60 | K/gguf.rs:73-80,114-117 | C |
| tokenizer.scores | kv:58 | read only for model = "llama" (K/gguf.rs:102-106) | C (inert here) |
| tokenizer.eos_token_id, add_eos_token | kv:53,51 | K/gguf.rs:46,49,76,82,118 | C |
| tokenizer.bos/unknown/add_bos | absent | K/gguf.rs:45,47,48 (read as `None`) | C |
| tokenizer.padding_token_id, add_padding_token | kv:56,52 | none: K/gguf.rs:40-49 lists no padding key | **G10** |
| tokenizer.presplit.* | L/src/llama-vocab.cpp:392-396 | `pretokenize` K/pretokenize.rs:63 implements the LLAMA3 alternation only; `encode_ordinary` calls it unconditionally for byte-level vocabs (K/pipe.rs:75-80) | **G9** |
| template.chat_template | kv:50 | none (H2) | out of range |
| detokenize.* | none | `decode_streamed_piece` residency_caches.rs:3395-3409, `decode_until_stop_or_budget` :3433-3470 | C |
| sampling.* | none | `ServingConfig` I/serving.rs:791-829 -> `SamplingConfig` K/sample.rs:96-123 -> `select_decoded_token` I/generate/decode.rs:1319-1337 | C |
| stop.ids | kv:53 | `vocab.eos_token_id() == Some(id)` residency_caches.rs:3448 | C |
| placement.weights | none | `ServingConfig` gpu layers; `WeightClassBytes.ssm_state_bytes` I/memory_fit.rs:74-81 | C |
| serving.speculative | none | `speculative_verify_program` default `None` (I/architecture.rs:327-335) gates verify (decode.rs:3645) | C |
| serving.prefill_rows_per_evaluation | derived | `BoundProgram.single_position_step` set `true` by hand (I/qwen35.rs:709), split loop decode.rs:3460-3487 | **G6** |
| serving.prompt_cache.* | none | `PromptCacheConfig` serving.rs:472-557 | C, but inert for this model (G12, G13) |
| attention.read (H12) | none | `git grep -n -i 'attention_read\|attention\.read\|AttentionRead\|read_set\|ReadSet' main -- proxima-model-interop/src proxima-tensor/src proxima-core` returned 0 lines | **G12b** (recorded, no qwen35 edit) |
| kernel threshold (H7) | `omega/omega-runtime.toml` | `[gated_delta_net] head_k_dim_max = 256` (omega/omega-runtime.toml:266-277) vs head_k_dim 128; fusion recognised by op shape `gated_delta_net_candidates` T/bind/gdn_moe_fusion_apply.rs:528, behind the `gated-delta-net-fusion` feature (:527) | C |

### b2. the gap list (stage, missing input, smallest generic change)

**G0 [H4] (inherited, not counted).** `ModelDescriptor` derives `Debug, Clone, PartialEq` only
(`T/descriptor.rs:53`): no serde, no `Settings`, no `Validate`, so no config can be loaded. On
main no family-profile file exists (`profiles/` holds five build-tier TOMLs). Closed by
architecture-as-data R2/R9 and AC9, which already say "at HEAD the descriptor has no serde". See
section w for the in-flight profile work staged in this checkout.

**G1 [H4/H6] `LayerKind` has no Gdn.** Missing input: the six GDN parameters.
Change: `LayerKind::Gdn(GdnConfig)` with `GdnConfig { key_heads, key_head_dim, value_heads,
inner_size, conv_kernel, v_head_reordered, output_gate: GdnOutputGate }`, all `Copy`
(`GdnOutputGate` already exists, `T/lfm2_qwen35_gdn.rs:1070`). Delete the two private enums
(`I/qwen35.rs:96`, `I/qwen35moe/hparams.rs:13`) in favour of it. `LayerSchedule.attention` stays
unread for Gdn exactly as it is unread for ShortConv (`T/attention_forward.rs:451-453`).

**G2 [H6] the attention mixer cannot express qwen35's attention.** Missing inputs on
`LayerAttentionConfig`: `rotary_dim: Option<u32>` (pairs would be `rotary_dim / 2`, and a pass
plane exists iff `head_dim > rotary_dim`; today `pairs = head_dim / 2` at
`T/attention_forward.rs:1464` and `:2386`), `output_gate: Option<AttentionOutputGate>` (q-projection
twice as wide, sigmoid-gated output, `T/hyperconn_qwen35_dense.rs:630-653,1078-1116`), and a
per-layer `qk_norm` replacing the model-global SingleRange-only flag (`descriptor.rs:100-109`).
Change: three fields on one struct; `None`/`false` reproduce every existing caller.

**G3 [H6] no engine runs a state-cached recurrent layer.** `build_forward` dispatches to
`lfm2_two_range_cached...` (Attention only, `T/lfm2_single_range_cached.rs:2-3`),
`lfm2_forward_program_with_experts` (cacheless; its own doc says the conv-state-cached
counterpart "is still a further step", `T/attention_forward.rs:1695-1699`), and the Mistral
single-range builder (uniform attention only, `descriptor.rs:382-391`). Change: the loop at
`T/attention_forward.rs:3046-3525` becomes `CacheStrategy::StateCached`'s engine, with the
predicate at `:3069` replaced by `descriptor.layers[layer].kind`; the per-layer builders it calls are
already `pub` (`T/hyperconn_qwen35_dense.rs:433,583,1172`, `T/lfm2_qwen35_gdn.rs:1294`). Constraint:
the digest hashes `{index}:{op:?}` for every op in order (`arch_data_baseline.rs:142-148`), so leaf
and constant declaration order (`:2976-3042` global leaves; per layer `:3047-3064`, then
`:3078-3288` attention or `:3327-3398` Gdn) is part of the contract and the lifted loop must keep it.

**G4 [H6] `ModelDescriptor` has no `rms_epsilon`.** It is needed at lowering (the `head_eps`
constant, `T/attention_forward.rs:3025-3032`, and `eps` is a runtime leaf, `:2991`) and at step
time (`build_position_inputs`, `I/generate/residency_caches.rs:1210,1218`). Today it travels
through `Qwen35Architecture` -> `ModelArchitecture` -> a loose function argument.
Change: one `f32` field.

**G5 [H6] `BuildForwardProgram`'s third element is too narrow.** It is `Vec<CachedLayerRoots>`
(`descriptor.rs:253-261`); the runtime consumes `Vec<Qwen35LayerRoots>` (`I/architecture.rs:153`),
whose variants include `Ssm` and `DenseAttention` (`T/attention_forward.rs:2847-2867`). Two callers
already re-wrap it (`I/dense.rs:149-151`, `I/gemma4/bind.rs:705`): the compensator points at the
return type. Change: widen to `Vec<LayerRoots>` (the enum is already generic; only its name is
family-shaped).

**G6 [H6/H11] no width input, and `single_position_step` is a hand-set bool.**
`build_forward(descriptor, last_row_only)` (`descriptor.rs:299-302`) cannot pin the prefill width the
mixer's M>1 branch needs (`prefill_width`, `T/lfm2_qwen35_gdn.rs:1389,1948-2056`); only a
qwen35moe wrapper builds such a program (`I/generate/decode.rs:3486-3512`). The dense program is
always built at `prefill_width = None` (the mixer call at `T/attention_forward.rs:3400-3427` goes
through the wrapper that passes `None`, `T/lfm2_qwen35_gdn.rs:1352`), so qwen35 prefill is
one evaluation per prompt position (`I/qwen35.rs:709`, `decode.rs:3460-3487`). Change: a width
parameter beside `last_row_only`, and `single_position_step` derived as "schedule has a Gdn layer
and the program was not built at a pinned width" (the digest records it: `digest: single_position_step=true`).

**G7 [H5/H6] four leaf names disagree with the file.** Three program leaves are not file tensor
names: `ssm_in.weight` (file `attn_qkv.weight`), `ssm_gate.weight` (`attn_gate.weight`), `ssm_dt.bias`
(`ssm_dt`) (`I/qwen35.rs:362-375,431-437`). The fourth is the FFN pre-norm: the generic builders
declare `ffn_norm.weight` (`T/attention_forward.rs:1819-1823,2494-2498`), qwen35 declares
`post_attention_norm.weight` (`:3059-3064`), and that is also the file name. Trap:
`LayerFfnConfig.post_attention_norm = true` would add Gemma's sandwich norm (`:306-310`), not select
this tensor. Two changes with two sites, split by who must keep the name stable. The three ssm
leaves are named by the Gdn arm itself, so they need a binder alias (descriptor-carried table, program
name to file name). The FFN pre-norm leaf name must stay `post_attention_norm.weight` in the program,
because `Op::Input` debug text is part of the pinned digest (`arch_data_baseline.rs:142-148`), so it
needs a lowering-side `LayerFfnConfig.pre_norm_leaf` defaulting to `ffn_norm.weight`. Also hand-listed
today: which tensors take `bind_dense` versus `bind_matmul_weight_*` (`I/qwen35.rs:313-437`). On this
blob the choice has no visible effect (all 321 `bound:` entries are `packed`, and `bind_dense` keeps
Q4_K/Q5_K/Q6_K/Q8_0 packed, `I/bind.rs:1252-1259`); for another codec it is UNMEASURED, because I read
only `bind_dense`'s codec handling (`:1186-1189`), not `bind_matmul_weight_*`'s. The comment at
`I/qwen35.rs:387-392` (`ssm_out`/`ssm_conv1d` "stay on bind_dense", read as dense) is contradicted by
`bound:` (`blk.0.ssm_out.weight Q8_0 2228224`, listed as packed).

**G8 [H4/H6/step inputs] RoPE: no descriptor field.** Missing: `dimension_sections`,
`interleaved`, `rotary_dim`, `base`, scaling. Today the base is in `ModelArchitecture`
(`I/qwen35.rs:682`) and the table is `base^(-2i/d)` for every pair with no section logic
(`residency_caches.rs:1232-1238`); the sections are never read (U3). Change: one `RopeSpec`
field group on the descriptor, consumed by both lowering (pairs) and `build_position_inputs`;
`Validate`: `sum(sections) == rotary_dim / 2`.

**G9 [H3] `tokenizer.ggml.pre` is never read, and the pre-split is one hard-coded alternation.**
llama selects the QWEN35 regex from `pre = "qwen35"` (`L/src/llama-vocab.cpp:2266-2267`); it
differs from the LLAMA3 regex that `K/pretokenize.rs:7-15` implements (digit run capped at 3 by
`run.min(3)`, `:168-177`; letters by `char::is_alphabetic`, `:31-33`) in the digit clause
(`\p{N}` versus `\p{N}{1,3}`, `L/src/llama-vocab.cpp:392-396` versus `:284-290`) and in using
`[\p{L}\p{M}]` for letters. Evidence of effect: by reading, the string "2026" splits into
`2|0|2|6` under qwen35 and `202|6` under the implemented clause; **no run was made**, so the id-level
consequence is UNMEASURED. Change: two parameters on the existing scanner (`digit_group: u32`,
`letters_include_marks: bool`) populated from `pre`, not a regex engine and not a new type.

**G10 [H3] padding keys unread.** `padding_token_id` and `add_padding_token` have no reader
(`K/gguf.rs:40-49`). Inert for generation. Change: `Vocab` carries `padding_token_id:
Option<u32>`.

**G11 [H8] placement is a predicate on cache kinds plus flags.** `device_kv_eligible` requires every
layer to be `Attention` or `SharedFromLayer` (`I/generate/decode.rs:3552-3563`), so a hybrid keeps its KV on
the host; dense-attention and ssm placement are separate switches (`decode.rs:3242,4406-4426`).
The load-time budget reads `ssm_state_bytes` from a second derivation, `qwen35_ssm_state_bytes`,
which multiplies one layer's elements by `block_count` (`I/qwen35.rs:595-598`) rather than by the
number of Gdn layers; for this checkpoint that is 24 x 1,122,304 = 26,935,296 B against 18 x 1,122,304
= 20,201,472 B, a 6,733,824 B difference (DERIVED; the gate is `metal`-only, `pregather.rs:2715-2720`).
The runtime sizes the real caches from the program's declared leaves instead (`decode.rs:1541-1548`).
Change: `placement` per cache kind as an enum field, and `ssm_state_bytes` derived from the lowered
program's declared leaves, the same source the caches use.

**G12 [H11/H13] "can this layer rewind?" is answered in three places, each with a bool or `None`.**
`rings_cover_speculation` returns `true` for every non-ring layer (`I/generate/kv_ring.rs:278-283`),
which answers `true` for a Gdn layer, a question that predicate was not written to ask; `RingCheckpoint::capture` returns `None` for any Ssm or
DenseAttention layer (`ring_checkpoint.rs:75`); `rewind_refusal` returns `UnrewindableLayer` for
anything that is not a plain or ring attention cache (`prompt_cache.rs:267`), which includes
qwen35's own rewindable 4-wide attention layers because `Qwen35DenseAttentionCache` has `append`
but no `truncate` (`residency_caches.rs:418-441`). Information destroyed: the real answer is
three-valued. The incumbent keeps it so: `common_context_seq_rm_type` is `NO / PART / FULL / RS`
(`L/common/common.h:1006-1011`), where RS means "partial removal bounded by `n_rs_seq`". Change: one
per-layer capability derived from the layer kind, `Rewind::{Truncate, Snapshot { bound }, None}`,
read by all three sites, plus `Qwen35DenseAttentionCache::truncate`.
**G12b [H12]:** `attention.read` has no consumer on main; qwen35 needs only the default (dense) and
the rule that a Gdn layer exposes no read set. No qwen35 edit.

**G13 [H13] the snapshot hook is ring-only.** `RingCheckpoint` stores per-ring-layer rows
(`ring_checkpoint.rs:26-39`). Change: `LayerSnapshot::{Rows, Ssm { conv_history, state }}` with
`capture` handling every kind; `PromptCacheConfig.checkpoint_interval` and `max_checkpoints`
(`serving.rs:492-503`) then apply unchanged, and one checkpoint costs 20,201,472 B on this model
(DERIVED), against the 12 MiB the field's doc quotes for gemma4-E2B.

**G14 [H11] Verify reachability and per-row state.** `Verify` is reachable only when a verify
program exists (`I/architecture.rs:311-335`, `decode.rs:3645`), and the FSM's `accept` consumes one
`Cache` per row (`serving_fsm.rs:186-190`). For a hybrid, per-row Gdn state exists only in the M>1
branch (`SsmMixerTaps.per_position_state_out`, `T/lfm2_qwen35_gdn.rs:1099-1105`), which needs G6's
width-pinned program. Change: reachable-states rule from config (section d, rule 3).

Counting rule for N: edits to one struct or one read-site are one change, so G9+G10 (both are
tokenizer-key readers) and G13+G14 (the snapshot type and the reachability rule that reads it) each
pair into one row, and G12b needs no qwen35 edit. That gives 12 changes (section e).

## c. can the one generic `build_forward` produce qwen3.5's program?

Today: no. `build_forward` is not one builder; it is a `match` over `cache_strategy` that calls
three whole-model builders (`descriptor.rs:303-426`). None of them can emit a Gdn layer or a
partial-rotary gated attention layer. The structures of the bespoke builder with no descriptor
expression, exactly (a bare `:N` in this table is `T/attention_forward.rs:N`):

| # | structure | bespoke location | descriptor today | gap |
|---|---|---|---|---|
| S1 | Gdn layer kind and its 6 parameters | T/attention_forward.rs:3326-3427 | `LayerKind` = {Attention, ShortConv} (T/single_range_moe_cached.rs:1452-1455) | G1 |
| S2 | partial rotary (`rotary_dim`, pass plane) | :2970, :3248, :3269-3278 | `pairs = head_dim / 2` (:1464, :2386) | G2 |
| S3 | gated q projection (2x width, sigmoid output gate) | :3078-3120; T/hyperconn_qwen35_dense.rs:630-653,1078-1116 | none | G2 |
| S4 | per-layer qk_norm | :3236-3247 | model-global flag, SingleRange only (descriptor.rs:109) | G2 |
| S5 | a 4-plane KV cache and a `cached_len` padding mask in the attention layer | T/hyperconn_qwen35_dense.rs:814-856 | engines mask differently (descriptor.rs:18-36) | G3 |
| S6 | a recurrent state cache (`ssm_cache.{l}.conv_history/state`) threaded as inputs and returned roots | :3346-3398, :3514-3520 | no recurrent state in any engine (:1695-1699) | G3 |
| S7 | schedule from `full_attention_interval` instead of `layers` | :3069 | `layers: Vec<LayerSchedule>` exists (descriptor.rs:82) but this loop ignores it | G3 |
| S8 | epsilon baked as a constant and bound as a leaf | :3025-3032, :2991 | no field | G4 |
| S9 | return type `Vec<Qwen35LayerRoots>` | :2962 | `Vec<CachedLayerRoots>` | G5 |
| S10 | prefill width for the mixer's M>1 branch | T/lfm2_qwen35_gdn.rs:1389 | no parameter | G6 |
| S11 | FFN pre-norm leaf named `post_attention_norm.weight`; 3 ssm leaves whose names differ from the file | :3059-3064; I/qwen35.rs:366,373,435 | leaf names fixed in the builders (:1819-1823) | G7 |
| S12 | RoPE with sections and an interleave flag | hyperconn_qwen35_dense.rs:698-704; residency_caches.rs:1227-1238 | `RopeTableSel` names two leaves only (attention_forward.rs:99-103) | G8 |

Input coverage: the bespoke builder takes 16 positional arguments (`T/attention_forward.rs:2946-2961`).
After the changes in section e each has a home in the configuration of section a, and none is left over:

| argument | config field |
|---|---|
| `vocab`, `embedding`, `feed_forward`, `query_heads`, `block_count` | `model.*` (all exist on `ModelDescriptor`) |
| `kv_heads` | `layer.attention.kv_heads` (exists) |
| `attn_head_dim` | `layer.attention.head_dim` (exists) |
| `head_dim`, which is the rotary width | `layer.attention.rotary_dim` (G2) |
| `full_attention_interval` | `layers.unit` / `layers.repeat` (G3: the loop reads the schedule) |
| `ssm_d_state`, `ssm_dt_rank`, `ssm_n_group`, `ssm_d_inner`, `ssm_d_conv` | `layer.gdn.key_head_dim`, `value_heads`, `key_heads`, `inner_size`, `conv_kernel` (G1) |
| `rms_eps` | `model.rms_epsilon` (G4) |
| `last_row_only` | per-call parameter, as `build_forward` already treats it (`descriptor.rs:276-280`) |

Expressible today, no gap: the dense SwiGLU FFN (`LayerFfnConfig::exclusive()` +
`append_dense_swiglu_ffn`, `:875-956`, same op chain as the two inline copies at `:3436-3506` and
`hyperconn_qwen35_dense.rs:1233-1324`), embedding lookup, output norm, last-row gather (`:532-544`),
the lm head, and the GDN fused kernel selection (H7, shape-recognised). Whether the generic loop
reproduces the bespoke op order for those, and so the digest `bind.ops=4389`, `logits_root=NodeId(4388)`,
`layer_roots=24` (`digest:`), is UNMEASURED: the inline FFN declares its three weight leaves after the
rmsnorm (`:3436-3454`) while `append_dense_swiglu_ffn` declares them first (`:884-901`), so an
order-preserving lift is a requirement of G3, not a given.

## d. H11 and H13: recurrent state versus KV rows

Can they share a hook? **H11: yes.** `ServingState<Cache>` is generic and already pairs a
`snapshot` with the live `cache` (`serving_fsm.rs:48-74`), requires only `Cache: Clone`
(`:158-161`), and accepts one cache per verified row (`:186-190`). That is the recurrent protocol;
KV rows are the case where the snapshot is unnecessary. The FSM is not driven by the live loop today
(`#![allow(dead_code)]`, `serving_fsm.rs:36-38`); the live speculative branch rewinds KV with
`LayerCache::truncate`. **H13: one hook, two granularities.** The
existing snapshot machinery (`RingCheckpoint`) is a per-layer, per-position snapshot; a KV block is
sealed by position range, a recurrent state only by a whole-state snapshot at one position.

The incumbent draws the same line: a recurrent layer "can't have a state partially erased at the end
of the sequence because their state isn't preserved for previous tokens"
(`L/src/llama-memory-recurrent.cpp:182-183`), and partial rollback works only through per-token
snapshots bounded by `n_rs_seq`, single-use (`:193-201`, `L/include/llama.h:370`). llama decides which
architectures qualify by name (`llm_arch_supports_rs_rollback`, `L/src/llama-arch.cpp:1114-1128`,
QWEN35 listed); here it must come from the layer kinds in the schedule.

Decision rules that differ (KV row layer / DenseAttention vs Gdn layer):

1. **Rewind unit.** KV: shrink `cached_len`; rows are positional (`LayerCache::truncate`,
   `residency_caches.rs:155-162`; a ring only moves `cached_len`, `:156-158`). Gdn: no rows. State
   at n is a function of every token up to n, so rewind is restoring a state stored at some k <= n.
2. **What `enter_verify` snapshots.** Today `snapshot: cache.clone()` clones the whole cache
   (`serving_fsm.rs:165`), which for KV is O(cached_len x row) for no benefit. Rule: snapshot only
   layers whose capability is `Snapshot`; `Truncate` layers record nothing. For this model that is
   18 x 1,122,304 = 20,201,472 B (DERIVED) per verify step, flat in context length, versus 24,576 B per
   position of KV (DERIVED): the two cost the same at 822 positions.
3. **Verify reachability (R2 "reachable from config").** `Verify` is reachable iff the speculative
   config is non-empty, a verify program exists (G6: width-pinned), and for every layer `Rewind != None`
   and, for `Snapshot { bound }`, `bound >= draft_limit + 1`. With `rollback_snapshots = 0` (the
   incumbent's default, `L/src/llama-context.cpp:3718`) a hybrid never enters `Verify`; today that
   outcome is reached by accident of `speculative_verify_program() == None`, and `rings_cover_speculation`
   would answer `true` for it (`kv_ring.rs:282`).
4. **Accept with n of K drafts.** KV: truncate to the `accepted + 1` positions that survive
   (`residency_caches.rs:150-154`). Gdn: select the state after row
   n, either `per_position_state_out[n]` from the verify program (`T/lfm2_qwen35_gdn.rs:1105`; K copies
   of the state, which is why the bound exists) or restore the snapshot and replay n + 1 single-row
   steps. `serving_fsm.rs:186-190` already takes the first form ("no replay").
5. **Device-resident state cannot rewind by parity.** The placed state alternates two buffers by
   `cached_len.is_multiple_of(2)` (`decode.rs:4420-4424`), which is correct only for +1 stepping. A
   rollback of k > 1 needs k + 1 buffers or the host path.
6. **Prefix reuse (H9/H13).** KV: any prefix of an entry, by LCP (`PrefixState::rewind_to`,
   `prompt_cache.rs:302-329`). Gdn: only at a stored checkpoint position <= LCP, then prefill the
   rest (the shape `ring_checkpoint.rs:10-15` attributes to llama-server); an extension (`target_len ==
   cached_len`) is always legal (`prompt_cache.rs:1892-1909`). Today qwen35 gets `UnrewindableLayer`
   for every rewind because the catch-all at `prompt_cache.rs:267` also refuses its 6 rewindable
   attention layers.
7. **Seal and tier (H13).** KV: seal by block `[i*B, (i+1)*B)`, demote oldest first, partial demotion
   legal (`PromptCacheConfig.block_tokens`, `serving.rs:545-550`). Gdn: seal is one snapshot per
   checkpoint position, all layers at once; demoting it makes every position past it unrecoverable
   without replay from the previous checkpoint. Chunk shift and blend (H9) are KV-only: Ssm and
   DenseAttention report `rows_are_movable() == false` (`chunk_shift.rs:287-291,1113-1120`).
8. **H12 read set.** KV layers: a subset of rows. A Gdn layer reads its state whole, so `attention.read`
   applies to `Attention` layers only; `Validate` rejects a non-dense read on a Gdn layer.
9. **Prefill shape (H11 `Prefill`).** KV: M rows in one pass. Gdn without a width-pinned program: 1 row
   per evaluation (G6), so `Prefill` is a loop of `Decode`-shaped evaluations until the pinned program exists.

No new pipe is required for any rule: each is a capability enum (G12), a snapshot enum (G13) and a
reachability predicate over config (G14).

## e. verdict field, and what is not a pipe

```
verdict = "fits after 12 named hook changes"
```

The 12 (each one edit to an existing structure; none a new pipe):

| # | change | hook |
|---|---|---|
| 1 | `LayerKind::Gdn(GdnConfig)`, delete the two private layer enums (G1) | H4/H6 |
| 2 | `LayerAttentionConfig` gains `rotary_dim`, `output_gate`, per-layer `qk_norm` (G2) | H6 |
| 3 | schedule-driven `StateCached` engine lifted from `attention_forward.rs:3046-3525` (G3) | H6 |
| 4 | `ModelDescriptor.rms_epsilon` (G4) | H6 |
| 5 | `BuildForwardProgram` roots widened to `Vec<LayerRoots>` (G5) | H6 |
| 6 | width parameter on `build_forward`; derive `single_position_step` (G6) | H6/H11 |
| 7 | binder alias table for the 3 ssm leaves, and `LayerFfnConfig.pre_norm_leaf` (G7) | H5/H6 |
| 8 | `RopeSpec` on the descriptor, consumed by lowering and `build_position_inputs` (G8) | H4/H6 |
| 9 | `tokenizer.ggml.pre` read; `digit_group`, `letters_include_marks` on the scanner (G9), and `padding_token_id` on `Vocab` (G10) | H3 |
| 10 | per-kind placement enum; `ssm_state_bytes` from the lowered program (G11) | H8 |
| 11 | per-layer `Rewind` capability read by the three sites, `Qwen35DenseAttentionCache::truncate` (G12) | H11/H13 |
| 12 | `LayerSnapshot` replacing `RingCheckpoint`'s row-only capture; `Verify` reachability and recurrent-only snapshot rule (G13, G14) | H11/H13 |

Counting rule: edits to one struct or one site are one change, so G9+G10 and G13+G14 each pair
into one row. The count is a function of that rule, not a measurement.

Evidence that it is not "does not fit": every change above is a field, an enum arm, or a
read-site, and the mixer and attention layer already exist as `pub` per-layer builders taking the
parameters those fields would hold (`T/lfm2_qwen35_gdn.rs:1294-1321`,
`T/hyperconn_qwen35_dense.rs:433-460`). Evidence that it is not "zero core edits": G1, G3 and G5
alone change `descriptor.rs`, `single_range_moe_cached.rs` and `attention_forward.rs`.

What is not a pipe, and why that is justified:

- `build_forward` and the layer loop: a pure function from descriptor to op list, sans-IO by R9, so
  `Pipe`'s async shape would add nothing.
- `ServingState` and its reachability rule: a discriminated-enum FSM (P11), the owner's stated shape.
- `GdnConfig`, `RopeSpec`, `Rewind`, `LayerSnapshot`: data. Each passes the second gate: a caller can
  now schedule a GDN layer, load a RoPE with sections, and checkpoint a hybrid; none of those had a call
  site before.
- the tokenizer pre-split: an existing scanner with two parameters; the tokenizer's own `encode` is
  unchanged.

Design abandoned: (1) `CacheStrategy::Qwen35` forwarding to the bespoke function (fails R9); (2)
cloning the whole `Cache` on `enter_verify` for every layer (rule 2); (3) a trait object per mixer
kind (P20, R2); (4) a regex engine for the pre-split (the six clauses llama composes need two
parameters here, and the crate is `no_std + alloc`).

## u. unexplained, and contradictions in the code read

- **U1.** The GGUF says `ssm.v_head_reordered = True` (`kv:35`). The dense qwen35 program passes
  `false`: `append_qwen35_ssm_mixer` forwards `false` (`T/lfm2_qwen35_gdn.rs:1418`) and
  `qwen35_forward_program_with_last_row` calls that wrapper (`T/attention_forward.rs:3400-3427`).
  qwen35moe passes the real value (`I/qwen35moe/program.rs:720`). For this checkpoint
  `group = 16 / 16 = 1`, so the two index maps (`:1805-1810` versus `:1828-1831`, `:2186-2200`)
  coincide numerically by inspection (not run), but they are textually different ops, so by reading a
  config that faithfully says `true` lowers different op text from the one the pinned digest hashes. Which value the AC0 digest should
  carry is the owner's call; I carried the GGUF value.
- **U2.** `T/hyperconn_qwen35_dense.rs:387-390` says qwen35 RoPE is split-half (NEOX); the same
  function applies `RopePairing::Interleaved` (`:698-704`, comment "Qwen3.5 uses interleaved MRoPE").
  The GGUF key is `mrope_interleaved = True` (`kv:28`), a different notion (interleaving of
  section axes) from pair layout. There is no oracle: llama f1ea20621 refuses these blobs
  (architecture-as-data, finding O1).
- **U3.** `build_position_inputs` says each MRoPE section "restarts its local frequency index"
  (`residency_caches.rs:1227-1230`); the code computes one flat `base^(-2i/d)` schedule
  (`:1232-1238`) and never reads the sections. The text-only argument for a flat table is at
  `hyperconn_qwen35_dense.rs:391-401`.
- **U4.** The fixture header says `Dumping 55 key/value pair(s)` and `GGUF.kv_count = 52`
  (`kv:5,8`); the dump lists 55 keys. Not explained.
- **U5.** I did not execute anything, so no claim here is a measurement. In particular: digest
  equality after the lift (G3), the id-level effect of the pre-split difference (G9), and the
  18-versus-24-layer state-bytes difference at runtime (G11) are all UNMEASURED.

## w. in-flight work staged in this checkout (index only, not main), checked against this sketch

`git status` on the shared checkout shows staged additions under `proxima-model-interop/src/profiles/`
(8 family strings: gemma4, llama, mistral, mixtral, qwen2, qwen3, qwen3moe, qwen3_moe; none for qwen35 or any
recurrent family) and a staged `FamilyProfile` in `proxima-tensor/src/spec/descriptor.rs`
(`git show :proxima-tensor/src/spec/descriptor.rs`, lines 144-205). Read, not built. Two points that
bear on this sketch:

- `FamilyProfile` has six fields (`embedding_scale`, `ffn`, `parallel_dense_moe`,
  `score_scale_inverse_sqrt_head_dim`, `value_norm`, `rope_split_half`): no field for a recurrent layer,
  for a rotary width, for an output gate, or for RoPE sections. G1, G2 and G8 are therefore not covered by it.
- Its `rope_pairing(head_dim, qk_norm_tensors)` returns `SplitHalf { pairs: head_dim / 2 }` whenever
  `rope_split_half || qk_norm_tensors` (staged `descriptor.rs:194-203`). qwen35 has QK-norm tensors
  (`bound:` lists `blk.3.attn_q_norm.weight`), so that rule selects split-half, while the bespoke layer
  applies `Interleaved` (`T/hyperconn_qwen35_dense.rs:698-704`; see U2). It also takes `pairs` from
  `head_dim`, which for qwen35 is 256 rather than the rotary 64 (G2). A profile for qwen35 under the
  staged rule would lower a different RoPE than the program the digest pins, so `rope_pairing` needs to be a
  stated field in the profile, not inferred from tensor presence.
