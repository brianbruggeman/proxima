# sketch 1: gemma4 E2B entirely from configuration (H2-H8, H14-H16)

status: paper test. Read at main 4b4be6cf (`git show main:<path>`). Nothing was built or run; every
"hand-derived" value is derived from cited lines, not executed. The E2B facts come from the vendored
fixtures `proxima-model-interop/tests/fixtures/llama-parity/gemma4_e2b*` ("fixture kv", `.digest`,
`.bound`, `swa_layers.txt`). Paths relative to the proxima repo root; `interop` =
`proxima-model-interop/src`, `tensor` = `proxima-tensor/src`, `tok` = `proxima-tokenizer/src`.

This sketch is the integration: it walks the whole request-to-response path for one real checkpoint and
points at the single-stage sketches (4 template, 5 pre-split, 6 sampler, 7 speculation) for the stages
they own, instead of repeating them. Its own new ground is H4 through H8, plus H16.

## shape chosen, and the contested decision

Shape: one TOML document, one section per stage, every section a plain configured record or ordered list.
The checkpoint's GGUF is a data layer under it (`"gguf:<key>"` references), so per-layer facts (which
layers slide, which share KV, per-layer FFN width) are never written out by hand.

Contested decision: where do the gemma4-specific constants live (unscaled attention score, value norm,
GeLU-tanh, sqrt embedding scale, split-half RoPE, the two table names, the four norms per layer)? Today
they are literals inside `gemma4_descriptor_from_gguf` (`tensor/spec/gguf_descriptor.rs:12-14`: "literals
here until they move to a family profile"; :53-86, :139-148, :152-176). Chosen: a `[model.profile]` record
that carries exactly those literals as data, interpreted by the same generic function. The family name
is then a label in a TOML file, not a code path. A second profile for another family is a new file.
Not a type: the call site before and after is `gemma4_descriptor_from_gguf(parsed, ring)` versus
`descriptor_from_gguf(parsed, &profile, ring)`; those differ (the profile is an input), so this is a
parameter, not a relocation.

## 1. ground: what a user does today to run gemma4 E2B

There is no configuration file. A gemma4 run is a Rust value plus code paths, stage by stage:

- H2 template: none (sketch 4). The gate test hand-frames `<|turn>user\n..<turn|>\n<|turn>model\n`
  (`proxima-model-interop/tests/gemma4_correctness_gate.rs:81-83`).
- H3 tokenize: `vocab_from_metadata` reads `tokenizer.ggml.model = 'gemma4'` (fixture kv :32), builds a
  merges-driven vocab and probes it char-level (`tok/gguf.rs:84-101`, `tok/vocab.rs:474-482`); the splitter
  is `pretokenize_newline_runs` (`tok/pipe.rs:72-74, 85-92`). `tokenizer.ggml.add_space_prefix = False`
  (kv :43) is never read (git grep `add_space_prefix` over `tok/`, `interop/src` finds nothing); the
  no-prefix behaviour is baked into `encode_char_level` (`pipe.rs:83-84`).
- H4 describe: `gemma4_descriptor_from_gguf(parsed, sliding_kv_ring)` (`gguf_descriptor.rs:29-150`). Keys
  read: `embedding_length`, `block_count`, `attention.head_count_kv`, `attention.sliding_window_pattern`,
  `feed_forward_length`, `expert_count`, `embedding_length_per_layer_input`, `attention.shared_kv_layers`,
  `attention.key_length`, `attention.key_length_swa`, `attention.sliding_window`,
  `final_logit_softcapping` (:36-50). The same keys are read AGAIN by `gemma4::hparams::from_metadata`
  (`interop/gemma4/hparams.rs:81-150`), which feeds `ModelArchitecture` and the weight bind
  (`interop/gemma4/bind.rs:669-741`). `bind_gemma4_with_last_row_only` calls both in one function
  (:669 and :675).
- H5 bind: `gemma4_tensor_names` (`bind.rs:62-130`) is a literal suffix list per layer, gated by
  `is_shared_kv`, `is_moe`, `is_sliding`, `ple_dim`; `bind_gemma4_weights` (:218) binds the same names with
  its own gates; the descriptor gates `ValueSourceKind` a third time (`gguf_descriptor.rs:72-77`). The
  repo's own test doc records that this once disagreed: "`bind` and the forward program each independently
  gate ... and nothing forced the two gates to agree" (`bind.rs:931-948`). Precision recode is
  `ServingConfig::weight_precision: &'model [WeightPrecisionRule]`, first match wins (`interop/serving.rs:
  92-134`); no serde.
- H6 lower: `build_forward(&descriptor, last_row_only)` dispatches on `CacheStrategy` (`tensor/spec/
  descriptor.rs:299-330`); gemma4 is `TwoRange` when every layer is `LayerKind::Attention`
  (`gguf_descriptor.rs:120-124`). The KV layout is not data: `KvLayout::SlidingRing` is a literal at the load
  entry points (`interop/generate/pregather.rs:2447, 2520, 2552`), and the descriptor takes it as the
  caller's `sliding_kv_ring` argument ("the caller's KV layout choice, not checkpoint data",
  `gguf_descriptor.rs:18-20`). `PROXIMA_HEAD_REPEATS` is read inside the graph builder
  (`tensor/spec/attention_forward.rs:2121`, `std::env::var`).
- H7 specialize: the kernel-selection thresholds are BUILD-time constants. `omega/omega-runtime.toml`
  opens "build-time execution-policy sizing ... read by build.rs, emitted as consts" and each key can be
  overridden by an `OMEGA_<SECTION>_<KEY>` env var at `cargo build` (lines 1-9). `tiled_gemm.min_tokens =
  160` is documented as measured on "gemma4-E2B Q4_0" (the key's own comment).
- H8 place: `gpu_layers: i32`, default `GPU_LAYERS_ALL` only when the crate is built with the `metal`
  feature, else `0` (`serving.rs:145-149`); `kv_offload` is rejected when true by admission (`interop/generate/prompt_cache_key.rs:119-120`
  comment).
- H14 sample: greedy by default (`serving.rs:1153-1159`). The GGUF carries `general.sampling.top_k = 64`,
  `top_p = 0.95`, `temp = 1.0` (fixture kv :6-8); no code reads `general.sampling.*` (git grep over
  `interop/src`, `tok/`, `proxima-gguf/src` returns nothing).
- H15 stop: one eos id (kv :38, `eos_token_id = 1`) or budget (`interop/generate/residency_caches.rs:3448`).
  The turn terminator `<turn|>` is id 106 and does not stop (sketch 4, GAP-4).
- H16 detokenize: `decode_streamed_piece` (`residency_caches.rs:3395-3409`): UTF-8 hold, then `▁` to space
  for unigram or char-level vocabs. Control-type tokens get empty text by an unconditional branch
  (:3449-3458).

## 2. the config a user would write

Real data, from the fixtures: 35 blocks, 20 shared-KV layers (so layers 15 to 34 own no KV), `head_count =
8`, `head_count_kv = 1` (a scalar, broadcast to every layer, `gguf_descriptor.rs:236`), `key_length = 512`,
`key_length_swa = 256`, `sliding_window = 512`, `final_logit_softcapping = 30.0`,
`embedding_length_per_layer_input = 256` (kv :12-30). Oracle full-attention layers (`is_swa = 0`): 4, 9,
14, 19, 24, 29, 34 (`swa_layers.txt`).

```toml
[template]
source = "turns"
bos = "tokenizer"
generation_prompt = "<|turn>model\n"

[template.turn.user]
prefix = "<|turn>user\n"
suffix = "<turn|>\n"
trim = true

[template.turn.assistant]
prefix = "<|turn>model\n"
suffix = "<turn|>\n"
trim = true

[tokenizer.pre]
source = "gguf"
missing = "shape"

[tokenizer.pre.rule.gemma4]
names = ["gemma4"]
kind = "newline_runs"

[model.profile]
family = "gemma4"
embedding_scale = "sqrt"
final_logit_softcap = "gguf:final_logit_softcapping"
cache = "two_range"

[model.ffn]
activation = "gelu_tanh"
post_attention_norm = true
output_scale = true
dense_post_norm = true
gating = "softmax"
expert_bias = false
ple_dim = "gguf:embedding_length_per_layer_input"

[model.layers]
kind_pattern = "gguf:attention.sliding_window_pattern"
kv_heads = "gguf:attention.head_count_kv"
dense_ffn_width = "gguf:feed_forward_length"
shared_kv_layers = "gguf:attention.shared_kv_layers"
shared_source = "last_own_kv_of_same_kind"

[model.attention.sliding]
head_dim = "gguf:attention.key_length_swa"
window = "gguf:attention.sliding_window"
rope = "sliding"
pairing = "split_half"
score_scale = "unscaled"
value_norm = true

[model.attention.full]
head_dim = "gguf:attention.key_length"
window = 0
rope = "full"
pairing = "split_half"
score_scale = "unscaled"
value_norm = true

[model.rope.full]
base = "gguf:rope.freq_base"
dims = "gguf:rope.dimension_count"
freq_factors = "tensor:rope_freqs.weight"

[model.rope.sliding]
base = "gguf:rope.freq_base_swa"
dims = "gguf:rope.dimension_count_swa"

[bind]
precision = []

[lower]
kv_layout = "sliding_ring"

[specialize]
source = "build"

[placement]
gpu_layers = "all"

[sampling]
source = "default"

[stop]
extra_ids = []

[detokenize]
control = "hide"
```

Defaults equal today: omitting a section leaves today's behaviour (greedy sampling, eos-only stop,
control tokens hidden, all layers on the GPU in a metal build, `precision = []` meaning "bind at the codec
stored on disk"). Every value above that is not `gguf:` or `tensor:` is the literal that
`gemma4_descriptor_from_gguf` carries today (:53-86 for attention, :139-148 for the model-global fields,
:152-176 for the FFN).

## 3. field map: every key to a consumer, or a gap

| key | consumer | status |
|---|---|---|
| `template.*` | none (sketch 4) | GAP (sketch 4: stage, segments, end-of-turn id) |
| `tokenizer.pre.*` | none: `pre` unread; gemma4 has no `pre` key (kv dump lists none); the splitter is shape-probed (sketch 5) | GAP (sketch 5) |
| `model.profile.embedding_scale/cache` | `Some(EmbeddingScale::Sqrt)` and the `cache_strategy` expression (`gguf_descriptor.rs:120-139`) | literal in code: GAP-2 |
| `model.profile.final_logit_softcap` | `metadata_f32_optional(.., "final_logit_softcapping", 0.0)` then `(softcap > 0.0).then_some` (:50, :140) | EXISTS as code, not data |
| `model.ffn.*` | `gemma4_ffn` (:152-176): literals | GAP-2 |
| `model.ffn.ple_dim` | `ple_dim`, `ple: ple_dim > 0` (:45, :143, :174) | EXISTS as code |
| `model.layers.*` | `sliding_window_pattern`, per-layer `head_count_kv`/`feed_forward_length`, `shared_kv_source_layer` (:38-48, :88-118, :182-188) | EXISTS as code; rule name `last_own_kv_of_same_kind` is today's function |
| `model.attention.sliding/full.*` | `LayerAttentionConfig` literals (:53-86) | GAP-2 |
| `model.rope.*` | `ModelArchitecture::sliding_rope`, `rope_freq_factors`, `step_inputs` (`bind.rs:737-740, 873-928`) | family override in code: GAP-2 |
| the whole `[model]` as TOML | `ModelDescriptor`, `LayerSchedule`, `LayerAttentionConfig`, `LayerFfnConfig`: no serde (`descriptor.rs:53`; `attention_forward.rs:304-305, 410-411, 454-455`); `RopeTableSel` holds `&'static str` (:100-103) | GAP-1 |
| `bind.precision` | `weight_precision` borrowed slice, no serde (`serving.rs:107-110`) | GAP-6 |
| `lower.kv_layout` | `KvLayout::SlidingRing` literal (`pregather.rs:2447, 2520, 2552`) | GAP-4 |
| `specialize.*` | build-time consts (`omega/omega-runtime.toml:1-9`) | GAP-5 |
| `placement.gpu_layers` | `ServingConfig::gpu_layers` (`serving.rs:145-149`) | EXISTS; loader GAP-6 |
| `sampling.source` | none for `general.sampling.*` | GAP-7 (rest: sketch 6) |
| `stop.extra_ids` | none (sketch 4, GAP-4) | GAP (sketch 4) |
| `detokenize.control` | unconditional `is_control` branch (`residency_caches.rs:3453-3458`) | GAP-8 |

## 4. can each decision be a pure function in core plus a pipe?

Decision halves, by stage:
- H4 describe: yes. `gemma4_descriptor_from_gguf` is a pure function of `(&ParsedGguf, bool)`; it needs
  only the profile as an extra input. Lives in `proxima-tensor`, which is `no_std`-capable
  (`tensor/spec/gguf_descriptor.rs` imports `alloc`).
- H5 bind: partly. Name resolution (which tensors does layer `i` have) is a pure function of the
  descriptor; today it is written three times (section 1). Codec choice is the first-match rule list
  (`serving.rs:121-134`), already pure.
- H6 lower: yes; `build_forward` is plain data to op graph, "no async/`Future`/`Box<dyn>` anywhere"
  (`descriptor.rs:263-298`).
- H7: the decision (a classifier per op shape, `omega/src/msl/emit_and_classify.rs`) is pure but its
  constants are baked at build.
- H8: the placement decision for a dense model is "all on device"; nothing to decide.
- H14-H16: sketches 6, 4, and GAP-8.

Placement half: for each of these, the second gate, call site both ways, answers "no new type". H4:
`gemma4_descriptor_from_gguf(parsed, ring)` versus `descriptor_from_gguf(parsed, &profile, ring)`: a
parameter. H5: `gemma4_tensor_names(&architecture)` versus `tensor_names(&descriptor)`: one function
replacing three, no wrapper. H6: nothing changes. No `Pipe` is introduced anywhere in this sketch; the
SPEC's "pipe slot" for these stages is a configured record read by an existing pure function. Where the
SPEC says "pipe", this sketch finds a function (H4, H5, H6) or a build-time constant (H7).

The one function needed beyond the sketches already written is the profile interpreter's core, shown as
a signature because the body is the existing 120 lines with literals replaced by reads:

```rust
pub fn descriptor_from_gguf(
    parsed: &ParsedGguf,
    profile: &ModelProfile,
    sliding_kv_ring: bool,
) -> Result<ModelDescriptor, TensorError>
```

## 5. worked example (doubles as the test)

Derive the E2B layer schedule from the fixtures and the cited code, by hand.

Inputs: `block_count = 35`, `shared_kv_layers = 20` (kv :12, :25), `first_shared_idx = 35 - 20 = 15`
(`gguf_descriptor.rs:88`). Sliding pattern: full layers are 4, 9, 14, 19, 24, 29, 34 (`swa_layers.txt`;
the repo's own doc names the first three as the full own-KV layers, `bind.rs:44-48`).

Own-KV layers 0 to 14: 12 sliding (head_dim 256, window 512, table `rope_cos_swa`, `value_norm = true`,
`ProjectedV`) and 3 full (4, 9, 14: head_dim 512, no window, table `rope_cos`, `ProjectedV` because
`shared_kv_layers > 0`, `gguf_descriptor.rs:72-77`). Shared layers 15 to 34 (20 layers): the source is
the last own-KV layer of the same kind before 15 (`shared_kv_source_layer`, :182-188): a sliding layer
takes layer 13 (14 is full), a full layer (19, 24, 29, 34) takes layer 14; both use
`SharedFromLayer(source)` for key and value and `value_norm = false` (:98-105). So layer 17 (sliding) is
`SharedFromLayer(13)`, layer 19 (full) is `SharedFromLayer(14)`. Cache-owning layers are the `ProjectedK`
ones: 15 (`bind.rs:691-694` counts them the same way); `layer_roots` has 35 entries, 20 of them
`SharedFromLayer` (digest line `bind.layer_roots=35`). Model-global: `ple_dim = Some(256)`,
`logit_softcap = Some(30.0)`, `embedding_scale = Some(Sqrt)`, `cache_strategy = TwoRange`.

Pass condition (the test): load the TOML profile above, build the descriptor, assert it equals
`gemma4_descriptor_from_gguf` on the same header (`ModelDescriptor` derives `PartialEq`,
`descriptor.rs:53`), and that `build_forward` on it yields `bind.ops = 6074` with `bind.ops_sha256 =
5765164066d03badcbe4cf9b43a79969dc984ae4fddc9c5ebad9f621bee69164` and `bind.layer_roots = 35` (all read from
`gemma4_e2b.digest`; the `arch_data_digest_` tests compare these today). The bound-weight table
(`gemma4_e2b.bound`: 542 weights; codecs counted from the file: `Q4_0` 275, `f32` 263, `Q6K` 3,
`Float16` 1) is the H5 check: a name set derived from the descriptor must equal that list.

Config parity (P4): TOML versus builder versus env, by the `PromptCacheSettings` triple
(`interop/prompt_cache_settings.rs:17-100`).

## 6. HOOK GAPS (stage, missing input, smallest generic change)

GAP-1. H4 describe. Missing input: the descriptor cannot be written in TOML. `ModelDescriptor` derives
`Debug, Clone, PartialEq` only (`descriptor.rs:53`), `LayerSchedule`, `LayerAttentionConfig`,
`LayerFfnConfig` are `Copy` data with no serde, and `RopeTableSel { cos_name: &'static str, sin_name:
&'static str }` (`attention_forward.rs:99-103`) cannot be deserialized into (a borrowed `'static` string).
Smallest change: serde on the descriptor family; `RopeTableSel` becomes a two-value enum (`Full`,
`Sliding`) mapped to the two leaf-name pairs inside the builder, or owned strings. Call site both ways: the
builder reads `attention.rope_table.cos_name` (`attention_forward.rs`, the leaf declaration); with the enum
it reads `rope_table.cos_name()`; identical lines, so this is a data-shape fix, not a type. Already owned
by architecture-as-data slice 10a per the sibling sketch 3, GAP-1.

GAP-2. H4 describe. Missing input: the family constants are code. Smallest change: `ModelProfile`, the
serde record of section 2's `[model.*]` tables, read by one generic `descriptor_from_gguf`. It also takes
the rope data now spread over `ModelArchitecture::sliding_rope`, `Gemma4Arch::step_inputs` with its
literal leaf names `rope_cos_swa`/`rope_sin_swa` (`bind.rs:873-892`), and `rope_freq_factors`
(`bind.rs:900-928`): the second per-family override seam (an `Architecture` trait object) that a profile
makes unnecessary for this family. The pass condition is section 5's digest equality.

GAP-3. H4 / H5. Missing input: one source for the GGUF key reads and the tensor name set. Today three
readers of the same keys and three gates for the same tensors (section 1). Smallest change: `bind` and
`hparams` take the descriptor (or its derived name set) instead of re-reading the header; `gemma4_tensor_names`
becomes a function of the descriptor. Evidence this is a real hazard, not tidiness: `bind.rs:931-948`
records the bug it already caused. Delete `hparams::from_metadata`'s duplicate reads once the bind takes the
descriptor.

GAP-4. H6 lower. Missing input: the KV layout and an env read. `kv_layout` is a literal at three load
entry points (`pregather.rs:2447, 2520, 2552`) and the descriptor's `sliding_kv_ring` is "the caller's
choice" (`gguf_descriptor.rs:18-20`); `PROXIMA_HEAD_REPEATS` is read inside the builder
(`attention_forward.rs:2121`) so the op graph depends on process env. Smallest change: `[lower]
kv_layout` consumed at load; the env knob becomes a descriptor field or is deleted (it is an
`instrument`-gated diagnostic per `attention_forward.rs:2065`; per the repo rule diagnostics are
structured events, not env-gated graph changes).

GAP-5. H7 specialize. Missing input: per-model specialization values. The stage's only configuration is
`omega/omega-runtime.toml`, consumed at `cargo build`, with numbers measured on one model (`min_tokens =
160`, "gemma4-E2B Q4_0"). So "gemma4 E2B entirely from configuration" is false for H7 by construction:
a different checkpoint on the same binary gets E2B's thresholds. Smallest change: a `[specialize]` table
read at plan time that overrides the build-time constants per model (the build-time value stays the
default, per principle 12), or an explicit statement in the SPEC that H7 is system-scoped, not
model-scoped. Status: the build-time-only fact is read from the file header; a `git ls-tree` search finds only `omega/omega-runtime.toml`; whether any runtime override
exists elsewhere was not checked.

GAP-6. H5 / H8. Missing input: loaders. `ServingConfig` is `Copy` with borrowed slices and has no
serde (`serving.rs:719`; the reason is documented at `speculative_settings.rs:10-13`), so `bind.precision`
and `placement.*` have no TOML path. Smallest change: the `SpeculativeSettings` pattern for each: an
owned, serde, `Settings` mirror lowered by an `as_*` method. Also `DEFAULT_GPU_LAYERS` depends on a cargo
feature (`serving.rs:145-149`), so the same TOML means "all on GPU" in one build and "CPU" in another;
the default should be `"all"` with admission failing loudly when the backend is absent.

GAP-7. H14 sample. Missing input: the GGUF's own sampling defaults as a layer. `general.sampling.top_k /
top_p / temp` (kv :6-8) are unread; the incumbent honours them as defaults (llama.cpp's
`COMMON_PARAMS_SAMPLING_CONFIG_*` flags, `common/common.h:160-171`, mark which fields a user set). Smallest
change: `[sampling] source = "gguf"` lowers them into the same record; default `"default"` keeps greedy,
which the parity gate needs (`temperature: 0` in the oracle runs, gate doc :25-27). Everything else: sketch 6.

GAP-8. H16 detokenize. Missing input: control-token visibility is a hard-coded branch
(`residency_caches.rs:3453-3458`), and the `▁` to space rule is keyed off vocab shape
(`:3404-3407`). A `[detokenize]` record `{ control = "hide" | "show" }` is the smallest change; the
shape-keyed rule stays (derived from the vocab, like `is_char_level_bpe`). There is also no stop-string
hold-back stage (hold text that might begin a stop string before emitting): `decode_streamed_piece` holds
only incomplete UTF-8 (:3395-3403). Not needed by this sketch; the SPEC lists it as thin-in-literature.

H2, H3, H15: see sketches 4, 5 and 4 (GAP-4) respectively.

## 7. what is not a pipe, and why

- `descriptor_from_gguf`, `build_forward`, `tensor_names`: pure functions run once per model load, not
  stream steps.
- The profile: plain data.
- The H7 constants: compile-time numbers. Principle 12 makes build-time the right home for per-system
  tuning; the gap is only that they also carry per-model measurements.
- The only stage that is a per-token stream step in this path is H14/H16, and both are synchronous
  reductions (sketches 6; `decode_streamed_piece`).

## 8. designs abandoned

- A `Gemma4Profile` type: one family name in a type is the per-instance rule the invariant forbids; the
  profile is a generic record.
- Generating the descriptor from the checkpoint with no profile: the constants (unscaled score,
  `value_norm`, GeLU-tanh) are model-card facts, not GGUF keys; the file does not carry them
  (`gguf_descriptor.rs:12-14` says exactly this).
- Making H7 a runtime stage list: the thresholds are consumed by kernel emission at build; a runtime
  table would add a lookup to the plan path for a decision made once.

## defects found in passing

- Triplicated reads of the same GGUF keys and tensor gates (GAP-3), with a documented past mismatch.
- `unreachable!` in non-test source: `bind_gemma4_with_last_row_only` (`bind.rs:707-709`), against the rule
  "no panic in production" (guarded by a length check at :695, so not reachable; still an `unreachable!`).
- `std::env::var` inside the op-graph builder (GAP-4).
- `DEFAULT_GPU_LAYERS` changes meaning by cargo feature (GAP-6).

## verdict

fits after 8 named hook changes here (GAP-1 through GAP-8), plus those the single-stage sketches name for
the stages they own (sketch 4: stage, segments, Jinja engine, end-of-turn stop, BOS ownership, request
variables; sketch 5: key read, scanner record, mark table, policies, `ignore_merges`; sketch 6: chain list,
loader, per-request sampling, argmax-preservation function, mask stage). The pass condition is stated and
checkable (section 5, digest equality) but no part of it was run. H7 is the stage where the claim "entirely
from configuration" is false by construction (GAP-5). Not claimed: that the TOML profile reproduces token
ids, or that configuration-driven lowering costs no speed; neither was measured.
