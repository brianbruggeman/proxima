# A model is a configuration

Every `file:line` below was read at commit `365fa7f6` plus this slice; line numbers drift, the
symbol names do not. Every code block is a runnable example, quoted from the file it names.
Run them from `proxima-windows/`:

```
cargo run -p proxima-model-interop --example model_config_expand --features std
cargo run -p proxima-model-interop --example model_config_load   --features std,metal
```

`model_config_expand` reads no checkpoint. `model_config_load` needs the Ollama
`granite3.1-moe:1b` blob (or `PROXIMA_ARCH_GRANITE_MOE_GGUF`) and a quiet Mac.

## The claim, and what it rests on

A Granite 3.1 MoE model runs through one engine with no Granite-specific Rust. What makes it
Granite is the file
`proxima-model-interop/tests/fixtures/model-configs/granite_moe.toml`, a `ModelDescriptor`
(`proxima-tensor/src/spec/descriptor.rs:67`) written by hand. Lowering is a pure function of that
descriptor (`build_forward`, `descriptor.rs:671`); binding takes the descriptor and the
checkpoint's tensor directory (`LoadedModel::load_with_descriptor`,
`proxima-model-interop/src/generate/pregather.rs:2561`).

What the config does not hold: tensors. Weights bind from the GGUF tensor directory by name.

## Where each value comes from

The file's own header (`granite_moe.toml:1-12`) says it was written field by field from the
published Hugging Face `config.json`. There are two sources, and they differ in spelling and in
what they carry. The table is checked two ways: the hand-written config must equal the descriptor
the loader derives from the GGUF header
(`hand_written_granite_moe_config_equals_the_gguf_header_descriptor`,
`proxima-model-interop/tests/arch_data_baseline.rs`, through `assert_hand_written_config_equals_header`),
and it must reproduce llama.cpp's ids (below).

| config field | value | Hugging Face `config.json` | GGUF header key (`llama-parity/granite_moe/gguf_kv.txt`) |
|---|---|---|---|
| `vocab` | 49155 | `vocab_size` | `granitemoe.vocab_size` |
| `embedding` | 1024 | `hidden_size` | `granitemoe.embedding_length` |
| `feed_forward`, `expert_feed_forward` | 512 | `intermediate_size` (width of each expert) | `granitemoe.feed_forward_length`; one key feeds both fields (`gqa_descriptor_from_shape`, `descriptor.rs:519`) |
| `query_heads` | 16 | `num_attention_heads` | `granitemoe.attention.head_count` |
| `kv_heads` | 8 | `num_key_value_heads` | `granitemoe.attention.head_count_kv` |
| `head_dim` | 64 | not used: the converter drops it (`granite.py:33-34`); the width is `hidden_size / num_attention_heads` | no key; `granitemoe.rope.dimension_count` is 64 and `require_full_rotary` (`dense.rs:206`) refuses a header where it differs |
| `block_count` | 24 | `num_hidden_layers` | `granitemoe.block_count` |
| `expert_count`, `expert_used_count` | 32, 8 | `num_local_experts`, `num_experts_per_tok` | `granitemoe.expert_count`, `granitemoe.expert_used_count` |
| `embedding_scale` | `Factor = 12.0` | `embedding_multiplier` | `granitemoe.embedding_scale` (`with_header_scales`, `dense.rs:193`) |
| `score_scale` | `Factor = 0.015625` | `attention_multiplier` | `granitemoe.attention.scale` (`dense.rs:178`) |
| `residual_scale` | 0.22 | `residual_multiplier` | `granitemoe.residual_scale` (`dense.rs:197`) |
| `logit_scale` | 6.0 | `logits_scaling` | `granitemoe.logit_scale` (`dense.rs:196`) |
| `activation` | `Silu` | `hidden_act` | none; the family profile (`profiles/granitemoe.toml`) |
| `routed_gating` | `Softmax` | none | none; the family profile |
| `qkv_biases`, `qk_norm` | false | `attention_bias` | the tensor directory: `checkpoint_qkv_biases`, `checkpoint_has_qk_norm` (`dense.rs:81-82`) |
| `rope_table` names | `rope_cos`, `rope_sin` | none | none; the names the lowering reads (`descriptor.rs:498-501`) |
| `last_row_only` | true | none | decided by task: false only when `classify_task` says embedding (`dense.rs:91`) |
| `cache_strategy`, `cache_mask`, `head_repeats`, `sliding_kv_ring` | `Cached`, `Bounded`, 1, false | none | none; engine choices, not architecture (`descriptor.rs:534-535`, `:544`) |

Two things this table makes visible. First, the GGUF header carries the four multipliers
`config.json` carries because llama.cpp's converter wrote them, renaming `_multiplier` to
`_scale` (`GraniteModel.set_gguf_parameters`, `conversion/granite.py:23-46` at llama.cpp
`f1ea20621`). Second, some fields come from neither: the activation, the gating function and
the rope layout are knowledge about the family that lives in `profiles/granitemoe.toml`, and a
hand-written config states them in the open.

## Why rope pairing is adjacent

`config.json` says nothing about how a head's channels pair for rotation. The fixture header
(`granite_moe.toml:6-10`) records that Hugging Face computes `rotate_half`, which pairs channel
`i` with channel `i + head_dim/2` (split-half). The config says `rope_pairing = "Interleaved"`
(pairs `(2i, 2i+1)`). Both are right, about different weights:

- llama.cpp's converter reorders the q and k projection rows when it writes the GGUF.
  `LlamaModel.undo_permute = True` (`conversion/llama.py:35`); `LlamaModel.permute` reshapes the
  rows to `(n_head, 2, rows/n_head/2)` and swaps the two inner axes (`llama.py:176-181`);
  `modify_tensors` applies it to every `q_proj` and `k_proj` weight and bias (`llama.py:260-264`).
  `GraniteModel(LlamaModel)` (`granite.py:19`) and `GraniteMoeModel(GraniteModel)`
  (`granite.py:181`) never set `undo_permute` (the only overrides in `granite.py` are on
  `GraniteSwitchModel` and `GraniteHybridModel`, lines 238 and 397), and
  `GraniteMoeModel.modify_tensors` ends in `super().modify_tensors` (`granite.py:227`).
- The runtime then reads those rows as consecutive pairs: `LLM_ARCH_GRANITE_MOE` is in the case
  list that returns `LLAMA_ROPE_TYPE_NORM` (`llama-model.cpp:3019-3039`).

So over this GGUF the pairing is `Interleaved`, and the family profile says so with
`rope_layout = "adjacent"` (`profiles/granitemoe.toml:3`, read by `rope_pairing`,
`descriptor.rs:444-449`).

The lowering guards the claim. For a routed layer it derives the one pairing it can express from
`qk_norm` (`descriptor.rs:799-811`): split-half with qk-norm, interleaved without. A config that
says `SplitHalf` for granite is refused before any weight is touched; guide two shows the
refusal text, and `granite_moe_config_with_the_hf_split_half_pairing_is_refused_by_the_lowering`
(`arch_data_baseline.rs`) asserts it.

## `repeat` and `pattern`

`layers` is a list of entries. In a file an entry is either one layer (`kind`, `attention`,
`ffn`) or a `pattern` of entries, and either one is repeated `repeat` times (default 1). The
loader expands them to one `LayerSchedule` per block (`LayerRun`,
`proxima-tensor/src/spec/layer_runs.rs:19`; `expand`, `:30`). An entry that is both a layer and
a pattern, or neither, is refused (`layer_runs.rs:35`), and a total over 65535 layers is refused
(`MAX_LAYERS`, `:8`). The descriptor serializes expanded, so a config written with `repeat`
round-trips to the long form.

The granite file is one entry, `repeat = 24`. The example proves the expansion, the round trip,
and a nested `pattern` built from the same layer text (a windowed layer then a full layer,
twelve times, the shape Gemma 4 uses: `layer_runs.rs` test
`a_pattern_repeats_in_order_like_gemma4_sliding_then_full`):

```rust
    assert_eq!(descriptor.layers.len(), 24, "`repeat = 24` expands to one schedule per block");
    assert!(descriptor.layers.iter().all(|layer| *layer == descriptor.layers[0]));

    let expanded = toml::to_string(&descriptor).expect("a descriptor serializes");
    let written_entries = expanded.matches("[[layers]]").count();
    assert_eq!(written_entries, 24, "a descriptor serializes expanded, one entry per block");
    let reloaded: ModelDescriptor = toml::from_str(&expanded).expect("the expanded form loads again");
    assert_eq!(reloaded, descriptor, "expanded and repeated forms are the same descriptor");

    let patterned: ModelDescriptor = toml::from_str(&windowed_then_full_text(&text)).expect("the pattern config loads");
    conflaguration::Validate::validate(&patterned).expect("a pattern config holds one entry per block");
    let windows: Vec<Option<u32>> = patterned.layers.iter().map(|layer| layer.attention.mask_window).collect();
    assert!(windows.chunks(2).all(|pair| pair == [Some(512), None]), "the pattern alternates in order: {windows:?}");
```

`Validate` checks that the expansion holds exactly `block_count` entries
(`descriptor.rs:258-262`). Output, `slice_13/expand.log`:

```
repeat: 1 entry -> 24 layers, all equal
round trip: 24 serialized entries reload to the same descriptor
pattern: repeat 12 x [windowed, full] -> 24 layers, windows [Some(512), None, Some(512), None]...
lowering: 24 layers -> 6394 ops
```

## How the descriptor lowers to a program

`build_forward` (`descriptor.rs:671`) takes the recurrent-hybrid engine when `gated_attention`
is set or any layer has kind `Gdn`; otherwise it matches `(cache_strategy, cache_mask)`:
`(Cached, Padded)` is the two-range engine, `(Cacheless, _)` the plain one, and
`(Cached, Bounded)` the engine granite uses, which is where `logit_scale` and `residual_scale`
are passed on (`descriptor.rs:836-838`). The result is a `ForwardProgram` whose `program` is a
`Vec<Op>`. The granite config lowers to 6394 ops (last line above). That program is also what the
`arch_data_digest_*` tests hash, together with the bound weight set, against
`llama-parity/<name>.digest`.

Nothing in `build_forward` names a family. A different model is a different descriptor value.

## How a config loads

```rust
    let descriptor: ModelDescriptor = conflaguration::from_file(&fixture("model-configs/granite_moe.toml")).expect("the config loads");
    conflaguration::Validate::validate(&descriptor).expect("the config is internally consistent");
    let model = LoadedModel::load_with_descriptor(&parsed, &mapping, &descriptor).expect("the config binds over the checkpoint");

    let ForwardProgram { program, .. } = build_forward(&descriptor).expect("the config lowers");
    assert_eq!(model.op_count(), program.len(), "the loaded model runs the config's lowering");
```

`conflaguration::from_file` is the loader and `Validate` the cross-field check; both are the std
boundary (`ModelDescriptor` derives `Settings` and `bon::Builder` under the `config` feature,
`descriptor.rs:63-65`). `load_with_descriptor` takes the KV layout from the config
(`sliding_kv_ring`, `pregather.rs:2566-2571`) and calls `load_inner` with the descriptor, so the
op graph, the verify program and the placed single-range program follow the config; the weights
still bind from the GGUF tensor directory (the doc comment above `:2561`). That comment also
names the other way to build a descriptor: `conflaguration::builder().value(base).env().file(path)
.validate()` over a header-derived base, so a variant is an edit of a base config.

Output of the load example, `slice_13/load.log`:

```
loaded: 6394 ops, the same count build_forward gives the config
```

## How the three parity checks prove it

A config that loads proves nothing. Three checks, all on real checkpoints, all in this repo:

1. Correctness (tests). `cargo nextest run -p proxima-tensor --cargo-profile gate` and
   `cargo nextest run -p proxima-model-interop --features std,metal --cargo-profile gate
   --profile slice-gate`. For granite: `arch_data_digest_granite_moe` pins the lowered program
   and the bound bytes; `generic_binder_granite_moe` binds every tensor; the hand-written config
   equals the header-derived descriptor; the split-half config is refused.
2. Semantic (the model's response). `hand_written_granite_moe_config_reproduces_llama_ids` loads
   the config with `load_with_descriptor` and decodes the vendored prompts; the ids must equal
   llama.cpp `f1ea20621`'s, recorded once in `llama-parity/granite_moe/llama_ids.json` and
   replayed, never re-queried. The example does it for the first record:

```rust
    assert!(!generated.is_empty(), "decoded zero tokens");
    assert_eq!(generated, llama_ids, "ids differ from llama.cpp f1ea20621");
```

```
semantic: 32 generated ids equal llama.cpp's
```

3. Performance (wall clock, memory). `decode_arms` against a baseline binary: 2 processes x 3
   runs per slice that moves a digest or touches a decode loop, 8 x 7 at the end of the run, with
   a base-versus-control arm so the memory bound is `max(2% of the base median, |control - base|
   measured in the same run)`. A slice whose digests are byte-identical and that touches no
   decode loop inherits the baseline. Tiers and commands: `proxima-tensor/specs/architecture-as-data/SPEC.md`,
   section "gate tiers".

The incumbent is the oracle for check two. When ids differ, the first assumption is that the
config or the lowering is wrong, not llama.cpp.
