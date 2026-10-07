# What configuration cannot do

Read `a-model-is-a-configuration.md` first. Every `file:line` here was read at commit
`365fa7f6` plus this slice. Every behavior quoted below is printed by one example, which
reads no checkpoint:

```
cargo run -p proxima-model-interop --example model_config_limits --features std
```

## The rule

A configuration composes compiled primitives; it does not add one. `ModelDescriptor`
(`proxima-tensor/src/spec/descriptor.rs:67`) is `#[serde(deny_unknown_fields)]` (`:66`) and its
layer enums are closed, so a config can only choose among the fields, variants and engines the
binary already has. A new architectural idea therefore takes two steps, in this order:

1. A compiled primitive: an op builder in `proxima-tensor`, a field or variant that names it on
   the descriptor, and a refusal in every engine that cannot lower it.
2. Configuration: from then on every model that uses the idea is a file.

Two ideas in this repository went through step one before any config used them, and the commits
show the order.

## Granite's scale settings

Granite multiplies the embeddings (`embedding_scale`), each sublayer output before it joins the
residual stream (`residual_scale`), the attention scores (`score_scale`) and divides the final
logits (`logit_scale`). The last two descriptor-level scales are fields with doc comments
(`descriptor.rs:96-102`). They landed as `c5be749b feat(tensor): carry a logit scale and refuse
engines that lack it` and `17753f99 feat(tensor): carry a residual scale and refuse engines that
lack it`: the field, the lowering in the engine that honours it, and a refusal in the engines
that do not.

The refusal is a config-visible fact. Of the three cache engines, `(Cached, Bounded)` passes both
scales to its builder (`descriptor.rs:836-838`); `(Cached, Padded)` (`descriptor.rs:680-690`) and
`(Cacheless, _)` (`descriptor.rs:721-731`) call `refuse_when` on them. The example flips
granite's `cache_mask` to `Padded` and keeps the scales:

```rust
    let padded = ModelDescriptor { cache_mask: CacheMask::Padded, ..granite() };
    let padded_refusal = refusal_of(&padded);
    assert!(padded_refusal.contains("a logit scale"), "{padded_refusal}");
```

```
granite's logit scale on the padded cache engine: build_forward(CacheMask::Padded) does not support a logit scale
the same config minus the two scales lowers: 2752 ops
```

The refusal names the engine and the missing feature; deleting the two scales from the same
config lowers it. A config that kept the scales and needed the padded engine would need step
one: teach that engine the scale.

## The gated delta net layer kind

`LayerKind` has three variants, `Attention`, `ShortConv` and `Gdn`
(`proxima-tensor/src/spec/single_range_moe_cached.rs:1240-1244`). `Gdn` landed as
`aba8621f feat(tensor): add the gated delta net layer kind`. A schedule holding a `Gdn` layer, or
a descriptor with `gated_attention`, is lowered by the recurrent-hybrid engine
(`build_forward`, `descriptor.rs:672-679`), which reads the `ssm_*` fields
(`ssm_conv_kernel` through `v_head_reordered`, `descriptor.rs:179-200`) and keeps a state cache beside the
KV cache. Every other engine refuses the kind
(`attention_forward.rs:2113-2118`: "LayerKind::Gdn (lowered by the hybrid engine)").

What a config cannot do is mix this kind into an engine that has no recurrence. The example
marks every granite layer `Gdn`:

```rust
    let gdn = with_layers(granite(), |layer| LayerSchedule { kind: LayerKind::Gdn, ..layer.clone() });
    let gdn_refusal = build_forward(&gdn).expect_err("the recurrent engine refuses a routed granite schedule");
    assert!(
        matches!(gdn_refusal, TensorError::UnsupportedInBuilder { builder: "build_forward(hybrid dense)", feature: "routed experts" }),
        "got {gdn_refusal:?}"
    );
```

```
every layer kind = Gdn on the granite moe schedule: UnsupportedInBuilder { builder: "build_forward(hybrid dense)", feature: "routed experts" }
```

The descriptor was routed to the hybrid engine because of the layer kind, and that engine's dense
path refuses routed experts (granite's FFN combination is `Exclusive`, not
`RoutedWithSharedExpert`, so `descriptor.rs:673-677` picks `hybrid_dense_forward`). The refusal
names a missing combination of compiled capabilities, not a missing config key. Closing it is a
change to the hybrid engine (its dense path routing experts for this combination), after
which a config can ask for it.

## Other refusals, all before any weight is read

```rust
    let split_half = with_layers(granite(), |layer| LayerSchedule {
        attention: LayerAttentionConfig { rope_pairing: RopePairing::SplitHalf { pairs: layer.attention.head_dim / 2 }, ..layer.attention.clone() },
        ..layer.clone()
    });
```

```
split-half rope on the moe layer: build_forward(CacheMask::Bounded) does not support a rope pairing the moe layer cannot express
```

The refusal (`descriptor.rs:799-811`) is about weights as much as engines: over a GGUF
whose q and k rows the converter already permuted (see the rope section of the first guide), the
lowering derives the one pairing it can express from `qk_norm` and refuses any other.

A key no field reads, and a variant no enum names, fail at parse time with the list of what is
allowed:

```
unknown key: TOML parse error at line 13, column 1
...
unknown field `sliding_window_size`, expected one of `vocab`, `embedding`, ... `gated_attention`

unknown activation: TOML parse error at line 55, column 14
...
unknown variant `Mish`, expected `Silu` or `GeluTanh`
```

Those lists are the compiled vocabulary. A new activation is a new `Activation` variant and its
elementwise expression first; the file naming it comes second.

## What this means for a new model

Ask three questions of the idea, in order:

1. Is it a value of an existing field (a width, a count, a scale, a pairing)? Then it is a file
   edit, or a header key the loader already reads.
2. Is it a combination of existing layer kinds, engines and fields the lowering already accepts?
   Then it is a file, and the three parity checks say whether the combination is right.
3. Neither? Add the primitive with its refusals and its test, then write the file. The refusal
   text in `build_forward` is the to-do list: it names the engine and the feature.
