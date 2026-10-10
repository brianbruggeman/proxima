# Card 13b2c2: resolve pinned LFM layers through binding profile data

## Contract

Remove the public LFM tensor-view structures and hand-selected layer binder.
Use one model-neutral safetensors program-view resolver for named `Op::Input`
leaves. It applies `BindingProfile` aliases, the existing program consumer-role
analysis, exact source-axis checks, dtype byte-width checks, and file-range
checks. Each result carries the program leaf name, resolved checkpoint name,
checkpoint dtype and shape, and a borrowed slice of the caller's file bytes.
The existing `bind_safetensors_program_leaves` consumes the same validated
views and retains its F32 decode, packed matmul, and layout behavior.

The binding profile maps these graph leaf names to pinned checkpoint suffixes:

| Graph leaf | Checkpoint tensor | Pinned axes |
|---|---|---|
| `blk.N.attn_norm.weight` | `operator_norm.weight` | `[2048]` |
| `blk.N.ffn_norm.weight` | `ffn_norm.weight` | `[2048]` |
| `blk.N.ffn_gate.weight` | `feed_forward.w1.weight` | `[8192,2048]` |
| `blk.N.ffn_down.weight` | `feed_forward.w2.weight` | `[2048,8192]` |
| `blk.N.ffn_up.weight` | `feed_forward.w3.weight` | `[8192,2048]` |
| `blk.N.shortconv.conv.weight` | `conv.conv.weight` | `[2048,1,3]` |
| `blk.N.shortconv.in_proj.weight` | `conv.in_proj.weight` | `[6144,2048]` |
| `blk.N.shortconv.out_proj.weight` | `conv.out_proj.weight` | `[2048,2048]` |
| `blk.N.attn_q.weight` | `self_attn.q_proj.weight` | `[2048,2048]` |
| `blk.N.attn_k.weight` | `self_attn.k_proj.weight` | `[512,2048]` |
| `blk.N.attn_v.weight` | `self_attn.v_proj.weight` | `[512,2048]` |
| `blk.N.attn_output.weight` | `self_attn.out_proj.weight` | `[2048,2048]` |
| `blk.N.attn_q_norm.weight` | `self_attn.q_layernorm.weight` | `[64]` |
| `blk.N.attn_k_norm.weight` | `self_attn.k_layernorm.weight` | `[64]` |

Each alias is a fully qualified Rename row from `blk.N.*` to
`model.layers.N.*` in the existing binding-profile schema. The pinned
16-layer schedule needs 146 rows: 10 × (5 common + 3 convolution) and
6 × (5 common + 6 attention). The test binds the eight layer-0 leaves and
the eleven layer-2 leaves through those rows. The pinned config's
16-entry schedule has 10 short-convolution and 6 attention layers. The
test gives all 10 FFN/attention matrix leaves real multiply/reduce consumers:
their program axes are `[in,out]` while the checkpoint axes are `[out,in]`.
Norm and convolution leaves use native axes. The generic resolver reuses the
same consumer-role analysis as the execution binder. The retained safetensors
header supplies real names, axes, BF16 dtype, and declared
byte lengths; synthetic bytes fill only the selected layer ranges. The pinned
upstream `Lfm2MLP` and `Lfm2DecoderLayer` source establishes the FFN width
adjustment and the shared versus mixer weights
(`fixtures/upstream-source/f399fa2a111dac8c7fc07b2717abb10ee3e82468851321f45553ebb10fbd28b1.py.gz`,
lines 105–119 and 527–570).

The current scheduled graph declares `blk.N.shortconv.conv.weight` with axes
`[2048,3]`, while the pinned safetensors source has `[2048,1,3]`. This card
checks the source view with its exact three axes. A later graph execution must
express the singleton-axis view using existing tensor operations before its
program leaf can pass the execution binder's exact-axis rule.

## Acceptance

Run one CPU test process:

```sh
nice -n 20 env CARGO_BUILD_JOBS=1 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= cargo test -p proxima-model-interop --features std --lib architecture_matrix_lfm_layer_bind -- --nocapture
```

Require one passing named test and exactly this marker:

```text
layers=16 shortconv=10 attention=6 profile_matches=19 bad_shapes_rejected=2 model_binder_symbols=0
```

For each of the 19 resolved leaves, assert the exact `blk.0`/`blk.2` graph
name and `model.layers.0`/`model.layers.2` source name,
shape, BF16 dtype, byte length, and pointer identity against the compact
file buffer. Mutate convolution kernel width from 3 to 2 and attention query
width from 2048 to 2047; both must return `SafetensorsAxesMismatch`. The
following source search must return zero matches:

```sh
rg -n 'Lfm(TensorView|CommonLayerWeights|MixerWeights|LayerWeights)|bind_lfm_layer|mod lfm_bind' proxima-model-interop/src
```

Record the command, host, compiler, full output, exit status, source-search
status, and `git diff --check` in
`evidence/card-13b2c2-lfm-profile-leaves.txt`. These assertions cover a
header-derived synthetic payload and source binding; they do not establish a
model invocation or numerical parity.
