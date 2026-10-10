# Card 12: bind the pinned LFM2 layer tensors

## Pinned tensor contracts

The pinned base config and the retained safetensors header establish two different mixer layers: layer 0 is short convolution, layer 2 is full attention. Both layers carry `operator_norm`, `ffn_norm`, and the three `feed_forward.wN` tensors (`fixtures/upstream-source/f399fa2a111dac8c7fc07b2717abb10ee3e82468851321f45553ebb10fbd28b1.py.gz:527-570`). The layer binder therefore returns common FFN/norm views plus one typed mixer variant.

The generic `intermediate_size` in the config is 12,288, but it is not the width used by the LFM2 MLP. The pinned source applies `int(2 * intermediate_size / 3)` when `block_auto_adjust_ff_dim` is true, then applies the optional multiplier and multiple rounding before constructing `w1/w2/w3` (`...py.gz:105-119`). For this checkpoint, the config has adjustment enabled, multiplier `1.0`, and multiple `256`; the resulting width is 8,192. The header independently records `w1/w3 [8192,2048]` and `w2 [2048,8192]`. `architecture_from_hf_config` now carries this effective width into `ModelHparams.feed_forward`.

| Layer | Required tensor | Pinned shape |
|---|---|---|
| 0, short convolution | `conv.conv.weight` | `[2048,1,3]` |
| 0, short convolution | `conv.in_proj.weight` | `[6144,2048]` |
| 0, short convolution | `conv.out_proj.weight` | `[2048,2048]` |
| 2, attention | `self_attn.q_proj.weight`, `self_attn.out_proj.weight` | `[2048,2048]` |
| 2, attention | `self_attn.k_proj.weight`, `self_attn.v_proj.weight` | `[512,2048]` |
| 2, attention | `self_attn.q_layernorm.weight`, `self_attn.k_layernorm.weight` | `[64]` |
| both | `operator_norm.weight`, `ffn_norm.weight` | `[2048]` |
| both | `feed_forward.w1.weight`, `feed_forward.w3.weight` | `[8192,2048]` |
| both | `feed_forward.w2.weight` | `[2048,8192]` |

All pinned tensors use BF16. The implementation exposes raw byte views borrowed from the caller's file buffer and keeps the checkpoint dtype/shape visible. These are source weight views, not evaluator-ready Proxima buffers: norms and the attention Q/K projections may need conversion or layout transforms in later graph-binding work.

## Borrowed-view fixture

The retained pinned artifact contains the 16,720-byte safetensors header only; its provenance states that the 2,340,697,936-byte weight file was not downloaded (`fixtures/lfm_text.safetensors.header.json`). The test parses that real header for names, dtypes, and shapes, then remaps the selected layer entries to a compact synthetic tensor payload. It checks that each returned slice has the exact shape and dtype and that its data pointer is the same pointer as the corresponding source-buffer range. This proves the binder's borrowing contract without claiming that the pinned model's tensor bytes were read.

Two negative controls mutate only the convolution kernel width (`3` to `2`) and attention query projection width (`2048` to `2047`); both binds must reject before returning views. The accepted layers are exactly one short-convolution layer and one attention layer, and the test counts all 19 returned views.

## Validation correction

The first acceptance spelling omitted `--features std`; because `proxima-model-interop` has `default = []` and this binder is std-gated, Cargo selected zero tests. The acceptance command now explicitly enables `std` and `--lib`. Its first compile then exposed that `DType::size_bytes()` returns `usize`; the binder now uses checked conversion before multiplying by the element count. The final capped-container run selected and passed the named test.
