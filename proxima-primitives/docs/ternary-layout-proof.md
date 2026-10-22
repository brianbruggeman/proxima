# ternary block layout proof

## value and scale composition

The numeric relation is a nonrecursive composition: a stored code maps to a small value, and an ordinary scalar scale maps that value into the model's numeric range. The block index map selects which scale applies. No scale contains another recursively scaled value.

For both GGUF TQ formats, the logical code is `q ∈ {0, 1, 2}` and the decoded value is `(q - 1) * d`, where `d` is the block's F16 scale. The byte layout differs between the formats; the value relation does not.

## worked example from llama.cpp reference

### TQ2_0

Inputs: one block with `d = 2` and packed byte `qs[0] = 0b11_10_01_00`.

The reference loops over four 2-bit lanes in increasing lane order. The extracted codes are `[0, 1, 2, 3]`. Applying `(q - 1) * d` gives `[-2, 0, 2, 4]`. Code 3 is not emitted by the reference quantizer for normalized inputs in `[-1, 1]`, but the reference dequantizer maps a stored code 3 to `2*d`; the decoder must preserve that behavior.

### TQ1_0

Inputs: one block with `d = 2`, `qs[0] = 85`, all other `qs` bytes zero, `qh[0] = 83`, and all other `qh` bytes zero.

The reference stores five base-3 digits in each `qs` byte using a value scaled from `[0, 242]` to `[0, 255]`. For a stored byte `q` and lane `n`, it extracts

```text
trit(q, n) = floor(3 * ((q * 3^n) mod 256) / 256)
```

The base-3 word for `[0, 2, 2, 2, 2]` is `80`, and the stored byte is `ceil(80 * 256 / 243) = 85`. For `q = 85`, lanes `n = 0..4` produce `[0, 2, 2, 2, 2]`; after subtracting one and multiplying by 2, the values are `[-2, 2, 2, 2, 2]`.

The three source regions map logical element indexes to packed bytes and lanes:

| logical indexes | source byte | lane | group width |
|---|---|---|---:|
| 0..159 | `qs[i % 32]` | `floor(i / 32)` | 32 |
| 160..239 | `qs[32 + (i - 160) % 16]` | `floor((i - 160) / 16)` | 16 |
| 240..255 | `qh[(i - 240) % 4]` | `floor((i - 240) / 4)` | 4 |

The padded `qh` word is built from digits `[0, 2, 2, 2]` as `0*27 + 2*9 + 2*3 + 2 = 26`, then padded to five digits (`26*3 = 78`) and stored as `ceil(78 * 256 / 243) = 83`. Thus for `qh[0] = 83`, lanes `0..3` produce `[0, 2, 2, 2]`, so output positions `[240, 244, 248, 252]` are `[-2, 2, 2, 2]`. The padded fifth digit is not emitted.

## algorithm

1. Select the element region containing the logical index.
2. Use that region's mixed-radix address map to read the packed source byte.
3. Extract the selected lane either as the existing binary plane value or as a radix digit using the expression above.
4. Apply the common value relation `(code - 1) * scale`.

TQ2 uses the existing binary-plane extraction with a new structural layout constant. TQ1 uses three fixed structural regions and radix 3 extraction. Both use the same scalar code layout, scale field, and CPU block walker. The generic MSL block-layout renderer can render these descriptors; this change does not establish a current Omega kernel call site or end-to-end model inference support.

## reference artifacts

- llama.cpp `ggml/src/ggml-common.h`: `block_tq1_0`, `block_tq2_0`.
- llama.cpp `ggml/src/ggml-quants.c`: `dequantize_row_tq1_0`, `dequantize_row_tq2_0`, `quantize_row_tq1_0_ref`, `quantize_row_tq2_0_ref`.
