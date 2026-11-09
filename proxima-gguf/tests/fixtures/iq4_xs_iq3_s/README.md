# IQ4_XS / IQ3_S parity vectors, from llama.cpp itself

Two codecs the flint capture stores its routed experts in. The oracle here is the incumbent's own decoder — not a
retyped reference, not one of proxima's own paths — so a mistranslation on either side of the port shows up as a
different `f32` bit pattern.

What is committed:

| file | what it is |
|---|---|
| `iq4_xs.bin` / `iq3_s.bin` | four real super-blocks of `blk.19.ffn_down_exps.weight` (iq4_xs) and `blk.2.ffn_gate_exps.weight` (iq3_s), lifted from the checkpoint's own data section |
| `iq4_xs.expected.txt` / `iq3_s.expected.txt` | llama.cpp's decoded `f32` for exactly those bytes, one bit pattern per element in hexadecimal |
| `manifest.json` | which tensor each run came from, its declared dims, its offset, and the checkpoint's sha256-bearing path |
| `iql_extract.py` | the standalone GGUF reader that produced the `.bin` files (shares no code with `proxima-gguf`) |
| `iql_oracle.py` | builds and runs the oracle: it lifts `dequantize_row_iq4_xs`, `dequantize_row_iq3_s` and the `ggml-common.h` tables VERBATIM out of a llama.cpp checkout and compiles them |

Checkpoint: `/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf`, sha256
`63c2ecc461ca545134b670c660e76fa20f47240f8d39b6510bfd21c4dd6e072f`, llama.cpp `cd959bae9`.

Regenerate (from this directory, with a llama.cpp checkout at `$LLAMA`):

```
python3 iql_extract.py /Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf . 4
python3 iql_oracle.py "$LLAMA" . cc
./iql_oracle iq4_xs.bin iq4_xs > iq4_xs.expected.txt
./iql_oracle iq3_s.bin iq3_s > iq3_s.expected.txt
```

`iql_oracle` and `iql_oracle.c` are generated and ignored. The consumer is
`proxima-gguf/tests/iq4_xs_iq3_s_llama_cpp_oracle.rs`, which compares raw `u32` bit patterns and carries a control that
a flipped packed byte moves at least one decoded element.
