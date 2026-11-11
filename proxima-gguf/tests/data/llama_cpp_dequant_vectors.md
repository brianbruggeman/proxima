# llama_cpp_dequant_vectors.bin

Golden outputs of llama.cpp's own `dequantize_row_<t>` for q4_0, q4_1, q5_0, q5_1, q8_0, q2_K, q3_K,
q4_K, q5_K and q6_K. Layout and hashing are documented in `../llama_cpp_dequant_vectors.rs`.

- dylib: `/Users/brianbruggeman/repos/others/llama.cpp/build/bin/libggml-base.dylib`
- build: Release, `GGML_NATIVE=ON`, Xcode clang, built 2026-09-06 14:34 (file mtime)
- llama.cpp commit (checkout the dylib was built from, HEAD at generation time):
  `f1ea206218210afb913ae2f5d2c51faed35915da`
- `git log --since='2026-09-06 14:34' -- ggml/src/ggml-quants.c` shows one commit,
  `f95b0d953 ggml : IQ1_M build prefix sums once per block (#28706)`, whose hunks are
  `@@ -4774,68 +4774,31 @@` and `@@ -4843,5 +4806,13 @@`, both inside `quantize_row_iq1_m_impl`.
  No `dequantize_row` line changed since the build.
- generator: Rust program `vecgen` (libloading over the dylib; fixed splitmix64 seed
  `0x6c6c616d61637070 ^ (format_index << 32)`), kept outside the repo in the author's scratch
  directory `.long_ctx_backups/kv-cards/vecgen`.
- NaN bits in `H_all` are those of the arm64 fused build; `H_canon` is target-independent.
