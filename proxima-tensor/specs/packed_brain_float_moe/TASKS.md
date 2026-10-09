# Packed brain-float MoE forward -- slices

Each slice: one behavior change, one validation command, under ~30 minutes.
Update the checkbox and note in the same change as the slice.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Audit packed-storage ownership and admit the representation contract | AC1-AC3 | rg -c '^VERDICT: ADMIT$' proxima-tensor/specs/packed_brain_float_moe/ADMISSION.md | 1 admitted revision recorded | [x] | The auditor admitted the bound Codec tags, packed carrier, CPU dispatch path, and exact-byte evidence requirements. |
| 2 | Add source-neutral codec identities and direct packed-byte CPU matmul | AC1-AC2 | cargo test -p proxima-primitives --lib codec_brain_float_tags_roundtrip && cargo test -p proxima-gguf --test brain_float_codecs_not_ggml brain_float_codecs_do_not_change_ggml_wire_types -- --exact && cargo test -p proxima-model-interop --features std --lib brain_float_codecs_have_no_ggml_target && cargo test -p proxima-tensor --test packed_brain_float_moe_matmul bf_packed_moe_matmul_matches_fp32_oracle -- --exact | 4 tests passed, 0 failed; tags map BF8=29/BF4=30, GGML IDs unchanged, no GGML target exists, and both encodings produce [1.5,-4,-5.5,1.5] | [x] | Unit-test filters omit `--exact` because Cargo prefixes unit test names with their module path; the first attempted exact filters selected zero tests. All four corrected commands each ran one test and passed. |
| 3 | Compose packed expert gather with direct CPU matmul | AC3 | cargo test -p proxima-tensor --features instrument --test packed_brain_float_moe_matmul packed_bf_computed_gather_passes_selected_bytes -- --exact | 1 passed, 0 failed; Computed-map evaluation returns [-5.5,1.5] and captured dispatch event contains the exact selected bytes | [x] | Route [1] selected BF8 [0xc0,0x3e,0x38,0xb8] and BF4 [0x3c,0x91]; captured event assertions passed. |
## resume

Last completed slice: 3
Next action: integrate packed BF4/BF8 operands with autograd and GPU execution under a separate admitted spec
Open question, if any: GPU and packed-autograd integration remain separate later phases

## struck

-
