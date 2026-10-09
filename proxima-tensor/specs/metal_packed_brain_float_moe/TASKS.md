# Metal packed brain-float MoE forward -- slices

Each slice: one behavior change, one validation command.
Update the checkbox and the note in the same change as the slice.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Admit packed BF Metal lowering contract | AC1-AC3 | `rg -c '^VERDICT: ADMIT$' proxima-tensor/specs/metal_packed_brain_float_moe/ADMISSION.md` | 1 admitted revision | [x] | Auditor admitted the explicit native index formula, odd-width nibble boundary, CPU transpose, fixed result, and unsupported-codec gate. |
| 2 | Add explicit MSL scalar decoders and codec admission | AC1, AC3 | `cargo test -p omega --lib msl::tests::bf8_e5m2_operand_read_decodes_each_byte -- --exact && cargo test -p omega --lib msl::tests::bf4_e2m1_operand_read_selects_each_nibble -- --exact && cargo test -p omega --lib metal::prepare_uniforms_pack::unsupported_packed_codec_tests::unsupported_packed_codec_stays_not_lowerable -- --exact` | 3 tests pass, 0 fail; flattened BF source preserves odd-row nibble addressing and Iq4Nl remains typed as NotLowerable | [x] | BF8 byte read, BF4 low/high nibble selection, and Iq4Nl typed rejection each passed their exact focused test. |
| 3 | Execute both packed formats through the computed Metal gather | AC2 | `test "$(uname -s)" = Darwin && cargo test -p omega --test packed_brain_float_moe_metal packed_computed_gather_matches_cpu -- --exact` | 1 test passes, 0 fails on macOS with an available Metal device; corrected native stride `[0,1,3]` maps packed `p = expert_base + output * in_dim + input`, and both Metal results equal CPU `[-4.5,-6]` for odd-width route `[1]` | [x] | The device test compiled and ran BF8 and BF4 kernels using raw packed bytes; CPU and Metal both yielded `[-4.5,-6]`, with corrected native strides `[0,1,3]`. |

## resume

Last completed slice: 3; all admitted slices have focused validation records
Next action: admit and implement Metal execution of the differentiated packed MoE graph, then carry its compact gradients through the existing sparse FP32 master update
Open question, if any: GPU execution evidence currently covers packed forward only

## struck

-
