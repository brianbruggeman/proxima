# Packed BF4/BF8 MoE CPU training step -- slices

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Admit the packed training-step contract | AC1-AC3 | `rg -c '^VERDICT: ADMIT$' proxima-tensor/specs/packed_brain_float_moe_training/ADMISSION.md` | one admitted revision | [x] | Auditor admitted the fixed fixture, dispatch-span oracle, FP32 optimizer outputs, and test-to-slice mapping. |
| 2 | Build the packed gathered CPU forward, sparse STE gradients, and FP32 master update | AC1-AC3 | `cargo test -p proxima-autograd --features instrument,proxima-tensor/instrument --test packed_brain_float_moe_training packed_gathered_step_updates_only_selected_master_rows -- --exact`; `cargo test -p proxima-autograd --features instrument,proxima-tensor/instrument --test packed_brain_float_moe_training packed_bf4_bf8_updates_match_fp32_ste_reference -- --exact`; `cargo test -p proxima-autograd --features instrument,proxima-tensor/instrument --test packed_brain_float_moe_training packed_training_payload_record_is_complete -- --exact` | 3 tests pass, 0 fail; exactly two records contain the pinned payloads, route-ordered dispatch spans, Adam state, seed, and CPU label | [x] | All three commands passed separately; AC3 emitted two persistent records with route-ordered dispatch spans. |

## resume

Last completed slice: 1
Next action: design the next admitted step from this CPU composition toward a trainable MoE model
Open question, if any: none; use the fixed CPU fixture and route pattern in the spec
