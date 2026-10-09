# BF4/BF8 MoE training discovery in Proxima -- slices

Each slice: one commit, one behaviour change, one validation command, under ~30 minutes.
Update the checkbox and the note IN THE SAME COMMIT as the slice.

| # | slice | discharges | validation command | expected | done | note |
|---|---|---|---|---|---|---|
| 1 | Audit the BF8/BF4 format contract, gradient rule, and pilot acceptance criteria | AC1-AC5 | `rg -c '^VERDICT: ADMIT$' proxima-tensor/specs/low_precision_moe_training/ADMISSION.md` | 1 admitted revision recorded | [x] | ADMIT record captures the auditor's count-completeness disposition and AC4 payload requirements. |
| 2 | Add explicit BF8 E5M2 and BF4 E2M1 scalar codecs in the existing allocation-free quant module | AC1 | `cargo test -p proxima-gguf --test bf8_e5m2_bf4_e2m1_encoding_contract bf8_e5m2_bf4_e2m1_encoding_contract -- --exact` | 1 passed, 0 failed; 272 raw encodings plus 4 signed midpoint cases and one BF4 pack/unpack pair match the pinned format tables and packing rule | [x] | The contract test passed; `cargo check -p proxima-gguf --no-default-features --features alloc` also passed after replacing std-only `powi` with bit-constructed power-of-two scales. These scalar primitives remain separate from GGUF's block-scaled MXFP4/NVFP4 tensor codecs. |
| 3 | Apply the specified fake-quantized weight view and straight-through gradient in the existing autograd program | AC2 | `cargo test -p proxima-autograd --test low_precision_weight_gradient low_precision_weight_gradient_matches_f32_reference -- --exact` | 1 passed, 0 failed; the same FP32 master tensor is updated while the forward consumes BF8/BF4-rounded values, FP32 accumulation is used, and Adam moments/update arithmetic remain FP32 | [x] | The hand-derived loss/gradient, FP32 graph dtypes, Adam moments, and master update assertions passed for both format views. |
| 4 | Compose the low-precision weight view with the fixed-router sparse MoE training fixture and paired-arm harness | AC3-AC4 | `cargo test -p proxima-autograd sparse_moe_training_step_updates_only_routed_experts && cargo test -p proxima-autograd --test low_precision_moe_comparison low_precision_moe_fixed_payload_comparison -- --exact` | 2 passed, 0 failed; exact hand-example route/gradient holds and 9 arm-seed records each contain 64 steps with 8 training/held-out input IDs, selected expert IDs, targets, losses, and finite/non-finite counts; zero tuning trials | [x] | Both focused tests passed; the report test checks every fixed token and route payload, all sample counts, and identical initial parameters. |
| 5 | Run the preregistered 64-step, 3-seed tiny pilot and retain payloads for each arm | AC4-AC5 | `cargo run -p proxima-autograd --example low_precision_moe_training_pilot -- --fixture tiny --steps 64 --seeds 3 --report json | jq . > proxima-tensor/specs/low_precision_moe_training/results/tiny-pilot.json` | 1 report, 9 arm-seed records, 576 step/time/throughput records each retaining 8 token IDs, 8 selected expert IDs, losses, and finite/non-finite counts, plus 9 resident-memory samples | [x] | AC4 reads the retained report; focused checks emitted 4 passing tests, and the alloc-only no-default-features check completed. The report has per-arm raw losses and resource samples; near-1B scale cost remains unmeasured. |

## resume

Last completed slice: 5; slices 1-5 have retained admission, code, checks, and pilot payload
Next action: none in the admitted tiny-pilot scope
Open question, if any: none; this spec fixes ACE-style BF8 E5M2 and BF4 E2M1 for the first CPU/reference experiment

## struck

-
