# BF4/BF8 MoE training discovery in Proxima

status: admitted
owner: brian
created: 2026-10-09

## problem

Proxima has no verified training path for a from-scratch MoE with BF8 or BF4 model weights; this work will measure whether either explicitly specified brain-float weight format can reduce held-out language-model loss against an FP32 MoE control on the same small corpus under the same fixed training configuration, zero hyperparameter-search trials, and final-step checkpoint rule while preserving the declared format and MoE routing semantics.

## refutation condition

If neither BF8 nor BF4 produces finite, improving held-out loss on the fixed pilot while the FP32 control does, the proposed low-precision weight path is refuted for this pilot. If the FP32 control fails to improve held-out loss, the pilot is invalid and the training fixture must be repaired before comparing formats. Any result from this pilot remains a plumbing result and cannot support a near-1B quality or performance claim.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Define the exact BF8 E5M2 and BF4 E2M1 encodings, conversion rules, non-finite handling, saturation, and BF4 byte packing used by the experiment; label them as the selected ACE-style formats, not universal BF4/BF8 encodings. | yes |
| R2 | Run forward and backward training with low-precision model-weight operands, FP32 accumulation, and FP32 master parameters and optimizer state; record the exact backward rule for quantized operands. | yes |
| R3 | Train a sparse MoE whose router, expert dispatch, and expert gradients use Proxima's existing MoE program semantics. | yes |
| R4 | Compare BF8, BF4, and FP32 control arms on identical initialization, token batches, optimizer steps, and held-out payloads; retain per-step losses, token inputs, routing assignments, and non-finite counts. Use zero hyperparameter-search trials for every arm and select the final step checkpoint for all arms. | yes |
| R5 | Report measured CPU wall time, tokens per second, and peak resident memory for the tiny pilot, with raw per-seed/per-step samples and an explicit statement that these do not predict near-1B training cost. | yes |

## architecture

The first experiment keeps FP32 master parameters and Adam state. Each training forward uses a low-precision projection-weight view: BF8 E5M2, BF4 E2M1, or FP32 control. Activations and accumulation remain FP32. BF8 conversion uses sign:exponent:fraction widths 1:5:2, exponent bias 15, subnormals, and round-to-nearest ties-to-even. The selected encoder follows Intel ACE v1.15 §9.2 SAT behavior: finite overflow and ±infinity inputs saturate to signed max finite (±57344); NaN input encodes as canonical positive quiet NaN `0x7e`, following the canonical NaN conversion in §16.1. The selected decoder follows the E5M2 special-value helper in §16.3: exponent 31 with fraction 0 is signed infinity (`0x7c`/`0xfc`), and exponent 31 with nonzero fraction is NaN. This explicitly chooses the helper's special-value behavior where it differs from the summary table in §2.4.1. BF4 conversion uses sign:exponent:fraction widths 1:2:1, exponent bias 1, finite values through ±6, the E2M1 subnormal ±0.5, round-to-nearest ties-to-even, and saturation to signed 6 for finite overflow. BF4 NaN and infinity inputs return a typed conversion error because the selected format has no such encodings. BF4 packs the first value in the low nibble and the second value in the high nibble. No block scaling is applied in the first scalar-format arm. The backward path uses the straight-through estimator: the forward reads the dequantized low-precision view, while `d(Q(w))/dw` is 1 for every FP32 master weight. The CPU evaluator is the first reference path. Existing GPU training evidence covers dense graphs; MoE gathered gradients are currently accumulated host-side, outside the GPU-executed differentiated program, so the pilot must not claim GPU MoE training.

MoE routing and token dispatch use the existing program representation; the experiment records each token's selected expert IDs so a loss change can be related to actual routed payloads. The pilot route is top-1 with a fixed router, so it measures low-precision expert-weight learning and sparse dispatch; trainable router learning is a later slice with its own objective and ablation.

### hand-derived update

For one token with input `x = 1`, the fixed router emits logits `[2, -1]`; top-1 selects expert 0. The selected expert has master weight `w0 = 0.5`; expert 1 has `w1 = -0.5`. Both values are exactly representable in the specified BF8 E5M2 and BF4 E2M1 formats, so each low-precision forward returns `y = 0.5`. With squared error `L = 0.5 * (y - 1)^2`, the loss is `0.125`; the straight-through gradient for `w0` is `(y - 1) * x = -0.5`; `w1` receives no gradient because its expert was not selected. This example specifies the sparse update boundary, while the held-out pilot measures whether the format helps or harms across varied weights.

The pilot uses a fixed vocabulary of 4 IDs, a fixed one-hot input representation, and two expert matrices of shape `[4, 4]` with no bias; the selected expert's four logits are the next-token scores. A fixed top-1 router sends even input IDs to expert 0 and odd input IDs to expert 1. Thus it has exactly 32 trainable parameters, all in expert matrices, and no trainable router or embedding parameters. Training input IDs are `[0, 1, 2, 3, 0, 1, 2, 3]` and next-token targets are `[1, 2, 3, 0, 1, 2, 3, 0]`. Held-out input IDs are `[0, 0, 1, 1, 2, 2, 3, 3]` and targets are `[0, 1, 1, 2, 2, 3, 3, 0]`. Initialize expert weights by taking 32 successive outputs from `proxima_tensor::test_support::Lcg(seed).next_unit()` and multiplying each by `0.5`, for seeds `[17, 29, 43]`; `next_unit` is the existing 64-bit LCG with multiplier `6364136223846793005`, increment 1, top 32-bit draw, mapped to `[-1, 1)`. Use Adam at learning rate `0.001`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, full eight-token training batch, 64 training steps, and no hyperparameter search in any arm. All arms start from byte-identical FP32 master tensors and consume batches in the same order. The report retains initial and final master tensors; the final step is the only selected checkpoint. This synthetic fixture is a plumbing control, not a corpus-quality evaluation or evidence that a near-1B model can be trained within any particular time, hardware, or budget.

For each arm and step, the training objective is mean next-token cross-entropy over the eight training tokens: for token `i`, `logsumexp(logits_i) - logits_i[target_i]`, averaged by dividing the sum by 8. After each optimizer update, held-out loss is computed by the same formula over the eight held-out tokens, also divided by 8, with no optimizer update on held-out payloads. Both use FP32 logits and stable log-sum-exp. The final-step held-out loss is the sole comparison value; per-step values are retained to show the trajectory, not used for checkpoint selection.

Each step's `non_finite_count` is the count of non-finite FP32 scalar entries across the master weights, averaged expert gradients, training loss, and held-out loss after the update. CPU wall time covers all eight training forward/backward evaluations, gathered-gradient reduction, the Adam update, and all eight held-out evaluations; tokens per second is eight training tokens divided by that whole-step wall time. Each arm-seed runs in its own worker process, so its `peak_rss_bytes` is the independent process high-water mark rather than a cumulative value shared across arms.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| First low-precision arm | BF8 E5M2 | It preserves more exponent range than the E2M1 BF4 comparison and is the lower-risk first measurement; the result still decides whether BF8 is useful. |
| Master parameter and optimizer precision | FP32 | Updating only rounded BF8/BF4 values can erase small optimizer updates; retaining master state isolates operand-format effects. |
| Initial activation and accumulation precision | FP32 | The first comparison should isolate weight-operand precision instead of conflating weight and activation quantization. |
| Model scale | tiny executable pilot before any scale-up | A toy fixture can expose format, gradient, and routing failures without treating a parameter-count target as a training-capability result. |
| Discovery outcome | held-out comparison with FP32 control, equal tuning budget, and fixed checkpoint selection rule | There is no known answer for Proxima's BF4/BF8 training behavior to reproduce. |

## acceptance criteria

Each criterion is a planned command and count-bearing expected result. The commands become admissible for implementation only after the spec is audited and their test names are added. The held-out corpus split, seed, tuning budget, and checkpoint selection rule must be pinned before the first comparison run.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1 | `cargo test -p proxima-gguf bf8_e5m2_bf4_e2m1_encoding_contract` | 1 passed, 0 failed; all 256 BF8 patterns decode by the explicit E5M2 rule in Intel ACE v1.15 §16.3, with its §2.4.1 table used only for finite values, and all 16 BF4 nibbles decode against §§2.4.2/2.5; two signed BF8 midpoint inputs (`±1.125`) round to even `0x3c`/`0xbc`, two signed BF4 midpoint inputs (`±1.25`) round to even `0x2`/`0xa`, BF8 finite overflow and ±infinity input saturate to signed max finite under §9.2 SAT, BF8 NaN input encodes as `0x7e` per §16.1, BF4 non-finite inputs error, finite overflow saturates to signed 6, and packing `[1.0, 0.5]` yields byte `0x12` (first value low nibble) and decodes back to the same pair |
| AC2 | R2 | `cargo test -p proxima-autograd low_precision_weight_gradient_matches_f32_reference` | 1 passed, 0 failed; the hand-derived spec example is the independent oracle: loss `0.125`, selected expert gradient `-0.5`, unselected expert gradient `0.0`; BF4/BF8 master updates use identity STE, forward accumulation is FP32, and Adam first/second moments and update arithmetic are FP32 |
| AC3 | R3 | `cargo test -p proxima-autograd sparse_moe_training_step_updates_only_routed_experts` | 1 passed, 0 failed; for the hand-derived one-token example, route is `[0]`, loss is `0.125`, expert-0 gradient is `-0.5`, and expert-1 gradient is `0.0` |
| AC4 | R4 | `cargo test -p proxima-autograd low_precision_moe_fixed_payload_comparison` | 1 passed, 0 failed; 9 arm-seed records (3 formats × 3 seeds); each has the byte-identical initial FP32 parameters and exactly 64 step records with 8 training input IDs, 8 selected expert IDs, 8 held-out input IDs, 8 held-out expert IDs, losses, and finite/non-finite counts; zero tuning trials and final-step checkpoint selection |
| AC5 | R5 | `cargo run -p proxima-autograd --example low_precision_moe_training_pilot -- --fixture tiny --steps 64 --seeds 3 --report json` | 1 JSON report with 9 arm-seed records, each retaining 32 initial and final master parameters, 64 step records, 64 CPU wall-time samples and 64 tokens-per-second samples; report totals are 576 step/time/throughput samples and 9 independent worker-process peak-resident-memory samples; each model reports 32 parameters and near-1B scale cost as unmeasured |

## out of scope

- claiming that the pilot establishes the quality or cost of training a model near one billion parameters
- using BF4 or BF8 for optimizer state, activations, gradients, or router logits in the first experiment
- distributed expert parallel training, multi-node execution, tokenizer research, and large-corpus acquisition
- substituting MXFP4, NVFP4, integer Q4/Q8, or another block-scaled encoding for the named BF encodings
- publishing or releasing a checkpoint

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| BF4/BF8 names do not identify the same encoding or scale layout across vendors | high | bytes or arithmetic could be mislabeled as brain-float behavior | pin an exact bit-layout and scaling contract in R1 before implementation |
| Proxima's autograd representation cannot express the chosen low-precision forward view and gradient rule without a semantic extension | medium | the pilot could measure a custom host path instead of Proxima training | trace the current differentiation and evaluation path before selecting the smallest API seam |
| A toy corpus and model do not predict large-model quality or performance | high | a successful pilot could be overread as a 1B training result | retain the scale boundary and treat the pilot only as a path gate |
| Expert routing collapses or sends too few tokens to an expert | medium | aggregate loss may hide missing expert learning | retain per-step token-to-expert assignments and per-expert update counts |

## context

- `proxima-tensor/src/dtype.rs`
- `proxima-autograd/src/train.rs`
- `proxima-autograd/examples/low_precision_moe_training_pilot.rs`
- `proxima-autograd/src/adjoint.rs`
- `proxima-tensor/src/op.rs`
- `proxima-tensor/src/bind/gdn_moe_fusion_apply.rs`
- `proxima-gguf/src/types.rs`
- `proxima-gguf/src/quant/dispatch.rs`
- `specs/attention-storage-format-parity/SPEC.md`
- `specs/omega-programmable/SPEC.md`
- AMD rocWMMA documentation for `bf8` as E5M2
- Intel ACE v1.15 specification, §§2.4.1, 2.4.2, 2.5, 9.2, 16.1, and 16.3: BF8 is E5M2 and BF4 is E2M1; the selected E5M2 decoder follows the §16.3 helper for special values where it differs from the §2.4.1 summary table, and the encoder follows §9.2 SAT; BF4 has no infinity or NaN encodings
- AMD Composable Kernel `float8.hpp`: `bf8_t` aliases `float8_e5m2_t`
- Exact references: `https://x86ecosystem.org/wp-content/uploads/2026/06/ACE_v1_Specification_public_1_15.pdf` and `https://rocm.docs.amd.com/projects/composable_kernel/en/docs-7.0.1/doxygen/html/float8_8hpp_source.html`
- `omega/tests/training_step_parity.rs`: existing GPU backward/optimizer parity is for dense graphs; its module documents gathered gradients as host-side
- `proxima-tensor/src/test_support.rs:73-81`: deterministic LCG recurrence and `[-1, 1)` conversion used for pilot master weights
- Hand-derived update above is the independent numerical reference for the pilot STE path
- Discovery-loop protocol: equal-tuning FP32 baseline, fixed held-out set, ablations, raw payloads, and scale gate
