# Packed-byte BF8/BF4 execution in the tiny MoE training pilot

status: proposed
owner: brian
created: 2026-10-09

## problem

The admitted 64-step tiny pilot compares BF8 E5M2 and BF4 E2M1 arms, but its
forward path calls `proxima_autograd::low_precision::weight_view`, which
materializes a decoded FP32 `Vec` before the model graph runs. Separate
fixtures prove packed-byte CPU and Metal execution on a three-expert squared
loss, but the 64-step cross-entropy pilot does not consume the packed bytes.
The pilot therefore does not yet measure the exact weight representation it
labels BF8 or BF4.

## refutation condition

If either low-precision arm's training and held-out forward still receives an
expanded decoded FP32 expert tensor, or if packed-byte execution disagrees with
the independent scalar decode reference on the same master rows, the pilot has
not been moved onto its declared packed representation.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Encode the existing two `[4,4]` expert master matrices into direct BF8 E5M2 bytes or BF4 E2M1 nibbles at each step boundary; bind those bytes to Proxima's existing CPU quantized gather path with explicit per-expert shape and codec. | yes |
| R2 | Run the admitted tiny pilot's fixed two-expert route and four-class softmax cross-entropy, gathered gradients, duplicate-route reduction, and FP32 Adam updates from packed-byte forwards. Use FP32 master weights and optimizer state; encode updated masters again before the next forward. | yes |
| R3 | For each codec and step, compare logits, token loss, gathered gradient rows, coalesced gradients, updated masters, and Adam moments against an independent scalar BF decoder/forward/STE/Adam implementation within absolute tolerance `1e-6`. FP32 control remains unchanged. | yes |
| R4 | Retain every step's input master rows, packed bytes, token IDs, selected expert IDs, targets, logits, per-token and mean losses, compact and coalesced gradients, output master rows, moments, backend labels, and independent scalar reference. Keep exactly the Cartesian product of formats `{bf8_e5m2, bf4_e2m1, fp32}` and seeds `{17, 29, 43}`, with all 64 steps per arm. | yes |
| R5 | The trained artifact remains a synthetic 2-expert, 4-token, 32-parameter pilot; make no corpus-quality, convergence, GPU-training, or near-1B cost claim from it. | yes |

## architecture

Replace only the BF8/BF4 arm's dequantized `weight_view` binding in
`proxima-autograd/examples/low_precision_moe_training_pilot.rs`. Preserve its
fixed inputs, labels, router, seeds, optimizer settings, token order, held-out
split, and final-step checkpoint rule from
`proxima-tensor/specs/low_precision_moe_training/SPEC.md`. At each training step, encode the step's `master_input` expert rows into a
contiguous codec-tagged arena; at held-out evaluation, encode the post-update
`updated_master_rows` into a second arena. Construct two borrowed
`ExpertEntry` values with `[out_dim=4, in_dim=4]`; bind an `ExpertSource` to
the expert-weight graph input; and run the existing CPU quantized evaluator.
The graph's differentiated gathered contribution remains compact by token and
is coalesced through `dedupe_and_sum_rows`; Adam continues to update the FP32
masters and moments. The next training step reuses bytes produced from the prior update only after
re-encoding and recording them for the new `master_input`. No decoded FP32
expert matrix is passed to a BF8/BF4 forward.

The independent scalar reference decodes retained bytes using the scalar BF8
or BF4 decoder and evaluates the same token's `[4,4]` selected expert matrix
with explicit loops, then computes stable softmax cross-entropy, identity STE
gradients, route-row coalescing, and FP32 Adam independently of Proxima's
quantized evaluator, autograd, sparse coalescer, and optimizer. Compare actual
per-step payload arrays; aggregate losses alone are insufficient evidence.

Retain the existing FP32 arm and all nine format/seed records. For each BF8 or
BF4 step record both the byte arena and per-expert byte spans, along with the
complete state and reference outputs. Every step has `master_input`, `first_moment_input`, `second_moment_input`, `train_packed_bytes`, `train_expert_byte_spans`, `held_out_packed_bytes`, `held_out_expert_byte_spans`, training and held-out token IDs/routes/targets, training and held-out logits and per-token losses, mean train and held-out losses, compact and coalesced gradients, updated master rows, updated first and second moment rows, `backend`, and a `scalar_reference` containing all corresponding numeric outputs plus the optimizer input state. Training bytes must independently re-encode `master_input`; held-out bytes must independently re-encode `updated_master_rows`; both span tables must have expert IDs 0 and 1 with exact offsets and codec-derived byte lengths. The FP32 arm records empty train/held-out packed bytes and spans, and `backend: "cpu-fp32"`; packed arms label evaluator backend `"cpu-packed"`. `validate_report.py` rejects any arm set other than the exact 3×3 product, any step count or required-field mismatch, any wrong byte/span length or span offset, any arena that differs from an independent encoding of its corresponding master state, and any scalar-reference numeric leaf differing by more than `1e-6`; it writes the validated report at
`proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json`.

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1-R3 | `cargo test -p proxima-autograd --test packed_low_precision_moe_training packed_pilot_step_matches_scalar_reference -- --exact` | 1 test passes; BF8, BF4, and FP32 one-step controls run the same fixed tokens and routes; the BF8/BF4 inputs are packed byte blocks, no decoded expert tensor is bound, all logits/losses/compact and coalesced gradients/updated masters/moments match scalar arrays within `1e-6`, and both routed expert rows receive the scalar-reference FP32 master and moment updates |
| AC2 | R1-R5 | `cargo run -p proxima-autograd --example low_precision_moe_training_pilot -- --fixture tiny --steps 64 --seeds 3 --report json > /tmp/packed-tiny-pilot.json && python3 proxima-tensor/specs/packed_pilot_moe_training/validate_report.py --input /tmp/packed-tiny-pilot.json --output proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json && jq -e '(.arms|length)==9 and ([.arms[].steps[]]|length)==576' proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json` | 1 report contains the exact 3-format × 3-seed set and 576 step records; validator checks all required state/payload fields, byte geometry, and every numeric output against its independent scalar reference within `1e-6`; jq confirms 9 arms and 576 steps |

## out of scope

- GPU execution of the 64-step cross-entropy pilot
- changing the model, training corpus, router, data split, seeds, optimizer, or tuning budget
- training or evaluating a one-billion-parameter model
- claiming quality from the synthetic four-ID corpus or claiming convergence from 64 steps
- checkpoint publishing or release

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| the CPU quantized gather path may not expose gradients for the same graph input shape | medium | the pilot cannot use the existing compact autograd path unchanged | require the one-step parity AC first and keep the training update boundary explicit |
| separate `ExpertEntry` spans may reorder expert or output rows | medium | routes update the wrong master rows | retain codec, expert ID, shape, byte offset, and byte length in each step record |
| float summation order differs between packed evaluator and scalar reference | medium | exact float equality would reject equivalent arithmetic | compare full arrays with the pinned absolute tolerance and retain both values |

## context

- `proxima-tensor/specs/low_precision_moe_training/SPEC.md`
- `proxima-tensor/specs/low_precision_moe_training/results/tiny-pilot.json`
- `proxima-autograd/examples/low_precision_moe_training_pilot.rs`
- `proxima-autograd/src/low_precision.rs`
- `proxima-autograd/src/adjoint.rs`
- `proxima-autograd/src/sparse.rs`
- `proxima-autograd/src/optimizer.rs`
- `proxima-autograd/tests/packed_brain_float_moe_training.rs`
- `proxima-tensor/src/cpu/quantized_eval.rs`
- `proxima-tensor/src/cpu/run_reduce_scan.rs`
- `proxima-tensor/src/cpu/epilogue.rs`
- `proxima-tensor/specs/packed_pilot_moe_training/validate_report.py`
- `proxima-gguf/src/quant/bf4_e2m1.rs`
- `proxima-gguf/src/quant/bf8_e5m2.rs`
