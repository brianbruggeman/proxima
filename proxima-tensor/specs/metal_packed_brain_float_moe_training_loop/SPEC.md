# Two-step Metal packed BF4/BF8 MoE training loop

status: admitted
owner: brian
created: 2026-10-09

## problem

The admitted Metal training fixture executes one BF4/BF8 forward/backward
step, but it resets Adam state and starts from the same FP32 masters each
time. It does not establish that a caller can carry optimizer state, update
the masters, repack the new values, and train on the next Metal step.

## refutation condition

If step two does not consume bytes freshly encoded from step one's updated
FP32 masters and carried Adam state, or its Metal outputs diverge from the
independent scalar reference, the path has not demonstrated a repeated
training loop.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Run two consecutive steps for BF8 and BF4 with FP32 masters and Adam moments carried across steps; Adam bias correction uses global steps one and two. | yes |
| R2 | Encode each step's current master rows into the selected packed format immediately before that step's Metal execution; step two's bytes differ from the initial bytes and match scalar encoding of the step-one master state. | yes |
| R3 | At both steps, Metal predictions, loss, compact gathered gradients, sparse coalesced gradients, updated masters, and Adam moments match CPU and an independent scalar FP32 Adam reference. | yes |
| R4 | Retain exactly four step records, one per `(codec, step)`, with input and output state, packed bytes, routes, activations, targets, forward values, compact/coalesced gradients, backend labels, and reference values. | yes |

## architecture

Extend the admitted fixture and Metal execution in
`proxima-tensor/specs/metal_packed_brain_float_moe_training/SPEC.md` to two
consecutive updates using the same batch and routes. Keep master parameters
and Adam first/second moments in FP32. Configure Adam with learning rate
`0.5`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`. Start at step 1 with the
existing master rows, zero moments, routes `[1,1,0]`, activations
`[[1,0],[1,0],[0,1]]`, and targets `[[0,0],[1,0],[0,0]]`. After each
coalesced sparse gradient update, carry all selected moments and the global
step number forward, encode the updated masters into the selected packed
format, and execute the next differentiated graph on Metal from those bytes.
The CPU reference runs the same update sequence from an independent scalar
implementation and independently encodes each step's packed rows.

Step one's predictions, loss, and gradients are those pinned in the admitted
one-step fixture. Its updated master rows are approximately
`e0=[1.1,0,0,0.6]`, `e1=[1.7,0,-0.6,0]`, `e2=[3.3,4,5,6]` within absolute
tolerance `1e-6`. The step-two packed bytes, in native `[expert,output,input]`
order, are BF8 `[60,0,0,57,63,0,185,0,67,68,69,70]` and BF4
`[2,16,3,9,101,118]`. These are direct encodings of step one's updated
masters, not the initial stack. On step two BF4 predicts
`[[1.5,-0.5],[1.5,-0.5],[0,0.5]]` with loss `1.625`; BF8 predicts
`[[1.75,-0.625],[1.75,-0.625],[0,0.625]]` with loss `2.3984375`. The full
step-two compact gradient is pinned as BF4
`[[1.5,-0.5,0,0],[0.5,-0.5,0,0],[0,0,0,0.5]]` and BF8
`[[1.75,-0.625,0,0],[0.75,-0.625,0,0],[0,0,0,0.625]]`, in token then
`[input,output]` order. Require step two's Adam moments to differ from a
zero-state step-two calculation, proving moment carry-forward.

Do not claim convergence, language-model quality, model-scale speed, or
training cost from this two-step deterministic fixture.

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1-R3 | `test "$(uname -s)" = Darwin && cargo test -p omega --test packed_brain_float_moe_training_metal metal_training_loop_reencodes_masters_and_carries_adam_state -- --exact` | 1 macOS Metal device test passes; both codecs execute two consecutive steps; step-one updated masters match the pinned rows, step-two packed bytes match the exact BF8/BF4 arrays above and differ from initial bytes, step-two predictions/loss/compact gradients match the pinned values, and all GPU/CPU/reference arrays match within `1e-6` |
| AC2 | R4 | `test "$(uname -s)" = Darwin && RUSTC_WRAPPER= cargo test -p omega --test packed_brain_float_moe_training_metal metal_training_loop_records_each_step -- --exact --nocapture > /tmp/metal-packed-training-loop-test.log && python3 -c 'from pathlib import Path; import json; log=Path("/tmp/metal-packed-training-loop-test.log").read_text(); marker="METAL_TRAINING_LOOP_JSON="; start=log.index(marker)+len(marker); records,_=json.JSONDecoder().raw_decode(log[start:].lstrip()); assert len(records)==4; out=Path("proxima-tensor/specs/metal_packed_brain_float_moe_training_loop/results/metal-packed-training-loop.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(records,indent=2)+"\n")' && jq -e 'length == 4 and ([.[] | .codec + ":" + (.step|tostring)] | unique | length == 4) and all(.[]; .backend == "metal" and .cpu_reference.backend == "cpu" and .scalar_reference.backend == "scalar")' proxima-tensor/specs/metal_packed_brain_float_moe_training_loop/results/metal-packed-training-loop.json` | 1 test passes; serialized JSON contains exactly four records for BF8/BF4 x steps 1/2. Each top-level record has `codec`, `step`, `backend: "metal"`, `master_input`, `moment_one_input`, `moment_two_input`, `routes`, `activations`, `targets`, `packed_bytes`, `predictions`, `loss`, `compact_gradient_rows`, `coalesced_gradient_rows`, `updated_master_rows`, `first_moment_rows`, `second_moment_rows`, nested `cpu_reference` with the same output fields and `backend: "cpu"`, and nested `scalar_reference` with `predictions`, `loss`, `compact_gradient_rows`, `coalesced_gradient_rows`, `updated_master_rows`, `first_moment_rows`, and `second_moment_rows` |

## out of scope

- production trainer API or generic data loader
- learned routing, corpus quality, convergence, and held-out language-model evaluation
- CUDA/WGSL execution, GPU coalescing, or GPU optimizer updates
- throughput, memory-cap, cost, or near-1B model claims
- checkpoint publishing or model release

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| moments are reset between steps or bias correction uses the batch index incorrectly | medium | step two does not represent continuation of step one | retain complete input/output moments and assert against the global-step scalar reference |
| step two reuses the initial packed bytes | medium | Metal backward can appear to train while forward weights never reflect updates | pin step-one packed bytes, require step-two bytes to equal fresh codec encoding and differ from initial bytes |
| discrete BF4/BF8 rounding masks an update in the fixture | medium | a test can pass without exercising repacking | choose learning rate `0.5` and pin step-two decoded predictions and packed bytes for both codecs |

## context

- `proxima-tensor/specs/metal_packed_brain_float_moe_training/SPEC.md`
- `proxima-tensor/specs/metal_packed_brain_float_moe_training/results/metal-packed-training.json`
- `omega/tests/packed_brain_float_moe_training_metal.rs`
- `proxima-autograd/src/optimizer.rs`
- `proxima-gguf/src/quant/bf4_e2m1.rs`
- `proxima-gguf/src/quant/bf8_e5m2.rs`
