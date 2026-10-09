# Metal packed pilot cross-entropy through all 64 steps

status: admitted
owner: brian
created: 2026-10-09

## problem

The packed CPU pilot executes 64 synthetic training steps, while Metal currently
executes only step one of its eight-token cross-entropy graph; therefore no
retained payload shows whether the same packed BF8/BF4 masters, sparse host
Adam updates, and re-encoded bytes carry across the pilot schedule on Metal.

## refutation condition

If any step for either codec fails to match the CPU packed evaluator and
independent scalar forward, gradient, optimizer, or held-out payload within
`1e-6`, or if a step consumes bytes that do not encode its recorded input
masters, the 64-step Metal training path does not reproduce the pilot loop.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Start each BF8 E5M2 and BF4 E2M1 arm from the exact seed-17 initial masters in `packed-tiny-pilot.json`; run global steps 1 through 64 with train token IDs `[0,1,2,3,0,1,2,3]`, target IDs `[1,2,3,0,1,2,3,0]`, and routes `[0,1,0,1,0,1,0,1]`. The held-out IDs are `[0,0,1,1,2,2,3,3]`, targets `[0,1,1,2,2,3,3,0]`, and routes `[0,0,1,1,0,0,1,1]`. | yes |
| R2 | Each step consumes direct packed expert bytes in a single Metal differentiated cross-entropy graph, exposes token logits/losses and compact `[8,4,4]` gathered gradients, then applies existing host sparse coalescing and FP32 Adam with carried moments and global step. | yes |
| R3 | Re-encode each updated master before the next forward. At every step compare Metal training outputs, compact/coalesced gradients and updated state to the CPU packed evaluator and an independent scalar reference within absolute tolerance `1e-6`; compare held-out logits and per-token losses after each update as well. | yes |
| R4 | Retain exactly 128 records: codecs `{Bf8E5M2,Bf4E2M1}` × steps `{1..64}` for seed `17`, including input state, packed bytes, fixed train and held-out inputs, all three backend outputs, gradients, moments, and update state. | yes |
| R5 | Keep the measured fixture at two experts, four input/output dimensions, four token IDs, and 32 parameters; report no large-model quality, convergence, throughput, memory-fit, or near-1B conclusion. | yes |

## architecture

Reuse the admitted graph, fixed routes and labels, scalar arithmetic, host
coalescer, and Adam boundary from
`metal_packed_pilot_moe_training/SPEC.md`. Reuse the same seed-17 initial
masters and fixed 64-step train/held-out split as
`packed_pilot_moe_training/SPEC.md`. Construct one Metal plan per codec for
the fixed graph and output nodes, then call `execute_plan_named` with the
current packed arena and current train or held-out batch; the plan API accepts
fresh named block data. Do not rebuild the graph or lower a new plan each step.
The CPU reference continues to use the packed `ExpertSource` evaluator. The
scalar reference independently decodes the step's bytes, computes the eight
routed token rows and losses, forms STE gradients, coalesces route rows, and
computes Adam from the carried master and moment inputs using learning rate `0.001`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, and global step `step`. After each host update,
encode the new FP32 master rows and use those exact bytes for the next training
step and the post-update held-out batch.

At step 1, both Adam moment inputs are exactly 32 zeros. For every later step,
the master, first-moment, and second-moment inputs exactly equal the previous
step's corresponding updated arrays. Training bytes at every step must equal
the selected codec encoding of `master_input`; held-out bytes must equal the
same codec encoding of `updated_masters`. The byte lengths are exactly 32 for
BF8 E5M2 and 16 for BF4 E2M1.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| Metal plan lifetime | one plan per codec reused across all steps and both fixed batches | per-step planning would repeatedly lower the same graph and obscure training-loop execution |
| optimizer placement | host sparse coalescing and host FP32 Adam | this slice measures the already admitted Metal gradient boundary; it does not add GPU scatter or optimizer semantics |
| arm set | seed 17, BF8 and BF4 | this is the exact seed already used by the one-step Metal pilot; additional seed coverage would multiply device runs without changing the state-carry contract |

Each record contains exactly `codec`, `seed`, `step`, `backend`,
`plan_build_count` (1 per codec), `plan_execution_count` (cumulative, equal to
`step × 2` per codec), `master_input` (32),
`first_moment_input` (32), `second_moment_input` (32), `train_packed_bytes`
(32 BF8 bytes or 16 BF4 bytes), `routes` (8), `token_ids` and `target_ids`
(each 8), `train_logits` (8×4), `train_token_losses` (8), `train_mean_loss`
(scalar), `compact_gradients` (8×16), `coalesced_gradients` (2×16),
`updated_masters` (32), `updated_first_moment` (32), `updated_second_moment`
(32), and `held_out_packed_bytes` with the same codec length,
`held_out_routes` (8), `held_out_token_ids` and `held_out_target_ids` (each 8),
`held_out_logits` (8×4), `held_out_token_losses` (8), and
`held_out_mean_loss` (scalar). `backend` is `metal`. Both `cpu_reference` and
`scalar_reference` contain exactly `backend`, `train_logits` (8×4),
`train_token_losses` (8), `train_mean_loss` (scalar), `compact_gradients`
(8×16), `coalesced_gradients` (2×16), `updated_masters` (32),
`updated_first_moment` (32), `updated_second_moment` (32), `held_out_logits`
(8×4), `held_out_token_losses` (8), and `held_out_mean_loss` (scalar), with
backend labels `cpu` and `scalar`. Record order is BF8 steps 1–64 followed by
BF4 steps 1–64. `validate_report.py` checks the exact 128 codec-step keys,
record and nested field sets, fixed IDs/routes, shapes, finite numeric leaves,
step-1 masters equal to the exact seed-17 pilot artifact, and two plan builds
with 128 executions per codec (one train and one held-out execution per step),
codec-specific packed lengths, byte-for-byte encoding of every training input
master and held-out updated master, zero step-1 moments, and exact master/moment
continuity between adjacent steps. The test assertions compare every Metal,
CPU, and scalar numeric payload within `1e-6`.

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1-R5 | `test "$(uname -s)" = Darwin && RUSTC_WRAPPER= cargo test -p omega --test packed_brain_float_moe_training_metal packed_pilot_cross_entropy_64step_records_payload -- --exact --nocapture > /tmp/metal-packed-pilot-64step.log && python3 -c 'from pathlib import Path; import json; log=Path("/tmp/metal-packed-pilot-64step.log").read_text(); marker="METAL_PACKED_PILOT_64STEP_JSON="; start=log.index(marker)+len(marker); records,_=json.JSONDecoder().raw_decode(log[start:].lstrip()); assert len(records)==128; out=Path("proxima-tensor/specs/metal_packed_pilot_moe_64step/results/metal-packed-pilot-64step.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(records,indent=2)+"\n")' && python3 proxima-tensor/specs/metal_packed_pilot_moe_64step/validate_report.py proxima-tensor/specs/metal_packed_pilot_moe_64step/results/metal-packed-pilot-64step.json` | 1 test passes; exactly 128 ordered records retain all 64 steps for both codecs; validator confirms state continuity and payload dimensions; every Metal/CPU/scalar training and held-out comparison is within `1e-6` |

## out of scope

- changing the model, token split, routes, labels, seed, optimizer, or learning rate
- learned-router training, GPU sparse coalescing, GPU Adam, or full-model kernels
- benchmarking throughput, peak memory, quality, convergence, or a one-billion-parameter run
- publishing or releasing a checkpoint

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| reused Metal plan retains stale packed input bytes | medium | later steps would not consume the just-updated masters | assert each record's byte arena against the independent scalar re-encoding and compare its forward payload |
| 64 sequential optimizer steps drift through a state-carry error | medium | a final-only comparison would hide the first divergent update | compare all three backends and moments at every global step and assert exact state continuity |
| repeated held-out evaluation changes training state | low | validation would advance an optimizer or reuse the wrong bytes | run held-out outputs only after the update and never feed their gradients into Adam |

## context

- `proxima-tensor/specs/metal_packed_pilot_moe_training/SPEC.md`
- `proxima-tensor/specs/metal_packed_pilot_moe_training/TASKS.md`
- `proxima-tensor/specs/metal_packed_pilot_moe_training/results/metal-packed-pilot-training.json`
- `proxima-tensor/specs/packed_pilot_moe_training/SPEC.md`
- `proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json`
- `proxima-autograd/examples/low_precision_moe_training_pilot.rs`
- `omega/tests/packed_brain_float_moe_training_metal.rs`
- `omega/src/backend.rs:1147-1154`
