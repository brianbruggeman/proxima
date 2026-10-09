# Metal packed BF4/BF8 MoE backward step

status: admitted
owner: brian
created: 2026-10-09

## problem

Omega can run the packed BF4/BF8 computed-gather forward on Metal, and
proxima-autograd can construct compact expert-gradient contributions, but
there is no single Metal execution that evaluates the differentiated packed
MoE graph and feeds those contributions into the existing sparse FP32 master
update.

## refutation condition

If Metal cannot evaluate the differentiated form of the fixed CPU training
fixture from the same packed byte stack and return the same predictions,
loss, and compact gathered-gradient payload as CPU, then Metal is not yet a
backend for this MoE training step.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Differentiate the admitted packed CPU training graph and execute its forward and gathered-gradient outputs on Metal using direct BF8 E5M2 or BF4 E2M1 packed `QuantizedBlock` bytes plus the computed route, with FP32 activations and accumulation. | yes |
| R2 | For both codecs, Metal's predictions, scalar loss, and compact `[token,input,output]` expert-gradient values match the CPU execution of the same `Differentiated.program` and same packed bytes. | yes |
| R3 | Feed the Metal compact gradient payload through the existing host `dedupe_and_sum_rows` and FP32 Adam update; retain equality with the admitted independent CPU STE reference for repeated routes and the untaken expert. | yes |
| R4 | Retain codec, packed bytes, routes, predictions, loss, compact and coalesced gradients, updated masters, and backend labels in the training result record. | yes |

## architecture

Reuse the exact three-expert fixture, route list, activations, targets, and
expected predictions and updates from
`proxima-tensor/specs/packed_brain_float_moe_training/SPEC.md`. Differentiate
the existing graph once, request its `loss`, `predictions`, and
`GatheredContribution.values` as outputs, then prepare and execute that
`Differentiated.program` on Metal with the contiguous packed `QuantizedBlock`
stack and computed route as separate graph inputs. The Metal path reads the
original packed span through its packed operand binding; it does not use the
CPU `ExpertSource` abstraction or an expanded FP32 weight table. CPU evaluates
the same differentiated program and binds per-expert entries with
`ExpertSource` as the numerical oracle. For each codec compare the complete compact
gradient rows before coalescing, then run the existing host sparse coalescer,
transpose compact `[input,output]` rows into master `[output,input]` rows, and
apply the existing Adam implementation only to selected rows. The retained
FP32 master weights and optimizer state remain caller-owned host buffers in
this slice; each next step re-encodes them into packed expert bytes.

This is a hybrid Metal forward/backward plus host sparse optimizer step. It
does not move gathered-gradient coalescing or Adam to the GPU. This distinction
preserves the existing `GatheredContribution` API and its sparse route-sized
work instead of materializing an expert-count-sized scatter. A later design
must measure host transfer and optimizer costs before replacing this boundary.

Use the admitted fixed payload: three experts with two-by-two native
`[output,input]` weights, routes `[1,1,0]`, activations
`[[1,0],[1,0],[0,1]]`, targets `[[0,0],[1,0],[0,0]]`, and batch loss
`sum(0.5 * (prediction-target)^2)`. Both packed codecs must produce
predictions `[[2,-1],[2,-1],[0,1]]`, loss `4.0`, and canonical coalesced
gradients `e0=[0,0,0,1]`, `e1=[3,0,-2,0]`, `e2=[0,0,0,0]`. Updated master
rows and Adam state must match the exact FP32 arrays in the CPU training
spec. The 12 compact gradient values, in token then `[input,output]` order,
are `[2,-1,0,0]`, `[1,-1,0,0]`, `[0,0,0,1]`. Compare GPU floating results
with absolute tolerance `1e-6` and retain
the complete observed payload on mismatch. No throughput or model-scale
claim is part of this fixture.

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1-R2 | `test "$(uname -s)" = Darwin && cargo test -p omega --test packed_brain_float_moe_training_metal metal_differentiated_packed_moe_matches_cpu -- --exact` | 1 macOS Metal device test passes; for each codec, Metal and CPU match predictions `[[2,-1],[2,-1],[0,1]]` and loss `4.0`; all 12 pre-coalescing compact-gradient values in `[token,input,output]` order equal `[2,-1,0,0,1,-1,0,0,0,0,0,1]`; source asserts direct packed decoders and no expanded FP32 Metal expert table |
| AC2 | R3 | `test "$(uname -s)" = Darwin && cargo test -p omega --test packed_brain_float_moe_training_metal metal_compact_gradient_drives_sparse_master_update -- --exact` | 1 test passes for both codecs; route `[1,1,0]` coalesces duplicate expert-1 rows, updates only experts 0 and 1, and matches the full independent CPU reference for gradients, master weights, and Adam moments within absolute tolerance `1e-6` |
| AC3 | R4 | `RUSTC_WRAPPER= cargo test -p omega --test packed_brain_float_moe_training_metal metal_training_payload_record_is_complete -- --exact --nocapture > /tmp/metal-packed-training-test.log && python3 -c 'from pathlib import Path; import json; log=Path("/tmp/metal-packed-training-test.log").read_text(); marker="METAL_TRAINING_JSON="; start=log.index(marker)+len(marker); records, _=json.JSONDecoder().raw_decode(log[start:].lstrip()); assert len(records)==2; out=Path("proxima-tensor/specs/metal_packed_brain_float_moe_training/results/metal-packed-training.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(records,indent=2)+"\n")' && jq -e 'length == 2 and all(.[]; .backend == "metal" and .cpu_reference.backend == "cpu" and .scalar_reference.backend == "scalar")' proxima-tensor/specs/metal_packed_brain_float_moe_training/results/metal-packed-training.json` | 1 test passes; extraction writes the test's actual JSON stdout to `results/metal-packed-training.json`, which contains exactly 2 records (one per codec). Each has top-level Metal outputs, nested `cpu_reference` from the differentiated CPU evaluator, and nested `scalar_reference` from the independent scalar FP32 calculation; both references include predictions, loss, compact/coalesced gradients, updated masters, and Adam moments |