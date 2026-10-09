# Metal packed-byte cross-entropy step for the tiny BF8/BF4 MoE pilot

status: admitted
owner: brian
created: 2026-10-09

## problem

The 64-step tiny BF8/BF4 MoE pilot now trains from packed bytes through the CPU
quantized gather evaluator and retains a per-step scalar reference. Metal has
separate packed BF8/BF4 forward and two-step squared-loss training fixtures,
but it has not executed the pilot's `[2,4,4]` two-expert graph, four-class
softmax cross-entropy, eight-token route batch, or its 16-value gathered rows.

## refutation condition

If Metal's actual packed-byte cross-entropy graph differs from both the CPU
quantized evaluator and independent scalar reference in logits, per-token
losses, compact gradients, or the host sparse update, the pilot graph is not
executing as the CPU pilot does on Metal.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Differentiate the exact tiny-pilot graph from `low_precision_moe_training_pilot.rs`: two expert matrices `[2,4,4]`, fixed top-1 parity router, one-hot four-class inputs/targets, stable softmax cross-entropy, and `GatheredContribution` values. Lift batch-one tensors to `expert_ids [8]`, `inputs [8,4]`, `targets [8,4]`, logits `[8,4]`, token losses `[8]`, and gathered rows `[8,4,4]`; row `i` retains source token `i`, with route `i % 2`. | yes |
| R2 | Execute the eight-token batch on Metal from the same direct packed BF8 E5M2 or BF4 E2M1 bytes used by the packed CPU pilot; preserve routes `[0,1,0,1,0,1,0,1]`, token order, and FP32 accumulation. | yes |
| R3 | For each codec, compare every token's four logits, token loss, 16 compact gathered-gradient elements, batch mean loss, coalesced expert gradients, and one host FP32 Adam update against the packed CPU evaluator and independent scalar reference, absolute tolerance `1e-6`. | yes |
| R4 | Retain exactly two records, one per codec, using the schema below with seed-17 input masters, packed bytes, routes, token IDs/targets, Metal/CPU/scalar outputs, full gradient rows, coalesced rows, and updated masters/moments. | yes |
| R5 | Keep this as a Metal backend parity fixture; it does not establish GPU 64-step throughput, learned-router behavior, corpus quality, convergence, or near-1B feasibility. | yes |

## architecture

Reuse the admitted graph and numerical definitions in
`proxima-tensor/specs/low_precision_moe_training/SPEC.md` and its packed pilot
continuation in `proxima-tensor/specs/packed_pilot_moe_training/SPEC.md`. Use
seed `17` and the exact initial FP32 masters and packed bytes in
`proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json`
step 1. Batch-lift the graph by concatenating the eight original batch-one
invocations in original token order: `expert_ids [8]`, `inputs [8,4]`,
`targets [8,4]`, logits `[8,4]`, token losses `[8]`, and gathered contribution
`[8,4,4]`; route index `i` is `i % 2`. These rows form one differentiated
program invocation, not eight sequential optimizer steps. Bind the packed
expert stack as the Metal computed-gather source, with route, activation, and
target buffers as separate inputs; do not expand BF8 or BF4 expert bytes into
a Metal FP32 expert table. Request logits, per-token cross-entropy, mean batch
loss, and the complete `[token,input,output]` gathered contribution from the
same differentiated program. The graph uses mean cross-entropy, so the host
Adam update consumes mean-scaled coalesced gradients.

The CPU path evaluates the same differentiated graph with its admitted
`ExpertSource` binding. The scalar path independently decodes the packed bytes,
computes each selected matrix row, stable softmax loss, token gradients, sparse
route coalescing, and the FP32 Adam update. Use the exact optimizer settings
from the CPU pilot: learning rate `0.001`, beta1 `0.9`, beta2 `0.999`, epsilon
`1e-8`, global step one. The sparse update remains host-side. Compare all
per-token arrays before aggregation and preserve both winning and losing
values in the result record when a mismatch occurs.

## acceptance criteria

### retained JSON schema

The root is an array with exactly two records. Each record has `codec`
(`Bf8E5M2` or `Bf4E2M1`), `seed` (`17`), `step` (`1`), `routes` (8 integers),
`token_ids` (`[0,1,2,3,0,1,2,3]`), `target_ids` (`[1,2,3,0,1,2,3,0]`), `targets` (8 rows × 4 floats),
`initial_masters` (32 floats), `packed_bytes` (the exact codec bytes consumed
by Metal), and `packed_byte_len` (32 for `Bf8E5M2`; 16 for `Bf4E2M1`, equal to the packed byte-array length). It contains
`backend: "metal"`, `logits` (8×4), `token_losses` (8), `mean_loss` (scalar),
`compact_gradients` (8×16 token-major values), `coalesced_gradients` (2×16),
`updated_masters` (32), `first_moment` (32), and `second_moment` (32).
`cpu_reference` and `scalar_reference` contain the same output and update
fields with backend values `cpu` and `scalar`. The Metal test asserts values;
the AC2 validator checks field presence, codec set, all shapes and lengths,
routes, token IDs, and backend labels for all three backends.

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1-R3 | `test "$(uname -s)" = Darwin && cargo test -p omega --test packed_brain_float_moe_training_metal packed_pilot_cross_entropy_matches_cpu_and_scalar -- --exact` | One Metal device test passes, exercising both codecs; eight token logits and losses, all 128 compact gradient values, batch loss, coalesced rows, updated masters, and Adam moments match CPU and scalar values within `1e-6` |
| AC2 | R4 | `test "$(uname -s)" = Darwin && RUSTC_WRAPPER= cargo test -p omega --test packed_brain_float_moe_training_metal packed_pilot_cross_entropy_records_payload -- --exact --nocapture > /tmp/metal-packed-pilot-training.log && python3 -c 'from pathlib import Path; import json; log=Path("/tmp/metal-packed-pilot-training.log").read_text(); marker="METAL_PACKED_PILOT_JSON="; start=log.index(marker)+len(marker); records,_=json.JSONDecoder().raw_decode(log[start:].lstrip()); assert len(records)==2; out=Path("proxima-tensor/specs/metal_packed_pilot_moe_training/results/metal-packed-pilot-training.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(records,indent=2)+"\n")' && python3 proxima-tensor/specs/metal_packed_pilot_moe_training/validate_report.py proxima-tensor/specs/metal_packed_pilot_moe_training/results/metal-packed-pilot-training.json` | One Metal device test passes; exactly two codec records satisfy the full schema and nested payload dimensions |

## out of scope

- changing the fixed router, tokens, targets, corpus, seed, or optimizer
- Metal execution of the full 64-step pilot or performance claims
- GPU sparse coalescing or GPU Adam
- training a one-billion-parameter model, checkpoint publishing, or release

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| four-class stable softmax lowering differs from the squared-loss fixture | medium | a generic Metal training result could be mistaken for the pilot graph | compare the complete logits and token-loss payloads before optimizer update |
| batch gather gradient order differs from `[token,input,output]` | medium | the host update writes the wrong expert elements | assert all 128 token-major compact values, then compare coalesced expert rows |
| the direct Metal input binding silently stages expanded weights | medium | memory behavior would not match packed inference | inspect the emitted Metal operand contract and retain packed byte input lengths in each record |

## context

- `proxima-tensor/specs/packed_pilot_moe_training/SPEC.md`
- `proxima-tensor/specs/packed_pilot_moe_training/results/packed-tiny-pilot.json`
- `proxima-tensor/specs/metal_packed_brain_float_moe_training/SPEC.md`
- `proxima-tensor/specs/metal_packed_brain_float_moe_training_loop/SPEC.md`
- `omega/tests/packed_brain_float_moe_training_metal.rs`
- `omega/tests/training_step_parity.rs`
- `omega/src/msl/signature_tokens_prelude.rs`
- `omega/src/msl/packed_row_blocked_ggml.rs`
- `proxima-autograd/examples/low_precision_moe_training_pilot.rs`
