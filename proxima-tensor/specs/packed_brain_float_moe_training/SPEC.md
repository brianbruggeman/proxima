# Packed BF4/BF8 MoE CPU training step

status: admitted
owner: brian
created: 2026-10-09

## problem

The retained BF4/BF8 pilot optimizes FP32 master weights in a synthetic
training loop, while the packed-byte MoE path now exists only as a forward
evaluation. No admitted evidence joins quantized packed forward execution,
autograd's gathered expert gradients, sparse duplicate-route accumulation, and
an optimizer update into one training step.

## refutation condition

If one CPU step cannot use the packed BF4/BF8 selected expert bytes for its
forward pass and produce the same routed FP32 master-weight update as an
independent straight-through reference, then this composition is not yet a
training path and must not be used to plan scale-up.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | A CPU training step reads BF8 E5M2 or BF4 E2M1 packed expert bytes through the computed gather and preserves FP32 activations, gradients, master weights, and optimizer state. | yes |
| R2 | Autograd's compact gathered contribution is coalesced by routed expert ID; repeated routes add row gradients and untouched experts receive no update. | yes |
| R3 | The update matches an independently computed FP32 straight-through reference for BF8 and BF4 on fixed payloads, including a repeated route and an untaken expert. | yes |
| R4 | The result retains per-step input bytes, routes, gathered gradients, updated master rows, and losses as JSON evidence with codec, seed, step, and backend labels. | yes |

## architecture

Keep trainable parameters and Adam state in FP32. At the start of each step,
encode the current expert master rows into the selected scalar format; feed
those packed bytes into the admitted `ExpertSource`/computed-gather CPU path.
Differentiate the same forward graph, evaluate each `GatheredContribution`,
then coalesce contributions by the route IDs using the existing sparse row
operation. The gathered contribution layout is `[token,input,output]`, so each
coalesced row is transposed into canonical `[output,input]` master storage
before the optimizer sees it. Apply the existing Adam implementation only to expert rows present
in the coalesced set, leaving all other rows bit-identical. The STE contract is
identity from decoded forward values to the corresponding FP32 master row;
the quantizer's rounding derivative is not included.

Use three experts with row-major `[out=2, in=2]` FP32 masters:
`e0=[1.1,0,0,1.1]`, `e1=[2.2,0,-1.1,0]`, `e2=[3.3,4,5,6]`. BF8 and BF4
round the selected e0/e1 values to `[1,0,0,1]` and `[2,0,-1,0]`, so bypassing
the packed forward would produce different predictions. For BF8 the full
step payload is `[3c,00,00,3c,40,00,bc,00,42,44,45,46]`; for BF4 it is
`[02,20,04,0a,65,76]` (hex). Routes are `[1,1,0]`, activations are
`[[1,0],[1,0],[0,1]]`, and targets are `[[0,0],[1,0],[0,0]]`. Define the
batch loss as the sum of `0.5 * (prediction - target)^2` over all six output
elements. The expected predictions are `[[2,-1],[2,-1],[0,1]]` and loss is
`4.0` for either encoding. The canonical row-major coalesced gradients are
`e0=[0,0,0,1]`, `e1=[3,0,-2,0]`, `e2=[0,0,0,0]`. Starting Adam moments at
zero, using default Adam settings and step 1, the expected masters are
`e0=[1.1,0,0,1.099]`, `e1=[2.199,0,-1.099,0]`, and
`e2=[3.3,4,5,6]` (f32 representations). With initial moments zero, the
updated first moments are `e0=[0,0,0,0.1]`, `e1=[0.3,0,-0.2,0]`, and
`e2=[0,0,0,0]`; updated second moments are `e0=[0,0,0,0.001]`,
`e1=[0.009,0,0.004,0]`, and `e2=[0,0,0,0]`. The exact Adam settings are
learning rate `0.001`, beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`, step `1`.
The reference is a separate scalar FP32 forward and
straight-through gradient calculation over those arrays, followed by the
Adam equations; it must not call the packed kernel, autograd transform, sparse
coalescer, or production optimizer. Use fixture seed `17` and backend label
`cpu` in each record.

Persist exactly two JSON records, one per codec, with non-null codec, backend,
seed (`17`), backend (`cpu`), step, master input, routes, activations, targets,
packed bytes, predictions, loss, three coalesced gradient rows, three final
master rows, updated first and second moments, and reference arrays. The first implementation is CPU-only; GPU execution,
optimizer sharding, model-scale throughput, and near-1B training remain
outside this spec.

## acceptance criteria

| id | discharges | command | expected |
|---|---|---|---|
| AC1 | R1-R2 | `cargo test -p proxima-autograd --features instrument,proxima-tensor/instrument --test packed_brain_float_moe_training packed_gathered_step_updates_only_selected_master_rows -- --exact` | 1 test passes; outputs are `[[2,-1],[2,-1],[0,1]]`, loss `4.0`, routes `[1,1,0]`, and six captured dispatch records prove the selected byte spans in route order for BF8 (`[40,00,bc,00]` twice, then `[3c,00,00,3c]`) and BF4 (`[04,0a]` twice, then `[02,20]`); coalesced gradients are `[0,0,0,1]`, `[3,0,-2,0]`, `[0,0,0,0]`, and expert 2's `[3.3,4,5,6]` master and zero moments are unchanged |
| AC2 | R3 | `cargo test -p proxima-autograd --features instrument,proxima-tensor/instrument --test packed_brain_float_moe_training packed_bf4_bf8_updates_match_fp32_ste_reference -- --exact` | 1 test passes; both codecs produce masters `[1.1,0,0,1.099]`, `[2.199,0,-1.099,0]`, `[3.3,4,5,6]`, first moments `[0,0,0,0.1]`, `[0.3,0,-0.2,0]`, `[0,0,0,0]`, second moments `[0,0,0,0.001]`, `[0.009,0,0.004,0]`, `[0,0,0,0]`, each matching the independent scalar FP32 STE plus pinned Adam reference |
| AC3 | R4 | `cargo test -p proxima-autograd --features instrument,proxima-tensor/instrument --test packed_brain_float_moe_training packed_training_payload_record_is_complete -- --exact` | 1 test passes; `results/packed-training.json` contains exactly 2 records with seed `17`, backend `cpu`, inputs, routes, full packed bytes, selected dispatch spans in route order, predictions, loss, full gradients, final masters, moments, and independent references specified above |

## out of scope

- GPU execution of gathered gradients or BF4/BF8 GPU kernels
- multi-GPU training, distributed expert ownership, optimizer sharding, or checkpoint publishing
- tokenization, corpus curation, full language-model architecture, or quality claims
- any statement about training a model near 1B parameters

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| autograd's gathered-contribution shape differs from the expert matrix layout | medium | sparse update can target wrong elements | assert complete per-row gradient arrays and compare to the independent reference |
| duplicate routes are overwritten instead of summed | medium | batch contributions are lost | routes deliberately repeat expert 1 and assert its coalesced payload |
| BF4 row geometry needs even input dimensions | high | odd-width matrices have padded or ambiguous trailing nibbles | fixture and API must reject non-even packed rows with a typed error |
| optimizer update uses packed values as parameters | medium | rounding becomes accumulated into the master copy | bind only FP32 master rows to Adam and assert untouched rows are bit-identical |

## context

- `proxima-tensor/specs/low_precision_moe_training/SPEC.md`
- `proxima-tensor/specs/packed_brain_float_moe/SPEC.md`
- `proxima-autograd/src/adjoint.rs`
- `proxima-autograd/src/sparse.rs`
- `proxima-autograd/src/optimizer.rs`
- `proxima-autograd/src/low_precision.rs`
- `proxima-tensor/src/cpu/epilogue.rs`
- `proxima-tensor/src/cpu/run_reduce_scan.rs`
