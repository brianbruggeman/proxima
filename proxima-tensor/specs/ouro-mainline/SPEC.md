# ouro-mainline

status: implementation under independently admitted component contracts
owner: root with delegated implementation and validation
created: 2026-10-03

## problem

Ouro currently exercises a lab-owned transformer and recurrence loop; migrate its
execution to mainline's descriptor, architecture binding and cached serving loop,
then measure the shared pipeline against the same checkpoint's llama runner.

## refutation condition

The adaptation fails its purpose if Ouro needs a second attention implementation,
executor, sampler, plan cache or KV owner; if recurrent visits alias each other's
KV state; if existing model descriptors change their graph semantics; or if the
reported wall-clock measurements come from the lab executor instead of mainline.

## requirements

| id | requirement | testable in isolation |
|---|---|---|
| R1 | Express physical weight reuse, logical layer visits, inter-pass normalization and completed-pass readouts as data consumed by the existing forward builder. | yes |
| R2 | Make Q/K normalization an explicit choice in the existing attention configuration. Existing callers retain their graph; Ouro uses raw projected Q/K before RoPE. | yes |
| R3 | Register Ouro through the existing architecture registry and bind the pinned GGUF through existing weight-binding primitives. Generation uses LoadedModel and its ordinary serving loop. | yes |
| R4 | Preserve separate KV histories per logical visit while sharing physical weights, including cached continuation, chunked prefill and reset. | yes |
| R5 | Preserve baseline, fixed-logit, adaptive-logit and fixed-hidden paper semantics, including first/terminal representation boundaries and deliberate wrong-sign/wrong-boundary controls. | yes |
| R6 | Use the common numeric policy, fusion and kernel paths; do not create Ouro-specific lowering. Keep unsupported resource/driver cases explicit. | yes |
| R7 | Replace the experiment's production execution owner with mainline. Retain independent references and historical payloads, not a competing production pipeline. | yes |
| R8 | Measure mainline prefill, cached decode and complete response wall time against pinned llama with frozen inputs, output payloads, warmups, repeated samples and CoV. Retain losses and errors. | yes |

## architecture

```text
checkpoint metadata + model configuration
                |
       ArchitectureRegistry / bind
                |
       existing ModelDescriptor
       physical weights + logical schedule
                |
       existing build_forward
       existing attention / FFN / readout primitives
                |
       LoadedModel serving lifecycle
       KV ownership / prefill / decode / sampling
                |
       existing Omega binding and lowering
                |
       backend Plan / buffers / kernels
```

The implementation contracts are SHARED-GRAPH-SPEC.md, INTEROP-SPEC.md and
CALLER-SPEC.md, each with its own independent audit artifact. WORK-PACKET.md
holds shared signatures and literal fixtures; the component specs bind runnable
acceptance counts and ownership.
The existing TwoRange builder is the reuse candidate because it already consumes
LayerAttentionConfig and LayerFfnConfig, including sandwich normalization.
Its forced Q/K normalization is a concrete semantic gap, not a missing Ouro
kernel. Existing KeySource::SharedFromLayer means shared history and must not be
used to express shared recurrent weights.

### decisions

| decision | chosen | why not the alternative |
|---|---|---|
| Development direction | Adapt the current shared graph and serving path; return generic gaps to those owners. | Extending the lab runtime duplicates lifecycle and hides production gaps. |
| Cache identity | Logical visit owns KV; physical layer owns weights. | Weight sharing does not make activations or attention history identical. |
| Q/K behavior | Optional normalization in the current attention helper. | Unit scale still normalizes the vectors and cannot represent absence. |
| Reference role | Existing literal controls, retained vectors and external llama remain references. | A second Proxima executor is not the deployment path under measurement. |
| Routing | Luna for bounded adapters/fixtures; Sol for recurrent semantics and cross-crate implementation; root integrates and reads payloads. | Cache/state and graph-boundary changes need the documented higher-capability envelope before execution. |

## acceptance criteria

| requirements | admitted contract | expected evidence |
|---|---|---|
| R1,R2,R4,R5 | SHARED-GRAPH-SPEC GAC1-GAC4 | 10 new literal cases; 3 frozen ordinary graph payloads inside them; 2 existing graph comparisons; 1 alloc library artifact; 0 compiler/clippy errors |
| R3,R4,R6 | INTEROP-SPEC IAC1-IAC4 | 8 CPU and 3 Metal cases; 1 no-default library artifact; 0 compiler/clippy errors |
| R5,R6,R7,R8 | CALLER-SPEC CAC1-CAC5 | 4 caller cases; 3 Omega forwarding cases; 1 example artifact; 12 cached requests per runtime across 2 frozen prefixes, including 10 measured samples and 2 warmups |

Full native commands, per-case assertions, negative controls and output payload
contracts live in those admitted documents. No graph fixture, test count or
measurement row substitutes for opening its actual payload and assertion.

## out of scope

- Publishing, committing, pushing, retraining, or adding an unavailable NPU driver.

## risks

| risk | likelihood | what it costs | what we do about it |
|---|---|---|---|
| Recurrent KV alias | concrete design hazard | Different model semantics | Trace each logical cache leaf and retained row in full-prefix versus cached fixtures. |
| Extra Q/K norm | observed in reuse candidate | Different logits | Assert absence/presence from literal non-unit projected vectors. |
| Ordinary dense fast path bypasses recurrence | concrete integration hazard | Wrong model | Give architecture capability data to load/prewarm/speculation paths; do not branch on the model name in serving. |
| Duplicate mapped weight uploads | possible | Memory and wall-clock regression | Inspect actual names, input node sharing and physical buffer ownership. |
| Benchmark lifecycle mismatch | observed in current oracle | Invalid decode comparison | Add an explicit cached oracle mode; retain the existing fresh-prefix mode separately. |
| Starting dirty work is overwritten | workspace condition | Lost changes | Preserve the starting patch and integrate only owned diffs; no automatic commits. |

## context

- `proxima-tensor/src/spec/descriptor.rs`: existing ModelDescriptor and build_forward.
- `proxima-tensor/src/spec/attention_forward.rs`: attention and FFN configuration/primitives.
- `proxima-tensor/src/spec/lfm2_single_range_cached.rs`: existing TwoRange graph.
- `proxima-model-interop/src/architecture.rs`: Architecture, BoundProgram and registry.
- `proxima-model-interop/src/generate/residency_caches.rs`: production plan/cache owner.
- `proxima-model-interop/src/generate/decode.rs`: production cached generation lifecycle.
- `examples/ouro_loopcd_lab/{graph,engine,model,performance}.rs`: execution to migrate.
- `examples/ouro_llama_oracle/main.cpp`: pinned independent oracle.
- `/private/tmp/omega-decode-inventory.md` and `/private/tmp/ouro-mainline-harness-inventory.md`: opened inventory records.
- `/private/tmp/ouro-mainline-start.patch`: captured tracked starting changes.
