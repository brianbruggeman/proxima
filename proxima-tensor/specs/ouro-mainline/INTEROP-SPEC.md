# recurrent model loading and serving

status: draft for independent admission
owner: Luna implementation; root caller integration

## problem

Load Ouro through the existing architecture registry and serving lifecycle: the
one-physical-layer/two-iteration fixture must bind one physical weight set and
two separate logical KV owners, and produce `[3,4]*sqrt(2/52)` for token zero
within absolute-plus-relative 0.000001. GGUF and safetensors use the same shared
descriptor. No alternative transformer evaluator is introduced.

## refutation condition

Wrong fixture values, recurrent histories sharing buffers, repeated physical
weight ownership, rebuilding a Mistral-only graph, or changing model-bound
recurrence identity after prefill refutes the corresponding requirement. A
deliberately wrong depth/readout identity must be rejected before plan or KV work.

## requirements

| id | requirement | isolated |
|---|---|---|
| I1 | Ouro resolves through ArchitectureRegistry and builds the admitted shared descriptor graph. | yes |
| I2 | Bind physical checkpoint tensors once, including sandwich norms, then derive logical KV sizing with checked arithmetic. | yes |
| I3 | GGUF and HF descriptors express equivalent normalization/recurrence/RoPE with their actual stored layouts: pinned GGUF interleaved, native HF split-half; absence of QK norms does not choose a permutation. | yes |
| I4 | Typed recurrent configuration has matching fluent/env/file routes; unsupported settings fail before execution. | yes |
| I5 | Serving validates model-bound settings before planning/cache mutation; prefix identity includes every value-affecting setting. | yes |
| I6 | Existing production prefill/decode/reset and diagnostic paths consume the shared graph on CPU and Metal; no lab graph is required. | yes |

## architecture and ownership

WORK-PACKET.md sections 4-7 bind signatures, fixtures and source ownership.
Luna owns interop sources and tests; Sol owns tensor graph and tensor initializer
updates. Coordinate shared types before editing the same initializer. Root owns
lab caller and oracle. No commits, scripts or Cargo.toml edits.

1. Add stateless `OuroArch` in `src/ouro.rs`, register its static value in the
   existing registry. Translate metadata into `ModelDescriptor`, call
   `build_forward_with_readouts`, and return `BoundProgram`. Keep loader IO and
   tensor execution in existing owners.
2. Add `RecurrentConfig` Copy data plus std `RecurrentSettings` using the existing
   Settings/Validate/serde/bon pattern. Fields: optional iterations, LoopCdMode,
   strength, one-based reference iteration. Resolve None from model metadata.
   Reject zero/overflow, invalid reference, nonfinite/negative strength, nonzero
   disabled strength and nonneutral settings on unsupported architectures.
3. Architecture trait options-aware bind and KV hooks default to existing hooks
   for neutral settings and reject nonneutral settings. Load constructors all
   delegate to the existing load_inner body. BoundProgram/LoadedModel carry
   optional RecurrentReadoutRoots and immutable resolved recurrence identity.
4. Keep physical metadata for binding; effective logical layer count and KV head
   schedule are expanded only afterward. Memory-fit counts physical weights once
   and logical KV owners separately. `KvCacheShape::Custom` excludes the dense
   Mistral single-range reconstruction. Ordinary host cache ownership remains
   explicit in evidence; do not describe it as device-resident KV.
5. HF companion takes a descriptor, sharing the old loader body. Existing caller
   derives its old descriptor. Ouro typed config retains total_ut_steps and norm
   choices. Explicit descriptor RopePairing controls HF row layout. Bind required
   branch norms through existing tensor binding helpers. Preserve old non-Ouro
   graph/layout through a tiny ordinary positive-control fixture.
   The pinned GGUF converter permutes Q/K rows (main.cpp:202-203,213-230),
   and pinned llama selects ROPE_TYPE_NORM (llama-model.cpp:3105-3107).
   GGUF therefore uses Interleaved, HF uses SplitHalf with no permutation.
6. Validate ServingConfig's resolved identity before any plan lookup/prefill/cache
   mutation, including zero-token calls. Equivalent None/checkpoint-depth and
   explicit checkpoint-depth resolve identically. Prefix-cache key carries depth,
   norm policy, mode, strength bits and reference. Disallow mismatched live state.
7. Expose frozen-token diagnostic evaluation as a small wrapper around existing
   `evaluate_forward_tokens`, accepting token IDs, requested roots and serving
   config; retain old text wrappers. Expose recurrent readout roots through a
   borrowed getter. Timing uses existing `generate_from_ids`, not this readback
   helper. Do not create another BackendRuntime owner.

### constraint-driven decisions

- Rejected one mapped tensor per logical visit: callers share checkpoint bytes
  and graph input identity while cache owners stay distinct.
- Rejected model-name branching in decode: architecture capabilities and resolved
  model data select admissible shared paths.
- Rejected live depth overrides: graph and prefix state bind the same identity.
- New generic config remains no_std data; std loaders/builders stay gated.

## fixtures and controls

The literal full GGUF/HF fixture is WORK-PACKET section 5, epsilon one, two loops.
Preserve actual logits, cache rows, buffer identities, shapes and leaf names in
test output. Compare full vectors, not argmax alone. The wrong-depth control,
wrong HF permutation control and deliberately aliased logical-cache control must
fail the same assertions used for the positive fixture. Shared tensor tests
independently cover branch normalization and nonzero attention/FFN arithmetic;
zero O/down in this fixture intentionally isolates loading and iteration norms.

## acceptance criteria

All runs delegated, no scripts. Preserve logs under results/validation. Exactly
eight CPU cases with `recurrent_mainline_` prefix, as WORK-PACKET section 6:
registry_ouro_exact, tiny_gguf_literal, physical_weights_once,
hf_shared_descriptor_literal, settings_three_routes, loader_fastpath_gate,
serving_identity_before_work, production_decode_literal. Each has the concrete
assertions in that packet; additionally hf_shared_descriptor_literal includes
the ordinary-loader positive control, and production_decode_literal includes
reset and full-prefill versus chunked continuation cache rows.
The HF case also requires a head-width-four asymmetric projection at position
one: split-half coordinates `[1,2,3,4]` correspond to interleaved `[1,3,2,4]`;
use cosines `[0,1]`, sines `[1,0]`. Rotated split-half is `[-3,2,1,4]`, rotated
interleaved is `[-3,1,2,4]`. After the inverse layout permutation they must
match exactly. Applying split-half pairing to the stored interleaved vector
must fail. Head width two cannot distinguish these layouts.

| id | discharges | native command | expected |
|---|---|---|---|
| IAC1 | I1-I6 | `cargo nextest run -p proxima-model-interop --features std --lib -E 'test(recurrent_mainline_)' --no-capture` | Exactly 8 named cases run, 0 skipped, full payloads and negative controls retained. |
| IAC2 | I2,I3,I6 | `cargo nextest run -p proxima-model-interop --features 'std metal metal-output-placement instrument' --lib -E 'test(recurrent_metal_)' --no-capture` | Exactly 3 named real-device cases: cached_literal, repeat_identity, loopcd_readouts; 0 skipped; same tiny CPU literals and full KV payloads. |
| IAC3 | I1-I5 | `cargo check -p proxima-model-interop --no-default-features --message-format=json` | 1 proxima_model_interop library artifact, 0 compiler errors, build-finished success true. |
| IAC4 | I1-I6 | `cargo clippy -p proxima-model-interop --features std --lib` | 0 compiler/clippy errors; preserve entire log. |

## slices

| slice | requirement | validation | expected |
|---|---|---|---|
| existing registry/load/config adaptation | I1-I6 | IAC1 exact command above, then IAC2-IAC4 | 8 CPU cases, 3 Metal cases, 1 no-default library artifact, 0 compiler/clippy errors |

## risks

| risk | response |
|---|---|
| Tensor signatures compile while interop is changing | Coordinate declared ownership; capture each actual command and retry only after source fix. |
| Model-bound identity differs through alternate entry | Validate in common decode admission, plus diagnostics and constructor validation. |
| Cache length/byte budget counts physical instead of logical | Assert two logical owners and capacity accounting from literal payload. |
| Positive tests accidentally use the lab implementation | Tests must construct LoadedModel and call its production methods; imports from lab are forbidden. |

## out of scope

- This adapter does not add checkpoint IO, tensor execution, backend kernels,
  an alternative sampler, or another KV/plan-cache owner.
- Publishing and committing are not authorized by this packet.

## context

`proxima-model-interop/src/{architecture,dense,bind,hf_bind,hf_config,serving,prompt_cache_settings}.rs`;
`proxima-model-interop/src/generate/{pregather,load_model,decode,residency_caches,prompt_cache_key}.rs`;
WORK-PACKET.md sections 4-7; SHARED-GRAPH-SPEC.md; loopcd-panel/ORACLE-SPEC.md.
