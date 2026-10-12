# Ouro through the production model pipeline

Status: original design packet, retained with its initial source inventory. The independently admitted contracts and executed artifacts are indexed in TASKS.md and EVIDENCE.md; statements about the original lab route below describe the starting point. Source tree: `/Users/brianbruggeman/repos/slot-0/proxima`. This packet supplements `/private/tmp/omega-decode-inventory.md` with the source boundaries opened below.

## 1. Actual reuse boundary

The production entry is `ArchitectureRegistry::with_builtin` / `Architecture::bind_with_kv_layout` -> `BoundProgram` -> `LoadedModel::load_inner` -> existing `run_decode_loop_observed_seeded` -> `BackendRuntime` cached Omega plans. Registry code is in `proxima-model-interop/src/architecture.rs`, not a registry.rs file. Ouro must register here and emit the existing `Op` graph through `ModelDescriptor` / `build_forward`.

The lab currently executes an uncached sequence graph through `examples/ouro_loopcd_lab/model.rs:402` and `graph.rs:150`; it repeatedly calls the layer executor, then normalization, then optional heads. It does not implement the production cache contract. The lab remains an evidence caller, adapted to the shared model load/forward interfaces; its checkpoint reader and private graph do not become a second production loader or engine.

The shared graph substrate already contains:

- `descriptor.rs:541` `build_forward`, whose TwoRange arm dispatches into `lfm2_single_range_cached.rs:1260` `lfm2_two_range_cached_forward_program_with_experts`.
- `lfm2_single_range_cached.rs:1434` per-layer attention/FFN construction and `:1697` final RMSNorm/head.
- `LayerFfnConfig.post_attention_norm` and `exclusive_dense_post_norm`, implemented before residual additions (`attention_forward.rs:1150–1205`); these express all four Ouro branch norms.
- Cached two-range attention, current-range/history masks, split-half RoPE, and `CachedLayerRoots`; retain these helpers and their existing fusion/numeric options.

Abandoned approaches: adding an Ouro-only attention builder; pretending unit gamma disables Q/K RMSNorm; using `SharedFromLayer` to share recurrent history; duplicating weights under every logical-layer name; copying the lab's host loop into production serving. Each would bypass an existing semantic boundary or introduce a second implementation.

## 2. Minimal shared data extensions

Add these plain build-time data fields to the existing tensor descriptor types (no runtime, cache, scheduler, allocator or new Op variant):

```rust
pub struct RecurrenceConfig {
    pub iterations: u32,
    pub normalize_each_iteration: bool,
}

pub enum LoopCdMode { Disabled, FixedLogits, AdaptiveLogits, FixedHidden }

pub struct LoopCdConfig {
    pub mode: LoopCdMode,
    pub strength: f32,
    pub reference_iteration: u32, // completed iteration, one based
}

// additions to existing types
ModelDescriptor::recurrence: Option<RecurrenceConfig>
ModelDescriptor::loopcd: LoopCdConfig
LayerAttentionConfig::qk_norm: bool
```

All current descriptor constructors set recurrence None and LoopCD Disabled; current TwoRange/Cacheless schedules set qk_norm true, preserving their existing per-head normalization. Mistral SingleRange retains its existing qk_norm choice. Ouro sets false: `references/ouro-modeling_ouro.py:304–312` projects Q/K and directly applies RoPE. Current TwoRange applies Q/K normalization unconditionally at `lfm2_single_range_cached.rs:477,506`; gamma=1 is still normalization, hence cannot represent Ouro.

`RecurrenceConfig` applies to the existing complete physical schedule. `ModelDescriptor.block_count` and `layers.len()` remain physical checkpoint counts. Effective logical count is checked `physical_count * iterations`; both counts must be positive. For logical index `logical`, physical weight index is `logical % physical_count`, iteration is `logical / physical_count`. No recurrence means exactly the original schedule and graph. Initially admit Attention schedules, including dense/routed FFNs; reject recurrent ShortConv before graph construction until its state index expression has a documented contract. Repeat schedule-local SharedFromLayer donors within each iteration: logical donor = iteration_base + physical donor, preserving the existing requirement donor precedes consumer. This sharing is an explicit model configuration and never an automatic consequence of recurrence.

The existing FFN helper currently uses `layer` for both `blk.{layer}` leaves and `MoeSite.layer`. Its recurrent adaptation must carry two indices: physical for every weight leaf, logical for sites/diagnostics/routing identity. Carry explicit physical/logical index arguments through the existing helper seam; do not add a record solely to pack parameters. Keep nonrecurrent callers at `logical == weight`; update all such callers explicitly. Apply the same distinction to per-layer embedding slices and shared-KV donors or reject the unsupported combination at descriptor validation before emitting any nodes. Do not silently reuse physical state for a logical recurrence.

## 3. Exact graph seams

`build_forward(&ModelDescriptor, last_row_only)` remains the public route. Refactor the current TwoRange body behind `build_two_range_forward(&ModelDescriptor, last_row_only)`, a private/shared implementation. The existing positional `lfm2_two_range_cached_forward_program_with_experts` becomes a compatibility wrapper constructing the existing descriptor with recurrence None/LoopCD Disabled and calling that same body. Do not maintain two bodies. The descriptor arm calls the body directly.

In that body:

1. Construct token embedding/masks/RoPE/resources using the existing helpers. Resource dedup remains by attention configuration, not logical weight identity.
2. For each logical layer, select its physical schedule entry. Use `blk.{physical}.*` for norm/QKV/output/FFN leaves; use `kv_cache.{logical}.{k_even,k_odd,v}` for history and return one cache-root triple for each actual logical owner. Repeated weight names must resolve to the same bound bytes; preferably reuse identical Input NodeIds through a cold weight-leaf prepass, avoiding duplicate device-weight objects in a plan. Shapes and codecs are checked at the existing binder boundary.
3. Change Q/K norm-weight arguments to `Option<NodeId>` in the existing attention helper and `KeySource::Projected`; Some retains the exact current path, None passes raw projection directly to existing `fused_rope_pair`. Apply the same representation to the cacheless/single-range siblings so their shared KeySource match remains exhaustive; preserve current behavior at their callers.
4. Keep both sandwich norms in the existing LayerFfnConfig helpers: input RMSNorm, post-attention RMSNorm, residual add, FFN-input RMSNorm, post-FFN RMSNorm, residual add.
5. At each completed iteration retain `raw_hidden`, apply the shared `output_norm.weight` RMSNorm when configured, retain `normalized_hidden`, and feed that value into the next iteration. Final iteration has one norm, not an inter-loop norm followed by a second final norm. Disabled recurrence retains the original single final norm/head nodes and existing output roots.
6. LoopCD Disabled computes only terminal head. FixedLogits computes reference and terminal heads after normalization, then `terminal + strength*(terminal-reference)` with existing elementwise/subtract/multiply/add. AdaptiveLogits preserves the lab rule `strength * (1 - (p_top1-p_top2))` from a stabilized terminal softmax, then the same contrast; top1/top2 are values, not token-index tie breaking. FixedHidden contrasts raw completed states BEFORE the final output RMSNorm, then computes one terminal head. Gather the last new row before reference/terminal vocabulary projections where last_row_only=true; do not remove whole-row loop norm needed by the next iteration.
7. Preserve cache roots from each layer before final hidden contrast. Hidden guidance changes readout only; it must not rewrite the already-computed KV rows or the next request's history.

Reuse the existing scalar/reduction builders; factor lab adaptive/contrast semantics into these shared spec helpers if absent. No new scalar opcode is required. Two top probabilities can be computed by max p then max over elements unequal to the maximum, except ties require preserving a second equal maximum; therefore implement via existing deterministic stable top-two machinery if present, or a documented max/count/reduction expression that explicitly handles equal maxima. A naive `p < max` filter fails on tied maxima and is inadmissible.

Expose boundaries through one small build result extension rather than overloading the existing `hidden_root` (which means normalized output activation): add `RecurrentReadoutRoots { reference_raw, terminal_raw, reference_normalized, terminal_normalized, reference_logits, terminal_logits }`, each optional NodeId, to a new named result wrapper around the existing `BuildForwardProgram` fields. Preserve the public tuple API via `build_forward`, and add `build_forward_with_readouts(&ModelDescriptor, bool) -> Result<ForwardBuild, TensorError>` as the single implementation called by both. `BoundProgram` and `LoadedModel` carry `Option<RecurrentReadoutRoots>`; callers request diagnostic roots explicitly. Ordinary serving requests only final logits plus existing cache roots. Diagnostic captures add measurable readback work and are absent from timing controls.

## 4. Load adaptation, weight sharing and settings

Owned files and entrypoints:

| File | Adaptation |
|---|---|
| tensor `src/spec/descriptor.rs` | recurrence/readout descriptor fields; validation; build_forward_with_readouts / compatibility projection |
| tensor `src/spec/attention_forward.rs` | qk_norm data; physical/logical FFN indices; shared readout algebra if placed here |
| tensor `src/spec/lfm2_single_range_cached.rs` | same TwoRange engine, physical leaves/logical caches, loop normalization/readouts, optional QK norms |
| tensor `src/spec/mod.rs`, `src/spec/tests.rs` | exports; native literal fixtures and graph controls |
| interop `src/ouro.rs` (new) | stateless Architecture adapter only; metadata -> shared descriptor; weight binding via existing helpers |
| interop `src/architecture.rs` | register static OURO; BoundProgram readout field; options hook described below |
| interop `src/bind.rs` | reuse bind_all_weights physical pass; add required post_attention_norm/post_ffw_norm binding via an explicit descriptor-aware norm seam, not name heuristics |
| interop `src/hf_config.rs`, `src/hf_bind.rs` | parse total_ut_steps and norm/RoPE config; physical tensor name mapping and no erroneous RoPE permutation |
| interop `src/generate/pregather.rs`, `load_model.rs` | same LoadedModel constructors; resolved recurrence identity, effective KV count; no Mistral reconstruction |
| interop `src/generate/decode.rs`, `prompt_cache_key.rs` | consume/check resolved identity; same loop/cache append; cache identity includes recurrence/LoopCD policy |
| interop `src/serving.rs`, `src/recurrent_settings.rs` (new), `lib.rs` | Copy config plus std Settings/bon mirror using existing SpeculativeSettings/PromptCacheSettings pattern |
| lab `examples/ouro_loopcd_lab/{engine,model,checkpoint,graph}.rs` | replace private computational route with shared LoadedModel route; retain provenance/raw-evidence serialization |

GGUF route already exists: `LoadedModel::load` at `pregather.rs:2413`, registry resolution at `:2638`, vocabulary from GGUF metadata. Register `OuroArch` exact architecture `ouro`; parse physical block count, dimensions, epsilon, theta and `ouro.num_loops` (pinned llama implementation reads LLM_KV_NUM_LOOPS at `src/models/ouro.cpp:27`). Load physical weights once, including the two post norms. `BoundProgram.architecture.block_count` used by serving must expose effective logical count only after physical binding; expand `kv_heads_by_layer` to the same logical length. Keep physical count in the recurrence descriptor/resolved model identity, never infer it from effective count later. `Architecture::kv_layers` must match logical cache-root owners; extend its options-aware hook alongside bind so a steps override does not size default-depth caches.

For options, add to the existing trait:

```rust
fn bind_with_recurrence<'file>(
    &self, parsed: &ParsedGguf, bytes: &'file [u8],
    layout: KvLayout, config: &RecurrentConfig,
) -> Result<BoundProgram<'file>, InteropError>;
fn kv_layers_with_recurrence(
    &self, parsed: &ParsedGguf, config: &RecurrentConfig,
) -> Result<Vec<(u32,u32,Option<u32>)>, InteropError>;
```

Default implementation accepts the neutral config and delegates to existing hooks; rejects nonneutral recurrence/LoopCD on architectures that do not declare recurrent support. `OuroArch` resolves model iterations from metadata unless explicitly overridden, builds the descriptor, and uses `build_forward_with_readouts`. Add `LoadedModel::load_with_recurrent_config(parsed, bytes, &RecurrentConfig)`; existing load calls it with neutral settings. Thread this through the existing load_inner, not a parallel constructor body.

`RecurrentConfig` is the Copy serving/load shape: `iterations: Option<u32>` (None = checkpoint), `mode: LoopCdMode`, `strength: f32`, `reference_iteration: u32` (default 1). `RecurrentSettings` is Settings+Validate+serde+bon, std-only with prefix `PROXIMA_RECURRENT`, lowers via `as_recurrent_config`. No separate Omega engine/backend knobs. Validate positive explicit iterations, finite nonnegative strength, reference in 1..=resolved iterations; Disabled requires strength zero to reject unused configuration. Typed model-bound config identity is fixed at load. ServingConfig carries the same config; decode admission rejects mismatch with the model's resolved graph BEFORE plan lookup/prefill/cache writes. A caller changes policy by loading/building a new graph sharing existing checkpoint bytes, rather than mutating a live prefix's recurrent depth. Settings are accepted only when consumed; unsupported options return typed InteropError.

HF route exists at `pregather.rs:2918`: `load_from_safetensors(manifest, bytes, data_start, ModelArchitecture, Vocab)` binds through `hf_bind.rs:547` then currently hardcodes the Mistral graph. `architecture_from_hf_config` discards model_type and does not preserve total_ut_steps. Add a descriptor-aware companion `load_from_safetensors_with_descriptor(manifest, bytes, data_start, architecture, vocab, &ModelDescriptor)` and make the old method derive the same current dense descriptor and call the shared implementation. `ouro_descriptor_from_hf_config` consumes the parsed typed Ouro fields, feeds that companion, and stores the same resolved recurrence identity. This preserves the existing caller-owned Vocab/single-shard route; no directory loader is needed for the pinned checkpoint.

HF mappings: input_layernorm -> attn_norm; input_layernorm_2 -> post_attention_norm; post_attention_layernorm -> ffn_norm; post_attention_layernorm_2 -> post_ffw_norm. Q/K weights preserve split-half native layout. Current HF binding `permute_rope_rows = !safetensors_has_qk_norm(manifest)` at `hf_bind.rs:573` would incorrectly permute Ouro because Ouro lacks Q/K norm tensors. Resolve row permutation from the shared descriptor's RopePairing, not QK-norm presence. GGUF mapping follows the existing pinned converter names. Its Q/K rows ARE permuted to interleaved (converter main.cpp:202-203,213-230; pinned llama-model.cpp:3105-3107 selects ROPE_TYPE_NORM), so the GGUF descriptor uses Interleaved while native HF uses SplitHalf. A width-four asymmetric fixture distinguishes them; width two cannot. BoundWeights stores seven physical matrix payloads per block, four norm vectors, shared embedding/final norm/head; recurrence increases logical KV/state, not weight payload ownership.

`OuroArch::kv_cache_shape()` returns Custom to prevent `pregather.rs:2680` from constructing the Mistral SingleRange graph. That graph would omit sandwich/inter-loop norms and request nonexistent logical weight names. Custom here is an exact fast-path capability gate, not an excuse to ignore cache shape: Ouro's two-range graph still uses ordinary cache-root triples and the shared host cache append route. Device placed-KV needs a subsequent shared builder capability adaptation in the same admitted task if required by wallclock acceptance; do not report ordinary Metal uploads/readbacks as device-resident KV.

## 5. Hand-worked tiny fixtures

### Separate history despite shared weights

Fixture uses one physical projection block, two completed iterations, head width 2, one head, identity K/V matrix and identity RoPE (cos=1, sin=0), zero Q. Inject each logical layer's input activation at the projection boundary; this fixture isolates KV indexing and does not claim to reproduce full Ouro normalization.

| logical / physical | token A input and stored K/V | token B input and new K/V | score vector for B | attention result B |
|---|---|---|---|---|
| 0 / 0 | [2,0] | [4,0] | [0,0] -> probabilities [1/2,1/2] | [3,0] |
| 1 / 0 | [6,0] | [8,0] | [0,0] -> probabilities [1/2,1/2] | [7,0] |

After both tokens: `kv_cache.0.v=[[2,0],[4,0]]`; `kv_cache.1.v=[[6,0],[8,0]]`; key halves have even rows [2,4] vs [6,8], odd rows [0,0] for both. Both projection nodes resolve the same physical identity-weight bytes, but every cache-owning logical root and buffer is distinct. Force logical1 to read logical0 history: result becomes [5,0], which must fail the literal [7,0] assertion. Include poisoned padded rows [1000,1000] outside cached_len to exercise the existing cache mask. Compare two-token prefill vs one-token prefill plus decode on the same generic attention program, preserving actual cache roots/values.

### LoopCD boundary arithmetic

Completed raw states reference=[3,4], terminal=[0,5], shared output norm gamma=[1,1], epsilon=0, identity head, strength=1/2:

- reference normalized/logits = [3*sqrt(2)/5, 4*sqrt(2)/5].
- terminal normalized/logits = [0,sqrt(2)].
- fixed-logits result = [-3*sqrt(2)/10, 11*sqrt(2)/10].
- fixed-hidden raw contrast = [-3/2,11/2]; mean square = 65/4; normalized/head result = [-3/sqrt(65),11/sqrt(65)].
- strength=0 restores terminal output in both modes. Disabled emits one head; fixed-logits emits two distinct depth heads; fixed-hidden emits one guided head. Reference iteration equal to terminal may reuse the head node, retaining semantic identity.
- Adaptive tie control terminal logits=[0,0]: probabilities=[1/2,1/2], margin=0, effective strength=max_strength. Unique top control [ln(3),0]: probabilities=[3/4,1/4], margin=1/2, effective strength=max_strength/2. Test literal effective strengths and contrasted coordinate values; never approximate top-two with a distinct-value filter.

### Full tiny load fixture

Use native `proxima_gguf::write_complete` with general.architecture=ouro, physical block_count=1, num_loops=2, embedding/head_dim=2, heads=kv_heads=1, vocab=2, FFN width=2. Write F32 tensors: token embeddings [[3,4],[0,5]], seven 2x2 matrices (Q all zero; K/V identity; output and down zero to suppress branches), four branch gamma vectors [1,1], shared output gamma [1,1], identity head, epsilon=1 (zero branch norms defined), theta explicit. Derive final vectors by repeated RMSNorm for epsilon=1, not by copying the epsilon=0 boundary fixture. For [3,4], first boundary scale = sqrt(2/27); second normalized row = [3,4]*sqrt(2/52), since the second norm acts on the first normed state. Expected baseline head = [3*sqrt(2/52),4*sqrt(2/52)]. Cache roots still differ by recurrence because K/V project differently normalized inputs. For cache independence use the separate injection fixture's exact integer rows; do not mix its assumptions into this full-load test.

## 6. Executable bounded packets and assertions

No tests run for this design task. Commands below are proposed acceptance commands; each namespace has a fixed count and concrete assertions.

S1 shared tensor changes first. `cargo nextest run -p proxima-tensor --features std -E 'test(recurrent_shared_)' --no-capture` -> 10 named cases:

1. disabled_descriptor_same_graph (complete Op vector/root equality to three immutable pre-edit TwoRange payloads: dense, routed MoE, and Gemma norms/shared donor; deliberately mutated graph/root comparison fails). Capture artifacts under results/baseline before product edits; the compatibility wrapper is not an independent oracle.
2. schedule_two_indices (physical 2 x iterations 2 -> physical [0,1,0,1], logical [0,1,2,3], exactly 12 logical cache roots components and no blk.2/blk.3 weights).
3. qk_norm_none_direct_projection (literal unnormalized [2,0], gamma-one negative control).
4. sandwich_norm_boundaries (four branch norm placement, no norm moved after residual).
5. norm_each_iteration_once (2 completed loop norms, next-loop input is first normalized output, no duplicate final norm).
6. distinct_kv_literal (integer cache vectors above, [3,0]/[7,0], poisoned padded tail, wrong donor fails).
7. cached_vs_prefill_literal (same prefix graph outputs/cache rows for prefill vs decode).
8. loopcd_logits_hidden_literal (both formulas above plus strength-zero).
9. adaptive_ties_literal (equal/unique top controls and full contrast coordinates).
10. logical_moe_and_shared_donors (recurrent routed sites labeled logical [0,1]; donor indices repeated within loop, physical weight identity unchanged; unsupported ShortConv/PLE combinations and overflow/ref out of range asserted here).

S2 registered GGUF/HF and serving identity. `cargo nextest run -p proxima-model-interop --features std -E 'test(recurrent_mainline_)' --no-capture` -> 8 named cases:

1. registry_ouro_exact (registered name and adapter invoked; normal dense fallback unchanged).
2. tiny_gguf_literal (full tiny fixture above through LoadedModel::load, literal logits/cache outputs).
3. physical_weights_once (mmap slices/owned matrix addresses identical across recurrent consumers; logical cache buffers distinct).
4. hf_shared_descriptor_literal (single native safetensors fixture plus typed config, no Q/K row permutation; same literal values as tiny GGUF).
5. settings_three_routes (env/file/bon equal resolved fields; negative depth/NaN/infinite/unused strength).
6. loader_fastpath_gate (single_range None, logical KV layer sizing equal cache-root owners; no Mistral reconstructed program).
7. serving_identity_before_work (different depth/mode rejected before plan-count/KV mutation, prefix cache key distinguishes all value-affecting settings).
8. production_decode_literal (existing serving loop prefills then decodes; exactly one cache append per logical layer per forwarded token; zero budget generation root guard preserved).

S3 adapt lab caller; production wall-time/payload evidence. `cargo nextest run -p proxima-model-interop --features 'std metal metal-output-placement instrument' -E 'test(recurrent_metal_)' --no-capture` -> 3 named real-device cases: cached literal vectors; repeat same model/prefix cache identity and raw stage events; production LoopCD readout vectors versus CPU fixture. Unsupported compiled/hardware feature paths remain typed and explicit. `cargo check -p proxima-model-interop --no-default-features` and alloc/std checks retain source compatibility. The std/metal/metal-output-placement/instrument feature names were read from the current interop manifest; the tensor std feature was read from its manifest.

For real-model native cells reuse canonical pinned BF16 GGUF `/private/tmp/ouro-llama-converted-v1/ouro-bf16.gguf`, the current GGUF tokenizer/load route, and canonical lab fixed token prefixes. Freeze parent mainline source identity and settings for each condition. Compare original shared pipeline / Ouro adaptation of that pipeline / shared mainline candidate; record full prefix IDs, every completed iteration boundary requested for diagnostic cells, both reference and terminal vectors, signed coordinate differences, and generated IDs. Native timing cells use the production serving loop with diagnostic readbacks disabled, synchronized cold load/plan and warm prefill/decode wall durations, raw sample arrays and CoV. Pinned llama.cpp is the performance oracle for the same token IDs/depth/metadata; do not interpret uncached lab timing as cached decode timing. Architecture integration alone provides no wallclock claim.

## 7. Hazards the admitted spec must close

- Logical KV depth multiplies payload capacity. Feed effective owners into existing memory-fit/kv_cache_budget before allocation; never multiply physical weight bytes by iterations. Preserve distinct buffer ownership and append exactly once per forwarded token for every logical layer.
- Host cache vs device placement changes wallclock materially. Current Custom gate avoids an incorrect Mistral graph but leaves ordinary two-range KV transfers; root must choose a shared placement-capability adaptation if real serving wallclock acceptance requires it. This is a concrete semantic/performance gap, not permission to leave a lab-only route.
- `BackendRuntime` measured candidate execution mutates placed caches if handed output placements. Candidate measurement requires caller-owned scratch KV snapshots or typed rejection until such trial state is owned; no repeated real cache append during tuning. Common configured resource subset currently rejects cached-attention workspace, so it cannot silently replace the production serving arm.
- Prefix cache/prewarm speculative verification must use identical resolved iterations/LoopCD mode/strength/reference and numeric policy. The existing exhaustive ServingConfig destructure in prompt_cache_key.rs is a useful compiler-enforced consumer seam; add the value fields there, not ignore them.
- HF row permutation must follow RoPE representation. Absence of Q/K norm is not permission to permute Ouro's split-half rows. GGUF BF16 native matrices retain existing packed storage convention; do not transpose based solely on logical dimensions.
- Shared-KV donor indices and routed MoeSite identity need logical rebasing. Recurrence means weight sharing only. An explicit schedule shared-KV donor remains the only reason to share history.
- LoopCD fixed hidden/logit modes operate at different boundaries. `hidden_root` normalized output cannot substitute for raw completed state; first/terminal roots must be from the exact same prefix, never preceding token states.
- A depth override becomes graph identity at load. Changing it mid-prefix without invalidating cache is not admitted. Learned early_exit_gate is not the current lab's fixed-depth mode; retain that distinction and reject early-exit settings until an actual contract is specified.

## 8. Opened source register

`architecture.rs:92–160` BoundProgram; `:273–434` Architecture hooks; `:499` KV memory sizing; `:606–676` registry. `dense.rs:71–168` existing bind -> descriptor -> build_forward. `descriptor.rs:1–150` descriptor; `:541–581` TwoRange dispatch. `lfm2_single_range_cached.rs:432–540` mandatory Q/K norms; `:1260–1455` schedule and cache resources; `:1485–1710` cache owner names/layer/FFN/finalnorm. `attention_forward.rs:1150–1215` norm/FFN weight naming. `bind.rs:2140–2280` physical dense binding/QK probe. `hf_bind.rs:547–690` names/bind/RoPE permutation. `hf_config.rs:171–200` current model shape conversion. `pregather.rs:2413,2638–2755,2918–2970` loader routes, fast path, KV allocation shape. `decode.rs:5900–6125` existing append/next-token and placed-KV contracts. `prompt_cache_key.rs:83–155` exhaustive config identity. `prompt_cache_settings.rs:17–94` existing Settings mirror. Lab `checkpoint.rs:16–145` pinned config/weights; `graph.rs:150–290` branch algebra; `model.rs:402–485` recurrence/boundaries; `guidance.rs:65–90` adaptive policy. Reference `ouro-modeling_ouro.py:304–322,367–424,592–606` projection/cache indices/branch and loop norm. Pinned llama source `src/models/ouro.cpp:23–95,185–208` independent load/shared weights/loop norm implementation.
