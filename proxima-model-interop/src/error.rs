//! Typed failures for the interop transform, on top of whatever the
//! underlying reader/writer surfaces.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::GgmlType;
use proxima_tensor::DType;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum InteropError {
    /// `tensor`'s `GgmlType` has no safetensors dtype counterpart — either
    /// it's block-quantized (packs multiple elements per scale/bias, which
    /// a flat typed array can't express without dequantizing) or otherwise
    /// has no fixed-width scalar equivalent.
    #[error(
        "tensor {tensor:?} has ggml type {ggml_type:?}, which has no safetensors dtype counterpart"
    )]
    UnrepresentableGgmlType { tensor: String, ggml_type: GgmlType },

    /// `tensor`'s `DType` has no `GgmlType` counterpart (`Bool` or an
    /// unsigned-integer / 128-bit width ggml never defined).
    #[error("tensor {tensor:?} has dtype {dtype:?}, which has no ggml type counterpart")]
    UnrepresentableDType { tensor: String, dtype: DType },

    /// `tensor`'s shape has more dimensions than GGUF's tensor directory
    /// can hold (`proxima_gguf::tensor::MAX_DIMS`, 4).
    #[error("tensor {tensor:?} has {found} dimensions, gguf supports at most {max}")]
    TooManyDimensions {
        tensor: String,
        found: usize,
        max: usize,
    },

    /// [`crate::bind::gguf_tensor_as_f32`] was asked for a name absent
    /// from the parsed tensor directory.
    #[error("no tensor named {name:?} in the gguf tensor directory")]
    UnknownTensor { name: String },

    #[error(transparent)]
    Gguf(#[from] proxima_gguf::GgufError),

    #[error(transparent)]
    Safetensors(#[from] proxima_safetensors::SafetensorsError),

    /// A sidecar writer failed while emitting its header, descriptors, or
    /// one expert payload to the caller-owned destination.
    #[cfg(feature = "std")]
    #[error("expert sidecar io: {0}")]
    SidecarIo(#[from] std::io::Error),

    /// A sidecar descriptor cannot encode its projection name in the fixed
    /// wire field.
    #[error("expert sidecar projection name has {found} bytes; maximum is {max}")]
    SidecarProjectionTooLong { found: usize, max: usize },

    /// A sidecar's header or data offsets overflow the representable format.
    #[error("expert sidecar size overflow")]
    SidecarSizeOverflow,

    /// A mapped sidecar did not satisfy its wire-format contract.
    #[cfg(feature = "std")]
    #[error("invalid expert sidecar: {0}")]
    InvalidExpertSidecar(String),

    /// A block-quantized tensor's bytes didn't fit its codec's own shape
    /// contract (not a whole block multiple, or an output-size mismatch)
    /// -- propagated from [`proxima_gguf::quant`] rather than re-derived.
    #[error(transparent)]
    Quant(#[from] proxima_gguf::quant::QuantError),

    /// `crate::loader::prefault`'s (`std`-gated) shared background pool failed to build,
    /// or a spawned page-touch chunk never reported back (a worker panic;
    /// `ProximaBackgroundPool` catches and discards worker panics rather
    /// than propagating them).
    #[error("prefault: {0}")]
    PrefaultPoolUnavailable(String),

    /// `crate::bind::gguf_tensor_as_packed_block` (`std`-gated) found `tensor` stored as
    /// `F32` but its absolute file offset is not a multiple of
    /// `align_of::<f32>()` -- reinterpreting the raw bytes as `&[f32]`
    /// without copying would be unsound, so the caller must fall back to
    /// [`crate::bind::gguf_tensor_as_f32`]'s owned, byte-at-a-time decode
    /// instead.
    #[error(
        "tensor {tensor:?} is f32 but its file offset is not 4-byte aligned, cannot borrow as &[f32]"
    )]
    MisalignedFloat32Tensor { tensor: String },

    /// [`crate::bind::architecture_from_metadata`] needed `key` (either
    /// `general.architecture` itself, or one of that architecture's own
    /// `{architecture}.*` dimension keys) and the parsed gguf metadata had
    /// no such key, or the key was present with the wrong `MetadataValue`
    /// variant.
    #[error("gguf metadata is missing required key {key:?}")]
    MissingMetadataKey { key: String },

    /// `crate::architecture::ArchitectureRegistry::resolve` (`std`-gated) read
    /// `general.architecture` as `name`, and no registered
    /// `crate::architecture::Architecture` (`std`-gated) declares that name, nor is
    /// the registry's own fallback architecture set to catch it -- a
    /// foreign crate loading a checkpoint whose architecture it never
    /// registered gets this instead of a panic or a silent misbind.
    #[error("no registered architecture matches general.architecture = {name:?}")]
    UnknownArchitecture { name: String },

    /// [`crate::bind::architecture_from_metadata`]'s vocab derivation: the
    /// `token_embd.weight` tensor's element count did not divide evenly by
    /// `embedding_length`.
    #[error(
        "token_embd.weight has {elements} elements, which does not divide evenly by embedding_length {embedding}"
    )]
    VocabShapeMismatch { elements: u64, embedding: u32 },

    /// [`crate::bind::architecture_from_metadata`] found `key` (e.g.
    /// `{architecture}.attention.head_count_kv`) stored as a per-layer
    /// [`proxima_gguf::value::MetadataArray`] whose `distinct_values` are not
    /// all equal -- confirmed against a real hybrid checkpoint
    /// (LFM2.5-8B-A1B, whose convolution layers report `0` kv heads and
    /// whose attention layers report a real count in the SAME array).
    /// [`crate::bind::ModelArchitecture`]'s single `u32` field cannot
    /// represent genuine per-layer variation, so this surfaces as a typed,
    /// named gap rather than silently picking one layer's value (the max,
    /// the first nonzero, ...) and presenting it as if it applied uniformly.
    #[error(
        "gguf metadata key {key:?} has {distinct_values} distinct per-layer values; ModelArchitecture cannot represent per-layer variation"
    )]
    HeterogeneousMetadataArray { key: String, distinct_values: usize },

    /// A per-layer GGUF metadata array did not provide exactly one value for
    /// every declared transformer block, so no architecture can align its
    /// configuration to the tensor directory safely.
    #[error(
        "gguf metadata key {key:?} has {found} per-layer values, expected block_count {expected}"
    )]
    MetadataArrayLengthMismatch {
        key: String,
        expected: usize,
        found: usize,
    },

    /// The checkpoint family was recognized from its header, but its
    /// architecture-specific forward program has not been supplied yet.
    /// This is deliberately distinct from an unknown architecture or a
    /// malformed generic configuration: callers can inspect or route the
    /// header without pretending a dense program is valid for a hybrid MoE.
    #[error("architecture {name:?} needs its own hybrid MoE forward program")]
    HybridMoeProgramUnsupported { name: String },

    /// `crate::generate`'s cached forward program failed to build or
    /// evaluate -- propagated from `proxima_tensor` rather than re-derived.
    #[error(transparent)]
    Tensor(#[from] proxima_tensor::TensorError),

    /// `crate::generate`'s prompt encode/decode step failed --
    /// propagated from `proxima_tokenizer` rather than re-derived.
    #[cfg(feature = "std")]
    #[error(transparent)]
    Tokenizer(#[from] proxima_tokenizer::TokenizerError),

    /// [`crate::generate::LoadedModel`]'s evaluator ran but `node` (one of
    /// the logits root or a per-layer cache root) is absent from its
    /// output -- an interpreter/program-construction invariant violation
    /// rather than a caller mistake, surfaced instead of panicking.
    #[cfg(feature = "std")]
    #[error("evaluator output is missing node {node:?}")]
    MissingEvaluatedNode { node: proxima_tensor::op::NodeId },

    /// [`crate::generate::LoadedModel`]'s greedy pick step ran against an
    /// empty logits slice.
    #[cfg(feature = "std")]
    #[error("greedy_pick: logits slice is empty")]
    EmptyLogits,

    /// [`crate::generate::LoadedModel`]'s decode loop read `logits_root` and
    /// found more than one row of `vocab` -- the `Architecture` contract
    /// (`crate::architecture`'s doc on `BoundProgram::logits_root`) requires
    /// exactly one row, the `lm_head_row`-gathered last position; a foreign
    /// `Architecture` that skips that gather would otherwise be silently
    /// sampled at row 0 instead of the last token.
    #[cfg(feature = "std")]
    #[error(
        "logits_root evaluated to {found_rows} row(s) of vocab {vocab}, expected exactly {expected_rows}"
    )]
    LogitsShapeMismatch {
        expected_rows: usize,
        found_rows: usize,
        vocab: usize,
    },

    /// [`crate::generate::LoadedModel`]'s decode loop asked
    /// [`omega::backend`] to plan or execute a forward step and the backend
    /// itself refused -- an unrecognized/uncompiled backend name, or a
    /// codec the chosen backend's driver has no kernel for.
    #[cfg(feature = "metal")]
    #[error(transparent)]
    Backend(#[from] omega::backend::BackendError),

    /// [`crate::generate::find_input_node`] scanned the single-range
    /// program for an [`proxima_tensor::Op::Input`] named `name` and found
    /// none -- would mean [`crate::generate::build_single_range_program`]'s
    /// own `kv_cache.{layer}.*` naming has drifted out of sync with
    /// [`proxima_tensor::spec::mistral_single_range_cached_forward_program`]'s.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[error("single-range program has no input node named {0:?}")]
    UnboundInputName(String),

    /// [`crate::generate::BackendRuntime::evaluate_with_placements`]'s
    /// direct `omega::metal` plan/execute call (bypassing the
    /// backend-polymorphic `omega::backend` entry point, since
    /// [`omega::PlacedBuffer`] is Metal-only) failed -- propagated from
    /// `omega::metal` rather than re-derived.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[error(transparent)]
    Metal(#[from] omega::metal::MetalError),

    /// [`crate::serving::apply_serving_config`]'s prompt-length precondition:
    /// `sequence` (the tokenized prompt length) exceeds the caller's own
    /// configured `context_length` (`-c`).
    #[error("prompt sequence {sequence} exceeds configured context_length {context_length} (-c)")]
    SequenceExceedsContextLength {
        sequence: usize,
        context_length: u32,
    },

    /// [`crate::serving::apply_serving_config`] found a [`crate::serving::ServingConfig`]
    /// field requesting behavior this forward path does not implement yet --
    /// what used to be a `todo!` naming the gap now surfaces as data, since a
    /// caller (not just this crate) picks the config fields.
    #[error("unsupported serving config: {0}")]
    UnsupportedServingConfig(String),

    /// The requested routed execution phase is not available for the bound
    /// architecture or its forward graph.  This is preferable to silently
    /// running the single-pass graph when a caller requires a residency
    /// transition before the expert gather.
    #[cfg(feature = "std")]
    #[error("pre-gather execution is unavailable for architecture {architecture:?}: {reason}")]
    PreGatherExecutionUnsupported {
        architecture: String,
        reason: String,
    },

    /// `crate::bind::transpose_expert_stack`'s decoded element count did
    /// not equal `expert_count * out_dim * in_dim` — the tensor directory's
    /// own declared shape for a stacked MoE weight disagrees with the
    /// `general.expert_count`/architecture hparams a malformed or
    /// adversarial GGUF file could set independently of it. Caught here
    /// rather than sliced past, which would otherwise panic mid-transpose.
    #[error(
        "moe expert stack {tensor:?} has {elements} elements, but expert_count={expert_count} \
         out_dim={out_dim} in_dim={in_dim} needs {expected}"
    )]
    MoeExpertShapeMismatch {
        tensor: String,
        elements: usize,
        expert_count: usize,
        out_dim: usize,
        in_dim: usize,
        expected: usize,
    },

    /// `crate::bind::transpose_out_in_to_in_out`'s decoded element count
    /// did not equal `out_dim * in_dim` — same disagreement as
    /// [`Self::MoeExpertShapeMismatch`], but for a plain (non-MoE) matmul
    /// weight: the tensor directory's own declared shape disagrees with
    /// the architecture hparams `out_dim`/`in_dim` were derived from.
    #[error(
        "weight {tensor:?} has {elements} elements, but out_dim={out_dim} in_dim={in_dim} needs {expected}"
    )]
    DenseWeightShapeMismatch {
        tensor: String,
        elements: usize,
        out_dim: usize,
        in_dim: usize,
        expected: usize,
    },

    /// A checkpoint declared only part of a layer's Q/K/V bias family, or a
    /// bias vector whose element count disagrees with the checkpoint's head
    /// geometry.  Biases are a semantic model parameter; silently dropping a
    /// partial family would make the model executable but incorrect.
    #[error(
        "layer {layer} QKV bias {projection:?} has {elements} elements, expected {expected}"
    )]
    QkvBiasShapeMismatch {
        layer: u32,
        projection: String,
        elements: usize,
        expected: usize,
    },

    /// One or more members of a layer's Q/K/V bias family were absent while
    /// another member was present.
    #[error("layer {layer} QKV bias family is incomplete; missing {missing:?}")]
    QkvBiasFamilyIncomplete { layer: u32, missing: Vec<String> },

    /// [`crate::hf_config::parse_hf_config`]'s `config.json` bytes were not
    /// valid JSON, or were missing/mis-typing one of [`crate::hf_config::HfConfig`]'s
    /// required fields (`hidden_size`, `num_attention_heads`,
    /// `num_hidden_layers`, `intermediate_size`, or `vocab_size`).
    #[error("malformed hf config.json: {reason}")]
    MalformedHfConfig { reason: String },

    /// `crate::hf_bind`'s weight binder found a safetensors tensor whose
    /// declared [`DType`] this crate has no decoder for -- only
    /// `Float32`/`Float16`/`BFloat16` dense weights are supported (an
    /// unquantized HF checkpoint's own on-disk types); an integer dtype, a
    /// quantized layout's packed integer payload (e.g. MLX's `U32`), or any
    /// other unhandled type surfaces here rather than misreading bytes.
    #[error(
        "tensor {tensor:?} has dtype {dtype:?}, which this crate has no dense-weight decoder for"
    )]
    UndecodableSafetensorsDType { tensor: String, dtype: DType },

    /// `crate::hf_bind::bind_all_weights_from_safetensors` was asked to
    /// bind a checkpoint whose [`crate::bind::ModelArchitecture::expert_count`]
    /// is nonzero -- HF's own mixture-of-experts tensor-naming convention
    /// (Mixtral's per-expert `block_sparse_moe.experts.{e}.*` vs. Qwen's
    /// `mlp.experts.{e}.*`, neither confirmed against a real on-disk
    /// safetensors checkpoint on this host, since the only MoE checkpoint
    /// available here is MLX's packed `weight`/`scales`/`biases` layout,
    /// explicitly out of scope) is not yet implemented -- a caller gets a
    /// typed, named gap rather than a silent wrong bind.
    #[error(
        "checkpoint has expert_count={expert_count}, but hf mixture-of-experts weight binding is not implemented (dense hf checkpoints only)"
    )]
    HfMoeWeightsUnsupported { expert_count: u32 },

    /// `crate::lfm2::bind_lfm2_shortconv_in_proj`'s fused `blk.{layer}.shortconv.in_proj.weight`
    /// did not have exactly `3 * embedding * embedding` elements -- the
    /// real checkpoint's own declared shape disagrees with the
    /// `embedding` this call derived from `lfm2moe.embedding_length`.
    #[error(
        "blk.{layer}.shortconv.in_proj.weight has {elements} elements, but embedding={embedding} needs {expected} (3 * embedding * embedding)"
    )]
    ShortConvInProjShapeMismatch {
        layer: u32,
        elements: u64,
        embedding: u32,
        expected: u64,
    },

    /// `crate::lfm2::bind_lfm2_shortconv_in_proj`'s row-split precondition:
    /// `embedding` (the row width, GGUF's `in_dim` axis) is not a whole
    /// multiple of the fused tensor's own codec `block_elements` -- never
    /// observed on the real checkpoint (`embedding = 2048 = 8 * 256`), but
    /// a row-boundary split is only provably block-aligned when this holds,
    /// so it is checked rather than assumed.
    #[error(
        "blk.{layer}.shortconv.in_proj.weight has ggml type {ggml_type:?}, whose block size does not evenly divide embedding={embedding}"
    )]
    ShortConvInProjNotBlockAligned {
        layer: u32,
        ggml_type: GgmlType,
        embedding: u32,
    },

    /// `crate::lfm2::lfm2_architecture_from_metadata`'s (`std`-gated) `key` (e.g.
    /// `lfm2moe.attention.head_count_kv`) is a per-layer array whose
    /// nonzero entries (the real attention layers' own kv head count)
    /// disagree with each other -- the zero entries (convolution layers)
    /// are expected and skipped, but every attention layer must still
    /// share one real kv head count for `crate::lfm2::Lfm2Architecture::kv_heads`
    /// to mean anything.
    #[error(
        "gguf metadata key {key:?} has {distinct_values} distinct nonzero per-layer values; Lfm2Architecture cannot represent per-layer variation"
    )]
    HeterogeneousNonzeroMetadataArray { key: String, distinct_values: usize },

    /// `crate::qwen35::bind_qwen35_attn_qkv_split`'s fused
    /// `blk.{layer}.attn_qkv.weight` did not have exactly
    /// `embedding * (2 * key_dim + value_dim)` elements -- the real
    /// checkpoint's own declared shape disagrees with the row boundaries
    /// this call derived from `qwen35.ssm.state_size` /
    /// `qwen35.ssm.group_count` / `qwen35.ssm.inner_size`.
    #[error(
        "blk.{layer}.attn_qkv.weight has {elements} elements, but embedding={embedding}, key_dim={key_dim}, value_dim={value_dim} needs {expected} (embedding * (2 * key_dim + value_dim))"
    )]
    QwenQkvShapeMismatch {
        layer: u32,
        elements: u64,
        embedding: u32,
        key_dim: u32,
        value_dim: u32,
        expected: u64,
    },

    /// `crate::qwen35::bind_qwen35_attn_qkv_split`'s row-split precondition:
    /// `embedding` (the row width, GGUF's `in_dim` axis) is not a whole
    /// multiple of the fused tensor's own codec `block_elements` -- a
    /// row-boundary split is only provably block-aligned when this holds.
    #[error(
        "blk.{layer}.attn_qkv.weight has ggml type {ggml_type:?}, whose block size does not evenly divide embedding={embedding}"
    )]
    QwenQkvNotBlockAligned {
        layer: u32,
        ggml_type: GgmlType,
        embedding: u32,
    },

    /// [`crate::quality::parse_prompts_jsonl`]'s line `line_number` (1-based)
    /// was not valid JSON, or was valid JSON missing/mis-typing one of
    /// [`crate::quality::Prompt`]'s required fields.
    #[cfg(feature = "std")]
    #[error("quality prompt fixture line {line_number}: {reason}")]
    MalformedQualityPrompt { line_number: usize, reason: String },

    /// [`crate::memory_fit::fit_context_length`]: even a context length of
    /// `1` cannot fit this checkpoint's own weights (by class -- dense,
    /// mixture-of-experts, embedding/output tables, SSM state) plus the
    /// fixed arena allowance inside `limit_bytes - os_headroom_bytes` --
    /// `generate_with_serving_config` returns this before
    /// `BackendRuntime::new` uploads a single weight
    /// (`ServingConfig::gpu_memory_fit`'s own doc). Every field names one
    /// class so a later allocation step (per-expert precision as a
    /// budget-constrained top-n selection, `crate::memory_fit`'s own module
    /// doc) can read the record without re-deriving the split.
    // mirrors `crate::memory_fit`'s own module gate exactly
    // (`lib.rs`'s `mod memory_fit`) rather than `feature = "std"`: that
    // module is forced into every `cargo test` build regardless of
    // features (so its own unit tests always run), so a variant it
    // constructs must be reachable there too -- `feature = "std"` alone
    // left this variant absent under a bare `cargo nextest run` with no
    // features, breaking the module that names it in its own doc comment.
    #[cfg(any(test, all(feature = "std", feature = "metal")))]
    #[error(
        "load-time memory budget exceeded: dense_weights_bytes={dense_weights_bytes} \
         expert_weights_bytes={expert_weights_bytes} table_weights_bytes={table_weights_bytes} \
         kv_cache_bytes={kv_cache_bytes} ssm_state_bytes={ssm_state_bytes} \
         arena_allowance_bytes={arena_allowance_bytes} exceeds limit_bytes={limit_bytes} \
         minus os_headroom_bytes={os_headroom_bytes}"
    )]
    MemoryBudgetExceeded {
        dense_weights_bytes: u64,
        expert_weights_bytes: u64,
        table_weights_bytes: u64,
        kv_cache_bytes: u64,
        ssm_state_bytes: u64,
        arena_allowance_bytes: u64,
        limit_bytes: u64,
        os_headroom_bytes: u64,
    },

    /// `crate::bind`'s per-tensor precision recode
    /// (`crate::serving::ServingConfig::weight_precision`) matched `tensor`
    /// against a rule naming `target`, but [`proxima_gguf::quant`] ships no
    /// encoder for `target` (`Q2_K`, the `Iq*` family, `Q4_1`/`Q5_1`/`Q8_1`,
    /// or an integer/`F64`/`Tq*` type) -- silently keeping the tensor at its
    /// on-disk codec instead would make the rule a no-op nobody could see,
    /// so this surfaces as a typed, named gap instead.
    #[cfg(feature = "std")]
    #[error(
        "weight_precision rule for {tensor:?} names target {target:?}, which proxima_gguf::quant has no encoder for"
    )]
    UnsupportedWeightPrecisionTarget { tensor: String, target: GgmlType },

    /// `crate::Architecture::step_inputs` (feature-gated behind `std`) returned a
    /// `crate::StepInput` (feature-gated behind `std`) whose name does not match any
    /// [`proxima_tensor::Op::Input`] leaf in this checkpoint's own forward
    /// program -- the architecture computed a leaf the program never
    /// declared, a caller mistake surfaced as data rather than the value
    /// silently sitting in `named_blocks` unread.
    #[error(
        "architecture step_inputs returned {name:?}, which this program declares no Op::Input leaf for"
    )]
    UnknownStepInput { name: String },

    /// A forward program leaf beyond the decode loop's own builtin
    /// `ids`/`eps`/`rope_cos`/`rope_sin`/`cached_len`/`kv_cache.*` set was
    /// left unbound after `crate::Architecture::step_inputs` (feature-gated
    /// behind `std`) ran -- either the architecture's own
    /// `crate::Architecture::bind` declared a leaf its
    /// `crate::Architecture::step_inputs` never feeds, or
    /// (the default, no-op override) an architecture with a custom leaf
    /// never overrode `crate::Architecture::step_inputs` at
    /// all.
    #[error(
        "forward program leaf {name:?} is left unbound; no builtin block and no architecture step_inputs supplied it"
    )]
    MissingStepInput { name: String },

    /// `crate::Architecture::step_inputs` (feature-gated behind `std`) returned a
    /// `crate::StepInput` (feature-gated behind `std`) whose `symbol` names a slot the
    /// decode loop itself already binds
    /// (`crate::symbols::NEW_COUNT`/`crate::symbols::KV_BOUND`)
    /// -- a foreign architecture's own slot must start at
    /// `crate::symbols::FIRST_FREE` (feature-gated behind `std`), never overwrite a
    /// builtin one out from under the loop.
    #[error(
        "architecture step_inputs named reserved symbol slot {slot}; foreign slots start at FIRST_FREE"
    )]
    ReservedSymbolSlot { slot: u16 },

    /// `crate::architecture::BoundProgram::single_position_step` is set
    /// (today: only `crate::qwen35::Qwen35Arch`'s own gated-DeltaNet mixer,
    /// whose `s`-axis reduce sums across positions instead of stepping
    /// through them -- `proxima_tensor::error::TensorError::SingleTokenStepOnly`'s
    /// own doc) but this call bound `new_count` (`crate::symbols::NEW_COUNT`)
    /// to more than one position -- a batched multi-token prefill, which
    /// this architecture's program would evaluate wrong rather than raise
    /// on its own, since `s` is `Extent::Symbolic` there and only resolved
    /// here, at bind time. The decode loop must feed this architecture one
    /// position per evaluation instead (prefill becomes `new_count`
    /// sequential evaluations of `new_count == 1`, the same path decode
    /// already takes).
    #[error(
        "architecture declares single_position_step but new_count = {new_count}; feed one position per evaluation"
    )]
    MultiPositionStepUnsupported { new_count: usize },

    /// `layer`'s `crate::architecture::BoundProgram::layer_roots` entry
    /// names a cache shape (`bound`) that disagrees with what the compiled
    /// program actually declares as `Op::Input` leaves for that layer
    /// (`declared`) -- an `crate::architecture::Architecture::bind` that
    /// tagged the wrong `proxima_tensor::spec::Qwen35LayerRoots` variant
    /// for a layer it built correctly otherwise. Caught once, at decode-loop
    /// setup, instead of surfacing later as a confusing
    /// [`Self::MissingStepInput`] on a leaf the decode loop never even
    /// tried to feed under the bound (wrong) shape.
    #[error(
        "layer {layer} cache shape mismatch: program declares {declared} inputs, layer_roots bound {bound}"
    )]
    LayerCacheKindMismatch {
        layer: usize,
        declared: &'static str,
        bound: &'static str,
    },

    /// `layer`'s `crate::architecture::BoundProgram::layer_roots` entry
    /// says `kind` (recurrent SSM state or attention KV state), but the
    /// compiled program declares NONE of `expected` as `Op::Input` leaves
    /// for that layer -- the architecture baked this layer's state as
    /// constants instead of feeding it through the decode loop. Unlike
    /// [`Self::LayerCacheKindMismatch`] (program declares a DIFFERENT
    /// shape than bound), this is the program declaring NO cache shape at
    /// all for a layer the architecture itself says is stateful. Left
    /// uncaught, the decode loop silently runs that layer from zero state
    /// on every step (`declared_cache_kind`'s own doc): nothing fails, and
    /// the output degrades with a period equal to the interval between
    /// stateful layers, which is invisible on short generations. Caught
    /// once, at decode-loop setup, before the first token.
    #[error(
        "layer {layer} is bound as {kind} but the program declares none of its cache leaves ({expected:?})"
    )]
    LayerCacheLeavesMissing {
        layer: usize,
        kind: &'static str,
        expected: &'static [&'static str],
    },

    /// `crate::expert_slab::ExpertSlab::page_expert`/`evict_expert` was
    /// called while a decode step was in progress -- the slab's own borrow
    /// contract (`ExpertSlab`'s doc: an `ExpertSource` snapshot is valid for
    /// exactly one step) forbids mutating an expert's bytes while a running
    /// step may still be reading them through that snapshot. Callable again
    /// once the step that produced the error has returned.
    #[error("cannot page or evict expert {expert} of layer {layer}: a decode step is in progress")]
    ExpertSwapDuringStep { layer: usize, expert: usize },

    /// `crate::expert_slab::ExpertSlab::page_expert` was asked to page an
    /// `(layer, expert)` pair the slab was never built with -- the slab's
    /// shape is fixed at construction from the checkpoint's own layer/expert
    /// count, never grown at runtime.
    #[error("expert {expert} of layer {layer} is out of range for this checkpoint's expert slab")]
    ExpertSlabIndexOutOfRange { layer: usize, expert: usize },

    /// The monolithic all-low diagnostic was requested after a residency
    /// transition replaced one sidecar-backed low projection. That arm binds
    /// the complete low table before routing and therefore cannot mix a high
    /// checkpoint entry into the same snapshot.
    #[error(
        "monolithic all-low expert source requires sidecar bytes for layer {layer} expert {expert} projection {projection}"
    )]
    ExpertAllLowSourceRequired {
        layer: usize,
        expert: usize,
        projection: &'static str,
    },

    /// `crate::expert_slab::ExpertSlab::page_expert_mapped` received a byte
    /// range that does not fit its supplied mmap. The range is rejected
    /// before the slab records the mapping, so no later evaluation can slice
    /// beyond the source file.
    #[error("expert mmap range {start}..{end} is outside the mapping of {mapping_len} bytes")]
    ExpertMappedRangeOutOfBounds {
        start: usize,
        end: usize,
        mapping_len: usize,
    },

    /// `layer` has at least one evicted expert with no paged replacement
    /// yet (`crate::expert_slab::ExpertSlab::first_incomplete_layer`), and
    /// the selected backend does not consult
    /// `crate::expert_slab::ExpertSlab`'s per-step table at all. Metal now
    /// consumes uniform packed-codec tables; this error remains for a
    /// backend that cannot honor the table rather than silently gathering
    /// the checkpoint's original, evicted bytes.
    #[error(
        "layer {layer} has an evicted expert with no replacement, and this backend cannot honor per-step expert routing"
    )]
    ExpertRoutingUnsupportedByBackend { layer: usize },

    /// A pad-scratch buffer's row width (`expected`, elements) came out
    /// smaller than the growing per-layer cache it was about to copy from
    /// (`found`) -- `crate::generate::Qwen35DenseAttentionPadScratch::fill`
    /// (and its `KvPadScratch` counterpart)'s own row widths are read back
    /// off `layer`'s `kv_cache.{layer}.*` `Op::Input` leaves as declared by
    /// the bound program, which is authoritative; this only fires if a
    /// foreign `crate::architecture::Architecture::bind` compiled a program
    /// whose declared cache-leaf shape is narrower than the cache rows it
    /// actually appends per step, a bind-time defect this scratch resize
    /// cannot self-heal from. Previously an unchecked `copy_from_slice`
    /// panic (`range end index out of range for slice of length N`); this
    /// is that same condition, named.
    #[error(
        "layer {layer} cache scratch {leaf:?} is sized for {expected} elements, but the cache holds {found}"
    )]
    CacheScratchShapeMismatch {
        layer: usize,
        leaf: &'static str,
        expected: usize,
        found: usize,
    },
}
