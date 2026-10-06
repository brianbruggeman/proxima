//! Typed failures for the interop transform, on top of whatever the
//! underlying reader/writer surfaces.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::GgmlType;
use proxima_primitives::Codec;
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

    /// A kv block file could not be opened or mapped.
    #[cfg(feature = "std")]
    #[error("kv block file io at {path:?}: {source}")]
    BlockFileIo { path: std::path::PathBuf, source: std::io::Error },

    /// A sidecar descriptor cannot encode its projection name in the fixed
    /// wire field.
    #[error("expert sidecar projection name has {found} bytes; maximum is {max}")]
    SidecarProjectionTooLong { found: usize, max: usize },

    /// A kv block file's header or planes disagree with each other.
    #[error("kv block file is malformed: {reason}")]
    BlockFileMalformed { reason: &'static str },

    /// A kv block file was written for a different model than the one loading it.
    #[error("kv block file was written for another model: expected digest {expected:02x?}, found {found:02x?}")]
    BlockFileDigestMismatch { expected: [u8; 16], found: [u8; 16] },

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

    /// `crate::bind::as_block` was asked for `codec`, and no
    /// `proxima_tensor::cpu::QuantizedBlock` variant decodes it -- one of
    /// the codecs [`Codec`] recognizes for GPU-select/sidecar identity but
    /// that this crate's CPU decode path has never implemented.
    #[error("codec {codec:?} has no QuantizedBlock decoder")]
    UnsupportedCodec { codec: Codec },

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

    /// an in-flight rewind reached rows a seal made immutable
    #[error("cannot rewind to {keep_positions} rows: rows below {sealed_end} are sealed")]
    RewindIntoSealed {
        keep_positions: usize,
        sealed_end: usize,
    },

    /// [`crate::profiles::family_profile`]: no embedded profile file exists for
    /// this family string. Never defaulted: a family whose profile is absent
    /// would lower with another family's activation, scales and norm shifts.
    #[error("no family profile for general.architecture or model_type = {family:?}")]
    MissingFamilyProfile { family: String },

    /// [`crate::profiles::family_profile`]: the embedded profile file for this
    /// family did not parse into a [`proxima_tensor::spec::FamilyProfile`].
    #[error("family profile for {family:?} does not parse: {message}")]
    InvalidFamilyProfile { family: String, message: String },

    /// [`crate::profiles::binding_profile`]: an embedded binding file did not
    /// parse into a [`crate::profiles::BindingProfile`].
    #[error("binding profile for {family:?} does not parse: {message}")]
    InvalidBindingProfile { family: String, message: String },

    /// [`crate::bind_leaves::bind_program_leaves`]: a program leaf and the
    /// tensor that satisfies it disagree on how many elements the weight has.
    #[error(
        "leaf {leaf:?} declares {leaf_elements} elements but tensor {tensor:?} holds {tensor_elements}"
    )]
    LeafShapeMismatch {
        leaf: String,
        leaf_elements: u64,
        tensor: String,
        tensor_elements: u64,
    },

    /// [`crate::bind_leaves::bind_program_leaves`]: the program contracts some
    /// of a leaf's axes and keeps others, interleaved, so no single byte order
    /// of the stored matrix is the one the program reads.
    #[error("leaf {leaf:?} interleaves its contracted and kept axes; a stored matrix cannot match that order")]
    LeafAxesInterleaved { leaf: String },

    /// [`crate::bind_leaves::bind_program_leaves`]: a `part` alias cannot cut
    /// `tensor` into equal whole-block row slices.
    #[error("tensor {tensor:?} has {rows} rows, which do not split into {of} whole-block parts")]
    TensorPartInvalid {
        tensor: String,
        rows: u64,
        of: u32,
    },

    /// The uniform header reader: the header's `<arch>.rope.dimension_count`
    /// differs from the head width, so RoPE rotates only part of each head.
    #[error("{family:?} rotates {rope_dimension_count} of {head_dim} head dims; the single-range dense program rotates the full head")]
    PartialRotaryUnsupported { family: String, rope_dimension_count: u32, head_dim: u32 },

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
    /// [`crate::bind::ModelHparams`]'s single `u32` field cannot
    /// represent genuine per-layer variation, so this surfaces as a typed,
    /// named gap rather than silently picking one layer's value (the max,
    /// the first nonzero, ...) and presenting it as if it applied uniformly.
    #[error(
        "gguf metadata key {key:?} has {distinct_values} distinct per-layer values; ModelHparams cannot represent per-layer variation"
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

    /// The serving state machine (`proxima_core::ServingState`) refused the
    /// transition the decode loop attempted -- the loop and the machine
    /// disagree about which evaluation shape is legal, surfaced instead of
    /// continuing from an unknown state.
    #[cfg(feature = "std")]
    #[error(transparent)]
    ServingFsm(#[from] proxima_core::ServingFsmError),

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
    /// found more than one row of `vocab` -- the bound-program contract
    /// (`crate::lowering`'s doc on `BoundProgram::logits_root`) requires
    /// exactly one row, the `lm_head_row`-gathered last position; a program
    /// that skips that gather would otherwise be silently sampled at row 0
    /// instead of the last token.
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

    /// `crate::generate::find_input_node` scanned the single-range
    /// program for an [`proxima_tensor::Op::Input`] named `name` and found
    /// none -- would mean `crate::generate::build_single_range_program`'s
    /// own `kv_cache.{layer}.*` naming has drifted out of sync with
    /// [`proxima_tensor::spec::mistral_single_range_cached_forward_program`]'s.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[error("single-range program has no input node named {0:?}")]
    UnboundInputName(String),

    /// `crate::generate::BackendRuntime::evaluate_with_placements`'s
    /// direct `omega::metal` plan/execute call (bypassing the
    /// backend-polymorphic `omega::backend` entry point, since
    /// [`omega::PlacedBuffer`] is Metal-only) failed -- propagated from
    /// `omega::metal` rather than re-derived.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    #[error(transparent)]
    Metal(#[from] omega::metal::MetalError),

    /// `write_attn_read_source_vectors`'s (`PROXIMA_ATTN_VECTORS_DIR`)
    /// per-node kernel dump: `omega::emit` or `omega::metal::
    /// pack_uniforms_for` failed for one of the 15 target nodes' own
    /// `BoundOp` from the real per-step bound plan.
    #[cfg(all(
        feature = "metal",
        feature = "metal-fuse-attn-decode",
        feature = "metal-output-placement",
        target_os = "macos"
    ))]
    #[error("attn node dump: node {node} emit failed: {reason}")]
    AttnNodeDumpFailed { node: u32, reason: String },

    /// [`crate::serving::apply_serving_config`]'s prompt-length precondition:
    /// `sequence` (the tokenized prompt length) exceeds the caller's own
    /// configured `context_length` (`-c`).
    #[error("prompt sequence {sequence} exceeds configured context_length {context_length} (-c)")]
    SequenceExceedsContextLength {
        sequence: usize,
        context_length: u32,
    },

    /// [`crate::serving::resolve_context_length`]: an explicit context
    /// length above what the checkpoint was trained for (or its
    /// [`crate::rope_scaling::RopeScaling`] admits) while
    /// `ContextLength::Within` is requested.
    #[error(
        "requested context {requested} exceeds the limit {limit} under rope scaling {scaling:?}; \
         request ContextLength::Extrapolate to run past it"
    )]
    ContextExceedsTrained {
        requested: u32,
        limit: u32,
        scaling: crate::rope_scaling::RopeScaling,
    },

    /// `{arch}.rope.scaling.type` named a scaling law this crate does not
    /// implement.
    #[error("unknown rope scaling type {found:?}: expected none, linear, or yarn")]
    UnknownRopeScalingType { found: String },

    /// [`crate::serving::apply_serving_config`] found a [`crate::serving::ServingConfig`]
    /// field requesting behavior this forward path does not implement yet --
    /// what used to be a `todo!` naming the gap now surfaces as data, since a
    /// caller (not just this crate) picks the config fields.
    #[error("unsupported serving config: {0}")]
    UnsupportedServingConfig(String),

    /// `generate::decode::run_decode_loop_placed_kv`'s per-step
    /// `omega::metal::MetalStageTotals::gpu_exec_calls` reading exceeded
    /// [`crate::serving::ServingConfig::max_command_buffers_per_token`].
    #[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
    #[error(
        "step {step}: {committed} metal command buffers committed, exceeds max_command_buffers_per_token={limit}"
    )]
    TooManyCommandBuffers {
        step: usize,
        committed: u64,
        limit: usize,
    },

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
    #[error("layer {layer} QKV bias {projection:?} has {elements} elements, expected {expected}")]
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
    /// bind a checkpoint whose [`crate::bind::ModelHparams::expert_count`]
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

    /// `crate::short_conv::lfm2_architecture_from_metadata`'s (`std`-gated) `key` (e.g.
    /// `lfm2moe.attention.head_count_kv`) is a per-layer array whose
    /// nonzero entries (the real attention layers' own kv head count)
    /// disagree with each other -- the zero entries (convolution layers)
    /// are expected and skipped, but every attention layer must still
    /// share one real kv head count for `crate::short_conv::Lfm2Hparams::kv_heads`
    /// to mean anything.
    #[error(
        "gguf metadata key {key:?} has {distinct_values} distinct nonzero per-layer values; Lfm2Hparams cannot represent per-layer variation"
    )]
    HeterogeneousNonzeroMetadataArray { key: String, distinct_values: usize },

    /// `crate::recurrent_interval::bind_qwen35_attn_qkv_split`'s fused
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

    /// `crate::recurrent_interval::bind_qwen35_attn_qkv_split`'s row-split precondition:
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

    /// `crate::memory_fit::fit_context_length`: even a context length of
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

    /// A `crate::StepInput` (feature-gated behind `std`) whose name does not
    /// match any [`proxima_tensor::Op::Input`] leaf in this checkpoint's own
    /// forward program -- the step computed a leaf the program never declared,
    /// a mistake surfaced as data rather than the value silently sitting in
    /// `named_blocks` unread.
    #[error(
        "step input {name:?} named, which this program declares no Op::Input leaf for"
    )]
    UnknownStepInput { name: String },

    /// A forward program leaf beyond the decode loop's own builtin
    /// `ids`/`eps`/`rope_cos`/`rope_sin`/`cached_len`/`kv_cache.*` set was
    /// left unbound after the step inputs ran -- the lowered program declared a
    /// leaf that neither the builtin blocks nor `crate::sliding_rope_inputs`
    /// feeds, so a config that adds such a leaf must also say where its values
    /// come from.
    #[error(
        "forward program leaf {name:?} is left unbound; no builtin block and no step input supplied it"
    )]
    MissingStepInput { name: String },

    /// A `crate::StepInput` (feature-gated behind `std`) whose `symbol` names a
    /// slot the decode loop itself already binds
    /// (`crate::symbols::NEW_COUNT`/`crate::symbols::KV_BOUND`)
    /// -- a step input's own slot must start at
    /// `crate::symbols::FIRST_FREE` (feature-gated behind `std`), never overwrite a
    /// builtin one out from under the loop.
    #[error(
        "step input named reserved symbol slot {slot}; step input slots start at FIRST_FREE"
    )]
    ReservedSymbolSlot { slot: u16 },

    /// `crate::lowering::BoundProgram::single_position_step` is set
    /// (today: only a schedule with a gated-DeltaNet layer, whose `s`-axis reduce sums across positions instead of stepping
    /// through them -- `proxima_tensor::error::TensorError::SingleTokenStepOnly`'s
    /// own doc) but this call bound `new_count` (`crate::symbols::NEW_COUNT`)
    /// to more than one position -- a batched multi-token prefill, which
    /// this program would evaluate wrong rather than raise on its own, since
    /// `s` is `Extent::Symbolic` there and only resolved here, at bind time.
    /// The decode loop must feed this program one position per evaluation
    /// instead (prefill becomes `new_count`
    /// sequential evaluations of `new_count == 1`, the same path decode
    /// already takes).
    #[error(
        "program declares single_position_step but new_count = {new_count}; feed one position per evaluation"
    )]
    MultiPositionStepUnsupported { new_count: usize },

    /// `layer`'s `crate::lowering::BoundProgram::layer_roots` entry
    /// names a cache shape (`bound`) that disagrees with what the compiled
    /// program actually declares as `Op::Input` leaves for that layer
    /// (`declared`) -- a bound program whose layer roots tag the wrong
    /// `proxima_tensor::spec::LayerCacheRoots` variant for a layer it built
    /// correctly otherwise. Caught once, at decode-loop
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

    /// `layer`'s `crate::lowering::BoundProgram::layer_roots` entry
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
    /// (`found`) -- `crate::generate::DenseAttentionPadScratch::fill`
    /// (and its `KvPadScratch` counterpart)'s own row widths are read back
    /// off `layer`'s `kv_cache.{layer}.*` `Op::Input` leaves as declared by
    /// the bound program, which is authoritative; this only fires if a
    /// lowering compiled a program
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

    /// The prompt cache stopped a prefill at `position` to snapshot the ring
    /// layers, and the state could not be brought back to exactly
    /// `position` tokens after the stretch's sampled token was forwarded.
    #[error("prompt cache prefill stop at {position} could not rewind: {reason}")]
    PromptCacheStopRewind {
        position: usize,
        reason: &'static str,
    },

    /// The prompt cache could not write a shifted chunk of cached rows at
    /// `position`, the first token of the chunk in the new prompt.
    #[error("prompt cache chunk shift at {position} refused: {reason}")]
    PromptCacheShift {
        position: usize,
        reason: &'static str,
    },

    /// The bound program bounds layer `layer`'s `kv_cache.{layer}.*` leaves
    /// by the sliding-ring slot (`proxima_tensor::spec::SLIDING_KV_SYMBOL`),
    /// but the checkpoint's own `kv_layers` gives that layer no window to size
    /// the ring by -- the program and the metadata disagree about whether the
    /// layer slides.
    #[error(
        "layer {layer} is bound as a sliding ring but its checkpoint metadata gives it no window"
    )]
    SlidingRingWindowMissing { layer: usize },

    /// `crate::generate::LoadedModel::load_inner`'s load-time residency
    /// gate: `mapped_bytes` (the checkpoint mapping's own length, registered
    /// whole as one no-copy `MTLBuffer`) exceeds `available_bytes` (the
    /// host's own reported limit minus OS headroom, the same
    /// `crate::memory_fit::HostMemoryLimit` shape
    /// [`Self::MemoryBudgetExceeded`] already reads). Unlike that error,
    /// there is no reduced value to retry -- a whole-mapping no-copy buffer
    /// is either entirely resident or it is not, so the bounded alternative
    /// is a caller-configured `ServingConfig::moe_residency_budget_bytes`
    /// pre-gather instead of the whole-mapping registration this gate
    /// refused. Set `PROXIMA_MAPPING_FIT_OVERRIDE=1` to skip this gate for a
    /// measurement run; never for a served request.
    // same dual gate as `Self::MemoryBudgetExceeded` (`crate::memory_fit`'s
    // own doc): the module is forced into a bare `cargo test` build
    // regardless of features, so a variant it constructs must stay
    // reachable there too.
    #[cfg(any(test, all(feature = "std", feature = "metal")))]
    #[error(
        "checkpoint mapping of {mapped_bytes} bytes exceeds the {available_bytes}-byte resident \
         budget; pre-gather into moe_residency_budget_bytes instead of a whole-mapping \
         no-copy buffer, or set PROXIMA_MAPPING_FIT_OVERRIDE=1 for a measurement run"
    )]
    MappingExceedsResidentBudget {
        mapped_bytes: u64,
        available_bytes: u64,
    },

    /// `crate::memory_fit::fit_per_class_budgets`'s load-time gate: one of
    /// `ServingConfig`'s four per-class caps (`dense_weights_budget_bytes`,
    /// `expert_weights_budget_bytes`, `activations_budget_bytes`,
    /// `kv_cache_budget_bytes` -- ROW 501/I2's "separate budgets and
    /// placement owners for expert weights, dense layers, activations, and
    /// KV; they must not collapse into one cache counter") is nonzero and
    /// `class`'s own computed bytes exceed it, independent of
    /// [`Self::MemoryBudgetExceeded`]'s aggregate-limit check -- a class can
    /// fit the whole-host limit and still blow its own configured cap.
    #[cfg(any(test, all(feature = "std", feature = "metal")))]
    #[error(
        "load-time per-class residency budget exceeded: class={class} bytes={bytes} \
         budget_bytes={budget_bytes}"
    )]
    PerClassResidencyBudgetExceeded {
        class: &'static str,
        bytes: u64,
        budget_bytes: u64,
    },

    /// `crate::mapping_residency::prove_resident` touched every page of the
    /// checkpoint mapping ([`crate::loader::prefault`]) and re-checked with
    /// `mincore(2)` once, and `bytes_missing` of `bytes_total` mapped bytes
    /// were still not resident -- the exact silent-zero-read hazard ROW 533
    /// named (`proxima-tensor/docs/discipline.md`): a no-copy GPU mapping
    /// reads a non-resident page as zero rather than faulting, so this
    /// fails the load instead of letting `omega::backend::register_checkpoint_mapping`
    /// hand a partially-resident mapping to the first dispatch.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error(
        "checkpoint mapping still has {bytes_missing} of {bytes_total} bytes non-resident after prefault"
    )]
    MappingNotResident {
        bytes_missing: u64,
        bytes_total: u64,
    },

    /// `PROXIMA_DISPATCH` (`generate::decode::resolve_dispatch_type_override`'s
    /// own doc) was set to a value other than `serial`/`concurrent`
    /// (case-insensitive) or was not valid Unicode. An explicit error here
    /// rather than a silent fallback to `ServingConfig::dispatch_type` --
    /// a typo in the override must not quietly run the configured default.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[error("PROXIMA_DISPATCH={value:?}: expected `serial` or `concurrent`")]
    InvalidDispatchTypeOverride { value: String },
}
