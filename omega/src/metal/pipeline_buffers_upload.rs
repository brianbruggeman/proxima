#[cfg(test)]
use super::warm_resident_buffers::release_all_mapping_residency;
use super::warm_resident_buffers::{release_mapping_residency, request_mapping_residency};
use super::*;
use proxima_numeric::float_layout::FloatLayout;
use std::sync::atomic::AtomicU64;

#[cfg(feature = "instrument")]
use sha2::{Digest, Sha256};

/// [`MTLCompileOptions::mathMode`], narrowed to the three values this
/// crate's kernels compile with. ROW 296 (`proxima-tensor/docs/discipline.md`)
/// measured `Fast` identical to `Relaxed` on that one kernel
/// (240.9-247.3 GB/s vs. 179.2 GB/s for `Safe`, same 0-1.9e-6 parity drift)
/// and dropped it as a knob nothing selected; ROW 338's best cell (nsg=4 +
/// fast) reopened the question on a different kernel, so `Fast` is back as a
/// selectable runtime value rather than a recompile.
///
/// A [`Plan`] carries one of these ([`Plan::set_math_mode`]); it feeds
/// `compile_pipeline` and folds into `pipeline_for`'s cache key so a
/// `Safe`-compiled kernel is never handed to a caller that asked for
/// `Relaxed` or `Fast`, or the reverse.
///
/// `objc2_metal::MTLMathMode`'s own doc (`MTLLibrary.rs`, upstreaming
/// [Apple's `MTLMathMode`](https://developer.apple.com/documentation/metal/mtlmathmode))
/// states the three rungs in these exact words: `Safe` "disables unsafe
/// floating-point optimizations"; `Relaxed` "allows aggressive, unsafe
/// floating-point optimizations but preserves infs and nans"; `Fast`
/// "allows aggressive, unsafe floating-point optimizations" with no such
/// preservation. `Relaxed`'s wording -- aggressive optimization, NaN/inf
/// preserved -- is exactly [`NumericPolicy::llama_relaxed`]: contraction and
/// reassociation are permitted, `nan_assumptions`/`signed_zero`/
/// `approx_functions` are withheld because Relaxed's own contract
/// explicitly preserves NaN/inf/zero behavior. See
/// `numeric_policy_as_metal_math_mode` and
/// `metal_math_mode_as_numeric_policy` for the full projection both
/// directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MathMode {
    /// IEEE-safe float math -- bit-parity with
    /// [`proxima_tensor::cpu::evaluate`], at 179.2 GB/s on the ROW 296
    /// packed-row Q4_K matvec kernel.
    Safe,
    /// Metal's relaxed-math kernels. ROW 296/297's measured default:
    /// 240.9-247.3 GB/s on the ROW 296 kernel (1.34x `Safe`), 28.19 vs.
    /// 33.82 ms/token steady-state on the whole ROW 297 decode program
    /// (1.20x), with generated text and quality metrics identical to
    /// `Safe`'s in both rows.
    #[default]
    Relaxed,
    /// Metal's fast-math kernels. Measured indistinguishable from
    /// `Relaxed` on the ROW 296 kernel; ROW 338's best cell (nsg=4 + fast)
    /// found a different kernel where it was not, so it is exposed here as
    /// a runtime choice rather than assumed identical everywhere.
    Fast,
}

impl MathMode {
    /// selects the widest compiler mode permitted by the bound numeric policy.
    #[must_use]
    pub const fn for_numeric_policy(policy: NumericPolicy) -> Self {
        numeric_policy_as_metal_math_mode(policy)
    }

    pub(super) const fn as_mtl(self) -> MTLMathMode {
        match self {
            MathMode::Safe => MTLMathMode::Safe,
            MathMode::Relaxed => MTLMathMode::Relaxed,
            MathMode::Fast => MTLMathMode::Fast,
        }
    }

    /// One character folded onto [`kernel_cache_key`]'s own identity string
    /// by every `PIPELINE_CACHE` call site ([`resolve_steps`],
    /// [`encode_op`], [`Plan::kernel_keys`]) -- NOT a duplicate of
    /// `identity::MetalOnlyExtras::numeric_policy_token`. That token is
    /// strictly finer for the axes `numeric_policy` fixes for a `Plan`'s
    /// whole life (chunking, contraction, reassociation); it does NOT track
    /// [`Plan::set_math_mode`], which narrows `compile_pipeline`'s
    /// `MTLCompileOptions.mathMode` for an UNCHANGED `numeric_policy`. Two
    /// resolutions of the same `Plan` that only differ by a `set_math_mode`
    /// call render byte-identical MSL source (`compile_pipeline` never
    /// touches source text) but must never share a `PIPELINE_CACHE` entry,
    /// since they were compiled with different `mathMode` compile options --
    /// this token is what keeps them apart.
    pub(super) const fn cache_token(self) -> char {
        match self {
            MathMode::Safe => 'S',
            MathMode::Relaxed => 'R',
            MathMode::Fast => 'F',
        }
    }
}

/// `MTLCompileOptions.mathMode` only distinguishes 3 rungs -- a compiler
/// flag governing how the SAME algebra compiles. [`NumericPolicy`] governs a
/// richer, orthogonal question: which algebra `bind`/the emitter are
/// permitted to choose in the first place (chunk count, contraction,
/// reduction order). [`MathMode`] is this narrower projection, not a
/// duplicate ladder. This selects the WIDEST mode whose own permission set
/// ([`metal_math_mode_as_numeric_policy`]) is a SUBSET of `policy` --
/// [`NumericPolicy::grants`] is the subset check, the same one
/// [`Plan::set_math_mode`] already uses to refuse a widening request. It
/// never rounds up past what `policy` actually grants: a `contraction`-only
/// policy does not grant `llama_relaxed()` (which also needs
/// `reassociation`), so it stays `Safe`; only a policy granting BOTH
/// `contraction` and `reassociation` gets `Relaxed`, and only one granting
/// every permission [`NumericPolicy::fast`] names gets `Fast`. See
/// [`metal_math_mode_as_numeric_policy`] for the inverse.
///
/// A free function, not an inherent `impl NumericPolicy` -- `NumericPolicy`
/// is defined in `proxima-tensor`, and the orphan rule forbids an inherent
/// `impl` for a foreign type from this crate.
#[must_use]
pub(super) const fn numeric_policy_as_metal_math_mode(policy: NumericPolicy) -> MathMode {
    if policy.grants(NumericPolicy::fast()) {
        MathMode::Fast
    } else if policy.grants(NumericPolicy::llama_relaxed()) {
        MathMode::Relaxed
    } else {
        MathMode::Safe
    }
}

/// The inverse of [`numeric_policy_as_metal_math_mode`]: the exact
/// permission set Apple's own `MTLMathMode` doc commits to for `mode` --
/// `Safe` grants nothing ([`NumericPolicy::bit_exact`]), `Relaxed` grants
/// contraction+reassociation only, NaN/inf/zero preserved per the header
/// ([`NumericPolicy::llama_relaxed`]), `Fast` grants everything
/// ([`NumericPolicy::fast`]). Loses nothing: it round-trips through
/// [`numeric_policy_as_metal_math_mode`] for all three modes. What the
/// OTHER direction loses: every point in the 32-state permission space that
/// isn't one of these 3 Metal natively supports (e.g. `contraction` alone
/// without `reassociation` compiles identically to both granted, since
/// Metal has no finer compiler flag).
#[must_use]
pub(super) const fn metal_math_mode_as_numeric_policy(mode: MathMode) -> NumericPolicy {
    match mode {
        MathMode::Safe => NumericPolicy::bit_exact(),
        MathMode::Relaxed => NumericPolicy::llama_relaxed(),
        MathMode::Fast => NumericPolicy::fast(),
    }
}

/// [`MTLComputeCommandEncoder`]'s dispatch-scheduling mode, narrowed to the
/// two values [`objc2_metal::MTLDispatchType`] exposes to a compute encoder.
/// A [`Plan`] carries one of these ([`Plan::set_dispatch_type`]); it decides
/// which encoder [`execute_plan_with_placements`] opens and whether its loop
/// runs `HazardTracker` at all.
///
/// ROW 311 (`proxima-tensor/docs/discipline.md`): llama.cpp encodes its whole
/// token on a serial compute encoder with zero explicit barriers.
/// `Concurrent` (this type's default) lets independent dispatches overlap
/// instead of draining the pipeline between every op, at the cost of the 323
/// per-token [`MTLBarrierScope::Buffers`] barriers `HazardTracker` inserts
/// to keep that overlap correct; `Serial` orders every dispatch for free and
/// emits none. See ROW 312 for the measured wall-clock comparison between
/// the two on the decode program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DispatchType {
    /// One dispatch completes before the next begins -- llama.cpp's own
    /// encoding shape, and the same guarantee an unmodified
    /// `computeCommandEncoder()` gives. No barrier is ever needed.
    Serial,
    /// Independent dispatches may overlap; `HazardTracker` inserts a
    /// [`MTLBarrierScope::Buffers`] barrier wherever a RAW/WAW/WAR hazard
    /// would otherwise let two overlapping dispatches race.
    #[default]
    Concurrent,
}

impl DispatchType {
    pub(super) const fn as_mtl(self) -> MTLDispatchType {
        match self {
            DispatchType::Serial => MTLDispatchType::Serial,
            DispatchType::Concurrent => MTLDispatchType::Concurrent,
        }
    }
}

/// [`pipeline_for`]/[`pipeline_for_kernel`]'s own creation record for a
/// compiled pipeline -- the identity a dispatch-site capture must reuse on
/// a cache HIT instead of recomputing `msl_sha256` from a different string.
/// See [`super::device_buffers_arena_plan::PIPELINE_CAPTURE`]'s own doc for
/// why this is keyed by pointer, not by [`PIPELINE_CACHE`]'s string key.
#[cfg(feature = "instrument")]
#[derive(Debug, Clone)]
pub(super) struct PipelineCaptureRecord {
    pub(super) cache_key: String,
    pub(super) entry: String,
    pub(super) msl_sha256: String,
}

/// Lowercase hex sha256 of `bytes` -- shared by [`capture_pipeline_source`]
/// (hashes a freshly-compiled kernel's source) and
/// [`audit_pipeline_key_on_hit`] (hashes a re-emitted one), so the two never
/// drift onto different digest formats.
#[cfg(feature = "instrument")]
fn sha256_hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return EMPTY_SOURCE_SKIPPED.to_string();
    }
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// stands in for the digest when there is no source to hash, so the event never carries the
/// empty-input sha256 (e3b0c442...) as if it named a real kernel.
#[cfg(feature = "instrument")]
const EMPTY_SOURCE_SKIPPED: &str = "<skipped: empty source>";

#[cfg(all(test, feature = "instrument"))]
mod sha256_hex_tests {
    use super::{EMPTY_SOURCE_SKIPPED, sha256_hex};

    const EMPTY_INPUT_SHA256: &str =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

    #[test]
    fn non_empty_msl_source_hashes_to_a_real_digest() {
        let source =
            "#include <metal_stdlib>\nkernel void add_f32(device float* out [[buffer(0)]]) {}\n";
        let digest = sha256_hex(source.as_bytes());
        assert_eq!(digest.len(), 64, "sha256 hex is 64 chars");
        assert_ne!(
            digest, EMPTY_INPUT_SHA256,
            "non-empty source must not hash to the empty digest"
        );
    }

    #[test]
    fn empty_msl_source_is_skipped_not_hashed() {
        let digest = sha256_hex(b"");
        assert_eq!(digest, EMPTY_SOURCE_SKIPPED);
        assert_ne!(
            digest, EMPTY_INPUT_SHA256,
            "empty input must never log the empty digest"
        );
    }
}

/// `PROXIMA_PIPELINE_CAPTURE=<dir>` writes the exact MSL source text this
/// process submitted to `newLibraryWithSource_options_error` to
/// `<dir>/pipeline_<sha256-16hex>.metal`, prints the `pipeline_create`
/// record, and returns the full sha256 hex digest so the caller can register
/// a [`PipelineCaptureRecord`] against the compiled pipeline's pointer. A
/// no-op (returns `None`) when the env var is unset, so a plain `instrument`
/// build with no capture directory pays one `env::var_os` lookup per miss.
#[cfg(feature = "instrument")]
fn capture_pipeline_source(kernel: &Kernel, cache_key: &str, math_mode: MathMode) -> String {
    let full_hex = sha256_hex(kernel.source.as_bytes());
    if let Some(dir) = std::env::var_os("PROXIMA_PIPELINE_CAPTURE") {
        let path = std::path::Path::new(&dir).join(format!("pipeline_{}.metal", &full_hex[..16]));
        if let Err(error) = std::fs::write(&path, &kernel.source) {
            debug!(path = ?path, ?error, "pipeline_capture_write_failed");
        }
    }
    debug!(
        cache_key = ?cache_key,
        entry = %kernel.entry,
        msl_sha256 = %full_hex,
        math_mode = ?math_mode,
        len = kernel.source.len() as u64,
        "pipeline_create"
    );
    full_hex
}

#[cfg(feature = "instrument")]
fn register_pipeline_capture(
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    cache_key: &str,
    entry: &str,
    msl_sha256: String,
) {
    let pointer = Retained::as_ptr(pipeline) as usize;
    super::device_buffers_arena_plan::PIPELINE_CAPTURE.with(|capture| {
        capture.borrow_mut().insert(
            pointer,
            PipelineCaptureRecord {
                cache_key: cache_key.to_string(),
                entry: entry.to_string(),
                msl_sha256,
            },
        );
    });
}

/// `PROXIMA_PIPELINE_KEY_AUDIT=1`: `pipeline_for` re-emits `bound`'s own
/// kernel on every cache hit and re-hashes it, catching a key-completeness
/// gap (an axis two structurally-different ops disagree on, but
/// `kernel_identity` never folded in) as a loud, typed error at the exact hit
/// that reused the wrong pipeline -- see `c4-7-reduction-literal.md`.
/// Default off; unset, empty, or any value other than `"1"` skips the
/// re-emit entirely, so a plain `instrument` build with the audit off pays
/// one `env::var_os` lookup per hit.
#[cfg(feature = "instrument")]
fn pipeline_key_audit_enabled() -> bool {
    matches!(std::env::var("PROXIMA_PIPELINE_KEY_AUDIT"), Ok(value) if value.trim() == "1")
}

/// Re-emits `bound`'s kernel and compares its sha256 against the
/// [`PipelineCaptureRecord`] this pipeline's pointer was registered under at
/// compile time. `Ok(())` when the audit is off, when no record exists for
/// this pointer (a pipeline built through [`pipeline_for_kernel`]'s own path
/// rather than this function's `emit`-driven one), or when the two hashes
/// agree.
#[cfg(feature = "instrument")]
#[expect(
    clippy::too_many_arguments,
    reason = "the audit re-emits with exactly the values pipeline_for resolved, which arrive as separate values from its caller"
)]
fn audit_pipeline_key_on_hit(
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    cache_key: &str,
    numeric_policy: NumericPolicy,
    knobs: &EmitKnobs,
    dense_mma_input_layout: FloatLayout,
    attention: AttentionKernelChoice,
) -> Result<(), MetalError> {
    if !pipeline_key_audit_enabled() {
        return Ok(());
    }
    let recorded = super::device_buffers_arena_plan::PIPELINE_CAPTURE.with(|capture| {
        capture
            .borrow()
            .get(&(Retained::as_ptr(pipeline) as usize))
            .cloned()
    });
    let Some(recorded) = recorded else {
        return Ok(());
    };
    let kernel = crate::msl::emit_inner_with_mma_selection_and_layout(
        bound,
        packed_operands,
        numeric_policy,
        knobs,
        false,
        attention,
        dense_mma_input_layout,
    )?;
    let recomputed_sha256 = sha256_hex(kernel.source.as_bytes());
    super::device_buffers_arena_plan::PIPELINE_KEY_AUDIT_AUDITED
        .with(|counter| counter.set(counter.get() + 1));
    if recorded.msl_sha256 != recomputed_sha256 {
        super::device_buffers_arena_plan::PIPELINE_KEY_AUDIT_MISMATCHED
            .with(|counter| counter.set(counter.get() + 1));
        return Err(MetalError::PipelineKeyAuditMismatch {
            cache_key: cache_key.to_string(),
            recorded_sha256: recorded.msl_sha256,
            recomputed_sha256,
        });
    }
    Ok(())
}

pub(super) fn compile_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: &Kernel,
    math_mode: MathMode,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    let options = MTLCompileOptions::new();
    // ROW 296 (`proxima-tensor/docs/discipline.md`): `Safe` and `Relaxed`
    // stream 179.2 vs. 240.9-247.3 GB/s with identical parity in every
    // shape-sweep cell, so the mode is a caller choice (`Plan::set_math_mode`),
    // not a fixed compile option.
    options.setMathMode(math_mode.as_mtl());

    let source = NSString::from_str(&kernel.source);
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| MetalError::CompileFailed {
            log: nserror_description(&error),
        })?;

    let entry = NSString::from_str(&kernel.entry);
    let function =
        library
            .newFunctionWithName(&entry)
            .ok_or_else(|| MetalError::CompileFailed {
                log: format!(
                    "kernel entry `{}` missing from its own compiled library",
                    kernel.entry
                ),
            })?;

    pipeline_for_function(device, &function, kernel, math_mode)
}

/// Resolves `bound`'s compiled pipeline against `cache_key`
/// ([`kernel_cache_key`]) rather than [`Kernel::source`] — a hit never builds
/// the MSL source text at all, only a genuine miss calls [`emit`] to render
/// it and compile. See this module's `ROW 92`/`ROW 93` discipline-log entries
/// for the measured cost `emit` paid on every call, hit or miss, before this
/// split existed.
#[expect(
    clippy::too_many_arguments,
    reason = "the cache key, both numeric settings and both attention selections arrive pre-resolved from each caller, which already holds them as separate values"
)]
pub(super) fn pipeline_for(
    device: &ProtocolObject<dyn MTLDevice>,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    cache_key: &str,
    math_mode: MathMode,
    numeric_policy: NumericPolicy,
    knobs: &EmitKnobs,
    dense_mma_input_layout: FloatLayout,
    attention_mma_selection: crate::msl::AttentionMmaSelection,
    attention_row_schedule: crate::msl::AttentionRowSchedule,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    // `cache_key` already carries every selected attention axis and both
    // numeric compilation settings `compile_pipeline` reads: the
    // numeric-policy token from `kernel_cache_key`
    // (`crate::identity::kernel_identity`, via
    // `MetalOnlyExtras::numeric_policy_token`) plus `math_mode`'s own
    // `MathMode::cache_token`, appended by every caller of this function
    // (`resolve_steps`, `encode_op`) before it gets here. Two BoundOps
    // agreeing on everything else but compiled under different policies OR
    // different math modes still never share a pipeline: `Safe`'s kernel
    // body is byte-identical to `Relaxed`'s (`compile_pipeline` never
    // touches source text, only `MTLCompileOptions`), so the key's tokens
    // are what keep them apart. The numeric-policy token alone is not
    // enough: `Plan::set_math_mode` narrows the compiled mode for an
    // UNCHANGED `numeric_policy`, so a `math_mode` token distinct from the
    // policy token is required too (see `MathMode::cache_token`'s own doc).
    let shared = MetalDevice::current()?;
    if let Some(pipeline) = shared.pipeline(cache_key) {
        trace!(cache_key = %cache_key, hit = true, "pipeline cache lookup");
        #[cfg(feature = "instrument")]
        counter!(PIPELINE_HITS, 1);
        #[cfg(feature = "instrument")]
        audit_pipeline_key_on_hit(
            &pipeline,
            bound,
            packed_operands,
            cache_key,
            numeric_policy,
            knobs,
            dense_mma_input_layout,
            AttentionKernelChoice {
                mma_selection: attention_mma_selection,
                row_schedule: attention_row_schedule,
            },
        )?;
        return Ok(pipeline);
    }
    trace!(cache_key = %cache_key, hit = false, "pipeline cache lookup");
    // fires only on a pipeline-cache MISS, before `compile_pipeline` runs --
    // evidence of the lowering choice this cache-key string encodes, never
    // of a later dispatch: a cache hit on the same key emits nothing here.
    #[cfg(feature = "instrument")]
    if let BoundOpKind::CachedAttention {
        query_groups,
        head_dim,
        cached_key_rows,
        new_key_rows,
        ..
    } = &bound.kind
    {
        let context_length = cached_key_rows + new_key_rows;
        debug!(
            context_length,
            context_chunks = crate::msl::context_chunks_for(
                context_length,
                *query_groups,
                *head_dim,
                numeric_policy
            ),
            numeric_policy = ?numeric_policy,
            kernel_identity = %cache_key,
            "lowering selected attention chunking"
        );
    }
    #[cfg(feature = "instrument")]
    let compile_started = read_ticks();
    let kernel = crate::msl::emit_inner_with_mma_selection_and_layout(
        bound,
        packed_operands,
        numeric_policy,
        knobs,
        false,
        AttentionKernelChoice {
            mma_selection: attention_mma_selection,
            row_schedule: attention_row_schedule,
        },
        dense_mma_input_layout,
    )?;
    trace!(
        node = ?bound.node,
        source = %kernel.source,
        "metal bound source emitted for a pipeline cache miss"
    );
    #[cfg(feature = "instrument")]
    let captured_sha256 = capture_pipeline_source(&kernel, cache_key, math_mode);
    // Owner addition to Intervention 5 (RUN.md "Intervention 5"): one line
    // per DISTINCT packed-row pipeline at creation, so the selected
    // rows-per-simdgroup variant and its launch geometry are provable from
    // the log rather than inferred -- `PIPELINE_CACHE`'s own miss gate above
    // already means this only fires once per distinct `cache_key`, never per
    // dispatch. Checked once here (a genuine cache miss, not a hot path),
    // not per dispatch, per the owner's own zero-per-dispatch-work
    // constraint. Gated on `instrument`, exactly like `token_breakdown_gpu`
    // (`placements_execute_named.rs`) -- the env-var check nests inside the
    // feature gate rather than standing alone, so this diagnostic never
    // compiles into a non-instrument build.
    #[cfg(feature = "instrument")]
    {
        let quantized = crate::identity::operand_codecs(bound, packed_operands);
        if let Some(block) = crate::msl::packed_row_block(bound, &quantized, knobs) {
            let geometry = crate::msl::packed_row_geometry(bound, &quantized, &block, knobs);
            let rows = geometry.rows_per_simdgroup;
            if let Ok((_, grid)) =
                crate::msl::kernel_dispatch_shape(bound, packed_operands, numeric_policy, knobs)
            {
                let feature_total: u64 = block
                    .feature_axes
                    .iter()
                    .map(|&axis| bound.extents[axis as usize])
                    .product();
                let k_extent = bound.extents[block.reduce_dim];
                debug!(
                    entry = %kernel.entry,
                    rows_per_simdgroup = rows as u64,
                    grid_threads = grid.threads,
                    threadgroup = ?grid.threadgroup_width,
                    rows = feature_total,
                    k_extent,
                    "packed_rows_pipeline"
                );
            }
        }
    }
    let pipeline = compile_pipeline(device, &kernel, math_mode)?;
    #[cfg(feature = "instrument")]
    {
        counter!(PIPELINE_MISSES, 1);
        counter!(PIPELINE_COMPILE_TICKS, elapsed_ticks(compile_started));
        register_pipeline_capture(&pipeline, cache_key, &kernel.entry, captured_sha256);
        debug!(
            entry = %kernel.entry,
            cache_key = %cache_key,
            static_threadgroup_memory = pipeline.staticThreadgroupMemoryLength() as u64,
            max_total_threads = pipeline.maxTotalThreadsPerThreadgroup() as u64,
            thread_execution_width = pipeline.threadExecutionWidth() as u64,
            device_max_threadgroup_memory = device.maxThreadgroupMemoryLength() as u64,
            "pipeline_footprint"
        );
    }
    Ok(shared.insert_pipeline(cache_key, pipeline))
}

/// [`pipeline_for`]'s counterpart for a [`Kernel`] already in hand (the
/// `CachedAttention` merge dispatch's own `emit_cached_attention_merge`
/// output) instead of one this function must `emit` itself from a `bound` --
/// its own pipeline-cache entry, keyed on `cache_key` (the split kernel's
/// own key plus a `_merge` suffix at the call site), so a merge kernel never
/// shares a compiled `MTLComputePipelineState` with its split sibling even
/// though both come from the SAME `BoundOp` position.
pub(super) fn pipeline_for_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    kernel: &Kernel,
    cache_key: &str,
    math_mode: MathMode,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    let shared = MetalDevice::current()?;
    if let Some(pipeline) = shared.pipeline(cache_key) {
        return Ok(pipeline);
    }
    #[cfg(feature = "instrument")]
    let captured_sha256 = capture_pipeline_source(kernel, cache_key, math_mode);
    let pipeline = compile_pipeline(device, kernel, math_mode)?;
    #[cfg(feature = "instrument")]
    register_pipeline_capture(&pipeline, cache_key, &kernel.entry, captured_sha256);
    Ok(shared.insert_pipeline(cache_key, pipeline))
}

pub(super) fn allocation_capacity(element_count: usize, dtype: DType) -> Result<usize, MetalError> {
    let bytes = element_count
        .max(1)
        .checked_mul(device_element_bytes(dtype))
        .ok_or(MetalError::ResourceOverflow {
            field: "buffer_capacity",
        })?;
    #[cfg(feature = "metal-buffer-pool")]
    let bytes = pool_bucket_checked(bytes)?;
    Ok(bytes)
}

#[cfg(feature = "metal-buffer-pool")]
fn pool_bucket_checked(byte_length: usize) -> Result<usize, MetalError> {
    byte_length
        .checked_next_power_of_two()
        .ok_or(MetalError::ResourceOverflow {
            field: "buffer_capacity",
        })
}

fn check_device_buffer_length(
    device: &ProtocolObject<dyn MTLDevice>,
    requested: usize,
) -> Result<(), MetalError> {
    let limit = device.maxBufferLength();
    if requested > limit {
        return Err(MetalError::ResourceLimit {
            field: "max_buffer_length",
            requested: requested as u64,
            limit: limit as u64,
        });
    }
    Ok(())
}

pub(super) fn allocate_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    element_count: usize,
    dtype: DType,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    allocate_buffer_for("output_buffer", "call", device, element_count, dtype)
}

#[cfg(not(feature = "metal-buffer-pool"))]
pub(super) fn allocate_buffer_for(
    owner: &'static str,
    lifetime: &'static str,
    device: &ProtocolObject<dyn MTLDevice>,
    element_count: usize,
    dtype: DType,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_length = allocation_capacity(element_count, dtype)?;
    check_device_buffer_length(device, byte_length)?;
    counter!(OUTPUT_BUFFER_ALLOCATIONS, 1);
    counter!(OUTPUT_BUFFER_ALLOCATED_BYTES, byte_length as u64);
    let buffer = device
        .newBufferWithLength_options(byte_length, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a shared buffer".to_string(),
        })?;
    #[cfg(feature = "instrument")]
    record_device_buffer(owner, lifetime, "", byte_length, false);
    #[cfg(not(feature = "instrument"))]
    let _ = (owner, lifetime);
    Ok(buffer)
}

/// Pool-backed counterpart of the `metal-buffer-pool`-off `allocate_buffer`
/// above -- identical contract (a buffer sized to hold `element_count`
/// `dtype` elements), but the ACTUAL Metal allocation, on both the pool-hit
/// and pool-miss paths, is always sized to `pool_bucket(byte_length)`, never
/// to the tight `byte_length`. See `pool_bucket`'s doc for why that is what
/// makes a bucketed pop sound, and `OUTPUT_BUFFER_POOL`'s doc for the
/// invariant this maintains across the pool's whole lifetime.
#[cfg(feature = "metal-buffer-pool")]
pub(super) fn allocate_buffer_for(
    owner: &'static str,
    lifetime: &'static str,
    device: &ProtocolObject<dyn MTLDevice>,
    element_count: usize,
    dtype: DType,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let bucket = allocation_capacity(element_count, dtype)?;
    check_device_buffer_length(device, bucket)?;
    let pooled = OUTPUT_BUFFER_POOL.with(|pool| {
        pool.borrow_mut()
            .get_mut(&(bucket, dtype))
            .and_then(Vec::pop)
    });
    if let Some(buffer) = pooled {
        counter!(OUTPUT_BUFFER_POOL_REUSES, 1);
        return Ok(buffer);
    }
    counter!(OUTPUT_BUFFER_ALLOCATIONS, 1);
    counter!(OUTPUT_BUFFER_ALLOCATED_BYTES, bucket as u64);
    let buffer = device
        .newBufferWithLength_options(bucket, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a shared buffer".to_string(),
        })?;
    #[cfg(feature = "instrument")]
    record_device_buffer(owner, lifetime, "", bucket, false);
    #[cfg(not(feature = "instrument"))]
    let _ = (owner, lifetime);
    Ok(buffer)
}

/// Rounds `byte_length` up to the next power of two -- the pool's bucket
/// function. A pool entry's KEY is always this bucket, and by
/// [`allocate_buffer`]'s own invariant (every fresh allocation under this
/// feature is sized to exactly `pool_bucket(request)` bytes, never to the
/// tight request), a buffer stored under bucket `B` always has REAL Metal
/// capacity `>= B` -- in fact exactly `B`, since nothing ever shrinks a
/// buffer once allocated. So a pop from bucket `B` is sound for any request
/// whose own bucket is `B`, i.e. any request in `(B/2, B]` bytes: never
/// smaller than what was asked for.
#[cfg(feature = "metal-buffer-pool")]
pub(super) fn pool_bucket(byte_length: usize) -> usize {
    byte_length.next_power_of_two()
}

#[cfg(feature = "metal-buffer-pool")]
thread_local! {
    /// Op-OUTPUT device buffers, reused across [`execute_plan`] calls instead
    /// of a fresh `newBufferWithLength_options` per op per token
    /// (`allocate_buffer` pops from here first). Keyed on `(bucket, dtype)`
    /// where `bucket` is [`pool_bucket`]'s power-of-two rounding of the
    /// requested byte length -- NOT the tight byte length itself. See
    /// `pool_bucket`'s doc for the size invariant this relies on: every
    /// buffer stored under bucket `B` has REAL Metal capacity exactly `B`,
    /// because [`allocate_buffer`] allocates fresh buffers at `B` too, never
    /// at the tight request. `dtype` stays part of the key alongside the
    /// bucket so the guarantee reads directly off the type at every call
    /// site.
    ///
    /// # Why bucketing (not exact-size keying)
    ///
    /// Decode's cached-attention extent grows every token, so an op whose
    /// output size tracks it (`BoundOp::output_len` scales with the KV extent)
    /// would get a NEW exact key every token under exact-size keying, which
    /// (a) never gets reused again -- an unbounded, one-orphaned-buffer-per-
    /// token leak over a long decode session -- and (b) MISSES every time on
    /// exactly those ops, so they keep paying `newBufferWithLength` per token
    /// regardless. Bucketing by power of two means a growing extent walks
    /// through a BOUNDED number of buckets (`log2(max_extent_bytes)`, not one
    /// per token) and hits the bucket it shared with the previous token's
    /// request whenever the two requests round to the same power of two.
    ///
    /// # Liveness: why this cannot alias two live tensors
    ///
    /// A buffer is pushed back here ONLY by [`execute_plan`], and only after
    /// that call's single `command_buffer.waitUntilCompleted()` has returned
    /// -- i.e. after every dispatch that could read or write it has finished
    /// running on the GPU. Nothing is ever returned mid-program:
    /// `prepared.retires` still drives `device_buffers` removal exactly as
    /// before this feature existed (see the `metal-buffer-pool`-off arm right
    /// below the retirement loop in `execute_plan`), the retired buffer is
    /// simply ALSO cloned into a same-call `reclaim_stash` rather than only
    /// dropped, and that stash is not drained into this pool until after the
    /// wait. So within one `execute_plan` call, every buffer this pool hands
    /// out via `allocate_buffer` was either freshly allocated or was a buffer
    /// whose prior GPU work is already complete -- it can never be a buffer
    /// some still-pending dispatch in the SAME command buffer is about to
    /// read or write, because nothing enters this pool until that command
    /// buffer no longer exists to have pending dispatches. This is the "pool
    /// only across calls" fallback, chosen over intra-program reuse (which
    /// would need to reason about `MTLDispatchTypeSerial` hazard-tracked
    /// ordering across a retired buffer's last read and a later op's first
    /// write) to keep the liveness argument this simple.
    ///
    /// # Never touches block-input caching
    ///
    /// Only buffers `encode_op` allocated for an op's OUTPUT are ever pushed
    /// here -- `execute_plan` filters by `output_meta`, built from
    /// `prepared.resolved`'s own node ids, before cloning anything into
    /// `reclaim_stash`. `NOCOPY_BUFFERS` and the resident-copy cache (block
    /// INPUTS, wrapping the caller's own weight memory) are untouched by this
    /// feature; a block-input buffer removed by the same retirement loop is
    /// dropped exactly as it always was, never reaching this map.
    ///
    /// # An over-sized buffer is only ever consulted through the bound op's
    /// own shape, never through its own `.length()`
    ///
    /// The one hazard bucketing introduces: a popped buffer can be LARGER
    /// than the op that requested it strictly needs. Every dispatch-affecting
    /// consumer -- `dispatch`, `bind_buffers`, `read_back` -- sizes its work
    /// from the bound op's own shape/uniforms, never from buffer capacity, so
    /// bucketing carries no dispatch/readback correctness hazard.
    ///
    /// # Bounded, not unbounded: worst-case memory overhead
    ///
    /// A power-of-two bucket wastes strictly less than 2x the tight request
    /// (a request just over `B/2` rounds up to `B`, the worst case; a request
    /// of exactly a power of two wastes nothing). DERIVED, not measured.
    /// [`crate::sized::OUTPUT_POOL_MAX_PER_BUCKET`] caps retained-buffer growth per bucket
    /// on top of that.
    pub(super) static OUTPUT_BUFFER_POOL: RefCell<HashMap<(usize, DType), Vec<MetalBuffer>>> =
        RefCell::new(HashMap::new());
}

/// Counts op-output buffers served from `OUTPUT_BUFFER_POOL` rather than
/// freshly allocated.
#[cfg(feature = "metal-buffer-pool")]
pub static OUTPUT_BUFFER_POOL_REUSES: Counter = Counter::new("omega.metal.output_pool.reuse");

/// Total buffers currently retained across every `(bucket, dtype)` slot in
/// `OUTPUT_BUFFER_POOL` on THIS thread. Test/diagnostic surface: proves
/// bucketing keeps the pool's retained-buffer count BOUNDED across many
/// distinct historical output sizes (a growing cached-attention extent)
/// rather than growing once per distinct size an exact-size-keyed pool
/// would.
#[cfg(feature = "metal-buffer-pool")]
#[must_use]
pub fn output_buffer_pool_len() -> usize {
    OUTPUT_BUFFER_POOL.with(|pool| pool.borrow().values().map(Vec::len).sum())
}

/// split-4019 per-token attribution counters — each is a (`_CALLS`, `_TICKS`)
/// pair over one named stage of [`execute_plan`], read back via `.get()`
/// (cumulative) or `.snapshot_and_reset()` (per-token delta) by a caller that
/// wants to attribute wall clock rather than assume it. Every wrap site notes
/// which term of the split-4019 table it feeds.
#[cfg(feature = "instrument")]
pub static PREPARE_CALLS: Counter = Counter::new("omega.metal.prepare_calls");
#[cfg(feature = "instrument")]
pub static PREPARE_TICKS: Counter = Counter::new("omega.metal.prepare_ticks");
#[cfg(feature = "instrument")]
pub static EMIT_CALLS: Counter = Counter::new("omega.metal.emit_calls");
#[cfg(feature = "instrument")]
pub static EMIT_TICKS: Counter = Counter::new("omega.metal.emit_ticks");
#[cfg(feature = "instrument")]
pub static PIPELINE_HITS: Counter = Counter::new("omega.metal.pipeline_hits");
#[cfg(feature = "instrument")]
pub static PIPELINE_MISSES: Counter = Counter::new("omega.metal.pipeline_misses");
#[cfg(feature = "instrument")]
pub static PIPELINE_COMPILE_TICKS: Counter = Counter::new("omega.metal.pipeline_compile_ticks");
#[cfg(feature = "instrument")]
pub static BLOCK_UPLOAD_CALLS: Counter = Counter::new("omega.metal.block_upload_calls");
/// Counts a block-upload-loop position whose [`block_identity_key`] this step
/// disagrees with what [`Plan::block_identity`] recorded for it last step --
/// a REBIND, not a first-ever upload (`block_identity[index]` was already
/// `Some`). ROW 551: the direct witness that [`block_buffer_reusable`]'s fast
/// skip is missing every step for a resident weight block, forcing
/// [`checkpoint_mapping_offset`] to re-bind it from the mmap every token
/// instead of reusing the prior step's `device_buffers` entry.
#[cfg(feature = "instrument")]
pub static MAPPING_REBOUND_BLOCKS: Counter = Counter::new("omega.metal.mapping_rebound_blocks");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_STAGE_CALLS: Counter =
    Counter::new("omega.metal.expert_source_stage_calls");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_STAGE_BYTES: Counter =
    Counter::new("omega.metal.expert_source_stage_bytes");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_STAGE_TICKS: Counter =
    Counter::new("omega.metal.expert_source_stage_ticks");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_CACHE_HITS: Counter = Counter::new("omega.metal.expert_source_cache_hits");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_CACHE_MISSES: Counter =
    Counter::new("omega.metal.expert_source_cache_misses");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_CACHE_COLD_MISSES: Counter =
    Counter::new("omega.metal.expert_source_cache_cold_misses");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_CACHE_REPLACEMENT_MISSES: Counter =
    Counter::new("omega.metal.expert_source_cache_replacement_misses");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_BUFFER_REUSES: Counter =
    Counter::new("omega.metal.expert_source_buffer_reuses");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_REUSE_COPY_BYTES: Counter =
    Counter::new("omega.metal.expert_source_reuse_copy_bytes");
#[cfg(feature = "instrument")]
pub static EXPERT_SOURCE_REUSE_COPY_TICKS: Counter =
    Counter::new("omega.metal.expert_source_reuse_copy_ticks");
#[cfg(feature = "instrument")]
pub static BLOCK_UPLOAD_TICKS: Counter = Counter::new("omega.metal.block_upload_ticks");
#[cfg(feature = "instrument")]
pub static BLOCK_OFFERED_BYTES: Counter = Counter::new("omega.metal.block_offered_bytes");
#[cfg(feature = "instrument")]
pub static OP_SETUP_CALLS: Counter = Counter::new("omega.metal.op_setup_calls");
#[cfg(feature = "instrument")]
pub static OP_SETUP_TICKS: Counter = Counter::new("omega.metal.op_setup_ticks");
/// ROW 93's split of ROW 92's "inside-backend residual": the whole
/// `pipeline_for` call (cache lookup on a hit, lookup+compile on a miss),
/// distinct from `EMIT_TICKS` (now the cheap `kernel_cache_key`/
/// `kernel_dispatch_shape` pair, no MSL text) and from
/// `PIPELINE_COMPILE_TICKS` (the compile-only sub-span that fires on a miss).
#[cfg(feature = "instrument")]
pub static PIPELINE_LOOKUP_CALLS: Counter = Counter::new("omega.metal.pipeline_lookup_calls");
#[cfg(feature = "instrument")]
pub static PIPELINE_LOOKUP_TICKS: Counter = Counter::new("omega.metal.pipeline_lookup_ticks");
/// ROW 93's split of ROW 92's residual, part two: `setComputePipelineState`
/// + `bind_buffers` + `dispatch`, `encode_op`'s three remaining calls.
#[cfg(feature = "instrument")]
pub static ENCODE_DISPATCH_CALLS: Counter = Counter::new("omega.metal.encode_dispatch_calls");
#[cfg(feature = "instrument")]
pub static ENCODE_DISPATCH_TICKS: Counter = Counter::new("omega.metal.encode_dispatch_ticks");
/// The sole physical `dispatchThreads_threadsPerThreadgroup` call site --
/// `resident_nocopy_cache::dispatch` -- fires once per PHYSICAL GPU launch,
/// while [`ENCODE_DISPATCH_CALLS`] counts `encode_op` calls, which can each
/// emit more than one physical dispatch (a split-reduce merge pass).
#[cfg(feature = "instrument")]
pub static PHYSICAL_DISPATCH_CALLS: Counter = Counter::new("omega.metal.physical_dispatch_calls");
/// Number of `MTLCommandBuffer`s committed for the call, not number of
/// `execute` calls -- `PROXIMA_COMMAND_BUFFER_CHUNKS=K` (K=1 by default)
/// splits one token's dispatch sequence into K buffers on the placements
/// path (`placements_execute_named.rs`), so this counts K there instead of
/// the flat 1 every other execution path still records.
#[cfg(feature = "instrument")]
pub static GPU_EXEC_CALLS: Counter = Counter::new("omega.metal.gpu_exec_calls");
#[cfg(feature = "instrument")]
pub static GPU_EXEC_TICKS: Counter = Counter::new("omega.metal.gpu_exec_ticks");
#[cfg(feature = "instrument")]
pub static READBACK_CALLS: Counter = Counter::new("omega.metal.readback_calls");
#[cfg(feature = "instrument")]
pub static READBACK_TICKS: Counter = Counter::new("omega.metal.readback_ticks");
#[cfg(feature = "instrument")]
pub static READBACK_BYTES: Counter = Counter::new("omega.metal.readback_bytes");
/// `execute_plan_inner`'s per-op buffer-retirement forward scan (ROW 13ms
/// attribution) -- calls counts retired-candidate iterations, not ops, so a
/// caller can compare it against `prepared.resolved.len()` directly.
#[cfg(feature = "instrument")]
pub static RETIRE_SCAN_CALLS: Counter = Counter::new("omega.metal.retire_scan_calls");
#[cfg(feature = "instrument")]
pub static RETIRE_SCAN_TICKS: Counter = Counter::new("omega.metal.retire_scan_ticks");
/// `execute_plan_inner`'s per-op `expert_buffers_for` lookup -- same 13ms
/// attribution effort as [`RETIRE_SCAN_TICKS`].
#[cfg(feature = "instrument")]
pub static EXPERT_BUFFERS_LOOKUP_CALLS: Counter =
    Counter::new("omega.metal.expert_buffers_lookup_calls");
#[cfg(feature = "instrument")]
pub static EXPERT_BUFFERS_LOOKUP_TICKS: Counter =
    Counter::new("omega.metal.expert_buffers_lookup_ticks");
/// `execute_plan_with_placements_inner`'s own per-position residual, never
/// wired to [`RETIRE_SCAN_TICKS`]/[`EXPERT_BUFFERS_LOOKUP_TICKS`]'s ROW-13ms
/// attribution: `arena_placement`, `plan_uniform_buffer`, and
/// `attention_scratch_buffer` -- the three lookups between `output_placed`
/// resolution and the `encode_op` call, on the placements path only.
#[cfg(feature = "instrument")]
pub static PLACEMENT_RESOLVE_CALLS: Counter = Counter::new("omega.metal.placement_resolve_calls");
#[cfg(feature = "instrument")]
pub static PLACEMENT_RESOLVE_TICKS: Counter = Counter::new("omega.metal.placement_resolve_ticks");
/// `execute_plan_with_placements_inner`'s per-position loop head: the
/// unconditional `trace!`-gate `BTreeMap` lookups over `bound.operands()`
/// plus `arena_placement` -- both run before [`PLACEMENT_RESOLVE_TICKS`]
/// starts.
#[cfg(feature = "instrument")]
pub static LOOP_HEAD_CALLS: Counter = Counter::new("omega.metal.loop_head_calls");
#[cfg(feature = "instrument")]
pub static LOOP_HEAD_TICKS: Counter = Counter::new("omega.metal.loop_head_ticks");
/// `execute_plan_with_placements_inner`'s own `encoder.finish()` call --
/// the sole real Metal API call between the last per-position dispatch and
/// [`GPU_EXEC_TICKS`]' own `commit()`.
#[cfg(feature = "instrument")]
pub static ENCODER_FINISH_TICKS: Counter = Counter::new("omega.metal.encoder_finish_ticks");
/// `execute_plan_with_placements_inner`'s own function-entry preamble --
/// expert-buffer merge plus `prepared`/`packed_operands` binds -- the sole
/// host span between the function's own entry and `step_encode_start`
/// (GAP.md's residual: unattributed host time between `resolve_cached_plan`
/// returning and the first `Instant::now()` clock this function starts).
#[cfg(feature = "instrument")]
pub static PRE_ENCODE_CALLS: Counter = Counter::new("omega.metal.pre_encode_calls");
#[cfg(feature = "instrument")]
pub static PRE_ENCODE_TICKS: Counter = Counter::new("omega.metal.pre_encode_ticks");

/// prefill599 retarget: `prepare`'s own internal phase split, five spans
/// covering every statement between `infer` and `Prepared`'s own
/// construction (`prepare_uniforms_pack::prepare`'s call sites name which
/// span each wraps). Distinct from [`PREPARE_TICKS`] (the whole-function
/// wall clock `plan_with_placed_inputs` already records) -- these are what
/// let a caller attribute that ~28 ms bucket-crossing cost to a specific
/// phase instead of treating `prepare` as one opaque block.
#[cfg(feature = "instrument")]
pub static PREPARE_INFER_TICKS: Counter = Counter::new("omega.metal.prepare_infer_ticks");
#[cfg(feature = "instrument")]
pub static PREPARE_VALIDATE_TICKS: Counter = Counter::new("omega.metal.prepare_validate_ticks");
#[cfg(feature = "instrument")]
pub static PREPARE_BIND_TICKS: Counter = Counter::new("omega.metal.prepare_bind_ticks");
#[cfg(feature = "instrument")]
pub static PREPARE_OPTIMIZE_TICKS: Counter = Counter::new("omega.metal.prepare_optimize_ticks");
#[cfg(feature = "instrument")]
pub static PREPARE_RETIRE_TICKS: Counter = Counter::new("omega.metal.prepare_retire_ticks");

/// `build_plan_uniforms`'s own wall clock, timed at its sole lazy call site
/// ([`arena_encode_dispatch_finish::plan_uniform_buffer`]) rather than inside
/// itself, since a warm call never re-enters the function at all (the
/// `OnceCell` is already populated) -- the counter only ever advances on the
/// same plan-cache-miss step [`PREPARE_TICKS`] does.
#[cfg(feature = "instrument")]
pub static BUILD_PLAN_UNIFORMS_TICKS: Counter =
    Counter::new("omega.metal.build_plan_uniforms_ticks");
/// [`arena_encode_dispatch_finish::resolve_steps`]'s own wall clock, timed
/// past its staleness check so a plan-cache HIT (the common case, an
/// immediate `return Ok(())`) never advances this counter -- only a genuine
/// miss or a `set_math_mode` change compiles pipelines and pays this.
#[cfg(feature = "instrument")]
pub static RESOLVE_STEPS_TICKS: Counter = Counter::new("omega.metal.resolve_steps_ticks");
/// One `newBufferWithLength_options` call inside [`build_plan_uniforms`] per
/// resolved position -- the uniform-buffer allocation count a bucket
/// crossing pays once per fresh `Plan`, never again for that plan's
/// lifetime.
#[cfg(feature = "instrument")]
pub static UNIFORM_BUFFER_ALLOCATIONS: Counter =
    Counter::new("omega.metal.uniform_buffer_allocations");

/// One [`execute_plan`] call's worth of the split-4019 counters above,
/// snapshot-and-reset so a caller (the metal decode test) can read a
/// PER-TOKEN delta rather than a cumulative mean over the whole run —
/// guiding-principle 19's "per-token records, not just a mean".
#[cfg(feature = "instrument")]
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalStageTotals {
    pub prepare_calls: u64,
    pub prepare_ticks: u64,
    pub emit_calls: u64,
    pub emit_ticks: u64,
    pub pipeline_hits: u64,
    pub pipeline_misses: u64,
    pub pipeline_compile_ticks: u64,
    pub block_upload_calls: u64,
    pub block_upload_ticks: u64,
    /// Every block's own declared byte length, summed regardless of which
    /// terminal upload path served it -- see [`BLOCK_OFFERED_BYTES`]'s own
    /// doc. `block_copied_bytes + block_nocopy_bound_bytes +
    /// block_offset_bound_bytes == block_offered_bytes` on every step EXCEPT
    /// one where a resident-copy cache hit served bytes for free -- see
    /// [`BLOCK_COPIED_BYTES`]'s own doc for why that hit contributes to
    /// neither term.
    pub block_offered_bytes: u64,
    /// Bytes bound through a real host->device copy this step -- see
    /// [`BLOCK_COPIED_BYTES`]'s own doc.
    pub block_copied_bytes: u64,
    /// Bytes bound zero-copy (no-copy path, cached or uncached) this step --
    /// see [`BLOCK_NOCOPY_BOUND_BYTES`]'s own doc.
    pub block_nocopy_bound_bytes: u64,
    /// Bytes bound at an offset into the shared checkpoint-mapping buffer
    /// this step -- see [`BLOCK_OFFSET_BOUND_BYTES`]'s own doc.
    pub block_offset_bound_bytes: u64,
    pub op_setup_calls: u64,
    pub op_setup_ticks: u64,
    pub pipeline_lookup_calls: u64,
    pub pipeline_lookup_ticks: u64,
    pub encode_dispatch_calls: u64,
    pub encode_dispatch_ticks: u64,
    /// [`PHYSICAL_DISPATCH_CALLS`]'s own per-step delta.
    pub physical_dispatch_calls: u64,
    pub gpu_exec_calls: u64,
    pub gpu_exec_ticks: u64,
    pub readback_calls: u64,
    pub readback_ticks: u64,
    pub readback_bytes: u64,
    /// Split of `block_upload_calls` above by host->device path — reads back
    /// the ALREADY-EXISTING [`NOCOPY_BUFFER_UPLOADS`]/[`COPYING_BUFFER_UPLOADS`]/
    /// [`NOCOPY_BUFFER_REUSES`] counters (see this module's "Host buffer
    /// upload" doc) rather than adding parallel ones, so a caller can tell
    /// whether the 380+ ms/token `block_upload_ticks` figure is genuine
    /// no-copy cache misses (a real `newBufferWithBytes*` driver call) or a
    /// growing count of small COPYING allocations (the KV cache's own
    /// re-`Vec`-allocated-every-append pointer, which can never take the
    /// no-copy path).
    pub nocopy_uploads: u64,
    pub copying_uploads: u64,
    pub nocopy_reuses: u64,
    /// How many of `block_upload_calls` were a caller-declared-static weight
    /// -- see [`Plan::mark_resident`] and `upload_resident_copy`. A resident
    /// upload happens once per distinct weight buffer; a resident reuse
    /// happens every token after that. `resident_reuses` growing while
    /// `resident_uploads` stays flat at the weight count is the direct
    /// witness that the ~5.84 GB/token copy `proxima-tensor/docs/discipline.md`
    /// ROW 82 measured moved exactly once, not every step.
    pub resident_uploads: u64,
    pub resident_reuses: u64,
    /// How many of `block_upload_calls` were served by
    /// `checkpoint_mapping_offset` -- addressed by offset into the ONE
    /// no-copy buffer spanning the whole checkpoint mapping, instead of
    /// falling to `resident_uploads`' per-tensor copy. `resident_uploads`
    /// staying at 0 while this climbs to the packed-weight count is the
    /// direct witness the 429,173,760-byte per-tensor copy this counter
    /// replaces never happens.
    pub mapping_offset_uploads: u64,
    /// [`MAPPING_REBOUND_BLOCKS`]'s own per-step delta -- see that counter's
    /// own doc. Zero on every step where `block_buffer_reusable` correctly
    /// fast-skips every already-bound resident block.
    pub mapping_rebound_blocks: u64,
    pub expert_mapping_candidate_uploads: u64,
    pub expert_mapping_missed_uploads: u64,
    /// CARD 6.5 census: [`OUTPUT_BUFFER_ALLOCATIONS`]'s own per-step delta --
    /// `op_count` every step with `metal-plan-stable-buffers` off, `op_count`
    /// only on the step that builds a plan (a plan-cache miss) and 0 on
    /// every following plan-cache-hit step with it on.
    pub output_buffer_allocations: u64,
    /// Device bytes requested by those fresh output-buffer allocations.
    pub output_buffer_allocated_bytes: u64,
    /// [`PLAN_UNIFORM_WRITES`]'s own per-step delta -- `op_count` on the
    /// first execution that initializes a stable plan's uniforms and zero
    /// on every warm execution of that plan.
    pub plan_uniform_writes: u64,
    /// [`BARRIERS_EMITTED`]'s own per-step delta -- 0 on
    /// [`DispatchType::Serial`], the count of dataflow hazards the private
    /// `HazardTracker` actually found on [`DispatchType::Concurrent`].
    pub barriers_emitted: u64,
    /// [`BARRIERS_RAW`]'s own per-step delta -- barriers caused by a genuine
    /// dataflow edge (an input was written since the last barrier).
    pub barriers_raw: u64,
    /// [`BARRIERS_WAW`]'s own per-step delta -- barriers caused by this op's
    /// output identity having been written since the last barrier.
    pub barriers_waw: u64,
    /// [`BARRIERS_WAR`]'s own per-step delta -- barriers caused by this op's
    /// output identity having been read since the last barrier.
    pub barriers_war: u64,
    /// [`BARRIERS_WAW_WAR_ARENA_RECYCLED`]'s own per-step delta -- of
    /// `barriers_waw + barriers_war`, how many collided on a
    /// [`BufferArena`] slot shared with an earlier, unrelated position (a
    /// false dependency slot reuse manufactured).
    pub barriers_waw_war_arena_recycled: u64,
    /// [`BARRIERS_WAW_WAR_PERSISTENT`]'s own per-step delta -- of
    /// `barriers_waw + barriers_war`, how many collided on a persistent or
    /// output-placed buffer genuinely written more than once.
    pub barriers_waw_war_persistent: u64,
    pub expert_source_cache_hits: u64,
    pub expert_source_cache_misses: u64,
    pub expert_source_cache_cold_misses: u64,
    pub expert_source_cache_replacement_misses: u64,
    pub expert_source_buffer_reuses: u64,
    pub expert_source_reuse_copy_bytes: u64,
    pub expert_source_reuse_copy_ticks: u64,
    pub plan_handoff_reuses: u64,
    pub expert_source_cache_entries: u64,
    pub nocopy_cache_entries: u64,
    /// [`RETIRE_SCAN_CALLS`]'s own per-step delta.
    pub retire_scan_calls: u64,
    /// [`RETIRE_SCAN_TICKS`]'s own per-step delta.
    pub retire_scan_ticks: u64,
    /// [`EXPERT_BUFFERS_LOOKUP_CALLS`]'s own per-step delta.
    pub expert_buffers_lookup_calls: u64,
    /// [`EXPERT_BUFFERS_LOOKUP_TICKS`]'s own per-step delta.
    pub expert_buffers_lookup_ticks: u64,
    /// [`PLACEMENT_RESOLVE_CALLS`]'s own per-step delta.
    pub placement_resolve_calls: u64,
    /// [`PLACEMENT_RESOLVE_TICKS`]'s own per-step delta.
    pub placement_resolve_ticks: u64,
    /// [`LOOP_HEAD_CALLS`]'s own per-step delta.
    pub loop_head_calls: u64,
    /// [`LOOP_HEAD_TICKS`]'s own per-step delta.
    pub loop_head_ticks: u64,
    /// [`ENCODER_FINISH_TICKS`]'s own per-step delta.
    pub encoder_finish_ticks: u64,
    /// [`PRE_ENCODE_CALLS`]'s own per-step delta.
    pub pre_encode_calls: u64,
    /// [`PRE_ENCODE_TICKS`]'s own per-step delta.
    pub pre_encode_ticks: u64,
    /// [`PREPARE_INFER_TICKS`]'s own per-step delta.
    pub prepare_infer_ticks: u64,
    /// [`PREPARE_VALIDATE_TICKS`]'s own per-step delta.
    pub prepare_validate_ticks: u64,
    /// [`PREPARE_BIND_TICKS`]'s own per-step delta.
    pub prepare_bind_ticks: u64,
    /// [`PREPARE_OPTIMIZE_TICKS`]'s own per-step delta.
    pub prepare_optimize_ticks: u64,
    /// [`PREPARE_RETIRE_TICKS`]'s own per-step delta.
    pub prepare_retire_ticks: u64,
    /// [`BUILD_PLAN_UNIFORMS_TICKS`]'s own per-step delta.
    pub build_plan_uniforms_ticks: u64,
    /// [`RESOLVE_STEPS_TICKS`]'s own per-step delta.
    pub resolve_steps_ticks: u64,
    /// [`UNIFORM_BUFFER_ALLOCATIONS`]'s own per-step delta.
    pub uniform_buffer_allocations: u64,
}

/// Reads and resets every split-4019 counter in one call — see
/// [`MetalStageTotals`]'s own doc for why snapshot-and-reset rather than
/// `.get()`.
#[cfg(feature = "instrument")]
pub fn metal_stage_totals() -> MetalStageTotals {
    MetalStageTotals {
        prepare_calls: PREPARE_CALLS.snapshot_and_reset(),
        prepare_ticks: PREPARE_TICKS.snapshot_and_reset(),
        emit_calls: EMIT_CALLS.snapshot_and_reset(),
        emit_ticks: EMIT_TICKS.snapshot_and_reset(),
        pipeline_hits: PIPELINE_HITS.snapshot_and_reset(),
        pipeline_misses: PIPELINE_MISSES.snapshot_and_reset(),
        pipeline_compile_ticks: PIPELINE_COMPILE_TICKS.snapshot_and_reset(),
        block_upload_calls: BLOCK_UPLOAD_CALLS.snapshot_and_reset(),
        block_upload_ticks: BLOCK_UPLOAD_TICKS.snapshot_and_reset(),
        block_offered_bytes: BLOCK_OFFERED_BYTES.snapshot_and_reset(),
        block_copied_bytes: BLOCK_COPIED_BYTES.snapshot_and_reset(),
        block_nocopy_bound_bytes: BLOCK_NOCOPY_BOUND_BYTES.snapshot_and_reset(),
        block_offset_bound_bytes: BLOCK_OFFSET_BOUND_BYTES.snapshot_and_reset(),
        op_setup_calls: OP_SETUP_CALLS.snapshot_and_reset(),
        op_setup_ticks: OP_SETUP_TICKS.snapshot_and_reset(),
        pipeline_lookup_calls: PIPELINE_LOOKUP_CALLS.snapshot_and_reset(),
        pipeline_lookup_ticks: PIPELINE_LOOKUP_TICKS.snapshot_and_reset(),
        encode_dispatch_calls: ENCODE_DISPATCH_CALLS.snapshot_and_reset(),
        encode_dispatch_ticks: ENCODE_DISPATCH_TICKS.snapshot_and_reset(),
        physical_dispatch_calls: PHYSICAL_DISPATCH_CALLS.snapshot_and_reset(),
        gpu_exec_calls: GPU_EXEC_CALLS.snapshot_and_reset(),
        gpu_exec_ticks: GPU_EXEC_TICKS.snapshot_and_reset(),
        readback_calls: READBACK_CALLS.snapshot_and_reset(),
        readback_ticks: READBACK_TICKS.snapshot_and_reset(),
        readback_bytes: READBACK_BYTES.snapshot_and_reset(),
        nocopy_uploads: NOCOPY_BUFFER_UPLOADS.snapshot_and_reset(),
        copying_uploads: COPYING_BUFFER_UPLOADS.snapshot_and_reset(),
        nocopy_reuses: NOCOPY_BUFFER_REUSES.snapshot_and_reset(),
        resident_uploads: RESIDENT_BUFFER_UPLOADS.snapshot_and_reset(),
        resident_reuses: RESIDENT_BUFFER_REUSES.snapshot_and_reset(),
        mapping_offset_uploads: MAPPING_OFFSET_UPLOADS.snapshot_and_reset(),
        mapping_rebound_blocks: MAPPING_REBOUND_BLOCKS.snapshot_and_reset(),
        expert_mapping_candidate_uploads: EXPERT_MAPPING_CANDIDATE_UPLOADS.snapshot_and_reset(),
        expert_mapping_missed_uploads: EXPERT_MAPPING_MISSED_UPLOADS.snapshot_and_reset(),
        output_buffer_allocations: OUTPUT_BUFFER_ALLOCATIONS.snapshot_and_reset(),
        output_buffer_allocated_bytes: OUTPUT_BUFFER_ALLOCATED_BYTES.snapshot_and_reset(),
        plan_uniform_writes: PLAN_UNIFORM_WRITES.snapshot_and_reset(),
        barriers_emitted: BARRIERS_EMITTED.snapshot_and_reset(),
        barriers_raw: BARRIERS_RAW.snapshot_and_reset(),
        barriers_waw: BARRIERS_WAW.snapshot_and_reset(),
        barriers_war: BARRIERS_WAR.snapshot_and_reset(),
        barriers_waw_war_arena_recycled: BARRIERS_WAW_WAR_ARENA_RECYCLED.snapshot_and_reset(),
        barriers_waw_war_persistent: BARRIERS_WAW_WAR_PERSISTENT.snapshot_and_reset(),
        expert_source_cache_hits: EXPERT_SOURCE_CACHE_HITS.snapshot_and_reset(),
        expert_source_cache_misses: EXPERT_SOURCE_CACHE_MISSES.snapshot_and_reset(),
        expert_source_cache_cold_misses: EXPERT_SOURCE_CACHE_COLD_MISSES.snapshot_and_reset(),
        expert_source_cache_replacement_misses: EXPERT_SOURCE_CACHE_REPLACEMENT_MISSES
            .snapshot_and_reset(),
        expert_source_buffer_reuses: EXPERT_SOURCE_BUFFER_REUSES.snapshot_and_reset(),
        expert_source_reuse_copy_bytes: EXPERT_SOURCE_REUSE_COPY_BYTES.snapshot_and_reset(),
        expert_source_reuse_copy_ticks: EXPERT_SOURCE_REUSE_COPY_TICKS.snapshot_and_reset(),
        plan_handoff_reuses: PLAN_HANDOFF_REUSES.snapshot_and_reset(),
        expert_source_cache_entries: EXPERT_SOURCE_CACHE.with(|cache| cache.borrow().len() as u64),
        nocopy_cache_entries: NOCOPY_BUFFERS.with(|cache| cache.borrow().len() as u64),
        retire_scan_calls: RETIRE_SCAN_CALLS.snapshot_and_reset(),
        retire_scan_ticks: RETIRE_SCAN_TICKS.snapshot_and_reset(),
        expert_buffers_lookup_calls: EXPERT_BUFFERS_LOOKUP_CALLS.snapshot_and_reset(),
        expert_buffers_lookup_ticks: EXPERT_BUFFERS_LOOKUP_TICKS.snapshot_and_reset(),
        placement_resolve_calls: PLACEMENT_RESOLVE_CALLS.snapshot_and_reset(),
        placement_resolve_ticks: PLACEMENT_RESOLVE_TICKS.snapshot_and_reset(),
        loop_head_calls: LOOP_HEAD_CALLS.snapshot_and_reset(),
        loop_head_ticks: LOOP_HEAD_TICKS.snapshot_and_reset(),
        encoder_finish_ticks: ENCODER_FINISH_TICKS.snapshot_and_reset(),
        pre_encode_calls: PRE_ENCODE_CALLS.snapshot_and_reset(),
        pre_encode_ticks: PRE_ENCODE_TICKS.snapshot_and_reset(),
        prepare_infer_ticks: PREPARE_INFER_TICKS.snapshot_and_reset(),
        prepare_validate_ticks: PREPARE_VALIDATE_TICKS.snapshot_and_reset(),
        prepare_bind_ticks: PREPARE_BIND_TICKS.snapshot_and_reset(),
        prepare_optimize_ticks: PREPARE_OPTIMIZE_TICKS.snapshot_and_reset(),
        prepare_retire_ticks: PREPARE_RETIRE_TICKS.snapshot_and_reset(),
        build_plan_uniforms_ticks: BUILD_PLAN_UNIFORMS_TICKS.snapshot_and_reset(),
        resolve_steps_ticks: RESOLVE_STEPS_TICKS.snapshot_and_reset(),
        uniform_buffer_allocations: UNIFORM_BUFFER_ALLOCATIONS.snapshot_and_reset(),
    }
}

/// Maps this backend's own counters into the backend-neutral
/// [`crate::backend::StageTotals`], snapshot-and-reset. Without `instrument`
/// only the copied-bytes counter exists, so only that field is filled.
#[cfg(feature = "instrument")]
pub fn shared_stage_totals() -> crate::backend::StageTotals {
    let stage = metal_stage_totals();
    let nanos = proxima_tensor::instrument::ticks_to_nanos;
    crate::backend::StageTotals {
        block_copied_bytes: stage.block_copied_bytes,
        pipeline_misses: stage.pipeline_misses,
        pipeline_compile_nanos: nanos(stage.pipeline_compile_ticks),
        prepare_nanos: nanos(stage.prepare_ticks),
        emit_nanos: nanos(stage.emit_ticks),
        pipeline_lookup_nanos: nanos(stage.pipeline_lookup_ticks),
        op_setup_nanos: nanos(stage.op_setup_ticks),
        gpu_exec_nanos: nanos(stage.gpu_exec_ticks),
        encode_dispatch_nanos: nanos(stage.encode_dispatch_ticks),
        readback_nanos: nanos(stage.readback_ticks),
        block_upload_nanos: nanos(stage.block_upload_ticks),
        resolve_steps_nanos: nanos(stage.resolve_steps_ticks),
    }
}

#[cfg(not(feature = "instrument"))]
pub fn shared_stage_totals() -> crate::backend::StageTotals {
    crate::backend::StageTotals {
        block_copied_bytes: BLOCK_COPIED_BYTES.snapshot_and_reset(),
        ..crate::backend::StageTotals::default()
    }
}

#[cfg(all(test, feature = "instrument"))]
pub(super) mod host_bookkeeping_instrument_tests {
    use proxima_telemetry::counter;

    use super::{
        EXPERT_BUFFERS_LOOKUP_CALLS, EXPERT_BUFFERS_LOOKUP_TICKS, RETIRE_SCAN_CALLS,
        RETIRE_SCAN_TICKS, metal_stage_totals,
    };

    /// The three new counters travel through [`metal_stage_totals`] the same
    /// way every other split-4019 counter does — a caller reading the
    /// snapshot after a run sees them printed in the struct's OWN fields, not
    /// silently zeroed out. `metal_stage_totals()` snapshot-and-resets
    /// process-global `Counter` statics that real dispatch code (and other
    /// concurrently running tests) also increments, so this asserts a FLOOR
    /// (this test's own contribution can only ever be added to by a
    /// concurrent writer, never subtracted) instead of an exact value against
    /// a shared, resettable counter — no lock, no serialization needed.
    #[test]
    fn new_host_bookkeeping_fields_print_in_stage_totals() {
        counter!(RETIRE_SCAN_CALLS, 3);
        counter!(RETIRE_SCAN_TICKS, 7);
        counter!(EXPERT_BUFFERS_LOOKUP_CALLS, 1);
        counter!(EXPERT_BUFFERS_LOOKUP_TICKS, 2);

        let totals = metal_stage_totals();

        assert!(
            totals.retire_scan_calls >= 3,
            "expected at least this test's own 3, got {}",
            totals.retire_scan_calls
        );
        assert!(
            totals.retire_scan_ticks >= 7,
            "expected at least this test's own 7, got {}",
            totals.retire_scan_ticks
        );
        assert!(
            totals.expert_buffers_lookup_calls >= 1,
            "expected at least this test's own 1, got {}",
            totals.expert_buffers_lookup_calls
        );
        assert!(
            totals.expert_buffers_lookup_ticks >= 2,
            "expected at least this test's own 2, got {}",
            totals.expert_buffers_lookup_ticks
        );
    }
}

/// How many real `upload_block` calls took each host->device path —
/// incremented once per call, never per byte, so a caller can read back the
/// no-copy hit rate after a run without an external profiler. See the
/// module doc's "Host buffer upload" section.
pub static NOCOPY_BUFFER_UPLOADS: Counter = Counter::new("omega.metal.upload_block.nocopy");
pub static COPYING_BUFFER_UPLOADS: Counter = Counter::new("omega.metal.upload_block.copy");

/// Bytes bound through a real host->device copy this call
/// (`upload_block_copy`, or `upload_resident_copy` on a cache MISS only) --
/// fired at the point the bytes actually move, never before the lookup that
/// can avoid moving them. `upload_resident_copy`'s own cache HIT does not
/// increment this counter (no bytes moved, the same buffer is reused), so
/// `BLOCK_COPIED_BYTES + BLOCK_NOCOPY_BOUND_BYTES + BLOCK_OFFSET_BOUND_BYTES`
/// sums to `BLOCK_OFFERED_BYTES` (feature-gated behind `instrument`) minus
/// whatever a resident-copy cache hit
/// served for free that step. See the module doc's "Host buffer upload"
/// section for why most weight bytes never reach this counter -- only a
/// misaligned, non-resident block (the KV cache, which is deliberately never
/// cached: see `upload_block_copy`'s own doc) pays a real copy every token.
pub static BLOCK_COPIED_BYTES: Counter = Counter::new("omega.metal.block_copied_bytes");
/// Bytes bound zero-copy, either `upload_block_no_copy` (cached, resident)
/// or `upload_block_no_copy_uncached` (uncached) — the other terminal-path
/// byte split of `BLOCK_OFFERED_BYTES` (feature-gated behind `instrument`).
pub static BLOCK_NOCOPY_BOUND_BYTES: Counter = Counter::new("omega.metal.block_nocopy_bound_bytes");
/// Bytes bound at an offset into the single whole-checkpoint no-copy buffer
/// (`checkpoint_mapping_offset`) — the third terminal-path byte split of
/// `BLOCK_OFFERED_BYTES` (feature-gated behind `instrument`), and the one that carries the bulk of a real
/// model's weight bytes once `register_checkpoint_mapping` is in effect.
pub static BLOCK_OFFSET_BOUND_BYTES: Counter = Counter::new("omega.metal.block_offset_bound_bytes");

/// The host's page size, queried once and cached — the alignment unit
/// `newBufferWithBytesNoCopy` requires for both the pointer and the length
/// (16384 on Apple silicon, but this asks the OS rather than hard-coding
/// that). Public so a caller building block inputs (e.g.
/// `proxima_tensor::AlignedBuffer::new`) can size an allocation to this
/// exact host's page size instead of duplicating the sysconf call.
pub fn page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    // SAFETY: `sysconf` takes a plain `c_int` name and has no preconditions;
    // `_SC_PAGESIZE` is POSIX-portable (macOS's `libc` crate has no
    // `getpagesize()` binding, unlike Linux's).
    *PAGE_SIZE.get_or_init(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize })
}

/// `newBufferWithBytesNoCopy`'s hard requirement: `pointer` and `length`
/// must both land on a page boundary.
pub(super) fn is_page_aligned(pointer: *const c_void, length: usize) -> bool {
    let page = page_size();
    (pointer as usize).is_multiple_of(page) && length.is_multiple_of(page)
}

/// Narrows the caller's f32 host data to `dtype`'s own width before
/// uploading — see this module's dtype doc for why that narrowing happens
/// exactly once, here, rather than the device buffer staying 4 bytes per
/// element regardless of `dtype`. `node` names the block input this upload
/// is for, used only to point an [`EmitError::UnsupportedDType`] at the
/// right place — [`reject_unsupported_gpu_dtype`] already keeps anything
/// but `Float32`/`Float16` from reaching this call inside [`execute`], so
/// the new arm below is a totality guard, not a path this driver's own
/// pipeline can actually hit.
pub(super) fn upload_block(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
    node: NodeId,
    dtype: DType,
    block_name: Option<&str>,
    resident_name: Option<&str>,
) -> Result<(MetalBuffer, usize), MetalError> {
    if !data.is_empty() {
        debug!(
            node = ?node,
            name = ?block_name,
            dtype = ?dtype,
            bytes = size_of_val(data) as u64,
            resident = ?resident_name,
            "metal_block_upload"
        );
    }
    match device_scalar_token(dtype) {
        // unreached by every program this driver compiles today (none
        // declares a `Float16` block input -- `proxima-tensor/src/spec.rs`'s
        // `gqa_cached_forward_program` is `Float32` throughout), so it is
        // not worth the residency cache's extra bookkeeping: the narrowed
        // `Vec<f16>` this allocates is dropped every call regardless.
        Some("half") => upload_block_as_half(device, data).map(|buffer| (buffer, 0)),
        Some(_) => upload_block_as_float(device, data, resident_name),
        None => Err(EmitError::UnsupportedDType { node, dtype, backend: "metal" }.into()),
    }
}

/// The only path that can take the no-copy upload: the caller's own
/// `&[f32]` slice is borrowed for [`execute`]'s entire call, which
/// `waitUntilCompleted`s its single command buffer (every op's reads
/// included) before that borrow can end, so handing the GPU the caller's
/// own pointer is sound whenever it is page-aligned. See the module doc's
/// "Host buffer upload" section for why [`upload_block_as_half`] can never
/// take this path. `resident` is [`Plan::mark_resident`]'s classification of
/// this block's own node -- see the module doc's "Resident blocks" section
/// for why a misaligned RESIDENT block still gets a cache, just a different
/// one than the no-copy path's.
///
/// CACHING the wrapper this creates is gated on `resident_name`, not on
/// `is_page_aligned` alone: page alignment is a property of an ADDRESS, not
/// of a LIFETIME, and `(pointer, byte_length)` is exactly the key an
/// ephemeral, growing buffer (a KV-cache row, say) can reuse after a
/// realloc moves a DIFFERENT allocation onto the same range. Only
/// `mark_resident`'s own NAME proof licenses remembering a buffer past this
/// one call -- see [`upload_resident_copy`]'s own doc for why that cache
/// keys on the name itself, never the address.
pub(super) fn upload_block_as_float(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
    resident_name: Option<&str>,
) -> Result<(MetalBuffer, usize), MetalError> {
    if data.is_empty() {
        return allocate_buffer(device, 0, DType::FLOAT32).map(|buffer| (buffer, 0));
    }
    let byte_length = size_of_val(data);
    let pointer = data.as_ptr().cast::<c_void>();
    // A mmap'd tensor is page-aligned by construction, so the alignment
    // check below always wins first if it runs first -- minting a dedicated
    // no-copy buffer that duplicates memory the whole-checkpoint mapping
    // (or the expert-sidecar mapping) already covers. Try the zero-cost
    // OFFSET views into an already-registered mapping before ever minting a
    // new buffer; only a genuinely separate allocation (a KV-cache row, a
    // scratch buffer) falls through to the page-aligned no-copy path.
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length, resident_name) {
        return result;
    }
    if let Some(result) = expert_mapping_offset(device, pointer, byte_length, resident_name) {
        return result;
    }
    if is_page_aligned(pointer, byte_length) {
        counter!(NOCOPY_BUFFER_UPLOADS, 1);
        counter!(BLOCK_NOCOPY_BOUND_BYTES, byte_length as u64);
        if let Some(name) = resident_name {
            return upload_block_no_copy(device, name, pointer, byte_length)
                .map(|buffer| (buffer, 0));
        }
        return upload_block_no_copy_uncached(device, pointer, byte_length)
            .map(|buffer| (buffer, 0));
    }
    if let Some(name) = resident_name {
        return upload_resident_copy(device, name, pointer, byte_length).map(|buffer| (buffer, 0));
    }
    record_block_copied(byte_length as u64);
    counter!(COPYING_BUFFER_UPLOADS, 1);
    upload_block_copy(device, pointer, byte_length).map(|buffer| (buffer, 0))
}

pub(super) fn upload_block_int32_as_float(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[i32],
    resident_name: Option<&str>,
) -> Result<(MetalBuffer, usize), MetalError> {
    let converted: Vec<f32> = data.iter().map(|value| *value as f32).collect();
    upload_block_as_float(device, converted.as_slice(), resident_name)
}

/// Uploads a packed quantized weight buffer as raw BYTES — no dequantize on
/// the host, which is the entire point. A 7B `Q4_K_S` checkpoint is 3.784 GB
/// packed against 14.5 GB as `f16`; decode is a weight sweep, so that 3.56x
/// in traffic IS the token rate. Reuses the same page-aligned no-copy path
/// [`upload_block_as_float`] uses, since a memory-mapped GGUF tensor is very
/// often already page-aligned. `resident_name` is the same "caller's own
/// static weight" classification [`upload_block_as_float`] takes; a packed
/// weight too misaligned for the no-copy path takes the same resident-copy
/// cache.
pub(super) fn upload_packed_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    resident_name: Option<&str>,
) -> Result<(MetalBuffer, usize), MetalError> {
    upload_packed_bytes_in(expert_mapping_identity(), device, bytes, resident_name)
}

/// [`upload_packed_bytes`] over an explicit expert-mapping identity; the
/// production entry point passes the process-global one.
pub(super) fn upload_packed_bytes_in(
    expert_mapping: Option<(usize, usize)>,
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    resident_name: Option<&str>,
) -> Result<(MetalBuffer, usize), MetalError> {
    if bytes.len() > 100 * 1024 * 1024 {
        debug!(
            bytes = bytes.len() as u64,
            resident = ?resident_name,
            "metal_large_packed_upload"
        );
    }
    if bytes.is_empty() {
        return allocate_buffer(device, 0, DType::FLOAT32).map(|buffer| (buffer, 0));
    }
    let byte_length = bytes.len();
    let pointer = bytes.as_ptr().cast::<c_void>();
    let mapping_published = expert_mapping.is_some();
    if mapping_published {
        counter!(EXPERT_MAPPING_CANDIDATE_UPLOADS, 1);
    }
    if let Some(result) =
        expert_mapping_offset_in(expert_mapping, device, pointer, byte_length, resident_name)
    {
        return result;
    }
    if mapping_published {
        counter!(EXPERT_MAPPING_MISSED_UPLOADS, 1);
    }
    // Same reorder as `upload_block_as_float`: a page-aligned quantized
    // tensor that also falls inside the whole-checkpoint mapping must be
    // served as a zero-cost offset view before minting a duplicate
    // dedicated buffer -- this is the path every GGUF quantized weight
    // (e.g. `blk.N.ffn_down_exps.weight`) actually takes.
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length, resident_name) {
        return result;
    }
    if is_page_aligned(pointer, byte_length) {
        counter!(NOCOPY_BUFFER_UPLOADS, 1);
        counter!(BLOCK_NOCOPY_BOUND_BYTES, byte_length as u64);
        if let Some(name) = resident_name {
            return upload_block_no_copy(device, name, pointer, byte_length)
                .map(|buffer| (buffer, 0));
        }
        return upload_block_no_copy_uncached(device, pointer, byte_length)
            .map(|buffer| (buffer, 0));
    }
    if let Some(name) = resident_name {
        return upload_resident_copy(device, name, pointer, byte_length).map(|buffer| (buffer, 0));
    }
    record_block_copied(byte_length as u64);
    counter!(COPYING_BUFFER_UPLOADS, 1);
    upload_block_copy(device, pointer, byte_length).map(|buffer| (buffer, 0))
}

thread_local! {
    /// The page-aligned, process-lifetime checkpoint mapping registered by
    /// [`register_checkpoint_mapping`] -- `(base_pointer, byte_length)` of the
    /// caller's own mmap. `None` until a loader calls it. A tensor whose byte
    /// range falls entirely inside this span never needs its own device
    /// buffer: see [`checkpoint_mapping_offset`].
    pub(super) static CHECKPOINT_MAPPING: RefCell<Option<(usize, usize)>> = const { RefCell::new(None) };
}

// The sidecar is registered by the host-side step boundary and consumed by
// the Metal execution thread.  This state cannot be thread-local: the
// mapped-window slice is process-owned and its address remains valid while a
// step is encoded.  Length zero is the publication fence, so a reader never
// observes a new length with the prior base address.
pub(super) static EXPERT_MAPPING_BASE: AtomicUsize = AtomicUsize::new(0);
pub(super) static EXPERT_MAPPING_LENGTH: AtomicUsize = AtomicUsize::new(0);

pub fn register_expert_mapping(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let identity = (bytes.as_ptr() as usize, bytes.len());
    let previous = expert_mapping_identity();
    EXPERT_MAPPING_LENGTH.store(0, Ordering::Release);
    EXPERT_MAPPING_BASE.store(identity.0, Ordering::Relaxed);
    EXPERT_MAPPING_LENGTH.store(identity.1, Ordering::Release);
    let changed = previous != Some(identity);
    if changed {
        EXPERT_SOURCE_CACHE.with(|cache| cache.borrow_mut().clear());
        evict_mapping_buffer(EXPERT_MAPPING_NOCOPY_NAME);
    }
}

pub fn unregister_expert_mapping(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let identity = (bytes.as_ptr() as usize, bytes.len());
    let matched = expert_mapping_identity() == Some(identity);
    if matched {
        EXPERT_MAPPING_LENGTH.store(0, Ordering::Release);
        EXPERT_MAPPING_BASE.store(0, Ordering::Relaxed);
    }
    if matched {
        EXPERT_SOURCE_CACHE.with(|cache| cache.borrow_mut().clear());
        evict_mapping_buffer(EXPERT_MAPPING_NOCOPY_NAME);
    }
}

/// Registers the whole-checkpoint memory mapping backing every packed
/// tensor's borrowed bytes, so a tensor whose own byte offset inside that
/// mapping is misaligned for `is_page_aligned` can still reach the GPU
/// without a copy -- by address, into ONE no-copy buffer spanning the whole
/// mapping, instead of one buffer per tensor. See `checkpoint_mapping_offset`
/// for the containment check and `upload_packed_bytes`'s doc for why the
/// per-tensor page-alignment test this replaces fails for every packed
/// tensor in a real GGUF layout (tensors are packed back-to-back at their
/// natural sizes; only the mapping's OWN base is page-aligned).
///
/// `bytes` must stay mapped, unchanged, at this address for the rest of the
/// process -- the same precondition `NOCOPY_BUFFERS` already rests on for
/// a single tensor, extended here to the whole file. Calling this again
/// replaces the previous registration; callers load one checkpoint mapping
/// per process in every reachable path today.
pub fn register_checkpoint_mapping(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let base = bytes.as_ptr() as usize;
    CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow_mut() = Some((base, bytes.len())));
    // this call is the caller EXPLICITLY declaring a new resident identity
    // for `CHECKPOINT_MAPPING_NOCOPY_NAME` -- drop whatever `NOCOPY_BUFFERS`
    // cached under that name for the PRIOR registration so the next
    // `checkpoint_mapping_offset` upload is a fresh, correctly-checked
    // entry rather than a `ResidentNameRebound` error against a mapping
    // this function itself just superseded.
    evict_mapping_buffer(CHECKPOINT_MAPPING_NOCOPY_NAME);
}

/// Advises the kernel that pages covering a checkpoint byte range are not
/// needed after an expert has been demoted to its sidecar copy. The mapping
/// remains valid; a later high promotion may fault the bytes back in.
pub fn discard_checkpoint_mmap_range(bytes: &[u8]) -> Result<(), MetalError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let page = page_size();
    let start = (bytes.as_ptr() as usize) & !(page - 1);
    let end = (bytes.as_ptr() as usize)
        .checked_add(bytes.len())
        .and_then(|value| value.checked_add(page - 1))
        .map(|value| value & !(page - 1))
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let length = end
        .checked_sub(start)
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let result = unsafe { libc::madvise(start as *mut libc::c_void, length, libc::MADV_FREE) };
    if result == 0 {
        Ok(())
    } else {
        Err(MetalError::CheckpointMmapDiscardFailed {
            errno: std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO),
        })
    }
}

/// Advises Darwin to reclaim mapped pages immediately when the caller is
/// enforcing a hard resident-memory ceiling. Unlike `MADV_FREE`, this does
/// not leave the pages in the process footprint until memory pressure arrives.
pub fn discard_checkpoint_mmap_range_immediate(bytes: &[u8]) -> Result<(), MetalError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let page = page_size();
    let start = (bytes.as_ptr() as usize) & !(page - 1);
    let end = (bytes.as_ptr() as usize)
        .checked_add(bytes.len())
        .and_then(|value| value.checked_add(page - 1))
        .map(|value| value & !(page - 1))
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let length = end
        .checked_sub(start)
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let result = unsafe { libc::madvise(start as *mut libc::c_void, length, libc::MADV_DONTNEED) };
    if result == 0 {
        Ok(())
    } else {
        Err(MetalError::CheckpointMmapDiscardFailed {
            errno: std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO),
        })
    }
}

/// Counts resident pages covering a checkpoint range using Darwin's `mincore`.
/// This is diagnostic-only and lets the serving harness distinguish mmap page
/// residency from retained heap allocations after an expert eviction.
pub fn checkpoint_mmap_resident_pages(bytes: &[u8]) -> Result<usize, MetalError> {
    if bytes.is_empty() {
        return Ok(0);
    }
    let page = page_size();
    let start = (bytes.as_ptr() as usize) & !(page - 1);
    let end = (bytes.as_ptr() as usize)
        .checked_add(bytes.len())
        .and_then(|value| value.checked_add(page - 1))
        .map(|value| value & !(page - 1))
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let length = end
        .checked_sub(start)
        .ok_or(MetalError::CheckpointMmapDiscardFailed {
            errno: libc::EINVAL,
        })?;
    let page_count = length / page;
    let mut residency = vec![0_i8; page_count];
    let result = unsafe {
        libc::mincore(
            start as *const libc::c_void,
            length,
            residency.as_mut_ptr().cast::<libc::c_char>(),
        )
    };
    if result != 0 {
        return Err(MetalError::CheckpointMmapDiscardFailed {
            errno: std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO),
        });
    }
    Ok(residency
        .iter()
        .filter(|state| **state & libc::MINCORE_INCORE as i8 != 0)
        .count())
}

/// Unregisters the checkpoint mapping [`register_checkpoint_mapping`]
/// installed for `bytes`, evicting its whole-mapping no-copy buffer from
/// `NOCOPY_BUFFERS` -- but ONLY when `bytes` is still the currently
/// registered mapping. [`register_checkpoint_mapping`] already evicts a
/// PRIOR mapping's entry the moment a new one is registered (see that
/// function's own doc), so a dropped model racing behind a second model's
/// load must not clear the second model's live mapping -- comparing the
/// base pointer and length is what tells "this is still mine" apart from
/// "someone else already superseded this". A no-op when `bytes` is empty
/// (matching [`register_checkpoint_mapping`]'s own early return) or when no
/// mapping this identity matches is currently registered.
pub fn unregister_checkpoint_mapping(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let identity = (bytes.as_ptr() as usize, bytes.len());
    let matched = CHECKPOINT_MAPPING.with(|mapping| {
        let mut mapping = mapping.borrow_mut();
        if *mapping == Some(identity) {
            *mapping = None;
            true
        } else {
            false
        }
    });
    if matched {
        evict_mapping_buffer(CHECKPOINT_MAPPING_NOCOPY_NAME);
    }
}

/// Counts uploads served by addressing the shared checkpoint-mapping buffer
/// at an offset, instead of copying the tensor into its own buffer -- the
/// direct witness for the census this mechanism is meant to zero out.
pub static MAPPING_OFFSET_UPLOADS: Counter =
    Counter::new("omega.metal.upload_block.mapping_offset");
static MAPPING_OFFSET_UPLOADS_TOTAL: AtomicU64 = AtomicU64::new(0);
static BLOCK_COPIED_BYTES_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Everything `MAPPING_OFFSET_UPLOADS` ever counted. `metal_stage_totals`
/// drains the counter after every decode step, so a reader taking a delta
/// across a decode reads this never-reset total instead.
pub fn mapping_offset_uploads_total() -> u64 {
    MAPPING_OFFSET_UPLOADS_TOTAL.load(Ordering::Relaxed)
}

/// Everything `BLOCK_COPIED_BYTES` ever counted; see `mapping_offset_uploads_total`.
pub fn block_copied_bytes_total() -> u64 {
    BLOCK_COPIED_BYTES_TOTAL.load(Ordering::Relaxed)
}

pub(super) fn record_block_copied(byte_length: u64) {
    counter!(BLOCK_COPIED_BYTES, byte_length);
    BLOCK_COPIED_BYTES_TOTAL.fetch_add(byte_length, Ordering::Relaxed);
}

fn record_mapping_offset_upload() {
    counter!(MAPPING_OFFSET_UPLOADS, 1);
    MAPPING_OFFSET_UPLOADS_TOTAL.fetch_add(1, Ordering::Relaxed);
}
/// Counts packed-source uploads observed while the bounded expert mapping was
/// published. The paired miss counter distinguishes a mapping that exists from
/// a source slice whose address/length falls outside that mapping.
pub static EXPERT_MAPPING_CANDIDATE_UPLOADS: Counter =
    Counter::new("omega.metal.expert_mapping_candidate_uploads");
pub static EXPERT_MAPPING_MISSED_UPLOADS: Counter =
    Counter::new("omega.metal.expert_mapping_missed_uploads");

/// If `pointer..pointer+byte_length` falls entirely inside the registered
/// checkpoint mapping, returns the whole-mapping no-copy buffer (created
/// once, reused after) plus this tensor's byte OFFSET into it. `None` when
/// no mapping is registered or the range falls outside it -- the scratch and
/// KV-cache buffers `upload_block_as_float`'s `Float32` arm also uploads
/// through never live inside the checkpoint's own mmap, so they fall
/// through unchanged.
///
/// The mapping's total length is rounded UP to a page boundary before
/// `newBufferWithBytesNoCopy` sees it, since that call requires a
/// page-aligned length; that rounding is sound because `mmap` only ever
/// backs a file with whole pages, so every byte up to the next page
/// boundary past the file's own length is already resident, zero-filled,
/// mapped memory (never past the region the OS mapped for this file) --
/// reading it is safe, and no kernel this driver emits ever reads past a
/// tensor's own declared byte length regardless.
pub(super) fn checkpoint_mapping_offset(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
    tensor_name: Option<&str>,
) -> Option<Result<(MetalBuffer, usize), MetalError>> {
    let (base, mapping_length) = CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow())?;
    let address = pointer as usize;
    if address < base || address + byte_length > base + mapping_length {
        return None;
    }
    if !mapping_offset_is_aligned(address - base, tensor_name) {
        return None;
    }
    record_mapping_offset_upload();
    counter!(BLOCK_OFFSET_BOUND_BYTES, byte_length as u64);
    Some(
        mapping_buffer(device, CHECKPOINT_MAPPING_NOCOPY_NAME, base, mapping_length)
            .map(|buffer| (buffer, address - base)),
    )
}

/// Byte alignment a tensor's offset inside a registered mapping must have to
/// be bound as a view. The packed matvec reads its weight with 4-byte and
/// wider vector loads that Metal does not guarantee at an unaligned address:
/// `bf16_matvec_mapping_offset_alignment` measures NaN logits at offsets 1
/// and 3 and exact ones at 0 and 8 (PROXNUM-569), and Metal's own
/// `setBuffer` offset rule is a multiple of the largest element loaded. 8 is
/// the widest scalar load the kernels issue.
pub(super) const MAPPING_VIEW_OFFSET_ALIGNMENT: usize = 8;

/// Counts tensors inside a registered mapping that took the copy path
/// because their offset broke `MAPPING_VIEW_OFFSET_ALIGNMENT`.
pub static MISALIGNED_MAPPING_OFFSET_COPIES: Counter =
    Counter::new("omega.metal.upload_block.misaligned_mapping_offset_copy");

/// Whether `offset` may be bound as a no-copy view. A misaligned offset
/// returns `false` after a `debug!` event naming the tensor, and the caller
/// falls through to the copy path: correct, only slower.
pub(super) fn mapping_offset_is_aligned(offset: usize, tensor_name: Option<&str>) -> bool {
    if offset.is_multiple_of(MAPPING_VIEW_OFFSET_ALIGNMENT) {
        return true;
    }
    counter!(MISALIGNED_MAPPING_OFFSET_COPIES, 1);
    debug!(
        tensor = ?tensor_name,
        offset = offset as u64,
        required_alignment = MAPPING_VIEW_OFFSET_ALIGNMENT as u64,
        "metal_misaligned_mapping_offset_copied"
    );
    false
}

/// The whole-mapping no-copy buffer for a registered mapping, created once
/// under `name` and served from the no-copy cache after. The length is
/// rounded up to a page because `newBufferWithBytesNoCopy` requires it.
pub(super) fn mapping_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    base: usize,
    mapping_length: usize,
) -> Result<MetalBuffer, MetalError> {
    let page = page_size();
    let rounded_length = mapping_length.div_ceil(page) * page;
    let cached = NOCOPY_BUFFERS.with(|cache| cache.borrow().contains_key(name));
    let buffer = upload_block_no_copy(device, name, base as *const c_void, rounded_length)?;
    if !cached {
        request_mapping_residency(device, name, &buffer)?;
    }
    Ok(buffer)
}

/// Drops the cached no-copy buffer for a mapping `name`, taking it out of
/// the residency set first so the set never outlives the mapping.
fn evict_mapping_buffer(name: &str) {
    release_mapping_residency(name);
    NOCOPY_BUFFERS.with(|cache| {
        cache.borrow_mut().remove(name);
    });
}

pub(super) fn expert_mapping_offset(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
    tensor_name: Option<&str>,
) -> Option<Result<(MetalBuffer, usize), MetalError>> {
    expert_mapping_offset_in(
        expert_mapping_identity(),
        device,
        pointer,
        byte_length,
        tensor_name,
    )
}

/// [`expert_mapping_offset`] over an explicit `(base, length)` identity
/// instead of the process-global one, so a caller owns the mapping it asks about.
pub(super) fn expert_mapping_offset_in(
    identity: Option<(usize, usize)>,
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
    tensor_name: Option<&str>,
) -> Option<Result<(MetalBuffer, usize), MetalError>> {
    let (base, mapping_length) = identity?;
    let address = pointer as usize;
    let end = address.checked_add(byte_length)?;
    let mapping_end = base.checked_add(mapping_length)?;
    if address < base || end > mapping_end {
        return None;
    }
    if !mapping_offset_is_aligned(address - base, tensor_name) {
        return None;
    }
    record_mapping_offset_upload();
    counter!(BLOCK_OFFSET_BOUND_BYTES, byte_length as u64);
    Some(
        mapping_buffer(device, EXPERT_MAPPING_NOCOPY_NAME, base, mapping_length)
            .map(|buffer| (buffer, address - base)),
    )
}

pub(super) fn expert_mapping_identity() -> Option<(usize, usize)> {
    let mapping_length = EXPERT_MAPPING_LENGTH.load(Ordering::Acquire);
    if mapping_length == 0 {
        return None;
    }
    let base = EXPERT_MAPPING_BASE.load(Ordering::Relaxed);
    (EXPERT_MAPPING_LENGTH.load(Ordering::Acquire) == mapping_length)
        .then_some((base, mapping_length))
}

/// The single [`NOCOPY_BUFFERS`] identity [`checkpoint_mapping_offset`]
/// caches under -- sound because [`CHECKPOINT_MAPPING`] itself is a single
/// thread-local slot (never more than one registration live at a time), so
/// this name never collides across two DIFFERENT live mappings the way a
/// per-tensor or per-weight name would need to. [`register_checkpoint_mapping`]
/// drops any stale entry under this name before installing a new mapping, so
/// a re-registration is a deliberate cache invalidation, never a
/// [`MetalError::ResidentNameRebound`].
pub(super) const CHECKPOINT_MAPPING_NOCOPY_NAME: &str = "__checkpoint_mapping__";
pub(super) const EXPERT_MAPPING_NOCOPY_NAME: &str = "__expert_mapping__";

/// Test-only reset -- the default std test harness reuses threads across
/// tests in the same binary (see [`reset_nocopy_cache_for_test`]'s own
/// doc), and [`CHECKPOINT_MAPPING`] is thread-local, so a prior test's
/// registration would otherwise leak into a later test on the same thread.
/// It leaves the process-global expert mapping alone: resetting it from any
/// test would clobber the one test that registers it.
#[cfg(test)]
pub(super) fn reset_checkpoint_mapping_for_test() {
    CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow_mut() = None);
    release_all_mapping_residency();
    NOCOPY_BUFFERS.with(|cache| {
        cache.borrow_mut().remove(CHECKPOINT_MAPPING_NOCOPY_NAME);
        cache.borrow_mut().remove(EXPERT_MAPPING_NOCOPY_NAME);
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod checkpoint_mapping_release_tests {
    use proxima_tensor::AlignedBuffer;

    use super::{
        CHECKPOINT_MAPPING_NOCOPY_NAME, EXPERT_MAPPING_NOCOPY_NAME,
        MISALIGNED_MAPPING_OFFSET_COPIES, checkpoint_mapping_offset, clear_expert_source_cache,
        device_and_queue, evict_mapping_buffer, expert_mapping_offset,
        mapping_buffer_allocated_bytes, mapping_offset_is_aligned, page_size,
        register_checkpoint_mapping, register_expert_mapping, reset_checkpoint_mapping_for_test,
        unregister_checkpoint_mapping, unregister_expert_mapping, upload_packed_bytes_in,
    };
    use crate::metal::warm_resident_buffers::{
        WARM_RESIDENCY_SET, held_mapping_residency_allocations, held_residency_allocations,
    };
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSObjectProtocol;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLDevice, MTLResourceOptions,
    };
    use std::io::Write;
    use std::os::fd::AsRawFd;

    /// The exact drop-ordering hazard `LoadedModel::drop` is written
    /// against: model A loads (registers its mapping), model B loads
    /// afterward (its own `register_checkpoint_mapping` call supersedes
    /// A's, per that function's own doc), and only THEN does A's `Drop`
    /// run. A's stale `unregister_checkpoint_mapping(bytes_a)` must be a
    /// no-op -- it is no longer the current registration -- so B's mapping
    /// stays resolvable. Only unregistering the CURRENTLY registered
    /// identity actually clears it.
    #[test]
    fn unregister_only_clears_the_currently_registered_identity() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_checkpoint_mapping_for_test();

        let page = page_size();
        // page-aligned, one page each -- `newBufferWithBytesNoCopy`'s own
        // precondition (`create_no_copy_buffer`'s SAFETY doc), which
        // `checkpoint_mapping_offset` exercises via `upload_block_no_copy`.
        let model_a = AlignedBuffer::new(page / core::mem::size_of::<f32>(), page)
            .expect("page-aligned fixture for fake model A's checkpoint bytes");
        let model_b = AlignedBuffer::new(page / core::mem::size_of::<f32>(), page)
            .expect("page-aligned fixture for fake model B's checkpoint bytes");
        let model_a_bytes =
            unsafe { core::slice::from_raw_parts(model_a.as_ptr().cast::<u8>(), page) };
        let model_b_bytes =
            unsafe { core::slice::from_raw_parts(model_b.as_ptr().cast::<u8>(), page) };

        register_checkpoint_mapping(model_a_bytes);
        register_checkpoint_mapping(model_b_bytes);

        // model A's drop races behind model B's load -- releasing A's own
        // (now-stale) identity must not disturb B's live mapping.
        unregister_checkpoint_mapping(model_a_bytes);
        assert!(
            checkpoint_mapping_offset(
                &device,
                model_b_bytes.as_ptr().cast(),
                model_b_bytes.len(),
                None
            )
            .is_some(),
            "model B's own mapping must survive a stale release of model A's superseded one"
        );

        // model B's own drop releases its own, still-current identity.
        unregister_checkpoint_mapping(model_b_bytes);
        assert!(
            checkpoint_mapping_offset(
                &device,
                model_b_bytes.as_ptr().cast(),
                model_b_bytes.len(),
                None
            )
            .is_none(),
            "releasing the CURRENTLY registered identity must actually clear it"
        );
    }

    // the only test that touches the process-global expert mapping
    #[test]
    fn expert_mapping_resolves_a_misaligned_arena_slice_and_unregisters() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_checkpoint_mapping_for_test();

        let page = page_size();
        let mapping = AlignedBuffer::new(page / core::mem::size_of::<f32>(), page)
            .expect("page-aligned fixture for an expert sidecar mapping");
        let mapping_bytes =
            unsafe { core::slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), page) };
        let arena = &mapping_bytes[16..101];

        register_expert_mapping(mapping_bytes);
        let (_buffer, offset) =
            expert_mapping_offset(&device, arena.as_ptr().cast(), arena.len(), None)
                .expect("the arena lies inside the registered expert mapping")
                .expect("the whole expert mapping binds without copying");
        assert_eq!(offset, 16);
        assert_eq!(mapping_buffer_allocated_bytes(), (0, page as u64));

        clear_expert_source_cache();
        assert_eq!(mapping_buffer_allocated_bytes(), (0, 0));
        assert!(
            expert_mapping_offset(&device, arena.as_ptr().cast(), arena.len(), None).is_some(),
            "clearing staged sources must preserve the registered expert mapping"
        );

        unregister_expert_mapping(mapping_bytes);
        assert_eq!(mapping_buffer_allocated_bytes(), (0, 0));
        assert!(
            expert_mapping_offset(&device, arena.as_ptr().cast(), arena.len(), None).is_none(),
            "unregistering the current expert mapping removes the offset alias"
        );
    }

    #[test]
    fn mapping_view_offsets_are_bound_only_when_eight_byte_aligned() {
        let table = [
            (0usize, true),
            (8, true),
            (16, true),
            (16384, true),
            (1, false),
            (3, false),
            (4, false),
            (7, false),
            (3611, false),
        ];
        for (offset, aligned) in table {
            assert_eq!(
                mapping_offset_is_aligned(offset, Some("table.weight")),
                aligned,
                "offset {offset}"
            );
        }
    }

    #[test]
    fn packed_upload_at_an_odd_checkpoint_offset_takes_the_copy_path() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_checkpoint_mapping_for_test();

        let page = page_size();
        let mapping = AlignedBuffer::new(page / core::mem::size_of::<f32>(), page)
            .expect("page-aligned fixture for a checkpoint mapping");
        let mapping_bytes =
            unsafe { core::slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), page) };
        register_checkpoint_mapping(mapping_bytes);

        let copies_before = MISALIGNED_MAPPING_OFFSET_COPIES.get();
        let (_view, aligned_offset) =
            upload_packed_bytes_in(None, &device, &mapping_bytes[8..72], Some("aligned.weight"))
                .expect("aligned tensor uploads");
        assert_eq!(
            aligned_offset, 8,
            "an aligned tensor stays a view into the mapping"
        );
        assert_eq!(MISALIGNED_MAPPING_OFFSET_COPIES.get(), copies_before);

        let (_copy, odd_offset) =
            upload_packed_bytes_in(None, &device, &mapping_bytes[3..67], Some("odd.weight"))
                .expect("misaligned tensor still uploads");
        assert_eq!(
            odd_offset, 0,
            "a misaligned tensor is copied into its own buffer"
        );
        assert!(MISALIGNED_MAPPING_OFFSET_COPIES.get() > copies_before);

        unregister_checkpoint_mapping(mapping_bytes);
    }

    /// the mapping's no-copy buffer joins the residency set when the GPU
    /// first receives it, once, and leaves it when the mapping is unregistered.
    #[test]
    fn mapping_buffer_joins_the_residency_set_once_and_leaves_on_unregister() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        if !device.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
            return;
        }
        reset_checkpoint_mapping_for_test();
        WARM_RESIDENCY_SET.with(|slot| *slot.borrow_mut() = None);

        let page = page_size();
        let mapping = AlignedBuffer::new(page / core::mem::size_of::<u32>(), page)
            .expect("page-aligned fixture for a checkpoint mapping");
        let mapping_bytes =
            unsafe { core::slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), page) };
        register_checkpoint_mapping(mapping_bytes);

        upload_packed_bytes_in(None, &device, &mapping_bytes[8..72], Some("first.weight"))
            .expect("first tensor uploads as a view");
        upload_packed_bytes_in(None, &device, &mapping_bytes[72..136], Some("second.weight"))
            .expect("second tensor uploads as a view");
        assert_eq!(
            held_mapping_residency_allocations(),
            vec![(CHECKPOINT_MAPPING_NOCOPY_NAME.to_string(), 1)],
            "two views of one mapping hold one set of one allocation"
        );
        assert_eq!(
            held_residency_allocations(),
            None,
            "the warm set is not touched by a mapping"
        );

        unregister_checkpoint_mapping(mapping_bytes);
        assert_eq!(held_mapping_residency_allocations(), vec![]);
    }

    const COLD_FILE_PAGES: usize = 512;

    /// reads `length` file bytes at `offset` through the GPU: a blit from the
    /// registered mapping's no-copy view into a shared buffer.
    fn gpu_read_through_mapping(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        address: *const u8,
        length: usize,
    ) -> Vec<u8> {
        let (view, view_offset) = checkpoint_mapping_offset(device, address.cast(), length, None)
            .expect("the range lies inside the registered mapping")
            .expect("the mapping view is created");
        let destination = device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .expect("device hands out a shared destination buffer");
        let command_buffer = queue
            .commandBuffer()
            .expect("queue hands out a command buffer");
        let blit = command_buffer
            .blitCommandEncoder()
            .expect("command buffer hands out a blit encoder");
        // SAFETY: both ranges lie inside their buffers; the view offset is 8-aligned.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &view,
                view_offset,
                &destination,
                0,
                length,
            );
        }
        blit.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        // SAFETY: the buffer is shared, `length` bytes long, and the GPU is done with it.
        unsafe { core::slice::from_raw_parts(destination.contents().as_ptr().cast::<u8>(), length) }
            .to_vec()
    }

    fn non_resident_pages(address: *const u8, length: usize) -> usize {
        let pages = length / page_size();
        let mut states = vec![0 as libc::c_char; pages];
        // SAFETY: `address..address+length` is a live page-aligned mapping and
        // `states` holds one byte per page.
        let status = unsafe { libc::mincore(address.cast(), length, states.as_mut_ptr()) };
        assert_eq!(status, 0, "mincore reads the mapping's residency");
        states
            .iter()
            .filter(|state| **state & libc::MINCORE_INCORE as libc::c_char == 0)
            .count()
    }

    /// writes generated bytes to a temp file with F_NOCACHE and maps it
    /// read-only; the file is returned so the descriptor outlives the mapping.
    fn cold_file_mapping() -> (std::fs::File, *const u8, Vec<u8>) {
        let file_length = COLD_FILE_PAGES * page_size();
        let contents: Vec<u8> = (0..file_length)
            .map(|index| (index.wrapping_mul(2_654_435_761) >> 13) as u8 ^ (index >> 9) as u8)
            .collect();
        let mut file = tempfile::tempfile().expect("a temp file for the checkpoint bytes");
        // F_NOCACHE keeps the written pages out of the page cache.
        // SAFETY: the descriptor is the live temp file's.
        let cache_off = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        assert_eq!(cache_off, 0, "F_NOCACHE accepted on the temp file");
        file.write_all(&contents)
            .expect("write the checkpoint bytes");
        file.sync_all().expect("flush the checkpoint bytes to disk");
        // SAFETY: a fresh read-only shared mapping of the whole file.
        let base = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                file_length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED, "map the temp file");
        (file, base.cast::<u8>().cast_const(), contents)
    }

    /// the contract that replaced `mlock` (commit 0a6ba7df23, ROW 533): a
    /// no-copy view over pages the kernel never loaded reads the file's bytes.
    #[test]
    fn a_gpu_read_through_a_cold_checkpoint_mapping_returns_the_file_bytes() {
        let Ok((device, queue)) = device_and_queue() else {
            return;
        };
        reset_checkpoint_mapping_for_test();

        let page = page_size();
        let file_length = COLD_FILE_PAGES * page;
        let (_file, base, contents) = cold_file_mapping();
        // SAFETY: the mapping is `file_length` bytes and outlives every use below.
        let mapped = unsafe { core::slice::from_raw_parts(base, file_length) };

        let offset = 64 * page + 64;
        let length = 16 * page;
        // SAFETY: `offset` is inside the mapping.
        let read_start = unsafe { base.add(offset) };
        // SAFETY: the page-aligned range around the read lies inside the mapping.
        let cold_in_read = non_resident_pages(unsafe { base.add(64 * page) }, 17 * page);
        eprintln!("cold probe: non-resident pages before the read: {cold_in_read} of 17");
        assert!(
            cold_in_read > 0,
            "the read range must hold non-resident pages or this test proves nothing \
             (non-resident pages in range: {cold_in_read} of 17)"
        );
        register_checkpoint_mapping(mapped);

        let read = gpu_read_through_mapping(&device, &queue, read_start, length);

        unregister_checkpoint_mapping(mapped);
        // SAFETY: the mapping was created above and nothing references it now.
        unsafe { libc::munmap(base.cast_mut().cast(), file_length) };
        assert_eq!(
            read,
            &contents[offset..offset + length],
            "the GPU read returns the file's bytes (non-resident pages before the read: {cold_in_read} of 17)"
        );
    }

    #[test]
    fn packed_upload_prefers_the_registered_expert_mapping() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_checkpoint_mapping_for_test();

        let page = page_size();
        let mapping = AlignedBuffer::new(2 * page / core::mem::size_of::<f32>(), page)
            .expect("page-aligned fixture for packed expert uploads");
        let mapping_bytes =
            unsafe { core::slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), 2 * page) };
        let first_page = &mapping_bytes[..page];
        let second_page = &mapping_bytes[page..];

        let identity = Some((mapping_bytes.as_ptr() as usize, mapping_bytes.len()));
        let (first_buffer, first_offset) =
            upload_packed_bytes_in(identity, &device, first_page, None)
                .expect("first page uses the owned expert mapping");
        let (second_buffer, second_offset) =
            upload_packed_bytes_in(identity, &device, second_page, None)
                .expect("second page uses the owned expert mapping");

        assert_eq!(first_offset, 0);
        assert_eq!(second_offset, page);
        assert_eq!(
            objc2::rc::Retained::as_ptr(&first_buffer),
            objc2::rc::Retained::as_ptr(&second_buffer)
        );
        assert_eq!(mapping_buffer_allocated_bytes(), (0, (2 * page) as u64));

        evict_mapping_buffer(EXPERT_MAPPING_NOCOPY_NAME);
        assert_eq!(mapping_buffer_allocated_bytes(), (0, 0));
    }
}

thread_local! {
    /// No-copy block buffers, keyed by the exact host range they wrap.
    ///
    /// `newBufferWithBytesNoCopy` does not copy, but it is NOT free: every
    /// call creates a fresh `MTLBuffer` and Metal has to wire those pages
    /// for GPU access. `execute` rebuilt every block buffer on every call,
    /// so a serving loop re-wired the entire weight set per token — a cost
    /// that scales with BYTES, which is exactly what made it invisible in a
    /// bytes-normalized probe.
    ///
    /// Keyed on the resident NAME [`Plan::mark_resident`] proved static for
    /// this node, never on `(pointer, byte_length)` alone -- ROW 334's own
    /// shape: an address-only key let a freed arm's `Vec<u8>` and a LATER,
    /// unrelated arm's same-byte-size `Vec<u8>` collide at the identical
    /// address, serving the later arm 100% stale bytes. `upload_block_no_copy`
    /// checks the offered pointer and byte length against what this name was
    /// first uploaded with on every lookup and refuses to serve a mismatch
    /// (see that function's own doc) -- the same contract
    /// [`upload_resident_copy`]/[`RESIDENT_BUFFERS`] enforce for the copy
    /// path, applied here to the no-copy path. See
    /// `proxima-tensor/docs/discipline.md` ROW 70/334.
    ///
    /// `upload_block_as_float` and `upload_packed_bytes` only route into this
    /// cache when [`Plan::mark_resident`] already proved a name for the
    /// node's own address -- true for mmap'd GGUF weights, false for an
    /// ephemeral, growing buffer (a KV-cache row, say) whose page-aligned
    /// address is a coincidence of a page-boundary crossing, not a lifetime
    /// proof. A page-aligned but non-resident (unnamed) block takes
    /// `upload_block_no_copy_uncached` instead: still zero-copy for this one
    /// `execute` call (sound for the same `waitUntilCompleted` reason), just
    /// never remembered past it, and never inserted here.
    ///
    /// Reuse is otherwise safe on the data-freshness axis precisely BECAUSE
    /// it is no-copy: writes through the caller's own slice are visible to
    /// the GPU, so a wrapper never goes stale. Copying uploads are
    /// deliberately NOT cached — those snapshot the data, and reuse would
    /// serve a stale snapshot.
    pub(super) static NOCOPY_BUFFERS: RefCell<BTreeMap<String, (usize, usize, MetalBuffer)>> =
        RefCell::new(BTreeMap::new());
}
