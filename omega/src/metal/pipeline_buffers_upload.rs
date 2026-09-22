use super::*;

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
    const fn as_mtl(self) -> MTLMathMode {
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

    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| MetalError::CompileFailed {
            log: nserror_description(&error),
        })
}

/// Resolves `bound`'s compiled pipeline against `cache_key`
/// ([`kernel_cache_key`]) rather than [`Kernel::source`] — a hit never builds
/// the MSL source text at all, only a genuine miss calls [`emit`] to render
/// it and compile. See this module's `ROW 92`/`ROW 93` discipline-log entries
/// for the measured cost `emit` paid on every call, hit or miss, before this
/// split existed.
pub(super) fn pipeline_for(
    device: &ProtocolObject<dyn MTLDevice>,
    bound: &BoundOp,
    packed_operands: &PackedOperands,
    cache_key: &str,
    math_mode: MathMode,
    numeric_policy: NumericPolicy,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, MetalError> {
    // `cache_key` already carries BOTH axes `compile_pipeline` reads: the
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
    if let Some(pipeline) = PIPELINE_CACHE.with(|cache| cache.borrow().get(cache_key).cloned()) {
        trace!(cache_key = %cache_key, hit = true, "pipeline cache lookup");
        #[cfg(feature = "instrument")]
        counter!(PIPELINE_HITS, 1);
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
    let kernel = emit(bound, packed_operands, numeric_policy)?;
    if std::env::var_os("PROXIMA_DEBUG_METAL_SOURCE").is_some()
        && std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE").ok() == Some(bound.node.0.to_string())
    {
        eprintln!(
            "metal_bound_source node={:?}\n{}",
            bound.node, kernel.source
        );
    }
    let pipeline = compile_pipeline(device, &kernel, math_mode)?;
    #[cfg(feature = "instrument")]
    {
        counter!(PIPELINE_MISSES, 1);
        counter!(PIPELINE_COMPILE_TICKS, elapsed_ticks(compile_started));
    }
    PIPELINE_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .insert(cache_key.to_string(), pipeline.clone());
    });
    Ok(pipeline)
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
    if let Some(pipeline) = PIPELINE_CACHE.with(|cache| cache.borrow().get(cache_key).cloned()) {
        return Ok(pipeline);
    }
    let pipeline = compile_pipeline(device, kernel, math_mode)?;
    PIPELINE_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .insert(cache_key.to_string(), pipeline.clone());
    });
    Ok(pipeline)
}

#[cfg(not(feature = "metal-buffer-pool"))]
pub(super) fn allocate_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    element_count: usize,
    dtype: DType,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_length = element_count.max(1) * dtype.size_bytes();
    counter!(OUTPUT_BUFFER_ALLOCATIONS, 1);
    counter!(OUTPUT_BUFFER_ALLOCATED_BYTES, byte_length as u64);
    device
        .newBufferWithLength_options(byte_length, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a shared buffer".to_string(),
        })
}

/// Pool-backed counterpart of the `metal-buffer-pool`-off `allocate_buffer`
/// above -- identical contract (a buffer sized to hold `element_count`
/// `dtype` elements), but the ACTUAL Metal allocation, on both the pool-hit
/// and pool-miss paths, is always sized to `pool_bucket(byte_length)`, never
/// to the tight `byte_length`. See `pool_bucket`'s doc for why that is what
/// makes a bucketed pop sound, and `OUTPUT_BUFFER_POOL`'s doc for the
/// invariant this maintains across the pool's whole lifetime.
#[cfg(feature = "metal-buffer-pool")]
pub(super) fn allocate_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    element_count: usize,
    dtype: DType,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_length = element_count.max(1) * dtype.size_bytes();
    let bucket = pool_bucket(byte_length);
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
    device
        .newBufferWithLength_options(bucket, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate a shared buffer".to_string(),
        })
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
    /// output size tracks it (`bound_output_len` scales with the KV extent)
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
    static OUTPUT_BUFFER_POOL: RefCell<HashMap<(usize, DType), Vec<MetalBuffer>>> =
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
    /// snapshot after a run sees them printed in `{:?}`, not silently zeroed
    /// out of the struct's `Debug` output.
    #[test]
    fn new_host_bookkeeping_fields_print_in_stage_totals() {
        counter!(RETIRE_SCAN_CALLS, 3);
        counter!(RETIRE_SCAN_TICKS, 7);
        counter!(EXPERT_BUFFERS_LOOKUP_CALLS, 1);
        counter!(EXPERT_BUFFERS_LOOKUP_TICKS, 2);

        let totals = metal_stage_totals();
        let rendered = format!("{totals:?}");

        assert!(rendered.contains("retire_scan_calls: 3"), "{rendered}");
        assert!(rendered.contains("retire_scan_ticks: 7"), "{rendered}");
        assert!(
            rendered.contains("expert_buffers_lookup_calls: 1"),
            "{rendered}"
        );
        assert!(
            rendered.contains("expert_buffers_lookup_ticks: 2"),
            "{rendered}"
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
    if std::env::var_os("PROXIMA_DEBUG_BLOCK_UPLOADS").is_some() && !data.is_empty() {
        eprintln!(
            "metal block upload node={node:?} name={block_name:?} dtype={dtype:?} bytes={} resident={resident_name:?}",
            size_of_val(data),
        );
    }
    match dtype {
        // unreached by every program this driver compiles today (none
        // declares a `Float16` block input -- `proxima-tensor/src/spec.rs`'s
        // `mistral_cached_forward_program` is `Float32` throughout), so it is
        // not worth the residency cache's extra bookkeeping: the narrowed
        // `Vec<f16>` this allocates is dropped every call regardless.
        DType::Float16 => upload_block_as_half(device, data).map(|buffer| (buffer, 0)),
        DType::Float32
        | DType::BFloat16
        | DType::Bool
        | DType::Int8
        | DType::UInt8
        | DType::Int32
        | DType::UInt32 => upload_block_as_float(device, data, resident_name),
        DType::Int16
        | DType::UInt16
        | DType::Int64
        | DType::UInt64
        | DType::Int128
        | DType::UInt128
        | DType::Float64 => Err(EmitError::UnsupportedDType { node, dtype }.into()),
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
        return allocate_buffer(device, 0, DType::Float32).map(|buffer| (buffer, 0));
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
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length) {
        return result;
    }
    if let Some(result) = expert_mapping_offset(device, pointer, byte_length) {
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
    counter!(BLOCK_COPIED_BYTES, byte_length as u64);
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
    if bytes.len() > 100 * 1024 * 1024 && std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some()
    {
        eprintln!(
            "metal large packed upload bytes={} resident={resident_name:?}",
            bytes.len()
        );
    }
    if bytes.is_empty() {
        return allocate_buffer(device, 0, DType::Float32).map(|buffer| (buffer, 0));
    }
    let byte_length = bytes.len();
    let pointer = bytes.as_ptr().cast::<c_void>();
    let mapping_published = expert_mapping_identity().is_some();
    if mapping_published {
        counter!(EXPERT_MAPPING_CANDIDATE_UPLOADS, 1);
    }
    if let Some(result) = expert_mapping_offset(device, pointer, byte_length) {
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
    if let Some(result) = checkpoint_mapping_offset(device, pointer, byte_length) {
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
    counter!(BLOCK_COPIED_BYTES, byte_length as u64);
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
        NOCOPY_BUFFERS.with(|cache| {
            cache.borrow_mut().remove(EXPERT_MAPPING_NOCOPY_NAME);
        });
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
        NOCOPY_BUFFERS.with(|cache| {
            cache.borrow_mut().remove(EXPERT_MAPPING_NOCOPY_NAME);
        });
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
    NOCOPY_BUFFERS.with(|cache| {
        cache.borrow_mut().remove(CHECKPOINT_MAPPING_NOCOPY_NAME);
    });
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
        NOCOPY_BUFFERS.with(|cache| {
            cache.borrow_mut().remove(CHECKPOINT_MAPPING_NOCOPY_NAME);
        });
    }
}

/// Counts uploads served by addressing the shared checkpoint-mapping buffer
/// at an offset, instead of copying the tensor into its own buffer -- the
/// direct witness for the census this mechanism is meant to zero out.
pub static MAPPING_OFFSET_UPLOADS: Counter =
    Counter::new("omega.metal.upload_block.mapping_offset");
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
) -> Option<Result<(MetalBuffer, usize), MetalError>> {
    let (base, mapping_length) = CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow())?;
    let address = pointer as usize;
    if address < base || address + byte_length > base + mapping_length {
        return None;
    }
    let page = page_size();
    let rounded_length = mapping_length.div_ceil(page) * page;
    counter!(MAPPING_OFFSET_UPLOADS, 1);
    counter!(BLOCK_OFFSET_BOUND_BYTES, byte_length as u64);
    Some(
        upload_block_no_copy(
            device,
            CHECKPOINT_MAPPING_NOCOPY_NAME,
            base as *const c_void,
            rounded_length,
        )
        .map(|buffer| (buffer, address - base)),
    )
}

pub(super) fn expert_mapping_offset(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Option<Result<(MetalBuffer, usize), MetalError>> {
    let (base, mapping_length) = expert_mapping_identity()?;
    let address = pointer as usize;
    let end = address.checked_add(byte_length)?;
    let mapping_end = base.checked_add(mapping_length)?;
    if address < base || end > mapping_end {
        return None;
    }
    let rounded_length = mapping_length.div_ceil(page_size()) * page_size();
    counter!(MAPPING_OFFSET_UPLOADS, 1);
    counter!(BLOCK_OFFSET_BOUND_BYTES, byte_length as u64);
    Some(
        upload_block_no_copy(
            device,
            EXPERT_MAPPING_NOCOPY_NAME,
            base as *const c_void,
            rounded_length,
        )
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
#[cfg(test)]
pub(super) fn reset_checkpoint_mapping_for_test() {
    CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow_mut() = None);
    EXPERT_MAPPING_LENGTH.store(0, Ordering::Release);
    EXPERT_MAPPING_BASE.store(0, Ordering::Relaxed);
    NOCOPY_BUFFERS.with(|cache| {
        cache.borrow_mut().remove(CHECKPOINT_MAPPING_NOCOPY_NAME);
        cache.borrow_mut().remove(EXPERT_MAPPING_NOCOPY_NAME);
    });
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod checkpoint_mapping_release_tests {
    use proxima_tensor::AlignedBuffer;
    use std::sync::Mutex;

    use super::{
        checkpoint_mapping_offset, clear_expert_source_cache, device_and_queue,
        expert_mapping_offset, mapping_buffer_allocated_bytes, page_size,
        register_checkpoint_mapping, register_expert_mapping, reset_checkpoint_mapping_for_test,
        unregister_checkpoint_mapping, unregister_expert_mapping, upload_packed_bytes,
    };

    static MAPPING_TEST_LOCK: Mutex<()> = Mutex::new(());

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
        let _guard = MAPPING_TEST_LOCK
            .lock()
            .expect("mapping test lock is healthy");
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
            checkpoint_mapping_offset(&device, model_b_bytes.as_ptr().cast(), model_b_bytes.len())
                .is_some(),
            "model B's own mapping must survive a stale release of model A's superseded one"
        );

        // model B's own drop releases its own, still-current identity.
        unregister_checkpoint_mapping(model_b_bytes);
        assert!(
            checkpoint_mapping_offset(&device, model_b_bytes.as_ptr().cast(), model_b_bytes.len())
                .is_none(),
            "releasing the CURRENTLY registered identity must actually clear it"
        );
    }

    #[test]
    fn expert_mapping_resolves_a_misaligned_arena_slice_and_unregisters() {
        let _guard = MAPPING_TEST_LOCK
            .lock()
            .expect("mapping test lock is healthy");
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_checkpoint_mapping_for_test();

        let page = page_size();
        let mapping = AlignedBuffer::new(page / core::mem::size_of::<f32>(), page)
            .expect("page-aligned fixture for an expert sidecar mapping");
        let mapping_bytes =
            unsafe { core::slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), page) };
        let arena = &mapping_bytes[17..101];

        register_expert_mapping(mapping_bytes);
        let (_buffer, offset) = expert_mapping_offset(&device, arena.as_ptr().cast(), arena.len())
            .expect("the arena lies inside the registered expert mapping")
            .expect("the whole expert mapping binds without copying");
        assert_eq!(offset, 17);
        assert_eq!(mapping_buffer_allocated_bytes(), (0, page as u64));

        clear_expert_source_cache();
        assert_eq!(mapping_buffer_allocated_bytes(), (0, 0));
        assert!(
            expert_mapping_offset(&device, arena.as_ptr().cast(), arena.len()).is_some(),
            "clearing staged sources must preserve the registered expert mapping"
        );

        unregister_expert_mapping(mapping_bytes);
        assert_eq!(mapping_buffer_allocated_bytes(), (0, 0));
        assert!(
            expert_mapping_offset(&device, arena.as_ptr().cast(), arena.len()).is_none(),
            "unregistering the current expert mapping removes the offset alias"
        );
    }

    #[test]
    fn packed_upload_prefers_the_registered_expert_mapping() {
        let _guard = MAPPING_TEST_LOCK
            .lock()
            .expect("mapping test lock is healthy");
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

        register_expert_mapping(mapping_bytes);
        let (first_buffer, first_offset) = upload_packed_bytes(&device, first_page, None)
            .expect("first page uses the registered expert mapping");
        let (second_buffer, second_offset) = upload_packed_bytes(&device, second_page, None)
            .expect("second page uses the registered expert mapping");

        assert_eq!(first_offset, 0);
        assert_eq!(second_offset, page);
        assert_eq!(
            objc2::rc::Retained::as_ptr(&first_buffer),
            objc2::rc::Retained::as_ptr(&second_buffer)
        );
        assert_eq!(mapping_buffer_allocated_bytes(), (0, (2 * page) as u64));

        unregister_expert_mapping(mapping_bytes);
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

