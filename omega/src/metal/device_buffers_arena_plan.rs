use super::*;

/// A live Metal buffer handle — the shape every device-buffer table and
/// return value in this file traffics in.
pub(super) type MetalBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
/// A device buffer plus this node's byte OFFSET into it -- most nodes own
/// their whole buffer (offset 0), but a tensor served by
/// [`checkpoint_mapping_offset`] shares one buffer across many nodes, each
/// at its own offset. Carrying the pair through `device_buffers` is what
/// lets [`bind_buffers`] bind the right slice with `setBuffer:offset:atIndex:`
/// instead of every binding assuming offset 0.
pub(super) type DeviceBuffer = (MetalBuffer, usize);

/// Owns a compute encoder's `endEncoding()` call so an early `?` return from
/// inside an op-encoding loop cannot leave it un-ended. Every `execute*`
/// entry point in this file used to open one encoder and call
/// `endEncoding()` exactly once, textually after its per-op loop (see the
/// module doc's "Execution model") — correct on the happy path, but any
/// error propagated with `?` from inside that loop skipped the call, and the
/// `Retained` handle's `dealloc` at autorelease-pool drain hit Metal's own
/// `-[_MTLCommandEncoder dealloc]: failed assertion 'Command encoder
/// released without endEncoding'`, a hard `SIGTRAP` that also swallowed the
/// real Rust `Err` that triggered it.
///
/// `Deref`s to the encoder so every existing call site (`&encoder`,
/// `encoder.memoryBarrierWithScope(..)`, passing it into [`encode_op`])
/// compiles unchanged. Call [`EncoderGuard::finish`] on the explicit success
/// path; anywhere else (including every `?`), [`Drop::drop`] ends it exactly
/// once instead.
pub(super) struct EncoderGuard {
    pub(super) encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    pub(super) finished: bool,
}

#[cfg(all(test, feature = "instrument"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod bounded_compare_tests {
    use super::{MetalError, compare_bound_f32};
    use proxima_tensor::op::NodeId;

    #[test]
    fn finite_equal_is_not_a_mismatch() {
        assert!(
            compare_bound_f32(NodeId(1), &[1.0, -2.0], &[1.0, -2.0])
                .expect("equal lengths")
                .is_none()
        );
    }

    #[test]
    fn finite_difference_reports_first_element_and_maximum_relative_error() {
        let mismatch = compare_bound_f32(NodeId(1), &[2.0, 4.0], &[1.0, 2.0])
            .expect("equal lengths")
            .expect("difference");
        assert_eq!(mismatch.0, 0);
        assert_eq!(mismatch.1, 2.0);
        assert_eq!(mismatch.2, 1.0);
        assert_eq!(mismatch.3, 1.0);
        assert_eq!(mismatch.4, 1.0);
    }

    #[test]
    fn nonfinite_difference_cannot_evade_tolerance() {
        let mismatch = compare_bound_f32(NodeId(1), &[f32::NAN], &[1.0])
            .expect("equal lengths")
            .expect("nonfinite difference");
        assert!(mismatch.3.is_infinite());
    }

    #[test]
    fn cpu_nonfinite_difference_is_reported() {
        let mismatch = compare_bound_f32(NodeId(1), &[1.0], &[f32::NAN])
            .expect("equal lengths")
            .expect("nonfinite difference");
        assert!(mismatch.3.is_infinite());
    }

    #[test]
    fn length_mismatch_is_typed() {
        let error = compare_bound_f32(NodeId(9), &[1.0], &[]).expect_err("length mismatch");
        assert!(matches!(
            error,
            MetalError::CpuBoundComparisonLengthMismatch {
                node: NodeId(9),
                metal_len: 1,
                cpu_len: 0,
            }
        ));
    }
}

impl EncoderGuard {
    pub(super) fn new(encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) -> Self {
        Self {
            encoder,
            finished: false,
        }
    }

    /// The success-path close: ends encoding now rather than at `Drop`, so
    /// intent at the call site reads the same as the un-guarded code it
    /// replaces (`encoder.endEncoding()` becomes `encoder.finish()`).
    pub(super) fn finish(mut self) {
        self.encoder.endEncoding();
        self.finished = true;
    }

    /// A cheap `Retained` clone of the underlying encoder for callers that
    /// need to alias it (e.g. hand it to [`encode_op`] by value) without
    /// taking over its `endEncoding()` obligation — only this guard's own
    /// `finish`/`Drop` ever closes the encoder. Only
    /// [`execute_plan_with_placements_dispatch_timed`]'s shared/stage-encoder
    /// reuse needs an alias rather than sole ownership, so this is gated
    /// identically to that function rather than left dead under every other
    /// feature combination.
    #[cfg(all(feature = "metal-output-placement", feature = "instrument"))]
    pub(super) fn clone_inner(&self) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        self.encoder.clone()
    }
}

impl Deref for EncoderGuard {
    type Target = ProtocolObject<dyn MTLComputeCommandEncoder>;

    fn deref(&self) -> &Self::Target {
        &self.encoder
    }
}

impl Drop for EncoderGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.encoder.endEncoding();
        }
    }
}

/// One gathering op's deferred fault check: the op it came from, its fault
/// buffer, and how many gather slots that buffer holds. [`encode_op`]
/// produces these; [`execute`] checks them all after its single
/// end-of-program wait (see the module doc's "Gather fault reporting").
pub(super) type PendingFault<'a> = (&'a BoundOp, MetalBuffer, usize);

/// Everything [`execute`] can fail with: a missing device, any device
/// operation that returned a Metal-side failure (compiling source, creating
/// a pipeline, or one of the handful of `Option`-returning calls that are
/// only ever `None` on a broken host), or a `BoundOp`/program-shaped fault
/// [`proxima_tensor`] or [`crate::msl`] already have a name for.
#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device available on this host")]
    NoDevice,
    #[error("metal driver error: {log}")]
    CompileFailed { log: String },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Emit(#[from] EmitError),
    #[error("metal expert source table for node {node} is not a uniform packed codec: {reason}")]
    ExpertSourceUnsupported { node: NodeId, reason: &'static str },
    #[error("metal expert source for node {node} is not resident for routed expert {expert}")]
    ExpertSourceMiss { node: NodeId, expert: u32 },
    #[error("checkpoint mmap page discard failed with errno {errno}")]
    CheckpointMmapDiscardFailed { errno: i32 },
    #[error("hazard tracking: operand {node} has no resolved device buffer")]
    UnresolvedHazardOperand { node: NodeId },
    /// `build_buffer_arena`'s own reuse pass still needed more transient
    /// bytes live at once than `cap_bytes` budgets -- MG-3's kill condition,
    /// now a typed error a caller can act on rather than a stderr line
    /// beside a silently returned `Ok`. `cap_bytes` is
    /// `ARENA_TRANSIENT_CAP * max(query_rows, 1)` (see
    /// `plan_query_rows`'s own doc for why the cap scales this way, not a
    /// fixed decode-shaped constant): a prefill's transient outputs grow
    /// linearly with its own row count, so a per-call budget that ignores
    /// row count rejects a prefill that fits the device just as readily as
    /// one that would not. `device_limit` (`MTLDevice::
    /// recommendedMaxWorkingSetSize`) is carried for diagnosis only -- it is
    /// not part of the accept/reject decision here.
    #[error(
        "arena peak_bytes={peak_bytes} exceeds arena_transient_cap={cap_bytes} \
         at query_rows={query_rows} (device_limit={device_limit})"
    )]
    ArenaOverCap {
        peak_bytes: usize,
        cap_bytes: usize,
        query_rows: u64,
        device_limit: u64,
    },
    /// `PROXIMA_METAL_KIND_FILTER`'s `kind:` term named a string
    /// `classify_kind` never returns -- ROW 308 found a stale or
    /// misspelled substring silently dropped the WHOLE plan instead of
    /// isolating one kind (`KindFilter::matches` returns `false` for every
    /// op, so `ablation_skip` becomes `true` for everything). This is now a
    /// typed error at the same point that env var is first consulted,
    /// rather than a silently degenerate ablation run.
    #[error("kind filter term {term:?} matches no classify_kind arm")]
    UnknownKindFilterTerm { term: String },
    /// A `PROXIMA_METAL_KIND_FILTER` value that, applied against THIS plan's
    /// own dispatch sequence, removes zero ops -- the same silent-degenerate
    /// failure [`UnknownKindFilterTerm`](Self::UnknownKindFilterTerm) covers
    /// for a `kind:` term, extended to `family:` terms (a family name is
    /// data-dependent on the loaded checkpoint, so it cannot be validated
    /// against a static list the way a `kind:` term can -- a typo there
    /// only ever shows up as "removed nothing").
    #[error("kind filter {filter:?} matched zero dispatches in this plan")]
    KindFilterMatchesNothing { filter: String },
    /// `upload_resident_copy`'s own contract, made typed rather than
    /// silently violated: [`Plan::mark_resident`]'s doc says a resident
    /// NAME's host buffer is "bound once at load and never mutated again",
    /// so a legitimate caller only ever offers the same host pointer and
    /// byte length under a given name. A hit under `name` whose offered
    /// host pointer or byte length differs from what was cached is a
    /// caller bug (ROW 331/332's shape, reproduced at the driver level): a
    /// second, unrelated host allocation reused a name a serving loop had
    /// already marked resident.
    #[error(
        "resident name {name:?} rebound to a different host buffer (cached len={cached_len}, offered len={offered_len})"
    )]
    ResidentNameRebound {
        name: String,
        cached_len: usize,
        offered_len: usize,
    },
    /// [`Plan::check_numeric_policy`]/[`Plan::set_math_mode`]: `bound` is the
    /// [`NumericPolicy`] this plan's program was actually compiled under
    /// ([`plan`]/[`plan_named`]'s own argument), `requested` is what the
    /// caller asked to confirm or narrow into. `Plan` never retains `blocks`
    /// (see [`Plan::numeric_policy`]'s own doc), so a rebind is not
    /// implementable in place -- the caller's only correct response is a
    /// fresh [`plan`]/[`plan_named`] call with `requested`.
    /// `PROXIMA_METAL_NAN_CHECK`'s own stop condition -- [`check_op_output_finite`]
    /// found this op's own output buffer holds a NaN/Inf, so
    /// [`execute_op_timed`] stops the step here rather than running every
    /// remaining op past a value already known bad. Diagnostic-only, same
    /// `instrument`-gated reachability as [`check_op_output_finite`] itself.
    #[cfg(feature = "instrument")]
    #[error("nan_check: op node={node} kind={kind} produced a non-finite output")]
    NonFiniteOpOutput { node: NodeId, kind: String },
    /// `PROXIMA_METAL_COMPARE_CPU`'s own stop condition -- [`compare_op_output_to_cpu`]
    /// found this op's own Metal output disagrees with the same node's CPU
    /// reference value beyond the 1e-2 relative-diff floor, so
    /// [`execute_op_timed`] stops the step at the FIRST divergent node
    /// rather than running every remaining op past a value already known
    /// wrong. Diagnostic-only, same `instrument`-gated reachability as
    /// [`check_op_output_finite`].
    #[cfg(feature = "instrument")]
    #[error("cpu_compare: op node={node} kind={kind} max_rel_diff={max_rel_diff} exceeds 1e-2")]
    CpuMetalDivergence {
        node: NodeId,
        kind: String,
        max_rel_diff: f32,
    },
    #[cfg(feature = "instrument")]
    #[error("cpu_compare: bound node {node} needs {bytes} snapshot bytes, above cap {cap_bytes}")]
    CpuBoundComparisonOverCap {
        node: NodeId,
        bytes: usize,
        cap_bytes: usize,
    },
    #[cfg(feature = "instrument")]
    #[error(
        "cpu_compare: bound node {node} output length differs (metal={metal_len}, cpu={cpu_len})"
    )]
    CpuBoundComparisonLengthMismatch {
        node: NodeId,
        metal_len: usize,
        cpu_len: usize,
    },
    #[cfg(feature = "instrument")]
    #[error("cpu_compare: selected bound node {node} is not present in this resolved plan")]
    CpuBoundComparisonNodeNotFound { node: NodeId },
    #[error(
        "plan bound under {bound:?}, caller requested {requested:?} -- rebuild via plan()/plan_named()"
    )]
    NumericPolicyMismatch {
        bound: NumericPolicy,
        requested: NumericPolicy,
    },
}

/// The plan-time description of one expert payload in an [`ExpertSource`].
///
/// The descriptor is deliberately independent of a Metal buffer: HOBBIT can
/// keep the payload in an mmap and a later lowering can bind one buffer per
/// codec run without changing the residency FSM or the tensor graph.  The
/// byte range is relative to the codec run's source, not to the checkpoint's
/// global mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertPayloadDescriptor {
    /// Original routed expert index. The descriptor table stays sparse in
    /// payload bytes but dense in route space so the kernel can retain O(1)
    /// `descriptor[expert]` addressing.
    pub expert_index: u32,
    pub codec: Codec,
    pub byte_offset: usize,
    pub byte_length: usize,
    pub out_dim: u32,
    pub in_dim: u32,
    pub epoch: u64,
}

/// The two device buffers a codec-tagged expert kernel needs for one source
/// node. They stay owned by the execution call through command completion;
/// the table decides bytes between steps, never while a kernel borrows them.
#[derive(Clone)]
pub(super) struct ExpertSourceBuffers {
    pub(super) node: NodeId,
    pub(super) payloads: MetalBuffer,
    pub(super) payload_offset: usize,
    pub(super) descriptors: MetalBuffer,
    pub(super) descriptor_records: Vec<ExpertPayloadDescriptor>,
}

/// Host staging that keeps a no-copy upload's borrowed bytes alive through
/// the command-buffer wait. A contiguous expert arena can later replace this
/// staging without changing the descriptor ABI or the emitted kernel.
pub(super) struct StagedExpertSource {
    pub(super) _payload_bytes: Option<Vec<u8>>,
    pub(super) _descriptor_bytes: Option<Vec<u8>>,
    pub(super) payload_alias_address: usize,
    pub(super) descriptor_alias_address: usize,
    pub(super) buffers: ExpertSourceBuffers,
}

/// This thread's Metal device paired with its command queue — both created
/// once per thread rather than per [`execute`] call.
pub(super) type DeviceAndQueue = (
    Retained<ProtocolObject<dyn MTLDevice>>,
    Retained<ProtocolObject<dyn MTLCommandQueue>>,
);

thread_local! {
    /// Compiled pipelines, keyed by [`kernel_cache_key`]'s cheap structural
    /// fingerprint (not [`Kernel::source`] — deriving that string is exactly
    /// the cost this cache exists to avoid on a hit), for the lifetime of
    /// the thread rather than of one [`execute`] call.
    ///
    /// This was per-call, which meant EVERY `execute` compiled every kernel
    /// from MSL source before dispatching it. A serving loop runs the same
    /// graph thousands of times, so that is thousands of redundant
    /// compiles — measured at 3.2 ms for a 2.36 MB matvec and 8.3 ms for a
    /// 9.44 MB one, against llama.cpp's 17.62 ms for an entire 7B token.
    /// `thread_local` rather than a process-wide `OnceLock`: `Retained<_>` of
    /// an `objc2` protocol object is not `Send`/`Sync`, and a per-thread
    /// cache needs no lock on the dispatch path anyway.
    pub(super) static PIPELINE_CACHE: RefCell<BTreeMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>> =
        RefCell::new(BTreeMap::new());

    /// Emitted mixed-expert kernels persist beside their compiled pipeline.
    /// The source body depends only on the bound shape, codec policy, math
    /// mode, and source node; the resident payload bytes are bound separately
    /// per step, so re-rendering MSL on every routed gather is unnecessary.
    pub(super) static MIXED_KERNEL_CACHE: RefCell<BTreeMap<String, crate::msl::Kernel>> =
        const { RefCell::new(BTreeMap::new()) };

    /// Staged HOBBIT payloads persist across token steps on this Metal
    /// execution thread. Segment programs intentionally reuse local `NodeId`s
    /// across layers, so the plan address is part of the key; otherwise one
    /// layer evicts another layer's staged table every token.
    pub(super) static EXPERT_SOURCE_CACHE:
        RefCell<BTreeMap<(usize, NodeId), (u64, StagedExpertSource)>> =
        const { RefCell::new(BTreeMap::new()) };

    /// The device and its command queue, created once per thread. Both were
    /// also per-call; `MTLCreateSystemDefaultDevice` plus `newCommandQueue`
    /// is not free, and nothing about either depends on the program being
    /// run.
    static DEVICE_AND_QUEUE: RefCell<Option<DeviceAndQueue>> = const { RefCell::new(None) };

}

/// This thread's `MTLDevice::currentAllocatedSize` -- Metal's own count of
/// bytes it has allocated for every buffer, texture and heap this device
/// owns, read directly rather than summed from this driver's own caches, so
/// it also catches anything the caches under- or over-count. `None` before
/// any Metal call on this thread has created a device.
#[must_use]
pub fn current_allocated_size() -> Option<u64> {
    let (device, _queue) = device_and_queue().ok()?;
    Some(device.currentAllocatedSize() as u64)
}

/// Drops staged HOBBIT payload buffers after an all-expert prefill has handed
/// routing observations to the next segmented step. The command buffer has
/// completed before the evaluator returns, so releasing these wrappers cannot
/// invalidate an in-flight kernel; keeping them would retain the entire
/// low-codec expert table while the bounded DynaExq table takes over.
pub fn clear_expert_source_cache() {
    EXPERT_SOURCE_CACHE.with(|cache| cache.borrow_mut().clear());
    NOCOPY_BUFFERS.with(|cache| {
        cache.borrow_mut().remove(EXPERT_MAPPING_NOCOPY_NAME);
    });
}

/// A plain record of the facts a load-time budget decision needs about this
/// host and its Metal device -- no policy, no threshold, just what the
/// device and the OS report right now (guiding-principles principle 1:
/// this is payload, not a new abstraction; the fit decision itself lives in
/// `proxima-model-interop`, which reads these fields). Every field is a
/// direct pass-through of one `MTLDevice` accessor or one `sysctlbyname`
/// call -- see [`system_memory_facts`]'s own doc for which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemMemoryFacts {
    /// `MTLDevice::recommendedMaxWorkingSetSize` -- the device's own advice
    /// for how many bytes of GPU-visible memory a well-behaved app should
    /// keep resident at once, in bytes.
    pub recommended_max_working_set_size: u64,
    /// `MTLDevice::maxBufferLength` -- the largest single `MTLBuffer` this
    /// device will create, in bytes.
    pub max_buffer_length: u64,
    /// `MTLDevice::hasUnifiedMemory` -- `true` on every Apple silicon Mac
    /// (the device and the host share one physical memory pool), `false` on
    /// a discrete GPU with its own VRAM.
    pub has_unified_memory: bool,
    /// `MTLDevice::currentAllocatedSize` at probe time -- bytes this device
    /// has already allocated for buffers/textures/heaps, before this load
    /// adds anything.
    pub current_allocated_size: u64,
    /// `sysctlbyname("hw.memsize")` -- the host's total physical memory, in
    /// bytes, independent of anything Metal reports.
    pub physical_memory_bytes: u64,
}

/// Reads [`SystemMemoryFacts`] off this thread's Metal device
/// (`device_and_queue`, the same lazily-created device/queue pair every
/// other driver call in this module shares) and the host's `sysctlbyname`,
/// and emits them as one structured `system_facts` telemetry event. Callers
/// building a load-time fit budget (`proxima-model-interop`'s own gate) call
/// this once, before any weight upload, and derive their own limit from the
/// fields it returns -- this function makes no fit decision itself.
///
/// # Errors
///
/// [`MetalError::NoDevice`] if this host has no Metal device.
pub fn system_memory_facts() -> Result<SystemMemoryFacts, MetalError> {
    let (device, _queue) = device_and_queue()?;
    let facts = SystemMemoryFacts {
        recommended_max_working_set_size: device.recommendedMaxWorkingSetSize(),
        max_buffer_length: device.maxBufferLength() as u64,
        has_unified_memory: device.hasUnifiedMemory(),
        current_allocated_size: device.currentAllocatedSize() as u64,
        physical_memory_bytes: physical_memory_bytes(),
    };
    info!(
        recommended_max_working_set_size = facts.recommended_max_working_set_size,
        max_buffer_length = facts.max_buffer_length,
        has_unified_memory = facts.has_unified_memory,
        current_allocated_size = facts.current_allocated_size,
        physical_memory_bytes = facts.physical_memory_bytes,
        "system_facts: host and device memory facts probed at load time"
    );
    Ok(facts)
}

/// The host's total physical memory, in bytes -- `sysctlbyname("hw.memsize")`
/// rather than `sysconf` (unlike [`page_size`]'s `_SC_PAGESIZE`, POSIX has no
/// portable name for "total RAM"; `hw.memsize` is the macOS-specific MIB
/// name, read the same way `page_size` already reads a host fact through
/// `libc`).
pub(super) fn physical_memory_bytes() -> u64 {
    let mut value: u64 = 0;
    let mut size = core::mem::size_of::<u64>();
    // SAFETY: `name` is a NUL-terminated C string naming a real MIB entry;
    // `value`/`size` point at a live `u64` and its own length, exactly what
    // `sysctlbyname` requires for an output buffer; the two trailing
    // pointers are `None`/`0` since this call has nothing to write.
    unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut value).cast::<c_void>(),
            &raw mut size,
            core::ptr::null_mut(),
            0,
        );
    }
    value
}

/// This thread's Metal device and command queue, created on first use.
pub(super) fn device_and_queue() -> Result<DeviceAndQueue, MetalError> {
    DEVICE_AND_QUEUE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(existing) = slot.as_ref() {
            return Ok(existing.clone());
        }
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| MetalError::CompileFailed {
                log: "device refused to create a command queue".to_string(),
            })?;
        let pair = (device, queue);
        *slot = Some(pair.clone());
        Ok(pair)
    })
}

/// Runs a tensor program on the system's default Metal device.
///
/// Everything about a program that does not change between runs, resolved
/// ONCE so a serving loop stops re-deriving it per token.
///
/// [`execute`] re-ran `infer` + `bind` on every call, then re-derived which
/// operands are packed, allocated fresh device buffers, and read the result
/// back. Measured on this box (`omega/examples/q4k_matvec_probe.rs`, two
/// problem sizes so the intercept separates from the slope): **0.191 ms of
/// fixed cost per call on the f32 arm and 0.400 ms on the packed arm**. A
/// real forward is 1196 nodes, so at one `execute` per node that is 228-478
/// ms per forward of overhead against llama.cpp Metal's 17.62 ms for the
/// whole token (`proxima-tensor/docs/discipline.md` ROW 71).
///
/// What a caller can do with this that they could not do before: prepare a
/// program once and run it many times. That is the entire justification for
/// the type — the shapes, the bound ops, the retirement schedule and the
/// codec set are all functions of the PROGRAM, and a serving loop holds the
/// program fixed while the block DATA changes every token.
pub struct Plan {
    /// The plan owns its program: `finish` needs it for output dtypes, and a
    /// plan that borrowed it could not outlive the caller's buffer.
    pub(super) program: Vec<Op>,
    pub(super) prepared: Prepared,
    pub(super) packed_operands: PackedOperands,
    pub(super) block_dtypes: Vec<DType>,
    /// Every block-input node a caller has told this plan, via
    /// [`Plan::mark_resident`], is bound to data that never changes across
    /// calls -- empty until that method runs, since [`plan`] itself has no
    /// way to know a caller's residency intent from codecs/shapes alone.
    pub(super) resident_nodes: BTreeSet<NodeId>,
    /// Which [`MTLCompileOptions::mathMode`] every kernel this plan compiles
    /// is compiled under -- projected from `numeric_policy` at construction
    /// ([`numeric_policy_as_metal_math_mode`]), narrowable afterward within
    /// that policy via [`Plan::set_math_mode`]. See [`MathMode`]'s own doc
    /// for the measured rationale.
    pub(super) math_mode: MathMode,
    /// Which bit-changing rewrites this plan's bound program was
    /// constructed with -- [`plan`]/[`plan_named`]'s own `numeric_policy`
    /// argument, fixed for this plan's whole life (see
    /// [`Plan::numeric_policy`]'s own doc for why there is no setter).
    /// `msl::context_chunks_for` (the cross-simdgroup attention
    /// context-chunk merge) consults it via [`Plan::numeric_policy`].
    pub(super) numeric_policy: NumericPolicy,
    /// Which [`MTLDispatchType`] [`execute_plan_with_placements`] opens its
    /// compute encoder with -- [`DispatchType::default`] (`Concurrent`)
    /// until a caller overrides it with [`Plan::set_dispatch_type`]. See
    /// [`DispatchType`]'s own doc for the measured rationale (ROW 311/312).
    pub(super) dispatch_type: DispatchType,
    /// ROW 329 diagnostic: when `Some(position)`,
    /// [`execute_plan_with_placements_dispatch_timed`]'s stage-boundary
    /// fallback (the only branch this device's own `AtStageBoundary`-only
    /// support ever takes, ROW 309's own finding) ends its compute encoder
    /// immediately BEFORE this plan position and opens a second one for the
    /// remainder, instead of ROW 309's original one-encoder-per-position
    /// fallback -- exactly the "one encoder per kind group" degradation
    /// that row's own doc named, generalized from size-1 groups to a
    /// caller-chosen two-way split. `None` (the default) keeps that
    /// original per-position behavior. Never invalidates `resolved_steps`:
    /// which encoder a dispatch lands in does not change its compiled
    /// pipeline.
    #[cfg(feature = "instrument")]
    pub(super) encoder_split_at: Option<usize>,
    /// CARD 6.5: whole-buffer, size-class-reused device output buffers for
    /// every position in `prepared.resolved`. Built lazily, on the first
    /// call that actually consults a placement (`arena_placement`) --
    /// [`plan`] itself no longer builds this eagerly, since the ordinary
    /// (unplaced) `execute_plan`/`execute_plan_op_timed` paths never read it
    /// and were paying its device allocation on every miss regardless. See
    /// [`BufferArena`]'s own doc.
    #[cfg(feature = "metal-plan-stable-buffers")]
    pub(super) arena: core::cell::OnceCell<BufferArena>,
    /// CARD 6.5: one uniform buffer per plan position, written in place by
    /// `encode_op` instead of going through the content-keyed
    /// `UNIFORM_BUFFERS` cache. Lazily built alongside `arena`, for the same
    /// reason -- see [`PlanUniforms`]'s own doc.
    #[cfg(feature = "metal-plan-stable-buffers")]
    pub(super) uniforms: core::cell::OnceCell<PlanUniforms>,
    /// One scratch buffer per plan position, `None` for every position
    /// other than a `CachedAttention` op -- redesign §4c
    /// ([`NumericRewrite::ContextSplitMerge`]): the split kernel's own
    /// write set, the merge kernel's own read set (`crate::msl::Binding::
    /// Scratch`'s own doc). Sized once, for the compiled MAXIMUM split
    /// count (`crate::sized::ATTENTION_SPLIT_MAX`) regardless of which
    /// `NumericPolicy` this call happens to run under -- the same "always
    /// allocate room for the compiled ceiling" stance [`BufferArena`]'s own
    /// output slots take, so a later call narrowing/widening the active
    /// policy never needs a resize. Deliberately plan-owned rather than
    /// folded into [`BufferArena`]'s own whole-buffer reuse: that arena's
    /// slot-sharing invariant ("whole-buffer sharing only", that struct's
    /// own doc) is unverified against a SECOND buffer per position in this
    /// pass -- see `attention-kernel-design.md` §4c risk 1. A future slice
    /// folding this into the arena's own reuse is free to do so without
    /// changing this field's read side ([`attention_scratch_buffer`]).
    #[cfg(feature = "metal-plan-stable-buffers")]
    pub(super) attention_scratch: core::cell::OnceCell<Vec<Option<MetalBuffer>>>,
    /// Per-position `(pipeline, bindings, grid)` resolved once, lazily, on
    /// this plan's first [`execute_plan_with_placements`] call -- see
    /// [`resolve_steps`]'s own doc for why a hit no longer builds
    /// [`kernel_cache_key`]'s `String` or [`kernel_dispatch_shape`]'s `Vec`
    /// per step. `None` until that first call. Rebuilt whole when
    /// [`Plan::set_math_mode`] changes the mode a prior resolution used,
    /// since [`pipeline_for`] compiles one pipeline per mode. Populated and
    /// read only by the placement executors; every other executor leaves it
    /// `None` for this plan's whole life, which costs nothing beyond the
    /// one empty `RefCell`.
    pub(super) resolved_steps: RefCell<Option<ResolvedSteps>>,
    /// Resolved once per plan, lazily, the first call whose `device_buffers`
    /// already names every candidate group's own weight/activation/output
    /// buffers -- `resolve_steps`'s own `merge_candidates` is pure structure
    /// (pipeline identity + no dataflow edge) and knows nothing about which
    /// physical buffer a `NodeId` resolves to; THIS is where a candidate is
    /// admitted or refused on buffer identity (`ensure_merged_dispatches`'s
    /// own doc) and where the `base_table` upload happens, once. `None`
    /// until that first call; cleared by [`resolve_steps`] on a math-mode
    /// rebuild, since a stale entry would dispatch a pipeline compiled under
    /// the wrong mode.
    #[cfg(feature = "metal-horizontal-merge")]
    pub(super) merged: RefCell<Option<MergedPlanState>>,
    /// [`HazardState`]'s own doc: [`execute_plan_with_placements`]'s hazard
    /// tracker and its per-step input-pointer scratch, reused call-to-call
    /// instead of rebuilt every call.
    pub(super) hazard_state: RefCell<HazardState>,
    /// [`execute_plan_with_placements`]'s per-node buffer map -- plan-owned
    /// and NEVER rebuilt fresh (`BTreeMap::new()`) call-to-call, so its
    /// already-allocated tree nodes are reused for every call's `insert`/
    /// `remove` cycle instead of an empty map paying that allocation again
    /// (ROW 303's residual). Every call still writes a fresh entry for every
    /// block/position exactly as before this landing -- only WHERE the map
    /// lives changed, so content correctness is unaffected regardless of
    /// residency; [`Plan::block_identity`] layers a further, residency-gated
    /// skip of the upload itself on top.
    pub(super) device_buffers: RefCell<BTreeMap<NodeId, DeviceBuffer>>,
    /// Per-[`Prepared::block_nodes`]-position `(pointer, byte_length)` this
    /// plan last saw for a RESIDENT block -- `None` for a position that is
    /// not in [`Plan::resident_nodes`], or that has not been uploaded yet.
    /// [`execute_plan_with_placements`] compares this against the CURRENT
    /// call's own block identity to decide whether that position's entry in
    /// [`Plan::device_buffers`] can be trusted as-is (no content hash: an
    /// address match alone would wrongly reuse a freshly reallocated,
    /// unrelated buffer that happens to land at a freed one's old address --
    /// the RESIDENT gate is the caller's own promise that this address never
    /// moves and never changes content, the same promise
    /// [`Plan::mark_resident`]'s no-copy upload path already trusts).
    pub(super) block_identity: RefCell<Vec<Option<(usize, usize)>>>,
}

/// [`Plan::resolved_steps`]'s payload -- the [`MathMode`] it was built
/// under, so a later [`Plan::set_math_mode`] call is detected and triggers a
/// rebuild rather than silently serving stale pipelines for the old mode.
/// Keyed on `math_mode` alone, not `numeric_policy`: `numeric_policy` cannot
/// move post-construction ([`Plan::numeric_policy`]'s own doc), so
/// `math_mode` -- narrowable within the bound policy via
/// [`Plan::set_math_mode`] -- is the only axis that can still go stale here.
pub(super) struct ResolvedSteps {
    pub(super) math_mode: MathMode,
    pub(super) steps: Vec<ResolvedStep>,
    /// Structural merge candidates (`group_mergeable_positions`, pipeline
    /// identity + no-dataflow-edge only -- no buffer identity check yet: that
    /// half needs `device_buffers`, which does not exist this early). Empty
    /// when the feature is off, or the plan hits its own [`Plan::merged`]
    /// cache first (see that field's own doc for why buffer identity is
    /// resolved lazily, once per plan, from THIS list).
    #[cfg(feature = "metal-horizontal-merge")]
    pub(super) merge_candidates: Vec<Vec<usize>>,
}

/// Plan-time grouping of positions that share one compiled pipeline
/// (`kernel_identity`/`kernel_cache_key` equality, already paid for by
/// [`resolve_steps`]'s own pipeline-cache lookup) and have no dataflow edge
/// between them -- the shape `append_moe_round_output`'s 8 routed-expert
/// gate/up/down gathers per (layer, projection) take
/// (`proxima-tensor/src/spec.rs:1963-2010`): one shared weight-stack NodeId,
/// one shared activation NodeId, a distinct per-round gather-index and
/// output NodeId. Such a group COULD be issued as one Metal dispatch with a
/// `grid.z = group.len()` axis instead of `group.len()` separate dispatches
/// (see `docs/discipline.md`'s horizontal-packed-merge design note) -- this
/// function computes the grouping only; nothing downstream of
/// [`resolve_steps`] consumes it yet (see that feature's own doc in
/// `omega/Cargo.toml`).
///
/// Pure over caller-supplied identity/read/write data, no Metal types, so it
/// is testable without a device: `identity[index]` is any equality key that
/// collapses exactly the positions sharing one pipeline (a raw pipeline
/// pointer on the real driver path, `usize`/`&str` in tests); `writes[index]`
/// is the position's own output [`NodeId`]; `reads[index]` is every NodeId it
/// consumes. A group is returned only when it has 2 or more members --
/// singletons carry nothing to merge, so [`resolve_steps`] leaves them alone.
#[cfg(feature = "metal-horizontal-merge")]
pub(super) fn group_mergeable_positions<Identity: Eq + core::hash::Hash + Copy>(
    identities: &[Identity],
    reads: &[Vec<NodeId>],
    writes: &[NodeId],
) -> Vec<Vec<usize>> {
    let mut buckets: HashMap<Identity, Vec<usize>> = HashMap::new();
    for (index, identity) in identities.iter().enumerate() {
        buckets.entry(*identity).or_default().push(index);
    }
    let mut groups = Vec::new();
    for bucket in buckets.into_values() {
        groups.extend(split_into_independent_groups(&bucket, reads, writes));
    }
    groups
}

/// [`group_mergeable_positions`]'s per-identity-bucket half: greedily packs
/// positions into the first group every one of its current members is
/// independent of (`no_dataflow_edge`, `docs/discipline.md`'s design note
/// §1), opening a new group otherwise. A RAW edge between two same-identity
/// positions (one reads the other's write) therefore lands them in separate
/// groups rather than blocking the merge outright -- exactly the "a RAW edge
/// between two of them splits the group" behavior this landing's own test
/// asserts.
#[cfg(feature = "metal-horizontal-merge")]
pub(super) fn split_into_independent_groups(
    bucket: &[usize],
    reads: &[Vec<NodeId>],
    writes: &[NodeId],
) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = Vec::new();
    'candidate: for &index in bucket {
        for group in &mut result {
            let independent = group.iter().all(|&member| {
                writes[member] != writes[index]
                    && !reads[index].contains(&writes[member])
                    && !reads[member].contains(&writes[index])
            });
            if independent {
                group.push(index);
                continue 'candidate;
            }
        }
        result.push(alloc::vec![index]);
    }
    result.into_iter().filter(|group| group.len() > 1).collect()
}

/// [`group_mergeable_positions`]'s structural candidates further split by
/// PHYSICAL weight-buffer identity and activation NODE identity -- two
/// positions sharing a pipeline can still read two unrelated weight stacks
/// (checked by raw buffer pointer, the same identity
/// [`resolve_hazard_inputs_into`] already uses, since weight is always an
/// `Op::Input` checkpoint leaf resolved this early). Activation is admitted
/// by NODE identity, not buffer identity (ROW 572): a `NodeId` resolves to
/// exactly one buffer for the whole plan-execution call once it is known, so
/// two positions reading the SAME activation `NodeId` are reading the SAME
/// buffer whenever that buffer exists -- the group does not need it to exist
/// YET at this call, only to agree on WHICH node it will be. ROW 571 measured
/// that requiring the activation buffer to already be resolved here (the
/// admission this replaces) refused 100% of the real qwen35moe decode
/// graph's own candidates, because a real activation is an intermediate
/// hidden state the per-position encode loop has not reached yet -- never an
/// `Op::Input` leaf like this crate's own synthetic fixtures use. Output is
/// no longer checked here at all: the group's shared output buffer does not
/// exist yet either (see [`build_merged_dispatch`]'s own doc for where it is
/// allocated instead). A position with a different operand arity or an
/// unresolved weight is dropped from consideration entirely (never merged,
/// never blocks the rest of the bucket) -- [`build_merged_dispatch`]'s own
/// `Ok(None)` path covers the remaining "wrong op shape" refusal (not a
/// packed weight).
#[cfg(feature = "metal-horizontal-merge")]
pub(super) fn split_by_shared_buffers(
    group: &[usize],
    resolved: &[BoundOp],
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
) -> Vec<Vec<usize>> {
    let mut buckets: HashMap<(usize, NodeId), Vec<usize>> = HashMap::new();
    for &position in group {
        let bound = &resolved[position];
        let operands = bound.operands();
        if operands.len() != 2 {
            continue;
        }
        let Some((weight_buffer, _)) = device_buffers.get(&operands[0].0) else {
            #[cfg(feature = "instrument")]
            counter!(MERGE_CANDIDATE_UNRESOLVED_WEIGHT, 1);
            continue;
        };
        let activation_node = operands[1].0;
        let key = (Retained::as_ptr(weight_buffer) as usize, activation_node);
        buckets.entry(key).or_default().push(position);
    }
    buckets.into_values().filter(|bucket| bucket.len() > 1).collect()
}

/// Resolved once per [`Plan`], the first call whose `device_buffers` and
/// `output_placed` already name every candidate's own operands: promotes
/// [`ResolvedSteps::merge_candidates`] (pure structure) into real,
/// dispatchable [`MergedDispatch`]es (real buffer identity, a compiled
/// `_z{n}` pipeline, an uploaded offset table) or refuses them, never both --
/// a refused candidate falls through to [`encode_op`]'s ordinary one-
/// dispatch-per-position path exactly as if `metal-horizontal-merge` were
/// off. No-op on a call that already populated [`Plan::merged`].
#[cfg(feature = "metal-horizontal-merge")]
pub(super) fn ensure_merged_dispatches(
    device: &ProtocolObject<dyn MTLDevice>,
    plan: &Plan,
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
) -> Result<(), MetalError> {
    if plan.merged.borrow().is_some() {
        return Ok(());
    }
    let candidates: Vec<Vec<usize>> = plan
        .resolved_steps
        .borrow()
        .as_ref()
        .map(|steps| steps.merge_candidates.clone())
        .unwrap_or_default();
    let mut groups: Vec<MergedDispatch> = Vec::new();
    let mut position_merge: Vec<Option<(usize, u32)>> = vec![None; plan.prepared.resolved.len()];
    for candidate in &candidates {
        for subgroup in split_by_shared_buffers(candidate, &plan.prepared.resolved, device_buffers) {
            let subgroup_len = subgroup.len();
            match build_merged_dispatch(device, plan, &subgroup)? {
                Some(dispatch) => {
                    let merge_index = groups.len();
                    debug!(
                        group_len = subgroup_len,
                        merge_index, "horizontal-merge candidate accepted"
                    );
                    for (z, &position) in dispatch.members.iter().enumerate() {
                        position_merge[position] = Some((merge_index, z as u32));
                    }
                    groups.push(dispatch);
                }
                None => debug!(
                    group_len = subgroup_len,
                    "horizontal-merge candidate refused: not a packed two-operand matvec shape"
                ),
            }
        }
    }
    *plan.merged.borrow_mut() = Some(MergedPlanState {
        groups,
        position_merge,
    });
    Ok(())
}

/// Builds one [`MergedDispatch`]'s STRUCTURAL half from a weight-buffer-and-
/// activation-node-confirmed group: emits the leader's own kernel, splices
/// the `SliceBase`/`base_table` preamble onto it
/// ([`crate::msl::splice_horizontal_merge_base_table`]), and compiles it
/// under a `_z{n}`-suffixed cache key so it never aliases the leader's own
/// N=1 pipeline. Needs no `device_buffers`/`output_placed` at all (ROW 572):
/// none of this depends on where any operand's bytes actually live, only on
/// the plan's own symbolic shape -- the group's shared output buffer and its
/// base table's actual offsets are resolved later, at the leader's own first
/// encode ([`ensure_merged_group_resolved`]), which is the earliest point a
/// real decode plan's own activation (an intermediate, not an `Op::Input`
/// leaf) is guaranteed to exist. `Ok(None)` -- not `Err` -- for every shape
/// this landing does not cover (not a packed weight, an operand this driver
/// cannot resolve): a candidate this function declines is not a defect, it
/// is `encode_op`'s ordinary path asked to keep handling that position.
#[cfg(feature = "metal-horizontal-merge")]
pub(super) fn build_merged_dispatch(
    device: &ProtocolObject<dyn MTLDevice>,
    plan: &Plan,
    group: &[usize],
) -> Result<Option<MergedDispatch>, MetalError> {
    let leader = &plan.prepared.resolved[group[0]];
    let operands = leader.operands();
    if operands.len() != 2 {
        return Ok(None);
    }
    let weight_node = operands[0].0;
    let activation_node = operands[1].0;
    if !plan.packed_operands.contains_key(&weight_node) {
        return Ok(None);
    }
    // A gathered operand's kernel binds a `Binding::Fault` slot
    // (`encode_op`'s own `gather_count(bound) > 0` branch allocates it);
    // `handle_merged_position` never resolves one (it always passes `fault:
    // None` to `bind_buffers`), so admitting a gathered candidate here
    // compiled and dispatched a kernel `bind_buffers` then rejected outright
    // ("kernel binds a fault buffer but none was allocated") -- refused
    // here instead, the same "not a shape this landing covers" refusal as
    // every other structural mismatch this function already declines.
    if gather_count(leader) > 0 {
        return Ok(None);
    }
    // A routed-expert weight stack (`execute_plan_named_with_placements_and_
    // expert_sources`'s own naming convention, `execute_plan_with_placements_
    // inner`'s `_exps.weight` check just above this function's own call
    // site) is bound via a PER-CALL substitution (`Binding::ExpertPayloads`/
    // `ExpertDescriptors`, resolved in `encode_op` from that call's own
    // `expert_buffers` argument, which this plan-resolution-time function
    // never sees) -- `crate::msl::emit` here compiles the ORDINARY,
    // non-substituted kernel for it, which reads the wrong buffer entirely.
    // Refused, the same "not a shape this landing covers" refusal as every
    // other structural mismatch.
    if plan.program[weight_node.0 as usize]
        .name()
        .is_some_and(|name| name.contains("_exps.weight"))
    {
        return Ok(None);
    }
    let mut kernel = crate::msl::emit(leader, &plan.packed_operands, plan.numeric_policy)?;
    let Some(weight_index) = kernel.bindings.iter().position(
        |binding| matches!(binding, Binding::Input(node) if *node == weight_node),
    ) else {
        return Ok(None);
    };
    let Some(other_index) = kernel.bindings.iter().position(
        |binding| matches!(binding, Binding::Input(node) if *node == activation_node),
    ) else {
        return Ok(None);
    };
    crate::msl::splice_horizontal_merge_base_table(
        &mut kernel,
        leader.node,
        weight_index,
        "uchar",
        other_index,
        "float",
        "float",
    )?;
    let mut cache_key = kernel_cache_key(leader, &plan.packed_operands, plan.numeric_policy)?;
    cache_key.push(plan.math_mode.cache_token());
    cache_key.push_str(&format!("_z{}", group.len()));
    let pipeline = pipeline_for_kernel(device, &kernel, &cache_key, plan.math_mode)?;
    let grid = GridSpec {
        threads: kernel.grid.threads,
        threadgroup_width: kernel.grid.threadgroup_width,
        depth: group.len() as u64,
    };
    let member_bytes = bound_output_len(leader).max(1) * leader.dtype.size_bytes();
    Ok(Some(MergedDispatch {
        members: group.to_vec(),
        pipeline,
        bindings: kernel.bindings,
        grid,
        member_bytes,
        dtype: leader.dtype,
        resolved: RefCell::new(None),
    }))
}

/// The group's own shared output buffer and uploaded base table, built once
/// -- lazily, at the leader's first real encode, never at plan-resolution
/// time -- and reused for every later decode step this same `Plan` serves
/// (ROW 572's own "cache per (plan, group) after the first step" call:
/// offsets are stable once resolved, because [`MergedDispatch::member_bytes`]
/// is a structural constant and this allocation is never handed back to a
/// [`BufferArena`] free list -- it lives exactly as long as the `Plan` does).
#[cfg(feature = "metal-horizontal-merge")]
pub(super) struct ResolvedMergedGroup {
    pub(super) base_table: MetalBuffer,
    /// `Some` for the fresh-allocation branch (member `i`'s own
    /// `device_buffers` entry is `(output, i * member_bytes)`, re-derivable
    /// on every call without touching `output_placed` again); `None` for the
    /// caller-placed branch, whose per-member entries are re-read from
    /// `output_placed` fresh on every call instead, since that map is
    /// already a per-call argument with nothing to cache.
    pub(super) output: Option<MetalBuffer>,
}

/// Resolves (once) or reuses (every call after) [`MergedDispatch::resolved`]
/// for the group at `merge_index`: allocates ONE fresh, dedicated buffer
/// sized `member_bytes * members.len()` -- [`BufferArena`]'s own doc rules
/// out sub-allocating a slot across two simultaneously-live positions, so
/// this bypasses the arena entirely and asks the device directly, the same
/// primitive [`allocate_buffer`] already is for any other non-placed,
/// non-arena output -- then registers EVERY member's own node in
/// `device_buffers` at `(output, i * member_bytes)` (the exact `(buffer,
/// offset)` shape the placement path already uses, proven correct by every
/// `metal_output_placement.rs` consumer), builds the base table from each
/// member's own resolved weight/activation offsets relative to the leader's,
/// and uploads it. Every member's activation operand is admitted by NODE
/// identity (`split_by_shared_buffers`'s own doc), so the `debug_assert!`
/// below is the promised buffer-identity check, demoted from an admission
/// gate to a cheap sanity assertion here at the one point the buffer
/// actually needs to exist.
#[cfg(feature = "metal-horizontal-merge")]
pub(super) fn ensure_merged_group_resolved(
    device: &ProtocolObject<dyn MTLDevice>,
    plan: &Plan,
    merge_index: usize,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    output_placed: &BTreeMap<NodeId, (&PlacedBuffer, usize)>,
) -> Result<(), MetalError> {
    // `device_buffers` is retired every position by the SAME `node_retirement`
    // liveness analysis every other node uses -- a merge member's own output
    // entry is gone again the step after its last reader consumed it, exactly
    // like any other intermediate. Caching the group's own allocation and
    // base table (the expensive half) across steps is safe -- their OWN
    // offsets are structural and never change -- but re-registering every
    // member's `device_buffers` entry from that cached buffer must run EVERY
    // call, not just the first, or a later step's own reader of a NON-leader
    // member's output fails exactly the way a caller-placed output never
    // could (`MetalError::UnresolvedHazardOperand`, measured against the
    // real qwen35moe decode graph's own second-and-later decode step).
    let already_resolved = {
        let merged_guard = plan.merged.borrow();
        let dispatch = &merged_guard.as_ref().ok_or(MetalError::CompileFailed {
            log: "horizontal-merge position_merge named a group but Plan::merged is empty".to_string(),
        })?.groups[merge_index];
        dispatch.resolved.borrow().is_some()
    };
    if already_resolved {
        let merged_guard = plan.merged.borrow();
        let merged_state = merged_guard.as_ref().ok_or(MetalError::CompileFailed {
            log: "horizontal-merge position_merge named a group but Plan::merged is empty".to_string(),
        })?;
        let dispatch = &merged_state.groups[merge_index];
        let resolved_guard = dispatch.resolved.borrow();
        let resolved = resolved_guard.as_ref().ok_or(MetalError::CompileFailed {
            log: "row 572: already_resolved just confirmed Some".to_string(),
        })?;
        for (index, &position) in dispatch.members.iter().enumerate() {
            let node = plan.prepared.resolved[position].node;
            let entry = match &resolved.output {
                Some(buffer) => (buffer.clone(), index * dispatch.member_bytes),
                None => output_placed
                    .get(&node)
                    .copied()
                    .map(|(buffer, offset)| (buffer.clone(), offset))
                    .ok_or(MetalError::CompileFailed {
                        log: "row 572: a caller-placed group's own placement must still be \
                              supplied on every call"
                            .to_string(),
                    })?,
            };
            device_buffers.insert(node, entry);
        }
        return Ok(());
    }
    let merged_guard = plan.merged.borrow();
    let merged_state = merged_guard.as_ref().ok_or(MetalError::CompileFailed {
        log: "horizontal-merge position_merge named a group but Plan::merged is empty".to_string(),
    })?;
    let dispatch = &merged_state.groups[merge_index];
    let leader_position = dispatch.members[0];
    let leader = &plan.prepared.resolved[leader_position];
    let leader_operands = leader.operands();
    let (_, leader_weight_offset) = buffer_for(device_buffers, leader_operands[0].0)?;
    let (leader_activation_buffer, leader_activation_offset) = buffer_for(device_buffers, leader_operands[1].0)?;

    // A caller that explicitly placed every member's own output (this
    // crate's own synthetic fixtures; ROW 568's own proof this shape is
    // correct) is honored exactly as before -- the group's shared buffer is
    // never substituted for a buffer the caller specifically asked for.
    // Production never places these (ROW 571's own measurement), so this
    // branch allocates a fresh, dedicated buffer sized for the whole group
    // instead: `BufferArena` cannot hand out a slot shared across several
    // simultaneously-live positions (its own "whole-buffer sharing only"
    // doc), so this bypasses the arena entirely and asks the device
    // directly, the same primitive `allocate_buffer` already is for any
    // other non-placed, non-arena output.
    let all_caller_placed = dispatch
        .members
        .iter()
        .all(|&position| output_placed.contains_key(&plan.prepared.resolved[position].node));
    let allocated_output = if all_caller_placed {
        None
    } else {
        let member_elements = dispatch.member_bytes / dispatch.dtype.size_bytes();
        Some(allocate_buffer(device, member_elements * dispatch.members.len(), dispatch.dtype)?)
    };
    // Every offset pushed below is relative to THIS value -- 0 for a fresh
    // allocation (the leader's own slice starts the buffer), or the leader's
    // own caller-placed offset otherwise -- matching `bind_buffers` binding
    // the leader's buffer at the leader's own `setBuffer:offset:` and the
    // spliced kernel body adding `base_table[z]` on top of that.
    let leader_output_offset = match &allocated_output {
        Some(_) => 0,
        None => {
            output_placed
                .get(&leader.node)
                .copied()
                .ok_or(MetalError::CompileFailed {
                    log: "row 572: all_caller_placed already confirmed the leader is placed".to_string(),
                })?
                .1
        }
    };

    let mut offsets: Vec<u64> = Vec::with_capacity(dispatch.members.len() * 3);
    for (index, &position) in dispatch.members.iter().enumerate() {
        let bound = &plan.prepared.resolved[position];
        let bound_operands = bound.operands();
        let (_, weight_offset) = buffer_for(device_buffers, bound_operands[0].0)?;
        let (activation_buffer, activation_offset) = buffer_for(device_buffers, bound_operands[1].0)?;
        // The promised buffer-identity check, demoted from an admission
        // gate (`split_by_shared_buffers` admits by NODE identity now) to a
        // cheap sanity assertion at the one point the buffer actually needs
        // to exist: a `NodeId` resolving to two different buffers within
        // one plan-execution call would violate the one invariant this
        // admission depends on.
        debug_assert!(
            Retained::as_ptr(&activation_buffer) == Retained::as_ptr(&leader_activation_buffer)
                && activation_offset == leader_activation_offset,
            "row 572: two positions admitted on the same activation NodeId must resolve to the \
             same buffer and offset"
        );
        let (output_buffer, output_offset) = match &allocated_output {
            Some(buffer) => (buffer.clone(), index * dispatch.member_bytes),
            None => output_placed
                .get(&bound.node)
                .copied()
                .map(|(buffer, offset)| (buffer.clone(), offset))
                .ok_or(MetalError::CompileFailed {
                    log: "row 572: all_caller_placed already confirmed every member is placed".to_string(),
                })?,
        };
        device_buffers.insert(bound.node, (output_buffer, output_offset));
        offsets.push(relative_byte_offset(weight_offset, leader_weight_offset));
        offsets.push(relative_byte_offset(activation_offset, leader_activation_offset));
        offsets.push(relative_byte_offset(output_offset, leader_output_offset));
    }
    let base_table = upload_base_table(device, &offsets)?;
    *dispatch.resolved.borrow_mut() = Some(ResolvedMergedGroup {
        base_table,
        output: allocated_output,
    });
    Ok(())
}

/// `member_offset - leader_offset` as a `u64`: both are byte offsets into the
/// SAME [`MetalBuffer`] ([`split_by_shared_buffers`]'s own admission
/// requirement), so this never wraps in practice, but the subtraction is
/// still done in `i64` first so a member ever placed BEFORE its leader (a
/// legal, if unusual, offset ordering) round-trips exactly rather than
/// panicking on an unsigned underflow.
#[cfg(feature = "metal-horizontal-merge")]
pub(super) fn relative_byte_offset(member_offset: usize, leader_offset: usize) -> u64 {
    (member_offset as i64 - leader_offset as i64) as u64
}

/// Uploads `offsets` (three `u64`s per merged member: weight/activation/
/// output base, [`splice_horizontal_merge_base_table`]'s own `SliceBase`
/// layout) as one small, plan-owned buffer -- built once per group inside
/// [`ensure_merged_dispatches`]'s own lazy resolution, never per dispatch.
#[cfg(feature = "metal-horizontal-merge")]
pub(super) fn upload_base_table(
    device: &ProtocolObject<dyn MTLDevice>,
    offsets: &[u64],
) -> Result<MetalBuffer, MetalError> {
    let bytes: &[u8] = unsafe {
        core::slice::from_raw_parts(offsets.as_ptr().cast::<u8>(), core::mem::size_of_val(offsets))
    };
    // SAFETY: `bytes` borrows `offsets`, which outlives this call, and its
    // length matches `offsets`'s own byte length exactly -- the same
    // single-copy contract `upload_uniforms` relies on for its own
    // `newBufferWithBytes_length_options` call just above it in this file.
    let pointer = unsafe { NonNull::new_unchecked(bytes.as_ptr() as *mut c_void) };
    unsafe { device.newBufferWithBytes_length_options(pointer, bytes.len(), MTLResourceOptions::StorageModeShared) }
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate the horizontal-merge base table".to_string(),
        })
}

/// [`execute_plan_with_placements_inner`]'s per-position loop, factored out
/// so the merge arm compiles ONLY under this feature ([`Plan::merged`] does
/// not exist otherwise) without an inline `#[cfg]` splitting one `if`
/// expression's branches. Returns `false` for any position `Plan::merged`
/// never claimed -- the caller falls through to its own ordinary
/// `encode_op` path exactly as if this feature were off. `true` covers BOTH
/// a real dispatch (`z == 0`) and a skipped member (`z > 0`): either way the
/// caller's own retirement loop still runs unconditionally afterward, so a
/// skipped member's buffer is retired at the same program position it
/// always was.
#[cfg(feature = "metal-horizontal-merge")]
#[allow(clippy::too_many_arguments)]
pub(super) fn handle_merged_position(
    device: &ProtocolObject<dyn MTLDevice>,
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    plan: &Plan,
    position: usize,
    bound: &BoundOp,
    dispatch_type: DispatchType,
    device_buffers: &mut BTreeMap<NodeId, DeviceBuffer>,
    hazard_state: &mut HazardState,
    output_placed: &BTreeMap<NodeId, (&PlacedBuffer, usize)>,
) -> Result<bool, MetalError> {
    let Some((merge_index, z)) = plan
        .merged
        .borrow()
        .as_ref()
        .and_then(|state| state.position_merge[position])
    else {
        return Ok(false);
    };
    // The leader (z == 0) is always the SMALLEST program position among the
    // group's members (`group_mergeable_positions`'s own bucket-preserving
    // order), so it is reached here before every other member -- the one
    // point this group's own activation is guaranteed to already be
    // resolved (an earlier position in the same program computed it) and
    // the one point nothing has read this group's shared output buffer yet.
    // A no-op on every call after the plan's first (ROW 572's own "cache per
    // (plan, group)" call).
    if z == 0 {
        ensure_merged_group_resolved(device, plan, merge_index, device_buffers, output_placed)?;
    }
    // Every member's own output still needs a hazard record against the
    // shared read set -- `HazardTracker::record`'s own signature is
    // unchanged (one `Option<Id>` output); this is the caller (design note
    // §4) looping it once per member instead of once per dispatch, since
    // only `z == 0` actually dispatches. A miss here (skipping hazard
    // bookkeeping for z>0) is exactly the silent RAW/WAW/WAR data race the
    // design note's own Risks section names -- every member is already
    // known to write into the SAME physical output buffer as its leader,
    // registered into `device_buffers` by `ensure_merged_group_resolved`
    // above (or, from step 2 onward, on a prior call).
    if dispatch_type == DispatchType::Concurrent {
        let operand_nodes: Vec<NodeId> = bound.operands().iter().map(|(node, ..)| *node).collect();
        resolve_hazard_inputs_into(operand_nodes.into_iter(), device_buffers, &mut hazard_state.inputs)?;
        let output_pointer = device_buffers
            .get(&bound.node)
            .map(|(buffer, _)| Retained::as_ptr(buffer))
            .ok_or(MetalError::UnresolvedHazardOperand { node: bound.node })?;
        if hazard_step(&mut hazard_state.tracker, &hazard_state.inputs, output_pointer) {
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            counter!(BARRIERS_EMITTED, 1);
        }
    }
    if z != 0 {
        return Ok(true);
    }
    let merged_guard = plan.merged.borrow();
    let merged_state = merged_guard.as_ref().ok_or(MetalError::CompileFailed {
        log: "horizontal-merge position_merge named a group but Plan::merged is empty".to_string(),
    })?;
    let merged_dispatch = &merged_state.groups[merge_index];
    let resolved_guard = merged_dispatch.resolved.borrow();
    let resolved = resolved_guard.as_ref().ok_or(MetalError::CompileFailed {
        log: "horizontal-merge leader dispatched before its own group was resolved".to_string(),
    })?;
    // The leader's own `(buffer, offset)` -- registered by
    // `ensure_merged_group_resolved` above, the same call this position's
    // own `z == 0` branch already made, so this is always populated here.
    let (output_buffer, output_offset) = device_buffers
        .get(&bound.node)
        .cloned()
        .ok_or(MetalError::UnresolvedHazardOperand { node: bound.node })?;
    let output = (&output_buffer, output_offset);
    let uniform_buffer = plan_uniform_buffer(plan, position)?;
    let owned_uniforms: MetalBuffer;
    let uniforms: &MetalBuffer = match uniform_buffer {
        Some(buffer) => buffer,
        None => {
            owned_uniforms = upload_uniforms(device, &pack_uniforms(bound, plan.numeric_policy)?)?;
            &owned_uniforms
        }
    };
    encoder.setComputePipelineState(&merged_dispatch.pipeline);
    bind_buffers(
        encoder,
        &merged_dispatch.bindings,
        device_buffers,
        output,
        None,
        uniforms,
        None,
        None,
        None,
        None,
    )?;
    // the ONE binding `splice_horizontal_merge_base_table` adds outside
    // `Kernel::bindings` (design note §1: no `NodeId` of its own, so it
    // never belongs in the shared `Binding` enum) -- bound at the buffer
    // index the splice's own MSL text literally names
    // (`kernel.bindings.len()` AT SPLICE TIME, which is
    // `merged_dispatch.bindings.len()` here since splicing never mutates
    // that `Vec`).
    unsafe {
        encoder.setBuffer_offset_atIndex(
            Some(&resolved.base_table),
            0,
            merged_dispatch.bindings.len(),
        );
    }
    dispatch(encoder, &merged_dispatch.pipeline, merged_dispatch.grid);
    #[cfg(feature = "instrument")]
    counter!(ENCODE_DISPATCH_CALLS, 1);
    Ok(true)
}

/// One [`Plan::prepared`] position's compiled pipeline plus the two other
/// per-op values [`encode_op`] needs to dispatch it -- resolved once by
/// [`resolve_steps`] instead of every step re-deriving [`kernel_cache_key`]
/// (a `String`) and [`kernel_dispatch_shape`] (a `Vec<Binding>`) just to
/// look the same pipeline up again.
pub(super) struct ResolvedStep {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) bindings: Vec<Binding>,
    pub(super) grid: GridSpec,
    /// Redesign §4c: `Some` exactly when [`crate::msl::
    /// cached_attention_merge_needed`] admits `ContextSplitMerge` for this
    /// position's `CachedAttention` op -- its own compiled pipeline (a
    /// SEPARATE `MTLLibrary`/pipeline-cache entry, keyed on the `_merge`
    /// entry name [`crate::msl::emit_cached_attention_merge`] builds), read
    /// back out of the scratch buffer `bindings`' own trailing
    /// `Binding::Scratch` slot wrote.
    pub(super) merge: Option<ResolvedMergeStep>,
}

/// [`ResolvedStep::merge`]'s payload -- the SAME triple `ResolvedStep`
/// itself carries, so [`encode_op`]'s second dispatch binds and dispatches
/// it identically to the first, just against a different pipeline/bindings/
/// grid and a scratch-buffer placement instead of the position's real
/// output.
pub(super) struct ResolvedMergeStep {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) bindings: Vec<Binding>,
    pub(super) grid: GridSpec,
}

/// One [`ResolvedSteps::merge_candidates`] group, promoted to a real,
/// dispatchable merged kernel: the leader's own bindings/grid (unchanged --
/// `base_table` is bound OUTSIDE this list and `merge_gid` is a widened
/// thread-position attribute, never a buffer binding at all -- see the design
/// note's own reasoning against widening the shared [`Binding`] enum for a
/// buffer with no `NodeId` of its own) plus the compiled `_z{n}` pipeline.
/// `member_bytes`/`dtype` are structural (known at plan-resolution time, no
/// buffer needed); `resolved` -- the group's shared output buffer and
/// uploaded base table -- is filled in lazily, at the leader's first real
/// encode, by [`ensure_merged_group_resolved`] (ROW 572).
#[cfg(feature = "metal-horizontal-merge")]
pub(super) struct MergedDispatch {
    /// Positions in `plan.prepared.resolved`, leader (z=0) first.
    pub(super) members: Vec<usize>,
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) bindings: Vec<Binding>,
    pub(super) grid: GridSpec,
    pub(super) member_bytes: usize,
    pub(super) dtype: DType,
    pub(super) resolved: RefCell<Option<ResolvedMergedGroup>>,
}

/// [`Plan::merged`]'s payload: every merge this plan resolved into a real
/// dispatch, plus a dense position -> `(merged index, z)` map so the encode
/// loop is one array read per position instead of a scan.
#[cfg(feature = "metal-horizontal-merge")]
pub(super) struct MergedPlanState {
    pub(super) groups: Vec<MergedDispatch>,
    pub(super) position_merge: Vec<Option<(usize, u32)>>,
}

impl Plan {
    /// Tells this plan which of its named block inputs are the CALLER's own
    /// static data -- model weights bound once at load and never mutated
    /// again -- so [`execute_plan`]'s upload loop may cache and reuse their
    /// device buffer on the copying path instead of re-copying every call.
    /// See the module doc's "Resident blocks" section for the full argument.
    ///
    /// `resident_names` is checked against each block-input node's own
    /// declared [`Op::name`] -- a one-time string compare over the program's
    /// declared inputs, off the per-token upload path -- never against the
    /// block's bytes or pointer. That is the necessary half of the
    /// invariant this driver's `NOCOPY_BUFFERS` cache cannot provide on its
    /// own: an address match alone cannot distinguish a weight buffer that
    /// never moves from a freshly reallocated `ids`/KV-cache buffer that
    /// coincidentally lands at a freed address of the same size. The NAME
    /// check happens here, once per plan, specifically so the per-token
    /// upload path never has to trust an address by itself.
    ///
    /// Safe and cheap to call every token even though [`plan`]/[`plan_named`]
    /// currently rebuild a fresh [`Plan`] every step (`plan_hits=0`,
    /// `proxima-tensor/docs/discipline.md` ROW 82): this is a scan over the
    /// block-input node list matching each node's name against
    /// `resident_names`, not a device operation.
    pub fn mark_resident(&mut self, resident_names: &BTreeSet<&str>) {
        self.resident_nodes = self
            .prepared
            .block_nodes
            .iter()
            .filter(|node| {
                self.program[node.0 as usize]
                    .name()
                    .is_some_and(|name| resident_names.contains(name))
            })
            .copied()
            .collect();
    }

    /// Extends [`Self::resident_nodes`] with every `Op::Iota`/`Op::Constant`
    /// leaf this plan's own `prepared.resolved` dispatches -- both are pure
    /// functions of their own `extent`/`value` (op.rs's own doc: "nothing
    /// external binds to it"), so the first call's real dispatch computes a
    /// value good for the plan's whole life, the same durability promise
    /// [`Self::mark_resident`] already gives a checkpoint weight. Joining the
    /// SAME `resident_nodes` set (not a second one) means both consumers of
    /// that set for free: the retirement loop's own `resident_nodes` check
    /// (`execute_plan_with_placements`'s own doc) stops dropping this node's
    /// `device_buffers` entry after the call that computed it, and
    /// [`resident_pinned_retires`] stops handing its `BufferArena` slot back
    /// to a later position -- without either site needing to know this node
    /// is a leaf rather than a checkpoint block.
    pub fn mark_plan_time_constants_resident(&mut self) {
        self.resident_nodes.extend(
            self.prepared
                .resolved
                .iter()
                .filter(|bound| matches!(bound.kind, BoundOpKind::Iota | BoundOpKind::Constant { .. }))
                .map(|bound| bound.node),
        );
    }
}

/// [`arena_placement`]'s own `retires` argument, filtered so a
/// [`Plan::resident_nodes`] member (a checkpoint weight OR a
/// [`Plan::mark_plan_time_constants_resident`] leaf) never has its
/// `BufferArena` slot handed back to a later position -- [`build_buffer_arena`]
/// is built exactly once, before `mark_resident`/`mark_plan_time_constants_resident`
/// ever narrow which nodes must outlive the call that wrote them, so this
/// filter is the one point downstream of both that actually enforces it. A
/// plan with no resident node (today's default) allocates nothing extra: the
/// filter still walks `retires`, but every `contains` check is a `BTreeSet`
/// miss and the resulting `Vec` is byte-identical to the input.
pub(super) fn resident_pinned_retires(plan: &Plan) -> Vec<Vec<NodeId>> {
    plan.prepared
        .retires
        .iter()
        .map(|retired| {
            retired
                .iter()
                .copied()
                .filter(|node| !plan.resident_nodes.contains(node))
                .collect()
        })
        .collect()
}

/// The name [`Plan::mark_resident`] proved this node's block input is bound
/// to for the life of the served model, or `None` when the node was never
/// classified resident. This is the identity [`upload_resident_copy`] caches
/// on -- see that function's own doc for why a host address cannot serve as
/// this identity instead.
pub(super) fn resident_name(plan: &Plan, node: NodeId) -> Option<&str> {
    plan.resident_nodes
        .contains(&node)
        .then(|| plan.program[node.0 as usize].name())
        .flatten()
}

/// A block's own `(pointer, byte_length)` -- every [`QuantizedBlock`] variant
/// is a borrowed slice, so this is the address identity
/// [`block_buffer_reusable`] compares, never the bytes themselves.
pub(super) fn block_identity_key(block: &QuantizedBlock<'_>) -> (usize, usize) {
    match block {
        QuantizedBlock::Float32(data) => (data.as_ptr().cast::<()>() as usize, size_of_val(*data)),
        QuantizedBlock::Int32(data) => (data.as_ptr().cast::<()>() as usize, size_of_val(*data)),
        QuantizedBlock::Packed { bytes, .. } => (bytes.as_ptr().cast::<()>() as usize, bytes.len()),
    }
}

/// Whether a RESIDENT block-input position's existing
/// [`Plan::device_buffers`] entry can be trusted as-is this call, so
/// [`execute_plan_with_placements`] can skip re-uploading it entirely.
/// `resident` alone is not enough -- see [`Plan::block_identity`]'s own doc
/// for why an address match still requires the caller's own residency
/// promise before it is trusted, and why no content hash backs it up.
pub(super) fn block_buffer_reusable(
    resident: bool,
    previous: Option<(usize, usize)>,
    current: (usize, usize),
) -> bool {
    resident && previous == Some(current)
}

// file-split artifact: this test module sits mid-file here (it trailed
// more code in the pre-split metal.rs); the items after it are unchanged
// production code, not newly misplaced.
#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::items_after_test_module
)]
pub(super) mod block_buffer_reusable_tests {
    //! Pure, GPU-free proof of the reuse decision
    //! [`execute_plan_with_placements`]'s block-upload loop relies on --
    //! `plan_pipeline_alloc_count.rs` proves the allocation count this
    //! decision buys end to end, on a real Metal device; this proves the
    //! decision itself, so a regression here fails fast without a GPU.

    use super::block_buffer_reusable;

    #[test]
    fn a_resident_block_at_the_same_address_and_length_is_reused() {
        let identity = (0x1000, 64);
        assert!(block_buffer_reusable(true, Some(identity), identity));
    }

    #[test]
    fn a_resident_block_whose_address_moved_is_rebuilt() {
        let previous = (0x1000, 64);
        let current = (0x2000, 64);
        assert!(!block_buffer_reusable(true, Some(previous), current));
    }

    #[test]
    fn a_resident_block_whose_length_changed_is_rebuilt() {
        let previous = (0x1000, 64);
        let current = (0x1000, 128);
        assert!(!block_buffer_reusable(true, Some(previous), current));
    }

    #[test]
    fn a_non_resident_block_is_never_reused_even_at_the_same_address() {
        let identity = (0x1000, 64);
        assert!(!block_buffer_reusable(false, Some(identity), identity));
    }

    #[test]
    fn a_first_call_with_no_recorded_identity_is_never_reused() {
        assert!(!block_buffer_reusable(true, None, (0x1000, 64)));
    }
}

impl Plan {
    /// Narrows this plan's compiled [`MathMode`] from [`MathMode::default`]
    /// (`Relaxed`) -- never widens it, and never touches
    /// [`Self::numeric_policy`]. Safe to call any time before an
    /// `execute_plan*` call -- `pipeline_for`'s cache key folds the mode in,
    /// so switching a plan's mode between calls never hands back a pipeline
    /// compiled for the other one.
    ///
    /// [`MathMode`] is a 3-rung compiler flag; [`NumericPolicy`] is the
    /// richer, orthogonal 5-permission set this plan's bound program was
    /// actually constructed under ([`plan`]/[`plan_named`]'s own argument,
    /// fixed for this plan's whole life -- see [`Self::numeric_policy`]'s
    /// own doc for why there is no setter for it). `math_mode` may only
    /// request what `numeric_policy` already grants
    /// (`metal_math_mode_as_numeric_policy`); requesting `Fast` on a plan
    /// bound under `bit_exact()` is a caller error, not a silent widening,
    /// so this returns [`MetalError::NumericPolicyMismatch`] instead of
    /// mutating `numeric_policy` the way this setter used to.
    ///
    /// # Errors
    /// [`MetalError::NumericPolicyMismatch`] when `math_mode` needs a
    /// permission `self.numeric_policy` does not grant.
    pub fn set_math_mode(&mut self, math_mode: MathMode) -> Result<(), MetalError> {
        let requested = metal_math_mode_as_numeric_policy(math_mode);
        if self.numeric_policy.grants(requested) {
            self.math_mode = math_mode;
            Ok(())
        } else {
            Err(MetalError::NumericPolicyMismatch {
                bound: self.numeric_policy,
                requested,
            })
        }
    }

    /// The [`NumericPolicy`] this plan's bound program was constructed under
    /// ([`plan`]/[`plan_named`]'s own `numeric_policy` argument) -- fixed
    /// for this plan's whole life. There is no setter: the bound program's
    /// topology (which identity eliminations fired, whether chain/reduce-
    /// epilogue fusion ran) is decided once, at bind time, and cannot be
    /// patched after without rebuilding it from `blocks`, which this type
    /// does not retain (the data is call-scoped, not plan-scoped -- see
    /// `Plan`'s own struct doc). Build a fresh [`Plan`] via [`plan`]/
    /// [`plan_named`] with a different policy instead; use
    /// [`Self::check_numeric_policy`] to confirm a `Plan` from elsewhere (a
    /// test fixture, a cached plan) matches before trusting it.
    #[must_use]
    pub fn numeric_policy(&self) -> NumericPolicy {
        self.numeric_policy
    }

    /// `Ok(())` when `desired` matches the policy this plan was bound
    /// under, `Err` otherwise. See [`Self::numeric_policy`]'s own doc for
    /// why there is no in-place rebind.
    ///
    /// # Errors
    /// [`MetalError::NumericPolicyMismatch`] when `desired` differs from
    /// [`Self::numeric_policy`].
    pub fn check_numeric_policy(&self, desired: NumericPolicy) -> Result<(), MetalError> {
        if self.numeric_policy == desired {
            Ok(())
        } else {
            Err(MetalError::NumericPolicyMismatch {
                bound: self.numeric_policy,
                requested: desired,
            })
        }
    }

    /// Overrides this plan's [`DispatchType`] from [`DispatchType::default`]
    /// (`Concurrent`). Safe to call any time before an `execute_plan*` call
    /// -- unlike [`Self::set_math_mode`], this never invalidates
    /// `resolved_steps`: a plan's compiled pipelines are the same regardless
    /// of which encoder dispatch mode runs them.
    pub fn set_dispatch_type(&mut self, dispatch_type: DispatchType) {
        self.dispatch_type = dispatch_type;
    }

    /// This plan's currently applied [`MathMode`] -- the read side of
    /// [`Self::set_math_mode`], added so a caller (or a test) can prove a
    /// build path actually applied a mode rather than only asserting that a
    /// setter was called somewhere upstream.
    #[must_use]
    pub fn math_mode(&self) -> MathMode {
        self.math_mode
    }

    /// This plan's currently applied [`DispatchType`] -- [`Self::math_mode`]'s
    /// counterpart for [`Self::set_dispatch_type`].
    #[must_use]
    pub fn dispatch_type(&self) -> DispatchType {
        self.dispatch_type
    }

    /// Overrides this plan's ROW 329 encoder-split position from `None`
    /// (one stage-sampled encoder per position, ROW 309's original
    /// fallback). Safe to call any time before an
    /// [`execute_plan_with_placements_dispatch_timed`] call; every other
    /// executor -- including this same function's `AtDispatchBoundary`
    /// branch, on a device that has it -- ignores this field entirely.
    #[cfg(feature = "instrument")]
    pub fn set_encoder_split_at(&mut self, position: Option<usize>) {
        self.encoder_split_at = position;
    }

    /// The arena's live-bytes high-water mark reached while it was built --
    /// the direct witness `build_buffer_arena`'s own `debug!` event already
    /// emits before allocating, kept queryable afterward for a census
    /// line. `None` when this feature is off, or when this plan has never
    /// executed a placed call and so never built its arena (see
    /// `Self::arena`'s own doc).
    #[must_use]
    pub fn arena_peak_bytes(&self) -> Option<usize> {
        #[cfg(feature = "metal-plan-stable-buffers")]
        {
            Some(self.arena.get()?.peak_bytes)
        }
        #[cfg(not(feature = "metal-plan-stable-buffers"))]
        {
            None
        }
    }

    /// Physical device buffers the arena actually holds -- `< op_count`
    /// whenever `build_buffer_arena`'s free list reused a slot across two
    /// or more non-overlapping positions. `None` when this feature is off,
    /// or when this plan has never built its arena.
    #[must_use]
    pub fn arena_slot_count(&self) -> Option<usize> {
        #[cfg(feature = "metal-plan-stable-buffers")]
        {
            Some(self.arena.get()?.slot_count())
        }
        #[cfg(not(feature = "metal-plan-stable-buffers"))]
        {
            None
        }
    }

    /// Total bytes retained by this plan's physical output slots. This is
    /// distinct from [`Self::arena_peak_bytes`]: the peak is a liveness
    /// high-water mark, while this sum is the device allocation kept alive
    /// for the plan's lifetime.
    #[must_use]
    pub fn arena_allocated_bytes(&self) -> Option<usize> {
        #[cfg(feature = "metal-plan-stable-buffers")]
        {
            Some(self.arena.get()?.slot_bytes.iter().sum())
        }
        #[cfg(not(feature = "metal-plan-stable-buffers"))]
        {
            None
        }
    }

    /// The byte length of the physical slot backing `position`'s output --
    /// `None` when this feature is off, `position` is out of range, or this
    /// plan has never built its arena.
    #[must_use]
    pub fn arena_position_byte_len(&self, position: usize) -> Option<usize> {
        #[cfg(feature = "metal-plan-stable-buffers")]
        {
            let arena = self.arena.get()?;
            let slot = *arena.position_slot.get(position)?;
            Some(arena.slot_byte_len(slot))
        }
        #[cfg(not(feature = "metal-plan-stable-buffers"))]
        {
            let _ = position;
            None
        }
    }

    /// The plan-cache key `resolve_steps` computes for each program
    /// position (`omega::msl::kernel_cache_key`'s own doc: the shared
    /// structural + compile-option identity of that position's resolved
    /// `BoundOp`, this plan's own [`MathMode`] included) -- exposed so a
    /// caller can assert exactly which compiled kernel variant a program
    /// resolves to (operand order, reduced axis, packed-row shape, math
    /// mode) without re-deriving `kernel_cache_key`'s own logic outside this
    /// crate.
    ///
    /// # Errors
    /// Propagates the same rejection `kernel_cache_key` raises for an
    /// unsupported dtype.
    pub fn kernel_keys(&self) -> Result<Vec<String>, MetalError> {
        self.prepared
            .resolved
            .iter()
            .map(|bound| {
                kernel_cache_key(bound, &self.packed_operands, self.numeric_policy)
                    .map(|mut key| {
                        key.push(self.math_mode.cache_token());
                        key
                    })
                    .map_err(MetalError::from)
            })
            .collect()
    }
}

/// Which of `block_nodes`' entries carry a codec [`crate::msl::emit`] has an
/// unpack kernel for (`Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q4_0`, `Q5_1`,
/// `Q5_0`, `Float16`, `BFloat16`), keyed to its [`Codec`] — the single place this crate
/// decides "packed AND which codec," shared by [`plan`] and [`prepare`] so
/// the two cannot drift on it. `Float16` earns a codec slot despite needing
/// no unpack FUNCTION (see `msl::FLOAT16_BLOCK_BYTES`'s own doc) because its
/// buffer still needs a non-`float`, non-`uchar` binding type -- `None`
/// would route it through `Float32`'s plain-array path and bind it as the
/// kernel's own accumulator type, which is wrong the moment a `Float16`
/// weight multiplies an `f32` activation.
pub(super) fn packed_operands_of(block_nodes: &[NodeId], blocks: &[QuantizedBlock<'_>]) -> PackedOperands {
    block_nodes
        .iter()
        .zip(blocks.iter())
        // decode-only codecs (CPU-only, see `proxima_tensor::cpu`) and the
        // two non-quantized carriers fall out of `Codec::
        // from_quantized_block` as `None`, exactly like before, and hit
        // `reject_unsupported_gpu_dtype`'s ordinary rejection rather than a
        // silent, wrong-shape upload.
        .filter_map(|(node, block)| crate::msl::codec_from_quantized_block(block).map(|codec| (*node, codec)))
        .collect()
}

/// The [`Codec`] one expert-table entry's block carries, restricted to
/// the four codecs mixed-expert lowering has a decoder for
/// (`Q2_K`/`Q3_K`/`Q4_K`/`Q6_K`) — the shared subset
/// [`expert_payload_descriptors`], [`selected_expert_payloads`], and
/// [`selected_expert_arena_descriptors`] each re-derived identically before
/// this. Any other block (including codecs [`Codec::
/// from_quantized_block`] itself recognizes, like `Q4_0`/`Q8_0`) is rejected
/// the same way an unrecognized block always was here.
fn expert_codec(node: NodeId, block: &QuantizedBlock<'_>) -> Result<Codec, MetalError> {
    match crate::msl::codec_from_quantized_block(block) {
        Some(codec @ (Codec::Q2K | Codec::Q3K | Codec::Q4K | Codec::Q6K)) => Ok(codec),
        _ => Err(MetalError::ExpertSourceUnsupported {
            node,
            reason: "mixed expert lowering only has Q2_K, Q3_K, Q4_K, and Q6_K decoders",
        }),
    }
}

/// Describes the borrowed payloads in one expert substitution table.
///
/// This is the lowering boundary for HOBBIT: the residency decision remains
/// a borrowed [`ExpertSource`] while Metal receives codec-aware byte spans.
/// Keeping the spans separate is important because Q2_K, Q4_K, and Q6_K
/// blocks do not have the same byte width; joining them into one packed operand would
/// make the second expert's address wrong.
///
/// # Errors
/// Returns the existing typed expert-source error when an entry is not a
/// packed codec. No bytes are copied.
pub fn expert_payload_descriptors(
    node: NodeId,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
) -> Result<Vec<ExpertPayloadDescriptor>, MetalError> {
    let mut descriptors = Vec::with_capacity(source.entries().len());
    let mut byte_offset = 0usize;
    for entry in source.entries() {
        let codec = expert_codec(node, &entry.block)?;
        let bytes = entry
            .block
            .packed_bytes()
            .ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert entries must use packed bytes",
            })?;
        let byte_length = bytes.len();
        descriptors.push(ExpertPayloadDescriptor {
            expert_index: u32::try_from(descriptors.len()).map_err(|_| {
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert index exceeds the Metal descriptor ABI",
                }
            })?,
            codec,
            byte_offset,
            byte_length,
            out_dim: entry.out_dim,
            in_dim: entry.in_dim,
            epoch: entry.epoch,
        });
        byte_offset =
            byte_offset
                .checked_add(byte_length)
                .ok_or(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert payload byte span overflowed",
                })?;
    }
    Ok(descriptors)
}

/// Builds the dense route descriptor table and a compact payload arena for a
/// source snapshot. The descriptor table keeps one record per original expert
/// index so the existing Metal gather ABI remains O(1); unselected records
/// carry no payload bytes and are never valid routed entries for this step.
pub fn selected_expert_payloads(
    node: NodeId,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
) -> Result<(Vec<u8>, Vec<ExpertPayloadDescriptor>), MetalError> {
    let selected_ids = source.selected_expert_ids();
    if selected_ids.is_some_and(|ids| ids.is_empty()) {
        return Err(MetalError::ExpertSourceUnsupported {
            node,
            reason: "selected expert ID list is empty",
        });
    }
    let mut payload_bytes = Vec::new();
    let mut descriptors = Vec::with_capacity(source.entries().len());
    for (expert_index, entry) in source.entries().iter().enumerate() {
        let codec = expert_codec(node, &entry.block)?;
        let bytes = entry
            .block
            .packed_bytes()
            .ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert entries must use packed bytes",
            })?;
        let selected =
            selected_ids.is_none_or(|ids| ids.contains(&(expert_index as u32)));
        let (byte_offset, byte_length) = if selected {
            let byte_offset = payload_bytes.len();
            payload_bytes.extend_from_slice(bytes);
            (byte_offset, bytes.len())
        } else {
            (0, 0)
        };
        descriptors.push(ExpertPayloadDescriptor {
            expert_index: u32::try_from(expert_index).map_err(|_| {
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert index exceeds the Metal descriptor ABI",
                }
            })?,
            codec,
            byte_offset,
            byte_length,
            out_dim: entry.out_dim,
            in_dim: entry.in_dim,
            epoch: entry.epoch,
        });
    }
    if let Some(ids) = selected_ids {
        for id in ids {
            if usize::try_from(*id)
                .ok()
                .is_none_or(|index| index >= source.entries().len())
            {
                return Err(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "selected expert ID is outside the source table",
                });
            }
        }
    }
    Ok((payload_bytes, descriptors))
}

pub(super) fn selected_expert_arena_descriptors(
    node: NodeId,
    source: &proxima_tensor::cpu::ExpertSource<'_>,
    arena: proxima_tensor::cpu::ExpertPayloadArena<'_>,
) -> Result<Vec<ExpertPayloadDescriptor>, MetalError> {
    let selected_ids = source.selected_expert_ids();
    let spans = arena.spans();
    let mut descriptors = Vec::with_capacity(source.entries().len());
    for (expert_index, entry) in source.entries().iter().enumerate() {
        let codec = expert_codec(node, &entry.block)?;
        let selected = selected_ids.is_none_or(|ids| {
            ids.iter()
                .any(|id| *id == u32::try_from(expert_index).unwrap_or(u32::MAX))
        });
        let span = spans.get(expert_index).and_then(|span| *span);
        let (byte_offset, byte_length) = if selected {
            let span = span.ok_or(MetalError::ExpertSourceUnsupported {
                node,
                reason: "selected expert has no arena span",
            })?;
            let end = usize::try_from(span.offset)
                .ok()
                .and_then(|offset| {
                    usize::try_from(span.length)
                        .ok()
                        .and_then(|length| offset.checked_add(length))
                })
                .ok_or(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert arena span endpoint overflowed",
                })?;
            if end > arena.bytes().len() {
                return Err(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert arena span exceeds payload bytes",
                });
            }
            (
                usize::try_from(span.offset).unwrap_or(usize::MAX),
                usize::try_from(span.length).unwrap_or(usize::MAX),
            )
        } else {
            (0, 0)
        };
        descriptors.push(ExpertPayloadDescriptor {
            expert_index: u32::try_from(expert_index).map_err(|_| {
                MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "expert index exceeds the Metal descriptor ABI",
                }
            })?,
            codec,
            byte_offset,
            byte_length,
            out_dim: entry.out_dim,
            in_dim: entry.in_dim,
            epoch: entry.epoch,
        });
    }
    Ok(descriptors)
}

/// Packs descriptor records into the byte layout consumed by the MSL
/// `ExpertPayloadDescriptor` struct. The caller owns the returned bytes and
/// may upload them beside the borrowed payload arena without touching the
/// tensor graph or changing the residency FSM.
pub fn pack_expert_payload_descriptors(
    node: NodeId,
    descriptors: &[ExpertPayloadDescriptor],
) -> Result<Vec<u8>, MetalError> {
    let mut bytes = Vec::with_capacity(descriptors.len() * 32);
    for descriptor in descriptors {
        let codec_tag: u32 = match descriptor.codec {
            Codec::Q2K => 1,
            Codec::Q3K => 4,
            Codec::Q4K => 2,
            Codec::Q6K => 3,
            _ => {
                return Err(MetalError::ExpertSourceUnsupported {
                    node,
                    reason: "mixed expert descriptor has no MSL decoder",
                });
            }
        };
        let byte_offset = u32::try_from(descriptor.byte_offset).map_err(|_| {
            MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert payload byte offset exceeds the Metal descriptor ABI",
            }
        })?;
        let byte_length = u32::try_from(descriptor.byte_length).map_err(|_| {
            MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert payload byte length exceeds the Metal descriptor ABI",
            }
        })?;
        let epoch =
            u32::try_from(descriptor.epoch).map_err(|_| MetalError::ExpertSourceUnsupported {
                node,
                reason: "expert promotion epoch exceeds the Metal descriptor ABI",
            })?;
        bytes.extend_from_slice(&descriptor.expert_index.to_ne_bytes());
        bytes.extend_from_slice(&codec_tag.to_ne_bytes());
        bytes.extend_from_slice(&byte_offset.to_ne_bytes());
        bytes.extend_from_slice(&byte_length.to_ne_bytes());
        bytes.extend_from_slice(&descriptor.out_dim.to_ne_bytes());
        bytes.extend_from_slice(&descriptor.in_dim.to_ne_bytes());
        bytes.extend_from_slice(&epoch.to_ne_bytes());
        bytes.extend_from_slice(&0_u32.to_ne_bytes());
    }
    Ok(bytes)
}

