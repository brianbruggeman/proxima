use super::*;

/// `PROXIMA_COMMAND_BUFFER_CHUNKS=K`, parsed and cached once per process
/// (matching every other `PROXIMA_*` knob this crate reads, e.g.
/// `packed_rows_override` in `kernel_types_identity.rs`) -- unlike the
/// config/plan-shape tier below, the env override is a genuine per-process
/// A/B switch, so caching it once is correct. Only a positive integer
/// literal is honored; unset, empty, zero, or unparsable is `None`, falling
/// through to the per-call config/default tier.
fn command_buffer_chunk_env_override() -> Option<usize> {
    static CACHED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("PROXIMA_COMMAND_BUFFER_CHUNKS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|parsed| *parsed >= 1)
    })
}

/// `PROXIMA_COMMAND_BUFFER_BOUNDARIES=<comma-separated op positions>`
/// (OWNER_BRIEF_largest_region, Step B): an explicit chunk-boundary list for
/// coarse per-region GPU timing, read and cached once like every other
/// `PROXIMA_*` knob this module reads. Only consulted when
/// `PROXIMA_COMMAND_BUFFER_CHUNKS` is unset (that env var's even-split path
/// stays authoritative when both are set) and only for decode-shaped plans
/// -- prefill keeps its existing `1` boundary count, same scoping as
/// `command_buffer_chunk_count`'s own doc. A boundary at position `p` starts
/// a new command buffer whose first op is `p`, reusing
/// [`command_buffer_chunk_boundaries`]'s own convention.
fn command_buffer_explicit_boundaries_env() -> Option<Vec<usize>> {
    static CACHED: std::sync::OnceLock<Option<Vec<usize>>> = std::sync::OnceLock::new();
    CACHED
        .get_or_init(|| {
            std::env::var("PROXIMA_COMMAND_BUFFER_BOUNDARIES")
                .ok()
                .map(|value| {
                    value
                        .split(',')
                        .filter_map(|entry| entry.trim().parse::<usize>().ok())
                        .collect::<Vec<usize>>()
                })
        })
        .clone()
}

/// `PROXIMA_COMMAND_BUFFER_CHUNKS=K` (OWNER_BRIEF_structural_difference,
/// Intervention 6): same-binary A/B for how many `MTLCommandBuffer`s one
/// [`execute_plan_with_placements_inner`] call splits its dispatch sequence
/// into. Resolved PER CALL, not cached -- the owner's own integration
/// scoping: the measured win covers only the single-new-token decode plan,
/// never prefill, so `plan_chunks`/`decode_shaped`
/// ([`Plan::command_buffer_chunks`]/[`Plan::command_buffer_chunks_decode_shaped`],
/// both threaded through `BackendRuntime`/`PlanNumerics` the same path
/// `ServingConfig::plan_time_constants` takes to
/// [`Plan::mark_plan_time_constants_resident`]) can legitimately differ
/// between two calls against the SAME process, one per new-token-count
/// plan shape. `decode_shaped == false` (prefill, or a plan built outside
/// the config-threading call sites) always resolves to `1`
/// (`source=default`) regardless of `plan_chunks`. `decode_shaped == true`
/// resolves `plan_chunks` through `config_or_default_chunks`
/// (`source=config` when a caller threaded a nonzero value, `source=default`
/// otherwise). The env var (`source=env`, [`command_buffer_chunk_env_override`])
/// wins unconditionally over both.
fn command_buffer_chunk_count(plan_chunks: u32, decode_shaped: bool) -> usize {
    let (chunks, source) = match command_buffer_chunk_env_override() {
        Some(parsed) => (parsed, "env"),
        None if decode_shaped => config_or_default_chunks(plan_chunks),
        None => (1, "default"),
    };
    // read unconditionally so a non-`instrument` build (where the only
    // consumer below is compiled out) does not trip `unused_variables`.
    let _ = source;
    #[cfg(feature = "instrument")]
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
        let plan_shape = if decode_shaped { "decode" } else { "prefill" };
        eprintln!("command_buffer_chunks={chunks} source={source} plan_shape={plan_shape}");
    }
    chunks
}

/// `plan_chunks == 0` (no caller threaded a config value) falls back to the
/// internal default of `1`; a nonzero value is the config-sourced tier.
/// Only consulted for a decode-shaped plan -- see
/// [`command_buffer_chunk_count`]'s own doc.
fn config_or_default_chunks(plan_chunks: u32) -> (usize, &'static str) {
    if plan_chunks == 0 {
        (1, "default")
    } else {
        (plan_chunks as usize, "config")
    }
}

/// Chunk boundary positions (program-order op indices) splitting
/// `total_ops` resolved ops into `chunk_count` contiguous groups as evenly
/// as an integer split allows: boundary `i` sits at
/// `floor(i * total_ops / chunk_count)`. Returns the empty vector for
/// `chunk_count <= 1` or `total_ops == 0` -- a `1..1` range has a
/// `(0, Some(0))` size hint, so `.collect()` never allocates, which is what
/// makes the default `PROXIMA_COMMAND_BUFFER_CHUNKS` unset path provably
/// free of the extra command-buffer machinery below rather than merely
/// untested. Duplicate boundary values (K exceeding `total_ops`) collapse
/// to fewer than `chunk_count - 1` entries instead of producing an empty
/// chunk.
fn command_buffer_chunk_boundaries(total_ops: usize, chunk_count: usize) -> Vec<usize> {
    if chunk_count <= 1 || total_ops == 0 {
        return Vec::new();
    }
    let mut boundaries = Vec::with_capacity(chunk_count - 1);
    let mut previous = 0usize;
    for i in 1..chunk_count {
        let boundary = (i * total_ops) / chunk_count;
        if boundary > previous && boundary < total_ops {
            boundaries.push(boundary);
            previous = boundary;
        }
    }
    boundaries
}

/// `PROXIMA_CHUNK_AUDIT=1` (OWNER_BRIEF_chunked_submission_audit): a per-thread
/// record of every CPU-side write into a Metal buffer's contents and every
/// dispatch's bound ranges, tagged with which K-chunk each belongs to --
/// built specifically to answer the audit's own question (does a chunked
/// commit let the CPU write memory an already-committed, possibly still
/// in-flight, chunk reads) from EXECUTION DATA rather than from reading the
/// encode loop alone. Read once per process like every other `PROXIMA_*`
/// knob this module already caches (`command_buffer_chunk_count`'s own doc).
#[cfg(feature = "instrument")]
struct ChunkAuditWrite {
    buffer_ptr: usize,
    offset: usize,
    len: usize,
    position: usize,
    committed_chunks: usize,
}

#[cfg(feature = "instrument")]
struct ChunkAuditDispatch {
    buffer_ptr: usize,
    offset: usize,
    len: usize,
    position: usize,
    chunk_index: usize,
    is_write: bool,
}

#[cfg(feature = "instrument")]
#[derive(Default)]
struct ChunkAuditState {
    current_chunk: usize,
    committed_chunks: usize,
    writes: Vec<ChunkAuditWrite>,
    dispatches: Vec<ChunkAuditDispatch>,
}

#[cfg(feature = "instrument")]
thread_local! {
    static CHUNK_AUDIT_ENABLED: core::cell::Cell<Option<bool>> = const { core::cell::Cell::new(None) };
    static CHUNK_AUDIT_STATE: core::cell::RefCell<ChunkAuditState> = const {
        core::cell::RefCell::new(ChunkAuditState {
            current_chunk: 1,
            committed_chunks: 0,
            writes: Vec::new(),
            dispatches: Vec::new(),
        })
    };
}

#[cfg(feature = "instrument")]
pub(super) fn chunk_audit_enabled() -> bool {
    CHUNK_AUDIT_ENABLED.with(|flag| {
        if let Some(value) = flag.get() {
            return value;
        }
        let value = std::env::var_os("PROXIMA_CHUNK_AUDIT").is_some();
        flag.set(Some(value));
        value
    })
}

/// Resets per-step audit state -- called once at the top of
/// [`execute_plan_with_placements_inner`], before this step's chunk
/// boundaries are even computed, so a warm call's leftover records from the
/// PRIOR decode step never bleed into this step's hazard intersection.
#[cfg(feature = "instrument")]
fn chunk_audit_begin_step() {
    if !chunk_audit_enabled() {
        return;
    }
    CHUNK_AUDIT_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.current_chunk = 1;
        state.committed_chunks = 0;
        state.writes.clear();
        state.dispatches.clear();
    });
}

/// Called right after an intermediate chunk's `commit()` -- marks that
/// chunk's dispatches as "already submitted to the queue" for every write
/// recorded from this point on, and advances `current_chunk` for every
/// dispatch recorded from this point on.
#[cfg(feature = "instrument")]
fn chunk_audit_on_commit() {
    if !chunk_audit_enabled() {
        return;
    }
    CHUNK_AUDIT_STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.committed_chunks = state.current_chunk;
        state.current_chunk += 1;
    });
}

/// Records one CPU write range into a Metal buffer's `contents()`. Writes
/// before the first commit (`committed_chunks == 0`) are dropped, matching
/// the audit brief's own scope ("every CPU write range... performed after
/// the FIRST commit") -- a write with nothing yet in flight cannot race
/// anything, and recording it would only pad `writes=` with noise the
/// hazard intersection below never uses (`committed_chunks == 0` can never
/// satisfy `dispatch.chunk_index <= write.committed_chunks` for a real
/// dispatch, since chunks are numbered from 1).
#[cfg(feature = "instrument")]
pub(super) fn chunk_audit_record_write(buffer_ptr: usize, offset: usize, len: usize, position: usize) {
    if !chunk_audit_enabled() {
        return;
    }
    CHUNK_AUDIT_STATE.with(|state| {
        let mut state = state.borrow_mut();
        if state.committed_chunks == 0 {
            return;
        }
        let committed_chunks = state.committed_chunks;
        state.writes.push(ChunkAuditWrite {
            buffer_ptr,
            offset,
            len,
            position,
            committed_chunks,
        });
    });
}

/// Records one dispatch's bound range (an input/indices read or the op's own
/// output write), tagged with the chunk it was encoded into.
#[cfg(feature = "instrument")]
pub(super) fn chunk_audit_record_dispatch(
    buffer_ptr: usize,
    offset: usize,
    len: usize,
    position: usize,
    is_write: bool,
) {
    if !chunk_audit_enabled() {
        return;
    }
    CHUNK_AUDIT_STATE.with(|state| {
        let mut state = state.borrow_mut();
        let chunk_index = state.current_chunk;
        state.dispatches.push(ChunkAuditDispatch {
            buffer_ptr,
            offset,
            len,
            position,
            chunk_index,
            is_write,
        });
    });
}

/// Intersects this step's recorded writes against this step's recorded
/// dispatches: a write recorded after chunk `c` committed, overlapping a
/// range some dispatch belonging to chunk `<= c` binds (read OR write --
/// both are a race, a WAR as much as a RAW), is a hazard. Printed
/// unconditionally when the audit is enabled, `hazards=0` included, so a
/// clean run is a MEASURED zero (`writes=`/`dispatches=` prove the counts
/// were non-trivial) rather than an absence of output.
#[cfg(feature = "instrument")]
pub(super) fn chunk_audit_finish(step: u64) {
    if !chunk_audit_enabled() {
        return;
    }
    CHUNK_AUDIT_STATE.with(|state| {
        let state = state.borrow();
        let mut hazards: Vec<String> = Vec::new();
        for write in &state.writes {
            for dispatch in &state.dispatches {
                if dispatch.chunk_index > write.committed_chunks {
                    continue;
                }
                let overlaps = dispatch.buffer_ptr == write.buffer_ptr
                    && dispatch.offset < write.offset + write.len
                    && write.offset < dispatch.offset + dispatch.len;
                if overlaps {
                    hazards.push(format!(
                        "write_position={} write_committed_chunks={} dispatch_position={} \
                         dispatch_chunk={} dispatch_is_write={} buffer_ptr={:#x} \
                         write_offset={} write_len={} dispatch_offset={} dispatch_len={}",
                        write.position,
                        write.committed_chunks,
                        dispatch.position,
                        dispatch.chunk_index,
                        dispatch.is_write,
                        write.buffer_ptr,
                        write.offset,
                        write.len,
                        dispatch.offset,
                        dispatch.len,
                    ));
                }
            }
        }
        eprintln!(
            "chunk_audit step={step} writes={} dispatches={} hazards={}",
            state.writes.len(),
            state.dispatches.len(),
            hazards.len()
        );
        for hazard in hazards.iter().take(3) {
            eprintln!("chunk_audit_hazard step={step} {hazard}");
        }
    });
}

/// `recycle` is the same pool [`proxima_tensor::cpu::evaluate_with_scratch`]
/// takes: a caller done reading a PREVIOUS call's `Evaluated` hands its
/// storage back with [`Evaluated::into_scratch`], and this call pops one
/// buffer from the pool (if any) to reuse for the root output's read-back
/// instead of allocating fresh. An empty pool (every call before this
/// landing was, implicitly) always allocates fresh, exactly as before -- see
/// `finish`'s own doc for the exact conditions a popped buffer is actually
/// reused under.
///
/// # Errors
/// Propagates block-codec and Metal driver failures, same as [`execute_plan`].
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_with_placements(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
    recycle: &mut Vec<Vec<f32>>,
) -> Result<Evaluated, MetalError> {
    execute_plan_with_placements_inner(
        plan,
        blocks,
        input_placements,
        output_placements,
        recycle,
        &BTreeMap::new(),
    )
}

#[cfg(feature = "metal-output-placement")]
pub(super) fn execute_plan_with_placements_inner(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
    recycle: &mut Vec<Vec<f32>>,
    expert_buffers: &BTreeMap<NodeId, ExpertSourceBuffers>,
) -> Result<Evaluated, MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;
    let mut effective_expert_buffers = expert_buffers.clone();
    for (source_node, buffers) in expert_buffers {
        let Some(source_name) = plan.program[source_node.0 as usize].name() else {
            continue;
        };
        for (position, operation) in plan.program.iter().enumerate() {
            if operation.name() == Some(source_name) {
                effective_expert_buffers
                    .entry(NodeId(position as u32))
                    .or_insert_with(|| buffers.clone());
            }
        }
    }
    let input_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = input_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    if std::env::var_os("PROXIMA_DEBUG_PLACEMENT_KEYS").is_some() {
        eprintln!(
            "metal placement keys inputs={:?} blocks={:?}",
            input_placed.keys().collect::<Vec<_>>(),
            prepared.block_nodes,
        );
    }
    let output_placed: BTreeMap<NodeId, (&PlacedBuffer, usize)> = output_placements
        .iter()
        .map(|(node, buffer, offset)| (*node, (*buffer, *offset)))
        .collect();
    let (device, queue) = device_and_queue()?;
    // On a plan-cache HIT this is a no-op (`resolve_steps` checks staleness
    // and returns immediately): the per-position loop below indexes
    // `plan.resolved_steps` instead of every step re-deriving
    // `kernel_cache_key`/`kernel_dispatch_shape`/`pipeline_for`'s own cache
    // key. Only a genuine plan-cache MISS or a `set_math_mode` change pays
    // this once, up front, rather than once per op per step.
    resolve_steps(&device, plan)?;
    let resolved_steps = plan.resolved_steps.borrow();

    // plan-owned and never rebuilt fresh -- see `Plan::device_buffers`'s own
    // doc for why this alone (independent of the per-node reuse skip below)
    // already removes ROW 303's residual `BTreeMap::new()` allocation on a
    // warm call: every key this loop and the position loop below touch is
    // still written fresh every call exactly as before, only the map's own
    // heap nodes now outlive one call instead of being dropped with it.
    let mut device_buffers = plan.device_buffers.borrow_mut();
    for (node, buffers) in &effective_expert_buffers {
        device_buffers.insert(*node, (buffers.payloads.clone(), buffers.payload_offset));
    }
    let mut block_identity = plan.block_identity.borrow_mut();
    if block_identity.len() != prepared.block_nodes.len() {
        block_identity.resize(prepared.block_nodes.len(), None);
    }
    for (index, ((node, block), dtype)) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
        .enumerate()
    {
        if !prepared.live_block_inputs[index] {
            block_identity[index] = None;
            continue;
        }
        if effective_expert_buffers.contains_key(node) {
            block_identity[index] = None;
            continue;
        }
        // an input-placed node skips the host round trip entirely: its
        // buffer is already on the device, owned by the caller, and this
        // call's own `block` entry for it (still required, positionally, to
        // keep `block_nodes.iter().zip(blocks.iter())` aligned) is unused.
        // The caller's own offset travels in the same `DeviceBuffer` tuple
        // every other node's buffer carries -- `buffer_for`/`bind_buffers`
        // read it generically, with no separate offset map required.
        if let Some((buffer, offset)) = input_placed.get(node) {
            device_buffers.insert(*node, ((*buffer).clone(), *offset));
            block_identity[index] = None;
            continue;
        }
        // The source table is the typed execution boundary.  Only omit the
        // packed expert stack when this call actually supplies substituted
        // expert buffers; configuration belongs to the serving plan, not an
        // ambient environment read in the Metal backend.  With no source
        // table, bind the checkpoint's ordinary expert blocks unchanged.
        if !effective_expert_buffers.is_empty()
            && plan.program[node.0 as usize]
                .name()
                .is_some_and(|name| name.contains("_exps.weight"))
        {
            // Router segments carry the full program input inventory, but
            // expert bytes are supplied only by the gather source table.
            // Do not bind the original packed stack on this placement path.
            block_identity[index] = None;
            continue;
        }
        let current_identity = block_identity_key(block);
        let resident = plan.resident_nodes.contains(node);
        #[cfg(feature = "instrument")]
        if let Some(previous_identity) = block_identity[index]
            && previous_identity != current_identity
        {
            counter!(MAPPING_REBOUND_BLOCKS, 1);
            debug!(
                block_name = %plan.program[node.0 as usize].name().unwrap_or("<unnamed>"),
                previous_pointer = previous_identity.0 as u64,
                previous_length = previous_identity.1 as u64,
                current_pointer = current_identity.0 as u64,
                current_length = current_identity.1 as u64,
                resident,
                "mapping_rebound_block: block_identity_key disagreed with the prior step's"
            );
        }
        if block_buffer_reusable(resident, block_identity[index], current_identity)
            && device_buffers.contains_key(node)
        {
            // this position's buffer is already correct in the plan-owned
            // map from a PRIOR call (never retired -- see the retirement
            // loop's own `resident_nodes` skip below) -- no upload, no
            // `device_buffers` write, this call's only cost for this node
            // is the identity comparison just made.
            continue;
        }
        block_identity[index] = Some(current_identity);
        let resident_name = resident_name(plan, *node);
        // a brand-new `Plan` (a KV-bucket-boundary reshape, say) starts with
        // an empty `device_buffers`/`block_identity` of its OWN, so the
        // fast-skip above always misses on plan 1 of a node's life even
        // though the GLOBAL, name-keyed caches
        // (`NOCOPY_BUFFERS`/`RESIDENT_BUFFERS`/the checkpoint mapping)
        // already hold this exact host range from the PRIOR plan. Consult
        // them here, before counting this node as "offered", so a reshape
        // does not re-walk every resident weight block through the upload
        // path just to have it resolve to the same cache hit it already was.
        if let Some((buffer, offset)) = cross_plan_resident_reuse(
            resident_name,
            current_identity.0 as *const c_void,
            current_identity.1,
        )? {
            device_buffers.insert(*node, (buffer, offset));
            continue;
        }
        // an input-placed node above never reaches here, so `BLOCK_OFFERED_BYTES`
        // fires only for a block that genuinely takes the host round trip
        // below -- the same "offered for upload" meaning `execute_plan`'s own
        // fire site carries, extended to this placed-buffer entry point so
        // the partition identity (`BLOCK_COPIED_BYTES + BLOCK_NOCOPY_BOUND_BYTES
        // + BLOCK_OFFSET_BOUND_BYTES == BLOCK_OFFERED_BYTES`) holds on this
        // loop too, not just `execute_plan`'s.
        #[cfg(feature = "instrument")]
        {
            counter!(BLOCK_UPLOAD_CALLS, 1);
            counter!(BLOCK_OFFERED_BYTES, block_byte_len(block) as u64);
        }
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                *node,
                *dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => {
                upload_block_int32_as_float(&device, data, resident_name)?
            }
            // `Q3_K` uploads its raw super-block bytes unchanged, same as
            // every other packed codec below -- `msl::Codec::Q3K`'s
            // own unpack kernel (`q3k_element`) reads them at the GPU side.
            QuantizedBlock::Packed { bytes, .. } => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(*node, buffer);
    }

    #[cfg(feature = "instrument")]
    let kind_filter = KindFilter::from_env()?;
    #[cfg(feature = "instrument")]
    if let Some(filter) = &kind_filter {
        validate_kind_filter(filter, prepared, packed_operands, &plan.program)?;
    }

    #[cfg(feature = "instrument")]
    chunk_audit_begin_step();
    let mut command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;
    // `Concurrent` lets independent dispatches (e.g. Q/K/V from one normed
    // input) overlap instead of draining the pipeline between every op --
    // see [`DispatchType`]'s own doc for why that requires [`HazardTracker`]
    // below to insert the barriers `Serial` gives for free by never
    // overlapping any two dispatches in the first place.
    let dispatch_type = plan.dispatch_type;
    let mut encoder = EncoderGuard::new(
        command_buffer
            .computeCommandEncoderWithDispatchType(dispatch_type.as_mtl())
            .ok_or_else(|| MetalError::CompileFailed {
                log: "command buffer refused to hand out a compute encoder".to_string(),
            })?,
    );
    // attn_parity followon (2026-09-22, OWNER_BRIEF_structural_difference):
    // command buffers submitted on one `MTLCommandQueue` execute in commit
    // order (Apple's Metal Programming Guide, "Command Queue" -- a queue
    // schedules the command buffers committed to it in the order `commit`
    // was called), so splitting one token's dispatch sequence into K
    // buffers, each committed as soon as it is encoded, needs no hazard or
    // barrier change: chunk i+1's dispatches still run only after every
    // dispatch chunk i wrote finishes, exactly as today's single buffer
    // orders them. `chunk_boundaries` is empty for the default K=1 (the
    // `1..1` range below has a `(0, Some(0))` size hint, so `.collect()`
    // allocates nothing), so the boundary check inside the loop never
    // matches and this call sequence is provably identical to before this
    // change for every existing caller.
    let total_ops = prepared.resolved.len();
    // OWNER_BRIEF_largest_region, Step B: an explicit boundary list wins over
    // the even-split path only when `PROXIMA_COMMAND_BUFFER_CHUNKS` is unset
    // and the plan is decode-shaped -- same scoping
    // `command_buffer_chunk_count` already applies to the config/default
    // tier, extended to this diagnostic-only env var.
    // `PROXIMA_BOUNDARIES_PREFILL=1` (OWNER_BRIEF_prefill_correction, coordinator
    // addition 2026-09-23): lets the explicit-boundary diagnostic also split a
    // prefill-shaped plan's step-0 command stream -- off by default so the
    // decode-only scoping above is unchanged for every existing caller.
    let boundaries_prefill_allowed =
        plan.command_buffer_chunks_decode_shaped || std::env::var_os("PROXIMA_BOUNDARIES_PREFILL").is_some();
    let explicit_boundaries = if command_buffer_chunk_env_override().is_none() && boundaries_prefill_allowed {
        command_buffer_explicit_boundaries_env()
    } else {
        None
    };
    let (chunk_boundaries, chunk_count, ignored_boundaries) =
        if let Some(positions) = explicit_boundaries {
            let mut boundaries: Vec<usize> = positions
                .iter()
                .copied()
                .filter(|position| *position > 0 && *position < total_ops)
                .collect();
            boundaries.sort_unstable();
            boundaries.dedup();
            let ignored = positions.len().saturating_sub(boundaries.len());
            let count = boundaries.len() + 1;
            (boundaries, count, ignored)
        } else {
            let count = command_buffer_chunk_count(
                plan.command_buffer_chunks,
                plan.command_buffer_chunks_decode_shaped,
            );
            (command_buffer_chunk_boundaries(total_ops, count), count, 0)
        };
    // read unconditionally so a non-`instrument` build (where the only
    // consumers are the `chunk_record`/`chunk_summary` prints below) does not
    // trip `unused_variables`, matching `command_buffer_chunk_count`'s own
    // `source`.
    let _ = (chunk_count, ignored_boundaries);
    let mut next_boundary = 0usize;
    #[cfg(feature = "instrument")]
    let mut first_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>> = None;
    #[cfg(feature = "instrument")]
    let mut first_commit_call_start_s: Option<f64> = None;
    #[cfg(feature = "instrument")]
    let mut post_first_commit_encode_started: Option<std::time::Instant> = None;
    // Deliverable 2 (OWNER_BRIEF_chunked_submission_audit): one record per
    // committed command buffer, host-clock start/end for the encode work
    // and commit call, GPU-clock start/end read from the buffer itself --
    // all `command_buffer`s stay alive (cloned `Retained` handles) until
    // AFTER the step's one `waitUntilCompleted`, so `GPUStartTime`/
    // `GPUEndTime` are valid to read for every chunk, not only the last
    // one waited on directly (Metal fills them in as each buffer completes;
    // completion happens in commit order on one serial queue, so the final
    // wait returning proves every earlier chunk already finished too).
    #[cfg(feature = "instrument")]
    let step_encode_start = std::time::Instant::now();
    #[cfg(feature = "instrument")]
    let step_encode_start_ticks = read_ticks();
    #[cfg(feature = "instrument")]
    let mut chunk_command_buffers: Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>> = Vec::new();
    #[cfg(feature = "instrument")]
    struct ChunkHostTiming {
        first_op: usize,
        last_op: usize,
        encode_start_ms: f64,
        encode_end_ms: f64,
        commit_ms: f64,
    }
    #[cfg(feature = "instrument")]
    let mut chunk_host_timings: Vec<ChunkHostTiming> = Vec::new();
    #[cfg(feature = "instrument")]
    let mut current_chunk_first_op = 0usize;
    #[cfg(feature = "instrument")]
    let mut current_chunk_encode_start_ms = 0.0f64;
    // plan-owned, reused across every call against this `Plan` (`HazardState`'s
    // own doc) -- `reset` clears both the tracker's sets and the input scratch
    // without releasing their capacity, so a warm call after the first never
    // pays their first-insert allocation again.
    let mut hazard_state = plan.hazard_state.borrow_mut();
    hazard_state.reset();
    // reborrowed as a plain `&mut` so `tracker`/`inputs` split into disjoint
    // field borrows below -- the borrow checker cannot see through
    // `RefMut`'s own `Deref`/`DerefMut` to know the two fields are disjoint.
    let hazard_state = &mut *hazard_state;

    // Resolved once per plan, lazily: needs only `device_buffers`' weight
    // identity (ROW 572 -- activation admits by NODE identity, output is
    // allocated at the leader's own first encode, never at plan-resolution
    // time, since `BufferArena` cannot hand out a slot shared across two
    // simultaneously-live positions; see `ensure_merged_group_resolved`).
    // A plan with no eligible group still costs one `Option::is_some()`
    // check per call after the first.
    #[cfg(feature = "metal-horizontal-merge")]
    ensure_merged_dispatches(&device, plan, &device_buffers)?;

    // diagnostic-only, `instrument`-gated, always emitted at `trace` level
    // (default-off via the runtime filter, never a bespoke env var): each
    // output-placed node's WRITE position and each of its aliased readers'
    // own position, the direct evidence a write-before-read ordering claim
    // needs (this function's own doc, "Within-call aliasing"). Silent unless
    // a caller raises `RUST_LOG` to `trace` for this target.
    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();
    for (position, bound) in prepared.resolved.iter().enumerate() {
        if next_boundary < chunk_boundaries.len() && chunk_boundaries[next_boundary] == position {
            let new_command_buffer =
                queue
                    .commandBuffer()
                    .ok_or_else(|| MetalError::CompileFailed {
                        log: "command queue refused to hand out a command buffer".to_string(),
                    })?;
            let new_encoder = EncoderGuard::new(
                new_command_buffer
                    .computeCommandEncoderWithDispatchType(dispatch_type.as_mtl())
                    .ok_or_else(|| MetalError::CompileFailed {
                        log: "command buffer refused to hand out a compute encoder".to_string(),
                    })?,
            );
            #[cfg(feature = "instrument")]
            if first_command_buffer.is_none() {
                first_command_buffer = Some(command_buffer.clone());
            }
            let closing_encoder = core::mem::replace(&mut encoder, new_encoder);
            let closing_command_buffer =
                core::mem::replace(&mut command_buffer, new_command_buffer);
            closing_encoder.finish();
            #[cfg(feature = "instrument")]
            let closing_encode_end_ms = step_encode_start.elapsed().as_secs_f64() * 1e3;
            #[cfg(feature = "instrument")]
            chunk_command_buffers.push(closing_command_buffer.clone());
            // Same clock-correlation argument as the single-buffer
            // `gpu_exec_started`/`commit_call_start_s` pair below: a tick
            // read immediately before `commit()`, converted through the
            // same `ticks_to_nanos`, is what's comparable to
            // `GPUStartTime`'s `mach_absolute_time`-based clock.
            #[cfg(feature = "instrument")]
            let this_commit_ticks = read_ticks();
            #[cfg(feature = "instrument")]
            let commit_call_started = std::time::Instant::now();
            closing_command_buffer.commit();
            #[cfg(feature = "instrument")]
            let closing_commit_ms = commit_call_started.elapsed().as_secs_f64() * 1e3;
            #[cfg(feature = "instrument")]
            {
                chunk_host_timings.push(ChunkHostTiming {
                    first_op: current_chunk_first_op,
                    last_op: position.saturating_sub(1),
                    encode_start_ms: current_chunk_encode_start_ms,
                    encode_end_ms: closing_encode_end_ms,
                    commit_ms: closing_commit_ms,
                });
                current_chunk_first_op = position;
                current_chunk_encode_start_ms = step_encode_start.elapsed().as_secs_f64() * 1e3;
            }
            #[cfg(feature = "instrument")]
            chunk_audit_on_commit();
            #[cfg(feature = "instrument")]
            if first_commit_call_start_s.is_none() {
                first_commit_call_start_s = Some(
                    proxima_tensor::instrument::ticks_to_nanos(this_commit_ticks.as_raw()) as f64
                        / 1e9,
                );
                post_first_commit_encode_started = Some(std::time::Instant::now());
            }
            next_boundary += 1;
        }
        // attribution2 followon: the trace-gate loop below unconditionally
        // walks `bound.operands()` and does an `input_placed`/`output_placed`
        // `BTreeMap::contains_key` per operand on EVERY `instrument` build,
        // regardless of whether `trace` level is enabled -- `trace!`'s own
        // callsite check is inside the `if`, not around it, so the lookup
        // cost is paid unconditionally at `debug`-level filtering
        // (`decode_gbps_baseline`'s own hardcoded `EnvFilter::parse("debug")`
        // excludes `trace`). Timed as `LOOP_HEAD_TICKS` together with the
        // `arena_placement` resolution just below, since both run before
        // `PLACEMENT_RESOLVE_TICKS` starts and neither has its own counter.
        #[cfg(feature = "instrument")]
        let loop_head_started = read_ticks();
        #[cfg(feature = "instrument")]
        {
            if output_placed.contains_key(&bound.node) {
                trace!(position, node = ?bound.node, "output-placed node write");
            }
            for (operand, _, _) in bound.operands() {
                if input_placed.contains_key(operand) {
                    trace!(position, node = ?bound.node, reads = ?operand, "placed-input node read");
                }
            }
        }
        // attn_parity followon (2026-09-23, OWNER_BRIEF_largest_region Step C):
        // `PROXIMA_DEBUG_RESOLVED_OPS=1` prints one `resolved_op` line per
        // resolved position, ONCE per process (the `AtomicBool` below), not
        // once per decode step -- the resolved list (node ids, kind, operand
        // ids, extents) is identical across every decode step against a
        // fixed-shape plan, so a single dump at the first call already
        // describes every later step. `entry`/`grid` are NOT included here:
        // they are only known at encode time (see `capture_dispatch`,
        // `arena_encode_dispatch_finish.rs`), which this print does not
        // duplicate -- join on `node` against a `PROXIMA_CAPTURE_NODES`-gated
        // `dispatch_capture` run instead.
        #[cfg(feature = "instrument")]
        if std::env::var_os("PROXIMA_DEBUG_RESOLVED_OPS").is_some() {
            // separate flags per plan shape -- prefill (step 0) and decode
            // build/resolve DIFFERENT `Plan`s, so a single shared flag would
            // let whichever shape resolves first (always prefill, step 0)
            // starve the other's dump for the rest of the process.
            static PREFILL_PRINTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            static DECODE_PRINTED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            let flag = if plan.command_buffer_chunks_decode_shaped {
                &DECODE_PRINTED
            } else {
                &PREFILL_PRINTED
            };
            if !flag.swap(true, core::sync::atomic::Ordering::Relaxed) {
                let plan_shape = if plan.command_buffer_chunks_decode_shaped {
                    "decode"
                } else {
                    "prefill"
                };
                for (dump_position, dump_bound) in prepared.resolved.iter().enumerate() {
                    let operand_ids: Vec<String> = dump_bound
                        .operands()
                        .iter()
                        .map(|(node, layout, _lookup)| format!("{}:{:?}", node.0, layout.strides))
                        .collect();
                    let output_len =
                        bound_output_len(dump_bound).max(1) * dump_bound.dtype.size_bytes();
                    eprintln!(
                        "resolved_op plan_shape={plan_shape} position={dump_position} node={} kind={} extents={:?} dtype={:?} operands=[{}] output_len={output_len}",
                        dump_bound.node.0,
                        dump_bound.kind.name(),
                        dump_bound.extents,
                        dump_bound.dtype,
                        operand_ids.join(", "),
                    );
                }
            }
        }
        // ROW 539: whether a `placement` below comes from the arena rather
        // than a caller-owned output-placed buffer -- the arena branch is
        // the one whose slot may be shared with an earlier, unrelated
        // position (`BufferArena::slot_is_recycled`), while an output-placed
        // buffer is a persistent identity the caller owns for the plan's
        // whole life.
        #[cfg(feature = "instrument")]
        let placement_is_arena_sourced = !output_placed.contains_key(&bound.node);
        let placement = match output_placed.get(&bound.node).copied() {
            Some(placement) => Some(placement),
            None => arena_placement(plan, position)?,
        };
        // attn_parity followon (2026-09-22, OWNER_BRIEF_gemma_head): per-node
        // pre-dispatch buffer identity + sentinel fill for the
        // `PROXIMA_HEAD_REPEATS` duplicate-head investigation. Placed here,
        // BEFORE the `merged_this_position`/`ablation_skip`/`resident_skip`
        // branch below, so a duplicate that takes one of those skip arms
        // still gets its identity printed and its buffer still gets
        // sentinel-filled -- an early exit is exactly the case this must
        // catch, and every one of those arms is reached only further down.
        #[cfg(feature = "instrument")]
        if let (Some(target_nodes), Some((buffer, offset))) =
            (head_debug_target_nodes(), placement)
            && let Some(target_index) = target_nodes.iter().position(|node| *node == bound.node)
        {
            let byte_length = bound_output_len(bound).max(1) * bound.dtype.size_bytes();
            let (_, grid) = kernel_dispatch_shape(bound, packed_operands, plan.numeric_policy)?;
            debug!(
                target_index = target_index as u64,
                node = bound.node.0 as u64,
                position = position as u64,
                buffer_pointer = Retained::as_ptr(buffer) as u64,
                offset = offset as u64,
                byte_length = byte_length as u64,
                output_total = bound_output_len(bound) as u64,
                grid_threads = grid.threads,
                grid_threadgroup_width = grid.threadgroup_width.unwrap_or(0),
                grid_depth = grid.depth,
                "head_debug: pre-dispatch buffer identity and dispatch shape"
            );
            if target_index > 0 && std::env::var_os("PROXIMA_HEAD_SENTINEL_FILL").is_some() {
                chunk_audit_record_write(
                    Retained::as_ptr(buffer) as usize,
                    offset,
                    byte_length,
                    position,
                );
                head_debug_fill_sentinel(buffer, offset, byte_length);
            }
        }
        // attn_parity followon (2026-09-22): `PROXIMA_REPEAT_VERIFY`'s own
        // pre-dispatch sentinel fill for every `PROXIMA_REPEAT_NODES` copy --
        // reuses `head_debug_fill_sentinel` unmodified (same quiet-NaN
        // pattern, same "write survives iff the dispatch actually ran"
        // contract `PROXIMA_HEAD_SENTINEL_FILL` established); the post-wait
        // compare lives in `arena_encode_dispatch_finish::finish`.
        #[cfg(feature = "instrument")]
        if let Some((buffer, offset)) = placement
            && std::env::var_os("PROXIMA_REPEAT_VERIFY").is_some()
            && plan
                .prepared
                .repeat_verify_pairs
                .iter()
                .any(|(_, copies)| copies.contains(&bound.node))
        {
            let byte_length = bound_output_len(bound).max(1) * bound.dtype.size_bytes();
            chunk_audit_record_write(
                Retained::as_ptr(buffer) as usize,
                offset,
                byte_length,
                position,
            );
            head_debug_fill_sentinel(buffer, offset, byte_length);
        }
        #[cfg(feature = "instrument")]
        {
            counter!(LOOP_HEAD_CALLS, 1);
            counter!(LOOP_HEAD_TICKS, elapsed_ticks(loop_head_started));
        }
        // row 555: `bound.node` above is the fused op's primary "out" node,
        // never the absorbed `state_out` node it also owns -- a caller's
        // `state_out` placement lives under a DIFFERENT key in the same
        // map, and `encode_op`'s own doc names why skipping this silently
        // discarded recurrent state on every fused `GatedDeltaNet` dispatch.
        let state_out_placement = match &bound.kind {
            BoundOpKind::GatedDeltaNet { state_out, .. } => {
                output_placed.get(state_out).copied()
            }
            _ => None,
        };
        // `ablation_skip` is `false` on every non-`instrument` build (the
        // `match` folds to the literal at compile time, so this costs
        // nothing in production) and `false` on every `instrument` build
        // where `PROXIMA_METAL_KIND_FILTER` is unset -- the only way into
        // the `register_skipped_output` arm below is an operator explicitly
        // setting that env var, which never happens outside this ablation's
        // own harness.
        #[cfg(feature = "instrument")]
        let ablation_skip = match &kind_filter {
            Some(filter) => !filter.matches(
                classify_kind(bound, packed_operands),
                bound_weight_family(bound, &plan.program).as_deref(),
            ),
            None => false,
        };
        #[cfg(not(feature = "instrument"))]
        let ablation_skip = false;

        // `Plan::mark_plan_time_constants_resident`'s own doc: a resident
        // `Iota`/`Constant` leaf's first real dispatch already wrote a value
        // good for the plan's whole life into a `BufferArena` slot
        // [`resident_pinned_retires`] now refuses to ever hand back -- once
        // `device_buffers` carries that entry (a warm call, never the plan's
        // very first), re-dispatching it here would recompute the identical
        // bytes into the identical buffer for nothing. `output_placed`
        // excluded: a caller-owned output buffer is refreshed every call by
        // its own contract, not this plan's residency promise.
        let resident_skip = plan.resident_nodes.contains(&bound.node)
            && matches!(bound.kind, BoundOpKind::Iota | BoundOpKind::Constant { .. })
            && !output_placed.contains_key(&bound.node)
            && device_buffers.contains_key(&bound.node);

        #[cfg(feature = "metal-horizontal-merge")]
        let merged_this_position = handle_merged_position(
            &device,
            &encoder,
            plan,
            position,
            bound,
            dispatch_type,
            &mut device_buffers,
            hazard_state,
            &output_placed,
        )?;
        #[cfg(not(feature = "metal-horizontal-merge"))]
        let merged_this_position = false;

        if merged_this_position {
            // encoded (or, for a z>0 member, hazard-recorded with no
            // dispatch) entirely inside `handle_merged_position` -- nothing
            // left to do here but fall through to the retirement loop below,
            // exactly like the ordinary path's own `else` arm does.
        } else if ablation_skip {
            // `ablation_skip` is always `false` on a non-`instrument` build
            // (see its own binding above), so this arm never runs there --
            // `register_skipped_output` itself is `instrument`-gated and
            // does not exist to call outside it.
            #[cfg(feature = "instrument")]
            register_skipped_output(&device, &mut device_buffers, bound, placement)?;
        } else if resident_skip {
            // `device_buffers[bound.node]` already holds this call's correct
            // value from a prior call -- no dispatch, no hazard bookkeeping
            // (a resident leaf has no operands to read and its own buffer is
            // never rewritten again, so no WAW/WAR edge can exist for it).
        } else {
            // A skipped op above never reaches this hazard tracking either
            // -- it neither reads nor writes a buffer THIS command buffer
            // touches, so the tracker must see it as absent: no
            // `hazard_inputs`/`hazard_output` collected, no
            // `needs_barrier`/`record` call, no barrier emitted on its
            // account.
            // `Serial` never overlaps two dispatches, so no hazard this
            // tracker catches could ever race -- skip it entirely rather
            // than pay the resolve/record bookkeeping for barriers that
            // would never fire.
            let resolved_step = resolved_steps
                .as_ref()
                .and_then(|resolved| resolved.steps.get(position));
            let resolved_output: Option<DeviceBuffer> = if dispatch_type == DispatchType::Concurrent
            {
                // The hazard read set is derived from `bindings` -- the exact
                // list `bind_buffers` will bind for this op -- rather than
                // enumerated separately from `bound.all_read_sources()`: ROW
                // 323 was exactly those two enumerations drifting apart when
                // `msl::bindings` grew a source (a fused epilogue operand)
                // this hazard check had not been taught to see. Reading
                // `bindings` itself makes that class of drift impossible --
                // there is only one list, and both the encoder bind loop
                // (`bind_buffers`) and this hazard walk read the same one.
                let owned_bindings: Vec<Binding>;
                let bindings_for_hazard: &[Binding] = match resolved_step {
                    Some(step) => step.bindings.as_slice(),
                    None => {
                        let (bindings, _grid) =
                            kernel_dispatch_shape(bound, packed_operands, plan.numeric_policy)?;
                        owned_bindings = bindings;
                        owned_bindings.as_slice()
                    }
                };
                resolve_hazard_inputs_into(
                    crate::msl::hazard_read_nodes(bindings_for_hazard),
                    &device_buffers,
                    &mut hazard_state.inputs,
                )?;
                // resolved ONCE, before the hazard check, from the exact same
                // placement -> arena -> fresh-allocation chain `encode_op`
                // would otherwise pick independently below: `needs_barrier`
                // and `record` used to see two different identities for a
                // fresh allocation (the check saw `None`, since nothing is
                // known before `encode_op` allocates; the record afterward
                // saw the real pointer), so a WAW/WAR hazard against an
                // address Metal happens to reuse for that allocation could
                // never be caught. Allocating here and handing the SAME
                // buffer into `encode_op` via `placement` closes that gap.
                let resolved: DeviceBuffer = match placement {
                    Some((buffer, offset)) => (buffer.clone(), offset),
                    None => (
                        allocate_buffer(&device, bound_output_len(bound), bound.dtype)?,
                        0,
                    ),
                };
                // the write side of the same "derived from bindings" guarantee:
                // whenever `bindings_for_hazard` names an explicit
                // `Binding::Output`, it must name THIS op's own node -- if it
                // ever didn't, the buffer the hazard tracker records as
                // written and the buffer `bind_buffers` actually binds as
                // this op's output would be two different things, which is a
                // worse bug than the one this refactor closes. Redesign §4c:
                // a `CachedAttention` split kernel under `ContextSplitMerge`
                // has NO `Binding::Output` at all (`Binding::Scratch`
                // replaces it -- that kernel writes scratch, never
                // `bound.node`'s own buffer) -- `hazard_write_node` returning
                // `None` there is the honest, by-design case, not a drift:
                // `resolved` below is still `bound.node`'s own real output
                // buffer (from `placement`, independent of `bindings`), and
                // the merge dispatch this position's `encode_op` call also
                // issues is what actually writes it, in the same encoder,
                // immediately after the split.
                debug_assert!(
                    matches!(
                        crate::msl::hazard_write_node(bindings_for_hazard),
                        Some(node) if node == bound.node
                    ) || crate::msl::hazard_write_node(bindings_for_hazard).is_none()
                );
                let hazard_output = Retained::as_ptr(&resolved.0);
                // Read-only, computed BEFORE `hazard_step` mutates the
                // tracker below, so it sees the exact same state
                // `hazard_step`'s own internal check will -- `classify`
                // never mutates, so precomputing it here for the counter
                // breakdown cannot change what `hazard_step` decides.
                #[cfg(feature = "instrument")]
                let hazard_class = hazard_state
                    .tracker
                    .classify(&hazard_state.inputs, Some(hazard_output));
                if hazard_step(
                    &mut hazard_state.tracker,
                    &hazard_state.inputs,
                    hazard_output,
                ) {
                    encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
                    counter!(BARRIERS_EMITTED, 1);
                    #[cfg(feature = "instrument")]
                    {
                        let arena_recycled = placement_is_arena_sourced
                            && plan
                                .arena
                                .get()
                                .is_some_and(|arena| arena.slot_is_recycled(position));
                        record_hazard_class(hazard_class, arena_recycled);
                    }
                }
                Some(resolved)
            } else {
                None
            };
            let placement = match &resolved_output {
                Some((buffer, offset)) => Some((buffer, *offset)),
                None => placement,
            };
            // attribution2 followon: the placements path never wired
            // `PLACEMENT_RESOLVE_TICKS`/`EXPERT_BUFFERS_LOOKUP_TICKS` the
            // sibling `execute_plan_inner` (`execute_and_hazards.rs`) already
            // has under its own ROW-13ms attribution -- these three lookups
            // plus `expert_buffers_for` below ran, per position, entirely
            // outside `OP_SETUP_TICKS`/`ENCODE_DISPATCH_TICKS` (both start
            // inside `encode_op` itself), which is exactly the gap between
            // `evaluate_ms` and the four counters `token_breakdown_metal`
            // already reports.
            #[cfg(feature = "instrument")]
            let placement_resolve_started = read_ticks();
            let uniform_buffer = plan_uniform_buffer(plan, position)?;
            // Redesign §4c: the ONE call site that resolves a scratch
            // buffer for `encode_op`'s two-dispatch `CachedAttention` form
            // -- every other `encode_op` caller passes `None` and rejects a
            // `Binding::Scratch` kernel instead (that function's own doc).
            let attention_scratch =
                attention_scratch_buffer(plan, position)?.map(|buffer| (buffer, 0usize));
            #[cfg(feature = "instrument")]
            {
                counter!(PLACEMENT_RESOLVE_CALLS, 1);
                counter!(
                    PLACEMENT_RESOLVE_TICKS,
                    elapsed_ticks(placement_resolve_started)
                );
            }
            // Only `Concurrent` needs the intra-op scratch write -> read edge
            // routed through the tracker (see `encode_op`'s own doc for
            // `hazard`) -- `Serial` orders the split before the merge for
            // free, the same reason `resolved_output` above is `None` there.
            let hazard =
                (dispatch_type == DispatchType::Concurrent).then_some(&mut hazard_state.tracker);
            #[cfg(feature = "instrument")]
            let expert_buffers_started = read_ticks();
            let bound_expert_buffers = expert_buffers_for(bound, &effective_expert_buffers)?;
            #[cfg(feature = "instrument")]
            {
                counter!(EXPERT_BUFFERS_LOOKUP_CALLS, 1);
                counter!(
                    EXPERT_BUFFERS_LOOKUP_TICKS,
                    elapsed_ticks(expert_buffers_started)
                );
            }
            let fault = encode_op(
                &device,
                &encoder,
                &mut device_buffers,
                bound,
                packed_operands,
                placement,
                state_out_placement,
                uniform_buffer,
                plan.math_mode,
                plan.numeric_policy,
                resolved_step,
                attention_scratch,
                hazard,
                bound_expert_buffers,
            )?;
            if let Some((fault_buffer, gathers)) = fault {
                pending_faults.push((bound, fault_buffer, gathers));
            }
        }
        // explicit liveness exclusion (see this function's doc): a placed
        // node, input or output, is externally owned and always live, so it
        // is never dropped from this call's own bookkeeping map, regardless
        // of what `prepared.retires` (a per-program-only liveness sweep)
        // says. A RESIDENT block node joins that exclusion for the same
        // reason: [`Plan::mark_resident`]'s caller-owned promise means its
        // buffer is live for the plan's whole life, and leaving its entry in
        // [`Plan::device_buffers`] across calls is exactly what lets
        // `block_buffer_reusable` skip re-uploading it next call.
        #[cfg(feature = "instrument")]
        let retire_scan_started = read_ticks();
        for retired in &prepared.retires[position] {
            if input_placed.contains_key(retired)
                || output_placed.contains_key(retired)
                || plan.resident_nodes.contains(retired)
            {
                continue;
            }
            // Keep the identity in this command buffer's hazard sets after
            // logical retirement. BufferArena may reuse the same persistent
            // MTLBuffer for a later output before this command buffer
            // commits; clearing it here would erase the WAR/WAW edge that
            // reuse needs. HazardState resets at the next command-buffer
            // call, after every dispatch in this one has retired, so the
            // identity cannot leak across submissions.
            device_buffers.remove(retired);
        }
        #[cfg(feature = "instrument")]
        {
            counter!(RETIRE_SCAN_CALLS, prepared.retires[position].len() as u64);
            counter!(RETIRE_SCAN_TICKS, elapsed_ticks(retire_scan_started));
        }
    }
    #[cfg(feature = "instrument")]
    chunk_audit_finish(CAPTURE_STEP.load(core::sync::atomic::Ordering::Relaxed));
    // attribution2 followon: `encoder.finish()` is a real Metal API call
    // (closes the command encoder) that ran entirely outside any timer --
    // between the last `RETIRE_SCAN_TICKS` sample and `GPU_EXEC_TICKS`'
    // own `commit()`/`waitUntilCompleted()` start.
    #[cfg(feature = "instrument")]
    let encoder_finish_started = read_ticks();
    encoder.finish();
    #[cfg(feature = "instrument")]
    counter!(ENCODER_FINISH_TICKS, elapsed_ticks(encoder_finish_started));
    // Deliverable 2: the LAST chunk's own encode_start/encode_end -- pushed
    // here (its `commit_ms` filled in right after the `commit()` call below)
    // so `chunk_host_timings.len() == chunk_command_buffers.len() ==
    // chunk_count` always holds, the same invariant every earlier chunk's
    // push already established.
    #[cfg(feature = "instrument")]
    let last_chunk_encode_end_ms = step_encode_start.elapsed().as_secs_f64() * 1e3;
    #[cfg(feature = "instrument")]
    chunk_command_buffers.push(command_buffer.clone());
    // "encode time elapsed after the first commit" (OWNER_BRIEF_structural_
    // difference): the host wall-clock span from the first chunk's `commit()`
    // call to the last chunk's `encoder.finish()`, i.e. the encode work the
    // GPU could in principle overlap with chunk 1's execution. `None` on the
    // default K=1 path (no swap ever set `post_first_commit_encode_started`),
    // so this reads 0.0 there, unchanged from before this change.
    #[cfg(feature = "instrument")]
    let encode_overlap_ms = post_first_commit_encode_started
        .map(|started| started.elapsed().as_secs_f64() * 1e3)
        .unwrap_or(0.0);

    #[cfg(feature = "instrument")]
    let gpu_exec_started = read_ticks();
    #[cfg(feature = "instrument")]
    let commit_call_started = std::time::Instant::now();
    command_buffer.commit();
    #[cfg(feature = "instrument")]
    let commit_call_ms = commit_call_started.elapsed().as_secs_f64() * 1e3;
    #[cfg(feature = "instrument")]
    chunk_host_timings.push(ChunkHostTiming {
        first_op: current_chunk_first_op,
        last_op: total_ops.saturating_sub(1),
        encode_start_ms: current_chunk_encode_start_ms,
        encode_end_ms: last_chunk_encode_end_ms,
        commit_ms: commit_call_ms,
    });
    #[cfg(feature = "instrument")]
    let wait_started = std::time::Instant::now();
    command_buffer.waitUntilCompleted();
    #[cfg(feature = "instrument")]
    {
        counter!(GPU_EXEC_CALLS, chunk_count as u64);
        counter!(GPU_EXEC_TICKS, elapsed_ticks(gpu_exec_started));
        // attribution3 followon (Part A): `GPUStartTime`/`GPUEndTime` are
        // read only by the diagnostic `execute_plan_timed` today (this
        // file's own doc there); this is the first read on the production
        // placements path. Apple's docs state both use the same time base
        // as `CACurrentMediaTime`, which is itself `mach_absolute_time`
        // scaled by the same `mach_timebase_info` `ticks_to_nanos` already
        // applies -- so `gpu_exec_started`'s tick, converted to seconds via
        // `ticks_to_nanos`, is directly comparable to `GPUStartTime`. That
        // correlation is asserted here, not proven; if the two clocks
        // disagree by more than noise the print below carries both raw
        // readings so a reader can judge without rerunning.
        if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
            let wait_return_ms = wait_started.elapsed().as_secs_f64() * 1e3;
            let gpu_exec_ms =
                proxima_tensor::instrument::ticks_to_nanos(elapsed_ticks(gpu_exec_started)) as f64
                    / 1e6;
            let commit_call_start_s = proxima_tensor::instrument::ticks_to_nanos(
                gpu_exec_started.as_raw(),
            ) as f64
                / 1e9;
            // Intervention 6 (OWNER_BRIEF_structural_difference): with
            // `chunks > 1` this is no longer one command buffer -- `gpu_busy`
            // must span the FIRST chunk's `GPUStartTime` to the LAST
            // chunk's `GPUEndTime` (the `command_buffer` binding here is
            // always the last chunk, waited on above; `first_command_buffer`
            // is `None` on the default K=1 path, so the fallback keeps this
            // read identical to before this change).
            let leading_command_buffer = first_command_buffer.as_ref().unwrap_or(&command_buffer);
            let gpu_start_s = leading_command_buffer.GPUStartTime();
            let gpu_end_s = command_buffer.GPUEndTime();
            let commit_to_gpu_start_ms = ((gpu_start_s - commit_call_start_s) * 1e3).max(0.0);
            let gpu_busy_ms = ((gpu_end_s - gpu_start_s) * 1e3).max(0.0);
            let gpu_end_to_wait_return_ms =
                (gpu_exec_ms - commit_call_ms - commit_to_gpu_start_ms - gpu_busy_ms).max(0.0);
            let leading_commit_start_s = first_commit_call_start_s.unwrap_or(commit_call_start_s);
            let first_commit_to_first_gpu_start_ms =
                ((gpu_start_s - leading_commit_start_s) * 1e3).max(0.0);
            eprintln!(
                "token_breakdown_gpu commit_call_ms={commit_call_ms} \
                 commit_to_gpu_start_ms={commit_to_gpu_start_ms} \
                 gpu_busy_ms={gpu_busy_ms} \
                 gpu_end_to_wait_return_ms={gpu_end_to_wait_return_ms} \
                 sum_ms={} gpu_exec_ms={gpu_exec_ms} \
                 gpu_start_raw_s={gpu_start_s} gpu_end_raw_s={gpu_end_s} \
                 commit_call_start_raw_s={commit_call_start_s} \
                 wait_return_extra_ms={wait_return_ms} \
                 chunks={chunk_count} \
                 first_commit_to_first_gpu_start_ms={first_commit_to_first_gpu_start_ms} \
                 encode_overlap_ms={encode_overlap_ms}",
                commit_call_ms + commit_to_gpu_start_ms + gpu_busy_ms + gpu_end_to_wait_return_ms,
            );
            // Deliverable 2 (OWNER_BRIEF_chunked_submission_audit): one
            // `chunk_record` per committed command buffer -- every
            // `GPUStartTime`/`GPUEndTime` read here is valid because this
            // runs AFTER `waitUntilCompleted` above, which (Serial commit
            // order on one queue) proves every earlier chunk's GPU work
            // completed too, not only the last one waited on directly.
            let step = CAPTURE_STEP.load(core::sync::atomic::Ordering::Relaxed);
            let plan_shape = if plan.command_buffer_chunks_decode_shaped {
                "decode"
            } else {
                "prefill"
            };
            let step_epoch_s = proxima_tensor::instrument::ticks_to_nanos(
                step_encode_start_ticks.as_raw(),
            ) as f64
                / 1e9;
            let mut sum_gpu_exec_ms = 0.0f64;
            let mut gpu_starts_ends: Vec<(f64, f64)> = Vec::with_capacity(chunk_command_buffers.len());
            for (index, buffer) in chunk_command_buffers.iter().enumerate() {
                let timing = &chunk_host_timings[index];
                let gpu_start_ms = ((buffer.GPUStartTime() - step_epoch_s) * 1e3).max(0.0);
                let gpu_end_ms = ((buffer.GPUEndTime() - step_epoch_s) * 1e3).max(0.0);
                sum_gpu_exec_ms += (gpu_end_ms - gpu_start_ms).max(0.0);
                gpu_starts_ends.push((gpu_start_ms, gpu_end_ms));
                eprintln!(
                    "chunk_record step={step} plan_shape={plan_shape} chunk={}/{chunk_count} ops={}..{} \
                     encode_start_ms={} encode_end_ms={} commit_ms={} \
                     gpu_start_ms={gpu_start_ms} gpu_end_ms={gpu_end_ms}",
                    index + 1,
                    timing.first_op,
                    timing.last_op,
                    timing.encode_start_ms,
                    timing.encode_end_ms,
                    timing.commit_ms,
                );
            }
            let mut inter_buffer_gaps_ms = 0.0f64;
            for window in gpu_starts_ends.windows(2) {
                inter_buffer_gaps_ms += (window[1].0 - window[0].1).max(0.0);
            }
            let first_start_to_last_end_ms = gpu_starts_ends
                .last()
                .zip(gpu_starts_ends.first())
                .map(|((_, last_end), (first_start, _))| last_end - first_start)
                .unwrap_or(0.0);
            let encode_total_ms: f64 = chunk_host_timings
                .iter()
                .map(|timing| timing.encode_end_ms - timing.encode_start_ms)
                .sum();
            eprintln!(
                "chunk_summary step={step} plan_shape={plan_shape} chunks={chunk_count} \
                 regions={chunk_count} ignored_boundaries={ignored_boundaries} \
                 sum_gpu_exec_ms={sum_gpu_exec_ms} \
                 first_start_to_last_end_ms={first_start_to_last_end_ms} \
                 inter_buffer_gaps_ms={inter_buffer_gaps_ms} \
                 encode_total_ms={encode_total_ms}",
            );
        }
    }

    for (bound, fault_buffer, gathers) in &pending_faults {
        check_gather_fault(bound, fault_buffer, *gathers)?;
    }

    let placed_output_nodes: BTreeSet<NodeId> = output_placed.keys().copied().collect();
    let evaluated = finish(plan, &device_buffers, &placed_output_nodes, recycle.pop())?;

    // Expert buffers are a per-step source snapshot, not plan-resident
    // weights. `finish` has already read every requested output, so retaining
    // these entries in the plan would keep the all-expert prefill arena alive
    // while the bounded DynaExq table takes over for decode.
    drop(device_buffers);
    {
        let mut plan_buffers = plan.device_buffers.borrow_mut();
        for node in effective_expert_buffers.keys() {
            plan_buffers.remove(node);
        }
    }
    Ok(evaluated)
}

/// [`plan`] against a name-keyed block set — the shape a model binds its
/// weights in. Resolution goes through
/// [`proxima_tensor::resolve_named_blocks`], the same function the CPU
/// evaluator uses, so the two backends cannot disagree about which name is
/// which position.
///
/// # Errors
/// Propagates name-resolution and planning failures.
pub fn plan_named(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
) -> Result<Plan, MetalError> {
    let blocks = resolve_named_blocks(program, named)?;
    plan_with_placed_inputs(program, symbols, &blocks, outputs, numeric_policy, &[], true)
}

/// [`plan`] against named blocks plus caller-owned input placements.
///
/// A placed input does not need a host payload during planning. The planner
/// still validates every ordinary named block, while the execution resolver
/// supplies an empty sentinel for each node whose bytes arrive from its
/// caller-owned [`PlacedBuffer`].
#[cfg(feature = "metal-output-placement")]
pub fn plan_named_with_placed_inputs(
    program: &[Op],
    symbols: &[u64],
    named: &[(&str, QuantizedBlock<'_>)],
    outputs: &[NodeId],
    numeric_policy: NumericPolicy,
    placed_input_nodes: &[NodeId],
    fuse_cached_attention: bool,
) -> Result<Plan, MetalError> {
    let blocks = resolve_named_blocks_with_placed_nodes(program, named, |node| {
        placed_input_nodes.contains(&node)
    })?;
    plan_with_placed_inputs(
        program,
        symbols,
        &blocks,
        outputs,
        numeric_policy,
        placed_input_nodes,
        fuse_cached_attention,
    )
}

/// [`execute_plan`] against a name-keyed block set. The plan owns its
/// program, so the caller hands over only the per-call data.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
pub fn execute_plan_named(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<Evaluated, MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan(plan, &blocks)
}

/// [`execute_plan_with_placements`] against a name-keyed block set -- the
/// name-resolving sibling of [`execute_plan_named`], mirroring how that
/// function wraps [`execute_plan`]. `blocks` (positional, for the per-node
/// upload path) is resolved once via [`resolve_named_blocks`], then handed
/// to `execute_plan_with_placements` unchanged; input-placed nodes skip that
/// resolved entry's upload just as they do in the positional call.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_named_with_placements(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<Evaluated, MetalError> {
    let blocks = resolve_named_blocks_with_placed_inputs(plan, named, input_placements)?;
    // no scratch pool for this name-keyed convenience wrapper -- a caller
    // wanting the recycle path calls `execute_plan_with_placements` directly.
    execute_plan_with_placements(
        plan,
        &blocks,
        input_placements,
        output_placements,
        &mut Vec::new(),
    )
}

/// Executes a named plan with both caller-owned buffers and per-step expert
/// substitutions. Routed recurrent models need both capabilities in the same
/// command buffer: placement keeps recurrent state on the device while the
/// expert table selects the codec and address for the current route.
///
/// # Errors
/// Propagates name resolution, expert-source validation, and Metal failures.
#[cfg(feature = "metal-output-placement")]
pub fn execute_plan_named_with_placements_and_expert_sources(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
    output_placements: &[(NodeId, &PlacedBuffer, usize)],
    expert_sources: &BTreeMap<NodeId, proxima_tensor::cpu::ExpertSource<'_>>,
) -> Result<Evaluated, MetalError> {
    let blocks = resolve_named_blocks_with_placed_inputs(plan, named, input_placements)?;
    #[cfg(feature = "instrument")]
    let expert_stage_started = std::time::Instant::now();
    let expert_buffers = stage_expert_sources(plan, &blocks, expert_sources)?;
    #[cfg(feature = "instrument")]
    if std::env::var_os("PROXIMA_DEBUG_METAL_STAGES").is_some() {
        eprintln!(
            "expert_source_stage_ms={} source_nodes={} staged_nodes={}",
            expert_stage_started.elapsed().as_secs_f64() * 1e3,
            expert_sources.len(),
            expert_buffers.len(),
        );
    }
    execute_plan_with_placements_inner(
        plan,
        &blocks,
        input_placements,
        output_placements,
        &mut Vec::new(),
        &expert_buffers,
    )
}

pub(super) fn resolve_named_blocks_with_placed_inputs<'blocks>(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'blocks>)],
    input_placements: &[(NodeId, &PlacedBuffer, usize)],
) -> Result<Vec<QuantizedBlock<'blocks>>, MetalError> {
    resolve_named_blocks_with_placed_nodes(&plan.program, named, |node| {
        input_placements
            .iter()
            .any(|(placed, _, _)| *placed == node)
    })
}

pub(super) fn resolve_named_blocks_with_placed_nodes<'blocks, IsPlaced>(
    program: &[Op],
    named: &[(&str, QuantizedBlock<'blocks>)],
    is_placed: IsPlaced,
) -> Result<Vec<QuantizedBlock<'blocks>>, MetalError>
where
    IsPlaced: Fn(NodeId) -> bool,
{
    let block_nodes = block_node_ids(program);
    let mut blocks = Vec::with_capacity(block_nodes.len());
    for node in &block_nodes {
        let name = program[node.0 as usize]
            .name()
            .ok_or(TensorError::UnnamedInput(*node))?;
        if let Some(block) = named.iter().find(|(candidate, _)| *candidate == name) {
            blocks.push(block.1);
        } else if is_placed(*node) {
            blocks.push(QuantizedBlock::Float32(&[]));
        } else {
            return Err(TensorError::UnboundInputName(String::from(name)).into());
        }
    }
    Ok(blocks)
}

/// [`execute_plan`] with the WHOLE program's own single command buffer's
/// `GPUStartTime`/`GPUEndTime` read back once, instead of
/// [`execute_plan_op_timed`]'s one-command-buffer-per-op attribution. ROW
/// 375 measured a batch of independent same-kind dispatches with the
/// per-op instrument and got a flat ~705 us reading dominated by that
/// function's own per-buffer submit/wait floor, 13-20x the in-program
/// per-dispatch cost `rmsnorm_fused_epilogue_cost.rs`'s ROW 368/372 batched
/// harness measures through this same one-encoder, one-command-buffer
/// shape `execute_plan` already uses for production. This function exists
/// so a caller batching N independent dispatches into one plan (this
/// crate's own `plan_named`/`execute_plan_named`) can read that batch's
/// real GPU occupancy without re-deriving `execute_plan`'s encode loop.
///
/// # Errors
/// Same as [`execute_plan`].
#[cfg(feature = "instrument")]
pub fn execute_plan_timed(
    plan: &Plan,
    blocks: &[QuantizedBlock<'_>],
) -> Result<(Evaluated, u64), MetalError> {
    let prepared = &plan.prepared;
    let packed_operands = &plan.packed_operands;

    let (device, queue) = device_and_queue()?;

    let mut device_buffers: BTreeMap<NodeId, DeviceBuffer> = BTreeMap::new();
    for (index, ((node, block), dtype)) in prepared
        .block_nodes
        .iter()
        .zip(blocks.iter())
        .zip(plan.block_dtypes.iter())
        .enumerate()
    {
        if !prepared.live_block_inputs[index] {
            continue;
        }
        let resident_name = resident_name(plan, *node);
        let buffer = match block {
            QuantizedBlock::Float32(data) => upload_block(
                &device,
                data,
                *node,
                *dtype,
                plan.program[node.0 as usize].name(),
                resident_name,
            )?,
            QuantizedBlock::Int32(data) => {
                upload_block_int32_as_float(&device, data, resident_name)?
            }
            QuantizedBlock::Packed { bytes, .. } => {
                upload_packed_bytes(&device, bytes, resident_name)?
            }
        };
        device_buffers.insert(*node, buffer);
    }

    let command_buffer = queue
        .commandBuffer()
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;
    let encoder = EncoderGuard::new(command_buffer.computeCommandEncoder().ok_or_else(|| {
        MetalError::CompileFailed {
            log: "command buffer refused to hand out a compute encoder".to_string(),
        }
    })?);

    let mut pending_faults: Vec<PendingFault<'_>> = Vec::new();
    for (position, bound) in prepared.resolved.iter().enumerate() {
        let fault = encode_op(
            &device,
            &encoder,
            &mut device_buffers,
            bound,
            packed_operands,
            None,
            None,
            None,
            plan.math_mode,
            plan.numeric_policy,
            None,
            None,
            None,
            None,
        )?;
        if let Some((fault_buffer, gathers)) = fault {
            pending_faults.push((bound, fault_buffer, gathers));
        }
        for retired in &prepared.retires[position] {
            device_buffers.remove(retired);
        }
    }
    encoder.finish();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();
    let gpu_ns =
        ((command_buffer.GPUEndTime() - command_buffer.GPUStartTime()) * 1e9).max(0.0) as u64;

    for (bound, fault_buffer, gathers) in &pending_faults {
        check_gather_fault(bound, fault_buffer, *gathers)?;
    }

    let evaluated = finish(plan, &device_buffers, &BTreeSet::new(), None)?;
    Ok((evaluated, gpu_ns))
}

/// [`execute_plan_timed`] against a name-keyed block set, mirroring
/// [`execute_plan_named`]'s own name resolution.
///
/// # Errors
/// Propagates name-resolution and Metal driver failures.
#[cfg(feature = "instrument")]
pub fn execute_plan_named_timed(
    plan: &Plan,
    named: &[(&str, QuantizedBlock<'_>)],
) -> Result<(Evaluated, u64), MetalError> {
    let blocks = resolve_named_blocks(&plan.program, named)?;
    execute_plan_timed(plan, &blocks)
}

/// One [`BoundOp`]'s GPU-only execution time and the operand bytes it read,
/// as gathered by [`execute_plan_op_timed`]. `weight_name` is
/// [`Op::name`](proxima_tensor::Op::name) off whichever operand is a named
/// block input (a model weight or the `ids`/`eps`/`rope_*`/KV-cache block
/// this program declares by name) -- `None` when every operand is itself a
/// computed node, which is the common case for a chained elementwise op.
#[cfg(feature = "instrument")]
#[derive(Debug, Clone)]
pub struct OpGpuTiming {
    pub node: NodeId,
    pub kind: &'static str,
    /// Full iteration extents of the bound operation that was timed.
    pub extents: Vec<u64>,
    /// Surviving axes for a reduction; empty for non-reductions.
    pub output_axes: Vec<u16>,
    /// This operand's TENSOR bytes -- `element_count(shape) * bytes_per_element`,
    /// where `bytes_per_element` is the operand's own dtype width for a plain
    /// buffer, or its `Codec::block_bytes`/block-elements ratio for a
    /// packed one. See `operand_tensor_bytes`. Distinct from
    /// [`Self::bound_buffer_bytes`]: since a checkpoint mapping upload binds
    /// ONE buffer spanning the whole mmap (`checkpoint_mapping_offset`), that
    /// buffer's own `length()` overstates every individual operand sharing it
    /// -- this field is what a per-tensor byte-share table needs instead.
    pub operand_bytes: u64,
    /// The device buffer's own `length()` this operand was bound against --
    /// the value `operand_bytes` used to report before it was corrected to
    /// the tensor's own byte count. Kept so a checkpoint-mapping-offset bind
    /// (one shared buffer, `bound_buffer_bytes` far larger than
    /// `operand_bytes`) stays observable rather than silently disappearing.
    pub bound_buffer_bytes: u64,
    pub gpu_ns: u64,
    pub weight_name: Option<String>,
    /// `bound.operands().len()` -- surfaced so a diagnostic caller can tell
    /// a two-operand (weight, activation) reduce, the shape
    /// `crate::msl::packed_row_block` requires, from a fused reduce whose
    /// element body absorbed a third operand (e.g. a gate/up product ahead
    /// of a down-projection), which disqualifies the row-blocked kernel via
    /// that same function's `quantized.len() != 2` check.
    pub operand_count: usize,
    /// The packed codec carried by this op's named operand, when present.
    pub packed_codec: Option<Codec>,
    /// The emitted packed-kernel body, kept separate from the broad op kind.
    pub packed_kernel_variant: &'static str,
    /// [`crate::msl::diagnose_packed_row_block`]'s own verdict on THIS op,
    /// against a REAL bound program rather than a synthetic symbolic one --
    /// `None` when the op is not a `Reduce { keep: Keep::Reduce, .. }` at
    /// all (the diagnosis does not apply), `Some("PASS")` when it took the
    /// row-blocked path, `Some(<rejection debug>)` otherwise. Exists
    /// because a synthetic probe's rejection table and a real production
    /// run's own `classify_kind` bucket disagreed on `ffn_down`/
    /// `output.weight` -- printing this against the REAL bound op is what
    /// settles which one was wrong, rather than trusting either by
    /// inference.
    pub packed_row_block_rejection: Option<String>,
}

/// `PROXIMA_METAL_NAN_CHECK`-gated NaN probe for exactly ONE op's own output
/// buffer, called from [`execute_op_timed`] right after `waitUntilCompleted`
/// and before that op's operands are retired -- the diagnostic this crate's
/// own `PROXIMA_METAL_OP_PROFILE_STEP` path lacked to bisect the 30B
/// qwen3moe Metal decode's all-zero logits down to the exact op that first
/// produced NaN data, rather than the whole step.
///
/// Checks `is_nan()`, deliberately NOT `!is_finite()` (which would also
/// catch `-inf`): `causal_mask` (`proxima-tensor/src/spec.rs`) bakes a
/// literal `f32::NEG_INFINITY` into the program as a `constant` op, fed
/// through a `Select` into every masked attention score BEFORE softmax --
/// the SAME literal a correct CPU decode also evaluates and zeroes out, so
/// a `-inf` reaching a masked score, or even an unmasked reduce that legally
/// saturates, is not evidence of anything. This codebase never constructs an
/// intentional `f32::NAN` scalar (grepped: zero production call sites), so a
/// NaN appearing on ANY node's output has no legitimate source and is
/// unambiguous evidence of the defect this diagnostic exists to find.
///
/// Reads the buffer back through [`read_back`] (the same narrow-to-f32 path
/// [`finish`] uses for every other output), which is why this is reachable
/// only behind `instrument`: it pays a device round trip per op, same cost
/// class [`execute_op_timed`]'s own doc already accepts for GPU-time
/// attribution. Returns `true` the first time this op's own output holds a
/// NaN, letting the caller stop the step at the FIRST offending op instead
/// of running every remaining op past a value already known bad.
// a type alias, not a new type: names the tuple `check_op_output_finite`
// returns so clippy's `type_complexity` lint reads it once instead of
// flagging the signature -- first NaN/Inf index, the op's shape, its
// readback values.
#[cfg(feature = "instrument")]
pub(super) type FiniteCheckFailure = (usize, Vec<u64>, alloc::vec::Vec<f32>);

#[cfg(feature = "instrument")]
pub(super) fn check_op_output_finite(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    program: &[Op],
    node: NodeId,
    kind: &str,
) -> Result<Option<FiniteCheckFailure>, MetalError> {
    let Some((buffer, offset)) = device_buffers.get(&node) else {
        return Ok(None);
    };
    let shape = prepared.shapes.of(node).to_vec();
    let dtype = gpu_dtype(program, &prepared.index_nodes, &prepared.resolved, node);
    let data = read_back(buffer, *offset, element_count(&shape), node, dtype)?;
    let nan_count = data.iter().filter(|value| value.is_nan()).count();
    if nan_count == 0 {
        return Ok(None);
    }
    let Some(first_index) = data.iter().position(|value| value.is_nan()) else {
        return Ok(None);
    };
    let first_values: alloc::vec::Vec<f32> = data.iter().copied().take(8).collect();
    let kind_owned = kind.to_string();
    debug!(
        node = node.0,
        kind = %kind_owned,
        shape = ?shape,
        first_values = ?first_values,
        nan_count = nan_count as u64,
        total_count = data.len() as u64,
        "nan_check: first nan op output found"
    );
    Ok(Some((first_index, shape, data)))
}

/// `PROXIMA_METAL_COMPARE_CPU`-gated per-node parity probe, called from
/// [`execute_op_timed`] alongside [`check_op_output_finite`] -- reads this
/// op's own Metal output back through [`read_back`] and diffs it against
/// `cpu_reference`'s entry for the SAME [`NodeId`], computed once up front by
/// evaluating the identical program/symbols/blocks on the CPU route for
/// every non-`Op::Input` node (`proxima-model-interop/src/generate.rs`'s
/// `evaluate_op_timed`, the one call site that builds `cpu_reference` and
/// has `program`/`symbols`/`named` all in scope to do it).
///
/// Relative diff per element uses a `1e-6` floor on the denominator so a
/// near-zero CPU reference value cannot inflate a genuinely tiny absolute
/// gap into a huge ratio. Returns the max relative diff across every element
/// (`None` when this node has no CPU reference entry -- weights and other
/// `Op::Input` nodes are never in `cpu_reference` by construction). The
/// caller stops the step at the first node whose max relative diff exceeds
/// `1e-2`, printing every field this diagnostic's own report needs: node,
/// kind, shape, and the first 8 values on both sides.
#[cfg(feature = "instrument")]
pub(super) fn compare_op_output_to_cpu(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    program: &[Op],
    node: NodeId,
    kind: &str,
    cpu_reference: &BTreeMap<NodeId, alloc::vec::Vec<f32>>,
) -> Result<Option<f32>, MetalError> {
    let Some(cpu_values) = cpu_reference.get(&node) else {
        return Ok(None);
    };
    let Some((buffer, offset)) = device_buffers.get(&node) else {
        return Ok(None);
    };
    let shape = prepared.shapes.of(node).to_vec();
    let dtype = gpu_dtype(program, &prepared.index_nodes, &prepared.resolved, node);
    let metal_values = read_back(buffer, *offset, element_count(&shape), node, dtype)?;
    let max_rel_diff = metal_values
        .iter()
        .zip(cpu_values.iter())
        .map(|(metal_value, cpu_value)| (metal_value - cpu_value).abs() / cpu_value.abs().max(1e-6))
        .fold(0.0_f32, f32::max);
    if max_rel_diff > 1e-2 {
        let metal_first: alloc::vec::Vec<f32> = metal_values.iter().copied().take(8).collect();
        let cpu_first: alloc::vec::Vec<f32> = cpu_values.iter().copied().take(8).collect();
        let kind_owned = kind.to_string();
        debug!(
            node = node.0,
            kind = %kind_owned,
            shape = ?shape,
            metal_first = ?metal_first,
            cpu_first = ?cpu_first,
            max_rel_diff = max_rel_diff as f64,
            "cpu_compare: divergent op output found"
        );
    }
    Ok(Some(max_rel_diff))
}

#[cfg(feature = "instrument")]
pub(super) const CPU_BOUND_COMPARE_CAP_BYTES: usize = 64 * 1024 * 1024;

#[cfg(feature = "instrument")]
pub(super) fn validate_selected_expert_routes(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    program: &[Op],
    bound: &BoundOp,
    expert_buffers: Option<&ExpertSourceBuffers>,
) -> Result<(), MetalError> {
    let Some(selected) = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map(NodeId)
    else {
        return Ok(());
    };
    if selected != bound.node {
        return Ok(());
    }
    let Some(expert_buffers) = expert_buffers else {
        eprintln!("metal_expert_routes_missing_buffers node={:?}", bound.node);
        return Ok(());
    };
    eprintln!(
        "metal_expert_routes_begin node={:?} source_node={:?} extents={:?} operands={:?}",
        bound.node,
        expert_buffers.node,
        bound.extents,
        bound
            .operands()
            .iter()
            .map(|(source, layout, lookup)| (
                *source,
                layout.clone(),
                lookup.as_ref().map(|value| (
                    value.indices,
                    value.index_layout.clone(),
                    value.element_stride,
                    value.extent
                ))
            ))
            .collect::<Vec<_>>()
    );
    for (source, _, lookup) in bound.all_read_sources() {
        if *source != expert_buffers.node {
            continue;
        }
        let Some(lookup) = lookup else {
            continue;
        };
        let Some((lookup_buffer, lookup_offset)) = device_buffers.get(&lookup.indices) else {
            return Err(MetalError::UnresolvedHazardOperand {
                node: lookup.indices,
            });
        };
        let lookup_values = read_back(
            lookup_buffer,
            *lookup_offset,
            element_count(prepared.shapes.of(lookup.indices)),
            lookup.indices,
            gpu_dtype(program, &prepared.index_nodes, &prepared.resolved, lookup.indices),
        )?;
        eprintln!(
            "metal_expert_lookup node={:?} values={:?}",
            lookup.indices, lookup_values
        );
        let mut seen = Vec::new();
        for value in lookup_values {
            let expert = value as u32;
            if value < 0.0 || value.fract() != 0.0 {
                return Err(MetalError::ExpertSourceUnsupported {
                    node: bound.node,
                    reason: "route index was not a non-negative integer",
                });
            }
            let Some(descriptor) = expert_buffers.descriptor_records.get(expert as usize) else {
                return Err(MetalError::ExpertSourceUnsupported {
                    node: bound.node,
                    reason: "route index exceeded the descriptor table",
                });
            };
            if descriptor.expert_index != expert
                || descriptor.byte_length == 0
                || descriptor.out_dim == 0
                || descriptor.in_dim == 0
            {
                eprintln!(
                    "metal_expert_route_invalid node={:?} expert={} descriptor={descriptor:?}",
                    bound.node, expert
                );
                return Err(MetalError::ExpertSourceUnsupported {
                    node: bound.node,
                    reason: "route selected an invalid expert descriptor",
                });
            }
            let descriptor_summary = (
                expert,
                descriptor.codec,
                descriptor.byte_offset,
                descriptor.byte_length,
            );
            if !seen.contains(&descriptor_summary) {
                seen.push(descriptor_summary);
            }
        }
        eprintln!(
            "metal_expert_routes node={:?} lookup={:?} routes={seen:?}",
            bound.node, lookup.indices
        );
    }
    Ok(())
}

/// Captures only the already-live dense operands of the selected bound op.
/// The snapshot happens before that op is encoded, so the CPU interpreter
/// receives the same payload the Metal kernel is about to read without
/// retaining or recursively evaluating any other graph node.
#[cfg(feature = "instrument")]
pub(super) fn evaluate_selected_bound_cpu(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
    bound: &BoundOp,
) -> Result<Option<alloc::vec::Vec<f32>>, MetalError> {
    let selected = std::env::var("PROXIMA_METAL_COMPARE_BOUND_NODE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .map(NodeId);
    if selected != Some(bound.node) {
        return Ok(None);
    }

    for (source, _, lookup) in bound.all_read_sources() {
        if lookup.is_some() {
            // Gathered operands require the full source-aware CPU evaluator;
            // this local snapshot intentionally cannot interpret descriptor
            // indexed packed bytes.
            return Ok(None);
        }
        if packed_operands.contains_key(source) {
            // A packed operand likewise has no valid dense snapshot here.
            return Ok(None);
        }
    }

    let sources: BTreeSet<NodeId> = bound
        .all_read_sources()
        .map(|(source, _, _)| *source)
        .collect();
    let output_elements = element_count(prepared.shapes.of(bound.node));
    let snapshot_elements = sources.iter().try_fold(output_elements, |total, source| {
        total.checked_add(element_count(prepared.shapes.of(*source)))
    });
    let snapshot_bytes = snapshot_elements
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .unwrap_or(usize::MAX);
    if snapshot_bytes > CPU_BOUND_COMPARE_CAP_BYTES {
        return Err(MetalError::CpuBoundComparisonOverCap {
            node: bound.node,
            bytes: snapshot_bytes,
            cap_bytes: CPU_BOUND_COMPARE_CAP_BYTES,
        });
    }

    let mut owned_buffers: alloc::vec::Vec<Option<alloc::vec::Vec<f32>>> =
        alloc::vec![None; program.len()];
    for source in sources {
        let (buffer, offset) = device_buffers
            .get(&source)
            .ok_or(MetalError::UnresolvedHazardOperand { node: source })?;
        let shape = prepared.shapes.of(source);
        let dtype = gpu_dtype(program, &prepared.index_nodes, &prepared.resolved, source);
        owned_buffers[source.0 as usize] = Some(read_back(
            buffer,
            *offset,
            element_count(shape),
            source,
            dtype,
        )?);
    }
    let borrowed_buffers: alloc::vec::Vec<Option<&[f32]>> = owned_buffers
        .iter()
        .map(|buffer| buffer.as_deref())
        .collect();
    let mut output = alloc::vec![0.0; output_elements];
    let packed_nodes: alloc::vec::Vec<NodeId> = packed_operands.keys().copied().collect();
    proxima_tensor::evaluate_bound_f32_into(bound, &borrowed_buffers, &packed_nodes, &mut output)?;
    Ok(Some(output))
}

// same reasoning as `FiniteCheckFailure` above: first divergent index,
// metal value, cpu value, absolute diff, relative diff.
#[cfg(feature = "instrument")]
pub(super) type BoundMismatch = (usize, f32, f32, f32, f32);

#[cfg(feature = "instrument")]
pub(super) fn compare_bound_f32(
    node: NodeId,
    metal_values: &[f32],
    cpu_values: &[f32],
) -> Result<Option<BoundMismatch>, MetalError> {
    if metal_values.len() != cpu_values.len() {
        return Err(MetalError::CpuBoundComparisonLengthMismatch {
            node,
            metal_len: metal_values.len(),
            cpu_len: cpu_values.len(),
        });
    }
    let mut max_relative = 0.0_f32;
    let mut first_mismatch = None;
    for (element, (metal_value, cpu_value)) in metal_values.iter().zip(cpu_values).enumerate() {
        let finite = metal_value.is_finite() && cpu_value.is_finite();
        let absolute = (metal_value - cpu_value).abs();
        let relative = if finite {
            absolute / cpu_value.abs().max(1e-6)
        } else {
            f32::INFINITY
        };
        max_relative = max_relative.max(relative);
        if (!finite || relative > 1e-2) && first_mismatch.is_none() {
            first_mismatch = Some((element, *metal_value, *cpu_value, relative));
        }
    }
    Ok(first_mismatch.map(|(element, metal, cpu, relative)| {
        (element, metal, cpu, relative, max_relative.max(relative))
    }))
}

#[cfg(feature = "instrument")]
pub(super) fn report_bound_operands(
    bound: &BoundOp,
    prepared: &Prepared,
    packed_operands: &PackedOperands,
    program: &[Op],
    expert_buffers: Option<&ExpertSourceBuffers>,
    device_buffers: Option<&BTreeMap<NodeId, DeviceBuffer>>,
    sample_element: Option<usize>,
) {
    let sample_coordinate = sample_element.map(|mut element| {
        let mut coordinate = alloc::vec![0_u64; prepared.shapes.of(bound.node).len()];
        for axis in (0..coordinate.len()).rev() {
            let extent = prepared.shapes.of(bound.node)[axis] as usize;
            coordinate[axis] = (element % extent) as u64;
            element /= extent;
        }
        coordinate
    });
    for (operand_index, (source, layout, lookup)) in bound.all_read_sources().enumerate() {
        eprintln!(
            "metal_bound_operand node={:?} operand={} source={source:?} source_name={:?} source_shape={:?} layout={layout:?} lookup={lookup:?} packed_codec={:?} expert_source={}",
            bound.node,
            operand_index,
            program[source.0 as usize].name(),
            prepared.shapes.of(*source),
            packed_operands.get(source),
            expert_buffers.is_some_and(|buffers| buffers.node == *source),
        );
        if let (Some(device_buffers), Some(coordinate)) =
            (device_buffers, sample_coordinate.as_deref())
            && expert_buffers.is_none()
            && let Some((buffer, offset)) = device_buffers.get(source)
            && let Ok(values) = read_back(
                buffer,
                *offset,
                element_count(prepared.shapes.of(*source)),
                *source,
                gpu_dtype(program, &prepared.index_nodes, &prepared.resolved, *source),
            )
        {
            let source_offset = layout.offset_of(coordinate);
            let source_value = usize::try_from(source_offset)
                .ok()
                .and_then(|index| values.get(index).copied());
            eprintln!(
                "metal_bound_operand_sample node={:?} operand={} coordinate={coordinate:?} source_offset={source_offset} value={source_value:?}",
                bound.node, operand_index
            );
        }
    }
    if let Some(expert_buffers) = expert_buffers
        && let Some((weight_source, weight_layout, Some(lookup))) = bound.all_read_sources().next()
        && let Some((lookup_buffer, lookup_offset)) =
            device_buffers.and_then(|buffers| buffers.get(&lookup.indices))
        && let Ok(route_values) = read_back(
            lookup_buffer,
            *lookup_offset,
            element_count(prepared.shapes.of(lookup.indices)),
            lookup.indices,
            gpu_dtype(program, &prepared.index_nodes, &prepared.resolved, lookup.indices),
        )
        && let Some(route) = route_values.first().copied().map(|value| value as usize)
        && let Some(descriptor) = expert_buffers.descriptor_records.get(route)
        && descriptor.codec == Codec::Q2K
        && let Some((_, _, None)) = bound.all_read_sources().nth(1)
        && let Some((activation_source, _, _)) = bound.all_read_sources().nth(1)
        && let Some((activation_buffer, activation_offset)) =
            device_buffers.and_then(|buffers| buffers.get(activation_source))
        && let Ok(activation_values) = read_back(
            activation_buffer,
            *activation_offset,
            element_count(prepared.shapes.of(*activation_source)),
            *activation_source,
            gpu_dtype(program, &prepared.index_nodes, &prepared.resolved, *activation_source),
        )
    {
        let payload_pointer = expert_buffers.payloads.contents().as_ptr().cast::<u8>();
        let payload_start = expert_buffers
            .payload_offset
            .saturating_add(descriptor.byte_offset);
        let payload_end = payload_start.saturating_add(descriptor.byte_length);
        if payload_end <= expert_buffers.payloads.length() {
            // `Q2_K` is the only dynamic low codec in this diagnostic arm;
            // decode the same 84-byte block geometry the emitted helper uses
            // and dot it with the already-read activation to separate a
            // payload/index fault from an upstream activation difference.
            let payload = unsafe {
                core::slice::from_raw_parts(
                    payload_pointer.add(payload_start),
                    descriptor.byte_length,
                )
            };
            let mut dot = 0.0_f32;
            for (index, activation) in activation_values.iter().enumerate() {
                if index >= usize::try_from(descriptor.in_dim).unwrap_or(0) {
                    break;
                }
                let block_start = (index / 256) * 84;
                let block = &payload[block_start..block_start + 84];
                let local = index % 256;
                let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
                let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
                let chunk = local / 128;
                let within = local % 128;
                let group = within / 32;
                let sub_block = chunk * 8 + group * 2 + usize::from(within % 32 >= 16);
                let scale_min = block[sub_block];
                let level = (block[16 + chunk * 32 + (within % 32)] >> (2 * group)) & 3;
                let weight = d * f32::from(scale_min & 0x0f) * f32::from(level)
                    - dmin * f32::from(scale_min >> 4);
                dot += weight * *activation;
            }
            eprintln!(
                "metal_expert_debug node={:?} route={} codec={:?} payload_offset={} activation_source={activation_source:?} weight_layout={weight_layout:?} weight0={} activation0={} dot={dot}",
                bound.node,
                route,
                descriptor.codec,
                payload_start,
                q2k_debug_value(payload, 0),
                activation_values.first().copied().unwrap_or_default(),
            );
        }
        let _ = weight_source;
    }
}

/// `PROXIMA_HEAD_DEBUG_NODES` (comma-separated `NodeId` integers) parsed
/// once per process -- the ordered target set `OWNER_BRIEF_gemma_head`'s
/// per-position identity print and `PROXIMA_HEAD_SENTINEL_FILL` both key
/// against. Index 0 is the production head root; every later index is a
/// `PROXIMA_HEAD_REPEATS` duplicate, per the caller's own convention when
/// setting the env var. `None` (the ordinary, non-debugging run) short-
/// circuits both call sites to a single `is_none()` check.
#[cfg(feature = "instrument")]
fn head_debug_target_nodes() -> Option<&'static [NodeId]> {
    static TARGETS: std::sync::OnceLock<Vec<NodeId>> = std::sync::OnceLock::new();
    let targets = TARGETS.get_or_init(|| {
        std::env::var("PROXIMA_HEAD_DEBUG_NODES")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|entry| entry.trim().parse::<u32>().ok())
                    .map(NodeId)
                    .collect()
            })
            .unwrap_or_default()
    });
    (!targets.is_empty()).then_some(targets.as_slice())
}

/// Overwrites `buffer[offset..offset + byte_length]` with the IEEE-754
/// quiet-NaN bit pattern `0x7fc0_0000`, repeated per `f32` lane -- the same
/// host-visible `contents()` write path `write_plan_uniform_bytes` uses to
/// seed a plan-owned uniform buffer, applied here to a duplicate head's own
/// output range BEFORE this command buffer is committed. If the encoded
/// dispatch for that position genuinely writes this range, `commit` +
/// `waitUntilCompleted` overwrites every sentinel byte; if the dispatch is
/// skipped, aliases an already-written range, or never reaches this buffer,
/// the sentinel survives the wait and the caller's post-wait bytes check
/// catches it directly rather than inferring non-execution from timing.
#[cfg(feature = "instrument")]
fn head_debug_fill_sentinel(buffer: &MetalBuffer, offset: usize, byte_length: usize) {
    const SENTINEL_BITS: u32 = 0x7fc0_0000;
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared` (every arena/output-placed
    // buffer this module hands `encode_op` is), and `offset + byte_length`
    // is the exact range `arena_placement`/`output_placed` already resolved
    // for this position's own output -- the same range `encode_op`'s later
    // write targets.
    let destination = unsafe {
        core::slice::from_raw_parts_mut(
            pointer.as_ptr().cast::<u8>().add(offset),
            byte_length,
        )
    };
    let (chunks, _remainder) = destination.as_chunks_mut::<4>();
    for chunk in chunks {
        *chunk = SENTINEL_BITS.to_ne_bytes();
    }
}

#[cfg(feature = "instrument")]
pub(super) fn q2k_debug_value(blocks: &[u8], index: usize) -> f32 {
    let block_start = (index / 256) * 84;
    let block = &blocks[block_start..block_start + 84];
    let local = index % 256;
    let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
    let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();
    let chunk = local / 128;
    let within = local % 128;
    let group = within / 32;
    let sub_block = chunk * 8 + group * 2 + usize::from(within % 32 >= 16);
    let scale_min = block[sub_block];
    let level = (block[16 + chunk * 32 + (within % 32)] >> (2 * group)) & 3;
    d * f32::from(scale_min & 0x0f) * f32::from(level) - dmin * f32::from(scale_min >> 4)
}

