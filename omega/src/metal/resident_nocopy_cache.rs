use super::*;

/// Shared identity check both the no-copy and resident-copy caches enforce:
/// a `name` hit is served only when the OFFERED host pointer and byte length
/// match what this name was first cached with; any other name hit is
/// [`MetalError::ResidentNameRebound`], never a stale serve and never a
/// silent replace. Composes with [`RESIDENT_BUFFERS`]/[`upload_resident_copy`]
/// and [`NOCOPY_BUFFERS`]/[`upload_block_no_copy`], the two callers that own
/// their own separate maps (different soundness arguments -- see each map's
/// own doc) but share this one lookup rule.
pub(super) fn resident_name_lookup(
    cache: &RefCell<BTreeMap<String, (usize, usize, MetalBuffer)>>,
    name: &str,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<Option<MetalBuffer>, MetalError> {
    let offered_address = pointer as usize;
    let Some((cached_address, cached_length, buffer)) = cache.borrow().get(name).cloned() else {
        return Ok(None);
    };
    if cached_address == offered_address && cached_length == byte_length {
        return Ok(Some(buffer));
    }
    Err(MetalError::ResidentNameRebound {
        name: name.to_string(),
        cached_len: cached_length,
        offered_len: byte_length,
    })
}

/// Counts a fresh [`Plan`]'s per-node fast path resolving a resident block
/// straight from the process-global caches (`NOCOPY_BUFFERS`,
/// `RESIDENT_BUFFERS`, the checkpoint mapping) instead of taking the
/// upload path -- the direct witness that a KV-bucket-boundary reshape does
/// not re-walk the checkpoint's weight blocks through
/// `upload_block`/`upload_packed_bytes` just to re-derive a buffer those
/// caches already hold.
pub static PLAN_HANDOFF_REUSES: Counter = Counter::new("omega.metal.plan_handoff_reuses");

/// Resolves `(pointer, byte_length)` against the GLOBAL, name-keyed residency
/// caches without recording it as a fresh "offer" -- the fast path a brand
/// new [`Plan`] takes so its own empty `device_buffers`/`block_identity`
/// (see [`plan`]'s doc) never forces a resident weight block through the
/// host round trip a PRIOR plan already paid once. `Ok(None)` means none of
/// the three caches have this host range yet, so the caller falls through to
/// `upload_block`/`upload_packed_bytes`'s normal offer path unchanged.
///
/// Never creates a buffer and never mutates a cache -- a rebind (the offered
/// `(pointer, byte_length)` disagreeing with what a name already holds) is
/// [`MetalError::ResidentNameRebound`] via [`resident_name_lookup`], the same
/// contract [`upload_block_no_copy`]/[`upload_resident_copy`] enforce, never
/// a silent reuse of a different allocation under the same name.
pub(super) fn cross_plan_resident_reuse(
    resident_name: Option<&str>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<Option<(MetalBuffer, usize)>, MetalError> {
    if let Some(name) = resident_name {
        if let Some(buffer) =
            NOCOPY_BUFFERS.with(|cache| resident_name_lookup(cache, name, pointer, byte_length))?
        {
            counter!(PLAN_HANDOFF_REUSES, 1);
            return Ok(Some((buffer, 0)));
        }
        if let Some(buffer) = RESIDENT_BUFFERS
            .with(|cache| resident_name_lookup(cache, name, pointer, byte_length))?
        {
            counter!(PLAN_HANDOFF_REUSES, 1);
            return Ok(Some((buffer, 0)));
        }
    }
    let Some((base, mapping_length)) = CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow()) else {
        return Ok(None);
    };
    let address = pointer as usize;
    if address < base || address + byte_length > base + mapping_length {
        return Ok(None);
    }
    let rounded_length = mapping_length.div_ceil(page_size()) * page_size();
    let Some(buffer) = NOCOPY_BUFFERS.with(|cache| {
        resident_name_lookup(
            cache,
            CHECKPOINT_MAPPING_NOCOPY_NAME,
            base as *const c_void,
            rounded_length,
        )
    })?
    else {
        return Ok(None);
    };
    counter!(PLAN_HANDOFF_REUSES, 1);
    Ok(Some((buffer, address - base)))
}

/// Counts entries `NOCOPY_BUFFERS` actually holds right now — the direct
/// witness that gating the cache on `resident` stops it growing without
/// bound. A non-resident page-aligned block (a KV-cache row that happened to
/// cross a page boundary) never reaches this map at all.
#[must_use]
pub fn nocopy_cache_len() -> usize {
    NOCOPY_BUFFERS.with(|cache| cache.borrow().len())
}

/// Returns the lengths of every retained no-copy buffer, keyed by the
/// resident identity that owns its lifetime. This is a load-time/diagnostic
/// census only; the serving path uses [`nocopy_cache_len`] and never walks it.
#[must_use]
pub fn nocopy_cache_lengths() -> Vec<(String, u64)> {
    NOCOPY_BUFFERS.with(|cache| {
        cache
            .borrow()
            .iter()
            .map(|(name, (_, length, buffer))| {
                (name.clone(), (*length).max(buffer.length()) as u64)
            })
            .collect()
    })
}

/// metal-visible byte lengths of the whole-file checkpoint and expert mmap
/// wrappers currently retained by the no-copy cache. these are the driver's
/// actual `MTLBuffer::length()` values, not tensor-shape estimates.
#[must_use]
pub fn mapping_buffer_allocated_bytes() -> (u64, u64) {
    NOCOPY_BUFFERS.with(|cache| {
        let cache = cache.borrow();
        let buffer_length = |name: &str| {
            cache
                .get(name)
                .map_or(0, |(_, _, buffer)| buffer.length() as u64)
        };
        (
            buffer_length(CHECKPOINT_MAPPING_NOCOPY_NAME),
            buffer_length(EXPERT_MAPPING_NOCOPY_NAME),
        )
    })
}

/// Counts the no-copy wrappers this thread reused instead of recreating —
/// the direct witness that a serving loop stops re-wiring its weights.
pub static NOCOPY_BUFFER_REUSES: Counter = Counter::new("omega.metal.upload_block.nocopy_reuse");

/// The zero-copy path: shares `pointer`'s memory directly with the GPU
/// instead of duplicating it. Sound only because every caller of
/// [`upload_block`] binds the returned buffer to a `device const float*`
/// kernel argument (see `msl::kernel_signature`) — the GPU never writes
/// through it, matching the `&[f32]` (never `&mut`) the caller handed us —
/// and because [`execute`] `waitUntilCompleted`s the one command buffer
/// every op (including this buffer's reads) is encoded into before
/// [`upload_block_as_float`]'s caller-owned slice's borrow can end.
///
/// Cached in [`NOCOPY_BUFFERS`] — reachable ONLY when the caller already
/// classified this address `resident` under `name` (see the call sites in
/// [`upload_block_as_float`]/[`upload_packed_bytes`]/[`checkpoint_mapping_offset`]);
/// a non-resident page-aligned block takes [`upload_block_no_copy_uncached`]
/// instead, which shares this function's Metal call but never remembers the
/// wrapper -- an unnamed upload is never cached here. A `name` hit whose
/// offered pointer or byte length disagrees with what is cached is a caller
/// contract violation ([`Plan::mark_resident`]'s doc), not a fresher version
/// of the same buffer, so it is [`MetalError::ResidentNameRebound`], never a
/// stale hit and never a silent re-upload. See [`resident_name_lookup`] and
/// [`NOCOPY_BUFFERS`]'s own doc.
pub(super) fn upload_block_no_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    if let Some(existing) =
        NOCOPY_BUFFERS.with(|cache| resident_name_lookup(cache, name, pointer, byte_length))?
    {
        counter!(NOCOPY_BUFFER_REUSES, 1);
        return Ok(existing);
    }
    let buffer = create_no_copy_buffer(device, pointer, byte_length)?;
    NOCOPY_BUFFERS.with(|cache| {
        cache.borrow_mut().insert(
            name.to_string(),
            (pointer as usize, byte_length, buffer.clone()),
        )
    });
    Ok(buffer)
}

/// The uncached counterpart to [`upload_block_no_copy`]: still hands Metal
/// the caller's own pointer directly (zero-copy, sound for this one
/// `execute` call for the exact reason [`upload_block_no_copy`]'s doc
/// gives), but never inserts the wrapper into [`NOCOPY_BUFFERS`]. Taken
/// whenever a page-aligned block's node is NOT [`Plan::mark_resident`]-classified
/// static — an ephemeral, growing buffer (a KV-cache row that crossed a page
/// boundary) can cross that same alignment by coincidence on every call, and
/// caching it would key a permanent entry off an address whose CONTENTS,
/// and even whose owning allocation, changes underneath it.
pub(super) fn upload_block_no_copy_uncached(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    create_no_copy_buffer(device, pointer, byte_length)
}

/// The `newBufferWithBytesNoCopy` FFI call itself, shared by
/// [`upload_block_no_copy`] and [`upload_block_no_copy_uncached`] — caching
/// is entirely the callers' concern, not this function's.
pub(super) fn create_no_copy_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    // SAFETY: `pointer` is non-null (it comes from a non-empty slice) and,
    // per `is_page_aligned`, page-aligned with a page-aligned `byte_length`
    // — `newBufferWithBytesNoCopy`'s documented precondition. Passing `None`
    // as the deallocator tells Metal it never owns this memory, so it is
    // never freed or written out from under the caller.
    let pointer = unsafe { NonNull::new_unchecked(pointer as *mut c_void) };
    unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            pointer,
            byte_length,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
    .ok_or_else(|| MetalError::CompileFailed {
        log: "device refused a no-copy shared buffer for a page-aligned block input".to_string(),
    })
}

pub(super) fn upload_block_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    // SAFETY: `pointer` is a live, non-null address for the duration of this
    // call (borrowed from the caller's own `&[f32]`, or a locally owned
    // narrowed `Vec<f16>` that outlives this call), so it stays valid while
    // `newBufferWithBytes_length_options` copies from it.
    let pointer = unsafe { NonNull::new_unchecked(pointer as *mut c_void) };
    unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            byte_length,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| MetalError::CompileFailed {
        log: "device refused to allocate a shared buffer for a block input".to_string(),
    })
}

thread_local! {
    /// Copied device buffers for blocks [`Plan::mark_resident`] classified as
    /// the caller's own static weights -- a SEPARATE map from
    /// [`NOCOPY_BUFFERS`], not a shared one, because the two caches rest on
    /// different soundness arguments: a no-copy entry is safe to reuse
    /// unconditionally because it aliases whatever is CURRENTLY at that
    /// address (never stale by construction); this cache instead reuses a
    /// SNAPSHOT taken at first upload, which is sound only because the
    /// caller already proved -- by name, once, in `mark_resident` -- that
    /// the NAME holds a model weight nothing overwrites again. Keyed on that
    /// name, never on `(pointer, byte_length)` alone: a host address is a
    /// property of an allocation's LIFETIME, and a short-lived buffer (an
    /// activation vector, say) can be freed and a same-sized,
    /// differently-contented allocation can land at the identical address on
    /// a later call -- `mark_resident`'s own proof is about the NAME the
    /// caller declared static, not about any address that name's data
    /// happened to occupy once. See `proxima-tensor/docs/discipline.md` ROW 70.
    ///
    /// The stored `(usize, usize, MetalBuffer)` is the host pointer and byte
    /// length this entry was uploaded from, alongside the device copy --
    /// ROW 332's shape was a caller marking a DIFFERENT host buffer resident
    /// under a REUSED name, which a name-only lookup cannot distinguish from
    /// a legitimate cache hit. [`upload_resident_copy`] checks both against
    /// what the caller offers on every lookup and refuses to serve a
    /// mismatch -- see that function's own doc.
    static RESIDENT_BUFFERS: RefCell<BTreeMap<String, (usize, usize, MetalBuffer)>> =
        RefCell::new(BTreeMap::new());
}

/// How many resident (caller-declared-static) blocks took a real copy versus
/// how many were served from `RESIDENT_BUFFERS` instead -- the direct
/// witness that the ~5.84 GB/token `proxima-tensor/docs/discipline.md` ROW 82
/// measured moving through `upload_block_copy` on every step now moves
/// exactly once per distinct weight buffer, never once per token.
pub static RESIDENT_BUFFER_UPLOADS: Counter =
    Counter::new("omega.metal.upload_block.resident_upload");
pub static RESIDENT_BUFFER_REUSES: Counter =
    Counter::new("omega.metal.upload_block.resident_reuse");

/// Entries `RESIDENT_BUFFERS` holds right now -- the direct witness that
/// residency reuse tracks the caller's declared NAME set, not the address
/// space: this stays exactly the plan's resident-name count across repeated
/// `execute_plan` calls for the same names, regardless of how many times the
/// host allocator has reused an address underneath them.
#[must_use]
pub fn resident_cache_len() -> usize {
    RESIDENT_BUFFERS.with(|cache| cache.borrow().len())
}

/// Returns the retained bytes in the caller-declared resident-copy cache.
/// This is diagnostic-only: the serving path uses [`resident_cache_len`] and
/// never walks the cache. The sum is the device-buffer length, not the host
/// slice length, so it can be compared directly with Metal's allocation
/// counter when a routed run unexpectedly retains the whole checkpoint.
#[must_use]
pub fn resident_cache_bytes() -> u64 {
    RESIDENT_BUFFERS.with(|cache| {
        cache
            .borrow()
            .values()
            .map(|(_, _, buffer)| buffer.length() as u64)
            .sum()
    })
}

/// Returns every retained resident-copy entry's name and device-buffer bytes.
/// This is a load-time/diagnostic census only; execution never enumerates it.
#[must_use]
pub fn resident_cache_lengths() -> Vec<(String, u64)> {
    RESIDENT_BUFFERS.with(|cache| {
        cache
            .borrow()
            .iter()
            .map(|(name, (_, _, buffer))| (name.clone(), buffer.length() as u64))
            .collect()
    })
}

/// The copy-path counterpart to [`upload_block_no_copy`]: called only for a
/// block [`upload_block_as_float`]/[`upload_packed_bytes`] already found
/// misaligned AND [`Plan::mark_resident`] already classified `name` as
/// static, so unlike [`upload_block_copy`] this one is allowed to remember
/// the buffer it creates and hand the SAME one back next time the SAME
/// `name` shows up -- sound only because that classification, not an
/// address guess, is what proves the bytes behind `name` never change
/// again.
///
/// [`Plan::mark_resident`]'s doc is the contract this enforces: under a
/// legitimate caller, a resident name's host pointer and byte length never
/// change once bound. A `name` hit whose OFFERED host pointer or byte length
/// disagrees with what is cached is therefore not a fresher version of the
/// same weight -- it is a different host allocation that happened to reuse a
/// name a serving loop already marked resident (ROW 331/332's shape). This
/// cache cannot tell "the weight changed" from "the caller has a bug" (the
/// module doc's residency precondition rules the former case out), so it
/// never guesses: it refuses to serve, and never silently replaces the
/// cached entry, via [`MetalError::ResidentNameRebound`].
pub(super) fn upload_resident_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    pointer: *const c_void,
    byte_length: usize,
) -> Result<MetalBuffer, MetalError> {
    if let Some(existing) =
        RESIDENT_BUFFERS.with(|cache| resident_name_lookup(cache, name, pointer, byte_length))?
    {
        counter!(RESIDENT_BUFFER_REUSES, 1);
        return Ok(existing);
    }
    counter!(COPYING_BUFFER_UPLOADS, 1);
    counter!(RESIDENT_BUFFER_UPLOADS, 1);
    counter!(BLOCK_COPIED_BYTES, byte_length as u64);
    let buffer = upload_block_copy(device, pointer, byte_length)?;
    RESIDENT_BUFFERS.with(|cache| {
        cache.borrow_mut().insert(
            name.to_string(),
            (pointer as usize, byte_length, buffer.clone()),
        )
    });
    Ok(buffer)
}

/// Evicts every buffer a checkpoint's own resident weight names cached in
/// `NOCOPY_BUFFERS`/`RESIDENT_BUFFERS` -- the drop-time counterpart to
/// `upload_block_no_copy`/`upload_resident_copy`'s insert. Removing by
/// NAME, never a blanket clear, is what lets a second, unrelated model
/// loaded on this same thread keep its own entries live after the first
/// model drops -- see `LoadedModel`'s own `Drop` impl in
/// `proxima-model-interop` for the caller. A name absent from either cache
/// (a resident block this run never actually uploaded) is silently
/// skipped.
pub fn release_resident_names<'name>(names: impl IntoIterator<Item = &'name str>) {
    for name in names {
        NOCOPY_BUFFERS.with(|cache| cache.borrow_mut().remove(name));
        RESIDENT_BUFFERS.with(|cache| cache.borrow_mut().remove(name));
    }
}

#[cfg(test)]
pub(super) fn reset_resident_cache_for_test() {
    RESIDENT_BUFFERS.with(|cache| cache.borrow_mut().clear());
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod resident_buffer_cache_tests {
    use core::mem::size_of_val;

    use super::{
        MetalBuffer, device_and_queue, read_back_as_device_f32, reset_resident_cache_for_test,
        resident_cache_len, upload_block_copy, upload_resident_copy,
    };

    fn pre_fix_address_keyed_lookup(
        cache: &mut alloc::collections::BTreeMap<(usize, usize), MetalBuffer>,
        device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
        pointer: *const core::ffi::c_void,
        byte_length: usize,
    ) -> MetalBuffer {
        let key = (pointer as usize, byte_length);
        if let Some(existing) = cache.get(&key) {
            return existing.clone();
        }
        let buffer = upload_block_copy(device, pointer, byte_length).expect("pre-fix copy upload");
        cache.insert(key, buffer.clone());
        buffer
    }

    #[test]
    fn pre_fix_address_keyed_cache_served_stale_content_across_names() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        let mut pre_fix_cache = alloc::collections::BTreeMap::new();

        let mut host_buffer = vec![1.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let byte_length = size_of_val(host_buffer.as_slice());

        let first = pre_fix_address_keyed_lookup(&mut pre_fix_cache, &device, pointer, byte_length);
        let first_content = read_back_as_device_f32(&first, 0, host_buffer.len());
        assert_eq!(first_content, vec![1.0_f32; 4096]);

        host_buffer.fill(2.0_f32);
        let second =
            pre_fix_address_keyed_lookup(&mut pre_fix_cache, &device, pointer, byte_length);
        let second_content = read_back_as_device_f32(&second, 0, host_buffer.len());
        assert_eq!(second_content, vec![1.0_f32; 4096]);
    }

    #[test]
    fn name_keyed_cache_never_serves_a_different_names_content() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let mut host_buffer = vec![1.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let byte_length = size_of_val(host_buffer.as_slice());

        let first = upload_resident_copy(&device, "resident_weight_one", pointer, byte_length)
            .expect("first resident upload");
        assert_eq!(
            read_back_as_device_f32(&first, 0, host_buffer.len()),
            vec![1.0_f32; 4096]
        );

        host_buffer.fill(2.0_f32);
        let second = upload_resident_copy(&device, "resident_weight_two", pointer, byte_length)
            .expect("second resident upload under a different name");
        assert_eq!(
            read_back_as_device_f32(&second, 0, host_buffer.len()),
            vec![2.0_f32; 4096]
        );
        assert_eq!(resident_cache_len(), 2);
    }

    #[test]
    fn the_same_resident_name_still_hits_the_cache() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let host_buffer = vec![3.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let byte_length = size_of_val(host_buffer.as_slice());

        upload_resident_copy(&device, "resident_weight_stable", pointer, byte_length)
            .expect("first upload for a stable resident name");
        assert_eq!(resident_cache_len(), 1);

        let reuses_before = super::RESIDENT_BUFFER_REUSES.get();
        upload_resident_copy(&device, "resident_weight_stable", pointer, byte_length)
            .expect("second upload of the same name must hit the cache");
        assert_eq!(super::RESIDENT_BUFFER_REUSES.get(), reuses_before + 1);
        assert_eq!(resident_cache_len(), 1);
    }

    #[test]
    fn a_resident_name_rebound_to_a_different_host_pointer_is_rejected() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let first_host_buffer = vec![4.0_f32; 4096];
        let byte_length = size_of_val(first_host_buffer.as_slice());
        upload_resident_copy(
            &device,
            "resident_weight_rebound",
            first_host_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect("first upload establishes the cached entry");

        // a second, unrelated host allocation of the SAME length reusing the
        // SAME name -- exactly ROW 331/332's shape, reproduced at the driver
        // level instead of relying on a harness never naming two arms alike.
        let second_host_buffer = vec![5.0_f32; 4096];
        let uploads_before = super::RESIDENT_BUFFER_UPLOADS.get();
        let reuses_before = super::RESIDENT_BUFFER_REUSES.get();
        let error = upload_resident_copy(
            &device,
            "resident_weight_rebound",
            second_host_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect_err("a different host pointer under the same name must never be served");

        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
        assert_eq!(super::RESIDENT_BUFFER_UPLOADS.get(), uploads_before);
        assert_eq!(super::RESIDENT_BUFFER_REUSES.get(), reuses_before);
        assert_eq!(resident_cache_len(), 1);
    }

    #[test]
    fn a_resident_name_rebound_to_a_different_length_is_rejected() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let host_buffer = vec![6.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let full_length = size_of_val(host_buffer.as_slice());
        upload_resident_copy(&device, "resident_weight_grown", pointer, full_length)
            .expect("first upload establishes the cached entry");

        let uploads_before = super::RESIDENT_BUFFER_UPLOADS.get();
        let reuses_before = super::RESIDENT_BUFFER_REUSES.get();
        let error =
            upload_resident_copy(&device, "resident_weight_grown", pointer, full_length / 2)
                .expect_err("a different byte length under the same name must never be served");

        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
        assert_eq!(super::RESIDENT_BUFFER_UPLOADS.get(), uploads_before);
        assert_eq!(super::RESIDENT_BUFFER_REUSES.get(), reuses_before);
        assert_eq!(resident_cache_len(), 1);
    }

    /// A cache HIT never moves bytes -- `BLOCK_COPIED_BYTES` must reflect
    /// that (ROW 369): the counter used to fire at the call site, before the
    /// lookup that can avoid the copy, so a resident-cache hit was double
    /// counted as if it had re-copied the whole block every time.
    #[test]
    fn a_resident_cache_hit_adds_no_bytes_to_the_copied_counter() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();
        let _ = super::BLOCK_COPIED_BYTES.snapshot_and_reset();

        let host_buffer = vec![7.0_f32; 4096];
        let pointer = host_buffer.as_ptr().cast();
        let byte_length = size_of_val(host_buffer.as_slice());

        let copied_before_miss = super::BLOCK_COPIED_BYTES.get();
        upload_resident_copy(
            &device,
            "resident_weight_copied_counter",
            pointer,
            byte_length,
        )
        .expect("first upload is a real copy (a cache miss)");
        assert!(
            super::BLOCK_COPIED_BYTES.get() >= copied_before_miss + byte_length as u64,
            "a genuine copy must add its own byte length"
        );

        let copied_before_hit = super::BLOCK_COPIED_BYTES.get();
        upload_resident_copy(
            &device,
            "resident_weight_copied_counter",
            pointer,
            byte_length,
        )
        .expect("second upload of the same name hits the cache");
        assert_eq!(
            super::BLOCK_COPIED_BYTES.get(),
            copied_before_hit,
            "a cache hit moves zero bytes, so it must add zero"
        );
    }

    /// [`super::release_resident_names`]'s copy-path counterpart to
    /// `nocopy_buffer_cache_tests`'s own
    /// `release_resident_names_evicts_only_the_named_entry` -- the same
    /// two-fake-model shape, this time through the misaligned/resident-copy
    /// cache a real GGUF's odd-byte-offset tensors actually take.
    #[test]
    fn release_resident_names_evicts_only_the_named_resident_copy() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_resident_cache_for_test();

        let model_a = vec![1.0_f32; 64];
        let model_b = vec![2.0_f32; 64];
        let byte_length = size_of_val(model_a.as_slice());

        upload_resident_copy(
            &device,
            "fake_model_a.weight",
            model_a.as_ptr().cast(),
            byte_length,
        )
        .expect("bind fake model A's resident copy");
        upload_resident_copy(
            &device,
            "fake_model_b.weight",
            model_b.as_ptr().cast(),
            byte_length,
        )
        .expect("bind fake model B's resident copy");
        assert_eq!(
            resident_cache_len(),
            2,
            "both fake models' copies are cached"
        );

        super::release_resident_names(["fake_model_a.weight"]);
        assert_eq!(
            resident_cache_len(),
            1,
            "releasing model A's own name must evict exactly one entry"
        );

        let reuses_before = super::RESIDENT_BUFFER_REUSES.get();
        upload_resident_copy(
            &device,
            "fake_model_b.weight",
            model_b.as_ptr().cast(),
            byte_length,
        )
        .expect("model B's own name must still resolve after model A's release");
        assert!(
            super::RESIDENT_BUFFER_REUSES.get() > reuses_before,
            "model B's entry must still be a cache HIT -- it was never released"
        );
    }
}

#[cfg(test)]
pub(super) fn reset_nocopy_cache_for_test() {
    NOCOPY_BUFFERS.with(|cache| cache.borrow_mut().clear());
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod nocopy_buffer_cache_tests {
    use core::mem::size_of;

    use proxima_tensor::AlignedBuffer;

    use super::{
        device_and_queue, nocopy_cache_len, page_size, reset_nocopy_cache_for_test,
        upload_block_no_copy, upload_block_no_copy_uncached,
    };

    #[test]
    fn the_same_nocopy_name_still_hits_the_cache() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let buffer =
            AlignedBuffer::new(page / size_of::<f32>(), page).expect("page-aligned test buffer");
        let pointer = buffer.as_ptr().cast();
        let byte_length = buffer.len() * size_of::<f32>();

        upload_block_no_copy(&device, "nocopy_weight_stable", pointer, byte_length)
            .expect("first upload for a stable no-copy name");
        assert_eq!(nocopy_cache_len(), 1);

        let reuses_before = super::NOCOPY_BUFFER_REUSES.get();
        upload_block_no_copy(&device, "nocopy_weight_stable", pointer, byte_length)
            .expect("second upload of the same name must hit the cache");
        assert_eq!(super::NOCOPY_BUFFER_REUSES.get(), reuses_before + 1);
        assert_eq!(nocopy_cache_len(), 1);
    }

    #[test]
    fn a_nocopy_name_rebound_to_a_different_host_pointer_is_rejected() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let first_buffer =
            AlignedBuffer::new(page / size_of::<f32>(), page).expect("page-aligned test buffer");
        let byte_length = first_buffer.len() * size_of::<f32>();
        upload_block_no_copy(
            &device,
            "nocopy_weight_rebound",
            first_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect("first upload establishes the cached entry");

        // a second, unrelated page-aligned host allocation of the SAME
        // length reusing the SAME name -- ROW 334's own shape (a freed
        // ladder arm's `Vec<u8>` reused by a later arm at the identical
        // address), reproduced at the driver level instead of relying on a
        // test harness never marking an ephemeral buffer resident under a
        // reused name.
        let second_buffer =
            AlignedBuffer::new(page / size_of::<f32>(), page).expect("page-aligned test buffer");
        let reuses_before = super::NOCOPY_BUFFER_REUSES.get();
        let error = upload_block_no_copy(
            &device,
            "nocopy_weight_rebound",
            second_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect_err("a different host pointer under the same name must never be served");

        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
        assert_eq!(super::NOCOPY_BUFFER_REUSES.get(), reuses_before);
        assert_eq!(nocopy_cache_len(), 1);
    }

    #[test]
    fn a_nocopy_name_rebound_to_a_different_length_is_rejected() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let buffer = AlignedBuffer::new(2 * page / size_of::<f32>(), page)
            .expect("page-aligned test buffer");
        let pointer = buffer.as_ptr().cast();
        let full_length = buffer.len() * size_of::<f32>();
        upload_block_no_copy(&device, "nocopy_weight_grown", pointer, full_length)
            .expect("first upload establishes the cached entry");

        let reuses_before = super::NOCOPY_BUFFER_REUSES.get();
        let error = upload_block_no_copy(&device, "nocopy_weight_grown", pointer, full_length / 2)
            .expect_err("a different byte length under the same name must never be served");

        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
        assert_eq!(super::NOCOPY_BUFFER_REUSES.get(), reuses_before);
        assert_eq!(nocopy_cache_len(), 1);
    }

    #[test]
    fn an_unnamed_upload_is_never_cached() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let buffer =
            AlignedBuffer::new(page / size_of::<f32>(), page).expect("page-aligned test buffer");
        let pointer = buffer.as_ptr().cast();
        let byte_length = buffer.len() * size_of::<f32>();

        upload_block_no_copy_uncached(&device, pointer, byte_length)
            .expect("first uncached upload");
        upload_block_no_copy_uncached(&device, pointer, byte_length)
            .expect("second uncached upload of the identical range");

        assert_eq!(
            nocopy_cache_len(),
            0,
            "an unnamed (uncached) upload must never populate NOCOPY_BUFFERS"
        );
    }

    /// ROW: `LoadedModel::drop` releases exactly the dropped checkpoint's
    /// own resident names -- a SECOND, unrelated model's own no-copy entry
    /// (a different name, same thread, both live at once, the two-model
    /// shape the caller's `Drop` impl exists for) must survive untouched.
    #[test]
    fn release_resident_names_evicts_only_the_named_entry() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let page = page_size();
        let model_a = AlignedBuffer::new(page / size_of::<f32>(), page)
            .expect("page-aligned fixture for fake model A");
        let model_b = AlignedBuffer::new(page / size_of::<f32>(), page)
            .expect("page-aligned fixture for fake model B");
        let byte_length = model_a.len() * size_of::<f32>();

        upload_block_no_copy(
            &device,
            "fake_model_a.weight",
            model_a.as_ptr().cast(),
            byte_length,
        )
        .expect("bind fake model A's weight");
        upload_block_no_copy(
            &device,
            "fake_model_b.weight",
            model_b.as_ptr().cast(),
            byte_length,
        )
        .expect("bind fake model B's weight");
        assert_eq!(
            nocopy_cache_len(),
            2,
            "both fake models' weights are cached"
        );

        super::release_resident_names(["fake_model_a.weight"]);
        assert_eq!(
            nocopy_cache_len(),
            1,
            "releasing model A's own name must evict exactly one entry"
        );

        let reuses_before = super::NOCOPY_BUFFER_REUSES.get();
        upload_block_no_copy(
            &device,
            "fake_model_b.weight",
            model_b.as_ptr().cast(),
            byte_length,
        )
        .expect("model B's own name must still resolve after model A's release");
        assert!(
            super::NOCOPY_BUFFER_REUSES.get() > reuses_before,
            "model B's entry must still be a cache HIT -- it was never released"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(super) mod cross_plan_resident_reuse_tests {
    //! ROW 369: a KV-bucket-boundary reshape builds a brand-new [`Plan`]
    //! whose own `device_buffers`/`block_identity` start empty. These prove
    //! [`cross_plan_resident_reuse`] -- the fast path that new `Plan` takes
    //! against the process-global caches -- resolves a still-resident block
    //! WITHOUT re-registering it as an "offer", and still refuses a genuine
    //! rebind. `execute_plan_with_placements`'s own per-node loop is the
    //! caller; these test the decision it delegates, GPU-free of any real
    //! `Plan`/program construction.

    use core::mem::size_of_val;

    use super::{
        PLAN_HANDOFF_REUSES, cross_plan_resident_reuse, device_and_queue,
        register_checkpoint_mapping, reset_nocopy_cache_for_test, upload_packed_bytes,
    };

    #[test]
    fn a_checkpoint_mapped_block_is_resolved_from_the_global_cache_without_an_offer() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();

        let checkpoint = vec![0_u8; super::page_size() * 4];
        register_checkpoint_mapping(&checkpoint);
        // one real upload registers `CHECKPOINT_MAPPING_NOCOPY_NAME` in
        // `NOCOPY_BUFFERS` -- the state a PRIOR `Plan`'s own weight walk
        // would have already left behind by the time a reshape builds a new
        // one.
        let tensor_bytes = &checkpoint[64..128];
        upload_packed_bytes(&device, tensor_bytes, None).expect("first checkpoint-backed upload");

        let handoffs_before = PLAN_HANDOFF_REUSES.get();
        let pointer = tensor_bytes.as_ptr().cast();
        let byte_length = tensor_bytes.len();
        let resolved = cross_plan_resident_reuse(None, pointer, byte_length)
            .expect("a resident block already in the checkpoint mapping must resolve")
            .expect("the checkpoint mapping is registered, so this must be Some");

        assert_eq!(resolved.1, 64, "the offset into the shared mapping buffer");
        assert_eq!(
            PLAN_HANDOFF_REUSES.get(),
            handoffs_before + 1,
            "a global-cache hit must be counted as a handoff, not an offer"
        );
    }

    #[test]
    fn a_host_range_the_global_caches_have_never_seen_is_not_resolved() {
        reset_nocopy_cache_for_test();
        let never_registered = vec![1.0_f32; 16];
        let byte_length = size_of_val(never_registered.as_slice());
        let resolved =
            cross_plan_resident_reuse(None, never_registered.as_ptr().cast(), byte_length)
                .expect("a miss is Ok(None), never an error");
        assert!(
            resolved.is_none(),
            "the caller must fall through to the normal offer path on a miss"
        );
    }

    #[test]
    fn a_resident_name_rebound_to_different_bytes_is_rejected_not_silently_reused() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        super::reset_resident_cache_for_test();

        let first_host_buffer = vec![8.0_f32; 4096];
        let byte_length = size_of_val(first_host_buffer.as_slice());
        super::upload_resident_copy(
            &device,
            "resident_weight_cross_plan_rebound",
            first_host_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect("first upload establishes the cached entry a later plan could try to reuse");

        // a later `Plan` offering the SAME resident name against a
        // DIFFERENT host allocation -- the exact contract violation
        // `cross_plan_resident_reuse` must never paper over with a silent
        // reuse of the wrong buffer.
        let second_host_buffer = vec![9.0_f32; 4096];
        let error = cross_plan_resident_reuse(
            Some("resident_weight_cross_plan_rebound"),
            second_host_buffer.as_ptr().cast(),
            byte_length,
        )
        .expect_err("a rebind under the same name must never resolve, even from this fast path");
        assert!(matches!(
            error,
            super::MetalError::ResidentNameRebound { .. }
        ));
    }
}

/// Always copies — see the module doc's "Host buffer upload" section for
/// why a freshly narrowed `Vec<f16>` can never take the no-copy path: it
/// drops the instant this function returns, so no-copy would hand Metal a
/// dangling pointer.
pub(super) fn upload_block_as_half(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
) -> Result<MetalBuffer, MetalError> {
    if data.is_empty() {
        return allocate_buffer(device, 0, DType::Float16);
    }
    let narrowed: Vec<f16> = data.iter().map(|value| f16::from_f32(*value)).collect();
    let byte_length = size_of_val(narrowed.as_slice());
    let pointer = narrowed.as_ptr().cast::<c_void>();
    counter!(COPYING_BUFFER_UPLOADS, 1);
    upload_block_copy(device, pointer, byte_length)
}

/// Allocates a `gather_count`-long `uint` buffer for a dispatch's gather
/// faults and zero-fills it — a freshly allocated `MTLBuffer`'s contents are
/// undefined, and a slot left as garbage would read as a spurious fault.
pub(super) fn allocate_fault_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    gather_count: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let byte_length = gather_count.max(1) * size_of::<u32>();
    let buffer = device
        .newBufferWithLength_options(byte_length, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| MetalError::CompileFailed {
            log: "device refused to allocate the gather fault buffer".to_string(),
        })?;
    zero_fault_buffer(&buffer, gather_count);
    Ok(buffer)
}

pub(super) fn zero_fault_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, gather_count: usize) {
    let pointer = buffer.contents();
    // SAFETY: `buffer` is `storageModeShared` and was sized to at least
    // `gather_count` `u32`s by `allocate_fault_buffer`, so this is a valid,
    // CPU-visible, mutable slice for the duration of this call.
    let slots = unsafe {
        core::slice::from_raw_parts_mut(pointer.as_ptr().cast::<u32>(), gather_count.max(1))
    };
    slots.fill(0);
}

thread_local! {
    /// Uniform blobs, keyed by their own bytes. A plan's uniforms are a
    /// function of the BOUND OP — extents, strides, bases — so they are
    /// byte-identical on every call, and `execute` was allocating a fresh
    /// `MTLBuffer` for each of them per op per call. Safe to share: the
    /// kernel binds them `constant` and never writes through them, and two
    /// ops with identical uniform bytes want identical contents by
    /// definition.
    ///
    /// Bounded to `crate::sized::UNIFORM_CACHE_ENTRIES` with least-recently-
    /// used eviction (the `u64` tick alongside each buffer) rather than left
    /// to grow forever: a workload whose uniform bytes vary per call
    /// (different shapes, different `cached_len` without bucketing) would
    /// otherwise retain one `MTLBuffer` per distinct blob ever seen.
    pub(super) static UNIFORM_BUFFERS: RefCell<BTreeMap<Vec<u8>, (MetalBuffer, u64)>> =
        RefCell::new(BTreeMap::new());

    /// Monotonic use counter driving LRU eviction -- incremented on every
    /// hit and every insert, so the entry with the smallest stored tick is
    /// always the one least recently touched.
    static UNIFORM_CACHE_CLOCK: RefCell<u64> = const { RefCell::new(0) };
}

/// Counts uniform buffers served from cache rather than allocated.
pub static UNIFORM_BUFFER_REUSES: Counter = Counter::new("omega.metal.uniforms.reuse");

/// CARD 6.5's census counter: every genuinely fresh device buffer
/// `allocate_buffer` hands out, on ANY path (the classic per-op-per-call
/// path below, or `build_buffer_arena`'s own size-class-miss path). Not
/// gated behind `metal-plan-stable-buffers` -- this counter's whole point is
/// to read the SAME number on both arms of the bake-off: `op_count` every
/// step with the feature off, `op_count` once (at the plan-cache miss that
/// builds the arena) and 0 on every following plan-cache-hit step with it on.
pub static OUTPUT_BUFFER_ALLOCATIONS: Counter =
    Counter::new("omega.metal.output_buffer.allocations");
pub static OUTPUT_BUFFER_ALLOCATED_BYTES: Counter =
    Counter::new("omega.metal.output_buffer.allocated_bytes");

/// Every initialization write into a plan-owned uniform buffer. These bytes
/// are immutable for the plan's life; a warm stable execution therefore
/// reports zero writes.
pub static PLAN_UNIFORM_WRITES: Counter = Counter::new("omega.metal.plan_uniforms.write");

/// [`DispatchType::Concurrent`]'s own census: every
/// `memoryBarrierWithScope(Buffers)` the private `HazardTracker` actually
/// emitted this step. Unconditional at declaration -- same convention as
/// [`OUTPUT_BUFFER_ALLOCATIONS`] above -- so it reads a stable 0 on
/// [`DispatchType::Serial`] rather than not existing at all.
pub static BARRIERS_EMITTED: Counter = Counter::new("omega.metal.concurrent.barriers_emitted");

/// Every `hazard_step` call -- one per bound-op position under
/// [`DispatchType::Concurrent`], including a horizontal-merge member with
/// `z > 0` that never dispatches: a miss there would silently drop that
/// member's own RAW/WAW/WAR bookkeeping (`handle_merged_position`'s own
/// doc), so this is the direct witness that every member's output was
/// recorded regardless of how many actual dispatches
/// `crate::metal::ENCODE_DISPATCH_CALLS` shows for the same step count
/// (`feature = "instrument"`-gated, unlike this counter, so not doc-linkable
/// under a default build).
pub static HAZARD_STEP_CALLS: Counter = Counter::new("omega.metal.concurrent.hazard_step_calls");

/// How many `split_by_shared_buffers` candidates were refused because the
/// WEIGHT operand had no `device_buffers` entry yet -- rare, weight is
/// normally a checkpoint leaf resolved up front. ROW 571 found this was
/// never the reason production's own candidates were refused (activation and
/// output were); ROW 572 removed those two checks (activation now admits by
/// NODE identity, output is allocated at encode time), leaving this the only
/// remaining refusal this function itself can report.
#[cfg(feature = "metal-horizontal-merge")]
#[cfg(feature = "instrument")]
pub static MERGE_CANDIDATE_UNRESOLVED_WEIGHT: Counter =
    Counter::new("omega.metal.horizontal_merge.candidate_unresolved_weight");

/// [`BARRIERS_EMITTED`]'s own hazard-cause breakdown -- ROW 539's question:
/// of the barriers a concurrent-dispatch step pays, how many are a genuine
/// dataflow edge (RAW) versus an output identity collision (WAW/WAR), and of
/// those, how many are a [`BufferArena`] slot handed to a new, unrelated node
/// while a prior user of that same physical buffer has not yet been
/// barriered off (a false dependency slot reuse manufactures) versus a
/// persistent/output-placed identity that is genuinely written more than
/// once. `instrument`-only: production only ever needed the boolean
/// [`HazardTracker::needs_barrier`] already gives it.
#[cfg(feature = "instrument")]
pub static BARRIERS_RAW: Counter = Counter::new("omega.metal.concurrent.barriers_raw");
#[cfg(feature = "instrument")]
pub static BARRIERS_WAW: Counter = Counter::new("omega.metal.concurrent.barriers_waw");
#[cfg(feature = "instrument")]
pub static BARRIERS_WAR: Counter = Counter::new("omega.metal.concurrent.barriers_war");
#[cfg(feature = "instrument")]
pub static BARRIERS_WAW_WAR_ARENA_RECYCLED: Counter =
    Counter::new("omega.metal.concurrent.barriers_waw_war_arena_recycled");
#[cfg(feature = "instrument")]
pub static BARRIERS_WAW_WAR_PERSISTENT: Counter =
    Counter::new("omega.metal.concurrent.barriers_waw_war_persistent");

/// Entries `UNIFORM_BUFFERS` holds right now -- the direct witness for D6
/// (round-4 synth S2): a caller that wants to know whether the cache grows
/// across a decode run reads this once per step rather than inferring
/// growth from `nocopy_cache_len`'s unrelated bound. Growing after the
/// plan-cache warms (roughly `op_count` per distinct token position, since
/// every `Uniforms` blob carries `reduction_total`, itself a function of
/// `cached_len`) up to `crate::sized::UNIFORM_CACHE_ENTRIES` is the
/// pre-registered prediction this counter exists to check; it never exceeds
/// that capacity.
#[must_use]
pub fn uniform_cache_len() -> usize {
    UNIFORM_BUFFERS.with(|cache| cache.borrow().len())
}

/// Evicts the entry with the smallest use-tick, making room for one more
/// insert. The map is bounded to `crate::sized::UNIFORM_CACHE_ENTRIES`
/// entries by construction, so a linear scan over its current contents to
/// find the minimum tick is cheap -- no ordered secondary index is needed
/// for a map this small.
pub(super) fn evict_least_recently_used(cache: &mut BTreeMap<Vec<u8>, (MetalBuffer, u64)>) {
    let Some(oldest_key) = cache
        .iter()
        .min_by_key(|(_, (_, tick))| *tick)
        .map(|(key, _)| key.clone())
    else {
        return;
    };
    cache.remove(&oldest_key);
}

pub(super) fn upload_uniforms(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let next_tick = UNIFORM_CACHE_CLOCK.with(|clock| {
        let mut clock = clock.borrow_mut();
        *clock += 1;
        *clock
    });

    if let Some(existing) = UNIFORM_BUFFERS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let hit = cache.get(bytes).map(|(buffer, _)| buffer.clone());
        if let Some(buffer) = &hit {
            cache.insert(bytes.to_vec(), (buffer.clone(), next_tick));
        }
        hit
    }) {
        counter!(UNIFORM_BUFFER_REUSES, 1);
        return Ok(existing);
    }
    // SAFETY: `bytes` is always non-empty (every `Uniforms` struct has at
    // least two `long` fields), so its first byte's address is valid and
    // stays valid while this call copies from it.
    let pointer = unsafe { NonNull::new_unchecked(bytes.as_ptr() as *mut c_void) };
    let buffer = unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| MetalError::CompileFailed {
        log: "device refused to allocate the uniforms buffer".to_string(),
    })?;
    UNIFORM_BUFFERS.with(|cache| {
        let mut cache = cache.borrow_mut();
        let capacity = crate::sized::UNIFORM_CACHE_ENTRIES as usize;
        if cache.len() >= capacity && !cache.contains_key(bytes) {
            evict_least_recently_used(&mut cache);
        }
        cache.insert(bytes.to_vec(), (buffer.clone(), next_tick));
    });
    Ok(buffer)
}

/// Test-only reset -- the default std test harness reuses threads across
/// tests in the same binary, and `UNIFORM_BUFFERS`/`UNIFORM_CACHE_CLOCK` are
/// thread-local, so a prior test's entries would otherwise leak into the
/// next one's capacity accounting.
#[cfg(test)]
pub(super) fn reset_uniform_cache_for_test() {
    UNIFORM_BUFFERS.with(|cache| cache.borrow_mut().clear());
    UNIFORM_CACHE_CLOCK.with(|clock| *clock.borrow_mut() = 0);
}

pub(super) fn buffer_for(
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    node: NodeId,
) -> Result<DeviceBuffer, MetalError> {
    if std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some()
        && !device_buffers.contains_key(&node)
    {
        eprintln!(
            "metal missing operand node={node:?} available={:?}",
            device_buffers.keys().collect::<Vec<_>>()
        );
    }
    device_buffers.get(&node).cloned().ok_or_else(|| {
        TensorError::NotLowerable {
            node,
            reason: "operand buffer missing at execution time",
        }
        .into()
    })
}

/// `output` is `(buffer, byte_offset)` rather than two separate parameters
/// so this function stays under clippy's argument-count lint without a
/// `#[allow]` — the pair is always passed and used together, never
/// independently. `byte_offset` is always `0` for a fresh, op-sized buffer
/// (the shape every call site used before `metal-output-placement`
/// existed); it is non-zero only when `buffer` is a caller-owned
/// [`PlacedBuffer`] the op is writing into at an offset (see
/// [`execute_plan_with_placements`]). An `Input`/`Indices` binding's own
/// offset travels with it already, in `device_buffers`' own
/// [`DeviceBuffer`] pair -- the same tuple `checkpoint_mapping_offset`
/// (an input-only placement predating this feature) already relied on, so
/// an input-placed node needs no separate offset map: its offset is
/// whatever [`execute_plan_with_placements`] inserted into `device_buffers`
/// for it. Uniforms and the fault buffer are always read from their own
/// start — neither is ever placed.
// three expert-routing params joined the fixed output/scratch/uniforms/fault
// set above -- each is load-bearing and independently optional, so grouping
// them into a struct would just move the argument count into a constructor.
#[allow(clippy::too_many_arguments)]
pub(super) fn bind_buffers(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    bindings: &[Binding],
    device_buffers: &BTreeMap<NodeId, DeviceBuffer>,
    output: (&Retained<ProtocolObject<dyn MTLBuffer>>, usize),
    // `Some` only for a `CachedAttention` position under `ContextSplitMerge`
    // -- redesign §4c: the split kernel's `Binding::Scratch` slot binds
    // THIS buffer (never `output`, unlike `Binding::Output`), and the merge
    // kernel's own `Binding::Scratch` slot reads it back. `None` for every
    // other op, and for any call site that never resolved
    // `crate::metal::attention_scratch_buffer` -- `Binding::Scratch`
    // appearing with `scratch == None` is exactly the "caller lacks the
    // merge dispatch this binding shape requires" gap `encode_op`'s own doc
    // names, and is rejected there before this function is ever reached.
    scratch: Option<(&Retained<ProtocolObject<dyn MTLBuffer>>, usize)>,
    uniforms: &Retained<ProtocolObject<dyn MTLBuffer>>,
    fault: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
    expert_source_node: Option<NodeId>,
    expert_payloads: Option<(&MetalBuffer, usize)>,
    expert_descriptors: Option<&MetalBuffer>,
) -> Result<(), MetalError> {
    let (output_buffer, output_offset) = output;
    for (index, binding) in bindings.iter().enumerate() {
        let (buffer, offset) = match binding {
            Binding::Input(node) if expert_source_node.is_some_and(|source| source == *node) => {
                expert_payloads
                    .map(|(buffer, offset)| (buffer.clone(), offset))
                    .ok_or_else(|| MetalError::CompileFailed {
                        log: "kernel replaces an expert input but no payload buffer was supplied"
                            .to_string(),
                    })?
            }
            Binding::Input(node) | Binding::Indices(node) => {
                if std::env::var_os("PROXIMA_DEBUG_SEGMENT_HOST").is_some()
                    && !device_buffers.contains_key(node)
                {
                    eprintln!("metal binding missing node={node:?} bindings={bindings:?}");
                }
                buffer_for(device_buffers, *node)?
            }
            Binding::ExpertPayloads(_) => expert_payloads
                .map(|(buffer, offset)| (buffer.clone(), offset))
                .ok_or_else(|| MetalError::CompileFailed {
                    log: "kernel binds expert payloads but none were supplied".to_string(),
                })?,
            Binding::ExpertDescriptors(_) => (
                expert_descriptors
                    .cloned()
                    .ok_or_else(|| MetalError::CompileFailed {
                        log: "kernel binds expert descriptors but none were supplied".to_string(),
                    })?,
                0,
            ),
            Binding::Output(_) => (output_buffer.clone(), output_offset),
            Binding::Scratch => {
                let (buffer, offset) = scratch.ok_or_else(|| MetalError::CompileFailed {
                    log: "kernel binds a scratch buffer but none was resolved".to_string(),
                })?;
                (buffer.clone(), offset)
            }
            Binding::Uniforms => (uniforms.clone(), 0),
            Binding::Fault => (
                fault.cloned().ok_or_else(|| MetalError::CompileFailed {
                    log: "kernel binds a fault buffer but none was allocated".to_string(),
                })?,
                0,
            ),
        };
        // SAFETY: `buffer`'s length was sized from the same op this kernel
        // was emitted from (or, for a placed input/output, the caller
        // guaranteed `offset + <this binding's own element count> *
        // dtype.size_bytes()` fits inside it — see
        // `execute_plan_with_placements`'s own doc), so every byte the
        // kernel indexes through this binding is in bounds, starting at
        // `offset` -- 0 for every binding except a tensor
        // `checkpoint_mapping_offset` or `metal-output-placement` address.
        unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), offset, index) };
    }
    Ok(())
}

/// `grid.threadgroup_width`, when present, is not an occupancy hint — it is
/// a correctness requirement a cooperative-reduce kernel's own coordinate
/// math depends on (`gid / SIMD_WIDTH` as an output index; see
/// `crate::msl::push_cooperative_reduce_body`'s doc), so it is honored
/// exactly rather than folded into the generic `min(threads, max)` pick.
pub(super) fn dispatch(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    grid: GridSpec,
) {
    if grid.threads == 0 {
        return;
    }
    let max_threadgroup = pipeline.maxTotalThreadsPerThreadgroup();
    let threadgroup_width = match grid.threadgroup_width {
        Some(width) => (width as usize).min(max_threadgroup).max(1),
        None => (grid.threads as usize).min(max_threadgroup).max(1),
    };
    let grid_size = MTLSize {
        width: grid.threads as usize,
        height: 1,
        depth: grid.depth as usize,
    };
    let threadgroup = MTLSize {
        width: threadgroup_width,
        height: 1,
        depth: 1,
    };
    #[cfg(feature = "instrument")]
    if let Some(requested_width) = grid.threadgroup_width {
        let entry_name = pipeline
            .label()
            .map(|label| label.to_string().replace(char::is_whitespace, "_"))
            .unwrap_or_else(|| "unknown".to_string());
        debug!(
            requested_width,
            compiled_width = threadgroup_width as u64,
            limit = max_threadgroup as u64,
            "cooperative-reduce threadgroup width"
        );
        eprintln!(
            "cooperative_reduce_width requested_width={requested_width} compiled_width={threadgroup_width} limit={max_threadgroup} entry_name={entry_name}"
        );
    }
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup);
    #[cfg(feature = "instrument")]
    counter!(PHYSICAL_DISPATCH_CALLS, 1);
}

