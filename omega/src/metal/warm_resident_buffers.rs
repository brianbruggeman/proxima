//! Moves the first-reference cost of the model's device buffers from the
//! first decode or prefill command buffer to model load.
//!
//! The first command buffer that references the checkpoint mapping and the
//! resident weight copies pays a driver-side cost between `commit` and
//! `scheduled` that later command buffers do not. [`warm_resident_buffers`]
//! pays it once, at load: it creates the registered mapping buffers through
//! the same [`mapping_buffer`] call the first upload would make, then
//! commits one command buffer whose compute encoder declares every buffer in
//! the no-copy and resident-copy caches with `useResource` and waits for it.
//! No bytes are allocated that the first step would not allocate; the
//! caches are thread-local, so call it on the thread that will serve.
//!
//! Composes: [`mapping_buffer`] (the mapping wrappers),
//! [`resident_buffers`] (the cache walk), [`new_labeled_command_buffer`] and
//! [`commit_and_wait`] (the one command buffer). Compose them directly when
//! a caller needs only part of the set.

use super::*;
use objc2::rc::Retained;
use objc2::sel;
use objc2_foundation::NSObjectProtocol;
use objc2_metal::{
    MTLCommandQueue, MTLResidencySet, MTLResidencySetDescriptor, MTLResourceUsage,
};
use std::cell::RefCell;
use std::time::Instant;

thread_local! {
    /// The residency set [`warm_resident_buffers`] requested, kept alive so
    /// the kernel does not reclaim the wired weight pages while the model
    /// serves. Thread-local like the buffer caches it was built from.
    static WARM_RESIDENCY_SET: RefCell<Option<Retained<ProtocolObject<dyn MTLResidencySet>>>> =
        const { RefCell::new(None) };
}

/// Wraps `MTLResidencySet` (macOS 15+): adds every buffer, `commit`s, attaches
/// the set to `queue` (`addResidencySet`), and `requestResidency`s it, which
/// asks the driver to wire the buffers' pages now instead of at the first
/// command buffer that references them. The same sequence ggml-metal's
/// `ggml_metal_buffer_rset_init` runs per model buffer. Returns `false` when
/// the OS has no residency sets.
fn request_residency(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    buffers: &[MetalBuffer],
) -> Result<bool, MetalError> {
    if !device.respondsToSelector(sel!(newResidencySetWithDescriptor:error:)) {
        return Ok(false);
    }
    let descriptor = MTLResidencySetDescriptor::new();
    // SAFETY: initialCapacity is a sizing hint the driver may exceed.
    unsafe { descriptor.setInitialCapacity(buffers.len()) };
    let set = device
        .newResidencySetWithDescriptor_error(&descriptor)
        .map_err(|error| MetalError::DriverRefusedObject {
            object: "residency set",
            reason: error.localizedDescription().to_string(),
        })?;
    for buffer in buffers {
        set.addAllocation(ProtocolObject::from_ref(&**buffer));
    }
    set.commit();
    queue.addResidencySet(&set);
    set.requestResidency();
    WARM_RESIDENCY_SET.with(|slot| *slot.borrow_mut() = Some(set));
    Ok(true)
}

/// Allocations in the residency set the last warm requested, `None` when no
/// set is held. Test-only: callers never need to read the set back.
#[cfg(test)]
fn held_residency_allocations() -> Option<usize> {
    WARM_RESIDENCY_SET.with(|slot| slot.borrow().as_ref().map(|set| set.allocationCount()))
}

/// Every buffer the no-copy and resident-copy caches hold on this thread.
pub(super) fn resident_buffers() -> Vec<MetalBuffer> {
    let nocopy = NOCOPY_BUFFERS.with(|cache| {
        cache
            .borrow()
            .values()
            .map(|(_, _, buffer)| buffer.clone())
            .collect::<Vec<_>>()
    });
    let copies = RESIDENT_BUFFERS.with(|cache| {
        cache
            .borrow()
            .values()
            .map(|(_, _, buffer)| buffer.clone())
            .collect::<Vec<_>>()
    });
    nocopy.into_iter().chain(copies).collect()
}

fn materialize_mapping_buffers(device: &ProtocolObject<dyn MTLDevice>) -> Result<(), MetalError> {
    if let Some((base, length)) = CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow()) {
        mapping_buffer(device, CHECKPOINT_MAPPING_NOCOPY_NAME, base, length)?;
    }
    if let Some((base, length)) = expert_mapping_identity() {
        mapping_buffer(device, EXPERT_MAPPING_NOCOPY_NAME, base, length)?;
    }
    Ok(())
}

fn nothing_to_warm() -> bool {
    CHECKPOINT_MAPPING.with(|mapping| mapping.borrow().is_none())
        && expert_mapping_identity().is_none()
        && resident_buffers().is_empty()
}

/// Declares every resident model buffer to the driver in one waited command
/// buffer and returns how many it declared. Returns `Ok(0)` without touching
/// the device when nothing is registered or resident.
pub fn warm_resident_buffers() -> Result<usize, MetalError> {
    if nothing_to_warm() {
        return Ok(0);
    }
    let (device, queue) = device_and_queue()?;
    materialize_mapping_buffers(&device)?;
    let buffers = resident_buffers();
    let residency_started = Instant::now();
    let residency_requested = request_residency(&device, &queue, &buffers)?;
    let residency_ms = residency_started.elapsed().as_secs_f64() * 1000.0;
    let command_buffer = new_labeled_command_buffer(&queue, "omega.warm_resident_buffers")
        .ok_or_else(|| MetalError::DriverRefusedObject {
            object: "command buffer",
            reason: "command queue returned none".to_string(),
        })?;
    let encoder = EncoderGuard::new(command_buffer.computeCommandEncoder().ok_or_else(|| {
        MetalError::DriverRefusedObject {
            object: "compute command encoder",
            reason: "command buffer returned none".to_string(),
        }
    })?);
    for buffer in &buffers {
        encoder.useResource_usage(ProtocolObject::from_ref(&**buffer), MTLResourceUsage::Read);
    }
    encoder.finish();
    commit_and_wait(
        &command_buffer,
        BufferDiagnostics {
            chunk_index: 0,
            chunk_count: 1,
            dispatch_count: 0,
            first_op_label: "warm_resident_buffers",
            last_op_label: "warm_resident_buffers",
            encoder_status_requested: encoder_error_status_requested(),
        },
    )?;
    debug!(
        buffers = buffers.len() as u64,
        residency_requested = residency_requested,
        residency_ms = residency_ms,
        "warmed resident buffers"
    );
    Ok(buffers.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_set_is_empty_before_any_upload() {
        reset_nocopy_cache_for_test();
        reset_resident_cache_for_test();
        assert!(resident_buffers().is_empty());
    }

    #[test]
    fn warm_up_with_nothing_registered_declares_zero_buffers() {
        reset_nocopy_cache_for_test();
        reset_resident_cache_for_test();
        reset_checkpoint_mapping_for_test();
        assert!(
            matches!(warm_resident_buffers(), Ok(0)),
            "an empty set needs no device"
        );
    }

    #[test]
    fn warming_two_resident_weights_holds_exactly_two_allocations_in_the_residency_set() {
        let Ok((device, _queue)) = device_and_queue() else {
            return;
        };
        reset_nocopy_cache_for_test();
        reset_resident_cache_for_test();
        reset_checkpoint_mapping_for_test();
        WARM_RESIDENCY_SET.with(|slot| *slot.borrow_mut() = None);
        let page = page_size();
        let first_weight: Vec<u8> = (0..page).map(|index| (index % 251) as u8).collect();
        let second_weight: Vec<u8> = (0..page * 2).map(|index| (index % 241) as u8).collect();
        for (name, weight) in [
            ("blk.0.attn_q.weight", &first_weight),
            ("blk.0.ffn_down.weight", &second_weight),
        ] {
            upload_resident_copy(&device, name, weight.as_ptr().cast(), weight.len())
                .expect("a resident weight uploads");
        }

        let warmed = warm_resident_buffers().expect("warming two real buffers succeeds");

        assert_eq!(warmed, 2, "both resident weights are declared");
        if device.respondsToSelector(sel!(newResidencySetWithDescriptor:error:)) {
            assert_eq!(
                held_residency_allocations(),
                Some(2),
                "the held residency set carries one allocation per warmed buffer"
            );
        } else {
            assert_eq!(held_residency_allocations(), None, "no residency sets on this OS");
        }
    }
}
