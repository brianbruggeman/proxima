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
use objc2_metal::MTLResourceUsage;

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
    let command_buffer = new_labeled_command_buffer(&queue, "omega.warm_resident_buffers")
        .ok_or_else(|| MetalError::CompileFailed {
            log: "command queue refused to hand out a command buffer".to_string(),
        })?;
    let encoder = EncoderGuard::new(command_buffer.computeCommandEncoder().ok_or_else(|| {
        MetalError::CompileFailed {
            log: "command buffer refused to hand out a compute encoder".to_string(),
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
    debug!(buffers = buffers.len() as u64, "warmed resident buffers");
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
        assert!(matches!(warm_resident_buffers(), Ok(0)), "an empty set needs no device");
    }
}
