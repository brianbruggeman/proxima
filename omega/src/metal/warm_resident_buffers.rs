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
use core::mem;
use objc2::rc::Retained;
use objc2::sel;
use objc2_foundation::NSObjectProtocol;
use objc2_metal::{MTLCommandQueue, MTLResidencySet, MTLResidencySetDescriptor, MTLResourceUsage};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

thread_local! {
    /// The residency set [`warm_resident_buffers`] requested, kept alive so
    /// the kernel does not reclaim the wired weight pages while the model
    /// serves. Thread-local while no context is active; an active [`MetalContext`] swaps its own
    /// set in, so the set follows the model that built it.
    pub(super) static WARM_RESIDENCY_SET: RefCell<Option<Retained<ProtocolObject<dyn MTLResidencySet>>>> =
        const { RefCell::new(None) };

    /// One residency set per registered mapping buffer, keyed by its
    /// no-copy cache name. Lives and dies with that cache entry, and an
    /// active [`MetalContext`] swaps it with the cache, so the set follows
    /// the mapping it covers.
    pub(super) static MAPPING_RESIDENCY: RefCell<BTreeMap<String, Retained<ProtocolObject<dyn MTLResidencySet>>>> =
        const { RefCell::new(BTreeMap::new()) };
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
    let Some(set) = attached_residency_set(device, queue, buffers)? else {
        return Ok(false);
    };
    WARM_RESIDENCY_SET.with(|slot| *slot.borrow_mut() = Some(set));
    Ok(true)
}

fn attached_residency_set(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    buffers: &[MetalBuffer],
) -> Result<Option<Retained<ProtocolObject<dyn MTLResidencySet>>>, MetalError> {
    if !device.respondsToSelector(sel!(newResidencySetWithDescriptor:error:)) {
        return Ok(None);
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
    Ok(Some(set))
}

/// Gives a mapping buffer its own residency set, keyed by the mapping's
/// cache name, so a GPU read through the registered mapping never meets a
/// non-resident page. Called where the GPU first receives the mapping, and
/// apart from the warm set so `warm` never reads it as a finished warm. A
/// no-op on an OS without residency sets.
pub(super) fn request_mapping_residency(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    buffer: &MetalBuffer,
) -> Result<(), MetalError> {
    let (_, queue) = device_and_queue()?;
    if let Some(set) = attached_residency_set(device, &queue, core::slice::from_ref(buffer))? {
        MAPPING_RESIDENCY.with(|held| held.borrow_mut().insert(name.to_string(), set));
    }
    Ok(())
}

/// Detaches and drops the residency set held for mapping `name`, before the
/// mapping is replaced or unregistered, so no set outlives its mapping.
pub(super) fn release_mapping_residency(name: &str) {
    let released = MAPPING_RESIDENCY.with(|held| held.borrow_mut().remove(name));
    detach_residency_sets(released.into_iter());
}

/// Detaches and drops every mapping residency set the active context holds.
pub(super) fn release_all_mapping_residency() {
    let released = MAPPING_RESIDENCY.with(|held| mem::take(&mut *held.borrow_mut()));
    detach_residency_sets(released.into_values());
}

/// Detaches and drops the warm residency set the active context holds; it
/// covers every cached buffer, mapping buffers included.
pub(super) fn release_warm_residency() {
    let released = WARM_RESIDENCY_SET.with(|slot| slot.borrow_mut().take());
    detach_residency_sets(released.into_iter());
}

fn detach_residency_sets(
    sets: impl Iterator<Item = Retained<ProtocolObject<dyn MTLResidencySet>>>,
) {
    if let Ok((_, queue)) = device_and_queue() {
        sets.for_each(|set| queue.removeResidencySet(&set));
    }
}

/// Allocation counts of the mapping residency sets held now, by name. Test-only.
#[cfg(test)]
pub(super) fn held_mapping_residency_allocations() -> Vec<(String, usize)> {
    MAPPING_RESIDENCY.with(|held| {
        held.borrow()
            .iter()
            .map(|(name, set)| (name.clone(), set.allocationCount()))
            .collect()
    })
}

/// Allocations in the residency set the last warm requested, `None` when no
/// set is held. Test-only: callers never need to read the set back.
#[cfg(test)]
pub(super) fn held_residency_allocations() -> Option<usize> {
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
    warm_body()
}

/// Warm for the active [`MetalContext`]: idempotent while its residency set is
/// held, and the set attaches to the context's queue, created here if absent.
pub(super) fn warm_active_context() -> Result<usize, MetalError> {
    let held =
        WARM_RESIDENCY_SET.with(|slot| slot.borrow().as_ref().map(|set| set.allocationCount()));
    if let Some(count) = held {
        return Ok(count);
    }
    if nothing_to_warm() {
        return Ok(0);
    }
    let (device, queue) = device_and_queue()?;
    ACTIVE_QUEUE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot =
                Some(
                    device
                        .newCommandQueue()
                        .ok_or_else(|| MetalError::DriverRefusedObject {
                            object: "command queue",
                            reason: "device returned none".to_string(),
                        })?,
                );
        }
        Ok::<(), MetalError>(())
    })?;
    drop(queue);
    warm_body()
}

fn warm_body() -> Result<usize, MetalError> {
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
            assert_eq!(
                held_residency_allocations(),
                None,
                "no residency sets on this OS"
            );
        }
    }
}

#[cfg(test)]
mod context_tests {
    use super::*;
    use crate::metal::MetalContext;
    use std::thread;

    const WEIGHT_NAMES: [&str; 2] = ["blk.0.attn_q.weight", "blk.0.ffn_down.weight"];

    fn queue_identity(queue: &ProtocolObject<dyn MTLCommandQueue>) -> *const () {
        core::ptr::from_ref(queue).cast()
    }

    /// Two real resident weights uploaded into a fresh context, then warmed on
    /// this thread. `None` when this host has no Metal device.
    fn warmed_context() -> Option<MetalContext> {
        let (device, _queue) = device_and_queue().ok()?;
        let mut context = MetalContext::default();
        {
            let _active = context.activate();
            let page = page_size();
            for (index, name) in WEIGHT_NAMES.iter().enumerate() {
                let weight: Vec<u8> = (0..page * (index + 1))
                    .map(|byte| (byte % 251) as u8)
                    .collect();
                upload_resident_copy(&device, name, weight.as_ptr().cast(), weight.len())
                    .expect("a resident weight uploads");
            }
        }
        assert_eq!(
            context.warm().expect("warming two real buffers succeeds"),
            2
        );
        assert_eq!(
            context.warm().expect("a second warm returns the held set"),
            2
        );
        Some(context)
    }

    /// The context is built and warmed on a spawned thread (its own empty
    /// thread-locals); `MetalContext` is not `Send` before S3, so the
    /// assertions run on that same thread.
    fn on_fresh_thread(assertions: fn(&mut MetalContext)) {
        thread::scope(|scope| {
            scope
                .spawn(|| {
                    if let Some(mut context) = warmed_context() {
                        assertions(&mut context);
                    }
                })
                .join()
                .expect("the context thread finishes without panicking");
        });
    }

    #[test]
    fn active_context_serves_its_own_queue_and_holds_its_two_allocation_residency_set() {
        on_fresh_thread(|context| {
            let own_queue = queue_identity(context.queue().expect("warm gave the context a queue"));
            let _active = context.activate();
            let (_device, queue) = device_and_queue().expect("device while active");
            assert_eq!(
                queue_identity(&queue),
                own_queue,
                "the active queue is the context's"
            );
            if _device.respondsToSelector(sel!(newResidencySetWithDescriptor:error:)) {
                assert_eq!(held_residency_allocations(), Some(2));
            }
        });
    }

    #[test]
    fn inactive_context_neither_owns_the_thread_queue_nor_the_thread_residency_set() {
        on_fresh_thread(|context| {
            let own_queue = queue_identity(context.queue().expect("warm gave the context a queue"));
            let (_device, queue) = device_and_queue().expect("device while inactive");
            assert_ne!(
                queue_identity(&queue),
                own_queue,
                "the thread queue is not the context's"
            );
            assert_eq!(
                held_residency_allocations(),
                None,
                "the set sits in the context, not the thread"
            );
        });
    }
}
