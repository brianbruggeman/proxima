use core::mem::size_of_val;
use std::panic::{AssertUnwindSafe, catch_unwind};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSObjectProtocol;
use objc2_metal::MTLDevice;

use super::super::MetalError;
use crate::metal_diagnostics::MetalDiagnostics;
use super::super::arena_encode_dispatch_finish::read_back_as_device_f32;
use super::super::device_buffers_arena_plan::device_and_queue;
use super::super::pipeline_buffers_upload::{
    CHECKPOINT_MAPPING, checkpoint_mapping_offset, page_size, register_checkpoint_mapping,
    reset_checkpoint_mapping_for_test, upload_packed_bytes_in,
};
use super::super::warm_resident_buffers::{
    held_mapping_residency_allocations, held_residency_allocations, warm_active_context_in,
};
use super::super::resident_nocopy_cache::{
    release_resident_names, reset_nocopy_cache_for_test, reset_resident_cache_for_test,
    resident_cache_len, uniform_cache_len, upload_block_no_copy, upload_resident_copy, upload_uniforms,
};
use super::{MetalContext, MetalContextGuard};

const TARGET_WEIGHT: &str = "resident_context_target.weight";
const DRAFT_WEIGHT: &str = "resident_context_draft.weight";

fn checkpoint_identity() -> Option<(usize, usize)> {
    CHECKPOINT_MAPPING.with(|mapping| *mapping.borrow())
}

fn upload_named_f32(
    device: &ProtocolObject<dyn MTLDevice>,
    name: &str,
    values: &[f32],
) -> super::MetalBuffer {
    upload_resident_copy(device, name, values.as_ptr().cast(), size_of_val(values))
        .expect("resident context accepts the named weight upload")
}

fn activate(context: &mut MetalContext) -> MetalContextGuard<'_> {
    context.activate()
}

#[test]
fn nested_contexts_restore_distinct_payloads_and_reuse_each_owner_buffer() {
    let (device, _queue) =
        device_and_queue().expect("Metal device is available for resident context tests");
    reset_resident_cache_for_test();

    let target_values = [1.0_f32, 2.0];
    let draft_values = [3.0_f32, 4.0];
    let mut target = MetalContext::default();
    let mut draft = MetalContext::default();
    let target_buffer;
    let draft_buffer;

    {
        let _target_guard = activate(&mut target);
        target_buffer = upload_named_f32(&device, TARGET_WEIGHT, &target_values);
        assert_eq!(resident_cache_len(), 1);
        assert_eq!(
            read_back_as_device_f32(&target_buffer, 0, target_values.len()),
            target_values
        );

        {
            let _draft_guard = activate(&mut draft);
            draft_buffer = upload_named_f32(&device, TARGET_WEIGHT, &draft_values);
            assert_ne!(
                Retained::as_ptr(&target_buffer),
                Retained::as_ptr(&draft_buffer),
                "different owners keep separate buffers for the same resident name"
            );
            assert_eq!(
                read_back_as_device_f32(&draft_buffer, 0, draft_values.len()),
                draft_values
            );

            let reused = upload_named_f32(&device, TARGET_WEIGHT, &draft_values);
            assert_eq!(
                Retained::as_ptr(&reused),
                Retained::as_ptr(&draft_buffer),
                "returning to draft restores its exact cached buffer"
            );
            assert_eq!(
                read_back_as_device_f32(&reused, 0, draft_values.len()),
                draft_values
            );
        }

        let reused = upload_named_f32(&device, TARGET_WEIGHT, &target_values);
        assert_eq!(Retained::as_ptr(&reused), Retained::as_ptr(&target_buffer));
        assert_eq!(
            read_back_as_device_f32(&reused, 0, target_values.len()),
            target_values,
            "dropping draft restores the target's prior cache"
        );
    }

    {
        let _draft_guard = activate(&mut draft);
        let reused = upload_named_f32(&device, TARGET_WEIGHT, &draft_values);
        assert_eq!(Retained::as_ptr(&reused), Retained::as_ptr(&draft_buffer));
        assert_eq!(
            read_back_as_device_f32(&reused, 0, draft_values.len()),
            draft_values,
            "draft's cache survives target reactivation"
        );
    }
}

#[test]
fn same_owner_rebound_is_rejected_without_replacing_its_payload() {
    let (device, _queue) =
        device_and_queue().expect("Metal device is available for resident context tests");
    reset_resident_cache_for_test();

    let first_values = [1.0_f32, 2.0];
    let rebound_values = [3.0_f32, 4.0];
    let mut context = MetalContext::default();
    let _guard = activate(&mut context);
    let first = upload_named_f32(&device, TARGET_WEIGHT, &first_values);
    let error = upload_resident_copy(
        &device,
        TARGET_WEIGHT,
        rebound_values.as_ptr().cast(),
        size_of_val(&rebound_values),
    )
    .expect_err("a resident name cannot be rebound within one context");

    assert!(matches!(error, MetalError::ResidentNameRebound { .. }));
    assert_eq!(resident_cache_len(), 1);
    assert_eq!(
        read_back_as_device_f32(&first, 0, first_values.len()),
        first_values
    );
}

#[test]
fn nested_contexts_keep_same_named_nocopy_payloads_separate() {
    let (device, _queue) =
        device_and_queue().expect("Metal device is available for resident context tests");
    reset_nocopy_cache_for_test();

    let page = page_size();
    let mut target_values = proxima_tensor::AlignedBuffer::new(2, page)
        .expect("allocate target's page-aligned host weights");
    target_values[0] = 1.0;
    target_values[1] = 2.0;
    let mut draft_values = proxima_tensor::AlignedBuffer::new(2, page)
        .expect("allocate draft's page-aligned host weights");
    draft_values[0] = 3.0;
    draft_values[1] = 4.0;
    let byte_length = target_values.len() * size_of::<f32>();
    let mut target = MetalContext::default();
    let mut draft = MetalContext::default();

    {
        let _target_guard = activate(&mut target);
        let target_buffer = upload_block_no_copy(
            &device,
            TARGET_WEIGHT,
            target_values.as_ptr().cast(),
            byte_length,
        )
        .expect("register target's no-copy weight");
        assert_eq!(read_back_as_device_f32(&target_buffer, 0, 2), [1.0, 2.0]);

        {
            let _draft_guard = activate(&mut draft);
            let draft_buffer = upload_block_no_copy(
                &device,
                TARGET_WEIGHT,
                draft_values.as_ptr().cast(),
                byte_length,
            )
            .expect("register draft's same-named no-copy weight");
            assert_ne!(
                Retained::as_ptr(&target_buffer),
                Retained::as_ptr(&draft_buffer)
            );
            assert_eq!(read_back_as_device_f32(&draft_buffer, 0, 2), [3.0, 4.0]);
        }

        let target_again = upload_block_no_copy(
            &device,
            TARGET_WEIGHT,
            target_values.as_ptr().cast(),
            byte_length,
        )
        .expect("restore target's no-copy entry after draft activation");
        assert_eq!(
            Retained::as_ptr(&target_buffer),
            Retained::as_ptr(&target_again)
        );
        assert_eq!(read_back_as_device_f32(&target_again, 0, 2), [1.0, 2.0]);
    }
}

#[test]
fn releasing_draft_names_does_not_evict_target_buffers() {
    let (device, _queue) =
        device_and_queue().expect("Metal device is available for resident context tests");
    reset_resident_cache_for_test();

    let target_values = [1.0_f32, 2.0];
    let draft_values = [3.0_f32, 4.0];
    let mut target = MetalContext::default();
    let mut draft = MetalContext::default();
    let target_buffer;

    {
        let _target_guard = activate(&mut target);
        target_buffer = upload_named_f32(&device, TARGET_WEIGHT, &target_values);
    }
    {
        let _draft_guard = activate(&mut draft);
        let _draft_buffer = upload_named_f32(&device, DRAFT_WEIGHT, &draft_values);
        release_resident_names([DRAFT_WEIGHT]);
        assert_eq!(resident_cache_len(), 0);
    }
    {
        let _target_guard = activate(&mut target);
        assert_eq!(resident_cache_len(), 1);
        let reused = upload_named_f32(&device, TARGET_WEIGHT, &target_values);
        assert_eq!(Retained::as_ptr(&reused), Retained::as_ptr(&target_buffer));
        assert_eq!(
            read_back_as_device_f32(&reused, 0, target_values.len()),
            target_values
        );
    }
}

#[test]
fn clear_releases_active_aliases_and_restores_the_enclosing_context() {
    let (device, _queue) =
        device_and_queue().expect("Metal device is available for resident context tests");
    reset_resident_cache_for_test();

    let target_values = [1.0_f32, 2.0];
    let draft_values = [3.0_f32, 4.0];
    let enclosing_checkpoint = [1_u8; 32];
    let target_checkpoint = [2_u8; 32];
    let draft_checkpoint = [3_u8; 32];
    reset_checkpoint_mapping_for_test();
    register_checkpoint_mapping(&enclosing_checkpoint);
    let mut target = MetalContext::default();
    let mut draft = MetalContext::default();
    let target_buffer;

    {
        let mut target_guard = activate(&mut target);
        register_checkpoint_mapping(&target_checkpoint);
        target_buffer = upload_named_f32(&device, TARGET_WEIGHT, &target_values);
        {
            let mut draft_guard = activate(&mut draft);
            register_checkpoint_mapping(&draft_checkpoint);
            let _draft_buffer = upload_named_f32(&device, DRAFT_WEIGHT, &draft_values);
            assert_eq!(resident_cache_len(), 1);
            draft_guard.clear();
            assert_eq!(resident_cache_len(), 0);
            assert_eq!(checkpoint_identity(), None);
        }

        assert_eq!(resident_cache_len(), 1);
        assert_eq!(
            checkpoint_identity(),
            Some((target_checkpoint.as_ptr() as usize, target_checkpoint.len())),
            "clearing a nested draft restores the target checkpoint mapping"
        );
        let reused = upload_named_f32(&device, TARGET_WEIGHT, &target_values);
        assert_eq!(
            Retained::as_ptr(&reused),
            Retained::as_ptr(&target_buffer),
            "clearing a nested draft context restores the target's live buffer"
        );
        target_guard.clear();
        assert_eq!(resident_cache_len(), 0);
        assert_eq!(checkpoint_identity(), None);
    }

    {
        let _draft_guard = activate(&mut draft);
        assert_eq!(resident_cache_len(), 0, "the draft context was cleared");
    }
    {
        let _target_guard = activate(&mut target);
        assert_eq!(resident_cache_len(), 0, "the target context was cleared");
        assert_eq!(checkpoint_identity(), None);
    }
    assert_eq!(
        checkpoint_identity(),
        Some((
            enclosing_checkpoint.as_ptr() as usize,
            enclosing_checkpoint.len()
        )),
        "clearing the active target restores its enclosing mapping"
    );
    reset_checkpoint_mapping_for_test();
}

fn return_error_after_checkpoint_registration(
    context: &mut MetalContext,
    checkpoint: &[u8],
) -> Result<(), ()> {
    let _guard = activate(context);
    register_checkpoint_mapping(checkpoint);
    Err(())
}

#[test]
fn checkpoint_mapping_restores_on_return_error_and_unwind() {
    let prior_checkpoint = [1_u8; 32];
    let target_checkpoint = [2_u8; 32];
    let draft_checkpoint = [3_u8; 32];
    reset_checkpoint_mapping_for_test();
    register_checkpoint_mapping(&prior_checkpoint);
    let mut target = MetalContext::default();
    let mut draft = MetalContext::default();

    {
        let _target_guard = activate(&mut target);
        register_checkpoint_mapping(&target_checkpoint);
        assert_eq!(
            checkpoint_identity(),
            Some((target_checkpoint.as_ptr() as usize, target_checkpoint.len()))
        );

        {
            let _draft_guard = activate(&mut draft);
            register_checkpoint_mapping(&draft_checkpoint);
            assert_eq!(
                checkpoint_identity(),
                Some((draft_checkpoint.as_ptr() as usize, draft_checkpoint.len()))
            );
        }
        assert_eq!(
            checkpoint_identity(),
            Some((target_checkpoint.as_ptr() as usize, target_checkpoint.len())),
            "normal nested return restores target mapping"
        );

        let result = return_error_after_checkpoint_registration(&mut draft, &draft_checkpoint);
        assert_eq!(result, Err(()));
        assert_eq!(
            checkpoint_identity(),
            Some((target_checkpoint.as_ptr() as usize, target_checkpoint.len())),
            "error return restores target mapping"
        );

        let unwind = catch_unwind(AssertUnwindSafe(|| {
            let _draft_guard = activate(&mut draft);
            register_checkpoint_mapping(&draft_checkpoint);
            panic!("exercise resident context unwind restoration");
        }));
        assert!(unwind.is_err());
        assert_eq!(
            checkpoint_identity(),
            Some((target_checkpoint.as_ptr() as usize, target_checkpoint.len())),
            "unwind restores target mapping"
        );
    }

    assert_eq!(
        checkpoint_identity(),
        Some((prior_checkpoint.as_ptr() as usize, prior_checkpoint.len())),
        "outer return restores the prior mapping"
    );
    reset_checkpoint_mapping_for_test();
}

#[test]
fn new_context_owns_the_queue_it_is_given_and_default_owns_none() {
    let Ok((_device, queue)) = device_and_queue() else {
        return;
    };
    let queue_address = Retained::as_ptr(&queue).cast::<()>();

    let context = MetalContext::new(Some(queue), None);

    assert_eq!(
        context
            .queue()
            .map(|owned| core::ptr::from_ref(owned).cast::<()>()),
        Some(queue_address)
    );
    assert!(context.residency().is_none());
    assert!(MetalContext::default().queue().is_none());
}

/// a context moves between threads only at quiescence; this test-only carrier
/// stands in for the owner (`ServingContext` in proxima-model-interop) whose
/// audited `Send` impl lets a model hand its context to a serving thread.
struct QuiescentContext(MetalContext);

// SAFETY: the context holds no buffers here and is used by one thread at a time.
unsafe impl Send for QuiescentContext {}

/// ragd serves each request from a fresh thread while the model loaded on
/// another. The mapping is registered into the model's context, so a thread
/// that activates that context resolves a tensor's offset inside it.
#[test]
fn checkpoint_mapping_resolves_on_a_thread_that_did_not_register_it() {
    let Ok((_device, _queue)) = device_and_queue() else {
        return;
    };
    reset_checkpoint_mapping_for_test();
    let page = page_size();
    let mapping = proxima_tensor::AlignedBuffer::new(2 * page / size_of::<f32>(), page)
        .expect("page-aligned fixture standing in for a loaded checkpoint mmap");
    // SAFETY: the buffer owns `2 * page` bytes for the whole test.
    let mapping_bytes = unsafe { core::slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), 2 * page) };
    let tensor = &mapping_bytes[page + 64..page + 64 + 4096];
    let (tensor_address, tensor_length) = (tensor.as_ptr() as usize, tensor.len());

    let mut context = MetalContext::default();
    {
        let _active = context.activate();
        register_checkpoint_mapping(mapping_bytes);
    }
    assert_eq!(checkpoint_identity(), None, "registration stays in the context, off the loading thread");

    let carried = QuiescentContext(context);
    let resolved = std::thread::spawn(move || {
        let mut carried = carried;
        let (device, _queue) = device_and_queue().expect("a second thread opens a device");
        let _active = carried.0.activate();
        checkpoint_mapping_offset(&device, tensor_address as *const _, tensor_length, None)
            .map(|resolved| resolved.expect("the whole mapping binds without copying").1)
    })
    .join()
    .expect("serving thread did not panic");

    assert_eq!(resolved, Some(page + 64));
}

/// the owner's release (what `ServingContext` runs on drop) leaves no residency
/// set over the mapping: the mapping set and the warm set both go.
#[test]
fn clear_detaches_the_mapping_and_warm_residency_sets() {
    let Ok((device, _queue)) = device_and_queue() else {
        return;
    };
    if !device.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
        return;
    }
    reset_checkpoint_mapping_for_test();
    let page = page_size();
    let mapping = proxima_tensor::AlignedBuffer::new(page / size_of::<u32>(), page)
        .expect("page-aligned fixture standing in for a loaded checkpoint mmap");
    // SAFETY: the buffer owns `page` bytes for the whole test.
    let mapping_bytes = unsafe { core::slice::from_raw_parts(mapping.as_ptr().cast::<u8>(), page) };

    let mut context = MetalContext::default();
    let mut active = context.activate();
    register_checkpoint_mapping(mapping_bytes);
    upload_packed_bytes_in(None, &device, &mapping_bytes[8..72], Some("view.weight"))
        .expect("an aligned tensor uploads as a view of the mapping");
    assert_eq!(held_mapping_residency_allocations().len(), 1, "the mapping holds one set");
    warm_active_context_in(None, &MetalDiagnostics::default())
        .expect("warming builds the context's warm set the way a model load does");
    assert!(
        held_residency_allocations().is_some_and(|count| count >= 1),
        "the warm set covers the cached mapping buffer before clear"
    );
    active.clear();

    assert_eq!(held_mapping_residency_allocations(), vec![], "no mapping set outlives clear");
    assert_eq!(held_residency_allocations(), None, "no warm set outlives clear");
}

/// the uniform blob a decode plan would bind: real extents and strides, 16 longs.
fn uniform_blob() -> Vec<u8> {
    (0_i64..16).flat_map(|value| (value * 4096).to_le_bytes()).collect()
}

/// thread A uploads the blob (cache miss) then the serving thread uploads the same bytes;
/// the returned handle's address says whether thread B was served from A's cache.
fn second_thread_buffer_is_first_threads(carry_context: bool) -> (bool, usize) {
    let blob = uniform_blob();
    let mut context = MetalContext::default();
    let first_address = {
        let _active = carry_context.then(|| context.activate());
        let (device, _queue) = device_and_queue().expect("thread A opens a device");
        let buffer = upload_uniforms(&device, &blob).expect("thread A uploads the uniform blob");
        (Retained::as_ptr(&buffer) as usize, uniform_cache_len())
    };
    let carried = QuiescentContext(context);
    let blob_for_b = blob.clone();
    let (second_address, held) = std::thread::spawn(move || {
        let mut carried = carried;
        let _active = carry_context.then(|| carried.0.activate());
        let (device, _queue) = device_and_queue().expect("thread B opens a device");
        let buffer = upload_uniforms(&device, &blob_for_b).expect("thread B uploads the uniform blob");
        (Retained::as_ptr(&buffer) as usize, uniform_cache_len())
    })
    .join()
    .expect("thread B did not panic");
    assert_eq!(first_address.1, 1, "thread A's cache holds the one blob it uploaded");
    (first_address.0 == second_address, held)
}

#[test]
fn uniform_cache_held_by_the_context_serves_a_second_thread_the_same_buffer() {
    let (reused, held) = second_thread_buffer_is_first_threads(true);
    assert!(reused, "thread B allocated its own uniform buffer instead of A's cached one");
    assert_eq!(held, 1, "the context's cache holds exactly the one blob on thread B");
}

#[test]
fn without_a_context_a_second_thread_rebuilds_its_own_uniform_buffer() {
    let (reused, held) = second_thread_buffer_is_first_threads(false);
    assert!(!reused, "control: thread B must not see thread A's thread-local cache");
    assert_eq!(held, 1, "control: thread B's own cache holds the one blob it uploaded");
}
