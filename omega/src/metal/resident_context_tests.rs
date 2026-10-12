use core::mem::size_of_val;
use std::panic::{AssertUnwindSafe, catch_unwind};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::super::MetalError;
use super::super::arena_encode_dispatch_finish::read_back_as_device_f32;
use super::super::device_buffers_arena_plan::device_and_queue;
use super::super::pipeline_buffers_upload::{
    CHECKPOINT_MAPPING, page_size, register_checkpoint_mapping, reset_checkpoint_mapping_for_test,
};
use super::super::resident_nocopy_cache::{
    release_resident_names, reset_nocopy_cache_for_test, reset_resident_cache_for_test,
    resident_cache_len, upload_block_no_copy, upload_resident_copy,
};
use super::{ResidentContext, ResidentContextGuard};

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

fn activate(context: &mut ResidentContext) -> ResidentContextGuard<'_> {
    context.activate()
}

#[test]
fn nested_contexts_restore_distinct_payloads_and_reuse_each_owner_buffer() {
    let (device, _queue) =
        device_and_queue().expect("Metal device is available for resident context tests");
    reset_resident_cache_for_test();

    let target_values = [1.0_f32, 2.0];
    let draft_values = [3.0_f32, 4.0];
    let mut target = ResidentContext::default();
    let mut draft = ResidentContext::default();
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
    let mut context = ResidentContext::default();
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
    let mut target = ResidentContext::default();
    let mut draft = ResidentContext::default();

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
    let mut target = ResidentContext::default();
    let mut draft = ResidentContext::default();
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
    let mut target = ResidentContext::default();
    let mut draft = ResidentContext::default();
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
    context: &mut ResidentContext,
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
    let mut target = ResidentContext::default();
    let mut draft = ResidentContext::default();

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
