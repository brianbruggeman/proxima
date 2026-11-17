//! The state one model's requests share, owned by the model instead of by
//! whichever thread happened to load or serve it.
//!
//! [`ServingContext`] holds an [`omega::metal::MetalContext`] (the checkpoint
//! mapping registration, the no-copy and resident buffer caches, the expert
//! sources, the queue and the residency set). `LoadedModel` keeps it behind a
//! [`proxima_primitives::sync::AsyncMutex`]: a request acquires the gate, then
//! [`MetalContext::activate`] swaps the context into the thread's ambient
//! slots for the request's duration and back out after. Compose the same way
//! from a sync entry point with `proxima_primitives::driver::block_on(gate.lock())`.
//! The gate is the exclusion; this type adds no lock of its own.

use omega::metal::MetalContext;

pub(super) struct ServingContext {
    metal: MetalContext,
}

// SAFETY: `ServingContext` holds one `MetalContext`: maps of `Retained<MTLBuffer>`
// (resident, no-copy and expert-source buffers), a `Vec<u8>` of staged expert
// bytes, a `(usize, usize)` mapping identity, and an optional queue and
// residency set. objc2-metal marks buffers and residency sets neither `Send`
// nor `Sync`, so this impl is what lets the aggregate cross threads. It holds
// because (1) every handle is owned by this value alone: nothing here is an
// `Rc`, a `Weak`, a clone shared with another thread's slot, or a borrow; the
// thread-local slots only ever hold a context's contents between `activate`
// and the guard's drop, which is on the same thread; (2) the value moves only
// as a whole, and is reached only through `&mut` taken from the model's
// `AsyncMutex`, so no two threads touch a handle at once; (3) it changes
// threads only between requests, when every command buffer that used these
// buffers has been waited, so no GPU work or other thread is mid-use; (4)
// Metal resource objects carry no thread affinity (ASSUMED, Apple's Metal
// documentation, not read here; the cross-thread acceptance test is the
// runtime evidence). The `plans` field the design adds in a later slice holds
// `Rc` leases into pools of this context, so that slice must re-argue this
// impl for the larger closed graph. Not `Sync`: access is exclusive.
unsafe impl Send for ServingContext {}

const fn assert_send<Value: Send>() {}

trait AmbiguousIfSync<Marker> {
    fn marker() {}
}
impl<Value: ?Sized> AmbiguousIfSync<()> for Value {}
impl<Value: ?Sized + Sync> AmbiguousIfSync<u8> for Value {}

// compile-time: `ServingContext` is `Send` and not `Sync`. The second item
// resolves to one impl only when `Sync` does not hold, else inference is ambiguous.
const _: () = {
    assert_send::<ServingContext>();
    let _ = <ServingContext as AmbiguousIfSync<_>>::marker;
};

impl ServingContext {
    pub(super) fn new() -> Self {
        Self {
            metal: MetalContext::new(None, None),
        }
    }

    /// registers the checkpoint mapping into this context rather than the
    /// calling thread's ambient state, so any thread that activates it resolves it.
    pub(super) fn register_checkpoint_mapping(&mut self, bytes: &[u8]) {
        let _active = self.metal.activate();
        omega::backend::register_checkpoint_mapping(bytes);
    }

    /// unregisters `bytes` from this context's mapping, as the model's own
    /// sidecar attach does once expert bytes stop living in the whole mapping.
    pub(super) fn unregister_checkpoint_mapping(&mut self, bytes: &[u8]) {
        let _active = self.metal.activate();
        omega::backend::unregister_checkpoint_mapping(bytes);
    }

    pub(super) fn metal_mut(&mut self) -> &mut MetalContext {
        &mut self.metal
    }
}

#[cfg(test)]
mod tests {
    use core::ops::ControlFlow;

    use omega::metal::{BLOCK_COPIED_BYTES, MAPPING_OFFSET_UPLOADS, page_size};
    use proxima_gguf::parse_complete;
    use proxima_tensor::AlignedBuffer;

    use crate::generate::chunked_prefill_tests::dense_checkpoint;
    use crate::generate::LoadedModel;
    use crate::serving::{GPU_LAYERS_ALL, PromptCacheConfig, ServingConfig};

    const PROMPT: [u32; 9] = [4, 17, 33, 8, 91, 12, 5, 60, 7];

    fn gpu_config() -> ServingConfig<'static> {
        ServingConfig {
            gpu_layers: GPU_LAYERS_ALL,
            prompt_cache: PromptCacheConfig::off(),
            ..ServingConfig::default()
        }
    }

    /// the checkpoint bytes at a page-aligned address, as a real mmap would be.
    fn page_aligned(bytes: &[u8]) -> AlignedBuffer {
        let page = page_size();
        let mut buffer = AlignedBuffer::new(bytes.len().div_ceil(size_of::<f32>()), page)
            .expect("page-aligned storage for the fixture checkpoint");
        // SAFETY: the buffer owns at least `bytes.len()` bytes and does not overlap `bytes`.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.as_mut_ptr().cast::<u8>(), bytes.len());
        }
        buffer
    }

    /// ragd loads on one thread and serves each request from a new one. The
    /// checkpoint mapping belongs to the model, so the serving thread binds
    /// weights by offset into it instead of copying them.
    #[test]
    fn a_thread_that_did_not_load_the_model_binds_weights_by_mapping_offset_not_by_copy() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads on thread A");
        let config = gpu_config();

        let (offset_uploads, copied_bytes) = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let offsets_before = MAPPING_OFFSET_UPLOADS.get();
                    let copied_before = BLOCK_COPIED_BYTES.get();
                    model
                        .generate_from_ids(&PROMPT, 3, &config, &mut |_| ControlFlow::Continue(()))
                        .expect("thread B decodes from the model loaded on thread A");
                    (
                        MAPPING_OFFSET_UPLOADS.get() - offsets_before,
                        BLOCK_COPIED_BYTES.get() - copied_before,
                    )
                })
                .join()
                .expect("thread B did not panic")
        });

        assert!(offset_uploads > 0, "thread B bound no weight by mapping offset: {offset_uploads}");
        assert!(
            (copied_bytes as usize) < fixture.len(),
            "thread B copied {copied_bytes} bytes against a {} byte checkpoint: the weights were copied",
            fixture.len()
        );
    }

    fn cached_gpu_config(follow_up_branches: u32) -> ServingConfig<'static> {
        let mut prompt_cache = PromptCacheConfig::standard();
        prompt_cache.follow_up_branches = follow_up_branches;
        prompt_cache.follow_up_max_tokens = 2;
        ServingConfig {
            gpu_layers: GPU_LAYERS_ALL,
            prompt_cache,
            ..ServingConfig::default()
        }
    }

    /// bytes the thread copied and weights it bound by offset while `work` ran on it.
    fn on_new_thread<Output: Send>(work: impl FnOnce() -> Output + Send) -> (Output, u64, u64) {
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let offsets_before = MAPPING_OFFSET_UPLOADS.get();
                    let copied_before = BLOCK_COPIED_BYTES.get();
                    let output = work();
                    (
                        output,
                        MAPPING_OFFSET_UPLOADS.get() - offsets_before,
                        BLOCK_COPIED_BYTES.get() - copied_before,
                    )
                })
                .join()
                .expect("the worker thread did not panic")
        })
    }

    fn assert_bound_by_offset(role: &str, offset_uploads: u64, copied_bytes: u64, checkpoint_len: usize) {
        assert!(offset_uploads > 0, "{role} bound no weight by mapping offset: {offset_uploads}");
        assert!(
            (copied_bytes as usize) < checkpoint_len,
            "{role} copied {copied_bytes} bytes against a {checkpoint_len} byte checkpoint: the weights were copied"
        );
    }

    /// a prewarm on a thread that never loaded the model evaluates inside the
    /// model's context, and a request on yet another thread follows it
    /// without rebinding a resident name or copying the weights.
    #[test]
    fn a_prewarm_then_a_request_on_other_threads_bind_by_offset_and_do_not_collide() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads on thread A");
        let config = cached_gpu_config(0);

        let (report, prewarm_offsets, prewarm_copied) = on_new_thread(|| {
            model.prewarm(&PROMPT, &config).expect("the prewarm runs on thread B")
        });
        let (generated, request_offsets, request_copied) = on_new_thread(|| {
            model
                .generate_from_ids(&PROMPT, 3, &config, &mut |_| ControlFlow::Continue(()))
                .expect("the request runs on thread C after the prewarm")
        });

        assert!(report.prefilled_tokens > 0, "the prewarm prefilled nothing: {report:?}");
        assert!(!generated.0.is_empty(), "the request generated nothing");
        assert_bound_by_offset("the prewarm thread", prewarm_offsets, prewarm_copied, fixture.len());
        // the prewarm already bound every weight into the shared context, so the request finds them resident and uploads none
        assert_eq!(request_offsets, 0, "the request re-bound weights the prewarm had already made resident");
        assert!(
            (request_copied as usize) < fixture.len(),
            "the request thread copied {request_copied} bytes against a {} byte checkpoint",
            fixture.len()
        );
    }

    /// the follow-up branch drafts and their restoring prefill run inside the
    /// model's context on a thread that never loaded it.
    #[test]
    fn follow_up_branches_on_a_thread_that_did_not_load_the_model_bind_by_offset() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads on thread A");
        let config = cached_gpu_config(1);
        model.set_follow_up_closing(&[5, 60]);

        model
            .generate_from_ids(&PROMPT, 3, &config, &mut |_| ControlFlow::Continue(()))
            .expect("the opening request stores the answer entry");
        let (branches, offsets, copied) = on_new_thread(|| {
            model
                .prewarm_follow_ups(&PROMPT, &config)
                .expect("the follow-up prewarm runs on thread B")
        });

        let _ = branches;
        // the opening request left every weight resident in the shared context, so the follow-up uploads none and copies none
        assert_eq!(offsets, 0, "the follow-up re-bound weights the opening request had made resident");
        assert!(
            (copied as usize) < fixture.len(),
            "the follow-up thread copied {copied} bytes against a {} byte checkpoint",
            fixture.len()
        );
    }
}
