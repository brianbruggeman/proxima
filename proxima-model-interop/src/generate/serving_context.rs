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

use omega::metal::{MetalContext, MetalContextGuard};

use super::resident_plans::{self, ResidentStore};

pub(super) struct ServingContext {
    plans: ResidentStore,
    metal: MetalContext,
}

// SAFETY: the aggregate is one closed graph. `metal` owns the maps of `Retained<MTLBuffer>`
// (resident, no-copy, expert-source, uniform and fault buffers), the two transient pools
// (a `Weak<MetalBuffer>` each), the queue and the residency set. `plans` owns every plan and
// every `Rc<MetalBuffer>` lease: each lease is taken from one of `metal`'s two pools
// (`ARENA_TRANSIENT`, `ATTENTION_SCRATCH_TRANSIENT`, swapped in by `activate`), so each `Rc`
// and `Weak` reachable from `plans` points at a buffer whose other handles are in `plans`
// or in `metal`'s pools, and nowhere else: no `Rc`, `Weak` or buffer handle is cloned to a
// thread-local that outlives an activation, since the guard swaps every slot back on drop
// (same thread) and `BackendRuntime` returns its plans before the guard drops. objc2-metal
// marks buffers and residency sets neither `Send` nor `Sync`, and `Rc` counts are non-atomic,
// so this impl is what lets the graph cross threads. It holds because (1) the graph is
// closed as above, so a move leaves no `Rc` count reachable from the old thread; (2) it moves
// only as a whole, and is reached only through `&mut` taken from the model's `AsyncMutex`,
// so no two threads touch a handle or a count at once; (3) it changes threads only between
// requests, when every command buffer that used these buffers has been waited
// (`shared_transient.rs` module doc), so no GPU work is mid-use; (4) Metal resource objects
// carry no thread affinity (ASSUMED, Apple's Metal documentation, not read here; the
// cross-thread tests are the runtime evidence). Not `Sync`: access is exclusive.
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

/// both contexts active on this thread; dropping it hands each back to its owner.
#[must_use]
pub(super) struct ServingGuard<'context> {
    plans: &'context mut ResidentStore,
    _metal: MetalContextGuard<'context>,
}

impl Drop for ServingGuard<'_> {
    fn drop(&mut self) {
        self.plans.swap_active();
    }
}

impl Drop for ServingContext {
    fn drop(&mut self) {
        self.clear_plans();
    }
}

impl ServingContext {
    pub(super) fn new() -> Self {
        Self {
            plans: ResidentStore::default(),
            metal: MetalContext::new(None, None),
        }
    }

    /// activates the metal context and this context's plans on the calling thread; both
    /// go back when the guard drops.
    pub(super) fn activate(&mut self) -> ServingGuard<'_> {
        let Self { plans, metal } = self;
        let metal = metal.activate();
        plans.swap_active();
        ServingGuard {
            plans,
            _metal: metal,
        }
    }

    /// frees every resident plan, while the context is active so their buffers
    /// drop against their own queue and mapping.
    pub(super) fn clear_plans(&mut self) {
        let _active = self.activate();
        resident_plans::clear_active();
    }

    /// decode and wide plans held while this context is inactive.
    #[cfg(test)]
    pub(super) fn plan_count(&self) -> usize {
        self.plans.plan_count()
    }

    #[cfg(test)]
    pub(super) fn wide_plan_count(&self) -> usize {
        self.plans.wide_count()
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

    #[cfg(test)]
    pub(super) fn metal(&self) -> &MetalContext {
        &self.metal
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

    /// ragd serves each request from a new thread. The plans belong to the model's context, so the
    /// second thread resumes what the first built instead of planning again.
    #[test]
    fn a_second_thread_resumes_the_plans_the_first_thread_built() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        let config = gpu_config();

        let (first, second) = std::thread::scope(|scope| {
            let request = || scope.spawn(|| decode_counts(&model, &config)).join().expect("the request thread did not panic");
            let first = request();
            let second = request();
            (first, second)
        });

        assert!(first.misses > 0, "thread A built no plan: {} misses", first.misses);
        assert_eq!(second.misses, 0, "thread B built {} plans instead of resuming", second.misses);
        assert!(second.hits > 0, "thread B hit no resident plan: {} hits", second.hits);
        assert!(
            proxima_primitives::driver::block_on(model.serving.lock()).plan_count() > 0,
            "the context holds no plan after the requests"
        );
    }

    /// the plans, arena lease and attention scratch a request leaves in the context are the
    /// only owners of their Metal buffers, so dropping the model must release every one.
    #[test]
    fn dropping_the_model_releases_the_metal_buffers_its_requests_left_resident() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        let config = gpu_config();
        decode_counts(&model, &config);

        let (plans, arena, attention) = {
            let serving = proxima_primitives::driver::block_on(model.serving.lock());
            let (arena, attention) = serving.metal().transient_handles();
            (serving.plan_count(), arena, attention)
        };
        assert!(plans > 0, "the request left no resident plan: {plans}");
        assert!(arena.upgrade().is_some(), "the request left no arena buffer for the model to free");
        assert!(attention.upgrade().is_some(), "the request left no attention scratch for the model to free");

        drop(model);

        assert!(arena.upgrade().is_none(), "the arena buffer outlived the model");
        assert!(attention.upgrade().is_none(), "the attention scratch outlived the model");
    }

    struct Counts {
        hits: usize,
        misses: usize,
    }

    fn decode_counts(model: &LoadedModel<'_>, config: &ServingConfig<'_>) -> Counts {
        let mut runtime = model.backend_runtime(config);
        model
            .run_decode_loop_through_cache(
                PROMPT.to_vec(),
                &[],
                3,
                config,
                &mut runtime,
                None,
                &mut crate::generate::LogitsSink::Discard,
                &mut crate::generate::NodeValuesSink::Discard,
                &mut |_| ControlFlow::Continue(()),
                None,
                false,
                None,
                None,
            )
            .expect("the request decodes on the dense fixture");
        Counts {
            hits: runtime.plan_hits,
            misses: runtime.plan_misses,
        }
    }
}
