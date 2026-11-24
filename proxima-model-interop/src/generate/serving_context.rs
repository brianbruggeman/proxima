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
#[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
use super::resident_slot::ResidentSlots;

pub(super) struct ServingContext {
    plans: ResidentStore,
    metal: MetalContext,
    /// Conversations whose device KV a request left in place; limits come from each request's `ServingConfig`.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    slots: ResidentSlots,
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
//
// `slots` extends the same closed graph. A `ResidentSlot` owns the `PlacedBuffer`s of its
// `DeviceKv` (`Retained<MTLBuffer>` planes and f32 staging), which no `Rc`, `Weak` or other
// handle shares once the decode loop released the plans' clones (`release_placed_buffers`).
// Under point (3) a slot enters `slots` only through `DeviceKv::into_slot`, which the decode
// loop calls after its last `evaluate`: every step of the request executed its one command
// buffer and `waitUntilCompleted` returned before the next step began
// (`omega/src/metal/pipeline_buffers_upload.rs:1365`), and the loop's final step read its
// logits back from the device, so no command buffer that touched a slot's planes is
// in flight when it leaves the loop. From there it moves only inside the context.
// `flush` and the drop of a slot run while the context is active on the calling thread.
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
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    slots: &'context mut ResidentSlots,
}

impl ServingGuard<'_> {
    /// the context's slots, reachable while it is active: a slot is flushed and dropped against its own queue.
    #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
    pub(super) fn slots(&mut self) -> &mut ResidentSlots {
        self.slots
    }
}

impl Drop for ServingGuard<'_> {
    fn drop(&mut self) {
        self.plans.swap_active();
    }
}

impl Drop for ServingContext {
    fn drop(&mut self) {
        self.clear_plans();
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        self.activate().slots().clear();
    }
}

impl ServingContext {
    pub(super) fn new() -> Self {
        Self {
            plans: ResidentStore::default(),
            metal: MetalContext::new(None, None),
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            slots: ResidentSlots::new(0, 0),
        }
    }

    /// activates the metal context and this context's plans on the calling thread; both
    /// go back when the guard drops.
    pub(super) fn activate(&mut self) -> ServingGuard<'_> {
        #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
        let Self { plans, metal, slots } = self;
        #[cfg(not(all(feature = "metal-output-placement", target_os = "macos")))]
        let Self { plans, metal } = self;
        let metal = metal.activate();
        plans.swap_active();
        ServingGuard {
            plans,
            _metal: metal,
            #[cfg(all(feature = "metal-output-placement", target_os = "macos"))]
            slots,
        }
    }

    /// frees every resident plan, while the context is active so their buffers
    /// drop against their own queue and mapping.
    pub(super) fn clear_plans(&mut self) {
        let _active = self.activate();
        resident_plans::clear_active();
    }

    /// frees what a warm-up at `chunk_rows` left that no request reuses, then fits the row
    /// template of each plan kept, while the context is active so the buffers drop against their
    /// own queue.
    pub(super) fn trim_warm_residue(&mut self, chunk_rows: usize) -> Result<usize, crate::error::InteropError> {
        let _active = self.activate();
        resident_plans::trim_active(chunk_rows);
        resident_plans::rejoin_active_transients();
        resident_plans::prime_active_row_templates()
    }

    #[cfg(test)]
    pub(super) fn footprint(&self) -> resident_plans::ResidentFootprint {
        self.plans.footprint()
    }

    #[cfg(test)]
    pub(super) fn wide_arenas(&self) -> Vec<(usize, usize, Vec<usize>, Option<usize>)> {
        self.plans.wide_arenas()
    }

    /// decode and wide plans held while this context is inactive.
    #[cfg(test)]
    pub(super) fn plan_count(&self) -> usize {
        self.plans.plan_count()
    }

    /// `(greedy builds, direct kv rows builds)` the resident programs cost so far.
    #[cfg(all(test, feature = "metal-output-placement", target_os = "macos"))]
    pub(super) fn program_builds(&self) -> (usize, usize) {
        let programs = self.plans.programs();
        (programs.greedy_builds(), programs.direct_kv_builds())
    }

    #[cfg(all(test, feature = "metal-output-placement", target_os = "macos"))]
    pub(super) fn program_footprint(&self) -> crate::generate::program_cache::ProgramFootprint {
        self.plans.programs().footprint()
    }

    #[cfg(test)]
    pub(super) fn wide_template_verdicts(&self) -> Vec<Option<bool>> {
        self.plans.wide_template_verdicts()
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

    use omega::metal::{block_copied_bytes_total, mapping_offset_uploads_total, page_size};
    use proxima_gguf::parse_complete;
    use proxima_numeric::NumericFormat;
    use proxima_tensor::AlignedBuffer;
    use proxima_tensor::op::NodeId;

    use crate::generate::chunked_prefill_tests::dense_checkpoint;
    use crate::generate::device_kv::with_direct_kv_rows;
    use crate::generate::program_cache::{GreedyKey, ProgramCache};
    use crate::generate::prompt_cache::CacheEntry;
    use crate::generate::residency_caches::KvFormats;
    use crate::generate::{
        CachePath, LoadedModel, PrefixRetention, PrefixState, ResidentAsk, ResidentRefusal, ResidentSlot, SlotTake,
        ring_slack_rows,
    };
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
                    let offsets_before = mapping_offset_uploads_total();
                    let copied_before = block_copied_bytes_total();
                    model
                        .generate_from_ids(&PROMPT, 3, &config, &mut |_| ControlFlow::Continue(()))
                        .expect("thread B decodes from the model loaded on thread A");
                    (
                        mapping_offset_uploads_total() - offsets_before,
                        block_copied_bytes_total() - copied_before,
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
                    let offsets_before = mapping_offset_uploads_total();
                    let copied_before = block_copied_bytes_total();
                    let output = work();
                    (
                        output,
                        mapping_offset_uploads_total() - offsets_before,
                        block_copied_bytes_total() - copied_before,
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

    const OTHER_PROMPT: [u32; 9] = [9, 41, 3, 77, 25, 6, 88, 14, 2];

    /// the decode loop as the request path runs it: the model's context active, the state the
    /// loop returns and the slot it left in the runtime.
    fn loop_under(
        model: &LoadedModel<'_>,
        config: &ServingConfig<'_>,
        retention: PrefixRetention,
    ) -> (PrefixState, Option<ResidentSlot>) {
        let mut runtime = model.backend_runtime(config);
        runtime.prefix_retention = retention;
        let mut serving = proxima_primitives::driver::block_on(model.serving.lock());
        let _active = serving.activate();
        runtime.resume_active();
        let (_, _, _, state) = model
            .run_decode_loop_from_ids(
                PROMPT.to_vec(),
                3,
                config,
                &mut runtime,
                None,
                &mut crate::generate::LogitsSink::Discard,
                &mut crate::generate::NodeValuesSink::Discard,
                &mut |_| ControlFlow::Continue(()),
                None,
                true,
                None,
                None,
            )
            .expect("the loop decodes on the dense fixture");
        runtime.park_active();
        (state, runtime.leased_slot.take())
    }

    /// a lease leaves the device rows where they are: no host rows come back, and the slot holds
    /// exactly the ids a `Keep` run of the same request flushes.
    #[test]
    fn a_leased_loop_flushes_no_rows_and_hands_out_a_slot_holding_the_request_ids() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        let config = cached_gpu_config(0);

        let (kept, no_slot) = loop_under(&model, &config, PrefixRetention::Keep);
        let (leased, slot) = loop_under(&model, &config, PrefixRetention::Lease);

        assert!(no_slot.is_none(), "a Keep run left a slot");
        assert!(kept.cached_len > PROMPT.len() && !kept.layer_caches.is_empty(), "the Keep run flushed no host rows: {}", kept.cached_len);
        assert_eq!(leased.cached_len, 0, "the leased loop flushed rows to the host");
        assert!(leased.layer_caches.is_empty(), "the leased loop returned host layer caches");
        let slot = slot.expect("the leased loop handed out no slot");
        assert_eq!(slot.ids, kept.ids, "the slot's ids are the request's final ids");
        assert!(slot.checkpoints.is_empty(), "the loop holds no checkpoints to pass");
    }

    /// through the cache with room for one slot: the first lease stays resident and writes no host
    /// entry; the second makes the table give the first back, flushed into the host cache.
    #[test]
    fn a_second_lease_in_a_table_of_one_flushes_and_stores_the_first_slot() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        let config = cached_gpu_config(0);
        assert_eq!(config.parallel_sequences, 1, "the table holds one slot");
        let mut runtime = model.backend_runtime(&config);
        runtime.prefix_retention = PrefixRetention::Lease;
        let mut request = |prompt: &[u32]| {
            model
                .run_decode_loop_through_cache(
                    prompt.to_vec(),
                    &[],
                    3,
                    &config,
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
                .expect("the request decodes on the dense fixture")
        };

        request(&PROMPT);
        let after_first = (
            proxima_primitives::driver::block_on(model.serving.lock()).slots.len(),
            model.prompt_cache.lock().held().len(),
        );
        request(&OTHER_PROMPT);
        let after_second = (
            proxima_primitives::driver::block_on(model.serving.lock()).slots.len(),
            model.prompt_cache.lock().held().len(),
        );
        let stored_ids: Vec<Vec<u32>> = model.prompt_cache.lock().entry_states().map(|state| state.ids.clone()).collect();

        assert_eq!(after_first, (1, 0), "after one lease: (slots resident, host entries)");
        assert_eq!(after_second, (1, 1), "after two leases the first was given up and stored");
        assert_eq!(&stored_ids[0][..PROMPT.len()], &PROMPT, "the stored entry is the first request's rows");
    }

    const GENERATED_TOKENS: usize = 6;

    fn greedy(config: ServingConfig<'static>) -> ServingConfig<'static> {
        ServingConfig { temperature: 0.0, ..config }
    }

    /// `length` token ids spread over the fixture's vocabulary, none of them 0.
    fn prompt_of(length: usize) -> Vec<u32> {
        (0..length).map(|index| (index as u32 * 37 + 11) % 250 + 1).collect()
    }

    /// `prompt` with every id from `from` on replaced by a different one.
    fn changed_from(prompt: &[u32], from: usize) -> Vec<u32> {
        prompt.iter().enumerate().map(|(index, id)| if index < from { *id } else { id % 250 + 1 }).collect()
    }

    fn with_dense_model<Output>(work: impl FnOnce(&LoadedModel<'_>) -> Output) -> Output {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        work(&model)
    }

    /// one request as ragd serves it: a new thread, the retention a discarding caller gets for this config.
    fn serve(model: &LoadedModel<'_>, config: &ServingConfig<'_>, prompt: &[u32]) -> Vec<u32> {
        serve_under(model, config, prompt, PrefixRetention::for_discarding_caller(config))
    }

    fn serve_under(model: &LoadedModel<'_>, config: &ServingConfig<'_>, prompt: &[u32], retention: PrefixRetention) -> Vec<u32> {
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let mut runtime = model.backend_runtime(config);
                    runtime.prefix_retention = retention;
                    let (generated, _, _, _) = model
                        .run_decode_loop_through_cache(
                            prompt.to_vec(),
                            &[],
                            GENERATED_TOKENS,
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
                    generated
                })
                .join()
                .expect("the request thread did not panic")
        })
    }

    /// (host entries, resident slots) the cache holds right now.
    fn held(model: &LoadedModel<'_>) -> (usize, usize) {
        (
            model.prompt_cache.lock().held().len(),
            proxima_primitives::driver::block_on(model.serving.lock()).slots.len(),
        )
    }

    struct ResidentCase {
        name: &'static str,
        first_prompt: usize,
        changed_from: usize,
        second_ubatch: u32,
        expected_path: CachePath,
        expected_keep: usize,
    }

    /// request 2 against the live conversation of request 1 for each case; returns the cases run.
    /// Request 2 answers what a cache-off run of the same prompt answers.
    fn run_resident_cases(cases: &[ResidentCase]) -> usize {
        let mut ran = 0;
        for case in cases {
            with_dense_model(|model| {
                let config = greedy(cached_gpu_config(0));
                let second_config = ServingConfig { ubatch_size: case.second_ubatch, ..config };
                let off = greedy(ServingConfig { ubatch_size: case.second_ubatch, ..gpu_config() });
                let first = prompt_of(case.first_prompt);
                let second = if case.changed_from == first.len() { first.clone() } else { changed_from(&first, case.changed_from) };

                serve(model, &config, &first);
                assert_eq!(held(model), (0, 1), "{}: request 1 leaves one slot and no host entry", case.name);
                let answered = serve(model, &second_config, &second);
                let report = model.last_prompt_cache_report().expect("request 2 reported its path");
                let reference = serve(model, &off, &second);

                assert_eq!(report.path, case.expected_path, "{}: {report:?}", case.name);
                assert_eq!(report.lcp, case.expected_keep, "{}: tokens kept", case.name);
                assert_eq!(report.prefilled_tokens, second.len() - case.expected_keep, "{}: tokens prefilled", case.name);
                assert_eq!(held(model), (0, 1), "{}: (host entries, slots) after request 2", case.name);
                assert_eq!(answered, reference, "{}: ids against the cache-off run", case.name);
            });
            ran += 1;
        }
        ran
    }

    /// a cache with `checkpoint_interval` 16, so a 40-id prompt plans stops at 16 and 32 wherever stops are planned.
    fn stopping_config() -> ServingConfig<'static> {
        let mut config = greedy(cached_gpu_config(0));
        config.prompt_cache.checkpoint_interval = 16;
        config
    }

    /// the lease miss prefills each token once, as cache-off does, and leaves one slot with the request's ids.
    #[test]
    fn a_miss_under_lease_prefills_the_prompt_once_and_leaves_one_slot_with_its_ids() {
        let mut ran = 0;
        with_dense_model(|model| {
            let config = stopping_config();
            let off = greedy(gpu_config());
            let prompt = prompt_of(40);
            assert_eq!(PrefixRetention::for_discarding_caller(&config), PrefixRetention::Lease);

            let answered = serve(model, &config, &prompt);
            let report = model.last_prompt_cache_report().expect("request 1 reported its path");

            assert_eq!(report.path, CachePath::Miss, "{report:?}");
            assert_eq!(report.prefilled_tokens, prompt.len(), "every token prefilled exactly once: {report:?}");
            assert_eq!(held(model), (0, 1), "(host entries, slots) after the miss");
            let key = {
                let runtime = model.backend_runtime(&config);
                model.cache_key(&config, &runtime, None)
            };
            let mut serving = proxima_primitives::driver::block_on(model.serving.lock());
            let SlotTake::Reused { slot, .. } = serving.slots.take(&prompt, &key, config.prompt_cache.min_similarity_milli) else {
                panic!("the miss left no slot sharing the prompt");
            };
            assert!(slot.ids.starts_with(&prompt) && slot.ids.len() >= prompt.len(), "the slot holds the request's ids: {:?}", slot.ids);
            assert!(slot.checkpoints.is_empty(), "a lease plans no stops, so captures no checkpoints");
            drop(serving);
            assert_eq!(answered, serve(model, &off, &prompt), "ids against the cache-off run");
            ran += 1;
        });
        assert_eq!(ran, 1, "cases run");
    }

    /// the repeat of a lease miss resumes the slot: one token prefilled.
    #[test]
    fn a_repeat_after_a_lease_miss_resumes_the_slot_and_prefills_one_token() {
        let mut ran = 0;
        with_dense_model(|model| {
            let config = stopping_config();
            let off = greedy(gpu_config());
            let prompt = prompt_of(40);

            serve(model, &config, &prompt);
            let answered = serve(model, &config, &prompt);
            let report = model.last_prompt_cache_report().expect("request 2 reported its path");

            assert_eq!(report.path, CachePath::Resident, "{report:?}");
            assert_eq!(report.prefilled_tokens, 1, "{report:?}");
            assert_eq!(answered, serve(model, &off, &prompt), "ids against the cache-off run");
            ran += 1;
        });
        assert_eq!(ran, 1, "cases run");
    }

    /// the stop loop under either caller retention runs its stretches on host state: the entry it returns holds the rows up
    /// to the last stop, no slot is leased, and the caller's retention is back as it was. The dense fixture has no ring
    /// layers, so `RingCheckpoint::capture` has nothing to snapshot on it and no checkpoint is counted here.
    #[test]
    fn the_stop_loop_returns_a_real_state_and_restores_the_callers_retention() {
        let mut ran = 0;
        for caller in [PrefixRetention::Lease, PrefixRetention::Keep] {
            with_dense_model(|model| {
                let config = stopping_config();
                let prompt = prompt_of(40);
                let stops = [16_usize, 32];
                let mut runtime = model.backend_runtime(&config);
                runtime.prefix_retention = caller;
                let key = model.cache_key(&config, &runtime, None);
                let (_, widths) = model.declared_layer_cache_names_and_widths().expect("the fixture declares its layers");
                let mut serving = proxima_primitives::driver::block_on(model.serving.lock());
                let _active = serving.activate();
                runtime.resume_active();

                let entry = model
                    .prefill_through_stops(
                        &prompt,
                        CacheEntry::empty(key),
                        &stops,
                        &stops,
                        &widths,
                        &config.prompt_cache,
                        &config,
                        &mut runtime,
                        None,
                        &mut |_position| ControlFlow::Continue(()),
                    )
                    .expect("the stops prefill on the dense fixture");
                runtime.park_active();

                assert_eq!(entry.state.cached_len, 32, "{caller:?}: the entry holds the rows up to the last stop");
                assert_eq!(&entry.state.ids[..32], &prompt[..32], "{caller:?}: the entry's ids");
                assert!(!entry.state.layer_caches.is_empty(), "{caller:?}: the state came back hollow");
                assert!(runtime.leased_slot.is_none(), "{caller:?}: a stop leased its rows to a slot");
                assert_eq!(runtime.prefix_retention, caller, "{caller:?}: the caller's retention was restored");
                ran += 1;
            });
        }
        assert_eq!(ran, 2, "cases run");
    }

    /// llama's identical repeat: 1 token re-evaluated, the slot rewound in place.
    #[test]
    fn a_repeated_prompt_resumes_the_live_slot_and_prefills_one_token() {
        let cases = [ResidentCase {
            name: "repeat",
            first_prompt: 40,
            changed_from: 40,
            second_ubatch: 512,
            expected_path: CachePath::Resident,
            expected_keep: 39,
        }];

        assert_eq!(run_resident_cases(&cases), 1, "cases run");
    }

    /// llama's B1: the last 12 of 40 ids changed, so 28 are kept and 12 prefilled, in place.
    #[test]
    fn a_prompt_with_its_last_12_ids_changed_resumes_the_live_slot_and_prefills_12() {
        let cases = [ResidentCase {
            name: "last_12_ids_changed",
            first_prompt: 40,
            changed_from: 28,
            second_ubatch: 512,
            expected_path: CachePath::Resident,
            expected_keep: 28,
        }];

        assert_eq!(run_resident_cases(&cases), 1, "cases run");
    }

    /// 60 tokens to prefill against a device step of 49 (the draft rows): the slot is flushed to a
    /// host entry and the host path serves the request, as it did before slots.
    #[test]
    fn a_tail_wider_than_the_device_step_flushes_the_slot_and_takes_the_host_path() {
        let cases = [ResidentCase {
            name: "tail_wider_than_the_device_step",
            first_prompt: 80,
            changed_from: 20,
            second_ubatch: 4,
            expected_path: CachePath::Rewind,
            expected_keep: 20,
        }];

        assert_eq!(run_resident_cases(&cases), 1, "cases run");
    }

    /// No `ServingConfig` field flips `device_kv_eligible` on the dense fixture while keeping the
    /// `CacheKey`: the runtime's engine (`gpu_layers`) and `ring_write_offset` are key fields, and
    /// `moe_pre_gather` only counts with experts, which the fixture has none of. So the refusal is
    /// asked directly of the slot request 1 left, with a CPU runtime standing in for the ineligible loop.
    #[test]
    fn a_resident_start_the_loop_cannot_adopt_is_refused_and_flushed_to_the_host_path() {
        with_dense_model(|model| {
            let config = greedy(cached_gpu_config(0));
            let off = greedy(gpu_config());
            let prompt = prompt_of(40);
            serve(model, &config, &prompt);
            assert_eq!(held(model), (0, 1), "request 1 leaves one slot and no host entry");

            let refusal = std::thread::scope(|scope| {
                scope
                    .spawn(|| {
                        let gpu_runtime = model.backend_runtime(&config);
                        let cpu_runtime = model.backend_runtime(&ServingConfig { gpu_layers: 0, ..config });
                        let key = model.cache_key(&config, &gpu_runtime, None);
                        let mut serving = proxima_primitives::driver::block_on(model.serving.lock());
                        let SlotTake::Reused { slot, keep } =
                            serving.slots.take(&prompt, &key, config.prompt_cache.min_similarity_milli)
                        else {
                            panic!("request 1's slot shares the prompt with request 2");
                        };
                        let _active = serving.activate();
                        let (names, widths) = model.declared_layer_cache_names_and_widths().expect("the fixture declares its layers");
                        let seed = model
                            .fresh_layer_caches(&KvFormats::of(&config), &names, &widths, 0, ring_slack_rows(&config, None))
                            .expect("hollow caches build");
                        let admit = |runtime| {
                            let ask = ResidentAsk { keep, prompt_tokens: prompt.len(), max_tokens: GENERATED_TOKENS, draft_limit: 0 };
                            model.admit_resident(&slot.device, ask, &config, runtime, &seed)
                        };
                        assert_eq!(admit(&gpu_runtime), Ok(()), "control: the same slot is admitted by an eligible loop");
                        let refusal = admit(&cpu_runtime).expect_err("an ineligible loop refuses the slot");
                        assert!(model.refuse_resident(slot, refusal, &config, None).is_none(), "the host path runs");
                        refusal
                    })
                    .join()
                    .expect("the refusal thread did not panic")
            });

            assert_eq!(refusal, ResidentRefusal::DeviceKvIneligible);
            assert_eq!(held(model), (1, 0), "the slot was flushed and stored as one host entry");
            let answered = serve(model, &config, &prompt);
            let report = model.last_prompt_cache_report().expect("request 2 reported its path");
            assert_ne!(report.path, CachePath::Resident, "{report:?}");
            assert_eq!(answered, serve(model, &off, &prompt), "ids against the cache-off run");
        });
    }

    /// the retention a caller that drops the state gets: a lease with the prompt cache on, a discard with it off.
    #[test]
    fn a_discarding_caller_leases_with_the_cache_on_and_discards_with_it_off() {
        let on = cached_gpu_config(0);
        let off = gpu_config();

        assert_eq!(PrefixRetention::for_discarding_caller(&on), PrefixRetention::Lease);
        assert_eq!(PrefixRetention::for_discarding_caller(&off), PrefixRetention::Discard);
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

    /// one greedy request on `model` through the cache path, a fresh runtime as the request path builds.
    fn greedy_ids(model: &LoadedModel<'_>, config: &ServingConfig<'_>, prompt: &[u32]) -> Vec<u32> {
        let mut runtime = model.backend_runtime(config);
        let (ids, ..) = model
            .run_decode_loop_through_cache(
                prompt.to_vec(),
                &[],
                4,
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
        ids
    }

    /// `(greedy builds, direct kv rows builds)` the model's resident programs have cost.
    fn program_builds(model: &LoadedModel<'_>) -> (usize, usize) {
        proxima_primitives::driver::block_on(model.serving.lock()).program_builds()
    }

    fn fresh_ids(config: &ServingConfig<'_>, prompt: &[u32]) -> Vec<u32> {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        greedy_ids(&model, config, prompt)
    }

    /// the second request of a model reuses the greedy program the first built, and decodes what a fresh model does.
    #[test]
    fn a_second_greedy_request_reuses_the_cached_program_and_decodes_the_same_ids() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        let config = gpu_config();
        let mut cases_run = 0;

        let first = greedy_ids(&model, &config, &PROMPT);
        let after_first = program_builds(&model);
        let second = greedy_ids(&model, &config, &PROMPT);
        let after_second = program_builds(&model);
        let fresh = fresh_ids(&config, &PROMPT);

        assert_eq!(after_first.0, 1, "the first request builds the greedy program once: {after_first:?}");
        cases_run += 1;
        assert_eq!(after_second, after_first, "the second request rebuilt a program: {after_first:?} then {after_second:?}");
        cases_run += 1;
        assert_eq!(second, fresh, "the reusing request decoded other ids than a fresh model");
        cases_run += 1;
        assert_eq!(first, fresh, "the building request decoded other ids than a fresh model");
        cases_run += 1;
        assert_eq!(cases_run, 4);
    }

    /// three requests of different lengths leave the cache at one greedy program and the direct rows of the one decode program.
    #[test]
    fn the_program_cache_holds_the_same_programs_after_one_two_and_three_requests() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        let config = gpu_config();

        let mut held = Vec::new();
        for prompt in [&PROMPT[..], &PROMPT[..5], &PROMPT[..7]] {
            greedy_ids(&model, &config, prompt);
            held.push(proxima_primitives::driver::block_on(model.serving.lock()).program_footprint());
        }

        let first = held[0];
        assert_eq!(first.greedy_entries, 1, "the fixture's decode and prefill cut share one greedy program: {first:?}");
        assert_eq!(first.greedy_ops, 256, "{first:?}");
        assert_eq!(first.greedy_logits_bytes, 257 * size_of::<f32>(), "one logits row of the fixture vocabulary: {first:?}");
        assert_eq!(first.direct_kv_entries, 1, "{first:?}");
        assert_eq!(first.direct_kv_ops, 235, "{first:?}");
        assert_eq!(held, [first; 3], "the cache grew with the request count: {held:?}");
    }

    /// a request at another cached length binds other symbol values to the same program, so it reuses the held greedy program
    /// and decodes as a fresh model: the built program never read the values, only `shape::infer` checked them.
    #[test]
    fn a_request_at_another_cached_length_reuses_the_greedy_program_and_decodes_the_same_ids_as_a_fresh_model() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        let config = gpu_config();
        let mut cases_run = 0;

        greedy_ids(&model, &config, &PROMPT);
        let base = program_builds(&model).0;

        let shorter = greedy_ids(&model, &config, &PROMPT[..5]);
        let after_shorter = program_builds(&model).0;
        assert_eq!(base, 1, "the first request builds the greedy program once");
        cases_run += 1;
        assert_eq!(after_shorter, base, "another prompt length (other symbol values) rebuilt the greedy program");
        cases_run += 1;
        assert_eq!(shorter, fresh_ids(&config, &PROMPT[..5]), "the program reused at another length decoded other ids");
        cases_run += 1;

        let again = greedy_ids(&model, &config, &PROMPT);
        assert_eq!(program_builds(&model).0, base, "returning to the first length rebuilt the greedy program");
        cases_run += 1;
        assert_eq!(again, fresh_ids(&config, &PROMPT), "the program reused at the first length decoded other ids");
        cases_run += 1;
        assert_eq!(cases_run, 5);
    }

    /// the fixture vocabulary has no end-of-generation id, so `ignore_eos` cannot reach the loop here: every other input the
    /// constructors read is checked at the key, where one differing field must make the keys unequal.
    #[test]
    fn a_greedy_key_differs_in_each_input_the_constructor_reads() {
        let base = GreedyKey::new(7, NodeId(3), 64, &[], false);
        let others = [
            ("program hash", GreedyKey::new(8, NodeId(3), 64, &[], false)),
            ("logits root", GreedyKey::new(7, NodeId(4), 64, &[], false)),
            ("vocab", GreedyKey::new(7, NodeId(3), 128, &[], false)),
            ("suppressed ids", GreedyKey::new(7, NodeId(3), 64, &[2], false)),
            ("prefill cut", GreedyKey::new(7, NodeId(3), 64, &[], true)),
        ];
        let mut cases_run = 0;
        for (input, other) in others {
            assert_ne!(base, other, "a key differing in the {input} compared equal");
            cases_run += 1;
        }
        assert_eq!(base, GreedyKey::new(7, NodeId(3), 64, &[], false));
        assert_eq!(cases_run, 5);
    }

    /// the direct kv rows program is reused for the same source, and rebuilt for other roots or other cache formats.
    #[test]
    fn the_direct_kv_rows_program_is_reused_for_one_key_and_rebuilt_for_another() {
        let fixture = dense_checkpoint();
        let storage = page_aligned(&fixture);
        // SAFETY: `storage` outlives every borrow below and holds `fixture.len()` initialized bytes.
        let checkpoint = unsafe { core::slice::from_raw_parts(storage.as_ptr().cast::<u8>(), fixture.len()) };
        let parsed = parse_complete(checkpoint).expect("the dense fixture parses");
        let model = LoadedModel::load(&parsed, checkpoint).expect("the dense fixture loads");
        let half = (NumericFormat::Float16, NumericFormat::Float16);
        let wide = (NumericFormat::Float32, NumericFormat::Float32);
        let mut other_roots = model.layer_roots.clone();
        other_roots.pop();
        let mut cache = ProgramCache::default();
        let mut cases_run = 0;
        let mut request = |formats, roots: &[crate::generate::LayerCacheRoots]| {
            let mut rows = cache.take_direct_kv(formats);
            let hash = rows
                .rows_for(&model.program, model.program_hash, roots)
                .expect("the rows build")
                .map(|(hash, _)| hash);
            cache.keep_direct_kv(rows);
            (hash, cache.direct_kv_builds())
        };

        let (built_hash, builds) = request(half, &model.layer_roots);
        assert!(built_hash.is_some() && builds == 1, "a binary16 cache takes direct rows, built once: {built_hash:?} {builds}");
        cases_run += 1;
        let (reused_hash, builds) = request(half, &model.layer_roots);
        assert!(reused_hash == built_hash && builds == 1, "the same key must reuse: {reused_hash:?} {builds}");
        cases_run += 1;
        let (_, builds) = request(half, &other_roots);
        assert_eq!(builds, 2, "other source roots must rebuild");
        cases_run += 1;
        let (none_hash, builds) = request(wide, &model.layer_roots);
        assert!(none_hash.is_none() && builds == 3, "other formats must rebuild, and f32 takes no direct rows: {none_hash:?} {builds}");
        cases_run += 1;
        let (back_hash, builds) = request(half, &model.layer_roots);
        let expected = with_direct_kv_rows(&model.program, &model.layer_roots, (&half.0, &half.1))
            .expect("the rows build")
            .map(|rows| crate::generate::program_hash(&rows.program));
        assert!(back_hash == expected && builds == 4, "the rebuilt rows are those a fresh build gives: {back_hash:?} {expected:?} {builds}");
        cases_run += 1;
        assert_eq!(cases_run, 5);
    }
}
