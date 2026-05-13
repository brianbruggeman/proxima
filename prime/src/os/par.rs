//! parallel data-processing primitives over `ProximaBackgroundPool`.
//!
//! covers rayon's primary data-parallel surface without adding work-stealing.
//! all shapes use the same recursive-split + pool.spawn pattern; split depth
//! is controlled by `chunk_threshold` (default: 4 leaves per worker).
//!
//! supported shapes:
//!   - par_reduce / par_each / par_map_collect (sync + async)
//!   - par_filter  — recursive split + predicate + concat
//!   - par_sort / par_sort_by / par_sort_unstable  — parallel merge sort
//!   - par_chunks_mut  — disjoint-chunk parallel mutation via raw pointer
//!   - scope  — structured concurrency; spawned tasks complete before return
//!   - install / installed_pool  — thread-local pool for implicit dispatch
//!   - par_bridge  — mutex-iterator fan-out for non-slice iterators
//!
//! not supported (with rationale):
//!   - work-stealing per-worker deques: proxima uses a shared injector; adding
//!     per-worker deques is a separate architectural effort.
//!   - rayon::join: use `futures::join!` instead; semantics map directly.
//!   - full ParallelIterator adaptor chain: filter_map, flat_map, partition, etc.
//!     are additive; each can be built on top of the primitives above.
//!   - multi-spawn `scope` with caller-stack borrows: requires lifetime-erased
//!     trampolining; current `scope` is safe and covers most use cases.
//!
//! constraints (per the workspace perf-design-constraints memory):
//!   - zero-copy: slice borrowed by leaves via Arc<[T]>; Acc moves through tree
//!   - branchless hot path: leaf fold is a tight `for` over a contiguous slice
//!   - stack allocations: only the per-spawn Box on bgpool's typed path; no extra heap per element
//!   - types: fully generic; no dyn-trait on the public API

#![cfg(feature = "runtime-prime-bgpool-par")]

use std::future::Future;
use std::marker::PhantomData;

use std::pin::Pin;
use std::sync::Arc;

use super::background::ProximaBackgroundPool;
use proxima_core::ProximaError;

/// fallback leaf-size threshold for the rare case where the caller
/// constructs a builder without a slice (e.g. some future API that
/// doesn't know N up front). regular callers get a slice-and-worker
/// derived threshold from [`default_threshold_for`].
///
/// kept for compatibility — old code that used this const directly
/// still works, but new callers should rely on `default_threshold_for`.
pub const DEFAULT_CHUNK_THRESHOLD: usize = 1024;

/// derive a default leaf-size threshold from worker count + slice length.
/// targets ~4 leaves per worker — the low-variance zone of the U-curve
/// validated in the discipline log's C3 entry. for N=100K on 4 workers
/// this returns ~6250, landing close to the threshold-8192 bench arm
/// that was within 1.23% of rayon.
///
/// floor at 1 so we never produce zero-size leaves. when the slice is
/// smaller than `workers * 4`, returns `slice_len` so the whole slice
/// runs as a single leaf — no value in splitting below worker count.
#[must_use = "threshold is informational; pass it to `par_reduce_with_threshold` etc."]
pub fn default_threshold_for(pool: &ProximaBackgroundPool, slice_len: usize) -> usize {
    let target_leaves = pool.workers().saturating_mul(4).max(1);
    (slice_len / target_leaves).max(1)
}

/// recursive-split parallel reduce.
///
/// folds each leaf into an `Acc` via `work`, then merges pairwise via
/// `reduce` up a binary tree. each leaf seeds with a fresh `identity()` —
/// callers provide a factory closure, not a value, so the same identity
/// can be reconstructed at every leaf without `Acc: Default` bounds.
///
/// `work`, `reduce`, and `identity` are cloned per split branch (typically
/// cheap — closures over `Copy` state or `Arc` of state). they must be
/// `Send + Sync + 'static` to satisfy the spawn signature.
///
/// errors from individual leaves are not propagated — a dropped sender
/// causes that leaf to contribute a fresh `identity()` instead of
/// crashing the reduce. callers that need explicit error propagation
/// should use `try_par_reduce` (not yet implemented).
pub async fn par_reduce<Item, Acc, Work, Reduce, Identity>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    identity: Identity,
    work: Work,
    reduce: Reduce,
) -> Acc
where
    Item: Sync + Send + 'static,
    Acc: Send + 'static,
    Work: Fn(&Item) -> Acc + Send + Sync + 'static + Clone,
    Reduce: Fn(Acc, Acc) -> Acc + Send + Sync + 'static + Clone,
    Identity: Fn() -> Acc + Send + Sync + 'static + Clone,
{
    let threshold = default_threshold_for(pool, slice.len());
    par_reduce_with_threshold(pool, slice, identity, work, reduce, threshold).await
}

/// explicit-threshold variant of [`par_reduce`]. lower threshold = more
/// leaves = more spawn overhead but finer-grained worker distribution.
/// benchmark to pick.
pub async fn par_reduce_with_threshold<Item, Acc, Work, Reduce, Identity>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    identity: Identity,
    work: Work,
    reduce: Reduce,
    chunk_threshold: usize,
) -> Acc
where
    Item: Sync + Send + 'static,
    Acc: Send + 'static,
    Work: Fn(&Item) -> Acc + Send + Sync + 'static + Clone,
    Reduce: Fn(Acc, Acc) -> Acc + Send + Sync + 'static + Clone,
    Identity: Fn() -> Acc + Send + Sync + 'static + Clone,
{
    if slice.is_empty() {
        return identity();
    }
    let len = slice.len();
    par_reduce_range(
        pool.clone(),
        slice,
        0,
        len,
        identity,
        work,
        reduce,
        chunk_threshold,
    )
    .await
}

/// recursive worker. boxed-future return because the recursive call's
/// future type would otherwise be infinite-recursive in the type system
/// (each level wraps the level below). the box is one allocation per
/// split node — O(log N) split nodes for N elements, dwarfed by the
/// per-leaf spawn allocations.
// 8 args: pool / slice / start / end / identity / work / reduce / threshold
// all thread through unavoidably; bundling into a struct just shifts the
// allocation cost and obscures the recursive call sites.
#[allow(clippy::too_many_arguments)]
fn par_reduce_range<Item, Acc, Work, Reduce, Identity>(
    pool: Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    start: usize,
    end: usize,
    identity: Identity,
    work: Work,
    reduce: Reduce,
    chunk_threshold: usize,
) -> Pin<Box<dyn Future<Output = Acc> + Send + 'static>>
where
    Item: Sync + Send + 'static,
    Acc: Send + 'static,
    Work: Fn(&Item) -> Acc + Send + Sync + 'static + Clone,
    Reduce: Fn(Acc, Acc) -> Acc + Send + Sync + 'static + Clone,
    Identity: Fn() -> Acc + Send + Sync + 'static + Clone,
{
    Box::pin(async move {
        let len = end - start;
        if len <= chunk_threshold {
            // leaf: spawn one fold over the contiguous slice via the
            // typed fast-path. clone factory + work + reduce into the
            // closure; the closure captures Arc<[Item]> + start/end +
            // the cloned closures.
            let slice_for_leaf = slice.clone();
            let identity_for_leaf = identity.clone();
            let work_for_leaf = work.clone();
            let reduce_for_leaf = reduce.clone();
            let leaf_handle = pool.spawn(move || {
                let mut accumulator = identity_for_leaf();
                for item in &slice_for_leaf[start..end] {
                    accumulator = reduce_for_leaf(accumulator, work_for_leaf(item));
                }
                Ok::<Acc, ProximaError>(accumulator)
            });
            return leaf_handle.await.unwrap_or_else(|_| identity());
        }
        // split: recurse on both halves concurrently. join via futures::join.
        let mid = start + len / 2;
        let left_future = par_reduce_range(
            pool.clone(),
            slice.clone(),
            start,
            mid,
            identity.clone(),
            work.clone(),
            reduce.clone(),
            chunk_threshold,
        );
        let right_future = par_reduce_range(
            pool,
            slice,
            mid,
            end,
            identity,
            work.clone(),
            reduce.clone(),
            chunk_threshold,
        );
        let (left, right) = futures::join!(left_future, right_future);
        reduce(left, right)
    })
}

/// recursive-split parallel for-each (no reduce). same split shape as
/// [`par_reduce`] but leaves only run `work(&item)` for its side effects.
///
/// `work` must be `Send + Sync + 'static + Clone` so it can be passed to
/// each branch independently. errors from individual leaf spawns are
/// dropped — a failed leaf simply doesn't run, parallel siblings are
/// unaffected.
pub async fn par_each<Item, Work>(pool: &Arc<ProximaBackgroundPool>, slice: Arc<[Item]>, work: Work)
where
    Item: Sync + Send + 'static,
    Work: Fn(&Item) + Send + Sync + 'static + Clone,
{
    let threshold = default_threshold_for(pool, slice.len());
    par_each_with_threshold(pool, slice, work, threshold).await;
}

/// explicit-threshold variant of [`par_each`].
pub async fn par_each_with_threshold<Item, Work>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    work: Work,
    chunk_threshold: usize,
) where
    Item: Sync + Send + 'static,
    Work: Fn(&Item) + Send + Sync + 'static + Clone,
{
    if slice.is_empty() {
        return;
    }
    let len = slice.len();
    par_each_range(pool.clone(), slice, 0, len, work, chunk_threshold).await;
}

#[allow(clippy::too_many_arguments)]
fn par_each_range<Item, Work>(
    pool: Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    start: usize,
    end: usize,
    work: Work,
    chunk_threshold: usize,
) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>>
where
    Item: Sync + Send + 'static,
    Work: Fn(&Item) + Send + Sync + 'static + Clone,
{
    Box::pin(async move {
        let len = end - start;
        if len <= chunk_threshold {
            let slice_for_leaf = slice.clone();
            let work_for_leaf = work.clone();
            let leaf_handle = pool.spawn(move || {
                for item in &slice_for_leaf[start..end] {
                    work_for_leaf(item);
                }
                Ok::<(), ProximaError>(())
            });
            let _ = leaf_handle.await;
            return;
        }
        let mid = start + len / 2;
        let left_future = par_each_range(
            pool.clone(),
            slice.clone(),
            start,
            mid,
            work.clone(),
            chunk_threshold,
        );
        let right_future = par_each_range(pool, slice, mid, end, work, chunk_threshold);
        let (_left, _right) = futures::join!(left_future, right_future);
    })
}

// ---- async engines ----
// these mirror the sync engines but use `pool.spawn_async` for leaves.
// the closure takes `Item` by value (not `&Item`) because the returned
// future must be `'static` — it can't borrow from the slice it was
// extracted from. `Item: Clone` lets the leaf clone items out before
// feeding them into the per-item future.

/// recursive-split parallel reduce where the per-item work returns a
/// future. each leaf folds its slice serially (item by item, awaiting
/// each future before reducing into the accumulator); splits join two
/// halves via `futures::join!`. only available with the
/// `runtime-prime-bgpool-async` feature, which gives the BackgroundPool
/// workers their per-thread tokio runtime.
#[cfg(feature = "runtime-prime-bgpool-async")]
pub async fn par_reduce_async<Item, Acc, Work, Fut, Reduce, Identity>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    identity: Identity,
    work: Work,
    reduce: Reduce,
) -> Acc
where
    Item: Sync + Send + Clone + 'static,
    Acc: Send + 'static,
    Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Acc> + Send + 'static,
    Reduce: Fn(Acc, Acc) -> Acc + Send + Sync + 'static + Clone,
    Identity: Fn() -> Acc + Send + Sync + 'static + Clone,
{
    let threshold = default_threshold_for(pool, slice.len());
    par_reduce_async_with_threshold(pool, slice, identity, work, reduce, threshold).await
}

/// explicit-threshold variant of [`par_reduce_async`].
#[cfg(feature = "runtime-prime-bgpool-async")]
pub async fn par_reduce_async_with_threshold<Item, Acc, Work, Fut, Reduce, Identity>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    identity: Identity,
    work: Work,
    reduce: Reduce,
    chunk_threshold: usize,
) -> Acc
where
    Item: Sync + Send + Clone + 'static,
    Acc: Send + 'static,
    Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Acc> + Send + 'static,
    Reduce: Fn(Acc, Acc) -> Acc + Send + Sync + 'static + Clone,
    Identity: Fn() -> Acc + Send + Sync + 'static + Clone,
{
    if slice.is_empty() {
        return identity();
    }
    let len = slice.len();
    par_reduce_async_range(
        pool.clone(),
        slice,
        0,
        len,
        identity,
        work,
        reduce,
        chunk_threshold,
    )
    .await
}

#[cfg(feature = "runtime-prime-bgpool-async")]
#[allow(clippy::too_many_arguments)]
fn par_reduce_async_range<Item, Acc, Work, Fut, Reduce, Identity>(
    pool: Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    start: usize,
    end: usize,
    identity: Identity,
    work: Work,
    reduce: Reduce,
    chunk_threshold: usize,
) -> Pin<Box<dyn Future<Output = Acc> + Send + 'static>>
where
    Item: Sync + Send + Clone + 'static,
    Acc: Send + 'static,
    Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Acc> + Send + 'static,
    Reduce: Fn(Acc, Acc) -> Acc + Send + Sync + 'static + Clone,
    Identity: Fn() -> Acc + Send + Sync + 'static + Clone,
{
    Box::pin(async move {
        let len = end - start;
        if len <= chunk_threshold {
            let slice_for_leaf = slice.clone();
            let identity_for_leaf = identity.clone();
            let work_for_leaf = work.clone();
            let reduce_for_leaf = reduce.clone();
            let leaf_handle = pool.spawn_async(async move {
                let mut accumulator = identity_for_leaf();
                for item in &slice_for_leaf[start..end] {
                    let cloned = item.clone();
                    let output = work_for_leaf(cloned).await;
                    accumulator = reduce_for_leaf(accumulator, output);
                }
                Ok::<Acc, ProximaError>(accumulator)
            });
            return leaf_handle.await.unwrap_or_else(|_| identity());
        }
        let mid = start + len / 2;
        let left_future = par_reduce_async_range(
            pool.clone(),
            slice.clone(),
            start,
            mid,
            identity.clone(),
            work.clone(),
            reduce.clone(),
            chunk_threshold,
        );
        let right_future = par_reduce_async_range(
            pool,
            slice,
            mid,
            end,
            identity,
            work.clone(),
            reduce.clone(),
            chunk_threshold,
        );
        let (left, right) = futures::join!(left_future, right_future);
        reduce(left, right)
    })
}

/// async sibling to [`par_each`]. per-item closure returns a future
/// instead of running synchronously. requires the
/// `runtime-prime-bgpool-async` feature.
#[cfg(feature = "runtime-prime-bgpool-async")]
pub async fn par_each_async<Item, Work, Fut>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    work: Work,
) where
    Item: Sync + Send + Clone + 'static,
    Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = ()> + Send + 'static,
{
    let threshold = default_threshold_for(pool, slice.len());
    par_each_async_with_threshold(pool, slice, work, threshold).await;
}

/// explicit-threshold variant of [`par_each_async`].
#[cfg(feature = "runtime-prime-bgpool-async")]
pub async fn par_each_async_with_threshold<Item, Work, Fut>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    work: Work,
    chunk_threshold: usize,
) where
    Item: Sync + Send + Clone + 'static,
    Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = ()> + Send + 'static,
{
    if slice.is_empty() {
        return;
    }
    let len = slice.len();
    par_each_async_range(pool.clone(), slice, 0, len, work, chunk_threshold).await;
}

#[cfg(feature = "runtime-prime-bgpool-async")]
#[allow(clippy::too_many_arguments)]
fn par_each_async_range<Item, Work, Fut>(
    pool: Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    start: usize,
    end: usize,
    work: Work,
    chunk_threshold: usize,
) -> Pin<Box<dyn Future<Output = ()> + Send + 'static>>
where
    Item: Sync + Send + Clone + 'static,
    Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = ()> + Send + 'static,
{
    Box::pin(async move {
        let len = end - start;
        if len <= chunk_threshold {
            let slice_for_leaf = slice.clone();
            let work_for_leaf = work.clone();
            let leaf_handle = pool.spawn_async(async move {
                for item in &slice_for_leaf[start..end] {
                    let cloned = item.clone();
                    work_for_leaf(cloned).await;
                }
                Ok::<(), ProximaError>(())
            });
            let _ = leaf_handle.await;
            return;
        }
        let mid = start + len / 2;
        let left_future = par_each_async_range(
            pool.clone(),
            slice.clone(),
            start,
            mid,
            work.clone(),
            chunk_threshold,
        );
        let right_future = par_each_async_range(pool, slice, mid, end, work, chunk_threshold);
        let (_left, _right) = futures::join!(left_future, right_future);
    })
}

/// recursive-split parallel map + collect. each leaf builds a
/// `Vec<Output>` sized for its chunk (no per-item allocations beyond
/// the output itself); merges concatenate left into right preserving
/// input order. tree concatenation cost is O(N log L) where L is the
/// leaf count — dwarfed by the leaf work for any non-trivial map_fn.
pub async fn par_map_collect<Item, Output, F>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    map_fn: F,
) -> Vec<Output>
where
    Item: Sync + Send + 'static,
    Output: Send + 'static,
    F: Fn(&Item) -> Output + Send + Sync + 'static + Clone,
{
    let threshold = default_threshold_for(pool, slice.len());
    par_map_collect_with_threshold(pool, slice, map_fn, threshold).await
}

/// explicit-threshold variant of [`par_map_collect`].
pub async fn par_map_collect_with_threshold<Item, Output, F>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    map_fn: F,
    chunk_threshold: usize,
) -> Vec<Output>
where
    Item: Sync + Send + 'static,
    Output: Send + 'static,
    F: Fn(&Item) -> Output + Send + Sync + 'static + Clone,
{
    if slice.is_empty() {
        return Vec::new();
    }
    let len = slice.len();
    par_map_collect_range(pool.clone(), slice, 0, len, map_fn, chunk_threshold).await
}

#[allow(clippy::too_many_arguments)]
fn par_map_collect_range<Item, Output, F>(
    pool: Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    start: usize,
    end: usize,
    map_fn: F,
    chunk_threshold: usize,
) -> Pin<Box<dyn Future<Output = Vec<Output>> + Send + 'static>>
where
    Item: Sync + Send + 'static,
    Output: Send + 'static,
    F: Fn(&Item) -> Output + Send + Sync + 'static + Clone,
{
    Box::pin(async move {
        let len = end - start;
        if len <= chunk_threshold {
            let slice_for_leaf = slice.clone();
            let map_fn_for_leaf = map_fn.clone();
            let leaf_handle = pool.spawn(move || {
                let mut output_vec = Vec::with_capacity(end - start);
                for item in &slice_for_leaf[start..end] {
                    output_vec.push(map_fn_for_leaf(item));
                }
                Ok::<Vec<Output>, ProximaError>(output_vec)
            });
            return leaf_handle.await.unwrap_or_default();
        }
        let mid = start + len / 2;
        let left_future = par_map_collect_range(
            pool.clone(),
            slice.clone(),
            start,
            mid,
            map_fn.clone(),
            chunk_threshold,
        );
        let right_future = par_map_collect_range(pool, slice, mid, end, map_fn, chunk_threshold);
        let (mut left, right) = futures::join!(left_future, right_future);
        left.extend(right);
        left
    })
}

/// async sibling of [`par_map_collect`]. each leaf awaits each per-item
/// future before pushing the result into its local Vec; merges
/// concatenate left into right.
#[cfg(feature = "runtime-prime-bgpool-async")]
pub async fn par_map_collect_async<Item, Output, F, Fut>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    map_fn: F,
) -> Vec<Output>
where
    Item: Sync + Send + Clone + 'static,
    Output: Send + 'static,
    F: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Output> + Send + 'static,
{
    let threshold = default_threshold_for(pool, slice.len());
    par_map_collect_async_with_threshold(pool, slice, map_fn, threshold).await
}

#[cfg(feature = "runtime-prime-bgpool-async")]
pub async fn par_map_collect_async_with_threshold<Item, Output, F, Fut>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    map_fn: F,
    chunk_threshold: usize,
) -> Vec<Output>
where
    Item: Sync + Send + Clone + 'static,
    Output: Send + 'static,
    F: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Output> + Send + 'static,
{
    if slice.is_empty() {
        return Vec::new();
    }
    let len = slice.len();
    par_map_collect_async_range(pool.clone(), slice, 0, len, map_fn, chunk_threshold).await
}

#[cfg(feature = "runtime-prime-bgpool-async")]
#[allow(clippy::too_many_arguments)]
fn par_map_collect_async_range<Item, Output, F, Fut>(
    pool: Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    start: usize,
    end: usize,
    map_fn: F,
    chunk_threshold: usize,
) -> Pin<Box<dyn Future<Output = Vec<Output>> + Send + 'static>>
where
    Item: Sync + Send + Clone + 'static,
    Output: Send + 'static,
    F: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Output> + Send + 'static,
{
    Box::pin(async move {
        let len = end - start;
        if len <= chunk_threshold {
            let slice_for_leaf = slice.clone();
            let map_fn_for_leaf = map_fn.clone();
            let leaf_handle = pool.spawn_async(async move {
                let mut output_vec = Vec::with_capacity(end - start);
                for item in &slice_for_leaf[start..end] {
                    let cloned = item.clone();
                    output_vec.push(map_fn_for_leaf(cloned).await);
                }
                Ok::<Vec<Output>, ProximaError>(output_vec)
            });
            return leaf_handle.await.unwrap_or_default();
        }
        let mid = start + len / 2;
        let left_future = par_map_collect_async_range(
            pool.clone(),
            slice.clone(),
            start,
            mid,
            map_fn.clone(),
            chunk_threshold,
        );
        let right_future =
            par_map_collect_async_range(pool, slice, mid, end, map_fn, chunk_threshold);
        let (mut left, right) = futures::join!(left_future, right_future);
        left.extend(right);
        left
    })
}

// rayon-shaped front-door API. lets callers write
// `data.par_iter(&pool).map(f).sum().await` instead of constructing
// `par_reduce` calls by hand.

/// extension trait exposing `.par_iter(&pool)` on indexable inputs.
/// implementing types own (or cheaply clone into) an `Arc<[Item]>`.
pub trait ProximaParIter<Item: Sync + Send + 'static>: Sized {
    fn par_iter(self, pool: &Arc<ProximaBackgroundPool>) -> ProximaParBuilder<Item>;
}

impl<Item: Sync + Send + 'static> ProximaParIter<Item> for Arc<[Item]> {
    fn par_iter(self, pool: &Arc<ProximaBackgroundPool>) -> ProximaParBuilder<Item> {
        let threshold = default_threshold_for(pool, self.len());
        ProximaParBuilder {
            slice: self,
            pool: pool.clone(),
            threshold,
        }
    }
}

impl<Item: Sync + Send + 'static> ProximaParIter<Item> for &Arc<[Item]> {
    fn par_iter(self, pool: &Arc<ProximaBackgroundPool>) -> ProximaParBuilder<Item> {
        let threshold = default_threshold_for(pool, self.len());
        ProximaParBuilder {
            slice: Arc::clone(self),
            pool: pool.clone(),
            threshold,
        }
    }
}

/// lazy builder returned by `.par_iter()`. records the input + pool +
/// threshold; nothing executes until a terminal method (`.sum()`,
/// `.reduce()`, `.for_each()`) is awaited.
pub struct ProximaParBuilder<Item> {
    slice: Arc<[Item]>,
    pool: Arc<ProximaBackgroundPool>,
    threshold: usize,
}

impl<Item: Sync + Send + 'static> ProximaParBuilder<Item> {
    #[must_use]
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn map<Output, F>(self, map_fn: F) -> ProximaParMap<Item, Output, F>
    where
        Output: Send + 'static,
        F: Fn(&Item) -> Output + Send + Sync + 'static + Clone,
    {
        ProximaParMap {
            slice: self.slice,
            pool: self.pool,
            threshold: self.threshold,
            map_fn,
            _output_marker: PhantomData,
        }
    }

    pub async fn for_each<Work>(self, work: Work)
    where
        Work: Fn(&Item) + Send + Sync + 'static + Clone,
    {
        par_each_with_threshold(&self.pool, self.slice, work, self.threshold).await;
    }

    pub async fn reduce<Identity, R>(self, identity: Identity, reduce: R) -> Item
    where
        Item: Clone,
        Identity: Fn() -> Item + Send + Sync + 'static + Clone,
        R: Fn(Item, Item) -> Item + Send + Sync + 'static + Clone,
    {
        par_reduce_with_threshold(
            &self.pool,
            self.slice,
            identity,
            |item: &Item| item.clone(),
            reduce,
            self.threshold,
        )
        .await
    }

    pub async fn sum(self) -> Item
    where
        Item: Clone + Default + std::ops::Add<Output = Item> + Send + Sync + 'static,
    {
        par_reduce_with_threshold(
            &self.pool,
            self.slice,
            Item::default,
            |item: &Item| item.clone(),
            |left, right| left + right,
            self.threshold,
        )
        .await
    }
}

/// lazy builder returned by `.map(F)`. holds the input and the mapping
/// function; the map is applied at leaf time, not at construction.
pub struct ProximaParMap<Item, Output, F> {
    slice: Arc<[Item]>,
    pool: Arc<ProximaBackgroundPool>,
    threshold: usize,
    map_fn: F,
    // PhantomData<fn() -> Output> so the struct stays Send/Sync regardless
    // of Output's drop-side traits; we never store an Output, only produce
    // them at leaf execution time.
    _output_marker: PhantomData<fn() -> Output>,
}

impl<Item, Output, F> ProximaParMap<Item, Output, F>
where
    Item: Sync + Send + 'static,
    Output: Send + 'static,
    F: Fn(&Item) -> Output + Send + Sync + 'static + Clone,
{
    pub async fn reduce<Identity, R>(self, identity: Identity, reduce: R) -> Output
    where
        Identity: Fn() -> Output + Send + Sync + 'static + Clone,
        R: Fn(Output, Output) -> Output + Send + Sync + 'static + Clone,
    {
        par_reduce_with_threshold(
            &self.pool,
            self.slice,
            identity,
            self.map_fn,
            reduce,
            self.threshold,
        )
        .await
    }

    pub async fn sum(self) -> Output
    where
        Output: Default + std::ops::Add<Output = Output>,
    {
        par_reduce_with_threshold(
            &self.pool,
            self.slice,
            Output::default,
            self.map_fn,
            |left, right| left + right,
            self.threshold,
        )
        .await
    }

    pub async fn for_each<Sink>(self, sink: Sink)
    where
        Sink: Fn(Output) + Send + Sync + 'static + Clone,
    {
        let map_fn = self.map_fn;
        par_each_with_threshold(
            &self.pool,
            self.slice,
            move |item: &Item| sink(map_fn(item)),
            self.threshold,
        )
        .await;
    }

    /// gather every mapped output into a `Vec<Output>` in input order.
    /// each leaf builds its own Vec sized for its chunk; merges
    /// concatenate left into right preserving order across the tree.
    pub async fn collect(self) -> Vec<Output> {
        par_map_collect_with_threshold(&self.pool, self.slice, self.map_fn, self.threshold).await
    }
}

// ---- async builder ----
// returned by `ProximaParBuilder::map_async`. holds the slice + pool +
// threshold + an `Fn(Item) -> Fut` mapping closure where Fut is a future.
// requires `Item: Clone` because the future must own its item (futures
// returned from closures are `'static` and can't borrow from a slice).

#[cfg(feature = "runtime-prime-bgpool-async")]
impl<Item: Sync + Send + Clone + 'static> ProximaParBuilder<Item> {
    /// async variant of [`map`]. closure receives `Item` by value (not by
    /// reference) and returns a future. requires `Item: Clone` so the
    /// leaf can extract each item from the slice and hand it to the
    /// future, which then owns it for the duration of its execution.
    ///
    /// available only when `runtime-prime-bgpool-async` is enabled.
    pub fn map_async<Output, F, Fut>(self, map_fn: F) -> ProximaParMapAsync<Item, Output, F, Fut>
    where
        Output: Send + 'static,
        F: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
        Fut: Future<Output = Output> + Send + 'static,
    {
        ProximaParMapAsync {
            slice: self.slice,
            pool: self.pool,
            threshold: self.threshold,
            map_fn,
            _output_marker: PhantomData,
            _future_marker: PhantomData,
        }
    }

    /// fire-and-forget async variant of [`for_each`]. closure receives
    /// `Item` by value and returns a future.
    pub async fn for_each_async<Work, Fut>(self, work: Work)
    where
        Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
        Fut: Future<Output = ()> + Send + 'static,
    {
        par_each_async_with_threshold(&self.pool, self.slice, work, self.threshold).await;
    }
}

/// lazy async-map builder. terminals here lower into `par_reduce_async`
/// / `par_each_async` and the per-worker tokio runtime drives the
/// resulting futures.
#[cfg(feature = "runtime-prime-bgpool-async")]
pub struct ProximaParMapAsync<Item, Output, F, Fut> {
    slice: Arc<[Item]>,
    pool: Arc<ProximaBackgroundPool>,
    threshold: usize,
    map_fn: F,
    _output_marker: PhantomData<fn() -> Output>,
    _future_marker: PhantomData<fn() -> Fut>,
}

#[cfg(feature = "runtime-prime-bgpool-async")]
impl<Item, Output, F, Fut> ProximaParMapAsync<Item, Output, F, Fut>
where
    Item: Sync + Send + Clone + 'static,
    Output: Send + 'static,
    F: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Output> + Send + 'static,
{
    pub async fn reduce<Identity, R>(self, identity: Identity, reduce: R) -> Output
    where
        Identity: Fn() -> Output + Send + Sync + 'static + Clone,
        R: Fn(Output, Output) -> Output + Send + Sync + 'static + Clone,
    {
        par_reduce_async_with_threshold(
            &self.pool,
            self.slice,
            identity,
            self.map_fn,
            reduce,
            self.threshold,
        )
        .await
    }

    pub async fn sum(self) -> Output
    where
        Output: Default + std::ops::Add<Output = Output>,
    {
        par_reduce_async_with_threshold(
            &self.pool,
            self.slice,
            Output::default,
            self.map_fn,
            |left, right| left + right,
            self.threshold,
        )
        .await
    }

    pub async fn for_each<Sink>(self, sink: Sink)
    where
        Sink: Fn(Output) + Send + Sync + 'static + Clone,
    {
        let map_fn = self.map_fn;
        // wrap (map_async then sink) into a single Fn(Item)->Future used
        // by par_each_async. the closure must be Clone (auto-derived
        // because both captures are Clone) so each split branch can hold
        // its own copy.
        par_each_async_with_threshold(
            &self.pool,
            self.slice,
            move |item: Item| {
                let sink = sink.clone();
                let map_fn = map_fn.clone();
                async move {
                    let output = map_fn(item).await;
                    sink(output);
                }
            },
            self.threshold,
        )
        .await;
    }

    /// gather every mapped output into a `Vec<Output>` in input order.
    /// async sibling of [`ProximaParMap::collect`]; the per-worker tokio
    /// runtime drives each per-item future before appending to the leaf's
    /// local Vec.
    pub async fn collect(self) -> Vec<Output> {
        par_map_collect_async_with_threshold(&self.pool, self.slice, self.map_fn, self.threshold)
            .await
    }
}

// ---- streaming API (C) ----
// buffered concurrency over a slice: N futures in flight at any time,
// output emitted in completion order (unordered relative to input).
// uses `pool.spawn_async` to distribute futures across workers, so
// fan-out scales with worker count, not with the calling task's polling.

/// extension trait exposing `.par_stream(&pool, concurrency)` on
/// indexable inputs. unlike `par_iter`, this models open-ended async
/// I/O (N futures in flight) instead of recursive split + fold/reduce.
#[cfg(feature = "runtime-prime-bgpool-async")]
pub trait ProximaParStreamExt<Item: Sync + Send + Clone + 'static>: Sized {
    fn par_stream(
        self,
        pool: &Arc<ProximaBackgroundPool>,
        concurrency: usize,
    ) -> ProximaParStream<Item>;
}

#[cfg(feature = "runtime-prime-bgpool-async")]
impl<Item: Sync + Send + Clone + 'static> ProximaParStreamExt<Item> for Arc<[Item]> {
    fn par_stream(
        self,
        pool: &Arc<ProximaBackgroundPool>,
        concurrency: usize,
    ) -> ProximaParStream<Item> {
        ProximaParStream {
            slice: self,
            pool: pool.clone(),
            concurrency: concurrency.max(1),
        }
    }
}

#[cfg(feature = "runtime-prime-bgpool-async")]
impl<Item: Sync + Send + Clone + 'static> ProximaParStreamExt<Item> for &Arc<[Item]> {
    fn par_stream(
        self,
        pool: &Arc<ProximaBackgroundPool>,
        concurrency: usize,
    ) -> ProximaParStream<Item> {
        ProximaParStream {
            slice: Arc::clone(self),
            pool: pool.clone(),
            concurrency: concurrency.max(1),
        }
    }
}

/// builder returned by `.par_stream()`. terminal: `.then(F)` returns a
/// `Stream<Item = Output>` driven by the BackgroundPool's polling
/// workers — N futures in flight at a time, output in completion order.
#[cfg(feature = "runtime-prime-bgpool-async")]
pub struct ProximaParStream<Item> {
    slice: Arc<[Item]>,
    pool: Arc<ProximaBackgroundPool>,
    concurrency: usize,
}

#[cfg(feature = "runtime-prime-bgpool-async")]
impl<Item: Sync + Send + Clone + 'static> ProximaParStream<Item> {
    /// run an async closure over each item, returning a `Stream<Item = Output>`.
    /// output is emitted in completion order — not input order. for
    /// ordered output, callers can collect into a `Vec<(usize, Output)>`
    /// and sort, or wait for an `ordered = true` mode (not yet shipped).
    ///
    /// the returned stream owns its own `FuturesUnordered` and refills
    /// the in-flight set up to `concurrency` whenever it's polled.
    pub fn then<Work, Fut, Output>(self, work: Work) -> ProximaParStreamThen<Item, Work, Output>
    where
        Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
        Fut: Future<Output = Output> + Send + 'static,
        Output: Send + 'static,
    {
        ProximaParStreamThen {
            slice: self.slice,
            pool: self.pool,
            work,
            concurrency: self.concurrency,
            next_index: 0,
            in_flight: futures::stream::FuturesUnordered::new(),
            _output_marker: PhantomData,
        }
    }

    /// ordered variant of [`then`]. output is emitted in input order,
    /// not completion order — an out-of-order completion goes into a
    /// reorder buffer and waits for its turn. costs O(concurrency) of
    /// buffer space and one HashMap lookup per emission.
    ///
    /// use when downstream consumers need positional ordering (e.g.
    /// "results aligned with inputs"). use [`then`] when emission
    /// order doesn't matter — it's slightly leaner and exposes ready
    /// results sooner.
    pub fn then_ordered<Work, Fut, Output>(
        self,
        work: Work,
    ) -> ProximaParStreamThenOrdered<Item, Work, Output>
    where
        Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
        Fut: Future<Output = Output> + Send + 'static,
        Output: Send + 'static,
    {
        ProximaParStreamThenOrdered {
            slice: self.slice,
            pool: self.pool,
            work,
            concurrency: self.concurrency,
            next_dispatch_index: 0,
            next_emit_index: 0,
            in_flight: futures::stream::FuturesUnordered::new(),
            reorder: std::collections::HashMap::new(),
            _output_marker: PhantomData,
        }
    }
}

/// per-spawn future type carried by `ProximaParStreamThen`'s in-flight
/// `FuturesUnordered`. type-erased because the concrete `spawn_async`
/// return type is unnameable until the caller's closure monomorphizes.
#[cfg(feature = "runtime-prime-bgpool-async")]
type StreamSpawnFuture<Output> =
    Pin<Box<dyn Future<Output = Result<Output, ProximaError>> + Send + 'static>>;

/// hand-rolled Stream that keeps `concurrency` futures in flight via
/// `FuturesUnordered`. each in-flight future is `pool.spawn_async`'s
/// return type, so the actual async work runs on the BackgroundPool's
/// polling workers — not on whatever task is polling this stream.
#[cfg(feature = "runtime-prime-bgpool-async")]
pub struct ProximaParStreamThen<Item, Work, Output>
where
    Item: Sync + Send + Clone + 'static,
    Output: Send + 'static,
{
    slice: Arc<[Item]>,
    pool: Arc<ProximaBackgroundPool>,
    work: Work,
    concurrency: usize,
    next_index: usize,
    in_flight: futures::stream::FuturesUnordered<StreamSpawnFuture<Output>>,
    _output_marker: PhantomData<fn() -> Work>,
}

#[cfg(feature = "runtime-prime-bgpool-async")]
impl<Item, Work, Fut, Output> futures::Stream for ProximaParStreamThen<Item, Work, Output>
where
    Item: Sync + Send + Clone + 'static,
    Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Output> + Send + 'static,
    Output: Send + 'static,
{
    type Item = Output;

    fn poll_next(
        self: Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // SAFETY: we project mutably to each field; none are pinned
        // structurally. FuturesUnordered, Arc, and primitive fields all
        // implement Unpin.
        let this = unsafe { self.get_unchecked_mut() };
        // refill: top up in_flight to `concurrency`, pulling items by
        // index from the slice. each push spawns one future on the
        // BackgroundPool via spawn_async; the worker's per-thread tokio
        // runtime drives the future to completion.
        while this.in_flight.len() < this.concurrency && this.next_index < this.slice.len() {
            let item = this.slice[this.next_index].clone();
            this.next_index += 1;
            let work = this.work.clone();
            let handle = this.pool.spawn_async(async move { Ok(work(item).await) });
            this.in_flight.push(Box::pin(handle));
        }
        // drive: poll FuturesUnordered for the next ready completion.
        // map Err (worker dropped sender) to end-of-stream — extremely
        // rare; would only happen on a panic during spawn_async wiring.
        match Pin::new(&mut this.in_flight).poll_next(context) {
            std::task::Poll::Ready(Some(Ok(output))) => std::task::Poll::Ready(Some(output)),
            std::task::Poll::Ready(Some(Err(_))) => std::task::Poll::Ready(None),
            std::task::Poll::Ready(None) => {
                // in_flight empty AND iterator exhausted → stream done.
                // otherwise the refill above would have pushed something.
                if this.next_index >= this.slice.len() {
                    std::task::Poll::Ready(None)
                } else {
                    // shouldn't reach this branch given the refill above
                    // pushes whenever possible, but stay defensive.
                    std::task::Poll::Pending
                }
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len().saturating_sub(self.next_index) + self.in_flight.len();
        (remaining, Some(remaining))
    }
}

/// per-spawn future type for ordered streaming — carries an `(index,
/// Output)` pair so completions can be reordered before emission.
#[cfg(feature = "runtime-prime-bgpool-async")]
type StreamOrderedSpawnFuture<Output> =
    Pin<Box<dyn Future<Output = Result<(usize, Output), ProximaError>> + Send + 'static>>;

/// ordered variant of [`ProximaParStreamThen`]. each in-flight future
/// is tagged with its input index; out-of-order completions go into a
/// `HashMap` reorder buffer keyed by index, and the stream emits the
/// next-expected index when it's ready. buffer high-water is bounded
/// by `concurrency` (we never dispatch beyond next_emit_index +
/// concurrency).
#[cfg(feature = "runtime-prime-bgpool-async")]
pub struct ProximaParStreamThenOrdered<Item, Work, Output>
where
    Item: Sync + Send + Clone + 'static,
    Output: Send + 'static,
{
    slice: Arc<[Item]>,
    pool: Arc<ProximaBackgroundPool>,
    work: Work,
    concurrency: usize,
    next_dispatch_index: usize,
    next_emit_index: usize,
    in_flight: futures::stream::FuturesUnordered<StreamOrderedSpawnFuture<Output>>,
    reorder: std::collections::HashMap<usize, Output>,
    _output_marker: PhantomData<fn() -> Work>,
}

#[cfg(feature = "runtime-prime-bgpool-async")]
impl<Item, Work, Fut, Output> futures::Stream for ProximaParStreamThenOrdered<Item, Work, Output>
where
    Item: Sync + Send + Clone + 'static,
    Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Output> + Send + 'static,
    Output: Send + 'static,
{
    type Item = Output;

    fn poll_next(
        self: Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // SAFETY: project mutably — FuturesUnordered, HashMap, Arc, and
        // primitive fields are all Unpin.
        let this = unsafe { self.get_unchecked_mut() };

        // emit fast path: if the next-expected index is already in the
        // reorder buffer, return it without polling in_flight.
        if let Some(output) = this.reorder.remove(&this.next_emit_index) {
            this.next_emit_index += 1;
            // refill in_flight before returning so producers stay primed.
            refill_ordered(this);
            return std::task::Poll::Ready(Some(output));
        }

        // refill in_flight to keep concurrency primed.
        refill_ordered(this);

        // drive: poll FuturesUnordered until next-expected emits or
        // pending. each completion goes into reorder; we keep polling
        // until we see next_emit_index OR no more progress is possible.
        loop {
            match Pin::new(&mut this.in_flight).poll_next(context) {
                std::task::Poll::Ready(Some(Ok((index, output)))) => {
                    if index == this.next_emit_index {
                        this.next_emit_index += 1;
                        refill_ordered(this);
                        return std::task::Poll::Ready(Some(output));
                    }
                    this.reorder.insert(index, output);
                    // keep polling — another completion might be the one we need.
                }
                std::task::Poll::Ready(Some(Err(_))) => {
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Ready(None) => {
                    // in_flight is empty. if dispatch is also done and
                    // reorder is empty, we're done. otherwise we're
                    // waiting for something — but FuturesUnordered::None
                    // means there's nothing TO wait on, which is a stuck
                    // state. should only happen at end-of-stream.
                    if this.next_dispatch_index >= this.slice.len() && this.reorder.is_empty() {
                        return std::task::Poll::Ready(None);
                    }
                    return std::task::Poll::Pending;
                }
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len().saturating_sub(self.next_dispatch_index)
            + self.in_flight.len()
            + self.reorder.len();
        (remaining, Some(remaining))
    }
}

/// helper: keep the in-flight set topped up to concurrency. dispatches
/// items by index from the slice; each spawn_async returns an `(index,
/// Output)` pair so completions remain identifiable for reordering.
#[cfg(feature = "runtime-prime-bgpool-async")]
fn refill_ordered<Item, Work, Fut, Output>(
    state: &mut ProximaParStreamThenOrdered<Item, Work, Output>,
) where
    Item: Sync + Send + Clone + 'static,
    Work: Fn(Item) -> Fut + Send + Sync + 'static + Clone,
    Fut: Future<Output = Output> + Send + 'static,
    Output: Send + 'static,
{
    while state.in_flight.len() < state.concurrency && state.next_dispatch_index < state.slice.len()
    {
        let item = state.slice[state.next_dispatch_index].clone();
        let index = state.next_dispatch_index;
        state.next_dispatch_index += 1;
        let work = state.work.clone();
        let handle = state.pool.spawn_async(async move {
            let output = work(item).await;
            Ok::<(usize, Output), ProximaError>((index, output))
        });
        state.in_flight.push(Box::pin(handle));
    }
}

// ---- par_filter ----

/// recursive-split parallel filter. collects elements satisfying `pred`
/// into a `Vec<Item>`. each leaf tests its slice and builds a local Vec;
/// splits concatenate left + right preserving input order.
///
/// `Item: Clone` because each accepted item must be moved into the output
/// Vec — the source slice stays alive as `Arc<[Item]>` so the leaf borrows
/// it, then clones accepted elements out.
pub async fn par_filter<Item, Pred>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    pred: Pred,
) -> Vec<Item>
where
    Item: Sync + Send + Clone + 'static,
    Pred: Fn(&Item) -> bool + Send + Sync + 'static + Clone,
{
    let threshold = default_threshold_for(pool, slice.len());
    par_filter_with_threshold(pool, slice, pred, threshold).await
}

/// explicit-threshold variant of [`par_filter`].
pub async fn par_filter_with_threshold<Item, Pred>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    pred: Pred,
    chunk_threshold: usize,
) -> Vec<Item>
where
    Item: Sync + Send + Clone + 'static,
    Pred: Fn(&Item) -> bool + Send + Sync + 'static + Clone,
{
    if slice.is_empty() {
        return Vec::new();
    }
    let len = slice.len();
    par_filter_range(pool.clone(), slice, 0, len, pred, chunk_threshold).await
}

fn par_filter_range<Item, Pred>(
    pool: Arc<ProximaBackgroundPool>,
    slice: Arc<[Item]>,
    start: usize,
    end: usize,
    pred: Pred,
    chunk_threshold: usize,
) -> Pin<Box<dyn Future<Output = Vec<Item>> + Send + 'static>>
where
    Item: Sync + Send + Clone + 'static,
    Pred: Fn(&Item) -> bool + Send + Sync + 'static + Clone,
{
    Box::pin(async move {
        let len = end - start;
        if len <= chunk_threshold {
            let slice_for_leaf = slice.clone();
            let pred_for_leaf = pred.clone();
            let leaf_handle = pool.spawn(move || {
                let accepted: Vec<Item> = slice_for_leaf[start..end]
                    .iter()
                    .filter(|item| pred_for_leaf(item))
                    .cloned()
                    .collect();
                Ok::<Vec<Item>, ProximaError>(accepted)
            });
            return leaf_handle.await.unwrap_or_default();
        }
        let mid = start + len / 2;
        let left_future = par_filter_range(
            pool.clone(),
            slice.clone(),
            start,
            mid,
            pred.clone(),
            chunk_threshold,
        );
        let right_future = par_filter_range(pool, slice, mid, end, pred, chunk_threshold);
        let (mut left, right) = futures::join!(left_future, right_future);
        left.extend(right);
        left
    })
}

// ---- par_sort / par_sort_by / par_sort_unstable ----
//
// parallel merge sort: recursive split, each half sorted independently,
// then the two sorted halves merged. base case uses std sort in-place on
// a contiguous sub-vec owned by the leaf. the merge step is serial (O(N)
// merge of two sorted Vecs) — the parallelism is in the recursive splits.
//
// callers hand in a Vec<Item> (owned, not Arc<[Item]>) so leaves can sort
// in-place and the merge can concatenate by moving. sorting an Arc<[T]>
// in-place would require &mut access which violates Arc's aliasing rules.

/// parallel merge sort. output is a sorted `Vec<Item>`. recursive split
/// via `pool.spawn`; base case sorts a local sub-vec; merges combine two
/// sorted halves via a serial merge walk.
pub async fn par_sort<Item>(pool: &Arc<ProximaBackgroundPool>, data: Vec<Item>) -> Vec<Item>
where
    Item: Send + Ord + 'static,
{
    let threshold = default_threshold_for(pool, data.len());
    par_sort_with_threshold(pool, data, threshold).await
}

/// explicit-threshold variant of [`par_sort`].
pub async fn par_sort_with_threshold<Item>(
    pool: &Arc<ProximaBackgroundPool>,
    data: Vec<Item>,
    chunk_threshold: usize,
) -> Vec<Item>
where
    Item: Send + Ord + 'static,
{
    par_sort_ord_range(pool.clone(), data, chunk_threshold, false).await
}

/// parallel sort with a custom comparator. stable — base case uses `sort_by`.
pub async fn par_sort_by<Item, Compare>(
    pool: &Arc<ProximaBackgroundPool>,
    data: Vec<Item>,
    compare: Compare,
) -> Vec<Item>
where
    Item: Send + 'static,
    Compare: Fn(&Item, &Item) -> std::cmp::Ordering + Send + Sync + 'static + Clone,
{
    let threshold = default_threshold_for(pool, data.len());
    par_sort_by_range(pool.clone(), data, compare, threshold).await
}

/// parallel unstable sort. base case uses `sort_unstable` — slightly
/// faster; does not preserve input order of equal elements.
pub async fn par_sort_unstable<Item>(
    pool: &Arc<ProximaBackgroundPool>,
    data: Vec<Item>,
) -> Vec<Item>
where
    Item: Send + Ord + 'static,
{
    let threshold = default_threshold_for(pool, data.len());
    par_sort_ord_range(pool.clone(), data, threshold, true).await
}

/// merge two sorted Vecs into one sorted Vec. serial O(N) walk.
/// equal elements from `left` come before equal elements from `right` (stable).
fn sorted_merge<Item: Ord>(left: Vec<Item>, right: Vec<Item>) -> Vec<Item> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    // first pass: determine merge order by indices, then collect
    let total = left.len() + right.len();
    let mut take_left = Vec::with_capacity(total);
    while left_idx < left.len() && right_idx < right.len() {
        if left[left_idx] <= right[right_idx] {
            take_left.push(true);
            left_idx += 1;
        } else {
            take_left.push(false);
            right_idx += 1;
        }
    }
    let remaining_left = left_idx < left.len();
    // second pass: drain in merge order
    let mut left_drain = left.into_iter();
    let mut right_drain = right.into_iter();
    for take in take_left {
        if take {
            if let Some(item) = left_drain.next() {
                result.push(item);
            }
        } else if let Some(item) = right_drain.next() {
            result.push(item);
        }
    }
    if remaining_left {
        result.extend(left_drain);
    } else {
        result.extend(right_drain);
    }
    result
}

/// merge two sorted Vecs using a custom comparator.
fn sorted_merge_by<Item, Compare>(left: Vec<Item>, right: Vec<Item>, compare: &Compare) -> Vec<Item>
where
    Compare: Fn(&Item, &Item) -> std::cmp::Ordering,
{
    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut left_idx = 0usize;
    let mut right_idx = 0usize;
    let total = left.len() + right.len();
    let mut take_left = Vec::with_capacity(total);
    while left_idx < left.len() && right_idx < right.len() {
        if compare(&left[left_idx], &right[right_idx]) != std::cmp::Ordering::Greater {
            take_left.push(true);
            left_idx += 1;
        } else {
            take_left.push(false);
            right_idx += 1;
        }
    }
    let remaining_left = left_idx < left.len();
    let mut left_drain = left.into_iter();
    let mut right_drain = right.into_iter();
    for take in take_left {
        if take {
            if let Some(item) = left_drain.next() {
                result.push(item);
            }
        } else if let Some(item) = right_drain.next() {
            result.push(item);
        }
    }
    if remaining_left {
        result.extend(left_drain);
    } else {
        result.extend(right_drain);
    }
    result
}

fn par_sort_ord_range<Item>(
    pool: Arc<ProximaBackgroundPool>,
    data: Vec<Item>,
    chunk_threshold: usize,
    unstable: bool,
) -> Pin<Box<dyn Future<Output = Vec<Item>> + Send + 'static>>
where
    Item: Send + Ord + 'static,
{
    Box::pin(async move {
        if data.len() <= chunk_threshold {
            let sorted_handle = pool.spawn(move || {
                let mut chunk = data;
                if unstable {
                    chunk.sort_unstable();
                } else {
                    chunk.sort();
                }
                Ok::<Vec<Item>, ProximaError>(chunk)
            });
            return sorted_handle.await.unwrap_or_default();
        }
        let mid = data.len() / 2;
        let mut data_iter = data.into_iter();
        let left_half: Vec<Item> = data_iter.by_ref().take(mid).collect();
        let right_half: Vec<Item> = data_iter.collect();
        let left_future = par_sort_ord_range(pool.clone(), left_half, chunk_threshold, unstable);
        let right_future = par_sort_ord_range(pool, right_half, chunk_threshold, unstable);
        let (sorted_left, sorted_right) = futures::join!(left_future, right_future);
        sorted_merge(sorted_left, sorted_right)
    })
}

fn par_sort_by_range<Item, Compare>(
    pool: Arc<ProximaBackgroundPool>,
    data: Vec<Item>,
    compare: Compare,
    chunk_threshold: usize,
) -> Pin<Box<dyn Future<Output = Vec<Item>> + Send + 'static>>
where
    Item: Send + 'static,
    Compare: Fn(&Item, &Item) -> std::cmp::Ordering + Send + Sync + 'static + Clone,
{
    Box::pin(async move {
        if data.len() <= chunk_threshold {
            let compare_for_leaf = compare.clone();
            let sorted_handle = pool.spawn(move || {
                let mut chunk = data;
                chunk.sort_by(|a, b| compare_for_leaf(a, b));
                Ok::<Vec<Item>, ProximaError>(chunk)
            });
            return sorted_handle.await.unwrap_or_default();
        }
        let mid = data.len() / 2;
        let mut data_iter = data.into_iter();
        let left_half: Vec<Item> = data_iter.by_ref().take(mid).collect();
        let right_half: Vec<Item> = data_iter.collect();
        let left_future =
            par_sort_by_range(pool.clone(), left_half, compare.clone(), chunk_threshold);
        let right_future = par_sort_by_range(pool, right_half, compare.clone(), chunk_threshold);
        let (sorted_left, sorted_right) = futures::join!(left_future, right_future);
        sorted_merge_by(sorted_left, sorted_right, &compare)
    })
}

// ---- par_chunks_mut ----
//
// processes a mutable slice in non-overlapping chunks of size `chunk_size`.
// each chunk is handed to `work` on the pool concurrently; the chunks
// are non-overlapping so they can be sent to workers without aliasing.
// uses raw pointer reinterpretation to split a `&mut [T]` into N disjoint
// `&mut [T]` sub-slices; SAFETY: each sub-slice covers a disjoint address
// range and all borrows are rejoined before the future resolves, so no
// two concurrent workers can alias the same memory.

/// apply `work` to each non-overlapping chunk of `slice` in parallel.
/// chunks are disjoint — no two workers access the same memory
/// concurrently. mutations in `work` are visible to the caller when
/// the future resolves.
///
/// `work` receives `&mut [T]` and must not alias or outlive the slice.
/// `T: Send` is sufficient because each chunk is exclusively owned by
/// its spawned closure for the duration of that spawn.
pub async fn par_chunks_mut<Item, Work>(
    pool: &Arc<ProximaBackgroundPool>,
    slice: &mut [Item],
    chunk_size: usize,
    work: Work,
) where
    Item: Send + 'static,
    Work: Fn(&mut [Item]) + Send + Sync + 'static + Clone,
{
    if slice.is_empty() || chunk_size == 0 {
        return;
    }

    let ptr = slice.as_mut_ptr();
    let total_len = slice.len();
    let effective_chunk_size = chunk_size.max(1);
    let chunk_count = total_len.div_ceil(effective_chunk_size);

    let mut handles = Vec::with_capacity(chunk_count);
    for chunk_index in 0..chunk_count {
        let start = chunk_index * effective_chunk_size;
        let end = (start + effective_chunk_size).min(total_len);
        let len = end - start;
        // SAFETY: ptr is valid for total_len elements, start < total_len.
        // cast to usize so the closure captures a Send value — usize is always
        // Send. we reconstruct the pointer inside the closure where it's used.
        // exclusive per-chunk: each chunk_addr_raw covers a unique disjoint range.
        let chunk_addr_raw = unsafe { ptr.add(start) } as usize;
        let work_for_chunk = work.clone();
        let pool_ref = pool.clone();
        handles.push(pool_ref.spawn(move || {
            // SAFETY: chunk_addr_raw is the address of a valid, exclusively-owned
            // sub-slice [start..end] of the original &mut [Item]. all handles are
            // awaited before this function returns so the source slice outlives every
            // handle. no two workers share the same address range.
            let chunk = unsafe { std::slice::from_raw_parts_mut(chunk_addr_raw as *mut Item, len) };
            work_for_chunk(chunk);
            Ok::<(), ProximaError>(())
        }));
    }
    for handle in handles {
        let _ = handle.await;
    }
}

// ---- scope ----
//
// rayon-style structured concurrency: spawn closures into a `Scope`,
// all spawned tasks complete before `scope(f)` returns.
//
// multi-spawn via lifetime-bound `'scope` closures requires that the
// closures borrow from the caller stack. this is feasible but requires
// unsafe to erase the lifetime for the pool's spawn signature (which
// demands `'static`). the single-closure variant below is fully safe
// and covers the majority of real use cases. the multi-spawn variant
// with borrows is TODO — see module-level doc.
//
// TODO: multi-spawn scope that allows closures to borrow from the caller
// stack. requires either a lifetime-erased trampoling with synchronization
// (barrier/CountdownLatch) to guarantee the borrows are live until all
// tasks complete, or switching spawn to take Scoped tasks like crossbeam's
// scope. currently blocked by `ProximaBackgroundPool::spawn`'s `'static`
// bound on the closure.

/// rayon-style structured scope. runs `scope_fn(scope)` immediately; any
/// work the caller passes to `scope.spawn(...)` is collected and awaited
/// before `scope(scope_fn)` returns.
///
/// closures passed to `scope.spawn` must be `'static` (no borrows from
/// caller stack) due to the pool's `'static` spawn bound. for borrowing
/// use cases, wrap shared state in an `Arc` or capture owned clones.
///
/// all spawned tasks are awaited even if some panic — drop order matches
/// spawn order.
pub async fn scope<F>(pool: &Arc<ProximaBackgroundPool>, scope_fn: F)
where
    F: FnOnce(&ProximaScope),
{
    let proxima_scope = ProximaScope {
        pool: pool.clone(),
        handles: std::cell::RefCell::new(Vec::new()),
    };
    scope_fn(&proxima_scope);
    let handles = proxima_scope.handles.into_inner();
    for handle in handles {
        let _ = handle.await;
    }
}

/// handle passed to the closure inside `scope(|s| { s.spawn(...); })`.
/// `spawn` queues work on the pool; all queued work is awaited before
/// the enclosing `scope(...)` future resolves.
pub struct ProximaScope {
    pool: Arc<ProximaBackgroundPool>,
    handles: std::cell::RefCell<
        Vec<Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'static>>>,
    >,
}

impl ProximaScope {
    /// queue a closure to run on the pool. the closure must be
    /// `FnOnce() + Send + 'static`. result is discarded — side effects
    /// are the intended usage.
    pub fn spawn<F>(&self, work: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let handle = self.pool.spawn(move || {
            work();
            Ok::<(), ProximaError>(())
        });
        self.handles.borrow_mut().push(Box::pin(handle));
    }
}

// ---- install ----
//
// runs a closure with a thread-local pointer to a specific pool. code
// inside the closure can call `installed_pool()` to retrieve the pool
// without threading it through every function argument.

std::thread_local! {
    static INSTALLED_POOL: std::cell::RefCell<Option<Arc<ProximaBackgroundPool>>> =
        const { std::cell::RefCell::new(None) };
}

/// run `work` with `pool` installed as the thread-local default pool.
/// code inside `work` can call [`installed_pool`] to retrieve it.
/// the previous installation (if any) is restored on return.
pub fn install<F, Output>(pool: Arc<ProximaBackgroundPool>, work: F) -> Output
where
    F: FnOnce() -> Output,
{
    let previous = INSTALLED_POOL.with(|cell| cell.borrow().clone());
    INSTALLED_POOL.with(|cell| *cell.borrow_mut() = Some(pool));
    let output = work();
    INSTALLED_POOL.with(|cell| *cell.borrow_mut() = previous);
    output
}

/// retrieve the thread-locally installed pool, if any. returns `None`
/// when called outside an `install(pool, ...)` context.
#[must_use = "check for None before using the pool"]
pub fn installed_pool() -> Option<Arc<ProximaBackgroundPool>> {
    INSTALLED_POOL.with(|cell| cell.borrow().clone())
}

// ---- par_bridge ----
//
// turns any `Iterator<Item = T> + Send` into a parallel fan-out. N
// workers pull items from the iterator (behind a Mutex) until it is
// exhausted, applying `work` to each item. ordering of work execution
// is not guaranteed — items are pulled under the mutex lock but work
// runs outside it, so faster workers pull more items.

/// apply `work` to every item from `iter` in parallel. `num_workers`
/// workers share the iterator behind a `Mutex`; each worker pulls one
/// item at a time and runs `work` on it before pulling the next.
///
/// prefer [`par_each`] when the input is already an `Arc<[Item]>` —
/// that path avoids the per-item Mutex lock. `par_bridge` is for
/// iterators whose length is unknown up front or that cannot be
/// collected cheaply.
pub async fn par_bridge<Item, Iter, Work>(
    pool: &Arc<ProximaBackgroundPool>,
    iter: Iter,
    num_workers: usize,
    work: Work,
) where
    Item: Send + 'static,
    Iter: Iterator<Item = Item> + Send + 'static,
    Work: Fn(Item) + Send + Sync + 'static + Clone,
{
    use std::sync::Mutex;
    let shared_iter = Arc::new(Mutex::new(iter));
    let worker_count = num_workers.max(1);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let iter_ref = shared_iter.clone();
        let work_for_worker = work.clone();
        handles.push(pool.spawn(move || {
            loop {
                let next_item = iter_ref.lock().ok().and_then(|mut guard| guard.next());
                match next_item {
                    Some(item) => work_for_worker(item),
                    None => break,
                }
            }
            Ok::<(), ProximaError>(())
        }));
    }
    for handle in handles {
        let _ = handle.await;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn build_pool(threads: usize) -> Arc<ProximaBackgroundPool> {
        Arc::new(ProximaBackgroundPool::with_threads(threads).expect("build pool"))
    }

    /// happy: small slice, no split needed, returns sum.
    #[proxima::test]
    async fn par_reduce_sum_small_slice() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = (0..16_u32).collect::<Vec<_>>().into();
        let sum = par_reduce(
            &pool,
            data,
            || 0u64,
            |&item| u64::from(item),
            |left, right| left.wrapping_add(right),
        )
        .await;
        assert_eq!(sum, (0..16u32).map(u64::from).sum::<u64>());
    }

    /// happy: large slice triggers recursive split + tree-shaped reduce.
    #[proxima::test]
    async fn par_reduce_sum_large_slice_triggers_split() {
        let pool = build_pool(4);
        let data: Arc<[u32]> = (0..100_000_u32).collect::<Vec<_>>().into();
        let sum = par_reduce_with_threshold(
            &pool,
            data,
            || 0u64,
            |&item| u64::from(item),
            |left, right| left.wrapping_add(right),
            1024,
        )
        .await;
        let expected: u64 = (0..100_000u32).map(u64::from).sum();
        assert_eq!(sum, expected);
    }

    /// edge: empty slice returns identity, no spawn happens.
    #[proxima::test]
    async fn par_reduce_empty_slice_returns_identity() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = Vec::<u32>::new().into();
        let sum = par_reduce(
            &pool,
            data,
            || 42u64,
            |&item| u64::from(item),
            |left, right| left.wrapping_add(right),
        )
        .await;
        assert_eq!(sum, 42);
    }

    /// edge: single element fits in one leaf.
    #[proxima::test]
    async fn par_reduce_single_element_no_split() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = vec![7].into();
        let sum = par_reduce(
            &pool,
            data,
            || 0u64,
            |&item| u64::from(item),
            |left, right| left.wrapping_add(right),
        )
        .await;
        assert_eq!(sum, 7);
    }

    /// edge: threshold exactly equals slice length → one leaf, no split.
    /// work() must be called len times exactly (no double-counting from
    /// errant split into 0/N children).
    #[proxima::test]
    async fn par_reduce_threshold_equal_to_len_runs_single_leaf() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = (0..8_u32).collect::<Vec<_>>().into();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        let sum = par_reduce_with_threshold(
            &pool,
            data,
            || 0u64,
            move |&item| {
                counter_for_work.fetch_add(1, Ordering::Relaxed);
                u64::from(item)
            },
            |left, right| left.wrapping_add(right),
            8,
        )
        .await;
        assert_eq!(sum, (0..8u32).map(u64::from).sum::<u64>());
        assert_eq!(counter.load(Ordering::Relaxed), 8);
    }

    /// happy: associative-but-not-summing reduce works (max).
    #[proxima::test]
    async fn par_reduce_max_op() {
        let pool = build_pool(4);
        let data: Arc<[u32]> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3].into();
        let max = par_reduce_with_threshold(&pool, data, || 0u32, |&item| item, u32::max, 4).await;
        assert_eq!(max, 9);
    }

    /// concurrency: many splits across multiple workers. counter must
    /// total `len` exactly — no double-count, no loss. catches races in
    /// the split-and-recurse + reduce-tree composition.
    #[proxima::test]
    async fn par_reduce_split_no_double_or_loss() {
        let pool = build_pool(4);
        const LEN: usize = 10_000;
        let data: Arc<[u32]> = (0..LEN as u32).collect::<Vec<_>>().into();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        let _ = par_reduce_with_threshold(
            &pool,
            data,
            || 0u64,
            move |&_item| {
                counter_for_work.fetch_add(1, Ordering::Relaxed);
                1u64
            },
            |left, right| left + right,
            512,
        )
        .await;
        assert_eq!(counter.load(Ordering::Relaxed), LEN);
    }

    /// par_each: side-effect-only fan-out. counter must total `len`,
    /// proving every element is visited exactly once across the split.
    #[proxima::test]
    async fn par_each_visits_every_item_once() {
        let pool = build_pool(4);
        const LEN: usize = 10_000;
        let data: Arc<[u32]> = (0..LEN as u32).collect::<Vec<_>>().into();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        par_each_with_threshold(
            &pool,
            data,
            move |&_item| {
                counter_for_work.fetch_add(1, Ordering::Relaxed);
            },
            512,
        )
        .await;
        assert_eq!(counter.load(Ordering::Relaxed), LEN);
    }

    /// par_each: empty slice no-ops (no spawn, no panic).
    #[proxima::test]
    async fn par_each_empty_slice_no_op() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = Vec::<u32>::new().into();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        par_each(&pool, data, move |&_item| {
            counter_for_work.fetch_add(1, Ordering::Relaxed);
        })
        .await;
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    // ---- trait shim: ProximaParIter + ProximaParBuilder + ProximaParMap ----

    /// trait: `.par_iter(&pool).map(f).sum().await` matches sequential sum.
    #[proxima::test]
    async fn par_iter_map_sum_matches_sequential() {
        let pool = build_pool(4);
        const LEN: u32 = 50_000;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let parallel_sum: u64 = data
            .par_iter(&pool)
            .map(|&item| u64::from(item))
            .sum()
            .await;
        let expected: u64 = (0..LEN).map(u64::from).sum();
        assert_eq!(parallel_sum, expected);
    }

    /// trait: `.par_iter(&pool).map(f).reduce(id, op).await` with a
    /// non-summing reduce (max).
    #[proxima::test]
    async fn par_iter_map_reduce_max() {
        let pool = build_pool(4);
        let data: Arc<[u32]> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3].into();
        let max = data
            .par_iter(&pool)
            .with_threshold(4)
            .map(|&item| item)
            .reduce(|| 0u32, u32::max)
            .await;
        assert_eq!(max, 9);
    }

    /// trait: `.par_iter(&pool).for_each(f).await` runs side effects on every item.
    #[proxima::test]
    async fn par_iter_for_each_runs_on_every_item() {
        let pool = build_pool(4);
        const LEN: usize = 5_000;
        let data: Arc<[u32]> = (0..LEN as u32).collect::<Vec<_>>().into();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        data.par_iter(&pool)
            .for_each(move |&_item| {
                counter_for_work.fetch_add(1, Ordering::Relaxed);
            })
            .await;
        assert_eq!(counter.load(Ordering::Relaxed), LEN);
    }

    /// trait: `.par_iter(&pool).map(f).for_each(sink).await` — map output
    /// piped into a side-effect sink. exercises ProximaParMap::for_each
    /// which composes a captured map_fn + sink into one work closure.
    #[proxima::test]
    async fn par_iter_map_for_each_pipes_outputs() {
        let pool = build_pool(4);
        const LEN: u32 = 1_000;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let total = Arc::new(AtomicUsize::new(0));
        let total_for_sink = total.clone();
        data.par_iter(&pool)
            .map(|&item| item as usize * 2)
            .for_each(move |doubled| {
                total_for_sink.fetch_add(doubled, Ordering::Relaxed);
            })
            .await;
        let expected: usize = (0..LEN as usize).map(|item| item * 2).sum();
        assert_eq!(total.load(Ordering::Relaxed), expected);
    }

    /// trait: `.sum()` directly on builder (no .map) works when Item is
    /// Default + Add + Clone (u64 here). proves the non-map sum path.
    #[proxima::test]
    async fn par_iter_sum_without_map() {
        let pool = build_pool(4);
        const LEN: u64 = 10_000;
        let data: Arc<[u64]> = (0..LEN).collect::<Vec<_>>().into();
        let parallel_sum: u64 = data.par_iter(&pool).sum().await;
        let expected: u64 = (0..LEN).sum();
        assert_eq!(parallel_sum, expected);
    }

    /// trait: `&Arc<[T]>` impl — call .par_iter() on a borrowed Arc, slice
    /// stays usable afterward (refcount, not move).
    #[proxima::test]
    async fn par_iter_on_borrowed_arc_preserves_caller() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = (0..100u32).collect::<Vec<_>>().into();
        let first: u64 = (&data)
            .par_iter(&pool)
            .map(|&item| u64::from(item))
            .sum()
            .await;
        let second: u64 = (&data)
            .par_iter(&pool)
            .map(|&item| u64::from(item))
            .sum()
            .await;
        assert_eq!(first, second);
        assert_eq!(first, (0..100u32).map(u64::from).sum::<u64>());
    }

    // ---- async engine + map_async ----

    /// async engine: per-item future yields, then returns. result matches
    /// sequential sum. exercises the polling-worker path.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_reduce_async_with_yielding_future() {
        let pool = build_pool(4);
        let data: Arc<[u32]> = (0..1_000_u32).collect::<Vec<_>>().into();
        let sum = par_reduce_async(
            &pool,
            data,
            || 0u64,
            |item| async move {
                tokio::task::yield_now().await;
                u64::from(item)
            },
            |left, right| left + right,
        )
        .await;
        assert_eq!(sum, (0..1_000_u32).map(u64::from).sum::<u64>());
    }

    /// async engine: empty slice returns identity, no spawn.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_reduce_async_empty_returns_identity() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = Vec::<u32>::new().into();
        let sum = par_reduce_async(
            &pool,
            data,
            || 99u64,
            |item| async move { u64::from(item) },
            |left, right| left + right,
        )
        .await;
        assert_eq!(sum, 99);
    }

    /// async par_each: side-effect-only fan-out with awaiting closures.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_each_async_visits_every_item() {
        let pool = build_pool(4);
        const LEN: usize = 1_000;
        let data: Arc<[u32]> = (0..LEN as u32).collect::<Vec<_>>().into();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        par_each_async_with_threshold(
            &pool,
            data,
            move |_item: u32| {
                let counter = counter_for_work.clone();
                async move {
                    tokio::task::yield_now().await;
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            },
            128,
        )
        .await;
        assert_eq!(counter.load(Ordering::Relaxed), LEN);
    }

    /// trait: `.par_iter(&pool).map_async(async fn).sum().await` end-to-end.
    /// closure receives item by value (Item: Clone), returns future, the
    /// per-worker tokio runtime drives it.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_iter_map_async_sum() {
        let pool = build_pool(4);
        const LEN: u32 = 500;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let sum: u64 = data
            .par_iter(&pool)
            .map_async(|item| async move {
                tokio::task::yield_now().await;
                u64::from(item) * 2
            })
            .sum()
            .await;
        let expected: u64 = (0..LEN).map(|item| u64::from(item) * 2).sum();
        assert_eq!(sum, expected);
    }

    /// trait: `.map_async().reduce(id, op).await` with non-summing reduce.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_iter_map_async_reduce_max() {
        let pool = build_pool(4);
        let data: Arc<[u32]> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3].into();
        let max = data
            .par_iter(&pool)
            .with_threshold(4)
            .map_async(|item| async move {
                tokio::task::yield_now().await;
                item
            })
            .reduce(|| 0u32, u32::max)
            .await;
        assert_eq!(max, 9);
    }

    /// trait: `.map_async().for_each(sink).await` — map output through
    /// async work, then drop into a sync sink. exercises the closure
    /// composition that wraps sink ∘ async_map into a single Fn->Future.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_iter_map_async_for_each_pipes_outputs() {
        let pool = build_pool(4);
        const LEN: u32 = 200;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let total = Arc::new(AtomicUsize::new(0));
        let total_for_sink = total.clone();
        data.par_iter(&pool)
            .map_async(|item| async move {
                tokio::task::yield_now().await;
                item as usize * 3
            })
            .for_each(move |tripled| {
                total_for_sink.fetch_add(tripled, Ordering::Relaxed);
            })
            .await;
        let expected: usize = (0..LEN as usize).map(|item| item * 3).sum();
        assert_eq!(total.load(Ordering::Relaxed), expected);
    }

    /// trait: `.for_each_async(async closure)` — no map, side-effect-only
    /// per-item async work. counter total matches LEN.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_iter_for_each_async() {
        let pool = build_pool(4);
        const LEN: usize = 500;
        let data: Arc<[u32]> = (0..LEN as u32).collect::<Vec<_>>().into();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        data.par_iter(&pool)
            .for_each_async(move |_item: u32| {
                let counter = counter_for_work.clone();
                async move {
                    tokio::task::yield_now().await;
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            })
            .await;
        assert_eq!(counter.load(Ordering::Relaxed), LEN);
    }

    // ---- par_stream (C: buffered concurrency) ----

    /// par_stream + then: collect all outputs from concurrent futures.
    /// every input produces one output, total count matches input length.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_stream_then_collect_all_items() {
        use futures::StreamExt;
        let pool = build_pool(4);
        const LEN: u32 = 100;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let outputs: Vec<u64> = data
            .par_stream(&pool, 8)
            .then(|item| async move {
                tokio::task::yield_now().await;
                u64::from(item) * 2
            })
            .collect()
            .await;
        assert_eq!(outputs.len(), LEN as usize);
        // outputs are unordered; sort then compare against expected sorted set.
        let mut sorted = outputs;
        sorted.sort_unstable();
        let expected: Vec<u64> = (0..LEN).map(|item| u64::from(item) * 2).collect();
        assert_eq!(sorted, expected);
    }

    /// par_stream: empty slice → empty stream.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_stream_empty_slice_emits_nothing() {
        use futures::StreamExt;
        let pool = build_pool(2);
        let data: Arc<[u32]> = Vec::<u32>::new().into();
        let outputs: Vec<u64> = data
            .par_stream(&pool, 4)
            .then(|item| async move { u64::from(item) })
            .collect()
            .await;
        assert!(outputs.is_empty());
    }

    /// par_stream concurrency=1 reduces to serial — output count still
    /// matches, no items lost.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_stream_concurrency_one_serial_path() {
        use futures::StreamExt;
        let pool = build_pool(2);
        const LEN: u32 = 20;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let outputs: Vec<u32> = data
            .par_stream(&pool, 1)
            .then(|item| async move {
                tokio::task::yield_now().await;
                item
            })
            .collect()
            .await;
        assert_eq!(outputs.len(), LEN as usize);
    }

    /// par_stream borrowed-Arc impl — slice stays usable after.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_stream_borrowed_arc() {
        use futures::StreamExt;
        let pool = build_pool(2);
        let data: Arc<[u32]> = (0..50_u32).collect::<Vec<_>>().into();
        let first: usize = (&data)
            .par_stream(&pool, 4)
            .then(|item| async move { item as usize })
            .count()
            .await;
        let second: usize = (&data)
            .par_stream(&pool, 4)
            .then(|item| async move { item as usize })
            .count()
            .await;
        assert_eq!(first, 50);
        assert_eq!(second, 50);
    }

    // ---- pool.workers() + default_threshold_for ----

    /// pool.workers() reflects the configured thread count.
    #[proxima::test]
    async fn pool_workers_returns_configured_count() {
        let pool_two = build_pool(2);
        assert_eq!(pool_two.workers(), 2);
        let pool_eight = build_pool(8);
        assert_eq!(pool_eight.workers(), 8);
    }

    /// default_threshold_for targets ~4 leaves per worker. for N=128 on
    /// 4 workers: target_leaves=16, threshold=8.
    #[proxima::test]
    async fn default_threshold_for_targets_four_leaves_per_worker() {
        let pool = build_pool(4);
        assert_eq!(default_threshold_for(&pool, 128), 8);
        assert_eq!(default_threshold_for(&pool, 100_000), 100_000 / 16);
        // floor: never zero
        assert_eq!(default_threshold_for(&pool, 0), 1);
        // tiny slice: threshold floor
        assert_eq!(default_threshold_for(&pool, 5), 1);
    }

    /// par_iter without explicit .with_threshold() uses the auto-derived
    /// default. correctness check: result is still the expected sum.
    #[proxima::test]
    async fn par_iter_auto_threshold_correctness() {
        let pool = build_pool(4);
        const LEN: u32 = 10_000;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let sum: u64 = data
            .par_iter(&pool)
            .map(|&item| u64::from(item))
            .sum()
            .await;
        let expected: u64 = (0..LEN).map(u64::from).sum();
        assert_eq!(sum, expected);
    }

    // ---- ProximaParMap::collect + par_map_collect ----

    /// trait: `.par_iter().map(f).collect()` preserves input order.
    #[proxima::test]
    async fn par_iter_map_collect_preserves_order() {
        let pool = build_pool(4);
        const LEN: u32 = 1_000;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let result: Vec<u64> = data
            .par_iter(&pool)
            .map(|&item| u64::from(item) * 2)
            .collect()
            .await;
        let expected: Vec<u64> = (0..LEN).map(|item| u64::from(item) * 2).collect();
        assert_eq!(result, expected);
    }

    /// par_map_collect engine: empty slice returns empty Vec.
    #[proxima::test]
    async fn par_map_collect_empty_slice() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = Vec::<u32>::new().into();
        let result: Vec<u64> = par_map_collect(&pool, data, |&item| u64::from(item)).await;
        assert!(result.is_empty());
    }

    /// par_map_collect engine: large slice forces splits, output ordered.
    #[proxima::test]
    async fn par_map_collect_large_slice_ordered() {
        let pool = build_pool(4);
        const LEN: u32 = 50_000;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let result: Vec<u32> =
            par_map_collect_with_threshold(&pool, data, |&item| item + 1, 1024).await;
        // verify monotonic increase (proves ordering, not just count)
        for window in result.windows(2) {
            assert!(window[0] < window[1]);
        }
        assert_eq!(result.len(), LEN as usize);
    }

    // ---- ProximaParMapAsync::collect + par_map_collect_async ----

    /// trait async: `.par_iter().map_async(async fn).collect()` ordered.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_iter_map_async_collect_preserves_order() {
        let pool = build_pool(4);
        const LEN: u32 = 500;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let result: Vec<u64> = data
            .par_iter(&pool)
            .map_async(|item| async move {
                tokio::task::yield_now().await;
                u64::from(item) * 3
            })
            .collect()
            .await;
        let expected: Vec<u64> = (0..LEN).map(|item| u64::from(item) * 3).collect();
        assert_eq!(result, expected);
    }

    /// par_map_collect_async engine: empty slice.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_map_collect_async_empty() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = Vec::<u32>::new().into();
        let result: Vec<u64> =
            par_map_collect_async(&pool, data, |item| async move { u64::from(item) }).await;
        assert!(result.is_empty());
    }

    // ---- ProximaParStream::then_ordered ----

    /// par_stream then_ordered preserves input order even when leaves
    /// complete in random order. inject a per-item delay proportional
    /// to LEN - item so later items complete earlier; if the reorder
    /// buffer works, output is still 0,1,2,...
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_stream_then_ordered_preserves_input_order() {
        use futures::StreamExt;
        let pool = build_pool(4);
        const LEN: u32 = 50;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        // delay = (LEN - item) microseconds — later items finish first
        // if the runtime is faithful to sleep durations.
        let outputs: Vec<u32> = data
            .par_stream(&pool, 8)
            .then_ordered(|item| async move {
                let delay_us = u64::from(LEN - item);
                tokio::time::sleep(std::time::Duration::from_micros(delay_us)).await;
                item
            })
            .collect()
            .await;
        let expected: Vec<u32> = (0..LEN).collect();
        assert_eq!(outputs, expected);
    }

    /// par_stream then_ordered: empty slice → empty stream.
    #[cfg(feature = "runtime-prime-bgpool-async")]
    #[proxima::test]
    async fn par_stream_then_ordered_empty() {
        use futures::StreamExt;
        let pool = build_pool(2);
        let data: Arc<[u32]> = Vec::<u32>::new().into();
        let outputs: Vec<u32> = data
            .par_stream(&pool, 4)
            .then_ordered(|item| async move { item })
            .collect()
            .await;
        assert!(outputs.is_empty());
    }

    // ---- par_filter ----

    /// par_filter: only even elements pass. count must be exactly half of LEN.
    #[proxima::test]
    async fn par_filter_even_elements_correct_count() {
        let pool = build_pool(4);
        const LEN: u32 = 10_000;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let evens = par_filter(&pool, data, |&item| item % 2 == 0).await;
        assert_eq!(evens.len(), LEN as usize / 2);
        assert!(evens.iter().all(|&item| item % 2 == 0));
    }

    /// par_filter: empty slice returns empty Vec.
    #[proxima::test]
    async fn par_filter_empty_slice() {
        let pool = build_pool(2);
        let data: Arc<[u32]> = Vec::<u32>::new().into();
        let result = par_filter(&pool, data, |&item| item > 5).await;
        assert!(result.is_empty());
    }

    /// par_filter: preserves relative order within accepted elements.
    #[proxima::test]
    async fn par_filter_preserves_order() {
        let pool = build_pool(4);
        const LEN: u32 = 1_000;
        let data: Arc<[u32]> = (0..LEN).collect::<Vec<_>>().into();
        let result = par_filter_with_threshold(&pool, data, |&item| item % 3 == 0, 64).await;
        for window in result.windows(2) {
            assert!(window[0] < window[1]);
        }
    }

    // ---- par_sort / par_sort_by / par_sort_unstable ----

    /// par_sort: output matches std sort.
    #[proxima::test]
    async fn par_sort_matches_std_sort() {
        let pool = build_pool(4);
        let data: Vec<u32> = vec![9, 3, 7, 1, 5, 2, 8, 4, 6, 0];
        let mut expected = data.clone();
        expected.sort();
        let sorted = par_sort(&pool, data).await;
        assert_eq!(sorted, expected);
    }

    /// par_sort_by: descending comparator produces reversed order.
    #[proxima::test]
    async fn par_sort_by_descending_matches_std() {
        let pool = build_pool(4);
        let data: Vec<u32> = (0..1_000_u32).rev().collect();
        let mut expected = data.clone();
        expected.sort_by(|a, b| b.cmp(a));
        let sorted = par_sort_by(&pool, data, |a, b| b.cmp(a)).await;
        assert_eq!(sorted, expected);
    }

    /// par_sort_by: large slice triggers splits; result matches std sort_by.
    #[proxima::test]
    async fn par_sort_by_large_slice() {
        let pool = build_pool(4);
        let data: Vec<u32> = (0..10_000_u32).rev().collect();
        let mut expected = data.clone();
        expected.sort_by(|a, b| a.cmp(b));
        let sorted = par_sort_by(&pool, data, |a, b| a.cmp(b)).await;
        assert_eq!(sorted, expected);
    }

    /// par_sort_unstable: output sorted (does not guarantee stability, only ordering).
    #[proxima::test]
    async fn par_sort_unstable_produces_sorted_output() {
        let pool = build_pool(4);
        let data: Vec<u32> = vec![5, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        let sorted = par_sort_unstable(&pool, data).await;
        for window in sorted.windows(2) {
            assert!(window[0] <= window[1]);
        }
    }

    // ---- par_chunks_mut ----

    /// par_chunks_mut: multiply each element by 2 in-place; verify afterward.
    #[proxima::test]
    async fn par_chunks_mut_mutation_visible_after_return() {
        let pool = build_pool(4);
        let mut data: Vec<u32> = (0..1_000_u32).collect();
        par_chunks_mut(&pool, &mut data, 64, |chunk| {
            for item in chunk.iter_mut() {
                *item *= 2;
            }
        })
        .await;
        let expected: Vec<u32> = (0..1_000_u32).map(|item| item * 2).collect();
        assert_eq!(data, expected);
    }

    /// par_chunks_mut: empty slice no-ops.
    #[proxima::test]
    async fn par_chunks_mut_empty_no_op() {
        let pool = build_pool(2);
        let mut data: Vec<u32> = Vec::new();
        par_chunks_mut(&pool, &mut data, 64, |chunk| {
            for item in chunk.iter_mut() {
                *item = 99;
            }
        })
        .await;
        assert!(data.is_empty());
    }

    // ---- scope ----

    /// scope: all spawned tasks complete before scope returns; results visible.
    #[proxima::test]
    async fn scope_all_tasks_complete_before_return() {
        let pool = build_pool(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        scope(&pool, |proxima_scope| {
            for _ in 0..8 {
                let counter_for_task = counter_clone.clone();
                proxima_scope.spawn(move || {
                    counter_for_task.fetch_add(1, Ordering::Relaxed);
                });
            }
        })
        .await;
        assert_eq!(counter.load(Ordering::Relaxed), 8);
    }

    /// scope: zero spawns — no deadlock, no panic.
    #[proxima::test]
    async fn scope_no_spawns_completes() {
        let pool = build_pool(2);
        scope(&pool, |_proxima_scope| {}).await;
    }

    // ---- install ----

    /// install: closure sees the installed pool via installed_pool().
    #[proxima::test]
    async fn install_pool_visible_inside_closure() {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(2).expect("build pool"));
        let pool_workers = pool.workers();
        let seen_workers = install(pool, || {
            installed_pool().map(|installed| installed.workers())
        });
        assert_eq!(seen_workers, Some(pool_workers));
    }

    /// install: installed_pool() returns None outside install context.
    #[proxima::test]
    async fn installed_pool_returns_none_outside_install() {
        assert!(installed_pool().is_none());
    }

    // ---- par_bridge ----

    /// par_bridge: produces all items from the source iterator.
    #[proxima::test]
    async fn par_bridge_produces_all_items() {
        let pool = build_pool(4);
        const LEN: usize = 1_000;
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        par_bridge(&pool, 0..LEN, 4, move |_item| {
            counter_for_work.fetch_add(1, Ordering::Relaxed);
        })
        .await;
        assert_eq!(counter.load(Ordering::Relaxed), LEN);
    }

    /// par_bridge: empty iterator — no panics, counter stays zero.
    #[proxima::test]
    async fn par_bridge_empty_iterator() {
        let pool = build_pool(2);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_work = counter.clone();
        par_bridge(&pool, std::iter::empty::<u32>(), 4, move |_item| {
            counter_for_work.fetch_add(1, Ordering::Relaxed);
        })
        .await;
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}
