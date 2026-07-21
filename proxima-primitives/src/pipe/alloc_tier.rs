//! Alloc tier: the erased form. Heterogeneous pipes lose their static type and
//! sit behind a trait object so routing tables, swap cells, and registries can
//! hold them side by side.
//!
//! This is the boundary where `ProximaError` lives: a typed subgraph converts
//! its `Err` into `ProximaError` exactly here, via `ProximaError: From<P::Err>`
//! at the erasure blanket. The typed tiers above never name the god-error.
//!
//! Two erased forms mirror the two root forms:
//! - [`DynPipe`] / [`LocalPipeHandle`] erase the no-Send [`Pipe`] (`Rc`).
//! - [`SendDynPipe`] / [`PipeHandle`] erase the cross-core [`SendPipe`] (`Arc`).
//!
//! Needs `alloc` (boxed futures + `Rc`/`Arc`), nothing from `std`.

use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::sync::Arc;
use core::future::Future;
use core::pin::Pin;

use proxima_core::ProximaError;

use crate::pipe::primitives::{Pipe, SendPipe};

/// Boxed `!Send` future — the erased return of [`DynPipe::call_dyn`].
pub type BoxFuture<'a, Output> = Pin<Box<dyn Future<Output = Output> + 'a>>;

/// Boxed `Send` future — the erased return of [`SendDynPipe::call_dyn`].
pub type SendBoxFuture<'a, Output> = Pin<Box<dyn Future<Output = Output> + Send + 'a>>;

/// Object-safe erasure of the no-Send root [`Pipe`]. The associated future and
/// `Err` are gone: the future is boxed, the error is pinned to [`ProximaError`].
/// This is what a `dyn` handle holds.
pub trait DynPipe<In, Out> {
    /// Erased dispatch — boxes the inner pipe's future and converts its error
    /// into [`ProximaError`].
    fn call_dyn(&self, input: In) -> BoxFuture<'_, Result<Out, ProximaError>>;
}

impl<P> DynPipe<P::In, P::Out> for P
where
    P: Pipe,
    ProximaError: From<P::Err>,
{
    fn call_dyn(&self, input: P::In) -> BoxFuture<'_, Result<P::Out, ProximaError>> {
        Box::pin(async move { self.call(input).await.map_err(ProximaError::from) })
    }
}

/// Erased, shareable no-Send pipe — `Rc` because the root form is per-core.
pub type LocalPipeHandle<In, Out> = Rc<dyn DynPipe<In, Out>>;

/// Erase a no-Send [`Pipe`] into a shareable [`LocalPipeHandle`].
pub fn into_local_handle<P>(pipe: P) -> LocalPipeHandle<P::In, P::Out>
where
    P: Pipe + 'static,
    ProximaError: From<P::Err>,
{
    Rc::new(pipe)
}

/// Object-safe erasure of the cross-core [`SendPipe`]. The future is boxed and
/// `Send`; the error is pinned to [`ProximaError`]. `Send + Sync` so the handle
/// crosses cores.
pub trait SendDynPipe<In, Out>: Send + Sync {
    /// Erased dispatch — boxes the inner pipe's `Send` future and converts its
    /// error into [`ProximaError`].
    fn call_dyn(&self, input: In) -> SendBoxFuture<'_, Result<Out, ProximaError>>;
}

impl<P> SendDynPipe<P::In, P::Out> for P
where
    P: SendPipe,
    P::In: Send,
    ProximaError: From<P::Err>,
{
    fn call_dyn(&self, input: P::In) -> SendBoxFuture<'_, Result<P::Out, ProximaError>> {
        Box::pin(async move { self.call(input).await.map_err(ProximaError::from) })
    }
}

/// Erased, shareable cross-core pipe — `Arc` so it dispatches across cores.
pub type PipeHandle<In, Out> = Arc<dyn SendDynPipe<In, Out>>;

/// Erase a cross-core [`SendPipe`] into a shareable [`PipeHandle`].
pub fn into_handle<P>(pipe: P) -> PipeHandle<P::In, P::Out>
where
    P: SendPipe,
    P::In: Send,
    ProximaError: From<P::Err>,
{
    Arc::new(pipe)
}

// Reflexive re-entry: an erased handle IS itself a pipe of the same form, so
// it composes wherever a `SendPipe`/`Pipe` is expected (e.g. as an `AndThen`
// stage, or fed straight back through the `Arc`/`Rc` forwarding impls below).
// Lives in the leaf (it owns both the erasure traits and the root forms) so
// this bridge is written ONCE instead of once per domain crate's own
// hand-rolled `impl SendPipe for dyn XyzDynPipe` copy.
impl<In: 'static, Out: 'static> SendPipe for dyn SendDynPipe<In, Out> {
    type In = In;
    type Out = Out;
    type Err = ProximaError;

    fn call(&self, input: In) -> impl Future<Output = Result<Out, ProximaError>> + Send {
        self.call_dyn(input)
    }
}

// The base-tier mirror of the impl above: every `SendPipe` is also required
// to be usable as a plain `Pipe` (the additive tiers never replace the root
// form), so `PipeHandle` (`Arc<dyn SendDynPipe<..>>`) reaches `PipeExt`'s
// `.and_then`/`.filter` sugar the same as any other pipe.
impl<In: 'static, Out: 'static> Pipe for dyn SendDynPipe<In, Out> {
    type In = In;
    type Out = Out;
    type Err = ProximaError;

    fn call(&self, input: In) -> impl Future<Output = Result<Out, ProximaError>> {
        self.call_dyn(input)
    }
}

impl<In, Out> Pipe for dyn DynPipe<In, Out> {
    type In = In;
    type Out = Out;
    type Err = ProximaError;

    fn call(&self, input: In) -> impl Future<Output = Result<Out, ProximaError>> {
        self.call_dyn(input)
    }
}

// Shared-ownership forwarding: an `Arc`/`Rc` of a pipe IS a pipe. Lives in the
// leaf (it owns the traits) so the cutover's `Arc<P>: Pipe` headers resolve here
// once, not at every consumer. `Arc` carries the cross-core [`SendPipe`]; `Rc`
// the per-core [`Pipe`].
impl<Inner> SendPipe for Arc<Inner>
where
    Inner: SendPipe + ?Sized,
{
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = Inner::Err;

    fn call(
        &self,
        input: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, Self::Err>> + Send {
        Inner::call(self, input)
    }
}

impl<Inner> Pipe for Arc<Inner>
where
    Inner: Pipe + ?Sized,
{
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = Inner::Err;

    fn call(
        &self,
        input: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, Self::Err>> {
        Inner::call(self, input)
    }
}

impl<Inner> Pipe for Rc<Inner>
where
    Inner: Pipe + ?Sized,
{
    type In = Inner::In;
    type Out = Inner::Out;
    type Err = Inner::Err;

    fn call(
        &self,
        input: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, Self::Err>> {
        Inner::call(self, input)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::{SendDynPipe, into_handle, into_local_handle};
    use crate::pipe::primitives::{Pipe, SendPipe};
    use core::future::Future;

    fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
        let mut pinned = core::pin::pin!(future);
        let mut context = core::task::Context::from_waker(core::task::Waker::noop());
        loop {
            if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
                return output;
            }
        }
    }

    // Err pinned to ProximaError at the erasure boundary, so the probe uses it
    // directly (reflexive From<ProximaError>); typed-error pipes supply their
    // own `impl From<E> for ProximaError`.
    struct Tagger(u64);

    impl Pipe for Tagger {
        type In = u64;
        type Out = u64;
        type Err = proxima_core::ProximaError;

        fn call(
            &self,
            input: u64,
        ) -> impl Future<Output = Result<u64, proxima_core::ProximaError>> {
            let tag = self.0;
            async move { Ok(input + tag) }
        }
    }

    impl SendPipe for Tagger {
        type In = u64;
        type Out = u64;
        type Err = proxima_core::ProximaError;

        fn call(
            &self,
            input: u64,
        ) -> impl Future<Output = Result<u64, proxima_core::ProximaError>> + Send {
            let tag = self.0;
            async move { Ok(input + tag) }
        }
    }

    #[test]
    fn local_handle_erases_a_no_send_pipe() {
        let handle = into_local_handle(Tagger(10));
        let out = block_on(handle.call_dyn(5)).expect("erased local dispatch");
        assert_eq!(out, 15);
    }

    #[test]
    fn send_handle_erases_a_cross_core_pipe() {
        let handle = into_handle(Tagger(100));
        let out =
            block_on(SendDynPipe::call_dyn(handle.as_ref(), 5)).expect("erased send dispatch");
        assert_eq!(out, 105);
    }

    fn assert_send_sync<Type: Send + Sync>(_: &Type) {}

    #[test]
    fn send_handle_is_send_and_sync() {
        let handle = into_handle(Tagger(1));
        assert_send_sync(&handle);
    }
}
