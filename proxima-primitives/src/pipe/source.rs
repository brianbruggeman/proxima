//! [`SourcePipe`] is the background-producer face of the [`SendPipe`] root
//! form: a `Signal -> ()` pipe with `Err` pinned to [`ProximaError`]. Where
//! [`Handler`](crate::pipe::handler::Handler) serves one request and returns one
//! response, a `SourcePipe` runs an unbounded loop that observes a
//! cancellation [`Signal`] cooperatively and returns once it has drained.
//!
//! `IntervalPipe` and `ScheduledTriggerPipe` are the canonical implementors:
//! each wraps a `run_*_loop(period, inner, factory, cancel)` free function
//! that already takes a `Signal` and only returns when the signal fires.
//! `SourcePipe` is a blanket impl over the right `SendPipe` shape — nothing
//! to implement directly.
//!
//! [`SourceHandle`] is the erased, shareable form [`App::source`](crate)
//! stores; [`ProducerLifecycle`](crate::pipe::lifecycle::ProducerLifecycle) drives
//! every registered source cooperatively via `spawn_from_source`.

use serde_json::Value;

use proxima_core::ProximaError;
use proxima_core::factory::Named;
use proxima_core::signal::Signal;

use crate::pipe::alloc_tier;
use crate::pipe::primitives::SendPipe;

/// The background-producer face of [`SendPipe`]: `Signal -> ()`, `Err`
/// pinned to [`ProximaError`]. Blanket-implemented for every qualifying
/// `SendPipe` — nothing to implement directly.
pub trait SourcePipe: SendPipe<In = Signal, Out = (), Err = ProximaError> {}

impl<Implementor> SourcePipe for Implementor where
    Implementor: SendPipe<In = Signal, Out = (), Err = ProximaError>
{
}

/// Erased, shareable source handle — the runtime-dispatch alias
/// `Arc<dyn SendDynPipe<Signal, ()>>` that `App::source` stores.
pub type SourceHandle = alloc_tier::PipeHandle<Signal, ()>;

/// Erase a [`SourcePipe`] into a shareable [`SourceHandle`].
pub fn into_source_handle<Implementor>(pipe: Implementor) -> SourceHandle
where
    Implementor: SendPipe<In = Signal, Out = (), Err = ProximaError> + 'static,
{
    alloc_tier::into_handle(pipe)
}

/// Config-driven source factory: the [`SourcePipe`] analogue of
/// [`PipeFactory`](crate::pipe::pipe_factory::PipeFactory). `IntervalPipe` and
/// `ScheduledTriggerPipe` each register a thin `SourceFactory` so
/// `App::apply_settings`'s producer-graph-config step resolves a TOML row
/// straight into a [`SourceHandle`] via the shared
/// [`FactoryRegistry`](proxima_core::factory::FactoryRegistry) primitive.
pub trait SourceFactory: Named {
    /// Build a source from a config spec.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the spec fails to parse or lower into a source.
    fn build(&self, spec: &Value) -> Result<SourceHandle, ProximaError>;
}

/// The source registry — [`proxima_core::FactoryRegistry`] specialized to
/// `dyn SourceFactory`, mirroring
/// [`PipeFactoryRegistry`](crate::pipe::pipe_factory::PipeFactoryRegistry).
pub type SourceFactoryRegistry = proxima_core::FactoryRegistry<dyn SourceFactory>;

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use core::future::Future;
    use core::sync::atomic::{AtomicUsize, Ordering};

    struct CountingSource {
        calls: alloc::sync::Arc<AtomicUsize>,
    }

    impl SendPipe for CountingSource {
        type In = Signal;
        type Out = ();
        type Err = ProximaError;

        fn call(&self, _cancel: Signal) -> impl Future<Output = Result<(), ProximaError>> + Send {
            self.calls.fetch_add(1, Ordering::SeqCst);
            async move { Ok(()) }
        }
    }

    #[proxima::test]
    async fn source_handle_dispatches_to_inner() {
        let calls = alloc::sync::Arc::new(AtomicUsize::new(0));
        let handle: SourceHandle = into_source_handle(CountingSource {
            calls: calls.clone(),
        });
        let signal = Signal::new();
        SendPipe::call(&handle, signal).await.expect("call");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
