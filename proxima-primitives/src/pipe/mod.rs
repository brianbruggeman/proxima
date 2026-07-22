//! Plugin surface for proxima.
//!
//! This module carries everything plugin authors need to implement a
//! [`Handler`](handler::Handler): request/response envelopes, the root
//! [`Pipe`](primitives::Pipe) / [`SendPipe`](primitives::SendPipe) forms,
//! [`PipeFactory`](pipe_factory::PipeFactory), the
//! [`TelemetryHandle`](telemetry_surface::TelemetryHandle) trait, the
//! [`CaptureContext`](capture_surface::CaptureContext) trait, the
//! upgrade primitives, and endpoint metadata value types.
//!
//! Concrete impls of the trait surfaces (recording sinks, Metrics with
//! dashmap+hdrhistogram, the per-core runtime) live downstream in
//! `proxima-telemetry`, `proxima-recording`, `proxima-runtime*`, etc.

/// Build-time sizing consts (heapless caps) baked from `proxima-primitives.toml`
/// by `build.rs`. Only the no-alloc tier reads `RETRY_STATUS_CAP`; under
/// `proxima_alloc` the unbounded `BTreeSet` form needs no cap.
#[cfg(not(proxima_alloc))]
mod sized {
    include!(concat!(env!("OUT_DIR"), "/proxima_primitives_pipe_sized.rs"));
}

#[cfg(feature = "std")]
pub mod batch;
#[cfg(feature = "alloc")]
pub mod body;
#[cfg(feature = "alloc")]
pub mod bounded;
#[cfg(feature = "std")]
pub mod bucket_table;
pub mod capture_surface;
#[cfg(feature = "alloc")]
pub mod chaos;
pub mod primitives;
#[cfg(feature = "alloc")]
pub mod alloc_tier;
pub mod batch_source;
pub mod capabilities;
#[cfg(feature = "alloc")]
pub mod delay;
// `Diff::call` used to require `tokio::join!` directly (predates the
// pipe/sync primitives merge); it's now `futures::join!` (runtime-agnostic),
// but the module stays `not(loom)` for parity with its sibling `lifecycle`
// — untested under the loom model, not because it needs tokio anymore.
#[cfg(all(feature = "std", not(loom)))]
pub mod diff;
#[cfg(feature = "alloc")]
pub mod demand;
#[cfg(feature = "alloc")]
pub mod endpoint;
#[cfg(feature = "alloc")]
pub mod fanout;
#[cfg(feature = "alloc")]
pub mod filter;
#[cfg(feature = "std")]
pub mod filter_registry;
#[cfg(feature = "std")]
pub mod fanout_registry;
pub mod drain_sink;
pub mod drain_source;
pub mod ext;
pub mod fan_in;
pub mod stream_bridge;
#[cfg(feature = "alloc")]
pub mod header_list;
#[cfg(feature = "alloc")]
pub mod handler;
#[cfg(feature = "std")]
pub mod interval_pipe;
#[cfg(feature = "std")]
pub mod isolate;
#[cfg(feature = "std")]
pub mod keyed_live_filter;
#[cfg(feature = "alloc")]
pub mod labeled;
// `not(loom)`: `ProducerLifecycle` wraps a real `tokio::task::JoinSet`
// directly (predates the pipe/sync primitives merge); the loom build keeps
// tokio out of the graph entirely (see `[target.'cfg(not(loom))'.dependencies]`
// in Cargo.toml), so this module is unavailable there — it is unrelated to
// the sync loom tests.
#[cfg(all(feature = "std", not(loom)))]
pub mod lifecycle;
#[cfg(feature = "std")]
pub mod live_filter;
#[cfg(feature = "alloc")]
pub mod mutate;
#[cfg(feature = "alloc")]
pub mod path_pattern;
#[cfg(feature = "alloc")]
pub mod pipe_factory;
#[cfg(feature = "fan-concurrent")]
pub mod race;
pub mod resilience;
pub mod retry_rules;
#[cfg(feature = "fan-concurrent")]
pub mod scatter_gather;
#[cfg(feature = "std")]
pub mod signal_source;
#[cfg(feature = "alloc")]
pub mod sink;
pub mod sink_front;
// production `Clock` for `Retry`, backed by proxima-time.
#[cfg(feature = "alloc")]
pub mod clock;
pub mod header_name;
pub mod method;
#[cfg(feature = "part-source")]
pub mod part;
#[cfg(feature = "alloc")]
pub mod plugin;
#[cfg(feature = "alloc")]
pub mod quiesce;
#[cfg(feature = "std")]
pub mod rate_limit;
#[cfg(feature = "alloc")]
pub mod request;
#[cfg(feature = "alloc")]
pub mod retry;
#[cfg(feature = "std")]
pub mod routing;
#[cfg(feature = "std")]
pub mod source;
#[cfg(feature = "std")]
pub mod swap_registry;
#[cfg(feature = "alloc")]
pub mod swap_surface;
#[cfg(feature = "alloc")]
pub mod telemetry_surface;
#[cfg(feature = "alloc")]
pub mod transform;
pub mod upgrade;
#[cfg(feature = "alloc")]
pub mod validate;
pub mod when;

pub use proxima_core::{ProximaError, ProximaResult};

// Top-level re-exports matching the existing `proxima` umbrella's plugin-facing
// surface. Lets plugin crates write `use proxima_primitives::pipe::{Pipe, Request,
// Body, ...}` without deeper nested module paths.
#[cfg(feature = "alloc")]
pub use body::{ChunkStream, RequestStream, ResponseStream};
#[cfg(feature = "alloc")]
pub use header_list::HeaderList;
pub use header_name::HeaderName;
#[cfg(feature = "std")]
pub use interval_pipe::{
    DEFAULT_TICK_METHOD, DEFAULT_TICK_PATH, IntervalBuildError, IntervalPipe, IntervalPipeBuilder,
    RequestFactory,
};
#[cfg(all(feature = "std", not(loom)))]
pub use lifecycle::{ProducerLifecycle, ShutdownReport};
pub use method::Method;
#[cfg(feature = "part-source")]
pub use part::{Part, PartSink, PartSource};
#[cfg(feature = "alloc")]
pub use handler::{
    Handler, PipeHandle, ThreadLocalHandler, ThreadLocalPipeHandle, into_handle,
    into_thread_local_handle,
};
#[cfg(feature = "alloc")]
pub use pipe_factory::PipeFactory;
#[cfg(feature = "alloc")]
pub use request::{Request, RequestBuilder, RequestContext, Response};
#[cfg(feature = "std")]
pub use source::{SourceFactory, SourceFactoryRegistry, SourceHandle, SourcePipe, into_source_handle};
#[cfg(feature = "alloc")]
pub use swap_surface::{StreamFramer, SwapSurface, Turn};

// The root form family — no-Send [`Pipe`], the additive `Send` constraint
// [`SendPipe`], and [`AndThen`] composition. `Handler` (above) and
// `SourcePipe` are the served-HTTP and background-producer instantiations;
// everything else in this crate composes over the root forms directly.
pub use primitives::{AndThen, Pipe, SendPipe, UnpinPipe, UnpinSendPipe};
// Fluent combinator sugar over the root form — `.and_then`/`.filter`/
// `.fanout`/`.fanin` at the call site instead of the bare constructors. One
// blanket impl over `Pipe` reaches every pipe (see `pipe::ext`'s module doc);
// the resulting combinator values still carry whatever higher tiers their own
// stages qualify for.
pub use ext::PipeExt;
#[cfg(feature = "alloc")]
pub use alloc_tier::{
    BoxFuture, DynPipe, LocalPipeHandle, SendBoxFuture, SendDynPipe, into_local_handle,
};
pub use batch_source::BatchSource;
// ApplyOps/BytePayload/CheckOutcome/Checkable/ExceededAction/KeyOf are the
// same canonical items already re-exported at the crate root via
// transform::/mutate::/validate::/rate_limit:: (each of those modules
// itself `pub use crate::pipe::capabilities::{..}`) — re-exporting them a second time
// here would just be the identical item under a second name, so only the
// capability traits with no other root-level path are added.
pub use capabilities::{Idempotent, Replayable, Retryable};
pub use drain_sink::{DrainFanOut, DrainSink, RingSink};
pub use drain_source::{DrainFanIn, DrainSource, DrainState, RingSource};
pub use fan_in::{Exhausted, FanIn, FanInStrategy, Select};
pub use stream_bridge::{AsSink, AsSinkError, AsStream, DrainSinkExt, PollSourceExt};
#[cfg(feature = "io-bridge")]
pub use stream_bridge::{IntoReader, IntoWriter};
#[cfg(feature = "io-async")]
pub use drain_sink::RingSinkWriteError;
#[cfg(feature = "std")]
pub use filter_registry::{FilterRegistry, FilterRegistryConfig};
#[cfg(feature = "std")]
pub use fanout_registry::{KeyedFanOut, SubscriptionId};
#[cfg(feature = "std")]
pub use live_filter::{FilterControl, FilterUpdate, IdSet, LiveFilter, live_filter, live_filter_ids};
pub use resilience::{Backoff, CircuitBreaker, CircuitState, Deadline, Jitter, RetryAction, RetryController};
#[cfg(feature = "alloc")]
pub use resilience::Fallback;
pub use retry_rules::RetryRules;
#[cfg(feature = "std")]
pub use signal_source::SignalSource;
// sink_front::SinkFront is the generic ring-backed engine struct, distinct from
// sink::SinkFront (the alloc-tier Arc-shared facade already at the crate root)
// — only the two concrete instantiations are re-exported here to avoid the name
// collision. Admission/DropReason/SinkCounters/SinkLifecycle are the same
// canonical items sink:: already re-exports.
pub use sink_front::StaticSinkFront;
#[cfg(feature = "alloc")]
pub use sink_front::HeapSinkFront;
pub use when::When;

#[cfg(feature = "std")]
pub use batch::Batch;
#[cfg(feature = "alloc")]
pub use bounded::{BoundedQueue, EnqueueOutcome, FailMode};
#[cfg(feature = "alloc")]
pub use chaos::{ChaosBuilder, ChaosConfig, LatencyFault, chaos};
#[cfg(all(feature = "std", not(loom)))]
pub use diff::{Diff, diff_handle};
#[cfg(feature = "std")]
pub use isolate::{Isolate, IsolateFactory};
#[cfg(feature = "std")]
pub use routing::{HostFilter, MethodFilter, Mount, Router, RoutingPipe};
#[cfg(feature = "std")]
pub use swap_registry::{SwapRegistry, SwappablePipe};
#[cfg(feature = "std")]
pub use delay::DelayFactory;
#[cfg(feature = "alloc")]
pub use delay::{Delay, DelayConfig, Dist};
#[cfg(feature = "alloc")]
pub use demand::{AlwaysArmed, AtomicGate, AtomicGateController, Demand, DemandGate};
#[cfg(feature = "alloc")]
pub use fanout::{AllOrNothing, BestEffort, FanOut, FanPolicy, IgnoreErrors};
#[cfg(feature = "std")]
pub use filter::FilterFactory;
#[cfg(feature = "alloc")]
pub use filter::{FilterConfig, Predicate, RejectMode};
#[cfg(feature = "std")]
pub use keyed_live_filter::{KeyedLiveFilter, keyed_live_filter_ids};
#[cfg(feature = "alloc")]
pub use labeled::Labeled;
#[cfg(feature = "alloc")]
pub use mutate::{BytePayload, MutateOp, Mutation};
#[cfg(feature = "fan-concurrent")]
pub use race::{Race, RaceBuildError};
#[cfg(feature = "std")]
pub use rate_limit::{
    ExceededAction, KeyExtractor, KeyOf, RateLimit, RateLimitCaps, RateLimitFactory,
    TokenBucketConfig,
};
#[cfg(feature = "std")]
pub use retry::RetryFactory;
#[cfg(feature = "alloc")]
pub use retry::{DeliveryOutcome, Retry, RetryBudget, RetryPredicate};
#[cfg(feature = "fan-concurrent")]
pub use scatter_gather::ScatterGather;
#[cfg(feature = "alloc")]
pub use sink::{Admission, DropReason, SinkCounters, SinkFront, SinkLifecycle};
#[cfg(feature = "alloc")]
pub use transform::{ApplyOps, Transform};
#[cfg(feature = "std")]
pub use transform::{RequestOp, ResponseOp, TransformFactory};
#[cfg(feature = "alloc")]
pub use validate::{CheckOutcome, Checkable, Validate};
#[cfg(feature = "std")]
pub use validate::{ValidateFactory, ValidateOp};

/// The algebra's central claim, asserted by the COMPILER instead of by grep.
///
/// "Everything is a pipe" is falsifiable, so falsify it mechanically: each line
/// below fails to compile the moment a primitive we teach stops being a pipe.
/// A grep for `impl .* Pipe for X` cannot answer this — it cannot see through
/// generics, re-exports, macros, or a renamed type parameter, and it has been
/// wrong every time it was asked. rustc is never wrong about it.
#[cfg(test)]
#[allow(dead_code)] // these exist to be TYPE-CHECKED, never called
mod algebra_claims {
    use super::fanout::FanPolicy;
    use super::primitives::{Pipe, SendPipe};

    fn assert_pipe<P: Pipe>() {}
    fn assert_send_pipe<P: SendPipe>() {}

    // fan-out IS a pipe, for any sink and any fan policy.
    fn _fan_out_is_a_pipe<S, Policy>()
    where
        S: SendPipe<Out = ()> + Clone,
        S::In: Clone + Send,
        S::Err: Send,
        Policy: FanPolicy,
    {
        assert_send_pipe::<super::fanout::FanOut<S, Policy>>();
    }

    // the chain of two pipes is itself a pipe — the composition law, checked.
    fn _a_chain_is_a_pipe<First, Second>()
    where
        First: Pipe,
        Second: Pipe<In = First::Out>,
        Second::Err: From<First::Err>,
    {
        assert_pipe::<super::primitives::AndThen<First, Second>>();
    }

    // fan-in IS a pipe, for any DropSafe UnpinPipe source.
    fn _fan_in_is_a_pipe<S, Strategy, const N: usize>()
    where
        S: super::primitives::UnpinPipe<In = (), Err = super::fan_in::Exhausted>
            + proxima_core::markers::DropSafe,
        Strategy: super::fan_in::FanInStrategy,
    {
        assert_pipe::<super::fan_in::FanIn<S, Strategy, N>>();
    }
}
