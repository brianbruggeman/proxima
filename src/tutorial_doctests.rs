//! Compile-checks every ```rust fenced block in `docs/tutorials/*.md`.
//!
//! `docs/tutorials/00-foundations.md` states its own contract: "every code
//! block below is copied verbatim from a real file in this repository —
//! either a doctest that `cargo test` compiles, a unit test, or a runnable
//! `examples/*/main.rs`". Nothing enforced that claim: `cargo test --doc`
//! never saw the tutorials at all, so a snippet could go stale the moment
//! the API it cites was renamed and nothing would fail.
//!
//! `#[cfg(doctest)]` items are only visible while rustdoc is extracting
//! doctests FROM this crate — they never link into a normal build, and (this
//! was verified empirically, not assumed) they are also invisible to the
//! doctest's own compiled snippet, which links against an ordinarily-built
//! `proxima` rlib. So `tutorial_gate_prelude` below is deliberately NOT
//! `cfg(doctest)`-gated: it is a real, always-compiled, `#[doc(hidden)]`
//! module of re-exports a fragment excerpted from a real file (which elides
//! its own imports "for space", per the tutorials' own convention) needs to
//! resolve. Each `docs/tutorials/*.md` line 5 words the same way: the
//! visible tutorial text never changes; `scripts/tutorials-gate.sh` writes a
//! transformed copy of each file into `.tutorial-gate-generated/` before
//! this compiles — untagged fences (terminal transcripts) retagged `text` so
//! rustdoc does not try to compile them, and one hidden
//! `# use proxima::tutorial_gate_prelude::*;` line injected after every
//! opening ` ```rust ` fence. Run `scripts/tutorials-gate.sh` before
//! `cargo test --doc`; without it these `include_str!` targets do not exist
//! and the crate fails to compile under `--doc`, which is the correct, loud
//! failure rather than a silently-skipped gate.
//!
//! The module count below must track `docs/tutorials/*.md` 1:1 (minus
//! `README.md`, which carries no code) — `scripts/tutorials-gate.sh` asserts
//! this so a new tutorial file can never land ungated.

#[doc(hidden)]
pub mod tutorial_gate_prelude {
    pub use crate::app::{App, IntoMountTarget, MountTarget, RunConfig};
    pub use crate::listen::ListenerSpec;
    pub use crate::listen::admission::{BlacklistConfig, ConnAdmission};
    pub use crate::listen::any::AnyHandler;
    pub use crate::prelude::*;
    pub use crate::selection::Selection;
    pub use crate::shutdown::ShutdownBarrier;
    pub use crate::upstreams::KvUpstream;
    pub use crate::{Fallthrough, KvCache, KvCaps, KvHandle, SynthUpstream, UpstreamRef, WriteBack};
    pub use crate::{PeerInfo, StreamConnection};
    #[cfg(feature = "runtime-tokio")]
    pub use crate::runtime::TokioPerCoreRuntime;
    pub use crate::runtime::{BackgroundHandle, BackgroundPool, CoreId, Runtime};
    pub use crate::{ProximaError, ProximaResult, Request, RequestBuilder, Response};
    pub use crate::{fanin, fanout, filter, fixture, instrument, main, pipe, piped, span, test};
    pub use bon::Builder;
    pub use bytes::Bytes;
    pub use conflaguration::{Settings, Validate, ValidationMessage};
    pub use futures::executor::block_on;
    pub use proxima_core::signal::Signal;
    pub use proxima_primitives::pipe::demand::{
        AlwaysArmed, AtomicGate, AtomicGateController, Demand,
    };
    pub use proxima_primitives::pipe::ext::PipeExt;
    pub use proxima_primitives::pipe::fan_in::{Exhausted, FanIn, FanInStrategy, Select};
    pub use proxima_primitives::pipe::fanout::FanOut;
    pub use proxima_primitives::pipe::handler::{Handler, PipeHandle, into_handle};
    pub use proxima_primitives::pipe::pipe_factory::{DynPipeFactory, PipeFactory};
    pub use proxima_primitives::pipe::plugin::PluginRegistry;
    pub use proxima_primitives::pipe::primitives::{
        AndThen, Pipe, SendPipe, UnpinPipe, UnpinSendPipe,
    };
    pub use proxima_primitives::pipe::routing::MethodFilter;
    pub use serde::{Deserialize, Serialize};
    pub use serde_json::Value;
    pub use std::cell::{Cell, RefCell};
    pub use std::collections::VecDeque;
    pub use std::convert::Infallible;
    pub use std::fmt::Debug;
    pub use std::future::Future;
    pub use std::net::{Ipv4Addr, SocketAddr};
    pub use std::pin::Pin;
    pub use std::sync::Arc;
    pub use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    pub use std::task::{Context, Poll};
    pub use std::time::Duration;

    #[cfg(all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ))]
    pub use crate::prime::PrimeRuntime;
}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/00-foundations.md")]
mod tutorial_00_foundations {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/01-ergonomics.md")]
mod tutorial_01_ergonomics {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/02-listener-builder.md")]
mod tutorial_02_listener_builder {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/03-native-runtime.md")]
mod tutorial_03_native_runtime {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/04-listener-hello.md")]
mod tutorial_04_listener_hello {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/05-listener-universal.md")]
mod tutorial_05_listener_universal {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/06-listener-production.md")]
mod tutorial_06_listener_production {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/07-sugar-composition.md")]
mod tutorial_07_sugar_composition {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/08-protocol-fleet.md")]
mod tutorial_08_protocol_fleet {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/09-extend-your-own-protocol.md")]
mod tutorial_09_extend_your_own_protocol {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/10-conflaguration.md")]
mod tutorial_10_conflaguration {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/11-any-transport-agnostic.md")]
mod tutorial_11_any_transport_agnostic {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-a-bare-metal-pipe.md")]
mod tutorial_build_a_bare_metal_pipe {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-a-caching-reverse-proxy.md")]
mod tutorial_build_a_caching_reverse_proxy {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-a-chaos-test-rig.md")]
mod tutorial_build_a_chaos_test_rig {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-a-crud-origin-service.md")]
mod tutorial_build_a_crud_origin_service {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-a-kafka-style-partitioner.md")]
mod tutorial_build_a_kafka_style_partitioner {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-a-load-balancer.md")]
mod tutorial_build_a_load_balancer {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-a-multi-runtime-service.md")]
mod tutorial_build_a_multi_runtime_service {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-a-plugin.md")]
mod tutorial_build_a_plugin {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-a-record-replay-harness.md")]
mod tutorial_build_a_record_replay_harness {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-an-api-gateway.md")]
mod tutorial_build_an_api_gateway {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-an-observability-pipeline.md")]
mod tutorial_build_an_observability_pipeline {}

#[cfg(all(doctest, tutorial_gate))]
#[doc = include_str!("../.tutorial-gate-generated/build-delivery-guarantees.md")]
mod tutorial_build_delivery_guarantees {}
