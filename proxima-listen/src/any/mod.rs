//! The open universal listener's classification primitives: an open,
//! registry-driven set of [`AnyProtocol`] candidates, arbitrated per
//! connection by [`Classifier`]. `std`-tier only — every piece here needs
//! [`proxima_primitives::stream::StreamConnection`],
//! [`proxima_primitives::pipe::handler::PipeHandle`], and `serde_json::Value`,
//! none of which exist below `std`.
//!
//! This module is scaffolding for `Listener::any()`
//! (`proxima-listen/src/handle.rs`): the trait, the registry, and the
//! per-connection classifier state machine. The accept loop that drives
//! them — reading bytes, calling [`Classifier::advance`], and replaying the
//! accumulated prefix into the matched candidate's
//! [`AnyProtocol::drive`] — lives in `proxima-http` (`any_listener` module),
//! since the two shipped candidates (h1, h2 prior-knowledge) need
//! `proxima-http`'s own connection drivers.

mod classifier;
mod deny;
#[cfg(feature = "framed-any")]
mod framed_any;
mod probe;
mod registry;

pub use classifier::{Classifier, ClassifyOutcome};
pub use deny::DenySignature;
#[cfg(feature = "framed-any")]
pub use framed_any::{AsFrame, FramedAny};
pub use probe::{
    AnyHandler, AnyProtocol, ProbeVerdict, RejectReason, downcast_handler, erase_handler,
};
pub use registry::AnyRegistry;
