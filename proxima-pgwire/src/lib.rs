//! PostgreSQL wire protocol facade — both halves of the connection.
//!
//! Composes the sans-IO [`proxima_protocols::pgwire_codec`] (message codec +
//! session FSM — see its docs for the wire layer) with the workspace's
//! one primitive, `proxima_primitives::pipe::Pipe`. A SQL engine is a `Pipe`:
//! it is `QueryRequest -> PgReply` (payload-no-cell — no `Request`/
//! `Response` envelope), matching on [`pipe_contract::QueryRequest`]'s
//! `verb` field. The driver owns wire framing and the text/binary encoding of
//! [`pipe_contract::SqlValue`], so the engine stays wire-agnostic — and
//! every proxima middleware (`Auth`, `RateLimit`, `Retry`, `Tee`, `Diff`,
//! record/replay, `RoutingPipe`) composes onto SQL with zero new code.
//!
//! Server side:
//!
//! - [`pipe_contract`] — the self-describing request enum + typed payloads
//!   a SQL `Pipe` exchanges
//! - [`connection`] — the runtime-agnostic per-connection driver over any
//!   `futures::io` stream; usable directly from prime, tests, or a bare
//!   event loop (`--no-default-features` keeps tokio out of the
//!   dependency graph entirely)
//! - [`pipe`] (feature `listen`) — [`pipe::PgWireConnectionPipe`], the
//!   connection layer as a `Pipe` whose `call` returns the upgrade that
//!   runs the session loop
//! - [`any_protocol`] / [`listen`] (feature `listen`, default) — the two
//!   mounts: a candidate for the universal listener's classifier, and a
//!   standalone `ListenProtocol` over the runtime-matched acceptor factory.
//!   Both answer SSLRequest through `proxima-tls`
//! - [`auth`] / [`config`] / [`store`] — trust / cleartext / MD5 / SCRAM
//!   policies, the conflaguration + bon config mirror, and the
//!   per-connection statement/portal slots
//!
//! Client side ([`client`], feature `scram`): a sans-IO
//! [`client::ClientSession`] mirroring the server FSM, driven either by the
//! blocking [`client::PgClient`] or — under feature `client` — by
//! `client::pipe::PgwireClientUpstream`, the `Pipe` that lets
//! `proxima::Client` speak pgwire as a registered protocol.
//!
//! `scripts/proxima-pgwire-gate.sh` (run by `.github/workflows/proxima-pgwire.yml`)
//! is the proof mechanism: the bare-metal codec build, the feature-matrix
//! clippy sweep, the real-PostgreSQL differential, and the invariant that
//! `--no-default-features` carries zero tokio.

#[cfg(feature = "listen")]
pub mod any_protocol;
pub mod auth;
pub mod broker;
#[cfg(feature = "scram")]
pub mod client;
pub mod config;
pub mod connection;
pub mod error;
pub mod handler;
#[cfg(feature = "listen")]
pub mod listen;
#[cfg(feature = "md5-auth")]
pub mod md5;
#[cfg(feature = "listen")]
pub mod pipe;
pub mod pipe_contract;
pub mod pipes;
#[cfg(feature = "scram")]
pub mod scram;
#[cfg(feature = "listen")]
mod spec;
pub mod store;

pub use proxima_protocols::pgwire_codec as codec;

// everything a SQL engine names — `impl SendPipe { type Err = ProximaError }`
// plus the handle it is erased into — re-exported so an engine author imports
// from proxima-pgwire alone and never reaches past it into proxima-primitives
// internals (teaching surface, principle 2)
pub use pipes::{PgPipeHandle, into_pg_handle};
pub use proxima_core::ProximaError;
pub use proxima_primitives::pipe::SendPipe;

#[cfg(feature = "listen")]
pub use any_protocol::PgWireAnyProtocol;
pub use auth::{PasswordVerifier, PgAuth, StaticCredentials};
pub use broker::{Notification, NotifyBroker};
#[cfg(feature = "client")]
pub use client::PgwireClientUpstream;
#[cfg(feature = "scram")]
pub use client::{
    ClientError, ClientSession, Column, ConfigError, PgClient, PgClientConfig, QueryResult, Step,
};
pub use config::{AuthConfig, PgServerConfig};
pub use connection::{
    BackendKey, CancelRegistry, Negotiation, RuntimeHandle, negotiate, serve_session,
};
pub use error::ServeError;
pub use handler::ErrorInfo;
#[cfg(feature = "listen")]
pub use listen::PgWireListenProtocol;
#[cfg(feature = "listen")]
pub use pipe::PgWireConnectionPipe;
pub use pipe_contract::{
    CancelToken, ColumnDesc, DescribeReply, ErrorReply, NoticeReply, PgReply, QueryReply,
    QueryRequest, RowStream, SqlValue, TxStatus, Verb, verb,
};
#[cfg(feature = "scram")]
pub use scram::ScramClient;
