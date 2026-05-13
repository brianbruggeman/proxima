//! PostgreSQL wire protocol server facade.
//!
//! Composes the sans-IO [`proxima_protocols::pgwire_codec`] (message codec +
//! session FSM — see its docs for the wire layer) with the workspace's
//! one primitive, `proxima_primitives::pipe::Pipe`. A SQL engine is a `Pipe`: it
//! matches on [`pipe_contract::verb`] verbs and returns a typed
//! [`pipe_contract::PgReply`] through `Carry`. The driver owns wire
//! framing and the text/binary encoding of [`pipe_contract::SqlValue`],
//! so the engine stays wire-agnostic — and every proxima middleware
//! (`Auth`, `RateLimit`, `Retry`, `Tee`, `Diff`, record/replay,
//! `RoutingPipe`) composes onto SQL with zero new code.
//!
//! - [`pipe_contract`] — the verb vocabulary + typed payloads a SQL
//!   `Pipe` exchanges
//! - [`connection`] — the runtime-agnostic per-connection driver over any
//!   `futures::io` stream; usable directly from prime, tests, or a bare
//!   event loop (`--no-default-features` keeps tokio out of the
//!   dependency graph entirely)
//! - [`pipe`] (feature `listen`) — [`pipe::PgWireConnectionPipe`], the
//!   connection layer as a `Pipe` whose `call` returns the upgrade that
//!   runs the session loop
//! - [`listen`] (feature `listen`, default) — `PgWireListenProtocol`
//!   mounting into `proxima-listen`'s registry over the runtime-matched
//!   acceptor factory, with SSLRequest TLS upgrades via `proxima-tls`
//! - [`auth`] / [`config`] / [`store`] — authentication policies, the
//!   conflaguration + bon config mirror, and the per-connection
//!   statement/portal slots
//!
//! The remaining staged surfaces are sequenced as named gates in
//! `docs/proxima-pgwire/discipline.md` (G8 CI/baseline substrate, G11
//! stream-listener upgrade-honor).

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
pub mod store;

pub use proxima_protocols::pgwire_codec as codec;

// the Handler surface a SQL engine builds against — re-exported so an engine
// author imports everything from proxima-pgwire and never reaches past it
// into proxima-pipe / proxima-core internals (teaching surface, principle 2)
pub use pipes::{PgPipeHandle, PgRequest, PgResponse, into_pg_handle};
pub use proxima_core::ProximaError;
pub use proxima_primitives::pipe::SendPipe;
pub use proxima_primitives::pipe::handler::{Handler, PipeHandle, into_handle};
pub use proxima_primitives::pipe::request::{Request, Response};

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
    QueryRequest, RowStream, SqlValue, TxStatus, verb,
};
#[cfg(feature = "scram")]
pub use scram::ScramClient;
