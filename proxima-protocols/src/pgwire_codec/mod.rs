//! Sans-IO PostgreSQL wire protocol (v3.x) codec and session FSM.
//!
//! Message shapes follow the PostgreSQL protocol message formats
//! reference (`https://www.postgresql.org/docs/current/protocol-message-formats.html`)
//! and the message flow chapter
//! (`https://www.postgresql.org/docs/current/protocol-flow.html`). Every
//! frontend and backend message the protocol documents is covered in
//! both directions, including the ones common implementations skip
//! (CancelRequest, GSSENCRequest, FunctionCall, FunctionCallResponse,
//! NegotiateProtocolVersion, CopyBothResponse).
//!
//! No I/O traits in the signature. No `async`. No `tokio`. No `std::net`.
//! No allocation: decode produces borrowed views over the caller's
//! buffer ([`types::PgStr`], the lazy iterators in [`views`]); encode
//! writes into caller-owned storage via [`writer::MessageWriter`] and the
//! streaming writers ([`backend::DataRowWriter`],
//! [`backend::RowDescriptionWriter`], [`backend::ErrorResponseWriter`],
//! [`frontend::BindWriter`]). The crate is tier-3: bare `no_std`,
//! no `alloc` — usable from prime, DPDK-style loops, embedded targets,
//! fuzzers, and test harnesses. The caller owns the wire and the heap
//! policy.
//!
//! # Layers
//!
//! - [`frontend`] / [`backend`] — stateless, re-entrant message codec:
//!   `parse_*(&[u8]) -> Ok(None)` until a full frame is buffered, then a
//!   borrowed message view plus the consumed byte count.
//! - [`session`] — the explicit connection FSM ([`session::Session`]):
//!   startup / SSL / GSS / cancel choreography, authentication flows,
//!   simple and extended query sequencing, error recovery until Sync,
//!   COPY sub-protocols, and the ReadyForQuery transaction status.
//! - the std facade (`proxima-pgwire`) composes these over
//!   `proxima-stream` / `proxima-listen` / `proxima-tls`; this crate is
//!   deliberately consumable without it.
//!
//! # Teaching pointers
//!
//! The primitives this crate composes: `memchr` for SIMD NUL scans (the
//! only dependency), and the workspace codec conventions established by
//! `proxima-h1-codec` (borrowed-view parse) and `proxima-quic-proto`
//! (sans-IO state machines, caller-owned output). See
//! `examples/pgwire_codec_session_walkthrough.rs` for the FSM driven
//! through every legal transition path.

mod cursor;
mod frame;

pub mod backend;
pub mod error;
pub mod frontend;
pub mod session;
pub mod types;
pub mod views;
pub mod writer;

pub use backend::{AuthRequest, BackendMessage, GssEncResponse, SslResponse};
pub use error::{EncodeError, ParseError};
pub use frontend::{FrontendMessage, InitialMessage};
pub use session::{AuthFlow, Disposition, Session, SessionError, WirePhase};
pub use types::{
    CopyFormat, FormatCode, Oid, PgStr, ProtocolVersion, StatementTarget, TransactionStatus,
};
