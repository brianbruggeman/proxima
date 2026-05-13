//! The SQL-over-Pipe contract: the verb vocabulary and typed payloads a
//! PostgreSQL connection exchanges with a `proxima_primitives::pipe::Pipe`.
//!
//! This is the RISC payoff (workspace principle 1). pgwire does not own a
//! bespoke handler trait — it speaks the one workspace primitive, `Pipe`,
//! exactly the way `proxima-telemetry` does: a single `Pipe` matches on
//! `Request.method` verbs and reads/writes typed values through `Carry`
//! (`proxima-telemetry` carries `SpanRecord`/`LogRecord`; pgwire carries
//! [`QueryRequest`] / [`QueryReply`]). Because a SQL engine is just a
//! `Pipe`, every proxima middleware — `Auth`, `RateLimit`, `Retry`,
//! `Tee`, `Diff`, record/replay, `RoutingPipe` — composes onto SQL with
//! zero new code.
//!
//! # Two layers
//!
//! - **connection** ([`verb::CONNECT`]): the listener calls the mounted
//!   `Pipe` once per accepted socket; the `Pipe` returns a `Response`
//!   whose `upgrade` runs the session loop. This is the same
//!   raw-socket-hijack seam WebSocket / CONNECT tunnels use
//!   (`proxima_primitives::pipe::upgrade`).
//! - **query** ([`verb::QUERY`] / [`verb::PARSE`] / [`verb::DESCRIBE`] /
//!   [`verb::EXECUTE`]): inside the session, each protocol operation
//!   becomes a `Pipe::call`. SQL text rides `Request.body`; bind
//!   parameters, statement metadata, and result rows ride `Carry`. The
//!   driver owns wire framing and text/binary format-code encoding so
//!   the SQL engine stays wire-agnostic.
//!
//! The neutral [`SqlValue`] lives here, not in any engine: format-code
//! encoding needs typed values (an `i64` becomes 8 big-endian bytes in
//! binary, decimal ASCII in text), so the engine yields typed cells and
//! the driver encodes. Engines map their own value type to [`SqlValue`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use proxima_protocols::pgwire_codec::{CopyFormat, FormatCode, Oid};

/// Request-method verbs the driver sets and a SQL `Pipe` matches on.
/// Bytes, not an enum, mirroring `proxima_telemetry`'s `METHOD_*` set —
/// the wire substrate is bytes-internal and the verb is a cheap
/// discriminant before the `Carry` downcast.
pub mod verb {
    /// a freshly accepted connection; the `Pipe` returns `Response.upgrade`
    pub const CONNECT: &[u8] = b"CONNECT";
    /// simple-protocol query; SQL in `QueryRequest.sql`
    pub const QUERY: &[u8] = b"QUERY";
    /// extended-protocol Parse; SQL in `QueryRequest.sql`, name in `path`
    pub const PARSE: &[u8] = b"PARSE";
    /// extended-protocol Describe; statement/portal name in `path`
    pub const DESCRIBE: &[u8] = b"DESCRIBE";
    /// extended-protocol Execute against a bound portal
    pub const EXECUTE: &[u8] = b"EXECUTE";
    /// COPY IN second phase: the driver collected the client's CopyData
    /// rows (or saw CopyFail) and hands them back for the engine to apply.
    /// SQL in `Request.body`; the rows ride `QueryRequest.copy_data` and the
    /// abort flag rides `QueryRequest.copy_failed`.
    pub const COPY_DATA: &[u8] = b"COPY_DATA";
}

/// A neutral SQL value, owned so it can ride `Carry`
/// (`Arc<dyn Any + Send + Sync>` is `'static`). The driver encodes each
/// cell to text or binary per the portal's result format codes.
#[derive(Debug, Clone, PartialEq)]
pub enum SqlValue {
    Null,
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Bytes(Vec<u8>),
}

/// A cooperative cancel signal the driver threads into every engine
/// request. A matching `CancelRequest` flips the connection's flag (the
/// `CancelRegistry` owns the write side); the engine reads
/// [`CancelToken::is_cancelled`] between units of work and aborts
/// cooperatively — the Pipe-native restoration of the old
/// `Execution.cancelled` capability. A never-cancelled token
/// ([`CancelToken::none`]) is the default for directly-driven sessions.
#[derive(Clone)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A token that is never cancelled — the `QueryRequest::new` default.
    #[must_use]
    pub fn none() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// True once a matching `CancelRequest` has fired for this connection.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

impl From<Arc<AtomicBool>> for CancelToken {
    fn from(flag: Arc<AtomicBool>) -> Self {
        Self(flag)
    }
}

impl core::fmt::Debug for CancelToken {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_tuple("CancelToken")
            .field(&self.is_cancelled())
            .finish()
    }
}

/// One result column's wire metadata, produced by Parse/Describe and by
/// a query reply that carries rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDesc {
    pub name: String,
    pub type_oid: Oid,
}

impl ColumnDesc {
    #[must_use]
    pub fn new(name: impl Into<String>, type_oid: Oid) -> Self {
        Self {
            name: name.into(),
            type_oid,
        }
    }
}

/// The typed payload the driver puts in `Request.payload` for QUERY /
/// EXECUTE. The SQL text rides `sql`; the out-of-band context is the
/// remaining fields.
#[derive(Debug, Clone)]
pub struct QueryRequest {
    /// SQL text for QUERY / PARSE / EXECUTE / COPY_DATA verbs
    pub sql: String,
    /// process-unique connection id; the engine keys per-connection
    /// state (transaction snapshots) off it
    pub connection_id: u64,
    /// bound parameters, already decoded from wire format to typed values
    pub parameters: Vec<SqlValue>,
    /// statement name (Parse/Describe/Execute target); empty = unnamed
    pub statement: String,
    /// portal name (Execute target); empty = unnamed
    pub portal: String,
    /// COPY IN payload (set only on the `COPY_DATA` verb): the client's
    /// CopyData rows, one raw COPY-format payload per `Vec<u8>`. Empty on
    /// every other verb.
    pub copy_data: Vec<Vec<u8>>,
    /// COPY IN abort flag (set only on the `COPY_DATA` verb): true when the
    /// client sent CopyFail instead of CopyDone, so the engine rolls back.
    pub copy_failed: bool,
    /// cooperative cancellation: the engine polls
    /// [`CancelToken::is_cancelled`] between units of work to abort a
    /// long-running query. Defaults to a never-cancelled token; the driver
    /// substitutes the connection's real flag.
    pub cancel: CancelToken,
}

impl QueryRequest {
    #[must_use]
    pub fn new(connection_id: u64) -> Self {
        Self {
            sql: String::new(),
            connection_id,
            parameters: Vec::new(),
            statement: String::new(),
            portal: String::new(),
            copy_data: Vec::new(),
            copy_failed: false,
            cancel: CancelToken::none(),
        }
    }
}

/// A lazy row source the engine hands the driver for [`PgReply::QueryStream`]
/// (gate G10). One channel item is one row (`Vec<SqlValue>`), the same
/// element shape as [`QueryReply::rows`]; the producer feeds the sender from
/// a spawned task or lazy iterator while the driver drains, encodes, and
/// flushes incrementally — so an unbounded result set rides bounded memory
/// instead of a fully-materialized `Vec`.
///
/// `async_channel::Receiver` is the deliberate choice: `Carry` requires
/// `Send + Sync + 'static`, which a `futures`/`tokio` mpsc `Receiver` is not
/// (`Sync` is missing), but `async_channel::Receiver` is `Send + Sync +
/// Clone` and executor-agnostic (no tokio in the bare graph).
#[derive(Debug, Clone)]
pub struct RowStream(async_channel::Receiver<Vec<SqlValue>>);

impl RowStream {
    #[must_use]
    pub fn new(receiver: async_channel::Receiver<Vec<SqlValue>>) -> Self {
        Self(receiver)
    }

    #[must_use]
    pub fn receiver(&self) -> &async_channel::Receiver<Vec<SqlValue>> {
        &self.0
    }
}

/// The typed payload a SQL `Pipe` puts in `Response.carry`. For a
/// row-returning statement, `columns` describes the result and `rows`
/// holds the typed cells; for a non-row statement (DDL/DML/tx-control),
/// `columns`/`rows` are empty and `command_tag` carries the completion
/// tag (`INSERT 0 2`, `BEGIN`, ...). When `columns` is non-empty the
/// driver derives the tag (`SELECT n`) unless `command_tag` overrides it.
///
/// v1 buffers `rows` (the engine materializes them anyway); the driver
/// still streams the *encoding* with high-water flush. A lazy/streaming
/// row source (a channel carried in `Carry`) is gate G10.
#[derive(Debug, Clone, Default)]
pub struct QueryReply {
    pub columns: Vec<ColumnDesc>,
    pub rows: Vec<Vec<SqlValue>>,
    pub command_tag: Option<String>,
    /// non-fatal notices to emit before the result (severity/code/message)
    pub notices: Vec<NoticeReply>,
    /// transaction status the engine moved to (BEGIN/COMMIT/ROLLBACK),
    /// or `None` to leave the driver's tracking unchanged
    pub transaction: Option<TxStatus>,
}

/// Transaction status an engine reports back to the driver, mapped to the
/// codec's `TransactionStatus` for ReadyForQuery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxStatus {
    Idle,
    InTransaction,
    Failed,
}

/// A NoticeResponse an engine asks the driver to emit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoticeReply {
    pub sqlstate: String,
    pub message: String,
}

/// What Parse/Describe answers: parameter types and result columns,
/// carried in `Response.carry`. Empty `parameter_types` means the engine
/// leaves parameter typing to bind time.
#[derive(Debug, Clone, Default)]
pub struct DescribeReply {
    pub parameter_types: Vec<Oid>,
    pub columns: Vec<ColumnDesc>,
}

/// A SQL error an engine reports — an expected outcome the driver turns
/// into an ErrorResponse and recovers from (vs `Err(ProximaError)`, which
/// is a transport/Pipe failure that aborts the connection).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorReply {
    pub severity: String,
    pub sqlstate: String,
    pub message: String,
    pub detail: Option<String>,
    pub hint: Option<String>,
}

impl ErrorReply {
    #[must_use]
    pub fn new(sqlstate: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: "ERROR".into(),
            sqlstate: sqlstate.into(),
            message: message.into(),
            detail: None,
            hint: None,
        }
    }
}

/// The single typed value a SQL `Pipe` returns in `Response.carry`. The
/// driver downcasts this one type and matches; it knows which verb it
/// sent, so a mismatched variant (e.g. `Describe` for a `QUERY`) is a
/// contract violation it reports.
#[derive(Debug, Clone)]
pub enum PgReply {
    Query(QueryReply),
    /// Like [`PgReply::Query`] but rows arrive lazily over a [`RowStream`]
    /// instead of a buffered `Vec` — the engine produces them on demand and
    /// the driver drains/encodes/flushes incrementally for bounded memory
    /// (gate G10). The driver derives `SELECT n` (n = rows drained) unless
    /// `command_tag` overrides it.
    QueryStream {
        columns: Vec<ColumnDesc>,
        rows: RowStream,
        command_tag: Option<String>,
    },
    Describe(DescribeReply),
    Error(ErrorReply),
    /// COPY OUT (`COPY t TO STDOUT`): the engine supplies the COPY rows as
    /// raw CopyData payloads. The driver emits CopyOutResponse, one CopyData
    /// per `data` entry, CopyDone, then `CommandComplete("COPY {n}")`.
    CopyOut {
        format: CopyFormat,
        column_formats: Vec<FormatCode>,
        data: Vec<Vec<u8>>,
    },
    /// COPY IN (`COPY t FROM STDIN`): the engine signals it will receive
    /// COPY data. The driver emits CopyInResponse, collects the client's
    /// CopyData/CopyDone/CopyFail, then re-calls the engine with the
    /// [`verb::COPY_DATA`] verb carrying the collected rows.
    CopyIn {
        format: CopyFormat,
        column_formats: Vec<FormatCode>,
    },
    /// `LISTEN chan` (one or more channels): the engine recognized a LISTEN
    /// and the driver subscribes the connection on the shared broker, then
    /// emits `CommandComplete("LISTEN")`. The engine never touches the
    /// broker — delivery is the driver's wire job, exactly like the socket.
    Listen {
        channels: Vec<String>,
    },
    /// `UNLISTEN chan` / `UNLISTEN *`: the driver unsubscribes the named
    /// channels (or every channel when `all`), then `CommandComplete("UNLISTEN")`.
    Unlisten {
        channels: Vec<String>,
        all: bool,
    },
    /// `NOTIFY chan[, 'payload']`: the driver publishes to every connection
    /// listening on `channel`, stamping the notification with *this*
    /// connection's backend pid, then `CommandComplete("NOTIFY")`. Self-notify
    /// is delivered (PostgreSQL behavior).
    Notify {
        channel: String,
        payload: String,
    },
}

impl QueryReply {
    #[must_use]
    pub fn rows(columns: Vec<ColumnDesc>, rows: Vec<Vec<SqlValue>>) -> Self {
        Self {
            columns,
            rows,
            command_tag: None,
            notices: Vec::new(),
            transaction: None,
        }
    }

    #[must_use]
    pub fn tag(command_tag: impl Into<String>) -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            command_tag: Some(command_tag.into()),
            notices: Vec::new(),
            transaction: None,
        }
    }

    #[must_use]
    pub fn with_transaction(mut self, status: TxStatus) -> Self {
        self.transaction = Some(status);
        self
    }
}
