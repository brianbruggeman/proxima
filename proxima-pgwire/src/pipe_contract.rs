//! The SQL-over-Pipe contract: the self-describing request enum and typed
//! payloads a PostgreSQL connection exchanges with a
//! `proxima_primitives::pipe::Pipe`.
//!
//! This is the RISC payoff (workspace principle 1). pgwire does not own a
//! bespoke handler trait — it speaks the one workspace primitive, `Pipe`,
//! the way `proxima-redis` and `proxima-kafka` do: a business-handler `Pipe`
//! is `P -> Q` (payload-no-cell, no `Request`/`Response` envelope), and
//! [`QueryRequest`] is already self-describing — the FSM/verb information
//! that used to live on `Request.method`/`.path` is now its `verb` field.
//! Because a SQL engine is just a `Pipe`, every proxima middleware —
//! `Auth`, `RateLimit`, `Retry`, `Tee`, `Diff`, record/replay,
//! `RoutingPipe` — composes onto SQL with zero new code.
//!
//! # Two layers
//!
//! - **connection** ([`verb::CONNECT`]): the listener calls the mounted
//!   connection `Pipe` once per accepted socket over the transport's own
//!   `Request<Bytes>`/`Response<Bytes>` envelope (a raw-socket-hijack seam,
//!   not the SQL contract below); the `Pipe` returns a `Response` whose
//!   `upgrade` runs the session loop. This is the same seam WebSocket /
//!   CONNECT tunnels use (`proxima_primitives::pipe::upgrade`).
//! - **query** ([`Verb`]'s `Query`/`Parse`/`Execute`/`CopyData` variants):
//!   inside the session, each protocol operation the driver actually
//!   dispatches to the engine becomes one `Pipe::call(QueryRequest) ->
//!   PgReply`. SQL text rides `QueryRequest::sql`; bind parameters and
//!   other verb-specific data ride the matched `verb` variant's fields;
//!   the driver owns wire framing and text/binary format-code encoding so
//!   the SQL engine stays wire-agnostic. `Describe` (extended-protocol
//!   Describe) is answered entirely from the driver's own statement/portal
//!   store and never reaches the engine, so it has no variant here.
//!
//! The neutral [`SqlValue`] lives here, not in any engine: format-code
//! encoding needs typed values (an `i64` becomes 8 big-endian bytes in
//! binary, decimal ASCII in text), so the engine yields typed cells and
//! the driver encodes. Engines map their own value type to [`SqlValue`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use proxima_protocols::pgwire_codec::{CopyFormat, FormatCode, Oid};

/// Bytes-verb vocabulary for translating a generic, untyped transport
/// (`Request<Bytes>.method`, e.g. `proxima::Client`'s universal envelope, or
/// a hand-built test request) into a [`QueryRequest`] variant. The driver
/// itself no longer builds requests from these bytes — it constructs each
/// variant directly at its one dispatch site — but a boundary that only has
/// raw method bytes to work with still needs a shared vocabulary to name
/// them. `DESCRIBE` has no entry: the driver answers Describe locally and
/// never dispatches it to the engine (see the module doc).
pub mod verb {
    /// a freshly accepted connection; the connection `Pipe` returns
    /// `Response.upgrade` (the transport's own `Request<Bytes>` envelope,
    /// not the `QueryRequest` contract)
    pub const CONNECT: &[u8] = b"CONNECT";
    /// simple-protocol query -> [`super::Verb::Query`]
    pub const QUERY: &[u8] = b"QUERY";
    /// extended-protocol Parse -> [`super::Verb::Parse`]
    pub const PARSE: &[u8] = b"PARSE";
    /// extended-protocol Execute against a bound portal ->
    /// [`super::Verb::Execute`]
    pub const EXECUTE: &[u8] = b"EXECUTE";
    /// COPY IN second phase -> [`super::Verb::CopyData`]
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
    /// A token that is never cancelled — the default for a directly-driven
    /// session (a client construction site, a test) that has no real
    /// `CancelRegistry` flag to substitute.
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

/// The business-handler pipe's request payload — a de-enveloped
/// (payload-no-cell) carry: a pipe is `QueryRequest -> PgReply`, no
/// `Request`/`Response` wrapper. `sql`/`connection_id`/`cancel` ride every
/// protocol operation, so they live on the request itself; [`Verb`] carries
/// only the fields specific to the operation the driver's own dispatch
/// actually calls the engine for — `Query` (simple-protocol query), `Parse`
/// (extended-protocol Parse), `Execute` (extended-protocol Execute against a
/// bound portal), and `CopyData` (the COPY IN second phase). Bind (`Bind`),
/// Describe, Close, Flush, and Sync are answered entirely from the driver's
/// own statement/portal store and never reach the engine, so they have no
/// variant here (see the module doc).
///
/// `connection_id` and `cancel` are shared across every verb: the engine
/// keys per-connection state (transaction snapshots) off `connection_id`,
/// and polls [`CancelToken::is_cancelled`] between units of work to abort a
/// long-running query cooperatively.
#[derive(Debug, Clone)]
pub struct QueryRequest {
    /// The client's literal SQL text — every verb operates against it.
    pub sql: String,
    /// Process-unique connection id shared across every verb so the engine
    /// can key per-connection state regardless of which operation
    /// dispatched.
    pub connection_id: u64,
    /// Cooperative cancel signal shared across every verb so the engine can
    /// poll [`CancelToken::is_cancelled`] regardless of which operation
    /// dispatched.
    pub cancel: CancelToken,
    /// The protocol operation and its verb-specific fields.
    pub verb: Verb,
}

/// The protocol-operation-specific fields a [`QueryRequest`] carries beyond
/// its shared `sql`/`connection_id`/`cancel`.
#[derive(Debug, Clone)]
pub enum Verb {
    /// Simple-protocol query (`verb::QUERY`): no extra fields — `sql` is the
    /// whole request.
    Query,
    /// Extended-protocol Parse (`verb::PARSE`): declares a prepared
    /// statement named `statement` (empty = unnamed) for `sql`. The engine
    /// answers with [`PgReply::Describe`].
    Parse { statement: String },
    /// Extended-protocol Execute (`verb::EXECUTE`) against a bound portal:
    /// `statement`/`portal` name the prepared statement and portal (empty =
    /// unnamed), `parameters` are the bind values already decoded from wire
    /// format to typed values.
    Execute {
        statement: String,
        portal: String,
        parameters: Vec<SqlValue>,
    },
    /// COPY IN second phase (`verb::COPY_DATA`): the driver collected the
    /// client's CopyData rows (or saw CopyFail) for the `sql` that opened
    /// the transfer and hands them back for the engine to apply.
    /// `copy_failed` is true when the client sent CopyFail instead of
    /// CopyDone, so the engine rolls back.
    CopyData {
        copy_data: Vec<Vec<u8>>,
        copy_failed: bool,
    },
}

impl QueryRequest {
    /// Builds a request from a generic bytes verb + wire fields — the
    /// construction site for a boundary that only has untyped transport
    /// fields to work with (`proxima::Client`'s universal `Request<Bytes>`
    /// envelope, a hand-built test request), mirroring
    /// `RedisRequest::from_args`. `statement` is used for `PARSE`;
    /// `parameters` for `EXECUTE`.
    ///
    /// An unrecognized verb is REJECTED, not widened to `Query`: this
    /// boundary sits directly in front of live SQL execution, so coercing
    /// garbage input into a query would silently execute the request body
    /// as SQL (a fail-open on a SQL boundary — the wrong default). The
    /// error message matches the pre-de-envelope client dispatch's own
    /// rejection (`ClientError::Protocol("unsupported client verb")`) so a
    /// caller sees the identical text either way.
    pub fn try_from_wire(
        verb: &[u8],
        statement: impl Into<String>,
        sql: impl Into<String>,
        parameters: Vec<SqlValue>,
    ) -> Result<Self, &'static str> {
        let sql = sql.into();
        let query_verb = match verb {
            self::verb::QUERY => Verb::Query,
            self::verb::PARSE => Verb::Parse {
                statement: statement.into(),
            },
            self::verb::EXECUTE => Verb::Execute {
                statement: statement.into(),
                portal: String::new(),
                parameters,
            },
            _ => return Err("unsupported client verb"),
        };
        Ok(Self {
            sql,
            connection_id: 0,
            cancel: CancelToken::none(),
            verb: query_verb,
        })
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn try_from_wire_query_verb_builds_the_query_variant() {
        let request = QueryRequest::try_from_wire(verb::QUERY, "", "select 1", Vec::new())
            .expect("QUERY must be accepted");
        assert_eq!(request.sql, "select 1");
        assert!(matches!(request.verb, Verb::Query));
    }

    #[test]
    fn try_from_wire_parse_verb_carries_the_statement_name() {
        let request = QueryRequest::try_from_wire(verb::PARSE, "stmt1", "select $1", Vec::new())
            .expect("PARSE must be accepted");
        assert_eq!(request.sql, "select $1");
        match request.verb {
            Verb::Parse { statement } => {
                assert_eq!(statement, "stmt1");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn try_from_wire_execute_verb_carries_statement_and_parameters() {
        let request = QueryRequest::try_from_wire(
            verb::EXECUTE,
            "stmt1",
            "select $1",
            vec![SqlValue::Int(7)],
        )
        .expect("EXECUTE must be accepted");
        assert_eq!(request.sql, "select $1");
        match request.verb {
            Verb::Execute {
                statement,
                parameters,
                ..
            } => {
                assert_eq!(statement, "stmt1");
                assert_eq!(parameters, vec![SqlValue::Int(7)]);
            }
            other => panic!("expected Execute, got {other:?}"),
        }
    }

    /// SQL-boundary fail-loud contract (post PC3-pgwire review): an
    /// unrecognized verb at the generic-bytes construction site must be
    /// REJECTED, never silently coerced into a live `Query` — the
    /// pre-de-envelope client dispatch's own default arm rejected exactly
    /// this input (`ClientError::Protocol("unsupported client verb")`), and
    /// this restores that fail-loud behavior instead of executing the
    /// request body as SQL.
    #[test]
    fn try_from_wire_unrecognized_verb_is_rejected_not_coerced_to_query() {
        let outcome = QueryRequest::try_from_wire(b"BOGUS", "", "select 1", Vec::new());
        assert_eq!(outcome.unwrap_err(), "unsupported client verb");
    }

    #[test]
    fn try_from_wire_rejects_a_plausible_but_unsupported_pg_verb() {
        // COPY_DATA is a real driver-internal verb (see verb::COPY_DATA) but
        // is never a valid CLIENT-dispatch verb — a garbage/mismatched verb
        // at this boundary must not silently run the body as a query either.
        let outcome =
            QueryRequest::try_from_wire(b"COPY_DATA", "", "delete from users", Vec::new());
        assert_eq!(outcome.unwrap_err(), "unsupported client verb");
    }

    #[test]
    fn connection_id_and_cancel_are_shared_fields_regardless_of_verb() {
        let flag = Arc::new(AtomicBool::new(true));
        let cancel = CancelToken::from(Arc::clone(&flag));
        let request = QueryRequest {
            sql: "copy t from stdin".to_string(),
            connection_id: 42,
            cancel,
            verb: Verb::CopyData {
                copy_data: vec![b"1\tx\n".to_vec()],
                copy_failed: false,
            },
        };
        assert_eq!(request.connection_id, 42);
        assert!(request.cancel.is_cancelled());
        assert_eq!(request.sql, "copy t from stdin");
    }
}
