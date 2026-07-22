//! The per-connection driver: reads frames, feeds the sans-IO session
//! FSM, builds a `Request` per protocol operation, calls the SQL engine
//! `Pipe`, and encodes the typed `PgReply` back onto the wire.
//!
//! Composes `proxima_protocols::pgwire_codec` (parse / encode / `Session`) over any
//! `futures::io` stream — no runtime, no socket type, no TLS knowledge.
//! TLS lives one layer up: [`negotiate`] returns
//! [`Negotiation::StartTls`] and the caller re-enters with the wrapped
//! stream (see the `listen` feature). The query surface is a
//! `proxima_primitives::pipe::Pipe`: the driver owns wire framing and the text/binary
//! format-code encoding of [`SqlValue`], so the engine stays
//! wire-agnostic (see [`crate::pipe_contract`]).

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};

use bytes::{Buf, Bytes, BytesMut};
use dashmap::DashMap;
use futures::FutureExt;
use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use futures::stream::StreamExt;

use proxima_protocols::pgwire_codec::backend::{
    DataRowWriter, ErrorResponseWriter, RowDescriptionWriter, encode_copy_in_response,
    encode_copy_out_response, encode_negotiate_protocol_version, encode_parameter_description,
};
use proxima_protocols::pgwire_codec::frontend::{
    Bind, FunctionCall, InitialMessage, ParseMessage, parse_frontend, parse_initial,
};
use proxima_protocols::pgwire_codec::session::StateName;
use proxima_protocols::pgwire_codec::views::FieldDescription;
use proxima_protocols::pgwire_codec::{
    AuthRequest, BackendMessage, CopyFormat, Disposition, FormatCode, FrontendMessage, Oid, PgStr,
    ProtocolVersion, Session, SessionError, SslResponse, StatementTarget, TransactionStatus,
};

use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::request::RequestContext;

use crate::pipes::{PgPipeHandle, PgRequest, PgResponse};

#[cfg(feature = "scram")]
use base64::Engine as _;

use crate::auth::PgAuth;
use crate::broker::{Notification, NotifyBroker};
use crate::config::PgServerConfig;
use crate::error::ServeError;
use crate::handler::{ErrorInfo, commit, error_response_size, reserve, write_error_fields};
use crate::pipe_contract::{
    CancelToken, ColumnDesc, DescribeReply, PgReply, QueryReply, QueryRequest, RowStream, SqlValue,
    TxStatus, verb,
};
use crate::store::{
    BoundParameter, NamedSlots, PendingRows, Portal, PreparedStatement, StoreError,
};

/// The runtime threaded down to the auth path so the CPU-bound SCRAM
/// PBKDF2 can be offloaded off the reactor core. With `listen` it is a
/// real `&dyn Runtime`; without it (the bare no-tokio graph) it collapses
/// to `Option<()>` and the auth path stays inline — mirrors the
/// `TlsAcceptor = ()` idiom in `pipe.rs`.
#[cfg(feature = "listen")]
pub type RuntimeHandle = Option<std::sync::Arc<dyn proxima_runtime::Runtime>>;
#[cfg(not(feature = "listen"))]
pub type RuntimeHandle = Option<()>;

/// The identity reported in BackendKeyData and matched by CancelRequest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendKey {
    pub process_id: i32,
    pub secret_key: i32,
}

/// Listener-wide map from backend process id to cancel key + flag.
/// Cancellation is engine-cooperative: a matching CancelRequest sets the
/// connection's flag (mirroring how PostgreSQL queries poll for
/// interrupts). v1 has no out-of-band engine notification hook.
#[derive(Debug, Default)]
pub struct CancelRegistry {
    next_process_id: AtomicI32,
    entries: DashMap<i32, (i32, Arc<AtomicBool>)>,
}

fn constant_time_eq_i32(left: i32, right: i32) -> bool {
    (left ^ right) == 0
}

impl CancelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_process_id: AtomicI32::new(1000),
            entries: DashMap::new(),
        }
    }

    /// Registers a new connection, returning its key and cancel flag.
    #[must_use]
    pub fn register(&self) -> (BackendKey, Arc<AtomicBool>) {
        let process_id = self.next_process_id.fetch_add(1, Ordering::Relaxed);
        let secret_key = rand::random::<i32>();
        let flag = Arc::new(AtomicBool::new(false));
        self.entries
            .insert(process_id, (secret_key, Arc::clone(&flag)));
        (
            BackendKey {
                process_id,
                secret_key,
            },
            flag,
        )
    }

    /// Applies a CancelRequest; returns true when pid + key matched.
    /// The protocol sends no reply either way.
    pub fn cancel(&self, process_id: i32, secret_key: &[u8]) -> bool {
        let Ok(key_bytes) = <[u8; 4]>::try_from(secret_key) else {
            return false;
        };
        let presented = i32::from_be_bytes(key_bytes);
        match self.entries.get(&process_id) {
            Some(entry) if constant_time_eq_i32(presented, entry.0) => {
                entry.1.store(true, Ordering::Relaxed);
                true
            }
            _ => false,
        }
    }

    pub fn deregister(&self, process_id: i32) {
        self.entries.remove(&process_id);
    }
}

/// Owned copy of the startup packet, surviving past the read buffer.
#[derive(Debug, Clone)]
pub struct StartupOwned {
    pub version: ProtocolVersion,
    pub user: String,
    pub database: Option<String>,
    pub parameters: Vec<(String, String)>,
    /// `_pq_.*` options the server does not recognize, reported inside
    /// NegotiateProtocolVersion
    pub negotiate_options: Vec<String>,
    /// the client asked for protocol 3.x minor > 0 or unknown `_pq_.*`
    /// options; the server answers NegotiateProtocolVersion before
    /// authentication (required for a bare minor bump even with zero
    /// unrecognized options)
    pub needs_negotiation: bool,
}

/// Outcome of the untagged startup phase.
pub enum Negotiation<S> {
    /// SSLRequest accepted (`S` written); wrap the stream in TLS, call
    /// `session.tls_established()`, and run [`negotiate`] again
    StartTls(S),
    /// startup accepted; run [`serve_session`], handing it the bytes the
    /// peer pipelined behind the startup packet
    Proceed {
        stream: S,
        startup: StartupOwned,
        leftover: BytesMut,
    },
    /// a cancel connection: apply against the registry and close
    Cancel {
        process_id: i32,
        secret_key: Vec<u8>,
    },
    /// peer closed before sending a complete startup message
    Closed,
}

async fn read_some<S: AsyncRead + Unpin>(
    stream: &mut S,
    buf: &mut BytesMut,
    scratch: &mut [u8],
) -> Result<usize, ServeError> {
    let read = stream.read(scratch).await?;
    buf.extend_from_slice(&scratch[..read]);
    Ok(read)
}

fn utf8_owned(value: PgStr<'_>, field: &'static str) -> Result<String, ServeError> {
    value
        .to_str()
        .map(str::to_owned)
        .map_err(|_| ServeError::InvalidUtf8 { field })
}

fn startup_owned(
    startup: &proxima_protocols::pgwire_codec::frontend::Startup<'_>,
) -> Result<StartupOwned, ServeError> {
    let mut user = None;
    let mut database = None;
    let mut parameters = Vec::new();
    let mut negotiate_options = Vec::new();
    for (key, value) in startup.parameters.iter() {
        let key_text = utf8_owned(key, "startup parameter name")?;
        let value_text = utf8_owned(value, "startup parameter value")?;
        if key_text == "user" {
            user = Some(value_text.clone());
        } else if key_text == "database" {
            database = Some(value_text.clone());
        }
        if key_text.starts_with("_pq_.") {
            negotiate_options.push(key_text.clone());
        }
        parameters.push((key_text, value_text));
    }
    negotiate_options.sort();
    negotiate_options.dedup();
    let needs_negotiation = startup.version.minor > 0 || !negotiate_options.is_empty();
    let user = user.ok_or(ServeError::MissingUser)?;
    Ok(StartupOwned {
        version: startup.version,
        user,
        database,
        parameters,
        negotiate_options,
        needs_negotiation,
    })
}

/// PostgreSQL rejects startup packets above 10000 bytes
/// (`MAX_STARTUP_PACKET_LENGTH`); enforcing the same bound caps what an
/// unauthenticated peer can make the server buffer.
const MAX_STARTUP_PACKET_BYTES: usize = 10_000;

/// Drives the untagged startup phase on a fresh (or freshly TLS-wrapped)
/// stream.
///
/// # Errors
/// [`ServeError`] on transport failure, malformed startup traffic, or a
/// protocol violation (e.g. bytes pipelined behind an SSLRequest).
pub async fn negotiate<S: AsyncRead + AsyncWrite + Unpin>(
    mut stream: S,
    session: &mut Session,
    tls_available: bool,
) -> Result<Negotiation<S>, ServeError> {
    let mut buf = BytesMut::with_capacity(1024);
    let mut scratch = [0_u8; 1024];
    loop {
        let parsed = parse_initial(&buf)?;
        let Some((message, consumed)) = parsed else {
            if buf.len() > MAX_STARTUP_PACKET_BYTES {
                return Err(ServeError::MessageTooLarge {
                    limit: MAX_STARTUP_PACKET_BYTES,
                });
            }
            if read_some(&mut stream, &mut buf, &mut scratch).await? == 0 {
                if buf.is_empty() {
                    return Ok(Negotiation::Closed);
                }
                return Err(ServeError::UnexpectedEof);
            }
            continue;
        };
        session.on_initial(&message)?;
        match message {
            InitialMessage::SslRequest => {
                if tls_available {
                    if buf.len() > consumed {
                        // no client may pipeline bytes behind SSLRequest;
                        // anything buffered would bypass the TLS boundary
                        // (CVE-2021-23222 class)
                        return Err(ServeError::Parse(
                            proxima_protocols::pgwire_codec::ParseError::TrailingBytes {
                                tag: 0,
                                trailing: buf.len() - consumed,
                            },
                        ));
                    }
                    session.ssl_accepted()?;
                    let mut byte = [0_u8; 1];
                    let written = SslResponse::Accept.encode(&mut byte)?;
                    stream.write_all(&byte[..written]).await?;
                    stream.flush().await?;
                    return Ok(Negotiation::StartTls(stream));
                }
                session.ssl_refused()?;
                let mut byte = [0_u8; 1];
                let written = SslResponse::Refuse.encode(&mut byte)?;
                stream.write_all(&byte[..written]).await?;
                stream.flush().await?;
                buf.advance(consumed);
            }
            InitialMessage::GssEncRequest => {
                session.gss_enc_refused()?;
                stream.write_all(b"N").await?;
                stream.flush().await?;
                buf.advance(consumed);
            }
            InitialMessage::Cancel(cancel) => {
                return Ok(Negotiation::Cancel {
                    process_id: cancel.process_id,
                    secret_key: cancel.secret_key.to_vec(),
                });
            }
            InitialMessage::Startup(startup) => {
                let owned = startup_owned(&startup)?;
                buf.advance(consumed);
                return Ok(Negotiation::Proceed {
                    stream,
                    startup: owned,
                    leftover: buf,
                });
            }
        }
    }
}

fn append_message(out: &mut Vec<u8>, size: usize, message: &BackendMessage<'_>) -> io::Result<()> {
    let start = reserve(out, size);
    let outcome = message.encode(&mut out[start..]);
    commit(out, start, outcome)
}

fn append_error(out: &mut Vec<u8>, info: &ErrorInfo) -> io::Result<()> {
    let start = reserve(out, error_response_size(info));
    let outcome = (|| {
        let mut writer = ErrorResponseWriter::error(&mut out[start..])?;
        write_error_fields(&mut writer, info)?;
        writer.finish()
    })();
    commit(out, start, outcome)
}

fn append_notice(out: &mut Vec<u8>, info: &ErrorInfo) -> io::Result<()> {
    let start = reserve(out, error_response_size(info));
    let outcome = (|| {
        let mut writer = ErrorResponseWriter::notice(&mut out[start..])?;
        write_error_fields(&mut writer, info)?;
        writer.finish()
    })();
    commit(out, start, outcome)
}

async fn flush_out<S: AsyncWrite + Unpin>(stream: &mut S, out: &mut Vec<u8>) -> io::Result<()> {
    if !out.is_empty() {
        stream.write_all(out).await?;
        out.clear();
    }
    stream.flush().await
}

fn protocol_violation(detail: impl std::fmt::Display) -> ErrorInfo {
    ErrorInfo::new("08P01", format!("protocol violation: {detail}")).fatal()
}

struct ConnState {
    statements: NamedSlots<PreparedStatement>,
    portals: NamedSlots<Portal>,
    connection_id: u64,
    /// the connection's cancel flag (set by a matching CancelRequest via the
    /// registry); threaded into every engine request as a [`CancelToken`] so
    /// an interruptible engine can abort a long-running query cooperatively
    cancel: CancelToken,
    /// the shared LISTEN/NOTIFY fabric; `None` for a directly-driven session
    /// (no listener) — LISTEN/NOTIFY then complete as no-ops over the wire
    broker: Option<Arc<NotifyBroker>>,
    /// this connection's backend pid, stamped into every NOTIFY it publishes
    /// so listeners learn who notified (PostgreSQL semantics)
    process_id: i32,
    /// the sender half handed to the broker on LISTEN; cloned per channel
    notify_tx: UnboundedSender<Notification>,
}

impl ConnState {
    /// A fresh [`QueryRequest`] keyed to this connection with the cancel
    /// token already threaded in — every engine call goes through here so
    /// the token reaches the engine uniformly (gate G12).
    fn request(&self) -> QueryRequest {
        let mut request = QueryRequest::new(self.connection_id);
        request.cancel = self.cancel.clone();
        request
    }
}

/// Process-unique connection ids, independent of the cancel registry so
/// directly-driven sessions (no registry) still get distinct identities.
static NEXT_CONNECTION_ID: AtomicU64 = AtomicU64::new(1);

fn resolve_format(codes: &[FormatCode], index: usize) -> FormatCode {
    match codes.len() {
        0 => FormatCode::Text,
        1 => codes[0],
        _ => codes.get(index).copied().unwrap_or_default(),
    }
}

fn tx_status_to_codec(status: TxStatus) -> TransactionStatus {
    match status {
        TxStatus::Idle => TransactionStatus::Idle,
        TxStatus::InTransaction => TransactionStatus::InTransaction,
        TxStatus::Failed => TransactionStatus::Failed,
    }
}

/// Builds the inbound `PgRequest` for a verb the engine matches on.
/// SQL text rides `query.sql`; the statement/portal name rides `path`.
fn build_request(method: &'static [u8], sql: &str, mut query: QueryRequest) -> PgRequest {
    use proxima_primitives::pipe::request::Request;
    query.sql = sql.to_owned();
    Request {
        method: proxima_primitives::pipe::method::Method::from_bytes(method),
        path: Bytes::from(query.statement.clone().into_bytes()),
        query: HeaderList::new(),
        metadata: HeaderList::new(),
        payload: query,
        stream: None,
        context: RequestContext::default(),
    }
}

/// Extracts the typed reply from an engine response.
fn downcast_reply(response: PgResponse) -> Result<PgReply, ServeError> {
    Ok(response.payload)
}

/// Encodes one `SqlValue` cell for the given result format code. The
/// column's `type_oid` drives the binary integer width — `SqlValue::Int`
/// is a single i64, but an int2/int4 column expects 2/4 big-endian bytes,
/// not 8. Returns `None` for SQL NULL.
fn encode_value(value: &SqlValue, format: FormatCode, type_oid: Oid) -> Option<Vec<u8>> {
    match value {
        SqlValue::Null => None,
        SqlValue::Int(number) => Some(match format {
            FormatCode::Binary => encode_binary_int(*number, type_oid),
            FormatCode::Text => number.to_string().into_bytes(),
        }),
        SqlValue::Float(number) => Some(match format {
            FormatCode::Binary => encode_binary_float(*number, type_oid),
            FormatCode::Text => number.to_string().into_bytes(),
        }),
        SqlValue::Bool(flag) => Some(match format {
            FormatCode::Binary => vec![u8::from(*flag)],
            FormatCode::Text => {
                if *flag {
                    b"t".to_vec()
                } else {
                    b"f".to_vec()
                }
            }
        }),
        SqlValue::Text(text) => Some(text.clone().into_bytes()),
        SqlValue::Bytes(bytes) => Some(bytes.clone()),
    }
}

fn encode_binary_int(number: i64, type_oid: Oid) -> Vec<u8> {
    match type_oid {
        Oid(21) => (number as i16).to_be_bytes().to_vec(),
        Oid(23) => (number as i32).to_be_bytes().to_vec(),
        _ => number.to_be_bytes().to_vec(),
    }
}

fn encode_binary_float(number: f64, type_oid: Oid) -> Vec<u8> {
    match type_oid {
        // float4 is 4 binary bytes; the engine yields f64, narrow to f32
        Oid(700) => (number as f32).to_be_bytes().to_vec(),
        _ => number.to_be_bytes().to_vec(),
    }
}

fn column_descriptor<'a>(column: &'a ColumnDesc, format: FormatCode) -> FieldDescription<'a> {
    FieldDescription {
        name: PgStr::new(column.name.as_bytes()),
        table_oid: 0,
        column_attr: 0,
        type_oid: column.type_oid,
        type_size: -1,
        type_modifier: -1,
        format,
    }
}

fn append_row_description(
    out: &mut Vec<u8>,
    columns: &[ColumnDesc],
    formats: Option<&[FormatCode]>,
) -> io::Result<()> {
    let body: usize = columns.iter().map(|column| column.name.len() + 19).sum();
    let start = reserve(out, 7 + body);
    let outcome = (|| {
        let mut writer = RowDescriptionWriter::begin(&mut out[start..])?;
        for (index, column) in columns.iter().enumerate() {
            let format = match formats {
                None => FormatCode::Text,
                Some(codes) => resolve_format(codes, index),
            };
            writer.field(&column_descriptor(column, format))?;
        }
        writer.finish()
    })();
    commit(out, start, outcome)
}

/// Emits the row-result prologue shared by the buffered and streaming paths:
/// any notices, the transaction-status update, and RowDescription (when
/// columns are present and the caller wants it — Describe owns it in the
/// extended path). Flushes past the high-water mark.
#[expect(
    clippy::too_many_arguments,
    reason = "shared row prologue spans the result metadata, the wire sink, and the session"
)]
async fn emit_rows_prologue<S>(
    columns: &[ColumnDesc],
    notices: &[crate::pipe_contract::NoticeReply],
    transaction: Option<TxStatus>,
    result_formats: &[FormatCode],
    stream: &mut S,
    out: &mut Vec<u8>,
    session: &mut Session,
    high_water: usize,
    has_row_description: bool,
) -> Result<(), ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    for notice in notices {
        append_notice(out, &ErrorInfo::from_notice(notice))?;
    }
    if let Some(status) = transaction {
        session.set_transaction_status(tx_status_to_codec(status));
    }
    if !columns.is_empty() && has_row_description {
        append_row_description(out, columns, Some(result_formats))?;
        flush_if_above(stream, out, high_water).await?;
    }
    Ok(())
}

/// Emits the CommandComplete that closes a row result: the engine's explicit
/// tag wins, else `SELECT n` from the drained row count. Shared by both paths.
async fn emit_command_complete<S>(
    command_tag: Option<&str>,
    row_count: u64,
    stream: &mut S,
    out: &mut Vec<u8>,
    high_water: usize,
) -> Result<(), ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    let tag = command_tag
        .map(str::to_owned)
        .unwrap_or_else(|| format!("SELECT {row_count}"));
    append_message(
        out,
        6 + tag.len(),
        &BackendMessage::CommandComplete {
            tag: PgStr::new(tag.as_bytes()),
        },
    )?;
    flush_if_above(stream, out, high_water).await?;
    Ok(())
}

/// Streams a `QueryReply` onto the wire: RowDescription (when columns are
/// present) + one DataRow per row, encoded per the portal's result format
/// codes, then CommandComplete. Flushes past the high-water mark so wide
/// result sets stream with bounded memory.
async fn emit_query_reply<S>(
    reply: &QueryReply,
    result_formats: &[FormatCode],
    stream: &mut S,
    out: &mut Vec<u8>,
    session: &mut Session,
    high_water: usize,
    has_row_description: bool,
) -> Result<(), ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    emit_rows_prologue(
        &reply.columns,
        &reply.notices,
        reply.transaction,
        result_formats,
        stream,
        out,
        session,
        high_water,
        has_row_description,
    )
    .await?;
    let mut row_count: u64 = 0;
    for row in &reply.rows {
        append_data_row(out, row, result_formats, &reply.columns)?;
        row_count += 1;
        flush_if_above(stream, out, high_water).await?;
    }
    emit_command_complete(
        reply.command_tag.as_deref(),
        row_count,
        stream,
        out,
        high_water,
    )
    .await
}

/// Streams a `PgReply::QueryStream` onto the wire: the same prologue +
/// per-row encode/flush + CommandComplete as [`emit_query_reply`], but rows
/// are drained from the [`RowStream`]'s channel (`recv().await` until the
/// sender closes) instead of iterating a buffered `Vec`. The driver never
/// collects the full result, so an unbounded engine result rides bounded
/// memory (only the high-water-bounded `out` buffer + one row at a time).
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors emit_query_reply; the column/tag pair replaces the reply struct"
)]
async fn emit_query_stream<S>(
    columns: &[ColumnDesc],
    rows: &RowStream,
    command_tag: Option<&str>,
    result_formats: &[FormatCode],
    stream: &mut S,
    out: &mut Vec<u8>,
    session: &mut Session,
    high_water: usize,
    has_row_description: bool,
) -> Result<(), ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    emit_rows_prologue(
        columns,
        &[],
        None,
        result_formats,
        stream,
        out,
        session,
        high_water,
        has_row_description,
    )
    .await?;
    let mut row_count: u64 = 0;
    while let Ok(row) = rows.receiver().recv().await {
        append_data_row(out, &row, result_formats, columns)?;
        row_count += 1;
        flush_if_above(stream, out, high_water).await?;
    }
    emit_command_complete(command_tag, row_count, stream, out, high_water).await
}

fn append_data_row(
    out: &mut Vec<u8>,
    row: &[SqlValue],
    formats: &[FormatCode],
    columns: &[ColumnDesc],
) -> io::Result<()> {
    let encoded: Vec<Option<Vec<u8>>> = row
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let type_oid = columns.get(index).map_or(Oid(0), |column| column.type_oid);
            encode_value(value, resolve_format(formats, index), type_oid)
        })
        .collect();
    let body: usize = encoded
        .iter()
        .map(|cell| 4 + cell.as_ref().map_or(0, Vec::len))
        .sum();
    let start = reserve(out, 7 + body);
    let outcome = (|| {
        let mut writer = DataRowWriter::begin(&mut out[start..])?;
        for cell in &encoded {
            match cell {
                Some(bytes) => writer.column(bytes)?,
                None => writer.null()?,
            };
        }
        writer.finish()
    })();
    commit(out, start, outcome)
}

async fn flush_if_above<S>(stream: &mut S, out: &mut Vec<u8>, high_water: usize) -> io::Result<()>
where
    S: AsyncWrite + Unpin + Send,
{
    if out.len() >= high_water {
        stream.write_all(out).await?;
        out.clear();
    }
    Ok(())
}

fn append_copy_out_response(
    out: &mut Vec<u8>,
    format: CopyFormat,
    column_formats: &[FormatCode],
) -> io::Result<()> {
    let start = reserve(out, 8 + 2 * column_formats.len());
    let outcome = encode_copy_out_response(&mut out[start..], format, column_formats);
    commit(out, start, outcome)
}

fn append_copy_in_response(
    out: &mut Vec<u8>,
    format: CopyFormat,
    column_formats: &[FormatCode],
) -> io::Result<()> {
    let start = reserve(out, 8 + 2 * column_formats.len());
    let outcome = encode_copy_in_response(&mut out[start..], format, column_formats);
    commit(out, start, outcome)
}

/// Streams a COPY OUT transfer: CopyOutResponse, one CopyData per row
/// payload (flushing past the high-water mark), CopyDone, then
/// `CommandComplete("COPY {n}")`. Drives the FSM through copy-out begun →
/// finished so the simple/extended tail proceeds normally.
async fn emit_copy_out<S>(
    format: CopyFormat,
    column_formats: &[FormatCode],
    data: &[Vec<u8>],
    stream: &mut S,
    out: &mut Vec<u8>,
    session: &mut Session,
    high_water: usize,
) -> Result<(), ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    session.copy_out_begun()?;
    append_copy_out_response(out, format, column_formats)?;
    flush_if_above(stream, out, high_water).await?;
    for payload in data {
        append_message(
            out,
            5 + payload.len(),
            &BackendMessage::CopyData { data: payload },
        )?;
        flush_if_above(stream, out, high_water).await?;
    }
    append_message(out, 5, &BackendMessage::CopyDone)?;
    session.copy_finished()?;
    let tag = format!("COPY {}", data.len());
    append_message(
        out,
        6 + tag.len(),
        &BackendMessage::CommandComplete {
            tag: PgStr::new(tag.as_bytes()),
        },
    )?;
    flush_if_above(stream, out, high_water).await?;
    Ok(())
}

/// Runs the COPY IN second phase: emit CopyInResponse, then read frontend
/// messages exactly like the main loop (buffering CopyData, stopping on
/// CopyDone/CopyFail), re-call the engine with the `COPY_DATA` verb, and
/// emit its CommandComplete (or ErrorResponse). Returns `false` when the
/// client terminated the connection mid-copy.
#[expect(
    clippy::too_many_arguments,
    reason = "copy-in spans the read loop and the engine re-call"
)]
async fn handle_copy_in<S>(
    format: CopyFormat,
    column_formats: &[FormatCode],
    sql: &str,
    connection_id: u64,
    cancel: &CancelToken,
    stream: &mut S,
    out: &mut Vec<u8>,
    buf: &mut BytesMut,
    session: &mut Session,
    query: &PgPipeHandle,
    config: &PgServerConfig,
) -> Result<bool, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    session.copy_in_begun()?;
    append_copy_in_response(out, format, column_formats)?;
    flush_out(stream, out).await?;

    let Some((rows, failed)) = collect_copy_data(stream, buf, session, config).await? else {
        return Ok(false);
    };

    let mut request = QueryRequest::new(connection_id);
    request.copy_data = rows;
    request.copy_failed = failed;
    request.cancel = cancel.clone();
    let response =
        SendPipe::call(query.as_ref(), build_request(verb::COPY_DATA, sql, request)).await?;
    match downcast_reply(response)? {
        PgReply::Query(reply) => {
            let tag = reply
                .command_tag
                .clone()
                .unwrap_or_else(|| "COPY 0".to_string());
            append_message(
                out,
                6 + tag.len(),
                &BackendMessage::CommandComplete {
                    tag: PgStr::new(tag.as_bytes()),
                },
            )?;
        }
        PgReply::Error(error) => append_error(out, &ErrorInfo::from_reply(&error))?,
        other => {
            return Err(ServeError::Config(format!(
                "sql pipe answered COPY_DATA with {} (contract violation)",
                copy_reply_kind(&other)
            )));
        }
    }
    Ok(true)
}

fn copy_reply_kind(reply: &PgReply) -> &'static str {
    match reply {
        PgReply::Query(_) => "Query",
        PgReply::QueryStream { .. } => "QueryStream",
        PgReply::Describe(_) => "Describe",
        PgReply::Error(_) => "Error",
        PgReply::CopyOut { .. } => "CopyOut",
        PgReply::CopyIn { .. } => "CopyIn",
        PgReply::Listen { .. } => "Listen",
        PgReply::Unlisten { .. } => "Unlisten",
        PgReply::Notify { .. } => "Notify",
    }
}

/// Applies a LISTEN engine reply: subscribes the connection on the shared
/// broker for each channel (a no-op without a broker, e.g. a directly-driven
/// session), then emits `CommandComplete("LISTEN")`.
fn apply_listen(out: &mut Vec<u8>, channels: &[String], state: &ConnState) -> io::Result<()> {
    if let Some(broker) = &state.broker {
        for channel in channels {
            broker.subscribe(channel, state.connection_id, state.notify_tx.clone());
        }
    }
    append_command_complete(out, "LISTEN")
}

/// Applies an UNLISTEN engine reply: unsubscribes the named channels, or
/// every channel when `all`, then `CommandComplete("UNLISTEN")`.
fn apply_unlisten(
    out: &mut Vec<u8>,
    channels: &[String],
    all: bool,
    state: &ConnState,
) -> io::Result<()> {
    if let Some(broker) = &state.broker {
        if all {
            broker.unsubscribe_all(state.connection_id);
        } else {
            for channel in channels {
                broker.unsubscribe(channel, state.connection_id);
            }
        }
    }
    append_command_complete(out, "UNLISTEN")
}

/// Applies a NOTIFY engine reply: publishes to every connection listening on
/// the channel, stamped with this connection's pid, then
/// `CommandComplete("NOTIFY")`. Self-notify is delivered (PostgreSQL
/// behavior); the receiving connection drains it at its next idle point.
fn apply_notify(
    out: &mut Vec<u8>,
    channel: &str,
    payload: &str,
    state: &ConnState,
) -> io::Result<()> {
    if let Some(broker) = &state.broker {
        broker.publish(
            channel,
            &Notification {
                process_id: state.process_id,
                channel: channel.to_owned(),
                payload: payload.to_owned(),
            },
        );
    }
    append_command_complete(out, "NOTIFY")
}

fn append_command_complete(out: &mut Vec<u8>, tag: &str) -> io::Result<()> {
    append_message(
        out,
        6 + tag.len(),
        &BackendMessage::CommandComplete {
            tag: PgStr::new(tag.as_bytes()),
        },
    )
}

/// Reads the client's COPY IN stream off the wire: each frontend message is
/// fed to the FSM and its disposition obeyed — CopyData payloads buffer,
/// CopyDone ends the stream, CopyFail aborts (returning the abort flag set),
/// Terminate closes the connection (`None`). Mirrors the main loop's buffer
/// and `max_message_bytes` handling.
async fn collect_copy_data<S>(
    stream: &mut S,
    buf: &mut BytesMut,
    session: &mut Session,
    config: &PgServerConfig,
) -> Result<Option<(Vec<Vec<u8>>, bool)>, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut rows: Vec<Vec<u8>> = Vec::new();
    let mut scratch = vec![0_u8; config.read_buffer_bytes];
    loop {
        let Some((message, consumed)) = parse_frontend(buf)? else {
            if buf.len() > config.max_message_bytes {
                return Err(ServeError::MessageTooLarge {
                    limit: config.max_message_bytes,
                });
            }
            if read_some(stream, buf, &mut scratch).await? == 0 {
                return Err(ServeError::UnexpectedEof);
            }
            continue;
        };
        // PostgreSQL ignores Flush/Sync during copy-in until CopyDone/CopyFail;
        // tokio-postgres pipelines a Sync right behind Execute, so skip them
        // without feeding the FSM (which rejects them in the CopyIn state)
        if matches!(message, FrontendMessage::Flush | FrontendMessage::Sync) {
            buf.advance(consumed);
            continue;
        }
        let disposition = session.on_frontend(&message)?;
        let outcome = match (&message, disposition) {
            (FrontendMessage::CopyData { data }, Disposition::Handle) => {
                rows.push(data.to_vec());
                None
            }
            (FrontendMessage::CopyDone, Disposition::Handle) => Some(Some((rows.clone(), false))),
            (FrontendMessage::CopyFail { .. }, Disposition::Handle) => {
                Some(Some((Vec::new(), true)))
            }
            (FrontendMessage::Terminate, Disposition::Handle) => Some(None),
            _ => None,
        };
        buf.advance(consumed);
        if let Some(result) = outcome {
            return Ok(result);
        }
    }
}

/// Decodes a bound parameter's raw wire bytes into a typed [`SqlValue`].
/// v1 keeps this lossless and conservative: binary int4/int8 decode to
/// `Int`, everything else rides as `Text`/`Bytes` and the engine
/// interprets it. NULL stays NULL.
fn decode_parameter(parameter: &BoundParameter, declared: Option<Oid>) -> SqlValue {
    let Some(bytes) = parameter.value.as_deref() else {
        return SqlValue::Null;
    };
    match parameter.format {
        FormatCode::Binary => decode_binary_parameter(bytes, declared),
        FormatCode::Text => match std::str::from_utf8(bytes) {
            Ok(text) => SqlValue::Text(text.to_owned()),
            Err(_) => SqlValue::Bytes(bytes.to_vec()),
        },
    }
}

fn decode_binary_parameter(bytes: &[u8], declared: Option<Oid>) -> SqlValue {
    match (declared, bytes.len()) {
        (Some(Oid(23)), 4) => SqlValue::Int(i64::from(i32::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
        ]))),
        (Some(Oid(20)), 8) => SqlValue::Int(i64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])),
        (Some(Oid(21)), 2) => SqlValue::Int(i64::from(i16::from_be_bytes([bytes[0], bytes[1]]))),
        _ => SqlValue::Bytes(bytes.to_vec()),
    }
}

/// Serves one authenticated-or-authenticating connection to completion.
/// `startup` comes from [`negotiate`]; the stream may be plaintext or
/// TLS-wrapped — the driver does not care. Each protocol operation is a
/// `Pipe::call` against `query`.
///
/// # Errors
/// [`ServeError`] on transport failure or protocol violations that the
/// wire could not express as an ErrorResponse.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the wire lifecycle one-to-one; bundling would invent a type with no other consumer"
)]
/// Behavior-preserving wrapper over [`serve_session_admitted`] for every
/// EXISTING caller (this crate's own test suite, `differential_realpg.rs`,
/// `client_smoke.rs`) — an unbounded, never-quiesced/-drained
/// [`proxima_listen::admission::ConnAdmission`], so none of those call
/// sites need to change. `PgWireAnyProtocol::drive` (the new
/// admission-aware caller) calls [`serve_session_admitted`] directly with
/// the listener's REAL shared handle instead.
pub async fn serve_session<S>(
    stream: S,
    session: Session,
    startup: StartupOwned,
    leftover: BytesMut,
    query: PgPipeHandle,
    auth: &PgAuth,
    config: &PgServerConfig,
    registry: Option<Arc<CancelRegistry>>,
    broker: Option<Arc<NotifyBroker>>,
    runtime: RuntimeHandle,
) -> Result<(), ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    serve_session_admitted(
        stream,
        session,
        startup,
        leftover,
        query,
        auth,
        config,
        registry,
        broker,
        runtime,
        proxima_listen::admission::ConnAdmission::unbounded(),
    )
    .await
}

/// Request-admission-aware sibling of [`serve_session`] — every frontend
/// message `main_loop` dispatches calls
/// [`proxima_listen::admission::ConnAdmission::request_admit`]/
/// `request_release` through `admission`, rendering an `ErrorResponse`
/// (SQLSTATE `57P03`, cannot_connect_now) instead of dispatching on `Shed`.
#[allow(clippy::too_many_arguments)]
pub async fn serve_session_admitted<S>(
    mut stream: S,
    mut session: Session,
    startup: StartupOwned,
    leftover: BytesMut,
    query: PgPipeHandle,
    auth: &PgAuth,
    config: &PgServerConfig,
    registry: Option<Arc<CancelRegistry>>,
    broker: Option<Arc<NotifyBroker>>,
    runtime: RuntimeHandle,
    admission: proxima_listen::admission::ConnAdmission,
) -> Result<(), ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut out = Vec::with_capacity(config.write_high_water_bytes + 4096);
    let mut buf = leftover;

    if startup.needs_negotiation {
        let options: Vec<&[u8]> = startup
            .negotiate_options
            .iter()
            .map(|option| option.as_bytes())
            .collect();
        let size = 13 + options.iter().map(|option| option.len() + 1).sum::<usize>();
        let start = reserve(&mut out, size);
        let outcome = encode_negotiate_protocol_version(&mut out[start..], 0, &options);
        commit(&mut out, start, outcome)?;
    }

    if !authenticate(
        &mut stream,
        &mut out,
        &mut buf,
        &mut session,
        &startup,
        auth,
        config,
        &runtime,
    )
    .await?
    {
        return Ok(());
    }

    append_message(
        &mut out,
        9,
        &BackendMessage::Authentication(AuthRequest::Ok),
    )?;
    for (name, value) in &config.parameters.0 {
        append_message(
            &mut out,
            7 + name.len() + value.len(),
            &BackendMessage::ParameterStatus {
                name: PgStr::new(name.as_bytes()),
                value: PgStr::new(value.as_bytes()),
            },
        )?;
    }
    // the registry registers the cancel flag (a matching CancelRequest sets
    // it); the driver keeps a handle and threads it into every engine
    // request as a CancelToken, so an interruptible engine can poll it
    // mid-query and abort cooperatively (gate G12)
    let (key, cancelled) = match &registry {
        Some(registry) => registry.register(),
        None => (
            BackendKey {
                process_id: 0,
                secret_key: 0,
            },
            Arc::new(AtomicBool::new(false)),
        ),
    };
    if registry.is_some() {
        append_message(
            &mut out,
            13,
            &BackendMessage::BackendKeyData {
                process_id: key.process_id,
                secret_key: &key.secret_key.to_be_bytes(),
            },
        )?;
    }
    let status = session.ready_for_query()?;
    append_message(&mut out, 6, &BackendMessage::ReadyForQuery { status })?;
    flush_out(&mut stream, &mut out).await?;

    let (notify_tx, notify_rx) = unbounded::<Notification>();
    let connection_id = NEXT_CONNECTION_ID.fetch_add(1, Ordering::Relaxed);
    let mut state = ConnState {
        statements: NamedSlots::new(config.max_statements),
        portals: NamedSlots::new(config.max_portals),
        connection_id,
        cancel: CancelToken::from(cancelled),
        broker: broker.clone(),
        process_id: key.process_id,
        notify_tx,
    };

    let outcome = main_loop(
        &mut stream,
        &mut out,
        &mut buf,
        &mut session,
        &mut state,
        &query,
        config,
        notify_rx,
        &admission,
    )
    .await;
    if let Some(registry) = &registry {
        registry.deregister(key.process_id);
    }
    if let Some(broker) = &broker {
        broker.unsubscribe_all(connection_id);
    }
    outcome
}

/// The one frontend message an auth flow waits for, owned past the read
/// buffer so each arm can act without holding a borrow.
enum AuthMessage {
    Password(String),
    #[cfg(feature = "scram")]
    SaslInitial {
        mechanism: String,
        data: Vec<u8>,
    },
    #[cfg(feature = "scram")]
    SaslResponse(Vec<u8>),
    Terminate,
}

/// Reads the next frontend message during authentication, feeding the
/// session FSM and returning an owned [`AuthMessage`]. Mirrors the buffer
/// and `max_message_bytes` handling of the simple-query loop.
async fn read_auth_message<S>(
    stream: &mut S,
    buf: &mut BytesMut,
    session: &mut Session,
    config: &PgServerConfig,
) -> Result<AuthMessage, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut scratch = [0_u8; 1024];
    loop {
        match parse_frontend(buf)? {
            None => {
                if buf.len() > config.max_message_bytes {
                    return Err(ServeError::MessageTooLarge {
                        limit: config.max_message_bytes,
                    });
                }
                if read_some(stream, buf, &mut scratch).await? == 0 {
                    return Err(ServeError::UnexpectedEof);
                }
            }
            Some((message, consumed)) => {
                session.on_frontend(&message)?;
                let owned = match &message {
                    FrontendMessage::AuthData(data) => auth_message_from(data, session)?,
                    FrontendMessage::Terminate => AuthMessage::Terminate,
                    _ => {
                        return Err(ServeError::Session(SessionError::IllegalMessage {
                            state: session.state_name(),
                            tag: message.tag(),
                        }));
                    }
                };
                buf.advance(consumed);
                return Ok(owned);
            }
        }
    }
}

fn auth_message_from(
    data: &proxima_protocols::pgwire_codec::frontend::AuthData<'_>,
    session: &Session,
) -> Result<AuthMessage, ServeError> {
    match session.auth_flow() {
        #[cfg(feature = "scram")]
        Some(proxima_protocols::pgwire_codec::AuthFlow::Sasl) => {
            // the first SASL message is a SASLInitialResponse (carries the
            // mechanism), every later one is a bare SASLResponse
            match data.as_sasl_initial() {
                Ok(initial) => {
                    let mechanism = initial
                        .mechanism
                        .to_str()
                        .map_err(|_| ServeError::InvalidUtf8 {
                            field: "sasl mechanism",
                        })?
                        .to_owned();
                    Ok(AuthMessage::SaslInitial {
                        mechanism,
                        data: initial.data.unwrap_or_default().to_vec(),
                    })
                }
                Err(_) => Ok(AuthMessage::SaslResponse(data.as_sasl_response().to_vec())),
            }
        }
        _ => {
            let password = data
                .as_password()?
                .to_str()
                .map_err(|_| ServeError::InvalidUtf8 { field: "password" })?
                .to_owned();
            Ok(AuthMessage::Password(password))
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the auth lifecycle; the runtime handle threads the kdf offload through"
)]
async fn authenticate<S>(
    stream: &mut S,
    out: &mut Vec<u8>,
    buf: &mut BytesMut,
    session: &mut Session,
    startup: &StartupOwned,
    auth: &PgAuth,
    config: &PgServerConfig,
    runtime: &RuntimeHandle,
) -> Result<bool, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    // the runtime is consumed only by the scram kdf offload; without that
    // feature the handle is threaded but unused
    #[cfg(not(feature = "scram"))]
    let _ = runtime;
    match auth {
        PgAuth::Trust => {
            session.auth_ok()?;
            Ok(true)
        }
        PgAuth::Cleartext(verifier) => {
            session.auth_requested(proxima_protocols::pgwire_codec::AuthFlow::Cleartext)?;
            append_message(
                out,
                9,
                &BackendMessage::Authentication(AuthRequest::CleartextPassword),
            )?;
            flush_out(stream, out).await?;
            match read_auth_message(stream, buf, session, config).await? {
                AuthMessage::Password(password) => {
                    if verifier.verify(&startup.user, startup.database.as_deref(), &password) {
                        session.auth_ok()?;
                        return Ok(true);
                    }
                    auth_reject(stream, out, session, &startup.user).await?;
                    Ok(false)
                }
                AuthMessage::Terminate => Ok(false),
                #[cfg(feature = "scram")]
                _ => Err(ServeError::Session(SessionError::IllegalMessage {
                    state: session.state_name(),
                    tag: b'p',
                })),
            }
        }
        #[cfg(feature = "md5-auth")]
        PgAuth::Md5(source) => {
            authenticate_md5(stream, out, buf, session, startup, source.as_ref(), config).await
        }
        #[cfg(feature = "scram")]
        PgAuth::Scram(source) => {
            authenticate_scram(
                stream,
                out,
                buf,
                session,
                startup,
                source.as_ref(),
                config,
                runtime,
            )
            .await
        }
        #[cfg(not(feature = "md5-auth"))]
        PgAuth::Md5(_) => Err(ServeError::Config(
            "md5 auth requires the md5-auth feature".into(),
        )),
        #[cfg(not(feature = "scram"))]
        PgAuth::Scram(_) => Err(ServeError::Config(
            "scram auth requires the scram feature".into(),
        )),
    }
}

async fn auth_reject<S>(
    stream: &mut S,
    out: &mut Vec<u8>,
    session: &mut Session,
    user: &str,
) -> Result<(), ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    session.auth_failed()?;
    append_error(out, &ErrorInfo::invalid_password(user).fatal())?;
    flush_out(stream, out).await?;
    Ok(())
}

#[cfg(feature = "md5-auth")]
async fn authenticate_md5<S>(
    stream: &mut S,
    out: &mut Vec<u8>,
    buf: &mut BytesMut,
    session: &mut Session,
    startup: &StartupOwned,
    source: &dyn crate::auth::PasswordSource,
    config: &PgServerConfig,
) -> Result<bool, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    session.auth_requested(proxima_protocols::pgwire_codec::AuthFlow::Md5)?;
    let salt: [u8; 4] = rand::random();
    append_message(
        out,
        13,
        &BackendMessage::Authentication(AuthRequest::Md5Password { salt }),
    )?;
    flush_out(stream, out).await?;
    match read_auth_message(stream, buf, session, config).await? {
        AuthMessage::Password(presented) => {
            let accepted = source.password_for(&startup.user).is_some_and(|password| {
                let expected = crate::md5::md5_password(&startup.user, password, salt);
                crate::auth::constant_time_eq(presented.as_bytes(), expected.as_bytes())
            });
            if accepted {
                session.auth_ok()?;
                return Ok(true);
            }
            auth_reject(stream, out, session, &startup.user).await?;
            Ok(false)
        }
        AuthMessage::Terminate => Ok(false),
        #[cfg(feature = "scram")]
        _ => Err(ServeError::Session(SessionError::IllegalMessage {
            state: session.state_name(),
            tag: b'p',
        })),
    }
}

/// Builds a `ScramServer` (running the 4096-iteration PBKDF2-HMAC-SHA256
/// key derivation) off the reactor core via the runtime's background
/// blocking pool when one is present; otherwise inline. The KDF is
/// ~0.5-1ms of pure CPU per auth — running it on the per-core async
/// reactor stalls that core, so with a runtime we route it through
/// `spawn_background_blocking` (the rayon-subsumed pool) and await the
/// `Send` handle, resuming on this core when the result is ready.
///
/// The outer `Result` is an infrastructure failure (the background pool
/// could not run the task); the inner `Result` is the KDF outcome the
/// caller maps to an auth rejection — preserving the inline path's
/// "SASLprep rejected the stored password → reject" behavior.
#[cfg(all(feature = "scram", feature = "listen"))]
async fn build_scram_server(
    runtime: &RuntimeHandle,
    password: &str,
) -> Result<Result<crate::scram::ScramServer, crate::scram::ScramError>, ServeError> {
    match runtime {
        Some(runtime) => {
            let password = password.to_owned();
            let handle = runtime.spawn_background_blocking(Box::new(move || {
                Ok(Box::new(crate::scram::ScramServer::new(&password))
                    as Box<dyn core::any::Any + Send>)
            }));
            let boxed = handle
                .await
                .map_err(|error| ServeError::BackgroundPool(error.to_string()))?;
            boxed
                .downcast::<Result<crate::scram::ScramServer, crate::scram::ScramError>>()
                .map(|result| *result)
                .map_err(|_| {
                    // contract: the closure boxes exactly this Result, so a
                    // type mismatch is unreachable; surfaced as an error
                    // rather than a panic per the no-panic discipline
                    ServeError::BackgroundPool(
                        "background scram task returned an unexpected type".into(),
                    )
                })
        }
        None => Ok(crate::scram::ScramServer::new(password)),
    }
}

/// Without the `listen` feature there is no runtime — the KDF runs inline
/// on whatever executor drives the session.
#[cfg(all(feature = "scram", not(feature = "listen")))]
async fn build_scram_server(
    _runtime: &RuntimeHandle,
    password: &str,
) -> Result<Result<crate::scram::ScramServer, crate::scram::ScramError>, ServeError> {
    Ok(crate::scram::ScramServer::new(password))
}

#[cfg(feature = "scram")]
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors the auth lifecycle; the runtime handle threads the kdf offload through"
)]
async fn authenticate_scram<S>(
    stream: &mut S,
    out: &mut Vec<u8>,
    buf: &mut BytesMut,
    session: &mut Session,
    startup: &StartupOwned,
    source: &dyn crate::auth::PasswordSource,
    config: &PgServerConfig,
    runtime: &RuntimeHandle,
) -> Result<bool, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    session.auth_requested(proxima_protocols::pgwire_codec::AuthFlow::Sasl)?;
    let mechanisms: [&[u8]; 1] = [b"SCRAM-SHA-256"];
    let size = 9 + mechanisms.iter().map(|name| name.len() + 1).sum::<usize>() + 1;
    let start = reserve(out, size);
    let outcome = proxima_protocols::pgwire_codec::backend::encode_auth_sasl(&mut out[start..], &mechanisms);
    commit(out, start, outcome)?;
    flush_out(stream, out).await?;

    let Some(password) = source.password_for(&startup.user).map(str::to_owned) else {
        // the user is unknown; we still run the exchange shape against a
        // throwaway password so timing does not distinguish the cases, then
        // reject on proof
        return scram_reject_unknown(stream, out, buf, session, startup, config, runtime).await;
    };

    let mut server = match build_scram_server(runtime, &password).await? {
        Ok(server) => server,
        Err(_) => {
            auth_reject(stream, out, session, &startup.user).await?;
            return Ok(false);
        }
    };

    let client_first = match read_auth_message(stream, buf, session, config).await? {
        AuthMessage::SaslInitial { mechanism, data } => {
            if mechanism != "SCRAM-SHA-256" {
                auth_reject(stream, out, session, &startup.user).await?;
                return Ok(false);
            }
            data
        }
        AuthMessage::Terminate => return Ok(false),
        _ => {
            auth_reject(stream, out, session, &startup.user).await?;
            return Ok(false);
        }
    };

    let server_first = match server.handle_client_first(&client_first) {
        Ok(bytes) => bytes,
        Err(_) => {
            auth_reject(stream, out, session, &startup.user).await?;
            return Ok(false);
        }
    };
    append_message(
        out,
        9 + server_first.len(),
        &BackendMessage::Authentication(AuthRequest::SaslContinue {
            data: &server_first,
        }),
    )?;
    flush_out(stream, out).await?;

    let client_final = match read_auth_message(stream, buf, session, config).await? {
        AuthMessage::SaslResponse(data) => data,
        AuthMessage::Terminate => return Ok(false),
        _ => {
            auth_reject(stream, out, session, &startup.user).await?;
            return Ok(false);
        }
    };

    match server.handle_client_final(&client_final) {
        Ok(server_final) => {
            append_message(
                out,
                9 + server_final.len(),
                &BackendMessage::Authentication(AuthRequest::SaslFinal {
                    data: &server_final,
                }),
            )?;
            session.auth_ok()?;
            Ok(true)
        }
        Err(_) => {
            auth_reject(stream, out, session, &startup.user).await?;
            Ok(false)
        }
    }
}

/// Runs the SASL message shape for an unknown user against a random
/// password so the failure is indistinguishable in timing from a wrong
/// password, then rejects.
#[cfg(feature = "scram")]
async fn scram_reject_unknown<S>(
    stream: &mut S,
    out: &mut Vec<u8>,
    buf: &mut BytesMut,
    session: &mut Session,
    startup: &StartupOwned,
    config: &PgServerConfig,
    runtime: &RuntimeHandle,
) -> Result<bool, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let decoy: [u8; 16] = rand::random();
    let decoy_password = base64::engine::general_purpose::STANDARD.encode(decoy);
    let mut server = match build_scram_server(runtime, &decoy_password).await? {
        Ok(server) => server,
        Err(_) => {
            auth_reject(stream, out, session, &startup.user).await?;
            return Ok(false);
        }
    };
    let client_first = match read_auth_message(stream, buf, session, config).await? {
        AuthMessage::SaslInitial { mechanism, data } => {
            // mirror the known-user path: a wrong mechanism rejects before
            // any decoy crypto, so unknown-user and known-user-wrong-mechanism
            // are timing-indistinguishable
            if mechanism != "SCRAM-SHA-256" {
                auth_reject(stream, out, session, &startup.user).await?;
                return Ok(false);
            }
            data
        }
        AuthMessage::Terminate => return Ok(false),
        _ => {
            auth_reject(stream, out, session, &startup.user).await?;
            return Ok(false);
        }
    };
    if let Ok(server_first) = server.handle_client_first(&client_first) {
        append_message(
            out,
            9 + server_first.len(),
            &BackendMessage::Authentication(AuthRequest::SaslContinue {
                data: &server_first,
            }),
        )?;
        flush_out(stream, out).await?;
        if let AuthMessage::SaslResponse(client_final) =
            read_auth_message(stream, buf, session, config).await?
        {
            let _ = server.handle_client_final(&client_final);
        }
    }
    auth_reject(stream, out, session, &startup.user).await?;
    Ok(false)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the notification receiver joins the wire lifecycle the loop already drives"
)]
async fn main_loop<S>(
    stream: &mut S,
    out: &mut Vec<u8>,
    buf: &mut BytesMut,
    session: &mut Session,
    state: &mut ConnState,
    query: &PgPipeHandle,
    config: &PgServerConfig,
    mut notify_rx: UnboundedReceiver<Notification>,
    admission: &proxima_listen::admission::ConnAdmission,
) -> Result<(), ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let mut scratch = vec![0_u8; config.read_buffer_bytes];
    loop {
        loop {
            if session.is_closed() {
                flush_out(stream, out).await?;
                return Ok(());
            }
            let parsed = match parse_frontend(buf) {
                Ok(parsed) => parsed,
                Err(error) => {
                    append_error(out, &protocol_violation(error))?;
                    flush_out(stream, out).await?;
                    session.fail();
                    return Ok(());
                }
            };
            let Some((message, consumed)) = parsed else {
                break;
            };
            let disposition = match session.on_frontend(&message) {
                Ok(disposition) => disposition,
                Err(SessionError::IllegalMessage {
                    state: state_name,
                    tag,
                }) => {
                    append_error(
                        out,
                        &protocol_violation(format!(
                            "message {:?} not allowed in state {state_name:?}",
                            char::from(tag)
                        )),
                    )?;
                    flush_out(stream, out).await?;
                    session.fail();
                    return Ok(());
                }
                Err(error) => return Err(error.into()),
            };
            let outcome = if disposition == Disposition::Handle {
                dispatch(&message, stream, out, session, state, query, config, admission).await?
            } else {
                DispatchOutcome::Done
            };
            buf.advance(consumed);
            if let DispatchOutcome::CopyIn(pending) = outcome {
                let proceed = handle_copy_in(
                    pending.format,
                    &pending.column_formats,
                    &pending.sql,
                    state.connection_id,
                    &state.cancel,
                    stream,
                    out,
                    buf,
                    session,
                    query,
                    config,
                )
                .await?;
                if !proceed {
                    flush_out(stream, out).await?;
                    return Ok(());
                }
                if pending.finish_simple {
                    finish_simple(stream, out, session).await?;
                }
            }
        }
        if buf.len() > config.max_message_bytes {
            append_error(
                out,
                &ErrorInfo::new(
                    "54000",
                    format!("message exceeds {} byte limit", config.max_message_bytes),
                )
                .fatal(),
            )?;
            flush_out(stream, out).await?;
            return Err(ServeError::MessageTooLarge {
                limit: config.max_message_bytes,
            });
        }
        // the safe delivery point: the inner loop drained `buf` (no partial
        // message buffered) AND the FSM is Idle (reached only via
        // ready_for_query, i.e. just after ReadyForQuery — never mid simple
        // or extended sequence). Only here do we race notifications against
        // the socket read; mid-sequence (state Extended/Syncing/...) we read
        // the socket alone and notifications stay queued in the unbounded
        // channel until the next idle pass.
        let read = if session.state_name() == StateName::Idle {
            idle_read(stream, buf, &mut scratch, out, &mut notify_rx).await?
        } else {
            read_some(stream, buf, &mut scratch).await?
        };
        if read == 0 {
            if buf.is_empty() {
                return Ok(());
            }
            return Err(ServeError::UnexpectedEof);
        }
    }
}

/// Waits at the idle safe point for either client bytes or a notification.
/// A notification is encoded as NotificationResponse and flushed, then the
/// wait resumes — so any number of notifications drain before the next
/// client message, and none ever interleaves a message sequence (this only
/// runs when the FSM is Idle). Returns the count of client bytes read (0 on
/// peer close); a notification returns a non-zero sentinel so the caller's
/// EOF check does not fire while bytes are still pending.
async fn idle_read<S>(
    stream: &mut S,
    buf: &mut BytesMut,
    scratch: &mut [u8],
    out: &mut Vec<u8>,
    notify_rx: &mut UnboundedReceiver<Notification>,
) -> Result<usize, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    loop {
        futures::select_biased! {
            notification = notify_rx.next().fuse() => {
                let Some(notification) = notification else {
                    // the sender is held by this same connection's state, so
                    // None means the connection is tearing down; fall back to
                    // a plain read so the loop's EOF handling takes over
                    return read_some(stream, buf, scratch).await;
                };
                emit_notification(stream, out, &notification).await?;
            }
            read = read_some(stream, buf, scratch).fuse() => {
                return read;
            }
        }
    }
}

/// Encodes one NotificationResponse and flushes it. The `process_id` on the
/// wire is the *notifying* connection's pid as carried on the
/// [`Notification`], not this listener's pid.
async fn emit_notification<S>(
    stream: &mut S,
    out: &mut Vec<u8>,
    notification: &Notification,
) -> Result<(), ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    let size = 11 + notification.channel.len() + notification.payload.len();
    append_message(
        out,
        size,
        &BackendMessage::NotificationResponse {
            process_id: notification.process_id,
            channel: PgStr::new(notification.channel.as_bytes()),
            payload: PgStr::new(notification.payload.as_bytes()),
        },
    )?;
    flush_out(stream, out).await?;
    Ok(())
}

/// A COPY IN transfer the engine signaled but the driver must run after
/// the triggering message is consumed from `buf` (the sub-loop reads more
/// frontend messages, so it cannot run while the trigger still borrows the
/// buffer). `finish` distinguishes the simple tail (ReadyForQuery) from the
/// extended tail.
struct PendingCopyIn {
    format: CopyFormat,
    column_formats: Vec<FormatCode>,
    sql: String,
    finish_simple: bool,
}

/// What `dispatch` asks the main loop to do after the trigger message is
/// consumed: nothing, or run a deferred COPY IN sub-loop.
enum DispatchOutcome {
    Done,
    CopyIn(PendingCopyIn),
}

#[allow(clippy::too_many_arguments)]
async fn dispatch<S>(
    message: &FrontendMessage<'_>,
    stream: &mut S,
    out: &mut Vec<u8>,
    session: &mut Session,
    state: &mut ConnState,
    query: &PgPipeHandle,
    config: &PgServerConfig,
    admission: &proxima_listen::admission::ConnAdmission,
) -> Result<DispatchOutcome, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    match message {
        FrontendMessage::Query { sql } => {
            state.portals.remove("");
            let Ok(sql_text) = sql.to_str() else {
                append_error(
                    out,
                    &ErrorInfo::new("08P01", "query text is not valid utf-8"),
                )?;
                finish_simple(stream, out, session).await?;
                return Ok(DispatchOutcome::Done);
            };
            if sql_text.trim().is_empty() {
                append_message(out, 5, &BackendMessage::EmptyQueryResponse)?;
                finish_simple(stream, out, session).await?;
                return Ok(DispatchOutcome::Done);
            }
            // Request-level admission: the simple-query boundary is
            // pgwire's natural "one request" unit (mirrors h1 per request,
            // h2 per stream, redis per command). On `Shed` the query never
            // reaches the engine; the listener's admission policy (quiesce
            // window, hard drain, or a configured in-flight cap) renders as
            // a real `ErrorResponse` (57P03, cannot_connect_now) instead —
            // the connection stays alive and ready for the next query.
            if let proxima_listen::admission::RequestAdmit::Shed { reason } =
                admission.request_admit()
            {
                append_error(
                    out,
                    &ErrorInfo::new(
                        "57P03",
                        format!("server is shedding requests ({reason:?}); retry shortly"),
                    ),
                )?;
                finish_simple(stream, out, session).await?;
                return Ok(DispatchOutcome::Done);
            }
            let request = build_request(verb::QUERY, sql_text, state.request());
            let response = SendPipe::call(query.as_ref(), request).await;
            admission.request_release();
            let response = response?;
            match downcast_reply(response)? {
                PgReply::Query(reply) => {
                    emit_query_reply(
                        &reply,
                        &[],
                        stream,
                        out,
                        session,
                        config.write_high_water_bytes,
                        true,
                    )
                    .await?;
                }
                PgReply::QueryStream {
                    columns,
                    rows,
                    command_tag,
                } => {
                    emit_query_stream(
                        &columns,
                        &rows,
                        command_tag.as_deref(),
                        &[],
                        stream,
                        out,
                        session,
                        config.write_high_water_bytes,
                        true,
                    )
                    .await?;
                }
                PgReply::Error(error) => append_error(out, &ErrorInfo::from_reply(&error))?,
                PgReply::CopyOut {
                    format,
                    column_formats,
                    data,
                } => {
                    emit_copy_out(
                        format,
                        &column_formats,
                        &data,
                        stream,
                        out,
                        session,
                        config.write_high_water_bytes,
                    )
                    .await?;
                }
                PgReply::CopyIn {
                    format,
                    column_formats,
                } => {
                    return Ok(DispatchOutcome::CopyIn(PendingCopyIn {
                        format,
                        column_formats,
                        sql: sql_text.to_owned(),
                        finish_simple: true,
                    }));
                }
                PgReply::Listen { channels } => apply_listen(out, &channels, state)?,
                PgReply::Unlisten { channels, all } => {
                    apply_unlisten(out, &channels, all, state)?;
                }
                PgReply::Notify { channel, payload } => {
                    apply_notify(out, &channel, &payload, state)?;
                }
                PgReply::Describe(_) => {
                    return Err(ServeError::Config(
                        "sql pipe answered QUERY with Describe (contract violation)".into(),
                    ));
                }
            }
            finish_simple(stream, out, session).await?;
        }
        FrontendMessage::Parse(parse) => {
            handle_parse(parse, out, session, state, query).await?;
        }
        FrontendMessage::Bind(bind) => {
            handle_bind(bind, out, session, state)?;
        }
        FrontendMessage::Describe { target, name } => {
            handle_describe(*target, *name, out, session, state)?;
        }
        FrontendMessage::Execute { portal, max_rows } => {
            return handle_execute(
                *portal, *max_rows, stream, out, session, state, query, config,
            )
            .await;
        }
        FrontendMessage::Close { target, name } => {
            let Ok(name_text) = name.to_str() else {
                extended_fail(
                    out,
                    session,
                    &ErrorInfo::new("08P01", "close target name is not valid utf-8"),
                )?;
                return Ok(DispatchOutcome::Done);
            };
            match target {
                StatementTarget::Statement => {
                    if state.statements.remove(name_text) {
                        let owned = name_text.to_string();
                        state
                            .portals
                            .remove_where(|portal| portal.statement_name == owned);
                    }
                }
                StatementTarget::Portal => {
                    state.portals.remove(name_text);
                }
            }
            append_message(out, 5, &BackendMessage::CloseComplete)?;
        }
        FrontendMessage::Flush => {
            flush_out(stream, out).await?;
        }
        FrontendMessage::Sync => {
            if session.transaction_status() == TransactionStatus::Idle {
                state.portals.clear();
            }
            let status = session.ready_for_query()?;
            append_message(out, 6, &BackendMessage::ReadyForQuery { status })?;
            flush_out(stream, out).await?;
        }
        FrontendMessage::FunctionCall(call) => {
            handle_function_call(call, stream, out, session)?;
            finish_simple(stream, out, session).await?;
        }
        FrontendMessage::Terminate => {}
        FrontendMessage::AuthData(_)
        | FrontendMessage::CopyData { .. }
        | FrontendMessage::CopyDone
        | FrontendMessage::CopyFail { .. } => {
            // the FSM only yields Handle for these inside flows the facade
            // starts; COPY IN runs its own sub-loop (handle_copy_in), so a
            // COPY message reaching the main dispatch is a facade bug, not a
            // peer error
            return Err(ServeError::Session(SessionError::IllegalMessage {
                state: session.state_name(),
                tag: message.tag(),
            }));
        }
    }
    Ok(DispatchOutcome::Done)
}

async fn finish_simple<S: AsyncRead + AsyncWrite + Unpin + Send>(
    stream: &mut S,
    out: &mut Vec<u8>,
    session: &mut Session,
) -> Result<(), ServeError> {
    let status = session.ready_for_query()?;
    append_message(out, 6, &BackendMessage::ReadyForQuery { status })?;
    flush_out(stream, out).await?;
    Ok(())
}

/// Reports an error inside the extended pipeline and arms
/// discard-until-Sync recovery.
fn extended_fail(
    out: &mut Vec<u8>,
    session: &mut Session,
    info: &ErrorInfo,
) -> Result<(), ServeError> {
    append_error(out, info)?;
    session.extended_error()?;
    Ok(())
}

async fn handle_parse(
    parse: &ParseMessage<'_>,
    out: &mut Vec<u8>,
    session: &mut Session,
    state: &mut ConnState,
    query: &PgPipeHandle,
) -> Result<(), ServeError> {
    let (Ok(name), Ok(sql)) = (parse.statement.to_str(), parse.sql.to_str()) else {
        extended_fail(
            out,
            session,
            &ErrorInfo::new("08P01", "parse message is not valid utf-8"),
        )?;
        return Ok(());
    };
    let declared: Vec<Oid> = parse.parameter_types.iter().collect();
    let is_empty_query = sql.trim().is_empty();
    let reply = if is_empty_query {
        DescribeReply::default()
    } else {
        let mut request = state.request();
        request.statement = name.to_string();
        let response =
            SendPipe::call(query.as_ref(), build_request(verb::PARSE, sql, request)).await?;
        match downcast_reply(response)? {
            PgReply::Describe(reply) => reply,
            PgReply::Error(error) => {
                extended_fail(out, session, &ErrorInfo::from_reply(&error))?;
                return Ok(());
            }
            other => {
                return Err(ServeError::Config(format!(
                    "sql pipe answered PARSE with {} (contract violation)",
                    copy_reply_kind(&other)
                )));
            }
        }
    };
    let parameter_types = if reply.parameter_types.is_empty() {
        declared
    } else {
        reply.parameter_types
    };
    let statement = PreparedStatement {
        sql: sql.to_string(),
        parameter_types,
        columns: reply.columns,
        is_empty_query,
    };
    match state.statements.insert(name, statement) {
        Ok(()) => append_message(out, 5, &BackendMessage::ParseComplete)?,
        Err(StoreError::Duplicate) => extended_fail(
            out,
            session,
            &ErrorInfo::new(
                "42P05",
                format!("prepared statement \"{name}\" already exists"),
            ),
        )?,
        Err(StoreError::Full) => extended_fail(
            out,
            session,
            &ErrorInfo::new("53000", "prepared statement slots exhausted"),
        )?,
    }
    Ok(())
}

fn handle_bind(
    bind: &Bind<'_>,
    out: &mut Vec<u8>,
    session: &mut Session,
    state: &mut ConnState,
) -> Result<(), ServeError> {
    let (Ok(portal_name), Ok(statement_name)) = (bind.portal.to_str(), bind.statement.to_str())
    else {
        extended_fail(
            out,
            session,
            &ErrorInfo::new("08P01", "bind message is not valid utf-8"),
        )?;
        return Ok(());
    };
    let Some(statement) = state.statements.get(statement_name) else {
        extended_fail(
            out,
            session,
            &ErrorInfo::new(
                "26000",
                format!("prepared statement \"{statement_name}\" does not exist"),
            ),
        )?;
        return Ok(());
    };
    let parameter_count = bind.parameters.len();
    // an empty declared list means the engine left parameter types
    // unspecified (it validates at execute); count is enforced only
    // against an explicit declaration
    if !statement.parameter_types.is_empty() && parameter_count != statement.parameter_types.len() {
        let expected = statement.parameter_types.len();
        extended_fail(
            out,
            session,
            &ErrorInfo::new(
                "08P01",
                format!("bind supplies {parameter_count} parameters, statement needs {expected}"),
            ),
        )?;
        return Ok(());
    }
    let format_count = bind.parameter_formats.len();
    if format_count > 1 && format_count != parameter_count {
        extended_fail(
            out,
            session,
            &ErrorInfo::new(
                "08P01",
                "parameter format count does not match parameter count",
            ),
        )?;
        return Ok(());
    }
    let result_format_count = bind.result_formats.len();
    if result_format_count > 1 && result_format_count != statement.columns.len() {
        extended_fail(
            out,
            session,
            &ErrorInfo::new(
                "08P01",
                "result format count does not match result column count",
            ),
        )?;
        return Ok(());
    }
    let parameters: Vec<BoundParameter> = bind
        .parameters
        .iter()
        .enumerate()
        .map(|(index, value)| BoundParameter {
            value: value.map(<[u8]>::to_vec),
            format: bind.parameter_format(index),
        })
        .collect();
    let result_formats: Vec<FormatCode> = bind.result_formats.iter().collect();
    let portal = Portal {
        statement_name: statement_name.to_string(),
        parameters,
        result_formats,
        pending: None,
    };
    match state.portals.insert(portal_name, portal) {
        Ok(()) => append_message(out, 5, &BackendMessage::BindComplete)?,
        Err(StoreError::Duplicate) => extended_fail(
            out,
            session,
            &ErrorInfo::new("42P03", format!("portal \"{portal_name}\" already exists")),
        )?,
        Err(StoreError::Full) => {
            extended_fail(
                out,
                session,
                &ErrorInfo::new("53000", "portal slots exhausted"),
            )?;
        }
    }
    Ok(())
}

fn handle_describe(
    target: StatementTarget,
    name: PgStr<'_>,
    out: &mut Vec<u8>,
    session: &mut Session,
    state: &ConnState,
) -> Result<(), ServeError> {
    let Ok(name_text) = name.to_str() else {
        extended_fail(
            out,
            session,
            &ErrorInfo::new("08P01", "describe name is not valid utf-8"),
        )?;
        return Ok(());
    };
    match target {
        StatementTarget::Statement => {
            let Some(statement) = state.statements.get(name_text) else {
                extended_fail(
                    out,
                    session,
                    &ErrorInfo::new(
                        "26000",
                        format!("prepared statement \"{name_text}\" does not exist"),
                    ),
                )?;
                return Ok(());
            };
            let oids = statement.parameter_types.clone();
            let start = reserve(out, 7 + 4 * oids.len());
            let outcome = encode_parameter_description(&mut out[start..], &oids);
            commit(out, start, outcome)?;
            if statement.columns.is_empty() {
                append_message(out, 5, &BackendMessage::NoData)?;
            } else {
                append_row_description(out, &statement.columns, None)?;
            }
        }
        StatementTarget::Portal => {
            let Some(portal) = state.portals.get(name_text) else {
                extended_fail(
                    out,
                    session,
                    &ErrorInfo::new("34000", format!("portal \"{name_text}\" does not exist")),
                )?;
                return Ok(());
            };
            let Some(statement) = state.statements.get(&portal.statement_name) else {
                extended_fail(
                    out,
                    session,
                    &ErrorInfo::new("26000", "portal references a closed statement"),
                )?;
                return Ok(());
            };
            if statement.columns.is_empty() {
                append_message(out, 5, &BackendMessage::NoData)?;
            } else {
                let formats = portal.result_formats.clone();
                append_row_description(out, &statement.columns, Some(&formats))?;
            }
        }
    }
    Ok(())
}

/// `max_rows <= 0` means unlimited (the protocol treats 0 as "fetch all";
/// a negative value is malformed and PostgreSQL also treats it as no cap).
fn execute_limit(max_rows: i32) -> Option<usize> {
    usize::try_from(max_rows).ok().filter(|limit| *limit > 0)
}

/// Streams up to `limit` (or all, when `None`) rows from the pending
/// buffer, advancing its cursor and running emitted count. Returns whether
/// rows remain — the caller decides PortalSuspended vs CommandComplete.
async fn stream_batch<S>(
    pending: &mut PendingRows,
    limit: Option<usize>,
    stream: &mut S,
    out: &mut Vec<u8>,
    high_water: usize,
) -> Result<bool, ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    let end = match limit {
        Some(count) => (pending.cursor + count).min(pending.rows.len()),
        None => pending.rows.len(),
    };
    for row in &pending.rows[pending.cursor..end] {
        append_data_row(out, row, &pending.result_formats, &pending.columns)?;
        pending.emitted += 1;
        flush_if_above(stream, out, high_water).await?;
    }
    pending.cursor = end;
    Ok(pending.cursor < pending.rows.len())
}

#[expect(
    clippy::too_many_arguments,
    reason = "execute touches every connection surface once"
)]
async fn handle_execute<S>(
    portal_name: PgStr<'_>,
    max_rows: i32,
    stream: &mut S,
    out: &mut Vec<u8>,
    session: &mut Session,
    state: &mut ConnState,
    query: &PgPipeHandle,
    config: &PgServerConfig,
) -> Result<DispatchOutcome, ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    let Ok(name_text) = portal_name.to_str() else {
        extended_fail(
            out,
            session,
            &ErrorInfo::new("08P01", "portal name is not valid utf-8"),
        )?;
        return Ok(DispatchOutcome::Done);
    };
    if state.portals.get(name_text).is_none() {
        extended_fail(
            out,
            session,
            &ErrorInfo::new("34000", format!("portal \"{name_text}\" does not exist")),
        )?;
        return Ok(DispatchOutcome::Done);
    }
    let limit = execute_limit(max_rows);
    if state
        .portals
        .get(name_text)
        .is_some_and(|portal| portal.pending.is_some())
    {
        resume_execute(name_text, limit, stream, out, state, config).await?;
        return Ok(DispatchOutcome::Done);
    }
    first_execute(name_text, limit, stream, out, session, state, query, config).await
}

/// Resumes a suspended portal: streams the next batch from its buffered
/// rows with no engine call, then PortalSuspended (more remain) or
/// CommandComplete + clears `pending`.
async fn resume_execute<S>(
    name_text: &str,
    limit: Option<usize>,
    stream: &mut S,
    out: &mut Vec<u8>,
    state: &mut ConnState,
    config: &PgServerConfig,
) -> Result<(), ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    let Some(portal) = state.portals.get_mut(name_text) else {
        return Ok(());
    };
    let Some(pending) = portal.pending.as_mut() else {
        return Ok(());
    };
    let suspended =
        stream_batch(pending, limit, stream, out, config.write_high_water_bytes).await?;
    if suspended {
        append_message(out, 5, &BackendMessage::PortalSuspended)?;
        return Ok(());
    }
    let tag = pending
        .command_tag
        .clone()
        .unwrap_or_else(|| format!("SELECT {}", pending.emitted));
    append_message(
        out,
        6 + tag.len(),
        &BackendMessage::CommandComplete {
            tag: PgStr::new(tag.as_bytes()),
        },
    )?;
    portal.pending = None;
    Ok(())
}

/// First Execute of a portal with no pending: calls the engine, then emits
/// all rows + CommandComplete (no cap, or the reply fits the cap) or the
/// first batch + PortalSuspended (more rows than the cap), buffering the
/// remainder on the portal.
#[expect(
    clippy::too_many_arguments,
    reason = "execute touches every connection surface once"
)]
async fn first_execute<S>(
    name_text: &str,
    limit: Option<usize>,
    stream: &mut S,
    out: &mut Vec<u8>,
    session: &mut Session,
    state: &mut ConnState,
    query: &PgPipeHandle,
    config: &PgServerConfig,
) -> Result<DispatchOutcome, ServeError>
where
    S: AsyncWrite + Unpin + Send,
{
    let Some(portal) = state.portals.get(name_text) else {
        return Ok(DispatchOutcome::Done);
    };
    let Some(statement) = state.statements.get(&portal.statement_name) else {
        extended_fail(
            out,
            session,
            &ErrorInfo::new("26000", "portal references a closed statement"),
        )?;
        return Ok(DispatchOutcome::Done);
    };
    if statement.is_empty_query {
        append_message(out, 5, &BackendMessage::EmptyQueryResponse)?;
        return Ok(DispatchOutcome::Done);
    }
    let mut request = state.request();
    request.statement = portal.statement_name.clone();
    request.portal = name_text.to_string();
    request.parameters = portal
        .parameters
        .iter()
        .enumerate()
        .map(|(index, parameter)| {
            decode_parameter(parameter, statement.parameter_types.get(index).copied())
        })
        .collect();
    let sql = statement.sql.clone();
    let result_formats = portal.result_formats.clone();
    let response =
        SendPipe::call(query.as_ref(), build_request(verb::EXECUTE, &sql, request)).await?;
    let reply = match downcast_reply(response)? {
        PgReply::Query(reply) => reply,
        PgReply::QueryStream {
            columns,
            rows,
            command_tag,
        } => {
            // portal suspension (G4) over a lazy stream is out of scope for
            // v1: a cursor would have to buffer the drained tail, defeating
            // the bounded-memory point. The honest simpler behavior is to
            // stream the whole result and ignore max_rows (no PortalSuspended).
            emit_query_stream(
                &columns,
                &rows,
                command_tag.as_deref(),
                &result_formats,
                stream,
                out,
                session,
                config.write_high_water_bytes,
                false,
            )
            .await?;
            return Ok(DispatchOutcome::Done);
        }
        PgReply::Error(error) => {
            extended_fail(out, session, &ErrorInfo::from_reply(&error))?;
            return Ok(DispatchOutcome::Done);
        }
        PgReply::CopyOut {
            format,
            column_formats,
            data,
        } => {
            emit_copy_out(
                format,
                &column_formats,
                &data,
                stream,
                out,
                session,
                config.write_high_water_bytes,
            )
            .await?;
            return Ok(DispatchOutcome::Done);
        }
        PgReply::CopyIn {
            format,
            column_formats,
        } => {
            return Ok(DispatchOutcome::CopyIn(PendingCopyIn {
                format,
                column_formats,
                sql,
                finish_simple: false,
            }));
        }
        PgReply::Listen { channels } => {
            apply_listen(out, &channels, state)?;
            return Ok(DispatchOutcome::Done);
        }
        PgReply::Unlisten { channels, all } => {
            apply_unlisten(out, &channels, all, state)?;
            return Ok(DispatchOutcome::Done);
        }
        PgReply::Notify { channel, payload } => {
            apply_notify(out, &channel, &payload, state)?;
            return Ok(DispatchOutcome::Done);
        }
        PgReply::Describe(_) => {
            return Err(ServeError::Config(
                "sql pipe answered EXECUTE with Describe (contract violation)".into(),
            ));
        }
    };

    let fits = limit.is_none_or(|count| reply.rows.len() <= count);
    if fits {
        emit_query_reply(
            &reply,
            &result_formats,
            stream,
            out,
            session,
            config.write_high_water_bytes,
            false,
        )
        .await?;
        return Ok(DispatchOutcome::Done);
    }

    for notice in &reply.notices {
        append_notice(out, &ErrorInfo::from_notice(notice))?;
    }
    if let Some(status) = reply.transaction {
        session.set_transaction_status(tx_status_to_codec(status));
    }
    let QueryReply {
        columns,
        rows,
        command_tag,
        ..
    } = reply;
    let mut pending = PendingRows {
        columns,
        rows,
        cursor: 0,
        result_formats,
        emitted: 0,
        command_tag,
    };
    stream_batch(
        &mut pending,
        limit,
        stream,
        out,
        config.write_high_water_bytes,
    )
    .await?;
    append_message(out, 5, &BackendMessage::PortalSuspended)?;
    if let Some(portal) = state.portals.get_mut(name_text) {
        portal.pending = Some(pending);
    }
    Ok(DispatchOutcome::Done)
}

/// Answers the legacy fast-path function call directly: the contract has
/// no verb for it, so the driver returns the typed `0A000` refusal that
/// the old `on_function_call` default produced.
fn handle_function_call<S>(
    call: &FunctionCall<'_>,
    _stream: &mut S,
    out: &mut Vec<u8>,
    _session: &mut Session,
) -> Result<(), ServeError>
where
    S: AsyncRead + AsyncWrite + Unpin + Send,
{
    append_error(
        out,
        &ErrorInfo::feature_not_supported(format!(
            "fast-path function call {} is not supported by this server",
            call.object.0
        )),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::pin::Pin;
    use std::task::{Context, Poll};

    use futures::io::{AsyncRead, AsyncWrite};
    use proxima_core::ProximaError;
    use proxima_protocols::pgwire_codec::backend::parse_backend;
    use proxima_protocols::pgwire_codec::types::error_field;
    use proxima_protocols::pgwire_codec::{BackendMessage, Oid, Session};
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::request::Response;

    use crate::auth::PgAuth;
    use crate::config::PgServerConfig;
    use crate::error::ServeError;
    use crate::pipe_contract::{ColumnDesc, DescribeReply, PgReply, QueryReply, SqlValue, verb};
    use crate::pipes::{PgPipeHandle, PgRequest, PgResponse, into_pg_handle};

    use super::*;

    struct TestStream {
        read_data: std::io::Cursor<Vec<u8>>,
        write_data: Vec<u8>,
    }

    impl TestStream {
        fn new(input: Vec<u8>) -> Self {
            Self {
                read_data: std::io::Cursor::new(input),
                write_data: Vec::new(),
            }
        }

        fn written(&self) -> &[u8] {
            &self.write_data
        }
    }

    impl AsyncRead for TestStream {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            use std::io::Read;
            Poll::Ready(self.read_data.read(buf))
        }
    }

    impl AsyncWrite for TestStream {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            self.write_data.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    /// A fake SQL engine that answers the verb contract: QUERY "select 1"
    /// and EXECUTE yield one int column one row; PARSE yields a Describe.
    struct EchoPipe;

    impl SendPipe for EchoPipe {
        type In = PgRequest;
        type Out = PgResponse;
        type Err = ProximaError;

        async fn call(&self, request: PgRequest) -> Result<PgResponse, ProximaError> {
            let sql = request.payload.sql.clone();
            let reply = match request.method.as_bytes() {
                verb::QUERY => {
                    if sql.trim() == "select 1" {
                        PgReply::Query(QueryReply::rows(
                            vec![ColumnDesc::new("?column?", Oid(23))],
                            vec![vec![SqlValue::Int(1)]],
                        ))
                    } else {
                        PgReply::Error(crate::pipe_contract::ErrorReply::new(
                            "42601",
                            format!("unknown: {sql}"),
                        ))
                    }
                }
                verb::PARSE => PgReply::Describe(DescribeReply {
                    parameter_types: vec![],
                    columns: vec![ColumnDesc::new("v", Oid(23))],
                }),
                verb::EXECUTE => PgReply::Query(QueryReply::rows(
                    vec![ColumnDesc::new("v", Oid(23))],
                    vec![vec![SqlValue::Int(1)]],
                )),
                other => {
                    return Err(ProximaError::Config(format!("unexpected verb {other:?}")));
                }
            };
            Ok(Response::typed(200, reply))
        }
    }

    fn echo_handle() -> PgPipeHandle {
        into_pg_handle(EchoPipe)
    }

    fn build_startup_bytes(user: &str, database: Option<&str>) -> Vec<u8> {
        let version_code: i32 = 196608;
        let mut params = Vec::new();
        params.extend_from_slice(b"user\0");
        params.extend_from_slice(user.as_bytes());
        params.push(0);
        if let Some(db) = database {
            params.extend_from_slice(b"database\0");
            params.extend_from_slice(db.as_bytes());
            params.push(0);
        }
        params.push(0);

        let total_len = 4 + 4 + params.len();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(total_len as i32).to_be_bytes());
        buf.extend_from_slice(&version_code.to_be_bytes());
        buf.extend_from_slice(&params);
        buf
    }

    fn build_ssl_request_bytes() -> Vec<u8> {
        let code: i32 = 80877103;
        let len: i32 = 8;
        let mut buf = Vec::new();
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&code.to_be_bytes());
        buf
    }

    fn build_cancel_request_bytes(process_id: i32, secret_key: &[u8]) -> Vec<u8> {
        let code: i32 = 80877102;
        let total_len = (4 + 4 + 4 + secret_key.len()) as i32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(&code.to_be_bytes());
        buf.extend_from_slice(&process_id.to_be_bytes());
        buf.extend_from_slice(secret_key);
        buf
    }

    fn build_query_message(sql: &str) -> Vec<u8> {
        let body_len = sql.len() + 1;
        let total_len = (4 + body_len) as i32;
        let mut buf = Vec::new();
        buf.push(b'Q');
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(sql.as_bytes());
        buf.push(0);
        buf
    }

    fn build_terminate_message() -> Vec<u8> {
        let len: i32 = 4;
        let mut buf = Vec::new();
        buf.push(b'X');
        buf.extend_from_slice(&len.to_be_bytes());
        buf
    }

    fn build_sync_message() -> Vec<u8> {
        let len: i32 = 4;
        let mut buf = Vec::new();
        buf.push(b'S');
        buf.extend_from_slice(&len.to_be_bytes());
        buf
    }

    fn build_bind_message(portal: &str, statement: &str) -> Vec<u8> {
        let body_len = portal.len() + 1 + statement.len() + 1 + 2 + 2 + 2;
        let total_len = (4 + body_len) as i32;
        let mut buf = Vec::new();
        buf.push(b'B');
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(portal.as_bytes());
        buf.push(0);
        buf.extend_from_slice(statement.as_bytes());
        buf.push(0);
        buf.extend_from_slice(&0i16.to_be_bytes());
        buf.extend_from_slice(&0i16.to_be_bytes());
        buf.extend_from_slice(&0i16.to_be_bytes());
        buf
    }

    fn has_message_tag(bytes: &[u8], wanted_tag: u8) -> bool {
        let mut offset = 0;
        while offset < bytes.len() {
            let Some(&tag) = bytes.get(offset) else {
                break;
            };
            let Some(len_slice) = bytes.get(offset + 1..offset + 5) else {
                break;
            };
            let len = i32::from_be_bytes([len_slice[0], len_slice[1], len_slice[2], len_slice[3]]);
            let Ok(frame_len) = usize::try_from(len) else {
                break;
            };
            if tag == wanted_tag {
                return true;
            }
            offset += 1 + frame_len;
        }
        false
    }

    #[test]
    fn cancel_registry_register_yields_distinct_pids() {
        let registry = CancelRegistry::new();

        let (key1, _flag1) = registry.register();
        let (key2, _flag2) = registry.register();

        assert_ne!(
            key1.process_id, key2.process_id,
            "each registration must get a distinct pid"
        );
    }

    #[test]
    fn cancel_registry_cancel_with_correct_pid_and_key_sets_flag_and_returns_true() {
        let registry = CancelRegistry::new();
        let (key, flag) = registry.register();

        let key_bytes = key.secret_key.to_be_bytes();
        let matched = registry.cancel(key.process_id, &key_bytes);

        assert!(matched, "cancel with correct pid+key must return true");
        assert!(
            flag.load(std::sync::atomic::Ordering::Relaxed),
            "flag must be set"
        );
    }

    #[test]
    fn cancel_registry_wrong_key_returns_false() {
        let registry = CancelRegistry::new();
        let (key, flag) = registry.register();

        let wrong_key = key.secret_key.wrapping_add(1).to_be_bytes();
        let matched = registry.cancel(key.process_id, &wrong_key);

        assert!(!matched, "wrong key must return false");
        assert!(
            !flag.load(std::sync::atomic::Ordering::Relaxed),
            "flag must not be set"
        );
    }

    #[test]
    fn cancel_registry_wrong_pid_returns_false() {
        let registry = CancelRegistry::new();
        let (key, flag) = registry.register();

        let key_bytes = key.secret_key.to_be_bytes();
        let matched = registry.cancel(key.process_id + 1, &key_bytes);

        assert!(!matched, "wrong pid must return false");
        assert!(
            !flag.load(std::sync::atomic::Ordering::Relaxed),
            "flag must not be set"
        );
    }

    #[test]
    fn cancel_registry_wrong_key_length_returns_false() {
        let registry = CancelRegistry::new();
        let (key, flag) = registry.register();

        let short_key = &key.secret_key.to_be_bytes()[..3];
        let matched = registry.cancel(key.process_id, short_key);

        assert!(!matched, "key of wrong length must return false");
        assert!(
            !flag.load(std::sync::atomic::Ordering::Relaxed),
            "flag must not be set"
        );
    }

    #[test]
    fn cancel_registry_deregister_then_cancel_returns_false() {
        let registry = CancelRegistry::new();
        let (key, _flag) = registry.register();

        registry.deregister(key.process_id);

        let key_bytes = key.secret_key.to_be_bytes();
        let matched = registry.cancel(key.process_id, &key_bytes);

        assert!(!matched, "cancel after deregister must return false");
    }

    #[test]
    fn encode_value_int_text_is_decimal_ascii() {
        let encoded = encode_value(&SqlValue::Int(42), FormatCode::Text, Oid(23));
        assert_eq!(encoded.as_deref(), Some(b"42".as_slice()));
    }

    #[test]
    fn encode_value_int4_binary_is_four_be_bytes() {
        let encoded = encode_value(&SqlValue::Int(7), FormatCode::Binary, Oid(23));
        assert_eq!(encoded, Some(7_i32.to_be_bytes().to_vec()));
    }

    #[test]
    fn encode_value_int8_binary_is_eight_be_bytes() {
        let encoded = encode_value(&SqlValue::Int(7), FormatCode::Binary, Oid(20));
        assert_eq!(encoded, Some(7_i64.to_be_bytes().to_vec()));
    }

    #[test]
    fn encode_value_null_is_none_regardless_of_format() {
        assert_eq!(
            encode_value(&SqlValue::Null, FormatCode::Text, Oid(23)),
            None
        );
        assert_eq!(
            encode_value(&SqlValue::Null, FormatCode::Binary, Oid(23)),
            None
        );
    }

    #[test]
    fn decode_binary_int4_parameter_becomes_int() {
        let parameter = BoundParameter {
            value: Some(7_i32.to_be_bytes().to_vec()),
            format: FormatCode::Binary,
        };
        assert_eq!(
            decode_parameter(&parameter, Some(Oid(23))),
            SqlValue::Int(7)
        );
    }

    #[test]
    fn decode_text_parameter_becomes_text() {
        let parameter = BoundParameter {
            value: Some(b"hello".to_vec()),
            format: FormatCode::Text,
        };
        assert_eq!(
            decode_parameter(&parameter, None),
            SqlValue::Text("hello".into())
        );
    }

    #[test]
    fn decode_null_parameter_becomes_null() {
        let parameter = BoundParameter {
            value: None,
            format: FormatCode::Binary,
        };
        assert_eq!(decode_parameter(&parameter, Some(Oid(23))), SqlValue::Null);
    }

    #[proxima::test(runtime = "tokio")]
    async fn negotiate_startup_with_user_and_database_proceeds() {
        let input = build_startup_bytes("alice", Some("appdb"));
        let stream = TestStream::new(input);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, false)
            .await
            .expect("negotiate must succeed");

        match result {
            Negotiation::Proceed { startup, .. } => {
                assert_eq!(startup.user, "alice");
                assert_eq!(startup.database.as_deref(), Some("appdb"));
                assert!(!startup.needs_negotiation, "plain 3.0 must not negotiate");
            }
            _ => panic!("expected Negotiation::Proceed"),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn startup_minor_version_bump_alone_requires_negotiate_protocol_version() {
        let mut input = build_startup_bytes("alice", None);
        let version_3_1: i32 = (3 << 16) | 1;
        input[4..8].copy_from_slice(&version_3_1.to_be_bytes());
        let stream = TestStream::new(input);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, false)
            .await
            .expect("negotiate must succeed");

        match result {
            Negotiation::Proceed { startup, .. } => {
                assert!(
                    startup.needs_negotiation,
                    "protocol 3.1 with zero _pq_ options must still trigger NegotiateProtocolVersion"
                );
                assert!(startup.negotiate_options.is_empty());
            }
            _ => panic!("expected Negotiation::Proceed"),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn negotiate_caps_unauthenticated_buffering_at_startup_packet_limit() {
        let mut input = Vec::new();
        input.extend_from_slice(&0x7fff_ffff_i32.to_be_bytes());
        input.extend_from_slice(&196_608_i32.to_be_bytes());
        input.extend_from_slice(&[0_u8; 12_000]);
        let stream = TestStream::new(input);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, false).await;

        assert!(
            matches!(result, Err(ServeError::MessageTooLarge { limit: 10_000 })),
            "oversized startup length must be rejected, not buffered"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn bytes_pipelined_behind_startup_survive_into_the_session() {
        let mut input = build_startup_bytes("alice", None);
        let query = FrontendMessage::Query {
            sql: PgStr::new(b"select 1"),
        };
        let mut query_bytes = [0_u8; 32];
        let written = query.encode(&mut query_bytes).expect("query must encode");
        input.extend_from_slice(&query_bytes[..written]);
        let stream = TestStream::new(input);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, false)
            .await
            .expect("negotiate must succeed");

        match result {
            Negotiation::Proceed { leftover, .. } => {
                assert_eq!(
                    leftover.as_ref(),
                    &query_bytes[..written],
                    "pipelined query bytes must carry into serve_session"
                );
            }
            _ => panic!("expected Negotiation::Proceed"),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn negotiate_startup_missing_user_returns_missing_user_error() {
        let version_code: i32 = 196608;
        let mut params = Vec::new();
        params.extend_from_slice(b"database\0appdb\0\0");
        let total_len = (8 + params.len()) as i32;
        let mut input = Vec::new();
        input.extend_from_slice(&total_len.to_be_bytes());
        input.extend_from_slice(&version_code.to_be_bytes());
        input.extend_from_slice(&params);

        let stream = TestStream::new(input);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, false).await;

        assert!(
            matches!(result, Err(ServeError::MissingUser)),
            "missing user must return MissingUser"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn negotiate_ssl_request_tls_unavailable_writes_n_then_startup_proceeds() {
        let mut input = build_ssl_request_bytes();
        input.extend_from_slice(&build_startup_bytes("bob", Some("testdb")));

        let mut stream = TestStream::new(input);
        let mut session = Session::new();

        let (n_written, user) = {
            let result = negotiate(&mut stream, &mut session, false)
                .await
                .expect("negotiate must succeed");
            match result {
                Negotiation::Proceed { startup, .. } => {
                    (stream.written().starts_with(b"N"), startup.user.clone())
                }
                _ => panic!("expected Negotiation::Proceed after SSL refusal"),
            }
        };

        assert!(n_written, "N must be written for SSL refusal");
        assert_eq!(user, "bob");
    }

    #[proxima::test(runtime = "tokio")]
    async fn negotiate_ssl_request_tls_available_writes_s_and_returns_start_tls() {
        let input = build_ssl_request_bytes();
        let stream = TestStream::new(input);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, true)
            .await
            .expect("negotiate must succeed");

        match result {
            Negotiation::StartTls(returned_stream) => {
                assert!(
                    returned_stream.written().starts_with(b"S"),
                    "S must be written for TLS acceptance"
                );
            }
            _ => panic!("expected Negotiation::StartTls"),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn negotiate_ssl_request_with_pipelined_bytes_tls_available_returns_error() {
        let mut input = build_ssl_request_bytes();
        input.extend_from_slice(&build_startup_bytes("carol", Some("db")));

        let stream = TestStream::new(input);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, true).await;

        assert!(
            matches!(result, Err(ServeError::Parse(_))),
            "pipelined bytes behind SSLRequest must be a parse error"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn negotiate_cancel_request_returns_cancel_with_pid_and_key() {
        let process_id = 12345_i32;
        let secret_key = [0xde, 0xad, 0xbe, 0xef_u8];
        let input = build_cancel_request_bytes(process_id, &secret_key);
        let stream = TestStream::new(input);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, false)
            .await
            .expect("negotiate must succeed");

        match result {
            Negotiation::Cancel {
                process_id: pid,
                secret_key: key,
            } => {
                assert_eq!(pid, 12345);
                assert_eq!(key.as_slice(), &secret_key);
            }
            _ => panic!("expected Negotiation::Cancel"),
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn negotiate_empty_input_returns_closed() {
        let stream = TestStream::new(vec![]);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, false)
            .await
            .expect("negotiate must succeed");

        assert!(
            matches!(result, Negotiation::Closed),
            "empty input must return Negotiation::Closed"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn negotiate_protocol_2_startup_returns_unsupported_protocol_error() {
        let proto_v2_code: i32 = 0x0002_0000;
        let mut params = Vec::new();
        params.extend_from_slice(b"user\0alice\0\0");
        let total_len = (8 + params.len()) as i32;
        let mut input = Vec::new();
        input.extend_from_slice(&total_len.to_be_bytes());
        input.extend_from_slice(&proto_v2_code.to_be_bytes());
        input.extend_from_slice(&params);

        let stream = TestStream::new(input);
        let mut session = Session::new();

        let result = negotiate(stream, &mut session, false).await;

        assert!(
            matches!(result, Err(ServeError::Parse(_))),
            "protocol 2.0 must return a parse error"
        );
    }

    fn default_startup() -> StartupOwned {
        StartupOwned {
            version: proxima_protocols::pgwire_codec::ProtocolVersion::V3_0,
            user: "alice".into(),
            database: Some("appdb".into()),
            parameters: vec![
                ("user".into(), "alice".into()),
                ("database".into(), "appdb".into()),
            ],
            negotiate_options: vec![],
            needs_negotiation: false,
        }
    }

    fn default_config() -> PgServerConfig {
        PgServerConfig::builder().parameters(vec![]).build()
    }

    fn session_after_startup() -> Session {
        let startup_bytes = build_startup_bytes("alice", Some("appdb"));
        let (parsed, _) = proxima_protocols::pgwire_codec::frontend::parse_initial(&startup_bytes)
            .expect("parse must succeed")
            .expect("complete startup packet");
        let mut session = Session::new();
        session
            .on_initial(&parsed)
            .expect("on_initial must succeed");
        session
    }

    #[proxima::test(runtime = "tokio")]
    async fn serve_session_trust_auth_produces_auth_ok_and_ready_for_query() {
        let session = session_after_startup();
        let mut ts = TestStream::new(build_terminate_message());

        serve_session(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            echo_handle(),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
        )
        .await
        .expect("serve_session must complete cleanly");

        assert!(
            has_message_tag(&ts.write_data, b'R'),
            "AuthenticationOk (R) must be present"
        );
        assert!(
            has_message_tag(&ts.write_data, b'Z'),
            "ReadyForQuery (Z) must be present"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn serve_session_trust_auth_simple_query_select_1_writes_row_description_and_data_row() {
        let session = session_after_startup();
        let mut input = build_query_message("select 1");
        input.extend_from_slice(&build_terminate_message());
        let mut ts = TestStream::new(input);

        serve_session(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            echo_handle(),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
        )
        .await
        .expect("serve_session must complete cleanly");

        assert!(
            has_message_tag(&ts.write_data, b'T'),
            "RowDescription (T) must be present"
        );
        assert!(
            has_message_tag(&ts.write_data, b'D'),
            "DataRow (D) must be present"
        );
        assert!(
            has_message_tag(&ts.write_data, b'C'),
            "CommandComplete (C) must be present"
        );
        assert!(
            has_message_tag(&ts.write_data, b'Z'),
            "ReadyForQuery (Z) must be present"
        );
    }

    // The listener's admission policy (quiesce/drain/capacity), not the
    // SQL engine, decides whether a simple query reaches it. Proves the
    // Shed path renders a real ErrorResponse (57P03, cannot_connect_now)
    // instead of dispatching to the engine, and that the connection stays
    // alive afterward (ReadyForQuery follows, ready for the next query).
    #[proxima::test(runtime = "tokio")]
    async fn simple_query_is_shed_with_a_57p03_error_while_admission_is_quiescing() {
        let session = session_after_startup();
        let mut input = build_query_message("select 1");
        input.extend_from_slice(&build_terminate_message());
        let mut ts = TestStream::new(input);

        let admission = proxima_listen::admission::ConnAdmission::unbounded();
        admission.begin_quiesce();

        serve_session_admitted(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            echo_handle(),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
            admission,
        )
        .await
        .expect("serve_session_admitted must complete cleanly");

        let found_57p03 = {
            let mut offset = 0;
            let mut found = false;
            while offset < ts.write_data.len() {
                let Some(&tag) = ts.write_data.get(offset) else {
                    break;
                };
                let Some(len_slice) = ts.write_data.get(offset + 1..offset + 5) else {
                    break;
                };
                let len =
                    i32::from_be_bytes([len_slice[0], len_slice[1], len_slice[2], len_slice[3]]);
                let Ok(frame_len) = usize::try_from(len) else {
                    break;
                };
                if tag == b'E'
                    && let Ok(Some((BackendMessage::ErrorResponse { fields }, _))) =
                        parse_backend(&ts.write_data[offset..])
                    && fields
                        .get(error_field::CODE)
                        .is_some_and(|code| code == "57P03")
                {
                    found = true;
                }
                offset += 1 + frame_len;
            }
            found
        };
        assert!(
            found_57p03,
            "expected a 57P03 (cannot_connect_now) ErrorResponse when admission sheds the query"
        );
        assert!(
            has_message_tag(&ts.write_data, b'Z'),
            "ReadyForQuery (Z) must follow — the connection stays alive through the shed"
        );
        assert!(
            !has_message_tag(&ts.write_data, b'D'),
            "the shed query must never reach the engine (no DataRow)"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn serve_session_simple_query_empty_string_produces_empty_query_response() {
        let session = session_after_startup();
        let mut input = build_query_message("");
        input.extend_from_slice(&build_terminate_message());
        let mut ts = TestStream::new(input);

        serve_session(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            echo_handle(),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
        )
        .await
        .expect("serve must complete");

        assert!(
            has_message_tag(&ts.write_data, b'I'),
            "EmptyQueryResponse (I) must be present"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn serve_session_bind_unknown_statement_produces_error_26000_then_ready_after_sync() {
        let session = session_after_startup();
        let mut input = build_bind_message("", "nonexistent_stmt");
        input.extend_from_slice(&build_sync_message());
        input.extend_from_slice(&build_terminate_message());
        let mut ts = TestStream::new(input);

        serve_session(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            echo_handle(),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
        )
        .await
        .expect("serve must complete");

        assert!(
            has_message_tag(&ts.write_data, b'E'),
            "ErrorResponse (E) must be present"
        );
        assert!(
            has_message_tag(&ts.write_data, b'Z'),
            "ReadyForQuery (Z) must follow Sync"
        );

        let found_26000 = {
            let mut offset = 0;
            let mut found = false;
            while offset < ts.write_data.len() {
                let Some(&tag) = ts.write_data.get(offset) else {
                    break;
                };
                let Some(len_slice) = ts.write_data.get(offset + 1..offset + 5) else {
                    break;
                };
                let len =
                    i32::from_be_bytes([len_slice[0], len_slice[1], len_slice[2], len_slice[3]]);
                let Ok(frame_len) = usize::try_from(len) else {
                    break;
                };
                if tag == b'E'
                    && let Ok(Some((BackendMessage::ErrorResponse { fields }, _))) =
                        parse_backend(&ts.write_data[offset..])
                    && fields
                        .get(error_field::CODE)
                        .is_some_and(|code| code == "26000")
                {
                    found = true;
                }
                offset += 1 + frame_len;
            }
            found
        };
        assert!(
            found_26000,
            "error code 26000 must appear for unknown statement"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn serve_session_terminate_message_ends_cleanly() {
        let session = session_after_startup();
        let mut ts = TestStream::new(build_terminate_message());

        let result = serve_session(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            echo_handle(),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
        )
        .await;

        assert!(result.is_ok(), "terminate must complete cleanly");
    }

    #[proxima::test(runtime = "tokio")]
    async fn serve_session_query_select_1_emits_data_row_with_value_1() {
        let session = session_after_startup();
        let mut input = build_query_message("select 1");
        input.extend_from_slice(&build_terminate_message());
        let mut ts = TestStream::new(input);

        serve_session(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            echo_handle(),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
        )
        .await
        .expect("serve must complete");

        let mut offset = 0;
        let mut data_row_value = None;
        while offset < ts.write_data.len() {
            let Ok(Some((message, consumed))) = parse_backend(&ts.write_data[offset..]) else {
                break;
            };
            if let BackendMessage::DataRow { columns } = message {
                let values: Vec<Option<&[u8]>> = columns.iter().collect();
                data_row_value = values.first().copied().flatten().map(<[u8]>::to_vec);
            }
            offset += consumed;
        }
        assert_eq!(
            data_row_value.as_deref(),
            Some(b"1".as_slice()),
            "select 1 must emit a data row whose single column encodes 1 in text"
        );
    }

    /// An engine that answers EXECUTE/QUERY with five int rows and PARSE
    /// with the matching one-column Describe — the fixture for portal
    /// suspension.
    struct FiveRowPipe;

    impl SendPipe for FiveRowPipe {
        type In = PgRequest;
        type Out = PgResponse;
        type Err = ProximaError;

        async fn call(&self, request: PgRequest) -> Result<PgResponse, ProximaError> {
            let reply = match request.method.as_bytes() {
                verb::PARSE => PgReply::Describe(DescribeReply {
                    parameter_types: vec![],
                    columns: vec![ColumnDesc::new("n", Oid(23))],
                }),
                verb::EXECUTE | verb::QUERY => PgReply::Query(QueryReply::rows(
                    vec![ColumnDesc::new("n", Oid(23))],
                    (1..=5).map(|number| vec![SqlValue::Int(number)]).collect(),
                )),
                other => return Err(ProximaError::Config(format!("unexpected verb {other:?}"))),
            };
            Ok(Response::typed(200, reply))
        }
    }

    fn build_parse_message(statement: &str, sql: &str) -> Vec<u8> {
        let body_len = statement.len() + 1 + sql.len() + 1 + 2;
        let total_len = (4 + body_len) as i32;
        let mut buf = vec![b'P'];
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(statement.as_bytes());
        buf.push(0);
        buf.extend_from_slice(sql.as_bytes());
        buf.push(0);
        buf.extend_from_slice(&0_i16.to_be_bytes());
        buf
    }

    fn build_execute_message(portal: &str, max_rows: i32) -> Vec<u8> {
        let body_len = portal.len() + 1 + 4;
        let total_len = (4 + body_len) as i32;
        let mut buf = vec![b'E'];
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(portal.as_bytes());
        buf.push(0);
        buf.extend_from_slice(&max_rows.to_be_bytes());
        buf
    }

    fn backend_tag_sequence(bytes: &[u8]) -> Vec<u8> {
        let mut tags = Vec::new();
        let mut offset = 0;
        while let Ok(Some((message, consumed))) = parse_backend(&bytes[offset..]) {
            if consumed == 0 {
                break;
            }
            tags.push(message.tag());
            offset += consumed;
        }
        tags
    }

    fn count_tag(bytes: &[u8], wanted: u8) -> usize {
        let mut count = 0;
        let mut offset = 0;
        while let Ok(Some((message, consumed))) = parse_backend(&bytes[offset..]) {
            if consumed == 0 {
                break;
            }
            if message.tag() == wanted {
                count += 1;
            }
            offset += consumed;
        }
        count
    }

    fn command_complete_tag(bytes: &[u8]) -> Option<String> {
        let mut offset = 0;
        while let Ok(Some((message, consumed))) = parse_backend(&bytes[offset..]) {
            if consumed == 0 {
                break;
            }
            if let BackendMessage::CommandComplete { tag } = message {
                return tag.to_str().ok().map(str::to_owned);
            }
            offset += consumed;
        }
        None
    }

    async fn run_portal_session(messages: Vec<u8>) -> Vec<u8> {
        let session = session_after_startup();
        let mut input = messages;
        input.extend_from_slice(&build_terminate_message());
        let mut ts = TestStream::new(input);
        serve_session(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            into_pg_handle(FiveRowPipe),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
        )
        .await
        .expect("portal session must complete cleanly");
        ts.write_data
    }

    #[proxima::test(runtime = "tokio")]
    async fn execute_max_rows_2_over_5_rows_suspends_then_completes_select_5() {
        let mut messages = build_parse_message("", "select n from generate_series(1,5)");
        messages.extend_from_slice(&build_bind_message("", ""));
        messages.extend_from_slice(&build_execute_message("", 2));
        messages.extend_from_slice(&build_execute_message("", 2));
        messages.extend_from_slice(&build_execute_message("", 2));
        messages.extend_from_slice(&build_sync_message());

        let written = run_portal_session(messages).await;

        let data_rows = count_tag(&written, b'D');
        let suspended = count_tag(&written, b's');
        let completes = count_tag(&written, b'C');
        assert_eq!(
            data_rows, 5,
            "all five rows must stream across the three batches"
        );
        assert_eq!(
            suspended, 2,
            "the first two batches must each PortalSuspend"
        );
        assert_eq!(
            completes, 1,
            "the final batch must CommandComplete, not suspend"
        );
        assert_eq!(
            command_complete_tag(&written).as_deref(),
            Some("SELECT 5"),
            "the tag must report the running total across all batches"
        );
        let tags = backend_tag_sequence(&written);
        assert!(
            !tags.contains(&b'T'),
            "Execute must not emit RowDescription (that is Describe's job)"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn execute_max_rows_0_streams_all_rows_without_suspension() {
        let mut messages = build_parse_message("", "select n from generate_series(1,5)");
        messages.extend_from_slice(&build_bind_message("", ""));
        messages.extend_from_slice(&build_execute_message("", 0));
        messages.extend_from_slice(&build_sync_message());

        let written = run_portal_session(messages).await;

        assert_eq!(
            count_tag(&written, b'D'),
            5,
            "max_rows=0 must stream all rows"
        );
        assert_eq!(count_tag(&written, b's'), 0, "max_rows=0 must not suspend");
        assert_eq!(
            count_tag(&written, b'C'),
            1,
            "a single CommandComplete must close the execute"
        );
        assert_eq!(command_complete_tag(&written).as_deref(), Some("SELECT 5"));
    }

    const COPY_OUT_ROWS: [&[u8]; 2] = [b"1\talice\n", b"2\tbob\n"];

    /// A COPY engine: `TO STDOUT` SQL yields CopyOut with two text rows;
    /// `FROM STDIN` yields CopyIn; the COPY_DATA second phase echoes the
    /// collected row count as the command tag (or ROLLBACK on failure).
    struct CopyPipe;

    impl SendPipe for CopyPipe {
        type In = PgRequest;
        type Out = PgResponse;
        type Err = ProximaError;

        async fn call(&self, request: PgRequest) -> Result<PgResponse, ProximaError> {
            let sql = request.payload.sql.clone();
            let reply = match request.method.as_bytes() {
                verb::QUERY | verb::EXECUTE if sql.contains("TO STDOUT") => PgReply::CopyOut {
                    format: CopyFormat::Text,
                    column_formats: vec![],
                    data: COPY_OUT_ROWS.iter().map(|row| row.to_vec()).collect(),
                },
                verb::QUERY | verb::EXECUTE if sql.contains("FROM STDIN") => PgReply::CopyIn {
                    format: CopyFormat::Text,
                    column_formats: vec![],
                },
                verb::COPY_DATA => {
                    let query = &request.payload;
                    if query.copy_failed {
                        PgReply::Query(QueryReply::tag("ROLLBACK"))
                    } else {
                        PgReply::Query(QueryReply::tag(format!("COPY {}", query.copy_data.len())))
                    }
                }
                verb::PARSE => PgReply::Describe(DescribeReply::default()),
                other => return Err(ProximaError::Config(format!("unexpected verb {other:?}"))),
            };
            Ok(Response::typed(200, reply))
        }
    }

    fn build_copy_data_message(payload: &[u8]) -> Vec<u8> {
        let total_len = (4 + payload.len()) as i32;
        let mut buf = vec![b'd'];
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(payload);
        buf
    }

    fn build_copy_done_message() -> Vec<u8> {
        let mut buf = vec![b'c'];
        buf.extend_from_slice(&4_i32.to_be_bytes());
        buf
    }

    fn build_copy_fail_message(message: &str) -> Vec<u8> {
        let total_len = (4 + message.len() + 1) as i32;
        let mut buf = vec![b'f'];
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(message.as_bytes());
        buf.push(0);
        buf
    }

    async fn run_copy_session(messages: Vec<u8>) -> Vec<u8> {
        let session = session_after_startup();
        let mut input = messages;
        input.extend_from_slice(&build_terminate_message());
        let mut ts = TestStream::new(input);
        serve_session(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            into_pg_handle(CopyPipe),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
        )
        .await
        .expect("copy session must complete cleanly");
        ts.write_data
    }

    fn copy_data_payloads(bytes: &[u8]) -> Vec<Vec<u8>> {
        let mut payloads = Vec::new();
        let mut offset = 0;
        while let Ok(Some((message, consumed))) = parse_backend(&bytes[offset..]) {
            if consumed == 0 {
                break;
            }
            if let BackendMessage::CopyData { data } = message {
                payloads.push(data.to_vec());
            }
            offset += consumed;
        }
        payloads
    }

    #[proxima::test(runtime = "tokio")]
    async fn copy_out_simple_query_emits_response_data_done_complete_in_order() {
        let written = run_copy_session(build_query_message("COPY t TO STDOUT")).await;

        let tags = backend_tag_sequence(&written);
        let copy_out = tags
            .iter()
            .position(|&tag| tag == b'H')
            .expect("CopyOutResponse (H)");
        let first_data = tags
            .iter()
            .position(|&tag| tag == b'd')
            .expect("CopyData (d)");
        let copy_done = tags
            .iter()
            .position(|&tag| tag == b'c')
            .expect("CopyDone (c)");
        let complete = tags
            .iter()
            .position(|&tag| tag == b'C')
            .expect("CommandComplete (C)");
        assert!(
            copy_out < first_data && first_data < copy_done && copy_done < complete,
            "order must be CopyOutResponse, CopyData, CopyDone, CommandComplete: {tags:?}"
        );
        assert_eq!(count_tag(&written, b'd'), 2, "one CopyData per row");
        assert_eq!(
            copy_data_payloads(&written),
            COPY_OUT_ROWS.map(<[u8]>::to_vec).to_vec()
        );
        assert_eq!(command_complete_tag(&written).as_deref(), Some("COPY 2"));
        assert!(
            tags.contains(&b'Z'),
            "simple-protocol copy-out must end with ReadyForQuery"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn copy_in_simple_query_collects_rows_and_reports_count() {
        let mut messages = build_query_message("COPY t FROM STDIN");
        messages.extend_from_slice(&build_copy_data_message(b"1\tx\n"));
        messages.extend_from_slice(&build_copy_data_message(b"2\ty\n"));
        messages.extend_from_slice(&build_copy_data_message(b"3\tz\n"));
        messages.extend_from_slice(&build_copy_done_message());

        let written = run_copy_session(messages).await;

        let tags = backend_tag_sequence(&written);
        assert!(tags.contains(&b'G'), "CopyInResponse (G) must be present");
        assert_eq!(
            command_complete_tag(&written).as_deref(),
            Some("COPY 3"),
            "the engine must see all three collected rows"
        );
        assert!(
            tags.contains(&b'Z'),
            "simple-protocol copy-in must end with ReadyForQuery"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn copy_in_copy_fail_routes_failed_flag_and_recovers() {
        let mut messages = build_query_message("COPY t FROM STDIN");
        messages.extend_from_slice(&build_copy_data_message(b"partial\n"));
        messages.extend_from_slice(&build_copy_fail_message("client aborted"));

        let written = run_copy_session(messages).await;

        let tags = backend_tag_sequence(&written);
        assert!(tags.contains(&b'G'), "CopyInResponse (G) must be present");
        assert_eq!(
            command_complete_tag(&written).as_deref(),
            Some("ROLLBACK"),
            "CopyFail must reach the engine as copy_failed=true"
        );
        assert!(
            tags.contains(&b'Z'),
            "copy-in must still reach ReadyForQuery after a CopyFail"
        );
    }

    /// A LISTEN/NOTIFY engine: `LISTEN c` subscribes channel "test",
    /// `NOTIFY c` publishes ("test","payload"); other queries answer empty.
    struct ListenNotifyPipe;

    impl SendPipe for ListenNotifyPipe {
        type In = PgRequest;
        type Out = PgResponse;
        type Err = ProximaError;

        async fn call(&self, request: PgRequest) -> Result<PgResponse, ProximaError> {
            let sql = request.payload.sql.clone();
            let reply = match request.method.as_bytes() {
                verb::QUERY if sql.starts_with("LISTEN") => PgReply::Listen {
                    channels: vec!["test".to_string()],
                },
                verb::QUERY if sql.starts_with("UNLISTEN") => PgReply::Unlisten {
                    channels: vec!["test".to_string()],
                    all: false,
                },
                verb::QUERY if sql.starts_with("NOTIFY") => PgReply::Notify {
                    channel: "test".to_string(),
                    payload: "payload".to_string(),
                },
                verb::QUERY => PgReply::Query(QueryReply::tag("OK")),
                other => return Err(ProximaError::Config(format!("unexpected verb {other:?}"))),
            };
            Ok(Response::typed(200, reply))
        }
    }

    /// A read stream for the listening connection: it yields the scripted
    /// startup bytes, then returns `Pending` exactly once (parking the
    /// session at the idle select so a notification can wake it), then yields
    /// the terminate tail. The broker publish wakes the parked task through
    /// its notification channel; the next idle-select poll reads the tail.
    struct GatedListener {
        head: std::io::Cursor<Vec<u8>>,
        tail: std::io::Cursor<Vec<u8>>,
        parked_once: bool,
        /// self-notify uses a pre-queued notification, so one self-wake
        /// re-polls the loop past the park to read the tail; the
        /// cross-connection case must NOT self-wake (the channel wake from
        /// the other connection's NOTIFY is the witness)
        self_wake: bool,
        write_data: Vec<u8>,
    }

    impl AsyncRead for GatedListener {
        fn poll_read(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            use std::io::Read;
            let head_read = self.head.read(buf).unwrap_or(0);
            if head_read > 0 {
                return Poll::Ready(Ok(head_read));
            }
            if !self.parked_once {
                self.parked_once = true;
                if self.self_wake {
                    context.waker().wake_by_ref();
                }
                return Poll::Pending;
            }
            Poll::Ready(Ok(self.tail.read(buf).unwrap_or(0)))
        }
    }

    impl AsyncWrite for GatedListener {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            self.write_data.extend_from_slice(buf);
            Poll::Ready(Ok(buf.len()))
        }

        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    fn notification_payloads(bytes: &[u8]) -> Vec<(i32, String, String)> {
        let mut found = Vec::new();
        let mut offset = 0;
        while let Ok(Some((message, consumed))) = parse_backend(&bytes[offset..]) {
            if consumed == 0 {
                break;
            }
            if let BackendMessage::NotificationResponse {
                process_id,
                channel,
                payload,
            } = message
            {
                found.push((
                    process_id,
                    channel.to_str().unwrap_or_default().to_owned(),
                    payload.to_str().unwrap_or_default().to_owned(),
                ));
            }
            offset += consumed;
        }
        found
    }

    /// The real delivery witness: two sessions share one broker. A runs
    /// `LISTEN test` then parks at idle; B runs `NOTIFY test`; A receives a
    /// NotificationResponse stamped with B's pid and the payload.
    #[proxima::test(runtime = "tokio")]
    async fn notify_on_one_connection_delivers_to_a_listener_on_another() {
        let broker = Arc::new(NotifyBroker::new());

        let listen_head = build_query_message("LISTEN test");
        let mut listen_tail = build_query_message("UNLISTEN test");
        listen_tail.extend_from_slice(&build_terminate_message());
        let mut listener = GatedListener {
            head: std::io::Cursor::new(listen_head),
            tail: std::io::Cursor::new(listen_tail),
            parked_once: false,
            self_wake: false,
            write_data: Vec::new(),
        };

        let mut notifier_input = build_query_message("NOTIFY test");
        notifier_input.extend_from_slice(&build_terminate_message());
        let mut notifier = TestStream::new(notifier_input);

        let listen_session = session_after_startup();
        let notify_session = session_after_startup();
        let engine = into_pg_handle(ListenNotifyPipe);
        let auth = PgAuth::Trust;
        let config = default_config();

        let listener_run = serve_session(
            &mut listener,
            listen_session,
            default_startup(),
            bytes::BytesMut::new(),
            engine.clone(),
            &auth,
            &config,
            None,
            Some(Arc::clone(&broker)),
            None,
        );
        let notifier_run = serve_session(
            &mut notifier,
            notify_session,
            default_startup(),
            bytes::BytesMut::new(),
            engine,
            &auth,
            &config,
            None,
            Some(Arc::clone(&broker)),
            None,
        );

        // both sessions run concurrently on one task: the listener subscribes
        // during LISTEN then parks at the idle select; the notifier publishes;
        // the publish wakes the parked listener, which delivers and finishes
        let (listener_outcome, notifier_outcome) = futures::join!(listener_run, notifier_run);
        listener_outcome.expect("listener session must complete cleanly");
        notifier_outcome.expect("notifier session must complete cleanly");

        let delivered = notification_payloads(&listener.write_data);
        assert_eq!(
            delivered.len(),
            1,
            "the listener must receive exactly one NotificationResponse"
        );
        let (_pid, channel, payload) = &delivered[0];
        assert_eq!(channel, "test", "channel must round-trip");
        assert_eq!(payload, "payload", "payload must round-trip");
    }

    /// A connection that LISTENs and NOTIFYs the same channel receives its
    /// own notification (PostgreSQL self-notify), drained at the idle point
    /// after the NOTIFY completes.
    #[proxima::test(runtime = "tokio")]
    async fn self_notify_is_delivered_to_the_same_connection() {
        let broker = Arc::new(NotifyBroker::new());

        // LISTEN then NOTIFY the same channel, then park so the idle select is
        // reached (a pipelined Terminate would be parsed in the inner loop
        // before idle). The self-notification, queued by NOTIFY's publish, is
        // drained at that idle point; only then does the tail Terminate arrive
        let mut head = build_query_message("LISTEN test");
        head.extend_from_slice(&build_query_message("NOTIFY test"));
        let mut stream = GatedListener {
            head: std::io::Cursor::new(head),
            tail: std::io::Cursor::new(build_terminate_message()),
            parked_once: false,
            self_wake: true,
            write_data: Vec::new(),
        };

        serve_session(
            &mut stream,
            session_after_startup(),
            default_startup(),
            bytes::BytesMut::new(),
            into_pg_handle(ListenNotifyPipe),
            &PgAuth::Trust,
            &default_config(),
            None,
            Some(Arc::clone(&broker)),
            None,
        )
        .await
        .expect("self-notify session must complete cleanly");

        let delivered = notification_payloads(&stream.write_data);
        assert_eq!(
            delivered.len(),
            1,
            "self-notify must deliver one NotificationResponse"
        );
        assert_eq!(delivered[0].1, "test");
        assert_eq!(delivered[0].2, "payload");
    }

    // ---- G12: engine-cooperative cancellation ----

    /// An engine that reports back what `cancel.is_cancelled()` read when its
    /// `call` ran — the observable contract for cooperative cancellation.
    struct CancelObserverPipe(Arc<AtomicBool>);

    impl SendPipe for CancelObserverPipe {
        type In = PgRequest;
        type Out = PgResponse;
        type Err = ProximaError;

        async fn call(&self, request: PgRequest) -> Result<PgResponse, ProximaError> {
            let observed = request.payload.cancel.is_cancelled();
            self.0.store(observed, Ordering::Relaxed);
            Ok(Response::typed(200, PgReply::Query(QueryReply::tag("OK"))))
        }
    }

    fn conn_state_with_cancel(cancel: CancelToken) -> ConnState {
        let (notify_tx, _notify_rx) = unbounded::<Notification>();
        ConnState {
            statements: NamedSlots::new(8),
            portals: NamedSlots::new(8),
            connection_id: 1,
            cancel,
            broker: None,
            process_id: 0,
            notify_tx,
        }
    }

    #[proxima::test(runtime = "tokio")]
    async fn engine_observes_cancel_token_set_before_the_call() {
        let flag = Arc::new(AtomicBool::new(false));
        let state = conn_state_with_cancel(CancelToken::from(Arc::clone(&flag)));
        // a matching CancelRequest would set the flag; simulate it firing
        // before the engine call so the engine observes it cooperatively
        flag.store(true, Ordering::Relaxed);

        let observed = Arc::new(AtomicBool::new(false));
        let engine = CancelObserverPipe(Arc::clone(&observed));
        let request = build_request(verb::QUERY, "select 1", state.request());
        engine
            .call(request)
            .await
            .expect("engine call must succeed");

        assert!(
            observed.load(Ordering::Relaxed),
            "engine must observe the cancel token as cancelled when the flag is set"
        );
    }

    #[proxima::test(runtime = "tokio")]
    async fn engine_observes_no_cancel_by_default() {
        let state = conn_state_with_cancel(CancelToken::none());

        let observed = Arc::new(AtomicBool::new(true));
        let engine = CancelObserverPipe(Arc::clone(&observed));
        let request = build_request(verb::QUERY, "select 1", state.request());
        engine
            .call(request)
            .await
            .expect("engine call must succeed");

        assert!(
            !observed.load(Ordering::Relaxed),
            "engine must observe an un-cancelled token by default"
        );
    }

    #[test]
    fn cancel_registry_flag_drives_the_token_the_driver_threads() {
        let registry = CancelRegistry::new();
        let (key, flag) = registry.register();
        let token = CancelToken::from(Arc::clone(&flag));

        assert!(!token.is_cancelled(), "fresh token must not be cancelled");
        registry.cancel(key.process_id, &key.secret_key.to_be_bytes());
        assert!(
            token.is_cancelled(),
            "a matching CancelRequest must surface through the threaded token"
        );
    }

    // ---- G10: streaming (lazy) row source ----

    /// A streaming engine: on `verb::QUERY`/`verb::EXECUTE` it spawns a task
    /// that feeds `count` int rows into an `async_channel` sender and returns
    /// `PgReply::QueryStream` over the receiver — the driver drains them
    /// lazily, never collecting the full set.
    struct StreamPipe {
        count: i64,
    }

    impl SendPipe for StreamPipe {
        type In = PgRequest;
        type Out = PgResponse;
        type Err = ProximaError;

        async fn call(&self, request: PgRequest) -> Result<PgResponse, ProximaError> {
            let reply = match request.method.as_bytes() {
                verb::PARSE => PgReply::Describe(DescribeReply {
                    parameter_types: vec![],
                    columns: vec![ColumnDesc::new("n", Oid(23))],
                }),
                verb::QUERY | verb::EXECUTE => {
                    let (sender, receiver) = async_channel::bounded::<Vec<SqlValue>>(4);
                    let count = self.count;
                    tokio::spawn(async move {
                        for number in 1..=count {
                            if sender.send(vec![SqlValue::Int(number)]).await.is_err() {
                                break;
                            }
                        }
                    });
                    PgReply::QueryStream {
                        columns: vec![ColumnDesc::new("n", Oid(23))],
                        rows: RowStream::new(receiver),
                        command_tag: None,
                    }
                }
                other => return Err(ProximaError::Config(format!("unexpected verb {other:?}"))),
            };
            Ok(Response::typed(200, reply))
        }
    }

    fn data_row_ints(bytes: &[u8]) -> Vec<i64> {
        let mut values = Vec::new();
        let mut offset = 0;
        while let Ok(Some((message, consumed))) = parse_backend(&bytes[offset..]) {
            if consumed == 0 {
                break;
            }
            if let BackendMessage::DataRow { columns } = message
                && let Some(Some(cell)) = columns.iter().next()
                && let Ok(text) = std::str::from_utf8(cell)
                && let Ok(number) = text.parse::<i64>()
            {
                values.push(number);
            }
            offset += consumed;
        }
        values
    }

    async fn run_stream_session(messages: Vec<u8>, count: i64) -> Vec<u8> {
        let session = session_after_startup();
        let mut input = messages;
        input.extend_from_slice(&build_terminate_message());
        let mut ts = TestStream::new(input);
        serve_session(
            &mut ts,
            session,
            default_startup(),
            bytes::BytesMut::new(),
            into_pg_handle(StreamPipe { count }),
            &PgAuth::Trust,
            &default_config(),
            None,
            None,
            None,
        )
        .await
        .expect("stream session must complete cleanly");
        ts.write_data
    }

    #[proxima::test(flavor = "multi_thread", worker_threads = 2)]
    async fn simple_query_stream_emits_all_rows_in_order_with_select_n_tag() {
        let written = run_stream_session(build_query_message("select streamed"), 4).await;

        assert!(
            has_message_tag(&written, b'T'),
            "RowDescription must precede the streamed rows"
        );
        assert_eq!(
            data_row_ints(&written),
            vec![1, 2, 3, 4],
            "all streamed rows must arrive in producer order"
        );
        assert_eq!(
            command_complete_tag(&written).as_deref(),
            Some("SELECT 4"),
            "the tag must report the count drained from the stream"
        );
    }

    #[proxima::test(flavor = "multi_thread", worker_threads = 2)]
    async fn extended_execute_stream_emits_all_rows_and_ignores_max_rows() {
        let mut messages = build_parse_message("", "select streamed");
        messages.extend_from_slice(&build_bind_message("", ""));
        // max_rows=2 is ignored for a stream (portal suspension over a lazy
        // source is out of scope for v1) — all rows stream, no PortalSuspended
        messages.extend_from_slice(&build_execute_message("", 2));
        messages.extend_from_slice(&build_sync_message());

        let written = run_stream_session(messages, 5).await;

        assert_eq!(
            data_row_ints(&written),
            vec![1, 2, 3, 4, 5],
            "a stream Execute must drain fully"
        );
        assert_eq!(
            count_tag(&written, b's'),
            0,
            "a streamed Execute must not PortalSuspend in v1"
        );
        assert_eq!(command_complete_tag(&written).as_deref(), Some("SELECT 5"));
        assert!(
            !backend_tag_sequence(&written).contains(&b'T'),
            "Execute must not emit RowDescription (Describe owns it)"
        );
    }
}
