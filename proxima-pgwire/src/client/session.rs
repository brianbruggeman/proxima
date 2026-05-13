//! Sans-IO PostgreSQL client session — the protocol state machine, no I/O.
//!
//! Bytes in (`feed`), bytes out (`take_outbound`), driven by `advance()`. This
//! is the client-side mirror of the codec's server `session::Session`: it owns
//! the startup + auth handshake (trust / cleartext / SCRAM-SHA-256) and the
//! simple/extended query exchange, but never touches a socket (principle 11).
//! A blocking driver, an async driver, and the `PipeFactory` client all wrap
//! it — that is what makes the client agnostic to the transport shape.

use proxima_protocols::pgwire_codec::backend::{AuthRequest, parse_backend};
use proxima_protocols::pgwire_codec::writer::MessageWriter;
use proxima_protocols::pgwire_codec::{BackendMessage, EncodeError, ParseError};

use crate::scram::{ScramClient, ScramError};

const PROTOCOL_3_0: i32 = 196_608;

/// A result column descriptor (the subset a caller compares).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    pub type_oid: u32,
}

/// One query's replies, collected.
#[derive(Debug, Clone, Default)]
pub struct QueryResult {
    pub columns: Vec<Column>,
    pub rows: Vec<Vec<Option<Vec<u8>>>>,
    pub command_tag: String,
}

impl QueryResult {
    /// The cell at `(row, col)` as UTF-8 text (text format), if present.
    #[must_use]
    pub fn text(&self, row: usize, col: usize) -> Option<&str> {
        self.rows
            .get(row)?
            .get(col)?
            .as_deref()
            .and_then(|bytes| core::str::from_utf8(bytes).ok())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode: {0}")]
    Encode(#[from] EncodeError),
    #[error("parse: {0}")]
    Parse(#[from] ParseError),
    #[error("scram: {0}")]
    Scram(#[from] ScramError),
    #[error("server connection closed mid-message")]
    Closed,
    #[error("unexpected auth request: {0}")]
    UnsupportedAuth(&'static str),
    #[error("server error {sqlstate}: {message}")]
    Server { sqlstate: String, message: String },
    #[error("protocol: {0}")]
    Protocol(&'static str),
}

/// What the driver must do next to advance the session. The driver owns I/O;
/// the session owns the protocol.
#[derive(Debug)]
pub enum Step {
    /// Bytes are queued — write `take_outbound()` to the transport, then
    /// call `advance()` again.
    Send,
    /// No progress without more inbound bytes — read, `feed()`, then
    /// `advance()` again.
    Recv,
    /// Startup + authentication complete; the session is idle and ready for
    /// `submit_simple` / `submit_extended`.
    Ready,
    /// The in-flight query finished.
    Complete(QueryResult),
}

#[derive(Debug, PartialEq, Eq)]
enum Phase {
    Connecting,
    Ready,
    Querying,
}

enum Outcome {
    Continue,
    Ready,
    Complete,
}

pub struct ClientSession {
    password: String,
    inbox: Vec<u8>,
    outbound: Vec<u8>,
    scram: Option<ScramClient>,
    phase: Phase,
    result: QueryResult,
    pending_error: Option<ClientError>,
}

impl ClientSession {
    /// Builds a session and queues the StartupMessage. `password` is unused for
    /// trust auth.
    ///
    /// # Errors
    /// [`ClientError::Encode`] if the startup parameters overflow the buffer.
    pub fn new(user: &str, password: &str, database: &str) -> Result<Self, ClientError> {
        let mut session = Self {
            password: password.to_string(),
            inbox: Vec::with_capacity(8192),
            outbound: Vec::with_capacity(256),
            scram: None,
            phase: Phase::Connecting,
            result: QueryResult::default(),
            pending_error: None,
        };
        session.queue_startup(user, database)?;
        Ok(session)
    }

    /// Drains the bytes the driver must send.
    pub fn take_outbound(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.outbound)
    }

    /// Appends bytes the driver read from the transport.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.inbox.extend_from_slice(bytes);
    }

    /// Queues a simple-protocol `Query`. Only valid once `Ready`.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if not ready, [`ClientError::Encode`] on overflow.
    pub fn submit_simple(&mut self, sql: &str) -> Result<(), ClientError> {
        self.begin_query()?;
        let mut buffer = vec![0_u8; sql.len() + 16];
        let mut writer = MessageWriter::tagged(&mut buffer, b'Q')?;
        writer.put_cstr(sql.as_bytes())?;
        let length = writer.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);
        Ok(())
    }

    /// Queues an extended-protocol round (Parse / Describe-statement / Bind /
    /// Execute / Sync) with text parameters. Only valid once `Ready`.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if not ready, [`ClientError::Encode`] on overflow.
    pub fn submit_extended(&mut self, sql: &str, params: &[&str]) -> Result<(), ClientError> {
        self.begin_query()?;
        let capacity = sql.len() + params.iter().map(|value| value.len()).sum::<usize>() + 64;
        let mut buffer = vec![0_u8; capacity];

        let mut parse = MessageWriter::tagged(&mut buffer, b'P')?;
        parse.put_cstr(b"")?;
        parse.put_cstr(sql.as_bytes())?;
        parse.put_i16(0)?;
        let length = parse.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);

        let mut describe = MessageWriter::tagged(&mut buffer, b'D')?;
        describe.put_u8(b'S')?;
        describe.put_cstr(b"")?;
        let length = describe.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);

        let mut bind = MessageWriter::tagged(&mut buffer, b'B')?;
        bind.put_cstr(b"")?;
        bind.put_cstr(b"")?;
        bind.put_i16(0)?;
        let count = i16::try_from(params.len()).map_err(|_| EncodeError::ValueTooLarge {
            field: "param count",
        })?;
        bind.put_i16(count)?;
        for value in params {
            let value_length =
                i32::try_from(value.len()).map_err(|_| EncodeError::ValueTooLarge {
                    field: "param length",
                })?;
            bind.put_i32(value_length)?;
            bind.put_bytes(value.as_bytes())?;
        }
        bind.put_i16(0)?;
        let length = bind.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);

        let mut execute = MessageWriter::tagged(&mut buffer, b'E')?;
        execute.put_cstr(b"")?;
        execute.put_i32(0)?;
        let length = execute.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);

        let sync = MessageWriter::tagged(&mut buffer, b'S')?;
        let length = sync.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);
        Ok(())
    }

    /// Queues a Terminate.
    ///
    /// # Errors
    /// [`ClientError::Encode`] on overflow (cannot happen for a 5-byte message).
    pub fn submit_terminate(&mut self) -> Result<(), ClientError> {
        let mut buffer = [0_u8; 8];
        let writer = MessageWriter::tagged(&mut buffer, b'X')?;
        let length = writer.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);
        Ok(())
    }

    /// Advances the state machine: sends queued bytes, then parses inbound
    /// messages until it needs more bytes or reaches a checkpoint.
    ///
    /// # Errors
    /// [`ClientError`] on a server `ErrorResponse`, a SCRAM failure, or a
    /// malformed message.
    pub fn advance(&mut self) -> Result<Step, ClientError> {
        if !self.outbound.is_empty() {
            return Ok(Step::Send);
        }
        loop {
            let (incoming, consumed) = match parse_backend(&self.inbox)? {
                None => return Ok(Step::Recv),
                Some((message, consumed)) => (to_incoming(&message), consumed),
            };
            self.inbox.drain(..consumed);
            match self.handle(incoming)? {
                Outcome::Continue => {
                    if !self.outbound.is_empty() {
                        return Ok(Step::Send);
                    }
                }
                Outcome::Ready => return Ok(Step::Ready),
                Outcome::Complete => {
                    if let Some(error) = self.pending_error.take() {
                        return Err(error);
                    }
                    return Ok(Step::Complete(core::mem::take(&mut self.result)));
                }
            }
        }
    }

    fn begin_query(&mut self) -> Result<(), ClientError> {
        if self.phase != Phase::Ready {
            return Err(ClientError::Protocol("submit before ready"));
        }
        self.result = QueryResult::default();
        self.pending_error = None;
        self.phase = Phase::Querying;
        Ok(())
    }

    fn handle(&mut self, incoming: Incoming) -> Result<Outcome, ClientError> {
        match self.phase {
            Phase::Connecting => self.handle_connecting(incoming),
            Phase::Querying => Ok(self.handle_querying(incoming)),
            Phase::Ready => Ok(Outcome::Continue),
        }
    }

    fn handle_connecting(&mut self, incoming: Incoming) -> Result<Outcome, ClientError> {
        match incoming {
            Incoming::AuthOk => Ok(Outcome::Continue),
            Incoming::AuthCleartext => {
                self.queue_password()?;
                Ok(Outcome::Continue)
            }
            Incoming::AuthSasl => {
                let mut scram = ScramClient::new(&self.password);
                let client_first = scram.client_first();
                self.queue_sasl_initial(&client_first)?;
                self.scram = Some(scram);
                Ok(Outcome::Continue)
            }
            Incoming::AuthSaslContinue(server_first) => {
                let scram = self
                    .scram
                    .as_mut()
                    .ok_or(ClientError::Protocol("SASLContinue before SASL"))?;
                let client_final = scram.client_final(&server_first)?;
                self.queue_sasl_response(&client_final)?;
                Ok(Outcome::Continue)
            }
            Incoming::AuthSaslFinal(server_final) => {
                let scram = self
                    .scram
                    .as_ref()
                    .ok_or(ClientError::Protocol("SASLFinal before SASL"))?;
                scram.verify_server_final(&server_final)?;
                Ok(Outcome::Continue)
            }
            Incoming::ReadyForQuery => {
                self.phase = Phase::Ready;
                Ok(Outcome::Ready)
            }
            Incoming::ErrorResponse { sqlstate, message } => {
                Err(ClientError::Server { sqlstate, message })
            }
            Incoming::UnsupportedAuth(kind) => Err(ClientError::UnsupportedAuth(kind)),
            _ => Ok(Outcome::Continue),
        }
    }

    fn handle_querying(&mut self, incoming: Incoming) -> Outcome {
        match incoming {
            Incoming::RowDescription(columns) => {
                self.result.columns = columns;
                Outcome::Continue
            }
            Incoming::DataRow(values) => {
                self.result.rows.push(values);
                Outcome::Continue
            }
            Incoming::CommandComplete(tag) => {
                self.result.command_tag = tag;
                Outcome::Continue
            }
            Incoming::ErrorResponse { sqlstate, message } => {
                // record it; the server still sends ReadyForQuery, so we
                // recover to Ready and surface the error from `advance`.
                self.pending_error = Some(ClientError::Server { sqlstate, message });
                Outcome::Continue
            }
            Incoming::ReadyForQuery => {
                self.phase = Phase::Ready;
                Outcome::Complete
            }
            _ => Outcome::Continue,
        }
    }

    fn queue_startup(&mut self, user: &str, database: &str) -> Result<(), ClientError> {
        let mut buffer = vec![0_u8; user.len() + database.len() + 64];
        let mut writer = MessageWriter::untagged(&mut buffer)?;
        writer.put_i32(PROTOCOL_3_0)?;
        writer.put_cstr(b"user")?;
        writer.put_cstr(user.as_bytes())?;
        writer.put_cstr(b"database")?;
        writer.put_cstr(database.as_bytes())?;
        writer.put_u8(0)?;
        let length = writer.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);
        Ok(())
    }

    fn queue_password(&mut self) -> Result<(), ClientError> {
        let mut buffer = vec![0_u8; self.password.len() + 8];
        let mut writer = MessageWriter::tagged(&mut buffer, b'p')?;
        writer.put_cstr(self.password.as_bytes())?;
        let length = writer.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);
        Ok(())
    }

    fn queue_sasl_initial(&mut self, client_first: &[u8]) -> Result<(), ClientError> {
        let mut buffer = vec![0_u8; client_first.len() + 32];
        let mut writer = MessageWriter::tagged(&mut buffer, b'p')?;
        writer.put_cstr(b"SCRAM-SHA-256")?;
        let length = i32::try_from(client_first.len()).map_err(|_| EncodeError::ValueTooLarge {
            field: "sasl initial",
        })?;
        writer.put_i32(length)?;
        writer.put_bytes(client_first)?;
        let total = writer.finish()?;
        self.outbound.extend_from_slice(&buffer[..total]);
        Ok(())
    }

    fn queue_sasl_response(&mut self, client_final: &[u8]) -> Result<(), ClientError> {
        let mut buffer = vec![0_u8; client_final.len() + 8];
        let mut writer = MessageWriter::tagged(&mut buffer, b'p')?;
        writer.put_bytes(client_final)?;
        let length = writer.finish()?;
        self.outbound.extend_from_slice(&buffer[..length]);
        Ok(())
    }
}

enum Incoming {
    AuthOk,
    AuthSasl,
    AuthSaslContinue(Vec<u8>),
    AuthSaslFinal(Vec<u8>),
    AuthCleartext,
    RowDescription(Vec<Column>),
    DataRow(Vec<Option<Vec<u8>>>),
    CommandComplete(String),
    ErrorResponse { sqlstate: String, message: String },
    ReadyForQuery,
    Ignored,
    UnsupportedAuth(&'static str),
}

fn to_incoming(message: &BackendMessage<'_>) -> Incoming {
    match message {
        BackendMessage::Authentication(AuthRequest::Ok) => Incoming::AuthOk,
        BackendMessage::Authentication(AuthRequest::Sasl { .. }) => Incoming::AuthSasl,
        BackendMessage::Authentication(AuthRequest::SaslContinue { data }) => {
            Incoming::AuthSaslContinue(data.to_vec())
        }
        BackendMessage::Authentication(AuthRequest::SaslFinal { data }) => {
            Incoming::AuthSaslFinal(data.to_vec())
        }
        BackendMessage::Authentication(AuthRequest::CleartextPassword) => Incoming::AuthCleartext,
        BackendMessage::Authentication(AuthRequest::Md5Password { .. }) => {
            Incoming::UnsupportedAuth("md5")
        }
        BackendMessage::Authentication(_) => Incoming::UnsupportedAuth("gss/kerberos/sspi"),
        BackendMessage::RowDescription { fields } => Incoming::RowDescription(
            fields
                .iter()
                .map(|field| Column {
                    name: field.name.to_string(),
                    type_oid: field.type_oid.0,
                })
                .collect(),
        ),
        BackendMessage::DataRow { columns } => Incoming::DataRow(
            columns
                .iter()
                .map(|value| value.map(<[u8]>::to_vec))
                .collect(),
        ),
        BackendMessage::CommandComplete { tag } => Incoming::CommandComplete(tag.to_string()),
        BackendMessage::ErrorResponse { fields } => {
            let mut sqlstate = String::new();
            let mut message = String::new();
            for (code, value) in fields.iter() {
                match code {
                    b'C' => sqlstate = value.to_string(),
                    b'M' => message = value.to_string(),
                    _ => {}
                }
            }
            Incoming::ErrorResponse { sqlstate, message }
        }
        BackendMessage::ReadyForQuery { .. } => Incoming::ReadyForQuery,
        _ => Incoming::Ignored,
    }
}
