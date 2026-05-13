//! Server-side connection session finite state machine.
//!
//! The message codec ([`super::frontend`], [`super::backend`]) is
//! stateless and re-entrant; this FSM owns protocol sequencing: which
//! message is legal in which state, the startup / TLS / authentication
//! choreography, extended-query error recovery (discard until Sync),
//! COPY sub-protocol transitions, and the transaction status byte that
//! ReadyForQuery reports.
//!
//! The FSM is sans-IO and driven from both sides:
//!
//! - the wire side calls [`Session::on_initial`] / [`Session::on_frontend`]
//!   with each parsed message and obeys the returned [`Disposition`]
//! - the server side calls the transition methods (`ssl_accepted`,
//!   `auth_ok`, `ready_for_query`, ...) as it emits backend messages
//!
//! Driver contract: feed exactly one parsed message at a time and do not
//! parse further input while a server-side transition is owed (see
//! [`Session::wire_phase`]). Backpressure is "stop reading", which is the
//! natural shape for a sans-IO read loop.
//!
//! State diagram (happy paths):
//!
//! ```text
//! AwaitingInitial --SslRequest--> SslDecision --accept--> TlsHandshake
//!       |                              |--refuse--> AwaitingInitial
//!       |--Startup--> StartupReceived --require_auth--> AuthInProgress
//!       |                  |--auth_ok (trust)--> Starting
//!       |--Cancel--> Cancelling
//! AuthInProgress --'p' + auth_ok--> Starting --ready_for_query--> Idle
//! Idle --Query--> SimpleQuery --ready_for_query--> Idle
//! Idle --Parse/Bind/...--> Extended --Sync--> Syncing --ready_for_query--> Idle
//! Extended --extended_error--> ExtendedFailed --Sync--> Syncing
//! SimpleQuery/Extended --copy_*_begun--> CopyIn/CopyOut/CopyBoth
//! any post-startup --Terminate--> Terminated
//! ```

use super::frontend::{FrontendMessage, InitialMessage};
use super::types::TransactionStatus;

/// Authentication exchange selected by the server after Startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthFlow {
    Cleartext,
    Md5,
    Sasl,
    Gss,
}

/// Which statement context a COPY sub-protocol was entered from — COPY
/// under the simple protocol returns to the simple-query flow, COPY
/// under the extended protocol returns to the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopySource {
    Simple,
    Extended,
}

/// What the wire side should do with a message it just parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// dispatch the message to the server logic
    Handle,
    /// drop the message: extended-pipeline recovery (skip until Sync) or
    /// frontend COPY messages during copy-out, which the protocol says
    /// to ignore
    Discard,
}

/// Which decode entry point the next read should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WirePhase {
    /// untagged startup-phase framing — [`super::frontend::parse_initial`]
    Initial,
    /// tagged framing — [`super::frontend::parse_frontend`]
    Tagged,
    /// no protocol read expected: the server owes a transition (TLS
    /// handshake in progress, auth verification, response generation)
    Quiescent,
    /// connection is finished (terminated, cancelled, or failed)
    Closed,
}

/// Session state names, exposed for error context and logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StateName {
    AwaitingInitial,
    SslDecision,
    GssDecision,
    TlsHandshake,
    StartupReceived,
    AuthInProgress,
    Starting,
    Idle,
    SimpleQuery,
    Extended,
    ExtendedFailed,
    Syncing,
    CopyIn,
    CopyOut,
    CopyBoth,
    FunctionCallActive,
    Cancelling,
    Terminated,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    AwaitingInitial { ssl_done: bool, gss_done: bool },
    SslDecision { ssl_done: bool, gss_done: bool },
    GssDecision { ssl_done: bool, gss_done: bool },
    TlsHandshake,
    StartupReceived,
    AuthInProgress { flow: AuthFlow },
    Starting,
    Idle,
    SimpleQuery,
    Extended,
    ExtendedFailed,
    Syncing,
    CopyIn { source: CopySource },
    CopyOut { source: CopySource },
    CopyBoth,
    FunctionCallActive,
    Cancelling,
    Terminated,
    Failed,
}

impl State {
    const fn name(self) -> StateName {
        match self {
            Self::AwaitingInitial { .. } => StateName::AwaitingInitial,
            Self::SslDecision { .. } => StateName::SslDecision,
            Self::GssDecision { .. } => StateName::GssDecision,
            Self::TlsHandshake => StateName::TlsHandshake,
            Self::StartupReceived => StateName::StartupReceived,
            Self::AuthInProgress { .. } => StateName::AuthInProgress,
            Self::Starting => StateName::Starting,
            Self::Idle => StateName::Idle,
            Self::SimpleQuery => StateName::SimpleQuery,
            Self::Extended => StateName::Extended,
            Self::ExtendedFailed => StateName::ExtendedFailed,
            Self::Syncing => StateName::Syncing,
            Self::CopyIn { .. } => StateName::CopyIn,
            Self::CopyOut { .. } => StateName::CopyOut,
            Self::CopyBoth => StateName::CopyBoth,
            Self::FunctionCallActive => StateName::FunctionCallActive,
            Self::Cancelling => StateName::Cancelling,
            Self::Terminated => StateName::Terminated,
            Self::Failed => StateName::Failed,
        }
    }
}

/// Protocol-sequencing violation: a message or server action that is not
/// legal in the current state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SessionError {
    /// the peer sent a message the protocol does not allow here
    IllegalMessage { state: StateName, tag: u8 },
    /// the server logic attempted a transition the protocol does not
    /// allow here — a driver bug, not a peer bug
    IllegalTransition {
        state: StateName,
        action: &'static str,
    },
}

impl core::fmt::Display for SessionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IllegalMessage { state, tag } => {
                write!(
                    formatter,
                    "illegal message tag {tag:#04x} in state {state:?}"
                )
            }
            Self::IllegalTransition { state, action } => {
                write!(formatter, "illegal transition {action} in state {state:?}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SessionError {}

/// Server-side PostgreSQL connection session FSM.
#[derive(Debug, Clone)]
pub struct Session {
    state: State,
    transaction: TransactionStatus,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: State::AwaitingInitial {
                ssl_done: false,
                gss_done: false,
            },
            transaction: TransactionStatus::Idle,
        }
    }

    #[must_use]
    pub const fn state_name(&self) -> StateName {
        self.state.name()
    }

    /// Which decode entry point applies to the next read.
    #[must_use]
    pub const fn wire_phase(&self) -> WirePhase {
        match self.state {
            State::AwaitingInitial { .. } => WirePhase::Initial,
            State::AuthInProgress { .. }
            | State::Idle
            | State::Extended
            | State::ExtendedFailed
            | State::CopyIn { .. }
            | State::CopyOut { .. }
            | State::CopyBoth => WirePhase::Tagged,
            State::SslDecision { .. }
            | State::GssDecision { .. }
            | State::TlsHandshake
            | State::StartupReceived
            | State::Starting
            | State::SimpleQuery
            | State::Syncing
            | State::FunctionCallActive => WirePhase::Quiescent,
            State::Cancelling | State::Terminated | State::Failed => WirePhase::Closed,
        }
    }

    #[must_use]
    pub const fn is_closed(&self) -> bool {
        matches!(self.wire_phase(), WirePhase::Closed)
    }

    /// Transaction status byte the next ReadyForQuery will carry. The
    /// server logic updates it via [`Session::set_transaction_status`]
    /// as BEGIN / COMMIT / ROLLBACK / errors change it.
    #[must_use]
    pub const fn transaction_status(&self) -> TransactionStatus {
        self.transaction
    }

    pub fn set_transaction_status(&mut self, status: TransactionStatus) {
        self.transaction = status;
    }

    /// Accepts one untagged startup-phase message.
    ///
    /// # Errors
    /// [`SessionError::IllegalMessage`] when startup-phase framing is not
    /// expected or the specific request repeats (e.g. second SSLRequest).
    pub fn on_initial(
        &mut self,
        message: &InitialMessage<'_>,
    ) -> Result<Disposition, SessionError> {
        let State::AwaitingInitial { ssl_done, gss_done } = self.state else {
            return Err(SessionError::IllegalMessage {
                state: self.state.name(),
                tag: 0,
            });
        };
        match message {
            InitialMessage::SslRequest => {
                if ssl_done {
                    return Err(SessionError::IllegalMessage {
                        state: self.state.name(),
                        tag: 0,
                    });
                }
                self.state = State::SslDecision { ssl_done, gss_done };
            }
            InitialMessage::GssEncRequest => {
                if gss_done {
                    return Err(SessionError::IllegalMessage {
                        state: self.state.name(),
                        tag: 0,
                    });
                }
                self.state = State::GssDecision { ssl_done, gss_done };
            }
            InitialMessage::Startup(_) => {
                self.state = State::StartupReceived;
            }
            InitialMessage::Cancel(_) => {
                self.state = State::Cancelling;
            }
        }
        Ok(Disposition::Handle)
    }

    /// Accepts one tagged frontend message.
    ///
    /// # Errors
    /// [`SessionError::IllegalMessage`] when the message is not legal in
    /// the current state.
    pub fn on_frontend(
        &mut self,
        message: &FrontendMessage<'_>,
    ) -> Result<Disposition, SessionError> {
        let illegal = SessionError::IllegalMessage {
            state: self.state.name(),
            tag: message.tag(),
        };
        match self.state {
            State::AuthInProgress { .. } => match message {
                FrontendMessage::AuthData(_) => Ok(Disposition::Handle),
                FrontendMessage::Terminate => {
                    self.state = State::Terminated;
                    Ok(Disposition::Handle)
                }
                _ => Err(illegal),
            },
            State::Idle => match message {
                FrontendMessage::Query { .. } => {
                    self.state = State::SimpleQuery;
                    Ok(Disposition::Handle)
                }
                FrontendMessage::Parse(_)
                | FrontendMessage::Bind(_)
                | FrontendMessage::Describe { .. }
                | FrontendMessage::Execute { .. }
                | FrontendMessage::Close { .. } => {
                    self.state = State::Extended;
                    Ok(Disposition::Handle)
                }
                // a Sync outside a pipeline still answers ReadyForQuery
                FrontendMessage::Sync => {
                    self.state = State::Syncing;
                    Ok(Disposition::Handle)
                }
                FrontendMessage::Flush => Ok(Disposition::Handle),
                FrontendMessage::FunctionCall(_) => {
                    self.state = State::FunctionCallActive;
                    Ok(Disposition::Handle)
                }
                FrontendMessage::Terminate => {
                    self.state = State::Terminated;
                    Ok(Disposition::Handle)
                }
                _ => Err(illegal),
            },
            State::Extended => match message {
                FrontendMessage::Parse(_)
                | FrontendMessage::Bind(_)
                | FrontendMessage::Describe { .. }
                | FrontendMessage::Execute { .. }
                | FrontendMessage::Close { .. }
                | FrontendMessage::Flush => Ok(Disposition::Handle),
                FrontendMessage::Sync => {
                    self.state = State::Syncing;
                    Ok(Disposition::Handle)
                }
                FrontendMessage::Terminate => {
                    self.state = State::Terminated;
                    Ok(Disposition::Handle)
                }
                _ => Err(illegal),
            },
            State::ExtendedFailed => match message {
                // protocol error recovery: discard until Sync
                FrontendMessage::Parse(_)
                | FrontendMessage::Bind(_)
                | FrontendMessage::Describe { .. }
                | FrontendMessage::Execute { .. }
                | FrontendMessage::Close { .. }
                | FrontendMessage::Flush
                | FrontendMessage::FunctionCall(_)
                | FrontendMessage::CopyData { .. }
                | FrontendMessage::CopyDone
                | FrontendMessage::CopyFail { .. } => Ok(Disposition::Discard),
                FrontendMessage::Sync => {
                    self.state = State::Syncing;
                    Ok(Disposition::Handle)
                }
                FrontendMessage::Terminate => {
                    self.state = State::Terminated;
                    Ok(Disposition::Handle)
                }
                _ => Err(illegal),
            },
            State::CopyIn { source } => match message {
                FrontendMessage::CopyData { .. } => Ok(Disposition::Handle),
                FrontendMessage::CopyDone => {
                    self.state = match source {
                        CopySource::Simple => State::SimpleQuery,
                        CopySource::Extended => State::Extended,
                    };
                    Ok(Disposition::Handle)
                }
                FrontendMessage::CopyFail { .. } => {
                    self.state = match source {
                        CopySource::Simple => State::SimpleQuery,
                        CopySource::Extended => State::ExtendedFailed,
                    };
                    Ok(Disposition::Handle)
                }
                // extended copy-in: Flush/Sync mid-copy are protocol
                // errors the server reports, then recovers from
                FrontendMessage::Terminate => {
                    self.state = State::Terminated;
                    Ok(Disposition::Handle)
                }
                _ => Err(illegal),
            },
            State::CopyOut { .. } => match message {
                // the protocol defines no frontend abort for copy-out;
                // stray copy-in-shaped messages are ignored per the COPY
                // robustness note
                FrontendMessage::CopyData { .. }
                | FrontendMessage::CopyDone
                | FrontendMessage::CopyFail { .. }
                | FrontendMessage::Flush
                | FrontendMessage::Sync => Ok(Disposition::Discard),
                FrontendMessage::Terminate => {
                    self.state = State::Terminated;
                    Ok(Disposition::Handle)
                }
                _ => Err(illegal),
            },
            State::CopyBoth => match message {
                FrontendMessage::CopyData { .. } | FrontendMessage::CopyDone => {
                    Ok(Disposition::Handle)
                }
                FrontendMessage::Terminate => {
                    self.state = State::Terminated;
                    Ok(Disposition::Handle)
                }
                _ => Err(illegal),
            },
            State::AwaitingInitial { .. }
            | State::SslDecision { .. }
            | State::GssDecision { .. }
            | State::TlsHandshake
            | State::StartupReceived
            | State::Starting
            | State::SimpleQuery
            | State::Syncing
            | State::FunctionCallActive
            | State::Cancelling
            | State::Terminated
            | State::Failed => Err(illegal),
        }
    }

    /// Server accepted SSLRequest (`S` sent); TLS handshake follows.
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `SslDecision`.
    pub fn ssl_accepted(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::SslDecision { .. } => {
                self.state = State::TlsHandshake;
                Ok(())
            }
            _ => Err(self.illegal_transition("ssl_accepted")),
        }
    }

    /// Server refused SSLRequest (`N` sent); cleartext startup continues.
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `SslDecision`.
    pub fn ssl_refused(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::SslDecision { gss_done, .. } => {
                self.state = State::AwaitingInitial {
                    ssl_done: true,
                    gss_done,
                };
                Ok(())
            }
            _ => Err(self.illegal_transition("ssl_refused")),
        }
    }

    /// Server refused GSSENCRequest (`N` sent).
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `GssDecision`.
    pub fn gss_enc_refused(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::GssDecision { ssl_done, .. } => {
                self.state = State::AwaitingInitial {
                    ssl_done,
                    gss_done: true,
                };
                Ok(())
            }
            _ => Err(self.illegal_transition("gss_enc_refused")),
        }
    }

    /// TLS handshake completed; the client now sends Startup over TLS.
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `TlsHandshake`.
    pub fn tls_established(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::TlsHandshake => {
                // ssl_done blocks a second SSLRequest inside the tunnel
                self.state = State::AwaitingInitial {
                    ssl_done: true,
                    gss_done: true,
                };
                Ok(())
            }
            _ => Err(self.illegal_transition("tls_established")),
        }
    }

    /// Server chose an authentication exchange and sent the matching
    /// AuthenticationCleartextPassword / MD5Password / SASL / GSS request.
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `StartupReceived`.
    pub fn auth_requested(&mut self, flow: AuthFlow) -> Result<(), SessionError> {
        match self.state {
            State::StartupReceived => {
                self.state = State::AuthInProgress { flow };
                Ok(())
            }
            _ => Err(self.illegal_transition("auth_requested")),
        }
    }

    /// Active authentication exchange, when one is in progress.
    #[must_use]
    pub const fn auth_flow(&self) -> Option<AuthFlow> {
        match self.state {
            State::AuthInProgress { flow } => Some(flow),
            _ => None,
        }
    }

    /// Server sent AuthenticationOk — either directly after Startup
    /// (trust) or after verifying the exchange. ParameterStatus /
    /// BackendKeyData emission follows, then [`Session::ready_for_query`].
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `StartupReceived` /
    /// `AuthInProgress`.
    pub fn auth_ok(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::StartupReceived | State::AuthInProgress { .. } => {
                self.state = State::Starting;
                Ok(())
            }
            _ => Err(self.illegal_transition("auth_ok")),
        }
    }

    /// Authentication failed; the server sends a fatal ErrorResponse and
    /// closes.
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `StartupReceived` /
    /// `AuthInProgress`.
    pub fn auth_failed(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::StartupReceived | State::AuthInProgress { .. } => {
                self.state = State::Failed;
                Ok(())
            }
            _ => Err(self.illegal_transition("auth_failed")),
        }
    }

    /// Server emitted ReadyForQuery; returns the transaction status byte
    /// it must carry.
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `Starting` /
    /// `SimpleQuery` / `Syncing` / `FunctionCallActive`.
    pub fn ready_for_query(&mut self) -> Result<TransactionStatus, SessionError> {
        match self.state {
            State::Starting | State::SimpleQuery | State::Syncing | State::FunctionCallActive => {
                self.state = State::Idle;
                Ok(self.transaction)
            }
            _ => Err(self.illegal_transition("ready_for_query")),
        }
    }

    /// An error occurred inside the extended pipeline; the server sent
    /// ErrorResponse and now discards messages until Sync.
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `Extended`.
    pub fn extended_error(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::Extended => {
                self.state = State::ExtendedFailed;
                Ok(())
            }
            _ => Err(self.illegal_transition("extended_error")),
        }
    }

    /// Server sent CopyInResponse.
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `SimpleQuery` /
    /// `Extended`.
    pub fn copy_in_begun(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::SimpleQuery => {
                self.state = State::CopyIn {
                    source: CopySource::Simple,
                };
                Ok(())
            }
            State::Extended => {
                self.state = State::CopyIn {
                    source: CopySource::Extended,
                };
                Ok(())
            }
            _ => Err(self.illegal_transition("copy_in_begun")),
        }
    }

    /// Server sent CopyOutResponse.
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `SimpleQuery` /
    /// `Extended`.
    pub fn copy_out_begun(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::SimpleQuery => {
                self.state = State::CopyOut {
                    source: CopySource::Simple,
                };
                Ok(())
            }
            State::Extended => {
                self.state = State::CopyOut {
                    source: CopySource::Extended,
                };
                Ok(())
            }
            _ => Err(self.illegal_transition("copy_out_begun")),
        }
    }

    /// Server sent CopyBothResponse (streaming replication).
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `SimpleQuery`.
    pub fn copy_both_begun(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::SimpleQuery => {
                self.state = State::CopyBoth;
                Ok(())
            }
            _ => Err(self.illegal_transition("copy_both_begun")),
        }
    }

    /// Server finished a copy-out / copy-both transfer (sent CopyDone +
    /// CommandComplete).
    ///
    /// # Errors
    /// [`SessionError::IllegalTransition`] outside `CopyOut` / `CopyBoth`.
    pub fn copy_finished(&mut self) -> Result<(), SessionError> {
        match self.state {
            State::CopyOut { source } => {
                self.state = match source {
                    CopySource::Simple => State::SimpleQuery,
                    CopySource::Extended => State::Extended,
                };
                Ok(())
            }
            State::CopyBoth => {
                self.state = State::SimpleQuery;
                Ok(())
            }
            _ => Err(self.illegal_transition("copy_finished")),
        }
    }

    /// A fatal error outside the extended pipeline; the server sends a
    /// fatal ErrorResponse and closes.
    pub fn fail(&mut self) {
        self.state = State::Failed;
    }

    const fn illegal_transition(&self, action: &'static str) -> SessionError {
        SessionError::IllegalTransition {
            state: self.state.name(),
            action,
        }
    }
}

// the test helpers build Vec<u8> frames; this crate carries no alloc
// dependency for its no_std tier, so the suite needs std, not just test
#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use rstest::rstest;

    use super::{AuthFlow, Disposition, Session, SessionError, StateName, WirePhase};
    use super::super::frontend::{FrontendMessage, InitialMessage, parse_frontend, parse_initial};
    use super::super::types::{ProtocolVersion, StatementTarget, TransactionStatus};

    fn ssl_request_bytes() -> [u8; 8] {
        let mut buf = [0u8; 8];
        InitialMessage::SslRequest
            .encode(&mut buf)
            .expect("ssl request encodes");
        buf
    }

    fn gss_enc_request_bytes() -> [u8; 8] {
        let mut buf = [0u8; 8];
        InitialMessage::GssEncRequest
            .encode(&mut buf)
            .expect("gss enc request encodes");
        buf
    }

    fn startup_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        let raw: &[u8] = b"user\0testuser\0database\0testdb\0\0";
        let version = ProtocolVersion::V3_0.as_code().to_be_bytes();
        let total_len = (4 + 4 + raw.len()) as i32;
        buf[..4].copy_from_slice(&total_len.to_be_bytes());
        buf[4..8].copy_from_slice(&version);
        buf[8..8 + raw.len()].copy_from_slice(raw);
        buf.truncate(8 + raw.len());
        buf
    }

    fn cancel_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 16];
        let cancel = InitialMessage::Cancel(super::super::frontend::CancelRequest {
            process_id: 42,
            secret_key: &[1, 2, 3, 4],
        });
        let size = cancel.encode(&mut buf).expect("cancel encodes");
        buf.truncate(size);
        buf
    }

    fn auth_data_bytes(data: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; 5 + data.len()];
        let msg = FrontendMessage::AuthData(super::super::frontend::AuthData { data });
        let size = msg.encode(&mut buf).expect("auth data encodes");
        buf.truncate(size);
        buf
    }

    fn query_bytes(sql: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; 5 + sql.len() + 1];
        let msg = FrontendMessage::Query {
            sql: super::super::types::PgStr::new(sql),
        };
        let size = msg.encode(&mut buf).expect("query encodes");
        buf.truncate(size);
        buf
    }

    fn parse_msg_bytes() -> Vec<u8> {
        let statement: &[u8] = b"\x00";
        let sql: &[u8] = b"SELECT 1\x00";
        let param_count: &[u8] = &[0x00, 0x00];
        let body_len = statement.len() + sql.len() + param_count.len();
        let length = (4 + body_len) as i32;
        let mut buf = Vec::with_capacity(5 + body_len);
        buf.push(b'P');
        buf.extend_from_slice(&length.to_be_bytes());
        buf.extend_from_slice(statement);
        buf.extend_from_slice(sql);
        buf.extend_from_slice(param_count);
        buf
    }

    fn bind_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 128];
        let size = super::super::frontend::BindWriter::begin(&mut buf, b"", b"", &[])
            .expect("bind writer begins")
            .finish(&[])
            .expect("bind writer finishes");
        buf.truncate(size);
        buf
    }

    fn describe_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 16];
        let msg = FrontendMessage::Describe {
            target: StatementTarget::Statement,
            name: super::super::types::PgStr::new(b""),
        };
        let size = msg.encode(&mut buf).expect("describe encodes");
        buf.truncate(size);
        buf
    }

    fn execute_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 16];
        let msg = FrontendMessage::Execute {
            portal: super::super::types::PgStr::new(b""),
            max_rows: 0,
        };
        let size = msg.encode(&mut buf).expect("execute encodes");
        buf.truncate(size);
        buf
    }

    fn close_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 16];
        let msg = FrontendMessage::Close {
            target: StatementTarget::Statement,
            name: super::super::types::PgStr::new(b""),
        };
        let size = msg.encode(&mut buf).expect("close encodes");
        buf.truncate(size);
        buf
    }

    fn flush_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        let size = FrontendMessage::Flush
            .encode(&mut buf)
            .expect("flush encodes");
        buf.truncate(size);
        buf
    }

    fn sync_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        let size = FrontendMessage::Sync
            .encode(&mut buf)
            .expect("sync encodes");
        buf.truncate(size);
        buf
    }

    fn copy_data_bytes(data: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; 5 + data.len()];
        let msg = FrontendMessage::CopyData { data };
        let size = msg.encode(&mut buf).expect("copy data encodes");
        buf.truncate(size);
        buf
    }

    fn copy_done_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        let size = FrontendMessage::CopyDone
            .encode(&mut buf)
            .expect("copy done encodes");
        buf.truncate(size);
        buf
    }

    fn copy_fail_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 16];
        let msg = FrontendMessage::CopyFail {
            message: super::super::types::PgStr::new(b"oops"),
        };
        let size = msg.encode(&mut buf).expect("copy fail encodes");
        buf.truncate(size);
        buf
    }

    fn function_call_bytes() -> Vec<u8> {
        let oid: u32 = 1234;
        let body_len: i32 = 4 + 2 + 2 + 2;
        let length: i32 = 4 + body_len;
        let mut buf = Vec::with_capacity(5 + body_len as usize);
        buf.push(b'F');
        buf.extend_from_slice(&length.to_be_bytes());
        buf.extend_from_slice(&oid.to_be_bytes());
        buf.extend_from_slice(&0i16.to_be_bytes());
        buf.extend_from_slice(&0i16.to_be_bytes());
        buf.extend_from_slice(&0i16.to_be_bytes());
        buf
    }

    fn terminate_bytes() -> Vec<u8> {
        let mut buf = vec![0u8; 8];
        let size = FrontendMessage::Terminate
            .encode(&mut buf)
            .expect("terminate encodes");
        buf.truncate(size);
        buf
    }

    fn feed_initial(session: &mut Session, bytes: &[u8]) -> Result<Disposition, SessionError> {
        let (msg, _) = parse_initial(bytes)
            .expect("initial bytes parse ok")
            .expect("initial message complete");
        session.on_initial(&msg)
    }

    fn feed_frontend(session: &mut Session, bytes: &[u8]) -> Result<Disposition, SessionError> {
        let (msg, _) = parse_frontend(bytes)
            .expect("frontend bytes parse ok")
            .expect("frontend message complete");
        session.on_frontend(&msg)
    }

    fn startup_to_idle(session: &mut Session) {
        let startup = startup_bytes();
        feed_initial(session, &startup).expect("startup accepted");
        session.auth_ok().expect("auth_ok accepted");
        session.ready_for_query().expect("ready_for_query accepted");
    }

    #[test]
    fn ssl_refused_then_cleartext_startup_reaches_idle() {
        let mut session = Session::new();

        let ssl_bytes = ssl_request_bytes();
        let disposition = feed_initial(&mut session, &ssl_bytes).expect("ssl request accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::SslDecision);

        session.ssl_refused().expect("ssl_refused accepted");
        assert_eq!(session.state_name(), StateName::AwaitingInitial);

        let startup = startup_bytes();
        let disposition = feed_initial(&mut session, &startup).expect("startup accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::StartupReceived);

        session
            .auth_requested(AuthFlow::Cleartext)
            .expect("auth_requested accepted");
        assert_eq!(session.state_name(), StateName::AuthInProgress);
        assert_eq!(session.auth_flow(), Some(AuthFlow::Cleartext));

        let auth = auth_data_bytes(b"password\0");
        let disposition = feed_frontend(&mut session, &auth).expect("auth data accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::AuthInProgress);

        session.auth_ok().expect("auth_ok accepted");
        assert_eq!(session.state_name(), StateName::Starting);

        let status = session.ready_for_query().expect("ready_for_query accepted");
        assert_eq!(status, TransactionStatus::Idle);
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn ssl_accepted_tls_established_then_startup_reaches_idle() {
        let mut session = Session::new();

        let ssl_bytes = ssl_request_bytes();
        feed_initial(&mut session, &ssl_bytes).expect("ssl request accepted");
        session.ssl_accepted().expect("ssl_accepted accepted");
        assert_eq!(session.state_name(), StateName::TlsHandshake);

        session.tls_established().expect("tls_established accepted");
        assert_eq!(session.state_name(), StateName::AwaitingInitial);

        let startup = startup_bytes();
        feed_initial(&mut session, &startup).expect("startup over tls accepted");
        session.auth_ok().expect("auth_ok accepted");
        session.ready_for_query().expect("ready_for_query accepted");
        assert_eq!(session.state_name(), StateName::Idle);

        let second_ssl = ssl_request_bytes();
        let result = feed_initial(&mut session, &second_ssl);
        assert!(
            result.is_err(),
            "second SSLRequest after tls_established must be illegal"
        );
    }

    #[test]
    fn trust_auth_startup_then_auth_ok_directly_reaches_idle() {
        let mut session = Session::new();

        let startup = startup_bytes();
        feed_initial(&mut session, &startup).expect("startup accepted");
        assert_eq!(session.state_name(), StateName::StartupReceived);

        session
            .auth_ok()
            .expect("auth_ok from StartupReceived accepted");
        assert_eq!(session.state_name(), StateName::Starting);

        session.ready_for_query().expect("ready_for_query accepted");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn sasl_multi_round_auth_reaches_idle() {
        let mut session = Session::new();

        let startup = startup_bytes();
        feed_initial(&mut session, &startup).expect("startup accepted");
        session
            .auth_requested(AuthFlow::Sasl)
            .expect("auth_requested sasl accepted");
        assert_eq!(session.auth_flow(), Some(AuthFlow::Sasl));

        for _ in 0..3 {
            let auth = auth_data_bytes(b"sasl-round");
            let disposition = feed_frontend(&mut session, &auth).expect("sasl auth data accepted");
            assert_eq!(disposition, Disposition::Handle);
            assert_eq!(session.state_name(), StateName::AuthInProgress);
        }

        session.auth_ok().expect("auth_ok after sasl accepted");
        session.ready_for_query().expect("ready_for_query accepted");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn simple_query_cycle_repeats() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        for _ in 0..2 {
            let query = query_bytes(b"SELECT 1");
            let disposition = feed_frontend(&mut session, &query).expect("query accepted");
            assert_eq!(disposition, Disposition::Handle);
            assert_eq!(session.state_name(), StateName::SimpleQuery);

            session.ready_for_query().expect("ready_for_query accepted");
            assert_eq!(session.state_name(), StateName::Idle);
        }
    }

    #[test]
    fn extended_pipeline_parse_bind_describe_execute_flush_sync() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let parse = parse_msg_bytes();
        let disposition = feed_frontend(&mut session, &parse).expect("parse accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::Extended);

        for bytes in [
            bind_bytes(),
            describe_bytes(),
            execute_bytes(),
            close_bytes(),
            flush_bytes(),
        ] {
            let disposition = feed_frontend(&mut session, &bytes).expect("extended msg accepted");
            assert_eq!(disposition, Disposition::Handle);
            assert_eq!(session.state_name(), StateName::Extended);
        }

        let sync = sync_bytes();
        let disposition = feed_frontend(&mut session, &sync).expect("sync accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::Syncing);

        session
            .ready_for_query()
            .expect("ready_for_query from syncing accepted");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn extended_error_recovery_discards_until_sync() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let parse = parse_msg_bytes();
        feed_frontend(&mut session, &parse).expect("parse accepted");

        session.extended_error().expect("extended_error accepted");
        assert_eq!(session.state_name(), StateName::ExtendedFailed);

        for bytes in [
            parse_msg_bytes(),
            bind_bytes(),
            execute_bytes(),
            copy_data_bytes(b"data"),
            copy_done_bytes(),
            copy_fail_bytes(),
            flush_bytes(),
        ] {
            let disposition =
                feed_frontend(&mut session, &bytes).expect("discardable msg accepted");
            assert_eq!(disposition, Disposition::Discard);
            assert_eq!(session.state_name(), StateName::ExtendedFailed);
        }

        let sync = sync_bytes();
        let disposition =
            feed_frontend(&mut session, &sync).expect("sync accepted in extended failed");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::Syncing);

        session
            .ready_for_query()
            .expect("ready_for_query from syncing accepted");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn copy_in_simple_copydone_returns_to_simple_query() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let query = query_bytes(b"COPY t FROM STDIN");
        feed_frontend(&mut session, &query).expect("query accepted");
        assert_eq!(session.state_name(), StateName::SimpleQuery);

        session.copy_in_begun().expect("copy_in_begun accepted");
        assert_eq!(session.state_name(), StateName::CopyIn);

        for _ in 0..2 {
            let data = copy_data_bytes(b"row\n");
            let disposition = feed_frontend(&mut session, &data).expect("copy data accepted");
            assert_eq!(disposition, Disposition::Handle);
        }

        let done = copy_done_bytes();
        let disposition = feed_frontend(&mut session, &done).expect("copy done accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::SimpleQuery);

        session
            .ready_for_query()
            .expect("ready_for_query from simple query accepted");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn copy_in_extended_copyfail_enters_extended_failed_then_sync_recovers() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let parse = parse_msg_bytes();
        feed_frontend(&mut session, &parse).expect("parse accepted");
        assert_eq!(session.state_name(), StateName::Extended);

        session
            .copy_in_begun()
            .expect("copy_in_begun in extended accepted");
        assert_eq!(session.state_name(), StateName::CopyIn);

        let fail = copy_fail_bytes();
        let disposition = feed_frontend(&mut session, &fail).expect("copy fail accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::ExtendedFailed);

        let sync = sync_bytes();
        feed_frontend(&mut session, &sync).expect("sync accepted");
        assert_eq!(session.state_name(), StateName::Syncing);

        session
            .ready_for_query()
            .expect("ready_for_query after extended recovery");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn copy_out_simple_frontend_messages_are_discarded_copy_finished_returns_to_simple_query() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let query = query_bytes(b"COPY t TO STDOUT");
        feed_frontend(&mut session, &query).expect("query accepted");

        session.copy_out_begun().expect("copy_out_begun accepted");
        assert_eq!(session.state_name(), StateName::CopyOut);

        for bytes in [
            copy_data_bytes(b"row\n"),
            copy_done_bytes(),
            copy_fail_bytes(),
            flush_bytes(),
            sync_bytes(),
        ] {
            let disposition =
                feed_frontend(&mut session, &bytes).expect("copy out frontend msg accepted");
            assert_eq!(disposition, Disposition::Discard);
            assert_eq!(session.state_name(), StateName::CopyOut);
        }

        session.copy_finished().expect("copy_finished accepted");
        assert_eq!(session.state_name(), StateName::SimpleQuery);

        session
            .ready_for_query()
            .expect("ready_for_query after copy out");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn copy_both_copydone_handle_then_copy_finished() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let query = query_bytes(b"START_REPLICATION");
        feed_frontend(&mut session, &query).expect("query accepted");

        session.copy_both_begun().expect("copy_both_begun accepted");
        assert_eq!(session.state_name(), StateName::CopyBoth);

        for bytes in [copy_data_bytes(b"wal"), copy_done_bytes()] {
            let disposition = feed_frontend(&mut session, &bytes).expect("copy both msg accepted");
            assert_eq!(disposition, Disposition::Handle);
        }

        session
            .copy_finished()
            .expect("copy_finished from copy_both accepted");
        assert_eq!(session.state_name(), StateName::SimpleQuery);

        session
            .ready_for_query()
            .expect("ready_for_query after copy both");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn function_call_from_idle_reaches_function_call_active_then_idle() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let func_bytes = function_call_bytes();
        let disposition = feed_frontend(&mut session, &func_bytes).expect("function call accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::FunctionCallActive);

        session
            .ready_for_query()
            .expect("ready_for_query from function call active");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn cancel_connection_enters_cancelling_and_closes() {
        let mut session = Session::new();
        let cancel = cancel_bytes();
        let disposition = feed_initial(&mut session, &cancel).expect("cancel accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::Cancelling);
        assert_eq!(session.wire_phase(), WirePhase::Closed);
        assert!(session.is_closed());
    }

    #[rstest]
    #[case::from_idle(StateName::Idle)]
    #[case::from_extended(StateName::Extended)]
    #[case::from_extended_failed(StateName::ExtendedFailed)]
    #[case::from_auth_in_progress(StateName::AuthInProgress)]
    #[case::from_copy_in(StateName::CopyIn)]
    fn terminate_from_various_states_enters_terminated(#[case] target_state: StateName) {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        match target_state {
            StateName::Extended => {
                let parse = parse_msg_bytes();
                feed_frontend(&mut session, &parse).expect("parse accepted");
            }
            StateName::ExtendedFailed => {
                let parse = parse_msg_bytes();
                feed_frontend(&mut session, &parse).expect("parse accepted");
                session.extended_error().expect("extended_error accepted");
            }
            StateName::AuthInProgress => {
                let mut session2 = Session::new();
                let startup = startup_bytes();
                feed_initial(&mut session2, &startup).expect("startup accepted");
                session2
                    .auth_requested(AuthFlow::Cleartext)
                    .expect("auth_requested accepted");
                let terminate = terminate_bytes();
                let disposition = feed_frontend(&mut session2, &terminate)
                    .expect("terminate accepted in auth in progress");
                assert_eq!(disposition, Disposition::Handle);
                assert_eq!(session2.state_name(), StateName::Terminated);
                return;
            }
            StateName::CopyIn => {
                let query = query_bytes(b"COPY t FROM STDIN");
                feed_frontend(&mut session, &query).expect("query accepted");
                session.copy_in_begun().expect("copy_in_begun accepted");
            }
            _ => {}
        }

        let terminate = terminate_bytes();
        let disposition = feed_frontend(&mut session, &terminate).expect("terminate accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::Terminated);
        assert!(session.is_closed());
    }

    #[test]
    fn transaction_status_propagates_through_ready_for_query() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        session.set_transaction_status(TransactionStatus::InTransaction);
        assert_eq!(
            session.transaction_status(),
            TransactionStatus::InTransaction
        );

        let query = query_bytes(b"SELECT 1");
        feed_frontend(&mut session, &query).expect("query accepted");

        let status = session.ready_for_query().expect("ready_for_query accepted");
        assert_eq!(status, TransactionStatus::InTransaction);
        assert_eq!(
            session.transaction_status(),
            TransactionStatus::InTransaction
        );

        session.set_transaction_status(TransactionStatus::Idle);
        let query2 = query_bytes(b"SELECT 2");
        feed_frontend(&mut session, &query2).expect("query accepted");
        let status2 = session.ready_for_query().expect("ready_for_query accepted");
        assert_eq!(status2, TransactionStatus::Idle);
    }

    #[rstest]
    #[case::query_during_auth(b'Q')]
    #[case::parse_during_auth(b'P')]
    #[case::sync_during_auth(b'S')]
    fn illegal_messages_during_auth_in_progress(#[case] tag: u8) {
        let mut session = Session::new();
        let startup = startup_bytes();
        feed_initial(&mut session, &startup).expect("startup accepted");
        session
            .auth_requested(AuthFlow::Cleartext)
            .expect("auth_requested accepted");

        let bytes: Vec<u8> = match tag {
            b'Q' => query_bytes(b"SELECT 1"),
            b'P' => parse_msg_bytes(),
            b'S' => sync_bytes(),
            _ => unreachable!(),
        };
        let result = feed_frontend(&mut session, &bytes);
        let err = result.expect_err("should be illegal in AuthInProgress");
        assert!(
            matches!(
                err,
                SessionError::IllegalMessage {
                    state: StateName::AuthInProgress,
                    ..
                }
            ),
            "error should reference AuthInProgress state, got {err:?}"
        );
    }

    #[rstest]
    #[case::auth_data_in_idle(b'p')]
    #[case::copy_data_in_idle(b'd')]
    fn illegal_messages_in_idle(#[case] tag: u8) {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let bytes: Vec<u8> = match tag {
            b'p' => auth_data_bytes(b"password\0"),
            b'd' => copy_data_bytes(b"data"),
            _ => unreachable!(),
        };
        let result = feed_frontend(&mut session, &bytes);
        let err = result.expect_err("should be illegal in Idle");
        assert!(
            matches!(
                err,
                SessionError::IllegalMessage {
                    state: StateName::Idle,
                    ..
                }
            ),
            "error should reference Idle state, got {err:?}"
        );
    }

    #[test]
    fn function_call_illegal_during_extended() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let parse = parse_msg_bytes();
        feed_frontend(&mut session, &parse).expect("parse accepted");
        assert_eq!(session.state_name(), StateName::Extended);

        let func_bytes = function_call_bytes();
        let result = feed_frontend(&mut session, &func_bytes);
        let err = result.expect_err("function call should be illegal in Extended");
        assert!(
            matches!(
                err,
                SessionError::IllegalMessage {
                    state: StateName::Extended,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[rstest]
    #[case::query_during_simple_query(false, b'Q')]
    #[case::auth_data_in_copy_out(true, b'p')]
    fn illegal_messages_in_quiescent_states(#[case] use_copy_out: bool, #[case] tag: u8) {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        if use_copy_out {
            let query = query_bytes(b"COPY t TO STDOUT");
            feed_frontend(&mut session, &query).expect("query accepted");
            session.copy_out_begun().expect("copy_out_begun accepted");
        } else {
            let query = query_bytes(b"SELECT 1");
            feed_frontend(&mut session, &query).expect("query accepted");
        }

        let bytes: Vec<u8> = match tag {
            b'Q' => query_bytes(b"SELECT 2"),
            b'p' => auth_data_bytes(b"x"),
            _ => unreachable!(),
        };
        let result = feed_frontend(&mut session, &bytes);
        assert!(
            result.is_err(),
            "msg with tag {tag:#04x} should be illegal in this state"
        );
    }

    #[test]
    fn any_frontend_message_in_terminated_is_illegal() {
        let mut session = Session::new();
        startup_to_idle(&mut session);
        let terminate = terminate_bytes();
        feed_frontend(&mut session, &terminate).expect("terminate accepted");
        assert_eq!(session.state_name(), StateName::Terminated);

        let query = query_bytes(b"SELECT 1");
        let result = feed_frontend(&mut session, &query);
        let err = result.expect_err("query should be illegal in Terminated");
        assert!(
            matches!(
                err,
                SessionError::IllegalMessage {
                    state: StateName::Terminated,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn on_initial_after_startup_is_illegal() {
        let mut session = Session::new();
        let startup = startup_bytes();
        feed_initial(&mut session, &startup).expect("startup accepted");
        assert_eq!(session.state_name(), StateName::StartupReceived);

        let second_startup = startup_bytes();
        let result = feed_initial(&mut session, &second_startup);
        let err = result.expect_err("second on_initial should be illegal after startup");
        assert!(
            matches!(
                err,
                SessionError::IllegalMessage {
                    state: StateName::StartupReceived,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn second_ssl_request_is_illegal() {
        let mut session = Session::new();
        let ssl = ssl_request_bytes();
        feed_initial(&mut session, &ssl).expect("first ssl request accepted");
        session.ssl_refused().expect("ssl_refused accepted");

        let second_ssl = ssl_request_bytes();
        let result = feed_initial(&mut session, &second_ssl);
        let err = result.expect_err("second SSLRequest should be illegal");
        assert!(
            matches!(
                err,
                SessionError::IllegalMessage {
                    state: StateName::AwaitingInitial,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn second_gss_enc_request_is_illegal() {
        let mut session = Session::new();
        let gss = gss_enc_request_bytes();
        feed_initial(&mut session, &gss).expect("first gss request accepted");
        session.gss_enc_refused().expect("gss_enc_refused accepted");

        let second_gss = gss_enc_request_bytes();
        let result = feed_initial(&mut session, &second_gss);
        let err = result.expect_err("second GSSENCRequest should be illegal");
        assert!(
            matches!(
                err,
                SessionError::IllegalMessage {
                    state: StateName::AwaitingInitial,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[rstest]
    #[case::ssl_accepted_in_idle("ssl_accepted")]
    #[case::auth_ok_in_idle("auth_ok")]
    #[case::ready_for_query_in_extended("ready_for_query")]
    #[case::extended_error_in_simple_query("extended_error")]
    #[case::copy_in_begun_in_idle("copy_in_begun")]
    #[case::copy_finished_in_copy_in("copy_finished")]
    #[case::tls_established_without_ssl_accepted("tls_established")]
    #[case::auth_requested_twice("auth_requested_twice")]
    fn illegal_server_transitions(#[case] scenario: &str) {
        let mut session = Session::new();

        let err = match scenario {
            "ssl_accepted" => {
                startup_to_idle(&mut session);
                session
                    .ssl_accepted()
                    .expect_err("ssl_accepted in Idle must fail")
            }
            "auth_ok" => {
                startup_to_idle(&mut session);
                session.auth_ok().expect_err("auth_ok in Idle must fail")
            }
            "ready_for_query" => {
                startup_to_idle(&mut session);
                let parse = parse_msg_bytes();
                feed_frontend(&mut session, &parse).expect("parse accepted");
                session
                    .ready_for_query()
                    .expect_err("ready_for_query in Extended must fail")
            }
            "extended_error" => {
                startup_to_idle(&mut session);
                let query = query_bytes(b"SELECT 1");
                feed_frontend(&mut session, &query).expect("query accepted");
                session
                    .extended_error()
                    .expect_err("extended_error in SimpleQuery must fail")
            }
            "copy_in_begun" => {
                startup_to_idle(&mut session);
                session
                    .copy_in_begun()
                    .expect_err("copy_in_begun in Idle must fail")
            }
            "copy_finished" => {
                startup_to_idle(&mut session);
                let query = query_bytes(b"COPY t FROM STDIN");
                feed_frontend(&mut session, &query).expect("query accepted");
                session.copy_in_begun().expect("copy_in_begun accepted");
                session
                    .copy_finished()
                    .expect_err("copy_finished in CopyIn must fail")
            }
            "tls_established" => session
                .tls_established()
                .expect_err("tls_established in AwaitingInitial must fail"),
            "auth_requested_twice" => {
                let startup = startup_bytes();
                feed_initial(&mut session, &startup).expect("startup accepted");
                session
                    .auth_requested(AuthFlow::Cleartext)
                    .expect("first auth_requested ok");
                session
                    .auth_requested(AuthFlow::Md5)
                    .expect_err("second auth_requested must fail")
            }
            _ => unreachable!("unknown scenario: {scenario}"),
        };

        assert!(
            matches!(err, SessionError::IllegalTransition { .. }),
            "expected IllegalTransition, got {err:?}"
        );
    }

    #[rstest]
    #[case::awaiting_initial(StateName::AwaitingInitial, WirePhase::Initial)]
    #[case::ssl_decision(StateName::SslDecision, WirePhase::Quiescent)]
    #[case::gss_decision(StateName::GssDecision, WirePhase::Quiescent)]
    #[case::tls_handshake(StateName::TlsHandshake, WirePhase::Quiescent)]
    #[case::startup_received(StateName::StartupReceived, WirePhase::Quiescent)]
    #[case::auth_in_progress(StateName::AuthInProgress, WirePhase::Tagged)]
    #[case::starting(StateName::Starting, WirePhase::Quiescent)]
    #[case::idle(StateName::Idle, WirePhase::Tagged)]
    #[case::simple_query(StateName::SimpleQuery, WirePhase::Quiescent)]
    #[case::extended(StateName::Extended, WirePhase::Tagged)]
    #[case::extended_failed(StateName::ExtendedFailed, WirePhase::Tagged)]
    #[case::syncing(StateName::Syncing, WirePhase::Quiescent)]
    #[case::copy_in(StateName::CopyIn, WirePhase::Tagged)]
    #[case::copy_out(StateName::CopyOut, WirePhase::Tagged)]
    #[case::copy_both(StateName::CopyBoth, WirePhase::Tagged)]
    #[case::function_call_active(StateName::FunctionCallActive, WirePhase::Quiescent)]
    #[case::cancelling(StateName::Cancelling, WirePhase::Closed)]
    #[case::terminated(StateName::Terminated, WirePhase::Closed)]
    #[case::failed(StateName::Failed, WirePhase::Closed)]
    fn wire_phase_matches_state(#[case] state_name: StateName, #[case] expected_phase: WirePhase) {
        let session = reach_state(state_name);
        assert_eq!(
            session.wire_phase(),
            expected_phase,
            "state {state_name:?} should map to {expected_phase:?}"
        );
    }

    fn reach_state(target: StateName) -> Session {
        let mut session = Session::new();
        match target {
            StateName::AwaitingInitial => {}
            StateName::SslDecision => {
                let ssl = ssl_request_bytes();
                feed_initial(&mut session, &ssl).expect("ssl request accepted");
            }
            StateName::GssDecision => {
                let gss = gss_enc_request_bytes();
                feed_initial(&mut session, &gss).expect("gss request accepted");
            }
            StateName::TlsHandshake => {
                let ssl = ssl_request_bytes();
                feed_initial(&mut session, &ssl).expect("ssl request accepted");
                session.ssl_accepted().expect("ssl_accepted accepted");
            }
            StateName::StartupReceived => {
                let startup = startup_bytes();
                feed_initial(&mut session, &startup).expect("startup accepted");
            }
            StateName::AuthInProgress => {
                let startup = startup_bytes();
                feed_initial(&mut session, &startup).expect("startup accepted");
                session
                    .auth_requested(AuthFlow::Cleartext)
                    .expect("auth_requested accepted");
            }
            StateName::Starting => {
                let startup = startup_bytes();
                feed_initial(&mut session, &startup).expect("startup accepted");
                session.auth_ok().expect("auth_ok accepted");
            }
            StateName::Idle => {
                startup_to_idle(&mut session);
            }
            StateName::SimpleQuery => {
                startup_to_idle(&mut session);
                let query = query_bytes(b"SELECT 1");
                feed_frontend(&mut session, &query).expect("query accepted");
            }
            StateName::Extended => {
                startup_to_idle(&mut session);
                let parse = parse_msg_bytes();
                feed_frontend(&mut session, &parse).expect("parse accepted");
            }
            StateName::ExtendedFailed => {
                startup_to_idle(&mut session);
                let parse = parse_msg_bytes();
                feed_frontend(&mut session, &parse).expect("parse accepted");
                session.extended_error().expect("extended_error accepted");
            }
            StateName::Syncing => {
                startup_to_idle(&mut session);
                let sync = sync_bytes();
                feed_frontend(&mut session, &sync).expect("sync accepted");
            }
            StateName::CopyIn => {
                startup_to_idle(&mut session);
                let query = query_bytes(b"COPY t FROM STDIN");
                feed_frontend(&mut session, &query).expect("query accepted");
                session.copy_in_begun().expect("copy_in_begun accepted");
            }
            StateName::CopyOut => {
                startup_to_idle(&mut session);
                let query = query_bytes(b"COPY t TO STDOUT");
                feed_frontend(&mut session, &query).expect("query accepted");
                session.copy_out_begun().expect("copy_out_begun accepted");
            }
            StateName::CopyBoth => {
                startup_to_idle(&mut session);
                let query = query_bytes(b"START_REPLICATION");
                feed_frontend(&mut session, &query).expect("query accepted");
                session.copy_both_begun().expect("copy_both_begun accepted");
            }
            StateName::FunctionCallActive => {
                startup_to_idle(&mut session);
                let func_bytes = function_call_bytes();
                feed_frontend(&mut session, &func_bytes).expect("function call accepted");
            }
            StateName::Cancelling => {
                let cancel = cancel_bytes();
                feed_initial(&mut session, &cancel).expect("cancel accepted");
            }
            StateName::Terminated => {
                startup_to_idle(&mut session);
                let terminate = terminate_bytes();
                feed_frontend(&mut session, &terminate).expect("terminate accepted");
            }
            StateName::Failed => {
                session.fail();
            }
        }
        session
    }

    #[test]
    fn auth_failed_moves_to_failed_and_closes() {
        let mut session = Session::new();
        let startup = startup_bytes();
        feed_initial(&mut session, &startup).expect("startup accepted");
        session
            .auth_requested(AuthFlow::Cleartext)
            .expect("auth_requested accepted");
        session.auth_failed().expect("auth_failed accepted");
        assert_eq!(session.state_name(), StateName::Failed);
        assert!(session.is_closed());
    }

    #[test]
    fn fail_from_idle_moves_to_failed_and_closes() {
        let mut session = Session::new();
        startup_to_idle(&mut session);
        session.fail();
        assert_eq!(session.state_name(), StateName::Failed);
        assert!(session.is_closed());
    }

    #[test]
    fn copy_fail_in_copy_both_is_illegal() {
        let mut session = Session::new();
        startup_to_idle(&mut session);
        let query = query_bytes(b"START_REPLICATION");
        feed_frontend(&mut session, &query).expect("query accepted");
        session.copy_both_begun().expect("copy_both_begun accepted");

        let fail = copy_fail_bytes();
        let result = feed_frontend(&mut session, &fail);
        let err = result.expect_err("copy_fail in CopyBoth must be illegal");
        assert!(
            matches!(
                err,
                SessionError::IllegalMessage {
                    state: StateName::CopyBoth,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn sync_from_idle_enters_syncing_then_idle() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let sync = sync_bytes();
        let disposition = feed_frontend(&mut session, &sync).expect("sync from idle accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::Syncing);

        session
            .ready_for_query()
            .expect("ready_for_query from syncing accepted");
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn flush_from_idle_is_handle_stays_idle() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let flush = flush_bytes();
        let disposition = feed_frontend(&mut session, &flush).expect("flush from idle accepted");
        assert_eq!(disposition, Disposition::Handle);
        assert_eq!(session.state_name(), StateName::Idle);
    }

    #[test]
    fn copy_out_extended_copy_finished_returns_to_extended() {
        let mut session = Session::new();
        startup_to_idle(&mut session);

        let parse = parse_msg_bytes();
        feed_frontend(&mut session, &parse).expect("parse accepted");
        assert_eq!(session.state_name(), StateName::Extended);

        session
            .copy_out_begun()
            .expect("copy_out_begun from extended accepted");
        assert_eq!(session.state_name(), StateName::CopyOut);

        session
            .copy_finished()
            .expect("copy_finished from extended copy_out accepted");
        assert_eq!(session.state_name(), StateName::Extended);
    }

    #[test]
    fn gss_refused_then_ssl_refused_then_startup() {
        let mut session = Session::new();

        let gss = gss_enc_request_bytes();
        feed_initial(&mut session, &gss).expect("gss request accepted");
        session.gss_enc_refused().expect("gss_enc_refused accepted");
        assert_eq!(session.state_name(), StateName::AwaitingInitial);

        let ssl = ssl_request_bytes();
        feed_initial(&mut session, &ssl).expect("ssl request after gss refused accepted");
        session.ssl_refused().expect("ssl_refused accepted");
        assert_eq!(session.state_name(), StateName::AwaitingInitial);

        let startup = startup_bytes();
        feed_initial(&mut session, &startup).expect("startup after both refused accepted");
        assert_eq!(session.state_name(), StateName::StartupReceived);
    }

    #[test]
    fn default_transaction_status_is_idle() {
        let session = Session::new();
        assert_eq!(session.transaction_status(), TransactionStatus::Idle);
    }

    #[test]
    fn auth_flow_returns_none_outside_auth_in_progress() {
        let mut session = Session::new();
        assert_eq!(session.auth_flow(), None);
        startup_to_idle(&mut session);
        assert_eq!(session.auth_flow(), None);
    }
}
