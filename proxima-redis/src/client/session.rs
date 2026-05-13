//! Sans-IO Redis/Valkey client session — the protocol state machine, no I/O.
//!
//! Bytes in (`feed`), bytes out (`take_outbound`), driven by `advance()`. The
//! client-side mirror of pgwire's `ClientSession`: it owns the startup
//! handshake (`HELLO 3` / `AUTH` / `SELECT`) and the request/reply exchange,
//! but never touches a socket (workspace principle 11). A blocking driver, an
//! async driver, and the `PipeFactory` client all wrap it — that is what makes
//! the client agnostic to the transport shape.
//!
//! The FSM is a two-state enum ([`Phase`]): `Handshake` drains a queue of
//! startup commands, validating each reply is not a server error; `Ready`
//! accepts one user command at a time and yields its single reply. Pub/sub and
//! MONITOR leave the request/reply rhythm — after the first reply the driver
//! reads pushed frames with [`ClientSession::poll_push`].

use std::collections::VecDeque;

use zeroize::Zeroizing;

use proxima_protocols::redis::{Frame, ParseError, RespValue, encode_command, parse};

use crate::client::config::{RedisClientConfig, RespProtocol};

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// A server `-ERR` / `!blob-error` reply during the startup handshake
    /// (bad password, unknown command on an old server, missing database).
    #[error("server: {0}")]
    Server(String),
    #[error("server connection closed mid-reply")]
    Closed,
    #[error("protocol: {0}")]
    Protocol(&'static str),
}

/// What the driver must do next to advance the session. The driver owns I/O;
/// the session owns the protocol.
#[derive(Debug)]
pub enum Step {
    /// Bytes are queued — write `take_outbound()` to the transport, then call
    /// `advance()` again.
    Send,
    /// No progress without more inbound bytes — read, `feed()`, then
    /// `advance()` again.
    Recv,
    /// Startup complete; the session is idle and ready for `submit`.
    Ready,
    /// The in-flight command's reply.
    Complete(RespValue),
}

/// One step of the pub/sub / MONITOR push loop, driven after the subscribe
/// reply lands.
#[derive(Debug)]
pub enum PushStep {
    /// Need more inbound bytes for a complete push frame.
    Recv,
    /// One pushed frame (a pub/sub message, an invalidation, a MONITOR line).
    Frame(RespValue),
}

#[derive(Debug, PartialEq, Eq)]
enum Phase {
    Handshake,
    Ready,
}

pub struct ClientSession {
    inbox: Vec<u8>,
    outbound: Vec<u8>,
    /// remaining startup commands (`HELLO` / `AUTH` / `SELECT`), each fully
    /// encoded; popped one at a time so a failure surfaces against its command.
    handshake: VecDeque<Vec<u8>>,
    /// a handshake command is on the wire awaiting its reply.
    handshake_awaiting: bool,
    /// a user command is on the wire awaiting its single reply.
    pending: bool,
    phase: Phase,
}

impl ClientSession {
    /// Builds a session and queues the startup handshake derived from `config`
    /// (RESP3 `HELLO 3 [AUTH ...]` or RESP2 `AUTH`, then `SELECT` for a
    /// non-zero database). The password is copied into a `Zeroizing` buffer only
    /// long enough to encode the command bytes, then wiped.
    #[must_use]
    pub fn new(config: &RedisClientConfig) -> Self {
        let mut handshake = VecDeque::new();
        let password = Zeroizing::new(config.password.clone().into_bytes());
        let user: &str = if config.username.is_empty() {
            "default"
        } else {
            config.username.as_str()
        };

        match config.protocol() {
            RespProtocol::Resp3 => {
                let mut argv: Vec<&[u8]> = vec![b"HELLO", b"3"];
                if !config.password.is_empty() {
                    argv.extend_from_slice(&[b"AUTH", user.as_bytes(), password.as_slice()]);
                }
                handshake.push_back(encode_argv(&argv));
            }
            RespProtocol::Resp2 => {
                if !config.password.is_empty() {
                    let mut argv: Vec<&[u8]> = vec![b"AUTH"];
                    if !config.username.is_empty() {
                        argv.push(user.as_bytes());
                    }
                    argv.push(password.as_slice());
                    handshake.push_back(encode_argv(&argv));
                }
            }
        }
        if config.db != 0 {
            let db = config.db.to_string();
            handshake.push_back(encode_argv(&[b"SELECT", db.as_bytes()]));
        }

        Self {
            inbox: Vec::with_capacity(8192),
            outbound: Vec::with_capacity(64),
            handshake,
            handshake_awaiting: false,
            pending: false,
            phase: Phase::Handshake,
        }
    }

    /// Drains the bytes the driver must send.
    pub fn take_outbound(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.outbound)
    }

    /// Appends bytes the driver read from the transport.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.inbox.extend_from_slice(bytes);
    }

    /// Encodes a command (verb + args) onto the outbound buffer without marking
    /// a reply pending — the pub/sub / MONITOR path, where the driver then reads
    /// pushed frames with [`Self::poll_push`].
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if the session is not yet ready.
    pub fn queue_command(&mut self, argv: &[&[u8]]) -> Result<(), ClientError> {
        if self.phase != Phase::Ready {
            return Err(ClientError::Protocol("command before ready"));
        }
        encode_command(argv, &mut self.outbound);
        Ok(())
    }

    /// Queues a request/reply command. Only valid once `Ready` and with no other
    /// reply outstanding.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if not ready or a reply is already pending.
    pub fn submit(&mut self, argv: &[&[u8]]) -> Result<(), ClientError> {
        if self.pending {
            return Err(ClientError::Protocol("submit while a reply is pending"));
        }
        self.queue_command(argv)?;
        self.pending = true;
        Ok(())
    }

    /// Advances the state machine: sends queued bytes, then parses inbound
    /// frames until it needs more bytes or reaches a checkpoint.
    ///
    /// # Errors
    /// [`ClientError`] on a server error during the handshake or a malformed
    /// frame.
    pub fn advance(&mut self) -> Result<Step, ClientError> {
        if !self.outbound.is_empty() {
            return Ok(Step::Send);
        }
        match self.phase {
            Phase::Handshake => self.advance_handshake(),
            Phase::Ready => self.advance_ready(),
        }
    }

    /// Reads one pushed frame (pub/sub message, MONITOR line) without sending.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] on a malformed frame.
    pub fn poll_push(&mut self) -> Result<PushStep, ClientError> {
        match self.next_reply()? {
            None => Ok(PushStep::Recv),
            Some(value) => Ok(PushStep::Frame(value)),
        }
    }

    fn advance_handshake(&mut self) -> Result<Step, ClientError> {
        if self.handshake_awaiting {
            match self.next_reply()? {
                None => return Ok(Step::Recv),
                Some(value) => {
                    if let Some(message) = value.as_error() {
                        return Err(ClientError::Server(message.to_string()));
                    }
                    self.handshake_awaiting = false;
                }
            }
        }
        if let Some(command) = self.handshake.pop_front() {
            self.outbound.extend_from_slice(&command);
            self.handshake_awaiting = true;
            return Ok(Step::Send);
        }
        self.phase = Phase::Ready;
        Ok(Step::Ready)
    }

    fn advance_ready(&mut self) -> Result<Step, ClientError> {
        if !self.pending {
            return Ok(Step::Ready);
        }
        match self.next_reply()? {
            None => Ok(Step::Recv),
            Some(value) => {
                self.pending = false;
                Ok(Step::Complete(value))
            }
        }
    }

    /// Parses one logical reply from the inbox, owning it and draining the
    /// consumed bytes. RESP3 attribute frames are out-of-band metadata that
    /// precede the real reply, so they are consumed and skipped transparently.
    fn next_reply(&mut self) -> Result<Option<RespValue>, ClientError> {
        loop {
            let (value, consumed, is_attribute) = match parse(&self.inbox) {
                Err(ParseError::NeedMore) => return Ok(None),
                Err(ParseError::Malformed(reason)) => return Err(ClientError::Protocol(reason)),
                Ok((frame, consumed)) => {
                    let is_attribute = matches!(frame, Frame::Attribute(_));
                    (RespValue::from_frame(&frame), consumed, is_attribute)
                }
            };
            self.inbox.drain(..consumed);
            if !is_attribute {
                return Ok(Some(value));
            }
        }
    }
}

fn encode_argv(argv: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    encode_command(argv, &mut out);
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn drive_handshake(session: &mut ClientSession, server_replies: &[&[u8]]) {
        let mut reply_index = 0;
        loop {
            match session.advance().expect("advance") {
                Step::Send => {
                    let _sent = session.take_outbound();
                    if reply_index < server_replies.len() {
                        session.feed(server_replies[reply_index]);
                        reply_index += 1;
                    }
                }
                Step::Recv => {
                    if reply_index < server_replies.len() {
                        session.feed(server_replies[reply_index]);
                        reply_index += 1;
                    } else {
                        panic!("session wants more bytes but the script is exhausted");
                    }
                }
                Step::Ready => return,
                Step::Complete(_) => panic!("unexpected reply during handshake"),
            }
        }
    }

    #[test]
    fn resp2_no_auth_is_ready_without_a_round_trip() {
        let config = RedisClientConfig::builder().resp3(false).build();
        let mut session = ClientSession::new(&config);
        match session.advance().expect("advance") {
            Step::Ready => {}
            other => panic!("expected immediate Ready, got {other:?}"),
        }
    }

    #[test]
    fn resp3_handshake_sends_hello_then_becomes_ready() {
        let config = RedisClientConfig::default();
        let mut session = ClientSession::new(&config);

        // first advance queues HELLO 3
        match session.advance().expect("advance") {
            Step::Send => {}
            other => panic!("expected Send (HELLO), got {other:?}"),
        }
        let sent = session.take_outbound();
        assert_eq!(sent, b"*2\r\n$5\r\nHELLO\r\n$1\r\n3\r\n");

        // server answers with a (minimal) HELLO map
        session.feed(b"%1\r\n$6\r\nserver\r\n$5\r\nredis\r\n");
        match session.advance().expect("advance") {
            Step::Ready => {}
            other => panic!("expected Ready after HELLO reply, got {other:?}"),
        }
    }

    #[test]
    fn resp3_auth_and_select_are_queued_in_order() {
        let config = RedisClientConfig::builder()
            .password("hunter2")
            .username("alice")
            .db(2)
            .build();
        let mut session = ClientSession::new(&config);

        match session.advance().expect("advance") {
            Step::Send => {}
            other => panic!("expected Send, got {other:?}"),
        }
        let hello = session.take_outbound();
        assert_eq!(
            hello,
            b"*5\r\n$5\r\nHELLO\r\n$1\r\n3\r\n$4\r\nAUTH\r\n$5\r\nalice\r\n$7\r\nhunter2\r\n"
        );
        session.feed(b"%0\r\n");

        // next advance queues SELECT 2
        match session.advance().expect("advance") {
            Step::Send => {}
            other => panic!("expected Send (SELECT), got {other:?}"),
        }
        assert_eq!(
            session.take_outbound(),
            b"*2\r\n$6\r\nSELECT\r\n$1\r\n2\r\n"
        );
        session.feed(b"+OK\r\n");
        assert!(matches!(session.advance().expect("advance"), Step::Ready));
    }

    #[test]
    fn handshake_surfaces_server_error() {
        let config = RedisClientConfig::builder().password("wrong").build();
        let mut session = ClientSession::new(&config);
        let _ = session.advance().expect("advance");
        let _ = session.take_outbound();
        session.feed(b"-WRONGPASS invalid username-password pair\r\n");
        match session.advance() {
            Err(ClientError::Server(message)) => assert!(message.contains("WRONGPASS")),
            other => panic!("expected server error, got {other:?}"),
        }
    }

    #[test]
    fn ready_command_round_trips_a_reply() {
        let config = RedisClientConfig::builder().resp3(false).build();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, &[]);

        session.submit(&[b"GET", b"mykey"]).expect("submit");
        assert_eq!(
            session.take_outbound(),
            b"*2\r\n$3\r\nGET\r\n$5\r\nmykey\r\n"
        );
        session.feed(b"$5\r\nhello\r\n");
        match session.advance().expect("advance") {
            Step::Complete(value) => assert_eq!(value, RespValue::BulkString(b"hello".to_vec())),
            other => panic!("expected Complete, got {other:?}"),
        }
        // back to idle-ready
        assert!(matches!(session.advance().expect("advance"), Step::Ready));
    }

    #[test]
    fn poll_push_reads_subsequent_messages() {
        let config = RedisClientConfig::builder().resp3(false).build();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, &[]);

        session
            .queue_command(&[b"SUBSCRIBE", b"news"])
            .expect("queue");
        let _ = session.take_outbound();
        session.feed(b">3\r\n$7\r\nmessage\r\n$4\r\nnews\r\n$5\r\nhello\r\n");
        match session.poll_push().expect("poll") {
            PushStep::Frame(RespValue::Push(items)) => {
                assert_eq!(items[0], RespValue::BulkString(b"message".to_vec()));
                assert_eq!(items[2], RespValue::BulkString(b"hello".to_vec()));
            }
            other => panic!("expected a push frame, got {other:?}"),
        }
        assert!(matches!(session.poll_push().expect("poll"), PushStep::Recv));
    }

    #[test]
    fn attribute_frame_is_skipped_before_the_real_reply() {
        let config = RedisClientConfig::builder().resp3(false).build();
        let mut session = ClientSession::new(&config);
        drive_handshake(&mut session, &[]);

        session.submit(&[b"GET", b"k"]).expect("submit");
        let _ = session.take_outbound();
        // an attribute map precedes the actual integer reply
        session.feed(b"|1\r\n$3\r\nttl\r\n:60\r\n:7\r\n");
        match session.advance().expect("advance") {
            Step::Complete(value) => assert_eq!(value, RespValue::Integer(7)),
            other => panic!("expected Complete(7), got {other:?}"),
        }
    }
}
