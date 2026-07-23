//! Sans-IO memcached client session — the protocol state machine, no I/O.
//!
//! Bytes in (`feed`), bytes out (`take_outbound`), driven by `advance()`.
//! The client-side mirror of `proxima_redis::client::ClientSession`: it
//! owns the request/reply exchange, but never touches a socket (workspace
//! principle 11). A blocking driver, an async driver, and the
//! `PipeFactory` client all wrap it — that is what makes the client
//! agnostic to the transport shape.
//!
//! Simpler than RESP's session: there is no startup handshake phase (see
//! [`crate::client::config::MemcachedClientConfig`]'s docs), so the FSM is
//! just "idle" vs "a reply is pending" — no `Phase::Handshake`.

use proxima_protocols::memcached::pipe_contract::encode_request;
use proxima_protocols::memcached::{MemcachedRequest, ParseError, Reply, ReplyHint, parse_reply};

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("server connection closed mid-reply")]
    Closed,
    #[error("submit while a reply is pending")]
    ReplyPending,
    #[error("reply: {0}")]
    Reply(#[from] ParseError),
}

/// What the driver must do next to advance the session. The driver owns
/// I/O; the session owns the protocol.
#[derive(Debug)]
pub enum Step {
    /// Bytes are queued — write `take_outbound()` to the transport, then
    /// call `advance()` again.
    Send,
    /// No progress without more inbound bytes — read, `feed()`, then
    /// `advance()` again.
    Recv,
    /// Idle: no reply outstanding, ready for `submit`.
    Ready,
    /// The in-flight command's reply.
    Complete(Reply),
}

pub struct ClientSession {
    inbox: Vec<u8>,
    outbound: Vec<u8>,
    /// `Some` while a reply is outstanding, carrying the hint that
    /// resolves [`parse_reply`]'s `get`/`stats` block-vs-bare-`END`
    /// ambiguity. `None` when idle, including right after submitting a
    /// `noreply`-flagged command (the wire contract guarantees no reply
    /// will ever arrive for it).
    pending: Option<ReplyHint>,
}

impl Default for ClientSession {
    fn default() -> Self {
        Self::new()
    }
}

fn reply_hint_for(request: &MemcachedRequest) -> ReplyHint {
    match request {
        MemcachedRequest::Get { .. } => ReplyHint::Get,
        MemcachedRequest::Stats { .. } => ReplyHint::Stats,
        _ => ReplyHint::Simple,
    }
}

impl ClientSession {
    /// Builds an idle session — there is no handshake to queue (see the
    /// module docs), so this never touches `config` beyond what a future
    /// SASL-auth extension would need.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inbox: Vec::with_capacity(8192),
            outbound: Vec::with_capacity(64),
            pending: None,
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

    /// Queues one command. A `noreply`-flagged request never sets
    /// `pending` — the next `advance()` reports `Ready` right after the
    /// bytes are sent, since the wire contract guarantees the server
    /// answers nothing.
    ///
    /// # Errors
    /// [`ClientError::ReplyPending`] if a previous reply is still
    /// outstanding.
    pub fn submit(&mut self, request: &MemcachedRequest) -> Result<(), ClientError> {
        if self.pending.is_some() {
            return Err(ClientError::ReplyPending);
        }
        encode_request(request, &mut self.outbound);
        if !request.is_noreply() {
            self.pending = Some(reply_hint_for(request));
        }
        Ok(())
    }

    /// Advances the state machine: sends queued bytes, then parses the
    /// inbound reply once one is outstanding.
    ///
    /// # Errors
    /// [`ClientError::Reply`] on a malformed reply.
    pub fn advance(&mut self) -> Result<Step, ClientError> {
        if !self.outbound.is_empty() {
            return Ok(Step::Send);
        }
        match self.pending {
            None => Ok(Step::Ready),
            Some(hint) => match self.next_reply(hint)? {
                None => Ok(Step::Recv),
                Some(reply) => {
                    self.pending = None;
                    Ok(Step::Complete(reply))
                }
            },
        }
    }

    /// Parses one reply from the inbox, owning it and draining the
    /// consumed bytes.
    fn next_reply(&mut self, hint: ReplyHint) -> Result<Option<Reply>, ClientError> {
        match parse_reply(&self.inbox, hint) {
            Err(ParseError::Short | ParseError::PartialValue(_)) => Ok(None),
            Err(error) => Err(ClientError::Reply(error)),
            Ok((reply, consumed)) => {
                self.inbox.drain(..consumed);
                Ok(Some(reply))
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use proxima_protocols::memcached::StoreMode;

    #[test]
    fn new_session_is_immediately_ready() {
        let mut session = ClientSession::new();
        assert!(matches!(session.advance().expect("advance"), Step::Ready));
    }

    #[test]
    fn get_round_trips_a_reply() {
        let mut session = ClientSession::new();
        session
            .submit(&MemcachedRequest::Get {
                keys: Bytes::from_static(b"mykey"),
                gets: false,
            })
            .expect("submit");

        match session.advance().expect("advance") {
            Step::Send => {}
            other => panic!("expected Send, got {other:?}"),
        }
        assert_eq!(session.take_outbound(), b"get mykey\r\n");

        match session.advance().expect("advance") {
            Step::Recv => {}
            other => panic!("expected Recv, got {other:?}"),
        }

        session.feed(b"VALUE mykey 0 5\r\nhello\r\nEND\r\n");
        match session.advance().expect("advance") {
            Step::Complete(Reply::Values(values)) => {
                assert_eq!(values.len(), 1);
                assert_eq!(values[0].data, b"hello");
            }
            other => panic!("expected Complete(Values), got {other:?}"),
        }
        assert!(matches!(session.advance().expect("advance"), Step::Ready));
    }

    #[test]
    fn set_round_trips_stored() {
        let mut session = ClientSession::new();
        session
            .submit(&MemcachedRequest::Store {
                mode: StoreMode::Set,
                key: Bytes::from_static(b"k"),
                flags: 0,
                exptime: 0,
                value: Bytes::from_static(b"abc"),
                noreply: false,
            })
            .expect("submit");
        let _ = session.take_outbound();
        session.feed(b"STORED\r\n");
        match session.advance().expect("advance") {
            Step::Complete(Reply::Stored) => {}
            other => panic!("expected Complete(Stored), got {other:?}"),
        }
    }

    #[test]
    fn noreply_command_returns_to_ready_without_waiting_for_a_reply() {
        let mut session = ClientSession::new();
        session
            .submit(&MemcachedRequest::Delete {
                key: Bytes::from_static(b"k"),
                noreply: true,
            })
            .expect("submit");
        assert_eq!(session.take_outbound(), b"delete k noreply\r\n");
        assert!(matches!(session.advance().expect("advance"), Step::Ready));
    }

    #[test]
    fn submit_while_a_reply_is_pending_is_rejected() {
        let mut session = ClientSession::new();
        session
            .submit(&MemcachedRequest::Version)
            .expect("first submit");
        let outcome = session.submit(&MemcachedRequest::Version);
        assert!(matches!(outcome, Err(ClientError::ReplyPending)));
    }

    #[test]
    fn malformed_reply_surfaces_as_a_client_error() {
        let mut session = ClientSession::new();
        session
            .submit(&MemcachedRequest::Version)
            .expect("submit");
        let _ = session.take_outbound();
        session.feed(b"BOGUS\r\n");
        assert!(matches!(session.advance(), Err(ClientError::Reply(_))));
    }
}
