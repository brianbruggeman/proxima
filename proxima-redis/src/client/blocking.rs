//! Blocking driver over the sans-IO [`ClientSession`] — a `std::io::Read +
//! Write` transport (e.g. `std::net::TcpStream`). Used by the real-server parity
//! harness (capture + differential); the async Pipe driver wraps the same
//! session over a futures-io transport. The session owns the protocol; this only
//! moves bytes.
//!
//! [`RedisClient`]'s second type parameter, `State`, is a client-side typestate
//! mirror of the server's runtime `ConnMode` FSM
//! (`proxima_protocols::redis::ConnMode` — see
//! `proxima-protocols/src/redis/connection.rs`): [`Active`] pairs with
//! `ConnMode::Command` (the general RESP command path), [`Subscribed`] with
//! `ConnMode::Subscriber` (only the pub/sub push loop and the subscriber-safe
//! verbs). `State` defaults to `Active` so every existing `RedisClient<S>`
//! call site keeps compiling unchanged. `subscribe`/`psubscribe` consume an
//! `Active` client and return a `Subscribed` one; `unsubscribe_all` consumes a
//! `Subscribed` client and returns to `Active` — an illegal call (`command` on
//! `Subscribed`) is a compile error, not a runtime gate check (workspace
//! principle 11).

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::marker::PhantomData;

use proxima_protocols::redis::{RespValue, encode_command};

use crate::client::config::RedisClientConfig;
use crate::client::session::{ClientError, ClientSession, PushStep, Step};

/// Client-side mirror of the server's `ConnMode::Command` — the general RESP
/// [`RedisClient::command`] path is available.
pub struct Active;

/// Client-side mirror of the server's `ConnMode::Subscriber`, reached via
/// [`RedisClient::subscribe`] / [`RedisClient::psubscribe`]. Only the pub/sub
/// push loop ([`RedisClient::next_push`]) and the return trip
/// ([`RedisClient::unsubscribe_all`]) are available — `command` is absent at
/// compile time, matching real Redis rejecting most commands in subscriber
/// context.
///
/// ```compile_fail
/// use std::net::TcpStream;
/// use proxima_redis::client::{RedisClient, Subscribed};
///
/// fn demo(mut client: RedisClient<TcpStream, Subscribed>) {
///     let _ = client.command(&[b"GET", b"k"]); // ERROR: no method `command` on Subscribed
/// }
/// ```
pub struct Subscribed;

pub struct RedisClient<S, State = Active> {
    stream: S,
    session: ClientSession,
    capture: bool,
    /// Every server byte read, when capture is enabled — the verbatim
    /// server->client stream for fixture vendoring.
    pub captured: Vec<u8>,
    /// Exact-channel subscriptions this client has open, tracked so
    /// `unsubscribe_all` knows exactly how many `unsubscribe` ack frames to
    /// drain — the server acks one frame per *distinct* open channel
    /// (`Connection::subscribe`/`SubscriberState` both dedupe into a set), so
    /// this mirrors that with a set rather than a bare count: a bare count of
    /// arguments passed would desync (and hang the drain loop) the moment the
    /// same channel is named twice.
    channels: BTreeSet<Vec<u8>>,
    /// Pattern subscriptions this client has open; same set-based bookkeeping
    /// as `channels`, for `PUNSUBSCRIBE`.
    patterns: BTreeSet<Vec<u8>>,
    _state: PhantomData<State>,
}

impl<S: Read + Write, State> RedisClient<S, State> {
    fn flush_outbound(&mut self) -> Result<(), ClientError> {
        let bytes = self.session.take_outbound();
        self.stream.write_all(&bytes)?;
        self.stream.flush()?;
        Ok(())
    }

    fn read_into_session(&mut self) -> Result<(), ClientError> {
        let mut chunk = [0_u8; 8192];
        let read = self.stream.read(&mut chunk)?;
        if read == 0 {
            return Err(ClientError::Closed);
        }
        if self.capture {
            self.captured.extend_from_slice(&chunk[..read]);
        }
        self.session.feed(&chunk[..read]);
        Ok(())
    }

    /// Reads the next frame off the pub/sub push loop: a subscribe/
    /// unsubscribe ack, or a `message`/`pmessage` push. Shared by
    /// `subscribe`/`psubscribe` (draining their own acks), `next_push`, and
    /// `unsubscribe_all` — the one place that drives
    /// `ClientSession::poll_push`.
    fn next_push_frame(&mut self) -> Result<RespValue, ClientError> {
        loop {
            match self.session.poll_push()? {
                PushStep::Frame(value) => return Ok(value),
                PushStep::Recv => self.read_into_session()?,
            }
        }
    }

    /// Reconstructs `self` under a different `State` marker — the compile-time
    /// transition a `subscribe`/`unsubscribe_all` call makes. Zero-cost: same
    /// fields, same session, same transport; only the `PhantomData` changes.
    fn into_state<NewState>(self) -> RedisClient<S, NewState> {
        RedisClient {
            stream: self.stream,
            session: self.session,
            capture: self.capture,
            captured: self.captured,
            channels: self.channels,
            patterns: self.patterns,
            _state: PhantomData,
        }
    }
}

impl<S: Read + Write> RedisClient<S, Active> {
    /// Startup handshake over `stream`, returning a ready client.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a server error during `HELLO`/`AUTH`/`SELECT`.
    pub fn connect(stream: S, config: &RedisClientConfig) -> Result<Self, ClientError> {
        Self::connect_inner(stream, config, false)
    }

    /// Like [`Self::connect`] but tees every server byte into `captured`.
    ///
    /// # Errors
    /// See [`Self::connect`].
    pub fn connect_capturing(stream: S, config: &RedisClientConfig) -> Result<Self, ClientError> {
        Self::connect_inner(stream, config, true)
    }

    fn connect_inner(
        stream: S,
        config: &RedisClientConfig,
        capture: bool,
    ) -> Result<Self, ClientError> {
        let session = ClientSession::new(config);
        let mut client = Self {
            stream,
            session,
            capture,
            captured: Vec::new(),
            channels: BTreeSet::new(),
            patterns: BTreeSet::new(),
            _state: PhantomData,
        };
        client.drive_until_ready()?;
        Ok(client)
    }

    /// Runs one command (verb + args) and collects the reply.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a malformed frame. A server `-ERR` reply is
    /// returned as a [`RespValue::Error`], not an `Err`.
    pub fn command(&mut self, argv: &[&[u8]]) -> Result<RespValue, ClientError> {
        self.session.submit(argv)?;
        self.run_command()
    }

    /// Sends `SUBSCRIBE` for every channel, drains each channel's
    /// subscribe-ack off the pub/sub push loop, and returns the client
    /// re-typed to [`Subscribed`] — the client-side transition mirroring the
    /// server entering `ConnMode::Subscriber`. `channels` may repeat a name;
    /// the server acks each argument unconditionally (one frame per loop
    /// iteration, not per distinct channel), so every ack is still drained,
    /// but only the distinct names are recorded for `unsubscribe_all`.
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if `channels` is empty (nothing would be
    /// sent, so nothing would ack — nothing to gate the state transition on).
    /// [`ClientError`] on I/O or a malformed frame.
    pub fn subscribe(mut self, channels: &[&[u8]]) -> Result<RedisClient<S, Subscribed>, ClientError> {
        if channels.is_empty() {
            return Err(ClientError::Protocol("subscribe requires at least one channel"));
        }
        self.enter_subscriber_mode(b"SUBSCRIBE", channels)?;
        for channel in channels {
            self.channels.insert((*channel).to_vec());
        }
        Ok(self.into_state())
    }

    /// Like [`Self::subscribe`] but for glob patterns (`PSUBSCRIBE`).
    ///
    /// # Errors
    /// [`ClientError::Protocol`] if `patterns` is empty. [`ClientError`] on
    /// I/O or a malformed frame.
    pub fn psubscribe(mut self, patterns: &[&[u8]]) -> Result<RedisClient<S, Subscribed>, ClientError> {
        if patterns.is_empty() {
            return Err(ClientError::Protocol("psubscribe requires at least one pattern"));
        }
        self.enter_subscriber_mode(b"PSUBSCRIBE", patterns)?;
        for pattern in patterns {
            self.patterns.insert((*pattern).to_vec());
        }
        Ok(self.into_state())
    }

    /// Sends `QUIT` and drops the connection (best-effort).
    ///
    /// # Errors
    /// [`ClientError::Io`] if the final write fails.
    pub fn close(mut self) -> Result<(), ClientError> {
        let mut bytes = Vec::new();
        encode_command(&[b"QUIT"], &mut bytes);
        self.stream.write_all(&bytes)?;
        self.stream.flush()?;
        Ok(())
    }

    fn enter_subscriber_mode(&mut self, verb: &'static [u8], targets: &[&[u8]]) -> Result<(), ClientError> {
        let mut argv: Vec<&[u8]> = Vec::with_capacity(targets.len() + 1);
        argv.push(verb);
        argv.extend_from_slice(targets);
        self.session.queue_command(&argv)?;
        self.flush_outbound()?;
        for _ in 0..targets.len() {
            self.next_push_frame()?;
        }
        Ok(())
    }

    fn drive_until_ready(&mut self) -> Result<(), ClientError> {
        loop {
            match self.session.advance()? {
                Step::Send => self.flush_outbound()?,
                Step::Recv => self.read_into_session()?,
                Step::Ready => return Ok(()),
                Step::Complete(_) => return Err(ClientError::Protocol("reply before ready")),
            }
        }
    }

    fn run_command(&mut self) -> Result<RespValue, ClientError> {
        loop {
            match self.session.advance()? {
                Step::Send => self.flush_outbound()?,
                Step::Recv => self.read_into_session()?,
                Step::Complete(value) => return Ok(value),
                Step::Ready => return Err(ClientError::Protocol("ready without a reply")),
            }
        }
    }
}

impl<S: Read + Write> RedisClient<S, Subscribed> {
    /// Reads the next pub/sub push frame — a `message`/`pmessage` array, or
    /// an `unsubscribe`/`punsubscribe` ack.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a malformed frame.
    pub fn next_push(&mut self) -> Result<RespValue, ClientError> {
        self.next_push_frame()
    }

    /// Sends `UNSUBSCRIBE`/`PUNSUBSCRIBE` for every subscription this client
    /// opened, drains each ack, and returns the client re-typed to [`Active`]
    /// — the client-side mirror of the server falling back to
    /// `ConnMode::Command` once every subscription is gone. Drains exactly
    /// `self.channels.len()` / `self.patterns.len()` acks — the server sends
    /// one per *distinct* open channel/pattern, so this must match the set,
    /// not a count of every name ever passed to `subscribe`/`psubscribe`.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a malformed frame.
    pub fn unsubscribe_all(mut self) -> Result<RedisClient<S, Active>, ClientError> {
        if !self.channels.is_empty() {
            self.session.queue_command(&[b"UNSUBSCRIBE"])?;
            self.flush_outbound()?;
            for _ in 0..self.channels.len() {
                self.next_push_frame()?;
            }
            self.channels.clear();
        }
        if !self.patterns.is_empty() {
            self.session.queue_command(&[b"PUNSUBSCRIBE"])?;
            self.flush_outbound()?;
            for _ in 0..self.patterns.len() {
                self.next_push_frame()?;
            }
            self.patterns.clear();
        }
        Ok(self.into_state())
    }
}
