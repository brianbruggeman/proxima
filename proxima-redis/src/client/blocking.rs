//! Blocking driver over the sans-IO [`ClientSession`] — a `std::io::Read +
//! Write` transport (e.g. `std::net::TcpStream`). Used by the real-server parity
//! harness (capture + differential); the async Pipe driver wraps the same
//! session over a futures-io transport. The session owns the protocol; this only
//! moves bytes.

use std::io::{Read, Write};

use proxima_protocols::redis::{RespValue, encode_command};

use crate::client::config::RedisClientConfig;
use crate::client::session::{ClientError, ClientSession, Step};

pub struct RedisClient<S> {
    stream: S,
    session: ClientSession,
    capture: bool,
    /// Every server byte read, when capture is enabled — the verbatim
    /// server->client stream for fixture vendoring.
    pub captured: Vec<u8>,
}

impl<S: Read + Write> RedisClient<S> {
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
}
