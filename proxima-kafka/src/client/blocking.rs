//! Blocking driver over the sans-IO [`ClientSession`] — a `std::io::Read +
//! Write` transport (e.g. `std::net::TcpStream`). Mirrors
//! `proxima_redis::client::blocking::RedisClient`; the async Pipe driver
//! ([`crate::client::pipe::KafkaClientUpstream`]) wraps the same session
//! over a futures-io transport. The session owns the protocol; this only
//! moves bytes.

use std::io::{Read, Write};

use crate::client::config::KafkaClientConfig;
use crate::client::session::{ClientError, ClientSession, Step};
use crate::wire::{FetchRequest, FetchResponse, ProduceRequest, ProduceResponse, ResponseBody};

pub struct KafkaClient<S> {
    stream: S,
    session: ClientSession,
}

impl<S: Read + Write> KafkaClient<S> {
    /// `ApiVersions` handshake over `stream`, returning a ready client.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a non-`NONE` `ApiVersions` error code.
    pub fn connect(stream: S, config: &KafkaClientConfig) -> Result<Self, ClientError> {
        let session = ClientSession::new(config);
        let mut client = Self { stream, session };
        client.drive_until_ready()?;
        Ok(client)
    }

    /// Produce one batch, returning the broker's reply.
    ///
    /// # Errors
    /// [`ClientError`] on I/O, a malformed reply, or a non-Produce reply
    /// shape.
    pub fn produce(&mut self, request: ProduceRequest) -> Result<ProduceResponse, ClientError> {
        match self.exchange(crate::wire::RequestBody::Produce(request))? {
            ResponseBody::Produce(response) => Ok(response),
            other => Err(ClientError::Protocol(format!(
                "expected a Produce reply, got {other:?}"
            ))),
        }
    }

    /// Fetch from an offset, returning the broker's reply.
    ///
    /// # Errors
    /// [`ClientError`] on I/O, a malformed reply, or a non-Fetch reply
    /// shape.
    pub fn fetch(&mut self, request: FetchRequest) -> Result<FetchResponse, ClientError> {
        match self.exchange(crate::wire::RequestBody::Fetch(request))? {
            ResponseBody::Fetch(response) => Ok(response),
            other => Err(ClientError::Protocol(format!(
                "expected a Fetch reply, got {other:?}"
            ))),
        }
    }

    fn exchange(&mut self, request: crate::wire::RequestBody) -> Result<ResponseBody, ClientError> {
        self.session.submit(request)?;
        self.run_request()
    }

    fn drive_until_ready(&mut self) -> Result<(), ClientError> {
        loop {
            match self.session.advance()? {
                Step::Send => self.flush_outbound()?,
                Step::Recv => self.read_into_session()?,
                Step::Ready => return Ok(()),
                Step::Complete(_) => {
                    return Err(ClientError::Protocol("reply before ready".into()));
                }
            }
        }
    }

    fn run_request(&mut self) -> Result<ResponseBody, ClientError> {
        loop {
            match self.session.advance()? {
                Step::Send => self.flush_outbound()?,
                Step::Recv => self.read_into_session()?,
                Step::Complete(response) => return Ok(response),
                Step::Ready => return Err(ClientError::Protocol("ready without a reply".into())),
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
        self.session.feed(&chunk[..read]);
        Ok(())
    }
}
