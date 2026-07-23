//! Blocking driver over the sans-IO [`ClientSession`] — a `std::io::Read +
//! Write` transport (e.g. `std::net::TcpStream`). Mirrors
//! `proxima_redis::client::blocking::RedisClient`: the session owns the
//! protocol, this only moves bytes.

use std::io::{Read, Write};

use crate::client::config::AmqpClientConfig;
use crate::client::session::{ClientError, ClientSession, Step};
use crate::frame::encode_method_frame;
use crate::method::Method;

/// One reassembled `basic.deliver`, returned by [`AmqpClient::next_delivery`].
#[derive(Debug, Clone)]
pub struct ClientDelivery {
    pub consumer_tag: Vec<u8>,
    pub delivery_tag: u64,
    pub redelivered: bool,
    pub exchange: Vec<u8>,
    pub routing_key: Vec<u8>,
    pub properties: Vec<u8>,
    pub body: Vec<u8>,
}

pub struct AmqpClient<S> {
    stream: S,
    session: ClientSession,
}

impl<S: Read + Write> AmqpClient<S> {
    /// Startup handshake over `stream`, returning a ready client.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a broker-reported close during the
    /// handshake.
    pub fn connect(stream: S, config: &AmqpClientConfig) -> Result<Self, ClientError> {
        let session = ClientSession::new(config);
        let mut client = Self { stream, session };
        client.drive_until_ready()?;
        Ok(client)
    }

    /// `basic.publish`. Fire-and-forget — see the crate-level gap notes on
    /// publisher confirms.
    ///
    /// # Errors
    /// [`ClientError`] on I/O.
    pub fn publish(
        &mut self,
        exchange: &[u8],
        routing_key: &[u8],
        body: &[u8],
    ) -> Result<(), ClientError> {
        self.session
            .queue_publish(exchange, routing_key, false, false, b"", body)?;
        self.flush_outbound()
    }

    /// `basic.consume`, returning the (possibly server-assigned) consumer
    /// tag once `basic.consume-ok` arrives. Every subsequent `basic.deliver`
    /// is read with [`Self::next_delivery`].
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a malformed reply.
    pub fn consume(&mut self, queue: &[u8]) -> Result<Vec<u8>, ClientError> {
        self.session.queue_consume(queue, b"", false)?;
        loop {
            match self.session.advance()? {
                Step::Send => self.flush_outbound()?,
                Step::Recv => self.read_into_session()?,
                Step::ConsumeOk { consumer_tag } => return Ok(consumer_tag),
                Step::Ready => {}
                Step::Delivery { .. } => {
                    return Err(ClientError::Protocol(
                        "delivery arrived before consume-ok".into(),
                    ));
                }
            }
        }
    }

    /// Blocks for the next `basic.deliver` on this connection.
    ///
    /// # Errors
    /// [`ClientError`] on I/O or a malformed frame.
    pub fn next_delivery(&mut self) -> Result<ClientDelivery, ClientError> {
        loop {
            match self.session.advance()? {
                Step::Recv => self.read_into_session()?,
                Step::Send => self.flush_outbound()?,
                Step::Delivery {
                    consumer_tag,
                    delivery_tag,
                    redelivered,
                    exchange,
                    routing_key,
                    properties,
                    body,
                } => {
                    return Ok(ClientDelivery {
                        consumer_tag,
                        delivery_tag,
                        redelivered,
                        exchange,
                        routing_key,
                        properties,
                        body,
                    });
                }
                Step::Ready | Step::ConsumeOk { .. } => {}
            }
        }
    }

    /// Sends `connection.close` and drops the connection (best-effort — does
    /// not wait for `connection.close-ok`).
    ///
    /// # Errors
    /// [`ClientError::Io`] if the final write fails.
    pub fn close(mut self) -> Result<(), ClientError> {
        let mut bytes = Vec::new();
        encode_method_frame(
            &mut bytes,
            0,
            &Method::ConnectionClose {
                reply_code: 200,
                reply_text: b"bye".to_vec(),
                class_id: 0,
                method_id: 0,
            },
        );
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
                Step::ConsumeOk { .. } | Step::Delivery { .. } => {
                    return Err(ClientError::Protocol(
                        "unexpected event before ready".into(),
                    ));
                }
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
