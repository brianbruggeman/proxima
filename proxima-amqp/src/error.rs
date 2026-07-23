//! Facade-level serve errors. Wire-level detail stays in
//! [`crate::wire::WireError`] / [`crate::method::MethodError`] /
//! [`proxima_protocols::amqp::ParseError`]; this layer adds transport,
//! buffer-policy, and configuration failures. Mirrors
//! `proxima_redis::error::RedisServeError`.

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AmqpServeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("handler pipe: {0}")]
    Pipe(#[from] proxima_core::ProximaError),
    #[error("inbound frame exceeds the {limit}-byte frame-max limit")]
    FrameTooLarge { limit: usize },
    #[error("reassembled message body exceeds the {limit}-byte limit")]
    MessageTooLarge { limit: usize },
    #[error("channel {channel} exceeds the {limit}-channel limit")]
    TooManyChannels { channel: u16, limit: u16 },
    #[error("protocol error: {reason}")]
    Protocol { reason: String },
    #[error("connection closed mid-frame")]
    UnexpectedEof,
    #[error("config: {0}")]
    Config(String),
}
