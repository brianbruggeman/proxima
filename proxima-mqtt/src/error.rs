//! Facade-level serve errors. Wire-level detail stays in
//! `proxima_protocols::mqtt` (`ParseError`); this layer adds transport,
//! buffer-policy, and configuration failures. Mirrors
//! `proxima_redis::error::RedisServeError`.

use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MqttServeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("handler pipe: {0}")]
    Pipe(#[from] proxima_core::ProximaError),
    #[error("inbound message exceeds the {limit}-byte buffer limit")]
    MessageTooLarge { limit: usize },
    #[error("protocol error: {reason}")]
    Protocol { reason: String },
    #[error("connection closed mid-message")]
    UnexpectedEof,
    #[error("config: {0}")]
    Config(String),
}
