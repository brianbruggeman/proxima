//! Facade-level serve errors. Wire-level detail stays in the codec error
//! types; this layer adds transport, buffer-policy, and configuration
//! failures.

use proxima_core::ProximaError;
use proxima_protocols::pgwire_codec::{EncodeError, ParseError, SessionError};
use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ServeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("sql pipe: {0}")]
    Pipe(#[from] ProximaError),
    #[error("wire parse: {0}")]
    Parse(#[from] ParseError),
    #[error("wire encode: {0}")]
    Encode(#[from] EncodeError),
    #[error("session: {0}")]
    Session(#[from] SessionError),
    #[error("inbound message exceeds the {limit}-byte buffer limit")]
    MessageTooLarge { limit: usize },
    #[error("connection closed mid-message")]
    UnexpectedEof,
    #[error("startup did not carry the required user parameter")]
    MissingUser,
    #[error("invalid utf-8 in {field}")]
    InvalidUtf8 { field: &'static str },
    #[error("config: {0}")]
    Config(String),
    #[error("background pool: {0}")]
    BackgroundPool(String),
}
