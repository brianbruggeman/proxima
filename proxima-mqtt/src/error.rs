//! Facade-level serve errors. Wire-level detail stays in
//! `proxima_protocols::mqtt` (`ParseError`); this layer adds the transport
//! and buffer-policy failures the connection driver can end on.
//!
//! MQTT v3.1.1 has no error-report packet, so a framing violation or a
//! half-arrived packet closes the connection rather than failing it — those
//! are `Ok(())` out of [`crate::serve_connection`], not variants here.

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
}
