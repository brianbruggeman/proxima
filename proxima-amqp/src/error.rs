//! Facade-level serve errors — the four ways
//! [`crate::connection::serve_connection`] can end other than cleanly.
//! Wire-level detail stays in [`crate::wire::WireError`] /
//! [`crate::method::MethodError`] / [`proxima_protocols::amqp::ParseError`];
//! this layer adds transport and DoS-cap failures. Mirrors
//! `proxima_redis::error::RedisServeError` in role, not variant-for-variant:
//! an AMQP protocol violation is not an error here at all, because the
//! driver renders it as `connection.close`/`channel.close` on the wire and
//! ends the connection cleanly (see `crate::connection`'s `Outcome`).

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AmqpServeError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The outer wait — the socket read, the consumer push channel, and
    /// shutdown merged through `proxima_listen::wait_for_wire_event` —
    /// failed. NOT the business handler: a handler error never reaches
    /// here, it is logged and the publish is dropped
    /// (`crate::connection::dispatch_publish`).
    #[error("connection pipe: {0}")]
    Pipe(#[from] proxima_core::ProximaError),
    #[error("inbound frame exceeds the {limit}-byte frame-max limit")]
    FrameTooLarge { limit: usize },
    #[error("reassembled message body exceeds the {limit}-byte limit")]
    MessageTooLarge { limit: usize },
}
