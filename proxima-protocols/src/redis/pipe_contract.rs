//! The RESP-over-Pipe contract: how a Redis/Valkey command maps onto a
//! `proxima_primitives::pipe::Pipe` request, and what rides back.
//!
//! This is the RISC payoff (workspace principle 1). proxima-redis does not own a
//! bespoke client trait — it speaks the one workspace primitive, `Pipe`, the
//! way pgwire and proxima-telemetry do. The request shape:
//!
//! - `Request.method` is the command verb (`GET`, `SET`, `SUBSCRIBE`, …).
//! - the argument list is `Request.body` (the single-arg convenience, e.g. the
//!   key for `GET`) OR, when a command needs more than one argument, the typed
//!   [`RedisRequest`] carry (`SET key value`, `MSET k1 v1 k2 v2`). When the
//!   carry is present it supplies the full argument list and `body` is ignored.
//!
//! The reply is typed protocol-out: it rides `Response.carry` as a
//! [`RespValue`](super::RespValue), downcast with `response.reply::<RespValue>()`.
//! Protocol-out is NOT pinned to protocol-in — a `GET` answers with whatever
//! shape the server returns (bulk string, null, error, …). Pub/sub and MONITOR
//! leave the request/reply rhythm: the reply rides `Response.stream` as RESP
//! wire bytes, one pushed frame per chunk.

use alloc::vec::Vec;

/// Command verbs a caller sets as `Request.method`. These are the literal
/// command names — Redis is case-insensitive on the verb, but the canonical
/// upper-case spelling is what the wire carries.
pub mod verb {
    pub const GET: &str = "GET";
    pub const SET: &str = "SET";
    pub const DEL: &str = "DEL";
    pub const EXISTS: &str = "EXISTS";
    pub const INCR: &str = "INCR";
    pub const EXPIRE: &str = "EXPIRE";
    pub const TTL: &str = "TTL";
    pub const PING: &str = "PING";
    pub const ECHO: &str = "ECHO";
    pub const HELLO: &str = "HELLO";
    pub const COMMAND: &str = "COMMAND";
    pub const MGET: &str = "MGET";
    pub const MSET: &str = "MSET";
    pub const HSET: &str = "HSET";
    pub const HGET: &str = "HGET";
    pub const HGETALL: &str = "HGETALL";
    pub const PUBLISH: &str = "PUBLISH";
    pub const SUBSCRIBE: &str = "SUBSCRIBE";
    pub const PSUBSCRIBE: &str = "PSUBSCRIBE";
    pub const SSUBSCRIBE: &str = "SSUBSCRIBE";
    pub const UNSUBSCRIBE: &str = "UNSUBSCRIBE";
    pub const MONITOR: &str = "MONITOR";
}

/// True for commands that switch the connection into a server-pushed stream
/// (pub/sub subscribe families and `MONITOR`): after the first reply the driver
/// keeps reading pushed frames rather than returning to request/reply.
#[must_use]
pub fn is_streaming(command: &str) -> bool {
    command.eq_ignore_ascii_case(verb::SUBSCRIBE)
        || command.eq_ignore_ascii_case(verb::PSUBSCRIBE)
        || command.eq_ignore_ascii_case(verb::SSUBSCRIBE)
        || command.eq_ignore_ascii_case(verb::MONITOR)
}

/// The typed payload a caller puts in `Request.carry` to supply a multi-argument
/// command's full argument list (everything after the verb). When present it
/// overrides `Request.body`. Arguments are raw bytes — binary safe, so a value
/// can be any blob.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RedisRequest {
    pub args: Vec<Vec<u8>>,
}

impl RedisRequest {
    /// A command's arguments from any iterator of byte-ish values.
    #[must_use]
    pub fn new(args: impl IntoIterator<Item = Vec<u8>>) -> Self {
        Self {
            args: args.into_iter().collect(),
        }
    }

    /// Convenience for the common all-text case (`SET key value`).
    #[must_use]
    pub fn text(args: &[&str]) -> Self {
        Self {
            args: args.iter().map(|arg| arg.as_bytes().to_vec()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_detection_is_case_insensitive() {
        assert!(is_streaming("subscribe"));
        assert!(is_streaming("PSUBSCRIBE"));
        assert!(is_streaming("Monitor"));
        assert!(!is_streaming("GET"));
        assert!(!is_streaming("publish"));
    }

    #[test]
    fn text_args_round_trip() {
        let request = RedisRequest::text(&["key", "value"]);
        assert_eq!(request.args, vec![b"key".to_vec(), b"value".to_vec()]);
    }
}
