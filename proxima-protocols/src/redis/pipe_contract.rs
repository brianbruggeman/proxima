//! The RESP-over-Pipe contract: how a Redis/Valkey command maps onto the
//! business-handler pipe's request shape, and what rides back.
//!
//! This is the RISC payoff (workspace principle 1). proxima-redis does not own a
//! bespoke client trait — it speaks the one workspace primitive, `Pipe`, the
//! way pgwire and proxima-telemetry do. [`RedisRequest`] is a de-enveloped,
//! FSM-aware carry: a pipe is `P -> Q` (payload-no-cell, no `Request`/
//! `Response` wrapper) and the variants mirror
//! [`super::connection::ConnMode`]'s transitions — the five commands that
//! actually change a connection's mode (or are gated by it) get a variant;
//! everything else (`GET`, `SET`, `PING`, `PUBLISH`, …) is `Command{verb,
//! args}`, dispatched to the business handler exactly as before.
//!
//! The reply is [`RespValue`](super::RespValue) directly — protocol-out is
//! NOT pinned to protocol-in: a `GET` answers with whatever shape the server
//! returns (bulk string, null, error, …). Pub/sub and MONITOR leave the
//! request/reply rhythm entirely: the driver answers those without ever
//! calling the business handler.

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

/// The business-handler pipe's request payload — a de-enveloped
/// (payload-no-cell) carry, FSM-aware over
/// [`super::connection::ConnMode`]'s two states. Only the commands that
/// actually move or are gated by that FSM get their own variant; every other
/// command (`GET`, `SET`, `PING`, `PUBLISH`, …) is `Command{verb, args}` —
/// the driver's own PING/QUIT/PUBLISH interception and `ConnMode::admits`
/// gate still apply to that variant exactly as before. Arguments are raw
/// bytes — binary safe, so a value can be any blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedisRequest {
    /// `SUBSCRIBE`/`SSUBSCRIBE` with at least one channel — enters
    /// `ConnMode::Subscriber`.
    Subscribe { channels: Vec<Vec<u8>> },
    /// `UNSUBSCRIBE`/`SUNSUBSCRIBE` — empty `channels` means "unsubscribe
    /// every exact-channel subscription".
    Unsubscribe { channels: Vec<Vec<u8>> },
    /// `PSUBSCRIBE` with at least one pattern — enters
    /// `ConnMode::Subscriber`.
    Psubscribe { patterns: Vec<Vec<u8>> },
    /// `PUNSUBSCRIBE` — empty `patterns` means "unsubscribe every pattern
    /// subscription".
    Punsubscribe { patterns: Vec<Vec<u8>> },
    /// Every other command, including the arity-empty (P)SUBSCRIBE forms
    /// (which surface as an unknown-command/arity error, same as before)
    /// and the driver-intercepted PING/QUIT/PUBLISH.
    Command { verb: Vec<u8>, args: Vec<Vec<u8>> },
}

impl RedisRequest {
    /// Splits a raw command argument list (verb first, then its arguments)
    /// into the FSM-aware carry. The verb is upper-cased before matching —
    /// Redis command verbs are case-insensitive on the wire.
    ///
    /// NOTE: `SSUBSCRIBE`/`SUNSUBSCRIBE` (sharded pub/sub) collapse into the
    /// regular `Subscribe`/`Unsubscribe` variants — this mirrors current
    /// behavior (the broker gives shard channels no distinct namespace
    /// today) rather than fixing the pre-existing sharded-pubsub compliance
    /// gap, which is a separate follow-on (PC3-redis-COMPLIANCE).
    #[must_use]
    pub fn from_args(args: Vec<Vec<u8>>) -> Self {
        let Some((verb, rest)) = args.split_first() else {
            return Self::Command {
                verb: Vec::new(),
                args: Vec::new(),
            };
        };
        let verb = verb.to_ascii_uppercase();
        let rest = rest.to_vec();
        match verb.as_slice() {
            b"SUBSCRIBE" | b"SSUBSCRIBE" if !rest.is_empty() => Self::Subscribe { channels: rest },
            b"UNSUBSCRIBE" | b"SUNSUBSCRIBE" => Self::Unsubscribe { channels: rest },
            b"PSUBSCRIBE" if !rest.is_empty() => Self::Psubscribe { patterns: rest },
            b"PUNSUBSCRIBE" => Self::Punsubscribe { patterns: rest },
            _ => Self::Command { verb, args: rest },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn streaming_detection_is_case_insensitive() {
        assert!(is_streaming("subscribe"));
        assert!(is_streaming("PSUBSCRIBE"));
        assert!(is_streaming("Monitor"));
        assert!(!is_streaming("GET"));
        assert!(!is_streaming("publish"));
    }

    #[test]
    fn from_args_splits_subscribe_into_channels() {
        let request = RedisRequest::from_args(vec![b"SUBSCRIBE".to_vec(), b"news".to_vec()]);
        assert_eq!(
            request,
            RedisRequest::Subscribe {
                channels: vec![b"news".to_vec()]
            }
        );
    }

    #[test]
    fn from_args_collapses_ssubscribe_into_subscribe() {
        let request = RedisRequest::from_args(vec![b"SSUBSCRIBE".to_vec(), b"shard".to_vec()]);
        assert_eq!(
            request,
            RedisRequest::Subscribe {
                channels: vec![b"shard".to_vec()]
            }
        );
    }

    #[test]
    fn from_args_empty_subscribe_falls_to_command() {
        let request = RedisRequest::from_args(vec![b"SUBSCRIBE".to_vec()]);
        assert_eq!(
            request,
            RedisRequest::Command {
                verb: b"SUBSCRIBE".to_vec(),
                args: Vec::new()
            }
        );
    }

    #[test]
    fn from_args_unsubscribe_with_no_targets_means_all() {
        let request = RedisRequest::from_args(vec![b"UNSUBSCRIBE".to_vec()]);
        assert_eq!(
            request,
            RedisRequest::Unsubscribe {
                channels: Vec::new()
            }
        );
    }

    #[test]
    fn from_args_psubscribe_splits_patterns() {
        let request = RedisRequest::from_args(vec![b"PSUBSCRIBE".to_vec(), b"news.*".to_vec()]);
        assert_eq!(
            request,
            RedisRequest::Psubscribe {
                patterns: vec![b"news.*".to_vec()]
            }
        );
    }

    #[test]
    fn from_args_punsubscribe_with_no_targets_means_all() {
        let request = RedisRequest::from_args(vec![b"PUNSUBSCRIBE".to_vec()]);
        assert_eq!(
            request,
            RedisRequest::Punsubscribe {
                patterns: Vec::new()
            }
        );
    }

    #[test]
    fn from_args_uppercases_the_verb_for_command() {
        let request = RedisRequest::from_args(vec![b"get".to_vec(), b"key".to_vec()]);
        assert_eq!(
            request,
            RedisRequest::Command {
                verb: b"GET".to_vec(),
                args: vec![b"key".to_vec()]
            }
        );
    }
}
