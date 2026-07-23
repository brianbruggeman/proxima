//! The MQTT-over-Pipe contract: how an MQTT packet maps onto a
//! `proxima_primitives::pipe::Pipe` request, and what rides back. Mirrors
//! `crate::redis::pipe_contract` — this is the RISC payoff (workspace
//! principle 1): proxima-mqtt does not own a bespoke client trait, it
//! speaks the one workspace primitive, `Pipe`.
//!
//! [`MqttRequest`] is the owned, `'static` counterpart of
//! [`super::Packet`] for the request-shaped packets a caller sends
//! (`CONNECT`, `PUBLISH`, `SUBSCRIBE`, `UNSUBSCRIBE`, `PINGREQ`,
//! `DISCONNECT`) — the same borrowed-to-owned lift
//! `crate::redis::RespValue::from_frame` does for `Frame<'a>`, so it can
//! ride a `Carry` across an async boundary. [`MqttReply`] is its
//! response-shaped mirror (`CONNACK`, `PUBACK`, `SUBACK`, `UNSUBACK`,
//! `PINGRESP`) — MQTT has no single universal value type the way RESP has
//! [`crate::redis::RespValue`], so the reply is enumerated per request
//! shape instead of one carry type.

use alloc::vec::Vec;

/// Command verbs a caller sets as `Request.method`.
pub mod verb {
    pub const CONNECT: &str = "CONNECT";
    pub const PUBLISH: &str = "PUBLISH";
    pub const SUBSCRIBE: &str = "SUBSCRIBE";
    pub const UNSUBSCRIBE: &str = "UNSUBSCRIBE";
    pub const PING: &str = "PING";
    pub const DISCONNECT: &str = "DISCONNECT";
}

/// True for verbs that switch the connection into a server-pushed stream:
/// after the `SUBACK` reply the driver keeps reading pushed `PUBLISH`
/// frames rather than returning to request/reply — mirrors
/// `crate::redis::pipe_contract::is_streaming`.
#[must_use]
pub fn is_streaming(command: &str) -> bool {
    command.eq_ignore_ascii_case(verb::SUBSCRIBE)
}

/// The typed payload a caller puts in `Request.payload` — one variant per
/// request-shaped MQTT packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MqttRequest {
    Connect {
        client_id: Vec<u8>,
        clean_session: bool,
        keep_alive: u16,
        username: Option<Vec<u8>>,
        password: Option<Vec<u8>>,
    },
    Publish {
        topic: Vec<u8>,
        payload: Vec<u8>,
        qos: u8,
        retain: bool,
    },
    Subscribe {
        filters: Vec<(Vec<u8>, u8)>,
    },
    Unsubscribe {
        filters: Vec<Vec<u8>>,
    },
    Ping,
    Disconnect,
}

/// The typed payload a reply rides on `Response.payload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MqttReply {
    ConnAck { session_present: bool, return_code: u8 },
    /// `qos == 0` `PUBLISH` completes with no wire acknowledgement.
    Published,
    PubAck { packet_id: u16 },
    SubAck { packet_id: u16, granted: Vec<u8> },
    UnsubAck { packet_id: u16 },
    Pong,
    Disconnected,
    /// One inbound `PUBLISH` the broker pushed to a subscribed client —
    /// the per-chunk item a `SUBSCRIBE` reply's `Response.stream` carries,
    /// mirroring how `crate::redis::RespValue` rides a pub/sub `message`
    /// frame on `proxima_redis`'s streamed reply.
    Publish {
        topic: Vec<u8>,
        payload: Vec<u8>,
        qos: u8,
        retain: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_detection_is_case_insensitive_and_scoped_to_subscribe() {
        assert!(is_streaming("subscribe"));
        assert!(is_streaming("SUBSCRIBE"));
        assert!(!is_streaming("PUBLISH"));
        assert!(!is_streaming("unsubscribe"));
    }
}
