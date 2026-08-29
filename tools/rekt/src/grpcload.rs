//! gRPC load — grpc IS http/2, so this is the multiplexed h2 drive with a grpc
//! request shape: `POST` + `content-type: application/grpc`, a length-prefixed
//! message body, and a `grpc-status` trailer closing the stream. It reuses
//! h2load's request-agnostic connection loop ([`crate::h2load::drive_h2_request`])
//! unchanged — the stream ends on the trailers (a HEADERS with END_STREAM), which
//! the h2 completion path already counts, so a completion is one full grpc unary
//! round-trip regardless of the `grpc-status` value.
//!
//! GET-less by design: an empty unary message (`grpc_frame(&[])`) keeps the
//! send-window irrelevant and measures the h2 + grpc-framing round-trip itself.

use std::time::Duration;

use bytes::{BufMut, Bytes, BytesMut};

use crate::engine::{LiveCounters, Throughput};
use crate::error::Error;
use crate::h2load::{H2Request, drive_h2_request, parse_target};

// a grpc length-prefixed message: 1 compression-flag byte (0 = uncompressed) +
// a 4-byte big-endian length + the message bytes.
fn grpc_frame(message: &[u8]) -> Bytes {
    let mut buffer = BytesMut::with_capacity(5 + message.len());
    buffer.put_u8(0);
    buffer.put_u32(message.len() as u32);
    buffer.put_slice(message);
    buffer.freeze()
}

// the grpc request headers: pseudo-headers first (HPACK requires it), then the
// grpc content-type and the `te: trailers` the status trailer rides on.
fn grpc_headers(authority: &str, path: &str) -> Vec<(Bytes, Bytes)> {
    vec![
        (Bytes::from_static(b":method"), Bytes::from_static(b"POST")),
        (Bytes::from_static(b":scheme"), Bytes::from_static(b"http")),
        (Bytes::from_static(b":authority"), Bytes::from(authority.to_string())),
        (Bytes::from_static(b":path"), Bytes::from(path.to_string())),
        (Bytes::from_static(b"content-type"), Bytes::from_static(b"application/grpc")),
        (Bytes::from_static(b"te"), Bytes::from_static(b"trailers")),
    ]
}

/// Closed-loop grpc drive: `cores` prime cores, each opening
/// `connections_per_core` h2 connections that keep `streams_per_conn` unary calls
/// to `path` (e.g. `/helloworld.Greeter/SayHello`) in flight until the deadline.
pub fn drive_grpc(url: &str, path: &str, connections_per_core: usize, streams_per_conn: usize, cores: usize, duration: Duration) -> Result<Throughput, Error> {
    drive_grpc_metered(url, path, connections_per_core, streams_per_conn, cores, duration, None)
}

/// Live sibling of [`drive_grpc`]: folds each connection's completions/errors
/// into the registered telemetry counters as they happen, so a dashboard can
/// sample the run in flight. The aggregate [`Throughput`] is unchanged.
pub fn drive_grpc_live(url: &str, path: &str, connections_per_core: usize, streams_per_conn: usize, cores: usize, duration: Duration, counters: LiveCounters) -> Result<Throughput, Error> {
    drive_grpc_metered(url, path, connections_per_core, streams_per_conn, cores, duration, Some(counters))
}

fn drive_grpc_metered(
    url: &str,
    path: &str,
    connections_per_core: usize,
    streams_per_conn: usize,
    cores: usize,
    duration: Duration,
    counters: Option<LiveCounters>,
) -> Result<Throughput, Error> {
    let (host, port, authority) = parse_target(url)?;
    let request = H2Request {
        headers: grpc_headers(&authority, path),
        body: Some(grpc_frame(&[])),
    };
    drive_h2_request(host, port, request, streams_per_conn, connections_per_core, cores, duration, counters)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn empty_message_frames_to_a_bare_5_byte_header() {
        let frame = grpc_frame(&[]);
        assert_eq!(&frame[..], &[0u8, 0, 0, 0, 0]);
    }

    #[test]
    fn frame_carries_big_endian_length_then_body() {
        let frame = grpc_frame(&[0xAA, 0xBB]);
        assert_eq!(&frame[..], &[0u8, 0, 0, 0, 2, 0xAA, 0xBB]);
    }

    #[test]
    fn headers_lead_with_pseudo_headers_then_grpc_content_type() {
        let headers = grpc_headers("127.0.0.1:8100", "/svc/Method");
        assert_eq!(headers[0].0, Bytes::from_static(b":method"));
        assert_eq!(headers[0].1, Bytes::from_static(b"POST"));
        assert!(
            headers
                .iter()
                .any(|(name, value)| *name == Bytes::from_static(b":path") && *value == Bytes::from_static(b"/svc/Method"))
        );
        assert!(
            headers
                .iter()
                .any(|(name, value)| *name == Bytes::from_static(b"content-type") && *value == Bytes::from_static(b"application/grpc"))
        );
    }
}
