//! Multiplexed HTTP/2 load — the "later layer" proxima-h2's unary client doc
//! names. proxima ships the sans-IO h2 `Connection` (multiplexing, flow control,
//! HPACK) but only a unary connect-per-request client wrapper; that's useless as
//! a load source. Here we drive `Connection` directly: one persistent socket,
//! `streams` concurrent streams kept in flight (refilled the instant one ends),
//! reusing the h1 drive harness (per-core prime factories, `Throughput`).
//!
//! GET `/` only, no request body — so the client send-window never matters and
//! the 2-byte `ok` response sits far inside the 64 KiB stream window; we never
//! touch `WindowGranted`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::io::{AsyncReadExt, AsyncWriteExt};
use proxima::h2::connection::{Connection, ConnectionEvent};
use proxima::h2::frame::StandardSettings;
use proxima::{PrimeTcpUpstream, StreamUpstreamExt};

use crate::engine::{Throughput, drive_replicated};
use crate::error::Error;

const READ_CHUNK: usize = 16_384;
/// h2 connections cannot carry more than the peer's SETTINGS_MAX_CONCURRENT_STREAMS
/// (proxima advertises 100); opening past it just earns RST_STREAMs.
const MAX_STREAMS_PER_CONN: usize = 100;

/// Client SETTINGS — mirror of proxima's native server/client defaults.
fn client_settings() -> StandardSettings {
    StandardSettings {
        header_table_size: Some(4096),
        enable_push: Some(false),
        max_concurrent_streams: Some(100),
        initial_window_size: Some(65_535),
        max_frame_size: Some(16_384),
        max_header_list_size: None,
        extensions: Default::default(),
    }
}

/// The `GET /` request as h2 pseudo-headers (order matters: pseudo-headers first).
fn get_headers(authority: &str) -> Vec<(Bytes, Bytes)> {
    vec![
        (Bytes::from_static(b":method"), Bytes::from_static(b"GET")),
        (Bytes::from_static(b":scheme"), Bytes::from_static(b"http")),
        (Bytes::from_static(b":authority"), Bytes::from(authority.to_string())),
        (Bytes::from_static(b":path"), Bytes::from_static(b"/")),
    ]
}

/// Parse `http://host[:port]/` into `(host, port, authority)`.
fn parse_target(url: &str) -> Result<(String, u16, String), Error> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| Error::Engine("h2 target must be http://host[:port]/".into()))?;
    let authority = rest.split('/').next().unwrap_or(rest).to_string();
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => (
            host.to_string(),
            port.parse::<u16>()
                .map_err(|err| Error::Engine(format!("h2 target port: {err}")))?,
        ),
        None => (authority.clone(), 80),
    };
    Ok((host, port, authority))
}

/// Closed-loop multiplexed h2 drive: `cores` prime cores, each opening
/// `connections_per_core` persistent h2 connections, each keeping
/// `streams_per_conn` streams in flight, firing `GET /` until the deadline.
/// Composes [`crate::engine::drive_replicated`] — see its doc-comment for why
/// this fans via `FuturesUnordered` and not
/// [`proxima_primitives::pipe::FanOut`]/[`proxima_primitives::pipe::ScatterGather`].
pub fn drive_h2(url: &str, connections_per_core: usize, streams_per_conn: usize, cores: usize, duration: Duration) -> Result<Throughput, Error> {
    let (host, port, authority) = parse_target(url)?;
    let streams_per_conn = streams_per_conn.clamp(1, MAX_STREAMS_PER_CONN);
    let authority = Arc::new(authority);
    drive_replicated(cores, connections_per_core, duration, move |deadline| {
        let host = host.clone();
        let authority = Arc::clone(&authority);
        async move { h2_connection(&host, port, &authority, streams_per_conn, deadline).await }
    })
}

// one persistent h2 connection multiplexing `streams` concurrent GETs: open the
// socket once, seed `streams` streams, then refill a stream the instant it ends
// (END_STREAM) so the connection stays saturated until the deadline.
async fn h2_connection(host: &str, port: u16, authority: &str, streams: usize, deadline: Instant) -> (u64, u64) {
    let upstream = PrimeTcpUpstream::with_host(host.to_string(), port);
    let mut socket = match upstream.connect().await {
        Ok(socket) => socket,
        Err(_) => return (0, 1),
    };
    let mut connection = Connection::new_client(client_settings());
    let headers = get_headers(authority);

    for _ in 0..streams {
        let stream_id = connection.next_local_stream_id();
        if connection
            .send_request_head(stream_id, headers.clone(), true)
            .is_err()
        {
            break;
        }
    }

    let (mut completed, mut errors) = (0u64, 0u64);
    let mut read_buf = vec![0u8; READ_CHUNK];
    let mut running = true;
    while running {
        let outbound = connection.take_output();
        if !outbound.is_empty() && socket.write_all(&outbound).await.is_err() {
            errors += 1;
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        let read = match socket.read(&mut read_buf).await {
            Ok(0) => break,
            Ok(read) => read,
            Err(_) => {
                errors += 1;
                break;
            }
        };
        if connection.feed(&read_buf[..read]).is_err() {
            errors += 1;
            break;
        }
        while let Some(event) = connection.next_event() {
            match event {
                ConnectionEvent::ResponseHead { end_stream: true, .. } | ConnectionEvent::BodyData { end_stream: true, .. } => {
                    completed += 1;
                    if Instant::now() < deadline {
                        let stream_id = connection.next_local_stream_id();
                        let _ = connection.send_request_head(stream_id, headers.clone(), true);
                    }
                }
                ConnectionEvent::StreamReset { .. } => {
                    errors += 1;
                    if Instant::now() < deadline {
                        let stream_id = connection.next_local_stream_id();
                        let _ = connection.send_request_head(stream_id, headers.clone(), true);
                    }
                }
                ConnectionEvent::PeerGoaway { .. } => running = false,
                _ => {}
            }
        }
    }
    (completed, errors)
}
