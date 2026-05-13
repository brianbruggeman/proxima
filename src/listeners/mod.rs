#[cfg(feature = "dpdk")]
pub mod dpdk_packet;
#[cfg(feature = "dpdk")]
pub mod dpdk_stream;
#[cfg(feature = "http1")]
pub use proxima_http::listener as http;
#[cfg(all(target_os = "linux", feature = "io-uring", feature = "http1"))]
pub mod http_uring;
// tokio stdio + UnixListener + tokio::sync::Mutex throughout — a genuine
// tokio capability (MCP stdio/socket transport), no prime equivalent today.
#[cfg(feature = "tokio")]
pub mod mcp;
// `quic_stream` (not `quic`): collapses each accepted QUIC
// connection to a single bidi stream so it fits the `StreamListener`
// trait. The full QUIC multiplexer lives at `crate::quic` and is what
// `crate::h3` rides on. Two distinct concerns, two distinct names.
// `stream_listener` only exists under proxima-quic's own `quinn-compat`
// feature (its native sans-IO facade doesn't build it), so gate on our
// forwarding `quinn-compat` feature rather than the broader `quic`.
#[cfg(feature = "quinn-compat")]
pub use proxima_quic::stream_listener as quic_stream;
// Strictly-one-wire-version listeners. `http` (the combiner) handles
// ALPN multiplex + TLS + UDS + SO_REUSEPORT for the full HTTP/1+2
// story; these three siblings each do exactly one wire and nothing
// else, for uniform composition.
#[cfg(feature = "http1")]
pub use proxima_http::http1::listener as h1;
#[cfg(feature = "http2")]
pub use proxima_http::http2::listener as h2;
#[cfg(feature = "http3-quinn-compat")]
pub use proxima_http::http3::listener as h3;
#[cfg(any(feature = "tcp", feature = "unix"))]
pub use proxima_listen::stream as stream_protocol;
#[cfg(feature = "tcp")]
pub use proxima_listen::stream::default_listener as stream_default;
#[cfg(feature = "udp")]
pub use proxima_net::tokio::tokio_packet;
#[cfg(any(feature = "tcp", feature = "unix"))]
pub use proxima_net::tokio::tokio_stream_listener as tokio_stream;
#[cfg(feature = "redis-listener")]
pub use proxima_redis as redis;
#[cfg(feature = "websocket")]
pub use proxima_http::websocket as websocket;
#[cfg(feature = "xdp")]
pub mod xdp_packet;

#[cfg(feature = "dpdk")]
pub use dpdk_packet::DpdkPacketListener;
#[cfg(feature = "dpdk")]
pub use dpdk_stream::{DpdkStreamConnection, DpdkStreamListener, DpdkStreamUpstream};
#[cfg(feature = "http1")]
pub use h1::H1ListenProtocol;
#[cfg(feature = "http1")]
pub use http::{HttpListenProtocol, HttpListenerSpec, serve_h1_connection};
#[cfg(feature = "tokio")]
pub use mcp::McpListenProtocol;
#[cfg(feature = "http2")]
pub use proxima_http::http2::listener::H2ListenProtocol;
// legacy quinn-backed listener; proxima-http's `http3::listener` module
// only exists under its own `http3-quinn-compat` feature (native h3 is
// tokio-free by default — see the umbrella `http3-quinn-compat` feature).
#[cfg(feature = "http3-quinn-compat")]
pub use proxima_http::http3::listener::H3ListenProtocol;
#[cfg(feature = "http3")]
pub use proxima_http::http3::native::H3NativeListenProtocol;
#[cfg(feature = "websocket")]
pub use proxima_http::websocket::{WebSocketConnection, WebSocketListener};
#[cfg(feature = "quinn-compat")]
pub use quic_stream::{QuicListener, QuicStreamConnection};
#[cfg(feature = "tcp")]
pub use stream_default::StreamListenProtocol;
#[cfg(any(feature = "tcp", feature = "unix"))]
pub use stream_protocol::{StreamListenerProtocol, reader_to_byte_stream};
// `ConnTransform`/`FramedListenProtocol` bake `TokioTcpConnection` into their
// public signature — see proxima-listen/src/stream/mod.rs's doc comment.
#[cfg(all(any(feature = "tcp", feature = "unix"), feature = "tokio"))]
pub use stream_protocol::{ConnTransform, FramedListenProtocol};
#[cfg(feature = "udp")]
pub use stream_protocol::DatagramListenProtocol;
#[cfg(feature = "udp")]
pub use tokio_packet::TokioUdpListener;
#[cfg(feature = "tcp")]
pub use tokio_stream::{TokioTcpConnection, TokioTcpListener};
#[cfg(all(feature = "unix", unix))]
pub use tokio_stream::{TokioUnixConnection, TokioUnixListener};
#[cfg(feature = "xdp")]
pub use xdp_packet::XdpUdpListener;
