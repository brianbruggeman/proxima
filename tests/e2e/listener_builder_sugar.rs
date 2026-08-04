//! End-to-end coverage for the builder-sugar surface: the three
//! type-specific extension-trait families (`ListenerTransportExt`/
//! `ListenerProtocolExt`, `ClientTransportExt`/`ClientSecurityExt`/
//! `ClientProtocolExt`), the `Transport` enum dissolution (`.http().quic()`
//! now genuinely dials/serves h3-native), `.dns()` now reaching BOTH
//! transports on one port via the `.any()` fan-in, and the runtime
//! validation for invalid axis compositions.
//!
//! Kept deliberately light per composition — proving the axis RESOLVES to
//! the right listen protocol / factory, not re-testing each protocol
//! crate's own wire conformance (that is `proxima-kafka`/`proxima-dns`/…'s
//! own test suite's job).

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(all(
    feature = "http1",
    any(
        feature = "runtime-tokio",
        all(
            feature = "serve-prime",
            feature = "runtime-prime-reactor",
            any(target_os = "linux", target_os = "macos")
        )
    )
))]

use std::future::Future;
use std::net::{SocketAddr, TcpListener as StdTcpListener, TcpStream as StdTcpStream};
use std::time::Duration;

use bytes::Bytes;
use serde_json::Value;

use proxima::SendPipe;
use proxima::error::ProximaError;
use proxima::pipe::into_handle;
use proxima::prelude::*;
use proxima::request::{Request, Response};

fn free_loopback_addr() -> SocketAddr {
    let probe = StdTcpListener::bind("127.0.0.1:0").expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

fn wait_until_listening(addr: SocketAddr) {
    for _ in 0..200 {
        if StdTcpStream::connect(addr).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("listener at {addr} never came up");
}

struct FixedOk;

impl SendPipe for FixedOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move { Ok(Response::ok("builder-sugar-ok")) }
    }
}

/// Bare `.http()` — the default h1+h2 ALPN combiner — compiles and serves.
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn bare_http_listener_serves() {
    let bind = free_loopback_addr();
    let server = Listener::http(bind)
        .handle(into_handle(FixedOk))
        .serve()
        .await
        .expect("bare .http() serves");
    wait_until_listening(bind);
    server.stop();
}

/// `.http().tcp().tls(cfg)` composes: the TLS decorator wraps the ALPN
/// combiner `.tcp()` also resolves to.
#[cfg(feature = "tls")]
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_tcp_tls_composes() {
    use proxima::tls::TlsConfig;

    let bind = free_loopback_addr();
    let server = Listener::builder()
        .bind(bind)
        .tcp()
        .tls(TlsConfig::self_signed())
        .handle(into_handle(FixedOk))
        .serve()
        .await
        .expect(".http().tcp().tls(cfg) serves");
    wait_until_listening(bind);
    server.stop();
}

/// `.http().quic()` genuinely resolves to (and binds) the native h3
/// `DatagramProtocol` listener — a real UDP bind, not the ALPN combiner. The
/// full h3 wire round-trip is `tests/e2e/listener_h3_native.rs`'s job; this
/// proves the AXIS resolves at all (it used to hard-error under the name
/// `.h3()` before the fold into `ListenerTransportExt`).
#[cfg(feature = "http3")]
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_quic_listener_resolves_h3_native() {
    let bind = free_loopback_addr();
    let server = Listener::builder()
        .bind(bind)
        .quic()
        .spec("dev_self_signed", Value::Bool(true))
        .spec("dev_sans", serde_json::json!(["localhost"]))
        .handle(into_handle(FixedOk))
        .serve()
        .await
        .expect(".http().quic() serves");
    server.stop();
}

#[cfg(all(
    feature = "kafka-listener",
    any(feature = "http1", feature = "http1-native")
))]
mod kafka_axis {
    use super::*;
    use proxima_kafka::wire::{ApiVersionsResponse, RequestBody, ResponseBody};
    use proxima_kafka::{KafkaPipeHandle, into_kafka_handle};

    struct StubKafka;

    impl SendPipe for StubKafka {
        type In = RequestBody;
        type Out = ResponseBody;
        type Err = ProximaError;

        async fn call(&self, _request: RequestBody) -> Result<ResponseBody, ProximaError> {
            Ok(ResponseBody::ApiVersions(ApiVersionsResponse::supported()))
        }
    }

    fn stub_handle() -> KafkaPipeHandle {
        into_kafka_handle(StubKafka)
    }

    /// `.kafka(handle).tcp()` — the typed-handle listener axis delegates to
    /// `.protocol()` and serves a real TCP bind.
    #[proxima::test(flavor = "multi_thread", worker_threads = 2)]
    async fn kafka_tcp_listener_serves() {
        let bind = free_loopback_addr();
        let server = Listener::builder()
            .bind(bind)
            .tcp()
            .handle(into_handle(FixedOk))
            .kafka(stub_handle())
            .serve()
            .await
            .expect(".kafka(handle).tcp() serves");
        wait_until_listening(bind);
        server.stop();
    }

    /// Invalid composition: `.kafka(handle).quic()` — kafka's `AnyProtocol`
    /// candidate is TCP-only (`AnyProtocol::drive` takes `Box<dyn
    /// StreamConnection>`); combining with `.quic()` must be a named config
    /// error, never a panic or a silent degrade.
    #[cfg(feature = "http3")]
    #[proxima::test]
    async fn kafka_quic_is_a_named_config_error() {
        let bind = free_loopback_addr();
        let outcome = Listener::builder()
            .bind(bind)
            .quic()
            .handle(into_handle(FixedOk))
            .kafka(stub_handle())
            .serve()
            .await;
        let err = match outcome {
            Ok(_) => panic!(".kafka(handle).quic() must not silently serve"),
            Err(err) => err,
        };
        let text = format!("{err}");
        assert!(
            text.contains(".kafka") || text.contains("TCP-only"),
            "got: {text}"
        );
    }
}

#[cfg(feature = "dns-listener")]
mod dns_axis {
    use super::*;
    use proxima_dns::{DnsAnswer, DnsPipeHandle, DnsPipeReply, DnsPipeRequest, into_dns_handle};

    struct StubDns;

    impl SendPipe for StubDns {
        type In = DnsPipeRequest;
        type Out = DnsPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: DnsPipeRequest) -> Result<DnsPipeReply, ProximaError> {
            Ok(DnsPipeReply::typed(200, DnsAnswer::name_error()))
        }
    }

    fn stub_handle() -> DnsPipeHandle {
        into_dns_handle(StubDns)
    }

    /// `.dns(handle).tcp()` (the default) resolves a TCP-shaped
    /// `AnyListenProtocol` candidate — a raw TCP connect succeeds.
    #[proxima::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dns_tcp_accepts_a_raw_tcp_connect() {
        let bind = free_loopback_addr();
        let server = Listener::builder()
            .bind(bind)
            .tcp()
            .handle(into_handle(FixedOk))
            .dns(stub_handle())
            .serve()
            .await
            .expect(".dns(handle).tcp() serves");
        wait_until_listening(bind);
        server.stop();
    }

    /// `.dns(handle)` reaches BOTH transports on the SAME port number,
    /// regardless of `.tcp()`/`.udp()` — the dual-transport AXIS this
    /// module used to describe (pick TCP xor UDP by reading
    /// `spec["transport"]`) is retired: `.dns(handler)` now registers
    /// `DnsAnyProtocol` (DNS-over-TCP, RFC 1035 §4.2.2 framing) and
    /// `DnsUdpAnyProtocol` (DNS-over-UDP, RFC 1035 §4.2.1, no length
    /// prefix) as two `AnyProtocol` candidates under ONE `.any()`-fanned
    /// listener. Proof: a real DNS-over-TCP query and a real DNS-over-UDP
    /// query, sent to the SAME bind address, both resolve — see
    /// `src/listener/handle.rs`'s `.dns(handler)` doc for why the old
    /// branch is gone.
    #[proxima::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dns_serves_both_transports_on_one_port() {
        use std::io::{Read, Write};
        use std::net::UdpSocket;

        let bind = free_loopback_addr();
        let server = Listener::builder()
            .bind(bind)
            .handle(into_handle(FixedOk))
            .dns(stub_handle())
            .serve()
            .await
            .expect(".dns(handle) serves both transports on one port");
        wait_until_listening(bind);

        let mut query = Vec::new();
        proxima_dns::encode_query(
            42,
            true,
            proxima_dns::EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &mut query,
        )
        .expect("encode query");

        // DNS-over-TCP: 2-byte big-endian length prefix, then the message.
        let mut tcp_conn = StdTcpStream::connect(bind).expect("dns tcp connect");
        let mut framed = Vec::new();
        framed.extend_from_slice(&u16::try_from(query.len()).unwrap().to_be_bytes());
        framed.extend_from_slice(&query);
        tcp_conn.write_all(&framed).expect("dns tcp write");
        let mut length_prefix = [0_u8; 2];
        tcp_conn
            .read_exact(&mut length_prefix)
            .expect("dns tcp read length");
        let reply_len = u16::from_be_bytes(length_prefix) as usize;
        let mut tcp_reply = vec![0_u8; reply_len];
        tcp_conn
            .read_exact(&mut tcp_reply)
            .expect("dns tcp read body");
        let tcp_message = proxima_dns::parse_message(&tcp_reply).expect("dns tcp reply parses");
        assert_eq!(tcp_message.header.id, 42);
        assert!(tcp_message.header.flags.is_response());

        // DNS-over-UDP: the raw message, no length prefix.
        let udp_socket = UdpSocket::bind("127.0.0.1:0").expect("dns udp client bind");
        udp_socket
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set read timeout");
        udp_socket.send_to(&query, bind).expect("dns udp send");
        let mut udp_buf = [0_u8; 512];
        let (udp_len, _) = udp_socket.recv_from(&mut udp_buf).expect("dns udp reply");
        let udp_message =
            proxima_dns::parse_message(&udp_buf[..udp_len]).expect("dns udp reply parses");
        assert_eq!(udp_message.header.id, 42);
        assert!(udp_message.header.flags.is_response());

        server.stop();
    }

    /// Invalid composition: `.dns(handle).quic()` — DNS-over-QUIC (DoQ) is
    /// unimplemented; must be a named config error.
    #[cfg(feature = "http3")]
    #[proxima::test]
    async fn dns_quic_is_a_named_config_error() {
        let bind = free_loopback_addr();
        let outcome = Listener::builder()
            .bind(bind)
            .quic()
            .handle(into_handle(FixedOk))
            .dns(stub_handle())
            .serve()
            .await;
        let err = match outcome {
            Ok(_) => panic!(".dns(handle).quic() must not silently serve"),
            Err(err) => err,
        };
        let text = format!("{err}");
        assert!(text.contains(".dns") || text.contains("DoQ"), "got: {text}");
    }
}

/// Invalid composition: `.grpc().quic()` — gRPC rides h2, not QUIC.
#[cfg(all(feature = "http3", feature = "http2"))]
#[proxima::test]
async fn grpc_quic_is_a_named_config_error() {
    let bind = free_loopback_addr();
    let outcome = Listener::builder()
        .bind(bind)
        .quic()
        .grpc()
        .handle(into_handle(FixedOk))
        .serve()
        .await;
    let err = match outcome {
        Ok(_) => panic!(".grpc().quic() must not silently serve"),
        Err(err) => err,
    };
    let text = format!("{err}");
    assert!(
        text.contains(".grpc") || text.contains("QUIC"),
        "got: {text}"
    );
}

#[cfg(all(
    feature = "websocket-upgrade",
    any(feature = "http1", feature = "http1-native"),
    feature = "http3"
))]
#[proxima::test]
async fn websocket_quic_is_a_named_config_error() {
    use proxima::listener::websocket::WebSocketHandler;
    use std::sync::Arc;

    let handler: WebSocketHandler = Arc::new(|_socket| Box::pin(async { Ok(()) }));
    let bind = free_loopback_addr();
    let outcome = Listener::builder()
        .bind(bind)
        .quic()
        .websocket(handler)
        .handle(into_handle(FixedOk))
        .serve()
        .await;
    let err = match outcome {
        Ok(_) => panic!(".websocket(handler).quic() must not silently serve"),
        Err(err) => err,
    };
    let text = format!("{err}");
    assert!(
        text.contains(".websocket") || text.contains("extended-CONNECT"),
        "got: {text}"
    );
}

/// THIRD-PARTY-style proof: a local `TestThriftExt` trait, defined right
/// here (never inside the umbrella crate), that adds `.thrift(protocol)`
/// mirroring `ListenerProtocolExt`'s own shape — delegating straight to the
/// same `.protocol()` seam `.kafka()`/`.mqtt()`/… use internally. Proves a
/// downstream crate can mint its own builder-sugar extension trait that
/// works IDENTICALLY to the umbrella's first-party ones.
#[cfg(any(feature = "http1", feature = "http1-native"))]
mod third_party_sugar {
    use super::*;
    use proxima::listen::admission::ConnAdmission;
    use proxima::stream::{PeerInfo, StreamConnection};
    use std::pin::Pin;

    const THRIFT_LITERAL: &[u8] = b"THRIFT/1\r\n";
    const THRIFT_REPLY: &[u8] = b"THRIFT/1 OK\r\n";

    struct ThriftProtocol;

    impl AnyProtocol for ThriftProtocol {
        fn name(&self) -> &str {
            "thrift"
        }

        fn max_prefix_bytes(&self) -> usize {
            THRIFT_LITERAL.len()
        }

        fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
            let compare_len = prefix.len().min(THRIFT_LITERAL.len());
            if prefix[..compare_len] != THRIFT_LITERAL[..compare_len] {
                return ProbeVerdict::No;
            }
            if prefix.len() < THRIFT_LITERAL.len() {
                return ProbeVerdict::NeedMore {
                    at_least: THRIFT_LITERAL.len(),
                };
            }
            ProbeVerdict::Match {
                consumed: THRIFT_LITERAL.len(),
            }
        }

        fn drive<'a>(
            &'a self,
            mut stream: Box<dyn StreamConnection>,
            _handler: proxima::listen::any::AnyHandler,
            _spec: &'a Value,
            _peer: Option<PeerInfo>,
            _admission: &'a ConnAdmission,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
            Box::pin(async move {
                use futures::AsyncWriteExt as _;
                stream.write_all(THRIFT_REPLY).await?;
                stream.close().await?;
                Ok(())
            })
        }
    }

    /// The third-party sugar trait itself — proving `.protocol()` is a real
    /// open seam an external crate builds its own axis method on top of.
    trait TestThriftExt: Sized {
        fn thrift(self, protocol: impl AnyProtocol) -> Self;
    }

    impl TestThriftExt for ListenerBuilder {
        fn thrift(self, protocol: impl AnyProtocol) -> Self {
            self.protocol(protocol)
        }
    }

    #[proxima::test(flavor = "multi_thread", worker_threads = 2)]
    async fn third_party_extension_trait_works_identically_to_first_party_sugar() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpStream;

        let bind = free_loopback_addr();
        let server = Listener::builder()
            .bind(bind)
            .tcp()
            .handle(into_handle(FixedOk))
            .any()
            .thrift(ThriftProtocol)
            .serve()
            .await
            .expect("third-party .thrift() sugar serves");
        wait_until_listening(bind);

        let mut conn = TcpStream::connect(bind).await.expect("connect");
        conn.write_all(THRIFT_LITERAL).await.expect("write");
        conn.flush().await.expect("flush");
        let mut collected = Vec::new();
        let _ = conn.read_to_end(&mut collected).await;
        assert_eq!(&collected[..], THRIFT_REPLY);

        server.stop();
    }
}
