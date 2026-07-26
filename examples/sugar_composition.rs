#![allow(clippy::unwrap_used, clippy::expect_used)]

//! The builder-sugar composition families, end to end: transport
//! (`.tcp()`/`.udp()`/`.quic()`), security (`.tls(cfg)`), and protocol
//! (`.http()`/`.grpc()`/`.kafka()`/`.dns()`), all riding the SAME
//! `Listener::builder()`/`Client::builder()` seam — each axis its own
//! TYPE-SPECIFIC extension trait (`ListenerTransportExt`/`ListenerProtocolExt`
//! vs. `ClientTransportExt`/`ClientProtocolExt`/`ClientSecurityExt`), no
//! blanket trait reaching across builders. `use proxima::prelude::*;` brings
//! every first-party axis trait into scope at once.
//!
//! Every section proves ONE composition; the last two prove the honest
//! failure mode — an invalid composition (`.kafka().quic()`, `.grpc().quic()`)
//! is a named `ProximaError::Config`, printed verbatim below, never a silent
//! degrade to some other wire.
//!
//! Grounded in `tests/e2e/listener_builder_sugar.rs`, which proves the same
//! compositions as `#[proxima::test]` assertions; this file is the
//! `cargo run`-able walkthrough.
//!
//! Run: `cargo run --example sugar_composition --features
//! "http1-native,tls,http3,kafka-listener,dns-listener"`

use std::net::{Ipv4Addr, SocketAddr, TcpStream as StdTcpStream};
use std::time::Duration;

use bytes::Bytes;
use serde_json::json;

use proxima::pipe::into_handle;
use proxima::prelude::*;
use proxima::request::{Request, Response};
use proxima::tls::TlsConfig;
use proxima::{ProximaError, SendPipe};

struct FixedOk;

impl SendPipe for FixedOk {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        Ok(Response::ok("sugar-composition-ok"))
    }
}

fn free_loopback_addr() -> Result<SocketAddr, ProximaError> {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let addr = probe.local_addr()?;
    drop(probe);
    Ok(addr)
}

fn tcp_connect_succeeds(addr: SocketAddr) -> bool {
    for _ in 0..20 {
        if StdTcpStream::connect(addr).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    // ── §1: bare `.http()` + a real client dial, both sides `.tcp()` ────────
    let bind_1 = free_loopback_addr()?;
    let server_1 = Listener::builder()
        .bind(bind_1)
        .tcp()
        .handle(into_handle(FixedOk))
        .serve()
        .await?;
    assert!(tcp_connect_succeeds(bind_1), "§1 listener never came up");

    let client = Client::builder()
        .http(format!("http://{bind_1}"))
        .tcp()
        .build()?;
    let response = client.call("GET", "/").send().await?;
    assert_eq!(response.status(), 200);
    println!("§1: .http().tcp() listener + .http(url).tcp() client -> {}", response.status());
    server_1.stop();

    // ── §2: `.http().tcp().tls(cfg)` composes — TLS as a decorator ──────────
    let bind_2 = free_loopback_addr()?;
    let server_2 = Listener::builder()
        .bind(bind_2)
        .tcp()
        .tls(TlsConfig::self_signed())
        .handle(into_handle(FixedOk))
        .serve()
        .await?;
    assert!(tcp_connect_succeeds(bind_2), "§2 TLS-wrapped listener never came up");
    println!(
        "§2: .http().tcp().tls(cfg) serves on {bind_2} — TLS terminates as a decorator over \
         the SAME h1+h2 combiner, not a spec key (see tests/e2e/listener_client_interop.rs for \
         the full handshake proof)"
    );
    server_2.stop();

    // ── §3: `.http().quic()` IS h3 — resolves the native h3 listener ────────
    let bind_3 = free_loopback_addr()?;
    let server_3 = Listener::builder()
        .bind(bind_3)
        .quic()
        .spec("dev_self_signed", json!(true))
        .spec("dev_sans", json!(["localhost"]))
        .handle(into_handle(FixedOk))
        .serve()
        .await?;
    println!(
        "§3: .http(bind).quic() resolves to the native h3-native DatagramProtocol listener on \
         {bind_3} — a real UDP bind, not the ALPN h1+h2 combiner. There is no separate `.h3()` \
         method any more; `.quic()` on `.http()` IS h3."
    );
    server_3.stop();

    // ── §4: `.dns(handler)` answers BOTH transports on one port ─────────────
    dns_axis().await?;

    // ── §5: `.kafka(handler).tcp()` succeeds; `.quic()` is a named error ────
    kafka_axis().await?;

    // ── §6: `.grpc().quic()` is a named error too (gRPC rides h2, not h3) ───
    let bind_6 = free_loopback_addr()?;
    let outcome = Listener::builder()
        .bind(bind_6)
        .quic()
        .grpc()
        .handle(into_handle(FixedOk))
        .serve()
        .await;
    let error = match outcome {
        Ok(_) => panic!(".grpc().quic() must not silently serve"),
        Err(error) => error,
    };
    println!("§6: .grpc().quic() -> named ProximaError::Config:\n    {error}");

    println!("\nsugar_composition: every transport/security/protocol composition above OK");
    Ok(())
}

/// `.dns(handler)` used to be a dual-transport protocol axis: `.tcp()`
/// (default) resolved a TCP `AnyListenProtocol`, `.udp()` a DIFFERENT listen
/// protocol (`DatagramProtocolListenProtocol`) — pick one. That branch is
/// retired: `.dns(handler)` now registers TWO `AnyProtocol` candidates
/// (DNS-over-TCP and DNS-over-UDP) under ONE `.any()`-fanned listener, so a
/// SINGLE bind answers both a DNS-over-TCP query (RFC 1035 §4.2.2, 2-byte
/// length prefix) and a DNS-over-UDP query (RFC 1035 §4.2.1, raw message) —
/// `.tcp()`/`.udp()` are no longer read by `.dns(handler)` at all. See
/// `docs/tutorials/11-any-transport-agnostic.md` for the general mechanism
/// (`AnyProtocol::wants_datagram`) this sugar is built on.
async fn dns_axis() -> Result<(), ProximaError> {
    use proxima_dns::{DnsAnswer, DnsPipeHandle, DnsPipeReply, DnsPipeRequest, into_dns_handle};

    struct NameErrorDns;

    impl SendPipe for NameErrorDns {
        type In = DnsPipeRequest;
        type Out = DnsPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: DnsPipeRequest) -> Result<DnsPipeReply, ProximaError> {
            Ok(DnsPipeReply::typed(200, DnsAnswer::name_error()))
        }
    }

    fn stub_handle() -> DnsPipeHandle {
        into_dns_handle(NameErrorDns)
    }

    let bind = free_loopback_addr()?;
    let server = Listener::builder()
        .bind(bind)
        .handle(into_handle(FixedOk))
        .dns(stub_handle())
        .serve()
        .await?;
    assert!(tcp_connect_succeeds(bind), ".dns(handler) must accept a raw TCP connect");

    let mut query = Vec::new();
    proxima_dns::encode_query(
        7,
        true,
        proxima_dns::EncodeQuestion {
            name: "example.test.",
            qtype: 1,
            qclass: 1,
        },
        &mut query,
    )
    .map_err(|error| ProximaError::Config(format!("encode dns query: {error}")))?;

    // DNS-over-TCP: 2-byte big-endian length prefix, then the message.
    let mut tcp_conn = StdTcpStream::connect(bind)?;
    let mut framed = Vec::new();
    framed.extend_from_slice(&u16::try_from(query.len()).unwrap_or(u16::MAX).to_be_bytes());
    framed.extend_from_slice(&query);
    std::io::Write::write_all(&mut tcp_conn, &framed)?;
    let mut length_prefix = [0_u8; 2];
    std::io::Read::read_exact(&mut tcp_conn, &mut length_prefix)?;
    let reply_len = u16::from_be_bytes(length_prefix) as usize;
    let mut tcp_reply = vec![0_u8; reply_len];
    std::io::Read::read_exact(&mut tcp_conn, &mut tcp_reply)?;
    let tcp_message = proxima_dns::parse_message(&tcp_reply)
        .map_err(|error| ProximaError::Config(format!("parse dns tcp reply: {error}")))?;
    assert_eq!(tcp_message.header.id, 7);

    // DNS-over-UDP: the SAME bind address, the raw message, no length prefix.
    let udp_socket = std::net::UdpSocket::bind("127.0.0.1:0")?;
    udp_socket.set_read_timeout(Some(Duration::from_secs(2)))?;
    udp_socket.send_to(&query, bind)?;
    let mut udp_buf = [0_u8; 512];
    let (udp_len, _) = udp_socket.recv_from(&mut udp_buf)?;
    let udp_message = proxima_dns::parse_message(&udp_buf[..udp_len])
        .map_err(|error| ProximaError::Config(format!("parse dns udp reply: {error}")))?;
    assert_eq!(udp_message.header.id, 7);

    println!(
        "§4: .dns(handler) on {bind} answers BOTH a DNS-over-TCP query (id {}) and a \
         DNS-over-UDP query (id {}) on the SAME port — no .tcp()/.udp() call needed",
        tcp_message.header.id, udp_message.header.id
    );
    server.stop();
    Ok(())
}

/// `.kafka(handler)` delegates to `.protocol()`, which only ever drives
/// `Box<dyn StreamConnection>` (a byte stream) — combining it with `.quic()`
/// has no meaning and is rejected BEFORE any socket work, not discovered at
/// request time.
async fn kafka_axis() -> Result<(), ProximaError> {
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

    let bind_ok = free_loopback_addr()?;
    let server_ok = Listener::builder()
        .bind(bind_ok)
        .tcp()
        .handle(into_handle(FixedOk))
        .kafka(stub_handle())
        .serve()
        .await?;
    assert!(tcp_connect_succeeds(bind_ok), ".kafka(handler).tcp() must serve");
    println!("§5: .kafka(handler).tcp() serves on {bind_ok}");
    server_ok.stop();

    let bind_bad = free_loopback_addr()?;
    let outcome = Listener::builder()
        .bind(bind_bad)
        .quic()
        .handle(into_handle(FixedOk))
        .kafka(stub_handle())
        .serve()
        .await;
    let error = match outcome {
        Ok(_) => panic!(".kafka(handler).quic() must not silently serve"),
        Err(error) => error,
    };
    println!("§5: .kafka(handler).quic() -> named ProximaError::Config:\n    {error}");
    Ok(())
}
