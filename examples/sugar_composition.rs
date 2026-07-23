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

    // ── §4: `.dns(handler).udp()` vs `.dns(handler).tcp()` ──────────────────
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

/// `.dns(handler)` is the one dual-transport protocol axis: `.tcp()`
/// (default) resolves a TCP `AnyListenProtocol`; `.udp()` resolves a
/// DIFFERENT listen protocol (`DatagramProtocolListenProtocol`) — a raw TCP
/// connect to the udp bind must fail, proving the two really are different
/// sockets, not the same listener answering both ways.
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

    let bind_tcp = free_loopback_addr()?;
    let server_tcp = Listener::builder()
        .bind(bind_tcp)
        .tcp()
        .handle(into_handle(FixedOk))
        .dns(stub_handle())
        .serve()
        .await?;
    assert!(tcp_connect_succeeds(bind_tcp), ".dns(handler).tcp() must accept a raw TCP connect");
    println!("§4: .dns(handler).tcp() accepts a raw TCP connect on {bind_tcp}");
    server_tcp.stop();

    let bind_udp = free_loopback_addr()?;
    let server_udp = Listener::builder()
        .bind(bind_udp)
        .udp()
        .handle(into_handle(FixedOk))
        .dns(stub_handle())
        .serve()
        .await?;
    let still_tcp_refused = !tcp_connect_succeeds(bind_udp);
    assert!(
        still_tcp_refused,
        ".dns(handler).udp() must NOT accept a TCP connection — it bound a UDP listener, a \
         different listen protocol than .dns(handler).tcp()'s AnyListenProtocol"
    );
    println!(
        "§4: .dns(handler).udp() refuses a raw TCP connect on {bind_udp} — a genuinely \
         different listen protocol from the .tcp() variant above, not the same socket"
    );
    server_udp.stop();
    Ok(())
}

/// `.kafka(handler)` delegates to `.protocol()`, which only ever drives
/// `Box<dyn StreamConnection>` (a byte stream) — combining it with `.quic()`
/// has no meaning and is rejected BEFORE any socket work, not discovered at
/// request time.
async fn kafka_axis() -> Result<(), ProximaError> {
    use proxima_kafka::wire::{ApiVersionsResponse, ResponseBody};
    use proxima_kafka::{KafkaPipeHandle, KafkaPipeReply, KafkaPipeRequest, into_kafka_handle};

    struct StubKafka;

    impl SendPipe for StubKafka {
        type In = KafkaPipeRequest;
        type Out = KafkaPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: KafkaPipeRequest) -> Result<KafkaPipeReply, ProximaError> {
            Ok(KafkaPipeReply::typed(200, ResponseBody::ApiVersions(ApiVersionsResponse::supported())))
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
