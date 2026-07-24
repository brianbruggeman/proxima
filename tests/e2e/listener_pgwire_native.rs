//! End-to-end test for `.pgwire(query)` through `Listener::builder()` — the
//! builder axis, not the hand-built `ListenRegistry` +
//! `PgWireListenProtocol::new` shortcut
//! `proxima-pgwire/tests/integration/prime_e2e.rs` uses. Same engine shape
//! (`select 1` echo) and the same blocking `PgClient` driver over a plain
//! `std::net::TcpStream` (no `tokio-postgres`), proving the typed query
//! engine `.pgwire(query)` carries actually reaches a real PostgreSQL wire
//! client.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "pgwire")]

use std::net::{Ipv4Addr, SocketAddr, TcpStream};

use bytes::Bytes;
use proxima::error::ProximaError;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt};
use proxima_pgwire::{ColumnDesc, PgReply, QueryReply, QueryRequest, SqlValue, into_pg_handle};
use proxima_primitives::pipe::SendPipe;
use proxima_protocols::pgwire_codec::Oid;

const OID_INT4: Oid = Oid(23);

struct EchoPipe;

impl SendPipe for EchoPipe {
    type In = QueryRequest;
    type Out = PgReply;
    type Err = ProximaError;

    async fn call(&self, request: QueryRequest) -> Result<PgReply, ProximaError> {
        let sql = request.sql().to_owned();
        let reply = if sql.trim().eq_ignore_ascii_case("select 1") {
            PgReply::Query(QueryReply::rows(
                vec![ColumnDesc::new("?column?", OID_INT4)],
                vec![vec![SqlValue::Int(1)]],
            ))
        } else {
            PgReply::Error(proxima_pgwire::ErrorReply::new(
                "42601",
                format!("test engine only knows select 1, got: {sql}"),
            ))
        };
        Ok(reply)
    }
}

struct NeverDispatch;

impl SendPipe for NeverDispatch {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        Err(ProximaError::Config(
            "registry dispatch must not be reached when the protocol carries its own engine".into(),
        ))
    }
}

fn free_loopback_addr() -> SocketAddr {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("probe bind");
    let addr = probe.local_addr().expect("probe addr");
    drop(probe);
    addr
}

fn wait_until_listening(addr: SocketAddr) -> TcpStream {
    for _ in 0..200 {
        if let Ok(stream) = TcpStream::connect(addr) {
            return stream;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("pgwire listener at {addr} never came up");
}

#[proxima::test]
async fn listener_builder_pgwire_serves_a_real_pgclient() {
    let bind = free_loopback_addr();

    let server = Listener::builder()
        .bind(bind)
        .pgwire(into_pg_handle(EchoPipe))
        .handle(into_handle(NeverDispatch))
        .serve()
        .await
        .expect("Listener::builder().pgwire(query) serve");

    let result = std::thread::spawn(move || {
        let stream = wait_until_listening(bind);
        let mut client = proxima_pgwire::PgClient::connect(stream, "smoke", "", "smokedb")
            .expect("pgwire startup + auth");
        let result = client.simple_query("select 1").expect("select 1");
        let _ = client.close();
        result
    })
    .join()
    .expect("client thread");

    assert_eq!(result.rows.len(), 1, "select 1 must return exactly one row");
    assert_eq!(
        result.text(0, 0),
        Some("1"),
        "the single column must decode to 1"
    );

    server.stop();
}
