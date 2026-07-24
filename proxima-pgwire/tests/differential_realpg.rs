//! Live differential parity against a real PostgreSQL server (gate G13),
//! driven by proxima's OWN client ([`proxima_pgwire::PgClient`]) — no
//! `tokio-postgres`. The same client talking to real PostgreSQL and to our
//! facade must observe an identical protocol contract for `select 1` (column
//! name, type OID, value). Driving real PG with our client also proves the
//! codec interoperates with the canonical incumbent end-to-end.
//!
//! `tokio` appears only for the facade-under-test (inherently async; the
//! production path is proven prime-native by G7) and `spawn_blocking` to host
//! the blocking client off the reactor. The client itself is pure `std::net` +
//! our codec.
//!
//! Requires a reachable server (principle 15 legitimate-deferral cat. 2:
//! external infra). Set `PGWIRE_REALPG_HOST` (+ `_PORT`/`_USER`/`_PASSWORD`/
//! `_DB`) — CI provides it via a `postgres` service. Absent it, the test logs
//! why it is skipping and returns; it is never `#[ignore]`'d.

#![cfg(all(feature = "scram", feature = "listen"))]
#![allow(clippy::expect_used)]

use std::net::{Ipv4Addr, SocketAddr, TcpStream};

use proxima_core::ProximaError;
use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;
use proxima_pgwire::codec::Session;
use proxima_pgwire::{
    ColumnDesc, DescribeReply, Negotiation, PgAuth, PgClient, PgReply, PgServerConfig, QueryReply,
    QueryRequest, SqlValue, Verb, into_pg_handle, negotiate, serve_session,
};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::stream::{StreamConnection, StreamListener, StreamListenerExt};
use proxima_protocols::pgwire_codec::Oid;

const OID_INT4: Oid = Oid(23);

/// Faithful `select 1` engine: matches what real PostgreSQL returns — one
/// `int4` column named `?column?` valued `1` — across simple and extended verbs.
struct SelectOnePipe;

impl SendPipe for SelectOnePipe {
    type In = QueryRequest;
    type Out = PgReply;
    type Err = ProximaError;

    async fn call(&self, request: QueryRequest) -> Result<PgReply, ProximaError> {
        let reply = match request.verb {
            Verb::Query | Verb::Execute { .. } => PgReply::Query(QueryReply::rows(
                vec![ColumnDesc::new("?column?", OID_INT4)],
                vec![vec![SqlValue::Int(1)]],
            )),
            Verb::Parse { .. } => PgReply::Describe(DescribeReply {
                parameter_types: vec![],
                columns: vec![ColumnDesc::new("?column?", OID_INT4)],
            }),
            other => {
                return Err(ProximaError::Config(format!(
                    "select-one pipe got verb {other:?}"
                )));
            }
        };
        Ok(reply)
    }
}

async fn spawn_facade() -> u16 {
    let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind loopback");
    let port = match listener.local_addr().expect("local addr") {
        proxima_primitives::stream::BindAddr::Tcp(addr) => addr.port(),
        other => panic!("expected tcp bind, got {other:?}"),
    };
    tokio::spawn(async move {
        let handle = into_pg_handle(SelectOnePipe);
        loop {
            let Ok(conn) = listener.accept().await else {
                return;
            };
            let handle = handle.clone();
            tokio::spawn(async move {
                let _peer = conn.peer();
                let config = PgServerConfig::default();
                let mut session = Session::new();
                let negotiated = negotiate(conn, &mut session, false)
                    .await
                    .expect("negotiate");
                if let Negotiation::Proceed {
                    stream,
                    startup,
                    leftover,
                } = negotiated
                {
                    serve_session(
                        stream,
                        session,
                        startup,
                        leftover,
                        handle,
                        &PgAuth::Trust,
                        &config,
                        None,
                        None,
                        None,
                    )
                    .await
                    .expect("session completes");
                }
            });
        }
    });
    port
}

/// (column name, type OID, value) of `select 1` via our own client. Blocking,
/// so callers run it on `spawn_blocking`.
fn probe_select_one(
    addr: String,
    user: String,
    password: String,
    database: String,
) -> (String, u32, i32) {
    let stream = TcpStream::connect(&addr).unwrap_or_else(|err| panic!("connect {addr}: {err}"));
    stream.set_nodelay(true).expect("nodelay");
    let mut client =
        PgClient::connect(stream, &user, &password, &database).expect("startup + auth");
    let result = client.simple_query("select 1").expect("select 1");
    let column = result.columns.first().expect("one column");
    let value: i32 = result
        .text(0, 0)
        .expect("one cell")
        .parse()
        .expect("int value");
    let probe = (column.name.clone(), column.type_oid, value);
    let _ = client.close();
    probe
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn select_one_contract_matches_real_postgres() {
    let Ok(host) = std::env::var("PGWIRE_REALPG_HOST") else {
        eprintln!(
            "skipping differential_realpg: PGWIRE_REALPG_HOST unset (no server). \
             CI provides a postgres service; locally run a docker postgres and set it."
        );
        return;
    };
    let port = std::env::var("PGWIRE_REALPG_PORT").unwrap_or_else(|_| "5432".to_string());
    let user = std::env::var("PGWIRE_REALPG_USER").unwrap_or_else(|_| "postgres".to_string());
    let password =
        std::env::var("PGWIRE_REALPG_PASSWORD").unwrap_or_else(|_| "proxima".to_string());
    let database = std::env::var("PGWIRE_REALPG_DB").unwrap_or_else(|_| "postgres".to_string());

    let real_addr = format!("{host}:{port}");
    let (ru, rp, rd) = (user.clone(), password.clone(), database.clone());
    let real = tokio::task::spawn_blocking(move || probe_select_one(real_addr, ru, rp, rd))
        .await
        .expect("real-pg probe");

    let facade_port = spawn_facade().await;
    let facade_addr = format!("127.0.0.1:{facade_port}");
    let facade = tokio::task::spawn_blocking(move || {
        probe_select_one(
            facade_addr,
            "postgres".to_string(),
            String::new(),
            "postgres".to_string(),
        )
    })
    .await
    .expect("facade probe");

    assert_eq!(
        facade, real,
        "our client must see an identical (name, type_oid, value) from our facade \
         and from real PostgreSQL for `select 1`"
    );
    assert_eq!(
        real.1, OID_INT4.0,
        "sanity: real PG types `select 1` as int4 (oid 23)"
    );
}
