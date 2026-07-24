#![allow(clippy::expect_used)]
//! G7 native witness: the SAME `PgWireListenProtocol` the tokio e2e tests
//! drive serves real SQL clients when mounted on the PRIME runtime via the
//! runtime-matched acceptor factory. The tokio tests in `client_smoke.rs`
//! prove the codec over `proxima_net::tokio`; this proves the
//! runtime-agnostic path the whole facade rests on — prime accept loop +
//! prime per-core executor running `PgWireConnectionPipe`'s upgrade and
//! `serve_session` over the prime-accepted `futures::io` stream.
//!
//! The mount is the canonical App shape: register the protocol in a
//! `ListenRegistry`, attach a dispatch, and `Listener::run_with_runtime`
//! with `PrimeRuntime` + `PrimeAcceptorFactory`. That constructs the
//! `ServeContext { runtime: prime, acceptor_factory: prime }` and calls
//! `PgWireListenProtocol::serve` on a prime worker core. The client is a
//! real tokio-postgres on an independent tokio runtime — the client's
//! executor is decoupled from the server's.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_listen::handle::{ListenerHandle, ListenerSpec, ShutdownPolicy};
use proxima_listen::{ListenProtocol, ListenRegistry};
use proxima_pgwire::{
    ColumnDesc, DescribeReply, ErrorReply, PgPipeHandle, PgReply, PgWireListenProtocol, QueryReply,
    QueryRequest, SqlValue, Verb, into_pg_handle,
};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::telemetry_surface::NoopTelemetry;
use proxima_protocols::pgwire_codec::Oid;
use proxima_runtime::Runtime;

use prime::PrimeRuntime;
use proxima_net::prime::PrimeAcceptorFactory;
use proxima_primitives::stream::AcceptorFactory;

const OID_INT4: Oid = Oid(23);

/// The SQL engine as a `Pipe` (mirrors `client_smoke.rs`): QUERY "select 1"
/// → one int column one row; PARSE → a Describe declaring the param +
/// result shape; EXECUTE → one int column one row whose value is the bound
/// parameter, round-tripped from the typed `SqlValue` the driver decoded.
struct EchoPipe;

impl SendPipe for EchoPipe {
    type In = QueryRequest;
    type Out = PgReply;
    type Err = ProximaError;

    async fn call(&self, request: QueryRequest) -> Result<PgReply, ProximaError> {
        let QueryRequest { sql, verb, .. } = request;
        let reply = match verb {
            Verb::Query => echo_query(&sql),
            Verb::Parse { .. } => PgReply::Describe(echo_describe(&sql)),
            Verb::Execute { parameters, .. } => echo_execute(&parameters),
            other => {
                return Err(ProximaError::Config(format!(
                    "echo pipe received unexpected verb {other:?}"
                )));
            }
        };
        Ok(reply)
    }
}

fn echo_query(sql: &str) -> PgReply {
    if sql.trim().eq_ignore_ascii_case("select 1") {
        return PgReply::Query(QueryReply::rows(
            vec![ColumnDesc::new("?column?", OID_INT4)],
            vec![vec![SqlValue::Int(1)]],
        ));
    }
    PgReply::Error(ErrorReply::new(
        "42601",
        format!("test engine only knows select 1, got: {sql}"),
    ))
}

fn echo_describe(sql: &str) -> DescribeReply {
    if sql.contains("$1") {
        return DescribeReply {
            parameter_types: vec![OID_INT4],
            columns: vec![ColumnDesc::new("v", OID_INT4)],
        };
    }
    DescribeReply {
        parameter_types: vec![],
        columns: vec![ColumnDesc::new("v", OID_INT4)],
    }
}

fn echo_execute(parameters: &[SqlValue]) -> PgReply {
    let value = match parameters.first() {
        Some(SqlValue::Int(number)) => *number,
        Some(SqlValue::Text(text)) => match text.parse::<i64>() {
            Ok(number) => number,
            Err(_) => return PgReply::Error(ErrorReply::new("22P02", "invalid int4 text")),
        },
        Some(SqlValue::Null) | None => 1,
        Some(_) => return PgReply::Error(ErrorReply::new("22023", "unsupported parameter type")),
    };
    PgReply::Query(QueryReply::rows(
        vec![ColumnDesc::new("v", OID_INT4)],
        vec![vec![SqlValue::Int(value)]],
    ))
}

/// Mount `PgWireListenProtocol` on the prime runtime exactly as an `App`
/// does: register in a `ListenRegistry`, build a `ListenerSpec` naming the
/// protocol, attach the engine as dispatch, and `run_with_runtime` with the
/// prime runtime + prime acceptor factory. Returns the live handle (owns
/// the prime runtime + shutdown sender) and the resolved bind address.
///
/// single-core prime: `PrimeAcceptorFactory` ignores SO_REUSEPORT (prime
/// serve is single-core today), so one lane avoids a same-port double-bind.
fn mount_on_prime(label: &str, query: PgPipeHandle) -> (ListenerHandle, SocketAddr) {
    let runtime: Arc<dyn Runtime> = Arc::new(PrimeRuntime::new(1).expect("build prime runtime"));
    let factory: Arc<dyn AcceptorFactory> = Arc::new(PrimeAcceptorFactory);

    let registry = ListenRegistry::new();
    let protocol: Arc<dyn ListenProtocol> = Arc::new(PgWireListenProtocol::new(label, query));
    registry
        .register(protocol)
        .expect("register pgwire protocol");

    let bind = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    let mut spec = ListenerSpec::http(bind).with_shutdown(ShutdownPolicy::Immediate);
    spec.protocol_name = label.to_string();
    let listener = spec.attach(into_handle(NeverDispatch));

    let handle = listener
        .run_with_runtime(
            &registry,
            NoopTelemetry::handle(),
            Some(runtime),
            Some(factory),
            None,
        )
        .expect("mount pgwire listener on prime");
    let addr = handle.bind_addr().expect("listener resolved a bind addr");
    (handle, addr)
}

/// The registry dispatch must never be exercised: `PgWireListenProtocol::new`
/// supplies its own query engine, so the dispatch pipe is unused. Answering
/// with an error if it is ever called makes a wiring regression loud.
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

fn tokio_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("client tokio runtime")
}

async fn connect_when_ready(conninfo: &str) -> tokio_postgres::Client {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match tokio_postgres::connect(conninfo, tokio_postgres::NoTls).await {
            Ok((client, connection)) => {
                tokio::spawn(connection);
                return client;
            }
            Err(error) => {
                assert!(
                    Instant::now() < deadline,
                    "tokio-postgres never connected to the prime-mounted listener: {error}"
                );
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }
}

#[test]
fn prime_mounted_listener_serves_select_1_to_a_real_client() {
    let (handle, addr) = mount_on_prime("pg-prime-simple", into_pg_handle(EchoPipe));

    let client_runtime = tokio_runtime();
    let port = addr.port();
    client_runtime.block_on(async move {
        let conninfo =
            format!("host=127.0.0.1 port={port} user=smoke dbname=smokedb connect_timeout=5");
        let client = connect_when_ready(&conninfo).await;
        let rows = client
            .query("select 1", &[])
            .await
            .expect("simple-protocol select 1 must round-trip through the prime serve path");
        assert_eq!(rows.len(), 1, "select 1 must return exactly one row");
        let value: i32 = rows[0].get(0);
        assert_eq!(value, 1, "the single column must decode to 1");
        drop(client);
    });

    handle.shutdown();
}

#[test]
fn prime_mounted_listener_round_trips_an_extended_query_param() {
    let (handle, addr) = mount_on_prime("pg-prime-extended", into_pg_handle(EchoPipe));

    let client_runtime = tokio_runtime();
    let port = addr.port();
    client_runtime.block_on(async move {
        let conninfo =
            format!("host=127.0.0.1 port={port} user=smoke dbname=smokedb connect_timeout=5");
        let client = connect_when_ready(&conninfo).await;
        let rows = client
            .query("select $1::int4 as v", &[&7_i32])
            .await
            .expect("extended query must round-trip through the prime serve path");
        assert_eq!(rows.len(), 1, "extended query must return one row");
        let value: i32 = rows[0].get("v");
        assert_eq!(
            value, 7,
            "the bound parameter must round-trip via Parse/Bind/Execute"
        );
        drop(client);
    });

    handle.shutdown();
}
