//! `.pgwire(query)` through `Listener::builder()`: binds
//! [`PgWireListenProtocol`](proxima_pgwire::PgWireListenProtocol) carrying a
//! typed SQL query engine, then proves it with a real PostgreSQL wire client
//! — [`PgClient`](proxima_pgwire::PgClient), the blocking sans-IO session
//! driver over a plain `std::net::TcpStream` (no `tokio-postgres`).
//!
//! ```sh
//! cargo run --example pgwire_server --features pgwire
//! ```
//!
//! The query engine here is the same shape as
//! `proxima-pgwire/tests/integration/prime_e2e.rs`'s `EchoPipe`: it only
//! knows `select 1`, echoing a single int column/row — enough to prove the
//! wire round trip without a real SQL executor.

use std::net::{Ipv4Addr, SocketAddr, TcpStream};

use bytes::Bytes;
use proxima::error::ProximaError;
use proxima::pipe::into_handle;
use proxima::request::{Request, Response};
use proxima::{Listener, ListenerBuilderEntry};
use proxima_pgwire::{
    ColumnDesc, ErrorReply, PgReply, PgRequest, PgResponse, QueryReply, SqlValue, into_pg_handle,
    verb,
};
use proxima_primitives::pipe::SendPipe;
use proxima_protocols::pgwire_codec::Oid;

const OID_INT4: Oid = Oid(23);

/// The SQL engine as a `Pipe`: `select 1` -> one int column, one row.
/// `.pgwire(query)` is the only place this typed handle is threaded
/// through — the general `.handle(pipe)` dispatch below is unused (a
/// pgwire listener with a constructor-supplied query engine never falls
/// back to it), which is why `NeverDispatch` errors loudly if it is ever
/// reached: a wiring regression should be loud, not silently wrong.
struct EchoPipe;

impl SendPipe for EchoPipe {
    type In = PgRequest;
    type Out = PgResponse;
    type Err = ProximaError;

    async fn call(&self, request: PgRequest) -> Result<PgResponse, ProximaError> {
        let sql = request.payload.sql.clone();
        let reply = if sql.trim().eq_ignore_ascii_case("select 1") {
            PgReply::Query(QueryReply::rows(
                vec![ColumnDesc::new("?column?", OID_INT4)],
                vec![vec![SqlValue::Int(1)]],
            ))
        } else if request.method.as_bytes() == verb::QUERY {
            PgReply::Error(ErrorReply::new(
                "42601",
                format!("example engine only knows select 1, got: {sql}"),
            ))
        } else {
            return Err(ProximaError::Config(format!(
                "example engine received unexpected verb {:?}",
                request.method
            )));
        };
        Ok(Response::typed(200, reply))
    }
}

struct NeverDispatch;

impl SendPipe for NeverDispatch {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
        Err(ProximaError::Config(
            "registry dispatch must not be reached when the protocol carries its own engine"
                .into(),
        ))
    }
}

/// Grab a free port with a plain std socket (closes synchronously on drop),
/// then hand the vacated port to `.bind()` — `Server` (unlike
/// `ListenerHandle`) has no `bind_addr()` to discover an ephemeral port
/// after the fact.
fn free_loopback_addr() -> Result<SocketAddr, ProximaError> {
    let probe = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let addr = probe.local_addr()?;
    drop(probe);
    Ok(addr)
}

fn wait_until_listening(addr: SocketAddr) -> Result<TcpStream, ProximaError> {
    for _ in 0..200 {
        if let Ok(stream) = TcpStream::connect(addr) {
            return Ok(stream);
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    Err(ProximaError::Config(format!(
        "pgwire listener at {addr} never came up"
    )))
}

/// The blocking client side, run on its own OS thread (`PgClient` is
/// synchronous `Read + Write`, not a `Future` — driving it inline on the
/// async task would block the whole worker).
fn run_client(addr: SocketAddr) -> Result<(), ProximaError> {
    let stream = wait_until_listening(addr)?;
    let mut client = proxima_pgwire::PgClient::connect(stream, "smoke", "", "smokedb")
        .map_err(|error| ProximaError::Config(format!("pgwire startup + auth: {error}")))?;
    let result = client
        .simple_query("select 1")
        .map_err(|error| ProximaError::Config(format!("select 1: {error}")))?;
    if result.rows.len() != 1 {
        return Err(ProximaError::Config(format!(
            "select 1 must return exactly one row, got {}",
            result.rows.len()
        )));
    }
    if result.text(0, 0) != Some("1") {
        return Err(ProximaError::Config(format!(
            "the single column must decode to 1, got {:?}",
            result.text(0, 0)
        )));
    }
    let _ = client.close();
    println!("pgwire_server: select 1 round trip through Listener::builder().pgwire() OK on {addr}");
    Ok(())
}

#[proxima::main]
async fn main() -> Result<(), ProximaError> {
    let bind = free_loopback_addr()?;

    let server = Listener::builder()
        .bind(bind)
        .pgwire(into_pg_handle(EchoPipe))
        .handle(into_handle(NeverDispatch))
        .serve()
        .await?;

    std::thread::spawn(move || run_client(bind))
        .join()
        .map_err(|_| ProximaError::Config("pgwire client thread panicked".into()))??;

    server.stop();
    Ok(())
}
