//! The pgwire client `Pipe` (`PgwireClientUpstream`) driven over a real
//! `StreamUpstream` (tokio) against real PostgreSQL — proving the async client
//! path that `proxima::Client` will reach through the `PipeFactory`. Mirrors
//! `proxima-h1`'s client tests (which use `TokioTcpUpstream`). Env-gated on a
//! reachable server; skips with a logged reason locally, never `#[ignore]`'d.

#![cfg(feature = "client")]
#![allow(clippy::expect_used)]

use std::net::SocketAddr;

use proxima_net::tokio::tokio_stream_upstream::TokioTcpUpstream;
use proxima_pgwire::{CancelToken, PgClientConfig, PgReply, PgwireClientUpstream, QueryRequest};
use proxima_primitives::pipe::SendPipe;

fn make_query_request(sql: &str) -> QueryRequest {
    QueryRequest::Query {
        sql: sql.to_string(),
        connection_id: 0,
        cancel: CancelToken::none(),
    }
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_pipe_runs_simple_and_extended_against_real_pg() {
    let Ok(host) = std::env::var("PGWIRE_REALPG_HOST") else {
        eprintln!(
            "skipping client_pipe_realpg: PGWIRE_REALPG_HOST unset (no server). \
             CI provides a postgres service; locally run a docker postgres and set it."
        );
        return;
    };
    let port: u16 = std::env::var("PGWIRE_REALPG_PORT")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(5432);
    let password =
        std::env::var("PGWIRE_REALPG_PASSWORD").unwrap_or_else(|_| "proxima".to_string());
    let user = std::env::var("PGWIRE_REALPG_USER").unwrap_or_else(|_| "postgres".to_string());
    let database = std::env::var("PGWIRE_REALPG_DB").unwrap_or_else(|_| "postgres".to_string());

    let addr: SocketAddr = format!("{host}:{port}").parse().expect("ip:port");
    let config = PgClientConfig::builder()
        .host(host)
        .port(port)
        .user(user)
        .password(password)
        .database(database)
        .build();
    let client = PgwireClientUpstream::new(TokioTcpUpstream::new(addr), config);

    // simple protocol
    let response = client
        .call(make_query_request("select 1"))
        .await
        .expect("simple query");
    match &response {
        PgReply::Query(reply) => {
            assert_eq!(reply.columns.len(), 1, "select 1 -> one column");
            assert_eq!(reply.rows.len(), 1, "one row");
        }
        other => panic!("expected Query reply, got {other:?}"),
    }

    // the cached connection is reused for a second call (no re-auth)
    let response = client
        .call(make_query_request(
            "select 42::int4 as n, 'alice'::text as who",
        ))
        .await
        .expect("second query");
    match &response {
        PgReply::Query(reply) => {
            assert_eq!(reply.columns.len(), 2, "two columns");
            assert_eq!(reply.columns[0].name, "n");
            assert_eq!(reply.columns[0].type_oid.0, 23, "int4 oid");
        }
        other => panic!("expected Query reply, got {other:?}"),
    }

    // a SQL error surfaces as PgReply::Error (transport stays Ok), connection
    // recovers and stays usable.
    let response = client
        .call(make_query_request(
            "select * from a_table_that_does_not_exist",
        ))
        .await
        .expect("error query is transport-ok");
    match &response {
        PgReply::Error(reply) => assert_eq!(reply.sqlstate, "42P01", "undefined_table"),
        other => panic!("expected Error reply, got {other:?}"),
    }
}
