//! `PipeFactory` for the `pgwire` protocol — a `proxima::Client` transport that
//! speaks the PostgreSQL wire protocol.
//!
//! Reached via the `type` discriminator (`{"type":"pgwire", "dsn": "..."}` or
//! `{"type":"pgwire", "host":..., "port":..., ...}`), so it needs no edit to the
//! spec precedence chain — the extensible terminal seam. Composes the sans-IO
//! pgwire client ([`PgwireClientUpstream`](proxima_pgwire::PgwireClientUpstream))
//! over the prime TCP transport ([`PrimeTcpUpstream`](crate::PrimeTcpUpstream)),
//! exactly like the prime `http`/`grpc` factories. A client and a server are the
//! same `Handler`; this is the initiating half. The SQL-over-Handler request shape
//! (`verb::QUERY`/`EXECUTE` + body SQL + `QueryRequest` carry) is the caller's;
//! this factory is purely the transport.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use serde_json::Value;

use proxima_pgwire::{PgClientConfig, PgReply, PgResponse, PgwireClientUpstream, QueryRequest};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::pipe_factory::PipeFactory;
use proxima_primitives::pipe::request::{Request, Response};

use crate::PrimeTcpUpstream;
use crate::client::handle::ClientProtocol;
use crate::error::ProximaError;

/// A [`PipeFactory`] for the `pgwire` key. Builds a client `Handler` from a
/// [`PgClientConfig`] parsed out of the spec.
#[derive(Debug, Default)]
pub struct PgwirePipeFactory;

impl PgwirePipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PipeFactory for PgwirePipeFactory {
    fn name(&self) -> &str {
        "pgwire"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config = config_from_spec(&spec)?;
            // DNS is resolved lazily by the prime upstream on connect, so
            // `build` stays side-effect-free (mirrors the prime http factory).
            let upstream = PrimeTcpUpstream::with_host(config.host.clone(), config.port);
            Ok(into_handle(PgwireClientPipe::new(
                PgwireClientUpstream::new(upstream, config),
            )))
        })
    }
}

struct PgwireClientPipe {
    inner: PgwireClientUpstream<PrimeTcpUpstream>,
}

impl PgwireClientPipe {
    fn new(inner: PgwireClientUpstream<PrimeTcpUpstream>) -> Self {
        Self { inner }
    }
}

impl SendPipe for PgwireClientPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let (request, body) = request.body_bytes().await?;
            let mut query = QueryRequest::new(0);
            query.sql = String::from_utf8_lossy(&body).into_owned();
            let request = Request {
                method: request.method,
                path: request.path,
                query: request.query,
                metadata: request.metadata,
                payload: query,
                stream: None,
                context: request.context,
            };
            let response = self.inner.call(request).await?;
            Ok(pg_response_to_bytes(response))
        }
    }
}


fn pg_response_to_bytes(response: PgResponse) -> Response<Bytes> {
    let Response {
        status,
        metadata,
        payload,
        stream,
        upgrade,
    } = response;
    let body = serde_json::to_vec(&pg_reply_to_json(&payload))
        .unwrap_or_else(|err| format!("{{\"encode_error\":\"{err}\"}}").into_bytes());
    let mut response = Response::new(status).with_payload(body);
    response.metadata = metadata;
    response.stream = stream;
    response.upgrade = upgrade;
    response
}

/// `Response<PgReply>` collapses to `Response<Bytes>` at the dynamic-dispatch
/// boundary every `PipeHandle` shares (one dispatch shape — see `Delay`/
/// `Filter`, which hardcode the same `Response<Bytes>` for the same reason).
/// The bytes need to actually decode back to something, so this JSON-encodes
/// the reply rather than the previous `format!("{reply:?}")` debug string —
/// mirrors how the `redis` client path round-trips real RESP bytes rather
/// than threading a typed value through `Response` itself.
///
/// The client driver (`proxima_pgwire::client::pipe`) only ever produces
/// `Query` and `Error`; the other variants are server/engine-role replies
/// this transport never sees from a real postgres server, so they fall back
/// to a debug string.
fn pg_reply_to_json(reply: &PgReply) -> Value {
    match reply {
        PgReply::Query(query) => serde_json::json!({
            "kind": "query",
            "columns": query.columns.iter().map(|column| serde_json::json!({
                "name": column.name,
                "type_oid": column.type_oid.0,
            })).collect::<Vec<_>>(),
            "rows": query.rows.iter().map(|row| {
                row.iter().map(sql_value_to_json).collect::<Vec<_>>()
            }).collect::<Vec<_>>(),
            "command_tag": query.command_tag,
        }),
        PgReply::Error(error) => serde_json::json!({
            "kind": "error",
            "severity": error.severity,
            "sqlstate": error.sqlstate,
            "message": error.message,
            "detail": error.detail,
            "hint": error.hint,
        }),
        other => serde_json::json!({ "kind": "unsupported", "debug": format!("{other:?}") }),
    }
}

fn sql_value_to_json(value: &proxima_pgwire::SqlValue) -> Value {
    match value {
        proxima_pgwire::SqlValue::Null => Value::Null,
        proxima_pgwire::SqlValue::Int(number) => serde_json::json!(number),
        proxima_pgwire::SqlValue::Float(number) => serde_json::json!(number),
        proxima_pgwire::SqlValue::Bool(flag) => serde_json::json!(flag),
        proxima_pgwire::SqlValue::Text(text) => serde_json::json!(text),
        proxima_pgwire::SqlValue::Bytes(bytes) => serde_json::json!(bytes),
    }
}

/// Parse a [`PgClientConfig`] from the spec: prefer a `dsn` string, else
/// deserialize the field form (serde ignores the `type` discriminator).
fn config_from_spec(spec: &Value) -> Result<PgClientConfig, ProximaError> {
    if let Some(dsn) = spec.get("dsn").and_then(Value::as_str) {
        return PgClientConfig::from_dsn(dsn)
            .map_err(|err| ProximaError::Config(format!("pgwire dsn: {err}")));
    }
    serde_json::from_value(spec.clone())
        .map_err(|err| ProximaError::Config(format!("pgwire config: {err}")))
}

/// The out-of-crate [`ClientProtocol`] a `.pgwire(dsn)` builder call merges —
/// migrated OFF the old bespoke inherent `ClientBuilder::pgwire` onto the
/// same `.protocol()` mechanism every other protocol terminal uses, wrapping
/// this SAME [`PgwirePipeFactory`] (net-zero runtime change; see Section E
/// of the builder-sugar design).
pub struct PgwireClientProtocol {
    dsn: String,
}

impl PgwireClientProtocol {
    /// Point at a PostgreSQL server by DSN (`postgres://user:pw@host:port/db`).
    #[must_use]
    pub fn dsn(dsn: impl Into<String>) -> Self {
        Self { dsn: dsn.into() }
    }
}

impl ClientProtocol for PgwireClientProtocol {
    fn spec(&self) -> Value {
        serde_json::json!({"type": "pgwire", "dsn": self.dsn})
    }

    fn factory(&self) -> Arc<dyn PipeFactory> {
        Arc::new(PgwirePipeFactory::new())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::client::protocol::ClientProtocolExt;

    #[test]
    fn config_from_dsn_spec() {
        let spec = serde_json::json!({ "type": "pgwire", "dsn": "postgres://u:p@h:6543/db" });
        let config = config_from_spec(&spec).expect("config");
        assert_eq!((config.host.as_str(), config.port), ("h", 6543));
        assert_eq!(
            (config.user.as_str(), config.database.as_str()),
            ("u", "db")
        );
    }

    #[test]
    fn config_from_fields_ignores_type_discriminator() {
        let spec =
            serde_json::json!({ "type": "pgwire", "host": "db", "port": 5544, "user": "alice" });
        let config = config_from_spec(&spec).expect("config");
        assert_eq!(
            (config.host.as_str(), config.port, config.user.as_str()),
            ("db", 5544, "alice")
        );
        assert_eq!(
            config.database, "postgres",
            "unspecified field falls back to default"
        );
    }

    #[test]
    fn factory_name_is_the_spec_key() {
        assert_eq!(PgwirePipeFactory::new().name(), "pgwire");
    }

    #[test]
    fn client_protocol_lowers_to_the_type_and_dsn_spec() {
        let protocol = PgwireClientProtocol::dsn("postgres://u:p@h:5432/db");
        let spec = protocol.spec();
        assert_eq!(spec["type"], "pgwire");
        assert_eq!(spec["dsn"], "postgres://u:p@h:5432/db");
        assert_eq!(protocol.factory().name(), "pgwire");
    }

    /// The headline: pgwire reached through `proxima::Client` like any other
    /// protocol. `.pgwire(dsn)` lowers to the `type` terminal, `load()`
    /// resolves this factory. The SQL reply is typed (`PgReply`) inside the
    /// driver, but every `PipeHandle` shares one dynamic-dispatch shape
    /// (`Response<Bytes>` — see `Delay`/`Filter`), so `pg_response_to_bytes`
    /// JSON-encodes the reply rather than growing `Response` a new field;
    /// this decodes that JSON back. Off-worker: `Client` auto-dispatches onto
    /// the shared prime runtime (the prime TCP upstream needs a worker), so
    /// this needs the full runtime bundle, not just the bare `runtime-prime`
    /// marker feature — see `shared_prime_runtime`'s gate in
    /// `src/client/handle.rs`. Env-gated on a server.
    #[cfg(all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ))]
    #[test]
    fn pgwire_through_client_queries_real_postgres() {
        let Ok(host) = std::env::var("PGWIRE_REALPG_HOST") else {
            eprintln!("skipping pgwire_through_client: PGWIRE_REALPG_HOST unset (no server)");
            return;
        };
        let port = std::env::var("PGWIRE_REALPG_PORT").unwrap_or_else(|_| "5432".to_string());
        let password =
            std::env::var("PGWIRE_REALPG_PASSWORD").unwrap_or_else(|_| "proxima".to_string());
        let dsn = format!("postgres://postgres:{password}@{host}:{port}/postgres");

        let ok = futures::executor::block_on(async move {
            let client = crate::Client::builder()
                .pgwire(&dsn)
                .build()
                .expect("build client");
            let response = client
                .call("QUERY", "/")
                .body("select 1")
                .send()
                .await
                .expect("send select 1");
            let bytes = response.bytes().await.expect("select 1 body");
            let reply: Value = serde_json::from_slice(&bytes).expect("decode pgwire reply");
            assert_eq!(
                reply["kind"], "query",
                "expected a query reply, got {reply}"
            );
            assert_eq!(
                reply["columns"].as_array().expect("columns array").len(),
                1,
                "select 1 -> one column"
            );
            assert_eq!(
                reply["rows"].as_array().expect("rows array").len(),
                1,
                "one row"
            );
            true
        });
        assert!(ok);
    }
}
