//! Async PostgreSQL client `Pipe` over any [`StreamUpstream`] — the same
//! transport seam `H1ClientUpstream` uses, so the client is agnostic to the
//! wire (prime, tokio, TLS-wrapped). It drives the sans-IO [`ClientSession`]
//! over a futures-io connection and maps the SQL-over-Pipe contract
//! ([`QueryRequest`]'s `verb` field — `Query`/`Parse`/`Execute`) to/from
//! [`PgReply`] — no `Request`/`Response` envelope (payload-no-cell). A
//! registered `PipeFactory` (see `crate::client::factory`) builds this, so
//! `proxima::Client` speaks pgwire as just another protocol.

use std::future::Future;
use std::sync::Arc;

use futures::io::{AsyncReadExt, AsyncWriteExt};
use futures::lock::Mutex;

use proxima_core::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::stream::{StreamConnection, StreamUpstream, StreamUpstreamExt};
use proxima_protocols::pgwire_codec::Oid;

use crate::client::config::PgClientConfig;
use crate::client::session::{ClientError, ClientSession, QueryResult, Step};
use crate::pipe_contract::{
    ColumnDesc, ErrorReply, PgReply, QueryReply, QueryRequest, SqlValue, Verb,
};

const READ_CHUNK_BYTES: usize = 16 * 1024;

/// PostgreSQL client `Pipe` over a `StreamUpstream`. One client owns one
/// upstream binding (host:port) and one cached authenticated connection
/// (pool of one) reused across calls — PostgreSQL auth (SCRAM PBKDF2) is
/// expensive, so keep-alive matters more here than for HTTP.
pub struct PgwireClientUpstream<U: StreamUpstream> {
    upstream: Arc<U>,
    config: PgClientConfig,
    cached: Arc<Mutex<Option<Cached<U::Conn>>>>,
}

struct Cached<C> {
    conn: C,
    session: ClientSession,
}

impl<U: StreamUpstream> PgwireClientUpstream<U> {
    /// Builds a client over `upstream` with `config` (connection params). The
    /// transport is injected (runtime object); the config is the declarative
    /// half — the same split as `H1ClientUpstream::from_config`.
    pub fn new(upstream: U, config: PgClientConfig) -> Self {
        Self {
            upstream: Arc::new(upstream),
            config,
            cached: Arc::new(Mutex::new(None)),
        }
    }

    async fn exchange(&self, request: QueryRequest) -> Result<PgReply, ProximaError> {
        let mut guard = self.cached.lock().await;
        if guard.is_none() {
            *guard = Some(self.connect().await?);
        }
        let cached = guard
            .as_mut()
            .ok_or_else(|| ProximaError::Upstream("pgwire cache empty".into()))?;

        let outcome = run_request(&mut cached.session, &mut cached.conn, request).await;
        match outcome {
            Ok(reply) => Ok(reply),
            Err(ClientError::Server { sqlstate, message }) => {
                Ok(PgReply::Error(ErrorReply::new(sqlstate, message)))
            }
            Err(error) => {
                *guard = None;
                Err(client_error_to_proxima(error))
            }
        }
    }

    async fn connect(&self) -> Result<Cached<U::Conn>, ProximaError> {
        let conn = self
            .upstream
            .connect()
            .await
            .map_err(|err| ProximaError::Upstream(format!("pgwire connect: {err}")))?;
        let mut session = ClientSession::new(
            &self.config.user,
            &self.config.password,
            &self.config.database,
        )
        .map_err(client_error_to_proxima)?;
        let mut conn = conn;
        drive_until_ready(&mut session, &mut conn)
            .await
            .map_err(client_error_to_proxima)?;
        Ok(Cached { conn, session })
    }
}

impl<U: StreamUpstream> SendPipe for PgwireClientUpstream<U> {
    type In = QueryRequest;
    type Out = PgReply;
    type Err = ProximaError;

    fn call(
        &self,
        request: QueryRequest,
    ) -> impl Future<Output = Result<PgReply, ProximaError>> + Send {
        async move { self.exchange(request).await }
    }
}

async fn run_request<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
    request: QueryRequest,
) -> Result<PgReply, ClientError> {
    let QueryRequest { sql, verb, .. } = request;
    let result = match verb {
        Verb::Query => {
            session.submit_simple(&sql)?;
            run_query(session, conn).await?
        }
        Verb::Parse { .. } => {
            session.submit_extended(&sql, &[])?;
            run_query(session, conn).await?
        }
        Verb::Execute { parameters, .. } => {
            let params = parameters
                .iter()
                .map(sql_value_to_text)
                .collect::<Result<Vec<_>, _>>()?;
            let borrowed = params.iter().map(String::as_str).collect::<Vec<_>>();
            session.submit_extended(&sql, &borrowed)?;
            run_query(session, conn).await?
        }
        Verb::CopyData { .. } => {
            return Err(ClientError::Protocol("unsupported client verb"));
        }
    };
    Ok(query_result_to_reply(result))
}

async fn drive_until_ready<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<(), ClientError> {
    loop {
        match session.advance()? {
            Step::Send => flush(session, conn).await?,
            Step::Recv => recv(session, conn).await?,
            Step::Ready => return Ok(()),
            Step::Complete(_) => return Err(ClientError::Protocol("reply before ready")),
        }
    }
}

async fn run_query<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<QueryResult, ClientError> {
    loop {
        match session.advance()? {
            Step::Send => flush(session, conn).await?,
            Step::Recv => recv(session, conn).await?,
            Step::Complete(result) => return Ok(result),
            Step::Ready => return Err(ClientError::Protocol("ready without a reply")),
        }
    }
}

async fn flush<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<(), ClientError> {
    let bytes = session.take_outbound();
    conn.write_all(&bytes).await?;
    conn.flush().await?;
    Ok(())
}

async fn recv<C: StreamConnection>(
    session: &mut ClientSession,
    conn: &mut C,
) -> Result<(), ClientError> {
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    let read = conn.read(&mut chunk).await?;
    if read == 0 {
        return Err(ClientError::Closed);
    }
    session.feed(&chunk[..read]);
    Ok(())
}

fn sql_value_to_text(value: &SqlValue) -> Result<String, ClientError> {
    match value {
        SqlValue::Int(number) => Ok(number.to_string()),
        SqlValue::Text(text) => Ok(text.clone()),
        SqlValue::Null => Err(ClientError::Protocol(
            "null bind parameter not supported yet",
        )),
        _ => Err(ClientError::Protocol("unsupported bind parameter type")),
    }
}

fn query_result_to_reply(result: QueryResult) -> PgReply {
    let columns = result
        .columns
        .iter()
        .map(|column| ColumnDesc::new(column.name.clone(), Oid(column.type_oid)))
        .collect::<Vec<_>>();
    let rows = result
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| match cell {
                    Some(bytes) => SqlValue::Text(String::from_utf8_lossy(bytes).into_owned()),
                    None => SqlValue::Null,
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    PgReply::Query(QueryReply::rows(columns, rows))
}

fn client_error_to_proxima(error: ClientError) -> ProximaError {
    ProximaError::Upstream(format!("pgwire client: {error}"))
}
