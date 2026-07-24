#![allow(clippy::expect_used)]
//! Real-client smoke tests: psql (libpq) over the simple protocol —
//! trust, SSL-refusal + cleartext password, and TLS — plus
//! tokio-postgres over the extended protocol with binary parameters.
//! These are the discipline-log witnesses that the facade speaks actual
//! PostgreSQL, not just our own codec round-tripped.

use std::net::{Ipv4Addr, SocketAddr};
use std::process::Stdio;
use std::sync::Arc;

use proxima_core::ProximaError;
use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;
use proxima_pgwire::codec::Session;
use proxima_pgwire::{
    ColumnDesc, DescribeReply, ErrorReply, Negotiation, PgAuth, PgPipeHandle, PgReply,
    PgServerConfig, QueryReply, QueryRequest, RowStream, SqlValue, StaticCredentials,
    into_pg_handle, negotiate, serve_session,
};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::stream::{StreamConnection, StreamListener, StreamListenerExt};
use proxima_protocols::pgwire_codec::{CopyFormat, Oid};
use zeroize::Zeroizing;

const OID_INT4: Oid = Oid(23);

/// The SQL engine as a `Pipe`: QUERY "select 1" → one int column one row;
/// PARSE → a Describe declaring the param + result shape; EXECUTE → one
/// int column one row whose value is the bound parameter (round-tripped
/// from the typed `SqlValue` the driver decoded).
struct EchoPipe;

impl SendPipe for EchoPipe {
    type In = QueryRequest;
    type Out = PgReply;
    type Err = ProximaError;

    async fn call(&self, request: QueryRequest) -> Result<PgReply, ProximaError> {
        let reply = match request {
            QueryRequest::Query { sql, .. } => echo_query(&sql),
            QueryRequest::Parse { sql, .. } => PgReply::Describe(echo_describe(&sql)),
            QueryRequest::Execute { parameters, .. } => echo_execute(&parameters),
            other => {
                return Err(ProximaError::Config(format!(
                    "echo pipe received unexpected request {other:?}"
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

async fn serve_one<S>(stream: S, auth: PgAuth, tls: Option<futures_rustls::TlsAcceptor>)
where
    S: futures::io::AsyncRead + futures::io::AsyncWrite + Unpin + Send,
{
    serve_one_with(stream, auth, tls, into_pg_handle(EchoPipe)).await;
}

async fn serve_one_with<S>(
    stream: S,
    auth: PgAuth,
    tls: Option<futures_rustls::TlsAcceptor>,
    query: PgPipeHandle,
) where
    S: futures::io::AsyncRead + futures::io::AsyncWrite + Unpin + Send,
{
    let config = PgServerConfig::default();
    let mut session = Session::new();
    let negotiated = negotiate(stream, &mut session, tls.is_some())
        .await
        .expect("startup negotiation must succeed");
    match negotiated {
        Negotiation::Proceed {
            stream,
            startup,
            leftover,
        } => {
            serve_session(
                stream, session, startup, leftover, query, &auth, &config, None, None, None,
            )
            .await
            .expect("session must complete cleanly");
        }
        Negotiation::StartTls(stream) => {
            let acceptor = tls.expect("StartTls only offered when tls is configured");
            let tls_stream = acceptor
                .accept(stream)
                .await
                .expect("tls handshake must succeed");
            session
                .tls_established()
                .expect("tls_established after accept");
            match negotiate(tls_stream, &mut session, false)
                .await
                .expect("post-tls negotiation must succeed")
            {
                Negotiation::Proceed {
                    stream,
                    startup,
                    leftover,
                } => {
                    serve_session(
                        stream, session, startup, leftover, query, &auth, &config, None, None, None,
                    )
                    .await
                    .expect("tls session must complete cleanly");
                }
                _ => panic!("expected startup after tls handshake"),
            }
        }
        Negotiation::Closed => {}
        Negotiation::Cancel { .. } => {}
    }
}

async fn spawn_server(auth: PgAuth, tls: Option<futures_rustls::TlsAcceptor>) -> u16 {
    let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind loopback");
    let port = match listener.local_addr().expect("local addr") {
        proxima_primitives::stream::BindAddr::Tcp(addr) => addr.port(),
        other => panic!("expected tcp bind, got {other:?}"),
    };
    tokio::spawn(async move {
        loop {
            let Ok(conn) = listener.accept().await else {
                return;
            };
            let auth = auth.clone();
            let tls = tls.clone();
            tokio::spawn(async move {
                let _peer = conn.peer();
                serve_one(conn, auth, tls).await;
            });
        }
    });
    port
}

/// The COPY witness engine: `TO STDOUT` yields CopyOut with two text rows;
/// `FROM STDIN` yields CopyIn; the COPY_DATA second phase reports the
/// collected row count as the command tag (ROLLBACK when the client
/// aborted via CopyFail).
struct CopyPipe;

const COPY_OUT_WITNESS: [&[u8]; 2] = [b"1\talice\n", b"2\tbob\n"];

impl SendPipe for CopyPipe {
    type In = QueryRequest;
    type Out = PgReply;
    type Err = ProximaError;

    async fn call(&self, request: QueryRequest) -> Result<PgReply, ProximaError> {
        let sql = request.sql().to_owned();
        let reply = match request {
            QueryRequest::Query { .. } | QueryRequest::Execute { .. }
                if sql.contains("TO STDOUT") =>
            {
                PgReply::CopyOut {
                    format: CopyFormat::Text,
                    column_formats: vec![],
                    data: COPY_OUT_WITNESS.iter().map(|row| row.to_vec()).collect(),
                }
            }
            QueryRequest::Query { .. } | QueryRequest::Execute { .. }
                if sql.contains("FROM STDIN") =>
            {
                PgReply::CopyIn {
                    format: CopyFormat::Text,
                    column_formats: vec![],
                }
            }
            QueryRequest::CopyData {
                copy_failed,
                copy_data,
                ..
            } => {
                if copy_failed {
                    PgReply::Query(QueryReply::tag("ROLLBACK"))
                } else {
                    PgReply::Query(QueryReply::tag(format!("COPY {}", copy_data.len())))
                }
            }
            QueryRequest::Parse { .. } => PgReply::Describe(DescribeReply::default()),
            _ => {
                return Err(ProximaError::Config(format!(
                    "copy pipe unexpected request sql {sql:?}"
                )));
            }
        };
        Ok(reply)
    }
}

/// The G10 streaming witness engine: a `select` yields `PgReply::QueryStream`
/// whose rows are produced lazily by a spawned task feeding an
/// `async_channel` sender. The driver drains/encodes/flushes incrementally —
/// proving the streaming path round-trips through a real client. The driver
/// never collects the full set (bounded memory); correctness is what the
/// client asserts.
struct StreamPipe {
    count: i64,
}

impl SendPipe for StreamPipe {
    type In = QueryRequest;
    type Out = PgReply;
    type Err = ProximaError;

    async fn call(&self, request: QueryRequest) -> Result<PgReply, ProximaError> {
        let reply = match request {
            QueryRequest::Query { .. } | QueryRequest::Execute { .. } => {
                let (sender, receiver) = async_channel::bounded::<Vec<SqlValue>>(4);
                let count = self.count;
                tokio::spawn(async move {
                    for number in 1..=count {
                        if sender.send(vec![SqlValue::Int(number)]).await.is_err() {
                            break;
                        }
                    }
                });
                PgReply::QueryStream {
                    columns: vec![ColumnDesc::new("n", OID_INT4)],
                    rows: RowStream::new(receiver),
                    command_tag: None,
                }
            }
            QueryRequest::Parse { .. } => PgReply::Describe(DescribeReply {
                parameter_types: vec![],
                columns: vec![ColumnDesc::new("n", OID_INT4)],
            }),
            other => {
                return Err(ProximaError::Config(format!(
                    "stream pipe unexpected request {other:?}"
                )));
            }
        };
        Ok(reply)
    }
}

async fn spawn_stream_server(count: i64) -> u16 {
    let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind loopback");
    let port = match listener.local_addr().expect("local addr") {
        proxima_primitives::stream::BindAddr::Tcp(addr) => addr.port(),
        other => panic!("expected tcp bind, got {other:?}"),
    };
    tokio::spawn(async move {
        loop {
            let Ok(conn) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                serve_one_with(
                    conn,
                    PgAuth::Trust,
                    None,
                    into_pg_handle(StreamPipe { count }),
                )
                .await;
            });
        }
    });
    port
}

async fn spawn_copy_server() -> u16 {
    let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind loopback");
    let port = match listener.local_addr().expect("local addr") {
        proxima_primitives::stream::BindAddr::Tcp(addr) => addr.port(),
        other => panic!("expected tcp bind, got {other:?}"),
    };
    tokio::spawn(async move {
        loop {
            let Ok(conn) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                serve_one_with(conn, PgAuth::Trust, None, into_pg_handle(CopyPipe)).await;
            });
        }
    });
    port
}

/// The LISTEN/NOTIFY witness engine: maps `LISTEN <chan>` and
/// `NOTIFY <chan>, '<payload>'` simple-query SQL to the matching
/// [`PgReply`] variants the driver applies against the shared broker. The
/// engine never touches the broker — it only classifies the SQL.
struct NotifyPipe;

impl SendPipe for NotifyPipe {
    type In = QueryRequest;
    type Out = PgReply;
    type Err = ProximaError;

    async fn call(&self, request: QueryRequest) -> Result<PgReply, ProximaError> {
        let QueryRequest::Query { sql, .. } = &request else {
            return Err(ProximaError::Config(format!(
                "notify pipe unexpected request {request:?}"
            )));
        };
        let sql = sql.trim().to_string();
        let reply = if sql.to_ascii_uppercase().starts_with("LISTEN ") {
            PgReply::Listen {
                channels: vec![parse_listen_channel(&sql)],
            }
        } else if sql.to_ascii_uppercase().starts_with("NOTIFY ") {
            let (channel, payload) = parse_notify(&sql);
            PgReply::Notify { channel, payload }
        } else {
            PgReply::Query(QueryReply::tag("OK"))
        };
        Ok(reply)
    }
}

fn parse_listen_channel(sql: &str) -> String {
    sql.split_whitespace()
        .nth(1)
        .unwrap_or_default()
        .trim_matches(';')
        .to_string()
}

fn parse_notify(sql: &str) -> (String, String) {
    let rest = sql["NOTIFY".len()..].trim().trim_end_matches(';');
    match rest.split_once(',') {
        Some((channel, payload)) => (
            channel.trim().to_string(),
            payload.trim().trim_matches('\'').to_string(),
        ),
        None => (rest.trim().to_string(), String::new()),
    }
}

/// Spawns a server whose accepted connections all share ONE broker, so a
/// NOTIFY on one connection reaches a LISTEN on another. Returns the port.
async fn spawn_notify_server() -> u16 {
    use proxima_pgwire::NotifyBroker;

    let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind loopback");
    let port = match listener.local_addr().expect("local addr") {
        proxima_primitives::stream::BindAddr::Tcp(addr) => addr.port(),
        other => panic!("expected tcp bind, got {other:?}"),
    };
    let broker = Arc::new(NotifyBroker::new());
    tokio::spawn(async move {
        loop {
            let Ok(conn) = listener.accept().await else {
                return;
            };
            let broker = Arc::clone(&broker);
            tokio::spawn(async move {
                let mut session = Session::new();
                let negotiated = match negotiate(conn, &mut session, false).await {
                    Ok(negotiated) => negotiated,
                    Err(_) => return,
                };
                if let Negotiation::Proceed {
                    stream,
                    startup,
                    leftover,
                } = negotiated
                {
                    let _ = serve_session(
                        stream,
                        session,
                        startup,
                        leftover,
                        into_pg_handle(NotifyPipe),
                        &PgAuth::Trust,
                        &PgServerConfig::default(),
                        None,
                        Some(broker),
                        None,
                    )
                    .await;
                }
            });
        }
    });
    port
}

async fn run_psql(port: u16, sslmode: &str, password: Option<&str>) -> (String, String) {
    let conninfo = format!(
        "host=127.0.0.1 port={port} user=smoke dbname=smokedb sslmode={sslmode} connect_timeout=5"
    );
    let mut command = tokio::process::Command::new("psql");
    command
        .arg("-X")
        .arg("-t")
        .arg("-A")
        .arg("-c")
        .arg("select 1")
        .arg(&conninfo)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(password) = password {
        command.env("PGPASSWORD", password);
    }
    let output = command.output().await.expect("psql must spawn");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn psql_trust_completes_startup_and_select_1() {
    let port = spawn_server(PgAuth::Trust, None).await;
    let (stdout, stderr) = run_psql(port, "disable", None).await;
    assert_eq!(
        stdout.trim(),
        "1",
        "psql must print the single column, stderr: {stderr}"
    );
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn psql_ssl_refusal_then_cleartext_password_auth_succeeds() {
    let auth = PgAuth::Cleartext(Arc::new(StaticCredentials {
        username: "smoke".into(),
        password: Zeroizing::new("open sesame".into()),
    }));
    let port = spawn_server(auth, None).await;
    let (stdout, stderr) = run_psql(port, "prefer", Some("open sesame")).await;
    assert_eq!(
        stdout.trim(),
        "1",
        "ssl-refused cleartext path failed, stderr: {stderr}"
    );
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn psql_rejects_wrong_password_with_28p01() {
    let auth = PgAuth::Cleartext(Arc::new(StaticCredentials {
        username: "smoke".into(),
        password: Zeroizing::new("right".into()),
    }));
    let port = spawn_server(auth, None).await;
    let (stdout, stderr) = run_psql(port, "disable", Some("wrong")).await;
    assert!(
        stdout.trim().is_empty(),
        "no result expected on auth failure"
    );
    assert!(
        stderr.contains("password authentication failed"),
        "stderr must carry the auth failure: {stderr}"
    );
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn psql_over_tls_completes_select_1() {
    let tls_config = proxima_tls::TlsConfig::self_signed();
    let acceptor =
        proxima_tls::build_acceptor_futures_io(&tls_config).expect("self-signed acceptor");
    let port = spawn_server(PgAuth::Trust, Some(acceptor)).await;
    let (stdout, stderr) = run_psql(port, "require", None).await;
    assert_eq!(stdout.trim(), "1", "tls path failed, stderr: {stderr}");
}

fn static_creds(username: &str, password: &str) -> Arc<StaticCredentials> {
    Arc::new(StaticCredentials {
        username: username.into(),
        password: Zeroizing::new(password.into()),
    })
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_scram_auth_succeeds_with_right_password() {
    let port = spawn_server(PgAuth::Scram(static_creds("smoke", "pencil")), None).await;
    let conninfo = format!(
        "host=127.0.0.1 port={port} user=smoke password=pencil dbname=smokedb connect_timeout=5"
    );
    let (client, connection) = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls)
        .await
        .expect("tokio-postgres must connect over scram");
    let connection_task = tokio::spawn(connection);
    let rows = client
        .query("select 1", &[])
        .await
        .expect("query after scram auth");
    assert_eq!(rows.len(), 1);
    drop(client);
    let _ = connection_task.await;
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_scram_auth_fails_with_wrong_password() {
    let port = spawn_server(PgAuth::Scram(static_creds("smoke", "pencil")), None).await;
    let conninfo = format!(
        "host=127.0.0.1 port={port} user=smoke password=wrong dbname=smokedb connect_timeout=5"
    );
    let result = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls).await;
    assert!(
        result.is_err(),
        "scram with wrong password must fail to connect"
    );
}

/// Serves one connection with the SCRAM KDF offloaded onto a real
/// runtime's background-blocking pool — the offload path under test.
async fn serve_one_scram_offloaded<S>(
    stream: S,
    auth: PgAuth,
    runtime: Arc<dyn proxima_runtime::Runtime>,
) where
    S: futures::io::AsyncRead + futures::io::AsyncWrite + Unpin + Send,
{
    let config = PgServerConfig::default();
    let mut session = Session::new();
    let negotiated = negotiate(stream, &mut session, false)
        .await
        .expect("startup negotiation must succeed");
    if let Negotiation::Proceed {
        stream,
        startup,
        leftover,
    } = negotiated
    {
        let _ = serve_session(
            stream,
            session,
            startup,
            leftover,
            into_pg_handle(EchoPipe),
            &auth,
            &config,
            None,
            None,
            Some(runtime),
        )
        .await;
    }
}

/// Spawns a server whose SCRAM auth offloads PBKDF2 onto a
/// `TokioPerCoreRuntime` background-blocking pool, exercising the real
/// `spawn_background_blocking` path. Returns the bound port.
async fn spawn_scram_offloaded_server(auth: PgAuth) -> u16 {
    let runtime: Arc<dyn proxima_runtime::Runtime> =
        Arc::new(proxima_runtime::tokio::TokioPerCoreRuntime::new(1).expect("runtime"));
    let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind loopback");
    let port = match listener.local_addr().expect("local addr") {
        proxima_primitives::stream::BindAddr::Tcp(addr) => addr.port(),
        other => panic!("expected tcp bind, got {other:?}"),
    };
    tokio::spawn(async move {
        loop {
            let Ok(conn) = listener.accept().await else {
                return;
            };
            let auth = auth.clone();
            let runtime = Arc::clone(&runtime);
            tokio::spawn(async move {
                serve_one_scram_offloaded(conn, auth, runtime).await;
            });
        }
    });
    port
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn scram_auth_with_offloaded_kdf_succeeds_with_right_password() {
    let port = spawn_scram_offloaded_server(PgAuth::Scram(static_creds("smoke", "pencil"))).await;
    let conninfo = format!(
        "host=127.0.0.1 port={port} user=smoke password=pencil dbname=smokedb connect_timeout=5"
    );
    let (client, connection) = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls)
        .await
        .expect("scram with offloaded kdf must connect");
    let connection_task = tokio::spawn(connection);
    let rows = client
        .query("select 1", &[])
        .await
        .expect("query after offloaded scram auth");
    assert_eq!(rows.len(), 1);
    drop(client);
    let _ = connection_task.await;
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn scram_auth_with_offloaded_kdf_fails_with_wrong_password() {
    let port = spawn_scram_offloaded_server(PgAuth::Scram(static_creds("smoke", "pencil"))).await;
    let conninfo = format!(
        "host=127.0.0.1 port={port} user=smoke password=wrong dbname=smokedb connect_timeout=5"
    );
    let result = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls).await;
    assert!(
        result.is_err(),
        "offloaded scram with wrong password must fail to connect"
    );
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn psql_md5_auth_succeeds_with_right_password() {
    let port = spawn_server(PgAuth::Md5(static_creds("smoke", "letmein")), None).await;
    let (stdout, stderr) = run_psql(port, "disable", Some("letmein")).await;
    assert_eq!(
        stdout.trim(),
        "1",
        "md5 right-password path failed, stderr: {stderr}"
    );
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn psql_md5_auth_fails_with_wrong_password() {
    let port = spawn_server(PgAuth::Md5(static_creds("smoke", "letmein")), None).await;
    let (stdout, stderr) = run_psql(port, "disable", Some("nope")).await;
    assert!(
        stdout.trim().is_empty(),
        "no result expected on md5 auth failure"
    );
    assert!(
        stderr.contains("password authentication failed"),
        "md5 wrong-password stderr must carry the failure: {stderr}"
    );
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn psql_scram_auth_succeeds_with_right_password() {
    let port = spawn_server(PgAuth::Scram(static_creds("smoke", "s3cr3t")), None).await;
    let (stdout, stderr) = run_psql(port, "disable", Some("s3cr3t")).await;
    assert_eq!(
        stdout.trim(),
        "1",
        "scram right-password path failed, stderr: {stderr}"
    );
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn psql_scram_auth_fails_with_wrong_password() {
    let port = spawn_server(PgAuth::Scram(static_creds("smoke", "s3cr3t")), None).await;
    let (stdout, stderr) = run_psql(port, "disable", Some("bogus")).await;
    assert!(
        stdout.trim().is_empty(),
        "no result expected on scram auth failure"
    );
    assert!(
        stderr.contains("password authentication failed"),
        "scram wrong-password stderr must carry the failure: {stderr}"
    );
}

#[cfg(feature = "tls")]
mod direct_tls {
    //! PostgreSQL 17 `sslnegotiation=direct`: the client opens with a TLS
    //! ClientHello and no SSLRequest. psql 16's libpq cannot drive this, so
    //! the witness is a raw futures-rustls client handshaking immediately,
    //! then running startup + `select 1` over the tunnel. The server side is
    //! the real `PgWireConnectionPipe` upgrade path so G9's first-byte
    //! detection is exercised, not re-implemented.

    use std::net::{Ipv4Addr, SocketAddr};
    use std::sync::Arc;

    use bytes::Bytes;
    use futures::io::{AsyncReadExt, AsyncWriteExt};
    use proxima_net::tokio::tokio_stream_listener::TokioTcpListener;
    use proxima_pgwire::{
        CancelRegistry, PgAuth, PgServerConfig, PgWireConnectionPipe, into_pg_handle,
    };
    use proxima_primitives::pipe::SendPipe;
    use proxima_primitives::pipe::request::{Request, RequestContext};
    use proxima_primitives::pipe::upgrade::HijackedSocket;
    use proxima_primitives::stream::{StreamListener, StreamListenerExt, StreamUpstreamExt};
    use proxima_protocols::pgwire_codec::backend::{BackendMessage, parse_backend};
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::{ClientConfig, DigitallySignedStruct, SignatureScheme};

    use super::EchoPipe;

    #[derive(Debug)]
    struct AcceptAnyCert;

    impl ServerCertVerifier for AcceptAnyCert {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp_response: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, rustls::Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }

        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }

        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            vec![
                SignatureScheme::ECDSA_NISTP256_SHA256,
                SignatureScheme::ED25519,
                SignatureScheme::RSA_PSS_SHA256,
                SignatureScheme::RSA_PKCS1_SHA256,
            ]
        }
    }

    fn no_verify_client_config() -> ClientConfig {
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .expect("client protocol versions")
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(AcceptAnyCert))
            .with_no_client_auth()
    }

    fn startup_bytes(user: &str) -> Vec<u8> {
        let version_code: i32 = 196_608;
        let mut params = Vec::new();
        params.extend_from_slice(b"user\0");
        params.extend_from_slice(user.as_bytes());
        params.push(0);
        params.push(0);
        let total_len = (4 + 4 + params.len()) as i32;
        let mut buf = Vec::new();
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(&version_code.to_be_bytes());
        buf.extend_from_slice(&params);
        buf
    }

    fn query_bytes(sql: &str) -> Vec<u8> {
        let total_len = (4 + sql.len() + 1) as i32;
        let mut buf = Vec::new();
        buf.push(b'Q');
        buf.extend_from_slice(&total_len.to_be_bytes());
        buf.extend_from_slice(sql.as_bytes());
        buf.push(0);
        buf
    }

    fn ready_for_query_count(buf: &[u8]) -> usize {
        let mut offset = 0;
        let mut count = 0;
        while let Ok(Some((message, consumed))) = parse_backend(&buf[offset..]) {
            if consumed == 0 {
                break;
            }
            if matches!(message, BackendMessage::ReadyForQuery { .. }) {
                count += 1;
            }
            offset += consumed;
        }
        count
    }

    async fn spawn_direct_tls_server(acceptor: futures_rustls::TlsAcceptor) -> u16 {
        let listener = TokioTcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .await
            .expect("bind loopback");
        let port = match listener.local_addr().expect("local addr") {
            proxima_primitives::stream::BindAddr::Tcp(addr) => addr.port(),
            other => panic!("expected tcp bind, got {other:?}"),
        };
        let pipe = PgWireConnectionPipe::new(
            "pg-direct-tls",
            into_pg_handle(EchoPipe),
            PgAuth::Trust,
            Arc::new(PgServerConfig::default()),
            Arc::new(CancelRegistry::new()),
        )
        .with_tls(Some(acceptor));
        tokio::spawn(async move {
            let Ok(conn) = listener.accept().await else {
                return;
            };
            let request = Request {
                method: proxima_primitives::pipe::method::Method::from_bytes(
                    proxima_pgwire::verb::CONNECT,
                ),
                path: Bytes::new(),
                query: proxima_primitives::pipe::header_list::HeaderList::new(),
                metadata: proxima_primitives::pipe::header_list::HeaderList::new(),
                payload: Bytes::new(),
                stream: None,
                context: RequestContext::default(),
            };
            let response = pipe
                .call(request)
                .await
                .expect("connect must answer with upgrade");
            let handler = response.upgrade.expect("upgrade must be present");
            let hijacked = HijackedSocket::new(Box::new(conn), Bytes::new());
            let _ = handler.invoke(hijacked).await;
        });
        port
    }

    #[proxima::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_tls_no_ssl_request_runs_startup_and_select_1() {
        let tls_config = proxima_tls::TlsConfig::self_signed();
        let acceptor =
            proxima_tls::build_acceptor_futures_io(&tls_config).expect("self-signed acceptor");
        let port = spawn_direct_tls_server(acceptor).await;

        let connector = futures_rustls::TlsConnector::from(Arc::new(no_verify_client_config()));
        let upstream = proxima_net::tokio::TokioTcpUpstream::new(SocketAddr::from((
            Ipv4Addr::LOCALHOST,
            port,
        )));
        let stream = upstream.connect().await.expect("client tcp connect");
        let server_name = ServerName::try_from("localhost").expect("server name");
        let mut tls = connector
            .connect(server_name, stream)
            .await
            .expect("direct tls handshake must succeed");

        tls.write_all(&startup_bytes("smoke"))
            .await
            .expect("startup write");
        tls.write_all(&query_bytes("select 1"))
            .await
            .expect("query write");
        tls.flush().await.expect("flush");

        let mut collected = Vec::new();
        let mut scratch = [0_u8; 1024];
        while ready_for_query_count(&collected) < 2 {
            let read =
                tokio::time::timeout(std::time::Duration::from_secs(5), tls.read(&mut scratch))
                    .await
                    .expect("server response must arrive before the guard fires")
                    .expect("server read");
            if read == 0 {
                break;
            }
            collected.extend_from_slice(&scratch[..read]);
        }

        let mut data_value = None;
        let mut offset = 0;
        while let Ok(Some((message, consumed))) = parse_backend(&collected[offset..]) {
            if let BackendMessage::DataRow { columns } = message {
                let cells: Vec<Option<&[u8]>> = columns.iter().collect();
                data_value = cells.first().copied().flatten().map(<[u8]>::to_vec);
            }
            offset += consumed;
            if consumed == 0 {
                break;
            }
        }
        assert_eq!(
            data_value.as_deref(),
            Some(b"1".as_slice()),
            "direct-tls select 1 must stream a data row encoding 1"
        );
    }
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_extended_query_round_trips_binary_param() {
    let port = spawn_server(PgAuth::Trust, None).await;
    let conninfo =
        format!("host=127.0.0.1 port={port} user=smoke dbname=smokedb connect_timeout=5");
    let (client, connection) = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls)
        .await
        .expect("tokio-postgres must connect");
    let connection_task = tokio::spawn(connection);
    let rows = client
        .query("select $1::int4 as v", &[&7_i32])
        .await
        .expect("extended query must succeed");
    assert_eq!(rows.len(), 1);
    let value: i32 = rows[0].get("v");
    assert_eq!(
        value, 7,
        "bound parameter must round-trip through Parse/Bind/Execute"
    );
    drop(client);
    let _ = connection_task.await;
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_simple_query_drains_a_streamed_row_source_in_order() {
    let port = spawn_stream_server(5).await;
    let conninfo =
        format!("host=127.0.0.1 port={port} user=smoke dbname=smokedb connect_timeout=5");
    let (client, connection) = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls)
        .await
        .expect("tokio-postgres must connect");
    let connection_task = tokio::spawn(connection);

    let rows = client
        .query("select streamed", &[])
        .await
        .expect("streamed query must succeed");
    let values: Vec<i32> = rows.iter().map(|row| row.get::<_, i32>("n")).collect();
    assert_eq!(
        values,
        vec![1, 2, 3, 4, 5],
        "all streamed rows must arrive in order"
    );

    drop(client);
    let _ = connection_task.await;
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_extended_query_drains_a_streamed_row_source() {
    let port = spawn_stream_server(3).await;
    let conninfo =
        format!("host=127.0.0.1 port={port} user=smoke dbname=smokedb connect_timeout=5");
    let (client, connection) = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls)
        .await
        .expect("tokio-postgres must connect");
    let connection_task = tokio::spawn(connection);

    let rows = client
        .query("select streamed", &[])
        .await
        .expect("extended streamed query");
    let values: Vec<i32> = rows.iter().map(|row| row.get::<_, i32>("n")).collect();
    assert_eq!(
        values,
        vec![1, 2, 3],
        "the extended path must drain the full stream"
    );

    drop(client);
    let _ = connection_task.await;
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_copy_out_streams_the_engines_rows() {
    use futures::TryStreamExt;

    let port = spawn_copy_server().await;
    let conninfo =
        format!("host=127.0.0.1 port={port} user=smoke dbname=smokedb connect_timeout=5");
    let (client, connection) = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls)
        .await
        .expect("tokio-postgres must connect");
    let connection_task = tokio::spawn(connection);

    let stream = client
        .copy_out("COPY t TO STDOUT")
        .await
        .expect("copy_out must start");
    let chunks: Vec<bytes::Bytes> = stream
        .try_collect()
        .await
        .expect("copy_out stream must complete");
    let collected: Vec<u8> = chunks.into_iter().flatten().collect();

    let expected: Vec<u8> = COPY_OUT_WITNESS
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    assert_eq!(
        collected, expected,
        "copy_out must stream exactly the engine's rows"
    );

    drop(client);
    let _ = connection_task.await;
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_copy_in_reports_the_collected_row_count() {
    use futures::SinkExt;

    let port = spawn_copy_server().await;
    let conninfo =
        format!("host=127.0.0.1 port={port} user=smoke dbname=smokedb connect_timeout=5");
    let (client, connection) = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls)
        .await
        .expect("tokio-postgres must connect");
    let connection_task = tokio::spawn(connection);

    let sink = client
        .copy_in("COPY t FROM STDIN")
        .await
        .expect("copy_in must start");
    futures::pin_mut!(sink);
    sink.send(bytes::Bytes::from_static(b"1\tx\n"))
        .await
        .expect("first row");
    sink.send(bytes::Bytes::from_static(b"2\ty\n"))
        .await
        .expect("second row");
    sink.send(bytes::Bytes::from_static(b"3\tz\n"))
        .await
        .expect("third row");
    let rows = sink.finish().await.expect("copy_in must finish");
    assert_eq!(
        rows, 3,
        "the CommandComplete tag count must match the rows sent"
    );

    drop(client);
    let _ = connection_task.await;
}

/// The real-client cross-connection LISTEN/NOTIFY witness: two tokio-postgres
/// clients share one server broker. Client A `LISTEN test`; client B
/// `NOTIFY test, 'payload'`; A surfaces the async notification through its
/// connection's message stream with channel/payload intact.
#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn tokio_postgres_notify_reaches_a_listener_on_another_connection() {
    use std::time::Duration;

    use futures::StreamExt;

    let port = spawn_notify_server().await;
    let conninfo =
        format!("host=127.0.0.1 port={port} user=smoke dbname=smokedb connect_timeout=5");

    let (client_a, mut connection_a) = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls)
        .await
        .expect("listener must connect");
    let (notified_tx, notified_rx) = tokio::sync::oneshot::channel();
    let listener_task = tokio::spawn(async move {
        let mut messages =
            futures::stream::poll_fn(move |context| connection_a.poll_message(context));
        let mut notified_tx = Some(notified_tx);
        while let Some(message) = messages.next().await {
            if let Ok(tokio_postgres::AsyncMessage::Notification(notification)) = message
                && let Some(sender) = notified_tx.take()
            {
                let _ = sender.send((
                    notification.channel().to_string(),
                    notification.payload().to_string(),
                ));
            }
        }
    });

    client_a
        .batch_execute("LISTEN test")
        .await
        .expect("listen must succeed");

    let (client_b, connection_b) = tokio_postgres::connect(&conninfo, tokio_postgres::NoTls)
        .await
        .expect("notifier must connect");
    let connection_b_task = tokio::spawn(connection_b);
    client_b
        .batch_execute("NOTIFY test, 'payload'")
        .await
        .expect("notify must succeed");

    let received = tokio::time::timeout(Duration::from_secs(5), notified_rx)
        .await
        .expect("notification must arrive within the timeout")
        .expect("notification channel must not drop");
    assert_eq!(
        received.0, "test",
        "channel must round-trip to the real client"
    );
    assert_eq!(
        received.1, "payload",
        "payload must round-trip to the real client"
    );

    drop(client_a);
    drop(client_b);
    listener_task.abort();
    let _ = connection_b_task.await;
}
