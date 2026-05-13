//! `proxima pipeline …` CLI surface — submits / inspects / resolves /
//! lists pipelines against a running `proximad`. Two transports today:
//! local UDS (`--socket /path/to/proximad.sock`) and SSH-stdio
//! (`--host host-b` which spawns `ssh host-b proximad serve
//! --stdio` and frames HTTP/1.1 across the pipe).

use std::path::PathBuf;
use std::process::Stdio;

use clap::{Args, Subcommand};
use proxima::ProximaError;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::process::{ChildStdin, ChildStdout, Command};

#[derive(Args, Debug)]
pub struct PipelineArgs {
    /// Path to a local proximad's UDS. Mutually exclusive with --host.
    #[arg(long, value_name = "PATH", env = "PROXIMAD_SOCKET")]
    socket: Option<PathBuf>,

    /// SSH host running proximad. The CLI spawns
    /// `ssh <host> proximad serve --stdio` and frames HTTP/1.1 over
    /// the pipe. Mutually exclusive with --socket.
    #[arg(long, value_name = "HOST")]
    host: Option<String>,

    #[command(subcommand)]
    action: PipelineAction,
}

#[derive(Subcommand, Debug)]
pub enum PipelineAction {
    /// Submit a pipeline spec (TOML or JSON). Returns the freshly
    /// allocated pipeline id.
    Submit {
        /// Path to a pipeline spec file. `.toml` → TOML, anything else → JSON.
        spec: PathBuf,
    },
    /// List submitted pipelines, newest first.
    List {
        /// Filter by submitted pipeline name.
        #[arg(long)]
        name: Option<String>,
        /// Filter by spec_hash_hex (full hash).
        #[arg(long)]
        spec_hash_hex: Option<String>,
    },
    /// Resolve a query string to a canonical pipeline id. Tries:
    /// exact ULID parse → exact `--name` match → unique id/name prefix.
    Resolve {
        /// Query string (id, name, or prefix thereof).
        query: String,
    },
    /// Inspect a pipeline by query (id / name / prefix).
    Inspect { query: String },
    /// Stream this pipeline's events as they arrive, one event per
    /// line (NDJSON). Closes when the pipeline reaches a terminal
    /// state.
    Tail { query: String },
    /// Stream every event for every pipeline on this proximad. Runs
    /// until Ctrl-C / SSH-pipe close.
    Events,
    /// Walk a stage's `depends_on` chain. Renders the queried stage
    /// followed by each ancestor as an indented arrow tree.
    Explain {
        /// Pipeline query (id / name / prefix).
        query: String,
        /// Stage name to start the walk from.
        #[arg(long)]
        stage: String,
    },
    /// Replay a recorded pipeline under a fresh id, optionally
    /// substituting selected stages with new spec files. Non-substituted
    /// stages replay their recorded events; substituted stages run live.
    Replay {
        /// Pipeline query (id / name / prefix).
        query: String,
        /// One or more `stage_name=path/to/stage.toml` substitutions.
        /// The path's contents are parsed as a StageSpec (TOML or JSON
        /// by extension) and replace the original stage entirely.
        #[arg(long, value_name = "STAGE=PATH")]
        substitute: Vec<String>,
    },
    /// Download a file from a stage's workspace on the daemon host.
    Artifact {
        /// Pipeline query (id / name / prefix).
        query: String,
        /// Stage name whose workspace contains the file.
        #[arg(long)]
        stage: String,
        /// Path within the stage's workspace (relative). `../` is
        /// rejected after canonicalization on the daemon side.
        #[arg(long)]
        path: String,
        /// Where to write the downloaded file. `-` writes to stdout.
        #[arg(long, default_value = "-")]
        output: PathBuf,
    },
}

pub async fn run(args: PipelineArgs) -> Result<(), ProximaError> {
    let transport = build_transport(&args)?;
    match args.action {
        PipelineAction::Submit { spec } => submit(transport, spec).await,
        PipelineAction::List {
            name,
            spec_hash_hex,
        } => list(transport, name, spec_hash_hex).await,
        PipelineAction::Resolve { query } => resolve(transport, &query).await,
        PipelineAction::Inspect { query } => inspect(transport, &query).await,
        PipelineAction::Tail { query } => tail(transport, &query).await,
        PipelineAction::Events => events(transport).await,
        PipelineAction::Explain { query, stage } => explain(transport, &query, &stage).await,
        PipelineAction::Replay { query, substitute } => replay(transport, &query, substitute).await,
        PipelineAction::Artifact {
            query,
            stage,
            path,
            output,
        } => artifact(transport, &query, &stage, &path, output).await,
    }
}

fn build_transport(args: &PipelineArgs) -> Result<PipelineTransport, ProximaError> {
    match (&args.socket, &args.host) {
        (Some(_), Some(_)) => Err(ProximaError::Config(
            "--socket and --host are mutually exclusive".into(),
        )),
        (Some(path), None) => Ok(PipelineTransport::LocalUds(path.clone())),
        (None, Some(host)) => Ok(PipelineTransport::SshStdio(host.clone())),
        (None, None) => Err(ProximaError::Config(
            "specify --socket <path> or --host <ssh-host>".into(),
        )),
    }
}

#[derive(Debug, Clone)]
pub enum PipelineTransport {
    LocalUds(PathBuf),
    SshStdio(String),
}

impl PipelineTransport {
    /// Open a fresh duplex connection. UDS opens a `UnixStream`;
    /// SSH-stdio spawns `ssh <host> proximad serve --stdio` and returns
    /// its piped stdin+stdout (the child is owned by the returned
    /// connection so the SSH process gets reaped on drop).
    pub async fn connect(&self) -> Result<Connection, ProximaError> {
        match self {
            PipelineTransport::LocalUds(path) => {
                let stream = UnixStream::connect(path).await.map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!(
                        "connect proximad at {path:?}: {err}"
                    )))
                })?;
                Ok(Connection::Uds(stream))
            }
            PipelineTransport::SshStdio(host) => {
                let mut child = Command::new("ssh")
                    .arg(host)
                    .arg("proximad")
                    .arg("serve")
                    .arg("--stdio")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .map_err(|err| {
                        ProximaError::Io(std::io::Error::other(format!("spawn ssh {host}: {err}")))
                    })?;
                let stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| ProximaError::Io(std::io::Error::other("ssh stdin")))?;
                let stdout = child
                    .stdout
                    .take()
                    .ok_or_else(|| ProximaError::Io(std::io::Error::other("ssh stdout")))?;
                Ok(Connection::Ssh {
                    child,
                    stdin,
                    stdout,
                })
            }
        }
    }
}

pub enum Connection {
    Uds(UnixStream),
    Ssh {
        child: tokio::process::Child,
        stdin: ChildStdin,
        stdout: ChildStdout,
    },
}

impl Connection {
    /// Send a fully-formed HTTP/1.1 request and read the full response
    /// to EOF or Content-Length. Streaming responses (chunked tail) get
    /// their own helper in G6 — this one is for unary calls.
    pub async fn http_call(
        mut self,
        method: &str,
        path: &str,
        body: Option<(&str, Vec<u8>)>,
    ) -> Result<HttpResponse, ProximaError> {
        match &mut self {
            Connection::Uds(stream) => http_call_io(stream, method, path, body).await,
            Connection::Ssh {
                stdin,
                stdout,
                child,
                ..
            } => {
                let result = http_call_split(stdin, stdout, method, path, body).await;
                // when the unary call is done, close stdin so the proximad child
                // sees EOF and exits cleanly; the ssh process exits when its
                // remote command exits.
                let _ = child;
                result
            }
        }
    }

    /// Stream an HTTP/1.1 chunked response line-by-line. Used by
    /// `pipeline tail` / `pipeline events` — server emits NDJSON
    /// inside chunked transfer-encoding, this iterator parses both
    /// layers and yields one JSON-event line per call to `recv()`.
    ///
    /// Returns `None` when the chunked terminator (`0\r\n`) arrives or
    /// the transport closes.
    pub async fn stream_chunked_ndjson(
        self,
        method: &str,
        path: &str,
    ) -> Result<NdjsonStream, ProximaError> {
        match self {
            Connection::Uds(mut stream) => {
                write_request(&mut stream, method, path, None).await?;
                let head = read_response_head(&mut stream).await?;
                let framing = framing_after_leftover(&head);
                Ok(NdjsonStream::new_uds(stream, head.leftover, framing))
            }
            Connection::Ssh {
                mut stdin,
                mut stdout,
                child,
            } => {
                write_request(&mut stdin, method, path, None).await?;
                stdin.flush().await.map_err(|err| {
                    ProximaError::Io(std::io::Error::other(format!("flush ssh stdin: {err}")))
                })?;
                let head = read_response_head(&mut stdout).await?;
                let framing = framing_after_leftover(&head);
                Ok(NdjsonStream::new_ssh(stdout, child, head.leftover, framing))
            }
        }
    }
}

struct ResponseHead {
    /// Bytes already consumed from the body after the head terminator,
    /// kept so the body decoder doesn't lose them.
    leftover: Vec<u8>,
    chunked: bool,
    content_length: Option<usize>,
}

/// How the tail/events body is framed on the wire. The server sends
/// chunked for a still-open stream and content-length for a stream that
/// was already complete-and-ready at request time (h1 lazy-seal collapses
/// a finite ready body into one write) — RFC 9112 requires the client to
/// accept both.
enum Framing {
    Chunked,
    /// Content-length body with this many payload bytes still to read.
    Sized(usize),
    /// No framing header: read raw until the transport closes.
    ToEof,
}

fn framing_after_leftover(head: &ResponseHead) -> Framing {
    if head.chunked {
        Framing::Chunked
    } else if let Some(length) = head.content_length {
        Framing::Sized(length.saturating_sub(head.leftover.len()))
    } else {
        Framing::ToEof
    }
}

async fn read_response_head<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<ResponseHead, ProximaError> {
    // read until we see \r\n\r\n; cap at 32 KiB head size as a safety net.
    let mut accumulated: Vec<u8> = Vec::with_capacity(2048);
    let mut buf = [0_u8; 1024];
    let limit = 32 * 1024;
    let terminator_at;
    loop {
        let read = reader.read(&mut buf).await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("read response head: {err}")))
        })?;
        if read == 0 {
            return Err(ProximaError::Decode(
                "response head EOF before terminator".into(),
            ));
        }
        accumulated.extend_from_slice(&buf[..read]);
        if let Some(at) = accumulated
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
        {
            terminator_at = at;
            break;
        }
        if accumulated.len() > limit {
            return Err(ProximaError::Decode(
                "response head exceeded 32KiB without terminator".into(),
            ));
        }
    }
    let head_text = std::str::from_utf8(&accumulated[..terminator_at])
        .map_err(|err| ProximaError::Decode(format!("response head not utf-8: {err}")))?;
    let mut lines = head_text.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| ProximaError::Decode("empty status line".into()))?;
    let status = status_line
        .split(' ')
        .nth(1)
        .and_then(|raw| raw.parse::<u16>().ok())
        .ok_or_else(|| ProximaError::Decode(format!("malformed status: {status_line:?}")))?;
    if status != 200 {
        return Err(ProximaError::Upstream(format!(
            "expected 200, got {status}"
        )));
    }
    let mut chunked = false;
    let mut content_length: Option<usize> = None;
    for header_line in lines {
        let Some((name, value)) = header_line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("transfer-encoding")
            && value.trim().eq_ignore_ascii_case("chunked")
        {
            chunked = true;
        } else if name.trim().eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse::<usize>().ok();
        }
    }
    let leftover = accumulated[terminator_at + 4..].to_vec();
    Ok(ResponseHead {
        leftover,
        chunked,
        content_length,
    })
}

/// Incremental NDJSON-inside-chunked reader. Each `recv` returns one
/// JSON event line (without trailing newline), or `None` at end of
/// stream. Holds whichever transport bytes are flowing.
pub struct NdjsonStream {
    transport: NdjsonTransport,
    /// Bytes still unparsed from previous reads.
    buffer: Vec<u8>,
    /// How the wire body is framed — decides how each read is decoded.
    framing: Framing,
    /// True once the body's end has been observed (chunk-size 0,
    /// content-length exhausted, or transport EOF).
    terminated: bool,
}

enum NdjsonTransport {
    Uds(UnixStream),
    Ssh {
        stdout: ChildStdout,
        _child: tokio::process::Child,
    },
}

impl NdjsonStream {
    fn new_uds(stream: UnixStream, leftover: Vec<u8>, framing: Framing) -> Self {
        let terminated = matches!(framing, Framing::Sized(0));
        Self {
            transport: NdjsonTransport::Uds(stream),
            buffer: leftover,
            framing,
            terminated,
        }
    }

    fn new_ssh(
        stdout: ChildStdout,
        child: tokio::process::Child,
        leftover: Vec<u8>,
        framing: Framing,
    ) -> Self {
        let terminated = matches!(framing, Framing::Sized(0));
        Self {
            transport: NdjsonTransport::Ssh {
                stdout,
                _child: child,
            },
            buffer: leftover,
            framing,
            terminated,
        }
    }

    /// Return the next decoded line (one JSON event), or `None` when
    /// the chunked terminator has been seen and no buffered lines
    /// remain.
    pub async fn recv(&mut self) -> Result<Option<Vec<u8>>, ProximaError> {
        loop {
            // try to produce one line from what's already buffered
            if let Some(line) = try_extract_line(&mut self.buffer) {
                return Ok(Some(line));
            }
            if self.terminated {
                return Ok(None);
            }
            // pull more bytes from transport
            let mut chunk = [0_u8; 4096];
            let read = match &mut self.transport {
                NdjsonTransport::Uds(stream) => stream.read(&mut chunk).await,
                NdjsonTransport::Ssh { stdout, .. } => stdout.read(&mut chunk).await,
            }
            .map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!("read tail body: {err}")))
            })?;
            if read == 0 {
                self.terminated = true;
                // any trailing non-newline-terminated line is dropped; tail/events
                // server-side always terminates each event with `\n`, so this only
                // catches mid-line EOF.
                return Ok(None);
            }
            match &mut self.framing {
                Framing::Chunked => {
                    self.buffer.extend_from_slice(
                        decode_chunked_incremental(&mut self.terminated, &chunk[..read])?
                            .as_slice(),
                    );
                }
                Framing::Sized(remaining) => {
                    let take = (*remaining).min(read);
                    self.buffer.extend_from_slice(&chunk[..take]);
                    *remaining -= take;
                    if *remaining == 0 {
                        self.terminated = true;
                    }
                }
                Framing::ToEof => self.buffer.extend_from_slice(&chunk[..read]),
            }
        }
    }
}

/// Strip chunk-size lines from `chunk` and return only the payload
/// bytes. Sets `terminated = true` when a size-0 chunk is observed.
///
/// Simplification: assumes each `chunk` argument contains complete
/// chunk-size lines (i.e., we don't split a chunk-size across reads).
/// The proximad server emits one JSON event per chunk and a small head
/// (`<size>\r\n…\r\n`), so for the demo each read fits one logical
/// chunk. A robust streaming chunked decoder would keep parser state
/// across reads; we punt on that until it bites us.
fn decode_chunked_incremental(
    terminated: &mut bool,
    bytes: &[u8],
) -> Result<Vec<u8>, ProximaError> {
    let mut output = Vec::with_capacity(bytes.len());
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let line_end = bytes[cursor..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .map(|offset| cursor + offset)
            .ok_or_else(|| {
                ProximaError::Decode(
                    "chunked: size CRLF split across reads (not yet supported)".into(),
                )
            })?;
        let size_str = std::str::from_utf8(&bytes[cursor..line_end])
            .map_err(|err| ProximaError::Decode(format!("chunked size not utf-8: {err}")))?;
        let size_hex = size_str.split(';').next().unwrap_or("");
        let chunk_size = usize::from_str_radix(size_hex.trim(), 16)
            .map_err(|err| ProximaError::Decode(format!("chunked size parse: {err}")))?;
        cursor = line_end + 2;
        if chunk_size == 0 {
            *terminated = true;
            break;
        }
        let chunk_end = cursor + chunk_size;
        if chunk_end > bytes.len() {
            return Err(ProximaError::Decode(
                "chunked: payload split across reads (not yet supported)".into(),
            ));
        }
        output.extend_from_slice(&bytes[cursor..chunk_end]);
        cursor = chunk_end + 2; // trailing CRLF after chunk data
    }
    Ok(output)
}

fn try_extract_line(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    let newline_at = buffer.iter().position(|byte| *byte == b'\n')?;
    let mut line = buffer.split_off(newline_at + 1);
    std::mem::swap(buffer, &mut line);
    let mut without_newline = line;
    // strip trailing \n (and \r if present, though server emits LF-only)
    if without_newline.last() == Some(&b'\n') {
        without_newline.pop();
    }
    if without_newline.last() == Some(&b'\r') {
        without_newline.pop();
    }
    Some(without_newline)
}

pub struct HttpResponse {
    pub status: u16,
    /// kept for G6 (chunked tail decoder needs to look at transfer-encoding)
    #[allow(dead_code)]
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

async fn http_call_io<S>(
    stream: &mut S,
    method: &str,
    path: &str,
    body: Option<(&str, Vec<u8>)>,
) -> Result<HttpResponse, ProximaError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    write_request(stream, method, path, body).await?;
    let mut response_bytes = Vec::with_capacity(4096);
    stream
        .read_to_end(&mut response_bytes)
        .await
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("read response: {err}"))))?;
    parse_response(&response_bytes)
}

async fn http_call_split<R, W>(
    writer: &mut W,
    reader: &mut R,
    method: &str,
    path: &str,
    body: Option<(&str, Vec<u8>)>,
) -> Result<HttpResponse, ProximaError>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    write_request(writer, method, path, body).await?;
    // proximad's --stdio mode serves one request and exits on stdin
    // EOF. Close stdin (drop) by flushing here; read_to_end on stdout
    // catches the response body.
    writer.flush().await.map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("flush ssh stdin: {err}")))
    })?;
    // Force EOF so proximad finishes the response and exits. We can't
    // drop the borrowed writer here; use shutdown.
    writer.shutdown().await.map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("shutdown ssh stdin: {err}")))
    })?;
    let mut response_bytes = Vec::with_capacity(4096);
    reader
        .read_to_end(&mut response_bytes)
        .await
        .map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!("read ssh stdout: {err}")))
        })?;
    parse_response(&response_bytes)
}

async fn write_request<W>(
    writer: &mut W,
    method: &str,
    path: &str,
    body: Option<(&str, Vec<u8>)>,
) -> Result<(), ProximaError>
where
    W: AsyncWrite + Unpin,
{
    let mut request =
        Vec::with_capacity(128 + path.len() + body.as_ref().map_or(0, |entry| entry.1.len()));
    request.extend_from_slice(method.as_bytes());
    request.push(b' ');
    request.extend_from_slice(path.as_bytes());
    request.extend_from_slice(b" HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
    if let Some((content_type, body_bytes)) = &body {
        request.extend_from_slice(b"Content-Type: ");
        request.extend_from_slice(content_type.as_bytes());
        request.extend_from_slice(b"\r\n");
        request.extend_from_slice(format!("Content-Length: {}\r\n", body_bytes.len()).as_bytes());
        request.extend_from_slice(b"\r\n");
        request.extend_from_slice(body_bytes);
    } else {
        request.extend_from_slice(b"Content-Length: 0\r\n\r\n");
    }
    writer
        .write_all(&request)
        .await
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("write request: {err}"))))?;
    Ok(())
}

fn parse_response(bytes: &[u8]) -> Result<HttpResponse, ProximaError> {
    let header_end = bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| ProximaError::Decode("response missing header terminator".into()))?;
    let head = &bytes[..header_end];
    let head_text = std::str::from_utf8(head)
        .map_err(|err| ProximaError::Decode(format!("response head not utf-8: {err}")))?;
    let mut lines = head_text.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| ProximaError::Decode("empty status line".into()))?;
    let status = status_line
        .split(' ')
        .nth(1)
        .and_then(|raw| raw.parse::<u16>().ok())
        .ok_or_else(|| ProximaError::Decode(format!("malformed status line: {status_line:?}")))?;
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for header_line in lines {
        let Some((name, value)) = header_line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_string();
        let value = value.trim().to_string();
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.parse::<usize>().ok();
        } else if name.eq_ignore_ascii_case("transfer-encoding")
            && value.eq_ignore_ascii_case("chunked")
        {
            chunked = true;
        }
        headers.push((name, value));
    }
    let body_start = header_end + 4;
    let body = if chunked {
        decode_chunked(&bytes[body_start..])?
    } else if let Some(length) = content_length {
        let end = body_start.saturating_add(length).min(bytes.len());
        bytes[body_start..end].to_vec()
    } else {
        bytes[body_start..].to_vec()
    };
    Ok(HttpResponse {
        status,
        headers,
        body,
    })
}

fn decode_chunked(bytes: &[u8]) -> Result<Vec<u8>, ProximaError> {
    let mut output = Vec::new();
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let line_end = bytes[cursor..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .map(|offset| cursor + offset)
            .ok_or_else(|| ProximaError::Decode("chunked: missing size CRLF".into()))?;
        let size_str = std::str::from_utf8(&bytes[cursor..line_end])
            .map_err(|err| ProximaError::Decode(format!("chunked size not utf-8: {err}")))?;
        let size_hex = size_str.split(';').next().unwrap_or("");
        let chunk_size = usize::from_str_radix(size_hex.trim(), 16)
            .map_err(|err| ProximaError::Decode(format!("chunked size parse: {err}")))?;
        cursor = line_end + 2;
        if chunk_size == 0 {
            break;
        }
        let chunk_end = cursor + chunk_size;
        if chunk_end > bytes.len() {
            return Err(ProximaError::Decode("chunked: payload short".into()));
        }
        output.extend_from_slice(&bytes[cursor..chunk_end]);
        cursor = chunk_end + 2;
    }
    Ok(output)
}

async fn submit(transport: PipelineTransport, spec: PathBuf) -> Result<(), ProximaError> {
    let extension = spec
        .extension()
        .and_then(|raw| raw.to_str())
        .map(str::to_ascii_lowercase);
    let content_type = if matches!(extension.as_deref(), Some("toml")) {
        "application/toml"
    } else {
        "application/json"
    };
    let body = tokio::fs::read(&spec).await.map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("read spec {spec:?}: {err}")))
    })?;
    let connection = transport.connect().await?;
    let response = connection
        .http_call("POST", "/pipelines/submit", Some((content_type, body)))
        .await?;
    if response.status != 201 {
        return Err(ProximaError::Upstream(format!(
            "submit failed: HTTP {} — {}",
            response.status,
            String::from_utf8_lossy(&response.body)
        )));
    }
    println!("{}", String::from_utf8_lossy(&response.body));
    Ok(())
}

async fn list(
    transport: PipelineTransport,
    name: Option<String>,
    spec_hash_hex: Option<String>,
) -> Result<(), ProximaError> {
    let mut path = String::from("/pipelines");
    let mut params: Vec<(String, String)> = Vec::new();
    if let Some(name) = name {
        params.push(("name".into(), name));
    }
    if let Some(hex) = spec_hash_hex {
        params.push(("spec_hash_hex".into(), hex));
    }
    if !params.is_empty() {
        path.push('?');
        for (index, (key, value)) in params.iter().enumerate() {
            if index > 0 {
                path.push('&');
            }
            path.push_str(key);
            path.push('=');
            path.push_str(value);
        }
    }
    let connection = transport.connect().await?;
    let response = connection.http_call("GET", &path, None).await?;
    if response.status != 200 {
        return Err(ProximaError::Upstream(format!(
            "list failed: HTTP {}",
            response.status
        )));
    }
    println!("{}", String::from_utf8_lossy(&response.body));
    Ok(())
}

async fn resolve(transport: PipelineTransport, query: &str) -> Result<(), ProximaError> {
    let encoded = urlencode(query);
    let path = format!("/pipelines/resolve?q={encoded}");
    let connection = transport.connect().await?;
    let response = connection.http_call("GET", &path, None).await?;
    if response.status != 200 {
        return Err(ProximaError::Upstream(format!(
            "resolve failed: HTTP {} — {}",
            response.status,
            String::from_utf8_lossy(&response.body)
        )));
    }
    println!("{}", String::from_utf8_lossy(&response.body));
    Ok(())
}

async fn artifact(
    transport: PipelineTransport,
    query: &str,
    stage: &str,
    artifact_path: &str,
    output: PathBuf,
) -> Result<(), ProximaError> {
    let id = resolve_to_id(&transport, query).await?;
    let path = format!(
        "/pipelines/{id}/artifact?stage={}&path={}",
        urlencode(stage),
        urlencode(artifact_path)
    );
    let connection = transport.connect().await?;
    // artifact body is chunked but for the CLI we just want the bytes —
    // http_call already chunked-decodes when the response advertises it.
    let response = connection.http_call("GET", &path, None).await?;
    if response.status != 200 {
        return Err(ProximaError::Upstream(format!(
            "artifact failed: HTTP {} — {}",
            response.status,
            String::from_utf8_lossy(&response.body)
        )));
    }
    if output.as_os_str() == "-" {
        use tokio::io::AsyncWriteExt;
        tokio::io::stdout()
            .write_all(&response.body)
            .await
            .map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!("write stdout: {err}")))
            })?;
    } else {
        tokio::fs::write(&output, &response.body)
            .await
            .map_err(|err| {
                ProximaError::Io(std::io::Error::other(format!("write {output:?}: {err}")))
            })?;
    }
    Ok(())
}

async fn replay(
    transport: PipelineTransport,
    query: &str,
    substitutes: Vec<String>,
) -> Result<(), ProximaError> {
    let id = resolve_to_id(&transport, query).await?;
    // parse each `stage_name=path` into a (name, StageSpec) pair, then
    // serialize the whole map as JSON for the POST body.
    let mut map: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    for entry in &substitutes {
        let Some((stage_name, path)) = entry.split_once('=') else {
            return Err(ProximaError::Config(format!(
                "--substitute must be `stage_name=path`, got `{entry}`"
            )));
        };
        let stage_path = std::path::Path::new(path);
        let raw = tokio::fs::read(stage_path).await.map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!(
                "read substitute spec {stage_path:?}: {err}"
            )))
        })?;
        let ext = stage_path
            .extension()
            .and_then(|raw| raw.to_str())
            .map(str::to_ascii_lowercase);
        let stage_spec: serde_json::Value = if matches!(ext.as_deref(), Some("toml")) {
            let text = std::str::from_utf8(&raw)
                .map_err(|err| ProximaError::Decode(format!("substitute toml not utf-8: {err}")))?;
            let value: toml::Value = toml::from_str(text)
                .map_err(|err| ProximaError::Decode(format!("parse substitute toml: {err}")))?;
            serde_json::to_value(value)
                .map_err(|err| ProximaError::Encode(format!("toml→json: {err}")))?
        } else {
            serde_json::from_slice(&raw)
                .map_err(|err| ProximaError::Decode(format!("parse substitute json: {err}")))?
        };
        map.insert(stage_name.into(), stage_spec);
    }
    let body = serde_json::to_vec(&map)
        .map_err(|err| ProximaError::Encode(format!("serialize substitutes: {err}")))?;
    let connection = transport.connect().await?;
    let response = connection
        .http_call(
            "POST",
            &format!("/pipelines/{id}/replay"),
            Some(("application/json", body)),
        )
        .await?;
    if response.status != 201 {
        return Err(ProximaError::Upstream(format!(
            "replay failed: HTTP {} — {}",
            response.status,
            String::from_utf8_lossy(&response.body)
        )));
    }
    println!("{}", String::from_utf8_lossy(&response.body));
    Ok(())
}

async fn explain(
    transport: PipelineTransport,
    query: &str,
    stage: &str,
) -> Result<(), ProximaError> {
    let id = resolve_to_id(&transport, query).await?;
    let connection = transport.connect().await?;
    let path = format!("/pipelines/{id}/explain?stage={}", urlencode(stage));
    let response = connection.http_call("GET", &path, None).await?;
    if response.status != 200 {
        return Err(ProximaError::Upstream(format!(
            "explain failed: HTTP {} — {}",
            response.status,
            String::from_utf8_lossy(&response.body)
        )));
    }
    let chain: Vec<serde_json::Value> = serde_json::from_slice(&response.body)
        .map_err(|err| ProximaError::Decode(format!("decode explain: {err}")))?;
    for step in chain {
        let stage_name = step
            .get("stage")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let depth = step
            .get("depth")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let indent = "  ".repeat(depth);
        if depth == 0 {
            println!("{indent}{stage_name}");
        } else {
            println!("{indent}↑ depends on {stage_name}");
        }
    }
    Ok(())
}

async fn tail(transport: PipelineTransport, query: &str) -> Result<(), ProximaError> {
    let id = resolve_to_id(&transport, query).await?;
    let connection = transport.connect().await?;
    let mut stream = connection
        .stream_chunked_ndjson("GET", &format!("/pipelines/{id}/tail"))
        .await?;
    while let Some(line) = stream.recv().await? {
        println!("{}", String::from_utf8_lossy(&line));
    }
    Ok(())
}

async fn events(transport: PipelineTransport) -> Result<(), ProximaError> {
    let connection = transport.connect().await?;
    let mut stream = connection.stream_chunked_ndjson("GET", "/events").await?;
    while let Some(line) = stream.recv().await? {
        println!("{}", String::from_utf8_lossy(&line));
    }
    Ok(())
}

async fn resolve_to_id(transport: &PipelineTransport, query: &str) -> Result<String, ProximaError> {
    let connection = transport.connect().await?;
    let path = format!("/pipelines/resolve?q={}", urlencode(query));
    let response = connection.http_call("GET", &path, None).await?;
    if response.status != 200 {
        return Err(ProximaError::Upstream(format!(
            "resolve failed: HTTP {} — {}",
            response.status,
            String::from_utf8_lossy(&response.body)
        )));
    }
    let resolved: serde_json::Value = serde_json::from_slice(&response.body)
        .map_err(|err| ProximaError::Decode(format!("decode resolve: {err}")))?;
    resolved
        .get("pipeline_id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| ProximaError::Decode("resolve response missing pipeline_id".into()))
}

async fn inspect(transport: PipelineTransport, query: &str) -> Result<(), ProximaError> {
    // resolve first, then inspect by canonical id
    let resolve_connection = transport.connect().await?;
    let resolve_path = format!("/pipelines/resolve?q={}", urlencode(query));
    let resolve_response = resolve_connection
        .http_call("GET", &resolve_path, None)
        .await?;
    if resolve_response.status != 200 {
        return Err(ProximaError::Upstream(format!(
            "inspect: resolve failed: HTTP {} — {}",
            resolve_response.status,
            String::from_utf8_lossy(&resolve_response.body)
        )));
    }
    let resolved: serde_json::Value = serde_json::from_slice(&resolve_response.body)
        .map_err(|err| ProximaError::Decode(format!("decode resolve: {err}")))?;
    let id = resolved
        .get("pipeline_id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ProximaError::Decode("resolve response missing pipeline_id".into()))?;
    let inspect_connection = transport.connect().await?;
    let inspect_response = inspect_connection
        .http_call("GET", &format!("/pipelines/{id}"), None)
        .await?;
    if inspect_response.status != 200 {
        return Err(ProximaError::Upstream(format!(
            "inspect failed: HTTP {} — {}",
            inspect_response.status,
            String::from_utf8_lossy(&inspect_response.body)
        )));
    }
    println!("{}", String::from_utf8_lossy(&inspect_response.body));
    Ok(())
}

/// Minimal URL-encode for query-string values. Only handles the chars
/// likely to appear in human-friendly names/queries (whitespace + a
/// few specials). Daemon-side parsing is already lenient.
fn urlencode(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(byte as char);
            }
            _ => {
                output.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    output
}
