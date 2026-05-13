use std::future::IntoFuture;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use futures::channel::oneshot;
use proxima::{
    DynPipelineControlPlane, FsPipelineControlPlane, HttpListenProtocol, ListenProtocol,
    ListenProtocolFluent, McpListenProtocol, PipelineControlPlanePipe, ProximaError, ServeContext,
    into_handle, serve_h1_connection,
};
use tokio::signal::unix::{SignalKind, signal};
use tokio_util::compat::TokioAsyncReadCompatExt;
use tracing::error;
use tracing_subscriber::EnvFilter;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
#[command(
    name = "proximad",
    about = "proxima pipeline daemon — runs declarative pipelines, streams their recordings",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Serve the pipeline control plane until SIGTERM/SIGINT.
    Serve {
        /// Listen on a Unix domain socket at this path. The path is
        /// recreated each launch (a stale socket file is removed first).
        #[arg(long, value_name = "PATH")]
        unix: Option<PathBuf>,

        /// Serve a single HTTP/1.1 connection over stdin/stdout. Used
        /// by the CLI's SSH-pipe transport.
        #[arg(long)]
        stdio: bool,

        /// Serve MCP (JSON-RPC 2.0) over stdin/stdout. Off by default
        /// — opt in only when an agent needs the pipeline tools.
        /// Cannot be combined with --stdio or --unix.
        #[arg(long)]
        mcp_stdio: bool,

        /// Serve MCP over a Unix domain socket at this path. Cannot
        /// be combined with --stdio or --unix.
        #[arg(long, value_name = "PATH")]
        mcp_unix: Option<PathBuf>,

        /// Root directory for persistent state. Defaults to
        /// `$HOME/.local/share/proximad/pipelines`.
        #[arg(long, value_name = "DIR")]
        state_dir: Option<PathBuf>,
    },
}

#[proxima::main(runtime = "tokio", flavor = "multi_thread")]
async fn main() -> ExitCode {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn,proxima=info,proximad=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    let cli = Cli::parse();
    match run(cli).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            error!(?error, "proximad exited with error");
            ExitCode::from(1)
        }
    }
}

async fn run(cli: Cli) -> Result<(), ProximaError> {
    match cli.command {
        Command::Serve {
            unix,
            stdio,
            mcp_stdio,
            mcp_unix,
            state_dir,
        } => serve(unix, stdio, mcp_stdio, mcp_unix, state_dir).await,
    }
}

async fn serve(
    unix: Option<PathBuf>,
    stdio: bool,
    mcp_stdio: bool,
    mcp_unix: Option<PathBuf>,
    state_dir: Option<PathBuf>,
) -> Result<(), ProximaError> {
    let mode_count = [stdio, unix.is_some(), mcp_stdio, mcp_unix.is_some()]
        .iter()
        .filter(|flag| **flag)
        .count();
    if mode_count > 1 {
        return Err(ProximaError::Config(
            "pick exactly one of --unix / --stdio / --mcp-stdio / --mcp-unix".into(),
        ));
    }
    let state_dir = state_dir.unwrap_or_else(default_state_dir);
    // arm the recording spigot with a tokio-backed runtime (we are under
    // #[tokio::main]); the FS control plane's per-pipeline durables stay inert
    // until armed (C7 spigot model).
    let recording_spigot = proxima::deferred_runtime();
    let _ = recording_spigot.set(Arc::new(proxima::runtime::TokioPerCoreRuntime::new(1)?)
        as Arc<dyn proxima::runtime::Runtime>);
    let plane = Arc::new(FsPipelineControlPlane::open(&state_dir, recording_spigot).await?);
    let dyn_plane: DynPipelineControlPlane = plane.clone();
    let pipe = PipelineControlPlanePipe::new(dyn_plane);
    let handle = into_handle(pipe);
    if stdio {
        return serve_stdio(handle).await;
    }
    if mcp_stdio {
        return serve_mcp(handle, McpTransport::Stdio).await;
    }
    if let Some(path) = mcp_unix {
        return serve_mcp(handle, McpTransport::Unix(path)).await;
    }
    let unix_path = unix.ok_or_else(|| {
        ProximaError::Config(
            "serve requires one of --unix <path> / --stdio / --mcp-stdio / --mcp-unix <path>"
                .into(),
        )
    })?;
    let protocol = HttpListenProtocol::default();
    let spec = serde_json::json!({ "path": unix_path.to_string_lossy() });
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let context = ServeContext::new(Arc::new(proxima::NoopTelemetry));
    // UDS dispatch uses spec.path; bind is required by the builder but ignored on the UDS leg.
    let bind = std::net::SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), 0);
    // ServeBuilder is IntoFuture (not itself a Future); convert before tokio::select.
    let mut serve: Pin<Box<dyn std::future::Future<Output = Result<(), ProximaError>> + Send>> =
        protocol
            .fluent()
            .bind(bind)
            .dispatch(handle)
            .spec(spec)
            .context(context)
            .shutdown(shutdown_rx)
            .into_future();

    let mut sigterm = signal(SignalKind::terminate()).map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("install sigterm: {err}")))
    })?;
    let mut sigint = signal(SignalKind::interrupt())
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("install sigint: {err}"))))?;

    // probe for the listener actually accepting before printing READY.
    // serve_default_uds binds in-future, so a naive println here would
    // race the bind. drive serve until a probe-connect succeeds; if
    // serve errors before then, surface that error.
    let probe = probe_uds_until_ready(&unix_path);
    tokio::pin!(probe);
    tokio::select! {
        outcome = &mut serve => return outcome,
        ready = &mut probe => ready?,
    }
    println!("READY {}", unix_path.display());
    use std::io::Write as _;
    let _ = std::io::stdout().flush();

    tokio::select! {
        outcome = &mut serve => outcome?,
        _ = sigterm.recv() => {
            let _ = shutdown_tx.send(());
            serve.await?;
        }
        _ = sigint.recv() => {
            let _ = shutdown_tx.send(());
            serve.await?;
        }
    }
    Ok(())
}

async fn probe_uds_until_ready(path: &std::path::Path) -> Result<(), ProximaError> {
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if tokio::net::UnixStream::connect(path).await.is_ok() {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(ProximaError::Io(std::io::Error::other(format!(
                "uds listener did not accept within 5s at {path:?}"
            ))));
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn default_state_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("proximad")
            .join("pipelines")
    } else {
        PathBuf::from("./proximad-state/pipelines")
    }
}

enum McpTransport {
    Stdio,
    Unix(PathBuf),
}

/// Serve MCP (JSON-RPC 2.0) backed by the pipeline control plane. Off
/// by default; opt in via `proximad serve --mcp-stdio` or
/// `--mcp-unix <path>`. The pipeline tool set lives in
/// proxima::listeners::mcp; this just wires that listener to our
/// PipelineControlPlanePipe.
async fn serve_mcp(
    handle: proxima::PipeHandle,
    transport: McpTransport,
) -> Result<(), ProximaError> {
    let protocol = McpListenProtocol::default();
    let spec = match &transport {
        McpTransport::Stdio => serde_json::json!({ "transport": "stdio" }),
        McpTransport::Unix(path) => {
            serde_json::json!({ "transport": "unix", "path": path.to_string_lossy() })
        }
    };
    let (_shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    let bind = std::net::SocketAddr::new(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), 0);
    let context = ServeContext::new(Arc::new(proxima::NoopTelemetry));
    protocol
        .serve(bind, handle, &spec, context, shutdown_rx)
        .await
}

/// Serve a single HTTP/1.1 connection over stdin/stdout. Used by the
/// CLI's SSH-pipe transport: `ssh <host> proximad serve --stdio` reads
/// requests off stdin, writes responses to stdout. The connection
/// terminates on stdin EOF (which happens when ssh closes the pipe).
async fn serve_stdio(handle: proxima::PipeHandle) -> Result<(), ProximaError> {
    // tokio::io::stdin/stdout each impl tokio AsyncRead/AsyncWrite;
    // serve_h1_connection wants futures-io traits, so .compat() the
    // joined duplex. No READY line — the caller (CLI's SSH transport)
    // sends a request and expects only HTTP bytes on stdout; a READY
    // prefix would corrupt the response stream.
    let duplex = tokio::io::join(tokio::io::stdin(), tokio::io::stdout()).compat();
    serve_h1_connection(duplex, handle, None, None).await
}
