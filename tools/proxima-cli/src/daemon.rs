#[cfg(unix)]
use std::future::IntoFuture;
use std::path::PathBuf;
#[cfg(unix)]
use std::pin::Pin;
use std::process::ExitCode;
use std::sync::Arc;
#[cfg(unix)]
use std::time::Duration;

use clap::{Parser, Subcommand};
use futures::channel::oneshot;
use proxima::{
    DynPipelineControlPlane, FsPipelineControlPlane, ListenProtocol, McpListenProtocol,
    PipelineControlPlanePipe, ProximaError, ServeContext, into_handle, serve_h1_connection,
};
#[cfg(unix)]
use proxima::{HttpListenProtocol, ListenProtocolFluent};
use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::error;
use proxima_telemetry::error::Error as TelemetryError;
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::recorder::Recorder;
#[cfg(unix)]
use tokio::signal::unix::{SignalKind, signal};
use tokio_util::compat::TokioAsyncReadCompatExt;

mod tensor_run;

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

        /// Also write telemetry as JSON lines to this file; console output is kept.
        #[arg(long, value_name = "PATH")]
        telemetry_file: Option<PathBuf>,
    },
}

#[proxima::main(runtime = "tokio", flavor = "multi_thread")]
async fn main() -> ExitCode {
    let directive =
        std::env::var("RUST_LOG").unwrap_or_else(|_| "warn,proxima=info,proximad=info".to_string());
    global::install(EnvFilter::parse(&directive));
    let cli = Cli::parse();
    let telemetry_file = match &cli.command {
        Command::Serve { telemetry_file, .. } => telemetry_file.clone(),
    };
    let Ok(recorder) = install_telemetry(telemetry_file) else {
        eprintln!("proximad: failed to install telemetry");
        return ExitCode::from(1);
    };

    let outcome = run(cli).await;
    recorder.drain();
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            error!(?error, "proximad exited with error");
            ExitCode::from(1)
        }
    }
}

/// Console-only unless `--telemetry-file` is given; then console plus a JSON-lines file.
fn install_telemetry(telemetry_file: Option<PathBuf>) -> Result<Arc<Recorder>, TelemetryError> {
    let Some(path) = telemetry_file else {
        return proxima::init_telemetry();
    };
    let exporter = Exporter::fan(vec![
        Exporter::std(),
        Exporter::file(path).format(Formatter::Json),
    ])?;
    let recorder = Recorder::builder().export(exporter)?.install()?;
    let pump = Arc::clone(&recorder);
    std::thread::Builder::new()
        .name("proximad-telemetry-drain".to_string())
        .spawn(move || pump.run_drain_loop())
        .map_err(|error| TelemetryError::ThreadSpawn(error.to_string()))?;
    Ok(recorder)
}

async fn run(cli: Cli) -> Result<(), ProximaError> {
    match cli.command {
        Command::Serve {
            unix,
            stdio,
            mcp_stdio,
            mcp_unix,
            state_dir,
            telemetry_file: _,
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
    #[cfg(windows)]
    if unix.is_some() || mcp_unix.is_some() {
        return Err(ProximaError::Config(
            "daemon unix sockets require Unix; use --stdio or --mcp-stdio".into(),
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
    let handle = tensor_run::mount(into_handle(pipe));
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
    serve_unix(handle, &unix_path).await
}

#[cfg(not(unix))]
async fn serve_unix(
    _handle: proxima::PipeHandle,
    _path: &std::path::Path,
) -> Result<(), ProximaError> {
    Err(ProximaError::Config(
        "daemon unix sockets require Unix; use --stdio".into(),
    ))
}

#[cfg(unix)]
async fn serve_unix(
    handle: proxima::PipeHandle,
    unix_path: &std::path::Path,
) -> Result<(), ProximaError> {
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
    let probe = probe_uds_until_ready(unix_path);
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

#[cfg(unix)]
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

#[cfg(all(test, windows))]
mod windows_tests {
    use super::{ProximaError, serve};

    #[proxima::test]
    async fn windows_cli_server_uds_rejects_before_state_creation() {
        let directory = tempfile::tempdir().expect("create daemon test directory");
        let socket = directory.path().join("daemon.sock");
        let state = directory.path().join("nonexistent-state");

        let http_error = serve(
            Some(socket.clone()),
            false,
            false,
            None,
            Some(state.clone()),
        )
        .await
        .expect_err("reject HTTP Unix socket mode on Windows");
        assert!(
            matches!(http_error, ProximaError::Config(message) if message.contains("require Unix"))
        );
        assert!(!state.exists(), "HTTP refusal must precede state creation");

        let mcp_error = serve(None, false, false, Some(socket), Some(state.clone()))
            .await
            .expect_err("reject MCP Unix socket mode on Windows");
        assert!(
            matches!(mcp_error, ProximaError::Config(message) if message.contains("require Unix"))
        );
        assert!(!state.exists(), "MCP refusal must precede state creation");
    }
}
