mod pipeline;

use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use pipeline::PipelineArgs;
use proxima::scenarios::{
    DurationSpec, WorkloadMode, discover_by_name, discover_scenario, run_scenario_with_sink,
};
use proxima::{
    App, LoadContext, MetricsSnapshot, MountTarget, ProximaError, Request, RunConfig, Scenario,
    ScenarioReport, SendPipe, Spec, default_config_format_registry, load,
};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::signal::unix::{SignalKind, signal};
use tracing::error;
use tracing_subscriber::EnvFilter;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
#[command(
    name = "proxima",
    about = "config-first proxy/middleware runtime",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Call {
        #[arg(long)]
        config: PathBuf,

        #[arg(long, default_value = "GET")]
        method: String,

        #[arg(long, default_value = "/")]
        path: String,

        #[arg(long)]
        body: Option<String>,

        #[arg(long = "header", value_parser = parse_header)]
        headers: Vec<(String, String)>,
    },
    /// Load a pipe config and serve it on an HTTP listener until
    /// SIGTERM/SIGINT. Prints `READY <bound-addr>` to stdout once the
    /// listener has accepted its first probe — scenario orchestrators
    /// poll this signal before driving traffic.
    Serve {
        /// Inline pipe spec, or a path. Strings starting with `{`,
        /// `[`, `<`, or containing a newline are treated as content;
        /// anything else is a path. Mutually exclusive with `--config`.
        spec: Option<String>,

        /// Explicit path to a pipe config file. Mutually exclusive
        /// with the positional `spec`.
        #[arg(long)]
        config: Option<PathBuf>,

        /// Bind address. Use `127.0.0.1:0` to let the OS pick a port —
        /// the chosen addr is reported via the READY line.
        #[arg(long, default_value = "127.0.0.1:0")]
        addr: SocketAddr,

        /// Path the pipe is mounted at (default `/`).
        #[arg(long, default_value = "/")]
        mount: String,
    },
    /// Load a config and emit the registered schemas in the chosen format.
    Describe {
        #[arg(long)]
        config: PathBuf,

        /// Output format: json-schema, openapi, or toml.
        #[arg(long, default_value = "json-schema")]
        format: String,

        /// Optional title for the OpenAPI document.
        #[arg(long, default_value = "proxima")]
        title: String,

        /// Optional version for the OpenAPI document.
        #[arg(long, default_value = "0.1.0")]
        version: String,
    },
    /// Talk to a running proxima daemon over its UDS HTTP/1.1 control
    /// plane. The daemon must be exposing its `ControlPlanePipe` on
    /// the configured socket path.
    Daemon {
        /// Path to the daemon's control-plane UDS.
        #[arg(long, default_value = "/tmp/proxima.sock", env = "PROXIMA_SOCKET")]
        socket: PathBuf,

        #[command(subcommand)]
        action: DaemonAction,
    },
    /// Hot-swap a running pipe to a new spec. Convenience alias
    /// for `proxima daemon apply <name> --spec <path>` — same wire
    /// (POST /pipes/<name>/apply over HTTP/1.1 on the daemon UDS).
    /// In-flight requests on the old impl complete; new requests hit
    /// the new impl.
    Apply {
        /// Name of the registered pipe to swap.
        name: String,

        /// Path to a file containing the new pipe spec.
        #[arg(long)]
        spec: PathBuf,

        /// Path to the daemon's control-plane UDS.
        #[arg(long, default_value = "/tmp/proxima.sock", env = "PROXIMA_SOCKET")]
        socket: PathBuf,
    },
    /// Walk the causal graph backward from a recorded output byte.
    /// Reads a JSONL causal index (one CausalEdge per line) and prints
    /// the chain of edges from the queried (node, offset) up to the
    /// earliest ancestor recorded.
    Explain {
        /// Path to the JSONL causal index file (one CausalEdge per line).
        #[arg(long)]
        index: PathBuf,

        /// Node id to start the walk from.
        #[arg(long)]
        node: String,

        /// Byte offset within that node's output to query.
        #[arg(long)]
        offset: u64,
    },
    /// Submit and inspect pipelines on a running `proximad`. Pipelines
    /// are DAGs of stages (each one a child process) submitted by toml
    /// or json spec; the CLI talks to proximad over either a local UDS
    /// or `ssh <host> proximad serve --stdio`.
    Pipeline(PipelineArgs),
    /// Walk a parsed spec and report policy violations. Zero-arg
    /// invocation at a project root discovers `proxima.toml` /
    /// `spec.toml` / `.proxima/spec.toml` and `proxima.policy.toml` /
    /// `.proxima/policy.toml`.
    ///
    /// Exit codes: 0 = clean (no FAIL; WARN-only without `--strict`),
    /// 1 = any FAIL (or WARN with `--strict`), 2 = invocation / IO /
    /// discovery / parse error.
    Verify {
        /// Path to the spec file. Optional; discovered if absent.
        spec: Option<PathBuf>,

        /// Path to a `policy.toml`. Optional; discovered if absent. If
        /// no policy exists at all, the built-in invariants run with
        /// default severities.
        #[arg(long)]
        policy: Option<PathBuf>,

        /// Output format: `text` (default, grep-friendly) or `json`.
        #[arg(long, default_value = "text")]
        format: String,

        /// Upgrade WARN to FAIL for exit-code purposes.
        #[arg(long)]
        strict: bool,

        /// Run the spec through coherence repair: project the spec to
        /// the largest subset that passes verification, emit a
        /// `REPAIR` line per dropped item, then report on the
        /// post-repair spec. Exit code is computed against the
        /// post-repair report.
        #[arg(long)]
        repair: bool,
    },
    /// Stream a recorded `.bin` session, run replay-policy
    /// assertions, and emit a verification report. Zero-arg
    /// invocation at a project root discovers the newest
    /// `*.bin` under `./.proxima/recordings/` or `./recordings/`
    /// and the policy from the same locations `verify` searches.
    ///
    /// Exit codes: same shape as `verify` (0 / 1 / 2).
    Replay {
        /// Path to a `.bin` recording. Optional; newest discovered
        /// if absent.
        #[arg(long)]
        recording: Option<PathBuf>,

        /// Path to a `policy.toml`. Optional; discovered if absent.
        #[arg(long = "verify")]
        policy: Option<PathBuf>,

        /// Path to a spec file. Required when policy declares
        /// `byte_identical_pipes`. Discovered from cwd if absent.
        #[arg(long)]
        spec: Option<PathBuf>,

        /// Output format: `text` (default) or `json`.
        #[arg(long, default_value = "text")]
        format: String,

        /// Upgrade WARN to FAIL for exit-code purposes.
        #[arg(long)]
        strict: bool,

        /// Revert the policy to match what the recording actually
        /// shows. Drops `must_derive_from_record` entries whose pipes
        /// produced inferred events (or are declared non-idempotent
        /// in the spec), emits a `REPAIR` line per dropped entry, and
        /// reports against the relaxed policy.
        #[arg(long)]
        repair: bool,
    },
    /// Drive a load scenario from a TOML file. Open-loop or
    /// closed-loop driver selection comes from the workload spec
    /// itself: `target_rps + duration` (or `profile`) means open-loop;
    /// `requests + concurrency` means closed-loop. Zero-arg
    /// invocation at a project root discovers
    /// `proxima.scenario.toml` / `scenario.toml` /
    /// `.proxima/scenario.toml`.
    ///
    /// Exit codes: 0 = expectations pass, 1 = any expectation fails,
    /// 2 = invocation / IO / parse error.
    Load {
        /// Path to a `scenario.toml`. Optional; discovered from cwd
        /// if absent.
        scenario: Option<PathBuf>,

        /// Look up scenario by stem name across tiered search paths
        /// (PROXIMA_SCENARIO_PATH env, cwd, XDG, /etc).
        #[arg(long, conflicts_with = "scenario")]
        name: Option<String>,

        /// Emit the final ScenarioReport as JSON to stdout instead of
        /// a text summary. Per-second snapshot rows are suppressed
        /// in JSON mode (use `--remote` event stream if you need
        /// live snapshots in a JSON pipeline).
        #[arg(long)]
        json: bool,

        /// Override `target_rps` from the scenario (open-loop only).
        #[arg(long)]
        rps: Option<u64>,

        /// Override `duration` from the scenario, accepting `30s` /
        /// `5m` / `2h` strings (open-loop only).
        #[arg(long)]
        duration: Option<String>,

        /// Target a remote proximad orchestrator. Stage-2 placeholder;
        /// not yet implemented — passing this flag exits 2 with a
        /// notice.
        #[arg(long)]
        remote: Option<String>,

        /// Capture traffic to a `.bin` recording for later replay /
        /// verify. Stage-2 placeholder; not yet wired — passing this
        /// flag prints a notice and continues without recording.
        #[arg(long)]
        record: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum DaemonAction {
    /// List every pipe the daemon knows about with its current state.
    List,
    /// Show one pipe's status by name.
    Status { name: String },
    /// Snapshot the daemon's metrics (counters / gauges / histogram p50+p99).
    Metrics,
    /// Tail recent stdout/stderr lines from one supervised pipe.
    Logs {
        name: String,

        /// Max number of recent lines to return (newest last). Default 100.
        #[arg(long, default_value_t = 100)]
        max_lines: usize,
    },
    /// Start a pipe (cascades through configured `requires` deps).
    Start { name: String },
    /// Stop a pipe.
    Stop { name: String },
    /// Restart a pipe (stop + start, bumps restart count).
    Restart { name: String },
    /// Reload the daemon's config from disk.
    Reload,
    /// Hot-swap a pipe to a new spec. In-flight requests on the old
    /// impl complete; new requests hit the new impl. Spec is read from
    /// the supplied file (any of the formats `proxima serve` accepts).
    Apply {
        /// Name of the registered pipe to swap.
        name: String,

        /// Path to a file containing the new pipe spec.
        #[arg(long)]
        spec: PathBuf,
    },
}

fn parse_header(raw: &str) -> Result<(String, String), String> {
    let (name, value) = raw
        .split_once(':')
        .ok_or_else(|| format!("expected 'name: value', got '{raw}'"))?;
    Ok((name.trim().to_string(), value.trim().to_string()))
}

#[proxima::main(runtime = "tokio", flavor = "multi_thread")]
async fn main() -> ExitCode {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("warn,proxima=info,proxima_cli=info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    let cli = Cli::parse();
    // Verify owns its exit code semantics (0/1/2); other commands map
    // Ok→0, Err→1.
    match cli.command {
        Command::Verify {
            spec,
            policy,
            format,
            strict,
            repair,
        } => run_verify(spec, policy, format, strict, repair).await,
        Command::Replay {
            recording,
            policy,
            spec,
            format,
            strict,
            repair,
        } => run_replay(recording, policy, spec, format, strict, repair).await,
        Command::Load {
            scenario,
            name,
            json,
            rps,
            duration,
            remote,
            record,
        } => run_load(scenario, name, json, rps, duration, remote, record).await,
        other => match run(Cli { command: other }).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                error!(?error, "command failed");
                ExitCode::from(1)
            }
        },
    }
}

async fn run(cli: Cli) -> Result<(), ProximaError> {
    match cli.command {
        Command::Call {
            config,
            method,
            path,
            body,
            headers,
        } => run_call(config, method, path, body, headers).await,
        Command::Serve {
            spec,
            config,
            addr,
            mount,
        } => run_serve(spec, config, addr, mount).await,
        Command::Describe {
            config,
            format,
            title,
            version,
        } => run_describe(config, format, title, version).await,
        Command::Daemon { socket, action } => run_daemon(socket, action).await,
        Command::Apply { name, spec, socket } => {
            // Top-level alias for `daemon apply` — same protocol path.
            run_daemon(socket, DaemonAction::Apply { name, spec }).await
        }
        Command::Explain {
            index,
            node,
            offset,
        } => run_explain(index, node, offset),
        Command::Pipeline(args) => pipeline::run(args).await,
        Command::Verify { .. } | Command::Replay { .. } | Command::Load { .. } => {
            unreachable!("Verify/Replay/Load dispatched in main for custom exit codes")
        }
    }
}

fn run_explain(index_path: PathBuf, node: String, offset: u64) -> Result<(), ProximaError> {
    let index = proxima::CausalIndex::read_jsonl(&index_path)?;
    let chain = index.explain(&node, offset);
    if chain.is_empty() {
        eprintln!("no causal edges cover ({node}, offset {offset})");
        return Ok(());
    }
    for (depth, edge) in chain.iter().enumerate() {
        let indent = "  ".repeat(depth);
        println!(
            "{indent}{} [{}..{})",
            edge.node_id, edge.output_range.start, edge.output_range.end
        );
        for (parent_id, parent_range) in &edge.parent_ranges {
            println!(
                "{indent}  ← {parent_id} [{}..{})",
                parent_range.start, parent_range.end
            );
        }
    }
    Ok(())
}

async fn run_daemon(socket: PathBuf, action: DaemonAction) -> Result<(), ProximaError> {
    let response = match action {
        DaemonAction::List => http_call_over_uds(&socket, "GET", "/pipes", Vec::new()).await?,
        DaemonAction::Status { ref name } => {
            http_call_over_uds(&socket, "GET", &format!("/pipes/{name}"), Vec::new()).await?
        }
        DaemonAction::Metrics => http_call_over_uds(&socket, "GET", "/metrics", Vec::new()).await?,
        DaemonAction::Logs {
            ref name,
            max_lines,
        } => {
            http_call_over_uds(
                &socket,
                "GET",
                &format!("/pipes/{name}/logs?max_lines={max_lines}"),
                Vec::new(),
            )
            .await?
        }
        DaemonAction::Start { ref name } => {
            http_call_over_uds(&socket, "POST", &format!("/pipes/{name}/start"), Vec::new()).await?
        }
        DaemonAction::Stop { ref name } => {
            http_call_over_uds(&socket, "POST", &format!("/pipes/{name}/stop"), Vec::new()).await?
        }
        DaemonAction::Restart { ref name } => {
            http_call_over_uds(
                &socket,
                "POST",
                &format!("/pipes/{name}/restart"),
                Vec::new(),
            )
            .await?
        }
        DaemonAction::Reload => http_call_over_uds(&socket, "POST", "/reload", Vec::new()).await?,
        DaemonAction::Apply { ref name, ref spec } => {
            let body = read_spec_as_json(spec)?;
            http_call_over_uds(&socket, "POST", &format!("/pipes/{name}/apply"), body).await?
        }
    };
    print_daemon_response(&action, response);
    Ok(())
}

/// Reads a spec file (json, yaml, toml, ron, json5) and re-serializes it
/// as JSON so the daemon's apply route can deserialize uniformly.
fn read_spec_as_json(path: &std::path::Path) -> Result<Vec<u8>, ProximaError> {
    let text = std::fs::read_to_string(path).map_err(ProximaError::Io)?;
    let hint = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_string);
    let context = LoadContext::with_default_registry()?;
    let parsed: Value = context
        .config_formats
        .parse_with_hint(&text, hint.as_deref())?;
    serde_json::to_vec(&parsed)
        .map_err(|err| ProximaError::Encode(format!("re-encode apply spec as JSON: {err}")))
}

/// Minimal HTTP/1.1 response captured from the daemon. Control plane
/// always answers with `Content-Length`; we never see chunked here.
struct DaemonResponse {
    status: u16,
    body: Vec<u8>,
}

/// HTTP/1.1 client over a Unix domain socket. Hand-rolled because:
/// 1. control plane responses are tiny + always Content-Length,
/// 2. zero need for the full hyper client surface here,
/// 3. CLI startup cost matters.
async fn http_call_over_uds(
    socket: &std::path::Path,
    method: &str,
    path_with_query: &str,
    body: Vec<u8>,
) -> Result<DaemonResponse, ProximaError> {
    let mut stream = UnixStream::connect(socket).await.map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!(
            "connect daemon at {socket:?}: {err}"
        )))
    })?;
    let mut request = Vec::with_capacity(64 + path_with_query.len() + body.len());
    request.extend_from_slice(method.as_bytes());
    request.push(b' ');
    request.extend_from_slice(path_with_query.as_bytes());
    request.extend_from_slice(b" HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n");
    if !body.is_empty() {
        request.extend_from_slice(b"Content-Type: application/json\r\n");
        request.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
    } else {
        request.extend_from_slice(b"Content-Length: 0\r\n");
    }
    request.extend_from_slice(b"\r\n");
    request.extend_from_slice(&body);
    stream
        .write_all(&request)
        .await
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("write request: {err}"))))?;
    stream
        .flush()
        .await
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("flush: {err}"))))?;
    let mut response_bytes = Vec::with_capacity(4096);
    stream
        .read_to_end(&mut response_bytes)
        .await
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("read response: {err}"))))?;
    parse_http_response(&response_bytes)
}

fn parse_http_response(bytes: &[u8]) -> Result<DaemonResponse, ProximaError> {
    let header_end = find_double_crlf(bytes)
        .ok_or_else(|| ProximaError::Decode("response missing header terminator".into()))?;
    let head = &bytes[..header_end];
    let body_start = header_end + 4;
    let head_text = std::str::from_utf8(head)
        .map_err(|err| ProximaError::Decode(format!("response head not utf-8: {err}")))?;
    let mut lines = head_text.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| ProximaError::Decode("empty response head".into()))?;
    let status = status_line
        .split(' ')
        .nth(1)
        .and_then(|raw| raw.parse::<u16>().ok())
        .ok_or_else(|| ProximaError::Decode(format!("malformed status line: {status_line:?}")))?;
    let mut content_length: Option<usize> = None;
    let mut transfer_encoding_chunked = false;
    for header_line in lines {
        let (name, value) = match header_line.split_once(':') {
            Some(pair) => pair,
            None => continue,
        };
        let name_lower = name.trim().to_ascii_lowercase();
        let value_trimmed = value.trim();
        if name_lower == "content-length" {
            content_length = value_trimmed.parse::<usize>().ok();
        } else if name_lower == "transfer-encoding" && value_trimmed.eq_ignore_ascii_case("chunked")
        {
            transfer_encoding_chunked = true;
        }
    }
    let body = if transfer_encoding_chunked {
        decode_chunked(&bytes[body_start..])?
    } else if let Some(length) = content_length {
        let end = body_start.saturating_add(length).min(bytes.len());
        bytes[body_start..end].to_vec()
    } else {
        // No Content-Length and not chunked — slurp to EOF.
        bytes[body_start..].to_vec()
    };
    Ok(DaemonResponse { status, body })
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
            return Err(ProximaError::Decode(
                "chunked: payload shorter than declared".into(),
            ));
        }
        output.extend_from_slice(&bytes[cursor..chunk_end]);
        cursor = chunk_end + 2; // skip trailing CRLF after chunk data
    }
    Ok(output)
}

fn find_double_crlf(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn print_daemon_response(action: &DaemonAction, response: DaemonResponse) {
    if response.status >= 400 {
        let body = String::from_utf8_lossy(&response.body);
        eprintln!("status: {} — {body}", response.status);
        return;
    }
    match action {
        DaemonAction::List => print_list(response.body),
        DaemonAction::Status { .. } => print_status(response.body),
        DaemonAction::Metrics => print_metrics(response.body),
        DaemonAction::Logs { .. } => print_logs(response.body),
        DaemonAction::Start { .. }
        | DaemonAction::Stop { .. }
        | DaemonAction::Restart { .. }
        | DaemonAction::Apply { .. } => print_status(response.body),
        DaemonAction::Reload => println!("ok"),
    }
}

fn print_logs(body: Vec<u8>) {
    match serde_json::from_slice::<Vec<String>>(&body) {
        Ok(lines) => {
            for line in lines {
                println!("{line}");
            }
        }
        Err(err) => eprintln!("decode logs response: {err}"),
    }
}

fn print_list(body: Vec<u8>) {
    let pipes: Vec<serde_json::Value> = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("decode list response: {err}");
            return;
        }
    };
    println!(
        "{:<24} {:<10} {:>10} {:>10}",
        "name", "state", "uptime_ms", "restarts"
    );
    for entry in pipes {
        let name = entry
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let state = entry
            .get("state")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let uptime = entry
            .get("uptime_ms")
            .and_then(serde_json::Value::as_u64)
            .map(|raw| raw.to_string())
            .unwrap_or_else(|| "-".into());
        let restarts = entry
            .get("restart_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        println!("{name:<24} {state:<10} {uptime:>10} {restarts:>10}");
    }
}

fn print_status(body: Vec<u8>) {
    match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(status) => match serde_json::to_string_pretty(&status) {
            Ok(pretty) => println!("{pretty}"),
            Err(_) => println!("{}", String::from_utf8_lossy(&body)),
        },
        Err(err) => eprintln!("decode status response: {err}"),
    }
}

fn print_metrics(body: Vec<u8>) {
    match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(snapshot) => match serde_json::to_string_pretty(&snapshot) {
            Ok(pretty) => println!("{pretty}"),
            Err(_) => println!("{}", String::from_utf8_lossy(&body)),
        },
        Err(err) => eprintln!("decode metrics response: {err}"),
    }
}

async fn run_describe(
    config: PathBuf,
    format: String,
    title: String,
    version: String,
) -> Result<(), ProximaError> {
    let raw_text = std::fs::read_to_string(&config).map_err(ProximaError::Io)?;
    let formats = default_config_format_registry()?;
    let hint = config
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_string);
    let parsed: Value = formats.parse_with_hint(&raw_text, hint.as_deref())?;
    let table = parsed
        .as_object()
        .ok_or_else(|| ProximaError::Config("describe expects a top-level object config".into()))?;
    let mut schemas = std::collections::BTreeMap::new();
    if let Some(entries) = table.get("schema").and_then(Value::as_array) {
        for entry in entries {
            let name = entry
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| ProximaError::Config("[[schema]] requires `name`".into()))?
                .to_string();
            let schema_value = entry.get("schema").cloned().ok_or_else(|| {
                ProximaError::Config(format!("[[schema]] `{name}` requires nested `schema`"))
            })?;
            let schema: proxima::Schema = serde_json::from_value(schema_value).map_err(|err| {
                ProximaError::Config(format!("[[schema]] `{name}` decode: {err}"))
            })?;
            schemas.insert(name, schema);
        }
    }
    if schemas.is_empty() {
        return Err(ProximaError::Config(
            "no [[schema]] blocks found in config".into(),
        ));
    }
    let rendered = match format.as_str() {
        "json-schema" => {
            let mut document = serde_json::Map::new();
            for (name, schema) in &schemas {
                document.insert(
                    name.clone(),
                    proxima::schema::emit::emit_json_schema(
                        schema,
                        &std::collections::BTreeMap::new(),
                    ),
                );
            }
            serde_json::to_string_pretty(&Value::Object(document))
                .map_err(|err| ProximaError::Encode(format!("json-schema render: {err}")))?
        }
        "openapi" => {
            let document = proxima::schema::emit::emit_openapi(
                &title,
                &version,
                &schemas,
                serde_json::json!({}),
            );
            serde_json::to_string_pretty(&document)
                .map_err(|err| ProximaError::Encode(format!("openapi render: {err}")))?
        }
        "toml" => proxima::schema::emit::emit_toml_schema(&schemas)?,
        other => {
            return Err(ProximaError::Config(format!(
                "unknown --format `{other}` (expected json-schema, openapi, toml)"
            )));
        }
    };
    println!("{rendered}");
    Ok(())
}

async fn run_serve(
    spec_arg: Option<String>,
    config: Option<PathBuf>,
    addr: SocketAddr,
    mount_path: String,
) -> Result<(), ProximaError> {
    let (raw_text, format_hint) = match (spec_arg, config) {
        (Some(_), Some(_)) => {
            return Err(ProximaError::Config(
                "provide either positional <spec> or --config <path>, not both".into(),
            ));
        }
        (None, None) => {
            return Err(ProximaError::Config(
                "missing pipe spec: pass positional <spec> or --config <path>".into(),
            ));
        }
        (Some(arg), None) => classify_positional_spec(arg)?,
        (None, Some(path)) => read_config_path(path)?,
    };

    let formats = default_config_format_registry()?;
    let parsed: Value = formats.parse_with_hint(&raw_text, format_hint.as_deref())?;

    let has_listen_block = parsed
        .as_object()
        .and_then(|map| map.get("listen"))
        .and_then(|value| value.as_array())
        .is_some_and(|arr| !arr.is_empty());

    if has_listen_block {
        preflight_full_listener_binds(&parsed)?;
        run_serve_full(Spec::Inline(parsed)).await
    } else {
        run_serve_single(Spec::Inline(parsed), addr, mount_path).await
    }
}

fn classify_positional_spec(arg: String) -> Result<(String, Option<String>), ProximaError> {
    let trimmed = arg.trim_start();
    let looks_like_content = trimmed.starts_with(['{', '[', '<']) || arg.contains('\n');
    if looks_like_content {
        Ok((arg, None))
    } else {
        read_config_path(PathBuf::from(&arg))
    }
}

fn read_config_path(path: PathBuf) -> Result<(String, Option<String>), ProximaError> {
    let text = std::fs::read_to_string(&path).map_err(ProximaError::Io)?;
    let hint = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_string);
    Ok((text, hint))
}

fn preflight_full_listener_binds(parsed: &Value) -> Result<(), ProximaError> {
    let Some(listeners) = parsed
        .as_object()
        .and_then(|map| map.get("listen"))
        .and_then(|value| value.as_array())
    else {
        return Ok(());
    };
    let mut reservations = Vec::with_capacity(listeners.len());
    for listen in listeners {
        let protocol = listen
            .get("type")
            .and_then(|value| value.as_str())
            .unwrap_or("http");
        if protocol != "http" {
            continue;
        }
        let bind_text = listen
            .get("bind")
            .and_then(|value| value.as_str())
            .ok_or_else(|| ProximaError::Config("[[listen]] requires `bind`".into()))?;
        let bind: SocketAddr = bind_text
            .parse()
            .map_err(|err| ProximaError::Config(format!("invalid bind '{bind_text}': {err}")))?;
        if bind.port() == 0 {
            return Err(ProximaError::Config(
                "[[listen]] bind must use a concrete port for CLI readiness".into(),
            ));
        }
        let listener = TcpListener::bind(bind).map_err(|err| {
            ProximaError::Io(std::io::Error::other(format!(
                "reserve listener at {bind}: {err}"
            )))
        })?;
        reservations.push(listener);
    }
    drop(reservations);
    Ok(())
}

async fn run_serve_single(
    spec: Spec,
    addr: SocketAddr,
    mount_path: String,
) -> Result<(), ProximaError> {
    let mut app = App::new()?;
    let handle = app.pipe("__served__", spec).await?;
    app.mount(&mount_path, MountTarget::Handle(handle))?;

    let bind_addr = resolve_bind_addr(addr)?;

    // run_until_signal blocks until the listener is actually accepting.
    let shutdown = app.run_until_signal(RunConfig::http(bind_addr)).await?;
    println!("READY {bind_addr}");
    use std::io::Write as _;
    std::io::stdout()
        .flush()
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("flush stdout: {err}"))))?;

    let mut sigterm = signal(SignalKind::terminate()).map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("install sigterm: {err}")))
    })?;
    let mut sigint = signal(SignalKind::interrupt())
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("install sigint: {err}"))))?;
    tokio::select! {
        _ = sigterm.recv() => {}
        _ = sigint.recv() => {}
    }
    shutdown.stop();
    Ok(())
}

fn resolve_bind_addr(addr: SocketAddr) -> Result<SocketAddr, ProximaError> {
    let listener = TcpListener::bind(addr).map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!(
            "reserve listener at {addr}: {err}"
        )))
    })?;
    let resolved = listener.local_addr().map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!(
            "resolve listener at {addr}: {err}"
        )))
    })?;
    drop(listener);
    Ok(resolved)
}

async fn run_serve_full(spec: Spec) -> Result<(), ProximaError> {
    let mut app = App::new()?;
    let handles = app.load_full(spec).await?;
    if handles.is_empty() {
        return Err(ProximaError::Config(
            "load_full produced no listeners".into(),
        ));
    }

    // wait for every TCP-shaped bind to be reachable, then print READY
    // lines so test harnesses can sync. handles already own their
    // shutdown senders.
    for handle in &handles {
        let addr = handle.bind_addr();
        if let Some(addr) = addr {
            wait_until_listening(addr).await?;
            println!("READY {addr}");
        }
    }
    use std::io::Write as _;
    std::io::stdout()
        .flush()
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("flush stdout: {err}"))))?;

    let mut sigterm = signal(SignalKind::terminate()).map_err(|err| {
        ProximaError::Io(std::io::Error::other(format!("install sigterm: {err}")))
    })?;
    let mut sigint = signal(SignalKind::interrupt())
        .map_err(|err| ProximaError::Io(std::io::Error::other(format!("install sigint: {err}"))))?;
    tokio::select! {
        _ = sigterm.recv() => {}
        _ = sigint.recv() => {}
    }
    for handle in handles {
        handle.shutdown();
    }
    Ok(())
}

async fn wait_until_listening(addr: SocketAddr) -> Result<(), ProximaError> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(ProximaError::Io(std::io::Error::other(format!(
                "listener did not come up at {addr} within 5s"
            ))));
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

async fn run_call(
    config: PathBuf,
    method: String,
    path: String,
    body: Option<String>,
    headers: Vec<(String, String)>,
) -> Result<(), ProximaError> {
    let context = LoadContext::with_default_registry()?;
    let pipe = load(config, &context).await?;
    let mut builder = Request::builder().method(method.as_str()).path(path);
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    if let Some(text) = body {
        builder = builder.body(text);
    }
    let request = builder.build()?;
    let response = SendPipe::call(&pipe, request).await?;
    let status = response.status;
    let headers = response.metadata.clone();
    let body_bytes = response.collect_body().await?;
    let body_text = String::from_utf8_lossy(&body_bytes);
    println!("status: {status}");
    for (name, value) in &headers {
        println!(
            "header: {}: {}",
            String::from_utf8_lossy(name),
            String::from_utf8_lossy(value),
        );
    }
    print!("{body_text}");
    if !body_text.ends_with('\n') {
        println!();
    }
    Ok(())
}

async fn run_verify(
    spec_path: Option<PathBuf>,
    policy_path: Option<PathBuf>,
    format: String,
    strict: bool,
    repair: bool,
) -> ExitCode {
    // Discovery — at the project root when zero-arg.
    let cwd = match std::env::current_dir() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("proxima verify: cannot read current directory: {err}");
            return ExitCode::from(2);
        }
    };
    let spec_path = match spec_path.or_else(|| proxima::verify::discover::discover_spec(&cwd)) {
        Some(path) => path,
        None => {
            eprintln!(
                "proxima verify: no spec found. Looked for: proxima.toml, spec.toml, \
                 proxima.json, .proxima/spec.toml. Pass an explicit path."
            );
            return ExitCode::from(2);
        }
    };
    let policy_path = policy_path.or_else(|| proxima::verify::discover::discover_policy(&cwd));

    // Load spec as serde_json::Value (the walker's input).
    let spec_value = match load_spec_value(&spec_path) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("proxima verify: load spec {}: {err}", spec_path.display());
            return ExitCode::from(2);
        }
    };

    // Load policy if present; default to empty policy (run defaults).
    let policy = match policy_path.as_ref() {
        Some(path) => match proxima::verify::Policy::from_path(path) {
            Ok(policy) => policy,
            Err(err) => {
                eprintln!("proxima verify: load policy {}: {err}", path.display());
                return ExitCode::from(2);
            }
        },
        None => proxima::verify::Policy::default(),
    };

    let (report, blame) = if repair {
        let outcome = proxima::verify::repair_static(&spec_value, &policy);
        (outcome.after, outcome.blame)
    } else {
        (
            proxima::verify::verify_static(&spec_value, &policy),
            Vec::new(),
        )
    };

    match format.as_str() {
        "text" => {
            for item in &blame {
                println!("REPAIR dropped {item}");
            }
            print!("{}", report.emit_text());
        }
        "json" => {
            let entries_value = match serde_json::to_value(&report.entries) {
                Ok(value) => value,
                Err(err) => {
                    eprintln!("proxima verify: encode entries: {err}");
                    return ExitCode::from(2);
                }
            };
            let blame_strings: Vec<String> = blame.iter().map(ToString::to_string).collect();
            let doc = serde_json::json!({
                "entries": entries_value,
                "blame": blame_strings,
                "pass": report.pass_count(),
                "warn": report.warn_count(),
                "fail": report.fail_count(),
            });
            match serde_json::to_string_pretty(&doc) {
                Ok(text) => println!("{text}"),
                Err(err) => {
                    eprintln!("proxima verify: emit json: {err}");
                    return ExitCode::from(2);
                }
            }
        }
        other => {
            eprintln!("proxima verify: unknown --format '{other}' (expected text|json)");
            return ExitCode::from(2);
        }
    }

    let fail = report.fail_count();
    let warn = report.warn_count();
    if fail > 0 || (strict && warn > 0) {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn load_spec_value(path: &PathBuf) -> Result<Value, ProximaError> {
    let text = std::fs::read_to_string(path).map_err(ProximaError::Io)?;
    let registry = default_config_format_registry()?;
    let hint = path.extension().and_then(|ext| ext.to_str());
    registry.parse_with_hint(&text, hint)
}

async fn run_replay(
    recording_path: Option<PathBuf>,
    policy_path: Option<PathBuf>,
    spec_path: Option<PathBuf>,
    format: String,
    strict: bool,
    repair: bool,
) -> ExitCode {
    let cwd = match std::env::current_dir() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("proxima replay: cannot read current directory: {err}");
            return ExitCode::from(2);
        }
    };
    let recording = match recording_path
        .or_else(|| proxima::verify::discover::discover_newest_recording(&cwd).unwrap_or_default())
    {
        Some(path) => path,
        None => {
            eprintln!(
                "proxima replay: no recording found. Looked for newest *.bin under \
                 ./.proxima/recordings/ or ./recordings/. Pass --recording."
            );
            return ExitCode::from(2);
        }
    };
    // offline tooling runtime: the recording sources offload their blocking
    // file reads onto this runtime's background pool (the read-side seam off
    // tokio::fs). One core is plenty for a single-pass walk.
    let runtime = match proxima::offline_runtime() {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("proxima replay: build offline runtime: {err}");
            return ExitCode::from(2);
        }
    };
    let policy_path = policy_path.or_else(|| proxima::verify::discover::discover_policy(&cwd));
    let original_policy = match policy_path.as_ref() {
        Some(path) => match proxima::verify::Policy::from_path(path) {
            Ok(policy) => policy,
            Err(err) => {
                eprintln!("proxima replay: load policy {}: {err}", path.display());
                return ExitCode::from(2);
            }
        },
        None => proxima::verify::Policy::default(),
    };
    let spec_path = spec_path.or_else(|| proxima::verify::discover::discover_spec(&cwd));

    // Load the spec once if available — both byte_drift and
    // idempotence_contract need it.
    let spec_value = match spec_path.as_ref() {
        Some(path) => match load_spec_value(path) {
            Ok(value) => Some(value),
            Err(err) => {
                eprintln!("proxima replay: load spec {}: {err}", path.display());
                return ExitCode::from(2);
            }
        },
        None => None,
    };

    // Repair mode reverts the policy to whatever the recording can
    // actually support, *then* verifies against that policy. Blame
    // is emitted before the verify report.
    let (policy, blame) = if repair {
        match proxima::verify::repair_from_recording_file(
            &recording,
            spec_value.as_ref(),
            &original_policy,
            &runtime,
        )
        .await
        {
            Ok(outcome) => (outcome.repaired_policy, outcome.blame),
            Err(err) => {
                eprintln!("proxima replay: repair walk: {err}");
                return ExitCode::from(2);
            }
        }
    } else {
        (original_policy, Vec::new())
    };

    let mut report = match proxima::verify::verify_replay_with_spec(
        &recording,
        &policy,
        spec_value.as_ref(),
        &runtime,
    )
    .await
    {
        Ok(report) => report,
        Err(err) => {
            eprintln!(
                "proxima replay: walk recording {}: {err}",
                recording.display()
            );
            return ExitCode::from(2);
        }
    };

    // byte_drift: needs the spec to build the live pipe. If the
    // policy flags any byte_identical_pipes and we have a spec,
    // dispatch each recorded request through the live pipe and
    // diff bytes. Otherwise emit a WARN documenting the elision.
    if !policy.replay.byte_identical_pipes.is_empty() {
        match spec_value.as_ref() {
            Some(spec) => {
                let context = match proxima::LoadContext::with_default_registry() {
                    Ok(ctx) => ctx,
                    Err(err) => {
                        eprintln!("proxima replay: build LoadContext: {err}");
                        return ExitCode::from(2);
                    }
                };
                if let Err(err) = proxima::verify::verify_byte_drift(
                    &recording,
                    &policy,
                    spec,
                    &context,
                    &runtime,
                    &mut report,
                )
                .await
                {
                    eprintln!("proxima replay: byte_drift walk: {err}");
                    return ExitCode::from(2);
                }
            }
            None => {
                proxima::verify::skip_byte_drift_without_spec(&policy, &mut report);
            }
        }
    } else {
        proxima::verify::skip_byte_drift_without_spec(&policy, &mut report);
    }

    match format.as_str() {
        "text" => {
            for item in &blame {
                println!("REPAIR dropped {item}");
            }
            print!("{}", report.emit_text());
        }
        "json" => {
            let entries_value = match serde_json::to_value(&report.entries) {
                Ok(value) => value,
                Err(err) => {
                    eprintln!("proxima replay: encode entries: {err}");
                    return ExitCode::from(2);
                }
            };
            let blame_strings: Vec<String> = blame.iter().map(ToString::to_string).collect();
            let doc = serde_json::json!({
                "entries": entries_value,
                "blame": blame_strings,
                "pass": report.pass_count(),
                "warn": report.warn_count(),
                "fail": report.fail_count(),
            });
            match serde_json::to_string_pretty(&doc) {
                Ok(text) => println!("{text}"),
                Err(err) => {
                    eprintln!("proxima replay: emit json: {err}");
                    return ExitCode::from(2);
                }
            }
        }
        other => {
            eprintln!("proxima replay: unknown --format '{other}' (expected text|json)");
            return ExitCode::from(2);
        }
    }

    let fail = report.fail_count();
    let warn = report.warn_count();
    if fail > 0 || (strict && warn > 0) {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

async fn run_load(
    scenario_arg: Option<PathBuf>,
    name: Option<String>,
    json: bool,
    rps_override: Option<u64>,
    duration_override: Option<String>,
    remote: Option<String>,
    record: Option<PathBuf>,
) -> ExitCode {
    let cwd = match std::env::current_dir() {
        Ok(path) => path,
        Err(err) => {
            eprintln!("proxima load: cannot read current directory: {err}");
            return ExitCode::from(2);
        }
    };

    let scenario_path = match resolve_scenario_path(scenario_arg, name.as_deref(), &cwd) {
        Ok(path) => path,
        Err(code) => return code,
    };

    let mut scenario = match Scenario::from_toml_file(&scenario_path) {
        Ok(scenario) => scenario,
        Err(err) => {
            eprintln!("proxima load: parse {}: {err}", scenario_path.display());
            return ExitCode::from(2);
        }
    };

    if let Some(rps) = rps_override {
        scenario.workload.target_rps = Some(rps);
    }
    if let Some(raw) = duration_override.as_ref() {
        match parse_duration_override(raw) {
            Ok(duration_spec) => scenario.workload.duration = Some(duration_spec),
            Err(err) => {
                eprintln!("proxima load: --duration: {err}");
                return ExitCode::from(2);
            }
        }
    }

    let mode = match scenario.workload.mode() {
        Ok(mode) => mode,
        Err(err) => {
            eprintln!("proxima load: workload misconfigured: {err}");
            return ExitCode::from(2);
        }
    };

    if let Some(addr) = remote {
        eprintln!("proxima load --remote: not yet implemented (Stage 2). target would be {addr}");
        return ExitCode::from(2);
    }

    if let Some(path) = record.as_ref() {
        eprintln!(
            "proxima load: --record is a Stage 2 placeholder; \
             would write to {} — continuing without recording.",
            path.display()
        );
    }

    let context = match LoadContext::with_default_registry() {
        Ok(context) => context,
        Err(err) => {
            eprintln!("proxima load: build load context: {err}");
            return ExitCode::from(2);
        }
    };

    let (sink, printer) = if matches!(mode, WorkloadMode::OpenLoop) && !json {
        let (sender, receiver) = std::sync::mpsc::channel::<MetricsSnapshot>();
        let handle = std::thread::spawn(move || print_windows_to_stderr(receiver));
        (Some(sender), Some(handle))
    } else {
        (None, None)
    };

    let report = match run_scenario_with_sink(&scenario, &context, sink).await {
        Ok(report) => report,
        Err(err) => {
            eprintln!("proxima load: run {}: {err}", scenario_path.display());
            return ExitCode::from(2);
        }
    };

    if let Some(handle) = printer {
        let _ = handle.join();
    }

    if json {
        if let Err(err) = emit_json_report(&scenario_path, &report) {
            eprintln!("proxima load: emit json: {err}");
            return ExitCode::from(2);
        }
    } else {
        print_text_summary(&scenario_path, &report);
    }

    if report.passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn resolve_scenario_path(
    explicit: Option<PathBuf>,
    name: Option<&str>,
    cwd: &std::path::Path,
) -> Result<PathBuf, ExitCode> {
    if let Some(path) = explicit {
        return Ok(path);
    }
    if let Some(name) = name {
        let matches = discover_by_name(name, cwd);
        return match matches.len() {
            0 => {
                eprintln!("proxima load: no scenario named '{name}' found across search paths");
                Err(ExitCode::from(2))
            }
            1 => matches.into_iter().next().ok_or_else(|| {
                eprintln!("proxima load: internal inconsistency selecting scenario '{name}'");
                ExitCode::from(2)
            }),
            _ => {
                let listing = matches
                    .iter()
                    .map(|path| format!("  {}", path.display()))
                    .collect::<Vec<_>>()
                    .join("\n");
                eprintln!(
                    "proxima load: '--name {name}' matched multiple files:\n{listing}\n\
                     Pass an explicit path."
                );
                Err(ExitCode::from(2))
            }
        };
    }
    match discover_scenario(cwd) {
        Some(path) => Ok(path),
        None => {
            eprintln!(
                "proxima load: no scenario found. Looked for: proxima.scenario.toml, \
                 scenario.toml, .proxima/scenario.toml. Pass an explicit path or use --name."
            );
            Err(ExitCode::from(1))
        }
    }
}

fn parse_duration_override(raw: &str) -> Result<DurationSpec, String> {
    // ride the same serde visitor that scenario.toml uses, so the CLI
    // accepts exactly the same forms as the file.
    serde_json::from_value::<DurationSpec>(Value::String(raw.to_string()))
        .map_err(|err| format!("invalid duration `{raw}`: {err}"))
}

fn print_windows_to_stderr(receiver: std::sync::mpsc::Receiver<MetricsSnapshot>) {
    eprintln!(
        "{:>8}  {:>8}  {:>8}  {:>8}",
        "elapsed", "rps", "p50_ms", "p99_ms"
    );
    let mut prev_count: u64 = 0;
    let mut elapsed_secs: u64 = 0;
    while let Ok(snapshot) = receiver.recv() {
        elapsed_secs += 1;
        let (count, p50, p99) = co_latency_summary(&snapshot);
        let delta = count.saturating_sub(prev_count);
        eprintln!(
            "{:>8}  {:>8}  {:>8.1}  {:>8.1}",
            format!("{elapsed_secs}s"),
            delta,
            p50,
            p99
        );
        prev_count = count;
    }
}

fn co_latency_summary(snapshot: &MetricsSnapshot) -> (u64, f64, f64) {
    snapshot
        .histograms
        .iter()
        .find(|(name, _, _)| name == "proxima.workload.co_latency_ms")
        .map(|(_, _, summary)| (summary.count, summary.p50, summary.p99))
        .unwrap_or((0, 0.0, 0.0))
}

fn print_text_summary(scenario_path: &Path, report: &ScenarioReport) {
    let status = if report.passed() { "PASS" } else { "FAIL" };
    let total = report.completed.max(1);
    let success_rate = (report.successes as f64) / (total as f64) * 100.0;
    println!("{status}  {}", scenario_path.display());
    println!("  completed:  {}", report.completed);
    println!("  successes:  {} ({:.2}%)", report.successes, success_rate);
    println!("  failures:   {}", report.failures);
    if !report.windows.is_empty() {
        println!("  windows:    {} snapshots", report.windows.len());
    }
    if !report.failed_expectations.is_empty() {
        println!("  failed expectations:");
        for reason in &report.failed_expectations {
            println!("    - {reason}");
        }
    }
}

fn emit_json_report(
    scenario_path: &Path,
    report: &ScenarioReport,
) -> Result<(), serde_json::Error> {
    let envelope = serde_json::json!({
        "scenario_path": scenario_path.display().to_string(),
        "passed": report.passed(),
        "completed": report.completed,
        "successes": report.successes,
        "failures": report.failures,
        "failed_expectations": report.failed_expectations,
        "windows": report.windows,
        "metrics_snapshot": report.metrics_snapshot,
    });
    let text = serde_json::to_string(&envelope)?;
    println!("{text}");
    Ok(())
}
