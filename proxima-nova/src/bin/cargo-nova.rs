//! `cargo nova run`: compile a single-package, zero-dependency project with the
//! nova codegen backend and run the result. Never calls `cargo`.
//!
//! Composes `proxima_process::CommandPipe` (unsandboxed chain: `grounds::Empty`)
//! driven by `prime::os::runtime::PrimeRuntime::block_on`. The child's end is read
//! from the response stream's trailers after the body drains (`exit_trailer.rs`).
use std::env::{args, current_dir};
use std::ffi::CString;
use std::fs::{create_dir_all, read_to_string};
use std::io::{Write, stdout};
use std::path::Path;
use std::process::ExitCode;

use anyhow::Context;
use bytes::Bytes;
use conflaguration::Settings;
use futures::StreamExt;
use prime::os::runtime::PrimeRuntime;
use proxima_primitives::pipe::{ProximaError, Request, SendPipe};
use proxima_process::grounds::Empty;
use proxima_process::{CommandDescriptor, CommandPipe};
use thiserror::Error;

const TOOLCHAIN: &str = "+nightly-2026-11-15";
const SIGNAL_EXIT_BASE: i32 = 128;

/// Env-backed config: `NOVA_BACKEND` is the codegen backend dylib path.
#[derive(Debug, Clone, Default, Settings)]
#[settings(prefix = "NOVA")]
struct NovaSettings {
    #[setting(default = "")]
    backend: String,
}

#[derive(Debug, Error)]
enum NovaError {
    #[error("usage: cargo nova run")]
    Usage,
    #[error("NOVA_BACKEND is not set: point it at the nova codegen backend dylib")]
    MissingBackend,
    #[error("src/main.rs is required in the current directory")]
    MissingMain,
    #[error("Cargo.toml has no [package] {0}")]
    MissingField(&'static str),
    #[error("argument contains a NUL byte: {0}")]
    Nul(String),
    #[error("spawn failed: {0}")]
    Spawn(ProximaError),
    #[error("child response carried no stream")]
    NoStream,
}

/// How a child ended, read from the `exit-status` / `exit-signal` trailers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ended {
    Code(i32),
    Signal(i32),
    Unreported,
}

impl Ended {
    fn from_trailers(status: Option<&str>, signal: Option<&str>) -> Self {
        let parse = |text: Option<&str>| text.and_then(|value| value.parse::<i32>().ok());
        match (parse(status), parse(signal)) {
            (Some(code), _) => Self::Code(code),
            (None, Some(number)) => Self::Signal(number),
            (None, None) => Self::Unreported,
        }
    }

    fn process_code(self) -> i32 {
        match self {
            Self::Code(code) => code,
            Self::Signal(number) => SIGNAL_EXIT_BASE + number,
            Self::Unreported => 1,
        }
    }
}

struct Package {
    name: String,
    edition: String,
}

fn read_package(directory: &Path) -> anyhow::Result<Package> {
    if !directory.join("src/main.rs").is_file() {
        return Err(NovaError::MissingMain.into());
    }
    let text = read_to_string(directory.join("Cargo.toml")).context("reading Cargo.toml")?;
    let table: toml::Table = text.parse().context("parsing Cargo.toml")?;
    let package = table.get("package").and_then(toml::Value::as_table);
    let field = |key: &'static str| {
        package
            .and_then(|section| section.get(key))
            .and_then(toml::Value::as_str)
            .map(str::to_owned)
    };
    Ok(Package {
        name: field("name").ok_or(NovaError::MissingField("name"))?,
        edition: field("edition").unwrap_or_else(|| "2015".to_owned()),
    })
}

fn rustc_args(package: &Package, backend: &str, output: &str) -> Vec<String> {
    let mut arguments: Vec<String> = [
        TOOLCHAIN,
        "--edition",
        &package.edition,
        "--crate-name",
        &package.name,
        "--crate-type",
        "bin",
        "-C",
        "panic=abort",
        &format!("-Zcodegen-backend={backend}"),
        "src/main.rs",
        "-o",
        output,
    ]
    .iter()
    .map(|argument| (*argument).to_owned())
    .collect();
    // macOS links a no_std no_main bin with -nodefaultlibs, leaving dyld_stub_binder undefined
    if cfg!(target_os = "macos") {
        arguments.extend(["-C".to_owned(), "link-arg=-lSystem".to_owned()]);
    }
    arguments
}

fn descriptor(program: &str, arguments: &[String]) -> Result<CommandDescriptor, NovaError> {
    let cstring = |text: &str| CString::new(text).map_err(|_| NovaError::Nul(text.to_owned()));
    let mut command = CommandDescriptor::new(cstring(program)?);
    for argument in arguments {
        command.arg(cstring(argument)?);
    }
    command.inherit_current_env();
    Ok(command)
}

/// Spawn through `CommandPipe`, forward the child's stdout, drain the body, then read trailers.
async fn spawn_and_wait(command: CommandDescriptor) -> Result<Ended, NovaError> {
    let pipe = CommandPipe::builder()
        .command(command)
        .dispatch(Empty::new())
        .build();
    let request = Request::<Bytes>::builder()
        .method("POST")
        .path("/")
        .build()
        .map_err(NovaError::Spawn)?;
    let response = SendPipe::call(&pipe, request)
        .await
        .map_err(NovaError::Spawn)?;
    let mut stream = response.stream.ok_or(NovaError::NoStream)?;
    let mut stdout = stdout();
    while let Some(chunk) = stream.chunk_stream_mut().next().await {
        let bytes = chunk.map_err(NovaError::Spawn)?;
        let _ = stdout.write_all(&bytes);
    }
    let _ = stdout.flush();
    Ok(stream.trailers().map_or(Ended::Unreported, |list| {
        Ended::from_trailers(list.get_str("exit-status"), list.get_str("exit-signal"))
    }))
}

fn run_child(runtime: &PrimeRuntime, command: CommandDescriptor) -> anyhow::Result<Ended> {
    runtime
        .block_on(spawn_and_wait(command))
        .map_err(|error| anyhow::anyhow!("prime block_on failed: {error}"))?
        .map_err(Into::into)
}

fn run() -> anyhow::Result<i32> {
    let mut arguments = args().skip(1).peekable();
    // cargo invokes `cargo-nova nova run`; a direct `cargo-nova run` works too
    if arguments.peek().map(String::as_str) == Some("nova") {
        arguments.next();
    }
    if arguments.next().as_deref() != Some("run") {
        return Err(NovaError::Usage.into());
    }
    let settings = NovaSettings::from_env().context("loading NOVA_* settings")?;
    if settings.backend.is_empty() {
        return Err(NovaError::MissingBackend.into());
    }
    let package = read_package(Path::new("."))?;
    let output = format!("target/nova/debug/{}", package.name);
    create_dir_all("target/nova/debug").context("creating target/nova/debug")?;

    let runtime = PrimeRuntime::new(1).context("starting prime runtime")?;
    let compile = descriptor("rustc", &rustc_args(&package, &settings.backend, &output))?;
    let compiled = run_child(&runtime, compile)?;
    if compiled != Ended::Code(0) {
        eprintln!("cargo-nova: rustc did not exit 0: {compiled:?}");
        return Ok(compiled.process_code().max(1));
    }
    let absolute = current_dir().context("reading current dir")?.join(&output);
    let program = absolute.to_str().context("program path is not UTF-8")?;
    Ok(run_child(&runtime, descriptor(program, &[])?)?.process_code())
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(u8::try_from(code & 0xff).unwrap_or(1)),
        Err(error) => {
            eprintln!("cargo-nova: {error:#}");
            ExitCode::from(2)
        }
    }
}
