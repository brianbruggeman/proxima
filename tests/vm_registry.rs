//! `[upstreams.x] type = "vm"` in a TOML file -- with no per-function Rust
//! anywhere -- resolves through `proxima`'s `PipeFactoryRegistry` to a live
//! `PipeHandle` that boots a real ELF guest under a real
//! hypervisor, feeds it the request body, and returns the bytes it wrote as
//! the response body (`tools/proxima-vm/ROADMAP.md`'s M2 exit criterion 1).
//!
//! This lives in the `proxima` crate (the owner of the `vm` upstream factory,
//! `src/upstreams/vm.rs`), not in `tools/proxima-vm/tests`: a `proxima-vm`
//! test target that dev-depends on `proxima` with its `vm` feature links
//! `proxima-vm` a second time into that crate's own unit-test binary, and the
//! `#[no_mangle]` trampolines (`proxima_vm_dispatch_hypercall`, ...) then
//! collide (`rust-lld: duplicate symbol`). A `[[bin]]` cannot host it either --
//! it only sees `[dependencies]`. So this file owns `fn main()` (`harness =
//! false`, `Cargo.toml`) and re-executes ITSELF as the signed
//! hypervisor-entitled child, dispatching on argv:
//! `--vm-registry-probe-child <config-path> <pipe-name>` runs the App/
//! registry/dispatch path with stdin as the request body and prints the
//! response body to stdout; `--vm-registry-bench-child ...` times repeated
//! calls; no argv runs the actual test cases, each of which builds the real
//! guest ELFs, writes a TOML config naming one, signs a copy of this very
//! executable, and drives it as the child.
//!
//! Hypervisor.framework answers `hv_vm_create` with `HV_DENIED` for a
//! process lacking the `com.apple.security.hypervisor` entitlement, and
//! applying it is a post-link `codesign` step -- same constraint
//! `tests/boot.rs`'s module doc explains for `SignedGuest`.
//!
//! `harness = false` hands this file `fn main()` outright, which would
//! normally make nextest unable to enumerate/run it (it drives the binary
//! with libtest's `--list`/`--format terse` protocol). `libtest_mimic`
//! reimplements just enough of that protocol to keep nextest's discovery
//! working while `main` still owns the child-dispatch branch below it.

// the workspace lint set denies expect_used everywhere; test files opt out
// the same way `tests/boot.rs`/`tests/dispatch_hypercall.rs` do.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::env;
use std::error::Error;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use bytes::Bytes;
use libtest_mimic::{Arguments, Trial};
use tempfile::TempDir;

const ENTITLEMENTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tools/proxima-vm/entitlements.plist"
);
const GUEST_MANIFEST_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tools/proxima-vm/guests/lambda"
);
const REPO_TEXT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/LICENSE-APACHE");
const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));
const PROBE_FLAG: &str = "--vm-registry-probe-child";
const SERVE_FLAG: &str = "--vm-registry-serve-child";
const LAMBDA_EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/lambda/proxima.toml");
const BENCH_FLAG: &str = "--vm-registry-bench-child";
const INSTRUMENT_FLAG: &str = "--vm-registry-instrument-child";
const ELF_GUEST: &str = "uppercase";
const PIPE: &str = "shout";
const TWO_MIB: u64 = 2 * 1024 * 1024;
const SIXTY_FOUR_MIB: u64 = 64 * 1024 * 1024;
const LATENCY_SAMPLES: usize = 200;
const BASELINE_WARMUP: usize = 10;
const BENCH_BODY: &str = "hello, lambda";
const LINUX_GUEST_MANIFEST_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tools/proxima-vm/guests/linux-init"
);
const LINUX_TARGET: &str = "x86_64-unknown-linux-musl";
const LINUX_GUEST: &str = "uppercase-init";
const LINUX_PIPE: &str = "shout-linux";
const IMAGE_PIPE: &str = "probe-image";
const DEFAULT_KERNEL: &str = "/boot/vmlinuz-linux";
const RESIDENT_FLAG: &str = "--vm-registry-resident-child";
const RESIDENT_PIPE: &str = "resident-whoami";
const WHOAMI_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tools/proxima-vm/tests/fixtures/oci/whoami-v1.10.2"
);
const WHOAMI_INDEX_DIGEST: &str =
    "sha256:1474027c316661cdec87df2623e13a41e7e1ce0ba99c24917631de8f300b5420";
const WHOAMI_BLOBS: [&str; 6] = [
    WHOAMI_INDEX_DIGEST,
    "sha256:7000846753fcc36eb2a1a3a21fe897da09c71e1d3b5381f912ea6e6e1c8871b6",
    "sha256:ab0ca2603e47e42f7d45e094b6f4cecc5d5b7b0ef6ca89170ce1a771ed1a0066",
    "sha256:1b5c60a1be3bf9cb17566bdad4cb1f36387367e56798bc307d394169f1e5d72d",
    "sha256:1d12c9005bc5cd6824edcfe37f412c35922508f76cb4b854beb52f85c12a929b",
    "sha256:0abc37d8af53c11b5d94e3b54eca06ac582fec2a9b56f7d3070bb54247aa8595",
];
const RESIDENT_REQUESTS: usize = 3;

const MAX_SEGMENTS: usize = 8;
const OUTPUT_CAPACITY: usize = 64 * 1024;
const MAX_HYPERCALLS: usize = 16;

fn main() {
    // the child-dispatch branch reads its own positional args, not
    // libtest_mimic's flags -- intercepted before `Arguments::from_args`
    // ever sees argv, so a real `cargo test`/nextest invocation (which
    // never passes the child flags) is unaffected.
    let arguments: Vec<String> = env::args().skip(1).collect();
    let outcome = match arguments.first().map(String::as_str) {
        Some(PROBE_FLAG) => Some(run_probe_child(&arguments[1], &arguments[2])),
        Some(SERVE_FLAG) => Some(run_serve_child(&arguments[1], &arguments[2])),
        Some(RESIDENT_FLAG) => Some(run_resident_child(&arguments[1], &arguments[2])),
        Some(BENCH_FLAG) => Some(run_bench_child(&arguments[1..])),
        Some(INSTRUMENT_FLAG) => Some(run_instrument_child(&arguments[1..])),
        _ => None,
    };
    if let Some(outcome) = outcome {
        match outcome {
            Ok(()) => return,
            Err(error) => {
                eprintln!("child failed: {error}");
                std::process::exit(1);
            }
        }
    }

    let linux_trials = if KVM_LANE {
        vec![
            trial(
                "config_only_linux_vm_upstream_uppercases_the_request_body",
                config_only_linux_vm_upstream_uppercases_the_request_body,
            ),
            trial(
                "relative_linux_init_path_boots_from_a_different_working_directory",
                relative_linux_init_path_boots_from_a_different_working_directory,
            ),
            trial(
                "config_only_linux_vm_upstream_runs_an_oci_image",
                config_only_linux_vm_upstream_runs_an_oci_image,
            ),
            trial(
                "a_tampered_cached_layer_fails_the_image_upstream_build_before_boot",
                a_tampered_cached_layer_fails_the_image_upstream_build_before_boot,
            ),
            trial(
                "a_resident_vm_upstream_boots_one_guest_and_serves_sequential_gets",
                a_resident_vm_upstream_boots_one_guest_and_serves_sequential_gets,
            ),
            trial(
                "a_resident_elf_guest_is_rejected_when_the_config_is_applied",
                a_resident_elf_guest_is_rejected_when_the_config_is_applied,
            ),
            trial(
                "missing_kernel_fails_when_the_config_is_applied",
                missing_kernel_fails_when_the_config_is_applied,
            ),
            trial(
                "lambda_example_config_serves_both_guests_over_http",
                lambda_example_config_serves_both_guests_over_http,
            ),
        ]
    } else {
        Vec::new()
    };
    let mut trials = if HYPERVISOR_LANE || KVM_LANE {
        vec![
            trial(
                "config_only_vm_upstream_uppercases_the_request_body",
                config_only_vm_upstream_uppercases_the_request_body,
            ),
            trial(
                "relative_elf_guest_path_boots_from_a_different_working_directory",
                relative_elf_guest_path_boots_from_a_different_working_directory,
            ),
            trial(
                "relative_elf_guest_path_nested_under_record_boots_from_a_different_working_directory",
                relative_elf_guest_path_nested_under_record_boots_from_a_different_working_directory,
            ),
            trial(
                "relative_elf_guest_path_serves_from_a_different_working_directory",
                relative_elf_guest_path_serves_from_a_different_working_directory,
            ),
            trial(
                "missing_relative_guest_error_names_the_resolved_path",
                missing_relative_guest_error_names_the_resolved_path,
            ),
            trial(
                "multi_chunk_input_round_trips_uppercased_byte_exact",
                multi_chunk_input_round_trips_uppercased_byte_exact,
            ),
            trial(
                "empty_input_yields_empty_output",
                empty_input_yields_empty_output,
            ),
            trial(
                "output_past_the_configured_capacity_is_a_named_error",
                output_past_the_configured_capacity_is_a_named_error,
            ),
            trial(
                "guest_memory_too_small_for_the_image_is_a_named_error",
                guest_memory_too_small_for_the_image_is_a_named_error,
            ),
            trial(
                "non_elf_image_fails_when_the_config_is_applied",
                non_elf_image_fails_when_the_config_is_applied,
            ),
            trial(
                "exhausted_hypercall_budget_is_a_named_error",
                exhausted_hypercall_budget_is_a_named_error,
            ),
            trial(
                "input_past_the_guest_buffer_fails_instead_of_truncating",
                input_past_the_guest_buffer_fails_instead_of_truncating,
            ),
            trial(
                "vm_upstream_latency_table",
                vm_upstream_latency_table,
            ),
        ]
    } else {
        Vec::new()
    };

    trials.extend(linux_trials);
    libtest_mimic::run(&Arguments::from_args(), trials).exit();
}

fn trial(name: &str, case: fn()) -> Trial {
    Trial::test(name, move || {
        case();
        Ok(())
    })
}

fn read_stdin() -> Result<Vec<u8>, Box<dyn Error>> {
    let mut body = Vec::new();
    std::io::stdin().read_to_end(&mut body)?;
    Ok(body)
}

/// Builds the `App` from a TOML file exactly as a deployment would. Reads
/// NOTHING guest- or function-specific from Rust -- only `config_path`/
/// `pipe_name` from argv -- which is the config-only claim under test. The
/// process installs a runtime first, as `#[proxima::main]` does, so the
/// factory wraps the pipe in `Offload` and the VM never runs on a tokio
/// worker.
async fn build_pipe(
    config_path: &str,
    pipe_name: &str,
) -> Result<proxima::PipeHandle, Box<dyn Error>> {
    let settings = proxima::settings::ProximaSettings::from_path(config_path)?;
    let mut app = proxima::App::builder().with_defaults()?.build()?;
    app.apply_settings(&settings).await?;
    app.lookup_pipe(pipe_name)
        .ok_or_else(|| format!("pipe `{pipe_name}` not registered by config").into())
}

fn request_with(body: &[u8]) -> Result<proxima::Request<Bytes>, Box<dyn Error>> {
    Ok(proxima::Request::builder()
        .method("POST")
        .path("/")
        .body(Bytes::copy_from_slice(body))
        .build()?)
}

fn child_runtime(offload: bool) -> Result<tokio::runtime::Runtime, Box<dyn Error>> {
    if offload {
        proxima::runtime::install_runtime(proxima::runtime::RuntimeSelection::prime(2)?);
    }
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

fn http_post(addr: SocketAddr, path: &str, body: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut stream = std::net::TcpStream::connect(addr)?;
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("response has no header terminator"))?;
    if !response.starts_with(b"HTTP/1.1 200") {
        return Err(std::io::Error::other(format!(
            "non-200 response: {}",
            String::from_utf8_lossy(&response)
        )));
    }
    Ok(response[split + 4..].to_vec())
}

/// args: `<config> <mount-path>`. Binds every `[[listen]]` of a multi-block
/// config the way `proxima serve` does, POSTs stdin to the mount over a real
/// socket, and prints the response body.
fn run_serve_child(config_path: &str, mount_path: &str) -> Result<(), Box<dyn Error>> {
    let body = read_stdin()?;
    let runtime = child_runtime(true)?;
    runtime.block_on(async move {
        let mut app = proxima::App::builder().with_defaults()?.build()?;
        let handles = app.load_full(config_path).await?;
        let addr = handles
            .first()
            .and_then(|handle| handle.bind_addr())
            .ok_or("listener reports no bind address")?;
        let mount_path = mount_path.to_string();
        let response =
            tokio::task::spawn_blocking(move || http_post(addr, &mount_path, &body)).await??;
        std::io::stdout().write_all(&response)?;
        for handle in handles {
            handle.stop().await;
        }
        Ok::<(), Box<dyn Error>>(())
    })
}

fn get_request() -> Result<proxima::Request<Bytes>, Box<dyn Error>> {
    Ok(proxima::Request::builder()
        .method("GET")
        .path("/")
        .build()?)
}

/// args: `<config> <pipe>`. Builds the resident upstream once, sends
/// `RESIDENT_REQUESTS` sequential GETs through its `PipeHandle`, and prints one
/// `status=<code>` line per response, the first response's body, and
/// `boots=<n>`: the count of guests this process started.
fn run_resident_child(config_path: &str, pipe_name: &str) -> Result<(), Box<dyn Error>> {
    let runtime = child_runtime(true)?;
    runtime.block_on(async move {
        let handle = build_pipe(config_path, pipe_name).await?;
        let mut report = String::new();
        for index in 0..RESIDENT_REQUESTS {
            let response =
                proxima_primitives::pipe::SendPipe::call(&handle, get_request()?).await?;
            report.push_str(&format!("status={}\n", response.status));
            let payload = response.collect_body().await?;
            if index == 0 {
                report.push_str(&format!("body={}\n", String::from_utf8_lossy(&payload)));
            }
        }
        report.push_str(&format!(
            "boots={}\n",
            proxima::upstreams::vm_resident::guests_booted()
        ));
        std::io::stdout().write_all(report.as_bytes())?;
        Ok::<(), Box<dyn Error>>(())
    })
}

fn run_probe_child(config_path: &str, pipe_name: &str) -> Result<(), Box<dyn Error>> {
    let body = read_stdin()?;
    let runtime = child_runtime(true)?;
    runtime.block_on(async move {
        let handle = build_pipe(config_path, pipe_name).await?;
        let response =
            proxima_primitives::pipe::SendPipe::call(&handle, request_with(&body)?).await?;
        let payload = response.collect_body().await?;
        std::io::stdout().write_all(&payload)?;
        Ok::<(), Box<dyn Error>>(())
    })
}

/// args: `<config> <pipe> <samples> <warmup> <offload|inline>`. Prints one
/// `latency_nanos <n>` line per timed call through the pipe. `offload`
/// installs a prime runtime first (the factory then wraps the pipe in
/// `Offload`); `inline` leaves none installed, so the VM runs on the polling
/// thread.
fn run_bench_child(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let [config_path, pipe_name, samples, warmup, placement] = arguments else {
        return Err("bench child expects 5 args".into());
    };
    let samples: usize = samples.parse()?;
    let warmup: usize = warmup.parse()?;
    let body = read_stdin()?;
    let expected = body.to_ascii_uppercase();

    let runtime = child_runtime(placement == "offload")?;
    let latencies = runtime.block_on(async {
        let handle = build_pipe(config_path, pipe_name).await?;
        let mut nanos = Vec::with_capacity(samples);
        for index in 0..warmup + samples {
            let request = request_with(&body)?;
            let start = Instant::now();
            let response = proxima_primitives::pipe::SendPipe::call(&handle, request).await?;
            let elapsed = start.elapsed();
            if response.payload.trim_ascii_end() != expected.trim_ascii_end() {
                return Err(format!(
                    "call {index} returned {:?}, expected {:?}",
                    response.payload, expected
                )
                .into());
            }
            if index >= warmup {
                nanos.push(elapsed.as_nanos());
            }
        }
        Ok::<Vec<u128>, Box<dyn Error>>(nanos)
    })?;
    for nanos in latencies {
        println!("latency_nanos {nanos}");
    }
    Ok(())
}

/// args: `<guest-elf> <guest-memory-size> <samples>`. Runs
/// `run_dispatch_loop` directly in a process with no runtime threads and
/// prints one `create_to_first_exit_nanos` and `touch_all_pages_nanos` line
/// per run: the M3 instrument numbers the upstream itself discards.
fn run_instrument_child(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let [guest_elf, guest_memory_size, samples] = arguments else {
        return Err("instrument child expects 3 args".into());
    };
    report_instrument(
        Path::new(guest_elf),
        BENCH_BODY.as_bytes(),
        guest_memory_size.parse()?,
        samples.parse()?,
    )
}

fn report_instrument(
    guest_elf: &Path,
    input: &[u8],
    guest_memory_size: u64,
    samples: usize,
) -> Result<(), Box<dyn Error>> {
    let image = std::fs::read(guest_elf)?;
    let (entry, segments) = proxima_vm::elf::parse_elf::<MAX_SEGMENTS>(&image)
        .map_err(|error| format!("parse {}: {error}", guest_elf.display()))?;
    for _ in 0..samples {
        let start = Instant::now();
        let observed = proxima_vm::dispatch::run_dispatch_loop(
            entry,
            &segments,
            input,
            OUTPUT_CAPACITY,
            MAX_HYPERCALLS,
            256,
            256,
            256,
            2048,
            256,
            guest_memory_size,
        )?;
        println!("run_dispatch_loop_nanos {}", start.elapsed().as_nanos());
        println!(
            "create_to_first_exit_nanos {}",
            observed.create_to_first_exit_nanos
        );
        println!("touch_all_pages_nanos {}", observed.touch_all_pages_nanos);
    }
    Ok(())
}

/// Builds the lambda guest crate's bins for this lane's bare-metal target
/// into a per-case tempdir's own `CARGO_TARGET_DIR` -- same discipline
/// `tools/proxima-vm/tests/dispatch_hypercall.rs`'s `build_guest_elfs` uses,
/// copied rather than shared because that helper lives in a `harness = true`
/// target this file (`harness = false`) cannot import as a sibling module.
fn build_guest_elf(target_dir: &Path, bin: &str) -> PathBuf {
    let target = if KVM_LANE {
        "x86_64-unknown-none"
    } else {
        "aarch64-unknown-none"
    };
    let status = Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            &format!("{GUEST_MANIFEST_DIR}/Cargo.toml"),
            "--target",
            target,
            "--release",
            "--bin",
            bin,
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("run cargo build for the guest ELF");
    assert!(status.success(), "guest ELF build failed");
    target_dir.join(target).join("release").join(bin)
}

/// A signed copy of this very test executable, ready to run as a child.
struct SignedSelf {
    _directory: TempDir,
    path: PathBuf,
}

impl SignedSelf {
    fn prepare() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed self-copy");
        let path = directory.path().join("vm-registry-probe");
        let current_exe = env::current_exe().expect("resolve this executable's own path");
        std::fs::copy(&current_exe, &path).expect("copy this executable");

        if HYPERVISOR_LANE {
            let status = Command::new("codesign")
                .arg("--force")
                .arg("--sign")
                .arg("-")
                .arg("--entitlements")
                .arg(ENTITLEMENTS)
                .arg(&path)
                .status()
                .expect("run codesign");
            assert!(status.success(), "codesign failed for {}", path.display());
        }

        Self {
            _directory: directory,
            path,
        }
    }

    fn run(&self, arguments: &[&str], stdin_body: &[u8]) -> Result<Vec<u8>, String> {
        self.run_from(None, arguments, stdin_body)
    }

    fn run_from(
        &self,
        working_directory: Option<&Path>,
        arguments: &[&str],
        stdin_body: &[u8],
    ) -> Result<Vec<u8>, String> {
        let mut command = Command::new(&self.path);
        command
            .args(arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(directory) = working_directory {
            command.current_dir(directory);
        }
        let mut child = command.spawn().expect("spawn the signed self-copy");
        child
            .stdin
            .take()
            .expect("child stdin is piped")
            .write_all(stdin_body)
            .expect("write the request body to the child");
        let output = child.wait_with_output().expect("wait for the child");
        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(String::from_utf8_lossy(&output.stderr).into_owned())
        }
    }

    fn call(&self, config_path: &Path, body: &[u8]) -> Result<Vec<u8>, String> {
        let config = config_path.to_string_lossy();
        self.run(&[PROBE_FLAG, &config, PIPE], body)
    }
}

/// One TOML-only upstream: `extra` is raw `key = value` lines appended to the
/// `[upstreams.shout]` table.
fn write_vm_upstream_config(directory: &Path, guest_elf: &Path, extra: &str) -> PathBuf {
    let config_path = directory.join("proxima.toml");
    // `ProximaSettings.upstreams` is the field name (`src/settings/mod.rs`);
    // `[upstreams.shout]` is the TOML table key it deserializes from --
    // `tests/units/settings_to_app.rs` pins the same plural spelling.
    let toml = format!(
        "[upstreams.{PIPE}]\ntype = \"vm\"\nguest_image_path = \"{}\"\n{extra}",
        guest_elf.display()
    );
    std::fs::write(&config_path, toml).expect("write TOML config");
    config_path
}

struct Fixture {
    directory: TempDir,
    guest_elf: PathBuf,
    signed_self: SignedSelf,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the guest build");
        let guest_elf = build_guest_elf(&directory.path().join("guest-target"), ELF_GUEST);
        Self {
            directory,
            guest_elf,
            signed_self: SignedSelf::prepare(),
        }
    }

    fn config(&self, extra: &str) -> PathBuf {
        write_vm_upstream_config(self.directory.path(), &self.guest_elf, extra)
    }
}

/// Exit proof a: a function registered purely via TOML -- zero new Rust --
/// uppercases the request body. `shout` names nothing this test file,
/// `VmPipeFactory`, or `VmConfig` hard-codes; it is entirely the TOML's
/// choice.
fn config_only_vm_upstream_uppercases_the_request_body() {
    let fixture = Fixture::new();
    let config = fixture.config("");

    let response = fixture
        .signed_self
        .call(&config, b"hello, lambda")
        .expect("config-registered vm upstream runs end to end");

    assert_eq!(
        response, b"HELLO, LAMBDA",
        "a TOML-only `type = \"vm\"` upstream must boot a real guest through a \
         real VM exit and return the uppercased request body"
    );
}

/// Exit proof b: an input of several KiB crosses many 256-byte read chunks
/// and write chunks, and still round-trips byte-exact.
fn multi_chunk_input_round_trips_uppercased_byte_exact() {
    let text = std::fs::read(REPO_TEXT).expect("read the repo's LICENSE-APACHE");
    assert!(text.len() > 8 * 1024, "the corpus must span many chunks");
    let fixture = Fixture::new();
    let config = fixture.config("max_hypercalls = 256\n");

    let response = fixture
        .signed_self
        .call(&config, &text)
        .expect("multi-chunk call");

    assert_eq!(response.len(), text.len());
    assert_eq!(response, text.to_ascii_uppercase());
    assert_ne!(response, text, "the corpus must contain lowercase to change");
}

/// Exit proof c, the degenerate control: nothing in, nothing out.
fn empty_input_yields_empty_output() {
    let fixture = Fixture::new();
    let config = fixture.config("");

    let response = fixture
        .signed_self
        .call(&config, b"")
        .expect("empty-body call");

    assert!(response.is_empty(), "empty input gave {response:?}");
}

/// Exit proof d: an output past `output_capacity` fails the call; it is
/// never silently truncated to the first 8 bytes.
fn output_past_the_configured_capacity_is_a_named_error() {
    let fixture = Fixture::new();
    let config = fixture.config("output_capacity = 8\n");

    let error = fixture
        .signed_self
        .call(&config, b"hello, lambda")
        .expect_err("13 bytes of output cannot fit an 8 byte capacity");

    assert!(
        error.contains("guest output exceeds output_capacity of 8 bytes"),
        "the failure must name the capacity: {error}"
    );
}

/// Exit proof d: a `guest_memory_size` below what the image's segments need
/// fails the call, naming both numbers.
fn guest_memory_too_small_for_the_image_is_a_named_error() {
    let fixture = Fixture::new();
    let config = fixture.config("guest_memory_size = 65536\n");

    let error = fixture
        .signed_self
        .call(&config, b"hello")
        .expect_err("a 64 KiB guest cannot hold the elf image");

    assert!(
        error.contains("guest_memory_size 65536 is too small")
            && error.contains("stack need 335872 bytes"),
        "the failure must name the configured size and what the image needs: {error}"
    );
}

/// Exit proof d: an image that is not an ELF is refused when the config is
/// applied, before any request arrives.
fn non_elf_image_fails_when_the_config_is_applied() {
    let fixture = Fixture::new();
    let not_elf = fixture.directory.path().join("broken.elf");
    std::fs::write(&not_elf, b"#!/bin/sh\necho not an elf\n").expect("write the non-ELF image");
    let config = write_vm_upstream_config(fixture.directory.path(), &not_elf, "");

    let error = fixture
        .signed_self
        .call(&config, b"hello")
        .expect_err("a shell script is not a guest image");

    assert!(
        error.contains("vm `shout`") && error.contains("parse guest ELF"),
        "the failure must come from the factory build and name the parse stage: {error}"
    );
}

fn exhausted_hypercall_budget_is_a_named_error() {
    let fixture = Fixture::new();
    let config = fixture.config("max_hypercalls = 2\n");

    let error = fixture
        .signed_self
        .call(&config, b"hello, lambda")
        .expect_err("read, write and halt need three hypercalls");

    assert!(
        error.contains("exceeded hypercall budget"),
        "the failure must name the budget: {error}"
    );
}

/// The guest's input buffer is 256 KiB; a larger body must fail the run (the
/// guest reports `InputTooLarge`, diagnostic bytes `[04, 00]`), never be
/// truncated and answered.
fn input_past_the_guest_buffer_fails_instead_of_truncating() {
    let fixture = Fixture::new();
    let config = fixture.config("max_hypercalls = 4096\n");

    let error = fixture
        .signed_self
        .call(&config, &vec![b'a'; 300 * 1024])
        .expect_err("300 KiB cannot fit the guest's 256 KiB input buffer");

    assert!(
        error.contains("guest emitted [04, 00] before failing"),
        "the guest's InputTooLarge diagnostic must reach the host error: {error}"
    );
}

fn host_kernel() -> String {
    env::var("PROXIMA_VM_KERNEL").unwrap_or_else(|_| DEFAULT_KERNEL.to_string())
}

/// Builds the Linux guest crate for musl into `target_dir` and returns the
/// static `uppercase-init` executable, the same build
/// `tools/proxima-vm/tests/linux_guest.rs` drives.
fn build_linux_init(target_dir: &Path) -> PathBuf {
    build_linux_bin(target_dir, LINUX_GUEST)
}

fn build_linux_bin(target_dir: &Path, bin: &str) -> PathBuf {
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            LINUX_TARGET,
            "--manifest-path",
            &format!("{LINUX_GUEST_MANIFEST_DIR}/Cargo.toml"),
            "--bin",
            bin,
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("run cargo build for the Linux guest");
    assert!(status.success(), "linux guest build failed");
    target_dir.join(LINUX_TARGET).join("release").join(bin)
}

fn write_linux_upstream_config(directory: &Path, kernel: &str, init: &Path) -> PathBuf {
    let config_path = directory.join("proxima-linux.toml");
    let toml = format!(
        "[upstreams.{LINUX_PIPE}]\ntype = \"vm\"\nkernel_path = \"{kernel}\"\ninit_path = \"{}\"\n",
        init.display()
    );
    std::fs::write(&config_path, toml).expect("write linux TOML config");
    config_path
}

/// The Linux guest through `type = "vm"` with zero new Rust per function: a
/// real bzImage boots under KVM with the built `uppercase-init` as `/init`,
/// and the request body comes back uppercased.
fn config_only_linux_vm_upstream_uppercases_the_request_body() {
    let directory = tempfile::tempdir().expect("create tempdir for the Linux guest build");
    let init = build_linux_init(&directory.path().join("linux-target"));
    let config = write_linux_upstream_config(directory.path(), &host_kernel(), &init);
    let signed_self = SignedSelf::prepare();

    let response = signed_self
        .run(
            &[PROBE_FLAG, &config.to_string_lossy(), LINUX_PIPE],
            b"hello, linux",
        )
        .expect("config-registered linux vm upstream runs end to end");

    assert_eq!(response, b"HELLO, LINUX");
}

/// Sad path: a kernel that does not exist fails when the config is applied,
/// and the error names the path.
fn missing_kernel_fails_when_the_config_is_applied() {
    let directory = tempfile::tempdir().expect("create tempdir");
    let init = directory.path().join("init");
    std::fs::write(&init, b"init").expect("write init stub");
    let config = write_linux_upstream_config(directory.path(), "/no/such/vmlinuz", &init);
    let signed_self = SignedSelf::prepare();

    let error = signed_self
        .run(
            &[PROBE_FLAG, &config.to_string_lossy(), LINUX_PIPE],
            b"hello, linux",
        )
        .expect_err("a missing kernel cannot boot");

    assert!(
        error.contains(&format!("vm `{LINUX_PIPE}`")) && error.contains("read kernel `/no/such/vmlinuz`"),
        "the failure must come from the factory build and name the kernel path: {error}"
    );
}

/// A guest named by a bare relative path resolves against the directory of
/// the config file, not the process's working directory: the child runs from a
/// directory that holds nothing, and the guest sits beside the TOML.
fn relative_elf_guest_path_boots_from_a_different_working_directory() {
    let fixture = Fixture::new();
    let config_directory = tempfile::tempdir().expect("create the config directory");
    let elsewhere = tempfile::tempdir().expect("create an unrelated working directory");
    std::fs::copy(&fixture.guest_elf, config_directory.path().join(ELF_GUEST))
        .expect("place the guest beside the config");
    let config = write_vm_upstream_config(config_directory.path(), Path::new(ELF_GUEST), "");

    let response = fixture
        .signed_self
        .run_from(
            Some(elsewhere.path()),
            &[PROBE_FLAG, &config.to_string_lossy(), PIPE],
            b"hello, relative",
        )
        .expect("a relative guest_image_path resolves against the config directory");

    assert_eq!(response, b"HELLO, RELATIVE");
}

/// The same contract when the `vm` pipe is the `inner` of a `record` pipe: the
/// nested spec is anchored to the config directory just like a top-level one.
fn relative_elf_guest_path_nested_under_record_boots_from_a_different_working_directory() {
    let fixture = Fixture::new();
    let config_directory = tempfile::tempdir().expect("create the config directory");
    let elsewhere = tempfile::tempdir().expect("create an unrelated working directory");
    std::fs::copy(&fixture.guest_elf, config_directory.path().join(ELF_GUEST))
        .expect("place the guest beside the config");
    let recording = config_directory.path().join("recording.bin");
    let config_path = config_directory.path().join("proxima.toml");
    let toml = format!(
        "[upstreams.{PIPE}]\ntype = \"record\"\n\n\
         [upstreams.{PIPE}.sink]\npath = \"{}\"\n\n\
         [upstreams.{PIPE}.inner]\ntype = \"vm\"\nguest_image_path = \"{ELF_GUEST}\"\n",
        recording.display()
    );
    std::fs::write(&config_path, toml).expect("write the record config");

    let response = fixture
        .signed_self
        .run_from(
            Some(elsewhere.path()),
            &[PROBE_FLAG, &config_path.to_string_lossy(), PIPE],
            b"hello, nested",
        )
        .expect("a relative guest_image_path nested under record resolves against the config directory");

    assert_eq!(response, b"HELLO, NESTED");
}

/// The same contract through `App::load_full`, the seam `proxima serve
/// --config` uses for a multi-block config.
fn relative_elf_guest_path_serves_from_a_different_working_directory() {
    let fixture = Fixture::new();
    let config_directory = tempfile::tempdir().expect("create the config directory");
    let elsewhere = tempfile::tempdir().expect("create an unrelated working directory");
    std::fs::copy(&fixture.guest_elf, config_directory.path().join(ELF_GUEST))
        .expect("place the guest beside the config");
    let config_path = config_directory.path().join("serve.toml");
    let toml = format!(
        "[[pipe]]\nname = \"elf\"\ntype = \"vm\"\nguest_image_path = \"{ELF_GUEST}\"\n\n\
         [[listen]]\ntype = \"http\"\nbind = \"127.0.0.1:0\"\n\n\
         [[listen.mount]]\npath = \"/elf\"\npipe = \"elf\"\n"
    );
    std::fs::write(&config_path, toml).expect("write the serve config");

    let response = fixture
        .signed_self
        .run_from(
            Some(elsewhere.path()),
            &[SERVE_FLAG, &config_path.to_string_lossy(), "/elf"],
            b"hello, serve",
        )
        .expect("a relative guest_image_path resolves through load_full");

    assert_eq!(response, b"HELLO, SERVE");
}

/// Sad path: a relative guest that is absent from the config directory fails
/// naming the resolved absolute path, not the bare name.
fn missing_relative_guest_error_names_the_resolved_path() {
    let config_directory = tempfile::tempdir().expect("create the config directory");
    let elsewhere = tempfile::tempdir().expect("create an unrelated working directory");
    let config = write_vm_upstream_config(config_directory.path(), Path::new("absent.elf"), "");
    let resolved = config_directory.path().join("absent.elf");
    let signed_self = SignedSelf::prepare();

    let error = signed_self
        .run_from(
            Some(elsewhere.path()),
            &[PROBE_FLAG, &config.to_string_lossy(), PIPE],
            b"hello",
        )
        .expect_err("an absent relative guest cannot boot");

    assert!(
        error.contains(&format!("read guest image `{}`", resolved.display())),
        "the failure must name the resolved path {}: {error}",
        resolved.display()
    );
}

/// The Linux guest with `kernel_path` absolute and `init_path` relative to
/// the config directory, run from an unrelated working directory.
fn relative_linux_init_path_boots_from_a_different_working_directory() {
    let directory = tempfile::tempdir().expect("create tempdir for the Linux guest build");
    let init = build_linux_init(&directory.path().join("linux-target"));
    let config_directory = tempfile::tempdir().expect("create the config directory");
    let elsewhere = tempfile::tempdir().expect("create an unrelated working directory");
    std::fs::copy(&init, config_directory.path().join(LINUX_GUEST))
        .expect("place the init beside the config");
    let config = write_linux_upstream_config(
        config_directory.path(),
        &host_kernel(),
        Path::new(LINUX_GUEST),
    );
    let signed_self = SignedSelf::prepare();

    let response = signed_self
        .run_from(
            Some(elsewhere.path()),
            &[PROBE_FLAG, &config.to_string_lossy(), LINUX_PIPE],
            b"hello, relative linux",
        )
        .expect("a relative init_path resolves against the config directory");

    assert_eq!(response, b"HELLO, RELATIVE LINUX");
}

/// `examples/lambda/proxima.toml` as shipped, with only the guest paths made
/// absolute and the bind port made ephemeral: `[[pipe]] type = "vm"` mounted
/// under `[[listen]]` answers real HTTP POSTs for both guest kinds.
fn lambda_example_config_serves_both_guests_over_http() {
    let directory = tempfile::tempdir().expect("create tempdir for the guest builds");
    let elf = build_guest_elf(&directory.path().join("guest-target"), ELF_GUEST);
    let linux = build_linux_init(&directory.path().join("linux-target"));
    let shipped = std::fs::read_to_string(LAMBDA_EXAMPLE).expect("read the shipped example config");
    let config = shipped
        .replace("\"uppercase\"", &format!("\"{}\"", elf.display()))
        .replace("\"uppercase-init\"", &format!("\"{}\"", linux.display()))
        .replace("127.0.0.1:8088", "127.0.0.1:0");
    assert_ne!(config, shipped, "the example's guest paths must be substituted");
    let config_path = directory.path().join("proxima.toml");
    std::fs::write(&config_path, config).expect("write the substituted example config");
    let signed_self = SignedSelf::prepare();
    let config_arg = config_path.to_string_lossy();

    let elf_response = signed_self
        .run(&[SERVE_FLAG, &config_arg, "/elf"], b"hello, elf")
        .expect("POST /elf through the example listener");
    let linux_request = "hello, linux ".repeat(1024);
    let linux_response = signed_self
        .run(&[SERVE_FLAG, &config_arg, "/linux"], linux_request.as_bytes())
        .expect("POST /linux through the example listener");

    assert_eq!(elf_response, b"HELLO, ELF");
    assert_eq!(
        linux_response,
        linux_request.to_ascii_uppercase().as_bytes(),
        "13 KiB exceeds the elf upstream's default hypercall budget, so only the Linux guest answers it"
    );
}

fn percentile(sorted: &[u64], percent: usize) -> Duration {
    Duration::from_nanos(sorted[(sorted.len() * percent / 100).min(sorted.len() - 1)])
}

fn samples_of(output: &str, key: &str) -> Vec<u64> {
    let mut values: Vec<u64> = output
        .lines()
        .filter_map(|line| line.strip_prefix(key)?.trim().parse().ok())
        .collect();
    values.sort_unstable();
    values
}

fn describe(label: &str, sorted: &[u64]) -> String {
    format!(
        "{label:<44} p50={:>10.1?} p99={:>10.1?} min={:>10.1?} max={:>10.1?} n={}",
        percentile(sorted, 50),
        percentile(sorted, 99),
        Duration::from_nanos(sorted[0]),
        Duration::from_nanos(sorted[sorted.len() - 1]),
        sorted.len()
    )
}

fn child_text(fixture: &Fixture, arguments: &[&str]) -> String {
    let output = fixture
        .signed_self
        .run(arguments, BENCH_BODY.as_bytes())
        .expect("bench child");
    String::from_utf8(output).expect("bench child prints text")
}

fn bench(fixture: &Fixture, config: &Path, warmup: usize, placement: &str) -> Vec<u64> {
    let output = child_text(
        fixture,
        &[
            BENCH_FLAG,
            &config.to_string_lossy(),
            PIPE,
            &LATENCY_SAMPLES.to_string(),
            &warmup.to_string(),
            placement,
        ],
    );
    let latency = samples_of(&output, "latency_nanos");
    assert_eq!(latency.len(), LATENCY_SAMPLES);
    latency
}

/// Exit proof e: request -> response latency through the upstream, fresh VM
/// per call (cold included in every sample), at two guest memory sizes, next
/// to a warm-process baseline. The baseline is `process_rpc`, not `process`:
/// `ProcessUpstream::call` ignores the request (`src/upstreams/process.rs:232`),
/// while `process_rpc` writes the body to a long-lived child's stdin and
/// returns the stdout line.
fn vm_upstream_latency_table() {
    let fixture = Fixture::new();
    let guest = fixture.guest_elf.to_string_lossy().into_owned();

    let mut report = vec![format!(
        "vm_upstream_latency: n={LATENCY_SAMPLES} body={BENCH_BODY:?} host_profile={} loadavg={}",
        if cfg!(debug_assertions) { "dev" } else { "release" },
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
    )];
    for size in [TWO_MIB, SIXTY_FOUR_MIB] {
        let config = fixture.config(&format!("guest_memory_size = {size}\n"));
        for placement in ["offload", "inline"] {
            report.push(describe(
                &format!("vm request->response, {size} B, {placement}"),
                &bench(&fixture, &config, 0, placement),
            ));
        }
        let output = child_text(
            &fixture,
            &[
                INSTRUMENT_FLAG,
                &guest,
                &size.to_string(),
                &LATENCY_SAMPLES.to_string(),
            ],
        );
        report.push(describe(
            &format!("  run_dispatch_loop total, {size} B"),
            &samples_of(&output, "run_dispatch_loop_nanos"),
        ));
        report.push(describe(
            &format!("  create_to_first_exit, {size} B"),
            &samples_of(&output, "create_to_first_exit_nanos"),
        ));
        report.push(describe(
            &format!("  touch_all_pages, {size} B"),
            &samples_of(&output, "touch_all_pages_nanos"),
        ));
    }

    let baseline_config = fixture.directory.path().join("baseline.toml");
    std::fs::write(
        &baseline_config,
        format!(
            "[upstreams.{PIPE}]\ntype = \"process_rpc\"\ncommand = \"stdbuf\"\nargs = [\"-oL\", \"tr\", \"a-z\", \"A-Z\"]\n"
        ),
    )
    .expect("write baseline config");
    report.push(describe(
        "warm process_rpc `stdbuf -oL tr a-z A-Z`",
        &bench(&fixture, &baseline_config, BASELINE_WARMUP, "offload"),
    ));

    println!("{}", report.join("\n"));
}

/// A one-layer docker-format image holding `program` as `/bin/entry-probe`, written into
/// `cache_dir` the way `proxima_vm::oci::cache` lays blobs out; returns the manifest digest
/// a config can pin. The pull then finds every blob locally and never dials a registry.
fn seed_probe_image(cache_dir: &Path, program: &[u8]) -> String {
    use proxima_vm::oci::cache;
    use proxima_vm::oci::digest::Digest;

    let mut builder = tar::Builder::new(Vec::new());
    for (kind, mode, path, contents) in [
        (tar::EntryType::Directory, 0o755, "bin/", &[][..]),
        (tar::EntryType::Regular, 0o755, "bin/entry-probe", program),
    ] {
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(kind);
        header.set_mode(mode);
        header.set_size(contents.len() as u64);
        header.set_uid(0);
        header.set_gid(0);
        header.set_mtime(0);
        builder.append_data(&mut header, path, contents).expect("append tar entry");
    }
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(&builder.into_inner().expect("finish tar")).expect("gzip layer");
    let layer = encoder.finish().expect("finish gzip");
    let config = br#"{"config":{"Entrypoint":["/bin/entry-probe"],"Cmd":["--serve"],"Env":["OCI_PROBE=via-registry"],"WorkingDir":"/"}}"#;
    let (layer_digest, config_digest) = (Digest::of(&layer), Digest::of(config));
    let manifest = format!(
        r#"{{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"{config_digest}","size":{}}},"layers":[{{"mediaType":"application/vnd.docker.image.rootfs.diff.tar.gzip","digest":"{layer_digest}","size":{}}}]}}"#,
        config.len(),
        layer.len()
    );
    let manifest_digest = Digest::of(manifest.as_bytes());
    for (digest, bytes) in [
        (&layer_digest, layer.as_slice()),
        (&config_digest, config.as_slice()),
        (&manifest_digest, manifest.as_bytes()),
    ] {
        cache::write(cache_dir, digest, bytes).expect("seed the blob cache");
    }
    manifest_digest.to_string()
}

fn write_image_upstream_config(directory: &Path, kernel: &str, oci_init: &Path, cache_dir: &Path, manifest_digest: &str) -> PathBuf {
    let config_path = directory.join("proxima-image.toml");
    let toml = format!(
        "[upstreams.{IMAGE_PIPE}]\ntype = \"vm\"\nkernel_path = \"{kernel}\"\nimage = \"probe/app@{manifest_digest}\"\noci_init_path = \"{}\"\nimage_cache_dir = \"{}\"\nwatchdog_millis = 15000\n",
        oci_init.display(),
        cache_dir.display()
    );
    std::fs::write(&config_path, toml).expect("write image TOML config");
    config_path
}

/// An OCI image through `type = "vm"` with zero new Rust: `image` + `oci_init_path` in TOML pull
/// (here from a pre-seeded content-addressed cache), flatten, boot, and the call returns the
/// fate `oci-init` observed for the image's entrypoint.
fn config_only_linux_vm_upstream_runs_an_oci_image() {
    let directory = tempfile::tempdir().expect("create tempdir for the guest build");
    let target = directory.path().join("linux-target");
    let oci_init = build_linux_bin(&target, "oci-init");
    let probe = build_linux_bin(&target, "entry-probe");
    let cache_dir = directory.path().join("oci-cache");
    let manifest_digest = seed_probe_image(&cache_dir, &std::fs::read(&probe).expect("read entry-probe"));
    let config = write_image_upstream_config(directory.path(), &host_kernel(), &oci_init, &cache_dir, &manifest_digest);
    let signed_self = SignedSelf::prepare();

    let response = signed_self
        .run(&[PROBE_FLAG, &config.to_string_lossy(), IMAGE_PIPE], b"300")
        .expect("config-registered image upstream runs end to end");

    assert_eq!(response, b"exited:0");
}

fn vendored_whoami_blob(digest: &str) -> Vec<u8> {
    let name = digest.strip_prefix("sha256:").expect("sha256 digest");
    let raw: PathBuf = [WHOAMI_FIXTURE, "blobs", "sha256", name].iter().collect();
    match std::fs::read(&raw) {
        Ok(bytes) => bytes,
        Err(_) => {
            let hex_path = raw.with_extension("hex");
            let text = std::fs::read_to_string(&hex_path)
                .unwrap_or_else(|error| panic!("blob {} must exist raw or as .hex: {error}", raw.display()));
            hex::decode(text.trim()).expect("vendored hex blob decodes")
        }
    }
}

fn seed_whoami_cache(cache_dir: &Path) {
    use proxima_vm::oci::cache;
    use proxima_vm::oci::digest::Digest;

    for digest in WHOAMI_BLOBS {
        cache::write(
            cache_dir,
            &Digest::parse(digest).expect("whoami digest parses"),
            &vendored_whoami_blob(digest),
        )
        .expect("seed the blob cache from the vendored whoami fixture");
    }
}

fn write_resident_config(directory: &Path, kernel: &str, extra: &str) -> PathBuf {
    let config_path = directory.join("proxima-resident.toml");
    let toml = format!(
        "[upstreams.{RESIDENT_PIPE}]\ntype = \"vm\"\nkernel_path = \"{kernel}\"\nresident = true\n{extra}"
    );
    std::fs::write(&config_path, toml).expect("write resident TOML config");
    config_path
}

/// The resident lifetime through `type = "vm"` with zero new Rust: the whoami
/// image boots once when the config is applied, three sequential GETs reach it
/// through `H1ClientUpstream` over `dial_guest`, all three answer 200, and the
/// process started exactly one guest.
fn a_resident_vm_upstream_boots_one_guest_and_serves_sequential_gets() {
    let directory = tempfile::tempdir().expect("create tempdir for the guest build");
    let oci_init = build_linux_bin(&directory.path().join("linux-target"), "oci-init");
    let cache_dir = directory.path().join("oci-cache");
    seed_whoami_cache(&cache_dir);
    let extra = format!(
        "image = \"traefik/whoami@{WHOAMI_INDEX_DIGEST}\"\noci_init_path = \"{}\"\nimage_cache_dir = \"{}\"\nguest_port = 80\n",
        oci_init.display(),
        cache_dir.display()
    );
    let config = write_resident_config(directory.path(), &host_kernel(), &extra);
    let signed_self = SignedSelf::prepare();

    let output = signed_self
        .run(&[RESIDENT_FLAG, &config.to_string_lossy(), RESIDENT_PIPE], b"")
        .expect("config-registered resident upstream serves sequential GETs");

    let report = String::from_utf8(output).expect("child prints text");
    let statuses: Vec<&str> = report.lines().filter(|line| line.starts_with("status=")).collect();
    assert_eq!(statuses, vec!["status=200"; RESIDENT_REQUESTS], "{report}");
    assert!(report.contains("GET / HTTP/1.1"), "whoami echoes the request line: {report}");
    assert!(report.lines().any(|line| line == "boots=1"), "exactly one guest boots: {report}");
}

/// Sad path: `resident = true` on an elf guest fails when the config is
/// applied, and the error names the field.
fn a_resident_elf_guest_is_rejected_when_the_config_is_applied() {
    let directory = tempfile::tempdir().expect("create tempdir");
    let config_path = directory.path().join("proxima-resident-elf.toml");
    std::fs::write(
        &config_path,
        format!("[upstreams.{PIPE}]\ntype = \"vm\"\nguest_image_path = \"/no/such/guest.elf\"\nresident = true\n"),
    )
    .expect("write resident elf config");
    let signed_self = SignedSelf::prepare();

    let error = signed_self
        .run(&[PROBE_FLAG, &config_path.to_string_lossy(), PIPE], b"")
        .expect_err("a resident elf guest cannot boot");

    assert!(
        error.contains("resident: only a Linux guest (kernel_path) can run resident"),
        "the failure must come from validate and name the field: {error}"
    );
}

/// Sad path: a layer altered in the cache fails the build of the upstream with both digests,
/// so nothing boots.
fn a_tampered_cached_layer_fails_the_image_upstream_build_before_boot() {
    let directory = tempfile::tempdir().expect("create tempdir for the guest build");
    let target = directory.path().join("linux-target");
    let oci_init = build_linux_bin(&target, "oci-init");
    let cache_dir = directory.path().join("oci-cache");
    let manifest_digest = seed_probe_image(&cache_dir, b"\x7fELF-not-really");
    let blobs = cache_dir.join("blobs").join("sha256");
    let layer = std::fs::read_dir(&blobs)
        .expect("list cached blobs")
        .map(|entry| entry.expect("blob entry").path())
        .find(|path| std::fs::read(path).is_ok_and(|bytes| bytes.starts_with(&[0x1f, 0x8b])))
        .expect("the gzip layer is in the cache");
    std::fs::write(&layer, b"tampered").expect("tamper with the cached layer");
    let config = write_image_upstream_config(directory.path(), &host_kernel(), &oci_init, &cache_dir, &manifest_digest);
    let signed_self = SignedSelf::prepare();

    let error = signed_self
        .run(&[PROBE_FLAG, &config.to_string_lossy(), IMAGE_PIPE], b"300")
        .expect_err("a tampered layer must fail the build");

    assert!(
        error.contains("failed its digest check: expected sha256:") && error.contains("actual sha256:"),
        "the failure must carry both digests: {error}"
    );
}
