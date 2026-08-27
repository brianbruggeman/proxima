//! M2's config-only exit proof, as a real signed process: `[upstream.my-fn]
//! type = "vm"` in a TOML file — with no per-function Rust anywhere —
//! resolves through `proxima`'s `PipeFactoryRegistry` to a live `PipeHandle`
//! that boots a real guest ELF under a real hypervisor and returns the
//! bytes it emitted (`tools/proxima-vm/ROADMAP.md`'s M2 exit criterion 1).
//!
//! `proxima` (the crate that owns the `vm` upstream factory,
//! `src/upstreams/vm.rs`) can only be a **dev**-dependency of `proxima-vm`:
//! `proxima` optionally depends on `proxima-vm` under its own `vm` feature,
//! so a real (non-dev) dependency in the other direction would cycle. That
//! rules out a `[[bin]]` target the way `tests/dispatch_hypercall.rs` uses
//! `dispatch_probe` — a `[[bin]]` only sees `[dependencies]`. So this file
//! owns `fn main()` (`harness = false` in `Cargo.toml`) and re-executes
//! ITSELF as the signed hypervisor-entitled child, dispatching on argv:
//! `--vm-registry-probe-child <config-path> <pipe-name>` runs the App/
//! registry/dispatch path and prints the guest's emitted bytes to stdout;
//! no argv runs the actual test cases, each of which builds a real guest
//! ELF, writes a TOML config naming it, signs a copy of this very
//! executable, and drives it as the child.
//!
//! Hypervisor.framework answers `hv_vm_create` with `HV_DENIED` for a
//! process lacking the `com.apple.security.hypervisor` entitlement, and
//! applying it is a post-link `codesign` step — same constraint
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
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use libtest_mimic::{Arguments, Trial};
use tempfile::TempDir;

const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");
const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));
const CHILD_FLAG: &str = "--vm-registry-probe-child";

fn main() {
    // the child-dispatch branch reads its own two positional args, not
    // libtest_mimic's flags — intercepted before `Arguments::from_args`
    // ever sees argv, so a real `cargo test`/nextest invocation (which
    // never passes `CHILD_FLAG`) is unaffected.
    let mut arguments = env::args().skip(1);
    if arguments.next().as_deref() == Some(CHILD_FLAG) {
        let config_path = arguments.next().expect("config path argv");
        let pipe_name = arguments.next().expect("pipe name argv");
        match run_child(&config_path, &pipe_name) {
            Ok(()) => return,
            Err(error) => {
                eprintln!("child failed: {error}");
                std::process::exit(1);
            }
        }
    }

    let trials = if HYPERVISOR_LANE || KVM_LANE {
        vec![
            Trial::test(
                "config_only_vm_upstream_boots_a_real_guest_and_returns_its_bytes",
                || {
                    config_only_vm_upstream_boots_a_real_guest_and_returns_its_bytes();
                    Ok(())
                },
            ),
            Trial::test("boot_run_teardown_latency_baseline", || {
                boot_run_teardown_latency_baseline();
                Ok(())
            }),
        ]
    } else {
        Vec::new()
    };

    libtest_mimic::run(&Arguments::from_args(), trials).exit();
}

/// Drives the actual config -> registry -> hypervisor path. Reads NOTHING
/// guest- or function-specific from Rust — only `config_path`/`pipe_name`
/// from argv — which is the config-only claim under test.
fn run_child(config_path: &str, pipe_name: &str) -> Result<(), Box<dyn Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let settings = proxima::settings::ProximaSettings::from_path(config_path)?;
        let mut app = proxima::App::builder().with_defaults()?.build()?;
        app.apply_settings(&settings).await?;
        let handle = app
            .lookup_pipe(pipe_name)
            .ok_or_else(|| format!("pipe `{pipe_name}` not registered by config"))?;
        let request = proxima::Request::builder()
            .method("GET")
            .path("/")
            .build()?;
        let response = proxima_primitives::pipe::SendPipe::call(&handle, request).await?;
        std::io::stdout().write_all(&response.payload)?;
        Ok::<(), Box<dyn Error>>(())
    })
}

/// Builds `proxima-vm-guest-lambda` for `aarch64-unknown-none` into a
/// per-case tempdir's own `CARGO_TARGET_DIR` — same discipline
/// `tests/dispatch_hypercall.rs`'s `build_guest_elf` uses, copied rather
/// than shared because that helper lives in a `harness = true` target this
/// file (`harness = false`) cannot import as a sibling module.
fn build_guest_elf(target_dir: &Path) -> PathBuf {
    let status = Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            &format!("{GUEST_MANIFEST_DIR}/Cargo.toml"),
            "--target",
            "aarch64-unknown-none",
            "--release",
        ])
        .env("CARGO_TARGET_DIR", target_dir)
        .status()
        .expect("run cargo build for the guest ELF");
    assert!(status.success(), "guest ELF build failed");
    target_dir
        .join("aarch64-unknown-none")
        .join("release")
        .join("proxima-vm-guest-lambda")
}

/// A signed copy of this very test executable, ready to run as
/// `--vm-registry-probe-child`.
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

    fn run_child(&self, config_path: &Path, pipe_name: &str) -> Result<Vec<u8>, String> {
        let output = Command::new(&self.path)
            .arg(CHILD_FLAG)
            .arg(config_path)
            .arg(pipe_name)
            .output()
            .expect("run the signed self-copy");
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(stderr)
        }
    }
}

fn write_vm_upstream_config(directory: &Path, guest_elf: &Path) -> PathBuf {
    let config_path = directory.join("proxima.toml");
    // `ProximaSettings.upstreams` is the field name (`src/settings/mod.rs`);
    // `[upstreams.my-fn]` is the TOML table key it deserializes from —
    // `tests/units/settings_to_app.rs` pins the same plural spelling.
    let toml = format!(
        "[upstreams.my-fn]\ntype = \"vm\"\nguest_image_path = \"{}\"\nresponse = \"vm-side-canned\"\n",
        guest_elf.display()
    );
    std::fs::write(&config_path, toml).expect("write TOML config");
    config_path
}

/// M2 exit criterion 1: a fixture registers a NOVEL function purely via
/// TOML — zero new Rust — and drives it end to end, returning the guest's
/// bytes. `my-fn` names nothing this test file, `VmPipeFactory`, or
/// `VmConfig` hard-codes; it is entirely the TOML's choice.
fn config_only_vm_upstream_boots_a_real_guest_and_returns_its_bytes() {
    let guest_directory = tempfile::tempdir().expect("create tempdir for the guest build");
    let guest_elf = build_guest_elf(guest_directory.path());
    let config_path = write_vm_upstream_config(guest_directory.path(), &guest_elf);

    let signed_self = SignedSelf::prepare();
    let emitted = signed_self
        .run_child(&config_path, "my-fn")
        .expect("config-registered vm upstream runs end to end");

    // the guest re-emits the first byte of the host's ChildResponse buffer
    // after each of its two hypercalls (`guests/lambda/src/main.rs`); the
    // dispatcher answers every hypercall with the same configured `Read`
    // response, so both emitted bytes are that variant's postcard
    // discriminant (0x00) — proof the guest ran to completion against a
    // real VM exit, not that it was replayed from a canned transcript.
    assert_eq!(
        emitted,
        [0x00, 0x00],
        "a TOML-only `type = \"vm\"` upstream must boot a real guest through \
         a real VM exit and return its two emitted bytes"
    );
}

/// M2 exit criterion 3: the boot -> run -> teardown cycle's p50/p99, fresh
/// VM per request, no snapshot, no fork — the baseline every later
/// milestone (M8's µsec claim) is measured against.
fn boot_run_teardown_latency_baseline() {
    const SAMPLES: usize = 20;

    let guest_directory = tempfile::tempdir().expect("create tempdir for the guest build");
    let guest_elf = build_guest_elf(guest_directory.path());
    let config_path = write_vm_upstream_config(guest_directory.path(), &guest_elf);
    let signed_self = SignedSelf::prepare();

    let mut durations: Vec<Duration> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        signed_self
            .run_child(&config_path, "my-fn")
            .expect("run for latency sample");
        durations.push(start.elapsed());
    }
    durations.sort();
    let p50 = durations[SAMPLES / 2];
    let p99 = durations[(SAMPLES * 99 / 100).min(SAMPLES - 1)];
    println!(
        "vm_registry: boot->run->teardown p50={p50:?} p99={p99:?} \
         (fresh VM per request, no snapshot, no fork, n={SAMPLES})"
    );
}
