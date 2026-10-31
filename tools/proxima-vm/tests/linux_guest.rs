//! The Linux guest against a real kernel: `/boot/vmlinuz-linux` (or
//! `PROXIMA_VM_KERNEL`) booted under KVM with the `linux-*` musl guests as
//! the initramfs `/init`. A missing kernel or guest is a loud failure, never a
//! skip.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use proxima_vm::linux::{LinuxEnd, LinuxGuestLimits, LinuxGuestRun, run_linux_guest};
use proxima_vm::x86_boot::SetupHeader;

const DEFAULT_KERNEL: &str = "/boot/vmlinuz-linux";
const GUEST_MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/linux-init/Cargo.toml");
const GUEST_TARGET_DIR: &str = concat!(env!("CARGO_TARGET_TMPDIR"), "/linux-guest");
const GUEST_TARGET: &str = "x86_64-unknown-linux-musl";
const REAL_TEXT: &str = include_str!("../ROADMAP.md");
const MEASURED_BOOTS: usize = 20;
const CMDLINE: &str = "console=ttyS0 reboot=t panic=-1 nomodules i8042.noaux loglevel=6";
const CMDLINE_RESET_PORT: &str = "console=ttyS0 reboot=k panic=-1 nomodules i8042.noaux loglevel=6";

fn kernel() -> &'static [u8] {
    static KERNEL: OnceLock<Vec<u8>> = OnceLock::new();
    KERNEL.get_or_init(|| {
        let path = std::env::var("PROXIMA_VM_KERNEL").unwrap_or_else(|_| DEFAULT_KERNEL.to_string());
        std::fs::read(&path).unwrap_or_else(|error| panic!("kernel {path} must be readable: {error}"))
    })
}

fn guest(name: &str) -> Vec<u8> {
    let status = Command::new("cargo")
        .args(["build", "--release", "--target", GUEST_TARGET, "--manifest-path", GUEST_MANIFEST])
        .env("CARGO_TARGET_DIR", GUEST_TARGET_DIR)
        .status()
        .expect("spawn cargo to build the linux guests");
    assert!(status.success(), "linux guest build failed");
    let path: PathBuf = [GUEST_TARGET_DIR, GUEST_TARGET, "release", name].iter().collect();
    std::fs::read(&path).unwrap_or_else(|error| panic!("guest {} must exist: {error}", path.display()))
}

fn uppercase_guest() -> &'static [u8] {
    static GUEST: OnceLock<Vec<u8>> = OnceLock::new();
    GUEST.get_or_init(|| guest("uppercase-init"))
}

fn invoke(input: &[u8]) -> LinuxGuestRun {
    run_linux_guest(kernel(), uppercase_guest(), CMDLINE, input, &LinuxGuestLimits::default())
        .unwrap_or_else(|error| panic!("linux guest failed: {error}"))
}

fn kernel_release() -> String {
    let image = kernel();
    let header = SetupHeader::parse(image).expect("the host kernel is a bzImage");
    let version = header.version_string(image).expect("the setup code names its version");
    String::from_utf8_lossy(version).split_whitespace().next().expect("release token").to_string()
}

#[test]
fn the_kernel_boots_and_prints_its_own_version_banner() {
    let release = kernel_release();

    let run = invoke(b"hello, linux");

    let console = String::from_utf8_lossy(&run.console);
    assert!(!run.console.is_empty(), "kernel console produced no bytes");
    assert!(console.contains(&format!("Linux version {release}")), "console lacks the banner for {release}:\n{console}");
    assert_eq!(run.end, LinuxEnd::Shutdown, "reboot=t ends in a triple fault");
}

#[test]
fn hello_linux_comes_back_uppercased() {
    let run = invoke(b"hello, linux");

    assert_eq!(run.output, b"HELLO, LINUX");
}

#[test]
fn a_multi_kib_repo_text_file_round_trips_uppercased_byte_exact() {
    assert!(REAL_TEXT.len() > 8 * 1024, "fixture must be multi-KiB, is {} bytes", REAL_TEXT.len());

    let run = invoke(REAL_TEXT.as_bytes());

    assert_eq!(run.output, REAL_TEXT.to_ascii_uppercase().into_bytes());
}

#[test]
fn empty_input_produces_empty_output() {
    let run = invoke(b"");

    assert_eq!(run.output, b"");
}

#[test]
fn every_byte_value_survives_the_raw_tty_unmangled() {
    let every_byte: Vec<u8> = (0..4).flat_map(|_| 0..=255_u8).collect();

    let run = invoke(&every_byte);

    assert_eq!(run.output, every_byte.to_ascii_uppercase());
}

#[test]
fn reboot_k_ends_through_the_i8042_reset_port() {
    let run = run_linux_guest(kernel(), uppercase_guest(), CMDLINE_RESET_PORT, b"hello, linux", &LinuxGuestLimits::default())
        .expect("reboot=k run");

    assert_eq!(run.end, LinuxEnd::ResetPort);
    assert_eq!(run.output, b"HELLO, LINUX");
}

#[test]
fn bytes_that_are_not_a_bzimage_are_rejected_by_the_header_parser() {
    let error = run_linux_guest(&[0x7f; 4096], uppercase_guest(), CMDLINE, b"x", &LinuxGuestLimits::default())
        .expect_err("garbage is not a kernel");

    assert_eq!(error.to_string(), "config: kernel rejected: not a bzImage: boot flag 0x7f7f, expected 0xaa55");
}

#[test]
fn ram_too_small_for_the_kernel_is_rejected_before_any_vm_exists() {
    let limits = LinuxGuestLimits { ram_mib: 16, ..LinuxGuestLimits::default() };

    let error = run_linux_guest(kernel(), uppercase_guest(), CMDLINE, b"x", &limits).expect_err("16 MiB cannot hold this kernel");

    assert!(error.to_string().starts_with("config: boot plan rejected: guest ram too small"), "{error}");
}

#[test]
fn ram_above_the_kvm_leafs_three_gib_ceiling_is_rejected() {
    let limits = LinuxGuestLimits { ram_mib: 4096, ..LinuxGuestLimits::default() };

    let error = run_linux_guest(kernel(), uppercase_guest(), CMDLINE, b"x", &limits).expect_err("4 GiB overlaps the tss pages");

    assert!(error.to_string().contains("linux ram must be a nonzero page multiple of at most 3 GiB"), "{error}");
}

#[test]
fn a_console_smaller_than_the_boot_log_reports_the_dropped_bytes() {
    let limits = LinuxGuestLimits { console_capacity: 256, ..LinuxGuestLimits::default() };

    let run = run_linux_guest(kernel(), uppercase_guest(), CMDLINE, b"x", &limits).expect("run with a tiny console");

    assert_eq!(run.console.len(), 256);
    assert!(run.console_dropped > 1000, "dropped {}", run.console_dropped);
    assert_eq!(run.output, b"X");
}

#[test]
fn watchdog_fires_on_an_init_that_never_powers_off() {
    let limits = LinuxGuestLimits { watchdog_millis: 1500, ..LinuxGuestLimits::default() };

    let error = run_linux_guest(kernel(), &guest("spin-init"), CMDLINE, b"x", &limits).expect_err("spin never ends");

    let message = error.to_string();
    assert!(message.contains("watchdog forced exit: guest never trapped within the watchdog window"), "{message}");
    assert!(message.contains("console tail:"), "{message}");
}

#[test]
fn a_frame_that_declares_more_bytes_than_it_sends_is_reported_with_both_counts() {
    let error = run_linux_guest(kernel(), &guest("truncated-init"), CMDLINE, b"x", &LinuxGuestLimits::default())
        .expect_err("declared 100, sent 3");

    assert!(error.to_string().contains("guest output frame truncated: declared 100 bytes, received 3"), "{error}");
}

#[test]
fn output_beyond_the_budget_is_reported_as_overflow() {
    let limits = LinuxGuestLimits { output_capacity: 1024, ..LinuxGuestLimits::default() };

    let error = run_linux_guest(kernel(), &guest("flood-init"), CMDLINE, b"x", &limits).expect_err("64 KiB into a 1 KiB budget");

    assert!(error.to_string().contains("guest output exceeded the 1024 byte budget"), "{error}");
}

fn percentile(sorted: &[Duration], fraction: f64) -> Duration {
    let rank = (fraction * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn summarize(label: &str, mut samples: Vec<Duration>) -> String {
    samples.sort();
    format!(
        "{label:<28} p50={:>9.3}ms p99={:>9.3}ms min={:>9.3}ms max={:>9.3}ms",
        percentile(&samples, 0.50).as_secs_f64() * 1e3,
        percentile(&samples, 0.99).as_secs_f64() * 1e3,
        samples[0].as_secs_f64() * 1e3,
        samples[samples.len() - 1].as_secs_f64() * 1e3,
    )
}

#[test]
fn twenty_boots_report_percentiles_console_size_and_unhandled_ports() {
    measure("reboot=t (triple fault)", CMDLINE);
}

#[test]
fn twenty_reset_port_boots_report_the_same_instruments() {
    measure("reboot=k (i8042 reset port)", CMDLINE_RESET_PORT);
}

fn measure(label: &str, cmdline: &str) {
    let loadavg = std::fs::read_to_string("/proc/loadavg").expect("read /proc/loadavg");
    let mut setup = Vec::new();
    let mut first_exit = Vec::new();
    let mut init_exec = Vec::new();
    let mut invocation_proper = Vec::new();
    let mut total = Vec::new();
    let mut invocation_wall = Vec::new();
    let mut console_sizes = Vec::new();
    let mut exit_counts = Vec::new();
    let mut port_tables = Vec::new();

    for _ in 0..MEASURED_BOOTS {
        let begun = std::time::Instant::now();
        let run = run_linux_guest(kernel(), uppercase_guest(), cmdline, b"hello, linux", &LinuxGuestLimits::default())
            .unwrap_or_else(|error| panic!("linux guest failed: {error}"));
        invocation_wall.push(begun.elapsed());
        assert_eq!(run.output, b"HELLO, LINUX");
        setup.push(run.timings.setup);
        first_exit.push(run.timings.first_exit);
        let init_at = run.timings.init_exec.expect("init wrote the marker");
        init_exec.push(init_at);
        invocation_proper.push(run.timings.total - init_at);
        total.push(run.timings.total);
        console_sizes.push(run.console.len());
        exit_counts.push(run.exit_count);
        port_tables.push(run.unhandled_ports.iter().map(|entry| (entry.port, (entry.reads, entry.writes))).collect::<BTreeMap<_, _>>());
    }

    eprintln!("== {label}: loadavg={} profile=debug-test-binary(release-guest) N={MEASURED_BOOTS}", loadavg.trim());
    eprintln!("{}", summarize("create->first KVM_RUN (setup)", setup));
    eprintln!("{}", summarize("create->first exit", first_exit));
    eprintln!("{}", summarize("create->init exec", init_exec));
    eprintln!("{}", summarize("init exec->total", invocation_proper));
    eprintln!("{}", summarize("total (C-side)", total));
    eprintln!("{}", summarize("invocation wall (incl. prep)", invocation_wall));
    eprintln!("console bytes: min={} max={}", console_sizes.iter().min().unwrap(), console_sizes.iter().max().unwrap());
    eprintln!("exits: min={} max={}", exit_counts.iter().min().unwrap(), exit_counts.iter().max().unwrap());
    for (port, (reads, writes)) in &port_tables[0] {
        eprintln!("unhandled port 0x{port:04x}: reads={reads} writes={writes}");
    }
    assert!(port_tables.iter().all(|table| table == &port_tables[0]), "unhandled port accesses differ between boots");
}
