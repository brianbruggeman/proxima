//! The M3 fault-count instrument, as tests
//! (`tools/proxima-vm/ROADMAP.md`'s "M3 — the fault-count instrument"):
//! `dispatch_probe`'s stderr summary line now carries the three numbers
//! `dispatch::run_dispatch_loop` reports alongside its M1 four `Vec`
//! fields — `create_to_first_exit_nanos`, `touch_all_pages_nanos`,
//! `mmio_trap_count` — so this file parses that line rather than
//! reintroducing a second probe binary.
//!
//! [`SignedProbe`] is copied from `tests/dispatch_hypercall.rs` rather than
//! shared: each integration-test file in this crate compiles as its own
//! binary, and there is no shared test-support crate here to import from
//! (`tests/boot.rs` and `tests/dispatch_hypercall.rs` each already carry
//! their own copy of this exact shape).

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

const PROBE_BINARY: &str = env!("CARGO_BIN_EXE_dispatch_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");
const GUEST_MANIFEST_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/guests/lambda");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));

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

struct SignedProbe {
    _directory: TempDir,
    path: PathBuf,
    guest_elf: PathBuf,
}

impl SignedProbe {
    fn prepare() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
        let guest_elf = build_guest_elf(&directory.path().join("guest-target"));
        Self::prepare_with_guest_elf(directory, guest_elf)
    }

    /// Same signing/copy setup as [`Self::prepare`], against a caller-built
    /// `guest_elf` rather than the lambda guest's own build — the
    /// degenerate-control case needs a guest that never touches mmio, which
    /// the lambda guest's fixed M1+M6 bring-up sequence does not provide.
    fn prepare_with_degenerate_guest() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed probe");
        let guest_elf = directory.path().join("halt-only-guest.elf");
        std::fs::write(&guest_elf, build_minimal_elf(&HALT_ONLY_GUEST_CODE))
            .expect("write synthetic halt-only guest ELF");
        Self::prepare_with_guest_elf(directory, guest_elf)
    }

    fn prepare_with_guest_elf(directory: TempDir, guest_elf: PathBuf) -> Self {
        let path = directory.path().join("dispatch-probe");
        std::fs::copy(Path::new(PROBE_BINARY), &path).expect("copy the built probe binary");

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
            guest_elf,
        }
    }

    fn run(&self) -> String {
        let output = Command::new(&self.path)
            .arg(&self.guest_elf)
            .arg("read")
            .output()
            .expect("run the probe");
        assert!(
            output.status.success(),
            "probe run failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stderr).into_owned()
    }
}

fn lane_is_supported() -> bool {
    HYPERVISOR_LANE || KVM_LANE
}

/// Encodes a minimal, valid ELF64 `ET_EXEC` image with exactly one
/// `PT_LOAD` entry (readable + executable, `virtual_address = 0`,
/// `file_offset = 0x1000`) carrying `content` — the same byte layout
/// `src/elf.rs`'s own `test_support::build_elf` +
/// `TestSegment::readable_executable(0, 0x1000, content)` produce, proven
/// accepted by `parse_elf` (`elf.rs`'s own `minimal_valid_elf` fixture uses
/// this exact shape), inlined here because `test_support` is `pub(crate)`
/// to `proxima-vm` and this file compiles as a separate crate. `file_offset
/// = 0` would place the segment's content over the ELF header itself
/// (bytes 0..64, including the magic number) once `build_elf`'s content
/// pass runs after its header pass — `0x1000` keeps the two apart.
fn build_minimal_elf(content: &[u8]) -> Vec<u8> {
    const ELF64_HEADER_LEN: usize = 64;
    const PROGRAM_HEADER_LEN: usize = 56;
    const PT_LOAD: u32 = 1;
    const PF_EXECUTE: u32 = 1;
    const PF_READ: u32 = 4;
    const ET_EXEC: u16 = 2;

    let program_header_table_offset = ELF64_HEADER_LEN;
    let mut image = vec![0_u8; program_header_table_offset + PROGRAM_HEADER_LEN];

    image[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    image[4] = 2; // ELFCLASS64
    image[5] = 1; // ELFDATA2LSB
    image[6] = 1; // EI_VERSION
    image[16..18].copy_from_slice(&ET_EXEC.to_le_bytes());
    image[20..24].copy_from_slice(&1_u32.to_le_bytes()); // e_version
    image[24..32].copy_from_slice(&0_u64.to_le_bytes()); // e_entry
    image[32..40].copy_from_slice(&(program_header_table_offset as u64).to_le_bytes());
    image[52..54].copy_from_slice(&(ELF64_HEADER_LEN as u16).to_le_bytes());
    image[54..56].copy_from_slice(&(PROGRAM_HEADER_LEN as u16).to_le_bytes());
    image[56..58].copy_from_slice(&1_u16.to_le_bytes()); // e_phnum

    const FILE_OFFSET: u64 = 0x1000;

    let phdr = program_header_table_offset;
    image[phdr..phdr + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
    image[phdr + 4..phdr + 8].copy_from_slice(&(PF_READ | PF_EXECUTE).to_le_bytes());
    image[phdr + 8..phdr + 16].copy_from_slice(&FILE_OFFSET.to_le_bytes()); // p_offset
    image[phdr + 16..phdr + 24].copy_from_slice(&0_u64.to_le_bytes()); // p_vaddr
    image[phdr + 24..phdr + 32].copy_from_slice(&0_u64.to_le_bytes()); // p_paddr
    image[phdr + 32..phdr + 40].copy_from_slice(&(content.len() as u64).to_le_bytes());
    image[phdr + 40..phdr + 48].copy_from_slice(&(content.len() as u64).to_le_bytes());
    image[phdr + 48..phdr + 56].copy_from_slice(&0x1000_u64.to_le_bytes()); // p_align

    let content_start = FILE_OFFSET as usize;
    let content_end = content_start + content.len();
    if image.len() < content_end {
        image.resize(content_end, 0);
    }
    image[content_start..content_end].copy_from_slice(content);
    image
}

/// `movz x0, #0xffff` (`PROXIMA_VM_HALT_VERB`, `src/dispatch_trampoline.h`)
/// followed by `hvc #0` — the two-instruction guest the C exit loop
/// (`src/backend_macos.c`'s `proxima_vm_run_dispatch_loop`) services via
/// its own `PROXIMA_VM_HALT_VERB` branch, before any mmio window or
/// dispatcher call is ever reached. This is the M3 degenerate control's "a
/// run that touches zero pages" guest.
const HALT_ONLY_GUEST_CODE: [u8; 8] = [0xe0, 0xff, 0x9f, 0xd2, 0x02, 0x00, 0x00, 0xd4];

/// Parses `dispatch_probe`'s
/// `m3 create_to_first_exit_nanos=<n> touch_all_pages_nanos=<n> mmio_trap_count=<n>`
/// summary line out of its stderr, returning the three numbers in that
/// order. Panics (test-only, per this crate's own convention for
/// integration-test parsing helpers) if the line is missing or malformed —
/// either is itself a real M3 regression worth failing loudly on.
fn parse_m3_line(stderr: &str) -> (u64, u64, u64) {
    let line = stderr
        .lines()
        .find(|line| line.starts_with("m3 "))
        .unwrap_or_else(|| panic!("no m3 summary line in probe stderr: {stderr}"));
    let mut fields = line.split_whitespace().skip(1);
    let parse_field = |field: Option<&str>| -> u64 {
        field
            .and_then(|entry| entry.split('=').nth(1))
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("malformed m3 summary line: {line}"))
    };
    let create_to_first_exit_nanos = parse_field(fields.next());
    let touch_all_pages_nanos = parse_field(fields.next());
    let mmio_trap_count = parse_field(fields.next());
    (
        create_to_first_exit_nanos,
        touch_all_pages_nanos,
        mmio_trap_count,
    )
}

/// The M3 exit table's per-run numbers must actually be present and
/// nonzero for a guest that does drive real hypercall/mmio traffic (the
/// lambda guest's fixed M1+M6 bring-up sequence) — a zero here would mean
/// the clock reads or the counter never fired, not that the guest was
/// fast.
#[proxima::test]
async fn fault_count_summary_reports_nonzero_wall_times_for_a_real_guest_run() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let stderr = probe.run();
    let (create_to_first_exit_nanos, touch_all_pages_nanos, _mmio_trap_count) =
        parse_m3_line(&stderr);

    assert!(
        create_to_first_exit_nanos > 0,
        "create-to-first-exit wall time must be nonzero for a real vCPU run: {stderr}"
    );
    assert!(
        touch_all_pages_nanos > 0,
        "touch-all-pages wall time must be nonzero for a 64 MiB guest reservation: {stderr}"
    );
}

/// Degenerate control (`ROADMAP.md`'s M3 section, mandatory): the lambda
/// guest issues at least one real mmio access as part of its fixed M1+M6
/// bring-up sequence (`guests/lambda/src/main.rs`), so `mmio_trap_count`
/// must be nonzero here — a run that touched mmio and still reported zero
/// traps would mean the counter measures nothing, and the instrument is
/// void per the roadmap's own words.
#[proxima::test]
async fn mmio_trap_count_is_nonzero_for_a_guest_that_touches_mmio() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let stderr = probe.run();
    let (_create_to_first_exit_nanos, _touch_all_pages_nanos, mmio_trap_count) =
        parse_m3_line(&stderr);

    assert!(
        mmio_trap_count > 0,
        "the lambda guest's own console/net/blk bring-up sequence must produce at least \
         one mmio trap: {stderr}"
    );
}

/// Two identical runs of the same guest, against the same host, must
/// report the same `mmio_trap_count` — the number of mmio accesses a fixed
/// instruction stream issues is deterministic, unlike wall-clock timing.
/// Wall times are reported here (not asserted on beyond the nonzero check
/// above) because host scheduling noise makes exact-equality across two
/// process launches an unreliable assertion; the variance is reported
/// verbatim rather than asserted away.
#[proxima::test]
async fn mmio_trap_count_is_stable_across_two_identical_runs() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare();
    let first_stderr = probe.run();
    let second_stderr = probe.run();
    let (_, _, first_mmio_trap_count) = parse_m3_line(&first_stderr);
    let (_, _, second_mmio_trap_count) = parse_m3_line(&second_stderr);

    assert_eq!(
        first_mmio_trap_count, second_mmio_trap_count,
        "two runs of the identical guest instruction stream must trap into mmio the same \
         number of times: first={first_stderr} second={second_stderr}"
    );
}

/// Degenerate control, mandatory (`ROADMAP.md`'s M3 section verbatim): "a
/// run that touches zero pages must report a near-zero fault count. If it
/// does not, the instrument measures something other than what it names
/// and every number after it is void." A guest whose entire instruction
/// stream is `movz x0, #HALT_VERB; hvc #0` never accesses an mmio window —
/// `mmio_trap_count` for this run must be exactly zero, or `mmio_trap_count`
/// is void as an instrument for every other test in this file.
#[proxima::test]
async fn a_guest_that_touches_no_mmio_window_reports_a_zero_mmio_trap_count() {
    if !lane_is_supported() {
        return;
    }

    let probe = SignedProbe::prepare_with_degenerate_guest();
    let stderr = probe.run();
    let (_, _, mmio_trap_count) = parse_m3_line(&stderr);

    assert_eq!(
        mmio_trap_count, 0,
        "a guest that never accesses an mmio window must report a zero mmio trap count, or \
         the instrument measures something other than what it names: {stderr}"
    );
}
