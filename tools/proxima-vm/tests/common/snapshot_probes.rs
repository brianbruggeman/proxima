#![allow(dead_code, clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

const CAPTURE_BINARY: &str = env!("CARGO_BIN_EXE_snapshot_capture_probe");
const RESTORE_BINARY: &str = env!("CARGO_BIN_EXE_snapshot_restore_probe");
const WARM_RESTORE_BINARY: &str = env!("CARGO_BIN_EXE_snapshot_warm_restore_probe");
const LAYERED_BINARY: &str = env!("CARGO_BIN_EXE_snapshot_layered_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");

pub const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
pub const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));

pub fn lane_is_supported() -> bool {
    HYPERVISOR_LANE || KVM_LANE
}

/// A signed, isolated pair of `snapshot_capture_probe`/`snapshot_restore_probe`
/// copies, mirroring `tests/boot.rs`'s own `SignedGuest`: nextest runs cases
/// in parallel processes, so each case signs its own private copies rather
/// than racing a shared path.
pub struct SignedProbes {
    pub directory: TempDir,
    pub capture_path: PathBuf,
    pub restore_path: PathBuf,
    pub warm_restore_path: PathBuf,
    pub layered_path: PathBuf,
}

impl SignedProbes {
    pub fn prepare() -> Self {
        let directory = tempfile::tempdir().expect("create tempdir for the signed probes");
        let capture_path = directory.path().join("snapshot-capture-probe");
        let restore_path = directory.path().join("snapshot-restore-probe");
        let warm_restore_path = directory.path().join("snapshot-warm-restore-probe");
        let layered_path = directory.path().join("snapshot-layered-probe");
        std::fs::copy(Path::new(CAPTURE_BINARY), &capture_path)
            .expect("copy the built capture probe");
        std::fs::copy(Path::new(RESTORE_BINARY), &restore_path)
            .expect("copy the built restore probe");
        std::fs::copy(Path::new(WARM_RESTORE_BINARY), &warm_restore_path)
            .expect("copy the built warm-restore probe");
        std::fs::copy(Path::new(LAYERED_BINARY), &layered_path)
            .expect("copy the built layered probe");

        if HYPERVISOR_LANE {
            for path in [
                &capture_path,
                &restore_path,
                &warm_restore_path,
                &layered_path,
            ] {
                let status = Command::new("codesign")
                    .arg("--force")
                    .arg("--sign")
                    .arg("-")
                    .arg("--entitlements")
                    .arg(ENTITLEMENTS)
                    .arg(path)
                    .status()
                    .expect("run codesign");
                assert!(status.success(), "codesign failed for {}", path.display());
            }
        }

        Self {
            directory,
            capture_path,
            restore_path,
            warm_restore_path,
            layered_path,
        }
    }

    /// Runs `snapshot_layered_probe` in `sweep` mode: adopts a base of
    /// `target_size` bytes, then runs `iterations` dirty-write/restore
    /// cycles dirtying `dirty_page_count` pages each time. Panics with the
    /// probe's own stderr on a nonzero exit.
    pub fn run_layered_sweep(
        &self,
        target_size: usize,
        dirty_page_count: u16,
        iterations: usize,
    ) -> HashMap<(String, String), String> {
        let output = Command::new(&self.layered_path)
            .arg("sweep")
            .arg(target_size.to_string())
            .arg(dirty_page_count.to_string())
            .arg(iterations.to_string())
            .output()
            .expect("run the layered probe");
        assert!(
            output.status.success(),
            "snapshot_layered_probe sweep failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout)
            .into_owned()
            .lines()
            .filter_map(|line| {
                let mut fields = line.splitn(3, ':');
                let key = fields.next()?.to_string();
                let second = fields.next()?.to_string();
                match fields.next() {
                    Some(value) => Some(((key, second), value.to_string())),
                    None => Some(((key, String::new()), second)),
                }
            })
            .collect()
    }

    /// Runs `snapshot_layered_probe` in `sharing` mode: two `WarmVm`s over
    /// one base, proven concurrently in one process. Panics with the
    /// probe's own stderr on a nonzero exit.
    pub fn run_layered_sharing(&self) -> HashMap<String, String> {
        let output = Command::new(&self.layered_path)
            .arg("sharing")
            .output()
            .expect("run the layered probe");
        assert!(
            output.status.success(),
            "snapshot_layered_probe sharing failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout)
            .into_owned()
            .lines()
            .filter_map(|line| {
                line.split_once(':')
                    .map(|(key, value)| (key.to_string(), value.to_string()))
            })
            .collect()
    }

    /// Runs `snapshot_capture_probe` then `snapshot_restore_probe` as two
    /// separate child processes connected by a snapshot file, parsing each
    /// probe's `key:value` stdout lines into one combined map. Panics with
    /// the failing probe's stderr on a nonzero exit.
    pub fn snapshot_and_restore(&self, message: &str, page_size: usize) -> HashMap<String, String> {
        let snapshot_path = self.directory.path().join("snapshot.postcard");

        let capture_output = Command::new(&self.capture_path)
            .arg(message)
            .arg(&snapshot_path)
            .output()
            .expect("run the capture probe");
        assert!(
            capture_output.status.success(),
            "snapshot_capture_probe failed: {}",
            String::from_utf8_lossy(&capture_output.stderr)
        );

        let restore_output = Command::new(&self.restore_path)
            .arg(&snapshot_path)
            .arg(page_size.to_string())
            .output()
            .expect("run the restore probe");
        assert!(
            restore_output.status.success(),
            "snapshot_restore_probe failed: {}",
            String::from_utf8_lossy(&restore_output.stderr)
        );

        [&capture_output.stdout, &restore_output.stdout]
            .into_iter()
            .flat_map(|stdout| {
                String::from_utf8_lossy(stdout)
                    .into_owned()
                    .lines()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .filter_map(|line| {
                line.split_once(':')
                    .map(|(key, value)| (key.to_string(), value.to_string()))
            })
            .collect()
    }

    /// Runs `snapshot_capture_probe` once, then `snapshot_warm_restore_probe`
    /// once against the resulting snapshot -- ONE `WarmVm` inside that one
    /// process, exercised `iterations` times, proving the µsec campaign's
    /// own "same-process warm restore avoids the second-`hv_vm_create`
    /// hang" claim empirically rather than asserting it. Returns every
    /// `key:iteration:value` line, keyed by `(key, iteration)`, plus the
    /// plain `key:value` summary lines under an empty-string iteration key.
    pub fn snapshot_and_warm_restore_many(
        &self,
        message: &str,
        page_size: usize,
        iterations: usize,
    ) -> HashMap<(String, String), String> {
        self.snapshot_and_warm_restore_sized(message, page_size, iterations, None, None)
    }

    /// The size-sweep form (µsec campaign slice 2): `target_size` scales the
    /// padded guest-memory `WarmVm::restore` copies per call
    /// (`VmSnapshot::with_padded_memory`), `content_mode` selects the
    /// degenerate content-bound control (`same`/`alternate`) -- both `None`
    /// reproduce `snapshot_and_warm_restore_many`'s own unsized behaviour
    /// exactly, so this is the one shape both callers share.
    pub fn snapshot_and_warm_restore_sized(
        &self,
        message: &str,
        page_size: usize,
        iterations: usize,
        target_size: Option<usize>,
        content_mode: Option<&str>,
    ) -> HashMap<(String, String), String> {
        let snapshot_path = self.directory.path().join("snapshot.postcard");

        let capture_output = Command::new(&self.capture_path)
            .arg(message)
            .arg(&snapshot_path)
            .output()
            .expect("run the capture probe");
        assert!(
            capture_output.status.success(),
            "snapshot_capture_probe failed: {}",
            String::from_utf8_lossy(&capture_output.stderr)
        );

        let mut command = Command::new(&self.warm_restore_path);
        command
            .arg(&snapshot_path)
            .arg(page_size.to_string())
            .arg(iterations.to_string());
        if let Some(target_size) = target_size {
            command.arg(target_size.to_string());
            command.arg(content_mode.unwrap_or("same"));
        }
        let warm_output = command.output().expect("run the warm-restore probe");
        assert!(
            warm_output.status.success(),
            "snapshot_warm_restore_probe failed: {}",
            String::from_utf8_lossy(&warm_output.stderr)
        );

        String::from_utf8_lossy(&warm_output.stdout)
            .into_owned()
            .lines()
            .filter_map(|line| {
                let mut fields = line.splitn(3, ':');
                let key = fields.next()?.to_string();
                let second = fields.next()?.to_string();
                match fields.next() {
                    Some(value) => Some(((key, second), value.to_string())),
                    None => Some(((key, String::new()), second)),
                }
            })
            .collect()
    }
}

/// p50/p99/CoV over a nanosecond series -- the µsec campaign's own bench
/// discipline (`BENCH_LOG.md`'s existing warm-restore rows), pulled out as a
/// free function so the size-sweep test below computes both the VM-side and
/// memcpy-control arms with the same formula.
pub fn percentiles_and_cov(mut samples: Vec<u64>) -> (u64, u64, f64) {
    samples.sort_unstable();
    let len = samples.len();
    let p50 = samples[len / 2];
    let p99 = samples[(len * 99 / 100).min(len - 1)];
    let mean = samples.iter().sum::<u64>() as f64 / len as f64;
    let variance = samples
        .iter()
        .map(|&sample| (sample as f64 - mean).powi(2))
        .sum::<f64>()
        / len as f64;
    let cov = if mean > 0.0 {
        variance.sqrt() / mean
    } else {
        0.0
    };
    (p50, p99, cov)
}

/// Pulls every `iteration_<field>:<index>:<value>` series out of
/// `snapshot_and_warm_restore_sized`'s returned map, parsed as `u64`
/// nanoseconds, in index order.
pub fn iteration_series(
    fields: &HashMap<(String, String), String>,
    field: &str,
    iterations: usize,
) -> Vec<u64> {
    (0..iterations)
        .map(|index| {
            fields[&(field.to_string(), index.to_string())]
                .parse()
                .unwrap_or_else(|_| panic!("parse {field}:{index}"))
        })
        .collect()
}

pub const SWEEP_SIZES: [(&str, usize); 5] = [
    ("64KiB", 64 * 1024),
    ("1MiB", 1024 * 1024),
    ("16MiB", 16 * 1024 * 1024),
    ("64MiB", 64 * 1024 * 1024),
    ("256MiB", 256 * 1024 * 1024),
];

pub fn assert_restores_are_byte_correct(
    fields: &HashMap<(String, String), String>,
    iterations: usize,
    label: &str,
) {
    let matched_count: usize = fields[&("matched_count".to_string(), String::new())]
        .parse()
        .expect("parse matched_count");
    assert_eq!(
        matched_count, iterations,
        "every warm restore at size={label} must re-trap correctly"
    );

    let sample_matches: Vec<bool> = fields
        .iter()
        .filter(|((key, _), _)| key == "sample_offset")
        .map(|(_, value)| value == "true")
        .collect();
    assert!(
        !sample_matches.is_empty(),
        "size={label} must report at least one sample_offset correctness line"
    );
    assert!(
        sample_matches.iter().all(|&matched| matched),
        "size={label}: a sampled post-restore byte did not match the deterministic pattern -- \
         a lazy no-op restore would still re-trap but fail this check"
    );
}
