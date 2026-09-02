//! M7 — snapshot, as tests (`tools/proxima-vm/ROADMAP.md`'s M7 section):
//! restore wall time and fault count at each page size, measured with the
//! M3 instrument, proven by a resumed vCPU re-trapping with the exact
//! register value the snapshot captured.
//!
//! `hv_vm_create` answers `HV_DENIED` for a process lacking
//! `com.apple.security.hypervisor`, so — the exact `SignedGuest` pattern
//! `tests/boot.rs` established — `proxima_vm::snapshot::{capture, restore}`
//! run inside two small codesigned probe binaries. They are two SEPARATE
//! processes, not one: `hv_vm_create` is once-per-process on the HVF lane
//! (`proxima_vm::snapshot`'s own module doc), connected by a
//! postcard-encoded [`proxima_vm::snapshot::VmSnapshot`] file on disk.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

const CAPTURE_BINARY: &str = env!("CARGO_BIN_EXE_snapshot_capture_probe");
const RESTORE_BINARY: &str = env!("CARGO_BIN_EXE_snapshot_restore_probe");
const WARM_RESTORE_BINARY: &str = env!("CARGO_BIN_EXE_snapshot_warm_restore_probe");
const LAYERED_BINARY: &str = env!("CARGO_BIN_EXE_snapshot_layered_probe");
const ENTITLEMENTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/entitlements.plist");

const HYPERVISOR_LANE: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));
const KVM_LANE: bool = cfg!(all(target_os = "linux", target_arch = "x86_64"));

fn lane_is_supported() -> bool {
    HYPERVISOR_LANE || KVM_LANE
}

/// A signed, isolated pair of `snapshot_capture_probe`/`snapshot_restore_probe`
/// copies, mirroring `tests/boot.rs`'s own `SignedGuest`: nextest runs cases
/// in parallel processes, so each case signs its own private copies rather
/// than racing a shared path.
struct SignedProbes {
    directory: TempDir,
    capture_path: PathBuf,
    restore_path: PathBuf,
    warm_restore_path: PathBuf,
    layered_path: PathBuf,
}

impl SignedProbes {
    fn prepare() -> Self {
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
    fn run_layered_sweep(
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
    fn run_layered_sharing(&self) -> HashMap<String, String> {
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
    fn snapshot_and_restore(&self, message: &str, page_size: usize) -> HashMap<String, String> {
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
    fn snapshot_and_warm_restore_many(
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
    fn snapshot_and_warm_restore_sized(
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
fn percentiles_and_cov(mut samples: Vec<u64>) -> (u64, u64, f64) {
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
fn iteration_series(
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

#[proxima::test]
async fn a_restored_vcpu_re_traps_with_the_same_register_the_snapshot_captured() {
    if !lane_is_supported() {
        return;
    }

    let probes = SignedProbes::prepare();
    let fields = probes.snapshot_and_restore("snapshot-proof", 4096);

    assert_eq!(
        fields.get("emitted").map(String::as_str),
        Some("snapshot-proof")
    );
    assert_eq!(
        fields.get("resumed_matched_trap").map(String::as_str),
        Some("true"),
        "a resumed vCPU seeded from the snapshot's registers must re-trap at the same halting instruction"
    );
    if HYPERVISOR_LANE {
        // KVM's scratch guest halts via `hlt`, which carries no sentinel
        // value in `rax` the way this lane's `hvc`-based halt carries
        // `TERMINAL_VALUE` in `x0` -- this exact readback is an HVF-lane
        // proof only.
        assert_eq!(
            fields.get("resumed_x0").map(String::as_str),
            Some("256"),
            "the resumed trap must read back the exact terminal value the snapshot captured"
        );
    }
}

/// The M7 exit criterion, verbatim: restore wall time and fault count at
/// each page size, measured with the M3 instrument.
#[proxima::test]
#[case::page_4kib(4096)]
#[case::page_16kib(16384)]
#[case::page_64kib(65536)]
async fn restore_reports_wall_time_and_fault_count_at_this_page_size(#[case] page_size: usize) {
    if !lane_is_supported() {
        return;
    }

    let probes = SignedProbes::prepare();
    let fields = probes.snapshot_and_restore("page-stride-proof", page_size);

    let restore_wall_nanos: u64 = fields["restore_wall_nanos"]
        .parse()
        .expect("parse restore_wall_nanos");
    let touch_all_pages_nanos: u64 = fields["touch_all_pages_nanos"]
        .parse()
        .expect("parse touch_all_pages_nanos");
    let fault_count: u64 = fields["fault_count"].parse().expect("parse fault_count");

    assert!(
        restore_wall_nanos > 0,
        "restore at page_size={page_size} must report a nonzero wall time, got {restore_wall_nanos}"
    );
    assert!(
        touch_all_pages_nanos > 0,
        "the page-strided memory copy at page_size={page_size} must report a nonzero wall time"
    );
    assert_eq!(
        fault_count, 0,
        "this guest touches no mmio window, so the resumed step's fault count must be exactly 0"
    );
    assert_eq!(
        fields.get("resumed_matched_trap").map(String::as_str),
        Some("true"),
        "restore at page_size={page_size} must still reproduce the halting trap"
    );

    eprintln!(
        "page_size={page_size} restore_wall_nanos={restore_wall_nanos} \
         touch_all_pages_nanos={touch_all_pages_nanos} fault_count={fault_count}"
    );
}

/// µsec campaign, warm-restore first slice's own exit criterion: `N >= 100`
/// consecutive warm restores against ONE `WarmVm`, in ONE process, all
/// re-trapping correctly -- the empirical proof that reusing a live
/// vm/vcpu never triggers the second-`hv_vm_create` hang
/// `proxima_vm::snapshot::WarmVm`'s own module doc names, since this test
/// would hang (and nextest's own timeout would fail it) rather than merely
/// assert-fail if it did.
#[proxima::test]
async fn n_consecutive_warm_restores_in_one_process_all_re_trap_correctly() {
    if !HYPERVISOR_LANE {
        // The warm-restore trio is implemented on the HVF lane only this
        // slice (`dispatch_trampoline.h`'s own doc on the warm-restore
        // trio); the KVM lane has no `hv_vm_create`-once-per-process
        // restriction to work around in the first place.
        return;
    }

    const ITERATIONS: usize = 150;

    let probes = SignedProbes::prepare();
    let fields = probes.snapshot_and_warm_restore_many("warm-restore-proof", 4096, ITERATIONS);

    let reported_iterations: usize = fields[&("iterations".to_string(), String::new())]
        .parse()
        .expect("parse iterations");
    let matched_count: usize = fields[&("matched_count".to_string(), String::new())]
        .parse()
        .expect("parse matched_count");

    assert_eq!(reported_iterations, ITERATIONS);
    assert_eq!(
        matched_count, ITERATIONS,
        "every one of {ITERATIONS} warm restores in this one process must re-trap correctly, \
         not merely most of them"
    );

    for index in 0..ITERATIONS {
        let key = index.to_string();
        assert_eq!(
            fields
                .get(&("iteration_resumed_matched_trap".to_string(), key.clone()))
                .map(String::as_str),
            Some("true"),
            "warm restore #{index} must re-trap"
        );
        assert_eq!(
            fields
                .get(&("iteration_resumed_x0".to_string(), key))
                .map(String::as_str),
            Some("256"),
            "warm restore #{index} must read back the exact terminal value"
        );
    }
}

/// µsec campaign slice 2 (`tools/proxima-vm/BENCH_LOG.md`): how warm-restore
/// time scales with snapshot memory size. 100 warm restores per size, a
/// same-size raw-`memcpy` control arm with no VM involved, and a byte-level
/// correctness check (not just the re-trap) that a lazy no-op restore could
/// not pass. Prints the size-sweep table `BENCH_LOG.md`'s own row cites.
#[proxima::test]
async fn warm_restore_wall_time_scales_with_snapshot_memory_size() {
    if !HYPERVISOR_LANE {
        // The warm-restore trio is implemented on the HVF lane only
        // (`dispatch_trampoline.h`'s own doc); see
        // `n_consecutive_warm_restores_in_one_process_all_re_trap_correctly`.
        return;
    }

    const ITERATIONS: usize = 100;
    const SIZES: [(&str, usize); 5] = [
        ("64KiB", 64 * 1024),
        ("1MiB", 1024 * 1024),
        ("16MiB", 16 * 1024 * 1024),
        ("64MiB", 64 * 1024 * 1024),
        ("256MiB", 256 * 1024 * 1024),
    ];

    println!(
        "| size | warm p50 (ns) | warm p99 (ns) | warm CoV | memcpy p50 (ns) | memcpy p99 (ns) | delta p50 (ns) |"
    );
    println!("|---|---|---|---|---|---|---|");

    for (label, size) in SIZES {
        let probes = SignedProbes::prepare();
        let fields = probes.snapshot_and_warm_restore_sized(
            "size-sweep-proof",
            4096,
            ITERATIONS,
            Some(size),
            Some("same"),
        );

        let matched_count: usize = fields[&("matched_count".to_string(), String::new())]
            .parse()
            .expect("parse matched_count");
        assert_eq!(
            matched_count, ITERATIONS,
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

        let warm_wall_nanos = iteration_series(&fields, "iteration_call_wall_nanos", ITERATIONS);
        let memcpy_control_nanos =
            iteration_series(&fields, "iteration_memcpy_control_nanos", ITERATIONS);

        let (warm_p50, warm_p99, warm_cov) = percentiles_and_cov(warm_wall_nanos);
        let (memcpy_p50, memcpy_p99, _memcpy_cov) = percentiles_and_cov(memcpy_control_nanos);
        let delta_p50 = warm_p50.saturating_sub(memcpy_p50);

        println!(
            "| {label} | {warm_p50} | {warm_p99} | {warm_cov:.3} | {memcpy_p50} | {memcpy_p99} | {delta_p50} |"
        );
    }
}

/// µsec campaign slice 2's degenerate control: a warm restore whose snapshot
/// content is unchanged from the region's current bytes still pays the same
/// `memcpy` -- so warm-restore wall time must be (within noise) identical
/// whether `content_mode` is `same` (content constant after the first call)
/// or `alternate` (content changes every call). Run at one representative
/// size (16MiB), not the full sweep -- this is one control experiment, not
/// a second sweep.
#[proxima::test]
async fn warm_restore_cost_is_size_bound_not_content_bound() {
    if !HYPERVISOR_LANE {
        return;
    }

    const ITERATIONS: usize = 100;
    const SIZE: usize = 16 * 1024 * 1024;

    let same_probes = SignedProbes::prepare();
    let same_fields = same_probes.snapshot_and_warm_restore_sized(
        "content-control-proof",
        4096,
        ITERATIONS,
        Some(SIZE),
        Some("same"),
    );
    let alternate_probes = SignedProbes::prepare();
    let alternate_fields = alternate_probes.snapshot_and_warm_restore_sized(
        "content-control-proof",
        4096,
        ITERATIONS,
        Some(SIZE),
        Some("alternate"),
    );

    let same_matched: usize = same_fields[&("matched_count".to_string(), String::new())]
        .parse()
        .expect("parse matched_count");
    let alternate_matched: usize = alternate_fields[&("matched_count".to_string(), String::new())]
        .parse()
        .expect("parse matched_count");
    assert_eq!(same_matched, ITERATIONS);
    assert_eq!(alternate_matched, ITERATIONS);

    let same_wall_nanos = iteration_series(&same_fields, "iteration_call_wall_nanos", ITERATIONS);
    let alternate_wall_nanos =
        iteration_series(&alternate_fields, "iteration_call_wall_nanos", ITERATIONS);

    let (same_p50, _same_p99, _same_cov) = percentiles_and_cov(same_wall_nanos);
    let (alternate_p50, _alternate_p99, _alternate_cov) = percentiles_and_cov(alternate_wall_nanos);

    let relative_difference =
        (same_p50 as f64 - alternate_p50 as f64).abs() / (same_p50.max(alternate_p50) as f64);

    println!(
        "content-bound control at {SIZE} bytes: same_content p50={same_p50}ns \
         alternating_content p50={alternate_p50}ns relative_difference={relative_difference:.3}"
    );

    assert!(
        relative_difference < 0.25,
        "warm-restore p50 differed by {:.1}% between same-content ({same_p50}ns) and \
         alternating-content ({alternate_p50}ns) restores at {SIZE} bytes -- expected the cost to \
         be size-bound, not content-bound",
        relative_difference * 100.0
    );
}

/// The layered rework's own correctness gate: after `iterations` dirty-write
/// + mapping-only-restore cycles, the base's own bytes must still match the
/// snapshot's ORIGINAL content byte-for-byte over the WHOLE region -- the
/// design's own "restore is a mapping, not a copy" claim proven at the
/// strongest level a bare re-trap check cannot reach (a lazy no-op restore
/// would still leave the base untouched by construction, so this is really
/// proving `run_dirty_write` never corrupts the base, not merely that
/// `restore_layered` is a no-op).
#[proxima::test]
async fn layered_restore_reproduces_the_base_byte_identically_over_the_whole_region() {
    if !HYPERVISOR_LANE {
        return;
    }

    let probes = SignedProbes::prepare();
    let fields = probes.run_layered_sweep(1024 * 1024, 32, 10);

    assert_eq!(
        fields
            .get(&("byte_identical_twin_oracle".to_string(), String::new()))
            .map(String::as_str),
        Some("true"),
        "the base's bytes must match the original snapshot content exactly after every dirty-write/restore cycle"
    );
}

/// The design's own "restore becomes a mapping, not a copy" claim, proven
/// mechanically: after `restore_layered`, re-running the IDENTICAL
/// dirty-write guest program must re-fault on every one of the same pages
/// (`post_restore_fault_count == dirty_page_count`) -- proof the mapping
/// genuinely reverted to read-only, not merely that the dirty bitmap was
/// cleared while the stage-2 permission stayed read-write.
#[proxima::test]
#[case::small_k(16)]
#[case::medium_k(256)]
async fn layered_restore_actually_remaps_dirtied_pages_read_only(#[case] dirty_page_count: u16) {
    if !HYPERVISOR_LANE {
        return;
    }

    let probes = SignedProbes::prepare();
    let fields = probes.run_layered_sweep(16 * 1024 * 1024, dirty_page_count, 3);

    let post_restore_fault_count: u16 = fields
        [&("post_restore_fault_count".to_string(), String::new())]
        .parse()
        .expect("parse post_restore_fault_count");
    assert_eq!(
        post_restore_fault_count, dirty_page_count,
        "every one of the {dirty_page_count} previously-dirtied pages must re-fault after restore"
    );
}

/// µsec campaign, layered rework: restore cost as a function of dirty page
/// count `K`, at two region sizes -- the design's own claim ("restore scales
/// with K remaps, not with size or copy volume") stated as a table, not a
/// single number. Prints the row `BENCH_LOG.md` cites.
#[proxima::test]
async fn layered_restore_cost_scales_with_dirty_page_count_not_region_size() {
    if !HYPERVISOR_LANE {
        return;
    }

    const ITERATIONS: usize = 50;
    const CASES: [(&str, usize, u16); 5] = [
        ("16MiB", 16 * 1024 * 1024, 16),
        ("16MiB", 16 * 1024 * 1024, 256),
        ("256MiB", 256 * 1024 * 1024, 16),
        ("256MiB", 256 * 1024 * 1024, 256),
        ("256MiB", 256 * 1024 * 1024, 4096),
    ];

    println!(
        "| size | K | restore p50 (ns) | restore p99 (ns) | run p50 (ns) | unprotected-run p50 (ns) |"
    );
    println!("|---|---|---|---|---|---|");

    for (label, size, dirty_page_count) in CASES {
        let probes = SignedProbes::prepare();
        let fields = probes.run_layered_sweep(size, dirty_page_count, ITERATIONS);

        let byte_identical = fields
            .get(&("byte_identical_twin_oracle".to_string(), String::new()))
            .map(String::as_str);
        assert_eq!(
            byte_identical,
            Some("true"),
            "size={label} K={dirty_page_count}: base must survive intact"
        );

        let post_restore_fault_count: u16 = fields
            [&("post_restore_fault_count".to_string(), String::new())]
            .parse()
            .expect("parse post_restore_fault_count");
        assert_eq!(
            post_restore_fault_count, dirty_page_count,
            "size={label} K={dirty_page_count}: must re-trap"
        );

        let restore_nanos = iteration_series(&fields, "iteration_restore_wall_nanos", ITERATIONS);
        let run_nanos = iteration_series(&fields, "iteration_run_wall_nanos", ITERATIONS);
        let unprotected_nanos =
            iteration_series(&fields, "iteration_unprotected_run_wall_nanos", ITERATIONS);

        let (restore_p50, restore_p99, _restore_cov) = percentiles_and_cov(restore_nanos);
        let (run_p50, _run_p99, _run_cov) = percentiles_and_cov(run_nanos);
        let (unprotected_p50, _unprotected_p99, _unprotected_cov) =
            percentiles_and_cov(unprotected_nanos);

        println!(
            "| {label} | {dirty_page_count} | {restore_p50} | {restore_p99} | {run_p50} | {unprotected_p50} |"
        );
    }
}

/// The design's own sharing proof, closing M4's deferred "two VMs map the
/// same region" criterion for this milestone: two `WarmVm`s over ONE shared
/// base, each with its own private delta, concurrently in one process --
/// a write through one must be invisible in the base and in the other.
#[proxima::test]
async fn two_warm_vms_share_one_base_without_observing_each_others_writes() {
    if !HYPERVISOR_LANE {
        return;
    }

    let probes = SignedProbes::prepare();
    let fields = probes.run_layered_sharing();

    assert_eq!(
        fields.get("vm_a_run_halted_ok").map(String::as_str),
        Some("true")
    );
    assert_eq!(
        fields.get("base_unaffected").map(String::as_str),
        Some("true"),
        "a write through vm_a's delta must never appear in the shared base"
    );
    assert_eq!(
        fields.get("vm_a_wrote_its_delta").map(String::as_str),
        Some("true"),
        "vm_a's own delta must carry the write it made"
    );
    assert_eq!(
        fields.get("vm_b_never_wrote").map(String::as_str),
        Some("true"),
        "vm_b's delta must be untouched by vm_a's write -- proof the two deltas are genuinely private"
    );
}
