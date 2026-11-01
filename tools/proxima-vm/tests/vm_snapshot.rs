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

#[path = "common/snapshot_probes.rs"]
mod snapshot_probes;

use snapshot_probes::{
    SignedProbes, assert_restores_are_byte_correct, iteration_series,
    lane_is_supported, percentiles_and_cov,
};

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
    assert_eq!(
        fields.get("resumed_x0").map(String::as_str),
        Some("256"),
        "the resumed trap must read back the exact terminal value the snapshot captured \
         (`x0` on HVF, `rax` on KVM -- the scratch guest loads 256 into `ax` before its `hlt`)"
    );
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
    if !lane_is_supported() {
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

/// Correctness half of the µsec campaign slice 2 size sweep (the timing half
/// is `benches/bench_warm_restore_sweep.rs`): at every snapshot size up to
/// 256MiB, every warm restore re-traps and every sampled post-restore byte
/// matches the deterministic pattern, which a lazy no-op restore would fail.
/// One case per size, so each size is its own bounded run.
#[proxima::test]
#[case::size_64kib("64KiB", 64 * 1024)]
#[case::size_1mib("1MiB", 1024 * 1024)]
#[case::size_16mib("16MiB", 16 * 1024 * 1024)]
#[case::size_64mib("64MiB", 64 * 1024 * 1024)]
#[case::size_256mib("256MiB", 256 * 1024 * 1024)]
async fn warm_restores_are_byte_correct_at_this_snapshot_size(#[case] label: &str, #[case] size: usize) {
    if !lane_is_supported() {
        return;
    }

    const ITERATIONS: usize = 2;

    let probes = SignedProbes::prepare();
    let fields = probes.snapshot_and_warm_restore_sized(
        "size-sweep-proof",
        4096,
        ITERATIONS,
        Some(size),
        Some("same"),
    );
    assert_restores_are_byte_correct(&fields, ITERATIONS, label);
}

/// µsec campaign slice 2's degenerate control: a warm restore whose snapshot
/// content is unchanged from the region's current bytes still pays the same
/// `memcpy` -- so warm-restore wall time must be (within noise) identical
/// whether `content_mode` is `same` (content constant after the first call)
/// or `alternate` (content changes every call). Run at one representative
/// size (16MiB), not the full sweep -- this is one control experiment, not
/// a second sweep.
#[proxima::test(timeout = "300s")]
async fn warm_restore_cost_is_size_bound_not_content_bound() {
    if !lane_is_supported() {
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
    let same_bytes = iteration_series(&same_fields, "iteration_bytes_copied", ITERATIONS);
    let alternate_bytes = iteration_series(&alternate_fields, "iteration_bytes_copied", ITERATIONS);

    let (same_p50, _same_p99, _same_cov) = percentiles_and_cov(same_wall_nanos);
    let (alternate_p50, _alternate_p99, _alternate_cov) = percentiles_and_cov(alternate_wall_nanos);

    println!(
        "content-bound control at {SIZE} bytes (reported, not asserted): \
         same_content p50={same_p50}ns alternating_content p50={alternate_p50}ns"
    );

    let expected_bytes = SIZE as u64;
    for (mode, series) in [("same", &same_bytes), ("alternate", &alternate_bytes)] {
        assert!(
            series.iter().all(|&copied| copied == expected_bytes),
            "{mode}-content restores must each move exactly the snapshot size ({expected_bytes} \
             bytes) regardless of content -- size-bound, not content-bound; saw {series:?}"
        );
    }
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
    if !lane_is_supported() {
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
    if !lane_is_supported() {
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
#[proxima::test(timeout = "300s")]
async fn layered_restore_cost_scales_with_dirty_page_count_not_region_size() {
    if !lane_is_supported() {
        return;
    }

    const ITERATIONS: usize = 5;
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

        let remapped_pages =
            iteration_series(&fields, "iteration_remapped_page_count", ITERATIONS);
        let newly_dirty_pages =
            iteration_series(&fields, "iteration_run_newly_dirty_page_count", ITERATIONS);
        let expected_pages = u64::from(dirty_page_count);
        assert!(
            remapped_pages.iter().all(|&pages| pages == expected_pages),
            "size={label} K={dirty_page_count}: every restore must remap exactly K pages \
             whatever the region size; saw {remapped_pages:?}"
        );
        assert!(
            newly_dirty_pages.iter().all(|&pages| pages == expected_pages),
            "size={label} K={dirty_page_count}: every run must dirty exactly K pages; \
             saw {newly_dirty_pages:?}"
        );

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
    if !lane_is_supported() {
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
