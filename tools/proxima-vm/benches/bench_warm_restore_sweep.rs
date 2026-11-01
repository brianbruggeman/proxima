//! µsec campaign slice 2 (`BENCH_LOG.md`): how warm-restore time scales with
//! snapshot memory size. 100 warm restores per size plus a same-size raw
//! `memcpy` control arm with no VM involved. Prints the size-sweep table
//! `BENCH_LOG.md` cites. Correctness at every size is asserted by
//! `warm_restores_are_byte_correct_at_this_snapshot_size` in
//! `tests/vm_snapshot.rs`; this target only measures.
//!
//! Run: `cargo bench -p proxima-vm --bench bench_warm_restore_sweep`

#[path = "../tests/common/snapshot_probes.rs"]
mod snapshot_probes;

use snapshot_probes::{
    SWEEP_SIZES, SignedProbes, assert_restores_are_byte_correct, iteration_series,
    lane_is_supported, percentiles_and_cov,
};

const ITERATIONS: usize = 100;

fn main() {
    if !lane_is_supported() {
        println!("no hypervisor lane on this host; nothing to measure");
        return;
    }

    println!(
        "| size | warm p50 (ns) | warm p99 (ns) | warm CoV | memcpy p50 (ns) | memcpy p99 (ns) | delta p50 (ns) |"
    );
    println!("|---|---|---|---|---|---|---|");

    for (label, size) in SWEEP_SIZES {
        let probes = SignedProbes::prepare();
        let fields = probes.snapshot_and_warm_restore_sized(
            "size-sweep-proof",
            4096,
            ITERATIONS,
            Some(size),
            Some("same"),
        );
        assert_restores_are_byte_correct(&fields, ITERATIONS, label);

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
