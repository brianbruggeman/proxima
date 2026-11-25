//! The one stage-totals surface every device backend feeds and every reader
//! consumes, so no caller needs a backend `cfg` to ask what a step cost.
//!
//! Built from [`proxima_telemetry::metric::Counter`] (the same instrument
//! `crate::metal` and `crate::wgpu_driver` already use) read with
//! `snapshot_and_reset`, so a reader sees a per-step delta rather than a
//! cumulative mean.
//!
//! Who fills what:
//! - Metal keeps its own `MetalStageTotals` counters and maps them into
//!   [`StageTotals`] at read time (`crate::metal::shared_stage_totals`).
//! - CUDA and wgpu call [`record_block_copied`] at each host-to-device upload
//!   of a named block and [`record_pipeline_compile`] after each kernel or
//!   pipeline build; the fields they cannot measure stay `0`. CUDA also calls
//!   [`record_stage_phases`] once per executed plan with its upload, submit,
//!   device-wait and readback times.
//!
//! `block_copied_bytes` is the same quantity as Metal's `BLOCK_COPIED_BYTES`:
//! bytes that crossed host-to-device because the block was copied, never bytes
//! a resident cache served for free.

use proxima_telemetry::counter;
use proxima_telemetry::metric::Counter;

/// Host-to-device bytes copied for named blocks by every non-Metal backend.
pub static BLOCK_COPIED_BYTES: Counter = Counter::new("omega.backend.block_copied_bytes");
/// Kernels or pipelines built because no cached one served the request.
pub static PIPELINE_MISSES: Counter = Counter::new("omega.backend.pipeline_misses");
/// Wall-clock nanoseconds spent building those kernels or pipelines.
pub static PIPELINE_COMPILE_NANOS: Counter = Counter::new("omega.backend.pipeline_compile_nanos");

/// Wall-clock nanoseconds a device backend spent resolving and uploading named inputs.
pub static BLOCK_UPLOAD_NANOS: Counter = Counter::new("omega.backend.block_upload_nanos");
/// Wall-clock nanoseconds spent submitting kernel launches to the device queue.
pub static ENCODE_DISPATCH_NANOS: Counter = Counter::new("omega.backend.encode_dispatch_nanos");
/// Wall-clock nanoseconds spent waiting for the device to finish submitted work.
pub static GPU_EXEC_NANOS: Counter = Counter::new("omega.backend.gpu_exec_nanos");
/// Wall-clock nanoseconds spent copying outputs from device to host.
pub static READBACK_NANOS: Counter = Counter::new("omega.backend.readback_nanos");

/// One read's worth of stage counters, in backend-neutral units (bytes,
/// counts, nanoseconds). A backend that cannot measure a field leaves it `0`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageTotals {
    /// Bytes bound through a real host-to-device copy of a named block.
    pub block_copied_bytes: u64,
    /// Pipelines or kernels compiled (cache misses).
    pub pipeline_misses: u64,
    pub pipeline_compile_nanos: u64,
    pub prepare_nanos: u64,
    pub emit_nanos: u64,
    pub pipeline_lookup_nanos: u64,
    pub op_setup_nanos: u64,
    pub gpu_exec_nanos: u64,
    pub encode_dispatch_nanos: u64,
    pub readback_nanos: u64,
    pub block_upload_nanos: u64,
    pub resolve_steps_nanos: u64,
}

impl StageTotals {
    /// Field-wise sum, so several backends' reads of one step add up.
    #[must_use]
    pub fn merged(self, other: Self) -> Self {
        Self {
            block_copied_bytes: self.block_copied_bytes + other.block_copied_bytes,
            pipeline_misses: self.pipeline_misses + other.pipeline_misses,
            pipeline_compile_nanos: self.pipeline_compile_nanos + other.pipeline_compile_nanos,
            prepare_nanos: self.prepare_nanos + other.prepare_nanos,
            emit_nanos: self.emit_nanos + other.emit_nanos,
            pipeline_lookup_nanos: self.pipeline_lookup_nanos + other.pipeline_lookup_nanos,
            op_setup_nanos: self.op_setup_nanos + other.op_setup_nanos,
            gpu_exec_nanos: self.gpu_exec_nanos + other.gpu_exec_nanos,
            encode_dispatch_nanos: self.encode_dispatch_nanos + other.encode_dispatch_nanos,
            readback_nanos: self.readback_nanos + other.readback_nanos,
            block_upload_nanos: self.block_upload_nanos + other.block_upload_nanos,
            resolve_steps_nanos: self.resolve_steps_nanos + other.resolve_steps_nanos,
        }
    }
}

/// Records one named block's host-to-device copy of `byte_length` bytes.
pub fn record_block_copied(byte_length: u64) {
    counter!(BLOCK_COPIED_BYTES, byte_length);
}

/// Records one kernel or pipeline build that took `nanos` nanoseconds.
pub fn record_pipeline_compile(nanos: u64) {
    counter!(PIPELINE_MISSES, 1);
    counter!(PIPELINE_COMPILE_NANOS, nanos);
}

/// One executed plan's phase costs in nanoseconds, recorded by a backend that
/// times its own upload, submit, device-wait and readback phases.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StagePhases {
    pub block_upload_nanos: u64,
    pub encode_dispatch_nanos: u64,
    pub gpu_exec_nanos: u64,
    pub readback_nanos: u64,
}

/// Adds one executed plan's phase costs to the shared counters.
pub fn record_stage_phases(phases: StagePhases) {
    counter!(BLOCK_UPLOAD_NANOS, phases.block_upload_nanos);
    counter!(ENCODE_DISPATCH_NANOS, phases.encode_dispatch_nanos);
    counter!(GPU_EXEC_NANOS, phases.gpu_exec_nanos);
    counter!(READBACK_NANOS, phases.readback_nanos);
}

/// Reads and resets every backend's stage counters in one call: the shared
/// counters CUDA and wgpu feed, plus Metal's own counters mapped into the same
/// fields. Snapshot-and-reset, so the result is the delta since the last read.
#[must_use]
pub fn stage_totals() -> StageTotals {
    let shared = StageTotals {
        block_copied_bytes: BLOCK_COPIED_BYTES.snapshot_and_reset(),
        pipeline_misses: PIPELINE_MISSES.snapshot_and_reset(),
        pipeline_compile_nanos: PIPELINE_COMPILE_NANOS.snapshot_and_reset(),
        block_upload_nanos: BLOCK_UPLOAD_NANOS.snapshot_and_reset(),
        encode_dispatch_nanos: ENCODE_DISPATCH_NANOS.snapshot_and_reset(),
        gpu_exec_nanos: GPU_EXEC_NANOS.snapshot_and_reset(),
        readback_nanos: READBACK_NANOS.snapshot_and_reset(),
        ..StageTotals::default()
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        shared.merged(crate::metal::shared_stage_totals())
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        shared
    }
}
