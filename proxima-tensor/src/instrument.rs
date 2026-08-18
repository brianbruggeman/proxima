//! Real execution-witness counters for [`crate::cpu`]'s bound-op kernels,
//! gated entirely behind the `instrument` feature.
//!
//! Every field here is incremented from a plain local accumulator inside
//! the kernel loop and committed to the process-wide [`proxima_telemetry`]
//! counters exactly once, at the end of the bound-op call — never as an
//! atomic increment inside a loop that can run ~1e9 times, or the
//! instrument would perturb the thing it measures.

use proxima_telemetry::counter;
use proxima_telemetry::metric::Counter;

pub static BOUND_OPS: Counter = Counter::new("proxima_tensor.bound_ops");
pub static MAC_OPS: Counter = Counter::new("proxima_tensor.mac_ops");
pub static OPERAND_LOADS: Counter = Counter::new("proxima_tensor.operand_loads");
pub static DISTINCT_OPERAND_ELEMENTS: Counter = Counter::new("proxima_tensor.distinct_operand_elements");
pub static OUTPUT_WRITES: Counter = Counter::new("proxima_tensor.output_writes");
pub static PATH_DOT_FAST: Counter = Counter::new("proxima_tensor.path.dot_fast");
pub static PATH_WIDTH_FAST: Counter = Counter::new("proxima_tensor.path.width_fast");
pub static PATH_GENERIC: Counter = Counter::new("proxima_tensor.path.generic");
pub static LEADING_ITERS: Counter = Counter::new("proxima_tensor.leading_iters");
pub static KERNEL_CALLS: Counter = Counter::new("proxima_tensor.kernel_calls");

/// Which straight-line branch a bound op actually took — set once per call,
/// never re-derived from tensor shape after the fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    DotFast,
    WidthFast,
    Generic,
}

/// One bound-op call's local tally. All fields are plain `u64`s bumped by
/// real per-branch loop bounds already in scope at the call site (`width`,
/// `reduction_total`, `raw.len()`, per-operand contiguity) — never a
/// quantity re-derived from the op's declared extents after the fact.
#[derive(Debug, Default)]
pub struct KernelCounters {
    pub mac_ops: u64,
    pub operand_loads: u64,
    pub output_writes: u64,
    pub leading_iters: u64,
    pub kernel_calls: u64,
}

impl KernelCounters {
    pub fn commit(&self, path: Path, distinct_operand_elements: u64) {
        counter!(BOUND_OPS, 1);
        counter!(MAC_OPS, self.mac_ops);
        counter!(OPERAND_LOADS, self.operand_loads);
        counter!(DISTINCT_OPERAND_ELEMENTS, distinct_operand_elements);
        counter!(OUTPUT_WRITES, self.output_writes);
        counter!(LEADING_ITERS, self.leading_iters);
        counter!(KERNEL_CALLS, self.kernel_calls);
        match path {
            Path::DotFast => counter!(PATH_DOT_FAST, 1),
            Path::WidthFast => counter!(PATH_WIDTH_FAST, 1),
            Path::Generic => counter!(PATH_GENERIC, 1),
        }
    }
}

/// Snapshot of every counter's running total, for a caller (the
/// `profile_hot` example) that wants to print them without pulling in the
/// full recorder/exporter machinery.
#[derive(Debug, Clone, Copy, Default)]
pub struct Totals {
    pub bound_ops: u64,
    pub mac_ops: u64,
    pub operand_loads: u64,
    pub distinct_operand_elements: u64,
    pub output_writes: u64,
    pub path_dot_fast: u64,
    pub path_width_fast: u64,
    pub path_generic: u64,
    pub leading_iters: u64,
    pub kernel_calls: u64,
}

#[must_use]
pub fn totals() -> Totals {
    Totals {
        bound_ops: BOUND_OPS.get(),
        mac_ops: MAC_OPS.get(),
        operand_loads: OPERAND_LOADS.get(),
        distinct_operand_elements: DISTINCT_OPERAND_ELEMENTS.get(),
        output_writes: OUTPUT_WRITES.get(),
        path_dot_fast: PATH_DOT_FAST.get(),
        path_width_fast: PATH_WIDTH_FAST.get(),
        path_generic: PATH_GENERIC.get(),
        leading_iters: LEADING_ITERS.get(),
        kernel_calls: KERNEL_CALLS.get(),
    }
}

/// Resets every counter to zero — lets a caller isolate one program's totals
/// from whatever ran earlier in the same process.
pub fn reset() {
    let _ = BOUND_OPS.snapshot_and_reset();
    let _ = MAC_OPS.snapshot_and_reset();
    let _ = OPERAND_LOADS.snapshot_and_reset();
    let _ = DISTINCT_OPERAND_ELEMENTS.snapshot_and_reset();
    let _ = OUTPUT_WRITES.snapshot_and_reset();
    let _ = PATH_DOT_FAST.snapshot_and_reset();
    let _ = PATH_WIDTH_FAST.snapshot_and_reset();
    let _ = PATH_GENERIC.snapshot_and_reset();
    let _ = LEADING_ITERS.snapshot_and_reset();
    let _ = KERNEL_CALLS.snapshot_and_reset();
}
