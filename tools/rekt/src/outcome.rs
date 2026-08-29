use std::time::Duration;

/// what one fired arrival produced. `timed_out` marks an in-flight overflow —
/// the gate was full so the arrival was shed as a timeout — and is kept
/// distinct from a server-side error so the report can tell coordinated-omission
/// shedding apart from a real failure.
pub struct Outcome {
    pub latency: Duration,
    pub ok: bool,
    pub timed_out: bool,
}
