//! Timing probes over the cpu evaluator's private dispatch internals, run on
//! demand because a wall-clock sweep is a measurement, not a pass/fail gate.
//!
//! `cargo run -p proxima-tensor --release --example cpu_probes -- <probe>`
//! with `row_oversubscribe`, `attention_tile_shapes` or
//! `narrow_width_tile_shapes` (the last two exist on aarch64 only).

use std::env;
use std::process::ExitCode;

use proxima_tensor::cpu::probes;

fn main() -> ExitCode {
    let probe = env::args().nth(1).unwrap_or_default();
    match probe.as_str() {
        "row_oversubscribe" => probes::row_oversubscribe_picks_the_multiplier(),
        #[cfg(target_arch = "aarch64")]
        "attention_tile_shapes" => probes::attention_tile_shapes_nano(),
        #[cfg(target_arch = "aarch64")]
        "narrow_width_tile_shapes" => probes::narrow_width_tile_shapes_nano(),
        other => {
            eprintln!("unknown probe {other:?}; expected row_oversubscribe, attention_tile_shapes or narrow_width_tile_shapes");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
