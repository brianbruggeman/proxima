//! ROW 570's ladder: wall-clock time around one call to
//! `execute_plan_named_with_placements` for the eight-round Q4_K fixture
//! shared with `tests/horizontal_merge_dispatch.rs`. Which of "8 separate
//! dispatches" or "1 depth-8 dispatch" this measures is decided by the
//! feature set the binary was built with (`metal-horizontal-merge` on or
//! off); run it under both to get both arms:
//!
//! `cargo run -p proxima-omega --release --features metal --example horizontal_merge_ladder`
//! `cargo run -p proxima-omega --release --features metal,metal-horizontal-merge --example horizontal_merge_ladder`

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "../tests/support/horizontal_merge_fixture.rs"]
mod fixture;

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    let fixture = fixture::build_fixture();
    let named = fixture.named();
    let input_placements = fixture.input_placements();
    let output_placements = fixture.output_placements();
    for _ in 0..2 {
        omega::execute_plan_named_with_placements(
            &fixture.plan,
            &named,
            &input_placements,
            &output_placements,
        )
        .expect("warmup run executes");
    }
    let mut samples_us: Vec<f64> = Vec::with_capacity(7);
    for _ in 0..7 {
        let started = std::time::Instant::now();
        omega::execute_plan_named_with_placements(
            &fixture.plan,
            &named,
            &input_placements,
            &output_placements,
        )
        .expect("timed run executes");
        samples_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
    }
    fixture.assert_outputs_match_reference(1e-2);
    let mut sorted = samples_us.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    eprintln!(
        "row570_ladder metal-horizontal-merge={} samples_us={samples_us:?} median_us={median}",
        cfg!(feature = "metal-horizontal-merge")
    );
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    println!("horizontal_merge_ladder requires --features metal on macOS; skipped");
}
