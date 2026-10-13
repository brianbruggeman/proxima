//! Drives the one door state machine through every legal transition and every
//! stage's failure exit and prints one line per transition.
//!
//! `onnx` runs the ONNX-in, CPU-out instance (`Door<OnnxReader, NoSink>`):
//! 5 legal transitions and 5 failure exits. `mil` runs the ONNX-in, MIL-out
//! instance (`Door<OnnxReader, MilSink>`): 8 and 8. With no argument both run.
//! Both instances are the same `Door` and the same `DoorStage`; the MIL one
//! enters the emit, package and verify stages the ONNX one never reaches.
//!
//! ```text
//! cargo run -p proxima-omega --features onnx-door,cpu,mil-writer --example door_walkthrough -- mil
//! ```

#[path = "../tests/door_support/mod.rs"]
mod door_support;
#[path = "../tests/door_support/walkthrough.rs"]
mod walkthrough;
#[path = "../tests/door_support/walkthrough_mil.rs"]
mod walkthrough_mil;

use anyhow::{Result, bail};

fn main() -> Result<()> {
    let lines = match std::env::args().nth(1).as_deref() {
        Some("onnx") => walkthrough::onnx_lines()?,
        Some("mil") => walkthrough_mil::mil_lines()?,
        None => [walkthrough::onnx_lines()?, walkthrough_mil::mil_lines()?].concat(),
        Some(other) => bail!("unknown instance `{other}`, expected `onnx` or `mil`"),
    };
    lines.iter().for_each(|line| println!("{line}"));
    Ok(())
}
