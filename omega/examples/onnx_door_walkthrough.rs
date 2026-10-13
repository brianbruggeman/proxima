//! Drives the ONNX door's one state machine through every legal transition and
//! every stage's failure exit and prints one line per transition.
//!
//! Five legal transitions (parsed to validated, validated to lowered, lowered
//! to planned, planned to executed, executed to planned) and five failure
//! exits, one per stage: parse, validate, lower, plan, execute. The door is
//! `Door<OnnxReader, NoSink>`, the same `Door` and `DoorStage` every format
//! instantiates; `door_walkthrough` adds the MIL-out instance behind
//! `mil-writer`.
//!
//! ```text
//! cargo run -p proxima-omega --features onnx-door,cpu --example onnx_door_walkthrough
//! ```

#[path = "../tests/door_support/mod.rs"]
mod door_support;
#[path = "../tests/door_support/walkthrough.rs"]
mod walkthrough;

use anyhow::Result;

fn main() -> Result<()> {
    walkthrough::onnx_lines()?
        .iter()
        .for_each(|line| println!("{line}"));
    Ok(())
}
