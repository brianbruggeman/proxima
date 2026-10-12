//! The MIL-out half of the door walkthrough: the writing stages (emit,
//! package, verify) the ONNX-in, CPU-out instance never reaches. Needs the
//! `mil-writer` feature; the reading half is in `walkthrough.rs`.

use anyhow::{Result, bail};
use omega::door::{DoorBatch, DoorStage, MilSink, StageKind, Writer};
use omega::onnx_door::DoorConfig;
use proxima_mil::writer::MilPackage;
use proxima_onnx::lower::lower_graph;
use proxima_onnx::messages::GraphProto;
use proxima_onnx::pipe::parse_complete;
use proxima_tensor::Lowered;

use crate::door_support::{fixed, float_input, node, two_layer_mlp_bytes, weights};
use crate::walkthrough::{
    Lines, ReadingDoor, describe, door, encode, reading_exits, record_exit, run_reading_path,
    untyped_output, walk,
};

const WRITING_STEPS: usize = 3;

fn conv_bytes() -> Vec<u8> {
    let kernel = vec![0_u8; 9 * size_of::<f32>()];
    encode(GraphProto {
        node: vec![node("Conv", &["image", "kernel"], "y")],
        initializer: vec![weights("kernel", &[1, 1, 3, 3], &kernel)],
        input: vec![float_input("image", [1, 1, 4, 4].map(fixed).to_vec())],
        output: vec![untyped_output("y")],
        ..GraphProto::default()
    })
}

fn batched_matmul_bytes() -> Vec<u8> {
    let right = vec![0_u8; 40 * size_of::<f32>()];
    encode(GraphProto {
        node: vec![node("MatMul", &["left", "right"], "y")],
        initializer: vec![weights("right", &[2, 4, 5], &right)],
        input: vec![float_input("left", [2, 2, 3, 4].map(fixed).to_vec())],
        output: vec![untyped_output("y")],
        ..GraphProto::default()
    })
}

fn lowered_of(bytes: &[u8]) -> Result<Lowered> {
    let model = parse_complete(bytes).map_err(describe)?;
    match model.graph {
        Some(graph) => lower_graph(&graph).map_err(describe),
        None => bail!("the walkthrough model carries no graph"),
    }
}

fn run_writing_path<W: Writer>(
    instance: &str,
    door: &ReadingDoor<W>,
    lines: &mut Lines,
) -> Result<()> {
    let entry = DoorStage::emit(door.writer(), lowered_of(&two_layer_mlp_bytes())?);
    walk(
        instance,
        entry,
        WRITING_STEPS,
        door,
        &DoorBatch::default(),
        lines,
    )
    .map_err(describe)?;
    Ok(())
}

fn writing_exits(instance: &str, lines: &mut Lines) -> Result<()> {
    let sink = door(MilSink, &[], &DoorConfig::default())?;
    let batch = DoorBatch::default();
    for bytes in [conv_bytes(), batched_matmul_bytes()] {
        let entry = DoorStage::emit(sink.writer(), lowered_of(&bytes)?);
        record_exit(instance, entry, &sink, &batch, lines)?;
    }
    let packaged = Ok(DoorStage::Packaged {
        lowered: lowered_of(&two_layer_mlp_bytes())?,
        package: emptied_model_package(&sink)?,
    });
    record_exit(instance, packaged, &sink, &batch, lines)
}

fn emptied_model_package(sink: &ReadingDoor<MilSink>) -> Result<MilPackage> {
    let package = sink
        .write(lowered_of(&two_layer_mlp_bytes())?)
        .map_err(describe)?;
    Ok(MilPackage {
        model: Vec::new(),
        ..package
    })
}

fn stage_rank(line: &str) -> usize {
    StageKind::ALL
        .iter()
        .position(|kind| line.contains(&format!("stage={} ", kind.name())))
        .unwrap_or(StageKind::ALL.len())
}

/// The ONNX-in, MIL-out instance: eight legal transitions, then the failure
/// exit of each of the eight stages in pipeline order.
///
/// # Errors
/// A stage that succeeded where the walkthrough needs it to fail, or the
/// reverse.
pub fn mil_lines() -> Result<Lines> {
    let mut lines = Lines::new();
    let sink = door(MilSink, &[], &DoorConfig::default())?;
    run_reading_path("mil", &sink, &mut lines)?;
    run_writing_path("mil", &sink, &mut lines)?;
    let mut exits = Lines::new();
    reading_exits("mil", MilSink, &mut exits)?;
    writing_exits("mil", &mut exits)?;
    exits.sort_by_key(|line| stage_rank(line));
    lines.extend(exits);
    Ok(lines)
}
