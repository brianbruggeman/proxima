//! The door walkthrough, shared by the `door_walkthrough` example and the
//! `door_shared_` tests: it drives the one [`DoorStage`] machine through every
//! legal transition of an instance and through every stage's failure exit,
//! and returns one line per transition.
//!
//! The ONNX-in, CPU-out instance has five legal transitions and five failure
//! exits. The ONNX-in, MIL-out instance has eight of each: the reader's three
//! and the run's three, plus emit to package, package to verify and verify
//! back to lower. A program that did not come from a reader enters the
//! writing stages through [`DoorStage::emit`], which is an entry and not a
//! transition, so it prints no line.

use std::collections::BTreeMap;
use std::fmt::Display;

use anyhow::{Result, anyhow, bail};
use omega::door::{Door, DoorBatch, DoorError, DoorStage, MilSink, NoSink, StageKind, Writer};
use omega::onnx_door::{DoorConfig, OnnxReader};
use proxima_mil::writer::MilPackage;
use proxima_onnx::lower::lower_graph;
use proxima_onnx::messages::{Dimension, DimensionValue, GraphProto, ModelProto, ValueInfoProto};
use proxima_onnx::pipe::parse_complete;
use proxima_onnx::write_model_proto;
use proxima_tensor::Lowered;

use crate::door_support::{
    MLP_INPUT, fixed, float_input, mlp_batch, node, two_layer_mlp_bytes, weights,
};

const READING_STEPS: usize = 5;
const WRITING_STEPS: usize = 3;
const EXIT_STEPS: usize = 5;

type Lines = Vec<String>;
type ReadingDoor<W> = Door<OnnxReader, W>;

fn untyped_output(name: &'static str) -> ValueInfoProto<'static> {
    ValueInfoProto {
        name,
        ..ValueInfoProto::default()
    }
}

fn encode(graph: GraphProto<'_>) -> Vec<u8> {
    write_model_proto(&ModelProto {
        ir_version: 8,
        graph: Some(graph),
        ..ModelProto::default()
    })
}

fn relu_over_symbolic_batch_bytes() -> Vec<u8> {
    let batch = Dimension {
        value: Some(DimensionValue::Param("batch_size")),
        denotation: "",
    };
    encode(GraphProto {
        node: vec![node("Relu", &["x"], "y")],
        input: vec![float_input("x", vec![batch, fixed(3)])],
        output: vec![untyped_output("y")],
        ..GraphProto::default()
    })
}

fn dangling_add_bytes() -> Vec<u8> {
    encode(GraphProto {
        node: vec![node("Add", &["x", "ghost"], "y")],
        input: vec![float_input("x", vec![fixed(2), fixed(3)])],
        output: vec![untyped_output("y")],
        ..GraphProto::default()
    })
}

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

fn describe(error: impl Display) -> anyhow::Error {
    anyhow!("{error}")
}

fn lowered_of(bytes: &[u8]) -> Result<Lowered> {
    let model = parse_complete(bytes).map_err(describe)?;
    match model.graph {
        Some(graph) => lower_graph(&graph).map_err(describe),
        None => bail!("the walkthrough model carries no graph"),
    }
}

fn door<W: Writer>(writer: W, bytes: &[u8], config: &DoorConfig) -> Result<ReadingDoor<W>> {
    let settings = config.settings().map_err(describe)?;
    Ok(Door::new(OnnxReader::new(), writer, bytes, settings))
}

fn failure_line<W: Writer>(instance: &str, error: &DoorError<OnnxReader, W>) -> String {
    let stage = error.stage().name();
    format!("transition instance={instance} stage={stage} outcome=failed error=\"{error}\"")
}

fn legal_line(instance: &str, from: &str, to: &str) -> String {
    format!("transition instance={instance} from={from} to={to} outcome=advanced")
}

fn advance<'bytes, W: Writer>(
    instance: &str,
    stage: DoorStage<'bytes, OnnxReader, W>,
    door: &ReadingDoor<W>,
    batch: &DoorBatch,
    lines: &mut Lines,
) -> Result<DoorStage<'bytes, OnnxReader, W>, DoorError<OnnxReader, W>> {
    let from = stage.kind().name();
    let next = stage.advance(door, batch)?;
    lines.push(legal_line(instance, from, next.kind().name()));
    Ok(next)
}

fn walk<'bytes, W: Writer>(
    instance: &str,
    entry: Result<DoorStage<'bytes, OnnxReader, W>, DoorError<OnnxReader, W>>,
    steps: usize,
    door: &ReadingDoor<W>,
    batch: &DoorBatch,
    lines: &mut Lines,
) -> Result<DoorStage<'bytes, OnnxReader, W>, DoorError<OnnxReader, W>> {
    let first = entry?;
    (0..steps).try_fold(first, |stage, _| {
        advance(instance, stage, door, batch, lines)
    })
}

fn record_exit<'bytes, W: Writer>(
    instance: &str,
    entry: Result<DoorStage<'bytes, OnnxReader, W>, DoorError<OnnxReader, W>>,
    door: &ReadingDoor<W>,
    batch: &DoorBatch,
    lines: &mut Lines,
) -> Result<()> {
    let mut discarded = Lines::new();
    match walk(instance, entry, EXIT_STEPS, door, batch, &mut discarded) {
        Err(error) => lines.push(failure_line(instance, &error)),
        Ok(stage) => bail!(
            "{instance} expected a failure exit, reached {:?}",
            stage.kind()
        ),
    }
    Ok(())
}

fn run_reading_path<W: Writer>(
    instance: &str,
    door: &ReadingDoor<W>,
    lines: &mut Lines,
) -> Result<()> {
    let bytes = two_layer_mlp_bytes();
    let entry = DoorStage::begin(door.reader(), &bytes);
    walk(instance, entry, READING_STEPS, door, &mlp_batch(), lines).map_err(describe)?;
    Ok(())
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

fn reading_exits<W: Writer + Copy>(instance: &str, writer: W, lines: &mut Lines) -> Result<()> {
    let wgpu = DoorConfig::builder()
        .engine("gpu")
        .gpu_driver("wgpu")
        .build();
    let mlp = two_layer_mlp_bytes();
    let short = DoorBatch {
        pins: BTreeMap::new(),
        inputs: vec![("x".to_owned(), MLP_INPUT[..5].to_vec())],
    };
    let cases = [
        (
            mlp[..mlp.len() - 3].to_vec(),
            DoorConfig::default(),
            mlp_batch(),
        ),
        (
            relu_over_symbolic_batch_bytes(),
            DoorConfig::default(),
            mlp_batch(),
        ),
        (dangling_add_bytes(), DoorConfig::default(), mlp_batch()),
        (mlp.clone(), wgpu, mlp_batch()),
        (mlp, DoorConfig::default(), short),
    ];
    for (bytes, config, batch) in cases {
        let reading = door(writer, &bytes, &config)?;
        record_exit(
            instance,
            DoorStage::begin(reading.reader(), &bytes),
            &reading,
            &batch,
            lines,
        )?;
    }
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

/// The ONNX-in, CPU-out instance: five legal transitions, then the failure
/// exit of parse, validate, lower, plan and execute.
///
/// # Errors
/// A stage that succeeded where the walkthrough needs it to fail, or the
/// reverse.
pub fn onnx_lines() -> Result<Lines> {
    let mut lines = Lines::new();
    run_reading_path(
        "onnx",
        &door(NoSink, &[], &DoorConfig::default())?,
        &mut lines,
    )?;
    reading_exits("onnx", NoSink, &mut lines)?;
    Ok(lines)
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
