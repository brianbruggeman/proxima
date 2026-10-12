//! The door walkthrough, shared by the `onnx_door_walkthrough` and
//! `door_walkthrough` examples and the `door_shared_` tests: it drives the one [`DoorStage`] machine through every
//! legal transition of an instance and through every stage's failure exit,
//! and returns one line per transition.
//!
//! The ONNX-in, CPU-out instance has five legal transitions and five failure
//! exits; this file holds that instance. The ONNX-in, MIL-out instance has
//! eight of each (the reader's three and the run's three, plus emit to
//! package, package to verify and verify back to lower) and lives in
//! `walkthrough_mil.rs`, which needs the `mil-writer` feature. A program that
//! did not come from a reader enters the writing stages through
//! [`DoorStage::emit`], which is an entry and not a transition, so it prints
//! no line.

use std::collections::BTreeMap;
use std::fmt::Display;

use anyhow::{Result, anyhow, bail};
use omega::door::{Door, DoorBatch, DoorError, DoorStage, NoSink, Writer};
use omega::onnx_door::{DoorConfig, OnnxReader};
use proxima_onnx::messages::{Dimension, DimensionValue, GraphProto, ModelProto, ValueInfoProto};
use proxima_onnx::write_model_proto;

use crate::door_support::{MLP_INPUT, fixed, float_input, mlp_batch, node, two_layer_mlp_bytes};

const READING_STEPS: usize = 5;
const EXIT_STEPS: usize = 5;

pub(crate) type Lines = Vec<String>;
pub(crate) type ReadingDoor<W> = Door<OnnxReader, W>;

pub(crate) fn untyped_output(name: &'static str) -> ValueInfoProto<'static> {
    ValueInfoProto {
        name,
        ..ValueInfoProto::default()
    }
}

pub(crate) fn encode(graph: GraphProto<'_>) -> Vec<u8> {
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

pub(crate) fn describe(error: impl Display) -> anyhow::Error {
    anyhow!("{error}")
}

pub(crate) fn door<W: Writer>(
    writer: W,
    bytes: &[u8],
    config: &DoorConfig,
) -> Result<ReadingDoor<W>> {
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

pub(crate) fn walk<'bytes, W: Writer>(
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

pub(crate) fn record_exit<'bytes, W: Writer>(
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

pub(crate) fn run_reading_path<W: Writer>(
    instance: &str,
    door: &ReadingDoor<W>,
    lines: &mut Lines,
) -> Result<()> {
    let bytes = two_layer_mlp_bytes();
    let entry = DoorStage::begin(door.reader(), &bytes);
    walk(instance, entry, READING_STEPS, door, &mlp_batch(), lines).map_err(describe)?;
    Ok(())
}

pub(crate) fn reading_exits<W: Writer + Copy>(
    instance: &str,
    writer: W,
    lines: &mut Lines,
) -> Result<()> {
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
