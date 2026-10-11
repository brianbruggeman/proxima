//! The tier-1 build of `proxima-mil` (`--no-default-features --features alloc`)
//! compiles the census, the wire encoder, the weight writer, the writer and
//! the sizing module, and the wire encoder and weight writer import nothing
//! from `alloc`. The emitted bytes are judged against a model derived by hand
//! from the protobuf wire rules and the field numbers of the vendored
//! `Model.proto` and `MIL.proto`, not against the encoder's own helpers.

#![cfg(feature = "alloc")]

use std::collections::BTreeSet;

use proxima_mil::MODULES;
use proxima_mil::census::{SCALAR_OPS, scalar_form};
use proxima_mil::weights::BlobWriter;
use proxima_mil::wire::{Encoder, WireError};
use proxima_mil::writer::{Context, MilRequest, MilWriter, emit_node};
use proxima_tensor::{
    DType, Extent, IndexMap, NodeId, Op, ScalarOp, Shapes, append, infer, projection,
};

const CARGO_TOML: &str = include_str!("../Cargo.toml");
const WIRE_SOURCE: &str = include_str!("../src/wire.rs");
const WEIGHTS_SOURCE: &str = include_str!("../src/weights.rs");
const ALLOC_PATH: &str = "alloc::";

const FLOAT16: u8 = 10;
const FLOAT32: u8 = 11;
const STRING: u8 = 2;

fn delimited(key: &[u8], body: &[u8]) -> Vec<u8> {
    let mut bytes = key.to_vec();
    let mut length = body.len();
    while length >= 0x80 {
        bytes.push((length & 0x7f) as u8 | 0x80);
        length >>= 7;
    }
    bytes.push(length as u8);
    bytes.extend_from_slice(body);
    bytes
}

fn text(key: u8, value: &str) -> Vec<u8> {
    delimited(&[key], value.as_bytes())
}

fn tensor_type(data_type: u8, dimension: Option<u8>) -> Vec<u8> {
    let mut body = vec![0x08, data_type, 0x10, u8::from(dimension.is_some())];
    if let Some(size) = dimension {
        let constant = delimited(&[0x0a], &[0x08, size]);
        body.extend(delimited(&[0x1a], &constant));
    }
    delimited(&[0x0a], &body)
}

fn feature(key: u8, name: &str, extent: u8) -> Vec<u8> {
    let array = [vec![0x0a, 0x01, extent], vec![0x10, 0xa0, 0x80, 0x04]].concat();
    let kind = delimited(&[0x2a], &array);
    let body = [text(0x0a, name), delimited(&[0x1a], &kind)].concat();
    delimited(&[key], &body)
}

fn named_type(key: u8, name: &str, data_type: u8, extent: u8) -> Vec<u8> {
    let value_type = delimited(&[0x12], &tensor_type(data_type, Some(extent)));
    delimited(&[key], &[text(0x0a, name), value_type].concat())
}

fn argument_entry(name: &str, binding: &[u8]) -> Vec<u8> {
    let value = delimited(&[0x12], &delimited(&[0x0a], binding));
    delimited(&[0x12], &[text(0x0a, name), value].concat())
}

fn cast_target(label: &str) -> Vec<u8> {
    let strings = delimited(&[0x22], &text(0x0a, label));
    let immediate = delimited(&[0x1a], &delimited(&[0x0a], &strings));
    let value = [delimited(&[0x12], &tensor_type(STRING, None)), immediate].concat();
    argument_entry("dtype", &delimited(&[0x12], &value))
}

fn operation(kind: &str, inputs: &[Vec<u8>], output: &[u8]) -> Vec<u8> {
    let body = [
        vec![text(0x0a, kind)],
        inputs.to_vec(),
        vec![output.to_vec()],
    ]
    .concat();
    delimited(&[0x1a], &body.concat())
}

fn named_input(slot: &str, source: &str) -> Vec<u8> {
    argument_entry(slot, &text(0x0a, source))
}

fn node_output(name: &str, data_type: u8) -> Vec<u8> {
    let value_type = delimited(&[0x12], &tensor_type(data_type, Some(4)));
    delimited(&[0x1a], &[text(0x0a, name), value_type].concat())
}

fn hand_derived_operations() -> [Vec<u8>; 3] {
    [
        operation(
            "cast",
            &[named_input("x", "x"), cast_target("fp16")],
            &node_output("n0", FLOAT16),
        ),
        operation(
            "add",
            &[named_input("x", "n0"), named_input("y", "n0")],
            &node_output("n1", FLOAT16),
        ),
        operation(
            "cast",
            &[named_input("x", "n1"), cast_target("fp32")],
            &node_output("y", FLOAT32),
        ),
    ]
}

fn hand_derived_model() -> Vec<u8> {
    let description = [feature(0x0a, "x", 4), feature(0x52, "y", 4)].concat();
    let block_body = [text(0x12, "y"), hand_derived_operations().concat()].concat();
    let block = [text(0x0a, "CoreML6"), delimited(&[0x12], &block_body)].concat();
    let function = [
        named_type(0x0a, "x", FLOAT32, 4),
        text(0x12, "CoreML6"),
        delimited(&[0x1a], &block),
    ]
    .concat();
    let entry = [text(0x0a, "main"), delimited(&[0x12], &function)].concat();
    let program = [vec![0x08, 0x01], delimited(&[0x12], &entry)].concat();
    [
        vec![0x08, 0x07],
        delimited(&[0x12], &description),
        delimited(&[0xb2, 0x1f], &program),
    ]
    .concat()
}

struct AddProgram {
    program: Vec<Op>,
    shapes: Shapes,
}

fn add_x_to_itself() -> AddProgram {
    let identity = || IndexMap::Affine(projection(1, &[0]));
    let mut program = vec![Op::Input {
        dtype: DType::Float32,
        shape: vec![Extent::Static(4)],
        name: Some("x".into()),
    }];
    append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![(NodeId(0), identity()), (NodeId(0), identity())],
            name: None,
        },
    );
    match infer(&program, &[]) {
        Ok(shapes) => AddProgram { program, shapes },
        Err(source) => panic!("the add program must have shapes: {source}"),
    }
}

fn emitted_model(fixture: &AddProgram) -> Vec<u8> {
    let request = MilRequest {
        program: &fixture.program,
        shapes: &fixture.shapes,
        weights: &[],
        output: "y",
    };
    match MilWriter::new().write(&request) {
        Ok(package) => package.model,
        Err(source) => panic!("the add program must emit: {source}"),
    }
}

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

#[test]
fn tier1_modules_built_are_census_wire_weights_writer_sized() {
    assert_eq!(MODULES, ["census", "wire", "weights", "writer", "sized"]);
    assert!(!MODULES.is_empty(), "a tier build of zero modules is red");
}

#[test]
fn tier1_default_features_are_empty() {
    let manifest = match CARGO_TOML.parse::<toml::Table>() {
        Ok(table) => table,
        Err(source) => panic!("Cargo.toml must parse: {source}"),
    };

    let defaults = manifest["features"]["default"].as_array();

    assert_eq!(defaults.map(Vec::len), Some(0));
}

#[test]
fn tier1_census_table_covers_all_17_scalar_ops() {
    let distinct: BTreeSet<_> = SCALAR_OPS.iter().map(|body| format!("{body:?}")).collect();

    let unnamed = SCALAR_OPS
        .iter()
        .filter(|body| scalar_form(**body).operators().is_empty())
        .count();

    assert_eq!(SCALAR_OPS.len(), 17);
    assert_eq!(distinct.len(), 17);
    assert_eq!(unnamed, 0);
}

#[test]
fn tier1_one_operation_program_emits_the_hand_derived_model_bytes() {
    let fixture = add_x_to_itself();

    let emitted = emitted_model(&fixture);

    assert_eq!(emitted, hand_derived_model());
}

#[test]
fn tier1_hand_derived_bytes_altered_by_one_fail() {
    let fixture = add_x_to_itself();
    let mut altered = hand_derived_model();
    let middle = altered.len() / 2;
    altered[middle] ^= 0x01;

    let emitted = emitted_model(&fixture);

    assert_ne!(emitted, altered);
}

#[test]
fn tier1_buffer_too_small_is_a_typed_error_carrying_the_needed_length() {
    let fixture = add_x_to_itself();
    let context = Context {
        program: &fixture.program,
        shapes: &fixture.shapes,
        weights: &[],
    };
    let [cast, add, _] = hand_derived_operations();
    let needed = cast.len() + add.len();
    let mut short = vec![0_u8; needed - 1];
    let mut encoder = Encoder::new(&mut short);
    let mut blob = BlobWriter::new(&mut []);

    for index in 0..fixture.program.len() {
        let emitted = emit_node(&context, NodeId(index as u32), &mut encoder, &mut blob);
        assert!(emitted.is_ok(), "node {index} must not fail on overflow");
    }

    assert_eq!(encoder.finish(), Err(WireError::BufferTooSmall { needed }));
}

#[test]
fn tier1_core_only_modules_import_nothing_from_alloc() {
    let wire = count_occurrences(WIRE_SOURCE, ALLOC_PATH);
    let weights = count_occurrences(WEIGHTS_SOURCE, ALLOC_PATH);

    assert_eq!((wire, weights), (0, 0));
}

#[test]
fn tier1_alloc_counter_control() {
    let planted = "use alloc::vec::Vec;";

    assert_eq!(count_occurrences(planted, ALLOC_PATH), 1);
    assert_eq!(count_occurrences("use core::fmt;", ALLOC_PATH), 0);
}
