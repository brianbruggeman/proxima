//! The `MIL.proto` messages the writer emits, by field number.
//!
//! Field numbers are those of `Program`, `Function`, `Block`, `Operation`,
//! `Argument`, `Value` and `TensorType`; the vendored protos under
//! `omega/tests/fixtures/coreml-lane/proto/coreml/` are the reference and
//! `protoc --decode` the oracle.

use core::fmt::{self, Write};

use half::f16;

use crate::wire::{Encoder, Open};

const BLOCK_OPERATIONS: u32 = 3;
const OPERATION_TYPE: u32 = 1;
const OPERATION_INPUTS: u32 = 2;
const OPERATION_OUTPUTS: u32 = 3;
const OPERATION_ATTRIBUTES: u32 = 5;
const MAP_KEY: u32 = 1;
const MAP_VALUE: u32 = 2;
const ARGUMENT_BINDINGS: u32 = 1;
const BINDING_NAME: u32 = 1;
const BINDING_VALUE: u32 = 2;
const VALUE_TYPE: u32 = 2;
const VALUE_IMMEDIATE: u32 = 3;
const VALUE_BLOB: u32 = 5;
const IMMEDIATE_TENSOR: u32 = 1;
const TENSOR_INTS: u32 = 2;
const TENSOR_BOOLS: u32 = 3;
const TENSOR_STRINGS: u32 = 4;
const TENSOR_BYTES: u32 = 7;
const REPEATED_VALUES: u32 = 1;
const BLOB_FILE_NAME: u32 = 1;
const BLOB_OFFSET: u32 = 2;
const TENSOR_TYPE: u32 = 1;
const TENSOR_DATA_TYPE: u32 = 1;
const TENSOR_RANK: u32 = 2;
const TENSOR_DIMENSIONS: u32 = 3;
const DIMENSION_CONSTANT: u32 = 1;
const DIMENSION_SIZE: u32 = 1;
const NAMED_NAME: u32 = 1;
const NAMED_TYPE: u32 = 2;

/// The path every `BLOBFILE` names, relative to the compiled model.
pub(super) const WEIGHT_BLOB_PATH: &[u8] = b"@model_path/weights/weight.bin";

/// The MIL element types the writer uses; the discriminant is `DataType` in `MIL.proto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Bool = 1,
    Text = 2,
    Float16 = 10,
    Float32 = 11,
    Int32 = 23,
}

impl Kind {
    /// The `cast` operator's `dtype` string for this element type.
    pub(super) const fn cast_name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Text => "string",
            Self::Float16 => "fp16",
            Self::Float32 => "fp32",
            Self::Int32 => "int32",
        }
    }
}

/// A fixed-capacity identifier, so naming a node never allocates.
#[derive(Debug, Clone, Copy)]
pub(super) struct Name {
    bytes: [u8; Name::CAPACITY],
    length: usize,
}

impl Name {
    const CAPACITY: usize = 40;

    fn render(arguments: fmt::Arguments<'_>) -> Self {
        let mut name = Self {
            bytes: [0; Self::CAPACITY],
            length: 0,
        };
        let _ = name.write_fmt(arguments);
        name
    }

    /// `n<index>`: the output of a program node.
    pub(super) fn node(index: u32) -> Self {
        Self::render(format_args!("n{index}"))
    }

    /// `n<index>_<tag><step>`: an intermediate a node's emission creates.
    pub(super) fn step(index: u32, tag: &str, step: usize) -> Self {
        Self::render(format_args!("n{index}_{tag}{step}"))
    }

    /// The identifier `text` when it matches `[A-Za-z_][A-Za-z0-9_@]*` and fits.
    pub(super) fn identifier(text: &str) -> Option<Self> {
        let mut chars = text.chars();
        let head = chars
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == '_');
        let tail = chars.all(|next| next.is_ascii_alphanumeric() || next == '_' || next == '@');
        let mut name = Self::render(format_args!(""));
        (head && tail && name.write_str(text).is_ok()).then_some(name)
    }

    pub(super) fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.length]
    }
}

impl Default for Name {
    fn default() -> Self {
        Self::render(format_args!(""))
    }
}

impl Write for Name {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        let end = self.length + text.len();
        let slot = self.bytes.get_mut(self.length..end).ok_or(fmt::Error)?;
        slot.copy_from_slice(text.as_bytes());
        self.length = end;
        Ok(())
    }
}

/// `ValueType { tensorType }` at `field`: element type and constant dimensions.
pub(super) fn value_type(encoder: &mut Encoder<'_>, field: u32, kind: Kind, dims: &[u64]) {
    let value = encoder.open(field);
    let tensor = encoder.open(TENSOR_TYPE);
    encoder.varint_field(TENSOR_DATA_TYPE, kind as u64);
    encoder.varint_field(TENSOR_RANK, dims.len() as u64);
    dims.iter().for_each(|extent| dimension(encoder, *extent));
    encoder.close(tensor);
    encoder.close(value);
}

fn dimension(encoder: &mut Encoder<'_>, extent: u64) {
    let outer = encoder.open(TENSOR_DIMENSIONS);
    let constant = encoder.open(DIMENSION_CONSTANT);
    encoder.varint_field(DIMENSION_SIZE, extent);
    encoder.close(constant);
    encoder.close(outer);
}

/// `NamedValueType` at `field`.
pub(super) fn named_type(
    encoder: &mut Encoder<'_>,
    field: u32,
    name: &[u8],
    kind: Kind,
    dims: &[u64],
) {
    let named = encoder.open(field);
    encoder.bytes_field(NAMED_NAME, name);
    value_type(encoder, NAMED_TYPE, kind, dims);
    encoder.close(named);
}

/// Starts one `Operation` of the block and names its `type`.
pub(super) fn begin_operation(encoder: &mut Encoder<'_>, operator: &str) -> Open {
    let operation = encoder.open(BLOCK_OPERATIONS);
    encoder.bytes_field(OPERATION_TYPE, operator.as_bytes());
    operation
}

/// The single output an operation declares.
pub(super) fn operation_output(encoder: &mut Encoder<'_>, name: &Name, kind: Kind, dims: &[u64]) {
    named_type(encoder, OPERATION_OUTPUTS, name.as_bytes(), kind, dims);
}

fn begin_argument(encoder: &mut Encoder<'_>, key: &str) -> (Open, Open) {
    let entry = encoder.open(OPERATION_INPUTS);
    encoder.bytes_field(MAP_KEY, key.as_bytes());
    let argument = encoder.open(MAP_VALUE);
    (entry, argument)
}

/// An input bound to the output of an earlier node or step.
pub(super) fn input_name(encoder: &mut Encoder<'_>, key: &str, name: &Name) {
    let (entry, argument) = begin_argument(encoder, key);
    let binding = encoder.open(ARGUMENT_BINDINGS);
    encoder.bytes_field(BINDING_NAME, name.as_bytes());
    encoder.close(binding);
    encoder.close(argument);
    encoder.close(entry);
}

/// An immediate a `Value` carries inline.
#[derive(Debug, Clone, Copy)]
pub(super) enum Immediate<'a> {
    Ints(&'a [i64]),
    Int(i64),
    Flag(bool),
    Text(&'a str),
    Half(f16),
    HalfFill { value: f16, count: u64 },
}

impl Immediate<'_> {
    fn type_and_dims(&self) -> (Kind, [u64; 1], usize) {
        match self {
            Self::Ints(values) => (Kind::Int32, [values.len() as u64], 1),
            Self::Int(_) => (Kind::Int32, [0], 0),
            Self::Flag(_) => (Kind::Bool, [0], 0),
            Self::Text(_) => (Kind::Text, [0], 0),
            Self::Half(_) => (Kind::Float16, [0], 0),
            Self::HalfFill { count, .. } => (Kind::Float16, [*count], 1),
        }
    }
}

/// An input bound to an immediate value.
pub(super) fn input_immediate(encoder: &mut Encoder<'_>, key: &str, immediate: Immediate<'_>) {
    let (entry, argument) = begin_argument(encoder, key);
    let binding = encoder.open(ARGUMENT_BINDINGS);
    value_immediate(encoder, BINDING_VALUE, immediate);
    encoder.close(binding);
    encoder.close(argument);
    encoder.close(entry);
}

/// A `Value` holding `immediate`, at `field` of the enclosing message.
pub(super) fn value_immediate(encoder: &mut Encoder<'_>, field: u32, immediate: Immediate<'_>) {
    let (kind, dims, rank) = immediate.type_and_dims();
    let value = encoder.open(field);
    value_type(encoder, VALUE_TYPE, kind, &dims[..rank]);
    let inline = encoder.open(VALUE_IMMEDIATE);
    let tensor = encoder.open(IMMEDIATE_TENSOR);
    tensor_body(encoder, immediate);
    encoder.close(tensor);
    encoder.close(inline);
    encoder.close(value);
}

fn tensor_body(encoder: &mut Encoder<'_>, immediate: Immediate<'_>) {
    match immediate {
        Immediate::Ints(values) => repeated_ints(encoder, values),
        Immediate::Int(value) => repeated_ints(encoder, &[value]),
        Immediate::Flag(flag) => repeated_flag(encoder, flag),
        Immediate::Text(text) => repeated_text(encoder, text),
        Immediate::Half(value) => half_bytes(encoder, value, 1),
        Immediate::HalfFill { value, count } => half_bytes(encoder, value, count),
    }
}

fn repeated_ints(encoder: &mut Encoder<'_>, values: &[i64]) {
    let field = encoder.open(TENSOR_INTS);
    encoder.packed_varints(REPEATED_VALUES, values);
    encoder.close(field);
}

fn repeated_flag(encoder: &mut Encoder<'_>, flag: bool) {
    let field = encoder.open(TENSOR_BOOLS);
    encoder.bytes_field(REPEATED_VALUES, &[u8::from(flag)]);
    encoder.close(field);
}

fn repeated_text(encoder: &mut Encoder<'_>, text: &str) {
    let field = encoder.open(TENSOR_STRINGS);
    encoder.bytes_field(REPEATED_VALUES, text.as_bytes());
    encoder.close(field);
}

fn half_bytes(encoder: &mut Encoder<'_>, value: f16, count: u64) {
    let field = encoder.open(TENSOR_BYTES);
    let body = encoder.open(REPEATED_VALUES);
    (0..count).for_each(|_| encoder.raw(&value.to_le_bytes()));
    encoder.close(body);
    encoder.close(field);
}

/// A `const` attribute `key` holding a tensor read from the weight blob at `offset`.
pub(super) fn attribute_blob(encoder: &mut Encoder<'_>, key: &str, dims: &[u64], offset: usize) {
    let entry = encoder.open(OPERATION_ATTRIBUTES);
    encoder.bytes_field(MAP_KEY, key.as_bytes());
    let value = encoder.open(MAP_VALUE);
    value_type(encoder, VALUE_TYPE, Kind::Float16, dims);
    let blob = encoder.open(VALUE_BLOB);
    encoder.bytes_field(BLOB_FILE_NAME, WEIGHT_BLOB_PATH);
    encoder.varint_field(BLOB_OFFSET, offset as u64);
    encoder.close(blob);
    encoder.close(value);
    encoder.close(entry);
}

/// A `const` attribute `key` holding an inline value.
pub(super) fn attribute_immediate(encoder: &mut Encoder<'_>, key: &str, immediate: Immediate<'_>) {
    let entry = encoder.open(OPERATION_ATTRIBUTES);
    encoder.bytes_field(MAP_KEY, key.as_bytes());
    value_immediate(encoder, MAP_VALUE, immediate);
    encoder.close(entry);
}
