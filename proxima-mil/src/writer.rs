//! `MilWriter`: a lowered program, its shapes and its weight views to an
//! `.mlpackage` held as three in-memory files.
//!
//! `MilWriter` is a [`Pipe`] (`In = MilRequest`, `Out = MilPackage`,
//! `Err = MilError`) with no I/O in its signature, so it is the Emit and
//! Package stages of omega's door and runs unchanged under a fuzzer or a
//! kernel-bypass loop. It composes [`crate::wire::Encoder`] for the protobuf,
//! [`crate::weights::BlobWriter`] for `weights/weight.bin` and the census's own
//! classification ([`crate::census`]) for what each node becomes; compose
//! those directly to emit one node ([`emit_node`]) or measure a program
//! ([`measure`]) without building a package.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::future::{Future, ready};
use core::marker::PhantomData;

use proxima_primitives::pipe::Pipe;
use proxima_tensor::{DType, NodeId, Op, Shapes, TensorError};

use crate::census::NoMilForm;
use crate::sized;
use crate::weights::{BlobWriter, SourceDType};
use crate::wire::{Encoder, WireError};

mod contraction;
mod message;
mod node;
mod reduce;
mod views;

pub use node::emit_node;

use message::{Kind, Name, named_type};

const MODEL_SPECIFICATION_VERSION: u32 = 1;
const MODEL_DESCRIPTION: u32 = 2;
const MODEL_PROGRAM: u32 = 502;
const DESCRIPTION_INPUT: u32 = 1;
const DESCRIPTION_OUTPUT: u32 = 10;
const FEATURE_NAME: u32 = 1;
const FEATURE_TYPE: u32 = 3;
const FEATURE_MULTI_ARRAY: u32 = 5;
const ARRAY_SHAPE: u32 = 1;
const ARRAY_DATA_TYPE: u32 = 2;
const ARRAY_FLOAT32: u64 = 65568;
const PROGRAM_VERSION: u32 = 1;
const PROGRAM_VERSION_VALUE: u64 = 1;
const PROGRAM_FUNCTIONS: u32 = 2;
const MAP_KEY: u32 = 1;
const MAP_VALUE: u32 = 2;
const FUNCTION_INPUT: u32 = 1;
const FUNCTION_OPSET: u32 = 2;
const FUNCTION_BLOCKS: u32 = 3;
const BLOCK_OUTPUTS: u32 = 2;
const MAIN_FUNCTION: &[u8] = b"main";

/// The package's relative path of each file, in the order [`MilPackage::files`] lists them.
pub const PACKAGE_PATHS: [&str; 3] = [
    "Manifest.json",
    "Data/com.apple.CoreML/model.mlmodel",
    "Data/com.apple.CoreML/weights/weight.bin",
];

const MANIFEST: &[u8] = br#"{
    "fileFormatVersion": "1.0.0",
    "itemInfoEntries": {
        "6D0A1C3E-5B7F-4A21-9C44-1E0F5A7B2D01": {
            "author": "com.apple.CoreML",
            "description": "CoreML Model Specification",
            "name": "model.mlmodel",
            "path": "com.apple.CoreML/model.mlmodel"
        },
        "6D0A1C3E-5B7F-4A21-9C44-1E0F5A7B2D02": {
            "author": "com.apple.CoreML",
            "description": "CoreML Model Weights",
            "name": "weights",
            "path": "com.apple.CoreML/weights"
        }
    },
    "rootModelIdentifier": "6D0A1C3E-5B7F-4A21-9C44-1E0F5A7B2D01"
}
"#;

/// Why a program could not become a package.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MilError {
    /// A buffer was too small or a weight source was malformed.
    #[error("wire: {0}")]
    Wire(#[from] WireError),
    /// The census refused these nodes: MIL has no form for any of them.
    #[error("{}", describe_offenders(.offenders))]
    NoMilForm { offenders: Vec<NoMilForm> },
    /// The census maps the node but the writer does not yet emit that form.
    #[error("node {node} has a mil form the writer does not emit: {form}")]
    NotEmitted { node: NodeId, form: &'static str },
    /// A node index outside the program.
    #[error("node {node} is outside the program")]
    UnknownNode { node: NodeId },
    /// A node element type MIL's fp16 interior does not carry.
    #[error("node {node} has element type {dtype:?}, which the writer does not carry")]
    UnsupportedDType { node: NodeId, dtype: DType },
    /// A tensor rank above what a view is planned for.
    #[error("node {node} has rank {rank}, above the writer's view capacity")]
    RankTooLarge { node: NodeId, rank: usize },
    /// A function input name MIL does not accept as an identifier.
    #[error("node {node} names its input `{name}`, which is not a mil identifier")]
    InvalidName { node: NodeId, name: String },
    /// A program with no node to be its output.
    #[error("the program has no nodes")]
    EmptyProgram,
    /// Shape inference refused the program.
    #[error("shape inference failed: {0}")]
    Shape(#[from] TensorError),
    /// A lowered program that declares no output.
    #[error("the program declares no output")]
    NoOutput,
    /// The declared output is not the last node, which is where the writer reads it.
    #[error("output node {node} is not the final node of the program")]
    OutputNotFinal { node: NodeId },
    /// A package file the compiler reads is empty.
    #[error("package file {path} is empty")]
    EmptyPackageFile { path: &'static str },
}

fn describe_offenders(offenders: &[NoMilForm]) -> String {
    offenders
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

/// One weight tensor as the source holds it, e.g. an ONNX initializer.
#[derive(Debug, Clone, Copy)]
pub struct WeightView<'a> {
    /// The `Op::Input` name this tensor binds to, e.g. `encoder.layer.0.attention.self.query.weight`.
    pub name: &'a str,
    /// The element type of `bytes`.
    pub dtype: SourceDType,
    /// The little-endian source bytes, read once and converted to fp16 in the blob.
    pub bytes: &'a [u8],
}

/// Everything the writer reads: the lowered program, its resolved shapes and its weights.
#[derive(Debug, Clone, Copy)]
pub struct MilRequest<'a> {
    /// The lowered program; the last node is the output.
    pub program: &'a [Op],
    /// `proxima_tensor::infer` of `program` at the pins.
    pub shapes: &'a Shapes,
    /// Weights by input name; an input with no weight is a function input.
    pub weights: &'a [WeightView<'a>],
    /// The name of the function's output feature, e.g. `last_hidden_state`.
    pub output: &'a str,
}

/// The `.mlpackage` as three in-memory files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MilPackage {
    /// `Manifest.json`.
    pub manifest: Vec<u8>,
    /// `Data/com.apple.CoreML/model.mlmodel`, a `Model` holding `mlProgram`.
    pub model: Vec<u8>,
    /// `Data/com.apple.CoreML/weights/weight.bin`.
    pub weights: Vec<u8>,
}

impl MilPackage {
    /// The files with their package-relative paths, in [`PACKAGE_PATHS`] order.
    #[must_use]
    pub fn files(&self) -> [(&'static str, &[u8]); 3] {
        [
            (PACKAGE_PATHS[0], &self.manifest),
            (PACKAGE_PATHS[1], &self.model),
            (PACKAGE_PATHS[2], &self.weights),
        ]
    }
}

/// Byte lengths of the two buffers a request needs; the manifest is a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measured {
    /// Bytes of the `Model` protobuf.
    pub model: usize,
    /// Bytes of the weight blob.
    pub weights: usize,
}

/// What `emit_node` reads about the program besides the node itself.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    /// The lowered program.
    pub program: &'a [Op],
    /// Its resolved shapes.
    pub shapes: &'a Shapes,
    /// Weights by input name.
    pub weights: &'a [WeightView<'a>],
}

impl<'a> Context<'a> {
    fn weight(&self, name: &str) -> Option<&'a WeightView<'a>> {
        self.weights.iter().find(|weight| weight.name == name)
    }

    fn dtype_of(&self, node: NodeId) -> Result<DType, MilError> {
        self.program
            .get(node.0 as usize)
            .map(Op::dtype)
            .ok_or(MilError::UnknownNode { node })
    }

    fn contraction_of(&self, node: NodeId) -> Option<contraction::Contraction<'a>> {
        contraction::detect(self, node)
    }
}

/// The measuring pre-pass: runs the writer over empty buffers.
///
/// # Errors
/// [`MilError`] for a node the writer cannot express.
pub fn measure(request: &MilRequest<'_>) -> Result<Measured, MilError> {
    let mut encoder = Encoder::new(&mut []);
    let mut blob = BlobWriter::new(&mut []);
    write_model(request, &mut encoder, &mut blob)?;
    Ok(Measured {
        model: encoder.position(),
        weights: blob.length(),
    })
}

fn assemble(request: &MilRequest<'_>) -> Result<MilPackage, MilError> {
    let measured = measure(request)?;
    let mut model = vec![0_u8; measured.model];
    let mut weights = vec![0_u8; measured.weights];
    let mut encoder = Encoder::new(&mut model);
    let mut blob = BlobWriter::new(&mut weights);
    write_model(request, &mut encoder, &mut blob)?;
    encoder.finish()?;
    blob.finish()?;
    Ok(MilPackage {
        manifest: Vec::from(MANIFEST),
        model,
        weights,
    })
}

fn write_model(
    request: &MilRequest<'_>,
    encoder: &mut Encoder<'_>,
    blob: &mut BlobWriter<'_>,
) -> Result<(), MilError> {
    let context = Context {
        program: request.program,
        shapes: request.shapes,
        weights: request.weights,
    };
    let root = NodeId(
        u32::try_from(request.program.len().saturating_sub(1))
            .map_err(|_| MilError::EmptyProgram)?,
    );
    if request.program.is_empty() {
        return Err(MilError::EmptyProgram);
    }
    let output = Name::identifier(request.output).ok_or_else(|| MilError::InvalidName {
        node: root,
        name: request.output.into(),
    })?;
    encoder.varint_field(
        MODEL_SPECIFICATION_VERSION,
        sized::SPECIFICATION_VERSION as u64,
    );
    write_description(&context, root, &output, encoder)?;
    write_program(&context, root, &output, encoder, blob)
}

fn function_inputs<'a>(
    context: &'a Context<'a>,
) -> impl Iterator<Item = (NodeId, Option<&'a str>)> + 'a {
    context
        .program
        .iter()
        .enumerate()
        .filter_map(|(index, op)| match op {
            Op::Input { name, .. }
                if name
                    .as_deref()
                    .and_then(|text| context.weight(text))
                    .is_none() =>
            {
                Some((NodeId(index as u32), name.as_deref()))
            }
            _ => None,
        })
}

fn write_description(
    context: &Context<'_>,
    root: NodeId,
    output: &Name,
    encoder: &mut Encoder<'_>,
) -> Result<(), MilError> {
    let description = encoder.open(MODEL_DESCRIPTION);
    for (node, name) in function_inputs(context) {
        let source = node::function_input_name(node, name)?;
        feature(encoder, DESCRIPTION_INPUT, &source, context.shapes.of(node));
    }
    feature(encoder, DESCRIPTION_OUTPUT, output, context.shapes.of(root));
    encoder.close(description);
    Ok(())
}

fn feature(encoder: &mut Encoder<'_>, field: u32, name: &Name, dims: &[u64]) {
    let description = encoder.open(field);
    encoder.bytes_field(FEATURE_NAME, name.as_bytes());
    let kind = encoder.open(FEATURE_TYPE);
    let array = encoder.open(FEATURE_MULTI_ARRAY);
    encoder.packed_extents(ARRAY_SHAPE, dims);
    encoder.varint_field(ARRAY_DATA_TYPE, ARRAY_FLOAT32);
    encoder.close(array);
    encoder.close(kind);
    encoder.close(description);
}

fn write_program(
    context: &Context<'_>,
    root: NodeId,
    output: &Name,
    encoder: &mut Encoder<'_>,
    blob: &mut BlobWriter<'_>,
) -> Result<(), MilError> {
    let program = encoder.open(MODEL_PROGRAM);
    encoder.varint_field(PROGRAM_VERSION, PROGRAM_VERSION_VALUE);
    let entry = encoder.open(PROGRAM_FUNCTIONS);
    encoder.bytes_field(MAP_KEY, MAIN_FUNCTION);
    let function = encoder.open(MAP_VALUE);
    for (node, name) in function_inputs(context) {
        let source = node::function_input_name(node, name)?;
        named_type(
            encoder,
            FUNCTION_INPUT,
            source.as_bytes(),
            Kind::Float32,
            context.shapes.of(node),
        );
    }
    encoder.bytes_field(FUNCTION_OPSET, sized::OPSET.as_bytes());
    write_block(context, root, output, encoder, blob)?;
    encoder.close(function);
    encoder.close(entry);
    encoder.close(program);
    Ok(())
}

fn write_block(
    context: &Context<'_>,
    root: NodeId,
    output: &Name,
    encoder: &mut Encoder<'_>,
    blob: &mut BlobWriter<'_>,
) -> Result<(), MilError> {
    let specialization = encoder.open(FUNCTION_BLOCKS);
    encoder.bytes_field(MAP_KEY, sized::OPSET.as_bytes());
    let block = encoder.open(MAP_VALUE);
    encoder.bytes_field(BLOCK_OUTPUTS, output.as_bytes());
    for index in 0..context.program.len() {
        emit_node(context, NodeId(index as u32), encoder, blob)?;
    }
    node::cast(
        encoder,
        &Name::node(root.0),
        output,
        Kind::Float32,
        context.shapes.of(root),
    );
    encoder.close(block);
    encoder.close(specialization);
    Ok(())
}

/// The sans-IO writer: [`MilRequest`] in, [`MilPackage`] out.
///
/// Built with [`MilWriter::new`] and driven with [`Pipe::call`]; omega's door
/// plugs it in as the Emit and Package stages. [`MilWriter::write`] is the
/// same work without a future for callers that are not in a pipe.
#[derive(Debug, Clone, Copy, Default)]
pub struct MilWriter<'request> {
    request: PhantomData<&'request ()>,
}

impl<'request> MilWriter<'request> {
    /// A writer; it holds no state, so one serves any number of requests.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            request: PhantomData,
        }
    }

    /// Writes the package for `request`.
    ///
    /// # Errors
    /// [`MilError`] for a program the writer cannot express.
    pub fn write(&self, request: &MilRequest<'request>) -> Result<MilPackage, MilError> {
        assemble(request)
    }
}

impl<'request> Pipe for MilWriter<'request> {
    type In = MilRequest<'request>;
    type Out = MilPackage;
    type Err = MilError;

    fn call(&self, input: Self::In) -> impl Future<Output = Result<Self::Out, Self::Err>> {
        ready(self.write(&input))
    }
}
