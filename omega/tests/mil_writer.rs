//! `MilWriter` turns the lowered two-layer MLP (`Gemm -> Relu -> Gemm ->
//! Softmax`, the program `onnx_metal_parity.rs` checks against hand-computed
//! softmax values) into an `.mlpackage`. The package is judged by tools that
//! share no code with the writer: `protoc` decodes the hand-encoded model
//! with the vendored Apple protos, and the operator sequence below was derived
//! from the lowered program before the writer existed.

#![cfg(feature = "mil-writer")]
// fixtures are hand-built to succeed; an expect failure here IS the test failing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

// the shared door fixture serves several binaries and this one reads only the MLP bytes.
#[allow(dead_code)]
mod door_support;
// the helper module serves two binaries and each uses part of it.
#[allow(dead_code)]
mod mil_support;

use door_support::two_layer_mlp_bytes;
use half::f16;
use mil_support::{operator_sequence, protoc_decode};
use proxima_mil::sized::{OPSET, SPECIFICATION_VERSION};
use proxima_mil::weights::SourceDType;
use proxima_mil::writer::{
    MilError, MilPackage, MilRequest, MilWriter, PACKAGE_PATHS, WeightView, measure,
};
use proxima_onnx::lower::lower_graph;
use proxima_onnx::pipe::parse_complete;
use proxima_primitives::pipe::Pipe;
use proxima_tensor::{Op, Shapes, infer};
use sha2::{Digest, Sha256};

const OUTPUT: &str = "y";

const MLP_OPERATORS: [&str; 27] = [
    "const",
    "const",
    "const",
    "const",
    "cast",
    "matmul",
    "const",
    "mul",
    "const",
    "mul",
    "add",
    "const",
    "maximum",
    "matmul",
    "const",
    "mul",
    "const",
    "mul",
    "add",
    "reduce_max",
    "expand_dims",
    "sub",
    "exp",
    "reduce_sum",
    "expand_dims",
    "real_div",
    "cast",
];

struct Mlp {
    program: Vec<Op>,
    shapes: Shapes,
    initializers: Vec<(String, Vec<f32>)>,
}

fn mlp() -> Mlp {
    let bytes = two_layer_mlp_bytes();
    let model = parse_complete(&bytes).expect("the mlp bytes parse");
    let graph = model.graph.as_ref().expect("the mlp has a graph");
    let lowered = lower_graph(graph).expect("the mlp lowers");
    let shapes = infer(&lowered.program, &[]).expect("the mlp program has shapes");
    Mlp {
        program: lowered.program,
        shapes,
        initializers: lowered.initializers,
    }
}

fn little_endian(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn encoded_weights(mlp: &Mlp) -> Vec<(String, Vec<u8>)> {
    mlp.initializers
        .iter()
        .map(|(name, values)| (name.clone(), little_endian(values)))
        .collect()
}

fn views(encoded: &[(String, Vec<u8>)]) -> Vec<WeightView<'_>> {
    encoded
        .iter()
        .map(|(name, bytes)| WeightView {
            name,
            dtype: SourceDType::Float32,
            bytes,
        })
        .collect()
}

fn write_package(mlp: &Mlp, encoded: &[(String, Vec<u8>)]) -> MilPackage {
    let weights = views(encoded);
    let request = MilRequest {
        program: &mlp.program,
        shapes: &mlp.shapes,
        weights: &weights,
        output: OUTPUT,
    };
    MilWriter::new()
        .write(&request)
        .expect("the writer emits the mlp")
}

fn decoded(package: &MilPackage) -> String {
    protoc_decode(&package.model).expect("protoc decodes the emitted model")
}

fn package_sha256(package: &MilPackage) -> String {
    let mut files = package.files().to_vec();
    files.sort_by_key(|(path, _)| *path);
    let mut digest = Sha256::new();
    for (path, bytes) in files {
        digest.update(path.as_bytes());
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
}

fn blob_offsets(text: &str) -> Vec<usize> {
    text.lines()
        .filter_map(|line| line.trim().strip_prefix("offset: "))
        .map(|number| number.parse().expect("an offset is a number"))
        .collect()
}

fn le_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn le_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"))
}

#[test]
fn mil_writer_package_file_list_is_manifest_model_and_weight_blob() {
    let mlp = mlp();

    let package = write_package(&mlp, &encoded_weights(&mlp));

    let paths: Vec<&str> = package.files().iter().map(|(path, _)| *path).collect();
    assert_eq!(paths, PACKAGE_PATHS);
    assert_eq!(
        paths,
        [
            "Manifest.json",
            "Data/com.apple.CoreML/model.mlmodel",
            "Data/com.apple.CoreML/weights/weight.bin"
        ]
    );
    assert!(package.files().iter().all(|(_, bytes)| !bytes.is_empty()));
    let manifest = String::from_utf8(package.manifest.clone()).expect("the manifest is text");
    assert!(manifest.contains("\"rootModelIdentifier\""), "{manifest}");
    assert!(manifest.contains("com.apple.CoreML/weights"), "{manifest}");
}

#[test]
fn mil_writer_emission_is_deterministic_byte_for_byte() {
    let mlp = mlp();
    let encoded = encoded_weights(&mlp);

    let first = write_package(&mlp, &encoded);
    let second = write_package(&mlp, &encoded);

    assert_eq!(first, second);
    assert_eq!(package_sha256(&first), package_sha256(&second));
}

#[test]
fn mil_writer_model_decodes_with_the_protoc_oracle() {
    let mlp = mlp();
    let package = write_package(&mlp, &encoded_weights(&mlp));

    let text = decoded(&package);

    assert!(text.contains("mlProgram {"), "{text}");
    assert!(text.contains("key: \"main\""), "{text}");
    assert!(text.contains("name: \"x\""), "{text}");
    assert_eq!(operator_sequence(&text).len(), MLP_OPERATORS.len());
}

#[test]
fn mil_writer_protoc_oracle_degenerate_truncated_model_fails_to_decode() {
    let mlp = mlp();
    let package = write_package(&mlp, &encoded_weights(&mlp));
    let truncated = &package.model[..package.model.len() - 3];

    let verdict = protoc_decode(truncated);

    assert!(
        verdict.is_err(),
        "protoc accepted a model cut 3 bytes short"
    );
}

#[test]
fn mil_writer_specification_version_and_opset_equal_the_sized_keys() {
    let mlp = mlp();
    let package = write_package(&mlp, &encoded_weights(&mlp));

    let text = decoded(&package);

    assert!(
        text.contains(&format!("specificationVersion: {SPECIFICATION_VERSION}")),
        "{text}"
    );
    assert!(text.contains(&format!("opset: \"{OPSET}\"")), "{text}");
    assert!(text.contains(&format!("key: \"{OPSET}\"")), "{text}");
}

#[test]
fn mil_writer_interior_dtype_is_fp16_and_io_dtype_is_fp32() {
    let mlp = mlp();
    let package = write_package(&mlp, &encoded_weights(&mlp));

    let text = decoded(&package);

    let float32 = text.matches("dataType: FLOAT32").count();
    let float16 = text.matches("dataType: FLOAT16").count();
    assert_eq!(
        float32, 4,
        "feature input, feature output, function input and the output cast are the only fp32 types"
    );
    assert_eq!(
        float16, 35,
        "4 weights and 5 constants types twice, plus 21 operation outputs"
    );
    assert!(text.contains("values: \"fp16\""), "{text}");
    assert!(text.contains("values: \"fp32\""), "{text}");
}

#[test]
fn mil_writer_weights_blob_holds_each_initializer_once_at_recorded_offsets() {
    let mlp = mlp();
    let package = write_package(&mlp, &encoded_weights(&mlp));
    let offsets = blob_offsets(&decoded(&package));
    let blob = &package.weights;

    assert_eq!(le_u32(blob, 0), 4, "the header counts four entries");
    assert_eq!(le_u32(blob, 4), 2, "the blob version");
    assert_eq!(offsets.len(), mlp.initializers.len());
    for (offset, (name, values)) in offsets.iter().zip(&mlp.initializers) {
        assert_eq!(offset % 64, 0, "{name} descriptor is 64-byte aligned");
        assert_eq!(le_u32(blob, *offset), 0xDEAD_BEEF, "{name} sentinel");
        assert_eq!(le_u32(blob, offset + 4), 1, "{name} element code is fp16");
        assert_eq!(
            le_u64(blob, offset + 8),
            (values.len() * 2) as u64,
            "{name} size"
        );
        let data = le_u64(blob, offset + 16) as usize;
        assert_eq!(data, offset + 64, "{name} data follows its descriptor");
        let expected: Vec<u8> = values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect();
        assert_eq!(
            &blob[data..data + expected.len()],
            expected.as_slice(),
            "{name}"
        );
    }
}

#[test]
fn mil_writer_altered_weight_byte_changes_the_package_sha256() {
    let mlp = mlp();
    let encoded = encoded_weights(&mlp);
    let mut altered = encoded.clone();
    altered[0].1[2] ^= 0x40;

    let original = package_sha256(&write_package(&mlp, &encoded));
    let changed = package_sha256(&write_package(&mlp, &altered));

    assert_ne!(original, changed);
}

fn assert_is_pipe<'request, P>()
where
    P: Pipe<In = MilRequest<'request>, Out = MilPackage, Err = MilError>,
{
}

#[proxima::test]
async fn mil_writer_writer_is_a_pipe_with_in_mil_request_out_mil_package_err_mil_error() {
    let mlp = mlp();
    let encoded = encoded_weights(&mlp);
    let weights = views(&encoded);
    assert_is_pipe::<MilWriter<'_>>();
    let request = MilRequest {
        program: &mlp.program,
        shapes: &mlp.shapes,
        weights: &weights,
        output: OUTPUT,
    };
    let measured = measure(&request).expect("the mlp measures");

    let piped = MilWriter::new()
        .call(request)
        .await
        .expect("the pipe emits the mlp");

    assert_eq!(piped, write_package(&mlp, &encoded));
    assert_eq!(piped.model.len(), measured.model);
    assert_eq!(piped.weights.len(), measured.weights);
}

#[test]
fn mil_writer_two_layer_mlp_emits_the_hand_derived_operator_sequence() {
    let mlp = mlp();
    let package = write_package(&mlp, &encoded_weights(&mlp));

    let sequence = operator_sequence(&decoded(&package));

    assert_eq!(sequence, MLP_OPERATORS);
}
