//! The Q6_K token-group matvec is rendered from `BlockLayout::Q6_K`
//! (`assembled_multi_row_matvec_msl`, `msl/sub_block_assembled_matvec.rs`). Output on real tensors must
//! equal, bit for bit, what the hand body `push_packed_row_multi_row_q6k_body` produced at main
//! 97eef4d7. The fixtures hold that body's raw f32 bits; `fixtures/q6k_multi_row_PROVENANCE.txt` records
//! how they were captured. Setting `PROXNUM69_CAPTURE_DIR` writes the bits instead of comparing.
#![cfg(all(feature = "metal", not(feature = "metal-q4k-split-k"), target_os = "macos"))]
// workspace lints deny unwrap/expect (Cargo.toml:93-94) and an integration test binary is not cfg(test), so a failed fixture read must be an allowed panic.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use proxima_gguf::parser::{GgufEvent, GgufParser};
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::types::GgmlType;
use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{AxisTerm, Extent, IndexMap, Keep, NodeId, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, affine, append, projection};

const FLINT_PATH: &str = "/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf";
const OLLAMA_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129";
const ACTIVATION_SEED: u64 = 2026;
const ROWS: usize = 2048;
const QK_K: usize = 256;
/// `crate::sized::PACKED_ROW_ACTIVATION_GROUP` (omega/omega-runtime.toml:268): the token group a dispatch folds at once.
const TOKEN_CAP: usize = 8;

fn header(file: &mut File, file_len: u64) -> ParsedGguf {
    let mut prefix_len = 1usize << 22;
    loop {
        let mut buffer = vec![0u8; prefix_len];
        file.seek(SeekFrom::Start(0)).unwrap();
        let read = file.read(&mut buffer).unwrap();
        buffer.truncate(read);
        if let Ok((parser, events)) = GgufParser::new().push(&buffer) {
            let (mut version, mut metadata, mut tensors, mut completion) = (None, Vec::new(), Vec::new(), None);
            for event in events {
                match event {
                    GgufEvent::Header { version: found, .. } => version = Some(found),
                    GgufEvent::Metadata { key, value } => metadata.push((key, value)),
                    GgufEvent::Tensor(tensor) => tensors.push(tensor),
                    GgufEvent::Complete { data_offset, alignment } => completion = Some((data_offset, alignment)),
                }
            }
            if let (Some(version), Some((data_offset, alignment))) = (version, completion) {
                parser.finish().unwrap();
                return ParsedGguf { version, tensor_count: tensors.len() as u64, kv_count: metadata.len() as u64, metadata, tensors, data_offset, alignment };
            }
        }
        assert!((prefix_len as u64) < file_len, "no gguf header in the first {prefix_len} bytes");
        prefix_len *= 2;
    }
}

fn tensor_rows(path: &str, name: &str) -> (Vec<u8>, usize) {
    let path = proxima_gguf::checkpoint_resolver::require_checkpoint("PROXIMA_Q6K_MULTI_ROW_PARITY_GGUF", path, 0);
    let mut file = File::open(&path).expect("the checkpoint opens");
    let file_len = file.metadata().unwrap().len();
    let parsed = header(&mut file, file_len);
    let tensor = parsed.tensors.iter().find(|tensor| tensor.name == name).expect("the tensor is in the directory");
    assert_eq!(tensor.ggml_type, GgmlType::Q6_K, "{name} is Q6_K");
    let reduction = tensor.dims[0] as usize;
    let range = parsed.tensor_data_range(tensor, file_len).unwrap();
    let mut bytes = vec![0u8; ROWS * (reduction / QK_K) * proxima_gguf::quant::q6_k::BLOCK_BYTES];
    file.seek(SeekFrom::Start(range.start)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    (bytes, reduction)
}

fn matmul_program(tokens: u32, reduction: u32, rows: u32) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let weight = append(&mut program, Op::Input { dtype: DType::UINT8, shape: vec![Extent::Static(rows), Extent::Static(reduction)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(tokens), Extent::Static(reduction)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (weight, IndexMap::Affine(projection(3, &[1, 2]))),
                (activation, IndexMap::Affine(affine(3, &[(&[AxisTerm::scaled(0, 1)], 0), (&[AxisTerm::scaled(2, 1)], 0)]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, sum)
}

fn matmul_bits(path: &str, name: &str, tokens: usize) -> Vec<u32> {
    let (weights, reduction) = tensor_rows(path, name);
    let mut lcg = Lcg(ACTIVATION_SEED);
    let activation: Vec<f32> = (0..tokens * reduction).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();
    let blocks = [QuantizedBlock::Packed { codec: Codec::Q6K, bytes: &weights }, QuantizedBlock::Float32(&activation)];
    let (program, sum) = matmul_program(tokens as u32, reduction as u32, ROWS as u32);
    let plan = omega::plan(&program, &[], &blocks, &[sum], NumericPolicy::default()).expect("metal plans the Q6_K matmul");
    let output = omega::execute_plan(&plan, &blocks).expect("metal runs the matmul on a real device");
    output.root().iter().map(|value| value.to_bits()).collect()
}

fn fixture_bits(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|word| u32::from_le_bytes(*word)).collect()
}

fn count_mismatches(actual: &[u32], expected: &[u32]) -> usize {
    actual.iter().zip(expected).filter(|(got, want)| got != want).count()
}

fn assert_bit_identical(path: &str, name: &str, tokens: usize, fixture_name: &str, fixture: &[u8]) {
    let actual = matmul_bits(path, name, tokens);
    if let Some(directory) = std::env::var_os("PROXNUM69_CAPTURE_DIR") {
        let bytes: Vec<u8> = actual.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        std::fs::write(std::path::Path::new(&directory).join(fixture_name), bytes).expect("the capture writes");
        return;
    }
    let expected = fixture_bits(fixture);
    assert_eq!(expected.len(), tokens * ROWS, "the fixture holds one f32 per (token, row)");
    assert_eq!(actual.len(), expected.len(), "{name}: output length");
    let mismatches = count_mismatches(&actual, &expected);
    assert_eq!(mismatches, 0, "{name} tokens={tokens}: {mismatches} of {} f32 outputs differ from the hand body's bits", expected.len());
}

macro_rules! fixture {
    ($file:literal) => {
        ($file, include_bytes!(concat!("fixtures/", $file)).as_slice())
    };
}

#[test]
fn ollama_token_embd_two_tokens_match_the_hand_body_bit_for_bit() {
    let (name, bytes) = fixture!("q6k_multi_row_ollama_token_embd_rows2048_tokens2.f32le");
    assert_bit_identical(OLLAMA_PATH, "token_embd.weight", 2, name, bytes);
}

#[test]
fn ollama_token_embd_cap_tokens_match_the_hand_body_bit_for_bit() {
    let (name, bytes) = fixture!("q6k_multi_row_ollama_token_embd_rows2048_tokens8.f32le");
    assert_bit_identical(OLLAMA_PATH, "token_embd.weight", TOKEN_CAP, name, bytes);
}

#[test]
fn flint_ffn_down_exps_two_tokens_match_the_hand_body_bit_for_bit() {
    let (name, bytes) = fixture!("q6k_multi_row_flint_blk8_ffn_down_exps_rows2048_tokens2.f32le");
    assert_bit_identical(FLINT_PATH, "blk.8.ffn_down_exps.weight", 2, name, bytes);
}

#[test]
fn flint_ffn_down_exps_cap_tokens_match_the_hand_body_bit_for_bit() {
    let (name, bytes) = fixture!("q6k_multi_row_flint_blk8_ffn_down_exps_rows2048_tokens8.f32le");
    assert_bit_identical(FLINT_PATH, "blk.8.ffn_down_exps.weight", TOKEN_CAP, name, bytes);
}

#[test]
fn one_flipped_output_bit_is_counted_as_one_mismatch() {
    let (_, bytes) = fixture!("q6k_multi_row_flint_blk8_ffn_down_exps_rows2048_tokens2.f32le");
    let expected = fixture_bits(bytes);
    let mut perturbed = expected.clone();
    perturbed[1234] ^= 1;
    assert_eq!(count_mismatches(&expected, &expected), 0, "an identical vector has no mismatches");
    assert_eq!(count_mismatches(&perturbed, &expected), 1, "the comparison must see a single low-bit flip");
}
