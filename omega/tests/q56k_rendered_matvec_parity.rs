//! The Q6_K decode matvec is rendered from `BlockLayout::Q6_K` (`msl/sub_block_assembled_matvec.rs`);
//! Q5_K keeps ggml's port (`push_q5k_ggml_port_body`) because its rendering measured slower. Output on
//! real tensors must equal, bit for bit, what the hand ports of ggml's
//! `kernel_mul_mv_q5_K_f32_impl<2,2,32>` / `kernel_mul_mv_q6_K_f32_impl<1,2,32>`
//! (`push_q6k_ggml_port_body` removed) produced at main 873ebbc3 (the rendered Q5_K matched them at
//! e436879f, 0 mismatches on both Q5_K tensors, before the port was selected again). The
//! fixtures hold that port's raw f32 bits; `fixtures/q56k_matvec_PROVENANCE.txt` records how they were
//! captured. Setting `PROXNUM67_CAPTURE_DIR` writes the bits instead of comparing (the capture run).
#![cfg(all(feature = "metal", target_os = "macos"))]
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
use proxima_tensor::{Extent, IndexMap, Keep, Op, QuantizedBlock, Reduce, ReduceInit, ScalarOp, append, map};

const FLINT_PATH: &str = "/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf";
const OLLAMA_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129";
const ACTIVATION_SEED: u64 = 2026;
const QK_K: usize = 256;

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

fn block_bytes(codec: Codec) -> usize {
    match codec {
        Codec::Q5K => proxima_gguf::quant::q5_k::BLOCK_BYTES,
        Codec::Q6K => proxima_gguf::quant::q6_k::BLOCK_BYTES,
        other => panic!("{other:?} is not a codec this test covers"),
    }
}

fn tensor_rows(path: &str, name: &str, ggml_type: GgmlType, codec: Codec, rows: usize) -> (Vec<u8>, usize) {
    let path = proxima_gguf::checkpoint_resolver::require_checkpoint("PROXIMA_Q56K_PARITY_GGUF", path, 0);
    let mut file = File::open(&path).expect("the checkpoint opens");
    let file_len = file.metadata().unwrap().len();
    let parsed = header(&mut file, file_len);
    let tensor = parsed.tensors.iter().find(|tensor| tensor.name == name).expect("the tensor is in the directory");
    assert_eq!(tensor.ggml_type, ggml_type, "{name} has the expected quantization");
    let reduction = tensor.dims[0] as usize;
    let range = parsed.tensor_data_range(tensor, file_len).unwrap();
    let mut bytes = vec![0u8; rows * (reduction / QK_K) * block_bytes(codec)];
    file.seek(SeekFrom::Start(range.start)).unwrap();
    file.read_exact(&mut bytes).unwrap();
    (bytes, reduction)
}

fn matvec_program(rows: u32, reduction: u32) -> (Vec<Op>, Vec<proxima_tensor::NodeId>) {
    let mut program = Vec::new();
    let weight = append(&mut program, Op::Input { dtype: DType::UINT8, shape: vec![Extent::Static(rows), Extent::Static(reduction)], name: None });
    let activation = append(&mut program, Op::Input { dtype: DType::FLOAT32, shape: vec![Extent::Static(reduction), Extent::Static(1)], name: None });
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![(weight, IndexMap::Affine(map::projection(3, &[0, 2]))), (activation, IndexMap::Affine(map::projection(3, &[2, 1])))],
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
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    (program, vec![sum])
}

fn matvec_bits(path: &str, name: &str, ggml_type: GgmlType, codec: Codec, rows: usize) -> Vec<u32> {
    let (weights, reduction) = tensor_rows(path, name, ggml_type, codec, rows);
    let mut lcg = Lcg(ACTIVATION_SEED);
    let activation: Vec<f32> = (0..reduction).map(|_| lcg.next_unit() * 4.0 - 2.0).collect();
    let blocks = [QuantizedBlock::Packed { codec, bytes: &weights }, QuantizedBlock::Float32(&activation)];
    let (program, roots) = matvec_program(rows as u32, reduction as u32);
    let output = omega::execute(&program, &[], &blocks, &roots, NumericPolicy::default()).expect("the matvec executes");
    output.root().iter().map(|value| value.to_bits()).collect()
}

fn fixture_bits(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|word| u32::from_le_bytes(*word)).collect()
}

fn assert_bit_identical(path: &str, name: &str, ggml_type: GgmlType, codec: Codec, rows: usize, fixture_name: &str, fixture: &[u8]) {
    let actual = matvec_bits(path, name, ggml_type, codec, rows);
    if let Some(directory) = std::env::var_os("PROXNUM67_CAPTURE_DIR") {
        let bytes: Vec<u8> = actual.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        std::fs::write(std::path::Path::new(&directory).join(fixture_name), bytes).expect("the capture writes");
        return;
    }
    let expected = fixture_bits(fixture);
    assert_eq!(expected.len(), rows, "the fixture holds one f32 per row");
    assert_eq!(actual.len(), expected.len(), "{name}: output length");
    let mismatches = actual.iter().zip(&expected).filter(|(got, want)| got != want).count();
    assert_eq!(mismatches, 0, "{name}: {mismatches} of {rows} f32 outputs differ from the ggml port's bits");
}

macro_rules! fixture {
    ($file:literal) => {
        ($file, include_bytes!(concat!("fixtures/", $file)).as_slice())
    };
}

#[test]
fn ollama_attn_v_q5k_matvec_matches_the_ggml_port_bit_for_bit() {
    let (name, bytes) = fixture!("q5k_matvec_ollama_blk0_attn_v_rows512.f32le");
    assert_bit_identical(OLLAMA_PATH, "blk.0.attn_v.weight", GgmlType::Q5_K, Codec::Q5K, 512, name, bytes);
}

#[test]
fn flint_ffn_gate_exps_q5k_matvec_matches_the_ggml_port_bit_for_bit() {
    let (name, bytes) = fixture!("q5k_matvec_flint_blk10_ffn_gate_exps_rows512.f32le");
    assert_bit_identical(FLINT_PATH, "blk.10.ffn_gate_exps.weight", GgmlType::Q5_K, Codec::Q5K, 512, name, bytes);
}

#[test]
fn ollama_token_embd_q6k_matvec_matches_the_ggml_port_bit_for_bit() {
    let (name, bytes) = fixture!("q6k_matvec_ollama_token_embd_rows2048.f32le");
    assert_bit_identical(OLLAMA_PATH, "token_embd.weight", GgmlType::Q6_K, Codec::Q6K, 2048, name, bytes);
}

#[test]
fn flint_ffn_down_exps_q6k_matvec_matches_the_ggml_port_bit_for_bit() {
    let (name, bytes) = fixture!("q6k_matvec_flint_blk8_ffn_down_exps_rows2048.f32le");
    assert_bit_identical(FLINT_PATH, "blk.8.ffn_down_exps.weight", GgmlType::Q6_K, Codec::Q6K, 2048, name, bytes);
}
