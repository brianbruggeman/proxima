//! Materializing one named GGUF tensor as an owned `f32` buffer, ready to
//! sit in `proxima_tensor::cpu::evaluate_named`'s `named: &[(&str, &[f32])]`
//! bind-by-name slice.
//!
//! Nothing before this crate joined the two: [`proxima_gguf`] parses a
//! tensor directory keyed by name and hands back raw bytes, and
//! [`proxima_tensor::cpu::evaluate_named`] binds an [`proxima_tensor::Op::Input`]
//! by the same kind of name -- but neither crate depends on the other, so
//! nothing turned "here are a GGUF file's bytes and a tensor name" into
//! "here is the `&[f32]` that name's `Op::Input` needs". [`gguf_tensor_as_f32`]
//! is that one step: look the name up, slice its bytes out of the file
//! buffer, and decode them (a straight copy for `F32`, dequantization for
//! a supported block-quantized type).
//!
//! Sans-IO like the rest of this crate: this module never opens a file.
//! The caller parses via [`proxima_gguf::pipe::parse_complete`] (or
//! [`proxima_gguf::edge::read_file`], std-only) and hands this module the
//! resulting [`ParsedGguf`] plus the byte buffer it was parsed from.

use alloc::vec;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::quant::{q4_k, q5_k, q6_k, q8_0};
use proxima_gguf::tensor::TensorInfo;
use proxima_gguf::types::GgmlType;

use crate::error::InteropError;

/// Looks `name` up in `parsed`'s tensor directory, slices its bytes out of
/// `file_bytes`, and decodes them to an owned `f32` buffer -- copied
/// as-is for `F32`, dequantized for `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0` (the four
/// codecs [`proxima_gguf::quant`] already ships). Every other `GgmlType`
/// (`F16`/`Bf16`/integer/any other quant family) has no decoder here yet
/// and errors rather than misreading bytes.
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if `name` isn't in `parsed.tensors`;
/// [`InteropError::Gguf`] if the tensor's declared byte range doesn't fit
/// `file_bytes`; [`InteropError::Quant`] if a block-quantized tensor's
/// byte length doesn't match its own codec's block-size contract;
/// [`InteropError::UnrepresentableGgmlType`] for an undecoded `GgmlType`.
pub fn gguf_tensor_as_f32(parsed: &ParsedGguf, file_bytes: &[u8], name: &str) -> Result<Vec<f32>, InteropError> {
    let tensor = find_tensor(parsed, name)?;
    let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
    let data = &file_bytes[range.start as usize..range.end as usize];
    let element_count = tensor.element_count() as usize;

    match tensor.ggml_type {
        GgmlType::F32 => Ok(reinterpret_f32(data)),
        GgmlType::Q4_K => dequantize(data, element_count, q4_k::dequantize),
        GgmlType::Q5_K => dequantize(data, element_count, q5_k::dequantize),
        GgmlType::Q6_K => dequantize(data, element_count, q6_k::dequantize),
        GgmlType::Q8_0 => dequantize(data, element_count, q8_0::dequantize),
        other => Err(InteropError::UnrepresentableGgmlType {
            tensor: tensor.name.clone(),
            ggml_type: other,
        }),
    }
}

fn find_tensor<'a>(parsed: &'a ParsedGguf, name: &str) -> Result<&'a TensorInfo, InteropError> {
    parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| InteropError::UnknownTensor { name: name.into() })
}

fn reinterpret_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(bytes)
        })
        .collect()
}

fn dequantize(
    data: &[u8],
    element_count: usize,
    decode: fn(&[u8], &mut [f32]) -> Result<(), proxima_gguf::quant::QuantError>,
) -> Result<Vec<f32>, InteropError> {
    let mut output = vec![0.0f32; element_count];
    decode(data, &mut output)?;
    Ok(output)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;

    use proxima_gguf::{GgmlType as WireType, GgufModel, TensorPayload, write_complete};

    use super::*;
    use crate::error::InteropError;

    fn dims(values: &[u64]) -> arrayvec::ArrayVec<u64, { proxima_gguf::tensor::MAX_DIMS }> {
        values.iter().copied().collect()
    }

    /// A round-tripped `F32` tensor comes back byte-identical as `f32`,
    /// not merely "close" -- reinterpretation, not conversion.
    #[test]
    fn f32_tensor_reinterprets_bytes_exactly() {
        let values = [1.0f32, -2.5, 3.25, 0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "weights".to_string(),
                dims: dims(&[4]),
                ggml_type: WireType::F32,
                data: bytes.as_slice(),
            }],
        };
        let file_bytes = write_complete(&model).expect("writes gguf");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses gguf");

        let decoded = gguf_tensor_as_f32(&parsed, &file_bytes, "weights").expect("bind f32 tensor by name");
        assert_eq!(decoded, values);
    }

    /// A `Q4_K` tensor decodes through the crate's own dequantizer and
    /// matches an independent hand computation of the block's `x =
    /// d*sc*q - dmin*m` formula for the one nonzero probe element this
    /// fixture packs.
    #[test]
    fn q4_k_tensor_dequantizes_through_bind_by_name() {
        let mut block = [0u8; q4_k::BLOCK_BYTES];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes()); // d
        block[2..4].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes()); // dmin
        // sub_block 0: scale code 3, min code 61 (sub_block < 4 packing).
        block[4] = 3;
        block[8] = 61;
        block[16] = 0x07; // qs[0] low nibble = 7 -> element 0 of sub_block 0

        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "blk.0.ffn_gate.weight".to_string(),
                dims: dims(&[q4_k::QK_K as u64]),
                ggml_type: WireType::Q4_K,
                data: &block,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes quantized gguf");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses quantized gguf");

        let decoded =
            gguf_tensor_as_f32(&parsed, &file_bytes, "blk.0.ffn_gate.weight").expect("bind q4_k tensor by name");
        assert_eq!(decoded.len(), q4_k::QK_K);
        // element 0: d*sc*q - dmin*m = 1.0*3.0*7.0 - 0.5*61.0 = -9.5
        assert!((decoded[0] - (-9.5)).abs() < 1e-6, "decoded[0]={}", decoded[0]);
        // every other element in sub_block 0 shares scale/min with q=0.
        assert!((decoded[1] - (-30.5)).abs() < 1e-6, "decoded[1]={}", decoded[1]);
    }

    #[test]
    fn unknown_name_errors_instead_of_panicking() {
        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: Vec::new(),
        };
        let file_bytes = write_complete(&model).expect("writes empty gguf");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses empty gguf");

        let outcome = gguf_tensor_as_f32(&parsed, &file_bytes, "missing");
        assert!(matches!(outcome, Err(InteropError::UnknownTensor { .. })));
    }

    #[test]
    fn unrepresentable_ggml_type_errors_instead_of_misreading_bytes() {
        let data = [0u8; 18]; // one Q4_0 block
        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![TensorPayload {
                name: "blk.0.attn_q.weight".to_string(),
                dims: dims(&[32]),
                ggml_type: WireType::Q4_0,
                data: &data,
            }],
        };
        let file_bytes = write_complete(&model).expect("writes q4_0 gguf");
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses q4_0 gguf");

        let outcome = gguf_tensor_as_f32(&parsed, &file_bytes, "blk.0.attn_q.weight");
        assert!(matches!(outcome, Err(InteropError::UnrepresentableGgmlType { .. })));
    }
}

// -- Real-data proof: bind a real Q4_K weight row out of a host-local
// checkpoint by name, feed it through `proxima_tensor::cpu::evaluate_named`
// against a known activation, and check the interpreter's result against a
// dequantize-then-multiply computed independently of both `bind` and
// `cpu`. Opportunistic like `proxima_gguf::restack::tests::real_mixtral_file`:
// `#[ignore]`d and skips cleanly when the host-local model cache is absent.
#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod real_openchat_file {
    use alloc::vec::Vec;

    use proxima_gguf::GgmlType;
    use proxima_gguf::quant::q4_k;
    use proxima_tensor::DType;
    use proxima_tensor::cpu::evaluate_named;
    use proxima_tensor::map::{self, IndexMap};
    use proxima_tensor::op::{Extent, Keep, Op, Reduce, ReduceInit, ScalarOp, append};

    use super::gguf_tensor_as_f32;

    const FIXTURE_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

    /// Builds `weight . activation` (elementwise multiply, then reduce to
    /// a scalar) over one super-block's worth (256 elements) of a real
    /// `Q4_K` tensor, evaluated two ways: through
    /// `proxima_model_interop::gguf_tensor_as_f32` bound by name into
    /// `evaluate_named`, and by hand-computing the same dot product
    /// straight from `q4_k::dequantize_block` on the tensor's raw bytes,
    /// bypassing this crate's `bind` module entirely. The interpreter's
    /// output must agree with the independent computation.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn binds_one_real_q4_k_block_and_matmuls_against_a_known_activation() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local openchat gguf fixture at {FIXTURE_PATH}");
            return;
        }

        let (parsed, file_bytes) = proxima_gguf::edge::read_file(path).expect("read host-local openchat gguf fixture");

        // a mid-network `ffn_gate` row, not `token_embd`'s row 0 -- that
        // row is the padding-token embedding and its first super-block
        // decodes to all zeros, which would make this test pass on a
        // degenerate zero-vs-zero comparison instead of a real one.
        let tensor_name = parsed
            .tensors
            .iter()
            .find(|tensor| {
                tensor.ggml_type == GgmlType::Q4_K
                    && tensor.element_count() as usize >= q4_k::QK_K
                    && tensor.name.contains("ffn_gate")
            })
            .map(|tensor| tensor.name.clone())
            .expect("openchat checkpoint has at least one q4_k ffn_gate tensor with a full super-block");

        let decoded = gguf_tensor_as_f32(&parsed, &file_bytes, &tensor_name).expect("bind real q4_k tensor by name");
        let weight_row: Vec<f32> = decoded[..q4_k::QK_K].to_vec();

        let activation: Vec<f32> = (0..q4_k::QK_K).map(|index| 0.01 * (index as f32) - 1.28).collect();

        let mut program = Vec::new();
        let weight_node = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(q4_k::QK_K as u32)],
                name: Some("weight".into()),
            },
        );
        let activation_node = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(q4_k::QK_K as u32)],
                name: Some("activation".into()),
            },
        );
        let identity_map = IndexMap::Affine(map::projection(1, &[0]));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(weight_node, identity_map.clone()), (activation_node, identity_map)],
                name: None,
            },
        );
        let dot = append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );

        let symbols: Vec<u64> = Vec::new();
        let named: [(&str, &[f32]); 2] = [("weight", weight_row.as_slice()), ("activation", activation.as_slice())];
        let evaluated = evaluate_named(&program, &symbols, &named, &[dot]).expect("evaluate_named binds by name");
        let (interpreter_output, _shape) = evaluated.get(dot).expect("dot product node present in output");

        // independent computation: raw bytes -> dequantize_block -> manual
        // dot product, never touching `bind::gguf_tensor_as_f32` or the
        // interpreter.
        let tensor = parsed.tensors.iter().find(|tensor| tensor.name == tensor_name).expect("tensor still present");
        let range = parsed
            .tensor_data_range(tensor, file_bytes.len() as u64)
            .expect("tensor byte range");
        let raw_block = &file_bytes[range.start as usize..range.start as usize + q4_k::BLOCK_BYTES];
        let mut independent_weights = [0.0f32; q4_k::QK_K];
        q4_k::dequantize_block(raw_block, &mut independent_weights);
        let expected: f32 = independent_weights
            .iter()
            .zip(activation.iter())
            .map(|(weight, value)| weight * value)
            .sum();

        let max_diff = (interpreter_output[0] - expected).abs();
        eprintln!(
            "real_q4_k_matmul tensor={tensor_name} interpreter={} independent={expected} max_diff={max_diff}",
            interpreter_output[0]
        );
        assert!(
            expected.abs() > 1e-3,
            "degenerate control: expected dot product is ~zero ({expected}), this run proves nothing about real agreement"
        );
        assert!(max_diff < 1e-3, "interpreter and independent dequantize-then-multiply diverged: max_diff={max_diff}");
    }
}
