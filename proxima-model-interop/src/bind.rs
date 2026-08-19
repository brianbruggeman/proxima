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
    use proxima_tensor::op::{self, Extent, Keep, Op, Reduce, ReduceInit, ScalarOp, append};
    use proxima_tokenizer::greedy_pick;

    use super::gguf_tensor_as_f32;

    const FIXTURE_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/TheBloke/openchat-3.5-1210-GGUF/openchat-3.5-1210.Q4_K_S.gguf";

    /// One real greedy-decoded token out of the real openchat-3.5 (Mistral
    /// architecture) checkpoint: every weight `mistral_forward_program`
    /// needs is bound by name through [`gguf_tensor_as_f32`], a real prompt
    /// is BPE-encoded through `proxima_tokenizer`, `evaluate_named` runs
    /// the whole 32-layer forward, and the last position's logits are
    /// greedy-picked and decoded back to text.
    ///
    /// GGUF stores every 2D projection weight `[in, out]` in `TensorInfo`'s
    /// `dims` (`dims[0]` is `ne[0]`, the file's contiguous/row-length axis
    /// -- confirmed against this exact file: `token_embd.weight` dims
    /// `[4096, 32002]` with `4096` the embedding width), but the flat byte
    /// layout that produces is row-major `[out, in]` (`out` rows of
    /// contiguous `in` values -- ggml's `nn.Linear`-style weight layout).
    /// `mistral_forward_program`'s access patterns (`"ihd->shdi"` for
    /// `wq`, `"dg->sdg"` for `w_gate`, `"gd->sgd"` for `w_down`, ...) all
    /// index their weight as `[in, ...out]`, so every projection weight
    /// except `token_embd.weight` (whose target `[vocab, embedding]` shape
    /// already equals the GGUF-native flat layout, since an embedding
    /// table is indexed by row, not projected) needs an explicit transpose
    /// at load time -- [`transpose_out_in_to_in_out`] below.
    ///
    /// Known finding, not fixed here: this checkpoint's `tokenizer.ggml.
    /// model` metadata key is `"llama"` (SentencePiece unigram, carrying a
    /// `tokenizer.ggml.scores` array) and has no `tokenizer.ggml.merges`
    /// key at all, but `proxima_tokenizer::pipe::encode` is a byte-level
    /// BPE codec keyed on merges -- this crate's own module doc says as
    /// much ("the variant `tokenizer.ggml.model = "gpt2"` ... identify on
    /// the real fixture this crate was built against"). With no merges to
    /// apply, `encode_with_bos_eos` degenerates to one token per input
    /// byte: encoding `"The capital of France is"` (24 characters) plus
    /// BOS yields `sequence == 25`, not the ~6 subword tokens a working
    /// tokenizer would produce. That is the tokenizer stage, not the
    /// forward pass or the weight transpose above, and it is why this
    /// test's picked token is not `"▁Paris"`.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo, and dequantizes ~29GB of weights"]
    fn runs_one_real_forward_pass_and_greedy_picks_a_real_token() {
        let path = std::path::Path::new(FIXTURE_PATH);
        if !path.exists() {
            eprintln!("skipping: no host-local openchat gguf fixture at {FIXTURE_PATH}");
            return;
        }

        const VOCAB: usize = 32_002;
        const EMBEDDING: usize = 4096;
        const FEED_FORWARD: usize = 14_336;
        const QUERY_HEADS: usize = 32;
        const KV_HEADS: usize = 8;
        const HEAD_DIM: usize = 128;
        const PAIRS: usize = HEAD_DIM / 2;
        const GROUP: usize = QUERY_HEADS / KV_HEADS;
        const BLOCK_COUNT: u32 = 32;
        const ROPE_FREQ_BASE: f32 = 10_000.0;
        const RMS_EPSILON: f32 = 1e-5;

        let load_start = std::time::Instant::now();
        let (parsed, file_bytes) = proxima_gguf::edge::read_file(path).expect("read host-local openchat gguf fixture");

        let mut resident_bytes: usize = file_bytes.len();
        let mut named: Vec<(alloc::string::String, Vec<f32>)> = Vec::new();

        let mut bind = |name: &str| -> Vec<f32> {
            let decoded = gguf_tensor_as_f32(&parsed, &file_bytes, name)
                .unwrap_or_else(|error| panic!("bind real tensor {name} by name: {error}"));
            resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            decoded
        };

        let table = bind("token_embd.weight");
        named.push(("token_embd.weight".into(), table));

        for layer in 0..BLOCK_COUNT {
            let wq = transpose_out_in_to_in_out(&bind(&alloc::format!("blk.{layer}.attn_q.weight")), EMBEDDING, EMBEDDING);
            let wk = transpose_out_in_to_in_out(&bind(&alloc::format!("blk.{layer}.attn_k.weight")), KV_HEADS * HEAD_DIM, EMBEDDING);
            let wv = transpose_out_in_to_in_out(&bind(&alloc::format!("blk.{layer}.attn_v.weight")), KV_HEADS * HEAD_DIM, EMBEDDING);
            let wo = transpose_out_in_to_in_out(&bind(&alloc::format!("blk.{layer}.attn_output.weight")), EMBEDDING, EMBEDDING);
            let w_gate = transpose_out_in_to_in_out(&bind(&alloc::format!("blk.{layer}.ffn_gate.weight")), FEED_FORWARD, EMBEDDING);
            let w_up = transpose_out_in_to_in_out(&bind(&alloc::format!("blk.{layer}.ffn_up.weight")), FEED_FORWARD, EMBEDDING);
            let w_down = transpose_out_in_to_in_out(&bind(&alloc::format!("blk.{layer}.ffn_down.weight")), EMBEDDING, FEED_FORWARD);

            named.push((alloc::format!("blk.{layer}.attn_q.weight"), wq));
            named.push((alloc::format!("blk.{layer}.attn_k.weight"), wk));
            named.push((alloc::format!("blk.{layer}.attn_v.weight"), wv));
            named.push((alloc::format!("blk.{layer}.attn_output.weight"), wo));
            named.push((alloc::format!("blk.{layer}.ffn_gate.weight"), w_gate));
            named.push((alloc::format!("blk.{layer}.ffn_up.weight"), w_up));
            named.push((alloc::format!("blk.{layer}.ffn_down.weight"), w_down));
        }

        let lm_head = transpose_out_in_to_in_out(&bind("output.weight"), VOCAB, EMBEDDING);
        named.push(("output.weight".into(), lm_head));

        let load_elapsed = load_start.elapsed();
        std::println!(
            "load: wall_clock={load_elapsed:?} resident_bytes={resident_bytes} ({:.2} GiB)",
            resident_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("build vocab from openchat gguf metadata");
        let prompt = "The capital of France is";
        let ids = proxima_tokenizer::encode_with_bos_eos(prompt, &vocab, true, false).expect("encode prompt");
        let sequence = ids.len();

        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();
        let inv_dim = alloc::vec![1.0 / EMBEDDING as f32; sequence];
        let epsilon = alloc::vec![RMS_EPSILON; sequence];
        let ones = alloc::vec![1.0f32; sequence];
        let group_ones = alloc::vec![1.0f32; KV_HEADS * GROUP];

        let mut cos = alloc::vec![0.0f32; sequence * PAIRS];
        let mut sin = alloc::vec![0.0f32; sequence * PAIRS];
        for position in 0..sequence {
            for pair in 0..PAIRS {
                let theta = (position as f32) * ROPE_FREQ_BASE.powf(-((2 * pair) as f32) / (HEAD_DIM as f32));
                cos[position * PAIRS + pair] = theta.cos();
                sin[position * PAIRS + pair] = theta.sin();
            }
        }

        let program = proxima_tensor::spec::mistral_forward_program(
            VOCAB as u32,
            EMBEDDING as u32,
            FEED_FORWARD as u32,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            BLOCK_COUNT,
        )
        .expect("the whole forward pass lowers to a program");

        let mut named_slices: Vec<(&str, &[f32])> = Vec::with_capacity(named.len() + 6);
        named_slices.push(("ids", ids_f32.as_slice()));
        for (name, data) in &named {
            named_slices.push((name.as_str(), data.as_slice()));
        }
        named_slices.push(("inv_dim", inv_dim.as_slice()));
        named_slices.push(("eps", epsilon.as_slice()));
        named_slices.push(("ones", ones.as_slice()));
        named_slices.push(("rope_cos", cos.as_slice()));
        named_slices.push(("rope_sin", sin.as_slice()));
        named_slices.push(("group_ones", group_ones.as_slice()));

        let root = op::NodeId(program.len() as u32 - 1);
        let symbols = [sequence as u64];

        let forward_start = std::time::Instant::now();
        let evaluated =
            evaluate_named(&program, &symbols, &named_slices, &[root]).expect("evaluate_named binds the whole forward pass by name");
        let forward_elapsed = forward_start.elapsed();

        let (logits, shape) = evaluated.get(root).expect("logits present in output");
        assert_eq!(shape, [sequence as u64, VOCAB as u64], "logits must be [seq, vocab]");

        let last_position = &logits[(sequence - 1) * VOCAB..sequence * VOCAB];

        // degenerate control: a dead forward pass (all-zero or collapsed
        // weights) produces constant logits, which would make greedy_pick
        // return a confident-looking but meaningless index-0 token.
        let first = last_position[0];
        assert!(
            last_position.iter().any(|&value| value != first),
            "degenerate control failed: logits are all-equal ({first}), forward pass produced no signal"
        );
        assert!(
            last_position.iter().any(|&value| value != 0.0),
            "degenerate control failed: logits are all-zero, forward pass produced no signal"
        );

        let token_id = greedy_pick(last_position).expect("logits are nonempty");
        let token_text = proxima_tokenizer::decode(&[token_id], &vocab).expect("decode picked token id");

        std::println!(
            "prompt={prompt:?} sequence={sequence} token_id={token_id} token={token_text:?} forward_wall_clock={forward_elapsed:?}"
        );
    }

    /// Row-major transpose from GGUF's native flat layout (`[out, in]`,
    /// `out` rows of contiguous `in` values, ggml's linear-weight layout)
    /// to `mistral_forward_program`'s expected `[in, out]` layout. See the
    /// doc comment on [`real_openchat_file::runs_one_real_forward_pass_and_greedy_picks_a_real_token`]
    /// for the derivation, checked against this file's own tensor shapes.
    fn transpose_out_in_to_in_out(flat: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        assert_eq!(flat.len(), out_dim * in_dim, "flat buffer length must match out_dim * in_dim");
        let mut transposed = alloc::vec![0.0f32; flat.len()];
        for out_index in 0..out_dim {
            for in_index in 0..in_dim {
                transposed[in_index * out_dim + out_index] = flat[out_index * in_dim + in_index];
            }
        }
        transposed
    }

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
