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

/// Zero-copy counterpart to [`gguf_tensor_as_f32`] for a k-quant tensor:
/// borrows `name`'s raw packed bytes straight out of `file_bytes` and wraps
/// them as the matching [`proxima_tensor::cpu::QuantizedBlock`] variant,
/// instead of dequantizing into an owned `Vec<f32>` first. No copy, no
/// allocation -- exactly the bytes GGUF already stored.
///
/// One function over `Q4_K`/`Q5_K`/`Q6_K` rather than three near-identical
/// ones: the three differ only in which variant carries the byte range, and
/// every k-quant super-block is stored the same way -- a contiguous
/// row-major `[out, in]` byte run whose per-row period is a function of the
/// type alone. A per-type entry point would be that `match` arm rewritten as
/// a signature, three times over.
///
/// This works without a transpose, unlike `gguf_tensor_as_f32`'s callers
/// for a 2D projection weight (see `transpose_out_in_to_in_out` in this
/// crate's real-forward-pass test): a packed [`proxima_tensor::cpu::QuantizedBlock`]
/// bypasses the interpreter's strided operand machinery entirely -- the
/// `proxima_tensor::cpu::matmul_q4k_f32` family walks `weights` as `rows`
/// contiguous per-row byte chunks and dot-products each row against the
/// activation directly, so it only ever needs GGUF's native on-disk
/// row-major `[out, in]` layout, the layout this function hands through
/// unchanged.
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if `name` isn't in `parsed.tensors`;
/// [`InteropError::Gguf`] if the tensor's declared byte range doesn't fit
/// `file_bytes`; [`InteropError::UnrepresentableGgmlType`] if `name`'s
/// tensor is not one of `Q4_K`/`Q5_K`/`Q6_K` -- callers route anything else
/// through [`gguf_tensor_as_f32`] instead, which is what this crate has a
/// decoder for.
#[cfg(feature = "std")]
pub fn gguf_tensor_as_packed_block<'a>(
    parsed: &ParsedGguf,
    file_bytes: &'a [u8],
    name: &str,
) -> Result<proxima_tensor::cpu::QuantizedBlock<'a>, InteropError> {
    let tensor = find_tensor(parsed, name)?;
    let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
    let bytes = &file_bytes[range.start as usize..range.end as usize];
    match tensor.ggml_type {
        GgmlType::Q4_K => Ok(proxima_tensor::cpu::QuantizedBlock::Q4K(bytes)),
        GgmlType::Q5_K => Ok(proxima_tensor::cpu::QuantizedBlock::Q5K(bytes)),
        GgmlType::Q6_K => Ok(proxima_tensor::cpu::QuantizedBlock::Q6K(bytes)),
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
    use core::ffi::c_void;
    use std::os::fd::AsFd;

    use proxima_gguf::GgmlType;
    use proxima_gguf::pipe::ParsedGguf;
    use proxima_gguf::quant::q4_k;
    use proxima_tensor::DType;
    use proxima_tensor::cpu::{QuantizedBlock, evaluate_named, evaluate_quantized_named};
    #[cfg(feature = "instrument")]
    use proxima_tensor::instrument;
    use proxima_tensor::map::{self, IndexMap};
    use proxima_tensor::op::{self, Extent, Keep, Op, Reduce, ReduceInit, ScalarOp, append};
    use proxima_tokenizer::greedy_pick;

    use crate::loader::prefault;
    use crate::serving::{ServingConfig, apply_serving_config};

    use super::{gguf_tensor_as_f32, gguf_tensor_as_packed_block};

    /// A read-only `mmap` of the fixture file (rustix, already a workspace
    /// dependency used the same way by `proxima-storage/src/dax/region.rs`
    /// for its own domain) -- the whole reason to bind `Q4_K` tensors
    /// packed is so the byte range GGUF already stored is the buffer
    /// `evaluate_quantized_named` reads, with no owned copy in between;
    /// `proxima_gguf::edge::read_file`'s `std::fs::read` would put that
    /// copy straight back, one whole-file `Vec<u8>` at a time (its own doc,
    /// `proxima-gguf/src/edge.rs:3`, names this exact tradeoff). `MapFlags::
    /// PRIVATE`/`ProtFlags::READ` because this test only ever reads the
    /// fixture; unmapped on drop.
    struct MappedGguf {
        base: *mut u8,
        len: usize,
        _file: std::fs::File,
    }

    impl MappedGguf {
        fn open(path: &std::path::Path) -> std::io::Result<Self> {
            let file = std::fs::File::open(path)?;
            let len = usize::try_from(file.metadata()?.len()).expect("fixture file length fits in usize");
            // SAFETY: `len` matches the just-opened file's own length; `file`
            // is kept alive in `_file` for as long as `base` is used, and the
            // mapping is read-only/private so no writer can observe or race it.
            let base = unsafe {
                rustix::mm::mmap(
                    core::ptr::null_mut(),
                    len,
                    rustix::mm::ProtFlags::READ,
                    rustix::mm::MapFlags::PRIVATE,
                    file.as_fd(),
                    0,
                )
            }
            .expect("mmap host-local openchat gguf fixture")
            .cast::<u8>();
            Ok(Self { base, len, _file: file })
        }

        fn as_slice(&self) -> &[u8] {
            // SAFETY: `base` points at `len` bytes mapped for `self`'s whole
            // lifetime; this borrows `self` immutably, so nothing can unmap
            // the region while the returned slice is alive.
            unsafe { core::slice::from_raw_parts(self.base, self.len) }
        }
    }

    impl Drop for MappedGguf {
        fn drop(&mut self) {
            // SAFETY: `base`/`len` are exactly what `open`'s `mmap` call
            // returned; nothing else unmaps this region.
            let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
        }
    }

    /// The load loop's own accumulators, grouped so `bind_dense`/
    /// `bind_matmul_weight` take one `&mut` argument instead of three --
    /// purely a call-site grouping local to this test, not a library type:
    /// nothing outside this module ever sees it.
    struct LoadState<'file> {
        resident_bytes: usize,
        owned: Vec<(alloc::string::String, Vec<f32>)>,
        packed: Vec<(alloc::string::String, QuantizedBlock<'file>)>,
    }

    /// A learned 1-D scale (RMSNorm weight) or `token_embd.weight` (indexed
    /// by row via `embedding_lookup`, never projected) -- always dequantized
    /// to owned `f32`, no `Q4_K` packed path: this checkpoint's norms are
    /// `F32` on disk already, and even a quantized `token_embd.weight`
    /// would not qualify for `reject_non_float32`'s matmul exemption
    /// (`is_quantized_matmul_operand` requires feeding a `Multiply`-then-
    /// `Add` reduce, which a gather is not).
    fn bind_dense(parsed: &ParsedGguf, file_bytes: &[u8], name: alloc::string::String, state: &mut LoadState) {
        let decoded = gguf_tensor_as_f32(parsed, file_bytes, &name)
            .unwrap_or_else(|error| panic!("bind real tensor {name} by name: {error}"));
        state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
        state.owned.push((name, decoded));
    }

    /// A 2-D projection weight `mistral_forward_program` uses as one
    /// `Multiply`-then-`Add`-reduce (matmul) operand -- the shape
    /// `reject_non_float32`'s quantized-weight exemption requires. Tries
    /// [`gguf_tensor_as_packed_block`] first: a `Q4_K`/`Q5_K`/`Q6_K` tensor
    /// binds packed, zero-copy, straight out of the mmap's bytes (the
    /// `matmul_q*k_f32` kernels want GGUF's native on-disk `[out, in]`
    /// row-major layout directly, so this arm skips
    /// `transpose_out_in_to_in_out` entirely -- see
    /// `gguf_tensor_as_packed_block`'s own doc). Only `F32` now falls back to
    /// dequantize-then-transpose; `Q5_K`/`Q6_K` used to take that path too,
    /// which meant the packed kernels for them were correct and unreachable.
    fn bind_matmul_weight<'file>(
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        name: alloc::string::String,
        out_dim: usize,
        in_dim: usize,
        state: &mut LoadState<'file>,
    ) {
        match gguf_tensor_as_packed_block(parsed, file_bytes, &name) {
            Ok(block) => state.packed.push((name, block)),
            Err(_) => {
                let decoded = gguf_tensor_as_f32(parsed, file_bytes, &name)
                    .unwrap_or_else(|error| panic!("bind real tensor {name} by name: {error}"));
                state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
                state.owned.push((name, transpose_out_in_to_in_out(&decoded, out_dim, in_dim)));
            }
        }
    }

    /// One real greedy-decoded token out of the real openchat-3.5 (Mistral
    /// architecture) checkpoint: every weight `mistral_forward_program`
    /// needs is bound by name -- `Q4_K` tensors packed straight out of an
    /// mmap via [`gguf_tensor_as_packed_block`], everything else dequantized
    /// through [`gguf_tensor_as_f32`] -- a real prompt is BPE-encoded
    /// through `proxima_tokenizer`, `evaluate_quantized_named` runs the
    /// whole 32-layer forward, and the last position's logits are
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
    /// index their weight as `[in, ...out]`, so every dequantized-f32
    /// projection weight except `token_embd.weight` (whose target
    /// `[vocab, embedding]` shape already equals the GGUF-native flat
    /// layout, since an embedding table is indexed by row, not projected)
    /// needs an explicit transpose at load time --
    /// [`transpose_out_in_to_in_out`] below. A packed `Q4_K` weight skips
    /// this transpose entirely -- see `bind_matmul_weight`'s own doc.
    ///
    /// This checkpoint's `tokenizer.ggml.model` metadata key is `"llama"`
    /// (SentencePiece/SPM, carrying a `tokenizer.ggml.scores` array, no
    /// `tokenizer.ggml.merges` key at all). `proxima_tokenizer::gguf::
    /// vocab_from_metadata` now dispatches on that key (`Vocab::
    /// new_unigram` for `"llama"`, `Vocab::new` for `"gpt2"`), and
    /// `proxima_tokenizer::pipe::encode`/`decode` select the matching
    /// encoder from `Vocab::is_unigram` -- see `proxima-tokenizer/src/
    /// unigram.rs`. Encoding `"The capital of France is"` now segments
    /// into the real vocab's five subword pieces (`sequence == 6` with
    /// BOS), not one token per byte.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo"]
    fn runs_one_real_forward_pass_and_greedy_picks_a_real_token() {
        // `ServingConfig::default()` is the TARGET invocation, so it trips the
        // first `todo!` by design. the overrides below are exactly the knobs
        // this forward does not implement yet -- the delta IS the gap list.
        let serving_config = ServingConfig {
            kv_cache_key_quant: GgmlType::F16,
            kv_cache_value_quant: GgmlType::F16,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            reasoning_budget: 0,
            ..ServingConfig::default()
        };
        let path = std::path::Path::new(serving_config.model_path);
        if !path.exists() {
            eprintln!(
                "skipping: no host-local openchat gguf fixture at {}",
                serving_config.model_path
            );
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

        // DIAGNOSTIC (proxima-debugger, remove before landing): phase
        // markers with epoch-ms timestamps so an external RSS/footprint
        // sampler on this process's pid can be correlated against the
        // load/parse/bind/forward phase boundaries.
        fn diag_now_ms() -> u128 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_millis()
        }
        let diag_pid = std::process::id();
        std::eprintln!("DIAG phase=process_start t_ms={} pid={diag_pid}", diag_now_ms());

        let load_start = std::time::Instant::now();
        let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
        std::eprintln!("DIAG phase=mmap_open t_ms={} pid={diag_pid}", diag_now_ms());
        let file_bytes = mapped.as_slice();
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes).expect("parse host-local openchat gguf fixture");
        std::eprintln!("DIAG phase=parse_complete t_ms={} pid={diag_pid}", diag_now_ms());

        // `resident_bytes` starts at the mapped file's own length: a full
        // 32-layer forward pass reads essentially every packed weight byte,
        // so the kernel demand-pages nearly the whole mapping into this
        // process's resident set even though nothing here copies it --
        // `MappedGguf` never allocates a second 3.94 GB buffer the way
        // `proxima_gguf::edge::read_file`'s `std::fs::read` did. Every
        // owned `f32` buffer `bind_dense`/`bind_matmul_weight` allocate
        // (norms, plus `token_embd.weight`, which is an embedding lookup
        // rather than a matmul and so has no packed kernel) adds on top.
        //
        // This is a DERIVED counter, not a measurement: it is the mmap
        // length plus the owned buffers, and it tracked 1.42 GiB below the
        // kernel's own `phys_footprint` even before `Q5_K`/`Q6_K` bound
        // packed. Quote max RSS, not this, for what serving costs.
        let mut state = LoadState {
            resident_bytes: file_bytes.len(),
            owned: Vec::new(),
            packed: Vec::new(),
        };

        bind_dense(&parsed, file_bytes, "token_embd.weight".into(), &mut state);

        for layer in 0..BLOCK_COUNT {
            // 1-D `[embedding]` learned RMSNorm scale -- no `[out, in]`
            // GGUF layout to undo, so it skips `transpose_out_in_to_in_out`
            // (that helper is for rank-2 projections only).
            bind_dense(&parsed, file_bytes, alloc::format!("blk.{layer}.attn_norm.weight"), &mut state);
            bind_dense(&parsed, file_bytes, alloc::format!("blk.{layer}.ffn_norm.weight"), &mut state);
            bind_matmul_weight(&parsed, file_bytes, alloc::format!("blk.{layer}.attn_q.weight"), EMBEDDING, EMBEDDING, &mut state);
            bind_matmul_weight(&parsed, file_bytes, alloc::format!("blk.{layer}.attn_k.weight"), KV_HEADS * HEAD_DIM, EMBEDDING, &mut state);
            bind_matmul_weight(&parsed, file_bytes, alloc::format!("blk.{layer}.attn_v.weight"), KV_HEADS * HEAD_DIM, EMBEDDING, &mut state);
            bind_matmul_weight(&parsed, file_bytes, alloc::format!("blk.{layer}.attn_output.weight"), EMBEDDING, EMBEDDING, &mut state);
            bind_matmul_weight(&parsed, file_bytes, alloc::format!("blk.{layer}.ffn_gate.weight"), FEED_FORWARD, EMBEDDING, &mut state);
            bind_matmul_weight(&parsed, file_bytes, alloc::format!("blk.{layer}.ffn_up.weight"), FEED_FORWARD, EMBEDDING, &mut state);
            bind_matmul_weight(&parsed, file_bytes, alloc::format!("blk.{layer}.ffn_down.weight"), EMBEDDING, FEED_FORWARD, &mut state);
        }

        bind_dense(&parsed, file_bytes, "output_norm.weight".into(), &mut state);
        bind_matmul_weight(&parsed, file_bytes, "output.weight".into(), VOCAB, EMBEDDING, &mut state);

        let load_elapsed = load_start.elapsed();
        let resident_bytes = state.resident_bytes;
        std::eprintln!("DIAG phase=bind_complete t_ms={} pid={diag_pid}", diag_now_ms());
        std::println!(
            "load: wall_clock={load_elapsed:?} resident_bytes={resident_bytes} ({:.2} GiB) packed_tensors={} owned_tensors={}",
            resident_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            state.packed.len(),
            state.owned.len()
        );
        // Names every Q5_K/Q6_K-packed weight. These ran single-threaded
        // until `matmul_q5k_f32`/`matmul_q6k_f32` gained the shared
        // `matmul_quantized_dispatch` -- 990ms of a 1595ms forward. They
        // dispatch now; the listing stays because these 9 tensors are the
        // ones whose codec differs from the other 216.
        for (name, block) in &state.packed {
            match block {
                QuantizedBlock::Q5K(bytes) => {
                    std::eprintln!("DIAG packed_tensor name={name} codec=Q5K bytes={}", bytes.len())
                }
                QuantizedBlock::Q6K(bytes) => {
                    std::eprintln!("DIAG packed_tensor name={name} codec=Q6K bytes={}", bytes.len())
                }
                QuantizedBlock::Q4K(_) | QuantizedBlock::Float32(_) => {}
            }
        }

        // DIAGNOSTIC (proxima-debugger, remove before landing): A/B toggle
        // for `crate::loader::prefault` at the loader's own phase boundary
        // -- bind just finished (every packed tensor's byte range is now
        // known), forward has not started. `PROXIMA_PREFAULT=1` warms every
        // page of the mmap through the shared background pool before the
        // timed forward runs; unset skips it entirely (the explicit-opt-in
        // this module's own doc names: a caller serving many small models
        // should not pay to warm a mapping it will only read a slice of).
        let prefault_enabled = std::env::var("PROXIMA_PREFAULT").is_ok_and(|value| value == "1");
        #[cfg(feature = "instrument")]
        let diag_minflt_prefault_before = instrument::ru_minflt();
        let prefault_start = std::time::Instant::now();
        if prefault_enabled {
            prefault(file_bytes).expect("prefault the host-local openchat gguf mapping");
        }
        let prefault_elapsed = prefault_start.elapsed();
        #[cfg(feature = "instrument")]
        let diag_minflt_prefault_after = instrument::ru_minflt();
        std::eprintln!(
            "DIAG phase=prefault t_ms={} pid={diag_pid} enabled={prefault_enabled}",
            diag_now_ms()
        );
        std::println!("prefault: enabled={prefault_enabled} wall_clock={prefault_elapsed:?}");
        #[cfg(feature = "instrument")]
        std::println!(
            "DIAG prefault_minflt_delta={}",
            diag_minflt_prefault_after.saturating_sub(diag_minflt_prefault_before)
        );

        let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("build vocab from openchat gguf metadata");
        let prompt = "The capital of France is";
        let ids = proxima_tokenizer::encode_with_bos_eos(prompt, &vocab, true, false).expect("encode prompt");
        let sequence = ids.len();
        apply_serving_config(&serving_config, sequence);

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

        let mut named_blocks: Vec<(&str, QuantizedBlock)> = Vec::with_capacity(state.owned.len() + state.packed.len() + 6);
        named_blocks.push(("ids", QuantizedBlock::Float32(ids_f32.as_slice())));
        for (name, data) in &state.owned {
            named_blocks.push((name.as_str(), QuantizedBlock::Float32(data.as_slice())));
        }
        for (name, block) in &state.packed {
            named_blocks.push((name.as_str(), *block));
        }
        named_blocks.push(("inv_dim", QuantizedBlock::Float32(inv_dim.as_slice())));
        named_blocks.push(("eps", QuantizedBlock::Float32(epsilon.as_slice())));
        named_blocks.push(("ones", QuantizedBlock::Float32(ones.as_slice())));
        named_blocks.push(("rope_cos", QuantizedBlock::Float32(cos.as_slice())));
        named_blocks.push(("rope_sin", QuantizedBlock::Float32(sin.as_slice())));
        named_blocks.push(("group_ones", QuantizedBlock::Float32(group_ones.as_slice())));

        let root = op::NodeId(program.len() as u32 - 1);
        let symbols = [sequence as u64];

        #[cfg(feature = "instrument")]
        {
            instrument::reset_parallel();
            instrument::reset_matmul_dispatch();
            instrument::reset_worker_cpu();
            instrument::reset_q4k_shape_buckets();
            instrument::cohort::reset();
        }
        std::eprintln!("DIAG phase=forward_start t_ms={} pid={diag_pid}", diag_now_ms());
        // `forward_start`/`forward_elapsed` are the one clock read per
        // forward this test always takes -- not under investigation, and
        // required for `forward_wall_clock` below whether or not
        // `instrument` (per-chunk/per-node reads) is compiled in.
        let forward_start = std::time::Instant::now();
        #[cfg(feature = "instrument")]
        let main_thread_cpu_start = instrument::thread_cpu_nanos();
        // DIAGNOSTIC (proxima-debugger, remove before landing): minor-fault
        // delta across the forward call. `evaluate_quantized_named` reads
        // essentially every `Q4_K` weight byte from the mmap; if the parser
        // no longer pre-faults it (established: load 1,473ms -> 1,605ms
        // when pre-fault was removed), first-touch page-in during the
        // forward should show up here as a nonzero minflt delta.
        #[cfg(feature = "instrument")]
        let diag_minflt_before = instrument::ru_minflt();
        let evaluated = evaluate_quantized_named(&program, &symbols, &named_blocks, &[root])
            .expect("evaluate_quantized_named binds the whole forward pass by name, packed weights included");
        #[cfg(feature = "instrument")]
        let diag_minflt_after = instrument::ru_minflt();
        #[cfg(feature = "instrument")]
        let main_thread_cpu_nanos = instrument::thread_cpu_nanos() - main_thread_cpu_start;
        let forward_elapsed = forward_start.elapsed();
        std::eprintln!("DIAG phase=forward_complete t_ms={} pid={diag_pid}", diag_now_ms());
        #[cfg(feature = "instrument")]
        let wall_ns = forward_elapsed.as_nanos() as u64;
        #[cfg(feature = "instrument")]
        let parallel = instrument::parallel_totals();
        #[cfg(feature = "instrument")]
        let matmul_dispatch = instrument::matmul_dispatch_totals();
        #[cfg(feature = "instrument")]
        std::println!(
            "wall_ns={wall_ns}  main_thread_cpu_ns={main_thread_cpu_nanos}  ratio={:.4}",
            main_thread_cpu_nanos as f64 / wall_ns as f64
        );
        #[cfg(feature = "instrument")]
        {
            std::println!(
                "nodes={}  parallel_nodes={}  parallel_chunks={}",
                program.len(),
                parallel.parallel_nodes,
                parallel.chunk_count
            );
            std::println!(
                "DIAG minflt_delta={} (before={diag_minflt_before} after={diag_minflt_after})",
                diag_minflt_after.saturating_sub(diag_minflt_before)
            );
            // every `_ticks` field below is a raw `proxima_clock::Ticks` delta
            // (`instrument::read_ticks`'s doc) -- converted to nanoseconds
            // exactly once, here at the print edge, via `ticks_to_nanos`,
            // never inside the loop that produced it.
            std::println!(
                "DIAG matmul_dispatch workers_calls={} workers_none={} threaded_calls={} setup_ms={:.3} available_parallelism_ms={:.3} spawn_ms={:.3} own_chunk_ms={:.3} recv_wait_ms={:.3} quantize_activation_ms={:.3}",
                matmul_dispatch.workers_calls,
                matmul_dispatch.workers_none,
                matmul_dispatch.calls,
                instrument::ticks_to_nanos(matmul_dispatch.setup_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(matmul_dispatch.available_parallelism_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(matmul_dispatch.spawn_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(matmul_dispatch.own_chunk_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(matmul_dispatch.recv_wait_ticks) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(matmul_dispatch.quantize_activation_ticks) as f64 / 1_000_000.0,
            );
            std::println!(
                "DIAG matmul_reduce_quantized_ms={:.3}",
                instrument::ticks_to_nanos(matmul_dispatch.reduce_quantized_ticks) as f64 / 1_000_000.0
            );
            std::println!(
                "DIAG matmul_q5k_f32 calls={} total_ms={:.3}  matmul_q6k_f32 calls={} total_ms={:.3}",
                matmul_dispatch.q5k_f32_calls,
                instrument::ticks_to_nanos(matmul_dispatch.q5k_f32_ticks) as f64 / 1_000_000.0,
                matmul_dispatch.q6k_f32_calls,
                instrument::ticks_to_nanos(matmul_dispatch.q6k_f32_ticks) as f64 / 1_000_000.0,
            );
            std::println!(
                "DIAG matmul_chunk_compute chunk_count={} sum_ms={:.3} min_us={:.3} max_us={:.3}",
                parallel.chunk_count,
                instrument::ticks_to_nanos(parallel.chunk_ticks_sum) as f64 / 1_000_000.0,
                instrument::ticks_to_nanos(parallel.chunk_ticks_min) as f64 / 1_000.0,
                instrument::ticks_to_nanos(parallel.chunk_ticks_max) as f64 / 1_000.0,
            );
            // DIAGNOSTIC (proxima-debugger, remove before landing): mac-count
            // and per-codec ns/mac, directly comparable against the isolated
            // single-threaded kernel bench (0.0334 ns/mac) and ggml's own
            // (0.0255 ns/mac) -- see instrument.rs's MATMUL_Q4K_MACS/etc doc.
            let total_matmul_macs = matmul_dispatch.q4k_macs + matmul_dispatch.q5k_macs + matmul_dispatch.q6k_macs;
            std::println!(
                "DIAG matmul_reduce_quantized_calls={} position_loop_iters={} total_macs={total_matmul_macs}",
                matmul_dispatch.reduce_quantized_calls, matmul_dispatch.position_loop_iters,
            );
            let reduce_quantized_nanos = instrument::ticks_to_nanos(matmul_dispatch.reduce_quantized_ticks);
            std::println!(
                "DIAG matmul_bucket_ns_per_mac={:.6}  (reduce_quantized_ms={:.3} / total_macs={total_matmul_macs})",
                reduce_quantized_nanos as f64 / total_matmul_macs as f64,
                reduce_quantized_nanos as f64 / 1_000_000.0,
            );
            let q4k_call_nanos = instrument::ticks_to_nanos(matmul_dispatch.q4k_call_ticks);
            std::println!(
                "DIAG q4k macs={} call_ns_sum_ms={:.3} ns_per_mac={:.6}",
                matmul_dispatch.q4k_macs,
                q4k_call_nanos as f64 / 1_000_000.0,
                q4k_call_nanos as f64 / matmul_dispatch.q4k_macs.max(1) as f64,
            );
            let q5k_call_nanos = instrument::ticks_to_nanos(matmul_dispatch.q5k_call_ticks);
            std::println!(
                "DIAG q5k macs={} call_ns_sum_ms={:.3} ns_per_mac={:.6}",
                matmul_dispatch.q5k_macs,
                q5k_call_nanos as f64 / 1_000_000.0,
                q5k_call_nanos as f64 / matmul_dispatch.q5k_macs.max(1) as f64,
            );
            let q6k_call_nanos = instrument::ticks_to_nanos(matmul_dispatch.q6k_call_ticks);
            std::println!(
                "DIAG q6k macs={} call_ns_sum_ms={:.3} ns_per_mac={:.6}",
                matmul_dispatch.q6k_macs,
                q6k_call_nanos as f64 / 1_000_000.0,
                q6k_call_nanos as f64 / matmul_dispatch.q6k_macs.max(1) as f64,
            );
            // spawn/own_chunk/recv_wait are timed as one sequential chain per
            // `matmul_rows_threaded` call (`cpu.rs`'s `diag_spawn_started` ->
            // `diag_own_chunk_started` -> `diag_recv_started`), so their sum
            // across every call is the total wall-clock time this process spent
            // inside that function across the whole forward pass -- the
            // denominator `chunk_ticks_sum` (total compute work, summed across
            // every chunk on every worker) needs to read achieved core count
            // directly against the 10 physical cores this box has.
            let dispatch_wall_ns = instrument::ticks_to_nanos(
                matmul_dispatch.setup_ticks
                    + matmul_dispatch.available_parallelism_ticks
                    + matmul_dispatch.spawn_ticks
                    + matmul_dispatch.own_chunk_ticks
                    + matmul_dispatch.recv_wait_ticks,
            );
            let chunk_compute_sum_nanos = instrument::ticks_to_nanos(parallel.chunk_ticks_sum);
            let achieved_parallel_cores = chunk_compute_sum_nanos as f64 / dispatch_wall_ns.max(1) as f64;
            std::println!(
                "DIAG achieved_parallel_cores={achieved_parallel_cores:.3}  (chunk_compute_sum_ms={:.3} / dispatch_wall_ms={:.3})",
                chunk_compute_sum_nanos as f64 / 1_000_000.0,
                dispatch_wall_ns as f64 / 1_000_000.0,
            );
            // DIAGNOSTIC (proxima-debugger, remove before landing): deschedule-
            // immune peer of `matmul_chunk_compute`'s wall-clock sum -- every
            // matmul row-chunk this run executed goes through Q4_K's
            // `matmul_rows_threaded` alone (Q5_K/Q6_K's int8 paths never reach
            // it, confirmed by `parallel_nodes == workers_calls`), so this sum
            // divided by `q4k_macs` is directly comparable to the isolated
            // single-threaded kernel bench's 0.0334 ns/mac WITHOUT host-load
            // wall-clock contamination.
            let worker_cpu_sum_nanos: u64 = instrument::worker_cpu_snapshot().iter().sum();
            std::println!(
                "DIAG matmul_chunk_cpu_sum_ms={:.3}  ns_per_mac_cpu={:.6}  (vs wall ns_per_mac={:.6}, isolated single-thread bench=0.0334)",
                worker_cpu_sum_nanos as f64 / 1_000_000.0,
                worker_cpu_sum_nanos as f64 / matmul_dispatch.q4k_macs.max(1) as f64,
                chunk_compute_sum_nanos as f64 / matmul_dispatch.q4k_macs.max(1) as f64,
            );
            // DIAGNOSTIC (proxima-debugger, remove before landing): per-shape
            // breakdown of the exact same in-situ measurement the aggregate
            // q4k ns_per_mac line above sums away -- settles whether the
            // 0.0462 vs 0.0332 gap is uniform across every matmul shape this
            // forward runs, or concentrated in the small (attn_k/attn_v,
            // rows=1024) shapes ggml's own t8 already regresses at.
            std::println!(
                "DIAG matmul_q4k_transpose_ms={:.3}",
                instrument::ticks_to_nanos(matmul_dispatch.q4k_transpose_ticks) as f64 / 1_000_000.0,
            );
            // DIAGNOSTIC (proxima-debugger, remove before landing): the
            // cohort's own view of a round -- how members wake (spin vs park),
            // and per slot how long after the round opened it claimed its
            // first chunk and how long it sat done before the leader closed.
            {
                use instrument::cohort;
                use core::sync::atomic::Ordering;
                std::println!(
                    "DIAG cohort rounds={} parks={} spin_hits={} immediate_hits={} arm_aborts={} unpark_rounds={} unpark_ms={:.3}",
                    cohort::ROUNDS.load(Ordering::Relaxed),
                    cohort::PARKS.load(Ordering::Relaxed),
                    cohort::SPIN_HITS.load(Ordering::Relaxed),
                    cohort::IMMEDIATE_HITS.load(Ordering::Relaxed),
                    cohort::ARM_ABORTS.load(Ordering::Relaxed),
                    cohort::UNPARK_ROUNDS.load(Ordering::Relaxed),
                    cohort::UNPARK_NANOS.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                );
                for slot in 0..cohort::MAX_SLOTS {
                    let rounds = cohort::SLOT_ROUNDS[slot].load(Ordering::Relaxed);
                    let chunks = cohort::SLOT_CHUNKS[slot].load(Ordering::Relaxed);
                    if rounds == 0 && chunks == 0 {
                        continue;
                    }
                    std::println!(
                        "DIAG cohort_slot slot={slot} rounds={rounds} chunks={chunks} first_claim_ms={:.3} compute_ms={:.3} tail_ms={:.3}",
                        cohort::SLOT_FIRST_CLAIM_NANOS[slot].load(Ordering::Relaxed) as f64 / 1_000_000.0,
                        cohort::SLOT_COMPUTE_NANOS[slot].load(Ordering::Relaxed) as f64 / 1_000_000.0,
                        cohort::SLOT_TAIL_NANOS[slot].load(Ordering::Relaxed) as f64 / 1_000_000.0,
                    );
                }
            }
            std::println!("DIAG q4k_shape_table rows k calls macs ns_per_mac");
            for (rows, k, calls, macs, ticks) in instrument::q4k_shape_snapshot() {
                let nanos = instrument::ticks_to_nanos(ticks);
                std::println!(
                    "DIAG q4k_shape rows={rows} k={k} calls={calls} macs={macs} ns_per_mac={:.6}",
                    nanos as f64 / macs.max(1) as f64,
                );
            }
        }

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

        // llama.cpp's own captured greedy answer for this exact prompt and
        // checkpoint (guiding-principle 14: the incumbent is the oracle).
        // Parallel row-chunking can reassociate float addition, so a small
        // numeric shift in the logits is expected, but the argmax must not
        // move -- if it does, the dispatch above changed the answer, not
        // just its speed.
        assert_eq!(token_id, 2651, "greedy token id drifted off llama.cpp's captured answer");
        assert_eq!(token_text, "known", "greedy token text drifted off llama.cpp's captured answer");
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
        let model_path = ServingConfig::default().model_path;
        let path = std::path::Path::new(model_path);
        if !path.exists() {
            eprintln!("skipping: no host-local openchat gguf fixture at {model_path}");
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
