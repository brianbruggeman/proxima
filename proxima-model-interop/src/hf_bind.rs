//! safetensors weights -> the same [`BoundWeights`] structures
//! [`crate::bind::bind_all_weights`] produces from a GGUF checkpoint --
//! HF-directory counterpart to that module, over the tensor manifest
//! [`proxima_safetensors::Manifest`] hands back instead of GGUF's
//! [`ParsedGguf`] tensor directory.
//!
//! Reuses `bind.rs`'s own decode/transpose/state machinery rather than
//! duplicating it: [`crate::bind::reinterpret_f32`], [`crate::bind::dequantize`],
//! [`crate::bind::aligned_f32_view`], [`crate::bind::transpose_out_in_to_in_out`],
//! and [`BoundWeights`] itself are all format-agnostic once handed raw bytes
//! and a declared dtype -- the only genuinely new work here is (a) mapping
//! HF's per-tensor names (`model.layers.0.self_attn.q_proj.weight`) to the
//! same weight slots GGUF's `blk.0.attn_q.weight` fills, and (b) decoding
//! [`DType::Float16`]/[`DType::BFloat16`] (safetensors' own on-disk element
//! types for an unquantized bf16/fp16 checkpoint), which `bind.rs`'s GGUF
//! path has never needed since no GGUF checkpoint this crate has evaluated
//! stores `F16`/`Bf16` weights.
//!
//! `std`-gated, matching `bind.rs`'s own `bind_all_weights`: both walk a
//! whole tensor directory and need [`proxima_gguf::restack`]-free but still
//! platform-shaped support (an owned `Vec<f32>` per non-packed weight, sized
//! by the checkpoint), and [`crate::generate::LoadedModel`] (the only
//! caller either bind function needs to serve) is itself `std`-only.
//!
//! Mixture-of-experts HF checkpoints are NOT bound here --
//! [`InteropError::HfMoeWeightsUnsupported`] names the gap rather than
//! guessing at Mixtral's or Qwen's per-expert tensor-naming convention
//! against zero real on-disk evidence (the only MoE-shaped checkpoint on
//! this host is MLX's packed `weight`/`scales`/`biases` layout, explicitly
//! out of scope for this crate).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_safetensors::Manifest;
use proxima_tensor::DType;
use proxima_tensor::cpu::QuantizedBlock;

use crate::bind::{
    BoundWeights, ModelArchitecture, aligned_f32_view, dequantize, reinterpret_f32,
    transpose_out_in_to_in_out,
};
use crate::error::InteropError;

fn find_entry<'manifest>(
    manifest: &'manifest Manifest,
    name: &str,
) -> Result<&'manifest proxima_safetensors::TensorEntry, InteropError> {
    manifest
        .tensor(name)
        .ok_or_else(|| InteropError::UnknownTensor { name: name.into() })
}

/// `entry.data_offsets` are relative to the first byte AFTER the header
/// (`Manifest`'s own doc, `proxima-safetensors/src/parser.rs`), never to
/// the start of the whole file -- `data_start` (`8 + header_len`, exactly
/// what a caller who just parsed `file_bytes`'s own header already knows,
/// see [`bind_all_weights_from_safetensors`]'s doc) is added here so every
/// caller of this module hands in the WHOLE file buffer, matching
/// `bind.rs`'s GGUF convention (`parsed.tensor_data_range`) instead of a
/// caller having to pre-slice the header off itself.
fn tensor_bytes<'file>(
    file_bytes: &'file [u8],
    data_start: u64,
    entry: &proxima_safetensors::TensorEntry,
) -> Result<&'file [u8], InteropError> {
    let start = data_start
        .checked_add(entry.data_offsets.0)
        .and_then(|value| usize::try_from(value).ok());
    let end = data_start
        .checked_add(entry.data_offsets.1)
        .and_then(|value| usize::try_from(value).ok());
    match (start, end) {
        (Some(start), Some(end)) => {
            file_bytes
                .get(start..end)
                .ok_or_else(|| InteropError::UnknownTensor {
                    name: format!(
                        "{} (byte range {start}..{end} outside a {}-byte buffer)",
                        entry.name,
                        file_bytes.len()
                    ),
                })
        }
        _ => Err(InteropError::UnknownTensor {
            name: format!(
                "{} (data_start {data_start} + declared offsets overflow usize)",
                entry.name
            ),
        }),
    }
}

/// Zero-copy counterpart to [`safetensors_tensor_as_f32`], mirroring
/// [`crate::bind::gguf_tensor_as_packed_block`]: borrows `name`'s raw bytes
/// straight out of `file_bytes` for [`DType::Float32`] (alignment-checked,
/// same reasoning as the GGUF path), [`DType::Float16`], or
/// [`DType::BFloat16`] -- safetensors never block-quantizes, so every
/// mapped dtype here is a flat per-element scalar array with no sub-block
/// scale to unpack, unlike GGUF's `Q4_K`/`Q5_K`/`Q6_K`.
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if `name` isn't in `manifest`'s tensor
/// directory, or its declared byte range doesn't fit `file_bytes`;
/// [`InteropError::UndecodableSafetensorsDType`] for any other `DType`
/// (an integer type, or a quantized layout's own packed payload type).
pub(crate) fn safetensors_tensor_as_packed_block<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    name: &str,
) -> Result<QuantizedBlock<'file>, InteropError> {
    let entry = find_entry(manifest, name)?;
    let bytes = tensor_bytes(file_bytes, data_start, entry)?;
    match entry.dtype {
        DType::Float32 => aligned_f32_view(bytes)
            .map(QuantizedBlock::Float32)
            .ok_or_else(|| InteropError::MisalignedFloat32Tensor {
                tensor: entry.name.clone(),
            }),
        DType::Float16 => Ok(QuantizedBlock::Float16(bytes)),
        DType::BFloat16 => Ok(QuantizedBlock::BFloat16(bytes)),
        other => Err(InteropError::UndecodableSafetensorsDType {
            tensor: entry.name.clone(),
            dtype: other,
        }),
    }
}

/// Owned-decode counterpart, mirroring [`crate::bind::gguf_tensor_as_f32`]:
/// reads `name`'s bytes and decodes to an owned `Vec<f32>` -- a straight
/// reinterpret for `Float32`, and [`proxima_gguf::quant::f16`]/
/// [`proxima_gguf::quant::bf16`]'s own `dequantize` (reused, not
/// reimplemented -- both already have exactly this crate's `fn(&[u8], &mut
/// [f32]) -> Result<(), QuantError>` shape) for `Float16`/`BFloat16`.
///
/// # Errors
///
/// Same as [`safetensors_tensor_as_packed_block`], plus
/// [`InteropError::Quant`] if a `Float16`/`BFloat16` tensor's byte length is
/// not a whole number of 2-byte elements.
pub(crate) fn safetensors_tensor_as_f32(
    manifest: &Manifest,
    file_bytes: &[u8],
    data_start: u64,
    name: &str,
) -> Result<Vec<f32>, InteropError> {
    let entry = find_entry(manifest, name)?;
    let bytes = tensor_bytes(file_bytes, data_start, entry)?;
    let element_count = entry.shape.iter().product::<u64>() as usize;
    match entry.dtype {
        DType::Float32 => Ok(reinterpret_f32(bytes)),
        DType::Float16 => dequantize(bytes, element_count, proxima_gguf::quant::f16::dequantize),
        DType::BFloat16 => dequantize(bytes, element_count, proxima_gguf::quant::bf16::dequantize),
        other => Err(InteropError::UndecodableSafetensorsDType {
            tensor: entry.name.clone(),
            dtype: other,
        }),
    }
}

/// A learned 1-D scale (RMSNorm weight) or the embedding table (indexed by
/// row via `embedding_lookup`, never projected) -- same role as
/// [`crate::bind::bind_dense`], reused under a new name only because its
/// GGUF counterpart is `pub(crate)` to `bind.rs`'s own module, not this one.
///
/// `lookup_name` is the name HF's own manifest carries on disk
/// (`model.layers.0.input_layernorm.weight`); `store_name` is the name
/// [`BoundWeights`] keys this tensor under -- the forward program's own
/// GGUF-convention node name (`blk.0.attn_norm.weight`), since
/// [`crate::generate::LoadedModel::call`] matches every bound weight against
/// the compiled program purely by that string (`generate.rs`'s
/// `named_blocks.push((name.as_str(), ..))` loop), never by position. The
/// two names differ for every weight this module binds; kept as two
/// parameters rather than reusing one, so a caller cannot silently store a
/// tensor under the very on-disk name the program will never ask for.
///
/// Only `Float32` binds packed/zero-copy here -- a packed `Float16`/
/// `BFloat16` block lands in the interpreter's `quantized_weights` map
/// (`proxima_tensor::cpu`'s own `block_nodes`/`blocks` split), read ONLY by
/// the quantized-matmul reduce kernel (`run_reduce_quantized`); a gather
/// (`embedding_lookup`) or a plain elementwise scale (RMSNorm) instead reads
/// the plain per-node `buffers` array via `buffer_of`, which never looks
/// there at all. Confirmed the hard way against the real, downloaded
/// `HuggingFaceTB/SmolLM2-135M-Instruct/model.safetensors` (BF16 weights
/// throughout): binding `token_embd.weight` packed here produced
/// `NotLowerable { reason: "operand buffer missing at evaluation time" }`
/// the first time this path ever ran against a real bf16 HF checkpoint --
/// this is that fix. [`hf_bind_matmul_weight`] is the one binder that feeds
/// a genuine 2-D matmul operand, where the packed `Float16`/`BFloat16` path
/// (`matmul_f16_f32`/`matmul_bf16_f32`) is correct and stays.
fn hf_bind_dense<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    lookup_name: String,
    store_name: String,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    match safetensors_tensor_as_packed_block(manifest, file_bytes, data_start, &lookup_name) {
        Ok(block @ QuantizedBlock::Float32(borrowed)) => {
            state.resident_bytes += core::mem::size_of_val(borrowed);
            state.packed.push((store_name, block));
        }
        Ok(_) | Err(_) => {
            let decoded =
                safetensors_tensor_as_f32(manifest, file_bytes, data_start, &lookup_name)?;
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            state.owned.push((store_name, decoded));
        }
    }
    Ok(())
}

/// Reorders `flat`'s (`[out_dim, in_dim]` row-major, `out_dim ==
/// head_count * head_dim`) rows from HF's own RoPE layout into the
/// interleaved layout [`proxima_tensor::spec`]'s RoPE math (`spec.rs`'s
/// `q_even`/`q_odd`, indexed `"2*i"`/`"2*i+1"`) consumes -- the same
/// per-head row permutation `llama.cpp`'s own `convert_hf_to_gguf.py`
/// applies (its `permute()`) before writing `blk.N.attn_q.weight`/
/// `blk.N.attn_k.weight` into a GGUF file, which is why [`crate::bind`]'s
/// GGUF path has never needed this: a GGUF checkpoint's bytes already
/// carry it.
///
/// HF/PyTorch's own rotary embedding (`transformers`' `rotate_half`)
/// treats a head's `head_dim` channels as two contiguous halves, `x1 =
/// x[..half]`, `x2 = x[half..]`, and rotates each `(x1[i], x2[i])` pair.
/// `spec.rs`'s RoPE instead rotates adjacent-channel pairs `(x[2*i],
/// x[2*i+1])` -- mathematically the same rotation, over a differently
/// numbered set of pairs, so it needs `x1[i]`/`x2[i]` delivered at channels
/// `2*i`/`2*i+1` instead of at `i`/`i+head_dim/2`. Per head:
///
/// ```text
/// permuted[head*head_dim + 2*i]     = original[head*head_dim + i]
/// permuted[head*head_dim + 2*i + 1] = original[head*head_dim + head_dim/2 + i]
/// ```
///
/// Confirmed against the real, downloaded `HuggingFaceTB/SmolLM2-135M-Instruct/model.safetensors`:
/// without this permutation, this crate's own forward pass matches
/// `llama.cpp`'s bit-for-bit at RoPE position 0 (where every pair's angle
/// is zero, so no permutation convention can matter) and diverges starting
/// at position 1 already inside layer 0 -- exactly the position dependence
/// this permutation, and only this permutation, explains.
fn permute_rope_rows(flat: &[f32], head_count: usize, head_dim: usize, in_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let mut permuted = alloc::vec![0.0f32; flat.len()];
    for head in 0..head_count {
        for i in 0..half {
            let source_even = (head * head_dim + i) * in_dim;
            let source_odd = (head * head_dim + half + i) * in_dim;
            let dest_even = (head * head_dim + 2 * i) * in_dim;
            let dest_odd = (head * head_dim + 2 * i + 1) * in_dim;
            permuted[dest_even..dest_even + in_dim]
                .copy_from_slice(&flat[source_even..source_even + in_dim]);
            permuted[dest_odd..dest_odd + in_dim]
                .copy_from_slice(&flat[source_odd..source_odd + in_dim]);
        }
    }
    permuted
}

/// [`hf_bind_matmul_weight`]'s counterpart for `q_proj`/`k_proj` alone:
/// always decodes to an owned `[out_dim, in_dim]` buffer (never takes the
/// packed zero-copy path -- [`permute_rope_rows`]'s reorder is a physical
/// byte move a borrowed packed block cannot express) and applies
/// [`permute_rope_rows`] before the same transpose
/// [`hf_bind_matmul_weight`] already does.
// one weight's own real shape (lookup/store names, head geometry, in_dim)
// plus the file/state every binder in this module threads through -- the
// same shape `proxima_tensor::spec::append_mistral_cached_moe_layer`
// carries its own `#[allow(clippy::too_many_arguments)]` for.
#[allow(clippy::too_many_arguments)]
fn hf_bind_rope_weight<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    lookup_name: String,
    store_name: String,
    head_count: usize,
    head_dim: usize,
    in_dim: usize,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let decoded = safetensors_tensor_as_f32(manifest, file_bytes, data_start, &lookup_name)?;
    let permuted = permute_rope_rows(&decoded, head_count, head_dim, in_dim);
    let out_dim = head_count * head_dim;
    state.resident_bytes += permuted.len() * core::mem::size_of::<f32>();
    state.owned.push((
        store_name,
        transpose_out_in_to_in_out(&permuted, &lookup_name, out_dim, in_dim)?,
    ));
    Ok(())
}

/// A 2-D projection weight bound as one matmul operand -- HF/PyTorch's
/// `nn.Linear` weight is `[out_features, in_features]`, row-major, the
/// identical on-disk convention GGUF's own `[out, in]` layout uses (see
/// [`crate::bind::transpose_out_in_to_in_out`]'s doc), so the same
/// dequantize-then-transpose fallback applies unchanged. Packed
/// `Float16`/`BFloat16` binds straight through zero-copy: [`QuantizedBlock`]'s
/// own matmul kernel (`proxima_tensor::cpu::matmul_f16_f32`/`matmul_bf16_f32`)
/// walks the packed bytes in their native `[out, in]` order directly, the
/// same way a packed `Q4_K` operand does, so no transpose is needed on that
/// path at all.
///
/// `Float32` is the one codec [`safetensors_tensor_as_packed_block`] can
/// decode but this function does NOT hand to the packed path, mirroring
/// [`crate::bind::bind_matmul_weight_as`]'s own fix: neither the CPU generic
/// evaluator (`proxima_tensor::cpu::evaluate_quantized_with_scratch`'s
/// `QuantizedBlock::Float32` arm binds as a plain buffer, never entering
/// `run_reduce_quantized`'s byte-native bypass) nor the GPU driver
/// (`omega::metal.rs`'s own `packed_operands_of` explicitly excludes
/// `QuantizedBlock::Float32` from `correct_packed_matmul_layouts`'s node set)
/// has any mechanism that rewrites a packed `F32` operand's layout from
/// safetensors' native `[out, in]` to the `[in, out]` every consuming
/// `IndexMap` declares. So `Float32` always takes the dequantize-then-transpose
/// path instead, the same as any other undecodable-packed dtype -- only the
/// OWNED f32 fallback needs the transpose, exactly mirroring
/// [`crate::bind::bind_matmul_weight`]/[`crate::bind::bind_matmul_weight_as`].
///
/// `lookup_name`/`store_name` split for the same reason [`hf_bind_dense`]'s
/// doc explains -- and this split is also what lets a tied-embedding
/// checkpoint bind its `output.weight` program node from
/// `model.embed_tokens.weight`'s on-disk bytes (`lookup_name`) while still
/// storing the result under `output.weight` (`store_name`): the identical
/// transpose this function already applies to any other `[out, in]`
/// projection weight, no new bind path.
// two names (lookup/store, see this doc) plus the tensor's own shape
// (out_dim/in_dim) plus the file/state every binder in this module threads
// through -- the same real parameter count `bind_moe_expert_weights`
// (`bind.rs`) carries for the identical reason.
#[allow(clippy::too_many_arguments)]
fn hf_bind_matmul_weight<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    lookup_name: String,
    store_name: String,
    out_dim: usize,
    in_dim: usize,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    match safetensors_tensor_as_packed_block(manifest, file_bytes, data_start, &lookup_name) {
        Ok(QuantizedBlock::Float32(_)) | Err(_) => {
            let decoded =
                safetensors_tensor_as_f32(manifest, file_bytes, data_start, &lookup_name)?;
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            state.owned.push((
                store_name,
                transpose_out_in_to_in_out(&decoded, &lookup_name, out_dim, in_dim)?,
            ));
        }
        Ok(block) => state.packed.push((store_name, block)),
    }
    Ok(())
}

/// Every dense-checkpoint weight name [`crate::bind::bind_all_weights`]'s
/// GGUF loop binds, HF's own naming instead -- the standard Llama/Mistral/
/// Qwen `transformers` layout (`model.layers.{layer}.*`), the convention
/// every dense checkpoint on HuggingFace this crate has been checked against
/// uses.
mod names {
    use alloc::format;
    use alloc::string::String;

    pub(super) fn embed_tokens() -> String {
        "model.embed_tokens.weight".into()
    }
    pub(super) fn input_layernorm(layer: u32) -> String {
        format!("model.layers.{layer}.input_layernorm.weight")
    }
    pub(super) fn post_attention_layernorm(layer: u32) -> String {
        format!("model.layers.{layer}.post_attention_layernorm.weight")
    }
    pub(super) fn q_proj(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.q_proj.weight")
    }
    pub(super) fn k_proj(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.k_proj.weight")
    }
    pub(super) fn v_proj(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.v_proj.weight")
    }
    pub(super) fn o_proj(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.o_proj.weight")
    }
    pub(super) fn gate_proj(layer: u32) -> String {
        format!("model.layers.{layer}.mlp.gate_proj.weight")
    }
    pub(super) fn up_proj(layer: u32) -> String {
        format!("model.layers.{layer}.mlp.up_proj.weight")
    }
    pub(super) fn down_proj(layer: u32) -> String {
        format!("model.layers.{layer}.mlp.down_proj.weight")
    }
    pub(super) fn final_norm() -> String {
        "model.norm.weight".into()
    }
    pub(super) fn lm_head() -> String {
        "lm_head.weight".into()
    }
}

/// The forward program's own node names -- exactly the GGUF-convention
/// strings [`proxima_tensor::spec::mistral_cached_forward_program_with_experts`]
/// builds its `Op::Input` leaves with (`spec.rs`'s own `blk.{layer}.*`/
/// `token_embd.weight`/`output_norm.weight`/`output.weight` literals), never
/// [`names`]'s HF-convention strings. [`crate::generate::LoadedModel::call`]
/// matches every bound weight against the compiled program purely by name
/// (see [`hf_bind_dense`]'s doc), so every [`BoundWeights`] entry this module
/// produces MUST be stored under one of these, regardless of what the
/// on-disk tensor was called.
mod node_names {
    use alloc::format;
    use alloc::string::String;

    pub(super) fn token_embd() -> String {
        "token_embd.weight".into()
    }
    pub(super) fn attn_norm(layer: u32) -> String {
        format!("blk.{layer}.attn_norm.weight")
    }
    pub(super) fn ffn_norm(layer: u32) -> String {
        format!("blk.{layer}.ffn_norm.weight")
    }
    pub(super) fn attn_q(layer: u32) -> String {
        format!("blk.{layer}.attn_q.weight")
    }
    pub(super) fn attn_k(layer: u32) -> String {
        format!("blk.{layer}.attn_k.weight")
    }
    pub(super) fn attn_v(layer: u32) -> String {
        format!("blk.{layer}.attn_v.weight")
    }
    pub(super) fn attn_output(layer: u32) -> String {
        format!("blk.{layer}.attn_output.weight")
    }
    pub(super) fn ffn_gate(layer: u32) -> String {
        format!("blk.{layer}.ffn_gate.weight")
    }
    pub(super) fn ffn_up(layer: u32) -> String {
        format!("blk.{layer}.ffn_up.weight")
    }
    pub(super) fn ffn_down(layer: u32) -> String {
        format!("blk.{layer}.ffn_down.weight")
    }
    pub(super) fn output_norm() -> String {
        "output_norm.weight".into()
    }
    pub(super) fn output_weight() -> String {
        "output.weight".into()
    }
}

/// HF/safetensors counterpart to [`crate::bind::bind_all_weights`]: binds
/// every weight [`proxima_tensor::spec::mistral_cached_forward_program`]
/// needs out of a single safetensors buffer's [`Manifest`], using HF's own
/// per-tensor names ([`names`]) in place of GGUF's `blk.{n}.*` convention.
///
/// `file_bytes` is the WHOLE safetensors buffer (matching `bind.rs`'s GGUF
/// convention), and `data_start` is the byte offset where tensor data
/// begins -- `8 + header_len`, exactly what a caller already computed while
/// parsing `file_bytes`'s own header into `manifest` (the same value
/// `bf16_real_checkpoint_parity.rs`'s own `real_manifest` helper derives by
/// hand for the identical reason: [`Manifest`]/[`proxima_safetensors::TensorEntry`]'s
/// own `data_offsets` are relative to the byte AFTER the header, never to
/// the start of the file, per that type's own doc).
///
/// Single-shard only: a real multi-file HF checkpoint (this crate's own
/// evidence, `~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit/`,
/// ships four `model-0000N-of-00004.safetensors` shards plus a
/// `model.safetensors.index.json` mapping each tensor name to its shard) is
/// NOT assembled here -- a caller with several shards' `(Manifest,
/// file_bytes, data_start)` triples calls this once per shard it owns and
/// merges the resulting [`BoundWeights`], or (unimplemented) this crate
/// grows an index-aware entry point that resolves a name to its shard
/// first. That index-following step is the concrete piece still missing
/// between this function and a full multi-shard HF directory load.
///
/// # Errors
///
/// [`InteropError::HfMoeWeightsUnsupported`] if `architecture.expert_count`
/// is nonzero; otherwise whatever [`hf_bind_dense`]/[`hf_bind_matmul_weight`]
/// can fail with (see [`safetensors_tensor_as_f32`]/
/// [`safetensors_tensor_as_packed_block`]).
pub(crate) fn bind_all_weights_from_safetensors<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    architecture: &ModelArchitecture,
) -> Result<BoundWeights<'file>, InteropError> {
    if architecture.expert_count != 0 {
        return Err(InteropError::HfMoeWeightsUnsupported {
            expert_count: architecture.expert_count,
        });
    }

    let mut state = BoundWeights {
        resident_bytes: file_bytes.len(),
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
    };

    let embedding = architecture.embedding as usize;
    let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
    let feed_forward = architecture.feed_forward as usize;
    let vocab = architecture.vocab as usize;

    hf_bind_dense(
        manifest,
        file_bytes,
        data_start,
        names::embed_tokens(),
        node_names::token_embd(),
        &mut state,
    )?;

    for layer in 0..architecture.block_count {
        hf_bind_dense(
            manifest,
            file_bytes,
            data_start,
            names::input_layernorm(layer),
            node_names::attn_norm(layer),
            &mut state,
        )?;
        hf_bind_dense(
            manifest,
            file_bytes,
            data_start,
            names::post_attention_layernorm(layer),
            node_names::ffn_norm(layer),
            &mut state,
        )?;
        hf_bind_rope_weight(
            manifest,
            file_bytes,
            data_start,
            names::q_proj(layer),
            node_names::attn_q(layer),
            architecture.query_heads as usize,
            architecture.head_dim as usize,
            embedding,
            &mut state,
        )?;
        hf_bind_rope_weight(
            manifest,
            file_bytes,
            data_start,
            names::k_proj(layer),
            node_names::attn_k(layer),
            architecture.kv_heads as usize,
            architecture.head_dim as usize,
            embedding,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::v_proj(layer),
            node_names::attn_v(layer),
            kv_dim,
            embedding,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::o_proj(layer),
            node_names::attn_output(layer),
            embedding,
            embedding,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::gate_proj(layer),
            node_names::ffn_gate(layer),
            feed_forward,
            embedding,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::up_proj(layer),
            node_names::ffn_up(layer),
            feed_forward,
            embedding,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::down_proj(layer),
            node_names::ffn_down(layer),
            embedding,
            feed_forward,
            &mut state,
        )?;
    }

    hf_bind_dense(
        manifest,
        file_bytes,
        data_start,
        names::final_norm(),
        node_names::output_norm(),
        &mut state,
    )?;

    // A tied checkpoint (HF's `tie_word_embeddings: true`, e.g. real
    // SmolLM2-135M-Instruct) ships no `lm_head.weight` tensor at all -- its
    // token-embedding table doubles as the LM head. Binding the SAME
    // on-disk tensor at the `output.weight` program node needs no new bind
    // path: `hf_bind_matmul_weight` already dequantizes-then-transposes (or
    // binds packed) any `[out, in]`-shaped weight, and `embed_tokens`'s own
    // on-disk shape (`[vocab, embedding]`) is exactly that for
    // `out_dim = vocab, in_dim = embedding` -- the same shape `lm_head.weight`
    // would have carried on an untied checkpoint.
    let output_lookup_name = if architecture.tied_embeddings {
        names::embed_tokens()
    } else {
        names::lm_head()
    };
    hf_bind_matmul_weight(
        manifest,
        file_bytes,
        data_start,
        output_lookup_name,
        node_names::output_weight(),
        vocab,
        embedding,
        &mut state,
    )?;
    Ok(state)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use proxima_safetensors::{SafetensorsModel, TensorPayload, write_complete};

    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    /// `8 + header_len`, read off a real written safetensors buffer's own
    /// 8-byte little-endian length prefix -- the `data_start` every
    /// `bind_all_weights_from_safetensors` call in this module needs, since
    /// `Manifest`'s own `data_offsets` are relative to this point (see
    /// `bind_all_weights_from_safetensors`'s own doc), not to byte 0.
    fn header_data_start(file_bytes: &[u8]) -> u64 {
        let mut length_prefix = [0u8; 8];
        length_prefix.copy_from_slice(&file_bytes[..8]);
        8 + u64::from_le_bytes(length_prefix)
    }

    /// The smallest real dense architecture this crate's weight names can
    /// bind: 1 layer, embedding=4, feed_forward=8, 2 query heads, 1 kv head
    /// (GQA), head_dim=2, vocab=3 -- every dimension distinct so a
    /// transposed or mis-shaped bind would produce a length mismatch, not
    /// silently pass.
    fn tiny_dense_architecture() -> ModelArchitecture {
        ModelArchitecture {
            vocab: 3,
            embedding: 4,
            feed_forward: 8,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            block_count: 1,
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
            tied_embeddings: false,
        }
    }

    /// Builds a real (in-memory-encoded, byte-for-byte real safetensors wire
    /// format via [`write_complete`]) checkpoint carrying every weight name
    /// [`bind_all_weights_from_safetensors`] looks up for
    /// [`tiny_dense_architecture`], each an `F32` tensor of the exact shape
    /// the architecture implies.
    fn tiny_dense_checkpoint() -> Vec<u8> {
        let architecture = tiny_dense_architecture();
        let embedding = architecture.embedding as usize;
        let feed_forward = architecture.feed_forward as usize;
        let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
        let vocab = architecture.vocab as usize;

        let mut owned_data: Vec<(String, Vec<u8>)> = Vec::new();
        let mut push = |name: String, elements: usize| {
            let values: Vec<f32> = (0..elements).map(|index| index as f32 * 0.5).collect();
            owned_data.push((name, f32_bytes(&values)));
        };

        push(names::embed_tokens(), vocab * embedding);
        push(names::input_layernorm(0), embedding);
        push(names::post_attention_layernorm(0), embedding);
        push(names::q_proj(0), embedding * embedding);
        push(names::k_proj(0), kv_dim * embedding);
        push(names::v_proj(0), kv_dim * embedding);
        push(names::o_proj(0), embedding * embedding);
        push(names::gate_proj(0), feed_forward * embedding);
        push(names::up_proj(0), feed_forward * embedding);
        push(names::down_proj(0), embedding * feed_forward);
        push(names::final_norm(), embedding);
        push(names::lm_head(), vocab * embedding);

        let tensors = owned_data
            .iter()
            .map(|(name, bytes)| {
                let elements = bytes.len() / 4;
                TensorPayload {
                    name: name.clone(),
                    dtype: DType::Float32,
                    shape: vec![elements as u64],
                    data: bytes.as_slice(),
                }
            })
            .collect();

        write_complete(&SafetensorsModel {
            tensors,
            metadata: alloc::collections::BTreeMap::new(),
        })
        .expect("writes a real safetensors buffer")
    }

    /// Every weight name [`bind_all_weights_from_safetensors`]'s loop looks
    /// up is present and binds without error, over a real (wire-format
    /// encoded) safetensors buffer -- proves the HF name mapping matches
    /// what a real dense checkpoint's own tensor directory would carry, not
    /// just that the function compiles.
    #[test]
    fn binds_every_dense_weight_name_from_a_real_safetensors_buffer() {
        let file_bytes = tiny_dense_checkpoint();
        let manifest = proxima_safetensors::parse_complete(&file_bytes)
            .expect("parses real safetensors buffer");
        let architecture = tiny_dense_architecture();
        let data_start = header_data_start(&file_bytes);

        let bound =
            bind_all_weights_from_safetensors(&manifest, &file_bytes, data_start, &architecture)
                .expect("binds every dense weight this architecture's forward program needs");

        // 12 tensors total: embed_tokens, 2 norms, 4 attention projections,
        // 3 ffn projections, final_norm, lm_head -- every one bound as an
        // owned f32 buffer here since none of this fixture's tensors are
        // 4-byte-misaligned within the written buffer.
        assert_eq!(
            bound.owned.len() + bound.packed.len(),
            12,
            "every named weight must bind, none silently skipped"
        );

        // Every stored key must be the forward program's own GGUF-convention
        // node name, NEVER the HF on-disk name this checkpoint's manifest
        // actually carries -- `LoadedModel::call` matches bound weights
        // against the compiled program purely by string (`generate.rs`'s
        // `named_blocks` loop), so a name mismatch here would bind silently
        // and then fail (or worse, mis-match) only once a real forward pass
        // runs.
        let stored_names: alloc::vec::Vec<&str> = bound
            .owned
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(bound.packed.iter().map(|(name, _)| name.as_str()))
            .collect();
        for expected in [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "output_norm.weight",
            "output.weight",
        ] {
            assert!(
                stored_names.contains(&expected),
                "expected the GGUF-convention node name {expected:?} among stored keys {stored_names:?}"
            );
        }
        for hf_name in [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            assert!(
                !stored_names.contains(&hf_name),
                "the HF on-disk name {hf_name:?} must never be the STORED key -- only the lookup key"
            );
        }
    }

    /// A real bf16 embedding table must bind as an OWNED f32 buffer, never
    /// packed -- the exact defect this row's own fix closes, reproduced
    /// directly rather than only through the full multi-tensor pipeline:
    /// binding `token_embd.weight` packed would land it in the
    /// interpreter's `quantized_weights` map, which `embedding_lookup`'s
    /// gather (`proxima_tensor::cpu::buffer_of`) never reads at all, so a
    /// packed bind here is a silent setup for `NotLowerable { reason:
    /// "operand buffer missing at evaluation time" }` at forward time
    /// rather than a bind-time failure -- this test catches it at bind time
    /// instead.
    #[test]
    fn bf16_embed_tokens_binds_as_owned_f32_not_packed() {
        let architecture = tiny_dense_architecture();
        let vocab = architecture.vocab as usize;
        let embedding = architecture.embedding as usize;
        let values: Vec<half::bf16> = (0..vocab * embedding)
            .map(|index| half::bf16::from_f32(index as f32 * 0.5))
            .collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let tensors = vec![TensorPayload {
            name: names::embed_tokens(),
            dtype: DType::BFloat16,
            shape: vec![vocab as u64, embedding as u64],
            data: bytes.as_slice(),
        }];
        let file_bytes = write_complete(&SafetensorsModel {
            tensors,
            metadata: alloc::collections::BTreeMap::new(),
        })
        .expect("writes a real bf16 safetensors buffer");
        let manifest = proxima_safetensors::parse_complete(&file_bytes)
            .expect("parses real safetensors buffer");
        let data_start = header_data_start(&file_bytes);

        let mut state = BoundWeights {
            resident_bytes: 0,
            owned: Vec::new(),
            packed: Vec::new(),
            packed_owned: Vec::new(),
        };
        hf_bind_dense(
            &manifest,
            &file_bytes,
            data_start,
            names::embed_tokens(),
            node_names::token_embd(),
            &mut state,
        )
        .expect("binds the real bf16 embedding table");

        assert_eq!(
            state.owned.len(),
            1,
            "the bf16 embedding table must decode to an OWNED f32 buffer"
        );
        assert_eq!(
            state.packed.len(),
            0,
            "the bf16 embedding table must NEVER bind packed -- gather cannot read it there"
        );
        assert_eq!(state.owned[0].0, "token_embd.weight");
        assert_eq!(state.owned[0].1.len(), vocab * embedding);
        // real-value check, not just shape: the first decoded element must
        // match an independent bf16 decode of the same bytes.
        assert!((state.owned[0].1[0] - 0.0).abs() < 1e-6);
        assert!((state.owned[0].1[1] - 0.5).abs() < 1e-6);
    }

    /// A tied checkpoint (real SmolLM2-135M-Instruct's own shape: no
    /// `lm_head.weight` tensor at all) must still bind an `output.weight`
    /// program node, reusing `embed_tokens`'s own on-disk bytes -- proves
    /// [`bind_all_weights_from_safetensors`]'s tied-embedding branch, not
    /// just that the untied path works.
    #[test]
    fn tied_embeddings_bind_output_weight_from_embed_tokens_with_no_lm_head_tensor() {
        let mut architecture = tiny_dense_architecture();
        architecture.tied_embeddings = true;
        let embedding = architecture.embedding as usize;
        let feed_forward = architecture.feed_forward as usize;
        let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
        let vocab = architecture.vocab as usize;

        let mut owned_data: Vec<(String, Vec<u8>)> = Vec::new();
        let mut push = |name: String, elements: usize| {
            let values: Vec<f32> = (0..elements).map(|index| index as f32 * 0.5).collect();
            owned_data.push((name, f32_bytes(&values)));
        };
        push(names::embed_tokens(), vocab * embedding);
        push(names::input_layernorm(0), embedding);
        push(names::post_attention_layernorm(0), embedding);
        push(names::q_proj(0), embedding * embedding);
        push(names::k_proj(0), kv_dim * embedding);
        push(names::v_proj(0), kv_dim * embedding);
        push(names::o_proj(0), embedding * embedding);
        push(names::gate_proj(0), feed_forward * embedding);
        push(names::up_proj(0), feed_forward * embedding);
        push(names::down_proj(0), embedding * feed_forward);
        push(names::final_norm(), embedding);
        // deliberately NO names::lm_head() tensor -- this is the whole point

        let tensors = owned_data
            .iter()
            .map(|(name, bytes)| TensorPayload {
                name: name.clone(),
                dtype: DType::Float32,
                shape: vec![(bytes.len() / 4) as u64],
                data: bytes.as_slice(),
            })
            .collect();
        let file_bytes = write_complete(&SafetensorsModel {
            tensors,
            metadata: alloc::collections::BTreeMap::new(),
        })
        .expect("writes a real safetensors buffer with no lm_head.weight tensor");
        let manifest = proxima_safetensors::parse_complete(&file_bytes)
            .expect("parses real safetensors buffer");
        let data_start = header_data_start(&file_bytes);

        let bound = bind_all_weights_from_safetensors(&manifest, &file_bytes, data_start, &architecture)
            .expect("a tied checkpoint must bind output.weight from embed_tokens, not error looking for lm_head.weight");

        let stored_names: alloc::vec::Vec<&str> = bound
            .owned
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(bound.packed.iter().map(|(name, _)| name.as_str()))
            .collect();
        assert!(
            stored_names.contains(&"output.weight"),
            "output.weight must still be bound: {stored_names:?}"
        );
        // 12 STORED entries, same as the untied fixture: 11 on-disk tensors,
        // but `embed_tokens` is read TWICE -- once for `token_embd.weight`,
        // once for `output.weight` -- so the program still gets 12 named
        // inputs even though the checkpoint carries one fewer on-disk tensor.
        assert_eq!(
            bound.owned.len() + bound.packed.len(),
            12,
            "embed_tokens binds twice: once as token_embd.weight, once as output.weight"
        );
        assert_eq!(
            manifest.tensors.len(),
            11,
            "the checkpoint fixture itself carries 11 on-disk tensors, one fewer than the untied fixture's 12, since lm_head.weight is absent"
        );
    }

    /// The one weight this test perturbs by transposing it wrong would prove
    /// [`transpose_out_in_to_in_out`] is actually being applied: corrupt
    /// `q_proj`'s declared shape so it disagrees with `out_dim*in_dim`, and
    /// the typed shape-mismatch error must fire, not a panic mid-transpose.
    /// Forces the OWNED decode-then-transpose fallback (not the zero-copy
    /// packed path, which never validates a shape against `out_dim`/`in_dim`
    /// at all -- see [`hf_bind_matmul_weight`]'s own doc) by prepending a
    /// dummy leading tensor whose byte length shifts `q_proj`'s absolute
    /// file offset off a 4-byte boundary: safetensors writes tensors
    /// back-to-back with zero padding between them (`writer.rs`'s own doc),
    /// so the padding length is fully under this test's control -- tried
    /// 0..4 until the real, measured offset actually lands on an unaligned
    /// byte, rather than assumed.
    #[test]
    fn shape_mismatched_weight_is_a_typed_error_not_a_panic() {
        let architecture = tiny_dense_architecture();
        let embedding = architecture.embedding as usize;
        let bad_q_proj = f32_bytes(&vec![0.0f32; embedding * embedding - 1]);

        for padding in 0..4u64 {
            let dummy_bytes = vec![0u8; padding as usize];
            let tensors = vec![
                TensorPayload {
                    name: "dummy_padding".into(),
                    dtype: DType::Int8,
                    shape: vec![padding],
                    data: dummy_bytes.as_slice(),
                },
                TensorPayload {
                    name: names::q_proj(0),
                    dtype: DType::Float32,
                    shape: vec![(embedding * embedding - 1) as u64],
                    data: bad_q_proj.as_slice(),
                },
            ];
            let file_bytes = write_complete(&SafetensorsModel {
                tensors,
                metadata: alloc::collections::BTreeMap::new(),
            })
            .expect("writes a real safetensors buffer");
            let manifest = proxima_safetensors::parse_complete(&file_bytes)
                .expect("parses real safetensors buffer");
            let entry = manifest
                .tensor(&names::q_proj(0))
                .expect("q_proj entry present");
            let data_start =
                8 + u64::from_le_bytes(file_bytes[..8].try_into().expect("8-byte length prefix"));
            let absolute_offset = data_start + entry.data_offsets.0;
            if absolute_offset.is_multiple_of(4) {
                continue; // still aligned with this padding length; try the next one
            }

            let outcome = hf_bind_matmul_weight(
                &manifest,
                &file_bytes,
                data_start,
                names::q_proj(0),
                node_names::attn_q(0),
                embedding,
                embedding,
                &mut BoundWeights {
                    resident_bytes: 0,
                    owned: Vec::new(),
                    packed: Vec::new(),
                    packed_owned: Vec::new(),
                },
            );

            assert!(
                matches!(outcome, Err(InteropError::DenseWeightShapeMismatch { .. })),
                "a short decoded buffer reached through the misaligned owned-decode fallback must \
                 surface as a typed error, got {outcome:?}"
            );
            return;
        }
        panic!(
            "could not construct a misaligned f32 tensor offset with padding in 0..4 -- test setup bug"
        );
    }

    /// The decisive proof for this file's own fix: a raw-packed `F32`
    /// matmul weight bound through [`hf_bind_matmul_weight`] must land in
    /// [`BoundWeights::owned`], transposed, and produce the exact same
    /// matmul result as an independent hand computation over the tensor's
    /// own on-disk bytes -- run through the real
    /// [`proxima_tensor::cpu::evaluate_quantized_named`] evaluation a forward
    /// program actually uses, not merely compared byte-for-byte against the
    /// dequantize-then-transpose path. Mirrors `bind.rs`'s own
    /// `raw_packed_f32_matmul_weight_matches_an_independent_hand_computed_matmul`
    /// against safetensors' wire format instead of GGUF's.
    ///
    /// `out_dim=4`/`in_dim=6` are deliberately asymmetric: a transpose of a
    /// square or single-row buffer can coincidentally read back correctly,
    /// or silently permute symmetric data, proving nothing. With `out_dim !=
    /// in_dim`, a buffer addressed through the wrong axis order reads
    /// flatly wrong values.
    ///
    /// A single-tensor safetensors buffer's data offset is NOT guaranteed
    /// 4-byte aligned (safetensors pads nothing between tensors, unlike
    /// GGUF), so [`safetensors_tensor_as_packed_block`]'s own alignment
    /// check on `Float32` can reject this tensor for reasons that have
    /// nothing to do with this fix, silently forcing the safe owned path
    /// either way and making the test vacuous. A dummy leading `Int8`
    /// tensor whose byte length shifts the target's absolute offset (same
    /// technique [`shape_mismatched_weight_is_a_typed_error_not_a_panic`]
    /// uses in reverse) is prepended, tried over `0..4` bytes of padding
    /// until the offset actually lands 4-byte ALIGNED, so this test
    /// genuinely exercises the packed-vs-owned dispatch rather than an
    /// accidental fallback.
    #[test]
    fn raw_packed_f32_matmul_weight_matches_an_independent_hand_computed_matmul() {
        let out_dim = 4usize;
        let in_dim = 6usize;
        // safetensors' own on-disk convention (identical to GGUF's): `out_dim`
        // rows, each a contiguous run of `in_dim` elements -- weight(out, in)
        // = out*10 + in, distinct per element so a scrambled read is
        // detectable.
        let mut on_disk = vec![0.0f32; out_dim * in_dim];
        for out_index in 0..out_dim {
            for in_index in 0..in_dim {
                on_disk[out_index * in_dim + in_index] = (out_index * 10 + in_index) as f32;
            }
        }
        let bytes = f32_bytes(&on_disk);

        let (file_bytes, manifest, data_start) = (0..4u64)
            .find_map(|padding| {
                let dummy_bytes = vec![0u8; padding as usize];
                let tensors = vec![
                    TensorPayload {
                        name: "dummy_padding".into(),
                        dtype: DType::Int8,
                        shape: vec![padding],
                        data: dummy_bytes.as_slice(),
                    },
                    TensorPayload {
                        name: "model.layers.0.mlp.gate_proj.weight".into(),
                        dtype: DType::Float32,
                        shape: vec![out_dim as u64, in_dim as u64],
                        data: bytes.as_slice(),
                    },
                ];
                let file_bytes = write_complete(&SafetensorsModel {
                    tensors,
                    metadata: alloc::collections::BTreeMap::new(),
                })
                .expect("writes a real safetensors buffer with an asymmetric f32 matmul weight");
                let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses real safetensors buffer");
                let data_start = header_data_start(&file_bytes);
                let entry = manifest.tensor("model.layers.0.mlp.gate_proj.weight").expect("gate_proj entry present");
                let absolute_offset = data_start + entry.data_offsets.0;
                absolute_offset.is_multiple_of(4).then_some((file_bytes, manifest, data_start))
            })
            .expect("could not construct a 4-byte-aligned f32 tensor offset with padding in 0..4 -- test setup bug");

        let mut state = BoundWeights {
            resident_bytes: 0,
            owned: Vec::new(),
            packed: Vec::new(),
            packed_owned: Vec::new(),
        };
        hf_bind_matmul_weight(
            &manifest,
            &file_bytes,
            data_start,
            "model.layers.0.mlp.gate_proj.weight".into(),
            "gate".into(),
            out_dim,
            in_dim,
            &mut state,
        )
        .expect("binds the asymmetric f32 matmul weight");
        assert!(
            state.packed.is_empty(),
            "an F32 matmul weight must never take the raw-packed path -- nothing downstream corrects its layout"
        );
        assert_eq!(
            state.owned.len(),
            1,
            "the F32 matmul weight must land in the transposed owned path"
        );
        let bound_weight = &state.owned[0].1;

        // Deliberately NOT symmetric around zero (`index - 2.5` sums to
        // zero over `0..6`, which cancels every `out_dim`-dependent term and
        // makes all four outputs identical regardless of axis order -- a
        // vacuous test that would pass under a transpose too). `index + 1`
        // sums to a nonzero, out-axis-coupled value instead.
        let activation: Vec<f32> = (0..in_dim).map(|index| (index as f32) + 1.0).collect();

        let mut program: Vec<proxima_tensor::op::Op> = Vec::new();
        let activation_node = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Input {
                dtype: proxima_tensor::dtype::DType::Float32,
                shape: alloc::vec![proxima_tensor::op::Extent::Static(in_dim as u32)],
                name: Some("activation".into()),
            },
        );
        let weight_node = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Input {
                dtype: proxima_tensor::dtype::DType::Float32,
                shape: alloc::vec![
                    proxima_tensor::op::Extent::Static(in_dim as u32),
                    proxima_tensor::op::Extent::Static(out_dim as u32)
                ],
                name: Some("gate".into()),
            },
        );
        let product = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Elementwise {
                dtype: proxima_tensor::dtype::DType::Float32,
                body: proxima_tensor::op::ScalarOp::Multiply,
                operands: alloc::vec![
                    (
                        activation_node,
                        proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                            2,
                            &[1]
                        ))
                    ),
                    (
                        weight_node,
                        proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                            2,
                            &[1, 0]
                        ))
                    ),
                ],
                name: None,
            },
        );
        let logits = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Reduce(proxima_tensor::op::Reduce {
                dtype: proxima_tensor::dtype::DType::Float32,
                body: proxima_tensor::op::ScalarOp::Add,
                init: proxima_tensor::op::ReduceInit::Zero,
                operand: product,
                in_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                    2,
                    &[0, 1],
                )),
                out_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                    2,
                    &[0],
                )),
                keep: proxima_tensor::op::Keep::Reduce,
                name: Some("logits".into()),
            }),
        );

        let named = [
            ("activation", QuantizedBlock::Float32(activation.as_slice())),
            ("gate", QuantizedBlock::Float32(bound_weight.as_slice())),
        ];
        let evaluated =
            proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[logits])
                .expect("evaluate the bound matmul weight through the real interpreter");
        let ours = evaluated.root();

        let mut oracle = vec![0.0f32; out_dim];
        for (out_index, logit) in oracle.iter_mut().enumerate() {
            let mut accumulator = 0.0f32;
            for in_index in 0..in_dim {
                accumulator += activation[in_index] * on_disk[out_index * in_dim + in_index];
            }
            *logit = accumulator;
        }

        std::println!("raw_packed_f32_matmul ours={ours:?} oracle={oracle:?}");
        for (out_index, (found, wanted)) in ours.iter().zip(&oracle).enumerate() {
            let diff = (found - wanted).abs();
            assert!(
                diff < 1e-4,
                "output {out_index}: found={found} wanted={wanted} diff={diff}"
            );
        }
    }

    #[test]
    fn moe_architecture_is_a_typed_error_not_an_attempted_wrong_bind() {
        let mut architecture = tiny_dense_architecture();
        architecture.expert_count = 8;
        architecture.expert_used_count = 2;
        let file_bytes: Vec<u8> = Vec::new();
        let manifest = Manifest::default();

        let outcome = bind_all_weights_from_safetensors(&manifest, &file_bytes, 0, &architecture);
        let Err(error) = outcome else {
            panic!("a moe architecture must surface as a typed error, not an attempted bind");
        };

        assert!(
            matches!(
                error,
                InteropError::HfMoeWeightsUnsupported { expert_count: 8 }
            ),
            "expected HfMoeWeightsUnsupported{{expert_count: 8}}, got {error:?}"
        );
    }
}
