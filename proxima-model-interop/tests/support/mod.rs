//! Synthetic-but-complete GGUF checkpoints for the capability matrix in
//! `../capability_matrix.rs`: a real GGUF byte stream (through
//! [`proxima_gguf::write_complete`]), parsed back through the real reader
//! (`proxima_gguf::parse_complete`) and bound through the crate's own
//! public [`proxima_model_interop::LoadedModel::load`] -- no host-local
//! model file, no network, no mmap. Every weight tensor's "in" dimension
//! (`EMBEDDING`/`FEED_FORWARD`, both 256) is chosen to be exactly one
//! `Q4_K`/`Q5_K`/`Q6_K` super-block (`QK_K`) wide, the one shape constraint
//! a K-quant codec imposes that a smaller from-scratch toy (`omega/tests/
//! support`'s 64-wide tensor-graph fixture, which never quantizes a
//! weight) does not need to satisfy.
//!
//! Scaled down (2 layers) so every case in the matrix runs in
//! milliseconds, but every op this crate's own `mistral_cached_forward_program`
//! runs on a real checkpoint (embedding lookup, RMSNorm, RoPE, grouped-query
//! attention with a KV cache, SwiGLU, output projection) runs here exactly
//! the same way.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use arrayvec::ArrayVec;
use proxima_gguf::quant::{QuantError, q4_k, q5_k, q6_k, q8_0};
use proxima_gguf::tensor::MAX_DIMS;
use proxima_gguf::{
    GgmlType, GgufModel, MetadataArray, MetadataValue, TensorPayload, write_complete,
};
use proxima_tensor::test_support::Lcg;
use proxima_tokenizer::byte_level::byte_to_char;

/// The "in" dimension of every projection weight this fixture writes --
/// exactly one `Q4_K`/`Q5_K`/`Q6_K` super-block (`QK_K == 256`) wide, so
/// every row of every quantized weight is exactly one packed block, with
/// no partial-block remainder for any codec under test (`Q8_0`'s 32-wide
/// blocks and `F32`'s 1-wide "blocks" both divide 256 too).
pub const EMBEDDING: u32 = 256;
pub const FEED_FORWARD: u32 = 256;
pub const QUERY_HEADS: u32 = 4;
pub const KV_HEADS: u32 = 2;
pub const HEAD_DIM: u32 = 64;
pub const BLOCK_COUNT: u32 = 2;
/// Every base byte (`Vocab::new`'s own requirement for a byte-level BPE
/// vocab) plus one dedicated end-of-sequence marker at the next id.
pub const VOCAB: u32 = 257;
pub const EOS_TOKEN_ID: u32 = 256;

fn dims(values: &[u64]) -> ArrayVec<u64, MAX_DIMS> {
    values.iter().copied().collect()
}

/// Deterministic, non-degenerate weight data -- the same small LCG
/// `omega/tests/support`'s real-forward-graph fixture uses, so a codec's
/// forward pass produces a real, checkable logit distribution instead of
/// an all-zero or all-equal one.
fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// Encodes `values` through `codec`'s real quantizer for the four codecs
/// `proxima_gguf::quant` ships (`Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`); `F32` is a
/// plain little-endian reinterpret. For a `GgmlType` this workspace has no
/// encoder for at all (`Q4_0`/`Q5_0`/`Q2_K`/`Q3_K`), returns a
/// correctly-*sized* buffer (`GgmlType::block_layout`, the same arithmetic
/// `ggml_nbytes` uses) with no encoder run over it: `bind::gguf_tensor_as_f32`
/// rejects those types by `GgmlType` alone before ever reading a byte
/// (`InteropError::UnrepresentableGgmlType`), so what those bytes contain
/// cannot affect the capability-matrix cells that construct one.
pub fn encode_weights(codec: GgmlType, values: &[f32]) -> Vec<u8> {
    let layout = codec.block_layout();
    let blocks = values.len() as u64 / layout.block_elements;
    let byte_len = (blocks * layout.block_bytes) as usize;
    match codec {
        GgmlType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        GgmlType::Q4_K => encode_with(byte_len, values, q4_k::quantize),
        GgmlType::Q5_K => encode_with(byte_len, values, q5_k::quantize),
        GgmlType::Q6_K => encode_with(byte_len, values, q6_k::quantize),
        GgmlType::Q8_0 => encode_with(byte_len, values, q8_0::quantize),
        _ => vec![0u8; byte_len],
    }
}

fn encode_with(
    byte_len: usize,
    values: &[f32],
    quantize: fn(&[f32], &mut [u8]) -> Result<(), QuantError>,
) -> Vec<u8> {
    let mut output = vec![0u8; byte_len];
    quantize(values, &mut output).expect("real codec quantizes this fixture's own weight data");
    output
}

/// A real GPT-2 byte-level BPE vocab (`tokenizer.ggml.model = "gpt2"`,
/// empty merges) covering every one of the 256 base bytes -- enough for
/// `proxima_tokenizer::encode_with_bos_eos` to tokenize any real UTF-8
/// prompt (one token per byte, since no merge rules are declared), and for
/// `LoadedModel`'s own decode loop to recognize [`EOS_TOKEN_ID`] as its
/// stopping signal.
fn tokenizer_metadata(metadata: &mut Vec<(String, MetadataValue)>) {
    let mut tokens: Vec<String> = (0..=255u8)
        .map(|byte| String::from(byte_to_char(byte)))
        .collect();
    tokens.push(String::from("<|endoftext|>"));
    metadata.push((
        String::from("tokenizer.ggml.model"),
        MetadataValue::String(String::from("gpt2")),
    ));
    metadata.push((
        String::from("tokenizer.ggml.tokens"),
        MetadataValue::Array(MetadataArray::String(tokens)),
    ));
    metadata.push((
        String::from("tokenizer.ggml.merges"),
        MetadataValue::Array(MetadataArray::String(Vec::new())),
    ));
    metadata.push((
        String::from("tokenizer.ggml.bos_token_id"),
        MetadataValue::U32(0),
    ));
    metadata.push((
        String::from("tokenizer.ggml.eos_token_id"),
        MetadataValue::U32(EOS_TOKEN_ID),
    ));
}

fn architecture_metadata(metadata: &mut Vec<(String, MetadataValue)>) {
    metadata.push((
        String::from("general.architecture"),
        MetadataValue::String(String::from("llama")),
    ));
    metadata.push((
        String::from("llama.embedding_length"),
        MetadataValue::U32(EMBEDDING),
    ));
    metadata.push((
        String::from("llama.feed_forward_length"),
        MetadataValue::U32(FEED_FORWARD),
    ));
    metadata.push((
        String::from("llama.attention.head_count"),
        MetadataValue::U32(QUERY_HEADS),
    ));
    metadata.push((
        String::from("llama.attention.head_count_kv"),
        MetadataValue::U32(KV_HEADS),
    ));
    metadata.push((
        String::from("llama.block_count"),
        MetadataValue::U32(BLOCK_COUNT),
    ));
    metadata.push((
        String::from("llama.rope.dimension_count"),
        MetadataValue::U32(HEAD_DIM),
    ));
}

/// [`architecture_metadata`] plus the two MoE-only keys
/// [`bind_all_weights`](proxima_model_interop) reads
/// (`bind.rs:263-264`'s own `metadata_u32_optional` lookups) to route a
/// layer's FFN through the routed path instead of the dense triple.
fn moe_architecture_metadata(
    metadata: &mut Vec<(String, MetadataValue)>,
    expert_count: u32,
    expert_used_count: u32,
) {
    architecture_metadata(metadata);
    metadata.push((
        String::from("llama.expert_count"),
        MetadataValue::U32(expert_count),
    ));
    metadata.push((
        String::from("llama.expert_used_count"),
        MetadataValue::U32(expert_used_count),
    ));
}

/// A checkpoint's projection weight -- always `codec`, `dims = [in_dim,
/// out_dim]` (ggml's own `[ne0, ne1]` row-length-first convention,
/// confirmed against `bind.rs`'s own `architecture_from_metadata_reads_real_keys...`
/// fixture) -- pushed into `buffers`/`specs` for [`build`] to assemble into
/// borrowed [`TensorPayload`]s once every buffer is final.
fn push_matmul_weight(
    buffers: &mut Vec<Vec<u8>>,
    specs: &mut Vec<(String, ArrayVec<u64, MAX_DIMS>, GgmlType)>,
    codec: GgmlType,
    name: String,
    seed: u64,
    in_dim: u32,
    out_dim: u32,
) {
    let values = random_vec(seed, in_dim as usize * out_dim as usize);
    buffers.push(encode_weights(codec, &values));
    specs.push((name, dims(&[u64::from(in_dim), u64::from(out_dim)]), codec));
}

/// One MoE-only stacked expert weight (`blk.{layer}.{ffn_gate,ffn_up,
/// ffn_down}_exps.weight`) -- `expert_count` back-to-back `[out_dim,
/// in_dim]` slabs in the same on-disk row-major layout
/// [`push_matmul_weight`] writes for a dense projection, one slab per
/// expert, each seeded distinctly so no two experts' weights collide.
/// [`bind_moe_expert_weights`](proxima_model_interop)'s own doc names this
/// exact tensor name and layout as the native-stacked convention it tries
/// first, before falling back to per-expert-tensor discovery.
#[allow(clippy::too_many_arguments)]
fn push_moe_expert_stack(
    buffers: &mut Vec<Vec<u8>>,
    specs: &mut Vec<(String, ArrayVec<u64, MAX_DIMS>, GgmlType)>,
    codec: GgmlType,
    name: String,
    seed: u64,
    expert_count: u32,
    in_dim: u32,
    out_dim: u32,
) {
    let values = random_vec(
        seed,
        expert_count as usize * in_dim as usize * out_dim as usize,
    );
    buffers.push(encode_weights(codec, &values));
    specs.push((
        name,
        dims(&[
            u64::from(in_dim),
            u64::from(out_dim),
            u64::from(expert_count),
        ]),
        codec,
    ));
}

/// `output.weight` -- like [`push_matmul_weight`], but every row is
/// exactly `0.0` rather than [`random_vec`] data. A logit is `dot(activation,
/// weight_row)`; nothing about *this* fixture's own forward pass (RMSNorm,
/// attention, SwiGLU -- none of them add a constant bias term anywhere)
/// can turn a genuinely-random-SIGN activation into a provably-positive or
/// provably-negative dot product against a fixed weight row, so no nonzero
/// row (however large in magnitude) can be pinned to *always* win or lose
/// the final argmax -- only the zero row can, since `0.0 * x == 0.0`
/// regardless of `x`'s sign or magnitude (checked directly: a per-row
/// large-negative pin was tried first and still occasionally lost to a
/// genuinely-negative real row's own logit, landing on a token id `>= 128`
/// -- a raw byte `>= 128` is a continuation/lead byte of a multi-byte
/// UTF-8 sequence, so an isolated one is exactly the reproducible
/// `TokenizerError::InvalidUtf8` this fixture hit on its first run).
/// An all-zero output projection makes every logit exactly `0.0`, so
/// [`proxima_tokenizer::greedy_pick`]'s own documented tie-break ("ties
/// resolve deterministically to the lowest id") always resolves to token
/// id `0` -- a raw byte `0x00`, independently valid UTF-8 on its own. This
/// is a fixture design constraint on the *last* matmul only: every other
/// weight in the checkpoint (attention, SwiGLU, all `block_count` layers)
/// is real [`random_vec`] data run through the codec under test, so a
/// broken kernel anywhere upstream (wrong shape, NaN, a codec that
/// silently misreads its own bytes) still has every opportunity to
/// surface as a non-`0` id, a panic, or a shape error.
fn push_output_projection(
    buffers: &mut Vec<Vec<u8>>,
    specs: &mut Vec<(String, ArrayVec<u64, MAX_DIMS>, GgmlType)>,
    codec: GgmlType,
    embedding: u32,
    vocab: u32,
) {
    let values = vec![0.0f32; embedding as usize * vocab as usize];
    buffers.push(encode_weights(codec, &values));
    specs.push((
        String::from("output.weight"),
        dims(&[u64::from(embedding), u64::from(vocab)]),
        codec,
    ));
}

/// A dense (always-`F32`) 1-D weight -- every real checkpoint's own norm
/// vectors, `bind.rs`'s own doc for why norms never carry a matmul codec.
fn push_dense_vector(
    buffers: &mut Vec<Vec<u8>>,
    specs: &mut Vec<(String, ArrayVec<u64, MAX_DIMS>, GgmlType)>,
    name: String,
    seed: u64,
    len: u32,
) {
    let values = random_vec(seed, len as usize);
    buffers.push(encode_weights(GgmlType::F32, &values));
    specs.push((name, dims(&[u64::from(len)]), GgmlType::F32));
}

/// A synthetic-but-complete Mistral/Llama-shaped GGUF checkpoint whose
/// every matmul-operand weight (`attn_{q,k,v,output}`, `ffn_{gate,up,down}`,
/// `output.weight`) is stored in `weight_codec` -- `token_embd.weight` and
/// every norm vector stay `F32` (real quantized checkpoints keep norms
/// dense too; `token_embd.weight` is bound via embedding lookup, never a
/// matmul operand, so this fixture keeps it out of the codec under test).
/// Returns the real, parseable GGUF byte stream
/// ([`proxima_gguf::write_complete`]) -- no file, no mmap, no host-local
/// dependency.
#[must_use]
pub fn checkpoint_bytes(weight_codec: GgmlType) -> Vec<u8> {
    let embedding = EMBEDDING;
    let feed_forward = FEED_FORWARD;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let vocab = VOCAB;

    let mut buffers: Vec<Vec<u8>> = Vec::new();
    let mut specs: Vec<(String, ArrayVec<u64, MAX_DIMS>, GgmlType)> = Vec::new();
    let mut seed = 1u64;
    let mut next_seed = || {
        seed += 1;
        seed
    };

    push_dense_vector(
        &mut buffers,
        &mut specs,
        String::from("token_embd.weight"),
        next_seed(),
        vocab * embedding,
    );

    for layer in 0..BLOCK_COUNT {
        push_dense_vector(
            &mut buffers,
            &mut specs,
            format!("blk.{layer}.attn_norm.weight"),
            next_seed(),
            embedding,
        );
        push_dense_vector(
            &mut buffers,
            &mut specs,
            format!("blk.{layer}.ffn_norm.weight"),
            next_seed(),
            embedding,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.attn_q.weight"),
            next_seed(),
            embedding,
            embedding,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.attn_k.weight"),
            next_seed(),
            embedding,
            kv_dim,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.attn_v.weight"),
            next_seed(),
            embedding,
            kv_dim,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.attn_output.weight"),
            next_seed(),
            embedding,
            embedding,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.ffn_gate.weight"),
            next_seed(),
            embedding,
            feed_forward,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.ffn_up.weight"),
            next_seed(),
            embedding,
            feed_forward,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.ffn_down.weight"),
            next_seed(),
            feed_forward,
            embedding,
        );
    }

    push_dense_vector(
        &mut buffers,
        &mut specs,
        String::from("output_norm.weight"),
        next_seed(),
        embedding,
    );
    push_output_projection(&mut buffers, &mut specs, weight_codec, embedding, vocab);

    let tensors: Vec<TensorPayload<'_>> = specs
        .iter()
        .zip(buffers.iter())
        .map(|((name, dims, ggml_type), data)| TensorPayload {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: *ggml_type,
            data: data.as_slice(),
        })
        .collect();

    let mut metadata = Vec::new();
    architecture_metadata(&mut metadata);
    tokenizer_metadata(&mut metadata);

    let model = GgufModel {
        version: 3,
        metadata,
        tensors,
    };
    write_complete(&model).expect("writes a well-formed synthetic checkpoint")
}

/// [`EXPERT_COUNT`]/[`EXPERT_USED_COUNT`] for [`checkpoint_bytes_moe`] --
/// Mixtral's own shape (8 experts, top-2) scaled down to a size the fixture's
/// `Q4_K`/`Q5_K`/`Q6_K` block-width constraint (`EMBEDDING`/`FEED_FORWARD`
/// both one `QK_K`-wide super-block) already keeps small.
pub const EXPERT_COUNT: u32 = 4;
pub const EXPERT_USED_COUNT: u32 = 2;

/// [`checkpoint_bytes`]'s mixture-of-experts counterpart: every dense
/// checkpoint tensor this fixture already writes, minus the dense
/// `ffn_{gate,up,down}.weight` triple, plus `ffn_gate_inp.weight` (the
/// router -- always `F32`, matching every real checkpoint's own norm/router
/// vectors never carrying a matmul quant codec in this fixture) and the
/// stacked `ffn_{gate,up,down}_exps.weight` routed experts in `weight_codec`
/// ([`push_moe_expert_stack`]), plus the `{architecture}.expert_count`/
/// `expert_used_count` metadata keys [`bind_all_weights`](proxima_model_interop)
/// reads to select the routed bind path at all. Same real GGUF byte stream
/// (`write_complete`), same real quantizer per codec, same 2-layer/256-wide
/// shape as [`checkpoint_bytes`] -- the only difference is which FFN weight
/// family each layer carries.
#[must_use]
pub fn checkpoint_bytes_moe(
    weight_codec: GgmlType,
    expert_count: u32,
    expert_used_count: u32,
) -> Vec<u8> {
    let embedding = EMBEDDING;
    let feed_forward = FEED_FORWARD;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let vocab = VOCAB;

    let mut buffers: Vec<Vec<u8>> = Vec::new();
    let mut specs: Vec<(String, ArrayVec<u64, MAX_DIMS>, GgmlType)> = Vec::new();
    let mut seed = 1u64;
    let mut next_seed = || {
        seed += 1;
        seed
    };

    push_dense_vector(
        &mut buffers,
        &mut specs,
        String::from("token_embd.weight"),
        next_seed(),
        vocab * embedding,
    );

    for layer in 0..BLOCK_COUNT {
        push_dense_vector(
            &mut buffers,
            &mut specs,
            format!("blk.{layer}.attn_norm.weight"),
            next_seed(),
            embedding,
        );
        push_dense_vector(
            &mut buffers,
            &mut specs,
            format!("blk.{layer}.ffn_norm.weight"),
            next_seed(),
            embedding,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.attn_q.weight"),
            next_seed(),
            embedding,
            embedding,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.attn_k.weight"),
            next_seed(),
            embedding,
            kv_dim,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.attn_v.weight"),
            next_seed(),
            embedding,
            kv_dim,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            weight_codec,
            format!("blk.{layer}.attn_output.weight"),
            next_seed(),
            embedding,
            embedding,
        );
        push_matmul_weight(
            &mut buffers,
            &mut specs,
            GgmlType::F32,
            format!("blk.{layer}.ffn_gate_inp.weight"),
            next_seed(),
            embedding,
            expert_count,
        );
        for (projection, out_dim) in [
            ("ffn_gate", feed_forward),
            ("ffn_up", feed_forward),
            ("ffn_down", embedding),
        ] {
            let in_dim = if projection == "ffn_down" {
                feed_forward
            } else {
                embedding
            };
            push_moe_expert_stack(
                &mut buffers,
                &mut specs,
                weight_codec,
                format!("blk.{layer}.{projection}_exps.weight"),
                next_seed(),
                expert_count,
                in_dim,
                out_dim,
            );
        }
    }

    push_dense_vector(
        &mut buffers,
        &mut specs,
        String::from("output_norm.weight"),
        next_seed(),
        embedding,
    );
    push_output_projection(&mut buffers, &mut specs, weight_codec, embedding, vocab);

    let tensors: Vec<TensorPayload<'_>> = specs
        .iter()
        .zip(buffers.iter())
        .map(|((name, dims, ggml_type), data)| TensorPayload {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: *ggml_type,
            data: data.as_slice(),
        })
        .collect();

    let mut metadata = Vec::new();
    moe_architecture_metadata(&mut metadata, expert_count, expert_used_count);
    tokenizer_metadata(&mut metadata);

    let model = GgufModel {
        version: 3,
        metadata,
        tensors,
    };
    write_complete(&model).expect("writes a well-formed synthetic MoE checkpoint")
}
