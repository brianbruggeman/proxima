//! Metal Shading Language kernel emission.
//!
//! [`emit`] turns one lowered [`BoundOp`] into one [`Kernel`]: MSL source text,
//! an entry name, the buffer-index -> data [`Binding`] list a driver needs to
//! set up a dispatch, and the thread-count [`GridSpec`] for this invocation.
//!
//! # Runtime uniforms, not baked constants
//!
//! A `BoundOp` node's extents and strides are read out of a `constant
//! Uniforms&` buffer at kernel runtime — never spliced into the source text
//! as literal numbers. What *does* vary the source is the node's STRUCTURE:
//! rank (operand and output coordinate arity), operand count, which
//! [`ScalarOp`]s the body and (if present) the reduction use, and whether a
//! reduction is present at all and which [`Keep`] it is. Concrete extent and
//! stride values remain uniforms. Elementwise kernels additionally specialize
//! on the stride layout's dense/strided addressing class so dense operands can
//! skip coordinate decoding; [`elementwise_addressing_cache_token`] records
//! that class in the pipeline identity. This makes a kernel cacheable by the
//! exact source it emits rather than by node identity.
//!
//! # Execution model (v1: correctness parity with `cpu.rs`, not peak speed)
//!
//! - **Elementwise** (no reduction): one thread per output element. A
//!   thread's linear id decodes into a coordinate via the same row-major
//!   div/mod chain `cpu::unflatten` uses, each operand's read offset is
//!   `base + sum(coord[d] * stride[d])`, and the body writes directly to the
//!   dense output at its own linear id — matching `cpu::run_elementwise`.
//! - **Fused fold, `Keep::Reduce`** (reduce): one thread per OUTPUT element
//!   (matmul is one thread per `(i, j)`), with a serial loop over the
//!   reduction dims inside the kernel. `ReduceInit` seeding — including
//!   `FirstElement`'s seed-on-first-step behavior — matches
//!   `cpu::run_reduce` exactly: the accumulator is seeded from the *first*
//!   reduction step's value rather than combined into an `init` constant.
//! - **`Keep::Scan`** (scan): one thread per non-folded coordinate line,
//!   serial along the folded (innermost) dim, writing every prefix through
//!   the output strides — matching `cpu::run_scan`.
//!
//! Parity extends to the sad path: `cpu.rs` returns
//! `TensorError::GatherIndexOutOfRange` for a fetched index outside
//! `[0, extent)` rather than clamping it, and a gather kernel here agrees —
//! it clamps for memory safety (a GPU kernel cannot propagate a `Result`)
//! but also records the fault into the `Fault` buffer `crate::metal` reads
//! back after dispatch and turns into the identical error. See
//! `push_gather_fetch`'s doc for where the check is emitted.
//!
//! # dtype
//!
//! `BoundOp` carries its own element type ([`proxima_tensor::BoundOp::dtype`],
//! read straight from the [`proxima_tensor::Op`] it was built from). Every
//! buffer/scratch/accumulator declaration this module renders is spelled
//! from `type_token` rather than hardcoding `float`, so a `Float16` node
//! emits a kernel of `half` declarations while a `Float32` node emits the
//! same `float` kernel this module always has. The *op logic* — which
//! `ScalarOp` token, which reduction init, how a body's steps chain — never
//! consults dtype at all: `op_token`, `scalar_op_expr`, `init_token`,
//! `fold_init_tokens` stay total over their enums exactly as before, and
//! only the declaration spelling varies. `cpu.rs`'s own evaluator remains
//! f32-only (`cpu::reject_non_float32`) — it is the reference oracle, not
//! this crate's dtype ceiling. `omega::execute` runs its own, narrower
//! upstream gate (`Float32` or `Float16` only) before a `BoundOp` ever
//! reaches [`emit`].

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_tensor::{
    BoundOp, BoundOpKind, ComposedBody, DType, Keep, Layout, Lookup, NodeId, NumericPolicy,
    NumericRewrite, ReduceInit, ScalarOp, StepArg, admit,
};

use crate::error::EmitError;
use crate::identity::{
    body_token, init_token, keep_token, op_token, operand_codecs, reduce_epilogue_is_identity,
    signed_name_part,
};
#[cfg(all(
    any(feature = "metal-packed-row-nsg2", feature = "metal-q4k-ggml-port"),
    not(feature = "metal-q4k-split-k")
))]
use crate::sized::PACKED_ROW_NSG;
use crate::sized::SIMD_WIDTH;

/// MSL source for one `Q2_K` element. The byte layout and index arithmetic
/// mirror `proxima_gguf::quant::q2_k::dequantize_block`: sixteen scale/min
/// bytes, sixty-four 2-bit payload bytes, then trailing `f16` `d`/`dmin`.
pub const Q2K_UNPACK_MSL: &str = r#"
static inline float q2k_element(device const uchar *block, uint index) {
    device const uchar *scales = block;
    device const uchar *qs = block + 16;
    ushort d_bits = (ushort)((uint)block[80] | ((uint)block[81] << 8));
    ushort dmin_bits = (ushort)((uint)block[82] | ((uint)block[83] << 8));
    float d = (float)as_type<half>(d_bits);
    float dmin = (float)as_type<half>(dmin_bits);

    uint chunk = index / 128u;
    uint within = index % 128u;
    uint group = within / 32u;
    uint local = within % 32u;
    uint sub_block = chunk * 8u + group * 2u + (local >= 16u ? 1u : 0u);
    uchar scale_min = scales[sub_block];
    float scale = d * (float)(scale_min & 0x0Fu);
    float minimum = dmin * (float)(scale_min >> 4u);
    uchar level = (qs[chunk * 32u + local] >> (2u * group)) & 0x03u;
    return scale * (float)level - minimum;
}
"#;

/// Bytes and elements in one `Q2_K` super-block, sourced from the GGUF
/// codec rather than restated numeric constants.
pub const Q2K_BLOCK_BYTES: usize = proxima_gguf::quant::q2_k::BLOCK_BYTES;
pub const Q2K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q2_k::QK_K;

/// Emits the runtime codec selector used by a mixed HOBBIT expert gather.
/// The payload pointer is one borrowed byte arena and each descriptor supplies
/// its own byte offset; a later lowering can bind this helper without changing
/// the graph's gather index map.
pub const MIXED_EXPERT_READ_MSL: &str = r#"
struct ExpertPayloadDescriptor {
    uint expert_index;
    uint codec;
    uint byte_offset;
    uint byte_length;
    uint out_dim;
    uint in_dim;
    uint epoch;
    uint reserved;
};
static inline float mixed_expert_element(device const uchar *payload,
                                         device const ExpertPayloadDescriptor *descriptor,
                                         uint expert,
                                         uint element) {
    ExpertPayloadDescriptor selected = descriptor[expert];
    device const uchar *block = payload + selected.byte_offset;
    if (selected.codec == 1u) {
        return q2k_element(block + (element / 256u) * 84u, element % 256u);
    }
    if (selected.codec == 2u) {
        return q4k_element(block + (element / 256u) * 144u, element % 256u);
    }
    if (selected.codec == 3u) {
        return q6k_element(block + (element / 256u) * 210u, element % 256u);
    }
    if (selected.codec == 4u) {
        return q3k_element(block + (element / 256u) * 110u, element % 256u);
    }
    return 0.0f;
}
static inline float mixed_expert_element_from_offset(
        device const uchar *payload,
        device const ExpertPayloadDescriptor *descriptor,
        uint expert,
        long full_offset,
        long expert_stride,
        uint codec) {
    long local = full_offset - (long)expert * expert_stride;
    if (codec == 1u) {
        device const uchar *block = payload + descriptor[expert].byte_offset +
            (uint)(local / 256l) * 84u;
        return q2k_element(block, (uint)(local % 256l));
    }
    if (codec == 2u) {
        device const uchar *block = payload + descriptor[expert].byte_offset +
            (uint)(local / 256l) * 144u;
        return q4k_element(block, (uint)(local % 256l));
    }
    if (codec == 3u) {
        device const uchar *block = payload + descriptor[expert].byte_offset +
            (uint)(local / 256l) * 210u;
        return q6k_element(block, (uint)(local % 256l));
    }
    if (codec == 4u) {
        device const uchar *block = payload + descriptor[expert].byte_offset +
            (uint)(local / 256l) * 110u;
        return q3k_element(block, (uint)(local % 256l));
    }
    return 0.0f;
}
static inline float mixed_expert_element_from_local(
        device const uchar *base,
        uint codec,
        long local) {
    if (codec == 1u) {
        return q2k_element(base + (uint)(local / 256l) * 84u, (uint)(local % 256l));
    }
    if (codec == 2u) {
        return q4k_element(base + (uint)(local / 256l) * 144u, (uint)(local % 256l));
    }
    if (codec == 3u) {
        return q6k_element(base + (uint)(local / 256l) * 210u, (uint)(local % 256l));
    }
    if (codec == 4u) {
        return q3k_element(base + (uint)(local / 256l) * 110u, (uint)(local % 256l));
    }
    return 0.0f;
}
"#;

/// Codec-specialized descriptor reads used by the packed row kernels.  The
/// descriptor still supplies the selected expert's offset, but the decoder is
/// fixed in the generated body so a uniform sidecar codec keeps the same
/// block-level work sharing as the ordinary packed path.
pub const UNIFORM_EXPERT_READ_MSL: &str = r#"
static inline float uniform_expert_element_from_offset_q2k(
        device const uchar *payload,
        device const ExpertPayloadDescriptor *descriptor,
        uint expert,
        long full_offset,
        long expert_stride) {
    long local = full_offset - (long)expert * expert_stride;
    device const uchar *block = payload + descriptor[expert].byte_offset +
        (uint)(local / 256l) * 84u;
    return q2k_element(block, (uint)(local % 256l));
}
static inline float uniform_expert_element_from_offset_q3k(
        device const uchar *payload,
        device const ExpertPayloadDescriptor *descriptor,
        uint expert,
        long full_offset,
        long expert_stride) {
    long local = full_offset - (long)expert * expert_stride;
    device const uchar *block = payload + descriptor[expert].byte_offset +
        (uint)(local / 256l) * 110u;
    return q3k_element(block, (uint)(local % 256l));
}
static inline float uniform_expert_element_from_offset_q4k(
        device const uchar *payload,
        device const ExpertPayloadDescriptor *descriptor,
        uint expert,
        long full_offset,
        long expert_stride) {
    long local = full_offset - (long)expert * expert_stride;
    device const uchar *block = payload + descriptor[expert].byte_offset +
        (uint)(local / 256l) * 144u;
    return q4k_element(block, (uint)(local % 256l));
}
static inline float uniform_expert_element_from_offset_q5k(
        device const uchar *payload,
        device const ExpertPayloadDescriptor *descriptor,
        uint expert,
        long full_offset,
        long expert_stride) {
    long local = full_offset - (long)expert * expert_stride;
    device const uchar *block = payload + descriptor[expert].byte_offset +
        (uint)(local / 256l) * 176u;
    return q5k_element(block, (uint)(local % 256l));
}
static inline float uniform_expert_element_from_offset_q6k(
        device const uchar *payload,
        device const ExpertPayloadDescriptor *descriptor,
        uint expert,
        long full_offset,
        long expert_stride) {
    long local = full_offset - (long)expert * expert_stride;
    device const uchar *block = payload + descriptor[expert].byte_offset +
        (uint)(local / 256l) * 210u;
    return q6k_element(block, (uint)(local % 256l));
}
"#;

/// One compiled kernel: MSL source, its entry point, the buffer-index ->
/// data mapping a driver needs to bind before dispatch, and the thread count
/// this particular op needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub source: String,
    pub entry: String,
    pub bindings: Vec<Binding>,
    pub grid: GridSpec,
}

/// What buffer index `n` in [`Kernel::bindings`] is for, in dispatch order:
/// index `0..operands.len()` are inputs, then one `Indices` buffer per
/// gathered operand (in operand order), then the output, then the uniforms
/// blob (extents/strides/bases for this dispatch — see the module doc), then
/// — only when the op gathers — the fault buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    Input(NodeId),
    /// Codec-specific payload bytes for a mixed expert source. The matching
    /// [`ExpertDescriptors`] binding selects the byte span and decoder for
    /// each routed expert index.
    ExpertPayloads(NodeId),
    /// Per-expert codec and byte-span records for a mixed expert source.
    /// Kept separate from [`ExpertPayloads`] so the payloads remain borrowed
    /// mapped ranges rather than one concatenated staging allocation.
    ExpertDescriptors(NodeId),
    /// The `indices` buffer a gathered operand fetches from.
    Indices(NodeId),
    Output(NodeId),
    Uniforms,
    /// Present only when `gather_count` is nonzero: a `gather_count`-long
    /// zero-initialized `atomic_uint` array. The kernel `atomic_fetch_max`s
    /// an out-of-range fetched index (plus one, so zero means "no fault")
    /// into its gathered operand's slot; the driver reads this back after
    /// dispatch and turns a nonzero slot into the same
    /// `TensorError::GatherIndexOutOfRange` `cpu::evaluate` would report —
    /// see `push_gather_fetch`'s doc for how the check is emitted.
    Fault,
    /// `CachedAttention`'s cross-threadgroup key-split scratch — present
    /// only when `cached_attention_merge_needed` admits
    /// [`NumericRewrite::ContextSplitMerge`]. The split kernel WRITES it
    /// (this binding replaces `Binding::Output` in that kernel's own
    /// `bindings` list, since the split no longer writes the op's real
    /// output directly); the merge kernel READS it. Not `NodeId`-keyed —
    /// unlike every other binding, this buffer has no program node of its
    /// own, so the identity a hazard tracker needs comes from
    /// `crate::metal`'s own scratch-buffer pointer, resolved outside
    /// `device_buffers` (see that crate's `BufferArena` scratch slot).
    Scratch,
}

/// How many threads a driver must dispatch for this op — one per
/// independent unit of work (output element for elementwise/reduce, output
/// line for a scan). Unlike [`Kernel::source`], this genuinely is a function
/// of the op's concrete extents, not just its structure: it is per-dispatch
/// data, the same way an argument to a function call is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridSpec {
    pub threads: u64,
    /// `Some(SIMD_WIDTH)` for a cooperative reduce (see `reduce_is_cooperative`):
    /// the driver must dispatch threadgroups exactly this wide so every
    /// SIMD-group boundary lands on an output-element boundary (`gid / SIMD_WIDTH`
    /// is only a valid output index under that alignment — see
    /// `push_cooperative_reduce_body`'s doc). `Some(query_groups * SIMD_WIDTH)`
    /// for `CachedAttention`: the kernel body cooperatively loads each K/V row
    /// into `threadgroup` memory once per threadgroup rather than once per
    /// simdgroup (`render_cached_attention`'s own doc), which only stays
    /// correct if every simdgroup sharing a kv_head lands in the same
    /// threadgroup. `None` for every other kernel, which keeps the
    /// occupancy-driven width the driver already picks.
    pub threadgroup_width: Option<u64>,
}

/// Emits an MSL kernel from a bound [`BoundOp`] — the GPU-emission half of
/// the same descriptor `proxima_tensor::cpu` interprets on CPU. See the
/// module doc for the runtime-uniforms stance and the per-[`Keep`]
/// execution model.
///
/// # Examples
///
/// ```
/// use proxima_tensor::{DType, Extent, IndexMap, Op, ScalarOp, append, map};
///
/// let mut program = Vec::new();
/// let source = append(
///     &mut program,
///     Op::Input {
///         dtype: DType::Float32,
///         shape: vec![Extent::Static(4)],
///         name: None,
///     },
/// );
/// append(
///     &mut program,
///     Op::Elementwise {
///         dtype: DType::Float32,
///         body: ScalarOp::Tanh,
///         operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
///         name: None,
///     },
/// );
///
/// let shapes = proxima_tensor::infer(&program, &[])?;
/// let bound_ops = proxima_tensor::bind(&program, &shapes, &[], proxima_tensor::NumericPolicy::default())?;
///
/// // no packed (quantized/half-precision) operand in this program, so an
/// // empty codec table is exactly right -- see `PackedOperands`'s own doc.
/// let packed_operands = omega::PackedOperands::new();
/// let kernel = omega::emit(&bound_ops[0], &packed_operands, proxima_tensor::NumericPolicy::default())?;
/// assert!(kernel.source.contains("kernel void"));
/// assert!(kernel.source.contains("tanh("));
/// assert_eq!(kernel.bindings.len(), 3); // one input, one output, uniforms
/// assert_eq!(kernel.grid.threads, 4);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// MSL source for unpacking one element of a `Q3_K` super-block. Ports
/// `proxima_gguf::quant::q3_k::dequantize_block` exactly: `x = d*sc*q`, `sc`
/// a signed 6-bit scale unpacked the same bit-interleaved way
/// `q4k_scale_min`/`q5k_scale_min` unpack their `(scale, min)` pairs (here
/// carrying one scale, no min -- see [`q3_k`'s module
/// doc](../../proxima_gguf/src/quant/q3_k.rs) for the derivation), and `q`
/// a 3-bit level assembled from a 2-bit `qs` lane plus one `hmask` high bit
/// (`level - 4` when the high bit is set, `level` otherwise -- ported here
/// as `level - (bit_set ? 0 : 4)`, algebraically the same correction).
///
/// Layout, 110 bytes per 256 elements: `hmask[32]` (one high bit per
/// element) at 0, `qs[64]` (2-bit lanes, four elements per byte) at 32,
/// `scales[12]` (packed 6-bit codes) at 96, `d` f16 at 108 -- `d` TRAILS the
/// block, the same trailing position [`Q6K_UNPACK_MSL`] uses, unlike
/// `Q4_K`/`Q5_K`/`Q8_0` where it leads.
///
/// Index arithmetic: for element `index` (0..256), `chunk = index/128`,
/// `j = (index%128)/32`, `local32 = index%32`. The `qs` byte is
/// `qs[chunk*32 + local32]`, read at bit-shift `2*j`; the `hmask` byte is
/// `hmask[local32]` (the SAME 32-byte range for both super-block halves,
/// same "shared bit, different byte range" trap `q3_k::dequantize_block`'s
/// own doc calls out, inverted from `Q5_K`'s `qh` trap) tested against bit
/// `1 << (4*chunk + j)`; the scale sub-block is
/// `8*chunk + 2*j + (local32 < 16 ? 0 : 1)`.
pub const Q3K_UNPACK_MSL: &str = r#"
// ports proxima_gguf::quant::q3_k::unpack_scale -- the same bit-interleaved
// 6-bit unpack q4k_scale_min/q5k_scale_min use for a (scale, min) pair,
// restated here for a scale-only code (no min), same posture as
// q5k_scale_min restating q4k_scale_min.
static inline int q3k_unpack_scale(device const uchar *scales, uint sub_block) {
    uchar low = (sub_block < 8u) ? (scales[sub_block] & 0x0Fu) : (scales[sub_block - 8u] >> 4u);
    uchar high = (scales[8u + sub_block % 4u] >> (2u * (sub_block / 4u))) & 0x03u;
    int combined = (int)(low | (high << 4u));
    return combined - 32;
}

// one Q3_K super-block's per-sub-block (16-element) scale and high-bit
// MASK, decoded ONCE for a run of elements sharing both -- same
// amortization q4k_header_for/q5k_header_for make, at Q3_K's own 16-element
// sub-block granularity rather than their 32-element one.
struct q3k_header { float scale; uchar mask; };

static inline q3k_header q3k_header_for(device const uchar *block, uint index) {
    device const uchar *scales = block + 96;
    ushort d_bits = (ushort)((uint)block[108] | ((uint)block[109] << 8));
    float d = (float)as_type<half>(d_bits);

    uint chunk = index / 128u;
    uint rem = index % 128u;
    uint j = rem / 32u;
    uint local32 = rem % 32u;
    bool low = local32 < 16u;
    uint sub_block = 8u * chunk + 2u * j + (low ? 0u : 1u);

    q3k_header header;
    header.scale = d * (float)q3k_unpack_scale(scales, sub_block);
    header.mask = (uchar)(1u << (4u * chunk + j));
    return header;
}

// one element, given its sub-block's already-decoded header.
static inline float q3k_value(device const uchar *block, uint index, q3k_header header) {
    device const uchar *hmask = block;
    device const uchar *qs = block + 32;

    uint chunk = index / 128u;
    uint rem = index % 128u;
    uint j = rem / 32u;
    uint local32 = rem % 32u;

    uchar level = (qs[chunk * 32u + local32] >> (2u * j)) & 0x03u;
    float correction = (hmask[local32] & header.mask) != 0u ? 0.0f : 4.0f;
    return header.scale * ((float)level - correction);
}

// element `index` of one Q3_K super-block, decoding its own header first --
// the generic per-element path (`operand_read`'s non-row-blocked callers)
// has no amortized header to reuse across elements, same posture as
// q5k_element/q6k_element.
static inline float q3k_element(device const uchar *block, uint index) {
    return q3k_value(block, index, q3k_header_for(block, index));
}
"#;

/// Bytes one `Q3_K` super-block occupies -- read from
/// `proxima_gguf::quant::q3_k::BLOCK_BYTES`, the one place this number is
/// defined; pinned in `omega/tests/q3k_unpack.rs`, same posture as
/// [`Q4K_BLOCK_BYTES`].
pub const Q3K_BLOCK_BYTES: usize = proxima_gguf::quant::q3_k::BLOCK_BYTES;

/// The paired plain-product body for `Q3_K`, structurally selected by
/// `PackedCodec::supports_pair_dot` (no new Cargo feature -- `Q3_K` always
/// takes this arm when the reduce is a plain product, the same unconditional
/// posture `Q4_K`'s own arm already has). Same `iq`/`ir` lane assignment and
/// `yl`/`yh` activation gather `q4k_pair_dot`/`q5k_pair_dot` use
/// (`plain_product`'s gather is codec-agnostic, built once above this match)
/// but a DIFFERENT byte addressing shape: `Q3_K`'s 2-bit `qs` lanes pack FOUR
/// levels per byte at shifts `0/2/4/6`, not `Q4_K`'s two nibbles or `Q5_K`'s
/// nibble-plus-plane, so `iq` here selects which SHIFT PAIR (`0,2` or `4,6`)
/// rather than a separate 32-byte `qs` window the way `Q4_K`/`Q5_K`'s
/// `32*iq` term does -- the byte offset is `8*ir + l` alone, shared by all
/// four of this call's levels (derived from `q3k_header_for`'s own
/// `chunk`/`j`/`local32` algebra: for `low_index = 64*iq + 8*ir`,
/// `low_index`/`low_index+32` share `chunk=0` and `local32=8*ir+l`,
/// differing only in `j` -> `shift`; adding 128 flips `chunk` to 1 without
/// changing `local32`, which is why `low_index+128`/`low_index+160` read the
/// SAME `hmask` byte too). Headers are looked up via the existing
/// `q3k_header_for` rather than re-derived, same restated-vs-shared posture
/// `q5k_pair_dot` takes for its own header lookups.
pub const Q3K_PAIR_DOT_MSL: &str = r#"
static inline float q3k_pair_dot(device const uchar *block, uint iq, uint ir, thread const float *yl, thread const float *yh) {
    device const uchar *hmask = block;
    device const uchar *qs = block + 32;
    uint low_index = 64u * iq + 8u * ir;
    q3k_header h0 = q3k_header_for(block, low_index);
    q3k_header h1 = q3k_header_for(block, low_index + 32u);
    q3k_header h2 = q3k_header_for(block, low_index + 128u);
    q3k_header h3 = q3k_header_for(block, low_index + 160u);
    uint shift0 = 4u * iq;
    uint shift1 = shift0 + 2u;
    uint byte_offset = 8u * ir;
    float result = 0.0f;
    for (uint l = 0u; l < 8u; ++l) {
        uchar q1 = qs[byte_offset + l];
        uchar q2 = qs[32u + byte_offset + l];
        uchar hm = hmask[byte_offset + l];
        float low0 = (float)((q1 >> shift0) & 0x03u) - ((hm & h0.mask) != 0u ? 0.0f : 4.0f);
        float low1 = (float)((q1 >> shift1) & 0x03u) - ((hm & h1.mask) != 0u ? 0.0f : 4.0f);
        float high0 = (float)((q2 >> shift0) & 0x03u) - ((hm & h2.mask) != 0u ? 0.0f : 4.0f);
        float high1 = (float)((q2 >> shift1) & 0x03u) - ((hm & h3.mask) != 0u ? 0.0f : 4.0f);
        result += h0.scale * low0 * yl[l];
        result += h1.scale * low1 * yl[l + 8u];
        result += h2.scale * high0 * yh[l];
        result += h3.scale * high1 * yh[l + 8u];
    }
    return result;
}
"#;

/// MSL source for unpacking one element of a `Q4_K` super-block, straight
/// out of the packed GGUF bytes with no `f32` weight tensor ever
/// materialized.
///
/// This is the whole GPU story for decode, not a convenience. Decode is a
/// weight sweep, so bytes-per-weight is the only variable that moves the
/// number: `f16` is 2.0 B/weight and `Q4_K` is 0.5625, a 3.56x difference in
/// traffic. Measured on an M1 Max, llama.cpp's Metal backend runs this same
/// 7B checkpoint at 17.62 ms/token (56.8 tok/s, 214.7 GB/s achieved) reading
/// packed `Q4_K`; the same sweep in `f16` is 14.5 GB per token, which at that
/// bandwidth is 67.4 ms/token — slower than our own CPU path. A float-only
/// GPU backend is not worth having (`proxima-tensor/docs/discipline.md`
/// ROW 69).
///
/// Ports `proxima_gguf::quant::q4_k::dequantize_block` exactly, including
/// the two details that are easy to get wrong and silently plausible:
/// `get_scale_min_k4`'s 6-bit scale/min unpacking (sub-blocks 4..8 take
/// their high bits from a DIFFERENT byte than their low bits), and the
/// nibble order — a `qs` byte's low and high nibbles land 32 elements apart,
/// not adjacent, so element `i`'s byte is NOT `qs[i / 2]`.
///
/// Layout, 144 bytes per 256 elements: `d` f16 at 0, `dmin` f16 at 2,
/// 12 packed scale/min bytes at 4, 128 nibble bytes at 16.
pub const Q4K_UNPACK_MSL: &str = r#"
// ports proxima_gguf::quant::q4_k::get_scale_min_k4
static inline uchar2 q4k_scale_min(device const uchar *scales, uint sub_block) {
    if (sub_block < 4u) {
        return uchar2(scales[sub_block] & 63, scales[sub_block + 4u] & 63);
    }
    uchar scale = (scales[sub_block + 4u] & 0x0F) | ((scales[sub_block - 4u] >> 6) << 4);
    uchar minimum = (scales[sub_block + 4u] >> 4) | ((scales[sub_block] >> 6) << 4);
    return uchar2(scale, minimum);
}

// element `index` (0..256) of one Q4_K super-block, byte-for-byte the value
// proxima_gguf::quant::q4_k::dequantize_block writes at the same index.
static inline float q4k_element(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    ushort dmin_bits = (ushort)((uint)block[2] | ((uint)block[3] << 8));
    float d = (float)as_type<half>(d_bits);
    float dmin = (float)as_type<half>(dmin_bits);

    device const uchar *scales = block + 4;
    device const uchar *qs = block + 16;

    uint group = index / 64u;
    uint within = index % 64u;
    bool low_nibble = within < 32u;
    uint sub_block = 2u * group + (low_nibble ? 0u : 1u);
    uint byte_index = group * 32u + (within % 32u);

    uchar2 scale_min = q4k_scale_min(scales, sub_block);
    float scale = d * (float)scale_min.x;
    float minimum = dmin * (float)scale_min.y;
    uchar nibble = low_nibble ? (qs[byte_index] & 0x0F) : (qs[byte_index] >> 4);
    return scale * (float)nibble - minimum;
}

// A super-block's per-sub-block scale and min, decoded ONCE for a run of
// elements inside that sub-block. `d`, `dmin` and the 6-bit scale/min pair
// are constant across all 32 elements of a sub-block, so deriving them per
// element (what `q4k_element` does) is 8-40x the arithmetic of the nibble
// extract it feeds. ggml decodes once per super-block and spends ~1.6 ops
// per weight; this is the same amortization.
struct q4k_header { float scale; float minimum; };

static inline q4k_header q4k_header_for(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    ushort dmin_bits = (ushort)((uint)block[2] | ((uint)block[3] << 8));
    device const uchar *scales = block + 4;
    uint group = index / 64u;
    uint within = index % 64u;
    uint sub_block = 2u * group + (within < 32u ? 0u : 1u);
    uchar2 scale_min = q4k_scale_min(scales, sub_block);
    q4k_header header;
    header.scale = (float)as_type<half>(d_bits) * (float)scale_min.x;
    header.minimum = (float)as_type<half>(dmin_bits) * (float)scale_min.y;
    return header;
}

// one element, given its sub-block's already-decoded header. This is the
// whole per-element cost in the tiled loop: one byte load, one mask or
// shift, one fma.
static inline float q4k_value(device const uchar *block, uint index, q4k_header header) {
    device const uchar *qs = block + 16;
    uint group = index / 64u;
    uint within = index % 64u;
    uint byte_index = group * 32u + (within % 32u);
    uchar nibble = (within < 32u) ? (qs[byte_index] & 0x0F) : (qs[byte_index] >> 4);
    return header.scale * (float)nibble - header.minimum;
}

static inline float q4k_pair_dot(device const uchar *block, uint iq, uint ir, thread const float *yl, thread const float *yh) {
    device const uchar *qs = block + 16;
    // ggml's q1/q2 pointers are uint16_t, so its +32 offset advances 64 bytes.
    uint byte_base = 32u * iq + 8u * ir;
    // `qs` sits at offset 16 in a 144-byte block (both even), and `byte_base`
    // is always a multiple of 8, so every `ushort` word below is 2-byte
    // aligned -- matches ggml's `(device const uint16_t *)qs + 16*iq + 4*ir`.
    device const ushort *word_low = (device const ushort *)(qs + byte_base);
    device const ushort *word_high = (device const ushort *)(qs + byte_base + 64u);
    uint low_index = 64u * iq + 8u * ir;
    q4k_header h0 = q4k_header_for(block, low_index);
    q4k_header h1 = q4k_header_for(block, low_index + 32u);
    q4k_header h2 = q4k_header_for(block, low_index + 128u);
    q4k_header h3 = q4k_header_for(block, low_index + 160u);
    float result = 0.0f;
    for (uint i = 0u; i < 4u; ++i) {
        uint word1 = (uint)word_low[i];
        uint word2 = (uint)word_high[i];
        result += (h0.scale * (float)(word1 & 0x0Fu) - h0.minimum) * yl[2u * i + 0u];
        result += (h0.scale * (float)((word1 >> 8) & 0x0Fu) - h0.minimum) * yl[2u * i + 1u];
        result += (h1.scale * (float)((word1 >> 4) & 0x0Fu) - h1.minimum) * yl[2u * i + 8u];
        result += (h1.scale * (float)((word1 >> 12) & 0x0Fu) - h1.minimum) * yl[2u * i + 9u];
        result += (h2.scale * (float)(word2 & 0x0Fu) - h2.minimum) * yh[2u * i + 0u];
        result += (h2.scale * (float)((word2 >> 8) & 0x0Fu) - h2.minimum) * yh[2u * i + 1u];
        result += (h3.scale * (float)((word2 >> 4) & 0x0Fu) - h3.minimum) * yh[2u * i + 8u];
        result += (h3.scale * (float)((word2 >> 12) & 0x0Fu) - h3.minimum) * yh[2u * i + 9u];
    }
    return result;
}

// `q4k_pair_dot` generalized across an activation GROUP: `push_packed_row_
// multi_row_body`'s own per-element `operand_read` loop (`docs/discipline.md`
// ROW 389) decoded this row's header/nibble words once PER TOKEN instead of
// once per weight-block iteration, because it never shared this function's
// `word_low`/`word_high`/`h0..h3` decode across the group. `q4k_pair_dot_mr`
// pulls that decode out of the per-token loop: `word_low`/`word_high` and
// `h0..h3` are read/derived exactly once, then the eight-fma accumulate
// below runs once per token, gathering that ONE token's 16+16 `yl`/`yh`
// floats from `other_ptr` (via `other_base[s]`/`other_stride`/
// `other_ib_offset`) immediately before folding them.
//
// CORRECTION (perf/prefill-packed-row-decode): a first version of this
// function took pre-gathered `yl_group[cap][16]`/`yh_group[cap][16]` thread
// arrays instead -- 256 live private floats at `cap = 8` -- and measured
// SLOWER on a 31-token prefill than the generic per-element body it
// replaced (step-0 GPU 13.0s on main vs 15.1-21.8s on that version). Arrays
// sized by `cap` force Metal to reserve that much thread/private storage
// for the whole per-block-iteration accumulate regardless of how many of
// those `cap` slots are actually live at once, collapsing occupancy. This
// version stages one token's `yl`/`yh` at a time inside the `s` loop below,
// so live private state is ~40 floats (`yl`, `yh`, `result_s`, `result[]`)
// no matter how large `cap` gets -- at the cost of re-issuing `other_ptr`'s
// device loads once per output row (`rows_per_simdgroup`, 4 for `Q4_K`)
// instead of once shared across all four; each repeat reads the SAME
// address as the other three (same lane, same token, same sub-block
// offset), so it costs a cache hit, not additional HBM traffic.
static inline void q4k_pair_dot_mr(device const uchar *block, uint iq, uint ir, device const float *other_ptr, thread const long *other_base, long other_stride, long other_ib_offset, uint cap, thread float *result) {
    device const uchar *qs = block + 16;
    uint byte_base = 32u * iq + 8u * ir;
    device const ushort *word_low = (device const ushort *)(qs + byte_base);
    device const ushort *word_high = (device const ushort *)(qs + byte_base + 64u);
    uint low_index = 64u * iq + 8u * ir;
    q4k_header h0 = q4k_header_for(block, low_index);
    q4k_header h1 = q4k_header_for(block, low_index + 32u);
    q4k_header h2 = q4k_header_for(block, low_index + 128u);
    q4k_header h3 = q4k_header_for(block, low_index + 160u);
    ushort w1[4]; ushort w2[4];
    for (uint i = 0u; i < 4u; ++i) { w1[i] = word_low[i]; w2[i] = word_high[i]; }
    long lane_offset = (long)(64u * iq + 8u * ir) * other_stride;
    for (uint s = 0u; s < cap; ++s) {
        device const float *y4 = other_ptr + other_base[s] + other_ib_offset + lane_offset;
        float yl[16];
        float yh[16];
        for (uint i = 0u; i < 8u; ++i) {
            yl[i] = y4[(long)i * other_stride];
            yl[i + 8u] = y4[(long)(i + 32u) * other_stride];
            yh[i] = y4[(long)(i + 128u) * other_stride];
            yh[i + 8u] = y4[(long)(i + 160u) * other_stride];
        }
        float result_s = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            uint word1 = (uint)w1[i];
            uint word2 = (uint)w2[i];
            result_s += (h0.scale * (float)(word1 & 0x0Fu) - h0.minimum) * yl[2u * i + 0u];
            result_s += (h0.scale * (float)((word1 >> 8) & 0x0Fu) - h0.minimum) * yl[2u * i + 1u];
            result_s += (h1.scale * (float)((word1 >> 4) & 0x0Fu) - h1.minimum) * yl[2u * i + 8u];
            result_s += (h1.scale * (float)((word1 >> 12) & 0x0Fu) - h1.minimum) * yl[2u * i + 9u];
            result_s += (h2.scale * (float)(word2 & 0x0Fu) - h2.minimum) * yh[2u * i + 0u];
            result_s += (h2.scale * (float)((word2 >> 8) & 0x0Fu) - h2.minimum) * yh[2u * i + 1u];
            result_s += (h3.scale * (float)((word2 >> 4) & 0x0Fu) - h3.minimum) * yh[2u * i + 8u];
            result_s += (h3.scale * (float)((word2 >> 12) & 0x0Fu) - h3.minimum) * yh[2u * i + 9u];
        }
        result[s] = result_s;
    }
}

// Eight consecutive levels from TWO 32-bit loads instead of eight byte
// loads. A lane's run is `slot .. slot+7` and never crosses a 32-element
// sub-block boundary, so all eight share a group and a nibble half, and
// their bytes are eight CONSECUTIVE bytes of `qs`. `slot % 32` is one of
// {0,8,16,24} and a super-block is 144 bytes (a multiple of 16), so the
// address is 4-byte aligned and the `uint` cast is sound.
//
// CORRECTION (perf/metal-q4k-mask-fma): the claim this comment used to make
// here -- "ggml does the same thing one width down ... the nibble extract is
// cheap and the LOAD is what costs" -- is FALSE, verified against
// `ggml-metal.metal:5157-5165` (`kernel_mul_mv_q4_K_f32_impl`). ggml does NOT
// shift at all: it masks four FIXED bit positions (`q1[i] & 0x000F/0x0F00/
// 0x00F0/0xF000`) straight off a `uint16_t` load and folds the resulting
// 1x/16x/256x residual scale into the per-sub-block combine
// (`ggml-metal.metal:5171-5175`). This function instead computes a RUNTIME
// `shift` (0 or 4, not known at compile time) and adds it into every one of
// the eight extractions below, which ggml's masked form has no equivalent
// of. See `push_q4k_product_reduce_body`'s `metal-q4k-mask-fma` arm (Rust,
// not MSL -- it generates the masked loop inline so it stays generic over
// this kernel's `element_type`) for the mask-without-shift port of ggml's
// actual technique, adapted to this function's one-nibble-per-byte layout
// (ggml's is two-nibbles-per-byte interleaved across two sub-blocks per
// load, a different packing this file's lane mapping does not share).
static inline void q4k_run8(device const uchar *block, uint index, thread float *out) {
    device const uchar *qs = block + 16;
    uint group = index / 64u;
    uint within = index % 64u;
    uint byte_index = group * 32u + (within % 32u);
    device const uint *words = (device const uint *)(qs + byte_index);
    uint w0 = words[0];
    uint w1 = words[1];
    uint shift = (within < 32u) ? 0u : 4u;
    out[0] = (float)((w0 >> (shift +  0u)) & 0xFu);
    out[1] = (float)((w0 >> (shift +  8u)) & 0xFu);
    out[2] = (float)((w0 >> (shift + 16u)) & 0xFu);
    out[3] = (float)((w0 >> (shift + 24u)) & 0xFu);
    out[4] = (float)((w1 >> (shift +  0u)) & 0xFu);
    out[5] = (float)((w1 >> (shift +  8u)) & 0xFu);
    out[6] = (float)((w1 >> (shift + 16u)) & 0xFu);
    out[7] = (float)((w1 >> (shift + 24u)) & 0xFu);
}

// `metal-q4k-single-fetch` (opt-in, see `push_q4k_single_fetch_body`): ONE
// 8-byte (two-word) load, BOTH nibble halves of it -- eight low-nibble
// levels (elements `low_index .. low_index+8`) AND the eight high-nibble
// levels of the SAME bytes (elements `low_index+32 .. low_index+40`, ggml's
// and `dequantize_row_q4_K`'s own "32 elements apart, same byte" pairing —
// see `q4_k.rs::dequantize_block`'s doc). `q4k_run8` above issues this exact
// load TWICE for the pair of lanes that owns a byte range (once per nibble
// half); this issues it ONCE and reads out both halves, which is the whole
// point of the feature. `low_index` must be a "low" index (`index % 64 <
// 32`) — callers only ever pass one of `sf_low_base + c*8`.
static inline void q4k_run8_dual(
    device const uchar *block,
    uint low_index,
    thread float *out_low,
    thread float *out_high
) {
    device const uchar *qs = block + 16;
    uint group = low_index / 64u;
    uint within = low_index % 64u;
    uint byte_index = group * 32u + within;
    device const uint *words = (device const uint *)(qs + byte_index);
    uint w0 = words[0];
    uint w1 = words[1];
    out_low[0] = (float)((w0 >>  0u) & 0xFu);
    out_low[1] = (float)((w0 >>  8u) & 0xFu);
    out_low[2] = (float)((w0 >> 16u) & 0xFu);
    out_low[3] = (float)((w0 >> 24u) & 0xFu);
    out_low[4] = (float)((w1 >>  0u) & 0xFu);
    out_low[5] = (float)((w1 >>  8u) & 0xFu);
    out_low[6] = (float)((w1 >> 16u) & 0xFu);
    out_low[7] = (float)((w1 >> 24u) & 0xFu);
    out_high[0] = (float)((w0 >>  4u) & 0xFu);
    out_high[1] = (float)((w0 >> 12u) & 0xFu);
    out_high[2] = (float)((w0 >> 20u) & 0xFu);
    out_high[3] = (float)((w0 >> 28u) & 0xFu);
    out_high[4] = (float)((w1 >>  4u) & 0xFu);
    out_high[5] = (float)((w1 >> 12u) & 0xFu);
    out_high[6] = (float)((w1 >> 20u) & 0xFu);
    out_high[7] = (float)((w1 >> 28u) & 0xFu);
}
"#;

/// `metal-q4k-mask-fma` (default-off): mask-without-shift ports of
/// `q4k_scale_min` and `q4k_run8`, onto ggml's actual technique
/// (`ggml-metal.metal:5096-5098,5157-5175`) rather than the shift-then-mask
/// scheme those two functions use. See the correction on `q4k_run8`'s own
/// doc comment for why that scheme was previously (falsely) attributed to
/// ggml, and `push_q4k_product_reduce_body`'s feature-gated arm (Rust, the
/// caller) for where the masked accumulate itself is generated -- inlined
/// directly rather than through a shared MSL function, so it can stay typed
/// to the kernel's own `element_type` (`half` or `float`) the way
/// `q4k_run8`'s callers already do.
///
/// Only the two functions used OUTSIDE the row-blocked accumulate loop live
/// here as shared MSL text: `q4k_scale_min_bf` (called once per header
/// decode, fixed `float`/`uchar` types regardless of kernel element type)
/// and `q4k_header_for_bf` (its caller, `push_q4k_header_decode`'s
/// feature-gated arm). Everything type-dependent is generated inline by
/// `push_q4k_product_reduce_body` itself.
#[cfg(feature = "metal-q4k-mask-fma")]
pub const Q4K_MASK_FMA_MSL: &str = r#"
// Branch-free port of ggml's kmask1/kmask2/kmask3 scale/min unpack
// (`ggml-metal.metal:5096-5098`), adapted to `scales` being indexed per
// BYTE here (`q4k_scale_min`'s own layout) rather than per `uint16_t` pair
// the way ggml reads it. `sub_block & 3u` is the SAME index
// `q4k_scale_min` reads whether `sub_block` names the low half (0..3, used
// directly) or the high half (4..7, used as `sub_block - 4`) --
// `x & 3u == x` for `x < 4` and `== x - 4` for `4 <= x < 8`. Both the
// low-half and high-half formulas are computed UNCONDITIONALLY and the
// result selected, never branched on -- `q4k_scale_min`'s
// `if (sub_block < 4u) { return ...; }` compiles to a real divergent
// branch taken by 4 of every 8 lanes, every call; this compiles to a
// `select`.
static inline uchar2 q4k_scale_min_bf(device const uchar *scales, uint sub_block) {
    uint low4 = sub_block & 3u;
    uchar byte_a = scales[low4];
    uchar byte_b = scales[low4 + 4u];
    uchar byte_c = scales[sub_block + 4u];
    bool hi = sub_block >= 4u;
    uchar lo_scale = byte_a & 63u;
    uchar lo_min = byte_b & 63u;
    uchar hi_scale = (byte_c & 0x0Fu) | ((byte_a >> 6u) << 4u);
    uchar hi_min = (byte_c >> 4u) | ((byte_b >> 6u) << 4u);
    return uchar2(hi ? hi_scale : lo_scale, hi ? hi_min : lo_min);
}

// same per-sub-block amortization `q4k_header_for` makes, over
// `q4k_scale_min_bf` instead of the branchy `q4k_scale_min`.
static inline q4k_header q4k_header_for_bf(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    ushort dmin_bits = (ushort)((uint)block[2] | ((uint)block[3] << 8));
    device const uchar *scales = block + 4;
    uint group = index / 64u;
    uint within = index % 64u;
    uint sub_block = 2u * group + (within < 32u ? 0u : 1u);
    uchar2 scale_min = q4k_scale_min_bf(scales, sub_block);
    q4k_header header;
    header.scale = (float)as_type<half>(d_bits) * (float)scale_min.x;
    header.minimum = (float)as_type<half>(dmin_bits) * (float)scale_min.y;
    return header;
}
"#;

/// Bytes one `Q4_K` super-block occupies, and elements it carries — the two
/// numbers a caller needs to index a packed weight row. Read from
/// `proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K}`, the one place this
/// codec's block geometry is defined.
pub const Q4K_BLOCK_BYTES: usize = proxima_gguf::quant::q4_k::BLOCK_BYTES;
/// Elements one `Q4_K` super-block carries. Shared by `Q5_K`/`Q6_K` too —
/// the whole K-quant super-block family is 256 elements wide
/// (`proxima-tensor/src/cpu.rs`'s own doc on its `Q6K_BLOCK_BYTES` makes the
/// same point); only the packed BYTE width differs per codec.
pub const Q4K_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_k::QK_K;

/// MSL source for unpacking one element of a `Q6_K` super-block. Ports
/// `proxima_gguf::quant::q6_k::dequantize_block`/`unpack_levels` exactly:
/// two 128-element halves, each split into four 32-wide lanes sharing one
/// `qh` byte per lane position (2 bits each), `ql`'s low/high nibble shared
/// between lanes 0/2 (`ql[l]`) and 1/3 (`ql[l+32]`), one signed 8-bit
/// sub-block scale (`x = d*sc*(level-32)`, no `dmin` term at all — a
/// genuinely different shape from `Q4_K`/`Q5_K`, not a small variation), and
/// `d` trailing the block (offset 208) rather than leading it.
///
/// Layout, 210 bytes per 256 elements: 128 bytes `ql` at 0, 64 bytes `qh` at
/// 128, 16 signed scale bytes at 192, `d` f16 at 208.
pub const Q6K_UNPACK_MSL: &str = r#"
// one Q6_K super-block's scale `d` -- decoded ONCE per super-block by the
// row-blocked path (see push_packed_row_blocked_body), since it is constant
// across all 256 elements (unlike Q4_K's per-sub-block header).
struct q6k_header { float d; };

static inline q6k_header q6k_header_for(device const uchar *block) {
    ushort d_bits = (ushort)((uint)block[208] | ((uint)block[209] << 8));
    q6k_header header;
    header.d = (float)as_type<half>(d_bits);
    return header;
}

// element `index` (0..256) of one Q6_K super-block, given its super-block's
// already-decoded `d` -- byte-for-byte the value
// proxima_gguf::quant::q6_k::dequantize_block writes at the same index.
static inline float q6k_value(device const uchar *block, uint index, q6k_header header) {
    uint half_index = index / 128u;
    uint local = index % 128u;
    uint l = local % 32u;
    uint lane = local / 32u;
    uint sub_block_in_half = l / 16u;

    device const uchar *ql = block + half_index * 64u;
    device const uchar *qh = block + 128u + half_index * 32u;
    device const uchar *scales = block + 192u;

    uchar ql_byte = (lane % 2u == 0u) ? ql[l] : ql[l + 32u];
    uchar nibble = (lane < 2u) ? (ql_byte & 0x0Fu) : (ql_byte >> 4u);
    uchar high2 = (qh[l] >> (uchar)(lane * 2u)) & 0x03u;
    uchar level = nibble | (high2 << 4u);

    uchar scale_byte = scales[half_index * 8u + sub_block_in_half + lane * 2u];
    float scale = (float)(char)scale_byte;
    float quant = (float)level - 32.0f;
    return header.d * scale * quant;
}

// element `index` of one Q6_K super-block, decoding its own header first --
// the generic per-element path (`operand_read`'s non-row-blocked callers)
// has no amortized header to reuse across elements, unlike the row-blocked
// path's `q6k_header_for` decoded once per super-block.
static inline float q6k_element(device const uchar *block, uint index) {
    return q6k_value(block, index, q6k_header_for(block));
}
"#;

/// Bytes one `Q6_K` super-block occupies -- read from
/// `proxima_gguf::quant::q6_k::BLOCK_BYTES`; pinned in
/// `omega/tests/q6k_unpack.rs`, same posture as [`Q4K_BLOCK_BYTES`].
pub const Q6K_BLOCK_BYTES: usize = proxima_gguf::quant::q6_k::BLOCK_BYTES;

/// `Q6_K`'s counterpart to `q4k_pair_dot`/`q5k_pair_dot`, selected by
/// `PackedCodec::supports_pair_dot` rather than a cargo feature -- routed
/// the same way, into the SAME codec-agnostic `yl`/`yh` activation gather
/// `push_packed_row_blocked_body`'s `plain_product` preamble already builds
/// for `Q4_K`/`Q5_K`. The lane
/// assignment (`iq = it/4`, `ir = it%4`) is unchanged from those two, so a
/// lane's 32 owned elements decompose into the same four 8-element groups
/// (`base+[0..7]`, `base+32+[0..7]`, `base+128+[0..7]`, `base+160+[0..7]`
/// with `base = 64*iq + 8*ir`) that feed `yl[0..7]`/`yl[8..15]`/
/// `yh[0..7]`/`yh[8..15]` respectively -- verified against `q6k_value`'s own
/// `half_index`/`local`/`l`/`lane`/`sub_block_in_half` derivation: group A
/// (`base+i`) and group C (`base+128+i`) share `lane = 2*iq` and `l =
/// 8*ir+i`, differing only in `half_index` (0 vs 1); group B/D share
/// `lane = 2*iq+1` with the same `l`. That gives four fixed (`half_index`,
/// `lane`) headers -- one signed `char` scale byte each, mirroring
/// `q4k_pair_dot`'s four `q4k_header`s -- and `d` is decoded once per
/// super-block, not per group, the same amortization `q4k_pair_dot`'s
/// `q4k_header_for` calls give `Q4_K`.
///
/// UNLIKE `q4k_pair_dot` (144-byte block, mult of 16) and `q5k_pair_dot`
/// (176-byte block, mult of 16, hence its `ulong` 8-byte loads), `Q6_K`'s
/// 210-byte block is NOT a multiple of 4 -- consecutive super-blocks'
/// addresses only preserve 2-byte alignment (`210` is even, so a `ql`/`qh`
/// byte offset that is itself even keeps `ushort` loads sound; a `uint` or
/// `ulong` load would straddle an odd 4-byte boundary on every other
/// super-block). Every wide load below is a `ushort` for exactly this
/// reason -- the same width `q4k_pair_dot` uses, but here it is the WIDEST
/// safe load rather than a stylistic choice.
///
/// Mirrors ggml's `kernel_mul_mv_q6_K_f32_impl`
/// (`ggml-metal.metal:5360-5420`, MIT-licensed) mask/shift technique
/// (`kmask1..4`, `(q1[l] & 0xF) | ((qh[l] & kmask) << shift)`) rather than
/// `q6k_value`'s own shift-then-mask-then-OR order, fused with the
/// accumulate the same non-deferred way `q4k_pair_dot`/`q5k_pair_dot`
/// already fold `scale*level` into each element's product immediately (no
/// `Q4_K`-style `dmin` term to defer here at all -- `Q6_K` has none).
pub const Q6K_PAIR_DOT_MSL: &str = r#"
static inline float q6k_pair_dot(device const uchar *block, uint iq, uint ir, thread const float *yl, thread const float *yh) {
    device const uchar *ql = block;
    device const uchar *qh = block + 128u;
    device const uchar *scales = block + 192u;
    q6k_header hdr = q6k_header_for(block);

    uint l_base = 8u * ir;
    // `l_base` (0/8/16/24) and its +32/+64/+96 siblings are all even, and
    // `block` itself always lands on an even byte (210-byte stride, always
    // even) -- so every `ushort` load below is 2-byte aligned. A `uint`
    // load would need `l_base` 4-byte aligned AND `block` 4-byte aligned;
    // neither holds for every super-block index, so `ushort` is the
    // widest load this layout can support unconditionally (see this
    // constant's own doc).
    device const ushort *ql_a = (device const ushort *)(ql + l_base);
    device const ushort *ql_b = (device const ushort *)(ql + 32u + l_base);
    device const ushort *ql_c = (device const ushort *)(ql + 64u + l_base);
    device const ushort *ql_d = (device const ushort *)(ql + 96u + l_base);
    device const ushort *qh_0 = (device const ushort *)(qh + l_base);
    device const ushort *qh_1 = (device const ushort *)(qh + 32u + l_base);

    uint sub = (l_base < 16u) ? 0u : 1u;
    float scale_a = (float)(char)scales[sub + 4u * iq];
    float scale_b = (float)(char)scales[sub + 4u * iq + 2u];
    float scale_c = (float)(char)scales[8u + sub + 4u * iq];
    float scale_d = (float)(char)scales[8u + sub + 4u * iq + 2u];

    uint shift_lo = 4u * iq;
    uint shift_hi = shift_lo + 2u;
    uchar nibble_shift = (iq == 0u) ? 0u : 4u;

    float result = 0.0f;
    for (uint w = 0u; w < 4u; ++w) {
        ushort ql_a_word = ql_a[w];
        ushort ql_b_word = ql_b[w];
        ushort ql_c_word = ql_c[w];
        ushort ql_d_word = ql_d[w];
        ushort qh_0_word = qh_0[w];
        ushort qh_1_word = qh_1[w];
        for (uint bshift = 0u; bshift < 16u; bshift += 8u) {
            uint i = 2u * w + (bshift / 8u);
            uchar ql_a_byte = (uchar)((ql_a_word >> bshift) & 0xFFu);
            uchar ql_b_byte = (uchar)((ql_b_word >> bshift) & 0xFFu);
            uchar ql_c_byte = (uchar)((ql_c_word >> bshift) & 0xFFu);
            uchar ql_d_byte = (uchar)((ql_d_word >> bshift) & 0xFFu);
            uchar qh_0_byte = (uchar)((qh_0_word >> bshift) & 0xFFu);
            uchar qh_1_byte = (uchar)((qh_1_word >> bshift) & 0xFFu);

            float level_a = (float)(((ql_a_byte >> nibble_shift) & 0x0Fu) | (((qh_0_byte >> shift_lo) & 0x03u) << 4u)) - 32.0f;
            float level_b = (float)(((ql_b_byte >> nibble_shift) & 0x0Fu) | (((qh_0_byte >> shift_hi) & 0x03u) << 4u)) - 32.0f;
            float level_c = (float)(((ql_c_byte >> nibble_shift) & 0x0Fu) | (((qh_1_byte >> shift_lo) & 0x03u) << 4u)) - 32.0f;
            float level_d = (float)(((ql_d_byte >> nibble_shift) & 0x0Fu) | (((qh_1_byte >> shift_hi) & 0x03u) << 4u)) - 32.0f;

            result += (hdr.d * scale_a * level_a) * yl[i];
            result += (hdr.d * scale_b * level_b) * yl[i + 8u];
            result += (hdr.d * scale_c * level_c) * yh[i];
            result += (hdr.d * scale_d * level_d) * yh[i + 8u];
        }
    }
    return result;
}
"#;

/// MSL source for unpacking one element of a `Q5_K` super-block. Ports
/// `proxima_gguf::quant::q5_k::dequantize_block`/`get_scale_min_k4`
/// exactly: the SAME super-block/sub-block shape and SAME bit-interleaved
/// `(scale, min)` packing as `Q4_K` (`q5k_scale_min` below is
/// `q4k_scale_min` unchanged, restated rather than shared -- see this
/// crate's own per-codec duplication precedent in `proxima_gguf::quant`),
/// plus a `qh` high-bit plane `Q4_K` does not have: each element's 5-bit
/// level is a `qs` nibble OR'd with one bit of `qh[offset]`, selected by a
/// mask that depends on which of the four 64-element chunks the element
/// falls in (`mask = 1 << (2*chunk)` for the chunk's low half, `2 <<
/// (2*chunk)` for its high half) -- genuinely a third bit layout, not a
/// `Q4_K` widening or a `Q6_K` narrowing, matching this landing's own
/// sizing note on why it needed its own kernel.
///
/// Layout, 176 bytes per 256 elements: `d` f16 at 0, `dmin` f16 at 2, 12
/// packed scale/min bytes at 4, 32 `qh` bytes at 16, 128 nibble bytes at 48.
pub const Q5K_UNPACK_MSL: &str = r#"
// ports proxima_gguf::quant::q5_k::get_scale_min_k4 -- byte-for-byte the
// same function q4k_scale_min above computes, restated here rather than
// shared (see this constant's own doc).
static inline uchar2 q5k_scale_min(device const uchar *scales, uint sub_block) {
    if (sub_block < 4u) {
        return uchar2(scales[sub_block] & 63, scales[sub_block + 4u] & 63);
    }
    uchar scale = (scales[sub_block + 4u] & 0x0F) | ((scales[sub_block - 4u] >> 6) << 4);
    uchar minimum = (scales[sub_block + 4u] >> 4) | ((scales[sub_block] >> 6) << 4);
    return uchar2(scale, minimum);
}

// one Q5_K super-block's per-sub-block scale, min, and high-bit MASK,
// decoded ONCE for a run of elements inside that sub-block -- the same
// amortization q4k_header_for makes, widened to also carry which `qh` bit
// this sub-block's elements read (constant across the whole 32-element
// sub-block: `chunk` and "low or high half" are both fixed for it).
struct q5k_header { float scale; float minimum; uchar mask; };

static inline q5k_header q5k_header_for(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    ushort dmin_bits = (ushort)((uint)block[2] | ((uint)block[3] << 8));
    device const uchar *scales = block + 4;

    uint chunk = index / 64u;
    uint within = index % 64u;
    bool low = within < 32u;
    uint sub_block = 2u * chunk + (low ? 0u : 1u);

    uchar2 scale_min = q5k_scale_min(scales, sub_block);
    q5k_header header;
    header.scale = (float)as_type<half>(d_bits) * (float)scale_min.x;
    header.minimum = (float)as_type<half>(dmin_bits) * (float)scale_min.y;
    header.mask = low ? (uchar)(1u << (2u * chunk)) : (uchar)(2u << (2u * chunk));
    return header;
}

// one element, given its sub-block's already-decoded header. `qh` is
// indexed by `offset` (0..32) alone, never by `chunk` -- the same
// within-sub-block-position indexing `dequantize_block`'s own doc calls
// out as `Q5_K`'s "easiest to get silently wrong" trap: two elements in
// DIFFERENT chunks but the SAME local offset read different BITS of the
// SAME `qh` byte (the header's own `mask` is what picks the right bit).
static inline float q5k_value(device const uchar *block, uint index, q5k_header header) {
    device const uchar *qh = block + 16;
    device const uchar *qs = block + 48;

    uint chunk = index / 64u;
    uint within = index % 64u;
    bool low = within < 32u;
    uint offset = within % 32u;

    uchar qs_byte = qs[chunk * 32u + offset];
    uchar nibble = low ? (qs_byte & 0x0Fu) : (qs_byte >> 4u);
    float high = (qh[offset] & header.mask) != 0u ? 16.0f : 0.0f;
    return header.scale * ((float)nibble + high) - header.minimum;
}

// element `index` of one Q5_K super-block, decoding its own header first --
// the generic per-element path (`operand_read`'s non-row-blocked callers)
// has no amortized header to reuse across elements, unlike the row-blocked
// path's `q5k_header_for` decoded once per sub-block.
static inline float q5k_element(device const uchar *block, uint index) {
    return q5k_value(block, index, q5k_header_for(block, index));
}
"#;

/// Bytes one `Q5_K` super-block occupies -- read from
/// `proxima_gguf::quant::q5_k::BLOCK_BYTES`; pinned in
/// `omega/tests/q5k_unpack.rs`, same posture as [`Q4K_BLOCK_BYTES`].
pub const Q5K_BLOCK_BYTES: usize = proxima_gguf::quant::q5_k::BLOCK_BYTES;

/// The same paired-nibble, packed-word-load body `q4k_pair_dot` gives
/// `Q4_K`'s `plain_product` arm, selected by
/// `PackedCodec::supports_pair_dot` rather than a cargo feature
/// (`push_packed_row_blocked_body`) -- one `ulong` load per 8-byte `qs`/`qh`
/// run, byte-extracted by shift rather than eight scalar `uchar` loads --
/// extended with `Q5_K`'s `qh` high-bit plane -- ONE extra mask-select per
/// pair of levels, no shift on the nibble itself,
/// mirroring llama.cpp's own `kernel_mul_mv_q5_K_f32_impl`
/// (`ggml-metal.metal:5209-5324`) lane assignment (`tid = tiisg/4`,
/// `ix = tiisg%4`, `iq = tid/4`, `ir = tid%4`, `l0 = 8*ir`) exactly: `iq`/`ir`
/// select the same `qs` byte range `q4k_pair_dot` reads (`32*iq + 8*ir`), and
/// `qh` is indexed by `8*ir + l` alone -- independent of `iq` -- with the
/// high bit selected by one of four fixed masks (`1<<2*iq`, that value
/// shifted left by 1 for the low sub-block's high nibble, and both again
/// shifted left by 4 for the "second half" pair of sub-blocks 128/160
/// elements on), matching this crate's own `q5k_header_for`'s `mask`
/// derivation (`low ? 1<<(2*chunk) : 2<<(2*chunk)`) rather than deferring the
/// scale the way ggml's `acc1`/`acc2` split does: each level's nibble and
/// high bit are folded into one `scale*(nibble+high)-minimum` term per
/// element, immediately, the same non-deferred style `q4k_pair_dot` itself
/// already uses (see `q5k_header_for`, restated per-pair here rather than
/// shared -- same posture as `q5k_scale_min` restating `q4k_scale_min`).
pub const Q5K_PAIR_DOT_MSL: &str = r#"
static inline float q5k_pair_dot(device const uchar *block, uint iq, uint ir, thread const float *yl, thread const float *yh) {
    device const uchar *qh = block + 16;
    device const uchar *qs = block + 48;
    uint byte_base = 32u * iq + 8u * ir;
    // `qs`/`qh` start at offsets 48/16 in a 176-byte block (all multiples of
    // 8), and `byte_base`/`8*ir` are themselves multiples of 8, so each
    // `ulong` below reads its 8-byte run of `l` in one 8-byte-aligned load
    // instead of eight scalar `uchar` loads.
    ulong q1_word = *(device const ulong *)(qs + byte_base);
    ulong q2_word = *(device const ulong *)(qs + byte_base + 64u);
    ulong h_word = *(device const ulong *)(qh + 8u * ir);
    uint low_index = 64u * iq + 8u * ir;
    q5k_header h0 = q5k_header_for(block, low_index);
    q5k_header h1 = q5k_header_for(block, low_index + 32u);
    q5k_header h2 = q5k_header_for(block, low_index + 128u);
    q5k_header h3 = q5k_header_for(block, low_index + 160u);
    uchar hm1 = (uchar)(1u << (2u * iq));
    uchar hm2 = (uchar)(hm1 << 1u);
    uchar hm3 = (uchar)(hm1 << 4u);
    uchar hm4 = (uchar)(hm2 << 4u);
    float result = 0.0f;
    for (uint l = 0u; l < 8u; ++l) {
        uchar q1 = (uchar)((q1_word >> (8u * l)) & 0xFFu);
        uchar q2 = (uchar)((q2_word >> (8u * l)) & 0xFFu);
        uchar h = (uchar)((h_word >> (8u * l)) & 0xFFu);
        float low0 = (float)(q1 & 0x0Fu) + ((h & hm1) != 0u ? 16.0f : 0.0f);
        float low1 = (float)(q1 >> 4u) + ((h & hm2) != 0u ? 16.0f : 0.0f);
        float high0 = (float)(q2 & 0x0Fu) + ((h & hm3) != 0u ? 16.0f : 0.0f);
        float high1 = (float)(q2 >> 4u) + ((h & hm4) != 0u ? 16.0f : 0.0f);
        result += (h0.scale * low0 - h0.minimum) * yl[l];
        result += (h1.scale * low1 - h1.minimum) * yl[l + 8u];
        result += (h2.scale * high0 - h2.minimum) * yh[l];
        result += (h3.scale * high1 - h3.minimum) * yh[l + 8u];
    }
    return result;
}
"#;

/// `Q8_0`: a flat 32-element block, one `f16` scale, no sub-block scale/min
/// pair and no bit-packing at all -- genuinely a different SHAPE from the
/// K-quant family above (no super-block; each level is already a full signed
/// byte, not a nibble), not a widening or narrowing of one. It slots into
/// the same PACKED-OPERAND mechanism ([`PackedCodec`], `operand_read`,
/// this preamble) as a fourth codec precisely because that mechanism is
/// generic over block byte width and element count; it does NOT take the
/// row-blocked (`classify_packed_row_block`) or tiled-GEMM
/// (`classify_tiled_gemm`) fast paths, both of which hard-code the
/// K-quants' shared 256-element super-block and 8-lane amortization scheme
/// this codec has no analogue for -- `Q8_0` always renders through the fully
/// generic per-element accessor below, same as any codec those two paths
/// reject.
///
/// Ports `proxima_gguf::quant::q8_0::dequantize_block` exactly: `x = q*d`
/// per element, no sub-block structure at all.
///
/// Layout, 34 bytes per 32 elements: `d` f16 at 0, 32 signed `int8_t` levels
/// at 2.
pub const Q8_0_UNPACK_MSL: &str = r#"
// element `index` (0..32) of one Q8_0 block, byte-for-byte the value
// proxima_gguf::quant::q8_0::dequantize_block writes at the same index.
static inline float q8_0_element(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    float d = (float)as_type<half>(d_bits);
    char level = (char)block[2u + index];
    return (float)level * d;
}
"#;

/// Bytes one `Q8_0` block occupies -- read from
/// `proxima_gguf::quant::q8_0::BLOCK_BYTES`; pinned in
/// `omega/tests/q8_0_unpack.rs`, same posture as [`Q4K_BLOCK_BYTES`].
pub const Q8_0_BLOCK_BYTES: usize = proxima_gguf::quant::q8_0::BLOCK_BYTES;

/// Elements one `Q8_0` block carries -- 32, NOT [`Q4K_BLOCK_ELEMENTS`]'s 256:
/// `Q8_0` has no super-block structure, so it does not share the K-quant
/// family's element count. Read from `proxima_gguf::quant::q8_0::QK8_0`.
pub const Q8_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q8_0::QK8_0;

/// `Q4_0`: llama.cpp's simplest and most widely distributed 4-bit legacy
/// format -- a flat 32-element block, one `f16` scale, no sub-block
/// scale/min pair (unlike `Q4_K`) and no second `min` field (unlike
/// `Q4_1`). Same KIND-difference from the K-quant family that `Q8_0`'s
/// own doc draws: no super-block, so this codec does not take the
/// row-blocked (`classify_packed_row_block`) or tiled-GEMM
/// (`classify_tiled_gemm`) fast paths either -- it always renders through
/// the fully generic per-element accessor below.
///
/// Ports `proxima_gguf::quant::q4_0::dequantize_block` exactly: each packed
/// byte carries two 4-bit levels, `value = (nibble - 8) * d`.
///
/// Layout, 18 bytes per 32 elements: `d` f16 at 0, 16 packed-nibble bytes
/// at 2 (element `j`'s low nibble, element `16 + j`'s high nibble).
pub const Q4_0_UNPACK_MSL: &str = r#"
// element `index` (0..32) of one Q4_0 block, byte-for-byte the value
// proxima_gguf::quant::q4_0::dequantize_block writes at the same index.
static inline float q4_0_element(device const uchar *block, uint index) {
    ushort d_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));
    float d = (float)as_type<half>(d_bits);
    uchar byte = block[2u + (index % 16u)];
    int nibble = (index < 16u) ? (int)(byte & 0x0Fu) : (int)(byte >> 4u);
    return (float)(nibble - 8) * d;
}
"#;

/// Bytes one `Q4_0` block occupies -- read from
/// `proxima_gguf::quant::q4_0::BLOCK_BYTES`; pinned in
/// `omega/tests/q4_0_unpack.rs`, same posture as [`Q8_0_BLOCK_BYTES`].
pub const Q4_0_BLOCK_BYTES: usize = proxima_gguf::quant::q4_0::BLOCK_BYTES;

/// Elements one `Q4_0` block carries -- 32, the same flat block width as
/// [`Q8_0_BLOCK_ELEMENTS`], NOT [`Q4K_BLOCK_ELEMENTS`]'s 256. Read from
/// `proxima_gguf::quant::q4_0::QK4_0`.
pub const Q4_0_BLOCK_ELEMENTS: usize = proxima_gguf::quant::q4_0::QK4_0;

/// `Float16`: not a quantization at all -- MSL's `half` is IEEE-754 binary16
/// natively, so a `Float16` weight's bytes ARE a valid `half` buffer with no
/// unpack function required. It still needs a [`PackedCodec`] slot (rather
/// than folding into the plain `operand_read`'s `None` arm) because the
/// buffer must bind as `device const half*`, not whatever `float`/`half`
/// `type_token` chose for the KERNEL's own accumulator dtype -- a router
/// weight (`Float16`) multiplied against an `f32` activation is exactly the
/// mixed-dtype case `type_token` never had to handle before this codec.
/// Same non-K-quant, flat, one-element-per-block shape `Q8_0`/`Q4_0`
/// take: never the row-blocked (`classify_packed_row_block`) or tiled-GEMM
/// path, always the generic per-element accessor.
pub const FLOAT16_BLOCK_BYTES: usize = proxima_gguf::quant::f16::BLOCK_BYTES;

/// One `Float16` block is one element -- there is no super-block or
/// sub-block structure to amortize over, unlike every K-quant codec. Read
/// from `proxima_gguf::quant::f16::QK_F16`.
pub const FLOAT16_BLOCK_ELEMENTS: usize = proxima_gguf::quant::f16::QK_F16;

/// `BFloat16`: unlike `Float16`/[`FLOAT16_BLOCK_BYTES`], MSL has no
/// native `bfloat` storage type on this driver's baseline toolchain, so a
/// `BFloat16` weight DOES need an unpack function -- [`BF16_UNPACK_MSL`]'s
/// widen-by-shift, not a bit-packed dequantize. `bfloat16` is the top 16
/// bits of an `f32` (1 sign + 8 exponent + 7 mantissa, IEEE binary32's
/// exponent width exactly), so reconstructing the `f32` is `bits << 16`
/// reinterpreted, no rounding or lookup table involved.
pub const BFLOAT16_BLOCK_BYTES: usize = proxima_gguf::quant::bf16::BLOCK_BYTES;

/// Same one-element-per-block shape as [`FLOAT16_BLOCK_ELEMENTS`]. Read
/// from `proxima_gguf::quant::bf16::QK_BF16`.
pub const BFLOAT16_BLOCK_ELEMENTS: usize = proxima_gguf::quant::bf16::QK_BF16;

/// Widens one `bfloat16` element (2 little-endian bytes, the top half of an
/// `f32`) to `float` by shifting it into the high 16 bits of a 32-bit word
/// and reinterpreting -- the exact inverse of truncating an `f32`'s mantissa
/// to 7 bits, no rounding needed since every bit already present is kept
/// unchanged. `index` is threaded for the same call shape every other
/// `*_element` accessor takes (`operand_read`'s block-offset arithmetic
/// already folds a `BFloat16` weight's element index into the pointer, so
/// this function always reads at `index == 0`), even though a flat
/// one-element block never uses it for addressing.
pub const BF16_UNPACK_MSL: &str = r#"
// element `index` (always 0 -- BFloat16 has no super-block) of one BFloat16
// block: the top 16 bits of the f32 this pair of bytes came from,
// reconstructed by shifting them back into place.
static inline float bf16_element(device const uchar *block, uint index) {
    (void)index;
    uint bits = ((uint)block[0] | ((uint)block[1] << 8)) << 16u;
    return as_type<float>(bits);
}
"#;

/// Which packed codec one operand's bytes are — the second axis [`emit`]
/// needs alongside "is this operand packed at all" (a plain `bool` cannot
/// distinguish `Q4_K`'s 144-byte super-block from `Q6_K`'s 210-byte one, or
/// which unpack function reads it). `Copy`/`Eq` so it can sit directly in
/// the `quantized` slice every render function already threads through,
/// with no allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// the ggml/gguf ecosystem's own name is `Q8_0` everywhere else this codec
// appears (`GgmlType::Q8_0`, `QuantizedBlock::Q8_0`) -- the K-quant variants
// dropped their underscore for a terser name, but there is no "Q80" spelling
// anyone else uses, so this one variant keeps it instead of drifting from
// its own wire name.
#[allow(non_camel_case_types)]
pub enum PackedCodec {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    /// Flat 32-element block, no super-block structure — see
    /// [`Q8_0_UNPACK_MSL`]'s own doc for how this differs in KIND from the
    /// three K-quants above, not just in size.
    Q8_0,
    /// Flat 32-element block, one `f16` scale, no sub-block structure --
    /// llama.cpp's simplest legacy 4-bit format. Same KIND-difference from
    /// the K-quant family as [`Self::Q8_0`]; see [`Q4_0_UNPACK_MSL`]'s own
    /// doc.
    Q4_0,
    /// Not a quantization: `half`-native bytes, read directly through a
    /// `device const half*` binding, no unpack function -- see
    /// [`FLOAT16_BLOCK_BYTES`]'s own doc for why this still needs a codec
    /// slot despite there being nothing to decode.
    Float16,
    /// Needs a real unpack (widen-by-shift, [`BF16_UNPACK_MSL`]) since MSL
    /// has no native `bfloat` storage type -- see [`BFLOAT16_BLOCK_BYTES`]'s
    /// own doc.
    BFloat16,
}

impl PackedCodec {
    pub(crate) const fn cache_token(self) -> &'static str {
        match self {
            PackedCodec::Q2K => "q2k",
            PackedCodec::Q3K => "q3k",
            PackedCodec::Q4K => "q4k",
            PackedCodec::Q5K => "q5k",
            PackedCodec::Q6K => "q6k",
            PackedCodec::Q8_0 => "q8_0",
            PackedCodec::Q4_0 => "q4_0",
            PackedCodec::Float16 => "f16",
            PackedCodec::BFloat16 => "bf16",
        }
    }

    /// Bytes one block of this codec occupies — the multiplier
    /// [`operand_read`] and the row-blocked path need to step between
    /// blocks. Element count per block is shared ([`Q4K_BLOCK_ELEMENTS`])
    /// across the K-quant family (`Q4K`/`Q5K`/`Q6K`) but NOT by `Q8_0`,
    /// which uses its own, much smaller [`Q8_0_BLOCK_ELEMENTS`].
    pub(crate) const fn block_bytes(self) -> usize {
        match self {
            PackedCodec::Q2K => Q2K_BLOCK_BYTES,
            PackedCodec::Q3K => Q3K_BLOCK_BYTES,
            PackedCodec::Q4K => Q4K_BLOCK_BYTES,
            PackedCodec::Q5K => Q5K_BLOCK_BYTES,
            PackedCodec::Q6K => Q6K_BLOCK_BYTES,
            PackedCodec::Q8_0 => Q8_0_BLOCK_BYTES,
            PackedCodec::Q4_0 => Q4_0_BLOCK_BYTES,
            PackedCodec::Float16 => FLOAT16_BLOCK_BYTES,
            PackedCodec::BFloat16 => BFLOAT16_BLOCK_BYTES,
        }
    }

    /// Elements one block of this codec carries — [`crate::wgsl`]'s WGSL
    /// codec table needs this alongside [`Self::block_bytes`] the same way
    /// `operand_read`'s own `{offset} / N_ELEMENTS` / `{offset} % N_ELEMENTS`
    /// split does here, and `crate::metal`'s `operand_tensor_bytes` needs it
    /// to turn a packed operand's element count into its real byte count —
    /// gated on either caller's own feature, since neither is compiled by
    /// default.
    #[cfg(any(feature = "wgpu-backend", feature = "instrument"))]
    pub(crate) const fn block_elements(self) -> usize {
        match self {
            PackedCodec::Q2K
            | PackedCodec::Q3K
            | PackedCodec::Q4K
            | PackedCodec::Q5K
            | PackedCodec::Q6K => Q4K_BLOCK_ELEMENTS,
            PackedCodec::Q8_0 => Q8_0_BLOCK_ELEMENTS,
            PackedCodec::Q4_0 => Q4_0_BLOCK_ELEMENTS,
            PackedCodec::Float16 => FLOAT16_BLOCK_ELEMENTS,
            PackedCodec::BFloat16 => BFLOAT16_BLOCK_ELEMENTS,
        }
    }

    /// Whether this codec's block layout has a paired-nibble/paired-lane
    /// decode body (`q4k_pair_dot`/`q5k_pair_dot`/`q6k_pair_dot`) at all --
    /// the structural fact `push_packed_row_blocked_body`'s `plain_product`
    /// gate reads, in place of a `cfg!(feature = "metal-q{5,6}k-pair-dot")`
    /// check. `Q4_K` (144 B, mult of 16), `Q5_K` (176 B, mult of 16, plus its
    /// `qh` high-bit plane), and `Q6_K` (210 B, NOT a mult of 4, hence the
    /// `ushort` loads in [`Q6K_PAIR_DOT_MSL`]) each have one; the flat 32-
    /// element legacy codecs and the two non-quantized codecs do not --
    /// `classify_packed_row_block` rejects all four before this is ever
    /// consulted (`NotKQuantCodec`), so this only needs to be honest about
    /// the four K-quants, not defensive about the rest. `Q3_K` (110 B) has
    /// its own paired-lane body ([`Q3K_PAIR_DOT_MSL`]) despite a DIFFERENT
    /// byte addressing shape from the other three (four 2-bit levels per
    /// byte, not a nibble or nibble-plus-plane) -- see that body's own doc.
    pub(crate) const fn supports_pair_dot(self) -> bool {
        matches!(
            self,
            PackedCodec::Q3K | PackedCodec::Q4K | PackedCodec::Q5K | PackedCodec::Q6K
        )
    }

    /// Output rows one SIMD group folds at once in the row-blocked packed
    /// path (`push_packed_row_blocked_body`'s generic `else` arm and
    /// `push_packed_row_multi_row_body`; NOT `push_q4k_single_fetch_body`/
    /// `push_q4k_ggml_port_body`, which are `Q4_K`-only and keep `4` baked
    /// into their own `lane % 8u` arithmetic regardless of this value —
    /// [`PACKED_ROWS_PER_GROUP`]'s doc). `Q6_K` decodes three raw-byte
    /// fields per element (`ql`, `qh`, `scale`) versus `Q4_K`'s effectively
    /// two, so batching 4 rows' worth of `sumf[q]` accumulators plus
    /// per-row `weight_base[q]`/`other_base[q]` state costs more live
    /// registers per lane for `Q6_K` than the same batching costs `Q4_K` —
    /// matches ggml's own choice (`N_R0_Q6_K = 1`, `N_R0_Q4_K = 4`,
    /// `ggml-metal-impl.h:32-39`). The lane assignment itself (`ix`/`it`/
    /// `slot` spreading all 32 lanes across the reduction axis) does not
    /// depend on this value — it only controls how many output rows share
    /// one activation load.
    pub(crate) const fn rows_per_simdgroup(self) -> usize {
        match self {
            PackedCodec::Q6K => 1,
            _ => PACKED_ROWS_PER_GROUP,
        }
    }
}

/// Every packed operand a bound program has, keyed by [`NodeId`] to its
/// codec — the single source of truth [`emit`] (via the `quantized` slice it
/// derives) and the Metal driver's `correct_packed_matmul_layouts` call both
/// need, generalizing the Q4_K-only `BTreeSet<NodeId>` this crate carried
/// before Q6_K support existed.
pub type PackedOperands = BTreeMap<NodeId, PackedCodec>;

pub fn emit(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<Kernel, EmitError> {
    emit_inner(resolved, packed_operands, numeric_policy, false)
}

fn emit_inner(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    expert_source_mode: bool,
) -> Result<Kernel, EmitError> {
    validate(resolved)?;
    let entry = entry_name(resolved);
    let quantized = operand_codecs(resolved, packed_operands);
    let source = match &resolved.kind {
        BoundOpKind::CachedAttention { .. } => {
            render_cached_attention(resolved, &entry, numeric_policy)
        }
        BoundOpKind::Elementwise { .. } => render_elementwise(resolved, &entry, &quantized),
        BoundOpKind::Reduce {
            keep: Keep::Reduce, ..
        } => render_reduce(resolved, &entry, &quantized, expert_source_mode),
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => render_scan(resolved, &entry, &quantized),
        BoundOpKind::Iota => render_iota(resolved, &entry),
        BoundOpKind::Constant { value } => render_constant(resolved, &entry, *value),
    }?;
    // Coupled to `render_cached_attention`'s own final-store branch by
    // construction: whenever the rendered SOURCE writes the scratch layout
    // instead of the real output (`cached_attention_merge_needed`), the
    // BINDINGS this call returns must say so too (`Binding::Scratch`
    // replacing `Binding::Output`) -- source and bindings are two views of
    // the SAME `emit` call, keyed on the SAME `numeric_policy`, so they can
    // never disagree with each other. What they CAN disagree with is a
    // caller that lacks the merge dispatch this binding shape requires --
    // `crate::metal::encode_op`'s own doc names that gap and the guard it
    // takes on its `resolved: None` (no plan-resolved merge sibling) path.
    let is_split_cached_attention = cached_attention_merge_needed(&resolved.kind, numeric_policy);
    Ok(Kernel {
        source,
        entry,
        bindings: if is_split_cached_attention {
            split_bindings_with_scratch(resolved)
        } else {
            bindings(resolved)
        },
        grid: GridSpec {
            threads: grid_threads(resolved, &quantized, numeric_policy)?,
            threadgroup_width: tiled_gemm_threadgroup_width(resolved, &quantized, numeric_policy),
        },
    })
}

/// Emits the ordinary kernel ABI plus the two buffers required by a HOBBIT
/// substitution table.  The decoder is deliberately not selected here: this
/// establishes the stable binding contract that the mixed-codec lowering will
/// consume, while ordinary [`emit`] callers remain byte-for-byte unchanged.
pub fn emit_with_expert_sources(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    source_node: NodeId,
) -> Result<Kernel, EmitError> {
    emit_with_expert_sources_mode(resolved, packed_operands, numeric_policy, source_node, None)
}

/// Emits a substituted expert kernel while retaining the bound operand's
/// packed decoder. This is valid only when every selected descriptor uses the
/// same codec; mixed selections must use [`emit_with_expert_sources`].
pub fn emit_with_uniform_expert_source(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    source_node: NodeId,
    codec: PackedCodec,
) -> Result<Kernel, EmitError> {
    emit_with_expert_sources_mode(
        resolved,
        packed_operands,
        numeric_policy,
        source_node,
        Some(codec),
    )
}

fn emit_with_expert_sources_mode(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
    source_node: NodeId,
    uniform_codec: Option<PackedCodec>,
) -> Result<Kernel, EmitError> {
    // A substituted expert source may select a different codec for every
    // routed expert. Do not specialize this operand to the checkpoint's
    // original codec: the packed-row bodies bake one decoder into the kernel
    // and would interpret a low-copy Q2_K expert as Q4_K. Rendering the source
    // operand as scalar first leaves every read visible to the descriptor-aware
    // replacement below, whose codec tag selects the decoder at runtime.
    let mut source_packed_operands = packed_operands.clone();
    if let Some(codec) = uniform_codec {
        source_packed_operands.insert(source_node, codec);
    } else {
        source_packed_operands.remove(&source_node);
    }
    let mut kernel = emit_inner(
        resolved,
        &source_packed_operands,
        numeric_policy,
        uniform_codec.is_none(),
    )?;
    // Replace the gathered weight read with the descriptor-aware form. The
    // ordinary emitter remains unchanged; this opt-in path is selected only
    // when a residency table is supplied for this operand.
    if let Some((weight_index, (_, _, Some(_)))) = resolved
        .operands()
        .iter()
        .enumerate()
        .find(|(_, (node, _, _lookup))| *node == source_node)
    {
        let gather_slot = resolved
            .operands()
            .iter()
            .take(weight_index)
            .filter(|(_, _, lookup)| lookup.is_some())
            .count();
        let row_block_source = kernel.source.contains("long weight_base[");
        let offset = format!("off{weight_index}");
        for codec in [
            None,
            Some(PackedCodec::Q2K),
            Some(PackedCodec::Q4K),
            Some(PackedCodec::Q5K),
            Some(PackedCodec::Q6K),
        ] {
            for offset_name in [
                offset.as_str(),
                &format!("walk{weight_index}"),
                &format!("read_off{weight_index}"),
                &format!("running{weight_index}"),
                &format!("(weight_base[q] + k)"),
            ] {
                let expert_stride = if row_block_source {
                    String::from("0l")
                } else {
                    format!("u.gather_element_stride[{gather_slot}]")
                };
                let old = operand_read(weight_index, offset_name, codec);
                let replacement = if offset_name == format!("walk{weight_index}") {
                    format!(
                        "mixed_expert_element_from_local(expert_base{weight_index}, expert_codec{weight_index}, {offset_name})"
                    )
                } else {
                    format!(
                        "mixed_expert_element_from_offset(expert_payloads, expert_descriptors, (uint)fetched{weight_index}, {offset_name}, {}, expert_codec{weight_index})",
                        expert_stride,
                    )
                };
                let replacement = if let Some(codec) = uniform_codec {
                    format!(
                        "uniform_expert_element_from_offset_{}(expert_payloads, expert_descriptors, (uint)fetched{}, {offset_name}, {})",
                        codec.cache_token(),
                        weight_index,
                        expert_stride,
                    )
                } else {
                    replacement
                };
                kernel.source = kernel.source.replace(&old, &replacement);
            }
        }
        let expert_base =
            format!("expert_payloads + expert_descriptors[expert_route_index[q]].byte_offset");
        // packed-row group bases fetch the route inside their per-row `q`
        // scope.  The row-blocked body consumes that route later, after the
        // scope has closed, so preserve it in a row-indexed array rather than
        // referring to the local `fetchedN` declaration out of scope.
        kernel.source = kernel
            .source
            .replace("long weight_base[", "long expert_route_index[");
        if let Some(start) = kernel.source.find("long expert_route_index[") {
            if let Some(end) = kernel.source[start..].find(';') {
                let declaration_end = start + end + 1;
                let declaration = kernel.source[start..declaration_end].to_owned();
                let dimension = declaration
                    .split_once('[')
                    .and_then(|(_, rest)| rest.split_once(']'))
                    .map(|(value, _)| value)
                    .unwrap_or("1");
                kernel.source = kernel.source.replacen(
                    &declaration,
                    &format!("long weight_base[{dimension}];\n    long expert_route_index[{dimension}];\n    long expert_row_base[{dimension}];"),
                    1,
                );
            }
        }
        kernel.source = kernel
            .source
            .lines()
            .map(|line| {
                let marker = format!("blk_ptr[q] = in{weight_index} + ");
                let packed_block_marker = format!(
                    "device const uchar *blk = in{weight_index} + "
                );
                let packed_block_marker_compact = format!(
                    "device const uchar* blk = in{weight_index} + "
                );
                if line.contains(&marker)
                    || (row_block_source
                        && (line.contains(&packed_block_marker)
                            || line.contains(&packed_block_marker_compact)))
                {
                    line.replace(
                        &format!("in{weight_index} + "),
                        &format!("{expert_base} + "),
                    )
                        .replace(
                            "weight_base[q]",
                            &format!(
                                "(expert_row_base[q] - u.operand_base[{weight_index}] - expert_route_index[q] * u.gather_element_stride[{gather_slot}])"
                            ),
                        )
                } else if line.contains(&format!(
                    "fetched{weight_index} = (long)simd_broadcast_first"
                )) {
                    if row_block_source {
                        format!(
                            "{line}\n        uint expert_codec{weight_index} = expert_descriptors[(uint)fetched{weight_index}].codec;\n        device const uchar *expert_base{weight_index} = expert_payloads + expert_descriptors[(uint)fetched{weight_index}].byte_offset;\n        expert_route_index[q] = fetched{weight_index};"
                        )
                    } else {
                        format!(
                            "{line}\n    uint expert_codec{weight_index} = expert_descriptors[(uint)fetched{weight_index}].codec;\n    device const uchar *expert_base{weight_index} = expert_payloads + expert_descriptors[(uint)fetched{weight_index}].byte_offset;"
                        )
                    }
                } else if !row_block_source
                    && (line.contains(&packed_block_marker)
                        || line.contains(&packed_block_marker_compact))
                {
                    // The ordinary packed body has no row index `q`; its
                    // fetched expert is broadcast in `fetchedN`.  Rebase the
                    // full-stack offset to that expert's compact payload
                    // before selecting the codec-specific block bytes.
                    let mut rewritten = line.replace(
                        &format!("in{weight_index} + "),
                        &format!(
                            "expert_payloads + expert_descriptors[(uint)fetched{weight_index}].byte_offset + "
                        ),
                    );
                    let full_base = format!("base{weight_index}");
                    rewritten = rewritten.replacen(
                        &full_base,
                        &format!(
                            "(base{weight_index} - fetched{weight_index} * u.gather_element_stride[{gather_slot}])"
                        ),
                        1,
                    );
                    let slot_base = String::from("slot_off");
                    rewritten.replacen(
                        &slot_base,
                        &format!(
                            "(slot_off - fetched{weight_index} * u.gather_element_stride[{gather_slot}])"
                        ),
                        1,
                    )
                } else if line.contains(&format!("int walk{weight_index} = (int)off{weight_index};")) {
                    format!(
                        "{line}\n        walk{weight_index} -= fetched{weight_index} * u.gather_element_stride[{gather_slot}];"
                    )
                } else if line.trim() == "weight_base[q] = wb;" {
                    format!("{line}\n        expert_row_base[q] = wb;")
                } else if line.trim().starts_with("weight_base[q] += fetched") {
                    format!("{line}\n        expert_row_base[q] = weight_base[q];")
                } else { line.to_owned() }
            })
            .collect::<Vec<_>>()
            .join("\n");
        if std::env::var_os("PROXIMA_DEBUG_EXPERT_EMIT").is_some() {
            for line in kernel.source.lines().filter(|line| {
                line.contains("mixed_expert")
                    || line.contains("expert_route_index")
                    || line.contains("fetched")
                    || line.contains("walk0")
                    || line.contains("base0")
            }) {
                eprintln!("qwen35 expert msl: {line}");
            }
        }
    }
    let payload_binding = Binding::ExpertPayloads(source_node);
    let descriptor_binding = Binding::ExpertDescriptors(source_node);
    let next_buffer = kernel.bindings.len();
    let signature = format!("kernel void {}(", kernel.entry);
    let signature_start = kernel
        .source
        .find(&signature)
        .ok_or(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "emitted kernel signature",
            found: "missing",
        })?;
    let body_start = kernel.source[signature_start..]
        .find(")\n{\n")
        .map(|offset| signature_start + offset)
        .ok_or(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "emitted kernel signature terminator",
            found: "missing",
        })?;
    // `MIXED_EXPERT_READ_MSL` contributes the descriptor declaration in the
    // shared preamble; do not splice a second definition into each kernel.
    let descriptor_struct = "";
    kernel.source.insert_str(signature_start, descriptor_struct);
    let adjusted_body_start = body_start + descriptor_struct.len();
    let parameters = format!(
        ",\n    device const uchar* expert_payloads [[buffer({next_buffer})]],\n    device const ExpertPayloadDescriptor* expert_descriptors [[buffer({})]]",
        next_buffer + 1
    );
    kernel.source.insert_str(adjusted_body_start, &parameters);
    kernel.bindings.push(payload_binding);
    kernel.bindings.push(descriptor_binding);
    Ok(kernel)
}

/// The row-blocked/tiled-GEMM structural shape [`kernel_cache_key`] folds
/// into [`crate::identity::MetalOnlyExtras::packed_row_block_shape`] — 'G'
/// (tiled `simdgroup_matrix` GEMM, checked FIRST: [`tiled_gemm_block`] only
/// ever returns `Some` when [`packed_row_block`] also would, since it is
/// built ON TOP of that same gate, so the two are mutually exclusive by
/// construction and this order costs nothing extra to get right), 'M'/'B'
/// (row-blocked, multiple/single activation row per streamed weight row),
/// 'S' (fully serial, every non-`Reduce` op and every `Reduce` neither path
/// claims). Kept here, not in `identity.rs`: every function it calls is
/// Metal-only private state ([`PackedRowBlock`], [`tiled_gemm_block`]).
// same gate as `kernel_cache_key`, its sole caller -- see that function's
// own comment for why `metal-core` alone (not `metal`) is the right feature.
#[cfg(any(test, feature = "metal-core"))]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
fn packed_row_block_shape_token(resolved: &BoundOp, quantized: &[Option<PackedCodec>]) -> char {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        ..
    } = &resolved.kind
    else {
        return 'S';
    };
    if tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some() {
        'G'
    } else if let Some(block) = packed_row_block(resolved, quantized) {
        if packed_row_block_token_total(&block, &resolved.extents) > 1 {
            'M'
        } else {
            'B'
        }
    } else {
        'S'
    }
}

/// Whether the packed row-block's non-weight operand reads unit stride on
/// the reduce axis — `push_packed_row_blocked_body`'s STRIDE-FREE
/// SPECIALIZATION (see that function's own doc) renders different source
/// text for the SAME 'B'/'M' structural shape depending on this CONCRETE
/// resolved stride, not just op structure. `None` when [`packed_row_block`]
/// has no match at all — [`packed_row_block_shape_token`]'s 'G'/'S' arms
/// never render that specialization, so a stray extra token there would
/// only cost an unnecessary cache miss, never a wrong hit; leaving it out
/// entirely is just as sound and keeps a non-matching op's identity free of
/// a token it has no reason to carry.
#[cfg(any(test, feature = "metal-core"))]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
fn packed_row_block_stride_is_one(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Option<bool> {
    let BoundOpKind::Reduce { .. } = &resolved.kind else {
        return None;
    };
    let block = packed_row_block(resolved, quantized)?;
    Some(
        resolved.operands()[block.other]
            .1
            .stride(block.reduce_dim as u16)
            == 1,
    )
}

/// Whether [`push_packed_row_group_bases`] rendered the direct single-axis
/// row addressing for this op, and on which axis — `None` when it did not
/// (no `packed_row_block` match, or a multi-row `M` block, which always
/// takes the generic path; see [`packed_row_direct_output_axis`]'s own doc).
#[cfg(any(test, feature = "metal-core"))]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
fn packed_row_block_direct_axis(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Option<u16> {
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        return None;
    };
    let block = packed_row_block(resolved, quantized)?;
    if packed_row_block_token_total(&block, &resolved.extents) > 1 {
        return None;
    }
    packed_row_direct_output_axis(resolved, output_axes)
}

/// Cheap structural + compile-option identity for the kernel [`emit`] would
/// produce from `resolved` — built without ever rendering the MSL body
/// text, so a caller can decide whether a pipeline compile is needed before
/// paying for one. A thin Metal-specific wrapper around
/// [`crate::identity::kernel_identity`]: this crate's own [`entry_name`]
/// stays a byte-identical, narrower fingerprint (embedded literally as the
/// compiled kernel's own declared name, so its format can never change
/// without changing emitted MSL source); this key is the fuller identity a
/// caller uses to decide pipeline-cache reuse, never embedded in source
/// text, so its format is free to change as the union of axes it must
/// distinguish grows. See [`crate::identity`]'s module doc for the full
/// axis census and why one shared function derives every renderer's own
/// version of this fact instead of three.
///
/// # Errors
/// Propagates [`type_token`]'s unsupported-dtype rejection — the same gate
/// [`emit`] enforces before ever building a kernel.
// production caller is `crate::metal::encode_op` (macOS-only driver, gated
// on `metal`, which now implies `metal-core`); the `mod tests` call sites
// below are the second, so `cfg(test)` keeps a non-macOS `cargo test` build
// honest. No macOS-specific code lives in this function body, so it is
// gated on `metal-core` alone -- it builds (and is exercised) on Linux too.
// `metal-core` without `metal` (the Linux emitter-only build) has no driver
// to call it, hence the `allow`: genuinely unreachable there, not a hidden bug.
#[cfg(any(test, feature = "metal-core"))]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
pub(crate) fn kernel_cache_key(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<String, EmitError> {
    // Called for its unsupported-dtype rejection alone -- `kernel_identity`
    // reads `resolved.dtype` directly for the actual half/wide classing (the
    // same partition every renderer's own `type_token` match already makes),
    // but `kernel_cache_key` has no `validate` call of its own upstream of
    // it, so this stays the one place that fails fast on a dtype `emit`
    // would also reject.
    type_token(resolved.node, resolved.dtype)?;
    let quantized = operand_codecs(resolved, packed_operands);
    let extras = crate::identity::MetalOnlyExtras {
        cooperative_width: tiled_gemm_threadgroup_width(resolved, &quantized, numeric_policy),
        packed_row_block_shape: Some(packed_row_block_shape_token(resolved, &quantized)),
        packed_row_block_stride_is_one: packed_row_block_stride_is_one(resolved, &quantized),
        packed_row_block_direct_axis: packed_row_block_direct_axis(resolved, &quantized),
        elementwise_addressing: elementwise_addressing_cache_token(resolved),
        numeric_policy_token: Some(crate::identity::numeric_policy_cache_token(numeric_policy)),
    };
    Ok(crate::identity::kernel_identity(
        crate::identity::KernelLanguage::Metal,
        resolved,
        packed_operands,
        extras,
        numeric_policy,
    ))
}

/// The dispatch-time shape of `resolved`'s kernel — buffer bindings and
/// thread count — without rendering any MSL body text. Cheap on every call
/// regardless of pipeline-cache hit or miss: [`emit`]'s `source`/`entry`
/// fields are needed only on a genuine cache miss (see
/// `crate::metal::encode_op`).
///
/// # Errors
/// Propagates [`validate`]'s structural rejection — the same gate [`emit`]
/// enforces before ever building a kernel.
// unlike `kernel_cache_key` above, this has no `mod tests` call site of its
// own, so a bare `cfg(test)` disjunct leaves it genuinely dead-code on a
// non-macOS `cargo test`/`nextest` build -- gate on the one real caller
// (`crate::metal::encode_op`, via `metal-core`, same rationale as above).
#[cfg(feature = "metal-core")]
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(dead_code, reason = "sole caller is the macOS-only metal driver")
)]
pub(crate) fn kernel_dispatch_shape(
    resolved: &BoundOp,
    packed_operands: &PackedOperands,
    numeric_policy: NumericPolicy,
) -> Result<(Vec<Binding>, GridSpec), EmitError> {
    validate(resolved)?;
    let quantized = operand_codecs(resolved, packed_operands);
    let is_split_cached_attention = cached_attention_merge_needed(&resolved.kind, numeric_policy);
    Ok((
        if is_split_cached_attention {
            split_bindings_with_scratch(resolved)
        } else {
            bindings(resolved)
        },
        GridSpec {
            threads: grid_threads(resolved, &quantized, numeric_policy)?,
            threadgroup_width: tiled_gemm_threadgroup_width(resolved, &quantized, numeric_policy),
        },
    ))
}

// `SIMD_WIDTH` moved to `crate::sized::SIMD_WIDTH` (the build-time floor's
// only configuration surface) -- imported at the top of this file.

/// Whether `resolved` is a `Keep::Reduce` fold whose `reduce_op` is
/// associative and commutative (`Add`, `Multiply`, `Maximum`, `Minimum`) with
/// a gather whose index lookup is invariant across every reduction axis, AND
/// whose reduced-axis extent meets
/// [`crate::sized::COOPERATIVE_REDUCE_MIN_LEN`] — the set [`render_reduce`]
/// emits a SIMD-group cooperative loop for instead of the one-thread-per-
/// output serial fold. `Subtract`/`Divide` are not associative, so
/// reordering their combination across lanes is not imprecise, it is wrong —
/// they and every other `ScalarOp` stay on the serial path. A gather is
/// admitted only when its index lookup is constant over every reduced axis;
/// the fetched index is then broadcast once per simdgroup and the existing
/// lane-0 fault reporting remains sufficient.
///
/// The length gate exists because a cooperative reduce always launches
/// `SIMD_WIDTH`(32) lanes per output regardless of how many elements each
/// output folds — a 34-long attention reduce pays a full `simd_sum` combine
/// for 34 elements of real work across 32 mostly-idle lanes. `min_len == 0`
/// (the sentinel, not a real length any reduce can be shorter than) makes
/// this check vacuous, so every op that clears the op/gather gate above
/// stays cooperative — the routing every build before this key existed used,
/// and the `omega-runtime.toml` default: that "mostly-idle lanes" framing
/// turned out to predict the wrong direction on real hardware (that file's
/// own `[cooperative_reduce]` doc has the measured numbers) — these
/// reduces are memory-latency-bound, and the serial route's 32x-fewer
/// threads hides less load latency than the idle lanes cost, so routing
/// short reduces off cooperative made them slower, not faster.
fn reduce_is_cooperative(resolved: &BoundOp) -> bool {
    match &resolved.kind {
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            reduce_op,
            output_axes,
            ..
        } => {
            is_cooperative_reduce_op(*reduce_op)
                && gather_is_reduction_invariant(resolved, output_axes)
                && meets_cooperative_min_len(reduction_len(resolved, output_axes))
        }
        _ => false,
    }
}

/// A route selected by the token axis has zero stride in contracted
/// dimensions, so one fetched expert index can be shared by all lanes. Any
/// nonzero stride would select different experts during the reduction and is
/// therefore kept on the serial path.
fn gather_is_reduction_invariant(resolved: &BoundOp, output_axes: &[u16]) -> bool {
    let reduce_dims = reduction_dims(resolved, output_axes);
    resolved.operands().iter().all(|(_, _, lookup)| {
        lookup.as_ref().is_none_or(|lookup| {
            reduce_dims
                .iter()
                .all(|&dim| lookup.index_layout.stride(dim) == 0)
        })
    })
}

/// `length >= COOPERATIVE_REDUCE_MIN_LEN`, factored out so clippy's
/// `absurd_extreme_comparisons` lint has one site to allow rather than every
/// call site: at the `omega-runtime.toml` default (0, `u64::MIN`) the
/// comparison IS always true, and that is the intended behavior (see
/// `reduce_is_cooperative`'s own doc) -- a config-driven threshold cannot be
/// assumed non-degenerate by the linter, but `OMEGA_COOPERATIVE_REDUCE_MIN_
/// LEN` overriding it to a real value at build time makes this a genuine
/// runtime-varying comparison, not dead code.
#[allow(clippy::absurd_extreme_comparisons)]
fn meets_cooperative_min_len(length: u64) -> bool {
    length >= crate::sized::COOPERATIVE_REDUCE_MIN_LEN
}

/// Total element count one output folds over: the product of the extents of
/// every dim [`reduction_dims`] names. Zero-rank (a scalar operand reduced
/// over nothing) has no `reduce_dims`, so `product()` over the empty
/// iterator correctly yields `1` — one element, itself.
fn reduction_len(resolved: &BoundOp, output_axes: &[u16]) -> u64 {
    reduction_dims(resolved, output_axes)
        .iter()
        .map(|&dim| resolved.extents[dim as usize])
        .product()
}

fn is_cooperative_reduce_op(op: ScalarOp) -> bool {
    matches!(
        op,
        ScalarOp::Add | ScalarOp::Multiply | ScalarOp::Maximum | ScalarOp::Minimum
    )
}

/// The MSL SIMD-group reduction builtin that combines one lane's private
/// accumulator across the whole 32-lane group — only called for a
/// [`is_cooperative_reduce_op`] body, so the non-cooperative arms below are
/// enumerated rather than wildcarded — adding a `ScalarOp` variant forces a
/// decision here instead of silently panicking.
fn simd_combine_fn(node: NodeId, op: ScalarOp) -> Result<&'static str, EmitError> {
    match op {
        ScalarOp::Add => Ok("simd_sum"),
        ScalarOp::Multiply => Ok("simd_product"),
        ScalarOp::Maximum => Ok("simd_max"),
        ScalarOp::Minimum => Ok("simd_min"),
        ScalarOp::Identity
        | ScalarOp::Subtract
        | ScalarOp::Divide
        | ScalarOp::Negate
        | ScalarOp::Reciprocal
        | ScalarOp::Exponential
        | ScalarOp::Logarithm
        | ScalarOp::SquareRoot
        | ScalarOp::Tanh
        | ScalarOp::Erf
        | ScalarOp::Greater
        | ScalarOp::Equal
        | ScalarOp::Select => Err(EmitError::NonCooperativeReduceOp {
            node,
            op: op_token(op),
        }),
    }
}

/// The algebraic identity `op` folds against without changing a value: `e op
/// x == x` for every `x`. Every SIMD lane but lane 0 seeds its private
/// accumulator with this (never with the `BoundOp`'s own `ReduceInit`, which
/// may be `FirstElement` or otherwise mismatched with `op`) — folding that
/// untouched identity into the final `simd_*` combine can never perturb the
/// result, because `e op e == e` holds for any identity by definition. Lane
/// 0 alone carries the real seed, so it is folded into the group exactly
/// once, matching `cpu::run_reduce`'s single-seed semantics regardless of
/// how many idle lanes there are.
fn cooperative_identity_token(node: NodeId, op: ScalarOp) -> Result<&'static str, EmitError> {
    match op {
        ScalarOp::Add => Ok("0.0f"),
        ScalarOp::Multiply => Ok("1.0f"),
        ScalarOp::Maximum => Ok("-INFINITY"),
        ScalarOp::Minimum => Ok("INFINITY"),
        ScalarOp::Identity
        | ScalarOp::Subtract
        | ScalarOp::Divide
        | ScalarOp::Negate
        | ScalarOp::Reciprocal
        | ScalarOp::Exponential
        | ScalarOp::Logarithm
        | ScalarOp::SquareRoot
        | ScalarOp::Tanh
        | ScalarOp::Erf
        | ScalarOp::Greater
        | ScalarOp::Equal
        | ScalarOp::Select => Err(EmitError::NonCooperativeReduceOp {
            node,
            op: op_token(op),
        }),
    }
}

/// Structural checks over a (possibly fused) [`ComposedBody`]: every step's
/// own arity matches its arg count — the same check [`validate`] always ran,
/// now per absorbed step instead of once for a single `ScalarOp`, since a
/// fused body can carry more than one.
fn validate_body(node: NodeId, body: &ComposedBody) -> Result<(), EmitError> {
    for step in &body.steps {
        let expected = step.op.arity();
        let found = step.args.len();
        if expected != found {
            return Err(EmitError::ArityMismatch {
                node,
                expected,
                found,
            });
        }
    }
    Ok(())
}

fn validate(resolved: &BoundOp) -> Result<(), EmitError> {
    validate_body(resolved.node, resolved.element_body())?;
    if let BoundOpKind::Reduce {
        reduce_op,
        keep,
        out_scatter,
        ..
    } = &resolved.kind
    {
        if out_scatter.is_some() {
            return Err(EmitError::ScatterNotSupported {
                node: resolved.node,
            });
        }
        if matches!(reduce_op, ScalarOp::Select) {
            return Err(EmitError::ReductionBodyIsSelect {
                node: resolved.node,
            });
        }
        if *keep == Keep::Scan && resolved.extents.is_empty() {
            return Err(EmitError::EmptyScan {
                node: resolved.node,
            });
        }
    }
    Ok(())
}

/// `pub(crate)`, not private: the Metal driver's uniforms packer
/// (`crate::metal::pack_reduce_uniforms`) needs the exact same reduce-dim set
/// this rendering uses, and duplicating the filter would risk the two
/// drifting apart.
pub(crate) fn reduction_dims(resolved: &BoundOp, output_axes: &[u16]) -> Vec<u16> {
    (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .collect()
}

fn bindings(resolved: &BoundOp) -> Vec<Binding> {
    // `all_read_sources`, not `operands` -- a `BoundOpKind::Reduce` with a
    // fused epilogue reads its `epilogue_operands` too, and those need a
    // buffer bound at the exact index `kernel_signature`'s `epi{index}`
    // params claim (see that function's own doc).
    let mut bindings: Vec<Binding> = resolved
        .all_read_sources()
        .map(|(node, _, _)| Binding::Input(*node))
        .collect();
    for (_, _, gather) in resolved.operands() {
        if let Some(gather_access) = gather {
            bindings.push(Binding::Indices(gather_access.indices));
        }
    }
    bindings.push(Binding::Output(resolved.node));
    bindings.push(Binding::Uniforms);
    if gather_count(resolved) > 0 {
        bindings.push(Binding::Fault);
    }
    bindings
}

/// [`bindings`]'s split-kernel counterpart for `BoundOpKind::CachedAttention`
/// once [`cached_attention_merge_needed`] admits `ContextSplitMerge`: the
/// same read set, but the LAST slot -- the op's own output -- becomes
/// [`Binding::Scratch`] instead of [`Binding::Output`], because the split
/// kernel no longer writes `resolved.node`'s real buffer; the merge kernel
/// does. See [`render_cached_attention_merge`]'s own doc for the kernel that
/// reads this scratch buffer back out.
fn split_bindings_with_scratch(resolved: &BoundOp) -> Vec<Binding> {
    let mut list = bindings(resolved);
    for binding in &mut list {
        if let Binding::Output(_) = binding {
            *binding = Binding::Scratch;
        }
    }
    list
}

/// [`render_cached_attention_merge`]'s own binding list: reads the scratch
/// buffer [`split_bindings_with_scratch`] wrote, writes the op's real
/// output, and needs its own (smaller) `Uniforms` blob -- no operand inputs,
/// no gather, no fault buffer, since the merge is pure scratch-to-output
/// rescale-and-copy.
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
fn merge_bindings(resolved: &BoundOp) -> Vec<Binding> {
    alloc::vec![
        Binding::Scratch,
        Binding::Output(resolved.node),
        Binding::Uniforms
    ]
}

/// Every `NodeId` `bindings` reads from a device buffer for — the exact
/// operand set `crate::metal::bind_buffers` resolves for a
/// `Binding::Input`/`Binding::Indices` slot, in bind order. This is the one
/// adapter a hazard tracker (or anything
/// else that needs "what does this dispatch read") should walk, instead of
/// re-deriving the read set from `BoundOp::all_read_sources()` independently
/// — ROW 323's bug was exactly two call sites (`bindings()` here and the
/// hazard tracker's own operand enumeration) drifting apart when
/// `bindings()` grew a source `all_read_sources()`'s caller had not been
/// updated to match. Walking `bindings` itself makes that drift impossible:
/// there is only one list, and both the encoder bind loop and the hazard
/// walk read the same one.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn hazard_read_nodes(bindings: &[Binding]) -> impl Iterator<Item = NodeId> + '_ {
    bindings.iter().filter_map(|binding| match binding {
        Binding::Input(node) | Binding::Indices(node) => Some(*node),
        Binding::Output(_)
        | Binding::ExpertPayloads(_)
        | Binding::ExpertDescriptors(_)
        | Binding::Uniforms
        | Binding::Fault
        | Binding::Scratch => None,
    })
}

/// The single `NodeId` `bindings` writes to — every bound op writes exactly
/// one device buffer (`crate::metal::bind_buffers`'s own doc), so a well-formed
/// `bindings` list always has exactly one `Binding::Output`. `None` only if
/// `bindings` is malformed (a validation bug upstream, not a runtime case a
/// caller should expect to hit).
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) fn hazard_write_node(bindings: &[Binding]) -> Option<NodeId> {
    bindings.iter().find_map(|binding| match binding {
        Binding::Output(node) => Some(*node),
        // `Binding::Scratch`'s write target has no `NodeId` -- see
        // `Binding::Scratch`'s own doc. A caller tracking hazards across the
        // split dispatch's scratch write needs the scratch buffer's own
        // pointer identity directly, not through this `NodeId` path.
        Binding::Input(_)
        | Binding::ExpertPayloads(_)
        | Binding::ExpertDescriptors(_)
        | Binding::Indices(_)
        | Binding::Uniforms
        | Binding::Fault
        | Binding::Scratch => None,
    })
}

/// For each operand, `Some(slot)` if it gathers — `slot` is its position
/// among only the gathered operands, 0-based, matching the order
/// [`bindings`] appends `Indices` buffers and the order the `Uniforms`
/// gather arrays are packed in. `pub(crate)` for the same reason
/// [`reduction_dims`] is: the Metal driver's uniforms packer needs the exact
/// same numbering.
pub(crate) fn gather_slots(resolved: &BoundOp) -> Vec<Option<usize>> {
    let mut next = 0usize;
    resolved
        .operands()
        .iter()
        .map(|(_, _, gather)| {
            gather.as_ref().map(|_| {
                let slot = next;
                next += 1;
                slot
            })
        })
        .collect()
}

pub(crate) fn gather_count(resolved: &BoundOp) -> usize {
    resolved
        .operands()
        .iter()
        .filter(|(_, _, gather)| gather.is_some())
        .count()
}

/// Total independent units of work `resolved` needs — see [`GridSpec`]'s doc
/// for why this, unlike [`Kernel::source`], is genuinely a function of
/// `resolved`'s concrete extents.
/// Output rows one SIMD group folds at once in the packed path — ggml's
/// `N_R0_Q4_K`. The point is the ACTIVATION: its run of
/// [`Q4K_BLOCK_ELEMENTS`]/[`SIMD_WIDTH`] values is loaded into registers
/// once and reused across all four rows, so activation traffic falls 4x and
/// the per-row work becomes one header decode plus the nibble extracts.
const PACKED_ROWS_PER_GROUP: usize = 4;

/// Edge length of the `simdgroup_matrix` tile `push_tiled_gemm_body` uses —
/// `simdgroup_float8x8`/`simdgroup_half8x8` are fixed 8x8 by the MSL type
/// itself on every Apple GPU family that supports them, the same "hardware
/// fact, not a policy knob" class [`SIMD_WIDTH`] is in
/// (`crate::sized`'s own module doc draws this exact line): there is no
/// tuning that would make this anything but 8, so it stays a bare `const`
/// rather than threading through the sizing-config mechanism
/// [`crate::sized::TILED_GEMM_MIN_TOKENS`] uses. Only [`push_tiled_gemm_body`]
/// reads it, so it is gated the same as that function -- see its `#[cfg(not(..))]`
/// stub's own doc for why the non-feature build never needs it.
#[cfg(feature = "metal-tiled-gemm")]
const TILE_DIM: usize = 8;

/// Number of `simdgroup`s cooperating in one [`push_tiled_gemm_body`]
/// threadgroup — ports `ggml-metal.metal:6500`'s `kernel_mul_mm` dispatch
/// (`ggml-metal.m:3102`'s `threadsPerThreadgroup:MTLSizeMake(128, 1, 1)`,
/// 128/32 = 4 `simdgroup`s). Fixed at 4 (a 2x2 grid: `sgitg & 1` selects
/// which half of [`crate::sized::TILED_GEMM_BLOCK_M`]'s rows, `sgitg >> 1`
/// selects which half of [`crate::sized::TILED_GEMM_BLOCK_N`]'s columns,
/// exactly ggml's own `mc[8]`/`THREAD_MAT_M`/`THREAD_MAT_N` split) rather
/// than threaded through the sizing-config mechanism: the 2x2 halving is
/// baked into the pointer arithmetic `push_tiled_gemm_body` emits, so a
/// value other than 4 would need a different kernel body, not just a
/// different constant — the same "hardware fact, not a policy knob" class
/// [`TILE_DIM`] and [`crate::sized::SIMD_WIDTH`] are in. `BLOCK_M`/`BLOCK_N`
/// themselves ARE the tunable axes (`crate::sized::TILED_GEMM_BLOCK_M`/
/// `TILED_GEMM_BLOCK_N`) — this only fixes how many simdgroups split them.
const TILED_GEMM_NSG: usize = 4;

/// The one decision that both [`grid_threads`] and
/// [`push_cooperative_reduce_body`] must reach identically: whether this
/// bound op takes the row-blocked packed path. They compute different things
/// from it (dispatch geometry, kernel body), and a disagreement would not
/// fail to compile — it would silently fold the wrong rows. So it is decided
/// once, here, from the bound layout.
struct PackedRowBlock {
    /// operand index of the packed weight
    weight: usize,
    /// operand index of the single non-packed operand (the activation)
    other: usize,
    reduce_dim: usize,
    /// which codec `weight`'s bytes are packed as — decides the block byte
    /// width and which unpack function the emitted body calls.
    codec: PackedCodec,
    /// output axes the activation owns exclusively, outermost first --
    /// empty when the op's output axes do not split cleanly into a
    /// token/feature ownership partition (every axis then counts as a
    /// feature axis; see `push_packed_row_blocked_body`'s single-row arm).
    token_axes: Vec<u16>,
    /// output axes the weight owns exclusively, outermost first -- every
    /// output axis when `token_axes` is empty.
    feature_axes: Vec<u16>,
}

/// The token/feature ownership split `push_packed_row_blocked_body` needs to
/// decide whether more than one activation row can be folded per streamed
/// weight row: every output axis partitions into a token group (nonzero
/// stride on `other`, zero on `weight`) and a feature group (the reverse),
/// each nesting contiguously in every layout that reads it. `None` when any
/// of those conditions fails -- the caller then treats every output axis as
/// a feature axis (`token_axes` empty), which is exactly today's row-blocked
/// behaviour for an op this split does not apply to.
fn split_token_feature_axes(
    output_axes: &[u16],
    weight_layout: &Layout,
    other_layout: &Layout,
    out_layout: &Layout,
    extents: &[u64],
) -> Option<(Vec<u16>, Vec<u16>)> {
    let mut token_axes: Vec<u16> = Vec::new();
    let mut feature_axes: Vec<u16> = Vec::new();
    for &axis in output_axes {
        match (
            weight_layout.stride(axis) == 0,
            other_layout.stride(axis) == 0,
        ) {
            (true, false) => token_axes.push(axis),
            (false, true) => feature_axes.push(axis),
            _ => return None,
        }
    }
    if feature_axes.is_empty() {
        return None;
    }
    let reassembled: Vec<u16> = token_axes
        .iter()
        .chain(feature_axes.iter())
        .copied()
        .collect();
    if reassembled != output_axes {
        return None;
    }
    let groups_contiguous = axes_fold_contiguously(&token_axes, extents, other_layout)
        && axes_fold_contiguously(&feature_axes, extents, weight_layout)
        && axes_fold_contiguously(&token_axes, extents, out_layout)
        && axes_fold_contiguously(&feature_axes, extents, out_layout);
    if !groups_contiguous {
        return None;
    }
    Some((token_axes, feature_axes))
}

/// Product of `block.token_axes`' extents -- `1` when empty (no distinct
/// token axis, or the split did not apply), matching an ordinary product
/// over zero terms. [`push_packed_row_blocked_body`]'s own branch on
/// whether this exceeds `1` is the single decision point for which kernel
/// body shape gets emitted; [`kernel_cache_key`] and the dispatch-geometry
/// functions below all re-derive the identical value from the identical
/// block so none of them can drift from what the body actually emits.
fn packed_row_block_token_total(block: &PackedRowBlock, extents: &[u64]) -> u64 {
    block
        .token_axes
        .iter()
        .map(|&axis| extents[axis as usize])
        .product()
}

/// Why a given [`BoundOp`] did NOT take the row-blocked packed kernel —
/// `classify_packed_row_block`'s error arm, one variant per gate in that
/// function's own condition order. `#[non_exhaustive]` so a new gate added
/// later is a compile error at every match site instead of a silently
/// unmatched `_`. Always compiled (not feature-gated itself) so
/// `classify_packed_row_block` — called from the unconditional emit path
/// — never needs a second copy of these seven conditions; only the public
/// accessor `diagnose_packed_row_block` is gated behind `instrument`, this
/// crate's diagnostic-only feature (see
/// `crate::metal::execute_plan_op_timed`'s own doc for why diagnostics
/// live behind that gate).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PackedRowBlockRejection {
    /// `reduce_is_cooperative` is false — not `Add`/`Multiply`/`Maximum`/
    /// `Minimum`, or the op gathers.
    NotCooperativeReduce,
    /// Not a `Reduce { keep: Keep::Reduce, .. }` at all.
    NotReduceKeepReduce,
    /// `quantized.len() != 2` — not a two-operand (weight, activation) op.
    OperandCountNotTwo,
    /// Neither exactly zero nor exactly one operand is packed.
    NotExactlyOnePackedOperand,
    /// Gathered packed weights need the cooperative gather-aware renderer;
    /// the row-blocked body has no index-buffer fetch path.
    GatheredOperand,
    /// The packed operand's codec is [`PackedCodec::Q8_0`] or
    /// [`PackedCodec::Q4_0`] — this path's lane amortization
    /// ([`Q4K_BLOCK_ELEMENTS`], 8 lanes per 32-element sub-block) is
    /// hard-coded to the K-quant family's shared 256-element super-block,
    /// which neither flat 32-element codec has an analogue for. Both
    /// always take the fully generic per-element path instead (see
    /// [`Q8_0_UNPACK_MSL`]/[`Q4_0_UNPACK_MSL`]'s own docs). Checked by
    /// WHITELISTING the three K-quant variants rather than blacklisting
    /// `Q8_0` alone -- an equality check against one non-K-quant codec
    /// silently admits any OTHER non-K-quant codec whose extent happens to
    /// be a multiple of 256 (`docs/discipline.md`'s own landmine: `Q8_0`'s
    /// addition was caught only because this was rewritten as a match, not
    /// because the single `==` check would have caught `Q4_0` too).
    NotKQuantCodec,
    /// The reduce folds ZERO axes into its output — degenerate, never
    /// observed on a real matmul (kept so the match stays exhaustive over
    /// every way `reduce_dims` (`reduction_dims`) can come back empty).
    NotExactlyOneReduceDim { reduce_dims: Vec<u16> },
    /// More than one reduce dim, but they do NOT nest contiguously for both
    /// operands (see `classify_packed_row_block`'s own doc for the
    /// contiguous-fold check this fails) — cannot be treated as one
    /// flattened reduction, so the generic per-element path runs instead.
    ReduceDimsNotContiguous { reduce_dims: Vec<u16> },
    /// The packed operand's stride at the innermost reduce dim is not 1.
    NonUnitWeightStride { stride: i64 },
    /// The flattened extent across every reduce dim is not a whole
    /// multiple of [`Q4K_BLOCK_ELEMENTS`].
    ExtentNotBlockMultiple { extent: u64 },
}

/// Whether `dims` (given OUTERMOST-first, i.e. `dims.last()` is the
/// fastest/innermost axis — the same convention [`reduction_dims`]'s own
/// callers already use) is one contiguous nested block in `layout`: each
/// outer axis's stride equals the extent of every axis nested inside it
/// times that inner axis's own stride. A single dim (or empty) trivially
/// passes (`windows(2)` yields nothing to check).
///
/// The one identity two independent folds both lean on: `classify_packed_row_block`'s
/// reduce-dim fold (below) and [`classify_tiled_gemm`]'s token/feature-axis-group
/// fold both need "a single flat index times the innermost axis's stride
/// addresses the same memory a full per-axis decomposition would" to be
/// true, and it is true exactly when this check passes — never a special
/// case for how many dims fold, or for reduce vs. output axes.
fn axes_fold_contiguously(dims: &[u16], extents: &[u64], layout: &Layout) -> bool {
    dims.windows(2).all(|window| {
        let [outer, inner] = window else {
            return false;
        };
        let inner_extent = extents[*inner as usize] as i64;
        layout.stride(*outer) == inner_extent * layout.stride(*inner)
    })
}

/// The one decision [`packed_row_block`] and [`diagnose_packed_row_block`]
/// both need — this function is the single source of truth;
/// `packed_row_block` is `.ok()` over it so there is exactly one place the
/// seven conditions are spelled out, never two copies that could drift.
fn classify_packed_row_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Result<PackedRowBlock, PackedRowBlockRejection> {
    if !reduce_is_cooperative(resolved) {
        return Err(PackedRowBlockRejection::NotCooperativeReduce);
    }
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        output_axes,
        out_layout,
        ..
    } = &resolved.kind
    else {
        return Err(PackedRowBlockRejection::NotReduceKeepReduce);
    };
    if quantized.len() != 2 {
        return Err(PackedRowBlockRejection::OperandCountNotTwo);
    }
    let packed: Vec<(usize, PackedCodec)> = quantized
        .iter()
        .enumerate()
        .filter_map(|(index, codec)| codec.map(|codec| (index, codec)))
        .collect();
    let [(weight, codec)] = packed[..] else {
        return Err(PackedRowBlockRejection::NotExactlyOnePackedOperand);
    };
    // Whitelist the K-quant family explicitly rather than blacklisting one
    // non-K-quant codec by `==` -- an equality check against `Q8_0` alone
    // would have silently admitted `Q4_0` (or any future flat-block codec)
    // the moment its extent happened to be a multiple of 256. This match
    // is exhaustive over `PackedCodec`, so a new codec added later forces a
    // decision here instead of slipping through.
    match codec {
        PackedCodec::Q3K | PackedCodec::Q4K | PackedCodec::Q5K | PackedCodec::Q6K => {}
        PackedCodec::Q2K
        | PackedCodec::Q8_0
        | PackedCodec::Q4_0
        | PackedCodec::Float16
        | PackedCodec::BFloat16 => {
            return Err(PackedRowBlockRejection::NotKQuantCodec);
        }
    }
    let other = 1 - weight;
    let reduce_dims: Vec<u16> = (0..resolved.extents.len() as u16)
        .filter(|dim| !output_axes.contains(dim))
        .collect();
    let Some(&innermost) = reduce_dims.last() else {
        return Err(PackedRowBlockRejection::NotExactlyOneReduceDim { reduce_dims });
    };
    // MULTIPLE reduce dims are only a single logical reduction if they are
    // CONTIGUOUS in memory for BOTH operands: `attn_output`'s own reduce
    // folds three axes (kv-head-group x query-group x head-dim) that are
    // exactly the row-major decomposition of one 4096-wide embedding axis
    // (`docs/discipline.md`'s "print the gate, don't infer it" table: weight
    // strides `[512, 128, 1]` against extents `[8, 4, 128]` — each outer
    // dim's stride equals the product of every dim nested inside it). The
    // row-blocked kernel body walks the flattened `reduction_total` range
    // with ONE stride per operand (`crate::metal::pack_reduce_uniforms`
    // already packs `reduction_total` as the product across every reduce
    // dim, generic in dim count), so folding is sound exactly when this
    // check passes — never a special case for three dims specifically.
    for operand in [weight, other] {
        let layout = &resolved.operands()[operand].1;
        if !axes_fold_contiguously(&reduce_dims, &resolved.extents, layout) {
            return Err(PackedRowBlockRejection::ReduceDimsNotContiguous {
                reduce_dims: reduce_dims.clone(),
            });
        }
    }
    // the packed operand must be contiguous along the INNERMOST (fastest)
    // reduce dim (its super-blocks run along `k`), and the flattened extent
    // across every folded reduce dim must be whole super-blocks.
    let stride = resolved.operands()[weight].1.stride(innermost);
    if stride != 1 {
        return Err(PackedRowBlockRejection::NonUnitWeightStride { stride });
    }
    let extent: u64 = reduce_dims
        .iter()
        .map(|&dim| resolved.extents[dim as usize])
        .product();
    if !(extent as usize).is_multiple_of(Q4K_BLOCK_ELEMENTS) {
        return Err(PackedRowBlockRejection::ExtentNotBlockMultiple { extent });
    }
    let weight_layout = &resolved.operands()[weight].1;
    let other_layout = &resolved.operands()[other].1;
    let (token_axes, feature_axes) = split_token_feature_axes(
        output_axes,
        weight_layout,
        other_layout,
        out_layout,
        &resolved.extents,
    )
    .unwrap_or_else(|| (Vec::new(), output_axes.to_vec()));
    if gather_count(resolved) != 0
        && token_axes
            .iter()
            .map(|&axis| resolved.extents[axis as usize])
            .product::<u64>()
            > 1
    {
        return Err(PackedRowBlockRejection::GatheredOperand);
    }
    Ok(PackedRowBlock {
        weight,
        other,
        reduce_dim: innermost as usize,
        codec,
        token_axes,
        feature_axes,
    })
}

fn packed_row_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Option<PackedRowBlock> {
    // The experimental mixed-source path must use the descriptor-aware
    // element reader; row-block pointer hoisting has a separate address ABI.
    if std::env::var_os("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES").is_some() {
        return None;
    }
    classify_packed_row_block(resolved, quantized).ok()
}

/// Public diagnostic seam for `classify_tiled_gemm`, same shape as
/// [`PackedRowBlockRejection`] for `classify_packed_row_block`: one variant
/// per `return`/`None` site in that function, in the order they are checked,
/// so a caller printing `{rejection:?}` sees exactly which condition gave up
/// on a real op instead of an inferred guess. `NotPackedRowBlock` wraps the
/// more basic gate's own rejection when that one fails first -- the tiled
/// path can never be more permissive than the row-blocked path it narrows.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TiledGemmRejection {
    /// The `metal-tiled-gemm` feature is not compiled in -- the tiled path
    /// does not exist as far as this build can observe (see
    /// `classify_tiled_gemm`'s own doc).
    FeatureDisabled,
    /// `classify_packed_row_block` itself rejected first; the tiled path
    /// can only narrow that gate's `Ok`, never rescue its `Err`.
    NotPackedRowBlock(PackedRowBlockRejection),
    /// The packed operand's codec is not [`PackedCodec::Q4K`] -- Q5_K/Q6_K
    /// have no batched-unpack helper for this path yet (see
    /// `classify_tiled_gemm`'s own comment).
    NotQ4K,
    /// `reduce_op`/`init` are not the plain `Add`-from-`Zero` shape
    /// `simdgroup_matrix` accumulation requires.
    NotAddZeroReduce,
    /// `is_plain_product_reduce` is false -- the fused body carries more
    /// than a bare `weight * activation` product.
    NotPlainProductReduce,
    /// Every output axis partitions into a token group (activation-owned)
    /// and a feature group (weight-owned) by nonzero-stride ownership; an
    /// axis neither or both operands depend on, an empty group, or the two
    /// groups interleaving in `output_axes` rather than token-group-then-
    /// feature-group (`native_packed_layout`'s own convention) is a
    /// broadcast/ordering shape this restricted path has never been
    /// measured against.
    AxisOwnershipAmbiguous,
    /// The token or feature group has more than one axis, but they do NOT
    /// nest contiguously (for the owning operand, or for the op's own
    /// output layout) — see `axes_fold_contiguously`.
    AxisGroupNotContiguous,
    /// The token group's flattened extent is below
    /// `crate::sized::TILED_GEMM_MIN_TOKENS` -- tiling overhead is not
    /// amortized at this size.
    TokenExtentBelowMinimum { token_extent: u64, min_tokens: u64 },
}

/// The additional narrowing [`push_tiled_gemm_body`]'s `simdgroup_matrix`
/// path requires on top of [`packed_row_block`]'s own row-blocked
/// eligibility -- the one decision [`grid_threads`] and
/// [`push_cooperative_reduce_body`] must reach IDENTICALLY, same discipline
/// [`PackedRowBlock`] itself follows (see its own doc): this reads a
/// CONCRETE extent (the activation/token axis, against
/// `crate::sized::TILED_GEMM_MIN_TOKENS`) on top of `packed_row_block`'s
/// own concrete-stride gate, so [`kernel_cache_key`] re-derives this too
/// rather than caching by structure alone (`docs/discipline.md` ROW 107).
struct TiledGemmBlock {
    // only [`push_tiled_gemm_body`] reads these -- gated the same as that
    // function, so the non-feature build does not carry never-read fields.
    #[cfg(feature = "metal-tiled-gemm")]
    weight: usize,
    #[cfg(feature = "metal-tiled-gemm")]
    other: usize,
    #[cfg(feature = "metal-tiled-gemm")]
    reduce_dim: usize,
    /// every output axis the ACTIVATION owns exclusively (nonzero stride on
    /// `other`, zero on `weight`), outermost first -- more than one only
    /// when [`axes_fold_contiguously`] validated them as one flattened
    /// block, the identical identity [`classify_packed_row_block`]'s own
    /// reduce-dim fold relies on. The tile loop's N side walks the
    /// flattened product of these.
    token_axes: Vec<u16>,
    /// every output axis the WEIGHT owns exclusively (nonzero stride on
    /// `weight`, zero on `other`), outermost first -- `attn_q`/`attn_k`/
    /// `attn_v`'s own `heads`/`head_dim` split folds here the same way
    /// `attn_output`'s reduce already folds three axes. The tile loop's M
    /// side walks the flattened product of these.
    feature_axes: Vec<u16>,
}

/// `resolved`/`quantized`/`reduce_op`/`init`/`output_axes` are exactly
/// [`push_cooperative_reduce_body`]'s own parameters -- this and
/// [`packed_row_block`] are the two gates that function consults, in order,
/// before falling back to the fully generic cooperative-reduce path.
///
/// Feature-gated: without `metal-tiled-gemm`,
/// [`crate::sized::TILED_GEMM_MIN_TOKENS`] does not exist (see that
/// constant's own doc), so this always returns
/// `Err(TiledGemmRejection::FeatureDisabled)` and every dispatch keeps taking
/// the row-blocked or generic path exactly as it does today — the tiled
/// kernel does not exist as far as the rest of this module can observe.
fn classify_tiled_gemm(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
) -> Result<TiledGemmBlock, TiledGemmRejection> {
    #[cfg(not(feature = "metal-tiled-gemm"))]
    {
        let _ = (resolved, quantized, reduce_op, init, output_axes);
        Err(TiledGemmRejection::FeatureDisabled)
    }
    #[cfg(feature = "metal-tiled-gemm")]
    {
        let PackedRowBlock {
            weight,
            other,
            reduce_dim,
            codec,
            ..
        } = classify_packed_row_block(resolved, quantized)
            .map_err(TiledGemmRejection::NotPackedRowBlock)?;
        // Q4_K only -- Q5_K/Q6_K have no batched-unpack helper yet
        // (`push_packed_row_blocked_body`'s own comment on their arms) and,
        // more to the point, have never been measured on this path.
        // Shipping them unmeasured on a correctness-critical GPU kernel
        // would violate the same discipline this landing's own gate
        // demands (principle 18).
        if codec != PackedCodec::Q4K {
            return Err(TiledGemmRejection::NotQ4K);
        }
        // `simdgroup_multiply_accumulate` IS a sum-of-products -- there is
        // no hardware knob for `Maximum`/`Subtract`/etc, so this only ever
        // applies to the exact shape a real matmul takes: an `Add`-reduce
        // over a plain `weight * activation` body, seeded from zero. Every
        // other combination keeps taking the row-blocked or generic path.
        if reduce_op != ScalarOp::Add || init != ReduceInit::Zero {
            return Err(TiledGemmRejection::NotAddZeroReduce);
        }
        if !is_plain_product_reduce(resolved, reduce_op, weight, other) {
            return Err(TiledGemmRejection::NotPlainProductReduce);
        }
        // A plain matmul: every output axis is EITHER token (activation-
        // owned) or feature (weight-owned) -- never both, never neither.
        // `attn_q`/`attn_k`/`attn_v` keep TWO weight-owned axes (`heads` and
        // `head_dim`, split by the einsum but one flat out-features run on
        // disk); folding them the same way `classify_packed_row_block`
        // already folds `attn_output`'s three reduce axes is what lets this
        // path reach them at all (ROW 114 -- ROW 107's "documented scope
        // limit" was this fold, not yet written).
        let weight_layout = &resolved.operands()[weight].1;
        let other_layout = &resolved.operands()[other].1;
        let mut token_axes: Vec<u16> = Vec::new();
        let mut feature_axes: Vec<u16> = Vec::new();
        for &axis in output_axes {
            match (
                weight_layout.stride(axis) == 0,
                other_layout.stride(axis) == 0,
            ) {
                (true, false) => token_axes.push(axis),
                (false, true) => feature_axes.push(axis),
                _ => return Err(TiledGemmRejection::AxisOwnershipAmbiguous),
            }
        }
        if token_axes.is_empty() || feature_axes.is_empty() {
            return Err(TiledGemmRejection::AxisOwnershipAmbiguous);
        }
        // `native_packed_layout`'s own doc: a packed weight's on-disk layout
        // is `[out_dim, in_dim]` row-major, reconstructed by walking
        // `output_axes` so the "out" (feature) side must sit LAST, after
        // every token axis -- checked here as "the two groups reassemble
        // `output_axes` in order", which also catches an interleaved shape
        // (token/feature/token) this path has never been measured against.
        let reassembled: Vec<u16> = token_axes
            .iter()
            .chain(feature_axes.iter())
            .copied()
            .collect();
        if reassembled != output_axes {
            return Err(TiledGemmRejection::AxisOwnershipAmbiguous);
        }
        // A group with more than one axis is only a single logical token/
        // feature dimension if it nests contiguously -- same identity
        // `classify_packed_row_block`'s reduce-dim fold already leans on,
        // checked for the OWNING operand (the other operand's stride is
        // uniformly zero across the group, trivially "contiguous") AND for
        // the op's own output layout, since the tile write-back below also
        // walks the flattened group with one stride.
        let BoundOpKind::Reduce { out_layout, .. } = &resolved.kind else {
            return Err(TiledGemmRejection::NotPackedRowBlock(
                PackedRowBlockRejection::NotReduceKeepReduce,
            ));
        };
        let groups_contiguous =
            axes_fold_contiguously(&token_axes, &resolved.extents, other_layout)
                && axes_fold_contiguously(&feature_axes, &resolved.extents, weight_layout)
                && axes_fold_contiguously(&token_axes, &resolved.extents, out_layout)
                && axes_fold_contiguously(&feature_axes, &resolved.extents, out_layout);
        if !groups_contiguous {
            return Err(TiledGemmRejection::AxisGroupNotContiguous);
        }
        let token_extent: u64 = token_axes
            .iter()
            .map(|&axis| resolved.extents[axis as usize])
            .product();
        if token_extent < crate::sized::TILED_GEMM_MIN_TOKENS {
            return Err(TiledGemmRejection::TokenExtentBelowMinimum {
                token_extent,
                min_tokens: crate::sized::TILED_GEMM_MIN_TOKENS,
            });
        }
        Ok(TiledGemmBlock {
            weight,
            other,
            reduce_dim,
            token_axes,
            feature_axes,
        })
    }
}

fn tiled_gemm_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
) -> Option<TiledGemmBlock> {
    if std::env::var_os("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES").is_some() {
        return None;
    }
    classify_tiled_gemm(resolved, quantized, reduce_op, init, output_axes).ok()
}

/// Public diagnostic seam: which condition, if any, rejected `resolved` from
/// the tiled-GEMM `simdgroup_matrix` kernel. `Ok(())` means it WOULD take (or
/// does take) the tiled path. Same shape as [`diagnose_packed_row_block`],
/// one narrowing further -- see [`TiledGemmRejection`]'s own doc.
///
/// # Errors
/// Returns the specific [`TiledGemmRejection`] gate that rejected this op.
#[cfg(feature = "instrument")]
pub fn diagnose_tiled_gemm_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
) -> Result<(), TiledGemmRejection> {
    classify_tiled_gemm(resolved, quantized, reduce_op, init, output_axes).map(drop)
}

/// Public diagnostic seam: which condition, if any, rejected `resolved`
/// from the row-blocked packed kernel. `Ok(())` means it WOULD take (or
/// does take) the fast path. Behind `instrument` — see
/// [`PackedRowBlockRejection`]'s own doc for why.
///
/// # Errors
/// Returns the specific [`PackedRowBlockRejection`] gate that rejected this
/// op.
#[cfg(feature = "instrument")]
pub fn diagnose_packed_row_block(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
) -> Result<(), PackedRowBlockRejection> {
    classify_packed_row_block(resolved, quantized).map(drop)
}

/// Thread count [`grid_threads`]' tiled-GEMM arm dispatches -- one
/// `TILED_GEMM_NSG * SIMD_WIDTH`-thread threadgroup per
/// `crate::sized::TILED_GEMM_BLOCK_M x TILED_GEMM_BLOCK_N` output tile,
/// tiling both `feature_extent` and `token_extent`. Only ever called from
/// behind `tiled_gemm_block(..).is_some()` (`grid_threads`' own call site),
/// which is itself only `Some` behind `feature = "metal-tiled-gemm"` (see
/// [`classify_tiled_gemm`]'s doc) -- the `#[cfg(not(..))]` arm is therefore
/// as unreachable as [`push_tiled_gemm_body`]'s own stub, for the same
/// reason.
fn tiled_gemm_threadgroups(
    node: NodeId,
    feature_extent: u64,
    token_extent: u64,
) -> Result<u64, EmitError> {
    #[cfg(not(feature = "metal-tiled-gemm"))]
    {
        let _ = (feature_extent, token_extent);
        Err(EmitError::TiledGemmFeatureDisabled { node })
    }
    #[cfg(feature = "metal-tiled-gemm")]
    {
        let _ = node;
        let row_tiles = feature_extent.div_ceil(crate::sized::TILED_GEMM_BLOCK_M);
        let col_tiles = token_extent.div_ceil(crate::sized::TILED_GEMM_BLOCK_N);
        Ok(row_tiles * col_tiles * (TILED_GEMM_NSG as u64) * SIMD_WIDTH)
    }
}

/// Split-K factor for a row-blocked packed matmul with `rows` OUTPUT rows
/// (`output_total`), dispatching `base_simdgroups` simdgroups
/// (`rows.div_ceil(codec.rows_per_simdgroup())`) -- how many simdgroups per
/// threadgroup cooperate on ONE row-group's reduction axis so the
/// dispatch's total simdgroup count reaches
/// [`crate::sized::PACKED_ROW_SPLIT_K_TARGET_SIMDGROUPS`], capped at
/// [`crate::sized::PACKED_ROW_SPLIT_K_MAX_SPLIT`].
///
/// Gated FIRST by [`crate::sized::PACKED_ROW_SPLIT_K_MAX_ROWS`] (`0` means
/// no ceiling): `target_simdgroups` alone lets integer division push a
/// shape's factor toward `1` as `rows` grows, but that fall-off is gradual,
/// not a hard cutoff -- a 4096-row op (attn_q/attn_output/ffn_down in the
/// measured decode-graph table) still clears `factor=2` on
/// `target_simdgroups` arithmetic alone even though its GB/s was already
/// close to the `ffn_*` rate at 1024 base simdgroups, which is exactly the
/// "applied to every packed matvec" loss this ceiling exists to cut off at
/// a build-time-tunable row count instead. Always `1` once
/// `base_simdgroups` already meets the target OR `rows` exceeds the
/// ceiling. Feature-off builds never see this: the caller only reaches
/// here behind `packed_row_block(..).is_some()` AND the `metal-q4k-split-k`
/// feature (see `push_cooperative_reduce_body`'s own gate).
#[cfg(feature = "metal-q4k-split-k")]
fn packed_row_split_factor(base_simdgroups: u64, rows: u64) -> u64 {
    if base_simdgroups == 0 {
        return 1;
    }
    let max_rows = crate::sized::PACKED_ROW_SPLIT_K_MAX_ROWS;
    if max_rows != 0 && rows > max_rows {
        return 1;
    }
    (crate::sized::PACKED_ROW_SPLIT_K_TARGET_SIMDGROUPS / base_simdgroups)
        .clamp(1, crate::sized::PACKED_ROW_SPLIT_K_MAX_SPLIT)
}

/// The `metal-q4k-split-k`-off arm: split-K never engages, so the factor is
/// always `1` -- see [`packed_row_split_factor`]'s feature-on twin for the
/// real policy. Kept as a separate function (not a `cfg!()` branch inline)
/// so [`crate::sized::PACKED_ROW_SPLIT_K_TARGET_SIMDGROUPS`] is never
/// referenced from a build that never generated it.
#[cfg(not(feature = "metal-q4k-split-k"))]
fn packed_row_split_factor(_base_simdgroups: u64, _rows: u64) -> u64 {
    1
}

/// Single source of truth for the row-blocked packed path's base simdgroup
/// count (one per [`PackedCodec::rows_per_simdgroup`] feature rows, tiled
/// again by `ceil(token_total / crate::sized::PACKED_ROW_ACTIVATION_GROUP)`
/// once more than one activation row folds per streamed weight row) and its
/// derived split-K factor -- both [`grid_threads`] and
/// [`tiled_gemm_threadgroup_width`] need the SAME pair, and
/// [`PackedRowBlock`]'s own doc already names the hazard of two independent
/// call sites silently disagreeing. `token_total == 1` (no distinct token
/// axis, or exactly one activation row) collapses the token factor to `1`,
/// so a caller passing `feature_total` for the whole output and `token_total
/// == 1` gets today's byte-identical single-row dispatch shape.
fn packed_row_dispatch(feature_total: u64, token_total: u64, codec: PackedCodec) -> (u64, u64) {
    let base = feature_total.div_ceil(codec.rows_per_simdgroup() as u64);
    let split = packed_row_split_factor(base, feature_total);
    let token_groups = token_total.div_ceil(crate::sized::PACKED_ROW_ACTIVATION_GROUP);
    (base * token_groups, split)
}

fn grid_threads(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    numeric_policy: NumericPolicy,
) -> Result<u64, EmitError> {
    let threads = match &resolved.kind {
        BoundOpKind::CachedAttention {
            head_dim,
            cached_key_rows,
            new_key_rows,
            ..
        } => {
            // The single-range fused path (nine operands) always dispatches
            // the compiled MAXIMUM chunk count (`cap`) -- redesign §5 option
            // 2, `render_cached_attention`'s own doc -- so idle simdgroups
            // above the live `context_chunks_for(...)` result are always
            // present in the grid and contribute the merge's identity
            // partial rather than being sized out of the dispatch. Redesign
            // §4c: `splits` multiplies the THREADGROUP count (this product
            // divided by `threadgroup_width`, `tiled_gemm_threadgroup_
            // width`'s own `CachedAttention` arm) by the compiled maximum
            // `ATTENTION_SPLIT_MAX` ONLY when [`cached_attention_merge_needed`]
            // holds for THIS op's own context length -- unlike `chunks`
            // (whose idle simdgroups stay inside one threadgroup and merge
            // there, so widening them costs nothing under `bit_exact`), an
            // idle SPLIT is a whole extra threadgroup that would otherwise
            // race the real one to write `out` directly (the single-dispatch
            // `final_store` branch has no scratch hop to arbitrate that
            // race). Gating this exactly like `render_cached_attention`'s own
            // `final_store` branch keeps grid width and kernel body in
            // lock-step: both flip together, at the same context length.
            let context_length = *cached_key_rows + *new_key_rows;
            // Only the single-range fused path (nine operands, `cached_key_rows
            // == 0`) widens to the compiled cap -- `two_range_cached_bound`
            // (nine operands, `cached_key_rows != 0`) already dispatches
            // against a bucket-padded, compile-time-fixed `context_length`, so
            // its live `chunks`/`splits` never grow between calls the way the
            // single-range path's do (`render_cached_attention`'s own doc).
            let single_range_dynamic = resolved.operands().len() == 9 && *cached_key_rows == 0;
            let (chunks, splits) = if single_range_dynamic {
                (
                    crate::sized::ATTENTION_CONTEXT_CHUNK_CAP,
                    if cached_attention_merge_needed(&resolved.kind, numeric_policy) {
                        crate::sized::ATTENTION_SPLIT_MAX
                    } else {
                        1
                    },
                )
            } else {
                (context_chunks_for(context_length, numeric_policy), 1)
            };
            resolved
                .extents
                .iter()
                .product::<u64>()
                .checked_div(*head_dim)
                .unwrap_or(0)
                * chunks
                * splits
                * SIMD_WIDTH
        }
        BoundOpKind::Elementwise { .. } => resolved.extents.iter().product(),
        BoundOpKind::Reduce {
            keep: Keep::Reduce,
            reduce_op,
            init,
            output_axes,
            ..
        } => {
            let output_total: u64 = output_axes
                .iter()
                .map(|dim| resolved.extents[*dim as usize])
                .product();
            if let Some(block) =
                tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes)
            {
                // TILED_GEMM_NSG simdgroups per BLOCK_M x BLOCK_N output
                // tile, tiled over BOTH the feature axis and the token axis
                // — the amortization the row-blocked path does not do (it
                // tiles the feature axis alone; see `push_tiled_gemm_body`'s
                // doc).
                tiled_gemm_threadgroups(
                    resolved.node,
                    block
                        .feature_axes
                        .iter()
                        .map(|&axis| resolved.extents[axis as usize])
                        .product(),
                    block
                        .token_axes
                        .iter()
                        .map(|&axis| resolved.extents[axis as usize])
                        .product(),
                )?
            } else if let Some(block) = packed_row_block(resolved, quantized) {
                // one simdgroup per `block.codec.rows_per_simdgroup()`
                // feature rows, times the split-K factor (1 = no-op unless
                // `metal-q4k-split-k` is active AND this shape is below the
                // target simdgroup count), tiled again by
                // `ceil(token_total / PACKED_ROW_ACTIVATION_GROUP)` once
                // `block.token_axes` folds more than one activation row per
                // streamed weight row -- `token_total == 1` collapses that
                // factor to `1`, the byte-identical single-row shape.
                let feature_total: u64 = block
                    .feature_axes
                    .iter()
                    .map(|&axis| resolved.extents[axis as usize])
                    .product();
                let token_total = packed_row_block_token_total(&block, &resolved.extents);
                let (base, split) = packed_row_dispatch(feature_total, token_total, block.codec);
                base * SIMD_WIDTH * split
            } else if reduce_is_cooperative(resolved) {
                // one cooperative-reduce threadgroup per output element,
                // `cooperative_reduce_width` lanes wide (SIMD_WIDTH with
                // `metal-wide-cooperative-reduce` off, matching
                // `reduce_is_cooperative`'s prior doc byte-for-byte) — see
                // that function's own doc for the scaling policy.
                let reduce_dims = reduction_dims(resolved, output_axes);
                output_total * cooperative_reduce_width(resolved, quantized, &reduce_dims)
            } else {
                output_total
            }
        }
        BoundOpKind::Reduce {
            keep: Keep::Scan, ..
        } => {
            let rank = resolved.extents.len();
            resolved.extents[..rank.saturating_sub(1)].iter().product()
        }
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => resolved.extents.iter().product(),
    };
    Ok(threads)
}

/// The MSL scalar type a `BoundOp`'s own dtype declares its buffers,
/// scratch array, and accumulator as. `Float16` is the one narrower type
/// this backend emits (`half`, MSL's IEEE-754 binary16) — every other
/// dtype that already reached the "float" bucket before `DType` widened
/// keeps emitting `float`, matching this module's stance before `BoundOp`
/// carried a dtype at all. `omega::execute`'s upstream gate is what keeps
/// anything other than `Float32`/`Float16` from ever reaching [`emit`], so
/// those are the only two cases that matter in practice, but the match
/// stays total over every [`DType`] variant rather than assuming that gate
/// ran — a width this backend has never emitted (the 64/128-bit integers,
/// `Float64`) is rejected here by name instead of silently folded into the
/// 4-byte `float` bucket it does not fit.
fn type_token(node: NodeId, dtype: DType) -> Result<&'static str, EmitError> {
    match dtype {
        DType::Float16 => Ok("half"),
        DType::Float32
        | DType::BFloat16
        | DType::Bool
        | DType::Int8
        | DType::UInt8
        | DType::Int32
        | DType::UInt32 => Ok("float"),
        DType::Int16
        | DType::UInt16
        | DType::Int64
        | DType::UInt64
        | DType::Int128
        | DType::UInt128
        | DType::Float64 => Err(EmitError::UnsupportedDType { node, dtype }),
    }
}

/// A structural fingerprint, not a hash of anything runtime: rank, operand
/// count, every `ScalarOp`/`ReduceInit`/`Keep` involved, and — since a gather
/// changes the generated source (extra buffer params, extra uniforms, extra
/// fetch code) — which operands gather. That last part is a suffix appended
/// only when at least one operand gathers, so a gather-free `BoundOp`'s name is
/// unchanged from before this existed.
fn entry_name(resolved: &BoundOp) -> String {
    let rank = resolved.extents.len();
    let operand_count = resolved.operands().len();
    let base = match &resolved.kind {
        BoundOpKind::CachedAttention {
            query_rows,
            cached_key_rows,
            new_key_rows,
            kv_heads,
            query_groups,
            head_dim,
            scale,
            cached_lower_inclusive,
            new_upper_inclusive,
            ..
        } => {
            // `operand_count == 9` names a runtime ninth operand, but that
            // operand carries two DIFFERENT scalars discriminated by
            // `cached_key_rows` (`BoundOpKind::CachedAttention`'s own doc):
            // `cached_key_rows == 0` is `single_range_dynamic` -- the ninth
            // operand IS the real `new_upper_inclusive`, so the static field
            // here is unused filler and the key names the STRUCTURE ("dyn")
            // rather than that filler value. `cached_key_rows != 0` is
            // `two_range_cached_bound` -- the ninth operand is the CACHED
            // range's own live row count, but `new_upper_inclusive` is still
            // the real, query-independent compiled bound (this path's causal
            // band never depends on it), so that token is real, not "dyn".
            let single_range_dynamic = operand_count == 9 && *cached_key_rows == 0;
            let two_range_cached_bound = operand_count == 9 && *cached_key_rows != 0;
            let upper_token = if single_range_dynamic {
                "dyn".to_string()
            } else {
                signed_name_part(*new_upper_inclusive)
            };
            if single_range_dynamic {
                // `_b{width}` names the build-time block-staging width
                // (`block_width_for`) -- a build-time constant, so a build
                // whose `OMEGA_ATTENTION_BLOCK_WIDTH` override changed emits
                // a distinct name rather than reusing a cached kernel
                // compiled for a different in-block staging width. `_qh{0|1}`
                // names `cached_attention_per_query_head_grid`'s own decision
                // -- unlike the row counts this token intentionally reduces
                // the bucket extent to a boolean, never the extent itself, so
                // every bucket ON ONE SIDE of the knee still shares a single
                // compiled kernel, but the two regimes render genuinely
                // different MSL text (a different `tgid` decode -- see
                // `render_cached_attention`'s own doc) and so cannot be
                // allowed to collide on the same cache key. On this SAME
                // path, `cached_key_rows`/`new_key_rows` are also runtime
                // `Uniforms` fields now (redesign §5 option 2's
                // `render_cached_attention`), so the row-count tokens drop
                // from the name entirely -- two different `kv-capacity-
                // bucket` extents share one compiled kernel, keyed only by
                // the compiled MAXIMUM chunk count (`_x{cap}`), never by the
                // live capacity.
                let per_query_head_grid =
                    cached_attention_per_query_head_grid(true, *cached_key_rows + *new_key_rows);
                format!(
                    "omega_cached_attention_q{query_rows}_h{kv_heads}_g{query_groups}_d{head_dim}_s{:08x}_l{}_u{upper_token}_x{}_b{}_qh{}",
                    scale.to_bits(),
                    signed_name_part(*cached_lower_inclusive),
                    crate::sized::ATTENTION_CONTEXT_CHUNK_CAP,
                    crate::sized::ATTENTION_BLOCK_WIDTH,
                    u8::from(per_query_head_grid),
                )
            } else if two_range_cached_bound {
                // `_cb` marks the ninth-operand, runtime-`cached_key_rows`
                // body (`long cached_key_rows = (long)in8[0];` --
                // `render_cached_attention`'s own doc) as structurally
                // distinct from the eight-operand, fully-`constexpr` body
                // below: same row counts, same upper token, same grid, but a
                // different buffer signature (the extra `in8` param) and a
                // different generated statement for `cached_key_rows`, so
                // the two must never share a compiled pipeline.
                format!(
                    "omega_cached_attention_q{query_rows}_c{cached_key_rows}_n{new_key_rows}_h{kv_heads}_g{query_groups}_d{head_dim}_s{:08x}_l{}_u{upper_token}_cb",
                    scale.to_bits(),
                    signed_name_part(*cached_lower_inclusive),
                )
            } else {
                format!(
                    "omega_cached_attention_q{query_rows}_c{cached_key_rows}_n{new_key_rows}_h{kv_heads}_g{query_groups}_d{head_dim}_s{:08x}_l{}_u{upper_token}",
                    scale.to_bits(),
                    signed_name_part(*cached_lower_inclusive),
                )
            }
        }
        BoundOpKind::Elementwise { .. } => {
            let body = body_token(resolved.element_body());
            format!("omega_elementwise_r{rank}_n{operand_count}_{body}")
        }
        BoundOpKind::Reduce {
            reduce_op,
            init,
            keep,
            output_axes,
            epilogue_body,
            epilogue_operands,
            ..
        } => {
            let body = body_token(resolved.element_body());
            let kind = keep_token(*keep);
            let reduce_body = op_token(*reduce_op);
            let init = init_token(*init);
            // `rank` alone does not fix the output/reduce split -- two folds
            // over the same total rank can keep a different number of axes
            // (e.g. one output axis folding three vs one folding one), which
            // sizes `output_extents`/`reduction_extents` differently in
            // `render_reduce`'s own uniform struct. Without `output_rank`
            // here, two such ops would share this name despite emitting
            // different source -- see `distinct_output_rank_at_same_total_rank_yields_distinct_entry_names`.
            let output_rank = output_axes.len();
            // A fused epilogue changes both the `Uniforms` layout (the extra
            // `epilogue_operand_base`/`_strides` fields) and the body text
            // (`push_reduce_epilogue_write`'s emitted tail) -- the untouched
            // identity default contributes nothing here, so a program with
            // no fused epilogue anywhere names exactly what it always did.
            let epilogue = if reduce_epilogue_is_identity(epilogue_body, epilogue_operands) {
                String::new()
            } else {
                format!(
                    "_epi{}_{}",
                    epilogue_operands.len(),
                    body_token(epilogue_body)
                )
            };
            format!(
                "omega_{kind}_r{rank}_o{output_rank}_n{operand_count}_{body}_{reduce_body}_{init}{epilogue}"
            )
        }
        // no operand count, no body: an `Iota`'s whole structure is its
        // rank (always 1 in practice, since `Op::Iota` resolves one
        // `Extent` — see `op.rs`'s doc — but this reads `extents.len()`
        // rather than assuming that, matching every other arm here).
        BoundOpKind::Iota => format!("omega_iota_r{rank}"),
        // the literal is baked into the source (see `render_constant`), so
        // it has to be part of the entry name too - otherwise two constants
        // of the same rank would share one cached kernel and the second
        // would run the first one's value. Raw bits, not the decimal, so
        // the name is exact and identifier-safe.
        BoundOpKind::Constant { value } => {
            format!("omega_constant_r{rank}_v{:08x}", value.to_bits())
        }
    };
    let gather_bits: String = resolved
        .operands()
        .iter()
        .map(|(_, _, gather)| if gather.is_some() { '1' } else { '0' })
        .collect();
    if gather_bits.contains('1') {
        format!("{base}_g{gather_bits}")
    } else {
        base
    }
}

fn scalar_op_expr(op: ScalarOp, args: &[&str]) -> String {
    match op {
        ScalarOp::Identity => (*args.first().unwrap_or(&"0.0f")).into(),
        ScalarOp::Add => format!("({} + {})", args[0], args[1]),
        ScalarOp::Subtract => format!("({} - {})", args[0], args[1]),
        ScalarOp::Multiply => format!("({} * {})", args[0], args[1]),
        ScalarOp::Divide => format!("({} / {})", args[0], args[1]),
        ScalarOp::Maximum => format!("max({}, {})", args[0], args[1]),
        ScalarOp::Minimum => format!("min({}, {})", args[0], args[1]),
        ScalarOp::Negate => format!("(-{})", args[0]),
        ScalarOp::Reciprocal => format!("(1.0f / {})", args[0]),
        ScalarOp::Exponential => format!("exp({})", args[0]),
        ScalarOp::Logarithm => format!("log({})", args[0]),
        ScalarOp::SquareRoot => format!("sqrt({})", args[0]),
        ScalarOp::Tanh => format!("tanh({})", args[0]),
        ScalarOp::Erf => format!("proxima_erf({})", args[0]),
        ScalarOp::Greater => format!("(({} > {}) ? 1.0f : 0.0f)", args[0], args[1]),
        ScalarOp::Equal => format!("((fabs({} - {}) == 0.0f) ? 1.0f : 0.0f)", args[0], args[1]),
        ScalarOp::Select => format!("(({} != 0.0f) ? {} : {})", args[0], args[1], args[2]),
    }
}

/// `(init expression, seeded-from-the-start)`. `FirstElement` mirrors
/// `cpu::initial_value`/`cpu::run_reduce`'s `seeded` flag: the accumulator
/// starts unseeded and is instead set from the first reduction step's value —
/// the init expression here is never actually read in that case.
fn fold_init_tokens(init: ReduceInit) -> (&'static str, &'static str) {
    match init {
        ReduceInit::Zero => ("0.0f", "true"),
        ReduceInit::One => ("1.0f", "true"),
        ReduceInit::NegativeInfinity => ("-INFINITY", "true"),
        ReduceInit::PositiveInfinity => ("INFINITY", "true"),
        ReduceInit::FirstElement => ("0.0f", "false"),
    }
}

/// Emits one `float step{n} = ...;` declaration per [`ComposedBody`] step,
/// each reading `scratch[i]` for an `Operand` arg or an earlier `step{k}`
/// for a `Step` arg — the MSL counterpart of `cpu::apply_body`'s scratch
/// walk. Returns the C expression for the body's own result (its last
/// step), which a caller splices directly into whatever it does with the
/// value (`out[gid] = ...` for elementwise, `float value = ...` for a
/// reduce/scan step).
fn push_body_steps(
    source: &mut String,
    body: &ComposedBody,
    indent: &str,
    element_type: &str,
) -> String {
    crate::epilogue::declare_steps(
        source,
        body,
        "scratch",
        "step",
        scalar_op_expr,
        |source, index, expr| {
            source.push_str(&format!("{indent}{element_type} step{index} = {expr};\n"));
        },
    )
}

// only the row-blocked packed-matmul path's caller ever passes `true` --
// every other kernel keeps the same signature it always has, so this stays
// off by construction wherever split-K does not engage (see
// `push_cooperative_reduce_body`'s own call site for the gate).
fn kernel_signature(
    source: &mut String,
    quantized: &[Option<PackedCodec>],
    epilogue_operand_count: usize,
    gather_count: usize,
    entry: &str,
    element_type: &str,
    include_threadgroup_width: bool,
) {
    let operand_count = quantized.len();
    source.push_str(&format!("kernel void {entry}(\n"));
    for (index, &codec) in quantized.iter().enumerate() {
        // a packed operand's buffer is BYTES, not elements — the shader
        // turns an element offset into a super-block plus a position inside
        // it at the read (`operand_read`), so the binding has to be typed
        // for what is actually in the buffer. `Float16` is the one codec
        // whose buffer is neither raw bytes nor the kernel's own
        // `element_type`: its bytes ARE valid `half` elements already, so it
        // binds as `half` directly rather than `uchar` -- see
        // `FLOAT16_BLOCK_BYTES`'s own doc.
        let binding_type = match codec {
            None => element_type,
            Some(PackedCodec::Float16) => "half",
            Some(_) => "uchar",
        };
        source.push_str(&format!(
            "    device const {binding_type}* in{index} [[buffer({index})]],\n"
        ));
    }
    // A [`BoundOpKind::Reduce::epilogue_operands`] entry is always a plain,
    // un-gathered, un-packed `element_type` buffer -- the same restriction
    // `cpu::apply_reduce_epilogue` already enforces (`operand_read` there has
    // no codec branch) -- so each gets one flat device buffer, positioned
    // right after the fold's own operands and before anything gather adds.
    for index in 0..epilogue_operand_count {
        source.push_str(&format!(
            "    device const {element_type}* epi{index} [[buffer({})]],\n",
            operand_count + index
        ));
    }
    let base = operand_count + epilogue_operand_count;
    for slot in 0..gather_count {
        // a gather's fetched index is always carried as an exact-integer
        // `float`, independent of the op's own element type — see this
        // crate's doc for `gather_idx` and `cpu::reject_non_float32`'s own
        // note on indices being the one deliberate non-dtype exception.
        source.push_str(&format!(
            "    device const float* gather_idx{slot} [[buffer({})]],\n",
            base + slot
        ));
    }
    source.push_str(&format!(
        "    device {element_type}* out [[buffer({})]],\n",
        base + gather_count
    ));
    source.push_str(&format!(
        "    constant Uniforms& u [[buffer({})]],\n",
        base + gather_count + 1
    ));
    if gather_count > 0 {
        source.push_str(&format!(
            "    device atomic_uint* fault [[buffer({})]],\n",
            base + gather_count + 2
        ));
    }
    source.push_str("    uint gid [[thread_position_in_grid]]");
    if include_threadgroup_width {
        // the actual per-dispatch threadgroup width -- `crate::metal::dispatch`
        // sets this from `GridSpec::threadgroup_width`, which
        // `tiled_gemm_threadgroup_width` computes FRESH per concrete dispatch
        // (unlike `Kernel::source`, cached and shared across every dispatch
        // with the same structural `kernel_cache_key`). Reading it back here
        // is what lets one compiled kernel body serve both a starved shape
        // (split > 1) and a saturated one (split == 1) without two kernel
        // bodies existing per structural shape.
        source.push_str(",\n    uint tptg [[threads_per_threadgroup]]");
    }
    source.push_str(")\n{\n");
}

/// Declares the `Uniforms` fields a gather needs — `index_base`/`index_strides`
/// (per-gather addressing into its `indices` buffer, over the *same* rank as
/// every other operand), `element_stride` (the operand's own stride along
/// its gathered dim), and `extent` (the gathered dim's size, for the clamp
/// [`push_gather_fetch`] emits). Declared only when `gather_count > 0`, so a
/// gather-free kernel's `Uniforms` struct is byte-for-byte what it was
/// before gather existed.
fn push_gather_uniform_fields(source: &mut String, gather_count: usize, rank_len: usize) {
    if gather_count == 0 {
        return;
    }
    source.push_str(&format!("    long gather_index_base[{gather_count}];\n"));
    source.push_str(&format!(
        "    long gather_index_strides[{gather_count}][{rank_len}];\n"
    ));
    source.push_str(&format!(
        "    long gather_element_stride[{gather_count}];\n"
    ));
    source.push_str(&format!("    long gather_extent[{gather_count}];\n"));
}

/// Emits the out-of-range check for one just-fetched, not-yet-clamped
/// `fetched{operand_index}`: when it falls outside
/// `[0, u.gather_extent[gather_slot])`, records it (plus one, so a slot
/// left at zero unambiguously means "no fault") into that gathered
/// operand's slot of the `fault` buffer via `atomic_fetch_max`. A negative
/// fetched index is reported as `0` (mapped through `max(fetched, 0)`
/// before the `+1`) rather than reinterpreting a negative `long` as a huge
/// `uint` — this crate's sad-path tests only exercise the far-more-common
/// too-large case, so that is the one case whose reported value round-trips
/// exactly. `atomic_fetch_max` (not a plain write) is what makes this safe
/// under concurrent threads without a CAS loop: whichever value "wins" the
/// max is still a genuine fault, and the driver only needs to know that one
/// occurred and at what value to build a `TensorError`.
fn push_gather_fault_check(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    indent: &str,
) {
    source.push_str(&format!(
        "{indent}if (fetched{operand_index} < 0 || fetched{operand_index} >= u.gather_extent[{gather_slot}]) {{\n"
    ));
    source.push_str(&format!(
        "{indent}    atomic_fetch_max_explicit(&fault[{gather_slot}], (uint)max(fetched{operand_index}, (long)0) + 1u, memory_order_relaxed);\n"
    ));
    source.push_str(&format!("{indent}}}\n"));
}

/// Emits the fetch for one gathered operand: reads its index from
/// `gather_idx{slot}` at the same coordinate `coord_var` addresses every
/// other buffer with, checks it against `[0, extent)` — recording a fault
/// (see [`push_gather_fault_check`]) since a GPU kernel cannot return a
/// `Result` the way `cpu::evaluate` does — then clamps it into `[0, extent)`
/// regardless, so the read this value drives always lands in bounds even
/// when a fault was just recorded, and adds the resulting offset into
/// `offset_var`.
fn push_gather_fetch(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    rank: usize,
    coord_var: &str,
    offset_var: &str,
) {
    source.push_str(&format!(
        "    long gather_off{operand_index} = u.gather_index_base[{gather_slot}];\n"
    ));
    for dim in 0..rank {
        source.push_str(&format!(
            "    gather_off{operand_index} += {coord_var}[{dim}] * u.gather_index_strides[{gather_slot}][{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "    long fetched{operand_index} = (long)gather_idx{gather_slot}[gather_off{operand_index}];\n"
    ));
    push_gather_fault_check(source, operand_index, gather_slot, "    ");
    source.push_str(&format!(
        "    fetched{operand_index} = max((long)0, min(fetched{operand_index}, u.gather_extent[{gather_slot}] - 1));\n"
    ));
    source.push_str(&format!(
        "    {offset_var} += fetched{operand_index} * u.gather_element_stride[{gather_slot}];\n"
    ));
}

fn push_cooperative_gather_fetch(
    source: &mut String,
    operand_index: usize,
    gather_slot: usize,
    rank: usize,
    coord_var: &str,
    offset_var: &str,
) {
    source.push_str(&format!(
        "    long gather_off{operand_index} = u.gather_index_base[{gather_slot}];\n"
    ));
    for dim in 0..rank {
        source.push_str(&format!(
            "    gather_off{operand_index} += {coord_var}[{dim}] * u.gather_index_strides[{gather_slot}][{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "    long fetched{operand_index} = (long)gather_idx{gather_slot}[gather_off{operand_index}];\n"
    ));
    source.push_str("    if (lane == 0u) {\n");
    push_gather_fault_check(source, operand_index, gather_slot, "    ");
    source.push_str(&format!(
        "    }}\n    fetched{operand_index} = max((long)0, min(fetched{operand_index}, u.gather_extent[{gather_slot}] - 1));\n"
    ));
    source.push_str(&format!(
        "    fetched{operand_index} = (long)simd_broadcast_first((uint)fetched{operand_index});\n    {offset_var} += fetched{operand_index} * u.gather_element_stride[{gather_slot}];\n"
    ));
}

/// `metal_stdlib` has no `erf` in any namespace — verified against the real
/// toolchain (`xcrun -sdk macosx metal -c`, `no member named 'erf'`, tried
/// bare, `metal::`, and `metal::precise::`), not assumed from the ONNX
/// survey that first named it. This is the same Abramowitz & Stegun 7.1.26
/// approximation [`crate cpu::erf_f32`](../../proxima_tensor/src/cpu.rs) uses
/// on the CPU path, so a kernel and the CPU interpreter it is checked
/// against agree on more than "close enough" — they run the identical
/// formula.
const PROXIMA_ERF_FN: &str = "\
inline float proxima_erf(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float magnitude = fabs(x);
    float t = 1.0f / fma(0.3275911f, magnitude, 1.0f);
    float poly = t * fma(fma(fma(fma(1.061405429f, t, -1.453152027f), t, 1.421413741f), t, -0.284496736f), t, 0.254829592f);
    return sign * fma(poly, -exp(-magnitude * magnitude), 1.0f);
}
";

fn preamble(source: &mut String) {
    source.push_str("#include <metal_stdlib>\n");
    source.push_str("using namespace metal;\n\n");
    source.push_str(PROXIMA_ERF_FN);
    source.push('\n');
    source.push_str(Q2K_UNPACK_MSL);
    source.push('\n');
    // emitted unconditionally, the same way `PROXIMA_ERF_FN` is: a
    // `static inline` the kernel never calls costs nothing in the compiled
    // AIR, and making it conditional would mean threading "does this kernel
    // read a packed operand" into the preamble for no gain.
    source.push_str(Q3K_UNPACK_MSL);
    source.push('\n');
    // structural, not feature-gated: `Q3K_PAIR_DOT_MSL` is always a real
    // Rust symbol (unlike `Q5K_PAIR_DOT_MSL`'s `#[cfg]`), so this splice is
    // unconditional the same way `Q4K_UNPACK_MSL`'s own splice below is --
    // `push_packed_row_blocked_body`'s `plain_product` check decides
    // whether the KERNEL calls it, not whether it compiles into the AIR.
    source.push_str(Q3K_PAIR_DOT_MSL);
    source.push('\n');
    source.push_str(Q4K_UNPACK_MSL);
    source.push('\n');
    // feature-gated, unlike the constants around it: `Q4K_MASK_FMA_MSL`
    // does not exist as a Rust symbol at all without `metal-q4k-mask-fma`
    // (see its own doc), so this splice is the one place in `preamble`
    // that must itself be `#[cfg]`'d rather than relying on "unused static
    // inline costs nothing" the way the codec preambles around it do.
    #[cfg(feature = "metal-q4k-mask-fma")]
    {
        source.push_str(Q4K_MASK_FMA_MSL);
        source.push('\n');
    }
    source.push_str(Q5K_UNPACK_MSL);
    source.push('\n');
    // unconditional, same posture as `Q4K_UNPACK_MSL` above: an unused
    // `static inline` the kernel never calls costs nothing in the compiled
    // AIR, and the selector is now the codec's own layout
    // ([`PackedCodec::supports_pair_dot`]), not a cargo feature.
    source.push_str(Q5K_PAIR_DOT_MSL);
    source.push('\n');
    source.push_str(Q6K_UNPACK_MSL);
    source.push('\n');
    source.push_str(Q6K_PAIR_DOT_MSL);
    source.push('\n');
    // The mixed selector calls the Q2_K, Q4_K, and Q6_K element decoders,
    // so it follows all three declarations in the generated translation unit.
    source.push_str(MIXED_EXPERT_READ_MSL);
    source.push('\n');
    source.push_str(UNIFORM_EXPERT_READ_MSL);
    source.push('\n');
    source.push_str(Q8_0_UNPACK_MSL);
    source.push('\n');
    source.push_str(Q4_0_UNPACK_MSL);
    source.push('\n');
    source.push_str(BF16_UNPACK_MSL);
    source.push('\n');
}

/// How operand `index` is READ, given the element-offset expression the
/// caller already computed. A float operand is a direct index. A `Q4_K`
/// operand's buffer is PACKED BYTES, so that same element offset splits into
/// a super-block and a position inside it: element `n` lives in super-block
/// `n / 256` at position `n % 256`, and that super-block starts at byte
/// `(n / 256) * 144`. The uniforms stay in elements either way — only the
/// read shape changes, which is the entire point of unpacking at the read
/// instead of materializing a dequantized tensor first.
fn operand_read(index: usize, offset: &str, codec: Option<PackedCodec>) -> String {
    match codec {
        None => format!("in{index}[{offset}]"),
        Some(PackedCodec::Q2K) => format!(
            "q2k_element(in{index} + ({offset} / {Q2K_BLOCK_ELEMENTS}) * {Q2K_BLOCK_BYTES}, (uint)({offset} % {Q2K_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q3K) => format!(
            "q3k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q3K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q4K) => format!(
            "q4k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q4K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q5K) => format!(
            "q5k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q5K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        Some(PackedCodec::Q6K) => format!(
            "q6k_element(in{index} + ({offset} / {Q4K_BLOCK_ELEMENTS}) * {Q6K_BLOCK_BYTES}, (uint)({offset} % {Q4K_BLOCK_ELEMENTS}))"
        ),
        // `Q8_0`'s block is 32 elements, not 256 -- its own
        // [`Q8_0_BLOCK_ELEMENTS`], never [`Q4K_BLOCK_ELEMENTS`].
        Some(PackedCodec::Q8_0) => format!(
            "q8_0_element(in{index} + ({offset} / {Q8_0_BLOCK_ELEMENTS}) * {Q8_0_BLOCK_BYTES}, (uint)({offset} % {Q8_0_BLOCK_ELEMENTS}))"
        ),
        // `Q4_0`'s block is 32 elements, not 256 -- its own
        // [`Q4_0_BLOCK_ELEMENTS`], never [`Q4K_BLOCK_ELEMENTS`].
        Some(PackedCodec::Q4_0) => format!(
            "q4_0_element(in{index} + ({offset} / {Q4_0_BLOCK_ELEMENTS}) * {Q4_0_BLOCK_BYTES}, (uint)({offset} % {Q4_0_BLOCK_ELEMENTS}))"
        ),
        // `Float16`'s buffer already binds as `device const half*`
        // (`kernel_signature`'s own match), so reading it is a plain index
        // exactly like a `None` operand -- MSL implicitly promotes the
        // resulting `half` to `float` wherever the body assigns it into a
        // `float` scratch slot, no cast needed.
        Some(PackedCodec::Float16) => format!("in{index}[{offset}]"),
        // `BFloat16`'s block is 1 element, 2 bytes -- its own
        // [`BFLOAT16_BLOCK_ELEMENTS`]/[`BFLOAT16_BLOCK_BYTES`], never
        // [`Q4K_BLOCK_ELEMENTS`].
        Some(PackedCodec::BFloat16) => format!(
            "bf16_element(in{index} + ({offset} / {BFLOAT16_BLOCK_ELEMENTS}) * {BFLOAT16_BLOCK_BYTES}, (uint)({offset} % {BFLOAT16_BLOCK_ELEMENTS}))"
        ),
    }
}

/// [`BoundOpKind::Iota`]'s kernel: no operand buffers, no gather, no body —
/// the output value at each position is the thread's own grid coordinate,
/// which every kernel already computes as `gid`, so there is nothing to
/// derive beyond casting it to the node's element type. Reuses
/// [`kernel_signature`] with `operand_count = 0`, `gather_count = 0` so the
/// buffer-index arithmetic (`out` at 0, `Uniforms` at 1) stays the one place
/// that owns it rather than being re-derived here.
fn render_iota(resolved: &BoundOp, entry: &str) -> Result<String, EmitError> {
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long total_elements;\n");
    source.push_str("};\n\n");

    kernel_signature(&mut source, &[], 0, 0, entry, element_type, false);
    source.push_str("    if ((long)gid >= u.total_elements) { return; }\n");
    source.push_str(&format!("    out[gid] = ({element_type})gid;\n"));
    source.push_str("}\n");
    Ok(source)
}

/// [`BoundOpKind::Constant`]'s kernel, the same shape as [`render_iota`]'s
/// with the position swapped for the literal. The literal is baked into the
/// source rather than passed as a uniform so the `Uniforms` struct stays
/// byte-identical to `render_iota`'s and both share
/// [`crate::metal`]'s `pack_leaf_uniforms`; `kernel_entry` folds the value's
/// bits into the entry name to keep the kernel cache correct.
fn render_constant(resolved: &BoundOp, entry: &str, value: f32) -> Result<String, EmitError> {
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long total_elements;\n");
    source.push_str("};\n\n");

    kernel_signature(&mut source, &[], 0, 0, entry, element_type, false);
    source.push_str("    if ((long)gid >= u.total_elements) { return; }\n");
    source.push_str(&format!(
        "    out[gid] = ({element_type}){};\n",
        msl_literal(value)
    ));
    source.push_str("}\n");
    Ok(source)
}

/// One `f32` as MSL source text. `Debug`'s shortest round-trip decimal is
/// what MSL's own float grammar accepts, except for the values it has no
/// decimal spelling for.
fn msl_literal(value: f32) -> String {
    if value.is_nan() {
        return "NAN".to_string();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-INFINITY".to_string()
        } else {
            "INFINITY".to_string()
        };
    }
    format!("{value:?}")
}

/// Number of simdgroups `render_cached_attention` splits one
/// `(query_row, kv_head, group)` triple's key range across -- llama.cpp's
/// `kernel_flash_attn_ext_vec` (ggml-metal.metal:4016-4017) partitions the
/// context across `nsg` simdgroups with a stride of `C*nsg`; this ports the
/// same stride partition to a `C=1` per-key loop. This is CHUNK SIZING, a
/// compile-time partitioning decision, and it stays bound to the compiled
/// range extent `cached_key_rows + new_key_rows` (the same static fields the
/// kernel body bakes as `constexpr`) -- a build-time-configurable but
/// bind-time-deterministic function of shape alone, never a runtime read.
/// This is a SEPARATE decision from the per-simdgroup stride LOOP's own
/// upper bound (`render_cached_attention`'s `last_key`), which as of the
/// merged-KV fix reads the live runtime prefix when one is carried (the
/// ninth operand) rather than always walking the full compiled extent.
/// Neither value is "the context length": one sizes the partition, the
/// other bounds each partition's own walk. See `omega-runtime.toml`'s
/// `[attention_context_chunks]` for the two knobs.
///
/// A chunk count above 1 folds partial online-softmax state across
/// simdgroups (`render_cached_attention`'s cross-simdgroup merge) -- a
/// reassociation of the reduce, [`NumericRewrite::ContextChunkMerge`].
/// `policy` gates it: without `reassociation` granted, [`admit`] rejects the
/// rewrite and this falls back to `1` (the single-pass
/// kernel `render_cached_attention` already renders for that case) rather
/// than failing the whole plan -- the same "reject/fallback at bind time"
/// shape `fuse_cached_attention: false` gives wgpu/cuda
/// (`proxima_tensor::bind::bind_with_fusion`'s own doc).
pub(crate) fn context_chunks_for(context_length: u64, policy: NumericPolicy) -> u64 {
    if admit(policy, NumericRewrite::ContextChunkMerge).is_err() {
        return 1;
    }
    context_length
        .div_ceil(crate::sized::ATTENTION_CONTEXT_KEYS_PER_CHUNK)
        .clamp(1, crate::sized::ATTENTION_CONTEXT_CHUNK_CAP)
}

/// [`render_cached_attention`]'s single-range dynamic path's in-block
/// Q·K/softmax/V staging width -- llama.cpp's `kernel_flash_attn_ext_vec`
/// keys-per-block constant `C` (ggml-metal.metal:4016-4017, `ic0 += C*nsg`),
/// ported onto our per-simdgroup key walk. Grouping `width` sequential
/// online-softmax updates into one block-level `simd_max`/`simd_sum` combine
/// reorders the fold as a tree instead of a left fold --
/// [`NumericRewrite::TreeReduce`], the same permission bit
/// [`context_chunks_for`]'s `NumericRewrite::ContextChunkMerge` already
/// needs. Under `bit_exact` (`admit` rejects the rewrite), this returns `1`,
/// which renders EXACTLY today's per-key loop, byte for byte -- the
/// sequential path IS the bit-exact lowering, by construction, the same way
/// `context_chunks_for` keeps chunk-count `1` under `bit_exact` today.
pub(crate) fn block_width_for(policy: NumericPolicy) -> u64 {
    if admit(policy, NumericRewrite::TreeReduce).is_err() {
        1
    } else {
        crate::sized::ATTENTION_BLOCK_WIDTH
    }
}

/// Number of THREADGROUPS `render_cached_attention`'s single-range dynamic
/// path splits one `(query_row, kv_head)` pair's key range across --
/// [`context_chunks_for`] one level down (simdgroups sharing ONE
/// threadgroup); this is the same partition one hardware level up
/// (threadgroups sharing one dispatch), llama.cpp's `nwg`
/// (`ggml-metal-ops.cpp:3457-3466`) ported as a compiled-capacity-derived
/// count rather than llama's fixed 32, because this backend's threadgroup is
/// heavier (cooperative `query_groups`-way K/V sharing across its `cap`
/// simdgroups) than llama's per-Q-head one.
///
/// `context_length` is the SAME compiled-capacity input
/// [`context_chunks_for`] already takes (`cached_key_rows + new_key_rows`),
/// not a per-call live value -- so, exactly like `chunks`/`cap`, one
/// compiled kernel serves every `kv-capacity-bucket` crossing: the dispatch
/// always issues the compiled MAXIMUM ([`crate::sized::ATTENTION_SPLIT_MAX`])
/// worth of threadgroups, and the live `splits` value this function returns
/// travels as a runtime `Uniforms` field so an idle split (`split >= splits`)
/// contributes the merge's own identity partial rather than being sized out
/// of the dispatch -- the same "idle unit, identity partial" shape `chunks`/
/// `cap` already uses.
///
/// Splitting the key range across dispatch boundaries reassociates the
/// online-softmax combine one hardware level above [`ContextChunkMerge`] --
/// [`NumericRewrite::ContextSplitMerge`]. Under `bit_exact` (`admit` rejects
/// the rewrite), this returns `1`, rendering exactly today's single-dispatch
/// `render_cached_attention` body, byte for byte -- the single-dispatch path
/// IS the bit-exact lowering, the same way `context_chunks_for`/
/// `block_width_for` fall back to `1` under `bit_exact`.
///
/// ROW 383: the divisor has a KNEE rather than being one constant. More
/// splits helps at both ends -- the tiny decode window (normally a single,
/// occupancy-starved threadgroup) and huge contexts (enough keys per split
/// to amortize the merge dispatch's fixed cost) -- but HURTS in the middle,
/// where occupancy is already adequate at a handful of splits and a finer
/// divisor only adds fixed per-split overhead with nothing to amortize it
/// against (measured: 512 keys at 8/16/32 splits all land 170-177us, no
/// better than each other, vs 89.5us at 4 splits). So `context_length`
/// below `ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE` uses the small divisor
/// (`ATTENTION_SPLIT_KEYS_PER_SPLIT`, ROW 381's decode-window fix); at or
/// above it, the ORIGINAL pre-ROW-381 divisor
/// (`ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`) applies, unchanged from what
/// already won at both 512 keys (splits=4) and 4096 keys (splits=32, the
/// same value the small divisor also reaches once clamped to `max`).
pub(crate) fn splits_for(context_length: u64, policy: NumericPolicy) -> u64 {
    if admit(policy, NumericRewrite::ContextSplitMerge).is_err() {
        return 1;
    }
    let keys_per_split = if context_length < crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE {
        crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT
    } else {
        crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE
    };
    context_length
        .div_ceil(keys_per_split)
        .clamp(1, crate::sized::ATTENTION_SPLIT_MAX)
}

/// Whether `render_cached_attention`'s single-range dynamic path renders as
/// TWO Metal dispatches (this function's `true`) or the single, unchanged
/// dispatch (`false`) -- delegates to [`splits_for`]'s own `context_length`
/// AND policy gate, factored out so [`render_cached_attention`]'s own
/// final-store branch, [`split_bindings_with_scratch`]'s caller, and
/// [`emit_cached_attention_merge`] all key off one boolean rather than three
/// independent `splits_for` calls that could drift. Taking `context_length`
/// (not policy alone) matters: a policy that admits `ContextSplitMerge` but
/// whose live capacity clamps `splits_for` to `1` (ROW 376's own 40-key
/// scoreboard window, well under `omega-runtime.toml`'s `keys_per_split`)
/// must still render the byte-identical single dispatch -- admitting the
/// rewrite is necessary but not sufficient for a SECOND dispatch to be
/// worth its own fixed overhead (design risk 2). ROW 385: a merge additionally
/// requires `context_length >= ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE` --
/// below that knee [`cached_attention_per_query_head_grid`] takes over the
/// short-context case with one threadgroup per query head instead, and
/// `splits_for`'s own small-divisor branch would otherwise still report
/// `> 1` there, which [`crate::metal::pack_cached_attention_uniforms`] would
/// then slice the live key range by with no second dispatch to merge the
/// slices back -- an out-of-bounds-shaped undercount, not merely a missed
/// optimization.
///
/// Takes `kind` (not a bare `context_length: u64`) because `ContextSplitMerge`
/// is implemented ONLY inside [`render_cached_attention`]'s `single_range_
/// dynamic` branch (`cached_key_rows == 0`, the nine-operand fused kind):
/// that branch alone writes the `(max, sum, weighted)` scratch layout
/// [`render_cached_attention_merge`] reads back. The `two_range_cached_bound`
/// branch (nine operands, `cached_key_rows != 0`, `render_cached_attention`'s
/// own two-range `context_chunks <= 1` / `> 1` arms) ALWAYS writes the
/// complete, correctly-normalized attention output straight to `out[]` in one
/// pass, regardless of `context_length` -- it has no split/merge protocol at
/// all. A bare `context_length: u64` signature let three of this function's
/// four call sites (`emit`, `kernel_dispatch_shape`,
/// `emit_cached_attention_merge`) answer `true` for the two-range kind once
/// `cached_key_rows + new_key_rows >= ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`
/// (measured: `cached_len=200`/`700`, `NumericPolicy::llama_relaxed()`,
/// `qwen3_gqa_qk_norm_forward_fixture` -- max_diff 12.6 / 10.4 against the CPU
/// oracle) -- `emit`/`kernel_dispatch_shape` then swapped the op's bindings to
/// `split_bindings_with_scratch` (treating `out` as a raw per-split scratch
/// table it was never written as) and `emit_cached_attention_merge` dispatched
/// a companion merge kernel that reinterpreted the already-correct, already-
/// normalized `out[]` bytes as `(max, sum, weighted[head_dim])` triples and
/// overwrote them with garbage. `single_range_dynamic` here reproduces the
/// SAME discriminator [`render_cached_attention`] (`:3414`) and `grid_threads`
/// (`:2477`) already compute correctly, so all four call sites now agree by
/// construction instead of by convention.
pub(crate) fn cached_attention_merge_needed(kind: &BoundOpKind, policy: NumericPolicy) -> bool {
    let BoundOpKind::CachedAttention {
        operands,
        cached_key_rows,
        new_key_rows,
        ..
    } = kind
    else {
        return false;
    };
    let single_range_dynamic = operands.len() == 9 && *cached_key_rows == 0;
    if !single_range_dynamic {
        return false;
    }
    let context_length = cached_key_rows + new_key_rows;
    splits_for(context_length, policy) > 1
        && context_length >= crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE
}

/// Whether `render_cached_attention`'s single-range dynamic path dispatches
/// one threadgroup per `(query_row, kv_head, group)` triple -- narrowing
/// [`tiled_gemm_threadgroup_width`]'s `CachedAttention` width from
/// `query_groups * chunks * SIMD_WIDTH` to `chunks * SIMD_WIDTH` and turning
/// `query_groups` into an extra THREADGROUP-count factor instead -- rather
/// than today's one threadgroup per `(query_row, kv_head)` pair shared by
/// every query head in the group. Below
/// [`crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE`] keys the
/// occupancy problem is too few threadgroups, not too little per-threadgroup
/// parallelism (ROW 383's own knee), so spending `query_groups` as more,
/// narrower threadgroups instead of more warps inside one threadgroup gives
/// the GPU more independent units of work to schedule at a window where it
/// is otherwise starved. `dynamic_cached_len` gates this to the single-range
/// fused (nine-operand) path alone -- the compiled fixed-shape path already
/// sizes its own dispatch from a KNOWN `context_length`, so it never carries
/// this trade. Not policy-gated: unlike [`cached_attention_merge_needed`],
/// this reshapes an EXISTING dispatch rather than reassociating the online-
/// softmax fold, so `bit_exact` renders it exactly like every other policy.
pub(crate) fn cached_attention_per_query_head_grid(
    dynamic_cached_len: bool,
    context_length: u64,
) -> bool {
    dynamic_cached_len && context_length < crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_AT_SCALE
}

/// `BoundOpKind::CachedAttention`'s Metal kernel: online (running max/sum,
/// register-resident weighted-value accumulator) softmax attention over a
/// cached range plus a new range, one 32-lane simdgroup per
/// `(query_row, kv_head, group[, chunk])` tuple. Each simdgroup reads ONLY
/// its own assigned keys straight from device memory into per-lane
/// registers (`head_dim / 2` even/odd K pairs, `head_dim` V elements,
/// lane-strided), reduces the QK dot with `simd_sum`, and folds the running
/// max/sum/weighted-value purely in registers — the same per-simdgroup
/// register-resident shape as llama.cpp's `kernel_flash_attn_ext_vec`
/// (ggml-metal.metal:4014-4110), which loads K/V per simdgroup from device
/// memory and reduces with `simd_shuffle_down`/`simd_max` rather than
/// staging through `threadgroup` memory. There is no `threadgroup_barrier`
/// inside the per-key loop: nothing is shared across simdgroups until the
/// keys are exhausted, so there is nothing to synchronize on every
/// iteration. The only `threadgroup_barrier` calls left are around the
/// cross-simdgroup merge below, when `context_chunks > 1` splits one
/// `(query_row, kv_head, group)` triple's key range across simdgroups that
/// must combine their partial online-softmax state afterward.
fn render_cached_attention(
    resolved: &BoundOp,
    entry: &str,
    numeric_policy: NumericPolicy,
) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        query_rows,
        cached_key_rows,
        new_key_rows,
        kv_heads,
        query_groups,
        head_dim,
        scale,
        cached_lower_inclusive,
        new_upper_inclusive,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "cached_attention",
            found: resolved.kind.name(),
        });
    };
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let cached_lower = if *cached_lower_inclusive == i64::MIN {
        "-9223372036854775807L".to_string()
    } else {
        format!("{cached_lower_inclusive}L")
    };
    // The ninth operand slot carries two DIFFERENT runtime scalars,
    // discriminated by `cached_key_rows` (`BoundOpKind::CachedAttention`'s
    // own doc): `cached_key_rows == 0` is the single-range fusion's true
    // `cached_len`, read here as `new_upper`; `cached_key_rows != 0` is the
    // two-range fusion's own live cached-row count, read further down as a
    // runtime `cached_key_rows` (`row_count_decl`) instead -- that shape
    // needs no dynamic `new_upper` at all, since its "new" range is never
    // bucketed. `entry_name`'s own "dyn"/"cb" markers are what let one
    // compiled kernel serve every live value on each path.
    let has_ninth_operand = resolved.operands().len() == 9;
    let single_range_dynamic = has_ninth_operand && *cached_key_rows == 0;
    let two_range_cached_bound = has_ninth_operand && *cached_key_rows != 0;
    let (cached_len_param, new_upper_decl) = if single_range_dynamic {
        (
            format!(", device const {element_type}* in8 [[buffer(8)]]"),
            "long new_upper = (long)in8[0];".to_string(),
        )
    } else if two_range_cached_bound {
        (
            format!(", device const {element_type}* in8 [[buffer(8)]]"),
            format!("constexpr long new_upper = {new_upper_inclusive}L;"),
        )
    } else {
        (
            String::new(),
            format!("constexpr long new_upper = {new_upper_inclusive}L;"),
        )
    };
    // The split kernel-plus-merge shape (redesign §4c) reads WHICH
    // threadgroup a simdgroup belongs to directly from Metal's own
    // per-dispatch coordinate rather than re-deriving it from `gid` --
    // `gid / threadgroup_width` would give the same value today, but that
    // equality holds only because `threadgroup_width` never itself depends
    // on `splits` (see `grid_threads`'s own doc); a hardware-native readback
    // does not carry that assumption. Only the single-range dynamic path
    // dispatches more than one threadgroup per `(query_row, kv_head)` pair,
    // so only that path declares the parameter.
    let tgid_param = if single_range_dynamic {
        ", uint tgid [[threadgroup_position_in_grid]]"
    } else {
        ""
    };
    let (out_buffer_index, uniforms_buffer_index) =
        if has_ninth_operand { (9, 10) } else { (8, 9) };
    // `cached_key_rows`/`new_key_rows` are runtime `Uniforms` fields on the
    // single-range fused path (`dynamic_cached_len`, the ninth-operand
    // form). Redesign §5 option 2: `context_chunks` joins them as a fourth
    // runtime field on that SAME path, so `entry_name` (below) can drop the
    // `_c{}`/`_n{}` row-count tokens entirely -- the compiled kernel text no
    // longer depends on the compiled `kv-capacity-bucket` extent at all, and
    // one pipeline serves every bucket, matching llama's own `ne11`-as-
    // runtime-field property (`ggml-metal.m:4790-4813`).
    // `splits` (redesign §4c, [`NumericRewrite::ContextSplitMerge`]) rides
    // the SAME "runtime `Uniforms` field, never a `constexpr`" precedent
    // `context_chunks` just set two lines below -- packed by
    // `crate::metal::pack_cached_attention_uniforms` from the SAME
    // compiled-capacity input `context_chunks_for` already takes, via
    // `splits_for`. Declared here so the packed byte layout the driver
    // writes (`pack_cached_attention_uniforms`) and the struct this text
    // declares never drift out of sync. Not yet read anywhere in this
    // kernel body -- `render_cached_attention_merge`, the split/scratch
    // write, and the encoder's second dispatch are the follow-up that
    // consumes it (see this crate's `attention-kernel-design.md` §4c, risk
    // 1: the scratch-buffer plumbing this needs is unverified against
    // `metal.rs`'s current allocation call sites and is out of scope here).
    let uniforms_struct = if single_range_dynamic {
        "struct Uniforms { long total_elements; long cached_key_rows; long new_key_rows; long context_chunks; long splits; };\n\n"
    } else {
        "struct Uniforms { long total_elements; };\n\n"
    };
    // `two_range_cached_bound` substitutes a runtime-read `cached_key_rows`
    // for the compiled `constexpr` everywhere downstream that name already
    // appears (`bool cached = key < cached_key_rows`, the `new_index`
    // offset subtraction, `last_key_decl`'s sum below) -- one declaration
    // site changes what every existing use of the variable name means,
    // rather than a new gate at each of those uses (`BoundOpKind::
    // CachedAttention`'s own doc on why this is the fewer-renderer-lines
    // shape). `new_key_rows` is never bucketed on this path, so it stays a
    // compiled constant.
    let row_count_decl = if single_range_dynamic {
        "long cached_key_rows = u.cached_key_rows; long new_key_rows = u.new_key_rows;"
    } else if two_range_cached_bound {
        "long cached_key_rows = (long)in8[0]; constexpr long new_key_rows = {new_key_rows};"
    } else {
        "constexpr long cached_key_rows = {cached_key_rows}; constexpr long new_key_rows = {new_key_rows};"
    };
    let row_count_decl = row_count_decl
        .replace("{cached_key_rows}", &cached_key_rows.to_string())
        .replace("{new_key_rows}", &new_key_rows.to_string());
    let mut source = String::new();
    preamble(&mut source);
    source.push_str(uniforms_struct);
    source.push_str(&format!(
        "kernel void {entry}(device const {element_type}* in0 [[buffer(0)]], device const {element_type}* in1 [[buffer(1)]], device const {element_type}* in2 [[buffer(2)]], device const {element_type}* in3 [[buffer(3)]], device const {element_type}* in4 [[buffer(4)]], device const {element_type}* in5 [[buffer(5)]], device const {element_type}* in6 [[buffer(6)]], device const {element_type}* in7 [[buffer(7)]]{cached_len_param}, device {element_type}* out [[buffer({out_buffer_index})]], constant Uniforms& u [[buffer({uniforms_buffer_index})]], uint gid [[thread_position_in_grid]]{tgid_param}) {{\n"
    ));
    source.push_str("    if ((long)gid >= u.total_elements * 32L) { return; }\n");
    source.push_str(&format!(
        "    {row_count_decl} constexpr long kv_heads = {kv_heads}; constexpr long query_groups = {query_groups}; constexpr long head_dim = {head_dim}; constexpr float scale = {}; constexpr long cached_lower = {cached_lower}; {new_upper_decl}\n",
        msl_literal(*scale),
    ));
    // One threadgroup per (query_row, kv_head) pair -- `tiled_gemm_
    // threadgroup_width`'s `CachedAttention` arm dispatches exactly
    // `query_groups * SIMD_WIDTH` threads per threadgroup, and `dispatchThreads_
    // threadsPerThreadgroup`'s linear grouping (`thread_position_in_grid =
    // threadgroup_position_in_grid * width + thread_position_in_threadgroup`)
    // lands every one of `query_groups` simdgroups (one per query head sharing
    // this kv_head) in the SAME threadgroup, because `vector_index`'s own
    // encoding already cycles `group` fastest, then `kv_head`, then
    // `query_row` -- see this function's doc. `tid`/`group_width` below are
    // therefore derivable from the existing `group`/`lane` split without a
    // new `[[thread_position_in_threadgroup]]` kernel parameter.
    // Two distinct decisions, both fed by the bind-time `cached_key_rows` /
    // `new_key_rows` fields but not the same computation: CHUNK SIZING
    // (`context_chunks`, immediately below -- how many simdgroups share this
    // key range) versus the per-simdgroup stride LOOP's own upper bound
    // (`last_key`, below that). A merged-KV bind that declares its empty
    // cached half as `cached_key_rows: 0` (rather than duplicating the
    // capacity into it) changes what this line feeds `context_chunks_for`,
    // which can change the chunk count -- a repartitioning, not merely a
    // loop-bound change.
    let context_chunks = context_chunks_for(*cached_key_rows + *new_key_rows, numeric_policy);
    // Redesign §5 option 2: on the single-range fused path, the compiled
    // MAXIMUM simdgroup count (`cap`) sizes both the dispatch grid and the
    // threadgroup-memory merge arrays -- never the bind's own compiled
    // `context_chunks_for(...)` result, which varies with the compiled
    // `kv-capacity-bucket` extent and is exactly the quantity `entry_name`
    // must stop depending on. A live `chunks <= cap` still travels as a
    // runtime `Uniforms` field (packed by `pack_cached_attention_uniforms`),
    // and simdgroups `chunk >= chunks` contribute the merge's own identity
    // (`maximum = -INFINITY`, `sum = 0.0`) rather than looping -- the same
    // "idle simdgroup, identity partial" shape llama's own dispatch-time
    // `nsg` uses against a compiled maximum (`ggml-metal.m:4887-4913`).
    let cap = crate::sized::ATTENTION_CONTEXT_CHUNK_CAP;
    // The block-staged body's `float4` K/Q loads (below) reinterpret each
    // real/imaginary plane offset as `device const {element_type}4*` --
    // legal only when every `qbase`/`kbase` offset this kernel computes is a
    // multiple of 4 floats. Both offsets are integer multiples of
    // `head_dim / 2`, so `head_dim / 2` itself being a multiple of 4
    // (equivalently, `head_dim` a multiple of 8) is sufficient regardless of
    // `kv_heads`/`query_row`/`kv_head` -- see [`EmitError::
    // AttentionBlockMisaligned`]'s own doc.
    let block_width = block_width_for(numeric_policy);
    if single_range_dynamic && block_width > 1 && !head_dim.is_multiple_of(8) {
        return Err(EmitError::AttentionBlockMisaligned {
            node: resolved.node,
            head_dim: *head_dim,
        });
    }
    // The stride loop's live upper bound: every key past this point would
    // hit the `relative > new_upper` / `relative < cached_lower` `continue`
    // on every remaining iteration within THIS simdgroup's own assigned
    // subset (both bands are monotone in `key`), so stopping here changes
    // no arithmetic and no read this simdgroup would have skipped anyway --
    // it removes iteration/address/control overhead, not any accumulate or
    // K/V load, and the dead-band checks already preceded every load and
    // multiply-add in the loop this replaces. This does NOT halve dispatched
    // work: chunks stride-interleave keys across simdgroups, so a "fake"
    // key was never confined to one simdgroup's idle range, and this bound
    // is evaluated per simdgroup, independently of the chunk-sizing decision
    // above. `new_key_rows` is still the compiled buffer extent (the read
    // bound never moves); only the iteration count does.
    let last_key_decl = if single_range_dynamic {
        "long last_key = cached_key_rows + min(new_key_rows - 1L, query_row + new_upper);\n"
    } else {
        "long last_key = cached_key_rows + new_key_rows - 1L;\n"
    };
    // `grid_threads` widens the dispatch by the compiled `ATTENTION_SPLIT_MAX`
    // (never the live `splits_for` result) exactly when a merge is needed --
    // `tgid`'s own decode below must divide out that SAME compiled multiplier,
    // not the live `u.splits` value, or a `tgid` beyond `rows * kv_heads *
    // live_splits` (which the widened grid always dispatches) decodes a
    // `query_row` past the real row count and reads/writes out of bounds.
    // Computed once here so the decode and `final_store`'s scratch-vs-direct
    // branch below can never disagree on which case this compiled kernel is.
    let merge_needed = cached_attention_merge_needed(&resolved.kind, numeric_policy);
    // ROW 385: below the split-at-scale knee, `query_groups` moves out of
    // this threadgroup's own width (`tiled_gemm_threadgroup_width`'s own
    // `CachedAttention` arm) and into a threadgroup-COUNT factor instead, so
    // `group` decodes off `tgid` here rather than off `vector_index`.
    // `entry_name` names this decision (`_qh{0|1}`) precisely because it
    // changes the text below -- the compiled `kv-capacity-bucket` extent
    // still never appears in the text ITSELF, only this boolean does.
    let per_query_head_grid = cached_attention_per_query_head_grid(
        single_range_dynamic,
        *cached_key_rows + *new_key_rows,
    );
    if single_range_dynamic {
        let grid_splits = if merge_needed {
            crate::sized::ATTENTION_SPLIT_MAX
        } else {
            1
        };
        let tgid_decode = if per_query_head_grid {
            "long kv_head_and_group = (long)tgid % (kv_heads * query_groups);\n    long query_row_and_split = (long)tgid / (kv_heads * query_groups);\n    long kv_head = kv_head_and_group / query_groups;\n    long group = kv_head_and_group % query_groups;\n    long chunk = vector_index % cap;\n"
        } else {
            "long chunk = vector_index % cap;\n    long group = (vector_index / cap) % query_groups;\n    long kv_head = (long)tgid % kv_heads;\n    long query_row_and_split = (long)tgid / kv_heads;\n"
        };
        source.push_str(&format!(
            "    long vector_index = (long)gid / 32L; uint lane = gid % 32u;\n    if (vector_index >= u.total_elements) {{ return; }}\n    constexpr long cap = {cap};\n    long chunks = u.context_chunks;\n    long splits = u.splits;\n    {tgid_decode}    constexpr long grid_splits = {grid_splits};\n    long split = query_row_and_split % grid_splits;\n    long query_row = query_row_and_split / grid_splits;\n    if (split >= splits) {{ return; }}\n    long query_index = query_row * (kv_heads * query_groups) + kv_head * query_groups + group;\n    long query_head = kv_head * query_groups + group;\n    long qbase = query_row * (kv_heads * query_groups * (head_dim / 2)) + query_head * (head_dim / 2);\n    long local_group_index = group * cap + chunk;\n    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];\n    for (long dimension = 0; dimension < (head_dim + 31) / 32; dimension++) {{ weighted[dimension] = 0.0f; }}\n"
        ));
        source.push_str(&format!("    {last_key_decl}"));
        // Declared once here (not per `block_width` arm) because the
        // cross-chunk merge below -- `shared_m`/`shared_l`/`shared_o` reads
        // and the `final_store` that follows it -- runs identically whether
        // `block_width <= 1` (bit_exact) or `> 1` (the block-staged body);
        // only the `block_width > 1` arm additionally reuses `shared_o` as
        // its own V-accumulate scratch (this function's own doc on that
        // arm), never a second, separate allocation.
        source.push_str("    threadgroup float shared_m[query_groups * cap]; threadgroup float shared_l[query_groups * cap]; threadgroup float shared_o[query_groups * cap * head_dim];\n");
        // Redesign §4c: this threadgroup's slice of the LIVE key range
        // (`last_key + 1`, ROW 366's live upper) -- `ceil_div(live, splits)`
        // sized so every split but the last is exactly `slice_len` keys and
        // the last absorbs the remainder; `lo >= hi` (a `split` past the end
        // of a short live range) leaves the per-key loop below empty, which
        // is already this kernel's identity partial (`maximum = -INFINITY`,
        // `sum = 0.0`, `weighted` zeroed above) -- no separate branch needed.
        source.push_str(
            "    long live = last_key + 1L;\n    long slice_len = (live + splits - 1L) / splits;\n    long lo = split * slice_len;\n    long hi = min(lo + slice_len, live);\n",
        );
        // Idle simdgroups (`chunk >= chunks`, live count below the compiled
        // maximum) skip the walk entirely and keep the identity partial
        // (`maximum = -INFINITY`, `sum = 0.0`, `weighted` zeroed above) --
        // the merge below already treats a `-INFINITY` partial as
        // zero-weight (`rescale = 0.0f`), so an idle simdgroup contributes
        // nothing to the merged result, bit-for-bit.
        // `block_width == 1` renders EXACTLY the strictly-sequential per-key
        // walk (byte for byte) -- see [`block_width_for`]'s own doc for why
        // this is the `bit_exact` lowering, not a fallback bolted on beside
        // it. `block_width > 1` renders §4b's block-staged walk: llama's
        // `kernel_flash_attn_ext_vec` shape (ggml-metal.metal:4014-4143)
        // ported onto this kernel's cached/new dual-range band masking --
        // `ss[]` stages `block_width` raw scores via a `float4` Q·K dot plus
        // an 8-lane `simd_shuffle_down` tree reduce (`ty = lane/8` selects
        // one of 4 concurrent keys, `tx = lane%8` selects a float4 chunk of
        // the real/imaginary planes), one `simd_max`/`simd_sum` combine per
        // 32-lane sub-block (not per key), then a threadgroup-broadcast read
        // of each key's softmax weight back out of the SAME `ss[]` slots for
        // the V accumulate -- llama's own `ss[]` reuse
        // (ggml-metal.metal:4114, `ss[tiisg] = vs;`). `ss[]` is sized and
        // indexed per `local_group_index` exactly like `shared_m`/
        // `shared_l`/`shared_o` below it, so concurrent simdgroups in the
        // same threadgroup never alias each other's staged scores.
        if block_width <= 1 {
            source.push_str("    if (chunk < chunks) {\n    for (long key = lo + chunk; key < hi; key += chunks) {\n        bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n        long relative = (cached ? key - cached_key_rows : new_index) - query_row;\n        if (cached && relative < cached_lower) { continue; }\n        if (!cached && relative > new_upper) { continue; }\n        long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);\n        float partial_score = 0.0f;\n        for (long pair = (long)lane; pair < head_dim / 2; pair += 32L) {\n            partial_score += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);\n            partial_score += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);\n        }\n        float score = simd_broadcast_first(simd_sum(partial_score)) * scale;\n        float next_max = max(maximum, score);\n        float weight = exp(score - next_max); float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);\n        sum = sum * rescale + weight;\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {\n            long local_dimension = dimension / 32L;\n            weighted[local_dimension] = weighted[local_dimension] * rescale + weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);\n        }\n        maximum = next_max;\n    }\n    }\n");
        } else {
            // The float4/ty-group V accumulate (llama's `kernel_flash_attn_
            // ext_vec` register form, ggml-metal.metal:4125-4143) needs each
            // `tx` lane's `v_registers` float4 loads to land on a 16-byte
            // boundary and never read past `head_dim` floats -- true only
            // when `head_dim` is itself a multiple of 32 (8 `tx` lanes times
            // 4 floats/float4). Smaller/odd `head_dim` (the `head_dim == 8`
            // fixtures below) keep today's scalar per-key V loop, byte for
            // byte, rather than mis-sizing the vector loads or rejecting a
            // shape the Q·K side already renders correctly.
            let v_accumulate = if head_dim.is_multiple_of(32) {
                let v_registers = head_dim / 8 / 4;
                format!(
                    "            {{\n                constexpr long v_registers = {v_registers};\n                float4 v_acc[v_registers];\n                for (long register_index = 0L; register_index < v_registers; register_index++) {{ v_acc[register_index] = float4(0.0f); }}\n                for (long cc4 = 0L; cc4 < 8L; cc4++) {{\n                    long local_index = block_start + sub + 4L * cc4 + (long)ty;\n                    bool valid = local_index < min(block_start + sub + 32L, num_local_keys);\n                    long key = slice_start + local_index * chunks;\n                    bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n                    long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);\n                    float key_weight = valid ? ss[local_group_index * block_width + (local_index - block_start)] : 0.0f;\n                    device const {element_type}4* v4 = (device const {element_type}4*)((cached ? in6 : in7) + kbase * 2);\n                    for (long register_index = 0L; register_index < v_registers; register_index++) {{\n                        v_acc[register_index] += valid ? float4(v4[(long)tx + 8L * register_index]) * key_weight : float4(0.0f);\n                    }}\n                }}\n                for (long register_index = 0L; register_index < v_registers; register_index++) {{\n                    v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 8);\n                    v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 16);\n                }}\n                if (ty == 0) {{\n                    threadgroup float4* shared_o4 = (threadgroup float4*)(shared_o + local_group_index * head_dim);\n                    for (long register_index = 0L; register_index < v_registers; register_index++) {{ shared_o4[(long)tx + 8L * register_index] = v_acc[register_index]; }}\n                }}\n                simdgroup_barrier(mem_flags::mem_threadgroup);\n                for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{\n                    long local_dimension = dimension / 32L;\n                    weighted[local_dimension] += shared_o[local_group_index * head_dim + dimension];\n                }}\n                simdgroup_barrier(mem_flags::mem_threadgroup);\n            }}\n"
                )
            } else {
                "            for (long local_index = block_start + sub; local_index < min(block_start + sub + 32L, num_local_keys); local_index++) {\n                long key = slice_start + local_index * chunks;\n                bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n                long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);\n                float key_weight = ss[local_group_index * block_width + (local_index - block_start)];\n                for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {\n                    long local_dimension = dimension / 32L;\n                    weighted[local_dimension] += key_weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);\n                }\n            }\n".to_string()
            };
            source.push_str(&format!(
                "    constexpr long block_width = {block_width};\n    short ty = (short)(lane / 8u); short tx = (short)(lane % 8u);\n    threadgroup float ss[query_groups * cap * block_width];\n    if (chunk < chunks) {{\n    long slice_start = lo + chunk;\n    long num_local_keys = (slice_start < hi) ? ((hi - 1L - slice_start) / chunks) + 1L : 0L;\n    for (long block_start = 0L; block_start < num_local_keys; block_start += block_width) {{\n        for (long cc = 0L; cc < block_width / 4L; cc++) {{\n            long local_index = block_start + 4L * cc + (long)ty;\n            bool valid = local_index < num_local_keys;\n            long key = slice_start + local_index * chunks;\n            bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n            if (valid) {{\n                long relative = (cached ? key - cached_key_rows : new_index) - query_row;\n                if (cached && relative < cached_lower) {{ valid = false; }}\n                if (!cached && relative > new_upper) {{ valid = false; }}\n            }}\n            long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);\n            float partial_score = 0.0f;\n            if (valid) {{\n                device const {element_type}4* qr4 = (device const {element_type}4*)(in0 + qbase);\n                device const {element_type}4* qi4 = (device const {element_type}4*)(in1 + qbase);\n                device const {element_type}4* kr4 = (device const {element_type}4*)((cached ? in2 : in4) + kbase);\n                device const {element_type}4* ki4 = (device const {element_type}4*)((cached ? in3 : in5) + kbase);\n                for (short index = tx; index < (short)((head_dim / 2) / 4L); index += 8) {{\n                    partial_score += dot(kr4[index], qr4[index]);\n                    partial_score += dot(ki4[index], qi4[index]);\n                }}\n            }}\n            partial_score += simd_shuffle_down(partial_score, 4);\n            partial_score += simd_shuffle_down(partial_score, 2);\n            partial_score += simd_shuffle_down(partial_score, 1);\n            if (tx == 0) {{ ss[local_group_index * block_width + 4L * cc + (long)ty] = valid ? partial_score * scale : -INFINITY; }}\n        }}\n        simdgroup_barrier(mem_flags::mem_threadgroup);\n        for (long sub = 0L; sub < block_width; sub += 32L) {{\n            long lane_index = sub + (long)lane;\n            float raw_score = ss[local_group_index * block_width + lane_index];\n            float next_max = simd_max(max(maximum, raw_score));\n            float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);\n            float weight = exp(raw_score - next_max);\n            sum = sum * rescale + simd_sum(weight);\n            ss[local_group_index * block_width + lane_index] = weight;\n            for (long dimension = 0L; dimension < (head_dim + 31) / 32; dimension++) {{ weighted[dimension] *= rescale; }}\n            maximum = next_max;\n            simdgroup_barrier(mem_flags::mem_threadgroup);\n{v_accumulate}            simdgroup_barrier(mem_flags::mem_threadgroup);\n        }}\n    }}\n    }}\n"
            ));
        }
        // Redesign §4c ([`NumericRewrite::ContextSplitMerge`]): once the
        // active policy admits it, this simdgroup-level combine's own
        // result (`merged_max`/`merged_sum`/`weighted`, exactly what the
        // `bit_exact` arm below normalizes straight into `out`) is instead
        // the SPLIT kernel's own partial for split 0 -- written raw
        // (un-normalized) into the scratch buffer this position's
        // `Binding::Scratch` slot backs, at `splits` (the LIVE `u.splits`
        // value, not the compiled `ATTENTION_SPLIT_MAX`) stride so a later
        // slice's real `split = threadgroup_index % splits` can address its
        // own slot with the identical layout. This must be the same live
        // value `crate::metal::cached_attention_scratch_len` sizes the
        // buffer with (`splits_for(cached_key_rows + new_key_rows, ..)` at
        // `query_rows > 1`, the compiled max only at `query_rows == 1`) --
        // striding by the compiled max here while the buffer holds only
        // `splits` slots per row was an out-of-bounds write on every
        // `query_index > 0` (production, 2026-09-07: turn two of a chat,
        // ~600 cached keys plus ~100 new rows, produced 1024 `!!!!` tokens).
        // `render_cached_attention_merge` reads exactly this layout back and
        // performs the online-softmax combine over every live split -- the
        // final `weighted[..] / sum` normalize this arm always did is
        // deferred to that kernel, generalizing this exact combine one
        // hardware level up (`ContextSplitMerge`'s own doc). `split` is the
        // real per-threadgroup value this function's own index-unpack
        // derived from `tgid` above -- at `u.splits == 1` it is always `0`,
        // so this reduces to exactly the prior forced-`0L` behaviour byte
        // for byte.
        let final_store = if merge_needed {
            "        device float* attn_scratch = (device float*)out;\n        long scratch_index = (query_index * splits + split) * (2L + head_dim);\n        if (lane == 0u) { attn_scratch[scratch_index] = merged_max; attn_scratch[scratch_index + 1] = merged_sum; }\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) { long local_dimension = dimension / 32L; attn_scratch[scratch_index + 2L + dimension] = weighted[local_dimension]; }\n".to_string()
        } else {
            format!(
                "        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ long local_dimension = dimension / 32L; out[query_index * head_dim + dimension] = ({element_type})(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum); }}\n"
            )
        };
        source.push_str(&format!(
            "    if (lane == 0u) {{ shared_m[local_group_index] = maximum; shared_l[local_group_index] = sum; }}\n    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ shared_o[local_group_index * head_dim + dimension] = weighted[dimension / 32L]; }}\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (chunk == 0L) {{\n        float merged_max = -INFINITY;\n        for (long c = 0; c < cap; c++) {{ merged_max = max(merged_max, shared_m[group * cap + c]); }}\n        float merged_sum = 0.0f;\n        for (long c = 0; c < cap; c++) {{\n            float partial_max = shared_m[group * cap + c];\n            float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);\n            merged_sum += shared_l[group * cap + c] * rescale;\n        }}\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{\n            long local_dimension = dimension / 32L;\n            float acc = 0.0f;\n            for (long c = 0; c < cap; c++) {{\n                float partial_max = shared_m[group * cap + c];\n                float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);\n                acc += shared_o[(group * cap + c) * head_dim + dimension] * rescale;\n            }}\n            weighted[local_dimension] = acc;\n        }}\n        sum = merged_sum;\n{final_store}    }}\n}}\n"
        ));
    } else if context_chunks <= 1 {
        source.push_str("    long vector_index = (long)gid / 32L; uint lane = gid % 32u;\n    if (vector_index >= u.total_elements) { return; }\n    long query_index = vector_index;\n    long query_row = query_index / (kv_heads * query_groups);\n    long remainder = query_index % (kv_heads * query_groups);\n    long kv_head = remainder / query_groups;\n    long group = remainder % query_groups;\n    long query_head = kv_head * query_groups + group;\n    long qbase = query_row * (kv_heads * query_groups * (head_dim / 2)) + query_head * (head_dim / 2);\n    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];\n    for (long dimension = 0; dimension < (head_dim + 31) / 32; dimension++) { weighted[dimension] = 0.0f; }\n");
        source.push_str(&format!("    {last_key_decl}"));
        // Each lane reads its own K/V elements straight from device memory
        // into registers, llama.cpp's `kernel_flash_attn_ext_vec` shape
        // (ggml-metal.metal:4037-4058 for K, the V accumulate mirrors it) --
        // no `threadgroup` staging, so no barrier is needed inside the loop:
        // nothing is shared across lanes or simdgroups until `simd_sum`
        // reduces the per-lane partial dot product within this simdgroup.
        source.push_str("    for (long key = 0; key <= last_key; key++) {\n        bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n        long relative = (cached ? key - cached_key_rows : new_index) - query_row;\n        if (cached && relative < cached_lower) { continue; }\n        if (!cached && relative > new_upper) { continue; }\n        long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);\n        float partial_score = 0.0f;\n        for (long pair = (long)lane; pair < head_dim / 2; pair += 32L) {\n            partial_score += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);\n            partial_score += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);\n        }\n        float score = simd_broadcast_first(simd_sum(partial_score)) * scale;\n        float next_max = max(maximum, score);\n        float weight = exp(score - next_max); float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);\n        sum = sum * rescale + weight;\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {\n            long local_dimension = dimension / 32L;\n            weighted[local_dimension] = weighted[local_dimension] * rescale + weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);\n        }\n        maximum = next_max;\n    }\n");
        source.push_str(&format!("    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ long local_dimension = dimension / 32L; out[query_index * head_dim + dimension] = ({element_type})(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum); }}\n}}\n"));
    } else {
        // `context_chunks` simdgroups per (query_row, kv_head, group) triple
        // now share one threadgroup -- `local_group_index` (`group *
        // context_chunks + chunk`) is what `tiled_gemm_threadgroup_width`'s
        // `CachedAttention` arm sizes the dispatch width against, and its
        // encoding cycles `chunk` fastest so the SAME `dispatchThreads_
        // threadsPerThreadgroup` linear-grouping invariant this function's
        // doc already relies on for `group` still holds. Each simdgroup
        // walks ONLY the keys assigned to its `chunk` (`key = chunk; key <
        // total; key += context_chunks`), llama.cpp's own context-stride
        // partition (`ic0 += C*nsg`, ggml-metal.metal:4016), reading K/V
        // straight from device memory into registers exactly like the
        // `chunks<=1` body above -- so there is no per-key barrier here
        // either, and the set of keys this chunk visits is identical to the
        // old `key % context_chunks == chunk` gate, visited in the same
        // increasing order, so the accumulated floats are bit-identical.
        source.push_str(&format!(
            "    long vector_index = (long)gid / 32L; uint lane = gid % 32u;\n    if (vector_index >= u.total_elements) {{ return; }}\n    constexpr long context_chunks = {context_chunks};\n    long query_index = vector_index / context_chunks;\n    long chunk = vector_index % context_chunks;\n    long query_row = query_index / (kv_heads * query_groups);\n    long remainder = query_index % (kv_heads * query_groups);\n    long kv_head = remainder / query_groups;\n    long group = remainder % query_groups;\n    long query_head = kv_head * query_groups + group;\n    long qbase = query_row * (kv_heads * query_groups * (head_dim / 2)) + query_head * (head_dim / 2);\n    long local_group_index = group * context_chunks + chunk;\n    float maximum = -INFINITY; float sum = 0.0f; float weighted[(head_dim + 31) / 32];\n    for (long dimension = 0; dimension < (head_dim + 31) / 32; dimension++) {{ weighted[dimension] = 0.0f; }}\n"
        ));
        source.push_str(&format!("    {last_key_decl}"));
        source.push_str("    for (long key = chunk; key <= last_key; key += context_chunks) {\n        bool cached = key < cached_key_rows; long new_index = key - cached_key_rows;\n        long relative = (cached ? key - cached_key_rows : new_index) - query_row;\n        if (cached && relative < cached_lower) { continue; }\n        if (!cached && relative > new_upper) { continue; }\n        long kbase = (cached ? key : new_index) * (kv_heads * (head_dim / 2)) + kv_head * (head_dim / 2);\n        float partial_score = 0.0f;\n        for (long pair = (long)lane; pair < head_dim / 2; pair += 32L) {\n            partial_score += in0[qbase + pair] * (cached ? in2[kbase + pair] : in4[kbase + pair]);\n            partial_score += in1[qbase + pair] * (cached ? in3[kbase + pair] : in5[kbase + pair]);\n        }\n        float score = simd_broadcast_first(simd_sum(partial_score)) * scale;\n        float next_max = max(maximum, score);\n        float weight = exp(score - next_max); float rescale = (maximum == -INFINITY) ? 0.0f : exp(maximum - next_max);\n        sum = sum * rescale + weight;\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {\n            long local_dimension = dimension / 32L;\n            weighted[local_dimension] = weighted[local_dimension] * rescale + weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);\n        }\n        maximum = next_max;\n    }\n");
        // Single cross-simdgroup merge: each chunk's own (max, sum,
        // weighted) is the same online-softmax state the `chunks<=1` body
        // already computes over its own key subset; combining them is the
        // one piece of NEW arithmetic (`render_cached_attention`'s own
        // doc) -- rescale each chunk's partial by `exp(m_i - m_max)`, sum
        // `l`, sum `o`, exactly llama.cpp's `kernel_flash_attn_ext_vec`
        // cross-simdgroup reduction (ggml-metal.metal). Only `chunk == 0`
        // writes the merged result and the final output -- the other
        // chunks' registers are dead past this point.
        source.push_str(&format!(
            "    threadgroup float shared_m[query_groups * context_chunks]; threadgroup float shared_l[query_groups * context_chunks]; threadgroup float shared_o[query_groups * context_chunks * head_dim];\n    if (lane == 0u) {{ shared_m[local_group_index] = maximum; shared_l[local_group_index] = sum; }}\n    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ shared_o[local_group_index * head_dim + dimension] = weighted[dimension / 32L]; }}\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    if (chunk == 0L) {{\n        float merged_max = -INFINITY;\n        for (long c = 0; c < context_chunks; c++) {{ merged_max = max(merged_max, shared_m[group * context_chunks + c]); }}\n        float merged_sum = 0.0f;\n        for (long c = 0; c < context_chunks; c++) {{\n            float partial_max = shared_m[group * context_chunks + c];\n            float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);\n            merged_sum += shared_l[group * context_chunks + c] * rescale;\n        }}\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{\n            long local_dimension = dimension / 32L;\n            float acc = 0.0f;\n            for (long c = 0; c < context_chunks; c++) {{\n                float partial_max = shared_m[group * context_chunks + c];\n                float rescale = (partial_max == -INFINITY) ? 0.0f : exp(partial_max - merged_max);\n                acc += shared_o[(group * context_chunks + c) * head_dim + dimension] * rescale;\n            }}\n            weighted[local_dimension] = acc;\n        }}\n        sum = merged_sum;\n        for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{ long local_dimension = dimension / 32L; out[query_index * head_dim + dimension] = ({element_type})(sum == 0.0f ? 0.0f : weighted[local_dimension] / sum); }}\n    }}\n}}\n"
        ));
    }
    let _ = query_rows;
    Ok(source)
}

/// `BoundOpKind::CachedAttention`'s MERGE kernel -- [`render_cached_attention`]'s
/// own doc, redesign §4c: reads back the `(max, sum, weighted[head_dim])`
/// partial each of `u.splits` split-kernel threadgroups wrote into the
/// scratch buffer (one simdgroup per output row) and combines them with the
/// SAME online-softmax rescale llama.cpp's own
/// `kernel_flash_attn_ext_vec_reduce` uses (`ggml-metal-ops.cpp:2063-2097`
/// on `origin/master`, this crate's `attention-kernel-design.md` §4c cites
/// it verbatim): lane `i` (`i < splits`) loads that split's own `(M_i,
/// S_i)`; `simd_max`/`simd_sum` combine the up-to-32 lanes (one lane per
/// split, `ATTENTION_SPLIT_MAX <= 32` so every live split fits in one
/// simdgroup's shuffle network, the same reason llama's own reduce needs no
/// tree); each lane then walks its own lane-strided subset of `head_dim`
/// dimensions, re-deriving every split's rescale weight straight from
/// device memory (cheap here -- this kernel is `O(splits * head_dim)`, not
/// the hot per-key loop) rather than shuffling `M`/`S` across lanes. At
/// `u.splits == 1` this reduces to exactly the prior forced-`split = 0`
/// copy-and-normalize, since the single live lane's rescale weight is
/// always `1.0` (or `0.0` under the all-`-INFINITY`/empty-context corner).
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
fn render_cached_attention_merge(resolved: &BoundOp, entry: &str) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention { head_dim, .. } = &resolved.kind else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "cached_attention_merge",
            found: resolved.kind.name(),
        });
    };
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let mut source = String::new();
    preamble(&mut source);
    source.push_str("struct Uniforms { long total_elements; long splits; };\n\n");
    source.push_str(&format!(
        "kernel void {entry}(device const float* in0 [[buffer(0)]], device {element_type}* out [[buffer(1)]], constant Uniforms& u [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{\n"
    ));
    source.push_str(
        "    long vector_index = (long)gid / 32L; uint lane = gid % 32u;\n    if (vector_index >= u.total_elements) { return; }\n",
    );
    // `splits` is the SAME live `u.splits` value the split kernel's own
    // `final_store` addresses its scratch write with
    // (`render_cached_attention`'s doc) -- and the same value
    // `crate::metal::cached_attention_scratch_len` sized this buffer's
    // per-row extent against. Reading it back with the compiled
    // `ATTENTION_SPLIT_MAX` instead (this function's own bug until fixed
    // alongside the split kernel) would address a stride the allocator
    // never reserved.
    source.push_str(&format!(
        "    constexpr long head_dim = {head_dim};\n    long query_index = vector_index;\n    long splits = u.splits;\n    long iwg = (long)lane;\n    long own_scratch = (query_index * splits + iwg) * (2L + head_dim);\n    float own_max = (iwg < splits) ? in0[own_scratch] : -INFINITY;\n    float own_sum = (iwg < splits) ? in0[own_scratch + 1] : 0.0f;\n    float global_max = simd_max(own_max);\n    float own_weight = (own_max == -INFINITY) ? 0.0f : exp(own_max - global_max);\n    float total_sum = simd_sum(own_weight * own_sum);\n    float inv_sum = (total_sum == 0.0f) ? 0.0f : 1.0f / total_sum;\n    for (long dimension = (long)lane; dimension < head_dim; dimension += 32L) {{\n        float accumulated = 0.0f;\n        for (long split = 0; split < splits; split++) {{\n            long scratch_index = (query_index * splits + split) * (2L + head_dim);\n            float split_max = in0[scratch_index];\n            float split_weight = (split_max == -INFINITY) ? 0.0f : exp(split_max - global_max);\n            accumulated += split_weight * in0[scratch_index + 2L + dimension];\n        }}\n        out[query_index * head_dim + dimension] = ({element_type})(accumulated * inv_sum);\n    }}\n}}\n"
    ));
    Ok(source)
}

/// Whether `resolved` needs a companion merge dispatch under `numeric_policy`
/// and, if so, that dispatch's [`Kernel`] -- `None` for every op kind other
/// than `CachedAttention` and for `CachedAttention` itself when
/// [`cached_attention_merge_needed`] withholds `ContextSplitMerge` (the
/// `bit_exact` lowering: single dispatch, byte-identical to before this
/// redesign). This is [`emit`]'s own companion rather than a change to
/// `emit`'s signature: every OTHER caller of `emit` (`cuda.rs`, `wgsl.rs`,
/// the module doc's own example) keeps binding one `BoundOp` to one
/// `Kernel`, and `CachedAttention`'s own two-dispatch shape is additive —
/// `crate::metal`'s plan-resolution path calls this alongside `emit` for
/// every position, exactly as it already calls `kernel_dispatch_shape`
/// alongside `emit`.
#[cfg(any(test, all(feature = "metal", target_os = "macos")))]
pub(crate) fn emit_cached_attention_merge(
    resolved: &BoundOp,
    numeric_policy: NumericPolicy,
) -> Result<Option<Kernel>, EmitError> {
    if !cached_attention_merge_needed(&resolved.kind, numeric_policy) {
        return Ok(None);
    }
    validate(resolved)?;
    let entry = alloc::format!("{}_merge", entry_name(resolved));
    let source = render_cached_attention_merge(resolved, &entry)?;
    let total_elements = resolved
        .extents
        .iter()
        .product::<u64>()
        .checked_div(match &resolved.kind {
            BoundOpKind::CachedAttention { head_dim, .. } => *head_dim,
            _ => 1,
        })
        .unwrap_or(0);
    Ok(Some(Kernel {
        source,
        entry,
        bindings: merge_bindings(resolved),
        grid: GridSpec {
            threads: total_elements * SIMD_WIDTH,
            threadgroup_width: None,
        },
    }))
}

fn render_elementwise(
    resolved: &BoundOp,
    entry: &str,
    quantized: &[Option<PackedCodec>],
) -> Result<String, EmitError> {
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let coordinate_dims = elementwise_coordinate_dims(resolved, gather_count);

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long total_elements;\n");
    source.push_str(&format!("    long extents[{rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(
        &mut source,
        quantized,
        0,
        gather_count,
        entry,
        element_type,
        false,
    );
    source.push_str("    if ((long)gid >= u.total_elements) { return; }\n");

    if !coordinate_dims.is_empty() {
        let coordinate_type = elementwise_coordinate_type(resolved);
        source.push_str(&format!("    {coordinate_type} coord[{rank_len}];\n"));
        source.push_str(&format!(
            "    {coordinate_type} remaining = ({coordinate_type})gid;\n"
        ));
        for dim in (0..resolved.extents.len()).rev() {
            if coordinate_dims.contains(&dim) {
                source.push_str(&format!(
                    "    coord[{dim}] = remaining % ({coordinate_type})u.extents[{dim}];\n"
                ));
            }
            source.push_str(&format!(
                "    remaining /= ({coordinate_type})u.extents[{dim}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        let dense_operand = dense_layout(&resolved.operands()[index].1, &resolved.extents);
        if dense_operand {
            source.push_str(&format!(
                "    long off{index} = u.operand_base[{index}] + (long)gid;\n"
            ));
        } else {
            source.push_str(&format!("    long off{index} = u.operand_base[{index}];\n"));
            for &dim in &coordinate_dims {
                source.push_str(&format!(
                    "    off{index} += (long)coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
                ));
            }
        }
        if let Some(slot) = gather_slot {
            push_gather_fetch(
                &mut source,
                index,
                *slot,
                rank,
                "coord",
                &format!("off{index}"),
            );
        }
    }

    source.push_str(&format!(
        "    {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "    scratch[{index}] = {};\n",
            operand_read(index, &format!("off{index}"), codec)
        ));
    }

    let result = push_body_steps(&mut source, resolved.element_body(), "    ", element_type);
    source.push_str(&format!("    out[gid] = {result};\n"));
    source.push_str("}\n");
    Ok(source)
}

fn elementwise_coordinate_dims(resolved: &BoundOp, gather_count: usize) -> Vec<usize> {
    let rank = resolved.extents.len();
    let mut coordinate_dims: Vec<usize> = if gather_count > 0 {
        (0..rank).collect()
    } else {
        resolved
            .operands()
            .iter()
            .filter(|(_, layout, _)| !dense_layout(layout, &resolved.extents))
            .flat_map(|(_, layout, _)| {
                layout
                    .strides
                    .iter()
                    .enumerate()
                    .filter_map(|(dimension, stride)| (*stride != 0).then_some(dimension))
            })
            .collect()
    };
    coordinate_dims.sort_unstable();
    coordinate_dims.dedup();
    coordinate_dims
}

fn elementwise_coordinate_type(resolved: &BoundOp) -> &'static str {
    if resolved.extents.iter().product::<u64>() <= u32::MAX as u64 {
        "uint"
    } else {
        "ulong"
    }
}

/// The exact stride-layout decisions [`render_elementwise`] bakes into MSL:
/// coordinate integer width, which axes it decodes, and which operands take
/// the dense `base + gid` path. Concrete stride values remain uniforms.
fn elementwise_addressing_cache_token(resolved: &BoundOp) -> Option<String> {
    if !matches!(resolved.kind, BoundOpKind::Elementwise { .. }) {
        return None;
    }

    let coordinate_dims = elementwise_coordinate_dims(resolved, gather_count(resolved));
    let mut token = String::from("_ea");
    token.push(if coordinate_dims.is_empty() {
        'n'
    } else if elementwise_coordinate_type(resolved) == "uint" {
        '4'
    } else {
        '8'
    });
    token.push('c');
    for dimension in coordinate_dims {
        token.push('_');
        token.push_str(&dimension.to_string());
    }
    token.push_str("_d");
    for (_, layout, _) in resolved.operands() {
        token.push(if dense_layout(layout, &resolved.extents) {
            '1'
        } else {
            '0'
        });
    }
    Some(token)
}

fn dense_layout(layout: &Layout, extents: &[u64]) -> bool {
    if layout.base < 0 || layout.strides.len() != extents.len() {
        return false;
    }
    let mut expected_stride = 1_i64;
    for (extent, stride) in extents.iter().zip(layout.strides.iter()).rev() {
        if *stride != expected_stride {
            return false;
        }
        expected_stride = expected_stride.saturating_mul(*extent as i64);
    }
    true
}

fn render_reduce(
    resolved: &BoundOp,
    entry: &str,
    quantized: &[Option<PackedCodec>],
    expert_source_mode: bool,
) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op,
        init,
        output_axes,
        epilogue_body,
        epilogue_operands,
        epilogue_broadcast_axes,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "keep::reduce fold",
            found: resolved.kind.name(),
        });
    };
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_dims = reduction_dims(resolved, output_axes);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    // `push_reduce_epilogue_write`'s own doc: `epilogue_operand_strides` is
    // declared over `output_rank` (`output_axes`'s own smaller space) for the
    // pre-existing PLAIN epilogue shape. A non-empty `epilogue_broadcast_axes`
    // (`bind::BoundOpKind::Reduce::epilogue_broadcast_axes`'s own doc: the
    // RMSNorm-shaped `x * inv_rms` "broadcast-reduce" epilogue) needs the
    // WIDER full-rank space instead -- `push_cooperative_reduce_tail` is the
    // only renderer that walks it (see its own doc), so anything else
    // rejects: a mismatched axis set (this bind-time invariant is proven at
    // `bind::epilogue_broadcast_axes_for`, but a stray value here would
    // otherwise silently read/write the wrong element count), a non-
    // cooperative reduce (`push_serial_reduce_body` has no broadcast write),
    // or a tiled-GEMM/row-blocked packed match (`push_tiled_gemm_body`/
    // `push_packed_row_blocked_body` each own their write tail entirely and
    // neither has one).
    let is_broadcast_epilogue = !epilogue_broadcast_axes.is_empty();
    if is_broadcast_epilogue {
        let matches_reduce_dims = epilogue_broadcast_axes.len() == reduce_dims.len()
            && epilogue_broadcast_axes
                .iter()
                .all(|axis| reduce_dims.contains(axis));
        if !matches_reduce_dims
            || !reduce_is_cooperative(resolved)
            || tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some()
            || packed_row_block(resolved, quantized).is_some()
        {
            return Err(EmitError::EpilogueNotSupported {
                node: resolved.node,
                reason: "the broadcast-reduce epilogue only has a Metal renderer for the plain cooperative-reduce path",
            });
        }
    }
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;
    let epilogue_operand_count = epilogue_operands.len();
    // The tiled `simdgroup_matrix` GEMM path (`push_tiled_gemm_body`) writes
    // its output through cooperative per-tile stores this module has no
    // single output-coordinate hook to splice an epilogue tail into -- every
    // other reduce renderer funnels its write through one of
    // `push_serial_reduce_body`/`push_cooperative_reduce_tail`/
    // `push_packed_row_combine_and_write`, which `push_reduce_epilogue_write`
    // now covers, so this is the one shape a fused epilogue is rejected for
    // rather than rendered, the same "no renderer, reject" contract
    // `BoundOpKind::Reduce::epilogue_body`'s own doc names.
    if !reduce_epilogue_is_identity(epilogue_body, epilogue_operands)
        && tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some()
    {
        return Err(EmitError::EpilogueNotSupported {
            node: resolved.node,
            reason: "the tiled simdgroup_matrix GEMM kernel has no epilogue tail yet",
        });
    }

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long output_total;\n");
    source.push_str("    long reduction_total;\n");
    source.push_str(&format!("    long output_extents[{output_rank_len}];\n"));
    source.push_str(&format!("    long reduction_extents[{reduce_rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    source.push_str("    long out_base;\n");
    source.push_str(&format!("    long out_strides[{rank_len}];\n"));
    // Broadcast case: `epilogue_operand_strides` widens from `output_rank_len`
    // to `rank_len` (`x`/`gamma` are read at the FULL `(s, d)` coordinate,
    // `push_cooperative_reduce_tail`'s own broadcast-write loop's doc), and
    // a fresh `broadcast_out_strides` row carries the CONTIGUOUS full-extents
    // layout the materialized output buffer is allocated with -- `out_strides`
    // above stays `out_layout`'s own compact, stride-0-on-broadcast-axes
    // addressing (still needed to locate the fold's own scalar), a genuinely
    // different address space from the widened write.
    let epilogue_stride_rank_len = if is_broadcast_epilogue {
        rank_len
    } else {
        output_rank_len
    };
    if epilogue_operand_count > 0 {
        source.push_str(&format!(
            "    long epilogue_operand_base[{epilogue_operand_count}];\n"
        ));
        source.push_str(&format!(
            "    long epilogue_operand_strides[{epilogue_operand_count}][{epilogue_stride_rank_len}];\n"
        ));
    }
    if is_broadcast_epilogue {
        source.push_str(&format!("    long broadcast_out_strides[{rank_len}];\n"));
    }
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    // split-K needs the actual per-dispatch threadgroup width back
    // (`kernel_signature`'s `tptg` param) ONLY on the row-blocked packed
    // path -- every other reduce kernel keeps its signature untouched, and
    // with the feature off this is always `false`, which is what makes
    // "split == 1 reproduces the current kernel exactly" hold at the source
    // level, not just numerically.
    let include_threadgroup_width =
        cfg!(feature = "metal-q4k-split-k") && packed_row_block(resolved, quantized).is_some();
    kernel_signature(
        &mut source,
        quantized,
        epilogue_operand_count,
        gather_count,
        entry,
        element_type,
        include_threadgroup_width,
    );

    if reduce_is_cooperative(resolved) {
        push_cooperative_reduce_body(
            &mut source,
            resolved,
            *reduce_op,
            *init,
            output_axes,
            &reduce_dims,
            rank,
            quantized,
            element_type,
            epilogue_body,
            epilogue_operands,
            is_broadcast_epilogue,
            expert_source_mode,
        )?;
    } else {
        push_serial_reduce_body(
            &mut source,
            resolved,
            *reduce_op,
            *init,
            output_axes,
            &reduce_dims,
            rank,
            rank_len,
            output_rank,
            output_rank_len,
            reduce_rank,
            reduce_rank_len,
            operand_count,
            &gather_slots,
            quantized,
            element_type,
            epilogue_body,
            epilogue_operands,
        );
    }
    source.push_str("}\n");
    Ok(source)
}

/// Shared write tail for every reduce renderer -- [`push_serial_reduce_body`],
/// [`push_cooperative_reduce_tail`], and [`push_packed_row_combine_and_write`]
/// -- so a fused [`BoundOpKind::Reduce::epilogue_body`] renders identically
/// regardless of which fold produced the value being written. `coord` gives
/// the OUTPUT-axis-order coordinate expression for axis `dim` (an
/// `output_coord[dim]`-style array read, or a plain `"0"` for a rank-0
/// output where no such array exists) -- [`BoundOpKind::Reduce::
/// epilogue_operands`]'s own doc is why that space, not `full_coord`'s full
/// iteration rank, is what `epilogue_operand_strides` is declared over.
/// When the epilogue is the untouched identity default
/// ([`reduce_epilogue_is_identity`]), this emits exactly the one-line
/// `out[...] = accumulator;` every caller emitted before epilogue fusion
/// existed -- byte-for-byte, so a program with no fused epilogue anywhere
/// renders the same kernel source it always did.
#[allow(clippy::too_many_arguments)]
fn push_reduce_epilogue_write(
    source: &mut String,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    output_rank: usize,
    element_type: &str,
    indent: &str,
    coord: impl Fn(usize) -> String,
    accumulator_expr: &str,
    out_offset_expr: &str,
) {
    if reduce_epilogue_is_identity(epilogue_body, epilogue_operands) {
        source.push_str(&format!(
            "{indent}out[{out_offset_expr}] = {accumulator_expr};\n"
        ));
        return;
    }
    let epilogue_operand_count = epilogue_operands.len();
    source.push_str(&format!(
        "{indent}{element_type} epi_scratch[{}];\n",
        epilogue_operand_count + 1
    ));
    push_epilogue_operand_reads(
        source,
        0..epilogue_operand_count,
        output_rank,
        indent,
        &coord,
    );
    source.push_str(&format!(
        "{indent}epi_scratch[{epilogue_operand_count}] = {accumulator_expr};\n"
    ));
    let epi_value = push_epilogue_body_steps(source, epilogue_body, indent, element_type);
    source.push_str(&format!("{indent}out[{out_offset_expr}] = {epi_value};\n"));
}

/// The per-element epilogue-operand read loop [`push_reduce_epilogue_write`]
/// and the hoisted broadcast write ([`push_broadcast_epilogue_write`]) both
/// need verbatim: read each epilogue operand (`gamma[d]`, `x[s,d]`, ...) at
/// the current coordinate into `epi_scratch`. Pulled out once the hoist
/// (ROW 370) needed to run it from inside a loop whose invariant steps are
/// declared outside that same loop -- keeping one copy is what stops the two
/// call sites drifting on the offset arithmetic.
fn push_epilogue_operand_reads(
    source: &mut String,
    operand_indices: impl Iterator<Item = usize>,
    output_rank: usize,
    indent: &str,
    coord: impl Fn(usize) -> String,
) {
    for index in operand_indices {
        source.push_str(&format!(
            "{indent}long epi_off{index} = u.epilogue_operand_base[{index}];\n"
        ));
        for dim in 0..output_rank {
            source.push_str(&format!(
                "{indent}epi_off{index} += {} * u.epilogue_operand_strides[{index}][{dim}];\n",
                coord(dim)
            ));
        }
        source.push_str(&format!(
            "{indent}epi_scratch[{index}] = epi{index}[epi_off{index}];\n"
        ));
    }
}

/// True when `epilogue_operands[operand_index]`'s own [`Layout`] never
/// varies along any of `reduce_dims` -- a genuine broadcast operand
/// (`inv_dim`, `eps` in an rmsnorm chain: rank-0, every stride 0) whose
/// value is the same on every pass of [`push_broadcast_epilogue_write`]'s
/// per-lane loop, as opposed to a per-element operand (`x[s,d]`, `gamma[d]`)
/// whose stride along the loop's own axis is nonzero. A gathered operand
/// (`Lookup` present) is never treated as invariant: its effective address
/// depends on an index buffer this analysis does not follow.
fn epilogue_operand_is_loop_invariant(
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    reduce_dims: &[u16],
    operand_index: usize,
) -> bool {
    let Some((_, layout, lookup)) = epilogue_operands.get(operand_index) else {
        return false;
    };
    lookup.is_none() && reduce_dims.iter().all(|&dim| layout.stride(dim) == 0)
}

/// [`push_body_steps`]'s counterpart for [`BoundOpKind::Reduce::
/// epilogue_body`]: identical step-emission shape over the same
/// [`scalar_op_expr`] table, reading `epi_scratch[i]`/`epi_step{k}` instead
/// of `push_body_steps`'s `scratch[i]`/`step{k}` -- the epilogue's operand
/// table is a SEPARATE array from the fold's own per-step `scratch`
/// (`push_reduce_epilogue_write`'s own doc), so the two never share a slot
/// even when a real operand index collides.
fn push_epilogue_body_steps(
    source: &mut String,
    body: &ComposedBody,
    indent: &str,
    element_type: &str,
) -> String {
    crate::epilogue::declare_steps(
        source,
        body,
        "epi_scratch",
        "epi_step",
        scalar_op_expr,
        |source, index, expr| {
            source.push_str(&format!(
                "{indent}{element_type} epi_step{index} = {expr};\n"
            ));
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn push_serial_reduce_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    reduce_dims: &[u16],
    rank: usize,
    rank_len: usize,
    output_rank: usize,
    output_rank_len: usize,
    reduce_rank: usize,
    reduce_rank_len: usize,
    operand_count: usize,
    gather_slots: &[Option<usize>],
    quantized: &[Option<PackedCodec>],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) {
    source.push_str("    if ((long)gid >= u.output_total) { return; }\n");

    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    long output_coord[{output_rank_len}];\n"));
        source.push_str("    long remaining = (long)gid;\n");
        for index in (0..output_rank).rev() {
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                 remaining /= u.output_extents[{index}];\n"
            ));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(init);
    source.push_str(&format!("    {element_type} accumulator = {init_expr};\n"));
    source.push_str(&format!("    bool seeded = {seeded_init};\n"));

    source.push_str("    for (long r = 0; r < u.reduction_total; r++) {\n");
    if reduce_rank > 0 {
        source.push_str(&format!(
            "        long reduction_coord[{reduce_rank_len}];\n"
        ));
        source.push_str("        long remaining_r = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r /= u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!(
                "        full_coord[{dim}] = reduction_coord[{index}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        source.push_str(&format!(
            "        long off{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..rank {
            source.push_str(&format!(
                "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            push_gather_fetch(
                source,
                index,
                *slot,
                rank,
                "full_coord",
                &format!("off{index}"),
            );
        }
    }
    source.push_str(&format!(
        "        {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "        scratch[{index}] = {};\n",
            operand_read(index, &format!("off{index}"), codec)
        ));
    }
    let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    source.push_str("    long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "    out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "    ",
        |dim| {
            if output_rank > 0 {
                format!("output_coord[{dim}]")
            } else {
                "0".to_string()
            }
        },
        "accumulator",
        "out_offset",
    );
}

/// The SIMD-group cooperative fold: `SIMD_WIDTH` lanes split one output
/// element's contraction axis, each striding through `reduction_total` by
/// `SIMD_WIDTH` so every element is visited by exactly one lane, then
/// combine via [`simd_combine_fn`]. Gathered expert products with a route
/// invariant across the reduction use the same fold; their route index is
/// fetched once per simdgroup and broadcast before the strided walk. Only lane 0 writes the result, and only
/// lane 0 seeds from the `BoundOp`'s real `ReduceInit` — every other lane
/// seeds from [`cooperative_identity_token`] so the true seed is folded into
/// the group exactly once (see that function's doc). `gid / SIMD_WIDTH` is a
/// valid output index, and `gid % SIMD_WIDTH` a valid lane-within-group
/// index, because [`GridSpec::threadgroup_width`] always pins the dispatched
/// threadgroup width to a whole multiple of `SIMD_WIDTH` — see
/// `crate::metal::dispatch` — and Metal's `dispatchThreads:
/// threadsPerThreadgroup:` (the API `dispatch` always calls) defines
/// `[[thread_position_in_grid]]` as `threadgroup_position_in_grid *
/// threadgroup_width + thread_position_in_threadgroup` even in the boundary
/// (non-full) threadgroup, so `gid` stays a flat global index unaffected by
/// how many `SIMD_WIDTH`-lane simdgroups the driver packs into one
/// threadgroup. The row-blocked packed path widens this multiple via
/// [`crate::sized::PACKED_ROW_BLOCK_SIMDGROUPS`]; every other cooperative
/// reduce stays at exactly one simdgroup per threadgroup.
/// Gather is out of scope here: [`reduce_is_cooperative`] never selects this
/// path when the op gathers, so operand offsets are read straight off
/// `operand_base`/`operand_strides` with no fetch/fault machinery.
/// See [`PackedRowBlock`]. Emits the whole body for the row-blocked packed
/// path; the caller has already emitted `output_index` (a GROUP index here)
/// and `lane`.
///
/// Whether the Q4_K arm below may defer a sub-block's scale/min to ONCE per
/// sub-block instead of once per element. The identity that makes this legal,
/// `sum_j (scale*nibble_j - min)*act_j == scale*sum(nibble_j*act_j) -
/// min*sum(act_j)`, holds only when the reduction is a plain sum of products:
/// `reduce_op == Add` (`Multiply`/`Maximum`/`Minimum` are also legal under
/// [`is_cooperative_reduce_op`] and all break the identity — a `Maximum`
/// reduce cannot be pulled outside a per-element scale at all) AND the fused
/// element body is EXACTLY `scratch[weight] * scratch[other]`, no other
/// steps (a fused body inserts arbitrary extra `ScalarOp`s between the raw
/// product and the reduce, any of which the identity does not survive).
/// Mirrors `ggml-metal.metal:5157-5175`'s `acc1`/`dall` shape
/// (`docs/discipline.md` ROW 106); the other two codecs are untouched — see
/// this function's own Q5_K/Q6_K arms for why.
fn is_plain_product_reduce(
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    weight: usize,
    other: usize,
) -> bool {
    if reduce_op != ScalarOp::Add {
        return false;
    }
    let [step] = resolved.element_body().steps.as_slice() else {
        return false;
    };
    if step.op != ScalarOp::Multiply {
        return false;
    }
    let weight = weight as u16;
    let other = other as u16;
    matches!(
        step.args.as_slice(),
        [StepArg::Operand(first), StepArg::Operand(second)]
            if (*first == weight && *second == other) || (*first == other && *second == weight)
    )
}

/// The row-blocked Q4_K header decode, one call site feature-gated between
/// `q4k_header_for` (the shift-then-branch original) and `q4k_header_for_bf`
/// (`metal-q4k-mask-fma`'s branch-free port, see [`Q4K_MASK_FMA_MSL`]).
/// Split out of [`push_packed_row_blocked_body`]'s `PackedCodec::Q4K` arm so
/// the two `#[cfg]` bodies stay next to each other rather than interleaved
/// with the surrounding match.
#[cfg(not(feature = "metal-q4k-mask-fma"))]
fn push_q4k_header_decode(source: &mut String) {
    source.push_str("            q4k_header hdr = q4k_header_for(blk, slot);\n");
}

#[cfg(feature = "metal-q4k-mask-fma")]
fn push_q4k_header_decode(source: &mut String) {
    source.push_str("            q4k_header hdr = q4k_header_for_bf(blk, slot);\n");
}

/// The Q4_K SCALE-DEFERRED matvec body (`docs/discipline.md` ROW 106):
/// accumulate the raw nibble x activation product and the activation sum
/// UNSCALED across the whole sub-block, then apply `hdr.scale`/`hdr.minimum`
/// ONCE at the end instead of once per element — legal because this
/// function is only reached when `is_plain_product_reduce` has already
/// proved `reduce_op` is `Add` and the body is exactly `weight * other`, so
/// `sum_j (scale*nibble_j - min)*act_j == scale*sum(nibble_j*act_j) -
/// min*sum(act_j)`.
///
/// Two bodies behind `metal-q4k-mask-fma`, same shape as
/// [`push_q4k_header_decode`]:
///
/// Without the feature: `q4k_run8`'s shift-then-mask extraction into a
/// `levels[8]` scratch array, then a `dot`-reduce into `raw_acc`/`act_sum` —
/// mirrors `ggml-metal.metal:5157-5175`'s `acc1`/`dall` split at the ALGEBRA
/// level (defer the scale) but not at the EXTRACTION level (ggml never
/// shifts; see `q4k_run8`'s own corrected doc).
///
/// With the feature: extraction and accumulate are ONE fused loop, masked
/// without any shift, ported from ggml's ACTUAL technique
/// (`ggml-metal.metal:5157-5165`) onto this file's one-nibble-per-byte
/// layout (every element in a lane's 32-element sub-block occupies its own
/// byte, unlike ggml's two-nibbles-per-byte interleave across two
/// sub-blocks — see `q4k_run8`'s doc for why that is a different packing,
/// not a narrower ggml). A `ushort` load of one byte pair yields both
/// nibbles this lane wants at two residual scales — 1x/256x for the low
/// nibble half, 16x/4096x for the high half — inlined directly against
/// `element_type` (not through a shared MSL function typed to `float`) so
/// this stays exactly as generic over `half`/`float` as `q4k_run8`'s own
/// callers are. The residual scale is IDENTICAL for all four `c` iterations
/// a lane makes (`within < 32u` cannot change within one lane's 32-element
/// run, `q4k_run8`'s own doc establishes why), so `q4k_corr` is computed
/// ONCE, outside the loop, and folded into `hdr.scale` at the same combine
/// point the deferred-scale algebra above already uses.
#[cfg(not(feature = "metal-q4k-mask-fma"))]
fn push_q4k_product_reduce_body(source: &mut String, sub: usize, run: usize, element_type: &str) {
    source.push_str(&format!("            {element_type} raw_acc = 0;\n"));
    source.push_str(&format!("            {element_type} act_sum = 0;\n"));
    source.push_str(&format!(
        "            for (int c = 0; c < {}; ++c) {{\n",
        sub / run
    ));
    // raw 4-bit levels (0..15) are exact in float regardless of the
    // kernel's element type; q4k_run8 takes `thread float *out`, narrowed
    // to element_type at the multiply below, same as the per-element path.
    source.push_str(&format!("                float levels[{run}];\n"));
    source.push_str(&format!(
        "                q4k_run8(blk, slot + (uint)(c * {run}), levels);\n"
    ));
    source.push_str("                raw_acc += dot(float4(levels[0], levels[1], levels[2], levels[3]), float4(acts[c * 8 + 0], acts[c * 8 + 1], acts[c * 8 + 2], acts[c * 8 + 3]));\n");
    source.push_str("                raw_acc += dot(float4(levels[4], levels[5], levels[6], levels[7]), float4(acts[c * 8 + 4], acts[c * 8 + 5], acts[c * 8 + 6], acts[c * 8 + 7]));\n");
    source.push_str("                act_sum += acts[c * 8 + 0] + acts[c * 8 + 1] + acts[c * 8 + 2] + acts[c * 8 + 3] + acts[c * 8 + 4] + acts[c * 8 + 5] + acts[c * 8 + 6] + acts[c * 8 + 7];\n");
    source.push_str("            }\n");
    source
        .push_str("            sumf[q] = sumf[q] + hdr.scale * raw_acc - hdr.minimum * act_sum;\n");
}

#[cfg(feature = "metal-q4k-mask-fma")]
fn push_q4k_product_reduce_body(source: &mut String, sub: usize, run: usize, element_type: &str) {
    source.push_str(&format!("            {element_type} raw_acc = 0;\n"));
    source.push_str(&format!("            {element_type} act_sum = 0;\n"));
    source.push_str("            bool q4k_hi = (slot % 64u) >= 32u;\n");
    source.push_str("            ushort q4k_mask_a = q4k_hi ? 0x00F0u : 0x000Fu;\n");
    source.push_str("            ushort q4k_mask_b = q4k_hi ? 0xF000u : 0x0F00u;\n");
    source.push_str("            float q4k_corr = q4k_hi ? (1.0f / 16.0f) : 1.0f;\n");
    source.push_str(&format!(
        "            for (int c = 0; c < {}; ++c) {{\n",
        sub / run
    ));
    source.push_str(&format!(
        "                uint q4k_index = slot + (uint)(c * {run});\n"
    ));
    source.push_str("                uint q4k_group = q4k_index / 64u;\n");
    source.push_str("                uint q4k_within = q4k_index % 64u;\n");
    source.push_str("                uint q4k_byte = q4k_group * 32u + (q4k_within % 32u);\n");
    source.push_str(
        "                device const ushort *q4k_pairs = (device const ushort *)(blk + 16 + q4k_byte);\n",
    );
    source.push_str(&format!(
        "                for (int p = 0; p < {}; ++p) {{\n",
        run / 2
    ));
    source.push_str("                    ushort q4k_word = q4k_pairs[p];\n");
    source.push_str("                    float q4k_level_a = (float)(q4k_word & q4k_mask_a);\n");
    source.push_str(
        "                    float q4k_level_b = (float)(q4k_word & q4k_mask_b) * (1.0f / 256.0f);\n",
    );
    source.push_str(&format!(
        "                    {element_type} act_a = acts[c * {run} + 2 * p];\n"
    ));
    source.push_str(&format!(
        "                    {element_type} act_b = acts[c * {run} + 2 * p + 1];\n"
    ));
    source.push_str(&format!(
        "                    raw_acc += ({element_type})(q4k_level_a * (float)act_a + q4k_level_b * (float)act_b);\n"
    ));
    source.push_str("                    act_sum += act_a + act_b;\n");
    source.push_str("                }\n");
    source.push_str("            }\n");
    source.push_str(
        "            sumf[q] = sumf[q] + hdr.scale * q4k_corr * raw_acc - hdr.minimum * act_sum;\n",
    );
}

/// The row-blocked packed path's tail: combine each simdgroup's per-row
/// `sumf[q]` and write the output. `metal-q4k-split-k`-off arm -- exactly
/// [`push_packed_row_blocked_body`]'s original tail, one simdgroup per
/// row-group, lane 0 writes straight from the SIMD combine.
///
/// Reads `coord_q_cache[q]`, NOT a fresh `flat % / u.output_extents` decode:
/// the preamble above (`push_packed_row_blocked_body`'s own `weight_base`/
/// `other_base` loop) already derives that same coordinate, per `q`, to
/// address the weight/activation operands -- `flat = group_first + q` is
/// identical in both places, so re-running the same division/modulo chain a
/// second time here just to re-derive `coord_q` was paying the ladder-vs-
/// production gap's own named cost twice per thread for zero new
/// information (`docs/bench-campaigns/2026-09-03-gpu-one-risc/design-2026-09-04/kernel-body-diff.md`).
#[cfg(not(feature = "metal-q4k-split-k"))]
#[allow(clippy::too_many_arguments)]
fn push_packed_row_combine_and_write(
    source: &mut String,
    node: NodeId,
    reduce_op: ScalarOp,
    rows: usize,
    rank: usize,
    output_axes: &[u16],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) -> Result<(), EmitError> {
    let combine_fn = simd_combine_fn(node, reduce_op)?;
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        {element_type} reduced = {combine_fn}(sumf[q]);\n"
    ));
    source.push_str("        long flat = group_first + q;\n");
    source.push_str("        if (lane == 0u && flat < u.output_total) {\n");
    source.push_str("            long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "            out_offset += coord_q_cache[q][{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_axes.len(),
        element_type,
        "            ",
        |dim| format!("coord_q_cache[q][{}]", output_axes[dim]),
        "reduced",
        "out_offset",
    );
    source.push_str("        }\n");
    source.push_str("    }\n");
    Ok(())
}

/// The row-blocked packed path's tail: combine each simdgroup's per-row
/// `sumf[q]` and write the output. `metal-q4k-split-k`-on arm -- SPLIT-K
/// COMBINE, option 1 from this landing's brief (multiple simdgroups in ONE
/// threadgroup, threadgroup memory + a barrier, one final fold), over the
/// second-choice atomic-accumulate (the output dtype can be `half`, which
/// Metal has no `atomic<half>` for) and the third-choice second-dispatch
/// partials pass (would double the kernel-launch and uniform-upload cost
/// this landing exists to avoid paying on the STARVED shapes specifically).
///
/// Each simdgroup already SIMD-folds its own interleaved slice of
/// super-blocks (`push_packed_row_blocked_body`'s `ib` loop, strided by
/// `4 * split` when split-K is active); this only combines the (at most
/// [`crate::sized::PACKED_ROW_SPLIT_K_MAX_SPLIT`]) per-simdgroup partials
/// left behind. At `split == 1` (`sgitg` always `0`) this degenerates to
/// exactly the feature-off tail: the loop over `s` never runs, so
/// `total == partial_sums[q][0] == reduced`, written by the SAME thread that
/// computed it -- same value, same order, only the intermediate trip through
/// `threadgroup` memory differs.
#[cfg(feature = "metal-q4k-split-k")]
#[allow(clippy::too_many_arguments)]
fn push_packed_row_combine_and_write(
    source: &mut String,
    node: NodeId,
    reduce_op: ScalarOp,
    rows: usize,
    rank: usize,
    output_axes: &[u16],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
) -> Result<(), EmitError> {
    let combine_fn = simd_combine_fn(node, reduce_op)?;
    let max_split = crate::sized::PACKED_ROW_SPLIT_K_MAX_SPLIT;
    source.push_str(&format!(
        "    threadgroup {element_type} partial_sums[{rows}][{max_split}];\n"
    ));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        {element_type} reduced = {combine_fn}(sumf[q]);\n"
    ));
    source.push_str("        if (lane == 0u) { partial_sums[q][sgitg] = reduced; }\n");
    source.push_str("    }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("    if (sgitg == 0u && lane == 0u) {\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "            {element_type} total = partial_sums[q][0];\n"
    ));
    source.push_str("            for (uint s = 1u; s < split; ++s) {\n");
    let combine_expr = scalar_op_expr(reduce_op, &["total", "partial_sums[q][s]"]);
    source.push_str(&format!("                total = {combine_expr};\n"));
    source.push_str("            }\n");
    source.push_str("            long flat = group_first + q;\n");
    source.push_str("            if (flat < u.output_total) {\n");
    source.push_str("                long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "                out_offset += coord_q_cache[q][{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_axes.len(),
        element_type,
        "                ",
        |dim| format!("coord_q_cache[q][{}]", output_axes[dim]),
        "total",
        "out_offset",
    );
    source.push_str("            }\n");
    source.push_str("        }\n");
    source.push_str("    }\n");
    Ok(())
}

/// [`push_packed_row_blocked_body`]'s `token_total > 1` branch: `s <=
/// crate::sized::PACKED_ROW_ACTIVATION_GROUP` activation ("token") rows
/// folded against ONE streamed weight row per accumulator group --
/// [`grid_threads`]'s own tiling of `token_total` above the cap handles the
/// rest by dispatching more groups, never by truncating `s`. Reuses the
/// fully generic [`operand_read`]/[`push_body_steps`] machinery the plain
/// cooperative/serial reduce paths already use, rather than the
/// `token_total <= 1` branch's hand-tuned per-codec lane-spread -- this
/// path's gate is the s-fold itself (a weight element read from device
/// memory ONCE per `(feature row, reduce-dim element)`, copied into
/// `scratch[weight]` and reused `s` times, never re-read), so every codec
/// and reduce body `packed_row_block` admits is covered by construction.
/// The amortized per-codec header/nibble decode the `token_total <= 1`
/// branch hand-tunes (`q4k_run8`, paired-lane loads) is a follow-up, not
/// folded in here yet -- this body pays one `operand_read` per weight
/// element, same as the generic serial path, just reused across `s` instead
/// of the reduce dim alone.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn push_packed_row_multi_row_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    rank: usize,
    quantized: &[Option<PackedCodec>],
    element_type: &str,
    block: &PackedRowBlock,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    expert_source_mode: bool,
) -> Result<(), EmitError> {
    let weight = block.weight;
    let other = block.other;
    let reduce_dim = block.reduce_dim;
    let token_axes = &block.token_axes;
    let feature_axes = &block.feature_axes;
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    let rows = block.codec.rows_per_simdgroup();
    let cap = crate::sized::PACKED_ROW_ACTIVATION_GROUP as usize;
    let (init_expr, _) = fold_init_tokens(init);
    let identity = cooperative_identity_token(resolved.node, reduce_op)?;
    let combine_fn = simd_combine_fn(resolved.node, reduce_op)?;

    source.push_str("    long feature_total = 1;\n");
    for index in 0..feature_axes.len() {
        source.push_str(&format!(
            "    feature_total *= u.output_extents[{}];\n",
            token_axes.len() + index
        ));
    }
    source.push_str("    long token_total = 1;\n");
    for index in 0..token_axes.len() {
        source.push_str(&format!("    token_total *= u.output_extents[{index}];\n"));
    }
    source.push_str(&format!(
        "    long feature_base = (feature_total + {rows} - 1) / {rows};\n"
    ));
    source.push_str("    long token_group = output_index / feature_base;\n");
    source.push_str("    long feature_group = output_index % feature_base;\n");
    source.push_str(&format!(
        "    long feature_first = feature_group * {rows};\n"
    ));
    source.push_str(&format!("    long token_first = token_group * {cap};\n"));

    source.push_str(&format!("    {element_type} sumf[{cap}][{rows}];\n"));
    source.push_str(&format!("    for (int s = 0; s < {cap}; ++s) {{\n"));
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "            sumf[s][q] = (lane == 0u) ? ({init_expr}) : ({identity});\n"
    ));
    source.push_str("        }\n    }\n");

    source.push_str(&format!("    long weight_base[{rows}];\n"));
    source.push_str(&format!("    long feature_coord[{rows}][{rank_len}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("        long flat = feature_first + q;\n");
    source.push_str(
        "        long remaining = (flat < feature_total) ? flat : (feature_total - 1);\n",
    );
    for dim in 0..rank {
        source.push_str(&format!("        feature_coord[q][{dim}] = 0;\n"));
    }
    for (index_from_end, &dim) in feature_axes.iter().enumerate().rev() {
        let full_index = token_axes.len() + index_from_end;
        source.push_str(&format!(
            "        feature_coord[q][{dim}] = remaining % u.output_extents[{full_index}]; remaining /= u.output_extents[{full_index}];\n"
        ));
    }
    source.push_str(&format!("        long wb = u.operand_base[{weight}];\n"));
    for &dim in feature_axes {
        source.push_str(&format!(
            "        wb += feature_coord[q][{dim}] * u.operand_strides[{weight}][{dim}];\n"
        ));
    }
    source.push_str("        weight_base[q] = wb;\n");
    source.push_str("    }\n");

    source.push_str(&format!("    long other_base[{cap}];\n"));
    source.push_str(&format!("    long token_coord[{cap}][{rank_len}];\n"));
    source.push_str(&format!("    for (int s = 0; s < {cap}; ++s) {{\n"));
    source.push_str("        long flat = token_first + s;\n");
    source.push_str("        long remaining = (flat < token_total) ? flat : (token_total - 1);\n");
    for dim in 0..rank {
        source.push_str(&format!("        token_coord[s][{dim}] = 0;\n"));
    }
    for (index, &dim) in token_axes.iter().enumerate().rev() {
        source.push_str(&format!(
            "        token_coord[s][{dim}] = remaining % u.output_extents[{index}]; remaining /= u.output_extents[{index}];\n"
        ));
    }
    source.push_str(&format!("        long ob = u.operand_base[{other}];\n"));
    for &dim in token_axes {
        source.push_str(&format!(
            "        ob += token_coord[s][{dim}] * u.operand_strides[{other}][{dim}];\n"
        ));
    }
    source.push_str("        other_base[s] = ob;\n");
    source.push_str("    }\n");

    source.push_str(&format!(
        "    long other_stride = u.operand_strides[{other}][{reduce_dim}];\n"
    ));
    // Q4_K FAST PATH (`docs/discipline.md` ROW 389): the weight block's
    // header/nibble decode is shared across the whole `cap`-token activation
    // group via `q4k_pair_dot_mr` (this file's own multi-row generalization
    // of the M=1 decode path's `q4k_pair_dot`), instead of the per-element
    // `operand_read` below paying that decode once per token. Every other
    // codec (`Q3_K`/`Q5_K`/`Q6_K`) keeps the generic loop -- they have no
    // multi-row port yet, this landing only proves the pattern on `Q4_K`,
    // the codec `ROW 389`'s own trace named as the dominant contributor.
    let fast_q4k = !expert_source_mode
        && block.codec == PackedCodec::Q4K
        && element_type == "float"
        && quantized[weight] == Some(PackedCodec::Q4K)
        && quantized[other].is_none()
        && is_plain_product_reduce(resolved, reduce_op, weight, other);
    if fast_q4k {
        push_packed_row_multi_row_q4k_body(
            source,
            weight,
            other,
            rows,
            cap,
            block.codec.block_bytes(),
        );
    } else {
        source.push_str("    for (long k = (long)lane; k < u.reduction_total; k += 32L) {\n");
        source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str(&format!(
            "            {element_type} scratch[{}];\n",
            operand_count.max(1)
        ));
        source.push_str(&format!(
            "            scratch[{weight}] = {};\n",
            operand_read(weight, "(weight_base[q] + k)", quantized[weight])
        ));
        source.push_str(&format!("            for (int s = 0; s < {cap}; ++s) {{\n"));
        source.push_str(&format!(
            "                scratch[{other}] = {};\n",
            operand_read(
                other,
                "(other_base[s] + k * other_stride)",
                quantized[other]
            )
        ));
        let value_expr = push_body_steps(
            source,
            resolved.element_body(),
            "                ",
            element_type,
        );
        source.push_str(&format!(
            "                {element_type} value = {value_expr};\n"
        ));
        let combine_expr = scalar_op_expr(reduce_op, &["sumf[s][q]", "value"]);
        source.push_str(&format!("                sumf[s][q] = {combine_expr};\n"));
        source.push_str("            }\n");
        source.push_str("        }\n");
        source.push_str("    }\n");
    }

    source.push_str(&format!("    for (int s = 0; s < {cap}; ++s) {{\n"));
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "            {element_type} reduced = {combine_fn}(sumf[s][q]);\n"
    ));
    source.push_str("            if (lane == 0u) {\n");
    source.push_str("                long token_flat = token_first + s;\n");
    source.push_str("                long feature_flat = feature_first + q;\n");
    source.push_str(
        "                if (token_flat < token_total && feature_flat < feature_total) {\n",
    );
    source.push_str("                    long out_offset = u.out_base;\n");
    for &dim in feature_axes {
        source.push_str(&format!(
            "                    out_offset += feature_coord[q][{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    for &dim in token_axes {
        source.push_str(&format!(
            "                    out_offset += token_coord[s][{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    let output_rank = token_axes.len() + feature_axes.len();
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "                    ",
        |dim| {
            if dim < token_axes.len() {
                format!("token_coord[s][{}]", token_axes[dim])
            } else {
                format!("feature_coord[q][{}]", feature_axes[dim - token_axes.len()])
            }
        },
        "reduced",
        "out_offset",
    );
    source.push_str("                }\n");
    source.push_str("            }\n");
    source.push_str("        }\n");
    source.push_str("    }\n");
    Ok(())
}

/// Renders [`push_packed_row_multi_row_body`]'s `Q4_K` fast-path reduction
/// loop: the SAME lane split (`ix`/`it`/`iq`/`ir`, `lanes_per_block = 8`) and
/// row-pointer hoist `push_q4k_ggml_port_body`/the M=1 `plain_product` arm
/// already use, generalized to a `cap`-token activation group. Each weight
/// block's header/nibble words are read ONCE per `(ib, q)` inside
/// `q4k_pair_dot_mr` and folded against every one of the `cap` activation
/// rows -- but the activation gather itself happens ONE token at a time,
/// inside `q4k_pair_dot_mr`'s own `s` loop, off `other_base`/`other_stride`/
/// `other_ib_offset` (all scalars/a `cap`-long array of `long`s, not floats)
/// rather than this function pre-gathering `yl_group[cap][16]`/
/// `yh_group[cap][16]` and handing those arrays in. See `q4k_pair_dot_mr`'s
/// own doc for why that pre-gather (256 live private floats at `cap = 8`)
/// measured SLOWER than the generic body this fast path replaced.
fn push_packed_row_multi_row_q4k_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    cap: usize,
    block_bytes: usize,
) {
    source.push_str("    uint ix = (uint)lane / 8u;\n");
    source.push_str("    uint it = (uint)lane % 8u;\n");
    source.push_str("    uint iq = it / 4u;\n    uint ir = it % 4u;\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 4;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {block_bytes};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    source.push_str(&format!(
        "    long y4_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
    ));
    source.push_str(&format!(
        "    long other_ib_offset = (long)ib_first * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
    ));
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!("            float result[{cap}];\n"));
    source.push_str(&format!(
        "            q4k_pair_dot_mr(blk_ptr[q], iq, ir, in{other}, other_base, other_stride, other_ib_offset, {cap}u, result);\n"
    ));
    source.push_str(&format!(
        "            for (int s = 0; s < {cap}; ++s) {{ sumf[s][q] = sumf[s][q] + result[s]; }}\n"
    ));
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        other_ib_offset += y4_step;\n");
    source.push_str("    }\n");
}

/// The sole output axis whose bound extent is `> 1`, when every OTHER
/// output axis is degenerate (extent `1`) -- the decode-matvec shape
/// `docs/discipline.md` ROW 350 measured (`output_extents = [1, rows]`).
/// `None` when zero or more than one output axis is non-unit, so the
/// generic N-D coordinate decomposition
/// [`push_packed_row_group_bases`]'s `else` arm renders stays correct for
/// every shape this does not apply to (batched decode, multi-row `M`
/// blocks routed through [`push_packed_row_multi_row_body`] instead, etc).
fn packed_row_direct_output_axis(resolved: &BoundOp, output_axes: &[u16]) -> Option<u16> {
    let mut found = None;
    for &axis in output_axes {
        if resolved.extents[axis as usize] > 1 {
            if found.is_some() {
                return None;
            }
            found = Some(axis);
        }
    }
    found
}

/// Renders `weight_base[q]`/`other_base[q]`/`coord_q_cache[q]` for one
/// row-blocked group -- the ONE place every [`push_packed_row_blocked_body`]
/// body variant (the default/mask-fma arm, [`push_q4k_ggml_port_body`],
/// [`push_q6k_ggml_port_body`]) gets its row addressing from, since all
/// three run this preamble before branching on which reduce body to emit.
/// ROW 350 named the generic `%`/`/` decomposition against
/// `u.output_extents`/`u.operand_strides` as 16 division-class instructions
/// per thread that are a provable no-op whenever [`packed_row_direct_output_axis`]
/// finds exactly one non-unit output axis (the decode matvec, `s=1`) --
/// this is the SAME emit-time specialization pattern the `other_stride_is_one`
/// stride-free arm above already applies, done here for the row-base
/// addresses instead of the activation stride.
fn push_packed_row_group_bases(
    source: &mut String,
    resolved: &BoundOp,
    output_axes: &[u16],
    rank: usize,
    rows: usize,
    weight: usize,
    other: usize,
) {
    let gather_slots = gather_slots(resolved);
    if let Some(axis) = packed_row_direct_output_axis(resolved, output_axes) {
        source.push_str(&format!(
            "    long weight_row_stride = u.operand_strides[{weight}][{axis}];\n"
        ));
        source.push_str(&format!(
            "    long other_row_stride = u.operand_strides[{other}][{axis}];\n"
        ));
        source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str("        long flat = group_first + q;\n");
        source.push_str(&format!(
            "        for (int d = 0; d < {rank}; ++d) {{ coord_q_cache[q][d] = 0; }}\n"
        ));
        source.push_str(&format!("        coord_q_cache[q][{axis}] = flat;\n"));
        source.push_str(&format!(
            "        weight_base[q] = u.operand_base[{weight}] + flat * weight_row_stride;\n"
        ));
        source.push_str(&format!(
            "        other_base[q] = u.operand_base[{other}] + flat * other_row_stride;\n"
        ));
        if let Some(slot) = gather_slots[weight] {
            push_cooperative_gather_fetch(
                source,
                weight,
                slot,
                rank,
                "coord_q_cache[q]",
                &format!("weight_base[q]"),
            );
        }
        if let Some(slot) = gather_slots[other] {
            push_cooperative_gather_fetch(
                source,
                other,
                slot,
                rank,
                "coord_q_cache[q]",
                &format!("other_base[q]"),
            );
        }
        source.push_str("    }\n");
        return;
    }
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("        long flat = group_first + q;\n");
    source.push_str("        long remaining_q = flat;\n");
    source.push_str(&format!(
        "        for (int d = 0; d < {rank}; ++d) {{ coord_q_cache[q][d] = 0; }}\n"
    ));
    for (index, dim) in output_axes.iter().enumerate().rev() {
        source.push_str(&format!(
            "        coord_q_cache[q][{dim}] = remaining_q % u.output_extents[{index}]; remaining_q /= u.output_extents[{index}];\n"
        ));
    }
    source.push_str(&format!("        long wb = u.operand_base[{weight}];\n"));
    source.push_str(&format!("        long ob = u.operand_base[{other}];\n"));
    // Iterate the OUTPUT axes directly rather than `0..rank` minus one
    // excluded dim: `reduce_dim` is now the innermost of possibly SEVERAL
    // folded reduce dims (see `classify_packed_row_block`'s contiguous-fold
    // check), so `output_axes` -- already the exact complement of every
    // reduce dim, however many there are -- is the correct and simpler set
    // to walk here regardless of reduce rank.
    for &dim in output_axes {
        source.push_str(&format!(
            "        wb += coord_q_cache[q][{dim}] * u.operand_strides[{weight}][{dim}];\n"
        ));
        source.push_str(&format!(
            "        ob += coord_q_cache[q][{dim}] * u.operand_strides[{other}][{dim}];\n"
        ));
    }
    source.push_str("        weight_base[q] = wb;\n");
    source.push_str("        other_base[q] = ob;\n");
    if let Some(slot) = gather_slots[weight] {
        push_cooperative_gather_fetch(
            source,
            weight,
            slot,
            rank,
            "coord_q_cache[q]",
            &format!("weight_base[q]"),
        );
    }
    if let Some(slot) = gather_slots[other] {
        push_cooperative_gather_fetch(
            source,
            other,
            slot,
            rank,
            "coord_q_cache[q]",
            &format!("other_base[q]"),
        );
    }
    source.push_str("    }\n");
}

/// Every source marker a packed-row-blocked body arm below can emit, owned
/// here next to the bodies that emit them so a new arm's marker is added in
/// ONE place. [`crate::metal::classify_kind`] (the profiler's own
/// classifier, source-greps this same emitted text over in `metal.rs`)
/// consults this list instead of restating its own copy -- the restated copy
/// is exactly what went stale: it carried `q4k_pair_dot(blk`/`q4k_run8(blk`/
/// `q5k_pair_dot(blk`/`q5k_value(blk`/`q6k_value(blk`/`acc1_0` but missed
/// `q3k_pair_dot(blk` (this file, the `Q3K if plain_product` arm above),
/// `q3k_element(blk` (the `Q3K` per-element fallback), and `q6k_pair_dot(blk`
/// (the `Q6K if plain_product` arm) -- every Q3_K row-blocked dispatch and
/// every plain-product Q6_K dispatch (the shape the openchat output head
/// actually takes) undercounted into `"reduce-cooperative"`. A fourth gap
/// found by the lowering census test this same landing adds: `metal`'s own
/// feature list turns `metal-q4k-ggml-port` ON by default (`Cargo.toml`'s
/// `metal = [.., "metal-q4k-ggml-port"]`), and [`push_q6k_ggml_port_body`]
/// -- unlike its `Q4_K`/`Q5_K` siblings, which share `acc1_0` -- calls none
/// of the named helpers and has no `acc1_0` accumulator either, so it fell
/// through this same list even after the first three additions. `"sums0"`
/// closes it (unique to that function's own per-thread partial sums).
// sole caller is `crate::metal::classify_kind`, gated on `metal` (macOS-only
// driver) AND `instrument` (diagnostic-only) -- this crate's `msl` module
// itself only needs `alloc`, so a build with neither compiles this table
// with nothing left to call it.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos", feature = "instrument")),
    allow(
        dead_code,
        reason = "sole caller is the macOS-only, instrument-gated profiler"
    )
)]
pub(crate) const PACKED_ROW_BODY_MARKERS: &[&str] = &[
    "q3k_pair_dot(blk",
    "q3k_element(blk",
    "q4k_pair_dot(blk",
    // `push_packed_row_multi_row_body`'s own Q4_K-specific staged decode
    // (landed on main while this fix was in flight, found again at rebase
    // time by this same list) -- same gap, same fix, one more marker.
    "q4k_pair_dot_mr(blk",
    "q4k_run8(blk",
    "q5k_pair_dot(blk",
    "q5k_value(blk",
    "q6k_pair_dot(blk",
    "q6k_value(blk",
    // unique to `push_q4k_ggml_port_body`/`push_q5k_ggml_port_body`'s
    // per-thread accumulator naming -- those bodies call none of the
    // helpers above by name, that is the whole point of the ggml port.
    "acc1_0",
    // unique to `push_q6k_ggml_port_body`'s own per-thread partial sums --
    // that body shares neither `acc1_0` nor any named helper call with its
    // Q4_K/Q5_K ggml-port siblings.
    "sums0",
];

#[allow(clippy::too_many_arguments)]
fn push_packed_row_blocked_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    rank: usize,
    quantized: &[Option<PackedCodec>],
    element_type: &str,
    block: &PackedRowBlock,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    expert_source_mode: bool,
) -> Result<(), EmitError> {
    if packed_row_block_token_total(block, &resolved.extents) > 1 {
        push_packed_row_multi_row_body(
            source,
            resolved,
            reduce_op,
            init,
            rank,
            quantized,
            element_type,
            block,
            epilogue_body,
            epilogue_operands,
            expert_source_mode,
        )?;
        return Ok(());
    }
    let PackedRowBlock {
        weight,
        other,
        reduce_dim,
        codec,
        ..
    } = *block;
    let block_bytes = codec.block_bytes();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    // seeded on lane 0 only, exactly as the general cooperative path does:
    // the true seed folds in once and every other lane starts at the
    // algebraic identity, so `simd_*` can combine them unconditionally.
    let (init_expr, _) = fold_init_tokens(init);
    let identity = cooperative_identity_token(resolved.node, reduce_op)?;
    // ROW-BLOCKED PACKED PATH. One SIMD group folds `codec.rows_per_
    // simdgroup()` output rows at once so the activation's run of 8 values
    // is loaded into registers ONCE and reused across all of them — ggml's
    // `float sumf[nr0]` (`N_R0_Q4_K 4`, `N_R0_Q6_K 1`). Combined with the
    // super-block header amortization below, the per-element cost becomes
    // one byte load, one mask, one fma (`docs/discipline.md` ROW 74).
    {
        let run = Q4K_BLOCK_ELEMENTS / SIMD_WIDTH as usize;
        let rows = codec.rows_per_simdgroup();
        source.push_str(&format!("    long group_first = output_index * {rows};\n"));
        source.push_str(&format!("    {element_type} sumf[{rows}];\n"));
        // when the seed and the algebraic identity are the textually same
        // token (true for every op this path actually reaches: Add/Zero,
        // Multiply/One, Maximum/NegativeInfinity, Minimum/PositiveInfinity),
        // every lane's `sumf[q]` starts at the same value regardless of
        // which lane it is -- the per-lane ternary is dead, a select over
        // two identical constants, so emit a plain assignment instead.
        if init_expr == identity {
            source.push_str(&format!(
                "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = {identity}; }}\n"
            ));
        } else if cfg!(feature = "metal-q4k-split-k") {
            // the true seed folds in exactly ONCE across the WHOLE
            // threadgroup, not once per simdgroup -- at split == 1 `sgitg`
            // is always `0`, so this collapses to the feature-off condition
            // (`lane == 0u`) exactly.
            source.push_str(&format!(
                "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = (lane == 0u && sgitg == 0u) ? ({init_expr}) : ({identity}); }}\n"
            ));
        } else {
            source.push_str(&format!(
                "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = (lane == 0u) ? ({init_expr}) : ({identity}); }}\n"
            ));
        }
        source.push_str(&format!("    long weight_base[{rows}];\n"));
        source.push_str(&format!("    long other_base[{rows}];\n"));
        // `coord_q_cache[q]` -- kept past this loop, not a scratch local --
        // is the SAME `flat = group_first + q` decode
        // `push_packed_row_combine_and_write`'s epilogue needs for the
        // output-write coordinate; caching it here instead of re-running the
        // `%`/`/` chain against `u.output_extents` a second time in the
        // epilogue is the fix for the ladder-vs-production kernel-body gap
        // this op's own diff found (`docs/bench-campaigns/2026-09-03-gpu-one-
        // risc/design-2026-09-04/kernel-body-diff.md`): the ladder's hand-
        // written dispatch never had a second uniform-driven coordinate
        // decode to pay, because it never had a first one either.
        source.push_str(&format!("    long coord_q_cache[{rows}][{rank_len}];\n"));
        push_packed_row_group_bases(source, resolved, output_axes, rank, rows, weight, other);
        // STRIDE-FREE SPECIALIZATION (`docs/discipline.md` perf/packed-row-
        // addressing row): ggml's own row-blocked kernel assumes a
        // contiguous activation and addresses it with pure element offsets
        // (`ggml-metal.metal:5132`'s `y4 += 4*QK_K`, no per-element stride
        // multiply anywhere). This op's activation is not always
        // contiguous along the reduce axis, but `resolved`'s `Layout`
        // already carries the answer here at EMIT time -- this function
        // renders one op's kernel text once, not once per dispatch -- so
        // every activation address below drops the runtime `other_stride`
        // multiply entirely in the source text whenever the layout proves
        // it would multiply by 1.
        let other_stride_is_one = resolved.operands()[other].1.stride(reduce_dim as u16) == 1;
        source.push_str(&format!(
            "    long other_stride = u.operand_strides[{other}][{reduce_dim}];\n"
        ));
        // LANE SPREAD, ggml's `ix = tiisg/8`. Putting all 32 lanes on ONE
        // super-block gives each lane 8 of its 256 elements, so the header
        // decode is amortized over 8. Putting EIGHT lanes on a super-block
        // and letting the 32 lanes span FOUR at once gives each lane a whole
        // 32-element sub-block per decode — 4x the amortization, and the
        // sub-block is exactly the granularity the header is constant over.
        //
        // `it` selects the sub-block, so `slot = it * 32` and every one of
        // that lane's 32 elements shares a group and a nibble half. Levels
        // are still pulled 8 at a time (`q4k_run8`) rather than 32, to keep
        // the live register count near ggml's `yl[16]+yh[16]+sumf[4]`.
        // eight lanes per super-block (ggml's `tiisg/8`), so the 32 lanes of
        // a SIMD group span four super-blocks and each lane owns exactly one
        // 32-element sub-block — the granularity the header is constant over.
        let lanes_per_block = 8;
        let sub = Q4K_BLOCK_ELEMENTS / lanes_per_block;
        // Structural, not feature-gated: the paired-nibble/paired-lane body
        // applies to any codec whose block layout has one
        // ([`PackedCodec::supports_pair_dot`]) when the dtype is real
        // `Float32` -- a `DType` match, not the `element_type == "float"`
        // MSL-type-token comparison this replaced, which also admitted
        // `Int32`/`UInt32`/`Bool`/`Int8`/`UInt8` (every dtype `type_token`
        // happens to lower to the same MSL `float` storage type) and would
        // have run the scale/minimum float algebra below on integer data.
        let plain_product = !expert_source_mode
            && codec.supports_pair_dot()
            && resolved.dtype == DType::Float32
            && is_plain_product_reduce(resolved, reduce_op, weight, other);
        // `metal-q4k-single-fetch` (default-off): eliminates the redundant
        // paired-lane load the default `Q4_K` lane assignment below makes --
        // see `push_q4k_single_fetch_body`'s own doc. Checked after
        // `plain_product` (`q4k_pair_dot`'s float-only arm keeps priority --
        // neither is a self-contained kernel body the way this one is, so
        // `q4k_pair_dot` stays the fastest-known path where it applies) and
        // takes priority over everything else below it (the scale-deferred/
        // mask-fma arm, the per-element fallback): `push_q4k_single_fetch_body`
        // is fully self-contained (its own dispatch loop, not routed through
        // the lane-spread preamble this `else` arm builds), so it replaces
        // that whole preamble+match rather than plugging into one arm of it.
        // ALSO requires `metal-q4k-split-k` off: `emit`'s own dispatch-geometry
        // setup (`kernel_dispatch_shape`) computes `sgitg`/`split` purely off
        // that feature flag, unconditionally, for every row-blocked op --
        // `push_q4k_single_fetch_body`'s `ib` loop has no `sgitg`/`split`
        // awareness of its own (measured: under `--all-features` its sum came
        // out ~7x too large, consistent with every one of `split` simdgroups
        // redundantly summing the SAME full reduction instead of a disjoint
        // 1/split slice). Correct fix, not a silent one: `plain_product`/the
        // default/mask-fma `else` arm are already split-K-aware (they read
        // `sgitg`/`split` when the feature is on), so gating single-fetch off
        // here just means split-K wins whenever BOTH features are compiled
        // in, same "not invented to compose" posture as its other arms.
        let use_single_fetch = matches!(codec, PackedCodec::Q4K)
            && !plain_product
            && cfg!(feature = "metal-q4k-single-fetch")
            && !cfg!(feature = "metal-q4k-split-k");
        // `metal-q4k-ggml-port` (default-off): the verbatim ggml transcription,
        // see [`push_q4k_ggml_port_body`]/[`push_q6k_ggml_port_body`]'s own
        // docs. Requires `plain_product` (float-only scale-deferred shape,
        // the same gate `q4k_pair_dot`'s own arm below needs) and takes
        // priority over it -- both target the exact same design point, and
        // when this feature is on it is the one under test. Not
        // `metal-q4k-split-k`-aware, same posture as `push_q4k_single_fetch_body`
        // above and for the identical reason: its `ib` loop has no
        // `sgitg`/`split` stride of its own. ALSO requires the codec be one
        // this file has a verbatim ggml body for (`Q4K`, `Q5K`, `Q6K`): each
        // of `push_q4k_ggml_port_body`/`push_q5k_ggml_port_body`/
        // `push_q6k_ggml_port_body` transcribes a SPECIFIC ggml kernel's own
        // fixed byte offsets and decode shape. `plain_product` alone is not
        // codec-specific (every codec with `supports_pair_dot` reaches this
        // arm) -- without this guard, enabling `metal-q4k-ggml-port` silently
        // routed every packed-row `Q5_K` matmul through the `Q4_K`-shaped
        // body too (found via
        // `metal_matmul_on_packed_q5k_weights_matches_the_dequantized_f32_cpu_path`,
        // relative=0.977, essentially uncorrelated output -- the qh plane was
        // simply never read). `Q5_K` now has its own body
        // ([`push_q5k_ggml_port_body`]) instead of falling through to `Q4_K`'s.
        let use_ggml_port = plain_product
            && matches!(
                codec,
                PackedCodec::Q4K | PackedCodec::Q5K | PackedCodec::Q6K
            )
            && cfg!(feature = "metal-q4k-ggml-port")
            && !cfg!(feature = "metal-q4k-split-k");
        if use_ggml_port && matches!(codec, PackedCodec::Q6K) {
            push_q6k_ggml_port_body(
                source,
                weight,
                other,
                rows,
                block_bytes,
                other_stride_is_one,
            );
        } else if use_ggml_port && matches!(codec, PackedCodec::Q5K) {
            push_q5k_ggml_port_body(
                source,
                weight,
                other,
                rows,
                block_bytes,
                other_stride_is_one,
            );
        } else if use_ggml_port {
            push_q4k_ggml_port_body(
                source,
                weight,
                other,
                rows,
                block_bytes,
                other_stride_is_one,
            );
        } else if use_single_fetch {
            push_q4k_single_fetch_body(
                source,
                resolved,
                reduce_op,
                weight,
                other,
                element_type,
                operand_count,
                rows,
                block_bytes,
            );
        } else {
            source.push_str(&format!("    uint ix = (uint)lane / {lanes_per_block}u;\n"));
            source.push_str(&format!("    uint it = (uint)lane % {lanes_per_block}u;\n"));
            source.push_str(&format!("    uint slot = it * {sub}u;\n"));
            if plain_product {
                source.push_str(
                    "    uint iq = it / 4u; uint ir = it % 4u;\n    float yl[16]; float yh[16];\n",
                );
            } else {
                source.push_str(&format!("    {element_type} acts[{sub}];\n"));
            }
            source.push_str(&format!(
                "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
            ));
            let ix_stride = SIMD_WIDTH as usize / lanes_per_block;
            if cfg!(feature = "metal-q4k-split-k") {
                // each simdgroup (`sgitg`, 0 at split == 1) owns a disjoint
                // interleaved slice of super-blocks -- a plain strided loop, so a
                // `super_blocks` not evenly divisible by `split` is handled by
                // construction (some simdgroups simply run one fewer iteration),
                // never a separate ragged-tail branch.
                source.push_str(&format!(
                "    int ib_first = (int)(ix + sgitg * {ix_stride}u);\n    int ib_step = (int)({ix_stride}u * split);\n"
            ));
            } else {
                source.push_str(&format!(
                    "    int ib_first = (int)ix;\n    int ib_step = {ix_stride};\n"
                ));
            }
            // HOIST + POINTER INCREMENT (`docs/discipline.md` perf/packed-row-
            // addressing row): `weight_base[q]/Q4K_BLOCK_ELEMENTS` and the y4
            // lane offset (`64*iq + 8*ir`) are invariant across every `ib` this
            // thread visits -- only `ib` itself varies. The prior form
            // recomputed `(weight_base[q]/256 + ib) * block_bytes` and
            // `ib*256*other_stride` from scratch every iteration (a 64-bit
            // multiply-add per row per iteration); ggml's own row/`y4` pointers
            // instead advance by a CONSTANT per iteration
            // (`ggml-metal.metal:5132,5182`'s `q1 += nb01/2`, `y4 += 4*QK_K`).
            // This computes each row's starting byte pointer and the
            // per-iteration byte step ONCE before the loop, then the loop body
            // only adds.
            source.push_str(&format!(
                "    long blk_step = (long)ib_step * {block_bytes};\n"
            ));
            source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
            source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
            source.push_str(&format!(
            "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
        ));
            source.push_str("    }\n");
            if plain_product {
                push_q4k_plain_product_y4_address(source, other, other_stride_is_one);
            } else if other_stride_is_one {
                // SAME HOIST, generic (non-plain-product) arm: `elem0 = ib*256 +
                // slot` was rebuilt every `ib` purely to feed `(elem0+j)*
                // other_stride` -- the identical 64-bit multiply-add-per-
                // iteration shape arm1 already removed from the plain-product
                // `y4` pointer above. `other_stride_is_one` additionally drops
                // the multiply itself, same as the `y4` arm just above.
                source.push_str(&format!(
                    "    long acts_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS};\n"
                ));
                source.push_str(&format!("    device const {element_type} *acts_row = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} + (long)slot;\n"));
            } else {
                source.push_str(&format!(
                    "    long acts_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
                ));
                source.push_str(&format!("    device const {element_type} *acts_row = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} * other_stride + (long)slot * other_stride;\n"));
            }
            source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
            if plain_product && other_stride_is_one {
                source.push_str("        for (uint i = 0u; i < 8u; ++i) { yl[i] = y4[i]; yl[i + 8u] = y4[i + 32u]; yh[i] = y4[i + 128u]; yh[i + 8u] = y4[i + 160u]; }\n");
            } else if plain_product {
                // CORRECTNESS FIX, not part of the stride-free specialization
                // above: this arm's `y4[i]`/`y4[i+32]`/... reads were pure
                // element offsets regardless of `other_stride` before this
                // landing -- correct only by accident, for every caller that
                // happened to hand this path a contiguous activation. Ported
                // from `push_q4k_ggml_port_body`'s own already-stride-aware
                // form (`y4_base + (long)i * other_stride`, below in this
                // file), the one sibling body that already got this right.
                source.push_str("        for (uint i = 0u; i < 8u; ++i) { yl[i] = y4[(long)i * other_stride]; yl[i + 8u] = y4[(long)(i + 32u) * other_stride]; yh[i] = y4[(long)(i + 128u) * other_stride]; yh[i + 8u] = y4[(long)(i + 160u) * other_stride]; }\n");
            } else {
                source.push_str(&format!("        for (int j = 0; j < {sub}; ++j) {{\n"));
                if other_stride_is_one {
                    source.push_str("            acts[j] = acts_row[j];\n");
                } else {
                    source.push_str("            acts[j] = acts_row[(long)j * other_stride];\n");
                }
                source.push_str("        }\n");
            }
            source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
            source.push_str("            device const uchar *blk = blk_ptr[q];\n");
            match codec {
                PackedCodec::Q2K => {
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q2k_element(blk, slot + (uint)e);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                PackedCodec::Q3K if plain_product => {
                    // `plain_product` is codec-agnostic (the `yl`/`yh` gather
                    // above is built once, shared by `Q4_K`/`Q3_K` and, when
                    // `metal-q5k-pair-dot` is on, `Q5_K` too) -- no separate
                    // activation load path needed here, same posture as
                    // `Q5_K`'s own `plain_product` arm.
                    source.push_str(
                        "            sumf[q] = sumf[q] + q3k_pair_dot(blk, iq, ir, yl, yh);\n",
                    );
                }
                PackedCodec::Q3K => {
                    // `Q3_K`'s sub-block width (16) is narrower than this
                    // loop's 32-element `sub` slot, unlike `Q5_K`'s matching
                    // 32-element sub-block -- amortizing one header decode
                    // across the whole slot the way the `Q5_K` arm below does
                    // would silently span two different sub-block scales. Each
                    // call to `q3k_element` decodes its own header, the same
                    // posture `Q6_K`'s per-element path takes for a different
                    // reason (its scale bytes are plain, not bit-packed, so the
                    // per-call cost is small either way). A follow-up
                    // optimization (a two-headers-per-slot amortization), not a
                    // correctness gap.
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q3k_element(blk, slot + (uint)e);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                PackedCodec::Q4K => {
                    if plain_product {
                        source.push_str(
                            "            sumf[q] = sumf[q] + q4k_pair_dot(blk, iq, ir, yl, yh);\n",
                        );
                    } else if is_plain_product_reduce(resolved, reduce_op, weight, other) {
                        // SCALE-DEFERRED PATH (`docs/discipline.md` ROW 106).
                        // Accumulate the raw nibble x activation product and the
                        // activation sum UNSCALED across the whole sub-block, then
                        // apply `hdr.scale`/`hdr.minimum` ONCE at the end instead
                        // of once per element — legal here because
                        // `is_plain_product_reduce` already proved reduce_op is
                        // `Add` and the body is exactly `weight * other`, so
                        // `sum_j (scale*nibble_j - min)*act_j == scale*sum(nibble_j
                        // *act_j) - min*sum(act_j)`. Mirrors
                        // `ggml-metal.metal:5157-5175`'s `acc1`/`dall` split.
                        // Two bodies behind `metal-q4k-mask-fma`: off, the
                        // shift-then-mask `q4k_run8` extraction into a `dot`
                        // reduce; on, ggml's actual mask-without-shift technique,
                        // fused with the accumulate -- see
                        // `push_q4k_product_reduce_body`'s own doc.
                        push_q4k_header_decode(source);
                        push_q4k_product_reduce_body(source, sub, run, element_type);
                    } else {
                        source.push_str(&format!(
                            "            for (int c = 0; c < {}; ++c) {{\n",
                            sub / run
                        ));
                        // raw 4-bit levels (0..15) are exact in float regardless of
                        // the kernel's element type; q4k_run8 takes `thread float
                        // *out`, and the narrowing to element_type happens where
                        // levels combine into scratch below, same as every other
                        // operand read.
                        source.push_str(&format!("                float levels[{run}];\n"));
                        source.push_str(&format!(
                            "                q4k_run8(blk, slot + (uint)(c * {run}), levels);\n"
                        ));
                        source.push_str(&format!(
                            "                for (int j = 0; j < {run}; ++j) {{\n"
                        ));
                        source.push_str(&format!(
                            "                    {element_type} scratch[{}];\n",
                            operand_count.max(1)
                        ));
                        source.push_str(&format!(
                        "                    scratch[{weight}] = hdr.scale * levels[j] - hdr.minimum;\n"
                    ));
                        source.push_str(&format!(
                            "                    scratch[{other}] = acts[c * {run} + j];\n"
                        ));
                        let value_expr = push_body_steps(
                            source,
                            resolved.element_body(),
                            "                    ",
                            element_type,
                        );
                        source.push_str(&format!(
                            "                    {element_type} value = {value_expr};\n"
                        ));
                        let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                        source
                            .push_str(&format!("                    sumf[q] = {combine_expr};\n"));
                        source.push_str("                }\n");
                        source.push_str("            }\n");
                    }
                }
                PackedCodec::Q5K if plain_product => {
                    // See [`Q5K_PAIR_DOT_MSL`]: the same `yl`/`yh` two-word-load pairing `Q4_K`'s own
                    // `plain_product` arm above uses, extended with `Q5_K`'s `qh`
                    // high-bit plane. Reads the SAME `yl`/`yh` activation gather
                    // this preamble already built for `Q4_K` (`plain_product`
                    // is codec-agnostic there), so no separate activation load
                    // path is needed for this codec.
                    source.push_str(
                        "            sumf[q] = sumf[q] + q5k_pair_dot(blk, iq, ir, yl, yh);\n",
                    );
                }
                PackedCodec::Q5K => {
                    // No `q5k_run8`-style batched unpack yet — `Q5_K`'s `qh`
                    // high-bit plane means each element needs a `qs` nibble AND
                    // a `qh` bit from a DIFFERENT byte, the same shape gap
                    // `Q6_K`'s own arm below documents. `d` and this sub-block's
                    // scale/min/mask ARE decoded once per 32-element run via
                    // `q5k_header_for` (the same granularity `q4k_header_for`
                    // amortizes over) — a follow-up optimization, not a
                    // correctness gap; see this landing's discipline row (ROW
                    // 92) for the measured cost of skipping it. The `plain_product`
                    // arm above replaces this whole per-element loop with the
                    // paired-nibble body whenever the reduce is a plain product.
                    source.push_str("            q5k_header hdr = q5k_header_for(blk, slot);\n");
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q5k_value(blk, slot + (uint)e, hdr);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                PackedCodec::Q6K if plain_product => {
                    // See [`Q6K_PAIR_DOT_MSL`]: the same paired-lane body `Q4_K`/`Q5_K`'s own
                    // `plain_product` arms use above, ported to `Q6_K`'s
                    // ql/qh/signed-scale layout. Reads the SAME `yl`/`yh`
                    // activation gather this preamble already built for
                    // `Q4_K` (`plain_product` is codec-agnostic there), so no
                    // separate activation load path is needed for this codec.
                    source.push_str(
                        "            sumf[q] = sumf[q] + q6k_pair_dot(blk, iq, ir, yl, yh);\n",
                    );
                }
                PackedCodec::Q6K => {
                    // No `q6k_run8`-style batched unpack yet — `Q6_K`'s bit
                    // layout does not reduce to two word loads the way `Q4_K`'s
                    // does (each element needs a `ql` byte, a `qh` byte, AND a
                    // sub-block scale byte, not one nibble out of an
                    // already-loaded word). Correct, one element at a time; `d`
                    // is still decoded ONCE per super-block via
                    // `q6k_header_for` rather than per element. The `plain_product`
                    // arm above replaces this whole per-element loop with the
                    // paired-lane body whenever the reduce is a plain product.
                    source.push_str("            q6k_header hdr = q6k_header_for(blk);\n");
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q6k_value(blk, slot + (uint)e, hdr);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                PackedCodec::Q8_0 => {
                    return Err(EmitError::NonKQuantPackedCodec {
                        node: resolved.node,
                        codec: "q8_0",
                    });
                }
                PackedCodec::Q4_0 => {
                    return Err(EmitError::NonKQuantPackedCodec {
                        node: resolved.node,
                        codec: "q4_0",
                    });
                }
                PackedCodec::Float16 => {
                    return Err(EmitError::NonKQuantPackedCodec {
                        node: resolved.node,
                        codec: "float16",
                    });
                }
                PackedCodec::BFloat16 => {
                    return Err(EmitError::NonKQuantPackedCodec {
                        node: resolved.node,
                        codec: "bfloat16",
                    });
                }
            }
            source.push_str("        }\n");
            source.push_str(&format!(
                "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
            ));
            if plain_product {
                source.push_str("        y4 += y4_step;\n");
            } else {
                source.push_str("        acts_row += acts_step;\n");
            }
            source.push_str("    }\n");
        }
        push_packed_row_combine_and_write(
            source,
            resolved.node,
            reduce_op,
            rows,
            rank,
            output_axes,
            element_type,
            epilogue_body,
            epilogue_operands,
        )?;
    }
    Ok(())
}

/// `metal-q4k-single-fetch` (default-off): the row-blocked packed-`Q4_K`
/// path's `it`/`slot` lane assignment, above, gives lanes `2r` and `2r+1`
/// (`r` = one of the super-block's 4 64-element groups) the SAME 32-byte
/// `qs` range — `q4k_run8` loads it twice, once per lane, differing only in
/// which nibble half each keeps (`shift` 0 vs 4). This function replaces
/// that pairing: `sf_half` (still `it % 2`) now selects a DISTINCT 16-byte
/// half of the group's 32 bytes, and `q4k_run8_dual`
/// extracts BOTH nibble halves from each byte it loads, so the pair's two
/// lanes together read the group's 32 bytes exactly once instead of twice.
///
/// Dispatch geometry is untouched: `ix = lane/8` and the `ib += 4` stride
/// are identical to the duplicate-fetch path, so this is a change to which
/// BYTES a lane owns and what it does with them, not to thread count,
/// simdgroups-per-threadgroup, or `PACKED_ROWS_PER_GROUP`. Not `split-K`
/// aware -- this function owns its own complete dispatch loop rather than
/// plugging into the shared lane-spread preamble `metal-q4k-split-k`
/// modifies, so the two features do not compose (see this feature's own
/// Cargo.toml doc).
///
/// Correctness hazard this function exists to get right: a `qs` byte's low
/// nibble and high nibble belong to DIFFERENT 32-element sub-blocks with
/// DIFFERENT 6-bit `(scale, min)` pairs (`q4_k.rs::dequantize_block`'s own
/// doc — elements land "32 output elements apart", not adjacent). A lane
/// that decodes both nibbles of a byte therefore needs BOTH sub-blocks'
/// headers (`hdr_low`/`hdr_high`), never one. `q4k_header_for(blk,
/// sf_low_base)` and `q4k_header_for(blk, sf_high_base)` resolve to the
/// same two sub-block indices for `sf_half == 0` and `sf_half == 1` alike
/// (`sf_low_base % 64` is `0` or `16`, both `< 32`; `sf_high_base % 64` is
/// `32` or `48`, both `>= 32`), so both this lane's low-half partial sum and
/// its pair-partner's low-half partial sum are scaled by the IDENTICAL
/// `hdr_low`, and summing them via `simd_sum` after the `ib` loop
/// reconstructs the same per-sub-block total the duplicate-fetch path
/// computes — see this function's own algebra note on the scale-deferred
/// arm below.
#[allow(clippy::too_many_arguments)]
fn push_q4k_single_fetch_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    weight: usize,
    other: usize,
    element_type: &str,
    operand_count: usize,
    rows: usize,
    block_bytes: usize,
) {
    source.push_str("    uint ix = (uint)lane / 8u;\n");
    source.push_str("    uint it = (uint)lane % 8u;\n");
    source.push_str("    uint sf_region = it / 2u;\n");
    source.push_str("    uint sf_half = it % 2u;\n");
    source.push_str("    uint sf_low_base = sf_region * 64u + sf_half * 16u;\n");
    source.push_str("    uint sf_high_base = sf_low_base + 32u;\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    for (int ib = (int)ix; ib < super_blocks; ib += 4) {\n");
    source.push_str("        int elem0_low = ib * 256 + (int)sf_low_base;\n");
    source.push_str("        int elem0_high = ib * 256 + (int)sf_high_base;\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "            device const uchar *blk = in{weight} + ((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS} + ib) * {block_bytes};\n"
    ));
    source.push_str("            q4k_header hdr_low = q4k_header_for(blk, sf_low_base);\n");
    source.push_str("            q4k_header hdr_high = q4k_header_for(blk, sf_high_base);\n");
    if is_plain_product_reduce(resolved, reduce_op, weight, other) {
        // SCALE-DEFERRED, split across TWO sub-blocks instead of one: this
        // lane covers 16 of sub-block-A's 32 elements (`raw_low`/`act_low`)
        // and 16 of sub-block-B's 32 (`raw_high`/`act_high`) — its pair
        // partner (`sf_half` flipped, same `sf_region`) covers the other 16
        // of each. `sum_j (scale*nibble_j - min)*act_j == scale*sum(nibble_j
        // *act_j) - min*sum(act_j)` (the same identity
        // `is_plain_product_reduce`'s caller already proved licenses) holds
        // per HALF exactly as it holds per whole sub-block, and addition
        // distributes over the two halves, so `simd_sum` over both lanes in
        // a pair reconstructs the identical two sub-block totals the
        // duplicate-fetch path computes in one lane each.
        source.push_str(&format!("            {element_type} raw_low = 0;\n"));
        source.push_str(&format!("            {element_type} act_low = 0;\n"));
        source.push_str(&format!("            {element_type} raw_high = 0;\n"));
        source.push_str(&format!("            {element_type} act_high = 0;\n"));
        source.push_str("            for (int c = 0; c < 2; ++c) {\n");
        source.push_str("                float low_levels[8];\n");
        source.push_str("                float high_levels[8];\n");
        source.push_str(
            "                q4k_run8_dual(blk, sf_low_base + (uint)(c * 8), low_levels, high_levels);\n",
        );
        source.push_str("                for (int j = 0; j < 8; ++j) {\n");
        source.push_str(&format!(
            "                    {element_type} act_l = in{other}[other_base[0] + (long)(elem0_low + c * 8 + j) * other_stride];\n"
        ));
        source.push_str(&format!(
            "                    {element_type} act_h = in{other}[other_base[0] + (long)(elem0_high + c * 8 + j) * other_stride];\n"
        ));
        source.push_str("                    raw_low += low_levels[j] * act_l;\n");
        source.push_str("                    act_low += act_l;\n");
        source.push_str("                    raw_high += high_levels[j] * act_h;\n");
        source.push_str("                    act_high += act_h;\n");
        source.push_str("                }\n");
        source.push_str("            }\n");
        source.push_str(
            "            sumf[q] = sumf[q] + hdr_low.scale * raw_low - hdr_low.minimum * act_low + hdr_high.scale * raw_high - hdr_high.minimum * act_high;\n",
        );
    } else {
        source.push_str("            for (int c = 0; c < 2; ++c) {\n");
        source.push_str("                float low_levels[8];\n");
        source.push_str("                float high_levels[8];\n");
        source.push_str(
            "                q4k_run8_dual(blk, sf_low_base + (uint)(c * 8), low_levels, high_levels);\n",
        );
        source.push_str("                for (int j = 0; j < 8; ++j) {\n");
        source.push_str(&format!(
            "                    {element_type} scratch[{}];\n",
            operand_count.max(1)
        ));
        source.push_str(&format!(
            "                    scratch[{weight}] = hdr_low.scale * low_levels[j] - hdr_low.minimum;\n"
        ));
        source.push_str(&format!(
            "                    scratch[{other}] = in{other}[other_base[0] + (long)(elem0_low + c * 8 + j) * other_stride];\n"
        ));
        let low_value_expr = push_body_steps(
            source,
            resolved.element_body(),
            "                    ",
            element_type,
        );
        source.push_str(&format!(
            "                    {element_type} value = {low_value_expr};\n"
        ));
        let low_combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
        source.push_str(&format!(
            "                    sumf[q] = {low_combine_expr};\n"
        ));
        source.push_str("                }\n");
        source.push_str("                for (int j = 0; j < 8; ++j) {\n");
        source.push_str(&format!(
            "                    {element_type} scratch[{}];\n",
            operand_count.max(1)
        ));
        source.push_str(&format!(
            "                    scratch[{weight}] = hdr_high.scale * high_levels[j] - hdr_high.minimum;\n"
        ));
        source.push_str(&format!(
            "                    scratch[{other}] = in{other}[other_base[0] + (long)(elem0_high + c * 8 + j) * other_stride];\n"
        ));
        let high_value_expr = push_body_steps(
            source,
            resolved.element_body(),
            "                    ",
            element_type,
        );
        source.push_str(&format!(
            "                    {element_type} value = {high_value_expr};\n"
        ));
        let high_combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
        source.push_str(&format!(
            "                    sumf[q] = {high_combine_expr};\n"
        ));
        source.push_str("                }\n");
        source.push_str("            }\n");
    }
    source.push_str("        }\n");
    source.push_str("    }\n");
}

/// The plain-product row-blocked bodies' shared activation address: the
/// `y4` pointer's base and per-iteration byte step. `push_packed_row_blocked_
/// body`'s own `plain_product` arm and `push_q4k_ggml_port_body` both read
/// `weight_base`/`other_base`/`other_stride`/`iq`/`ir` from the same
/// preamble and must have already declared `ib_first`/`ib_step` in scope --
/// this is the ONE place either renders the pointer, so the stride-free
/// specialization (drop the runtime `other_stride` multiply when the layout
/// proves it is 1, see the caller's own `other_stride_is_one` doc) applies
/// identically to both instead of drifting.
fn push_q4k_plain_product_y4_address(source: &mut String, other: usize, other_stride_is_one: bool) {
    push_packed_row_plain_product_y4_address(
        source,
        other,
        other_stride_is_one,
        "64u * iq + 8u * ir",
    );
}

/// [`push_q4k_plain_product_y4_address`] generalized to any ggml-port body's
/// own lane-derived activation offset -- [`push_q6k_ggml_port_body`] shares
/// this instead of a second copy of the address arithmetic, passing
/// `"128u * ip + l0"` (llama's `y_offset`, `ggml-metal.metal:5383`) in place
/// of `Q4_K`'s `"64u * iq + 8u * ir"`.
fn push_packed_row_plain_product_y4_address(
    source: &mut String,
    other: usize,
    other_stride_is_one: bool,
    lane_offset: &str,
) {
    if other_stride_is_one {
        source.push_str(&format!(
            "    long y4_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS};\n"
        ));
        source.push_str(&format!("    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} + (long)({lane_offset});\n"));
    } else {
        source.push_str(&format!(
            "    long y4_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
        ));
        source.push_str(&format!("    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} * other_stride + (long)({lane_offset}) * other_stride;\n"));
    }
}

// ggml (llama.cpp, MIT license: https://github.com/ggml-org/llama.cpp/blob/
// master/LICENSE) `kernel_mul_mv_q4_K_f32_impl<4,2,32>`
// (ggml-metal.metal:5086-5193), transcribed line-for-line onto this crate's
// operand-base/stride addressing. See `push_q4k_ggml_port_body`'s own doc for
// what is and is not identical to the upstream source.
//
// Copyright (c) 2023-2024 The ggml authors. MIT-licensed; see THIRD_PARTY.md.
//
/// `metal-q4k-ggml-port` (default-off): a VERBATIM port of ggml's
/// `kernel_mul_mv_q4_K_f32_impl<nr0=4, nsg=2, nw=32>`
/// (`ggml-metal.metal:5086-5193`) -- every prior landing on this path
/// (`q4k_pair_dot`'s `plain_product` arm above, `metal-q4k-mask-fma`,
/// `metal-q4k-single-fetch`) is this crate's own RE-DERIVATION of pieces of
/// ggml's technique through its `q4k_header`/`q4k_run8` abstractions; this
/// function instead transcribes ggml's actual per-thread math with no
/// intermediate abstraction, so the only remaining difference from upstream
/// is address computation (`weight_base[q]`/`other_base[0]`/`other_stride`,
/// this crate's per-axis strided reads, instead of ggml's raw `nb01` pointer
/// walk -- ggml's own `q1 += args.nb01/2` row-advance is behaviorally
/// identical to this function's per-row `blk` recompute for the contiguous
/// packed-row layout this crate always uses). The activation address itself
/// shares `push_q4k_plain_product_y4_address` with
/// `push_packed_row_blocked_body`'s own `plain_product` arm, so when the
/// layout proves the reduce-axis stride is 1 this body drops the runtime
/// multiply the same way ggml's raw pointer walk always did.
///
/// Per-thread split (ggml-metal.metal:5100-5103), unchanged from the
/// existing row-blocked preamble's own lane assignment: `ix = lane/8`
/// (0..3, which of 4 super-blocks in today's `ib` stride this lane owns),
/// `it = lane%8` (0..7), `iq = it/4` (0 or 1, selects `q1` vs `q2`'s 64-byte
/// `qs` half), `ir = it%4` (0..3, a 4-uint16 stride within that half).
///
/// Super-block iteration (ggml-metal.metal:5132,5182): `for (ib = ix; ib <
/// nb; ib += 4)`, `nb = reduction_total / 256`; `y4` (the activation gather
/// base) advances by `4 * QK_K` (1024) elements per iteration, matching
/// ggml's `y4 += 4 * QK_K`.
///
/// Scale/min extraction (ggml-metal.metal:5096-5098,5142-5150): three fixed
/// masks, `kmask1 = 0x3f3f` (two 6-bit scale/min fields), `kmask2 = 0x0f0f`
/// (two 4-bit high-scale/high-min fields), `kmask3 = 0xc0c0` (the two
/// leftover high bits of the LOW fields, shifted into place with `>> 2`) --
/// no shift-then-branch the way this file's own `q4k_scale_min` reads it.
/// `sc16[0..3]` (aliased as 8 bytes `sc8[0..7]`) hold, in order: low-group
/// low-half scale, low-group low-half min, low-group high-half scale,
/// low-group high-half min, high-group low-half scale, high-group low-half
/// min, high-group high-half scale, high-group high-half min.
///
/// Nibble extraction (ggml-metal.metal:5157-5166): FOUR fixed bit-position
/// masks off each raw `uint16_t` word -- `& 0x000F`, `& 0x0F00`, `& 0x00F0`,
/// `& 0xF000` -- no runtime shift at all. The `0x0F00`/`0xF000` masks leave
/// their nibble sitting at bit 8/12, so the corresponding accumulator lane
/// (`acc1[1]`/`acc1[3]`/`acc2[1]`/`acc2[3]`) is 256x too large; `0x00F0`
/// leaves its nibble at bit 4, 16x too large. Both residuals are folded into
/// the FINAL per-sub-block combine below (ggml-metal.metal:5171-5175)
/// rather than corrected per element -- `1.0f/256.0f` on the odd
/// accumulator lanes, `1.0f/16.0f` on the whole second scale/min term --
/// this is the "mask-without-shift" technique `metal-q4k-mask-fma`'s own doc
/// names but only ports for the header decode, not this accumulate.
///
/// SIMD reduction (ggml-metal.metal:5187-5192): unchanged from every other
/// row-blocked arm -- `simd_sum(sumf[row])` combines the 32 lanes of one
/// simdgroup, lane 0 alone writes -- handled by the shared
/// `push_packed_row_combine_and_write` tail this function's caller still
/// invokes after it returns.
///
/// Dispatch geometry: `nr0 = 4` is already this crate's own
/// `PACKED_ROWS_PER_GROUP`; `nsg = 2` is wired separately, in
/// `tiled_gemm_threadgroup_width`'s own `metal-q4k-ggml-port` arm.
#[allow(clippy::too_many_arguments)]
fn push_q4k_ggml_port_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    block_bytes: usize,
    other_stride_is_one: bool,
) {
    source.push_str("    uint ix = (uint)lane / 8u;\n");
    source.push_str("    uint it = (uint)lane % 8u;\n");
    source.push_str("    uint iq = it / 4u;\n");
    source.push_str("    uint ir = it % 4u;\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    float yl[16];\n");
    source.push_str("    float yh[16];\n");
    // HOIST + POINTER INCREMENT, same as `push_packed_row_blocked_body`'s
    // own arm above and for the identical reason: `weight_base[q]/
    // Q4K_BLOCK_ELEMENTS` and the y4 lane offset are invariant across every
    // `ib` this thread visits (`ix` alone selects the starting super-block,
    // the loop always steps by the fixed `4`), so both this row's byte
    // pointer and the activation base are computed ONCE and advanced by a
    // constant per iteration instead of rebuilt from `ib` every time.
    // `ib_first`/`ib_step` (rather than `ix`/the literal `4`) are the same
    // names `push_packed_row_blocked_body`'s own arm declares, so
    // `push_q4k_plain_product_y4_address` renders identical text in both
    // callers.
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 4;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {block_bytes};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    push_q4k_plain_product_y4_address(source, other, other_stride_is_one);
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    source.push_str(
        "        float sumy0 = 0.0f; float sumy1 = 0.0f; float sumy2 = 0.0f; float sumy3 = 0.0f;\n",
    );
    if other_stride_is_one {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[i]; sumy0 += yl[i];\n",
        );
        source.push_str("            yl[i + 8u] = y4[i + 32u]; sumy1 += yl[i + 8u];\n");
        source.push_str("            yh[i] = y4[i + 128u]; sumy2 += yh[i];\n");
        source.push_str("            yh[i + 8u] = y4[i + 160u]; sumy3 += yh[i + 8u];\n");
    } else {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[(long)i * other_stride]; sumy0 += yl[i];\n",
        );
        source.push_str(
            "            yl[i + 8u] = y4[(long)(i + 32u) * other_stride]; sumy1 += yl[i + 8u];\n",
        );
        source
            .push_str("            yh[i] = y4[(long)(i + 128u) * other_stride]; sumy2 += yh[i];\n");
        source.push_str(
            "            yh[i + 8u] = y4[(long)(i + 160u) * other_stride]; sumy3 += yh[i + 8u];\n",
        );
    }
    source.push_str("        }\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("            device const uchar *blk = blk_ptr[q];\n");
    source
        .push_str("            device const ushort *sc = (device const ushort *)(blk + 4) + iq;\n");
    source.push_str("            device const ushort *q1 = (device const ushort *)(blk + 16) + 16u * iq + 4u * ir;\n");
    source.push_str("            device const ushort *q2 = q1 + 32;\n");
    source.push_str("            device const half *dh = (device const half *)blk;\n");
    source.push_str("            ushort sc16_0 = sc[0] & (ushort)0x3f3fu;\n");
    source.push_str("            ushort sc16_1 = sc[2] & (ushort)0x3f3fu;\n");
    source.push_str(
        "            ushort sc16_2 = (ushort)(((sc[4] >> 0) & (ushort)0x0f0fu) | ((sc[0] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            ushort sc16_3 = (ushort)(((sc[4] >> 4) & (ushort)0x0f0fu) | ((sc[2] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            uchar sc8_0 = (uchar)(sc16_0 & 0xffu); uchar sc8_1 = (uchar)(sc16_0 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_2 = (uchar)(sc16_1 & 0xffu); uchar sc8_3 = (uchar)(sc16_1 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_4 = (uchar)(sc16_2 & 0xffu); uchar sc8_5 = (uchar)(sc16_2 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_6 = (uchar)(sc16_3 & 0xffu); uchar sc8_7 = (uchar)(sc16_3 >> 8);\n",
    );
    source.push_str(
        "            float acc1_0 = 0.0f; float acc1_1 = 0.0f; float acc1_2 = 0.0f; float acc1_3 = 0.0f;\n",
    );
    source.push_str(
        "            float acc2_0 = 0.0f; float acc2_1 = 0.0f; float acc2_2 = 0.0f; float acc2_3 = 0.0f;\n",
    );
    source.push_str("            for (uint i = 0u; i < 4u; ++i) {\n");
    source.push_str("                ushort word1 = q1[i];\n");
    source.push_str("                ushort word2 = q2[i];\n");
    source.push_str(
        "                acc1_0 += yl[2u * i + 0u] * (float)(word1 & (ushort)0x000Fu);\n",
    );
    source.push_str(
        "                acc1_1 += yl[2u * i + 1u] * (float)(word1 & (ushort)0x0F00u);\n",
    );
    source.push_str(
        "                acc1_2 += yl[2u * i + 8u] * (float)(word1 & (ushort)0x00F0u);\n",
    );
    source.push_str(
        "                acc1_3 += yl[2u * i + 9u] * (float)(word1 & (ushort)0xF000u);\n",
    );
    source.push_str(
        "                acc2_0 += yh[2u * i + 0u] * (float)(word2 & (ushort)0x000Fu);\n",
    );
    source.push_str(
        "                acc2_1 += yh[2u * i + 1u] * (float)(word2 & (ushort)0x0F00u);\n",
    );
    source.push_str(
        "                acc2_2 += yh[2u * i + 8u] * (float)(word2 & (ushort)0x00F0u);\n",
    );
    source.push_str(
        "                acc2_3 += yh[2u * i + 9u] * (float)(word2 & (ushort)0xF000u);\n",
    );
    source.push_str("            }\n");
    source.push_str("            float dall = (float)dh[0];\n");
    source.push_str("            float dmin = (float)dh[1];\n");
    source.push_str(
        "            sumf[q] = sumf[q] + dall * ((acc1_0 + (1.0f/256.0f) * acc1_1) * (float)sc8_0 +\n",
    );
    source.push_str(
        "                                       (acc1_2 + (1.0f/256.0f) * acc1_3) * (float)sc8_1 * (1.0f/16.0f) +\n",
    );
    source.push_str(
        "                                       (acc2_0 + (1.0f/256.0f) * acc2_1) * (float)sc8_4 +\n",
    );
    source.push_str(
        "                                       (acc2_2 + (1.0f/256.0f) * acc2_3) * (float)sc8_5 * (1.0f/16.0f)) -\n",
    );
    source.push_str(
        "                      dmin * (sumy0 * (float)sc8_2 + sumy1 * (float)sc8_3 + sumy2 * (float)sc8_6 + sumy3 * (float)sc8_7);\n",
    );
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        y4 += y4_step;\n");
    source.push_str("    }\n");
}

// ggml (llama.cpp, MIT license: https://github.com/ggml-org/llama.cpp/blob/
// master/LICENSE) `kernel_mul_mv_q5_K_f32_impl<2,2,32>`
// (ggml-metal.metal:5209-5324), transcribed line-for-line onto this crate's
// operand-base/stride addressing, the same posture as
// [`push_q4k_ggml_port_body`].
//
// Copyright (c) 2023-2024 The ggml authors. MIT-licensed; see THIRD_PARTY.md.
//
/// `metal-q4k-ggml-port` (default-off, same feature `push_q4k_ggml_port_body`
/// answers to): a VERBATIM port of ggml's `kernel_mul_mv_q5_K_f32_impl<nr0=2,
/// nsg=2, nw=32>` (`ggml-metal.metal:5209-5324`) -- the fix for this crate's
/// own `q5k_pair_dot`, whose non-deferred `(scale*value - minimum)*activation`
/// per lane (`docs/discipline.md` ROW 361/384) does the scale/min multiply-
/// subtract EIGHT times per sub-block; this body applies `dall`/`dmin` ONCE
/// per sub-block over pre-summed `acc1`/`acc2`/`sumy`, the same deferral
/// `push_q4k_ggml_port_body` already ports for `Q4_K`.
///
/// Lane assignment (`ggml-metal.metal:5244-5250`), same shape as
/// [`push_q4k_ggml_port_body`]'s own `tid`/`ix`/`iq`/`ir` (`Q5_K`'s `tid =
/// tiisg/4` divides by 4 where `Q4_K`'s divides by 8, because `Q5_K` has no
/// second `it/4` split -- `iq = tid/4`, `ir = tid%4` read directly off `tid`):
/// `l0 = 8*ir`; `q_offset = 32*iq + l0` (byte offset into `qs`, unused here --
/// folded into the `48 + q_offset` pointer below); `y_offset = 64*iq + l0`,
/// IDENTICAL to `Q4_K`'s own `y4` lane offset, so this body shares
/// [`push_q4k_plain_product_y4_address`] rather than a second copy.
///
/// Super-block iteration (`ggml-metal.metal:5263`): `for (i = ix; i < nb; i
/// += 4)`, `ix = tiisg%4` -- same stride `4` as `Q4_K`'s own `ib_step`
/// despite `Q5_K`'s `nr0=2` (this function still iterates `rows` generically,
/// the same posture [`push_q6k_ggml_port_body`]'s own doc calls out for its
/// `nr0=1`).
///
/// Activation gather (`ggml-metal.metal:5269-5276`): `y2 = y1 + 128`, so
/// `y2[l] == y1[128+l]` and `y2[l+32] == y1[160+l]` -- textually the SAME
/// `yl[i]/yl[i+8]/yh[i]/yh[i+8]` gather from `y4[i]/y4[i+32]/y4[i+128]/
/// y4[i+160]` [`push_q4k_ggml_port_body`] already renders, so this body
/// reuses that exact source text rather than a second copy.
///
/// Scale/min extraction (`ggml-metal.metal:5281-5284`): `Q5_K`'s
/// `scales`/`kmask1`/`kmask2`/`kmask3` bit layout is byte-for-byte `Q4_K`'s
/// own (`sc16_0 = sc[0] & 0x3f3f`, ..., `sc16_2 = ((sc[4]>>0) & 0x0f0f) |
/// ((sc[0]&0xc0c0)>>2)`, ...) at the SAME `blk + 4` byte offset -- restated
/// here rather than shared, same posture `q5k_scale_min` already takes on
/// `q4k_scale_min`.
///
/// High-bit plane (`ggml-metal.metal:5253-5256,5289-5297`, the part `Q4_K`
/// has no analogue for): `hm1 = 1 << (2*iq)`, `hm2 = hm1<<1`, `hm3 = hm1<<4`,
/// `hm4 = hm2<<4`, each fixed per-thread (depends only on `iq`, hoisted once
/// outside the `ib` loop). `qh = blk + 16 + l0`; per lane `l`, `q1 = blk + 48
/// + q_offset`, `q2 = q1 + 64`: `acc1[k]` sums `yl_or_yh[l(+8)]` times the
/// masked nibble byte (low or high half of `q1`/`q2`), `acc2[k]` sums the
/// same activation gated by `qh[l] & hm_k` -- no scale multiply inside this
/// loop at all, unlike `q5k_pair_dot`'s per-element
/// `scale*(nibble+high)-minimum`.
///
/// Final combine (`ggml-metal.metal:5299-5305`): `dall`/`dmin` applied ONCE,
/// `sumf[row] += dall*(sc8_0*(acc1_0+16*acc2_0) + sc8_1*(acc1_1/16 +
/// 16*acc2_1) + sc8_4*(acc1_2+16*acc2_2) + sc8_5*(acc1_3/16+16*acc2_3)) -
/// dmin*(sumy0*sc8_2 + sumy1*sc8_3 + sumy2*sc8_6 + sumy3*sc8_7)` -- the
/// `/16`/`*16` folds are `Q5_K`'s own bit-position residuals (the low
/// sub-block's high-nibble term `q1[l]&0xF0` sits 16x too large), not
/// `Q4_K`'s `1/256` word-packed residual, because this body reads plain
/// `uchar` bytes rather than `Q4_K`'s packed `ushort` words.
///
/// SIMD reduction: unchanged from every other row-blocked arm, handled by
/// the shared `push_packed_row_combine_and_write` tail this function's
/// caller still invokes after it returns.
#[allow(clippy::too_many_arguments)]
fn push_q5k_ggml_port_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    block_bytes: usize,
    other_stride_is_one: bool,
) {
    source.push_str("    uint tid = (uint)lane / 4u;\n");
    source.push_str("    uint ix = (uint)lane % 4u;\n");
    source.push_str("    uint iq = tid / 4u;\n");
    source.push_str("    uint ir = tid % 4u;\n");
    source.push_str("    uint l0 = 8u * ir;\n");
    source.push_str("    uint q_offset = 32u * iq + l0;\n");
    source.push_str("    uchar hm1 = (uchar)(1u << (2u * iq));\n");
    source.push_str("    uchar hm2 = (uchar)(hm1 << 1u);\n");
    source.push_str("    uchar hm3 = (uchar)(hm1 << 4u);\n");
    source.push_str("    uchar hm4 = (uchar)(hm2 << 4u);\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    float yl[16];\n");
    source.push_str("    float yh[16];\n");
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 4;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {block_bytes};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    push_q4k_plain_product_y4_address(source, other, other_stride_is_one);
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    source.push_str(
        "        float sumy0 = 0.0f; float sumy1 = 0.0f; float sumy2 = 0.0f; float sumy3 = 0.0f;\n",
    );
    if other_stride_is_one {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[i]; sumy0 += yl[i];\n",
        );
        source.push_str("            yl[i + 8u] = y4[i + 32u]; sumy1 += yl[i + 8u];\n");
        source.push_str("            yh[i] = y4[i + 128u]; sumy2 += yh[i];\n");
        source.push_str("            yh[i + 8u] = y4[i + 160u]; sumy3 += yh[i + 8u];\n");
    } else {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[(long)i * other_stride]; sumy0 += yl[i];\n",
        );
        source.push_str(
            "            yl[i + 8u] = y4[(long)(i + 32u) * other_stride]; sumy1 += yl[i + 8u];\n",
        );
        source
            .push_str("            yh[i] = y4[(long)(i + 128u) * other_stride]; sumy2 += yh[i];\n");
        source.push_str(
            "            yh[i + 8u] = y4[(long)(i + 160u) * other_stride]; sumy3 += yh[i + 8u];\n",
        );
    }
    source.push_str("        }\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("            device const uchar *blk = blk_ptr[q];\n");
    source
        .push_str("            device const ushort *sc = (device const ushort *)(blk + 4) + iq;\n");
    source.push_str("            device const uchar *qh = blk + 16u + l0;\n");
    source.push_str("            device const uchar *q1 = blk + 48u + q_offset;\n");
    source.push_str("            device const uchar *q2 = q1 + 64u;\n");
    source.push_str("            device const half *dh = (device const half *)blk;\n");
    source.push_str("            ushort sc16_0 = sc[0] & (ushort)0x3f3fu;\n");
    source.push_str("            ushort sc16_1 = sc[2] & (ushort)0x3f3fu;\n");
    source.push_str(
        "            ushort sc16_2 = (ushort)(((sc[4] >> 0) & (ushort)0x0f0fu) | ((sc[0] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            ushort sc16_3 = (ushort)(((sc[4] >> 4) & (ushort)0x0f0fu) | ((sc[2] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            uchar sc8_0 = (uchar)(sc16_0 & 0xffu); uchar sc8_1 = (uchar)(sc16_0 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_2 = (uchar)(sc16_1 & 0xffu); uchar sc8_3 = (uchar)(sc16_1 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_4 = (uchar)(sc16_2 & 0xffu); uchar sc8_5 = (uchar)(sc16_2 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_6 = (uchar)(sc16_3 & 0xffu); uchar sc8_7 = (uchar)(sc16_3 >> 8);\n",
    );
    source.push_str(
        "            float acc1_0 = 0.0f; float acc1_1 = 0.0f; float acc1_2 = 0.0f; float acc1_3 = 0.0f;\n",
    );
    source.push_str(
        "            float acc2_0 = 0.0f; float acc2_1 = 0.0f; float acc2_2 = 0.0f; float acc2_3 = 0.0f;\n",
    );
    source.push_str("            for (uint l = 0u; l < 8u; ++l) {\n");
    source.push_str("                uchar h = qh[l];\n");
    source.push_str("                acc1_0 += yl[l] * (float)(q1[l] & 0x0Fu);\n");
    source.push_str("                acc1_1 += yl[l + 8u] * (float)(q1[l] & 0xF0u);\n");
    source.push_str("                acc1_2 += yh[l] * (float)(q2[l] & 0x0Fu);\n");
    source.push_str("                acc1_3 += yh[l + 8u] * (float)(q2[l] & 0xF0u);\n");
    source.push_str("                acc2_0 += (h & hm1) != 0u ? yl[l] : 0.0f;\n");
    source.push_str("                acc2_1 += (h & hm2) != 0u ? yl[l + 8u] : 0.0f;\n");
    source.push_str("                acc2_2 += (h & hm3) != 0u ? yh[l] : 0.0f;\n");
    source.push_str("                acc2_3 += (h & hm4) != 0u ? yh[l + 8u] : 0.0f;\n");
    source.push_str("            }\n");
    source.push_str("            float dall = (float)dh[0];\n");
    source.push_str("            float dmin = (float)dh[1];\n");
    source.push_str(
        "            sumf[q] = sumf[q] + dall * ((float)sc8_0 * (acc1_0 + 16.0f * acc2_0) +\n",
    );
    source.push_str(
        "                                       (float)sc8_1 * (acc1_1 * (1.0f/16.0f) + 16.0f * acc2_1) +\n",
    );
    source.push_str(
        "                                       (float)sc8_4 * (acc1_2 + 16.0f * acc2_2) +\n",
    );
    source.push_str(
        "                                       (float)sc8_5 * (acc1_3 * (1.0f/16.0f) + 16.0f * acc2_3)) -\n",
    );
    source.push_str(
        "                      dmin * (sumy0 * (float)sc8_2 + sumy1 * (float)sc8_3 + sumy2 * (float)sc8_6 + sumy3 * (float)sc8_7);\n",
    );
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        y4 += y4_step;\n");
    source.push_str("    }\n");
}

// ggml (llama.cpp, MIT license: https://github.com/ggml-org/llama.cpp/blob/
// master/LICENSE) `kernel_mul_mv_q6_K_f32_impl<1,2,32>`
// (ggml-metal.metal:5340-5433), transcribed line-for-line onto this crate's
// operand-base/stride addressing, the same posture as
// [`push_q4k_ggml_port_body`].
//
// Copyright (c) 2023-2024 The ggml authors. MIT-licensed; see THIRD_PARTY.md.
//
/// `metal-q4k-ggml-port` (default-off, same feature `push_q4k_ggml_port_body`
/// answers to -- ROW 316 named the feature after the first codec it ported,
/// not the mechanism): a VERBATIM port of ggml's
/// `kernel_mul_mv_q6_K_f32_impl<nr0=1, nsg=2, nw=32>`
/// (`ggml-metal.metal:5340-5433`).
///
/// Lane variable mapping, llama -> this function (`ggml-metal.metal:5376-5385`):
/// `tid = tiisg/2` -> `tid = lane/2u`; `ix = tiisg%2` -> `ix = lane%2u`;
/// `ip = tid/8` -> `ip = tid/8u`; `il = tid%8` -> `il = tid%8u`;
/// `l0 = 4*il` -> `l0 = 4u*il`; `y_offset = 128*ip+l0` -> the `lane_offset`
/// this shares with [`push_packed_row_plain_product_y4_address`];
/// `q_offset_l = 64*ip+l0`/`q_offset_h = 32*ip+l0` -> identical names;
/// `is = 8*ip+l0/16` -> identical name. `N_R0_Q6_K = 1`
/// (`ggml-metal-impl.h:38`) collapses ggml's own `row`/`nr0` loop to a
/// single iteration, still written as `for q in 0..rows` so this stays in
/// step with [`PackedCodec::rows_per_simdgroup`] rather than baking `1` in
/// the way `push_q4k_ggml_port_body` bakes `4`.
///
/// Super-block iteration (`ggml-metal.metal:5387`): `for (i = ix; i < nb; i
/// += 2)` -- stride `2` (`N_SIMDWIDTH/2` lanes-per-super-block-half), not
/// `Q4_K`'s stride `4`.
///
/// 6-bit reconstruct (`ggml-metal.metal:5409-5412`): `kmask1=0x03`,
/// `kmask2=0x0C`, `kmask3=0x30`, `kmask4=0xC0`, folded directly into the
/// nibble OR with no runtime shift beyond what each mask already leaves in
/// place, then `- 32` (`Q6_K` has no `dmin` term at all, unlike `Q4_K`/`Q5_K`
/// -- ggml's own `sumf[row] += dall * (...)` has a single scale multiply,
/// no minimum subtraction).
///
/// SIMD reduction: unchanged from every other row-blocked arm, handled by
/// the shared `push_packed_row_combine_and_write` tail this function's
/// caller still invokes after it returns.
#[allow(clippy::too_many_arguments)]
fn push_q6k_ggml_port_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    block_bytes: usize,
    other_stride_is_one: bool,
) {
    source.push_str("    uint tid = (uint)lane / 2u;\n");
    source.push_str("    uint ix = (uint)lane % 2u;\n");
    source.push_str("    uint ip = tid / 8u;\n");
    source.push_str("    uint il = tid % 8u;\n");
    source.push_str("    uint l0 = 4u * il;\n");
    source.push_str("    uint is = 8u * ip + l0 / 16u;\n");
    source.push_str("    uint q_offset_l = 64u * ip + l0;\n");
    source.push_str("    uint q_offset_h = 32u * ip + l0;\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    float yl[16];\n");
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 2;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {block_bytes};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    push_packed_row_plain_product_y4_address(source, other, other_stride_is_one, "128u * ip + l0");
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    if other_stride_is_one {
        source.push_str("        for (uint l = 0u; l < 4u; ++l) {\n            yl[4u * l + 0u] = y4[l]; yl[4u * l + 1u] = y4[l + 32u]; yl[4u * l + 2u] = y4[l + 64u]; yl[4u * l + 3u] = y4[l + 96u];\n        }\n");
    } else {
        source.push_str("        for (uint l = 0u; l < 4u; ++l) {\n            yl[4u * l + 0u] = y4[(long)l * other_stride]; yl[4u * l + 1u] = y4[(long)(l + 32u) * other_stride]; yl[4u * l + 2u] = y4[(long)(l + 64u) * other_stride]; yl[4u * l + 3u] = y4[(long)(l + 96u) * other_stride];\n        }\n");
    }
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("            device const uchar *blk = blk_ptr[q];\n");
    source.push_str("            device const uchar *ql = blk;\n");
    source.push_str("            device const uchar *qh = blk + 128u;\n");
    source.push_str("            device const uchar *sc = blk + 192u + is;\n");
    source.push_str("            device const half *dh = (device const half *)(blk + 208u);\n");
    source.push_str("            float dall = (float)dh[0];\n");
    source.push_str(
        "            float sums0 = 0.0f; float sums1 = 0.0f; float sums2 = 0.0f; float sums3 = 0.0f;\n",
    );
    source.push_str("            for (uint l = 0u; l < 4u; ++l) {\n");
    source.push_str("                uchar q1l = ql[q_offset_l + l];\n");
    source.push_str("                uchar q2l = ql[q_offset_l + 32u + l];\n");
    source.push_str("                uchar qhl = qh[q_offset_h + l];\n");
    source.push_str(
        "                uint q_lo0 = (uint)(q1l & 0x0Fu) | (((uint)qhl & 0x03u) << 4u);\n",
    );
    source.push_str(
        "                uint q_lo1 = (uint)(q2l & 0x0Fu) | (((uint)qhl & 0x0Cu) << 2u);\n",
    );
    source.push_str("                uint q_hi0 = (uint)(q1l >> 4u) | ((uint)qhl & 0x30u);\n");
    source.push_str(
        "                uint q_hi1 = (uint)(q2l >> 4u) | (((uint)qhl & 0xC0u) >> 2u);\n",
    );
    source.push_str("                sums0 += yl[4u * l + 0u] * (float)((int)q_lo0 - 32);\n");
    source.push_str("                sums1 += yl[4u * l + 1u] * (float)((int)q_lo1 - 32);\n");
    source.push_str("                sums2 += yl[4u * l + 2u] * (float)((int)q_hi0 - 32);\n");
    source.push_str("                sums3 += yl[4u * l + 3u] * (float)((int)q_hi1 - 32);\n");
    source.push_str("            }\n");
    source.push_str(
        "            sumf[q] = sumf[q] + dall * (sums0 * (float)(char)sc[0] + sums1 * (float)(char)sc[2] + sums2 * (float)(char)sc[4] + sums3 * (float)(char)sc[6]);\n",
    );
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        y4 += y4_step;\n");
    source.push_str("    }\n");
}

/// `simdgroup_matrix`-tiled Q4_K x F32 GEMM (`docs/discipline.md` ROW 109,
/// superseding ROW 107's single-simdgroup design) -- ports
/// `ggml-metal.metal:6500-6600`'s `kernel_mul_mm` GEOMETRY, not just its
/// `simdgroup_float8x8` primitives: [`TILED_GEMM_NSG`] (4) `simdgroup`s
/// cooperate in ONE threadgroup, each owning a
/// `crate::sized::TILED_GEMM_BLOCK_M`/2 x `crate::sized::TILED_GEMM_BLOCK_N`/2
/// sub-tile of the threadgroup's full `BLOCK_M x BLOCK_N` output block (a
/// 2x2 simdgroup grid -- `sgitg & 1` the row half, `sgitg >> 1` the column
/// half, exactly ggml's own split), and the reduction steps by
/// `crate::sized::TILED_GEMM_BLOCK_K` (ggml's `BLOCK_SIZE_K`) rather than by
/// `TILE_DIM` alone: ROW 107's own root cause was pairing ONE simdgroup with
/// an 8-wide K-step, paying two `threadgroup_barrier`s per 8 elements of K
/// (up to 512 barrier round-trips at k=4096) for 64 output elements each --
/// this design pays the same two barriers per `BLOCK_K`(32)-wide step (128
/// round-trips at k=4096, 4x fewer) and each pair now amortizes across
/// `TILED_GEMM_NSG` simdgroups x `BLOCK_K`/`TILE_DIM` K-substeps computing
/// `BLOCK_M x BLOCK_N`(2048) output elements, not 64 -- the "work per
/// barrier" ROW 107's own recommendation named as the actual fix.
///
/// This crate's operand model reads through generic per-axis strides
/// (never assumes row-major-contiguous device memory the way ggml's raw
/// `nb01` byte strides do), so both operand tiles are staged the same way
/// [`push_packed_row_blocked_body`] already reads a strided operand, just
/// written into a fixed `threadgroup` array instead of a private register.
///
/// ROW 113 correction: the weight-tile staging loop itself now decodes with
/// [`push_packed_row_blocked_body`]'s OWN amortized pattern (`q4k_header_for`
/// once per 32-element sub-block, `q4k_run8` batching the nibble extract 8
/// at a time), matching ggml's `dequantize_q4_K` (`ggml-metal.metal:336-352`,
/// which computes `dl`/`ml` once and loops 16 elements). Before this row it
/// called the generic [`operand_read`] (`q4k_element`), which rederives the
/// full header from `device` memory on every element -- correct (cross-token
/// tile reuse via `threadgroup` staging was always real, confirmed by
/// reading the emitted MSL) but roughly 8-40x more device reads and
/// arithmetic per weight element than necessary, which a per-op profiling
/// harness measured as 58.35x slower than decode (ROW 112) even though the
/// tile itself was never re-streamed per token.
///
/// Threadgroup memory is three FIXED-SIZE local arrays declared directly in
/// the kernel body (`weight_tile`: `BLOCK_M * BLOCK_K` `half`; `act_tile`:
/// `BLOCK_N * BLOCK_K` `float`; `out_tile`: `BLOCK_M * BLOCK_N` `float`,
/// reused across `k0` steps but allocated once) -- every dimension is a
/// compile-time constant (`crate::sized::TILED_GEMM_BLOCK_M`/`_N`/`_K`), so
/// this needs no `[[threadgroup(n)]]` kernel parameter and no
/// `setThreadgroupMemoryLength` call on the driver side, unlike ggml's
/// dynamically-sized `shmem` (`ggml-metal.m:3101`): every existing call
/// site in `crate::metal` keeps dispatching through the same
/// `dispatchThreads:threadsPerThreadgroup:` path unchanged, now with
/// [`TILED_GEMM_NSG`] `* SIMD_WIDTH` (128) threads per threadgroup instead
/// of one simdgroup ([`crate::msl::tiled_gemm_threadgroup_width`]).
///
/// Boundary tiles (feature or token extent not a whole multiple of
/// `BLOCK_M`/`BLOCK_N`) are handled by zero-padding out-of-range reads
/// during staging (a true-zero contribution changes nothing) and skipping
/// out-of-range writes entirely during the final scatter -- the same
/// n_rows/n_cols masking `ggml-metal.metal`'s own kernel applies, at
/// `BLOCK_M`/`BLOCK_N` granularity instead of `TILE_DIM`'s. The reduction
/// dimension needs no such mask: [`PackedRowBlock`] already guarantees it
/// is a whole number of [`Q4K_BLOCK_ELEMENTS`] (256) super-blocks, and
/// `build.rs`'s `require_divides_q4k_block` guarantees `BLOCK_K` divides
/// 256 evenly.
///
/// `weight_tile`/`act_tile` are both stored simple row-major (`weight_tile`:
/// feature-row-major, `act_tile`: token-row-major, K fastest in both --
/// UNLIKE ggml's own custom bit-shuffled `sa`/`sb` packing, which exists
/// only so its `simdgroup_load` calls can omit `elements_per_row` and read
/// each fragment pre-packed). `a_frag` reads a `feature x k` fragment
/// straight off `weight_tile`, but `b_frag` reads `act_tile` in its
/// NATURAL `token x k` orientation -- the wrong shape for
/// `simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc)`, which needs
/// its second operand `k x token` for the inner (`k`) dimensions to align.
/// `simdgroup_load`'s `transpose_matrix` flag supplies that without
/// restructuring the staging loop: `b_frag` is loaded with
/// `transpose_matrix = true`, turning the physical `token x k` read into
/// the logical `k x token` fragment the multiply needs. (A first pass
/// without this flag measured `relative=0.497` against the CPU oracle --
/// dimensionally valid MSL, semantically wrong matrix product -- caught by
/// `metal_matmul_on_packed_q4k_weights_matches_the_dequantized_f32_cpu_path_at_tile_scale`.)
#[cfg(feature = "metal-tiled-gemm")]
fn push_tiled_gemm_body(
    source: &mut String,
    node: NodeId,
    output_axes: &[u16],
    rank: usize,
    block: &TiledGemmBlock,
    element_type: &str,
) -> Result<(), EmitError> {
    let TiledGemmBlock {
        weight,
        other,
        reduce_dim,
        ref token_axes,
        ref feature_axes,
    } = *block;
    // innermost (fastest, last-listed) of each group -- the single stride
    // the per-element reads below use; see `TiledGemmBlock`'s own doc.
    let Some(&token_axis) = token_axes.last() else {
        return Err(EmitError::EmptyAxisGroup {
            node,
            group: "token",
        });
    };
    let Some(&feature_axis) = feature_axes.last() else {
        return Err(EmitError::EmptyAxisGroup {
            node,
            group: "feature",
        });
    };
    let rank_len = rank.max(1);

    let block_m = crate::sized::TILED_GEMM_BLOCK_M;
    let block_n = crate::sized::TILED_GEMM_BLOCK_N;
    let block_k = crate::sized::TILED_GEMM_BLOCK_K;
    let block_threads = (TILED_GEMM_NSG as u64) * SIMD_WIDTH;
    // 2 row-halves x TILE_DIM(8)-wide simdgroup-matrix fragments per half --
    // ggml's own `THREAD_MAT_M`/`THREAD_MAT_N` (`ggml-metal.metal:6490-6491`).
    let thread_mat_m = block_m / (TILE_DIM as u64 * 2);
    let thread_mat_n = block_n / (TILE_DIM as u64 * 2);
    let mc_count = thread_mat_m * thread_mat_n;
    let sub_k_steps = block_k / TILE_DIM as u64;
    let weight_tile_elems = block_m * block_k;
    let act_tile_elems = block_n * block_k;
    let out_tile_elems = block_m * block_n;

    // `attn_q`/`attn_k`/`attn_v` fold TWO weight-owned axes (`heads`,
    // `head_dim`) into one flattened feature dimension -- `axes_fold_
    // contiguously` already proved the group is one contiguous block, so
    // the runtime extent is the PRODUCT of every axis's own
    // `u.output_extents` entry, read fresh per dispatch the same way a
    // single-axis group already was (the kernel source is reused across
    // concrete shapes; see `TiledGemmBlock`'s own doc). Every real matmul
    // this path has measured keeps `token_axes` a single axis, but the
    // product generalizes to that case for free (one factor, no-op).
    let group_extent_expr = |group: &[u16]| -> Result<String, EmitError> {
        let mut terms = Vec::with_capacity(group.len());
        for &dim in group {
            let Some(index) = output_axes.iter().position(|&candidate| candidate == dim) else {
                return Err(EmitError::AxisNotInOutputAxes { node, axis: dim });
            };
            terms.push(format!("u.output_extents[{index}]"));
        }
        Ok(terms.join(" * "))
    };

    source.push_str(&format!(
        "    long feature_extent = {};\n",
        group_extent_expr(feature_axes)?
    ));
    source.push_str(&format!(
        "    long token_extent = {};\n",
        group_extent_expr(token_axes)?
    ));
    source.push_str(&format!(
        "    long num_col_tiles = (token_extent + {}) / {block_n};\n",
        block_n - 1
    ));
    source.push_str(&format!("    long tiitg = (long)gid % {block_threads};\n"));
    source.push_str(&format!("    long sgitg = tiitg / {SIMD_WIDTH};\n"));
    source.push_str(&format!(
        "    long tile_index = (long)gid / {block_threads};\n"
    ));
    source.push_str("    long row_tile = tile_index / num_col_tiles;\n");
    source.push_str("    long col_tile = tile_index % num_col_tiles;\n");
    source.push_str("    long row_half = sgitg & 1;\n");
    source.push_str("    long col_half = sgitg >> 1;\n");
    source.push_str(&format!(
        "    threadgroup half weight_tile[{weight_tile_elems}];\n"
    ));
    source.push_str(&format!(
        "    threadgroup float act_tile[{act_tile_elems}];\n"
    ));
    source.push_str(&format!("    simdgroup_float8x8 acc[{mc_count}];\n"));
    source.push_str(&format!(
        "    for (int i = 0; i < {mc_count}; ++i) {{ acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }}\n"
    ));
    source.push_str(&format!(
        "    for (long k0 = 0; k0 < u.reduction_total; k0 += {block_k}) {{\n"
    ));
    // ROW 113: weight staging amortizes the Q4_K sub-block header the same
    // way `push_packed_row_blocked_body` and ggml's own `dequantize_q4_K`
    // (ggml-metal.metal:336-352) both do -- one `q4k_header_for` per
    // 32-element sub-block, `q4k_run8` batching the nibble extract 8 at a
    // time -- instead of `operand_read`'s generic `q4k_element`, which
    // rederives the header (two `device` header reads plus the 6-bit
    // scale/min unpack) from scratch on every one of the tile's individual
    // elements. Staged by ROW rather than by flat index: `block_threads`
    // (128) exceeds `block_m` (64) with the default sizing, so the first
    // `block_m` threads each own exactly one row of the tile for this phase
    // and the rest do no extra weight work (`act_tile`'s own load below
    // still uses every thread).
    source.push_str(&format!(
        "        for (long w_row = tiitg; w_row < {block_m}; w_row += {block_threads}) {{\n"
    ));
    source.push_str(&format!(
        "            long w_feat = row_tile * {block_m} + w_row;\n"
    ));
    source.push_str("            if (w_feat < feature_extent) {\n");
    source.push_str(&format!(
        "                long row_base = u.operand_base[{weight}] + w_feat * u.operand_strides[{weight}][{feature_axis}] + k0 * u.operand_strides[{weight}][{reduce_dim}];\n"
    ));
    // `block_k` divides 256 (`Q4K_BLOCK_ELEMENTS`, `build.rs`'s
    // `require_divides_q4k_block`) and is a multiple of 8 (`build.rs`'s own
    // `require_multiple_of_eight`, added alongside this row), so it is
    // always either <= the Q4_K sub-block width (32) or a whole multiple of
    // it -- `chunk_width` picks the smaller, `num_chunks` covers `block_k`
    // exactly with no ragged remainder either way.
    let q4k_subblock_width: u64 = (Q4K_BLOCK_ELEMENTS / 8) as u64;
    let chunk_width = q4k_subblock_width.min(block_k);
    let num_chunks = block_k.div_ceil(chunk_width);
    for chunk_index in 0..num_chunks {
        let chunk_offset = chunk_index * chunk_width;
        source.push_str("                {\n");
        source.push_str(&format!(
            "                    long slot_off = row_base + {chunk_offset};\n"
        ));
        source.push_str(&format!(
            "                    device const uchar *blk = in{weight} + (slot_off / {Q4K_BLOCK_ELEMENTS}) * {Q4K_BLOCK_BYTES};\n"
        ));
        source.push_str(&format!(
            "                    uint slot = (uint)(slot_off % {Q4K_BLOCK_ELEMENTS});\n"
        ));
        source.push_str("                    q4k_header hdr = q4k_header_for(blk, slot);\n");
        let runs = chunk_width / 8;
        for run_index in 0..runs {
            let run_offset = run_index * 8;
            source.push_str("                    {\n");
            source.push_str("                        float levels[8];\n");
            source.push_str(&format!(
                "                        q4k_run8(blk, slot + {run_offset}u, levels);\n"
            ));
            source.push_str(&format!(
                "                        for (int j = 0; j < 8; ++j) {{ weight_tile[w_row * {block_k} + {chunk_offset} + {run_offset} + j] = (half)(hdr.scale * levels[j] - hdr.minimum); }}\n"
            ));
            source.push_str("                    }\n");
        }
        source.push_str("                }\n");
    }
    source.push_str("            } else {\n");
    source.push_str(&format!(
        "                for (long fill_k = 0; fill_k < {block_k}; ++fill_k) {{ weight_tile[w_row * {block_k} + fill_k] = 0.0h; }}\n"
    ));
    source.push_str("            }\n");
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (long idx = tiitg; idx < {act_tile_elems}; idx += {block_threads}) {{\n"
    ));
    source.push_str(&format!("            long a_col = idx / {block_k};\n"));
    source.push_str(&format!("            long a_k = idx % {block_k};\n"));
    source.push_str(&format!(
        "            long a_tok = col_tile * {block_n} + a_col;\n"
    ));
    source.push_str("            long a_k_global = k0 + a_k;\n");
    source.push_str("            float a_value = 0.0f;\n");
    source.push_str("            if (a_tok < token_extent) {\n");
    source.push_str(&format!(
        "                long aoff = u.operand_base[{other}] + a_tok * u.operand_strides[{other}][{token_axis}] + a_k_global * u.operand_strides[{other}][{reduce_dim}];\n"
    ));
    source.push_str(&format!(
        "                a_value = {};\n",
        operand_read(other, "aoff", None)
    ));
    source.push_str("            }\n");
    source.push_str(&format!(
        "            act_tile[a_col * {block_k} + a_k] = a_value;\n"
    ));
    source.push_str("        }\n");
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str(&format!(
        "        for (int sub_k = 0; sub_k < {sub_k_steps}; ++sub_k) {{\n"
    ));
    source.push_str(&format!(
        "            simdgroup_half8x8 a_frag[{thread_mat_m}];\n"
    ));
    source.push_str(&format!(
        "            for (int i = 0; i < {thread_mat_m}; ++i) {{\n"
    ));
    source.push_str(&format!(
        "                simdgroup_load(a_frag[i], weight_tile + (row_half * {thread_mat_m} + i) * 8 * {block_k} + sub_k * 8, {block_k});\n"
    ));
    source.push_str("            }\n");
    source.push_str("            simdgroup_barrier(mem_flags::mem_none);\n");
    source.push_str(&format!(
        "            simdgroup_float8x8 b_frag[{thread_mat_n}];\n"
    ));
    source.push_str(&format!(
        "            for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
    ));
    source.push_str(&format!(
        "                simdgroup_load(b_frag[j], act_tile + (col_half * {thread_mat_n} + j) * 8 * {block_k} + sub_k * 8, {block_k}, ulong2(0), true);\n"
    ));
    source.push_str("            }\n");
    source.push_str(&format!(
        "            for (int i = 0; i < {thread_mat_m}; ++i) {{\n"
    ));
    source.push_str(&format!(
        "                for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
    ));
    source.push_str(&format!(
        "                    simdgroup_multiply_accumulate(acc[i * {thread_mat_n} + j], a_frag[i], b_frag[j], acc[i * {thread_mat_n} + j]);\n"
    ));
    source.push_str("                }\n");
    source.push_str("            }\n");
    source.push_str("        }\n");
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("    }\n");
    source.push_str(&format!(
        "    threadgroup float out_tile[{out_tile_elems}];\n"
    ));
    source.push_str(&format!(
        "    for (int i = 0; i < {thread_mat_m}; ++i) {{\n"
    ));
    source.push_str(&format!(
        "        for (int j = 0; j < {thread_mat_n}; ++j) {{\n"
    ));
    source.push_str(&format!(
        "            simdgroup_store(acc[i * {thread_mat_n} + j], out_tile + (row_half * {thread_mat_m} + i) * 8 * {block_n} + (col_half * {thread_mat_n} + j) * 8, {block_n});\n"
    ));
    source.push_str("        }\n");
    source.push_str("    }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str(&format!(
        "    for (long idx = tiitg; idx < {out_tile_elems}; idx += {block_threads}) {{\n"
    ));
    source.push_str(&format!("        long o_row = idx / {block_n};\n"));
    source.push_str(&format!("        long o_col = idx % {block_n};\n"));
    source.push_str(&format!(
        "        long o_feat = row_tile * {block_m} + o_row;\n"
    ));
    source.push_str(&format!(
        "        long o_tok = col_tile * {block_n} + o_col;\n"
    ));
    source.push_str("        if (o_feat < feature_extent && o_tok < token_extent) {\n");
    source.push_str(&format!("            long coord[{rank_len}];\n"));
    source.push_str(&format!(
        "            for (int d = 0; d < {rank}; ++d) {{ coord[d] = 0; }}\n"
    ));
    source.push_str(&format!("            coord[{feature_axis}] = o_feat;\n"));
    source.push_str(&format!("            coord[{token_axis}] = o_tok;\n"));
    source.push_str("            long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "            out_offset += coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    source.push_str(&format!(
        "            out[out_offset] = ({element_type})out_tile[idx];\n"
    ));
    source.push_str("        }\n");
    source.push_str("    }\n");
    Ok(())
}

/// Never actually invoked: [`classify_tiled_gemm`]'s own `#[cfg(not(feature
/// = "metal-tiled-gemm"))]` arm always returns `None`, so no caller ever
/// holds a `&TiledGemmBlock` to pass here without the feature -- this stub
/// exists only so [`push_cooperative_reduce_body`]'s `if let Some(block) =
/// tiled_gemm_block(...)` arm still type-checks in that build.
#[cfg(not(feature = "metal-tiled-gemm"))]
fn push_tiled_gemm_body(
    source: &mut String,
    node: NodeId,
    output_axes: &[u16],
    rank: usize,
    block: &TiledGemmBlock,
    element_type: &str,
) -> Result<(), EmitError> {
    let _ = (source, output_axes, rank, block, element_type);
    Err(EmitError::TiledGemmFeatureDisabled { node })
}

/// The threadgroup width [`emit`]/[`kernel_dispatch_shape`] must dispatch
/// with -- [`TILED_GEMM_NSG`]` * SIMD_WIDTH` (128) when `resolved` takes
/// [`push_tiled_gemm_body`]'s multi-simdgroup path (its coordinate math
/// depends on exactly this many threads per threadgroup, the same
/// correctness requirement `crate::metal::dispatch`'s own doc states for
/// `SIMD_WIDTH`), `SIMD_WIDTH * split` for the row-blocked packed path when a
/// `Reduce`'s own shape carries `reduce_op`/`init`/`output_axes` (see
/// [`packed_row_dispatch`] -- `split` is `1`, i.e. plain `SIMD_WIDTH`, unless
/// `metal-q4k-split-k` is active and this shape is below the target
/// simdgroup count), [`crate::sized::PACKED_ROW_BLOCK_SIMDGROUPS`] *
/// `SIMD_WIDTH` for a packed row-block reached outside that match (widens the
/// packed row-block arm's threadgroup beyond one simdgroup — see that
/// constant's doc for why this is dispatch-only and never touches the kernel
/// body), [`cooperative_reduce_width`] for every other cooperative-reduce
/// kernel, `None` otherwise. Single source of truth both dispatch-shape
/// functions read, so they cannot drift the way two independent copies of
/// this `if`/`else` could. Ordered after the tiled-GEMM check and before the
/// generic cooperative-reduce fallback, matching [`grid_threads`]' own
/// priority (the two paths are mutually exclusive by construction —
/// [`kernel_cache_key`]'s doc).
fn tiled_gemm_threadgroup_width(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    numeric_policy: NumericPolicy,
) -> Option<u64> {
    // `query_groups * SIMD_WIDTH` threads per threadgroup -- one simdgroup
    // per query head sharing this kv_head, cooperatively loading that
    // kv_head's K/V row once per key into `threadgroup` memory instead of
    // each of the `query_groups` simdgroups re-reading it from device memory
    // (`render_cached_attention`'s own doc). Correctness-load-bearing, not an
    // occupancy hint: the body's `tid`/`group_width` split assumes exactly
    // this many threads land in the same threadgroup. Below
    // `cached_attention_per_query_head_grid`'s own knee, `query_groups` moves
    // out of this width entirely -- one query head per threadgroup, decoded
    // from `tgid` instead of shared threadgroup memory -- so the width there
    // is `chunks * SIMD_WIDTH` alone.
    if let BoundOpKind::CachedAttention {
        query_groups,
        cached_key_rows,
        new_key_rows,
        ..
    } = &resolved.kind
    {
        // Must agree with `grid_threads`'s own `CachedAttention` arm: the
        // single-range fused (dynamic) path always dispatches the compiled
        // MAXIMUM chunk count (`cap`), so the threadgroup width has to widen
        // to match -- a mismatch here puts fewer threads in the threadgroup
        // than `local_group_index`'s own `cap`-sized addressing assumes,
        // which is an out-of-bounds `threadgroup` memory write, not merely a
        // wrong answer.
        // Same `cached_key_rows == 0` discriminator as `grid_threads`'s own
        // `CachedAttention` arm -- `two_range_cached_bound` never widens to
        // the compiled cap, since its `context_length` is already the
        // bucket-padded compile-time value.
        let dynamic_cached_len = resolved.operands().len() == 9 && *cached_key_rows == 0;
        let context_length = *cached_key_rows + *new_key_rows;
        let chunks = if dynamic_cached_len {
            crate::sized::ATTENTION_CONTEXT_CHUNK_CAP
        } else {
            context_chunks_for(context_length, numeric_policy)
        };
        // Below the split-at-scale knee, `cached_attention_per_query_head_grid`
        // moves `query_groups` out of this width and into a threadgroup-count
        // factor instead (`grid_threads`' own total stays unchanged -- see
        // that function's doc) -- one query head per threadgroup rather than
        // `query_groups` of them sharing one.
        let width = if cached_attention_per_query_head_grid(dynamic_cached_len, context_length) {
            chunks * SIMD_WIDTH
        } else {
            *query_groups * chunks * SIMD_WIDTH
        };
        return Some(width);
    }
    if let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        reduce_op,
        init,
        output_axes,
        ..
    } = &resolved.kind
    {
        if tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some() {
            return Some((TILED_GEMM_NSG as u64) * SIMD_WIDTH);
        }
        if let Some(block) = packed_row_block(resolved, quantized) {
            let feature_total: u64 = block
                .feature_axes
                .iter()
                .map(|&axis| resolved.extents[axis as usize])
                .product();
            let token_total = packed_row_block_token_total(&block, &resolved.extents);
            let (_base, split) = packed_row_dispatch(feature_total, token_total, block.codec);
            // `metal-packed-row-nsg2`'s own `nsg=2` geometry (ggml's
            // `N_SG_Q4_K`, `ggml-metal-impl.h:33`) has to be applied HERE,
            // not in the `#[cfg(feature = "metal-packed-row-nsg2")]` arm
            // below -- this `if let` block's own `packed_row_block` check
            // returns unconditionally whenever it matches, so that arm below
            // is unreachable dead code for `Keep::Reduce` ops (every
            // `packed_row_block` match IS a `Keep::Reduce` op by
            // construction -- see `PackedRowBlock`'s own classification).
            // `metal-q4k-ggml-port` needs the identical nsg=2 width (its own
            // kernel body is ggml's, dispatched at ggml's own `N_SG_Q4_K`) --
            // [`packed_row_nsg_factor`] is the one place both features widen
            // from, so they cannot drift into two competing nsg constants.
            // `!metal-q4k-split-k` too: split-K's own combine (`push_packed_
            // row_combine_and_write`'s split-K arm) already picks a
            // cooperating `split` simdgroup count for a REAL reason -- a
            // starved shape's simdgroups share one output group via
            // `sgitg`/`threadgroup` memory/a barrier -- and doubling the
            // dispatched width again on top of that here, unconditionally,
            // would desync the combine's own `split` from the width the
            // driver actually dispatches (confirmed: `--all-features`,
            // ggml-port + split-K together, broke Q4_K parity outright,
            // relative=1). Every row-blocked body variant addresses its
            // output group purely from `gid / SIMD_WIDTH`
            // (`metal-packed-row-nsg2`'s own doc, still true here), so nsg=2
            // is correctness-neutral whenever split-K is off, regardless of
            // which body actually runs.
            return Some(SIMD_WIDTH * split * packed_row_nsg_factor());
        }
    }
    if packed_row_block(resolved, quantized).is_some() {
        return Some(crate::sized::PACKED_ROW_BLOCK_SIMDGROUPS * SIMD_WIDTH);
    }
    if !reduce_is_cooperative(resolved) {
        return None;
    }
    let BoundOpKind::Reduce {
        keep: Keep::Reduce,
        output_axes,
        ..
    } = &resolved.kind
    else {
        return None;
    };
    let reduce_dims = reduction_dims(resolved, output_axes);
    Some(cooperative_reduce_width(resolved, quantized, &reduce_dims))
}

/// Whether `resolved` takes [`push_cooperative_reduce_body`]'s "SUPER-BLOCK
/// TILED PACKED READ" arm -- exactly one Q4_K-packed operand, contiguous
/// along the single reduction dim, whose extent is a whole number of
/// super-blocks. That arm's lane math (`Q4K_BLOCK_ELEMENTS / SIMD_WIDTH`
/// contiguous elements per lane, `slot = lane * run`) is fixed to
/// `SIMD_WIDTH` lanes by construction -- widening the dispatch would push
/// `slot` past the super-block it is meant to stay inside. Extracted so
/// [`cooperative_reduce_width`] and the body can never disagree on which
/// shape a given op takes (mirrors [`tiled_gemm_threadgroup_width`]'s own
/// "single source of truth" doc).
fn q4k_super_block_tiled(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    reduce_dims: &[u16],
) -> bool {
    // Mixed expert sources have per-entry codecs and compact payload bases;
    // this Q4-only specialization cannot represent that ABI.
    if std::env::var_os("PROXIMA_ENABLE_UNSAFE_METAL_EXPERT_SOURCES").is_some() {
        return false;
    }
    if reduce_dims.len() != 1 {
        return false;
    }
    let reduce_dim = reduce_dims[0];
    let packed: Vec<usize> = quantized
        .iter()
        .enumerate()
        .filter_map(|(index, codec)| matches!(codec, Some(PackedCodec::Q4K)).then_some(index))
        .collect();
    packed.len() == 1
        && resolved.operands()[packed[0]].1.stride(reduce_dim) == 1
        && (resolved.extents[reduce_dim as usize] as usize).is_multiple_of(Q4K_BLOCK_ELEMENTS)
}

/// Cooperative-reduce threadgroup width -- `SIMD_WIDTH` (32) with this
/// feature off, matching the byte-identical prior behaviour every existing
/// gate baselines against. With `metal-wide-cooperative-reduce` on, scales
/// with the reduction extent instead of pinning every cooperative reduce to
/// one simdgroup regardless of size (the measured defect:
/// `docs/discipline.md`'s row for this initiative -- a 4096-element
/// RMS-norm sum launches 32 threads and each lane loops 128 times
/// serially). `reduction_total / 4` rounded up to the next multiple of
/// `SIMD_WIDTH`, clamped to `[SIMD_WIDTH,
/// WIDE_COOPERATIVE_REDUCE_MAX_WIDTH]` -- four elements of serial work per
/// lane keeps a short reduction (64, 128) from over-launching (more
/// threadgroup-barrier / partial-fold overhead than the serial work it
/// removes) while a long one (4096+) saturates the cap. Never applied to
/// [`q4k_super_block_tiled`]'s arm: that lane math is fixed to `SIMD_WIDTH`
/// by construction, not a policy choice this scaling could touch.
///
/// `WIDE_COOPERATIVE_REDUCE_MAX_WIDTH` is a build-time-configured cap
/// (`omega-runtime.toml`'s `[wide_cooperative_reduce]` section,
/// `crate::sized`), NOT a query of the device's real
/// `maxTotalThreadsPerThreadgroup` -- emission has no device handle
/// (`crate::sized::SIMD_WIDTH`'s own doc states the same constraint for the
/// hardware-fixed 32). `crate::metal::dispatch` already clamps
/// `grid.threadgroup_width` to the pipeline's real cap before dispatching,
/// so an emit-time cap above the true hardware limit is a wasted grid, not
/// a correctness hazard -- 256 (8 simdgroups) is conservative against every
/// Apple GPU family this crate targets.
#[cfg(feature = "metal-wide-cooperative-reduce")]
fn cooperative_reduce_width(
    resolved: &BoundOp,
    quantized: &[Option<PackedCodec>],
    reduce_dims: &[u16],
) -> u64 {
    if q4k_super_block_tiled(resolved, quantized, reduce_dims) {
        return SIMD_WIDTH;
    }
    let reduction_total: u64 = reduce_dims
        .iter()
        .map(|&dim| resolved.extents[dim as usize])
        .product();
    let quarter = reduction_total.div_ceil(4).max(1);
    quarter
        .next_multiple_of(SIMD_WIDTH)
        .clamp(SIMD_WIDTH, crate::sized::WIDE_COOPERATIVE_REDUCE_MAX_WIDTH)
}

/// Feature off: always `SIMD_WIDTH`, the byte-identical prior dispatch
/// shape -- see [`cooperative_reduce_width`]'s doc (the `metal-wide-
/// cooperative-reduce` arm) for the scaling this default-off build never
/// takes.
#[cfg(not(feature = "metal-wide-cooperative-reduce"))]
fn cooperative_reduce_width(
    _resolved: &BoundOp,
    _quantized: &[Option<PackedCodec>],
    _reduce_dims: &[u16],
) -> u64 {
    SIMD_WIDTH
}

/// [`tiled_gemm_threadgroup_width`]'s own nsg multiplier for the packed
/// row-blocked path -- `PACKED_ROW_NSG` with either nsg2 feature on and
/// `metal-q4k-split-k` off (see that call site's own doc for why split-K
/// must win when both are compiled in), `1` otherwise. Two functions, not a
/// `cfg!()` branch inline, so a feature-off build never references
/// `PACKED_ROW_NSG` from code it does not generate (mirrors
/// [`packed_row_split_factor`]'s own on/off pair).
#[cfg(all(
    any(feature = "metal-packed-row-nsg2", feature = "metal-q4k-ggml-port"),
    not(feature = "metal-q4k-split-k")
))]
fn packed_row_nsg_factor() -> u64 {
    PACKED_ROW_NSG as u64
}

/// The nsg2-features-off (or `metal-q4k-split-k`-on) arm: nsg widening never
/// engages, so the factor is always `1` -- see [`packed_row_nsg_factor`]'s
/// feature-on twin for the real policy.
#[cfg(not(all(
    any(feature = "metal-packed-row-nsg2", feature = "metal-q4k-ggml-port"),
    not(feature = "metal-q4k-split-k")
)))]
fn packed_row_nsg_factor() -> u64 {
    1
}

// the emitter threads a bound op's full shape (rank, axes, reduce op, init,
// element type, codec flags) into one kernel body; splitting that into a
// struct would relocate the arguments, not remove them.
#[allow(clippy::too_many_arguments)]
fn push_cooperative_reduce_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    reduce_dims: &[u16],
    rank: usize,
    quantized: &[Option<PackedCodec>],
    element_type: &str,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    is_broadcast_epilogue: bool,
    expert_source_mode: bool,
) -> Result<(), EmitError> {
    let rank_len = rank.max(1);
    let output_rank = output_axes.len();
    let output_rank_len = output_rank.max(1);
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);
    let operand_count = resolved.operands().len();
    let gather_slots = gather_slots(resolved);

    // the tiled GEMM path owns its own preamble entirely (`tiitg`/`sgitg`/
    // `tile_index`, derived straight from `gid` against `TILED_GEMM_NSG *
    // SIMD_WIDTH` threads per threadgroup, ROW 109) -- it needs neither
    // `output_index` nor `lane` the way the row-blocked path below does, see
    // `kernel_cache_key`'s own comment for why the two are mutually
    // exclusive by construction.
    if let Some(block) = tiled_gemm_block(resolved, quantized, reduce_op, init, output_axes) {
        push_tiled_gemm_body(
            source,
            resolved.node,
            output_axes,
            rank,
            &block,
            element_type,
        )?;
        return Ok(());
    }

    // the row-blocked packed path owns its own preamble: `output_index` is a
    // GROUP index there, not an output index, so the guard below would be
    // wrong for it. It always dispatches at SIMD_WIDTH regardless of
    // `metal-wide-cooperative-reduce` -- ROW-BLOCKED is a separate,
    // untouched investigation (see this file's own history), not the
    // reduction-extent-driven scaling below. Covers both the plain
    // single-activation-row shape and the multi-row fold
    // [`push_packed_row_blocked_body`]'s own `token_total > 1` branch emits
    // -- one preamble, since both branches dispatch the identical
    // `output_index`/`lane` pair at `SIMD_WIDTH` (`metal-q4k-split-k`'s own
    // `tptg`-derived preamble below is the only variant on this pair).
    if let Some(block) = packed_row_block(resolved, quantized) {
        if cfg!(feature = "metal-q4k-split-k") {
            // `tptg` is the ACTUAL per-dispatch threadgroup width
            // (`kernel_signature`'s new param, wired on for this exact
            // path -- see its call site's own gate). `split == 1` (the
            // feature-off-equivalent case) makes every line below collapse
            // to the plain `output_index`/`lane` pair the non-split-K arm
            // emits: `tptg_width == SIMD_WIDTH`, so `tiitg == gid % SIMD_WIDTH`
            // (today's `lane`), `sgitg == 0`, and `lane == tiitg` -- the same
            // value, same bits, same order.
            source.push_str("    uint tptg_width = tptg;\n");
            source.push_str("    long output_index = (long)gid / (long)tptg_width;\n");
            source.push_str("    uint tiitg = (uint)((long)gid % (long)tptg_width);\n");
            source.push_str(&format!("    uint sgitg = tiitg / {SIMD_WIDTH}u;\n"));
            source.push_str(&format!("    uint lane = tiitg % {SIMD_WIDTH}u;\n"));
            source.push_str(&format!("    uint split = tptg_width / {SIMD_WIDTH}u;\n"));
        } else {
            source.push_str(&format!(
                "    long output_index = (long)gid / {SIMD_WIDTH};\n"
            ));
            source.push_str(&format!("    uint lane = gid % {SIMD_WIDTH}u;\n"));
        }
        push_packed_row_blocked_body(
            source,
            resolved,
            reduce_op,
            init,
            output_axes,
            rank,
            quantized,
            element_type,
            &block,
            epilogue_body,
            epilogue_operands,
            expert_source_mode,
        )?;
        return Ok(());
    }

    // Single source of truth with the dispatch shape `grid_threads`/
    // `tiled_gemm_threadgroup_width` compute -- `width` here MUST equal
    // `cooperative_reduce_width`'s return for this exact op, or the grid
    // launched and the lane math emitted below disagree.
    let width = cooperative_reduce_width(resolved, quantized, reduce_dims);
    source.push_str(&format!("    long output_index = (long)gid / {width};\n"));
    source.push_str("    if (output_index >= u.output_total) { return; }\n");
    source.push_str(&format!("    uint lane = gid % {width}u;\n"));

    source.push_str(&format!("    long full_coord[{rank_len}];\n"));
    for dim in 0..rank {
        source.push_str(&format!("    full_coord[{dim}] = 0;\n"));
    }

    if output_rank > 0 {
        source.push_str(&format!("    long output_coord[{output_rank_len}];\n"));
        source.push_str("    long remaining = output_index;\n");
        for index in (0..output_rank).rev() {
            source.push_str(&format!(
                "    output_coord[{index}] = remaining % u.output_extents[{index}]; \
                 remaining /= u.output_extents[{index}];\n"
            ));
        }
        for (index, dim) in output_axes.iter().enumerate() {
            source.push_str(&format!("    full_coord[{dim}] = output_coord[{index}];\n"));
        }
    }

    let (init_expr, seeded_init) = fold_init_tokens(init);
    let identity = cooperative_identity_token(resolved.node, reduce_op)?;
    source.push_str(&format!("    {element_type} accumulator;\n"));
    source.push_str("    bool seeded;\n");
    source.push_str("    if (lane == 0u) {\n");
    source.push_str(&format!("        accumulator = {init_expr};\n"));
    source.push_str(&format!("        seeded = {seeded_init};\n"));
    source.push_str("    } else {\n");
    source.push_str(&format!("        accumulator = {identity};\n"));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    // A SINGLE reduction dim is the shape every matmul takes, and it makes
    // the whole per-element index computation redundant. `r` already IS the
    // reduction coordinate (`r < reduction_total == reduction_extents[0]`),
    // so the unflatten is an identity; and every operand's offset then
    // advances by a CONSTANT stride per step, so the base can be hoisted and
    // the step folded into one add.
    //
    // What the general path below costs per element, measured on the emitted
    // MSL: a 64-bit integer `%` and `/` against a runtime extent (Apple GPUs
    // have no integer divider — that is an emulated multi-instruction
    // sequence), a write into a thread-local `long` array, and `rank`
    // 64-bit multiply-adds per operand. For a 4096x4096 matvec that is all
    // of it: the probe measured 1.6 GB/s against llama.cpp Metal's 214.7.
    if reduce_rank == 1 {
        let reduce_dim = reduce_dims[0] as usize;
        // SUPER-BLOCK TILED PACKED READ. `q4k_element` derives `d`, `dmin` and
        // the 6-bit scale/min per ELEMENT, but all three are constant across a
        // 32-element sub-block, so the strided walk above pays that decode 256
        // times per super-block. Measured: packed marginal 12.3 GB/s = 21.9 G
        // elem/s against llama.cpp Metal's 381 G elem/s, while the f32 kernel on
        // the SAME loop hits 60.5 G elem/s reading 7.1x more bytes — Q4 was
        // compute-bound, not bandwidth-bound (`docs/discipline.md` ROW 72).
        //
        // Giving each lane a CONTIGUOUS run of `Q4K_BLOCK_ELEMENTS / SIMD_WIDTH`
        // elements keeps that run inside one sub-block (lane*8 .. lane*8+7 never
        // crosses a 32 boundary), so the header decodes once per run. Same shape
        // as ggml's `for (short i = 0; i < 8; ++i)`.
        //
        // Requires: exactly one packed operand, contiguous along the reduction
        // dim, and a reduction extent that is a whole number of super-blocks —
        // all known here, from the bound layout, not at runtime.
        // Q4_K-only: the body below calls `q4k_header_for`/`q4k_value` by name,
        // so this fallback requires the packed operand specifically to be that
        // codec — a `Q6_K` operand that somehow reaches here (it never does in
        // practice: `packed_row_block` above already claims every real
        // `Q6_K` matmul this repo's checkpoint carries) falls through to the
        // fully generic scalar path below instead of emitting the wrong codec's
        // unpack call.
        let packed: Vec<usize> = quantized
            .iter()
            .enumerate()
            .filter_map(|(index, codec)| matches!(codec, Some(PackedCodec::Q4K)).then_some(index))
            .collect();
        let run = Q4K_BLOCK_ELEMENTS / SIMD_WIDTH as usize;
        // The broadcast-write tail (`push_cooperative_reduce_tail`'s own doc)
        // only ever walks the plain per-lane strided loop below -- the Q4K
        // super-block-tiled specialization is Q4K-only (packed weight
        // operand), never the shape a broadcast-reduce epilogue's f32
        // activation fold takes, and `render_reduce`'s own gate already
        // rejects a packed-row-block match, so this stays `false` in
        // practice; forced off here rather than relied upon so a future
        // packed operand slipping past that gate still falls through to the
        // supported loop instead of silently skipping the epilogue write.
        let tiled =
            !is_broadcast_epilogue && q4k_super_block_tiled(resolved, quantized, reduce_dims);
        if tiled {
            let weight = packed[0];
            for index in 0..operand_count {
                source.push_str(&format!(
                    "    long base{index} = u.operand_base[{index}];\n"
                ));
                for dim in 0..rank {
                    if dim == reduce_dim {
                        continue;
                    }
                    source.push_str(&format!(
                    "    base{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
                ));
                }
                if index != weight {
                    source.push_str(&format!(
                        "    long stride{index} = u.operand_strides[{index}][{reduce_dim}];\n"
                    ));
                }
                if let Some(slot) = gather_slots[index] {
                    push_cooperative_gather_fetch(
                        source,
                        index,
                        slot,
                        rank,
                        "full_coord",
                        &format!("base{index}"),
                    );
                }
            }
            source.push_str(&format!("    uint slot = (uint)lane * {run}u;\n"));
            source.push_str(&format!(
            "    for (int block_start = 0; block_start < (int)u.reduction_total; block_start += {Q4K_BLOCK_ELEMENTS}) {{\n"
        ));
            source.push_str(&format!(
            "        device const uchar *blk = in{weight} + (((int)base{weight} + block_start) / {Q4K_BLOCK_ELEMENTS}) * {Q4K_BLOCK_BYTES};\n"
        ));
            source.push_str("        q4k_header hdr = q4k_header_for(blk, slot);\n");
            source.push_str(&format!("        for (int j = 0; j < {run}; ++j) {{\n"));
            source.push_str(&format!(
                "            {element_type} scratch[{}];\n",
                operand_count.max(1)
            ));
            source.push_str(&format!(
                "            scratch[{weight}] = q4k_value(blk, slot + (uint)j, hdr);\n"
            ));
            for index in 0..operand_count {
                if index == weight {
                    continue;
                }
                source.push_str(&format!(
                "            scratch[{index}] = in{index}[base{index} + (long)(block_start + (int)slot + j) * stride{index}];\n"
            ));
            }
            let value_expr = push_body_steps(
                source,
                resolved.element_body(),
                "            ",
                element_type,
            );
            source.push_str(&format!(
                "            {element_type} value = {value_expr};\n"
            ));
            let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
            source.push_str(&format!(
                "            accumulator = seeded ? {combine_expr} : value;\n"
            ));
            source.push_str("            seeded = true;\n");
            source.push_str("        }\n");
            source.push_str("    }\n");
            push_cooperative_reduce_tail(
                source,
                resolved.node,
                reduce_op,
                rank,
                width,
                element_type,
                output_rank,
                epilogue_body,
                epilogue_operands,
                reduce_dims,
                is_broadcast_epilogue,
            )?;
            return Ok(());
        }

        for index in 0..operand_count {
            source.push_str(&format!(
                "    long stride{index} = u.operand_strides[{index}][{reduce_dim}];\n"
            ));
            source.push_str(&format!("    long off{index} = u.operand_base[{index}];\n"));
            for dim in 0..rank {
                if dim == reduce_dim {
                    continue;
                }
                source.push_str(&format!(
                    "    off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
                ));
            }
            if let Some(slot) = gather_slots[index] {
                push_cooperative_gather_fetch(
                    source,
                    index,
                    slot,
                    rank,
                    "full_coord",
                    &format!("off{index}"),
                );
            }
            source.push_str(&format!("    off{index} += (long)lane * stride{index};\n"));
            // 32-bit from here down. The offsets ABOVE stay `long` because a
            // layout base can legitimately be one; the per-element WALK never
            // needs that range, and Apple GPUs are 32-bit machines where
            // 64-bit integer arithmetic is emulated. `u.walk_fits_int` is the
            // runtime guard — when an operand's span really does exceed
            // `int`, the 64-bit walk below runs instead.
            source.push_str(&format!("    int walk{index} = (int)off{index};\n"));
            source.push_str(&format!(
                "    int advance{index} = (int)(stride{index} * {width});\n"
            ));
        }
        source.push_str(&format!(
            "    for (int r = (int)lane; r < (int)u.reduction_total; r += {width}) {{\n"
        ));
        source.push_str(&format!(
            "        {element_type} scratch[{}];\n",
            operand_count.max(1)
        ));
        for (index, &codec) in quantized.iter().enumerate() {
            source.push_str(&format!(
                "        scratch[{index}] = {};\n",
                operand_read(index, &format!("walk{index}"), codec)
            ));
        }
        let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
        source.push_str(&format!("        {element_type} value = {value_expr};\n"));
        let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
        source.push_str(&format!(
            "        accumulator = seeded ? {combine_expr} : value;\n"
        ));
        source.push_str("        seeded = true;\n");
        for index in 0..operand_count {
            source.push_str(&format!("        walk{index} += advance{index};\n"));
        }
        source.push_str("    }\n");
        push_cooperative_reduce_tail(
            source,
            resolved.node,
            reduce_op,
            rank,
            width,
            element_type,
            output_rank,
            epilogue_body,
            epilogue_operands,
            reduce_dims,
            is_broadcast_epilogue,
        )?;
        return Ok(());
    }

    source.push_str(&format!(
        "    for (long r = (long)lane; r < u.reduction_total; r += {width}) {{\n"
    ));
    if reduce_rank > 0 {
        source.push_str(&format!(
            "        long reduction_coord[{reduce_rank_len}];\n"
        ));
        source.push_str("        long remaining_r = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r /= u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!(
                "        full_coord[{dim}] = reduction_coord[{index}];\n"
            ));
        }
    }

    for index in 0..operand_count {
        source.push_str(&format!(
            "        long off{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..rank {
            source.push_str(&format!(
                "        off{index} += full_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slots[index] {
            push_cooperative_gather_fetch(
                source,
                index,
                slot,
                rank,
                "full_coord",
                &format!("off{index}"),
            );
        }
    }
    source.push_str(&format!(
        "        {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, &codec) in quantized.iter().enumerate() {
        source.push_str(&format!(
            "        scratch[{index}] = {};\n",
            operand_read(index, &format!("off{index}"), codec)
        ));
    }
    let value_expr = push_body_steps(source, resolved.element_body(), "        ", element_type);
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    source.push_str("    }\n");

    push_cooperative_reduce_tail(
        source,
        resolved.node,
        reduce_op,
        rank,
        width,
        element_type,
        output_rank,
        epilogue_body,
        epilogue_operands,
        reduce_dims,
        is_broadcast_epilogue,
    )?;
    Ok(())
}

/// The per-lane `simd_sum` fold and the final store both cooperative loop
/// shapes end with — shared so the strength-reduced single-reduction-dim
/// path, the general path, and the Q4_K super-block-tiled path (which
/// always calls this with `width == SIMD_WIDTH`, see
/// [`q4k_super_block_tiled`]) cannot drift on how the result is written
/// out.
///
/// `width == SIMD_WIDTH` (32, one simdgroup, the byte-identical prior
/// shape): a single `simd_sum`-class fold and a lane-0 store, unchanged.
///
/// `width > SIMD_WIDTH` (`metal-wide-cooperative-reduce` only --
/// [`cooperative_reduce_width`] never returns a wider value with the
/// feature off): a two-level fold. Each simdgroup folds its own 32 lanes
/// with `simd_combine_fn`, its lane 0 stores that partial into a
/// `threadgroup` array sized to the EXACT simdgroup count this kernel
/// dispatches (`width / SIMD_WIDTH`, baked into the source as a literal --
/// not a uniform, so there is no way to index it out of bounds or read an
/// element no lane wrote). A barrier orders the writes before thread 0
/// folds the partials serially and stores the result. Every lane in every
/// simdgroup of a `width`-wide threadgroup is real (the grid this pairs
/// with is always an exact multiple of `width`, `grid_threads`' own
/// invariant), so every partial slot is written before the fold reads it —
/// there is no ragged-tail case here to guard, unlike the per-lane
/// accumulator seed above (which already handles `reduction_total < width`
/// via `cooperative_identity_token`, both before and after this feature).
#[allow(clippy::too_many_arguments)]
fn push_cooperative_reduce_tail(
    source: &mut String,
    node: NodeId,
    reduce_op: ScalarOp,
    rank: usize,
    width: u64,
    element_type: &str,
    output_rank: usize,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    reduce_dims: &[u16],
    is_broadcast_epilogue: bool,
) -> Result<(), EmitError> {
    let combine_fn = simd_combine_fn(node, reduce_op)?;
    let simdgroups = width / SIMD_WIDTH;
    let coord = |dim: usize| {
        if output_rank > 0 {
            format!("output_coord[{dim}]")
        } else {
            "0".to_string()
        }
    };
    if simdgroups <= 1 {
        // `combine_fn` is a `simd_*` reduction (`simd_combine_fn`'s own
        // doc): Metal broadcasts its result to every lane of the simdgroup
        // already, so a broadcast-reduce epilogue needs no extra barrier
        // here to make `reduced` visible everywhere `push_broadcast_
        // epilogue_write` reads it from.
        source.push_str(&format!(
            "    {element_type} reduced = {combine_fn}(accumulator);\n"
        ));
        if is_broadcast_epilogue {
            push_broadcast_epilogue_write(
                source,
                rank,
                reduce_dims,
                width,
                epilogue_body,
                epilogue_operands,
                element_type,
                "reduced",
            );
            return Ok(());
        }
        source.push_str("    if (lane == 0u) {\n");
        source.push_str("        long out_offset = u.out_base;\n");
        for dim in 0..rank {
            source.push_str(&format!(
                "        out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
            ));
        }
        push_reduce_epilogue_write(
            source,
            epilogue_body,
            epilogue_operands,
            output_rank,
            element_type,
            "        ",
            coord,
            "reduced",
            "out_offset",
        );
        source.push_str("    }\n");
        return Ok(());
    }

    source.push_str(&format!(
        "    {element_type} partial = {combine_fn}(accumulator);\n"
    ));
    source.push_str(&format!(
        "    threadgroup {element_type} partials[{simdgroups}];\n"
    ));
    source.push_str(&format!(
        "    if (lane % {SIMD_WIDTH}u == 0u) {{ partials[lane / {SIMD_WIDTH}u] = partial; }}\n"
    ));
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    if is_broadcast_epilogue {
        // Every lane needs `reduced`, not just lane 0 -- fold on lane 0 same
        // as below, publish it back through `partials[0]`, and a second
        // barrier before every lane reads it, matching llama.cpp's own
        // `kernel_rms_norm` shape (`ggml-metal.metal`: cooperative fold,
        // scalar broadcast through shared memory, every lane writes its own
        // elements).
        source.push_str("    if (lane == 0u) {\n");
        source.push_str(&format!("        {element_type} reduced = partials[0];\n"));
        source.push_str(&format!(
            "        for (uint fold_index = 1u; fold_index < {simdgroups}u; ++fold_index) {{\n"
        ));
        let fold_expr = scalar_op_expr(reduce_op, &["reduced", "partials[fold_index]"]);
        source.push_str(&format!("            reduced = {fold_expr};\n"));
        source.push_str("        }\n");
        source.push_str("        partials[0] = reduced;\n");
        source.push_str("    }\n");
        source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
        source.push_str(&format!("    {element_type} reduced = partials[0];\n"));
        push_broadcast_epilogue_write(
            source,
            rank,
            reduce_dims,
            width,
            epilogue_body,
            epilogue_operands,
            element_type,
            "reduced",
        );
        return Ok(());
    }
    source.push_str("    if (lane == 0u) {\n");
    source.push_str(&format!("        {element_type} reduced = partials[0];\n"));
    source.push_str(&format!(
        "        for (uint fold_index = 1u; fold_index < {simdgroups}u; ++fold_index) {{\n"
    ));
    let fold_expr = scalar_op_expr(reduce_op, &["reduced", "partials[fold_index]"]);
    source.push_str(&format!("            reduced = {fold_expr};\n"));
    source.push_str("        }\n");
    source.push_str("        long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "        out_offset += full_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }
    push_reduce_epilogue_write(
        source,
        epilogue_body,
        epilogue_operands,
        output_rank,
        element_type,
        "        ",
        coord,
        "reduced",
        "out_offset",
    );
    source.push_str("    }\n");
    Ok(())
}

/// The broadcast-reduce epilogue's own write tail
/// ([`BoundOpKind::Reduce::epilogue_broadcast_axes`]'s own doc): once the
/// fold's scalar (`reduced_expr`, already visible to every lane -- see each
/// [`push_cooperative_reduce_tail`] call site) is available, every lane
/// re-walks the SAME per-lane strided range over the reduction dims the
/// accumulation loop above just folded (`bind::epilogue_broadcast_axes_for`'s
/// own doc: a non-empty value is always exactly this fold's `reduce_dims`,
/// nothing else -- `render_reduce`'s own gate rejects anything that would
/// disagree), this time writing one output element per step instead of
/// reading one. Same shape llama.cpp's `kernel_rms_norm` takes (`ggml-metal.
/// metal`): a cooperative fold, then every thread writes its own share of
/// the row. `epilogue_operand_strides`/`broadcast_out_strides` are the two
/// uniform rows [`pack_reduce_uniforms`] packs at FULL rank for exactly this
/// loop -- `output_rank` handed to [`push_reduce_epilogue_write`] here is
/// `rank` itself (the widened space), and `coord` reads `full_coord[dim]`
/// rather than `output_coord[dim]`.
#[allow(clippy::too_many_arguments)]
fn push_broadcast_epilogue_write(
    source: &mut String,
    rank: usize,
    reduce_dims: &[u16],
    width: u64,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    element_type: &str,
    reduced_expr: &str,
) {
    let reduce_rank = reduce_dims.len();
    let reduce_rank_len = reduce_rank.max(1);

    // ROW 370: the fold scalar (`reduced_expr`) is the same value on every
    // pass of the loop below -- any epilogue step that only ever reads it
    // (mean_square + eps, sqrt, reciprocal for rmsnorm) is loop-invariant and
    // is declared exactly once here, before the loop, instead of being
    // re-derived on all 16 iterations a [1,4096] row takes at width 256.
    let epilogue_operand_count = epilogue_operands.len();
    let is_identity = reduce_epilogue_is_identity(epilogue_body, epilogue_operands);
    let is_invariant_operand = |operand_index: usize| {
        operand_index == epilogue_operand_count
            || epilogue_operand_is_loop_invariant(epilogue_operands, reduce_dims, operand_index)
    };
    let mut element_body = String::new();
    let epi_value = if is_identity {
        reduced_expr.to_string()
    } else {
        source.push_str(&format!(
            "    {element_type} epi_scratch[{}];\n",
            epilogue_operand_count + 1
        ));
        push_epilogue_operand_reads(
            source,
            (0..epilogue_operand_count).filter(|&index| is_invariant_operand(index)),
            rank,
            "    ",
            |dim| format!("full_coord[{dim}]"),
        );
        source.push_str(&format!(
            "    epi_scratch[{epilogue_operand_count}] = {reduced_expr};\n"
        ));
        crate::epilogue::declare_steps_partitioned(
            source,
            &mut element_body,
            epilogue_body,
            "epi_scratch",
            "epi_step",
            is_invariant_operand,
            scalar_op_expr,
            |source, index, expr| {
                source.push_str(&format!("    {element_type} epi_step{index} = {expr};\n"));
            },
            |source, index, expr| {
                source.push_str(&format!(
                    "        {element_type} epi_step{index} = {expr};\n"
                ));
            },
        )
    };

    source.push_str(&format!(
        "    for (long r = (long)lane; r < u.reduction_total; r += {width}) {{\n"
    ));
    if reduce_rank > 0 {
        source.push_str(&format!(
            "        long reduction_coord[{reduce_rank_len}];\n"
        ));
        source.push_str("        long remaining_r = r;\n");
        for index in (0..reduce_rank).rev() {
            source.push_str(&format!(
                "        reduction_coord[{index}] = remaining_r % u.reduction_extents[{index}]; \
                 remaining_r /= u.reduction_extents[{index}];\n"
            ));
        }
        for (index, dim) in reduce_dims.iter().enumerate() {
            source.push_str(&format!(
                "        full_coord[{dim}] = reduction_coord[{index}];\n"
            ));
        }
    }
    source.push_str("        long out_offset = u.out_base;\n");
    for dim in 0..rank {
        source.push_str(&format!(
            "        out_offset += full_coord[{dim}] * u.broadcast_out_strides[{dim}];\n"
        ));
    }
    if is_identity {
        source.push_str(&format!("        out[out_offset] = {epi_value};\n"));
    } else {
        push_epilogue_operand_reads(
            source,
            (0..epilogue_operand_count).filter(|&index| !is_invariant_operand(index)),
            rank,
            "        ",
            |dim| format!("full_coord[{dim}]"),
        );
        source.push_str(&element_body);
        source.push_str(&format!("        out[out_offset] = {epi_value};\n"));
    }
    source.push_str("    }\n");
}

fn render_scan(
    resolved: &BoundOp,
    entry: &str,
    quantized: &[Option<PackedCodec>],
) -> Result<String, EmitError> {
    let BoundOpKind::Reduce {
        reduce_op, init, ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "keep::scan fold",
            found: resolved.kind.name(),
        });
    };
    let rank = resolved.extents.len();
    let rank_len = rank.max(1);
    let outer_rank = rank.saturating_sub(1);
    let outer_rank_len = outer_rank.max(1);
    let last_dim = rank.saturating_sub(1);
    let operand_count = resolved.operands().len();
    let gather_count = gather_count(resolved);
    let gather_slots = gather_slots(resolved);
    let element_type = type_token(resolved.node, resolved.dtype)?;

    let mut source = String::new();
    preamble(&mut source);

    source.push_str("struct Uniforms {\n");
    source.push_str("    long outer_total;\n");
    source.push_str("    long inner_len;\n");
    source.push_str(&format!("    long outer_extents[{outer_rank_len}];\n"));
    source.push_str(&format!("    long operand_base[{operand_count}];\n"));
    source.push_str(&format!(
        "    long operand_strides[{operand_count}][{rank_len}];\n"
    ));
    source.push_str("    long out_base;\n");
    source.push_str(&format!("    long out_strides[{rank_len}];\n"));
    push_gather_uniform_fields(&mut source, gather_count, rank_len);
    source.push_str("};\n\n");

    kernel_signature(
        &mut source,
        quantized,
        0,
        gather_count,
        entry,
        element_type,
        false,
    );
    source.push_str("    if ((long)gid >= u.outer_total) { return; }\n");

    if outer_rank > 0 {
        source.push_str(&format!("    long outer_coord[{outer_rank_len}];\n"));
        source.push_str("    long remaining = (long)gid;\n");
        for dim in (0..outer_rank).rev() {
            source.push_str(&format!(
                "    outer_coord[{dim}] = remaining % u.outer_extents[{dim}]; \
                 remaining /= u.outer_extents[{dim}];\n"
            ));
        }
    }

    for (index, gather_slot) in gather_slots.iter().enumerate() {
        source.push_str(&format!(
            "    long running{index} = u.operand_base[{index}];\n"
        ));
        for dim in 0..outer_rank {
            source.push_str(&format!(
                "    running{index} += outer_coord[{dim}] * u.operand_strides[{index}][{dim}];\n"
            ));
        }
        if let Some(slot) = gather_slot {
            source.push_str(&format!(
                "    long gather_running{index} = u.gather_index_base[{slot}];\n"
            ));
            for dim in 0..outer_rank {
                source.push_str(&format!(
                    "    gather_running{index} += outer_coord[{dim}] * u.gather_index_strides[{slot}][{dim}];\n"
                ));
            }
        }
    }
    source.push_str("    long out_running = u.out_base;\n");
    for dim in 0..outer_rank {
        source.push_str(&format!(
            "    out_running += outer_coord[{dim}] * u.out_strides[{dim}];\n"
        ));
    }

    let (init_expr, seeded_init) = fold_init_tokens(*init);
    source.push_str(&format!("    {element_type} accumulator = {init_expr};\n"));
    source.push_str(&format!("    bool seeded = {seeded_init};\n"));

    source.push_str("    for (long step = 0; step < u.inner_len; step++) {\n");
    source.push_str(&format!(
        "        {element_type} scratch[{}];\n",
        operand_count.max(1)
    ));
    for (index, gather_slot) in gather_slots.iter().enumerate() {
        // the gathered dim's contribution is per-step (the fetched index
        // varies along the scanned dim too, in general), so it is combined
        // into a fresh `read_off` here rather than folded permanently into
        // `running{index}`, which must keep advancing by its own stride
        // alone — see the module doc's Uniforms-packing note for why.
        if let Some(slot) = gather_slot {
            source.push_str(&format!(
                "        long fetched{index} = (long)gather_idx{slot}[gather_running{index}];\n"
            ));
            push_gather_fault_check(&mut source, index, *slot, "        ");
            source.push_str(&format!(
                "        fetched{index} = max((long)0, min(fetched{index}, u.gather_extent[{slot}] - 1));\n"
            ));
            source.push_str(&format!(
                "        long read_off{index} = running{index} + fetched{index} * u.gather_element_stride[{slot}];\n"
            ));
            source.push_str(&format!(
                "        scratch[{index}] = {};\n",
                operand_read(index, &format!("read_off{index}"), quantized[index])
            ));
            source.push_str(&format!(
                "        gather_running{index} += u.gather_index_strides[{slot}][{last_dim}];\n"
            ));
        } else {
            source.push_str(&format!(
                "        scratch[{index}] = {};\n",
                operand_read(index, &format!("running{index}"), quantized[index])
            ));
        }
        source.push_str(&format!(
            "        running{index} += u.operand_strides[{index}][{last_dim}];\n"
        ));
    }
    let value_expr = push_body_steps(
        &mut source,
        resolved.element_body(),
        "        ",
        element_type,
    );
    source.push_str(&format!("        {element_type} value = {value_expr};\n"));
    let combine_expr = scalar_op_expr(*reduce_op, &["accumulator", "value"]);
    source.push_str(&format!(
        "        accumulator = seeded ? {combine_expr} : value;\n"
    ));
    source.push_str("        seeded = true;\n");
    source.push_str("        out[out_running] = accumulator;\n");
    source.push_str(&format!(
        "        out_running += u.out_strides[{last_dim}];\n"
    ));
    source.push_str("    }\n");
    source.push_str("}\n");
    Ok(source)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::collections::BTreeSet;
    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_tensor::{
        AxisTerm, DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, bind,
        infer, map,
    };

    use super::*;

    #[test]
    fn dense_layout_accepts_contiguous_and_rejects_broadcast() {
        let dense = Layout {
            base: 0,
            strides: vec![8_i64, 4, 1].into(),
        };
        assert!(dense_layout(&dense, &[2, 2, 4]));

        let broadcast = Layout {
            base: 0,
            strides: vec![0_i64, 4, 1].into(),
        };
        assert!(!dense_layout(&broadcast, &[2, 2, 4]));
    }

    #[test]
    fn elementwise_broadcast_decodes_omitted_axes_before_selected_axes() {
        let mut program = Vec::new();
        let matrix = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(7), Extent::Static(256)],
                name: None,
            },
        );
        let row = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(7)],
                name: None,
            },
        );
        let equal = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Equal,
                operands: vec![
                    (matrix, IndexMap::Affine(map::projection(2, &[0, 1]))),
                    (row, IndexMap::Affine(map::projection(2, &[0]))),
                ],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("broadcast elementwise infers");
        let bound = bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("broadcast elementwise lowers")
            .into_iter()
            .find(|candidate| candidate.node == equal)
            .expect("broadcast elementwise bound op");
        let source = render_elementwise(&bound, "broadcast_test", &[None, None])
            .expect("broadcast elementwise renders");
        let divide_axis_one = "remaining /= (uint)u.extents[1];";
        let assign_axis_zero = "coord[0] = remaining % (uint)u.extents[0];";
        assert!(source.find(divide_axis_one) < source.find(assign_axis_zero));
    }

    fn elementwise_tanh_op(extent: u32) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("elementwise infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("elementwise lowers")
            .into_iter()
            .next()
            .expect("one bound emitted")
    }

    fn matmul_op(m: u32, k: u32, n: u32) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(m), Extent::Static(k)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k), Extent::Static(n)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("matmul infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    fn gathered_matmul_op(tokens: u32, experts: u32, rows: u32, k: u32) -> BoundOp {
        let mut program = Vec::new();
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![
                    Extent::Static(experts),
                    Extent::Static(rows),
                    Extent::Static(k),
                ],
                name: None,
            },
        );
        let route = append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: vec![Extent::Static(tokens)],
                name: None,
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(tokens), Extent::Static(k)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (
                        weight,
                        IndexMap::Computed {
                            indices: route,
                            index_map: map::projection(3, &[0]),
                            base: map::IndexPattern {
                                iter_rank: 3,
                                axes: vec![
                                    map::AxisIndex::default(),
                                    map::AxisIndex {
                                        terms: core::iter::once(AxisTerm::projection(1)).collect(),
                                        offset: 0,
                                        len: None,
                                    },
                                    map::AxisIndex {
                                        terms: core::iter::once(AxisTerm::projection(2)).collect(),
                                        offset: 0,
                                        len: None,
                                    },
                                ],
                            },
                            gathered_dim: 0,
                        },
                    ),
                    (activation, IndexMap::Affine(map::projection(3, &[0, 2]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let shapes = infer(&program, &[]).expect("gathered matmul infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("gathered matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    #[expect(
        dead_code,
        reason = "fixture retained for the rejected physical-axis permutation case"
    )]
    fn permuted_reduction_matmul_op() -> BoundOp {
        let mut program = Vec::new();
        let gated = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(64), Extent::Static(2), Extent::Static(2)],
                name: None,
            },
        );
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(3), Extent::Static(256)],
                name: None,
            },
        );
        let product = proxima_tensor::spec::elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(gated, "jug->jugd"), (weight, "d,4*j+2*u+g->jugd")],
        )
        .expect("permuted product builds");
        proxima_tensor::spec::reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            product,
            "jugd->jugd",
            "d->jugd",
        )
        .expect("permuted reduce builds");
        let shapes = infer(&program, &[]).expect("permuted reduction matmul infers");
        let mut resolved = bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("permuted reduction matmul lowers");
        let mut packed = BTreeSet::new();
        packed.insert(weight);
        proxima_tensor::correct_packed_matmul_layouts(&mut resolved, &packed);
        resolved
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    fn cached_attention_op() -> BoundOp {
        let operands = (0..8)
            .map(|index| {
                (
                    NodeId(index),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    None,
                )
            })
            .collect();
        BoundOp {
            node: NodeId(8),
            dtype: DType::Float32,
            extents: vec![1, 1, 1, 4],
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows: 1,
                cached_key_rows: 1,
                new_key_rows: 1,
                kv_heads: 1,
                query_groups: 1,
                head_dim: 4,
                scale: 0.5,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        }
    }

    /// The single-range fused (nine-operand, `single_range_dynamic`) sibling
    /// of [`cached_attention_op`] -- the ninth operand carries the live
    /// `cached_len` at run time (`BoundOpKind::CachedAttention`'s own doc),
    /// and that shape is real, structural `cached_key_rows == 0`
    /// (`cached_attention_single_range_candidates`'s own bind-time
    /// construction folds the WHOLE context into the "new" slot). The two
    /// `u64` parameters here are kept as the COMPILED `kv-capacity-bucket`
    /// extent's two halves for every existing call site's own doc/comments
    /// ("capacity N") to stay accurate, but both fold into `new_key_rows`
    /// alone so the constructed op matches the real single-range invariant
    /// -- a test can still vary their SUM freely to prove `entry_name`/the
    /// rendered body do not key on it (redesign §5 option 2), since neither
    /// field is ever baked as a literal on this path (`row_count_decl`'s own
    /// `single_range_dynamic` arm reads both off `u.cached_key_rows`/
    /// `u.new_key_rows` at run time instead).
    fn cached_attention_op_dynamic(cached_key_rows: u64, new_key_rows: u64) -> BoundOp {
        let operands = (0..9)
            .map(|index| {
                (
                    NodeId(index),
                    Layout {
                        base: 0,
                        strides: vec![1].into(),
                    },
                    None,
                )
            })
            .collect();
        BoundOp {
            node: NodeId(9),
            dtype: DType::Float32,
            extents: vec![1, 1, 1, 4],
            kind: BoundOpKind::CachedAttention {
                operands,
                query_rows: 1,
                cached_key_rows: 0,
                new_key_rows: cached_key_rows + new_key_rows,
                kv_heads: 1,
                query_groups: 1,
                head_dim: 4,
                scale: 0.5,
                cached_lower_inclusive: i64::MIN,
                new_upper_inclusive: 0,
            },
        }
    }

    /// Same shape as [`matmul_op`] but with a caller-chosen reduce op, so a
    /// test can hold the fused `weight * activation` body fixed and vary only
    /// `reduce_op` — the one axis [`is_plain_product_reduce`] gates on beyond
    /// the body shape itself.
    fn matmul_op_with_reduce(m: u32, k: u32, n: u32, reduce_op: ScalarOp) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(m), Extent::Static(k)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k), Extent::Static(n)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: reduce_op,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul_reduce".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("matmul infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    /// [`matmul_op`]'s `Float16` counterpart -- `q4k_pair_dot`'s own
    /// `plain_product` arm (`push_packed_row_blocked_body`'s own gate) is
    /// `DType::Float32`-only, so a fixture that needs to reach a
    /// DIFFERENT row-blocked Q4_K arm (mask-fma's, single-fetch's) must NOT
    /// be plain-`Float32`-shaped, or `q4k_pair_dot` wins over it every time.
    #[cfg(all(feature = "metal-q4k-single-fetch", not(feature = "metal-q4k-split-k")))]
    fn matmul_op_f16(m: u32, k: u32, n: u32) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float16,
                shape: vec![Extent::Static(m), Extent::Static(k)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float16,
                shape: vec![Extent::Static(k), Extent::Static(n)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float16,
                body: ScalarOp::Multiply,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float16,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul_f16".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("f16 matmul infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("f16 matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    #[test]
    fn q4k_row_blocked_matmul_uses_paired_nibble_decode() {
        // 256 == Q4K_BLOCK_ELEMENTS exactly: one super-block, so
        // packed_row_block matches and this is the real matmul shape the
        // the paired decode path exists for (`docs/discipline.md` ROW 257).
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        assert!(
            packed_row_block(&bound, &operand_codecs(&bound, &q4k)).is_some(),
            "test fixture must actually take the row-blocked path for this assertion to mean anything"
        );

        let source = emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            source.contains("q4k_pair_dot"),
            "Add-reduce over a plain weight*activation body must take the paired decode path:\n{source}"
        );
        assert!(
            !source.contains("hdr.scale * levels[j] - hdr.minimum"),
            "the per-element dequant expression must not remain once the scale-deferred path is taken:\n{source}"
        );
    }

    /// `Q5_K` sibling of the test above: the same Add-reduce-over-plain-
    /// product shape must select `q5k_pair_dot` (`PackedCodec::supports_pair_dot`,
    /// a structural fact of `Q5_K`'s block layout, not a cargo feature)
    /// rather than the scalar per-element `q5k_value` loop
    /// `push_packed_row_blocked_body`'s `PackedCodec::Q5K` arm falls back to
    /// when the reduce is not a plain product.
    #[test]
    fn q5k_row_blocked_matmul_uses_paired_nibble_decode() {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q5k = BTreeMap::new();
        q5k.insert(weight_node, PackedCodec::Q5K);

        assert!(
            packed_row_block(&bound, &operand_codecs(&bound, &q5k)).is_some(),
            "test fixture must actually take the row-blocked path for this assertion to mean anything"
        );

        let source = emit(&bound, &q5k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            source.contains("q5k_pair_dot"),
            "Add-reduce over a plain weight*activation body must take the paired decode path by default:\n{source}"
        );
        assert!(
            !source.contains("q5k_value(blk"),
            "the scalar per-element q5k_value dequant expression must not remain once the paired path is taken:\n{source}"
        );
    }

    /// `Q6_K` sibling of `q5k_row_blocked_matmul_uses_paired_nibble_decode`:
    /// the same Add-reduce-over-plain-product shape must select
    /// `q6k_pair_dot` (`PackedCodec::supports_pair_dot`, a structural fact
    /// of `Q6_K`'s block layout, not a cargo feature) rather than the
    /// scalar per-element `q6k_value` loop `push_packed_row_blocked_body`'s
    /// `PackedCodec::Q6K` arm falls back to when the reduce is not a plain
    /// product. This subsumes the pair of feature-gated marker tests this
    /// landing replaced (`q6k_row_blocked_matmul_uses_paired_nibble_decode`/
    /// `_uses_scalar_decode_by_default`) -- there is now exactly one
    /// selection to assert, not a feature-on/feature-off pair.
    #[test]
    fn q6k_row_blocked_matmul_uses_paired_nibble_decode() {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q6k = BTreeMap::new();
        q6k.insert(weight_node, PackedCodec::Q6K);

        assert!(
            packed_row_block(&bound, &operand_codecs(&bound, &q6k)).is_some(),
            "test fixture must actually take the row-blocked path for this assertion to mean anything"
        );

        let source = emit(&bound, &q6k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            source.contains("q6k_pair_dot"),
            "Add-reduce over a plain weight*activation body must take the paired decode path by default:\n{source}"
        );
        assert!(
            !source.contains("q6k_value(blk"),
            "the scalar per-element q6k_value dequant expression must not remain once the paired path is taken:\n{source}"
        );
    }

    #[test]
    fn one_token_gathered_q4k_matmul_uses_row_blocked_gather_body() {
        let bound = gathered_matmul_op(1, 3, 4, 256);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);
        let codecs = operand_codecs(&bound, &q4k);

        assert!(
            classify_packed_row_block(&bound, &codecs).is_ok(),
            "one routed activation row selects one expert before the row-blocked reduction"
        );

        let source = emit(&bound, &q4k, NumericPolicy::default())
            .expect("gathered q4k matmul emits")
            .source;
        assert!(
            source.contains("gather_idx0")
                && source.contains("weight_base[q] += fetched0 * u.gather_element_stride[0]")
                && source.contains("q4k_pair_dot"),
            "the row-blocked body must select the routed expert slab before its paired q4k decode:\n{source}"
        );
    }

    #[test]
    fn expert_source_uses_descriptor_codec_instead_of_checkpoint_codec() {
        let bound = gathered_matmul_op(1, 3, 4, 256);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);
        let kernel = emit_with_expert_sources(&bound, &q4k, NumericPolicy::default(), weight_node)
            .expect("emits expert source kernel");
        assert!(kernel.source.contains("mixed_expert_element_from_offset"));
        assert!(
            kernel
                .source
                .contains("mixed_expert_element_from_local(expert_base0, expert_codec0, walk0)"),
            "the selected descriptor must be hoisted into the hot loop:\n{}",
            kernel.source
        );
        assert!(
            kernel.source.matches("q4k_pair_dot(").count() == 1,
            "a mixed source must not retain the checkpoint's fixed Q4_K decoder:\n{}",
            kernel.source
        );
        assert!(!kernel.source.contains("q4k_element(in0"));
        assert!(!kernel.source.contains("q6k_element(in0"));
        if std::env::var_os("PROXIMA_DUMP_EXPERT_TEST_SOURCE").is_some() {
            eprintln!("{}", kernel.source);
        }
    }

    #[test]
    fn uniform_expert_source_retains_packed_row_decoder() {
        let bound = gathered_matmul_op(1, 3, 4, 256);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);
        let kernel = emit_with_uniform_expert_source(
            &bound,
            &q4k,
            NumericPolicy::default(),
            weight_node,
            PackedCodec::Q4K,
        )
        .expect("emits uniform expert source kernel");
        assert!(kernel.source.contains("q4k_pair_dot("));
        assert_eq!(
            kernel
                .source
                .matches("mixed_expert_element_from_offset(")
                .count(),
            1,
            "uniform lowering keeps only the shared helper declaration"
        );
        assert!(
            kernel
                .source
                .contains("expert_descriptors[expert_route_index[q]]")
        );
    }

    #[test]
    fn flattened_selected_axis_gathered_q4k_matmul_is_not_row_blocked() {
        // `[sequence = 1, selected = 2]` flattened to two rows has the same
        // physical `[2, d_in]` activation shape as a two-token gather. Each
        // row may name a different expert, so it cannot reuse the packed
        // multi-row body's one weight row across its activation group.
        let flattened_sequence_selected = 2;
        let bound = gathered_matmul_op(flattened_sequence_selected, 3, 4, 256);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        assert_eq!(
            classify_packed_row_block(&bound, &operand_codecs(&bound, &q4k)).err(),
            Some(PackedRowBlockRejection::GatheredOperand),
            "flattened selected rows may route to different expert slabs, so the packed multi-row body must not share one weight row"
        );
    }

    /// `metal-q4k-single-fetch` sibling of the test above: the lane remap
    /// renames the scale-deferred accumulators (`raw_low`/`raw_high` in
    /// place of `q4k_pair_dot`, one pair per sub-block half — see
    /// `push_q4k_single_fetch_body`'s own algebra note) but the SAME
    /// dichotomy holds — Add-reduce over a plain product still defers the
    /// scale, never falls back to per-element dequant. `matmul_op_f16`, not
    /// `matmul_op`: `q4k_pair_dot`'s own `plain_product` arm is
    /// `DType::Float32`-only and takes priority over this feature
    /// (see `metal-q4k-single-fetch`'s own Cargo.toml doc), so a `Float32`
    /// fixture would silently exercise `q4k_pair_dot` instead and this
    /// test would assert nothing about single-fetch at all.
    #[cfg(all(feature = "metal-q4k-single-fetch", not(feature = "metal-q4k-split-k")))]
    #[test]
    fn q4k_row_blocked_matmul_defers_scale_to_once_per_sub_block_single_fetch() {
        let bound = matmul_op_f16(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        assert!(
            packed_row_block(&bound, &operand_codecs(&bound, &q4k)).is_some(),
            "test fixture must actually take the row-blocked path for this assertion to mean anything"
        );

        let source = emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            source.contains("raw_low") && source.contains("raw_high"),
            "Add-reduce over a plain weight*activation body must take the scale-deferred path, split across both sub-block halves:\n{source}"
        );
        assert!(
            !source.contains("hdr_low.scale * low_levels[j] - hdr_low.minimum")
                && !source.contains("hdr_high.scale * high_levels[j] - hdr_high.minimum"),
            "the per-element dequant expression must not remain once the scale-deferred path is taken:\n{source}"
        );
    }

    /// The landmine `Q8_0`'s own landing closed (`PackedRowBlockRejection::
    /// NotKQuantCodec`, added because an EARLIER equality-only check would
    /// have silently admitted any non-K-quant codec whose extent happened
    /// to be a multiple of 256): `Q4_0`'s own block is 32 elements, and 256
    /// is ALSO a whole multiple of that, so an extent-only gate could
    /// wrongly admit it into the K-quant row-blocked kernel. The codec
    /// check must reject `Q4_0` explicitly, before the extent is ever
    /// consulted.
    #[test]
    fn q4_0_codec_never_takes_the_row_blocked_path_even_at_a_256_extent() {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4_0 = BTreeMap::new();
        q4_0.insert(weight_node, PackedCodec::Q4_0);

        assert_eq!(
            classify_packed_row_block(&bound, &operand_codecs(&bound, &q4_0)).err(),
            Some(PackedRowBlockRejection::NotKQuantCodec),
            "Q4_0 must be rejected by codec, not admitted just because 256 is a multiple of its own block size"
        );
        assert!(
            packed_row_block(&bound, &operand_codecs(&bound, &q4_0)).is_none(),
            "packed_row_block must agree with classify_packed_row_block's own rejection"
        );

        let source = emit(&bound, &q4_0, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            source.contains("q4_0_element("),
            "a Q4_0 weight must render through the generic per-element accessor:\n{source}"
        );
        assert!(
            !source.contains("q4k_run8(blk")
                && !source.contains("q5k_value(blk")
                && !source.contains("q6k_value(blk"),
            "a Q4_0 weight must never emit a K-quant row-blocked unpack call:\n{source}"
        );
    }

    /// Same landmine `q4_0_codec_never_takes_the_row_blocked_path_even_at_a_256_extent`
    /// closes, proven for BOTH half-precision codecs at once: neither is a
    /// K-quant, so both must reject via `NotKQuantCodec` and render through
    /// the generic per-element accessor -- `Float16`'s direct `half` index,
    /// `BFloat16`'s `bf16_element` widen -- never the row-blocked path's
    /// `q4k_run8`/`q5k_value`/`q6k_value` calls.
    #[proxima::test]
    #[case::float16(PackedCodec::Float16, "in0[")]
    #[case::bfloat16(PackedCodec::BFloat16, "bf16_element(")]
    async fn half_precision_codec_never_takes_the_row_blocked_path_even_at_a_256_extent(
        #[case] codec: PackedCodec,
        #[case] expected_read: &str,
    ) {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut operands = BTreeMap::new();
        operands.insert(weight_node, codec);

        assert_eq!(
            classify_packed_row_block(&bound, &operand_codecs(&bound, &operands)).err(),
            Some(PackedRowBlockRejection::NotKQuantCodec),
            "{codec:?} must be rejected by codec, not admitted just because 256 is a multiple of its own block size"
        );
        assert!(
            packed_row_block(&bound, &operand_codecs(&bound, &operands)).is_none(),
            "packed_row_block must agree with classify_packed_row_block's own rejection"
        );

        let source = emit(&bound, &operands, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            source.contains(expected_read),
            "a {codec:?} weight must render through its own generic per-element accessor:\n{source}"
        );
        assert!(
            !source.contains("q4k_run8(blk")
                && !source.contains("q5k_value(blk")
                && !source.contains("q6k_value(blk"),
            "a {codec:?} weight must never emit a K-quant row-blocked unpack call:\n{source}"
        );
    }

    #[cfg(not(all(feature = "metal-q4k-single-fetch", not(feature = "metal-q4k-split-k"))))]
    #[test]
    fn q4k_row_blocked_non_add_reduce_keeps_the_per_element_path() {
        // Same fused `weight * activation` body as the matmul shape above,
        // but `Maximum` in place of `Add` — the identity
        // `sum_j (scale*nibble_j - min)*act_j == scale*sum(...) - min*sum(...)`
        // does not hold under `max`, so this must fall back to dequantizing
        // per element exactly as before this landing.
        let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Maximum);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        assert!(
            packed_row_block(&bound, &operand_codecs(&bound, &q4k)).is_some(),
            "test fixture must actually take the row-blocked path for this assertion to mean anything"
        );

        let source = emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            !source.contains("raw_acc"),
            "a Maximum reduce must never take the scale-deferred path, its identity does not hold under max:\n{source}"
        );
        assert!(
            source.contains("hdr.scale * levels[j] - hdr.minimum"),
            "a Maximum reduce must keep dequantizing per element:\n{source}"
        );
    }

    /// `metal-q4k-single-fetch` sibling of the test above: the lane remap
    /// renames the per-element dequant expression (`hdr_low`/`hdr_high` in
    /// place of `hdr`, one pair per sub-block half — see
    /// `push_q4k_single_fetch_body`'s own doc) but the SAME dichotomy holds
    /// — a Maximum reduce still falls back to dequantizing per element,
    /// never the scale-deferred accumulators either arm uses for Add.
    #[cfg(all(feature = "metal-q4k-single-fetch", not(feature = "metal-q4k-split-k")))]
    #[test]
    fn q4k_row_blocked_non_add_reduce_keeps_the_per_element_path_single_fetch() {
        let bound = matmul_op_with_reduce(4, 256, 5, ScalarOp::Maximum);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        assert!(
            packed_row_block(&bound, &operand_codecs(&bound, &q4k)).is_some(),
            "test fixture must actually take the row-blocked path for this assertion to mean anything"
        );

        let source = emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            !source.contains("raw_low") && !source.contains("raw_high"),
            "a Maximum reduce must never take the scale-deferred path, its identity does not hold under max:\n{source}"
        );
        assert!(
            source.contains("hdr_low.scale * low_levels[j] - hdr_low.minimum")
                && source.contains("hdr_high.scale * high_levels[j] - hdr_high.minimum"),
            "a Maximum reduce must keep dequantizing per element, both sub-block halves:\n{source}"
        );
    }

    /// Same shape as [`matmul_op`] (`lhs=[features,k]` weight,
    /// `rhs=[k,tokens]` activation), but with the out_map listing the TOKEN
    /// axis before the feature axis -- `output_axes = [1, 0]` instead of
    /// `matmul_op`'s `[0, 1]`. This is the convention every real matmul in
    /// `proxima-tensor/src/spec.rs` follows (`"sg->sdg"`, `"so->sugdo"`,
    /// ...: token/sequence letters listed first, the weight's own letters
    /// last) and [`classify_tiled_gemm`]'s own doc names as load-bearing for
    /// `native_packed_layout`'s packed-stride reconstruction — `matmul_op`'s
    /// own `[0, 1]` order fails that check by construction, so the tiled
    /// path needs its own fixture rather than reusing `matmul_op` (which
    /// several PRE-EXISTING structural tests already pin to its current
    /// order).
    fn tiled_gemm_op(tokens: u32, k: u32, features: u32) -> BoundOp {
        let mut program = Vec::new();
        let lhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(features), Extent::Static(k)],
                name: None,
            },
        );
        let rhs = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(k), Extent::Static(tokens)],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[1, 0])),
                keep: Keep::Reduce,
                name: Some("tiled_gemm".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("tiled gemm op infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("tiled gemm op lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    /// Same fused body as [`matmul_op`], but a 3-output-axis shape (`h`, `d`
    /// weight-owned, `s` activation-owned) mirroring the multi-head Q/K/V
    /// projections `proxima-tensor/src/spec.rs`'s `"ihd->shdi"` pattern
    /// takes — `classify_tiled_gemm`'s own doc names this the documented
    /// scope limit (ROW 107), not a silent gap: [`push_tiled_gemm_body`]
    /// only understands a 2-D tile, so this shape must always stay on the
    /// row-blocked path regardless of token count.
    #[cfg(feature = "metal-tiled-gemm")]
    fn multi_head_matmul_op(seq: u32, heads: u32, head_dim: u32, embed: u32) -> BoundOp {
        let mut program = Vec::new();
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(seq), Extent::Static(embed)],
                name: None,
            },
        );
        let weight = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![
                    Extent::Static(embed),
                    Extent::Static(heads),
                    Extent::Static(head_dim),
                ],
                name: None,
            },
        );
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (weight, IndexMap::Affine(map::projection(4, &[3, 1, 2]))),
                    (activation, IndexMap::Affine(map::projection(4, &[0, 3]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(map::projection(4, &[0, 1, 2])),
                keep: Keep::Reduce,
                name: Some("multi_head_matmul".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("multi-head matmul infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("multi-head matmul lowers")
            .into_iter()
            .next()
            .expect("one fused bound emitted")
    }

    /// [`push_tiled_gemm_body`]'s empty-group guard, driven with a hand-built
    /// [`TiledGemmBlock`] -- [`classify_tiled_gemm`]'s own `is_empty()` gate
    /// never lets a real caller build one of these, so this drives the
    /// emitter's internal contract directly.
    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn push_tiled_gemm_body_rejects_an_empty_token_axis_group() {
        let bound = tiled_gemm_op(16, 256, 4);
        let block = TiledGemmBlock {
            weight: 0,
            other: 1,
            reduce_dim: 1,
            token_axes: Vec::new(),
            feature_axes: vec![0],
        };
        let mut source = String::new();
        let error = push_tiled_gemm_body(&mut source, bound.node, &[0], 2, &block, "float")
            .expect_err("an empty token axis group is never built by classify_tiled_gemm");
        assert!(matches!(
            error,
            EmitError::EmptyAxisGroup { group: "token", .. }
        ));
    }

    /// [`push_tiled_gemm_body`]'s axis-lookup guard: a hand-built
    /// [`TiledGemmBlock`] naming an axis outside `output_axes` --
    /// [`classify_tiled_gemm`] only ever builds `token_axes`/`feature_axes`
    /// as a subset of `output_axes`, so this too drives the internal
    /// contract directly.
    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn push_tiled_gemm_body_rejects_an_axis_not_in_output_axes() {
        let bound = tiled_gemm_op(16, 256, 4);
        let block = TiledGemmBlock {
            weight: 0,
            other: 1,
            reduce_dim: 1,
            token_axes: vec![5],
            feature_axes: vec![0],
        };
        let mut source = String::new();
        let error = push_tiled_gemm_body(&mut source, bound.node, &[0], 2, &block, "float")
            .expect_err("axis 5 is never in output_axes [0]");
        assert!(matches!(
            error,
            EmitError::AxisNotInOutputAxes { axis: 5, .. }
        ));
    }

    /// [`push_tiled_gemm_body`]'s `#[cfg(not(feature = "metal-tiled-gemm"))]`
    /// stub and [`tiled_gemm_threadgroups`]'s own non-feature arm both
    /// name this exact state: the tiled path reached without the feature
    /// that alone can build a real `TiledGemmBlock`.
    #[cfg(not(feature = "metal-tiled-gemm"))]
    #[test]
    fn push_tiled_gemm_body_is_disabled_without_the_metal_tiled_gemm_feature() {
        let bound = tiled_gemm_op(16, 256, 4);
        let block = TiledGemmBlock {
            token_axes: Vec::new(),
            feature_axes: Vec::new(),
        };
        let mut source = String::new();
        let error = push_tiled_gemm_body(&mut source, bound.node, &[0], 2, &block, "float")
            .expect_err("the tiled path never exists without metal-tiled-gemm");
        assert!(matches!(error, EmitError::TiledGemmFeatureDisabled { .. }));

        let error = tiled_gemm_threadgroups(bound.node, 4, 16)
            .expect_err("the tiled path never exists without metal-tiled-gemm");
        assert!(matches!(error, EmitError::TiledGemmFeatureDisabled { .. }));
    }

    #[cfg(not(feature = "metal-tiled-gemm"))]
    #[test]
    fn tiled_gemm_never_triggers_without_the_metal_tiled_gemm_feature() {
        // 16 tokens clears every plausible threshold; without the feature
        // compiled in, `TILED_GEMM_MIN_TOKENS` does not exist at all and
        // `classify_tiled_gemm` always returns `None` — see that function's
        // own doc. This test is cfg-gated the OPPOSITE way from the
        // `metal-tiled-gemm`-only tests below: it proves the tiled path is
        // invisible in the build that does not opt into it.
        let bound = tiled_gemm_op(16, 256, 4);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        let source = emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            !source.contains("simdgroup_multiply_accumulate"),
            "the tiled GEMM path must not exist at all without `metal-tiled-gemm`:\n{source}"
        );
        assert!(
            source.contains("sumf["),
            "16 tokens must still take the row-blocked path when the feature is off:\n{source}"
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn decode_shape_stays_on_the_row_blocked_path_with_tiled_gemm_compiled_in() {
        // ONE token (real decode's own shape) is below
        // `TILED_GEMM_MIN_TOKENS` (8) regardless of how large the feature
        // axis is — proves decode keeps taking the vector path even when
        // the tiled kernel is compiled into the binary, the exact
        // correctness requirement ROW 107 states.
        let bound = tiled_gemm_op(1, 256, 4096);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        assert!(
            tiled_gemm_block(
                &bound,
                &operand_codecs(&bound, &q4k),
                ScalarOp::Add,
                ReduceInit::Zero,
                &[1, 0]
            )
            .is_none(),
            "one token must never clear TILED_GEMM_MIN_TOKENS"
        );
        let source = emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            !source.contains("simdgroup_multiply_accumulate"),
            "a one-token (decode-shaped) dispatch must not take the tiled GEMM path:\n{source}"
        );
        assert!(
            source.contains("sumf["),
            "a one-token dispatch must still take the row-blocked path:\n{source}"
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn many_token_matmul_takes_the_tiled_gemm_path() {
        // 16 tokens clears TILED_GEMM_MIN_TOKENS (8); 4 weight rows is
        // deliberately NOT a multiple of TILE_DIM (8), exercising the
        // boundary-tile mask on the feature axis in the same test that
        // proves the path is taken at all.
        let bound = tiled_gemm_op(16, 256, 4);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        assert!(
            tiled_gemm_block(
                &bound,
                &operand_codecs(&bound, &q4k),
                ScalarOp::Add,
                ReduceInit::Zero,
                &[1, 0]
            )
            .is_some(),
            "16 tokens must clear TILED_GEMM_MIN_TOKENS"
        );
        let source = emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            source.contains("simdgroup_multiply_accumulate"),
            "a 16-token dispatch must take the tiled GEMM path:\n{source}"
        );
        assert!(
            source.contains("simdgroup_load"),
            "the tiled path must stage both operand tiles:\n{source}"
        );
        assert!(
            source.contains("feature_extent"),
            "the boundary mask must read the feature extent from uniforms, never bake it in:\n{source}"
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn non_q4k_codec_never_takes_the_tiled_gemm_path() {
        // Q5_K/Q6_K are explicitly out of scope (ROW 107) -- unmeasured on
        // this path, and their unpack has no batched form to reuse.
        let bound = tiled_gemm_op(16, 256, 4);
        let weight_node = bound.operands()[0].0;
        let mut q6k = BTreeMap::new();
        q6k.insert(weight_node, PackedCodec::Q6K);

        assert!(
            tiled_gemm_block(
                &bound,
                &operand_codecs(&bound, &q6k),
                ScalarOp::Add,
                ReduceInit::Zero,
                &[1, 0]
            )
            .is_none(),
            "a Q6_K weight must never take the tiled GEMM path"
        );
        let source = emit(&bound, &q6k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            !source.contains("simdgroup_multiply_accumulate"),
            "a Q6_K weight must not emit the tiled GEMM kernel:\n{source}"
        );
    }

    #[cfg(feature = "metal-tiled-gemm")]
    #[test]
    fn multi_head_shaped_matmul_stays_on_the_row_blocked_path_regardless_of_token_count() {
        // 32 sequence positions clears TILED_GEMM_MIN_TOKENS handily, but
        // this op keeps TWO weight-owned output axes (`heads`, `head_dim`)
        // -- `classify_tiled_gemm`'s documented scope limit, not a silent
        // gap.
        let bound = multi_head_matmul_op(32, 8, 128, 4096);
        let weight_node = bound.operands()[1].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);

        let codecs = operand_codecs(&bound, &q4k);
        assert!(
            packed_row_block(&bound, &codecs).is_some(),
            "test fixture must actually clear the row-blocked gate for this assertion to mean anything"
        );
        let BoundOpKind::Reduce {
            reduce_op,
            init,
            output_axes,
            ..
        } = &bound.kind
        else {
            panic!("multi_head_matmul_op always builds a Keep::Reduce fold")
        };
        assert!(
            tiled_gemm_block(&bound, &codecs, *reduce_op, *init, output_axes).is_none(),
            "a 3-output-axis matmul must never take the 2-D tiled GEMM path"
        );
        let source = emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert!(
            !source.contains("simdgroup_multiply_accumulate"),
            "a multi-head-shaped matmul must stay on the row-blocked path:\n{source}"
        );
    }

    fn cumsum_op(extent: u32) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(extent)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[0])),
                keep: Keep::Scan,
                name: None,
            }),
        );
        let shapes = infer(&program, &[]).expect("cumsum infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("cumsum lowers")
            .into_iter()
            .next()
            .expect("one bound emitted")
    }

    /// `table[ids[s], d]` over iteration space `(s, d)`: the same worked
    /// example `map.rs`'s docs use, as a standalone elementwise gather.
    fn embedding_lookup_op(vocab: u32, dim: u32, seq: u32) -> BoundOp {
        let mut program = Vec::new();
        let table = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(vocab), Extent::Static(dim)],
                name: None,
            },
        );
        let ids = append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: vec![Extent::Static(seq)],
                name: None,
            },
        );
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(2, &[0]),
            base: map::IndexPattern {
                iter_rank: 2,
                axes: vec![
                    map::AxisIndex::default(),
                    map::AxisIndex {
                        terms: vec![AxisTerm::projection(1)].into(),
                        offset: 0,
                        len: None,
                    },
                ],
            },
            gathered_dim: 0,
        };
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: vec![(table, gathered_map)],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("embedding lookup infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("embedding lookup lowers")
            .into_iter()
            .next()
            .expect("one bound emitted")
    }

    #[test]
    fn a_gather_op_emits_an_indices_binding_and_the_fetch_uniforms() {
        let bound = embedding_lookup_op(50_000, 8, 4);
        let kernel =
            emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("gather emits");

        assert_eq!(
            kernel.entry, "omega_elementwise_r2_n1_identity_g1",
            "the gather bit is part of the structural fingerprint"
        );
        assert_eq!(
            kernel.bindings,
            vec![
                Binding::Input(bound.operands()[0].0),
                Binding::Indices(
                    bound.operands()[0]
                        .2
                        .as_ref()
                        .expect("operand 0 gathers")
                        .indices
                ),
                Binding::Output(bound.node),
                Binding::Uniforms,
                Binding::Fault,
            ],
            "inputs, then indices, then output, then uniforms, then the fault buffer"
        );
        assert!(kernel.source.contains("gather_idx0"));
        assert!(kernel.source.contains("gather_index_base"));
        assert!(kernel.source.contains("gather_element_stride"));
        assert!(kernel.source.contains("gather_extent"));
        assert_eq!(kernel.grid.threads, 4 * 8, "seq x feature, vocab absent");
    }

    #[test]
    fn expert_source_emission_appends_payload_and_descriptor_bindings() {
        let bound = embedding_lookup_op(50_000, 8, 4);
        let kernel = emit_with_expert_sources(
            &bound,
            &BTreeMap::new(),
            NumericPolicy::default(),
            bound.operands()[0].0,
        )
        .expect("expert-source ABI emits");

        assert_eq!(
            &kernel.bindings[kernel.bindings.len() - 2..],
            &[
                Binding::ExpertPayloads(bound.operands()[0].0),
                Binding::ExpertDescriptors(bound.operands()[0].0),
            ],
            "expert buffers follow the ordinary input/index/output/uniform ABI"
        );
        assert!(kernel.source.contains("expert_payloads"));
        assert!(kernel.source.contains("expert_descriptors"));
        assert!(kernel.source.contains("struct ExpertPayloadDescriptor"));
    }

    #[test]
    fn preamble_contains_mixed_expert_codec_selector() {
        let source = MIXED_EXPERT_READ_MSL;
        assert!(source.contains("mixed_expert_element"));
        assert!(source.contains("selected.codec == 1u"));
        assert!(source.contains("selected.codec == 2u"));
        assert!(source.contains("selected.codec == 3u"));
        assert!(source.contains("selected.codec == 4u"));
    }

    #[test]
    fn routed_q4k_reduce_uses_cooperative_gather_fetch() {
        let mut program = Vec::new();
        let stack = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(3), Extent::Static(256), Extent::Static(2)],
                name: Some("expert_stack".into()),
            },
        );
        let route = append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: vec![Extent::Static(4)],
                name: Some("route".into()),
            },
        );
        let activation = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4), Extent::Static(256)],
                name: Some("activation".into()),
            },
        );
        let gathered =
            proxima_tensor::spec::gathered_expert_product(&mut program, stack, route, activation);
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: gathered,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
                keep: Keep::Reduce,
                name: Some("routed_q4k".into()),
            }),
        );
        let shapes = infer(&program, &[]).expect("routed reduction infers");
        let bound = bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("routed reduction lowers")
            .into_iter()
            .find(|bound| matches!(bound.kind, BoundOpKind::Reduce { .. }))
            .expect("reduction bound op exists");
        let mut packed = BTreeMap::new();
        packed.insert(bound.operands()[0].0, PackedCodec::Q4K);
        let kernel =
            emit(&bound, &packed, NumericPolicy::default()).expect("routed reduction emits");
        assert!(kernel.source.contains("simd_sum(accumulator)"));
        assert!(
            kernel
                .source
                .contains("simd_broadcast_first((uint)fetched0)")
        );
        assert!(kernel.source.contains("q4k_header_for"));
    }

    #[test]
    fn a_gather_kernel_binds_and_declares_the_fault_buffer() {
        let bound = embedding_lookup_op(50_000, 8, 4);
        let kernel =
            emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("gather emits");

        assert!(
            kernel.bindings.contains(&Binding::Fault),
            "a gather kernel must bind a fault buffer"
        );
        assert!(kernel.source.contains("device atomic_uint* fault"));
        assert!(
            kernel
                .source
                .contains("atomic_fetch_max_explicit(&fault[0]")
        );
        assert!(
            kernel
                .source
                .contains("fetched0 < 0 || fetched0 >= u.gather_extent[0]"),
            "the fault check must run before the clamp, on the unclamped fetched value"
        );
    }

    #[test]
    fn a_gather_free_op_names_and_binds_exactly_as_before_gather_existed() {
        let bound = elementwise_tanh_op(10);
        let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
            .expect("gather-free elementwise emits");
        assert!(
            !kernel.entry.contains("_g"),
            "a gather-free kernel's name must not grow a gather suffix"
        );
        assert!(!kernel.source.contains("gather_idx"));
        assert!(
            !kernel.source.contains("fault") && !kernel.source.contains("atomic_uint"),
            "a gather-free kernel must not gain any fault-reporting machinery"
        );
        assert_eq!(
            kernel.bindings,
            vec![
                Binding::Input(bound.operands()[0].0),
                Binding::Output(bound.node),
                Binding::Uniforms,
            ],
            "gather-free bindings are unchanged: input, output, uniforms — no fault buffer"
        );
    }

    #[test]
    fn elementwise_op_emits_one_input_one_output_and_a_matching_grid() {
        let bound = elementwise_tanh_op(10);
        let kernel =
            emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("elementwise emits");

        assert_eq!(kernel.entry, "omega_elementwise_r1_n1_tanh");
        assert_eq!(
            kernel.bindings,
            vec![
                Binding::Input(bound.operands()[0].0),
                Binding::Output(bound.node),
                Binding::Uniforms
            ]
        );
        assert!(
            kernel
                .source
                .contains("kernel void omega_elementwise_r1_n1_tanh")
        );
        assert!(kernel.source.contains("tanh(scratch[0])"));
        assert_eq!(kernel.grid.threads, 10);
    }

    /// A plain, unfused `Reduce` (identity element body, `Add`/`Zero`) over a
    /// 3D input, keeping exactly `output_rank_axes` of its 3 iteration axes
    /// and folding the rest — the minimal-pair generator ROW 93's
    /// `kernel_cache_key` regression test needs: two calls with the SAME
    /// `rank` (3) and operand count (1) but a DIFFERENT `output_rank_axes.len()`
    /// share every field `entry_name` recorded before this row (rank, operand
    /// count, body, reduce op, keep, init) while `render_reduce` still sizes
    /// `output_extents`/`reduction_extents` differently for each — proving
    /// `output_axes.len()` had to join the key, not just decorate a doc-comment.
    fn rank3_identity_sum_op(output_rank_axes: &[u16]) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(2), Extent::Static(2), Extent::Static(2)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, output_rank_axes)),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let shapes = infer(&program, &[]).expect("rank3 identity sum infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("rank3 identity sum lowers")
            .into_iter()
            .next()
            .expect("one bound emitted")
    }

    #[test]
    fn distinct_output_rank_at_same_total_rank_yields_distinct_cache_keys_and_source() {
        let keeps_two_axes = rank3_identity_sum_op(&[0, 1]);
        let keeps_one_axis = rank3_identity_sum_op(&[0]);

        assert_eq!(
            keeps_two_axes.extents.len(),
            keeps_one_axis.extents.len(),
            "same total rank"
        );
        assert_eq!(
            keeps_two_axes.operands().len(),
            keeps_one_axis.operands().len(),
            "same operand count"
        );

        let empty = BTreeMap::new();
        let key_two_axes = kernel_cache_key(&keeps_two_axes, &empty, NumericPolicy::default())
            .expect("cache key builds");
        let key_one_axis = kernel_cache_key(&keeps_one_axis, &empty, NumericPolicy::default())
            .expect("cache key builds");
        assert_ne!(
            key_two_axes, key_one_axis,
            "a coarser key would let a 1-output-axis fold hit the 2-output-axis pipeline"
        );

        let source_two_axes = emit(&keeps_two_axes, &empty, NumericPolicy::default())
            .expect("emits")
            .source;
        let source_one_axis = emit(&keeps_one_axis, &empty, NumericPolicy::default())
            .expect("emits")
            .source;
        assert_ne!(
            source_two_axes, source_one_axis,
            "output_extents/reduction_extents array sizes must differ in the rendered source"
        );
    }

    /// The test the class defect this module fixes needed: sweep every
    /// render option one axis at a time from a shared base op, and for each
    /// assert BOTH the emitted source changes AND `kernel_cache_key`'s
    /// identity changes with it. ROW 290 (a forgotten cooperative-reduce
    /// width) and main 7312713 (wgsl/cuda forgetting the reduce epilogue)
    /// are exactly the shape this would have caught: a renderer whose body
    /// text moved on some axis while its own identity function stayed
    /// silent about it.
    #[test]
    fn every_axis_that_changes_emitted_source_also_changes_kernel_cache_key() {
        let empty = BTreeMap::new();

        // reduce op.
        let add = matmul_op_with_reduce(4, 8, 5, ScalarOp::Add);
        let max = matmul_op_with_reduce(4, 8, 5, ScalarOp::Maximum);
        assert_ne!(
            emit(&add, &empty, NumericPolicy::default())
                .expect("emits")
                .source,
            emit(&max, &empty, NumericPolicy::default())
                .expect("emits")
                .source,
            "reduce op must change the emitted body"
        );
        assert_ne!(
            kernel_cache_key(&add, &empty, NumericPolicy::default()).expect("cache key builds"),
            kernel_cache_key(&max, &empty, NumericPolicy::default()).expect("cache key builds"),
            "reduce op must change the identity"
        );

        // dtype (half vs wide).
        let mut program = Vec::new();
        let f32_source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: vec![(f32_source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let mut f16_program = Vec::new();
        let f16_source = append(
            &mut f16_program,
            Op::Input {
                dtype: DType::Float16,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut f16_program,
            Op::Elementwise {
                dtype: DType::Float16,
                body: ScalarOp::Tanh,
                operands: vec![(f16_source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let f32_shapes = infer(&program, &[]).expect("f32 infers");
        let f32_bound = bind(&program, &f32_shapes, &[], NumericPolicy::default())
            .expect("f32 lowers")
            .into_iter()
            .next()
            .expect("one bound emitted");
        let f16_shapes = infer(&f16_program, &[]).expect("f16 infers");
        let f16_bound = bind(&f16_program, &f16_shapes, &[], NumericPolicy::default())
            .expect("f16 lowers")
            .into_iter()
            .next()
            .expect("one bound emitted");
        assert_ne!(
            emit(&f32_bound, &empty, NumericPolicy::default())
                .expect("emits")
                .source,
            emit(&f16_bound, &empty, NumericPolicy::default())
                .expect("emits")
                .source,
            "dtype must change the emitted body (half vs. float declarations)"
        );
        assert_ne!(
            kernel_cache_key(&f32_bound, &empty, NumericPolicy::default())
                .expect("cache key builds"),
            kernel_cache_key(&f16_bound, &empty, NumericPolicy::default())
                .expect("cache key builds"),
            "dtype must change the identity"
        );

        // packed codec (the census's own finding for wgsl/cuda -- confirmed
        // here it was ALREADY correct for Metal).
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);
        let mut q5k = BTreeMap::new();
        q5k.insert(weight_node, PackedCodec::Q5K);
        assert_ne!(
            emit(&bound, &q4k, NumericPolicy::default())
                .expect("emits")
                .source,
            emit(&bound, &q5k, NumericPolicy::default())
                .expect("emits")
                .source,
            "packed codec must change the emitted body"
        );
        assert_ne!(
            kernel_cache_key(&bound, &q4k, NumericPolicy::default()).expect("cache key builds"),
            kernel_cache_key(&bound, &q5k, NumericPolicy::default()).expect("cache key builds"),
            "packed codec must change the identity"
        );

        // numeric policy -- folded into `kernel_cache_key` directly now
        // (never embedded in `emit`'s own source text, so no source-side
        // assertion here). A permission not reflected in `MathMode` (e.g.
        // `contraction` alone) must still change the identity, or two
        // kernels compiling the SAME `MathMode` under different
        // `NumericPolicy`s could share one PIPELINE_CACHE entry.
        assert_ne!(
            kernel_cache_key(&bound, &q4k, NumericPolicy::bit_exact()).expect("cache key builds"),
            kernel_cache_key(
                &bound,
                &q4k,
                NumericPolicy::bit_exact().with_contraction(true)
            )
            .expect("cache key builds"),
            "numeric policy must change the identity, or a bit_exact- and a \
             contraction-only-compiled kernel could share one PIPELINE_CACHE \
             entry even though both compile MathMode::Safe or MathMode::Relaxed"
        );
        assert_ne!(
            kernel_cache_key(&bound, &q4k, NumericPolicy::bit_exact()).expect("cache key builds"),
            kernel_cache_key(&bound, &q4k, NumericPolicy::llama_relaxed())
                .expect("cache key builds"),
            "numeric policy must change the identity, or a bit_exact- and a \
             llama_relaxed-compiled kernel could share one PIPELINE_CACHE entry"
        );
        assert_ne!(
            kernel_cache_key(&bound, &q4k, NumericPolicy::llama_relaxed())
                .expect("cache key builds"),
            kernel_cache_key(&bound, &q4k, NumericPolicy::fast()).expect("cache key builds"),
            "numeric policy must change the identity, or a llama_relaxed- and a \
             fast-compiled kernel could share one PIPELINE_CACHE entry"
        );

        // cooperative-reduce width -- ROW 290's own defect, at the same two
        // reduce lengths that test proved COOPERATIVE_REDUCE_MIN_LEN against.
        #[cfg(feature = "metal-wide-cooperative-reduce")]
        {
            let narrow = single_axis_sum_op(34);
            let wide = single_axis_sum_op(4096);
            assert_ne!(
                kernel_cache_key(&narrow, &empty, NumericPolicy::default())
                    .expect("cache key builds"),
                kernel_cache_key(&wide, &empty, NumericPolicy::default())
                    .expect("cache key builds"),
                "two cooperative reduces at different widths must never share a pipeline \
                 (ROW 290: a stale narrower kernel silently drops reduction terms)"
            );
        }
    }

    /// The regression this row's first cut of `kernel_cache_key` actually
    /// shipped with (caught by `omega::metal_parity
    /// attention_block_spec_parity_matches_within_epsilon` and
    /// `omega::backend_parity the_wrapper_agrees_with_itself_across_cpu_and_metal`
    /// going from PASS to FAIL against a real, unrelated forward): two folds
    /// can share `rank` AND `output_axes.len()` (so the SAME "how many axes
    /// this key" check the prior test guards would still pass both) while
    /// keeping a DIFFERENT axis SET or the same set in a DIFFERENT ORDER --
    /// `render_reduce`/`push_cooperative_reduce_body` bake the literal `dim`
    /// tied to each `u.output_extents[index]` slot straight into the source,
    /// so either change alone must also change the key.
    #[test]
    fn distinct_output_axis_set_at_the_same_output_rank_yields_distinct_cache_keys_and_source() {
        let keeps_first_and_second = rank3_identity_sum_op(&[0, 1]);
        let keeps_first_and_third = rank3_identity_sum_op(&[0, 2]);
        let empty = BTreeMap::new();

        assert_eq!(
            keeps_first_and_second.extents.len(),
            keeps_first_and_third.extents.len(),
            "same total rank"
        );
        let key_first_second =
            kernel_cache_key(&keeps_first_and_second, &empty, NumericPolicy::default())
                .expect("cache key builds");
        let key_first_third =
            kernel_cache_key(&keeps_first_and_third, &empty, NumericPolicy::default())
                .expect("cache key builds");
        assert_ne!(
            key_first_second, key_first_third,
            "output_axes.len() alone cannot tell {{0,1}} from {{0,2}}"
        );

        let source_first_second = emit(&keeps_first_and_second, &empty, NumericPolicy::default())
            .expect("emits")
            .source;
        let source_first_third = emit(&keeps_first_and_third, &empty, NumericPolicy::default())
            .expect("emits")
            .source;
        assert_ne!(
            source_first_second, source_first_third,
            "the reduce dim, and every operand_strides[..][dim] read, must differ"
        );
    }

    #[test]
    fn output_axis_order_at_the_same_axis_set_yields_distinct_cache_keys_and_source() {
        let ascending = rank3_identity_sum_op(&[0, 1]);
        let descending = rank3_identity_sum_op(&[1, 0]);
        let empty = BTreeMap::new();

        let key_ascending = kernel_cache_key(&ascending, &empty, NumericPolicy::default())
            .expect("cache key builds");
        let key_descending = kernel_cache_key(&descending, &empty, NumericPolicy::default())
            .expect("cache key builds");
        assert_ne!(
            key_ascending, key_descending,
            "the SEQUENCE order of output_axes selects which u.output_extents slot each dim reads"
        );

        let source_ascending = emit(&ascending, &empty, NumericPolicy::default())
            .expect("emits")
            .source;
        let source_descending = emit(&descending, &empty, NumericPolicy::default())
            .expect("emits")
            .source;
        assert_ne!(
            source_ascending, source_descending,
            "reversing output_axes must reverse which dim each output_extents index feeds"
        );
    }

    #[test]
    fn distinct_packed_codec_on_the_same_shape_yields_distinct_cache_keys_and_source() {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;

        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);
        let mut q6k = BTreeMap::new();
        q6k.insert(weight_node, PackedCodec::Q6K);

        let key_q4k =
            kernel_cache_key(&bound, &q4k, NumericPolicy::default()).expect("cache key builds");
        let key_q6k =
            kernel_cache_key(&bound, &q6k, NumericPolicy::default()).expect("cache key builds");
        assert_ne!(
            key_q4k, key_q6k,
            "entry_name alone cannot see which codec an operand reads through"
        );

        let source_q4k = emit(&bound, &q4k, NumericPolicy::default())
            .expect("emits")
            .source;
        let source_q6k = emit(&bound, &q6k, NumericPolicy::default())
            .expect("emits")
            .source;
        assert_ne!(
            source_q4k, source_q6k,
            "Q4_K and Q6_K unpack through different MSL functions"
        );
    }

    #[test]
    fn distinct_dtype_on_the_same_shape_yields_distinct_cache_keys_and_source() {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Tanh,
                operands: vec![(source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let shapes = infer(&program, &[]).expect("f32 elementwise infers");
        let f32_bound = bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("f32 elementwise lowers")
            .into_iter()
            .next()
            .expect("one bound emitted");

        let mut half_program = Vec::new();
        let half_source = append(
            &mut half_program,
            Op::Input {
                dtype: DType::Float16,
                shape: vec![Extent::Static(4)],
                name: None,
            },
        );
        append(
            &mut half_program,
            Op::Elementwise {
                dtype: DType::Float16,
                body: ScalarOp::Tanh,
                operands: vec![(half_source, IndexMap::Affine(map::projection(1, &[0])))],
                name: None,
            },
        );
        let half_shapes = infer(&half_program, &[]).expect("f16 elementwise infers");
        let f16_bound = bind(&half_program, &half_shapes, &[], NumericPolicy::default())
            .expect("f16 elementwise lowers")
            .into_iter()
            .next()
            .expect("one bound emitted");

        let empty = BTreeMap::new();
        let key_f32 = kernel_cache_key(&f32_bound, &empty, NumericPolicy::default())
            .expect("cache key builds");
        let key_f16 = kernel_cache_key(&f16_bound, &empty, NumericPolicy::default())
            .expect("cache key builds");
        assert_ne!(
            key_f32, key_f16,
            "entry_name does not encode dtype on its own"
        );

        let source_f32 = emit(&f32_bound, &empty, NumericPolicy::default())
            .expect("emits")
            .source;
        let source_f16 = emit(&f16_bound, &empty, NumericPolicy::default())
            .expect("emits")
            .source;
        assert_ne!(
            source_f32, source_f16,
            "float vs half declarations must differ in source"
        );
    }

    #[test]
    fn same_structure_different_extents_share_one_cache_key() {
        let small = elementwise_tanh_op(4);
        let large = elementwise_tanh_op(4096);
        let empty = BTreeMap::new();

        assert_eq!(
            kernel_cache_key(&small, &empty, NumericPolicy::default()).expect("cache key builds"),
            kernel_cache_key(&large, &empty, NumericPolicy::default()).expect("cache key builds"),
            "a cache keyed on structure must still hit across concrete extents"
        );
    }

    #[test]
    fn fused_matmul_op_emits_two_inputs_a_reduction_loop_and_a_row_by_col_grid() {
        let bound = matmul_op(4, 3, 5);
        assert!(
            matches!(bound.kind, BoundOpKind::Reduce { .. }),
            "the elementwise op must have fused into the reduce"
        );
        let kernel =
            emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("matmul emits");

        assert_eq!(kernel.entry, "omega_reduce_r3_o2_n2_multiply_add_zero");
        assert_eq!(kernel.bindings.len(), 4, "two inputs, one output, uniforms");
        assert!(matches!(kernel.bindings[2], Binding::Output(_)));
        assert!(matches!(kernel.bindings[3], Binding::Uniforms));
        assert!(
            kernel
                .source
                .contains("kernel void omega_reduce_r3_o2_n2_multiply_add_zero")
        );
        assert!(kernel.source.contains("reduction_total"));
        assert!(kernel.source.contains("(scratch[0] * scratch[1])"));
        assert!(kernel.source.contains("(accumulator + value)"));
        assert!(
            kernel.source.contains("simd_sum(accumulator)"),
            "an Add-reduce body must take the cooperative SIMD-group path"
        );
        assert_eq!(
            kernel.grid.threads,
            4 * 5 * 32,
            "one SIMD-group (32 lanes) per (row, col), not one thread"
        );
        assert_eq!(
            kernel.grid.threadgroup_width,
            Some(32),
            "the driver must dispatch exactly one SIMD-group per threadgroup"
        );
    }

    /// A single 1-D input folded fully to a scalar via `Add` — the plain
    /// shape [`reduce_is_cooperative`]'s length gate reasons about, without
    /// `matmul_op`'s fused elementwise-multiply step muddying which reduce
    /// length is under test.
    fn single_axis_sum_op(reduce_len: u32) -> BoundOp {
        let mut program = Vec::new();
        let source = append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: vec![Extent::Static(reduce_len)],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: source,
                in_map: IndexMap::Affine(map::projection(1, &[0])),
                out_map: IndexMap::Affine(map::projection(1, &[])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let shapes = infer(&program, &[]).expect("single-axis sum infers");
        bind(&program, &shapes, &[], NumericPolicy::default())
            .expect("single-axis sum lowers")
            .into_iter()
            .next()
            .expect("one bound emitted")
    }

    /// Proven against the COMPILED `COOPERATIVE_REDUCE_MIN_LEN` constant,
    /// not a hardcoded 128 — this same test body is the re-prove artifact for
    /// BOTH claims the short-reduce initiative makes: at the
    /// `omega-runtime.toml` default (2) it covers the unit-axis boundary
    /// (1/2) while the 34 `attended`, 64 `score_even`/`score_odd`, and 4096
    /// `sum_squares` shapes stay cooperative; re-run under
    /// `OMEGA_COOPERATIVE_REDUCE_MIN_LEN=0`
    /// (a distinct build — the constant is compile-time) it proves 0
    /// restores every-qualifying-reduce-stays-cooperative, the routing every
    /// build before this key existed used, because `>= 0` is vacuously true
    /// for every case including the 34-length one.
    #[proxima::test]
    #[case::unit_axis(1)]
    #[case::at_threshold_2(2)]
    #[case::attended_34(34)]
    #[case::score_even_odd_64(64)]
    #[case::sum_squares_4096(4096)]
    async fn reduce_routes_on_reduced_axis_length_against_min_len(#[case] reduce_len: u32) {
        let bound = single_axis_sum_op(reduce_len);
        let expected_cooperative = meets_cooperative_min_len(u64::from(reduce_len));

        assert_eq!(
            reduce_is_cooperative(&bound),
            expected_cooperative,
            "reduce_len={reduce_len} vs COOPERATIVE_REDUCE_MIN_LEN={}",
            crate::sized::COOPERATIVE_REDUCE_MIN_LEN
        );

        let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
            .expect("single-axis sum emits");
        assert_eq!(
            kernel.source.contains("simd_sum(accumulator)"),
            expected_cooperative,
            "emitted kernel source must agree with reduce_is_cooperative's own routing decision"
        );
    }

    #[test]
    fn cached_attention_emits_one_online_softmax_dispatch() {
        let bound = cached_attention_op();
        let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
            .expect("cached attention emits");

        let BoundOpKind::CachedAttention {
            cached_key_rows,
            new_key_rows,
            query_groups,
            ..
        } = &bound.kind
        else {
            panic!("cached_attention_op must build a CachedAttention bound op");
        };
        let context_chunks =
            context_chunks_for(*cached_key_rows + *new_key_rows, NumericPolicy::default());

        assert!(kernel.source.contains("long relative ="));
        assert!(kernel.source.contains("simd_sum(partial_score)"));
        assert!(kernel.source.contains("vector_index = (long)gid / 32L"));
        assert!(kernel.source.contains("weighted[local_dimension] / sum"));
        assert_eq!(kernel.bindings.len(), 10, "eight inputs, output, uniforms");
        assert_eq!(kernel.grid.threads, 32 * context_chunks);
        assert_eq!(
            kernel.grid.threadgroup_width,
            Some(query_groups * context_chunks * SIMD_WIDTH),
            "query_groups=1 -- one threadgroup per (query_row, kv_head), width \
             widened by context_chunks under `[attention_context_chunks]`"
        );
        assert!(
            !kernel.source.contains("threadgroup float shared_k_even"),
            "K/V rows are read straight from device memory into registers, \
             never staged through threadgroup memory"
        );
        assert_eq!(
            kernel.source.contains("threadgroup float shared_m["),
            context_chunks > 1,
            "the cross-simdgroup merge only exists when context splits across \
             more than one simdgroup"
        );
        assert_eq!(
            kernel
                .source
                .contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
            context_chunks > 1,
            "a threadgroup_barrier only exists for the cross-simdgroup merge, \
             never inside the per-key loop"
        );
    }

    /// A one-chunk dispatch must render EXACTLY the pre-context-parallel
    /// body: no `context_chunks`/`chunk`/merge tokens anywhere in the
    /// source, and -- since the key loop no longer stages K/V through
    /// `threadgroup` memory -- no `threadgroup_barrier` token anywhere
    /// either, because chunks==1 has nothing left to synchronize on. This is
    /// the byte-identity guarantee `render_cached_attention`'s
    /// `context_chunks <= 1` branch reuses the original string construction
    /// for, verified here rather than by a stored fixture (none pre-existed
    /// to snapshot against). Skipped (not failed) under a sizing override
    /// aggressive enough that even this tiny fixture's context splits --
    /// `cached_attention_emits_one_online_softmax_dispatch` already checks
    /// grid shape agrees with `context_chunks_for` at ANY sizing, this test
    /// only adds the byte-identity claim for the sizing where chunks==1.
    #[test]
    fn cached_attention_at_one_chunk_never_emits_context_chunk_machinery() {
        let bound = cached_attention_op();
        let BoundOpKind::CachedAttention {
            cached_key_rows,
            new_key_rows,
            ..
        } = &bound.kind
        else {
            panic!("cached_attention_op must build a CachedAttention bound op");
        };
        if context_chunks_for(*cached_key_rows + *new_key_rows, NumericPolicy::default()) != 1 {
            return;
        }

        let kernel = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
            .expect("cached attention emits");
        assert!(!kernel.source.contains("context_chunks"));
        assert!(!kernel.source.contains("long chunk ="));
        assert!(!kernel.source.contains("shared_m["));
        assert!(!kernel.source.contains("threadgroup_barrier"));
    }

    #[test]
    fn cumsum_op_emits_a_scan_kernel_with_one_thread_per_line() {
        let bound = cumsum_op(8);
        let kernel =
            emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("cumsum emits");

        assert_eq!(kernel.entry, "omega_scan_r1_o1_n1_identity_add_zero");
        assert!(kernel.source.contains("inner_len"));
        assert!(kernel.source.contains("out_running"));
        assert_eq!(
            kernel.grid.threads, 1,
            "no leading dims: a single scan line"
        );
    }

    #[test]
    fn emit_is_deterministic_byte_equal() {
        let bound = matmul_op(4, 3, 5);
        let first =
            emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("first emit succeeds");
        let second =
            emit(&bound, &BTreeMap::new(), NumericPolicy::default()).expect("second emit succeeds");
        assert_eq!(first, second);
    }

    #[test]
    fn same_structure_different_extents_yield_identical_source_but_different_grid() {
        let small = elementwise_tanh_op(4);
        let large = elementwise_tanh_op(4096);

        let small_kernel =
            emit(&small, &BTreeMap::new(), NumericPolicy::default()).expect("small emits");
        let large_kernel =
            emit(&large, &BTreeMap::new(), NumericPolicy::default()).expect("large emits");

        assert_eq!(small_kernel.source, large_kernel.source);
        assert_eq!(small_kernel.entry, large_kernel.entry);
        assert_ne!(small_kernel.grid.threads, large_kernel.grid.threads);
    }

    #[test]
    fn an_arity_mismatched_op_is_rejected() {
        let mut bound = elementwise_tanh_op(4);
        if let BoundOpKind::Elementwise { body, .. } = &mut bound.kind {
            body.steps[0].op = ScalarOp::Add; // arity 2, but the step still carries 1 arg
        }

        let error = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
            .expect_err("mismatched arity is rejected");
        assert!(matches!(error, EmitError::ArityMismatch { .. }), "{error}");
    }

    #[test]
    fn a_select_reduction_body_is_rejected() {
        let mut bound = matmul_op(4, 3, 5);
        if let BoundOpKind::Reduce { reduce_op, .. } = &mut bound.kind {
            *reduce_op = ScalarOp::Select;
        }

        let error = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
            .expect_err("select reduction body is rejected");
        assert!(
            matches!(error, EmitError::ReductionBodyIsSelect { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_keep_scan_over_zero_axes_is_rejected() {
        let mut bound = cumsum_op(8);
        bound.extents.clear();
        if let BoundOpKind::Reduce { output_axes, .. } = &mut bound.kind {
            output_axes.clear();
        }

        let error = emit(&bound, &BTreeMap::new(), NumericPolicy::default())
            .expect_err("an empty scan is rejected");
        assert!(matches!(error, EmitError::EmptyScan { .. }), "{error}");
    }

    #[test]
    fn render_reduce_rejects_an_elementwise_bound_op() {
        let bound = elementwise_tanh_op(8);
        let error = render_reduce(&bound, "entry", &[None], false)
            .expect_err("an elementwise chain is not a Reduce fold");
        assert!(matches!(
            error,
            EmitError::RenderKindMismatch {
                expected: "keep::reduce fold",
                found: "elementwise",
                ..
            }
        ));
    }

    /// `bind::BoundOpKind::Reduce::epilogue_broadcast_axes`'s own doc: a
    /// non-empty value is the RMSNorm-shaped "broadcast-reduce" epilogue
    /// (`x * inv_rms`, re-broadcasting the fold's scalar back over the
    /// reduced axis) — no Metal kernel here widens `push_reduce_epilogue_
    /// write`'s own `output_rank`-only uniform declaration to match, so this
    /// must reject with a typed error rather than emit a kernel that reads
    /// or writes the wrong element count. The Metal HALF of this fusion is
    /// the next slice; this proves main stays correct if the bind-time half
    /// lands first.
    #[test]
    fn render_reduce_rejects_a_broadcast_reduce_epilogue() {
        let mut bound = matmul_op(4, 8, 4);
        let BoundOpKind::Reduce {
            epilogue_broadcast_axes,
            ..
        } = &mut bound.kind
        else {
            panic!("matmul_op always builds a Keep::Reduce fold")
        };
        epilogue_broadcast_axes.push(1);

        let error = render_reduce(&bound, "entry", &[None], false)
            .expect_err("a broadcast-reduce epilogue has no Metal renderer yet");
        assert!(
            matches!(error, EmitError::EpilogueNotSupported { .. }),
            "{error}"
        );
    }

    #[test]
    fn render_scan_rejects_an_elementwise_bound_op() {
        let bound = elementwise_tanh_op(8);
        let error = render_scan(&bound, "entry", &[None])
            .expect_err("an elementwise chain is not a Reduce fold");
        assert!(matches!(
            error,
            EmitError::RenderKindMismatch {
                expected: "keep::scan fold",
                found: "elementwise",
                ..
            }
        ));
    }

    #[test]
    fn render_cached_attention_rejects_an_elementwise_bound_op() {
        let bound = elementwise_tanh_op(8);
        let error = render_cached_attention(&bound, "entry", NumericPolicy::default())
            .expect_err("an elementwise chain is not a CachedAttention op");
        assert!(matches!(
            error,
            EmitError::RenderKindMismatch {
                expected: "cached_attention",
                found: "elementwise",
                ..
            }
        ));
    }

    #[test]
    fn simd_combine_fn_rejects_a_non_cooperative_reduce_op() {
        let bound = matmul_op_with_reduce(4, 8, 3, ScalarOp::Subtract);
        let error = simd_combine_fn(bound.node, ScalarOp::Subtract)
            .expect_err("subtract is not associative-commutative");
        assert!(matches!(
            error,
            EmitError::NonCooperativeReduceOp { op: "subtract", .. }
        ));
    }

    #[test]
    fn cooperative_identity_token_rejects_a_non_cooperative_reduce_op() {
        let bound = matmul_op_with_reduce(4, 8, 3, ScalarOp::Subtract);
        let error = cooperative_identity_token(bound.node, ScalarOp::Subtract)
            .expect_err("subtract has no cooperative SIMD-group identity");
        assert!(matches!(
            error,
            EmitError::NonCooperativeReduceOp { op: "subtract", .. }
        ));
    }

    /// [`push_packed_row_blocked_body`]'s per-codec match, reached with a
    /// hand-built [`PackedRowBlock`] naming a non-K-quant codec --
    /// [`classify_packed_row_block`]'s own `NotKQuantCodec` gate never
    /// builds one of these in practice, so this drives the emitter's
    /// internal contract directly rather than through [`emit`].
    #[test]
    fn push_packed_row_blocked_body_rejects_a_non_k_quant_codec() {
        let bound = matmul_op(4, 256, 3);
        let block = PackedRowBlock {
            weight: 0,
            other: 1,
            reduce_dim: 1,
            codec: PackedCodec::Q8_0,
            token_axes: Vec::new(),
            feature_axes: vec![0, 1],
        };
        let mut source = String::new();
        let error = push_packed_row_blocked_body(
            &mut source,
            &bound,
            ScalarOp::Add,
            ReduceInit::Zero,
            &[0, 1],
            2,
            &[Some(PackedCodec::Q8_0), None],
            "float",
            &block,
            &ComposedBody::leaf(ScalarOp::Identity),
            &[],
            false,
        )
        .expect_err("Q8_0 never reaches the row-blocked path");
        assert!(matches!(
            error,
            EmitError::NonKQuantPackedCodec { codec: "q8_0", .. }
        ));
    }

    /// Reachability proof for [`packed_row_split_factor`]'s row-count gate
    /// (`omega-runtime.toml`'s `[packed_row_block].split_k_max_rows`,
    /// default 4096): a 1024-row op (the `attn_k`/`attn_v` shape) sits
    /// under both the row ceiling and `target_simdgroups`, so split-K must
    /// engage (`split > 1`); a 14336-row op (the `ffn_up`/`ffn_gate` shape)
    /// sits well past the row ceiling, so split-K must stay a no-op
    /// (`split == 1`) regardless of what the simdgroup-target arithmetic
    /// alone would compute. Calls [`packed_row_dispatch`] directly -- the
    /// SAME function [`grid_threads`] and [`tiled_gemm_threadgroup_width`]
    /// call -- so a passing test here is a guarantee those call sites see
    /// the identical factor, not a duplicate policy that could drift.
    #[cfg(feature = "metal-q4k-split-k")]
    #[test]
    fn split_k_engages_for_a_1024_row_op_and_declines_for_a_14336_row_op() {
        let (base_starved, split_starved) = packed_row_dispatch(1024, 1, PackedCodec::Q4K);
        assert!(
            split_starved > 1,
            "a 1024-row op (attn_k/attn_v shape) must engage split-K under the default \
             split_k_max_rows(4096)/target_simdgroups gate: got split={split_starved} at \
             base_simdgroups={base_starved}"
        );

        let (base_wide, split_wide) = packed_row_dispatch(14336, 1, PackedCodec::Q4K);
        assert_eq!(
            split_wide, 1,
            "a 14336-row op (ffn_up/ffn_gate shape) must stay split-K's no-op factor: got \
             split={split_wide} at base_simdgroups={base_wide}"
        );
    }

    /// The row-count ceiling itself, isolated from `target_simdgroups`: a
    /// shape whose base-simdgroup count would otherwise clear
    /// `packed_row_split_factor`'s target (so the simdgroup arithmetic
    /// alone would still pick a factor > 1) must nonetheless collapse to
    /// `1` once its row count crosses [`crate::sized::PACKED_ROW_SPLIT_K_MAX_ROWS`].
    /// Proves the ceiling is a genuine additional gate, not merely
    /// redundant with the simdgroup-target fall-off already in place.
    #[cfg(feature = "metal-q4k-split-k")]
    #[test]
    fn the_row_ceiling_overrides_a_simdgroup_target_that_would_otherwise_split() {
        let max_rows = crate::sized::PACKED_ROW_SPLIT_K_MAX_ROWS;
        assert_ne!(
            max_rows, 0,
            "this test requires a non-zero configured ceiling"
        );

        let rows_at_ceiling = max_rows;
        let rows_past_ceiling = max_rows + PACKED_ROWS_PER_GROUP as u64;

        let base_at = rows_at_ceiling.div_ceil(PACKED_ROWS_PER_GROUP as u64);
        let split_at = packed_row_split_factor(base_at, rows_at_ceiling);
        let base_past = rows_past_ceiling.div_ceil(PACKED_ROWS_PER_GROUP as u64);
        let split_past = packed_row_split_factor(base_past, rows_past_ceiling);

        assert_eq!(
            split_past, 1,
            "one row-group past the configured ceiling must decline split-K even though its \
             base_simdgroups({base_past}) barely differs from the still-eligible shape's \
             ({base_at}), which split at factor {split_at}"
        );
    }

    /// Reachability proof for [`packed_row_nsg_factor`] (found dead: the
    /// `#[cfg(feature = "metal-packed-row-nsg2")]` arm in
    /// `tiled_gemm_threadgroup_width` sat AFTER an unconditional
    /// `packed_row_block` return in the `Keep::Reduce` `if let` above it, so
    /// it could never run for any op that reaches this function -- every
    /// `packed_row_block` match IS a `Keep::Reduce` op by construction, see
    /// `PackedRowBlock`'s own classification). With `metal-packed-row-nsg2`
    /// on (and `metal-q4k-split-k` off, so `split == 1`), the packed
    /// row-blocked matmul's threadgroup width must be exactly double the
    /// one-simdgroup default.
    #[cfg(all(feature = "metal-packed-row-nsg2", not(feature = "metal-q4k-split-k")))]
    #[test]
    fn packed_row_nsg2_doubles_the_threadgroup_width_for_a_packed_row_blocked_matmul() {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);
        let quantized = operand_codecs(&bound, &q4k);

        assert!(
            packed_row_block(&bound, &quantized).is_some(),
            "test fixture must actually take the row-blocked path for this assertion to mean anything"
        );

        let width = tiled_gemm_threadgroup_width(&bound, &quantized, NumericPolicy::default())
            .expect("a packed row-blocked reduce always has a threadgroup width");
        assert_eq!(
            width,
            SIMD_WIDTH * 2,
            "metal-packed-row-nsg2 must double the one-simdgroup default width to 2 \
             simdgroups (PACKED_ROW_NSG); got {width}"
        );
    }

    /// [`packed_row_nsg2_doubles_the_threadgroup_width_for_a_packed_row_blocked_matmul`]'s
    /// own twin for the OTHER feature that shares [`packed_row_nsg_factor`]:
    /// `metal-q4k-ggml-port` dispatches ggml's own body at ggml's own
    /// nsg=2, so it must double the threadgroup width the identical way
    /// `metal-packed-row-nsg2` does -- same assertion, different feature,
    /// proving the two features compose through one factor rather than two
    /// competing nsg constants.
    #[cfg(all(feature = "metal-q4k-ggml-port", not(feature = "metal-q4k-split-k")))]
    #[test]
    fn ggml_port_doubles_the_threadgroup_width_for_a_packed_row_blocked_matmul() {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);
        let quantized = operand_codecs(&bound, &q4k);

        assert!(
            packed_row_block(&bound, &quantized).is_some(),
            "test fixture must actually take the row-blocked path for this assertion to mean anything"
        );

        let width = tiled_gemm_threadgroup_width(&bound, &quantized, NumericPolicy::default())
            .expect("a packed row-blocked reduce always has a threadgroup width");
        assert_eq!(
            width,
            SIMD_WIDTH * 2,
            "metal-q4k-ggml-port must double the one-simdgroup default width to 2 \
             simdgroups (PACKED_ROW_NSG); got {width}"
        );
    }

    /// [`packed_row_nsg2_doubles_the_threadgroup_width_for_a_packed_row_blocked_matmul`]'s
    /// feature-off twin: without EITHER nsg2 feature compiled in, the NSG
    /// factor itself must stay at one simdgroup -- proves the nsg2 fix is
    /// additive, not a change to the default dispatch geometry. This cfg arm
    /// is also the one `metal-q4k-split-k` alone reaches (neither nsg2
    /// feature is on), and split-K widens the SAME threadgroup for an
    /// unrelated, real reason (starved shapes get more simdgroups
    /// cooperating on one reduction, [`packed_row_split_factor`]'s own doc),
    /// so the expected width is derived from the SAME
    /// [`packed_row_dispatch`]/[`packed_row_nsg_factor`] production reads
    /// rather than a literal -- a hardcoded `SIMD_WIDTH` here is feature-blind
    /// to split-K's own widening and fails every cell it does not predict.
    #[cfg(not(any(feature = "metal-packed-row-nsg2", feature = "metal-q4k-ggml-port")))]
    #[test]
    fn packed_row_nsg2_off_leaves_the_threadgroup_width_unchanged() {
        let bound = matmul_op(4, 256, 5);
        let weight_node = bound.operands()[0].0;
        let mut q4k = BTreeMap::new();
        q4k.insert(weight_node, PackedCodec::Q4K);
        let quantized = operand_codecs(&bound, &q4k);

        let block = packed_row_block(&bound, &quantized)
            .expect("test fixture must actually take the row-blocked path for this assertion to mean anything");

        let feature_total: u64 = block
            .feature_axes
            .iter()
            .map(|&axis| bound.extents[axis as usize])
            .product();
        let token_total = packed_row_block_token_total(&block, &bound.extents);
        let (_base, split) = packed_row_dispatch(feature_total, token_total, block.codec);
        let expected_width = SIMD_WIDTH * split * packed_row_nsg_factor();

        let width = tiled_gemm_threadgroup_width(&bound, &quantized, NumericPolicy::default())
            .expect("a packed row-blocked reduce always has a threadgroup width");
        assert_eq!(
            width, expected_width,
            "without either nsg2 feature the packed row-blocked path's width must match \
             production's own SIMD_WIDTH * split * packed_row_nsg_factor() derivation; got {width}"
        );
    }

    /// `omega-runtime.toml`'s `[attention_context_chunks]` ships
    /// `keys_per_chunk = 16`, so a real 64-key merged context
    /// (`cached_key_rows + new_key_rows`) is exactly the shape
    /// `context_chunks_for` splits across simdgroups today -- the algebra
    /// design's own worked example (`op.rs:107`'s `is_associative` has no
    /// caller that gates this reassociation; this is the gate).
    #[test]
    fn context_chunk_merge_is_gated_by_numeric_policy_for_a_real_64_key_context() {
        let context_length: u64 = 64;
        assert_eq!(
            context_chunks_for(context_length, NumericPolicy::bit_exact()),
            1,
            "bit_exact() withholds reassociation, ContextChunkMerge, so this falls back to the \
             single-pass chunk<=1 kernel `render_cached_attention` already renders"
        );
        let chunks = context_chunks_for(context_length, NumericPolicy::llama_relaxed());
        assert!(
            chunks > 1,
            "llama_relaxed() grants reassociation, clearing NumericRewrite::ContextChunkMerge, \
             so a 64-key context (4x omega-runtime.toml's 16-key chunk) must split across more \
             than one simdgroup; got {chunks}"
        );
    }

    /// The same gate, exercised through the real emitter
    /// ([`render_cached_attention`]) instead of `context_chunks_for` in
    /// isolation -- the rendered kernel source must not contain the
    /// cross-simdgroup merge block under `BitExact`, and must contain it
    /// once the caller opts up.
    #[test]
    fn render_cached_attention_omits_the_merge_block_under_bit_exact_policy() {
        let mut bound = cached_attention_op();
        let BoundOpKind::CachedAttention {
            cached_key_rows,
            new_key_rows,
            ..
        } = &mut bound.kind
        else {
            unreachable!("cached_attention_op always returns a CachedAttention kind");
        };
        *cached_key_rows = 48;
        *new_key_rows = 16;

        let rejected = render_cached_attention(&bound, "entry", NumericPolicy::bit_exact())
            .expect("bit_exact() still renders -- it falls back to the single-pass kernel");
        let admitted = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
            .expect("llama_relaxed() renders the cross-simdgroup merge kernel");

        assert!(
            !rejected.contains("merged_max"),
            "bit_exact() must never reassociate the online-softmax fold across simdgroups"
        );
        assert!(
            admitted.contains("merged_max"),
            "llama_relaxed() is expected to emit the cross-simdgroup merge block"
        );
    }

    /// Redesign §4c's own inert proof: `bit_exact()` renders the split
    /// kernel's SOURCE and BINDINGS exactly as `render_cached_attention`
    /// already did before `ContextSplitMerge` existed (no `attn_scratch`,
    /// no `Binding::Scratch`, still a single `Binding::Output`), and emits
    /// no companion merge kernel at all. `llama_relaxed()` is the ONLY
    /// policy that changes either: the split kernel's final store moves to
    /// `attn_scratch` and its output binding becomes `Binding::Scratch`,
    /// and `emit_cached_attention_merge` returns the companion kernel that
    /// reads that scratch layout back and writes the real output. 256 keys
    /// (`omega-runtime.toml`'s `keys_per_split_at_scale = 128`, ROW 383) is
    /// deliberately past the split threshold under EITHER policy that admits
    /// the rewrite --
    /// `cached_attention_merge_needed` now takes context length, not just
    /// policy (its own doc), so a fixture at or below `keys_per_split`
    /// would stay single-dispatch even under `llama_relaxed`, proving
    /// nothing about the two-dispatch form this test exists to check.
    #[test]
    fn cached_attention_two_dispatch_form_is_inert_under_bit_exact() {
        let mut bound = cached_attention_op_dynamic(200, 56);
        let BoundOpKind::CachedAttention { head_dim, .. } = &mut bound.kind else {
            unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
        };
        // multiple of 8, `render_cached_attention`'s own alignment
        // requirement once `block_width_for` admits `TreeReduce`
        // (`llama_relaxed()` below) -- `AttentionBlockMisaligned`'s own doc.
        *head_dim = 8;
        let packed_operands = PackedOperands::new();

        let bit_exact_kernel = emit(&bound, &packed_operands, NumericPolicy::bit_exact())
            .expect("bit_exact() still renders the single-dispatch kernel");
        assert!(
            !bit_exact_kernel.source.contains("attn_scratch"),
            "bit_exact() must render the byte-identical, pre-redesign kernel body"
        );
        assert!(
            bit_exact_kernel
                .bindings
                .iter()
                .any(|binding| matches!(binding, Binding::Output(node) if *node == bound.node)),
            "bit_exact() must still bind its own node as a real Binding::Output"
        );
        assert!(
            !bit_exact_kernel
                .bindings
                .iter()
                .any(|binding| matches!(binding, Binding::Scratch)),
            "bit_exact() must never bind a scratch slot"
        );
        assert!(
            emit_cached_attention_merge(&bound, NumericPolicy::bit_exact())
                .expect("emit_cached_attention_merge never errors on a well-formed op")
                .is_none(),
            "bit_exact() must not need a merge dispatch at all"
        );

        let relaxed_kernel = emit(&bound, &packed_operands, NumericPolicy::llama_relaxed())
            .expect("llama_relaxed() renders the split kernel");
        assert!(
            relaxed_kernel.source.contains("attn_scratch"),
            "llama_relaxed() must write its final partial into scratch, not `out`"
        );
        assert!(
            relaxed_kernel
                .bindings
                .iter()
                .any(|binding| matches!(binding, Binding::Scratch)),
            "llama_relaxed()'s split kernel binds a scratch slot instead of Binding::Output"
        );
        assert!(
            !relaxed_kernel
                .bindings
                .iter()
                .any(|binding| matches!(binding, Binding::Output(_))),
            "llama_relaxed()'s split kernel must not ALSO claim to write the real output"
        );

        let merge_kernel = emit_cached_attention_merge(&bound, NumericPolicy::llama_relaxed())
            .expect("emit_cached_attention_merge never errors on a well-formed op")
            .expect("llama_relaxed() needs a companion merge dispatch");
        assert!(merge_kernel.entry.ends_with("_merge"));
        assert_eq!(
            merge_kernel.bindings,
            alloc::vec![
                Binding::Scratch,
                Binding::Output(bound.node),
                Binding::Uniforms
            ],
            "the merge kernel reads the split's scratch and writes the real output"
        );
    }

    /// Redesign §4c, item 3, corrected by ROW 385: a ROW 376-scoreboard-shaped
    /// context (40 keys) renders the single, byte-identical-in-STRUCTURE
    /// kernel under EITHER policy now -- [`cached_attention_merge_needed`]
    /// additionally requires `context_length >= ATTENTION_SPLIT_KEYS_PER_
    /// SPLIT_AT_SCALE` (128), so a 40-key context never engages the scratch
    /// hop regardless of what `splits_for`'s own small-divisor branch would
    /// report in isolation. ROW 381's split form measured faster at this
    /// window than the byte-identical sequential body (36.1us vs 43.6us
    /// bare), but ROW 385 supersedes it: the per-query-head grid
    /// ([`cached_attention_per_query_head_grid`]) is the mechanism that wins
    /// this window now, by giving the GPU more, narrower threadgroups to
    /// schedule instead of a second dispatch to pay for.
    #[test]
    fn forty_key_plan_uses_the_per_query_head_grid_with_no_merge_under_either_policy() {
        let mut bound = cached_attention_op_dynamic(32, 8);
        let BoundOpKind::CachedAttention { head_dim, .. } = &mut bound.kind else {
            unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
        };
        *head_dim = 8;
        let packed_operands = PackedOperands::new();

        for policy in [NumericPolicy::bit_exact(), NumericPolicy::llama_relaxed()] {
            let kernel =
                emit(&bound, &packed_operands, policy).expect("a 40-key context always renders");
            assert!(
                !kernel.source.contains("attn_scratch"),
                "a 40-key context must never engage the scratch hop under {policy:?}"
            );
            assert!(
                kernel
                    .source
                    .contains("long kv_head_and_group = (long)tgid % (kv_heads * query_groups);"),
                "a 40-key context must decode `group` off `tgid`, not shared threadgroup memory, \
                 under {policy:?}"
            );
            assert!(
                emit_cached_attention_merge(&bound, policy)
                    .expect("emit_cached_attention_merge never errors on a well-formed op")
                    .is_none(),
                "a 40-key context must never need a companion merge dispatch under {policy:?}"
            );
        }
    }

    /// ROW 385: below the split-at-scale knee the per-query-head grid must
    /// widen the THREADGROUP COUNT by `query_groups`, never the total thread
    /// count -- [`grid_threads`] is untouched by this feature by
    /// construction ([`cached_attention_per_query_head_grid`]'s own doc), so
    /// this is the one source of truth that the reshaping is real: the same
    /// total threads land in more, narrower threadgroups instead of fewer,
    /// wider ones.
    #[test]
    fn per_query_head_grid_narrows_threadgroup_width_without_changing_total_threads() {
        let mut narrow = cached_attention_op_dynamic(32, 8);
        let BoundOpKind::CachedAttention {
            head_dim,
            query_groups,
            ..
        } = &mut narrow.kind
        else {
            unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
        };
        *head_dim = 8;
        *query_groups = 4;

        let mut wide = narrow.clone();
        let BoundOpKind::CachedAttention {
            cached_key_rows,
            new_key_rows,
            ..
        } = &mut wide.kind
        else {
            unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
        };
        // Structural `cached_key_rows == 0` (the real single-range
        // invariant, `cached_attention_op_dynamic`'s own doc) -- the past-
        // the-knee 256-key total this test's own doc names lands entirely
        // in `new_key_rows`.
        *cached_key_rows = 0;
        *new_key_rows = 256;

        let quantized: Vec<Option<PackedCodec>> = Vec::new();
        let narrow_width =
            tiled_gemm_threadgroup_width(&narrow, &quantized, NumericPolicy::bit_exact())
                .expect("a CachedAttention op always has a threadgroup width");
        let wide_width =
            tiled_gemm_threadgroup_width(&wide, &quantized, NumericPolicy::bit_exact())
                .expect("a CachedAttention op always has a threadgroup width");
        assert_eq!(
            wide_width,
            4 * narrow_width,
            "past the knee (256 keys) the width must still carry the full query_groups factor \
             (4x the below-the-knee width): narrow={narrow_width} wide={wide_width}"
        );

        let narrow_threads = grid_threads(&narrow, &quantized, NumericPolicy::bit_exact())
            .expect("a CachedAttention op always has a thread count");
        let wide_threads = grid_threads(&wide, &quantized, NumericPolicy::bit_exact())
            .expect("a CachedAttention op always has a thread count");
        assert_eq!(
            narrow_threads, wide_threads,
            "total dispatched threads must be identical across the knee -- only the \
             threadgroup width narrows, never grid_threads' own total: narrow={narrow_threads} \
             wide={wide_threads}"
        );
    }

    /// Redesign §4c, item 1: the split kernel's own slice formula and
    /// scratch index expression, read straight out of the emitted MSL text
    /// -- an off-by-one in either is a device OOB write (design risk 3), so
    /// this pins the exact rendered arithmetic rather than trusting the
    /// hand-derivation.
    #[test]
    fn split_kernel_emits_the_slice_formula_and_scratch_index() {
        let mut bound = cached_attention_op_dynamic(200, 56);
        let BoundOpKind::CachedAttention { head_dim, .. } = &mut bound.kind else {
            unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
        };
        *head_dim = 8;

        let source = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
            .expect("llama_relaxed renders the split kernel for a 256-key context");

        assert!(
            source.contains("uint tgid [[threadgroup_position_in_grid]]"),
            "the split kernel must read its threadgroup index off Metal's own coordinate"
        );
        assert!(
            source.contains(
                "long slice_len = (live + splits - 1L) / splits;\n    long lo = split * slice_len;\n    long hi = min(lo + slice_len, live);"
            ),
            "the slice formula must be exactly ceil_div(live, splits), [lo, hi)"
        );
        assert!(
            source
                .contains("long scratch_index = (query_index * splits + split) * (2L + head_dim);"),
            "the scratch write must index by (query_index * splits + split) using the LIVE \
             u.splits value cached_attention_scratch_len sized the buffer with, not the \
             compiled ATTENTION_SPLIT_MAX -- ROW: production, 2026-09-07, striding by the \
             compiled max wrote past every query_index > 0's allocated slot"
        );
        assert!(
            source.contains("constexpr long grid_splits = 32;"),
            "the decode modulus must be the compiled grid multiplier (ATTENTION_SPLIT_MAX = 32 \
             here), the SAME constant grid_threads widened the dispatch by -- never the live \
             u.splits, or a tgid the widened grid always dispatches decodes a query_row past \
             the real row count"
        );
        assert!(
            source.contains("long split = query_row_and_split % grid_splits;")
                && source.contains("long query_row = query_row_and_split / grid_splits;"),
            "query_row/split must divide out grid_splits, not the live splits field"
        );
        assert!(
            source.contains("if (split >= splits) { return; }"),
            "an idle split (split >= the live u.splits) must return before touching Q/K/V or \
             scratch -- the merge kernel already reads only i < u.splits, so it must not write \
             an identity partial either"
        );
        assert!(
            source.contains("long slice_start = lo + chunk;"),
            "the block-staged walk (llama_relaxed grants TreeReduce too) must start from this \
             threadgroup's own [lo, hi) slice, not the whole live range"
        );

        let sequential_source = render_cached_attention(&bound, "entry", NumericPolicy::default())
            .expect("bit_exact still renders (splits collapse to 1, but the shape is shared)");
        assert!(
            sequential_source.contains("for (long key = lo + chunk; key < hi; key += chunks)"),
            "the strictly-sequential per-key walk must be bounded to this threadgroup's own \
             [lo, hi) slice"
        );
    }

    /// Redesign §4c: the merge kernel's own combine, read straight out of
    /// the emitted MSL text -- `simd_max`/`simd_sum` over the up-to-32
    /// per-split partials, exactly llama.cpp's `kernel_flash_attn_ext_vec_
    /// reduce` shape (`ggml-metal-ops.cpp:2063-2097` on `origin/master`).
    #[test]
    fn merge_kernel_emits_the_online_softmax_combine() {
        let bound = cached_attention_op_dynamic(200, 56);
        let source = render_cached_attention_merge(&bound, "entry")
            .expect("the merge kernel always renders");

        assert!(source.contains("long splits = u.splits;"));
        assert!(source.contains("float global_max = simd_max(own_max);"));
        assert!(source.contains("float total_sum = simd_sum(own_weight * own_sum);"));
        assert!(
            source.contains("for (long split = 0; split < splits; split++)"),
            "each lane's own dimensions must accumulate over every live split"
        );
    }

    /// Redesign §5 option 2, the ROW 369 residual this closes: crossing a
    /// `kv-capacity-bucket` boundary (39 rows of compiled capacity vs. 71)
    /// on the single-range fused (dynamic) path must neither change
    /// `entry_name` (the plan-cache key) nor the rendered MSL text -- both
    /// values now depend only on the compiled MAXIMUM chunk count (`cap`),
    /// never the live capacity, so one compiled `MTLComputePipelineState`
    /// serves both buckets and `pipeline_compile_ms` never fires again at a
    /// crossing (ROW 369's own residual, item 3).
    #[test]
    fn dynamic_cached_attention_kernel_identity_is_stable_across_kv_capacity_buckets() {
        let smaller_bucket = cached_attention_op_dynamic(32, 7); // capacity 39
        let larger_bucket = cached_attention_op_dynamic(64, 7); // capacity 71
        // `cached_attention_op_dynamic` folds both arguments into
        // `new_key_rows` alone (`cached_key_rows` stays the real, structural
        // `0` every single-range op carries) -- the compiled capacity this
        // test crosses is that SUM, so the fixture-sanity check reads it off
        // `new_key_rows`, not `cached_key_rows`.
        assert_ne!(
            match &smaller_bucket.kind {
                BoundOpKind::CachedAttention { new_key_rows, .. } => *new_key_rows,
                _ => unreachable!(),
            },
            match &larger_bucket.kind {
                BoundOpKind::CachedAttention { new_key_rows, .. } => *new_key_rows,
                _ => unreachable!(),
            },
            "the fixture must actually cross a different compiled capacity"
        );

        let smaller_name = entry_name(&smaller_bucket);
        let larger_name = entry_name(&larger_bucket);
        assert_eq!(
            smaller_name, larger_name,
            "kernel identity must be capacity-free on the dynamic path: got {smaller_name:?} \
             vs {larger_name:?}"
        );
        assert!(
            !smaller_name.contains("_c32") && !smaller_name.contains("_n7"),
            "row-count tokens must not appear in the dynamic path's entry name: {smaller_name:?}"
        );

        let smaller_source =
            render_cached_attention(&smaller_bucket, "entry", NumericPolicy::default())
                .expect("smaller bucket renders");
        let larger_source =
            render_cached_attention(&larger_bucket, "entry", NumericPolicy::default())
                .expect("larger bucket renders");
        assert_eq!(
            smaller_source, larger_source,
            "the generated MSL text itself must be capacity-free on the dynamic path"
        );
        assert!(
            smaller_source.contains("shared_m["),
            "the merge block is always present in the unified dynamic body"
        );
        assert!(
            smaller_source.contains("if (chunk < chunks)"),
            "chunk activity is a runtime guard, not a compile-time branch"
        );
        assert!(
            !smaller_source.contains("constexpr long context_chunks"),
            "context_chunks must be a runtime uniform, never baked as constexpr, \
             on the dynamic path"
        );
    }

    /// §4b's `block_width == 1` arm ([`block_width_for`] under `bit_exact`,
    /// where `NumericRewrite::TreeReduce` is rejected) must render the exact
    /// strictly-sequential per-key body -- never the `ss[]`-staged,
    /// `simd_shuffle_down`-reducing block body -- so a `bit_exact` plan's
    /// generated MSL text never regresses onto the reassociated path.
    #[test]
    fn block_width_one_renders_the_strictly_sequential_body_never_the_block_staged_one() {
        let bound = cached_attention_op_dynamic(32, 7);

        let source = render_cached_attention(&bound, "entry", NumericPolicy::bit_exact())
            .expect("bit_exact renders");

        assert!(
            !source.contains("block_width"),
            "bit_exact (block_width=1) must not emit the block-staging constant"
        );
        assert!(
            !source.contains("simd_shuffle_down"),
            "bit_exact (block_width=1) must not emit the block's tree-reduce shuffles"
        );
        assert!(
            source.contains("simd_broadcast_first(simd_sum(partial_score))"),
            "bit_exact (block_width=1) must still emit today's per-key simd_sum reduce"
        );
    }

    /// §4b's `block_width > 1` arm ([`block_width_for`] under
    /// `llama_relaxed`, which grants `NumericRewrite::TreeReduce`) must
    /// render the block-staged Q·K/softmax/V body: `ss[]` staging, the
    /// 8-lane `simd_shuffle_down` tree reduce, and a `simd_max`/`simd_sum`
    /// combine fired once per 32-lane sub-block rather than once per key.
    #[test]
    fn block_width_above_one_renders_the_block_staged_body_under_llama_relaxed() {
        let mut bound = cached_attention_op_dynamic(32, 7);
        let BoundOpKind::CachedAttention { head_dim, .. } = &mut bound.kind else {
            unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
        };
        *head_dim = 8; // a multiple of 8, so the float4 K/Q loads stay aligned

        let source = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
            .expect("llama_relaxed renders the block-staged body for an aligned head_dim");

        assert!(
            source.contains("block_width"),
            "the block-staging constant must be emitted"
        );
        assert!(
            source.contains("threadgroup float ss["),
            "scores must stage into ss[]"
        );
        assert!(
            source.contains("simd_shuffle_down(partial_score, 4)"),
            "the Q*K reduce must tree-reduce over the 8-lane key group"
        );
        assert!(
            source.contains("simd_max("),
            "one simd_max per 32-lane sub-block"
        );
        assert!(
            !source.contains("simd_broadcast_first(simd_sum(partial_score)) * scale"),
            "the block-staged body must not keep the old per-key reduce expression"
        );
    }

    /// §4b's V-accumulate port (this crate's `attention-kernel-design.md`
    /// §4b, llama's `kernel_flash_attn_ext_vec` register form,
    /// ggml-metal.metal:4125-4197): once `head_dim` is a multiple of 32 the
    /// block-staged body must load V as `float4` per `tx` lane and fold the
    /// four `ty` groups' partial sums back together with a `simd_shuffle_xor`
    /// cross-group reduce, rather than the scalar per-key V loop the
    /// smaller/unaligned `head_dim` fixtures above still render.
    #[test]
    fn block_staged_v_accumulate_uses_float4_and_a_cross_ty_reduce_when_aligned() {
        let mut bound = cached_attention_op_dynamic(32, 7);
        let BoundOpKind::CachedAttention { head_dim, .. } = &mut bound.kind else {
            unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
        };
        *head_dim = 32; // a multiple of 32, so the V float4 loads stay aligned

        let source = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
            .expect("llama_relaxed renders the block-staged body for a 32-aligned head_dim");

        assert!(
            source
                .contains("float4* v4 = (device const float4*)((cached ? in6 : in7) + kbase * 2)"),
            "V must be reinterpreted as a float4 pointer at the same kbase * 2 offset the \
             scalar form used"
        );
        assert!(
            source.contains("v_acc[register_index] += valid ? float4(v4[(long)tx + 8L * register_index]) * key_weight"),
            "each tx lane accumulates its own float4 chunk of V, weighted by its ty group's own key"
        );
        assert!(
            source.contains("v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 8)")
                && source.contains(
                    "v_acc[register_index] += simd_shuffle_xor(v_acc[register_index], 16)"
                ),
            "the four ty groups' partials must be folded together by a butterfly \
             simd_shuffle_xor reduce (strides 8 then 16), not left un-combined"
        );
        assert!(
            !source.contains(
                "weighted[local_dimension] += key_weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);"
            ),
            "an aligned head_dim must not keep the scalar per-key V loop the unaligned fixtures use"
        );
    }

    /// The scalar V-accumulate fallback (§4b: `head_dim` not a multiple of
    /// 32) must still be exactly today's per-key loop, byte for byte --
    /// `block_width_above_one_renders_the_block_staged_body_under_llama_relaxed`
    /// covers the Q·K side at this same `head_dim = 8`; this pins the V side.
    #[test]
    fn block_staged_v_accumulate_falls_back_to_scalar_when_head_dim_not_32_aligned() {
        let mut bound = cached_attention_op_dynamic(32, 7);
        let BoundOpKind::CachedAttention { head_dim, .. } = &mut bound.kind else {
            unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
        };
        *head_dim = 8; // multiple of 8 (Q/K stays aligned) but not of 32

        let source = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
            .expect("llama_relaxed renders the block-staged body for an unaligned head_dim");

        assert!(
            source.contains(
                "weighted[local_dimension] += key_weight * (cached ? in6[kbase * 2 + dimension] : in7[kbase * 2 + dimension]);"
            ),
            "an unaligned head_dim must keep the scalar per-key V loop"
        );
        assert!(
            !source.contains("simd_shuffle_xor"),
            "an unaligned head_dim must never emit the float4 cross-ty reduce"
        );
    }

    /// `bit_exact()` still withholds [`NumericRewrite::ContextSplitMerge`]
    /// at the scoreboard window (40 keys) regardless of `keys_per_split` --
    /// bit-identical output has no split/merge path at all under that
    /// policy. Under `llama_relaxed()`, ROW 381 lowered `omega-runtime.toml`'s
    /// `[attention_splits] keys_per_split` from 128 to 16 (measured 36.1us
    /// vs 43.6us bare at this window, beyond 2x CoV), so this window now
    /// DOES split: `ceil(40/16) = 3`, not the old single-threadgroup form.
    #[test]
    fn forty_keys_stays_one_split_under_bit_exact_but_splits_under_llama_relaxed() {
        assert_eq!(splits_for(40, NumericPolicy::bit_exact()), 1);
        assert_eq!(splits_for(40, NumericPolicy::llama_relaxed()), 3);
    }

    /// `bit_exact()` withholds `ContextSplitMerge` regardless of context
    /// length -- the split kernel-plus-merge machinery must never engage
    /// under a policy that demands bit-identical output, the same
    /// `admit`-rejects-so-fall-back-to-`1` shape [`context_chunks_for`]/
    /// [`block_width_for`] already use.
    #[test]
    fn long_contexts_stay_one_split_under_bit_exact() {
        assert_eq!(splits_for(512, NumericPolicy::bit_exact()), 1);
        assert_eq!(splits_for(4096, NumericPolicy::bit_exact()), 1);
    }

    /// `llama_relaxed()` grants reassociation, clearing
    /// `ContextSplitMerge`, so ROW 376's longer decode shapes (512/4096
    /// keys) must split across more than one threadgroup -- the whole
    /// point of the port: more of the GPU's 32 cores get a share of one
    /// attention op on the contexts where today's 8-threadgroup dispatch
    /// (`query_rows * kv_heads` for openchat's shape) leaves most of the
    /// GPU idle.
    #[test]
    fn long_contexts_split_across_more_than_one_threadgroup_under_llama_relaxed() {
        let splits_512 = splits_for(512, NumericPolicy::llama_relaxed());
        let splits_4096 = splits_for(4096, NumericPolicy::llama_relaxed());
        assert!(
            splits_512 > 1,
            "512 keys (4x omega-runtime.toml's 128-key keys_per_split_at_scale) must split; got {splits_512}"
        );
        assert!(splits_4096 > 1, "4096 keys must split; got {splits_4096}");
        assert!(
            splits_4096 >= splits_512,
            "a longer context must never split into FEWER threadgroups than a shorter one"
        );
    }

    /// ROW 383's own knee, pinned at the five lengths its own log row
    /// tabulates: below `keys_per_split_at_scale` (128) the small divisor
    /// (16) applies (40 keys -> 3 splits, ROW 381's decode-window win,
    /// unchanged); at or above it the large divisor (128, the ORIGINAL
    /// pre-ROW-381 value) applies, which is what measured fastest at 512
    /// keys (splits=4, 89.5us bare vs 170-177us at 8/16/32) and reaches the
    /// SAME split count (32) as the small divisor already did once clamped
    /// to `max` at 4096 keys, so nothing regresses at scale.
    #[test]
    fn splits_for_has_a_knee_at_the_scaled_divisor_threshold() {
        assert_eq!(splits_for(40, NumericPolicy::llama_relaxed()), 3);
        assert_eq!(splits_for(256, NumericPolicy::llama_relaxed()), 2);
        assert_eq!(splits_for(512, NumericPolicy::llama_relaxed()), 4);
        assert_eq!(splits_for(1024, NumericPolicy::llama_relaxed()), 8);
        assert_eq!(splits_for(4096, NumericPolicy::llama_relaxed()), 32);
    }

    /// `ATTENTION_SPLIT_MAX` (`omega-runtime.toml`'s `[attention_splits].max`,
    /// default 32) is a hard ceiling regardless of how long the context
    /// grows -- the compiled dispatch always issues exactly this many
    /// threadgroups per `(query_row, kv_head)` pair on the dynamic path
    /// (`grid_threads`'s own doc), so the live count this function returns
    /// can never exceed it without under-provisioning the grid.
    #[test]
    fn split_count_never_exceeds_the_compiled_maximum() {
        let splits = splits_for(1_000_000, NumericPolicy::llama_relaxed());
        assert_eq!(splits, crate::sized::ATTENTION_SPLIT_MAX);
    }

    /// [`EmitError::AttentionBlockMisaligned`]'s own reachable gate: a
    /// `head_dim` not a multiple of 8 leaves `head_dim / 2` not a multiple
    /// of 4, so the block-staged body's `float4` K/Q reinterpretation would
    /// read past an unaligned offset -- rejected here, named, rather than
    /// emitted.
    #[test]
    fn block_staged_attention_rejects_a_head_dim_not_a_multiple_of_eight() {
        let mut bound = cached_attention_op_dynamic(32, 7);
        let BoundOpKind::CachedAttention { head_dim, .. } = &mut bound.kind else {
            unreachable!("cached_attention_op_dynamic always returns a CachedAttention kind");
        };
        *head_dim = 12; // not a multiple of 8

        let error = render_cached_attention(&bound, "entry", NumericPolicy::llama_relaxed())
            .expect_err("a misaligned head_dim must be rejected under the block-staged policy");

        assert_eq!(
            error,
            EmitError::AttentionBlockMisaligned {
                node: bound.node,
                head_dim: 12,
            }
        );
    }
}
